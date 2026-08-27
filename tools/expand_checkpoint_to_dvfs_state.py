"""Zero-initialised expansion of a 347-D three-head checkpoint to the 425-D DVFS state.

The DVFS-aware upper state keeps the production layout and only widens the
per-server block from 12 to 15 features::

    [task 4] [server_features 26 x F] [server_mask 26] [global 5]

so every existing input column has an exact image in the new layout.  This
tool rebuilds ``backbone.0.weight`` with the old columns copied into their new
positions and the three new columns per server set to **zero**.  Every other
parameter (including ``backbone.0.bias``, all deeper layers and both duelling
heads) is copied verbatim.

Consequently the expanded network computes *exactly* the same Q values as the
source checkpoint until training moves the new columns off zero.  The
DVFS-feature ablation is therefore a strict increment on a shared starting
point rather than a separate from-scratch run.

Usage::

    py tools/expand_checkpoint_to_dvfs_state.py \
        --source results/.../checkpoint_ep000300.pt \
        --output results/.../checkpoint_ep000300_dvfs425.pt
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.D3QN.state import (
    GLOBAL_FEATURE_DIM,
    SERVER_FEATURE_DIM,
    TASK_FEATURE_DIM,
)
from src.D3QN.dvfs_aware_state import DVFS_SERVER_FEATURE_DIM

HEAD_KEYS = (
    "energy_q_eval",
    "energy_q_target",
    "deadline_q_eval",
    "deadline_q_target",
    "reliability_q_eval",
    "reliability_q_target",
)


def column_map(num_servers):
    """Return ``old_index -> new_index`` for every input column."""
    old_dim = (
        TASK_FEATURE_DIM
        + num_servers * SERVER_FEATURE_DIM
        + num_servers
        + GLOBAL_FEATURE_DIM
    )
    mapping = {}
    for index in range(TASK_FEATURE_DIM):
        mapping[index] = index
    for server in range(num_servers):
        for feature in range(SERVER_FEATURE_DIM):
            old = TASK_FEATURE_DIM + server * SERVER_FEATURE_DIM + feature
            new = TASK_FEATURE_DIM + server * DVFS_SERVER_FEATURE_DIM + feature
            mapping[old] = new
    old_tail = TASK_FEATURE_DIM + num_servers * SERVER_FEATURE_DIM
    new_tail = TASK_FEATURE_DIM + num_servers * DVFS_SERVER_FEATURE_DIM
    for offset in range(num_servers + GLOBAL_FEATURE_DIM):
        mapping[old_tail + offset] = new_tail + offset
    if len(mapping) != old_dim:
        raise RuntimeError("Column map does not cover the whole source state.")
    return mapping


def payload_new_dim(num_servers):
    return (
        TASK_FEATURE_DIM
        + num_servers * DVFS_SERVER_FEATURE_DIM
        + num_servers
        + GLOBAL_FEATURE_DIM
    )


def expand_head(head_state, num_servers):
    head_state = {key: value.clone() for key, value in head_state.items()}
    weight = head_state["backbone.0.weight"]
    hidden, old_dim = weight.shape
    new_dim = (
        TASK_FEATURE_DIM
        + num_servers * DVFS_SERVER_FEATURE_DIM
        + num_servers
        + GLOBAL_FEATURE_DIM
    )
    mapping = column_map(num_servers)
    if old_dim != len(mapping):
        raise ValueError(
            f"Source layer expects {len(mapping)} inputs, checkpoint has {old_dim}."
        )
    expanded = torch.zeros((hidden, new_dim), dtype=weight.dtype)
    for old_index, new_index in mapping.items():
        expanded[:, new_index] = weight[:, old_index]
    head_state["backbone.0.weight"] = expanded
    return head_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-servers", type=int, default=None)
    parser.add_argument(
        "--checkpoint-format",
        default="hierarchical_coupled_dvfs_d3qn_v1",
        help="Format string written to the expanded checkpoint.",
    )
    args = parser.parse_args()

    payload = torch.load(str(args.source), map_location="cpu", weights_only=False)
    saved = payload.get("dual_head_state_dict")
    if saved is None:
        raise RuntimeError("Source is not a dual-head / three-head checkpoint.")
    num_servers = args.num_servers
    if num_servers is None:
        num_servers = int(
            payload["config"]["resources"]["num_edge"]
            + payload["config"]["resources"]["num_cloud"]
        )
    old_dim = saved["energy_q_eval"]["backbone.0.weight"].shape[1]
    expected_old = (
        TASK_FEATURE_DIM
        + num_servers * SERVER_FEATURE_DIM
        + num_servers
        + GLOBAL_FEATURE_DIM
    )
    if old_dim != expected_old:
        raise ValueError(
            f"num_servers={num_servers} implies a {expected_old}-D state but the "
            f"checkpoint has {old_dim}-D."
        )

    expanded = copy.deepcopy(saved)
    for key in HEAD_KEYS:
        if key not in saved:
            raise RuntimeError(f"Source checkpoint is missing head {key!r}.")
        expanded[key] = expand_head(saved[key], num_servers)
    # Adam moments for the first layer are shape-bound to the old width and are
    # expanded with the same column map, so momentum for existing features is
    # preserved and the new columns start with zero moment.
    mapping = column_map(num_servers)
    for key in ("energy_optimizer", "deadline_optimizer", "reliability_optimizer"):
        optimizer_state = expanded.get(key)
        if optimizer_state is None:
            continue
        for entry in optimizer_state.get("state", {}).values():
            for name, tensor in list(entry.items()):
                if not torch.is_tensor(tensor) or tensor.ndim != 2:
                    continue
                if tensor.shape[1] != old_dim:
                    continue
                grown = torch.zeros(
                    (tensor.shape[0], payload_new_dim(num_servers)), dtype=tensor.dtype
                )
                for old_index, new_index in mapping.items():
                    grown[:, new_index] = tensor[:, old_index]
                entry[name] = grown
    expanded["replay"] = None
    expanded["reliability_costs"] = None

    payload["dual_head_state_dict"] = expanded
    # The saved lower SAC critics were fitted to the replay-z-scored reward and
    # would mis-value the physical reward, so the lower level starts fresh.
    payload["low_model_state_dict"] = None
    payload["checkpoint_format"] = args.checkpoint_format
    payload["expanded_from"] = str(Path(args.source).resolve())
    payload["expanded_state_dim"] = int(
        TASK_FEATURE_DIM
        + num_servers * DVFS_SERVER_FEATURE_DIM
        + num_servers
        + GLOBAL_FEATURE_DIM
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(output))
    print(
        f"expanded {old_dim} -> {payload['expanded_state_dim']} "
        f"(servers={num_servers}, {len(HEAD_KEYS)} heads, new columns zero-initialised)"
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
