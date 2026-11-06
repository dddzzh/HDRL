"""Run the Small-scenario ablations from the current future-ESP baseline."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import train_coupled_hierarchical_d3qn_sac as coupled
from tools.hierarchical_common import load_json


BASE_CONFIG = ROOT / "configs" / "v2_small_20edge_6cloud_dvfs_v3_future_esp.json"
LOW_VARIANTS = {"low_full", "low_no_deadline", "low_no_reliability"}
VARIANTS = {
    "heft_sac",
    "high_full_rho1",
    "high_no_deadline_rho1",
    *LOW_VARIANTS,
}


def build_variant_config(base_config, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant!r}.")
    config = copy.deepcopy(base_config)
    generator = config.setdefault("generator", {})
    resources = config.setdefault("resources", {})
    if str(generator.get("dataset_scale", "")).lower() != "small":
        raise ValueError("Small ablations require generator.dataset_scale='small'.")
    if (int(resources.get("num_edge", -1)), int(resources.get("num_cloud", -1))) != (20, 6):
        raise ValueError("Small ablations require exactly 20 edge and 6 cloud servers.")

    agent = config.setdefault("agent", {})
    high = agent.setdefault("high", {})
    low = agent.setdefault("low", {})
    training = config.setdefault("training", {})
    training["phase_name"] = f"ablation_{variant}"
    training["checkpoint_dir"] = f"results/ablations/small_future_esp/{variant}"

    if variant == "heft_sac":
        agent["high_algorithm"] = "heft"
    elif variant == "high_no_deadline_rho1":
        high["deadline_head_enabled"] = False
    elif variant == "low_no_deadline":
        low.setdefault("reward", {})["mode"] = "fixed_deadline_penalty"
        low["reward"]["deadline_violation_weight"] = 0.0
        low.setdefault("lagrangian", {})["enabled"] = False
    elif variant == "low_no_reliability":
        low.setdefault("reliability_lagrangian", {})["enabled"] = False
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one Small future-ESP ablation with the production trainer."
    )
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--config", default=str(BASE_CONFIG))
    parser.add_argument("--high-checkpoint")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.variant in LOW_VARIANTS and not args.high_checkpoint:
        raise ValueError(
            f"{args.variant} freezes one common upper policy and therefore "
            "requires --high-checkpoint."
        )
    if args.variant not in LOW_VARIANTS and args.high_checkpoint:
        raise ValueError("--high-checkpoint is only valid for low_* variants.")

    config = build_variant_config(load_json(args.config), args.variant)
    fixed_rho_one = args.variant.startswith("high_")
    freeze_high = args.variant in LOW_VARIANTS
    result = coupled.run_training(
        config,
        smoke_test=args.smoke_test,
        episodes_override=args.episodes,
        seed_override=args.seed,
        fixed_low_rho_one=fixed_rho_one,
        high_algorithm_override=("heft" if args.variant == "heft_sac" else None),
        freeze_high=freeze_high,
        high_checkpoint=args.high_checkpoint,
    )
    print(f"variant={args.variant}")
    print(f"output_dir={result['output_dir']}")


if __name__ == "__main__":
    main()
