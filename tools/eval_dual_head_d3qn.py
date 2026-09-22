from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.D3QN.dual_head_constrained import DualHeadHierarchicalD3QNAgent
from src.D3QN.reward import HierarchicalObjective
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from tools.hierarchical_common import (
    build_env,
    build_resources,
    build_workflows,
    configure_agent_state,
)
from tools.train_hierarchical_d3qn import collect_d3qn_episode


REPORT_METRICS = (
    "deadline_success_rate",
    "reliability_success_rate",
    "avg_workflow_reliability",
    "total_energy",
    "average_workflow_response_time",
    "episode_makespan",
    "avg_workflow_makespan",
    "max_workflow_makespan",
    "global_makespan",
    "rho_mean",
    "low_sac_total_energy_saving_ratio",
    "total_low_sac_energy_reference",
    "total_low_sac_energy_actual",
    "total_low_sac_energy_saved",
    "dual_energy_action_std_mean",
    "dual_deadline_action_std_mean",
    "dual_deadline_to_energy_scale_mean",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a dual-head D3QN checkpoint with fixed rho=1 or its SAC."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-start", type=int, default=242000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--deadline-type",
        choices=("tight", "medium", "loose"),
        default=None,
        help="Override the checkpoint's deadline stringency for evaluation only.",
    )
    parser.add_argument(
        "--use-low-sac",
        action="store_true",
        help="Use the lower SAC stored in the combined checkpoint instead of fixed rho=1.",
    )
    parser.add_argument(
        "--rho-floor",
        type=float,
        default=None,
        help="Evaluation-only lower bound applied to deterministic SAC rho.",
    )
    parser.add_argument(
        "--enforce-rho-safe",
        action="store_true",
        help="Evaluation-only: clamp deterministic SAC rho to the per-task rho_safe.",
    )
    return parser.parse_args()


def describe(values):
    array = np.asarray(values, dtype=np.float64)
    count = int(array.size)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    t_critical = 2.093 if count == 20 else 1.96
    ci95 = float(t_critical * std / math.sqrt(count)) if count > 1 else 0.0
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95_half_width": ci95,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main():
    args = parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    checkpoint = Path(args.checkpoint).resolve()
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload.get("config") or {})
    if not config:
        raise RuntimeError("Checkpoint does not contain its training configuration.")
    if args.deadline_type is not None:
        config.setdefault("simulation", {})["deadline_type"] = args.deadline_type
    agent_config = configure_agent_state(
        config,
        build_resources(config, int(config.get("training", {}).get("seed", 42))),
        smoke_test=False,
    )
    agent_config["use_low_agent"] = bool(args.use_low_sac)
    saved_high = dict(payload.get("dual_head_state_dict") or {})
    if "reliability_q_eval" in saved_high:
        from tools.train_reliability_aware_hierarchical_d3qn import (
            ReliabilityAwareHierarchicalD3QNAgent,
        )
        agent = ReliabilityAwareHierarchicalD3QNAgent(
            agent_config, seed=args.seed_start
        )
    else:
        agent = DualHeadHierarchicalD3QNAgent(agent_config, seed=args.seed_start)
    agent.load_checkpoint(checkpoint)
    if args.rho_floor is not None or args.enforce_rho_safe:
        if not args.use_low_sac:
            raise ValueError("DVFS evaluation floors require --use-low-sac.")
    if args.rho_floor is not None:
        rho_floor = float(args.rho_floor)
        if not 0.0 < rho_floor <= 1.0:
            raise ValueError("--rho-floor must be in (0, 1].")
    if args.rho_floor is not None or args.enforce_rho_safe:
        original_select_frequency = agent.low_agent.select_frequency

        def select_frequency_with_floor(*select_args, **select_kwargs):
            decision = dict(original_select_frequency(*select_args, **select_kwargs))
            rho = float(decision["rho"])
            if args.rho_floor is not None:
                rho = max(float(args.rho_floor), rho)
            if args.enforce_rho_safe:
                rho = max(float(decision.get("rho_safe", rho)), rho)
            decision["rho"] = rho
            decision["action_rho"] = rho
            decision["action_u"] = rho
            return decision

        agent.low_agent.select_frequency = select_frequency_with_floor
    agent.high_agent.epsilon = 0.0
    objective = HierarchicalObjective(config.get("objective", {}))
    resources = build_resources(
        config, int(config.get("training", {}).get("seed", 42))
    )

    rows = []
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        with redirect_stdout(StringIO()):
            workflows = build_workflows(config, resources, seed, smoke_test=False)
            env = build_env(
                workflows,
                resources,
                dict(config.get("simulation", {})),
                seed,
            )
            result = collect_d3qn_episode(
                env,
                agent,
                objective,
                HighRolloutBuffer(),
                LowRolloutBuffer(),
                training=False,
                high_deterministic=True,
                low_deterministic=True,
                fixed_low_rho_one=not args.use_low_sac,
                high_algorithm="d3qn",
            )
        metrics = result["metrics"]
        row = {
            "eval_seed": seed,
            **{name: float(metrics.get(name, 0.0)) for name in REPORT_METRICS},
        }
        rows.append(row)
        print(
            f"[eval] seed={seed} deadline={row['deadline_success_rate']:.2%} "
            f"energy_kJ={row['total_energy'] / 1000.0:.2f} "
            f"response={row['average_workflow_response_time']:.2f} "
            f"rho={row['rho_mean']:.3f}",
            flush=True,
        )

    summary = {name: describe([row[name] for row in rows]) for name in REPORT_METRICS}
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_episode": int(payload.get("episode", 0)),
        "checkpoint_format": payload.get("checkpoint_format"),
        "deadline_head_cost_mode": payload.get("deadline_head_cost_mode"),
        "fixed_low_rho_one": not args.use_low_sac,
        "low_agent_used": bool(args.use_low_sac),
        "evaluation_rho_floor": args.rho_floor,
        "evaluation_enforce_rho_safe": bool(args.enforce_rho_safe),
        "deadline_type": config.get("simulation", {}).get("deadline_type", "medium"),
        "seed_start": args.seed_start,
        "seed_end": args.seed_start + args.num_seeds - 1,
        "rows": rows,
        "summary": summary,
    }
    output = (
        Path(args.output).resolve()
        if args.output
        else checkpoint.parent
        / ("eval_best_sac_20seed.json" if args.use_low_sac else "eval_best_20seed.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[summary] "
        f"episode={report['checkpoint_episode']} "
        f"deadline={summary['deadline_success_rate']['mean']:.2%} "
        f"energy_kJ={summary['total_energy']['mean'] / 1000.0:.2f} "
        f"response={summary['average_workflow_response_time']['mean']:.2f} "
        f"rho={summary['rho_mean']['mean']:.3f}",
        flush=True,
    )
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
