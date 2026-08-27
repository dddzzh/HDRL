"""Run the A1/A3/A4 upper-layer ablation with paired seeds.

A1 uses the random upper scheduler, A3 requires a separately trained
single-head DQN checkpoint, and A4 uses the joint dual-head checkpoint.
The script writes one JSON report containing per-seed rows and summaries.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.D3QN.dual_head_constrained import DualHeadHierarchicalD3QNAgent
from src.D3QN.reward import HierarchicalObjective
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from tools.hierarchical_common import build_env, build_resources, build_workflows, configure_agent_state
from tools.train_hierarchical_d3qn import collect_d3qn_episode

ROOT = Path(__file__).resolve().parents[1]


def describe(values):
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {"n": int(len(values)), "mean": float(values.mean()), "std": std,
            "ci95_half_width": float(2.093 * std / math.sqrt(len(values))) if len(values) == 20 else 0.0}


def evaluate_variant(label, variant, config, seed_start, num_seeds, checkpoint_override=None):
    checkpoint_value = checkpoint_override or variant.get("checkpoint")
    checkpoint = (ROOT / checkpoint_value).resolve() if checkpoint_value else None
    resources = build_resources(config, int(config.get("training", {}).get("seed", 42)))
    agent_config = configure_agent_state(copy.deepcopy(config), resources, smoke_test=False)
    agent_config["use_low_agent"] = True
    if checkpoint is None:
        raise RuntimeError(f"{label} requires --{label.lower()}-checkpoint.")
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    agent = DualHeadHierarchicalD3QNAgent(agent_config, seed=seed_start)
    agent.load_checkpoint(checkpoint)
    agent.high_agent.epsilon = 0.0
    objective = HierarchicalObjective(config.get("objective", {}))
    rows = []
    for offset in range(num_seeds):
        seed = seed_start + offset
        workflows = build_workflows(config, resources, seed, smoke_test=False)
        env = build_env(workflows, resources, dict(config.get("simulation", {})), seed)
        result = collect_d3qn_episode(
            env, agent, objective, HighRolloutBuffer(), LowRolloutBuffer(),
            training=False, high_deterministic=True, low_deterministic=True,
            fixed_low_rho_one=False, high_algorithm=variant["high_algorithm"],
        )
        metrics = result["metrics"]
        rows.append({"eval_seed": seed, "label": label,
                     "total_energy": float(metrics.get("total_energy", 0.0)),
                     "deadline_success_rate": float(metrics.get("deadline_success_rate", 0.0)),
                     "episode_makespan": float(metrics.get("episode_makespan", 0.0)),
                     "average_workflow_response_time": float(metrics.get("average_workflow_response_time", 0.0)),
                     "rho_mean": float(metrics.get("rho_mean", 0.0))})
    keys = ["total_energy", "deadline_success_rate", "episode_makespan", "average_workflow_response_time", "rho_mean"]
    return {"label": label, "checkpoint": str(checkpoint), "rows": rows,
            "summary": {key: describe([row[key] for row in rows]) for key in keys}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablation_a1_a3_a4.json")
    parser.add_argument("--a3-checkpoint", required=True, help="Single-head DQN checkpoint for A3.")
    parser.add_argument("--a4-checkpoint", default=None)
    parser.add_argument("--seed-start", type=int, default=242000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    spec = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    config = json.loads((ROOT / spec["base_config"]).read_text(encoding="utf-8"))
    variants = spec["variants"]
    # A1 is evaluated with the same trained checkpoint only to keep the lower
    # SAC fixed; its upper scheduler is replaced by the random scheduler.
    a1 = dict(variants["A4"])
    a1["high_algorithm"] = "random"
    reports = [
        evaluate_variant("A1", a1, config, args.seed_start, args.num_seeds, args.a4_checkpoint),
        evaluate_variant("A3", variants["A3"], config, args.seed_start, args.num_seeds, args.a3_checkpoint),
        evaluate_variant("A4", variants["A4"], config, args.seed_start, args.num_seeds, args.a4_checkpoint),
    ]
    output = Path(args.output) if args.output else ROOT / "results" / "ablation_a1_a3_a4" / "ablation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"config": str((ROOT / args.config).resolve()), "reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output={output.resolve()}")


if __name__ == "__main__":
    main()
