from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.baselines.dqn_edge import DQNEdgeConfig
from tools.hierarchical_common import (
    DEADLINE_TYPES,
    apply_deadline_type,
    build_env,
    build_resources,
    build_workflows,
    load_json,
)
from tools.train_dqn_edge import build_dqn_edge_runtime, run_episode


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN-Edge checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="configs/dqn_edge_small_20edge_6cloud.json")
    parser.add_argument(
        "--deadline-type",
        choices=DEADLINE_TYPES,
        default=None,
        help=(
            "Deadline regime to evaluate under (tight=P(2x)=0.8, medium=0.5, "
            "loose=0.2). Defaults to the config value."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=242000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def evaluate(config, args):
    resource_seed = int(config["training"].get("seed", 42))
    resources = build_resources(config, resource_seed)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    values = dict(payload.get("config", config["agent"]["dqn_edge"]))
    values["device"] = args.device
    agent_config = DQNEdgeConfig.from_mapping(values)
    paper_state_dim = 3 + 2 * len(resources["servers"])
    variant = payload.get("variant")
    if variant is None:
        # Checkpoints created before the common-state adapter are unambiguously
        # paper-mode because Eq. 27 has a fixed 3 + 2|VM| dimension.
        variant = (
            "paper"
            if int(payload.get("state_dim", paper_state_dim)) == paper_state_dim
            else config.get("agent", {}).get("variant", "common_reward")
        )
    runtime = build_dqn_edge_runtime(
        config,
        resources,
        agent_config,
        variant=variant,
        state_config=payload.get("state_config"),
        objective_config=payload.get("objective_config"),
    )
    agent, scheduler = runtime["agent"], runtime["scheduler"]
    agent.load_checkpoint(payload)
    rows = []
    count = min(args.num_seeds, 2) if args.smoke_test else args.num_seeds
    for seed in range(args.seed_start, args.seed_start + count):
        workflows = build_workflows(config, resources, seed, smoke_test=args.smoke_test)
        simulation = {**config.get("simulation", {}), "enable_backup_duplication": False, "default_rho": 1.0}
        env = build_env(workflows, resources, simulation, seed)
        row = run_episode(env, scheduler, training=False); row["seed"] = seed
        rows.append(row)
        print(json.dumps({"seed": seed, "total_energy": row["total_energy"],
                          "deadline_success_rate": row["deadline_success_rate"],
                          "reliability_success_rate": row["reliability_success_rate"],
                          "avg_workflow_response_time": row["avg_workflow_response_time"]}, ensure_ascii=False))
    metrics = ("total_energy", "deadline_success_rate", "reliability_success_rate", "avg_workflow_response_time")
    aggregate = {name: {"mean": float(np.mean([row[name] for row in rows])),
                        "std": float(np.std([row[name] for row in rows]))} for name in metrics}
    result = {"algorithm": "adapted_dqn_edge", "variant": runtime["variant"],
              "checkpoint": str(Path(args.checkpoint).resolve()),
              "resource_seed": resource_seed, "workflow_seed_start": args.seed_start,
              "num_seeds": count, "rho": 1.0, "backup_duplication": False,
              "deadline_type": config.get("simulation", {}).get("deadline_type", "medium"),
              "aggregate": aggregate, "rows": rows}
    output = Path(args.output) if args.output else Path(args.checkpoint).resolve().parent / "evaluation.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output={output}")
    return result


def main():
    args = parse_args()
    evaluate(apply_deadline_type(load_json(args.config), args.deadline_type), args)


if __name__ == "__main__":
    main()
