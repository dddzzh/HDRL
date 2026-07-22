from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.workflow_factory import create_workflows
from src.ppo.ppo import PPO
from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.metrics import compute_summary
from src.simulator.resource import load_resources


def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO demo: Pegasus workflow arrivals, train then evaluate."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-episodes", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--num-workflows", type=int, default=20)
    parser.add_argument("--mean-interarrival", type=float, default=45.0)
    parser.add_argument("--num-edge", type=int, default=8)
    parser.add_argument("--num-cloud", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="results/ppo_demo/ppo_demo.pt")
    parser.add_argument("--device", type=str, default="cpu", help="Agent device: cpu | cuda | auto")
    return parser.parse_args()


def _workflow_generator_config(args) -> dict:
    return {
        "num_workflows": args.num_workflows,
        "mean_interarrival": args.mean_interarrival,
        "min_tasks": 3,
        "max_tasks": 8,
        "workload_gc_range": [0.5, 20.0],
        "edge_data_mb": 1.0,
    }


def _agent_config(args, checkpoint: str | None = None) -> dict:
    config = {
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "ppo_epochs": 4,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "learning_rate": 3e-4,
        "hidden_dim": 128,
        "max_ready_tasks": 32,
        "max_servers": 16,
        "max_rho_levels": 6,
        "device": args.device,
    }
    if checkpoint:
        config["checkpoint"] = checkpoint
    return config


def _build_env(workflows, resources, seed: int) -> EdgeCloudWorkflowEnv:
    return EdgeCloudWorkflowEnv(
        workflows=workflows,
        resources=resources,
        sim_config={"default_rho": 1.0, "rho_min": 0.5},
        seed=seed,
    )


def _print_summary(title: str, episode_result: dict, resources: dict) -> None:
    summary = compute_summary(
        episode_result["workflow_results"],
        episode_result["event_log"],
        resources["servers"],
    )
    print(f"\n=== {title} ===")
    print(f"workflows: {len(episode_result['workflow_results'])}")
    print(f"steps: {episode_result['steps']}")
    print(f"finished: {summary['total_finished_workflows']}")
    print(f"total_energy: {summary['total_energy']:.4f}")
    print(f"avg_response_time: {summary['avg_workflow_response_time']:.4f}")
    print(f"avg_makespan: {summary['avg_workflow_makespan']:.4f}")
    print(f"avg_utilization: {summary['avg_server_utilization']:.4f}")
    if episode_result.get("train_metrics"):
        metrics = episode_result["train_metrics"]
        loss = metrics.get("loss")
        ret = metrics.get("return_mean")
        clip = metrics.get("clip_fraction")
        loss_str = f"{loss:.6f}" if loss is not None else "n/a"
        ret_str = f"{ret:.6f}" if ret is not None else "n/a"
        clip_str = f"{clip:.6f}" if clip is not None else "n/a"
        print(f"train loss={loss_str} return={ret_str} clip={clip_str}")


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    resources = load_resources(
        "configs/resources_edge_cloud_default.json",
        num_edge=args.num_edge,
        num_cloud=args.num_cloud,
        seed=args.seed,
    )

    ppo = PPO(config=_agent_config(args), seed=args.seed)
    gen_cfg = _workflow_generator_config(args)
    train_history = []

    print("PPO Demo — fixed workflow count")
    print(
        f"num_workflows={args.num_workflows}, "
        f"mean_interarrival={args.mean_interarrival}s"
    )

    for episode in range(1, args.train_episodes + 1):
        workflows = create_workflows(
            gen_cfg,
            source="pegasus",
            seed=args.seed + episode,
            resources=resources,
        )
        env = _build_env(workflows, resources, seed=args.seed + episode)
        episode_result = ppo.train_episode(env)
        train_history.append({"episode": episode, **episode_result.get("train_metrics", {})})
        _print_summary(f"Train episode {episode}", episode_result, resources)

    ppo.save(str(checkpoint_path), extra={"demo": "ppo_pegasus_arrival"})
    print(f"\nCheckpoint saved: {checkpoint_path}")

    eval_ppo = PPO(config=_agent_config(args, checkpoint=str(checkpoint_path)), seed=args.seed)
    eval_summaries = []

    for episode in range(1, args.eval_episodes + 1):
        eval_seed = args.seed + 1000 + episode
        workflows = create_workflows(
            gen_cfg,
            source="pegasus",
            seed=eval_seed,
            resources=resources,
        )
        env = _build_env(workflows, resources, seed=eval_seed)
        episode_result = eval_ppo.collect_episode(env, deterministic=True)
        summary = compute_summary(
            episode_result["workflow_results"],
            episode_result["event_log"],
            resources["servers"],
        )
        eval_summaries.append(summary)
        _print_summary(f"Eval episode {episode}", episode_result, resources)

    avg_eval_energy = sum(row["total_energy"] for row in eval_summaries) / len(eval_summaries)
    avg_eval_response = sum(row["avg_workflow_response_time"] for row in eval_summaries) / len(
        eval_summaries
    )
    print("\n=== Eval average ===")
    print(f"avg_total_energy: {avg_eval_energy:.4f}")
    print(f"avg_response_time: {avg_eval_response:.4f}")

    history_path = checkpoint_path.parent / "demo_history.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "train_history": train_history,
                "eval_summaries": eval_summaries,
                "generator": gen_cfg,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"History saved: {history_path}")


if __name__ == "__main__":
    main()
