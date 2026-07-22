from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.workflow_factory import create_workflows
from src.experiments.runner import ExperimentRunner
from src.schedulers.registry import create_scheduler
from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.metrics import compute_summary
from src.simulator.resource import load_resources
from src.device_utils import describe_runtime_devices, resolve_agent_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO scheduler.")
    parser.add_argument("--config", default="configs/ppo_train_default.json")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | auto")
    return parser.parse_args()


def load_workflows(
    runner: ExperimentRunner,
    dataset_config: Mapping[str, Any],
    seed: int,
    resources: Mapping[str, Any] | None = None,
):
    dataset_type = dataset_config.get("type", "alibaba_json")
    if dataset_type == "alibaba_json":
        return runner._load_dataset(dataset_config, seed, resources=resources)
    if dataset_type == "pegasus":
        generator_cfg = {key: value for key, value in dataset_config.items() if key != "type"}
        return create_workflows(generator_cfg, source="pegasus", seed=seed, resources=resources)
    raise ValueError(f"Unknown dataset type: {dataset_type}")


def build_env(workflows, resources, sim_config, seed: int) -> EdgeCloudWorkflowEnv:
    return EdgeCloudWorkflowEnv(
        workflows=workflows,
        resources=resources,
        sim_config=sim_config,
        seed=seed,
    )


def run_evaluation(
    agent,
    eval_workflows,
    resources,
    sim_config,
    eval_episodes: int,
    eval_seed: int,
) -> dict[str, Any]:
    episode_rows = []
    for index in range(1, eval_episodes + 1):
        env = build_env(
            eval_workflows,
            resources,
            sim_config,
            seed=eval_seed + index,
        )
        episode_result = agent.collect_episode(env, deterministic=True)
        summary = compute_summary(
            episode_result["workflow_results"],
            episode_result["event_log"],
            resources["servers"],
        )
        episode_rows.append(
            {
                "eval_episode": index,
                "steps": episode_result["steps"],
                **summary,
            }
        )

    count = max(len(episode_rows), 1)
    return {
        "eval_episodes": eval_episodes,
        "avg_total_energy": sum(row["total_energy"] for row in episode_rows) / count,
        "avg_response_time": sum(row["avg_workflow_response_time"] for row in episode_rows) / count,
        "avg_makespan": sum(row["avg_workflow_makespan"] for row in episode_rows) / count,
        "avg_utilization": sum(row["avg_server_utilization"] for row in episode_rows) / count,
        "avg_finished": sum(row["total_finished_workflows"] for row in episode_rows) / count,
        "episodes": episode_rows,
    }


def print_eval_summary(episode: int, metrics: Mapping[str, Any]) -> None:
    print(
        f"[eval] episode={episode} "
        f"energy={metrics['avg_total_energy']:.4f} "
        f"response={metrics['avg_response_time']:.4f} "
        f"makespan={metrics['avg_makespan']:.4f} "
        f"util={metrics['avg_utilization']:.4f} "
        f"finished={metrics['avg_finished']:.1f}"
    )


def main():
    args = parse_args()
    runner = ExperimentRunner(args.config, project_root=PROJECT_ROOT)
    config = runner.config
    training_cfg = config.get("training", {})
    episodes = args.episodes or int(training_cfg.get("episodes", 100))
    save_every = int(training_cfg.get("save_every", 10))
    eval_every = args.eval_every or int(training_cfg.get("eval_every", 5))
    eval_episodes = int(training_cfg.get("eval_episodes", 3))
    eval_seed = int(training_cfg.get("eval_seed", args.seed + 10000))
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "results/ppo_checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    dataset_config = config["dataset"]
    dataset_type = dataset_config.get("type", "alibaba_json")
    sim_config = config.get("simulation", {})

    resources = load_resources(
        config["resources"]["path"],
        num_edge=config["resources"]["num_edge"],
        num_cloud=config["resources"]["num_cloud"],
        seed=seed,
    )

    train_workflows = None
    if dataset_type != "pegasus":
        train_workflows = load_workflows(runner, dataset_config, seed, resources=resources)

    eval_dataset_config = copy.deepcopy(dataset_config)
    eval_dataset_config.update(config.get("eval_dataset", {}))
    eval_workflows = load_workflows(
        runner,
        eval_dataset_config,
        eval_seed,
        resources=resources,
    )

    agent_params = config["scheduler"]["params"].setdefault("agent", {})
    if args.device is not None:
        agent_params["device"] = args.device
    agent_params.setdefault("device", "cpu")

    scheduler = create_scheduler(
        "ppo",
        params=config["scheduler"]["params"],
        seed=seed,
    )
    print(describe_runtime_devices(scheduler.ppo.device))
    agent = scheduler.ppo

    history = []
    eval_history = []
    best_eval_energy = float("inf")
    best_checkpoint = checkpoint_dir / "ppo_best.pt"

    print(
        f"PPO train: episodes={episodes}, eval_every={eval_every}, "
        f"eval_episodes={eval_episodes}, dataset={dataset_type}"
    )

    for episode in range(1, episodes + 1):
        episode_seed = seed + episode
        if dataset_type == "pegasus":
            episode_workflows = load_workflows(
                runner,
                dataset_config,
                episode_seed,
                resources=resources,
            )
        else:
            episode_workflows = train_workflows

        env = build_env(
            episode_workflows,
            resources,
            sim_config,
            seed=episode_seed,
        )
        episode_result = agent.train_episode(env)
        metrics = episode_result.get("train_metrics", {})
        history.append({"episode": episode, **metrics})
        print(
            f"[train] episode={episode} "
            f"workflows={len(episode_workflows)} "
            f"steps={episode_result['steps']} "
            f"loss={metrics.get('loss', 'n/a')} "
            f"return={metrics.get('return_mean', 'n/a')} "
            f"clip={metrics.get('clip_fraction', 'n/a')}"
        )

        if eval_every > 0 and episode % eval_every == 0:
            eval_metrics = run_evaluation(
                agent,
                eval_workflows,
                resources,
                sim_config,
                eval_episodes=eval_episodes,
                eval_seed=eval_seed,
            )
            eval_metrics["episode"] = episode
            eval_history.append(eval_metrics)
            print_eval_summary(episode, eval_metrics)

            if eval_metrics["avg_total_energy"] < best_eval_energy:
                best_eval_energy = eval_metrics["avg_total_energy"]
                scheduler.save(str(best_checkpoint))
                print(f"best checkpoint updated: {best_checkpoint}")

        if episode % save_every == 0:
            checkpoint_path = checkpoint_dir / f"ppo_ep{episode}.pt"
            scheduler.save(str(checkpoint_path))
            metrics_path = checkpoint_dir / "train_history.json"
            with metrics_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {"train": history, "eval": eval_history},
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
