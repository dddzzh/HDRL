from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.runner import ExperimentRunner
from src.schedulers.registry import create_scheduler
from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.resource import load_resources


def parse_args():
    parser = argparse.ArgumentParser(description="Train A2C scheduler.")
    parser.add_argument("--config", default="configs/a2c_train_default.json")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    runner = ExperimentRunner(args.config, project_root=PROJECT_ROOT)
    config = runner.config
    training_cfg = config.get("training", {})
    episodes = args.episodes or int(training_cfg.get("episodes", 100))
    save_every = int(training_cfg.get("save_every", 10))
    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "results/a2c_checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    resources = load_resources(
        config["resources"]["path"],
        num_edge=config["resources"]["num_edge"],
        num_cloud=config["resources"]["num_cloud"],
        seed=seed,
    )
    workflows = runner._load_dataset(config["dataset"], seed, resources=resources)
    scheduler = create_scheduler(
        "a2c",
        params=config["scheduler"]["params"],
        seed=seed,
    )

    env = EdgeCloudWorkflowEnv(
        workflows=workflows,
        resources=resources,
        sim_config=config.get("simulation", {}),
        seed=seed,
    )

    history = []
    for episode in range(1, episodes + 1):
        episode_result = scheduler.a2c.train_episode(env)
        metrics = episode_result.get("train_metrics", {})
        history.append({"episode": episode, **metrics})
        print(
            f"episode={episode} "
            f"steps={episode_result['steps']} "
            f"loss={metrics.get('loss', 'n/a')} "
            f"return={metrics.get('return_mean', 'n/a')}"
        )

        if episode % save_every == 0:
            checkpoint_path = checkpoint_dir / f"a2c_ep{episode}.pt"
            scheduler.save(str(checkpoint_path))
            metrics_path = checkpoint_dir / "train_history.json"
            with metrics_path.open("w", encoding="utf-8") as file:
                json.dump(history, file, ensure_ascii=False, indent=2)
            print(f"checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
