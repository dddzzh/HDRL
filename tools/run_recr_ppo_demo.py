from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.workflow_factory import WorkflowSampler, create_workflows
from src.device_utils import describe_runtime_devices, resolve_agent_device
from src.recr.recr_ppo import RECRPPO
from src.recr.state_reward import RECRStateReward
from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.metrics import compute_summary
from src.simulator.resource import load_resources


def parse_args():
    parser = argparse.ArgumentParser(
        description="RECR state/reward + PPO demo (Poisson arrivals, layered bandwidth)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/recr_ppo_demo.json",
        help="Path to RECR demo config JSON",
    )
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None, help="Run eval every N train episodes")
    parser.add_argument("--save-every", type=int, default=None, help="Save checkpoint every N train episodes")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workflows", type=int, default=None, help="Fixed workflow count per episode")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["pegasus", "alibaba_json", "alibaba_pool"],
        help=(
            "Workflow source: Pegasus generator, a fixed Alibaba JSON, or alibaba_pool "
            "(sample N workflows per episode from train/test pools)"
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0 | auto (default auto)")
    parser.add_argument(
        "--no-final-eval",
        action="store_true",
        help="Skip the final eval block after training (periodic eval still runs if eval_every > 0)",
    )
    return parser.parse_args()


def _load_config(path: str) -> dict:
    config_path = PROJECT_ROOT / path
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_env(workflows, resources, sim_config, seed: int) -> EdgeCloudWorkflowEnv:
    return EdgeCloudWorkflowEnv(
        workflows=workflows,
        resources=resources,
        sim_config=sim_config,
        seed=seed,
    )


def _generate_workflows(gen_cfg: dict, resources: dict, seed: int):
    return create_workflows(gen_cfg, source="pegasus", seed=seed, resources=resources)


def _resolve_dataset_type(config: dict, args) -> str:
    if args.dataset is not None:
        return args.dataset
    return str(config.get("dataset", {}).get("type", "pegasus"))


def _load_alibaba_workflows(config: dict, args, seed: int) -> list:
    dataset_cfg = dict(config.get("dataset", {}))
    if args.num_workflows is not None:
        dataset_cfg["max_workflows"] = args.num_workflows
    return create_workflows(dataset_cfg, source="alibaba_json", seed=seed)


def _build_pool_samplers(config: dict, resources: dict):
    dataset_cfg = dict(config.get("dataset", {}))
    gen_cfg = dict(config.get("generator", {}))
    train_path = dataset_cfg.get("train_path", "outputs/alibaba_train.json")
    test_path = dataset_cfg.get("test_path", "outputs/alibaba_test.json")
    mean_interarrival = float(
        dataset_cfg.get("mean_interarrival", gen_cfg.get("mean_interarrival", 60.0))
    )
    train_sampler = WorkflowSampler.from_file(
        train_path,
        mean_interarrival=mean_interarrival,
        mapping_config=dataset_cfg,
        resources=resources,
    )
    test_sampler = WorkflowSampler.from_file(
        test_path,
        mean_interarrival=mean_interarrival,
        mapping_config=dataset_cfg,
        resources=resources,
    )
    return train_sampler, test_sampler


def _resolve_pool_sample_count(config: dict, args) -> int:
    if args.num_workflows is not None:
        return int(args.num_workflows)
    dataset_cfg = dict(config.get("dataset", {}))
    gen_cfg = dict(config.get("generator", {}))
    return int(dataset_cfg.get("num_workflows", gen_cfg.get("num_workflows", 100)))


def _build_eval_workflow_sets(
    gen_cfg: dict,
    resources: dict,
    eval_episodes: int,
    eval_seed: int,
) -> list:
    return [
        _generate_workflows(gen_cfg, resources, eval_seed + index)
        for index in range(1, eval_episodes + 1)
    ]


def run_evaluation(
    agent: RECRPPO,
    eval_workflow_sets: list,
    resources: dict,
    sim_config: dict,
    eval_episodes: int,
    eval_seed: int,
) -> dict[str, Any]:
    episode_rows = []
    for index in range(1, eval_episodes + 1):
        env = _build_env(
            eval_workflow_sets[index - 1],
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
        "deadline_success_rate": _avg_optional(row.get("deadline_success_rate") for row in episode_rows),
        "episodes": episode_rows,
    }


def _avg_optional(values) -> float | None:
    rows = [float(value) for value in values if value is not None]
    return sum(rows) / len(rows) if rows else None


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
        print(
            "train "
            f"loss={metrics.get('loss', 'n/a'):.6f} "
            f"return={metrics.get('return_mean', 'n/a'):.6f} "
            f"clip={metrics.get('clip_fraction', 'n/a'):.6f}"
        )


def _print_eval_summary(train_episode: int, metrics: Mapping[str, Any]) -> None:
    deadline = metrics.get("deadline_success_rate")
    deadline_text = f"{deadline:.4f}" if deadline is not None else "n/a"
    print(
        f"\n=== [eval @ train episode {train_episode}] === "
        f"energy={metrics['avg_total_energy']:.4f} "
        f"response={metrics['avg_response_time']:.4f} "
        f"makespan={metrics['avg_makespan']:.4f} "
        f"finished={metrics['avg_finished']:.1f} "
        f"deadline_rate={deadline_text}"
    )


def _demo_state_reward(env_state: dict) -> None:
    recr = RECRStateReward()
    obs = recr.encode_state(env_state)
    print("\n--- RECR state snapshot ---")
    print(f"obs_dim={obs.vector.shape[0]}")
    print(f"ready_tasks={int(obs.ready_task_mask.sum())}, servers={int(obs.server_mask.sum())}")
    print(f"global_features={obs.global_features}")
    if obs.ready_task_mask.any():
        idx = int(obs.ready_task_mask.argmax())
        print(f"first_task_features={obs.ready_task_features[idx]}")


def _save_history(
    path: Path,
    train_history: list,
    eval_history: list,
    final_eval_summaries: list,
    gen_cfg: dict,
    reward_cfg: dict,
) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "train_history": train_history,
                "eval_history": eval_history,
                "eval_summaries": final_eval_summaries,
                "generator": gen_cfg,
                "reward": reward_cfg,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():
    args = parse_args()
    config = _load_config(args.config)

    train_cfg = config["training"]
    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    train_episodes = args.train_episodes or int(train_cfg.get("train_episodes", 5))
    eval_episodes = args.eval_episodes or int(train_cfg.get("eval_episodes", 2))
    eval_every = args.eval_every if args.eval_every is not None else int(train_cfg.get("eval_every", 5))
    save_every = args.save_every if args.save_every is not None else int(train_cfg.get("save_every", 10))
    eval_seed = int(train_cfg.get("eval_seed", seed + 10000))
    run_final_eval = not args.no_final_eval and bool(train_cfg.get("final_eval", True))

    checkpoint_path = PROJECT_ROOT / train_cfg.get("checkpoint", "results/recr_ppo_demo/recr_ppo.pt")
    best_checkpoint_path = checkpoint_path.parent / "recr_ppo_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_path.parent / "recr_demo_history.json"

    resources = load_resources(
        str(PROJECT_ROOT / config["resources"]["path"]),
        num_edge=config["resources"]["num_edge"],
        num_cloud=config["resources"]["num_cloud"],
        seed=seed,
    )
    sim_config = config["simulation"]
    agent_config = dict(config["agent"])
    if args.device is not None:
        agent_config["device"] = args.device
    device = resolve_agent_device(agent_config.get("device", "cpu"))

    dataset_type = _resolve_dataset_type(config, args)
    gen_cfg = dict(config.get("generator", {}))
    if args.num_workflows is not None and "num_workflows" in gen_cfg:
        gen_cfg["num_workflows"] = args.num_workflows

    train_sampler = None
    pool_sample_count = 0
    if dataset_type == "alibaba_pool":
        train_sampler, test_sampler = _build_pool_samplers(config, resources)
        pool_sample_count = _resolve_pool_sample_count(config, args)
        train_workflows = None
        eval_workflow_sets = [
            test_sampler.sample(pool_sample_count, eval_seed + index)
            for index in range(1, eval_episodes + 1)
        ]
        dataset_desc = (
            f"alibaba_pool (sample {pool_sample_count}/episode; "
            f"train pool={len(train_sampler)}, test pool={len(test_sampler)}; "
            f"fresh Poisson arrivals per episode)"
        )
    elif dataset_type == "alibaba_json":
        train_workflows = _load_alibaba_workflows(config, args, seed)
        eval_workflow_sets = [train_workflows for _ in range(eval_episodes)]
        dataset_desc = f"alibaba_json (fixed {len(train_workflows)} workflows per episode)"
    else:
        train_workflows = None
        eval_workflow_sets = _build_eval_workflow_sets(gen_cfg, resources, eval_episodes, eval_seed)
        dataset_desc = (
            f"random (num_workflows={gen_cfg.get('num_workflows')}, "
            f"mean_interarrival={gen_cfg.get('mean_interarrival')}s, regenerated per episode)"
        )

    print("RECR PPO Demo")
    print(describe_runtime_devices(device))
    print(f"dataset: {dataset_desc}")
    print(
        f"train_episodes={train_episodes}, eval_every={eval_every}, "
        f"eval_episodes={eval_episodes}, save_every={save_every}"
    )
    print(
        "Reward weights (energy min): "
        f"α={agent_config['reward']['alpha']} "
        f"β={agent_config['reward']['beta']} "
        f"γ={agent_config['reward']['gamma']}"
    )

    agent = RECRPPO(config=agent_config, seed=seed)
    train_history = []
    eval_history = []
    best_eval_energy = float("inf")

    for episode in range(1, train_episodes + 1):
        episode_seed = seed + episode
        if dataset_type == "alibaba_pool":
            workflows = train_sampler.sample(pool_sample_count, episode_seed)
        elif dataset_type == "alibaba_json":
            workflows = train_workflows
        else:
            workflows = _generate_workflows(gen_cfg, resources, episode_seed)
        env = _build_env(workflows, resources, sim_config, seed=episode_seed)
        if episode == 1:
            state = env.reset()
            _demo_state_reward(state)
        episode_result = agent.train_episode(env)
        train_history.append({"episode": episode, **episode_result.get("train_metrics", {})})
        _print_summary(f"Train episode {episode}", episode_result, resources)

        if eval_every > 0 and episode % eval_every == 0:
            eval_metrics = run_evaluation(
                agent,
                eval_workflow_sets,
                resources,
                sim_config,
                eval_episodes=eval_episodes,
                eval_seed=eval_seed,
            )
            eval_metrics["train_episode"] = episode
            eval_history.append(eval_metrics)
            _print_eval_summary(episode, eval_metrics)

            if eval_metrics["avg_total_energy"] < best_eval_energy:
                best_eval_energy = eval_metrics["avg_total_energy"]
                agent.save(str(best_checkpoint_path), extra={"demo": "recr_ppo_best", "config": config})
                print(f"best checkpoint updated: {best_checkpoint_path}")

        if save_every > 0 and episode % save_every == 0:
            periodic_path = checkpoint_path.parent / f"recr_ppo_ep{episode}.pt"
            agent.save(str(periodic_path), extra={"demo": "recr_ppo", "episode": episode, "config": config})
            _save_history(
                history_path,
                train_history,
                eval_history,
                [],
                gen_cfg,
                agent_config["reward"],
            )
            print(f"checkpoint saved: {periodic_path}")

    agent.save(str(checkpoint_path), extra={"demo": "recr_ppo", "config": config})
    print(f"\nCheckpoint saved: {checkpoint_path}")

    final_eval_summaries = []
    if run_final_eval:
        print("\n--- Final eval (deterministic) ---")
        final_eval = run_evaluation(
            agent,
            eval_workflow_sets,
            resources,
            sim_config,
            eval_episodes=eval_episodes,
            eval_seed=eval_seed,
        )
        final_eval["train_episode"] = train_episodes
        final_eval_summaries = final_eval["episodes"]
        _print_eval_summary(train_episodes, final_eval)
        for index, summary in enumerate(final_eval_summaries, start=1):
            print(
                f"  final eval {index}: "
                f"energy={summary['total_energy']:.4f} "
                f"response={summary['avg_workflow_response_time']:.4f} "
                f"makespan={summary['avg_workflow_makespan']:.4f}"
            )

    _save_history(
        history_path,
        train_history,
        eval_history,
        final_eval_summaries,
        gen_cfg,
        agent_config["reward"],
    )
    print(f"History saved: {history_path}")


if __name__ == "__main__":
    main()
