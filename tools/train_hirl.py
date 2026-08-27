from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.baselines.dream.dream_state import SharedEdgeCloudStateBuilder
from src.baselines.hirl import HIRLAgent, HIRLConfig, HIRLScheduler
from src.simulator.metrics import compute_summary
from tools.hierarchical_common import build_env, build_resources, build_workflows, load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train adapted HIRL (TD3 + DDQN).")
    parser.add_argument("--config", default="configs/hirl_td3_ddqn.json")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--resume")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def run_episode(env, scheduler, training=True):
    env.reset()
    steps = 0
    total_reward = upper_total = lower_total = 0.0
    rho_values, loads, updates, reliability_costs = [], [], [], []
    max_steps = sum(env._workflow_task_count(row["workflow"]) for row in env.workflow_states) + 1000
    while not env.done and steps < max_steps:
        state = env.get_state()
        if not state["ready_tasks"]:
            if not env._advance_to_next_decision_point():
                env._update_done()
                break
            continue
        task = state["ready_tasks"][0]
        policy, decision, env_action = scheduler.select_action(env, training=training)
        rho1_reference = env.estimate_action_metrics_ref_rho1(env_action)
        next_state, _, done, info = env.step(env_action)
        while not done and not next_state["ready_tasks"]:
            if not env._advance_to_next_decision_point():
                env._update_done()
                break
            next_state, done = env.get_state(), env.done
        if done or not next_state["ready_tasks"]:
            next_vector = np.zeros(scheduler.agent.state_dim, np.float32)
        else:
            next_task = next_state["ready_tasks"][0]
            next_workflow = next_state["workflows"][next_task["workflow_id"]]
            next_vector = scheduler.state_builder.build_state(env, next_task, next_workflow)
        upper, lower, failure, reward_info = scheduler.rewards(
            env, state, task, info["event"], rho1_reference
        )
        if training:
            scheduler.agent.remember(policy, upper, lower, next_vector, done, failure)
            update = scheduler.agent.update()
            if update:
                updates.append(update)
        total_reward += upper + lower
        upper_total += upper
        lower_total += lower
        rho_values.append(decision.rho)
        loads.append(decision.load)
        reliability_costs.append(float(
            reward_info.get("reliability_violation_delta", 0.0)
        ))
        steps += 1
    if steps >= max_steps and not env.done:
        raise RuntimeError("HIRL episode exceeded max steps.")
    summary = compute_summary(env._build_workflow_results(), env.event_log, env.servers)
    metrics = {
        "episode_reward": total_reward,
        "upper_reward": upper_total,
        "lower_reward": lower_total,
        "steps": steps,
        "rho_mean": float(np.mean(rho_values)) if rho_values else None,
        "load_mean": float(np.mean(loads)) if loads else None,
        "low_reliability_cost_mean": (
            float(np.mean(reliability_costs)) if reliability_costs else 0.0
        ),
        "low_reliability_cost_total": float(np.sum(reliability_costs)),
        **summary,
    }
    for key in (
        "td3_critic_loss", "td3_actor_loss", "td3_td_abs_mean", "ddqn_loss",
        "ddqn_td_abs_mean", "ddqn_learning_rate",
    ):
        values = [float(row[key]) for row in updates if row.get(key) is not None]
        metrics[key] = float(np.mean(values)) if values else None
    return metrics


def build_scheduler(config, resources, seed, smoke_test=False, device=None):
    state_cfg = dict(config.get("agent", {}).get("shared", {}).get("state", {}))
    state_cfg["max_servers"] = len(resources["servers"])
    state_builder = SharedEdgeCloudStateBuilder(state_cfg)
    values = dict(config.get("agent", {}).get("hirl", {}))
    values["seed"] = int(seed)
    if device:
        values["device"] = device
    if smoke_test:
        values.update(batch_size=4, td3_replay_size=64, ddqn_replay_size=64, hidden_dim=32)
    agent = HIRLAgent(
        state_builder.state_dim, len(resources["servers"]), HIRLConfig.from_mapping(values)
    )
    return HIRLScheduler(agent, state_builder)


def run_training(config, args):
    seed = args.seed if args.seed is not None else int(config.get("training", {}).get("seed", 42))
    resources = build_resources(config, seed)
    scheduler = build_scheduler(config, resources, seed, args.smoke_test, args.device)
    start_episode, best = 0, -float("inf")
    if args.resume:
        payload = torch.load(args.resume, map_location=scheduler.agent.device, weights_only=False)
        scheduler.agent.load_checkpoint(
            payload,
            load_replay=("td3_replay" in payload and "ddqn_replay" in payload),
        )
        start_episode = int(payload["episode"]) + 1
        best = float(payload["best_episode_reward"])
    episodes = args.episodes if args.episodes is not None else int(config["training"].get("total_episodes", 800))
    if args.smoke_test:
        episodes = min(episodes, 2)
    output = (
        Path(args.resume).resolve().parent if args.resume else
        PROJECT_ROOT / config["training"].get("checkpoint_dir", "results/hirl_td3_ddqn") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=True)
    resolved = {
        **config,
        "agent": {**config.get("agent", {}), "hirl": scheduler.agent.config.as_dict()},
        "training": {**config.get("training", {}), "total_episodes": episodes},
        "adaptation": {
            "paper": "HIRL: Hierarchical Reinforcement Learning for Coordinated Resource Management in Heterogeneous Edge Computing",
            "state": "project shared high-level state",
            "action": "project server_id + continuous rho",
            "tier_order": "TD3 rho then DDQN server allocation"
        }
    }
    (output / "resolved_config.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")
    history_path = output / "history.csv"
    save_every = max(1, int(config.get("training", {}).get("save_every", 20)))
    fields = [
        "episode", "episode_reward", "upper_reward", "lower_reward", "steps",
        "rho_mean", "load_mean", "td3_critic_loss", "td3_actor_loss", "ddqn_loss",
        "ddqn_learning_rate",
        "total_energy", "deadline_success_rate", "reliability_success_rate",
        "low_reliability_cost_mean", "low_reliability_cost_total",
        "avg_workflow_response_time", "avg_server_utilization",
    ]
    for episode in range(start_episode, episodes):
        workflows = build_workflows(config, resources, seed + episode, smoke_test=args.smoke_test)
        simulation = {**config.get("simulation", {}), "enable_backup_duplication": False}
        env = build_env(workflows, resources, simulation, seed + episode)
        row = run_episode(env, scheduler, training=True)
        row["episode"] = episode
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        is_best = row["episode_reward"] > best
        best = max(best, row["episode_reward"])
        if is_best:
            torch.save(
                scheduler.agent.checkpoint(episode, best, include_replay=False)
                | {"experiment_config": resolved},
                output / "best_checkpoint.pt",
            )
        if (episode + 1) % save_every == 0 or episode + 1 == episodes:
            payload = scheduler.agent.checkpoint(episode, best, include_replay=False)
            payload["experiment_config"] = resolved
            torch.save(payload, output / "latest_checkpoint.pt")
            torch.save(
                payload,
                output / f"checkpoint_ep{episode + 1:06d}.pt",
            )
        print(json.dumps({key: row.get(key) for key in fields}, ensure_ascii=False))
    print(f"output_dir={output}")
    return output


def main():
    args = parse_args()
    run_training(load_json(args.config), args)


if __name__ == "__main__":
    main()
