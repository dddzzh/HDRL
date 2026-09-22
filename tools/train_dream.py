from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_CPU_THREADS = os.environ.get("DRLEC_CPU_THREADS", "1")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = _CPU_THREADS

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.device_utils import configure_torch_runtime, describe_device

from src.baselines.dream import DReaMAgent, DReaMConfig, DReaMScheduler
from src.baselines.dream.dream_action import DREAM_RHO_LEVELS, action_dimension
from src.baselines.dream.dream_reward import ExistingEnvironmentReward
from src.baselines.dream.dream_state import SharedEdgeCloudStateBuilder
from src.simulator.metrics import compute_summary
from src.D3QN.reward import HierarchicalObjective
from src.baselines.dream.dream_reward import SharedWorkflowReward
from tools.hierarchical_common import (
    attach_satisfaction_metrics,
    build_env,
    build_resources,
    build_workflows,
    checkpoint_validation_seeds,
    combine_checkpoint_metrics,
    format_satisfaction_summary,
    load_json,
    metric_rank,
    satisfaction_field_names,
    write_checkpoint_selection,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Adapted DReaM-DQN.")
    parser.add_argument("--config", default="configs/dream_dqn.json")
    parser.add_argument("--episodes", type=int); parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float); parser.add_argument("--gamma", type=float)
    parser.add_argument("--batch-size", type=int); parser.add_argument("--epsilon-decay", type=float)
    parser.add_argument("--device"); parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume")
    return parser.parse_args()


def run_episode(env, scheduler, training=True):
    env.reset(); episode_reward = 0.0; updates = []; steps = 0; rho_values = []
    reliability_costs = []
    if hasattr(scheduler.reward_function, "reset_episode"): scheduler.reward_function.reset_episode()
    max_steps = sum(env._workflow_task_count(x["workflow"]) for x in env.workflow_states) + 1000
    while not env.done and steps < max_steps:
        state = env.get_state()
        if not state["ready_tasks"]:
            if not env._advance_to_next_decision_point(): env._update_done(); break
            continue
        vector, action_id, decision, env_action = scheduler.select_action(env, training=training)
        task = state["ready_tasks"][0]; workflow = state["workflows"][task["workflow_id"]]
        next_observation, environment_reward, done, info = env.step(env_action)
        rho_values.append(float(env_action["rho"]))
        while not done and not next_observation["ready_tasks"]:
            if not env._advance_to_next_decision_point(): env._update_done(); break
            next_observation, done = env.get_state(), env.done
        next_task = next_observation["ready_tasks"][0] if next_observation["ready_tasks"] else None
        next_workflow = next_observation["workflows"].get(next_task["workflow_id"]) if next_task else None
        next_vector = scheduler.state_builder.build_state(env, next_task, next_workflow)
        reward, reward_info = scheduler.reward_function.compute_reward(
            env, task, workflow, decision,
            {"pre_state": state, "env_action": env_action,
             "environment_reward": environment_reward, "environment_info": info},
        )
        if training:
            scheduler.agent.replay_buffer.add(vector, action_id, reward, next_vector, done)
            scheduler.agent.after_environment_step()
        reliability_costs.append(float(
            reward_info.get("components", {}).get(
                "reliability_violation_delta", 0.0
            )
        ))
        episode_reward += reward; steps += 1
    if steps >= max_steps and not env.done: raise RuntimeError("DReaM episode exceeded max steps.")
    if training:
        for _ in range(scheduler.agent.config.training_per_episode):
            update = scheduler.agent.update()
            if update:
                updates.append(update)
    summary = compute_summary(env._build_workflow_results(), env.event_log, env.servers)
    update_metrics = {}
    if updates:
        mean_keys = (
            "loss", "q_mean", "target_mean", "td_abs_mean",
            "grad_norm_before_clip", "grad_norm_after_clip",
        )
        max_keys = (
            "q_abs_max", "target_abs_max", "td_abs_max",
            "parameter_abs_max",
        )
        update_metrics.update({
            f"mean_{key}": float(np.mean([row[key] for row in updates]))
            for key in mean_keys
        })
        update_metrics.update({
            f"max_{key}": float(np.max([row[key] for row in updates]))
            for key in max_keys
        })
    return {"episode_reward": episode_reward,
            "mean_loss": update_metrics.get("mean_loss"),
            "steps": steps, "rho_mean": float(np.mean(rho_values)) if rho_values else None,
            "low_reliability_cost_mean": (
                float(np.mean(reliability_costs)) if reliability_costs else 0.0
            ),
            "low_reliability_cost_total": float(np.sum(reliability_costs)),
            "reward_source": reward_info["source"] if steps else None,
            **update_metrics, **summary}


def evaluate_scheduler(config, resources, scheduler, *, seed, episodes, smoke_test=False):
    rows = []
    for offset in range(int(episodes)):
        episode_seed = int(seed) + offset
        workflows = build_workflows(
            config, resources, episode_seed, smoke_test=smoke_test
        )
        env = build_env(
            workflows,
            resources,
            {
                **config.get("simulation", {}),
                "enable_backup_duplication": False,
            },
            episode_seed,
        )
        rows.append(run_episode(env, scheduler, training=False))
    metrics = {}
    for key in (
        "episode_reward",
        "total_energy",
        "deadline_success_rate",
        "reliability_success_rate",
        "rho_mean",
        "avg_workflow_response_time",
    ):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        metrics[key] = float(np.mean(values)) if values else None
    metrics["eval_episodes"] = int(episodes)
    metrics["eval_seed"] = int(seed)
    return attach_satisfaction_metrics(metrics, rows)


def run_training(config, args):
    runtime_threads = configure_torch_runtime(
        config.get("training", {}).get("torch_cpu_threads", 1)
    )
    seed = args.seed if args.seed is not None else int(config["training"].get("seed", 42))
    resources = build_resources(config, seed); state_cfg = dict(config["agent"].get("shared", {}).get("state", {}))
    state_cfg["max_servers"] = len(resources["servers"])
    builder = SharedEdgeCloudStateBuilder(state_cfg)
    dream_values = dict(config["agent"].get("dream", {})); dream_values["seed"] = seed
    for key, value in (("learning_rate", args.lr), ("gamma", args.gamma), ("batch_size", args.batch_size), ("epsilon_decay", args.epsilon_decay), ("device", args.device)):
        if value is not None: dream_values[key] = value
    if args.smoke_test: dream_values.update(batch_size=8, replay_capacity=64)
    dream_cfg = DReaMConfig.from_mapping(dream_values)
    agent = DReaMAgent(builder.state_dim, action_dimension(len(resources["servers"]), False), dream_cfg)
    print(
        "[device] "
        f"dream={describe_device(agent.device)} simulation=cpu "
        f"dvfs=server_x_{len(DREAM_RHO_LEVELS)}levels[rho_ee,1] "
        f"update=episode_end_x{dream_cfg.training_per_episode} "
        f"torch_threads={runtime_threads['torch_cpu_threads']} "
        f"interop_threads={runtime_threads['torch_interop_threads']}"
    )
    objective_config = dict(
        config.get("objective", {"reward_mode": "deadline_incremental_only"})
    )
    objective = HierarchicalObjective(objective_config)
    scheduler = DReaMScheduler(
        agent,
        builder,
        SharedWorkflowReward(
            objective,
            fixed_reliability_lambda=objective_config.get(
                "fixed_reliability_lambda", 0.0
            ),
        ),
        False,
    )
    training_config = config["training"]
    checkpoint_deadline_target = training_config.get(
        "checkpoint_deadline_target"
    )
    if checkpoint_deadline_target is not None:
        checkpoint_deadline_target = float(checkpoint_deadline_target)
    checkpoint_reliability_target = float(
        training_config.get("checkpoint_reliability_target", 1.0)
    )
    checkpoint_deadline_esp_threshold = training_config.get(
        "checkpoint_deadline_esp_threshold"
    )
    checkpoint_deadline_esp_target = training_config.get(
        "checkpoint_deadline_esp_target"
    )
    if checkpoint_deadline_esp_threshold is not None:
        checkpoint_deadline_esp_threshold = float(
            checkpoint_deadline_esp_threshold
        )
    if checkpoint_deadline_esp_target is not None:
        checkpoint_deadline_esp_target = float(checkpoint_deadline_esp_target)
    eval_seeds = checkpoint_validation_seeds(training_config, seed + 100000)
    save_replay = bool(training_config.get("save_replay_buffer", False))
    checkpoint_policy = {
        "mode": "all_blocks_esp_feasible_then_min_mean_energy",
        "reliability_target": checkpoint_reliability_target,
        "deadline_esp_threshold": checkpoint_deadline_esp_threshold,
        "deadline_esp_target": checkpoint_deadline_esp_target,
        "eval_seeds": eval_seeds,
    }
    if checkpoint_deadline_target is not None:
        checkpoint_policy["deadline_target"] = checkpoint_deadline_target
    start_episode, best = 0, -float("inf")
    best_metric_rank = None
    best_metrics = None
    if args.resume:
        payload = torch.load(args.resume, map_location=agent.device, weights_only=False); agent.load_checkpoint(payload)
        if "replay_buffer" in payload: agent.load_replay_state_dict(payload["replay_buffer"])
        start_episode, best = int(payload["episode"]) + 1, float(payload["best_episode_reward"])
        saved_rank = payload.get("best_metric_rank")
        if saved_rank is not None and payload.get("checkpoint_policy") == checkpoint_policy:
            best_metric_rank = tuple(float(value) for value in saved_rank)
            best_metrics = payload.get("best_metrics")
        elif saved_rank is not None:
            print("[checkpoint] reset legacy best rank because the selection policy changed")
    episodes = args.episodes if args.episodes is not None else int(config["training"].get("total_episodes", dream_cfg.training_episodes))
    if args.smoke_test: episodes = min(episodes, 2)
    checkpoint_root = PROJECT_ROOT / config["training"].get("checkpoint_dir", "results/dream_dqn")
    if args.resume:
        output = Path(args.resume).resolve().parent
    else:
        output = checkpoint_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        **config,
        "agent": {**config.get("agent", {}), "dream": dream_cfg.as_dict()},
        "training": {**config.get("training", {}), "total_episodes": episodes},
    }
    (output / "resolved_config.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    history_path = output / "history.csv"
    fields = [
        "episode", "episode_reward", "mean_loss", "epsilon", "replay_size",
        "environment_steps", "gradient_steps", "mean_q_mean", "max_q_abs_max",
        "mean_target_mean", "max_target_abs_max", "mean_td_abs_mean",
        "max_td_abs_max", "mean_grad_norm_before_clip",
        "mean_grad_norm_after_clip", "max_parameter_abs_max", "total_energy",
        "deadline_success_rate", "reliability_success_rate", "rho_mean",
        "low_reliability_cost_mean", "low_reliability_cost_total",
        "total_finished_workflows",
        "avg_workflow_response_time", "avg_server_utilization",
    ]
    eval_path = output / "eval_history.csv"
    eval_fields = [
        "episode", "eval_seed", "eval_episodes", "episode_reward",
        "total_energy", "deadline_success_rate", "reliability_success_rate",
        "rho_mean", "low_reliability_cost_mean",
        "low_reliability_cost_total", "avg_workflow_response_time",
        *satisfaction_field_names(),
    ]
    eval_interval = int(config["training"].get("eval_interval", 20))
    eval_episodes = int(config["training"].get("eval_episodes", 10))
    save_every = int(config["training"].get("save_every", 0))
    esp_description = (
        "ESP=disabled"
        if checkpoint_deadline_esp_threshold is None
        else (
            f"ESP@{checkpoint_deadline_esp_threshold:.2f}>="
            f"{checkpoint_deadline_esp_target:.2f}"
        )
    )
    print(
        "[checkpoint] mode=all_blocks_esp_feasible_then_min_mean_energy "
        "mean_deadline=diagnostic_only "
        f"{esp_description} "
        f"reliability_target={checkpoint_reliability_target:.4f} "
        f"eval_seeds={eval_seeds} eval_episodes={eval_episodes}"
    )
    if args.smoke_test:
        eval_interval = 1
        eval_episodes = 1
    for episode in range(start_episode, episodes):
        workflows = build_workflows(config, resources, seed + episode, smoke_test=args.smoke_test)
        env = build_env(workflows, resources, {**config.get("simulation", {}), "enable_backup_duplication": False}, seed + episode)
        try:
            row = run_episode(env, scheduler, True)
            health = agent.assert_healthy(f"episode {episode}")
        except FloatingPointError as error:
            failure = {
                "episode": episode,
                "environment_steps": agent.environment_steps,
                "gradient_steps": agent.gradient_steps,
                "error": str(error),
            }
            (output / "divergence_failure.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise RuntimeError(
                f"DReaM training stopped by divergence guard at episode {episode}: {error}"
            ) from error
        row.update(
            episode=episode,
            epsilon=agent.epsilon,
            replay_size=len(agent.replay_buffer),
            environment_steps=agent.environment_steps,
            gradient_steps=agent.gradient_steps,
            max_parameter_abs_max=max(
                float(row.get("max_parameter_abs_max", 0.0) or 0.0),
                float(health["parameter_abs_max"]),
            ),
        )
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            if stream.tell() == 0: writer.writeheader()
            writer.writerow(row)
        best = max(best, float(row["episode_reward"]))
        should_evaluate = (
            eval_interval > 0
            and ((episode + 1) % eval_interval == 0 or episode == episodes - 1)
        )
        if should_evaluate:
            validation_blocks = []
            eval_rows = []
            for eval_seed in eval_seeds:
                eval_metrics = evaluate_scheduler(
                    config,
                    resources,
                    scheduler,
                    seed=eval_seed,
                    episodes=eval_episodes,
                    smoke_test=args.smoke_test,
                )
                eval_row = {"episode": int(episode), **eval_metrics}
                eval_rows.append(eval_row)
                validation_blocks.append(eval_metrics)
                with eval_path.open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, eval_fields, extrasaction="ignore")
                    if stream.tell() == 0:
                        writer.writeheader()
                    writer.writerow(eval_row)
                print(
                    f"[eval] ep={int(episode)} seed={int(eval_seed)} "
                    f"{format_satisfaction_summary(eval_row, include_mean=False)} "
                    f"E={float(eval_metrics['total_energy']) / 1000:.2f}kJ",
                    flush=True,
                )
            checkpoint_metrics = combine_checkpoint_metrics(validation_blocks)
            current_rank = metric_rank(
                checkpoint_metrics,
                deadline_target=checkpoint_deadline_target,
                reliability_target=checkpoint_reliability_target,
                deadline_esp_threshold=checkpoint_deadline_esp_threshold,
                deadline_esp_target=checkpoint_deadline_esp_target,
            )
        else:
            current_rank = None
        if current_rank is not None and (
            best_metric_rank is None or current_rank > best_metric_rank
        ):
            best_metric_rank = current_rank
            best_metrics = {
                "episode": int(episode),
                **checkpoint_metrics,
                "validation_blocks": eval_rows,
            }
            checkpoint = agent.checkpoint(episode, best)
            checkpoint["experiment_config"] = resolved_config
            checkpoint["checkpoint_policy"] = checkpoint_policy
            checkpoint["best_metric_rank"] = list(best_metric_rank)
            checkpoint["best_metrics"] = best_metrics
            checkpoint["checkpoint_feasible"] = bool(best_metric_rank[0] >= 1.0)
            torch.save(checkpoint, output / "best_checkpoint.pt")
        if current_rank is not None:
            write_checkpoint_selection(
                output / "checkpoint_selection.json",
                checkpoint_policy,
                best_metrics,
                best_metric_rank,
                "best_checkpoint.pt",
            )
        latest = agent.checkpoint(episode, best)
        latest["experiment_config"] = resolved_config
        latest["checkpoint_policy"] = checkpoint_policy
        latest["best_metric_rank"] = (
            None if best_metric_rank is None else list(best_metric_rank)
        )
        latest["best_metrics"] = best_metrics
        latest["checkpoint_feasible"] = bool(
            best_metric_rank is not None and best_metric_rank[0] >= 1.0
        )
        if save_replay:
            latest["replay_buffer"] = agent.replay_state_dict()
        torch.save(latest, output / "latest_checkpoint.pt")
        if save_every > 0 and (episode + 1) % save_every == 0:
            periodic = agent.checkpoint(episode, best)
            periodic["experiment_config"] = resolved_config
            periodic["checkpoint_policy"] = checkpoint_policy
            periodic["best_metric_rank"] = (
                None if best_metric_rank is None else list(best_metric_rank)
            )
            periodic["best_metrics"] = best_metrics
            torch.save(
                periodic,
                output / f"checkpoint_ep{episode + 1:04d}.pt",
            )
        print(json.dumps({
            key: row.get(key)
            for key in fields
            if key != "deadline_success_rate"
        }, ensure_ascii=False))
    print(f"output_dir={output}")


def main():
    args = parse_args(); run_training(load_json(args.config), args)


if __name__ == "__main__": main()
