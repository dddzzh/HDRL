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

from src.baselines.dqn_edge import (
    DQNEdgeAgent,
    DQNEdgeCommonReward,
    DQNEdgeCommonStateBuilder,
    DQNEdgeConfig,
    DQNEdgeNormalizedPaperReward,
    DQNEdgePaperReward,
    DQNEdgeScheduler,
    DQNEdgeStateBuilder,
)
from src.simulator.metrics import compute_summary
from tools.hierarchical_common import (
    attach_satisfaction_metrics,
    build_env,
    build_resources,
    build_workflows,
    checkpoint_validation_seeds,
    combine_checkpoint_metrics,
    configure_agent_state,
    format_satisfaction_summary,
    load_json,
    metric_rank,
    satisfaction_field_names,
    write_checkpoint_selection,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the adapted DQN-Edge reproduction.")
    parser.add_argument("--config", default="configs/dqn_edge_small_20edge_6cloud.json")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume")
    return parser.parse_args()


def build_dqn_edge_runtime(
    config,
    resources,
    agent_config,
    *,
    smoke_test=False,
    variant=None,
    state_config=None,
    objective_config=None,
):
    variant = str(
        variant or config.get("agent", {}).get("variant", "paper")
    ).strip().lower()
    if variant == "paper":
        builder = DQNEdgeStateBuilder(len(resources["servers"]))
        reward = DQNEdgePaperReward()
        effective_state_config = None
        effective_objective_config = None
    elif variant == "paper_normalized":
        builder = DQNEdgeStateBuilder(len(resources["servers"]))
        reward = DQNEdgeNormalizedPaperReward()
        effective_state_config = None
        effective_objective_config = None
    elif variant == "common_state_paper_reward":
        # Fair simulator/state adapter, but retain the paper's two independent
        # objective rewards (Eq. 28).  The latency and energy DQNs must not be
        # trained with an already-composed scalar upper reward because their
        # Q-values are fused later by DQN-Edge's learned attention weights.
        if state_config is None:
            configured_agent = configure_agent_state(
                config, resources, smoke_test=smoke_test
            )
            state_config = configured_agent["shared"]["state"]
        effective_state_config = dict(state_config)
        effective_state_config["max_servers"] = len(resources["servers"])
        builder = DQNEdgeCommonStateBuilder(
            len(resources["servers"]), effective_state_config
        )
        reward = DQNEdgePaperReward()
        effective_objective_config = None
    elif variant in {"common_reward", "common_upper"}:
        effective_objective_config = dict(
            objective_config
            if objective_config is not None
            else config.get("objective", {})
        )
        reward = DQNEdgeCommonReward(effective_objective_config)
        if variant == "common_reward":
            # Only the two objective rewards change. Task ranking, Eq. 27
            # state, VM action, dual Q networks, and attention remain paper-side.
            builder = DQNEdgeStateBuilder(len(resources["servers"]))
            effective_state_config = None
        else:
            if state_config is None:
                configured_agent = configure_agent_state(
                    config, resources, smoke_test=smoke_test
                )
                state_config = configured_agent["shared"]["state"]
            effective_state_config = dict(state_config)
            effective_state_config["max_servers"] = len(resources["servers"])
            builder = DQNEdgeCommonStateBuilder(
                len(resources["servers"]), effective_state_config
            )
    else:
        raise ValueError(
            "DQN-Edge agent.variant must be 'paper', "
            "'paper_normalized', 'common_state_paper_reward', 'common_reward', or "
            "'common_upper'."
        )
    agent = DQNEdgeAgent(
        builder.state_dim, len(resources["servers"]), agent_config
    )
    scheduler = DQNEdgeScheduler(
        agent, builder, reward, agent_config.rank_critical_beta
    )
    return {
        "variant": variant,
        "state_config": effective_state_config,
        "objective_config": effective_objective_config,
        "builder": builder,
        "reward": reward,
        "agent": agent,
        "scheduler": scheduler,
    }


def run_episode(env, scheduler, training=True):
    env.reset(); scheduler.reward_function.reset_episode(); scheduler.state_builder.reset_episode()
    total_reward, steps, updates, weights = 0.0, 0, [], []
    max_steps = sum(env._workflow_task_count(row["workflow"]) for row in env.workflow_states) + 1000
    while not env.done and steps < max_steps:
        state = env.get_state()
        if not state["ready_tasks"]:
            if not env._advance_to_next_decision_point():
                env._update_done(); break
            continue
        vector, context, action_id, attention_weights, task, workflow_state, action = scheduler.select_action(env, training)
        reward_state = state
        first_task = state["ready_tasks"][0]
        selected_key = (str(task["workflow_id"]), int(task["task_id"]))
        first_key = (str(first_task["workflow_id"]), int(first_task["task_id"]))
        if selected_key != first_key:
            # The paper's upward-rank dispatcher may choose a task other than
            # the simulator's common first ready task. Rebuild only the reward
            # decision context so energy/deadline references match the action.
            reward_state = dict(state)
            reward_state["ready_tasks"] = [
                task,
                *[
                    row
                    for row in state["ready_tasks"]
                    if (str(row["workflow_id"]), int(row["task_id"]))
                    != selected_key
                ],
            ]
            reward_state["decision_context"] = (
                env.build_hierarchical_decision_context(
                    task["workflow_id"], task["task_id"]
                )
            )
        next_observation, env_reward, done, info = env.step(action)
        time_reward, energy_reward, attention_target, reward_info = (
            scheduler.reward_function.compute(
                task,
                workflow_state,
                info,
                env=env,
                state=reward_state,
                action=action,
                env_reward=env_reward,
            )
        )
        while not done and not next_observation["ready_tasks"]:
            if not env._advance_to_next_decision_point():
                env._update_done(); break
            next_observation, done = env.get_state(), env.done
        next_task = scheduler.state_builder.select_ranked_ready_task(env, scheduler.rank_critical_beta) if next_observation["ready_tasks"] else None
        next_workflow = next_observation["workflows"][next_task["workflow_id"]] if next_task else None
        next_vector = scheduler.state_builder.build_state(env, next_task, next_workflow) if next_task else np.zeros_like(vector)
        if training:
            scheduler.agent.replay_buffer.add(vector, action_id, time_reward, energy_reward,
                                              next_vector, done, context, attention_target)
            update = scheduler.agent.update()
            if update: updates.append(update)
            scheduler.agent.after_environment_step()
        combined_reward = float(attention_weights[0] * time_reward + attention_weights[1] * energy_reward)
        total_reward += combined_reward; weights.append(attention_weights); steps += 1
    if steps >= max_steps and not env.done:
        raise RuntimeError("DQN-Edge episode exceeded maximum steps.")
    result = compute_summary(env._build_workflow_results(), env.event_log, env.servers)
    return {"episode_reward": total_reward, "steps": steps,
            "mean_time_loss": float(np.mean([x["time_loss"] for x in updates])) if updates else None,
            "mean_energy_loss": float(np.mean([x["energy_loss"] for x in updates])) if updates else None,
            "mean_attention_loss": float(np.mean([x["attention_loss"] for x in updates])) if updates else None,
            "mean_time_q_action_std": float(np.mean([x["time_q_action_std"] for x in updates])) if updates else None,
            "mean_energy_q_action_std": float(np.mean([x["energy_q_action_std"] for x in updates])) if updates else None,
            "mean_weighted_q_scale_ratio": float(np.mean([x["weighted_q_scale_ratio"] for x in updates])) if updates else None,
            "mean_time_weight": float(np.mean([x[0] for x in weights])) if weights else None,
            "mean_energy_weight": float(np.mean([x[1] for x in weights])) if weights else None,
            "reward_source": reward_info["source"] if steps else None, **result}


def evaluate_scheduler(config, resources, scheduler, *, seed, episodes, smoke_test=False):
    rows = []
    for offset in range(int(episodes)):
        episode_seed = int(seed) + offset
        workflows = build_workflows(
            config, resources, episode_seed, smoke_test=smoke_test
        )
        simulation = {
            **config.get("simulation", {}),
            "enable_backup_duplication": False,
            "default_rho": 1.0,
        }
        env = build_env(workflows, resources, simulation, episode_seed)
        rows.append(run_episode(env, scheduler, training=False))
    metrics = {}
    for key in (
        "episode_reward",
        "total_energy",
        "deadline_success_rate",
        "reliability_success_rate",
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
    resources = build_resources(config, seed)
    values = dict(config["agent"]["dqn_edge"]); values["seed"] = seed
    if args.device: values["device"] = args.device
    if args.smoke_test: values.update(batch_size=8, replay_capacity=64)
    agent_config = DQNEdgeConfig.from_mapping(values)
    runtime = build_dqn_edge_runtime(
        config,
        resources,
        agent_config,
        smoke_test=args.smoke_test,
    )
    agent, scheduler = runtime["agent"], runtime["scheduler"]
    print(
        "[device] "
        f"dqn_edge={describe_device(agent.device)} simulation=cpu "
        "rho=fixed_1 "
        f"torch_threads={runtime_threads['torch_cpu_threads']} "
        f"interop_threads={runtime_threads['torch_interop_threads']} "
        f"variant={runtime['variant']} "
        f"reward={type(runtime['reward']).__name__}"
    )
    episodes = args.episodes if args.episodes is not None else int(config["training"].get("total_episodes", agent_config.training_episodes))
    if args.smoke_test: episodes = min(episodes, 2)
    checkpoint_deadline_target = config["training"].get(
        "checkpoint_deadline_target"
    )
    if checkpoint_deadline_target is not None:
        checkpoint_deadline_target = float(checkpoint_deadline_target)
    checkpoint_reliability_target = float(
        config["training"].get("checkpoint_reliability_target", 1.0)
    )
    checkpoint_deadline_esp_threshold = config["training"].get(
        "checkpoint_deadline_esp_threshold"
    )
    checkpoint_deadline_esp_target = config["training"].get(
        "checkpoint_deadline_esp_target"
    )
    if checkpoint_deadline_esp_threshold is not None:
        checkpoint_deadline_esp_threshold = float(
            checkpoint_deadline_esp_threshold
        )
    if checkpoint_deadline_esp_target is not None:
        checkpoint_deadline_esp_target = float(checkpoint_deadline_esp_target)
    eval_seeds = checkpoint_validation_seeds(
        config["training"], seed + 100000
    )
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
        payload = torch.load(args.resume, map_location=agent.device, weights_only=False)
        saved_variant = str(payload.get("variant", "paper")).strip().lower()
        if saved_variant != runtime["variant"]:
            raise ValueError(
                "DQN-Edge checkpoint variant mismatch: "
                f"checkpoint={saved_variant}, config={runtime['variant']}."
            )
        agent.load_checkpoint(payload); start_episode = int(payload["episode"]) + 1
        best = float(payload["best_episode_reward"])
        saved_rank = payload.get("best_metric_rank")
        if saved_rank is not None and payload.get("checkpoint_policy") == checkpoint_policy:
            best_metric_rank = tuple(float(value) for value in saved_rank)
            best_metrics = payload.get("best_metrics")
        elif saved_rank is not None:
            print("[checkpoint] reset legacy best rank because the selection policy changed")
    output = (Path(args.resume).resolve().parent if args.resume else
              PROJECT_ROOT / config["training"].get("checkpoint_dir", "results/dqn_edge") / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=True)
    resolved = {
        **config,
        "agent": {
            **config["agent"],
            "variant": runtime["variant"],
            "dqn_edge": agent_config.as_dict(),
        },
    }
    (output / "resolved_config.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["episode", "episode_reward", "epsilon", "steps", "mean_time_loss", "mean_energy_loss",
              "mean_attention_loss", "mean_time_q_action_std", "mean_energy_q_action_std",
              "mean_weighted_q_scale_ratio", "mean_time_weight", "mean_energy_weight", "total_energy",
              "deadline_success_rate", "reliability_success_rate", "avg_workflow_response_time"]
    history = output / "history.csv"
    eval_history = output / "eval_history.csv"
    eval_fields = [
        "episode",
        "eval_seed",
        "eval_episodes",
        "episode_reward",
        "total_energy",
        "deadline_success_rate",
        "reliability_success_rate",
        "avg_workflow_response_time",
        *satisfaction_field_names(),
    ]
    eval_interval = int(config["training"].get("eval_interval", 20))
    eval_episodes = int(config["training"].get("eval_episodes", 10))
    save_every = int(config["training"].get("save_every", 0))
    if args.smoke_test:
        eval_interval = 1
        eval_episodes = 1
    for episode in range(start_episode, episodes):
        workflows = build_workflows(config, resources, seed + episode, smoke_test=args.smoke_test)
        simulation = {**config.get("simulation", {}), "enable_backup_duplication": False, "default_rho": 1.0}
        env = build_env(workflows, resources, simulation, seed + episode)
        row = run_episode(env, scheduler, True)
        row.update(episode=episode, epsilon=agent.epsilon)
        with history.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fields, extrasaction="ignore")
            if stream.tell() == 0: writer.writeheader()
            writer.writerow(row)
        best = max(best, float(row["episode_reward"]))
        checkpoint = agent.checkpoint(episode, best)
        checkpoint.update(
            {
                "variant": runtime["variant"],
                "state_config": runtime["state_config"],
                "objective_config": runtime["objective_config"],
                "reward_source": row.get("reward_source"),
                "experiment_config": resolved,
                "checkpoint_policy": checkpoint_policy,
            }
        )
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
                with eval_history.open("a", newline="", encoding="utf-8") as stream:
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
            if best_metric_rank is None or current_rank > best_metric_rank:
                best_metric_rank = current_rank
                best_metrics = {
                    "episode": int(episode),
                    **checkpoint_metrics,
                    "validation_blocks": eval_rows,
                }
                checkpoint["best_metric_rank"] = list(best_metric_rank)
                checkpoint["best_metrics"] = best_metrics
                checkpoint["checkpoint_feasible"] = bool(best_metric_rank[0] >= 1.0)
                torch.save(checkpoint, output / "best_checkpoint.pt")
            write_checkpoint_selection(
                output / "checkpoint_selection.json",
                checkpoint_policy,
                best_metrics,
                best_metric_rank,
                "best_checkpoint.pt",
            )
        checkpoint["best_episode_reward"] = float(best)
        checkpoint["best_metric_rank"] = (
            None if best_metric_rank is None else list(best_metric_rank)
        )
        checkpoint["best_metrics"] = best_metrics
        checkpoint["checkpoint_feasible"] = bool(
            best_metric_rank is not None and best_metric_rank[0] >= 1.0
        )
        torch.save(checkpoint, output / "latest_checkpoint.pt")
        if save_every > 0 and (episode + 1) % save_every == 0:
            torch.save(checkpoint, output / f"checkpoint_ep{episode + 1:04d}.pt")
        print(json.dumps({
            key: row.get(key)
            for key in fields
            if key != "deadline_success_rate"
        }, ensure_ascii=False))
    print(f"output_dir={output}")


def main():
    args = parse_args(); run_training(load_json(args.config), args)


if __name__ == "__main__":
    main()
