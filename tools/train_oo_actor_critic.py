from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.oo_actor_critic import (
    OOActorCriticAgent,
    evaluate_upper_reward_candidate,
)
from src.D3QN.reward import HierarchicalObjective
from src.D3QN.state import HierarchicalStateEncoder
from src.device_utils import describe_device
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from src.simulator.metrics import compute_summary
from tools.hierarchical_common import (
    _assert_metrics_finite,
    _patch_terminal,
    append_history_row,
    average_metrics,
    build_env,
    build_episode_metrics_from_buffers,
    build_resources,
    build_workflows,
    completed_workflows,
    configure_agent_state,
    initialize_history_files,
    load_json,
    metric_rank,
    print_metrics_block,
    validate_training_config,
    write_history_files,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the adapted ordinal-optimization actor-critic baseline."
    )
    parser.add_argument("--config", default="configs/oo_actor_critic.json")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _padded_primary_mask(context, num_servers):
    raw = np.asarray(context["primary_server_mask"], dtype=bool)
    if raw.shape != (int(num_servers),):
        raise ValueError(
            f"OO context server mask shape mismatch: {raw.shape} != {(int(num_servers),)}."
        )
    # Current environment reserves False only for structurally absent servers.
    # No deadline or reliability feasibility is used as an action shield.
    return raw.copy()


def _high_decision_payload(observation, decision, context):
    server_index = int(decision["server_index"])
    server = observation.servers[server_index]
    task = observation.ready_tasks[0]
    primary_mask = np.zeros(observation.server_mask.shape, dtype=bool)
    primary_mask[: len(context["primary_server_mask"])] = np.asarray(
        context["primary_server_mask"], dtype=bool
    )
    backup_pair_mask = np.zeros(
        (primary_mask.shape[0], primary_mask.shape[0]), dtype=bool
    )
    return {
        "observation": observation,
        "state": observation.vector,
        "task_index": 0,
        "primary_server_index": server_index,
        "backup_server_index": None,
        "server_index": server_index,
        "workflow_id": str(task["workflow_id"]),
        "task_id": int(task["task_id"]),
        "primary_server_id": int(server["id"]),
        "backup_server_id": None,
        "server_id": int(server["id"]),
        "primary_server_mask": primary_mask,
        "backup_pair_mask": backup_pair_mask,
        "effective_backup_mask": np.zeros_like(primary_mask),
        "log_prob": 0.0,
        "value": float(decision["q_value"]),
        "deadline_value": 0.0,
        "reliability_value": 0.0,
        "entropy": 0.0,
        "action_mask_mode": "structural_only",
        "selector_mode": "oo_actor_critic",
    }


def collect_oo_episode(
    env,
    encoder,
    agent,
    objective,
    *,
    deterministic,
    deadline_cost_reward_weight,
):
    if bool(env.enable_backup_duplication):
        raise ValueError(
            "The paper baseline has no primary-backup action; set "
            "simulation.enable_backup_duplication=false."
        )
    state = env.reset()
    objective.reset_episode()
    high_buffer = HighRolloutBuffer()
    low_buffer = LowRolloutBuffer()
    completed_seen = set()
    steps = 0
    max_steps = sum(
        env._workflow_task_count(item["workflow"]) for item in env.workflow_states
    ) + 1000
    primary_counts = {}
    rho_values = []
    rho_safe_values = []
    candidate_counts = []
    top_s_counts = []
    hamming_distances = []
    exact_scores = []
    q_values = []

    while not env.done and steps < max_steps:
        if not state.get("ready_tasks"):
            if not env._advance_to_next_decision_point():
                env._update_done()
                break
            state = env.get_state()
            for workflow_result in completed_workflows(env, completed_seen):
                _patch_terminal(objective, high_buffer, workflow_result, completed_seen)
            continue
        if "decision_context" not in state:
            raise KeyError("OO decision state must contain decision_context.")

        observation = encoder.encode_high_state(state)
        context = state["decision_context"]
        valid_server_mask = _padded_primary_mask(context, agent.num_servers)
        candidate_cache = {}

        def exact_score(server_index, frequency_index):
            key = (int(server_index), int(frequency_index))
            if key not in candidate_cache:
                server_id = int(observation.servers[key[0]]["id"])
                candidate_cache[key] = evaluate_upper_reward_candidate(
                    env,
                    objective,
                    state,
                    server_id=server_id,
                    rho=agent.rho_levels[key[1]],
                    deadline_cost_reward_weight=deadline_cost_reward_weight,
                )
            return candidate_cache[key]["replay_reward"]

        decision = agent.select_action(
            observation.vector,
            valid_server_mask,
            exact_score,
            deterministic=bool(deterministic),
        )
        high_decision = _high_decision_payload(observation, decision, context)
        selected_key = (
            int(decision["server_index"]),
            int(decision["frequency_index"]),
        )
        if selected_key not in candidate_cache:
            exact_score(*selected_key)
        selected_prediction = candidate_cache[selected_key]
        env_action = {
            "workflow_id": high_decision["workflow_id"],
            "task_id": high_decision["task_id"],
            "server_id": high_decision["primary_server_id"],
            "rho": float(selected_prediction["rho"]),
            "backup": False,
            "backup_server_id": None,
        }
        selected_metrics = context["server_safety_metrics"][
            str(high_decision["primary_server_id"])
        ]
        env_action["primary_energy_ref_rho1"] = float(
            selected_metrics["primary_energy_ref_rho1"]
        )
        next_state, env_reward, done, info = env.step(env_action)
        feedback = objective.compute_step_feedback(
            env,
            state,
            high_decision,
            decision,
            env_action,
            env_reward,
            info,
        )
        actual_replay_reward = float(feedback["high_reward"]) - (
            float(deadline_cost_reward_weight)
            * float(feedback["deadline_cost"])
            / max(float(feedback["deadline_limit"]), 1e-8)
        )
        if not math.isclose(
            actual_replay_reward,
            float(selected_prediction["replay_reward"]),
            rel_tol=1e-6,
            abs_tol=1e-7,
        ):
            raise AssertionError(
                "OO candidate scoring drifted from the executed upper reward: "
                f"predicted={selected_prediction['replay_reward']}, "
                f"actual={actual_replay_reward}."
            )
        feedback.update(
            {
                "oo_frequency_index": int(decision["frequency_index"]),
                "oo_action_bits": np.asarray(decision["action_bits"], dtype=np.float32),
                "oo_candidate_count": int(decision["candidate_count"]),
                "oo_top_s_count": int(decision["top_s_count"]),
                "oo_selected_hamming_distance": int(
                    decision["selected_hamming_distance"]
                ),
                "oo_exact_score": float(decision["exact_score"]),
                "oo_q_value": float(decision["q_value"]),
            }
        )
        high_buffer.add(
            state=high_decision["state"],
            primary_action=high_decision["primary_server_index"],
            backup_action=None,
            old_log_prob=0.0,
            reward_value=high_decision["value"],
            deadline_value=0.0,
            reliability_value=0.0,
            reward=feedback["high_reward"],
            deadline_cost=feedback["deadline_cost"],
            reliability_cost=feedback["reliability_cost"],
            deadline_limit=feedback["deadline_limit"],
            reliability_limit=feedback["reliability_limit"],
            partial_makespan=feedback["current_partial_makespan"],
            partial_log_risk=feedback["current_partial_log_risk"],
            workflow_id=feedback["workflow_id"],
            workflow_done=False,
            episode_done=done,
            next_state=None,
            primary_mask=high_decision["primary_server_mask"],
            backup_pair_mask=high_decision["backup_pair_mask"],
            info={
                **feedback,
                "high_decision": {
                    key: value
                    for key, value in high_decision.items()
                    if key != "observation"
                },
            },
        )

        server_key = str(high_decision["primary_server_id"])
        primary_counts[server_key] = primary_counts.get(server_key, 0) + 1
        rho_values.append(float(env_action["rho"]))
        rho_safe_values.append(float(selected_metrics.get("rho_safe", 0.0)))
        candidate_counts.append(int(decision["candidate_count"]))
        top_s_counts.append(int(decision["top_s_count"]))
        hamming_distances.append(int(decision["selected_hamming_distance"]))
        exact_scores.append(float(decision["exact_score"]))
        q_values.append(float(decision["q_value"]))
        steps += 1

        while not done and not next_state.get("ready_tasks"):
            if not env._advance_to_next_decision_point():
                env._update_done()
                break
            next_state = env.get_state()
            done = env.done
            for workflow_result in completed_workflows(env, completed_seen):
                _patch_terminal(objective, high_buffer, workflow_result, completed_seen)
        for workflow_result in completed_workflows(env, completed_seen):
            _patch_terminal(objective, high_buffer, workflow_result, completed_seen)
        state = next_state

    if steps >= max_steps and not env.done:
        raise RuntimeError(f"OO episode exceeded max steps: {max_steps}.")
    env._update_done()
    while not env.done and env._advance_to_next_decision_point():
        env._update_done()
    for workflow_result in completed_workflows(env, completed_seen):
        _patch_terminal(objective, high_buffer, workflow_result, completed_seen)
    unfinished = [
        row["workflow_id"]
        for row in env._build_workflow_results()
        if row.get("response_time") is None
    ]
    if unfinished:
        raise RuntimeError(f"OO episode ended with unfinished workflows: {unfinished[:10]}.")

    high_buffer.validate()
    low_buffer.validate()
    workflow_results = env._build_workflow_results()
    summary = compute_summary(workflow_results, env.event_log, env.servers)
    diagnostics = {
        "steps": int(steps),
        "primary_selection_counts": primary_counts,
        "backup_selection_counts": {},
        "rho_mean": float(np.mean(rho_values)) if rho_values else 0.0,
        "rho_std": float(np.std(rho_values)) if rho_values else 0.0,
        "rho_safe_mean": float(np.mean(rho_safe_values)) if rho_safe_values else 0.0,
        "rho_safe_at_one_rate": (
            float(np.mean(np.asarray(rho_safe_values) >= 1.0 - 1e-6))
            if rho_safe_values
            else 0.0
        ),
        "forced_max_frequency_rate": 0.0,
        "shield_relaxed_count": 0,
        "backup_shield_relaxed_count": 0,
        "oo_candidate_count_mean": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "oo_top_s_count_mean": float(np.mean(top_s_counts)) if top_s_counts else 0.0,
        "oo_selected_hamming_distance_mean": (
            float(np.mean(hamming_distances)) if hamming_distances else 0.0
        ),
        "oo_exact_score_mean": float(np.mean(exact_scores)) if exact_scores else 0.0,
        "oo_selected_q_mean": float(np.mean(q_values)) if q_values else 0.0,
    }
    metrics = build_episode_metrics_from_buffers(
        high_buffer, low_buffer, summary, env.event_log, diagnostics
    )
    _assert_metrics_finite(metrics)
    return {
        "high_buffer": high_buffer,
        "low_buffer": low_buffer,
        "metrics": metrics,
        "summary": summary,
        "workflow_results": workflow_results,
        "event_log": list(env.event_log),
    }


def evaluate_agent(
    config,
    resources,
    encoder,
    agent,
    objective,
    *,
    seed,
    eval_episodes,
    smoke_test,
    deadline_cost_reward_weight,
):
    rows = []
    for offset in range(int(eval_episodes)):
        episode_seed = int(seed) + offset
        workflows = build_workflows(
            config, resources, episode_seed, smoke_test=smoke_test
        )
        env = build_env(
            workflows,
            resources,
            dict(config.get("simulation", {})),
            episode_seed,
        )
        result = collect_oo_episode(
            env,
            encoder,
            agent,
            objective,
            deterministic=True,
            deadline_cost_reward_weight=deadline_cost_reward_weight,
        )
        rows.append(result["metrics"])
    return average_metrics(rows)


def _print_oo_metrics(label, row):
    print_metrics_block(label, row)
    pairs = (
        ("oo_replay_size", "replay"),
        ("oo_updates", "updates"),
        ("oo_actor_loss", "actor"),
        ("oo_critic_loss", "critic"),
        ("oo_q_mean", "q_mean"),
        ("oo_target_mean", "target"),
        ("oo_td_abs_mean", "td_abs"),
        ("oo_candidate_count_mean", "candidates"),
        ("oo_top_s_count_mean", "top_s"),
        ("oo_selected_hamming_distance_mean", "hamming"),
        ("oo_exact_score_mean", "exact"),
    )
    values = []
    for key, title in pairs:
        if key not in row:
            continue
        value = row[key]
        formatted = f"{value:.6g}" if isinstance(value, float) else str(value)
        values.append(f"{title:<12} {formatted:>12}")
    if values:
        print("  oo_ac         " + "  ".join(values))


def run_training(config, *, smoke_test=False, episodes_override=None, seed_override=None):
    config = copy.deepcopy(config)
    training_cfg = copy.deepcopy(config.get("training", {}))
    seed = int(seed_override if seed_override is not None else training_cfg.get("seed", 42))
    if smoke_test:
        training_cfg.update(
            {
                "training_mode": "joint",
                "update_mode": "episode_end",
                "total_episodes": 3,
                "eval_interval": 1,
                "eval_episodes": 1,
                "save_every": 1,
            }
        )
        config.setdefault("resources", {})["num_edge"] = 2
        config.setdefault("resources", {})["num_cloud"] = 1
    validate_training_config(training_cfg)
    total_episodes = int(
        episodes_override
        if episodes_override is not None
        else training_cfg.get("total_episodes", 1500)
    )
    resources = build_resources(config, seed)
    configured_agent = configure_agent_state(config, resources, smoke_test=smoke_test)
    shared = dict(configured_agent.get("shared", {}))
    oo_cfg = {**shared, **dict(configured_agent.get("oo", {}))}
    if smoke_test:
        oo_cfg["batch_size"] = min(int(oo_cfg.get("batch_size", 32)), 8)
        oo_cfg["replay_size"] = min(int(oo_cfg.get("replay_size", 50000)), 1000)
        oo_cfg["training_per_episode"] = max(
            int(oo_cfg.get("training_per_episode", 10)), 2
        )
    state_cfg = dict(shared.get("state", {}))
    encoder = HierarchicalStateEncoder(state_cfg)
    rho_levels = oo_cfg.get("rho_levels", (0.2, 0.4, 0.6, 0.8, 1.0))
    agent = OOActorCriticAgent(
        state_dim=encoder.high_state_dim,
        num_servers=len(resources.get("servers", [])),
        rho_levels=rho_levels,
        config=oo_cfg,
        seed=seed,
    )
    objective = HierarchicalObjective(config.get("objective", {}))
    deadline_cost_reward_weight = float(
        oo_cfg.get("deadline_cost_reward_weight", 1.0)
    )
    config["agent"] = configured_agent
    config["training"] = training_cfg

    print(
        "[device] "
        f"oo_actor_critic={describe_device(agent.device)} "
        f"joint_action=server_x_{len(agent.rho_levels)}rho "
        f"bits={agent.server_bits}+{agent.frequency_bits} "
        f"candidates={agent.num_candidates} top_s={agent.top_s} "
        "constraints=reward_only action_mask=structural_only simulation=cpu"
    )

    checkpoint_root = Path(
        training_cfg.get("checkpoint_dir", "results/oo_actor_critic")
    )
    if smoke_test:
        checkpoint_root = Path("results/oo_actor_critic_smoke")
    if not checkpoint_root.is_absolute():
        checkpoint_root = PROJECT_ROOT / checkpoint_root
    run_name = training_cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = checkpoint_root / str(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    initialize_history_files(output_dir, "train_history")
    initialize_history_files(output_dir, "eval_history")
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)

    history = []
    eval_history = []
    best_metrics = None
    update_episode_count = 0
    eval_seed = int(training_cfg.get("eval_seed", seed + 100000))
    save_replay = bool(training_cfg.get("save_replay_buffer", False))

    for episode in range(1, total_episodes + 1):
        episode_seed = seed + episode
        workflows = build_workflows(
            config, resources, episode_seed, smoke_test=smoke_test
        )
        env = build_env(
            workflows,
            resources,
            dict(config.get("simulation", {})),
            episode_seed,
        )
        result = collect_oo_episode(
            env,
            encoder,
            agent,
            objective,
            deterministic=False,
            deadline_cost_reward_weight=deadline_cost_reward_weight,
        )
        store_metrics = agent.remember_from_high_buffer(
            result["high_buffer"],
            deadline_cost_reward_weight=deadline_cost_reward_weight,
        )
        update_metrics = agent.update()
        update_episode_count += int(update_metrics.get("updates", 0.0) > 0.0)
        row = {
            "episode": episode,
            "algorithm": "adapted_oo_actor_critic",
            "training_mode": "oo_joint_server_frequency",
            "update_mode": "episode_end_replay",
            "constraint_handling": "reward_only",
            "high_action_mask_mode": "structural_only",
            "high_update_count": update_episode_count,
            "low_update_count": 0,
            "low_update_skipped_due_to_empty_buffer": True,
            **result["metrics"],
            **{f"oo_{key}": value for key, value in store_metrics.items()},
            **{f"oo_{key}": value for key, value in update_metrics.items()},
        }
        history.append(row)
        append_history_row(output_dir, "train_history", row)
        _print_oo_metrics("train", row)

        agent.save_checkpoint(
            output_dir / "latest_checkpoint.pt",
            objective_state=objective.state_dict(),
            episode=episode,
            config=config,
            include_replay=save_replay,
        )
        save_every = int(training_cfg.get("save_every", 20))
        if save_every > 0 and episode % save_every == 0:
            agent.save_checkpoint(
                output_dir / f"checkpoint_ep{episode:06d}.pt",
                objective_state=objective.state_dict(),
                episode=episode,
                config=config,
                include_replay=save_replay,
            )
            write_history_files(output_dir, "train_history", history, rewrite_jsonl=False)

        eval_interval = int(training_cfg.get("eval_interval", 20))
        if eval_interval > 0 and episode % eval_interval == 0:
            eval_metrics = evaluate_agent(
                config,
                resources,
                encoder,
                agent,
                objective,
                seed=eval_seed,
                eval_episodes=int(training_cfg.get("eval_episodes", 10)),
                smoke_test=smoke_test,
                deadline_cost_reward_weight=deadline_cost_reward_weight,
            )
            eval_row = {
                "episode": episode,
                "algorithm": "adapted_oo_actor_critic",
                "training_mode": "oo_joint_server_frequency_eval",
                "constraint_handling": "reward_only",
                "high_action_mask_mode": "structural_only",
                "eval_seed": eval_seed,
                "eval_episodes": int(training_cfg.get("eval_episodes", 10)),
                **eval_metrics,
            }
            eval_history.append(eval_row)
            append_history_row(output_dir, "eval_history", eval_row)
            write_history_files(
                output_dir, "eval_history", eval_history, rewrite_jsonl=False
            )
            _print_oo_metrics("eval", eval_row)
            if best_metrics is None or metric_rank(eval_metrics) > metric_rank(best_metrics):
                best_metrics = eval_metrics
                agent.save_checkpoint(
                    output_dir / "best_oo_actor_critic_checkpoint.pt",
                    objective_state=objective.state_dict(),
                    episode=episode,
                    config=config,
                    include_replay=save_replay,
                )

    agent.save_checkpoint(
        output_dir / "final_checkpoint.pt",
        objective_state=objective.state_dict(),
        episode=total_episodes,
        config=config,
        include_replay=save_replay,
    )
    write_history_files(output_dir, "train_history", history)
    write_history_files(output_dir, "eval_history", eval_history)
    return {
        "output_dir": str(output_dir),
        "history": history,
        "eval_history": eval_history,
        "best_metrics": best_metrics,
    }


def main():
    args = parse_args()
    config = load_json(args.config)
    result = run_training(
        config,
        smoke_test=args.smoke_test,
        episodes_override=args.episodes,
        seed_override=args.seed,
    )
    print(f"output_dir={result['output_dir']}")


if __name__ == "__main__":
    main()
