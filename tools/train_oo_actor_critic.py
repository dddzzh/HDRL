from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CPU_THREADS = os.environ.get("DRLEC_CPU_THREADS", "1")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = _CPU_THREADS
LOCAL_PY310_SITE = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_PY310_SITE.is_dir() and str(LOCAL_PY310_SITE) not in sys.path:
    sys.path.insert(0, str(LOCAL_PY310_SITE))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.baselines.oo_actor_critic import (
    OOActorCriticAgent,
    evaluate_upper_reward_candidate,
)
from src.D3QN.reward import HierarchicalObjective
from src.device_utils import configure_torch_runtime, describe_device
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from src.simulator.metrics import compute_summary
from src.shared.hierarchical_state_reward import (
    FIXED_LAMBDA_UPPER_REWARD_VERSION,
    SHARED_REWARD_VERSION,
    SHARED_UPPER_REWARD_VERSION,
    build_shared_state_encoder,
    compose_fixed_lambda_upper_reward,
    compose_upper_deadline_energy_reward,
    shared_state_version,
)
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


def _patch_oo_terminal(objective, agent, high_buffer, workflow_result, completed_seen):
    """Use the fixed OO Deadline multiplier on the terminal telescope too."""
    terminal = _patch_terminal(
        objective, high_buffer, workflow_result, completed_seen
    )
    if agent.reward_mode not in {
        "fixed_lambda_upper_v1",
        "hierarchical_fixed_lambda_v2",
    }:
        return terminal

    workflow_id = str(terminal["workflow_id"])
    normalized_increment = float(
        terminal.get("normalized_terminal_makespan_increment", 0.0)
    )
    correction = -(
        float(objective.fixed_deadline_lambda) - 1.0
    ) * normalized_increment
    if abs(correction) <= 1e-15:
        return terminal

    for index in range(high_buffer.size - 1, -1, -1):
        if high_buffer.workflow_ids[index] != workflow_id:
            continue
        high_buffer.rewards[index] = float(high_buffer.rewards[index]) + correction
        high_buffer.infos[index]["reward"] = float(high_buffer.rewards[index])
        high_buffer.infos[index]["high_reward"] = float(
            high_buffer.rewards[index]
        )
        high_buffer.infos[index]["oo_terminal_deadline_reward"] = -float(
            objective.fixed_deadline_lambda
        ) * normalized_increment
        high_buffer.infos[index]["oo_terminal_reward_scale_correction"] = float(
            correction
        )
        return terminal
    raise RuntimeError(
        f"No OO transition found for completed workflow {workflow_id}."
    )


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
    random_warmup_flags = []
    energy_savings = []
    energy_zscores = []
    energy_rewards = []
    slack_time_costs = []
    slack_time_zscores = []
    weighted_low_rewards = []
    reliability_violation_deltas = []

    while not env.done and steps < max_steps:
        if not state.get("ready_tasks"):
            if not env._advance_to_next_decision_point():
                env._update_done()
                break
            state = env.get_state()
            for workflow_result in completed_workflows(env, completed_seen):
                _patch_oo_terminal(
                    objective, agent, high_buffer, workflow_result, completed_seen
                )
            continue
        if "decision_context" not in state:
            raise KeyError("OO decision state must contain decision_context.")

        observation = encoder.encode_high_state(state)
        context = state["decision_context"]
        valid_server_mask = _padded_primary_mask(context, agent.num_servers)
        energy_statistics = agent.energy_statistics()
        reward_statistics = agent.reward_statistics()
        active_energy_statistics = (
            reward_statistics["energy"]
            if agent.shared_reward is not None
            else energy_statistics
        )
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
                    energy_reward_weight=agent.energy_reward_weight,
                    energy_statistics=energy_statistics,
                    shared_reward=agent.shared_reward,
                    shared_reward_mode=agent.reward_mode,
                    reward_statistics=reward_statistics,
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
        reliability_violation_delta = float(
            selected_prediction.get("reliability_violation_delta", 0.0)
        )
        if objective.reward_mode == "deadline_incremental_only":
            if agent.reward_mode == "fixed_lambda_upper_v1":
                energy_reference = float(feedback["task_energy_scale"])
                energy_saving = energy_reference - float(
                    feedback["primary_energy_actual"]
                )
                upper_reward = compose_fixed_lambda_upper_reward(
                    normalized_makespan_increment=float(
                        feedback["normalized_makespan_increment"]
                    ),
                    deadline_violation_delta=float(
                        feedback["high_deadline_violation_delta"]
                    ),
                    energy_scale=energy_reference,
                    energy_actual=float(feedback["primary_energy_actual"]),
                    workflow_task_count=float(feedback["workflow_task_count"]),
                    fixed_deadline_lambda=objective.fixed_deadline_lambda,
                    fixed_deadline_miss_penalty=(
                        objective.deadline_incremental_violation_weight
                    ),
                )
                deadline_reward = float(upper_reward["deadline_reward"])
                composed_reward = {
                    "reward": float(upper_reward["reward"]),
                    "energy_reward": float(upper_reward["energy_reward"]),
                    "energy_zscore": 0.0,
                    "slack_time_zscore": 0.0,
                    "slack_time_reward": 0.0,
                    "low_deadline_penalty": 0.0,
                    "weighted_low_reward": float(upper_reward["energy_reward"]),
                }
                slack_time_cost = 0.0
                deadline_violation_delta = 0.0
                oo_reward_mode = agent.reward_mode
            elif agent.reward_mode == "hierarchical_fixed_lambda_v2":
                # Upper placement credit uses rho=1; lower cooperative credit
                # evaluates only the selected server's DVFS change.
                deadline_reward = float(
                    selected_prediction["hierarchical_high_reward"]
                )
            else:
                deadline_reward = float(feedback["high_reward"])
            if (
                agent.shared_reward is not None
                and agent.reward_mode != "shared_upper_same_server_energy_v2"
            ):
                energy_reference = float(selected_prediction["energy_reference"])
                energy_saving = float(selected_prediction["energy_saving"])
                slack_time_cost = float(selected_prediction["slack_time_cost"])
                deadline_violation_delta = float(
                    selected_prediction["deadline_violation_delta"]
                )
                composed_reward = agent.compose_reward(
                    deadline_reward,
                    energy_saving,
                    statistics=reward_statistics,
                    slack_time_cost=slack_time_cost,
                    deadline_violation_delta=deadline_violation_delta,
                    reliability_violation_delta=(
                        reliability_violation_delta
                    ),
                )
                oo_reward_mode = (
                    agent.reward_mode
                    if agent.reward_mode == "hierarchical_fixed_lambda_v2"
                    else "shared_hierarchical_v1"
                )
            elif (
                agent.shared_reward is None
                and agent.reward_mode != "fixed_lambda_upper_v1"
            ):
                energy_reference = float(context["task_energy_scale"])
                energy_saving = energy_reference - float(
                    feedback["primary_energy_actual"]
                )
                slack_time_cost = 0.0
                deadline_violation_delta = 0.0
                composed_reward = agent.compose_reward(
                    deadline_reward,
                    energy_saving,
                    statistics=energy_statistics,
                )
                oo_reward_mode = "deadline_incremental_energy_replay_zscore"
            actual_replay_reward = float(composed_reward["reward"])
            feedback.update(
                {
                    "high_reward": actual_replay_reward,
                    "high_reward_base": actual_replay_reward,
                    "high_makespan_reward": deadline_reward,
                    "high_makespan_reward_component": deadline_reward,
                    "high_time_reward_raw": deadline_reward,
                    "high_energy_reward_raw": energy_saving,
                    "high_energy_reward": composed_reward["energy_reward"],
                    "high_energy_reward_component": composed_reward["energy_reward"],
                    "high_low_reward_bonus": composed_reward.get(
                        "weighted_low_reward", composed_reward["energy_reward"]
                    ),
                    "high_energy_saving_joules": energy_saving,
                    "oo_deadline_reward": deadline_reward,
                    "fixed_deadline_cost": float(-deadline_reward),
                    "oo_energy_reference_joules": energy_reference,
                    "oo_energy_saving_joules": energy_saving,
                    "oo_energy_saving_ratio": float(
                        energy_saving / max(energy_reference, 1e-8)
                    ),
                    "oo_energy_zscore": composed_reward["energy_zscore"],
                    "oo_energy_reward": composed_reward["energy_reward"],
                    "oo_slack_time_cost": float(slack_time_cost),
                    "oo_slack_time_zscore": float(
                        composed_reward.get("slack_time_zscore", 0.0)
                    ),
                    "oo_slack_time_reward": float(
                        composed_reward.get("slack_time_reward", 0.0)
                    ),
                    "oo_deadline_violation_delta": float(
                        deadline_violation_delta
                    ),
                    "oo_reliability_violation_delta": float(
                        reliability_violation_delta
                    ),
                    "oo_low_reliability_penalty": float(
                        composed_reward.get("low_reliability_penalty", 0.0)
                    ),
                    "oo_low_deadline_penalty": float(
                        composed_reward.get("low_deadline_penalty", 0.0)
                    ),
                    "oo_weighted_low_reward": float(
                        composed_reward.get(
                            "weighted_low_reward",
                            composed_reward["energy_reward"],
                        )
                    ),
                    "oo_energy_replay_mean": float(
                        active_energy_statistics["mean"]
                    ),
                    "oo_energy_replay_std": float(
                        active_energy_statistics["std"]
                    ),
                    "oo_slack_time_replay_mean": float(
                        reward_statistics["slack_time"]["mean"]
                    ),
                    "oo_slack_time_replay_std": float(
                        reward_statistics["slack_time"]["std"]
                    ),
                    "oo_reward_mode": oo_reward_mode,
                }
            )
        else:
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
        random_warmup_flags.append(float(bool(decision.get("random_warmup", False))))
        if objective.reward_mode == "deadline_incremental_only":
            energy_savings.append(float(feedback["oo_energy_saving_joules"]))
            energy_zscores.append(float(feedback["oo_energy_zscore"]))
            energy_rewards.append(float(feedback["oo_energy_reward"]))
            slack_time_costs.append(float(feedback["oo_slack_time_cost"]))
            slack_time_zscores.append(float(feedback["oo_slack_time_zscore"]))
            weighted_low_rewards.append(float(feedback["oo_weighted_low_reward"]))
            reliability_violation_deltas.append(float(
                feedback.get("oo_reliability_violation_delta", 0.0)
            ))
        steps += 1

        while not done and not next_state.get("ready_tasks"):
            if not env._advance_to_next_decision_point():
                env._update_done()
                break
            next_state = env.get_state()
            done = env.done
            for workflow_result in completed_workflows(env, completed_seen):
                _patch_oo_terminal(
                    objective, agent, high_buffer, workflow_result, completed_seen
                )
        for workflow_result in completed_workflows(env, completed_seen):
            _patch_oo_terminal(
                objective, agent, high_buffer, workflow_result, completed_seen
            )
        state = next_state

    if steps >= max_steps and not env.done:
        raise RuntimeError(f"OO episode exceeded max steps: {max_steps}.")
    env._update_done()
    while not env.done and env._advance_to_next_decision_point():
        env._update_done()
    for workflow_result in completed_workflows(env, completed_seen):
        _patch_oo_terminal(
            objective, agent, high_buffer, workflow_result, completed_seen
        )
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
    state_version = shared_state_version(encoder)
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
        "oo_random_warmup_rate": (
            float(np.mean(random_warmup_flags)) if random_warmup_flags else 0.0
        ),
        "mean_high_energy_saving_joules": (
            float(np.mean(energy_savings)) if energy_savings else 0.0
        ),
        "total_high_energy_saving_joules": float(np.sum(energy_savings)),
        "oo_energy_zscore_mean": (
            float(np.mean(energy_zscores)) if energy_zscores else 0.0
        ),
        "oo_energy_reward_mean": (
            float(np.mean(energy_rewards)) if energy_rewards else 0.0
        ),
        "oo_slack_time_cost_mean": (
            float(np.mean(slack_time_costs)) if slack_time_costs else 0.0
        ),
        "oo_slack_time_zscore_mean": (
            float(np.mean(slack_time_zscores)) if slack_time_zscores else 0.0
        ),
        "oo_weighted_low_reward_mean": (
            float(np.mean(weighted_low_rewards)) if weighted_low_rewards else 0.0
        ),
        "oo_low_reliability_cost_mean": (
            float(np.mean(reliability_violation_deltas))
            if reliability_violation_deltas
            else 0.0
        ),
        "oo_low_reliability_cost_total": float(
            np.sum(reliability_violation_deltas)
        ),
        "oo_state_version": state_version,
        "oo_reward_version": (
            FIXED_LAMBDA_UPPER_REWARD_VERSION
            if agent.reward_mode == "fixed_lambda_upper_v1"
            else (
                SHARED_UPPER_REWARD_VERSION
                if agent.reward_mode == "shared_upper_same_server_energy_v2"
                else (
                    SHARED_REWARD_VERSION
                    if agent.shared_reward is not None
                    else "legacy_deadline_energy"
                )
            )
        ),
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
        ("oo_random_warmup_rate", "warmup"),
        ("oo_exact_score_mean", "exact"),
        ("oo_energy_zscore_mean", "e_z"),
        ("oo_energy_reward_mean", "e_reward"),
        ("oo_energy_replay_mean", "e_mean_J"),
        ("oo_energy_replay_std", "e_std_J"),
        ("oo_slack_time_cost_mean", "g_mean"),
        ("oo_slack_time_zscore_mean", "g_z"),
        ("oo_slack_time_replay_mean", "g_rep_mean"),
        ("oo_slack_time_replay_std", "g_rep_std"),
        ("oo_weighted_low_reward_mean", "beta_rL"),
        ("oo_low_reliability_cost_mean", "r_cost"),
        ("oo_low_reliability_cost_total", "r_cost_total"),
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
    runtime_threads = configure_torch_runtime(
        config.get("training", {}).get("torch_cpu_threads", 1)
    )
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
    encoder = build_shared_state_encoder(state_cfg)
    state_version = shared_state_version(encoder)
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
        oo_cfg.get(
            "deadline_cost_reward_weight",
            0.0 if objective.reward_mode == "deadline_incremental_only" else 1.0,
        )
    )
    config["agent"] = configured_agent
    config["training"] = training_cfg

    if objective.reward_mode == "legacy_weighted":
        reward_description = (
            "best_non_lagrangian_20260806"
            f"(slack_w={objective.high_makespan_weight:g},"
            f"energy_w={objective.high_energy_weight:g},"
            f"deadline_delta_w={deadline_cost_reward_weight:g},"
            f"reliability_cross_w={objective.constraint_crossing_weight:g},"
            f"reliability_severity_w={objective.constraint_severity_weight:g})"
        )
    elif agent.reward_mode == "fixed_lambda_upper_v1":
        reward_description = (
            f"{FIXED_LAMBDA_UPPER_REWARD_VERSION}"
            f"(lambda={objective.fixed_deadline_lambda:g},"
            "energy_ref=task_scale,divide_by=Nw)"
        )
    elif agent.shared_reward is not None:
        if agent.reward_mode == "shared_upper_same_server_energy_v2":
            reward_description = (
                f"{agent.reward_mode}(deadline_w="
                f"{objective.deadline_incremental_violation_weight:g},"
                f"energy_w={objective.high_energy_weight:g},"
                "energy_ref=same_server_rho1,divide_by=Nw)"
            )
        else:
            reward_description = (
                f"{SHARED_REWARD_VERSION}(beta="
                f"{agent.shared_reward.joint_low_reward_weight:g},"
                f"upper_lambda={objective.fixed_deadline_lambda:g},"
                f"upper_miss_w={objective.deadline_incremental_violation_weight:g},"
                f"deadline_w={agent.shared_reward.low_deadline_violation_weight:g},"
                f"reliability_w={agent.shared_reward.low_reliability_violation_weight:g},"
                f"slack_floor={agent.shared_reward.slack_floor_ratio:g})"
            )
    else:
        reward_description = (
            f"{objective.reward_mode}+energy_zscore"
            f"(w={agent.energy_reward_weight:g})"
        )
    print(
        "[device] "
        f"oo_actor_critic={describe_device(agent.device)} "
        f"joint_action=server_x_{len(agent.rho_levels)}rho "
        f"bits={agent.server_bits}+{agent.frequency_bits} "
        f"candidates={agent.num_candidates} top_s={agent.top_s} "
        f"state={state_version} reward={reward_description} "
        f"torch_threads={runtime_threads['torch_cpu_threads']} "
        f"interop_threads={runtime_threads['torch_interop_threads']} "
        f"task_order={config.get('simulation', {}).get('ready_task_order', 'smallest_workload')} "
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
