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

from src.D3QN.hierarchical_d3qn_agent import HierarchicalD3QNAgent
from src.D3QN.reward import HierarchicalObjective
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from src.sac.reward import build_low_sac_reward_components
from src.schedulers.heft import HEFTScheduler
from src.simulator.metrics import compute_summary
from tools.hierarchical_common import (
    _assert_metrics_finite,
    append_history_row,
    average_metrics,
    build_env,
    build_episode_metrics_from_buffers,
    build_resources,
    build_workflows,
    completed_workflows,
    configure_agent_state,
    load_json,
    metric_rank,
    validate_training_config,
    write_history_files,
    initialize_history_files,
    _patch_terminal,
)
from src.device_utils import describe_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train D3QN/HEFT high-level + low-level DVFS learner.")
    parser.add_argument("--config", default="configs/hierarchical_d3qn_sac.json")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--fixed-low-rho-one", action="store_true", help="Use rho=1 for diagnosis; low PPO is not updated.")
    parser.add_argument("--high-algorithm", choices=["d3qn", "heft"], default=None, help="Upper-level task/server selector.")
    parser.add_argument("--low-algorithm", choices=["ppo", "sac"], default=None, help="Low-level DVFS learner.")
    return parser.parse_args()


def _normalize_high_algorithm(value):
    result = str(value or "d3qn").strip().lower()
    if result not in {"d3qn", "heft"}:
        raise ValueError(f"Unsupported high_algorithm: {value!r}; expected 'd3qn' or 'heft'.")
    return result


def _reordered_state_for_task(env, state, workflow_id, task_id):
    workflow_id = str(workflow_id)
    task_id = int(task_id)
    selected = None
    remaining = []
    for task in state.get("ready_tasks", []):
        if str(task["workflow_id"]) == workflow_id and int(task["task_id"]) == task_id:
            selected = task
        else:
            remaining.append(task)
    if selected is None:
        raise RuntimeError(f"HEFT selected a non-ready task: workflow={workflow_id}, task={task_id}.")
    result = dict(state)
    result["ready_tasks"] = [selected, *remaining]
    result["decision_context"] = env.build_hierarchical_decision_context(workflow_id, task_id)
    return result


def _server_index(observation, server_id):
    server_id = int(server_id)
    for index, server in enumerate(observation.servers):
        if int(server["id"]) == server_id:
            return int(index)
    raise RuntimeError(f"HEFT selected unknown server_id={server_id}.")


def _fixed_low_decision():
    return {
        "state": np.zeros(1, dtype=np.float32),
        "action_rho": 1.0,
        "action_u": 1.0,
        "rho": 1.0,
        "log_prob": 0.0,
        "value": 0.0,
        "deadline_value": 0.0,
        "rho_alpha": 0.0,
        "rho_beta": 0.0,
        "entropy": 0.0,
        "rho_safe": 1.0,
        "low_action_feasible": True,
        "has_action_span": False,
        "actor_update_mask": False,
        "forced_max_frequency": False,
        "fixed_rho_one": True,
    }


def _select_heft_action(env, agent, state, scheduler, *, low_deterministic):
    heft_action = scheduler.select_action(
        state,
        context_builder=env.build_hierarchical_decision_context,
    )
    decision_state = _reordered_state_for_task(
        env, state, heft_action["workflow_id"], heft_action["task_id"]
    )
    observation = agent.high_agent.build_high_state(decision_state)
    context = decision_state["decision_context"]
    primary_index = _server_index(observation, heft_action["server_id"])
    primary_mask, backup_pair_mask = agent.high_agent._context_masks(context)
    backup_pair_mask[:] = False
    selected_metrics = context["server_safety_metrics"][str(heft_action["server_id"])]
    if agent.low_agent is None:
        low_decision = _fixed_low_decision()
    else:
        low_decision = agent.low_agent.select_frequency(
            observation,
            0,
            primary_index,
            deterministic=bool(low_deterministic),
            safety_metrics=selected_metrics,
        )
    high_decision = {
        "observation": observation,
        "state": observation.vector,
        "task_index": 0,
        "primary_server_index": primary_index,
        "backup_server_index": None,
        "server_index": primary_index,
        "workflow_id": str(heft_action["workflow_id"]),
        "task_id": int(heft_action["task_id"]),
        "primary_server_id": int(heft_action["server_id"]),
        "backup_server_id": None,
        "server_id": int(heft_action["server_id"]),
        "primary_server_mask": primary_mask,
        "backup_pair_mask": backup_pair_mask,
        "effective_backup_mask": np.zeros(agent.high_agent.max_servers, dtype=bool),
        "log_prob": 0.0,
        "value": 0.0,
        "deadline_value": 0.0,
        "reliability_value": 0.0,
        "entropy": 0.0,
        "action_mask_mode": "structural_only",
        "shield_relaxed": False,
        "shield_relaxation_reason": None,
        "backup_shield_relaxed": False,
        "backup_selector": "disabled",
        "selector_mode": "heft",
        "heft_algorithm": str(heft_action["algorithm"]),
        "heft_upward_rank": float(heft_action["upward_rank"]),
        "heft_estimated_finish_time": float(heft_action["estimated_finish_time"]),
        "heft_online_queue_aware": bool(heft_action["online_queue_aware"]),
    }
    env_action = {
        "workflow_id": high_decision["workflow_id"],
        "task_id": high_decision["task_id"],
        "server_id": high_decision["primary_server_id"],
        "rho": float(low_decision["rho"]),
        "backup": False,
        "backup_server_id": None,
    }
    return decision_state, high_decision, low_decision, env_action


def collect_d3qn_episode(
    env,
    agent,
    objective,
    high_buffer,
    low_buffer,
    *,
    training,
    high_deterministic,
    low_deterministic,
    fixed_low_rho_one=False,
    high_algorithm="d3qn",
):
    high_algorithm = _normalize_high_algorithm(high_algorithm)
    state = env.reset()
    heft_scheduler = None
    if high_algorithm == "heft":
        heft_scheduler = HEFTScheduler(params={"rho": 1.0}, seed=env.seed)
        heft_scheduler.reset()
    objective.reset_episode()
    high_buffer.clear()
    low_buffer.clear()
    completed_seen = set()
    steps = 0
    primary_counts = {}
    backup_counts = {}
    rho_values = []
    rho_safe_values = []
    low_candidate_count = 0
    low_accepted_count = 0
    forced_max_count = 0
    rho_safe_at_one_count = 0
    shield_relaxed_count = 0
    backup_shield_relaxed_count = 0
    heft_online_queue_aware_count = 0
    pending_low_sac = None
    low_sac_stored_count = 0
    max_steps = sum(env._workflow_task_count(item["workflow"]) for item in env.workflow_states) + 1000

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
            raise KeyError("decision state must contain decision_context.")
        decision_state = state
        if high_algorithm == "heft":
            decision_state, high_decision, low_decision, env_action = _select_heft_action(
                env,
                agent,
                state,
                heft_scheduler,
                low_deterministic=low_deterministic,
            )
        else:
            high_decision, low_decision, env_action = agent.select_action(
                state,
                deterministic=high_deterministic,
                low_deterministic=low_deterministic,
                high_selector_mode="policy",
            )
        if fixed_low_rho_one:
            low_decision = dict(low_decision)
            low_decision["rho"] = 1.0
            low_decision["actor_update_mask"] = False
            low_decision["forced_max_frequency"] = False
            low_decision["fixed_rho_one"] = True
            env_action = dict(env_action)
            env_action["rho"] = 1.0
        selected_metrics = decision_state["decision_context"]["server_safety_metrics"][str(high_decision["primary_server_id"])]
        env_action["primary_energy_ref_rho1"] = float(selected_metrics["primary_energy_ref_rho1"])
        low_sac_reference = None
        if _is_low_sac_agent(agent):
            low_sac_reference = {
                **selected_metrics,
                **env.estimate_action_metrics_ref_rho1(env_action),
            }
        next_state, env_reward, done, info = env.step(env_action)
        feedback = objective.compute_step_feedback(
            env, decision_state, high_decision, low_decision, env_action, env_reward, info
        )
        low_sac_reward_components = None
        if low_sac_reference is not None:
            low_sac_reward_components = build_low_sac_reward_components(
                env, env_action, info, low_sac_reference
            )
            feedback.update({f"low_sac_{key}": value for key, value in low_sac_reward_components.items()})
        if training and _is_low_sac_agent(agent):
            if bool(low_decision.get("actor_update_mask", False)):
                current_low_sac = {
                    "state": np.asarray(low_decision["state"], dtype=np.float32),
                    "action_rho": float(low_decision["action_rho"]),
                    "reward_components": low_sac_reward_components,
                }
                if pending_low_sac is not None:
                    agent.low_agent.remember_transition(
                        pending_low_sac["state"],
                        pending_low_sac["action_rho"],
                        pending_low_sac["reward_components"],
                        current_low_sac["state"],
                        False,
                    )
                    low_sac_stored_count += 1
                pending_low_sac = current_low_sac
            elif pending_low_sac is not None:
                agent.low_agent.remember_transition(
                    pending_low_sac["state"],
                    pending_low_sac["action_rho"],
                    pending_low_sac["reward_components"],
                    np.zeros_like(pending_low_sac["state"], dtype=np.float32),
                    True,
                )
                low_sac_stored_count += 1
                pending_low_sac = None
        high_buffer.add(
            state=high_decision["state"],
            primary_action=high_decision["primary_server_index"],
            backup_action=high_decision["backup_server_index"],
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
            info={**feedback, **{"high_decision": {k: v for k, v in high_decision.items() if k != "observation"}}},
        )
        low_candidate_count += 1
        if bool(low_decision.get("forced_max_frequency", False)):
            forced_max_count += 1
        if float(low_decision.get("rho_safe", 0.0)) >= 1.0 - 1e-6:
            rho_safe_at_one_count += 1
        if (not fixed_low_rho_one) and bool(low_decision.get("actor_update_mask", False)):
            low_accepted_count += 1
            low_buffer.add(
                state=low_decision["state"],
                action_u=low_decision["action_rho"],
                old_log_prob=low_decision["log_prob"],
                value=low_decision["value"],
                reward=feedback["low_reward"],
                server_id=high_decision["primary_server_id"],
                deadline_value=low_decision["deadline_value"],
                deadline_cost=feedback["deadline_cost"],
                deadline_limit=feedback["deadline_limit"],
                info={**feedback, **low_decision},
            )
        primary_counts[str(high_decision["primary_server_id"])] = primary_counts.get(str(high_decision["primary_server_id"]), 0) + 1
        if high_decision["backup_server_id"] is not None:
            backup_key = str(high_decision["backup_server_id"])
            backup_counts[backup_key] = backup_counts.get(backup_key, 0) + 1
        shield_relaxed_count += int(bool(high_decision.get("shield_relaxed", False)))
        backup_shield_relaxed_count += int(bool(high_decision.get("backup_shield_relaxed", False)))
        heft_online_queue_aware_count += int(
            bool(high_decision.get("heft_online_queue_aware", False))
        )
        rho_values.append(float(low_decision["rho"]))
        rho_safe_values.append(float(low_decision["rho_safe"]))
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
        raise RuntimeError(f"Episode exceeded max steps: {max_steps}")
    env._update_done()
    while not env.done and env._advance_to_next_decision_point():
        env._update_done()
    for workflow_result in completed_workflows(env, completed_seen):
        _patch_terminal(objective, high_buffer, workflow_result, completed_seen)
    if training and pending_low_sac is not None:
        agent.low_agent.remember_transition(
            pending_low_sac["state"],
            pending_low_sac["action_rho"],
            pending_low_sac["reward_components"],
            np.zeros_like(pending_low_sac["state"], dtype=np.float32),
            True,
        )
        low_sac_stored_count += 1
    unfinished = [row["workflow_id"] for row in env._build_workflow_results() if row.get("response_time") is None]
    if unfinished:
        raise RuntimeError(f"Episode ended with unfinished workflows: {unfinished[:10]}")
    high_buffer.validate()
    low_buffer.validate()
    workflow_results = env._build_workflow_results()
    summary = compute_summary(workflow_results, env.event_log, env.servers)
    diagnostics = {
        "steps": steps,
        "low_candidate_count": low_candidate_count,
        "low_buffer_acceptance_rate": 0.0 if low_candidate_count == 0 else low_accepted_count / low_candidate_count,
        "rho_safe_at_one_rate": 0.0 if low_candidate_count == 0 else rho_safe_at_one_count / low_candidate_count,
        "forced_max_frequency_rate": 0.0 if low_candidate_count == 0 else forced_max_count / low_candidate_count,
        "primary_selection_counts": primary_counts,
        "backup_selection_counts": backup_counts,
        "rho_mean": float(np.mean(rho_values)) if rho_values else 0.0,
        "rho_std": float(np.std(rho_values)) if rho_values else 0.0,
        "rho_safe_mean": float(np.mean(rho_safe_values)) if rho_safe_values else 0.0,
        "shield_relaxed_count": shield_relaxed_count,
        "backup_shield_relaxed_count": backup_shield_relaxed_count,
        "heft_online_queue_aware_rate": (
            0.0 if high_algorithm != "heft" or steps == 0
            else heft_online_queue_aware_count / steps
        ),
        "low_sac_stored_count": low_sac_stored_count,
    }
    metrics = build_episode_metrics_from_buffers(high_buffer, low_buffer, summary, env.event_log, diagnostics)
    _assert_metrics_finite(metrics)
    return {
        "high_buffer": high_buffer,
        "low_buffer": low_buffer,
        "metrics": metrics,
        "summary": summary,
        "workflow_results": workflow_results,
        "event_log": list(env.event_log),
    }


def _is_low_sac_agent(agent):
    return bool(
        getattr(agent, "low_algorithm", None) == "sac"
        and getattr(agent, "low_agent", None) is not None
        and hasattr(agent.low_agent, "remember_transition")
    )


def evaluate_agent(
    config,
    resources,
    agent,
    objective,
    seed,
    eval_episodes,
    smoke_test=False,
    fixed_low_rho_one=False,
    high_algorithm="d3qn",
):
    high_algorithm = _normalize_high_algorithm(high_algorithm)
    rows = []
    old_epsilon = None
    if high_algorithm == "d3qn":
        old_epsilon = agent.high_agent.epsilon
        agent.high_agent.epsilon = 0.0
    try:
        for offset in range(int(eval_episodes)):
            workflows = build_workflows(config, resources, seed + offset, smoke_test=smoke_test)
            env = build_env(workflows, resources, dict(config.get("simulation", {})), seed + offset)
            high_buffer = HighRolloutBuffer()
            low_buffer = LowRolloutBuffer()
            result = collect_d3qn_episode(
                env,
                agent,
                objective,
                high_buffer,
                low_buffer,
                training=False,
                high_deterministic=True,
                low_deterministic=True,
                fixed_low_rho_one=fixed_low_rho_one,
                high_algorithm=high_algorithm,
            )
            rows.append(result["metrics"])
    finally:
        if old_epsilon is not None:
            agent.high_agent.epsilon = old_epsilon
    return average_metrics(rows)


def configure_d3qn_agent_state(config, resources, smoke_test=False):
    result = configure_agent_state(config, resources, smoke_test=smoke_test)
    result["high_algorithm"] = _normalize_high_algorithm(result.get("high_algorithm", "d3qn"))
    low = result.setdefault("low", {})
    result["low_algorithm"] = str(
        config.get("low_algorithm", result.get("low_algorithm", low.get("algorithm", "ppo")))
    ).lower()
    low["algorithm"] = result["low_algorithm"]
    high = result.setdefault("high", {})
    high.setdefault("gamma", high.get("reward_gamma", 0.99))
    high.setdefault("tau", 0.005)
    high.setdefault("epsilon", 1.0)
    high.setdefault("epsilon_min", 0.05)
    high.setdefault("epsilon_decay", 2e-4)
    high.setdefault("replay_size", 100000)
    high.setdefault("training_per_episode", 10)
    if smoke_test:
        high["batch_size"] = min(int(high.get("batch_size", 256)), 8)
        high["replay_size"] = min(int(high.get("replay_size", 100000)), 1000)
        high["training_per_episode"] = max(int(high.get("training_per_episode", 10)), 2)
        if result["low_algorithm"] == "sac":
            low = result.setdefault("low", {})
            low["batch_size"] = min(int(low.get("batch_size", 128)), 8)
            low["replay_size"] = min(int(low.get("replay_size", 100000)), 1000)
            low["training_per_episode"] = max(int(low.get("training_per_episode", 10)), 2)
    return result


def print_d3qn_metrics_block(label, row):
    from tools.hierarchical_common import print_metrics_block

    print_metrics_block(label, row)
    pairs = [
        ("high_epsilon", "eps"),
        ("high_replay_size", "replay"),
        ("high_updates", "updates"),
        ("high_loss", "loss"),
        ("high_q_mean", "q_mean"),
        ("high_target_mean", "target"),
        ("high_td_abs_mean", "td_abs"),
        ("high_grad_norm_before_clip", "grad_pre"),
        ("high_deadline_reward_penalty_total", "d_penalty"),
        ("high_deadline_cost_reward_weight", "d_w"),
        ("heft_online_queue_aware_rate", "online_eft"),
        ("low_replay_size", "low_replay"),
        ("low_updates", "low_updates"),
        ("low_actor_loss", "low_actor"),
        ("low_critic_1_loss", "low_c1"),
        ("low_alpha", "low_alpha"),
        ("low_reward_mean", "sac_r_mean"),
        ("low_reward_std", "sac_r_std"),
    ]
    values = []
    for key, title in pairs:
        if key not in row:
            continue
        value = row.get(key)
        if isinstance(value, float):
            formatted = f"{value:.6g}"
        else:
            formatted = str(value)
        values.append(f"{title:<15} {formatted:>13}")
    if values:
        high_label = str(row.get("high_algorithm", "d3qn"))[:13]
        print(f"  {high_label:<13} " + "  ".join(values))


def run_training(
    config,
    *,
    smoke_test=False,
    episodes_override=None,
    seed_override=None,
    fixed_low_rho_one=False,
    low_algorithm_override=None,
    high_algorithm_override=None,
):
    config = copy.deepcopy(config)
    training_cfg = copy.deepcopy(config.get("training", {}))
    seed = int(seed_override if seed_override is not None else training_cfg.get("seed", 42))
    if smoke_test:
        training_cfg.update({
            "training_mode": "joint",
            "update_mode": "episode_end",
            "total_episodes": 3,
            "eval_interval": 1,
            "eval_episodes": 1,
            "save_every": 1,
        })
        config.setdefault("resources", {})["num_edge"] = 2
        config.setdefault("resources", {})["num_cloud"] = 1
    validate_training_config(training_cfg)
    total_episodes = int(episodes_override if episodes_override is not None else training_cfg.get("total_episodes", 1000))
    resources = build_resources(config, seed)
    if high_algorithm_override is not None:
        config.setdefault("agent", {})["high_algorithm"] = _normalize_high_algorithm(
            high_algorithm_override
        )
    if low_algorithm_override is not None:
        config.setdefault("agent", {})["low_algorithm"] = str(low_algorithm_override).lower()
        config["agent"].setdefault("low", {})["algorithm"] = str(low_algorithm_override).lower()
    agent_config = configure_d3qn_agent_state(config, resources, smoke_test=smoke_test)
    agent_config["use_low_agent"] = not bool(fixed_low_rho_one)
    config = copy.deepcopy(config)
    config["agent"] = agent_config
    config["training"] = training_cfg
    objective = HierarchicalObjective(config.get("objective", {}))
    agent = HierarchicalD3QNAgent(agent_config, seed=seed)
    high_algorithm = _normalize_high_algorithm(agent_config.get("high_algorithm", "d3qn"))
    low_algorithm = getattr(agent, "low_algorithm", "none" if agent.low_agent is None else "ppo")
    high_device = agent.high_agent.agent_R.device
    low_device = agent.low_agent.device if agent.low_agent is not None else None
    print(
        "[device] "
        f"high={'heft(cpu)' if high_algorithm == 'heft' else describe_device(high_device)} "
        f"low={describe_device(low_device) if low_device is not None else 'fixed-rho1'} "
        f"high_algorithm={high_algorithm} low_algorithm={low_algorithm} "
        "constraints=reward_only high_action_mask=structural_only simulation=cpu"
    )
    checkpoint_root = Path(training_cfg.get("checkpoint_dir", "results/hierarchical_d3qn"))
    if str(checkpoint_root).endswith("hierarchical_safe_ppo"):
        checkpoint_root = Path("results/hierarchical_d3qn")
    if smoke_test:
        checkpoint_root = Path(f"results/hierarchical_{high_algorithm}_smoke")
    if not checkpoint_root.is_absolute():
        checkpoint_root = PROJECT_ROOT / checkpoint_root
    run_name = training_cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = checkpoint_root / str(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    initialize_history_files(output_dir, "train_history")
    initialize_history_files(output_dir, "eval_history")
    best_metrics = None
    history = []
    eval_history = []
    high_update_count = 0
    low_update_count = 0
    save_replay_buffer = bool(training_cfg.get("save_replay_buffer", False))
    eval_seed = int(training_cfg.get("eval_seed", seed + 100000))

    for episode in range(1, total_episodes + 1):
        workflows = build_workflows(config, resources, seed + episode, smoke_test=smoke_test)
        env = build_env(workflows, resources, dict(config.get("simulation", {})), seed + episode)
        high_buffer = HighRolloutBuffer()
        low_buffer = LowRolloutBuffer()
        result = collect_d3qn_episode(
            env,
            agent,
            objective,
            high_buffer,
            low_buffer,
            training=True,
            high_deterministic=False,
            low_deterministic=False,
            fixed_low_rho_one=fixed_low_rho_one,
            high_algorithm=high_algorithm,
        )
        if high_algorithm == "d3qn":
            high_metrics = agent.high_agent.update(high_buffer)
        else:
            high_metrics = {
                "updates": 0.0,
                "replay_size": 0.0,
                "stored_transitions": 0.0,
                "epsilon": 0.0,
            }
        high_update_count += int(high_metrics.get("updates", 0.0) > 0.0)
        if (not fixed_low_rho_one) and _is_low_sac_agent(agent):
            low_metrics = agent.low_agent.update(low_buffer)
            low_update_count += int(low_metrics.get("updates", 0.0) > 0.0)
            low_update_skipped = float(low_metrics.get("updates", 0.0)) <= 0.0
        elif (not fixed_low_rho_one) and low_buffer.size > 0:
            low_metrics = agent.low_agent.update(low_buffer)
            low_update_count += 1
            low_update_skipped = False
        else:
            low_metrics = {}
            low_update_skipped = True
        row = {
            "episode": episode,
            "high_algorithm": high_algorithm,
            "constraint_handling": "reward_only",
            "high_action_mask_mode": "structural_only",
            "training_mode": f"{high_algorithm}_high_low_{low_algorithm}" if not fixed_low_rho_one else f"{high_algorithm}_high_fixed_rho1",
            "update_mode": "episode_end_replay" if high_algorithm == "d3qn" else "low_episode_end_replay",
            "high_update_count": high_update_count,
            "low_update_count": low_update_count,
            "low_update_skipped_due_to_empty_buffer": low_update_skipped,
            **result["metrics"],
        }
        row.update({f"high_{key}": value for key, value in high_metrics.items()})
        row.update({f"low_{key}": value for key, value in low_metrics.items()})
        history.append(row)
        append_history_row(output_dir, "train_history", row)
        print_d3qn_metrics_block("train", row)
        latest_path = output_dir / "latest_checkpoint.pt"
        agent.save_checkpoint(
            latest_path,
            objective_state=objective.state_dict(),
            episode=episode,
            training_phase=high_algorithm,
            config=config,
            include_replay=save_replay_buffer,
        )
        if int(training_cfg.get("save_every", 20)) > 0 and episode % int(training_cfg.get("save_every", 20)) == 0:
            agent.save_checkpoint(
                output_dir / f"checkpoint_ep{episode:06d}.pt",
                objective_state=objective.state_dict(),
                episode=episode,
                training_phase=high_algorithm,
                config=config,
                include_replay=save_replay_buffer,
            )
            write_history_files(output_dir, "train_history", history, rewrite_jsonl=False)
        if int(training_cfg.get("eval_interval", 20)) > 0 and episode % int(training_cfg.get("eval_interval", 20)) == 0:
            eval_metrics = evaluate_agent(
                config,
                resources,
                agent,
                objective,
                eval_seed,
                int(training_cfg.get("eval_episodes", 5)),
                smoke_test=smoke_test,
                fixed_low_rho_one=fixed_low_rho_one,
                high_algorithm=high_algorithm,
            )
            eval_row = {
                "episode": episode,
                "high_algorithm": high_algorithm,
                "constraint_handling": "reward_only",
                "high_action_mask_mode": "structural_only",
                "eval_seed": eval_seed,
                "eval_episodes": int(training_cfg.get("eval_episodes", 5)),
                **eval_metrics,
            }
            eval_history.append(eval_row)
            append_history_row(output_dir, "eval_history", eval_row)
            write_history_files(output_dir, "eval_history", eval_history, rewrite_jsonl=False)
            print_d3qn_metrics_block("eval", eval_row)
            if best_metrics is None or metric_rank(eval_metrics) > metric_rank(best_metrics):
                best_metrics = eval_metrics
                agent.save_checkpoint(
                    output_dir / f"best_{high_algorithm}_checkpoint.pt",
                    objective_state=objective.state_dict(),
                    episode=episode,
                    training_phase=high_algorithm,
                    config=config,
                    include_replay=save_replay_buffer,
                )
    agent.save_checkpoint(
        output_dir / "final_checkpoint.pt",
        objective_state=objective.state_dict(),
        episode=total_episodes,
        training_phase=high_algorithm,
        config=config,
        include_replay=save_replay_buffer,
    )
    write_history_files(output_dir, "train_history", history)
    write_history_files(output_dir, "eval_history", eval_history)
    return {"output_dir": str(output_dir), "history": history, "eval_history": eval_history, "best_metrics": best_metrics}


def main():
    args = parse_args()
    config = load_json(args.config)
    result = run_training(
        config,
        smoke_test=args.smoke_test,
        episodes_override=args.episodes,
        seed_override=args.seed,
        fixed_low_rho_one=args.fixed_low_rho_one,
        low_algorithm_override=args.low_algorithm,
        high_algorithm_override=args.high_algorithm,
    )
    print(f"output_dir={result['output_dir']}")


if __name__ == "__main__":
    main()
