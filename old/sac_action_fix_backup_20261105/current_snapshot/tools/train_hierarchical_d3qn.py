from __future__ import annotations

import argparse
import copy
import heapq
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
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.D3QN.hierarchical_d3qn_agent import HierarchicalD3QNAgent
from src.D3QN.queue_weight_calibrator import DeadlineQueueWeightCalibrator
from src.D3QN.reward import HierarchicalObjective
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from src.sac.reward import build_low_sac_reward_components
from src.schedulers.heft import HEFTScheduler
from src.schedulers.fastest_server_scheduler import FastestServerScheduler
from src.schedulers.fcfs_scheduler import FCFSScheduler
from src.schedulers.random_scheduler import RandomScheduler
from src.simulator.metrics import compute_summary
from tools.hierarchical_common import (
    _assert_metrics_finite,
    append_history_row,
    attach_satisfaction_metrics,
    average_metrics,
    build_env,
    build_episode_metrics_from_buffers,
    build_resources,
    build_workflows,
    checkpoint_validation_seeds,
    combine_checkpoint_metrics,
    completed_workflows,
    configure_agent_state,
    format_satisfaction_summary,
    load_json,
    metric_rank,
    validate_training_config,
    write_history_files,
    initialize_history_files,
    _patch_terminal,
)
from src.device_utils import configure_torch_runtime, describe_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train D3QN/HEFT high-level + low-level DVFS learner.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--fixed-low-rho-one", action="store_true", help="Use rho=1 for diagnosis; low PPO is not updated.")
    parser.add_argument(
        "--freeze-high",
        action="store_true",
        help="Freeze a loaded D3QN high policy and train only the low-level agent.",
    )
    parser.add_argument(
        "--freeze-low",
        action="store_true",
        help=(
            "Freeze the low-level policy loaded from a full checkpoint and train "
            "only the D3QN high policy."
        ),
    )
    parser.add_argument(
        "--high-checkpoint",
        default=None,
        help="Primary-only D3QN full/high checkpoint used to initialize the upper policy.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help=(
            "Full D3QN+SAC checkpoint used to initialize joint fine-tuning. "
            "Replay is fresh unless it was explicitly saved in the checkpoint."
        ),
    )
    parser.add_argument(
        "--high-learning-rate",
        type=float,
        default=None,
        help="Optional D3QN learning-rate override for joint fine-tuning.",
    )
    parser.add_argument(
        "--high-updates-per-episode",
        type=int,
        default=None,
        help="Optional D3QN replay updates per episode.",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=None,
        help="Optional D3QN epsilon reset used when joint fine-tuning begins.",
    )
    parser.add_argument(
        "--high-algorithm",
        choices=["d3qn", "heft", "fcfs", "fastest", "random"],
        default=None,
        help="Upper-level task/server selector.",
    )
    parser.add_argument("--low-algorithm", choices=["ppo", "sac"], default=None, help="Low-level DVFS learner.")
    return parser.parse_args()


def _normalize_high_algorithm(value):
    result = str(value or "d3qn").strip().lower()
    allowed = {"d3qn", "heft", "fcfs", "fastest", "random"}
    if result not in allowed:
        raise ValueError(
            f"Unsupported high_algorithm: {value!r}; expected one of {sorted(allowed)}."
        )
    return result


def _build_high_scheduler(high_algorithm, seed):
    scheduler_types = {
        "heft": HEFTScheduler,
        "fcfs": FCFSScheduler,
        "fastest": FastestServerScheduler,
        "random": RandomScheduler,
    }
    high_algorithm = _normalize_high_algorithm(high_algorithm)
    if high_algorithm == "d3qn":
        return None
    scheduler = scheduler_types[high_algorithm](params={"rho": 1.0}, seed=seed)
    scheduler.reset()
    return scheduler


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


def _select_scheduler_action(
    env,
    agent,
    state,
    scheduler,
    *,
    high_algorithm,
    low_deterministic,
):
    high_algorithm = _normalize_high_algorithm(high_algorithm)
    if high_algorithm == "heft":
        scheduler_action = scheduler.select_action(
            state,
            context_builder=env.build_hierarchical_decision_context,
        )
    else:
        scheduler_action = scheduler.select_action(state)
    decision_state = _reordered_state_for_task(
        env,
        state,
        scheduler_action["workflow_id"],
        scheduler_action["task_id"],
    )
    observation = agent.high_agent.build_high_state(decision_state)
    context = decision_state["decision_context"]
    primary_index = _server_index(observation, scheduler_action["server_id"])
    primary_mask, backup_pair_mask = agent.high_agent._context_masks(context)
    backup_pair_mask[:] = False
    selected_metrics = context["server_safety_metrics"][str(scheduler_action["server_id"])]
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
        "workflow_id": str(scheduler_action["workflow_id"]),
        "task_id": int(scheduler_action["task_id"]),
        "primary_server_id": int(scheduler_action["server_id"]),
        "backup_server_id": None,
        "server_id": int(scheduler_action["server_id"]),
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
        "selector_mode": high_algorithm,
        "heuristic_algorithm": str(scheduler_action.get("algorithm", high_algorithm)),
        "heft_algorithm": str(scheduler_action.get("algorithm", "")),
        "heft_upward_rank": float(scheduler_action.get("upward_rank", 0.0)),
        "heft_estimated_finish_time": float(
            scheduler_action.get("estimated_finish_time", 0.0)
        ),
        "heft_online_queue_aware": bool(
            scheduler_action.get("online_queue_aware", False)
        ),
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


def _select_heft_action(env, agent, state, scheduler, *, low_deterministic):
    return _select_scheduler_action(
        env,
        agent,
        state,
        scheduler,
        high_algorithm="heft",
        low_deterministic=low_deterministic,
    )


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
    low_dual_attribution="workflow_counterfactual",
):
    high_algorithm = _normalize_high_algorithm(high_algorithm)
    low_dual_attribution = str(low_dual_attribution).strip().lower()
    if low_dual_attribution not in {"counterfactual", "workflow_counterfactual"}:
        raise ValueError(
            "low_dual_attribution must be 'counterfactual' or "
            f"'workflow_counterfactual'; got {low_dual_attribution!r}."
        )
    if getattr(agent.high_agent, "deadline_lagrangian", None) is not None:
        objective.set_deadline_lambda(agent.high_agent.deadline_lambda)
    state = env.reset()
    high_scheduler = _build_high_scheduler(high_algorithm, env.seed)
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
    rho_floor_binding_count = 0
    rho_safe_at_one_count = 0
    shield_relaxed_count = 0
    backup_shield_relaxed_count = 0
    heft_online_queue_aware_count = 0
    coordination_guided_count = 0
    coordination_changed_count = 0
    coordination_beta_values = []
    pending_low_sac = None
    episode_low_sac_transitions = []
    low_sac_stored_count = 0
    low_deadline_step_cost_by_workflow = {}
    low_deadline_crossed_by_workflow = {}
    low_deadline_constraint_crossing_count = 0
    low_deadline_counterfactual_crossing_count = 0
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
        if high_algorithm != "d3qn":
            decision_state, high_decision, low_decision, env_action = _select_scheduler_action(
                env,
                agent,
                state,
                high_scheduler,
                high_algorithm=high_algorithm,
                low_deterministic=low_deterministic,
            )
        else:
            high_decision, low_decision, env_action = agent.select_action(
                state,
                deterministic=high_deterministic,
                low_deterministic=low_deterministic,
                high_selector_mode="policy",
            )
            if bool(high_decision.get("coordination_guidance_active", False)):
                coordination_guided_count += 1
                coordination_changed_count += int(
                    bool(
                        high_decision.get(
                            "coordination_changed_greedy_action", False
                        )
                    )
                )
                coordination_beta_values.append(
                    float(high_decision.get("coordination_beta", 0.0))
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
                env,
                env_action,
                info,
                low_sac_reference,
                slack_floor_ratio=agent.low_agent.slack_floor_ratio,
                reward_profile=objective.reward_profile,
            )
            feedback.update({f"low_sac_{key}": value for key, value in low_sac_reward_components.items()})
            if (not fixed_low_rho_one) and bool(
                low_decision.get("actor_update_mask", False)
            ):
                low_workflow_id = str(env_action["workflow_id"])
                low_deadline_step_cost_by_workflow[low_workflow_id] = float(
                    low_deadline_step_cost_by_workflow.get(low_workflow_id, 0.0)
                ) + float(low_sac_reward_components["deadline_constraint_cost"])
                low_deadline_crossed_by_workflow[low_workflow_id] = max(
                    float(low_deadline_crossed_by_workflow.get(low_workflow_id, 0.0)),
                    float(low_sac_reward_components["deadline_constraint_crossed"]),
                )
                low_deadline_constraint_crossing_count += int(
                    bool(
                        low_sac_reward_components[
                            "deadline_constraint_crossed"
                        ]
                    )
                )
                low_deadline_counterfactual_crossing_count += int(
                    bool(
                        low_sac_reward_components[
                            "deadline_counterfactual_crossed"
                        ]
                    )
                )
        if training and _is_low_sac_agent(agent):
            if bool(low_decision.get("actor_update_mask", False)):
                current_low_sac = {
                    "state": np.asarray(low_decision["state"], dtype=np.float32),
                    "critic_state": np.asarray(
                        low_decision.get("critic_state", low_decision["state"]),
                        dtype=np.float32,
                    ),
                    "action_rho": float(low_decision["action_rho"]),
                    "reward_components": low_sac_reward_components,
                    "workflow_id": str(env_action["workflow_id"]),
                }
                if pending_low_sac is not None:
                    episode_low_sac_transitions.append(
                        {
                            **pending_low_sac,
                            "next_state": current_low_sac["state"],
                            "next_critic_state": current_low_sac["critic_state"],
                            "done": False,
                        }
                    )
                pending_low_sac = current_low_sac
            elif pending_low_sac is not None:
                episode_low_sac_transitions.append(
                    {
                        **pending_low_sac,
                        "next_state": np.zeros_like(
                            pending_low_sac["state"], dtype=np.float32
                        ),
                        "next_critic_state": np.zeros_like(
                            pending_low_sac["critic_state"], dtype=np.float32
                        ),
                        "done": True,
                    }
                )
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
            info={
                **feedback,
                "coordination_low_rho": float(low_decision["rho"]),
                "high_decision": {
                    k: v
                    for k, v in high_decision.items()
                    if k != "observation"
                },
            },
        )
        low_candidate_count += 1
        if bool(low_decision.get("forced_max_frequency", False)):
            forced_max_count += 1
        # How often the sampled rho fell below the server's action floor and was
        # clipped.  The actor's log-prob refers to the pre-clip value, so a large
        # rate means the policy is credited for a frequency it never executes.
        if float(low_decision.get("rho_floor_binding", 0.0)) > 0.5:
            rho_floor_binding_count += 1
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
                deadline_cost=(
                    float(
                        low_sac_reward_components[
                            "deadline_constraint_cost"
                        ]
                    )
                    if low_sac_reward_components is not None
                    else feedback["deadline_cost"]
                ),
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
        episode_low_sac_transitions.append(
            {
                **pending_low_sac,
                "next_state": np.zeros_like(
                    pending_low_sac["state"], dtype=np.float32
                ),
                "next_critic_state": np.zeros_like(
                    pending_low_sac["critic_state"], dtype=np.float32
                ),
                "done": True,
            }
        )
    unfinished = [row["workflow_id"] for row in env._build_workflow_results() if row.get("response_time") is None]
    if unfinished:
        raise RuntimeError(f"Episode ended with unfinished workflows: {unfinished[:10]}")
    high_buffer.validate()
    low_buffer.validate()
    workflow_results = env._build_workflow_results()
    low_deadline_step_constraint_costs = [
        float(
            low_deadline_step_cost_by_workflow.get(
                str(row["workflow_id"]), 0.0
            )
        )
        for row in workflow_results
    ]
    deadline_usage = np.asarray(
        _completed_workflow_deadline_usage(workflow_results), dtype=np.float64
    )
    if deadline_usage.size != len(workflow_results):
        raise RuntimeError(
            "Every completed workflow must provide response_time and relative_deadline."
        )
    workflow_ids = [str(row["workflow_id"]) for row in workflow_results]
    actual_misses = (deadline_usage > 1.0).astype(np.float64)
    high_sub_deadline_costs = [
        float(
            sum(
                max(
                    0.0,
                    float(
                        high_buffer.infos[index].get(
                            "sub_deadline_constraint_cost", 0.0
                        )
                    ),
                )
                for index in indices
            )
        )
        for _workflow_id, indices in high_buffer.workflow_groups()
    ]
    low_deadline_dual_costs = _low_causal_deadline_misses(
        workflow_ids,
        actual_misses,
        low_deadline_crossed_by_workflow,
    )
    low_reliability_dual_costs = []
    for _workflow_id, indices in high_buffer.workflow_groups():
        low_reliability_dual_costs.append(
            float(sum(
                max(float(high_buffer.infos[index].get(
                    "low_sac_reliability_violation", 0.0
                )), 0.0)
                for index in indices
            ))
    )
    cumulative_low_deadline_details = {}
    low_deadline_replay_costs = None
    if low_dual_attribution == "workflow_counterfactual":
        (
            low_deadline_replay_costs,
            low_deadline_dual_costs,
            cumulative_low_deadline_details,
        ) = _workflow_counterfactual_low_deadline_credit(
            workflow_results, episode_low_sac_transitions
        )
    elif objective.reward_profile == "ctde_placement_cumulative_v2":
        (
            low_deadline_dual_costs,
            cumulative_low_deadline_details,
        ) = _cumulative_low_deadline_credit(
            workflow_results, episode_low_sac_transitions
        )
    if objective.reward_profile in {
        "ep740_lagrangian_v1",
        "sub_deadline_lagrangian_v1",
    } and low_dual_attribution != "workflow_counterfactual":
        # Exact ep740 credit semantics: keep each upper Delta-C/D cost and
        # each lower counterfactual DVFS cost unchanged.  Lambda is still
        # updated at episode end; only the transition credit is restored.
        low_deadline_constraint_costs = list(
            low_deadline_step_constraint_costs
        )
        if objective.reward_profile == "sub_deadline_lagrangian_v1":
            low_deadline_dual_costs = list(
                low_deadline_step_constraint_costs
            )
        low_causal_miss_by_workflow = dict(
            zip(workflow_ids, low_deadline_constraint_costs)
        )
        workflow_team_credit = {}
        if training:
            for transition in episode_low_sac_transitions:
                reward_components = dict(transition["reward_components"])
                reward_components["team_energy_reward"] = 0.0
                agent.low_agent.remember_transition(
                    transition["state"],
                    transition["action_rho"],
                    reward_components,
                    transition["next_state"],
                    transition["done"],
                    critic_state=transition.get("critic_state"),
                    next_critic_state=transition.get("next_critic_state"),
                )
                low_sac_stored_count += 1
    else:
        # A miss belongs to SAC only when rho=1 would have remained feasible
        # and the chosen rho crossed the workflow Deadline boundary. Merely
        # worsening a workflow that already misses at rho=1 remains an upper
        # placement failure.
        low_deadline_constraint_costs = list(
            low_deadline_dual_costs
            if low_deadline_replay_costs is None
            else low_deadline_replay_costs
        )
        low_causal_miss_by_workflow = dict(
            zip(workflow_ids, low_deadline_constraint_costs)
        )
        workflow_team_credit = _apply_cooperative_workflow_credit(
            objective,
            high_buffer,
            workflow_results,
            low_causal_miss_by_workflow,
        )
        if training and episode_low_sac_transitions:
            low_transition_counts = {}
            low_raw_cost_totals = {}
            raw_cost_key = (
                "time_increase_seconds"
                if low_dual_attribution == "workflow_counterfactual"
                else "workflow_makespan_delay_seconds"
                if objective.reward_profile == "ctde_placement_cumulative_v2"
                else "deadline_constraint_cost"
            )
            for transition in episode_low_sac_transitions:
                workflow_id = transition["workflow_id"]
                low_transition_counts[workflow_id] = (
                    low_transition_counts.get(workflow_id, 0) + 1
                )
                low_raw_cost_totals[workflow_id] = float(
                    low_raw_cost_totals.get(workflow_id, 0.0)
                ) + max(
                    float(
                        transition["reward_components"].get(
                            raw_cost_key, 0.0
                        )
                    ),
                    0.0,
                )
            for transition in episode_low_sac_transitions:
                workflow_id = transition["workflow_id"]
                reward_components = dict(transition["reward_components"])
                reward_components["team_energy_reward"] = float(
                    workflow_team_credit.get(workflow_id, {}).get(
                        "team_energy_ratio", 0.0
                    )
                ) / max(low_transition_counts[workflow_id], 1)
                raw_deadline_cost = max(
                    float(reward_components.get(raw_cost_key, 0.0)),
                    0.0,
                )
                raw_deadline_total = float(
                    low_raw_cost_totals.get(workflow_id, 0.0)
                )
                if raw_deadline_total > 1e-8:
                    deadline_share = raw_deadline_cost / raw_deadline_total
                else:
                    deadline_share = 1.0 / max(
                        low_transition_counts[workflow_id], 1
                    )
                reward_components["deadline_constraint_cost"] = float(
                    low_causal_miss_by_workflow.get(workflow_id, 0.0)
                ) * deadline_share
                if (
                    objective.reward_profile == "ctde_placement_cumulative_v2"
                    or low_dual_attribution == "workflow_counterfactual"
                ):
                    responsible_share = float(
                        reward_components["deadline_constraint_cost"]
                    )
                    reward_components["deadline_counterfactual_cost"] = (
                        responsible_share
                    )
                    reward_components["deadline_counterfactual_crossed"] = (
                        float(
                            cumulative_low_deadline_details.get(
                                workflow_id, {}
                            ).get("responsible_crossing", 0.0)
                        )
                        if low_dual_attribution == "workflow_counterfactual"
                        else float(responsible_share > 0.0)
                    )
                agent.low_agent.remember_transition(
                    transition["state"],
                    transition["action_rho"],
                    reward_components,
                    transition["next_state"],
                    transition["done"],
                    critic_state=transition.get("critic_state"),
                    next_critic_state=transition.get("next_critic_state"),
                )
                low_sac_stored_count += 1
    summary = compute_summary(workflow_results, env.event_log, env.servers)
    diagnostics = {
        "steps": steps,
        "low_candidate_count": low_candidate_count,
        "low_buffer_acceptance_rate": 0.0 if low_candidate_count == 0 else low_accepted_count / low_candidate_count,
        "rho_safe_at_one_rate": 0.0 if low_candidate_count == 0 else rho_safe_at_one_count / low_candidate_count,
        "forced_max_frequency_rate": 0.0 if low_candidate_count == 0 else forced_max_count / low_candidate_count,
        "rho_floor_binding_rate": 0.0 if low_candidate_count == 0 else rho_floor_binding_count / low_candidate_count,
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
        "coordination_guidance_rate": (
            0.0 if steps == 0 else coordination_guided_count / steps
        ),
        "coordination_changed_greedy_rate": (
            0.0
            if coordination_guided_count == 0
            else coordination_changed_count / coordination_guided_count
        ),
        "coordination_beta_mean": (
            float(np.mean(coordination_beta_values))
            if coordination_beta_values
            else 0.0
        ),
        "low_deadline_constraint_cost_mean": (
            float(np.mean(low_deadline_constraint_costs))
            if low_deadline_constraint_costs
            else 0.0
        ),
        "low_deadline_constraint_cost_total": float(
            np.sum(low_deadline_constraint_costs)
        ),
        "low_deadline_constraint_crossing_count": int(
            low_deadline_constraint_crossing_count
        ),
        "low_deadline_counterfactual_crossing_count": int(
            low_deadline_counterfactual_crossing_count
        ),
        "low_deadline_step_constraint_cost_mean": (
            float(np.mean(low_deadline_step_constraint_costs))
            if low_deadline_step_constraint_costs
            else 0.0
        ),
        "low_deadline_step_constraint_cost_total": float(
            np.sum(low_deadline_step_constraint_costs)
        ),
        "low_cumulative_responsible_workflow_count": int(
            sum(
                float(row.get("responsible", 0.0)) > 0.5
                for row in cumulative_low_deadline_details.values()
            )
        ),
        "low_cumulative_makespan_delay_seconds": float(
            sum(
                float(row.get("accumulated_delay_seconds", 0.0))
                for row in cumulative_low_deadline_details.values()
            )
        ),
    }
    metrics = build_episode_metrics_from_buffers(high_buffer, low_buffer, summary, env.event_log, diagnostics)
    if getattr(agent.high_agent, "deadline_lagrangian", None) is not None:
        deadline_usage = np.asarray(
            _completed_workflow_deadline_usage(workflow_results),
            dtype=np.float64,
        )
        high_constraint_values = np.asarray(
            high_sub_deadline_costs
            if objective.reward_profile == "sub_deadline_lagrangian_v1"
            else (deadline_usage > 1.0).astype(np.float64),
            dtype=np.float64,
        )
        high_lagrangian = agent.high_agent.deadline_lagrangian
        if (
            getattr(high_lagrangian, "constraint_metric", "")
            in getattr(high_lagrangian, "EPISODE_METRICS", ())
        ):
            # The episode metric's cost is this episode's pass/fail bit, not the
            # pooled miss rate, so the two cannot share one excess expression.
            episode_cost = float(
                1.0 - float(np.mean(high_constraint_values))
                < high_lagrangian.episode_deadline_target - 1e-9
            )
        else:
            episode_cost = float(np.mean(high_constraint_values))
        metrics.update(
            {
                "lambda_deadline": agent.high_agent.deadline_lambda,
                "mean_C_over_D": float(np.mean(deadline_usage)),
                "p95_C_over_D": float(np.percentile(deadline_usage, 95)),
                "max_C_over_D": float(np.max(deadline_usage)),
                "deadline_constraint_metric": (
                    "workflow_miss_rate"
                    if agent.shared_deadline_lagrangian_enabled
                    else getattr(
                        agent.high_agent.deadline_lagrangian,
                        "constraint_metric",
                        "workflow_miss_rate",
                    )
                ),
                "deadline_constraint_budget": float(
                    agent.high_agent.deadline_lagrangian.budget
                ),
                "deadline_excess": float(
                    episode_cost - agent.high_agent.deadline_lagrangian.budget
                ),
                "sub_deadline_constraint_mean": (
                    float(np.mean(high_sub_deadline_costs))
                    if high_sub_deadline_costs
                    else 0.0
                ),
                "deadline_miss_rate": float(
                    np.mean(deadline_usage > 1.0)
                ),
                "completed_workflow_count": int(deadline_usage.size),
            }
        )
    _assert_metrics_finite(metrics)
    return {
        "high_buffer": high_buffer,
        "low_buffer": low_buffer,
        "metrics": metrics,
        "summary": summary,
        "workflow_results": workflow_results,
        "event_log": list(env.event_log),
        "low_deadline_constraint_costs": low_deadline_constraint_costs,
        "low_deadline_dual_costs": low_deadline_dual_costs,
        "low_reliability_dual_costs": low_reliability_dual_costs,
        "high_deadline_dual_costs": (
            high_sub_deadline_costs
            if objective.reward_profile == "sub_deadline_lagrangian_v1"
            else list(actual_misses)
        ),
        "low_deadline_step_constraint_costs": (
            low_deadline_step_constraint_costs
        ),
    }


def _is_low_sac_agent(agent):
    return bool(
        getattr(agent, "low_algorithm", None) == "sac"
        and getattr(agent, "low_agent", None) is not None
        and hasattr(agent.low_agent, "remember_transition")
    )


def _completed_workflow_deadline_usage(workflow_results):
    usage = []
    for row in workflow_results:
        response_time = row.get("response_time")
        deadline = row.get("relative_deadline")
        if response_time is None or deadline is None:
            continue
        deadline = float(deadline)
        if deadline <= 0.0:
            raise ValueError("Completed workflow relative_deadline must be positive.")
        usage.append(float(response_time) / deadline)
    return usage


def _low_causal_deadline_misses(
    workflow_ids,
    actual_misses,
    low_deadline_crossed_by_workflow,
):
    """Return binary workflow misses causally attributable to low-level DVFS.

    The SAC replay reward may use a continuous counterfactual Deadline cost,
    especially for the restored ``ep740_lagrangian_v1`` profile. The dual
    controller constrains workflow miss probability and therefore observes
    one binary indicator per completed workflow.
    """
    misses = np.asarray(actual_misses, dtype=np.float64)
    if misses.ndim != 1 or misses.size != len(workflow_ids):
        raise ValueError(
            "actual_misses must contain one value per completed workflow."
        )
    return [
        float(
            bool(misses[index] > 0.5)
            and float(
                low_deadline_crossed_by_workflow.get(workflow_id, 0.0)
            )
            > 0.5
        )
        for index, workflow_id in enumerate(workflow_ids)
    ]


def _cumulative_low_deadline_credit(workflow_results, low_transitions):
    """Estimate terminal SAC responsibility from accumulated DVFS delay.

    Each rho<1 transition contributes only the delay by which it extends the
    workflow partial makespan relative to rho=1. SAC owns a terminal miss when
    removing the accumulated delay makes that completed workflow feasible.
    The caller distributes the resulting binary workflow cost over the
    contributing transitions in proportion to these delays.
    """
    delay_by_workflow = {}
    for transition in low_transitions:
        workflow_id = str(transition["workflow_id"])
        components = transition.get("reward_components") or {}
        delay = max(
            0.0,
            float(components.get("workflow_makespan_delay_seconds", 0.0)),
        )
        delay_by_workflow[workflow_id] = (
            float(delay_by_workflow.get(workflow_id, 0.0)) + delay
        )

    costs = []
    details = {}
    for row in workflow_results:
        workflow_id = str(row["workflow_id"])
        deadline = max(float(row["relative_deadline"]), 1e-8)
        response_time = float(row["response_time"])
        accumulated_delay = float(delay_by_workflow.get(workflow_id, 0.0))
        counterfactual_response = max(0.0, response_time - accumulated_delay)
        responsible = float(
            response_time > deadline + 1e-8
            and accumulated_delay > 1e-8
            and counterfactual_response <= deadline + 1e-8
        )
        costs.append(responsible)
        details[workflow_id] = {
            "responsible": responsible,
            "accumulated_delay_seconds": accumulated_delay,
            "counterfactual_response_time": counterfactual_response,
            "actual_response_time": response_time,
            "deadline": deadline,
        }
    return costs, details


def _fixed_schedule_counterfactual_model(workflow_results):
    """Build the fixed placement/order DAG used by DVFS counterfactual replay."""
    nodes = {}
    workflow_meta = {}
    server_rows = {}
    for result in workflow_results:
        workflow_id = str(result["workflow_id"])
        task_plans = result.get("task_plans") or []
        if not task_plans or any(
            "parents" not in row or "parent_transfer_times_primary" not in row
            for row in task_plans
        ):
            return None
        workflow_meta[workflow_id] = {
            "arrival_time": float(result["arrival_time"]),
            "download_time": max(
                0.0,
                float(result["finish_time"]) - float(result["compute_finish_time"]),
            ),
        }
        for row in task_plans:
            key = (workflow_id, int(row["task_id"]))
            nodes[key] = row
            server_rows.setdefault(int(row["primary_server"]), []).append(key)

    predecessors = {key: {} for key in nodes}
    for key, row in nodes.items():
        workflow_id, _task_id = key
        transfer_times = {
            int(parent_id): float(value)
            for parent_id, value in dict(
                row.get("parent_transfer_times_primary", {})
            ).items()
        }
        for parent_id in row.get("parents", []):
            parent_key = (workflow_id, int(parent_id))
            if parent_key not in nodes:
                raise RuntimeError(f"Counterfactual replay missing parent {parent_key}.")
            predecessors[key][parent_key] = max(
                predecessors[key].get(parent_key, 0.0),
                transfer_times.get(int(parent_id), 0.0),
            )

    for server_id, keys in server_rows.items():
        ordered = sorted(
            keys,
            key=lambda key: (
                float(nodes[key]["primary_ast"]),
                float(nodes[key]["primary_aft"]),
                key[0],
                key[1],
            ),
        )
        for previous, current in zip(ordered, ordered[1:]):
            predecessors[current][previous] = max(
                predecessors[current].get(previous, 0.0), 0.0
            )

    successors = {key: [] for key in nodes}
    indegree = {key: len(parents) for key, parents in predecessors.items()}
    for key, parents in predecessors.items():
        for parent in parents:
            successors[parent].append(key)
    ready = [key for key, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        key = heapq.heappop(ready)
        order.append(key)
        for child in successors[key]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(nodes):
        raise RuntimeError("Fixed schedule counterfactual graph contains a cycle.")
    return nodes, predecessors, order, workflow_meta


def _replay_fixed_schedule(model, restored_workflow_ids):
    """Replay one schedule with selected workflows restored to rho=1."""
    nodes, predecessors, order, workflow_meta = model
    restored_workflow_ids = {str(value) for value in restored_workflow_ids}
    finish_times = {}
    workflow_compute_finish = {workflow_id: 0.0 for workflow_id in workflow_meta}
    for key in order:
        workflow_id, _task_id = key
        row = nodes[key]
        ready_time = max(
            float(row.get("current_time", workflow_meta[workflow_id]["arrival_time"])),
            workflow_meta[workflow_id]["arrival_time"],
        )
        for parent, transfer_time in predecessors[key].items():
            ready_time = max(
                ready_time, finish_times[parent] + float(transfer_time)
            )
        execution_time = float(row["primary_exec_time"])
        if workflow_id in restored_workflow_ids:
            execution_time *= float(row.get("primary_rho", 1.0))
        finish_times[key] = ready_time + max(execution_time, 0.0)
        workflow_compute_finish[workflow_id] = max(
            workflow_compute_finish[workflow_id], finish_times[key]
        )
    return {
        workflow_id: max(
            0.0,
            compute_finish
            + workflow_meta[workflow_id]["download_time"]
            - workflow_meta[workflow_id]["arrival_time"],
        )
        for workflow_id, compute_finish in workflow_compute_finish.items()
    }


def _legacy_workflow_counterfactual_credit(workflow_results, delay_by_workflow):
    """Fallback for synthetic/legacy rows that do not carry task plans."""
    costs = []
    details = {}
    for row in workflow_results:
        workflow_id = str(row["workflow_id"])
        deadline = max(float(row["relative_deadline"]), 1e-8)
        actual_response = float(row["response_time"])
        accumulated_delay = float(delay_by_workflow.get(workflow_id, 0.0))
        counterfactual_response = max(0.0, actual_response - accumulated_delay)
        actual_violation = max(0.0, (actual_response - deadline) / deadline)
        counterfactual_violation = max(
            0.0, (counterfactual_response - deadline) / deadline
        )
        responsible_cost = float(
            np.clip(actual_violation - counterfactual_violation, 0.0, 1.0)
        )
        costs.append(responsible_cost)
        details[workflow_id] = {
            "responsible": float(responsible_cost > 0.0),
            "responsible_cost": responsible_cost,
            "self_responsible_cost": responsible_cost,
            "external_responsible_cost": 0.0,
            "affected_workflow_count": int(responsible_cost > 0.0),
            "responsible_crossing": float(
                actual_response > deadline + 1e-8
                and counterfactual_response <= deadline + 1e-8
            ),
            "accumulated_delay_seconds": accumulated_delay,
            "counterfactual_response_time": counterfactual_response,
            "actual_response_time": actual_response,
            "actual_violation": actual_violation,
            "counterfactual_violation": counterfactual_violation,
            "deadline": deadline,
        }
    return costs, list(costs), details


def _workflow_counterfactual_low_deadline_credit(
    workflow_results, low_transitions
):
    """Return source replay credit and victim dual costs under fixed decisions.

    Placement and per-server order are fixed.  Restoring a source workflow to
    rho=1 propagates through DAG and server-precedence edges, so the resulting
    marginal includes delays imposed on other workflows.  Victim cost is then
    conserved when allocated back to source workflows, preventing duplicate
    punishment when several slow workflows contributed to one miss.
    """
    delay_by_workflow = {}
    for transition in low_transitions:
        workflow_id = str(transition["workflow_id"])
        components = transition.get("reward_components") or {}
        delay = max(0.0, float(components.get("time_increase_seconds", 0.0)))
        delay_by_workflow[workflow_id] = (
            float(delay_by_workflow.get(workflow_id, 0.0)) + delay
        )

    model = _fixed_schedule_counterfactual_model(workflow_results)
    if model is None:
        return _legacy_workflow_counterfactual_credit(
            workflow_results, delay_by_workflow
        )

    workflow_ids = [str(row["workflow_id"]) for row in workflow_results]
    result_by_id = {str(row["workflow_id"]): row for row in workflow_results}
    actual_violation = {
        workflow_id: max(
            0.0,
            (
                float(result_by_id[workflow_id]["response_time"])
                - max(float(result_by_id[workflow_id]["relative_deadline"]), 1e-8)
            )
            / max(float(result_by_id[workflow_id]["relative_deadline"]), 1e-8),
        )
        for workflow_id in workflow_ids
    }
    all_rho1_response = _replay_fixed_schedule(model, workflow_ids)
    dual_cost_by_victim = {
        workflow_id: float(
            np.clip(
                actual_violation[workflow_id]
                - max(
                    0.0,
                    (
                        all_rho1_response[workflow_id]
                        - max(
                            float(result_by_id[workflow_id]["relative_deadline"]),
                            1e-8,
                        )
                    )
                    / max(
                        float(result_by_id[workflow_id]["relative_deadline"]),
                        1e-8,
                    ),
                ),
                0.0,
                1.0,
            )
        )
        for workflow_id in workflow_ids
    }

    marginal = {source: {} for source in workflow_ids}
    source_responses = {}
    for source in workflow_ids:
        responses = _replay_fixed_schedule(model, [source])
        source_responses[source] = responses
        for victim in workflow_ids:
            deadline = max(
                float(result_by_id[victim]["relative_deadline"]), 1e-8
            )
            counterfactual_violation = max(
                0.0, (responses[victim] - deadline) / deadline
            )
            marginal[source][victim] = max(
                0.0, actual_violation[victim] - counterfactual_violation
            )

    replay_credit = {workflow_id: 0.0 for workflow_id in workflow_ids}
    allocations = {source: {} for source in workflow_ids}
    delay_total = sum(delay_by_workflow.values())
    for victim in workflow_ids:
        victim_cost = dual_cost_by_victim[victim]
        if victim_cost <= 0.0:
            continue
        denominator = sum(marginal[source][victim] for source in workflow_ids)
        for source in workflow_ids:
            if denominator > 1e-12:
                share = marginal[source][victim] / denominator
            elif delay_total > 1e-12:
                # Interaction-only failure: no single restoration repairs the
                # victim, so conserve its cost using each source's slowdown mass.
                share = delay_by_workflow.get(source, 0.0) / delay_total
            else:
                share = 0.0
            allocated = victim_cost * share
            replay_credit[source] += allocated
            allocations[source][victim] = allocated

    details = {}
    for source in workflow_ids:
        row = result_by_id[source]
        deadline = max(float(row["relative_deadline"]), 1e-8)
        source_response = source_responses[source][source]
        responsible_crossing = any(
            float(result_by_id[victim]["response_time"])
            > float(result_by_id[victim]["relative_deadline"]) + 1e-8
            and source_responses[source][victim]
            <= float(result_by_id[victim]["relative_deadline"]) + 1e-8
            for victim in workflow_ids
        )
        details[source] = {
            "responsible": float(replay_credit[source] > 0.0),
            "responsible_cost": float(replay_credit[source]),
            "self_responsible_cost": float(allocations[source].get(source, 0.0)),
            "external_responsible_cost": float(
                sum(
                    value
                    for victim, value in allocations[source].items()
                    if victim != source
                )
            ),
            "affected_workflow_count": int(
                sum(value > 1e-12 for value in allocations[source].values())
            ),
            "responsible_crossing": float(responsible_crossing),
            "accumulated_delay_seconds": float(
                delay_by_workflow.get(source, 0.0)
            ),
            "counterfactual_response_time": float(source_response),
            "actual_response_time": float(row["response_time"]),
            "actual_violation": float(actual_violation[source]),
            "counterfactual_violation": max(
                0.0, (source_response - deadline) / deadline
            ),
            "deadline": deadline,
        }
    return (
        [float(replay_credit[workflow_id]) for workflow_id in workflow_ids],
        [float(dual_cost_by_victim[workflow_id]) for workflow_id in workflow_ids],
        details,
    )


def _apply_cooperative_workflow_credit(
    objective,
    high_buffer,
    workflow_results,
    low_causal_miss_by_workflow,
):
    """Patch workflow-level team energy and hierarchical Deadline credit.

    D3QN always receives the end-to-end workflow miss because server placement
    is responsible for preserving the system Deadline envelope. SAC receives
    only the causal miss introduced by rho < 1. Upper costs are distributed by
    task makespan increments so their workflow sum is the binary system miss.
    """
    result_by_id = {
        str(row["workflow_id"]): row for row in workflow_results
    }
    credit = {}
    for workflow_id, indices in high_buffer.workflow_groups():
        if workflow_id not in result_by_id:
            raise RuntimeError(
                f"Missing terminal result for workflow {workflow_id}."
            )
        workflow_result = result_by_id[workflow_id]
        deadline = max(
            float(workflow_result["relative_deadline"]), 1e-8
        )
        response_time = float(workflow_result["response_time"])
        actual_miss = float(response_time > deadline + 1e-8)
        low_miss = float(
            actual_miss > 0.5
            and float(low_causal_miss_by_workflow.get(workflow_id, 0.0)) > 0.5
        )
        high_miss = actual_miss

        increments = np.asarray(
            [max(0.0, float(high_buffer.deadline_costs[index])) for index in indices],
            dtype=np.float64,
        )
        increment_sum = float(np.sum(increments))
        if increment_sum <= 1e-8:
            weights = np.full(len(indices), 1.0 / max(len(indices), 1))
        else:
            weights = increments / increment_sum

        reference_energy = float(
            sum(
                max(float(high_buffer.infos[index].get("task_energy_scale", 0.0)), 0.0)
                for index in indices
            )
        )
        actual_energy = float(
            sum(
                max(float(high_buffer.infos[index].get("total_energy_actual", 0.0)), 0.0)
                for index in indices
            )
        )
        team_energy_ratio = (
            (reference_energy - actual_energy) / reference_energy
            if reference_energy > 1e-8
            else 0.0
        )
        team_share = team_energy_ratio / max(len(indices), 1)

        for position, index in enumerate(indices):
            info = high_buffer.infos[index]
            local_energy_reward = float(
                info.get(
                    "high_base_energy_reward_local",
                    info.get("high_base_energy_reward", 0.0),
                )
            )
            team_reward = float(objective.team_energy_weight) * team_share
            deadline_cost = high_miss * float(weights[position])
            base_reward = local_energy_reward + team_reward
            lagrangian_penalty = float(objective.deadline_lambda) * deadline_cost
            training_reward = base_reward - lagrangian_penalty
            info.update(
                {
                    "high_base_energy_reward_local": local_energy_reward,
                    "high_team_energy_reward_raw": team_share,
                    "high_team_energy_reward": team_reward,
                    "high_base_energy_reward": base_reward,
                    "high_deadline_constraint_cost": deadline_cost,
                    "terminal_high_deadline_constraint_cost": 0.0,
                    "high_lagrangian_penalty": lagrangian_penalty,
                    "high_training_reward": training_reward,
                    "high_causal_workflow_miss": high_miss,
                    "low_causal_workflow_miss": low_miss,
                    "workflow_team_energy_reference": reference_energy,
                    "workflow_team_energy_actual": actual_energy,
                    "workflow_team_energy_ratio": team_energy_ratio,
                }
            )
            high_buffer.rewards[index] = training_reward
            info["reward"] = training_reward

        credit[workflow_id] = {
            "team_energy_ratio": team_energy_ratio,
            "team_energy_reference": reference_energy,
            "team_energy_actual": actual_energy,
            "actual_miss": actual_miss,
            "high_causal_miss": high_miss,
            "low_causal_miss": low_miss,
        }
    return credit


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
    workers = int(os.environ.get("DRLEC_EVAL_WORKERS", config.get("training", {}).get("eval_workers", 1)))
    if workers < 1:
        raise ValueError("Evaluation workers must be at least one")
    if workers > 1:
        from tools.parallel_evaluation import evaluate_parallel
        return evaluate_parallel(config, resources, agent, objective, seed, eval_episodes,
                                 workers, smoke_test, fixed_low_rho_one, high_algorithm)
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
    return attach_satisfaction_metrics(average_metrics(rows), rows)


def configure_d3qn_agent_state(config, resources, smoke_test=False):
    result = configure_agent_state(config, resources, smoke_test=smoke_test)
    result["high_algorithm"] = _normalize_high_algorithm(result.get("high_algorithm", "d3qn"))
    low = result.setdefault("low", {})
    result["low_algorithm"] = str(
        config.get("low_algorithm", result.get("low_algorithm", low.get("algorithm", "ppo")))
    ).lower()
    low["algorithm"] = result["low_algorithm"]
    high = result.setdefault("high", {})
    high["objective_reward_mode"] = str(
        config.get("objective", {}).get("reward_mode", "legacy_weighted")
    ).strip().lower()
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
        coordination = result.get("coordination_critic")
        if coordination and bool(coordination.get("enabled", False)):
            coordination["batch_size"] = min(
                int(coordination.get("batch_size", 256)), 8
            )
            coordination["replay_size"] = min(
                int(coordination.get("replay_size", 50000)), 1000
            )
            coordination["training_per_episode"] = max(
                int(coordination.get("training_per_episode", 5)), 2
            )
            coordination["warmup_transitions"] = 0
            coordination["beta_anneal_transitions"] = 1
    return result


def print_d3qn_metrics_block(label, row):
    from tools.hierarchical_common import print_metrics_block

    if "phase_name" in row:
        print(
            f"[{label}-stage] "
            f"phase={int(row.get('phase_index', 1))}/{int(row.get('phase_count', 1))} "
            f"name={row.get('phase_name')} "
            f"local_ep={int(row.get('episode', 0))} "
            f"global_ep={int(row.get('global_episode', row.get('episode', 0)))}"
        )
        print(
            "  upper_status  "
            f"policy={row.get('high_algorithm', 'd3qn')} "
            f"state={row.get('high_policy_status', 'unknown')} "
            f"reward={row.get('high_reward_usage', 'unknown')} "
            f"active_ep={int(row.get('high_active_episode', 0))} "
            f"updates_now={int(row.get('high_updates_this_episode', 0))} "
            f"updated_episodes={int(row.get('high_update_count', 0))} "
            f"lr={float(row.get('high_learning_rate', 0.0)):.6g} "
            f"lambda={float(row.get('high_deadline_lambda_current', 0.0)):.6g} "
            f"lambda_update={bool(row.get('high_lambda_updated_this_episode', False))} "
            f"warmup={bool(row.get('deadline_dual_warmup_active', False))}"
        )
        print(
            "  lower_status  "
            f"policy=sac "
            f"state={row.get('low_policy_status', 'unknown')} "
            f"reward={row.get('low_reward_usage', 'unknown')} "
            f"active_ep={int(row.get('low_active_episode', 0))} "
            f"updates_now={int(row.get('low_updates_this_episode', 0))} "
            f"updated_episodes={int(row.get('low_update_count', 0))} "
            f"lr={float(row.get('low_learning_rate', 0.0)):.6g} "
            f"lambda={float(row.get('low_deadline_lambda_current', 0.0)):.6g} "
            f"lambda_update={bool(row.get('low_lambda_updated_this_episode', False))} "
            f"warmup={bool(row.get('low_deadline_dual_warmup_active', False))} "
            f"train_reward={float(row.get('low_reward_mean', 0.0)):.6g}"
        )
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
        ("high_reward_mean", "replay_r_mean"),
        ("high_reward_std", "replay_r_std"),
        ("high_replay_workflow_slack_ratio_mean", "buf_s_mean"),
        ("high_replay_workflow_slack_ratio_std", "buf_s_std"),
        ("high_replay_energy_saving_mean", "buf_e_mean"),
        ("high_replay_energy_saving_std", "buf_e_std"),
        ("high_deadline_crossing_rate", "batch_cross"),
        ("high_deadline_violation_mean", "batch_violation"),
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
        ("low_replay_energy_saving_mean", "buf_e_mean"),
        ("low_replay_energy_saving_std", "buf_e_std"),
        ("low_replay_slack_time_cost_mean", "buf_g_mean"),
        ("low_replay_slack_time_cost_std", "buf_g_std"),
        ("low_replay_team_energy_reward_mean", "buf_team_mean"),
        ("low_replay_team_energy_reward_std", "buf_team_std"),
        ("low_replay_deadline_violation_mean", "buf_dv_mean"),
        ("low_replay_reliability_violation_mean", "buf_rv_mean"),
        ("low_replay_responsible_sample_rate", "resp_sample"),
        ("low_replay_responsibility_severity_mean", "resp_sev"),
        ("low_replay_importance_weight_mean", "is_weight"),
    ]
    values = []
    for key, title in pairs:
        if (
            row.get("high_reward_mode") == "replay_zscore"
            and key in {
                "high_deadline_reward_penalty_total",
                "high_deadline_cost_reward_weight",
            }
        ):
            continue
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
    if "coord_replay_size" in row:
        print(
            "  hcc           "
            f"replay={int(row.get('coord_replay_size', 0)):>7} "
            f"seen={int(row.get('coord_observed_transitions', 0)):>7} "
            f"updates={int(row.get('coord_updates', 0)):>3} "
            f"total_upd={int(row.get('coord_learn_updates_total', 0)):>6} "
            f"beta={float(row.get('coord_beta', 0.0)):.4f} "
            f"active={bool(row.get('coord_guidance_active', False))} "
            f"changed={float(row.get('coordination_changed_greedy_rate', 0.0)):.3f} "
            f"c1={float(row.get('coord_critic_1_loss', 0.0)):.5g} "
            f"c2={float(row.get('coord_critic_2_loss', 0.0)):.5g} "
            f"E_Q={float(row.get('coord_energy_q_mean', 0.0)):.5g} "
            f"D_Q={float(row.get('coord_deadline_q_mean', 0.0)):.5g}"
        )
    if row.get("deadline_lagrangian_mode") == "shared":
        print(
            "[SHARED-D-LAG] "
            f"lambda={float(row.get('shared_deadline_lambda', row.get('lambda_deadline', 0.0))):.6g} "
            f"before={float(row.get('shared_deadline_lambda_before_update', row.get('lambda_deadline', 0.0))):.6g} "
            f"after={float(row.get('shared_deadline_lambda_after_update', row.get('lambda_deadline', 0.0))):.6g} "
            f"rolling_miss={float(row.get('shared_deadline_constraint_mean', row.get('deadline_miss_rate', 0.0))):.6g} "
            f"batch_miss={float(row.get('shared_deadline_constraint_batch_mean', row.get('deadline_miss_rate', 0.0))):.6g} "
            f"budget={float(row.get('shared_deadline_constraint_budget', row.get('deadline_constraint_budget', 0.0))):.6g} "
            f"target_success={float(row.get('shared_deadline_success_target', row.get('deadline_success_target', 0.0))):.4f} "
            f"window_n={int(row.get('shared_deadline_constraint_workflow_count', 0))} "
            f"updated={int(row.get('shared_deadline_dual_updated', 0.0))} "
            f"warmup={int(row.get('shared_deadline_dual_warmup_active', 0.0))}"
        )
    elif (
        row.get("high_reward_mode") == "lagrangian_deadline"
        and row.get("high_reward_usage") == "replay_training"
    ):
        print(
            "[D3QN-H-LAG] "
            f"lambda={float(row.get('lambda_deadline_before_update', row.get('lambda_deadline', 0.0))):.6g} "
            f"lambda_before={float(row.get('lambda_deadline_before_update', row.get('lambda_deadline', 0.0))):.6g} "
            f"lambda_after={float(row.get('lambda_deadline_after_update', row.get('lambda_deadline', 0.0))):.6g} "
            f"metric={row.get('deadline_constraint_metric', 'workflow_miss_rate')} "
            f"miss_rate={float(row.get('deadline_miss_rate', 0.0)):.6g} "
            f"budget={float(row.get('deadline_constraint_budget', 1.0)):.6g} "
            f"dual_cost={float(row.get('deadline_constraint_batch_mean', row.get('sub_deadline_constraint_mean', 0.0))):.6g} "
            # c_e is this episode's pass/fail bit and esp_fail its window rate;
            # under constraint_metric=episode_deadline_failure the latter is
            # what the multiplier above is actually regulating.
            f"c_e={int(row.get('deadline_episode_failed', 0.0))} "
            f"esp_fail={float(row.get('deadline_episode_failure_rate', 0.0)):.4f} "
            f"C/D_mean={float(row.get('mean_C_over_D', 0.0)):.6g} "
            f"C/D_p95={float(row.get('p95_C_over_D', 0.0)):.6g} "
            f"C/D_max={float(row.get('max_C_over_D', 0.0)):.6g} "
            f"excess={float(row.get('deadline_excess', 0.0)):.6g} "
            f"success_rate={float(row.get('deadline_success_rate', 0.0)):.4f} "
            f"base_E={float(row.get('mean_high_base_energy_reward', 0.0)):.6g} "
            f"deadline_cost={float(row.get('mean_high_deadline_constraint_cost', 0.0)):.6g} "
            f"lag_penalty={float(row.get('mean_high_lagrangian_penalty', 0.0)):.6g} "
            f"reward={float(row.get('mean_high_training_reward', 0.0)):.6g} "
            f"completed={int(row.get('completed_workflow_count', 0))} "
            f"updated={int(row.get('deadline_dual_updated', 0.0))} "
            f"warmup={int(row.get('deadline_dual_warmup_active', 0.0))}"
        )
    elif row.get("high_reward_mode") == "lagrangian_deadline":
        print(
            "[HIGH-LAG-INACTIVE] "
            f"reason={row.get('high_reward_usage', 'not_training')} "
            f"lambda={float(row.get('high_deadline_lambda_current', 0.0)):.6g} "
            "dual_update=False metrics_above=diagnostic_only"
        )
    if (
        row.get("deadline_lagrangian_mode") != "shared"
        and "low_deadline_lambda" in row
        and row.get("low_reward_usage") == "replay_training"
    ):
        print(
            "[SAC-L-LAG] "
            f"lambda={float(row.get('low_deadline_lambda', 0.0)):.6g} "
            f"lambda_before={float(row.get('low_deadline_lambda_before_update', row.get('low_deadline_lambda', 0.0))):.6g} "
            f"lambda_after={float(row.get('low_deadline_lambda_after_update', row.get('low_deadline_lambda', 0.0))):.6g} "
            f"metric={row.get('low_deadline_constraint_metric', 'workflow_miss_rate')} "
            f"budget={float(row.get('low_deadline_constraint_budget', 0.0)):.6g} "
            f"J_D_mean={float(row.get('low_deadline_constraint_mean', row.get('low_deadline_constraint_cost_mean', 0.0))):.6g} "
            f"J_D_total={float(row.get('low_deadline_constraint_total', row.get('low_deadline_constraint_cost_total', 0.0))):.6g} "
            f"window_n={int(row.get('low_deadline_constraint_workflow_count', 0))} "
            f"batch_cost={float(row.get('low_deadline_constraint_batch_mean', row.get('low_deadline_constraint_cost_mean', 0.0))):.6g} "
            f"step_cross={int(row.get('low_deadline_constraint_crossing_count', 0))} "
            f"cumulative_resp={int(row.get('low_cumulative_responsible_workflow_count', 0))} "
            f"cumulative_delay_s={float(row.get('low_cumulative_makespan_delay_seconds', 0.0)):.6g} "
            f"base={float(row.get('low_base_reward_mean', 0.0)):.6g} "
            f"lag_penalty={float(row.get('low_lagrangian_penalty_mean', 0.0)):.6g} "
            f"reward={float(row.get('low_training_reward_mean', row.get('low_reward_mean', 0.0))):.6g} "
            f"updated={int(row.get('low_deadline_dual_updated', 0.0))} "
            f"warmup={int(row.get('low_deadline_dual_warmup_active', 0.0))}"
        )
    elif (
        row.get("deadline_lagrangian_mode") != "shared"
        and "low_deadline_lambda" in row
    ):
        print(
            "[LOW-LAG-INACTIVE] "
            f"reason={row.get('low_reward_usage', 'not_training')} "
            f"lambda={float(row.get('low_deadline_lambda_current', row.get('low_deadline_lambda', 0.0))):.6g} "
            "dual_update=False metrics_above=diagnostic_only"
        )


def run_training(
    config,
    *,
    smoke_test=False,
    episodes_override=None,
    seed_override=None,
    fixed_low_rho_one=False,
    low_algorithm_override=None,
    high_algorithm_override=None,
    freeze_high=False,
    freeze_low=False,
    high_checkpoint=None,
    resume_checkpoint=None,
    high_learning_rate_override=None,
    high_updates_override=None,
    epsilon_start_override=None,
    episode_index_offset=0,
):
    config = copy.deepcopy(config)
    training_cfg = copy.deepcopy(config.get("training", {}))
    runtime_threads = configure_torch_runtime(
        training_cfg.get("torch_cpu_threads", 1)
    )
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
    episode_index_offset = int(
        training_cfg.get("episode_index_offset", episode_index_offset)
    )
    if episode_index_offset < 0:
        raise ValueError("episode_index_offset must be non-negative.")
    high_active_episode_offset = int(
        training_cfg.get("high_active_episode_offset", episode_index_offset)
    )
    low_active_episode_offset = int(
        training_cfg.get("low_active_episode_offset", episode_index_offset)
    )
    if high_active_episode_offset < 0 or low_active_episode_offset < 0:
        raise ValueError("active learner episode offsets must be non-negative.")
    resources = build_resources(config, seed)
    if high_algorithm_override is not None:
        config.setdefault("agent", {})["high_algorithm"] = _normalize_high_algorithm(
            high_algorithm_override
        )
    if low_algorithm_override is not None:
        config.setdefault("agent", {})["low_algorithm"] = str(low_algorithm_override).lower()
        config["agent"].setdefault("low", {})["algorithm"] = str(low_algorithm_override).lower()
    if high_learning_rate_override is not None:
        if float(high_learning_rate_override) <= 0.0:
            raise ValueError("high learning rate override must be positive.")
        config.setdefault("agent", {}).setdefault("high", {})[
            "learning_rate"
        ] = float(high_learning_rate_override)
    if high_updates_override is not None:
        if int(high_updates_override) <= 0:
            raise ValueError("high updates per episode must be positive.")
        config.setdefault("agent", {}).setdefault("high", {})[
            "training_per_episode"
        ] = int(high_updates_override)
    agent_config = configure_d3qn_agent_state(config, resources, smoke_test=smoke_test)
    agent_config["use_low_agent"] = not bool(fixed_low_rho_one)
    config = copy.deepcopy(config)
    config["agent"] = agent_config
    config["training"] = training_cfg
    objective = HierarchicalObjective(config.get("objective", {}))
    # beta_q is read by two independent consumers: the objective normalizes the
    # per-step wait, and the Deadline head weights it when it rebuilds its
    # target from the rollout buffer.  objective.deadline_queue_weight is the
    # single source of truth; a conflicting explicit agent value is a config
    # error, not something to silently resolve.
    high_queue_weight = agent_config.setdefault("high", {}).get(
        "deadline_queue_weight"
    )
    if (
        high_queue_weight is not None
        and abs(float(high_queue_weight) - objective.deadline_queue_weight) > 1e-12
    ):
        raise ValueError(
            "agent.high.deadline_queue_weight="
            f"{float(high_queue_weight)} conflicts with "
            f"objective.deadline_queue_weight={objective.deadline_queue_weight}"
        )
    agent_config["high"]["deadline_queue_weight"] = (
        objective.deadline_queue_weight
    )
    high_sub_weight = agent_config["high"].get("deadline_sub_weight")
    if (
        high_sub_weight is not None
        and abs(float(high_sub_weight) - objective.deadline_sub_weight) > 1e-12
    ):
        raise ValueError(
            "agent.high.deadline_sub_weight="
            f"{float(high_sub_weight)} conflicts with "
            f"objective.deadline_sub_weight={objective.deadline_sub_weight}"
        )
    agent_config["high"]["deadline_sub_weight"] = (
        objective.deadline_sub_weight
    )
    # beta_q is solved from a share target rather than configured directly, so
    # that the same config value works on every testbed size.  Until the window
    # closes both consumers keep the configured weight (0.0 by default), which
    # is exactly the pre-calibration behaviour.
    objective_cfg = dict(config.get("objective", {}))
    queue_weight_calibrator = (
        DeadlineQueueWeightCalibrator(
            objective.deadline_queue_share,
            start_episode=int(
                objective_cfg.get("deadline_queue_calibration_start", 150)
            ),
            window_episodes=int(
                objective_cfg.get("deadline_queue_calibration_episodes", 50)
            ),
            sub_deadline_weight=objective.deadline_sub_weight,
        )
        if objective.deadline_queue_share > 0.0
        else None
    )
    if objective.deadline_sub_weight > 0.0:
        sub_deadline_cfg = dict(
            config.get("simulation", {}).get("sub_deadline", {})
        )
        if not bool(sub_deadline_cfg.get("enabled", False)):
            raise ValueError(
                "deadline_sub_weight > 0 requires "
                "simulation.sub_deadline.enabled=true"
            )
    if objective.reward_profile == "sub_deadline_lagrangian_v1":
        sub_deadline_cfg = dict(
            config.get("simulation", {}).get("sub_deadline", {})
        )
        if not bool(sub_deadline_cfg.get("enabled", False)):
            raise ValueError(
                "sub_deadline_lagrangian_v1 requires "
                "simulation.sub_deadline.enabled=true"
            )
        if bool(
            agent_config.get("shared_deadline_lagrangian", {}).get(
                "enabled", False
            )
        ):
            raise ValueError(
                "sub-Deadline costs are layer-causal and cannot use one shared "
                "Deadline multiplier"
            )
    agent = HierarchicalD3QNAgent(agent_config, seed=seed)
    high_algorithm = _normalize_high_algorithm(agent_config.get("high_algorithm", "d3qn"))
    low_algorithm = getattr(agent, "low_algorithm", "none" if agent.low_agent is None else "ppo")
    freeze_high = bool(freeze_high or training_cfg.get("freeze_high", False))
    freeze_low = bool(freeze_low or training_cfg.get("freeze_low", False))
    high_checkpoint = high_checkpoint or training_cfg.get("initial_high_checkpoint")
    resume_checkpoint = resume_checkpoint or training_cfg.get("initial_full_checkpoint")
    if high_checkpoint and resume_checkpoint:
        raise ValueError(
            "Use either a high-only checkpoint or a full resume checkpoint, not both."
        )
    if freeze_high and high_algorithm != "d3qn":
        raise ValueError("freeze_high is supported only when high_algorithm='d3qn'.")
    if freeze_high and freeze_low:
        raise ValueError("freeze_high and freeze_low cannot both be enabled during training.")
    if freeze_high and not (high_checkpoint or resume_checkpoint):
        raise ValueError(
            "freeze_high requires a high-only or full initialization checkpoint."
        )
    if freeze_low and fixed_low_rho_one:
        raise ValueError("freeze_low cannot be combined with fixed_low_rho_one.")
    if freeze_low and agent.low_agent is None:
        raise ValueError("freeze_low requires an enabled low-level agent.")
    if freeze_low and not resume_checkpoint:
        raise ValueError("freeze_low requires a full initialization checkpoint.")
    high_checkpoint_path = None
    high_checkpoint_payload = None
    resume_checkpoint_path = None
    resume_checkpoint_payload = None
    if high_checkpoint:
        high_checkpoint_path = Path(high_checkpoint)
        if not high_checkpoint_path.is_absolute():
            high_checkpoint_path = PROJECT_ROOT / high_checkpoint_path
        if not high_checkpoint_path.is_file():
            raise FileNotFoundError(f"High checkpoint not found: {high_checkpoint_path}")
        high_checkpoint_payload = agent.load_high_checkpoint(
            high_checkpoint_path, freeze=freeze_high
        )
        training_cfg["initial_high_checkpoint"] = str(high_checkpoint_path)
        training_cfg["initial_high_checkpoint_episode"] = int(
            high_checkpoint_payload.get("episode", 0)
        )
        # A high-only initialization may restore optimizer state from the
        # source run.  In joint fine-tuning, honor the new configuration's
        # conservative learning rates instead of silently reusing the source
        # checkpoint rates.
        if not freeze_high:
            configured_high_lr = float(
                agent_config.get("high", {}).get("learning_rate", 3e-4)
            )
            configured_deadline_lr = float(
                agent_config.get("high", {}).get(
                    "deadline_learning_rate", configured_high_lr
                )
            )
            high_learner = agent.high_agent.agent_R
            high_optimizers = [
                (getattr(high_learner, "energy_optimizer", None), configured_high_lr),
                (
                    getattr(high_learner, "deadline_optimizer", None),
                    configured_deadline_lr,
                ),
                (
                    getattr(high_learner, "reliability_optimizer", None),
                    float(
                        agent_config.get("high", {}).get(
                            "reliability_learning_rate", configured_high_lr
                        )
                    ),
                ),
            ]
            if not any(optimizer is not None for optimizer, _lr in high_optimizers):
                high_optimizers = [(high_learner.optimizer, configured_high_lr)]
            seen_optimizers = set()
            for optimizer, learning_rate in high_optimizers:
                if optimizer is None or id(optimizer) in seen_optimizers:
                    continue
                seen_optimizers.add(id(optimizer))
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
    if resume_checkpoint:
        resume_checkpoint_path = Path(resume_checkpoint)
        if not resume_checkpoint_path.is_absolute():
            resume_checkpoint_path = PROJECT_ROOT / resume_checkpoint_path
        if not resume_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Full checkpoint not found: {resume_checkpoint_path}"
            )
        resume_checkpoint_payload = agent.load_checkpoint(
            resume_checkpoint_path
        )
        objective.load_state_dict(
            resume_checkpoint_payload.get("objective_state", {})
        )
        if queue_weight_calibrator is not None:
            queue_weight_calibrator.load_state_dict(
                objective.queue_calibration_state
            )
        # Both consumers of beta_q have to come back in step; the agent reads
        # its own copy and never sees the objective's.
        agent.high_agent.deadline_queue_weight = (
            objective.deadline_queue_weight
        )
        training_cfg["initial_full_checkpoint"] = str(
            resume_checkpoint_path
        )
        training_cfg["initial_full_checkpoint_episode"] = int(
            resume_checkpoint_payload.get("episode", 0)
        )
        configured_high_lr = float(
            agent_config.get("high", {}).get("learning_rate", 3e-4)
        )
        configured_deadline_lr = float(
            agent_config.get("high", {}).get(
                "deadline_learning_rate", configured_high_lr
            )
        )
        high_learner = agent.high_agent.agent_R
        high_optimizers = [
            (getattr(high_learner, "energy_optimizer", None), configured_high_lr),
            (
                getattr(high_learner, "deadline_optimizer", None),
                configured_deadline_lr,
            ),
            (
                getattr(high_learner, "reliability_optimizer", None),
                float(
                    agent_config.get("high", {}).get(
                        "reliability_learning_rate", configured_high_lr
                    )
                ),
            ),
        ]
        if not any(optimizer is not None for optimizer, _lr in high_optimizers):
            high_optimizers = [(high_learner.optimizer, configured_high_lr)]
        seen_optimizers = set()
        for optimizer, learning_rate in high_optimizers:
            if optimizer is None or id(optimizer) in seen_optimizers:
                continue
            seen_optimizers.add(id(optimizer))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        if _is_low_sac_agent(agent):
            configured_low_lr = float(
                agent_config.get("low", {}).get("learning_rate", 3e-4)
            )
            agent.low_agent.learning_rate = configured_low_lr
            low_optimizers = (
                agent.low_agent.actor_optimizer,
                agent.low_agent.critic_1_optimizer,
                agent.low_agent.critic_2_optimizer,
                agent.low_agent.alpha_optimizer,
            )
            for optimizer in low_optimizers:
                if optimizer is None:
                    continue
                for group in optimizer.param_groups:
                    group["lr"] = configured_low_lr
    if freeze_high:
        high_learner = agent.high_agent.agent_R
        if hasattr(high_learner, "reliability_q_eval"):
            high_networks = (
                high_learner.energy_q_eval,
                high_learner.energy_q_target,
                high_learner.deadline_q_eval,
                high_learner.deadline_q_target,
                high_learner.reliability_q_eval,
                high_learner.reliability_q_target,
            )
        elif hasattr(high_learner, "q_eval") and hasattr(high_learner, "q_target"):
            high_networks = (
                high_learner.q_eval,
                high_learner.q_target,
            )
        elif all(
            hasattr(high_learner, name)
            for name in (
                "energy_q_eval",
                "energy_q_target",
                "deadline_q_eval",
                "deadline_q_target",
            )
        ):
            high_networks = (
                high_learner.energy_q_eval,
                high_learner.energy_q_target,
                high_learner.deadline_q_eval,
                high_learner.deadline_q_target,
            )
        else:
            raise TypeError("Unsupported high learner for freeze_high.")
        for network in high_networks:
            network.eval()
            for parameter in network.parameters():
                parameter.requires_grad_(False)
    if freeze_low:
        low_modules = []
        for name in (
            "actor",
            "critic_1",
            "critic_2",
            "target_critic_1",
            "target_critic_2",
            "network",
        ):
            module = getattr(agent.low_agent, name, None)
            if module is not None:
                low_modules.append(module)
        for module in low_modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    if epsilon_start_override is not None:
        epsilon_start_override = float(epsilon_start_override)
        if not 0.0 <= epsilon_start_override <= 1.0:
            raise ValueError("epsilon start override must be in [0, 1].")
        agent.high_agent.epsilon = epsilon_start_override
        training_cfg["joint_epsilon_start"] = epsilon_start_override
    training_cfg["freeze_high"] = bool(freeze_high)
    training_cfg["freeze_low"] = bool(freeze_low)
    training_cfg["episode_index_offset"] = int(episode_index_offset)
    training_cfg["high_active_episode_offset"] = int(
        high_active_episode_offset
    )
    training_cfg["low_active_episode_offset"] = int(
        low_active_episode_offset
    )
    high_device = agent.high_agent.agent_R.device
    low_device = agent.low_agent.device if agent.low_agent is not None else None
    coordination_device = (
        agent.coordination_critic.device
        if agent.coordination_critic is not None
        else None
    )
    high_constraint_mode = (
        "shared_lagrangian_deadline"
        if agent.shared_deadline_lagrangian_enabled
        else str(agent.high_agent.constraint_handling_mode)
        if getattr(agent.high_agent, "constraint_handling_mode", None)
        else "lagrangian_deadline"
        if getattr(agent.high_agent, "deadline_lagrangian", None) is not None
        else "reward_only"
    )
    low_constraint_mode = (
        "shared_lagrangian_deadline"
        if agent.shared_deadline_lagrangian_enabled
        else "lagrangian_deadline"
        if _is_low_sac_agent(agent)
        and getattr(agent.low_agent, "deadline_lagrangian", None) is not None
        else "reward_only"
    )
    print(
        "[device] "
        f"high={f'{high_algorithm}(cpu)' if high_algorithm != 'd3qn' else describe_device(high_device)} "
        f"low={describe_device(low_device) if low_device is not None else 'fixed-rho1'} "
        f"hcc={describe_device(coordination_device) if coordination_device is not None else 'disabled'} "
        f"high_algorithm={high_algorithm} low_algorithm={low_algorithm} "
        f"high_training={('heuristic_fixed' if high_algorithm != 'd3qn' else 'frozen_deterministic' if freeze_high else 'trainable')} "
        f"low_training={'frozen_deterministic' if freeze_low else 'trainable'} "
        f"reward_profile={objective.reward_profile} "
        f"states=high:{agent.high_agent.encoder.high_state_dim},"
        f"low_actor:{getattr(agent.low_agent, 'actor_state_dim', 0)},"
        f"low_critic:{getattr(agent.low_agent, 'critic_state_dim', 0)}"
        f"({getattr(agent.low_agent, 'state_mode', 'fixed')}) "
        f"responsibility_replay={bool(getattr(getattr(agent.low_agent, 'replay', None), 'priority_enabled', False))} "
        f"constraints=high:{high_constraint_mode},low:{low_constraint_mode} "
        f"torch_threads={runtime_threads['torch_cpu_threads']} "
        f"interop_threads={runtime_threads['torch_interop_threads']} "
        f"task_order={config.get('simulation', {}).get('ready_task_order', 'smallest_workload')} "
        "high_action_mask=structural_only simulation=cpu"
    )
    if high_checkpoint_payload is not None:
        print(
            "[init] "
            f"high_checkpoint={high_checkpoint_path} "
            f"episode={int(high_checkpoint_payload.get('episode', 0))} "
            f"freeze_high={freeze_high} low_initialized=fresh"
        )
    if resume_checkpoint_payload is not None:
        print(
            "[init] "
            f"full_checkpoint={resume_checkpoint_path} "
            f"episode={int(resume_checkpoint_payload.get('episode', 0))} "
            "resume_full=True"
        )
    checkpoint_root = Path(training_cfg.get("checkpoint_dir", "results/hierarchical_d3qn"))
    if str(checkpoint_root).endswith("hierarchical_safe_ppo"):
        checkpoint_root = Path("results/hierarchical_d3qn")
    if smoke_test and not training_cfg.get("run_name"):
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
    # Checkpoints are ranked on every configured validation block, matching the
    # three baselines.  ``eval_seed_secondary`` adds a second disjoint 20-seed
    # block whose runs are pooled with the first, so selection sees 40 instances
    # instead of 20; the reported numbers still come from a third block that
    # neither of these touches (tools/accept_v2_checkpoints.py).
    eval_seeds = checkpoint_validation_seeds(training_cfg, seed + 100000)
    eval_seed = eval_seeds[0]
    checkpoint_deadline_target = training_cfg.get("checkpoint_deadline_target")
    if checkpoint_deadline_target is not None:
        checkpoint_deadline_target = float(checkpoint_deadline_target)
        if not 0.0 <= checkpoint_deadline_target <= 1.0:
            raise ValueError("training.checkpoint_deadline_target must be in [0, 1].")
    checkpoint_reliability_target = training_cfg.get(
        "checkpoint_reliability_target"
    )
    if checkpoint_reliability_target is not None:
        checkpoint_reliability_target = float(checkpoint_reliability_target)
        if not 0.0 <= checkpoint_reliability_target <= 1.0:
            raise ValueError(
                "training.checkpoint_reliability_target must be in [0, 1]."
            )
    checkpoint_deadline_esp_threshold = training_cfg.get(
        "checkpoint_deadline_esp_threshold"
    )
    checkpoint_deadline_esp_target = training_cfg.get(
        "checkpoint_deadline_esp_target"
    )
    if (checkpoint_deadline_esp_threshold is None) != (
        checkpoint_deadline_esp_target is None
    ):
        raise ValueError(
            "training.checkpoint_deadline_esp_threshold and "
            "training.checkpoint_deadline_esp_target must be set together."
        )
    if checkpoint_deadline_esp_threshold is not None:
        checkpoint_deadline_esp_threshold = float(
            checkpoint_deadline_esp_threshold
        )
        checkpoint_deadline_esp_target = float(checkpoint_deadline_esp_target)
        if not 0.0 <= checkpoint_deadline_esp_threshold <= 1.0:
            raise ValueError(
                "training.checkpoint_deadline_esp_threshold must be in [0, 1]."
            )
        if not 0.0 <= checkpoint_deadline_esp_target <= 1.0:
            raise ValueError(
                "training.checkpoint_deadline_esp_target must be in [0, 1]."
            )

    for episode in range(1, total_episodes + 1):
        global_episode = episode_index_offset + episode
        episode_seed = seed + global_episode
        workflows = build_workflows(config, resources, episode_seed, smoke_test=smoke_test)
        env = build_env(
            workflows,
            resources,
            dict(config.get("simulation", {})),
            episode_seed,
        )
        high_buffer = HighRolloutBuffer()
        low_buffer = LowRolloutBuffer()
        result = collect_d3qn_episode(
            env,
            agent,
            objective,
            high_buffer,
            low_buffer,
            training=not freeze_low,
            high_deterministic=freeze_high,
            low_deterministic=freeze_low,
            fixed_low_rho_one=fixed_low_rho_one,
            high_algorithm=high_algorithm,
        )
        if queue_weight_calibrator is not None:
            # Solve before this episode's update so the head starts training on
            # the calibrated target immediately, rather than one episode late.
            solved = queue_weight_calibrator.observe(
                high_buffer, global_episode - 1
            )
            objective.queue_calibration_state = (
                queue_weight_calibrator.state_dict()
            )
            if solved is not None:
                objective.set_deadline_queue_weight(solved)
                agent.high_agent.deadline_queue_weight = solved
                realized = queue_weight_calibrator.realized_share()
                print(
                    f"[queue-calibration] episode {global_episode}: "
                    f"beta_q={solved:.4f} for share target "
                    f"{queue_weight_calibrator.share:.3f} "
                    f"(realized {realized:.3f}) from "
                    f"{queue_weight_calibrator.observed_workflows} workflows; "
                    f"I={queue_weight_calibrator.increment_mass:.4f} "
                    f"Q={queue_weight_calibrator.queue_mass:.4f}"
                )
                if solved <= 0.0:
                    print(
                        "[queue-calibration] WARNING: no contention was "
                        "observed, so beta_q resolved to 0 and the queue term "
                        "is inert. Check that the step events carry "
                        "'primary_ast' and 'base_ready_time_primary'."
                    )
                # beta_q is not in the config, so stdout must not be its only
                # record.  Drop a standalone file next to the checkpoints.
                solve_record = queue_weight_calibrator.diagnostics()
                solve_record["solved_at_episode"] = int(global_episode)
                with open(
                    output_dir / "queue_weight_calibration.json",
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(solve_record, handle, indent=2)
        if high_algorithm == "d3qn" and not freeze_high:
            high_metrics = agent.high_agent.update(high_buffer)
        else:
            high_metrics = {
                "updates": 0.0,
                "replay_size": 0.0,
                "stored_transitions": 0.0,
                "epsilon": float(agent.high_agent.epsilon) if high_algorithm == "d3qn" else 0.0,
                "frozen": float(freeze_high),
            }
        high_update_count += int(high_metrics.get("updates", 0.0) > 0.0)
        if (not fixed_low_rho_one) and (not freeze_low) and _is_low_sac_agent(agent):
            low_metrics = agent.low_agent.update(low_buffer)
            low_update_count += int(low_metrics.get("updates", 0.0) > 0.0)
            low_update_skipped = float(low_metrics.get("updates", 0.0)) <= 0.0
        elif (not fixed_low_rho_one) and (not freeze_low) and low_buffer.size > 0:
            low_metrics = agent.low_agent.update(low_buffer)
            low_update_count += 1
            low_update_skipped = False
        else:
            low_metrics = {}
            low_update_skipped = True
        if agent.coordination_critic is not None:
            coordination_metrics = agent.coordination_critic.update(high_buffer)
        else:
            coordination_metrics = {}
        deadline_usage = np.asarray(
            _completed_workflow_deadline_usage(result["workflow_results"]),
            dtype=np.float64,
        )
        deadline_misses = (deadline_usage > 1.0).astype(np.float64)
        low_reliability_dual_metrics = {}
        if agent.shared_deadline_lagrangian_enabled:
            # One system constraint, one observation stream, one dual update.
            # Both replay learners see the updated shared value next episode.
            dual_metrics = agent.update_shared_deadline_lagrangian(
                deadline_misses,
                # Zero-based index: warmup=50 freezes episodes 1 through 50.
                episode=global_episode - 1,
            )
            low_dual_metrics = {}
            objective.set_deadline_lambda(agent.high_agent.deadline_lambda)
        else:
            if (
                (not fixed_low_rho_one)
                and (not freeze_low)
                and _is_low_sac_agent(agent)
                and getattr(agent.low_agent, "deadline_lagrangian", None)
                is not None
            ):
                low_dual_metrics = agent.low_agent.update_deadline_lagrangian(
                    result["low_deadline_dual_costs"],
                    episode=low_active_episode_offset + episode - 1,
                )
            else:
                low_dual_metrics = {}
            if (
                (not fixed_low_rho_one)
                and (not freeze_low)
                and _is_low_sac_agent(agent)
                and getattr(agent.low_agent, "reliability_lagrangian", None)
                is not None
            ):
                low_reliability_dual_metrics = agent.low_agent.update_reliability_lagrangian(
                    result["low_reliability_dual_costs"],
                    episode=low_active_episode_offset + episode - 1,
                )
            else:
                low_reliability_dual_metrics = {}
            if (
                high_algorithm == "d3qn"
                and not freeze_high
                and getattr(agent.high_agent, "deadline_lagrangian", None)
                is not None
            ):
                dual_metrics = agent.high_agent.update_deadline_lagrangian(
                    result["high_deadline_dual_costs"],
                    episode=high_active_episode_offset + episode - 1,
                )
                objective.set_deadline_lambda(
                    agent.high_agent.deadline_lambda
                )
            else:
                dual_metrics = {}
        constraint_handling = (
            "shared_lagrangian_deadline"
            if agent.shared_deadline_lagrangian_enabled
            else str(agent.high_agent.constraint_handling_mode)
            if getattr(agent.high_agent, "constraint_handling_mode", None)
            else "lagrangian_deadline"
            if getattr(agent.high_agent, "deadline_lagrangian", None) is not None
            else "reward_only"
        )
        row = {
            "episode": episode,
            "global_episode": global_episode,
            "episode_seed": episode_seed,
            "phase_name": str(training_cfg.get("phase_name", "single_stage")),
            "phase_index": int(training_cfg.get("phase_index", 1)),
            "phase_count": int(training_cfg.get("phase_count", 1)),
            "high_active_episode": (
                high_active_episode_offset + episode
                if high_algorithm == "d3qn" and not freeze_high
                else high_active_episode_offset
            ),
            "low_active_episode": (
                low_active_episode_offset + episode
                if not freeze_low and not fixed_low_rho_one
                else low_active_episode_offset
            ),
            "high_policy_status": (
                "fixed_heuristic"
                if high_algorithm != "d3qn"
                else "frozen_deterministic"
                if freeze_high
                else "trainable_exploratory"
            ),
            "low_policy_status": (
                "fixed_rho1"
                if fixed_low_rho_one
                else "frozen_deterministic"
                if freeze_low
                else "trainable_stochastic"
            ),
            "high_reward_usage": (
                "diagnostic_only"
                if high_algorithm != "d3qn"
                else "evaluation_only"
                if freeze_high
                else "replay_training"
            ),
            "low_reward_usage": (
                "disabled"
                if fixed_low_rho_one
                else "evaluation_only"
                if freeze_low
                else "replay_training"
            ),
            "high_learning_rate": float(
                0.0
                if high_algorithm != "d3qn"
                else agent.high_agent.agent_R.optimizer.param_groups[0]["lr"]
            ),
            "low_learning_rate": (
                float(agent.low_agent.actor_optimizer.param_groups[0]["lr"])
                if _is_low_sac_agent(agent)
                else 0.0
            ),
            "high_deadline_lambda_current": float(
                agent.high_agent.deadline_lambda
            ),
            "low_deadline_lambda_current": (
                float(agent.low_agent.deadline_lambda)
                if _is_low_sac_agent(agent)
                else 0.0
            ),
            "high_algorithm": high_algorithm,
            "constraint_handling": constraint_handling,
            "deadline_lagrangian_mode": agent.deadline_lagrangian_mode,
            "low_constraint_handling": (
                "shared_lagrangian_deadline"
                if agent.shared_deadline_lagrangian_enabled
                else "lagrangian_deadline"
                if _is_low_sac_agent(agent)
                and getattr(agent.low_agent, "deadline_lagrangian", None)
                is not None
                else "lagrangian_reliability"
                if _is_low_sac_agent(agent)
                and getattr(agent.low_agent, "reliability_lagrangian", None)
                is not None
                else "reward_only"
            ),
            "high_action_mask_mode": "structural_only",
            "training_mode": (
                f"{high_algorithm}_high_frozen_low_{low_algorithm}"
                if freeze_high
                else f"{high_algorithm}_high_low_{low_algorithm}_frozen"
                if freeze_low
                else (
                    f"{high_algorithm}_high_low_{low_algorithm}"
                    if not fixed_low_rho_one
                    else f"{high_algorithm}_high_fixed_rho1"
                )
            ),
            "update_mode": (
                "low_episode_end_replay"
                if freeze_high or high_algorithm != "d3qn"
                else "high_episode_end_replay"
                if freeze_low
                else "episode_end_replay"
            ),
            "high_update_count": high_update_count,
            "low_update_count": low_update_count,
            "low_update_skipped_due_to_empty_buffer": bool(
                low_update_skipped and not freeze_low
            ),
            "low_update_skipped_due_to_freeze": bool(freeze_low),
            "high_updates_this_episode": int(high_metrics.get("updates", 0.0)),
            "low_updates_this_episode": int(low_metrics.get("updates", 0.0)),
            "high_lambda_updated_this_episode": bool(
                dual_metrics.get("deadline_dual_updated", 0.0)
            ),
            "low_lambda_updated_this_episode": bool(
                low_dual_metrics.get("low_deadline_dual_updated", 0.0)
            ),
            **result["metrics"],
            **dual_metrics,
            **low_dual_metrics,
            **low_reliability_dual_metrics,
        }
        row.update({f"high_{key}": value for key, value in high_metrics.items()})
        row.update({f"low_{key}": value for key, value in low_metrics.items()})
        row.update(
            {f"coord_{key}": value for key, value in coordination_metrics.items()}
        )
        if low_dual_metrics:
            row["low_deadline_lambda"] = float(
                low_dual_metrics["low_deadline_lambda_after_update"]
            )
        if queue_weight_calibrator is not None:
            # Emitted every episode, not only on the solve, so the CSV keeps a
            # stable header and the accumulating I/Q masses stay inspectable
            # while the window is still open.
            row.update(queue_weight_calibrator.diagnostics())
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
        if int(training_cfg.get("eval_interval", 20)) > 0 and (
            global_episode % int(training_cfg.get("eval_interval", 20)) == 0
            or episode == total_episodes
        ):
            validation_blocks = []
            for block_index, eval_seed in enumerate(eval_seeds):
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
                validation_blocks.append(eval_metrics)
                eval_row = {
                    "episode": episode,
                    "validation_block_index": block_index,
                    "validation_block_count": len(eval_seeds),
                    "global_episode": global_episode,
                    "phase_name": str(
                        training_cfg.get("phase_name", "single_stage")
                    ),
                    "phase_index": int(training_cfg.get("phase_index", 1)),
                    "phase_count": int(training_cfg.get("phase_count", 1)),
                    "high_active_episode": (
                        high_active_episode_offset + episode
                        if high_algorithm == "d3qn" and not freeze_high
                        else high_active_episode_offset
                    ),
                    "low_active_episode": (
                        low_active_episode_offset + episode
                        if not freeze_low and not fixed_low_rho_one
                        else low_active_episode_offset
                    ),
                    "high_policy_status": (
                        "fixed_heuristic_eval"
                        if high_algorithm != "d3qn"
                        else "evaluation_deterministic"
                    ),
                    "low_policy_status": (
                        "fixed_rho1"
                        if fixed_low_rho_one
                        else "evaluation_deterministic"
                    ),
                    "high_reward_usage": "evaluation_only",
                    "low_reward_usage": (
                        "disabled" if fixed_low_rho_one else "evaluation_only"
                    ),
                    "high_updates_this_episode": 0,
                    "low_updates_this_episode": 0,
                    "high_lambda_updated_this_episode": False,
                    "low_lambda_updated_this_episode": False,
                    "checkpoint_selection_mode": (
                        "esp_feasible_energy_min"
                        if checkpoint_deadline_esp_threshold is not None
                        else "feasible_energy_min"
                        if checkpoint_deadline_target is not None
                        else "deadline_lexicographic"
                    ),
                    "checkpoint_deadline_target": checkpoint_deadline_target,
                    "checkpoint_reliability_target": checkpoint_reliability_target,
                    "checkpoint_deadline_esp_threshold": (
                        checkpoint_deadline_esp_threshold
                    ),
                    "checkpoint_deadline_esp_target": checkpoint_deadline_esp_target,
                    "high_learning_rate": float(
                        0.0
                        if high_algorithm != "d3qn"
                        else agent.high_agent.agent_R.optimizer.param_groups[0]["lr"]
                    ),
                    "low_learning_rate": (
                        float(agent.low_agent.actor_optimizer.param_groups[0]["lr"])
                        if _is_low_sac_agent(agent)
                        else 0.0
                    ),
                    "high_deadline_lambda_current": float(
                        agent.high_agent.deadline_lambda
                    ),
                    "low_deadline_lambda_current": (
                        float(agent.low_agent.deadline_lambda)
                        if _is_low_sac_agent(agent)
                        else 0.0
                    ),
                    "high_algorithm": high_algorithm,
                    "constraint_handling": constraint_handling,
                    "deadline_lagrangian_mode": agent.deadline_lagrangian_mode,
                    "low_constraint_handling": (
                        "shared_lagrangian_deadline"
                        if agent.shared_deadline_lagrangian_enabled
                        else "lagrangian_deadline"
                        if _is_low_sac_agent(agent)
                        and getattr(agent.low_agent, "deadline_lagrangian", None)
                        is not None
                        else "lagrangian_reliability"
                        if _is_low_sac_agent(agent)
                        and getattr(agent.low_agent, "reliability_lagrangian", None)
                        is not None
                        else "reward_only"
                    ),
                    "high_action_mask_mode": "structural_only",
                    "eval_seed": eval_seed,
                    "eval_episodes": int(training_cfg.get("eval_episodes", 5)),
                    **eval_metrics,
                }
                if getattr(agent.high_agent, "deadline_lagrangian", None) is not None:
                    eval_row["lambda_deadline"] = agent.high_agent.deadline_lambda
                if (
                    _is_low_sac_agent(agent)
                    and getattr(agent.low_agent, "deadline_lagrangian", None) is not None
                ):
                    eval_row["low_deadline_lambda"] = (
                        agent.low_agent.deadline_lambda
                    )
                if agent.shared_deadline_lagrangian_enabled:
                    shared_state = agent.shared_deadline_lagrangian.state_dict()
                    shared_window = np.asarray(
                        shared_state.get("workflow_cost_window", []),
                        dtype=np.float64,
                    )
                    eval_row.update(
                        {
                            "shared_deadline_lambda": agent.high_agent.deadline_lambda,
                            "shared_deadline_lambda_before_update": agent.high_agent.deadline_lambda,
                            "shared_deadline_lambda_after_update": agent.high_agent.deadline_lambda,
                            "shared_deadline_constraint_budget": float(
                                shared_state["deadline_budget"]
                            ),
                            "shared_deadline_success_target": float(
                                1.0 - shared_state["deadline_budget"]
                            ),
                            "shared_deadline_constraint_mean": (
                                float(np.mean(shared_window))
                                if shared_window.size
                                else 0.0
                            ),
                            "shared_deadline_constraint_batch_mean": float(
                                eval_metrics.get("deadline_miss_rate", 0.0)
                            ),
                            "shared_deadline_constraint_workflow_count": int(
                                shared_window.size
                            ),
                            "shared_deadline_dual_updated": 0.0,
                            "shared_deadline_dual_warmup_active": 0.0,
                        }
                    )
                eval_history.append(eval_row)
                append_history_row(output_dir, "eval_history", eval_row)
                write_history_files(output_dir, "eval_history", eval_history, rewrite_jsonl=False)
                print_d3qn_metrics_block("eval", eval_row)
            checkpoint_metrics = combine_checkpoint_metrics(validation_blocks)
            if len(validation_blocks) > 1:
                print(
                    f"[eval-pooled] ep={episode} "
                    f"blocks={len(validation_blocks)} "
                    f"{format_satisfaction_summary(checkpoint_metrics)} "
                    f"E={float(checkpoint_metrics['total_energy']) / 1000:.2f}kJ",
                    flush=True,
                )
            candidate_rank = metric_rank(
                checkpoint_metrics,
                deadline_target=checkpoint_deadline_target,
                reliability_target=checkpoint_reliability_target,
                deadline_esp_threshold=checkpoint_deadline_esp_threshold,
                deadline_esp_target=checkpoint_deadline_esp_target,
            )
            best_rank = None if best_metrics is None else metric_rank(
                best_metrics,
                deadline_target=checkpoint_deadline_target,
                reliability_target=checkpoint_reliability_target,
                deadline_esp_threshold=checkpoint_deadline_esp_threshold,
                deadline_esp_target=checkpoint_deadline_esp_target,
            )
            candidate_is_officially_feasible = (
                checkpoint_deadline_esp_threshold is None or candidate_rank[0] >= 1.0
            )
            if candidate_is_officially_feasible and (
                best_rank is None or candidate_rank > best_rank
            ):
                best_metrics = checkpoint_metrics
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
        freeze_high=args.freeze_high,
        freeze_low=args.freeze_low,
        high_checkpoint=args.high_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
        high_learning_rate_override=args.high_learning_rate,
        high_updates_override=args.high_updates_per_episode,
        epsilon_start_override=args.epsilon_start,
    )
    print(f"output_dir={result['output_dir']}")


if __name__ == "__main__":
    main()
