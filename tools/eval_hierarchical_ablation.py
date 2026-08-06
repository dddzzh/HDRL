from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.D3QN.hierarchical_d3qn_agent import HierarchicalD3QNAgent
from src.D3QN.reward import HierarchicalObjective
from src.D3QN.state import HierarchicalStateEncoder
from src.ppo.hierarchical_agent import LowLevelDVFSPPOAgent
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from src.schedulers.heft import HEFTScheduler
from src.simulator.metrics import compute_summary
from tools.train_hierarchical_d3qn import collect_d3qn_episode
from tools.hierarchical_common import (
    _assert_metrics_finite,
    _patch_terminal,
    average_metrics,
    build_env,
    build_episode_metrics_from_buffers,
    build_resources,
    build_workflows,
    completed_workflows,
    configure_agent_state,
    format_metrics,
    load_json,
    print_metrics_block,
    write_history_files,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical ablations.")
    parser.add_argument("--config", default="configs/hierarchical_safe_ppo.json")
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "d3qn_low_ppo",
            "d3qn_low_sac",
            "d3qn_rho1",
            "heft_low_ppo",
            "heft_low_sac",
        ),
        help=(
            "d3qn_low_sac: trained D3QN high + trained low SAC; "
            "d3qn_rho1: trained D3QN high + rho=1; "
            "heft_low_sac: HEFT high + SAC loaded from a full D3QN checkpoint; "
            "d3qn_low_ppo/heft_low_ppo: legacy PPO modes."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Full D3QN checkpoint for D3QN modes or heft_low_sac.",
    )
    parser.add_argument("--low-checkpoint", default=None, help="Low PPO component/full checkpoint for heft_low_ppo.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=242000)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def _resolve_path(path):
    if path is None:
        return None
    result = Path(path)
    if not result.is_absolute():
        result = PROJECT_ROOT / result
    return result


def _load_low_checkpoint(low_agent, path):
    path = _resolve_path(path)
    if path is None:
        raise ValueError("--low-checkpoint is required for mode=heft_low_ppo.")
    payload = torch.load(str(path), map_location=low_agent.device, weights_only=False)
    state_dict = payload.get("low_model_state_dict")
    optimizer_state = payload.get("low_optimizer_state_dict")
    if state_dict is None and payload.get("checkpoint_format") == "hierarchical_safe_ppo_v1":
        state_dict = payload.get("low_model_state_dict")
        optimizer_state = payload.get("low_optimizer_state_dict")
    if state_dict is None:
        raise RuntimeError(f"Checkpoint does not contain low_model_state_dict: {path}")
    low_agent.network.load_state_dict(state_dict)
    if optimizer_state is not None:
        low_agent.optimizer.load_state_dict(optimizer_state)
    return payload


def _reordered_state_for_task(env, state, workflow_id, task_id):
    workflow_id = str(workflow_id)
    task_id = int(task_id)
    ready = list(state.get("ready_tasks", []))
    selected = None
    rest = []
    for task in ready:
        if str(task["workflow_id"]) == workflow_id and int(task["task_id"]) == task_id:
            selected = task
        else:
            rest.append(task)
    if selected is None:
        raise RuntimeError(f"HEFT selected a task that is not ready: workflow={workflow_id}, task={task_id}")
    result = dict(state)
    result["ready_tasks"] = [selected, *rest]
    result["decision_context"] = env.build_hierarchical_decision_context(workflow_id, task_id)
    return result


def _server_index(observation, server_id):
    server_id = int(server_id)
    for index, server in enumerate(observation.servers):
        if int(server["id"]) == server_id:
            return index
    raise RuntimeError(f"Selected server_id={server_id} is not in observation servers.")


def _padded_masks(context, max_servers):
    primary_raw = np.asarray(context["primary_server_mask"], dtype=bool)
    pair_raw = np.asarray(context["backup_pair_mask"], dtype=bool)
    primary_mask = np.zeros(int(max_servers), dtype=bool)
    backup_pair_mask = np.zeros((int(max_servers), int(max_servers)), dtype=bool)
    primary_mask[: primary_raw.shape[0]] = primary_raw
    backup_pair_mask[: pair_raw.shape[0], : pair_raw.shape[1]] = pair_raw
    return primary_mask, backup_pair_mask


def _choose_backup_index(primary_index, backup_pair_mask, primary_mask):
    row = np.asarray(backup_pair_mask[int(primary_index)], dtype=bool)
    candidates = np.flatnonzero(row)
    if candidates.size > 0:
        return int(candidates[0])
    fallback = np.asarray(primary_mask, dtype=bool).copy()
    fallback[int(primary_index)] = False
    candidates = np.flatnonzero(fallback)
    if candidates.size <= 0:
        raise RuntimeError("No legal backup action for HEFT ablation buffer row.")
    return int(candidates[0])


def collect_heft_low_ppo_episode(env, low_agent, objective, high_buffer, low_buffer, *, encoder):
    state = env.reset()
    objective.reset_episode()
    high_buffer.clear()
    low_buffer.clear()
    scheduler = HEFTScheduler(params={"rho": 1.0}, seed=env.seed)
    scheduler.reset()
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
    heft_primary_mask_violation_count = 0
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

        heft_action = scheduler.select_action(
            state,
            context_builder=env.build_hierarchical_decision_context,
        )
        state_for_task = _reordered_state_for_task(env, state, heft_action["workflow_id"], heft_action["task_id"])
        observation = encoder.encode_high_state(state_for_task)
        context = state_for_task["decision_context"]
        primary_index = _server_index(observation, heft_action["server_id"])
        primary_mask, backup_pair_mask = _padded_masks(context, encoder.max_servers)
        heft_primary_mask_violation = not bool(primary_mask[int(primary_index)])
        primary_mask[int(primary_index)] = True
        backup_pair_mask[int(primary_index), int(primary_index)] = False
        backup_index = _choose_backup_index(primary_index, backup_pair_mask, primary_mask)
        backup_pair_mask[int(primary_index), int(backup_index)] = True
        backup_server_id = int(observation.servers[backup_index]["id"])
        selected_metrics = context["server_safety_metrics"][str(heft_action["server_id"])]
        low_decision = low_agent.select_frequency(
            observation,
            0,
            primary_index,
            deterministic=True,
            safety_metrics=selected_metrics,
        )
        high_decision = {
            "observation": observation,
            "state": observation.vector,
            "task_index": 0,
            "primary_server_index": int(primary_index),
            "backup_server_index": int(backup_index),
            "server_index": int(primary_index),
            "workflow_id": str(heft_action["workflow_id"]),
            "task_id": int(heft_action["task_id"]),
            "primary_server_id": int(heft_action["server_id"]),
            "backup_server_id": int(backup_server_id),
            "server_id": int(heft_action["server_id"]),
            "primary_server_mask": primary_mask,
            "backup_pair_mask": backup_pair_mask,
            "log_prob": 0.0,
            "value": 0.0,
            "deadline_value": 0.0,
            "reliability_value": 0.0,
            "entropy": 0.0,
            "shield_relaxed": bool(context.get("shield_relaxed", False)),
            "backup_shield_relaxed": not bool(backup_pair_mask[int(primary_index)].any()),
            "heft_primary_mask_violation": bool(heft_primary_mask_violation),
            "selector_mode": "heft",
        }
        env_action = {
            "workflow_id": high_decision["workflow_id"],
            "task_id": high_decision["task_id"],
            "server_id": high_decision["primary_server_id"],
            "rho": low_decision["rho"],
            "backup": True,
            "backup_server_id": high_decision["backup_server_id"],
            "primary_energy_ref_rho1": float(selected_metrics["primary_energy_ref_rho1"]),
        }
        next_state, env_reward, done, info = env.step(env_action)
        feedback = objective.compute_step_feedback(
            env,
            state_for_task,
            high_decision,
            low_decision,
            env_action,
            env_reward,
            info,
        )
        high_buffer.add(
            state=high_decision["state"],
            primary_action=high_decision["primary_server_index"],
            backup_action=high_decision["backup_server_index"],
            old_log_prob=0.0,
            reward_value=0.0,
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
        if bool(low_decision.get("actor_update_mask", False)):
            low_accepted_count += 1
            low_buffer.add(
                state=low_decision["state"],
                action_u=low_decision["action_u"],
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
        backup_counts[str(high_decision["backup_server_id"])] = backup_counts.get(str(high_decision["backup_server_id"]), 0) + 1
        shield_relaxed_count += int(bool(high_decision.get("shield_relaxed", False)))
        backup_shield_relaxed_count += int(bool(high_decision.get("backup_shield_relaxed", False)))
        heft_primary_mask_violation_count += int(bool(high_decision.get("heft_primary_mask_violation", False)))
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
        "heft_primary_mask_violation_count": heft_primary_mask_violation_count,
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


def evaluate(config, args):
    resources = build_resources(config, args.seed)
    agent_config = configure_agent_state(config, resources, smoke_test=args.smoke_test)
    config = copy.deepcopy(config)
    config["agent"] = agent_config
    objective = HierarchicalObjective(config.get("objective", {}))
    rows = []

    if args.mode in {"d3qn_low_ppo", "d3qn_low_sac", "d3qn_rho1", "heft_low_sac"}:
        agent_config["use_low_agent"] = args.mode != "d3qn_rho1"
        agent = HierarchicalD3QNAgent(agent_config, seed=args.seed)
        if not args.checkpoint:
            raise ValueError(f"--checkpoint is required for mode={args.mode}.")
        agent.load_checkpoint(_resolve_path(args.checkpoint))
        agent.high_agent.epsilon = 0.0
        high_algorithm = "heft" if args.mode == "heft_low_sac" else "d3qn"
        for offset in range(int(args.episodes)):
            workflows = build_workflows(config, resources, args.seed + offset, smoke_test=args.smoke_test)
            env = build_env(workflows, resources, dict(config.get("simulation", {})), args.seed + offset)
            result = collect_d3qn_episode(
                env,
                agent,
                objective,
                HighRolloutBuffer(),
                LowRolloutBuffer(),
                training=False,
                high_deterministic=True,
                low_deterministic=True,
                fixed_low_rho_one=args.mode == "d3qn_rho1",
                high_algorithm=high_algorithm,
            )
            rows.append({"episode": offset + 1, "mode": args.mode, **result["metrics"]})
    else:
        low_agent = LowLevelDVFSPPOAgent({**agent_config.get("shared", {}), **agent_config.get("low", {})}, seed=args.seed)
        _load_low_checkpoint(low_agent, args.low_checkpoint)
        encoder = HierarchicalStateEncoder(agent_config.get("shared", {}).get("state", {}))
        for offset in range(int(args.episodes)):
            workflows = build_workflows(config, resources, args.seed + offset, smoke_test=args.smoke_test)
            env = build_env(workflows, resources, dict(config.get("simulation", {})), args.seed + offset)
            result = collect_heft_low_ppo_episode(
                env,
                low_agent,
                objective,
                HighRolloutBuffer(),
                LowRolloutBuffer(),
                encoder=encoder,
            )
            rows.append({"episode": offset + 1, "mode": args.mode, **result["metrics"]})
    return rows


def main():
    args = parse_args()
    config = load_json(args.config)
    rows = evaluate(config, args)
    summary = average_metrics(rows)
    output_root = PROJECT_ROOT / "results" / "hierarchical_ablation" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)
    write_history_files(output_root, args.mode, rows)
    with (output_root / f"{args.mode}_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print_metrics_block(args.mode, {"episode": int(args.episodes), **summary})
    print(f"[summary] {format_metrics(summary)}")
    print(f"output_dir={output_root}")


if __name__ == "__main__":
    main()
