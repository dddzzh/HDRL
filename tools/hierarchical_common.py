from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.workflow_factory import create_workflows
from src.simulator.deadline import DEADLINE_PROBABILITIES, validate_deadline_type
from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.resource import load_resources


DEADLINE_TYPES = tuple(sorted(DEADLINE_PROBABILITIES, key=DEADLINE_PROBABILITIES.get, reverse=True))


def apply_deadline_type(config, deadline_type):
    """Return ``config`` re-targeted at a different Deadline regime.

    Only ``simulation.deadline_type`` changes. The current sec-job generator
    uses an independent, per-workflow multiplier stream, so arrivals, payloads
    and the shared FCFS reference stay identical across regimes for a fixed
    seed and resource pool. Tight/medium/loose are paired comparisons.
    """
    if deadline_type is None:
        return config
    result = copy.deepcopy(config)
    result.setdefault("simulation", {})["deadline_type"] = validate_deadline_type(
        deadline_type
    )
    return result


def config_deadline_type(config):
    return validate_deadline_type(
        config.get("simulation", {}).get("deadline_type", "medium")
    )


def satisfaction_statistics(
    success_rates,
    thresholds=(0.90, 0.92, 0.95, 1.0),
    tail_fraction=0.25,
):
    """Summarize per-run Deadline success rates as ESP and CVaR.

    ``esp[gamma]`` is the fraction of runs whose own success rate reaches
    ``gamma`` -- the empirical satisfaction probability.  With M runs it only
    resolves to 1/M and discards the magnitude of a miss, so it is a reporting
    statistic.  ``cvar`` is the mean of the worst ``tail_fraction`` of runs: it
    is continuous, tail-sensitive, and is what checkpoint selection should rank
    on when the goal is to raise ESP.
    """
    values = np.asarray([float(value) for value in success_rates], dtype=np.float64)
    if values.size == 0:
        return {}
    ordered = np.sort(values)
    tail = max(1, int(math.ceil(float(tail_fraction) * values.size)))
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(ordered[0]),
        "esp": {f"{value:.2f}": float(np.mean(values >= value - 1e-9))
                for value in thresholds},
        "cvar_tail_fraction": float(tail) / float(values.size),
        "cvar": float(ordered[:tail].mean()),
    }


# Which per-run rates each evaluation block reports as ESP, and at which
# thresholds.  Deadline carries the full ladder because the acceptance rule
# (Gamma_D = 0.92) sits inside it; reliability is only ever reported at 100%,
# which is the checkpoint target, so a ladder there would be dead columns.
SATISFACTION_METRICS = (
    ("deadline_success_rate", "deadline", (0.90, 0.92, 0.95, 1.0)),
    ("reliability_success_rate", "reliability", (1.0,)),
)


def satisfaction_field_names(metrics=SATISFACTION_METRICS):
    """Flat column names produced by :func:`attach_satisfaction_metrics`.

    CSV writers in the baseline trainers use ``extrasaction="ignore"``, so the
    header list has to be extended with exactly these names or the columns are
    silently dropped.
    """
    names = []
    for _, prefix, thresholds in metrics:
        names.extend(f"{prefix}_esp_{value:.2f}".replace(".", "p")
                     for value in thresholds)
        names.extend((f"{prefix}_cvar25", f"{prefix}_min", f"{prefix}_std"))
    return tuple(names)


def satisfaction_sample_key(metric_key):
    return f"{metric_key}_samples"


def attach_satisfaction_metrics(metrics, rows, satisfaction_metrics=SATISFACTION_METRICS):
    """Add flat ESP/CVaR columns to an averaged evaluation metrics dict.

    ``metrics`` is the mean over the evaluation block and hides the shape of the
    per-run distribution: a block that misses the Deadline on a few congested
    instances and one that misses a little everywhere have the same mean but
    very different ESP.  ``rows`` are the per-run metric dicts the mean was
    taken over, so the spread is recovered here rather than re-derived later.

    The per-run values themselves are kept under ``<metric>_samples`` so two
    validation blocks can be pooled exactly (see
    :func:`combine_checkpoint_metrics`) instead of having their block-level
    summaries combined, which no ESP or CVaR can be recovered from.  The CSV
    writers drop these list columns; the JSON/JSONL histories keep them.
    Mutates and returns ``metrics``.
    """
    rows = list(rows)
    for key, prefix, thresholds in satisfaction_metrics:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        metrics[satisfaction_sample_key(key)] = values
        stats = satisfaction_statistics(values, thresholds=thresholds)
        for value in thresholds:
            name = f"{prefix}_esp_{value:.2f}".replace(".", "p")
            metrics[name] = stats.get("esp", {}).get(f"{value:.2f}")
        metrics[f"{prefix}_cvar25"] = stats.get("cvar")
        metrics[f"{prefix}_min"] = stats.get("min")
        metrics[f"{prefix}_std"] = stats.get("std")
    return metrics


def load_json(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def build_resources(config, seed):
    resource_cfg = dict(config.get("resources", {}))
    path = Path(resource_cfg.get("path", "configs/resources_edge_cloud_default.json"))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return load_resources(
        str(path),
        num_edge=int(resource_cfg.get("num_edge", 2)),
        num_cloud=int(resource_cfg.get("num_cloud", 1)),
        seed=seed,
    )


def build_workflows(config, resources, seed, smoke_test=False):
    generator_cfg = copy.deepcopy(config.get("generator", {}))
    workflow_source = str(
        generator_cfg.get("source", generator_cfg.get("type", "pegasus"))
    ).strip().lower()
    simulation_cfg = config.get("simulation", {})
    generator_cfg["deadline_type"] = simulation_cfg.get("deadline_type", "medium")
    generator_cfg["workflow_reliability_requirement"] = simulation_cfg.get(
        "workflow_reliability_requirement",
        0.9,
    )
    if smoke_test:
        generator_cfg.update({
            "num_workflows": 2,
            "max_workflows": 2,
            "dataset_scale": "small",
            "arrival_rate": 0.05,
        })
    return create_workflows(
        generator_cfg,
        source=workflow_source,
        seed=seed,
        resources=resources,
    )


def configure_agent_state(config, resources, smoke_test=False):
    result = copy.deepcopy(config.get("agent", {}))
    result.setdefault("shared", {}).setdefault("state", {})
    state_cfg = result["shared"]["state"]
    if smoke_test:
        state_cfg["max_ready_tasks"] = min(int(state_cfg.get("max_ready_tasks", 32)), 20)
    state_cfg["max_servers"] = len(resources.get("servers", []))
    result.setdefault("low", {})["rho_min"] = float(config.get("simulation", {}).get("rho_min", result.get("low", {}).get("rho_min", 0.1)))
    return result


def build_env(workflows, resources, sim_config, seed):
    return EdgeCloudWorkflowEnv(workflows=workflows, resources=resources, sim_config=sim_config, seed=seed)


def completed_workflows(env, seen):
    rows = []
    for row in env._build_workflow_results():
        workflow_id = str(row["workflow_id"])
        if workflow_id not in seen and row.get("response_time") is not None:
            rows.append(row)
    return rows


_RISK_CALIBRATION_BINS = np.asarray(
    [
        0.0,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0 + 1e-8,
    ],
    dtype=np.float64,
)


def _workflow_balanced_risk_calibration(high_buffer, high_decisions):
    probabilities = []
    labels = []
    weights = []
    for _workflow_id, indices in high_buffer.workflow_groups():
        if not indices:
            continue
        final_index = indices[-1]
        deadline = max(float(high_buffer.deadline_limits[final_index]), 1e-8)
        completion = float(high_buffer.partial_makespans[final_index])
        label = float(completion > deadline + 1e-8)
        workflow_weight = 1.0 / len(indices)
        for index in indices:
            if "risk_selected" not in high_decisions[index]:
                continue
            probabilities.append(float(high_decisions[index]["risk_selected"]))
            labels.append(label)
            weights.append(workflow_weight)
    if not probabilities:
        return {}
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = max(float(weights.sum()), 1e-8)
    brier = float(
        np.sum(weights * np.square(probabilities - labels)) / weight_sum
    )
    ece = 0.0
    bins = {}
    occupied = 0
    for lower, upper in zip(
        _RISK_CALIBRATION_BINS[:-1], _RISK_CALIBRATION_BINS[1:]
    ):
        member = (probabilities >= lower) & (probabilities < upper)
        if not np.any(member):
            continue
        bucket_weight = float(weights[member].sum())
        predicted = float(
            np.sum(weights[member] * probabilities[member]) / bucket_weight
        )
        observed = float(
            np.sum(weights[member] * labels[member]) / bucket_weight
        )
        ece += bucket_weight / weight_sum * abs(predicted - observed)
        bins[f"{lower:.3f}:{min(upper, 1.0):.3f}"] = {
            "predicted": predicted,
            "observed": observed,
            "workflow_weight": bucket_weight,
            "transition_count": int(member.sum()),
        }
        occupied += 1
    return {
        "risk_rollout_brier_score": brier,
        "risk_rollout_ece": float(ece),
        "risk_rollout_occupied_bins": float(occupied),
        "risk_reliability_diagram": bins,
    }




def build_episode_metrics_from_buffers(high_buffer, low_buffer, summary, event_log, diagnostics=None):
    diagnostics = dict(diagnostics or {})
    high_decisions = [
        dict(info.get("high_decision", {})) for info in high_buffer.infos
    ]
    dual_energy_action_std = np.asarray(
        [float(row.get("dual_energy_action_std", 0.0)) for row in high_decisions],
        dtype=np.float64,
    )
    dual_deadline_action_std = np.asarray(
        [float(row.get("dual_deadline_action_std", 0.0)) for row in high_decisions],
        dtype=np.float64,
    )
    dual_deadline_to_energy_scale = np.asarray(
        [
            float(row.get("dual_deadline_to_energy_scale", 0.0))
            for row in high_decisions
        ],
        dtype=np.float64,
    )
    has_dual_head_scale_metrics = any(
        "dual_energy_action_std" in decision for decision in high_decisions
    )
    has_feasible_risk_metrics = any(
        "risk_selected" in decision for decision in high_decisions
    )
    has_low_sac_metrics = any("low_sac_energy_saved" in info for info in high_buffer.infos)
    has_high_replay_zscore_metrics = any(
        "high_workflow_slack_ratio" in info for info in high_buffer.infos
    )
    has_high_deadline_incremental_metrics = any(
        info.get("high_reward_mode") == "deadline_incremental_only"
        for info in high_buffer.infos
    )
    has_high_lagrangian_metrics = any(
        info.get("high_reward_mode") in {
            "lagrangian_deadline",
            "feasible_deadline_risk",
        }
        for info in high_buffer.infos
    )
    has_high_feasible_risk_metrics = any(
        info.get("high_reward_mode") == "feasible_deadline_risk"
        for info in high_buffer.infos
    )
    has_sub_deadline_metrics = any(
        info.get("high_reward_profile") == "sub_deadline_lagrangian_v1"
        or float(info.get("deadline_sub_weight", 0.0)) > 0.0
        for info in high_buffer.infos
    )
    high_rewards = np.asarray(high_buffer.rewards, dtype=np.float64)
    high_workflow_slack_ratios = np.asarray(
        _info_values(high_buffer.infos, "high_workflow_slack_ratio"),
        dtype=np.float64,
    )
    high_deadline_distances = np.asarray(
        _info_values(high_buffer.infos, "high_deadline_distance_seconds"),
        dtype=np.float64,
    )
    high_energy_savings = np.asarray(
        _info_values(high_buffer.infos, "high_energy_saving_joules"),
        dtype=np.float64,
    )
    high_deadline_missed = np.asarray(
        _info_values(high_buffer.infos, "high_deadline_missed"),
        dtype=np.float64,
    )
    risk_selected = np.asarray(
        [float(row.get("risk_selected", 0.0)) for row in high_decisions],
        dtype=np.float64,
    )
    risk_selected_rank = np.asarray(
        [float(row.get("risk_selected_rank", 0.0)) for row in high_decisions],
        dtype=np.float64,
    )
    feasible_action_counts = np.asarray(
        [float(row.get("feasible_action_count", 0.0)) for row in high_decisions],
        dtype=np.float64,
    )
    no_feasible_actions = np.asarray(
        [float(row.get("no_feasible_action", 0.0)) for row in high_decisions],
        dtype=np.float64,
    )
    risk_thresholds = np.asarray(
        [float(row.get("risk_threshold", 0.0)) for row in high_decisions],
        dtype=np.float64,
    )
    risk_gate_active = np.asarray(
        [float(row.get("risk_gate_active", 0.0)) for row in high_decisions],
        dtype=np.float64,
    )
    high_low_reward_bonuses = np.asarray(_info_values(high_buffer.infos, "high_low_reward_bonus"), dtype=np.float64)
    low_energy_saving_ratios = np.asarray(_info_values(high_buffer.infos, "low_energy_saving_ratio"), dtype=np.float64)
    task_time_regrets = np.asarray(_info_values(high_buffer.infos, "task_time_regret"), dtype=np.float64)
    task_complete_time_ratios = np.asarray(_info_values(high_buffer.infos, "task_complete_time_ratio"), dtype=np.float64)
    task_actual_times = np.asarray(_info_values(high_buffer.infos, "task_actual_time"), dtype=np.float64)
    task_fastest_times = np.asarray(_info_values(high_buffer.infos, "task_fastest_time"), dtype=np.float64)
    current_slack_ratios = np.asarray(_info_values(high_buffer.infos, "current_slack_ratio"), dtype=np.float64)
    normalized_makespan_increments = np.asarray(
        _info_values(high_buffer.infos, "normalized_makespan_increment"),
        dtype=np.float64,
    )
    sub_deadline_relative = np.asarray(
        _info_values(high_buffer.infos, "sub_deadline_relative"),
        dtype=np.float64,
    )
    sub_deadline_finish_relative = np.asarray(
        _info_values(high_buffer.infos, "sub_deadline_finish_relative"),
        dtype=np.float64,
    )
    sub_deadline_constraint_costs = np.asarray(
        _info_values(high_buffer.infos, "sub_deadline_constraint_cost"),
        dtype=np.float64,
    )
    high_base_energy_rewards = np.asarray(
        _info_values(high_buffer.infos, "high_base_energy_reward"),
        dtype=np.float64,
    )
    high_local_energy_rewards = np.asarray(
        _info_values(high_buffer.infos, "high_base_energy_reward_local"),
        dtype=np.float64,
    )
    high_team_energy_rewards = np.asarray(
        _info_values(high_buffer.infos, "high_team_energy_reward"),
        dtype=np.float64,
    )
    workflow_team_energy_ratios = np.asarray(
        _info_values(high_buffer.infos, "workflow_team_energy_ratio"),
        dtype=np.float64,
    )
    high_deadline_constraint_costs = np.asarray(
        _info_values(high_buffer.infos, "high_deadline_constraint_cost"),
        dtype=np.float64,
    ) + np.asarray(
        _info_values(
            high_buffer.infos, "terminal_high_deadline_constraint_cost"
        ),
        dtype=np.float64,
    )
    # beta_q diagnostics.  ``high_queue_constraint_cost`` is the part of the
    # Deadline-head cost above that comes from server contention rather than
    # from Delta-C_t, so its share says how much of the head's signal the
    # queue term now carries.  Both are zero when beta_q = 0.
    high_queue_delay_seconds = np.asarray(
        _info_values(high_buffer.infos, "high_queue_delay_seconds"),
        dtype=np.float64,
    )
    high_queue_delay_normalized = np.asarray(
        _info_values(high_buffer.infos, "high_queue_delay_normalized"),
        dtype=np.float64,
    )
    high_queue_constraint_costs = np.asarray(
        _info_values(high_buffer.infos, "high_queue_constraint_cost"),
        dtype=np.float64,
    )
    high_deadline_lambdas = np.asarray(
        _info_values(high_buffer.infos, "deadline_lambda"), dtype=np.float64
    )
    high_lagrangian_penalties = (
        high_deadline_lambdas * high_deadline_constraint_costs
    )
    high_lagrangian_training_rewards = (
        high_base_energy_rewards - high_lagrangian_penalties
    )
    high_slack_reward_raw = np.asarray(_info_values(high_buffer.infos, "high_slack_reward_raw"), dtype=np.float64)
    high_makespan_rewards = np.asarray(
        _info_values(high_buffer.infos, "high_makespan_reward_component"), dtype=np.float64
    )
    high_makespan_rewards += np.asarray(
        _info_values(high_buffer.infos, "terminal_high_makespan_reward"), dtype=np.float64
    )
    high_deadline_penalties = np.asarray(
        _info_values(high_buffer.infos, "high_deadline_penalty"), dtype=np.float64
    )
    high_deadline_crossings = np.asarray(
        _info_values(high_buffer.infos, "high_deadline_crossed"), dtype=np.float64
    )
    high_deadline_violations = np.asarray(
        _info_values(high_buffer.infos, "high_deadline_violation"), dtype=np.float64
    )
    high_deadline_violation_deltas = np.asarray(
        _info_values(high_buffer.infos, "high_deadline_violation_delta"),
        dtype=np.float64,
    )
    high_reliability_penalties = np.asarray(
        _info_values(high_buffer.infos, "high_reliability_penalty"), dtype=np.float64
    )
    reliability_crossings = np.asarray(
        _info_values(high_buffer.infos, "reliability_crossed"), dtype=np.float64
    )
    reliability_violation_deltas = np.asarray(
        _info_values(high_buffer.infos, "reliability_violation_delta"), dtype=np.float64
    )
    low_sac_energy_saved = np.asarray(
        _info_values(high_buffer.infos, "low_sac_energy_saved"), dtype=np.float64
    )
    low_sac_energy_saving = np.asarray(
        _info_values(high_buffer.infos, "low_sac_energy_saving_ratio"), dtype=np.float64
    )
    low_sac_energy_reference = np.asarray(
        _info_values(high_buffer.infos, "low_sac_energy_reference"), dtype=np.float64
    )
    low_sac_energy_actual = np.asarray(
        _info_values(high_buffer.infos, "low_sac_energy_actual"), dtype=np.float64
    )
    low_sac_time_increase = np.asarray(
        _info_values(high_buffer.infos, "low_sac_time_increase"), dtype=np.float64
    )
    low_sac_time_increase_seconds = np.asarray(
        _info_values(high_buffer.infos, "low_sac_time_increase_seconds"), dtype=np.float64
    )
    low_sac_slack_time_cost = np.asarray(
        _info_values(high_buffer.infos, "low_sac_slack_time_cost"), dtype=np.float64
    )
    low_sac_slack_before_seconds = np.asarray(
        _info_values(high_buffer.infos, "low_sac_slack_before_seconds"),
        dtype=np.float64,
    )
    low_sac_reference_sub_cost = np.asarray(
        _info_values(high_buffer.infos, "low_sac_reference_sub_deadline_cost"),
        dtype=np.float64,
    )
    low_sac_actual_sub_cost = np.asarray(
        _info_values(high_buffer.infos, "low_sac_actual_sub_deadline_cost"),
        dtype=np.float64,
    )
    low_sac_incremental_sub_cost = np.asarray(
        _info_values(high_buffer.infos, "low_sac_sub_deadline_constraint_cost"),
        dtype=np.float64,
    )
    low_sac_deadline_violation = np.asarray(
        _info_values(high_buffer.infos, "low_sac_deadline_violation"), dtype=np.float64
    )
    low_sac_reliability_violation = np.asarray(
        _info_values(high_buffer.infos, "low_sac_reliability_violation"), dtype=np.float64
    )
    low_sac_deadline_crossed = np.asarray(
        _info_values(high_buffer.infos, "low_sac_deadline_crossed"), dtype=np.float64
    )
    low_sac_reliability_crossed = np.asarray(
        _info_values(high_buffer.infos, "low_sac_reliability_crossed"), dtype=np.float64
    )
    deadline_costs = np.asarray(high_buffer.deadline_costs, dtype=np.float64)
    reliability_costs = np.asarray(high_buffer.reliability_costs, dtype=np.float64)
    low_rewards = np.asarray(low_buffer.rewards, dtype=np.float64)
    low_deadline_costs = np.asarray(getattr(low_buffer, "deadline_costs", []), dtype=np.float64)
    primary_energy = float(sum(float(row.get("energy", 0.0) or 0.0) for row in event_log if row.get("replica_role") == "primary"))
    backup_energy = float(sum(float(row.get("energy", 0.0) or 0.0) for row in event_log if row.get("replica_role") == "backup"))
    global_makespan_row = _global_makespan_event(event_log)
    metrics = {
        "high_buffer_size": high_buffer.size,
        "low_buffer_size": low_buffer.size,
        "mean_high_reward": _array_mean(high_rewards),
        "total_high_reward": _array_sum(high_rewards),
        "mean_high_makespan_reward": _array_mean(high_makespan_rewards),
        "total_high_makespan_reward": _array_sum(high_makespan_rewards),
        "mean_high_task_time_reward": _array_mean(high_makespan_rewards),
        "total_high_task_time_reward": _array_sum(high_makespan_rewards),
        "mean_task_time_regret": _array_mean(task_time_regrets),
        "total_task_time_regret": _array_sum(task_time_regrets),
        "mean_task_complete_time_ratio": _array_mean(task_complete_time_ratios),
        "max_task_complete_time_ratio": _array_max(task_complete_time_ratios),
        "mean_task_actual_time": _array_mean(task_actual_times),
        "mean_task_fastest_time": _array_mean(task_fastest_times),
        "mean_current_slack_ratio": _array_mean(current_slack_ratios),
        "min_current_slack_ratio": _array_min(current_slack_ratios),
        "mean_normalized_makespan_increment": _array_mean(
            normalized_makespan_increments
        ),
        "total_normalized_makespan_increment": _array_sum(
            normalized_makespan_increments
        ),
        "mean_high_slack_reward_raw": _array_mean(high_slack_reward_raw),
        "total_high_slack_reward_raw": _array_sum(high_slack_reward_raw),
        "mean_high_low_reward_bonus": _array_mean(high_low_reward_bonuses),
        "total_high_low_reward_bonus": _array_sum(high_low_reward_bonuses),
        "mean_high_local_energy_reward": _array_mean(high_local_energy_rewards),
        "total_high_local_energy_reward": _array_sum(high_local_energy_rewards),
        "mean_high_team_energy_reward": _array_mean(high_team_energy_rewards),
        "total_high_team_energy_reward": _array_sum(high_team_energy_rewards),
        "mean_workflow_team_energy_ratio": _array_mean(
            workflow_team_energy_ratios
        ),
        "mean_high_deadline_penalty": _array_mean(high_deadline_penalties),
        "total_high_deadline_penalty": _array_sum(high_deadline_penalties),
        "high_deadline_crossing_count": _array_sum(high_deadline_crossings),
        "mean_high_deadline_violation": _array_mean(high_deadline_violations),
        "total_high_deadline_violation": _array_sum(high_deadline_violations),
        "mean_high_deadline_violation_delta": _array_mean(
            high_deadline_violation_deltas
        ),
        "total_high_deadline_violation_delta": _array_sum(
            high_deadline_violation_deltas
        ),
        "mean_high_reliability_penalty": _array_mean(high_reliability_penalties),
        "total_high_reliability_penalty": _array_sum(high_reliability_penalties),
        "reliability_crossing_count": _array_sum(reliability_crossings),
        "mean_reliability_violation_delta": _array_mean(reliability_violation_deltas),
        "total_reliability_violation_delta": _array_sum(reliability_violation_deltas),
        "mean_low_energy_saving_ratio": _array_mean(low_energy_saving_ratios),
        "mean_low_reward": _array_mean(low_rewards),
        "total_low_reward": _array_sum(low_rewards),
        "mean_low_deadline_cost": _array_mean(low_deadline_costs),
        "total_low_deadline_cost": _array_sum(low_deadline_costs),
        "mean_deadline_cost": _array_mean(deadline_costs),
        "total_deadline_cost": _array_sum(deadline_costs),
        "mean_reliability_cost": _array_mean(reliability_costs),
        "total_reliability_cost": _array_sum(reliability_costs),
        "deadline_success_rate": summary.get("deadline_success_rate"),
        "reliability_success_rate": summary.get("reliability_success_rate"),
        "avg_workflow_reliability": summary.get("avg_workflow_reliability", 0.0),
        "total_energy": summary.get("total_energy"),
        "primary_energy": primary_energy,
        "backup_energy": backup_energy,
        "average_workflow_response_time": summary.get("avg_workflow_response_time"),
        "average_response_time": summary.get("avg_workflow_response_time"),
        "episode_makespan": summary.get("episode_makespan", 0.0),
        "avg_workflow_makespan": summary.get("avg_workflow_makespan", 0.0),
        "max_workflow_makespan": summary.get("max_workflow_makespan", 0.0),
        "global_makespan": float(global_makespan_row.get("finish_time", 0.0)),
        "global_makespan_workflow_id": global_makespan_row.get("workflow_id"),
        "global_makespan_task_id": global_makespan_row.get("task_id"),
        "global_makespan_server_id": global_makespan_row.get("server_id"),
    }
    if has_sub_deadline_metrics:
        metrics.update(
            {
                "mean_task_sub_deadline": _array_mean(
                    sub_deadline_relative
                ),
                "mean_task_finish_relative": _array_mean(
                    sub_deadline_finish_relative
                ),
                "mean_sub_deadline_constraint_cost": _array_mean(
                    sub_deadline_constraint_costs
                ),
                "total_sub_deadline_constraint_cost": _array_sum(
                    sub_deadline_constraint_costs
                ),
                "sub_deadline_miss_count": _array_sum(
                    sub_deadline_constraint_costs > 0.0
                ),
            }
        )
    if has_low_sac_metrics:
        total_low_sac_energy_saved = _array_sum(low_sac_energy_saved)
        total_low_sac_energy_reference = _array_sum(low_sac_energy_reference)
        metrics.update(
            {
                "mean_low_sac_energy_saved": _array_mean(low_sac_energy_saved),
                "total_low_sac_energy_saved": total_low_sac_energy_saved,
                "mean_low_sac_energy_saving_ratio": _array_mean(low_sac_energy_saving),
                "low_sac_total_energy_saving_ratio": (
                    total_low_sac_energy_saved / max(total_low_sac_energy_reference, 1e-8)
                ),
                "total_low_sac_energy_reference": total_low_sac_energy_reference,
                "total_low_sac_energy_actual": _array_sum(low_sac_energy_actual),
                "mean_low_sac_time_increase": _array_mean(low_sac_time_increase),
                "mean_low_sac_time_increase_seconds": _array_mean(low_sac_time_increase_seconds),
                "total_low_sac_time_increase_seconds": _array_sum(low_sac_time_increase_seconds),
                "mean_low_sac_slack_time_cost": _array_mean(low_sac_slack_time_cost),
                "mean_low_sac_slack_before_seconds": _array_mean(
                    low_sac_slack_before_seconds
                ),
                "mean_low_sac_deadline_violation": _array_mean(low_sac_deadline_violation),
                "mean_low_sac_reliability_violation": _array_mean(low_sac_reliability_violation),
                "low_sac_deadline_crossing_count": _array_sum(low_sac_deadline_crossed),
                "low_sac_reliability_crossing_count": _array_sum(low_sac_reliability_crossed),
            }
        )
        if has_sub_deadline_metrics:
            metrics.update(
                {
                    "mean_low_sac_reference_sub_deadline_cost": _array_mean(
                        low_sac_reference_sub_cost
                    ),
                    "mean_low_sac_actual_sub_deadline_cost": _array_mean(
                        low_sac_actual_sub_cost
                    ),
                    "mean_low_sac_incremental_sub_deadline_cost": _array_mean(
                        low_sac_incremental_sub_cost
                    ),
                }
            )
    if has_high_replay_zscore_metrics:
        metrics.update(
            {
                "high_reward_mode": "replay_zscore",
                "mean_high_workflow_slack_ratio": _array_mean(
                    high_workflow_slack_ratios
                ),
                "min_high_workflow_slack_ratio": _array_min(
                    high_workflow_slack_ratios
                ),
                "mean_high_deadline_distance_seconds": _array_mean(
                    high_deadline_distances
                ),
                "min_high_deadline_distance_seconds": _array_min(
                    high_deadline_distances
                ),
                "mean_high_energy_saving_joules": _array_mean(
                    high_energy_savings
                ),
                "total_high_energy_saving_joules": _array_sum(
                    high_energy_savings
                ),
                "high_deadline_missed_count": _array_sum(
                    high_deadline_missed
                ),
                "high_deadline_missed_rate": _array_mean(
                    high_deadline_missed
                ),
            }
        )
    elif has_high_deadline_incremental_metrics:
        metrics["high_reward_mode"] = "deadline_incremental_only"
    elif has_high_lagrangian_metrics:
        metrics.update(
            {
                "high_reward_mode": (
                    "feasible_deadline_risk"
                    if has_high_feasible_risk_metrics
                    else "lagrangian_deadline"
                ),
                "mean_high_base_energy_reward": _array_mean(
                    high_base_energy_rewards
                ),
                "total_high_base_energy_reward": _array_sum(
                    high_base_energy_rewards
                ),
                "mean_high_deadline_constraint_cost": _array_mean(
                    high_deadline_constraint_costs
                ),
                "total_high_deadline_constraint_cost": _array_sum(
                    high_deadline_constraint_costs
                ),
                "mean_high_queue_delay_seconds": _array_mean(
                    high_queue_delay_seconds
                ),
                "total_high_queue_delay_seconds": _array_sum(
                    high_queue_delay_seconds
                ),
                "mean_high_queue_delay_normalized": _array_mean(
                    high_queue_delay_normalized
                ),
                "mean_high_queue_constraint_cost": _array_mean(
                    high_queue_constraint_costs
                ),
                "total_high_queue_constraint_cost": _array_sum(
                    high_queue_constraint_costs
                ),
                # Share of the Deadline-head cost carried by the queue term.
                "high_queue_term_share": (
                    float(
                        _array_sum(high_queue_constraint_costs)
                        / _array_sum(high_deadline_constraint_costs)
                    )
                    if abs(float(_array_sum(high_deadline_constraint_costs)))
                    > 1e-12
                    else 0.0
                ),
                "mean_high_lagrangian_penalty": _array_mean(
                    high_lagrangian_penalties
                ),
                "total_high_lagrangian_penalty": _array_sum(
                    high_lagrangian_penalties
                ),
                "mean_high_training_reward": _array_mean(
                    high_lagrangian_training_rewards
                ),
                "total_high_training_reward": _array_sum(
                    high_lagrangian_training_rewards
                ),
            }
        )
    if has_dual_head_scale_metrics:
        metrics.update(
            {
                "dual_energy_action_std_mean": _array_mean(
                    dual_energy_action_std
                ),
                "dual_deadline_action_std_mean": _array_mean(
                    dual_deadline_action_std
                ),
                "dual_deadline_to_energy_scale_mean": _array_mean(
                    dual_deadline_to_energy_scale
                ),
            }
        )
    if has_feasible_risk_metrics:
        metrics.update(
            {
                "risk_selected_mean": _array_mean(risk_selected),
                "risk_selected_p95": float(
                    np.percentile(risk_selected, 95)
                    if risk_selected.size
                    else 0.0
                ),
                "risk_selected_rank_mean": _array_mean(risk_selected_rank),
                "feasible_action_count_mean": _array_mean(
                    feasible_action_counts
                ),
                "no_feasible_action_rate": _array_mean(
                    no_feasible_actions
                ),
                "risk_threshold_mean": _array_mean(risk_thresholds),
                "risk_gate_active_rate": _array_mean(risk_gate_active),
            }
        )
        metrics.update(
            _workflow_balanced_risk_calibration(
                high_buffer, high_decisions
            )
        )
    metrics.update(diagnostics)
    _validate_mean_total(metrics, "high_reward", high_buffer.size)
    _validate_mean_total(metrics, "high_makespan_reward", high_buffer.size)
    _validate_mean_total(metrics, "high_task_time_reward", high_buffer.size)
    _validate_mean_total(metrics, "task_time_regret", high_buffer.size)
    _validate_mean_total(metrics, "high_low_reward_bonus", high_buffer.size)
    _validate_mean_total(metrics, "high_deadline_penalty", high_buffer.size)
    _validate_mean_total(metrics, "high_deadline_violation", high_buffer.size)
    _validate_mean_total(
        metrics, "normalized_makespan_increment", high_buffer.size
    )
    _validate_mean_total(
        metrics, "high_deadline_violation_delta", high_buffer.size
    )
    _validate_mean_total(metrics, "high_reliability_penalty", high_buffer.size)
    _validate_mean_total(metrics, "reliability_violation_delta", high_buffer.size)
    _validate_mean_total(metrics, "deadline_cost", high_buffer.size)
    _validate_mean_total(metrics, "reliability_cost", high_buffer.size)
    _validate_mean_total(metrics, "low_reward", low_buffer.size)
    _validate_mean_total(metrics, "low_deadline_cost", low_buffer.size)
    return metrics


def _info_values(infos, key):
    values = []
    for info in infos:
        value = info.get(key, 0.0)
        values.append(float(value or 0.0))
    return values


def _array_mean(array):
    return float(np.mean(array)) if array.size > 0 else 0.0


def _array_sum(array):
    return float(np.sum(array)) if array.size > 0 else 0.0


def _array_max(array):
    return float(np.max(array)) if array.size > 0 else 0.0


def _array_min(array):
    return float(np.min(array)) if array.size > 0 else 0.0


def _global_makespan_event(event_log):
    best = {}
    best_finish = -float("inf")
    for event in event_log:
        finish_value = event.get(
            "finish_time",
            event.get("final_aft", event.get("logical_finish_time")),
        )
        if finish_value is None:
            continue
        finish_time = float(finish_value)
        if finish_time > best_finish:
            best_finish = finish_time
            best = {
                "finish_time": finish_time,
                "workflow_id": event.get("workflow_id"),
                "task_id": event.get("task_id"),
                "server_id": event.get("server_id"),
            }
    return best if best else {"finish_time": 0.0, "workflow_id": None, "task_id": None, "server_id": None}


def _validate_mean_total(metrics, name, count):
    if int(count) <= 0:
        return
    mean_value = float(metrics[f"mean_{name}"])
    total_value = float(metrics[f"total_{name}"])
    if abs(mean_value * int(count) - total_value) > 1e-6 * max(1.0, abs(total_value)):
        raise RuntimeError(f"Metric mismatch for {name}: mean * count != total.")


def _patch_terminal(objective, high_buffer, workflow_result, completed_seen):
    terminal = objective.compute_terminal_feedback(None, workflow_result)
    info_updates = terminal
    if getattr(objective, "reward_mode", None) in {
        "deadline_energy_replay_zscore",
        "deadline_incremental_only",
        "lagrangian_deadline",
        "feasible_deadline_risk",
    }:
        # The final task step already records the crossing and violation.
        # Terminal bookkeeping must not overwrite those step diagnostics with
        # the normally-zero terminal correction.
        info_updates = {
            key: value
            for key, value in terminal.items()
            if key
            not in {
                "high_deadline_crossed",
                "high_deadline_violation",
                "high_deadline_violation_delta",
                "high_deadline_penalty",
            }
        }
    patched = high_buffer.patch_last_workflow_transition(
        terminal["workflow_id"],
        reward_delta=terminal["high_reward_delta"],
        deadline_cost_delta=terminal["deadline_cost_delta"],
        partial_makespan=terminal["final_partial_makespan"],
        info_updates=info_updates,
    )
    if not patched:
        raise RuntimeError(f"No transition found for completed workflow {terminal['workflow_id']}.")
    high_buffer.mark_workflow_done(terminal["workflow_id"])
    completed_seen.add(str(terminal["workflow_id"]))
    return terminal


def _assert_metrics_finite(metrics):
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise FloatingPointError(f"Episode metric {key} is not finite: {value}")




def average_metrics(rows):
    result = {}
    if not rows:
        return result
    keys = set().union(*(row.keys() for row in rows))
    for key in keys:
        values = [row.get(key) for row in rows]
        if all(isinstance(value, (int, float)) or value is None for value in values):
            finite = [float(value) for value in values if value is not None]
            result[key] = float(np.mean(finite)) if finite else None
        else:
            result[key] = values[-1]
    _recompute_weighted_mean(result, "high_reward", "high_buffer_size")
    _recompute_weighted_mean(result, "high_makespan_reward", "high_buffer_size")
    _recompute_weighted_mean(result, "high_task_time_reward", "high_buffer_size")
    _recompute_weighted_mean(result, "task_time_regret", "high_buffer_size")
    _recompute_weighted_mean(result, "task_complete_time_ratio", "high_buffer_size")
    _recompute_weighted_mean(result, "high_low_reward_bonus", "high_buffer_size")
    _recompute_weighted_mean(result, "high_deadline_penalty", "high_buffer_size")
    _recompute_weighted_mean(result, "high_deadline_violation", "high_buffer_size")
    _recompute_weighted_mean(result, "high_reliability_penalty", "high_buffer_size")
    _recompute_weighted_mean(result, "reliability_violation_delta", "high_buffer_size")
    _recompute_weighted_mean(result, "deadline_cost", "high_buffer_size")
    _recompute_weighted_mean(result, "reliability_cost", "high_buffer_size")
    _recompute_weighted_mean(result, "low_reward", "low_buffer_size")
    return result


def _recompute_weighted_mean(metrics, name, count_key):
    total_key = f"total_{name}"
    mean_key = f"mean_{name}"
    count = metrics.get(count_key)
    total = metrics.get(total_key)
    if isinstance(count, (int, float)) and isinstance(total, (int, float)) and float(count) > 0.0:
        metrics[mean_key] = float(total) / float(count)


def metric_rank(
    metrics,
    deadline_target=None,
    reliability_target=None,
    deadline_esp_threshold=None,
    deadline_esp_target=None,
):
    deadline = metrics.get("deadline_success_rate")
    reliability = metrics.get("reliability_success_rate")
    energy = metrics.get("total_energy")
    response = metrics.get(
        "average_workflow_response_time",
        metrics.get("avg_workflow_response_time", metrics.get("average_response_time")),
    )
    deadline_value = -1.0 if deadline is None else float(deadline)
    reliability_value = -1.0 if reliability is None else float(reliability)
    energy_value = float("inf") if energy is None else float(energy)
    response_value = float("inf") if response is None else float(response)
    esp_enabled = deadline_esp_threshold is not None or deadline_esp_target is not None
    if esp_enabled and (
        deadline_esp_threshold is None or deadline_esp_target is None
    ):
        raise ValueError(
            "deadline_esp_threshold and deadline_esp_target must be set together."
        )
    esp_value = None
    if esp_enabled:
        esp_threshold = float(deadline_esp_threshold)
        esp_required = float(deadline_esp_target)
        if not 0.0 <= esp_threshold <= 1.0:
            raise ValueError("deadline_esp_threshold must be in [0, 1].")
        if not 0.0 <= esp_required <= 1.0:
            raise ValueError("deadline_esp_target must be in [0, 1].")
        esp_name = f"deadline_esp_{esp_threshold:.2f}".replace(".", "p")
        esp_value = metrics.get(f"{esp_name}_min_block", metrics.get(esp_name))
        if esp_value is None:
            raise KeyError(f"Missing checkpoint ESP metric: {esp_name}")
        esp_value = float(esp_value)
    if deadline_target is not None or reliability_target is not None or esp_enabled:
        target = None if deadline_target is None else float(deadline_target)
        reliability_required = (
            None if reliability_target is None else float(reliability_target)
        )
        if target is not None and not 0.0 <= target <= 1.0:
            raise ValueError("deadline_target must be in [0, 1].")
        if (
            reliability_required is not None
            and not 0.0 <= reliability_required <= 1.0
        ):
            raise ValueError("reliability_target must be in [0, 1].")
        deadline_feasible = target is None or deadline_value >= target
        reliability_feasible = (
            reliability_required is None
            or reliability_value >= reliability_required
        )
        esp_feasible = not esp_enabled or esp_value >= esp_required
        feasible = deadline_feasible and reliability_feasible and esp_feasible
        if esp_enabled:
            if feasible:
                # ESP is the only Deadline gate in this mode.  The per-block
                # mean remains available as a diagnostic, but it neither gates
                # nor ranks checkpoints.
                return (
                    1.0,
                    -energy_value,
                    reliability_value,
                    esp_value,
                    -response_value,
                )
            return (
                0.0,
                float(reliability_feasible) + float(esp_feasible),
                esp_value,
                reliability_value,
                -energy_value,
                -response_value,
            )
        if feasible:
            # Once the deadline requirement is met, scheduling is a constrained
            # energy-minimization problem.  Deadline is only a tie-breaker.
            return (
                1.0,
                -energy_value,
                reliability_value,
                1.0 if esp_value is None else esp_value,
                deadline_value,
                -response_value,
            )
        # Before feasibility is reached, recover the deadline constraint first.
        return (
            0.0,
            float(deadline_feasible)
            + float(reliability_feasible)
            + float(esp_feasible),
            -1.0 if esp_value is None else esp_value,
            deadline_value,
            reliability_value,
            -energy_value,
            -response_value,
        )
    return (
        deadline_value,
        reliability_value,
        -energy_value,
        -response_value,
    )


def checkpoint_validation_seeds(training_cfg, default_seed):
    seeds = [int(training_cfg.get("eval_seed", default_seed))]
    secondary = training_cfg.get("eval_seed_secondary")
    if secondary is not None and int(secondary) not in seeds:
        seeds.append(int(secondary))
    return seeds


def combine_checkpoint_metrics(blocks):
    """Merge the validation blocks a checkpoint is ranked on into one metrics dict.

    Where the per-run samples survive (``attach_satisfaction_metrics``), the
    blocks are **pooled** and every constraint statistic is recomputed over the
    union.  Two disjoint 20-seed blocks then give a 40-seed estimate: the
    standard error drops by sqrt(2) and ESP resolves to 1/40 instead of 1/20.
    Combining the block *summaries* instead cannot do this -- ``min`` of two
    20-seed means is biased low by roughly ``sigma/sqrt(20) * 0.56`` and no
    combination of two ESP values recovers the pooled ESP.

    Protection against selecting on noise does not come from this function: the
    accepted numbers are reported on a third block that selection never touches.

    Energy and response time are averaged, never min-ed: once the constraint
    gate is passed the ranking is a straight energy minimization
    (see :func:`metric_rank`), so a pessimistic energy estimate would distort
    the objective rather than guard the constraint.
    """
    if not blocks:
        raise ValueError("At least one validation block is required.")
    result = dict(blocks[0])
    pooled = {}
    for metric_key, _, _ in SATISFACTION_METRICS:
        samples = []
        complete = True
        for row in blocks:
            values = row.get(satisfaction_sample_key(metric_key))
            if values is None:
                complete = False
                break
            samples.extend(float(value) for value in values)
        if complete and samples:
            pooled[metric_key] = samples

    for metric_key in ("deadline_success_rate", "reliability_success_rate"):
        if metric_key in pooled:
            result[metric_key] = float(np.mean(pooled[metric_key]))
        else:
            # Pre-pooling blocks: fall back to the conservative worst block.
            result[metric_key] = min(float(row[metric_key]) for row in blocks)

    for name in (
        "total_energy",
        "average_workflow_response_time",
        "avg_workflow_response_time",
        "average_response_time",
        "episode_reward",
    ):
        values = [float(row[name]) for row in blocks if row.get(name) is not None]
        if values:
            result[name] = float(np.mean(values))

    if pooled:
        for metric_key, prefix, thresholds in SATISFACTION_METRICS:
            if metric_key not in pooled:
                continue
            result[satisfaction_sample_key(metric_key)] = pooled[metric_key]
            stats = satisfaction_statistics(pooled[metric_key], thresholds=thresholds)
            for value in thresholds:
                name = f"{prefix}_esp_{value:.2f}".replace(".", "p")
                result[name] = stats["esp"][f"{value:.2f}"]
            result[f"{prefix}_cvar25"] = stats["cvar"]
            result[f"{prefix}_min"] = stats["min"]
            result[f"{prefix}_std"] = stats["std"]
    else:
        # ``result`` started as a copy of the first block, so every satisfaction
        # column would otherwise report block 0 alone under a combined label.
        for name in satisfaction_field_names():
            values = [float(row[name]) for row in blocks if row.get(name) is not None]
            if values:
                result[name] = float(
                    max(values) if name.endswith("_std") else min(values)
                )
    # Checkpoint feasibility is deliberately block-wise: a good first block
    # must not hide a weak second block in the pooled ESP.
    for name in satisfaction_field_names():
        if "_esp_" not in name:
            continue
        values = [float(row[name]) for row in blocks if row.get(name) is not None]
        if values:
            result[f"{name}_min_block"] = min(values)
    result["validation_block_count"] = len(blocks)
    return result


def write_checkpoint_selection(path, policy, best_metrics, best_rank, checkpoint_name):
    rank = None if best_rank is None else [float(value) for value in best_rank]
    feasible = bool(rank is not None and rank[0] >= 1.0)
    payload = {
        "policy": policy,
        "feasible": feasible,
        "selected_checkpoint": checkpoint_name if feasible else None,
        "best_observed_checkpoint": checkpoint_name if rank is not None else None,
        "rank": rank,
        "metrics": best_metrics,
    }
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def validate_training_config(training_cfg):
    if str(training_cfg.get("training_mode", "joint")) != "joint":
        raise ValueError("Only training_mode='joint' is supported.")
    if str(training_cfg.get("update_mode", "episode_end")) != "episode_end":
        raise ValueError("Only update_mode='episode_end' is supported.")
    forbidden = [
        "low_pretrain_episodes",
        "high_train_episodes",
        "joint_finetune_episodes",
        "joint_low_update_interval",
        "joint_low_learning_rate_scale",
    ]
    present = [key for key in forbidden if key in training_cfg]
    if present:
        raise ValueError(f"Staged training keys are not supported: {present}")




def append_history_row(output_dir, stem, row):
    with (output_dir / f"{stem}.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def initialize_history_files(output_dir, stem):
    (output_dir / f"{stem}.jsonl").write_text("", encoding="utf-8")
    (output_dir / f"{stem}.json").write_text("[]\n", encoding="utf-8")


def write_history_files(output_dir, stem, rows, *, rewrite_json=True, rewrite_jsonl=True):
    if rewrite_json:
        with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as file:
            json.dump(rows, file, ensure_ascii=False, indent=2)
    if rewrite_jsonl:
        with (output_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")




def format_metrics(metrics):
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def format_satisfaction_summary(row, *, include_mean=True):
    """One-line ESP/CVaR digest of an evaluation row, for console logs.

    The block mean alone cannot tell a run that misses the Deadline on a few
    congested instances from one that misses a little everywhere, which is the
    difference ESP is there to expose.
    """
    parts = []
    fields = [
        ("deadline_esp_0p90", "ESP@0.90"),
        ("deadline_esp_0p92", "ESP@0.92"),
        ("deadline_esp_0p95", "ESP@0.95"),
        ("deadline_cvar25", "CVaR25"),
        ("deadline_min", "min"),
        ("reliability_esp_1p00", "R_ESP@1.00"),
    ]
    if include_mean:
        fields.insert(0, ("deadline_success_rate", "SR"))
    else:
        fields = [
            ("deadline_esp_0p90", "ESP@0.90"),
            ("reliability_esp_1p00", "R_ESP@1.00"),
        ]
    for key, title in fields:
        value = row.get(key)
        if value is not None:
            parts.append(f"{title}={float(value):.4f}")
    return " ".join(parts)


def print_metrics_block(
    label,
    row,
    *,
    include_deadline_success_rate=True,
    satisfaction_target_only=False,
):
    replay_zscore_high_reward = row.get("high_reward_mode") == "replay_zscore"
    header = [
        ("episode", "ep"),
        ("training_mode", "mode"),
        ("update_mode", "update"),
        ("high_update_count", "high_epupd"),
        ("low_update_count", "low_epupd"),
        ("low_update_skipped_due_to_empty_buffer", "low_empty"),
        ("low_update_skipped_due_to_freeze", "low_frozen"),
    ]
    print(f"[{label}] " + "  ".join(_format_pair(row, key, title) for key, title in header if key in row))
    sections = [
        ("buffers", [
            ("high_buffer_size", "high"),
            ("low_buffer_size", "low"),
            ("low_candidate_count", "low_candidates"),
            ("low_buffer_acceptance_rate", "low_accept"),
        ]),
        ("reward", [
            ("mean_high_reward", "pen_mean" if replay_zscore_high_reward else "high_mean"),
            ("total_high_reward", "pen_total" if replay_zscore_high_reward else "high_total"),
            ("mean_low_reward", "low_mean"),
            ("total_low_reward", "low_total"),
        ]),
        ("high_parts", [
            ("mean_high_task_time_reward", "time_mean"),
            ("total_high_task_time_reward", "time_total"),
            ("mean_current_slack_ratio", "slack_mean"),
            ("min_current_slack_ratio", "slack_min"),
            ("mean_task_time_regret", "regret_mean"),
            ("mean_task_complete_time_ratio", "ratio_mean"),
            ("max_task_complete_time_ratio", "ratio_max"),
            ("mean_high_low_reward_bonus", "energy_mean"),
            ("total_high_low_reward_bonus", "energy_total"),
            ("mean_high_team_energy_reward", "team_mean"),
            ("total_high_team_energy_reward", "team_total"),
        ]),
        ("upper_raw", [
            ("mean_high_workflow_slack_ratio", "slack_mean"),
            ("min_high_workflow_slack_ratio", "slack_min"),
            ("mean_high_energy_saving_joules", "saved_mean_J"),
            ("total_high_energy_saving_joules", "saved_total_J"),
            ("high_deadline_crossing_count", "cross_count"),
            ("mean_high_deadline_violation", "violation"),
        ]),
        ("upper_deadline", [
            ("mean_normalized_makespan_increment", "delta_C_mean"),
            ("total_normalized_makespan_increment", "delta_C_total"),
            ("mean_high_deadline_violation", "V_mean"),
            ("mean_high_deadline_violation_delta", "delta_V_mean"),
            ("total_high_deadline_violation_delta", "delta_V_total"),
            ("high_deadline_crossing_count", "cross_count"),
        ]),
        ("sub_deadline", [
            ("mean_task_sub_deadline", "task_D_mean"),
            ("mean_task_finish_relative", "finish_mean"),
            ("mean_sub_deadline_constraint_cost", "high_cost_mean"),
            ("total_sub_deadline_constraint_cost", "high_cost_total"),
            ("sub_deadline_miss_count", "task_miss"),
            ("mean_low_sac_reference_sub_deadline_cost", "low_ref"),
            ("mean_low_sac_actual_sub_deadline_cost", "low_actual"),
            ("mean_low_sac_incremental_sub_deadline_cost", "low_delta"),
        ]),
        ("low_energy", [
            ("mean_low_energy_saving_ratio", "low_save_ratio"),
            ("mean_low_deadline_cost", "low_d_cost"),
        ]),
        ("low_sac", [
            ("total_low_sac_energy_saved", "saved_J"),
            ("low_sac_total_energy_saving_ratio", "save_ratio"),
            ("mean_low_sac_energy_saving_ratio", "mean_ratio"),
            ("total_low_sac_energy_reference", "ref_J"),
            ("total_low_sac_energy_actual", "actual_J"),
            ("mean_low_sac_time_increase_seconds", "time_inc_s"),
            ("mean_low_sac_time_increase", "time_ratio"),
            ("mean_low_sac_slack_time_cost", "slack_cost"),
            ("mean_low_sac_slack_before_seconds", "slack_s"),
            ("mean_low_sac_deadline_violation", "d_violation"),
            ("low_sac_deadline_crossing_count", "d_cross"),
            ("mean_low_sac_reliability_violation", "r_violation"),
            ("low_sac_reliability_crossing_count", "r_cross"),
        ]),
        ("constraints", [
            ("mean_deadline_cost", "deadline_mean"),
            ("total_deadline_cost", "deadline_total"),
            ("deadline_success_rate", "deadline_ok"),
            ("mean_high_deadline_penalty", "d_pen_mean"),
            ("total_high_deadline_penalty", "d_pen_total"),
            ("high_deadline_crossing_count", "d_cross"),
            ("mean_reliability_cost", "reliab_mean"),
            ("total_reliability_cost", "reliab_total"),
            ("reliability_success_rate", "reliab_ok"),
            ("mean_high_reliability_penalty", "rel_pen_mean"),
            ("total_high_reliability_penalty", "rel_pen_total"),
            ("reliability_crossing_count", "rel_cross"),
        ]),
        # Only evaluation rows carry these keys, so this section is absent from
        # per-episode training output and appears once per eval block.
        ("satisfaction", [
            ("deadline_esp_0p90", "ESP@0.90"),
            ("deadline_esp_0p92", "ESP@0.92"),
            ("deadline_esp_0p95", "ESP@0.95"),
            ("deadline_esp_1p00", "ESP@1.00"),
            ("deadline_cvar25", "D_CVaR25"),
            ("deadline_min", "D_min"),
            ("deadline_std", "D_std"),
            ("reliability_esp_1p00", "R_ESP@1.00"),
            ("reliability_min", "R_min"),
        ]),
        ("p3o_deadline", [
            ("high_deadline_raw_surrogate", "raw"),
            ("high_deadline_margin", "margin"),
            ("high_deadline_pre_relu", "pre_relu"),
            ("high_deadline_penalty", "penalty"),
            ("high_deadline_relu_active_rate", "active"),
        ]),
        ("p3o_reliab", [
            ("high_reliability_raw_surrogate", "raw"),
            ("high_reliability_margin", "margin"),
            ("high_reliability_pre_relu", "pre_relu"),
            ("high_reliability_penalty", "penalty"),
            ("high_reliability_relu_active_rate", "active"),
        ]),
        ("high_scale", [
            ("high_reward_adv_std", "r_adv_std"),
            ("high_deadline_adv_std", "d_adv_std"),
            ("high_reliability_adv_std", "rel_adv_std"),
            ("high_grad_norm_before_clip", "grad_pre"),
            ("high_grad_norm_after_clip", "grad_post"),
        ]),
        ("rho", [
            ("rho_mean", "mean"),
            ("rho_std", "std"),
            ("rho_safe_mean", "safe_mean"),
            ("rho_safe_at_one_rate", "safe_at_one"),
            ("forced_max_frequency_rate", "forced_max"),
            ("rho_floor_binding_rate", "floor_bind"),
        ]),
        ("low_deadline", [
            ("low_deadline_surrogate", "surrogate"),
            ("low_deadline_value_loss", "value_loss"),
            ("low_deadline_return_mean", "return"),
            ("low_deadline_adv_std", "adv_std"),
        ]),
        ("system", [
            ("total_energy", "energy"),
            ("primary_energy", "primary"),
            ("backup_energy", "backup"),
            ("average_workflow_response_time", "avg_response"),
            ("global_makespan", "global_ms"),
            ("global_makespan_workflow_id", "ms_wf"),
            ("global_makespan_task_id", "ms_task"),
            ("global_makespan_server_id", "ms_server"),
        ]),
        ("selection", [
            ("primary_selection_counts", "primary"),
            ("backup_selection_counts", "backup"),
        ]),
        ("shield_diag", [
            ("shield_relaxed_count", "primary_relaxed"),
            ("backup_shield_relaxed_count", "backup_relaxed"),
        ]),
    ]
    for name, pairs in sections:
        values = [
            _format_pair(row, key, title)
            for key, title in pairs
            if key in row
            and (include_deadline_success_rate or key != "deadline_success_rate")
            and (
                not satisfaction_target_only
                or name != "satisfaction"
                or key in {"deadline_esp_0p90", "reliability_esp_1p00"}
            )
        ]
        if values:
            print(f"  {name:<13} " + "  ".join(values))


def _format_pair(row, key, title):
    return f"{title:<15} {_format_value(row.get(key), key=key):>13}"


def _format_value(value, key=None):
    if key == "deadline_success_rate" and isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)
