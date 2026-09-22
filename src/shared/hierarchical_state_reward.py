from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.D3QN.state import HierarchicalStateEncoder, safe_float
from src.a2c.a2c import Observation


EPS = 1e-8
SHARED_STATE_VERSION = "d3qn_high_deadline_queue_v2"
BEST_NON_LAGRANGIAN_STATE_VERSION = "d3qn_high_580_best_non_lagrangian_v1"
SHARED_REWARD_VERSION = "hierarchical_deadline_dvfs_v1"
SHARED_UPPER_REWARD_VERSION = "upper_deadline_energy_per_workflow_v2"
FIXED_LAMBDA_UPPER_REWARD_VERSION = "upper_energy_fixed_lambda_increment_v1"
FIXED_LAMBDA_UPPER_MISS_REWARD_VERSION = (
    "upper_energy_fixed_lambda_increment_miss_v2"
)


def low_reliability_counterfactual_cost(
    env,
    workflow_state,
    task_id,
    reference_task_reliability,
    actual_task_reliability,
):
    """Reliability damage caused only by lowering the selected task's rho.

    Both candidates are projected to the final workflow log-reliability using
    the same already-scheduled tasks and the same remaining-task estimate.  A
    server-placement reliability deficit present at rho=1 is therefore not
    charged to the DVFS controller or to a joint-action baseline's frequency
    choice.
    """

    required = float(env._workflow_reliability_requirement(workflow_state))
    if required <= 0.0:
        return {
            "reference_reliability_violation": 0.0,
            "actual_reliability_violation": 0.0,
            "reliability_violation_delta": 0.0,
            "reliability_crossed": 0.0,
        }
    log_required = math.log(max(required, EPS))
    risk_budget = max(-log_required, EPS)

    def violation(candidate):
        log_projected = float(
            env._estimated_workflow_log_reliability(
                workflow_state,
                int(task_id),
                float(candidate),
            )
        )
        return max(0.0, log_required - log_projected) / risk_budget

    reference = violation(reference_task_reliability)
    actual = violation(actual_task_reliability)
    delta = max(0.0, actual - reference)
    return {
        "reference_reliability_violation": float(reference),
        "actual_reliability_violation": float(actual),
        "reliability_violation_delta": float(delta),
        "reliability_crossed": float(actual > EPS and reference <= EPS),
    }


def build_shared_state_encoder(state_config=None):
    """Return the requested D3QN high-state encoder for fair comparisons."""
    config = dict(state_config or {})
    version = str(
        config.get("version", config.get("variant", SHARED_STATE_VERSION))
    ).strip().lower()
    if version in {
        BEST_NON_LAGRANGIAN_STATE_VERSION,
        "best_non_lagrangian_580",
        "legacy_580",
    }:
        return BestNonLagrangianStateEncoder(config)
    if version not in {
        SHARED_STATE_VERSION,
        "current",
        "d3qn_high_347",
        "d3qn_high_347_v1",
    }:
        raise ValueError(f"Unsupported shared high-state version: {version!r}.")
    encoder = HierarchicalStateEncoder(config)
    encoder.version = SHARED_STATE_VERSION
    return encoder


def shared_state_version(encoder):
    return str(getattr(encoder, "version", SHARED_STATE_VERSION))


class BestNonLagrangianStateEncoder(HierarchicalStateEncoder):
    """Exact 580-D upper state used by the validated 2026-08-06 model.

    This preserves the old three task features and twenty-one task/server
    features without reverting the current D3QN encoder.  Comparison agents
    can therefore opt into the validated state explicitly while newer models
    keep using the reduced 347-D state.
    """

    task_feature_dim = 3
    server_feature_dim = 21
    global_feature_dim = 5
    version = BEST_NON_LAGRANGIAN_STATE_VERSION

    @property
    def high_state_dim(self):
        return (
            self.task_feature_dim
            + self.max_servers * self.server_feature_dim
            + self.max_servers
            + self.global_feature_dim
        )

    def encode_high_state(self, env_state):
        ready_tasks = tuple(env_state.get("ready_tasks", []))
        servers = tuple(env_state.get("servers", []))
        current_time = safe_float(env_state.get("time", 0.0))
        context = env_state.get("decision_context", {})
        metrics_by_id = (
            context.get("server_safety_metrics", {})
            if isinstance(context, dict)
            else {}
        )
        max_server_freq_hz = self.max_server_freq_hz(servers)

        task_features = np.zeros(
            (self.max_ready_tasks, self.task_feature_dim), dtype=np.float32
        )
        task_mask = np.zeros(self.max_ready_tasks, dtype=bool)
        for index, task in enumerate(ready_tasks[: self.max_ready_tasks]):
            task_mask[index] = True
            task_features[index] = self._best_task_features(task, current_time)

        current_task = ready_tasks[0] if ready_tasks else {}
        server_features = np.zeros(
            (self.max_servers, self.server_feature_dim), dtype=np.float32
        )
        server_mask = np.zeros(self.max_servers, dtype=bool)
        for index, server in enumerate(servers[: self.max_servers]):
            server_mask[index] = True
            server_features[index] = self._best_server_features(
                server,
                current_task,
                current_time,
                max_server_freq_hz,
                safety_metrics=metrics_by_id.get(str(server.get("id")), {}),
            )

        global_features = self.high_global_features(
            ready_tasks, servers, current_time
        )
        current_task_features = (
            task_features[0]
            if ready_tasks
            else np.zeros(self.task_feature_dim, dtype=np.float32)
        )
        state_vector = np.concatenate(
            (
                current_task_features,
                server_features.reshape(-1),
                server_mask.astype(np.float32),
                global_features,
            )
        ).astype(np.float32)
        return Observation(
            ready_task_features=task_features,
            ready_task_mask=task_mask,
            server_features=server_features,
            server_mask=server_mask,
            global_features=global_features,
            ready_tasks=ready_tasks[: self.max_ready_tasks],
            servers=servers[: self.max_servers],
            rho_levels=(),
            vector_override=state_vector,
        )

    def _best_task_features(self, task, current_time):
        workload = safe_float(task.get("workload_cycles", 0.0))
        deadline = self.workflow_deadline_scale(task, current_time)
        absolute_deadline = safe_float(
            task.get("absolute_deadline", current_time + deadline)
        )
        remaining_deadline = absolute_deadline - current_time
        workflow_task_count = safe_float(
            task.get("workflow_task_count", 1.0), 1.0
        )
        return np.asarray(
            (
                np.tanh(workload / 1e10),
                np.clip(remaining_deadline / deadline, -10.0, 10.0),
                workflow_task_count / max(self.max_ready_tasks, 1),
            ),
            dtype=np.float32,
        )

    def _best_server_features(
        self,
        server,
        task,
        current_time,
        max_server_freq_hz,
        safety_metrics=None,
    ):
        metrics = self.high_server_pair_metrics(
            server,
            task,
            current_time,
            max_server_freq_hz,
            safety_metrics=safety_metrics,
        )
        deadline = max(float(metrics["workflow_relative_deadline"]), EPS)
        return np.asarray(
            (
                self.norm_deadline_time(metrics["wait_time"], deadline),
                self.norm_count(metrics["queue_count"]),
                self.norm_deadline_time(metrics["queue_load"], deadline),
                self.norm_deadline_time(
                    metrics["queued_processing_time"], deadline
                ),
                self.norm_deadline_time(
                    metrics["estimated_execution_time"], deadline
                ),
                self.norm_deadline_time(
                    metrics["estimated_ast"] - current_time, deadline
                ),
                self.norm_deadline_time(
                    metrics["estimated_finish_time"]
                    - metrics["workflow_arrival_time"],
                    deadline,
                ),
                np.clip(metrics["slack_after_assign"] / deadline, -10.0, 10.0),
                np.clip(metrics["safe_execution_budget"] / deadline, -10.0, 10.0),
                self.norm_deadline_time(metrics["communication_time"], deadline),
                np.clip(metrics["rho_safe"], 0.0, 1.0),
                1.0 if bool(metrics["low_action_feasible"]) else 0.0,
                np.tanh(metrics["bandwidth_mbps"] / 10000.0),
                metrics["max_freq_hz"] / max(max_server_freq_hz, EPS),
                np.tanh(metrics["total_power"] / 250.0),
                np.tanh(metrics["failure_rate"] / 1e-3),
                self.norm_deadline_time(metrics["wait_time_ema"], deadline),
                self.norm_deadline_time(
                    metrics["queued_processing_time_ema"], deadline
                ),
                self.norm_count(metrics["queue_count_ema"]),
                self.norm_count(metrics["recent_primary_assignments"]),
                self.norm_count(metrics["recent_backup_assignments"]),
            ),
            dtype=np.float32,
        )


def deadline_violation_with_unit_jump(makespan, deadline):
    makespan = float(makespan)
    deadline = max(float(deadline), EPS)
    if makespan <= deadline + EPS:
        return 0.0
    return 1.0 + (makespan - deadline) / deadline


def normalized_sub_deadline_violation(
    *,
    finish_time,
    workflow_arrival_time,
    sub_deadline_relative,
    workflow_task_count,
):
    """Return a dimensionless per-task local Deadline constraint cost.

    The cost uses the same ``1 / N_w`` workflow-size normalization as the
    upper energy reward.  A late task contributes at least ``1 / N_w`` and an
    on-time task contributes zero.
    """

    finish_relative = max(
        0.0, float(finish_time) - float(workflow_arrival_time)
    )
    sub_deadline = max(float(sub_deadline_relative), EPS)
    task_count = max(float(workflow_task_count), 1.0)
    if finish_relative <= sub_deadline + EPS:
        return 0.0
    severity = (finish_relative - sub_deadline) / sub_deadline
    return float((1.0 + severity) / task_count)


def compose_upper_deadline_energy_reward(
    *,
    normalized_makespan_increment,
    deadline_violation_delta,
    energy_reference,
    energy_actual,
    workflow_task_count,
    deadline_violation_weight=1.0,
    energy_weight=1.0,
):
    """Compose the common upper reward used by D3QN and OOAC.

    The caller chooses the energy reference. D3QN supplies the cross-server
    rho=1 median, while OOAC supplies rho=1 energy on its selected server.
    """
    energy_reference = max(float(energy_reference), EPS)
    workflow_task_count = max(float(workflow_task_count), 1.0)
    energy_saving = energy_reference - float(energy_actual)
    energy_saving_ratio = energy_saving / energy_reference
    time_reward = -max(0.0, float(normalized_makespan_increment))
    energy_reward = (
        float(energy_weight) * energy_saving_ratio / workflow_task_count
    )
    deadline_penalty = (
        float(deadline_violation_weight)
        * max(0.0, float(deadline_violation_delta))
    )
    return {
        "version": SHARED_UPPER_REWARD_VERSION,
        "reward": float(time_reward + energy_reward - deadline_penalty),
        "time_reward": float(time_reward),
        "energy_reward": float(energy_reward),
        "energy_saving": float(energy_saving),
        "energy_saving_ratio": float(energy_saving_ratio),
        "energy_reference": float(energy_reference),
        "energy_actual": float(energy_actual),
        "workflow_task_count": float(workflow_task_count),
        "deadline_violation_delta": float(
            max(0.0, float(deadline_violation_delta))
        ),
        "deadline_penalty": float(deadline_penalty),
    }


def compose_fixed_lambda_upper_reward(
    *,
    normalized_makespan_increment,
    deadline_violation_delta=0.0,
    energy_scale,
    energy_actual,
    workflow_task_count,
    energy_weight=1.0,
    fixed_deadline_lambda=10.0,
    fixed_deadline_miss_penalty=0.0,
):
    """Return the common baseline reward with a fixed Deadline multiplier.

    r_t = w_E * ((E_scale - E_actual) / E_scale) / N_w
          - lambda_H * (Delta C_t / D_w)
          - P_miss * Delta V_t

    ``Delta V_t`` uses the caller's unit-jump Deadline violation increment, so
    crossing from feasible to infeasible has an explicit fixed cost.  This
    intentionally has no replay normalization or adaptive multiplier update.
    """
    energy_scale = max(float(energy_scale), EPS)
    energy_actual = float(energy_actual)
    workflow_task_count = max(float(workflow_task_count), 1.0)
    energy_weight = float(energy_weight)
    fixed_deadline_lambda = float(fixed_deadline_lambda)
    fixed_deadline_miss_penalty = float(fixed_deadline_miss_penalty)
    if (
        energy_weight < 0.0
        or fixed_deadline_lambda < 0.0
        or fixed_deadline_miss_penalty < 0.0
    ):
        raise ValueError("Fixed Deadline coefficients must be non-negative.")

    normalized_makespan_increment = max(
        0.0, float(normalized_makespan_increment)
    )
    energy_saving = energy_scale - energy_actual
    energy_saving_ratio = energy_saving / energy_scale
    energy_reward = (
        energy_weight * energy_saving_ratio / workflow_task_count
    )
    deadline_violation_delta = max(0.0, float(deadline_violation_delta))
    incremental_deadline_cost = (
        fixed_deadline_lambda * normalized_makespan_increment
    )
    deadline_miss_cost = (
        fixed_deadline_miss_penalty * deadline_violation_delta
    )
    deadline_cost = incremental_deadline_cost + deadline_miss_cost
    deadline_reward = -deadline_cost
    return {
        "version": (
            FIXED_LAMBDA_UPPER_MISS_REWARD_VERSION
            if fixed_deadline_miss_penalty > 0.0
            else FIXED_LAMBDA_UPPER_REWARD_VERSION
        ),
        "reward": float(energy_reward + deadline_reward),
        "energy_reward": float(energy_reward),
        "energy_weight": float(energy_weight),
        "deadline_reward": float(deadline_reward),
        "deadline_cost": float(deadline_cost),
        "fixed_deadline_lambda": float(fixed_deadline_lambda),
        "fixed_deadline_miss_penalty": float(fixed_deadline_miss_penalty),
        "deadline_violation_delta": float(deadline_violation_delta),
        "incremental_deadline_cost": float(incremental_deadline_cost),
        "deadline_miss_cost": float(deadline_miss_cost),
        "normalized_makespan_increment": float(
            normalized_makespan_increment
        ),
        "energy_saving": float(energy_saving),
        "energy_saving_ratio": float(energy_saving_ratio),
        "energy_scale": float(energy_scale),
        "energy_actual": float(energy_actual),
        "workflow_task_count": float(workflow_task_count),
    }


def compute_dvfs_reward_components(
    *,
    deadline,
    partial_makespan_before,
    workflow_arrival_time,
    reference_aft,
    actual_aft,
    reference_energy,
    actual_energy,
    slack_floor_ratio=0.05,
):
    """Compute action-local DVFS components shared by SAC and joint baselines."""
    deadline = max(float(deadline), EPS)
    partial_makespan_before = max(0.0, float(partial_makespan_before))
    workflow_arrival_time = float(workflow_arrival_time)
    reference_aft = float(reference_aft)
    actual_aft = float(actual_aft)
    reference_energy = max(float(reference_energy), EPS)
    actual_energy = float(actual_energy)

    energy_saving = reference_energy - actual_energy
    time_increase_seconds = max(0.0, actual_aft - reference_aft)
    slack_before_seconds = deadline - partial_makespan_before
    slack_floor_seconds = max(float(slack_floor_ratio), EPS) * deadline
    effective_slack_seconds = max(slack_before_seconds, slack_floor_seconds)
    slack_time_cost = time_increase_seconds / effective_slack_seconds

    reference_partial_makespan = max(
        partial_makespan_before,
        max(0.0, reference_aft - workflow_arrival_time),
    )
    current_partial_makespan = max(
        partial_makespan_before,
        max(0.0, actual_aft - workflow_arrival_time),
    )
    workflow_makespan_delay_seconds = max(
        0.0, current_partial_makespan - reference_partial_makespan
    )
    previous_violation = deadline_violation_with_unit_jump(
        partial_makespan_before, deadline
    )
    current_violation = deadline_violation_with_unit_jump(
        current_partial_makespan, deadline
    )
    violation_delta = max(0.0, current_violation - previous_violation)
    reference_violation = deadline_violation_with_unit_jump(
        reference_partial_makespan, deadline
    )
    # Attribute only the extra Deadline damage caused by lowering rho on the
    # already-selected server.  If rho=1 also misses, SAC is not charged for
    # the upper scheduler's placement error.
    dvfs_violation_delta = max(0.0, current_violation - reference_violation)

    return {
        "energy_saving": float(energy_saving),
        "energy_reference": float(reference_energy),
        "energy_actual": float(actual_energy),
        "energy_saving_ratio": float(energy_saving / reference_energy),
        "slack_time_cost": float(slack_time_cost),
        "time_increase_seconds": float(time_increase_seconds),
        "time_increase": float(time_increase_seconds / deadline),
        "slack_before_seconds": float(slack_before_seconds),
        "slack_floor_seconds": float(slack_floor_seconds),
        "effective_slack_seconds": float(effective_slack_seconds),
        "deadline_violation": float(violation_delta),
        "previous_deadline_violation": float(previous_violation),
        "current_deadline_violation": float(current_violation),
        "deadline_crossed": float(
            current_violation > EPS and previous_violation <= EPS
        ),
        "deadline_constraint_cost": float(dvfs_violation_delta),
        "deadline_constraint_crossed": float(
            current_violation > EPS and reference_violation <= EPS
        ),
        "reference_partial_makespan_after": float(reference_partial_makespan),
        "reference_deadline_violation": float(reference_violation),
        "workflow_partial_makespan_before": float(partial_makespan_before),
        "workflow_partial_makespan_after": float(current_partial_makespan),
        "workflow_makespan_delay_seconds": float(
            workflow_makespan_delay_seconds
        ),
    }


@dataclass(frozen=True)
class SharedRewardStatistics:
    energy_count: int = 0
    energy_mean: float = 0.0
    energy_std: float = 0.05
    slack_time_count: int = 0
    slack_time_mean: float = 0.0
    slack_time_std: float = 0.005

    @classmethod
    def from_mapping(cls, values=None):
        values = dict(values or {})
        energy = dict(values.get("energy", {}))
        slack_time = dict(values.get("slack_time", {}))
        return cls(
            energy_count=int(energy.get("count", 0)),
            energy_mean=float(energy.get("mean", 0.0)),
            energy_std=float(energy.get("std", 0.05)),
            slack_time_count=int(slack_time.get("count", 0)),
            slack_time_mean=float(slack_time.get("mean", 0.0)),
            slack_time_std=float(slack_time.get("std", 0.005)),
        )

    def as_dict(self):
        return {
            "energy": {
                "count": int(self.energy_count),
                "mean": float(self.energy_mean),
                "std": float(self.energy_std),
            },
            "slack_time": {
                "count": int(self.slack_time_count),
                "mean": float(self.slack_time_mean),
                "std": float(self.slack_time_std),
            },
        }


class SharedHierarchicalReward:
    """Common reward adapter for agents that jointly choose server and DVFS."""

    def __init__(self, config=None):
        config = dict(config or {})
        self.version = SHARED_REWARD_VERSION
        self.joint_low_reward_weight = float(
            config.get("joint_low_reward_weight", 0.1)
        )
        self.low_deadline_violation_weight = float(
            config.get("low_deadline_violation_weight", 5.0)
        )
        self.low_reliability_violation_weight = float(
            config.get("low_reliability_violation_weight", 0.0)
        )
        self.slack_floor_ratio = float(config.get("slack_floor_ratio", 0.05))
        self.component_clip = float(config.get("component_clip", 3.0))
        self.energy_std_floor = float(config.get("energy_std_floor", 0.05))
        self.slack_time_std_floor = float(
            config.get("slack_time_std_floor", 0.005)
        )
        values = (
            self.joint_low_reward_weight,
            self.low_deadline_violation_weight,
            self.low_reliability_violation_weight,
            self.slack_floor_ratio,
            self.component_clip,
            self.energy_std_floor,
            self.slack_time_std_floor,
        )
        if any(value < 0.0 for value in values[:3]):
            raise ValueError("Shared reward weights must be non-negative.")
        if any(value <= 0.0 for value in values[3:]):
            raise ValueError("Shared reward floors and clipping must be positive.")

    def compose(
        self,
        high_reward,
        *,
        energy_saving,
        slack_time_cost,
        deadline_violation_delta,
        reliability_violation_delta=0.0,
        statistics=None,
    ):
        statistics = SharedRewardStatistics.from_mapping(statistics)
        energy_zscore = self._zscore(
            energy_saving,
            statistics.energy_count,
            statistics.energy_mean,
            max(statistics.energy_std, self.energy_std_floor),
        )
        slack_time_zscore = self._zscore(
            slack_time_cost,
            statistics.slack_time_count,
            statistics.slack_time_mean,
            max(statistics.slack_time_std, self.slack_time_std_floor),
        )
        low_deadline_penalty = (
            self.low_deadline_violation_weight
            * max(0.0, float(deadline_violation_delta))
        )
        low_reliability_penalty = (
            self.low_reliability_violation_weight
            * max(0.0, float(reliability_violation_delta))
        )
        low_reward = (
            energy_zscore
            - slack_time_zscore
            - low_deadline_penalty
            - low_reliability_penalty
        )
        weighted_low_reward = self.joint_low_reward_weight * low_reward
        return {
            "reward": float(high_reward) + float(weighted_low_reward),
            "high_reward": float(high_reward),
            "low_reward": float(low_reward),
            "weighted_low_reward": float(weighted_low_reward),
            "energy_saving": float(energy_saving),
            "energy_zscore": float(energy_zscore),
            "energy_reward": float(
                self.joint_low_reward_weight * energy_zscore
            ),
            "slack_time_cost": float(slack_time_cost),
            "slack_time_zscore": float(slack_time_zscore),
            "slack_time_reward": float(
                -self.joint_low_reward_weight * slack_time_zscore
            ),
            "deadline_violation_delta": float(deadline_violation_delta),
            "low_deadline_penalty": float(low_deadline_penalty),
            "weighted_low_deadline_penalty": float(
                self.joint_low_reward_weight * low_deadline_penalty
            ),
            "reliability_violation_delta": float(reliability_violation_delta),
            "low_reliability_penalty": float(low_reliability_penalty),
            "weighted_low_reliability_penalty": float(
                self.joint_low_reward_weight * low_reliability_penalty
            ),
        }

    def _zscore(self, value, count, mean, std):
        if int(count) < 2:
            return 0.0
        normalized = (float(value) - float(mean)) / max(float(std), EPS)
        return float(np.clip(normalized, -self.component_clip, self.component_clip))
