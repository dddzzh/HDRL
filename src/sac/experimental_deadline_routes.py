"""Isolated lower-SAC routes; legacy learners and checkpoints are unchanged."""
from __future__ import annotations

import copy
import math

from src.sac.hierarchical_low_sac import LowLevelDVFSContinuousSACAgent
from src.D3QN.deadline_lagrangian import DeadlineLagrangianController


DENSE_RISK_COST_MODE = "dense_budget_share_v1"
BATCH_FAILURE_DUAL_MODE = "episode_deadline_failure"
DEADLINE_ROUTES = ("a_slack", "b_dual")


class BatchFailureLowDeadlineController(DeadlineLagrangianController):
    """Reuse the upper PI rule, exposing the lower trainer's metric names."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.constraint_metric != BATCH_FAILURE_DUAL_MODE:
            raise ValueError("Experimental lower Deadline dual requires episode_deadline_failure.")

    def update(self, workflow_misses, episode):
        misses = list(workflow_misses)
        row = super().update(misses, episode)
        metrics = {}
        for key, value in row.items():
            if key.startswith("lambda_deadline"):
                metrics["low_deadline_lambda" + key[len("lambda_deadline"):]] = value
            elif key.startswith("deadline_"):
                metrics["low_" + key] = value
        metrics.update(
            low_deadline_constraint_excess=row["deadline_excess"],
            low_deadline_constraint_batch_total=float(sum(misses)),
            low_deadline_constraint_batch_workflow_count=row["completed_workflow_count"],
            low_deadline_constraint_window_workflows=self.window_workflows,
        )
        return metrics

    def state_dict(self):
        return {**super().state_dict(), "low_deadline_dual_mode": BATCH_FAILURE_DUAL_MODE}

    def validate_state(self, state):
        if not isinstance(state, dict) or (
            state.get("low_deadline_dual_mode") != BATCH_FAILURE_DUAL_MODE
            or state.get("deadline_constraint_metric") != self.constraint_metric
            or state.get("deadline_constraint_statistic") != self.constraint_statistic
        ):
            raise ValueError("Lower Deadline checkpoint is not a matching batch-failure controller.")

    def load_state_dict(self, state):
        self.validate_state(state)
        super().load_state_dict(state)


class DenseRiskLowSACAgent(LowLevelDVFSContinuousSACAgent):
    """Keep the original SAC/base reward; replace only reliability replay cost."""

    def __init__(self, config=None, seed=42):
        config = copy.deepcopy(dict(config or {}))
        self.deadline_route = config.get("deadline_route")
        self.reliability_cost_mode = config.get("reliability_cost_mode")
        if self.deadline_route not in DEADLINE_ROUTES:
            raise ValueError(f"Unsupported experimental deadline_route: {self.deadline_route!r}.")
        if self.reliability_cost_mode != DENSE_RISK_COST_MODE:
            raise ValueError("Experimental lower SAC requires dense_budget_share_v1.")
        lagrangian = dict(config.get("lagrangian", {}))
        if not lagrangian.get("enabled", False):
            raise ValueError("Keep the lower Deadline controller enabled; route A fixes its value to zero.")
        time_weight = float(dict(config.get("reward", {})).get("slack_time_weight", 1.0))
        if self.deadline_route == "a_slack":
            if time_weight <= 0 or any(float(lagrangian.get(key, 0.0)) != 0.0 for key in (
                "deadline_lambda_init", "dual_learning_rate", "proportional_gain"
            )):
                raise ValueError("Route A requires positive slack_time weight and a permanently zero Deadline dual.")
        elif time_weight != 0.0:
            raise ValueError("Route B requires slack_time_weight=0.")
        # The legacy low constructor does not accept the episode-level metric.
        # Its temporary controller is replaced before any rollout or learning.
        base_config = copy.deepcopy(config)
        base_config["lagrangian"]["constraint_metric"] = "workflow_miss_rate"
        super().__init__(base_config, seed=seed)
        if self.deadline_route == "b_dual":
            self.deadline_lagrangian = BatchFailureLowDeadlineController.from_config(lagrangian)

    def remember_transition(self, state, action_rho, reward_components, next_state, done,
                            *, critic_state=None, next_critic_state=None):
        components = dict(reward_components)
        dense_cost = float(components["reliability_budget_share_spent"])
        if not math.isfinite(dense_cost) or dense_cost < 0.0:
            raise ValueError("Dense reliability budget share must be finite and non-negative.")
        components["reliability_violation"] = dense_cost
        return super().remember_transition(
            state, action_rho, components, next_state, done,
            critic_state=critic_state, next_critic_state=next_critic_state,
        )

    def state_dict(self, include_replay=False):
        return {
            **super().state_dict(include_replay=include_replay),
            "deadline_route": self.deadline_route,
            "reliability_cost_mode": self.reliability_cost_mode,
        }

    def load_state_dict(self, payload):
        if (payload.get("deadline_route") != self.deadline_route
                or payload.get("reliability_cost_mode") != self.reliability_cost_mode):
            raise ValueError("Experimental lower SAC checkpoint route/cost mismatch; no legacy warm start.")
        state = payload.get("deadline_lagrangian")
        if self.deadline_route == "b_dual":
            self.deadline_lagrangian.validate_state(state)
        elif not isinstance(state, dict):
            raise ValueError("Route A checkpoint is missing its fixed-zero controller.")
        if self.deadline_route == "a_slack" and any(
            float(state.get(key, float("nan"))) != 0.0
            for key in ("deadline_lambda", "deadline_lambda_integral")
        ):
            raise ValueError("Route A cannot restore a nonzero lower Deadline multiplier.")
        if self.reliability_lagrangian is not None and payload.get("reliability_lagrangian") is None:
            raise ValueError("Experimental lower SAC checkpoint is missing its reliability controller.")
        return super().load_state_dict(payload)
