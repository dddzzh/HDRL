from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping

import numpy as np


class LowReliabilityLagrangianController:
    """Workflow-level multiplier for the low-level reliability constraint."""

    def __init__(self, lambda_init=1.0, dual_lr=0.001, budget=0.0,
                 lambda_max=100.0, warmup_episodes=0, window_workflows=100):
        self.lambda_init = float(lambda_init)
        self.dual_lr = float(dual_lr)
        self.budget = float(budget)
        self.lambda_max = float(lambda_max)
        self.warmup_episodes = int(warmup_episodes)
        self.window_workflows = int(window_workflows)
        if not 0.0 <= self.lambda_init <= self.lambda_max:
            raise ValueError("Reliability lambda_init must be in [0, lambda_max].")
        if self.dual_lr < 0.0 or self.budget < 0.0 or self.lambda_max <= 0.0:
            raise ValueError("Invalid reliability Lagrangian configuration.")
        if self.warmup_episodes < 0 or self.window_workflows <= 0:
            raise ValueError("Invalid reliability Lagrangian window configuration.")
        self.lambda_reliability = self.lambda_init
        self.workflow_cost_window = deque(maxlen=self.window_workflows)

    @classmethod
    def from_config(cls, config: Mapping | None):
        config = dict(config or {})
        if str(config.get("update_interval", "episode_end")).lower() != "episode_end":
            raise ValueError("Reliability Lagrangian supports update_interval='episode_end'.")
        return cls(
            lambda_init=config.get("reliability_lambda_init", 1.0),
            dual_lr=config.get("reliability_dual_learning_rate", 0.001),
            budget=config.get("reliability_budget", 0.0),
            lambda_max=config.get("reliability_lambda_max", 100.0),
            warmup_episodes=config.get("reliability_warmup_episodes", 0),
            window_workflows=config.get("reliability_window_workflows", 100),
        )

    @property
    def value(self):
        return float(self.lambda_reliability)

    def update(self, workflow_costs: Iterable[float], episode: int):
        costs = np.asarray(list(workflow_costs), dtype=np.float64)
        if costs.ndim != 1 or (costs.size and
                               (not np.isfinite(costs).all() or np.any(costs < 0.0))):
            raise ValueError("Reliability workflow costs must be finite and non-negative.")
        self.workflow_cost_window.extend(float(x) for x in costs)
        window = np.asarray(self.workflow_cost_window, dtype=np.float64)
        before = self.value
        batch_mean = float(np.mean(costs)) if costs.size else 0.0
        mean_cost = float(np.mean(window)) if window.size else 0.0
        warmup = int(episode) < self.warmup_episodes
        updated = bool(window.size and not warmup)
        if updated:
            self.lambda_reliability = float(np.clip(
                before + self.dual_lr * (mean_cost - self.budget),
                0.0, self.lambda_max,
            ))
        return {
            "low_reliability_lambda": self.value,
            "low_reliability_lambda_before_update": before,
            "low_reliability_lambda_after_update": self.value,
            "low_reliability_constraint_budget": self.budget,
            "low_reliability_constraint_mean": mean_cost,
            "low_reliability_constraint_excess": mean_cost - self.budget if window.size else 0.0,
            "low_reliability_constraint_batch_mean": batch_mean,
            "low_reliability_constraint_batch_total": float(np.sum(costs)),
            "low_reliability_constraint_workflow_count": int(window.size),
            "low_reliability_constraint_batch_workflow_count": int(costs.size),
            "low_reliability_dual_updated": float(updated),
            "low_reliability_dual_warmup_active": float(warmup),
        }

    def state_dict(self):
        return {
            "reliability_lambda": self.value,
            "reliability_lambda_init": self.lambda_init,
            "reliability_dual_learning_rate": self.dual_lr,
            "reliability_budget": self.budget,
            "reliability_lambda_max": self.lambda_max,
            "reliability_warmup_episodes": self.warmup_episodes,
            "reliability_window_workflows": self.window_workflows,
            "workflow_cost_window": list(self.workflow_cost_window),
            "update_interval": "episode_end",
        }

    def load_state_dict(self, state):
        state = dict(state or {})
        value = float(state.get("reliability_lambda", self.lambda_init))
        if not np.isfinite(value):
            raise ValueError("Restored reliability lambda must be finite.")
        self.lambda_reliability = float(np.clip(value, 0.0, self.lambda_max))
        costs = np.asarray(state.get("workflow_cost_window", []), dtype=np.float64)
        if costs.ndim != 1 or (costs.size and
                               (not np.isfinite(costs).all() or np.any(costs < 0.0))):
            raise ValueError("Restored reliability workflow costs are invalid.")
        self.workflow_cost_window = deque(
            (float(x) for x in costs[-self.window_workflows:]),
            maxlen=self.window_workflows,
        )
