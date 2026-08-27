from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping

import numpy as np


class LowDeadlineLagrangianController:
    """Multiplier driven by a rolling workflow-level Deadline cost."""

    def __init__(
        self,
        lambda_init,
        dual_lr,
        budget=0.0,
        lambda_max=100.0,
        warmup_episodes=0,
        window_workflows=100,
        constraint_metric="workflow_miss_rate",
    ):
        self.lambda_init = float(lambda_init)
        self.dual_lr = float(dual_lr)
        self.budget = float(budget)
        self.lambda_max = float(lambda_max)
        self.warmup_episodes = int(warmup_episodes)
        self.window_workflows = int(window_workflows)
        self.constraint_metric = str(constraint_metric).strip().lower()
        if self.constraint_metric not in {
            "workflow_miss_rate",
            "sub_deadline_violation",
        }:
            raise ValueError(
                "low constraint_metric must be 'workflow_miss_rate' or "
                "'sub_deadline_violation'"
            )
        if not 0.0 <= self.lambda_init <= self.lambda_max:
            raise ValueError("low Deadline lambda_init must be in [0, lambda_max].")
        if self.dual_lr < 0.0:
            raise ValueError("low Deadline dual_lr must be non-negative.")
        if self.budget < 0.0 or (
            self.constraint_metric == "workflow_miss_rate" and self.budget > 1.0
        ):
            raise ValueError("low Deadline constraint budget is invalid.")
        if self.lambda_max <= 0.0:
            raise ValueError("low Deadline lambda_max must be positive.")
        if self.warmup_episodes < 0:
            raise ValueError("low Deadline warmup_episodes must be non-negative.")
        if self.window_workflows <= 0:
            raise ValueError("low Deadline window_workflows must be positive.")
        self.lambda_deadline = self.lambda_init
        self.workflow_cost_window = deque(maxlen=self.window_workflows)

    @classmethod
    def from_config(cls, config: Mapping | None):
        config = dict(config or {})
        update_interval = str(
            config.get("update_interval", "episode_end")
        ).strip().lower()
        if update_interval != "episode_end":
            raise ValueError(
                "Low Deadline Lagrangian currently supports only "
                "update_interval='episode_end'."
            )
        return cls(
            lambda_init=config.get("deadline_lambda_init", 10.0),
            dual_lr=config.get("dual_learning_rate", 1e-2),
            budget=config.get("deadline_budget", 0.0),
            lambda_max=config.get("lambda_max", 100.0),
            warmup_episodes=config.get("warmup_episodes", 50),
            window_workflows=config.get("window_workflows", 100),
            constraint_metric=config.get(
                "constraint_metric", "workflow_miss_rate"
            ),
        )

    @property
    def value(self):
        return float(self.lambda_deadline)

    def update(self, workflow_costs: Iterable[float], episode: int):
        costs = np.asarray(list(workflow_costs), dtype=np.float64)
        if costs.ndim != 1:
            raise ValueError("low workflow Deadline costs must be one-dimensional.")
        if costs.size > 0 and (
            not np.isfinite(costs).all()
            or np.any(costs < 0.0)
            or (
                self.constraint_metric == "workflow_miss_rate"
                and np.any(costs > 1.0)
            )
        ):
            raise ValueError(
                "low workflow Deadline costs are incompatible with the "
                "configured constraint metric."
            )

        self.workflow_cost_window.extend(float(value) for value in costs)
        window_costs = np.asarray(
            self.workflow_cost_window, dtype=np.float64
        )
        lambda_before = self.value
        batch_mean = float(np.mean(costs)) if costs.size else 0.0
        mean_cost = float(np.mean(window_costs)) if window_costs.size else 0.0
        warmup_active = int(episode) < self.warmup_episodes
        updated = bool(window_costs.size and not warmup_active)
        if updated:
            self.lambda_deadline = float(
                np.clip(
                    lambda_before + self.dual_lr * (mean_cost - self.budget),
                    0.0,
                    self.lambda_max,
                )
            )
        return {
            "low_deadline_lambda": self.value,
            "low_deadline_lambda_before_update": float(lambda_before),
            "low_deadline_lambda_after_update": self.value,
            "low_deadline_constraint_budget": float(self.budget),
            "low_deadline_constraint_metric": self.constraint_metric,
            "low_deadline_constraint_mean": mean_cost,
            "low_deadline_constraint_total": float(np.sum(window_costs)),
            "low_deadline_constraint_excess": (
                mean_cost - self.budget if window_costs.size else 0.0
            ),
            "low_deadline_constraint_workflow_count": int(window_costs.size),
            "low_deadline_constraint_batch_mean": batch_mean,
            "low_deadline_constraint_batch_total": float(np.sum(costs)),
            "low_deadline_constraint_batch_workflow_count": int(costs.size),
            "low_deadline_constraint_window_workflows": int(
                self.window_workflows
            ),
            "low_deadline_dual_updated": float(updated),
            "low_deadline_dual_warmup_active": float(warmup_active),
        }

    def state_dict(self):
        return {
            "deadline_lambda": self.value,
            "deadline_lambda_init": float(self.lambda_init),
            "dual_learning_rate": float(self.dual_lr),
            "deadline_budget": float(self.budget),
            "deadline_constraint_metric": self.constraint_metric,
            "lambda_max": float(self.lambda_max),
            "warmup_episodes": int(self.warmup_episodes),
            "window_workflows": int(self.window_workflows),
            "workflow_cost_window": list(self.workflow_cost_window),
            "update_interval": "episode_end",
        }

    def load_state_dict(self, state):
        """Restore the dual *state*; the configured hyper-parameters always win.

        See ``src.D3QN.deadline_lagrangian.DeadlineLagrangianController`` for the
        measurement that motivated this: restoring the tuning as well made every
        warm-started run silently inherit the checkpoint's ``dual_learning_rate``
        and ignore its own config.  The low controller had the same defect; both
        reliability controllers already restored state only.
        """
        state = dict(state or {})
        saved_metric = str(
            state.get("deadline_constraint_metric", self.constraint_metric)
        ).strip().lower()
        if saved_metric != self.constraint_metric:
            return
        for message in self.hyperparameter_overrides(state):
            print(f"[low-deadline-lagrangian] config wins: {message}", flush=True)
        value = float(state.get("deadline_lambda", self.lambda_init))
        if not np.isfinite(value):
            raise ValueError("Restored low Deadline lambda must be finite.")
        self.lambda_deadline = float(np.clip(value, 0.0, self.lambda_max))
        restored_costs = np.asarray(
            state.get("workflow_cost_window", []), dtype=np.float64
        )
        if restored_costs.ndim != 1 or (
            restored_costs.size
            and (
                not np.isfinite(restored_costs).all()
                or np.any(restored_costs < 0.0)
                or (
                    self.constraint_metric == "workflow_miss_rate"
                    and np.any(restored_costs > 1.0)
                )
            )
        ):
            raise ValueError(
                "Restored low Deadline workflow costs are incompatible with "
                "the configured metric."
            )
        self.workflow_cost_window = deque(
            (float(item) for item in restored_costs[-self.window_workflows :]),
            maxlen=self.window_workflows,
        )

    def hyperparameter_overrides(self, state):
        """Return a description of every checkpoint value the config overrides."""
        state = dict(state or {})
        fields = (
            ("deadline_lambda_init", "lambda_init", float),
            ("dual_learning_rate", "dual_lr", float),
            ("deadline_budget", "budget", float),
            ("lambda_max", "lambda_max", float),
            ("warmup_episodes", "warmup_episodes", int),
            ("window_workflows", "window_workflows", int),
        )
        messages = []
        for key, attribute, cast in fields:
            if key not in state:
                continue
            saved = cast(state[key])
            configured = cast(getattr(self, attribute))
            if saved != configured:
                messages.append(f"{key} checkpoint={saved} config={configured}")
        return messages
