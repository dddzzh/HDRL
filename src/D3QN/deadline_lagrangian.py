from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping

import numpy as np


def tail_conditional_mean(values, tail_fraction):
    """Mean of the worst ``tail_fraction`` of ``values`` (CVaR at that level).

    Costs here are miss rates, so "worst" is largest.  At least one sample is
    always included, which makes the statistic equal to the maximum when the
    window is short.
    """
    array = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    if array.size == 0:
        return 0.0
    tail = max(1, int(math.ceil(float(tail_fraction) * array.size)))
    return float(array[:tail].mean())


class DeadlineLagrangianController:
    """Multiplier for a rolling workflow-level Deadline constraint cost."""

    #: Statistic of the rolling cost window that the dual regulates.
    CONSTRAINT_STATISTICS = ("mean", "cvar")

    #: Metric whose cost is a per-episode pass/fail indicator rather than a
    #: per-workflow miss.  Kept as a set so the validation below reads as a
    #: membership test instead of a string comparison repeated five times.
    EPISODE_METRICS = ("episode_deadline_failure",)

    def __init__(
        self,
        lambda_init,
        dual_lr,
        budget=0.01,
        lambda_max=100.0,
        warmup_episodes=0,
        window_workflows=100,
        constraint_metric="workflow_miss_rate",
        proportional_gain=0.0,
        constraint_statistic="mean",
        cvar_tail_fraction=0.25,
        window_episodes=40,
        episode_deadline_target=0.92,
    ):
        self.lambda_init = float(lambda_init)
        self.dual_lr = float(dual_lr)
        # Proportional (PI-Lagrangian) gain.  Pure integral control is badly
        # asymmetric here: the ascent step is dual_lr*(miss_rate - budget), up to
        # dual_lr*(1 - budget), while the descent step once the constraint holds
        # is only dual_lr*budget.  With budget=0.01 that is a 99x asymmetry, so
        # lambda saturates at lambda_max and stays there long after the
        # constraint is met -- the policy then buys deadline margin it does not
        # need with energy.  A proportional term responds to the *current*
        # excess, so lambda drops the moment the constraint is satisfied.
        # See Stooke, Achiam & Abbeel, "Responsive Safety in Reinforcement
        # Learning by PID Lagrangian Methods", ICML 2020.
        # Default 0.0 keeps the original pure-integral behaviour bit-identical.
        self.proportional_gain = float(proportional_gain)
        self.budget = float(budget)
        self.lambda_max = float(lambda_max)
        self.warmup_episodes = int(warmup_episodes)
        self.window_workflows = int(window_workflows)
        self.constraint_metric = str(constraint_metric).strip().lower()
        if self.constraint_metric not in {
            "workflow_miss_rate",
            "sub_deadline_violation",
            "episode_deadline_failure",
        }:
            raise ValueError(
                "constraint_metric must be 'workflow_miss_rate', "
                "'sub_deadline_violation' or 'episode_deadline_failure'"
            )
        # 'episode_deadline_failure' regulates the quantity the paper reports.
        # ESP is a per-seed pass/fail rate -- a run counts only if its own
        # Deadline success rate clears the target -- so a dual driven by the
        # *pooled* miss rate optimises the wrong thing: it will happily trade
        # one clean run against another that misses four workflows, because
        # both move the pooled average identically.  Here the per-episode cost
        # is c_e = 1{DSR_e < target} and the dual regulates mean(c_e), which is
        # exactly (1 - ESP) on the training distribution.  The D3QN Deadline
        # head is untouched and keeps its dense credit (Delta-C/D + beta_q
        # queue + miss credit): only the multiplier in front of it changes.
        self.episode_deadline_target = float(episode_deadline_target)
        if not 0.0 < self.episode_deadline_target <= 1.0:
            raise ValueError("episode_deadline_target must be in (0, 1].")
        # 'mean' constrains the average miss rate; satisfying it exactly at
        # budget b leaves roughly half the runs above b, i.e. ESP ~ 0.5 at the
        # matching threshold.  'cvar' constrains the mean of the worst tail of
        # per-episode miss rates instead, which is the tractable surrogate for
        # "every run stays under b" -- the quantity ESP actually measures.  It
        # also charges nothing extra on easy episodes, so the multiplier does
        # not buy deadline margin with energy where none is needed.
        self.constraint_statistic = str(constraint_statistic).strip().lower()
        if self.constraint_statistic not in self.CONSTRAINT_STATISTICS:
            raise ValueError(
                "constraint_statistic must be one of "
                f"{list(self.CONSTRAINT_STATISTICS)}"
            )
        if (
            self.constraint_metric in self.EPISODE_METRICS
            and self.constraint_statistic != "mean"
        ):
            # The per-episode cost is already binary, so its CVaR is 1.0 for
            # every failure rate above the tail fraction and 0.0 below it: a
            # step function with no gradient anywhere the dual could use.  The
            # mean of the binary window *is* the tail-sensitive quantity here.
            raise ValueError(
                "constraint_metric='episode_deadline_failure' requires "
                "constraint_statistic='mean'; the per-episode cost is binary, "
                "so its CVaR carries no gradient."
            )
        self.cvar_tail_fraction = float(cvar_tail_fraction)
        if not 0.0 < self.cvar_tail_fraction <= 1.0:
            raise ValueError("cvar_tail_fraction must be in (0, 1].")
        self.window_episodes = int(window_episodes)
        if self.window_episodes <= 0:
            raise ValueError("window_episodes must be positive.")
        if not 0.0 <= self.lambda_init <= self.lambda_max:
            raise ValueError("deadline lambda_init must be in [0, lambda_max].")
        if self.dual_lr < 0.0:
            raise ValueError("deadline dual_lr must be non-negative.")
        if self.proportional_gain < 0.0:
            raise ValueError("deadline proportional_gain must be non-negative.")
        if self.budget < 0.0 or (
            self.constraint_metric
            in {"workflow_miss_rate", *self.EPISODE_METRICS}
            and self.budget > 1.0
        ):
            raise ValueError("deadline constraint budget is invalid")
        if self.lambda_max <= 0.0:
            raise ValueError("deadline lambda_max must be positive.")
        if self.warmup_episodes < 0:
            raise ValueError("deadline warmup_episodes must be non-negative.")
        if self.window_workflows <= 0:
            raise ValueError("deadline window_workflows must be positive.")
        # lambda_integral is the accumulator; lambda_deadline is what the policy
        # sees (integral + proportional).  They coincide when the gain is 0.
        self.lambda_integral = self.lambda_init
        self.lambda_deadline = self.lambda_init
        self.workflow_cost_window = deque(maxlen=self.window_workflows)
        # Per-episode miss rates.  The CVaR has to be taken over run-level
        # rates, not over the flat per-workflow window: with a binary
        # miss indicator the tail of individual workflows is 1.0 whenever the
        # miss rate exceeds the tail fraction, which carries no gradient.
        self.episode_cost_window = deque(maxlen=self.window_episodes)
        # Per-episode pass/fail indicators, c_e = 1{DSR_e < target}.  Filled
        # for every metric so the failure rate stays readable in the training
        # history even when the dual is regulating something else.
        self.episode_failure_window = deque(maxlen=self.window_episodes)

    @classmethod
    def from_config(cls, config: Mapping | None):
        config = dict(config or {})
        update_interval = str(
            config.get("update_interval", "episode_end")
        ).strip().lower()
        if update_interval != "episode_end":
            raise ValueError(
                "Deadline Lagrangian currently supports only "
                "update_interval='episode_end'."
            )
        return cls(
            lambda_init=config.get("deadline_lambda_init", 1.0),
            dual_lr=config.get("dual_learning_rate", 5e-4),
            budget=config.get("deadline_budget", 0.01),
            lambda_max=config.get("lambda_max", 100.0),
            warmup_episodes=config.get("warmup_episodes", 50),
            window_workflows=config.get("window_workflows", 100),
            constraint_metric=config.get(
                "constraint_metric", "workflow_miss_rate"
            ),
            proportional_gain=config.get("proportional_gain", 0.0),
            constraint_statistic=config.get("constraint_statistic", "mean"),
            cvar_tail_fraction=config.get("cvar_tail_fraction", 0.25),
            window_episodes=config.get("window_episodes", 40),
            episode_deadline_target=config.get(
                "episode_deadline_target", 0.92
            ),
        )

    @property
    def min_episodes_for_failure_rate(self):
        """Episodes needed before the binary failure rate is worth acting on.

        A rate over n Bernoulli samples has standard error up to 0.5/sqrt(n);
        below ten episodes that is larger than any budget worth setting, so the
        dual would be chasing sampling noise.
        """
        return min(self.window_episodes, 10)

    @property
    def min_episodes_for_cvar(self):
        """Episodes needed before the tail estimate is worth acting on.

        Below this the CVaR collapses onto the single worst episode seen and
        the dual reacts to one unlucky instance.
        """
        return max(4, int(math.ceil(1.0 / self.cvar_tail_fraction)))

    @property
    def value(self):
        return float(self.lambda_deadline)

    def update(self, workflow_misses: Iterable[float], episode: int):
        costs = np.asarray(list(workflow_misses), dtype=np.float64)
        if costs.ndim != 1:
            raise ValueError("workflow_misses must be one-dimensional.")
        if costs.size > 0 and (
            not np.isfinite(costs).all()
            or np.any(costs < 0.0)
            or (
                self.constraint_metric
                in {"workflow_miss_rate", *self.EPISODE_METRICS}
                and np.any(costs > 1.0)
            )
        ):
            raise ValueError(
                "workflow Deadline costs are incompatible with the configured metric."
            )

        self.workflow_cost_window.extend(float(value) for value in costs)
        window_costs = np.asarray(self.workflow_cost_window, dtype=np.float64)
        lambda_before = self.value
        batch_mean = float(np.mean(costs)) if costs.size else 0.0
        # 1e-9 absorbs the binary-fraction round-off: with 20 workflows and a
        # 0.90 target, 2/20 must compare equal to 1 - 0.90, not above it.
        episode_success_rate = 1.0 - batch_mean
        episode_failed = float(
            episode_success_rate < self.episode_deadline_target - 1e-9
        )
        if costs.size:
            self.episode_cost_window.append(batch_mean)
            self.episode_failure_window.append(episode_failed)
        episode_costs = np.asarray(self.episode_cost_window, dtype=np.float64)
        episode_failures = np.asarray(
            self.episode_failure_window, dtype=np.float64
        )
        mean_cost = float(np.mean(window_costs)) if window_costs.size else 0.0
        cvar_cost = tail_conditional_mean(episode_costs, self.cvar_tail_fraction)
        failure_rate = (
            float(np.mean(episode_failures)) if episode_failures.size else 0.0
        )
        warmup_active = int(episode) < self.warmup_episodes
        if self.constraint_metric in self.EPISODE_METRICS:
            constraint_value = failure_rate
            ready = episode_failures.size >= self.min_episodes_for_failure_rate
        elif self.constraint_statistic == "cvar":
            constraint_value = cvar_cost
            ready = episode_costs.size >= self.min_episodes_for_cvar
        else:
            constraint_value = mean_cost
            ready = bool(window_costs.size)
        updated = bool(ready and not warmup_active)
        excess = constraint_value - self.budget
        if updated:
            self.lambda_integral = float(
                np.clip(
                    self.lambda_integral + self.dual_lr * excess,
                    0.0,
                    self.lambda_max,
                )
            )
            self.lambda_deadline = float(
                np.clip(
                    self.lambda_integral + self.proportional_gain * excess,
                    0.0,
                    self.lambda_max,
                )
            )
        result = {
            "lambda_deadline": self.value,
            "lambda_deadline_before_update": lambda_before,
            "lambda_deadline_after_update": self.value,
            "lambda_deadline_integral": float(self.lambda_integral),
            "lambda_deadline_proportional": float(self.proportional_gain * excess)
            if updated
            else 0.0,
            "deadline_constraint_metric": self.constraint_metric,
            "deadline_constraint_statistic": self.constraint_statistic,
            "deadline_constraint_budget": self.budget,
            "deadline_constraint_mean": mean_cost,
            "deadline_constraint_cvar": cvar_cost,
            "deadline_constraint_cvar_tail_fraction": self.cvar_tail_fraction,
            "deadline_constraint_value": constraint_value,
            "deadline_constraint_batch_mean": batch_mean,
            "deadline_constraint_workflow_count": int(window_costs.size),
            "deadline_constraint_episode_count": int(episode_costs.size),
            "deadline_episode_target": self.episode_deadline_target,
            "deadline_episode_success_rate": episode_success_rate,
            "deadline_episode_failed": episode_failed,
            "deadline_episode_failure_rate": failure_rate,
            "deadline_episode_failure_count": int(episode_failures.size),
            "deadline_excess": excess if ready else 0.0,
            "completed_workflow_count": int(costs.size),
            "deadline_dual_updated": float(updated),
            "deadline_dual_warmup_active": float(warmup_active),
        }
        if self.constraint_metric == "workflow_miss_rate":
            result["deadline_miss_rate"] = batch_mean
        return result

    def state_dict(self):
        return {
            "deadline_lambda": self.value,
            "deadline_lambda_integral": float(self.lambda_integral),
            "deadline_lambda_init": self.lambda_init,
            "dual_learning_rate": self.dual_lr,
            "deadline_budget": self.budget,
            "deadline_constraint_metric": self.constraint_metric,
            "deadline_constraint_statistic": self.constraint_statistic,
            "cvar_tail_fraction": self.cvar_tail_fraction,
            "episode_deadline_target": self.episode_deadline_target,
            "lambda_max": self.lambda_max,
            "warmup_episodes": self.warmup_episodes,
            "window_workflows": self.window_workflows,
            "window_episodes": self.window_episodes,
            "workflow_cost_window": list(self.workflow_cost_window),
            "episode_cost_window": list(self.episode_cost_window),
            "episode_failure_window": list(self.episode_failure_window),
            "update_interval": "episode_end",
        }

    def load_state_dict(self, state):
        """Restore the dual *state*; the configured hyper-parameters always win.

        A checkpoint carries both the multiplier (state) and the controller's
        tuning (hyper-parameters).  Restoring the tuning too made every
        warm-started run silently inherit the checkpoint's ``dual_learning_rate``
        and ``deadline_budget`` and ignore its own config, so a config change to
        the dual step size had no effect at all.  With the shipped
        ``dual_learning_rate = 0.001`` and a measured rolling miss rate of
        0.05-0.08 that gives ``d(lambda)/d(episode) ~ 6e-5``: over 1200 episodes
        lambda moved 1.1419 -> 1.1986 while the value needed to satisfy the 99%
        Deadline target was ~2.0, i.e. a ~14000-episode time constant.  The
        constraint loop was therefore never closed.

        Restored here: ``lambda_deadline`` and ``workflow_cost_window``.
        Taken from the config: ``dual_lr``, ``budget``, ``lambda_max``,
        ``warmup_episodes``, ``window_workflows``, ``lambda_init``.

        The two reliability controllers already behaved this way; only the two
        Deadline controllers restored their tuning.
        """
        state = dict(state or {})
        saved_metric = str(
            state.get("deadline_constraint_metric", self.constraint_metric)
        )
        if saved_metric != self.constraint_metric:
            # Old mean-C/D multipliers are semantically incompatible with the
            # binary miss-rate controller; keep the configured fresh value.
            return
        saved_statistic = str(
            state.get("deadline_constraint_statistic", self.constraint_statistic)
        )
        if saved_statistic != self.constraint_statistic:
            # A multiplier that settled against a mean constraint sits at a
            # different level than one that settled against a tail constraint,
            # so restoring it would start the dual in the wrong place.
            return
        for message in self.hyperparameter_overrides(state):
            print(f"[deadline-lagrangian] config wins: {message}", flush=True)
        value = float(state.get("deadline_lambda", self.lambda_init))
        if not np.isfinite(value):
            raise ValueError("Restored Deadline lambda must be finite.")
        self.lambda_deadline = float(np.clip(value, 0.0, self.lambda_max))
        # Older checkpoints predate the split and carry only the combined value.
        self.lambda_integral = float(
            np.clip(float(state.get("deadline_lambda_integral", value)), 0.0, self.lambda_max)
        )
        self.workflow_cost_window = deque(
            (
                float(item)
                for item in list(state.get("workflow_cost_window", []))[
                    -self.window_workflows :
                ]
            ),
            maxlen=self.window_workflows,
        )
        self.episode_cost_window = deque(
            (
                float(item)
                for item in list(state.get("episode_cost_window", []))[
                    -self.window_episodes :
                ]
            ),
            maxlen=self.window_episodes,
        )
        self.episode_failure_window = deque(
            (
                float(item)
                for item in list(state.get("episode_failure_window", []))[
                    -self.window_episodes :
                ]
            ),
            maxlen=self.window_episodes,
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
            ("window_episodes", "window_episodes", int),
            ("proportional_gain", "proportional_gain", float),
            ("cvar_tail_fraction", "cvar_tail_fraction", float),
            ("episode_deadline_target", "episode_deadline_target", float),
        )
        messages = []
        for key, attribute, cast in fields:
            if key not in state:
                continue
            saved = cast(state[key])
            configured = cast(getattr(self, attribute))
            if saved != configured:
                messages.append(
                    f"{key} checkpoint={saved} config={configured}"
                )
        return messages
