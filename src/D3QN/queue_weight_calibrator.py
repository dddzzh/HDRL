"""Solve beta_q from a scale-free share target instead of hand-setting it.

The Deadline head's dense cost is ``Delta-C_t/D_w + beta_q*q_t/(N_w*D_w)``
(plus the optional sub-Deadline term).  Picking ``beta_q`` directly does not
survive a change of scale: measured on trained checkpoints over seeds
642000-642019, the queue mass is 2.6% of the Delta-C mass on the 28+9 testbed
but 1.4% on the 20+6 one, so the same weight buys a 1.87x different share of
the signal.  Dividing ``q_t`` by ``N_w`` fixes the *units* but not the
*magnitude* -- the bigger testbed genuinely queues harder (aggregate wait per
workflow reaches 101% of its own span, against 38% on the small one).

Writing two different ``beta_q`` values into two configs would make it the
only learning hyperparameter that differs between scales; every other one of
the 53 is shared.  So the configured quantity is the *share* -- what fraction
of the dense cost the queue term should carry -- and this class solves for the
weight that realizes it:

    share = beta_q*Q / (I + beta_q*Q + beta_sd*S)
    beta_q = share/(1-share) * (I + beta_sd*S) / Q

``I``, ``Q`` and ``S`` are accumulated from real rollouts with ``beta_q = 0``,
summed exactly the way ``workflow_dense_miss_credit_deadline_costs`` sums them
so the realized share matches the requested one.

Two deliberate choices:

* **Measure late, not at episode 0.**  A barely-trained policy scatters tasks
  across the pool and therefore queues *less* than the energy-seeking policy it
  becomes, which would understate ``Q`` and overshoot ``beta_q``.  The window
  opens after the policy has settled.

* **Freeze after solving.**  Tracking the share continuously is a positive
  feedback loop: the term suppresses queueing, ``Q`` shrinks, and holding the
  share fixed pushes ``beta_q`` back up.  A frozen value sizes the correction
  against the blind spot as it existed before the correction, which is the
  right reference point, and keeps runs reproducible.
"""
from __future__ import annotations

import math

import numpy as np


EPS = 1e-9


class DeadlineQueueWeightCalibrator:
    """Accumulate the Deadline head's cost masses, then solve for beta_q."""

    def __init__(
        self,
        share,
        *,
        start_episode=150,
        window_episodes=50,
        sub_deadline_weight=0.0,
        weight_max=1000.0,
    ):
        share = float(share)
        if not math.isfinite(share) or not 0.0 <= share < 1.0:
            raise ValueError(
                f"deadline_queue_share must be in [0, 1), got {share!r}."
            )
        self.share = share
        self.start_episode = int(start_episode)
        if self.start_episode < 0:
            raise ValueError("deadline_queue_calibration_start must be >= 0.")
        self.window_episodes = int(window_episodes)
        if self.window_episodes <= 0:
            raise ValueError(
                "deadline_queue_calibration_episodes must be positive."
            )
        self.sub_deadline_weight = float(sub_deadline_weight)
        if self.sub_deadline_weight < 0.0:
            raise ValueError("sub_deadline_weight must be non-negative.")
        self.weight_max = float(weight_max)
        if self.weight_max <= 0.0:
            raise ValueError("weight_max must be positive.")

        self.increment_mass = 0.0
        self.queue_mass = 0.0
        self.sub_deadline_mass = 0.0
        self.observed_episodes = 0
        self.observed_workflows = 0
        self.weight = None

    @property
    def enabled(self):
        return self.share > 0.0

    @property
    def resolved(self):
        return self.weight is not None

    def observe(self, buffer, episode):
        """Fold one episode's rollout in; return the solved weight or None.

        ``episode`` is the zero-based global index, matching the convention the
        Deadline Lagrangian's warmup already uses.
        """
        if not self.enabled or self.resolved:
            return None
        if int(episode) < self.start_episode:
            return None
        if buffer.size == 0:
            return None

        for _workflow_id, indices in buffer.workflow_groups():
            deadlines = np.asarray(
                [
                    max(float(buffer.deadline_limits[index]), EPS)
                    for index in indices
                ],
                dtype=np.float64,
            )
            increments = np.asarray(
                [
                    max(0.0, float(buffer.deadline_costs[index]))
                    for index in indices
                ],
                dtype=np.float64,
            ) / deadlines
            queue = np.asarray(
                [
                    max(
                        0.0,
                        float(
                            buffer.infos[index].get(
                                "high_queue_delay_normalized", 0.0
                            )
                        ),
                    )
                    for index in indices
                ],
                dtype=np.float64,
            )
            sub_deadline = np.asarray(
                [
                    max(
                        0.0,
                        float(
                            buffer.infos[index].get(
                                "high_sub_deadline_normalized", 0.0
                            )
                        ),
                    )
                    for index in indices
                ],
                dtype=np.float64,
            )
            self.increment_mass += float(increments.sum())
            self.queue_mass += float(queue.sum())
            self.sub_deadline_mass += float(sub_deadline.sum())
            self.observed_workflows += 1

        self.observed_episodes += 1
        if self.observed_episodes < self.window_episodes:
            return None
        return self._solve()

    def _solve(self):
        if self.queue_mass <= EPS:
            # No contention was ever observed, so the term has nothing to
            # charge.  Resolving to zero is the honest outcome; a division here
            # would manufacture an enormous weight out of numerical noise.
            self.weight = 0.0
            return self.weight
        numerator = (
            self.increment_mass
            + self.sub_deadline_weight * self.sub_deadline_mass
        )
        weight = (
            self.share / max(1.0 - self.share, EPS) * numerator / self.queue_mass
        )
        self.weight = float(min(max(weight, 0.0), self.weight_max))
        return self.weight

    def realized_share(self, weight=None):
        """Share the given weight actually buys, for logging the solve."""
        weight = self.weight if weight is None else float(weight)
        if weight is None:
            return 0.0
        queue_term = weight * self.queue_mass
        total = (
            self.increment_mass
            + queue_term
            + self.sub_deadline_weight * self.sub_deadline_mass
        )
        return float(queue_term / total) if total > EPS else 0.0

    def diagnostics(self):
        return {
            "deadline_queue_share_target": float(self.share),
            "deadline_queue_share_realized": float(self.realized_share()),
            "deadline_queue_weight_solved": (
                float("nan") if self.weight is None else float(self.weight)
            ),
            "deadline_queue_calibration_episodes": float(
                self.observed_episodes
            ),
            "deadline_queue_calibration_workflows": float(
                self.observed_workflows
            ),
            "deadline_queue_increment_mass": float(self.increment_mass),
            "deadline_queue_queue_mass": float(self.queue_mass),
        }

    def state_dict(self):
        return {
            "share": float(self.share),
            "start_episode": int(self.start_episode),
            "window_episodes": int(self.window_episodes),
            "sub_deadline_weight": float(self.sub_deadline_weight),
            "weight_max": float(self.weight_max),
            "increment_mass": float(self.increment_mass),
            "queue_mass": float(self.queue_mass),
            "sub_deadline_mass": float(self.sub_deadline_mass),
            "observed_episodes": int(self.observed_episodes),
            "observed_workflows": int(self.observed_workflows),
            "weight": None if self.weight is None else float(self.weight),
        }

    def load_state_dict(self, payload):
        """Restore a solve in progress, or an already-frozen weight.

        Resume has to carry the solved weight forward: it is not in the config,
        so a run restarted past the calibration window would otherwise train
        the rest of the way with beta_q = 0 and silently lose the term.
        """
        payload = dict(payload or {})
        saved_share = float(payload.get("share", self.share))
        if not np.isclose(saved_share, self.share):
            raise ValueError(
                "deadline_queue_share mismatch on resume: "
                f"checkpoint={saved_share!r}, config={self.share!r}."
            )
        self.increment_mass = float(payload.get("increment_mass", 0.0))
        self.queue_mass = float(payload.get("queue_mass", 0.0))
        self.sub_deadline_mass = float(payload.get("sub_deadline_mass", 0.0))
        self.observed_episodes = int(payload.get("observed_episodes", 0))
        self.observed_workflows = int(payload.get("observed_workflows", 0))
        weight = payload.get("weight")
        self.weight = None if weight is None else float(weight)
        return self.weight
