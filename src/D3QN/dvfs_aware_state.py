"""Upper-level state that exposes each server's DVFS operating point.

Why this exists
---------------

The production upper state evaluates every candidate server *at rho = 1*:
``estimated_execution_time``, ``slack_after_assign``, ``safe_execution_budget``
and ``primary_energy_ref_rho1`` are all rho=1 quantities.  The upper policy
therefore commits to a server without any representation of what the lower
DVFS policy will subsequently do there.

That decoupling is measurable.  Replaying the frozen ep300 upper policy on the
20 acceptance seeds while forcing the per-server energy-optimal frequency
``rho* = (P_s / 2 P_d)^(1/3)`` gives:

===================  ===========  =================
metric               rho = 1      forced rho*
===================  ===========  =================
system energy        215.45 kJ    318.52 kJ (+48%)
deadline success     96.75%       40.00%
edge_efficient       7780 tasks   4689 tasks
                     1272.7 kJ    463.1 kJ (-64%)
edge_legacy           845 tasks   2275 tasks
                     1284.8 kJ    3260.6 kJ
edge_lowend          2044 tasks   3387 tasks
===================  ===========  =================

Every per-server saving materialises, yet system energy rises by 48% because
slowing the cheap servers inflates their queues and pushes work onto
``edge_legacy`` (299 W) and ``edge_lowend`` (119 W).  The upper policy cannot
anticipate this because rho is invisible to it.  Consequently the hierarchy has
a fixed point at "the lower level does nothing", which is where the trained SAC
sits (rho = 0.962, capturing 3.6% of the 22.6% available DVFS headroom).

What is added
-------------

Three features per server, appended to the existing twelve.  Everything they
need is already computed by ``EdgeCloudWorkflowEnv.compute_low_safety_metrics``
and carried in ``decision_context['server_safety_metrics']``; no new
environment call is made.

1. ``rho_econ``   -- ``clip(rho*_j, rho_safe_j, 1)``, the energy-optimal
   frequency on server ``j`` that still keeps this task feasible.  When the
   placement is already infeasible the environment reports ``rho_safe = 1`` and
   this feature degenerates to 1, i.e. "no DVFS room here".
2. ``option energy`` -- ``tanh(E_j(rho_econ) / E_median)``, directly comparable
   with existing feature #8 ``tanh(E_j(1) / E_median)``.  Their difference is
   exactly the DVFS advantage of choosing server ``j``.
3. ``option time price`` -- ``tanh(T_exec_j(1) * (1 / rho_econ - 1) / D_w)``,
   the normalized execution-time inflation that operating point costs.

The upper reward is unchanged: under ``reward_profile='ep740_lagrangian_v1'``
``energy_actual_for_reward`` is already the *realized* task energy after the
lower action, so the upper Q_E head already receives DVFS credit.  Only the
observation was missing.
"""
from __future__ import annotations

import numpy as np

from src.a2c.a2c import Observation
from src.D3QN.state import (
    EPS,
    GLOBAL_FEATURE_DIM,
    SERVER_FEATURE_DIM,
    TASK_FEATURE_DIM,
    HierarchicalStateEncoder,
    safe_float,
)

DVFS_EXTRA_FEATURE_DIM = 3
DVFS_SERVER_FEATURE_DIM = SERVER_FEATURE_DIM + DVFS_EXTRA_FEATURE_DIM
DVFS_AWARE_STATE_VERSION = "d3qn_high_dvfs_aware_v1"


def energy_optimal_rho(static_power, dynamic_power):
    """Interior minimiser of ``P_s / rho + P_d * rho^2`` on ``(0, 1]``.

    ``E(rho) = (L / f_max) * (P_s / rho + P_d * rho^2)`` so
    ``dE/drho = 0`` at ``rho = (P_s / 2 P_d)^(1/3)``.  Servers whose static
    power dominates (``edge_lowend``: 86 W static, 31 W dynamic) return 1.0,
    meaning DVFS never saves energy there.
    """
    static_power = max(float(static_power), 0.0)
    dynamic_power = float(dynamic_power)
    if dynamic_power <= 0.0:
        return 1.0
    return float(min(1.0, (static_power / (2.0 * dynamic_power)) ** (1.0 / 3.0)))


def relative_energy_at_rho(static_power, dynamic_power, rho):
    """``E(rho) / E(1)`` for the shared ``P_s / rho + P_d rho^2`` model."""
    static_power = max(float(static_power), 0.0)
    dynamic_power = max(float(dynamic_power), 0.0)
    rho = float(min(1.0, max(rho, EPS)))
    total = static_power + dynamic_power
    if total <= EPS:
        return 1.0
    return float((static_power / rho + dynamic_power * rho * rho) / total)


class DVFSAwareHighStateEncoder(HierarchicalStateEncoder):
    """Production 12-feature server encoding plus three DVFS-option features."""

    version = DVFS_AWARE_STATE_VERSION

    @property
    def server_feature_dim(self):
        return DVFS_SERVER_FEATURE_DIM

    @property
    def high_state_dim(self):
        return (
            TASK_FEATURE_DIM
            + self.max_servers * DVFS_SERVER_FEATURE_DIM
            + self.max_servers
            + GLOBAL_FEATURE_DIM
        )

    def dvfs_option_features(self, server, metrics, task_energy_scale):
        """Return ``(rho_econ, option_energy, option_time_price)`` for a server."""
        static_power = safe_float(server.get("static_power", 0.0))
        dynamic_power = safe_float(server.get("dynamic_power", 0.0))
        rho_safe = float(np.clip(safe_float(metrics.get("rho_safe", 1.0), 1.0), 0.0, 1.0))
        rho_star = energy_optimal_rho(static_power, dynamic_power)
        rho_econ = float(np.clip(max(rho_star, rho_safe), EPS, 1.0))
        energy_ref_rho1 = max(
            safe_float(metrics.get("primary_energy_ref_rho1", 0.0)), 0.0
        )
        option_energy = energy_ref_rho1 * relative_energy_at_rho(
            static_power, dynamic_power, rho_econ
        )
        deadline = max(safe_float(metrics.get("workflow_relative_deadline", 1.0), 1.0), EPS)
        execution_time_rho1 = max(
            safe_float(metrics.get("estimated_execution_time", 0.0)), 0.0
        )
        time_price = execution_time_rho1 * (1.0 / rho_econ - 1.0) / deadline
        return np.asarray(
            [
                rho_econ,
                np.tanh(option_energy / max(float(task_energy_scale), EPS)),
                np.tanh(max(time_price, 0.0)),
            ],
            dtype=np.float32,
        )

    def high_server_features(
        self,
        server,
        task,
        current_time,
        max_server_freq_hz,
        safety_metrics=None,
        task_energy_scale=1.0,
    ):
        base = super().high_server_features(
            server,
            task,
            current_time,
            max_server_freq_hz,
            safety_metrics=safety_metrics,
            task_energy_scale=task_energy_scale,
        )
        metrics = self.high_server_pair_metrics(
            server,
            task,
            current_time,
            max_server_freq_hz,
            safety_metrics=safety_metrics,
        )
        extra = self.dvfs_option_features(server, metrics, task_energy_scale)
        return np.concatenate((base, extra)).astype(np.float32)

    def encode_high_state(self, env_state):
        # Same layout as the production encoder, widened to 15 server features.
        ready_tasks = tuple(env_state.get("ready_tasks", []))
        servers = tuple(env_state.get("servers", []))
        current_time = safe_float(env_state.get("time", 0.0))
        context = env_state.get("decision_context", {})
        metrics_by_id = (
            context.get("server_safety_metrics", {})
            if isinstance(context, dict)
            else {}
        )
        task_energy_scale = max(
            safe_float(context.get("task_energy_scale", 1.0), 1.0)
            if isinstance(context, dict)
            else 1.0,
            EPS,
        )
        max_server_freq_hz = self.max_server_freq_hz(servers)

        task_features = np.zeros(
            (self.max_ready_tasks, TASK_FEATURE_DIM), dtype=np.float32
        )
        task_mask = np.zeros(self.max_ready_tasks, dtype=bool)
        for index, task in enumerate(ready_tasks[: self.max_ready_tasks]):
            task_mask[index] = True
            task_features[index] = self.high_task_features(task, current_time)

        current_task = ready_tasks[0] if ready_tasks else {}
        server_features = np.zeros(
            (self.max_servers, DVFS_SERVER_FEATURE_DIM), dtype=np.float32
        )
        server_mask = np.zeros(self.max_servers, dtype=bool)
        for index, server in enumerate(servers[: self.max_servers]):
            server_mask[index] = True
            server_features[index] = self.high_server_features(
                server,
                current_task,
                current_time,
                max_server_freq_hz,
                safety_metrics=metrics_by_id.get(str(server.get("id")), {}),
                task_energy_scale=task_energy_scale,
            )

        global_features = self.high_global_features(
            ready_tasks, servers, current_time
        )
        current_task_features = (
            task_features[0]
            if ready_tasks
            else np.zeros(TASK_FEATURE_DIM, dtype=np.float32)
        )
        state_vector = np.concatenate(
            [
                current_task_features,
                server_features.reshape(-1),
                server_mask.astype(np.float32),
                global_features,
            ]
        ).astype(np.float32)
        if state_vector.shape != (self.high_state_dim,) or not np.isfinite(
            state_vector
        ).all():
            raise ValueError("DVFS-aware high state vector is invalid.")
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
