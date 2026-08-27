from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from src.a2c.a2c import _require_torch
from src.device_utils import resolve_agent_device

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = object
    F = None


EPS = 1e-8


@dataclass(frozen=True)
class C3CFeatureLayout:
    task_dim: int = 5
    server_dim: int = 12
    low_dim: int = 9
    global_dim: int = 5
    pool_dim: int = 8

    @property
    def low_start(self):
        return self.task_dim + self.server_dim

    @property
    def low_stop(self):
        return self.low_start + self.low_dim

    @property
    def base_dim(self):
        return (
            self.task_dim
            + self.server_dim
            + self.low_dim
            + self.global_dim
            + self.pool_dim
        )

    @property
    def network_input_dim(self):
        return self.base_dim + 1  # continuous rho


FEATURE_LAYOUT = C3CFeatureLayout()


def compute_c3c_step_components(info, *, energy_weight=1.0, deadline_lambda=0.0):
    """Split the realized joint reward into rho=1 placement and DVFS residuals.

    All terms are dimensionless.  The same normalized makespan increment is
    used on both sides, so the decomposition is exact up to floating-point
    precision when lowering rho cannot finish earlier than rho=1.
    """

    energy_scale = max(float(info["task_energy_scale"]), EPS)
    workflow_task_count = max(float(info["workflow_task_count"]), 1.0)
    energy_rho1 = float(
        info.get("low_sac_energy_reference", info["high_placement_energy_rho1"])
    )
    energy_actual = float(
        info.get("low_sac_energy_actual", info["total_energy_actual"])
    )
    reward_profile = str(
        info.get("high_reward_profile", "ep740_lagrangian_v1")
    ).strip().lower()
    if reward_profile == "sub_deadline_lagrangian_v1":
        # Use exactly the same local constraint as the two base agents.  The
        # rho=1 term belongs to placement; only the extra cost caused by the
        # realized DVFS action belongs to the cross-level residual.
        deadline_rho1 = max(
            0.0,
            float(info["low_sac_reference_sub_deadline_cost"]),
        )
        deadline_actual = max(
            0.0,
            float(info["low_sac_actual_sub_deadline_cost"]),
        )
        if deadline_actual + 1e-6 < deadline_rho1:
            raise RuntimeError(
                "C3C sub-Deadline decomposition is non-monotonic: "
                f"rho1={deadline_rho1:.8g}, actual={deadline_actual:.8g}."
            )
        deadline_residual = max(0.0, deadline_actual - deadline_rho1)
    else:
        deadline = max(float(info["deadline_limit"]), EPS)
        partial_before = float(
            info.get("low_sac_workflow_partial_makespan_before", 0.0)
        )
        partial_rho1 = float(
            info.get("low_sac_reference_partial_makespan_after", partial_before)
        )
        partial_actual = float(
            info.get(
                "low_sac_workflow_partial_makespan_after",
                info.get("current_partial_makespan", partial_before),
            )
        )

        deadline_rho1 = max(0.0, partial_rho1 - partial_before) / deadline
        deadline_actual = max(0.0, partial_actual - partial_before) / deadline
        deadline_residual = max(0.0, deadline_actual - deadline_rho1)

    scale = workflow_task_count * energy_scale
    base_energy = float(energy_weight) * (energy_scale - energy_rho1) / scale
    residual_energy = float(energy_weight) * (energy_rho1 - energy_actual) / scale
    actual_energy_reward = (
        float(energy_weight) * (energy_scale - energy_actual) / scale
    )
    base_reward = base_energy - float(deadline_lambda) * deadline_rho1
    residual_reward = (
        residual_energy - float(deadline_lambda) * deadline_residual
    )
    realized_reward = (
        actual_energy_reward - float(deadline_lambda) * deadline_actual
    )
    decomposition_error = base_reward + residual_reward - realized_reward
    values = np.asarray(
        [
            base_energy,
            residual_energy,
            actual_energy_reward,
            deadline_rho1,
            deadline_actual,
            deadline_residual,
            base_reward,
            residual_reward,
            realized_reward,
            decomposition_error,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("C3C reward decomposition contains NaN or Inf.")
    if abs(decomposition_error) > 1e-5:
        raise RuntimeError(
            "C3C reward decomposition is inconsistent: "
            f"error={decomposition_error:.8g}, d_rho1={deadline_rho1:.8g}, "
            f"d_actual={deadline_actual:.8g}."
        )
    return {
        "base_energy_reward": float(base_energy),
        "residual_energy_reward": float(residual_energy),
        "actual_energy_reward": float(actual_energy_reward),
        "base_deadline_cost": float(deadline_rho1),
        "actual_deadline_cost": float(deadline_actual),
        "residual_deadline_cost": float(deadline_residual),
        "base_reward": float(base_reward),
        "residual_reward": float(residual_reward),
        "realized_reward": float(realized_reward),
        "decomposition_error": float(decomposition_error),
        "deadline_crossed_by_dvfs": float(
            info.get("low_sac_deadline_constraint_crossed", 0.0)
        ),
    }


class C3CCandidateEncoder:
    """Permutation-reusable task/server/SAC context for one candidate server."""

    def __init__(self, layout=FEATURE_LAYOUT):
        self.layout = layout

    @property
    def base_dim(self):
        return self.layout.base_dim

    def encode(self, observation, context, low_agent):
        server_features = np.asarray(observation.server_features, dtype=np.float32)
        server_mask = np.asarray(observation.server_mask, dtype=bool)
        action_dim = int(server_features.shape[0])
        if server_features.shape[1] != self.layout.server_dim:
            raise ValueError("C3C server feature dimension is incompatible.")

        task_features = np.asarray(
            observation.ready_task_features[0], dtype=np.float32
        )
        if task_features.shape != (4,):
            raise ValueError("C3C expects the current four-dimensional task state.")
        current_task = observation.ready_tasks[0] if observation.ready_tasks else {}
        deadline = max(float(current_task.get("relative_deadline", 1.0)), EPS)
        upward_rank = np.clip(
            float(current_task.get("upward_rank", 0.0)) / deadline, 0.0, 10.0
        )
        task = np.concatenate(
            (task_features, np.asarray([upward_rank], dtype=np.float32))
        ).astype(np.float32)

        global_features = np.asarray(observation.global_features, dtype=np.float32)
        if global_features.shape != (self.layout.global_dim,):
            raise ValueError("C3C global feature dimension is incompatible.")

        pool = self._pool_summary(server_features, server_mask)
        bases = np.zeros((action_dim, self.base_dim), dtype=np.float32)
        hardware_mins = np.ones(action_dim, dtype=np.float32)
        metrics_by_id = context.get("server_safety_metrics", {})
        for server_index in np.flatnonzero(server_mask):
            server = observation.servers[int(server_index)]
            server_id = int(server["id"])
            safety = metrics_by_id.get(str(server_id))
            if safety is None:
                raise KeyError(f"Missing safety metrics for server {server_id}.")
            low_state = low_agent.build_low_state(
                observation,
                0,
                int(server_index),
                safety_metrics=safety,
            )
            bases[int(server_index)] = np.concatenate(
                (
                    task,
                    server_features[int(server_index)],
                    np.asarray(low_state, dtype=np.float32),
                    global_features,
                    pool,
                )
            )
            hardware_mins[int(server_index)] = np.clip(
                float(safety.get("rho_hardware_min", low_agent.rho_min)),
                low_agent.rho_min,
                1.0,
            )
        if not np.isfinite(bases).all():
            raise FloatingPointError("C3C candidate features contain NaN or Inf.")
        return {
            "base_features": bases,
            "valid_servers": server_mask.copy(),
            "hardware_mins": hardware_mins,
        }

    @staticmethod
    def _pool_summary(server_features, server_mask):
        valid = np.asarray(server_features[server_mask], dtype=np.float32)
        if valid.size <= 0:
            return np.zeros(8, dtype=np.float32)
        # queue processing, AFT, rho_safe, and rho=1 energy; mean + max.
        selected = valid[:, [0, 3, 6, 7]]
        return np.concatenate((selected.mean(axis=0), selected.max(axis=0))).astype(
            np.float32
        )


class C3CReplayBuffer:
    def __init__(
        self,
        *,
        high_state_dim,
        action_dim,
        base_dim,
        max_size=50000,
        batch_size=256,
        harmful_fraction=0.25,
        candidate_storage_dtype="float16",
    ):
        self.high_state_dim = int(high_state_dim)
        self.action_dim = int(action_dim)
        self.base_dim = int(base_dim)
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        self.harmful_fraction = float(harmful_fraction)
        self.mem_count = 0
        candidate_dtype = (
            np.float16
            if str(candidate_storage_dtype).strip().lower() == "float16"
            else np.float32
        )
        self.high_states = np.zeros(
            (self.max_size, self.high_state_dim), dtype=np.float32
        )
        self.actions = np.zeros(self.max_size, dtype=np.int64)
        self.base_energy_rewards = np.zeros(self.max_size, dtype=np.float32)
        self.base_deadline_costs = np.zeros(self.max_size, dtype=np.float32)
        self.selected_bases = np.zeros(
            (self.max_size, self.base_dim), dtype=np.float32
        )
        self.hardware_mins = np.ones(self.max_size, dtype=np.float32)
        self.rhos = np.ones(self.max_size, dtype=np.float32)
        self.residual_energy_rewards = np.zeros(self.max_size, dtype=np.float32)
        self.residual_deadline_costs = np.zeros(self.max_size, dtype=np.float32)
        self.crossed = np.zeros(self.max_size, dtype=np.float32)
        self.next_high_states = np.zeros_like(self.high_states)
        self.next_candidate_bases = np.zeros(
            (self.max_size, self.action_dim, self.base_dim),
            dtype=candidate_dtype,
        )
        self.next_hardware_mins = np.ones(
            (self.max_size, self.action_dim), dtype=np.float32
        )
        self.next_valid_servers = np.zeros(
            (self.max_size, self.action_dim), dtype=bool
        )
        self.dones = np.zeros(self.max_size, dtype=bool)

    @property
    def size(self):
        return min(self.mem_count, self.max_size)

    def ready(self):
        return self.size >= self.batch_size

    def store(self, transition):
        index = self.mem_count % self.max_size
        action = int(transition["action"])
        if not 0 <= action < self.action_dim:
            raise ValueError("C3C action is out of range.")
        self.high_states[index] = transition["high_state"]
        self.actions[index] = action
        self.base_energy_rewards[index] = float(transition["base_energy_reward"])
        self.base_deadline_costs[index] = float(transition["base_deadline_cost"])
        self.selected_bases[index] = transition["selected_base"]
        self.hardware_mins[index] = float(transition["hardware_min"])
        self.rhos[index] = float(transition["rho"])
        self.residual_energy_rewards[index] = float(
            transition["residual_energy_reward"]
        )
        self.residual_deadline_costs[index] = float(
            transition["residual_deadline_cost"]
        )
        self.crossed[index] = float(transition.get("crossed", 0.0))
        self.next_high_states[index] = transition["next_high_state"]
        self.next_candidate_bases[index] = transition["next_candidate_bases"]
        self.next_hardware_mins[index] = transition["next_hardware_mins"]
        self.next_valid_servers[index] = transition["next_valid_servers"]
        self.dones[index] = bool(transition["done"])
        finite = np.asarray(
            [
                self.base_energy_rewards[index],
                self.base_deadline_costs[index],
                self.hardware_mins[index],
                self.rhos[index],
                self.residual_energy_rewards[index],
                self.residual_deadline_costs[index],
                self.crossed[index],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(finite).all():
            raise FloatingPointError("C3C replay contains NaN or Inf.")
        if self.base_deadline_costs[index] < -EPS:
            raise ValueError("C3C base Deadline cost must be non-negative.")
        if self.residual_deadline_costs[index] < -EPS:
            raise ValueError("C3C residual Deadline cost must be non-negative.")
        if not 0.0 <= self.rhos[index] <= 1.0:
            raise ValueError("C3C rho must be in [0, 1].")
        self.mem_count += 1

    def sample(self, rng):
        if not self.ready():
            return None
        size = self.size
        harmful = np.flatnonzero(
            (self.residual_deadline_costs[:size] > EPS)
            | (self.crossed[:size] > 0.5)
        )
        harmful_count = min(
            self.batch_size,
            int(round(self.batch_size * self.harmful_fraction)),
        )
        chosen = []
        if harmful_count > 0 and harmful.size > 0:
            chosen.extend(
                rng.choice(
                    harmful,
                    harmful_count,
                    replace=harmful.size < harmful_count,
                ).tolist()
            )
        remaining = self.batch_size - len(chosen)
        if remaining > 0:
            chosen.extend(
                rng.choice(size, remaining, replace=size < remaining).tolist()
            )
        indices = np.asarray(chosen, dtype=np.int64)
        rng.shuffle(indices)
        return {
            "indices": indices,
            "high_states": self.high_states[indices],
            "actions": self.actions[indices],
            "base_energy_rewards": self.base_energy_rewards[indices],
            "base_deadline_costs": self.base_deadline_costs[indices],
            "selected_bases": self.selected_bases[indices],
            "hardware_mins": self.hardware_mins[indices],
            "rhos": self.rhos[indices],
            "residual_energy_rewards": self.residual_energy_rewards[indices],
            "residual_deadline_costs": self.residual_deadline_costs[indices],
            "crossed": self.crossed[indices],
            "next_high_states": self.next_high_states[indices],
            "next_candidate_bases": self.next_candidate_bases[indices].astype(
                np.float32
            ),
            "next_hardware_mins": self.next_hardware_mins[indices],
            "next_valid_servers": self.next_valid_servers[indices],
            "dones": self.dones[indices],
        }

    def reward_statistics(self):
        if self.size <= 0:
            return {}
        streams = {
            "base_energy": self.base_energy_rewards[: self.size],
            "base_deadline": self.base_deadline_costs[: self.size],
            "residual_energy": self.residual_energy_rewards[: self.size],
            "residual_deadline": self.residual_deadline_costs[: self.size],
        }
        result = {}
        for name, values in streams.items():
            array = np.asarray(values, dtype=np.float64)
            result.update(
                {
                    f"{name}_mean": float(array.mean()),
                    f"{name}_std": float(array.std()),
                    f"{name}_min": float(array.min()),
                    f"{name}_max": float(array.max()),
                    f"{name}_zero_rate": float(np.mean(np.abs(array) <= 1e-12)),
                    f"{name}_unique_ratio": float(
                        np.unique(np.round(array, decimals=8)).size / array.size
                    ),
                }
            )
        result["harmful_rate"] = float(
            np.mean(self.residual_deadline_costs[: self.size] > EPS)
        )
        result["crossing_rate"] = float(np.mean(self.crossed[: self.size] > 0.5))
        return result

    def state_dict(self):
        size = self.size
        return {
            "high_state_dim": self.high_state_dim,
            "action_dim": self.action_dim,
            "base_dim": self.base_dim,
            "max_size": self.max_size,
            "batch_size": self.batch_size,
            "harmful_fraction": self.harmful_fraction,
            "mem_count": self.mem_count,
            **{
                name: getattr(self, name)[:size].copy()
                for name in (
                    "high_states",
                    "actions",
                    "base_energy_rewards",
                    "base_deadline_costs",
                    "selected_bases",
                    "hardware_mins",
                    "rhos",
                    "residual_energy_rewards",
                    "residual_deadline_costs",
                    "crossed",
                    "next_high_states",
                    "next_candidate_bases",
                    "next_hardware_mins",
                    "next_valid_servers",
                    "dones",
                )
            },
        }

    def load_state_dict(self, payload):
        for key, expected in (
            ("high_state_dim", self.high_state_dim),
            ("action_dim", self.action_dim),
            ("base_dim", self.base_dim),
        ):
            if int(payload[key]) != int(expected):
                raise ValueError(f"C3C replay {key} is incompatible.")
        size = min(int(payload["mem_count"]), self.max_size)
        self.mem_count = size
        for name in (
            "high_states",
            "actions",
            "base_energy_rewards",
            "base_deadline_costs",
            "selected_bases",
            "hardware_mins",
            "rhos",
            "residual_energy_rewards",
            "residual_deadline_costs",
            "crossed",
            "next_high_states",
            "next_candidate_bases",
            "next_hardware_mins",
            "next_valid_servers",
            "dones",
        ):
            getattr(self, name)[:size] = payload[name][:size]


class C3CResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, hidden_layers=2):
        _require_torch()
        super().__init__()
        layers = []
        current = int(input_dim)
        for layer_index in range(max(int(hidden_layers), 1)):
            layers.append(nn.Linear(current, int(hidden_dim)))
            if layer_index == 0:
                layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.ReLU())
            current = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.energy_head = nn.Linear(current, 1)
        self.deadline_head = nn.Linear(current, 1)

    def forward(self, features):
        hidden = self.backbone(features)
        return (
            self.energy_head(hidden).squeeze(-1),
            F.softplus(self.deadline_head(hidden)).squeeze(-1),
        )


class CounterfactualCrossLevelCritic:
    """Twin residual critic for D3QN placement and SAC DVFS coordination."""

    def __init__(self, *, high_state_dim, action_dim, config=None, seed=42):
        _require_torch()
        config = dict(config or {})
        self.config = copy.deepcopy(config)
        self.layout = FEATURE_LAYOUT
        self.encoder = C3CCandidateEncoder(self.layout)
        self.high_state_dim = int(high_state_dim)
        self.action_dim = int(action_dim)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.device = resolve_agent_device(config.get("device", "auto"))
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.learning_rate = float(config.get("learning_rate", 1e-4))
        self.training_per_episode = int(config.get("training_per_episode", 5))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.deadline_loss_weight = float(config.get("deadline_loss_weight", 1.0))
        self.beta_final = float(config.get("beta_final", 0.25))
        self.warmup_transitions = int(config.get("warmup_transitions", 20000))
        self.beta_anneal_transitions = int(
            config.get("beta_anneal_transitions", 40000)
        )
        self.uncertainty_scale_floor = float(
            config.get("uncertainty_scale_floor", 0.02)
        )
        self.uncertainty_ema_alpha = float(
            config.get("uncertainty_ema_alpha", 0.99)
        )
        self.low_guidance_enabled = bool(config.get("low_guidance_enabled", True))
        self.low_guidance_weight_final = float(
            config.get("low_guidance_weight_final", 0.02)
        )
        self.low_guidance_updates = int(config.get("low_guidance_updates", 1))
        self.q_scale_alignment_enabled = bool(
            config.get("q_scale_alignment_enabled", False)
        )
        self.correction_q_std_clip = float(
            config.get("correction_q_std_clip", 0.5)
        )
        self.correction_q_std_floor = float(
            config.get("correction_q_std_floor", 1e-3)
        )
        self.correction_advantage_std_floor = float(
            config.get("correction_advantage_std_floor", 1e-4)
        )
        self.relative_to_d3qn_action = bool(
            config.get("relative_to_d3qn_action", False)
        )
        self.extra_deadline_risk_weight = float(
            config.get("extra_deadline_risk_weight", 0.0)
        )
        self.observed_transitions = 0
        self.learn_updates = 0
        self.advantage_scale_ema = self.uncertainty_scale_floor
        self.target_low_actor = None
        if not 0.0 <= self.beta_final <= 1.0:
            raise ValueError("C3C beta_final must be in [0, 1].")
        if self.deadline_loss_weight < 0.0:
            raise ValueError("C3C deadline_loss_weight must be non-negative.")
        if self.uncertainty_scale_floor <= 0.0:
            raise ValueError("C3C uncertainty_scale_floor must be positive.")
        if self.extra_deadline_risk_weight < 0.0:
            raise ValueError(
                "C3C extra_deadline_risk_weight must be non-negative."
            )

        hidden_dim = int(config.get("hidden_dim", 128))
        hidden_layers = int(config.get("hidden_layers", 2))
        with torch.random.fork_rng(
            devices=list(range(torch.cuda.device_count()))
            if torch.cuda.is_available()
            else []
        ):
            torch.manual_seed(self.seed)
            self.critic_1 = C3CResidualNetwork(
                self.layout.network_input_dim, hidden_dim, hidden_layers
            ).to(self.device)
            self.critic_2 = C3CResidualNetwork(
                self.layout.network_input_dim, hidden_dim, hidden_layers
            ).to(self.device)
            self.target_1 = C3CResidualNetwork(
                self.layout.network_input_dim, hidden_dim, hidden_layers
            ).to(self.device)
            self.target_2 = C3CResidualNetwork(
                self.layout.network_input_dim, hidden_dim, hidden_layers
            ).to(self.device)
        self.optimizer_1 = torch.optim.Adam(
            self.critic_1.parameters(), lr=self.learning_rate
        )
        self.optimizer_2 = torch.optim.Adam(
            self.critic_2.parameters(), lr=self.learning_rate
        )
        self.replay = C3CReplayBuffer(
            high_state_dim=self.high_state_dim,
            action_dim=self.action_dim,
            base_dim=self.layout.base_dim,
            max_size=int(config.get("replay_size", 50000)),
            batch_size=int(config.get("batch_size", 256)),
            harmful_fraction=float(config.get("harmful_fraction", 0.25)),
            candidate_storage_dtype=config.get(
                "candidate_storage_dtype", "float16"
            ),
        )
        self.update_targets(tau=1.0)

    @property
    def beta(self):
        excess = max(0, self.observed_transitions - self.warmup_transitions)
        if excess <= 0:
            return 0.0
        if self.beta_anneal_transitions <= 0:
            return self.beta_final
        return float(
            self.beta_final
            * min(1.0, excess / float(self.beta_anneal_transitions))
        )

    @property
    def low_guidance_weight(self):
        if not self.low_guidance_enabled:
            return 0.0
        if self.beta_final <= 0.0:
            return 0.0
        return float(
            self.low_guidance_weight_final * self.beta / self.beta_final
        )

    @property
    def guidance_active(self):
        return bool(self.beta > 0.0 and self.learn_updates > 0)

    def attach_low_actor(self, low_actor):
        self.target_low_actor = copy.deepcopy(low_actor).to(self.device)
        self.target_low_actor.eval()
        for parameter in self.target_low_actor.parameters():
            parameter.requires_grad_(False)

    def predict_rhos(self, candidate_bases, hardware_mins, low_actor, *, target=False):
        bases = np.asarray(candidate_bases, dtype=np.float32)
        low_states = bases[:, self.layout.low_start : self.layout.low_stop]
        actor = (
            self.target_low_actor
            if target and self.target_low_actor is not None
            else low_actor
        )
        states_t = torch.as_tensor(low_states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            rhos, _, _, _ = actor.sample(states_t, deterministic=True)
        values = rhos.detach().cpu().numpy().reshape(-1)
        return np.clip(
            np.maximum(values, np.asarray(hardware_mins, dtype=np.float32)),
            0.0,
            1.0,
        ).astype(np.float32)

    def candidate_guidance(self, candidate_bases, rhos, deadline_lambda):
        bases_t = torch.as_tensor(
            np.asarray(candidate_bases, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        rhos_t = torch.as_tensor(
            np.asarray(rhos, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            result = self._guidance_tensors(
                bases_t, rhos_t, float(deadline_lambda), target=False
            )
        return {
            key: value.detach().cpu().numpy().astype(np.float64)
            for key, value in result.items()
        }

    def _guidance_tensors(self, bases, rhos, deadline_lambda, *, target):
        features = torch.cat((bases, rhos.unsqueeze(-1)), dim=-1)
        networks = (self.target_1, self.target_2) if target else (self.critic_1, self.critic_2)
        e1, d1 = networks[0](features)
        e2, d2 = networks[1](features)
        a1 = e1 - float(deadline_lambda) * d1
        a2 = e2 - float(deadline_lambda) * d2
        uncertainty = (a1 - a2).abs()
        scale = max(float(self.advantage_scale_ema), self.uncertainty_scale_floor)
        confidence = torch.exp(-uncertainty / scale)
        safe_energy = torch.minimum(e1, e2)
        safe_deadline = torch.maximum(d1, d2)
        safe_advantage = safe_energy - float(deadline_lambda) * safe_deadline
        beta_effective = float(self.beta) * confidence
        return {
            "energy": safe_energy,
            "deadline": safe_deadline,
            "advantage": safe_advantage,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "beta_effective": beta_effective,
            "correction": beta_effective * safe_advantage,
        }

    def remember_from_high_buffer(self, high_buffer):
        stored = 0
        if high_buffer.size <= 0:
            return stored
        zero_high = np.zeros(self.high_state_dim, dtype=np.float32)
        zero_candidates = np.zeros(
            (self.action_dim, self.layout.base_dim), dtype=np.float32
        )
        zero_valid = np.zeros(self.action_dim, dtype=bool)
        one_hardware = np.ones(self.action_dim, dtype=np.float32)
        for index in range(high_buffer.size):
            info = high_buffer.infos[index]
            decision = info["high_decision"]
            bases = np.asarray(decision["c3c_candidate_bases"], dtype=np.float32)
            hardware = np.asarray(
                decision["c3c_hardware_mins"], dtype=np.float32
            )
            action = int(high_buffer.primary_actions[index])
            done = index == high_buffer.size - 1
            if done:
                next_high = zero_high
                next_bases = zero_candidates
                next_hardware = one_hardware
                next_valid = zero_valid
            else:
                next_high = np.asarray(high_buffer.states[index + 1], dtype=np.float32)
                next_decision = high_buffer.infos[index + 1]["high_decision"]
                next_bases = np.asarray(
                    next_decision["c3c_candidate_bases"], dtype=np.float32
                )
                next_hardware = np.asarray(
                    next_decision["c3c_hardware_mins"], dtype=np.float32
                )
                next_valid = np.asarray(
                    next_decision["c3c_valid_servers"], dtype=bool
                )
            self.replay.store(
                {
                    "high_state": np.asarray(high_buffer.states[index], dtype=np.float32),
                    "action": action,
                    "base_energy_reward": info["c3c_base_energy_reward"],
                    "base_deadline_cost": info["c3c_base_deadline_cost"],
                    "selected_base": bases[action],
                    "hardware_min": hardware[action],
                    "rho": info["c3c_actual_rho"],
                    "residual_energy_reward": info[
                        "c3c_residual_energy_reward"
                    ],
                    "residual_deadline_cost": info[
                        "c3c_residual_deadline_cost"
                    ],
                    "crossed": info["c3c_deadline_crossed_by_dvfs"],
                    "next_high_state": next_high,
                    "next_candidate_bases": next_bases,
                    "next_hardware_mins": next_hardware,
                    "next_valid_servers": next_valid,
                    "done": done,
                }
            )
            stored += 1
        self.observed_transitions += stored
        return stored

    def _sample_tensors(self):
        sample = self.replay.sample(self.rng)
        if sample is None:
            return None
        result = {}
        long_names = {"actions"}
        bool_names = {"next_valid_servers", "dones"}
        for key, value in sample.items():
            if key == "indices":
                result[key] = value
                continue
            dtype = (
                torch.long
                if key in long_names
                else torch.bool
                if key in bool_names
                else torch.float32
            )
            result[key] = torch.as_tensor(value, dtype=dtype, device=self.device)
        return result

    def _aligned_joint_correction(
        self,
        high_values,
        guidance,
        valid_servers,
        coordination_factor,
        deadline_lambda,
    ):
        correction = guidance["correction"].reshape_as(high_values)
        advantage = guidance["advantage"].reshape_as(high_values)
        deadline = guidance["deadline"].reshape_as(high_values)
        if self.relative_to_d3qn_action:
            masked_high = high_values.masked_fill(~valid_servers, -1e9)
            base_actions = torch.argmax(masked_high, dim=-1, keepdim=True)
            base_advantage = advantage.gather(1, base_actions)
            base_deadline = deadline.gather(1, base_actions)
            extra_risk = torch.relu(deadline - base_deadline)
            advantage = (
                advantage
                - base_advantage
                - self.extra_deadline_risk_weight
                * max(float(deadline_lambda), 0.0)
                * extra_risk
            )
            correction = (
                guidance["beta_effective"].reshape_as(high_values)
                * advantage
            )
        if self.q_scale_alignment_enabled:
            mask = valid_servers.float()
            counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            q_mean = (high_values * mask).sum(dim=-1, keepdim=True) / counts
            advantage_mean = (
                torch.zeros_like(q_mean)
                if self.relative_to_d3qn_action
                else (advantage * mask).sum(dim=-1, keepdim=True) / counts
            )
            q_var = (((high_values - q_mean) * mask) ** 2).sum(
                dim=-1, keepdim=True
            ) / counts
            advantage_var = (((advantage - advantage_mean) * mask) ** 2).sum(
                dim=-1, keepdim=True
            ) / counts
            q_std = q_var.sqrt().clamp_min(self.correction_q_std_floor)
            advantage_std = advantage_var.sqrt()
            beta_effective = guidance["beta_effective"].reshape_as(high_values)
            aligned = (
                beta_effective
                * (advantage - advantage_mean)
                * q_std
                / advantage_std.clamp_min(
                    self.correction_advantage_std_floor
                )
            )
            clip = self.correction_q_std_clip * q_std
            aligned = torch.maximum(torch.minimum(aligned, clip), -clip)
            usable = advantage_std >= self.correction_advantage_std_floor
            correction = torch.where(usable, aligned, torch.zeros_like(aligned))
        return (
            correction
            * valid_servers.float()
            * float(coordination_factor)
        )

    def _next_joint_action(
        self,
        batch,
        high_agent,
        low_agent,
        deadline_lambda,
        coordination_factor=1.0,
    ):
        batch_size, action_dim, base_dim = batch["next_candidate_bases"].shape
        flat_bases = batch["next_candidate_bases"].reshape(-1, base_dim)
        low_states = flat_bases[:, self.layout.low_start : self.layout.low_stop]
        actor = (
            self.target_low_actor
            if self.target_low_actor is not None
            else low_agent.actor
        )
        with torch.no_grad():
            rho_flat, _, _, _ = actor.sample(low_states, deterministic=True)
            rho_flat = torch.maximum(
                rho_flat.reshape(batch_size, action_dim),
                batch["next_hardware_mins"],
            ).clamp(0.0, 1.0)
            guidance = self._guidance_tensors(
                flat_bases,
                rho_flat.reshape(-1),
                deadline_lambda,
                target=False,
            )
            high_values = high_agent.agent_R.q_eval(batch["next_high_states"])
            correction = self._aligned_joint_correction(
                high_values,
                guidance,
                batch["next_valid_servers"],
                coordination_factor,
                deadline_lambda,
            )
            joint = high_values + correction
            joint = joint.masked_fill(~batch["next_valid_servers"], -1e9)
            no_valid = batch["next_valid_servers"].sum(dim=-1) <= 0
            actions = torch.argmax(joint, dim=-1)
            actions = torch.where(no_valid, torch.zeros_like(actions), actions)
        return actions, rho_flat, no_valid

    def update_critics(
        self,
        high_agent,
        low_agent,
        deadline_lambda,
        coordination_factor=1.0,
    ):
        totals = {
            "replay_size": float(self.replay.size),
            "observed_transitions": float(self.observed_transitions),
            "beta": float(self.beta),
        }
        updates = 0
        for _ in range(self.training_per_episode):
            batch = self._sample_tensors()
            if batch is None:
                break
            metrics = self._learn_critics_batch(
                batch,
                high_agent,
                low_agent,
                float(deadline_lambda),
                float(coordination_factor),
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            updates += 1
        if updates:
            for key in list(totals):
                if key in {"replay_size", "observed_transitions", "beta"}:
                    continue
                totals[key] /= updates
        totals["updates"] = float(updates)
        totals["learn_updates_total"] = float(self.learn_updates)
        totals["advantage_scale_ema"] = float(self.advantage_scale_ema)
        totals.update(self.replay.reward_statistics())
        return totals

    def _learn_critics_batch(
        self,
        batch,
        high_agent,
        low_agent,
        deadline_lambda,
        coordination_factor=1.0,
    ):
        actions_next, next_rhos, no_valid = self._next_joint_action(
            batch,
            high_agent,
            low_agent,
            deadline_lambda,
            coordination_factor,
        )
        batch_index = torch.arange(
            batch["high_states"].shape[0], dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            next_bases = batch["next_candidate_bases"][batch_index, actions_next]
            next_rho = next_rhos[batch_index, actions_next]
            next_input = torch.cat((next_bases, next_rho.unsqueeze(-1)), dim=-1)
            te1, td1 = self.target_1(next_input)
            te2, td2 = self.target_2(next_input)
            next_energy = 0.5 * (te1 + te2)
            next_deadline = 0.5 * (td1 + td2)
            terminal = batch["dones"] | no_valid
            non_terminal = (~terminal).float()
            target_energy = (
                batch["residual_energy_rewards"]
                + self.gamma * non_terminal * next_energy
            )
            target_deadline = (
                batch["residual_deadline_costs"]
                + self.gamma * non_terminal * next_deadline
            )
        current_input = torch.cat(
            (batch["selected_bases"], batch["rhos"].unsqueeze(-1)), dim=-1
        )
        e1, d1 = self.critic_1(current_input)
        e2, d2 = self.critic_2(current_input)
        loss_1 = F.smooth_l1_loss(e1, target_energy) + self.deadline_loss_weight * F.smooth_l1_loss(d1, target_deadline)
        loss_2 = F.smooth_l1_loss(e2, target_energy) + self.deadline_loss_weight * F.smooth_l1_loss(d2, target_deadline)
        if not bool(torch.isfinite(loss_1 + loss_2).all().item()):
            raise FloatingPointError("C3C critic loss contains NaN or Inf.")
        grad_norms = []
        for optimizer, network, loss in (
            (self.optimizer_1, self.critic_1, loss_1),
            (self.optimizer_2, self.critic_2, loss_2),
        ):
            optimizer.zero_grad()
            loss.backward()
            grad = nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)
            optimizer.step()
            grad_norms.append(float(grad.item() if torch.is_tensor(grad) else grad))
        with torch.no_grad():
            a1 = e1 - float(deadline_lambda) * d1
            a2 = e2 - float(deadline_lambda) * d2
            scale_value = float((0.5 * (a1.abs() + a2.abs())).mean().item())
            self.advantage_scale_ema = (
                self.uncertainty_ema_alpha * self.advantage_scale_ema
                + (1.0 - self.uncertainty_ema_alpha)
                * max(scale_value, self.uncertainty_scale_floor)
            )
        self.learn_updates += 1
        self.update_targets()
        self.update_target_actor(low_agent.actor)
        return {
            "critic_1_loss": float(loss_1.item()),
            "critic_2_loss": float(loss_2.item()),
            "energy_q_mean": float(torch.minimum(e1, e2).mean().item()),
            "deadline_q_mean": float(torch.maximum(d1, d2).mean().item()),
            "energy_target_mean": float(target_energy.mean().item()),
            "deadline_target_mean": float(target_deadline.mean().item()),
            "energy_td_abs_mean": float((target_energy - e1.detach()).abs().mean().item()),
            "deadline_td_abs_mean": float((target_deadline - d1.detach()).abs().mean().item()),
            "grad_norm_before_clip": float(np.mean(grad_norms)),
        }

    def update_high(
        self,
        high_agent,
        low_agent,
        deadline_lambda,
        coordination_factor=1.0,
    ):
        totals = {"replay_size": float(self.replay.size)}
        updates = 0
        for _ in range(int(high_agent.training_per_episode)):
            batch = self._sample_tensors()
            if batch is None:
                break
            metrics = self._learn_high_batch(
                batch,
                high_agent,
                low_agent,
                float(deadline_lambda),
                float(coordination_factor),
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            updates += 1
        if updates:
            for key in list(totals):
                if key == "replay_size":
                    continue
                totals[key] /= updates
        totals["updates"] = float(updates)
        totals["epsilon"] = float(high_agent.epsilon)
        return totals

    def _learn_high_batch(
        self,
        batch,
        high_agent,
        low_agent,
        deadline_lambda,
        coordination_factor=1.0,
    ):
        batch_size = batch["high_states"].shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.long, device=self.device)
        q = high_agent.agent_R.q_eval(batch["high_states"])[
            batch_index, batch["actions"]
        ]
        rewards = (
            batch["base_energy_rewards"]
            - float(deadline_lambda) * batch["base_deadline_costs"]
        )
        with torch.no_grad():
            actions_next, _, no_valid = self._next_joint_action(
                batch,
                high_agent,
                low_agent,
                deadline_lambda,
                coordination_factor,
            )
            next_q_all = high_agent.agent_R.q_target(batch["next_high_states"])
            next_q = next_q_all[batch_index, actions_next]
            terminal = batch["dones"] | no_valid
            next_q = torch.where(terminal, torch.zeros_like(next_q), next_q)
            target = rewards + high_agent.agent_R.gamma * next_q
        loss = F.smooth_l1_loss(q, target)
        if not bool(torch.isfinite(loss).all().item()):
            raise FloatingPointError("C3C D3QN loss contains NaN or Inf.")
        optimizer = high_agent.agent_R.optimizer
        optimizer.zero_grad()
        loss.backward()
        grad = nn.utils.clip_grad_norm_(
            high_agent.agent_R.q_eval.parameters(), high_agent.agent_R.max_grad_norm
        )
        optimizer.step()
        high_agent.agent_R.update_network_parameters()
        high_agent.agent_R.decrement_epsilon()
        return {
            "loss": float(loss.item()),
            "q_mean": float(q.detach().mean().item()),
            "target_mean": float(target.mean().item()),
            "td_abs_mean": float((target - q.detach()).abs().mean().item()),
            "grad_norm_before_clip": float(
                grad.item() if torch.is_tensor(grad) else grad
            ),
            "reward_mean": float(rewards.mean().item()),
            "reward_std": float(rewards.std(unbiased=False).item()),
        }

    def update_low_safety(self, low_agent, deadline_lambda):
        weight = self.low_guidance_weight
        if weight <= 0.0 or not self.guidance_active:
            return {"updates": 0.0, "weight": float(weight)}
        totals = {}
        updates = 0
        for _ in range(max(self.low_guidance_updates, 0)):
            batch = self._sample_tensors()
            if batch is None:
                break
            low_states = batch["selected_bases"][
                :, self.layout.low_start : self.layout.low_stop
            ]
            rhos, _, _, _ = low_agent.actor.sample(low_states)
            rhos = torch.maximum(rhos.squeeze(-1), batch["hardware_mins"]).clamp(
                0.0, 1.0
            )
            features = torch.cat(
                (batch["selected_bases"], rhos.unsqueeze(-1)), dim=-1
            )
            critic_parameters = list(self.critic_1.parameters()) + list(
                self.critic_2.parameters()
            )
            old_requires_grad = [parameter.requires_grad for parameter in critic_parameters]
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            e1, d1 = self.critic_1(features)
            e2, d2 = self.critic_2(features)
            a1 = e1 - float(deadline_lambda) * d1
            a2 = e2 - float(deadline_lambda) * d2
            uncertainty = (a1 - a2).abs()
            scale = max(self.advantage_scale_ema, self.uncertainty_scale_floor)
            confidence = torch.exp(-uncertainty / scale).detach()
            safe_advantage = torch.minimum(e1, e2) - float(deadline_lambda) * torch.maximum(d1, d2)
            penalty = float(weight) * (
                confidence * F.relu(-safe_advantage)
            ).mean()
            low_agent.actor_optimizer.zero_grad()
            penalty.backward()
            grad = nn.utils.clip_grad_norm_(
                low_agent.actor.parameters(), low_agent.max_grad_norm
            )
            low_agent.actor_optimizer.step()
            for parameter, requires_grad in zip(critic_parameters, old_requires_grad):
                parameter.requires_grad_(requires_grad)
            totals["loss"] = totals.get("loss", 0.0) + float(penalty.item())
            totals["harmful_rate"] = totals.get("harmful_rate", 0.0) + float(
                (safe_advantage.detach() < 0.0).float().mean().item()
            )
            totals["confidence_mean"] = totals.get("confidence_mean", 0.0) + float(
                confidence.mean().item()
            )
            totals["actor_grad_norm_before_clip"] = totals.get(
                "actor_grad_norm_before_clip", 0.0
            ) + float(grad.item() if torch.is_tensor(grad) else grad)
            updates += 1
        if updates:
            totals = {key: value / updates for key, value in totals.items()}
        totals["updates"] = float(updates)
        totals["weight"] = float(weight)
        return totals

    def update_targets(self, tau=None):
        tau = self.tau if tau is None else float(tau)
        for target, online in (
            (self.target_1, self.critic_1),
            (self.target_2, self.critic_2),
        ):
            for target_parameter, online_parameter in zip(
                target.parameters(), online.parameters()
            ):
                target_parameter.data.copy_(
                    tau * online_parameter.data
                    + (1.0 - tau) * target_parameter.data
                )

    def update_target_actor(self, low_actor, tau=None):
        if self.target_low_actor is None:
            self.attach_low_actor(low_actor)
            return
        tau = self.tau if tau is None else float(tau)
        for target_parameter, online_parameter in zip(
            self.target_low_actor.parameters(), low_actor.parameters()
        ):
            target_parameter.data.copy_(
                tau * online_parameter.data
                + (1.0 - tau) * target_parameter.data
            )

    def state_dict(self, include_replay=False):
        return {
            "checkpoint_format": "counterfactual_cross_level_critic_v1",
            "config": copy.deepcopy(self.config),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_1": self.target_1.state_dict(),
            "target_2": self.target_2.state_dict(),
            "optimizer_1": self.optimizer_1.state_dict(),
            "optimizer_2": self.optimizer_2.state_dict(),
            "target_low_actor": (
                None if self.target_low_actor is None else self.target_low_actor.state_dict()
            ),
            "observed_transitions": int(self.observed_transitions),
            "learn_updates": int(self.learn_updates),
            "advantage_scale_ema": float(self.advantage_scale_ema),
            "replay": self.replay.state_dict() if include_replay else None,
        }

    def load_state_dict(self, payload):
        if payload.get("checkpoint_format") != "counterfactual_cross_level_critic_v1":
            raise ValueError("C3C checkpoint format is incompatible.")
        self.critic_1.load_state_dict(payload["critic_1"])
        self.critic_2.load_state_dict(payload["critic_2"])
        self.target_1.load_state_dict(payload["target_1"])
        self.target_2.load_state_dict(payload["target_2"])
        self.optimizer_1.load_state_dict(payload["optimizer_1"])
        self.optimizer_2.load_state_dict(payload["optimizer_2"])
        if self.target_low_actor is not None and payload.get("target_low_actor") is not None:
            self.target_low_actor.load_state_dict(payload["target_low_actor"])
        self.observed_transitions = int(payload.get("observed_transitions", 0))
        self.learn_updates = int(payload.get("learn_updates", 0))
        self.advantage_scale_ema = float(
            payload.get("advantage_scale_ema", self.uncertainty_scale_floor)
        )
        if payload.get("replay") is not None:
            self.replay.load_state_dict(payload["replay"])
