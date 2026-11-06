from __future__ import annotations

import copy

import numpy as np

from src.a2c.a2c import _require_torch
from src.device_utils import resolve_agent_device
from src.ppo.hierarchical_agent import _assert_finite_tensor, _build_mlp
from src.ppo.hierarchical_state import HierarchicalStateEncoder
from src.sac.deadline_lagrangian import LowDeadlineLagrangianController
from src.sac.reliability_lagrangian import LowReliabilityLagrangianController

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = object
    F = None


EPS = 1e-8
REWARD_COMPONENT_NAMES = (
    "energy_saving",
    "slack_time_cost",
    "team_energy_reward",
    "deadline_violation",
    "reliability_violation",
)
LEGACY_REWARD_COMPONENT_NAMES = (
    "energy_saving",
    "time_increase",
    "deadline_violation",
    "reliability_violation",
)
CONSTRAINT_FLAG_NAMES = ("deadline_crossed", "reliability_crossed")


class LowSACReplayBuffer:
    def __init__(
        self,
        state_dim,
        max_size=100000,
        batch_size=128,
        action_low=0.0,
        critic_state_dim=None,
        priority_config=None,
    ):
        self.state_dim = int(state_dim)
        self.critic_state_dim = int(
            self.state_dim if critic_state_dim is None else critic_state_dim
        )
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        self.action_low = float(action_low)
        priority_config = dict(priority_config or {})
        self.priority_enabled = bool(priority_config.get("enabled", False))
        self.priority_alpha = float(priority_config.get("alpha", 0.6))
        self.importance_beta = float(priority_config.get("importance_beta", 0.4))
        self.uniform_mix = float(priority_config.get("uniform_mix", 0.6))
        self.responsible_bonus = float(
            priority_config.get("responsible_crossing_bonus", 2.0)
        )
        self.severity_weight = float(priority_config.get("severity_weight", 1.0))
        self.priority_epsilon = float(priority_config.get("epsilon", 1e-4))
        self.priority_cap = float(priority_config.get("priority_cap", 100.0))
        if not 0.0 <= self.priority_alpha <= 1.0:
            raise ValueError("Replay priority alpha must be in [0, 1].")
        if not 0.0 <= self.importance_beta <= 1.0:
            raise ValueError("Replay importance beta must be in [0, 1].")
        if not 0.0 <= self.uniform_mix <= 1.0:
            raise ValueError("Replay uniform_mix must be in [0, 1].")
        if min(self.responsible_bonus, self.severity_weight) < 0.0:
            raise ValueError("Replay priority parameters must be non-negative.")
        if self.priority_epsilon <= 0.0 or self.priority_cap <= 0.0:
            raise ValueError("Replay priority epsilon and cap must be positive.")
        self.mem_count = 0
        self.max_base_priority = 1.0
        self.states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.critic_states = np.zeros(
            (self.max_size, self.critic_state_dim), dtype=np.float32
        )
        self.actions = np.zeros((self.max_size, 1), dtype=np.float32)
        self.reward_components = np.zeros(
            (self.max_size, len(REWARD_COMPONENT_NAMES)), dtype=np.float32
        )
        self.constraint_flags = np.zeros(
            (self.max_size, len(CONSTRAINT_FLAG_NAMES)), dtype=np.float32
        )
        self.deadline_constraint_costs = np.zeros(self.max_size, dtype=np.float32)
        self.responsible_crossings = np.zeros(self.max_size, dtype=np.float32)
        self.responsibility_severity = np.zeros(self.max_size, dtype=np.float32)
        self.priorities = np.ones(self.max_size, dtype=np.float32)
        self.next_states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.next_critic_states = np.zeros(
            (self.max_size, self.critic_state_dim), dtype=np.float32
        )
        self.dones = np.zeros(self.max_size, dtype=bool)

    @property
    def size(self):
        return min(self.mem_count, self.max_size)

    def store_transition(
        self,
        state,
        action,
        reward_components,
        next_state,
        done,
        *,
        critic_state=None,
        next_critic_state=None,
    ):
        index = self.mem_count % self.max_size
        self.states[index] = np.asarray(state, dtype=np.float32)
        self.critic_states[index] = np.asarray(
            state if critic_state is None else critic_state, dtype=np.float32
        )
        action = float(action)
        if not np.isfinite(action) or not self.action_low <= action <= 1.0:
            raise ValueError(
                f"Stored low SAC rho must be in [{self.action_low}, 1], got {action}."
            )
        self.actions[index, 0] = action
        self.reward_components[index] = [
            float(reward_components[name]) for name in REWARD_COMPONENT_NAMES
        ]
        self.constraint_flags[index] = [
            float(reward_components[name]) for name in CONSTRAINT_FLAG_NAMES
        ]
        self.deadline_constraint_costs[index] = float(
            reward_components.get(
                "deadline_constraint_cost",
                reward_components["deadline_violation"],
            )
        )
        if (
            not np.isfinite(self.deadline_constraint_costs[index])
            or self.deadline_constraint_costs[index] < 0.0
        ):
            raise ValueError(
                "Stored low SAC Deadline constraint cost must be finite and non-negative."
            )
        responsible_crossing = float(
            reward_components.get("deadline_counterfactual_crossed", 0.0)
        )
        if responsible_crossing not in {0.0, 1.0}:
            raise ValueError("Responsible Deadline crossing flag must be binary.")
        counterfactual_cost = max(
            float(reward_components.get("deadline_counterfactual_cost", 0.0)),
            0.0,
        )
        self.responsible_crossings[index] = responsible_crossing
        # Severity is used for priority only when rho=1 was feasible and the
        # actual rho crossed the workflow Deadline.  Infeasible placements
        # remain an upper-level responsibility and are not over-sampled here.
        self.responsibility_severity[index] = (
            counterfactual_cost if responsible_crossing > 0.5 else 0.0
        )
        responsibility_multiplier = (
            1.0
            + self.responsible_bonus * responsible_crossing
            + self.severity_weight * self.responsibility_severity[index]
        )
        self.priorities[index] = min(
            self.priority_cap,
            max(self.max_base_priority, self.priority_epsilon)
            * responsibility_multiplier,
        )
        self.next_states[index] = np.asarray(next_state, dtype=np.float32)
        self.next_critic_states[index] = np.asarray(
            next_state if next_critic_state is None else next_critic_state,
            dtype=np.float32,
        )
        self.dones[index] = bool(done)
        self.mem_count += 1

    def ready(self):
        return self.size >= self.batch_size

    def sample_buffer(self, rng, *, return_info=False):
        if not self.ready():
            return None
        if self.priority_enabled:
            raw_priorities = np.maximum(
                self.priorities[: self.size].astype(np.float64),
                self.priority_epsilon,
            )
            priority_probabilities = raw_priorities / raw_priorities.sum()
            probabilities = (
                (1.0 - self.uniform_mix) * priority_probabilities
                + self.uniform_mix / self.size
            )
            probabilities /= probabilities.sum()
            indices = rng.choice(
                self.size,
                self.batch_size,
                replace=False,
                p=probabilities,
            )
            importance_weights = np.power(
                self.size * probabilities[indices], -self.importance_beta
            )
            importance_weights /= max(float(importance_weights.max()), EPS)
        else:
            probabilities = np.full(self.size, 1.0 / self.size, dtype=np.float64)
            indices = rng.choice(self.size, self.batch_size, replace=False)
            importance_weights = np.ones(self.batch_size, dtype=np.float64)
        base_sample = (
            self.states[indices],
            self.actions[indices],
            self.reward_components[indices],
            self.constraint_flags[indices],
            self.deadline_constraint_costs[indices],
            self.next_states[indices],
            self.dones[indices],
        )
        if not return_info:
            return base_sample
        return (
            self.states[indices],
            self.critic_states[indices],
            self.actions[indices],
            self.reward_components[indices],
            self.constraint_flags[indices],
            self.deadline_constraint_costs[indices],
            self.next_states[indices],
            self.next_critic_states[indices],
            self.dones[indices],
            indices,
            importance_weights.astype(np.float32),
            self.responsible_crossings[indices],
            self.responsibility_severity[indices],
            probabilities[indices].astype(np.float32),
        )

    def update_priorities(self, indices, td_errors):
        if not self.priority_enabled:
            return
        indices = np.asarray(indices, dtype=np.int64)
        td_errors = np.asarray(td_errors, dtype=np.float64).reshape(-1)
        if indices.shape != td_errors.shape:
            raise ValueError("Replay priority indices and TD errors must align.")
        base = np.power(np.abs(td_errors) + self.priority_epsilon, self.priority_alpha)
        if base.size:
            self.max_base_priority = max(
                self.max_base_priority, float(np.max(base))
            )
        multiplier = (
            1.0
            + self.responsible_bonus * self.responsible_crossings[indices]
            + self.severity_weight * self.responsibility_severity[indices]
        )
        updated = np.clip(
            base * multiplier,
            self.priority_epsilon,
            self.priority_cap,
        )
        if not np.isfinite(updated).all():
            raise FloatingPointError("Replay priorities contain NaN or Inf.")
        self.priorities[indices] = updated.astype(np.float32)

    def state_dict(self):
        return {
            "state_dim": self.state_dim,
            "critic_state_dim": self.critic_state_dim,
            "action_dim": 1,
            "max_size": self.max_size,
            "batch_size": self.batch_size,
            "action_low": self.action_low,
            "mem_count": self.mem_count,
            "reward_component_names": REWARD_COMPONENT_NAMES,
            "states": self.states,
            "critic_states": self.critic_states,
            "actions": self.actions,
            "reward_components": self.reward_components,
            "constraint_flags": self.constraint_flags,
            "deadline_constraint_costs": self.deadline_constraint_costs,
            "responsible_crossings": self.responsible_crossings,
            "responsibility_severity": self.responsibility_severity,
            "priorities": self.priorities,
            "next_states": self.next_states,
            "next_critic_states": self.next_critic_states,
            "dones": self.dones,
            "priority_enabled": self.priority_enabled,
            "max_base_priority": self.max_base_priority,
        }

    def load_state_dict(self, payload):
        saved_names = tuple(
            payload.get("reward_component_names", LEGACY_REWARD_COMPONENT_NAMES)
        )
        if saved_names != REWARD_COMPONENT_NAMES:
            raise ValueError(
                "Low SAC replay reward components are incompatible with the current "
                f"reward: saved={saved_names}, current={REWARD_COMPONENT_NAMES}."
            )
        self.mem_count = int(payload["mem_count"])
        for name in (
            "states",
            "actions",
            "reward_components",
            "constraint_flags",
            "next_states",
            "dones",
        ):
            getattr(self, name)[:] = payload[name]
        self.critic_states[:] = payload.get("critic_states", payload["states"])
        self.next_critic_states[:] = payload.get(
            "next_critic_states", payload["next_states"]
        )
        self.deadline_constraint_costs[:] = payload.get(
            "deadline_constraint_costs",
            self.reward_components[:, REWARD_COMPONENT_NAMES.index("deadline_violation")],
        )
        self.responsible_crossings[:] = payload.get(
            "responsible_crossings", np.zeros(self.max_size, dtype=np.float32)
        )
        self.responsibility_severity[:] = payload.get(
            "responsibility_severity", np.zeros(self.max_size, dtype=np.float32)
        )
        self.priorities[:] = payload.get(
            "priorities", np.ones(self.max_size, dtype=np.float32)
        )
        self.max_base_priority = float(payload.get("max_base_priority", 1.0))

    def reward_statistics(self, std_floor):
        if self.size <= 0:
            return None
        values = self.reward_components[: self.size]
        floors = np.asarray(std_floor, dtype=np.float64)
        if floors.ndim == 0:
            floors = np.full(len(REWARD_COMPONENT_NAMES), float(floors), dtype=np.float64)
        if floors.shape != (len(REWARD_COMPONENT_NAMES),):
            raise ValueError(
                "Low SAC reward std floors must be scalar or have shape "
                f"{(len(REWARD_COMPONENT_NAMES),)}, got {floors.shape}."
            )
        return {
            "mean": values.mean(axis=0, dtype=np.float64).astype(np.float32),
            "std": np.maximum(values.std(axis=0, dtype=np.float64), floors).astype(
                np.float32
            ),
        }


class LowSACActor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim=1,
        hidden_dim=128,
        hidden_layers=2,
        log_std_min=-5.0,
        log_std_max=2.0,
        action_low=0.0,
        action_high=1.0,
    ):
        _require_torch()
        super().__init__()
        self.backbone = _build_mlp(state_dim, hidden_dim, hidden_layers)
        self.mean_head = nn.Linear(int(hidden_dim), int(action_dim))
        self.log_std_head = nn.Linear(int(hidden_dim), int(action_dim))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        if not 0.0 <= self.action_low < self.action_high <= 1.0:
            raise ValueError("SAC actor action bounds must satisfy 0 <= low < high <= 1.")

    def forward(self, state):
        features = self.backbone(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        pre_tanh = mean if deterministic else distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        unit_action = 0.5 * (squashed + 1.0)
        action_span = self.action_high - self.action_low
        action = self.action_low + action_span * unit_action
        log_prob = distribution.log_prob(pre_tanh)
        log_prob -= torch.log(1.0 - squashed.pow(2) + EPS)
        # Change of variables: z -> tanh(z) -> rho in [action_low, action_high].
        log_prob += np.log(2.0) - np.log(action_span)
        return action, log_prob.sum(dim=-1), mean, log_std


class LowSACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, hidden_layers=2):
        _require_torch()
        super().__init__()
        self.backbone = _build_mlp(state_dim + action_dim, hidden_dim, hidden_layers)
        self.q_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, state, action):
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        features = self.backbone(torch.cat((state, action), dim=-1))
        return self.q_head(features).squeeze(-1)


class LowLevelDVFSContinuousSACAgent:
    def __init__(self, config=None, seed=42):
        _require_torch()
        config = dict(config or {})
        self.config = copy.deepcopy(config)
        self.seed = int(seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.rng = np.random.default_rng(self.seed)
        state_cfg = config.get("state", config)
        self.encoder = HierarchicalStateEncoder(state_cfg)
        self.state_mode = str(
            config.get("state_mode", "local_9")
        ).strip().lower()
        if self.state_mode not in {
            "local_9",
            "shared_high_selected_server",
            "action_conditioned_ctde",
        }:
            raise ValueError(
                "low SAC state_mode must be 'local_9', "
                "'shared_high_selected_server', or 'action_conditioned_ctde'."
            )
        self.shared_high_server_feature_dim = int(
            config.get("shared_high_server_feature_dim", 0)
        )
        self.shared_high_global_feature_dim = int(
            config.get("shared_high_global_feature_dim", 0)
        )
        self.actor_server_feature_indices = tuple(
            int(value)
            for value in config.get("actor_server_feature_indices", [0, 9, 10])
        )
        if self.state_mode == "shared_high_selected_server":
            shared_high_state_dim = int(
                config.get("shared_high_state_dim", 0)
            )
            if shared_high_state_dim <= 0:
                raise ValueError(
                    "shared_high_selected_server requires a positive "
                    "shared_high_state_dim."
                )
            self._low_state_dim = (
                shared_high_state_dim + self.encoder.max_servers
            )
            self._critic_state_dim = self._low_state_dim
        elif self.state_mode == "action_conditioned_ctde":
            if (
                self.shared_high_server_feature_dim <= 0
                or self.shared_high_global_feature_dim <= 0
            ):
                raise ValueError(
                    "action_conditioned_ctde requires positive shared high "
                    "server/global feature dimensions."
                )
            if (
                not self.actor_server_feature_indices
                or min(self.actor_server_feature_indices) < 0
                or max(self.actor_server_feature_indices)
                >= self.shared_high_server_feature_dim
            ):
                raise ValueError(
                    "actor_server_feature_indices must address valid high-level "
                    "server features."
                )
            self._low_state_dim = int(
                self.encoder.low_state_dim + len(self.actor_server_feature_indices)
            )
            # The actor receives only the selected server.  Existing twin
            # critics additionally receive permutation-invariant mean/max
            # summaries of all valid servers and the compact global features.
            self._critic_state_dim = int(
                self._low_state_dim
                + 2 * self.shared_high_server_feature_dim
                + self.shared_high_global_feature_dim
            )
        else:
            self._low_state_dim = int(self.encoder.low_state_dim)
            self._critic_state_dim = self._low_state_dim
        self.device = resolve_agent_device(config.get("device", "auto"))
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.learning_rate = float(config.get("learning_rate", 3e-4))
        self.batch_size = int(config.get("batch_size", 128))
        self.training_per_episode = int(config.get("training_per_episode", 10))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.rho_min = float(config.get("rho_min", 0.1))
        reward_config = dict(config.get("reward", {}))
        self.reward_mode = str(
            reward_config.get("mode", "fixed_deadline_penalty")
        ).strip().lower()
        if self.reward_mode not in {
            "fixed_deadline_penalty",
            "lagrangian_deadline",
            "sub_deadline_lagrangian",
        }:
            raise ValueError(
                "low SAC reward.mode must be 'fixed_deadline_penalty' or "
                "'lagrangian_deadline' or 'sub_deadline_lagrangian'."
            )
        lagrangian_config = dict(config.get("lagrangian", {}))
        lagrangian_requested = bool(lagrangian_config.get("enabled", False))
        if self.reward_mode in {
            "lagrangian_deadline",
            "sub_deadline_lagrangian",
        } and not lagrangian_requested:
            raise ValueError(
                "low reward.mode='lagrangian_deadline' requires "
                "low.lagrangian.enabled=true."
            )
        self.deadline_lagrangian = (
            LowDeadlineLagrangianController.from_config(lagrangian_config)
            if self.reward_mode in {
                "lagrangian_deadline",
                "sub_deadline_lagrangian",
            }
            else None
        )
        reliability_lagrangian_config = dict(
            config.get("reliability_lagrangian", {})
        )
        self.reliability_lagrangian = (
            LowReliabilityLagrangianController.from_config(
                reliability_lagrangian_config
            )
            if bool(reliability_lagrangian_config.get("enabled", False))
            else None
        )
        self.slack_floor_ratio = float(reward_config.get("slack_floor_ratio", 0.05))
        self.reward_weights = np.asarray(
            [
                reward_config.get("energy_weight", 1.0),
                reward_config.get(
                    "slack_time_weight", reward_config.get("time_weight", 1.0)
                ),
                reward_config.get("team_energy_weight", 0.0),
                reward_config.get(
                    "deadline_violation_weight",
                    reward_config.get("deadline_severity_weight", 5.0),
                ),
            ],
            dtype=np.float32,
        )
        legacy_std_floor = float(reward_config.get("std_floor", 0.05))
        self.reward_std_floors = np.asarray(
            [
                reward_config.get("energy_std_floor", legacy_std_floor),
                reward_config.get("slack_time_std_floor", 0.005),
                reward_config.get("team_energy_std_floor", 0.01),
                reward_config.get("deadline_std_floor", legacy_std_floor),
                reward_config.get("reliability_std_floor", legacy_std_floor),
            ],
            dtype=np.float32,
        )
        self.reward_component_clip = float(reward_config.get("component_clip", 3.0))
        if np.any(self.reward_weights < 0.0):
            raise ValueError("low SAC reward weights must be non-negative.")
        if self.slack_floor_ratio <= 0.0:
            raise ValueError("low SAC reward slack_floor_ratio must be positive.")
        if self.reward_component_clip <= 0.0 or np.any(self.reward_std_floors <= 0.0):
            raise ValueError(
                "low SAC reward component_clip and component std floors must be positive."
            )
        self.action_dim = int(config.get("action_dim", 1))
        if self.action_dim != 1:
            raise ValueError("low continuous SAC supports exactly one rho action.")
        self.alpha = float(config.get("alpha", 0.2))
        self.automatic_entropy_tuning = bool(config.get("automatic_entropy_tuning", True))
        hidden_dim = int(config.get("hidden_dim", 128))
        hidden_layers = int(config.get("hidden_layers", 2))
        log_std_min = float(config.get("log_std_min", -5.0))
        log_std_max = float(config.get("log_std_max", 2.0))
        if log_std_min >= log_std_max:
            raise ValueError("low continuous SAC log_std_min must be smaller than log_std_max.")
        self.actor = LowSACActor(
            self.low_state_dim,
            self.action_dim,
            hidden_dim,
            hidden_layers,
            log_std_min,
            log_std_max,
            action_low=self.rho_min,
            action_high=1.0,
        ).to(self.device)
        self.initial_rho_mean = config.get("initial_rho_mean")
        self.initial_log_std = config.get("initial_log_std")
        if self.initial_rho_mean is not None:
            initial_rho = float(self.initial_rho_mean)
            if not self.rho_min < initial_rho < 1.0:
                raise ValueError(
                    "initial_rho_mean must lie strictly between rho_min and 1."
                )
            unit_action = (initial_rho - self.rho_min) / (1.0 - self.rho_min)
            squashed = float(np.clip(2.0 * unit_action - 1.0, -0.999999, 0.999999))
            mean_bias = float(np.arctanh(squashed))
            nn.init.zeros_(self.actor.mean_head.weight)
            nn.init.constant_(self.actor.mean_head.bias, mean_bias)
        if self.initial_log_std is not None:
            initial_log_std = float(self.initial_log_std)
            if not log_std_min <= initial_log_std <= log_std_max:
                raise ValueError(
                    "initial_log_std must lie within [log_std_min, log_std_max]."
                )
            nn.init.zeros_(self.actor.log_std_head.weight)
            nn.init.constant_(self.actor.log_std_head.bias, initial_log_std)
        self.critic_1 = LowSACCritic(self.critic_state_dim, self.action_dim, hidden_dim, hidden_layers).to(self.device)
        self.critic_2 = LowSACCritic(self.critic_state_dim, self.action_dim, hidden_dim, hidden_layers).to(self.device)
        self.target_critic_1 = LowSACCritic(self.critic_state_dim, self.action_dim, hidden_dim, hidden_layers).to(self.device)
        self.target_critic_2 = LowSACCritic(self.critic_state_dim, self.action_dim, hidden_dim, hidden_layers).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.learning_rate)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.learning_rate)
        if self.automatic_entropy_tuning:
            self.target_entropy = float(config.get("target_entropy", -float(self.action_dim)))
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)
        else:
            self.target_entropy = 0.0
            self.log_alpha = None
            self.alpha_optimizer = None
        self.replay = LowSACReplayBuffer(
            self.low_state_dim,
            max_size=int(config.get("replay_size", config.get("max_size", 100000))),
            batch_size=self.batch_size,
            action_low=self.rho_min,
            critic_state_dim=self.critic_state_dim,
            priority_config=config.get("replay_priority", {}),
        )
        self.update_target_networks(tau=1.0)

    @property
    def low_state_dim(self):
        return int(self._low_state_dim)

    @property
    def actor_state_dim(self):
        return int(self._low_state_dim)

    @property
    def critic_state_dim(self):
        return int(self._critic_state_dim)

    def build_low_state(self, high_observation, task_index, server_index, safety_metrics=None):
        if self.state_mode == "shared_high_selected_server":
            high_state = np.asarray(
                high_observation.vector, dtype=np.float32
            ).reshape(-1)
            expected_high_dim = self.low_state_dim - self.encoder.max_servers
            if high_state.size != expected_high_dim:
                raise ValueError(
                    "shared high observation dimension mismatch: "
                    f"expected {expected_high_dim}, got {high_state.size}."
                )
            server_index = int(server_index)
            if not 0 <= server_index < self.encoder.max_servers:
                raise ValueError(
                    f"selected server index out of range: {server_index}."
                )
            selected_server = np.zeros(
                self.encoder.max_servers, dtype=np.float32
            )
            selected_server[server_index] = 1.0
            return np.concatenate(
                (high_state, selected_server)
            ).astype(np.float32)
        local_state = self.encoder.encode_low_state(
            high_observation,
            task_index,
            server_index,
            safety_metrics=safety_metrics,
        )
        if self.state_mode != "action_conditioned_ctde":
            return local_state
        server_index = int(server_index)
        server_features = np.asarray(
            high_observation.server_features, dtype=np.float32
        )
        if not 0 <= server_index < server_features.shape[0]:
            raise ValueError(f"selected server index out of range: {server_index}.")
        selected_features = server_features[
            server_index, list(self.actor_server_feature_indices)
        ]
        result = np.concatenate((local_state, selected_features)).astype(np.float32)
        if result.shape != (self.actor_state_dim,) or not np.isfinite(result).all():
            raise ValueError("Action-conditioned SAC actor state is invalid.")
        return result

    def build_critic_state(self, high_observation, actor_state):
        actor_state = np.asarray(actor_state, dtype=np.float32).reshape(-1)
        if actor_state.shape != (self.actor_state_dim,):
            raise ValueError("SAC actor state dimension mismatch.")
        if self.state_mode != "action_conditioned_ctde":
            return actor_state.copy()
        server_features = np.asarray(
            high_observation.server_features, dtype=np.float32
        )
        server_mask = np.asarray(high_observation.server_mask, dtype=bool).reshape(-1)
        if (
            server_features.ndim != 2
            or server_features.shape[1] != self.shared_high_server_feature_dim
            or server_features.shape[0] != server_mask.size
        ):
            raise ValueError("High-level server feature layout is incompatible with CTDE.")
        valid_features = server_features[server_mask]
        if valid_features.size:
            server_mean = valid_features.mean(axis=0, dtype=np.float64)
            server_max = valid_features.max(axis=0)
        else:
            server_mean = np.zeros(self.shared_high_server_feature_dim, dtype=np.float32)
            server_max = np.zeros(self.shared_high_server_feature_dim, dtype=np.float32)
        global_features = np.asarray(
            high_observation.global_features, dtype=np.float32
        ).reshape(-1)
        if global_features.shape != (self.shared_high_global_feature_dim,):
            raise ValueError("High-level global feature layout is incompatible with CTDE.")
        result = np.concatenate(
            (actor_state, global_features, server_mean, server_max)
        ).astype(np.float32)
        if result.shape != (self.critic_state_dim,) or not np.isfinite(result).all():
            raise ValueError("Action-conditioned SAC critic state is invalid.")
        return result

    def select_frequency(self, high_observation, task_index, server_index, deterministic=False, safety_metrics=None):
        safety_metrics = dict(safety_metrics or {})
        low_state = self.build_low_state(high_observation, task_index, server_index, safety_metrics=safety_metrics)
        critic_state = self.build_critic_state(high_observation, low_state)
        rho_safe = min(1.0, max(self.rho_min, float(safety_metrics.get("rho_safe", self.rho_min))))
        rho_hardware_min = min(
            1.0,
            max(self.rho_min, float(safety_metrics.get("rho_hardware_min", self.rho_min))),
        )
        low_action_feasible = bool(safety_metrics.get("low_action_feasible", True))
        has_action_span = bool(rho_hardware_min < 1.0 - 1e-6)
        state_tensor = torch.as_tensor(low_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        critic_state_tensor = torch.as_tensor(
            critic_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if has_action_span:
            with torch.no_grad():
                action_tensor, log_prob, mean, log_std = self.actor.sample(
                    state_tensor, deterministic=deterministic
                )
                value = float(
                    torch.min(
                        self.critic_1(critic_state_tensor, action_tensor),
                        self.critic_2(critic_state_tensor, action_tensor),
                    ).item()
                )
            # The SAC action is the actual normalized DVFS frequency rho.
            # A final clip only enforces the server-specific action floor.
            rho_raw = float(action_tensor.item())
            rho = float(min(1.0, max(rho_hardware_min, rho_raw)))
            # The actor's log-prob refers to rho_raw while the environment executes
            # the clipped rho, so a high binding rate means the actor is being
            # rewarded for a frequency it never runs at.  Track it: if this stays
            # small the clip is a harmless guardrail, if it grows the action needs
            # a per-server affine map instead of a clip.
            rho_floor_binding = float(rho_raw < rho_hardware_min - 1e-9)
            action_rho = rho
            log_prob_value = float(log_prob.item())
            entropy_value = -log_prob_value
            policy_mean = float(mean.item())
            policy_log_std = float(log_std.item())
            actor_update_mask = True
        else:
            action_rho = 1.0
            rho = 1.0
            rho_floor_binding = 0.0
            log_prob_value = 0.0
            entropy_value = 0.0
            value = 0.0
            policy_mean = 0.0
            policy_log_std = 0.0
            actor_update_mask = False
        rho = float(min(1.0, max(rho_hardware_min, rho)))
        return {
            "state": low_state,
            "critic_state": critic_state,
            "action_rho": action_rho,
            # Compatibility alias for the shared low rollout container.  Its
            # value is now rho itself, not a separate unit action.
            "action_u": action_rho,
            "rho": rho,
            "log_prob": log_prob_value,
            "value": value,
            "deadline_value": 0.0,
            "rho_alpha": 0.0,
            "rho_beta": 0.0,
            "entropy": entropy_value,
            "policy_mean": policy_mean,
            "policy_log_std": policy_log_std,
            "rho_safe": float(rho_safe),
            "rho_hardware_min": float(rho_hardware_min),
            "rho_energy_efficient": float(
                safety_metrics.get("rho_energy_efficient", rho_hardware_min)
            ),
            "rho_floor_binding": float(rho_floor_binding),
            "low_action_feasible": low_action_feasible,
            "has_action_span": has_action_span,
            "actor_update_mask": actor_update_mask,
            "forced_max_frequency": not has_action_span,
            "low_algorithm": "sac",
        }

    def remember_transition(
        self,
        state,
        action_rho,
        reward_components,
        next_state,
        done,
        *,
        critic_state=None,
        next_critic_state=None,
    ):
        self.replay.store_transition(
            state,
            action_rho,
            reward_components,
            next_state,
            done,
            critic_state=critic_state,
            next_critic_state=next_critic_state,
        )

    @property
    def deadline_lambda(self):
        if self.deadline_lagrangian is None:
            return 0.0
        return self.deadline_lagrangian.value

    def update_deadline_lagrangian(self, workflow_costs, episode):
        if self.deadline_lagrangian is None:
            return {}
        return self.deadline_lagrangian.update(workflow_costs, episode)

    @property
    def reliability_lambda(self):
        if self.reliability_lagrangian is None:
            return 0.0
        return self.reliability_lagrangian.value

    def update_reliability_lagrangian(self, workflow_costs, episode):
        if self.reliability_lagrangian is None:
            return {}
        return self.reliability_lagrangian.update(workflow_costs, episode)

    def update(self, buffer=None, learning_rate_scale=1.0):
        totals = {"replay_size": float(self.replay.size)}
        updates = 0
        old_lrs = []
        for optimizer in (self.actor_optimizer, self.critic_1_optimizer, self.critic_2_optimizer):
            for group in optimizer.param_groups:
                old_lrs.append((group, group["lr"]))
                group["lr"] = self.learning_rate * float(learning_rate_scale)
        reward_statistics = self.replay.reward_statistics(self.reward_std_floors)
        for _ in range(self.training_per_episode):
            metrics = self.learn(reward_statistics)
            if not metrics:
                continue
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            updates += 1
        for group, old_lr in old_lrs:
            group["lr"] = old_lr
        if updates > 0:
            for key in list(totals.keys()):
                if key == "replay_size":
                    continue
                totals[key] /= updates
        totals["updates"] = float(updates)
        totals["replay_size"] = float(self.replay.size)
        return totals

    def learn(self, reward_statistics=None):
        sample = self.replay.sample_buffer(self.rng, return_info=True)
        if sample is None:
            return {}
        (
            states,
            critic_states,
            actions,
            reward_components,
            constraint_flags,
            deadline_constraint_costs,
            next_states,
            next_critic_states,
            dones,
            replay_indices,
            importance_weights,
            responsible_crossings,
            responsibility_severity,
            sample_probabilities,
        ) = sample
        if reward_statistics is None:
            reward_statistics = self.replay.reward_statistics(self.reward_std_floors)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        critic_states_t = torch.as_tensor(
            critic_states, dtype=torch.float32, device=self.device
        )
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        components_t = torch.as_tensor(reward_components, dtype=torch.float32, device=self.device)
        flags_t = torch.as_tensor(constraint_flags, dtype=torch.float32, device=self.device)
        deadline_costs_t = torch.as_tensor(
            deadline_constraint_costs, dtype=torch.float32, device=self.device
        )
        reliability_costs_t = components_t[:, 4].clamp_min(0.0)
        rewards_t = self._compose_rewards(
            components_t,
            flags_t,
            reward_statistics,
            deadline_constraint_costs=deadline_costs_t,
            reliability_constraint_costs=reliability_costs_t,
        )
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        next_critic_states_t = torch.as_tensor(
            next_critic_states, dtype=torch.float32, device=self.device
        )
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        importance_weights_t = torch.as_tensor(
            importance_weights, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            next_actions, next_log_probs, _, _ = self.actor.sample(next_states_t)
            next_q = torch.min(
                self.target_critic_1(next_critic_states_t, next_actions),
                self.target_critic_2(next_critic_states_t, next_actions),
            )
            next_values = next_q - self.alpha * next_log_probs
            target = rewards_t + self.gamma * (~dones_t).float() * next_values
        q1 = self.critic_1(critic_states_t, actions_t)
        q2 = self.critic_2(critic_states_t, actions_t)
        critic_1_loss = (importance_weights_t * (q1 - target).pow(2)).mean()
        critic_2_loss = (importance_weights_t * (q2 - target).pow(2)).mean()
        _assert_finite_tensor("low_sac_critic_loss", critic_1_loss + critic_2_loss)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_1_grad = nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        critic_2_grad = nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)
        self.critic_2_optimizer.step()
        policy_actions, log_probs, policy_means, policy_log_stds = self.actor.sample(states_t)
        q_min = torch.min(
            self.critic_1(critic_states_t, policy_actions),
            self.critic_2(critic_states_t, policy_actions),
        )
        actor_loss = (self.alpha * log_probs - q_min).mean()
        _assert_finite_tensor("low_sac_actor_loss", actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        entropy = -log_probs.mean()
        alpha_loss_value = 0.0
        if self.automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha * (log_probs.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = float(self.log_alpha.exp().detach().item())
            alpha_loss_value = float(alpha_loss.item())
        self.update_target_networks()
        td_error = target.detach() - q1.detach()
        replay_td_errors = torch.maximum(
            (target.detach() - q1.detach()).abs(),
            (target.detach() - q2.detach()).abs(),
        )
        self.replay.update_priorities(
            replay_indices, replay_td_errors.cpu().numpy()
        )
        metrics = {
            "critic_1_loss": float(critic_1_loss.item()),
            "critic_2_loss": float(critic_2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha),
            "entropy": float(entropy.item()),
            "q_mean": float(q1.detach().mean().item()),
            "target_mean": float(target.detach().mean().item()),
            "td_abs_mean": float(td_error.abs().mean().item()),
            "critic_1_grad_norm_before_clip": self._grad_to_float(critic_1_grad),
            "critic_2_grad_norm_before_clip": self._grad_to_float(critic_2_grad),
            "actor_grad_norm_before_clip": self._grad_to_float(actor_grad),
            "reward_mean": float(rewards_t.mean().item()),
            "reward_std": float(rewards_t.std(unbiased=False).item()),
            "policy_action_mean": float(policy_actions.detach().mean().item()),
            "policy_action_std": float(policy_actions.detach().std(unbiased=False).item()),
            "policy_mean": float(policy_means.detach().mean().item()),
            "policy_log_std": float(policy_log_stds.detach().mean().item()),
            "replay_priority_enabled": float(self.replay.priority_enabled),
            "replay_responsible_sample_rate": float(
                np.mean(responsible_crossings)
            ),
            "replay_responsibility_severity_mean": float(
                np.mean(responsibility_severity)
            ),
            "replay_importance_weight_mean": float(
                np.mean(importance_weights)
            ),
            "replay_sample_probability_mean": float(
                np.mean(sample_probabilities)
            ),
        }
        for index, name in enumerate(REWARD_COMPONENT_NAMES):
            metrics[f"replay_{name}_mean"] = float(reward_statistics["mean"][index])
            metrics[f"replay_{name}_std"] = float(reward_statistics["std"][index])
        if self.deadline_lagrangian is not None:
            base_reward = self._base_reward(components_t, reward_statistics)
            metrics.update(
                {
                    "deadline_lambda": self.deadline_lambda,
                    "base_reward_mean": float(base_reward.mean().item()),
                    "deadline_constraint_cost_mean": float(
                        deadline_costs_t.mean().item()
                    ),
                    "reliability_lambda": self.reliability_lambda,
                    "reliability_constraint_cost_mean": float(
                        reliability_costs_t.mean().item()
                    ),
                    "reliability_lagrangian_penalty_mean": float(
                        (self.reliability_lambda * reliability_costs_t).mean().item()
                    ),
                    "lagrangian_penalty_mean": float(
                        (self.deadline_lambda * deadline_costs_t).mean().item()
                    ),
                    "training_reward_mean": float(rewards_t.mean().item()),
                }
            )
        return metrics

    def _base_reward(self, components, statistics):
        """Energy/time part of the low reward, before any constraint penalty.

        Subclasses may override this to change how the two components are put
        on a common scale without touching the Lagrangian penalties below.
        """
        weights = torch.as_tensor(self.reward_weights, dtype=torch.float32, device=self.device)
        if self.reward_mode == "sub_deadline_lagrangian":
            # The energy component is already a dimensionless ratio divided
            # by N_w, exactly like the sub-Deadline constraint cost.
            return weights[0] * components[:, 0]
        mean = torch.as_tensor(statistics["mean"], dtype=torch.float32, device=self.device)
        std = torch.as_tensor(statistics["std"], dtype=torch.float32, device=self.device)
        normalized = ((components - mean) / std).clamp(
            -self.reward_component_clip, self.reward_component_clip
        )
        return (
            weights[0] * normalized[:, 0]
            - weights[1] * normalized[:, 1]
            + weights[2] * normalized[:, 2]
        )

    def _compose_rewards(
        self,
        components,
        constraint_flags,
        statistics,
        deadline_constraint_costs=None,
        reliability_constraint_costs=None,
    ):
        del constraint_flags
        weights = torch.as_tensor(self.reward_weights, dtype=torch.float32, device=self.device)
        base_reward = self._base_reward(components, statistics)
        if self.deadline_lagrangian is not None:
            if deadline_constraint_costs is None:
                deadline_constraint_costs = components[:, 2]
            reward = base_reward - self.deadline_lambda * deadline_constraint_costs.clamp_min(0.0)
        else:
            reward = base_reward - weights[3] * components[:, 3].clamp_min(0.0)
        if self.reliability_lagrangian is not None:
            if reliability_constraint_costs is None:
                reliability_constraint_costs = components[:, 4]
            reward = reward - self.reliability_lambda * reliability_constraint_costs.clamp_min(0.0)
        return reward

    def state_dict(self, include_replay=False):
        return {
            "state_mode": self.state_mode,
            "actor_state_dim": self.actor_state_dim,
            "critic_state_dim": self.critic_state_dim,
            "actor_state_dict": self.actor.state_dict(),
            "critic_1_state_dict": self.critic_1.state_dict(),
            "critic_2_state_dict": self.critic_2.state_dict(),
            "target_critic_1_state_dict": self.target_critic_1.state_dict(),
            "target_critic_2_state_dict": self.target_critic_2.state_dict(),
            "reliability_lagrangian": (
                None if self.reliability_lagrangian is None
                else self.reliability_lagrangian.state_dict()
            ),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_1_optimizer_state_dict": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer_state_dict": self.critic_2_optimizer.state_dict(),
            "alpha": float(self.alpha),
            "log_alpha": None if self.log_alpha is None else self.log_alpha.detach().cpu(),
            "alpha_optimizer_state_dict": None if self.alpha_optimizer is None else self.alpha_optimizer.state_dict(),
            "reward_mode": self.reward_mode,
            "deadline_lagrangian": (
                None
                if self.deadline_lagrangian is None
                else self.deadline_lagrangian.state_dict()
            ),
            "replay_state_dict": self.replay.state_dict() if include_replay else None,
        }

    def load_state_dict(self, payload):
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.critic_1.load_state_dict(payload["critic_1_state_dict"])
        self.critic_2.load_state_dict(payload["critic_2_state_dict"])
        self.target_critic_1.load_state_dict(payload["target_critic_1_state_dict"])
        self.target_critic_2.load_state_dict(payload["target_critic_2_state_dict"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        self.critic_1_optimizer.load_state_dict(payload["critic_1_optimizer_state_dict"])
        self.critic_2_optimizer.load_state_dict(payload["critic_2_optimizer_state_dict"])
        self.alpha = float(payload.get("alpha", self.alpha))
        if self.log_alpha is not None and payload.get("log_alpha") is not None:
            self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
        if self.alpha_optimizer is not None and payload.get("alpha_optimizer_state_dict") is not None:
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer_state_dict"])
        if (
            self.deadline_lagrangian is not None
            and payload.get("deadline_lagrangian") is not None
        ):
            self.deadline_lagrangian.load_state_dict(
                payload["deadline_lagrangian"]
            )
        if (
            self.reliability_lagrangian is not None
            and payload.get("reliability_lagrangian") is not None
        ):
            self.reliability_lagrangian.load_state_dict(
                payload["reliability_lagrangian"]
            )
        if payload.get("replay_state_dict") is not None:
            self.replay.load_state_dict(payload["replay_state_dict"])

    def update_target_networks(self, tau=None):
        tau = self.tau if tau is None else float(tau)
        self._soft_update(self.target_critic_1, self.critic_1, tau)
        self._soft_update(self.target_critic_2, self.critic_2, tau)

    @staticmethod
    def _soft_update(target_network, source_network, tau):
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def _grad_to_float(value):
        return float(value.item() if torch.is_tensor(value) else value)
