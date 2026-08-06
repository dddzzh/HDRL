from __future__ import annotations

import copy

import numpy as np

from src.a2c.a2c import _require_torch
from src.device_utils import resolve_agent_device
from src.ppo.hierarchical_agent import _assert_finite_tensor, _build_mlp
from src.ppo.hierarchical_state import HierarchicalStateEncoder

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
    "time_increase",
    "deadline_violation",
    "reliability_violation",
)
CONSTRAINT_FLAG_NAMES = ("deadline_crossed", "reliability_crossed")


class LowSACReplayBuffer:
    def __init__(self, state_dim, max_size=100000, batch_size=128, action_low=0.0):
        self.state_dim = int(state_dim)
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        self.action_low = float(action_low)
        self.mem_count = 0
        self.states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, 1), dtype=np.float32)
        self.reward_components = np.zeros(
            (self.max_size, len(REWARD_COMPONENT_NAMES)), dtype=np.float32
        )
        self.constraint_flags = np.zeros(
            (self.max_size, len(CONSTRAINT_FLAG_NAMES)), dtype=np.float32
        )
        self.next_states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=bool)

    @property
    def size(self):
        return min(self.mem_count, self.max_size)

    def store_transition(self, state, action, reward_components, next_state, done):
        index = self.mem_count % self.max_size
        self.states[index] = np.asarray(state, dtype=np.float32)
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
        self.next_states[index] = np.asarray(next_state, dtype=np.float32)
        self.dones[index] = bool(done)
        self.mem_count += 1

    def ready(self):
        return self.size >= self.batch_size

    def sample_buffer(self, rng):
        if not self.ready():
            return None
        indices = rng.choice(self.size, self.batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.reward_components[indices],
            self.constraint_flags[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def state_dict(self):
        return {
            "state_dim": self.state_dim,
            "action_dim": 1,
            "max_size": self.max_size,
            "batch_size": self.batch_size,
            "action_low": self.action_low,
            "mem_count": self.mem_count,
            "states": self.states,
            "actions": self.actions,
            "reward_components": self.reward_components,
            "constraint_flags": self.constraint_flags,
            "next_states": self.next_states,
            "dones": self.dones,
        }

    def load_state_dict(self, payload):
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

    def reward_statistics(self, std_floor):
        if self.size <= 0:
            return None
        values = self.reward_components[: self.size]
        return {
            "mean": values.mean(axis=0, dtype=np.float64).astype(np.float32),
            "std": np.maximum(values.std(axis=0, dtype=np.float64), float(std_floor)).astype(
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
        self.device = resolve_agent_device(config.get("device", "auto"))
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.learning_rate = float(config.get("learning_rate", 3e-4))
        self.batch_size = int(config.get("batch_size", 128))
        self.training_per_episode = int(config.get("training_per_episode", 10))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.rho_min = float(config.get("rho_min", 0.1))
        reward_config = dict(config.get("reward", {}))
        self.reward_weights = np.asarray(
            [
                reward_config.get("energy_weight", 1.0),
                reward_config.get("time_weight", 1.0),
                reward_config.get("deadline_severity_weight", 2.0),
                reward_config.get("reliability_severity_weight", 2.0),
            ],
            dtype=np.float32,
        )
        self.constraint_fixed_penalties = np.asarray(
            [
                reward_config.get("deadline_fixed_penalty", 5.0),
                reward_config.get("reliability_fixed_penalty", 5.0),
            ],
            dtype=np.float32,
        )
        self.reward_component_clip = float(reward_config.get("component_clip", 3.0))
        self.reward_std_floor = float(reward_config.get("std_floor", 0.05))
        if np.any(self.reward_weights < 0.0) or np.any(self.constraint_fixed_penalties < 0.0):
            raise ValueError("low SAC reward weights must be non-negative.")
        if self.reward_component_clip <= 0.0 or self.reward_std_floor <= 0.0:
            raise ValueError("low SAC reward component_clip and std_floor must be positive.")
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
            self.encoder.low_state_dim,
            self.action_dim,
            hidden_dim,
            hidden_layers,
            log_std_min,
            log_std_max,
            action_low=self.rho_min,
            action_high=1.0,
        ).to(self.device)
        self.critic_1 = LowSACCritic(self.encoder.low_state_dim, self.action_dim, hidden_dim, hidden_layers).to(self.device)
        self.critic_2 = LowSACCritic(self.encoder.low_state_dim, self.action_dim, hidden_dim, hidden_layers).to(self.device)
        self.target_critic_1 = LowSACCritic(self.encoder.low_state_dim, self.action_dim, hidden_dim, hidden_layers).to(self.device)
        self.target_critic_2 = LowSACCritic(self.encoder.low_state_dim, self.action_dim, hidden_dim, hidden_layers).to(self.device)
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
            self.encoder.low_state_dim,
            max_size=int(config.get("replay_size", config.get("max_size", 100000))),
            batch_size=self.batch_size,
            action_low=self.rho_min,
        )
        self.update_target_networks(tau=1.0)

    def build_low_state(self, high_observation, task_index, server_index, safety_metrics=None):
        return self.encoder.encode_low_state(high_observation, task_index, server_index, safety_metrics=safety_metrics)

    def select_frequency(self, high_observation, task_index, server_index, deterministic=False, safety_metrics=None):
        safety_metrics = dict(safety_metrics or {})
        low_state = self.build_low_state(high_observation, task_index, server_index, safety_metrics=safety_metrics)
        rho_safe = min(1.0, max(self.rho_min, float(safety_metrics.get("rho_safe", self.rho_min))))
        rho_hardware_min = min(
            1.0,
            max(self.rho_min, float(safety_metrics.get("rho_hardware_min", self.rho_min))),
        )
        low_action_feasible = bool(safety_metrics.get("low_action_feasible", True))
        has_action_span = bool(rho_hardware_min < 1.0 - 1e-6)
        state_tensor = torch.as_tensor(low_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if has_action_span:
            with torch.no_grad():
                action_tensor, log_prob, mean, log_std = self.actor.sample(
                    state_tensor, deterministic=deterministic
                )
                value = float(
                    torch.min(
                        self.critic_1(state_tensor, action_tensor),
                        self.critic_2(state_tensor, action_tensor),
                    ).item()
                )
            # The SAC action is the actual normalized DVFS frequency rho.
            # A final clip only enforces a server-specific hardware minimum.
            rho = float(action_tensor.item())
            rho = float(min(1.0, max(rho_hardware_min, rho)))
            action_rho = rho
            log_prob_value = float(log_prob.item())
            entropy_value = -log_prob_value
            policy_mean = float(mean.item())
            policy_log_std = float(log_std.item())
            actor_update_mask = True
        else:
            action_rho = 1.0
            rho = 1.0
            log_prob_value = 0.0
            entropy_value = 0.0
            value = 0.0
            policy_mean = 0.0
            policy_log_std = 0.0
            actor_update_mask = False
        rho = float(min(1.0, max(rho_hardware_min, rho)))
        return {
            "state": low_state,
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
            "low_action_feasible": low_action_feasible,
            "has_action_span": has_action_span,
            "actor_update_mask": actor_update_mask,
            "forced_max_frequency": not has_action_span,
            "low_algorithm": "sac",
        }

    def remember_transition(self, state, action_rho, reward_components, next_state, done):
        self.replay.store_transition(state, action_rho, reward_components, next_state, done)

    def update(self, buffer=None, learning_rate_scale=1.0):
        totals = {"replay_size": float(self.replay.size)}
        updates = 0
        old_lrs = []
        for optimizer in (self.actor_optimizer, self.critic_1_optimizer, self.critic_2_optimizer):
            for group in optimizer.param_groups:
                old_lrs.append((group, group["lr"]))
                group["lr"] = self.learning_rate * float(learning_rate_scale)
        reward_statistics = self.replay.reward_statistics(self.reward_std_floor)
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
        sample = self.replay.sample_buffer(self.rng)
        if sample is None:
            return {}
        (
            states,
            actions,
            reward_components,
            constraint_flags,
            next_states,
            dones,
        ) = sample
        if reward_statistics is None:
            reward_statistics = self.replay.reward_statistics(self.reward_std_floor)
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        components_t = torch.as_tensor(reward_components, dtype=torch.float32, device=self.device)
        flags_t = torch.as_tensor(constraint_flags, dtype=torch.float32, device=self.device)
        rewards_t = self._compose_rewards(components_t, flags_t, reward_statistics)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            next_actions, next_log_probs, _, _ = self.actor.sample(next_states_t)
            next_q = torch.min(
                self.target_critic_1(next_states_t, next_actions),
                self.target_critic_2(next_states_t, next_actions),
            )
            next_values = next_q - self.alpha * next_log_probs
            target = rewards_t + self.gamma * (~dones_t).float() * next_values
        q1 = self.critic_1(states_t, actions_t)
        q2 = self.critic_2(states_t, actions_t)
        critic_1_loss = F.mse_loss(q1, target)
        critic_2_loss = F.mse_loss(q2, target)
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
            self.critic_1(states_t, policy_actions),
            self.critic_2(states_t, policy_actions),
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
        }
        for index, name in enumerate(REWARD_COMPONENT_NAMES):
            metrics[f"replay_{name}_mean"] = float(reward_statistics["mean"][index])
            metrics[f"replay_{name}_std"] = float(reward_statistics["std"][index])
        return metrics

    def _compose_rewards(self, components, constraint_flags, statistics):
        mean = torch.as_tensor(statistics["mean"], dtype=torch.float32, device=self.device)
        std = torch.as_tensor(statistics["std"], dtype=torch.float32, device=self.device)
        normalized = ((components - mean) / std).clamp(
            -self.reward_component_clip, self.reward_component_clip
        )
        weights = torch.as_tensor(self.reward_weights, dtype=torch.float32, device=self.device)
        fixed_penalties = torch.as_tensor(
            self.constraint_fixed_penalties, dtype=torch.float32, device=self.device
        )
        return (
            weights[0] * normalized[:, 0]
            - weights[1] * normalized[:, 1]
            - weights[2] * F.relu(normalized[:, 2])
            - weights[3] * F.relu(normalized[:, 3])
            - (constraint_flags * fixed_penalties).sum(dim=-1)
        )

    def state_dict(self, include_replay=False):
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_1_state_dict": self.critic_1.state_dict(),
            "critic_2_state_dict": self.critic_2.state_dict(),
            "target_critic_1_state_dict": self.target_critic_1.state_dict(),
            "target_critic_2_state_dict": self.target_critic_2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_1_optimizer_state_dict": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer_state_dict": self.critic_2_optimizer.state_dict(),
            "alpha": float(self.alpha),
            "log_alpha": None if self.log_alpha is None else self.log_alpha.detach().cpu(),
            "alpha_optimizer_state_dict": None if self.alpha_optimizer is None else self.alpha_optimizer.state_dict(),
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
