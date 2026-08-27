from __future__ import annotations

import numpy as np

from src.a2c.a2c import _require_torch
from src.D3QN.buffer import REWARD_COMPONENT_NAMES, ReplayBuffer
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


def _build_mlp(input_dim, hidden_dim, hidden_layers):
    layers = []
    current_dim = int(input_dim)
    for _ in range(max(int(hidden_layers), 1)):
        layers.append(nn.Linear(current_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        current_dim = int(hidden_dim)
    return nn.Sequential(*layers)


class DuelingDeepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, hidden_layers=2):
        _require_torch()
        super().__init__()
        self.backbone = _build_mlp(state_dim, hidden_dim, hidden_layers)
        self.V = nn.Linear(int(hidden_dim), 1)
        self.A = nn.Linear(int(hidden_dim), int(action_dim))

    def forward(self, state):
        x = self.backbone(state)
        value = self.V(x)
        advantage = self.A(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class D3QN:
    def __init__(
        self,
        *,
        state_dim,
        action_dim,
        alpha=3e-4,
        hidden_dim=256,
        hidden_layers=2,
        gamma=0.99,
        tau=0.005,
        epsilon=1.0,
        eps_end=0.05,
        eps_dec=2e-4,
        max_size=100000,
        batch_size=256,
        max_grad_norm=1.0,
        reward_mode="scalar",
        reward_weights=(1.0, 1.0),
        deadline_violation_weight=2.0,
        reward_std_floor=0.05,
        lagrangian_energy_clip=3.0,
        lagrangian_energy_normalization="replay_zscore",
        device="auto",
        seed=42,
    ):
        _require_torch()
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        self.eps_min = float(eps_end)
        self.eps_dec = float(eps_dec)
        self.batch_size = int(batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.action_dim = int(action_dim)
        self.reward_mode = str(reward_mode).strip().lower()
        if self.reward_mode not in {"scalar", "replay_zscore"}:
            raise ValueError("D3QN reward_mode must be 'scalar' or 'replay_zscore'.")
        self.reward_weights = np.asarray(reward_weights, dtype=np.float32)
        if self.reward_weights.shape != (len(REWARD_COMPONENT_NAMES),):
            raise ValueError(
                f"D3QN reward_weights must have shape {(len(REWARD_COMPONENT_NAMES),)}."
            )
        self.deadline_violation_weight = float(deadline_violation_weight)
        self.reward_std_floor = float(reward_std_floor)
        self.lagrangian_energy_clip = float(lagrangian_energy_clip)
        if self.lagrangian_energy_clip <= 0.0:
            raise ValueError("lagrangian_energy_clip must be positive.")
        self.lagrangian_energy_normalization = str(
            lagrangian_energy_normalization
        ).strip().lower()
        if self.lagrangian_energy_normalization not in {
            "none",
            "replay_zscore",
        }:
            raise ValueError(
                "lagrangian_energy_normalization must be 'none' or "
                "'replay_zscore'."
            )
        self.action_space = [index for index in range(self.action_dim)]
        self.rng = np.random.default_rng(int(seed))
        self.device = resolve_agent_device(device)
        self.q_eval = DuelingDeepQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)
        self.q_target = DuelingDeepQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_eval.parameters(), lr=float(alpha))
        self.memory = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=max_size,
            batch_size=batch_size,
        )
        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        tau = self.tau if tau is None else float(tau)
        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params.data + (1.0 - tau) * q_target_params.data)

    def remember(
        self,
        state,
        action,
        reward,
        state_,
        done,
        action_mask,
        next_action_mask,
        *,
        reward_components=None,
        deadline_crossed=0.0,
        deadline_violation=0.0,
        base_energy_reward=None,
        deadline_constraint_cost=0.0,
    ):
        self.memory.store_transition(
            state,
            action,
            reward,
            state_,
            done,
            action_mask,
            next_action_mask,
            reward_components=reward_components,
            deadline_crossed=deadline_crossed,
            deadline_violation=deadline_violation,
            base_energy_reward=base_energy_reward,
            deadline_constraint_cost=deadline_constraint_cost,
        )

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation, action_mask, isTrain=True):
        q_values = self.action_q_values(observation)
        return self.choose_action_from_values(q_values, action_mask, isTrain=isTrain)

    def action_q_values(self, observation):
        """Return every discrete action value for candidate re-ranking."""
        state = torch.as_tensor(
            np.asarray([observation], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            return self.q_eval.forward(state)[0].detach().cpu().numpy().astype(
                np.float64
            )

    def choose_action_from_values(self, q_values, action_mask, isTrain=True):
        action_mask = np.asarray(action_mask, dtype=bool)
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size <= 0:
            raise ValueError("D3QN action mask contains no valid action.")
        if (np.random.random() < self.epsilon) and isTrain:
            return int(self.rng.choice(valid_actions))
        q_values = np.asarray(q_values, dtype=np.float64)
        if q_values.shape != (self.action_dim,) or not np.isfinite(q_values).all():
            raise ValueError(
                "D3QN candidate values must be a finite vector with shape "
                f"{(self.action_dim,)}."
            )
        masked = np.where(action_mask, q_values, -np.inf)
        return int(np.argmax(masked))

    def action_q_value(self, observation, action):
        state = torch.as_tensor(np.asarray([observation], dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(self.q_eval.forward(state)[0, int(action)].item())

    def learn(self, reward_statistics=None, deadline_lambda=None):
        sample = self.memory.sample_buffer(self.rng)
        if sample is None:
            return {}
        (
            states,
            actions,
            scalar_rewards,
            base_energy_rewards,
            deadline_constraint_costs,
            reward_components,
            deadline_crossed,
            deadline_violation,
            next_states,
            terminals,
            _,
            next_action_masks,
        ) = sample
        batch_idx = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        if deadline_lambda is not None:
            if (
                self.lagrangian_energy_normalization == "replay_zscore"
                and reward_statistics is None
            ):
                reward_statistics = self.memory.base_energy_reward_statistics(
                    self.reward_std_floor
                )
            elif self.lagrangian_energy_normalization == "none":
                reward_statistics = None
            rewards_tensor = self._compose_lagrangian_rewards(
                base_energy_rewards,
                deadline_constraint_costs,
                deadline_lambda,
                reward_statistics,
            )
        elif self.reward_mode == "replay_zscore":
            if reward_statistics is None:
                reward_statistics = self.memory.reward_statistics(self.reward_std_floor)
            rewards_tensor = self._compose_component_rewards(
                reward_components,
                deadline_crossed,
                deadline_violation,
                reward_statistics,
            )
        else:
            rewards_tensor = torch.as_tensor(
                scalar_rewards, dtype=torch.float32, device=self.device
            )
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        terminals_tensor = torch.as_tensor(terminals, dtype=torch.bool, device=self.device)
        next_masks_tensor = torch.as_tensor(next_action_masks, dtype=torch.bool, device=self.device)

        q = self.q_eval.forward(states_tensor)[batch_idx, actions_tensor]
        with torch.no_grad():
            q_eval_next = self.q_eval.forward(next_states_tensor).masked_fill(~next_masks_tensor, -1e9)
            max_actions = torch.argmax(q_eval_next, dim=-1)
            q_target_next = self.q_target.forward(next_states_tensor).masked_fill(~next_masks_tensor, -1e9)
            next_q = q_target_next[batch_idx, max_actions]
            no_valid_next = next_masks_tensor.sum(dim=-1) <= 0
            next_q = torch.where(terminals_tensor | no_valid_next, torch.zeros_like(next_q), next_q)
            target = rewards_tensor + self.gamma * next_q

        loss = F.mse_loss(q, target.detach())
        if not bool(torch.isfinite(loss).all().item()):
            raise FloatingPointError("D3QN loss contains NaN or Inf.")
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q_eval.parameters(), self.max_grad_norm)
        grad_norm = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
        self.optimizer.step()
        self.update_network_parameters()
        self.decrement_epsilon()
        td_error = (target - q).detach()
        metrics = {
            "loss": float(loss.item()),
            "q_mean": float(q.detach().mean().item()),
            "target_mean": float(target.detach().mean().item()),
            "td_abs_mean": float(td_error.abs().mean().item()),
            "grad_norm_before_clip": grad_norm,
            "reward_mean": float(rewards_tensor.mean().item()),
            "reward_std": float(rewards_tensor.std(unbiased=False).item()),
            "deadline_crossing_rate": float(np.mean(deadline_crossed)),
            "deadline_violation_mean": float(np.mean(deadline_violation)),
        }
        if deadline_lambda is not None:
            base_energy = torch.as_tensor(
                base_energy_rewards, dtype=torch.float32, device=self.device
            )
            if reward_statistics is None:
                normalized_base_energy = base_energy
                replay_energy_mean = 0.0
                replay_energy_std = 1.0
            else:
                normalized_base_energy = (
                    (base_energy - float(reward_statistics["mean"]))
                    / max(float(reward_statistics["std"]), EPS)
                ).clamp(
                    -self.lagrangian_energy_clip,
                    self.lagrangian_energy_clip,
                )
                replay_energy_mean = float(reward_statistics["mean"])
                replay_energy_std = float(reward_statistics["std"])
            deadline_cost = torch.as_tensor(
                deadline_constraint_costs,
                dtype=torch.float32,
                device=self.device,
            )
            metrics.update(
                {
                    "deadline_lambda": float(deadline_lambda),
                    "base_energy_reward_mean": float(base_energy.mean().item()),
                    "base_energy_zscore_mean": float(
                        normalized_base_energy.mean().item()
                    ),
                    "lagrangian_energy_normalization_none": float(
                        self.lagrangian_energy_normalization == "none"
                    ),
                    "replay_base_energy_mean": float(
                        replay_energy_mean
                    ),
                    "replay_base_energy_std": float(
                        replay_energy_std
                    ),
                    "deadline_constraint_cost_mean": float(
                        deadline_cost.mean().item()
                    ),
                    "lagrangian_penalty_mean": float(
                        (float(deadline_lambda) * deadline_cost).mean().item()
                    ),
                    "training_reward_mean": float(rewards_tensor.mean().item()),
                }
            )
        if self.reward_mode == "replay_zscore":
            for index, name in enumerate(REWARD_COMPONENT_NAMES):
                metrics[f"replay_{name}_mean"] = float(
                    reward_statistics["mean"][index]
                )
                metrics[f"replay_{name}_std"] = float(
                    reward_statistics["std"][index]
                )
        return metrics

    def _compose_lagrangian_rewards(
        self,
        base_energy_rewards,
        deadline_constraint_costs,
        deadline_lambda,
        reward_statistics=None,
    ):
        deadline_lambda = float(deadline_lambda)
        if not np.isfinite(deadline_lambda) or deadline_lambda < 0.0:
            raise ValueError("Current Deadline lambda must be finite and non-negative.")
        base_energy = torch.as_tensor(
            base_energy_rewards, dtype=torch.float32, device=self.device
        )
        if reward_statistics is not None:
            energy_mean = float(reward_statistics["mean"])
            energy_std = max(float(reward_statistics["std"]), EPS)
            base_energy = ((base_energy - energy_mean) / energy_std).clamp(
                -self.lagrangian_energy_clip,
                self.lagrangian_energy_clip,
            )
        deadline_cost = torch.as_tensor(
            deadline_constraint_costs, dtype=torch.float32, device=self.device
        )
        rewards = base_energy - deadline_lambda * deadline_cost
        if not bool(torch.isfinite(rewards).all().item()):
            raise FloatingPointError("D3QN Lagrangian reward contains NaN or Inf.")
        return rewards

    def _compose_component_rewards(
        self,
        reward_components,
        deadline_crossed,
        deadline_violation,
        reward_statistics,
    ):
        components = torch.as_tensor(
            reward_components, dtype=torch.float32, device=self.device
        )
        mean = torch.as_tensor(
            reward_statistics["mean"], dtype=torch.float32, device=self.device
        )
        std = torch.as_tensor(
            reward_statistics["std"], dtype=torch.float32, device=self.device
        )
        normalized = (components - mean) / std
        weights = torch.as_tensor(
            self.reward_weights, dtype=torch.float32, device=self.device
        )
        crossed = torch.as_tensor(
            deadline_crossed, dtype=torch.float32, device=self.device
        )
        violation = torch.as_tensor(
            deadline_violation, dtype=torch.float32, device=self.device
        )
        rewards = (
            normalized.matmul(weights)
            - self.deadline_violation_weight * violation
        )
        if not bool(torch.isfinite(rewards).all().item()):
            raise FloatingPointError("D3QN replay-normalized reward contains NaN or Inf.")
        return rewards
