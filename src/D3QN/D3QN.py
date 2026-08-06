from __future__ import annotations

import numpy as np

from src.a2c.a2c import _require_torch
from src.D3QN.buffer import ReplayBuffer
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

    def remember(self, state, action, reward, state_, done, action_mask, next_action_mask):
        self.memory.store_transition(state, action, reward, state_, done, action_mask, next_action_mask)

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation, action_mask, isTrain=True):
        action_mask = np.asarray(action_mask, dtype=bool)
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size <= 0:
            raise ValueError("D3QN action mask contains no valid action.")
        if (np.random.random() < self.epsilon) and isTrain:
            return int(self.rng.choice(valid_actions))
        state = torch.as_tensor(np.asarray([observation], dtype=np.float32), dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_vals = self.q_eval.forward(state).masked_fill(~mask, -1e9)
            return int(torch.argmax(q_vals, dim=-1).item())

    def action_q_value(self, observation, action):
        state = torch.as_tensor(np.asarray([observation], dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(self.q_eval.forward(state)[0, int(action)].item())

    def learn(self):
        sample = self.memory.sample_buffer(self.rng)
        if sample is None:
            return {}
        states, actions, rewards, next_states, terminals, _, next_action_masks = sample
        batch_idx = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
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
        return {
            "loss": float(loss.item()),
            "q_mean": float(q.detach().mean().item()),
            "target_mean": float(target.detach().mean().item()),
            "td_abs_mean": float(td_error.abs().mean().item()),
            "grad_norm_before_clip": grad_norm,
        }
