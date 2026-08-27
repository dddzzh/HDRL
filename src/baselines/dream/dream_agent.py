from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch import nn

from .dream_config import DReaMConfig
from .dream_network import DReaMQNetwork


class DReaMReplayBuffer:
    def __init__(self, capacity, state_dim, seed=42):
        self.capacity, self.state_dim = int(capacity), int(state_dim)
        self.rng = np.random.default_rng(seed)
        self.position, self.size = 0, 0
        self.states = np.zeros((self.capacity, self.state_dim), np.float32)
        self.actions = np.zeros(self.capacity, np.int64)
        self.rewards = np.zeros(self.capacity, np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), np.float32)
        self.dones = np.zeros(self.capacity, np.float32)

    def __len__(self): return self.size

    def add(self, state, action, reward, next_state, done):
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if state.shape != (self.state_dim,) or next_state.shape != (self.state_dim,):
            raise ValueError("Invalid DReaM replay state shape.")
        if not np.isfinite(state).all() or not np.isfinite(next_state).all():
            raise FloatingPointError("DReaM replay received a non-finite state.")
        if not math.isfinite(float(reward)):
            raise FloatingPointError("DReaM replay received a non-finite reward.")
        i = self.position
        self.states[i], self.actions[i], self.rewards[i] = state, int(action), float(reward)
        self.next_states[i], self.dones[i] = next_state, float(done)
        self.position, self.size = (i + 1) % self.capacity, min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.choice(self.size, int(batch_size), replace=False)
        return tuple(x[idx] for x in (self.states, self.actions, self.rewards, self.next_states, self.dones))


class DReaMAgent:
    def __init__(self, state_dim, action_dim, config=None):
        self.config = config if isinstance(config, DReaMConfig) else DReaMConfig.from_mapping(config)
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        if str(self.config.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "DReaM is configured for CUDA, but torch.cuda.is_available() is false. "
                "Install the GPU PyTorch build in conda py310 or explicitly pass --device cpu."
            )
        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.seed)
        self.rng = random.Random(self.config.seed)
        self.online_net = DReaMQNetwork(self.state_dim, self.action_dim, self.config.hidden_dim).to(self.device)
        self.target_net = DReaMQNetwork(self.state_dim, self.action_dim, self.config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        if not hasattr(torch.optim, "NAdam"):
            raise RuntimeError("This PyTorch build lacks torch.optim.NAdam; no silent optimizer substitution is allowed.")
        self.optimizer = torch.optim.NAdam(self.online_net.parameters(), lr=self.config.learning_rate)
        self.replay_buffer = DReaMReplayBuffer(self.config.replay_capacity, self.state_dim, self.config.seed)
        self.epsilon = self.config.epsilon_start
        self.environment_steps = self.gradient_steps = 0

    def select_action(self, state, training=True):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.state_dim,): raise ValueError("Invalid DReaM state shape.")
        if not np.isfinite(state).all():
            raise FloatingPointError("DReaM policy received a non-finite state.")
        if training and self.rng.random() < self.epsilon: return self.rng.randrange(self.action_dim)
        self.online_net.eval()
        with torch.no_grad():
            q = self.online_net(torch.as_tensor(state, device=self.device).unsqueeze(0))
        self._assert_tensor_finite(q, "policy Q")
        q_abs_max = float(q.detach().abs().max().cpu())
        if q_abs_max > self.config.max_abs_q:
            raise FloatingPointError(
                f"DReaM policy Q diverged: max_abs={q_abs_max:.6g} "
                f"> limit={self.config.max_abs_q:.6g}."
            )
        return int(q.argmax(dim=1).item())

    def update(self):
        if len(self.replay_buffer) < self.config.batch_size: return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)
        states = torch.as_tensor(states, device=self.device)
        actions = torch.as_tensor(actions, device=self.device).long().unsqueeze(1)
        rewards = torch.as_tensor(rewards, device=self.device)
        next_states = torch.as_tensor(next_states, device=self.device)
        dones = torch.as_tensor(dones, device=self.device)
        self.online_net.train()
        selected = self.online_net(states).gather(1, actions).squeeze(1)
        self.target_net.eval()
        with torch.no_grad():
            target = rewards + self.config.gamma * (1.0 - dones) * self.target_net(next_states).max(1).values
        self._assert_tensor_finite(selected, "selected Q")
        self._assert_tensor_finite(target, "TD target")
        q_abs_max = float(selected.detach().abs().max().cpu())
        target_abs_max = float(target.detach().abs().max().cpu())
        if max(q_abs_max, target_abs_max) > self.config.max_abs_q:
            raise FloatingPointError(
                "DReaM Q/target divergence detected before optimizer step: "
                f"q_abs_max={q_abs_max:.6g}, target_abs_max={target_abs_max:.6g}, "
                f"limit={self.config.max_abs_q:.6g}."
            )
        loss = nn.functional.huber_loss(
            selected,
            target,
            delta=self.config.huber_delta,
        )
        self._assert_tensor_finite(loss, "Huber loss")
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            self.config.max_grad_norm,
            error_if_nonfinite=True,
        )
        grad_norm_value = float(grad_norm.detach().cpu())
        self.optimizer.step()
        parameter_abs_max = self._parameter_abs_max()
        if parameter_abs_max > self.config.max_abs_parameter:
            raise FloatingPointError(
                "DReaM parameter divergence detected after optimizer step: "
                f"max_abs={parameter_abs_max:.6g} "
                f"> limit={self.config.max_abs_parameter:.6g}."
            )
        self.gradient_steps += 1
        td_abs = (target - selected).detach().abs()
        return {
            "loss": float(loss.detach().cpu()),
            "q_mean": float(selected.detach().mean().cpu()),
            "q_abs_max": q_abs_max,
            "target_mean": float(target.detach().mean().cpu()),
            "target_abs_max": target_abs_max,
            "td_abs_mean": float(td_abs.mean().cpu()),
            "td_abs_max": float(td_abs.max().cpu()),
            "grad_norm_before_clip": grad_norm_value,
            "grad_norm_after_clip": min(grad_norm_value, self.config.max_grad_norm),
            "parameter_abs_max": parameter_abs_max,
            "gradient_steps": self.gradient_steps,
        }

    def after_environment_step(self):
        self.environment_steps += 1
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        updated = self.environment_steps % self.config.target_update_frequency == 0
        if updated:
            self.target_net.load_state_dict(self.online_net.state_dict()); self.target_net.eval()
        return updated

    def checkpoint(self, episode, best_episode_reward):
        return {"online_network": self.online_net.state_dict(), "target_network": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(), "epsilon": self.epsilon,
                "environment_steps": self.environment_steps, "gradient_steps": self.gradient_steps,
                "episode": int(episode), "best_episode_reward": float(best_episode_reward), "config": self.config.as_dict()}

    def replay_state_dict(self):
        buffer = self.replay_buffer
        return {"capacity": buffer.capacity, "state_dim": buffer.state_dim,
                "position": buffer.position, "size": buffer.size, "states": buffer.states,
                "actions": buffer.actions, "rewards": buffer.rewards,
                "next_states": buffer.next_states, "dones": buffer.dones}

    def load_replay_state_dict(self, payload):
        if int(payload["capacity"]) != self.replay_buffer.capacity or int(payload["state_dim"]) != self.state_dim:
            raise ValueError("DReaM replay checkpoint shape mismatch.")
        for name in ("states", "actions", "rewards", "next_states", "dones"):
            getattr(self.replay_buffer, name)[:] = payload[name]
        self.replay_buffer.position = int(payload["position"]); self.replay_buffer.size = int(payload["size"])

    def load_checkpoint(self, payload):
        self.online_net.load_state_dict(payload["online_network"]); self.target_net.load_state_dict(payload["target_network"])
        self.optimizer.load_state_dict(payload["optimizer"]); self.epsilon = float(payload["epsilon"])
        self.environment_steps = int(payload["environment_steps"]); self.gradient_steps = int(payload["gradient_steps"])
        self.target_net.eval()
        self.assert_healthy("loaded checkpoint")

    def assert_healthy(self, context="agent"):
        parameter_abs_max = self._parameter_abs_max()
        if not math.isfinite(parameter_abs_max):
            raise FloatingPointError(f"DReaM {context} contains non-finite parameters.")
        if parameter_abs_max > self.config.max_abs_parameter:
            raise FloatingPointError(
                f"DReaM {context} parameters exceed the safety limit: "
                f"max_abs={parameter_abs_max:.6g} "
                f"> limit={self.config.max_abs_parameter:.6g}."
            )
        return {"parameter_abs_max": parameter_abs_max}

    def _parameter_abs_max(self):
        maxima = []
        for parameter in self.online_net.parameters():
            self._assert_tensor_finite(parameter, "online parameter")
            maxima.append(float(parameter.detach().abs().max().cpu()))
        return max(maxima, default=0.0)

    @staticmethod
    def _assert_tensor_finite(value, label):
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"DReaM {label} contains NaN or Inf.")
