from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from .config import DQNEdgeConfig
from .network import ContextAttention, ObjectiveQNetwork


class DQNEdgeReplayBuffer:
    def __init__(self, capacity, state_dim, seed=42):
        self.capacity, self.state_dim = int(capacity), int(state_dim)
        self.rng = np.random.default_rng(seed)
        self.position = self.size = 0
        self.states = np.zeros((self.capacity, self.state_dim), np.float32)
        self.actions = np.zeros(self.capacity, np.int64)
        self.time_rewards = np.zeros(self.capacity, np.float32)
        self.energy_rewards = np.zeros(self.capacity, np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), np.float32)
        self.dones = np.zeros(self.capacity, np.float32)
        self.contexts = np.zeros((self.capacity, 2), np.float32)
        self.attention_targets = np.full((self.capacity, 2), 0.5, np.float32)

    def __len__(self):
        return self.size

    def add(self, state, action, time_reward, energy_reward, next_state, done, context, attention_target):
        index = self.position
        self.states[index] = np.asarray(state, np.float32)
        self.actions[index] = int(action)
        self.time_rewards[index] = float(time_reward)
        self.energy_rewards[index] = float(energy_reward)
        self.next_states[index] = np.asarray(next_state, np.float32)
        self.dones[index] = float(done)
        self.contexts[index] = np.asarray(context, np.float32)
        self.attention_targets[index] = np.asarray(attention_target, np.float32)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = self.rng.choice(self.size, int(batch_size), replace=False)
        arrays = (self.states, self.actions, self.time_rewards, self.energy_rewards,
                  self.next_states, self.dones, self.contexts, self.attention_targets)
        return tuple(array[indices] for array in arrays)


class DQNEdgeAgent:
    def __init__(self, state_dim, action_dim, config=None):
        self.config = config if isinstance(config, DQNEdgeConfig) else DQNEdgeConfig.from_mapping(config)
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        if str(self.config.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("DQN-Edge requested CUDA, but CUDA is unavailable.")
        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.seed)
        self.rng = random.Random(self.config.seed)
        args = (self.state_dim, self.action_dim, self.config.hidden_dim)
        self.time_online, self.time_target = ObjectiveQNetwork(*args).to(self.device), ObjectiveQNetwork(*args).to(self.device)
        self.energy_online, self.energy_target = ObjectiveQNetwork(*args).to(self.device), ObjectiveQNetwork(*args).to(self.device)
        self.time_target.load_state_dict(self.time_online.state_dict())
        self.energy_target.load_state_dict(self.energy_online.state_dict())
        # Per candidate VM: [Q_latency, Q_energy, mean_latency, total_energy].
        self.attention = ContextAttention(4, self.config.attention_hidden_dim).to(self.device)
        self.time_optimizer = torch.optim.Adam(self.time_online.parameters(), lr=self.config.learning_rate)
        self.energy_optimizer = torch.optim.Adam(self.energy_online.parameters(), lr=self.config.learning_rate)
        self.attention_optimizer = torch.optim.Adam(self.attention.parameters(), lr=self.config.attention_learning_rate)
        self.replay_buffer = DQNEdgeReplayBuffer(self.config.replay_capacity, self.state_dim, self.config.seed)
        self.epsilon = float(self.config.epsilon_start)
        self.environment_steps = self.gradient_steps = 0
        self.smoothed_weights = np.full((self.action_dim, 2), 0.5, dtype=np.float32)

    def select_action(self, state, context, training=True):
        if training and self.rng.random() < self.epsilon:
            action = self.rng.randrange(self.action_dim)
            return action, self.smoothed_weights[action].copy()
        state_t = torch.as_tensor(state, device=self.device).unsqueeze(0)
        context_t = torch.as_tensor(context, device=self.device).unsqueeze(0)
        with torch.no_grad():
            time_q, energy_q = self.time_online(state_t), self.energy_online(state_t)
            repeated_context = context_t.unsqueeze(1).expand(-1, self.action_dim, -1)
            attention_input = torch.cat(
                (time_q.unsqueeze(-1), energy_q.unsqueeze(-1), repeated_context), dim=-1
            )
            weights = self.attention(attention_input).squeeze(0).cpu().numpy()
        alpha = self.config.attention_ema_alpha
        self.smoothed_weights = alpha * weights + (1.0 - alpha) * self.smoothed_weights
        smoothed = torch.as_tensor(self.smoothed_weights, device=self.device)
        combined = smoothed[:, 0].unsqueeze(0) * time_q + smoothed[:, 1].unsqueeze(0) * energy_q
        action = int(combined.argmax(1).item())
        return action, self.smoothed_weights[action].copy()

    def update(self):
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        batch = self.replay_buffer.sample(self.config.batch_size)
        states, actions, time_rewards, energy_rewards, next_states, dones, contexts, targets = (
            torch.as_tensor(value, device=self.device) for value in batch
        )
        actions = actions.long().unsqueeze(1)
        losses = []
        for online, target, optimizer, rewards in (
            (self.time_online, self.time_target, self.time_optimizer, time_rewards),
            (self.energy_online, self.energy_target, self.energy_optimizer, energy_rewards),
        ):
            predicted = online(states).gather(1, actions).squeeze(1)
            with torch.no_grad():
                next_action = online(next_states).argmax(1, keepdim=True)
                bootstrap = target(next_states).gather(1, next_action).squeeze(1)
                expected = rewards + self.config.gamma * (1.0 - dones) * bootstrap
            loss = nn.functional.mse_loss(predicted, expected)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), self.config.max_grad_norm)
            optimizer.step(); losses.append(loss)
        with torch.no_grad():
            selected_time_q = self.time_online(states).gather(1, actions).squeeze(1)
            selected_energy_q = self.energy_online(states).gather(1, actions).squeeze(1)
        attention_input = torch.cat(
            (selected_time_q.unsqueeze(1), selected_energy_q.unsqueeze(1), contexts), dim=1
        )
        attention_loss = nn.functional.mse_loss(self.attention(attention_input), targets)
        self.attention_optimizer.zero_grad(); attention_loss.backward()
        nn.utils.clip_grad_norm_(self.attention.parameters(), self.config.max_grad_norm)
        self.attention_optimizer.step()
        self.gradient_steps += 1
        return {"time_loss": float(losses[0].detach().cpu()),
                "energy_loss": float(losses[1].detach().cpu()),
                "attention_loss": float(attention_loss.detach().cpu())}

    def after_environment_step(self):
        self.environment_steps += 1
        self.epsilon = max(self.config.epsilon_min, self.epsilon - self.config.epsilon_decay_per_step)
        if self.environment_steps % self.config.target_update_frequency == 0:
            self.time_target.load_state_dict(self.time_online.state_dict())
            self.energy_target.load_state_dict(self.energy_online.state_dict())

    def checkpoint(self, episode, best_episode_reward):
        return {
            "time_online": self.time_online.state_dict(), "time_target": self.time_target.state_dict(),
            "energy_online": self.energy_online.state_dict(), "energy_target": self.energy_target.state_dict(),
            "attention": self.attention.state_dict(), "time_optimizer": self.time_optimizer.state_dict(),
            "energy_optimizer": self.energy_optimizer.state_dict(), "attention_optimizer": self.attention_optimizer.state_dict(),
            "epsilon": self.epsilon, "smoothed_weights": self.smoothed_weights,
            "environment_steps": self.environment_steps, "gradient_steps": self.gradient_steps,
            "episode": int(episode), "best_episode_reward": float(best_episode_reward),
            "state_dim": self.state_dim, "action_dim": self.action_dim, "config": self.config.as_dict(),
        }

    def load_checkpoint(self, payload):
        for name in ("time_online", "time_target", "energy_online", "energy_target", "attention"):
            getattr(self, name).load_state_dict(payload[name])
        for name in ("time_optimizer", "energy_optimizer", "attention_optimizer"):
            getattr(self, name).load_state_dict(payload[name])
        self.epsilon = float(payload["epsilon"])
        self.smoothed_weights = np.asarray(payload["smoothed_weights"], np.float32)
        self.environment_steps, self.gradient_steps = int(payload["environment_steps"]), int(payload["gradient_steps"])
