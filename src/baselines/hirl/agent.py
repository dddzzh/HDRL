from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch
from torch import nn

from .config import HIRLConfig
from .networks import TwinCritic, mlp
from .replay import FailureAwareReplay


class HIRLAgent:
    """TD3 power control followed by DDQN server allocation.

    The paper-specific state is deliberately replaced by the project's shared
    state.  The DDQN receives the TD3 rho as one additional coordination value.
    """

    CHECKPOINT_FORMAT = "adapted_hirl_td3_ddqn_v1"

    def __init__(self, state_dim, num_servers, config=None):
        self.config = config if isinstance(config, HIRLConfig) else HIRLConfig.from_mapping(config)
        self.state_dim = int(state_dim)
        self.num_servers = int(num_servers)
        if self.num_servers <= 0:
            raise ValueError("HIRL requires at least one server action.")
        requested = self.config.device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("HIRL requested CUDA, but CUDA is unavailable.")
        self.device = torch.device(requested)
        torch.manual_seed(self.config.seed)
        self.rng = random.Random(self.config.seed)

        hidden = self.config.hidden_dim
        self.actor = mlp(self.state_dim, 1, hidden, nn.Tanh).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).eval()
        self.critic = TwinCritic(self.state_dim, 1, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).eval()
        self.q_network = mlp(self.state_dim + 1, self.num_servers, hidden).to(self.device)
        self.q_target = copy.deepcopy(self.q_network).eval()

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.config.ddqn_learning_rate
        )
        self.td3_replay = FailureAwareReplay(
            self.config.td3_replay_size, self.state_dim, (1,), self.config.seed
        )
        self.ddqn_replay = FailureAwareReplay(
            self.config.ddqn_replay_size, self.state_dim + 1, (1,), self.config.seed + 1
        )
        self.environment_steps = self.gradient_steps = 0
        self.current_load = 0.0

    def select_action(self, state, action_mask, load, training=True):
        state = self._state(state, self.state_dim)
        action_mask = np.asarray(action_mask, dtype=bool)
        if action_mask.shape != (self.num_servers,) or not action_mask.any():
            raise ValueError("HIRL requires a non-empty server action mask.")
        load = float(np.clip(load, 0.0, 1.0))
        self.current_load = load
        state_tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            normalized_rho = float(self.actor(state_tensor).item())
        if training:
            # Paper Eq. (adaptive noise): more perturbation as measured load grows.
            sigma = self.config.exploration_noise * (1.0 + self.config.load_noise_alpha * load)
            normalized_rho += self.rng.gauss(0.0, sigma)
        normalized_rho = float(np.clip(normalized_rho, -1.0, 1.0))
        rho = self._rho_from_normalized(normalized_rho)

        ddqn_state = np.concatenate([state, np.array([rho], np.float32)])
        epsilon = self.epsilon(load) if training else 0.0
        valid = np.flatnonzero(action_mask)
        if training and self.rng.random() < epsilon:
            server_index = int(self.rng.choice(valid.tolist()))
        else:
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(
                    torch.as_tensor(ddqn_state, device=self.device).unsqueeze(0)
                ).squeeze(0)
            masked = q_values.detach().cpu().numpy()
            masked[~action_mask] = -np.inf
            server_index = int(np.argmax(masked))
        return {
            "state": state,
            "ddqn_state": ddqn_state.astype(np.float32),
            "normalized_rho": normalized_rho,
            "rho": rho,
            "server_index": server_index,
            "action_mask": action_mask,
            "load": load,
            "epsilon": epsilon,
        }

    def remember(self, decision, upper_reward, lower_reward, next_state, done, failure):
        next_state = self._state(next_state, self.state_dim)
        # The next DDQN coordination value is generated deterministically by the
        # target actor, preserving TD3 -> DDQN ordering in the bootstrap target.
        with torch.no_grad():
            next_action = self.actor_target(
                torch.as_tensor(next_state, device=self.device).unsqueeze(0)
            ).item()
        next_rho = self._rho_from_normalized(float(next_action))
        next_ddqn_state = np.concatenate([next_state, np.array([next_rho], np.float32)])
        self.td3_replay.add(
            decision["state"], [decision["normalized_rho"]], upper_reward,
            next_state, done, failure,
        )
        self.ddqn_replay.add(
            decision["ddqn_state"], [decision["server_index"]], lower_reward,
            next_ddqn_state, done, failure,
        )
        self.environment_steps += 1

    def update(self):
        if min(len(self.td3_replay), len(self.ddqn_replay)) < self.config.batch_size:
            return None
        td3 = self._update_td3()
        ddqn = self._update_ddqn()
        self.gradient_steps += 1
        if self.gradient_steps % self.config.target_update_frequency == 0:
            self.q_target.load_state_dict(self.q_network.state_dict())
            self.q_target.eval()
        return {**td3, **ddqn, "gradient_steps": self.gradient_steps}

    def _update_td3(self):
        indices, weights, batch = self.td3_replay.sample(
            self.config.batch_size, self.config.priority_alpha, self.config.priority_beta
        )
        states, actions, rewards, next_states, dones, _ = self._tensors(batch)
        weights = torch.as_tensor(weights, device=self.device).unsqueeze(1)
        with torch.no_grad():
            noise = torch.randn_like(actions) * self.config.target_noise
            noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target = rewards.unsqueeze(1) + self.config.gamma * (1.0 - dones.unsqueeze(1)) * torch.minimum(target_q1, target_q2)
        q1, q2 = self.critic(states, actions)
        td_error = target - torch.minimum(q1, q2)
        critic_loss = (weights * ((q1 - target).square() + (q2 - target).square())).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()
        self.td3_replay.update_priorities(
            indices, td_error.detach().squeeze(1).cpu().numpy(), self.config.failure_priority
        )

        actor_loss_value = None
        if (self.gradient_steps + 1) % self.config.policy_delay == 0:
            actor_loss = -self.critic.first(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            actor_loss_value = float(actor_loss.detach().cpu())
        return {
            "td3_critic_loss": float(critic_loss.detach().cpu()),
            "td3_actor_loss": actor_loss_value,
            "td3_td_abs_mean": float(td_error.detach().abs().mean().cpu()),
        }

    def _update_ddqn(self):
        adaptive_learning_rate = self.config.ddqn_learning_rate * (
            1.0 + self.config.load_learning_rate_alpha * self.current_load
        )
        for group in self.q_optimizer.param_groups:
            group["lr"] = adaptive_learning_rate
        indices, weights, batch = self.ddqn_replay.sample(
            self.config.batch_size, self.config.priority_alpha, self.config.priority_beta
        )
        states, actions, rewards, next_states, dones, _ = self._tensors(batch)
        action_indices = actions.long()
        selected = self.q_network(states).gather(1, action_indices)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_values = self.q_target(next_states).gather(1, next_actions)
            target = rewards.unsqueeze(1) + self.config.gamma * (1.0 - dones.unsqueeze(1)) * next_values
        td_error = target - selected
        weights = torch.as_tensor(weights, device=self.device).unsqueeze(1)
        loss = (weights * nn.functional.smooth_l1_loss(selected, target, reduction="none")).mean()
        self.q_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.q_optimizer.step()
        self.ddqn_replay.update_priorities(
            indices, td_error.detach().squeeze(1).cpu().numpy(), self.config.failure_priority
        )
        return {
            "ddqn_loss": float(loss.detach().cpu()),
            "ddqn_td_abs_mean": float(td_error.detach().abs().mean().cpu()),
            "ddqn_learning_rate": float(adaptive_learning_rate),
        }

    def epsilon(self, load):
        cfg = self.config
        return cfg.epsilon_min + (cfg.epsilon_max - cfg.epsilon_min) * math.exp(-cfg.epsilon_beta * float(load))

    def checkpoint(self, episode, best_episode_reward, include_replay=True):
        payload = {
            "checkpoint_format": self.CHECKPOINT_FORMAT,
            "episode": int(episode), "best_episode_reward": float(best_episode_reward),
            "state_dim": self.state_dim, "num_servers": self.num_servers,
            "config": self.config.as_dict(), "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "current_load": float(self.current_load),
            "actor": self.actor.state_dict(), "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(), "critic_target": self.critic_target.state_dict(),
            "q_network": self.q_network.state_dict(), "q_target": self.q_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
        }
        if include_replay:
            payload["td3_replay"] = self.td3_replay.state_dict()
            payload["ddqn_replay"] = self.ddqn_replay.state_dict()
        return payload

    def load_checkpoint(self, payload, load_replay=False):
        if payload.get("checkpoint_format") != self.CHECKPOINT_FORMAT:
            raise ValueError("Unsupported HIRL checkpoint format.")
        if int(payload["state_dim"]) != self.state_dim or int(payload["num_servers"]) != self.num_servers:
            raise ValueError("HIRL checkpoint state/action dimensions do not match.")
        for name in ("actor", "actor_target", "critic", "critic_target", "q_network", "q_target"):
            getattr(self, name).load_state_dict(payload[name])
        for name in ("actor_optimizer", "critic_optimizer", "q_optimizer"):
            getattr(self, name).load_state_dict(payload[name])
        self.environment_steps = int(payload.get("environment_steps", 0))
        self.gradient_steps = int(payload.get("gradient_steps", 0))
        self.current_load = float(payload.get("current_load", 0.0))
        if load_replay:
            self.td3_replay.load_state_dict(payload["td3_replay"])
            self.ddqn_replay.load_state_dict(payload["ddqn_replay"])
        return payload

    def _tensors(self, batch):
        return tuple(torch.as_tensor(value, device=self.device) for value in batch)

    def _rho_from_normalized(self, value):
        ratio = (float(value) + 1.0) / 2.0
        return self.config.min_rho + ratio * (self.config.max_rho - self.config.min_rho)

    def _soft_update(self, source, target):
        with torch.no_grad():
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.mul_(1.0 - self.config.tau).add_(source_parameter, alpha=self.config.tau)

    @staticmethod
    def _state(value, expected_dim):
        result = np.asarray(value, np.float32)
        if result.shape != (expected_dim,):
            raise ValueError(f"HIRL state shape {result.shape}, expected {(expected_dim,)}.")
        if not np.isfinite(result).all():
            raise FloatingPointError("HIRL received a non-finite state.")
        return result
