from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from src.a2c.a2c import (
    GLOBAL_FEATURE_DIM,
    SERVER_FEATURE_DIM,
    TASK_FEATURE_DIM,
    ActionDecision,
    Observation,
    StepContext,
    _require_torch,
)
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


class SACTransition:
    def __init__(
        self,
        observation: Observation,
        action: dict[str, Any],
        reward: float,
        next_observation: Observation,
        done: bool,
    ):
        self.observation = observation
        self.action = action
        self.reward = float(reward)
        self.next_observation = next_observation
        self.done = bool(done)


class DiscreteReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size=100000, batch_size=256):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        self.mem_count = 0
        self.states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.max_size, dtype=np.int64)
        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.next_states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=bool)
        self.action_masks = np.zeros((self.max_size, self.action_dim), dtype=bool)
        self.next_action_masks = np.zeros((self.max_size, self.action_dim), dtype=bool)

    @property
    def size(self):
        return min(self.mem_count, self.max_size)

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        action_mask,
        next_action_mask,
    ):
        mem_index = self.mem_count % self.max_size
        self.states[mem_index] = np.asarray(state, dtype=np.float32)
        self.actions[mem_index] = int(action)
        self.rewards[mem_index] = float(reward)
        self.next_states[mem_index] = np.asarray(next_state, dtype=np.float32)
        self.dones[mem_index] = bool(done)
        self.action_masks[mem_index] = np.asarray(action_mask, dtype=bool)
        self.next_action_masks[mem_index] = np.asarray(next_action_mask, dtype=bool)
        if not bool(self.action_masks[mem_index, self.actions[mem_index]]):
            raise ValueError("Stored SAC action violates its action mask.")
        self.mem_count += 1

    def ready(self):
        return self.size >= self.batch_size

    def sample_buffer(self, rng):
        if not self.ready():
            return None
        batch_indices = rng.choice(self.size, self.batch_size, replace=False)
        return (
            self.states[batch_indices],
            self.actions[batch_indices],
            self.rewards[batch_indices],
            self.next_states[batch_indices],
            self.dones[batch_indices],
            self.action_masks[batch_indices],
            self.next_action_masks[batch_indices],
        )

    def state_dict(self):
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_size": self.max_size,
            "batch_size": self.batch_size,
            "mem_count": self.mem_count,
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "dones": self.dones,
            "action_masks": self.action_masks,
            "next_action_masks": self.next_action_masks,
        }

    def load_state_dict(self, payload):
        self.mem_count = int(payload["mem_count"])
        for name in (
            "states",
            "actions",
            "rewards",
            "next_states",
            "dones",
            "action_masks",
            "next_action_masks",
        ):
            getattr(self, name)[:] = payload[name]


def _build_mlp(input_dim, hidden_dim, hidden_layers):
    layers = []
    current_dim = int(input_dim)
    for _ in range(max(int(hidden_layers), 1)):
        layers.append(nn.Linear(current_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        current_dim = int(hidden_dim)
    return nn.Sequential(*layers)


class DiscreteSACActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, hidden_layers=2):
        _require_torch()
        super().__init__()
        self.backbone = _build_mlp(state_dim, hidden_dim, hidden_layers)
        self.policy_head = nn.Linear(int(hidden_dim), int(action_dim))

    def forward(self, state):
        return self.policy_head(self.backbone(state))


class DiscreteSACCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, hidden_layers=2):
        _require_torch()
        super().__init__()
        self.backbone = _build_mlp(state_dim, hidden_dim, hidden_layers)
        self.value_head = nn.Linear(int(hidden_dim), int(action_dim))

    def forward(self, state):
        return self.value_head(self.backbone(state))


class SAC:
    def __init__(self, config=None, seed: int = 42):
        _require_torch()

        config = dict(config or {})
        state_cfg = config.get("state", config)

        self.seed = int(seed)
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.batch_size = int(config.get("batch_size", 256))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.target_entropy_scale = float(config.get("target_entropy_scale", 0.98))
        self.automatic_entropy_tuning = bool(config.get("automatic_entropy_tuning", True))
        self.alpha = float(config.get("alpha", 0.2))
        self.rng = np.random.default_rng(self.seed)
        self.device = resolve_agent_device(config.get("device", "cpu"))

        self.max_ready_tasks = int(state_cfg.get("max_ready_tasks", 32))
        self.max_servers = int(state_cfg.get("max_servers", 16))
        self.max_rho_levels = int(config.get("max_rho_levels", 6))
        self.action_dim = self.max_ready_tasks * self.max_servers * self.max_rho_levels

        state_dim = len(self.encode_state(self._empty_env_state()).vector)
        hidden_dim = int(config.get("hidden_dim", 256))
        hidden_layers = int(config.get("hidden_layers", 2))
        learning_rate = float(config.get("learning_rate", 3e-4))

        self.actor = DiscreteSACActor(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)
        self.critic_1 = DiscreteSACCritic(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)
        self.critic_2 = DiscreteSACCritic(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)
        self.target_critic_1 = DiscreteSACCritic(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)
        self.target_critic_2 = DiscreteSACCritic(
            state_dim=state_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=learning_rate)
        if self.automatic_entropy_tuning:
            self.target_entropy = -self.target_entropy_scale * np.log(1.0 / float(self.action_dim))
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.target_entropy = 0.0
            self.log_alpha = None
            self.alpha_optimizer = None

        self.memory = DiscreteReplayBuffer(
            state_dim=state_dim,
            action_dim=self.action_dim,
            max_size=int(config.get("replay_size", config.get("max_size", 100000))),
            batch_size=self.batch_size,
        )
        self.update_target_networks(tau=1.0)

        checkpoint = config.get("checkpoint")
        if checkpoint:
            self.load(checkpoint)

    def encode_state(self, env_state: Mapping[str, Any]) -> Observation:
        ready_tasks = tuple(env_state.get("ready_tasks", []))
        servers = tuple(env_state.get("servers", []))
        rho_levels = tuple(float(rho) for rho in env_state.get("rho_levels", [1.0]))

        ready_task_mask = np.zeros(self.max_ready_tasks, dtype=bool)
        for index in range(min(len(ready_tasks), self.max_ready_tasks)):
            ready_task_mask[index] = True

        server_mask = np.zeros(self.max_servers, dtype=bool)
        for index in range(min(len(servers), self.max_servers)):
            server_mask[index] = True

        return Observation(
            ready_task_features=np.zeros((self.max_ready_tasks, TASK_FEATURE_DIM), dtype=np.float32),
            ready_task_mask=ready_task_mask,
            server_features=np.zeros((self.max_servers, SERVER_FEATURE_DIM), dtype=np.float32),
            server_mask=server_mask,
            global_features=np.zeros(GLOBAL_FEATURE_DIM, dtype=np.float32),
            ready_tasks=ready_tasks[: self.max_ready_tasks],
            servers=servers[: self.max_servers],
            rho_levels=rho_levels,
        )

    def reset_reward(self) -> None:
        pass

    def compute_reward(self, context: StepContext) -> float:
        return float(context.env_reward)

    def select_action(self, env_state: Mapping[str, Any], deterministic: bool = False) -> ActionDecision:
        observation = self.encode_state(env_state)
        action_mask = self._joint_action_mask(observation)
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size <= 0:
            raise ValueError("SAC action mask contains no valid action.")

        state_tensor = torch.as_tensor(observation.vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs, log_action_probs = self._policy_distribution(state_tensor, mask_tensor)
            if deterministic:
                action_index = int(torch.argmax(action_probs, dim=-1).item())
            else:
                action_index = int(torch.distributions.Categorical(probs=action_probs).sample().item())
            log_prob = float(log_action_probs[0, action_index].item())

        task_index, server_index, rho_index = self.decode_action(action_index)
        selected_task = observation.ready_tasks[task_index]
        selected_server = observation.servers[server_index]
        action = {
            "workflow_id": selected_task["workflow_id"],
            "task_id": int(selected_task["task_id"]),
            "server_id": int(selected_server["id"]),
            "rho": float(observation.rho_levels[rho_index]),
            "backup": False,
            "backup_server_id": None,
            "_action_index": action_index,
            "_task_index": task_index,
            "_server_index": server_index,
            "_rho_index": rho_index,
        }
        q_value = self.action_q_value(observation.vector, action_index)
        return ActionDecision(
            action=action,
            task_index=task_index,
            server_index=server_index,
            rho_index=rho_index,
            log_prob=log_prob,
            value=q_value,
            observation=observation,
        )

    def build_transition(
        self,
        decision: ActionDecision,
        reward: float,
        next_state: Mapping[str, Any],
        done: bool,
    ) -> SACTransition:
        return SACTransition(
            observation=decision.observation,
            action=dict(decision.action),
            reward=reward,
            next_observation=self.encode_state(next_state),
            done=done,
        )

    def remember(self, transition: SACTransition):
        action_index = int(transition.action["_action_index"])
        self.memory.store_transition(
            transition.observation.vector,
            action_index,
            transition.reward,
            transition.next_observation.vector,
            transition.done,
            self._joint_action_mask(transition.observation),
            self._joint_action_mask(transition.next_observation),
        )

    def update(self, updates=1) -> dict[str, float]:
        metrics = {}
        for _ in range(int(updates)):
            metrics = self.learn()
        return metrics

    def learn(self) -> dict[str, float]:
        sample = self.memory.sample_buffer(self.rng)
        if sample is None:
            return {}
        states, actions, rewards, next_states, dones, action_masks, next_action_masks = sample
        batch_indices = torch.arange(self.batch_size, dtype=torch.long, device=self.device)
        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        masks_tensor = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
        next_masks_tensor = torch.as_tensor(next_action_masks, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            next_probs, next_log_probs = self._policy_distribution(next_states_tensor, next_masks_tensor)
            next_q_min = torch.min(
                self.target_critic_1(next_states_tensor),
                self.target_critic_2(next_states_tensor),
            )
            next_soft_values = (next_probs * (next_q_min - self.alpha * next_log_probs)).sum(dim=-1)
            no_valid_next = next_masks_tensor.sum(dim=-1) <= 0
            next_soft_values = torch.where(dones_tensor | no_valid_next, torch.zeros_like(next_soft_values), next_soft_values)
            target_values = rewards_tensor + self.gamma * next_soft_values

        critic_1_values = self.critic_1(states_tensor)[batch_indices, actions_tensor]
        critic_2_values = self.critic_2(states_tensor)[batch_indices, actions_tensor]
        critic_1_loss = F.mse_loss(critic_1_values, target_values)
        critic_2_loss = F.mse_loss(critic_2_values, target_values)
        if not bool(torch.isfinite(critic_1_loss + critic_2_loss).all().item()):
            raise FloatingPointError("SAC critic loss contains NaN or Inf.")

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        critic_1_grad_norm = nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        critic_2_grad_norm = nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)
        self.critic_2_optimizer.step()

        action_probs, log_action_probs = self._policy_distribution(states_tensor, masks_tensor)
        q_min = torch.min(self.critic_1(states_tensor), self.critic_2(states_tensor))
        actor_loss = (action_probs * (self.alpha * log_action_probs - q_min)).sum(dim=-1).mean()
        if not bool(torch.isfinite(actor_loss).all().item()):
            raise FloatingPointError("SAC actor loss contains NaN or Inf.")

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        entropy = -(action_probs * log_action_probs).sum(dim=-1).mean()
        alpha_loss_value = 0.0
        if self.automatic_entropy_tuning:
            alpha_loss = (self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = float(self.log_alpha.exp().detach().item())
            alpha_loss_value = float(alpha_loss.item())

        self.update_target_networks()
        td_error = target_values.detach() - critic_1_values.detach()
        return {
            "critic_1_loss": float(critic_1_loss.item()),
            "critic_2_loss": float(critic_2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha),
            "entropy": float(entropy.item()),
            "q_mean": float(critic_1_values.detach().mean().item()),
            "target_mean": float(target_values.detach().mean().item()),
            "td_abs_mean": float(td_error.abs().mean().item()),
            "replay_size": float(self.memory.size),
            "critic_1_grad_norm_before_clip": self._grad_norm_to_float(critic_1_grad_norm),
            "critic_2_grad_norm_before_clip": self._grad_norm_to_float(critic_2_grad_norm),
            "actor_grad_norm_before_clip": self._grad_norm_to_float(actor_grad_norm),
        }

    def collect_episode(self, env, deterministic: bool = False, updates_per_step: int = 1) -> dict[str, Any]:
        env.in_run = False
        state = env.reset()
        self.reset_reward()

        transitions: list[SACTransition] = []
        train_metrics = {}
        steps = 0
        max_steps = sum(env._workflow_task_count(item["workflow"]) for item in env.workflow_states) + 1000

        while not env.done and steps < max_steps:
            if not state["ready_tasks"]:
                if not env._advance_to_next_decision_point():
                    env._update_done()
                    break
                state = env.get_state()
                continue

            decision = self.select_action(state, deterministic=deterministic)
            next_state, env_reward, done, info = env.step(decision.action)
            reward = self.compute_reward(
                StepContext(
                    prev_state=state,
                    action=decision.action,
                    next_state=next_state,
                    env_reward=env_reward,
                    done=done,
                    info=info,
                )
            )
            transition = self.build_transition(decision, reward, next_state, done)
            transitions.append(transition)
            self.remember(transition)
            if not deterministic:
                train_metrics = self.update(updates=updates_per_step)
            state = next_state
            steps += 1

        return {
            "transitions": transitions,
            "workflow_results": env._build_workflow_results(),
            "event_log": list(env.event_log),
            "steps": steps,
            "done": env.done,
            "train_metrics": train_metrics,
        }

    def train_episode(self, env, updates_per_step: int = 1) -> dict[str, Any]:
        return self.collect_episode(env, deterministic=False, updates_per_step=updates_per_step)

    def action_q_value(self, observation, action):
        state_tensor = torch.as_tensor(np.asarray([observation], dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_min = torch.min(self.critic_1(state_tensor), self.critic_2(state_tensor))
            return float(q_min[0, int(action)].item())

    def update_target_networks(self, tau=None):
        tau = self.tau if tau is None else float(tau)
        self._soft_update(self.target_critic_1, self.critic_1, tau)
        self._soft_update(self.target_critic_2, self.critic_2, tau)

    def encode_action(self, task_index, server_index, rho_index):
        return (
            int(task_index) * self.max_servers * self.max_rho_levels
            + int(server_index) * self.max_rho_levels
            + int(rho_index)
        )

    def decode_action(self, action):
        action = int(action)
        task_index = action // (self.max_servers * self.max_rho_levels)
        remainder = action % (self.max_servers * self.max_rho_levels)
        server_index = remainder // self.max_rho_levels
        rho_index = remainder % self.max_rho_levels
        return task_index, server_index, rho_index

    def save(self, path: str, extra: Optional[dict] = None, include_replay: bool = False) -> None:
        payload = {
            "model_format": "discrete_sac_v1",
            "actor_state_dict": self.actor.state_dict(),
            "critic_1_state_dict": self.critic_1.state_dict(),
            "critic_2_state_dict": self.critic_2.state_dict(),
            "target_critic_1_state_dict": self.target_critic_1.state_dict(),
            "target_critic_2_state_dict": self.target_critic_2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_1_optimizer_state_dict": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer_state_dict": self.critic_2_optimizer.state_dict(),
            "alpha": float(self.alpha),
            "extra": extra or {},
            "replay_state_dict": self.memory.state_dict() if include_replay else None,
        }
        if self.automatic_entropy_tuning:
            payload["log_alpha"] = self.log_alpha.detach().cpu()
            payload["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()
        torch.save(payload, path)

    def load(self, path: str) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.critic_1.load_state_dict(payload["critic_1_state_dict"])
        self.critic_2.load_state_dict(payload["critic_2_state_dict"])
        self.target_critic_1.load_state_dict(payload["target_critic_1_state_dict"])
        self.target_critic_2.load_state_dict(payload["target_critic_2_state_dict"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        self.critic_1_optimizer.load_state_dict(payload["critic_1_optimizer_state_dict"])
        self.critic_2_optimizer.load_state_dict(payload["critic_2_optimizer_state_dict"])
        self.alpha = float(payload.get("alpha", self.alpha))
        if self.automatic_entropy_tuning and payload.get("log_alpha") is not None:
            self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
            if payload.get("alpha_optimizer_state_dict") is not None:
                self.alpha_optimizer.load_state_dict(payload["alpha_optimizer_state_dict"])
        if payload.get("replay_state_dict") is not None:
            self.memory.load_state_dict(payload["replay_state_dict"])
        return payload.get("extra", {})

    def _joint_action_mask(self, observation: Observation) -> np.ndarray:
        action_mask = np.zeros(self.action_dim, dtype=bool)
        rho_count = max(1, min(len(observation.rho_levels), self.max_rho_levels))
        for task_index, task_valid in enumerate(observation.ready_task_mask):
            if not bool(task_valid):
                continue
            for server_index, server_valid in enumerate(observation.server_mask):
                if not bool(server_valid):
                    continue
                for rho_index in range(rho_count):
                    action_mask[self.encode_action(task_index, server_index, rho_index)] = True
        return action_mask

    def _policy_distribution(self, states_tensor, masks_tensor):
        logits = self.actor(states_tensor).masked_fill(~masks_tensor, -1e9)
        action_probs = F.softmax(logits, dim=-1)
        action_probs = action_probs * masks_tensor.to(dtype=action_probs.dtype)
        normalizer = action_probs.sum(dim=-1, keepdim=True).clamp_min(EPS)
        action_probs = action_probs / normalizer
        log_action_probs = torch.log(action_probs.clamp_min(EPS))
        return action_probs, log_action_probs

    @staticmethod
    def _soft_update(target_network, source_network, tau):
        for target_params, source_params in zip(target_network.parameters(), source_network.parameters()):
            target_params.data.copy_(tau * source_params.data + (1.0 - tau) * target_params.data)

    @staticmethod
    def _empty_env_state() -> dict[str, Any]:
        return {"time": 0.0, "ready_tasks": [], "servers": [], "rho_levels": [1.0]}

    @staticmethod
    def _grad_norm_to_float(grad_norm):
        return float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
