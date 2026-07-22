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
    Transition,
    _ActorCriticNetwork,
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


class PPO:
    def __init__(self, config=None, seed: int = 42):
        _require_torch()

        config = config or {}
        state_cfg = config.get("state", config)
        _ = config.get("reward", {})

        self.seed = seed
        self.gamma = float(config.get("gamma", 0.99))
        self.gae_lambda = float(config.get("gae_lambda", 0.95))
        self.clip_ratio = float(config.get("clip_ratio", 0.2))
        self.ppo_epochs = int(config.get("ppo_epochs", 4))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.learning_rate = float(config.get("learning_rate", 3e-4))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.device = resolve_agent_device(config.get("device", "cpu"))

        self.max_ready_tasks = int(state_cfg.get("max_ready_tasks", 32))
        self.max_servers = int(state_cfg.get("max_servers", 16))
        self.max_rho_levels = int(config.get("max_rho_levels", 6))

        obs_dim = len(self.encode_state(self._empty_env_state()).vector)
        self.network = _ActorCriticNetwork(
            obs_dim=obs_dim,
            max_ready_tasks=self.max_ready_tasks,
            max_servers=self.max_servers,
            max_rho_levels=self.max_rho_levels,
            hidden_dim=int(config.get("hidden_dim", 128)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        checkpoint = config.get("checkpoint")
        if checkpoint:
            self.load(checkpoint)

    # ------------------------------------------------------------------ state
    def encode_state(self, env_state: Mapping[str, Any]) -> Observation:
        """State 特征在 CPU(numpy) 上编码；仅 network forward 使用 self.device。"""
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
            ready_task_features=np.zeros(
                (self.max_ready_tasks, TASK_FEATURE_DIM), dtype=np.float32
            ),
            ready_task_mask=ready_task_mask,
            server_features=np.zeros((self.max_servers, SERVER_FEATURE_DIM), dtype=np.float32),
            server_mask=server_mask,
            global_features=np.zeros(GLOBAL_FEATURE_DIM, dtype=np.float32),
            ready_tasks=ready_tasks[: self.max_ready_tasks],
            servers=servers[: self.max_servers],
            rho_levels=rho_levels,
        )

    # ----------------------------------------------------------------- reward
    def reset_reward(self) -> None:
        pass

    def compute_reward(self, context: StepContext) -> float:
        """占位：后续在这里设计 reward。"""
        return float(context.env_reward)

    # ----------------------------------------------------------------- policy
    def select_action(
        self,
        env_state: Mapping[str, Any],
        deterministic: bool = False,
    ) -> ActionDecision:
        observation = self.encode_state(env_state)
        outputs = self.network(
            torch.as_tensor(observation.vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        )

        task_mask = torch.as_tensor(observation.ready_task_mask, device=self.device).unsqueeze(0)
        server_mask = torch.as_tensor(observation.server_mask, device=self.device).unsqueeze(0)
        rho_count = max(1, min(len(observation.rho_levels), self.max_rho_levels))
        rho_mask = torch.zeros((1, self.max_rho_levels), dtype=torch.bool, device=self.device)
        rho_mask[0, :rho_count] = True

        task_probs = self._masked_softmax(outputs["task_logits"], task_mask)
        server_probs = self._masked_softmax(outputs["server_logits"], server_mask)
        rho_probs = self._masked_softmax(outputs["rho_logits"], rho_mask)

        task_index, task_log_prob = self._sample_categorical(task_probs, deterministic)
        server_index, server_log_prob = self._sample_categorical(server_probs, deterministic)
        rho_index, rho_log_prob = self._sample_categorical(rho_probs, deterministic)

        task_idx = int(task_index.item())
        server_idx = int(server_index.item())
        rho_idx = int(rho_index.item())

        selected_task = observation.ready_tasks[task_idx]
        selected_server = observation.servers[server_idx]

        action = {
            "workflow_id": selected_task["workflow_id"],
            "task_id": int(selected_task["task_id"]),
            "server_id": int(selected_server["id"]),
            "rho": float(observation.rho_levels[rho_idx]),
            "backup": False,
            "backup_server_id": None,
        }
        return ActionDecision(
            action=action,
            task_index=task_idx,
            server_index=server_idx,
            rho_index=rho_idx,
            log_prob=float((task_log_prob + server_log_prob + rho_log_prob).item()),
            value=float(outputs["value"].item()),
            observation=observation,
        )

    def build_transition(
        self,
        decision: ActionDecision,
        reward: float,
        done: bool,
    ) -> Transition:
        action = dict(decision.action)
        action["_task_index"] = decision.task_index
        action["_server_index"] = decision.server_index
        action["_rho_index"] = decision.rho_index
        return Transition(
            observation=decision.observation,
            action=action,
            reward=reward,
            done=done,
            log_prob=decision.log_prob,
            value=decision.value,
        )

    def update(self, transitions: list[Transition]) -> dict[str, float]:
        if not transitions:
            return {}

        rewards = torch.as_tensor(
            [transition.reward for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        old_log_probs = torch.as_tensor(
            [transition.log_prob for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.as_tensor(
            [transition.value for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )

        # advantages are computed per-workflow: independent workflows must not leak
        # reward credit across each other (no cross-workflow discounting).
        advantages = self._compute_advantages(rewards, old_values, transitions)
        # value target = GAE advantage + baseline, keeps returns/advantages consistent
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
        }

        for _ in range(self.ppo_epochs):
            new_log_probs, values, entropies = self._evaluate_actions(transitions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = -entropies.mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(-entropy_loss.item())
            metrics["clip_fraction"] += float(
                (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean().item()
            )

        epoch_count = float(self.ppo_epochs)
        return {
            "loss": metrics["policy_loss"] / epoch_count + self.value_coef * metrics["value_loss"] / epoch_count,
            "policy_loss": metrics["policy_loss"] / epoch_count,
            "value_loss": metrics["value_loss"] / epoch_count,
            "entropy": metrics["entropy"] / epoch_count,
            "clip_fraction": metrics["clip_fraction"] / epoch_count,
            "return_mean": float(returns.mean().item()),
        }

    def _evaluate_actions(
        self,
        transitions: list[Transition],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_probs = []
        values = []
        entropies = []

        for transition in transitions:
            outputs = self.network(
                torch.as_tensor(
                    transition.observation.vector,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
            )

            task_mask = torch.as_tensor(
                transition.observation.ready_task_mask,
                device=self.device,
            ).unsqueeze(0)
            server_mask = torch.as_tensor(
                transition.observation.server_mask,
                device=self.device,
            ).unsqueeze(0)
            rho_count = max(1, min(len(transition.observation.rho_levels), self.max_rho_levels))
            rho_mask = torch.zeros((1, self.max_rho_levels), dtype=torch.bool, device=self.device)
            rho_mask[0, :rho_count] = True

            task_probs = self._masked_softmax(outputs["task_logits"], task_mask)
            server_probs = self._masked_softmax(outputs["server_logits"], server_mask)
            rho_probs = self._masked_softmax(outputs["rho_logits"], rho_mask)

            task_dist = torch.distributions.Categorical(task_probs)
            server_dist = torch.distributions.Categorical(server_probs)
            rho_dist = torch.distributions.Categorical(rho_probs)

            task_index = torch.as_tensor(transition.action["_task_index"], device=self.device)
            server_index = torch.as_tensor(transition.action["_server_index"], device=self.device)
            rho_index = torch.as_tensor(transition.action["_rho_index"], device=self.device)

            log_prob = (
                task_dist.log_prob(task_index)
                + server_dist.log_prob(server_index)
                + rho_dist.log_prob(rho_index)
            )
            entropy = task_dist.entropy() + server_dist.entropy() + rho_dist.entropy()

            log_probs.append(log_prob.squeeze(0))
            values.append(outputs["value"].squeeze(0))
            entropies.append(entropy.squeeze(0))

        return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)

    # --------------------------------------------------------------- training
    def collect_episode(self, env, deterministic: bool = False) -> dict[str, Any]:
        env.in_run = False
        state = env.reset()
        self.reset_reward()

        transitions: list[Transition] = []
        prev_state = state
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
                    prev_state=prev_state,
                    action=decision.action,
                    next_state=next_state,
                    env_reward=env_reward,
                    done=done,
                    info=info,
                )
            )
            transitions.append(self.build_transition(decision, reward, done))
            prev_state = next_state
            state = next_state
            steps += 1

        return {
            "transitions": transitions,
            "workflow_results": env._build_workflow_results(),
            "event_log": list(env.event_log),
            "steps": steps,
            "done": env.done,
        }

    def train_episode(self, env) -> dict[str, Any]:
        episode = self.collect_episode(env, deterministic=False)
        episode["train_metrics"] = self.update(episode["transitions"])
        return episode

    # ----------------------------------------------------------------- persist
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        torch.save(
            {
                "model_state_dict": self.network.state_dict(),
                "extra": extra or {},
            },
            path,
        )

    def load(self, path: str) -> dict:
        payload = torch.load(path, map_location=self.device)
        self.network.load_state_dict(payload["model_state_dict"])
        return payload.get("extra", {})

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _empty_env_state() -> dict[str, Any]:
        return {"time": 0.0, "ready_tasks": [], "servers": [], "rho_levels": [1.0]}

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=torch.bool)
        return F.softmax(logits.masked_fill(~mask, -1e9), dim=-1)

    @staticmethod
    def _sample_categorical(
        probs: torch.Tensor,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Categorical(probs=probs)
        action = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action)

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        transitions: list[Transition],
    ) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        for segment in self._workflow_segments(transitions):
            self._gae_for_segment(rewards, values, segment, advantages)
        return advantages

    def _gae_for_segment(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        segment: list[int],
        advantages: torch.Tensor,
    ) -> None:
        # each workflow is an independent finite sub-episode: its last scheduling
        # step is terminal (bootstrap 0) and GAE resets at the segment boundary.
        gae = 0.0
        next_value = 0.0
        for pos in range(len(segment) - 1, -1, -1):
            index = segment[pos]
            mask = 0.0 if pos == len(segment) - 1 else 1.0
            delta = (
                float(rewards[index].item())
                + self.gamma * next_value * mask
                - float(values[index].item())
            )
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[index] = gae
            next_value = float(values[index].item())

    @staticmethod
    def _workflow_segments(transitions: list[Transition]) -> list[list[int]]:
        """Group transition indices by workflow_id, preserving temporal order."""
        segments: dict[str, list[int]] = {}
        order: list[str] = []
        for index, transition in enumerate(transitions):
            workflow_id = str(transition.action.get("workflow_id"))
            if workflow_id not in segments:
                segments[workflow_id] = []
                order.append(workflow_id)
            segments[workflow_id].append(index)
        return [segments[workflow_id] for workflow_id in order]
