from __future__ import annotations

import numpy as np
import warnings

from src.a2c.a2c import _require_torch
from src.device_utils import resolve_agent_device
from src.ppo.hierarchical_state import HierarchicalStateEncoder

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = object
    F = None


def _masked_softmax(logits, mask):
    mask = mask.to(dtype=torch.bool)
    if (~mask).all(dim=-1).any():
        raise ValueError("Action mask contains no valid action.")
    return F.softmax(logits.masked_fill(~mask, -1e9), dim=-1)


def _sample_categorical(probs, deterministic):
    dist = torch.distributions.Categorical(probs=probs)
    action = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
    return action, dist.log_prob(action), dist.entropy()


def _normalize_advantages(advantages):
    if advantages.numel() > 1:
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages


def _build_mlp(input_dim, hidden_dim, hidden_layers):
    layers = []
    current_dim = input_dim
    for _ in range(max(int(hidden_layers), 1)):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        current_dim = hidden_dim
    return nn.Sequential(*layers)


class _HighLevelNetwork(nn.Module):
    def __init__(self, input_dim, max_servers, hidden_dim=256, hidden_layers=2):
        _require_torch()
        super().__init__()
        self.backbone = _build_mlp(input_dim, hidden_dim, hidden_layers)
        self.primary_server_actor = nn.Linear(hidden_dim, max_servers)
        self.backup_server_actor = nn.Linear(hidden_dim, max_servers)
        self.critic = nn.Linear(hidden_dim, 1)
        self.deadline_critic = nn.Linear(hidden_dim, 1)
        self.reliability_critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        hidden = self.backbone(state)
        return {
            "primary_server_logits": self.primary_server_actor(hidden),
            "backup_server_logits": self.backup_server_actor(hidden),
            "value": self.critic(hidden).squeeze(-1),
            "deadline_value": self.deadline_critic(hidden).squeeze(-1),
            "reliability_value": self.reliability_critic(hidden).squeeze(-1),
        }


class _LowLevelDVFSNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, hidden_layers=2):
        _require_torch()
        super().__init__()
        self.backbone = _build_mlp(input_dim, hidden_dim, hidden_layers)
        self.rho_alpha = nn.Linear(hidden_dim, 1)
        self.rho_beta = nn.Linear(hidden_dim, 1)
        self.critic = nn.Linear(hidden_dim, 1)
        self.deadline_critic = nn.Linear(hidden_dim, 1)
        self.reliability_critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        hidden = self.backbone(state)
        return {
            "rho_alpha_raw": self.rho_alpha(hidden).squeeze(-1),
            "rho_beta_raw": self.rho_beta(hidden).squeeze(-1),
            "value": self.critic(hidden).squeeze(-1),
            "deadline_value": self.deadline_critic(hidden).squeeze(-1),
            "reliability_value": self.reliability_critic(hidden).squeeze(-1),
        }


class _PPOBase:
    def _init_ppo_config(self, config):
        config = config or {}
        self.reward_gamma = float(config.get("reward_gamma", config.get("gamma", 0.99)))
        self.reward_gae_lambda = float(config.get("reward_gae_lambda", config.get("gae_lambda", 0.95)))
        self.cost_gamma = float(config.get("cost_gamma", 1.0))
        self.cost_gae_lambda = float(config.get("cost_gae_lambda", 1.0))
        self.gamma = self.reward_gamma
        self.gae_lambda = self.reward_gae_lambda
        self.clip_ratio = float(config.get("clip_ratio", 0.2))
        self.ppo_epochs = int(config.get("ppo_epochs", 4))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.learning_rate = float(config.get("learning_rate", 3e-4))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.batch_size = int(config.get("batch_size", 64))
        self.device = resolve_agent_device(config.get("device", "auto"))
        self.use_p3o = bool(config.get("use_p3o", config.get("use_p3o_constraints", False)))
        self.kappa_deadline = float(config.get("kappa_deadline", 5.0))
        self.kappa_reliability = float(config.get("kappa_reliability", 5.0))
        self.constraint_margin_coef = float(config.get("constraint_margin_coef", 1.0))
        self.deadline_gate_margin_ratio = float(config.get("deadline_gate_margin_ratio", 0.05))
        self.deadline_severity_coef = float(config.get("deadline_severity_coef", 2.0))
        self.deadline_violation_clip = float(config.get("deadline_violation_clip", 1.0))
        self.reliability_gate_margin_ratio = float(config.get("reliability_gate_margin_ratio", 0.05))
        self.reliability_severity_coef = float(config.get("reliability_severity_coef", 2.0))
        self.reliability_violation_clip_ratio = float(config.get("reliability_violation_clip_ratio", 1.0))
        self.deadline_cost_limit = config.get("deadline_cost_limit")
        self.reliability_cost_limit = config.get("reliability_cost_limit")
        if self.use_p3o and (
            self.deadline_cost_limit is not None or self.reliability_cost_limit is not None
        ):
            warnings.warn(
                "deadline_cost_limit/reliability_cost_limit are ignored by P3O; "
                "workflow-specific limits from rollout transitions are used.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.cost_value_coef = float(config.get("cost_value_coef", 1.0))
        self.trajectory_group_key = config.get("trajectory_group_key")
        self.server_balanced_minibatch = bool(config.get("server_balanced_minibatch", False))

    def _update_from_buffer(self, buffer):
        if buffer.size == 0:
            return {}
        self._validate_buffer_masks(buffer)

        states = torch.as_tensor(np.asarray(buffer.states), dtype=torch.float32, device=self.device)
        actions = self._actions_tensor(buffer.actions)
        old_log_probs = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        actor_update_masks = torch.as_tensor(
            np.asarray(getattr(buffer, "actor_update_masks", [True] * buffer.size)),
            dtype=torch.bool,
            device=self.device,
        )
        last_value = self._bootstrap_value(buffer)
        reward_stream = np.asarray(buffer.rewards, dtype=np.float32)
        returns, advantages = self._compute_stream_returns_and_advantages(
            buffer,
            reward_stream,
            buffer.values,
            last_value=last_value,
        )
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = _normalize_advantages(advantages)

        p3o_batch = self._build_p3o_batch(buffer) if self.use_p3o else None

        masks = {
            name: torch.as_tensor(np.asarray(values), dtype=torch.bool, device=self.device)
            for name, values in buffer.masks.items()
        }

        total_policy_loss = 0.0
        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_deadline_value_loss = 0.0
        total_reliability_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_deadline_constraint_loss = 0.0
        total_reliability_constraint_loss = 0.0
        max_deadline_constraint_loss = None
        max_reliability_constraint_loss = None
        total_deadline_penalty = 0.0
        total_reliability_penalty = 0.0
        total_deadline_relu_active_rate = 0.0
        total_reliability_relu_active_rate = 0.0
        total_deadline_active_rate = 0.0
        total_reliability_active_rate = 0.0
        total_deadline_violation_rate = 0.0
        total_reliability_violation_rate = 0.0
        total_deadline_violation_mean = 0.0
        total_reliability_violation_mean = 0.0
        max_deadline_violation = None
        max_reliability_violation = None
        total_deadline_surrogate_mean = 0.0
        total_reliability_surrogate_mean = 0.0
        total_deadline_policy_loss = 0.0
        total_reliability_policy_loss = 0.0
        total_deadline_violation_but_inactive_rate = 0.0
        update_count = 0
        count = buffer.size

        for _ in range(self.ppo_epochs):
            if p3o_batch is None:
                if self.server_balanced_minibatch:
                    minibatches = [
                        torch.as_tensor(indices, dtype=torch.long, device=self.device)
                        for indices in buffer.server_balanced_minibatches(self.batch_size)
                    ]
                else:
                    permutation = torch.randperm(count, device=self.device)
                    minibatches = [
                        permutation[start : start + self.batch_size]
                        for start in range(0, count, self.batch_size)
                    ]
            else:
                minibatches = [
                    torch.as_tensor(indices, dtype=torch.long, device=self.device)
                    for indices in buffer.workflow_minibatches(self.batch_size)
                ]
            for batch_idx in minibatches:
                batch_masks = {name: value[batch_idx] for name, value in masks.items()}
                eval_result = self.evaluate_actions(
                    states[batch_idx],
                    actions[batch_idx],
                    batch_masks,
                )
                if len(eval_result) == 3:
                    new_log_probs, values, entropy = eval_result
                    deadline_values = reliability_values = None
                else:
                    new_log_probs, values, deadline_values, reliability_values, entropy = eval_result
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                if not bool(torch.isfinite(ratio).all().item()):
                    raise FloatingPointError("P3O/PPO ratio contains nan or inf.")
                batch_advantages = advantages[batch_idx]
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                    * batch_advantages
                )
                batch_actor_mask = actor_update_masks[batch_idx]
                if batch_actor_mask.any():
                    policy_loss = -torch.min(surr1, surr2)[batch_actor_mask].mean()
                    entropy_term = entropy[batch_actor_mask].mean()
                    clip_fraction = (
                        (torch.abs(ratio[batch_actor_mask] - 1.0) > self.clip_ratio)
                        .float()
                        .mean()
                    )
                else:
                    policy_loss = values.new_tensor(0.0)
                    entropy_term = values.new_tensor(0.0)
                    clip_fraction = values.new_tensor(0.0)
                value_loss = F.mse_loss(values, returns[batch_idx])
                actor_loss = policy_loss - self.entropy_coef * entropy_term

                deadline_value_loss = values.new_tensor(0.0)
                reliability_value_loss = values.new_tensor(0.0)
                deadline_constraint_loss = values.new_tensor(0.0)
                reliability_constraint_loss = values.new_tensor(0.0)
                deadline_penalty = values.new_tensor(0.0)
                reliability_penalty = values.new_tensor(0.0)
                if p3o_batch is not None:
                    if deadline_values is None or reliability_values is None:
                        raise RuntimeError("P3O requires deadline and reliability value heads.")
                    p3o_metrics = self._p3o_actor_terms(
                        ratio,
                        p3o_batch,
                        batch_idx,
                    )
                    deadline_constraint_loss = p3o_metrics["deadline_constraint_loss"]
                    reliability_constraint_loss = p3o_metrics["reliability_constraint_loss"]
                    deadline_penalty = p3o_metrics["deadline_penalty"]
                    reliability_penalty = p3o_metrics["reliability_penalty"]
                    actor_loss = actor_loss + deadline_penalty + reliability_penalty
                    deadline_value_loss = F.mse_loss(
                        deadline_values,
                        p3o_batch["deadline_returns"][batch_idx],
                    )
                    reliability_value_loss = F.mse_loss(
                        reliability_values,
                        p3o_batch["reliability_returns"][batch_idx],
                    )

                loss = actor_loss + self.value_coef * (
                    value_loss
                    + self.cost_value_coef * deadline_value_loss
                    + self.cost_value_coef * reliability_value_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_actor_loss += float(actor_loss.item())
                total_value_loss += float(value_loss.item())
                total_deadline_value_loss += float(deadline_value_loss.item())
                total_reliability_value_loss += float(reliability_value_loss.item())
                total_entropy += float(entropy_term.item())
                total_clip_fraction += float(clip_fraction.item())
                total_deadline_constraint_loss += float(deadline_constraint_loss.item())
                total_reliability_constraint_loss += float(reliability_constraint_loss.item())
                if p3o_batch is not None:
                    batch_max_deadline = float(p3o_metrics["max_deadline_constraint_loss"].item())
                    batch_max_reliability = float(p3o_metrics["max_reliability_constraint_loss"].item())
                    max_deadline_constraint_loss = (
                        batch_max_deadline
                        if max_deadline_constraint_loss is None
                        else max(max_deadline_constraint_loss, batch_max_deadline)
                    )
                    max_reliability_constraint_loss = (
                        batch_max_reliability
                        if max_reliability_constraint_loss is None
                        else max(max_reliability_constraint_loss, batch_max_reliability)
                    )
                    total_deadline_relu_active_rate += float(
                        p3o_metrics["deadline_relu_active_rate"].item()
                    )
                    total_reliability_relu_active_rate += float(
                        p3o_metrics["reliability_relu_active_rate"].item()
                    )
                    total_deadline_active_rate += float(p3o_metrics["deadline_active_rate"].item())
                    total_reliability_active_rate += float(p3o_metrics["reliability_active_rate"].item())
                    total_deadline_violation_rate += float(p3o_metrics["deadline_violation_rate"].item())
                    total_reliability_violation_rate += float(p3o_metrics["reliability_violation_rate"].item())
                    total_deadline_violation_mean += float(p3o_metrics["deadline_violation_mean"].item())
                    total_reliability_violation_mean += float(p3o_metrics["reliability_violation_mean"].item())
                    batch_deadline_violation = float(p3o_metrics["deadline_violation_max"].item())
                    batch_reliability_violation = float(p3o_metrics["reliability_violation_max"].item())
                    max_deadline_violation = (
                        batch_deadline_violation
                        if max_deadline_violation is None
                        else max(max_deadline_violation, batch_deadline_violation)
                    )
                    max_reliability_violation = (
                        batch_reliability_violation
                        if max_reliability_violation is None
                        else max(max_reliability_violation, batch_reliability_violation)
                    )
                    total_deadline_surrogate_mean += float(p3o_metrics["deadline_surrogate_mean"].item())
                    total_reliability_surrogate_mean += float(p3o_metrics["reliability_surrogate_mean"].item())
                    total_deadline_policy_loss += float(p3o_metrics["deadline_policy_loss"].item())
                    total_reliability_policy_loss += float(p3o_metrics["reliability_policy_loss"].item())
                    total_deadline_violation_but_inactive_rate += float(
                        p3o_metrics["deadline_violation_but_inactive_rate"].item()
                    )
                total_deadline_penalty += float(deadline_penalty.item())
                total_reliability_penalty += float(reliability_penalty.item())
                update_count += 1

        denom = max(update_count, 1)
        metrics = {
            "loss": total_actor_loss / denom
            + self.value_coef
            * (
                total_value_loss / denom
                + self.cost_value_coef * total_deadline_value_loss / denom
                + self.cost_value_coef * total_reliability_value_loss / denom
            ),
            "actor_loss": total_actor_loss / denom,
            "policy_loss": total_policy_loss / denom,
            "reward_clip_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "reward_value_loss": total_value_loss / denom,
            "deadline_value_loss": total_deadline_value_loss / denom,
            "reliability_value_loss": total_reliability_value_loss / denom,
            "entropy": total_entropy / denom,
            "clip_fraction": total_clip_fraction / denom,
            "return_mean": float(returns.mean().item()),
        }
        if p3o_batch is not None:
            metrics.update(
                {
                    "deadline_constraint_loss": total_deadline_constraint_loss / denom,
                    "reliability_constraint_loss": total_reliability_constraint_loss / denom,
                    "mean_deadline_constraint_loss": total_deadline_constraint_loss / denom,
                    "max_deadline_constraint_loss": (
                        0.0 if max_deadline_constraint_loss is None else max_deadline_constraint_loss
                    ),
                    "mean_reliability_constraint_loss": total_reliability_constraint_loss / denom,
                    "max_reliability_constraint_loss": (
                        0.0 if max_reliability_constraint_loss is None else max_reliability_constraint_loss
                    ),
                    "p3o_penalty_deadline": total_deadline_penalty / denom,
                    "p3o_penalty_reliability": total_reliability_penalty / denom,
                    "deadline_relu_active_rate": total_deadline_relu_active_rate / denom,
                    "reliability_relu_active_rate": total_reliability_relu_active_rate / denom,
                    "deadline_penalty_after_kappa": total_deadline_penalty / denom,
                    "reliability_penalty_after_kappa": total_reliability_penalty / denom,
                    "high_deadline_active_rate": total_deadline_active_rate / denom,
                    "high_reliability_active_rate": total_reliability_active_rate / denom,
                    "high_deadline_violation_rate": total_deadline_violation_rate / denom,
                    "high_reliability_violation_rate": total_reliability_violation_rate / denom,
                    "high_deadline_violation_mean": total_deadline_violation_mean / denom,
                    "high_reliability_violation_mean": total_reliability_violation_mean / denom,
                    "high_deadline_violation_max": 0.0 if max_deadline_violation is None else max_deadline_violation,
                    "high_reliability_violation_max": 0.0 if max_reliability_violation is None else max_reliability_violation,
                    "high_deadline_surrogate_mean": total_deadline_surrogate_mean / denom,
                    "high_reliability_surrogate_mean": total_reliability_surrogate_mean / denom,
                    "high_deadline_policy_loss": total_deadline_policy_loss / denom,
                    "high_reliability_policy_loss": total_reliability_policy_loss / denom,
                    "high_deadline_penalty_after_kappa": total_deadline_penalty / denom,
                    "high_reliability_penalty_after_kappa": total_reliability_penalty / denom,
                    "high_deadline_violation_but_inactive_rate": total_deadline_violation_but_inactive_rate / denom,
                    "high_deadline_usage_mean": p3o_batch["workflow_metrics"].get("mean_partial_makespan", 0.0),
                    "high_deadline_usage_max": p3o_batch["workflow_metrics"].get("max_partial_makespan", 0.0),
                    "J_deadline": float(p3o_batch["J_deadline"].item()),
                    "J_reliability": float(p3o_batch["J_reliability"].item()),
                    "workflow_constraint_count": int(p3o_batch["workflow_constraint_count"]),
                    "completed_workflow_count": int(p3o_batch["completed_workflow_count"]),
                    "mean_reward": float(torch.as_tensor(reward_stream, dtype=torch.float32).mean().item()),
                    "mean_energy": self._mean_raw_task_energy(buffer),
                    "energy_norm": self._mean_info_value(buffer, "energy_norm"),
                    "mean_energy_ref": self._mean_info_value(buffer, "energy_ref"),
                    "mean_energy_actual": self._mean_info_value(buffer, "energy_actual"),
                    "mean_energy_saving": self._mean_info_value(buffer, "energy_saving"),
                    "clamp_rate": 0.0,
                    "boundary_rate": self._mean_info_bool(buffer, "rho_boundary"),
                    "u_mean": self._mean_info_value(buffer, "action_u"),
                    "u_std": self._std_info_value(buffer, "action_u"),
                    "rho_mean": self._mean_info_value(buffer, "rho"),
                    "rho_std": self._std_info_value(buffer, "rho"),
                    "alpha_mean": self._mean_info_value(buffer, "rho_alpha"),
                    "beta_mean": self._mean_info_value(buffer, "rho_beta"),
                    **p3o_batch["workflow_metrics"],
                    "kappa_deadline": self.kappa_deadline,
                    "kappa_reliability": self.kappa_reliability,
                }
            )
        if isinstance(self, LowLevelDVFSPPOAgent):
            metrics.update(self._server_level_low_stats(buffer))
            metrics.update(
                {
                    "mean_reward": float(torch.as_tensor(reward_stream, dtype=torch.float32).mean().item()),
                    "mean_energy": self._mean_raw_task_energy(buffer),
                    "boundary_rate": self._mean_info_bool(buffer, "rho_boundary"),
                    "u_mean": self._mean_info_value(buffer, "action_u"),
                    "u_std": self._std_info_value(buffer, "action_u"),
                    "rho_mean": self._mean_info_value(buffer, "rho"),
                    "rho_safe_mean": self._mean_info_value(buffer, "rho_safe"),
                    "rho_margin_mean": self._mean_info_value(buffer, "rho_margin"),
                    "below_safe_rate": self._mean_info_bool(buffer, "below_safe"),
                    "forced_max_frequency_rate": self._mean_info_bool(buffer, "forced_max_frequency"),
                    "actor_update_rate": self._mean_info_bool(buffer, "actor_update_mask"),
                    "low_energy_reward_mean": self._mean_info_value(buffer, "low_energy_reward"),
                    "low_local_deadline_penalty_mean": self._mean_info_value(buffer, "low_local_deadline_penalty"),
                    "low_workflow_deadline_penalty_mean": self._mean_info_value(buffer, "low_workflow_deadline_penalty"),
                    "low_infeasible_penalty_mean": self._mean_info_value(buffer, "low_infeasible_penalty"),
                }
            )
        if isinstance(self, HighLevelPPOAgent):
            metrics.update(self._primary_backup_high_stats(buffer))
        return metrics

    @staticmethod
    def _mean_info_value(buffer, key):
        values = [float(info[key]) for info in buffer.infos if key in info and info[key] is not None]
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _std_info_value(buffer, key):
        values = [float(info[key]) for info in buffer.infos if key in info and info[key] is not None]
        return float(np.std(values)) if values else 0.0

    def _compute_stream_returns_and_advantages(self, buffer, rewards, values, last_value=0.0):
        if self.trajectory_group_key:
            return buffer.compute_grouped_stream_returns_and_advantages(
                rewards,
                values,
                self.gamma,
                self.gae_lambda,
                group_key=self.trajectory_group_key,
            )
        return buffer.compute_stream_returns_and_advantages(
            rewards,
            values,
            self.gamma,
            self.gae_lambda,
            last_value=last_value,
        )

    @staticmethod
    def _mean_raw_task_energy(buffer):
        return float(np.mean(buffer.raw_task_energies)) if buffer.raw_task_energies else 0.0

    @staticmethod
    def _mean_info_bool(buffer, key):
        values = [bool(info[key]) for info in buffer.infos if key in info and info[key] is not None]
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _server_level_low_stats(buffer):
        per_server = {}
        for index, info in enumerate(buffer.infos):
            server_id = info.get("server_id")
            if server_id is None:
                continue
            key = str(server_id)
            row = per_server.setdefault(
                key,
                {
                    "count": 0,
                    "rho_sum": 0.0,
                    "energy_sum": 0.0,
                },
            )
            row["count"] += 1
            row["rho_sum"] += float(info.get("rho", 0.0))
            row["energy_sum"] += float(
                info.get(
                    "raw_task_energy",
                    buffer.raw_task_energies[index] if index < len(buffer.raw_task_energies) else 0.0,
                )
            )
        stats = {}
        for server_id, row in sorted(per_server.items(), key=lambda item: int(item[0])):
            count = max(row["count"], 1)
            stats[server_id] = {
                "transitions": row["count"],
                "mean_rho": row["rho_sum"] / count,
                "mean_energy": row["energy_sum"] / count,
            }
        return {
            "trajectory_group_key": "server_id",
            "server_trajectory_count": len(stats),
            "server_transition_counts": {
                server_id: row["transitions"] for server_id, row in stats.items()
            },
            "server_mean_rho": {
                server_id: row["mean_rho"] for server_id, row in stats.items()
            },
            "server_mean_energy": {
                server_id: row["mean_energy"] for server_id, row in stats.items()
            },
        }

    @staticmethod
    def _primary_backup_high_stats(buffer):
        actions = np.asarray(buffer.actions, dtype=np.int64)
        if actions.ndim != 2 or actions.shape[1] != 2 or actions.size == 0:
            return {
                "primary_backup_same_rate": 0.0,
                "primary_selection_count": {},
                "backup_selection_count": {},
            }
        primary_counts = {}
        backup_counts = {}
        for info, action in zip(buffer.infos, actions):
            primary_id = info.get("primary_server_id", info.get("server_id"))
            backup_id = info.get("backup_server_id")
            if primary_id is None:
                primary_id = int(action[0])
            if backup_id is None:
                backup_id = int(action[1])
            primary_key = str(int(primary_id))
            backup_key = str(int(backup_id))
            primary_counts[primary_key] = primary_counts.get(primary_key, 0) + 1
            backup_counts[backup_key] = backup_counts.get(backup_key, 0) + 1
        return {
            "primary_backup_same_rate": float(np.mean(actions[:, 0] == actions[:, 1])),
            "primary_selection_count": dict(sorted(primary_counts.items(), key=lambda item: int(item[0]))),
            "backup_selection_count": dict(sorted(backup_counts.items(), key=lambda item: int(item[0]))),
            "primary_log_prob_mean": HighLevelPPOAgent._mean_info_value(buffer, "primary_log_prob"),
            "backup_log_prob_mean": HighLevelPPOAgent._mean_info_value(buffer, "backup_log_prob"),
            "joint_log_prob_mean": float(np.mean(buffer.log_probs)) if buffer.log_probs else 0.0,
            "primary_entropy_mean": HighLevelPPOAgent._mean_info_value(buffer, "primary_entropy"),
            "backup_entropy_mean": HighLevelPPOAgent._mean_info_value(buffer, "backup_entropy"),
            "joint_entropy_mean": HighLevelPPOAgent._mean_info_value(buffer, "entropy"),
        }

    def _build_p3o_batch(self, buffer):
        buffer.validate_p3o_metadata()
        deadline_costs = torch.as_tensor(buffer.deadline_costs, dtype=torch.float32, device=self.device)
        reliability_costs = torch.as_tensor(buffer.reliability_costs, dtype=torch.float32, device=self.device)
        if (deadline_costs < -1e-8).any() or (reliability_costs < -1e-8).any():
            raise ValueError("P3O cost streams must be non-negative.")
        deadline_bootstrap_values = self._bootstrap_head_values_by_transition(buffer, "deadline_value")
        reliability_bootstrap_values = self._bootstrap_head_values_by_transition(buffer, "reliability_value")
        deadline_returns, deadline_advantages = buffer.compute_workflow_cost_returns_and_advantages(
            buffer.deadline_costs,
            buffer.deadline_values,
            self.cost_gamma,
            self.cost_gae_lambda,
            bootstrap_values=deadline_bootstrap_values,
        )
        reliability_returns, reliability_advantages = buffer.compute_workflow_cost_returns_and_advantages(
            buffer.reliability_costs,
            buffer.reliability_values,
            self.cost_gamma,
            self.cost_gae_lambda,
            bootstrap_values=reliability_bootstrap_values,
        )
        buffer.deadline_returns = np.asarray(deadline_returns, dtype=np.float32).tolist()
        buffer.reliability_returns = np.asarray(reliability_returns, dtype=np.float32).tolist()
        buffer.deadline_advantages = np.asarray(deadline_advantages, dtype=np.float32).tolist()
        buffer.reliability_advantages = np.asarray(reliability_advantages, dtype=np.float32).tolist()
        deadline_advantages = torch.as_tensor(deadline_advantages, dtype=torch.float32, device=self.device)
        reliability_advantages = torch.as_tensor(reliability_advantages, dtype=torch.float32, device=self.device)
        workflow_rows = self._workflow_constraint_rows(buffer)
        j_deadline_values = [row["J_deadline"] for row in workflow_rows]
        j_reliability_values = [row["J_reliability"] for row in workflow_rows]
        deadline_limits = [row["deadline_limit"] for row in workflow_rows]
        reliability_limits = [row["reliability_limit"] for row in workflow_rows]
        completed_count = sum(int(row["workflow_done"]) for row in workflow_rows)
        workflow_metrics = self._workflow_constraint_metrics(
            j_deadline_values,
            deadline_limits,
            j_reliability_values,
            reliability_limits,
        )
        return {
            "deadline_costs": deadline_costs,
            "reliability_costs": reliability_costs,
            "deadline_returns": torch.as_tensor(deadline_returns, dtype=torch.float32, device=self.device),
            "reliability_returns": torch.as_tensor(reliability_returns, dtype=torch.float32, device=self.device),
            "deadline_advantages": deadline_advantages,
            "reliability_advantages": reliability_advantages,
            "workflow_ids": list(buffer.workflow_ids),
            "deadline_limits": torch.as_tensor(buffer.deadline_limits, dtype=torch.float32, device=self.device),
            "reliability_limits": torch.as_tensor(buffer.reliability_limits, dtype=torch.float32, device=self.device),
            "partial_makespans": torch.as_tensor(buffer.partial_makespans, dtype=torch.float32, device=self.device),
            "partial_log_risks": torch.as_tensor(buffer.partial_log_risks, dtype=torch.float32, device=self.device),
            "J_deadline": deadline_costs.new_tensor(float(np.mean(j_deadline_values)) if j_deadline_values else 0.0),
            "J_reliability": reliability_costs.new_tensor(float(np.mean(j_reliability_values)) if j_reliability_values else 0.0),
            "workflow_constraint_count": len(workflow_rows),
            "completed_workflow_count": completed_count,
            "workflow_metrics": workflow_metrics,
        }

    def _p3o_actor_terms(self, ratio, p3o_batch, batch_idx):
        # Violation-gated workflow-level clipped cost surrogate.
        # The gate/severity come from environment-observed workflow usage and
        # are detached; the surrogate provides the policy-gradient direction.
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        deadline_adv = p3o_batch["deadline_advantages"][batch_idx]
        reliability_adv = p3o_batch["reliability_advantages"][batch_idx]
        if deadline_adv.shape != ratio.shape or reliability_adv.shape != ratio.shape:
            raise ValueError("P3O advantage shapes must match policy ratio shape.")
        batch_workflow_ids = [p3o_batch["workflow_ids"][int(index.item())] for index in batch_idx]
        deadline_policy_terms = []
        reliability_policy_terms = []
        deadline_active_flags = []
        reliability_active_flags = []
        deadline_violation_values = []
        reliability_violation_values = []
        deadline_surrogate_values = []
        reliability_surrogate_values = []
        for workflow_id in dict.fromkeys(batch_workflow_ids):
            local_positions = [
                pos for pos, value in enumerate(batch_workflow_ids) if value == workflow_id
            ]
            local_idx = torch.as_tensor(local_positions, dtype=torch.long, device=self.device)
            global_idx = batch_idx[local_idx]
            deadline_surrogate = torch.max(
                ratio[local_idx] * deadline_adv[local_idx],
                clipped_ratio[local_idx] * deadline_adv[local_idx],
            ).mean()
            reliability_surrogate = torch.max(
                ratio[local_idx] * reliability_adv[local_idx],
                clipped_ratio[local_idx] * reliability_adv[local_idx],
            ).mean()
            latest_pos = int(global_idx[-1].item())
            j_deadline = p3o_batch["partial_makespans"][latest_pos].detach()
            deadline_limit = p3o_batch["deadline_limits"][latest_pos].detach()
            j_reliability = p3o_batch["partial_log_risks"][latest_pos].detach()
            reliability_limit = p3o_batch["reliability_limits"][latest_pos].detach()
            deadline_violation = (j_deadline - deadline_limit).detach()
            deadline_gate_margin = (
                self.deadline_gate_margin_ratio * deadline_limit.abs().clamp_min(1e-8)
            ).detach()
            deadline_active = (j_deadline >= deadline_limit - deadline_gate_margin).float().detach()
            deadline_severity = (
                1.0
                + self.deadline_severity_coef
                * deadline_violation.clamp(min=0.0, max=self.deadline_violation_clip)
            ).detach()
            deadline_weight = deadline_active * deadline_severity

            reliability_violation = (j_reliability - reliability_limit).detach()
            reliability_gate_margin = (
                self.reliability_gate_margin_ratio * reliability_limit.abs().clamp_min(1e-8)
            ).detach()
            reliability_active = (
                j_reliability >= reliability_limit - reliability_gate_margin
            ).float().detach()
            reliability_relative_violation = (
                reliability_violation / reliability_limit.abs().clamp_min(1e-8)
            ).detach()
            reliability_severity = (
                1.0
                + self.reliability_severity_coef
                * reliability_relative_violation.clamp(
                    min=0.0,
                    max=self.reliability_violation_clip_ratio,
                )
            ).detach()
            reliability_weight = reliability_active * reliability_severity

            deadline_policy_terms.append(deadline_weight * deadline_surrogate)
            reliability_policy_terms.append(reliability_weight * reliability_surrogate)
            deadline_active_flags.append(deadline_active)
            reliability_active_flags.append(reliability_active)
            deadline_violation_values.append(deadline_violation)
            reliability_violation_values.append(reliability_violation)
            deadline_surrogate_values.append(deadline_surrogate.detach())
            reliability_surrogate_values.append(reliability_surrogate.detach())

        deadline_active_tensor = torch.stack(deadline_active_flags)
        reliability_active_tensor = torch.stack(reliability_active_flags)
        deadline_policy_tensor = torch.stack(deadline_policy_terms)
        reliability_policy_tensor = torch.stack(reliability_policy_terms)
        deadline_violation_tensor = torch.stack(deadline_violation_values)
        reliability_violation_tensor = torch.stack(reliability_violation_values)
        deadline_surrogate_tensor = torch.stack(deadline_surrogate_values)
        reliability_surrogate_tensor = torch.stack(reliability_surrogate_values)

        deadline_active_count = deadline_active_tensor.sum()
        reliability_active_count = reliability_active_tensor.sum()
        if bool((deadline_active_count > 0).item()):
            deadline_constraint_loss = (
                deadline_policy_tensor.sum() / deadline_active_count.clamp_min(1.0)
            )
        else:
            deadline_constraint_loss = ratio.new_tensor(0.0)
        if bool((reliability_active_count > 0).item()):
            reliability_constraint_loss = (
                reliability_policy_tensor.sum() / reliability_active_count.clamp_min(1.0)
            )
        else:
            reliability_constraint_loss = ratio.new_tensor(0.0)

        deadline_violation_flags = deadline_violation_tensor > 0.0
        reliability_violation_flags = reliability_violation_tensor > 0.0
        deadline_violation_but_inactive = (
            deadline_violation_flags & (deadline_active_tensor <= 0.0)
        ).float()
        return {
            "deadline_constraint_loss": deadline_constraint_loss,
            "reliability_constraint_loss": reliability_constraint_loss,
            "max_deadline_constraint_loss": deadline_violation_tensor.max(),
            "max_reliability_constraint_loss": reliability_violation_tensor.max(),
            "deadline_penalty": self.kappa_deadline * deadline_constraint_loss,
            "reliability_penalty": self.kappa_reliability * reliability_constraint_loss,
            "deadline_relu_active_rate": deadline_active_tensor.mean(),
            "reliability_relu_active_rate": reliability_active_tensor.mean(),
            "deadline_active_rate": deadline_active_tensor.mean(),
            "reliability_active_rate": reliability_active_tensor.mean(),
            "deadline_violation_rate": deadline_violation_flags.float().mean(),
            "reliability_violation_rate": reliability_violation_flags.float().mean(),
            "deadline_violation_mean": deadline_violation_tensor.mean(),
            "reliability_violation_mean": reliability_violation_tensor.mean(),
            "deadline_violation_max": deadline_violation_tensor.max(),
            "reliability_violation_max": reliability_violation_tensor.max(),
            "deadline_surrogate_mean": deadline_surrogate_tensor.mean(),
            "reliability_surrogate_mean": reliability_surrogate_tensor.mean(),
            "deadline_policy_loss": deadline_constraint_loss,
            "reliability_policy_loss": reliability_constraint_loss,
            "deadline_violation_but_inactive_rate": deadline_violation_but_inactive.mean(),
        }

    def _workflow_constraint_rows(self, buffer):
        rows = []
        for workflow_id, indices in buffer.workflow_groups():
            last_index = indices[-1]
            rows.append(
                {
                    "workflow_id": workflow_id,
                    "J_deadline": float(buffer.partial_makespans[last_index]),
                    "deadline_limit": float(buffer.deadline_limits[last_index]),
                    "J_reliability": float(buffer.partial_log_risks[last_index]),
                    "reliability_limit": float(buffer.reliability_limits[last_index]),
                    "workflow_done": any(buffer.workflow_dones[index] for index in indices),
                }
            )
        return rows

    @staticmethod
    def _workflow_constraint_metrics(j_deadline, deadline_limits, j_reliability, reliability_limits):
        if not j_deadline:
            return {}
        jd = np.asarray(j_deadline, dtype=np.float64)
        dl = np.asarray(deadline_limits, dtype=np.float64)
        jr = np.asarray(j_reliability, dtype=np.float64)
        rl = np.asarray(reliability_limits, dtype=np.float64)
        dm = jd - dl
        rm = jr - rl
        return {
            "mean_partial_makespan": float(jd.mean()),
            "max_partial_makespan": float(jd.max()),
            "mean_workflow_deadline": float(dl.mean()),
            "min_workflow_deadline": float(dl.min()),
            "max_workflow_deadline": float(dl.max()),
            "mean_deadline_margin": float(dm.mean()),
            "max_deadline_margin": float(dm.max()),
            "deadline_violation_rate": float((dm > 0.0).mean()),
            "mean_partial_log_risk": float(jr.mean()),
            "max_partial_log_risk": float(jr.max()),
            "mean_allowed_log_risk": float(rl.mean()),
            "min_allowed_log_risk": float(rl.min()),
            "max_allowed_log_risk": float(rl.max()),
            "mean_reliability_margin": float(rm.mean()),
            "max_reliability_margin": float(rm.max()),
            "reliability_violation_rate": float((rm > 0.0).mean()),
        }

    def _bootstrap_head_values_by_transition(self, buffer, head_name):
        values = np.zeros(buffer.size, dtype=np.float32)
        if buffer.size == 0:
            return values
        with torch.no_grad():
            for _, indices in buffer.workflow_groups():
                tail = indices[-1]
                if buffer.workflow_dones[tail] or buffer.dones[tail] or buffer.next_states[tail] is None:
                    continue
                state = torch.as_tensor(
                    buffer.next_states[tail],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                values[tail] = float(self.network(state)[head_name].item())
        return values

    def _bootstrap_value(self, buffer):
        if buffer.size == 0 or buffer.dones[-1] or buffer.next_states[-1] is None:
            return 0.0
        with torch.no_grad():
            state = torch.as_tensor(
                buffer.next_states[-1],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            return float(self.network(state)["value"].item())

    def _bootstrap_head_value(self, buffer, head_name):
        if buffer.size == 0 or buffer.dones[-1] or buffer.next_states[-1] is None:
            return 0.0
        with torch.no_grad():
            state = torch.as_tensor(
                buffer.next_states[-1],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            return float(self.network(state)[head_name].item())

    def _actions_tensor(self, actions):
        return torch.as_tensor(actions, dtype=torch.long, device=self.device)

    def _validate_buffer_masks(self, buffer):
        missing = [name for name in self.required_mask_names if name not in buffer.masks]
        if missing:
            raise KeyError(
                f"Rollout buffer missing required mask(s): {missing}. "
                f"Available masks: {sorted(buffer.masks.keys())}"
            )
        for name in self.required_mask_names:
            if len(buffer.masks[name]) != buffer.size:
                raise ValueError(
                    f"Mask '{name}' length {len(buffer.masks[name])} does not match "
                    f"buffer size {buffer.size}"
                )


class HighLevelPPOAgent(_PPOBase):
    LEGACY_SERVER_ACTOR_KEYS = ("server_actor.weight", "server_actor.bias")

    def __init__(self, config=None, seed=42):
        _require_torch()
        config = dict(config or {})
        self.seed = seed
        self.migrate_legacy_high_checkpoint = bool(
            config.get("migrate_legacy_high_checkpoint", False)
        )
        state_cfg = config.get("state", config)
        self.encoder = HierarchicalStateEncoder(state_cfg)
        self._init_ppo_config(config)
        self.network = _HighLevelNetwork(
            input_dim=self.encoder.high_state_dim,
            max_servers=self.encoder.max_servers,
            hidden_dim=int(config.get("hidden_dim", 256)),
            hidden_layers=int(config.get("hidden_layers", 2)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.required_mask_names = ("server_mask",)
        checkpoint = config.get("checkpoint")
        if checkpoint:
            self.load(checkpoint)

    def build_high_state(self, env_state):
        return self.encoder.encode_high_state(env_state)

    def select_assignment(self, env_state, deterministic=False):
        observation = self.build_high_state(env_state)
        state_tensor = torch.as_tensor(
            observation.vector,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        outputs = self.network(state_tensor)
        server_mask = torch.as_tensor(observation.server_mask, device=self.device).unsqueeze(0)
        if int(server_mask.sum().item()) < 2:
            raise ValueError(
                "High-level primary-backup assignment requires at least two valid servers."
            )
        primary_probs = _masked_softmax(outputs["primary_server_logits"], server_mask)
        primary_index, primary_log_prob, primary_entropy = _sample_categorical(
            primary_probs,
            deterministic,
        )
        primary_idx = int(primary_index.item())
        backup_mask = server_mask.clone()
        backup_mask[0, primary_idx] = False
        backup_probs = _masked_softmax(outputs["backup_server_logits"], backup_mask)
        backup_index, backup_log_prob, backup_entropy = _sample_categorical(
            backup_probs,
            deterministic,
        )
        task_idx = 0
        backup_idx = int(backup_index.item())
        if primary_idx == backup_idx:
            raise ValueError("High-level primary and backup server actions must differ.")
        selected_task = observation.ready_tasks[task_idx]
        primary_server = observation.servers[primary_idx]
        backup_server = observation.servers[backup_idx]
        joint_log_prob = primary_log_prob + backup_log_prob
        joint_entropy = primary_entropy + backup_entropy
        return {
            "observation": observation,
            "state": observation.vector,
            "task_index": task_idx,
            "primary_server_index": primary_idx,
            "backup_server_index": backup_idx,
            "server_index": primary_idx,
            "workflow_id": selected_task["workflow_id"],
            "task_id": int(selected_task["task_id"]),
            "primary_server_id": int(primary_server["id"]),
            "backup_server_id": int(backup_server["id"]),
            "server_id": int(primary_server["id"]),
            "server_mask": observation.server_mask.copy(),
            "log_prob": float(joint_log_prob.item()),
            "primary_log_prob": float(primary_log_prob.item()),
            "backup_log_prob": float(backup_log_prob.item()),
            "value": float(outputs["value"].item()),
            "deadline_value": float(outputs["deadline_value"].item()),
            "reliability_value": float(outputs["reliability_value"].item()),
            "entropy": float(joint_entropy.item()),
            "primary_entropy": float(primary_entropy.item()),
            "backup_entropy": float(backup_entropy.item()),
        }

    def evaluate_actions(self, states, actions, masks):
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError(
                "High-level actions must have shape [batch_size, 2] with "
                "[primary_server_index, backup_server_index]."
            )
        outputs = self.network(states)
        server_mask = masks["server_mask"].to(dtype=torch.bool)
        if int(server_mask.ndim) != 2:
            raise ValueError("server_mask must have shape [batch_size, max_servers].")
        if (server_mask.sum(dim=-1) < 2).any():
            raise ValueError(
                "High-level primary-backup assignment requires at least two valid servers."
            )
        primary_actions = actions[:, 0].long()
        backup_actions = actions[:, 1].long()
        batch_size, max_servers = server_mask.shape
        if (
            (primary_actions < 0).any()
            or (primary_actions >= max_servers).any()
            or (backup_actions < 0).any()
            or (backup_actions >= max_servers).any()
        ):
            raise ValueError("High-level server action index is out of range.")
        row_index = torch.arange(batch_size, device=self.device)
        if not bool(server_mask[row_index, primary_actions].all().item()):
            raise ValueError("Primary server action violates server_mask.")
        if not bool(server_mask[row_index, backup_actions].all().item()):
            raise ValueError("Backup server action violates server_mask.")
        if bool((primary_actions == backup_actions).any().item()):
            raise ValueError("High-level primary and backup server actions must differ.")

        primary_probs = _masked_softmax(outputs["primary_server_logits"], server_mask)
        primary_dist = torch.distributions.Categorical(primary_probs)
        primary_log_probs = primary_dist.log_prob(primary_actions)

        backup_mask = server_mask.clone()
        backup_mask[row_index, primary_actions] = False
        if not bool(backup_mask[row_index, backup_actions].all().item()):
            raise ValueError("Backup server action violates primary-excluded backup mask.")
        backup_probs = _masked_softmax(outputs["backup_server_logits"], backup_mask)
        backup_dist = torch.distributions.Categorical(backup_probs)
        backup_log_probs = backup_dist.log_prob(backup_actions)
        log_probs = primary_log_probs + backup_log_probs
        entropy = primary_dist.entropy() + backup_dist.entropy()
        return (
            log_probs,
            outputs["value"],
            outputs["deadline_value"],
            outputs["reliability_value"],
            entropy,
        )

    def update(self, buffer):
        return self._update_from_buffer(buffer)

    def save(self, path, extra=None):
        torch.save({"model_state_dict": self.network.state_dict(), "extra": extra or {}}, path)

    def load(self, path):
        payload = torch.load(path, map_location=self.device)
        state_dict = payload["model_state_dict"]
        if any(key in state_dict for key in self.LEGACY_SERVER_ACTOR_KEYS):
            if not self.migrate_legacy_high_checkpoint:
                raise RuntimeError(
                    "Legacy high-level checkpoint uses single server_actor.* parameters "
                    "and is incompatible with the primary-backup high-level action network. "
                    "Retrain the high-level policy, or set migrate_legacy_high_checkpoint=True "
                    "to explicitly copy server_actor weights into both new actor heads."
                )
            state_dict = dict(state_dict)
            if "server_actor.weight" in state_dict:
                state_dict["primary_server_actor.weight"] = state_dict["server_actor.weight"]
                state_dict["backup_server_actor.weight"] = state_dict["server_actor.weight"]
                del state_dict["server_actor.weight"]
            if "server_actor.bias" in state_dict:
                state_dict["primary_server_actor.bias"] = state_dict["server_actor.bias"]
                state_dict["backup_server_actor.bias"] = state_dict["server_actor.bias"]
                del state_dict["server_actor.bias"]
            print(
                "Migrated legacy high-level server_actor weights to "
                "primary_server_actor and backup_server_actor."
            )
        self.network.load_state_dict(state_dict)
        return payload.get("extra", {})


class LowLevelDVFSPPOAgent(_PPOBase):
    def __init__(self, config=None, seed=42):
        _require_torch()
        config = dict(config or {})
        self.seed = seed
        state_cfg = config.get("state", config)
        self.encoder = HierarchicalStateEncoder(state_cfg)
        self.rho_min = float(config.get("rho_min", 0.1))
        self.beta_concentration_min = float(config.get("beta_concentration_min", 1.0))
        self.beta_action_eps = float(config.get("beta_action_eps", 1e-6))
        config.setdefault("reward_gamma", 0.0)
        config.setdefault("gamma", 0.0)
        config.setdefault("use_p3o", False)
        config.setdefault("server_balanced_minibatch", True)
        self._init_ppo_config(config)
        self.use_p3o = False
        self.trajectory_group_key = None
        self.network = _LowLevelDVFSNetwork(
            input_dim=self.encoder.low_state_dim,
            hidden_dim=int(config.get("hidden_dim", 128)),
            hidden_layers=int(config.get("hidden_layers", 2)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.required_mask_names = ()
        checkpoint = config.get("checkpoint")
        if checkpoint:
            self.load(checkpoint)

    def build_low_state(self, high_observation, task_index, server_index, safety_metrics=None):
        return self.encoder.encode_low_state(
            high_observation,
            task_index,
            server_index,
            safety_metrics=safety_metrics,
        )

    def select_frequency(self, high_observation, task_index, server_index, deterministic=False, safety_metrics=None):
        safety_metrics = dict(safety_metrics or {})
        low_state = self.build_low_state(
            high_observation,
            task_index,
            server_index,
            safety_metrics=safety_metrics,
        )
        rho_safe = float(safety_metrics.get("rho_safe", self.rho_min))
        low_action_feasible = bool(safety_metrics.get("low_action_feasible", True))
        forced_max_frequency = not low_action_feasible
        outputs = self.network(
            torch.as_tensor(low_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        )
        dist, alpha, beta = self._rho_distribution(outputs)
        u_tensor = alpha / (alpha + beta) if deterministic else dist.sample()
        u_safe = self._safe_beta_action(u_tensor)
        log_prob = dist.log_prob(u_safe).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        if forced_max_frequency:
            rho_tensor = torch.ones_like(u_safe)
            actor_update_mask = False
            log_prob_value = 0.0
        else:
            rho_tensor = self._rho_from_unit_action(u_safe, rho_safe=rho_safe)
            actor_update_mask = True
            log_prob_value = float(log_prob.item())
        rho_value = float(rho_tensor.item())
        if rho_value + 1e-8 < rho_safe:
            raise RuntimeError(f"Selected rho={rho_value} below rho_safe={rho_safe}")
        return {
            "state": low_state,
            "action_u": float(u_safe.squeeze(-1).item()),
            "rho": rho_value,
            "log_prob": log_prob_value,
            "value": float(outputs["value"].item()),
            "deadline_value": float(outputs["deadline_value"].item()),
            "reliability_value": float(outputs["reliability_value"].item()),
            "rho_boundary": bool(((u_safe < 0.02) | (u_safe > 0.98)).any().item()),
            "rho_alpha": float(alpha.squeeze(-1).item()),
            "rho_beta": float(beta.squeeze(-1).item()),
            "entropy": float(entropy.item()),
            "rho_safe": rho_safe,
            "low_action_feasible": low_action_feasible,
            "forced_max_frequency": forced_max_frequency,
            "actor_update_mask": actor_update_mask,
        }

    def evaluate_actions(self, states, actions, masks):
        outputs = self.network(states)
        dist, _, _ = self._rho_distribution(outputs)
        u_actions = actions if actions.ndim == 1 else actions[:, 0]
        u_actions = self._safe_beta_action(u_actions).unsqueeze(-1)
        log_probs = dist.log_prob(u_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return (
            log_probs,
            outputs["value"],
            outputs["deadline_value"],
            outputs["reliability_value"],
            entropy,
        )

    def _actions_tensor(self, actions):
        return torch.as_tensor(actions, dtype=torch.float32, device=self.device)

    def _rho_distribution(self, outputs):
        alpha = F.softplus(outputs["rho_alpha_raw"]).unsqueeze(-1) + self.beta_concentration_min
        beta = F.softplus(outputs["rho_beta_raw"]).unsqueeze(-1) + self.beta_concentration_min
        return torch.distributions.Beta(alpha, beta), alpha, beta

    def _safe_beta_action(self, action):
        return action.clamp(self.beta_action_eps, 1.0 - self.beta_action_eps)

    def _rho_from_unit_action(self, action_u, rho_safe=None):
        lower = self.rho_min if rho_safe is None else max(float(rho_safe), self.rho_min)
        lower = min(max(lower, 0.0), 1.0)
        return lower + (1.0 - lower) * action_u

    def update(self, buffer):
        return self._update_from_buffer(buffer)

    def save(self, path, extra=None):
        torch.save({"model_state_dict": self.network.state_dict(), "extra": extra or {}}, path)

    def load(self, path):
        payload = torch.load(path, map_location=self.device)
        self.network.load_state_dict(payload["model_state_dict"])
        return payload.get("extra", {})


class SharedServerLowAgentPool:
    """Logical server-level low agents backed by one shared actor-critic.

    Each server has a logical low-level DVFS agent, but all server-level low
    agents share the same actor and critic parameters. This improves sample
    efficiency and keeps the policy scalable as the number of servers changes.
    Server-specific behavior comes from local state features for the selected
    server, not from independent neural networks.
    """

    def __init__(self, shared_low_agent):
        self.shared_low_agent = shared_low_agent

    def build_low_state(self, high_observation, task_index, server_index, safety_metrics=None):
        return self.shared_low_agent.build_low_state(
            high_observation,
            task_index,
            server_index,
            safety_metrics=safety_metrics,
        )

    def select_frequency_for_server(
        self,
        high_observation,
        task_index,
        server_index,
        deterministic=False,
        safety_metrics=None,
    ):
        decision = self.shared_low_agent.select_frequency(
            high_observation,
            task_index,
            server_index,
            deterministic=deterministic,
            safety_metrics=safety_metrics,
        )
        decision["server_index"] = int(server_index)
        return decision


class HierarchicalPPOAgent:
    def __init__(self, config=None, seed=42):
        config = dict(config or {})
        shared = dict(config.get("shared", {}))
        high_cfg = {**shared, **dict(config.get("high", config.get("center", {})))}
        low_cfg = {**shared, **dict(config.get("low", config.get("adjustment", {})))}
        high_cfg.setdefault("hidden_dim", 256)
        high_cfg.setdefault("hidden_layers", 2)
        low_cfg.setdefault("hidden_dim", 128)
        low_cfg.setdefault("hidden_layers", 2)
        self.high_agent = HighLevelPPOAgent(high_cfg, seed=seed)
        self.low_dvfs_agent = LowLevelDVFSPPOAgent(low_cfg, seed=seed)
        self.server_low_agent_pool = SharedServerLowAgentPool(self.low_dvfs_agent)
        self._validate_encoder_config()

    def _validate_encoder_config(self):
        high_encoder = self.high_agent.encoder
        low_encoder = self.low_dvfs_agent.encoder
        fields = ("max_ready_tasks", "max_servers")
        mismatches = []
        for field in fields:
            high_value = getattr(high_encoder, field)
            low_value = getattr(low_encoder, field)
            if high_value != low_value:
                mismatches.append(f"{field}: high={high_value}, low={low_value}")
        if mismatches:
            raise ValueError(
                "High-level and low-level encoder state config must be consistent: "
                + "; ".join(mismatches)
            )

    def select_action(self, env_state, deterministic=False):
        high_decision = self.high_agent.select_assignment(env_state, deterministic=deterministic)
        primary_server_index = high_decision["primary_server_index"]
        backup_server_id = high_decision["backup_server_id"]
        low_decision = self.server_low_agent_pool.select_frequency_for_server(
            high_observation=high_decision["observation"],
            task_index=high_decision["task_index"],
            server_index=primary_server_index,
            deterministic=deterministic,
        )
        env_action = {
            "workflow_id": high_decision["workflow_id"],
            "task_id": high_decision["task_id"],
            "server_id": high_decision["primary_server_id"],
            "rho": low_decision["rho"],
            "backup": True,
            "backup_server_id": backup_server_id,
        }
        return high_decision, low_decision, env_action

    def save(self, path, extra=None):
        from pathlib import Path

        base = Path(path)
        self.high_agent.save(
            str(base.with_name(f"{base.stem}_high{base.suffix}")),
            extra={"agent": "high_level_ppo", **(extra or {})},
        )
        self.low_dvfs_agent.save(
            str(base.with_name(f"{base.stem}_low{base.suffix}")),
            extra={"agent": "low_dvfs_ppo", **(extra or {})},
        )


CenterPPOAgent = HighLevelPPOAgent
AdjustmentPPOAgent = LowLevelDVFSPPOAgent
CenterAgent = HighLevelPPOAgent
AdjustmentAgent = LowLevelDVFSPPOAgent
DualPPOAgent = HierarchicalPPOAgent
