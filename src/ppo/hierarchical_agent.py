
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

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

EPS = 1e-8
LEGACY_CHECKPOINT_ERROR = (
    "Legacy hierarchical checkpoint is incompatible with conditional backup policy. "
    "Retraining is required."
)


def _build_mlp(input_dim, hidden_dim, hidden_layers):
    layers = []
    current_dim = int(input_dim)
    for _ in range(max(int(hidden_layers), 1)):
        layers.append(nn.Linear(current_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        current_dim = int(hidden_dim)
    return nn.Sequential(*layers)


def _masked_logits(logits, mask):
    mask = mask.to(dtype=torch.bool, device=logits.device)
    if logits.ndim != mask.ndim:
        raise ValueError("logits and mask must have the same number of dimensions.")
    if (mask.sum(dim=-1) <= 0).any():
        raise ValueError("Action mask contains no valid action.")
    return logits.masked_fill(~mask, -1e9)


def _categorical_from_logits(logits, mask):
    return torch.distributions.Categorical(logits=_masked_logits(logits, mask))


def _normalize_advantages(values):
    if values.numel() <= 1:
        return values
    return (values - values.mean()) / (values.std(unbiased=False) + EPS)


def _assert_finite_tensor(name, value):
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains NaN or Inf.")


def _as_torch_rng_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype=torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8)


class HighLevelP3ONetwork(nn.Module):
    def __init__(self, input_dim, max_servers, hidden_dim=256, hidden_layers=2, primary_embedding_dim=32):
        _require_torch()
        super().__init__()
        self.max_servers = int(max_servers)
        self.backbone = _build_mlp(input_dim, hidden_dim, hidden_layers)
        self.primary_server_actor = nn.Linear(int(hidden_dim), self.max_servers)
        self.primary_embedding = nn.Embedding(self.max_servers, int(primary_embedding_dim))
        self.backup_backbone = nn.Sequential(
            nn.Linear(int(hidden_dim) + int(primary_embedding_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.backup_server_actor = nn.Linear(int(hidden_dim), self.max_servers)
        self.reward_critic = nn.Linear(int(hidden_dim), 1)
        self.deadline_cost_critic = nn.Linear(int(hidden_dim), 1)
        self.reliability_cost_critic = nn.Linear(int(hidden_dim), 1)

    def encode(self, state):
        return self.backbone(state)

    def primary_logits(self, hidden):
        return self.primary_server_actor(hidden)

    def backup_logits(self, hidden, primary_action):
        primary_action = primary_action.long()
        embedding = self.primary_embedding(primary_action)
        backup_hidden = self.backup_backbone(torch.cat([hidden, embedding], dim=-1))
        return self.backup_server_actor(backup_hidden)

    def values(self, hidden):
        return (
            self.reward_critic(hidden).squeeze(-1),
            self.deadline_cost_critic(hidden).squeeze(-1),
            self.reliability_cost_critic(hidden).squeeze(-1),
        )


class LowLevelDVFSNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, hidden_layers=2):
        _require_torch()
        super().__init__()
        self.backbone = _build_mlp(input_dim, hidden_dim, hidden_layers)
        self.alpha_head = nn.Linear(int(hidden_dim), 1)
        self.beta_head = nn.Linear(int(hidden_dim), 1)
        self.reward_critic = nn.Linear(int(hidden_dim), 1)
        self.deadline_cost_critic = nn.Linear(int(hidden_dim), 1)

    def forward(self, state):
        hidden = self.backbone(state)
        return {
            "alpha_raw": self.alpha_head(hidden).squeeze(-1),
            "beta_raw": self.beta_head(hidden).squeeze(-1),
            "value": self.reward_critic(hidden).squeeze(-1),
            "deadline_value": self.deadline_cost_critic(hidden).squeeze(-1),
        }

class HighLevelP3OAgent:
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
        self.clip_ratio = float(config.get("clip_ratio", 0.2))
        self.ppo_epochs = int(config.get("ppo_epochs", 4))
        self.batch_size = int(config.get("batch_size", 256))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.cost_value_coef = float(config.get("cost_value_coef", 0.5))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.learning_rate = float(config.get("learning_rate", 3e-4))
        self.reward_gamma = float(config.get("reward_gamma", config.get("gamma", 0.99)))
        self.reward_gae_lambda = float(config.get("reward_gae_lambda", config.get("gae_lambda", 0.95)))
        self.cost_gamma = float(config.get("cost_gamma", 1.0))
        self.cost_gae_lambda = float(config.get("cost_gae_lambda", 1.0))
        self.use_p3o = bool(config.get("use_p3o", True))
        self.kappa_deadline = float(config.get("kappa_deadline", 5.0))
        self.kappa_reliability = float(config.get("kappa_reliability", 5.0))
        self.network = HighLevelP3ONetwork(
            input_dim=self.encoder.high_state_dim,
            max_servers=self.encoder.max_servers,
            hidden_dim=int(config.get("hidden_dim", 256)),
            hidden_layers=int(config.get("hidden_layers", 2)),
            primary_embedding_dim=int(config.get("primary_embedding_dim", 32)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def build_high_state(self, env_state):
        return self.encoder.encode_high_state(env_state)

    def select_assignment(self, env_state, deterministic=False, selector_mode="policy"):
        observation = self.build_high_state(env_state)
        context = self._require_context(env_state)
        primary_mask, backup_pair_mask = self._context_masks(context)
        if selector_mode != "policy":
            raise ValueError("Only selector_mode='policy' is supported for joint hierarchical training.")
        state_tensor = torch.as_tensor(observation.vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            hidden = self.network.encode(state_tensor)
            primary_logits = self.network.primary_logits(hidden)
            reward_value, deadline_value, reliability_value = self.network.values(hidden)
            primary_mask_tensor = torch.as_tensor(primary_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            primary_dist = _categorical_from_logits(primary_logits, primary_mask_tensor)
            primary_action = torch.argmax(_masked_logits(primary_logits, primary_mask_tensor), dim=-1) if deterministic else primary_dist.sample()
            primary_idx = int(primary_action.item())
            backup_mask = self._effective_backup_mask(primary_idx, primary_mask, backup_pair_mask)
            backup_mask_tensor = torch.as_tensor(backup_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            backup_logits = self.network.backup_logits(hidden, primary_action)
            backup_dist = _categorical_from_logits(backup_logits, backup_mask_tensor)
            backup_action = torch.argmax(_masked_logits(backup_logits, backup_mask_tensor), dim=-1) if deterministic else backup_dist.sample()
            log_prob = primary_dist.log_prob(primary_action) + backup_dist.log_prob(backup_action)
            entropy = primary_dist.entropy() + backup_dist.entropy()
        return self._decision_payload(
            observation, context, primary_mask, backup_pair_mask, primary_idx, int(backup_action.item()),
            float(log_prob.item()), float(reward_value.item()), float(deadline_value.item()),
            float(reliability_value.item()), float(entropy.item()), selector_mode="policy",
        )

    def evaluate_actions(self, states, primary_actions, backup_actions, primary_masks, backup_pair_masks):
        states = states.to(self.device)
        primary_actions = primary_actions.long().to(self.device)
        backup_actions = backup_actions.long().to(self.device)
        primary_masks = primary_masks.to(dtype=torch.bool, device=self.device)
        backup_pair_masks = backup_pair_masks.to(dtype=torch.bool, device=self.device)
        hidden = self.network.encode(states)
        primary_logits = self.network.primary_logits(hidden)
        primary_dist = _categorical_from_logits(primary_logits, primary_masks)
        primary_log_probs = primary_dist.log_prob(primary_actions)
        row_index = torch.arange(states.shape[0], device=self.device)
        if not bool(primary_masks[row_index, primary_actions].all().item()):
            raise ValueError("Primary action violates stored primary mask.")
        backup_masks = backup_pair_masks[row_index, primary_actions, :].clone()
        invalid_rows = backup_masks.sum(dim=-1) <= 0
        if bool(invalid_rows.any().item()):
            backup_masks[invalid_rows] = primary_masks[invalid_rows]
            backup_masks[invalid_rows, primary_actions[invalid_rows]] = False
        if not bool(backup_masks[row_index, backup_actions].all().item()):
            raise ValueError("Backup action violates stored backup pair mask.")
        if bool((primary_actions == backup_actions).any().item()):
            raise ValueError("Primary and backup actions must differ.")
        backup_logits = self.network.backup_logits(hidden, primary_actions)
        backup_dist = _categorical_from_logits(backup_logits, backup_masks)
        backup_log_probs = backup_dist.log_prob(backup_actions)
        reward_values, deadline_values, reliability_values = self.network.values(hidden)
        entropy = primary_dist.entropy() + backup_dist.entropy()
        return primary_log_probs + backup_log_probs, reward_values, deadline_values, reliability_values, entropy

    def update(self, buffer):
        if buffer.size == 0:
            return {}
        buffer.validate()
        reward_returns, reward_advantages = buffer.compute_reward_returns_and_advantages(self.reward_gamma, self.reward_gae_lambda)
        deadline_returns, deadline_advantages = buffer.compute_normalized_deadline_returns_and_advantages(self.cost_gamma, self.cost_gae_lambda)
        reliability_returns, reliability_advantages = buffer.compute_cost_returns_and_advantages("reliability", self.cost_gamma, self.cost_gae_lambda)
        normalized_partial_makespans = buffer.normalized_partial_makespans()
        normalized_deadline_limits = np.ones_like(normalized_partial_makespans, dtype=np.float32)
        states = torch.as_tensor(np.asarray(buffer.states), dtype=torch.float32, device=self.device)
        primary_actions = torch.as_tensor(buffer.primary_actions, dtype=torch.long, device=self.device)
        backup_actions = torch.as_tensor(buffer.backup_actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor(buffer.old_log_probs, dtype=torch.float32, device=self.device)
        reward_returns_t = torch.as_tensor(reward_returns, dtype=torch.float32, device=self.device)
        reward_advantages_t = _normalize_advantages(torch.as_tensor(reward_advantages, dtype=torch.float32, device=self.device))
        deadline_returns_t = torch.as_tensor(deadline_returns, dtype=torch.float32, device=self.device)
        reliability_returns_t = torch.as_tensor(reliability_returns, dtype=torch.float32, device=self.device)
        deadline_advantages_t = torch.as_tensor(deadline_advantages, dtype=torch.float32, device=self.device)
        reliability_advantages_t = torch.as_tensor(reliability_advantages, dtype=torch.float32, device=self.device)
        primary_masks = torch.as_tensor(np.asarray(buffer.primary_masks), dtype=torch.bool, device=self.device)
        backup_pair_masks = torch.as_tensor(np.asarray(buffer.backup_pair_masks), dtype=torch.bool, device=self.device)
        reward_adv_std = float(np.std(reward_advantages)) if len(reward_advantages) else 0.0
        deadline_adv_std = float(np.std(deadline_advantages)) if len(deadline_advantages) else 0.0
        reliability_adv_std = float(np.std(reliability_advantages)) if len(reliability_advantages) else 0.0
        totals = {}
        update_count = 0
        for _ in range(self.ppo_epochs):
            for batch_idx_np in buffer.workflow_minibatches(self.batch_size, rng=self.rng):
                batch_idx = torch.as_tensor(batch_idx_np, dtype=torch.long, device=self.device)
                log_probs, reward_values, deadline_values, reliability_values, entropy = self.evaluate_actions(
                    states[batch_idx], primary_actions[batch_idx], backup_actions[batch_idx],
                    primary_masks[batch_idx], backup_pair_masks[batch_idx],
                )
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                _assert_finite_tensor("high_ratio", ratio)
                surr1 = ratio * reward_advantages_t[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * reward_advantages_t[batch_idx]
                reward_clip_loss = -torch.min(surr1, surr2).mean()
                entropy_term = entropy.mean()
                actor_loss = reward_clip_loss - self.entropy_coef * entropy_term
                p3o_metrics = self.compute_p3o_terms(
                    ratio,
                    batch_idx_np,
                    deadline_advantages_t[batch_idx],
                    reliability_advantages_t[batch_idx],
                    buffer,
                    partial_makespans=normalized_partial_makespans,
                    deadline_limits=normalized_deadline_limits,
                )
                if self.use_p3o:
                    actor_loss = actor_loss + self.kappa_deadline * p3o_metrics["deadline_penalty"] + self.kappa_reliability * p3o_metrics["reliability_penalty"]
                reward_value_loss = F.mse_loss(reward_values, reward_returns_t[batch_idx])
                deadline_value_loss = F.mse_loss(deadline_values, deadline_returns_t[batch_idx])
                reliability_value_loss = F.mse_loss(reliability_values, reliability_returns_t[batch_idx])
                critic_loss = reward_value_loss + self.cost_value_coef * (deadline_value_loss + reliability_value_loss)
                loss = actor_loss + self.value_coef * critic_loss
                _assert_finite_tensor("high_loss", loss)
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm_before_clip = nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                grad_norm_before_clip = float(grad_norm_before_clip.item() if torch.is_tensor(grad_norm_before_clip) else grad_norm_before_clip)
                grad_norm_after_clip = min(grad_norm_before_clip, self.max_grad_norm)
                self.optimizer.step()
                batch_metrics = {
                    "loss": loss.item(), "actor_loss": actor_loss.item(), "reward_clip_loss": reward_clip_loss.item(),
                    "reward_value_loss": reward_value_loss.item(), "deadline_value_loss": deadline_value_loss.item(),
                    "reliability_value_loss": reliability_value_loss.item(), "entropy": entropy_term.item(),
                    "clip_fraction": ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item(),
                    "reward_adv_std": reward_adv_std,
                    "deadline_adv_std": deadline_adv_std,
                    "reliability_adv_std": reliability_adv_std,
                    "grad_norm_before_clip": grad_norm_before_clip,
                    "grad_norm_after_clip": grad_norm_after_clip,
                }
                batch_metrics.update({key: value.item() if torch.is_tensor(value) else float(value) for key, value in p3o_metrics.items()})
                for key, value in batch_metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                update_count += 1
        denom = max(update_count, 1)
        return {key: value / denom for key, value in totals.items()}

    def compute_p3o_terms(
        self,
        ratio,
        batch_indices_np,
        deadline_advantages,
        reliability_advantages,
        buffer,
        partial_makespans=None,
        deadline_limits=None,
    ):
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
        partial_makespans = buffer.partial_makespans if partial_makespans is None else partial_makespans
        deadline_limits = buffer.deadline_limits if deadline_limits is None else deadline_limits
        return self._workflow_p3o_terms(
            ratio, clipped_ratio, batch_indices_np, deadline_advantages, reliability_advantages,
            buffer.workflow_ids, partial_makespans, deadline_limits,
            buffer.partial_log_risks, buffer.reliability_limits,
        )
    @staticmethod
    def _workflow_p3o_terms(ratio, clipped_ratio, batch_indices_np, deadline_advantages, reliability_advantages, workflow_ids, partial_makespans, deadline_limits, partial_log_risks, reliability_limits):
        workflow_to_positions = {}
        for local_pos, global_index in enumerate(batch_indices_np):
            workflow_to_positions.setdefault(str(workflow_ids[int(global_index)]), []).append(local_pos)
        deadline_pre_values = []
        reliability_pre_values = []
        deadline_raw_values = []
        reliability_raw_values = []
        deadline_margin_values = []
        reliability_margin_values = []
        for _, local_positions in workflow_to_positions.items():
            pos_tensor = torch.as_tensor(local_positions, dtype=torch.long, device=ratio.device)
            global_indices = [int(batch_indices_np[pos]) for pos in local_positions]
            group_workflow_id = str(workflow_ids[global_indices[0]])
            group_deadline = float(deadline_limits[global_indices[0]])
            if group_deadline <= 0.0:
                raise RuntimeError(f"P3O deadline must be positive for workflow {group_workflow_id}.")
            for index in global_indices:
                if str(workflow_ids[index]) != group_workflow_id:
                    raise RuntimeError("P3O workflow group contains mixed workflows.")
                current_deadline = float(deadline_limits[index])
                if abs(current_deadline - group_deadline) > 1e-6:
                    raise RuntimeError(
                        "P3O deadline mismatch within workflow: "
                        f"workflow={group_workflow_id}, "
                        f"reference={group_deadline}, "
                        f"value={current_deadline}"
                    )
            deadline_surrogate = torch.max(
                ratio[pos_tensor] * deadline_advantages[pos_tensor],
                clipped_ratio[pos_tensor] * deadline_advantages[pos_tensor],
            ).mean()
            reliability_surrogate = torch.max(
                ratio[pos_tensor] * reliability_advantages[pos_tensor],
                clipped_ratio[pos_tensor] * reliability_advantages[pos_tensor],
            ).mean()
            last_index = global_indices[-1]
            workflow_partial_makespan = float(partial_makespans[last_index])
            workflow_deadline = float(deadline_limits[last_index])
            deadline_margin = workflow_partial_makespan - workflow_deadline
            reliability_margin = float(partial_log_risks[last_index]) - float(reliability_limits[last_index])
            deadline_pre = deadline_surrogate + ratio.new_tensor(deadline_margin)
            reliability_pre = reliability_surrogate + ratio.new_tensor(reliability_margin)
            deadline_raw_values.append(deadline_surrogate)
            reliability_raw_values.append(reliability_surrogate)
            deadline_margin_values.append(ratio.new_tensor(deadline_margin))
            reliability_margin_values.append(ratio.new_tensor(reliability_margin))
            deadline_pre_values.append(deadline_pre)
            reliability_pre_values.append(reliability_pre)
        deadline_pre_tensor = torch.stack(deadline_pre_values) if deadline_pre_values else ratio.new_zeros(1)
        reliability_pre_tensor = torch.stack(reliability_pre_values) if reliability_pre_values else ratio.new_zeros(1)
        deadline_penalty_values = F.relu(deadline_pre_tensor)
        reliability_penalty_values = F.relu(reliability_pre_tensor)
        deadline_penalty = deadline_penalty_values.mean()
        reliability_penalty = reliability_penalty_values.mean()
        if deadline_penalty.item() < -EPS or reliability_penalty.item() < -EPS:
            raise FloatingPointError("P3O penalty must be non-negative.")
        return {
            "deadline_raw_surrogate": torch.stack(deadline_raw_values).mean() if deadline_raw_values else ratio.new_tensor(0.0),
            "deadline_margin": torch.stack(deadline_margin_values).mean() if deadline_margin_values else ratio.new_tensor(0.0),
            "deadline_pre_relu": deadline_pre_tensor.mean(),
            "deadline_penalty": deadline_penalty,
            "deadline_relu_active_rate": (deadline_pre_tensor > 0.0).float().mean(),
            "reliability_raw_surrogate": torch.stack(reliability_raw_values).mean() if reliability_raw_values else ratio.new_tensor(0.0),
            "reliability_margin": torch.stack(reliability_margin_values).mean() if reliability_margin_values else ratio.new_tensor(0.0),
            "reliability_pre_relu": reliability_pre_tensor.mean(),
            "reliability_penalty": reliability_penalty,
            "reliability_relu_active_rate": (reliability_pre_tensor > 0.0).float().mean(),
        }

    def evaluate_action_log_prob(self, state_vector, primary_action, backup_action, primary_mask, backup_pair_mask):
        states = torch.as_tensor(np.asarray([state_vector], dtype=np.float32), dtype=torch.float32, device=self.device)
        primary_actions = torch.as_tensor([int(primary_action)], dtype=torch.long, device=self.device)
        backup_actions = torch.as_tensor([int(backup_action)], dtype=torch.long, device=self.device)
        primary_masks = torch.as_tensor(np.asarray([primary_mask], dtype=bool), dtype=torch.bool, device=self.device)
        backup_pair_masks = torch.as_tensor(np.asarray([backup_pair_mask], dtype=bool), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            log_probs, _, _, _, _ = self.evaluate_actions(states, primary_actions, backup_actions, primary_masks, backup_pair_masks)
        return float(log_probs.item())

    def _decision_payload(self, observation, context, primary_mask, backup_pair_mask, primary_idx, backup_idx, log_prob, reward_value, deadline_value, reliability_value, entropy, selector_mode):
        if primary_idx >= len(observation.servers) or backup_idx >= len(observation.servers):
            raise ValueError("Selected server index exceeds observed server count.")
        task = observation.ready_tasks[0]
        primary_server = observation.servers[primary_idx]
        backup_server = observation.servers[backup_idx]
        effective_backup_mask = self._effective_backup_mask(primary_idx, primary_mask, backup_pair_mask)
        return {
            "observation": observation,
            "state": observation.vector,
            "task_index": 0,
            "primary_server_index": int(primary_idx),
            "backup_server_index": int(backup_idx),
            "server_index": int(primary_idx),
            "workflow_id": str(task["workflow_id"]),
            "task_id": int(task["task_id"]),
            "primary_server_id": int(primary_server["id"]),
            "backup_server_id": int(backup_server["id"]),
            "server_id": int(primary_server["id"]),
            "primary_server_mask": primary_mask.copy(),
            "backup_pair_mask": backup_pair_mask.copy(),
            "effective_backup_mask": effective_backup_mask.copy(),
            "log_prob": float(log_prob),
            "value": float(reward_value),
            "deadline_value": float(deadline_value),
            "reliability_value": float(reliability_value),
            "entropy": float(entropy),
            "shield_relaxed": bool(context.get("shield_relaxed", False)),
            "shield_relaxation_reason": context.get("shield_relaxation_reason"),
            "backup_shield_relaxed": not bool(backup_pair_mask[int(primary_idx)].any()),
            "selector_mode": selector_mode,
        }

    def _context_masks(self, context):
        primary_raw = np.asarray(context["primary_server_mask"], dtype=bool)
        pair_raw = np.asarray(context["backup_pair_mask"], dtype=bool)
        if primary_raw.ndim != 1:
            raise ValueError("decision_context primary_server_mask must be one-dimensional.")
        if pair_raw.shape != (primary_raw.shape[0], primary_raw.shape[0]):
            raise ValueError("decision_context backup_pair_mask must be [server_count, server_count].")
        if int(primary_raw.sum()) <= 0:
            raise ValueError("decision_context primary_server_mask contains no valid action.")
        max_servers = self.encoder.max_servers
        if primary_raw.shape[0] > max_servers:
            raise ValueError("server count exceeds encoder max_servers.")
        primary_mask = np.zeros(max_servers, dtype=bool)
        backup_pair_mask = np.zeros((max_servers, max_servers), dtype=bool)
        primary_mask[: primary_raw.shape[0]] = primary_raw
        backup_pair_mask[: pair_raw.shape[0], : pair_raw.shape[1]] = pair_raw
        return primary_mask, backup_pair_mask

    def _effective_backup_mask(self, primary_idx, primary_mask, backup_pair_mask):
        backup_mask = np.asarray(backup_pair_mask[int(primary_idx)], dtype=bool).copy()
        if int(backup_mask.sum()) == 0:
            backup_mask = np.asarray(primary_mask, dtype=bool).copy()
            backup_mask[int(primary_idx)] = False
        if int(backup_mask.sum()) <= 0:
            raise ValueError("backup mask contains no valid action after fallback.")
        return backup_mask

    def _require_context(self, env_state):
        context = env_state.get("decision_context")
        if context is None:
            raise KeyError("env_state must contain decision_context for hierarchical safe PPO.")
        return context

class LowLevelDVFSPPOAgent:
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
        self.learning_rate = float(config.get("learning_rate", 3e-4))
        self.gamma = 0.0
        self.clip_ratio = float(config.get("clip_ratio", 0.2))
        self.ppo_epochs = int(config.get("ppo_epochs", 4))
        self.batch_size = int(config.get("batch_size", 128))
        self.entropy_coef = float(config.get("entropy_coef", 0.01))
        self.value_coef = float(config.get("value_coef", 0.5))
        self.cost_value_coef = float(config.get("cost_value_coef", 0.5))
        self.kappa_deadline = float(config.get("kappa_deadline", 5.0))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.rho_min = float(config.get("rho_min", 0.1))
        self.beta_concentration_min = float(config.get("beta_concentration_min", 1.0))
        self.beta_action_eps = float(config.get("beta_action_eps", 1e-6))
        self.network = LowLevelDVFSNetwork(
            input_dim=self.encoder.low_state_dim,
            hidden_dim=int(config.get("hidden_dim", 128)),
            hidden_layers=int(config.get("hidden_layers", 2)),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def build_low_state(self, high_observation, task_index, server_index, safety_metrics=None):
        return self.encoder.encode_low_state(high_observation, task_index, server_index, safety_metrics=safety_metrics)

    def select_frequency(self, high_observation, task_index, server_index, deterministic=False, safety_metrics=None):
        safety_metrics = dict(safety_metrics or {})
        low_state = self.build_low_state(high_observation, task_index, server_index, safety_metrics=safety_metrics)
        rho_safe = min(1.0, max(self.rho_min, float(safety_metrics.get("rho_safe", self.rho_min))))
        low_action_feasible = bool(safety_metrics.get("low_action_feasible", True))
        has_action_span = bool(low_action_feasible and rho_safe < 1.0 - 1e-6)
        state_tensor = torch.as_tensor(low_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.network(state_tensor)
            dist, alpha, beta = self._distribution(outputs)
            action_u = alpha / (alpha + beta) if deterministic else dist.sample()
            action_u = self._safe_action(action_u)
            log_prob = dist.log_prob(action_u).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        if not has_action_span:
            rho = 1.0
            actor_update_mask = False
            log_prob_value = 0.0
        else:
            action_u_value = float(action_u.item())
            # 使用 Python float 计算边界映射，避免 torch float32 在 rho_safe 接近 1 时产生微小越界。
            rho = float(rho_safe + (1.0 - rho_safe) * action_u_value)
            actor_update_mask = True
            log_prob_value = float(log_prob.item())
        rho = float(min(1.0, max(rho_safe, rho)))
        if rho + 1e-8 < rho_safe or rho > 1.0 + 1e-8:
            raise RuntimeError(f"Selected rho={rho} outside safe interval [{rho_safe}, 1].")
        return {
            "state": low_state,
            "action_u": float(action_u.item()),
            "rho": rho,
            "log_prob": log_prob_value,
            "value": float(outputs["value"].item()),
            "deadline_value": float(outputs["deadline_value"].item()),
            "rho_alpha": float(alpha.item()),
            "rho_beta": float(beta.item()),
            "entropy": float(entropy.item()),
            "rho_safe": float(rho_safe),
            "low_action_feasible": low_action_feasible,
            "has_action_span": has_action_span,
            "actor_update_mask": actor_update_mask,
            "forced_max_frequency": not has_action_span,
        }

    def evaluate_actions(self, states, actions_u):
        outputs = self.network(states.to(self.device))
        dist, _, _ = self._distribution(outputs)
        actions_u = self._safe_action(actions_u.to(self.device).float()).unsqueeze(-1)
        log_probs = dist.log_prob(actions_u).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, outputs["value"], outputs["deadline_value"], entropy

    def update(self, buffer, learning_rate_scale=1.0):
        if buffer.size == 0:
            return {}
        buffer.validate()
        returns, advantages = buffer.compute_returns_and_advantages()
        deadline_returns, deadline_advantages = buffer.compute_deadline_returns_and_advantages()
        states = torch.as_tensor(np.asarray(buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(buffer.actions_u, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(buffer.old_log_probs, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = _normalize_advantages(torch.as_tensor(advantages, dtype=torch.float32, device=self.device))
        deadline_returns_t = torch.as_tensor(deadline_returns, dtype=torch.float32, device=self.device)
        deadline_advantages_t = torch.as_tensor(deadline_advantages, dtype=torch.float32, device=self.device)
        deadline_adv_std = float(np.std(deadline_advantages)) if len(deadline_advantages) else 0.0
        old_lrs = [group["lr"] for group in self.optimizer.param_groups]
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate * float(learning_rate_scale)
        totals = {}
        updates = 0
        count = buffer.size
        for _ in range(self.ppo_epochs):
            order = self.rng.permutation(count)
            for start in range(0, count, self.batch_size):
                batch_np = order[start : start + self.batch_size]
                batch = torch.as_tensor(batch_np, dtype=torch.long, device=self.device)
                log_probs, values, deadline_values, entropy = self.evaluate_actions(states[batch], actions[batch])
                ratio = torch.exp(log_probs - old_log_probs[batch])
                _assert_finite_tensor("low_ratio", ratio)
                surr1 = ratio * advantages_t[batch]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_t[batch]
                policy_loss = -torch.min(surr1, surr2).mean()
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                deadline_surr = torch.max(
                    ratio * deadline_advantages_t[batch],
                    clipped_ratio * deadline_advantages_t[batch],
                ).mean()
                value_loss = F.mse_loss(values, returns_t[batch])
                deadline_value_loss = F.mse_loss(deadline_values, deadline_returns_t[batch])
                entropy_term = entropy.mean()
                loss = (
                    policy_loss
                    + self.kappa_deadline * deadline_surr
                    + self.value_coef * value_loss
                    + self.cost_value_coef * deadline_value_loss
                    - self.entropy_coef * entropy_term
                )
                _assert_finite_tensor("low_loss", loss)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                metrics = {
                    "loss": loss.item(),
                    "policy_loss": policy_loss.item(),
                    "deadline_surrogate": deadline_surr.item(),
                    "value_loss": value_loss.item(),
                    "deadline_value_loss": deadline_value_loss.item(),
                    "entropy": entropy_term.item(),
                    "clip_fraction": ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item(),
                    "return_mean": returns_t[batch].mean().item(),
                    "deadline_return_mean": deadline_returns_t[batch].mean().item(),
                    "deadline_adv_std": deadline_adv_std,
                }
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value)
                updates += 1
        for group, old_lr in zip(self.optimizer.param_groups, old_lrs):
            group["lr"] = old_lr
        denom = max(updates, 1)
        return {key: value / denom for key, value in totals.items()}

    def _distribution(self, outputs):
        alpha = F.softplus(outputs["alpha_raw"]).unsqueeze(-1) + self.beta_concentration_min
        beta = F.softplus(outputs["beta_raw"]).unsqueeze(-1) + self.beta_concentration_min
        return torch.distributions.Beta(alpha, beta), alpha, beta

    def _safe_action(self, action):
        return action.clamp(self.beta_action_eps, 1.0 - self.beta_action_eps)

class HierarchicalSafeAgent:
    def __init__(self, config=None, seed=42):
        config = dict(config or {})
        self.config = copy.deepcopy(config)
        self.seed = int(seed)
        shared = dict(config.get("shared", {}))
        high_cfg = {**shared, **dict(config.get("high", {}))}
        low_cfg = {**shared, **dict(config.get("low", {}))}
        self.high_agent = HighLevelP3OAgent(high_cfg, seed=self.seed)
        self.low_agent = LowLevelDVFSPPOAgent(low_cfg, seed=self.seed)
        self._validate_encoder_config()

    def _validate_encoder_config(self):
        for field in ("max_ready_tasks", "max_servers"):
            if getattr(self.high_agent.encoder, field) != getattr(self.low_agent.encoder, field):
                raise ValueError(f"High/low encoder config mismatch for {field}.")

    def select_action(self, env_state, deterministic=False, low_deterministic=None, high_selector_mode="policy"):
        if low_deterministic is None:
            low_deterministic = deterministic
        high_decision = self.high_agent.select_assignment(
            env_state,
            deterministic=deterministic,
            selector_mode=high_selector_mode,
        )
        context = env_state["decision_context"]
        metrics_by_id = context.get("server_safety_metrics", {})
        safety_metrics = metrics_by_id.get(str(high_decision["primary_server_id"]))
        if safety_metrics is None:
            raise KeyError(f"Missing safety metrics for server {high_decision['primary_server_id']}.")
        low_decision = self.low_agent.select_frequency(
            high_decision["observation"],
            high_decision["task_index"],
            high_decision["primary_server_index"],
            deterministic=bool(low_deterministic),
            safety_metrics=safety_metrics,
        )
        env_action = {
            "workflow_id": high_decision["workflow_id"],
            "task_id": high_decision["task_id"],
            "server_id": high_decision["primary_server_id"],
            "rho": low_decision["rho"],
            "backup": True,
            "backup_server_id": high_decision["backup_server_id"],
        }
        return high_decision, low_decision, env_action

    def save_checkpoint(self, path, *, objective_state=None, episode=0, training_phase=None, config=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_format": "hierarchical_safe_ppo_v1",
            "high_model_state_dict": self.high_agent.network.state_dict(),
            "high_optimizer_state_dict": self.high_agent.optimizer.state_dict(),
            "low_model_state_dict": self.low_agent.network.state_dict(),
            "low_optimizer_state_dict": self.low_agent.optimizer.state_dict(),
            "objective_state": dict(objective_state or {}),
            "episode": int(episode),
            "training_phase": training_phase,
            "config": copy.deepcopy(config if config is not None else self.config),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(payload, str(path))
        self._save_component_checkpoints(
            path,
            objective_state=objective_state,
            episode=episode,
            training_phase=training_phase,
            config=config,
        )

    def _save_component_checkpoints(self, path, *, objective_state=None, episode=0, training_phase=None, config=None):
        common = {
            "episode": int(episode),
            "training_phase": training_phase,
            "config": copy.deepcopy(config if config is not None else self.config),
            "objective_state": dict(objective_state or {}),
        }
        high_payload = {
            "checkpoint_format": "hierarchical_safe_ppo_high_v1",
            "high_model_state_dict": self.high_agent.network.state_dict(),
            "high_optimizer_state_dict": self.high_agent.optimizer.state_dict(),
            **common,
        }
        low_payload = {
            "checkpoint_format": "hierarchical_safe_ppo_low_v1",
            "low_model_state_dict": self.low_agent.network.state_dict(),
            "low_optimizer_state_dict": self.low_agent.optimizer.state_dict(),
            "low_has_deadline_cost_critic": True,
            **common,
        }
        torch.save(high_payload, str(self._component_checkpoint_path(path, "high")))
        torch.save(low_payload, str(self._component_checkpoint_path(path, "low")))

    @staticmethod
    def _component_checkpoint_path(path, component):
        path = Path(path)
        name = path.name
        if name.endswith("_checkpoint.pt"):
            return path.with_name(name.replace("_checkpoint.pt", f"_{component}.pt"))
        if name.endswith(".pt"):
            return path.with_name(f"{path.stem}_{component}.pt")
        return path.with_name(f"{name}_{component}.pt")

    def load_checkpoint(self, path):
        payload = torch.load(str(path), map_location=self.high_agent.device, weights_only=False)
        if payload.get("checkpoint_format") != "hierarchical_safe_ppo_v1":
            raise RuntimeError(LEGACY_CHECKPOINT_ERROR)
        self.high_agent.network.load_state_dict(payload["high_model_state_dict"])
        self.high_agent.optimizer.load_state_dict(payload["high_optimizer_state_dict"])
        self.low_agent.network.load_state_dict(payload["low_model_state_dict"])
        self.low_agent.optimizer.load_state_dict(payload["low_optimizer_state_dict"])
        if payload.get("numpy_random_state") is not None:
            np.random.set_state(payload["numpy_random_state"])
        if payload.get("torch_random_state") is not None:
            torch.random.set_rng_state(_as_torch_rng_state(payload["torch_random_state"]))
        if torch.cuda.is_available() and payload.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all([_as_torch_rng_state(state) for state in payload["cuda_random_state"]])
        return payload

    def save(self, path, extra=None):
        extra = dict(extra or {})
        self.save_checkpoint(
            path,
            objective_state=extra.get("objective_state", {}),
            episode=extra.get("episode", 0),
            training_phase=extra.get("training_phase"),
            config=extra.get("config", self.config),
        )

    def load(self, path):
        return self.load_checkpoint(path)
