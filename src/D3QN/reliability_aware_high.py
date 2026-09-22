"""Standalone reliability-aware extension of the production high-level D3QN.

The production two-head implementation is not modified. This module reuses
its ReplayBuffer, action-mask checks and Double-DQN next-action convention,
and adds one parallel replay field and one Q head for cumulative reliability
violation cost. SAC is not imported or changed here.
"""
from __future__ import annotations

import math
import numpy as np
from src.D3QN.dual_head_constrained import DualHeadConstrainedD3QN
from src.D3QN.dual_head_constrained import DualHeadHighLevelD3QNAgent

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

EPS = 1e-8


def reliability_cost(workflow_reliability, required_reliability=0.9):
    required = float(np.clip(required_reliability, EPS, 1.0))
    actual = float(np.clip(workflow_reliability, EPS, 1.0))
    return float(max(0.0, math.log(required) - math.log(actual)))


def reliability_cost_from_log_values(log_workflow_reliability, log_required_reliability):
    return float(max(0.0, float(log_required_reliability) - float(log_workflow_reliability)))


class ReliabilityLagrangian:
    """Multiplier updated once for each completed workflow."""
    def __init__(self, lambda_init=1.0, dual_lr=0.001, lambda_max=100.0):
        if lambda_init < 0 or dual_lr < 0 or lambda_max <= 0:
            raise ValueError("Invalid reliability Lagrangian parameters.")
        self.lambda_reliability = float(lambda_init)
        self.dual_lr = float(dual_lr)
        self.lambda_max = float(lambda_max)

    def update(self, cost):
        value = float(cost)
        if not np.isfinite(value) or value < 0:
            raise ValueError("Reliability cost must be finite and non-negative.")
        before = self.lambda_reliability
        self.lambda_reliability = float(np.clip(before + self.dual_lr * value, 0.0, self.lambda_max))
        return {"lambda_reliability_before_update": before,
                "lambda_reliability": self.lambda_reliability,
                "lambda_reliability_after_update": self.lambda_reliability,
                "reliability_cost": value}

    def state_dict(self):
        return {
            "lambda_reliability": self.lambda_reliability,
            "dual_learning_rate": self.dual_lr,
            "lambda_reliability_max": self.lambda_max,
        }

    def load_state_dict(self, state):
        value = float(dict(state or {}).get(
            "lambda_reliability", self.lambda_reliability
        ))
        if not np.isfinite(value):
            raise ValueError("Restored reliability lambda must be finite.")
        self.lambda_reliability = float(np.clip(value, 0.0, self.lambda_max))


class ReliabilityAwareConstrainedD3QN(DualHeadConstrainedD3QN):
    """Production dual-head learner with a third reliability-cost head."""
    def __init__(self, *args, reliability_learning_rate=None, reliability_gamma=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required for Reliability D3QN.")
        from src.D3QN.D3QN import DuelingDeepQNetwork
        hidden_dim = int(kwargs.get("hidden_dim", 256))
        hidden_layers = int(kwargs.get("hidden_layers", 2))
        net_args = dict(state_dim=self.state_dim, action_dim=self.action_dim,
                        hidden_dim=hidden_dim, hidden_layers=hidden_layers)
        self.reliability_gamma = float(reliability_gamma)
        self.reliability_q_eval = DuelingDeepQNetwork(**net_args).to(self.device)
        self.reliability_q_target = DuelingDeepQNetwork(**net_args).to(self.device)
        lr = self.energy_optimizer.param_groups[0]["lr"] if reliability_learning_rate is None else reliability_learning_rate
        self.reliability_optimizer = torch.optim.Adam(self.reliability_q_eval.parameters(), lr=float(lr))
        self.reliability_costs = np.zeros(self.memory.max_size, dtype=np.float32)
        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        super().update_network_parameters(tau=tau)
        if not hasattr(self, "reliability_q_target"):
            return
        rate = self.tau if tau is None else float(tau)
        for target, online in ((self.reliability_q_target, self.reliability_q_eval),):
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.data.copy_(rate * op.data + (1.0 - rate) * tp.data)

    def remember(self, state, action, energy_reward, deadline_cost, reliability_cost_value,
                 next_state, done, action_mask, next_action_mask):
        index = self.memory.mem_count % self.memory.max_size
        super().remember(state, action, energy_reward, deadline_cost, next_state, done,
                         action_mask, next_action_mask)
        value = float(reliability_cost_value)
        if not np.isfinite(value) or value < 0:
            raise ValueError("reliability_cost must be finite and non-negative.")
        self.reliability_costs[index] = value

    def action_q_values(self, observation):
        energy, deadline = super().action_q_values(observation)
        state = torch.as_tensor(np.asarray([observation], dtype=np.float32), device=self.device)
        with torch.no_grad():
            reliability = self.reliability_q_eval(state)[0]
        return energy, deadline, reliability.detach().cpu().numpy().astype(np.float64)

    def combined_action_q_values(self, observation, deadline_lambda, reliability_lambda):
        energy, deadline, reliability = self.action_q_values(observation)
        return (energy - float(deadline_lambda) * deadline - float(reliability_lambda) * reliability,
                energy, deadline, reliability)

    def _next_actions(self, next_states, next_masks, deadline_lambda, reliability_lambda=1.0):
        with torch.no_grad():
            deadline = self.deadline_q_eval(next_states)
            if self.nonnegative_deadline_q:
                deadline = torch.clamp(deadline, min=0.0)
            combined = (self.energy_q_eval(next_states)
                        - float(deadline_lambda) * deadline
                        - float(reliability_lambda) * self.reliability_q_eval(next_states))
            return torch.argmax(combined.masked_fill(~next_masks, -1e9), dim=-1)

    def learn(self, deadline_lambda, reliability_lambda=1.0):
        indices = self._uniform_indices()
        if indices is None:
            return {}
        batch = self._batch(indices)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.bool, device=self.device)
        row = torch.arange(actions.shape[0], device=self.device)
        rel_rewards = torch.as_tensor(self.reliability_costs[indices], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_actions = self._next_actions(next_states, next_masks, deadline_lambda, reliability_lambda)

        def loss_for(online, target, rewards, gamma, *, nonnegative_target=False):
            q = online(states)[row, actions]
            with torch.no_grad():
                target_q = target(next_states)[row, next_actions]
                if nonnegative_target:
                    target_q = torch.clamp(target_q, min=0.0)
                target_q = torch.where(dones, torch.zeros_like(target_q), target_q)
                y = rewards + float(gamma) * target_q
            return F.smooth_l1_loss(q, y), q, y

        energy_loss, energy_q, _ = loss_for(self.energy_q_eval, self.energy_q_target,
                                             torch.as_tensor(batch["energy_rewards"], dtype=torch.float32, device=self.device), self.energy_gamma)
        deadline_loss, deadline_q, _ = loss_for(self.deadline_q_eval, self.deadline_q_target,
                                                 torch.as_tensor(batch["deadline_costs"], dtype=torch.float32, device=self.device), self.deadline_gamma,
                                                 nonnegative_target=self.nonnegative_deadline_q)
        reliability_loss, reliability_q, reliability_target = loss_for(self.reliability_q_eval, self.reliability_q_target,
                                                                         rel_rewards, self.reliability_gamma)
        total = energy_loss + deadline_loss + reliability_loss
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError("Reliability-aware D3QN loss contains NaN or Inf.")
        gradients = []
        for optimizer, loss, network in ((self.energy_optimizer, energy_loss, self.energy_q_eval),
                                         (self.deadline_optimizer, deadline_loss, self.deadline_q_eval),
                                         (self.reliability_optimizer, reliability_loss, self.reliability_q_eval)):
            optimizer.zero_grad(); loss.backward()
            gradients.append(float(torch.nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)))
            optimizer.step()
        self.update_network_parameters(); self.decrement_epsilon()
        return {"loss": float(total.item()), "energy_loss": float(energy_loss.item()),
                "deadline_loss": float(deadline_loss.item()), "reliability_loss": float(reliability_loss.item()),
                "energy_q_mean": float(energy_q.mean().item()), "deadline_q_mean": float(deadline_q.mean().item()),
                "reliability_q_mean": float(reliability_q.mean().item()),
                "reliability_cost_batch_mean": float(rel_rewards.mean().item()),
                "reliability_td_abs_mean": float((reliability_target - reliability_q).abs().mean().item()),
                "grad_norm_before_clip": float(max(gradients)),
                "reliability_lambda_for_action": float(reliability_lambda)}

    def state_dict(self, *, include_replay=False):
        payload = super().state_dict(include_replay=include_replay)
        payload.update({"reliability_q_eval": self.reliability_q_eval.state_dict(),
                        "reliability_q_target": self.reliability_q_target.state_dict(),
                        "reliability_optimizer": self.reliability_optimizer.state_dict(),
                        "reliability_costs": self.reliability_costs.copy() if include_replay else None})
        return payload

    def load_state_dict(self, payload):
        super().load_state_dict(payload)
        self.reliability_q_eval.load_state_dict(payload["reliability_q_eval"])
        self.reliability_q_target.load_state_dict(payload["reliability_q_target"])
        self.reliability_optimizer.load_state_dict(payload["reliability_optimizer"])
        if payload.get("reliability_costs") is not None:
            self.reliability_costs[:] = payload["reliability_costs"]


def augment_high_state(base_state, current_log_reliability, required_log_reliability,
                       candidate_server_failure_rate, failure_rate_scale=1e-3):
    """Append current log reliability, required log reliability, margin and rates."""
    base = np.asarray(base_state, dtype=np.float32)
    rates = np.asarray(candidate_server_failure_rate, dtype=np.float32)
    if base.ndim != 1 or rates.ndim != 1:
        raise ValueError("High state and failure-rate features must be vectors.")
    margin = float(current_log_reliability) - float(required_log_reliability)
    extra = np.asarray([np.tanh(float(current_log_reliability)),
                        np.tanh(float(required_log_reliability)), np.tanh(margin)], dtype=np.float32)
    return np.concatenate([base, extra, np.tanh(rates / max(float(failure_rate_scale), EPS))]).astype(np.float32)


class ReliabilityAwareHighLevelD3QNAgent(DualHeadHighLevelD3QNAgent):
    """Production high-level wrapper with workflow-terminal reliability costs.

    The wrapper is intentionally high-level only. It accepts the existing
    ``HighRolloutBuffer`` and leaves all low-level SAC objects untouched.
    The three requested state fields are exposed by ``augment_high_state``;
    callers that use the augmented state must construct the high learner with
    the corresponding augmented dimension.
    """

    def __init__(self, config=None, seed=42):
        super().__init__(config=config, seed=seed)
        config = dict(config or {})
        old = self.agent_R
        self.reliability_required = float(config.get("reliability_required", 0.9))
        self.reliability_lagrangian = ReliabilityLagrangian(
            lambda_init=float(config.get("lambda_reliability_init", 1.0)),
            dual_lr=float(config.get("dual_lr_reliability", 0.001)),
            lambda_max=float(config.get("lambda_reliability_max", 100.0)),
        )
        self.agent_R = ReliabilityAwareConstrainedD3QN(
            state_dim=old.state_dim, action_dim=old.action_dim,
            learning_rate=float(config.get("learning_rate", 3e-4)),
            deadline_learning_rate=float(config.get("deadline_learning_rate", config.get("learning_rate", 3e-4))),
            reliability_learning_rate=float(config.get("reliability_learning_rate", config.get("learning_rate", 3e-4))),
            hidden_dim=int(config.get("hidden_dim", 256)), hidden_layers=int(config.get("hidden_layers", 2)),
            energy_gamma=float(config.get("gamma", config.get("reward_gamma", 0.99))),
            deadline_gamma=float(config.get("deadline_gamma", 0.99)),
            reliability_gamma=float(config.get("reliability_gamma", 0.99)),
            tau=float(config.get("tau", 0.005)), epsilon=float(config.get("epsilon", 1.0)),
            epsilon_min=float(config.get("epsilon_min", 0.1)), epsilon_decay=float(config.get("epsilon_decay", 1.8e-4)),
            replay_size=int(config.get("replay_size", 50000)), batch_size=int(config.get("batch_size", 256)),
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            deadline_positive_fraction=float(config.get("deadline_positive_fraction", 0.3)),
            nonnegative_deadline_q=bool(config.get("nonnegative_deadline_q", False)),
            device=config.get("device", "auto"), seed=seed,
        )
        # Compatibility names used by the production checkpoint/logger path.
        self.agent_R.q_eval = self.agent_R.energy_q_eval
        self.agent_R.q_target = self.agent_R.energy_q_target
        self.agent_R.optimizer = self.agent_R.energy_optimizer

    @property
    def lambda_reliability(self):
        return float(self.reliability_lagrangian.lambda_reliability)

    def select_assignment(self, env_state, deterministic=False, selector_mode="policy",
                          action_values=None, selection_diagnostics=None):
        if selector_mode != "policy" or action_values is not None:
            raise ValueError("Reliability-aware D3QN supports policy selection only.")
        observation = self.build_high_state(env_state)
        context = self._require_context(env_state)
        primary_mask, backup_pair_mask = self._context_masks(context)
        combined, energy, deadline, reliability = self.agent_R.combined_action_q_values(
            observation.vector, self.deadline_lambda, self.lambda_reliability
        )
        action = self.agent_R.choose_action_from_values(
            combined, primary_mask, is_train=not bool(deterministic)
        )
        primary_idx = self.decode_action(action)
        decision = self._decision_payload(
            observation, context, primary_mask, backup_pair_mask, primary_idx, None,
            action, float(combined[action]), selector_mode,
            np.zeros(self.max_servers, dtype=bool), False,
        )
        decision.update({"energy_q_value": float(energy[action]),
                         "deadline_q_value": float(deadline[action]),
                         "reliability_q_value": float(reliability[action]),
                         "combined_q_value": float(combined[action]),
                         "action_reliability_lambda": self.lambda_reliability,
                         **dict(selection_diagnostics or {})})
        return decision

    def remember_from_high_buffer(self, buffer):
        if buffer.size == 0:
            return {"stored": 0.0, "completed_workflows": 0.0, "reliability_cost_total": 0.0}
        # Keep the production dual-head Deadline target unchanged.  The
        # rollout buffer stores raw Delta-makespan seconds, while the Deadline
        # head is trained on the configured normalized/miss-credit signal.
        episode_failed = 0.0
        responsibility_total = 0.0
        if self.deadline_head_cost_mode == "dense_normalized_makespan_miss_credit":
            deadline_costs, deadline_misses = (
                self.workflow_dense_miss_credit_deadline_costs(
                    buffer,
                    miss_penalty=self.deadline_miss_penalty,
                    queue_weight=self.deadline_queue_weight,
                    sub_deadline_weight=self.deadline_sub_weight,
                    future_regret_weight=self.deadline_future_regret_weight,
                )
            )
        elif self.deadline_head_cost_mode == "episode_failure_counterfactual_credit":
            (
                deadline_costs,
                deadline_misses,
                episode_failed,
                responsibility_total,
            ) = self.episode_failure_counterfactual_deadline_costs(
                buffer,
                episode_deadline_target=float(
                    getattr(
                        self.deadline_lagrangian,
                        "episode_deadline_target",
                        0.92,
                    )
                ),
            )
        elif self.deadline_head_cost_mode == "dense_miss_plus_episode_failure_credit":
            (
                deadline_costs,
                deadline_misses,
                episode_failed,
                responsibility_total,
            ) = self.dense_miss_plus_episode_failure_deadline_costs(
                buffer,
                episode_deadline_target=float(
                    getattr(
                        self.deadline_lagrangian,
                        "episode_deadline_target",
                        0.92,
                    )
                ),
                miss_penalty=self.deadline_miss_penalty,
                queue_weight=self.deadline_queue_weight,
                sub_deadline_weight=self.deadline_sub_weight,
                future_regret_weight=self.deadline_future_regret_weight,
            )
        elif self.deadline_head_cost_mode == "dense_normalized_makespan":
            deadline_costs, deadline_misses = self.workflow_dense_deadline_costs(buffer)
        elif self.deadline_head_cost_mode == "dense_normalized_makespan_crossing":
            deadline_costs, deadline_misses, _ = (
                self.workflow_dense_crossing_deadline_costs(
                    buffer,
                    crossing_penalty=self.deadline_crossing_penalty,
                )
            )
        else:
            deadline_costs, deadline_misses = self.workflow_binary_deadline_credits(buffer)
        terminal_costs = np.zeros(buffer.size, dtype=np.float32)
        completed = {}
        for workflow_id, indices in buffer.workflow_groups():
            last = indices[-1]
            cost = reliability_cost_from_log_values(
                -float(buffer.partial_log_risks[last]),
                -float(buffer.reliability_limits[last]),
            )
            terminal_costs[last] = cost
            completed[str(workflow_id)] = cost
        zero_state = np.zeros(self.agent_R.state_dim, dtype=np.float32)
        zero_mask = np.zeros(self.action_dim, dtype=bool)
        for index in range(buffer.size):
            done = index == buffer.size - 1
            next_state = zero_state if done else buffer.states[index + 1]
            next_mask = zero_mask if done else np.asarray(buffer.primary_masks[index + 1], dtype=bool)
            info = buffer.infos[index]
            info["dual_head_workflow_miss"] = float(
                deadline_misses[str(buffer.workflow_ids[index])]
            )
            info["dual_head_episode_failed"] = float(episode_failed)
            info["dual_head_episode_credit"] = float(deadline_costs[index])
            self.agent_R.remember(
                buffer.states[index], self.encode_action(buffer.primary_actions[index]),
                float(info["high_base_energy_reward"]), float(deadline_costs[index]),
                float(terminal_costs[index]), next_state, done,
                np.asarray(buffer.primary_masks[index], dtype=bool), next_mask,
            )
        # The dual update is deliberately after the workflow-level costs have
        # been assembled, never once per task transition.
        self.update_reliability_lagrangian(list(completed.values()))
        return {"stored": float(buffer.size),
                "completed_workflows": float(len(completed)),
                "reliability_cost_total": float(terminal_costs.sum()),
                "reliability_cost_mean": float(terminal_costs.mean()),
                "deadline_head_cost_total": float(deadline_costs.sum()),
                "deadline_head_episode_failed": float(episode_failed),
                "deadline_head_responsibility_total": float(responsibility_total),
                "deadline_future_regret_weight": float(
                    self.deadline_future_regret_weight
                )}

    def update_reliability_lagrangian(self, completed_costs):
        rows = []
        for cost in completed_costs:
            rows.append(self.reliability_lagrangian.update(cost))
        return {"lambda_reliability": self.lambda_reliability,
                "completed_workflow_count": float(len(rows)),
                "reliability_cost_mean": float(np.mean(completed_costs)) if rows else 0.0}

    def update(self, buffer=None):
        stored = self.remember_from_high_buffer(buffer) if buffer is not None else {"stored": 0.0}
        totals = dict(stored)
        updates = []
        for _ in range(int(getattr(self, "training_per_episode", 1))):
            row = self.agent_R.learn(self.deadline_lambda, self.lambda_reliability)
            if row:
                updates.append(row)
        if updates:
            for key in updates[0]:
                if isinstance(updates[0][key], (int, float)):
                    totals[key] = float(np.mean([r[key] for r in updates]))
        totals["updates"] = float(len(updates))
        totals["lambda_reliability"] = self.lambda_reliability
        return totals
