from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from src.D3QN.D3QN import DuelingDeepQNetwork
from src.D3QN.buffer import ReplayBuffer
from src.D3QN.hierarchical_d3qn_agent import (
    HighLevelD3QNAgent,
    HierarchicalD3QNAgent,
)
from src.D3QN.utils import as_torch_rng_state
from src.device_utils import resolve_agent_device

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


EPS = 1e-8


class DualHeadConstrainedD3QN:
    """Independent energy-reward and Deadline-cost Double-DQN heads.

    The two Bellman targets are never mixed.  The current Lagrange multiplier
    is used only to select an action from ``Q_E - lambda * Q_D`` and to choose
    the shared Double-DQN next action.
    """

    def __init__(
        self,
        *,
        state_dim,
        action_dim,
        learning_rate=3e-4,
        deadline_learning_rate=None,
        hidden_dim=256,
        hidden_layers=2,
        energy_gamma=0.99,
        deadline_gamma=0.99,
        tau=0.005,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=1.8e-4,
        replay_size=50000,
        batch_size=256,
        max_grad_norm=1.0,
        deadline_positive_fraction=0.3,
        device="auto",
        seed=42,
    ):
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required for dual-head D3QN.")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.energy_gamma = float(energy_gamma)
        self.deadline_gamma = float(deadline_gamma)
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        self.eps_min = float(epsilon_min)
        self.eps_dec = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.deadline_positive_fraction = float(deadline_positive_fraction)
        if not 0.0 <= self.deadline_positive_fraction <= 1.0:
            raise ValueError("deadline_positive_fraction must be in [0, 1].")
        self.rng = np.random.default_rng(int(seed))
        self.device = resolve_agent_device(device)

        network_args = dict(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=int(hidden_dim),
            hidden_layers=int(hidden_layers),
        )
        self.energy_q_eval = DuelingDeepQNetwork(**network_args).to(self.device)
        self.energy_q_target = DuelingDeepQNetwork(**network_args).to(self.device)
        self.deadline_q_eval = DuelingDeepQNetwork(**network_args).to(self.device)
        self.deadline_q_target = DuelingDeepQNetwork(**network_args).to(self.device)
        deadline_lr = (
            float(learning_rate)
            if deadline_learning_rate is None
            else float(deadline_learning_rate)
        )
        self.energy_optimizer = torch.optim.Adam(
            self.energy_q_eval.parameters(), lr=float(learning_rate)
        )
        self.deadline_optimizer = torch.optim.Adam(
            self.deadline_q_eval.parameters(), lr=deadline_lr
        )
        # Compatibility aliases used only by the common trainer's LR logger.
        self.optimizer = self.energy_optimizer
        self.memory = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_size=int(replay_size),
            batch_size=self.batch_size,
        )
        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        tau = self.tau if tau is None else float(tau)
        pairs = (
            (self.energy_q_target, self.energy_q_eval),
            (self.deadline_q_target, self.deadline_q_eval),
        )
        for target, online in pairs:
            for target_param, online_param in zip(
                target.parameters(), online.parameters()
            ):
                target_param.data.copy_(
                    tau * online_param.data
                    + (1.0 - tau) * target_param.data
                )

    def decrement_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

    def action_q_values(self, observation):
        state = torch.as_tensor(
            np.asarray([observation], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            energy = self.energy_q_eval(state)[0]
            deadline = self.deadline_q_eval(state)[0]
        return (
            energy.detach().cpu().numpy().astype(np.float64),
            deadline.detach().cpu().numpy().astype(np.float64),
        )

    def combined_action_q_values(self, observation, deadline_lambda):
        energy, deadline = self.action_q_values(observation)
        return energy - float(deadline_lambda) * deadline, energy, deadline

    def choose_action_from_values(self, values, action_mask, *, is_train=True):
        values = np.asarray(values, dtype=np.float64)
        action_mask = np.asarray(action_mask, dtype=bool)
        valid = np.flatnonzero(action_mask)
        if valid.size <= 0:
            raise ValueError("Dual-head D3QN action mask contains no valid action.")
        if values.shape != (self.action_dim,) or not np.isfinite(values).all():
            raise ValueError("Dual-head D3QN action values are invalid.")
        if bool(is_train) and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid))
        return int(np.argmax(np.where(action_mask, values, -np.inf)))

    def remember(
        self,
        state,
        action,
        energy_reward,
        deadline_cost,
        next_state,
        done,
        action_mask,
        next_action_mask,
    ):
        self.memory.store_transition(
            state,
            action,
            energy_reward,
            next_state,
            done,
            action_mask,
            next_action_mask,
            base_energy_reward=energy_reward,
            deadline_constraint_cost=deadline_cost,
        )

    def _uniform_indices(self):
        if not self.memory.ready():
            return None
        return self.rng.choice(
            self.memory.size, self.batch_size, replace=False
        ).astype(np.int64)

    def _deadline_indices_and_weights(self):
        if not self.memory.ready():
            return None, None
        size = self.memory.size
        costs = self.memory.deadline_constraint_costs[:size]
        positive = np.flatnonzero(costs > EPS)
        negative = np.flatnonzero(costs <= EPS)
        if positive.size == 0 or negative.size == 0:
            indices = self.rng.choice(size, self.batch_size, replace=False)
            return indices.astype(np.int64), np.ones(self.batch_size, dtype=np.float32)

        wanted_positive = max(
            1, int(round(self.batch_size * self.deadline_positive_fraction))
        )
        positive_count = min(wanted_positive, positive.size)
        negative_count = min(self.batch_size - positive_count, negative.size)
        missing = self.batch_size - positive_count - negative_count
        if missing > 0:
            positive_count += min(missing, positive.size - positive_count)
            missing = self.batch_size - positive_count - negative_count
        if missing > 0:
            negative_count += min(missing, negative.size - negative_count)
        if positive_count + negative_count != self.batch_size:
            raise RuntimeError("Unable to construct a full Deadline replay batch.")

        positive_batch = self.rng.choice(
            positive, positive_count, replace=False
        )
        negative_batch = self.rng.choice(
            negative, negative_count, replace=False
        )
        indices = np.concatenate([positive_batch, negative_batch]).astype(np.int64)
        # Horvitz-Thompson class weights remove the bias introduced by the
        # positive/negative quota while retaining more miss examples per batch.
        weights = np.concatenate(
            [
                np.full(positive_count, positive.size / positive_count),
                np.full(negative_count, negative.size / negative_count),
            ]
        ).astype(np.float32)
        order = self.rng.permutation(self.batch_size)
        indices = indices[order]
        weights = weights[order]
        weights /= max(float(weights.mean()), EPS)
        return indices, weights

    def _batch(self, indices):
        return {
            "states": self.memory.states[indices],
            "actions": self.memory.actions[indices],
            "energy_rewards": self.memory.base_energy_rewards[indices],
            "deadline_costs": self.memory.deadline_constraint_costs[indices],
            "next_states": self.memory.next_states[indices],
            "dones": self.memory.dones[indices],
            "next_masks": self.memory.next_action_masks[indices],
        }

    def _next_actions(self, next_states, next_masks, deadline_lambda):
        with torch.no_grad():
            energy = self.energy_q_eval(next_states)
            deadline = self.deadline_q_eval(next_states)
            combined = energy - float(deadline_lambda) * deadline
            combined = combined.masked_fill(~next_masks, -1e9)
            return torch.argmax(combined, dim=-1)

    def _loss_for_head(
        self,
        batch,
        *,
        online,
        target_network,
        rewards_key,
        gamma,
        deadline_lambda,
        sample_weights=None,
    ):
        states = torch.as_tensor(
            batch["states"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.long, device=self.device
        )
        rewards = torch.as_tensor(
            batch[rewards_key], dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            batch["next_states"], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            batch["dones"], dtype=torch.bool, device=self.device
        )
        next_masks = torch.as_tensor(
            batch["next_masks"], dtype=torch.bool, device=self.device
        )
        row = torch.arange(actions.shape[0], device=self.device)
        q_value = online(states)[row, actions]
        with torch.no_grad():
            next_actions = self._next_actions(
                next_states, next_masks, deadline_lambda
            )
            next_q = target_network(next_states)[row, next_actions]
            no_valid = next_masks.sum(dim=-1) <= 0
            next_q = torch.where(
                dones | no_valid, torch.zeros_like(next_q), next_q
            )
            target = rewards + float(gamma) * next_q
        losses = F.smooth_l1_loss(q_value, target, reduction="none")
        if sample_weights is not None:
            weights = torch.as_tensor(
                sample_weights, dtype=torch.float32, device=self.device
            )
            loss = (losses * weights).mean()
        else:
            loss = losses.mean()
        return loss, q_value, target, rewards

    def learn(self, deadline_lambda):
        energy_indices = self._uniform_indices()
        if energy_indices is None:
            return {}
        deadline_indices, deadline_weights = self._deadline_indices_and_weights()
        energy_batch = self._batch(energy_indices)
        deadline_batch = self._batch(deadline_indices)

        energy_loss, energy_q, energy_target, energy_reward = self._loss_for_head(
            energy_batch,
            online=self.energy_q_eval,
            target_network=self.energy_q_target,
            rewards_key="energy_rewards",
            gamma=self.energy_gamma,
            deadline_lambda=deadline_lambda,
        )
        deadline_loss, deadline_q, deadline_target, deadline_cost = self._loss_for_head(
            deadline_batch,
            online=self.deadline_q_eval,
            target_network=self.deadline_q_target,
            rewards_key="deadline_costs",
            gamma=self.deadline_gamma,
            deadline_lambda=deadline_lambda,
            sample_weights=deadline_weights,
        )
        if not bool(torch.isfinite(energy_loss + deadline_loss).item()):
            raise FloatingPointError("Dual-head D3QN loss contains NaN or Inf.")

        self.energy_optimizer.zero_grad()
        energy_loss.backward()
        energy_grad = torch.nn.utils.clip_grad_norm_(
            self.energy_q_eval.parameters(), self.max_grad_norm
        )
        self.energy_optimizer.step()

        self.deadline_optimizer.zero_grad()
        deadline_loss.backward()
        deadline_grad = torch.nn.utils.clip_grad_norm_(
            self.deadline_q_eval.parameters(), self.max_grad_norm
        )
        self.deadline_optimizer.step()
        self.update_network_parameters()
        self.decrement_epsilon()

        energy_td = (energy_target - energy_q).detach()
        deadline_td = (deadline_target - deadline_q).detach()
        return {
            "loss": float((energy_loss + deadline_loss).item()),
            "energy_loss": float(energy_loss.item()),
            "deadline_loss": float(deadline_loss.item()),
            "q_mean": float(
                (energy_q - float(deadline_lambda) * deadline_q).mean().item()
            ),
            "energy_q_mean": float(energy_q.detach().mean().item()),
            "deadline_q_mean": float(deadline_q.detach().mean().item()),
            "target_mean": float(energy_target.detach().mean().item()),
            "energy_target_mean": float(energy_target.detach().mean().item()),
            "deadline_target_mean": float(deadline_target.detach().mean().item()),
            "td_abs_mean": float(
                0.5
                * (
                    energy_td.abs().mean().item()
                    + deadline_td.abs().mean().item()
                )
            ),
            "energy_td_abs_mean": float(energy_td.abs().mean().item()),
            "deadline_td_abs_mean": float(deadline_td.abs().mean().item()),
            "grad_norm_before_clip": float(
                max(float(energy_grad), float(deadline_grad))
            ),
            "energy_grad_norm_before_clip": float(energy_grad),
            "deadline_grad_norm_before_clip": float(deadline_grad),
            "reward_mean": float(energy_reward.mean().item()),
            "reward_std": float(energy_reward.std(unbiased=False).item()),
            "deadline_cost_batch_mean": float(deadline_cost.mean().item()),
            "deadline_positive_replay_fraction": float(
                np.mean(deadline_batch["deadline_costs"] > EPS)
            ),
            "deadline_lambda_for_action": float(deadline_lambda),
        }

    def state_dict(self, *, include_replay=False):
        return {
            "energy_q_eval": self.energy_q_eval.state_dict(),
            "energy_q_target": self.energy_q_target.state_dict(),
            "deadline_q_eval": self.deadline_q_eval.state_dict(),
            "deadline_q_target": self.deadline_q_target.state_dict(),
            "energy_optimizer": self.energy_optimizer.state_dict(),
            "deadline_optimizer": self.deadline_optimizer.state_dict(),
            "epsilon": self.epsilon,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "replay": self.memory.state_dict() if include_replay else None,
        }

    def load_state_dict(self, payload):
        self.energy_q_eval.load_state_dict(payload["energy_q_eval"])
        self.energy_q_target.load_state_dict(payload["energy_q_target"])
        self.deadline_q_eval.load_state_dict(payload["deadline_q_eval"])
        self.deadline_q_target.load_state_dict(payload["deadline_q_target"])
        self.energy_optimizer.load_state_dict(payload["energy_optimizer"])
        self.deadline_optimizer.load_state_dict(payload["deadline_optimizer"])
        self.epsilon = float(payload.get("epsilon", self.epsilon))
        if payload.get("rng_state") is not None:
            self.rng.bit_generator.state = copy.deepcopy(payload["rng_state"])
        if payload.get("replay") is not None:
            self.memory.load_state_dict(payload["replay"])


class DualHeadHighLevelD3QNAgent(HighLevelD3QNAgent):
    DEADLINE_HEAD_COST_MODES = {
        "binary_miss_credit",
        "dense_normalized_makespan",
        "dense_normalized_makespan_crossing",
        "dense_normalized_makespan_miss_credit",
    }

    def __init__(self, config=None, seed=42):
        super().__init__(config=config, seed=seed)
        config = dict(config or {})
        self.deadline_head_cost_mode = str(
            config.get("deadline_head_cost_mode", "binary_miss_credit")
        ).strip().lower()
        if self.deadline_head_cost_mode not in self.DEADLINE_HEAD_COST_MODES:
            raise ValueError(
                "deadline_head_cost_mode must be one of "
                f"{sorted(self.DEADLINE_HEAD_COST_MODES)}, got "
                f"{self.deadline_head_cost_mode!r}."
            )
        self.deadline_crossing_penalty = float(
            config.get("deadline_crossing_penalty", 1.0)
        )
        if self.deadline_crossing_penalty < 0.0:
            raise ValueError("deadline_crossing_penalty must be non-negative.")
        self.deadline_miss_penalty = float(
            config.get("deadline_miss_penalty", 1.0)
        )
        if self.deadline_miss_penalty < 0.0:
            raise ValueError("deadline_miss_penalty must be non-negative.")
        self.agent_R = DualHeadConstrainedD3QN(
            state_dim=self.encoder.high_state_dim,
            action_dim=self.action_dim,
            learning_rate=float(config.get("learning_rate", 3e-4)),
            deadline_learning_rate=float(
                config.get(
                    "deadline_learning_rate",
                    config.get("learning_rate", 3e-4),
                )
            ),
            hidden_dim=int(config.get("hidden_dim", 256)),
            hidden_layers=int(config.get("hidden_layers", 2)),
            energy_gamma=float(config.get("gamma", config.get("reward_gamma", 0.99))),
            deadline_gamma=float(config.get("deadline_gamma", 0.99)),
            tau=float(config.get("tau", 0.005)),
            epsilon=float(config.get("epsilon", 1.0)),
            epsilon_min=float(config.get("epsilon_min", 0.1)),
            epsilon_decay=float(config.get("epsilon_decay", 1.8e-4)),
            replay_size=int(config.get("replay_size", 50000)),
            batch_size=int(config.get("batch_size", 256)),
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            deadline_positive_fraction=float(
                config.get("deadline_positive_fraction", 0.3)
            ),
            device=config.get("device", "auto"),
            seed=seed,
        )

    def select_assignment(
        self,
        env_state,
        deterministic=False,
        selector_mode="policy",
        action_values=None,
        selection_diagnostics=None,
    ):
        if selector_mode != "policy":
            raise ValueError("Dual-head D3QN supports only policy selection.")
        if action_values is not None:
            raise ValueError("External action re-ranking is disabled for dual-head D3QN.")
        observation = self.build_high_state(env_state)
        context = self._require_context(env_state)
        primary_mask, backup_pair_mask = self._context_masks(context)
        combined, energy_values, deadline_values = (
            self.agent_R.combined_action_q_values(
                observation.vector, self.deadline_lambda
            )
        )
        valid = np.asarray(primary_mask, dtype=bool)
        energy_action_std = float(np.std(energy_values[valid]))
        deadline_action_std = float(np.std(deadline_values[valid]))
        deadline_to_energy_scale = float(
            float(self.deadline_lambda)
            * deadline_action_std
            / max(energy_action_std, EPS)
        )
        action = self.agent_R.choose_action_from_values(
            combined, primary_mask, is_train=not bool(deterministic)
        )
        primary_idx = self.decode_action(action)
        decision = self._decision_payload(
            observation,
            context,
            primary_mask,
            backup_pair_mask,
            primary_idx,
            None,
            action,
            float(combined[action]),
            selector_mode,
            np.zeros(self.max_servers, dtype=bool),
            False,
        )
        decision.update(
            {
                "energy_q_value": float(energy_values[action]),
                "deadline_q_value": float(deadline_values[action]),
                "combined_q_value": float(combined[action]),
                "action_deadline_lambda": float(self.deadline_lambda),
                "dual_energy_action_std": energy_action_std,
                "dual_deadline_action_std": deadline_action_std,
                "dual_deadline_to_energy_scale": deadline_to_energy_scale,
                **dict(selection_diagnostics or {}),
            }
        )
        return decision

    @staticmethod
    def workflow_binary_deadline_credits(buffer):
        credits = np.zeros(buffer.size, dtype=np.float32)
        misses = {}
        for workflow_id, indices in buffer.workflow_groups():
            final_index = indices[-1]
            deadline = max(float(buffer.deadline_limits[final_index]), EPS)
            completion = float(buffer.partial_makespans[final_index])
            miss = float(completion > deadline + EPS)
            increments = np.asarray(
                [max(0.0, float(buffer.deadline_costs[index])) for index in indices],
                dtype=np.float64,
            )
            total = float(increments.sum())
            if total <= EPS:
                weights = np.full(len(indices), 1.0 / max(len(indices), 1))
            else:
                weights = increments / total
            credits[np.asarray(indices, dtype=np.int64)] = (
                miss * weights
            ).astype(np.float32)
            misses[str(workflow_id)] = miss
        return credits, misses

    @staticmethod
    def workflow_dense_deadline_costs(buffer):
        """Return dense per-step Delta-C / D costs and terminal miss labels."""
        costs = np.zeros(buffer.size, dtype=np.float32)
        misses = {}
        for workflow_id, indices in buffer.workflow_groups():
            final_index = indices[-1]
            final_deadline = max(
                float(buffer.deadline_limits[final_index]), EPS
            )
            completion = float(buffer.partial_makespans[final_index])
            misses[str(workflow_id)] = float(
                completion > final_deadline + EPS
            )
            for index in indices:
                deadline = max(float(buffer.deadline_limits[index]), EPS)
                increment = max(0.0, float(buffer.deadline_costs[index]))
                costs[index] = float(increment / deadline)
        return costs, misses

    @staticmethod
    def workflow_dense_crossing_deadline_costs(
        buffer, *, crossing_penalty=1.0
    ):
        """Return Delta-C/D plus a one-off cost when a workflow crosses D."""
        crossing_penalty = float(crossing_penalty)
        if crossing_penalty < 0.0:
            raise ValueError("crossing_penalty must be non-negative.")
        costs = np.zeros(buffer.size, dtype=np.float32)
        misses = {}
        crossing_count = 0
        for workflow_id, indices in buffer.workflow_groups():
            final_index = indices[-1]
            final_deadline = max(
                float(buffer.deadline_limits[final_index]), EPS
            )
            completion = float(buffer.partial_makespans[final_index])
            misses[str(workflow_id)] = float(
                completion > final_deadline + EPS
            )
            previous_completion = 0.0
            for index in indices:
                deadline = max(float(buffer.deadline_limits[index]), EPS)
                current_completion = float(buffer.partial_makespans[index])
                increment = max(0.0, float(buffer.deadline_costs[index]))
                crossed = bool(
                    previous_completion <= deadline + EPS
                    and current_completion > deadline + EPS
                )
                costs[index] = float(
                    increment / deadline
                    + crossing_penalty * float(crossed)
                )
                crossing_count += int(crossed)
                previous_completion = current_completion
        return costs, misses, crossing_count

    @staticmethod
    def workflow_dense_miss_credit_deadline_costs(
        buffer, *, miss_penalty=1.0
    ):
        """Return Delta-C/D plus final miss credit distributed by Delta-C."""
        miss_penalty = float(miss_penalty)
        if miss_penalty < 0.0:
            raise ValueError("miss_penalty must be non-negative.")
        costs = np.zeros(buffer.size, dtype=np.float32)
        misses = {}
        for workflow_id, indices in buffer.workflow_groups():
            final_index = indices[-1]
            final_deadline = max(
                float(buffer.deadline_limits[final_index]), EPS
            )
            completion = float(buffer.partial_makespans[final_index])
            miss = float(completion > final_deadline + EPS)
            misses[str(workflow_id)] = miss
            increments = np.asarray(
                [max(0.0, float(buffer.deadline_costs[index])) for index in indices],
                dtype=np.float64,
            )
            total_increment = float(increments.sum())
            if total_increment <= EPS:
                weights = np.full(len(indices), 1.0 / max(len(indices), 1))
            else:
                weights = increments / total_increment
            for local_index, index in enumerate(indices):
                deadline = max(float(buffer.deadline_limits[index]), EPS)
                costs[index] = float(
                    increments[local_index] / deadline
                    + miss_penalty * miss * weights[local_index]
                )
        return costs, misses

    def remember_from_high_buffer(self, buffer):
        if buffer.size == 0:
            return {
                "stored": 0.0,
                "deadline_reward_penalty_total": 0.0,
                "binary_deadline_cost_total": 0.0,
            }
        crossing_count = 0
        if self.deadline_head_cost_mode == "dense_normalized_makespan_miss_credit":
            credits, misses = self.workflow_dense_miss_credit_deadline_costs(
                buffer,
                miss_penalty=self.deadline_miss_penalty,
            )
        elif (
            self.deadline_head_cost_mode
            == "dense_normalized_makespan_crossing"
        ):
            credits, misses, crossing_count = (
                self.workflow_dense_crossing_deadline_costs(
                    buffer,
                    crossing_penalty=self.deadline_crossing_penalty,
                )
            )
        elif self.deadline_head_cost_mode == "dense_normalized_makespan":
            credits, misses = self.workflow_dense_deadline_costs(buffer)
        else:
            credits, misses = self.workflow_binary_deadline_credits(buffer)
        zero_state = np.zeros(self.encoder.high_state_dim, dtype=np.float32)
        zero_mask = np.zeros(self.action_dim, dtype=bool)
        for index in range(buffer.size):
            done = index == buffer.size - 1
            next_state = zero_state if done else buffer.states[index + 1]
            next_mask = zero_mask if done else np.asarray(
                buffer.primary_masks[index + 1], dtype=bool
            ).copy()
            info = buffer.infos[index]
            energy_reward = float(info["high_base_energy_reward"])
            deadline_cost = float(credits[index])
            info["dual_head_energy_reward"] = energy_reward
            info["dual_head_deadline_cost"] = deadline_cost
            info["dual_head_dense_deadline_cost"] = float(
                self.deadline_head_cost_mode
                in {
                    "dense_normalized_makespan",
                    "dense_normalized_makespan_crossing",
                    "dense_normalized_makespan_miss_credit",
                }
            )
            info["dual_head_deadline_crossing_cost"] = float(
                self.deadline_head_cost_mode
                == "dense_normalized_makespan_crossing"
            )
            info["dual_head_workflow_miss"] = float(
                misses[str(buffer.workflow_ids[index])]
            )
            self.agent_R.remember(
                buffer.states[index],
                self.encode_action(buffer.primary_actions[index]),
                energy_reward,
                deadline_cost,
                next_state,
                done,
                np.asarray(buffer.primary_masks[index], dtype=bool).copy(),
                next_mask,
            )
        return {
            "stored": float(buffer.size),
            "deadline_reward_penalty_total": 0.0,
            "deadline_head_cost_total": float(credits.sum()),
            "binary_deadline_cost_total": float(sum(misses.values())),
            "binary_missed_workflows": float(sum(misses.values())),
            "deadline_head_crossing_count": float(crossing_count),
            "deadline_crossing_penalty": float(
                self.deadline_crossing_penalty
            ),
            "deadline_miss_penalty": float(self.deadline_miss_penalty),
            "dense_deadline_cost_enabled": float(
                self.deadline_head_cost_mode
                in {
                    "dense_normalized_makespan",
                    "dense_normalized_makespan_crossing",
                    "dense_normalized_makespan_miss_credit",
                }
            ),
        }

    def update(self, buffer=None):
        remember = (
            self.remember_from_high_buffer(buffer)
            if buffer is not None
            else {
                "stored": 0.0,
                "deadline_reward_penalty_total": 0.0,
                "deadline_head_cost_total": 0.0,
                "binary_deadline_cost_total": 0.0,
                "binary_missed_workflows": 0.0,
                "deadline_head_crossing_count": 0.0,
                "deadline_crossing_penalty": float(
                    self.deadline_crossing_penalty
                ),
                "deadline_miss_penalty": float(self.deadline_miss_penalty),
                "dense_deadline_cost_enabled": float(
                    self.deadline_head_cost_mode
                    in {
                        "dense_normalized_makespan",
                        "dense_normalized_makespan_crossing",
                        "dense_normalized_makespan_miss_credit",
                    }
                ),
            }
        )
        totals = {
            "replay_size": float(self.replay.size),
            "epsilon": float(self.epsilon),
            "stored_transitions": float(remember["stored"]),
            "deadline_reward_penalty_total": 0.0,
            "deadline_cost_reward_weight": 0.0,
            "deadline_head_cost_total": float(
                remember.get("deadline_head_cost_total", 0.0)
            ),
            "binary_deadline_cost_total": float(
                remember.get("binary_deadline_cost_total", 0.0)
            ),
            "binary_missed_workflows": float(
                remember.get("binary_missed_workflows", 0.0)
            ),
            "deadline_head_crossing_count": float(
                remember.get("deadline_head_crossing_count", 0.0)
            ),
            "deadline_crossing_penalty": float(
                remember.get(
                    "deadline_crossing_penalty",
                    self.deadline_crossing_penalty,
                )
            ),
            "deadline_miss_penalty": float(
                remember.get("deadline_miss_penalty", self.deadline_miss_penalty)
            ),
            "dense_deadline_cost_enabled": float(
                remember.get("dense_deadline_cost_enabled", 0.0)
            ),
        }
        updates = 0
        for _ in range(self.training_per_episode):
            metrics = self.agent_R.learn(self.deadline_lambda)
            if not metrics:
                continue
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            updates += 1
        fixed = {
            "replay_size",
            "epsilon",
            "stored_transitions",
            "deadline_reward_penalty_total",
            "deadline_cost_reward_weight",
            "deadline_head_cost_total",
            "binary_deadline_cost_total",
            "binary_missed_workflows",
            "deadline_head_crossing_count",
            "deadline_crossing_penalty",
            "deadline_miss_penalty",
            "dense_deadline_cost_enabled",
        }
        if updates:
            for key in list(totals):
                if key not in fixed:
                    totals[key] /= updates
        totals["updates"] = float(updates)
        totals["epsilon"] = float(self.epsilon)
        return totals


class DualHeadHierarchicalD3QNAgent(HierarchicalD3QNAgent):
    """Compatibility wrapper for the existing environment/rollout trainer."""

    CHECKPOINT_FORMAT = "hierarchical_dual_head_d3qn_v1"

    def __init__(self, config=None, seed=42):
        config = copy.deepcopy(dict(config or {}))
        use_low_agent = bool(config.get("use_low_agent", False))
        coordination = dict(config.get("coordination_critic", {}))
        if bool(coordination.get("enabled", False)):
            raise ValueError("Dual-head experiment does not use C3C/HCC.")
        super().__init__(config=config, seed=seed)
        shared = dict(config.get("shared", {}))
        high_config = {**shared, **dict(config.get("high", {}))}
        self.high_agent = DualHeadHighLevelD3QNAgent(high_config, seed=seed)
        if not use_low_agent:
            self.low_agent = None
            self.use_low_agent = False
        self.coordination_critic = None
        self.coordination_critic_enabled = False

    def save_checkpoint(
        self,
        path,
        *,
        objective_state=None,
        episode=0,
        training_phase=None,
        config=None,
        include_replay=False,
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_format": self.CHECKPOINT_FORMAT,
            "deadline_head_cost_mode": (
                self.high_agent.deadline_head_cost_mode
            ),
            "deadline_crossing_penalty": float(
                self.high_agent.deadline_crossing_penalty
            ),
            "deadline_miss_penalty": float(
                self.high_agent.deadline_miss_penalty
            ),
            "dual_head_state_dict": self.high_agent.agent_R.state_dict(
                include_replay=include_replay
            ),
            "deadline_lagrangian": (
                None
                if self.high_agent.deadline_lagrangian is None
                else self.high_agent.deadline_lagrangian.state_dict()
            ),
            "use_low_agent": bool(self.use_low_agent),
            "low_algorithm": self.low_algorithm,
            "low_model_state_dict": self._low_model_state_dict(
                include_replay=include_replay
            ),
            "objective_state": dict(objective_state or {}),
            "episode": int(episode),
            "training_phase": training_phase,
            "config": copy.deepcopy(config if config is not None else self.config),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        torch.save(payload, str(path))
        component_path = self._component_checkpoint_path(path, "high")
        high_payload = {
            **payload,
            "use_low_agent": False,
            "low_model_state_dict": None,
        }
        torch.save(high_payload, str(component_path))
        if self.low_agent is not None:
            low_payload = {
                "checkpoint_format": "hierarchical_dual_head_low_sac_v1",
                "low_algorithm": self.low_algorithm,
                "low_model_state_dict": payload["low_model_state_dict"],
                "objective_state": dict(objective_state or {}),
                "episode": int(episode),
                "training_phase": training_phase,
                "config": copy.deepcopy(
                    config if config is not None else self.config
                ),
            }
            torch.save(
                low_payload,
                str(self._component_checkpoint_path(path, "low")),
            )

    def load_checkpoint(self, path):
        payload = torch.load(
            str(path), map_location=self.high_agent.agent_R.device, weights_only=False
        )
        if payload.get("checkpoint_format") != self.CHECKPOINT_FORMAT:
            raise RuntimeError("Checkpoint is not a dual-head constrained D3QN checkpoint.")
        saved_cost_mode = payload.get("deadline_head_cost_mode")
        if (
            saved_cost_mode is not None
            and str(saved_cost_mode) != self.high_agent.deadline_head_cost_mode
        ):
            raise RuntimeError(
                "Dual-head checkpoint Deadline-cost mode mismatch: "
                f"checkpoint={saved_cost_mode!r}, "
                f"config={self.high_agent.deadline_head_cost_mode!r}."
            )
        saved_crossing_penalty = payload.get("deadline_crossing_penalty")
        if (
            saved_crossing_penalty is not None
            and not np.isclose(
                float(saved_crossing_penalty),
                self.high_agent.deadline_crossing_penalty,
            )
        ):
            raise RuntimeError(
                "Dual-head checkpoint crossing-penalty mismatch: "
                f"checkpoint={float(saved_crossing_penalty)!r}, "
                f"config={self.high_agent.deadline_crossing_penalty!r}."
            )
        saved_miss_penalty = payload.get("deadline_miss_penalty")
        if (
            saved_miss_penalty is not None
            and not np.isclose(
                float(saved_miss_penalty),
                self.high_agent.deadline_miss_penalty,
            )
        ):
            raise ValueError(
                "Dual-head checkpoint miss-penalty mismatch: "
                f"checkpoint={float(saved_miss_penalty)!r}, "
                f"config={self.high_agent.deadline_miss_penalty!r}."
            )
        self.high_agent.agent_R.load_state_dict(payload["dual_head_state_dict"])
        low_state = payload.get("low_model_state_dict")
        if self.low_agent is not None and low_state is not None:
            saved_low_algorithm = str(
                payload.get("low_algorithm", self.low_algorithm)
            ).strip().lower()
            if saved_low_algorithm != self.low_algorithm:
                raise RuntimeError(
                    "Dual-head checkpoint low-algorithm mismatch: "
                    f"checkpoint={saved_low_algorithm!r}, "
                    f"config={self.low_algorithm!r}."
                )
            if self.low_algorithm != "sac":
                raise RuntimeError("Dual-head lower-level resume supports SAC only.")
            self.low_agent.load_state_dict(copy.deepcopy(low_state))
        if (
            self.high_agent.deadline_lagrangian is not None
            and payload.get("deadline_lagrangian") is not None
        ):
            self.high_agent.deadline_lagrangian.load_state_dict(
                payload["deadline_lagrangian"]
            )
        if payload.get("numpy_random_state") is not None:
            np.random.set_state(payload["numpy_random_state"])
        if payload.get("torch_random_state") is not None:
            torch.random.set_rng_state(
                as_torch_rng_state(payload["torch_random_state"])
            )
        if torch.cuda.is_available() and payload.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(
                [as_torch_rng_state(item) for item in payload["cuda_random_state"]]
            )
        return payload

    def load_high_checkpoint(self, path, *, freeze=False):
        payload = self.load_checkpoint(path)
        if freeze:
            networks = (
                self.high_agent.agent_R.energy_q_eval,
                self.high_agent.agent_R.energy_q_target,
                self.high_agent.agent_R.deadline_q_eval,
                self.high_agent.agent_R.deadline_q_target,
            )
            for network in networks:
                network.eval()
                for parameter in network.parameters():
                    parameter.requires_grad_(False)
        return payload
