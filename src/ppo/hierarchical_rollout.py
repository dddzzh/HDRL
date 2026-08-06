from __future__ import annotations

import math

import numpy as np


EPS = 1e-8


def _assert_finite_array(name, values):
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains NaN or Inf.")
    return array


class HighRolloutBuffer:
    def __init__(self):
        self.clear()

    @property
    def size(self):
        return len(self.states)

    def clear(self):
        self.states = []
        self.primary_actions = []
        self.backup_actions = []
        self.old_log_probs = []
        self.log_probs = self.old_log_probs
        self.reward_values = []
        self.deadline_values = []
        self.reliability_values = []
        self.rewards = []
        self.deadline_costs = []
        self.reliability_costs = []
        self.deadline_limits = []
        self.reliability_limits = []
        self.partial_makespans = []
        self.partial_log_risks = []
        self.workflow_ids = []
        self.workflow_dones = []
        self.episode_dones = []
        self.next_states = []
        self.primary_masks = []
        self.backup_pair_masks = []
        self.infos = []
        self.reward_returns = []
        self.reward_advantages = []
        self.deadline_returns = []
        self.deadline_advantages = []
        self.reliability_returns = []
        self.reliability_advantages = []
        self.completed_workflow_ids = set()

    def add(
        self,
        *,
        state,
        primary_action,
        backup_action,
        old_log_prob,
        reward_value,
        deadline_value,
        reliability_value,
        reward,
        deadline_cost,
        reliability_cost,
        deadline_limit,
        reliability_limit,
        partial_makespan,
        partial_log_risk,
        workflow_id,
        workflow_done,
        episode_done,
        next_state,
        primary_mask,
        backup_pair_mask,
        info=None,
    ):
        row = {
            "reward": float(reward),
            "deadline_cost": float(deadline_cost),
            "reliability_cost": float(reliability_cost),
            "deadline_limit": float(deadline_limit),
            "reliability_limit": float(reliability_limit),
            "partial_makespan": float(partial_makespan),
            "partial_log_risk": float(partial_log_risk),
        }
        for key, value in row.items():
            if not math.isfinite(value):
                raise FloatingPointError(f"High rollout field {key} is not finite: {value}")
        primary_mask = np.asarray(primary_mask, dtype=bool)
        backup_pair_mask = np.asarray(backup_pair_mask, dtype=bool)
        if primary_mask.ndim != 1:
            raise ValueError("primary_mask must be one-dimensional.")
        if backup_pair_mask.ndim != 2 or backup_pair_mask.shape[0] != backup_pair_mask.shape[1]:
            raise ValueError("backup_pair_mask must be square.")
        if backup_pair_mask.shape[0] != primary_mask.shape[0]:
            raise ValueError("backup_pair_mask and primary_mask sizes do not match.")
        if int(primary_mask.sum()) <= 0:
            raise ValueError("primary_mask has no valid action.")
        primary_action = int(primary_action)
        backup_action = -1 if backup_action is None else int(backup_action)
        if not bool(primary_mask[primary_action]):
            raise ValueError("primary_action violates primary_mask.")
        backup_disabled = backup_action < 0
        if backup_disabled:
            if bool(backup_pair_mask.any()):
                raise ValueError("disabled backup_action requires an empty backup_pair_mask.")
        else:
            if not bool(backup_pair_mask[primary_action, backup_action]):
                raise ValueError("backup_action violates backup_pair_mask for selected primary.")
            if primary_action == backup_action:
                raise ValueError("primary and backup actions must differ.")

        self.states.append(np.asarray(state, dtype=np.float32))
        self.primary_actions.append(primary_action)
        self.backup_actions.append(backup_action)
        self.old_log_probs.append(float(old_log_prob))
        self.reward_values.append(float(reward_value))
        self.deadline_values.append(float(deadline_value))
        self.reliability_values.append(float(reliability_value))
        self.rewards.append(float(reward))
        self.deadline_costs.append(float(deadline_cost))
        self.reliability_costs.append(float(reliability_cost))
        self.deadline_limits.append(float(deadline_limit))
        self.reliability_limits.append(float(reliability_limit))
        self.partial_makespans.append(float(partial_makespan))
        self.partial_log_risks.append(float(partial_log_risk))
        self.workflow_ids.append(str(workflow_id))
        self.workflow_dones.append(bool(workflow_done))
        self.episode_dones.append(bool(episode_done))
        self.next_states.append(None if next_state is None else np.asarray(next_state, dtype=np.float32))
        self.primary_masks.append(primary_mask.copy())
        self.backup_pair_masks.append(backup_pair_mask.copy())
        info_row = dict(info or {})
        info_row.update(row)
        info_row["workflow_id"] = str(workflow_id)
        info_row["workflow_done"] = bool(workflow_done)
        self.infos.append(info_row)

    def patch_last_workflow_transition(self, workflow_id, *, reward_delta=0.0, deadline_cost_delta=0.0, partial_makespan=None, info_updates=None):
        workflow_id = str(workflow_id)
        for index in range(self.size - 1, -1, -1):
            if self.workflow_ids[index] != workflow_id:
                continue
            self.rewards[index] = float(self.rewards[index]) + float(reward_delta)
            self.deadline_costs[index] = float(self.deadline_costs[index]) + float(deadline_cost_delta)
            self.infos[index]["reward"] = float(self.rewards[index])
            self.infos[index]["deadline_cost"] = float(self.deadline_costs[index])
            if partial_makespan is not None:
                self.partial_makespans[index] = float(partial_makespan)
                self.infos[index]["partial_makespan"] = float(partial_makespan)
            if info_updates:
                self.infos[index].update(info_updates)
            return True
        return False

    def mark_workflow_done(self, workflow_id):
        workflow_id = str(workflow_id)
        if workflow_id in self.completed_workflow_ids:
            return False
        for index in range(self.size - 1, -1, -1):
            if self.workflow_ids[index] == workflow_id:
                self.workflow_dones[index] = True
                self.infos[index]["workflow_done"] = True
                self.completed_workflow_ids.add(workflow_id)
                return True
        return False

    def workflow_groups(self):
        groups = {}
        order = []
        for index, workflow_id in enumerate(self.workflow_ids):
            if workflow_id not in groups:
                groups[workflow_id] = []
                order.append(workflow_id)
            groups[workflow_id].append(index)
        return [(workflow_id, groups[workflow_id]) for workflow_id in order]

    def workflow_minibatches(self, batch_size, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        groups = self.workflow_groups()
        order = np.arange(len(groups))
        rng.shuffle(order)
        batches = []
        current = []
        current_size = 0
        for group_pos in order:
            _, indices = groups[int(group_pos)]
            if current and current_size + len(indices) > int(batch_size):
                batches.append(np.asarray(current, dtype=np.int64))
                current = []
                current_size = 0
            if len(indices) > int(batch_size):
                batches.append(np.asarray(indices, dtype=np.int64))
                continue
            current.extend(indices)
            current_size += len(indices)
        if current:
            batches.append(np.asarray(current, dtype=np.int64))
        return batches

    def compute_reward_returns_and_advantages(self, gamma, gae_lambda):
        self.reward_returns, self.reward_advantages = self._compute_returns_by_workflow(
            self.rewards,
            self.reward_values,
            gamma,
            gae_lambda,
        )
        return self.reward_returns, self.reward_advantages

    def compute_cost_returns_and_advantages(self, cost_name, gamma=1.0, gae_lambda=1.0):
        if cost_name == "deadline":
            costs = self.deadline_costs
            values = self.deadline_values
        elif cost_name == "reliability":
            costs = self.reliability_costs
            values = self.reliability_values
        else:
            raise ValueError(f"Unknown cost stream: {cost_name}")
        returns, advantages = self._compute_returns_by_workflow(costs, values, gamma, gae_lambda)
        if cost_name == "deadline":
            self.deadline_returns = returns
            self.deadline_advantages = advantages
        else:
            self.reliability_returns = returns
            self.reliability_advantages = advantages
        return returns, advantages

    def compute_normalized_deadline_returns_and_advantages(self, gamma=1.0, gae_lambda=1.0):
        limits = _assert_finite_array("deadline_limits", self.deadline_limits)
        if np.any(limits <= EPS):
            raise RuntimeError("deadline_limits must be positive for normalized deadline training.")
        costs = _assert_finite_array("deadline_costs", self.deadline_costs) / limits
        returns, advantages = self._compute_returns_by_workflow(
            costs,
            self.deadline_values,
            gamma,
            gae_lambda,
        )
        self.deadline_returns = returns
        self.deadline_advantages = advantages
        return returns, advantages

    def normalized_partial_makespans(self):
        limits = _assert_finite_array("deadline_limits", self.deadline_limits)
        if np.any(limits <= EPS):
            raise RuntimeError("deadline_limits must be positive for normalized partial makespans.")
        partials = _assert_finite_array("partial_makespans", self.partial_makespans)
        return (partials / limits).astype(np.float32)

    def _compute_returns_by_workflow(self, signals, values, gamma, gae_lambda):
        signals = _assert_finite_array("workflow_signals", signals)
        values = _assert_finite_array("workflow_values", values)
        advantages = np.zeros_like(signals, dtype=np.float32)
        for _, indices in self.workflow_groups():
            if not any(self.workflow_dones[index] for index in indices):
                raise RuntimeError("Workflow-level P3O requires complete workflow rollouts.")
            gae = 0.0
            next_value = 0.0
            for pos in range(len(indices) - 1, -1, -1):
                index = indices[pos]
                non_terminal = 0.0 if self.workflow_dones[index] or pos == len(indices) - 1 else 1.0
                delta = signals[index] + float(gamma) * next_value * non_terminal - values[index]
                gae = delta + float(gamma) * float(gae_lambda) * non_terminal * gae
                advantages[index] = gae
                next_value = values[index]
        returns = advantages + values
        return returns.astype(np.float32), advantages.astype(np.float32)

    def validate(self):
        lengths = [
            len(self.primary_actions), len(self.backup_actions), len(self.old_log_probs),
            len(self.reward_values), len(self.deadline_values), len(self.reliability_values),
            len(self.rewards), len(self.deadline_costs), len(self.reliability_costs),
            len(self.deadline_limits), len(self.reliability_limits), len(self.partial_makespans),
            len(self.partial_log_risks), len(self.workflow_ids), len(self.workflow_dones),
            len(self.episode_dones), len(self.primary_masks), len(self.backup_pair_masks), len(self.infos),
        ]
        if any(length != self.size for length in lengths):
            raise RuntimeError("High rollout buffer column lengths are inconsistent.")
        for name in ("rewards", "deadline_costs", "reliability_costs", "partial_makespans", "partial_log_risks"):
            _assert_finite_array(name, getattr(self, name))
        if np.any(np.asarray(self.deadline_costs) < -EPS):
            raise RuntimeError("deadline_costs must be non-negative.")
        if np.any(np.asarray(self.reliability_costs) < -EPS):
            raise RuntimeError("reliability_costs must be non-negative.")
        for workflow_id, indices in self.workflow_groups():
            if not any(self.workflow_dones[index] for index in indices):
                raise RuntimeError(f"Workflow {workflow_id} is incomplete in high rollout buffer.")
            for index in indices:
                info_workflow_id = str(self.infos[index].get("workflow_id", workflow_id))
                if info_workflow_id != workflow_id:
                    raise RuntimeError(
                        "High rollout workflow_id mismatch: "
                        f"group={workflow_id}, info={info_workflow_id}, index={index}"
                    )
            workflow_deadlines = [float(self.deadline_limits[index]) for index in indices]
            reference_deadline = workflow_deadlines[0]
            if reference_deadline <= 0.0:
                raise RuntimeError(f"Deadline limit must be positive for workflow {workflow_id}.")
            for index, deadline_value in zip(indices, workflow_deadlines):
                if abs(deadline_value - reference_deadline) > 1e-6:
                    raise RuntimeError(
                        "Deadline limit changed within one workflow: "
                        f"workflow={workflow_id}, "
                        f"reference={reference_deadline}, "
                        f"value={deadline_value}, "
                        f"index={index}"
                    )
            previous_partial = -EPS
            for index in indices:
                current_partial = float(self.partial_makespans[index])
                if current_partial + 1e-6 < previous_partial:
                    raise RuntimeError(
                        "Partial makespan decreased within workflow: "
                        f"workflow={workflow_id}, "
                        f"previous={previous_partial}, "
                        f"current={current_partial}, "
                        f"index={index}"
                    )
                previous_partial = current_partial
            deadline_cost_sum = sum(float(self.deadline_costs[index]) for index in indices)
            final_partial_makespan = float(self.partial_makespans[indices[-1]])
            if abs(deadline_cost_sum - final_partial_makespan) > 1e-5:
                raise RuntimeError(
                    "Deadline cost telescope mismatch: "
                    f"workflow={workflow_id}, "
                    f"cost_sum={deadline_cost_sum}, "
                    f"final_partial={final_partial_makespan}"
                )
        return True


class LowRolloutBuffer:
    def __init__(self):
        self.clear()

    @property
    def size(self):
        return len(self.states)

    def clear(self):
        self.states = []
        self.actions_u = []
        self.old_log_probs = []
        self.log_probs = self.old_log_probs
        self.values = []
        self.deadline_values = []
        self.rewards = []
        self.deadline_costs = []
        self.deadline_limits = []
        self.server_ids = []
        self.infos = []
        self.returns = []
        self.advantages = []
        self.deadline_returns = []
        self.deadline_advantages = []

    def add(
        self,
        *,
        state,
        action_u,
        old_log_prob,
        value,
        reward,
        server_id,
        deadline_value=0.0,
        deadline_cost=0.0,
        deadline_limit=1.0,
        info=None,
    ):
        deadline_limit = max(float(deadline_limit), EPS)
        normalized_deadline_cost = max(0.0, float(deadline_cost)) / deadline_limit
        numeric_values = {
            "action_u": float(action_u),
            "old_log_prob": float(old_log_prob),
            "value": float(value),
            "reward": float(reward),
            "deadline_value": float(deadline_value),
            "deadline_cost": normalized_deadline_cost,
            "deadline_limit": deadline_limit,
        }
        for key, numeric_value in numeric_values.items():
            if not math.isfinite(numeric_value):
                raise FloatingPointError(f"Low rollout field {key} is not finite: {numeric_value}")
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions_u.append(float(action_u))
        self.old_log_probs.append(float(old_log_prob))
        self.values.append(float(value))
        self.deadline_values.append(float(deadline_value))
        self.rewards.append(float(reward))
        self.deadline_costs.append(normalized_deadline_cost)
        self.deadline_limits.append(deadline_limit)
        self.server_ids.append(int(server_id))
        info_row = dict(info or {})
        info_row["low_deadline_cost"] = normalized_deadline_cost
        info_row["low_deadline_limit"] = deadline_limit
        self.infos.append(info_row)

    def compute_returns_and_advantages(self):
        rewards = _assert_finite_array("low_rewards", self.rewards)
        values = _assert_finite_array("low_values", self.values)
        self.returns = rewards.astype(np.float32)
        self.advantages = (rewards - values).astype(np.float32)
        return self.returns, self.advantages

    def compute_deadline_returns_and_advantages(self):
        costs = _assert_finite_array("low_deadline_costs", self.deadline_costs)
        values = _assert_finite_array("low_deadline_values", self.deadline_values)
        self.deadline_returns = costs.astype(np.float32)
        self.deadline_advantages = (costs - values).astype(np.float32)
        return self.deadline_returns, self.deadline_advantages

    def validate(self):
        lengths = [
            len(self.actions_u),
            len(self.old_log_probs),
            len(self.values),
            len(self.deadline_values),
            len(self.rewards),
            len(self.deadline_costs),
            len(self.deadline_limits),
            len(self.server_ids),
            len(self.infos),
        ]
        if any(length != self.size for length in lengths):
            raise RuntimeError("Low rollout buffer column lengths are inconsistent.")
        _assert_finite_array("low_rewards", self.rewards)
        deadline_costs = _assert_finite_array("low_deadline_costs", self.deadline_costs)
        if np.any(deadline_costs < -EPS):
            raise RuntimeError("low deadline costs must be non-negative.")
        if np.any(_assert_finite_array("low_deadline_limits", self.deadline_limits) <= EPS):
            raise RuntimeError("low deadline limits must be positive.")
        return True
