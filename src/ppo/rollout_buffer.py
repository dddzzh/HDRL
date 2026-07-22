from __future__ import annotations

import numpy as np


class PPORolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.deadline_values = []
        self.reliability_values = []
        self.rewards = []
        self.raw_task_energies = []
        self.deadline_costs = []
        self.reliability_costs = []
        self.deadline_limits = []
        self.reliability_limits = []
        self.partial_makespans = []
        self.partial_log_risks = []
        self.deadline_returns = []
        self.reliability_returns = []
        self.deadline_advantages = []
        self.reliability_advantages = []
        self.dones = []
        self.next_states = []
        self.masks = {}
        self.infos = []
        self.workflow_ids = []
        self.workflow_dones = []
        self.actor_update_masks = []
        self.completed_workflow_ids = set()
        self._workflow_deadline_limits = {}
        self._workflow_reliability_limits = {}

    @property
    def size(self):
        return len(self.states)

    def _format_action_item(self, item):
        if isinstance(item, (int, np.integer)):
            return int(item)
        return float(item)

    def add(
        self,
        state,
        action,
        log_prob,
        value,
        reward,
        done,
        next_state=None,
        masks=None,
        info=None,
        deadline_cost=0.0,
        reliability_cost=0.0,
        deadline_value=0.0,
        reliability_value=0.0,
        raw_task_energy=None,
        workflow_id=None,
        deadline_limit=None,
        reliability_limit=None,
        current_partial_makespan=None,
        current_partial_log_risk=None,
        workflow_done=False,
        actor_update_mask=True,
    ):
        self.states.append(np.asarray(state, dtype=np.float32))
        if isinstance(action, (tuple, list, np.ndarray)):
            self.actions.append([self._format_action_item(item) for item in action])
        else:
            self.actions.append([self._format_action_item(action)])
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.raw_task_energies.append(
            0.0 if raw_task_energy is None else float(raw_task_energy)
        )
        row_info = dict(info or {})
        if workflow_id is None:
            workflow_id = row_info.get("workflow_id")
        workflow_key = None if workflow_id is None else str(workflow_id)
        deadline_limit = row_info.get("deadline_limit", deadline_limit)
        reliability_limit = row_info.get("reliability_limit", reliability_limit)
        current_partial_makespan = row_info.get("current_partial_makespan", current_partial_makespan)
        current_partial_log_risk = row_info.get("current_partial_log_risk", current_partial_log_risk)

        deadline_limit_value = None if deadline_limit is None else float(deadline_limit)
        reliability_limit_value = None if reliability_limit is None else float(reliability_limit)
        if workflow_key is not None:
            self._validate_workflow_limit(
                self._workflow_deadline_limits,
                workflow_key,
                deadline_limit_value,
                "deadline_limit",
            )
            self._validate_workflow_limit(
                self._workflow_reliability_limits,
                workflow_key,
                reliability_limit_value,
                "reliability_limit",
            )

        self.deadline_costs.append(float(deadline_cost))
        self.reliability_costs.append(float(reliability_cost))
        self.deadline_limits.append(deadline_limit_value)
        self.reliability_limits.append(reliability_limit_value)
        self.partial_makespans.append(
            None if current_partial_makespan is None else float(current_partial_makespan)
        )
        self.partial_log_risks.append(
            None if current_partial_log_risk is None else float(current_partial_log_risk)
        )
        self.deadline_values.append(float(deadline_value))
        self.reliability_values.append(float(reliability_value))
        self.dones.append(bool(done))
        self.next_states.append(None if next_state is None else np.asarray(next_state, dtype=np.float32))
        for name, mask in (masks or {}).items():
            self.masks.setdefault(name, []).append(np.asarray(mask, dtype=bool))
        row_info["workflow_id"] = workflow_id
        row_info["deadline_cost"] = float(deadline_cost)
        row_info["reliability_cost"] = float(reliability_cost)
        row_info["deadline_limit"] = deadline_limit_value
        row_info["reliability_limit"] = reliability_limit_value
        row_info["current_partial_makespan"] = (
            None if current_partial_makespan is None else float(current_partial_makespan)
        )
        row_info["current_partial_log_risk"] = (
            None if current_partial_log_risk is None else float(current_partial_log_risk)
        )
        row_info["workflow_done"] = bool(workflow_done)
        row_info["actor_update_mask"] = bool(row_info.get("actor_update_mask", actor_update_mask))
        self.workflow_ids.append(workflow_key)
        self.workflow_dones.append(bool(workflow_done))
        self.actor_update_masks.append(bool(row_info["actor_update_mask"]))
        self.infos.append(row_info)

    @staticmethod
    def _validate_workflow_limit(limit_map, workflow_id, value, field_name, eps=1e-8):
        if value is None:
            return
        if workflow_id not in limit_map:
            limit_map[workflow_id] = float(value)
            return
        if abs(float(limit_map[workflow_id]) - float(value)) > eps:
            raise RuntimeError(
                f"Inconsistent {field_name} for workflow_id={workflow_id}: "
                f"previous={limit_map[workflow_id]}, current={value}"
            )

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

    def add_to_last_workflow_transition(
        self,
        workflow_id,
        reward_delta=0.0,
        deadline_cost_delta=0.0,
        partial_makespan=None,
        info_updates=None,
    ):
        workflow_id = str(workflow_id)
        for index in range(self.size - 1, -1, -1):
            if self.workflow_ids[index] == workflow_id:
                reward_delta = float(reward_delta)
                deadline_cost_delta = float(deadline_cost_delta)
                self.rewards[index] = float(self.rewards[index]) + reward_delta
                self.deadline_costs[index] = float(self.deadline_costs[index]) + deadline_cost_delta
                self.infos[index]["deadline_cost"] = self.deadline_costs[index]
                if partial_makespan is not None:
                    self.partial_makespans[index] = float(partial_makespan)
                    self.infos[index]["current_partial_makespan"] = float(partial_makespan)
                if info_updates:
                    self.infos[index].update(info_updates)
                return True
        return False

    def add_reward_to_indices(self, indices, reward_deltas, info_key=None):
        for index, delta in zip(indices, reward_deltas):
            delta = float(delta)
            self.rewards[index] = float(self.rewards[index]) + delta
            if info_key is not None:
                self.infos[index][info_key] = float(self.infos[index].get(info_key, 0.0)) + delta
            self.infos[index]["low_total_reward"] = float(self.rewards[index])

    def compute_returns_and_advantages(self, gamma, gae_lambda, last_value=0.0):
        return self.compute_stream_returns_and_advantages(
            self.rewards,
            self.values,
            gamma,
            gae_lambda,
            last_value=last_value,
        )

    def compute_stream_returns_and_advantages(self, rewards, values, gamma, gae_lambda, last_value=0.0):
        # P3O uses the same GAE recurrence for reward and each cost stream.
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)

        for index in range(len(rewards) - 1, -1, -1):
            non_terminal = 1.0 - dones[index]
            delta = rewards[index] + gamma * next_value * non_terminal - values[index]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[index] = gae
            next_value = values[index]

        returns = advantages + values
        return returns.astype(np.float32), advantages.astype(np.float32)

    def compute_grouped_stream_returns_and_advantages(
        self,
        rewards,
        values,
        gamma,
        gae_lambda,
        group_key,
    ):
        # Server-level low agents share one buffer, but their temporal credit
        # assignment follows each server's own task sequence.
        rewards = np.asarray(rewards, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)

        groups = {}
        order = []
        for index, info in enumerate(self.infos):
            key = info.get(group_key)
            if key is None:
                key = "__missing__"
            key = str(key)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(index)

        for key in order:
            indices = groups[key]
            gae = 0.0
            next_value = 0.0
            for pos in range(len(indices) - 1, -1, -1):
                index = indices[pos]
                is_group_tail = pos == len(indices) - 1
                non_terminal = 0.0 if is_group_tail or dones[index] else 1.0
                delta = rewards[index] + gamma * next_value * non_terminal - values[index]
                gae = delta + gamma * gae_lambda * non_terminal * gae
                advantages[index] = gae
                next_value = values[index]

        returns = advantages + values
        return returns.astype(np.float32), advantages.astype(np.float32)

    def compute_workflow_cost_returns_and_advantages(
        self,
        costs,
        values,
        gamma,
        gae_lambda,
        bootstrap_values=None,
    ):
        costs = np.asarray(costs, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        advantages = np.zeros_like(costs, dtype=np.float32)
        if len(costs) == 0:
            return advantages, advantages

        groups = {}
        order = []
        for index, workflow_id in enumerate(self.workflow_ids):
            key = "__missing__" if workflow_id is None else str(workflow_id)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(index)

        for key in order:
            indices = groups[key]
            gae = 0.0
            next_value = 0.0
            for pos in range(len(indices) - 1, -1, -1):
                index = indices[pos]
                is_tail = pos == len(indices) - 1
                if self.workflow_dones[index] or self.dones[index]:
                    non_terminal = 0.0
                    next_value = 0.0
                elif is_tail:
                    non_terminal = 1.0
                    next_value = (
                        0.0
                        if bootstrap_values is None
                        else float(bootstrap_values[index])
                    )
                else:
                    non_terminal = 1.0
                delta = costs[index] + gamma * next_value * non_terminal - values[index]
                gae = delta + gamma * gae_lambda * non_terminal * gae
                advantages[index] = gae
                next_value = values[index]

        returns = advantages + values
        return returns.astype(np.float32), advantages.astype(np.float32)

    def workflow_groups(self):
        groups = {}
        order = []
        for index, workflow_id in enumerate(self.workflow_ids):
            key = "__missing__" if workflow_id is None else str(workflow_id)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(index)
        return [(key, groups[key]) for key in order]

    def workflow_minibatches(self, batch_size, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        groups = self.workflow_groups()
        if not groups:
            return []
        order = np.arange(len(groups))
        rng.shuffle(order)
        minibatches = []
        current = []
        current_size = 0
        for group_pos in order:
            _, indices = groups[int(group_pos)]
            group_size = len(indices)
            if current and current_size + group_size > int(batch_size):
                minibatches.append(np.asarray(current, dtype=np.int64))
                current = []
                current_size = 0
            if group_size > int(batch_size):
                minibatches.append(np.asarray(indices, dtype=np.int64))
                continue
            current.extend(indices)
            current_size += group_size
        if current:
            minibatches.append(np.asarray(current, dtype=np.int64))
        return minibatches

    def server_balanced_minibatches(self, batch_size, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        if self.size == 0:
            return []
        groups = {}
        for index, info in enumerate(self.infos):
            groups.setdefault(str(info.get("server_id", "__missing__")), []).append(index)
        for values in groups.values():
            rng.shuffle(values)
        keys = list(groups.keys())
        rng.shuffle(keys)
        batches = []
        current = []
        exhausted = False
        while not exhausted:
            exhausted = True
            for key in keys:
                if groups[key]:
                    current.append(groups[key].pop())
                    exhausted = False
                    if len(current) >= int(batch_size):
                        batches.append(np.asarray(current, dtype=np.int64))
                        current = []
        if current:
            batches.append(np.asarray(current, dtype=np.int64))
        return batches

    def validate_p3o_metadata(self):
        required = {
            "deadline_limits": self.deadline_limits,
            "reliability_limits": self.reliability_limits,
            "partial_makespans": self.partial_makespans,
            "partial_log_risks": self.partial_log_risks,
        }
        for name, values in required.items():
            if len(values) != self.size:
                raise RuntimeError(f"{name} length does not match buffer size.")
            missing = [
                index
                for index, value in enumerate(values)
                if value is None or not np.isfinite(float(value))
            ]
            if missing:
                raise RuntimeError(f"P3O requires finite {name}; missing indices={missing[:10]}")
        missing_workflow = [index for index, value in enumerate(self.workflow_ids) if value is None]
        if missing_workflow:
            raise RuntimeError(f"P3O requires workflow_id for every transition; missing indices={missing_workflow[:10]}")

    def discounted_workflow_cost_mean(self, costs, gamma):
        costs = np.asarray(costs, dtype=np.float32)
        workflow_returns = []
        seen = set()
        for workflow_id in self.workflow_ids:
            if workflow_id is not None:
                seen.add(str(workflow_id))
        for workflow_id in seen:
            indices = [index for index, value in enumerate(self.workflow_ids) if value == workflow_id]
            terminal_positions = [
                pos for pos, index in enumerate(indices) if self.workflow_dones[index]
            ]
            if not terminal_positions:
                continue
            last_pos = terminal_positions[-1]
            total = 0.0
            discount = 1.0
            for index in indices[: last_pos + 1]:
                total += discount * float(costs[index])
                discount *= float(gamma)
            workflow_returns.append(total)
        if not workflow_returns:
            return 0.0, 0
        return float(np.mean(workflow_returns)), len(workflow_returns)
