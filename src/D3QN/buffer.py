from __future__ import annotations

import numpy as np


REWARD_COMPONENT_NAMES = ("workflow_slack_ratio", "energy_saving")


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=100000, batch_size=256):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        self.mem_count = 0
        self.states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.max_size, dtype=np.int64)
        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.base_energy_rewards = np.zeros(self.max_size, dtype=np.float32)
        self.deadline_constraint_costs = np.zeros(self.max_size, dtype=np.float32)
        self.reward_components = np.zeros(
            (self.max_size, len(REWARD_COMPONENT_NAMES)), dtype=np.float32
        )
        self.deadline_crossed = np.zeros(self.max_size, dtype=np.float32)
        self.deadline_violation = np.zeros(self.max_size, dtype=np.float32)
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
        state_,
        done,
        action_mask,
        next_action_mask,
        *,
        reward_components=None,
        deadline_crossed=0.0,
        deadline_violation=0.0,
        base_energy_reward=None,
        deadline_constraint_cost=0.0,
    ):
        mem_idx = self.mem_count % self.max_size
        self.states[mem_idx] = np.asarray(state, dtype=np.float32)
        self.actions[mem_idx] = int(action)
        self.rewards[mem_idx] = float(reward)
        if base_energy_reward is None:
            base_energy_reward = float(reward)
        base_energy_reward = float(base_energy_reward)
        deadline_constraint_cost = float(deadline_constraint_cost)
        if (
            not np.isfinite(base_energy_reward)
            or not np.isfinite(deadline_constraint_cost)
            or deadline_constraint_cost < 0.0
        ):
            raise FloatingPointError(
                "D3QN Lagrangian reward components contain invalid values."
            )
        self.base_energy_rewards[mem_idx] = base_energy_reward
        self.deadline_constraint_costs[mem_idx] = deadline_constraint_cost
        if reward_components is None:
            reward_components = (float(reward), 0.0)
        components = np.asarray(reward_components, dtype=np.float32)
        if components.shape != (len(REWARD_COMPONENT_NAMES),):
            raise ValueError(
                "D3QN reward_components must contain workflow slack and "
                "energy saving."
            )
        crossed = float(deadline_crossed)
        violation = float(deadline_violation)
        if (
            not bool(np.isfinite(components).all())
            or not np.isfinite(crossed)
            or not np.isfinite(violation)
        ):
            raise FloatingPointError("D3QN reward components contain NaN or Inf.")
        self.reward_components[mem_idx] = components
        self.deadline_crossed[mem_idx] = crossed
        self.deadline_violation[mem_idx] = violation
        self.next_states[mem_idx] = np.asarray(state_, dtype=np.float32)
        self.dones[mem_idx] = bool(done)
        self.action_masks[mem_idx] = np.asarray(action_mask, dtype=bool)
        self.next_action_masks[mem_idx] = np.asarray(next_action_mask, dtype=bool)
        if not bool(self.action_masks[mem_idx, self.actions[mem_idx]]):
            raise ValueError("Stored D3QN action violates its action mask.")
        self.mem_count += 1

    def ready(self):
        return self.size >= self.batch_size

    def sample_buffer(self, rng):
        if not self.ready():
            return None
        batch = rng.choice(self.size, self.batch_size, replace=False)
        return (
            self.states[batch],
            self.actions[batch],
            self.rewards[batch],
            self.base_energy_rewards[batch],
            self.deadline_constraint_costs[batch],
            self.reward_components[batch],
            self.deadline_crossed[batch],
            self.deadline_violation[batch],
            self.next_states[batch],
            self.dones[batch],
            self.action_masks[batch],
            self.next_action_masks[batch],
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
            "base_energy_rewards": self.base_energy_rewards,
            "deadline_constraint_costs": self.deadline_constraint_costs,
            "reward_component_names": REWARD_COMPONENT_NAMES,
            "reward_components": self.reward_components,
            "deadline_crossed": self.deadline_crossed,
            "deadline_violation": self.deadline_violation,
            "next_states": self.next_states,
            "dones": self.dones,
            "action_masks": self.action_masks,
            "next_action_masks": self.next_action_masks,
        }

    def load_state_dict(self, payload):
        if "reward_components" in payload:
            saved_names = tuple(payload.get("reward_component_names", ()))
            if saved_names != REWARD_COMPONENT_NAMES:
                raise ValueError(
                    "D3QN replay reward components belong to an older reward formula."
                )
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
        self.base_energy_rewards[:] = payload.get(
            "base_energy_rewards", self.rewards
        )
        self.deadline_constraint_costs[:] = payload.get(
            "deadline_constraint_costs",
            np.zeros_like(self.deadline_constraint_costs),
        )
        if "reward_components" in payload:
            self.reward_components[:] = payload["reward_components"]
            self.deadline_crossed[:] = payload.get(
                "deadline_crossed", np.zeros_like(self.deadline_crossed)
            )
            self.deadline_violation[:] = payload.get(
                "deadline_violation", np.zeros_like(self.deadline_violation)
            )
        else:
            # Old scalar-reward replay is loadable for diagnostics only.
            self.reward_components.fill(0.0)
            self.reward_components[:, 0] = self.rewards
            self.deadline_crossed.fill(0.0)
            self.deadline_violation.fill(0.0)

    def reward_statistics(self, std_floor=0.05):
        if self.size <= 0:
            return None
        values = self.reward_components[: self.size]
        return {
            "mean": values.mean(axis=0, dtype=np.float64).astype(np.float32),
            "std": np.maximum(
                values.std(axis=0, dtype=np.float64), float(std_floor)
            ).astype(np.float32),
        }

    def base_energy_reward_statistics(self, std_floor=0.05):
        """Statistics for the raw upper placement-energy signal."""
        if self.size <= 0:
            return None
        values = self.base_energy_rewards[: self.size].astype(np.float64)
        return {
            "count": int(values.size),
            "mean": float(values.mean()),
            "std": float(max(values.std(), float(std_floor))),
        }
