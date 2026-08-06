from __future__ import annotations

import numpy as np


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
        self.next_states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=bool)
        self.action_masks = np.zeros((self.max_size, self.action_dim), dtype=bool)
        self.next_action_masks = np.zeros((self.max_size, self.action_dim), dtype=bool)

    @property
    def size(self):
        return min(self.mem_count, self.max_size)

    def store_transition(self, state, action, reward, state_, done, action_mask, next_action_mask):
        mem_idx = self.mem_count % self.max_size
        self.states[mem_idx] = np.asarray(state, dtype=np.float32)
        self.actions[mem_idx] = int(action)
        self.rewards[mem_idx] = float(reward)
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
