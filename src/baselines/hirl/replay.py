from __future__ import annotations

import numpy as np


class FailureAwareReplay:
    """Prioritized replay using |TD| * (1 + gamma_fail * I_fail), Eq. (14)."""

    def __init__(self, capacity, state_dim, action_shape, seed=42):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_shape = tuple(action_shape)
        self.rng = np.random.default_rng(seed)
        self.position = self.size = 0
        self.states = np.zeros((self.capacity, self.state_dim), np.float32)
        self.actions = np.zeros((self.capacity, *self.action_shape), np.float32)
        self.rewards = np.zeros(self.capacity, np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), np.float32)
        self.dones = np.zeros(self.capacity, np.float32)
        self.failures = np.zeros(self.capacity, np.float32)
        self.priorities = np.ones(self.capacity, np.float32)

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, done, failure):
        state = np.asarray(state, np.float32)
        next_state = np.asarray(next_state, np.float32)
        action = np.asarray(action, np.float32).reshape(self.action_shape)
        if state.shape != (self.state_dim,) or next_state.shape != (self.state_dim,):
            raise ValueError("HIRL replay state shape mismatch.")
        if not np.isfinite(state).all() or not np.isfinite(next_state).all():
            raise FloatingPointError("HIRL replay received a non-finite state.")
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = float(reward)
        self.next_states[index] = next_state
        self.dones[index] = float(done)
        self.failures[index] = float(bool(failure))
        previous_max = float(self.priorities[: self.size].max()) if self.size else 1.0
        self.priorities[index] = max(previous_max, 1.0)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, alpha, beta):
        count = int(batch_size)
        scaled = np.maximum(self.priorities[: self.size], 1e-6) ** float(alpha)
        probabilities = scaled / scaled.sum()
        indices = self.rng.choice(self.size, count, replace=False, p=probabilities)
        weights = (self.size * probabilities[indices]) ** (-float(beta))
        weights /= max(float(weights.max()), 1e-8)
        values = (
            self.states[indices], self.actions[indices], self.rewards[indices],
            self.next_states[indices], self.dones[indices], self.failures[indices],
        )
        return indices, weights.astype(np.float32), values

    def update_priorities(self, indices, td_errors, failure_priority):
        td_errors = np.asarray(td_errors, np.float32).reshape(-1)
        failures = self.failures[np.asarray(indices)]
        self.priorities[np.asarray(indices)] = np.maximum(
            np.abs(td_errors) * (1.0 + float(failure_priority) * failures), 1e-6
        )

    def state_dict(self):
        return {
            "capacity": self.capacity, "state_dim": self.state_dim,
            "action_shape": self.action_shape, "position": self.position,
            "size": self.size, "states": self.states, "actions": self.actions,
            "rewards": self.rewards, "next_states": self.next_states,
            "dones": self.dones, "failures": self.failures,
            "priorities": self.priorities,
        }

    def load_state_dict(self, payload):
        if int(payload["capacity"]) != self.capacity or int(payload["state_dim"]) != self.state_dim:
            raise ValueError("HIRL replay checkpoint shape mismatch.")
        for name in (
            "states", "actions", "rewards", "next_states", "dones",
            "failures", "priorities",
        ):
            getattr(self, name)[:] = payload[name]
        self.position = int(payload["position"])
        self.size = int(payload["size"])
