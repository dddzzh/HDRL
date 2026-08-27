from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.shared.hierarchical_state_reward import build_shared_state_encoder


class DReaMStateBuilder(ABC):
    @property
    @abstractmethod
    def state_dim(self): ...

    @abstractmethod
    def build_state(self, env, task, workflow, context=None): ...


class SharedEdgeCloudStateBuilder(DReaMStateBuilder):
    """Adapter to the project's comparison-state encoder; no DReaM state semantics."""

    def __init__(self, config=None):
        self.encoder = build_shared_state_encoder(config)

    @property
    def state_dim(self):
        return int(self.encoder.high_state_dim)

    def build_state(self, env, task, workflow, context=None):
        del workflow
        if task is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        context = context or env.build_hierarchical_decision_context(
            task["workflow_id"], task["task_id"]
        )
        env_state = env.get_state()
        env_state["decision_context"] = context
        state = np.asarray(self.encoder.encode_high_state(env_state).vector, dtype=np.float32)
        if state.shape != (self.state_dim,):
            raise ValueError(f"DReaM state builder returned {state.shape}, expected {(self.state_dim,)}.")
        return state
