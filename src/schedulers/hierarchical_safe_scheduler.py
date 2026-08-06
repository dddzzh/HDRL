from __future__ import annotations

from pathlib import Path

from src.ppo.hierarchical_agent import HierarchicalSafeAgent
from src.schedulers.base_scheduler import BaseScheduler


class HierarchicalSafeScheduler(BaseScheduler):
    """分层安全 PPO 推理入口；安全信息必须来自 state['decision_context']。"""

    def __init__(self, params=None, seed=42):
        super().__init__(params=params, seed=seed)
        self.training = bool(self.params.get("training", False))
        self.deterministic = bool(self.params.get("deterministic", not self.training))
        self.agent = HierarchicalSafeAgent(config=dict(self.params.get("agent", {})), seed=seed)
        checkpoint = self.params.get("checkpoint")
        if checkpoint:
            self.agent.load_checkpoint(checkpoint)

    def reset(self):
        pass

    def select_action(self, state):
        if not state.get("ready_tasks"):
            raise ValueError("Hierarchical safe scheduler received empty ready task queue.")
        if "decision_context" not in state:
            raise KeyError("state must contain decision_context before scheduler inference.")
        _, _, env_action = self.agent.select_action(
            state,
            deterministic=self.deterministic,
            low_deterministic=self.deterministic,
        )
        return env_action

    def update(self):
        return {}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save_checkpoint(path, training_phase="scheduler", config=self.params.get("agent", {}))
