from __future__ import annotations

from pathlib import Path

from src.ppo.dual_agent import HierarchicalPPOAgent
from src.schedulers.base_scheduler import BaseScheduler


class DualPPOScheduler(BaseScheduler):
    """Scheduler wrapper for hierarchical PPO inference/evaluation.

    Training is handled by tools/train_hierarchical_ppo.py because high/low PPO
    require separate rollout buffers and updates.
    """

    def __init__(self, params=None, seed=42):
        super().__init__(params=params, seed=seed)
        self.training = bool(self.params.get("training", False))
        self.deterministic = bool(self.params.get("deterministic", not self.training))
        self.ppo = HierarchicalPPOAgent(config=dict(self.params.get("agent", {})), seed=seed)

    def reset(self):
        pass

    def select_action(self, state):
        if not state["ready_tasks"]:
            raise ValueError("Hierarchical PPO scheduler received empty ready task queue")
        _, _, env_action = self.ppo.select_action(state, deterministic=self.deterministic)
        return env_action

    def update(self) -> dict[str, float]:
        return {}

    def save(self, path: str) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        self.ppo.save(str(base), extra={"scheduler": "dual_ppo", "params": self.params})
