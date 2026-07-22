from __future__ import annotations

from typing import Any, Mapping, Optional

from src.a2c.a2c import ActionDecision, StepContext
from src.recr.recr_ppo import RECRPPO
from src.schedulers.base_scheduler import BaseScheduler


class RECRPPOScheduler(BaseScheduler):
    def __init__(self, params=None, seed=42):
        super().__init__(params=params, seed=seed)
        self.training = bool(self.params.get("training", False))
        self.deterministic = bool(self.params.get("deterministic", not self.training))

        agent_config = dict(self.params.get("agent", {}))
        if "state" not in agent_config:
            agent_config["state"] = self.params.get("state", {})
        if "reward" not in agent_config:
            agent_config["reward"] = self.params.get("reward", {})

        self.ppo = RECRPPO(config=agent_config, seed=seed)
        self._transitions = []
        self._last_decision: Optional[ActionDecision] = None

    def reset(self):
        self.ppo.reset_reward()
        self._transitions = []
        self._last_decision = None

    def select_action(self, state):
        if not state["ready_tasks"]:
            raise ValueError("RECR PPO scheduler received empty ready task queue")
        decision = self.ppo.select_action(state, deterministic=self.deterministic)
        self._last_decision = decision
        return decision.action

    def on_step(
        self,
        prev_state: Mapping[str, Any],
        action: Mapping[str, Any],
        next_state: Mapping[str, Any],
        env_reward: float,
        done: bool,
        info: Mapping[str, Any],
    ) -> float:
        if not self.training or self._last_decision is None:
            return float(env_reward)

        reward = self.ppo.compute_reward(
            StepContext(
                prev_state=prev_state,
                action=action,
                next_state=next_state,
                env_reward=env_reward,
                done=done,
                info=info,
            )
        )
        self._transitions.append(self.ppo.build_transition(self._last_decision, reward, done))
        self._last_decision = None
        return reward

    def update(self) -> dict[str, float]:
        metrics = self.ppo.update(self._transitions)
        self._transitions = []
        return metrics

    def save(self, path: str) -> None:
        self.ppo.save(path, extra={"scheduler": "recr_ppo", "params": self.params})
