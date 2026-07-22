from __future__ import annotations

from typing import Any, Mapping, Optional

from src.a2c.a2c import StepContext, _require_torch
from src.ppo.ppo import PPO
from src.recr.state_reward import RECRStateReward


class RECRPPO(PPO):
    """PPO + RECR 论文启发的 state / reward。"""

    def __init__(self, config=None, seed: int = 42):
        _require_torch()
        config = dict(config or {})
        self.recr = RECRStateReward(
            state_cfg=config.get("state", config),
            reward_cfg=config.get("reward", {}),
        )
        agent_config = dict(config)
        agent_config.setdefault("state", {})
        agent_config["state"] = {
            **agent_config["state"],
            "max_ready_tasks": self.recr.state_cfg.max_ready_tasks,
            "max_servers": self.recr.state_cfg.max_servers,
        }
        super().__init__(config=agent_config, seed=seed)

    def encode_state(self, env_state: Mapping[str, Any]):
        return self.recr.encode_state(env_state)

    def reset_reward(self) -> None:
        self.recr.reset_reward()

    def compute_reward(self, context: StepContext) -> float:
        return self.recr.compute_reward(context)
