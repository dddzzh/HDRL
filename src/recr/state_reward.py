from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import numpy as np

from src.a2c.a2c import Observation, StepContext
from src.simulator.resource import compute_energy, compute_execution_time

# RECR-inspired feature sizes (deadline / energy / reliability aware).
RECR_TASK_FEATURE_DIM = 8
RECR_SERVER_FEATURE_DIM = 5
RECR_GLOBAL_FEATURE_DIM = 5


class RECRRewardConfig:
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.2,
        gamma: float = 0.1,
        budget_coef: float = 0.5,
        energy_scale: float = 100.0,
        reliability_target: float = 0.9,
        terminal_deadline_bonus: float = 1.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.budget_coef = budget_coef
        self.energy_scale = energy_scale
        self.reliability_target = reliability_target
        self.terminal_deadline_bonus = terminal_deadline_bonus


class RECRStateConfig:
    def __init__(
        self,
        max_ready_tasks: int = 32,
        max_servers: int = 16,
        time_horizon: float = 3600.0,
        ref_freq_hz: float = 3.0e9,
    ):
        self.max_ready_tasks = max_ready_tasks
        self.max_servers = max_servers
        self.time_horizon = time_horizon
        # reference frequency in Hz (cycles/s) used to normalize cycle-based features.
        self.ref_freq_hz = ref_freq_hz


class RECRStateReward:
    """RECR 论文约束启发式 state / reward（用于 RL 实验）。"""

    def __init__(
        self,
        state_cfg: Optional[Mapping[str, Any]] = None,
        reward_cfg: Optional[Mapping[str, Any]] = None,
    ):
        state_cfg = state_cfg or {}
        reward_cfg = reward_cfg or {}
        self.state_cfg = RECRStateConfig(
            max_ready_tasks=int(state_cfg.get("max_ready_tasks", 32)),
            max_servers=int(state_cfg.get("max_servers", 16)),
            time_horizon=float(state_cfg.get("time_horizon", 3600.0)),
            ref_freq_hz=float(state_cfg.get("ref_freq_hz", 3.0e9)),
        )
        self.reward_cfg = RECRRewardConfig(
            alpha=float(reward_cfg.get("alpha", 1.0)),
            beta=float(reward_cfg.get("beta", 0.2)),
            gamma=float(reward_cfg.get("gamma", 0.1)),
            budget_coef=float(reward_cfg.get("budget_coef", 0.5)),
            energy_scale=float(reward_cfg.get("energy_scale", 100.0)),
            reliability_target=float(reward_cfg.get("reliability_target", 0.9)),
            terminal_deadline_bonus=float(reward_cfg.get("terminal_deadline_bonus", 1.0)),
        )
        self._workflow_energy_used: dict[str, float] = {}

    def reset_reward(self) -> None:
        self._workflow_energy_used = {}

    def encode_state(self, env_state: Mapping[str, Any]) -> Observation:
        cfg = self.state_cfg
        ready_tasks = tuple(env_state.get("ready_tasks", []))
        servers = tuple(env_state.get("servers", []))
        rho_levels = tuple(float(rho) for rho in env_state.get("rho_levels", [1.0]))
        current_time = float(env_state.get("time", 0.0))
        horizon = max(cfg.time_horizon, 1.0)

        ready_task_features = np.zeros((cfg.max_ready_tasks, RECR_TASK_FEATURE_DIM), dtype=np.float32)
        ready_task_mask = np.zeros(cfg.max_ready_tasks, dtype=bool)
        for index, task in enumerate(ready_tasks[: cfg.max_ready_tasks]):
            ready_task_mask[index] = True
            ready_task_features[index] = self._encode_task(task, current_time, horizon)

        server_features = np.zeros((cfg.max_servers, RECR_SERVER_FEATURE_DIM), dtype=np.float32)
        server_mask = np.zeros(cfg.max_servers, dtype=bool)
        for index, server in enumerate(servers[: cfg.max_servers]):
            server_mask[index] = True
            server_features[index] = self._encode_server(server, current_time, horizon)

        global_features = self._encode_global(ready_tasks, servers, current_time, horizon)

        return Observation(
            ready_task_features=ready_task_features,
            ready_task_mask=ready_task_mask,
            server_features=server_features,
            server_mask=server_mask,
            global_features=global_features,
            ready_tasks=ready_tasks[: cfg.max_ready_tasks],
            servers=servers[: cfg.max_servers],
            rho_levels=rho_levels,
        )

    def compute_reward(self, context: StepContext) -> float:
        event = context.info.get("event", {})
        workflow_id = str(event.get("workflow_id", context.action.get("workflow_id", "")))
        step_energy = float(-context.env_reward) if context.env_reward else float(event.get("energy", 0.0))

        if workflow_id:
            self._workflow_energy_used[workflow_id] = (
                self._workflow_energy_used.get(workflow_id, 0.0) + step_energy
            )

        task_meta = self._find_task(context.prev_state, workflow_id, int(event.get("task_id", -1)))
        finish_time = float(event.get("finish_time", context.next_state.get("time", 0.0)))

        task_reliability = float(event.get("task_reliability", 1.0))
        energy_term = step_energy / max(self.reward_cfg.energy_scale, 1e-6)
        budget_overrun = self._energy_budget_overrun(task_meta, step_energy)
        pn_d = self._deadline_penalty(task_meta, finish_time)
        pn_r = self._reliability_penalty(task_reliability)

        cfg = self.reward_cfg
        step_reward = -(
            cfg.alpha * energy_term
            + cfg.budget_coef * budget_overrun
            + cfg.beta * pn_d
            + cfg.gamma * pn_r
        )

        if context.done:
            step_reward += self._terminal_bonus(context.next_state)

        return float(step_reward)

    def _encode_task(self, task: Mapping[str, Any], current_time: float, horizon: float) -> np.ndarray:
        workload_cycles = float(task.get("workload_cycles", 0.0))
        workload_gc = workload_cycles / 1e9
        arrival = float(task.get("arrival_time", 0.0))
        absolute_deadline = float(task.get("absolute_deadline", arrival + horizon))
        energy_budget = max(float(task.get("energy_budget", 1.0)), 1e-6)
        energy_used = float(task.get("workflow_energy_used", 0.0))
        reliability = float(task.get("workflow_reliability", 1.0))

        slack = max(absolute_deadline - current_time, 1e-6)
        min_exec = workload_cycles / max(self.state_cfg.ref_freq_hz, 1.0)
        urgency = min(min_exec / slack, 1.0)
        energy_remain = max(energy_budget - energy_used, 0.0) / energy_budget

        return np.array(
            [
                np.tanh(workload_gc / 10.0),
                arrival / horizon,
                (current_time - arrival) / horizon,
                slack / horizon,
                energy_remain,
                reliability,
                urgency,
                min_exec / horizon,
            ],
            dtype=np.float32,
        )

    def _encode_server(self, server: Mapping[str, Any], current_time: float, horizon: float) -> np.ndarray:
        available = float(server.get("execution_available_time", server.get("available_time", 0.0)))
        freq_hz = float(server.get("max_freq_hz", self.state_cfg.ref_freq_hz))
        server_type = str(server.get("type", "edge")).lower()
        is_cloud = 1.0 if server_type == "cloud" else 0.0
        queue_delay = max(available - current_time, 0.0) / horizon
        total_power = float(server.get("static_power", 0.0)) + float(
            server.get("dynamic_power", 0.0)
        )
        return np.array(
            [
                available / horizon,
                freq_hz / max(self.state_cfg.ref_freq_hz, 1.0),
                is_cloud,
                queue_delay,
                np.tanh(total_power / 100.0),
            ],
            dtype=np.float32,
        )

    def _encode_global(
        self,
        ready_tasks: tuple[Mapping[str, Any], ...],
        servers: tuple[Mapping[str, Any], ...],
        current_time: float,
        horizon: float,
    ) -> np.ndarray:
        n_ready = len(ready_tasks)
        max_ready = max(self.state_cfg.max_ready_tasks, 1)
        if ready_tasks:
            slacks = [
                max(float(task.get("absolute_deadline", current_time + horizon)) - current_time, 0.0)
                for task in ready_tasks
            ]
            mean_slack = float(np.mean(slacks)) / horizon
            energy_pressure = float(
                np.mean(
                    [
                        float(task.get("workflow_energy_used", 0.0))
                        / max(float(task.get("energy_budget", 1.0)), 1e-6)
                        for task in ready_tasks
                    ]
                )
            )
        else:
            mean_slack = 1.0
            energy_pressure = 0.0

        cloud_ratio = 0.0
        if servers:
            cloud_ratio = sum(1 for server in servers if str(server.get("type")) == "cloud") / len(
                servers
            )

        return np.array(
            [
                current_time / horizon,
                n_ready / max_ready,
                mean_slack,
                energy_pressure,
                cloud_ratio,
            ],
            dtype=np.float32,
        )

    def _find_task(
        self,
        env_state: Mapping[str, Any],
        workflow_id: str,
        task_id: int,
    ) -> dict[str, Any]:
        for task in env_state.get("ready_tasks", []):
            if str(task.get("workflow_id")) == workflow_id and int(task.get("task_id")) == task_id:
                return dict(task)
        return {
            "absolute_deadline": env_state.get("time", 0.0) + self.state_cfg.time_horizon,
            "energy_budget": 1.0,
            "workflow_energy_used": self._workflow_energy_used.get(workflow_id, 0.0),
            "reliability": self.reward_cfg.reliability_target,
        }

    def _deadline_penalty(self, task_meta: Mapping[str, Any], finish_time: float) -> float:
        absolute_deadline = float(task_meta.get("absolute_deadline", finish_time))
        if absolute_deadline <= 0:
            return 0.0
        return max(0.0, finish_time - absolute_deadline) / absolute_deadline

    def _energy_budget_overrun(self, task_meta: Mapping[str, Any], step_energy: float) -> float:
        budget = max(float(task_meta.get("energy_budget", self.reward_cfg.energy_scale)), 1e-6)
        used = float(task_meta.get("workflow_energy_used", 0.0)) + step_energy
        return max(0.0, used - budget) / budget

    def _reliability_penalty(self, task_reliability: float) -> float:
        # per-task failure probability 1 - exp(-lambda * exec_time); always active,
        # rewards picking low-failure-rate servers and shorter executions.
        return max(0.0, 1.0 - float(task_reliability))

    def _terminal_bonus(self, next_state: Mapping[str, Any]) -> float:
        cfg = self.reward_cfg
        bonus = 0.0
        for task in next_state.get("ready_tasks", []):
            absolute_deadline = float(task.get("absolute_deadline", 0.0))
            if absolute_deadline <= 0:
                continue
            if float(next_state.get("time", 0.0)) <= absolute_deadline:
                bonus += cfg.terminal_deadline_bonus * 0.01
        return bonus

    def estimate_step_penalties(
        self,
        env_state: Mapping[str, Any],
        action: Mapping[str, Any],
        servers: list[Mapping[str, Any]],
    ) -> dict[str, float]:
        """调试：估算单步 RECR 惩罚分量。"""
        task = self._find_task(env_state, str(action["workflow_id"]), int(action["task_id"]))
        server = next(item for item in servers if int(item["id"]) == int(action["server_id"]))
        rho = float(action.get("rho", 1.0))
        exec_time = compute_execution_time(float(task.get("workload_cycles", 1.0)), server, rho)
        available = float(server.get("execution_available_time", server.get("available_time", 0.0)))
        finish_time = max(float(env_state.get("time", 0.0)), available) + exec_time
        energy = compute_energy(float(task.get("workload_cycles", 1.0)), server, rho)
        task_reliability = math.exp(-float(server.get("failure_rate", 0.0)) * exec_time)
        return {
            "energy_term": energy / max(self.reward_cfg.energy_scale, 1e-6),
            "budget_overrun": self._energy_budget_overrun(task, energy),
            "pn_d": self._deadline_penalty(task, finish_time),
            "pn_r": self._reliability_penalty(task_reliability),
        }
