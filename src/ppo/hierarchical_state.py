from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from src.a2c.a2c import Observation
from src.simulator.resource import compute_execution_time


TASK_FEATURE_DIM = 3
SERVER_FEATURE_DIM = 16
GLOBAL_FEATURE_DIM = 5
LOW_STATE_FEATURE_DIM = 9
EPS = 1e-8


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def first_present(mapping, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


class HierarchicalStateEncoder:
    """State builder for hierarchical PPO.

    Edit this file when changing high-level or low-level agent state features.
    """

    def __init__(self, state_cfg=None):
        state_cfg = state_cfg or {}
        self.max_ready_tasks = int(state_cfg.get("max_ready_tasks", 32))
        self.max_servers = int(state_cfg.get("max_servers", 16))
        self.time_horizon = float(state_cfg.get("time_horizon", 3600.0))
        self.queue_time_scale = float(state_cfg.get("queue_time_scale", min(self.time_horizon, 60.0)))
        self.debug_high_state = bool(state_cfg.get("debug_high_state", False))
        self._debug_high_state_printed = False

    @property
    def high_state_dim(self):
        return (
            TASK_FEATURE_DIM
            + self.max_servers * SERVER_FEATURE_DIM
            + self.max_servers
            + GLOBAL_FEATURE_DIM
        )

    @property
    def low_state_dim(self):
        return LOW_STATE_FEATURE_DIM

    def encode_high_state(self, env_state: Mapping[str, Any]) -> Observation:
        ready_tasks = tuple(env_state.get("ready_tasks", []))
        servers = tuple(env_state.get("servers", []))
        current_time = safe_float(env_state.get("time", 0.0))
        horizon = max(self.time_horizon, 1.0)
        max_server_freq_hz = self.max_server_freq_hz(servers)
        # 添加server负载
        #server_workload

        task_features = np.zeros((self.max_ready_tasks, TASK_FEATURE_DIM), dtype=np.float32)
        task_mask = np.zeros(self.max_ready_tasks, dtype=bool)
        for index, task in enumerate(ready_tasks[: self.max_ready_tasks]):
            task_mask[index] = True
            task_features[index] = self.high_task_features(task, current_time, horizon)

        current_task = ready_tasks[0] if ready_tasks else {}
        server_features = np.zeros((self.max_servers, SERVER_FEATURE_DIM), dtype=np.float32)
        server_mask = np.zeros(self.max_servers, dtype=bool)
        for index, server in enumerate(servers[: self.max_servers]):
            server_mask[index] = True
            server_features[index] = self.high_server_features(
                server,
                current_task,
                current_time,
                horizon,
                max_server_freq_hz,
            )
        if self.debug_high_state and not self._debug_high_state_printed:
            self._debug_high_state_printed = True
            self.print_high_server_debug(ready_tasks, servers, current_time, horizon, max_server_freq_hz)

        global_features = self.high_global_features(ready_tasks, servers, current_time, horizon)
        current_task_features = task_features[0] if ready_tasks else np.zeros(
            TASK_FEATURE_DIM,
            dtype=np.float32,
        )
        state_vector = np.concatenate(
            [
                current_task_features,
                server_features.reshape(-1),
                server_mask.astype(np.float32),
                global_features,
            ]
        ).astype(np.float32)
        return Observation(
            ready_task_features=task_features,
            ready_task_mask=task_mask,
            server_features=server_features,
            server_mask=server_mask,
            global_features=global_features,
            ready_tasks=ready_tasks[: self.max_ready_tasks],
            servers=servers[: self.max_servers],
            rho_levels=(),
            vector_override=state_vector,
        )

    def encode_low_state(self, observation: Observation, task_index: int, server_index: int, safety_metrics=None):
        return self.low_dvfs_features(
            observation,
            task_index,
            server_index,
            safety_metrics=safety_metrics,
        ).astype(np.float32)

    def high_task_features(self, task, current_time, horizon):
        workload = safe_float(task.get("workload_cycles", 0.0))
        absolute_deadline = safe_float(task.get("absolute_deadline", current_time + horizon))
        workflow_task_count = safe_float(task.get("workflow_task_count", 1.0), 1.0)
        return np.array(
            [
                np.tanh(workload / 1e10),
                absolute_deadline / horizon,
                workflow_task_count / max(self.max_ready_tasks, 1),
            ],
            dtype=np.float32,
        )

    def max_server_freq_hz(self, servers):
        values = [
            safe_float(server.get("max_freq_hz", 0.0))
            for server in servers[: self.max_servers]
        ]
        return max(max(values, default=1.0), 1.0)

    def high_server_features(self, server, task, current_time, horizon, max_server_freq_hz):
        metrics = self.high_server_pair_metrics(server, task, current_time, horizon, max_server_freq_hz)
        return np.array(
            [
                self.norm_queue_time(metrics["wait_time"]),
                self.norm_count(metrics["queue_count"]),
                self.norm_queue_time(metrics["queue_load"]),
                self.norm_queue_time(metrics["queued_processing_time"]),
                self.norm_queue_time(metrics["estimated_execution_time"]),
                self.norm_abs_time(metrics["estimated_start_time"]),
                self.norm_abs_time(metrics["estimated_finish_time"]),
                self.norm_signed_time(metrics["slack_after_assign"]),
                self.norm_slack_ratio(metrics["slack_after_assign"], metrics["absolute_deadline"], current_time),
                self.norm_queue_time(metrics["communication_time"]),
                metrics["bandwidth_mbps"] / 10000.0,
                metrics["max_freq_hz"] / max(max_server_freq_hz, EPS),
                np.tanh(metrics["total_power"] / 250.0),
                np.tanh(metrics["failure_rate"] / 1e-3),
                np.clip(metrics["utilization"], 0.0, 1.0),
                1.0 if str(server.get("type", "")).lower() == "cloud" else 0.0,
            ],
            dtype=np.float32,
        )

    def high_server_pair_metrics(self, server, task, current_time, horizon, max_server_freq_hz):
        """Raw task-server pair metrics used by high-level server features."""
        max_freq_hz = safe_float(server.get("max_freq_hz", max_server_freq_hz), max_server_freq_hz)
        static_power = safe_float(server.get("static_power", 0.0))
        dynamic_power = safe_float(server.get("dynamic_power", 0.0))
        bandwidth_mbps = safe_float(server.get("bandwidth_mbps", 0.0))
        available_time = safe_float(
            first_present(server, ("execution_available_time", "available_time"), current_time),
            current_time,
        )
        wait_time = max(available_time - current_time, 0.0)
        queue_load = max(
            safe_float(first_present(server, ("execution_queue_load", "queue_load"), wait_time), wait_time),
            0.0,
        )
        queue_count = max(
            safe_float(first_present(server, ("execution_queue_count", "queue_count"), 0.0), 0.0),
            0.0,
        )
        # The simulator currently exposes queue backlog mainly as time load.
        # If env later provides queued workload cycles, map it to processing
        # time there and pass one of these fields into state.
        queued_processing_time = max(
            safe_float(
                first_present(
                    server,
                    (
                        "execution_queue_processing_time",
                        "queue_processing_time",
                        "execution_queue_workload_time",
                        "queue_workload_time",
                        "execution_queue_load",
                        "queue_load",
                    ),
                    queue_load,
                ),
                queue_load,
            ),
            0.0,
        )
        failure_rate = safe_float(server.get("failure_rate", 0.0))
        utilization = safe_float(server.get("cpu_utilization", 0.0))
        total_power = static_power + dynamic_power

        workload = safe_float(task.get("workload_cycles", 0.0))
        estimated_execution_time = self.estimate_execution_time(workload, server)
        task_ready_time = safe_float(
            first_present(task, ("task_ready_time", "ready_time", "arrival_time"), current_time),
            current_time,
        )
        # Exact communication depends on parent placement and edge data. The
        # high-level state keeps this hook for env-provided estimates; otherwise
        # it stays zero instead of inventing an inconsistent transfer model.
        communication_time = max(
            safe_float(
                first_present(
                    server,
                    ("estimated_comm_time", "communication_time", "transfer_time", "upload_time"),
                    first_present(
                        task,
                        ("estimated_comm_time", "communication_time", "transfer_time", "upload_time"),
                        0.0,
                    ),
                ),
                0.0,
            ),
            0.0,
        )
        estimated_start_time = max(current_time, task_ready_time + communication_time, available_time)
        estimated_finish_time = estimated_start_time + estimated_execution_time
        absolute_deadline = safe_float(
            task.get("absolute_deadline", current_time + horizon),
            current_time + horizon,
        )
        slack_after_assign = absolute_deadline - estimated_finish_time

        return {
            "wait_time": wait_time,
            "queue_load": queue_load,
            "queue_count": queue_count,
            "queued_processing_time": queued_processing_time,
            "estimated_execution_time": estimated_execution_time,
            "estimated_start_time": estimated_start_time,
            "estimated_finish_time": estimated_finish_time,
            "slack_after_assign": slack_after_assign,
            "communication_time": communication_time,
            "absolute_deadline": absolute_deadline,
            "bandwidth_mbps": bandwidth_mbps,
            "max_freq_hz": max_freq_hz,
            "total_power": total_power,
            "failure_rate": failure_rate,
            "utilization": utilization,
        }

    def estimate_execution_time(self, workload_cycles, server):
        if workload_cycles <= 0:
            return 0.0
        try:
            return max(float(compute_execution_time(workload_cycles, server, rho=1.0)), 0.0)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            max_freq_hz = max(
                safe_float(server.get("max_freq_hz", 1.0), 1.0),
                EPS,
            )
            return max(float(workload_cycles) / max_freq_hz, 0.0)

    def norm_abs_time(self, value):
        return np.tanh(max(safe_float(value), 0.0) / max(self.time_horizon, 1.0))

    def norm_queue_time(self, value):
        return np.tanh(max(safe_float(value), 0.0) / max(self.queue_time_scale, 1.0))

    def norm_signed_time(self, value):
        return np.tanh(safe_float(value) / max(self.time_horizon, 1.0))

    def norm_slack_ratio(self, slack, deadline, current_time):
        scale = max(abs(safe_float(deadline) - safe_float(current_time)), 1.0)
        return np.tanh(safe_float(slack) / scale)

    def norm_count(self, value):
        return np.tanh(max(safe_float(value), 0.0) / max(self.max_ready_tasks, 1))

    def high_server_debug_rows(self, env_state: Mapping[str, Any], limit=None):
        """Inspect unnormalized high-level task-server pair metrics."""
        ready_tasks = tuple(env_state.get("ready_tasks", []))
        servers = tuple(env_state.get("servers", []))
        current_time = safe_float(env_state.get("time", 0.0))
        horizon = max(self.time_horizon, 1.0)
        max_server_freq_hz = self.max_server_freq_hz(servers)
        task = ready_tasks[0] if ready_tasks else {}
        rows = []
        for server in servers[: limit or self.max_servers]:
            metrics = self.high_server_pair_metrics(server, task, current_time, horizon, max_server_freq_hz)
            rows.append(
                {
                    "server_id": server.get("id"),
                    "wait_time": metrics["wait_time"],
                    "queue_count": metrics["queue_count"],
                    "queued_processing_time": metrics["queued_processing_time"],
                    "estimated_finish_time": metrics["estimated_finish_time"],
                    "slack_after_assign": metrics["slack_after_assign"],
                }
            )
        return rows

    def print_high_server_debug(self, ready_tasks, servers, current_time, horizon, max_server_freq_hz, limit=5):
        task = ready_tasks[0] if ready_tasks else {}
        print("[high_state_debug] task-server pair metrics")
        for server in servers[:limit]:
            metrics = self.high_server_pair_metrics(server, task, current_time, horizon, max_server_freq_hz)
            print(
                "  "
                f"server={server.get('id')} "
                f"wait={metrics['wait_time']:.4f}s "
                f"q={metrics['queue_count']:.0f} "
                f"backlog={metrics['queued_processing_time']:.4f}s "
                f"finish={metrics['estimated_finish_time']:.4f}s "
                f"slack={metrics['slack_after_assign']:.4f}s"
            )

    def high_global_features(self, ready_tasks, servers, current_time, horizon):
        ready_ratio = len(ready_tasks) / max(self.max_ready_tasks, 1)
        server_ratio = len(servers) / max(self.max_servers, 1)
        if ready_tasks:
            mean_slack = float(
                np.mean(
                    [
                        max(safe_float(task.get("absolute_deadline", current_time + horizon)) - current_time, 0.0)
                        for task in ready_tasks
                    ]
                )
            ) / horizon
        else:
            mean_slack = 1.0
        busy_ratio = 0.0
        if servers:
            busy_ratio = sum(
                1
                for server in servers
                if safe_float(
                    server.get("execution_available_time", server.get("available_time", 0.0))
                )
                > current_time
            ) / len(servers)
        return np.array(
            [current_time / horizon, ready_ratio, server_ratio, mean_slack, busy_ratio],
            dtype=np.float32,
        )

    def low_task_features(self, observation, task_index):
        return observation.ready_task_features[task_index]

    def low_server_features(self, observation, server_index):
        return observation.server_features[server_index]

    def low_dvfs_features(self, observation, task_index, server_index, safety_metrics=None):
        horizon = max(self.time_horizon, 1.0)
        current_time = safe_float(observation.global_features[0]) * horizon

        task = {}
        if 0 <= task_index < len(observation.ready_tasks):
            task = observation.ready_tasks[task_index]

        server = {}
        if 0 <= server_index < len(observation.servers):
            server = observation.servers[server_index]
        max_server_freq_hz = self.max_server_freq_hz(observation.servers)

        available_time = safe_float(
            server.get("execution_available_time", server.get("available_time", current_time)),
            current_time,
        )
        estimated_start_time = max(current_time, available_time)
        absolute_deadline = safe_float(task.get("absolute_deadline", current_time + horizon), current_time + horizon)
        workflow_task_count = safe_float(task.get("workflow_task_count", 1.0), 1.0)
        max_freq_hz = safe_float(server.get("max_freq_hz", max_server_freq_hz), max_server_freq_hz)
        queue_count = safe_float(server.get("execution_queue_count", server.get("queue_count", 0.0)))
        static_power = safe_float(server.get("static_power", 0.0))
        dynamic_power = safe_float(server.get("dynamic_power", 0.0))
        reliability_req = safe_float(task.get("reliability", 1.0), 1.0)
        safety_metrics = safety_metrics or {}
        workflow_relative_deadline = max(
            safe_float(
                safety_metrics.get(
                    "workflow_relative_deadline",
                    task.get("relative_deadline", absolute_deadline - current_time),
                ),
                1.0,
            ),
            EPS,
        )
        estimated_ast = safe_float(safety_metrics.get("estimated_ast", estimated_start_time), estimated_start_time)
        execution_time_rho1 = safe_float(
            safety_metrics.get(
                "execution_time_rho_1",
                self.estimate_execution_time(safe_float(task.get("workload_cycles", 0.0)), server),
            ),
            0.0,
        )
        workflow_absolute_deadline = safe_float(
            safety_metrics.get("workflow_absolute_deadline", absolute_deadline),
            absolute_deadline,
        )
        safe_execution_budget = safe_float(
            safety_metrics.get("safe_execution_budget", workflow_absolute_deadline - estimated_ast),
            workflow_absolute_deadline - estimated_ast,
        )
        rho_safe = safe_float(safety_metrics.get("rho_safe", server.get("min_rho", 0.1)), 0.1)
        workload = safe_float(task.get("workload_cycles", 0.0))

        return np.array(
            [
                np.tanh(workload / 1e10),
                np.clip(execution_time_rho1 / workflow_relative_deadline, 0.0, 10.0),
                np.clip(max(estimated_ast - current_time, 0.0) / workflow_relative_deadline, 0.0, 10.0),
                np.clip((workflow_absolute_deadline - current_time) / workflow_relative_deadline, -10.0, 10.0),
                np.clip(safe_execution_budget / workflow_relative_deadline, -10.0, 10.0),
                np.clip(rho_safe, 0.0, 1.0),
                np.tanh(static_power / 100.0),
                np.tanh(dynamic_power / 250.0),
                reliability_req,
            ],
            dtype=np.float32,
        )
