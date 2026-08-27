from __future__ import annotations

import numpy as np

from src.shared.hierarchical_state_reward import build_shared_state_encoder
from src.simulator.resource import compute_execution_time
from src.simulator.workflow import Workflow


EPS = 1e-8


class DQNEdgeStateBuilder:
    """Eq. 27: task id, load, critical-path flag, and two resources per VM."""

    def __init__(self, num_servers, workload_scale=1e10):
        self.num_servers = int(num_servers)
        self.workload_scale = float(workload_scale)
        self._rank_cache = {}
        self._critical_cache = {}

    @property
    def state_dim(self):
        return 3 + 2 * self.num_servers

    def reset_episode(self):
        # Workflow objects are rebuilt every episode. Python may reuse an old
        # object id, so an id-keyed rank cache must never cross episodes.
        self._rank_cache.clear()
        self._critical_cache.clear()

    def rank_tasks(self, workflow, servers, network=None, beta=1.0):
        key = id(workflow)
        if key in self._rank_cache:
            return self._rank_cache[key]
        tasks, children, edge_data = self._workflow_parts(workflow)
        mean_frequency = max(np.mean([float(s["max_freq_hz"]) for s in servers]), EPS)
        bandwidth = self._representative_bandwidth(network, servers)
        base_rank = {}

        def upward(task_id):
            if task_id in base_rank:
                return base_rank[task_id]
            execution = tasks[task_id] / mean_frequency
            successors = children.get(task_id, ())
            tail = max(
                (edge_data.get((task_id, child), 0.0) * 8.0 / bandwidth + upward(child)
                 for child in successors),
                default=0.0,
            )
            base_rank[task_id] = execution + tail
            return base_rank[task_id]

        for task_id in tasks:
            upward(task_id)
        critical = self._critical_path(tasks, children, edge_data, mean_frequency, bandwidth)
        ranks = {task_id: value + float(beta) * float(task_id in critical)
                 for task_id, value in base_rank.items()}
        self._rank_cache[key], self._critical_cache[key] = ranks, critical
        return ranks

    def select_ranked_ready_task(self, env, beta=1.0):
        state = env.get_state()
        if getattr(env, "ready_task_order", None) == "upward_rank":
            # Fair-comparison mode: use the exact same standard HEFT UpRank
            # ordering as every other algorithm in the environment.
            return state["ready_tasks"][0] if state["ready_tasks"] else None
        candidates = []
        for task in state["ready_tasks"]:
            workflow = state["workflows"][task["workflow_id"]]
            ranks = self.rank_tasks(workflow, env.servers, getattr(env, "network", None), beta)
            task_id = int(task["task_id"])
            if task_id not in ranks:
                # Defensive recovery for callers that replace workflow objects
                # without resetting the episode-level cache.
                self._rank_cache.pop(id(workflow), None)
                self._critical_cache.pop(id(workflow), None)
                ranks = self.rank_tasks(workflow, env.servers, getattr(env, "network", None), beta)
            candidates.append((ranks[task_id], -float(task["arrival_time"]), task))
        if not candidates:
            return None
        return max(candidates, key=lambda row: (row[0], row[1], -int(row[2]["task_id"])))[2]

    def build_state(self, env, task, workflow):
        if task is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        ranks = self.rank_tasks(workflow, env.servers, getattr(env, "network", None))
        critical = self._critical_cache[id(workflow)]
        task_ids = self._workflow_parts(workflow)[0]
        max_task_id = max(max(task_ids), 1)
        deadline = max(float(task.get("relative_deadline", 1.0)), EPS)
        values = [
            float(task["task_id"]) / max_task_id,
            np.tanh(float(task["workload_cycles"]) / self.workload_scale),
            float(int(task["task_id"]) in critical),
        ]
        for server in env.servers:
            available = float(env._server_execution_available_time(server))
            queue_delay = max(0.0, available - float(env.current_time))
            time_availability = float(np.exp(-queue_delay / deadline))
            memory = float(server.get("memory_gb", 0.0))
            values.extend((time_availability, memory))
        vector = np.asarray(values, dtype=np.float32)
        memory_columns = np.arange(4, self.state_dim, 2)
        if len(memory_columns):
            maximum = max(float(vector[memory_columns].max()), EPS)
            vector[memory_columns] /= maximum
        if vector.shape != (self.state_dim,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid DQN-Edge state vector: shape={vector.shape}.")
        return vector

    @staticmethod
    def context(env):
        completed = [row for row in env.event_log if row.get("event") == "task_finish"]
        mean_latency = np.mean([float(row.get("finish_time", row.get("time", 0.0))) for row in completed]) if completed else 0.0
        total_energy = sum(float(state.get("energy_used", 0.0)) for state in env.workflow_states)
        return np.asarray([np.log1p(mean_latency), np.log1p(total_energy)], dtype=np.float32)

    @staticmethod
    def _representative_bandwidth(network, servers):
        if isinstance(network, dict):
            for key in ("bandwidth_mbps", "inter_bandwidth_mbps", "intra_bandwidth_mbps"):
                if key in network:
                    return max(float(network[key]), EPS)
        values = [float(server.get("bandwidth_mbps", 1000.0)) for server in servers]
        return max(float(np.mean(values)), EPS)

    @staticmethod
    def _workflow_parts(workflow):
        if isinstance(workflow, Workflow):
            tasks = {task.id: float(task.workload_cycles) for task in workflow.tasks}
            children = {task.id: tuple(task.children) for task in workflow.tasks}
            edges = {(edge.src, edge.dst): float(edge.data_size_mb) for edge in workflow.edges}
        else:
            tasks = {int(task["id"]): float(task["workload_cycles"]) for task in workflow["tasks"]}
            children = {int(k): tuple(map(int, v)) for k, v in workflow["children"].items()}
            edges = {(int(edge["src"]), int(edge["dst"])): float(edge["data_size_mb"])
                     for edge in workflow["edges"]}
        return tasks, children, edges

    @staticmethod
    def _critical_path(tasks, children, edges, frequency, bandwidth):
        memo, choice = {}, {}
        def longest(task_id):
            if task_id in memo:
                return memo[task_id]
            candidates = [(edges.get((task_id, child), 0.0) * 8.0 / bandwidth + longest(child), child)
                          for child in children.get(task_id, ())]
            tail, child = max(candidates, default=(0.0, None))
            memo[task_id] = tasks[task_id] / frequency + tail
            choice[task_id] = child
            return memo[task_id]
        start = max(tasks, key=longest)
        path = set()
        while start is not None:
            path.add(start); start = choice[start]
        return path


class DQNEdgeCommonStateBuilder:
    """Use the project's common upper state while retaining a VM-only action.

    This is the fair-comparison adapter.  It deliberately uses the same
    environment dispatch order as hierarchical D3QN instead of DQN-Edge's
    paper-specific upward-rank dispatcher.  The DQN-Edge policy still makes
    the only learned upper decision: selecting one server for the current
    task.
    """

    def __init__(self, num_servers, state_config=None):
        values = dict(state_config or {})
        values["max_servers"] = int(num_servers)
        self.encoder = build_shared_state_encoder(values)

    @property
    def state_dim(self):
        return self.encoder.high_state_dim

    def reset_episode(self):
        return None

    def select_ranked_ready_task(self, env, beta=1.0):
        del beta
        ready_tasks = env.get_state()["ready_tasks"]
        return ready_tasks[0] if ready_tasks else None

    def build_state(self, env, task, workflow):
        del workflow
        if task is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        observation = self.encoder.encode_high_state(env.get_state())
        return np.asarray(observation.vector, dtype=np.float32)

    @staticmethod
    def context(env):
        # Keep DQN-Edge's context-attention input.  It is part of the policy
        # architecture being compared, not part of the shared state vector.
        return DQNEdgeStateBuilder.context(env)
