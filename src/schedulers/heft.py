from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

from src.schedulers.base_scheduler import BaseScheduler
from src.simulator.network import get_link_bandwidth
from src.simulator.resource import compute_execution_time
from src.simulator.workflow import Workflow


WorkflowLike = Union[Workflow, Mapping[str, Any]]


def calculate_upward_ranks(
    workflow: WorkflowLike,
    servers: Sequence[Mapping[str, Any]],
    link_bandwidth_mbps: Optional[Mapping[str, float]] = None,
    *,
    rho: float = 1.0,
) -> dict[int, float]:
    """Compute standard HEFT ranks for the project's edge-cloud DAG model.

    Only data on DAG dependency edges contributes communication cost. There is
    no IoT-device upload for entry tasks and no result download for exit tasks.
    """
    _validate_inputs(workflow, servers)
    rho = float(rho)
    server_rows = [dict(server) for server in servers]
    mean_transfer_seconds_per_mb = _mean_transfer_seconds_per_mb(
        server_rows,
        link_bandwidth_mbps,
    )

    ranks: dict[int, float] = {}
    for task_id in reversed(_topological_order(workflow)):
        workload = _task_lengths(workflow)[task_id]
        computation_time = sum(
            compute_execution_time(workload, server, rho)
            for server in server_rows
        ) / len(server_rows)
        children = _get_children(workflow, task_id)
        max_successor_cost = max(
            (
                _edge_data_mb(workflow, task_id, child_id)
                * mean_transfer_seconds_per_mb
                + ranks[child_id]
                for child_id in children
            ),
            default=0.0,
        )
        ranks[task_id] = float(computation_time + max_successor_cost)

    _write_upward_ranks(workflow, ranks)
    return ranks


def run_heft(
    workflow: WorkflowLike,
    servers: Sequence[Mapping[str, Any]],
    link_bandwidth_mbps: Optional[Mapping[str, float]] = None,
    *,
    rho: float = 1.0,
    cache: Optional[MutableMapping[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run queue-tail HEFT using only inter-task DAG communication."""
    signature = workflow_signature(
        workflow,
        servers,
        link_bandwidth_mbps,
        rho=rho,
    )
    if cache is not None and signature in cache:
        result = copy.deepcopy(cache[signature])
        _write_upward_ranks(workflow, result["upward_ranks"])
        return result

    ranks = calculate_upward_ranks(
        workflow,
        servers,
        link_bandwidth_mbps,
        rho=rho,
    )
    server_rows = sorted((dict(server) for server in servers), key=lambda item: int(item["id"]))
    server_available = {int(server["id"]): 0.0 for server in server_rows}
    server_timelines: dict[int, list[dict[str, float | int]]] = {
        int(server["id"]): [] for server in server_rows
    }
    task_server: dict[int, int] = {}
    task_ast: dict[int, float] = {}
    task_processing_aft: dict[int, float] = {}
    task_aft: dict[int, float] = {}
    task_data_ready_time: dict[int, float] = {}
    task_execution_time: dict[int, float] = {}
    task_download_time: dict[int, float] = {}
    task_order: list[int] = []
    unscheduled = set(_task_ids(workflow))

    while unscheduled:
        ready = [
            task_id
            for task_id in unscheduled
            if all(parent_id not in unscheduled for parent_id in _get_parents(workflow, task_id))
        ]
        if not ready:
            raise ValueError("workflow DAG contains a cycle")
        task_id = min(ready, key=lambda value: (-ranks[value], value))
        workload = _task_lengths(workflow)[task_id]
        candidate_rows = []
        for server in server_rows:
            server_id = int(server["id"])
            data_ready_time = max(
                (
                    task_aft[parent_id]
                    + _interserver_transfer_time(
                        _edge_data_mb(workflow, parent_id, task_id),
                        _server_by_id(server_rows, task_server[parent_id]),
                        server,
                        link_bandwidth_mbps,
                    )
                    for parent_id in _get_parents(workflow, task_id)
                ),
                default=0.0,
            )
            ast = max(server_available[server_id], data_ready_time)
            execution_time = float(compute_execution_time(workload, server, rho))
            processing_aft = ast + execution_time
            eft = processing_aft
            candidate_rows.append(
                (eft, ast, server_id, data_ready_time, execution_time, processing_aft)
            )

        eft, ast, server_id, data_ready, execution_time, processing_aft = min(
            candidate_rows, key=lambda item: item[:3]
        )
        task_order.append(task_id)
        task_server[task_id] = int(server_id)
        task_ast[task_id] = float(ast)
        task_processing_aft[task_id] = float(processing_aft)
        task_aft[task_id] = float(eft)
        task_data_ready_time[task_id] = float(data_ready)
        task_execution_time[task_id] = float(execution_time)
        task_download_time[task_id] = 0.0
        server_available[server_id] = float(processing_aft)
        server_timelines[server_id].append(
            {"task_id": int(task_id), "start": float(ast), "finish": float(processing_aft)}
        )
        unscheduled.remove(task_id)

    exit_ids = [task_id for task_id in _task_ids(workflow) if not _get_children(workflow, task_id)]
    last_exit_task = max(exit_ids, key=lambda task_id: (task_aft[task_id], -task_id))
    reference_makespan = float(task_aft[last_exit_task])
    compute_finish_time = float(task_processing_aft[last_exit_task])
    download_time = 0.0
    if reference_makespan <= 0.0:
        raise ValueError(f"reference_makespan must be positive, got {reference_makespan}")

    result = {
        "algorithm": "heft_dag_edge_cloud_eft",
        "source": "edge_cloud_dag_model",
        "upward_ranks": dict(ranks),
        "task_order": task_order,
        "task_server": task_server,
        "task_ast": task_ast,
        "task_processing_aft": task_processing_aft,
        "task_aft": task_aft,
        "task_data_ready_time": task_data_ready_time,
        "task_execution_time": task_execution_time,
        "task_download_time": task_download_time,
        "server_timelines": server_timelines,
        "compute_finish_time": compute_finish_time,
        "download_time": download_time,
        "reference_makespan": reference_makespan,
        "last_exit_task": last_exit_task,
    }
    if cache is not None:
        cache[signature] = copy.deepcopy(result)
    return result


def reference_heft_makespan(
    workflow: WorkflowLike,
    servers: Sequence[Mapping[str, Any]],
    link_bandwidth_mbps: Optional[Mapping[str, float]] = None,
    *,
    rho: float = 1.0,
    cache: Optional[MutableMapping[str, dict[str, Any]]] = None,
) -> float:
    return float(
        run_heft(
            workflow,
            servers,
            link_bandwidth_mbps,
            rho=rho,
            cache=cache,
        )["reference_makespan"]
    )


class HEFTScheduler(BaseScheduler):
    """Online UpwardRank + EFT scheduler for edge-cloud DAG execution."""

    def __init__(self, params=None, seed=42):
        super().__init__(params=params, seed=seed)
        self.reset()

    def reset(self):
        self._rank_cache: dict[str, dict[int, float]] = {}

    def select_action(self, state, context_builder=None):
        ready_tasks = list(state.get("ready_tasks", []))
        servers = list(state.get("servers", []))
        workflows = state.get("workflows", {})
        if not ready_tasks:
            raise ValueError("HEFT scheduler received empty ready task queue")
        if not servers:
            raise ValueError("HEFT scheduler received empty server list")

        rho = float(self.params.get("rho", 1.0))
        link_bandwidth_mbps = state.get("link_bandwidth_mbps")
        ranked_ready = []
        for task in ready_tasks:
            workflow_id = str(task["workflow_id"])
            workflow = workflows.get(workflow_id)
            if workflow is None:
                raise ValueError(f"HEFT workflow is missing from state: {workflow_id}")
            signature = workflow_signature(
                workflow,
                servers,
                link_bandwidth_mbps,
                rho=rho,
            )
            ranks = self._rank_cache.get(signature)
            if ranks is None:
                ranks = calculate_upward_ranks(
                    workflow,
                    servers,
                    link_bandwidth_mbps,
                    rho=rho,
                )
                self._rank_cache[signature] = dict(ranks)
            task_id = int(task["task_id"])
            ranked_ready.append(
                (-float(ranks[task_id]), float(task.get("arrival_time", 0.0)), workflow_id, task_id, task)
            )

        _negative_rank, _arrival, workflow_id, task_id, task = min(ranked_ready)
        workflow = workflows[workflow_id]
        context = None
        if context_builder is not None:
            context = context_builder(workflow_id, task_id)
        else:
            candidate = state.get("decision_context")
            if (
                isinstance(candidate, Mapping)
                and str(candidate.get("workflow_id")) == workflow_id
                and int(candidate.get("task_id", -1)) == task_id
            ):
                context = candidate

        server_scores = []
        metrics_by_id = {} if context is None else context.get("server_safety_metrics", {})
        for server in servers:
            server_id = int(server["id"])
            metrics = metrics_by_id.get(str(server_id), {})
            if metrics:
                ast = float(metrics["estimated_ast"])
                execution_time = float(metrics["execution_time_rho_1"])
            else:
                ast = max(
                    float(state.get("time", 0.0)),
                    float(task.get("task_ready_time", task.get("arrival_time", 0.0))),
                    float(server.get("execution_available_time", server.get("available_time", 0.0))),
                )
                execution_time = float(
                    compute_execution_time(float(task["workload_cycles"]), server, rho)
                )
            eft = ast + execution_time
            server_scores.append((eft, ast, server_id, execution_time))

        eft, ast, server_id, execution_time = min(
            server_scores, key=lambda item: item[:3]
        )
        return {
            "workflow_id": workflow_id,
            "task_id": task_id,
            "server_id": int(server_id),
            "rho": rho,
            "backup": False,
            "backup_server_id": None,
            "upward_rank": float(-_negative_rank),
            "estimated_ast": float(ast),
            "execution_time_rho_1": float(execution_time),
            "download_time": 0.0,
            "estimated_finish_time": float(eft),
            "algorithm": "heft_dag_edge_cloud_eft",
            "online_queue_aware": context is not None,
        }


def workflow_signature(
    workflow: WorkflowLike,
    servers: Sequence[Mapping[str, Any]],
    link_bandwidth_mbps: Optional[Mapping[str, float]] = None,
    *,
    rho: float = 1.0,
) -> str:
    payload = {
        "tasks": sorted(_task_lengths(workflow).items()),
        "parents": sorted((task_id, tuple(_get_parents(workflow, task_id))) for task_id in _task_ids(workflow)),
        "edges": sorted((src, dst, _edge_data_mb(workflow, src, dst)) for src, dst in _edge_pairs(workflow)),
        "links": sorted(
            (str(key), float(value))
            for key, value in (link_bandwidth_mbps or {}).items()
        ),
        "servers": sorted(
            (
                int(server["id"]),
                float(server["max_freq_hz"]),
                _server_layer(server),
                _server_bandwidth(server),
            )
            for server in servers
        ),
        "rho": float(rho),
        "algorithm": "heft_dag_edge_cloud_eft_v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_inputs(workflow: WorkflowLike, servers: Sequence[Mapping[str, Any]]) -> None:
    if not servers:
        raise ValueError("servers must not be empty")
    if not _task_ids(workflow):
        raise ValueError("workflow must contain at least one task")
    for task_id, workload in _task_lengths(workflow).items():
        if workload < 0.0:
            raise ValueError(f"task {task_id} workload_cycles must be non-negative")
    for server in servers:
        if float(server.get("max_freq_hz", 0.0)) <= 0.0:
            raise ValueError(f"server {server.get('id')} max_freq_hz must be positive")
        if _server_bandwidth(server) <= 0.0:
            raise ValueError(f"server {server.get('id')} bandwidth must be positive")


def _server_bandwidth(server: Mapping[str, Any]) -> float:
    value = server.get("bandwidth_mbps", server.get("bandwidth"))
    if value is None:
        raise ValueError(f"server {server.get('id')} is missing bandwidth_mbps")
    return float(value)


def _server_layer(server: Mapping[str, Any]) -> str:
    return str(server.get("layer", server.get("type", ""))).lower()


def _server_by_id(
    servers: Sequence[Mapping[str, Any]],
    server_id: int,
) -> Mapping[str, Any]:
    for server in servers:
        if int(server["id"]) == int(server_id):
            return server
    raise KeyError(f"HEFT server not found: {server_id}")


def _pair_bandwidth_mbps(
    source_server: Mapping[str, Any],
    target_server: Mapping[str, Any],
    link_bandwidth_mbps: Optional[Mapping[str, float]],
) -> float:
    if link_bandwidth_mbps:
        bandwidth = get_link_bandwidth(
            link_bandwidth_mbps,
            _server_layer(source_server),
            _server_layer(target_server),
        )
    else:
        # Backward-compatible fallback for standalone HEFT callers. Production
        # runs pass the explicit edge-cloud link table through the environment.
        bandwidth = min(
            _server_bandwidth(source_server),
            _server_bandwidth(target_server),
        )
    if bandwidth <= 0.0:
        raise ValueError("HEFT link bandwidth must be positive")
    return float(bandwidth)


def _interserver_transfer_time(
    data_size_mb: float,
    source_server: Mapping[str, Any],
    target_server: Mapping[str, Any],
    link_bandwidth_mbps: Optional[Mapping[str, float]],
) -> float:
    data_size_mb = float(data_size_mb)
    if data_size_mb < 0.0:
        raise ValueError("HEFT edge data_size_mb must be non-negative")
    if int(source_server["id"]) == int(target_server["id"]) or data_size_mb == 0.0:
        return 0.0
    return 8.0 * data_size_mb / _pair_bandwidth_mbps(
        source_server,
        target_server,
        link_bandwidth_mbps,
    )


def _mean_transfer_seconds_per_mb(
    servers: Sequence[Mapping[str, Any]],
    link_bandwidth_mbps: Optional[Mapping[str, float]],
) -> float:
    pair_costs = [
        8.0 / _pair_bandwidth_mbps(source, target, link_bandwidth_mbps)
        for source in servers
        for target in servers
        if int(source["id"]) != int(target["id"])
    ]
    return float(sum(pair_costs) / len(pair_costs)) if pair_costs else 0.0


def _write_upward_ranks(workflow: WorkflowLike, ranks: Mapping[int, float]) -> None:
    if isinstance(workflow, Workflow):
        for task in workflow.tasks:
            task.upward_rank = float(ranks[task.id])
        return
    for task in workflow["tasks"]:
        task["upward_rank"] = float(ranks[int(task["id"])])


def _topological_order(workflow: WorkflowLike) -> list[int]:
    indegree = {task_id: len(_get_parents(workflow, task_id)) for task_id in _task_ids(workflow)}
    queue = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
    order = []
    while queue:
        task_id = queue.pop(0)
        order.append(task_id)
        for child_id in _get_children(workflow, task_id):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                queue.append(child_id)
                queue.sort()
    if len(order) != len(indegree):
        raise ValueError("workflow DAG contains a cycle")
    return order


def _task_ids(workflow: WorkflowLike) -> list[int]:
    if isinstance(workflow, Workflow):
        return [int(task.id) for task in workflow.tasks]
    return [int(task["id"]) for task in workflow["tasks"]]


def _task_lengths(workflow: WorkflowLike) -> dict[int, float]:
    if isinstance(workflow, Workflow):
        return {int(task.id): float(task.workload_cycles) for task in workflow.tasks}
    return {int(task["id"]): float(task["workload_cycles"]) for task in workflow["tasks"]}


def _get_parents(workflow: WorkflowLike, task_id: int) -> list[int]:
    if isinstance(workflow, Workflow):
        return [int(value) for value in workflow.get_parents(task_id)]
    parents = workflow.get("parents", {})
    values = parents.get(task_id, parents.get(str(task_id)))
    if values is None:
        task = next(item for item in workflow["tasks"] if int(item["id"]) == task_id)
        values = task.get("parents", [])
    return [int(value) for value in values]


def _get_children(workflow: WorkflowLike, task_id: int) -> list[int]:
    if isinstance(workflow, Workflow):
        return [int(value) for value in workflow.get_children(task_id)]
    children = workflow.get("children", {})
    values = children.get(task_id, children.get(str(task_id)))
    if values is None:
        task = next(item for item in workflow["tasks"] if int(item["id"]) == task_id)
        values = task.get("children", [])
    return [int(value) for value in values]


def _edge_pairs(workflow: WorkflowLike) -> list[tuple[int, int]]:
    if isinstance(workflow, Workflow):
        return [(int(src), int(dst)) for src, dst in workflow.edge_data_mb]
    if workflow.get("edges"):
        return [(int(edge["src"]), int(edge["dst"])) for edge in workflow["edges"]]
    edge_data = workflow.get("edge_data_mb", {})
    return [(int(src), int(dst)) for src, dst in edge_data]


def _edge_data_mb(workflow: WorkflowLike, src: int, dst: int) -> float:
    if isinstance(workflow, Workflow):
        return float(workflow.get_edge_data_mb(src, dst))
    edge_data = workflow.get("edge_data_mb", {})
    if (src, dst) in edge_data:
        return float(edge_data[(src, dst)])
    if str((src, dst)) in edge_data:
        return float(edge_data[str((src, dst))])
    for edge in workflow.get("edges", []):
        if int(edge["src"]) == src and int(edge["dst"]) == dst:
            return float(edge.get("data_size_mb", 0.0))
    return 0.0
