from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

from src.schedulers.base_scheduler import BaseScheduler
from src.simulator.network import get_link_bandwidth
from src.simulator.resource import compute_execution_time, compute_transfer_time
from src.simulator.workflow import Workflow, get_exit_tasks


WorkflowLike = Union[Workflow, Mapping[str, Any]]
EPS = 1e-9


def run_heft(
    workflow: WorkflowLike,
    servers: Sequence[Mapping[str, Any]],
    link_bandwidth_mbps: Optional[Mapping[str, float]] = None,
    *,
    rho: float = 1.0,
    cache: Optional[MutableMapping[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Run the single project-wide HEFT implementation on a relative time axis."""
    signature = workflow_signature(workflow, servers, link_bandwidth_mbps, rho=rho)
    if cache is not None and signature in cache:
        result = copy.deepcopy(cache[signature])
        _write_upward_ranks(workflow, result["upward_ranks"])
        return result

    _validate_inputs(workflow, servers)
    server_rows = sorted((dict(server) for server in servers), key=lambda item: int(item["id"]))
    task_ids = _task_ids(workflow)
    topo_order = _topological_order(workflow)
    task_lengths = _task_lengths(workflow)

    avg_exec = _average_execution_times(task_lengths, server_rows, rho)
    avg_comm = _average_communication_times(workflow, server_rows, link_bandwidth_mbps)
    upward_ranks = _compute_upward_ranks(workflow, topo_order, avg_exec, avg_comm)
    _write_upward_ranks(workflow, upward_ranks)
    task_order = sorted(task_ids, key=lambda task_id: (-upward_ranks[task_id], task_id))

    server_timelines: dict[int, list[dict[str, float | int]]] = {
        int(server["id"]): [] for server in server_rows
    }
    task_server: dict[int, int] = {}
    task_ast: dict[int, float] = {}
    task_aft: dict[int, float] = {}
    task_data_ready_time: dict[int, float] = {}
    task_execution_time: dict[int, float] = {}

    for task_id in task_order:
        best = None
        for server in server_rows:
            server_id = int(server["id"])
            data_ready = _data_ready_time(
                workflow,
                task_id,
                server,
                task_server,
                task_aft,
                server_rows,
                link_bandwidth_mbps,
            )
            execution_time = _execution_time(task_lengths[task_id], server, rho)
            est = earliest_idle_slot(server_timelines[server_id], data_ready, execution_time)
            eft = est + execution_time
            candidate = (eft, est, server_id, data_ready, execution_time)
            if best is None or candidate[:3] < best[:3]:
                best = candidate

        eft, est, server_id, data_ready, execution_time = best
        task_server[task_id] = int(server_id)
        task_ast[task_id] = float(est)
        task_aft[task_id] = float(eft)
        task_data_ready_time[task_id] = float(data_ready)
        task_execution_time[task_id] = float(execution_time)
        _insert_interval(server_timelines[server_id], task_id, est, eft)

    exit_ids = get_exit_tasks(workflow)
    compute_finish_time = max((task_aft[task_id] for task_id in exit_ids), default=0.0)
    last_exit_task = (
        max(exit_ids, key=lambda task_id: (task_aft[task_id], -task_id))
        if exit_ids
        else None
    )
    download_time = 0.0
    if last_exit_task is not None:
        exit_server = _server_by_id(server_rows, task_server[last_exit_task])
        download_time = _download_time(workflow, exit_server, link_bandwidth_mbps)
    reference_makespan = compute_finish_time + download_time
    if reference_makespan <= 0:
        raise ValueError(f"reference_makespan must be positive, got {reference_makespan}")

    result = {
        "upward_ranks": {task_id: float(upward_ranks[task_id]) for task_id in task_ids},
        "task_order": list(task_order),
        "task_server": dict(task_server),
        "task_ast": dict(task_ast),
        "task_aft": dict(task_aft),
        "task_data_ready_time": dict(task_data_ready_time),
        "task_execution_time": dict(task_execution_time),
        "server_timelines": copy.deepcopy(server_timelines),
        "compute_finish_time": float(compute_finish_time),
        "download_time": float(download_time),
        "reference_makespan": float(reference_makespan),
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


def workflow_signature(
    workflow: WorkflowLike,
    servers: Sequence[Mapping[str, Any]],
    link_bandwidth_mbps: Optional[Mapping[str, float]] = None,
    *,
    rho: float = 1.0,
) -> str:
    payload = {
        "tasks": sorted(_task_lengths(workflow).items()),
        "edges": sorted(
            (src, dst, _edge_data_mb(workflow, src, dst))
            for src, dst in _edge_pairs(workflow)
        ),
        "input_data_mb": _io_data_mb(workflow)[0],
        "output_data_mb": _io_data_mb(workflow)[1],
        "servers": sorted(_server_signature(server) for server in servers),
        "links": sorted((str(key), float(value)) for key, value in (link_bandwidth_mbps or {}).items()),
        "rho": float(rho),
        "includes_ingress": True,
        "includes_egress": True,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def earliest_idle_slot(
    timeline: Sequence[Mapping[str, float]],
    data_ready_time: float,
    execution_time: float,
    *,
    eps: float = EPS,
) -> float:
    if execution_time < -eps:
        raise ValueError(f"execution_time must be non-negative, got {execution_time}")
    candidate_start = float(data_ready_time)
    for interval in sorted(timeline, key=lambda item: (float(item["start"]), float(item["finish"]))):
        start = float(interval["start"])
        finish = float(interval["finish"])
        if candidate_start + float(execution_time) <= start + eps:
            return candidate_start
        candidate_start = max(candidate_start, finish)
    return candidate_start


class HEFTScheduler(BaseScheduler):
    """HEFT baseline backed by the single run_heft implementation."""

    def reset(self):
        self._plans = {}
        self._submitted = set()

    def select_action(self, state):
        ready_tasks = state.get("ready_tasks", [])
        if not ready_tasks:
            raise ValueError("HEFT scheduler received empty ready task queue")
        servers = state.get("servers", [])
        if not servers:
            raise ValueError("HEFT scheduler received empty server list")

        workflows = state.get("workflows", {})
        link_bandwidth_mbps = state.get("link_bandwidth_mbps")
        rho = float(self.params.get("rho", 1.0))

        ranked_ready = []
        for task in ready_tasks:
            workflow_id = str(task["workflow_id"])
            if workflow_id not in self._plans:
                if workflow_id not in workflows:
                    raise ValueError("HEFTScheduler requires state['workflows'] for standard HEFT planning")
                self._plans[workflow_id] = run_heft(
                    workflows[workflow_id],
                    servers,
                    link_bandwidth_mbps,
                    rho=rho,
                )
            plan = self._plans[workflow_id]
            task_id = int(task["task_id"])
            order_index = plan["task_order"].index(task_id)
            ranked_ready.append((order_index, workflow_id, task_id, plan))

        _order_index, workflow_id, task_id, plan = min(ranked_ready, key=lambda item: item[:3])
        self._submitted.add((workflow_id, task_id))
        return {
            "workflow_id": workflow_id,
            "task_id": task_id,
            "server_id": plan["task_server"][task_id],
            "rho": rho,
            "backup": False,
            "backup_server_id": None,
        }


def _validate_inputs(workflow: WorkflowLike, servers: Sequence[Mapping[str, Any]]) -> None:
    if not servers:
        raise ValueError("servers must not be empty")
    if not _task_ids(workflow):
        raise ValueError("workflow must contain at least one task")
    for task_id, workload in _task_lengths(workflow).items():
        if workload < 0:
            raise ValueError(f"task {task_id} workload_cycles must be non-negative")


def _average_execution_times(
    task_lengths: Mapping[int, float],
    servers: Sequence[Mapping[str, Any]],
    rho: float,
) -> dict[int, float]:
    result = {}
    for task_id, workload in task_lengths.items():
        values = [_execution_time(workload, server, rho) for server in servers]
        result[task_id] = sum(values) / len(values)
    return result


def _average_communication_times(
    workflow: WorkflowLike,
    servers: Sequence[Mapping[str, Any]],
    link_bandwidth_mbps: Optional[Mapping[str, float]],
) -> dict[tuple[int, int], float]:
    result = {}
    for src, dst in _edge_pairs(workflow):
        data_mb = _edge_data_mb(workflow, src, dst)
        if data_mb <= 0 or len(servers) <= 1 or not link_bandwidth_mbps:
            result[(src, dst)] = 0.0
            continue
        values = []
        for source in servers:
            for target in servers:
                if int(source["id"]) == int(target["id"]):
                    continue
                bandwidth = get_link_bandwidth(
                    link_bandwidth_mbps,
                    _server_layer(source),
                    _server_layer(target),
                )
                values.append(compute_transfer_time(data_mb, bandwidth))
        result[(src, dst)] = sum(values) / len(values) if values else 0.0
    return result


def _compute_upward_ranks(
    workflow: WorkflowLike,
    topo_order: Sequence[int],
    avg_exec: Mapping[int, float],
    avg_comm: Mapping[tuple[int, int], float],
) -> dict[int, float]:
    ranks = {}
    for task_id in reversed(topo_order):
        children = _get_children(workflow, task_id)
        if children:
            child_path = max(
                float(avg_comm.get((task_id, child_id), 0.0)) + ranks[child_id]
                for child_id in children
            )
            rank = float(avg_exec[task_id]) + child_path
        else:
            rank = float(avg_exec[task_id])
        if not math.isfinite(rank) or rank < 0:
            raise ValueError(f"invalid upward_rank for task {task_id}: {rank}")
        ranks[task_id] = rank
    return ranks


def _write_upward_ranks(workflow: WorkflowLike, ranks: Mapping[int, float]) -> None:
    if isinstance(workflow, Workflow):
        for task in workflow.tasks:
            task.upward_rank = float(ranks[task.id])
        return
    for task in workflow["tasks"]:
        task["upward_rank"] = float(ranks[int(task["id"])])


def _data_ready_time(
    workflow: WorkflowLike,
    task_id: int,
    candidate_server: Mapping[str, Any],
    task_server: Mapping[int, int],
    task_aft: Mapping[int, float],
    servers: Sequence[Mapping[str, Any]],
    link_bandwidth_mbps: Optional[Mapping[str, float]],
) -> float:
    parents = _get_parents(workflow, task_id)
    candidate_layer = _server_layer(candidate_server)
    if not parents:
        if not link_bandwidth_mbps:
            return 0.0
        input_mb, _ = _io_data_mb(workflow)
        bandwidth = get_link_bandwidth(link_bandwidth_mbps, "iot", candidate_layer)
        return compute_transfer_time(input_mb, bandwidth)

    ready_times = []
    candidate_server_id = int(candidate_server["id"])
    for parent_id in parents:
        parent_server_id = int(task_server[parent_id])
        transfer_time = 0.0
        if parent_server_id != candidate_server_id and link_bandwidth_mbps:
            parent_server = _server_by_id(servers, parent_server_id)
            bandwidth = get_link_bandwidth(
                link_bandwidth_mbps,
                _server_layer(parent_server),
                candidate_layer,
            )
            transfer_time = compute_transfer_time(
                _edge_data_mb(workflow, parent_id, task_id),
                bandwidth,
            )
        ready_times.append(float(task_aft[parent_id]) + transfer_time)
    return max(ready_times)


def _download_time(
    workflow: WorkflowLike,
    exit_server: Mapping[str, Any],
    link_bandwidth_mbps: Optional[Mapping[str, float]],
) -> float:
    _, output_mb = _io_data_mb(workflow)
    if output_mb <= 0 or not link_bandwidth_mbps:
        return 0.0
    bandwidth = get_link_bandwidth(link_bandwidth_mbps, _server_layer(exit_server), "iot")
    return compute_transfer_time(output_mb, bandwidth)


def _execution_time(workload_cycles: float, server: Mapping[str, Any], rho: float) -> float:
    value = float(compute_execution_time(workload_cycles, server, rho))
    if value < -EPS:
        raise ValueError(f"execution time must be non-negative, got {value}")
    return max(0.0, value)


def _insert_interval(timeline: list[dict[str, float | int]], task_id: int, start: float, finish: float) -> None:
    if start > finish + EPS:
        raise ValueError(f"invalid interval: start={start}, finish={finish}")
    timeline.append({"task_id": int(task_id), "start": float(start), "finish": float(finish)})
    timeline.sort(key=lambda item: (float(item["start"]), float(item["finish"]), int(item["task_id"])))
    _validate_timeline(timeline)


def _validate_timeline(timeline: Sequence[Mapping[str, float]]) -> None:
    previous_finish = None
    for interval in timeline:
        start = float(interval["start"])
        finish = float(interval["finish"])
        if start > finish + EPS:
            raise ValueError(f"invalid interval: start={start}, finish={finish}")
        if previous_finish is not None and start < previous_finish - EPS:
            raise ValueError("server timeline intervals overlap")
        previous_finish = finish


def _topological_order(workflow: WorkflowLike) -> list[int]:
    indegree = {task_id: len(_get_parents(workflow, task_id)) for task_id in _task_ids(workflow)}
    queue = [task_id for task_id, degree in indegree.items() if degree == 0]
    order = []
    while queue:
        queue.sort()
        task_id = queue.pop(0)
        order.append(task_id)
        for child_id in _get_children(workflow, task_id):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                queue.append(child_id)
    if len(order) != len(indegree):
        raise ValueError("workflow DAG contains a cycle")
    return order


def _task_ids(workflow: WorkflowLike) -> list[int]:
    if isinstance(workflow, Workflow):
        return [task.id for task in workflow.tasks]
    return [int(task["id"]) for task in workflow["tasks"]]


def _task_lengths(workflow: WorkflowLike) -> dict[int, float]:
    if isinstance(workflow, Workflow):
        return {task.id: float(task.workload_cycles) for task in workflow.tasks}
    lengths = {}
    for task in workflow["tasks"]:
        task_id = int(task["id"])
        if "workload_cycles" not in task:
            raise ValueError(f"Task {task_id} missing 'workload_cycles'")
        lengths[task_id] = float(task["workload_cycles"])
    return lengths


def _get_parents(workflow: WorkflowLike, task_id: int) -> list[int]:
    if isinstance(workflow, Workflow):
        return workflow.get_parents(task_id)
    return list(workflow.get("parents", {}).get(task_id, []))


def _get_children(workflow: WorkflowLike, task_id: int) -> list[int]:
    if isinstance(workflow, Workflow):
        return workflow.get_children(task_id)
    return list(workflow.get("children", {}).get(task_id, []))


def _edge_pairs(workflow: WorkflowLike) -> list[tuple[int, int]]:
    if isinstance(workflow, Workflow):
        return list(workflow.edge_data_mb.keys())
    edge_data = workflow.get("edge_data_mb", {})
    if edge_data:
        return [(int(src), int(dst)) for src, dst in edge_data.keys()]
    return [(int(edge["src"]), int(edge["dst"])) for edge in workflow.get("edges", [])]


def _edge_data_mb(workflow: WorkflowLike, src: int, dst: int) -> float:
    if isinstance(workflow, Workflow):
        return workflow.get_edge_data_mb(src, dst)
    edge_data = workflow.get("edge_data_mb", {})
    if edge_data:
        return float(edge_data[(src, dst)])
    for edge in workflow.get("edges", []):
        if int(edge["src"]) == int(src) and int(edge["dst"]) == int(dst):
            return float(edge.get("data_size_mb", 0.0))
    return 0.0


def _io_data_mb(workflow: WorkflowLike) -> tuple[float, float]:
    if isinstance(workflow, Workflow):
        return float(getattr(workflow, "input_data_mb", 0.0)), float(getattr(workflow, "output_data_mb", 0.0))
    return float(workflow.get("input_data_mb", 0.0)), float(workflow.get("output_data_mb", 0.0))


def _server_by_id(servers: Sequence[Mapping[str, Any]], server_id: int) -> Mapping[str, Any]:
    for server in servers:
        if int(server["id"]) == int(server_id):
            return server
    raise KeyError(f"server not found: {server_id}")


def _server_layer(server: Mapping[str, Any]) -> str:
    return str(server.get("type", server.get("layer", "edge")))


def _server_signature(server: Mapping[str, Any]) -> tuple[Any, ...]:
    keys = (
        "id",
        "type",
        "layer",
        "server_type",
        "max_freq_hz",
        "min_rho",
    )
    return tuple((key, server.get(key)) for key in keys)
