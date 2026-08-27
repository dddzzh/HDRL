from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from src.schedulers.heft import run_heft
from src.simulator.workflow import Workflow


EPS = 1e-9
WorkflowLike = Workflow | Mapping[str, Any]


@dataclass(frozen=True)
class SubDeadlineStats:
    """Compact diagnostics for one sub-Deadline decomposition."""

    relative_deadline: float
    base_makespan: float
    scale_ratio: float
    slack_seconds: float
    num_tasks: int
    min_sub_deadline: float
    max_sub_deadline: float
    reference_feasible: bool
    method: str


@dataclass(frozen=True)
class SubDeadlineDetails:
    """Per-task details; keys are task identifiers."""

    relative_by_task: dict[int, float]
    absolute_by_task: dict[int, float]
    heft_ast_by_task: dict[int, float]
    heft_aft_by_task: dict[int, float]
    heft_server_by_task: dict[int, int]
    method: str


def distribute_sub_deadlines_heft_scaled(
    workflow: WorkflowLike,
    plan: Mapping[str, Any],
    relative_deadline: float,
    *,
    return_details: bool = True,
) -> tuple[SubDeadlineStats, Optional[SubDeadlineDetails]]:
    """Scale HEFT task completion times into precedence-safe sub-Deadlines.

    For task ``i`` with reference HEFT completion time ``F_i`` and reference
    makespan ``M``, the relative sub-Deadline is

        d_i = (D / M) * F_i.

    This gives every task the same multiplicative relaxation as the workflow,
    preserves HEFT precedence order, and makes the reference last-exit task's
    sub-Deadline equal to the workflow Deadline.

    This is a HEFT-scaled decomposition.  It must not be described as an exact
    reproduction of a separately published ``IEE_pure`` algorithm unless that
    algorithm's equations are supplied and verified.
    """

    relative_deadline = float(relative_deadline)
    base_makespan = float(plan["reference_makespan"])
    if relative_deadline <= 0.0:
        raise ValueError("relative_deadline must be positive")
    if base_makespan <= 0.0:
        raise ValueError("HEFT reference makespan must be positive")

    task_ids = _task_ids(workflow)
    task_aft = {int(key): float(value) for key, value in plan["task_aft"].items()}
    task_ast = {int(key): float(value) for key, value in plan["task_ast"].items()}
    task_server = {
        int(key): int(value) for key, value in plan["task_server"].items()
    }
    if set(task_aft) != set(task_ids):
        raise ValueError("HEFT plan task_aft does not match workflow tasks")
    if set(task_ast) != set(task_ids) or set(task_server) != set(task_ids):
        raise ValueError("HEFT plan is missing task AST or server assignments")

    scale_ratio = relative_deadline / base_makespan
    relative_by_task = {
        task_id: scale_ratio * task_aft[task_id] for task_id in task_ids
    }
    last_exit_task = int(plan["last_exit_task"])
    relative_by_task[last_exit_task] = relative_deadline
    _validate_sub_deadlines(workflow, relative_by_task, relative_deadline)

    arrival_time = _arrival_time(workflow)
    absolute_by_task = {
        task_id: arrival_time + value
        for task_id, value in relative_by_task.items()
    }
    values = list(relative_by_task.values())
    stats = SubDeadlineStats(
        relative_deadline=relative_deadline,
        base_makespan=base_makespan,
        scale_ratio=scale_ratio,
        slack_seconds=relative_deadline - base_makespan,
        num_tasks=len(task_ids),
        min_sub_deadline=min(values),
        max_sub_deadline=max(values),
        reference_feasible=relative_deadline + EPS >= base_makespan,
        method="heft_aft_scaled_v1",
    )
    details = None
    if return_details:
        details = SubDeadlineDetails(
            relative_by_task=relative_by_task,
            absolute_by_task=absolute_by_task,
            heft_ast_by_task=task_ast,
            heft_aft_by_task=task_aft,
            heft_server_by_task=task_server,
            method=stats.method,
        )
    return stats, details


def distribute_existing_workflow_deadline(
    workflow: WorkflowLike,
    servers: Sequence[Mapping[str, Any]],
    *,
    link_bandwidth_mbps: Optional[Mapping[str, float]] = None,
    rho: float = 1.0,
    return_details: bool = True,
    cache: Optional[MutableMapping[str, dict[str, Any]]] = None,
) -> tuple[SubDeadlineStats, Optional[SubDeadlineDetails]]:
    """Distribute an already-defined workflow Deadline over its tasks.

    The function never generates or changes the workflow-level Deadline.  It
    reads ``relative_deadline`` (or derives it from ``absolute_deadline`` and
    ``arrival_time``), runs the project's reference HEFT once, and only
    returns per-task sub-Deadlines and diagnostics.
    """

    rho = float(rho)
    if rho <= 0.0:
        raise ValueError("rho must be positive")
    relative_deadline = _existing_relative_deadline(workflow)
    plan = run_heft(
        workflow,
        servers,
        link_bandwidth_mbps,
        rho=rho,
        cache=cache,
    )
    return distribute_sub_deadlines_heft_scaled(
        workflow,
        plan,
        relative_deadline,
        return_details=return_details,
    )


def _arrival_time(workflow: WorkflowLike) -> float:
    if isinstance(workflow, Workflow):
        return float(workflow.arrival_time)
    return float(workflow.get("arrival_time", 0.0))


def _existing_relative_deadline(workflow: WorkflowLike) -> float:
    if isinstance(workflow, Workflow):
        value = workflow.relative_deadline
        if value is None and workflow.absolute_deadline is not None:
            value = float(workflow.absolute_deadline) - _arrival_time(workflow)
    else:
        value = workflow.get("relative_deadline")
        if value is None and workflow.get("absolute_deadline") is not None:
            value = float(workflow["absolute_deadline"]) - _arrival_time(workflow)
    if value is None:
        raise ValueError(
            "workflow must already define relative_deadline or absolute_deadline"
        )
    value = float(value)
    if value <= 0.0:
        raise ValueError("existing workflow relative_deadline must be positive")
    return value


def _task_ids(workflow: WorkflowLike) -> list[int]:
    if isinstance(workflow, Workflow):
        return sorted(int(task.id) for task in workflow.tasks)
    return sorted(int(task["id"]) for task in workflow["tasks"])


def _parents(workflow: WorkflowLike, task_id: int) -> list[int]:
    if isinstance(workflow, Workflow):
        return workflow.get_parents(task_id)
    task_rows = {int(task["id"]): task for task in workflow["tasks"]}
    if "parents" in task_rows[task_id]:
        return [int(value) for value in task_rows[task_id].get("parents", [])]
    return [
        int(edge["src"])
        for edge in workflow.get("edges", [])
        if int(edge["dst"]) == task_id
    ]


def _validate_sub_deadlines(
    workflow: WorkflowLike,
    relative_by_task: Mapping[int, float],
    relative_deadline: float,
) -> None:
    for task_id, sub_deadline in relative_by_task.items():
        if not 0.0 < float(sub_deadline) <= relative_deadline + EPS:
            raise ValueError(
                f"invalid sub-Deadline for task {task_id}: {sub_deadline}"
            )
        for parent_id in _parents(workflow, task_id):
            if relative_by_task[parent_id] > sub_deadline + EPS:
                raise ValueError(
                    "sub-Deadline precedence violation: "
                    f"parent={parent_id}, child={task_id}"
                )
