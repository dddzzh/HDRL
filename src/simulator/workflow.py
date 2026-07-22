from __future__ import annotations

from typing import Any, Mapping, Optional


class Task:
    __slots__ = (
        "id",
        "workload_cycles",
        "parents",
        "children",
        "upward_rank",
    )

    def __init__(
        self,
        id: int,
        workload_cycles: float,
        parents: tuple[int, ...] = (),
        children: tuple[int, ...] = (),
        upward_rank: Optional[float] = None,
    ):
        self.id = id
        self.workload_cycles = workload_cycles
        self.parents = parents
        self.children = children
        self.upward_rank = upward_rank

    @property
    def workload_gc(self) -> float:
        """Workload in Giga-cycles (for logging/config readability)."""
        return self.workload_cycles / 1e9

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Task:
        if "id" not in data:
            raise ValueError("Task missing required field 'id'")

        # workload must be in cycles. Legacy fields (task_length_mi / task_length /
        # workload) are NOT trusted as cycles and are intentionally not accepted here;
        # the dataset/sampler layer is responsible for producing workload_cycles.
        if "workload_cycles" not in data:
            raise ValueError(
                f"Task {data.get('id')} missing required field 'workload_cycles' "
                "(workload must be expressed in cycles)"
            )

        parents = tuple(sorted(int(parent_id) for parent_id in data.get("parents", [])))
        children = tuple(sorted(int(child_id) for child_id in data.get("children", [])))
        return cls(
            id=int(data["id"]),
            workload_cycles=float(data["workload_cycles"]),
            parents=parents,
            children=children,
            upward_rank=_optional_float(data, "upward_rank"),
        )


class DependencyEdge:
    __slots__ = ("src", "dst", "data_size_mb")

    def __init__(self, src: int, dst: int, data_size_mb: float):
        self.src = src
        self.dst = dst
        self.data_size_mb = data_size_mb

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        default_data_size_mb: Optional[float] = None,
    ) -> DependencyEdge:
        if "src" not in data or "dst" not in data:
            raise ValueError("Dependency edge missing required fields 'src' and 'dst'")

        if "data_size_mb" in data:
            data_size_mb = float(data["data_size_mb"])
        elif default_data_size_mb is not None:
            data_size_mb = float(default_data_size_mb)
        else:
            raise ValueError(
                f"Dependency edge {data.get('src')}->{data.get('dst')} "
                "missing required field 'data_size_mb'"
            )

        return cls(src=int(data["src"]), dst=int(data["dst"]), data_size_mb=data_size_mb)


class Workflow:
    def __init__(
        self,
        workflow_id: str,
        tasks: tuple[Task, ...],
        edges: tuple[DependencyEdge, ...],
        arrival_time: float,
        reliability: Optional[float] = None,
        energy_budget: Optional[float] = None,
        input_data_mb: float = 1.0,
        output_data_mb: float = 1.0,
        reference_makespan: Optional[float] = None,
        deadline_multiplier: Optional[float] = None,
        relative_deadline: Optional[float] = None,
        absolute_deadline: Optional[float] = None,
        deadline_type: Optional[str] = None,
    ):
        self.workflow_id = workflow_id
        self.tasks = tasks
        self.edges = edges
        self.arrival_time = arrival_time
        self.reference_makespan = reference_makespan
        self.deadline_multiplier = deadline_multiplier
        self.relative_deadline = relative_deadline
        self.absolute_deadline = absolute_deadline
        self.deadline_type = deadline_type
        self.reliability = reliability
        self.energy_budget = energy_budget
        self.input_data_mb = input_data_mb
        self.output_data_mb = output_data_mb
        self.tasks, self.edge_data_mb = _build_tasks_with_adjacency(
            self.workflow_id,
            self.tasks,
            self.edges,
        )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        default_edge_data_mb: Optional[float] = None,
    ) -> Workflow:
        workflow_id = data.get("workflow_id")
        if workflow_id is None:
            raise ValueError("Workflow missing required field 'workflow_id'")

        if "arrival_time" not in data:
            raise ValueError(f"Workflow {workflow_id} missing required field 'arrival_time'")
        if "tasks" not in data or not isinstance(data["tasks"], list):
            raise ValueError(f"Workflow {workflow_id} missing list field 'tasks'")
        if "edges" not in data or not isinstance(data["edges"], list):
            raise ValueError(f"Workflow {workflow_id} missing list field 'edges'")

        tasks = tuple(Task.from_mapping(task) for task in data["tasks"])
        tasks = tuple(
            Task(
                id=task.id,
                workload_cycles=task.workload_cycles,
                upward_rank=task.upward_rank,
            )
            for task in tasks
        )
        edges = tuple(
            DependencyEdge.from_mapping(edge, default_data_size_mb=default_edge_data_mb)
            for edge in data["edges"]
        )

        return cls(
            workflow_id=str(workflow_id),
            tasks=tasks,
            edges=edges,
            arrival_time=float(data["arrival_time"]),
            reliability=_optional_float(data, "reliability", "reliability_constraint"),
            energy_budget=_optional_float(data, "energy_budget", "energy_constraint"),
            input_data_mb=float(data.get("input_data_mb", default_edge_data_mb or 1.0)),
            output_data_mb=float(data.get("output_data_mb", default_edge_data_mb or 1.0)),
            reference_makespan=_optional_float(data, "reference_makespan"),
            deadline_multiplier=_optional_float(data, "deadline_multiplier"),
            relative_deadline=_optional_float(data, "relative_deadline"),
            absolute_deadline=_optional_float(data, "absolute_deadline"),
            deadline_type=data.get("deadline_type"),
        )

    def get_task(self, task_id: int) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(f"Task {task_id} not found in workflow {self.workflow_id}")

    def get_parents(self, task_id: int) -> list[int]:
        return list(self.get_task(task_id).parents)

    def get_children(self, task_id: int) -> list[int]:
        return list(self.get_task(task_id).children)

    def get_edge_data_mb(self, src: int, dst: int) -> float:
        return self.edge_data_mb[(src, dst)]

    def task_ids(self) -> frozenset[int]:
        return frozenset(task.id for task in self.tasks)

    def entry_tasks(self) -> list[int]:
        return [task.id for task in self.tasks if not task.parents]

    def exit_tasks(self) -> list[int]:
        return [task.id for task in self.tasks if not task.children]

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "arrival_time": self.arrival_time,
            "tasks": [
                _task_to_mapping(task)
                for task in self.tasks
            ],
            "edges": [
                {
                    "src": edge.src,
                    "dst": edge.dst,
                    "data_size_mb": edge.data_size_mb,
                }
                for edge in self.edges
            ],
            "n_tasks": len(self.tasks),
            "n_edges": len(self.edges),
            "edge_data_mb": dict(self.edge_data_mb),
            "input_data_mb": self.input_data_mb,
            "output_data_mb": self.output_data_mb,
        }
        if self.absolute_deadline is not None:
            payload["absolute_deadline"] = self.absolute_deadline
        if self.reference_makespan is not None:
            payload["reference_makespan"] = self.reference_makespan
        if self.deadline_multiplier is not None:
            payload["deadline_multiplier"] = self.deadline_multiplier
        if self.relative_deadline is not None:
            payload["relative_deadline"] = self.relative_deadline
        if self.deadline_type is not None:
            payload["deadline_type"] = self.deadline_type
        if self.reliability is not None:
            payload["reliability"] = self.reliability
        if self.energy_budget is not None:
            payload["energy_budget"] = self.energy_budget
        for field in (
            "arrival_index",
            "inter_arrival_time",
            "arrival_process",
            "arrival_rate",
            "source_device_id",
            "workflow_type",
        ):
            if hasattr(self, field):
                payload[field] = getattr(self, field)
        return payload


def _task_to_mapping(task: Task) -> dict[str, Any]:
    payload = {
        "id": task.id,
        "workload_cycles": task.workload_cycles,
        "parents": list(task.parents),
        "children": list(task.children),
    }
    if task.upward_rank is not None:
        payload["upward_rank"] = task.upward_rank
    return payload


def _build_tasks_with_adjacency(
    workflow_id: str,
    tasks: tuple[Task, ...],
    edges: tuple[DependencyEdge, ...],
) -> tuple[tuple[Task, ...], dict[tuple[int, int], float]]:
    task_ids = frozenset(task.id for task in tasks)
    parents_map = {task_id: [] for task_id in task_ids}
    children_map = {task_id: [] for task_id in task_ids}
    edge_data_mb: dict[tuple[int, int], float] = {}

    for edge in edges:
        if edge.src not in task_ids or edge.dst not in task_ids:
            raise ValueError(
                f"Workflow {workflow_id} edge references unknown task: "
                f"{edge.src}->{edge.dst}"
            )
        parents_map[edge.dst].append(edge.src)
        children_map[edge.src].append(edge.dst)
        edge_data_mb[(edge.src, edge.dst)] = edge.data_size_mb

    rebuilt_tasks = tuple(
        Task(
            id=task.id,
            workload_cycles=task.workload_cycles,
            upward_rank=task.upward_rank,
            parents=tuple(sorted(parents_map[task.id])),
            children=tuple(sorted(children_map[task.id])),
        )
        for task in tasks
    )
    return rebuilt_tasks, edge_data_mb


def _optional_float(data: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in data and data[key] is not None:
            return float(data[key])
    return None


def build_parents_children(workflow, default_edge_data_mb=None):
    """Compute parents/children/edge_data_mb adjacency directly from edges.

    Decoupled from task workload so it can run before the workload-mapping layer
    has assigned ``workload_cycles`` (e.g. when loading raw Alibaba pools).
    """
    if isinstance(workflow, Workflow):
        return workflow

    task_ids = {int(task["id"]) for task in workflow["tasks"]}
    parents: dict[int, list[int]] = {task_id: [] for task_id in task_ids}
    children: dict[int, list[int]] = {task_id: [] for task_id in task_ids}
    edge_data_mb: dict[tuple[int, int], float] = {}

    for edge in workflow.get("edges", []):
        src = int(edge["src"])
        dst = int(edge["dst"])
        if src not in task_ids or dst not in task_ids:
            raise ValueError(
                f"Workflow {workflow.get('workflow_id')} edge references unknown task: "
                f"{src}->{dst}"
            )
        parents[dst].append(src)
        children[src].append(dst)
        if "data_size_mb" in edge and edge["data_size_mb"] is not None:
            edge_data_mb[(src, dst)] = float(edge["data_size_mb"])
        elif default_edge_data_mb is not None:
            edge_data_mb[(src, dst)] = float(default_edge_data_mb)

    workflow["parents"] = {task_id: sorted(parents[task_id]) for task_id in task_ids}
    workflow["children"] = {task_id: sorted(children[task_id]) for task_id in task_ids}
    workflow["edge_data_mb"] = edge_data_mb
    return workflow


def get_entry_tasks(workflow):
    if isinstance(workflow, Workflow):
        return workflow.entry_tasks()
    return [
        task_id
        for task_id, parent_ids in workflow.get("parents", {}).items()
        if not parent_ids
    ]


def get_exit_tasks(workflow):
    if isinstance(workflow, Workflow):
        return workflow.exit_tasks()
    return [
        task_id
        for task_id, child_ids in workflow.get("children", {}).items()
        if not child_ids
    ]


def is_task_ready(workflow_state, task_id):
    status = workflow_state["task_status"].get(task_id)
    if status not in ("waiting", "ready"):
        return False

    workflow = workflow_state["workflow"]
    if isinstance(workflow, Workflow):
        parent_ids = workflow.get_parents(task_id)
    else:
        parent_ids = workflow["parents"].get(task_id, [])

    return all(
        workflow_state["task_status"].get(parent_id) == "finished"
        for parent_id in parent_ids
    )


def create_workflow_state(workflow):
    if isinstance(workflow, Workflow):
        workflow_obj = workflow
    else:
        workflow_obj = Workflow.from_mapping(workflow)

    task_ids = [task.id for task in workflow_obj.tasks]
    return {
        "workflow_id": workflow_obj.workflow_id,
        "workflow": workflow_obj,
        "arrival_time": workflow_obj.arrival_time,
        "released": False,
        "completed": False,
        "task_status": {task_id: "waiting" for task_id in task_ids},
        "task_start_times": {},
        "task_finish_times": {},
        "task_server": {},
        "task_rho": {},
        "task_energy": {},
        "task_reliability": {},
        "task_primary_finish_times": {},
        "task_backup_finish_times": {},
        "task_primary_server": {},
        "task_backup_server": {},
        "task_primary_plan": {},
        "task_backup_plan": {},
        "task_final_plan": {},
        "upload_time": None,
        "download_time": None,
        "egress_finish_time": None,
        "egress_applied": False,
        "energy_used": 0.0,
        "reliability_product": 1.0,
    }
