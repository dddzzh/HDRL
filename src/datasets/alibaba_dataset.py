import copy
import json
import random
from pathlib import Path

from src.datasets.base_dataset import BaseWorkflowDataset
from src.datasets.arrival_process import (
    resolve_arrival_rate,
    sample_configured_arrivals,
)
from src.simulator.workflow import build_parents_children


class AlibabaWorkflowDataset(BaseWorkflowDataset):
    def __init__(self, config, seed=42):
        super().__init__(config)
        self.seed = seed

    def load(self):
        path = Path(self.config.get("path", ""))
        if not path.exists():
            raise FileNotFoundError(f"Alibaba workflow JSON not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if "workflows" not in data or not isinstance(data["workflows"], list):
            raise ValueError(f"Invalid workflow JSON, missing list field 'workflows': {path}")

        # Alibaba edges carry no data_size_mb; fall back to the JSON network block,
        # then the dataset config, then a sane default so loading never crashes.
        default_edge_data_mb = data.get("network", {}).get("edge_data_mb")
        if default_edge_data_mb is None:
            default_edge_data_mb = self.config.get("edge_data_mb", 1.0)
        workflows = [
            self._normalize_workflow(workflow, default_edge_data_mb)
            for workflow in data["workflows"]
        ]

        if self.config.get("shuffle", False):
            rng = random.Random(self.seed)
            rng.shuffle(workflows)

        max_workflows = self.config.get("max_workflows")
        if max_workflows is not None:
            workflows = workflows[: int(max_workflows)]

        self._ensure_arrival_times(workflows)
        workflows.sort(key=lambda workflow: workflow["arrival_time"])
        return workflows

    def _ensure_arrival_times(self, workflows):
        """为缺少 arrival_time 的工作流按泊松过程补上到达时间。

        build_alibaba_workflows.py 的原始产物不含 arrival_time（由
        add_poisson_arrivals.py 另加）。这里兜底，让任何产物都能直接加载。
        """
        pending = [workflow for workflow in workflows if workflow.get("_needs_arrival")]
        if not pending:
            return
        arrival_rate = resolve_arrival_rate(self.config)
        start_time = float(self.config.get("arrival_start_time", self.config.get("start_time", 0.0)))
        arrival_times, inter_arrival_times, source_device_ids, arrival_process = sample_configured_arrivals(
            len(pending),
            self.config,
            start_time,
            random.Random(self.seed),
        )
        for index, workflow in enumerate(pending):
            workflow["arrival_time"] = float(arrival_times[index])
            workflow["inter_arrival_time"] = float(inter_arrival_times[index])
            workflow["arrival_process"] = arrival_process
            workflow["arrival_rate"] = float(arrival_rate)
            workflow["source_device_id"] = source_device_ids[index]
            workflow.pop("_needs_arrival", None)

    def _normalize_workflow(self, workflow, default_edge_data_mb=None):
        workflow_id = workflow.get("workflow_id")
        if workflow_id is None:
            raise ValueError("Workflow missing required field 'workflow_id'")
        if "tasks" not in workflow or not isinstance(workflow["tasks"], list):
            raise ValueError(f"Workflow {workflow_id} missing list field 'tasks'")
        if "edges" not in workflow or not isinstance(workflow["edges"], list):
            raise ValueError(f"Workflow {workflow_id} missing list field 'edges'")

        normalized = copy.deepcopy(workflow)
        normalized["workflow_id"] = str(workflow_id)
        # arrival_time may be absent in raw build output; use a placeholder now and
        # synthesize a Poisson arrival later in load() (marked via _needs_arrival).
        if "arrival_time" in workflow and workflow["arrival_time"] is not None:
            normalized["arrival_time"] = float(workflow["arrival_time"])
        else:
            normalized["arrival_time"] = 0.0
            normalized["_needs_arrival"] = True
        normalized["tasks"] = [self._normalize_task(workflow_id, task) for task in workflow["tasks"]]
        normalized["edges"] = [
            self._normalize_edge(workflow_id, edge, default_edge_data_mb)
            for edge in workflow["edges"]
        ]
        normalized["n_tasks"] = len(normalized["tasks"])
        normalized["n_edges"] = len(normalized["edges"])
        build_parents_children(normalized, default_edge_data_mb=default_edge_data_mb)
        return normalized

    def _normalize_task(self, workflow_id, task):
        if "id" not in task:
            raise ValueError(f"Task in workflow {workflow_id} missing required field 'id'")

        # Keep trace-derived features (duration / plan_cpu / instance_num / raw_weight)
        # as-is. workload_cycles is NOT set here; the WorkloadMapper layer assigns it
        # from the relative weight. Legacy length fields are intentionally ignored.
        normalized = copy.deepcopy(task)
        normalized["id"] = int(task["id"])
        return normalized

    def _normalize_edge(self, workflow_id, edge, default_edge_data_mb=None):
        if "src" not in edge or "dst" not in edge:
            raise ValueError(f"Edge in workflow {workflow_id} missing 'src' or 'dst'")
        normalized = {"src": int(edge["src"]), "dst": int(edge["dst"])}
        if "data_size_mb" in edge:
            normalized["data_size_mb"] = float(edge["data_size_mb"])
        elif default_edge_data_mb is not None:
            normalized["data_size_mb"] = float(default_edge_data_mb)
        return normalized
