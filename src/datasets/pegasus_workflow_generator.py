from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.datasets.arrival_process import (
    DEFAULT_POISSON_ARRIVAL_RATE,
    resolve_arrival_rate,
    sample_configured_arrivals,
)
from src.simulator.deadline import cai_2025_deadline_fields, validate_deadline_type
from src.simulator.workflow import DependencyEdge, Task, Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JUVE_ROOT = PROJECT_ROOT / "third_party" / "WorkflowGenerator-master" / "juve"


class PegasusWorkflowGenerator:
    """Adapter around Pegasus WorkflowGenerator/Juve random workflow models."""

    def __init__(self, config: Optional[dict[str, Any]] = None, seed: int = 42):
        config = config or {}
        self.config = dict(config)
        self.seed = int(seed)
        self.np_rng = np.random.default_rng(seed)

        self.num_workflows = int(config.get("num_workflows", 100))
        self.min_tasks = int(config.get("min_tasks", config.get("num_tasks", 10)))
        self.max_tasks = int(config.get("max_tasks", self.min_tasks))
        self.min_levels = int(config.get("min_levels", config.get("num_levels", 3)))
        self.max_levels = int(config.get("max_levels", self.min_levels))

        self.arrival_rate = resolve_arrival_rate(config, default=DEFAULT_POISSON_ARRIVAL_RATE)
        self.arrival_process = "multi_device_poisson"
        self.num_mobile_devices = int(
            config.get("num_mobile_devices", config.get("num_devices", config.get("mobile_devices", 1)))
        )
        self.start_time = float(config.get("start_time", 0.0))

        self.task_length_cycles_min = float(config.get("task_length_cycles_min", 1.0e8))
        self.task_length_cycles_max = float(config.get("task_length_cycles_max", 15.0e8))
        self.task_size_kb_min = float(config.get("task_size_kb_min", 100.0))
        self.task_size_kb_max = float(config.get("task_size_kb_max", 1000.0))
        self.runtime_to_cycles_hz = float(config.get("runtime_to_cycles_hz", 3.0e9))
        self.edge_data_mb_default = float(config.get("edge_data_mb", self.task_size_kb_min / 1024.0))
        self.input_data_mb = float(config.get("input_data_mb", self.edge_data_mb_default))
        self.output_data_mb = float(config.get("output_data_mb", self.edge_data_mb_default))
        self.energy_eta = float(config.get("energy_eta", 0.5))
        self.energy_min_factor = float(config.get("energy_min_factor", 2.0))
        self.energy_max_factor = float(config.get("energy_max_factor", 20.0))
        self.reliability_target = self._resolve_reliability_requirement(config)
        self.deadline_type = validate_deadline_type(config.get("deadline_type", "medium"))
        self.deadline_heft_rho = float(config.get("deadline_heft_rho", config.get("heft_rho", 1.0)))
        self.deadline_rng = random.Random(seed)

    def generate(
        self,
        servers: Optional[list[dict[str, Any]]] = None,
        link_bandwidth_mbps: Optional[dict[str, float]] = None,
    ) -> list[Workflow]:
        if self.num_workflows <= 0:
            raise ValueError("num_workflows must be positive")
        if self.min_tasks <= 0 or self.max_tasks < self.min_tasks:
            raise ValueError("invalid Pegasus task count range")
        if self.min_levels <= 1 or self.max_levels < self.min_levels:
            raise ValueError("Pegasus randlevel requires at least two levels")

        arrival_times, inter_arrival_times, source_device_ids, arrival_process = sample_configured_arrivals(
            self.num_workflows,
            self.config,
            self.start_time,
            random.Random(self.seed),
        )
        workflows = []
        random_state = random.getstate()
        try:
            random.seed(self.seed)
            reference_makespan_cache = {}
            for index in range(self.num_workflows):
                task_count = random.randint(self.min_tasks, self.max_tasks)
                level_count = min(task_count, random.randint(self.min_levels, self.max_levels))
                pegasus_workflow = self._generate_juve_randlevel(task_count, level_count)
                workflow = self._convert_workflow(pegasus_workflow, index, arrival_times[index])
                workflow.arrival_index = index
                workflow.inter_arrival_time = inter_arrival_times[index]
                workflow.arrival_process = arrival_process
                workflow.arrival_rate = self.arrival_rate
                workflow.source_device_id = source_device_ids[index]
                if servers is not None:
                    deadline_info = cai_2025_deadline_fields(
                        workflow,
                        workflow.arrival_time,
                        servers,
                        deadline_type=self.deadline_type,
                        rng=self.deadline_rng,
                        link_bandwidth_mbps=link_bandwidth_mbps,
                        rho=self.deadline_heft_rho,
                        cache=reference_makespan_cache,
                    )
                    self._set_deadline_fields(workflow, deadline_info)
                    self._log_deadline(workflow, deadline_info)

                workflows.append(workflow)
        finally:
            random.setstate(random_state)
        return workflows

    def _generate_juve_randlevel(self, task_count: int, level_count: int):
        self._ensure_juve_path()
        from generator import UniformDistribution
        from synthetic.randlevel import randlevel

        return randlevel(
            task_count,
            level_count,
            UniformDistribution(1.0, 1.0),
            UniformDistribution(self.task_size_kb_min / (1024.0 * 1024.0), self.task_size_kb_max / (1024.0 * 1024.0)),
        )

    def _convert_workflow(self, pegasus_workflow, index: int, arrival_time: float) -> Workflow:
        pegasus_workflow._computeDataDependencies()
        jobs = sorted(pegasus_workflow.jobs, key=lambda job: self._job_sort_key(job.id))
        id_map = {job.id: idx for idx, job in enumerate(jobs)}

        tasks = tuple(
            self._convert_task(job, id_map[job.id])
            for job in jobs
        )
        edges = []
        for child in jobs:
            child_id = id_map[child.id]
            for parent in sorted(child.parents, key=lambda job: self._job_sort_key(job.id)):
                parent_id = id_map[parent.id]
                edges.append(
                    DependencyEdge(
                        src=parent_id,
                        dst=child_id,
                        data_size_mb=self._edge_data_mb(parent, child),
                    )
                )

        total_cycles = sum(task.workload_cycles for task in tasks)
        min_makespan = total_cycles / max(self.runtime_to_cycles_hz, 1.0)
        e_min = self.energy_min_factor * min_makespan
        e_max = self.energy_max_factor * min_makespan
        energy_budget = e_min + self.energy_eta * (e_max - e_min)
        return Workflow(
            workflow_id=f"pegasus_{index}",
            tasks=tasks,
            edges=tuple(edges),
            arrival_time=float(arrival_time),
            reliability=self.reliability_target,
            energy_budget=energy_budget,
            input_data_mb=self.input_data_mb,
            output_data_mb=self.output_data_mb,
        )

    def _convert_task(self, job, task_id: int) -> Task:
        workload_cycles = float(
            self.np_rng.uniform(self.task_length_cycles_min, self.task_length_cycles_max)
        )
        return Task(
            id=task_id,
            workload_cycles=workload_cycles,
        )

    def _edge_data_mb(self, parent, child) -> float:
        shared_files = parent.outputs.intersection(child.inputs)
        total_bytes = sum(float(file_.size) for file_ in shared_files)
        if total_bytes <= 0:
            return self.edge_data_mb_default
        return total_bytes / (1024.0 * 1024.0)

    @staticmethod
    def _job_sort_key(job_id):
        text = str(job_id)
        suffix = text.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else text

    @staticmethod
    def _ensure_juve_path():
        juve_path = str(JUVE_ROOT)
        if juve_path not in sys.path:
            sys.path.insert(0, juve_path)

    def _set_deadline_fields(self, workflow: Workflow, deadline_info: dict[str, Any]) -> None:
        workflow.reference_makespan = float(deadline_info["reference_makespan"])
        workflow.deadline_multiplier = float(deadline_info["deadline_multiplier"])
        workflow.relative_deadline = float(deadline_info["relative_deadline"])
        workflow.absolute_deadline = float(deadline_info["absolute_deadline"])
        workflow.deadline_type = str(deadline_info["deadline_type"])

    def _log_deadline(self, workflow: Workflow, deadline_info: dict[str, Any]) -> None:
        if not bool(self.config.get("log_deadline_generation", self.config.get("trace_deadline_generation", False))):
            return
        print(
            "DEADLINE "
            f"workflow_id={workflow.workflow_id} "
            "workflow_type=pegasus "
            f"arrival_time={workflow.arrival_time:.6f} "
            f"reference_makespan={float(deadline_info['reference_makespan']):.6f} "
            f"deadline_type={deadline_info['deadline_type']} "
            f"deadline_multiplier={float(deadline_info['deadline_multiplier']):.1f} "
            f"relative_deadline={float(deadline_info['relative_deadline']):.6f} "
            f"absolute_deadline={float(deadline_info['absolute_deadline']):.6f}"
        )

    @staticmethod
    def _resolve_reliability_requirement(config: dict[str, Any]) -> float:
        for key in ("Rreq", "rreq", "reliability_req", "reliability_target", "default_reliability"):
            if key in config and config[key] is not None:
                return float(config[key])
        return 0.9
