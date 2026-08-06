from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

from src.datasets.arrival_process import (
    CAI_2025_ARRIVAL_PROCESS,
    DEFAULT_POISSON_ARRIVAL_RATE,
    resolve_arrival_rate,
    sample_configured_arrivals,
)
from src.simulator.deadline import cai_2025_deadline_fields, validate_deadline_type
from src.simulator.workflow import DependencyEdge, Task, Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# Keep the same XML grouping and cumulative sampling rule as
# D3QN_for_Fog_Computing(1.6)/CreateDataset.py.
PEGASUS_XML_FILES = (
    "CyberShake_30.xml",
    "Epigenomics_24.xml",
    "Inspiral_30.xml",
    "Montage_25.xml",
    "Sipht_30.xml",
    "CyberShake_50.xml",
    "Epigenomics_46.xml",
    "Inspiral_50.xml",
    "Montage_50.xml",
    "Sipht_60.xml",
    "CyberShake_100.xml",
    "Epigenomics_100.xml",
    "Inspiral_100.xml",
    "Montage_100.xml",
    "Sipht_100.xml",
    "CyberShake_1000.xml",
    "Epigenomics_997.xml",
    "Inspiral_1000.xml",
    "Montage_1000.xml",
    "Sipht_1000.xml",
)

DATASET_SCALE_LIMITS = {
    "small": 5,
    "middle": 10,
    "large": 15,
    "huge": 20,
}

# These are the physical ranges already used by this project.  They are kept
# as model constants instead of repeated generator configuration fields.
WORKLOAD_GC_MIN = 0.5
WORKLOAD_GC_MAX = 20.0
DATA_SIZE_MB_MIN = 1.0
DATA_SIZE_MB_MAX = 30.0


class PegasusWorkflowGenerator:
    """Sample real Pegasus XML DAG templates using the reference D3QN rule."""

    def __init__(self, config: Optional[dict[str, Any]] = None, seed: int = 42):
        config = config or {}
        self.config = dict(config)
        self.seed = int(seed)
        self.num_workflows = int(config.get("num_workflows", 20))
        self.dataset_scale = str(
            config.get("dataset_scale", config.get("datasetScale", "small"))
        ).lower()
        self.data_dir = self._resolve_data_dir(config.get("path", "data"))
        self.arrival_rate = resolve_arrival_rate(config, default=DEFAULT_POISSON_ARRIVAL_RATE)
        self.arrival_process = str(
            config.get("arrival_process", CAI_2025_ARRIVAL_PROCESS)
        )
        self.start_time = float(config.get("start_time", 0.0))
        self.reliability_target = self._resolve_reliability_requirement(config)
        self.deadline_type = validate_deadline_type(config.get("deadline_type", "medium"))
        self.deadline_heft_rho = float(config.get("deadline_heft_rho", config.get("heft_rho", 1.0)))
        self.energy_eta = float(config.get("energy_eta", 0.5))

    def generate(
        self,
        servers: Optional[list[dict[str, Any]]] = None,
        link_bandwidth_mbps: Optional[dict[str, float]] = None,
    ) -> list[Workflow]:
        if self.num_workflows <= 0:
            raise ValueError("num_workflows must be positive")
        if self.dataset_scale not in DATASET_SCALE_LIMITS:
            raise ValueError(
                f"invalid dataset_scale={self.dataset_scale!r}; expected one of "
                f"{sorted(DATASET_SCALE_LIMITS)}"
            )

        rng = random.Random(self.seed)
        arrival_times, inter_arrival_times, source_device_ids, arrival_process = (
            sample_configured_arrivals(
                self.num_workflows,
                self.config,
                self.start_time,
                rng,
            )
        )
        template_names = PEGASUS_XML_FILES[: DATASET_SCALE_LIMITS[self.dataset_scale]]
        reference_makespan_cache: dict[str, dict[str, Any]] = {}
        workflows = []
        for index in range(self.num_workflows):
            template_name = rng.choice(template_names)
            workflow = self._workflow_from_xml(
                self.data_dir / template_name,
                index=index,
                arrival_time=arrival_times[index],
                rng=rng,
            )
            workflow.arrival_index = index
            workflow.inter_arrival_time = inter_arrival_times[index]
            workflow.arrival_process = arrival_process
            workflow.arrival_rate = self.arrival_rate
            workflow.source_device_id = source_device_ids[index]
            workflow.dataset_scale = self.dataset_scale
            workflow.template_name = template_name

            if servers is not None:
                deadline_info = cai_2025_deadline_fields(
                    workflow,
                    workflow.arrival_time,
                    servers,
                    deadline_type=self.deadline_type,
                    rng=rng,
                    link_bandwidth_mbps=link_bandwidth_mbps,
                    rho=self.deadline_heft_rho,
                    cache=reference_makespan_cache,
                )
                self._set_deadline_fields(workflow, deadline_info)
                self._log_deadline(workflow, deadline_info)
            workflows.append(workflow)
        return workflows

    def _workflow_from_xml(
        self,
        path: Path,
        *,
        index: int,
        arrival_time: float,
        rng: random.Random,
    ) -> Workflow:
        task_ids, edge_pairs = self._read_xml_topology(path)
        id_map = {xml_id: task_id for task_id, xml_id in enumerate(task_ids)}
        tasks = tuple(
            Task(
                id=id_map[xml_id],
                workload_cycles=rng.uniform(WORKLOAD_GC_MIN, WORKLOAD_GC_MAX) * 1.0e9,
            )
            for xml_id in task_ids
        )
        edges = tuple(
            DependencyEdge(
                src=id_map[parent_id],
                dst=id_map[child_id],
                data_size_mb=rng.uniform(DATA_SIZE_MB_MIN, DATA_SIZE_MB_MAX),
            )
            for parent_id, child_id in edge_pairs
        )

        total_cycles = sum(task.workload_cycles for task in tasks)
        max_reference_frequency_hz = 3.0e9
        min_makespan = total_cycles / max_reference_frequency_hz
        e_min = 2.0 * min_makespan
        e_max = 20.0 * min_makespan
        energy_budget = e_min + self.energy_eta * (e_max - e_min)
        return Workflow(
            workflow_id=f"pegasus_{index}",
            tasks=tasks,
            edges=edges,
            arrival_time=float(arrival_time),
            reliability=self.reliability_target,
            energy_budget=energy_budget,
            # IoT devices generate workflow arrivals only. Initial ingestion
            # and final result delivery are outside the scheduling boundary.
            input_data_mb=0.0,
            output_data_mb=0.0,
        )

    @staticmethod
    def _read_xml_topology(path: Path) -> tuple[list[str], list[tuple[str, str]]]:
        if not path.is_file():
            raise FileNotFoundError(f"Pegasus XML template not found: {path}")
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Invalid Pegasus XML template: {path}") from exc

        namespace = "{http://pegasus.isi.edu/schema/DAX}"
        job_nodes = root.findall(f"{namespace}job")
        task_ids = [str(node.get("id")) for node in job_nodes if node.get("id") is not None]
        task_id_set = set(task_ids)
        if not task_ids:
            raise ValueError(f"Pegasus XML template contains no jobs: {path}")
        if len(task_id_set) != len(task_ids):
            raise ValueError(f"Pegasus XML template contains duplicate job ids: {path}")

        edge_pairs: list[tuple[str, str]] = []
        seen_edges = set()
        for child_node in root.findall(f"{namespace}child"):
            child_id = child_node.get("ref")
            if child_id not in task_id_set:
                continue
            for parent_node in child_node.findall(f"{namespace}parent"):
                parent_id = parent_node.get("ref")
                edge = (str(parent_id), str(child_id))
                if parent_id in task_id_set and edge not in seen_edges:
                    edge_pairs.append(edge)
                    seen_edges.add(edge)
        return task_ids, edge_pairs

    @staticmethod
    def _resolve_data_dir(value: Any) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @staticmethod
    def _set_deadline_fields(workflow: Workflow, deadline_info: dict[str, Any]) -> None:
        workflow.reference_makespan = float(deadline_info["reference_makespan"])
        workflow.deadline_multiplier = float(deadline_info["deadline_multiplier"])
        workflow.relative_deadline = float(deadline_info["relative_deadline"])
        workflow.absolute_deadline = float(deadline_info["absolute_deadline"])
        workflow.deadline_type = str(deadline_info["deadline_type"])

    def _log_deadline(self, workflow: Workflow, deadline_info: dict[str, Any]) -> None:
        if not bool(self.config.get("log_deadline_generation", False)):
            return
        print(
            "DEADLINE "
            f"workflow_id={workflow.workflow_id} "
            f"template={workflow.template_name} "
            f"arrival_time={workflow.arrival_time:.6f} "
            f"reference_makespan={float(deadline_info['reference_makespan']):.6f} "
            f"deadline_type={deadline_info['deadline_type']} "
            f"deadline_multiplier={float(deadline_info['deadline_multiplier']):.1f} "
            f"relative_deadline={float(deadline_info['relative_deadline']):.6f} "
            f"absolute_deadline={float(deadline_info['absolute_deadline']):.6f}"
        )

    @staticmethod
    def _resolve_reliability_requirement(config: dict[str, Any]) -> float:
        for key in (
            "workflow_reliability_requirement",
            "Rreq",
            "rreq",
            "reliability_req",
            "reliability_target",
        ):
            if key in config and config[key] is not None:
                return float(config[key])
        return 0.9
