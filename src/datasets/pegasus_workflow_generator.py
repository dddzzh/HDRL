from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

from src.datasets.arrival_process import resolve_arrival_rate
from src.simulator.deadline import validate_deadline_type
from src.simulator.workflow import DependencyEdge, Task, Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sec_job_reference"

# Canonical source templates: small, medium, then large groups.
PEGASUS_XML_FILES = (
    "CyberShake_30.xml", "Epigenomics_24.xml", "Ligo_30.xml", "Montage_25.xml", "Sipht_29.xml",
    "CyberShake_50.xml", "Epigenomics_47.xml", "Ligo_50.xml", "Montage_50.xml", "Sipht_58.xml",
    "CyberShake_100.xml", "Epigenomics_100.xml", "Ligo_100.xml", "Montage_100.xml", "Sipht_97.xml",
)
DATASET_SCALE_LIMITS = {"small": 5, "middle": 10, "large": 15}

# These are the physical ranges already used by this project.  They are kept
# as model constants instead of repeated generator configuration fields.
WORKLOAD_GC_MIN = 0.5
WORKLOAD_GC_MAX = 20.0
DATA_SIZE_MB_MIN = 1.0
DATA_SIZE_MB_MAX = 30.0


class PegasusWorkflowGenerator:
    """Physical XML parser and compatibility entry for the current sec-job protocol."""

    def __init__(self, config: Optional[dict[str, Any]] = None, seed: int = 42):
        config = config or {}
        self.config = dict(config)
        self.seed = int(seed)
        self.num_workflows = int(config.get("num_workflows", 50))
        self.dataset_scale = str(
            config.get("dataset_scale", config.get("datasetScale", "small"))
        ).lower()
        self.data_dir = self._resolve_data_dir(config.get("path", DEFAULT_DATA_DIR))
        self.arrival_rate = resolve_arrival_rate(config, default=0.03)
        self.arrival_process = str(
            config.get("arrival_process", "sec_job_exponential")
        )
        self.start_time = float(config.get("start_time", 0.0))
        self.reliability_target = self._resolve_reliability_requirement(config)
        self.deadline_type = validate_deadline_type(config.get("deadline_type", "medium"))
        self.energy_eta = float(config.get("energy_eta", 0.5))

    def generate(
        self,
        servers: Optional[list[dict[str, Any]]] = None,
        link_bandwidth_mbps: Optional[dict[str, float]] = None,
    ) -> list[Workflow]:
        # Keep existing callers, but never silently run the retired generator.
        from src.datasets.sec_job_workflow_generator import SecJobWorkflowGenerator
        return SecJobWorkflowGenerator(self.config, self.seed).generate(
            servers=servers, link_bandwidth_mbps=link_bandwidth_mbps,
        )

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
