"""Pre-sec-job sampling and isolated HEFT deadlines; reuse the physical XML parser."""
import random

from src.datasets.arrival_process import sample_configured_arrivals
from src.datasets.pegasus_workflow_generator import PegasusWorkflowGenerator
from src.simulator.deadline import cai_2025_deadline_fields


TEMPLATES = (
    "CyberShake_30.xml", "Epigenomics_24.xml", "Inspiral_30.xml", "Montage_25.xml", "Sipht_30.xml",
    "CyberShake_50.xml", "Epigenomics_46.xml", "Inspiral_50.xml", "Montage_50.xml", "Sipht_60.xml",
    "CyberShake_100.xml", "Epigenomics_100.xml", "Inspiral_100.xml", "Montage_100.xml", "Sipht_100.xml",
    "CyberShake_1000.xml", "Epigenomics_997.xml", "Inspiral_1000.xml", "Montage_1000.xml", "Sipht_1000.xml",
)
SCALE_LIMITS = {"small": 5, "middle": 10, "large": 15, "huge": 20}


class LegacyWorkflowGenerator(PegasusWorkflowGenerator):
    def __init__(self, config=None, seed=42):
        config = dict(config or {})
        config.setdefault("path", "data")
        config.setdefault("num_workflows", 20)
        config.setdefault("arrival_rate", 0.01)
        config.setdefault("arrival_process", "cai_2025_exponential")
        super().__init__(config, seed)
        self.deadline_heft_rho = float(config.get("deadline_heft_rho", config.get("heft_rho", 1.0)))

    def generate(self, servers=None, link_bandwidth_mbps=None):
        if self.num_workflows <= 0:
            raise ValueError("num_workflows must be positive")
        if self.dataset_scale not in SCALE_LIMITS:
            raise ValueError(f"Invalid legacy dataset_scale: {self.dataset_scale!r}")
        rng = random.Random(self.seed)
        arrivals, intervals, devices, process = sample_configured_arrivals(
            self.num_workflows, self.config, self.start_time, rng,
        )
        templates = TEMPLATES[:SCALE_LIMITS[self.dataset_scale]]
        cache = {}
        workflows = []
        for index, arrival in enumerate(arrivals):
            name = rng.choice(templates)
            workflow = self._workflow_from_xml(
                self.data_dir / name, index=index, arrival_time=arrival, rng=rng,
            )
            workflow.arrival_index = index
            workflow.inter_arrival_time = intervals[index]
            workflow.arrival_process = process
            workflow.arrival_rate = self.arrival_rate
            workflow.source_device_id = devices[index]
            workflow.dataset_scale = self.dataset_scale
            workflow.template_name = name
            if servers is not None:
                info = cai_2025_deadline_fields(
                    workflow, arrival, servers, deadline_type=self.deadline_type,
                    rng=rng, link_bandwidth_mbps=link_bandwidth_mbps,
                    rho=self.deadline_heft_rho, cache=cache,
                )
                self._set_deadline_fields(workflow, info)
                self._log_deadline(workflow, info)
            workflows.append(workflow)
        return workflows
