import json
import unittest
from pathlib import Path

from src.datasets.pegasus_workflow_generator import (
    DATASET_SCALE_LIMITS,
    DATA_SIZE_MB_MAX,
    DATA_SIZE_MB_MIN,
    PEGASUS_XML_FILES,
    WORKLOAD_GC_MAX,
    WORKLOAD_GC_MIN,
    PegasusWorkflowGenerator,
)
from tools.hierarchical_common import build_resources, build_workflows


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PegasusXmlGeneratorTest(unittest.TestCase):
    def test_root_data_contains_reference_templates(self):
        for name in PEGASUS_XML_FILES:
            self.assertTrue((PROJECT_ROOT / "data" / name).is_file(), name)

    def test_xml_topology_is_loaded_instead_of_random_dag(self):
        path = PROJECT_ROOT / "data" / "Montage_100.xml"
        task_ids, edges = PegasusWorkflowGenerator._read_xml_topology(path)
        self.assertEqual(len(task_ids), 100)
        self.assertGreater(len(edges), 0)
        self.assertTrue(all(parent in task_ids and child in task_ids for parent, child in edges))

    def test_small_scale_matches_reference_cumulative_group(self):
        config = {
            "path": "data",
            "num_workflows": 6,
            "dataset_scale": "small",
            "arrival_process": "cai_2025_exponential",
            "arrival_rate": 0.05,
            "Rreq": 0.9,
        }
        workflows = PegasusWorkflowGenerator(config, seed=42).generate()
        allowed = set(PEGASUS_XML_FILES[: DATASET_SCALE_LIMITS["small"]])
        self.assertEqual(len(workflows), 6)
        self.assertTrue(all(workflow.template_name in allowed for workflow in workflows))
        self.assertTrue(
            all(workflows[index].arrival_time <= workflows[index + 1].arrival_time for index in range(5))
        )
        self.assertEqual(sum(workflow.arrival_time == 0.0 for workflow in workflows), 1)
        self.assertTrue(all(workflow.source_device_id == 0 for workflow in workflows))
        self.assertTrue(
            all(workflow.arrival_process == "cai_2025_exponential" for workflow in workflows)
        )
        self.assertTrue(all(workflow.reliability == 0.9 for workflow in workflows))

        for workflow in workflows:
            self.assertEqual(workflow.input_data_mb, 0.0)
            self.assertEqual(workflow.output_data_mb, 0.0)
            for task in workflow.tasks:
                workload_gc = task.workload_cycles / 1.0e9
                self.assertGreaterEqual(workload_gc, WORKLOAD_GC_MIN)
                self.assertLessEqual(workload_gc, WORKLOAD_GC_MAX)
            for edge in workflow.edges:
                self.assertGreaterEqual(edge.data_size_mb, DATA_SIZE_MB_MIN)
                self.assertLessEqual(edge.data_size_mb, DATA_SIZE_MB_MAX)

    def test_same_seed_reproduces_templates_and_physical_values(self):
        config = {
            "path": "data",
            "num_workflows": 3,
            "dataset_scale": "large",
            "arrival_process": "cai_2025_exponential",
            "arrival_rate": 0.05,
        }
        first = PegasusWorkflowGenerator(config, seed=7).generate()
        second = PegasusWorkflowGenerator(config, seed=7).generate()
        self.assertEqual(
            [(workflow.template_name, workflow.arrival_time) for workflow in first],
            [(workflow.template_name, workflow.arrival_time) for workflow in second],
        )
        self.assertEqual(
            [task.workload_cycles for workflow in first for task in workflow.tasks],
            [task.workload_cycles for workflow in second for task in workflow.tasks],
        )

    def test_hierarchical_config_is_compact_and_builds_deadlines(self):
        config_path = PROJECT_ROOT / "configs" / "hierarchical_d3qn_sac.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(config["generator"]),
            {"num_workflows", "dataset_scale", "arrival_process", "arrival_rate"},
        )
        self.assertEqual(config["generator"]["arrival_process"], "cai_2025_exponential")
        self.assertNotIn("dataset", config)
        self.assertEqual(config["simulation"]["deadline_type"], "medium")
        self.assertEqual(config["simulation"]["workflow_reliability_requirement"], 0.9)

        smoke_config = json.loads(config_path.read_text(encoding="utf-8"))
        smoke_config["generator"]["num_workflows"] = 2
        smoke_config["generator"]["dataset_scale"] = "small"
        resources = build_resources(smoke_config, seed=11)
        workflows = build_workflows(smoke_config, resources, seed=11)
        self.assertEqual(len(workflows), 2)
        for workflow in workflows:
            self.assertIsNotNone(workflow.absolute_deadline)
            self.assertGreater(workflow.absolute_deadline, workflow.arrival_time)
            self.assertEqual(workflow.reliability, 0.9)


if __name__ == "__main__":
    unittest.main()
