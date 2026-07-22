import copy
import random
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schedulers.heft import earliest_idle_slot, reference_heft_makespan, run_heft
from src.simulator.deadline import cai_2025_deadline_fields, sample_deadline_multiplier
from src.simulator.workflow import DependencyEdge, Task, Workflow


def servers():
    return [
        {"id": 0, "type": "edge", "layer": "edge", "max_freq_hz": 10.0, "min_rho": 1.0},
        {"id": 1, "type": "cloud", "layer": "cloud", "max_freq_hz": 5.0, "min_rho": 1.0},
    ]


def links():
    return {
        "iot_edge": 10.0,
        "edge_iot": 10.0,
        "iot_cloud": 20.0,
        "cloud_iot": 20.0,
        "edge_cloud": 5.0,
        "edge_edge": 100.0,
        "cloud_cloud": 100.0,
    }


class HEFTTests(unittest.TestCase):
    def test_single_task_rank_equals_average_execution_time(self):
        workflow = Workflow("wf", (Task(0, 10.0),), (), 0.0, input_data_mb=0.0, output_data_mb=0.0)
        result = run_heft(workflow, servers(), links())
        self.assertAlmostEqual(result["upward_ranks"][0], 1.5)
        self.assertAlmostEqual(workflow.get_task(0).upward_rank, 1.5)

    def test_chain_ranks_descend(self):
        workflow = Workflow(
            "wf",
            (Task(0, 10.0), Task(1, 10.0), Task(2, 10.0)),
            (DependencyEdge(0, 1, 1.0), DependencyEdge(1, 2, 1.0)),
            0.0,
            input_data_mb=0.0,
            output_data_mb=0.0,
        )
        result = run_heft(workflow, servers(), links())
        ranks = result["upward_ranks"]
        self.assertGreater(ranks[0], ranks[1])
        self.assertGreater(ranks[1], ranks[2])

    def test_fork_rank_uses_max_child_path(self):
        workflow = Workflow(
            "wf",
            (Task(0, 10.0), Task(1, 10.0), Task(2, 100.0)),
            (DependencyEdge(0, 1, 1.0), DependencyEdge(0, 2, 1.0)),
            0.0,
            input_data_mb=0.0,
            output_data_mb=0.0,
        )
        result = run_heft(workflow, servers(), links())
        ranks = result["upward_ranks"]
        avg_exec_parent = 1.5
        avg_comm = 8.0 / 5.0
        self.assertAlmostEqual(ranks[0], avg_exec_parent + max(avg_comm + ranks[1], avg_comm + ranks[2]))

    def test_join_uses_max_parent_data_ready_time(self):
        workflow = Workflow(
            "wf",
            (Task(0, 10.0), Task(1, 20.0), Task(2, 10.0)),
            (DependencyEdge(0, 2, 1.0), DependencyEdge(1, 2, 1.0)),
            0.0,
            input_data_mb=0.0,
            output_data_mb=0.0,
        )
        result = run_heft(workflow, servers(), links())
        parent_ready = []
        for parent_id in (0, 1):
            transfer = 0.0
            if result["task_server"][parent_id] != result["task_server"][2]:
                transfer = 8.0 / 5.0
            parent_ready.append(result["task_aft"][parent_id] + transfer)
        self.assertAlmostEqual(result["task_data_ready_time"][2], max(parent_ready))

    def test_same_server_transfer_is_zero(self):
        workflow = Workflow(
            "wf",
            (Task(0, 10.0), Task(1, 10.0)),
            (DependencyEdge(0, 1, 100.0),),
            0.0,
            input_data_mb=0.0,
            output_data_mb=0.0,
        )
        one_server = [servers()[0]]
        result = run_heft(workflow, one_server, links())
        self.assertEqual(result["task_server"][0], result["task_server"][1])
        self.assertAlmostEqual(result["task_data_ready_time"][1], result["task_aft"][0])

    def test_different_server_transfer_uses_actual_link(self):
        workflow = Workflow(
            "wf",
            (Task(0, 1.0), Task(1, 100.0)),
            (DependencyEdge(0, 1, 5.0),),
            0.0,
            input_data_mb=0.0,
            output_data_mb=0.0,
        )
        result = run_heft(workflow, servers(), links())
        if result["task_server"][0] != result["task_server"][1]:
            self.assertAlmostEqual(
                result["task_data_ready_time"][1],
                result["task_aft"][0] + 8.0,
            )

    def test_entry_upload_and_final_download(self):
        workflow = Workflow("wf", (Task(0, 10.0),), (), 0.0, input_data_mb=5.0, output_data_mb=5.0)
        result = run_heft(workflow, servers(), links())
        server_id = result["task_server"][0]
        expected_upload = 4.0 if server_id == 0 else 2.0
        expected_download = 4.0 if server_id == 0 else 2.0
        self.assertAlmostEqual(result["task_data_ready_time"][0], expected_upload)
        self.assertAlmostEqual(result["download_time"], expected_download)
        self.assertAlmostEqual(
            result["reference_makespan"],
            result["compute_finish_time"] + result["download_time"],
        )

    def test_zero_output_has_no_download(self):
        workflow = Workflow("wf", (Task(0, 10.0),), (), 0.0, input_data_mb=0.0, output_data_mb=0.0)
        result = run_heft(workflow, servers(), links())
        self.assertEqual(result["download_time"], 0.0)

    def test_earliest_idle_slot_inserts_and_fills_gap(self):
        timeline = [
            {"task_id": 0, "start": 0.0, "finish": 2.0},
            {"task_id": 1, "start": 8.0, "finish": 10.0},
        ]
        self.assertEqual(earliest_idle_slot(timeline, 3.0, 2.0), 3.0)
        filled = [
            {"task_id": 0, "start": 0.0, "finish": 2.0},
            {"task_id": 1, "start": 5.0, "finish": 8.0},
        ]
        self.assertEqual(earliest_idle_slot(filled, 2.0, 3.0), 2.0)

    def test_timelines_do_not_overlap(self):
        workflow = Workflow(
            "wf",
            (Task(0, 10.0), Task(1, 10.0), Task(2, 10.0), Task(3, 10.0)),
            (DependencyEdge(0, 2, 1.0), DependencyEdge(1, 3, 1.0)),
            0.0,
            input_data_mb=0.0,
            output_data_mb=0.0,
        )
        result = run_heft(workflow, servers(), links())
        for timeline in result["server_timelines"].values():
            for prev, cur in zip(timeline, timeline[1:]):
                self.assertGreaterEqual(cur["start"], prev["finish"] - 1e-9)

    def test_upward_rank_serialization_and_deepcopy(self):
        workflow = Workflow("wf", (Task(0, 10.0),), (), 0.0, input_data_mb=0.0, output_data_mb=0.0)
        run_heft(workflow, servers(), links())
        payload = workflow.to_mapping()
        self.assertIn("upward_rank", payload["tasks"][0])
        loaded = Workflow.from_mapping(payload)
        self.assertEqual(loaded.get_task(0).upward_rank, workflow.get_task(0).upward_rank)
        cloned = copy.deepcopy(workflow)
        self.assertEqual(cloned.get_task(0).upward_rank, workflow.get_task(0).upward_rank)

    def test_cache_hit_returns_copy_and_writes_ranks(self):
        cache = {}
        workflow_a = Workflow("wf", (Task(0, 10.0),), (), 0.0, input_data_mb=0.0, output_data_mb=0.0)
        first = run_heft(workflow_a, servers(), links(), cache=cache)
        workflow_b = Workflow("wf", (Task(0, 10.0),), (), 0.0, input_data_mb=0.0, output_data_mb=0.0)
        second = run_heft(workflow_b, servers(), links(), cache=cache)
        second["task_order"].append(99)
        third = run_heft(workflow_b, servers(), links(), cache=cache)
        self.assertEqual(first["reference_makespan"], third["reference_makespan"])
        self.assertNotIn(99, third["task_order"])
        self.assertIsNotNone(workflow_b.get_task(0).upward_rank)

    def test_deadline_fields_use_reference_makespan(self):
        workflow = Workflow("wf", (Task(0, 10.0),), (), 7.0, input_data_mb=0.0, output_data_mb=0.0)
        info = cai_2025_deadline_fields(
            workflow,
            workflow.arrival_time,
            servers(),
            deadline_type="medium",
            rng=random.Random(1),
            link_bandwidth_mbps=links(),
        )
        self.assertEqual(info["relative_deadline"], info["deadline_multiplier"] * info["reference_makespan"])
        self.assertEqual(info["absolute_deadline"], workflow.arrival_time + info["relative_deadline"])

    def test_deadline_sampling_probabilities(self):
        for deadline_type, expected in {"tight": 0.8, "medium": 0.5, "loose": 0.2}.items():
            rng = random.Random(123)
            values = [sample_deadline_multiplier(deadline_type, rng) for _ in range(10000)]
            self.assertAlmostEqual(values.count(2.0) / len(values), expected, delta=0.03)

    def test_reproducible_heft_schedule(self):
        workflow_a = Workflow(
            "wf",
            (Task(0, 10.0), Task(1, 20.0)),
            (DependencyEdge(0, 1, 1.0),),
            0.0,
            input_data_mb=1.0,
            output_data_mb=1.0,
        )
        workflow_b = copy.deepcopy(workflow_a)
        first = run_heft(workflow_a, servers(), links())
        second = run_heft(workflow_b, servers(), links())
        for key in ("task_order", "task_server", "task_ast", "task_aft"):
            self.assertEqual(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
