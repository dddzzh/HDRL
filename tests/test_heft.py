import copy
import random
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schedulers.heft import (
    HEFTScheduler,
    calculate_upward_ranks,
    reference_heft_makespan,
    run_heft,
)
from src.simulator.deadline import cai_2025_deadline_fields, sample_deadline_multiplier
from src.simulator.workflow import DependencyEdge, Task, Workflow


def servers():
    return [
        {
            "id": 0,
            "type": "edge",
            "layer": "edge",
            "max_freq_hz": 10.0,
            "bandwidth_mbps": 10.0,
            "min_rho": 1.0,
        },
        {
            "id": 1,
            "type": "cloud",
            "layer": "cloud",
            "max_freq_hz": 5.0,
            "bandwidth_mbps": 20.0,
            "min_rho": 1.0,
        },
    ]


def links():
    return {
        "edge_cloud": 5.0,
        "edge_edge": 100.0,
        "cloud_cloud": 100.0,
    }


class HEFTTests(unittest.TestCase):
    def test_upward_rank_ignores_workflow_ingress_and_egress(self):
        workflow = Workflow(
            "wf",
            (Task(0, 10.0),),
            (),
            0.0,
            input_data_mb=1.0,
            output_data_mb=2.0,
        )

        ranks = calculate_upward_ranks(workflow, servers(), links())

        expected = (10.0 / 10.0 + 10.0 / 5.0) / 2.0
        self.assertAlmostEqual(ranks[0], expected)
        self.assertAlmostEqual(workflow.get_task(0).upward_rank, expected)

    def test_upward_rank_is_own_cost_plus_max_successor_rank(self):
        workflow = Workflow(
            "wf",
            (Task(0, 10.0), Task(1, 10.0), Task(2, 100.0)),
            (DependencyEdge(0, 1, 1.0), DependencyEdge(0, 2, 1.0)),
            0.0,
            input_data_mb=0.0,
            output_data_mb=0.0,
        )

        ranks = calculate_upward_ranks(workflow, servers(), links())

        parent_compute = (10.0 / 10.0 + 10.0 / 5.0) / 2.0
        mean_edge_cost = 8.0 / links()["edge_cloud"]
        self.assertAlmostEqual(
            ranks[0],
            parent_compute + mean_edge_cost + max(ranks[1], ranks[2]),
        )
        self.assertGreater(ranks[0], ranks[2])
        self.assertGreater(ranks[2], ranks[1])

    def test_run_heft_reports_reference_port_and_queue_tail_schedule(self):
        workflow = Workflow(
            "wf",
            (Task(0, 10.0), Task(1, 20.0), Task(2, 10.0)),
            (DependencyEdge(0, 2, 1.0), DependencyEdge(1, 2, 1.0)),
            0.0,
            input_data_mb=0.0,
            output_data_mb=0.0,
        )

        result = run_heft(workflow, servers(), links())

        self.assertEqual(result["algorithm"], "heft_dag_edge_cloud_eft")
        self.assertEqual(result["source"], "edge_cloud_dag_model")
        self.assertEqual(result["task_order"][-1], 2)
        for timeline in result["server_timelines"].values():
            for previous, current in zip(timeline, timeline[1:]):
                self.assertGreaterEqual(current["start"], previous["finish"] - 1e-9)

    def test_join_waits_for_each_parent_and_only_cross_server_transfer(self):
        workflow = Workflow(
            "wf",
            (Task(0, 100.0), Task(1, 100.0), Task(2, 1.0)),
            (DependencyEdge(0, 2, 1.0), DependencyEdge(1, 2, 1.0)),
            0.0,
            input_data_mb=0.0,
            output_data_mb=0.0,
        )

        result = run_heft(workflow, servers(), links())

        child_server = result["task_server"][2]
        parent_ready_times = []
        for parent_id in (0, 1):
            transfer = 0.0
            if result["task_server"][parent_id] != child_server:
                transfer = 8.0 / links()["edge_cloud"]
            parent_ready_times.append(result["task_aft"][parent_id] + transfer)
        expected = max(parent_ready_times)
        self.assertAlmostEqual(result["task_data_ready_time"][2], expected)
        self.assertNotEqual(result["task_server"][0], result["task_server"][1])

    def test_entry_task_has_no_upload_and_exit_has_no_download(self):
        workflow = Workflow(
            "wf",
            (Task(0, 10.0),),
            (),
            0.0,
            input_data_mb=1.0,
            output_data_mb=2.0,
        )

        result = run_heft(workflow, servers(), links())

        self.assertEqual(result["task_server"][0], 0)
        self.assertEqual(result["task_data_ready_time"][0], 0.0)
        self.assertAlmostEqual(result["task_execution_time"][0], 1.0)
        self.assertEqual(result["download_time"], 0.0)
        self.assertAlmostEqual(result["reference_makespan"], 1.0)

    def test_scheduler_selects_highest_rank_ready_task_then_online_eft(self):
        workflow = Workflow(
            "wf",
            (Task(0, 10.0), Task(1, 20.0)),
            (),
            0.0,
            input_data_mb=0.0,
            output_data_mb=0.0,
        )
        state = {
            "time": 0.0,
            "ready_tasks": [
                {"workflow_id": "wf", "task_id": 0, "workload_cycles": 10.0},
                {"workflow_id": "wf", "task_id": 1, "workload_cycles": 20.0},
            ],
            "workflows": {"wf": workflow},
            "servers": servers(),
            "link_bandwidth_mbps": links(),
        }
        requested = []

        def build_context(workflow_id, task_id):
            requested.append((workflow_id, task_id))
            return {
                "workflow_id": workflow_id,
                "task_id": task_id,
                "server_safety_metrics": {
                    "0": {"estimated_ast": 100.0, "execution_time_rho_1": 1.0},
                    "1": {"estimated_ast": 0.0, "execution_time_rho_1": 4.0},
                },
            }

        scheduler = HEFTScheduler(params={"rho": 1.0}, seed=1)
        action = scheduler.select_action(state, context_builder=build_context)

        self.assertEqual(requested, [("wf", 1)])
        self.assertEqual(action["task_id"], 1)
        self.assertEqual(action["server_id"], 1)
        self.assertTrue(action["online_queue_aware"])
        self.assertEqual(action["algorithm"], "heft_dag_edge_cloud_eft")
        self.assertFalse(hasattr(scheduler, "_plans"))

    def test_cache_hit_returns_copy_and_writes_ranks(self):
        cache = {}
        workflow_a = Workflow(
            "wf", (Task(0, 10.0),), (), 0.0, input_data_mb=0.0, output_data_mb=0.0
        )
        first = run_heft(workflow_a, servers(), links(), cache=cache)
        workflow_b = copy.deepcopy(workflow_a)
        workflow_b.get_task(0).upward_rank = None
        second = run_heft(workflow_b, servers(), links(), cache=cache)
        second["task_order"].append(99)
        third = run_heft(workflow_b, servers(), links(), cache=cache)
        self.assertEqual(first["reference_makespan"], third["reference_makespan"])
        self.assertNotIn(99, third["task_order"])
        self.assertIsNotNone(workflow_b.get_task(0).upward_rank)

    def test_deadline_fields_use_reference_makespan(self):
        workflow = Workflow(
            "wf", (Task(0, 10.0),), (), 7.0, input_data_mb=0.0, output_data_mb=0.0
        )
        info = cai_2025_deadline_fields(
            workflow,
            workflow.arrival_time,
            servers(),
            deadline_type="medium",
            rng=random.Random(1),
            link_bandwidth_mbps=links(),
        )
        self.assertEqual(
            info["relative_deadline"],
            info["deadline_multiplier"] * info["reference_makespan"],
        )
        self.assertEqual(
            info["absolute_deadline"],
            workflow.arrival_time + info["relative_deadline"],
        )

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
        self.assertEqual(
            reference_heft_makespan(workflow_a, servers(), links()),
            first["reference_makespan"],
        )


if __name__ == "__main__":
    unittest.main()
