import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator.sub_deadline import distribute_existing_workflow_deadline
from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.shared.hierarchical_state_reward import normalized_sub_deadline_violation
from src.simulator.workflow import DependencyEdge, Task, Workflow
from tests.test_time_model import resources as runtime_resources


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
    return {"edge_cloud": 5.0, "edge_edge": 100.0, "cloud_cloud": 100.0}


class SubDeadlineTests(unittest.TestCase):
    def test_existing_deadline_scales_heft_finish_times_without_replacing_it(self):
        workflow = Workflow(
            "wf",
            (Task(0, 10.0), Task(1, 20.0), Task(2, 10.0)),
            (DependencyEdge(0, 1, 1.0), DependencyEdge(1, 2, 1.0)),
            arrival_time=7.0,
            relative_deadline=20.0,
            absolute_deadline=27.0,
        )

        stats, details = distribute_existing_workflow_deadline(
            workflow,
            servers(),
            link_bandwidth_mbps=links(),
        )

        self.assertEqual(workflow.relative_deadline, 20.0)
        self.assertEqual(workflow.absolute_deadline, 27.0)
        self.assertEqual(stats.relative_deadline, 20.0)
        self.assertEqual(stats.num_tasks, 3)
        self.assertEqual(stats.method, "heft_aft_scaled_v1")
        self.assertIsNotNone(details)
        for task_id, aft in details.heft_aft_by_task.items():
            self.assertAlmostEqual(
                details.relative_by_task[task_id],
                stats.scale_ratio * aft,
            )
        self.assertLessEqual(
            details.relative_by_task[0],
            details.relative_by_task[1],
        )
        self.assertLessEqual(
            details.relative_by_task[1],
            details.relative_by_task[2],
        )
        self.assertAlmostEqual(
            max(details.relative_by_task.values()),
            workflow.relative_deadline,
        )

    def test_can_skip_large_per_task_details(self):
        workflow = Workflow(
            "wf",
            (Task(0, 10.0),),
            (),
            arrival_time=0.0,
            relative_deadline=3.0,
        )
        stats, details = distribute_existing_workflow_deadline(
            workflow,
            servers(),
            link_bandwidth_mbps=links(),
            return_details=False,
        )
        self.assertIsNone(details)
        self.assertEqual(stats.num_tasks, 1)
        self.assertAlmostEqual(stats.max_sub_deadline, 3.0)

    def test_rejects_missing_existing_deadline(self):
        workflow = Workflow("wf", (Task(0, 10.0),), (), arrival_time=0.0)
        with self.assertRaises(ValueError):
            distribute_existing_workflow_deadline(
                workflow,
                servers(),
                link_bandwidth_mbps=links(),
            )

    def test_environment_exposes_existing_task_sub_deadline(self):
        workflow = Workflow(
            "wf",
            (Task(0, 10.0), Task(1, 10.0)),
            (DependencyEdge(0, 1, 0.0),),
            arrival_time=0.0,
            relative_deadline=40.0,
            absolute_deadline=40.0,
        )
        env = EdgeCloudWorkflowEnv(
            [workflow],
            runtime_resources(),
            {
                "ready_task_order": "upward_rank",
                "sub_deadline": {
                    "enabled": True,
                    "method": "heft_aft_scaled_v1",
                },
            },
            seed=1,
        )
        ready = env.get_state()["ready_tasks"][0]
        self.assertIsNotNone(ready["sub_deadline_relative"])
        self.assertAlmostEqual(
            ready["sub_deadline_absolute"],
            ready["arrival_time"] + ready["sub_deadline_relative"],
        )
        self.assertEqual(env.workflow_states[0]["relative_deadline"], 40.0)

    def test_unit_jump_sub_deadline_cost_is_divided_by_task_count(self):
        cost = normalized_sub_deadline_violation(
            finish_time=11.0,
            workflow_arrival_time=0.0,
            sub_deadline_relative=10.0,
            workflow_task_count=4,
        )
        self.assertAlmostEqual(cost, 0.275)


if __name__ == "__main__":
    unittest.main()
