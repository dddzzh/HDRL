import math
import random
import unittest
from pathlib import Path

from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.network import apply_edge_device_shannon, shannon_rate_mbps
from src.simulator.resource import load_resources
from src.simulator.workflow import Task, Workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CommunicationBoundaryTests(unittest.TestCase):
    def test_formula_matches_reference_create_dataset(self):
        distance_m = 75.0
        expected_bps = 1.0e8 * math.log2(
            1.0 + distance_m ** -4.0 / (1.0e-13 + 1.0e-11)
        )
        self.assertAlmostEqual(shannon_rate_mbps(distance_m), expected_bps / 1.0e6)

    def test_default_resources_do_not_enable_iot_shannon_links(self):
        resources = load_resources(
            str(PROJECT_ROOT / "configs" / "resources_edge_cloud_default.json"),
            num_edge=3,
            num_cloud=1,
            seed=17,
        )
        edge_servers = [server for server in resources["servers"] if server["layer"] == "edge"]
        self.assertEqual(len(edge_servers), 3)
        for server in edge_servers:
            self.assertNotIn("device_distance_m", server)
            self.assertNotIn("device_bandwidth_mbps", server)

        cloud_servers = [server for server in resources["servers"] if server["layer"] == "cloud"]
        self.assertEqual(len(cloud_servers), 1)
        self.assertNotIn("device_distance_m", cloud_servers[0])
        self.assertNotIn("device_bandwidth_mbps", cloud_servers[0])

        self.assertNotIn("iot_edge", resources["network"]["link_bandwidth_mbps"])
        self.assertNotIn("iot_cloud", resources["network"]["link_bandwidth_mbps"])
        self.assertEqual(resources["network"]["link_bandwidth_mbps"]["edge_cloud"], 100.0)

    def test_optional_shannon_model_still_applies_only_to_edge(self):
        resources = {
            "network": {"edge_device": {"model": "shannon_reference"}},
            "servers": [
                {"id": 0, "layer": "edge", "bandwidth_mbps": 1000.0},
                {"id": 1, "layer": "cloud", "bandwidth_mbps": 10000.0},
            ],
        }
        apply_edge_device_shannon(resources, random.Random(7))

        edge, cloud = resources["servers"]
        self.assertIn("device_bandwidth_mbps", edge)
        self.assertIn("device_distance_m", edge)
        self.assertNotIn("device_bandwidth_mbps", cloud)
        self.assertNotIn("device_distance_m", cloud)

    def test_entry_task_is_ready_without_device_upload(self):
        resources = load_resources(
            str(PROJECT_ROOT / "configs" / "resources_edge_cloud_default.json"),
            num_edge=2,
            num_cloud=0,
            seed=23,
        )
        workflow = Workflow(
            workflow_id="wf",
            tasks=(Task(0, workload_cycles=1.0e9),),
            edges=(),
            arrival_time=0.0,
            input_data_mb=8.0,
            output_data_mb=0.0,
        )
        env = EdgeCloudWorkflowEnv([workflow], resources, {"deadline_type": "medium"}, seed=23)
        workflow_state = env.workflow_states[0]
        for server in env.servers:
            ready = env.compute_replica_ready_times(
                workflow_state,
                task_id=0,
                target_server_id=server["id"],
            )
            self.assertEqual(ready["dependency_ready_time"], workflow.arrival_time)
            self.assertEqual(ready["data_ready_time"], workflow.arrival_time)
            self.assertEqual(ready["transfer_time"], 0.0)
            self.assertEqual(ready["upload_time"], 0.0)


if __name__ == "__main__":
    unittest.main()
