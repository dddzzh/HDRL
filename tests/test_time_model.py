import unittest
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.deadline import sample_deadline_multiplier
from src.simulator.metrics import compute_summary
from src.simulator.workflow import DependencyEdge, Task, Workflow


def resources():
    return {
        "network": {
            "bandwidth_mbps": 100.0,
            "link_bandwidth_mbps": {
                "edge_edge": 100.0,
                "edge_cloud": 100.0,
                "cloud_edge": 100.0,
                "cloud_cloud": 100.0,
            },
        },
        "servers": [
            {
                "id": 0,
                "name": "edge_0",
                "type": "edge",
                "max_freq_hz": 1.0,
                "max_speed_mips": 1.0,
                "static_power": 1.0,
                "dynamic_power": 1.0,
                "memory_gb": 1.0,
                "bandwidth_mbps": 100.0,
                "failure_rate": 0.0,
                "min_rho": 1.0,
                "execution_available_time": 0.0,
                "backup_available_time": 0.0,
                "available_time": 0.0,
            },
            {
                "id": 1,
                "name": "edge_1",
                "type": "edge",
                "max_freq_hz": 1.0,
                "max_speed_mips": 1.0,
                "static_power": 1.0,
                "dynamic_power": 1.0,
                "memory_gb": 1.0,
                "bandwidth_mbps": 100.0,
                "failure_rate": 0.0,
                "min_rho": 1.0,
                "execution_available_time": 0.0,
                "backup_available_time": 0.0,
                "available_time": 0.0,
            },
            {
                "id": 2,
                "name": "edge_2",
                "type": "edge",
                "max_freq_hz": 1.0,
                "max_speed_mips": 1.0,
                "static_power": 1.0,
                "dynamic_power": 1.0,
                "memory_gb": 1.0,
                "bandwidth_mbps": 100.0,
                "failure_rate": 0.0,
                "min_rho": 1.0,
                "execution_available_time": 0.0,
                "backup_available_time": 0.0,
                "available_time": 0.0,
            },
        ],
    }


def make_env(workflow, sim_config=None):
    env = EdgeCloudWorkflowEnv([workflow], resources(), sim_config or {}, seed=1)
    state = env.workflow_states[0]
    state["released"] = True
    return env, state


def mark_parent_finished(state, task_id, finish_time, server_id):
    state["task_status"][task_id] = "finished"
    state["task_finish_times"][task_id] = float(finish_time)
    state["task_server"][task_id] = int(server_id)
    state["task_primary_plan"][task_id] = {
        "aft": float(finish_time),
        "server_id": int(server_id),
    }


class TimeModelTests(unittest.TestCase):
    def test_ready_queue_can_use_upward_rank_instead_of_smallest_workload(self):
        workflow = Workflow(
            workflow_id="wf",
            arrival_time=0.0,
            absolute_deadline=1000.0,
            tasks=(
                Task(0, workload_cycles=20.0),
                Task(1, workload_cycles=100.0),
                Task(2, workload_cycles=1.0),
            ),
            edges=(DependencyEdge(0, 1, 0.0),),
        )

        uprank_env, _ = make_env(workflow, {"ready_task_order": "upward_rank"})
        sjf_env, _ = make_env(workflow, {"ready_task_order": "smallest_workload"})

        uprank_ready = uprank_env._get_ready_tasks()
        sjf_ready = sjf_env._get_ready_tasks()
        self.assertEqual(uprank_ready[0]["task_id"], 0)
        self.assertGreater(uprank_ready[0]["upward_rank"], uprank_ready[1]["upward_rank"])
        self.assertEqual(sjf_ready[0]["task_id"], 2)

    def test_ast_never_goes_before_current_time(self):
        workflow = Workflow(
            workflow_id="wf",
            arrival_time=0.0,
            absolute_deadline=200.0,
            tasks=(
                Task(0, workload_cycles=1.0),
                Task(1, workload_cycles=10.0),
            ),
            edges=(DependencyEdge(0, 1, 0.0),),
        )
        env, state = make_env(workflow)
        mark_parent_finished(state, 0, 80.0, 0)
        env.current_time = 100.0
        env.servers[0]["execution_available_time"] = 90.0
        env.servers[0]["available_time"] = 90.0

        plan = env.estimate_replica_plan(
            task=env._get_task(workflow, 1),
            server=env.servers[0],
            rho=1.0,
            workflow_state=state,
        )

        self.assertEqual(plan["ast"], 100.0)
        self.assertEqual(plan["aft"], 110.0)
        self.assertGreaterEqual(plan["ast"], env.current_time)

    def test_ready_queue_waiting_is_not_missed_by_deadline(self):
        workflow = Workflow(
            workflow_id="wf",
            arrival_time=0.0,
            absolute_deadline=60.0,
            tasks=(
                Task(0, workload_cycles=1.0),
                Task(1, workload_cycles=10.0),
            ),
            edges=(DependencyEdge(0, 1, 0.0),),
        )
        env, state = make_env(workflow)
        mark_parent_finished(state, 0, 10.0, 0)
        env.current_time = 100.0

        plan = env.estimate_replica_plan(
            task=env._get_task(workflow, 1),
            server=env.servers[0],
            rho=1.0,
            workflow_state=state,
        )
        final_plan = env.combine_primary_backup(
            plan,
            None,
            task=env._get_task(workflow, 1),
            workflow_state=state,
        )

        self.assertEqual(plan["task_ready_time"], 10.0)
        self.assertEqual(plan["ready_queue_waiting_time"], 90.0)
        self.assertEqual(final_plan["absolute_deadline"], 60.0)
        self.assertFalse(final_plan["deadline_satisfied"])
        self.assertEqual(final_plan["deadline_violation"], 50.0)

    def test_backup_uses_real_execution_timeline_without_overlap(self):
        workflow = Workflow(
            workflow_id="wf",
            arrival_time=0.0,
            absolute_deadline=200.0,
            tasks=(
                Task(0, workload_cycles=10.0),
                Task(1, workload_cycles=10.0),
            ),
            edges=(),
        )
        env, state = make_env(workflow)
        env.current_time = 10.0
        env.update_server_timeline(
            env.servers[1],
            {
                "workflow_id": "old",
                "task_id": 99,
                "ast": 0.0,
                "aft": 50.0,
            },
            "primary",
        )
        env.servers[1]["execution_available_time"] = 50.0
        env.servers[1]["available_time"] = 50.0

        plan = env.estimate_replica_plan(
            task=env._get_task(workflow, 1),
            server=env.servers[1],
            rho=1.0,
            workflow_state=state,
            queue_type="backup",
        )
        env.update_server_timeline(env.servers[1], plan, "backup")

        self.assertEqual(plan["ast"], 50.0)
        self.assertTrue(env.validate_no_overlap(1))

    def test_successor_uses_earliest_replica_source_server(self):
        workflow = Workflow(
            workflow_id="wf",
            arrival_time=0.0,
            absolute_deadline=200.0,
            tasks=(
                Task(0, workload_cycles=1.0),
                Task(1, workload_cycles=1.0),
            ),
            edges=(DependencyEdge(0, 1, 100.0),),
        )
        env, state = make_env(workflow)
        mark_parent_finished(state, 0, 80.0, 1)
        env.current_time = 80.0

        data_ready_time, transfer_time, _ = env.compute_data_ready_time(state, 1, 2)

        self.assertEqual(state["task_server"][0], 1)
        self.assertEqual(transfer_time, 8.0)
        self.assertEqual(data_ready_time, 88.0)

        same_ready_time, same_transfer_time, _ = env.compute_data_ready_time(state, 1, 1)
        self.assertEqual(same_transfer_time, 0.0)
        self.assertEqual(same_ready_time, 80.0)

    def test_explicit_input_output_sizes_do_not_add_ingress_or_egress(self):
        workflow = Workflow(
            workflow_id="wf",
            arrival_time=5.0,
            absolute_deadline=100.0,
            tasks=(Task(0, workload_cycles=10.0),),
            edges=(),
            input_data_mb=30.0,
            output_data_mb=30.0,
        )
        env = EdgeCloudWorkflowEnv([workflow], resources(), {}, seed=1)
        env._process_events_through(5.0)
        ready = env.compute_replica_ready_times(env.workflow_states[0], 0, 0)
        self.assertEqual(ready["data_ready_time"], 5.0)
        self.assertEqual(ready["upload_time"], 0.0)

        env.step({"workflow_id": "wf", "task_id": 0, "server_id": 0, "rho": 1.0})
        env._process_events_through(15.0)
        state = env.workflow_states[0]
        self.assertEqual(state["download_time"], 0.0)
        self.assertEqual(state["egress_finish_time"], state["task_finish_times"][0])

    def test_absolute_deadline_count_is_reported(self):
        workflow = Workflow(
            workflow_id="wf",
            arrival_time=0.0,
            absolute_deadline=50.0,
            tasks=(Task(0, workload_cycles=10.0),),
            edges=(),
        )
        env = EdgeCloudWorkflowEnv([workflow], resources(), {}, seed=1)
        env.current_time = 0.0
        env._process_events_through(0.0)
        env.step(
            {
                "workflow_id": "wf",
                "task_id": 0,
                "server_id": 0,
                "rho": 1.0,
            }
        )
        env.current_time = 11.0
        env._process_events_through(11.0)

        workflow_results = env._build_workflow_results()
        summary = compute_summary(workflow_results, env.event_log, env.servers)

        self.assertEqual(summary["deadline_total_count"], 1)
        self.assertEqual(summary["deadline_success_count"], 1)
        self.assertEqual(summary["deadline_violation_sum"], 0.0)


class Cai2025DeadlineTests(unittest.TestCase):
    def test_deadline_multiplier_sampling_probabilities(self):
        cases = {
            "tight": 0.8,
            "medium": 0.5,
            "loose": 0.2,
        }
        for deadline_type, expected in cases.items():
            rng = random.Random(123)
            samples = [
                sample_deadline_multiplier(deadline_type, rng)
                for _ in range(10000)
            ]
            ratio_2x = samples.count(2.0) / len(samples)
            self.assertAlmostEqual(ratio_2x, expected, delta=0.03)

    def test_environment_generates_reproducible_workflow_level_deadlines(self):
        workflows_a = [
            Workflow(
                workflow_id=f"wf_{idx}",
                arrival_time=float(idx * 5),
                tasks=(Task(0, workload_cycles=10.0 + idx),),
                edges=(),
            )
            for idx in range(20)
        ]
        workflows_b = [
            Workflow(
                workflow_id=f"wf_{idx}",
                arrival_time=float(idx * 5),
                tasks=(Task(0, workload_cycles=10.0 + idx),),
                edges=(),
            )
            for idx in range(20)
        ]
        env_a = EdgeCloudWorkflowEnv(workflows_a, resources(), {"deadline_type": "tight"}, seed=99)
        env_b = EdgeCloudWorkflowEnv(workflows_b, resources(), {"deadline_type": "tight"}, seed=99)

        seq_a = [state["deadline_multiplier"] for state in env_a.workflow_states]
        seq_b = [state["deadline_multiplier"] for state in env_b.workflow_states]
        self.assertEqual(seq_a, seq_b)
        self.assertGreater(len(set(seq_a)), 1)
        for state in env_a.workflow_states:
            self.assertIn(state["deadline_multiplier"], {2.0, 3.0})
            self.assertGreater(state["reference_makespan"], 0.0)
            self.assertGreater(state["relative_deadline"], 0.0)
            self.assertEqual(
                state["absolute_deadline"],
                state["arrival_time"] + state["relative_deadline"],
            )

    def test_deadline_generation_does_not_use_dynamic_finish_time(self):
        workflow = Workflow(
            workflow_id="wf",
            arrival_time=10.0,
            tasks=(Task(0, workload_cycles=10.0),),
            edges=(),
        )
        env = EdgeCloudWorkflowEnv([workflow], resources(), {"deadline_type": "medium"}, seed=7)
        state = env.workflow_states[0]
        initial_deadline = state["absolute_deadline"]

        state["task_finish_times"][0] = 9999.0
        state["egress_finish_time"] = 10000.0
        self.assertEqual(env.workflow_absolute_deadline(state), initial_deadline)
        self.assertNotEqual(initial_deadline, state["egress_finish_time"])


if __name__ == "__main__":
    unittest.main()
