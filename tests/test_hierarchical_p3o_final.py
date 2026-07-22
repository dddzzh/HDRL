import copy
import importlib.util
import math
from pathlib import Path
import random
import unittest

import numpy as np

from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.workflow import Task, Workflow

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

if torch is not None:
    from src.ppo.dual_agent import HighLevelPPOAgent, HierarchicalPPOAgent, LowLevelDVFSPPOAgent
else:  # pragma: no cover
    HighLevelPPOAgent = None
    HierarchicalPPOAgent = None
    LowLevelDVFSPPOAgent = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hierarchical_reward = load_module("hierarchical_reward_under_test", "src/ppo/hierarchical_reward.py")
rollout_buffer = load_module("rollout_buffer_under_test", "src/ppo/rollout_buffer.py")
arrival_process = load_module("arrival_process_under_test", "src/datasets/arrival_process.py")

compute_allowed_log_risk = hierarchical_reward.compute_allowed_log_risk
compute_deadline_cost_increment = hierarchical_reward.compute_deadline_cost_increment
compute_energy_decomposed_rewards = hierarchical_reward.compute_energy_decomposed_rewards
compute_low_energy_saving_reward = hierarchical_reward.compute_low_energy_saving_reward
compute_partial_workflow_log_risk = hierarchical_reward.compute_partial_workflow_log_risk
compute_partial_workflow_makespan = hierarchical_reward.compute_partial_workflow_makespan
compute_reliability_cost_increment = hierarchical_reward.compute_reliability_cost_increment
compute_workflow_deadline_penalty_distribution = hierarchical_reward.compute_workflow_deadline_penalty_distribution
PPORolloutBuffer = rollout_buffer.PPORolloutBuffer
create_time_table = arrival_process.create_time_table
resolve_arrival_rate = arrival_process.resolve_arrival_rate


def resources():
    return {
        "network": {
            "bandwidth_mbps": 100.0,
            "link_bandwidth_mbps": {
                "edge_edge": 100.0,
                "edge_cloud": 100.0,
                "cloud_edge": 100.0,
                "cloud_cloud": 100.0,
                "iot_edge": 100.0,
                "edge_iot": 100.0,
                "iot_cloud": 100.0,
                "cloud_iot": 100.0,
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
                "failure_rate": 0.0,
                "min_rho": 0.1,
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
                "failure_rate": 0.0,
                "min_rho": 0.1,
                "execution_available_time": 0.0,
                "backup_available_time": 0.0,
                "available_time": 0.0,
            },
        ],
    }


def add_transition(
    buffer,
    workflow_id,
    deadline_cost=0.0,
    reliability_cost=0.0,
    deadline_limit=100.0,
    reliability_limit=None,
    partial_makespan=0.0,
    partial_log_risk=0.0,
    workflow_done=False,
    server_id=0,
    deadline_value=0.0,
    reliability_value=0.0,
    done=False,
):
    if reliability_limit is None:
        reliability_limit = compute_allowed_log_risk(0.9)
    buffer.add(
        state=np.zeros(4, dtype=np.float32),
        action=0.5,
        log_prob=0.0,
        value=0.0,
        reward=0.0,
        done=done,
        next_state=np.zeros(4, dtype=np.float32),
        deadline_cost=deadline_cost,
        reliability_cost=reliability_cost,
        deadline_value=deadline_value,
        reliability_value=reliability_value,
        workflow_id=workflow_id,
        deadline_limit=deadline_limit,
        reliability_limit=reliability_limit,
        current_partial_makespan=partial_makespan,
        current_partial_log_risk=partial_log_risk,
        workflow_done=workflow_done,
        info={"workflow_id": workflow_id, "server_id": server_id},
    )


def workflow_state_snapshot(env):
    rows = []
    for state in env.workflow_states:
        rows.append(
            {
                "workflow_id": state["workflow_id"],
                "arrival_time": state["arrival_time"],
                "released": state["released"],
                "task_status": copy.deepcopy(state["task_status"]),
                "task_finish_times": copy.deepcopy(state["task_finish_times"]),
                "task_server": copy.deepcopy(state["task_server"]),
                "task_primary_plan": copy.deepcopy(state["task_primary_plan"]),
                "task_backup_plan": copy.deepcopy(state["task_backup_plan"]),
                "task_reliability": copy.deepcopy(state["task_reliability"]),
                "absolute_deadline": state.get("absolute_deadline"),
                "relative_deadline": state.get("relative_deadline"),
            }
        )
    return rows


class HierarchicalP3OFinalTests(unittest.TestCase):
    def test_energy_decomposition_identity(self):
        high, low = compute_energy_decomposed_rewards(energy_ref=80.0, energy_actual=50.0, energy_norm=10.0)
        self.assertAlmostEqual(high + low, -5.0, places=6)

    def test_rho_one_low_reward_zero(self):
        high, low = compute_energy_decomposed_rewards(energy_ref=42.0, energy_actual=42.0, energy_norm=7.0)
        self.assertAlmostEqual(high, -6.0, places=6)
        self.assertAlmostEqual(low, 0.0, places=6)

    def test_lower_frequency_saves_energy_positive_low_reward(self):
        _, low = compute_energy_decomposed_rewards(energy_ref=100.0, energy_actual=80.0, energy_norm=10.0)
        self.assertGreater(low, 0.0)

    def test_lower_frequency_can_cost_energy_negative_low_reward(self):
        _, low = compute_energy_decomposed_rewards(energy_ref=100.0, energy_actual=120.0, energy_norm=10.0)
        self.assertLess(low, 0.0)

    def test_deadline_makespan_increment_telescopes(self):
        partials = [10.0, 25.0, 40.0]
        previous = 0.0
        costs = []
        for current in partials:
            costs.append(compute_deadline_cost_increment(previous, current))
            previous = current
        self.assertEqual(costs, [10.0, 15.0, 15.0])
        self.assertEqual(sum(costs), 40.0)
        self.assertNotEqual(sum(partials), sum(costs))

    def test_different_workflows_use_different_deadlines(self):
        makespans = np.asarray([80.0, 120.0])
        deadlines = np.asarray([100.0, 90.0])
        self.assertEqual((makespans <= deadlines).tolist(), [True, False])
        self.assertEqual((makespans - deadlines).tolist(), [-20.0, 30.0])

    def test_deadline_is_not_normalized(self):
        self.assertEqual(compute_partial_workflow_makespan([110.0], 30.0), 80.0)
        self.assertEqual(compute_deadline_cost_increment(20.0, 30.0), 10.0)

    def test_deadline_step_cost_is_not_violation_cost(self):
        self.assertEqual(compute_deadline_cost_increment(20.0, 30.0), 10.0)

    def test_reliability_log_risk_increment_telescopes(self):
        q0 = compute_partial_workflow_log_risk(1.0)
        q1 = compute_partial_workflow_log_risk(0.99)
        q2 = compute_partial_workflow_log_risk(0.95)
        costs = [
            compute_reliability_cost_increment(q0, q1),
            compute_reliability_cost_increment(q1, q2),
        ]
        self.assertAlmostEqual(sum(costs), q2, places=8)

    def test_different_workflows_use_different_reliability_requirements(self):
        q = np.asarray([compute_partial_workflow_log_risk(0.95), compute_partial_workflow_log_risk(0.85)])
        limits = np.asarray([compute_allowed_log_risk(0.90), compute_allowed_log_risk(0.90)])
        self.assertEqual((q <= limits).tolist(), [True, False])

    def test_reliability_is_not_normalized(self):
        q = compute_partial_workflow_log_risk(0.95)
        q_req = compute_allowed_log_risk(0.90)
        self.assertAlmostEqual(q, -math.log(0.95), places=8)
        self.assertAlmostEqual(q_req, -math.log(0.90), places=8)

    @unittest.skipIf(torch is None, "torch is required")
    def test_relu_is_applied_per_workflow_before_mean(self):
        losses = torch.tensor([-20.0, 10.0])
        self.assertEqual(float(torch.relu(losses).mean().item()), 5.0)
        self.assertEqual(float(torch.relu(losses.mean()).item()), 0.0)

    def test_global_limit_does_not_override_transition_limits(self):
        buffer = PPORolloutBuffer()
        add_transition(buffer, "w1", deadline_limit=100.0, partial_makespan=80.0)
        add_transition(buffer, "w2", deadline_limit=90.0, partial_makespan=120.0)
        rows = {workflow_id: indices for workflow_id, indices in buffer.workflow_groups()}
        self.assertEqual(buffer.deadline_limits[rows["w1"][-1]], 100.0)
        self.assertEqual(buffer.deadline_limits[rows["w2"][-1]], 90.0)

    def test_same_workflow_limit_consistency(self):
        buffer = PPORolloutBuffer()
        add_transition(buffer, "w1", deadline_limit=100.0, reliability_limit=0.1)
        with self.assertRaises(RuntimeError):
            add_transition(buffer, "w1", deadline_limit=101.0, reliability_limit=0.1)
        buffer = PPORolloutBuffer()
        add_transition(buffer, "w1", deadline_limit=100.0, reliability_limit=0.1)
        with self.assertRaises(RuntimeError):
            add_transition(buffer, "w1", deadline_limit=100.0, reliability_limit=0.2)

    def test_cost_is_grouped_by_workflow_not_server(self):
        buffer = PPORolloutBuffer()
        add_transition(buffer, "w1", deadline_cost=1.0, partial_makespan=1.0, server_id=0)
        add_transition(buffer, "w2", deadline_cost=10.0, partial_makespan=10.0, server_id=0)
        add_transition(buffer, "w1", deadline_cost=2.0, partial_makespan=3.0, server_id=1)
        workflow_sums = {
            workflow_id: sum(buffer.deadline_costs[index] for index in indices)
            for workflow_id, indices in buffer.workflow_groups()
        }
        self.assertEqual(workflow_sums, {"w1": 3.0, "w2": 10.0})

    def test_incomplete_workflow_uses_bootstrap_and_keeps_partial_constraint(self):
        buffer = PPORolloutBuffer()
        add_transition(buffer, "w1", deadline_cost=10.0, partial_makespan=10.0, workflow_done=False)
        returns, _ = buffer.compute_workflow_cost_returns_and_advantages(
            buffer.deadline_costs,
            buffer.deadline_values,
            gamma=0.99,
            gae_lambda=1.0,
            bootstrap_values=np.asarray([5.0], dtype=np.float32),
        )
        self.assertAlmostEqual(float(returns[0]), 14.95, places=5)
        self.assertEqual(buffer.partial_makespans[0], 10.0)

    def test_workflow_done_reaches_high_buffer(self):
        buffer = PPORolloutBuffer()
        add_transition(buffer, "w1", workflow_done=True, done=False)
        self.assertEqual(buffer.workflow_dones, [True])
        self.assertTrue(buffer.infos[0]["workflow_done"])

    def test_workflow_done_reaches_low_buffer(self):
        buffer = PPORolloutBuffer()
        add_transition(buffer, "w1", workflow_done=False, done=True)
        self.assertEqual(buffer.workflow_dones, [False])
        self.assertTrue(buffer.dones[0])

    def test_completed_workflow_not_bootstrapped(self):
        buffer = PPORolloutBuffer()
        add_transition(buffer, "w1", deadline_cost=10.0, partial_makespan=10.0, workflow_done=True, done=False)
        returns, _ = buffer.compute_workflow_cost_returns_and_advantages(
            buffer.deadline_costs,
            buffer.deadline_values,
            gamma=0.99,
            gae_lambda=1.0,
            bootstrap_values=np.asarray([100.0], dtype=np.float32),
        )
        self.assertAlmostEqual(float(returns[0]), 10.0, places=6)

    def test_rollout_truncation_is_not_workflow_done(self):
        buffer = PPORolloutBuffer()
        add_transition(buffer, "w1", deadline_cost=10.0, partial_makespan=10.0, workflow_done=False, done=False)
        returns, _ = buffer.compute_workflow_cost_returns_and_advantages(
            buffer.deadline_costs,
            buffer.deadline_values,
            gamma=1.0,
            gae_lambda=1.0,
            bootstrap_values=np.asarray([5.0], dtype=np.float32),
        )
        self.assertAlmostEqual(float(returns[0]), 15.0, places=6)

    def test_completed_workflow_count_matches_environment(self):
        buffer = PPORolloutBuffer()
        for idx in range(20):
            add_transition(buffer, f"w{idx}", workflow_done=True)
        completed = sum(1 for _, indices in buffer.workflow_groups() if any(buffer.workflow_dones[i] for i in indices))
        self.assertEqual(completed, 20)

    def test_deadline_increment_telescopes_with_cost_gamma_one(self):
        buffer = PPORolloutBuffer()
        add_transition(buffer, "w1", deadline_cost=10.0, partial_makespan=10.0)
        add_transition(buffer, "w1", deadline_cost=15.0, partial_makespan=25.0)
        add_transition(buffer, "w1", deadline_cost=15.0, partial_makespan=40.0, workflow_done=True)
        returns, _ = buffer.compute_workflow_cost_returns_and_advantages(
            buffer.deadline_costs,
            buffer.deadline_values,
            gamma=1.0,
            gae_lambda=1.0,
        )
        self.assertAlmostEqual(float(returns[0]), 40.0, places=6)

    def test_normalized_deadline_cost_telescopes_to_response_ratio(self):
        relative_deadline = 100.0
        raw_increments = [10.0, 15.0, 15.0, 5.0]
        normalized_costs = [value / relative_deadline for value in raw_increments]
        self.assertAlmostEqual(sum(normalized_costs), 45.0 / relative_deadline, places=6)

    def test_reliability_increment_telescopes_with_cost_gamma_one(self):
        q0 = compute_partial_workflow_log_risk(1.0)
        q1 = compute_partial_workflow_log_risk(0.99)
        q2 = compute_partial_workflow_log_risk(0.95)
        buffer = PPORolloutBuffer()
        add_transition(buffer, "w1", reliability_cost=q1 - q0, partial_log_risk=q1)
        add_transition(buffer, "w1", reliability_cost=q2 - q1, partial_log_risk=q2, workflow_done=True)
        returns, _ = buffer.compute_workflow_cost_returns_and_advantages(
            buffer.reliability_costs,
            buffer.reliability_values,
            gamma=1.0,
            gae_lambda=1.0,
        )
        self.assertAlmostEqual(float(returns[0]), q2, places=8)

    def test_reward_gamma_does_not_affect_cost_return(self):
        buffer = PPORolloutBuffer()
        add_transition(buffer, "w1", deadline_cost=10.0, partial_makespan=10.0)
        add_transition(buffer, "w1", deadline_cost=15.0, partial_makespan=25.0)
        add_transition(buffer, "w1", deadline_cost=15.0, partial_makespan=40.0, workflow_done=True)
        discounted_returns, _ = buffer.compute_workflow_cost_returns_and_advantages(
            buffer.deadline_costs,
            buffer.deadline_values,
            gamma=0.99,
            gae_lambda=1.0,
        )
        cost_returns, _ = buffer.compute_workflow_cost_returns_and_advantages(
            buffer.deadline_costs,
            buffer.deadline_values,
            gamma=1.0,
            gae_lambda=1.0,
        )
        self.assertNotAlmostEqual(float(discounted_returns[0]), 40.0, places=4)
        self.assertAlmostEqual(float(cost_returns[0]), 40.0, places=6)

    @unittest.skipIf(torch is None, "torch is required")
    def test_p3o_actor_loss_uses_reward_plus_penalties_minus_entropy(self):
        agent = HighLevelPPOAgent({"device": "cpu", "use_p3o": True, "kappa_deadline": 2.0, "kappa_reliability": 3.0})
        ratio = torch.ones(2)
        p3o_batch = {
            "deadline_advantages": torch.tensor([0.0, 1.0]),
            "reliability_advantages": torch.tensor([0.0, 1.0]),
            "workflow_ids": ["w1", "w2"],
            "partial_makespans": torch.tensor([0.80, 1.20]),
            "partial_log_risks": torch.tensor([0.05, 0.20]),
            "deadline_limits": torch.tensor([1.0, 1.0]),
            "reliability_limits": torch.tensor([0.10, 0.10]),
        }
        metrics = agent._p3o_actor_terms(ratio, p3o_batch, torch.tensor([0, 1]))
        reward_clip_loss = torch.tensor(1.25)
        entropy = torch.tensor([0.5, 0.5])
        actor_loss = reward_clip_loss + metrics["deadline_penalty"] + metrics["reliability_penalty"] - agent.entropy_coef * entropy.mean()
        expected_deadline = 2.0 * (1.0 + 2.0 * 0.2) * 1.0
        expected_reliability = 3.0 * (1.0 + 2.0 * 1.0) * 1.0
        expected = 1.25 + expected_deadline + expected_reliability - agent.entropy_coef * 0.5
        self.assertAlmostEqual(float(actor_loss.item()), float(expected), places=6)

    @unittest.skipIf(torch is None, "torch is required")
    def test_violation_gated_deadline_active_even_with_negative_surrogate(self):
        agent = HighLevelPPOAgent({"device": "cpu", "use_p3o": True})
        metrics = agent._p3o_actor_terms(
            torch.ones(1),
            {
                "deadline_advantages": torch.tensor([-3.0]),
                "reliability_advantages": torch.zeros(1),
                "workflow_ids": ["w1"],
                "partial_makespans": torch.tensor([1.20]),
                "partial_log_risks": torch.tensor([0.20]),
                "deadline_limits": torch.tensor([1.0]),
                "reliability_limits": torch.tensor([0.10]),
            },
            torch.tensor([0]),
        )
        self.assertAlmostEqual(float(metrics["deadline_active_rate"].item()), 1.0)
        self.assertAlmostEqual(float(metrics["deadline_violation_but_inactive_rate"].item()), 0.0)
        self.assertLess(float(metrics["deadline_policy_loss"].item()), 0.0)

    @unittest.skipIf(torch is None, "torch is required")
    def test_deadline_gate_near_deadline_and_inactive_when_loose(self):
        agent = HighLevelPPOAgent({"device": "cpu", "use_p3o": True, "deadline_gate_margin_ratio": 0.05})
        metrics = agent._p3o_actor_terms(
            torch.ones(2),
            {
                "deadline_advantages": torch.ones(2),
                "reliability_advantages": torch.zeros(2),
                "workflow_ids": ["near", "loose"],
                "partial_makespans": torch.tensor([0.97, 0.70]),
                "partial_log_risks": torch.tensor([0.01, 0.01]),
                "deadline_limits": torch.tensor([1.0, 1.0]),
                "reliability_limits": torch.tensor([1.0, 1.0]),
            },
            torch.tensor([0, 1]),
        )
        self.assertAlmostEqual(float(metrics["deadline_active_rate"].item()), 0.5)
        self.assertAlmostEqual(float(metrics["deadline_policy_loss"].item()), 1.0)

    @unittest.skipIf(torch is None, "torch is required")
    def test_high_cost_gamma_lambda_defaults(self):
        agent = HighLevelPPOAgent({"device": "cpu", "use_p3o": True})
        self.assertAlmostEqual(agent.cost_gamma, 1.0)
        self.assertAlmostEqual(agent.cost_gae_lambda, 1.0)

    @unittest.skipIf(torch is None, "torch is required")
    def test_use_p3o_false_has_no_p3o_terms(self):
        agent = LowLevelDVFSPPOAgent({"device": "cpu", "use_p3o": False})
        self.assertFalse(agent.use_p3o)
        reward_clip_loss = torch.tensor(2.0)
        entropy = torch.tensor([0.25])
        actor_loss = reward_clip_loss - agent.entropy_coef * entropy.mean()
        self.assertAlmostEqual(float(actor_loss.item()), 2.0 - agent.entropy_coef * 0.25, places=6)

    def test_energy_ref_estimation_has_no_side_effects(self):
        workflow = Workflow(
            workflow_id="wf",
            arrival_time=0.0,
            absolute_deadline=100.0,
            tasks=(Task(0, workload_cycles=10.0),),
            edges=(),
            reliability=0.9,
        )
        env = EdgeCloudWorkflowEnv(
            [workflow],
            resources(),
            {
                "enable_backup_duplication": True,
                "default_reliability": 0.9,
                "backup_rho_levels": [1.0],
            },
            seed=1,
        )
        action = {
            "workflow_id": "wf",
            "task_id": 0,
            "server_id": 0,
            "backup_server_id": 1,
            "rho": 0.5,
        }
        before = {
            "current_time": env.current_time,
            "event_queue": [
                (item.time, item.sequence, item.event_type, copy.deepcopy(item.payload))
                for item in env.event_queue._heap
            ],
            "servers": copy.deepcopy(env.servers),
            "timeline": copy.deepcopy(env.server_real_timeline),
            "workflow_states": workflow_state_snapshot(env),
        }

        energy_ref = env.estimate_action_energy_ref(action)

        after = {
            "current_time": env.current_time,
            "event_queue": [
                (item.time, item.sequence, item.event_type, copy.deepcopy(item.payload))
                for item in env.event_queue._heap
            ],
            "servers": copy.deepcopy(env.servers),
            "timeline": copy.deepcopy(env.server_real_timeline),
            "workflow_states": workflow_state_snapshot(env),
        }
        self.assertGreater(energy_ref, 0.0)
        self.assertEqual(before, after)

class LowDvfsRefactorTests(unittest.TestCase):
    def test_low_energy_reward_ratios(self):
        self.assertAlmostEqual(compute_low_energy_saving_reward(100.0, 100.0), 0.0)
        self.assertAlmostEqual(compute_low_energy_saving_reward(100.0, 80.0), 0.2)
        self.assertAlmostEqual(compute_low_energy_saving_reward(100.0, 110.0), -0.1)

    def test_no_server_trajectory_gae_for_low_gamma_zero(self):
        if LowLevelDVFSPPOAgent is None:
            self.skipTest("torch unavailable")
        agent = LowLevelDVFSPPOAgent({"state": {"max_ready_tasks": 1, "max_servers": 1}, "reward_gamma": 0.0})
        buffer = PPORolloutBuffer()
        for reward in (1.0, 2.0, 3.0):
            buffer.add(
                state=np.zeros(agent.encoder.low_state_dim, dtype=np.float32),
                action=0.5,
                log_prob=0.0,
                value=0.0,
                reward=reward,
                done=False,
                info={"server_id": 0, "actor_update_mask": True},
            )
        returns, _ = agent._compute_stream_returns_and_advantages(buffer, buffer.rewards, buffer.values)
        np.testing.assert_allclose(returns, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
        self.assertIsNone(agent.trajectory_group_key)

    def test_safe_rho_mapping_range_and_deterministic(self):
        if LowLevelDVFSPPOAgent is None:
            self.skipTest("torch unavailable")
        agent = LowLevelDVFSPPOAgent({"state": {"max_ready_tasks": 1, "max_servers": 1}, "rho_min": 0.1})
        action = torch.linspace(0.0, 1.0, 11).unsqueeze(-1)
        rho = agent._rho_from_unit_action(action, rho_safe=0.8)
        self.assertTrue(bool((rho >= 0.8 - 1e-8).all()))
        self.assertTrue(bool((rho <= 1.0 + 1e-8).all()))
        self.assertAlmostEqual(float(agent._rho_from_unit_action(torch.tensor([[0.5]]), rho_safe=0.8).item()), 0.9)

    def test_infeasible_forces_max_frequency_and_masks_actor(self):
        if LowLevelDVFSPPOAgent is None:
            self.skipTest("torch unavailable")
        agent = LowLevelDVFSPPOAgent({"state": {"max_ready_tasks": 1, "max_servers": 1}, "rho_min": 0.1})
        obs = type(
            "Obs",
            (),
            {
                "global_features": np.zeros(5, dtype=np.float32),
                "ready_tasks": ({"workload_cycles": 1e9, "relative_deadline": 1.0, "absolute_deadline": 1.0},),
                "servers": ({"id": 0, "max_freq_hz": 1e9, "static_power": 10.0, "dynamic_power": 20.0, "min_rho": 0.1},),
            },
        )()
        decision = agent.select_frequency(
            obs,
            0,
            0,
            deterministic=True,
            safety_metrics={"rho_safe": 1.0, "low_action_feasible": False, "workflow_relative_deadline": 1.0},
        )
        self.assertEqual(decision["rho"], 1.0)
        self.assertFalse(decision["actor_update_mask"])
        self.assertTrue(decision["forced_max_frequency"])

    def test_low_state_shape_finite_and_tighter_deadline_raises_rho_safe(self):
        if LowLevelDVFSPPOAgent is None:
            self.skipTest("torch unavailable")
        agent = LowLevelDVFSPPOAgent({"state": {"max_ready_tasks": 1, "max_servers": 1}})
        obs = type(
            "Obs",
            (),
            {
                "global_features": np.zeros(5, dtype=np.float32),
                "ready_tasks": ({"workload_cycles": 1e9, "relative_deadline": 10.0, "absolute_deadline": 10.0},),
                "servers": ({"id": 0, "max_freq_hz": 1e9, "static_power": 10.0, "dynamic_power": 20.0, "min_rho": 0.1},),
            },
        )()
        loose = agent.build_low_state(obs, 0, 0, safety_metrics={"rho_safe": 0.2, "workflow_relative_deadline": 10.0})
        tight = agent.build_low_state(obs, 0, 0, safety_metrics={"rho_safe": 0.8, "workflow_relative_deadline": 10.0})
        self.assertEqual(loose.shape, (9,))
        self.assertTrue(np.isfinite(loose).all())
        self.assertGreater(tight[5], loose[5])

    def test_workflow_penalty_distribution(self):
        penalties = compute_workflow_deadline_penalty_distribution([1.0, 3.0], 0.2, coef=5.0)
        self.assertAlmostEqual(sum(penalties), 1.0)
        self.assertGreater(penalties[1], penalties[0])
        uniform = compute_workflow_deadline_penalty_distribution([0.0, 0.0], 0.2, coef=5.0)
        self.assertAlmostEqual(uniform[0], uniform[1])

    def test_shared_low_network_single_object(self):
        if HierarchicalPPOAgent is None:
            self.skipTest("torch unavailable")
        agent = HierarchicalPPOAgent(
            {
                "shared": {"state": {"max_ready_tasks": 1, "max_servers": 1}},
                "high": {"hidden_dim": 8, "hidden_layers": 1},
                "low": {"hidden_dim": 8, "hidden_layers": 1},
            }
        )
        self.assertIs(agent.server_low_agent_pool.shared_low_agent.network, agent.low_dvfs_agent.network)
        self.assertIs(agent.server_low_agent_pool.shared_low_agent.optimizer, agent.low_dvfs_agent.optimizer)


class ReferenceArrivalProcessTests(unittest.TestCase):
    def test_create_time_table_matches_reference_shape(self):
        rng = random.Random(7)
        rows = create_time_table(num_mobile_devices=3, num_workflows=5, lambda_rate=0.03, rng=rng)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0], (0.0, 0))
        self.assertEqual(rows[1], (0.0, 1))
        self.assertEqual(rows[2], (0.0, 2))
        self.assertTrue(all(rows[index][0] <= rows[index + 1][0] for index in range(len(rows) - 1)))

    def test_mean_interarrival_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_arrival_rate({"arrival_rate": 0.03, "mean_interarrival": 2.0})


if __name__ == "__main__":
    unittest.main()
