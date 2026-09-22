from __future__ import annotations

import math
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.ppo.hierarchical_agent import HighLevelP3OAgent, HierarchicalSafeAgent
from src.ppo.hierarchical_objective import HierarchicalObjective
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from src.ppo.hierarchical_state import (
    GLOBAL_FEATURE_DIM,
    LOW_STATE_FEATURE_DIM,
    SERVER_FEATURE_DIM,
    TASK_FEATURE_DIM,
    HierarchicalStateEncoder,
)
from src.schedulers.hierarchical_safe_scheduler import HierarchicalSafeScheduler
from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.workflow import DependencyEdge, Task, Workflow
from tools import hierarchical_common as train_safe


class FakeObjectiveEnv:
    def __init__(self, task_count=1):
        self.workflow_by_id = {
            "w": {
                "workflow_id": "w",
                "arrival_time": 0.0,
                "task_finish_times": {0: 0.0},
                "reliability_product": 1.0,
                "workflow": {"workflow_id": "w", "tasks": [{"id": i} for i in range(int(task_count))]},
            }
        }

    def workflow_relative_deadline(self, workflow_state):
        return 100.0

    def estimate_primary_energy_ref_rho1(self, action):
        return float(action["primary_energy_ref_rho1"])

    def _workflow_reliability_requirement(self, workflow_state):
        return 0.9


@unittest.skipIf(torch is None, "torch is required")
class TestHierarchicalSafePPO(unittest.TestCase):
    def high_state_dim_for_three_servers(self):
        return TASK_FEATURE_DIM + 3 * SERVER_FEATURE_DIM + 3 + GLOBAL_FEATURE_DIM

    def agent_config(self):
        return {
            "shared": {
                "device": "cpu",
                "state": {
                    "max_ready_tasks": 4,
                    "max_servers": 3,
                },
            },
            "high": {
                "hidden_dim": 32,
                "hidden_layers": 1,
                "primary_embedding_dim": 8,
                "batch_size": 8,
                "ppo_epochs": 1,
            },
            "low": {
                "hidden_dim": 16,
                "hidden_layers": 1,
                "batch_size": 8,
                "ppo_epochs": 1,
                "rho_min": 0.1,
            },
        }

    def sample_state(self, feasible=True):
        primary_mask = np.array([True, False, True], dtype=bool)
        backup_pair_mask = np.array(
            [
                [False, True, True],
                [True, False, True],
                [True, True, False],
            ],
            dtype=bool,
        )
        safety = {}
        for server_id in range(3):
            safety[str(server_id)] = {
                "server_id": server_id,
                "rho_safe": 0.65,
                "rho_deadline": 0.65,
                "rho_reliability": None,
                "low_action_feasible": bool(feasible),
                "estimated_ast": 0.0,
                "execution_time_rho_1": 1.0,
                "safe_execution_budget": 2.0,
                "task_latest_finish_time": 90.0,
                "workflow_relative_deadline": 100.0,
                "workflow_absolute_deadline": 100.0,
                "primary_energy_ref_rho1": 10.0 + server_id,
                "wait_time_ema": float(server_id),
                "queued_processing_time_ema": float(server_id + 1),
                "queue_count_ema": float(server_id),
                "recent_primary_assignments": 1.0 if server_id == 0 else 0.0,
                "recent_backup_assignments": 1.0 if server_id == 1 else 0.0,
            }
        return {
            "time": 0.0,
            "ready_tasks": [
                {
                    "workflow_id": "w0",
                    "task_id": 0,
                    "arrival_time": 0.0,
                    "workload_cycles": 1e8,
                    "absolute_deadline": 100.0,
                    "relative_deadline": 100.0,
                    "workflow_task_count": 1,
                    "reliability": 0.9,
                }
            ],
            "servers": [
                {
                    "id": i,
                    "type": "edge" if i < 2 else "cloud",
                    "server_type": "edge" if i < 2 else "cloud",
                    "execution_available_time": 0.0,
                    "available_time": 0.0,
                    "backup_available_time": 0.0,
                    "max_freq_hz": 1e9 * (i + 1),
                    "max_freq_ghz": float(i + 1),
                    "static_power": 10.0,
                    "dynamic_power": 20.0,
                    "bandwidth_mbps": 100.0,
                    "queue_load": 0.0,
                    "queue_count": 0,
                    "execution_queue_load": 0.0,
                    "execution_queue_count": 0,
                    "backup_queue_load": 0.0,
                    "backup_queue_count": 0,
                    "failure_rate": 1e-6,
                    "min_rho": 0.1,
                    "cpu_utilization": 0.0,
                }
                for i in range(3)
            ],
            "decision_context": {
                "workflow_id": "w0",
                "task_id": 0,
                "primary_server_mask": primary_mask,
                "backup_pair_mask": backup_pair_mask,
                "server_safety_metrics": safety,
                "task_energy_scale": 11.0,
                "shield_relaxed": False,
                "shield_relaxation_reason": None,
            },
        }

    def add_high_transition(self, buffer, workflow_id, reward, value=0.0, done=False):
        mask = np.array([True, True, True], dtype=bool)
        pair = np.array([[False, True, True], [True, False, True], [True, True, False]], dtype=bool)
        buffer.add(
            state=np.zeros(self.high_state_dim_for_three_servers(), dtype=np.float32),
            primary_action=0,
            backup_action=1,
            old_log_prob=0.0,
            reward_value=value,
            deadline_value=0.0,
            reliability_value=0.0,
            reward=reward,
            deadline_cost=1.0 if done else 0.0,
            reliability_cost=0.0,
            deadline_limit=1.0,
            reliability_limit=1.0,
            partial_makespan=1.0 if done else 0.0,
            partial_log_risk=0.0,
            workflow_id=workflow_id,
            workflow_done=done,
            episode_done=False,
            next_state=None,
            primary_mask=mask,
            backup_pair_mask=pair,
        )

    def add_high_cost_transition(self, buffer, workflow_id, *, deadline_cost, deadline_limit, partial_makespan, done=False):
        mask = np.array([True, True, True], dtype=bool)
        pair = np.array([[False, True, True], [True, False, True], [True, True, False]], dtype=bool)
        buffer.add(
            state=np.zeros(self.high_state_dim_for_three_servers(), dtype=np.float32),
            primary_action=0,
            backup_action=1,
            old_log_prob=0.0,
            reward_value=0.0,
            deadline_value=0.0,
            reliability_value=0.0,
            reward=0.0,
            deadline_cost=float(deadline_cost),
            reliability_cost=0.0,
            deadline_limit=float(deadline_limit),
            reliability_limit=1.0,
            partial_makespan=float(partial_makespan),
            partial_log_risk=0.0,
            workflow_id=workflow_id,
            workflow_done=done,
            episode_done=False,
            next_state=None,
            primary_mask=mask,
            backup_pair_mask=pair,
            info={"workflow_id": workflow_id},
        )

    def test_state_dimensions_unchanged(self):
        self.assertEqual(TASK_FEATURE_DIM, 3)
        self.assertEqual(SERVER_FEATURE_DIM, 21)
        self.assertEqual(GLOBAL_FEATURE_DIM, 5)
        self.assertEqual(LOW_STATE_FEATURE_DIM, 11)
        encoder = HierarchicalStateEncoder({"max_servers": 3, "max_ready_tasks": 4})
        self.assertEqual(encoder.high_state_dim, self.high_state_dim_for_three_servers())
        self.assertEqual(encoder.low_state_dim, 11)

    def test_high_state_uses_server_history_metrics(self):
        encoder = HierarchicalStateEncoder({"max_servers": 3, "max_ready_tasks": 4})
        state = self.sample_state()
        encoded = encoder.encode_high_state(state)
        server0 = encoded.server_features[0]
        self.assertEqual(server0[16], 0.0)
        self.assertGreater(server0[17], 0.0)
        self.assertGreaterEqual(server0[18], 0.0)
        self.assertGreater(server0[19], 0.0)
        self.assertEqual(server0[20], 0.0)

    def test_env_updates_server_history_metrics_after_assignment(self):
        workflow = Workflow(
            workflow_id="wf",
            arrival_time=0.0,
            absolute_deadline=100.0,
            tasks=(Task(0, workload_cycles=10.0), Task(1, workload_cycles=10.0)),
            edges=(DependencyEdge(0, 1, 0.0),),
            relative_deadline=100.0,
        )
        resources = {
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
                    "type": "edge",
                    "max_freq_hz": 1.0,
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
                    "type": "edge",
                    "max_freq_hz": 1.0,
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
        env = EdgeCloudWorkflowEnv(
            [workflow],
            resources,
            {"enable_backup_duplication": False, "server_history_ema_alpha": 0.5},
            seed=1,
        )
        _, _, _, info = env.step(
            {
                "workflow_id": "wf",
                "task_id": 0,
                "server_id": 0,
                "rho": 1.0,
                "backup": True,
                "backup_server_id": 1,
            }
        )
        self.assertFalse(info["event"]["has_backup"])
        self.assertEqual(info["event"]["backup_energy"], 0.0)
        self.assertFalse(any(row.get("replica_role") == "backup" for row in env.event_log))
        env._advance_to_next_decision_point()
        context = env.get_state()["decision_context"]
        self.assertFalse(context["backup_enabled"])
        self.assertFalse(context["backup_pair_mask"].any())
        selected_metrics = context["server_safety_metrics"]["0"]
        other_metrics = context["server_safety_metrics"]["1"]
        self.assertGreater(selected_metrics["recent_primary_assignments"], 0.0)
        self.assertGreater(selected_metrics["wait_time_ema"], 0.0)
        self.assertEqual(other_metrics["recent_primary_assignments"], 0.0)

    def test_primary_backup_action_legal(self):
        agent = HierarchicalSafeAgent(self.agent_config(), seed=1)
        state = self.sample_state()
        for _ in range(30):
            high, _, _ = agent.select_action(state, deterministic=False)
            self.assertTrue(state["decision_context"]["primary_server_mask"][high["primary_server_index"]])
            self.assertTrue(state["decision_context"]["backup_pair_mask"][high["primary_server_index"], high["backup_server_index"]])
            self.assertNotEqual(high["primary_server_index"], high["backup_server_index"])

    def test_sample_and_evaluate_log_probability_match(self):
        agent = HierarchicalSafeAgent(self.agent_config(), seed=2)
        state = self.sample_state()
        high, _, _ = agent.select_action(state, deterministic=False)
        evaluated = agent.high_agent.evaluate_action_log_prob(
            high["state"],
            high["primary_server_index"],
            high["backup_server_index"],
            high["primary_server_mask"],
            high["backup_pair_mask"],
        )
        self.assertLess(abs(high["log_prob"] - evaluated), 1e-5)

    def test_low_safe_action(self):
        agent = HierarchicalSafeAgent(self.agent_config(), seed=3)
        state = self.sample_state(feasible=True)
        for _ in range(20):
            _, low, _ = agent.select_action(state, deterministic=False)
            self.assertGreaterEqual(low["rho"], 0.65 - 1e-8)
            self.assertLessEqual(low["rho"], 1.0 + 1e-8)
        _, low_det, _ = agent.select_action(state, deterministic=True, low_deterministic=True)
        mean_u = low_det["rho_alpha"] / (low_det["rho_alpha"] + low_det["rho_beta"])
        self.assertAlmostEqual(low_det["rho"], 0.65 + (1.0 - 0.65) * mean_u, places=6)

    def test_infeasible_low_action_not_added_to_buffer(self):
        agent = HierarchicalSafeAgent(self.agent_config(), seed=4)
        state = self.sample_state(feasible=False)
        _, low, _ = agent.select_action(state, deterministic=False)
        buffer = LowRolloutBuffer()
        if low["actor_update_mask"]:
            buffer.add(state=low["state"], action_u=low["action_u"], old_log_prob=low["log_prob"], value=low["value"], reward=0.0, server_id=0)
        self.assertEqual(low["rho"], 1.0)
        self.assertFalse(low["actor_update_mask"])
        self.assertEqual(buffer.size, 0)

    def test_rho_safe_one_not_added_to_low_buffer(self):
        agent = HierarchicalSafeAgent(self.agent_config(), seed=14)
        state = self.sample_state(feasible=True)
        for metrics in state["decision_context"]["server_safety_metrics"].values():
            metrics["rho_safe"] = 1.0
        _, low, _ = agent.select_action(state, deterministic=False)
        buffer = LowRolloutBuffer()
        if low["actor_update_mask"]:
            buffer.add(state=low["state"], action_u=low["action_u"], old_log_prob=low["log_prob"], value=low["value"], reward=0.0, server_id=0)
        self.assertEqual(low["rho"], 1.0)
        self.assertFalse(low["has_action_span"])
        self.assertFalse(low["actor_update_mask"])
        self.assertEqual(buffer.size, 0)

    def test_rho_safe_near_one_float_boundary(self):
        agent = HierarchicalSafeAgent(self.agent_config(), seed=10)
        state = self.sample_state(feasible=True)
        rho_safe = 0.9984181174794319
        for metrics in state["decision_context"]["server_safety_metrics"].values():
            metrics["rho_safe"] = rho_safe
            metrics["low_action_feasible"] = True
        for _ in range(10):
            _, low, _ = agent.select_action(state, deterministic=False)
            self.assertGreaterEqual(low["rho"], rho_safe)
            self.assertLessEqual(low["rho"], 1.0)
            self.assertTrue(low["actor_update_mask"])

    def test_low_gamma_zero(self):
        buffer = LowRolloutBuffer()
        buffer.add(
            state=np.zeros(9),
            action_u=0.5,
            old_log_prob=0.0,
            value=0.2,
            reward=0.7,
            server_id=0,
            deadline_value=0.1,
            deadline_cost=5.0,
            deadline_limit=10.0,
        )
        returns, _ = buffer.compute_returns_and_advantages()
        self.assertTrue(np.allclose(returns, np.asarray(buffer.rewards, dtype=np.float32)))
        deadline_returns, deadline_advantages = buffer.compute_deadline_returns_and_advantages()
        self.assertTrue(np.allclose(deadline_returns, np.asarray([0.5], dtype=np.float32)))
        self.assertTrue(np.allclose(deadline_advantages, np.asarray([0.4], dtype=np.float32)))

    def test_low_network_outputs_deadline_value(self):
        agent = HierarchicalSafeAgent(self.agent_config(), seed=15)
        state = self.sample_state(feasible=True)
        _, low, _ = agent.select_action(state, deterministic=False)
        self.assertIn("deadline_value", low)
        self.assertTrue(math.isfinite(float(low["deadline_value"])))

    def test_p3o_relu(self):
        ratio = torch.ones(1)
        clipped = torch.ones(1)
        workflows = ["a"]
        adv = torch.zeros(1)
        negative = HighLevelP3OAgent._workflow_p3o_terms(ratio, clipped, np.array([0]), adv, adv, workflows, [0.2], [1.0], [0.1], [1.0])
        self.assertEqual(float(negative["deadline_penalty"]), 0.0)
        positive = HighLevelP3OAgent._workflow_p3o_terms(ratio, clipped, np.array([0]), adv, adv, workflows, [2.0], [1.0], [2.0], [1.0])
        self.assertGreater(float(positive["deadline_penalty"]), 0.0)
        self.assertGreaterEqual(float(positive["reliability_penalty"]), 0.0)

    def test_workflow_level_relu(self):
        ratio = torch.ones(2)
        clipped = torch.ones(2)
        adv = torch.zeros(2)
        metrics = HighLevelP3OAgent._workflow_p3o_terms(
            ratio, clipped, np.array([0, 1]), adv, adv, ["a", "b"], [2.0, 0.0], [1.0, 1.0], [2.0, 0.0], [1.0, 1.0]
        )
        self.assertAlmostEqual(float(metrics["deadline_penalty"]), 0.5, places=6)

    def test_reward_gae_is_grouped_by_workflow(self):
        def compute_advantages(b_reward):
            buffer = HighRolloutBuffer()
            self.add_high_transition(buffer, "A", 1.0, done=False)
            self.add_high_transition(buffer, "B", b_reward, done=False)
            self.add_high_transition(buffer, "A", 2.0, done=True)
            self.add_high_transition(buffer, "B", b_reward, done=True)
            _, advantages = buffer.compute_reward_returns_and_advantages(0.9, 1.0)
            return advantages
        base = compute_advantages(10.0)
        changed = compute_advantages(-10.0)
        self.assertTrue(np.allclose(base[[0, 2]], changed[[0, 2]]))

    def test_terminal_telescope(self):
        objective = HierarchicalObjective({"high_makespan_weight": 0.7, "high_low_reward_weight": 0.3})
        objective.previous_partial_makespan["w"] = 80.0
        objective.deadline_cost_sums["w"] = 80.0
        objective.high_makespan_component_sums["w"] = -0.7 * 0.8
        terminal = objective.compute_terminal_feedback(None, {"workflow_id": "w", "response_time": 105.0, "relative_deadline": 100.0})
        self.assertAlmostEqual(objective.deadline_cost_sums["w"], 105.0, places=6)
        self.assertAlmostEqual(objective.high_makespan_component_sums["w"], -0.7 * 0.8, places=6)
        self.assertAlmostEqual(terminal["deadline_cost_delta"], 25.0, places=6)
        self.assertAlmostEqual(terminal["high_reward_delta"], 0.0, places=6)
        self.assertAlmostEqual(terminal["final_partial_makespan"], 105.0, places=6)

    def test_episode_metrics_are_recomputed_after_terminal_patch(self):
        buffer = HighRolloutBuffer()
        mask = np.array([True, True, True], dtype=bool)
        pair = np.array([[False, True, True], [True, False, True], [True, True, False]], dtype=bool)
        buffer.add(
            state=np.zeros(self.high_state_dim_for_three_servers(), dtype=np.float32),
            primary_action=0,
            backup_action=1,
            old_log_prob=0.0,
            reward_value=0.0,
            deadline_value=0.0,
            reliability_value=0.0,
            reward=-0.8,
            deadline_cost=80.0,
            reliability_cost=0.1,
            deadline_limit=100.0,
            reliability_limit=1.0,
            partial_makespan=80.0,
            partial_log_risk=0.1,
            workflow_id="w",
            workflow_done=False,
            episode_done=False,
            next_state=None,
            primary_mask=mask,
            backup_pair_mask=pair,
            info={"high_low_reward_bonus": 0.0},
        )
        buffer.patch_last_workflow_transition("w", reward_delta=-0.25, deadline_cost_delta=25.0, partial_makespan=105.0)
        buffer.mark_workflow_done("w")
        buffer.validate()
        metrics = train_safe.build_episode_metrics_from_buffers(
            buffer,
            LowRolloutBuffer(),
            {"deadline_success_rate": 0.0, "reliability_success_rate": 1.0, "total_energy": 10.0, "avg_workflow_response_time": 1.0},
            [],
            {},
        )
        self.assertAlmostEqual(metrics["mean_high_reward"] * metrics["high_buffer_size"], metrics["total_high_reward"], places=8)
        self.assertAlmostEqual(metrics["mean_high_makespan_reward"] * metrics["high_buffer_size"], metrics["total_high_makespan_reward"], places=8)
        self.assertAlmostEqual(metrics["mean_deadline_cost"] * metrics["high_buffer_size"], metrics["total_deadline_cost"], places=8)
        self.assertAlmostEqual(metrics["total_high_reward"], -1.05, places=8)
        self.assertAlmostEqual(metrics["total_deadline_cost"], 105.0, places=8)

    def test_interleaved_workflows_keep_own_deadline_limits(self):
        buffer = HighRolloutBuffer()
        self.add_high_cost_transition(buffer, "A", deadline_cost=20.0, deadline_limit=100.0, partial_makespan=20.0)
        self.add_high_cost_transition(buffer, "B", deadline_cost=50.0, deadline_limit=200.0, partial_makespan=50.0)
        self.add_high_cost_transition(buffer, "A", deadline_cost=30.0, deadline_limit=100.0, partial_makespan=50.0)
        self.add_high_cost_transition(buffer, "B", deadline_cost=70.0, deadline_limit=200.0, partial_makespan=120.0)
        self.add_high_cost_transition(buffer, "A", deadline_cost=30.0, deadline_limit=100.0, partial_makespan=80.0, done=True)
        self.add_high_cost_transition(buffer, "B", deadline_cost=100.0, deadline_limit=200.0, partial_makespan=220.0, done=True)
        buffer.validate()
        groups = dict(buffer.workflow_groups())
        self.assertAlmostEqual(sum(buffer.deadline_costs[index] for index in groups["A"]), 80.0, places=8)
        self.assertAlmostEqual(sum(buffer.deadline_costs[index] for index in groups["B"]), 220.0, places=8)
        self.assertLessEqual(buffer.partial_makespans[groups["A"][-1]], buffer.deadline_limits[groups["A"][-1]])
        self.assertGreater(buffer.partial_makespans[groups["B"][-1]], buffer.deadline_limits[groups["B"][-1]])

        def deadline_margin_for(indices):
            ratio = torch.ones(len(indices), dtype=torch.float32)
            clipped = torch.ones(len(indices), dtype=torch.float32)
            zeros = torch.zeros(len(indices), dtype=torch.float32)
            metrics = HighLevelP3OAgent._workflow_p3o_terms(
                ratio,
                clipped,
                np.asarray(indices, dtype=np.int64),
                zeros,
                zeros,
                buffer.workflow_ids,
                buffer.partial_makespans,
                buffer.deadline_limits,
                buffer.partial_log_risks,
                buffer.reliability_limits,
            )
            return float(metrics["deadline_margin"])

        self.assertAlmostEqual(deadline_margin_for(groups["A"]), -20.0, places=8)
        self.assertAlmostEqual(deadline_margin_for(groups["B"]), 20.0, places=8)

    def test_normalized_deadline_training_stream_keeps_raw_buffer_values(self):
        buffer = HighRolloutBuffer()
        self.add_high_cost_transition(buffer, "A", deadline_cost=20.0, deadline_limit=100.0, partial_makespan=20.0)
        self.add_high_cost_transition(buffer, "A", deadline_cost=60.0, deadline_limit=100.0, partial_makespan=80.0, done=True)
        buffer.validate()
        returns, advantages = buffer.compute_normalized_deadline_returns_and_advantages(1.0, 1.0)
        normalized_partials = buffer.normalized_partial_makespans()
        self.assertAlmostEqual(sum(buffer.deadline_costs), 80.0, places=8)
        self.assertAlmostEqual(buffer.partial_makespans[-1], 80.0, places=8)
        self.assertAlmostEqual(buffer.deadline_limits[-1], 100.0, places=8)
        self.assertAlmostEqual(float(returns[0]), 0.8, places=6)
        self.assertAlmostEqual(float(returns[1]), 0.6, places=6)
        self.assertAlmostEqual(float(advantages[0]), 0.8, places=6)
        self.assertAlmostEqual(float(normalized_partials[-1]), 0.8, places=6)

    def test_terminal_patch_uses_workflow_last_transition_not_global_last(self):
        buffer = HighRolloutBuffer()
        self.add_high_cost_transition(buffer, "B", deadline_cost=50.0, deadline_limit=200.0, partial_makespan=50.0)
        self.add_high_cost_transition(buffer, "B", deadline_cost=70.0, deadline_limit=200.0, partial_makespan=120.0)
        self.add_high_cost_transition(buffer, "A", deadline_cost=20.0, deadline_limit=100.0, partial_makespan=20.0)

        patched = buffer.patch_last_workflow_transition(
            "B",
            reward_delta=-0.1,
            deadline_cost_delta=100.0,
            partial_makespan=220.0,
            info_updates={"workflow_id": "B", "final_partial_makespan": 220.0},
        )
        self.assertTrue(patched)
        self.assertAlmostEqual(buffer.deadline_costs[1], 170.0, places=8)
        self.assertAlmostEqual(buffer.partial_makespans[1], 220.0, places=8)
        self.assertAlmostEqual(buffer.deadline_costs[2], 20.0, places=8)
        self.assertAlmostEqual(buffer.partial_makespans[2], 20.0, places=8)
        self.assertTrue(buffer.mark_workflow_done("B"))
        self.assertFalse(buffer.workflow_dones[2])

    def test_train_and_eval_use_multiline_log_formatter(self):
        row = {
            "episode": 1,
            "training_mode": "joint",
            "update_mode": "episode_end",
            "high_buffer_size": 2,
            "low_buffer_size": 1,
            "mean_high_reward": -0.5,
            "total_high_reward": -1.0,
            "mean_deadline_cost": 0.5,
            "total_deadline_cost": 1.0,
            "deadline_success_rate": 1.0,
            "primary_selection_counts": {"0": 2},
        }
        train_out = StringIO()
        eval_out = StringIO()
        with redirect_stdout(train_out):
            train_safe.print_metrics_block("train", row)
        with redirect_stdout(eval_out):
            train_safe.print_metrics_block("eval", row)
        self.assertGreaterEqual(train_out.getvalue().count("\n"), 4)
        self.assertIn("reward", train_out.getvalue())
        self.assertIn("constraints", train_out.getvalue())
        self.assertEqual(train_out.getvalue().replace("[train]", "[eval]", 1), eval_out.getvalue())

    def test_shield_flags_do_not_change_high_reward(self):
        objective = HierarchicalObjective({"high_makespan_weight": 0.7, "high_low_reward_weight": 0.3})
        env = FakeObjectiveEnv()
        state = {
            "decision_context": {}
        }
        action = {"workflow_id": "w", "task_id": 0, "server_id": 0, "primary_energy_ref_rho1": 100.0}
        info = {"event": {"energy": 80.0, "primary_energy": 80.0, "backup_energy": 0.0}}
        reward_a = objective.compute_step_feedback(env, state, {"shield_relaxed": False, "backup_shield_relaxed": False}, {}, action, -80.0, info)["high_reward"]
        objective.reset_episode()
        env = FakeObjectiveEnv()
        reward_b = objective.compute_step_feedback(env, state, {"shield_relaxed": True, "backup_shield_relaxed": True}, {}, action, -80.0, info)["high_reward"]
        self.assertAlmostEqual(reward_a, reward_b, places=12)

    def test_high_low_bonus_uses_normalized_energy_saving(self):
        objective = HierarchicalObjective({"high_makespan_weight": 0.7, "high_low_reward_weight": 0.3})
        state = {"decision_context": {}}
        env = FakeObjectiveEnv(task_count=1)
        feedback = objective.compute_step_feedback(
            env,
            state,
            {},
            {},
            {"workflow_id": "w", "task_id": 0, "server_id": 0, "primary_energy_ref_rho1": 100.0},
            -80.0,
            {"event": {"energy": 80.0, "primary_energy": 80.0, "backup_energy": 0.0}},
        )
        self.assertAlmostEqual(feedback["low_reward"], 20.0, places=8)
        self.assertAlmostEqual(feedback["low_energy_saving_ratio"], 0.2, places=8)
        self.assertAlmostEqual(feedback["high_low_reward_bonus"], 0.3 * 0.2, places=8)

    def test_high_low_bonus_is_spread_over_workflow_tasks(self):
        objective = HierarchicalObjective({"high_makespan_weight": 0.7, "high_low_reward_weight": 0.3})
        state = {"decision_context": {}}
        env = FakeObjectiveEnv(task_count=4)
        feedback = objective.compute_step_feedback(
            env,
            state,
            {},
            {},
            {"workflow_id": "w", "task_id": 0, "server_id": 0, "primary_energy_ref_rho1": 100.0},
            -80.0,
            {"event": {"energy": 80.0, "primary_energy": 80.0, "backup_energy": 0.0}},
        )
        self.assertAlmostEqual(feedback["low_energy_saving_ratio"], 0.2, places=8)
        self.assertAlmostEqual(feedback["workflow_task_count"], 4.0, places=8)
        self.assertAlmostEqual(feedback["high_low_reward_bonus"], 0.3 * 0.2 / 4.0, places=8)

    def test_high_time_reward_uses_fastest_complete_time_not_exec_time(self):
        objective = HierarchicalObjective({"high_makespan_weight": 0.7, "high_low_reward_weight": 0.3})
        env = FakeObjectiveEnv(task_count=1)
        env.workflow_by_id["w"]["task_finish_times"][0] = 12.0
        state = {
            "servers": [{"id": 0}, {"id": 1}, {"id": 2}],
            "decision_context": {
                "primary_server_mask": np.array([True, True, False], dtype=bool),
                "server_safety_metrics": {
                    "0": {
                        "estimated_ast": 10.0,
                        "execution_time_rho_1": 1.0,
                        "estimated_finish_time": 11.0,
                        "current_time": 0.0,
                    },
                    "1": {
                        "estimated_ast": 0.0,
                        "execution_time_rho_1": 5.0,
                        "estimated_finish_time": 5.0,
                        "current_time": 0.0,
                    },
                    "2": {
                        "estimated_ast": 0.0,
                        "execution_time_rho_1": 1.0,
                        "estimated_finish_time": 1.0,
                        "current_time": 0.0,
                    },
                },
            },
        }
        feedback = objective.compute_step_feedback(
            env,
            state,
            {},
            {},
            {"workflow_id": "w", "task_id": 0, "server_id": 0, "primary_energy_ref_rho1": 100.0},
            -100.0,
            {
                "event": {
                    "task_ready_time": 0.0,
                    "final_aft": 12.0,
                    "energy": 100.0,
                    "primary_energy": 100.0,
                    "backup_energy": 0.0,
                }
            },
        )
        self.assertAlmostEqual(feedback["task_fastest_time"], 5.0, places=8)
        self.assertAlmostEqual(feedback["task_actual_time"], 12.0, places=8)
        self.assertAlmostEqual(feedback["task_time_regret"], (12.0 - 5.0) / 5.0, places=8)
        self.assertAlmostEqual(feedback["task_complete_time_ratio"], 12.0 / 5.0, places=8)
        self.assertAlmostEqual(feedback["normalized_makespan_increment"], 12.0 / 100.0, places=8)
        self.assertAlmostEqual(feedback["current_slack_ratio"], (100.0 - 12.0) / 100.0, places=8)
        self.assertAlmostEqual(feedback["high_task_time_reward"], 0.7 * ((100.0 - 12.0) / 100.0), places=8)

    def test_high_reward_excludes_high_level_energy_cost(self):
        objective = HierarchicalObjective({"high_makespan_weight": 0.7, "high_low_reward_weight": 0.3})
        state = {"decision_context": {}}
        env = FakeObjectiveEnv(task_count=1)
        feedback = objective.compute_step_feedback(
            env,
            state,
            {},
            {},
            {"workflow_id": "w", "task_id": 0, "server_id": 0, "primary_energy_ref_rho1": 10.0},
            -10.0,
            {"event": {"energy": 10.0, "primary_energy": 10.0, "backup_energy": 0.0}},
        )
        self.assertNotIn("high_energy_reward", feedback)
        self.assertAlmostEqual(
            feedback["high_reward"],
            feedback["high_makespan_reward"] + feedback["high_low_reward_bonus"],
            places=8,
        )

    def test_training_eval_scheduler_action_consistent(self):
        config = self.agent_config()
        state = self.sample_state()
        agent = HierarchicalSafeAgent(config, seed=5)
        scheduler = HierarchicalSafeScheduler(params={"agent": config, "deterministic": True}, seed=5)
        scheduler.agent.high_agent.network.load_state_dict(agent.high_agent.network.state_dict())
        scheduler.agent.low_agent.network.load_state_dict(agent.low_agent.network.state_dict())
        _, _, eval_action = agent.select_action(state, deterministic=True, low_deterministic=True)
        sched_action = scheduler.select_action(state)
        self.assertEqual(eval_action["server_id"], sched_action["server_id"])
        self.assertEqual(eval_action["backup_server_id"], sched_action["backup_server_id"])
        self.assertAlmostEqual(eval_action["rho"], sched_action["rho"], places=8)

    def test_checkpoint_resume(self):
        config = self.agent_config()
        agent = HierarchicalSafeAgent(config, seed=6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            agent.save_checkpoint(path, episode=7, training_phase="joint", config=config)
            loaded = HierarchicalSafeAgent(config, seed=8)
            payload = loaded.load_checkpoint(path)
            for key, value in agent.high_agent.network.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.high_agent.network.state_dict()[key]))
            self.assertIn("state", loaded.high_agent.optimizer.state_dict())
            self.assertEqual(payload["episode"], 7)
            self.assertEqual(payload["training_phase"], "joint")


if __name__ == "__main__":
    unittest.main()
