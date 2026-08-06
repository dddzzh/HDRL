from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.a2c.a2c import Observation
from src.D3QN.hierarchical_d3qn_agent import HighLevelD3QNAgent
from src.D3QN.reward import HierarchicalObjective
from src.sac.hierarchical_low_sac import LowLevelDVFSContinuousSACAgent, LowSACReplayBuffer
from src.sac.reward import build_low_sac_reward_components
from tools.train_hierarchical_d3qn import _normalize_high_algorithm, _select_heft_action


class FakeHighRewardEnv:
    def __init__(self, reliability, task_count=1):
        self.workflow_by_id = {
            "w": {
                "workflow_id": "w",
                "arrival_time": 0.0,
                "task_finish_times": {0: 1.0},
                "reliability_product": float(reliability),
                "workflow": {
                    "workflow_id": "w",
                    "tasks": [{"id": index} for index in range(int(task_count))],
                },
            }
        }

    def workflow_relative_deadline(self, workflow_state):
        return 10.0

    def _workflow_reliability_requirement(self, workflow_state):
        return 0.9


class FakeLowRewardEnv:
    def __init__(self):
        self.workflow_by_id = {"w": {"workflow_id": "w"}}

    def _workflow_reliability_requirement(self, workflow_state):
        return 0.0


@unittest.skipIf(torch is None, "torch is required")
class TestHierarchicalSAC(unittest.TestCase):
    def agent(self):
        return LowLevelDVFSContinuousSACAgent(
            {
                "device": "cpu",
                "hidden_dim": 16,
                "hidden_layers": 1,
                "action_dim": 1,
                "batch_size": 2,
                "automatic_entropy_tuning": False,
            },
            seed=7,
        )

    def test_replay_uses_standard_zscore_statistics(self):
        replay = LowSACReplayBuffer(2, max_size=4, batch_size=2)
        for value in (
            {"energy_saving": 1, "time_increase": 2, "deadline_violation": 0, "reliability_violation": 0},
            {"energy_saving": 3, "time_increase": 6, "deadline_violation": 2, "reliability_violation": 4},
        ):
            replay.store_transition(
                np.zeros(2),
                0.37,
                {**value, "deadline_crossed": 0, "reliability_crossed": 0},
                np.zeros(2),
                False,
            )
        statistics = replay.reward_statistics(std_floor=0.01)
        np.testing.assert_allclose(statistics["mean"], [2, 4, 1, 2])
        np.testing.assert_allclose(statistics["std"], [1, 2, 1, 2])

    def test_reward_combines_zscores_and_hard_penalties(self):
        agent = self.agent()
        components = torch.tensor([[3.0, 6.0, 2.0, 4.0]])
        flags = torch.tensor([[1.0, 1.0]])
        statistics = {
            "mean": np.array([2, 4, 1, 2], dtype=np.float32),
            "std": np.array([1, 2, 1, 2], dtype=np.float32),
        }
        reward = agent._compose_rewards(components, flags, statistics)
        self.assertAlmostEqual(float(reward.item()), -14.0)

    def test_energy_reward_component_uses_absolute_joule_saving(self):
        components = build_low_sac_reward_components(
            FakeLowRewardEnv(),
            {"workflow_id": "w", "task_id": 0},
            {"event": {"energy": 60.0, "final_aft": 2.0, "final_reliability": 1.0}},
            {
                "workflow_relative_deadline": 10.0,
                "total_energy": 100.0,
                "final_aft": 1.0,
                "task_latest_finish_time": 10.0,
                "final_reliability": 1.0,
            },
        )
        self.assertAlmostEqual(components["energy_saving"], 40.0)
        self.assertAlmostEqual(components["energy_saved"], 40.0)
        self.assertAlmostEqual(components["energy_saving_ratio"], 0.4)

    def test_fixed_alpha_disables_entropy_optimizer(self):
        agent = self.agent()
        self.assertFalse(agent.automatic_entropy_tuning)
        self.assertAlmostEqual(agent.alpha, 0.2)
        self.assertIsNone(agent.log_alpha)
        self.assertIsNone(agent.alpha_optimizer)

    def test_heft_high_switch_keeps_only_sac_frequency_action(self):
        class FakeHighAgent:
            max_servers = 2

            def build_high_state(self, env_state):
                return Observation(
                    ready_task_features=np.zeros((2, 3), dtype=np.float32),
                    ready_task_mask=np.ones(2, dtype=bool),
                    server_features=np.zeros((2, 21), dtype=np.float32),
                    server_mask=np.ones(2, dtype=bool),
                    global_features=np.zeros(5, dtype=np.float32),
                    ready_tasks=tuple(env_state["ready_tasks"]),
                    servers=({"id": 3}, {"id": 7}),
                    rho_levels=(),
                    vector_override=np.zeros(8, dtype=np.float32),
                )

            def _context_masks(self, context):
                return (
                    np.ones_like(context["primary_server_mask"], dtype=bool),
                    np.asarray(context["backup_pair_mask"], dtype=bool).copy(),
                )

        class FakeLowAgent:
            def select_frequency(self, observation, task_index, server_index, **kwargs):
                self.selected = (task_index, server_index, kwargs["safety_metrics"])
                return {"rho": 0.63, "state": np.zeros(1), "action_u": 0.5}

        context = {
            "primary_server_mask": np.array([True, False]),
            "backup_pair_mask": np.zeros((2, 2), dtype=bool),
            "server_safety_metrics": {"7": {"rho_hardware_min": 0.2}},
        }
        env = type(
            "FakeEnv",
            (),
            {"build_hierarchical_decision_context": lambda self, workflow_id, task_id: context},
        )()
        scheduler = type(
            "FakeHEFT",
            (),
            {
                "select_action": lambda self, state, context_builder=None: {
                    "workflow_id": "w",
                    "task_id": 1,
                    "server_id": 7,
                    "algorithm": "heft_dag_edge_cloud_eft",
                    "upward_rank": 2.0,
                    "estimated_finish_time": 3.0,
                    "online_queue_aware": context_builder is not None,
                }
            },
        )()
        agent = type("FakeAgent", (), {"high_agent": FakeHighAgent(), "low_agent": FakeLowAgent()})()
        state = {
            "ready_tasks": [
                {"workflow_id": "w", "task_id": 0},
                {"workflow_id": "w", "task_id": 1},
            ]
        }

        decision_state, high_decision, _, env_action = _select_heft_action(
            env, agent, state, scheduler, low_deterministic=False
        )

        self.assertEqual(_normalize_high_algorithm("HEFT"), "heft")
        self.assertEqual(decision_state["ready_tasks"][0]["task_id"], 1)
        self.assertEqual(high_decision["primary_server_index"], 1)
        self.assertIsNone(high_decision["backup_server_id"])
        self.assertTrue(high_decision["primary_server_mask"].all())
        self.assertEqual(high_decision["action_mask_mode"], "structural_only")
        self.assertTrue(high_decision["heft_online_queue_aware"])
        self.assertNotIn("heft_primary_mask_violation", high_decision)
        self.assertFalse(env_action["backup"])
        self.assertAlmostEqual(env_action["rho"], 0.63)
        self.assertEqual(agent.low_agent.selected[:2], (0, 1))

    def test_frequency_action_is_direct_rho_and_not_clamped_by_rho_safe(self):
        agent = self.agent()
        with torch.no_grad():
            for parameter in agent.actor.parameters():
                parameter.zero_()
            desired_rho = 0.37
            desired_unit_action = (desired_rho - agent.rho_min) / (1.0 - agent.rho_min)
            agent.actor.mean_head.bias[0] = np.arctanh(2.0 * desired_unit_action - 1.0)
        observation = Observation(
            ready_task_features=np.zeros((1, 3), dtype=np.float32),
            ready_task_mask=np.ones(1, dtype=bool),
            server_features=np.zeros((1, 21), dtype=np.float32),
            server_mask=np.ones(1, dtype=bool),
            global_features=np.zeros(5, dtype=np.float32),
            ready_tasks=({"workload_cycles": 1.0, "relative_deadline": 10.0},),
            servers=({"min_rho": 0.2},),
            rho_levels=(),
        )
        decision = agent.select_frequency(
            observation,
            0,
            0,
            deterministic=True,
            safety_metrics={
                "rho_hardware_min": 0.2,
                "rho_safe": 0.8,
                "low_action_feasible": False,
                "workflow_relative_deadline": 10.0,
                "workflow_absolute_deadline": 10.0,
            },
        )
        self.assertAlmostEqual(decision["action_rho"], desired_rho, places=6)
        self.assertAlmostEqual(decision["action_u"], desired_rho, places=6)
        self.assertAlmostEqual(decision["rho"], desired_rho, places=6)
        self.assertTrue(decision["actor_update_mask"])

    def test_stochastic_action_is_continuous_rho_in_hardware_interval(self):
        agent = self.agent()
        observation = Observation(
            ready_task_features=np.zeros((1, 3), dtype=np.float32),
            ready_task_mask=np.ones(1, dtype=bool),
            server_features=np.zeros((1, 21), dtype=np.float32),
            server_mask=np.ones(1, dtype=bool),
            global_features=np.zeros(5, dtype=np.float32),
            ready_tasks=({"workload_cycles": 1.0, "relative_deadline": 10.0},),
            servers=({"min_rho": 0.2},),
            rho_levels=(),
        )
        actions = [
            agent.select_frequency(
                observation,
                0,
                0,
                deterministic=False,
                safety_metrics={"rho_hardware_min": 0.2},
            )["action_u"]
            for _ in range(16)
        ]
        self.assertTrue(all(0.2 <= action <= 1.0 for action in actions))
        self.assertTrue(any(abs(action * 10.0 - round(action * 10.0)) > 1e-4 for action in actions))

    def test_high_d3qn_outputs_only_primary_and_selects_reliable_backup(self):
        agent = HighLevelD3QNAgent(
            {
                "device": "cpu",
                "hidden_dim": 16,
                "hidden_layers": 1,
                "state": {"max_servers": 3},
            },
            seed=11,
        )
        with torch.no_grad():
            for parameter in agent.agent_R.q_eval.parameters():
                parameter.zero_()
        observation = Observation(
            ready_task_features=np.zeros((1, 3), dtype=np.float32),
            ready_task_mask=np.ones(1, dtype=bool),
            server_features=np.zeros((3, 21), dtype=np.float32),
            server_mask=np.ones(3, dtype=bool),
            global_features=np.zeros(5, dtype=np.float32),
            ready_tasks=({"workflow_id": "w", "task_id": 0},),
            servers=({"id": 0}, {"id": 1}, {"id": 2}),
            rho_levels=(),
            vector_override=np.zeros(agent.encoder.high_state_dim, dtype=np.float32),
        )
        agent.build_high_state = lambda env_state: observation
        backup_pair_mask = np.array(
            [
                [False, True, True],
                [True, False, True],
                [True, True, False],
            ],
            dtype=bool,
        )
        decision = agent.select_assignment(
            {
                "decision_context": {
                    "primary_server_mask": np.ones(3, dtype=bool),
                    "backup_pair_mask": backup_pair_mask,
                    "server_safety_metrics": {
                        "0": {"task_reliability_rho_1": 0.90},
                        "1": {
                            "task_reliability_rho_1": 0.95,
                            "estimated_ast": 1.0,
                            "execution_time_rho_1": 1.0,
                            "primary_energy_ref_rho1": 5.0,
                        },
                        "2": {
                            "task_reliability_rho_1": 0.99,
                            "estimated_ast": 8.0,
                            "execution_time_rho_1": 1.0,
                            "primary_energy_ref_rho1": 20.0,
                        },
                    },
                }
            },
            deterministic=True,
        )
        self.assertEqual(agent.action_dim, 3)
        self.assertEqual(decision["primary_server_index"], 0)
        self.assertEqual(decision["primary_action"], 0)
        self.assertEqual(decision["backup_server_index"], 2)
        self.assertEqual(decision["backup_selector"], "reliability_then_finish_time")
        self.assertNotIn("joint_action", decision)

    def test_high_d3qn_disabled_backup_returns_primary_only(self):
        agent = HighLevelD3QNAgent(
            {
                "device": "cpu",
                "hidden_dim": 16,
                "hidden_layers": 1,
                "backup_selection": "disabled",
                "state": {"max_servers": 2},
            },
            seed=12,
        )
        with torch.no_grad():
            for parameter in agent.agent_R.q_eval.parameters():
                parameter.zero_()
        observation = Observation(
            ready_task_features=np.zeros((1, 3), dtype=np.float32),
            ready_task_mask=np.ones(1, dtype=bool),
            server_features=np.zeros((2, 21), dtype=np.float32),
            server_mask=np.ones(2, dtype=bool),
            global_features=np.zeros(5, dtype=np.float32),
            ready_tasks=({"workflow_id": "w", "task_id": 0},),
            servers=({"id": 0}, {"id": 1}),
            rho_levels=(),
            vector_override=np.zeros(agent.encoder.high_state_dim, dtype=np.float32),
        )
        agent.build_high_state = lambda env_state: observation
        decision = agent.select_assignment(
            {
                "decision_context": {
                    "primary_server_mask": np.zeros(2, dtype=bool),
                    "backup_pair_mask": np.zeros((2, 2), dtype=bool),
                    "backup_enabled": False,
                    "server_safety_metrics": {},
                }
            },
            deterministic=True,
        )
        self.assertEqual(decision["primary_server_index"], 0)
        self.assertTrue(decision["primary_server_mask"].all())
        self.assertEqual(decision["action_mask_mode"], "structural_only")
        self.assertIsNone(decision["backup_server_index"])
        self.assertIsNone(decision["backup_server_id"])
        self.assertFalse(decision["effective_backup_mask"].any())
        self.assertFalse(decision["backup_shield_relaxed"])

    def test_high_reward_applies_reliability_crossing_penalty(self):
        objective = HierarchicalObjective(
            {
                "high_makespan_weight": 1.0,
                "high_low_reward_weight": 0.0,
                "high_reliability_fixed_penalty": 5.0,
                "high_reliability_severity_weight": 2.0,
            }
        )
        env = FakeHighRewardEnv(reliability=0.85)
        feedback = objective.compute_step_feedback(
            env,
            {"servers": [], "decision_context": {}},
            {},
            {},
            {"workflow_id": "w", "task_id": 0, "server_id": 0, "primary_energy_ref_rho1": 1.0},
            -1.0,
            {"event": {"energy": 1.0, "primary_energy": 1.0, "task_ready_time": 0.0, "final_aft": 1.0}},
        )
        self.assertEqual(feedback["reliability_crossed"], 1.0)
        self.assertGreater(feedback["reliability_violation_delta"], 0.0)
        self.assertGreater(feedback["high_reliability_penalty"], 5.0)
        self.assertAlmostEqual(
            feedback["high_reward"],
            feedback["high_reward_base"] - feedback["high_reliability_penalty"],
        )

    def test_high_energy_reward_uses_cross_server_absolute_energy_scale(self):
        objective = HierarchicalObjective(
            {
                "high_makespan_weight": 0.0,
                "high_energy_weight": 1.0,
                "high_reliability_fixed_penalty": 5.0,
                "high_reliability_severity_weight": 2.0,
            }
        )
        state = {
            "servers": [],
            "decision_context": {"task_energy_scale": 100.0},
        }

        efficient = objective.compute_step_feedback(
            FakeHighRewardEnv(reliability=1.0),
            state,
            {},
            {},
            {
                "workflow_id": "w",
                "task_id": 0,
                "server_id": 0,
                "primary_energy_ref_rho1": 50.0,
            },
            -45.0,
            {
                "event": {
                    "energy": 45.0,
                    "primary_energy": 45.0,
                    "task_ready_time": 0.0,
                    "final_aft": 1.0,
                }
            },
        )
        objective.reset_episode()
        inefficient = objective.compute_step_feedback(
            FakeHighRewardEnv(reliability=1.0),
            state,
            {},
            {},
            {
                "workflow_id": "w",
                "task_id": 0,
                "server_id": 1,
                "primary_energy_ref_rho1": 400.0,
            },
            -360.0,
            {
                "event": {
                    "energy": 360.0,
                    "primary_energy": 360.0,
                    "task_ready_time": 0.0,
                    "final_aft": 1.0,
                }
            },
        )

        self.assertAlmostEqual(efficient["low_energy_saving_ratio"], 0.1)
        self.assertAlmostEqual(inefficient["low_energy_saving_ratio"], 0.1)
        self.assertAlmostEqual(efficient["high_energy_reward_raw"], 0.55)
        self.assertAlmostEqual(inefficient["high_energy_reward_raw"], -2.6)
        self.assertGreater(efficient["high_reward"], inefficient["high_reward"])
        self.assertEqual(efficient["high_energy_reward"], efficient["high_low_reward_bonus"])

    def test_high_reward_is_divided_by_workflow_task_count(self):
        objective = HierarchicalObjective(
            {
                "high_makespan_weight": 0.7,
                "high_energy_weight": 0.3,
            }
        )
        feedback = objective.compute_step_feedback(
            FakeHighRewardEnv(reliability=1.0, task_count=4),
            {"servers": [], "decision_context": {"task_energy_scale": 100.0}},
            {},
            {},
            {
                "workflow_id": "w",
                "task_id": 0,
                "server_id": 0,
                "primary_energy_ref_rho1": 100.0,
            },
            -80.0,
            {
                "event": {
                    "energy": 80.0,
                    "primary_energy": 80.0,
                    "task_ready_time": 0.0,
                    "final_aft": 1.0,
                }
            },
        )
        self.assertEqual(feedback["workflow_task_count"], 4.0)
        self.assertAlmostEqual(feedback["high_makespan_reward"], 0.7 * 0.9 / 4.0)
        self.assertAlmostEqual(feedback["high_energy_reward"], 0.3 * 0.2 / 4.0)
        self.assertAlmostEqual(feedback["high_reward"], (0.7 * 0.9 + 0.3 * 0.2) / 4.0)

    def test_incremental_high_reward_normalizes_only_energy_by_task_count(self):
        objective = HierarchicalObjective(
            {
                "time_reward_mode": "incremental_cost",
                "high_makespan_weight": 0.4,
                "high_energy_weight": 0.6,
                "constraint_penalty": {
                    "crossing_weight": 5.0,
                    "severity_weight": 2.0,
                },
            }
        )
        feedback = objective.compute_step_feedback(
            FakeHighRewardEnv(reliability=1.0, task_count=4),
            {"servers": [], "decision_context": {"task_energy_scale": 100.0}},
            {},
            {},
            {
                "workflow_id": "w",
                "task_id": 0,
                "server_id": 0,
                "primary_energy_ref_rho1": 100.0,
            },
            -80.0,
            {
                "event": {
                    "energy": 80.0,
                    "primary_energy": 80.0,
                    "task_ready_time": 0.0,
                    "final_aft": 1.0,
                }
            },
        )
        self.assertAlmostEqual(feedback["normalized_makespan_increment"], 0.1)
        self.assertAlmostEqual(feedback["high_makespan_reward"], -0.4 * 0.1)
        self.assertAlmostEqual(feedback["high_energy_reward"], 0.6 * 0.2 / 4.0)
        self.assertAlmostEqual(feedback["high_reward"], -0.4 * 0.1 + 0.6 * 0.2 / 4.0)

        terminal = objective.compute_terminal_feedback(
            None,
            {
                "workflow_id": "w",
                "response_time": 12.0,
                "relative_deadline": 10.0,
            },
        )
        self.assertAlmostEqual(terminal["high_makespan_reward_delta"], -0.4 * 1.1)
        self.assertEqual(terminal["high_deadline_crossed"], 1.0)
        self.assertAlmostEqual(terminal["high_deadline_violation"], 0.2)
        self.assertAlmostEqual(terminal["high_deadline_penalty"], 5.0 + 2.0 * 0.2)
        self.assertAlmostEqual(
            terminal["high_reward_delta"],
            -0.4 * 1.1 - (5.0 + 2.0 * 0.2),
        )
        self.assertAlmostEqual(objective.high_makespan_component_sums["w"], -0.4 * 1.2)


if __name__ == "__main__":
    unittest.main()
