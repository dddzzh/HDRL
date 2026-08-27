from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.D3QN.deadline_lagrangian import DeadlineLagrangianController
from src.D3QN.D3QN import D3QN
from src.D3QN.hierarchical_d3qn_agent import (
    HighLevelD3QNAgent,
    HierarchicalD3QNAgent,
)
from src.D3QN.reward import HierarchicalObjective
from src.ppo.hierarchical_rollout import HighRolloutBuffer
from tools.hierarchical_common import _patch_terminal
from tools.train_hierarchical_d3qn import (
    _apply_cooperative_workflow_credit,
    _cumulative_low_deadline_credit,
    _low_causal_deadline_misses,
)


class FakeLagrangianEnv:
    def __init__(self, partial_makespan, task_count=1):
        self.workflow_by_id = {
            "w": {
                "workflow_id": "w",
                "arrival_time": 0.0,
                "task_finish_times": {0: float(partial_makespan)},
                "reliability_product": 1.0,
                "workflow": {
                    "workflow_id": "w",
                    "tasks": [
                        {"id": index} for index in range(int(task_count))
                    ],
                },
            }
        }

    def workflow_relative_deadline(self, workflow_state):
        return 100.0

    def _workflow_reliability_requirement(self, workflow_state):
        return 0.9


class TestDeadlineLagrangian(unittest.TestCase):
    def test_low_dual_uses_binary_causal_workflow_misses(self):
        costs = _low_causal_deadline_misses(
            ["miss_due_to_dvfs", "upper_only_miss", "met_deadline"],
            np.asarray([1.0, 1.0, 0.0]),
            {
                "miss_due_to_dvfs": 1.0,
                "upper_only_miss": 0.0,
                "met_deadline": 1.0,
            },
        )
        self.assertEqual(costs, [1.0, 0.0, 0.0])

    def test_cumulative_low_credit_detects_only_recoverable_miss(self):
        results = [
            {
                "workflow_id": "low_caused",
                "response_time": 105.0,
                "relative_deadline": 100.0,
            },
            {
                "workflow_id": "upper_caused",
                "response_time": 130.0,
                "relative_deadline": 100.0,
            },
            {
                "workflow_id": "on_time",
                "response_time": 95.0,
                "relative_deadline": 100.0,
            },
        ]
        transitions = [
            {
                "workflow_id": "low_caused",
                "reward_components": {
                    "workflow_makespan_delay_seconds": 2.0
                },
            },
            {
                "workflow_id": "low_caused",
                "reward_components": {
                    "workflow_makespan_delay_seconds": 4.0
                },
            },
            {
                "workflow_id": "upper_caused",
                "reward_components": {
                    "workflow_makespan_delay_seconds": 10.0
                },
            },
            {
                "workflow_id": "on_time",
                "reward_components": {
                    "workflow_makespan_delay_seconds": 20.0
                },
            },
        ]
        costs, details = _cumulative_low_deadline_credit(results, transitions)
        self.assertEqual(costs, [1.0, 0.0, 0.0])
        self.assertAlmostEqual(
            details["low_caused"]["counterfactual_response_time"], 99.0
        )
        self.assertAlmostEqual(
            details["upper_caused"]["counterfactual_response_time"], 120.0
        )

    def controller(self, **overrides):
        values = {
            "lambda_init": 1.0,
            "dual_lr": 0.1,
            "budget": 0.01,
            "lambda_max": 100.0,
            "warmup_episodes": 0,
        }
        values.update(overrides)
        return DeadlineLagrangianController(**values)

    def test_deadline_cost_telescopes(self):
        partial = [0.0, 20.0, 20.0, 50.0, 80.0]
        costs = [
            max(0.0, after - before) / 100.0
            for before, after in zip(partial[:-1], partial[1:])
        ]
        self.assertAlmostEqual(sum(costs), 0.8)
        self.assertAlmostEqual(sum(costs), partial[-1] / 100.0)

    def test_lambda_increases(self):
        row = self.controller().update([1.0], episode=1)
        self.assertAlmostEqual(row["lambda_deadline_after_update"], 1.099)
        self.assertAlmostEqual(row["deadline_miss_rate"], 1.0)
        self.assertAlmostEqual(row["deadline_constraint_mean"], 1.0)

    def test_sub_deadline_metric_accepts_continuous_workflow_cost(self):
        controller = self.controller(
            budget=0.0,
            constraint_metric="sub_deadline_violation",
        )
        row = controller.update([1.5, 0.5], episode=1)
        self.assertAlmostEqual(row["deadline_constraint_batch_mean"], 1.0)
        self.assertAlmostEqual(row["lambda_deadline_after_update"], 1.1)
        self.assertNotIn("deadline_miss_rate", row)

    def test_default_config_targets_one_percent_miss_rate(self):
        controller = DeadlineLagrangianController.from_config({})
        row = controller.update([0.0], episode=0)
        self.assertEqual(row["deadline_constraint_budget"], 0.01)
        self.assertEqual(row["deadline_constraint_mean"], 0.0)

    def test_lambda_decreases(self):
        row = self.controller().update([0.0], episode=1)
        self.assertAlmostEqual(row["lambda_deadline_after_update"], 0.999)

    def test_lambda_unchanged_at_budget(self):
        row = self.controller().update([0.01], episode=1)
        self.assertAlmostEqual(row["lambda_deadline_after_update"], 1.0)

    def test_projection_lower_bound(self):
        controller = self.controller(lambda_init=0.01, dual_lr=10.0)
        controller.update([0.0], episode=1)
        self.assertEqual(controller.value, 0.0)

    def test_projection_upper_bound(self):
        controller = self.controller(lambda_init=1.0, dual_lr=10.0, lambda_max=2.0)
        controller.update([1.0], episode=1)
        self.assertEqual(controller.value, 2.0)

    def test_no_completed_workflow_skips_update(self):
        controller = self.controller()
        row = controller.update([], episode=1)
        self.assertEqual(controller.value, 1.0)
        self.assertEqual(row["completed_workflow_count"], 0)
        self.assertEqual(row["deadline_dual_updated"], 0.0)

    def test_warmup_uses_but_does_not_update_lambda(self):
        controller = self.controller(warmup_episodes=50)
        for episode in range(50):
            controller.update([1.0], episode=episode)
        self.assertEqual(controller.value, 1.0)
        controller.update([1.0], episode=50)
        self.assertAlmostEqual(controller.value, 1.099)

    def test_upper_gets_system_miss_while_lower_keeps_causal_miss(self):
        class FakeHighBuffer:
            def __init__(self):
                self.deadline_costs = [0.2, 0.8]
                self.rewards = [0.0, 0.0]
                self.infos = [
                    {
                        "task_energy_scale": 100.0,
                        "total_energy_actual": 90.0,
                        "high_base_energy_reward_local": 0.2,
                    },
                    {
                        "task_energy_scale": 100.0,
                        "total_energy_actual": 90.0,
                        "high_base_energy_reward_local": 0.2,
                    },
                ]

            def workflow_groups(self):
                return [("w", [0, 1])]

        high_buffer = FakeHighBuffer()
        credit = _apply_cooperative_workflow_credit(
            SimpleNamespace(team_energy_weight=0.1, deadline_lambda=2.0),
            high_buffer,
            [
                {
                    "workflow_id": "w",
                    "response_time": 110.0,
                    "relative_deadline": 100.0,
                }
            ],
            {"w": 1.0},
        )
        self.assertEqual(credit["w"]["high_causal_miss"], 1.0)
        self.assertEqual(credit["w"]["low_causal_miss"], 1.0)
        self.assertAlmostEqual(
            sum(
                row["high_deadline_constraint_cost"]
                for row in high_buffer.infos
            ),
            1.0,
        )
        self.assertAlmostEqual(
            sum(row["high_team_energy_reward"] for row in high_buffer.infos),
            0.01,
        )

    @unittest.skipIf(torch is None, "torch is required")
    def test_reward_composition_and_current_lambda(self):
        agent = D3QN(
            state_dim=2,
            action_dim=2,
            hidden_dim=4,
            hidden_layers=1,
            batch_size=1,
            device="cpu",
        )
        statistics = {"mean": 0.0, "std": 0.2}
        first = agent._compose_lagrangian_rewards(
            [0.2], [0.1], 1.0, statistics
        )
        second = agent._compose_lagrangian_rewards(
            [0.2], [0.1], 3.0, statistics
        )
        self.assertAlmostEqual(float(first.item()), 0.9)
        self.assertAlmostEqual(float(second.item()), 0.7)

        agent.remember(
            np.zeros(2),
            0,
            999.0,
            np.zeros(2),
            True,
            np.array([True, True]),
            np.array([False, False]),
            base_energy_reward=0.2,
            deadline_constraint_cost=0.1,
        )
        first_sample = agent.learn(deadline_lambda=1.0)
        second_sample = agent.learn(deadline_lambda=3.0)
        # A one-sample replay has zero centered energy reward; only the
        # current Lagrangian cost remains.
        self.assertAlmostEqual(first_sample["reward_mean"], -0.1, places=6)
        self.assertAlmostEqual(second_sample["reward_mean"], -0.3, places=6)

    def test_lagrangian_reward_has_no_double_deadline_penalty(self):
        objective = HierarchicalObjective(
            {
                "reward_mode": "lagrangian_deadline",
                "high_energy_weight": 1.0,
                # This old fixed coefficient must be ignored in this mode.
                "deadline_violation_weight": 999.0,
            }
        )
        objective.set_deadline_lambda(2.0)
        env = FakeLagrangianEnv(120.0)
        feedback = objective.compute_step_feedback(
            env,
            {
                "decision_context": {"task_energy_scale": 100.0},
                "servers": [],
            },
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
                    "final_aft": 120.0,
                }
            },
        )
        # The selected-server rho=1 energy equals the median reference, so its
        # energy reward is zero. Lower-level actual energy does not enter it.
        self.assertAlmostEqual(feedback["high_base_energy_reward"], 0.0)
        self.assertAlmostEqual(feedback["high_deadline_constraint_cost"], 1.2)
        self.assertAlmostEqual(feedback["high_reward"], -2.0 * 1.2)
        self.assertEqual(feedback["high_energy_includes_low_dvfs"], False)
        self.assertEqual(feedback["high_deadline_penalty"], 0.0)
        self.assertEqual(feedback["high_deadline_crossed"], 1.0)
        self.assertAlmostEqual(feedback["high_deadline_violation"], 1.2)

    def test_upper_energy_uses_median_ratio_and_ignores_low_dvfs(self):
        objective = HierarchicalObjective(
            {"reward_mode": "lagrangian_deadline", "high_energy_weight": 1.0}
        )
        state = {
            "decision_context": {"task_energy_scale": 100.0},
            "servers": [],
        }

        def feedback(actual_energy):
            objective.reset_episode()
            return objective.compute_step_feedback(
                FakeLagrangianEnv(80.0),
                state,
                {},
                {},
                {
                    "workflow_id": "w",
                    "task_id": 0,
                    "server_id": 0,
                    "primary_energy_ref_rho1": 80.0,
                },
                -float(actual_energy),
                {
                    "event": {
                        "energy": float(actual_energy),
                        "primary_energy": float(actual_energy),
                        "final_aft": 80.0,
                    }
                },
            )

        rho_one = feedback(80.0)
        lower_dvfs = feedback(50.0)
        self.assertAlmostEqual(rho_one["high_base_energy_reward"], 0.2)
        self.assertAlmostEqual(lower_dvfs["high_base_energy_reward"], 0.2)
        self.assertAlmostEqual(
            lower_dvfs["high_placement_energy_saving_joules"], 20.0
        )
        self.assertAlmostEqual(
            lower_dvfs["high_placement_energy_saving_ratio"], 0.2
        )
        self.assertFalse(lower_dvfs["high_energy_includes_low_dvfs"])

    def test_ctde_v2_upper_energy_ignores_low_dvfs(self):
        objective = HierarchicalObjective(
            {
                "reward_mode": "lagrangian_deadline",
                "reward_profile": "ctde_placement_cumulative_v2",
                "high_energy_weight": 1.0,
            }
        )
        feedback = objective.compute_step_feedback(
            FakeLagrangianEnv(80.0, task_count=4),
            {
                "decision_context": {"task_energy_scale": 100.0},
                "servers": [],
            },
            {},
            {},
            {
                "workflow_id": "w",
                "task_id": 0,
                "server_id": 0,
                "primary_energy_ref_rho1": 80.0,
            },
            -50.0,
            {
                "event": {
                    "energy": 50.0,
                    "primary_energy": 50.0,
                    "final_aft": 80.0,
                }
            },
        )
        self.assertAlmostEqual(feedback["high_base_energy_reward"], 0.05)
        self.assertAlmostEqual(feedback["high_energy_actual_for_reward"], 80.0)
        self.assertFalse(feedback["high_energy_includes_low_dvfs"])

    def test_ep740_upper_energy_uses_realized_low_dvfs_energy(self):
        objective = HierarchicalObjective(
            {
                "reward_mode": "lagrangian_deadline",
                "reward_profile": "ep740_lagrangian_v1",
                "high_energy_weight": 1.0,
            }
        )
        feedback = objective.compute_step_feedback(
            FakeLagrangianEnv(80.0, task_count=4),
            {
                "decision_context": {"task_energy_scale": 100.0},
                "servers": [],
            },
            {},
            {},
            {
                "workflow_id": "w",
                "task_id": 0,
                "server_id": 0,
                "primary_energy_ref_rho1": 80.0,
            },
            -50.0,
            {
                "event": {
                    "energy": 50.0,
                    "primary_energy": 50.0,
                    "final_aft": 80.0,
                }
            },
        )
        self.assertAlmostEqual(feedback["high_energy_saving_ratio"], 0.5)
        self.assertAlmostEqual(feedback["high_base_energy_reward"], 0.125)
        self.assertTrue(feedback["high_energy_includes_low_dvfs"])
        self.assertEqual(
            feedback["high_reward_profile"], "ep740_lagrangian_v1"
        )

    def test_upper_median_ratio_energy_is_divided_by_workflow_task_count(self):
        objective = HierarchicalObjective(
            {"reward_mode": "lagrangian_deadline", "high_energy_weight": 1.0}
        )
        state = {
            "decision_context": {"task_energy_scale": 100.0},
            "servers": [],
        }
        feedback = objective.compute_step_feedback(
            FakeLagrangianEnv(80.0, task_count=4),
            state,
            {},
            {},
            {
                "workflow_id": "w",
                "task_id": 0,
                "server_id": 0,
                "primary_energy_ref_rho1": 80.0,
            },
            -80.0,
            {
                "event": {
                    "energy": 80.0,
                    "primary_energy": 80.0,
                    "final_aft": 80.0,
                }
            },
        )
        self.assertAlmostEqual(
            feedback["high_placement_energy_saving_joules"], 20.0
        )
        self.assertAlmostEqual(
            feedback["high_placement_energy_saving_ratio"], 0.2
        )
        self.assertAlmostEqual(feedback["high_base_energy_reward"], 0.05)
        self.assertEqual(feedback["workflow_task_count"], 4.0)

    def test_terminal_patch_updates_cost_component_not_scalar_reward(self):
        objective = HierarchicalObjective(
            {"reward_mode": "lagrangian_deadline", "high_energy_weight": 1.0}
        )
        objective.previous_partial_makespan["w"] = 80.0
        objective.deadline_cost_sums["w"] = 80.0
        buffer = HighRolloutBuffer()
        buffer.add(
            state=np.zeros(2),
            primary_action=0,
            backup_action=None,
            old_log_prob=0.0,
            reward_value=0.0,
            deadline_value=0.0,
            reliability_value=0.0,
            reward=0.2,
            deadline_cost=80.0,
            reliability_cost=0.0,
            deadline_limit=100.0,
            reliability_limit=1.0,
            partial_makespan=80.0,
            partial_log_risk=0.0,
            workflow_id="w",
            workflow_done=False,
            episode_done=False,
            next_state=None,
            primary_mask=np.array([True, True]),
            backup_pair_mask=np.zeros((2, 2), dtype=bool),
            info={
                "high_reward_mode": "lagrangian_deadline",
                "high_base_energy_reward": 0.2,
                "high_deadline_constraint_cost": 0.8,
                "deadline_lambda": 1.0,
            },
        )
        _patch_terminal(
            objective,
            buffer,
            {
                "workflow_id": "w",
                "response_time": 95.0,
                "relative_deadline": 100.0,
            },
            set(),
        )
        self.assertAlmostEqual(buffer.rewards[0], 0.2)
        self.assertAlmostEqual(buffer.deadline_costs[0] / 100.0, 0.95)
        self.assertAlmostEqual(
            buffer.infos[0]["terminal_high_deadline_constraint_cost"], 0.15
        )
        self.assertTrue(buffer.validate())

    def test_controller_checkpoint_round_trip(self):
        source = self.controller()
        source.update([1.0], episode=1)
        restored = self.controller(lambda_init=0.0)
        restored.load_state_dict(source.state_dict())
        self.assertAlmostEqual(restored.value, source.value)

    @unittest.skipIf(torch is None, "torch is required")
    def test_agent_checkpoint_restores_deadline_lambda(self):
        config = {
            "shared": {
                "device": "cpu",
                "state": {"max_ready_tasks": 2, "max_servers": 2},
            },
            "high": {
                "hidden_dim": 8,
                "hidden_layers": 1,
                "batch_size": 2,
                "objective_reward_mode": "lagrangian_deadline",
                "lagrangian": {
                    "enabled": True,
                    "deadline_lambda_init": 1.0,
                    "dual_learning_rate": 0.1,
                    "deadline_budget": 1.0,
                    "lambda_max": 100.0,
                    "warmup_episodes": 0,
                    "update_interval": "episode_end",
                },
            },
            "use_low_agent": False,
        }
        source = HierarchicalD3QNAgent(config, seed=1)
        source.high_agent.update_deadline_lagrangian([1.0], episode=1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "lagrangian_checkpoint.pt"
            source.save_checkpoint(path, episode=7, config=config)
            restored = HierarchicalD3QNAgent(config, seed=2)
            payload = restored.load_checkpoint(path)
        self.assertEqual(payload["episode"], 7)
        self.assertAlmostEqual(
            restored.high_agent.deadline_lambda,
            source.high_agent.deadline_lambda,
        )

    @staticmethod
    def shared_agent_config():
        return {
            "shared": {
                "device": "cpu",
                "state": {"max_ready_tasks": 2, "max_servers": 2},
            },
            "shared_deadline_lagrangian": {
                "enabled": True,
                "deadline_lambda_init": 4.0,
                "dual_learning_rate": 0.5,
                "deadline_budget": 0.0,
                "lambda_max": 100.0,
                "warmup_episodes": 0,
                "window_workflows": 4,
                "update_interval": "episode_end",
            },
            "high": {
                "hidden_dim": 8,
                "hidden_layers": 1,
                "batch_size": 2,
                "objective_reward_mode": "lagrangian_deadline",
                "lagrangian": {"enabled": True},
            },
            "low_algorithm": "sac",
            "low": {
                "algorithm": "sac",
                "hidden_dim": 8,
                "hidden_layers": 1,
                "batch_size": 2,
                "automatic_entropy_tuning": False,
                "action_dim": 1,
                "reward": {"mode": "lagrangian_deadline"},
                "lagrangian": {"enabled": True},
            },
        }

    @unittest.skipIf(torch is None, "torch is required")
    def test_shared_lambda_is_one_object_and_updates_once(self):
        agent = HierarchicalD3QNAgent(self.shared_agent_config(), seed=3)
        self.assertIs(
            agent.high_agent.deadline_lagrangian,
            agent.low_agent.deadline_lagrangian,
        )
        metrics = agent.update_shared_deadline_lagrangian(
            [1.0, 0.0], episode=0
        )
        self.assertAlmostEqual(metrics["shared_deadline_lambda"], 4.25)
        self.assertAlmostEqual(agent.high_agent.deadline_lambda, 4.25)
        self.assertAlmostEqual(agent.low_agent.deadline_lambda, 4.25)
        self.assertAlmostEqual(
            metrics["shared_deadline_constraint_batch_mean"], 0.5
        )

    @unittest.skipIf(torch is None, "torch is required")
    def test_shared_lambda_checkpoint_has_single_state_and_restores(self):
        config = self.shared_agent_config()
        source = HierarchicalD3QNAgent(config, seed=4)
        source.update_shared_deadline_lagrangian([1.0], episode=0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shared_lagrangian_checkpoint.pt"
            source.save_checkpoint(path, episode=9, config=config)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["deadline_lagrangian_mode"], "shared")
            self.assertIsNotNone(payload["shared_deadline_lagrangian"])
            self.assertIsNone(payload["deadline_lagrangian"])
            self.assertIsNone(
                payload["low_model_state_dict"]["deadline_lagrangian"]
            )
            restored = HierarchicalD3QNAgent(config, seed=5)
            restored.load_checkpoint(path)
        self.assertIs(
            restored.high_agent.deadline_lagrangian,
            restored.low_agent.deadline_lagrangian,
        )
        self.assertAlmostEqual(restored.high_agent.deadline_lambda, 4.5)

    @unittest.skipIf(torch is None, "torch is required")
    def test_old_reward_mode_ignores_lagrangian_block_for_one_switch_rollback(self):
        agent = HighLevelD3QNAgent(
            {
                "device": "cpu",
                "state": {"max_ready_tasks": 2, "max_servers": 2},
                "hidden_dim": 8,
                "hidden_layers": 1,
                "batch_size": 2,
                "objective_reward_mode": "deadline_incremental_only",
                "lagrangian": {"enabled": True},
            },
            seed=1,
        )
        self.assertFalse(agent.deadline_lagrangian_enabled)
        self.assertIsNone(agent.deadline_lagrangian)


if __name__ == "__main__":
    unittest.main()
