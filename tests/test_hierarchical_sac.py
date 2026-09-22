from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.a2c.a2c import Observation
from src.D3QN.hierarchical_d3qn_agent import HighLevelD3QNAgent, HierarchicalD3QNAgent
from src.D3QN.reward import HierarchicalObjective
from src.D3QN.state import HierarchicalStateEncoder, SERVER_FEATURE_DIM, TASK_FEATURE_DIM
from src.sac.deadline_lagrangian import LowDeadlineLagrangianController
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
        self.workflow_by_id = {"w": {"workflow_id": "w", "arrival_time": 0.0}}

    def _workflow_reliability_requirement(self, workflow_state):
        return 0.0


@unittest.skipIf(torch is None, "torch is required")
class TestHierarchicalSAC(unittest.TestCase):
    def test_action_conditioned_ctde_uses_local_actor_and_pooled_global_critic(self):
        high_encoder = HierarchicalStateEncoder(
            {"max_ready_tasks": 4, "max_servers": 2}
        )
        agent = LowLevelDVFSContinuousSACAgent(
            {
                "device": "cpu",
                "state": {"max_ready_tasks": 4, "max_servers": 2},
                "state_mode": "action_conditioned_ctde",
                "shared_high_server_feature_dim": SERVER_FEATURE_DIM,
                "shared_high_global_feature_dim": 5,
                "actor_server_feature_indices": [0, 9, 10],
                "hidden_dim": 8,
                "hidden_layers": 1,
                "batch_size": 2,
                "automatic_entropy_tuning": False,
            },
            seed=7,
        )
        task = {
            "workload_cycles": 2e9,
            "relative_deadline": 10.0,
            "absolute_deadline": 10.0,
            "reliability": 0.9,
        }
        servers = (
            {
                "max_freq_hz": 2e9,
                "static_power": 20.0,
                "dynamic_power": 80.0,
                "execution_available_time": 1.0,
            },
            {
                "max_freq_hz": 3e9,
                "static_power": 30.0,
                "dynamic_power": 100.0,
                "execution_available_time": 2.0,
            },
        )
        server_features = np.arange(
            2 * SERVER_FEATURE_DIM, dtype=np.float32
        ).reshape(2, SERVER_FEATURE_DIM) / 100.0
        observation = Observation(
            ready_task_features=np.zeros((4, TASK_FEATURE_DIM), dtype=np.float32),
            ready_task_mask=np.asarray([True, False, False, False]),
            server_features=server_features,
            server_mask=np.asarray([True, True]),
            global_features=np.arange(5, dtype=np.float32) / 10.0,
            ready_tasks=(task,),
            servers=servers,
            rho_levels=(),
            vector_override=np.zeros(high_encoder.high_state_dim, dtype=np.float32),
        )
        safety = {
            "workflow_relative_deadline": 10.0,
            "workflow_absolute_deadline": 10.0,
            "current_time": 0.0,
            "estimated_ast": 2.0,
            "execution_time_rho_1": 1.0,
            "safe_execution_budget": 8.0,
            "rho_safe": 0.4,
        }

        actor_state = agent.build_low_state(
            observation, task_index=0, server_index=1, safety_metrics=safety
        )
        critic_state = agent.build_critic_state(observation, actor_state)

        self.assertEqual(agent.actor_state_dim, 14)
        self.assertEqual(
            agent.critic_state_dim,
            14 + 5 + 2 * SERVER_FEATURE_DIM,
        )
        np.testing.assert_allclose(
            actor_state[-3:], server_features[1, [0, 9, 10]]
        )
        self.assertEqual(critic_state.shape, (agent.critic_state_dim,))
        swapped = Observation(
            ready_task_features=observation.ready_task_features,
            ready_task_mask=observation.ready_task_mask,
            server_features=server_features[::-1].copy(),
            server_mask=observation.server_mask,
            global_features=observation.global_features,
            ready_tasks=observation.ready_tasks,
            servers=observation.servers,
            rho_levels=(),
            vector_override=observation.vector,
        )
        np.testing.assert_allclose(
            critic_state,
            agent.build_critic_state(swapped, actor_state),
        )

        decision = agent.select_frequency(
            observation,
            task_index=0,
            server_index=1,
            deterministic=True,
            safety_metrics=safety,
        )
        self.assertEqual(decision["state"].shape, (agent.actor_state_dim,))
        self.assertEqual(
            decision["critic_state"].shape, (agent.critic_state_dim,)
        )

    def test_low_sac_can_share_high_state_with_selected_server_identity(self):
        high_encoder = HierarchicalStateEncoder(
            {"max_ready_tasks": 4, "max_servers": 2}
        )
        agent = LowLevelDVFSContinuousSACAgent(
            {
                "device": "cpu",
                "state": {"max_ready_tasks": 4, "max_servers": 2},
                "state_mode": "shared_high_selected_server",
                "shared_high_state_dim": high_encoder.high_state_dim,
                "hidden_dim": 8,
                "hidden_layers": 1,
                "batch_size": 2,
                "automatic_entropy_tuning": False,
            },
            seed=7,
        )
        high_vector = np.arange(
            high_encoder.high_state_dim, dtype=np.float32
        )
        observation = Observation(
            ready_task_features=np.zeros((4, TASK_FEATURE_DIM), dtype=np.float32),
            ready_task_mask=np.asarray([True, False, False, False]),
            server_features=np.zeros((2, SERVER_FEATURE_DIM), dtype=np.float32),
            server_mask=np.asarray([True, True]),
            global_features=np.zeros(5, dtype=np.float32),
            ready_tasks=({},),
            servers=({}, {}),
            rho_levels=(),
            vector_override=high_vector,
        )

        low_state = agent.build_low_state(
            observation, task_index=0, server_index=1
        )

        self.assertEqual(
            agent.low_state_dim, high_encoder.high_state_dim + 2
        )
        np.testing.assert_array_equal(
            low_state[: high_encoder.high_state_dim], high_vector
        )
        np.testing.assert_array_equal(low_state[-2:], [0.0, 1.0])

    def test_load_high_checkpoint_freezes_only_high_networks(self):
        config = {
            "shared": {
                "device": "cpu",
                "state": {"max_ready_tasks": 2, "max_servers": 2},
            },
            "high": {
                "hidden_dim": 8,
                "hidden_layers": 1,
                "batch_size": 2,
            },
            "low": {
                "algorithm": "sac",
                "hidden_dim": 8,
                "hidden_layers": 1,
                "batch_size": 2,
                "automatic_entropy_tuning": False,
            },
            "low_algorithm": "sac",
        }
        source = HierarchicalD3QNAgent(config, seed=1)
        with torch.no_grad():
            for parameter in source.high_agent.agent_R.q_eval.parameters():
                parameter.fill_(0.25)
            source.high_agent.agent_R.update_network_parameters(tau=1.0)

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "source_checkpoint.pt"
            source.save_checkpoint(checkpoint, episode=840, config=config)
            target = HierarchicalD3QNAgent(config, seed=2)
            low_before = {
                key: value.detach().clone()
                for key, value in target.low_agent.actor.state_dict().items()
            }
            payload = target.load_high_checkpoint(checkpoint, freeze=True)

        self.assertEqual(payload["episode"], 840)
        for parameter in target.high_agent.agent_R.q_eval.parameters():
            self.assertTrue(torch.allclose(parameter, torch.full_like(parameter, 0.25)))
            self.assertFalse(parameter.requires_grad)
        for parameter in target.high_agent.agent_R.q_target.parameters():
            self.assertFalse(parameter.requires_grad)
        for key, value in target.low_agent.actor.state_dict().items():
            self.assertTrue(torch.equal(value, low_before[key]))

    def test_high_task_features_use_job_and_workflow_deadline_remaining(self):
        encoder = HierarchicalStateEncoder({"max_ready_tasks": 32, "max_servers": 2})
        features = encoder.high_task_features(
            {
                "workload_cycles": 1e10,
                "absolute_deadline": 50.0,
                "relative_deadline": 20.0,
                "task_ready_time": 10.0,
                # This field must no longer affect the high-level task state.
                "workflow_task_count": 999,
            },
            current_time=30.0,
        )

        np.testing.assert_allclose(
            features,
            [np.tanh(1.0), 2.0, 1.0, 1.0],
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertEqual(TASK_FEATURE_DIM, 4)

    def test_high_server_features_keep_only_selected_pair_metrics(self):
        encoder = HierarchicalStateEncoder({"max_ready_tasks": 32, "max_servers": 2})
        features = encoder.high_server_features(
            {
                "id": 7,
                "max_freq_hz": 2e9,
                "static_power": 20.0,
                "dynamic_power": 80.0,
                "execution_available_time": 12.0,
                "execution_queue_processing_time": 2.0,
                "failure_rate": 0.9,
                "bandwidth_mbps": 99999.0,
            },
            {
                "workload_cycles": 2e9,
                "absolute_deadline": 30.0,
                "relative_deadline": 20.0,
                "task_ready_time": 10.0,
            },
            current_time=10.0,
            max_server_freq_hz=2e9,
            safety_metrics={
                "workflow_relative_deadline": 20.0,
                "workflow_arrival_time": 0.0,
                "workflow_absolute_deadline": 30.0,
                "estimated_ast": 12.0,
                "execution_time_rho_1": 1.0,
                "safe_execution_budget": 17.0,
                "rho_safe": 0.4,
                "low_action_feasible": True,
                "primary_energy_ref_rho1": 150.0,
                "task_reliability_rho_1": 0.97,
                "queued_processing_time_ema": 4.0,
                "recent_primary_assignments": 3.0,
                "recent_backup_assignments": 1.0,
            },
            task_energy_scale=100.0,
        )

        self.assertEqual(SERVER_FEATURE_DIM, 12)
        self.assertEqual(features.shape, (SERVER_FEATURE_DIM,))
        np.testing.assert_allclose(
            features,
            [
                np.tanh(2.0 / 20.0),
                np.tanh(1.0 / 20.0),
                np.tanh(2.0 / 20.0),
                np.tanh(13.0 / 20.0),
                17.0 / 20.0,
                0.0,
                0.4,
                np.tanh(1.5),
                0.03,
                np.tanh(4.0 / 20.0),
                np.tanh(3.0 / 32.0),
                np.tanh(1.0 / 32.0),
            ],
            rtol=1e-6,
            atol=1e-6,
        )

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
            {"energy_saving": 1, "slack_time_cost": 2, "team_energy_reward": -1, "deadline_violation": 0, "reliability_violation": 0},
            {"energy_saving": 3, "slack_time_cost": 6, "team_energy_reward": 1, "deadline_violation": 2, "reliability_violation": 4},
        ):
            replay.store_transition(
                np.zeros(2),
                0.37,
                {**value, "deadline_crossed": 0, "reliability_crossed": 0},
                np.zeros(2),
                False,
            )
        statistics = replay.reward_statistics(std_floor=0.01)
        np.testing.assert_allclose(statistics["mean"], [2, 4, 0, 1, 2])
        np.testing.assert_allclose(statistics["std"], [1, 2, 1, 1, 2])

    def test_responsibility_aware_replay_prioritizes_only_causal_crossings(self):
        replay = LowSACReplayBuffer(
            2,
            critic_state_dim=3,
            max_size=8,
            batch_size=2,
            priority_config={
                "enabled": True,
                "uniform_mix": 0.0,
                "responsible_crossing_bonus": 2.0,
                "severity_weight": 1.0,
            },
        )
        base = {
            "energy_saving": 1.0,
            "slack_time_cost": 0.0,
            "team_energy_reward": 0.0,
            "deadline_violation": 0.0,
            "reliability_violation": 0.0,
            "deadline_crossed": 0.0,
            "reliability_crossed": 0.0,
            "deadline_constraint_cost": 0.0,
        }
        replay.store_transition(
            np.zeros(2),
            0.8,
            {
                **base,
                "deadline_counterfactual_crossed": 0.0,
                "deadline_counterfactual_cost": 4.0,
            },
            np.ones(2),
            False,
            critic_state=np.zeros(3),
            next_critic_state=np.ones(3),
        )
        replay.store_transition(
            np.ones(2),
            0.7,
            {
                **base,
                "deadline_counterfactual_crossed": 1.0,
                "deadline_counterfactual_cost": 0.25,
            },
            np.zeros(2),
            False,
            critic_state=np.ones(3),
            next_critic_state=np.zeros(3),
        )
        self.assertEqual(replay.responsibility_severity[0], 0.0)
        self.assertAlmostEqual(replay.responsibility_severity[1], 0.25)
        replay.update_priorities([0, 1], [1.0, 1.0])
        self.assertGreater(replay.priorities[1], replay.priorities[0])
        sample = replay.sample_buffer(np.random.default_rng(1), return_info=True)
        self.assertEqual(sample[0].shape, (2, 2))
        self.assertEqual(sample[1].shape, (2, 3))
        self.assertEqual(sample[10].shape, (2,))
        restored = LowSACReplayBuffer(
            2,
            critic_state_dim=3,
            max_size=8,
            batch_size=2,
            priority_config={"enabled": True},
        )
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(restored.mem_count, 2)
        np.testing.assert_allclose(restored.critic_states, replay.critic_states)
        np.testing.assert_allclose(restored.priorities, replay.priorities)
        self.assertAlmostEqual(
            restored.max_base_priority, replay.max_base_priority
        )

    def test_reward_combines_zscores_and_hard_penalties(self):
        agent = self.agent()
        components = torch.tensor([[3.0, 6.0, 0.0, 2.0, 4.0]])
        flags = torch.tensor([[1.0, 1.0]])
        statistics = {
            "mean": np.array([2, 4, 0, 1, 2], dtype=np.float32),
            "std": np.array([1, 2, 1, 1, 2], dtype=np.float32),
        }
        reward = agent._compose_rewards(components, flags, statistics)
        self.assertAlmostEqual(float(reward.item()), -10.0)

    def test_deadline_unit_jump_penalty_is_not_zscore_normalized(self):
        agent = self.agent()
        components = torch.tensor([[2.0, 4.0, 0.0, 1.25, 2.0]])
        flags = torch.zeros((1, 2))
        statistics = {
            "mean": np.array([2, 4, 0, 999, 2], dtype=np.float32),
            "std": np.array([1, 2, 1, 1000, 2], dtype=np.float32),
        }
        reward = agent._compose_rewards(components, flags, statistics)
        self.assertAlmostEqual(float(reward.item()), -6.25)

    def test_team_energy_credit_is_a_small_positive_cooperative_term(self):
        agent = LowLevelDVFSContinuousSACAgent(
            {
                "device": "cpu",
                "hidden_dim": 16,
                "hidden_layers": 1,
                "action_dim": 1,
                "batch_size": 2,
                "automatic_entropy_tuning": False,
                "reward": {"team_energy_weight": 0.1},
            },
            seed=7,
        )
        components = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])
        flags = torch.zeros((1, 2))
        statistics = {
            "mean": np.zeros(5, dtype=np.float32),
            "std": np.ones(5, dtype=np.float32),
        }
        reward = agent._compose_rewards(components, flags, statistics)
        self.assertAlmostEqual(float(reward.item()), 0.1)

    def test_low_lagrangian_recomposes_replay_reward_with_current_lambda(self):
        agent = LowLevelDVFSContinuousSACAgent(
            {
                "device": "cpu",
                "hidden_dim": 16,
                "hidden_layers": 1,
                "action_dim": 1,
                "batch_size": 2,
                "automatic_entropy_tuning": False,
                "reward": {
                    "mode": "lagrangian_deadline",
                    "energy_weight": 1.0,
                    "slack_time_weight": 1.0,
                },
                "lagrangian": {
                    "enabled": True,
                    "deadline_lambda_init": 2.0,
                    "dual_learning_rate": 0.1,
                    "deadline_budget": 0.0,
                    "warmup_episodes": 0,
                },
            },
            seed=7,
        )
        components = torch.tensor([[3.0, 6.0, 0.0, 999.0, 0.0]])
        flags = torch.zeros((1, 2))
        statistics = {
            "mean": np.array([2, 4, 0, 0, 0], dtype=np.float32),
            "std": np.array([1, 2, 1, 1, 1], dtype=np.float32),
        }
        constraint_cost = torch.tensor([0.5])

        reward = agent._compose_rewards(
            components,
            flags,
            statistics,
            deadline_constraint_costs=constraint_cost,
        )
        self.assertAlmostEqual(float(reward.item()), -1.0)

        metrics = agent.update_deadline_lagrangian([1.0], episode=0)
        self.assertAlmostEqual(metrics["low_deadline_lambda_after_update"], 2.1)
        reward = agent._compose_rewards(
            components,
            flags,
            statistics,
            deadline_constraint_costs=constraint_cost,
        )
        self.assertAlmostEqual(float(reward.item()), -1.05)

    def test_low_lagrangian_controller_keeps_warmup_and_restores_state(self):
        controller = LowDeadlineLagrangianController(
            lambda_init=10.0,
            dual_lr=0.1,
            budget=0.0,
            lambda_max=20.0,
            warmup_episodes=2,
            window_workflows=3,
        )
        frozen = controller.update([1.0], episode=1)
        self.assertAlmostEqual(frozen["low_deadline_lambda_after_update"], 10.0)
        updated = controller.update([0.0, 1.0], episode=2)
        self.assertAlmostEqual(
            updated["low_deadline_constraint_mean"], 2.0 / 3.0
        )
        self.assertAlmostEqual(
            updated["low_deadline_lambda_after_update"],
            10.0 + 0.1 * (2.0 / 3.0),
        )
        self.assertEqual(updated["low_deadline_constraint_workflow_count"], 3)

        restored = LowDeadlineLagrangianController(
            lambda_init=0.0,
            dual_lr=0.0,
        )
        restored.load_state_dict(controller.state_dict())
        # The multiplier and the rolling constraint window are state and are
        # restored from the checkpoint.
        self.assertAlmostEqual(restored.value, 10.0 + 0.1 * (2.0 / 3.0))
        self.assertEqual(list(restored.workflow_cost_window), [1.0, 0.0, 1.0])
        # The dual step size is a hyper-parameter, not state: it stays at the
        # configured 0.0 instead of adopting the checkpoint's 0.1.  Restoring it
        # used to make every warm-started run silently inherit the checkpoint's
        # tuning and ignore its own config; see
        # tests/test_lagrangian_config_precedence.py.
        self.assertAlmostEqual(restored.dual_lr, 0.0)

    def test_low_agent_checkpoint_restores_deadline_lambda(self):
        config = {
            "device": "cpu",
            "hidden_dim": 16,
            "hidden_layers": 1,
            "action_dim": 1,
            "batch_size": 2,
            "automatic_entropy_tuning": False,
            "reward": {"mode": "lagrangian_deadline"},
            "lagrangian": {
                "enabled": True,
                "deadline_lambda_init": 10.0,
                "dual_learning_rate": 0.1,
                "deadline_budget": 0.0,
                "warmup_episodes": 0,
            },
        }
        source = LowLevelDVFSContinuousSACAgent(config, seed=1)
        source.update_deadline_lagrangian([1.0], episode=0)
        payload = source.state_dict(include_replay=False)

        target = LowLevelDVFSContinuousSACAgent(config, seed=2)
        target.load_state_dict(payload)
        self.assertAlmostEqual(target.deadline_lambda, 10.1)

    def test_low_lagrangian_uses_only_most_recent_workflow_window(self):
        controller = LowDeadlineLagrangianController(
            lambda_init=1.0,
            dual_lr=1.0,
            budget=0.25,
            warmup_episodes=0,
            window_workflows=4,
        )
        controller.update([1.0, 1.0], episode=0)
        metrics = controller.update([0.0, 0.0, 0.0, 0.0], episode=1)

        self.assertEqual(
            metrics["low_deadline_constraint_workflow_count"], 4
        )
        self.assertEqual(list(controller.workflow_cost_window), [0.0] * 4)
        self.assertAlmostEqual(
            metrics["low_deadline_constraint_mean"], 0.0
        )
        self.assertAlmostEqual(
            metrics["low_deadline_lambda_after_update"], 1.5
        )

    def test_energy_reward_component_uses_absolute_joule_saving(self):
        components = build_low_sac_reward_components(
            FakeLowRewardEnv(),
            {"workflow_id": "w", "task_id": 0},
            {"event": {"energy": 60.0, "final_aft": 2.0, "final_reliability": 1.0}},
            {
                "workflow_relative_deadline": 10.0,
                "workflow_arrival_time": 0.0,
                "workflow_partial_makespan_before": 0.0,
                "total_energy": 100.0,
                "final_aft": 1.0,
                "task_latest_finish_time": 10.0,
                "final_reliability": 1.0,
            },
        )
        self.assertAlmostEqual(components["energy_saving"], 40.0)
        self.assertAlmostEqual(components["energy_saved"], 40.0)
        self.assertAlmostEqual(components["energy_saving_ratio"], 0.4)
        self.assertAlmostEqual(components["slack_time_cost"], 0.1)

    def test_sub_deadline_reward_uses_same_per_workflow_scale_as_energy(self):
        components = build_low_sac_reward_components(
            FakeLowRewardEnv(),
            {"workflow_id": "w", "task_id": 0},
            {
                "event": {
                    "energy": 60.0,
                    "final_aft": 11.0,
                    "final_reliability": 1.0,
                }
            },
            {
                "workflow_relative_deadline": 20.0,
                "workflow_arrival_time": 0.0,
                "workflow_partial_makespan_before": 0.0,
                "workflow_task_count": 4,
                "task_sub_deadline_relative": 10.0,
                "total_energy": 100.0,
                "final_aft": 9.0,
                "final_reliability": 1.0,
            },
            reward_profile="sub_deadline_lagrangian_v1",
        )
        self.assertAlmostEqual(components["energy_saved"], 40.0)
        self.assertAlmostEqual(components["energy_saving"], 0.1)
        self.assertAlmostEqual(components["reference_sub_deadline_cost"], 0.0)
        self.assertAlmostEqual(components["actual_sub_deadline_cost"], 0.275)
        self.assertAlmostEqual(components["deadline_constraint_cost"], 0.275)
        self.assertEqual(components["deadline_constraint_crossed"], 1.0)

    def test_low_sub_deadline_controller_accepts_continuous_cost(self):
        controller = LowDeadlineLagrangianController(
            lambda_init=1.0,
            dual_lr=0.1,
            budget=0.0,
            warmup_episodes=0,
            constraint_metric="sub_deadline_violation",
        )
        row = controller.update([1.5, 0.5], episode=0)
        self.assertAlmostEqual(row["low_deadline_constraint_batch_mean"], 1.0)
        self.assertAlmostEqual(row["low_deadline_lambda_after_update"], 1.1)

    def test_slack_time_cost_uses_five_percent_deadline_floor(self):
        components = build_low_sac_reward_components(
            FakeLowRewardEnv(),
            {"workflow_id": "w", "task_id": 0},
            {"event": {"energy": 60.0, "final_aft": 102.0, "final_reliability": 1.0}},
            {
                "workflow_relative_deadline": 100.0,
                "workflow_arrival_time": 0.0,
                "workflow_partial_makespan_before": 98.0,
                "total_energy": 100.0,
                "final_aft": 98.0,
                "final_reliability": 1.0,
            },
            slack_floor_ratio=0.05,
        )
        self.assertAlmostEqual(components["slack_before_seconds"], 2.0)
        self.assertAlmostEqual(components["effective_slack_seconds"], 5.0)
        self.assertAlmostEqual(components["slack_time_cost"], 0.8)
        self.assertAlmostEqual(components["deadline_violation"], 1.02)
        self.assertEqual(components["deadline_crossed"], 1.0)
        self.assertAlmostEqual(components["deadline_constraint_cost"], 1.02)
        self.assertEqual(components["deadline_constraint_crossed"], 1.0)

    def test_low_constraint_charges_only_damage_beyond_rho_one_reference(self):
        components = build_low_sac_reward_components(
            FakeLowRewardEnv(),
            {"workflow_id": "w", "task_id": 0},
            {
                "event": {
                    "energy": 60.0,
                    "final_aft": 12.0,
                    "final_reliability": 1.0,
                }
            },
            {
                "workflow_relative_deadline": 10.0,
                "workflow_arrival_time": 0.0,
                "workflow_partial_makespan_before": 9.0,
                "total_energy": 100.0,
                # Even rho=1 misses at 11 seconds. SAC is charged only for
                # the additional normalized overrun from 11 to 12 seconds.
                "final_aft": 11.0,
                "final_reliability": 1.0,
            },
        )
        self.assertAlmostEqual(components["deadline_violation"], 1.2)
        self.assertAlmostEqual(components["reference_deadline_violation"], 1.1)
        self.assertAlmostEqual(components["deadline_constraint_cost"], 0.1)
        self.assertEqual(components["deadline_constraint_crossed"], 0.0)
        self.assertAlmostEqual(components["deadline_counterfactual_cost"], 0.1)
        self.assertEqual(components["deadline_counterfactual_crossed"], 0.0)

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
            "server_safety_metrics": {
                "3": {
                    "rho_hardware_min": 0.2,
                    "estimated_finish_time": 4.0,
                },
                "7": {
                    "rho_hardware_min": 0.2,
                    "estimated_finish_time": 3.0,
                },
            },
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
        self.assertEqual(high_decision["primary_server_id"], 7)
        self.assertIsNone(high_decision["backup_server_id"])
        self.assertTrue(high_decision["primary_server_mask"].all())
        self.assertEqual(high_decision["action_mask_mode"], "structural_only")
        self.assertTrue(high_decision["heft_online_queue_aware"])
        self.assertNotIn("heft_primary_mask_violation", high_decision)
        self.assertFalse(env_action["backup"])
        self.assertAlmostEqual(env_action["rho"], 0.63)
        self.assertEqual(agent.low_agent.selected[:2], (0, 1))

    def test_all_heuristic_high_algorithms_are_supported(self):
        self.assertEqual(_normalize_high_algorithm("FCFS"), "fcfs")
        self.assertEqual(_normalize_high_algorithm("Fastest"), "fastest")
        self.assertEqual(_normalize_high_algorithm("Random"), "random")
        with self.assertRaises(ValueError):
            _normalize_high_algorithm("unknown")

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

    def test_replay_zscore_high_reward_emits_slack_energy_crossing_and_violation(self):
        objective = HierarchicalObjective(
            {"reward_mode": "deadline_energy_replay_zscore"}
        )
        env = FakeHighRewardEnv(reliability=1.0)
        env.workflow_by_id["w"]["task_finish_times"] = {0: 8.0}
        feedback = objective.compute_step_feedback(
            env,
            {"servers": [], "decision_context": {}},
            {},
            {},
            {
                "workflow_id": "w",
                "task_id": 0,
                "server_id": 0,
                "primary_energy_ref_rho1": 100.0,
            },
            -60.0,
            {
                "event": {
                    "energy": 60.0,
                    "primary_energy": 60.0,
                    "task_ready_time": 0.0,
                    "final_aft": 8.0,
                }
            },
        )
        self.assertAlmostEqual(feedback["high_workflow_slack_ratio"], 0.2)
        self.assertAlmostEqual(feedback["high_energy_saving_joules"], 40.0)
        self.assertEqual(feedback["high_deadline_missed"], 0.0)
        self.assertEqual(feedback["high_deadline_crossed"], 0.0)
        self.assertEqual(feedback["high_deadline_violation"], 0.0)
        self.assertEqual(feedback["high_reward"], 0.0)
        self.assertEqual(feedback["high_reliability_penalty"], 0.0)

        objective.reset_episode()
        env.workflow_by_id["w"]["task_finish_times"] = {0: 12.0}
        missed = objective.compute_step_feedback(
            env,
            {"servers": [], "decision_context": {}},
            {},
            {},
            {
                "workflow_id": "w",
                "task_id": 0,
                "server_id": 0,
                "primary_energy_ref_rho1": 100.0,
            },
            -60.0,
            {
                "event": {
                    "energy": 60.0,
                    "primary_energy": 60.0,
                    "task_ready_time": 0.0,
                    "final_aft": 12.0,
                }
            },
        )
        self.assertAlmostEqual(missed["high_workflow_slack_ratio"], -0.2)
        self.assertEqual(missed["high_deadline_missed"], 1.0)
        self.assertEqual(missed["high_deadline_crossed"], 1.0)
        self.assertAlmostEqual(missed["high_deadline_violation"], 0.2)
        self.assertAlmostEqual(missed["high_reward"], -0.4)

        env.workflow_by_id["w"]["task_finish_times"] = {0: 13.0}
        still_late = objective.compute_step_feedback(
            env,
            {"servers": [], "decision_context": {}},
            {},
            {},
            {
                "workflow_id": "w",
                "task_id": 0,
                "server_id": 0,
                "primary_energy_ref_rho1": 100.0,
            },
            -60.0,
            {
                "event": {
                    "energy": 60.0,
                    "primary_energy": 60.0,
                    "task_ready_time": 0.0,
                    "final_aft": 13.0,
                }
            },
        )
        self.assertEqual(still_late["high_deadline_crossed"], 0.0)
        self.assertAlmostEqual(still_late["high_deadline_violation"], 0.3)
        self.assertAlmostEqual(still_late["high_reward"], -0.6)

    def test_deadline_incremental_reward_has_unit_jump_and_telescopes(self):
        objective = HierarchicalObjective(
            {
                "reward_mode": "deadline_incremental_only",
                "deadline_violation_weight": 1.0,
            }
        )
        env = FakeHighRewardEnv(reliability=1.0)

        def step(finish_time):
            env.workflow_by_id["w"]["task_finish_times"] = {0: finish_time}
            return objective.compute_step_feedback(
                env,
                {"servers": [], "decision_context": {}},
                {},
                {},
                {
                    "workflow_id": "w",
                    "task_id": 0,
                    "server_id": 0,
                    "primary_energy_ref_rho1": 100.0,
                },
                -60.0,
                {
                    "event": {
                        "energy": 60.0,
                        "primary_energy": 60.0,
                        "task_ready_time": 0.0,
                        "final_aft": finish_time,
                    }
                },
            )

        safe = step(8.0)
        self.assertAlmostEqual(safe["normalized_makespan_increment"], 0.8)
        self.assertEqual(safe["high_deadline_violation"], 0.0)
        self.assertEqual(safe["high_deadline_violation_delta"], 0.0)
        self.assertAlmostEqual(safe["high_reward"], -0.8)
        self.assertEqual(safe["high_energy_reward"], 0.0)

        crossed = step(12.0)
        self.assertAlmostEqual(crossed["normalized_makespan_increment"], 0.4)
        self.assertEqual(crossed["high_deadline_crossed"], 1.0)
        self.assertAlmostEqual(crossed["high_deadline_violation"], 1.2)
        self.assertAlmostEqual(crossed["high_deadline_violation_delta"], 1.2)
        self.assertAlmostEqual(crossed["high_deadline_penalty"], 1.2)
        self.assertAlmostEqual(crossed["high_reward"], -1.6)

        still_late = step(13.0)
        self.assertAlmostEqual(still_late["normalized_makespan_increment"], 0.1)
        self.assertEqual(still_late["high_deadline_crossed"], 0.0)
        self.assertAlmostEqual(still_late["high_deadline_violation"], 1.3)
        self.assertAlmostEqual(still_late["high_deadline_violation_delta"], 0.1)
        self.assertAlmostEqual(still_late["high_deadline_penalty"], 0.1)
        self.assertAlmostEqual(still_late["high_reward"], -0.2)

        unchanged = step(13.0)
        self.assertEqual(unchanged["normalized_makespan_increment"], 0.0)
        self.assertEqual(unchanged["high_deadline_violation_delta"], 0.0)
        self.assertEqual(unchanged["high_reward"], 0.0)

        total_reward = sum(
            row["high_reward"] for row in (safe, crossed, still_late, unchanged)
        )
        self.assertAlmostEqual(total_reward, -(13.0 / 10.0) - 1.3)

    def test_deadline_incremental_reward_normalizes_energy_by_task_count(self):
        objective = HierarchicalObjective(
            {
                "reward_mode": "deadline_incremental_only",
                "deadline_violation_weight": 1.0,
                "high_energy_weight": 0.6,
            }
        )
        env = FakeHighRewardEnv(reliability=1.0, task_count=4)
        env.workflow_by_id["w"]["task_finish_times"] = {0: 8.0}
        feedback = objective.compute_step_feedback(
            env,
            {
                "servers": [],
                "decision_context": {"task_energy_scale": 100.0},
            },
            {},
            {},
            {
                "workflow_id": "w",
                "task_id": 0,
                "server_id": 0,
                "primary_energy_ref_rho1": 100.0,
            },
            -60.0,
            {
                "event": {
                    "energy": 60.0,
                    "primary_energy": 60.0,
                    "task_ready_time": 0.0,
                    "final_aft": 8.0,
                }
            },
        )
        self.assertAlmostEqual(feedback["high_energy_reward_raw"], 0.4)
        self.assertAlmostEqual(feedback["high_energy_reward"], 0.24 / 4.0)
        self.assertAlmostEqual(feedback["high_reward"], -0.8 + 0.24 / 4.0)
        self.assertEqual(feedback["workflow_task_count"], 4.0)

    def test_deadline_incremental_terminal_patch_preserves_telescope(self):
        objective = HierarchicalObjective(
            {
                "reward_mode": "deadline_incremental_only",
                "deadline_violation_weight": 1.0,
            }
        )
        env = FakeHighRewardEnv(reliability=1.0)
        env.workflow_by_id["w"]["task_finish_times"] = {0: 8.0}
        step = objective.compute_step_feedback(
            env,
            {"servers": [], "decision_context": {}},
            {},
            {},
            {
                "workflow_id": "w",
                "task_id": 0,
                "server_id": 0,
                "primary_energy_ref_rho1": 100.0,
            },
            -60.0,
            {
                "event": {
                    "energy": 60.0,
                    "primary_energy": 60.0,
                    "task_ready_time": 0.0,
                    "final_aft": 8.0,
                }
            },
        )
        terminal = objective.compute_terminal_feedback(
            None,
            {
                "workflow_id": "w",
                "response_time": 12.0,
                "relative_deadline": 10.0,
            },
        )

        self.assertAlmostEqual(terminal["normalized_terminal_makespan_increment"], 0.4)
        self.assertEqual(terminal["high_deadline_crossed"], 1.0)
        self.assertAlmostEqual(terminal["high_deadline_violation"], 1.2)
        self.assertAlmostEqual(terminal["high_deadline_violation_delta"], 1.2)
        self.assertAlmostEqual(terminal["high_reward_delta"], -1.6)
        self.assertAlmostEqual(
            step["high_reward"] + terminal["high_reward_delta"],
            -(12.0 / 10.0) - 1.2,
        )

    def test_d3qn_replay_reward_uses_ordinary_component_zscores(self):
        agent = HighLevelD3QNAgent(
            {
                "state": {"max_servers": 2},
                "device": "cpu",
                "hidden_dim": 16,
                "hidden_layers": 1,
                "batch_size": 2,
                "reward": {
                    "mode": "replay_zscore",
                    "workflow_slack_weight": 1.0,
                    "energy_saving_weight": 1.0,
                    "deadline_violation_weight": 2.0,
                },
            },
            seed=7,
        )
        rewards = agent.agent_R._compose_component_rewards(
            np.asarray([[0.2, 30.0], [-0.2, 10.0]], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.2], dtype=np.float32),
            {
                "mean": np.asarray([0.0, 20.0], dtype=np.float32),
                "std": np.asarray([0.2, 10.0], dtype=np.float32),
            },
        )
        np.testing.assert_allclose(
            rewards.detach().cpu().numpy(),
            [2.0, -2.4],
            rtol=1e-6,
            atol=1e-6,
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
