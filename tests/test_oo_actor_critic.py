import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.baselines.oo_actor_critic import (
    OOActorCriticAgent,
    evaluate_upper_reward_candidate,
)
from src.D3QN.reward import HierarchicalObjective
from src.ppo.hierarchical_rollout import HighRolloutBuffer
from src.shared.hierarchical_state_reward import SharedHierarchicalReward
from src.shared.hierarchical_state_reward import (
    BEST_NON_LAGRANGIAN_STATE_VERSION,
    build_shared_state_encoder,
    shared_state_version,
)
from tools.train_oo_actor_critic import _patch_oo_terminal


class OOActorCriticTests(unittest.TestCase):
    def _agent(self, *, num_servers=3, rho_levels=(0.2, 0.6, 1.0), **overrides):
        config = {
            "device": "cpu",
            "hidden_dim": 16,
            "hidden_layers": 2,
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "target_update_interval": 3,
            "num_candidates": num_servers * len(rho_levels),
            "top_s": num_servers * len(rho_levels),
            "bit_flip_probability": 0.2,
            "replay_size": 64,
            "batch_size": 4,
            "training_per_episode": 2,
        }
        config.update(overrides)
        return OOActorCriticAgent(
            state_dim=7,
            num_servers=num_servers,
            rho_levels=rho_levels,
            config=config,
            seed=7,
        )

    def test_fixed_lambda_terminal_telescope_uses_same_multiplier(self):
        objective = HierarchicalObjective(
            {
                "reward_mode": "deadline_incremental_only",
                "deadline_violation_weight": 0.0,
                "fixed_deadline_lambda": 10.0,
            }
        )
        objective.previous_partial_makespan["w"] = 8.0
        objective.deadline_cost_sums["w"] = 8.0
        buffer = HighRolloutBuffer()
        buffer.add(
            state=np.zeros(7, dtype=np.float32),
            primary_action=0,
            backup_action=None,
            old_log_prob=0.0,
            reward_value=0.0,
            deadline_value=0.0,
            reliability_value=0.0,
            reward=-7.9,
            deadline_cost=8.0,
            reliability_cost=0.0,
            deadline_limit=10.0,
            reliability_limit=1.0,
            partial_makespan=8.0,
            partial_log_risk=0.0,
            workflow_id="w",
            workflow_done=False,
            episode_done=False,
            next_state=None,
            primary_mask=np.array([True]),
            backup_pair_mask=np.zeros((1, 1), dtype=bool),
            info={"high_reward": -7.9},
        )

        _patch_oo_terminal(
            objective,
            SimpleNamespace(reward_mode="fixed_lambda_upper_v1"),
            buffer,
            {
                "workflow_id": "w",
                "response_time": 9.0,
                "relative_deadline": 10.0,
            },
            set(),
        )

        self.assertAlmostEqual(buffer.rewards[0], -8.9)
        self.assertAlmostEqual(
            buffer.infos[0]["oo_terminal_deadline_reward"], -1.0
        )

    def test_binary_joint_action_round_trip_and_invalid_code(self):
        agent = self._agent()
        for server_index in range(agent.num_servers):
            for frequency_index in range(len(agent.rho_levels)):
                bits = agent.encode_action(server_index, frequency_index)
                self.assertEqual(bits.shape, (agent.action_dim,))
                self.assertTrue(np.isin(bits, (0.0, 1.0)).all())
                self.assertEqual(
                    agent.decode_action(bits), (server_index, frequency_index)
                )

        invalid_server = np.concatenate(
            (np.ones(agent.server_bits, dtype=np.float32), np.zeros(agent.frequency_bits))
        )
        self.assertIsNone(agent.decode_action(invalid_server))

    def test_best_non_lagrangian_state_is_exactly_580_dimensions(self):
        encoder = build_shared_state_encoder(
            {
                "version": BEST_NON_LAGRANGIAN_STATE_VERSION,
                "max_ready_tasks": 32,
                "max_servers": 26,
            }
        )
        self.assertEqual(encoder.high_state_dim, 580)
        self.assertEqual(
            shared_state_version(encoder), BEST_NON_LAGRANGIAN_STATE_VERSION
        )

    def test_oo_precise_stage_selects_best_valid_joint_action(self):
        agent = self._agent(num_servers=2, rho_levels=(0.2, 1.0))
        for parameter in agent.critic.parameters():
            parameter.data.zero_()
        state = np.linspace(-0.5, 0.5, agent.state_dim, dtype=np.float32)
        valid_mask = np.asarray([True, True], dtype=bool)

        def exact_score(server_index, frequency_index):
            return 10.0 if (server_index, frequency_index) == (1, 0) else -1.0

        decision = agent.select_action(
            state, valid_mask, exact_score, deterministic=True
        )
        self.assertEqual(decision["server_index"], 1)
        self.assertEqual(decision["frequency_index"], 0)
        self.assertEqual(decision["candidate_count"], 4)
        self.assertEqual(decision["top_s_count"], 4)
        self.assertEqual(agent.decode_action(decision["action_bits"]), (1, 0))

    def test_actor_critic_update_and_checkpoint_are_finite(self):
        agent = self._agent()
        rng = np.random.default_rng(11)
        for index in range(8):
            server_index = index % agent.num_servers
            frequency_index = index % len(agent.rho_levels)
            agent.replay.store(
                rng.normal(size=agent.state_dim).astype(np.float32),
                agent.encode_action(server_index, frequency_index),
                float(index) / 10.0,
                rng.normal(size=agent.state_dim).astype(np.float32),
                index == 7,
            )
        metrics = agent.update()
        self.assertEqual(metrics["updates"], 2.0)
        for key, value in metrics.items():
            self.assertTrue(math.isfinite(float(value)), key)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oo.pt"
            agent.save_checkpoint(path, episode=3, include_replay=True)
            restored = self._agent()
            payload = restored.load_checkpoint(path, load_replay=True)
            self.assertEqual(payload["episode"], 3)
            self.assertEqual(restored.replay.size, agent.replay.size)
            for original, loaded in zip(
                agent.actor.parameters(), restored.actor.parameters()
            ):
                np.testing.assert_allclose(
                    original.detach().cpu().numpy(),
                    loaded.detach().cpu().numpy(),
                )

    def test_energy_reward_uses_replay_zscore_and_checkpoint_statistics(self):
        agent = self._agent(energy_reward_weight=2.0, energy_std_floor=0.05)
        state = np.zeros(agent.state_dim, dtype=np.float32)
        action = agent.encode_action(0, 0)
        agent.replay.store(
            state,
            action,
            -0.2,
            state,
            False,
            deadline_reward=-0.2,
            energy_saving=10.0,
        )
        agent.replay.store(
            state,
            action,
            -0.4,
            state,
            True,
            deadline_reward=-0.4,
            energy_saving=30.0,
        )
        statistics = agent.energy_statistics()
        self.assertEqual(statistics["count"], 2)
        self.assertAlmostEqual(statistics["mean"], 20.0)
        self.assertAlmostEqual(statistics["std"], 10.0)
        self.assertAlmostEqual(agent.compose_reward(-0.5, 10.0)["reward"], -2.5)
        self.assertAlmostEqual(agent.compose_reward(-0.5, 30.0)["reward"], 1.5)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oo_energy_stats.pt"
            agent.save_checkpoint(path, include_replay=False)
            restored = self._agent(energy_reward_weight=2.0, energy_std_floor=0.05)
            restored.load_checkpoint(path, load_replay=False)
            self.assertEqual(restored.energy_statistics()["count"], 2)
            self.assertAlmostEqual(restored.energy_statistics()["mean"], 20.0)
            self.assertAlmostEqual(restored.energy_statistics()["std"], 10.0)

    def test_new_oo_candidate_reward_combines_deadline_and_action_energy(self):
        class CandidateEnv:
            def __init__(self):
                task = {"id": 0, "workload_cycles": 1.0}
                self.workflow_by_id = {
                    "w": {
                        "workflow_id": "w",
                        "workflow": {"tasks": [task]},
                        "arrival_time": 0.0,
                        "task_finish_times": {},
                    }
                }
                self.task = task
                self.server = {"id": 0}

            def _get_task(self, workflow, task_id):
                return self.task

            def _get_server(self, server_id):
                return self.server

            def _normalize_rho(self, rho, server):
                return float(rho)

            def workflow_relative_deadline(self, workflow_state):
                return 10.0

            def estimate_replica_plan(self, **kwargs):
                return {
                    "aft": 12.0,
                    "energy": 60.0,
                    "reliability": 1.0,
                }

        objective = HierarchicalObjective(
            {
                "reward_mode": "deadline_incremental_only",
                "deadline_violation_weight": 1.0,
            }
        )
        values = evaluate_upper_reward_candidate(
            CandidateEnv(),
            objective,
            {
                "ready_tasks": ({"workflow_id": "w", "task_id": 0},),
                "decision_context": {"task_energy_scale": 100.0},
            },
            server_id=0,
            rho=0.6,
            energy_reward_weight=0.5,
            energy_statistics={"count": 10, "mean": 20.0, "std": 10.0},
        )
        self.assertAlmostEqual(values["deadline_reward"], -2.4)
        self.assertAlmostEqual(values["energy_saving"], 40.0)
        self.assertAlmostEqual(values["energy_zscore"], 2.0)
        self.assertAlmostEqual(values["energy_reward"], 1.0)
        self.assertAlmostEqual(values["replay_reward"], -1.4)

    def test_best_non_lagrangian_candidate_matches_validated_old_reward(self):
        class CandidateEnv:
            def __init__(self):
                tasks = [
                    {"id": index, "workload_cycles": 1.0}
                    for index in range(4)
                ]
                self.workflow_by_id = {
                    "w": {
                        "workflow_id": "w",
                        "workflow": {"tasks": tasks},
                        "arrival_time": 0.0,
                        "task_finish_times": {},
                        "task_reliability": {},
                    }
                }
                self.task = tasks[0]
                self.server = {"id": 0}

            def _get_task(self, workflow, task_id):
                return self.task

            def _get_server(self, server_id):
                return self.server

            def _normalize_rho(self, rho, server):
                return float(rho)

            def workflow_relative_deadline(self, workflow_state):
                return 10.0

            def _workflow_reliability_requirement(self, workflow_state):
                return 0.9

            def estimate_replica_plan(self, **kwargs):
                return {"aft": 8.0, "energy": 60.0, "reliability": 1.0}

        objective = HierarchicalObjective(
            {
                "reward_mode": "legacy_weighted",
                "time_reward_mode": "legacy_slack",
                "high_makespan_weight": 0.4,
                "high_energy_weight": 0.6,
                "high_reliability_fixed_penalty": 5.0,
                "high_reliability_severity_weight": 2.0,
            }
        )
        values = evaluate_upper_reward_candidate(
            CandidateEnv(),
            objective,
            {
                "ready_tasks": ({"workflow_id": "w", "task_id": 0},),
                "decision_context": {"task_energy_scale": 100.0},
            },
            server_id=0,
            rho=0.6,
            deadline_cost_reward_weight=1.5,
        )
        self.assertAlmostEqual(values["high_time_reward"], 0.4 * 0.2 / 4.0)
        self.assertAlmostEqual(values["high_energy_reward"], 0.6 * 0.4 / 4.0)
        self.assertAlmostEqual(values["deadline_penalty"], 1.5 * 0.8)
        self.assertAlmostEqual(values["replay_reward"], -1.12)

    def test_shared_reward_combines_same_server_dvfs_and_deadline_terms(self):
        reward = SharedHierarchicalReward(
            {
                "joint_low_reward_weight": 0.1,
                "low_deadline_violation_weight": 5.0,
            }
        )
        values = reward.compose(
            -0.2,
            energy_saving=30.0,
            slack_time_cost=0.2,
            deadline_violation_delta=1.1,
            statistics={
                "energy": {"count": 10, "mean": 20.0, "std": 10.0},
                "slack_time": {"count": 10, "mean": 0.1, "std": 0.1},
            },
        )
        self.assertAlmostEqual(values["energy_zscore"], 1.0)
        self.assertAlmostEqual(values["slack_time_zscore"], 1.0)
        self.assertAlmostEqual(values["low_reward"], -5.5)
        self.assertAlmostEqual(values["weighted_low_reward"], -0.55)
        self.assertAlmostEqual(values["reward"], -0.75)

    def test_shared_candidate_uses_selected_server_rho_one_reference(self):
        class CandidateEnv:
            def __init__(self):
                task = {"id": 0, "workload_cycles": 1.0}
                self.workflow_by_id = {
                    "w": {
                        "workflow_id": "w",
                        "workflow": {"tasks": [task]},
                        "arrival_time": 0.0,
                        "task_finish_times": {},
                    }
                }
                self.task = task
                self.server = {"id": 0}

            def _get_task(self, workflow, task_id):
                return self.task

            def _get_server(self, server_id):
                return self.server

            def _normalize_rho(self, rho, server):
                return float(rho)

            def workflow_relative_deadline(self, workflow_state):
                return 10.0

            def estimate_replica_plan(self, **kwargs):
                rho = float(kwargs["rho"])
                return {
                    "aft": 10.0 if math.isclose(rho, 1.0) else 12.0,
                    "energy": 100.0 if math.isclose(rho, 1.0) else 60.0,
                    "reliability": 1.0,
                }

        objective = HierarchicalObjective(
            {
                "reward_mode": "deadline_incremental_only",
                "deadline_violation_weight": 1.0,
            }
        )
        shared_reward = SharedHierarchicalReward(
            {
                "joint_low_reward_weight": 0.1,
                "low_deadline_violation_weight": 5.0,
            }
        )
        values = evaluate_upper_reward_candidate(
            CandidateEnv(),
            objective,
            {
                "ready_tasks": ({"workflow_id": "w", "task_id": 0},),
                "decision_context": {"task_energy_scale": 999.0},
            },
            server_id=0,
            rho=0.6,
            shared_reward=shared_reward,
            reward_statistics={
                "energy": {"count": 10, "mean": 20.0, "std": 10.0},
                "slack_time": {"count": 10, "mean": 0.1, "std": 0.1},
            },
        )
        self.assertAlmostEqual(values["energy_reference"], 100.0)
        self.assertAlmostEqual(values["energy_saving"], 40.0)
        self.assertAlmostEqual(values["deadline_reward"], -2.4)
        self.assertAlmostEqual(values["joint_energy_zscore"], 2.0)
        self.assertAlmostEqual(values["joint_slack_time_zscore"], 1.0)
        self.assertAlmostEqual(values["joint_low_reward"], -5.0)
        self.assertAlmostEqual(values["replay_reward"], -2.9)

    def test_new_shared_upper_reward_matches_d3qn_with_same_server_energy(self):
        class CandidateEnv:
            def __init__(self):
                tasks = [{"id": index, "workload_cycles": 1.0} for index in range(4)]
                self.workflow_by_id = {
                    "w": {
                        "workflow_id": "w",
                        "workflow": {"tasks": tasks},
                        "arrival_time": 0.0,
                        "task_finish_times": {},
                    }
                }
                self.task = tasks[0]
                self.server = {"id": 0}

            def _get_task(self, workflow, task_id):
                return self.task

            def _get_server(self, server_id):
                return self.server

            def _normalize_rho(self, rho, server):
                return float(rho)

            def workflow_relative_deadline(self, workflow_state):
                return 10.0

            def estimate_replica_plan(self, **kwargs):
                rho = float(kwargs["rho"])
                return {
                    "aft": 8.0,
                    "energy": 100.0 if math.isclose(rho, 1.0) else 60.0,
                    "reliability": 1.0,
                }

        objective = HierarchicalObjective(
            {
                "reward_mode": "deadline_incremental_only",
                "deadline_violation_weight": 1.5,
                "high_energy_weight": 1.0,
            }
        )
        shared_reward = SharedHierarchicalReward()
        values = evaluate_upper_reward_candidate(
            CandidateEnv(),
            objective,
            {
                "ready_tasks": ({"workflow_id": "w", "task_id": 0},),
                "decision_context": {"task_energy_scale": 999.0},
            },
            server_id=0,
            rho=0.6,
            shared_reward=shared_reward,
            shared_reward_mode="shared_upper_same_server_energy_v2",
        )
        self.assertAlmostEqual(values["energy_reference"], 100.0)
        self.assertAlmostEqual(values["energy_saving"], 40.0)
        self.assertAlmostEqual(values["energy_saving_ratio"], 0.4)
        self.assertAlmostEqual(values["energy_reward"], 0.4 / 4.0)
        self.assertAlmostEqual(values["deadline_reward"], -0.8)
        self.assertAlmostEqual(values["replay_reward"], -0.8 + 0.4 / 4.0)

    def test_fixed_lambda_candidate_uses_task_energy_scale_and_no_crossing_jump(self):
        class CandidateEnv:
            def __init__(self):
                tasks = [{"id": index, "workload_cycles": 1.0} for index in range(4)]
                self.workflow_by_id = {
                    "w": {
                        "workflow_id": "w",
                        "workflow": {"tasks": tasks},
                        "arrival_time": 0.0,
                        "task_finish_times": {},
                    }
                }
                self.task = tasks[0]
                self.server = {"id": 0}

            def _get_task(self, workflow, task_id):
                return self.task

            def _get_server(self, server_id):
                return self.server

            def _normalize_rho(self, rho, server):
                return float(rho)

            def workflow_relative_deadline(self, workflow_state):
                return 10.0

            def estimate_replica_plan(self, **kwargs):
                return {"aft": 8.0, "energy": 60.0, "reliability": 1.0}

        objective = HierarchicalObjective(
            {
                "reward_mode": "deadline_incremental_only",
                "deadline_violation_weight": 999.0,
                "fixed_deadline_lambda": 10.0,
            }
        )
        values = evaluate_upper_reward_candidate(
            CandidateEnv(),
            objective,
            {
                "ready_tasks": ({"workflow_id": "w", "task_id": 0},),
                "decision_context": {"task_energy_scale": 100.0},
            },
            server_id=0,
            rho=0.6,
            shared_reward_mode="fixed_lambda_upper_v1",
        )
        self.assertAlmostEqual(values["energy_reference"], 100.0)
        self.assertAlmostEqual(values["energy_reward"], 0.4 / 4.0)
        self.assertAlmostEqual(values["deadline_reward"], -10.0 * 0.8)
        self.assertAlmostEqual(values["replay_reward"], -7.9)


if __name__ == "__main__":
    unittest.main()
