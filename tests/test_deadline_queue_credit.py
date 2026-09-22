from __future__ import annotations

import unittest

import numpy as np

from src.D3QN.dual_head_constrained import EPS, DualHeadHighLevelD3QNAgent
from src.D3QN.reward import HierarchicalObjective
from src.ppo.hierarchical_rollout import HighRolloutBuffer


def _add_transition(
    buffer,
    *,
    workflow_id,
    increment,
    partial,
    deadline,
    queue_normalized=0.0,
    sub_deadline_normalized=0.0,
    future_regret_normalized=0.0,
):
    buffer.add(
        state=np.zeros(3, dtype=np.float32),
        primary_action=0,
        backup_action=None,
        old_log_prob=0.0,
        reward_value=0.0,
        deadline_value=0.0,
        reliability_value=0.0,
        reward=0.0,
        deadline_cost=increment,
        reliability_cost=0.0,
        deadline_limit=deadline,
        reliability_limit=1.0,
        partial_makespan=partial,
        partial_log_risk=0.0,
        workflow_id=workflow_id,
        workflow_done=False,
        episode_done=False,
        next_state=None,
        primary_mask=np.ones(2, dtype=bool),
        backup_pair_mask=np.zeros((2, 2), dtype=bool),
        info={
            "high_base_energy_reward": 0.1,
            "high_queue_delay_normalized": queue_normalized,
            "high_sub_deadline_normalized": sub_deadline_normalized,
            "high_future_finish_regret_normalized": future_regret_normalized,
        },
    )


def _legacy_dense_miss_credit_costs(buffer, *, miss_penalty=1.0):
    """The pre-beta_q implementation, kept verbatim as the regression oracle.

    Copied from ``workflow_dense_miss_credit_deadline_costs`` as it stood
    before the queue term was added, so the beta_q = 0 path can be compared
    against the real previous behaviour rather than a paraphrase of it.
    """
    miss_penalty = float(miss_penalty)
    costs = np.zeros(buffer.size, dtype=np.float32)
    misses = {}
    for workflow_id, indices in buffer.workflow_groups():
        final_index = indices[-1]
        final_deadline = max(float(buffer.deadline_limits[final_index]), EPS)
        completion = float(buffer.partial_makespans[final_index])
        miss = float(completion > final_deadline + EPS)
        misses[str(workflow_id)] = miss
        increments = np.asarray(
            [max(0.0, float(buffer.deadline_costs[index])) for index in indices],
            dtype=np.float64,
        )
        total_increment = float(increments.sum())
        if total_increment <= EPS:
            weights = np.full(len(indices), 1.0 / max(len(indices), 1))
        else:
            weights = increments / total_increment
        for local_index, index in enumerate(indices):
            deadline = max(float(buffer.deadline_limits[index]), EPS)
            costs[index] = float(
                increments[local_index] / deadline
                + miss_penalty * miss * weights[local_index]
            )
    return costs, misses


class DeadlineQueueCreditRegressionTests(unittest.TestCase):
    """beta_q = 0 must leave the Deadline head target untouched."""

    def test_episode_failure_credit_matches_esp_indicator(self):
        buffer = HighRolloutBuffer()
        for workflow_index in range(4):
            missed = workflow_index >= 2
            for task_index in range(2):
                _add_transition(
                    buffer,
                    workflow_id=f"wf{workflow_index}",
                    increment=1.0,
                    partial=11.0 if missed else 9.0,
                    deadline=10.0,
                    future_regret_normalized=float(task_index + 1),
                )
            buffer.mark_workflow_done(f"wf{workflow_index}")
        costs, misses, failed, responsibility = (
            DualHeadHighLevelD3QNAgent.
            episode_failure_counterfactual_deadline_costs(
                buffer, episode_deadline_target=0.75
            )
        )
        self.assertEqual(sum(misses.values()), 2.0)
        self.assertEqual(failed, 1.0)
        self.assertGreater(responsibility, 0.0)
        self.assertAlmostEqual(float(costs.sum()), 1.0)
        self.assertGreater(float(costs[1]), float(costs[0]))

    def test_successful_episode_has_zero_esp_credit(self):
        buffer = HighRolloutBuffer()
        for workflow_index in range(4):
            _add_transition(
                buffer,
                workflow_id=f"wf{workflow_index}",
                increment=1.0,
                partial=11.0 if workflow_index == 3 else 9.0,
                deadline=10.0,
                future_regret_normalized=1.0,
            )
            buffer.mark_workflow_done(f"wf{workflow_index}")
        costs, _, failed, _ = (
            DualHeadHighLevelD3QNAgent.
            episode_failure_counterfactual_deadline_costs(
                buffer, episode_deadline_target=0.75
            )
        )
        self.assertEqual(failed, 0.0)
        self.assertAlmostEqual(float(costs.sum()), 0.0)

    def test_dense_plus_episode_failure_keeps_old_cost_and_adds_one(self):
        buffer = HighRolloutBuffer()
        for workflow_index in range(4):
            missed = workflow_index >= 2
            for task_index in range(2):
                _add_transition(
                    buffer,
                    workflow_id=f"wf{workflow_index}",
                    increment=1.0,
                    partial=11.0 if missed else 9.0,
                    deadline=10.0,
                    future_regret_normalized=float(task_index + 1),
                )
            buffer.mark_workflow_done(f"wf{workflow_index}")
        dense, _ = (
            DualHeadHighLevelD3QNAgent.
            workflow_dense_miss_credit_deadline_costs(buffer)
        )
        costs, _, failed, _ = (
            DualHeadHighLevelD3QNAgent.
            dense_miss_plus_episode_failure_deadline_costs(
                buffer, episode_deadline_target=0.75
            )
        )
        self.assertEqual(failed, 1.0)
        self.assertTrue(np.all(costs >= dense))
        self.assertAlmostEqual(float(costs.sum() - dense.sum()), 1.0)

    def _random_buffer(self, rng, *, with_queue_values):
        buffer = HighRolloutBuffer()
        for workflow_index in range(6):
            workflow_id = f"wf{workflow_index}"
            deadline = float(rng.uniform(50.0, 800.0))
            partial = 0.0
            for _ in range(int(rng.integers(2, 12))):
                # Half the tasks are off the critical path: increment 0.
                increment = (
                    0.0 if rng.random() < 0.5 else float(rng.uniform(0.0, 90.0))
                )
                partial += increment
                _add_transition(
                    buffer,
                    workflow_id=workflow_id,
                    increment=increment,
                    partial=partial,
                    deadline=deadline,
                    queue_normalized=(
                        float(rng.uniform(0.0, 0.05)) if with_queue_values else 0.0
                    ),
                )
            buffer.mark_workflow_done(workflow_id)
        return buffer

    def test_zero_weight_is_bit_identical_to_legacy(self):
        rng = np.random.default_rng(20260903)
        for trial in range(25):
            buffer = self._random_buffer(rng, with_queue_values=False)
            new_costs, new_misses = (
                DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                    buffer, miss_penalty=1.0, queue_weight=0.0
                )
            )
            old_costs, old_misses = _legacy_dense_miss_credit_costs(
                buffer, miss_penalty=1.0
            )
            np.testing.assert_array_equal(
                new_costs, old_costs, err_msg=f"trial {trial}"
            )
            self.assertEqual(new_misses, old_misses)

    def test_zero_weight_ignores_recorded_queue_values(self):
        """Queue values in the buffer must not leak in when beta_q = 0."""
        rng = np.random.default_rng(20260904)
        for _ in range(25):
            buffer = self._random_buffer(rng, with_queue_values=True)
            new_costs, _ = (
                DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                    buffer, miss_penalty=1.0, queue_weight=0.0
                )
            )
            old_costs, _ = _legacy_dense_miss_credit_costs(buffer, miss_penalty=1.0)
            np.testing.assert_array_equal(new_costs, old_costs)

    def test_zero_sub_weight_ignores_recorded_sub_deadline_values(self):
        buffer = HighRolloutBuffer()
        _add_transition(
            buffer,
            workflow_id="w",
            increment=20.0,
            partial=20.0,
            deadline=100.0,
            sub_deadline_normalized=0.4,
        )
        buffer.mark_workflow_done("w")
        costs, _ = (
            DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                buffer, sub_deadline_weight=0.0
            )
        )
        self.assertAlmostEqual(float(costs[0]), 0.2)

    def test_sub_deadline_credit_enters_dense_cost(self):
        buffer = HighRolloutBuffer()
        _add_transition(
            buffer,
            workflow_id="w",
            increment=0.0,
            partial=20.0,
            deadline=100.0,
            sub_deadline_normalized=0.04,
        )
        buffer.mark_workflow_done("w")
        costs, misses = (
            DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                buffer, sub_deadline_weight=2.0
            )
        )
        self.assertEqual(misses, {"w": 0.0})
        self.assertAlmostEqual(float(costs[0]), 0.08)

    def test_future_finish_regret_enters_dense_cost(self):
        buffer = HighRolloutBuffer()
        _add_transition(
            buffer,
            workflow_id="w",
            increment=20.0,
            partial=20.0,
            deadline=100.0,
            future_regret_normalized=0.4,
        )
        buffer.mark_workflow_done("w")
        costs, misses = (
            DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                buffer, future_regret_weight=0.1
            )
        )
        self.assertEqual(misses, {"w": 0.0})
        self.assertAlmostEqual(float(costs[0]), 0.24)

    def test_zero_future_weight_ignores_recorded_regret(self):
        buffer = HighRolloutBuffer()
        _add_transition(
            buffer,
            workflow_id="w",
            increment=20.0,
            partial=20.0,
            deadline=100.0,
            future_regret_normalized=9.0,
        )
        buffer.mark_workflow_done("w")
        costs, _ = (
            DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                buffer, future_regret_weight=0.0
            )
        )
        self.assertAlmostEqual(float(costs[0]), 0.2)

    def test_all_zero_increments_fall_back_to_uniform_credit_in_both(self):
        """The one place the guard could diverge: no makespan growth at all."""
        buffer = HighRolloutBuffer()
        for _ in range(4):
            _add_transition(
                buffer, workflow_id="flat", increment=0.0, partial=90.0, deadline=40.0
            )
        buffer.mark_workflow_done("flat")

        new_costs, _ = (
            DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                buffer, miss_penalty=1.0, queue_weight=0.0
            )
        )
        old_costs, _ = _legacy_dense_miss_credit_costs(buffer, miss_penalty=1.0)
        np.testing.assert_array_equal(new_costs, old_costs)
        self.assertTrue(np.allclose(new_costs, 0.25))

    def test_negative_queue_weight_is_rejected(self):
        buffer = HighRolloutBuffer()
        _add_transition(
            buffer, workflow_id="w", increment=1.0, partial=1.0, deadline=4.0
        )
        buffer.mark_workflow_done("w")
        with self.assertRaises(ValueError):
            DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                buffer, queue_weight=-0.1
            )


class NonCriticalQueueingBlindSpotTests(unittest.TestCase):
    """The root cause, pinned as a test.

    Workflow ``w`` runs two tasks.  Task A is the longest pole and grows the
    partial makespan; task B is a parallel branch that waits 30 s behind
    another job on the cheap server it was placed on, but finishes before A,
    so ``Delta-C_B = 0``.  Without beta_q, B is invisible to the Deadline head
    in *both* terms while the energy head still credits it for being cheap.
    """

    def _buffer(self, *, deadline, queue_normalized, partial_final):
        buffer = HighRolloutBuffer()
        _add_transition(
            buffer,
            workflow_id="w",
            increment=100.0,
            partial=100.0,
            deadline=deadline,
        )  # task A: the longest pole
        _add_transition(
            buffer,
            workflow_id="w",
            increment=0.0,
            partial=partial_final,
            deadline=deadline,
            queue_normalized=queue_normalized,
        )  # task B: queued, but not the max
        buffer.mark_workflow_done("w")
        return buffer

    def test_queueing_non_critical_task_pays_nothing_without_beta_q(self):
        buffer = self._buffer(
            deadline=200.0, queue_normalized=0.075, partial_final=100.0
        )
        costs, misses = (
            DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                buffer, miss_penalty=1.0, queue_weight=0.0
            )
        )
        self.assertEqual(misses, {"w": 0.0})  # workflow met its deadline
        self.assertEqual(float(costs[1]), 0.0)  # task B: exactly zero
        self.assertGreater(float(costs[0]), 0.0)

    def test_beta_q_charges_the_queueing_task(self):
        buffer = self._buffer(
            deadline=200.0, queue_normalized=0.075, partial_final=100.0
        )
        costs, _ = (
            DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                buffer, miss_penalty=1.0, queue_weight=2.0
            )
        )
        self.assertAlmostEqual(float(costs[1]), 2.0 * 0.075, places=6)
        # Task A is unchanged: it did not queue.
        self.assertAlmostEqual(float(costs[0]), 100.0 / 200.0, places=6)

    def test_miss_credit_share_is_zero_without_beta_q_and_positive_with_it(self):
        """The second, worse zero: weights proportional to Delta-C_t."""
        buffer = self._buffer(
            deadline=80.0, queue_normalized=0.075, partial_final=100.0
        )
        without, misses = (
            DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                buffer, miss_penalty=1.0, queue_weight=0.0
            )
        )
        self.assertEqual(misses, {"w": 1.0})  # 100 s completion vs 80 s deadline
        # Task B gets no dense cost AND no share of the miss penalty.
        self.assertEqual(float(without[1]), 0.0)
        self.assertAlmostEqual(float(without[0]), 100.0 / 80.0 + 1.0, places=5)

        with_queue, _ = (
            DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                buffer, miss_penalty=1.0, queue_weight=2.0
            )
        )
        dense_a = 100.0 / 80.0
        dense_b = 2.0 * 0.075
        total = dense_a + dense_b
        self.assertAlmostEqual(
            float(with_queue[1]), dense_b + dense_b / total, places=5
        )
        self.assertGreater(float(with_queue[1]), 0.0)
        # The credit is redistributed, not invented: the total is unchanged.
        self.assertAlmostEqual(
            float(with_queue.sum()) - float(without.sum()), dense_b, places=5
        )


class ServerQueueDelayTests(unittest.TestCase):
    """``ast - base_ready_time`` isolates the contention the action caused."""

    def test_wait_is_the_gap_between_base_ready_and_actual_start(self):
        delay = HierarchicalObjective._server_queue_delay(
            {"primary_ast": 130.0, "base_ready_time_primary": 100.0}
        )
        self.assertAlmostEqual(delay, 30.0)

    def test_no_wait_when_the_server_is_idle(self):
        delay = HierarchicalObjective._server_queue_delay(
            {"primary_ast": 100.0, "base_ready_time_primary": 100.0}
        )
        self.assertAlmostEqual(delay, 0.0)

    def test_legacy_ast_key_is_accepted(self):
        delay = HierarchicalObjective._server_queue_delay(
            {"AST_primary": 130.0, "base_ready_time_primary": 100.0}
        )
        self.assertAlmostEqual(delay, 30.0)

    def test_missing_timestamps_return_none_rather_than_a_silent_zero(self):
        self.assertIsNone(
            HierarchicalObjective._server_queue_delay({"primary_ast": 130.0})
        )
        self.assertIsNone(
            HierarchicalObjective._server_queue_delay(
                {"base_ready_time_primary": 100.0}
            )
        )

    def test_excludes_the_ready_queue_delay(self):
        """Distinguishes our quantity from ``server_queue_waiting_time``.

        The environment already exposes ``ast - data_ready_time``, which also
        counts the time the task sat in the global ready queue.  That delay is
        incurred wherever the task goes, so charging it would penalise actions
        for something they did not cause.
        """
        event = {
            "primary_ast": 130.0,
            "base_ready_time_primary": 120.0,  # includes current_time = 120
            "data_ready_time_primary": 100.0,  # ready 20 s before dispatch
        }
        self.assertAlmostEqual(
            HierarchicalObjective._server_queue_delay(event), 10.0
        )


class ObjectiveQueueWeightConfigTests(unittest.TestCase):
    def test_default_weight_is_zero(self):
        objective = HierarchicalObjective({})
        self.assertEqual(objective.deadline_queue_weight, 0.0)

    def test_negative_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            HierarchicalObjective({"deadline_queue_weight": -1.0})


if __name__ == "__main__":
    unittest.main()
