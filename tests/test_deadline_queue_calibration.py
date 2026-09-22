"""beta_q must be solved, not configured, and the solve must be scale-free.

The point of the calibrator is that one config value -- a share -- produces a
working weight on any testbed size. These tests pin that property directly:
two rollouts whose queue mass differs by a factor of four must still land on
the requested share, because that is what lets small and middle ship the same
config.
"""
from __future__ import annotations

import unittest

import numpy as np

from src.D3QN.dual_head_constrained import DualHeadHighLevelD3QNAgent
from src.D3QN.queue_weight_calibrator import DeadlineQueueWeightCalibrator
from src.D3QN.reward import HierarchicalObjective
from src.ppo.hierarchical_rollout import HighRolloutBuffer

from tests.test_deadline_queue_credit import _add_transition


def _episode(workflows=3, tasks=4, increment=1.0, queue=0.05, deadline=40.0):
    """One episode of rollout with a known Delta-C and queue mass."""
    buffer = HighRolloutBuffer()
    for workflow in range(workflows):
        partial = 0.0
        for _ in range(tasks):
            partial += increment
            _add_transition(
                buffer,
                workflow_id=f"w{workflow}",
                increment=increment,
                partial=partial,
                deadline=deadline,
                queue_normalized=queue,
            )
    return buffer


class QueueWeightCalibratorTest(unittest.TestCase):
    def test_solved_weight_realizes_the_requested_share(self):
        calibrator = DeadlineQueueWeightCalibrator(
            0.20, start_episode=0, window_episodes=1
        )
        weight = calibrator.observe(_episode(), episode=0)
        self.assertIsNotNone(weight)
        self.assertAlmostEqual(calibrator.realized_share(), 0.20, places=9)

    def test_share_is_scale_free_where_the_raw_weight_is_not(self):
        """Same share target, rollouts that queue 4x differently.

        This is the whole reason the config holds a share: the measured queue
        mass differs by 1.87x between the 20+6 and 28+9 testbeds, so a shared
        raw weight would buy different amounts of signal on each.
        """
        weights = []
        for queue in (0.02, 0.08):
            calibrator = DeadlineQueueWeightCalibrator(
                0.25, start_episode=0, window_episodes=1
            )
            weights.append(calibrator.observe(_episode(queue=queue), episode=0))
            self.assertAlmostEqual(calibrator.realized_share(), 0.25, places=9)
        # The shares match while the weights that produce them do not.
        self.assertAlmostEqual(weights[0] / weights[1], 4.0, places=6)

    def test_solve_matches_the_production_credit_function(self):
        """The realized share must show up in the head's actual target.

        The calibrator sums the masses itself; if that summation drifts from
        ``workflow_dense_miss_credit_deadline_costs`` the requested share would
        be silently wrong. Compare against the real function with beta_q on.
        """
        buffer = _episode(queue=0.03)
        calibrator = DeadlineQueueWeightCalibrator(
            0.30, start_episode=0, window_episodes=1
        )
        weight = calibrator.observe(buffer, episode=0)

        with_queue, _ = (
            DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                buffer, miss_penalty=0.0, queue_weight=weight
            )
        )
        without_queue, _ = (
            DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                buffer, miss_penalty=0.0, queue_weight=0.0
            )
        )
        total = float(np.sum(with_queue))
        queue_share = (total - float(np.sum(without_queue))) / total
        self.assertAlmostEqual(queue_share, 0.30, places=6)

    def test_window_does_not_open_before_the_start_episode(self):
        calibrator = DeadlineQueueWeightCalibrator(
            0.20, start_episode=150, window_episodes=1
        )
        self.assertIsNone(calibrator.observe(_episode(), episode=149))
        self.assertEqual(calibrator.observed_episodes, 0)
        self.assertIsNotNone(calibrator.observe(_episode(), episode=150))

    def test_solve_waits_for_the_full_window(self):
        calibrator = DeadlineQueueWeightCalibrator(
            0.20, start_episode=0, window_episodes=3
        )
        self.assertIsNone(calibrator.observe(_episode(), episode=0))
        self.assertIsNone(calibrator.observe(_episode(), episode=1))
        self.assertIsNotNone(calibrator.observe(_episode(), episode=2))

    def test_weight_freezes_after_solving(self):
        """Re-solving each episode is a feedback loop, so it must not happen.

        beta_q suppresses queueing, which shrinks the queue mass, which would
        push beta_q back up to hold the share.
        """
        calibrator = DeadlineQueueWeightCalibrator(
            0.20, start_episode=0, window_episodes=1
        )
        weight = calibrator.observe(_episode(queue=0.05), episode=0)
        self.assertIsNone(calibrator.observe(_episode(queue=0.001), episode=1))
        self.assertEqual(calibrator.weight, weight)

    def test_no_contention_resolves_to_zero_not_infinity(self):
        calibrator = DeadlineQueueWeightCalibrator(
            0.20, start_episode=0, window_episodes=1
        )
        self.assertEqual(calibrator.observe(_episode(queue=0.0), episode=0), 0.0)

    def test_share_zero_disables_the_calibrator(self):
        calibrator = DeadlineQueueWeightCalibrator(
            0.0, start_episode=0, window_episodes=1
        )
        self.assertFalse(calibrator.enabled)
        self.assertIsNone(calibrator.observe(_episode(), episode=0))

    def test_rejects_out_of_range_share(self):
        for bad in (-0.1, 1.0, 1.5):
            with self.assertRaises(ValueError):
                DeadlineQueueWeightCalibrator(bad)

    def test_state_round_trip_carries_the_frozen_weight(self):
        """A resumed run has no other source for beta_q; it is not in the config."""
        original = DeadlineQueueWeightCalibrator(
            0.20, start_episode=0, window_episodes=1
        )
        weight = original.observe(_episode(), episode=0)
        restored = DeadlineQueueWeightCalibrator(
            0.20, start_episode=0, window_episodes=1
        )
        restored.load_state_dict(original.state_dict())
        self.assertEqual(restored.weight, weight)
        self.assertTrue(restored.resolved)
        self.assertIsNone(restored.observe(_episode(), episode=1))

    def test_state_round_trip_resumes_a_partial_window(self):
        original = DeadlineQueueWeightCalibrator(
            0.20, start_episode=0, window_episodes=2
        )
        original.observe(_episode(), episode=0)
        restored = DeadlineQueueWeightCalibrator(
            0.20, start_episode=0, window_episodes=2
        )
        restored.load_state_dict(original.state_dict())
        self.assertEqual(restored.observed_episodes, 1)
        self.assertIsNotNone(restored.observe(_episode(), episode=1))
        self.assertAlmostEqual(restored.realized_share(), 0.20, places=9)

    def test_resume_rejects_a_changed_share(self):
        original = DeadlineQueueWeightCalibrator(
            0.20, start_episode=0, window_episodes=1
        )
        original.observe(_episode(), episode=0)
        with self.assertRaises(ValueError):
            DeadlineQueueWeightCalibrator(
                0.30, start_episode=0, window_episodes=1
            ).load_state_dict(original.state_dict())


class ObjectiveQueueShareTest(unittest.TestCase):
    def test_share_and_weight_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            HierarchicalObjective(
                {"deadline_queue_share": 0.2, "deadline_queue_weight": 5.0}
            )

    def test_share_defaults_to_disabled(self):
        objective = HierarchicalObjective({})
        self.assertEqual(objective.deadline_queue_share, 0.0)
        self.assertEqual(objective.deadline_queue_weight, 0.0)

    def test_rejects_out_of_range_share(self):
        with self.assertRaises(ValueError):
            HierarchicalObjective({"deadline_queue_share": 1.0})

    def test_state_dict_carries_the_calibrated_weight(self):
        objective = HierarchicalObjective({"deadline_queue_share": 0.2})
        objective.set_deadline_queue_weight(9.5)
        objective.queue_calibration_state = {"share": 0.2, "weight": 9.5}

        restored = HierarchicalObjective({"deadline_queue_share": 0.2})
        restored.load_state_dict(objective.state_dict())
        self.assertEqual(restored.deadline_queue_weight, 9.5)
        self.assertEqual(restored.queue_calibration_state["weight"], 9.5)

    def test_set_weight_rejects_negative(self):
        objective = HierarchicalObjective({"deadline_queue_share": 0.2})
        with self.assertRaises(ValueError):
            objective.set_deadline_queue_weight(-1.0)


if __name__ == "__main__":
    unittest.main()
