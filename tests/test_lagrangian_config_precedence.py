"""Checkpoints restore dual *state*; the configured hyper-parameters win.

Regression guard for the defect measured on
``results/hierarchical_dual_head_d3qn_reliability_aware/20260826_091038``:
``load_state_dict`` used to restore ``dual_learning_rate``/``deadline_budget``
from the checkpoint, so every warm-started run silently kept the checkpoint's
tuning and ignored its own config.  With the shipped ``dual_learning_rate =
0.001`` and a rolling miss rate of 0.05-0.08 that gives
``d(lambda)/d(episode) ~ 6e-5``; lambda moved 1.1419 -> 1.1986 over 1200
episodes while ~2.0 was needed to hold Deadline >= 99%.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.D3QN.deadline_lagrangian import DeadlineLagrangianController
from src.sac.deadline_lagrangian import LowDeadlineLagrangianController
from src.sac.reliability_lagrangian import LowReliabilityLagrangianController


TUNING = ("dual_lr", "budget", "lambda_max", "warmup_episodes", "window_workflows")


class DeadlineLagrangianConfigPrecedenceTest(unittest.TestCase):
    """The two Deadline controllers are the ones that had the defect."""

    controllers = (DeadlineLagrangianController, LowDeadlineLagrangianController)

    def _stale(self, factory):
        return factory(
            lambda_init=1.0,
            dual_lr=0.001,
            budget=0.0,
            lambda_max=100.0,
            warmup_episodes=50,
            window_workflows=100,
        )

    def _configured(self, factory):
        return factory(
            lambda_init=1.0,
            dual_lr=0.2,
            budget=0.01,
            lambda_max=20.0,
            warmup_episodes=10,
            window_workflows=40,
        )

    def test_config_hyperparameters_survive_a_checkpoint_load(self):
        for factory in self.controllers:
            with self.subTest(controller=factory.__name__):
                stale = self._stale(factory)
                fresh = self._configured(factory)
                expected = {name: getattr(fresh, name) for name in TUNING}

                fresh.load_state_dict(stale.state_dict())

                for name in TUNING:
                    self.assertEqual(
                        getattr(fresh, name),
                        expected[name],
                        f"{factory.__name__}.{name} was overwritten by the checkpoint",
                    )

    def test_multiplier_and_window_are_still_restored(self):
        for factory in self.controllers:
            with self.subTest(controller=factory.__name__):
                stale = self._stale(factory)
                stale.update([1.0, 1.0, 0.0], episode=1000)
                fresh = self._configured(factory)
                fresh.lambda_deadline = 5.0

                fresh.load_state_dict(stale.state_dict())

                self.assertAlmostEqual(fresh.value, stale.value)
                self.assertEqual(
                    list(fresh.workflow_cost_window),
                    list(stale.workflow_cost_window),
                )

    def test_a_shrunk_window_keeps_only_the_most_recent_costs(self):
        for factory in self.controllers:
            with self.subTest(controller=factory.__name__):
                stale = self._stale(factory)
                stale.update([float(i % 2) for i in range(100)], episode=1000)
                fresh = self._configured(factory)  # window_workflows = 40

                fresh.load_state_dict(stale.state_dict())

                self.assertEqual(len(fresh.workflow_cost_window), 40)
                self.assertEqual(
                    list(fresh.workflow_cost_window),
                    list(stale.workflow_cost_window)[-40:],
                )

    def test_a_smaller_configured_lambda_max_clips_the_restored_multiplier(self):
        for factory in self.controllers:
            with self.subTest(controller=factory.__name__):
                stale = self._stale(factory)
                stale.lambda_deadline = 90.0
                fresh = self._configured(factory)  # lambda_max = 20.0

                fresh.load_state_dict(stale.state_dict())

                self.assertEqual(fresh.value, 20.0)

    def test_overrides_are_reported_rather_than_applied_silently(self):
        for factory in self.controllers:
            with self.subTest(controller=factory.__name__):
                stale = self._stale(factory)
                fresh = self._configured(factory)
                messages = fresh.hyperparameter_overrides(stale.state_dict())
                joined = " ".join(messages)
                self.assertIn("dual_learning_rate", joined)
                self.assertIn("deadline_budget", joined)

    def test_matching_checkpoint_and_config_report_nothing(self):
        for factory in self.controllers:
            with self.subTest(controller=factory.__name__):
                source = self._configured(factory)
                target = self._configured(factory)
                self.assertEqual(
                    target.hyperparameter_overrides(source.state_dict()), []
                )

    def test_dual_step_uses_the_configured_learning_rate_after_a_load(self):
        # The end-to-end symptom: before the fix the restored 0.001 made this
        # step ~200x too small.
        for factory in self.controllers:
            with self.subTest(controller=factory.__name__):
                stale = self._stale(factory)
                fresh = self._configured(factory)
                fresh.load_state_dict(stale.state_dict())
                before = fresh.value
                fresh.update([1.0] * 20, episode=1000)
                self.assertAlmostEqual(fresh.value - before, 0.2 * (1.0 - 0.01))


class ReliabilityLagrangianAlreadyCorrectTest(unittest.TestCase):
    """The reliability controller never had the defect; keep it that way."""

    def test_config_hyperparameters_survive_a_checkpoint_load(self):
        stale = LowReliabilityLagrangianController(
            lambda_init=1.0, dual_lr=0.001, budget=0.0,
            lambda_max=100.0, warmup_episodes=0, window_workflows=100,
        )
        fresh = LowReliabilityLagrangianController(
            lambda_init=5.0, dual_lr=0.5, budget=0.005,
            lambda_max=200.0, warmup_episodes=10, window_workflows=40,
        )
        fresh.load_state_dict(stale.state_dict())
        self.assertEqual(fresh.dual_lr, 0.5)
        self.assertEqual(fresh.budget, 0.005)
        self.assertEqual(fresh.lambda_max, 200.0)
        self.assertEqual(fresh.window_workflows, 40)


if __name__ == "__main__":
    unittest.main()
