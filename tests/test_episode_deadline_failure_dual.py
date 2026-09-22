"""The dual must regulate ESP, which is a per-episode pass/fail rate.

The paper reports ESP@0.92 -- the fraction of seeds whose *own* Deadline
success rate clears 0.92 -- and never a pooled average.  A dual driven by the
pooled miss rate cannot see that quantity: one run missing four workflows and
four runs missing one each move the pooled mean identically, while the first
costs one ESP point and the second costs none.  These tests pin the episode
metric to that distinction, and pin the shipped tuning to a loop that actually
moves the multiplier instead of clipping it.

Scope guard: this metric changes only the multiplier.  The D3QN Deadline head
keeps its dense credit, so nothing here touches the head's targets.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.D3QN.deadline_lagrangian import DeadlineLagrangianController

CONFIGS = (
    "configs/v2_medium_28edge_9cloud_dvfs_v3.json",
    "configs/v2_small_20edge_6cloud_dvfs_v3.json",
)


def _episode(n_miss, n_workflows=20):
    """One episode's per-workflow binary Deadline misses."""
    return np.array(
        [1.0] * int(n_miss) + [0.0] * (int(n_workflows) - int(n_miss)),
        dtype=np.float64,
    )


def _controller(**overrides):
    settings = {
        "constraint_metric": "episode_deadline_failure",
        "constraint_statistic": "mean",
        "episode_deadline_target": 0.92,
        "deadline_budget": 0.10,
        "deadline_lambda_init": 1.0,
        "dual_learning_rate": 0.1,
        "lambda_max": 6.0,
        "warmup_episodes": 0,
        "window_episodes": 60,
        "window_workflows": 100,
        "proportional_gain": 3.0,
    }
    settings.update(overrides)
    return DeadlineLagrangianController.from_config(settings)


class EpisodeCostTest(unittest.TestCase):
    def test_threshold_is_the_esp_rule_not_the_miss_rate(self):
        """ESP@0.92 over 20 workflows means at most one miss, exactly."""
        for n_miss, expected in ((0, 0.0), (1, 0.0), (2, 1.0), (5, 1.0)):
            metrics = _controller().update(_episode(n_miss), episode=0)
            self.assertEqual(
                metrics["deadline_episode_failed"],
                expected,
                f"{n_miss} misses out of 20",
            )

    def test_boundary_is_not_lost_to_binary_fraction_round_off(self):
        """DSR exactly at the target passes.

        ``2/20`` is exact but ``1 - 0.90`` is 0.09999999999999998, so a naive
        ``miss_rate > 1 - target`` fails a run that met the target precisely.
        """
        metrics = _controller(episode_deadline_target=0.90).update(
            _episode(2), episode=0
        )
        self.assertEqual(metrics["deadline_episode_failed"], 0.0)
        self.assertAlmostEqual(metrics["deadline_episode_success_rate"], 0.90)

    def test_pooled_mean_cannot_distinguish_what_esp_does(self):
        """The reason this metric exists, stated as an assertion.

        Four episodes with one miss each and four with (0, 0, 0, 4) misses have
        the same pooled miss rate.  ESP says the first set is perfect and the
        second loses a quarter of its runs; the dual must agree with ESP.
        """
        spread = _controller()
        for _ in range(4):
            spread_metrics = spread.update(_episode(1), episode=0)
        concentrated = _controller()
        for n_miss in (0, 0, 0, 4):
            concentrated_metrics = concentrated.update(
                _episode(n_miss), episode=0
            )

        self.assertAlmostEqual(
            spread_metrics["deadline_constraint_mean"],
            concentrated_metrics["deadline_constraint_mean"],
        )
        self.assertEqual(spread_metrics["deadline_episode_failure_rate"], 0.00)
        self.assertEqual(
            concentrated_metrics["deadline_episode_failure_rate"], 0.25
        )

    def test_constraint_value_is_the_episode_failure_rate(self):
        controller = _controller()
        for n_miss in (0, 3, 0, 3):
            metrics = controller.update(_episode(n_miss), episode=0)
        self.assertEqual(metrics["deadline_constraint_value"], 0.5)
        self.assertEqual(metrics["deadline_constraint_metric"],
                         "episode_deadline_failure")

    def test_binary_cost_rejects_cvar(self):
        """CVaR of a 0/1 cost is a step function with no gradient to follow."""
        with self.assertRaises(ValueError):
            _controller(constraint_statistic="cvar")

    def test_rate_waits_for_enough_bernoulli_samples(self):
        controller = _controller()
        for episode in range(9):
            metrics = controller.update(_episode(5), episode=episode)
            self.assertEqual(metrics["deadline_dual_updated"], 0.0)
        metrics = controller.update(_episode(5), episode=9)
        self.assertEqual(metrics["deadline_dual_updated"], 1.0)

    def test_target_must_be_a_rate(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                _controller(episode_deadline_target=bad)


class DualRegulationTest(unittest.TestCase):
    """The failure this replaces: lambda pinned at its cap from episode 200 on.

    A railed multiplier makes the constraint metric a no-op -- whatever the
    dual measures, the policy sees the same constant.  The shipped tuning has
    to move lambda in both directions on the failure rates this testbed
    actually produces.
    """

    def _run(self, controller, failure_rate, episodes, rng):
        for episode in range(episodes):
            n_miss = 2 + rng.integers(0, 3) if rng.random() < failure_rate else 0
            controller.update(_episode(int(n_miss)), episode=episode)
        return controller.value

    def test_infeasible_run_climbs_to_the_cap_and_stays(self):
        """When the target is out of reach the cap *is* the operating point.

        lambda_max = 6.0 is the two-block optimum of the frozen-checkpoint
        sweep (min ESP@0.92 = 0.90 on both 542000 and 642000), so railing here
        lands on the best measured multiplier rather than short of it.
        """
        controller = _controller()
        value = self._run(controller, 0.44, 400, np.random.default_rng(0))
        self.assertAlmostEqual(value, 6.0, places=6)

    def test_lambda_rises_well_before_the_run_is_half_over(self):
        """The old dual_learning_rate reached the cap around episode 1100.

        Most of training then happened at a multiplier the sweep scores at
        ESP 0.15, which is worse than useless.
        """
        controller = _controller()
        self._run(controller, 0.44, 200, np.random.default_rng(0))
        self.assertGreater(controller.value, 5.0)

    def test_slack_is_returned_to_energy(self):
        """Beating the target must buy energy back, or the dual is one-way."""
        controller = _controller()
        rng = np.random.default_rng(1)
        self._run(controller, 0.44, 300, rng)
        railed = controller.value
        self._run(controller, 0.02, 900, rng)
        self.assertGreater(railed, 5.0)
        self.assertLess(controller.value, 2.0)

    def test_forgiving_is_slower_than_punishing(self):
        """Asymmetry is deliberate here.

        A symmetric loop collapses lambda to zero the moment the constraint
        holds, the policy immediately loses the margin it just bought, and the
        pair limit-cycles.  Punish fast, forgive slowly.
        """
        controller = _controller()
        ascent = controller.dual_lr * (1.0 - controller.budget)
        descent = controller.dual_lr * controller.budget
        self.assertGreater(ascent, descent)


class ShippedConfigTest(unittest.TestCase):
    def test_both_scales_ship_the_same_dual(self):
        """beta_q is calibrated per scale; the dual is not, so it must match."""
        blocks = []
        for path in CONFIGS:
            with open(Path(path), encoding="utf-8") as handle:
                blocks.append(json.load(handle)["agent"]["high"]["lagrangian"])
        self.assertEqual(blocks[0], blocks[1])

    def test_shipped_high_dual_is_the_episode_metric_at_the_esp_target(self):
        with open(Path(CONFIGS[0]), encoding="utf-8") as handle:
            config = json.load(handle)
        block = config["agent"]["high"]["lagrangian"]
        self.assertEqual(block["constraint_metric"], "episode_deadline_failure")
        # The dual's episode target and the checkpoint rule must agree, or the
        # run optimises one threshold and is accepted on another.
        self.assertEqual(
            block["episode_deadline_target"],
            config["training"]["checkpoint_deadline_target"],
        )
        # ESP@0.92 >= 0.90 is the paper's claim; the budget is its complement.
        self.assertAlmostEqual(block["deadline_budget"], 0.10)
        DeadlineLagrangianController.from_config(block)

    def test_low_dual_keeps_the_workflow_metric(self):
        """SAC is charged for the overrun it causes, per workflow.

        Its counterfactual attribution is per-workflow by construction, so the
        episode metric would throw away exactly the signal that made it work.
        """
        for path in CONFIGS:
            with open(Path(path), encoding="utf-8") as handle:
                block = json.load(handle)["agent"]["low"]["lagrangian"]
            self.assertEqual(block["constraint_metric"], "workflow_miss_rate")


class StateRoundTripTest(unittest.TestCase):
    def test_failure_window_survives_a_resume(self):
        original = _controller()
        for n_miss in (0, 3, 0, 3):
            original.update(_episode(n_miss), episode=0)
        restored = _controller()
        restored.load_state_dict(original.state_dict())
        self.assertEqual(
            list(restored.episode_failure_window),
            list(original.episode_failure_window),
        )
        self.assertAlmostEqual(restored.value, original.value)

    def test_a_multiplier_from_the_old_metric_is_not_restored(self):
        """An old lambda settled against the pooled CVaR sits elsewhere."""
        old = DeadlineLagrangianController.from_config(
            {
                "constraint_metric": "workflow_miss_rate",
                "constraint_statistic": "cvar",
                "deadline_budget": 0.08,
                "lambda_max": 4.0,
                "warmup_episodes": 0,
            }
        )
        old.lambda_deadline = 4.0
        fresh = _controller()
        fresh.load_state_dict(old.state_dict())
        self.assertEqual(fresh.value, fresh.lambda_init)


if __name__ == "__main__":
    unittest.main()
