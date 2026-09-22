import unittest

from tools.hierarchical_common import combine_checkpoint_metrics, metric_rank


class CheckpointSelectionTest(unittest.TestCase):
    def test_esp_gate_uses_worst_validation_block_then_minimizes_energy(self):
        feasible = combine_checkpoint_metrics(
            [
                {"deadline_success_rate": 0.98, "reliability_success_rate": 1.0,
                 "total_energy": 320.0, "deadline_esp_0p90": 0.95},
                {"deadline_success_rate": 0.96, "reliability_success_rate": 1.0,
                 "total_energy": 300.0, "deadline_esp_0p90": 0.90},
            ]
        )
        cheap_but_infeasible = combine_checkpoint_metrics(
            [
                {"deadline_success_rate": 0.96, "reliability_success_rate": 1.0,
                 "total_energy": 250.0, "deadline_esp_0p90": 0.95},
                {"deadline_success_rate": 0.93, "reliability_success_rate": 1.0,
                 "total_energy": 230.0, "deadline_esp_0p90": 0.85},
            ]
        )
        kwargs = dict(deadline_esp_threshold=0.90, deadline_esp_target=0.90,
                      reliability_target=1.0)
        self.assertEqual(feasible["deadline_esp_0p90_min_block"], 0.90)
        self.assertGreater(metric_rank(feasible, **kwargs),
                           metric_rank(cheap_but_infeasible, **kwargs))

        cheaper_feasible = dict(feasible, total_energy=290.0)
        self.assertGreater(metric_rank(cheaper_feasible, **kwargs),
                           metric_rank(feasible, **kwargs))

    def test_esp_mode_ignores_mean_deadline_when_no_mean_target_is_set(self):
        high_mean = {
            "deadline_success_rate": 0.99,
            "reliability_success_rate": 1.0,
            "deadline_esp_0p90": 0.90,
            "total_energy": 400.0,
        }
        low_mean = {
            "deadline_success_rate": 0.91,
            "reliability_success_rate": 1.0,
            "deadline_esp_0p90": 0.90,
            "total_energy": 300.0,
        }
        kwargs = dict(
            deadline_esp_threshold=0.90,
            deadline_esp_target=0.90,
            reliability_target=1.0,
        )
        self.assertGreater(
            metric_rank(low_mean, **kwargs),
            metric_rank(high_mean, **kwargs),
        )

    def test_multiple_validation_blocks_require_each_block_to_be_feasible(self):
        combined = combine_checkpoint_metrics(
            [
                {
                    "deadline_success_rate": 0.95,
                    "reliability_success_rate": 1.0,
                    "total_energy": 300.0,
                },
                {
                    "deadline_success_rate": 0.90,
                    "reliability_success_rate": 1.0,
                    "total_energy": 500.0,
                },
            ]
        )
        self.assertEqual(combined["deadline_success_rate"], 0.90)
        self.assertEqual(combined["reliability_success_rate"], 1.0)
        self.assertEqual(combined["total_energy"], 400.0)
        self.assertEqual(
            metric_rank(combined, deadline_target=0.92, reliability_target=1.0)[0],
            0.0,
        )

    def test_reliability_target_blocks_lower_energy_infeasible_checkpoint(self):
        reliable = {
            "deadline_success_rate": 0.95,
            "reliability_success_rate": 1.0,
            "total_energy": 250.0,
            "average_workflow_response_time": 10.0,
        }
        unreliable = {
            "deadline_success_rate": 1.0,
            "reliability_success_rate": 0.95,
            "total_energy": 100.0,
            "average_workflow_response_time": 5.0,
        }
        self.assertGreater(
            metric_rank(
                reliable, deadline_target=0.95, reliability_target=1.0
            ),
            metric_rank(
                unreliable, deadline_target=0.95, reliability_target=1.0
            ),
        )

    def test_feasible_checkpoint_minimizes_energy_before_extra_deadline_margin(self):
        exact_deadline = {
            "deadline_success_rate": 1.0,
            "reliability_success_rate": 1.0,
            "total_energy": 360.0,
            "average_response_time": 20.0,
        }
        lower_energy = {
            "deadline_success_rate": 0.9925,
            "reliability_success_rate": 1.0,
            "total_energy": 299.0,
            "average_response_time": 25.0,
        }
        self.assertGreater(
            metric_rank(lower_energy, deadline_target=0.99),
            metric_rank(exact_deadline, deadline_target=0.99),
        )

    def test_feasible_checkpoint_always_beats_infeasible_checkpoint(self):
        feasible = {
            "deadline_success_rate": 0.99,
            "reliability_success_rate": 1.0,
            "total_energy": 400.0,
        }
        infeasible = {
            "deadline_success_rate": 0.9875,
            "reliability_success_rate": 1.0,
            "total_energy": 100.0,
        }
        self.assertGreater(
            metric_rank(feasible, deadline_target=0.99),
            metric_rank(infeasible, deadline_target=0.99),
        )

    def test_infeasible_checkpoint_maximizes_deadline_first(self):
        higher_deadline = {
            "deadline_success_rate": 0.80,
            "reliability_success_rate": 1.0,
            "total_energy": 400.0,
        }
        lower_deadline = {
            "deadline_success_rate": 0.70,
            "reliability_success_rate": 1.0,
            "total_energy": 100.0,
        }
        self.assertGreater(
            metric_rank(higher_deadline, deadline_target=0.99),
            metric_rank(lower_deadline, deadline_target=0.99),
        )

    def test_legacy_ranking_is_unchanged_without_target(self):
        higher_deadline = {
            "deadline_success_rate": 1.0,
            "reliability_success_rate": 1.0,
            "total_energy": 360.0,
        }
        lower_energy = {
            "deadline_success_rate": 0.9925,
            "reliability_success_rate": 1.0,
            "total_energy": 299.0,
        }
        self.assertGreater(metric_rank(higher_deadline), metric_rank(lower_energy))


if __name__ == "__main__":
    unittest.main()
