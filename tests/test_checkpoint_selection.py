import unittest

from tools.hierarchical_common import metric_rank


class CheckpointSelectionTest(unittest.TestCase):
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
