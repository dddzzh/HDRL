from __future__ import annotations

import unittest

from tools.eval_all_baselines import (
    DEFAULT_MODES,
    paired_energy_deltas,
    parse_csv_values,
    summarize_rows,
)


class TestAllBaselineEvaluation(unittest.TestCase):
    def test_mode_and_scale_parsing_is_ordered_and_unique(self):
        self.assertEqual(
            parse_csv_values("small,medium,small", {"small", "medium"}, "scale"),
            ["small", "medium"],
        )
        self.assertEqual(
            parse_csv_values("fcfs_sac,d3qn_sac", set(DEFAULT_MODES), "mode"),
            ["fcfs_sac", "d3qn_sac"],
        )

    def test_paired_energy_delta_uses_matching_seed(self):
        rows = [
            {"scale": "small", "mode": "d3qn_sac", "eval_seed": 1, "total_energy": 100.0},
            {"scale": "small", "mode": "fcfs_sac", "eval_seed": 1, "total_energy": 120.0},
            {"scale": "small", "mode": "d3qn_sac", "eval_seed": 2, "total_energy": 200.0},
            {"scale": "small", "mode": "fcfs_sac", "eval_seed": 2, "total_energy": 180.0},
        ]
        deltas = paired_energy_deltas(rows)
        self.assertAlmostEqual(deltas[("small", "fcfs_sac")]["mean"], 5.0)

    def test_summary_keeps_complete_system_energy(self):
        base = {
            "scale": "small",
            "mode": "d3qn_sac",
            "num_workflows": 20,
            "total_energy": 100.0,
            "energy_per_workflow": 5.0,
            "energy_per_task": 1.0,
            "average_response_time": 2.0,
            "global_makespan": 3.0,
            "deadline_success_rate": 1.0,
            "reliability_success_rate": 1.0,
            "rho_mean": 0.8,
            "low_sac_total_energy_saving_ratio": 0.1,
            "steps": 100,
        }
        rows = [
            {**base, "eval_seed": 1},
            {**base, "eval_seed": 2, "total_energy": 300.0},
        ]
        summary = summarize_rows(rows)[0]
        self.assertAlmostEqual(summary["metrics"]["total_energy"]["mean"], 200.0)
        self.assertEqual(summary["metrics"]["total_energy"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
