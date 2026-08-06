import unittest

from tools.eval_hierarchical_generalization import (
    add_normalized_metrics,
    build_cases,
    parse_modes,
    parse_positive_ints,
    parse_vm_layouts,
    summarize_rows,
)


class HierarchicalGeneralizationTest(unittest.TestCase):
    def test_scale_parsers_reject_invalid_values(self):
        self.assertEqual(parse_positive_ints("5,10,10,20"), [5, 10, 20])
        self.assertEqual(parse_vm_layouts("5:2,10:3"), [(5, 2), (10, 3)])
        self.assertEqual(parse_modes("d3qn_sac,heft_same_sac"), ["d3qn_sac", "heft_same_sac"])
        with self.assertRaises(ValueError):
            parse_positive_ints("0")
        with self.assertRaises(ValueError):
            parse_vm_layouts("20")
        with self.assertRaises(ValueError):
            parse_modes("ppo")

    def test_cases_keep_other_dimension_at_training_baseline(self):
        config = {
            "generator": {"num_workflows": 10},
            "resources": {"num_edge": 20, "num_cloud": 6},
        }
        cases = build_cases(config, [5, 20], [(10, 3)])
        self.assertEqual(cases[0]["num_workflows"], 5)
        self.assertEqual(cases[0]["num_edge"], 20)
        self.assertEqual(cases[1]["num_workflows"], 20)
        self.assertEqual(cases[2]["num_workflows"], 10)
        self.assertEqual(cases[2]["num_edge"] + cases[2]["num_cloud"], 13)

    def test_normalized_energy_and_confidence_interval_summary(self):
        case = {"num_workflows": 10}
        metrics = add_normalized_metrics({"total_energy": 1000.0, "steps": 200}, case)
        self.assertAlmostEqual(metrics["energy_per_workflow"], 100.0)
        self.assertAlmostEqual(metrics["energy_per_task"], 5.0)
        base = {
            "case": "workflows_10",
            "dimension": "workflow_count",
            "scale_value": 10,
            "num_workflows": 10,
            "num_edge": 20,
            "num_cloud": 6,
            "num_servers": 26,
            "mode": "d3qn_sac",
            "average_response_time": 1.5,
            "global_makespan": 10.0,
            "deadline_success_rate": 1.0,
            "reliability_success_rate": 1.0,
            "mean_task_time_regret": 0.2,
            "low_sac_total_energy_saving_ratio": 0.1,
            "rho_mean": 0.9,
        }
        rows = [
            {**base, "total_energy": 1000.0, "energy_per_workflow": 100.0, "energy_per_task": 5.0},
            {**base, "total_energy": 1200.0, "energy_per_workflow": 120.0, "energy_per_task": 6.0},
        ]
        summary = summarize_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["metrics"]["total_energy"]["mean"], 1100.0)
        self.assertGreater(summary[0]["metrics"]["total_energy"]["ci95_half_width"], 0.0)


if __name__ == "__main__":
    unittest.main()
