from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def load(name):
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


class TestBaselineConfigSync(unittest.TestCase):
    def test_baselines_use_the_current_v3_environment(self):
        cases = {
            "v2_small_20edge_6cloud_dvfs_v3.json": (
                "dqn_edge_small_20edge_6cloud.json",
                "dream_dqn.json",
                "oo_actor_critic_small_20edge_6cloud.json",
            ),
            "v2_medium_28edge_9cloud_dvfs_v3.json": (
                "dqn_edge_medium_28edge_9cloud.json",
                "dream_dqn_medium_28edge_9cloud.json",
                "oo_actor_critic_medium_28edge_9cloud.json",
            ),
        }
        for reference_name, baseline_names in cases.items():
            reference = load(reference_name)
            for baseline_name in baseline_names:
                baseline = load(baseline_name)
                with self.subTest(config=baseline_name):
                    self.assertEqual(baseline["generator"], reference["generator"])
                    self.assertEqual(baseline["resources"], reference["resources"])
                    self.assertEqual(baseline["simulation"], reference["simulation"])
                    training = baseline["training"]
                    expected_eval_episodes = (
                        5 if "medium_28edge_9cloud" in baseline_name else 20
                    )
                    self.assertEqual(
                        training["eval_episodes"], expected_eval_episodes
                    )
                    self.assertEqual(training["eval_seed"], 542000)
                    if baseline_name == "dream_dqn.json":
                        self.assertEqual(
                            training["checkpoint_deadline_target"], 0.92
                        )
                    else:
                        self.assertNotIn("checkpoint_deadline_target", training)
                        self.assertEqual(
                            training["checkpoint_deadline_esp_threshold"], 0.90
                        )
                        self.assertEqual(
                            training["checkpoint_deadline_esp_target"], 0.90
                        )
                    if "medium_28edge_9cloud" in baseline_name:
                        self.assertEqual(training["eval_interval"], 50)
                        self.assertEqual(training["save_every"], 50)
                    self.assertEqual(training["checkpoint_reliability_target"], 1.0)

    def test_obsolete_tuned_config_is_removed(self):
        self.assertFalse(
            (CONFIGS / "dqn_edge_medium_28edge_9cloud_tuned_b.json").exists()
        )


if __name__ == "__main__":
    unittest.main()
