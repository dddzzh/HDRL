import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE = ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE.is_dir() and str(LOCAL_SITE) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE))

import numpy as np

from src.D3QN.reliability_aware_high import (
    ReliabilityAwareConstrainedD3QN,
    ReliabilityLagrangian,
    augment_high_state,
    reliability_cost,
)


class TestReliabilityAwareHigh(unittest.TestCase):
    def test_log_cost_and_lagrangian(self):
        self.assertAlmostEqual(reliability_cost(0.9, 0.9), 0.0)
        self.assertGreater(reliability_cost(0.8, 0.9), 0.0)
        controller = ReliabilityLagrangian()
        row = controller.update(reliability_cost(0.8, 0.9))
        self.assertGreater(row["lambda_reliability"], 1.0)

    def test_state_augmentation_only_changes_high_state(self):
        state = augment_high_state(np.zeros(4), np.log(0.95), np.log(0.9), [1e-4, 2e-4])
        self.assertEqual(state.shape, (9,))
        self.assertTrue(np.isfinite(state).all())

    def test_three_heads_forward_select_and_learn(self):
        agent = ReliabilityAwareConstrainedD3QN(
            state_dim=5, action_dim=3, hidden_dim=16, hidden_layers=1,
            batch_size=2, replay_size=8, device="cpu", seed=7,
        )
        state = np.zeros(5, dtype=np.float32)
        next_state = np.ones(5, dtype=np.float32)
        for i in range(2):
            agent.remember(state, i, 1.0, 0.1, 0.2, next_state, i == 1,
                           [1, 1, 1], [1, 1, 1])
        combined, energy, deadline, reliability = agent.combined_action_q_values(state, 1.0, 1.0)
        self.assertEqual(reliability.shape, (3,))
        metrics = agent.learn(1.0, 1.0)
        self.assertIn("reliability_loss", metrics)


if __name__ == "__main__":
    unittest.main()
