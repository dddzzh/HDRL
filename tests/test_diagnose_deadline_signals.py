import unittest

from tools.diagnose_deadline_signals import (
    beta_for_target_share,
    sub_deadline_lateness,
)


class DeadlineSignalDiagnosticTest(unittest.TestCase):
    def test_linear_sub_deadline_lateness_has_no_unit_jump(self):
        self.assertEqual(sub_deadline_lateness(10.0, 10.0, 20, 100.0), 0.0)
        self.assertAlmostEqual(
            sub_deadline_lateness(11.0, 10.0, 20, 100.0), 1.0 / 2000.0
        )

    def test_beta_hits_requested_share(self):
        beta = beta_for_target_share(8.0, 2.0, 0.2)
        self.assertAlmostEqual(beta, 1.0)
        self.assertAlmostEqual(beta * 2.0 / (8.0 + beta * 2.0), 0.2)

    def test_zero_sub_signal_has_no_finite_beta(self):
        self.assertIsNone(beta_for_target_share(1.0, 0.0, 0.2))


if __name__ == "__main__":
    unittest.main()
