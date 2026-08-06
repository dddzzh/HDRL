import random
import unittest

from src.datasets.arrival_process import (
    CAI_2025_ARRIVAL_PROCESS,
    create_cai_2025_time_table,
    sample_configured_arrivals,
)


class Cai2025ArrivalProcessTests(unittest.TestCase):
    def test_single_stream_has_only_one_initial_arrival(self):
        arrivals, intervals, source_ids, process = sample_configured_arrivals(
            20,
            {
                "arrival_process": CAI_2025_ARRIVAL_PROCESS,
                "arrival_rate": 0.05,
            },
            rng=random.Random(42),
        )

        self.assertEqual(process, CAI_2025_ARRIVAL_PROCESS)
        self.assertEqual(arrivals[0], 0.0)
        self.assertEqual(sum(value == 0.0 for value in arrivals), 1)
        self.assertEqual(intervals[0], 0.0)
        self.assertTrue(all(value > 0.0 for value in intervals[1:]))
        self.assertTrue(all(arrivals[index] > arrivals[index - 1] for index in range(1, 20)))
        self.assertEqual(source_ids, [0] * 20)

    def test_interarrival_mean_matches_inverse_lambda(self):
        lambda_rate = 0.05
        arrivals = create_cai_2025_time_table(
            20000,
            lambda_rate,
            rng=random.Random(7),
        )
        intervals = [arrivals[index] - arrivals[index - 1] for index in range(1, len(arrivals))]
        empirical_mean = sum(intervals) / len(intervals)
        self.assertAlmostEqual(empirical_mean, 1.0 / lambda_rate, delta=0.5)

    def test_same_seed_reproduces_arrivals(self):
        config = {
            "arrival_process": CAI_2025_ARRIVAL_PROCESS,
            "arrival_rate": 0.05,
        }
        first = sample_configured_arrivals(50, config, rng=random.Random(99))
        second = sample_configured_arrivals(50, config, rng=random.Random(99))
        self.assertEqual(first, second)

    def test_legacy_multi_device_process_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cai_2025_exponential"):
            sample_configured_arrivals(
                5,
                {
                    "arrival_process": "multi_device_poisson",
                    "arrival_rate": 0.05,
                    "num_mobile_devices": 5,
                },
                rng=random.Random(1),
            )


if __name__ == "__main__":
    unittest.main()
