import math
import unittest

from src.simulator.resource import (
    compute_energy,
    compute_rho_energy_efficient,
    map_discrete_rho_to_server_range,
    _normalize_server,
)


def make_server(static_power, dynamic_power, min_rho=0.1):
    server = {
        "id": 0,
        "name": "s0",
        "type": "edge",
        "max_freq_ghz": 3.0,
        "static_power": static_power,
        "dynamic_power": dynamic_power,
        "min_rho": min_rho,
    }
    _normalize_server(server)
    return server


class RhoEnergyEfficientTest(unittest.TestCase):
    def test_matches_closed_form_cube_root(self):
        server = make_server(static_power=4.08, dynamic_power=46.92)
        expected = (4.08 / (2.0 * 46.92)) ** (1.0 / 3.0)
        self.assertAlmostEqual(server["rho_energy_efficient"], expected, places=12)

    def test_is_the_actual_energy_minimizer(self):
        # Scan the whole range and confirm the closed form really is the argmin.
        server = make_server(static_power=25.74, dynamic_power=91.26)
        rho_ee = server["rho_energy_efficient"]
        workload = 5.0e9
        best_rho, best_energy = None, math.inf
        for step in range(1, 1001):
            rho = step / 1000.0
            energy = compute_energy(workload, server, rho)
            if energy < best_energy:
                best_rho, best_energy = rho, energy
        self.assertAlmostEqual(best_rho, rho_ee, places=2)
        # And going below it really does cost more.
        self.assertGreater(
            compute_energy(workload, server, max(rho_ee - 0.15, 0.01)),
            compute_energy(workload, server, rho_ee),
        )

    def test_clamped_to_hardware_minimum_and_one(self):
        # P_ind > 2 C_ef makes the unconstrained optimum exceed 1: never scale down.
        self.assertEqual(compute_rho_energy_efficient(make_server(86.0, 31.0)), 1.0)
        # A tiny P_ind pushes it under the hardware floor.
        floored = make_server(1e-6, 100.0, min_rho=0.3)
        self.assertEqual(floored["rho_energy_efficient"], 0.3)

    def test_old_and_new_power_split_agree_at_rho_one(self):
        # The v3 config re-splits P_s/P_d while preserving their sum, so rho=1
        # energy is unchanged.
        old = make_server(static_power=86.0, dynamic_power=31.0)
        new = make_server(static_power=25.74, dynamic_power=91.26)
        self.assertAlmostEqual(
            compute_energy(5.0e9, old, 1.0), compute_energy(5.0e9, new, 1.0), places=9
        )
        # ... but the new split has real DVFS room and the old one has none.
        self.assertEqual(old["rho_energy_efficient"], 1.0)
        self.assertLess(new["rho_energy_efficient"], 0.6)

    def test_min_freq_hz_stays_at_the_hardware_limit(self):
        # f_min normalizes the fault-rate model and must not follow rho_ee.
        server = make_server(static_power=25.74, dynamic_power=91.26, min_rho=0.1)
        self.assertAlmostEqual(server["min_freq_hz"], 0.1 * server["max_freq_hz"])
        self.assertGreater(server["rho_energy_efficient"], 0.1)

    def test_discrete_levels_preserve_count_and_spacing_above_rho_ee(self):
        server = make_server(static_power=25.74, dynamic_power=91.26)
        original = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        mapped = [
            map_discrete_rho_to_server_range(server, original, index)
            for index in range(len(original))
        ]
        self.assertEqual(len(mapped), len(original))
        self.assertAlmostEqual(mapped[0], server["rho_energy_efficient"])
        self.assertAlmostEqual(mapped[-1], 1.0)
        self.assertTrue(all(left < right for left, right in zip(mapped, mapped[1:])))


if __name__ == "__main__":
    unittest.main()
