"""Closed-form audit of the DVFS operating window.

Answers two questions without running the simulator:

1. Energy side -- for each server profile, where does ``E(rho)`` bottom out?
   ``E(rho) = (L / f_max) * (P_s / rho + P_d * rho^2)`` has an interior minimum at
   ``rho_ee = (P_s / (2 P_d))^(1/3)``.  Below ``rho_ee`` slowing down *costs*
   energy, so any action below it is provably dominated.

2. Reliability side -- how much of the workflow risk budget
   ``B = -ln(R_req)`` is already spent at ``rho = 1``?  DVFS multiplies the
   spent risk by ``g(rho) = 10^(d (1-rho)/(1-rho_min)) / rho``, so the lowest
   admissible frequency solves ``g(rho) = B / risk(rho=1)``.

Usage::

    py -3.13 tools/analyze_dvfs_feasible_window.py
    py -3.13 tools/analyze_dvfs_feasible_window.py --lambda-scale 0.02 --d 3.0
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Mirrors src/datasets/pegasus_workflow_generator.py
PEGASUS_XML_FILES = (
    "CyberShake_30.xml",
    "Epigenomics_24.xml",
    "Inspiral_30.xml",
    "Montage_25.xml",
    "Sipht_30.xml",
    "CyberShake_50.xml",
    "Epigenomics_46.xml",
    "Inspiral_50.xml",
    "Montage_50.xml",
    "Sipht_60.xml",
)
DATASET_SCALE_LIMITS = {"small": 5, "middle": 10}
MEAN_WORKLOAD_GC = (0.5 + 20.0) / 2.0  # rng.uniform(WORKLOAD_GC_MIN, WORKLOAD_GC_MAX)


def count_tasks(path: Path) -> int:
    namespace = "{http://pegasus.isi.edu/schema/DAX}"
    root = ET.parse(path).getroot()
    return len([n for n in root.findall(f"{namespace}job") if n.get("id") is not None])


def rho_energy_efficient(p_static: float, p_dynamic: float) -> float:
    """rho* = (P_s / (2 P_d))^(1/3), clipped to (0, 1]."""
    return min(1.0, (p_static / (2.0 * p_dynamic)) ** (1.0 / 3.0))


def energy_ratio(p_static: float, p_dynamic: float, rho: float) -> float:
    """E(rho) / E(1) at fixed workload."""
    return (p_static / rho + p_dynamic * rho * rho) / (p_static + p_dynamic)


def risk_multiplier(rho: float, d: float, rho_min: float) -> float:
    """g(rho): how much the per-task failure risk lambda*t grows at frequency rho."""
    return 10.0 ** (d * (1.0 - rho) / (1.0 - rho_min)) / rho


def solve_min_rho(budget_ratio: float, d: float, rho_min: float) -> float:
    """Smallest rho with g(rho) <= budget_ratio (bisection; g is decreasing in rho)."""
    if budget_ratio <= 1.0:
        return 1.0
    if risk_multiplier(rho_min, d, rho_min) <= budget_ratio:
        return rho_min
    lo, hi = rho_min, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if risk_multiplier(mid, d, rho_min) > budget_ratio:
            lo = mid
        else:
            hi = mid
    return hi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", default="configs/resources_edge_cloud_default.json")
    parser.add_argument("--r-req", type=float, default=0.9, help="workflow reliability requirement")
    parser.add_argument("--d", type=float, default=None, help="override failure_rate_exponent")
    parser.add_argument(
        "--lambda-scale",
        type=float,
        default=1.0,
        help="multiply every profile failure_rate (1.0 = current config)",
    )
    parser.add_argument(
        "--mean-freq-ghz",
        type=float,
        default=3.078,
        help="task-count-weighted mean execution frequency observed in the small run",
    )
    args = parser.parse_args()

    resources = json.loads(
        io.open(PROJECT_ROOT / args.resources, encoding="utf-8").read()
    )
    profiles = resources["server_types"]

    print("=" * 78)
    print("1) ENERGY SIDE -- minimum energy-efficient frequency per profile")
    print("=" * 78)
    print(f"{'profile':<16}{'P_s':>7}{'P_d':>7}{'P_s/P_d':>9}{'rho_ee':>9}"
          f"{'E(rho_ee)/E(1)':>16}{'max saving':>12}")
    for name, spec in profiles.items():
        p_s = float(spec["static_power"])
        p_d = float(spec["dynamic_power"])
        rho_ee = rho_energy_efficient(p_s, p_d)
        ratio = energy_ratio(p_s, p_d, rho_ee)
        print(f"{name:<16}{p_s:>7.0f}{p_d:>7.0f}{p_s / p_d:>9.2f}{rho_ee:>9.3f}"
              f"{ratio:>16.3f}{1.0 - ratio:>11.1%}")

    print()
    print("=" * 78)
    print("2) RELIABILITY SIDE -- risk budget already spent at rho = 1")
    print("=" * 78)
    budget = -math.log(args.r_req)
    print(f"R_req = {args.r_req}  ->  risk budget B = -ln(R_req) = {budget:.5f}")
    print(f"mean task workload = {MEAN_WORKLOAD_GC:.2f} GC, "
          f"mean execution frequency = {args.mean_freq_ghz:.3f} GHz")
    print()

    lam_edge = float(profiles["edge_efficient"]["failure_rate"]) * args.lambda_scale
    lam_cloud = float(profiles["cloud_fast"]["failure_rate"]) * args.lambda_scale
    d_default = float(profiles["edge_efficient"].get("failure_rate_exponent", 1.0))
    d = args.d if args.d is not None else d_default
    rho_min = float(profiles["edge_efficient"]["min_rho"])
    print(f"lambda_edge = {lam_edge:.3e} /s, lambda_cloud = {lam_cloud:.3e} /s, "
          f"d = {d}, rho_min = {rho_min}")
    print()

    header = (f"{'template':<18}{'n':>5}{'sum_t(s)':>10}"
              f"{'risk@1':>10}{'used':>8}{'R@1':>9}{'rho_min_feas':>14}")
    for scale, limit in DATASET_SCALE_LIMITS.items():
        print(f"--- dataset_scale = {scale} ({limit} templates) ---")
        print(header)
        infeasible = 0
        rho_floors = []
        for name in PEGASUS_XML_FILES[:limit]:
            n = count_tasks(PROJECT_ROOT / "data" / name)
            sum_t = n * MEAN_WORKLOAD_GC / args.mean_freq_ghz
            risk = lam_edge * sum_t
            used = risk / budget
            r_at_one = math.exp(-risk)
            if used >= 1.0:
                infeasible += 1
                floor_text = "INFEASIBLE"
            else:
                floor = solve_min_rho(budget / risk, d, rho_min)
                rho_floors.append(floor)
                floor_text = f"{floor:.3f}"
            print(f"{name[:-4]:<18}{n:>5}{sum_t:>10.1f}{risk:>10.4f}"
                  f"{used:>7.1%}{r_at_one:>9.4f}{floor_text:>14}")
        if rho_floors:
            print(f"  feasible templates: {len(rho_floors)}/{limit}, "
                  f"rho floor range [{min(rho_floors):.3f}, {max(rho_floors):.3f}]")
        print(f"  reliability-infeasible at rho=1 (all-edge placement): "
              f"{infeasible}/{limit} templates")
        print()

    print("=" * 78)
    print("3) OVERLAP -- can the DVFS action reach the energy optimum?")
    print("=" * 78)
    rho_ee_all = {
        name: rho_energy_efficient(
            float(spec["static_power"]), float(spec["dynamic_power"])
        )
        for name, spec in profiles.items()
    }
    best_energy_rho = min(rho_ee_all.values())
    n_mid = sum(
        count_tasks(PROJECT_ROOT / "data" / f) for f in PEGASUS_XML_FILES[:10]
    ) / 10.0
    sum_t_mid = n_mid * MEAN_WORKLOAD_GC / args.mean_freq_ghz
    risk_mid = lam_edge * sum_t_mid
    if risk_mid >= budget:
        print("mean 'middle' workflow is already reliability-INFEASIBLE at rho = 1")
        print("=> the admissible DVFS window is EMPTY; rho is pinned at 1 by the "
              "reliability constraint alone.")
        return
    floor_mid = solve_min_rho(budget / risk_mid, d, rho_min)
    print(f"mean 'middle' workflow: n = {n_mid:.1f} tasks, "
          f"sum_t = {sum_t_mid:.1f} s, R(rho=1) = {math.exp(-risk_mid):.5f}")
    print(f"reliability floor    rho >= {floor_mid:.3f}")
    print(f"energy optimum       rho*  = {best_energy_rho:.3f} "
          f"(best profile: {min(rho_ee_all, key=rho_ee_all.get)})")
    if floor_mid >= 1.0 - 1e-6:
        print("=> window is EMPTY.")
    elif floor_mid > best_energy_rho:
        print(f"=> window = [{floor_mid:.3f}, 1.0], width {1.0 - floor_mid:.3f}; "
              "reliability binds BEFORE the energy optimum.")
    else:
        print(f"=> window = [{floor_mid:.3f}, 1.0] contains rho* = "
              f"{best_energy_rho:.3f}; energy optimum is reachable.")


if __name__ == "__main__":
    main()
