"""What does the Deadline yardstick actually assume, and what does it cost?

``D_w = multiplier * reference_heft_makespan`` and the reference is
latency-greedy HEFT, contention-free, on the whole pool.  HEFT therefore
picks servers to minimise EFT, with no energy term anywhere in its objective
-- so the yardstick is implicitly denominated in *fast* servers, while the
policy under test is minimising energy and lives on cheap ones.

The gap between those two placements is a floor on inflation that no amount
of Lagrangian pressure can remove, because it is baked into the denominator.
This measures it directly, with no learning involved: run HEFT on the same
workflows, take its server mix, and compare effective clock rates.

Effective clock is the *harmonic* mean weighted by cycles, since execution
time adds as ``work / f``; the arithmetic mean would understate the slow
servers.  The ratio ``f_ref / f_policy`` is the pure placement slowdown.
Multiplying it by the measured contention factor should reproduce the
observed inflation -- if it does, the inflation is fully explained by these
two terms and nothing exotic is going on.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.schedulers.heft import run_heft
from tools.hierarchical_common import build_resources, build_workflows


EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-start", type=int, default=642000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument(
        "--policy-mix",
        help="JSON from diagnose_deadline_energy_tradeoff.py; its placement "
        "counts are compared against HEFT's on the same seeds.",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def effective_clock(counts, freq_by_type):
    """Cycle-weighted harmonic mean clock of a placement mix.

    Execution time is ``work / f``, so mixing placements averages 1/f, not f.
    Treating every task as carrying the same expected work makes this the
    ratio the makespan actually scales by.
    """
    total = sum(counts.values())
    if not total:
        return float("nan")
    inverse = sum(
        count / max(freq_by_type[name], EPS) for name, count in counts.items()
    )
    return total / max(inverse, EPS)


def main():
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    with redirect_stdout(StringIO()):
        resources = build_resources(
            config, int(config.get("training", {}).get("seed", 42))
        )
    servers = resources["servers"] if isinstance(resources, dict) else resources
    link_bandwidth = (
        resources.get("link_bandwidth_mbps") if isinstance(resources, dict) else None
    )
    type_by_id = {
        server["id"]: str(server.get("server_type", server["id"]))
        for server in servers
    }
    freq_by_type = {}
    power_by_type = {}
    for server in servers:
        name = str(server.get("server_type", server["id"]))
        freq_by_type[name] = float(server["max_freq_ghz"])
        power_by_type[name] = float(
            server.get(
                "active_power_reference",
                float(server.get("static_power", 0.0))
                + float(server.get("dynamic_power", 0.0)),
            )
        )

    heft_mix = Counter()
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        with redirect_stdout(StringIO()):
            workflows = build_workflows(config, resources, seed, smoke_test=False)
        cache = {}
        for workflow in workflows:
            plan = run_heft(workflow, servers, link_bandwidth, rho=1.0, cache=cache)
            for server_id in plan["task_server"].values():
                heft_mix[type_by_id[int(server_id)]] += 1

    seeds = f"{args.seed_start}-{args.seed_start + args.num_seeds - 1}"
    print("=" * 70)
    print(f"DEADLINE REFERENCE PLACEMENT (latency-greedy HEFT)  seeds {seeds}")
    print("=" * 70)
    heft_total = max(sum(heft_mix.values()), 1)
    for name, count in heft_mix.most_common():
        print(f"  {name:<18}{count:>8}  {count / heft_total:>7.1%}"
              f"   f={freq_by_type[name]:.2f} GHz"
              f"   P/f={power_by_type[name] / freq_by_type[name]:.1f}")
    f_ref = effective_clock(heft_mix, freq_by_type)
    e_ref = sum(
        count * power_by_type[name] / max(freq_by_type[name], EPS)
        for name, count in heft_mix.items()
    ) / heft_total
    print(f"\n  effective clock (harmonic)  {f_ref:.3f} GHz")
    print(f"  mean energy per cycle       {e_ref:.2f}")

    payload = {
        "config": str(Path(args.config)),
        "seeds": seeds,
        "heft_mix": {str(k): int(v) for k, v in heft_mix.most_common()},
        "heft_effective_clock_ghz": f_ref,
        "heft_mean_energy_per_cycle": e_ref,
    }

    if args.policy_mix:
        blob = json.loads(Path(args.policy_mix).read_text(encoding="utf-8"))
        policy_mix = Counter(
            {name: int(count) for name, count in blob["placement"].items()}
        )
        policy_total = max(sum(policy_mix.values()), 1)
        print()
        print("=" * 70)
        print(f"POLICY PLACEMENT  [{blob.get('mode', '?')}]")
        print("=" * 70)
        for name, count in policy_mix.most_common():
            print(f"  {name:<18}{count:>8}  {count / policy_total:>7.1%}"
                  f"   f={freq_by_type[name]:.2f} GHz"
                  f"   P/f={power_by_type[name] / freq_by_type[name]:.1f}")
        f_pol = effective_clock(policy_mix, freq_by_type)
        e_pol = sum(
            count * power_by_type[name] / max(freq_by_type[name], EPS)
            for name, count in policy_mix.items()
        ) / policy_total
        placement_slowdown = f_ref / max(f_pol, EPS)
        energy_saving = 1.0 - e_pol / max(e_ref, EPS)
        print(f"\n  effective clock (harmonic)  {f_pol:.3f} GHz")
        print(f"  mean energy per cycle       {e_pol:.2f}")

        print()
        print("=" * 70)
        print("THE TRADE THE POLICY MADE")
        print("=" * 70)
        print(f"  placement slowdown vs reference   x{placement_slowdown:.3f}")
        print(f"  energy per cycle vs reference     {energy_saving:+.1%}")
        buckets = blob.get("multiplier_buckets", {})
        queueing = blob.get("queueing", {})
        print()
        print("  observed inflation = placement slowdown x contention factor:")
        for label, bucket in sorted(buckets.items()):
            observed = bucket["inflation"]["p50"]
            implied = observed / max(placement_slowdown, EPS)
            print(f"    mult {label:>4}:  observed p50 {observed:.3f}"
                  f"  =  {placement_slowdown:.3f} (placement)"
                  f"  x  {implied:.3f} (residual/queueing)")
        if queueing:
            print(f"\n  measured waiters {queueing['queued_tasks']}/"
                  f"{queueing['tasks']} tasks; contention "
                  f"{queueing['contention_seconds']:.0f} s")
        payload.update(
            {
                "policy_mix": {str(k): int(v) for k, v in policy_mix.most_common()},
                "policy_effective_clock_ghz": f_pol,
                "policy_mean_energy_per_cycle": e_pol,
                "placement_slowdown": placement_slowdown,
                "energy_per_cycle_saving": energy_saving,
            }
        )

    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
