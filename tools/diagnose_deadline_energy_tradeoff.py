"""Why are Deadlines missed: resource asymmetry, queueing, or a weak penalty?

Three hypotheses, one replay each, all read-only.  Nothing here trains, and
no checkpoint or config is modified.

H1  RESOURCE ASYMMETRY.  ``D_w = multiplier * reference_heft_makespan`` where
    the reference is the *contention-free, latency-greedy* HEFT makespan.  If
    the cheapest server is much slower than the one HEFT used, an energy-
    minimizing policy inflates the makespan by construction and the 2.0x
    workflows have no room left.  Measured by bucketing every workflow by its
    multiplier and comparing the realized inflation against the allowance.

H2  UNLEARNED QUEUEING.  Contention only counts against the policy if a
    better server was actually free.  For every task that waited, this counts
    the servers idle at its ``base_ready_time`` under three tightening tests:
    any idle server at all; one at least as fast; and one that is *both* at
    least as fast *and* no more expensive per cycle.  Only that last case is
    unambiguous -- it is a strict Pareto improvement the policy declined, so
    it costs no energy to take and can only be a learning failure.  Waiting
    behind a fully busy pool, by contrast, is a capacity fact.

H3  WEAK VIOLATION PENALTY.  Locates the realized policy on the
    energy/latency frontier: the mean energy-per-cycle of the servers it
    picked, against the pool's cheapest and fastest.  A policy sitting on the
    cheap end while missing Deadlines is under-penalized; one already near
    the fast end is capacity-bound and a larger penalty cannot help.

The per-server frontier table (H1's static half) needs no replay and is
printed from the resource set alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.D3QN.reward import HierarchicalObjective
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from tools.hierarchical_common import (
    build_env,
    build_resources,
    build_workflows,
    configure_agent_state,
)
from tools.train_hierarchical_d3qn import collect_d3qn_episode
from tools.train_coupled_hierarchical_d3qn_sac import select_agent_class


EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-start", type=int, default=642000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument(
        "--fixed-rho-one",
        action="store_true",
        help="Isolate placement from DVFS: run the upper level alone.",
    )
    parser.add_argument(
        "--lambda-deadline",
        type=float,
        help="Override lambda_D at selection time (see diagnose_lambda_sweep). "
        "Used to check *why* a larger penalty backfires: if the placement mix "
        "shifts onto the fast tier and contention rises with it, the failure "
        "at high lambda is herding, not slowness.",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def _f(mapping, key, default=0.0):
    value = mapping.get(key)
    return default if value is None else float(value)


# --------------------------------------------------------------------------
# H1 static half: the speed/energy frontier of the pool itself.
# --------------------------------------------------------------------------
def frontier_table(servers):
    """Energy per cycle at rho=1 for each distinct server profile.

    ``active_power_reference`` is the rho=1 power and ``max_freq_ghz`` the
    rho=1 clock, so ``P / f`` is joules per cycle -- the quantity a placement
    decision actually trades against latency.
    """
    by_type = {}
    for server in servers:
        name = str(server.get("server_type", server.get("type", "?")))
        freq = float(server["max_freq_ghz"])
        power = float(
            server.get(
                "active_power_reference",
                float(server.get("static_power", 0.0))
                + float(server.get("dynamic_power", 0.0)),
            )
        )
        entry = by_type.setdefault(
            name, {"count": 0, "freq": freq, "power": power}
        )
        entry["count"] += 1
    rows = []
    for name, entry in by_type.items():
        rows.append(
            {
                "server_type": name,
                "count": entry["count"],
                "max_freq_ghz": entry["freq"],
                "power_w": entry["power"],
                "energy_per_cycle": entry["power"] / max(entry["freq"], EPS),
            }
        )
    rows.sort(key=lambda row: row["energy_per_cycle"])
    cheapest = rows[0]
    fastest = max(rows, key=lambda row: row["max_freq_ghz"])
    for row in rows:
        row["energy_rel_to_cheapest"] = (
            row["energy_per_cycle"] / max(cheapest["energy_per_cycle"], EPS)
        )
        row["speed_rel_to_cheapest"] = (
            row["max_freq_ghz"] / max(cheapest["max_freq_ghz"], EPS)
        )
    speed_span = fastest["max_freq_ghz"] / max(cheapest["max_freq_ghz"], EPS)
    energy_span = fastest["energy_per_cycle"] / max(
        cheapest["energy_per_cycle"], EPS
    )
    # d(log energy)/d(log speed): 2.0 would be the classic P ~ C f^3 result at
    # constant C_ef.  Anything above that means C_ef itself grows with the
    # tier, which steepens the price of speed.
    elasticity = float(np.log(energy_span) / max(np.log(speed_span), EPS))
    return {
        "rows": rows,
        "speed_span": float(speed_span),
        "energy_span": float(energy_span),
        "energy_speed_elasticity": elasticity,
        "cheapest": cheapest["server_type"],
        "fastest": fastest["server_type"],
    }


# --------------------------------------------------------------------------
# H2/H3: per-episode replay analysis.
# --------------------------------------------------------------------------
def server_busy_intervals(event_log):
    """Compute occupancy per server from the primary replicas."""
    intervals = defaultdict(list)
    for event in event_log:
        if str(event.get("replica_role", "primary")) != "primary":
            continue
        server = event.get("server_id")
        start = _f(event, "start_time")
        intervals[server].append((start, start + _f(event, "exec_time")))
    for server in intervals:
        intervals[server].sort()
    return intervals


def _busy_at(intervals, server, moment):
    for start, end in intervals.get(server, ()):  # pools are small; linear is fine
        if start - EPS <= moment < end - EPS:
            return True
    return False


def episode_analysis(env, workflow_rows, speed_by_server, energy_by_server):
    intervals = server_busy_intervals(env.event_log)
    all_servers = list(speed_by_server)

    placement = Counter()
    energy_per_cycle_used = []
    queued_tasks = 0
    tasks = 0
    idle_available = 0          # some server was free
    idle_faster_available = 0   # a free server at least as fast as the pick
    idle_pareto_available = 0   # ... and no more expensive per cycle
    contention_seconds = 0.0
    pareto_wasted_seconds = 0.0  # waiting a Pareto move would have removed

    for event in env.event_log:
        if str(event.get("replica_role", "primary")) != "primary":
            continue
        tasks += 1
        server = event.get("server_id")
        placement[str(event.get("server_type", server))] += 1
        energy_per_cycle_used.append(energy_by_server.get(server, np.nan))

        ast = _f(event, "start_time")
        base_ready = _f(event, "base_ready_time")
        wait = ast - base_ready
        if wait <= 1e-9:
            continue
        queued_tasks += 1
        contention_seconds += wait
        chosen_speed = speed_by_server.get(server, 0.0)
        chosen_cost = energy_by_server.get(server, float("inf"))
        free = [
            other
            for other in all_servers
            if other != server and not _busy_at(intervals, other, base_ready)
        ]
        if free:
            idle_available += 1
            faster = [
                other for other in free
                if speed_by_server[other] >= chosen_speed - EPS
            ]
            if faster:
                idle_faster_available += 1
                if any(
                    energy_by_server[other] <= chosen_cost + EPS
                    for other in faster
                ):
                    idle_pareto_available += 1
                    pareto_wasted_seconds += wait

    workflows = []
    for row in workflow_rows:
        response = row.get("response_time")
        reference = row.get("reference_makespan")
        if response is None or reference is None or float(reference) <= 0.0:
            continue
        multiplier = row.get("deadline_multiplier")
        workflows.append(
            {
                "multiplier": None if multiplier is None else float(multiplier),
                "inflation": float(response) / float(reference),
                "usage": float(response) / max(float(row["relative_deadline"]), EPS),
                "met": bool(row.get("deadline_satisfied")),
                "energy": float(row.get("total_energy", 0.0)),
            }
        )

    return {
        "workflows": workflows,
        "placement": placement,
        "mean_energy_per_cycle_used": float(np.nanmean(energy_per_cycle_used))
        if energy_per_cycle_used
        else float("nan"),
        "tasks": tasks,
        "queued_tasks": queued_tasks,
        "idle_available": idle_available,
        "idle_faster_available": idle_faster_available,
        "idle_pareto_available": idle_pareto_available,
        "contention_seconds": contention_seconds,
        "pareto_wasted_seconds": pareto_wasted_seconds,
    }


def _quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def main():
    args = parse_args()
    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    agent_class = select_agent_class(config)
    with redirect_stdout(StringIO()):
        resources = build_resources(
            config, int(config.get("training", {}).get("seed", 42))
        )
    objective = HierarchicalObjective(config.get("objective", {}))
    use_low = not args.fixed_rho_one

    servers = resources["servers"] if isinstance(resources, dict) else resources
    frontier = frontier_table(servers)
    speed_by_server = {
        server["id"]: float(server["max_freq_ghz"]) for server in servers
    }
    energy_by_server = {}
    for server in servers:
        power = float(
            server.get(
                "active_power_reference",
                float(server.get("static_power", 0.0))
                + float(server.get("dynamic_power", 0.0)),
            )
        )
        energy_by_server[server["id"]] = power / max(
            float(server["max_freq_ghz"]), EPS
        )

    print("=" * 74)
    print("H1a  POOL FRONTIER (static, no replay)")
    print("=" * 74)
    print(f"{'profile':<16}{'n':>4}{'f_max':>8}{'P(W)':>10}{'P/f':>10}"
          f"{'xE':>8}{'xSpeed':>9}")
    for row in frontier["rows"]:
        print(f"{row['server_type']:<16}{row['count']:>4}"
              f"{row['max_freq_ghz']:>8.2f}{row['power_w']:>10.1f}"
              f"{row['energy_per_cycle']:>10.1f}"
              f"{row['energy_rel_to_cheapest']:>8.2f}"
              f"{row['speed_rel_to_cheapest']:>9.2f}")
    print(f"\n  fastest/cheapest: speed x{frontier['speed_span']:.2f}, "
          f"energy x{frontier['energy_span']:.2f}")
    print(f"  energy-speed elasticity d(logE)/d(logF) = "
          f"{frontier['energy_speed_elasticity']:.2f}   "
          f"(2.00 = constant C_ef)")

    agent_config = configure_agent_state(config, resources, smoke_test=False)
    agent_config["use_low_agent"] = bool(use_low)
    agent = agent_class(agent_config, seed=args.seed_start)
    agent.load_checkpoint(Path(args.checkpoint).resolve())
    agent.high_agent.epsilon = 0.0
    if args.lambda_deadline is not None:
        from tools.diagnose_lambda_sweep import set_deadline_lambda

        set_deadline_lambda(agent, args.lambda_deadline)
    lambda_used = float(agent.high_agent.deadline_lambda)

    all_workflows = []
    placement = Counter()
    totals = Counter()
    energy_per_cycle_used = []
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        with redirect_stdout(StringIO()):
            workflows = build_workflows(config, resources, seed, smoke_test=False)
            env = build_env(
                workflows, resources, dict(config.get("simulation", {})), seed
            )
            collect_d3qn_episode(
                env,
                agent,
                objective,
                HighRolloutBuffer(),
                LowRolloutBuffer(),
                training=False,
                high_deterministic=True,
                low_deterministic=True,
                fixed_low_rho_one=not use_low,
                high_algorithm="d3qn",
            )
            rows = env._build_workflow_results()
        analysis = episode_analysis(
            env, rows, speed_by_server, energy_by_server
        )
        all_workflows.extend(analysis["workflows"])
        placement.update(analysis["placement"])
        for key in ("tasks", "queued_tasks", "idle_available",
                    "idle_faster_available", "idle_pareto_available"):
            totals[key] += analysis[key]
        totals["contention_seconds"] += analysis["contention_seconds"]
        totals["pareto_wasted_seconds"] += analysis["pareto_wasted_seconds"]
        if np.isfinite(analysis["mean_energy_per_cycle_used"]):
            energy_per_cycle_used.append(analysis["mean_energy_per_cycle_used"])

    mode = ("rho1" if args.fixed_rho_one else "sac") + f" lam={lambda_used:g}"
    print()
    print("=" * 74)
    print(f"H1b  MISSES BY DEADLINE MULTIPLIER   [{mode}]  seeds "
          f"{args.seed_start}-{args.seed_start + args.num_seeds - 1}")
    print("=" * 74)
    buckets = defaultdict(list)
    for row in all_workflows:
        buckets[row["multiplier"]].append(row)
    print(f"{'mult':>6}{'n':>6}{'DSR':>9}{'infl_p50':>11}{'infl_p90':>11}"
          f"{'infl_max':>11}{'headroom':>11}")
    summary_buckets = {}
    for multiplier in sorted(buckets, key=lambda value: (value is None, value)):
        rows = buckets[multiplier]
        inflation = [row["inflation"] for row in rows]
        met = float(np.mean([row["met"] for row in rows]))
        stats = _quantiles(inflation)
        headroom = (
            (float(multiplier) - stats["p90"]) / float(multiplier)
            if multiplier
            else float("nan")
        )
        label = "None" if multiplier is None else f"{multiplier:.1f}"
        print(f"{label:>6}{len(rows):>6}{met:>9.4f}{stats['p50']:>11.3f}"
              f"{stats['p90']:>11.3f}{stats['max']:>11.3f}{headroom:>11.1%}")
        summary_buckets[label] = {
            "n": len(rows), "deadline_success_rate": met,
            "inflation": stats, "p90_headroom": headroom,
        }

    print()
    print("=" * 74)
    print(f"H2  IS THE QUEUEING AVOIDABLE?   [{mode}]")
    print("=" * 74)
    queued = max(int(totals["queued_tasks"]), 1)
    print(f"  tasks                       {int(totals['tasks'])}")
    print(f"  tasks that waited           {int(totals['queued_tasks'])} "
          f"({totals['queued_tasks'] / max(int(totals['tasks']), 1):.1%})")
    print(f"  ... some server was idle    {int(totals['idle_available'])} "
          f"({totals['idle_available'] / queued:.1%} of waiters)")
    print(f"  ... an idle server was      {int(totals['idle_faster_available'])} "
          f"({totals['idle_faster_available'] / queued:.1%} of waiters)")
    print(f"      at least as fast")
    print(f"  ... AND no more expensive   {int(totals['idle_pareto_available'])} "
          f"({totals['idle_pareto_available'] / queued:.1%} of waiters)")
    print(f"      per cycle  <- free to take, pure loss")
    print(f"  total contention            {totals['contention_seconds']:.1f} s")
    print(f"  of which behind a free      {totals['pareto_wasted_seconds']:.1f} s "
          f"({totals['pareto_wasted_seconds'] / max(totals['contention_seconds'], EPS):.1%})")
    print(f"      Pareto-equal server")

    print()
    print("=" * 74)
    print(f"H3  WHERE ON THE FRONTIER DID THE POLICY LAND?   [{mode}]")
    print("=" * 74)
    cheapest_epc = frontier["rows"][0]["energy_per_cycle"]
    fastest_epc = max(
        row["energy_per_cycle"] for row in frontier["rows"]
    )
    realized = float(np.mean(energy_per_cycle_used)) if energy_per_cycle_used else float("nan")
    position = (realized - cheapest_epc) / max(fastest_epc - cheapest_epc, EPS)
    total_placed = max(sum(placement.values()), 1)
    for name, count in placement.most_common():
        print(f"  {name:<18}{count:>7}  {count / total_placed:>7.1%}")
    print(f"\n  mean energy/cycle chosen    {realized:.2f}")
    print(f"  pool cheapest / fastest     {cheapest_epc:.2f} / {fastest_epc:.2f}")
    print(f"  position on frontier        {position:.1%}   "
          f"(0% = all-cheapest, 100% = all-fastest)")

    if args.output:
        payload = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "mode": mode,
            "seeds": f"{args.seed_start}-{args.seed_start + args.num_seeds - 1}",
            "frontier": frontier,
            "multiplier_buckets": summary_buckets,
            "queueing": {
                "tasks": int(totals["tasks"]),
                "queued_tasks": int(totals["queued_tasks"]),
                "idle_available": int(totals["idle_available"]),
                "idle_faster_available": int(totals["idle_faster_available"]),
                "idle_pareto_available": int(totals["idle_pareto_available"]),
                "contention_seconds": float(totals["contention_seconds"]),
                "pareto_wasted_seconds": float(totals["pareto_wasted_seconds"]),
            },
            "placement": {str(k): int(v) for k, v in placement.most_common()},
            "frontier_position": float(position),
            "mean_energy_per_cycle_used": realized,
        }
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
