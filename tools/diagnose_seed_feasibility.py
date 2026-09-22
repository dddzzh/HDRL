"""Are the seeds that break ESP actually schedulable, by anyone?

ESP@0.92 fails on a handful of episodes, not on the average.  Before trying to
make the policy better on those episodes it is worth asking whether they are
winnable at all: run a scheduler that ignores energy entirely -- online HEFT
at rho=1, contention-aware earliest-finish-time -- on the same seeds and see
what *it* misses.

HEFT is not optimal, but it is the same rule the Deadline was calibrated with
(``D_w = multiplier * reference_heft_makespan``), differing only in that this
run faces real contention while the reference was computed contention-free.
So its per-seed miss count is a fair practical ceiling:

* HEFT misses too  ->  the episode is over capacity.  No reweighting of an
  energy-aware policy reaches ESP = 1.00 there, and the honest move is to fix
  the acceptance statistic rather than the agent.
* HEFT clears it   ->  the schedule exists, the learned policy is not finding
  it, and the gap is worth attacking.

Also reports each seed's offered load (total cycles / arrival span), which is
what separates the hard episodes from the easy ones.
"""
from __future__ import annotations

import argparse
import json
import sys
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Only used to build the state encoder; with "
                             "--high-algorithm heft the learned upper policy "
                             "never selects anything.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-start", type=int, default=642000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--high-algorithm", default="heft")
    parser.add_argument("--target", type=float, default=0.92)
    parser.add_argument("--output")
    return parser.parse_args()


def offered_load(workflows):
    """Total cycles divided by the arrival span: the episode's demand rate."""
    arrivals = np.array([float(w.arrival_time) for w in workflows])
    cycles = sum(
        float(w.get_task(task_id).workload_cycles)
        for w in workflows
        for task_id in w.task_ids()
    )
    span = float(arrivals.max() - arrivals.min())
    x2 = sum(
        1 for w in workflows
        if abs(float(w.deadline_multiplier) - 2.0) < 1e-9
    )
    return {
        "workflows": len(workflows),
        "total_cycles": cycles,
        "arrival_span": span,
        "offered_load_gcps": cycles / 1e9 / max(span, 1e-9),
        "x2_workflows": x2,
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

    agent_config = configure_agent_state(config, resources, smoke_test=False)
    agent_config["use_low_agent"] = False
    agent = agent_class(agent_config, seed=args.seed_start)
    agent.load_checkpoint(Path(args.checkpoint).resolve())
    agent.high_agent.epsilon = 0.0

    print(f"[{args.high_algorithm}, rho=1] seeds {args.seed_start}-"
          f"{args.seed_start + args.num_seeds - 1}")
    print(f"{'seed':>8}{'load':>8}{'x2':>5}{'DSR':>8}{'miss':>6}"
          f"{'pass@' + f'{args.target:g}':>10}{'energy_kJ':>11}{'rel':>7}")
    rows = []
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        with redirect_stdout(StringIO()):
            workflows = build_workflows(config, resources, seed, smoke_test=False)
            env = build_env(
                workflows, resources, dict(config.get("simulation", {})), seed
            )
            result = collect_d3qn_episode(
                env,
                agent,
                objective,
                HighRolloutBuffer(),
                LowRolloutBuffer(),
                training=False,
                high_deterministic=True,
                low_deterministic=True,
                fixed_low_rho_one=True,
                high_algorithm=args.high_algorithm,
            )
        load = offered_load(workflows)
        metrics = result["metrics"]
        dsr = float(metrics.get("deadline_success_rate", 0.0))
        misses = int(round((1.0 - dsr) * load["workflows"]))
        passed = dsr >= args.target - 1e-12
        energy = float(metrics.get("total_energy", 0.0))
        rel = float(metrics.get("reliability_success_rate", 0.0))
        print(f"{seed:>8}{load['offered_load_gcps']:>8.1f}"
              f"{load['x2_workflows']:>5}{dsr:>8.4f}{misses:>6}"
              f"{'yes' if passed else 'NO':>10}{energy / 1000.0:>11.1f}{rel:>7.3f}")
        rows.append(
            {
                "eval_seed": seed,
                **load,
                "deadline_success_rate": dsr,
                "misses": misses,
                "passes_target": bool(passed),
                "total_energy": energy,
                "reliability_success_rate": rel,
            }
        )

    dsr = np.array([row["deadline_success_rate"] for row in rows])
    esp = float(np.mean([row["passes_target"] for row in rows]))
    energy = float(np.mean([row["total_energy"] for row in rows]))
    print(f"\n  mean DSR {dsr.mean():.4f}   min {dsr.min():.4f}   "
          f"ESP@{args.target:g} {esp:.2f}   mean E {energy / 1000.0:.1f} kJ")
    failed = [row for row in rows if not row["passes_target"]]
    if failed:
        print(f"  seeds this energy-blind scheduler still cannot clear: "
              f"{[row['eval_seed'] for row in failed]}")
        print("  -> those episodes are over capacity, not mis-scheduled.")
    else:
        print("  a feasible schedule exists on every seed; the gap is the "
              "policy's, not the pool's.")

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "high_algorithm": args.high_algorithm,
                    "target": args.target,
                    "mean_deadline_success_rate": float(dsr.mean()),
                    "min_deadline_success_rate": float(dsr.min()),
                    "esp_at_target": esp,
                    "mean_total_energy": energy,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
