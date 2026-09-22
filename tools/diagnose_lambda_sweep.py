"""Is the Deadline penalty too small, or is the schedule simply unbuyable?

The upper level picks its action by ranking

    combined = Q_E(s,a) - lambda_D * Q_D(s,a) - lambda_R * Q_R(s,a)

(``reliability_aware_high.py:116-118``), and ``lambda_D`` is read straight off
the PI controller at selection time (``high_agent.deadline_lambda`` is a
read-only view of ``deadline_lagrangian.lambda_deadline``).  So lambda can be
overridden *after* loading a checkpoint and the whole penalty axis swept
without retraining a thing.

That makes this a decisive test rather than a suggestive one:

* If DSR climbs to the 0.92 target as lambda grows, the trained policy *does*
  contain a Deadline-feasible schedule and the deployed lambda was simply too
  small -- a tuning problem.
* If DSR saturates below the target no matter how large lambda gets, then no
  reweighting of the same Q-heads can reach feasibility.  The ceiling is the
  resource pool or the state, not the penalty.

Sweeping also prices feasibility directly: the energy at the smallest lambda
that clears the target is what the Deadline constraint actually costs.

Second output, from the same replay: the *scale* of the two heads.  lambda
only matters through ``lambda_D * spread(Q_D)`` versus ``spread(Q_E)``, where
the spread is taken across the legal actions at one decision -- constants
shift every action equally and cannot change an argmax.  The ratio says
whether lambda=5 is a light touch or already overwhelming, which is the same
question a new beta_sd term would raise.
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


DEFAULT_LAMBDAS = "0,1,2.5,5,10,25,100,1000"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-start", type=int, default=642000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument(
        "--fixed-rho-one",
        action="store_true",
        help="Sweep the upper level alone, so DVFS cannot confound placement.",
    )
    parser.add_argument("--target", type=float, default=0.92)
    parser.add_argument("--output")
    return parser.parse_args()


class HeadScaleProbe:
    """Record the per-decision spread of each Q-head by wrapping selection.

    Only the spread across the legal actions of a single state matters: adding
    a constant to every action's Q leaves the argmax alone.  Averaging the
    spreads over all decisions gives the scale at which lambda operates.
    """

    def __init__(self, agent_r):
        self.agent_r = agent_r
        self.original = agent_r.combined_action_q_values
        self.energy_spread = []
        self.deadline_spread = []
        self.reliability_spread = []

    def __enter__(self):
        def wrapped(observation, deadline_lambda, reliability_lambda):
            result = self.original(observation, deadline_lambda, reliability_lambda)
            _, energy, deadline, reliability = result
            for values, sink in (
                (energy, self.energy_spread),
                (deadline, self.deadline_spread),
                (reliability, self.reliability_spread),
            ):
                array = np.asarray(values, dtype=np.float64).ravel()
                if array.size:
                    sink.append(float(array.max() - array.min()))
            return result

        self.agent_r.combined_action_q_values = wrapped
        return self

    def __exit__(self, *exc):
        self.agent_r.combined_action_q_values = self.original
        return False

    def summary(self):
        def mean(values):
            return float(np.mean(values)) if values else float("nan")

        return {
            "energy_q_spread": mean(self.energy_spread),
            "deadline_q_spread": mean(self.deadline_spread),
            "reliability_q_spread": mean(self.reliability_spread),
            "decisions": len(self.energy_spread),
        }


def set_deadline_lambda(agent, value):
    """Override lambda_D on the PI controller the selector reads from.

    ``high_agent.deadline_lambda`` is a read-only property, and the sweep
    deliberately goes past ``lambda_max`` -- the point is to find where the
    DSR curve flattens, which may be above the ceiling training was allowed
    to reach.  ``lambda_max`` is only applied inside the dual update, which
    never runs here, but it is raised anyway so nothing clips silently.
    """
    controller = agent.high_agent.deadline_lagrangian
    if controller is None:
        raise RuntimeError("checkpoint has no Deadline Lagrangian controller")
    controller.lambda_max = max(float(controller.lambda_max), float(value))
    controller.lambda_deadline = float(value)
    applied = float(agent.high_agent.deadline_lambda)
    if abs(applied - float(value)) > 1e-9:
        raise RuntimeError(
            f"lambda override did not take: asked {value}, got {applied}"
        )


def evaluate(agent, config, resources, objective, seed_start, num_seeds, use_low):
    rows = []
    for offset in range(num_seeds):
        seed = seed_start + offset
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
                fixed_low_rho_one=not use_low,
                high_algorithm="d3qn",
            )
        metrics = result["metrics"]
        rows.append(
            {
                "eval_seed": seed,
                "deadline_success_rate": float(metrics.get("deadline_success_rate", 0.0)),
                "reliability_success_rate": float(
                    metrics.get("reliability_success_rate", 0.0)
                ),
                "total_energy": float(metrics.get("total_energy", 0.0)),
            }
        )
    return rows


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

    agent_config = configure_agent_state(config, resources, smoke_test=False)
    agent_config["use_low_agent"] = bool(use_low)
    agent = agent_class(agent_config, seed=args.seed_start)
    agent.load_checkpoint(Path(args.checkpoint).resolve())
    agent.high_agent.epsilon = 0.0
    trained_lambda = float(agent.high_agent.deadline_lambda)

    lambdas = [float(value) for value in args.lambdas.split(",") if value.strip()]
    mode = "rho1" if args.fixed_rho_one else "sac"
    seeds = f"{args.seed_start}-{args.seed_start + args.num_seeds - 1}"
    print(f"[{mode}] seeds {seeds}   trained lambda_D = {trained_lambda:.4f}")
    print(f"{'lambda':>9}{'DSR':>9}{'min':>8}{'ESP@' + f'{args.target:g}':>9}"
          f"{'rel':>7}{'energy_kJ':>12}{'dE%':>8}")

    baseline_energy = None
    results = []
    head_scale = None
    for value in lambdas:
        set_deadline_lambda(agent, value)
        if head_scale is None:
            with HeadScaleProbe(agent.high_agent.agent_R) as probe:
                rows = evaluate(
                    agent, config, resources, objective,
                    args.seed_start, args.num_seeds, use_low,
                )
            head_scale = probe.summary()
        else:
            rows = evaluate(
                agent, config, resources, objective,
                args.seed_start, args.num_seeds, use_low,
            )
        dsr = np.array([row["deadline_success_rate"] for row in rows])
        energy = float(np.mean([row["total_energy"] for row in rows]))
        rel = float(np.mean([row["reliability_success_rate"] for row in rows]))
        esp = float(np.mean(dsr >= args.target - 1e-12))
        if baseline_energy is None:
            baseline_energy = energy
        delta = (energy - baseline_energy) / max(baseline_energy, 1e-9)
        print(f"{value:>9.4g}{dsr.mean():>9.4f}{dsr.min():>8.3f}{esp:>9.2f}"
              f"{rel:>7.3f}{energy / 1000.0:>12.2f}{delta:>7.1%}")
        results.append(
            {
                "lambda_deadline": value,
                "mean_deadline_success_rate": float(dsr.mean()),
                "min_deadline_success_rate": float(dsr.min()),
                "esp_at_target": esp,
                "mean_reliability_success_rate": rel,
                "mean_total_energy": energy,
                "energy_delta_vs_lambda0": delta,
                "rows": rows,
            }
        )
    set_deadline_lambda(agent, trained_lambda)

    print()
    print("Q-head scale (mean spread across the legal actions of one decision;")
    print("only the spread can move an argmax, a constant offset cannot):")
    print(f"  spread(Q_E)               {head_scale['energy_q_spread']:.6g}")
    print(f"  spread(Q_D)               {head_scale['deadline_q_spread']:.6g}")
    print(f"  spread(Q_R)               {head_scale['reliability_q_spread']:.6g}")
    ratio = (
        trained_lambda * head_scale["deadline_q_spread"]
        / max(head_scale["energy_q_spread"], 1e-12)
    )
    print(f"  lambda_D*spread(Q_D) / spread(Q_E) at the trained lambda = "
          f"{ratio:.3f}")
    print(f"  decisions probed          {head_scale['decisions']}")

    feasible = [row for row in results if row["esp_at_target"] >= 1.0]
    print()
    if feasible:
        best = min(feasible, key=lambda row: row["mean_total_energy"])
        print(f"cheapest lambda reaching ESP@{args.target:g} = 1.00: "
              f"lambda={best['lambda_deadline']:g}, "
              f"E={best['mean_total_energy'] / 1000.0:.2f} kJ, "
              f"DSR={best['mean_deadline_success_rate']:.4f}")
    else:
        best = max(results, key=lambda row: row["mean_deadline_success_rate"])
        print(f"no lambda reached ESP@{args.target:g} = 1.00 over {seeds}.")
        print(f"best DSR was {best['mean_deadline_success_rate']:.4f} at "
              f"lambda={best['lambda_deadline']:g} "
              f"(E={best['mean_total_energy'] / 1000.0:.2f} kJ).")
        print("A saturating DSR curve means the penalty is not the binding "
              "constraint.")

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "mode": mode,
                    "seeds": seeds,
                    "target": args.target,
                    "trained_lambda_deadline": trained_lambda,
                    "head_scale": head_scale,
                    "sweep": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
