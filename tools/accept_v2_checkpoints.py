"""Two-block checkpoint selection and acceptance for the v2 hierarchical runs.

Selection and acceptance use **disjoint** seed blocks so the reported 20-seed
numbers are not the numbers the checkpoint was picked on:

* selection block  -- default 542000-542019, ranked by
  ``deadline >= 0.99`` then ``reliability == 1.0`` then lowest total energy
  (the ordering required by the handoff document, section 6.6);
* acceptance block -- default 242000-242019, evaluated once on the winner.

Each checkpoint is reported three ways so the lower level's contribution is
never inferred from a single number:

* ``rho1``      -- upper level alone at fixed ``rho = 1``;
* ``sac``       -- the full system with the checkpoint's own lower policy;
* paired system saving ``rho1 - sac`` per seed, with a 95% CI, plus the
  same-trajectory local counterfactual saving reported by the simulator.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
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
    metric_rank,
)
from tools.train_hierarchical_d3qn import collect_d3qn_episode
from tools.train_coupled_hierarchical_d3qn_sac import select_agent_class

REPORT = (
    "deadline_success_rate",
    "reliability_success_rate",
    "avg_workflow_reliability",
    "total_energy",
    "average_workflow_response_time",
    "avg_workflow_makespan",
    "max_workflow_makespan",
    "global_makespan",
    "episode_makespan",
    "rho_mean",
    "low_sac_total_energy_saving_ratio",
    "total_low_sac_energy_reference",
    "total_low_sac_energy_actual",
    "total_low_sac_energy_saved",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True, help="Training config of the run.")
    parser.add_argument("--select-seed-start", type=int, default=542000)
    parser.add_argument("--accept-seed-start", type=int, default=242000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--deadline-target", type=float, default=0.99)
    parser.add_argument("--reliability-target", type=float, default=1.0)
    parser.add_argument(
        "--select-stride",
        type=int,
        default=1,
        help="Evaluate every Nth saved checkpoint during selection.",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Skip the selection scan and accept this checkpoint directly. Use the "
            "trainer's own best_*_checkpoint.pt, which is already picked on the "
            "selection block with the required deadline>=target, reliability>=target, "
            "lowest-energy ordering."
        ),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def describe(values):
    array = np.asarray(list(values), dtype=np.float64)
    count = int(array.size)
    if count == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci95_half_width": 0.0}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    t_critical = 2.093 if count == 20 else 1.96
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95_half_width": float(t_critical * std / math.sqrt(count)) if count > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def evaluate(agent, config, resources, objective, seed_start, num_seeds, use_low):
    rows = []
    for offset in range(num_seeds):
        seed = seed_start + offset
        with redirect_stdout(StringIO()):
            workflows = build_workflows(config, resources, seed, smoke_test=False)
            env = build_env(workflows, resources, dict(config.get("simulation", {})), seed)
            result = collect_d3qn_episode(
                env, agent, objective, HighRolloutBuffer(), LowRolloutBuffer(),
                training=False, high_deterministic=True, low_deterministic=True,
                fixed_low_rho_one=not use_low, high_algorithm="d3qn",
            )
        metrics = result["metrics"]
        rows.append({"eval_seed": seed,
                     **{name: float(metrics.get(name, 0.0)) for name in REPORT}})
    return rows


def aggregate(rows):
    return {name: describe([row[name] for row in rows]) for name in REPORT}


def load_agent(checkpoint, config, resources, agent_class, use_low, seed):
    agent_config = configure_agent_state(config, resources, smoke_test=False)
    agent_config["use_low_agent"] = bool(use_low)
    agent = agent_class(agent_config, seed=seed)
    agent.load_checkpoint(checkpoint)
    agent.high_agent.epsilon = 0.0
    return agent


def main():
    args = parse_args()
    torch.set_num_threads(1)
    run_dir = Path(args.run_dir).resolve()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    agent_class = select_agent_class(config)
    with redirect_stdout(StringIO()):
        resources = build_resources(config, int(config.get("training", {}).get("seed", 42)))
    objective = HierarchicalObjective(config.get("objective", {}))

    selection = []
    if args.checkpoint is not None:
        best_path = Path(args.checkpoint).resolve()
        if not best_path.is_file():
            raise RuntimeError(f"{best_path} does not exist.")
        history = run_dir / "eval_history.json"
        picked_episode = None
        if history.is_file():
            rows = json.loads(history.read_text(encoding="utf-8"))
            feasible = [
                row for row in rows
                if float(row.get("deadline_success_rate", 0.0)) >= args.deadline_target
                and float(row.get("reliability_success_rate", 0.0)) >= args.reliability_target
            ]
            selection = [
                {
                    "checkpoint": f"eval_ep{int(row['episode']):06d}",
                    "rank_feasible": float(row in feasible),
                    **{name: float(row.get(name, 0.0)) for name in REPORT},
                }
                for row in rows
            ]
            if feasible:
                winner = min(feasible, key=lambda row: float(row["total_energy"]))
                picked_episode = int(winner["episode"])
        print(
            f"[select] using trainer-selected checkpoint {best_path.name} "
            f"(selection block {args.select_seed_start}-"
            f"{args.select_seed_start + args.num_seeds - 1}, "
            f"feasible evals={sum(1 for r in selection if r['rank_feasible'] > 0)}/"
            f"{len(selection)}, best feasible episode={picked_episode})",
            flush=True,
        )
        best_rank = (1.0 if picked_episode is not None else 0.0,)
        best_point = {}
        return _accept(args, run_dir, config, resources, objective, agent_class,
                       best_path, best_rank, selection)

    checkpoints = sorted(
        p for p in run_dir.glob("checkpoint_ep*.pt")
        if "_high" not in p.name and "_low" not in p.name
    )
    if not checkpoints:
        raise RuntimeError(f"No full checkpoints under {run_dir}.")
    checkpoints = checkpoints[:: max(int(args.select_stride), 1)]
    print(f"[select] {len(checkpoints)} checkpoints on seeds "
          f"{args.select_seed_start}-{args.select_seed_start + args.num_seeds - 1}", flush=True)

    best = None
    for path in checkpoints:
        with redirect_stdout(StringIO()):
            agent = load_agent(path, config, resources, agent_class, True, args.select_seed_start)
            rows = evaluate(agent, config, resources, objective,
                            args.select_seed_start, args.num_seeds, True)
        summary = aggregate(rows)
        point = {name: summary[name]["mean"] for name in REPORT}
        rank = metric_rank(point, deadline_target=args.deadline_target,
                           reliability_target=args.reliability_target)
        selection.append({"checkpoint": path.name, "rank_feasible": float(rank[0]),
                          **point})
        feasible = "OK " if rank[0] >= 1.0 else "-- "
        print(f"[select] {feasible}{path.name} deadline={point['deadline_success_rate']:.4f} "
              f"rel={point['reliability_success_rate']:.4f} "
              f"E={point['total_energy'] / 1000:.2f}kJ rho={point['rho_mean']:.4f}", flush=True)
        if best is None or rank > best[0]:
            best = (rank, path, point)

    best_rank, best_path, best_point = best
    print(f"[select] winner={best_path.name} feasible={best_rank[0] >= 1.0}", flush=True)
    return _accept(args, run_dir, config, resources, objective, agent_class,
                   best_path, best_rank, selection)


def _accept(args, run_dir, config, resources, objective, agent_class,
            best_path, best_rank, selection):
    accept = {}
    for mode, use_low in (("rho1", False), ("sac", True)):
        with redirect_stdout(StringIO()):
            agent = load_agent(best_path, config, resources, agent_class, use_low,
                               args.accept_seed_start)
            rows = evaluate(agent, config, resources, objective,
                            args.accept_seed_start, args.num_seeds, use_low)
        accept[mode] = {"rows": rows, "summary": aggregate(rows)}
        s = accept[mode]["summary"]
        print(f"[accept:{mode}] deadline={s['deadline_success_rate']['mean']:.4f} "
              f"rel={s['reliability_success_rate']['mean']:.4f} "
              f"E={s['total_energy']['mean'] / 1000:.2f}kJ "
              f"resp={s['average_workflow_response_time']['mean']:.2f} "
              f"rho={s['rho_mean']['mean']:.4f}", flush=True)

    paired = [
        float(a["total_energy"]) - float(b["total_energy"])
        for a, b in zip(accept["rho1"]["rows"], accept["sac"]["rows"])
    ]
    local = [
        float(row["total_low_sac_energy_saved"]) for row in accept["sac"]["rows"]
    ]
    local_ratio = [
        float(row["low_sac_total_energy_saving_ratio"]) for row in accept["sac"]["rows"]
    ]
    report = {
        "label": args.label or run_dir.name,
        "run_dir": str(run_dir),
        "config": str(Path(args.config).resolve()),
        "agent_class": agent_class.__name__,
        "selection_seed_block": [args.select_seed_start,
                                 args.select_seed_start + args.num_seeds - 1],
        "acceptance_seed_block": [args.accept_seed_start,
                                  args.accept_seed_start + args.num_seeds - 1],
        "deadline_target": args.deadline_target,
        "reliability_target": args.reliability_target,
        "selected_checkpoint": best_path.name,
        "selection_feasible": bool(best_rank[0] >= 1.0),
        "selection_table": selection,
        "acceptance": accept,
        "sac_system_paired_saving_J": describe(paired),
        "sac_local_counterfactual_saving_J": describe(local),
        "sac_local_counterfactual_saving_ratio": describe(local_ratio),
        "sac_seeds_with_system_saving": int(sum(1 for v in paired if v > 0.0)),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    paired_stats = report["sac_system_paired_saving_J"]
    print(
        f"[paired] system saving {paired_stats['mean'] / 1000:.3f} kJ "
        f"+/- {paired_stats['ci95_half_width'] / 1000:.3f} "
        f"({report['sac_seeds_with_system_saving']}/{args.num_seeds} seeds improved); "
        f"local counterfactual {report['sac_local_counterfactual_saving_ratio']['mean']:.4%}",
        flush=True,
    )
    print(f"output={output}", flush=True)


if __name__ == "__main__":
    main()
