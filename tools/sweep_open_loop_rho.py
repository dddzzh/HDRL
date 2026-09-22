"""Open-loop rho sweep with a fixed HEFT placement policy.

The closed-form audit in ``analyze_dvfs_feasible_window.py`` ignores queueing.
This sweep runs the real simulator, so it also captures the externality that made
every open-loop frequency *worse* in docs/V2_RESULTS_20260827.md section 4:
slowing a task down keeps a server busy longer, which pushes later tasks onto
less efficient machines.

Placement is HEFT (deterministic, no learning), so any difference between rho
levels is attributable to DVFS alone.

Usage::

    py -3.13 tools/sweep_open_loop_rho.py --resources configs/resources_edge_cloud_default.json
    py -3.13 tools/sweep_open_loop_rho.py --resources configs/resources_edge_cloud_dvfs_v3.json
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.pegasus_workflow_generator import PegasusWorkflowGenerator
from src.schedulers.heft import HEFTScheduler
from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.resource import load_resources


def run_one(resource_path, rho, seed, num_edge, num_cloud, sim_config, generator_config):
    resources = load_resources(
        resource_path, num_edge=num_edge, num_cloud=num_cloud, seed=seed
    )
    generator = PegasusWorkflowGenerator(generator_config, seed=seed)
    workflows = generator.generate(
        servers=resources["servers"],
        link_bandwidth_mbps=resources["network"].get("links_mbps"),
    )
    env = EdgeCloudWorkflowEnv(
        workflows=workflows,
        resources=resources,
        sim_config={**sim_config, "default_rho": rho},
        seed=seed,
    )
    result = env.run(HEFTScheduler(params={"rho": rho}, seed=seed))
    summary = result["summary"]
    rhos = [float(event["rho"]) for event in result["event_log"] if "rho" in event]
    return {
        "energy": float(summary["total_energy"]),
        "deadline": float(summary.get("deadline_success_rate", float("nan"))),
        "reliability_rate": float(summary.get("reliability_success_rate", float("nan"))),
        "avg_reliability": float(summary.get("avg_workflow_reliability", float("nan"))),
        "response": float(summary.get("avg_workflow_response_time", float("nan"))),
        "rho_mean": statistics.fmean(rhos) if rhos else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", default="configs/resources_edge_cloud_default.json")
    parser.add_argument("--num-edge", type=int, default=28)
    parser.add_argument("--num-cloud", type=int, default=9)
    parser.add_argument("--num-workflows", type=int, default=20)
    parser.add_argument("--dataset-scale", default="middle")
    parser.add_argument("--r-req", type=float, default=0.9)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(542000, 542005)))
    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=[1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5],
    )
    args = parser.parse_args()

    generator_config = {
        "num_workflows": args.num_workflows,
        "dataset_scale": args.dataset_scale,
        "arrival_process": "cai_2025_exponential",
        "arrival_rate": 0.1,
        "deadline_type": "medium",
        "workflow_reliability_requirement": args.r_req,
        "energy_eta": 0.5,
    }
    sim_config = {
        "ready_task_order": "smallest_workload",
        "deadline_type": "medium",
        "workflow_reliability_requirement": args.r_req,
        "rho_min": 0.1,
        "energy_eta": 0.5,
        "default_reliability": 0.95,
        "reliability_exponent": 1.0,
        "enable_backup_duplication": False,
        "remaining_task_reliability_estimate": 0.999,
        "server_history_ema_alpha": 0.9,
    }

    print(f"resources = {args.resources}")
    print(f"seeds = {args.seeds[0]}..{args.seeds[-1]} ({len(args.seeds)}), "
          f"{args.num_edge} edge + {args.num_cloud} cloud, scale={args.dataset_scale}, "
          f"R_req={args.r_req}")
    print()
    print(f"{'rho':>6}{'energy (kJ)':>14}{'vs rho=1':>10}{'deadline':>10}"
          f"{'rel rate':>10}{'avg rel':>9}{'resp (s)':>10}")

    baseline = None
    for rho in args.rhos:
        rows = [
            run_one(
                args.resources, rho, seed, args.num_edge, args.num_cloud,
                sim_config, generator_config,
            )
            for seed in args.seeds
        ]
        energy = statistics.fmean(r["energy"] for r in rows) / 1000.0
        if baseline is None:
            baseline = energy
        deadline = statistics.fmean(r["deadline"] for r in rows)
        rel_rate = statistics.fmean(r["reliability_rate"] for r in rows)
        avg_rel = statistics.fmean(r["avg_reliability"] for r in rows)
        resp = statistics.fmean(r["response"] for r in rows)
        delta = (energy - baseline) / baseline
        print(f"{rho:>6.2f}{energy:>14.2f}{delta:>+9.1%}{deadline:>10.3f}"
              f"{rel_rate:>10.3f}{avg_rel:>9.4f}{resp:>10.2f}")


if __name__ == "__main__":
    main()
