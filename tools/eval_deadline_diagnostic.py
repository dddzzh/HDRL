from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.D3QN.hierarchical_d3qn_agent import HierarchicalD3QNAgent
from src.D3QN.reward import HierarchicalObjective
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from tools.hierarchical_common import (
    build_env,
    build_resources,
    build_workflows,
    configure_agent_state,
)
from tools.train_hierarchical_d3qn import collect_d3qn_episode


CASES = (
    ("heft_rho1", "heft", True),
    ("d3qn_rho1", "d3qn", True),
    ("d3qn_sac", "d3qn", False),
)
REPORT_METRICS = (
    "deadline_success_rate",
    "reliability_success_rate",
    "total_energy",
    "average_workflow_response_time",
    "global_makespan",
    "rho_mean",
    "mean_task_time_regret",
    "high_deadline_crossing_count",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose upper scheduling and lower DVFS contributions on paired seeds."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-start", type=int, default=242000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument(
        "--dataset-scale",
        choices=("small", "medium", "large"),
        default="small",
    )
    parser.add_argument(
        "--dataset-source",
        choices=("pegasus", "alibaba_json"),
        default="pegasus",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Alibaba workflow pool JSON; used only with --dataset-source alibaba_json.",
    )
    parser.add_argument(
        "--num-workflows",
        type=int,
        default=None,
        help="Override workflows sampled per evaluation seed.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def describe(values):
    array = np.asarray(values, dtype=np.float64)
    count = int(array.size)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if count > 1 else 0.0
    t_critical = 2.093 if count == 20 else 1.96
    ci95 = float(t_critical * std / math.sqrt(count)) if count > 1 else 0.0
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95_half_width": ci95,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def evaluate_case(config, resources, agent, objective, seed, high_algorithm, fixed_rho1):
    workflows = build_workflows(config, resources, seed, smoke_test=False)
    env = build_env(workflows, resources, dict(config.get("simulation", {})), seed)
    result = collect_d3qn_episode(
        env,
        agent,
        objective,
        HighRolloutBuffer(),
        LowRolloutBuffer(),
        training=False,
        high_deterministic=True,
        low_deterministic=True,
        fixed_low_rho_one=fixed_rho1,
        high_algorithm=high_algorithm,
    )
    return result["metrics"]


def write_outputs(output_dir, rows, summary, manifest):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "diagnostic_rows.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with (output_dir / "experiment_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    csv_rows = []
    for case_name, metrics in summary.items():
        for metric, stats in metrics.items():
            csv_rows.append({"case": case_name, "metric": metric, **stats})
    with (output_dir / "diagnostic_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "case",
                "metric",
                "n",
                "mean",
                "std",
                "ci95_half_width",
                "min",
                "max",
            ),
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# Deadline diagnostic",
        "",
        "All cases use the same checkpoint configuration and paired workflow seeds.",
        "Energy is complete-system total energy per seed, averaged over seeds.",
        "",
        "| Case | Deadline | Reliability | Energy (kJ) | Response (s) | rho | Crossings |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case_name, _, _ in CASES:
        metrics = summary[case_name]
        lines.append(
            f"| {case_name} "
            f"| {metrics['deadline_success_rate']['mean']:.2%} "
            f"| {metrics['reliability_success_rate']['mean']:.2%} "
            f"| {metrics['total_energy']['mean'] / 1000.0:.2f} "
            f"| {metrics['average_workflow_response_time']['mean']:.2f} "
            f"| {metrics['rho_mean']['mean']:.3f} "
            f"| {metrics['high_deadline_crossing_count']['mean']:.2f} |"
        )
    (output_dir / "diagnostic_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if args.num_workflows is not None and args.num_workflows <= 0:
        raise ValueError("--num-workflows must be positive.")
    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload.get("config") or {})
    if not config:
        raise RuntimeError("Checkpoint does not contain its training configuration.")
    generator_cfg = config.setdefault("generator", {})
    if args.dataset_source == "alibaba_json":
        generator_cfg["type"] = "alibaba_json"
        generator_cfg.pop("source", None)
        generator_cfg.pop("dataset_scale", None)
        generator_cfg["path"] = args.dataset_path or "outputs/alibaba_test.json"
        generator_cfg["shuffle"] = True
        generator_cfg["max_workflows"] = int(args.num_workflows or 20)
        generator_cfg["num_workflows"] = int(args.num_workflows or 20)
    else:
        generator_cfg["type"] = "pegasus"
        scale_name = "middle" if args.dataset_scale == "medium" else args.dataset_scale
        generator_cfg["dataset_scale"] = scale_name
        if args.num_workflows is not None:
            generator_cfg["num_workflows"] = int(args.num_workflows)

    server_seed = int(config.get("training", {}).get("seed", 42))
    with redirect_stdout(io.StringIO()):
        resources = build_resources(config, server_seed)
    agent_config = configure_agent_state(config, resources, smoke_test=False)
    agent_config["use_low_agent"] = True
    agent = HierarchicalD3QNAgent(agent_config, seed=args.seed_start)
    agent.load_checkpoint(checkpoint_path)
    agent.high_agent.epsilon = 0.0
    objective = HierarchicalObjective(config.get("objective", {}))

    rows = []
    for offset in range(args.num_seeds):
        seed = args.seed_start + offset
        for case_name, high_algorithm, fixed_rho1 in CASES:
            with redirect_stdout(io.StringIO()):
                metrics = evaluate_case(
                    config,
                    resources,
                    agent,
                    objective,
                    seed,
                    high_algorithm,
                    fixed_rho1,
                )
            row = {
                "case": case_name,
                "eval_seed": seed,
                **{key: metrics.get(key) for key in REPORT_METRICS},
            }
            rows.append(row)
            print(
                f"[eval] seed={seed} case={case_name:<10} "
                f"deadline={float(row['deadline_success_rate']):6.2%} "
                f"energy_kJ={float(row['total_energy']) / 1000.0:8.2f} "
                f"response={float(row['average_workflow_response_time']):7.2f} "
                f"rho={float(row['rho_mean']):.3f}",
                flush=True,
            )

    summary = {}
    for case_name, _, _ in CASES:
        case_rows = [row for row in rows if row["case"] == case_name]
        summary[case_name] = {
            metric: describe([float(row[metric]) for row in case_rows])
            for metric in REPORT_METRICS
        }
    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_episode": int(payload.get("episode", 0)),
        "checkpoint_config": config,
        "server_seed": server_seed,
        "workflow_seed_start": args.seed_start,
        "workflow_seed_end": args.seed_start + args.num_seeds - 1,
        "num_seeds": args.num_seeds,
        "dataset_source": args.dataset_source,
        "dataset_scale": (
            args.dataset_scale if args.dataset_source == "pegasus" else None
        ),
        "dataset_path": (
            str(generator_cfg.get("path"))
            if args.dataset_source == "alibaba_json"
            else None
        ),
        "num_workflows": int(
            generator_cfg.get(
                "max_workflows",
                generator_cfg.get("num_workflows", 0),
            )
        ),
        "cases": [
            {
                "name": case_name,
                "high_algorithm": high_algorithm,
                "fixed_low_rho_one": fixed_rho1,
            }
            for case_name, high_algorithm, fixed_rho1 in CASES
        ],
    }
    write_outputs(output_dir, rows, summary, manifest)
    print(f"output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
