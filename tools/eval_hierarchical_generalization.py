from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.D3QN.hierarchical_d3qn_agent import HierarchicalD3QNAgent
from src.D3QN.reward import HierarchicalObjective
from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
from tools.hierarchical_common import build_env, build_resources, build_workflows, load_json
from tools.train_hierarchical_d3qn import collect_d3qn_episode


DEFAULT_WORKFLOW_COUNTS = (5, 10, 20, 30)
DEFAULT_VM_LAYOUTS = ((5, 2), (10, 3), (15, 5), (20, 6))
REPORT_METRICS = (
    "total_energy",
    "energy_per_workflow",
    "energy_per_task",
    "average_response_time",
    "global_makespan",
    "deadline_success_rate",
    "reliability_success_rate",
    "mean_task_time_regret",
    "low_sac_total_energy_saving_ratio",
    "rho_mean",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate D3QN+SAC generalization over workflow count and VM count, "
            "with HEFT using the exact same trained SAC as a paired baseline."
        )
    )
    parser.add_argument("--config", default="configs/hierarchical_d3qn_sac.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-start", type=int, default=342000)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument(
        "--workflow-counts",
        default=",".join(str(value) for value in DEFAULT_WORKFLOW_COUNTS),
        help="Comma-separated workflow counts, for example 5,10,20,30.",
    )
    parser.add_argument(
        "--vm-layouts",
        default=",".join(f"{edge}:{cloud}" for edge, cloud in DEFAULT_VM_LAYOUTS),
        help="Comma-separated edge:cloud layouts, for example 5:2,10:3,20:6.",
    )
    parser.add_argument(
        "--modes",
        default="d3qn_sac,heft_same_sac",
        help="Subset of d3qn_sac,heft_same_sac.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def parse_positive_ints(value):
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        number = int(item)
        if number <= 0:
            raise ValueError("Counts must be positive integers.")
        if number not in result:
            result.append(number)
    if not result:
        raise ValueError("At least one count is required.")
    return result


def parse_vm_layouts(value):
    result = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid VM layout {item!r}; expected edge:cloud.")
        edge, cloud = (int(part) for part in parts)
        if edge < 0 or cloud < 0 or edge + cloud <= 0:
            raise ValueError(f"Invalid VM layout {item!r}.")
        layout = (edge, cloud)
        if layout not in result:
            result.append(layout)
    if not result:
        raise ValueError("At least one VM layout is required.")
    return result


def parse_modes(value):
    allowed = {"d3qn_sac", "heft_same_sac"}
    result = []
    for item in str(value).split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unsupported mode {item!r}; expected one of {sorted(allowed)}.")
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError("At least one evaluation mode is required.")
    return result


def resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_checkpoint_metadata(checkpoint_path):
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") != "hierarchical_d3qn_primary_v3":
        raise RuntimeError("Generalization evaluation requires a primary-only D3QN v3 checkpoint.")
    checkpoint_config = copy.deepcopy(payload.get("config") or {})
    agent_cfg = checkpoint_config.get("agent", {})
    state_cfg = agent_cfg.get("shared", {}).get("state", {})
    max_servers = int(state_cfg.get("max_servers", 0))
    if max_servers <= 0:
        state_dict = payload.get("high_q_eval_state_dict", {})
        output_weights = [
            value
            for key, value in state_dict.items()
            if key.startswith("advantage.") and key.endswith(".weight")
        ]
        if not output_weights:
            raise RuntimeError("Cannot infer checkpoint server capacity.")
        max_servers = int(output_weights[-1].shape[0])
    return payload, checkpoint_config, max_servers


def build_cases(config, workflow_counts, vm_layouts):
    base_workflows = int(config.get("generator", {}).get("num_workflows", 10))
    base_edge = int(config.get("resources", {}).get("num_edge", 20))
    base_cloud = int(config.get("resources", {}).get("num_cloud", 6))
    cases = []
    for count in workflow_counts:
        cases.append({
            "case": f"workflows_{count}",
            "dimension": "workflow_count",
            "scale_value": int(count),
            "num_workflows": int(count),
            "num_edge": base_edge,
            "num_cloud": base_cloud,
        })
    for edge, cloud in vm_layouts:
        cases.append({
            "case": f"vms_{edge}e_{cloud}c",
            "dimension": "vm_count",
            "scale_value": int(edge + cloud),
            "num_workflows": base_workflows,
            "num_edge": int(edge),
            "num_cloud": int(cloud),
        })
    return cases


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    count = int(values.size)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if count > 1 else 0.0
    t_critical = 2.093 if count == 20 else 1.96
    ci95 = float(t_critical * std / math.sqrt(count)) if count > 1 else 0.0
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95_half_width": ci95,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def add_normalized_metrics(metrics, case):
    result = dict(metrics)
    workflows = max(1, int(case["num_workflows"]))
    steps = max(1.0, float(metrics.get("steps", 0.0)))
    energy = float(metrics.get("total_energy", 0.0))
    result["energy_per_workflow"] = energy / workflows
    result["energy_per_task"] = energy / steps
    return result


def summarize_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["case"], row["mode"])
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (case_name, mode), group in sorted(grouped.items()):
        first = group[0]
        metric_stats = {}
        for metric in REPORT_METRICS:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            if values:
                metric_stats[metric] = describe(values)
        summaries.append({
            "case": case_name,
            "dimension": first["dimension"],
            "scale_value": first["scale_value"],
            "num_workflows": first["num_workflows"],
            "num_edge": first["num_edge"],
            "num_cloud": first["num_cloud"],
            "num_servers": first["num_servers"],
            "mode": mode,
            "metrics": metric_stats,
        })
    return summaries


def scalar_row(row):
    result = {}
    for key, value in row.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            result[key] = value
    return result


def write_outputs(output_dir, rows, summaries, manifest):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generalization_rows.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "generalization_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)
    with (output_dir / "experiment_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    flat_rows = [scalar_row(row) for row in rows]
    fieldnames = sorted({key for row in flat_rows for key in row})
    with (output_dir / "generalization_rows.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    summary_rows = []
    for row in summaries:
        base = {key: value for key, value in row.items() if key != "metrics"}
        for metric, stats in row["metrics"].items():
            summary_rows.append({"metric": metric, **base, **stats})
    summary_fields = [
        "case", "dimension", "scale_value", "num_workflows", "num_edge", "num_cloud",
        "num_servers", "mode", "metric", "n", "mean", "std", "ci95_half_width", "min", "max",
    ]
    with (output_dir / "generalization_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def print_summary(summaries):
    print("[generalization] mean over evaluation seeds")
    for row in summaries:
        metrics = row["metrics"]
        energy = metrics.get("total_energy", {}).get("mean", float("nan"))
        per_workflow = metrics.get("energy_per_workflow", {}).get("mean", float("nan"))
        response = metrics.get("average_response_time", {}).get("mean", float("nan"))
        deadline = metrics.get("deadline_success_rate", {}).get("mean", float("nan"))
        reliability = metrics.get("reliability_success_rate", {}).get("mean", float("nan"))
        print(
            f"  case={row['case']:<18} mode={row['mode']:<14} "
            f"energy_J={energy:9.2f} energy_per_wf={per_workflow:8.2f} "
            f"response_s={response:6.3f} deadline={deadline:6.2%} reliability={reliability:6.2%}"
        )


def evaluate(config, checkpoint_path, workflow_counts, vm_layouts, modes, seed_start, num_seeds, smoke_test=False):
    payload, checkpoint_config, max_servers = load_checkpoint_metadata(checkpoint_path)
    agent_cfg = copy.deepcopy(checkpoint_config.get("agent") or config.get("agent", {}))
    agent_cfg["use_low_agent"] = True
    agent_cfg.setdefault("shared", {}).setdefault("state", {})["max_servers"] = max_servers
    agent = HierarchicalD3QNAgent(agent_cfg, seed=seed_start)
    agent.load_checkpoint(checkpoint_path)
    agent.high_agent.epsilon = 0.0
    objective_cfg = checkpoint_config.get("objective", config.get("objective", {}))
    objective = HierarchicalObjective(objective_cfg)

    cases = build_cases(config, workflow_counts, vm_layouts)
    oversized = [case for case in cases if case["num_edge"] + case["num_cloud"] > max_servers]
    if oversized:
        details = ", ".join(
            f"{case['case']}={case['num_edge'] + case['num_cloud']}" for case in oversized
        )
        raise ValueError(
            f"VM count exceeds checkpoint action capacity max_servers={max_servers}: {details}. "
            "A flat D3QN output head cannot zero-shot select newly added VM slots; train a matched-size "
            "checkpoint or replace it with a shared per-server Q scorer."
        )

    rows = []
    cache = {}
    for case in cases:
        scenario_config = copy.deepcopy(config)
        scenario_config.setdefault("generator", {})["num_workflows"] = int(case["num_workflows"])
        scenario_config.setdefault("resources", {})["num_edge"] = int(case["num_edge"])
        scenario_config.setdefault("resources", {})["num_cloud"] = int(case["num_cloud"])
        with redirect_stdout(io.StringIO()):
            resources = build_resources(scenario_config, seed_start)
        print(
            f"[case] {case['case']} workflows={case['num_workflows']} "
            f"edge={case['num_edge']} cloud={case['num_cloud']} seeds={num_seeds}"
        )
        for mode in modes:
            high_algorithm = "heft" if mode == "heft_same_sac" else "d3qn"
            for offset in range(int(num_seeds)):
                eval_seed = int(seed_start + offset)
                cache_key = (
                    case["num_workflows"], case["num_edge"], case["num_cloud"], mode, eval_seed
                )
                if cache_key not in cache:
                    workflows = build_workflows(
                        scenario_config, resources, eval_seed, smoke_test=smoke_test
                    )
                    env = build_env(
                        workflows,
                        resources,
                        dict(scenario_config.get("simulation", {})),
                        eval_seed,
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
                        fixed_low_rho_one=False,
                        high_algorithm=high_algorithm,
                    )
                    cache[cache_key] = result["metrics"]
                metrics = add_normalized_metrics(cache[cache_key], case)
                rows.append({
                    **case,
                    "num_servers": int(case["num_edge"] + case["num_cloud"]),
                    "mode": mode,
                    "eval_seed": eval_seed,
                    **metrics,
                })
    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_episode": int(payload.get("episode", 0)),
        "checkpoint_max_servers": max_servers,
        "seed_start": int(seed_start),
        "seed_end": int(seed_start + num_seeds - 1),
        "num_seeds": int(num_seeds),
        "workflow_counts": workflow_counts,
        "vm_layouts": [{"num_edge": edge, "num_cloud": cloud} for edge, cloud in vm_layouts],
        "modes": modes,
        "vm_generalization_note": (
            "VM layouts do not exceed the checkpoint's fixed action capacity. Counts below the "
            "training layout measure resource-scale sensitivity; counts above the capacity require retraining "
            "or a variable-cardinality per-server scorer."
        ),
    }
    return rows, summarize_rows(rows), manifest


def main():
    args = parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    config = load_json(args.config)
    checkpoint_path = resolve_path(args.checkpoint)
    workflow_counts = parse_positive_ints(args.workflow_counts)
    vm_layouts = parse_vm_layouts(args.vm_layouts)
    modes = parse_modes(args.modes)
    rows, summaries, manifest = evaluate(
        config,
        checkpoint_path,
        workflow_counts,
        vm_layouts,
        modes,
        args.seed_start,
        args.num_seeds,
        smoke_test=args.smoke_test,
    )
    if args.output_dir:
        output_dir = resolve_path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / "hierarchical_generalization" / datetime.now().strftime("%Y%m%d_%H%M%S")
    write_outputs(output_dir, rows, summaries, manifest)
    print_summary(summaries)
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
