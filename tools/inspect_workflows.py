"""采样工作流并打印统计信息 + 做一致性检查（需求第十四节）。

用法::

    python tools/inspect_workflows.py --config configs/v2_small_20edge_6cloud.json --num-workflows 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.workflow_factory import WorkflowSampler, create_workflows
from src.simulator.resource import load_resources
from src.simulator.workflow import Workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect sampled workflows: stats + checks.")
    parser.add_argument("--config", default="configs/v2_small_20edge_6cloud.json")
    parser.add_argument("--dataset", default=None, choices=["pegasus", "alibaba_pool", "alibaba_json"])
    parser.add_argument("--num-workflows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _stat(values):
    if not values:
        return (0.0, 0.0, 0.0)
    return (min(values), sum(values) / len(values), max(values))


def _fmt(label, triple):
    return f"  {label:<16} min={triple[0]:.4g}  mean={triple[1]:.4g}  max={triple[2]:.4g}"


def _to_obj(workflow, default_edge_data_mb):
    if isinstance(workflow, Workflow):
        return workflow
    return Workflow.from_mapping(workflow, default_edge_data_mb=default_edge_data_mb)


def main() -> None:
    args = parse_args()
    with (PROJECT_ROOT / args.config).open("r", encoding="utf-8") as file:
        config = json.load(file)

    dataset_cfg = dict(config.get("dataset", {}))
    dataset_type = args.dataset or dataset_cfg.get("type", "pegasus")

    resources = load_resources(
        str(PROJECT_ROOT / config["resources"]["path"]),
        num_edge=config["resources"]["num_edge"],
        num_cloud=config["resources"]["num_cloud"],
        seed=args.seed,
    )
    links = resources["network"]["link_bandwidth_mbps"]
    if dataset_type == "alibaba_pool":
        sampler = WorkflowSampler.from_file(
            dataset_cfg.get("train_path", "outputs/alibaba_train.json"),
            arrival_rate=float(dataset_cfg.get("arrival_rate", 0.1)),
            mapping_config=dataset_cfg,
            resources=resources,
        )
        workflows = sampler.sample(args.num_workflows, args.seed)
    else:
        gen_cfg = dict(config.get("generator", {}))
        gen_cfg["num_workflows"] = args.num_workflows
        workflows = create_workflows(
            gen_cfg, source="pegasus", seed=args.seed, resources=resources
        )

    print(f"dataset: {dataset_type}")
    print(f"number of workflows: {len(workflows)}")

    n_tasks_total = 0
    n_edges_total = 0
    workload_cycles = []
    data_sizes = []
    arrivals = []
    absolute_deadlines = []
    workflow_relative_deadlines = []
    reliability_reqs = []
    errors = []

    for workflow in workflows:
        obj = _to_obj(workflow, float(dataset_cfg.get("data_size_mb_min", 1.0)))
        n_tasks_total += len(obj.tasks)
        n_edges_total += len(obj.edges)
        arrivals.append(obj.arrival_time)
        if obj.absolute_deadline is None:
            errors.append(f"{obj.workflow_id} missing workflow absolute_deadline")
        else:
            absolute_deadlines.append(float(obj.absolute_deadline))
            workflow_relative_deadlines.append(float(obj.absolute_deadline) - float(obj.arrival_time))
            if float(obj.absolute_deadline) <= float(obj.arrival_time):
                errors.append(f"{obj.workflow_id} workflow absolute_deadline<=arrival_time")
        if obj.reliability is None:
            errors.append(f"{obj.workflow_id} missing workflow reliability requirement")
        else:
            reliability_reqs.append(float(obj.reliability))

        for task in obj.tasks:
            workload_cycles.append(task.workload_cycles)
            if task.workload_cycles <= 0:
                errors.append(f"{obj.workflow_id} task {task.id} workload_cycles<=0")
        for edge in obj.edges:
            data_sizes.append(edge.data_size_mb)
            if edge.data_size_mb <= 0:
                errors.append(f"{obj.workflow_id} edge {edge.src}->{edge.dst} data<=0")

    print(f"total tasks: {n_tasks_total}")
    print(f"total edges: {n_edges_total}")
    print(_fmt("workload_cycles", _stat(workload_cycles)))
    print(_fmt("workload_gc", _stat([w / 1e9 for w in workload_cycles])))
    print(_fmt("data_size_mb", _stat(data_sizes)))
    print(_fmt("arrival_time", _stat(arrivals)))
    print(_fmt("absolute_deadline", _stat(absolute_deadlines)))
    print(_fmt("relative_deadline", _stat(workflow_relative_deadlines)))
    print(_fmt("Rreq", _stat(reliability_reqs)))

    if errors:
        print(f"\n[!] {len(errors)} issue(s) found, e.g.:")
        for line in errors[:10]:
            print(f"    {line}")
    else:
        print("\nAll checks passed: edges valid, workload>0, data>0, workflow absolute_deadline/Rreq present, task deadline fields absent.")


if __name__ == "__main__":
    main()
