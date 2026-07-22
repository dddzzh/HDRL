from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.workflow_factory import create_workflows
from src.simulator.resource import load_resources


SMALL_TASK_COUNTS = (10, 20, 30, 40, 50)
LARGE_TASK_COUNTS = (200, 400, 600, 800, 1000)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Pegasus workflow sets with workflow deadlines.")
    parser.add_argument("--output-dir", default="outputs/pegasus_workflow_sets")
    parser.add_argument("--seeds-per-class", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--arrival-rate", type=float, default=0.5)
    parser.add_argument("--deadline-type", dest="deadline_type", choices=["tight", "medium", "loose"], default="medium")
    parser.add_argument("--Rreq", "--rreq", dest="rreq", type=float, default=0.9)
    parser.add_argument("--resource-config", default="configs/resources_edge_cloud_default.json")
    parser.add_argument("--num-edge", type=int, default=20)
    parser.add_argument("--num-cloud", type=int, default=6)
    return parser.parse_args()


def generator_config(task_count: int, arrival_rate: float, deadline_type: str, rreq: float) -> dict:
    return {
        "num_workflows": 1,
        "num_tasks": task_count,
        "num_levels": max(2, int(math.sqrt(task_count))),
        "arrival_rate": arrival_rate,
        "task_length_cycles_min": 1.0e8,
        "task_length_cycles_max": 15.0e8,
        "task_size_kb_min": 100.0,
        "task_size_kb_max": 1000.0,
        "deadline_type": deadline_type,
        "Rreq": float(rreq),
        "reliability_target": float(rreq),
    }


def generate_class(
    scale: str,
    task_count: int,
    seeds_per_class: int,
    base_seed: int,
    arrival_rate: float,
    deadline_type: str,
    rreq: float,
    resources: dict,
):
    workflows = []
    for offset in range(seeds_per_class):
        seed = base_seed + offset
        generated = create_workflows(
            generator_config(task_count, arrival_rate, deadline_type, rreq),
            source="pegasus",
            seed=seed,
            resources=resources,
        )
        workflow = generated[0]
        workflow.workflow_id = f"{scale}_task{task_count}_seed{seed:04d}"
        payload = workflow.to_mapping()
        payload.pop("edge_data_mb", None)
        payload["seed"] = seed
        payload["scale"] = scale
        payload["task_count_class"] = task_count
        workflows.append(payload)
    return workflows


def write_class(
    output_dir: Path,
    scale: str,
    task_count: int,
    workflows: list[dict],
    deadline_type: str,
    rreq: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{scale}_tasks_{task_count}.json"
    payload = {
        "source": "pegasus_workflow_generator",
        "scale": scale,
        "task_count": task_count,
        "num_workflows": len(workflows),
        "task_length_cycles_range": [1.0e8, 15.0e8],
        "task_size_kb_range": [100.0, 1000.0],
        "deadline_type": deadline_type,
        "Rreq": float(rreq),
        "deadline_semantics": (
            "workflow-level absolute_deadline; relative_deadline = "
            "deadline_multiplier * reference_makespan"
        ),
        "workflows": workflows,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def main():
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    resources = load_resources(
        str(PROJECT_ROOT / args.resource_config),
        num_edge=args.num_edge,
        num_cloud=args.num_cloud,
        seed=args.base_seed,
    )
    written = []
    for scale, task_counts in (("small", SMALL_TASK_COUNTS), ("large", LARGE_TASK_COUNTS)):
        for task_count in task_counts:
            workflows = generate_class(
                scale,
                task_count,
                args.seeds_per_class,
                args.base_seed,
                args.arrival_rate,
                args.deadline_type,
                args.rreq,
                resources,
            )
            path = write_class(output_dir, scale, task_count, workflows, args.deadline_type, args.rreq)
            written.append(path)
            print(f"wrote {path} ({len(workflows)} workflows)")
    print(f"done: {len(written)} files")


if __name__ == "__main__":
    main()
