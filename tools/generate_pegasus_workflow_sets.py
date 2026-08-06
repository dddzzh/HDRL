from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.pegasus_workflow_generator import DATASET_SCALE_LIMITS
from src.datasets.workflow_factory import create_workflows
from src.simulator.resource import load_resources


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate workflow sets by sampling the root data/*.xml templates."
    )
    parser.add_argument("--output-dir", default="outputs/pegasus_workflow_sets")
    parser.add_argument("--dataset-scales", default="small,middle,large")
    parser.add_argument("--seeds-per-scale", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--arrival-rate", type=float, default=0.05)
    parser.add_argument(
        "--deadline-type",
        choices=["tight", "medium", "loose"],
        default="medium",
    )
    parser.add_argument("--Rreq", "--rreq", dest="rreq", type=float, default=0.9)
    parser.add_argument("--resource-config", default="configs/resources_edge_cloud_default.json")
    parser.add_argument("--num-edge", type=int, default=20)
    parser.add_argument("--num-cloud", type=int, default=6)
    return parser.parse_args()


def parse_scales(value: str) -> list[str]:
    scales = []
    for item in value.split(","):
        scale = item.strip().lower()
        if not scale:
            continue
        if scale not in DATASET_SCALE_LIMITS:
            raise ValueError(
                f"invalid dataset scale {scale!r}; expected one of {sorted(DATASET_SCALE_LIMITS)}"
            )
        if scale not in scales:
            scales.append(scale)
    if not scales:
        raise ValueError("at least one dataset scale is required")
    return scales


def generator_config(args, dataset_scale: str) -> dict:
    return {
        "path": "data",
        "num_workflows": 1,
        "dataset_scale": dataset_scale,
        "arrival_process": "cai_2025_exponential",
        "arrival_rate": args.arrival_rate,
        "deadline_type": args.deadline_type,
        "Rreq": float(args.rreq),
    }


def generate_scale(args, dataset_scale: str, resources: dict) -> list[dict]:
    workflows = []
    for offset in range(args.seeds_per_scale):
        seed = args.base_seed + offset
        workflow = create_workflows(
            generator_config(args, dataset_scale),
            source="pegasus",
            seed=seed,
            resources=resources,
        )[0]
        workflow.workflow_id = f"{dataset_scale}_seed{seed:04d}"
        payload = workflow.to_mapping()
        payload.pop("edge_data_mb", None)
        payload.update(
            {
                "seed": seed,
                "dataset_scale": dataset_scale,
                "template_name": workflow.template_name,
            }
        )
        workflows.append(payload)
    return workflows


def write_scale(output_dir: Path, args, dataset_scale: str, workflows: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{dataset_scale}.json"
    payload = {
        "source": "pegasus_xml_templates",
        "data_path": "data",
        "dataset_scale": dataset_scale,
        "num_workflows": len(workflows),
        "deadline_type": args.deadline_type,
        "Rreq": float(args.rreq),
        "workflows": workflows,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return path


def main():
    args = parse_args()
    scales = parse_scales(args.dataset_scales)
    resources = load_resources(
        str(PROJECT_ROOT / args.resource_config),
        num_edge=args.num_edge,
        num_cloud=args.num_cloud,
        seed=args.base_seed,
    )
    output_dir = PROJECT_ROOT / args.output_dir
    for dataset_scale in scales:
        workflows = generate_scale(args, dataset_scale, resources)
        path = write_scale(output_dir, args, dataset_scale, workflows)
        print(f"wrote {path} ({len(workflows)} workflows)")


if __name__ == "__main__":
    main()
