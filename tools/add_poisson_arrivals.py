from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Any

from src.datasets.arrival_process import (
    CAI_2025_ARRIVAL_PROCESS,
    POISSON_ARRIVAL_DEFINITION,
    sample_configured_arrivals,
)


DEFAULT_INPUT = "outputs/alibaba_workflows.json"
DEFAULT_OUTPUT = "outputs/alibaba_workflows_with_arrivals.json"
DEFAULT_SUMMARY = "outputs/alibaba_workflow_arrival_summary.csv"
ARRIVAL_DEFINITION = POISSON_ARRIVAL_DEFINITION


def load_workflows(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def assign_poisson_arrivals(
    data: dict[str, Any],
    arrival_rate: float,
    start_time: float,
    seed: int,
    shuffle: bool,
    edge_data_mb: float | None = None,
) -> dict[str, Any]:
    arrival_rate = float(arrival_rate)
    if arrival_rate <= 0.0:
        raise ValueError("arrival_rate must be positive")
    arrival_config = {
        "arrival_process": CAI_2025_ARRIVAL_PROCESS,
        "arrival_rate": arrival_rate,
    }

    result = copy.deepcopy(data)
    workflows = list(result.get("workflows", []))
    rng = random.Random(seed)

    if shuffle:
        rng.shuffle(workflows)

    arrival_times, inter_arrival_times, _, arrival_process = sample_configured_arrivals(
        len(workflows),
        arrival_config,
        start_time,
        rng,
    )
    for index, workflow in enumerate(workflows):
        workflow["arrival_index"] = index
        workflow["arrival_time"] = float(arrival_times[index])
        workflow["inter_arrival_time"] = float(inter_arrival_times[index])
        workflow["arrival_process"] = arrival_process
        workflow["arrival_rate"] = float(arrival_rate)

    if edge_data_mb is None:
        edge_data_mb = result.get("network", {}).get("edge_data_mb")

    if edge_data_mb is not None:
        result.setdefault("network", {})
        result["network"]["edge_data_mb"] = float(edge_data_mb)
        for workflow in workflows:
            for edge in workflow.get("edges", []):
                edge.setdefault("data_size_mb", float(edge_data_mb))

    result["workflows"] = workflows
    result["num_workflows"] = len(workflows)
    result["arrival_process"] = CAI_2025_ARRIVAL_PROCESS
    result["arrival_definition"] = ARRIVAL_DEFINITION
    result["mean_interarrival"] = 1.0 / float(arrival_rate)
    result["arrival_rate"] = float(arrival_rate)
    result["arrival_start_time"] = float(start_time)
    result["arrival_time_unit"] = "second"
    result["seed"] = seed
    result["arrival_shuffle"] = bool(shuffle)
    return result


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _total_task_length_mi(workflow: dict[str, Any]) -> Any:
    if "total_task_length_mi" in workflow:
        return workflow["total_task_length_mi"]
    return workflow.get("total_workload", "")


def save_summary(data: dict[str, Any], path: str | Path) -> None:
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "arrival_index",
        "workflow_id",
        "arrival_time",
        "inter_arrival_time",
        "n_tasks",
        "n_edges",
        "trace_makespan",
        "total_task_length_mi",
    ]

    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for workflow in data.get("workflows", []):
            writer.writerow(
                {
                    "arrival_index": workflow.get("arrival_index", ""),
                    "workflow_id": workflow.get("workflow_id", ""),
                    "arrival_time": workflow.get("arrival_time", ""),
                    "inter_arrival_time": workflow.get("inter_arrival_time", ""),
                    "n_tasks": workflow.get("n_tasks", ""),
                    "n_edges": workflow.get("n_edges", ""),
                    "trace_makespan": workflow.get("trace_makespan", ""),
                    "total_task_length_mi": _total_task_length_mi(workflow),
                }
            )


def _arrival_stats(data: dict[str, Any]) -> tuple[float, float, float]:
    workflows = data.get("workflows", [])
    if not workflows:
        return 0.0, 0.0, 0.0

    first_arrival = float(workflows[0]["arrival_time"])
    last_arrival = float(workflows[-1]["arrival_time"])
    intervals = [
        float(workflow["inter_arrival_time"])
        for workflow in workflows[1:]
    ]
    average_interval = sum(intervals) / len(intervals) if intervals else 0.0
    return first_arrival, last_arrival, average_interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign Poisson arrival times to Alibaba workflow JSON."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=0.1,
        help="Cai 2025 system-wide Poisson rate lambda in workflows/second.",
    )
    parser.add_argument("--start_time", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--edge-data-mb",
        type=float,
        default=None,
        help="Fill dependency edge data size in MB; preserves input network.edge_data_mb if omitted",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    input_data = load_workflows(input_path)
    output_data = assign_poisson_arrivals(
        data=input_data,
        arrival_rate=args.arrival_rate,
        start_time=args.start_time,
        seed=args.seed,
        shuffle=args.shuffle,
        edge_data_mb=args.edge_data_mb,
    )
    save_json(output_data, output_path)
    save_summary(output_data, summary_path)

    first_arrival, last_arrival, average_interval = _arrival_stats(output_data)

    print(f"Input workflows: {len(input_data.get('workflows', []))}")
    print(f"Output workflows: {len(output_data.get('workflows', []))}")
    print("Arrival process: poisson")
    print(f"Arrival rate lambda: {float(output_data['arrival_rate']):.8f} workflows/s")
    print(f"Mean inter-arrival time: {float(output_data['mean_interarrival']):.6f}")
    print(f"First arrival time: {first_arrival:.6f}")
    print(f"Last arrival time: {last_arrival:.6f}")
    print(f"Average generated inter-arrival time: {average_interval:.6f}")
    print(f"Output JSON: {output_path}")
    print(f"Output summary: {summary_path}")


if __name__ == "__main__":
    main()
