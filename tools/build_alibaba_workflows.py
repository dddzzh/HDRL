from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd


COLUMNS = [
    "task_name",
    "instance_num",
    "job_name",
    "task_type",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = str(PROJECT_ROOT / "dataset" / "cluster-trace-v2018" / "batch_task.csv")
DEFAULT_OUTPUT = "outputs/alibaba_workflows.json"
DEFAULT_SUMMARY = "outputs/alibaba_workflow_summary.csv"
TASK_NAME_RE = re.compile(r"^[A-Za-z]\d+(_\d+)*$")


def is_valid_task_name(task_name: Any) -> bool:
    if not isinstance(task_name, str):
        return False
    return TASK_NAME_RE.fullmatch(task_name.strip()) is not None


def parse_task_name(task_name: str) -> tuple[int, list[int]] | None:
    if not is_valid_task_name(task_name):
        return None
    try:
        parts = task_name.strip()[1:].split("_")
        current_task_original_id = int(parts[0])
        parent_original_ids = [int(part) for part in parts[1:]]
    except (TypeError, ValueError):
        return None
    return current_task_original_id, parent_original_ids


def topological_sort(n_tasks: int, edges: list[tuple[int, int]]) -> list[int]:
    adjacency: list[list[int]] = [[] for _ in range(n_tasks)]
    indegree = [0] * n_tasks

    for src, dst in edges:
        if src < 0 or src >= n_tasks or dst < 0 or dst >= n_tasks:
            return []
        adjacency[src].append(dst)
        indegree[dst] += 1

    ready = deque(node for node in range(n_tasks) if indegree[node] == 0)
    order: list[int] = []

    while ready:
        node = ready.popleft()
        order.append(node)
        for child in adjacency[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    if len(order) != n_tasks:
        return []
    return order


def is_dag(n_tasks: int, edges: list[tuple[int, int]]) -> bool:
    return len(topological_sort(n_tasks, edges)) == n_tasks


def _to_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    task_name = str(row.get("task_name", "")).strip()
    job_name = str(row.get("job_name", "")).strip()
    if not task_name or not job_name:
        return None

    instance_num = _to_int(row.get("instance_num"))
    start_time = _to_float(row.get("start_time"))
    end_time = _to_float(row.get("end_time"))
    plan_cpu = _to_float(row.get("plan_cpu"))
    plan_mem = _to_float(row.get("plan_mem"))
    if any(
        value is None
        for value in (instance_num, start_time, end_time, plan_cpu, plan_mem)
    ):
        return None

    return {
        "task_name": task_name,
        "instance_num": instance_num,
        "job_name": job_name,
        "start_time": start_time,
        "end_time": end_time,
        "plan_cpu": plan_cpu,
        "plan_mem": plan_mem,
    }


def build_workflow(
    job_name: str,
    rows: list[dict[str, Any]],
    min_tasks: int,
    max_tasks: int,
    edge_data_mb: float,
) -> dict[str, Any] | None:
    parsed_rows: list[tuple[int, list[int], dict[str, Any]]] = []
    original_ids: set[int] = set()

    for row in rows:
        parsed = parse_task_name(row["task_name"])
        if parsed is None:
            continue
        original_id, parent_original_ids = parsed
        if original_id in original_ids:
            return None
        original_ids.add(original_id)
        parsed_rows.append((original_id, parent_original_ids, row))

    n_tasks = len(parsed_rows)
    if n_tasks < min_tasks or n_tasks > max_tasks:
        return None

    for _, parent_original_ids, _ in parsed_rows:
        if any(parent_id not in original_ids for parent_id in parent_original_ids):
            return None

    original_to_temp_id = {
        original_id: index for index, original_id in enumerate(sorted(original_ids))
    }
    temp_rows: dict[int, tuple[int, dict[str, Any]]] = {}
    temp_edge_set: set[tuple[int, int]] = set()

    for original_id, parent_original_ids, row in parsed_rows:
        dst = original_to_temp_id[original_id]
        temp_rows[dst] = (original_id, row)
        for parent_original_id in parent_original_ids:
            src = original_to_temp_id[parent_original_id]
            temp_edge_set.add((src, dst))

    temp_edges = sorted(temp_edge_set)
    if not is_dag(n_tasks, temp_edges):
        return None

    topo_order = topological_sort(n_tasks, temp_edges)
    temp_to_final_id = {
        temp_id: final_id for final_id, temp_id in enumerate(topo_order)
    }

    tasks: list[dict[str, Any]] = []
    for temp_id in topo_order:
        original_id, row = temp_rows[temp_id]
        start_time = row["start_time"]
        end_time = row["end_time"]
        duration = max(1.0, end_time - start_time)
        instance_num = row["instance_num"]
        workload = duration * (row["plan_cpu"] / 100.0) * max(1, instance_num)
        tasks.append(
            {
                "id": temp_to_final_id[temp_id],
                "original_id": original_id,
                "task_name": row["task_name"],
                "instance_num": instance_num,
                "plan_cpu": row["plan_cpu"],
                "plan_mem": row["plan_mem"],
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "workload": workload,
            }
        )

    edges = [
        {
            "src": temp_to_final_id[src],
            "dst": temp_to_final_id[dst],
            "data_size_mb": edge_data_mb,
        }
        for src, dst in temp_edges
    ]
    edges.sort(key=lambda edge: (edge["src"], edge["dst"]))

    trace_start_time = min(task["start_time"] for task in tasks)
    trace_end_time = max(task["end_time"] for task in tasks)
    trace_makespan = max(1.0, trace_end_time - trace_start_time)
    total_workload = sum(task["workload"] for task in tasks)

    return {
        "workflow_id": job_name,
        "n_tasks": len(tasks),
        "n_edges": len(edges),
        "trace_start_time": trace_start_time,
        "trace_end_time": trace_end_time,
        "trace_makespan": trace_makespan,
        "total_workload": total_workload,
        "tasks": tasks,
        "edges": edges,
    }


def collect_rows_by_job(
    input_path: Path,
    chunksize: int,
) -> tuple[dict[str, list[dict[str, Any]]], int, int]:
    rows_by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_rows = 0
    valid_dag_rows = 0

    for chunk in pd.read_csv(
        input_path,
        names=COLUMNS,
        chunksize=chunksize,
        dtype=str,
        on_bad_lines="skip",
    ):
        total_rows += len(chunk)
        valid_mask = chunk["task_name"].map(is_valid_task_name)
        valid_chunk = chunk.loc[
            valid_mask,
            [
                "task_name",
                "instance_num",
                "job_name",
                "start_time",
                "end_time",
                "plan_cpu",
                "plan_mem",
            ],
        ]
        valid_dag_rows += len(valid_chunk)

        for row in valid_chunk.to_dict("records"):
            normalized = _normalize_row(row)
            if normalized is None:
                continue
            rows_by_job[normalized["job_name"]].append(normalized)

    return rows_by_job, total_rows, valid_dag_rows


def write_json(output_path: Path, workflows: list[dict[str, Any]], edge_data_mb: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "Alibaba Cluster Trace v2018 batch_task.csv",
        "description": "DAG workflows constructed from job_name and task_name dependencies.",
        "network": {
            "edge_data_mb": edge_data_mb,
        },
        "num_workflows": len(workflows),
        "workflows": workflows,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_summary(summary_path: Path, workflows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workflow_id",
        "n_tasks",
        "n_edges",
        "trace_start_time",
        "trace_end_time",
        "trace_makespan",
        "total_workload",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for workflow in workflows:
            writer.writerow({field: workflow[field] for field in fieldnames})


def _average(workflows: list[dict[str, Any]], field: str) -> float:
    if not workflows:
        return 0.0
    return sum(float(workflow[field]) for workflow in workflows) / len(workflows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Alibaba Cluster Trace v2018 DAG workflows from batch_task.csv."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--num_workflows", type=int, default=200)
    parser.add_argument("--min_tasks", type=int, default=5)
    parser.add_argument("--max_tasks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=200000)
    parser.add_argument(
        "--edge-data-mb",
        type=float,
        default=1.0,
        help="Data size in MB for each workflow dependency edge",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    rows_by_job, total_rows, valid_dag_rows = collect_rows_by_job(
        input_path=input_path,
        chunksize=args.chunksize,
    )

    workflows = []
    for job_name, rows in rows_by_job.items():
        try:
            workflow = build_workflow(
                job_name=job_name,
                rows=rows,
                min_tasks=args.min_tasks,
                max_tasks=args.max_tasks,
                edge_data_mb=args.edge_data_mb,
            )
        except Exception:
            workflow = None
        if workflow is not None:
            workflows.append(workflow)

    candidate_workflow_count = len(workflows)
    if args.num_workflows >= 0 and len(workflows) > args.num_workflows:
        rng = random.Random(args.seed)
        workflows = rng.sample(workflows, args.num_workflows)
        workflows.sort(key=lambda workflow: workflow["workflow_id"])

    write_json(output_path, workflows, edge_data_mb=args.edge_data_mb)
    write_summary(summary_path, workflows)

    print(f"Total rows scanned: {total_rows}")
    print(f"Valid DAG rows: {valid_dag_rows}")
    print(f"Raw job count: {len(rows_by_job)}")
    print(f"Candidate workflow count: {candidate_workflow_count}")
    print(f"Exported workflow count: {len(workflows)}")
    print(f"Average tasks: {_average(workflows, 'n_tasks'):.6f}")
    print(f"Average edges: {_average(workflows, 'n_edges'):.6f}")
    print(f"Average trace makespan: {_average(workflows, 'trace_makespan'):.6f}")
    print(f"Output JSON: {output_path}")
    print(f"Output summary: {summary_path}")


if __name__ == "__main__":
    main()
