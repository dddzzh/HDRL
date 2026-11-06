"""Generate and audit the paired 100-DAG/111-server pilot; no model or training."""
import argparse
from collections import defaultdict, deque
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.datasets.alibaba_sec_job import load_pool, validate_trace_dag
from src.datasets.sec_job_workflow_generator import shared_fcfs_reference
from src.datasets.workload_mapping import WorkloadMapper
from src.schedulers.heft import _interserver_transfer_time
from src.simulator.resource import compute_execution_time
from src.schedulers.fcfs_scheduler import FCFSScheduler
from tools.hierarchical_common import build_resources, build_workflows, build_env


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {"min": float(a.min()), "median": float(np.median(a)),
            "max": float(a.max()), "mean": float(a.mean())}


def audit_reference(workflows, resources, seed):
    servers = {int(s["id"]): s for s in resources["servers"]}
    links = resources["network"]["link_bandwidth_mbps"]
    responses, timeline = shared_fcfs_reference(workflows, list(servers.values()), links, seed)
    placements = {(w, t): (s, start, end) for w, t, s, start, end in timeline}
    assert len(placements) == len(timeline) == sum(len(w.tasks) for w in workflows)
    by_server = defaultdict(list)
    ready_events = []
    for wi, w in enumerate(workflows):
        ends = []
        for task in w.tasks:
            server, start, end = placements[wi, task.id]
            assert start >= w.arrival_time - 1e-9
            assert math.isclose(end - start, compute_execution_time(task.workload_cycles, servers[server], 1), rel_tol=1e-9, abs_tol=1e-9)
            ready_at = max([w.arrival_time] + [placements[wi, p][2] for p in task.parents])
            ready_events.extend([(ready_at, 0, 1), (start, 1, -1)])
            for parent in task.parents:
                ps, _, pe = placements[wi, parent]
                transfer = _interserver_transfer_time(w.edge_data_mb[parent, task.id], servers[ps], servers[server], links)
                assert start >= pe + transfer - 1e-8
            by_server[server].append((start, end))
            ends.append(end)
        assert math.isclose(max(ends) - w.arrival_time, responses[wi], abs_tol=1e-9)
        assert responses[wi] > 0 and math.isfinite(responses[wi])
        assert w.reference_makespan == responses[wi]
    for slots in by_server.values():
        slots.sort()
        assert all(a[1] <= b[0] + 1e-9 for a, b in zip(slots, slots[1:]))
    ready, peak = 0, 0
    for _, _, delta in sorted(ready_events):
        ready += delta
        peak = max(peak, ready)
    assert ready == 0
    return timeline, peak


def audit_raw_jobs(selected, raw_path):
    """Rebuild only the chosen jobs from the original CSV, not from cached JSON."""
    jobs = {w.workflow_id for w in selected}
    rows = defaultdict(list)
    scanned = 0
    with open(raw_path, encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            scanned += 1
            if len(row) == 9 and row[2] in jobs and re.fullmatch(r"[A-Za-z]\d+(_\d+)*", row[0]):
                parts = [int(p) for p in row[0][1:].split("_")]
                rows[row[2]].append((parts[0], parts[1:], row))
    pool = {str(w["workflow_id"]): w for w in load_pool("outputs/alibaba_test.json")}
    for job in jobs:
        original = {task_id: (parents, row) for task_id, parents, row in rows[job]}
        assert original and len(original) == len(rows[job]), f"Missing/duplicate source tasks: {job}"
        edges = sorted({(p, task_id) for task_id, (parents, _) in original.items() for p in parents})
        assert all(p in original and t in original for p, t in edges)
        indegree = {i: 0 for i in original}
        children = defaultdict(list)
        for parent, child in edges:
            indegree[child] += 1
            children[parent].append(child)
        ready = deque(i for i in sorted(original) if indegree[i] == 0)
        order = []
        while ready:
            parent = ready.popleft()
            order.append(parent)
            for child in children[parent]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        assert len(order) == len(original), f"Original cycle: {job}"
        id_map = {original_id: i for i, original_id in enumerate(order)}
        expected_edges = sorted((id_map[p], id_map[t]) for p, t in edges)
        assert expected_edges == [(e["src"], e["dst"]) for e in pool[job]["edges"]]
        assert len(order) == len(pool[job]["tasks"])
        for task, original_id in zip(pool[job]["tasks"], order):
            _, row = original[original_id]
            duration = max(1.0, float(row[6]) - float(row[5]))
            cpu, memory, instances = float(row[7]), float(row[8]), int(float(row[1]))
            expected = {"id": id_map[original_id], "duration": duration, "plan_cpu": cpu,
                        "plan_mem": memory, "instance_num": instances,
                        "raw_weight": duration * cpu / 100 * max(1, instances)}
            assert all(math.isclose(task[k], value, rel_tol=1e-10, abs_tol=1e-10)
                       for k, value in expected.items()), f"Source mismatch for {job}"
    return {"original_csv_rows_scanned": scanned, "reconstructed_jobs_matched": len(jobs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/alibaba_sl_111_sec_job_100wf.json")
    parser.add_argument("--output-dir", default="results/alibaba_sl111_generation_preflight_20261015")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    output = (ROOT / args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite preflight artifacts: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["evaluation"]["preflight_seed"])
    input_paths = [config_path, ROOT / config["generator"]["alibaba_path"],
                   ROOT / config["generator"]["mapping_calibration_path"],
                   ROOT / config["resources"]["path"],
                   ROOT / "src/datasets/alibaba_sec_job.py",
                   ROOT / "src/datasets/sec_job_workflow_generator.py",
                   ROOT / "src/datasets/workload_mapping.py",
                   ROOT / "src/datasets/workflow_factory.py",
                   Path(__file__).resolve(),
                   ROOT / "dataset/cluster-trace-v2018/batch_task.csv"]
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in input_paths}
    print("[preflight] source hashes frozen; checking both pools", flush=True)
    pool = load_pool(config["generator"]["alibaba_path"])
    calibration = load_pool(config["generator"]["mapping_calibration_path"])
    for w in pool + calibration:
        validate_trace_dag(w)
    assert not ({w["workflow_id"] for w in pool} & {w["workflow_id"] for w in calibration})
    mapper = WorkloadMapper(calibration, config["generator"])
    with contextlib.redirect_stdout(io.StringIO()):
        resources = build_resources(config, config["evaluation"]["resource_seed"])
    assert len(resources["servers"]) == 111
    assert sum(s["layer"] == "edge" for s in resources["servers"]) == 84
    batches, summaries = {}, {}
    expected_arrivals = np.cumsum(np.random.default_rng(seed).exponential(1 / 0.03, 100))
    for deadline in config["evaluation"]["deadline_types"]:
        cfg = copy.deepcopy(config)
        cfg["simulation"]["deadline_type"] = deadline
        workflows = build_workflows(cfg, resources, seed)
        assert len(workflows) == len({w.workflow_id for w in workflows}) == 100
        assert all(w.arrival_time == expected_arrivals[i] for i, w in enumerate(workflows))
        for w in workflows:
            assert all(0.5e9 <= t.workload_cycles <= 20e9 for t in w.tasks)
            assert all(1 <= e.data_size_mb <= 30 for e in w.edges)
            assert w.reliability == 0.9
            assert w.relative_deadline == w.reference_makespan * w.deadline_multiplier
            assert w.absolute_deadline == w.arrival_time + w.relative_deadline
            assert w.absolute_deadline > w.arrival_time
            assert w.deadline_reference == "shared_fcfs_first_idle_rho1"
        timeline, peak_ready = audit_reference(workflows, resources, seed)
        # Check the actual environment accepts and preserves the generated constraints.
        with contextlib.redirect_stdout(io.StringIO()):
            env = build_env(workflows, resources, cfg["simulation"], seed)
        assert [s["absolute_deadline"] for s in env.workflow_states] == [w.absolute_deadline for w in workflows]
        with contextlib.redirect_stdout(io.StringIO()):
            smoke = env.run(FCFSScheduler(seed=seed))
        assert env.done and smoke["summary"]["total_finished_workflows"] == 100
        assert all(status == "finished" for s in env.workflow_states for status in s["task_status"].values())
        batches[deadline] = workflows
        summaries[deadline] = {
            "workflows": len(workflows), "tasks": sum(len(w.tasks) for w in workflows),
            "edges": sum(len(w.edges) for w in workflows),
            "multiplier_2_count": sum(w.deadline_multiplier == 2 for w in workflows),
            "multiplier_3_count": sum(w.deadline_multiplier == 3 for w in workflows),
            "relative_deadline_seconds": distribution([w.relative_deadline for w in workflows]),
            "reference_response_seconds": distribution([w.reference_makespan for w in workflows]),
            "reference_timeline_tasks_checked": len(timeline),
            "reference_peak_ready_including_same_time_releases": peak_ready,
            "environment_deadlines_preserved": True,
            "environment_fcfs_smoke_finished_workflows": smoke["summary"]["total_finished_workflows"],
        }
        print(f"[preflight] {deadline}: {len(workflows)} DAGs; DDL and timeline passed", flush=True)
    for index, triple in enumerate(zip(*(batches[d] for d in ("loose", "medium", "tight")))):
        first = triple[0]
        signature = lambda w: (w.workflow_id, w.arrival_time,
                               [(t.id, t.workload_cycles) for t in w.tasks],
                               [(e.src, e.dst, e.data_size_mb) for e in w.edges], w.reference_makespan)
        assert all(signature(w) == signature(first) for w in triple)
        assert triple[0].absolute_deadline >= triple[1].absolute_deadline >= triple[2].absolute_deadline
    repeated = build_workflows({**config, "simulation": {**config["simulation"], "deadline_type": "loose"}}, resources, seed)
    assert [w.to_mapping() for w in repeated] == [w.to_mapping() for w in batches["loose"]]
    scientific_config = copy.deepcopy(config)
    scientific_config["generator"]["source"] = "pegasus"
    scientific = build_workflows(scientific_config, resources, seed)
    assert len(scientific) == 100 and all(w.workflow_id.startswith("pegasus_") for w in scientific)
    assert all(w.arrival_time == expected_arrivals[i] for i, w in enumerate(scientific))
    with contextlib.redirect_stdout(io.StringIO()):
        scientific_smoke = build_env(scientific, resources, scientific_config["simulation"], seed).run(FCFSScheduler(seed=seed))
    assert scientific_smoke["summary"]["total_finished_workflows"] == 100
    print("[preflight] paired equality and reproducibility passed; reconstructing original CSV jobs", flush=True)
    raw_check = audit_raw_jobs(batches["loose"], input_paths[-1])
    assert all(sha256(ROOT / p) == value for p, value in source_hashes.items())
    summary = {
        "status": "generation_preflight_passed_not_model_evaluation", "seed": seed,
        "resource_seed": config["evaluation"]["resource_seed"], "servers": 111,
        "edge": 84, "cloud": 27, "test_pool_jobs": len(pool),
        "calibration_pool_jobs": len(calibration), "source_sha256": source_hashes,
        "calibration_log1p_weight_p5": mapper.z_p5, "calibration_log1p_weight_p95": mapper.z_p95,
        "task_count_per_job": distribution([len(w.tasks) for w in batches["loose"]]),
        "task_workload_gc": distribution([t.workload_gc for w in batches["loose"] for t in w.tasks]),
        "arrival_window_seconds": [float(expected_arrivals[0]), float(expected_arrivals[-1])],
        "paired_physical_workloads_equal": True, "nested_deadlines": True,
        "same_seed_reproducible": True, "train_test_job_disjoint": True,
        "pool_dags_validated": len(pool) + len(calibration),
        "single_field_source_switch_scientific_finished_workflows": 100,
        "raw_reconstruction": raw_check, "scenarios": summaries, "config": config,
    }
    output.mkdir(parents=True, exist_ok=True)
    for deadline, workflows in batches.items():
        data = [{**w.to_mapping(), "source_workflow_id": w.source_workflow_id,
                 "deadline_reference": w.deadline_reference} for w in workflows]
        for row in data:
            row.pop("edge_data_mb", None)  # Derived tuple-key lookup; edges already preserve the data.
        (output / f"{deadline}_workflows.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Alibaba / 111 servers / 100-workflow generation preflight", "",
             "No model inference or training. Same 100 real test job IDs across all three deadlines.",
             "Trace topology/intensity are real; cycles, edge data, arrivals and constraints are simulated.",
             "", "| Deadline | Workflows | Tasks | Edges | 2x | 3x | Median relative DDL (s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for deadline, s in summaries.items():
        lines.append(f"| {deadline} | {s['workflows']} | {s['tasks']} | {s['edges']} | {s['multiplier_2_count']} | {s['multiplier_3_count']} | {s['relative_deadline_seconds']['median']:.6f} |")
    lines += ["", "All pool DAGs/weights validated; calibration/test job IDs disjoint; selected jobs reconstructed from raw CSV.",
              "Reference timeline checked for arrival, dependency+transfer, cycles/Hz duration, server non-overlap and response-based DDL.",
              "Detailed frozen sources, configuration, numeric checks and caveats: audit_summary.json."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "status": summary["status"], "scenarios": summaries}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
