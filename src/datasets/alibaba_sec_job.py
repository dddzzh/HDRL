"""Trace DAGs with frozen train-pool workload calibration and current deadlines.

Only the topology and relative compute intensity are trace-derived. Cycles,
communication, arrivals, reliability targets and deadlines are simulation inputs.
"""
import copy
from functools import lru_cache
from graphlib import TopologicalSorter
import json
import math
from pathlib import Path
import random

import numpy as np

from src.datasets.arrival_process import resolve_arrival_rate
from src.datasets.sec_job_workflow_generator import assign_shared_fcfs_deadlines
from src.datasets.workload_mapping import WorkloadMapper, compute_raw_weight
from src.simulator.workflow import Workflow

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=2)
def _read_pool(path, mtime_ns, size):
    # File identity is part of the key; never cache across an edited source file.
    with open(path, encoding="utf-8") as handle:
        pool = json.load(handle)["workflows"]
    ids = [str(w["workflow_id"]) for w in pool]
    if not pool or len(set(ids)) != len(ids):
        raise ValueError("Alibaba pool must have nonempty, unique job IDs")
    return pool


def load_pool(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    stat = path.stat()
    return _read_pool(str(path), stat.st_mtime_ns, stat.st_size)


def validate_trace_dag(workflow):
    tasks = workflow["tasks"]
    ids = [int(t["id"]) for t in tasks]
    if not tasks or len(set(ids)) != len(ids):
        raise ValueError("Trace DAG must have nonempty, unique task IDs")
    predecessors = {i: set() for i in ids}
    edges = set()
    for edge in workflow["edges"]:
        src, dst = int(edge["src"]), int(edge["dst"])
        if src not in predecessors or dst not in predecessors or (src, dst) in edges:
            raise ValueError("Invalid or duplicate trace dependency")
        edges.add((src, dst))
        predecessors[dst].add(src)
    list(TopologicalSorter(predecessors).static_order())
    for task in tasks:
        weight = compute_raw_weight(task)
        expected = float(task["duration"]) * float(task["plan_cpu"]) / 100 * max(1, int(task["instance_num"]))
        if not math.isfinite(weight) or weight <= 0 or not math.isclose(weight, expected, rel_tol=1e-10):
            raise ValueError("Invalid trace relative-work weight (not cycles)")


def generate_alibaba_sec_job(config, seed, resources):
    if resources is None or not resources.get("servers"):
        raise ValueError("Alibaba sec-job generation requires actual resources")
    if config.get("arrival_process") != "sec_job_exponential":
        raise ValueError("Alibaba sec-job profile requires sec_job_exponential arrivals")
    pool = load_pool(config.get("alibaba_path", config.get("path", "outputs/alibaba_test.json")))
    calibration = load_pool(config["mapping_calibration_path"])
    if {str(w["workflow_id"]) for w in pool} & {str(w["workflow_id"]) for w in calibration}:
        raise ValueError("Calibration and test job IDs must be disjoint")
    count = int(config["num_workflows"])
    if count <= 0 or count > len(pool):
        raise ValueError("num_workflows must be positive and not exceed test pool size")
    if "max_workflows" in config and int(config["max_workflows"]) != count:
        raise ValueError("Conflicting num_workflows and max_workflows")
    mapper = WorkloadMapper(calibration, config)
    if not mapper.has_spread:
        raise ValueError("Calibration pool has no positive workload spread")
    bounds = mapper.cfg
    if not (0 < bounds.workload_gc_min <= bounds.workload_gc_max < float("inf")
            and 0 < bounds.jitter_min <= bounds.jitter_max < float("inf")
            and 0 <= bounds.data_size_mb_min <= bounds.data_size_mb_max < float("inf")):
        raise ValueError("Invalid workload, jitter or communication mapping bounds")
    rng = random.Random(seed)
    selected = rng.sample(pool, count)
    rate = resolve_arrival_rate(config)
    start = float(config.get("start_time", 0.0))
    if not math.isfinite(rate) or not math.isfinite(start) or start < 0:
        raise ValueError("Arrival rate and start must be finite; start must be nonnegative")
    arrivals = start + np.cumsum(np.random.default_rng(seed).exponential(1 / rate, count))
    reliability = float(config.get("workflow_reliability_requirement", 0.9))
    if not 0 < reliability <= 1:
        raise ValueError("Workflow reliability requirement must be in (0, 1]")
    workflows = []
    for index, (template, arrival) in enumerate(zip(selected, arrivals)):
        validate_trace_dag(template)
        mapped = mapper.apply(copy.deepcopy(template), random.Random(seed + index))
        mapped["arrival_time"], mapped["reliability"] = float(arrival), reliability
        # Ignore any stale constraints from a previously processed input pool.
        for field in ("deadline_type", "reference_makespan", "deadline_multiplier",
                      "relative_deadline", "absolute_deadline", "energy_budget"):
            mapped.pop(field, None)
        workflow = Workflow.from_mapping(mapped)
        workflow.source_workflow_id = str(template["workflow_id"])
        workflow.workflow_type = "alibaba_v2018"
        workflow.arrival_index, workflow.source_device_id = index, 0
        workflow.arrival_process, workflow.arrival_rate = "sec_job_exponential", rate
        workflow.inter_arrival_time = float(arrival - (arrivals[index - 1] if index else start))
        workflows.append(workflow)
    return assign_shared_fcfs_deadlines(
        workflows, resources["servers"], resources["network"]["link_bandwidth_mbps"],
        seed, config.get("deadline_type", "medium"))
