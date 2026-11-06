"""Sec-job sampling protocol adapted to DRLEC's cycles/Hz and DAG-link model.

This is not a reproduction of its MI/MIPS, external-file I/O or host power model.
Deadlines use one shared-load FCFS/first-idle reference run, not isolated HEFT.
"""
from functools import lru_cache
import heapq
import random

import numpy as np

from src.datasets.pegasus_workflow_generator import PegasusWorkflowGenerator, PEGASUS_XML_FILES
from src.schedulers.heft import _interserver_transfer_time
from src.simulator.deadline import DEADLINE_PROBABILITIES, validate_deadline_type
from src.simulator.resource import compute_execution_time

GROUPS = tuple(PEGASUS_XML_FILES[i:i + 5] for i in (0, 5, 10))
GROUP_WEIGHTS = {"small": (1, 0, 0), "middle": (3, 7, 0), "large": (2, 3, 5)}


def shared_fcfs_reference(workflows, servers, links, seed):
    specs = tuple((w.arrival_time,
                   tuple((t.id, t.workload_cycles, t.parents, t.children) for t in w.tasks),
                   tuple((e.src, e.dst, e.data_size_mb) for e in w.edges)) for w in workflows)
    server_specs = tuple((int(s["id"]), float(s["max_freq_hz"]), str(s["layer"]),
                          float(s.get("bandwidth_mbps", 100))) for s in servers)
    return _shared_fcfs_reference(specs, server_specs, tuple(sorted((links or {}).items())), int(seed))


@lru_cache(maxsize=64)
def _shared_fcfs_reference(specs, server_specs, link_specs, seed):
    """Event simulation; only idle servers receive tasks, FCFS by dependency-ready time."""
    servers = {i: {"id": i, "max_freq_hz": f, "layer": layer, "bandwidth_mbps": bw}
               for i, f, layer, bw in server_specs}
    if not specs or not servers:
        raise ValueError("FCFS reference needs workflows and servers")
    tasks = {(w, tid): (cycles, parents, children) for w, (_, ts, _) in enumerate(specs)
             for tid, cycles, parents, children in ts}
    edge_data = {(w, a, b): d for w, (_, _, edges) in enumerate(specs) for a, b, d in edges}
    events = [(float(arrival), 0, w, -1) for w, (arrival, _, _) in enumerate(specs)]
    heapq.heapify(events)
    available = {i: 0.0 for i in servers}
    ready, completed, finish, placement, ready_time = [], set(), {}, {}, {}
    timeline = []
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    while events:
        now = events[0][0]
        while events and events[0][0] <= now:
            time, kind, w, tid = heapq.heappop(events)
            if kind == 0:
                candidates = [i for i, _, parents, _ in specs[w][1] if not parents]
            else:
                completed.add((w, tid))
                candidates = [i for i in tasks[w, tid][2]
                              if all((w, p) in completed for p in tasks[w, i][1])]
            for i in candidates:
                key = (w, i)
                assert key not in ready_time, "Task became ready twice"
                ready_time[key] = time
                ready.append(key)
        rng.shuffle(ready)
        ready.sort(key=ready_time.get)
        while ready:
            idle = [i for i in sorted(servers) if available[i] <= now]
            if not idle:
                break
            earliest = min(available[i] for i in idle)
            chosen = int(rng.choice([i for i in idle if abs(available[i] - earliest) <= 1e-12]))
            w, tid = ready.pop(0)
            cycles, parents, _ = tasks[w, tid]
            if cycles <= 0:
                raise ValueError("Task cycles must be positive")
            data_ready = max((finish[w, p] + _interserver_transfer_time(
                edge_data[w, p, tid], servers[placement[w, p]], servers[chosen], dict(link_specs))
                for p in parents), default=specs[w][0])
            start = max(now, data_ready)
            end = start + compute_execution_time(cycles, servers[chosen], 1.0)
            finish[w, tid], placement[w, tid], available[chosen] = end, chosen, end
            timeline.append((w, tid, chosen, start, end))
            heapq.heappush(events, (end, 1, w, tid))
    if len(completed) != len(tasks):
        raise ValueError("FCFS reference did not complete every task; check DAG")
    responses = tuple(max(finish[w, tid] for tid, *_ in ts) - arrival
                      for w, (arrival, ts, _) in enumerate(specs))
    return responses, tuple(timeline)


def assign_shared_fcfs_deadlines(workflows, servers, links, seed, deadline_type):
    """Assign the same protocol to Pegasus and explicitly opted-in trace DAGs."""
    deadline_type = validate_deadline_type(deadline_type)
    responses, _ = shared_fcfs_reference(workflows, servers, links, seed)
    for index, (w, reference) in enumerate(zip(workflows, responses)):
        local_seed = (seed * 1000003 + index * 9176 + 20260316) & 0xFFFFFFFF
        multiplier = 2.0 if np.random.RandomState(local_seed).rand() < DEADLINE_PROBABILITIES[deadline_type] else 3.0
        PegasusWorkflowGenerator._set_deadline_fields(w, {"reference_makespan": reference,
            "deadline_multiplier": multiplier, "relative_deadline": multiplier * reference,
            "absolute_deadline": w.arrival_time + multiplier * reference,
            "deadline_type": deadline_type})
        w.deadline_reference = "shared_fcfs_first_idle_rho1"
    return workflows


class SecJobWorkflowGenerator(PegasusWorkflowGenerator):
    def generate(self, servers=None, link_bandwidth_mbps=None):
        if servers is None:
            raise ValueError("Sec-job shared reference requires actual resources")
        if self.dataset_scale not in GROUP_WEIGHTS:
            raise ValueError("Sec-job scale must be small, middle or large")
        if self.num_workflows <= 0:
            raise ValueError("num_workflows must be positive")
        if self.arrival_process != "sec_job_exponential":
            raise ValueError("Sec-job profile requires sec_job_exponential arrivals")
        arrivals = self.start_time + np.cumsum(np.random.default_rng(self.seed).exponential(
            1 / self.arrival_rate, self.num_workflows))
        templates = [name for group, weight in zip(GROUPS, GROUP_WEIGHTS[self.dataset_scale])
                     for _ in range(weight) for name in group]
        template_rng = np.random.RandomState(self.seed & 0xFFFFFFFF)
        workflows = []
        for index, arrival in enumerate(arrivals):
            name = templates[int(template_rng.randint(len(templates)))]
            w = self._workflow_from_xml(self.data_dir / name, index=index,
                                       arrival_time=float(arrival), rng=random.Random(self.seed + index))
            w.template_name, w.dataset_scale = name, self.dataset_scale
            w.arrival_index, w.source_device_id = index, 0
            w.arrival_process, w.arrival_rate = self.arrival_process, self.arrival_rate
            w.inter_arrival_time = float(arrival - (arrivals[index - 1] if index else self.start_time))
            workflows.append(w)
        return assign_shared_fcfs_deadlines(
            workflows, servers, link_bandwidth_mbps, self.seed, self.deadline_type)
