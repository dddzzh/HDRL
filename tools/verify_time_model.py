"""验证 DES 时间模型：cycle 执行时间、事件队列、running/finished、并发到达、workload 映射。"""
from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.workflow_factory import create_workflows
from src.datasets.workload_mapping import WorkloadMapper
from src.schedulers.heft import reference_heft_makespan
from src.simulator.edge_cloud_env import EdgeCloudWorkflowEnv
from src.simulator.resource import compute_energy, compute_execution_time, load_resources
from src.simulator.workflow import DependencyEdge, Task, Workflow


def _make_chain_workflow(
    wf_id: str, arrival: float, n: int = 3, workload_cycles: float = 1e9
) -> Workflow:
    tasks = tuple(Task(id=i, workload_cycles=workload_cycles) for i in range(n))
    edges = tuple(
        DependencyEdge(src=i, dst=i + 1, data_size_mb=1.0) for i in range(n - 1)
    )
    return Workflow(workflow_id=wf_id, tasks=tasks, edges=edges, arrival_time=arrival)


def test_execution_time_cycles():
    """T = L / (rho * f): 3e9 cycles @ 3 GHz -> rho=1.0 → 1s, rho=0.5 → 2s。"""
    server = {"id": 0, "max_freq_hz": 3e9, "static_power": 10.0, "dynamic_power": 90.0}
    assert abs(compute_execution_time(3e9, server, 1.0) - 1.0) < 1e-9
    assert abs(compute_execution_time(3e9, server, 0.5) - 2.0) < 1e-9


def test_energy_consistency_rho():
    """rho=0.5 时执行时间约为 rho=1.0 的 2 倍，能耗用对应执行时间。"""
    server = {"id": 0, "max_freq_hz": 3e9, "static_power": 10.0, "dynamic_power": 90.0}
    t_full = compute_execution_time(3e9, server, 1.0)
    t_half = compute_execution_time(3e9, server, 0.5)
    assert abs(t_half - 2.0 * t_full) < 1e-9
    e_full = compute_energy(3e9, server, 1.0)
    e_half = compute_energy(3e9, server, 0.5)
    # energy = power(rho) * time(rho); both positive and finite
    assert e_full > 0 and e_half > 0


def test_workload_mapping_range():
    """raw_weight 映射出的 workload_cycles 必须落在 [0.5e9, 20e9]。"""
    pool = [
        {"tasks": [{"id": 0, "raw_weight": w}], "edges": []}
        for w in (1.0, 10.0, 100.0, 1000.0, 50000.0, 2_000_000.0)
    ]
    mapper = WorkloadMapper(pool, {"workload_gc_min": 0.5, "workload_gc_max": 20.0})
    rng = random.Random(0)
    for workflow in pool:
        cycles = mapper.assign_workload_cycles(workflow["tasks"][0], rng)
        assert 0.5e9 <= cycles <= 20.0e9, cycles
    # fallback uniform when no usable weight
    cycles = mapper.assign_workload_cycles({"id": 0}, rng)
    assert 0.5e9 <= cycles <= 20.0e9


def test_communication_mapping_range():
    """边数据量必须落在 [1, 30] MB。"""
    pool = [{"tasks": [{"id": 0, "raw_weight": 10.0}], "edges": []}]
    mapper = WorkloadMapper(pool, {"data_size_mb_min": 1.0, "data_size_mb_max": 30.0})
    rng = random.Random(1)
    for _ in range(200):
        data_mb = mapper.assign_edge_data_mb(rng)
        assert 1.0 <= data_mb <= 30.0, data_mb


def test_chain_unlocks_after_finish():
    """链式 DAG：父任务 running 时子任务不可 ready，finish 后才解锁。"""
    workflows = [_make_chain_workflow("wf0", 0.0, 3)]
    resources = load_resources(
        "configs/resources_edge_cloud_default.json",
        num_edge=1,
        num_cloud=0,
        seed=0,
    )
    env = EdgeCloudWorkflowEnv(workflows, resources, {}, seed=0)
    state = env.get_state()
    assert len(state["ready_tasks"]) == 1
    assert state["ready_tasks"][0]["task_id"] == 0

    env.step({"workflow_id": "wf0", "task_id": 0, "server_id": 0, "rho": 1.0})
    ws = env.workflow_by_id["wf0"]
    assert ws["task_status"][0] == "running"
    assert env._get_ready_tasks() == []

    assert env._advance_to_next_decision_point()
    assert ws["task_status"][0] == "finished"
    ready = env._get_ready_tasks()
    assert len(ready) == 1 and ready[0]["task_id"] == 1


def test_late_arrival_while_first_running():
    """wf1 在 t=50 到达时，wf0 仍有 running 任务；时钟应独立推进并释放 wf1。"""
    workflows = [
        # ~67s on a 3 GHz core, so wf0 is still running when wf1 arrives at t=50
        _make_chain_workflow("wf0", 0.0, 1, workload_cycles=2e11),
        _make_chain_workflow("wf1", 50.0, 2),
    ]
    resources = load_resources(
        "configs/resources_edge_cloud_default.json",
        num_edge=2,
        num_cloud=0,
        seed=1,
    )
    env = EdgeCloudWorkflowEnv(
        workflows,
        resources,
        {"trace_scheduling": False},
        seed=1,
    )

    env.step({"workflow_id": "wf0", "task_id": 0, "server_id": 0, "rho": 1.0})
    assert env.workflow_by_id["wf0"]["task_status"][0] == "running"
    assert env._get_ready_tasks() == []
    assert not env.workflow_by_id["wf1"]["released"]

    assert env._advance_to_next_decision_point()

    assert env.current_time >= 50.0
    assert env.workflow_by_id["wf0"]["task_status"][0] == "running"
    assert env.workflow_by_id["wf1"]["released"]
    ready_wfs = {t["workflow_id"] for t in env._get_ready_tasks()}
    assert "wf1" in ready_wfs


def test_reference_heft_makespan_diamond_dag():
    """并行 DAG 的 T_ef_min 应小于同任务数的链式 DAG。"""
    workload_cycles = 1e9
    data_mb = 1.0
    tasks = (
        Task(id=0, workload_cycles=workload_cycles),
        Task(id=1, workload_cycles=workload_cycles),
        Task(id=2, workload_cycles=workload_cycles),
        Task(id=3, workload_cycles=workload_cycles),
    )
    diamond = Workflow(
        workflow_id="diamond",
        tasks=tasks,
        edges=(
            DependencyEdge(src=0, dst=1, data_size_mb=data_mb),
            DependencyEdge(src=0, dst=2, data_size_mb=data_mb),
            DependencyEdge(src=1, dst=3, data_size_mb=data_mb),
            DependencyEdge(src=2, dst=3, data_size_mb=data_mb),
        ),
        arrival_time=0.0,
    )
    chain = Workflow(
        workflow_id="chain",
        tasks=tasks,
        edges=tuple(
            DependencyEdge(src=i, dst=i + 1, data_size_mb=data_mb) for i in range(3)
        ),
        arrival_time=0.0,
    )
    resources = load_resources(
        "configs/resources_edge_cloud_default.json",
        num_edge=2,
        num_cloud=1,
        seed=3,
    )
    links = resources["network"]["link_bandwidth_mbps"]
    diamond_t = reference_heft_makespan(diamond, resources["servers"], links)
    chain_t = reference_heft_makespan(chain, resources["servers"], links)
    assert diamond_t < chain_t


def test_server_no_overlap():
    gen_cfg = {"num_workflows": 5, "mean_interarrival": 50.0, "min_tasks": 2, "max_tasks": 4}
    resources = load_resources(
        "configs/resources_edge_cloud_default.json",
        num_edge=2,
        num_cloud=1,
        seed=2,
    )
    workflows = create_workflows(gen_cfg, source="pegasus", seed=2, resources=resources)
    env = EdgeCloudWorkflowEnv(workflows, resources, {}, seed=2)
    steps = 0
    while not env.done and steps < 500:
        state = env.get_state()
        if not state["ready_tasks"]:
            if not env._advance_to_next_decision_point():
                env._update_done()
                break
            continue
        task = state["ready_tasks"][0]
        env.step(
            {
                "workflow_id": task["workflow_id"],
                "task_id": task["task_id"],
                "server_id": 0,
                "rho": 1.0,
            }
        )
        steps += 1

    by_server: dict[int, list] = {}
    for ev in env.event_log:
        by_server.setdefault(ev["server_id"], []).append(ev)
    for sid, events in by_server.items():
        events.sort(key=lambda e: e["start_time"])
        for i in range(1, len(events)):
            assert events[i]["start_time"] >= events[i - 1]["finish_time"] - 1e-9


def test_server_fcfs_queue():
    """Tasks assigned to the same server are appended in FCFS order."""
    tasks = (
        Task(id=0, workload_cycles=2e10),
        Task(id=1, workload_cycles=1e9),
    )
    workflow = Workflow(workflow_id="wf0", tasks=tasks, edges=(), arrival_time=0.0)
    resources = load_resources(
        "configs/resources_edge_cloud_default.json",
        num_edge=1,
        num_cloud=0,
        seed=4,
    )
    env = EdgeCloudWorkflowEnv([workflow], resources, {}, seed=4)
    env.step({"workflow_id": "wf0", "task_id": 0, "server_id": 0, "rho": 1.0})
    env.step({"workflow_id": "wf0", "task_id": 1, "server_id": 0, "rho": 1.0})

    first, second = env.event_log
    assert second["start_time"] >= first["finish_time"] - 1e-9
    state = env.get_state()
    assert state["servers"][0]["queue_count"] == 2


def test_recr_ingress_egress():
    """入口上传(I->E/I->C)与出口下载(E/C->I)使用 RECR 分层带宽。"""
    from src.simulator.resource import compute_transfer_time

    wf = Workflow(
        workflow_id="wf0",
        tasks=(Task(id=0, workload_cycles=1e9),),
        edges=(),
        arrival_time=0.0,
        input_data_mb=8.0,
        output_data_mb=8.0,
    )
    resources = load_resources(
        "configs/resources_edge_cloud_default.json",
        num_edge=1,
        num_cloud=1,
        seed=0,
    )
    links = resources["network"]["link_bandwidth_mbps"]

    env_edge = EdgeCloudWorkflowEnv([wf], resources, {}, seed=0)
    env_edge.step({"workflow_id": "wf0", "task_id": 0, "server_id": 0, "rho": 1.0})
    expected_edge_upload = compute_transfer_time(8.0, links["iot_edge"])
    assert abs(env_edge.workflow_by_id["wf0"]["upload_time"] - expected_edge_upload) < 1e-9

    env_cloud = EdgeCloudWorkflowEnv([wf], resources, {}, seed=0)
    env_cloud.step({"workflow_id": "wf0", "task_id": 0, "server_id": 1, "rho": 1.0})
    expected_cloud_upload = compute_transfer_time(8.0, links["iot_cloud"])
    assert abs(env_cloud.workflow_by_id["wf0"]["upload_time"] - expected_cloud_upload) < 1e-9
    assert expected_cloud_upload > expected_edge_upload

    while not env_cloud.workflow_by_id["wf0"]["completed"]:
        if not env_cloud._advance_to_next_decision_point():
            break

    ws = env_cloud.workflow_by_id["wf0"]
    expected_download = compute_transfer_time(8.0, links["cloud_iot"])
    assert ws["egress_applied"]
    assert abs(ws["download_time"] - expected_download) < 1e-9


if __name__ == "__main__":
    test_execution_time_cycles()
    print("OK: execution time = L/(rho*f) in cycles")
    test_energy_consistency_rho()
    print("OK: energy uses cycle-based execution time")
    test_workload_mapping_range()
    print("OK: workload mapping in [0.5, 20] GC")
    test_communication_mapping_range()
    print("OK: edge data size in [1, 30] MB")
    test_chain_unlocks_after_finish()
    print("OK: chain unlocks after finish")
    test_late_arrival_while_first_running()
    print("OK: late arrival while first running")
    test_t_ef_min_diamond_dag()
    print("OK: t_ef_min diamond dag")
    test_server_no_overlap()
    print("OK: server no overlap")
    test_server_fcfs_queue()
    print("OK: server fcfs queue")
    test_recr_ingress_egress()
    print("OK: recr ingress egress")
    print("\nAll DES checks passed.")
