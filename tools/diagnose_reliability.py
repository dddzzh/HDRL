"""Audit original reliability evidence and replay one frozen 20-seed diagnosis."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import math
import multiprocessing
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/reliability_diagnostic_20261015"
PANEL = ROOT / "results/success_seed_screening_20261013/panel_plan.json"
SEEDS = list(range(742000, 742020))
GRID = [1.0, .9, .8, .7, .6, .5]
_runtime = None


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def describe(values):
    import numpy as np
    a = [float(v) for v in values]
    if not a:
        return {"n": 0}
    if not all(math.isfinite(v) for v in a):
        raise ValueError("Nonfinite observation")
    return dict(n=len(a), mean=statistics.mean(a), min=min(a),
                p01=float(np.quantile(a, .01)), p05=float(np.quantile(a, .05)),
                median=statistics.median(a), max=max(a))


def audit():
    """Only original unselected records; no inference, training or model edits."""
    panel = read(PANEL)
    source_hashes = {relative(PANEL): sha(PANEL)}
    records, training = [], []
    for run in panel["runs"]:
        path = Path(run["source"]) / "comparison_rows.jsonl"
        expected = panel["original_file_sha256"][str(path)]
        if sha(path) != expected:
            raise ValueError(f"Original data changed: {path}")
        source_hashes[relative(path)] = expected
        data = list(rows(path))
        keys = [(r["scale"], r["deadline_type"], r["eval_seed"]) for r in data]
        expected_keys = {(s, d, n) for s in ["small", "medium", "large"]
                         for d in ["loose", "medium", "tight"] for n in SEEDS}
        if len(data) != 180 or len(set(keys)) != 180 or set(keys) != expected_keys:
            raise ValueError("Original panel incomplete")
        groups = defaultdict(list)
        for row in data:
            if row["mode"] != run["mode"] or row["num_workflows"] != 50:
                raise ValueError("Original protocol differs")
            for key in ("avg_workflow_reliability", "reliability_success_rate", "rho_mean"):
                if not 0 <= float(row[key]) <= 1:
                    raise ValueError("Invalid reliability or rho")
            groups[row["scale"], row["deadline_type"]].append(row)
        for (scale, deadline), group in groups.items():
            records.append(dict(run=run["id"], scale=scale, deadline_type=deadline,
                                batch_average_R=describe(r["avg_workflow_reliability"] for r in group),
                                batch_average_rho=describe(r["rho_mean"] for r in group),
                                batches_all_reliable=sum(r["reliability_success_rate"] >= 1-1e-9 for r in group),
                                mean_workflow_attainment=statistics.mean(r["reliability_success_rate"] for r in group)))
        if run["mode"] != "d3qn_sac":
            continue
        histories = [Path(run["checkpoint"]).parent / "train_history.jsonl"]
        if run["resource"] == "sl":
            histories += sorted((ROOT / "results/v3_sl_111_legacy_train_20wf_resume_ep1050").glob("*/train_history.jsonl"))
        for history in histories:
            if not history.exists():
                continue
            source_hashes[relative(history)] = sha(history)
            data = list(rows(history))
            for scope, sample in [("all_logged_training", data),
                                  ("through_selected_checkpoint", [r for r in data if int(r.get("global_episode", r["episode"])) <= run["source_episode"]])]:
                trace = [{k: r[k] for k in ("episode", "global_episode", "rho_mean", "avg_workflow_reliability",
                          "reliability_success_rate", "low_reliability_lambda", "low_reliability_constraint_batch_total",
                          "low_reliability_lagrangian_penalty_mean", "high_reliability_cost_total",
                          "high_lambda_reliability", "mean_low_sac_reliability_violation") if k in r} for r in sample]
                summary = dict(run=run["id"], history=relative(history), scope=scope, episodes=len(sample), trace=trace)
                for key in ("rho_mean", "avg_workflow_reliability", "reliability_success_rate", "low_reliability_lambda",
                            "low_reliability_lagrangian_penalty_mean", "high_reliability_cost_total",
                            "mean_low_sac_reliability_violation", "low_reliability_constraint_batch_total"):
                    values = [r[key] for r in sample if r.get(key) is not None]
                    summary[key] = {**describe(values), "positive_episodes": sum(float(v)>1e-12 for v in values)}
                summary["episodes_with_reliability_failure"] = sum(float(r["reliability_success_rate"]) < 1-1e-9 for r in sample)
                training.append(summary)
    write(OUT / "existing_evidence.json", dict(original_records=2160, groups=records, training=training,
                                               source_sha256=source_hashes))
    return panel, source_hashes


def prepare():
    if OUT.exists():
        raise ValueError("Diagnosis directory exists; do not prepare or overwrite twice")
    OUT.mkdir()
    panel, hashes = audit()
    run = next(r for r in panel["runs"] if r["id"] == "sm_74_d3qn_sac")
    for key in ["checkpoint", "config"]:
        path = Path(run[key])
        if sha(path) != run[key + "_sha256"]:
            raise ValueError("Frozen model/config changed")
        hashes[relative(path)] = sha(path)
    paths = ["tools/diagnose_reliability.py", "docs/RELIABILITY_DIAGNOSTIC_20261015.md",
             "tools/eval_all_baselines.py", "tools/train_hierarchical_d3qn.py",
             "tools/hierarchical_common.py", "src/simulator/reliability.py",
             "src/simulator/resource.py", "src/simulator/edge_cloud_env.py",
             "src/simulator/metrics.py", "src/shared/hierarchical_state_reward.py",
             "src/sac/reward.py", "src/D3QN/reliability_aware_high.py"]
    for name in paths:
        hashes[name] = sha(ROOT / name)
    write(OUT / "frozen_plan.json", dict(run=run, seeds=SEEDS, grid=GRID, scale="large", deadline_type="tight",
          model_initialization_seed=742000, server_seed=42, workers=4, num_workflows=50,
          threshold=.9, selection="original_unfiltered_known_test_diagnostic_not_new_holdout",
          scan="analytic_fixed_placement_only_no_deadline_or_rescheduling_claim", source_sha256=hashes))
    print("[reliability-prepared] original_records=2160 seeds=20", flush=True)


def check_sources(plan):
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")


def independent_reliability(server, rho, duration):
    """Explicitly recompute, without calling the simulator reliability helper."""
    fmax = float(server["max_freq_hz"])
    fmin = float(server["min_freq_hz"])
    freq = max(fmin, min(fmax, float(rho)*fmax))
    rate = float(server["failure_rate"]) * 10**(float(server.get("failure_rate_exponent", 3)) * (fmax-freq)/(fmax-fmin))
    return math.exp(-rate*float(duration))


def workflow_diagnostics(workflow, tasks):
    if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Missing or duplicate workflow tasks")
    reliability = math.prod(t["task_reliability"] for t in tasks)
    if not math.isclose(reliability, float(workflow["reliability"]), abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError("Workflow reliability product mismatch")
    required = float(workflow["reliability_req"])
    met = reliability >= required
    if met != bool(workflow["reliability_met"]):
        raise ValueError("Workflow attainment mismatch")
    return dict(workflow_id=workflow["workflow_id"], reliability=reliability, requirement=required,
                margin=reliability-required, risk_budget_fraction=-math.log(reliability)/-math.log(required),
                reliability_met=met, num_tasks=len(tasks), rho_mean=statistics.mean(t["rho"] for t in tasks),
                total_energy_j=float(workflow["total_energy"]), response_time=workflow["response_time"],
                relative_deadline=workflow["relative_deadline"], deadline_satisfied=workflow["deadline_satisfied"])


def fixed_placement_scan(tasks, servers, requirements, grid):
    from src.simulator.resource import compute_execution_time, compute_energy
    result = []
    for requested in grid:
        groups, energy, frequencies = defaultdict(list), 0.0, []
        for task in tasks:
            server = servers[task["server_id"]]
            rho = max(float(requested), float(server["rho_energy_efficient"]), float(server["min_rho"]))
            duration = compute_execution_time(task["workload_cycles"], server, rho)
            groups[task["workflow_id"]].append(independent_reliability(server, rho, duration))
            energy += compute_energy(task["workload_cycles"], server, rho)
            frequencies.append(rho)
        rels = {name: math.prod(values) for name, values in groups.items()}
        if set(rels) != set(requirements):
            raise ValueError("Scan lost workflows")
        result.append(dict(requested_rho=requested, actual_rho_mean=statistics.mean(frequencies),
                           floor_binding_fraction=sum(f>requested+1e-12 for f in frequencies)/len(frequencies),
                           total_compute_energy_j=energy, reliability=describe(rels.values()),
                           workflow_attainment=statistics.mean(rels[k]>=requirements[k] for k in rels),
                           workflows=[dict(workflow_id=k, reliability=v, margin=v-requirements[k]) for k,v in rels.items()]))
    return result


def initialize():
    global _runtime
    from tools import eval_all_baselines as e
    e.torch.set_num_threads(1)
    plan = read(OUT / "frozen_plan.json")
    config = e.load_json(plan["run"]["config"])
    config["generator"]["dataset_scale"] = "large"
    config = e.apply_deadline_type(config, "tight")
    with redirect_stdout(io.StringIO()):
        resources = e.build_resources(config, 42)
        _, _, agent, objective = e.load_d3qn(plan["run"]["checkpoint"], 742000)
    _runtime = (plan, config, resources, agent, objective)


def diagnose_seed(seed):
    from tools import eval_all_baselines as e
    plan, config, resources, agent, objective = _runtime
    with redirect_stdout(io.StringIO()):
        workflows = e.build_workflows(config, resources, seed, smoke_test=False)
        env = e.build_env(workflows, resources, dict(config["simulation"]), seed)
        result = e.collect_d3qn_episode(env, agent, objective, e.HighRolloutBuffer(), e.LowRolloutBuffer(),
                    training=False, high_deterministic=True, low_deterministic=True,
                    fixed_low_rho_one=False, high_algorithm="d3qn")
    reference = next(r for r in rows(Path(plan["run"]["source"])/"comparison_rows.jsonl")
                     if (r["scale"], r["deadline_type"], r["eval_seed"]) == ("large", "tight", seed))
    metrics = e.json_ready(e.add_normalized_metrics(result["metrics"], 50))
    checked = 0
    for key, value in metrics.items():
        if key in reference:
            same = math.isclose(float(value), float(reference[key]), rel_tol=1e-10, abs_tol=1e-10) if isinstance(value,(float,int)) else value == reference[key]
            if not same:
                raise ValueError(f"Original metric differs seed={seed} key={key}: {value!r} vs {reference[key]!r}")
            checked += 1
    servers = {int(s["id"]): s for s in env.servers}
    task_rows, grouped = [], defaultdict(list)
    for event in result["event_log"]:
        if event.get("replica_role") != "primary" or event.get("backup_enabled"):
            raise ValueError("This audit requires the frozen no-backup protocol")
        server = servers[int(event["server_id"])]
        rho, duration = float(event["rho"]), float(event["exec_time"])
        actual = float(event["task_reliability"])
        if not math.isclose(independent_reliability(server, rho, duration), actual, abs_tol=1e-13, rel_tol=1e-13):
            raise ValueError("Task reliability formula mismatch")
        state = env.workflow_by_id[str(event["workflow_id"])]
        workload = float(env._get_task(state["workflow"], int(event["task_id"]))["workload_cycles"])
        task = dict(workflow_id=str(event["workflow_id"]), task_id=int(event["task_id"]),
                    server_id=int(event["server_id"]), server_type=event["server_type"], rho=rho,
                    exec_time=duration, workload_cycles=workload, energy_j=float(event["energy"]),
                    task_reliability=actual, base_failure_rate=server["failure_rate"],
                    failure_rate=event["failure_rate"], frequency_exponent=server.get("failure_rate_exponent",3),
                    max_freq_hz=server["max_freq_hz"], min_freq_hz=server["min_freq_hz"],
                    rho_energy_efficient=server["rho_energy_efficient"])
        grouped[task["workflow_id"]].append(task)
        task_rows.append(task)
    workflow_rows = [workflow_diagnostics(w, grouped[str(w["workflow_id"])]) for w in result["workflow_results"]]
    if len(workflow_rows) != 50 or len(grouped) != 50 or len(task_rows) != int(metrics["steps"]):
        raise ValueError("Incomplete diagnostic episode")
    attainment = statistics.mean(w["reliability_met"] for w in workflow_rows)
    if not math.isclose(attainment, metrics["reliability_success_rate"], abs_tol=1e-12):
        raise ValueError("Episode attainment mismatch")
    scans = fixed_placement_scan(task_rows, servers, {w["workflow_id"]:w["requirement"] for w in workflow_rows}, plan["grid"])
    return dict(seed=seed, metrics=metrics, original_metrics_verified=checked, workflows=workflow_rows,
                tasks=task_rows, fixed_placement_scan=scans,
                workflow_reliability=describe(w["reliability"] for w in workflow_rows),
                risk_budget_fraction=describe(w["risk_budget_fraction"] for w in workflow_rows))


def run():
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    directory = OUT / "detailed_replay"
    directory.mkdir()  # Refuse restarts or partial replays; retain failures as-is.
    write(directory / "execution.json", dict(workers=plan["workers"], seeds=SEEDS, status="running"))
    with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=multiprocessing.get_context("spawn"), initializer=initialize) as pool:
        for item in pool.map(diagnose_seed, SEEDS):
            write(directory / f"seed_{item['seed']}.json", item)
            print(f"[reliability-seed] seed={item['seed']} workflows=50 minR={item['workflow_reliability']['min']:.8f} original_metrics={item['original_metrics_verified']}", flush=True)
    check_sources(plan)
    write(directory / "completion.json", dict(seeds=20, workflows=1000, source_hashes_verified=True))
    report()
    print("[reliability-complete] seeds=20 workflows=1000", flush=True)


def report():
    plan, audit_data = read(OUT / "frozen_plan.json"), read(OUT / "existing_evidence.json")
    check_sources(plan)
    detail = [read(p) for p in sorted((OUT / "detailed_replay").glob("seed_*.json"))]
    lines = ["# 可靠性诊断结果", "", "独立诊断：不覆盖主表；原始未筛选样本。100%为解析可靠性门槛达标率，不是故障试验零失败。", "",
             "## 原始2160条记录审计", "", "| 模型 | 记录数 | 每批平均R的均值 | 平均rho | 全工作流可靠性达标的批数 |", "|---|---:|---:|---:|---:|"]
    by_run = defaultdict(list)
    for group in audit_data["groups"]:
        by_run[group["run"]].append(group)
    for name, groups in by_run.items():
        lines.append(f"| {name} | 180 | {statistics.mean(g['batch_average_R']['mean'] for g in groups):.8f} | {statistics.mean(g['batch_average_rho']['mean'] for g in groups):.6f} | {sum(g['batches_all_reliable'] for g in groups)}/180 |")
    lines += ["", "上表均值不能解释为最小单工作流R。所有原始失败均保留。", "", "## Full训练中约束是否激活", "",
              "| 模型/历史段 | 范围 | 轮数 | 出现可靠性违约的轮数 | SAC可靠性惩罚非零轮数 | 下层lambda范围 |", "|---|---|---:|---:|---:|---|"]
    for t in audit_data["training"]:
        lam=t["low_reliability_lambda"]
        lines.append(f"| {t['run']} ({Path(t['history']).parent.name}) | {t['scope']} | {t['episodes']} | {t['episodes_with_reliability_failure']} | {t['low_reliability_lagrangian_penalty_mean']['positive_episodes']} | {lam.get('min','NA')}–{lam.get('max','NA')} |")
    lines += ["", "历史激活只能说明训练中观察过该信号，不能替代独立重训消融；选定checkpoint之后的历史不能用于解释其训练。", "", f"## 固定20种子详细复放：{len(detail)}/20", ""]
    if detail:
        workflows = [w for d in detail for w in d["workflows"]]
        stats = describe(w["reliability"] for w in workflows)
        budget = describe(w["risk_budget_fraction"] for w in workflows)
        lines += [f"实际观测工作流数：{len(workflows)}；R最小值={stats['min']:.8f}，P01={stats['p01']:.8f}，中位数={stats['median']:.8f}。",
                  f"风险预算消耗：均值={budget['mean']:.2%}，最大={budget['max']:.2%}；预算定义 -ln(R)/-ln(0.9)，不是平均rho。", "",
                  "| 请求rho | 实际rho均值 | 固定放置计算能耗均值(J) | 工作流R最小值 | 工作流可靠性达标比例 |", "|---|---:|---:|---:|---:|"]
        for index, requested in enumerate(GRID):
            scans = [d["fixed_placement_scan"][index] for d in detail]
            lines.append(f"| {requested:.1f} | {statistics.mean(s['actual_rho_mean'] for s in scans):.6f} | {statistics.mean(s['total_compute_energy_j'] for s in scans):.2f} | {min(s['reliability']['min'] for s in scans):.8f} | {statistics.mean(s['workflow_attainment'] for s in scans):.2%} |")
        lines += ["", "扫描仅固定放置下的解析物理诊断；没有重调度，不计算或宣称扫描的Deadline达标率。实际rho受原能效下界约束。"]
    complete = (OUT / "detailed_replay/completion.json").exists()
    lines += ["", f"完整状态：{'已完成并核验' if complete else '尚未完成，不作最终结论'}。",
              "可靠性模块因果贡献：本实验未检验。没有更改原门槛、故障率、模型或种子。"]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"[reliability-report] detailed_seeds={len(detail)} complete={complete}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "report"])
    action = parser.parse_args().action
    {"prepare": prepare, "run": run, "report": report}[action]()
