"""Frozen 111-server models, Alibaba 100-workflow paired evaluation and reporting."""
import argparse
import copy
import hashlib
import io
import json
import math
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "results/alibaba_sl111_100wf_20261015"
MATRIX = ROOT / "results/legacy_train_new_eval_20261009"
CONFIG = ROOT / "configs/alibaba_sl_111_eval_100wf.json"
MODES = ["d3qn_sac", "dqn_edge", "dream_dqn", "oo_actor_critic"]
DEADLINES = ["loose", "medium", "tight"]
LABELS = dict(zip(MODES, ["D3QN+SAC", "DQN-Edge", "DReaM-DQN", "OOAC"]))
SEEDS = list(range(742000, 742020))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(workflows):
    physical = [{"id": w.workflow_id, "arrival": w.arrival_time, "reliability": w.reliability,
                 "tasks": [(t.id, t.workload_cycles) for t in w.tasks],
                 "edges": [(e.src, e.dst, e.data_size_mb) for e in w.edges]} for w in workflows]
    deadlines = [(w.workflow_id, w.reference_makespan, w.deadline_multiplier,
                  w.relative_deadline, w.absolute_deadline) for w in workflows]
    hash_json = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()
    return dict(job_ids=[w.workflow_id for w in workflows], physical_sha256=hash_json(physical),
                deadlines_sha256=hash_json(deadlines), tasks=sum(len(w.tasks) for w in workflows),
                edges=sum(len(w.edges) for w in workflows))


def save(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_sources(plan):
    hashes = {**plan["source_sha256"], **plan["original_data_sha256"]}
    for run in plan["models"]:
        hashes.update({run["checkpoint"]: run["checkpoint_sha256"],
                       run["training_config"]: run["training_config_sha256"]})
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")


def validate_rows(rows, mode, plan, complete=False):
    expected = [(d, s) for d in DEADLINES for s in SEEDS]
    keys = [(r["deadline_type"], r["eval_seed"]) for r in rows]
    if keys != expected[:len(keys)] or len(keys) > 60 or (complete and len(keys) != 60):
        raise ValueError("Missing, reordered, duplicated or unexpected Alibaba rows")
    for row in rows:
        sample = plan["samples"][row["deadline_type"]][str(row["eval_seed"])]
        if (row["mode"] != mode or row["num_workflows"] != 100
                or row["scale"] != "small" or row["dataset_source"] != "alibaba_v2018"
                or row["physical_sha256"] != sample["physical_sha256"]
                or row["deadlines_sha256"] != sample["deadlines_sha256"]
                or row["tasks"] != sample["tasks"] or row["edges"] != sample["edges"]):
            raise ValueError("Wrong model, dataset, sample or workflow count")
        values = [float(row[k]) for k in ("total_energy", "deadline_success_rate", "reliability_success_rate")]
        if not all(math.isfinite(v) for v in values) or values[0] <= 0 or not all(0 <= v <= 1 for v in values[1:]):
            raise ValueError("Invalid evaluation metrics")


def prepare():
    from tools.hierarchical_common import build_resources, build_workflows
    from tools import eval_all_baselines as evaluation
    if OUTPUT.exists():
        raise FileExistsError("Experiment already exists; never prepare/restart it automatically")
    source_plan = ROOT / "results/success_seed_screening_20261013/panel_plan.json"
    old = read(source_plan)
    models = [r for r in old["runs"] if r["resource"] == "sl"]
    assert [r["mode"] for r in models] == MODES
    code = ["src/datasets/alibaba_sec_job.py", "src/datasets/workload_mapping.py",
            "src/datasets/sec_job_workflow_generator.py", "src/datasets/workflow_factory.py",
            "src/datasets/pegasus_workflow_generator.py", "src/simulator/edge_cloud_env.py",
            "src/simulator/resource.py", "src/schedulers/heft.py", "tools/hierarchical_common.py",
            "tools/eval_all_baselines.py", "tools/eval_alibaba_sl111.py"]
    paths = [CONFIG, source_plan, ROOT / "outputs/alibaba_test.json", ROOT / "outputs/alibaba_train.json",
             ROOT / "configs/resources_edge_cloud_dvfs_v3.json"] + [ROOT / p for p in code]
    plan = dict(dataset="alibaba_v2018", num_workflows=100, num_servers=111, edge=84, cloud=27,
                seeds=SEEDS, deadline_types=DEADLINES, workers_per_model=1, torch_threads=1,
                model_initialization_seed=742000, resource_seed=42,
                selection_is_conditional=False, models=models,
                source_sha256={str(p): digest(p) for p in paths},
                original_data_sha256=old["original_file_sha256"], samples={})
    check_sources(plan)
    # All chosen checkpoints must actually come from the fixed scientific-workflow runs.
    for run in models:
        payload = evaluation.torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
        cfg = (read(run["training_config"]) if run["training_config_storage"] == "json"
               else payload["config"])
        if cfg.get("generator", {}).get("source", cfg.get("generator", {}).get("type", "pegasus")) != "pegasus":
            raise ValueError("Checkpoint is not from the scientific-workflow source")
        assert cfg["resources"]["num_edge"] == 84 and cfg["resources"]["num_cloud"] == 27
        # DQN-Edge/DReaM store learner-only config; their frozen resolved JSON is the loader source.
        if "generator" in payload["config"] and payload["config"] != cfg:
            raise ValueError("Resolved loader config differs from checkpoint config")
        del payload
    cfg = read(CONFIG)
    with redirect_stdout(io.StringIO()):
        resources = build_resources(cfg, 42)
    assert len(resources["servers"]) == 111
    for deadline in DEADLINES:
        scenario = copy.deepcopy(cfg)
        scenario["simulation"]["deadline_type"] = deadline
        plan["samples"][deadline] = {}
        for seed in SEEDS:
            workflows = build_workflows(scenario, resources, seed)
            assert len(workflows) == len({w.workflow_id for w in workflows}) == 100
            sample = identity(workflows)
            if deadline != "loose":
                assert sample["physical_sha256"] == plan["samples"]["loose"][str(seed)]["physical_sha256"]
            plan["samples"][deadline][str(seed)] = sample
        print(f"[prepare] {deadline}: 20 paired 100-job samples frozen", flush=True)
    check_sources(plan)
    OUTPUT.mkdir(parents=True)
    schedule = {"small": {d: SEEDS for d in DEADLINES}}
    save(OUTPUT / "seed_schedule.json", schedule)
    plan["source_sha256"][str(OUTPUT / "seed_schedule.json")] = digest(OUTPUT / "seed_schedule.json")
    save(OUTPUT / "evaluation_plan.json", plan)
    shutil.copy2(MATRIX / "SCENARIO_MATRIX.md", OUTPUT / "SCENARIO_MATRIX_before_alibaba.md")
    report()
    print(f"[prepared] output_dir={OUTPUT}", flush=True)


def run_one(mode):
    from tools import eval_all_baselines as evaluation
    plan = read(OUTPUT / "evaluation_plan.json")
    check_sources(plan)
    out = OUTPUT / mode
    if out.exists():
        raise FileExistsError("Model run exists: no automatic restart, partial resume or duplicate launch")
    run = next(r for r in plan["models"] if r["mode"] == mode)
    prefix = dict(d3qn_sac="d3qn", dqn_edge="dqn-edge", dream_dqn="dream", oo_actor_critic="oo")[mode]
    argv = ["eval_all_baselines", "--config", str(CONFIG), f"--{prefix}-checkpoint", run["checkpoint"],
            "--modes", mode, "--scales", "small", "--deadline-types", ",".join(DEADLINES),
            "--seed-start", "742000", "--num-seeds", "20", "--workers", "1",
            "--seed-schedule", str(OUTPUT / "seed_schedule.json"), "--output-dir", str(out)]
    if mode != "d3qn_sac":
        argv += [f"--{prefix}-config", run["training_config"]]
    with patch.object(sys, "argv", argv):
        args = evaluation.parse_args()
    evaluation.torch.set_num_threads(1)
    out.mkdir()
    save(out / "run_provenance.json", dict(model=run, config=str(CONFIG), plan_sha256=digest(OUTPUT / "evaluation_plan.json"),
                                         unselected=True, workers=1, model_initialization_seed=742000,
                                         order="loose,medium,tight; each 742000..742019; one continuous model instance",
                                         scale_field="small is evaluator transport slot, not Alibaba size classification"))
    original_build, original_writer = evaluation.build_workflows, evaluation.write_rows
    generated = {}

    def checked_build(config, resources, seed, smoke_test=False):
        if smoke_test or config["generator"]["source"] != "alibaba_json":
            raise ValueError("Wrong Alibaba evaluation source/protocol")
        workflows = original_build(config, resources, seed, smoke_test=False)
        key = (config["simulation"]["deadline_type"], seed)
        actual = identity(workflows)
        if actual != plan["samples"][key[0]][str(seed)]:
            raise ValueError("Generated workload differs from frozen paired sample")
        generated[key] = actual
        return workflows

    def checked_write(directory, rows):
        for row in rows:
            sample = generated[row["deadline_type"], row["eval_seed"]]
            row.update(dataset_source="alibaba_v2018", servers=111,
                       **{k: sample[k] for k in ("physical_sha256", "deadlines_sha256", "tasks", "edges")})
        validate_rows(rows, mode, plan)
        original_writer(directory, rows)
        report()

    with patch.object(evaluation, "build_workflows", checked_build), patch.object(evaluation, "write_rows", checked_write):
        _, rows, _ = evaluation.evaluate(args)
    validate_rows(rows, mode, plan, complete=True)
    manifest = read(out / "experiment_manifest.json")
    if manifest["workers"] != 1 or manifest["model_initialization_seed"] != 742000 or manifest["server_seed"] != 42:
        raise ValueError("Model initialization or worker protocol changed")
    check_sources(plan)
    failures = [dict(deadline_type=r["deadline_type"], eval_seed=r["eval_seed"],
                     deadline_success_rate=r["deadline_success_rate"], reliability_success_rate=r["reliability_success_rate"])
                for r in rows if r["deadline_success_rate"] < .9 - 1e-9 or r["reliability_success_rate"] < 1 - 1e-9]
    save(out / "completion.json", dict(records=60, source_hashes_verified=True, failures=failures))
    report()
    print(f"[alibaba-complete] mode={mode} records=60 failures={len(failures)}", flush=True)


def report():
    from tools.eval_v3_scenarios import refresh
    from tools.eval_all_baselines import describe
    import os
    plan = read(OUTPUT / "evaluation_plan.json")
    samples = plan["samples"]["loose"].values()
    task_counts = [s["tasks"] for s in samples]
    lines = ["## Alibaba跨来源泛化：111台服务器／100工作流（未筛选测试）", "",
             "本节单独使用100工作流，不套用上方科学工作流50工作流的45/50门槛。",
             "固定原111台四算法模型，不在Alibaba重新训练或选checkpoint；每档20个固定seed（742000–742019），保留全部失败。",
             "84 edge+27 cloud；到达率0.03/s；sec_job_fcfs_v1共享FCFS参考DDL。每seed至少90/100按时才计入ESP@90；可靠性达标要求该seed的100个工作流均达到0.9。",
             "真实trace提供DAG/相对强度，cycles、通信、到达与约束为仿真映射；不是阿里线上能耗实测。DQN-Edge按原顺序串行保留跨seed EMA，无梯度学习。",
             f"20批样本每批总任务数：{min(task_counts)}–{max(task_counts)}，均值{sum(task_counts)/len(task_counts):.2f}；100工作流数量不代表100个并发工作流或高负载。",
             "[冻结模型、数据/代码SHA与真实job名单](../alibaba_sl111_100wf_20261015/evaluation_plan.json)。",
             "", "| 数据来源 | Deadline | 方法 | 能耗（J，均值±样本SD） | ESP@90 | 可靠性达标 | 状态 |",
             "|---|---|---|---:|---:|---:|---|"]
    data, done, failures = {}, {}, []
    for mode in MODES:
        path = OUTPUT / mode / "comparison_rows.jsonl"
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path.exists() else []
        validate_rows(rows, mode, plan)
        completion = OUTPUT / mode / "completion.json"
        done[mode] = completion.exists()
        if done[mode]:
            validate_rows(rows, mode, plan, complete=True)
            raw_failures = [dict(deadline_type=r["deadline_type"], eval_seed=r["eval_seed"],
                                 deadline_success_rate=r["deadline_success_rate"], reliability_success_rate=r["reliability_success_rate"])
                            for r in rows if r["deadline_success_rate"] < .9 - 1e-9 or r["reliability_success_rate"] < 1 - 1e-9]
            completion_data = read(completion)
            if completion_data["records"] != 60 or completion_data["failures"] != raw_failures:
                raise ValueError("Completion marker differs from actual rows")
            failures += [dict(mode=mode, **f) for f in raw_failures]
        data[mode] = rows
    for deadline in DEADLINES:
        for mode in MODES:
            group = [r for r in data[mode] if r["deadline_type"] == deadline]
            if not group:
                cells = "— | — | — | 待评估"
            else:
                e = describe([r["total_energy"] for r in group])
                sd = f"{e['std']:.2f}" if len(group) > 1 else "—"
                esp = sum(r["deadline_success_rate"] >= .9 - 1e-9 for r in group) / len(group)
                rel = sum(r["reliability_success_rate"] >= 1 - 1e-9 for r in group) / len(group)
                status = "完成并核验" if done[mode] else f"非最终 {len(data[mode])}/60"
                cells = f"{e['mean']:.2f} ± {sd} | {esp:.0%} | {rel:.0%} | {status}"
            lines.append(f"| Alibaba-111 | {deadline.title()} | {LABELS[mode]} | {cells} |")
    lines += ["", f"完整模型：{sum(done.values())}/4；已完成模型未通过记录：{len(failures)}。", ""]
    for mode in MODES:
        lines.append(f"{LABELS[mode]}逐种子原始数据：[CSV](../alibaba_sl111_100wf_20261015/{mode}/comparison_rows.csv)。")
    path = MATRIX / "ALIBABA_SL111_100WF.md"
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)
    refresh(MATRIX)
    return dict(all_complete=all(done.values()), counts={m: len(v) for m, v in data.items()}, failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prepare", action="store_true")
    choice.add_argument("--run", choices=MODES)
    choice.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        run_one(args.run)
    else:
        print(json.dumps(report(), ensure_ascii=False))


if __name__ == "__main__":
    main()
