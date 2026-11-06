"""Run the nine original baselines on the same contiguous block; report all four methods."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.eval_sm74_ablation import read, rows, sha, save, check_sources
from tools.eval_success_seed_panel import validate_rows, passed

OUT = ROOT / "results/baselines_seed_block_0_19_20261026"
OURS = ROOT / "results/legacy_seed_block_0_19_20261026"
SCALES = ("small", "medium", "large")
DDLS = ("loose", "medium", "tight")
SEEDS = list(range(20))
RESOURCES = ("ss_37", "sm_74", "sl_111")
MODES = ("dqn_edge", "dream_dqn", "oo_actor_critic", "d3qn_sac")
LABELS = dict(zip(MODES, ("DQN-Edge", "DReaM-DQN", "OOAC", "CAHDRL（原方案）")))
PREFIXES = {"dqn_edge": "dqn-edge", "dream_dqn": "dream", "oo_actor_critic": "oo"}


def schedule():
    return {s: {d: SEEDS for d in DDLS} for s in SCALES}


def validate(data, model, complete=False):
    validate_rows(data, model, schedule(), complete=complete)


def failures(data):
    return [{k: r[k] for k in ("scale", "deadline_type", "eval_seed", "deadline_success_rate", "reliability_success_rate")}
            for r in data if not passed(r)]


def argv_for(model):
    prefix = PREFIXES[model["mode"]]
    return ["eval_all_baselines", "--config", str(ROOT / model["config"]),
            f"--{prefix}-config", str(ROOT / model["training_config"]),
            f"--{prefix}-checkpoint", str(ROOT / model["checkpoint"]), "--modes", model["mode"],
            "--scales", ",".join(SCALES), "--deadline-types", ",".join(DDLS),
            "--seed-start", "742000", "--num-seeds", "20", "--seed-schedule", str(OURS / "seed_schedule.json"),
            "--workers", "1", "--output-dir", str(OUT / model["id"])]


def check_manifest(model, plan):
    manifest = read(OUT / model["id"] / "experiment_manifest.json")
    prefix = PREFIXES[model["mode"]].replace("-", "_")
    if (manifest["workers"], manifest["model_initialization_seed"], manifest["server_seed"],
            manifest["num_workflows"], manifest[f"{prefix}_checkpoint_episode"]) != (1, 742000, 42, 50, model["episode"]):
        raise ValueError("Workers, initialization or checkpoint episode differs")
    for key, expected in (("config", model["config"]), (f"{prefix}_config", model["training_config"]),
                          (f"{prefix}_checkpoint", model["checkpoint"])):
        if Path(manifest[key]).resolve() != (ROOT / expected).resolve():
            raise ValueError(f"Wrong model/config executed: {key}")
    if (manifest["modes"] != [model["mode"]] or manifest["scales"] != list(SCALES)
            or manifest["deadline_types"] != list(DDLS) or manifest["evaluation_seed_schedule"] != schedule()
            or manifest["evaluation_seed_schedule_sha256"] != plan["seed_schedule_sha256"]):
        raise ValueError("Actual ordered seed schedule or scenario traversal differs")


def run(run_id):
    from tools import eval_all_baselines as evaluation
    from src.device_utils import configure_torch_runtime
    plan = read(OUT / "frozen_plan.json")
    model = next(m for m in plan["models"] if m["id"] == run_id)
    check_sources(plan)
    directory = OUT / run_id
    directory.mkdir()  # Atomic refusal of duplicate or partial-row starts.
    save(directory / "run_provenance.json", dict(model=model, pid=os.getpid(), workers=1, torch_threads=1,
         model_initialization_seed=742000, server_seed=42, workflow_seeds=SEEDS,
         plan_sha256=sha(OUT / "frozen_plan.json"), training=False,
         dqn_edge_state_continuity="single runtime across all 180 rows; no seed reload or splitting"))
    configure_torch_runtime(1)
    with (OUT / f"{run_id}.log").open("x", encoding="utf-8", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        original_writer = evaluation.write_rows
        last_report = time.monotonic()

        def checked_writer(output, data):
            nonlocal last_report
            validate(data, model)
            original_writer(output, data)
            if len(data) % 20 == 0 or time.monotonic() - last_report >= 30:
                report()
                last_report = time.monotonic()

        try:
            with patch.object(sys, "argv", argv_for(model)):
                args = evaluation.parse_args()
            evaluation.write_rows = checked_writer
            _, data, _ = evaluation.evaluate(args)
            validate(data, model, complete=True)
            check_manifest(model, plan)
            check_sources(plan)
            save(directory / "completion.json", dict(records=180, failures=failures(data),
                 source_hashes_verified=True, workers=1, model_initialization_seed=742000, workflow_seeds=SEEDS))
            print(f"[baseline-block-complete] {run_id} records=180 failures={len(failures(data))}", flush=True)
            report()
        except Exception:
            save(directory / "failure.json", dict(traceback=traceback.format_exc(), automatic_restart=False))
            raise
        finally:
            evaluation.write_rows = original_writer


def report():
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    lines = ["# 连续种子0—19：四方法统一对比", "",
             "37/74/111台，Small/Medium/Large × Loose/Medium/Tight，每档同20个真实连续种子0—19。",
             "50wf/seed、0.03/s、原共享FCFS期限；模型初始化742000、资源seed42。固定原模型及各自run配置，未重选checkpoint、未训练。",
             "CAHDRL引用已核验的540条原方案记录，不是Raw，也不重复评估；三个对比算法新评估1620条。",
             "DQN-Edge每个模型内workers=1，完整有序180条共用一个运行时，attention EMA跨种子连续；仅在模型之间并行。",
             "本种子组是在本方法通过后追加对比算法的探索性配对检查；按结果采用该组属于条件样本，不能当作未筛选总体成功率。全部失败保留，不自动换种子。",
             "能耗J均值±样本SD；ESP@90为每批至少45/50按时的种子占比；Rel为每批所有工作流可靠性达到0.9的种子占比。", "",
             "| 场景 | DDL | 方法 | 能耗 J | ESP@90 | Rel | 状态 |", "|---|---|---|---:|---:|---:|---|"]
    datasets, completion = {}, {}
    new_records, finished = 0, 0
    for resource in RESOURCES:
        own = rows(OURS / resource / "comparison_rows.jsonl")
        validate(own, {"mode": "d3qn_sac"}, complete=True)
        datasets[resource, "d3qn_sac"] = own
        completion[resource, "d3qn_sac"] = True
    for model in plan["models"]:
        directory = OUT / model["id"]
        path = directory / "comparison_rows.jsonl"
        data = rows(path) if path.exists() else []
        done = (directory / "completion.json").exists()
        validate(data, model, complete=done)
        if done:
            check_manifest(model, plan)
            if read(directory / "completion.json")["failures"] != failures(data):
                raise ValueError("Completion failures differ from true records")
        datasets[model["resource"], model["mode"]] = data
        completion[model["resource"], model["mode"]] = done
        new_records += len(data)
        finished += int(done)
    cells = []
    for resource in RESOURCES:
        for scale in SCALES:
            scenario = scale[0].upper() + resource[1].upper()
            for ddl in DDLS:
                for mode in MODES:
                    group = [r for r in datasets[resource, mode] if r["scale"] == scale and r["deadline_type"] == ddl]
                    if not group:
                        lines.append(f"| {scenario} | {ddl} | {LABELS[mode]} | — | — | — | 待评估 |")
                        continue
                    energy = [r["total_energy"] for r in group]
                    sd = statistics.stdev(energy) if len(energy) > 1 else None
                    esp = statistics.mean(r["deadline_success_rate"] >= .9 - 1e-9 for r in group)
                    rel = statistics.mean(r["reliability_success_rate"] >= 1 - 1e-9 for r in group)
                    cell = dict(resource=resource, scenario=scenario, scale=scale, deadline_type=ddl, mode=mode,
                                n=len(group), energy_mean_J=statistics.mean(energy), energy_sample_sd_J=sd,
                                esp_at_90=esp, reliability_attainment=rel, complete=len(group) == 20)
                    cells.append(cell)
                    spread = f"{sd:.2f}" if sd is not None else "—"
                    status = "完成" if len(group) == 20 else f"进行中{len(group)}/20，非最终"
                    lines.append(f"| {scenario} | {ddl} | {LABELS[mode]} | {cell['energy_mean_J']:.2f} ± {spread} | {esp:.0%} | {rel:.0%} | {status} |")
    all_complete = finished == 9
    lines += ["", f"对比算法已写入：{new_records}/1620；完整模型：{finished}/9；四方法全部完成：{all_complete}。",
              "", "[固定9模型及来源SHA](frozen_plan.json)；原始CSV/JSONL、执行manifest、日志保存在各模型子目录及同名log。",
              "", "CAHDRL原始记录仍保留在../legacy_seed_block_0_19_20261026/；旧主表、旧数据及论文不覆盖。",
              "", "表格每个模型完成20条或写入后距上次更新满30秒时刷新；部分样本不得视为最终结果。"]
    save(OUT / "summary.json", dict(all_complete=all_complete, new_records=new_records, baseline_models_complete=finished,
         total_rows_including_reference=new_records + 540, cells=cells, workflow_seeds=SEEDS,
         result_conditioned_adoption=True))
    temp = (OUT / "COMPARISON_0_19.md").with_suffix(f".{os.getpid()}.tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(OUT / "COMPARISON_0_19.md")
    print(f"[baseline-block-report] records={new_records}/1620 complete_models={finished}/9", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run")
    choice.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(args.run)
    else:
        report()
