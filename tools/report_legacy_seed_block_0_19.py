"""Validate and separately summarize one exploratory block; never run evaluation."""
import argparse
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.eval_sm74_ablation import read, rows, sha, save, check_sources

OUT = ROOT / "results/legacy_seed_block_0_19_20261026"
SCALES = ("small", "medium", "large")
DDLS = ("loose", "medium", "tight")
SEEDS = list(range(20))
IDS = ("ss_37", "sm_74", "sl_111")


def validate(data, complete=False):
    expected = [(s, d, n) for s in SCALES for d in DDLS for n in SEEDS]
    actual = [(r["scale"], r["deadline_type"], r["eval_seed"]) for r in data]
    if len(actual) > 180 or actual != expected[:len(actual)] or (complete and len(actual) != 180):
        raise ValueError("Rows must follow the real contiguous 0--19 schedule exactly")
    for row in data:
        metrics = [row[k] for k in ("total_energy", "deadline_success_rate", "reliability_success_rate")]
        if (row["mode"] != "d3qn_sac" or row["num_workflows"] != 50 or not all(math.isfinite(v) for v in metrics)
                or metrics[0] <= 0 or not all(0 <= v <= 1 for v in metrics[1:])):
            raise ValueError("Invalid model, workflow count or metrics")


def failure_rows(data):
    return [{k: r[k] for k in ("scale", "deadline_type", "eval_seed", "deadline_success_rate", "reliability_success_rate")}
            for r in data if r["deadline_success_rate"] < .9 - 1e-9 or r["reliability_success_rate"] < 1 - 1e-9]


def check_manifest(directory, model, plan):
    manifest = read(directory / "experiment_manifest.json")
    schedule = read(OUT / "seed_schedule.json")
    if (manifest["workers"], manifest["model_initialization_seed"], manifest["server_seed"],
            manifest["num_workflows"], manifest["d3qn_checkpoint_episode"]) != (4, 742000, 42, 50, model["episode"]):
        raise ValueError("Worker/model/initialization identity changed")
    if (manifest["evaluation_seed_schedule"] != schedule
            or manifest["evaluation_seed_schedule_sha256"] != plan["seed_schedule_sha256"]
            or Path(manifest["d3qn_checkpoint"]).resolve() != (ROOT / model["checkpoint"]).resolve()
            or Path(manifest["config"]).resolve() != (ROOT / model["config"]).resolve()
            or manifest["modes"] != ["d3qn_sac"]):
        raise ValueError("Checkpoint, config or actual workflow seed schedule changed")


def complete(resource):
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    model = next(m for m in plan["models"] if m["id"] == resource)
    directory = OUT / resource
    data = rows(directory / "comparison_rows.jsonl")
    validate(data, complete=True)
    check_manifest(directory, model, plan)
    if not (directory / "comparison_summary.json").exists():
        raise ValueError("Evaluator did not finish writing its final summary")
    save(directory / "completion.json", dict(records=180, failures=failure_rows(data),
         source_hashes_verified=True, workflow_seeds=SEEDS, model_initialization_seed=742000,
         selection_context="One exploratory contiguous block; all failures retained"))
    print(f"[seed-block-complete] {resource} records=180 failures={len(failure_rows(data))}", flush=True)


def report():
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    lines = ["# 旧方案：连续工作流种子0—19单组检查", "",
             "独立探索性结果，不覆盖原始742000—742019数据，不与旧种子或筛选结果拼接。",
             "用户提出寻找全达标连续段；本次只固定并运行0—19一组，不自动追加。若因结果全达标才采用，属于按结果选择的条件样本，不能作为无条件泛化成功率。",
             "仅本方法旧模型，未评估其他算法，不能直接替换原四法配对表。",
             "50wf/seed、0.03/s、原共享FCFS期限；模型初始化742000、资源seed42。能耗J均值±样本SD。",
             "ESP@90为每批至少45/50按时；Rel为每批所有工作流可靠性达到0.9。", "",
             "| 场景 | DDL | 能耗 J | ESP@90 | Rel | 状态 |", "|---|---|---:|---:|---:|---|"]
    cells, all_complete = [], True
    for model in plan["models"]:
        directory = OUT / model["id"]
        path = directory / "comparison_rows.jsonl"
        data = rows(path) if path.exists() else []
        done = (directory / "completion.json").exists()
        validate(data, complete=done)
        if done and read(directory / "completion.json")["failures"] != failure_rows(data):
            raise ValueError("Completion failures differ from raw data")
        all_complete = all_complete and done
        for scale in SCALES:
            label = scale[0].upper() + model["id"][1].upper()
            for ddl in DDLS:
                group = [r for r in data if r["scale"] == scale and r["deadline_type"] == ddl]
                if not group:
                    lines.append(f"| {label} | {ddl} | — | — | — | 待评估 |")
                    continue
                energy = [r["total_energy"] for r in group]
                esp = statistics.mean(r["deadline_success_rate"] >= .9 - 1e-9 for r in group)
                rel = statistics.mean(r["reliability_success_rate"] >= 1 - 1e-9 for r in group)
                sd = statistics.stdev(energy) if len(energy) > 1 else None
                cell = dict(resource=model["id"], scenario=label, scale=scale, deadline_type=ddl,
                            n=len(group), energy_mean_J=statistics.mean(energy), energy_sample_sd_J=sd,
                            esp_at_90=esp, reliability_attainment=rel, complete=len(group) == 20)
                cells.append(cell)
                spread = f"{sd:.2f}" if sd is not None else "—"
                status = "完成" if len(group) == 20 else f"进行中{len(group)}/20，非最终"
                lines.append(f"| {label} | {ddl} | {cell['energy_mean_J']:.2f} ± {spread} | {esp:.0%} | {rel:.0%} | {status} |")
    passed = all_complete and all(c["esp_at_90"] == 1 and c["reliability_attainment"] == 1 for c in cells)
    lines += ["", f"全部完成：{all_complete}；整组所有27档ESP及Rel均100%：{'是' if passed else ('否' if all_complete else '尚未确定')}。",
              "", "各资源子目录保存原始CSV/JSONL及完整评估器摘要；本表只展示J、ESP和Rel，不用平均DSR代替ESP。"]
    save(OUT / "summary.json", dict(all_complete=all_complete, all_cells_joint_100=passed if all_complete else None,
         cells=cells, workflow_seeds=SEEDS, result_conditioned_adoption=True))
    temp = (OUT / "REPORT.md").with_suffix(f".{os.getpid()}.tmp")
    temp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temp.replace(OUT / "REPORT.md")
    print(f"[seed-block-report] all_complete={all_complete}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete", choices=IDS)
    args = parser.parse_args()
    if args.complete:
        complete(args.complete)
    report()
