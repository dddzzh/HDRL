"""Extend the existing four Deadline-feedback versions to Large/74 servers."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import eval_deadline_credit_ablation as credit
from tools.eval_sm74_ablation import read, rows, sha, save, check_sources

OLD = ROOT / "results/deadline_credit_ablation_sm74_20261023"
RAW = ROOT / "results/raw_sm74_generalization_20261024"
FULL = ROOT / "results/legacy_train_new_eval_20261009/sm_74"
OUT = ROOT / "results/lm_deadline_credit_20261025"
NEW = ("cf_self", "cf_uniform")
LABELS = {"full_historical": "Full（历史模型）", "cf_self": "CF-Self",
          "cf_uniform": "CF-Uniform", "raw_self": "Raw-Self"}
DDLS = ("loose", "medium", "tight")
WORKERS = 4


def validate(data, variant, plan, complete=False):
    expected = [(d, n) for d in DDLS for n in credit.SEEDS]
    actual = [(r["deadline_type"], r["eval_seed"]) for r in data]
    if len(actual) > 60 or actual != expected[:len(actual)] or (complete and len(actual) != 60):
        raise ValueError("Missing, duplicate, unexpected or reordered LM records")
    if any(r["scale"] != "large" or r["arrival_rate"] != .03 for r in data):
        raise ValueError("Expected Large workflows and arrival rate .03")
    credit.validate([dict(r, scale="small") for r in data], variant, plan)


def failures(data):
    return [{k: r[k] for k in ("deadline_type", "eval_seed", "deadline_success_rate", "reliability_success_rate")}
            for r in data if r["deadline_success_rate"] < .9 - 1e-9 or r["reliability_success_rate"] < 1 - 1e-9]


def prepare():
    from tools import eval_all_baselines as e
    if OUT.exists():
        raise FileExistsError("LM experiment exists; refuse duplicate preparation")
    old, raw = read(OLD / "frozen_plan.json"), read(RAW / "frozen_plan.json")
    check_sources(old)
    check_sources(raw)
    if not read(RAW / "completion.json")["source_hashes_verified"]:
        raise ValueError("Raw reference not completed/verified")
    models = {}
    for variant in NEW:
        prior = old["models"][variant]
        selected = credit.selected_model((ROOT / prior["checkpoint"]).parent, variant)
        if selected != prior:
            raise ValueError("The original validation-selected model changed")
        models[variant] = selected
    models["raw_self"] = old["models"]["raw_self"]
    if models["raw_self"]["checkpoint_sha256"] != raw["models"]["raw_self"]["checkpoint_sha256"]:
        raise ValueError("Raw reference model differs")
    provenance = read(FULL / "checkpoint_provenance.json")
    if (provenance["checkpoint_sha256"] != sha(ROOT / old["full_reference_checkpoint"])
            or provenance["config_sha256"] != sha(ROOT / old["config"])
            or provenance["seed_start"] != 742000 or provenance["num_seeds"] != 20):
        raise ValueError("Historical Full checkpoint or unfiltered evaluation protocol differs")
    models["full_historical"] = dict(checkpoint=old["full_reference_checkpoint"],
        checkpoint_sha256=provenance["checkpoint_sha256"], episode=provenance["episode"])
    paths = [Path(__file__), ROOT / "tests/test_lm_deadline_credit.py", OLD / "frozen_plan.json",
             RAW / "frozen_plan.json", RAW / "comparison_rows.jsonl", RAW / "completion.json",
             FULL / "checkpoint_provenance.json", FULL / "comparison_rows.jsonl"]
    sources = {**old["source_sha256"], **raw["source_sha256"]}
    sources.update({p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
    plan = dict(models=models, config=old["config"], samples=raw["samples"]["large"],
        source_sha256=sources, workers_per_model=WORKERS, torch_threads=1, num_workflows=50,
        servers=74, model_initialization_seed=742000, server_seed=42, seeds=credit.SEEDS,
        scale="large", arrival_rate=.03, training=False, selection_is_conditional=False,
        new_variants=NEW, reused_rows={"full_historical": str((FULL / "comparison_rows.jsonl").relative_to(ROOT)),
                                     "raw_self": str((RAW / "comparison_rows.jsonl").relative_to(ROOT))},
        limitation="Exploratory extension after viewing Small results; one training seed. Full is historical, not a freshly retrained matched fourth model. LS/LL not evaluated here.")
    references = {}
    for variant, source in plan["reused_rows"].items():
        data = [r for r in rows(ROOT / source) if r["scale"] == "large"]
        data.sort(key=lambda r: (DDLS.index(r["deadline_type"]), r["eval_seed"]))
        converted = []
        for row in data:
            sample = plan["samples"][credit.tag(.03, row["deadline_type"])][str(row["eval_seed"])]
            if variant == "raw_self" and any(row[k] != sample[k] for k in credit.SAMPLE_KEYS):
                raise ValueError("Raw reference workload identity differs")
            converted.append(dict(row, **sample, mode=variant, arrival_rate=.03,
                condition=credit.tag(.03, row["deadline_type"]), selection_is_conditional=False))
        validate(converted, variant, plan, complete=True)
        references[variant] = converted
    check_sources(plan)
    OUT.mkdir()
    save(OUT / "frozen_plan.json", plan)
    for variant, data in references.items():
        directory = OUT / variant
        directory.mkdir()
        e.write_rows(directory, data)
        save(directory / "completion.json", dict(records=60, reused_from=plan["reused_rows"][variant],
             failures=failures(data), source_hashes_verified=True))
    report()
    print("[lm-prepared] reused=120 new=120 workers_per_new_model=4", flush=True)


def initialize(variant):
    credit.OUT = OUT
    credit.initialize(variant)
    credit._runtime[2]["generator"]["dataset_scale"] = "large"


def evaluate_case(case):
    row, detail = credit.evaluate_case((.03, *case))
    row["scale"] = "large"
    row["workflow_deadline_miss_count"] = sum(not w["deadline_satisfied"] for w in detail)
    row["normalized_lateness_sum"] = sum(max(0., w["response_time"] / w["relative_deadline"] - 1.) for w in detail)
    return row, detail


def run(variant):
    from tools import eval_all_baselines as e
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    directory = OUT / variant
    directory.mkdir()  # Atomic no-duplicate/no-partial-resume guard.
    (directory / "workflow_details").mkdir()
    save(directory / "experiment_manifest.json", dict(plan["models"][variant], pid=os.getpid(),
         workers=WORKERS, torch_threads=1, model_initialization_seed=742000, server_seed=42,
         scale="large", evaluation_seed_schedule={d: credit.SEEDS for d in DDLS}, training=False,
         selection_is_conditional=False, plan_sha256=sha(OUT / "frozen_plan.json")))
    with (OUT / f"{variant}.log").open("x", encoding="utf-8", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            data = []
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=initialize, initargs=(variant,)) as pool:
                for row, detail in pool.map(evaluate_case, [(d, n) for d in DDLS for n in credit.SEEDS]):
                    data.append(row)
                    validate(data, variant, plan)
                    save(directory / "workflow_details" / f"{row['deadline_type']}_seed{row['eval_seed']}.json", detail)
                    e.write_rows(directory, data)
                    print(f"[lm-eval] {variant} {len(data)}/60 {row['deadline_type']} seed={row['eval_seed']} E_J={row['total_energy']:.2f} DDL={row['deadline_success_rate']:.4f} Rel={row['reliability_success_rate']:.4f}", flush=True)
            validate(data, variant, plan, complete=True)
            check_sources(plan)
            save(directory / "completion.json", dict(records=60, failures=failures(data), source_hashes_verified=True))
            report()
            print(f"[lm-complete] {variant} records=60 failures={len(failures(data))}", flush=True)
        except Exception:
            save(directory / "failure.json", dict(traceback=traceback.format_exc(), automatic_restart=False))
            raise


def report():
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    datasets, complete = {}, True
    for variant in LABELS:
        directory = OUT / variant
        data = rows(directory / "comparison_rows.jsonl") if (directory / "comparison_rows.jsonl").exists() else []
        done = (directory / "completion.json").exists()
        validate(data, variant, plan, complete=done)
        if done and read(directory / "completion.json")["failures"] != failures(data):
            raise ValueError("Completion failure list differs from true raw records")
        datasets[variant] = data
        complete = complete and done
    lines = ["# LM：四版本Deadline反馈扩展评估", "",
             "74台、Large工作流、50wf/seed、0.03/s、原共享FCFS期限；连续未筛选742000—742019。",
             "Full和Raw复用已完成LM真实记录；仅CF-Self/CF-Uniform新增各60条。原始数据及论文未修改。",
             "Full为历史模型，非同代码新重训第四组；单训练seed42，不能把差异当作严格单变量因果证明。",
             "本扩展是在查看Small结果后开展的探索性分析，不是预注册的独立确认实验。",
             "能耗J均值±样本SD；ESP@90为每批至少45/50按时；Rel为每批所有工作流满足可靠性0.9。", "",
             "| DDL | 版本 | 能耗 J | ESP@90 | Rel | 迟到工作流数 | 状态 |",
             "|---|---|---:|---:|---:|---:|---|"]
    for ddl in DDLS:
        for variant, label in LABELS.items():
            data = [r for r in datasets[variant] if r["deadline_type"] == ddl]
            if not data:
                lines.append(f"| {ddl} | {label} | — | — | — | — | 待评估 |")
                continue
            energy = [r["total_energy"] for r in data]
            sd = f"{statistics.stdev(energy):.2f}" if len(energy) > 1 else "—"
            esp = statistics.mean(r["deadline_success_rate"] >= .9 - 1e-9 for r in data)
            rel = statistics.mean(r["reliability_success_rate"] >= 1 - 1e-9 for r in data)
            missed = sum(round(50 * (1 - r["deadline_success_rate"])) for r in data)
            status = "完成" if len(data) == 20 else f"进行中{len(data)}/20，非最终"
            lines.append(f"| {ddl} | {label} | {statistics.mean(energy):.2f} ± {sd} | {esp:.0%} | {rel:.0%} | {missed}/{len(data)*50} | {status} |")
    lines += ["", f"全部完成：{complete}。全部失败保留，不更换种子或测试后重选checkpoint。",
              "", "逐种子CSV/JSONL位于各版本子目录；新评估工作流级明细在workflow_details。"]
    temporary = (OUT / "REPORT.md").with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(OUT / "REPORT.md")
    print(f"[lm-report] all_complete={complete}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("variant", nargs="?", choices=NEW)
    args = parser.parse_args()
    if args.action == "run":
        if not args.variant:
            parser.error("run requires a variant")
        run(args.variant)
    else:
        {"prepare": prepare, "report": report}[args.action]()
