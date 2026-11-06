"""Evaluate only the completed isolated-HEFT budget=.01 Full pilot.

Reuse the frozen five-condition credit-ablation inputs and rollout code.
Historical controls keep budget=.08; this is exploratory, not a matched ablation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
from contextlib import redirect_stdout, redirect_stderr
import multiprocessing
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import eval_deadline_credit_ablation as credit
from tools.eval_sm74_ablation import read, rows, sha, save, check_sources, rank_rounds

RUN = ROOT / "results/deadline_reference_sm74_budget001/isolated_heft/seed42_20261023_200141_483735_49412"
OLD = ROOT / "results/deadline_credit_ablation_sm74_20261023"
OUT = ROOT / "results/deadline_budget001_isolated_sm74_20261024"
MODE = "full_budget001_isolated"
EXPECTED_SHA = "a2bdefeeab3c1c4ab755fa48520affbfaab9ffbee6531b8c9864a5426ea1b9f0"


def select_model():
    from tools import eval_all_baselines as e
    from tools.train_deadline_reference_experiment import validate_config
    manifest = read(RUN / "deadline_reference_manifest.json")
    if manifest["status"] != "completed" or manifest["training_reference"] != "isolated_heft":
        raise ValueError("Not a completed isolated HEFT training run")
    check_sources(manifest)
    config = read(RUN / "resolved_config.json")
    if config != manifest["config"]:
        raise ValueError("Resolved config and manifest disagree")
    clean = copy.deepcopy(config)
    clean["training"].pop("run_name")
    validate_config(clean)
    history = rows(RUN / "train_history.jsonl")
    if [r["episode"] for r in history] != list(range(1, 1201)):
        raise ValueError("Training is not exactly 1200 ordered episodes")
    inputs = rows(RUN / "training_inputs.jsonl")
    if [r["seed"] for r in inputs] != list(range(43, 1243)) or any(
        r["training_reference"] != "isolated_heft" or r["deadlines_sha256"] != r["legacy_deadlines_sha256"] for r in inputs
    ):
        raise ValueError("Training input identity is not the unchanged isolated protocol")
    candidates = rank_rounds([RUN / "eval_history.jsonl"])
    if [c["global_episode"] for c in candidates] != list(range(20, 1201, 20)):
        raise ValueError("Missing dual-validation rounds")
    best = max(candidates, key=lambda c: c["rank"])
    if best["rank"][0] != 1 or best["local_episode"] != 1100 or not manifest["validation_feasible"]:
        raise ValueError("Validation selection differs from the verified episode 1100")
    checkpoint = RUN / "best_d3qn_checkpoint.pt"
    if sha(checkpoint) != EXPECTED_SHA:
        raise ValueError("Wrong checkpoint hash")
    payload = e.torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["episode"] != 1100 or payload["config"] != config:
        raise ValueError("Wrong checkpoint episode/config")
    if payload["checkpoint_format"] != "hierarchical_reliability_aware_d3qn_v1":
        raise ValueError("Wrong model format")
    return dict(checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_sha256=EXPECTED_SHA,
                episode=1100, validation_feasible=True, selected_validation=best,
                validation_candidates=candidates, training_budget=.01, training_reference="isolated_heft")


def prepare():
    if OUT.exists():
        raise FileExistsError("Output already exists; never prepare twice")
    model = select_model()
    old_plan = read(OLD / "frozen_plan.json")
    check_sources(old_plan)
    if not read(OLD / "summary.json")["all_complete"]:
        raise ValueError("Historical controls are incomplete")
    references = {}
    for name in ("full_historical", *credit.VARIANTS):
        path = OLD / "comparison_rows.jsonl" if name == "full_historical" else OLD / name / "comparison_rows.jsonl"
        data = rows(path)
        credit.validate(data, name, old_plan, complete=True)
        references[name] = path.relative_to(ROOT).as_posix()
    sources = dict(old_plan["source_sha256"])
    files = [Path(__file__), OLD / "frozen_plan.json", OLD / "summary.json",
             ROOT / model["checkpoint"], *(ROOT / p for p in references.values())]
    files += [RUN / name for name in ("resolved_config.json", "deadline_reference_manifest.json",
                                     "eval_history.jsonl", "train_history.jsonl", "training_inputs.jsonl")]
    files += [ROOT / "tests/test_deadline_budget001_evaluation.py"]
    sources.update({p.relative_to(ROOT).as_posix(): sha(p) for p in files})
    plan = dict(models={MODE: model}, config=old_plan["config"], samples=old_plan["samples"],
                conditions=credit.CONDITIONS, seeds=credit.SEEDS, num_workflows=50, servers=74,
                workers_per_model=2, torch_threads=1, model_initialization_seed=742000, server_seed=42,
                training=False, selection_is_conditional=False, historical_rows=references, source_sha256=sources,
                limitation="Exploratory: new Full budget=0.01 versus historical controls budget=0.08; single training seed; previously viewed test seeds.")
    OUT.mkdir()
    save(OUT / "frozen_plan.json", plan)
    save(OUT / "validation_selection.json", model)
    report()
    print(f"[prepared] checkpoint_ep=1100 records=100 unfiltered=True output={OUT}", flush=True)


def initialize():
    credit.OUT = OUT
    credit.initialize(MODE)


def run():
    from tools import eval_all_baselines as e
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    directory = OUT / MODE
    directory.mkdir()  # Duplicate launch / partial resume is forbidden.
    save(directory / "experiment_manifest.json", dict(plan["models"][MODE], pid=os.getpid(), workers=2,
         torch_threads=1, model_initialization_seed=742000, training=False,
         conditions=credit.CONDITIONS, seeds=credit.SEEDS, plan_sha256=sha(OUT / "frozen_plan.json")))
    (directory / "workflow_details").mkdir()
    data = []
    with (OUT / "evaluation.log").open("x", encoding="utf-8", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        cases = [(rate, deadline, seed) for rate, deadline in credit.CONDITIONS for seed in credit.SEEDS]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"), initializer=initialize) as pool:
            for row, detail in pool.map(credit.evaluate_case, cases):
                data.append(row)
                credit.validate(data, MODE, plan)
                save(directory / "workflow_details" / f"{row['condition']}_seed{row['eval_seed']}.json", detail)
                e.write_rows(directory, data)
                print(f"[budget-eval] {len(data)}/100 {row['condition']} seed={row['eval_seed']} E_J={row['total_energy']:.2f} DDL={row['deadline_success_rate']:.4f} Rel={row['reliability_success_rate']:.4f}", flush=True)
        credit.validate(data, MODE, plan, complete=True)
        check_sources(plan)
        failures = [{k: r[k] for k in ("condition", "eval_seed", "deadline_success_rate", "reliability_success_rate")}
                    for r in data if r["deadline_success_rate"] < .9-1e-9 or r["reliability_success_rate"] < 1-1e-9]
        save(directory / "completion.json", dict(records=100, failures=failures, source_hashes_verified=True))
        report()
        print(f"[budget-eval-complete] records=100 failures={len(failures)}", flush=True)


def report():
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    path = OUT / MODE / "comparison_rows.jsonl"
    data = rows(path) if path.exists() else []
    complete = (OUT / MODE / "completion.json").exists()
    credit.validate(data, MODE, plan, complete=complete)
    datasets = {name: rows(ROOT / path) for name, path in plan["historical_rows"].items()}
    datasets[MODE] = data
    cells = []
    for rate, deadline in credit.CONDITIONS:
        for name, dataset in datasets.items():
            group = [r for r in dataset if r["condition"] == credit.tag(rate, deadline)]
            if not group:
                continue
            energy = [r["total_energy"] for r in group]
            cells.append(dict(variant=name, budget=.01 if name==MODE else .08, arrival_rate=rate,
                 deadline=deadline, n=len(group), energy_mean_J=statistics.mean(energy),
                 energy_sample_sd_J=statistics.stdev(energy) if len(group)>1 else None,
                 esp_at_90=statistics.mean(r["deadline_success_rate"] >= .9-1e-9 for r in group),
                 reliability_attainment=statistics.mean(r["reliability_success_rate"] >= 1-1e-9 for r in group)))
    save(OUT / "summary.json", dict(all_complete=complete, cells=cells, limitation=plan["limitation"]))
    lines = ["# Full budget0.01：原HEFT训练，五条件配对测试", "",
             "新模型是第1100轮验证最优检查点；不是第1200轮最终模型。", "",
             "SM74、50工作流、连续未筛选742000—742019；.03三档DDL，.10/.20 Tight保持.03生成的工作量与相对期限。",
             "ESP@90指每批至少45/50按时；Rel指每批所有工作流可靠性≥0.9。", "",
             "注意：旧对照budget0.08，新Full budget0.01；仅探索性比较，不是同预算归因消融。", "",
             "| Arrival /s | DDL | Version | Budget | Energy J mean ± sample SD | ESP@90 | Rel | n |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for c in cells:
        sd = "—" if c["energy_sample_sd_J"] is None else f"{c['energy_sample_sd_J']:.2f}"
        lines.append(f"| {c['arrival_rate']:.2f} | {c['deadline']} | {c['variant']} | {c['budget']:.2f} | {c['energy_mean_J']:.2f} ± {sd} | {c['esp_at_90']:.0%} | {c['reliability_attainment']:.0%} | {c['n']} |")
    lines += ["", f"新评估全部完成：{complete}；新记录{len(data)}/100。未完成的格子不是最终结果。"]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "report": report}[args.action]()
