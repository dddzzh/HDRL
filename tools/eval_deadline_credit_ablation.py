"""Frozen, unfiltered five-condition evaluation of the three credit ablations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import copy
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_sm74_ablation import read, rows, sha, save, rank_rounds, check_sources
from tools.eval_alibaba_sl111 import identity
from tools.eval_sm74_arrival_rate import retime, sample_identity, diagnostics

OUT = ROOT / "results/deadline_credit_ablation_sm74_20261023"
TRAIN = ROOT / "results/ablations/sm_74_deadline_credit_20wf"
CONFIG = ROOT / "configs/v3_sm_74_new_eval_50wf.json"
VARIANTS = ("cf_self", "cf_uniform", "raw_self")
CONDITIONS = ((.03, "loose"), (.03, "medium"), (.03, "tight"), (.10, "tight"), (.20, "tight"))
SEEDS = list(range(742000, 742020))
SAMPLE_KEYS = ("physical_sha256", "deadlines_sha256", "workload_sha256", "relative_budgets_sha256", "tasks", "edges")
_runtime = None


def tag(rate, deadline):
    return f"rate_{rate:.2f}_{deadline}".replace(".", "p")


def selected_model(run, variant):
    """Reproduce the original validation ranking, then verify payload identity."""
    from tools import eval_all_baselines as e
    manifest = read(run / "credit_ablation_manifest.json")
    if (manifest["variant"], manifest["status"], manifest["seed"], manifest["requested_episodes"]) != (variant, "completed", 42, 1200):
        raise ValueError("Wrong variant, incomplete training, seed, or episode budget")
    for path, expected in manifest["source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Training source drift: {path}")
    history = rows(run / "train_history.jsonl")
    if [r["episode"] for r in history] != list(range(1, 1201)):
        raise ValueError("Training must have exactly 1200 ordered episodes")
    candidates = rank_rounds([run / "eval_history.jsonl"])
    if [c["global_episode"] for c in candidates] != list(range(20, 1201, 20)):
        raise ValueError("Expected all 60 dual-validation rounds")
    best = max(candidates, key=lambda r: r["rank"])
    feasible = best["rank"][0] == 1
    checkpoint = run / ("best_d3qn_checkpoint.pt" if feasible else "final_checkpoint.pt")
    expected_episode = best["local_episode"] if feasible else 1200
    payload = e.torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    if (payload["episode"], config.get("deadline_credit_ablation", {}).get("variant")) != (expected_episode, variant):
        raise ValueError("Checkpoint payload episode or credit variant differs from selected model")
    if payload["checkpoint_format"] != "hierarchical_reliability_aware_d3qn_v1":
        raise ValueError("Not the full reliability-aware hierarchical D3QN + SAC checkpoint")
    if (config["resources"]["num_edge"], config["resources"]["num_cloud"], config["training"]["seed"]) != (56, 18, 42):
        raise ValueError("Wrong resource scale or model training seed")
    if not config["agent"].get("use_low_agent", True) or not config["agent"]["high"].get("deadline_head_enabled", True):
        raise ValueError("SAC or Deadline Head disabled")
    inputs = rows(run / "workload_inputs.jsonl")
    if [r["seed"] for r in inputs] != list(range(43, 1243)):
        raise ValueError("Missing or reordered training workload inputs")
    return dict(variant=variant, checkpoint=str(checkpoint.relative_to(ROOT)),
                checkpoint_sha256=sha(checkpoint), episode=expected_episode,
                validation_feasible=feasible, selected_validation=best,
                validation_candidates=candidates, training_input_sha256=sha(run / "workload_inputs.jsonl"))


def build_sample(config, resources, rate, deadline, seed):
    from tools import eval_all_baselines as e
    base = e.apply_deadline_type(config, deadline)
    base["generator"]["arrival_rate"] = .03
    with redirect_stdout(io.StringIO()):
        original = e.build_workflows(base, resources, seed, smoke_test=False)
    workflows = retime(original, rate)
    sample = sample_identity(workflows)
    reference = sample_identity(original)
    if any(sample[key] != reference[key] for key in ("workload_sha256", "relative_budgets_sha256")):
        raise ValueError("Retiming changed tasks or relative Deadline budgets")
    if len(workflows) != 50 or any(w.absolute_deadline <= w.arrival_time for w in workflows):
        raise ValueError("Invalid workflow count or deadline")
    return workflows, sample


def prepare():
    from tools import eval_all_baselines as e
    if OUT.exists():
        raise FileExistsError("Experiment already exists; do not prepare/launch twice")
    config = read(CONFIG)
    models = {}
    sources = {CONFIG}
    for variant in VARIANTS:
        runs = list((TRAIN / variant).glob("*/credit_ablation_manifest.json"))
        if len(runs) != 1:
            raise ValueError(f"Ambiguous training runs for {variant}: {len(runs)}")
        run = runs[0].parent
        models[variant] = selected_model(run, variant)
        sources.update(run / name for name in ("credit_ablation_manifest.json", "launch_config.json", "eval_history.jsonl", "train_history.jsonl", "workload_inputs.jsonl"))
        sources.add(ROOT / models[variant]["checkpoint"])
        print(f"[selected] {variant} ep={models[variant]['episode']} feasible={models[variant]['validation_feasible']}", flush=True)
    if len({m["training_input_sha256"] for m in models.values()}) != 1:
        raise ValueError("The three training runs did not receive identical paired workloads")

    full_root = ROOT / "results/legacy_train_new_eval_20261009/sm_74"
    full_provenance = read(full_root / "checkpoint_provenance.json")
    full_checkpoint = Path(full_provenance["source_checkpoint"])
    if sha(full_checkpoint) != full_provenance["checkpoint_sha256"] or sha(CONFIG) != full_provenance["config_sha256"]:
        raise ValueError("Historical Full reference sources differ")
    old_plan_path = ROOT / "results/ablation_sm74_new_eval_20261015/frozen_plan.json"
    old_plan = read(old_plan_path)
    rate_root = ROOT / "results/arrival_rate_sm74_20261017"
    rate_plan = read(rate_root / "frozen_plan.json")
    rate_full = next(m for m in rate_plan["models"] if m["mode"] == "d3qn_sac")
    if rate_full["checkpoint_sha256"] != sha(full_checkpoint):
        raise ValueError("Full arrival-rate reference uses another model")
    sources.update((full_checkpoint, full_root / "checkpoint_provenance.json", full_root / "comparison_rows.jsonl", old_plan_path, rate_root / "frozen_plan.json"))
    with redirect_stdout(io.StringIO()):
        resources = e.build_resources(config, 42)
    samples, full_rows = {}, []
    original_full = rows(full_root / "comparison_rows.jsonl")
    for rate, deadline in CONDITIONS:
        condition = tag(rate, deadline)
        samples[condition] = {}
        if rate == .03:
            reference = [r for r in original_full if r["scale"] == "small" and r["deadline_type"] == deadline]
        else:
            path = rate_root / "d3qn_sac" / f"rate_{rate:.2f}".replace(".", "p") / "comparison_rows.jsonl"
            sources.add(path)
            reference = rows(path)
        if [r["eval_seed"] for r in reference] != SEEDS:
            raise ValueError("Full reference seeds differ from the unfiltered consecutive schedule")
        for seed, row in zip(SEEDS, reference):
            _, sample = build_sample(config, resources, rate, deadline, seed)
            expected = old_plan["samples"][deadline][str(seed)] if rate == .03 else rate_plan["samples"][f"rate_{rate:.2f}".replace(".", "p")][str(seed)]
            keys = ("physical_sha256", "deadlines_sha256", "tasks", "edges") if rate == .03 else SAMPLE_KEYS
            if any(sample[key] != expected[key] for key in keys):
                raise ValueError("Current samples do not match historical Full evaluation inputs")
            if rate != .03 and any(row[key] != sample[key] for key in SAMPLE_KEYS):
                raise ValueError("Historical Full record has mismatched input identity")
            samples[condition][str(seed)] = sample
            full_rows.append(dict(row, mode="full_historical", condition=condition, arrival_rate=rate,
                                  selection_is_conditional=False, **sample))
        print(f"[inputs-verified] {condition} seeds=20 historical_full_reused=True", flush=True)
    sources.update(ROOT / name for name in (
        "tools/eval_deadline_credit_ablation.py", "tools/eval_sm74_ablation.py", "tools/eval_sm74_arrival_rate.py",
        "tools/eval_all_baselines.py", "tools/hierarchical_common.py", "tools/eval_alibaba_sl111.py",
        "tests/test_deadline_credit_evaluation.py", "docs/COUNTERFACTUAL_ABLATION_PLAN_20261021.md",
        "tools/train_hierarchical_d3qn.py", "tools/train_reliability_aware_hierarchical_d3qn.py",
        "src/D3QN/hierarchical_d3qn_agent.py", "src/D3QN/reliability_aware_high.py", "src/D3QN/dual_head_constrained.py",
        "src/sac/hierarchical_low_sac.py", "src/shared/hierarchical_state_reward.py",
        "src/simulator/edge_cloud_env.py", "src/simulator/metrics.py", "src/simulator/resource.py", "src/simulator/reliability.py",
        "src/datasets/sec_job_workflow_generator.py", "configs/resources_edge_cloud_dvfs_v3.json"))
    sources.update((ROOT / "data/sec_job_reference").glob("*.xml"))
    plan = dict(models=models, conditions=CONDITIONS, seeds=SEEDS, num_workflows=50, servers=74,
                config=str(CONFIG.relative_to(ROOT)), samples=samples, workers_per_model=2, torch_threads=1,
                model_initialization_seed=742000, server_seed=42, training=False, selection_is_conditional=False,
                full_reference_checkpoint=str(full_checkpoint.relative_to(ROOT)),
                source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
                limitation="Single training seed42; Full is historical, not a freshly retrained matched fourth version.")
    OUT.mkdir()
    save(OUT / "frozen_plan.json", plan)
    e.write_rows(OUT, full_rows)
    save(OUT / "validation_selection.json", models)
    print(f"[prepared] new_models=3 new_records=300 reused_full_records=100 output={OUT}", flush=True)


def validate(data, variant, plan, complete=False):
    expected = [(tag(rate, deadline), seed) for rate, deadline in CONDITIONS for seed in SEEDS]
    actual = [(r["condition"], r["eval_seed"]) for r in data]
    if actual != expected[:len(actual)] or len(actual) > 100 or (complete and len(actual) != 100):
        raise ValueError("Missing, duplicate, or reordered evaluation records")
    for row in data:
        if row["mode"] != variant or row["scale"] != "small" or row["num_workflows"] != 50 or row["selection_is_conditional"]:
            raise ValueError("Evaluation protocol changed")
        if row["condition"] != tag(row["arrival_rate"], row["deadline_type"]):
            raise ValueError("Condition labels differ from the actual inputs")
        sample = plan["samples"][row["condition"]][str(row["eval_seed"])]
        if any(row[key] != sample[key] for key in SAMPLE_KEYS):
            raise ValueError("Unpaired workload/deadline identity")
        if not math.isfinite(row["total_energy"]) or row["total_energy"] <= 0:
            raise ValueError("Invalid energy")
        for key in ("deadline_success_rate", "reliability_success_rate"):
            if not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError("Invalid success rate")


def initialize(variant):
    global _runtime
    from tools import eval_all_baselines as e
    from src.device_utils import configure_torch_runtime
    configure_torch_runtime(1)
    plan = read(OUT / "frozen_plan.json")
    model = plan["models"][variant]
    with redirect_stdout(io.StringIO()):
        payload, _, agent, objective = e.load_d3qn(ROOT / model["checkpoint"], 742000)
        objective.load_state_dict(payload.get("objective_state", {}))
        config = read(ROOT / plan["config"])
        resources = e.build_resources(config, 42)
    _runtime = variant, plan, config, resources, agent, objective


def evaluate_case(case):
    from tools import eval_all_baselines as e
    variant, plan, config, resources, agent, objective = _runtime
    rate, deadline, seed = case
    started = time.perf_counter()
    workflows, sample = build_sample(config, resources, rate, deadline, seed)
    if sample != plan["samples"][tag(rate, deadline)][str(seed)]:
        raise ValueError("Runtime sample differs from frozen plan")
    simulation = dict(config["simulation"], deadline_type=deadline)
    with redirect_stdout(io.StringIO()):
        env = e.build_env(workflows, resources, simulation, seed)
        result = e.collect_d3qn_episode(env, agent, objective, e.HighRolloutBuffer(), e.LowRolloutBuffer(),
                    training=False, high_deterministic=True, low_deterministic=True,
                    fixed_low_rho_one=False, high_algorithm="d3qn")
    extra, detail = diagnostics(env, result["metrics"])
    row = e.json_ready({**e.add_normalized_metrics(result["metrics"], 50), **extra, **sample,
                    "scale": "small", "deadline_type": deadline, "arrival_rate": rate,
                    "condition": tag(rate, deadline), "eval_seed": seed, "mode": variant, "num_workflows": 50,
                    "selection_is_conditional": False, "evaluation_wall_seconds": time.perf_counter() - started})
    return row, detail


def run(variant):
    from tools import eval_all_baselines as e
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    directory = OUT / variant
    directory.mkdir()  # Refuse duplicate starts and partial-row resumption.
    save(directory / "experiment_manifest.json", dict(plan["models"][variant], workers=2, torch_threads=1,
         training=False, plan_sha256=sha(OUT / "frozen_plan.json"), pid=os.getpid(),
         conditions=CONDITIONS, seeds=SEEDS, num_workflows=50, selection_is_conditional=False,
         model_initialization_seed=742000, server_seed=42))
    (directory / "workflow_details").mkdir()
    cases = [(rate, deadline, seed) for rate, deadline in CONDITIONS for seed in SEEDS]
    data = []
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(variant,)) as pool:
        for row, detail in pool.map(evaluate_case, cases):
            data.append(row)
            validate(data, variant, plan)
            save(directory / "workflow_details" / f"{row['condition']}_seed{row['eval_seed']}.json", detail)
            e.write_rows(directory, data)
            print(f"[credit-eval] {variant} {len(data)}/100 {row['condition']} seed={row['eval_seed']} E_J={row['total_energy']:.2f} DDL={row['deadline_success_rate']:.4f} Rel={row['reliability_success_rate']:.4f}", flush=True)
    validate(data, variant, plan, complete=True)
    check_sources(plan)
    failed = [dict(condition=r["condition"], eval_seed=r["eval_seed"], deadline_success_rate=r["deadline_success_rate"],
                   reliability_success_rate=r["reliability_success_rate"]) for r in data
              if r["deadline_success_rate"] < .9 - 1e-9 or r["reliability_success_rate"] < 1 - 1e-9]
    save(directory / "completion.json", dict(records=100, failures=failed, source_hashes_verified=True))
    print(f"[credit-eval-complete] {variant} records=100 failures={len(failed)}", flush=True)


def report():
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    datasets = {"full_historical": rows(OUT / "comparison_rows.jsonl")}
    all_complete = True
    for variant in VARIANTS:
        path = OUT / variant / "comparison_rows.jsonl"
        data = rows(path) if path.exists() else []
        done = (OUT / variant / "completion.json").exists()
        validate(data, variant, plan, complete=done)
        all_complete = all_complete and done
        datasets[variant] = data
    cells = []
    for rate, deadline in CONDITIONS:
        for variant, data in datasets.items():
            group = [r for r in data if r["condition"] == tag(rate, deadline)]
            if not group:
                continue
            energy = [r["total_energy"] for r in group]
            cells.append(dict(variant=variant, arrival_rate=rate, deadline=deadline, n=len(group),
                 energy_mean_J=statistics.mean(energy), energy_sample_sd_J=statistics.stdev(energy) if len(group) > 1 else None,
                 esp_at_90=statistics.mean(r["deadline_success_rate"] >= .9 - 1e-9 for r in group),
                 reliability_attainment=statistics.mean(r["reliability_success_rate"] >= 1 - 1e-9 for r in group),
                 joint_attainment=statistics.mean(r["deadline_success_rate"] >= .9 - 1e-9 and r["reliability_success_rate"] >= 1 - 1e-9 for r in group)))
    save(OUT / "summary.json", dict(all_complete=all_complete, cells=cells, limitation=plan["limitation"]))
    lines = ["# Deadline责任反馈消融（SM74，未筛选种子）", "", "Small、74台、50工作流/批、连续742000—742019。能耗为J均值±测试种子间样本SD。",
             "ESP@90：每批至少45/50按时；Rel：每批所有工作流可靠性达到0.9。失败均保留。", "",
             "三组新训练均为seed42；Full复用同模型、同输入口径的历史未筛选结果，不是同代码新重训的第四组，不能宣称三训练重复或严格完整四组证据。", "",
             "| Arrival /s | DDL | Version | Energy J | ESP@90 | Rel | Joint | n |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for cell in cells:
        spread = "—" if cell["energy_sample_sd_J"] is None else f"{cell['energy_sample_sd_J']:.2f}"
        lines.append(f"| {cell['arrival_rate']:.2f} | {cell['deadline']} | {cell['variant']} | {cell['energy_mean_J']:.2f} ± {spread} | {cell['esp_at_90']:.0%} | {cell['reliability_attainment']:.0%} | {cell['joint_attainment']:.0%} | {cell['n']} |")
    lines += ["", f"全部完成：{all_complete}。n不足20的格子为进行中，不是最终结果。", "", "模型选择、episode、checkpoint SHA256见validation_selection.json和frozen_plan.json。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[credit-report] all_complete={all_complete}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("variant", nargs="?", choices=VARIANTS)
    args = parser.parse_args()
    if args.action == "run":
        if args.variant is None:
            parser.error("run requires a variant")
        run(args.variant)
    else:
        {"prepare": prepare, "report": report}[args.action]()
