"""Evaluate the completed Raw111 model with the original frozen FCFS protocol."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import io
import multiprocessing
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import eval_raw_ss37_generalization as reference
from tools import eval_deadline_credit_ablation as credit
from tools.eval_sm74_ablation import read, rows, sha, save, check_sources, rank_rounds
from tools.eval_v3_scenarios import result_cells

RUN = ROOT / "results/raw_self_resource_models_20261025/sl_111/seed42_20261025_115207_654763_35688"
OUT = ROOT / "results/raw_sl111_generalization_20261026"
CONFIG = ROOT / "configs/v3_sl_111_new_eval_50wf.json"
SCALES, DDLS, WORKERS = reference.SCALES, reference.DDLS, 8


def select_model():
    from tools import eval_all_baselines as e
    manifest = read(RUN / "credit_ablation_manifest.json")
    if (manifest["variant"], manifest["status"], manifest["seed"], manifest["requested_episodes"],
            manifest["smoke_test"]) != ("raw_self", "completed", 42, 1200, False):
        raise ValueError("Not the completed 1200-episode Raw111 training")
    check_sources(manifest)
    if [r["episode"] for r in rows(RUN / "train_history.jsonl")] != list(range(1, 1201)):
        raise ValueError("Training episodes are missing or reordered")
    if [r["seed"] for r in rows(RUN / "workload_inputs.jsonl")] != list(range(43, 1243)):
        raise ValueError("Training workload seeds differ")
    candidates = rank_rounds([RUN / "eval_history.jsonl"])
    if [c["global_episode"] for c in candidates] != list(range(20, 1201, 20)):
        raise ValueError("Expected all 60 dual-validation rounds")
    best = max(candidates, key=lambda c: c["rank"])
    if best["rank"][0] != 1 or not manifest["validation_feasible"]:
        raise ValueError("No validation-feasible checkpoint: request user direction")
    checkpoint = RUN / "best_d3qn_checkpoint.pt"
    payload = e.torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    if payload["episode"] != best["local_episode"] or config != manifest["config"]:
        raise ValueError("Selected checkpoint episode/config differs from training manifest")
    if payload["checkpoint_format"] != "hierarchical_reliability_aware_d3qn_v1":
        raise ValueError("Wrong checkpoint format")
    if (config["resources"]["num_edge"], config["resources"]["num_cloud"], config["training"]["seed"],
            config["agent"]["shared"]["state"]["max_servers"], config["deadline_credit_ablation"]["variant"],
            config["agent"]["low"]["lagrangian"]["deadline_budget"]) != (84, 27, 42, 111, "raw_self", .08):
        raise ValueError("Wrong resource/model/Raw feedback/budget identity")
    if not config["agent"].get("use_low_agent", True) or not config["agent"]["high"].get("deadline_head_enabled", True):
        raise ValueError("SAC or Deadline Head disabled")
    return dict(variant="raw_self", checkpoint=checkpoint.relative_to(ROOT).as_posix(),
                checkpoint_sha256=sha(checkpoint), episode=payload["episode"], validation_feasible=True,
                selected_validation=best, validation_candidates=candidates)


def validate(data, plan, complete=False):
    expected = [(s, d, n) for s in SCALES for d in DDLS for n in credit.SEEDS]
    actual = [(r["scale"], r["deadline_type"], r["eval_seed"]) for r in data]
    if len(actual) > 180 or actual != expected[:len(actual)] or (complete and len(actual) != 180):
        raise ValueError("Missing, duplicate, unexpected or reordered Raw111 cases")
    for scale in SCALES:
        credit.validate([dict(r, scale="small") for r in data if r["scale"] == scale],
                        "raw_self", dict(samples=plan["samples"][scale]))
    if any(r["arrival_rate"] != .03 for r in data):
        raise ValueError("Arrival rate changed")


def prepare():
    from tools import eval_all_baselines as e
    if OUT.exists():
        raise FileExistsError("Experiment exists; refuse duplicate preparation")
    model = select_model()
    config = read(CONFIG)
    if (config["generator"]["profile"], config["generator"]["num_workflows"],
            config["generator"]["arrival_rate"]) != ("sec_job_fcfs_v1", 50, .03):
        raise ValueError("Wrong evaluation protocol")
    with redirect_stdout(io.StringIO()):
        resources = e.build_resources(config, 42)
    if len(resources["servers"]) != 111 or config["resources"] != read(RUN / "credit_ablation_manifest.json")["config"]["resources"]:
        raise ValueError("Expected the original 84 edge + 27 cloud resources")
    # Reuse the checked dependency inventory, not any Raw37 experimental data.
    prior = read(reference.OUT / "frozen_plan.json")
    check_sources(prior)
    paths = [ROOT / p for p in prior["source_sha256"]
             if not p.replace("\\", "/").startswith("results/") and p != prior["config"]]
    paths += [CONFIG, ROOT / model["checkpoint"], Path(__file__), ROOT / "tests/test_raw_sl111_generalization.py"]
    paths += [RUN / n for n in ("credit_ablation_manifest.json", "launch_config.json", "train_history.jsonl",
                                "eval_history.jsonl", "workload_inputs.jsonl")]
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    sources.update(read(RUN / "credit_ablation_manifest.json")["source_sha256"])
    samples = {}
    for scale in SCALES:
        local = copy.deepcopy(config)
        local["generator"]["dataset_scale"] = SCALES[scale]
        samples[scale] = {}
        for ddl in DDLS:
            samples[scale][credit.tag(.03, ddl)] = {
                str(n): credit.build_sample(local, resources, .03, ddl, n)[1] for n in credit.SEEDS}
            print(f"[prepare-inputs] {scale}/{ddl} seeds=20", flush=True)
    plan = dict(models={"raw_self": model}, config=CONFIG.relative_to(ROOT).as_posix(), samples=samples,
                source_sha256=sources, workers=WORKERS, torch_threads=1, model_initialization_seed=742000,
                server_seed=42, seeds=credit.SEEDS, servers=111, num_workflows=50, training=False,
                selection_is_conditional=False, scope="SL/ML/LL x loose/medium/tight; 180 new cases")
    check_sources(plan)
    OUT.mkdir()
    save(OUT / "frozen_plan.json", plan)
    save(OUT / "validation_selection.json", model)
    report()
    print(f"[prepared] episode={model['episode']} workers={WORKERS} output={OUT}", flush=True)


def initialize():
    reference.OUT = OUT
    reference.initialize()


# The existing scale adapter operates on the worker runtime set above.
evaluate_case = reference.evaluate_case


def report():
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    data = rows(OUT / "comparison_rows.jsonl") if (OUT / "comparison_rows.jsonl").exists() else []
    complete = (OUT / "completion.json").exists()
    validate(data, plan, complete=complete)
    lines = ["# Raw-Self 111台完整泛化评估", "",
             f"固定原双验证选中ep{plan['models']['raw_self']['episode']}；84 edge+27 cloud、50wf/seed、0.03/s、sec_job_fcfs_v1。",
             "连续未筛选742000—742019；SL/ML/LL各三档，每档20条，失败全部保留。原始数据及论文不修改。",
             "能耗J均值±样本SD；ESP@90为每批至少45/50工作流按时；Rel为每批所有工作流可靠性≥0.9。", "",
             "| 场景 | DDL | 能耗 J | ESP@90 | Rel | 状态 |", "|---|---|---:|---:|---:|---|"]
    for scale, label in (("small", "SL"), ("medium", "ML"), ("large", "LL")):
        for ddl in DDLS:
            group = [r for r in data if r["scale"] == scale and r["deadline_type"] == ddl]
            lines.append(f"| {label} | {ddl} | {result_cells(group, 20)} |")
    lines += ["", f"完成标记：{complete}；{len(data)}/180。", "",
              "[逐种子CSV](comparison_rows.csv)；[模型选择](validation_selection.json)；[冻结协议](frozen_plan.json)。"]
    temporary = (OUT / "REPORT.md").with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(OUT / "REPORT.md")


def run():
    from tools import eval_all_baselines as e
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    detail_dir = OUT / "workflow_details"
    detail_dir.mkdir()  # Atomic refusal of duplicate launches or partial resumption.
    save(OUT / "experiment_manifest.json", dict(plan["models"]["raw_self"], pid=os.getpid(), workers=WORKERS,
         torch_threads=1, model_initialization_seed=742000, server_seed=42, servers=111, num_workflows=50,
         evaluation_seed_schedule=[(s, d, n) for s in SCALES for d in DDLS for n in credit.SEEDS],
         plan_sha256=sha(OUT / "frozen_plan.json"), training=False, selection_is_conditional=False))
    with (OUT / "evaluation.log").open("x", encoding="utf-8", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            data = []
            cases = [(s, d, n) for s in SCALES for d in DDLS for n in credit.SEEDS]
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn"), initializer=initialize) as pool:
                for row, detail in pool.map(evaluate_case, cases):
                    data.append(row)
                    validate(data, plan)
                    save(detail_dir / f"{row['scale']}_{row['deadline_type']}_seed{row['eval_seed']}.json", detail)
                    e.write_rows(OUT, data)
                    print(f"[raw111-eval] {len(data)}/180 {row['scale']}/{row['deadline_type']} seed={row['eval_seed']} E_J={row['total_energy']:.2f} DDL={row['deadline_success_rate']:.4f} Rel={row['reliability_success_rate']:.4f}", flush=True)
                    if len(data) % 20 == 0:
                        report()
            validate(data, plan, complete=True)
            check_sources(plan)
            failed = [{k: r[k] for k in ("scale", "deadline_type", "eval_seed", "deadline_success_rate", "reliability_success_rate")}
                      for r in data if r["deadline_success_rate"] < .9 - 1e-9 or r["reliability_success_rate"] < 1 - 1e-9]
            save(OUT / "completion.json", dict(records=180, failures=failed, source_hashes_verified=True))
            report()
            print(f"[raw111-complete] records=180 failures={len(failed)}", flush=True)
        except Exception:
            save(OUT / "failure.json", dict(traceback=traceback.format_exc(), automatic_restart=False))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "report": report}[args.action]()
