"""Complete the 74-server Raw model's SM/MM/LM grid, without new training.

Small's 60 identical cases are reused; only Medium/Large's 120 cases are new.
Rollouts, input construction, loader, diagnostics and validation reuse the
existing credit-ablation evaluator. No 37/111-server transfer is implied.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import io
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import eval_deadline_credit_ablation as credit
from tools.eval_sm74_ablation import read, rows, sha, save, check_sources
from tools.eval_v3_scenarios import result_cells

OUT = ROOT / "results/raw_sm74_generalization_20261024"
OLD = ROOT / "results/deadline_credit_ablation_sm74_20261023"
RUN = ROOT / "results/ablations/sm_74_deadline_credit_20wf/raw_self/seed42_20261023_093836_720976_27828"
SCALES = {"small": "small", "medium": "middle", "large": "large"}
DDLS = ("loose", "medium", "tight")
_base_runtime = None


def validate(data, plan, complete=False):
    expected = [(s, d, n) for s in SCALES for d in DDLS for n in credit.SEEDS]
    keys = [(r["scale"], r["deadline_type"], r["eval_seed"]) for r in data]
    if keys != expected[:len(keys)] or (complete and len(keys) != 180):
        raise ValueError("Missing, duplicate, or reordered 74-server cases")
    for scale in SCALES:
        group = [dict(r, scale="small") for r in data if r["scale"] == scale]
        # The reused validator's first 60 cases are exactly .03, three deadlines.
        credit.validate(group, "raw_self", dict(samples=plan["samples"][scale]))


def prepare():
    from tools import eval_all_baselines as e
    if OUT.exists():
        raise FileExistsError("Already prepared; no duplicate evaluation")
    model = credit.selected_model(RUN, "raw_self")
    if model["checkpoint_sha256"] != "d3911fdb917f0ece732aa776af46b2ddd34d0a3d1817264d4aac6609eed1f046":
        raise ValueError("Unexpected Raw model")
    old = read(OLD / "frozen_plan.json")
    check_sources(old)
    historical = rows(OLD / "raw_self/comparison_rows.jsonl")
    credit.validate(historical, "raw_self", old, complete=True)
    reused = [r for r in historical if r["arrival_rate"] == .03]
    config = read(credit.CONFIG)
    with redirect_stdout(io.StringIO()):
        resources = e.build_resources(config, 42)
    if len(resources["servers"]) != 74:
        raise ValueError("This checkpoint supports the original 74-server protocol only")
    samples = {"small": old["samples"]}
    for scale in ("medium", "large"):
        scenario = copy.deepcopy(config)
        scenario["generator"]["dataset_scale"] = SCALES[scale]
        samples[scale] = {}
        for ddl in DDLS:
            samples[scale][credit.tag(.03, ddl)] = {
                str(seed): credit.build_sample(scenario, resources, .03, ddl, seed)[1]
                for seed in credit.SEEDS
            }
            print(f"[prepare-inputs] {scale}/{ddl} paired_seeds=20", flush=True)
    sources = dict(old["source_sha256"])
    extra = [Path(__file__), ROOT / "tests/test_raw_sm74_generalization.py", OLD / "frozen_plan.json",
             OLD / "raw_self/comparison_rows.jsonl", OLD / "raw_self/completion.json"]
    sources.update({p.relative_to(ROOT).as_posix(): sha(p) for p in extra})
    plan = dict(models={"raw_self": model}, config=credit.CONFIG.relative_to(ROOT).as_posix(),
                samples=samples, source_sha256=sources, workers=2, training=False,
                seeds=credit.SEEDS, servers=74, num_workflows=50, model_initialization_seed=742000,
                reused_small_source=(OLD / "raw_self/comparison_rows.jsonl").relative_to(ROOT).as_posix(),
                scope="SM/MM/LM x three deadlines; 37 and 111 server Raw models not trained")
    validate(reused, plan)
    OUT.mkdir()
    save(OUT / "frozen_plan.json", plan)
    e.write_rows(OUT, reused)
    report()
    print("[prepared] reused=60 new=120 target=180", flush=True)


def initialize():
    global _base_runtime
    credit.OUT = OUT
    credit.initialize("raw_self")
    _base_runtime = credit._runtime


def evaluate_case(case):
    scale, ddl, seed = case
    variant, plan, config, resources, agent, objective = _base_runtime
    local = copy.deepcopy(config)
    local["generator"]["dataset_scale"] = SCALES[scale]
    local_plan = dict(plan, samples=plan["samples"][scale])
    credit._runtime = (variant, local_plan, local, resources, agent, objective)
    row, detail = credit.evaluate_case((.03, ddl, seed))
    row["scale"] = scale
    return row, detail


def run():
    from tools import eval_all_baselines as e
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    detail_dir = OUT / "workflow_details"
    detail_dir.mkdir()  # Atomic no-duplicate/no-resume guard.
    data = rows(OUT / "comparison_rows.jsonl")
    if len(data) != 60:
        raise ValueError("Must start with only the 60 reused Small cases")
    validate(data, plan)
    save(OUT / "execution.json", dict(pid=os.getpid(), workers=2, status="running", new_records=120))
    with (OUT / "evaluation.log").open("x", encoding="utf-8", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        cases = [(s,d,n) for s in ("medium", "large") for d in DDLS for n in credit.SEEDS]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"), initializer=initialize) as pool:
            for row, detail in pool.map(evaluate_case, cases):
                data.append(row)
                validate(data, plan)
                save(detail_dir / f"{row['scale']}_{row['deadline_type']}_seed{row['eval_seed']}.json", detail)
                e.write_rows(OUT, data)
                print(f"[raw-generalization] {len(data)}/180 {row['scale']}/{row['deadline_type']} seed={row['eval_seed']} energy_J={row['total_energy']:.2f} DDL={row['deadline_success_rate']:.4f} Rel={row['reliability_success_rate']:.4f}", flush=True)
        validate(data, plan, complete=True)
        check_sources(plan)
        failures = [{k:r[k] for k in ("scale", "deadline_type", "eval_seed", "deadline_success_rate", "reliability_success_rate")}
                    for r in data if r["deadline_success_rate"] < .9-1e-9 or r["reliability_success_rate"] < 1-1e-9]
        save(OUT / "completion.json", dict(records=180, reused=60, newly_evaluated=120, failures=failures, source_hashes_verified=True))
        save(OUT / "execution.json", dict(pid=os.getpid(), workers=2, status="completed", new_records=120))
        report()
        print(f"[raw-generalization-complete] records=180 failures={len(failures)}", flush=True)


def report():
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    data = rows(OUT / "comparison_rows.jsonl")
    complete = (OUT / "completion.json").exists()
    validate(data, plan, complete=complete)
    lines = ["# Raw-Self 74台全工作流规模评估", "",
             "固定Raw ep1200、budget0.08、74台、50工作流、0.03/s、共享FCFS期限；742000—742019未筛选连续种子。",
             "Small的60条复用相同模型/输入的已完成消融；Medium/Large新评估120条。不是37/111台结果。",
             "能耗J均值±样本SD；ESP@90：每批至少45/50按时；Rel：每批所有工作流可靠性≥0.9。", "",
             "| 场景 | DDL | 能耗 J | ESP@90 | Rel | 状态 |", "|---|---|---:|---:|---:|---|"]
    for scale, label in (("small","SM"),("medium","MM"),("large","LM")):
        for ddl in DDLS:
            group = [r for r in data if r["scale"]==scale and r["deadline_type"]==ddl]
            lines.append(f"| {label} | {ddl} | {result_cells(group,20)} |")
    lines += ["", f"完成标记：{complete}；{len(data)}/180。原模型及历史结果未覆盖。"]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","run","report"))
    args = parser.parse_args()
    {"prepare":prepare,"run":run,"report":report}[args.action]()
