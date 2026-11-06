"""One frozen, validation-only closed-loop DVFS precheck; never trains or resumes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
import multiprocessing
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.diagnose_reliability import read, sha, write
from tools.eval_alibaba_sl111 import identity

OUT = ROOT / "results/reliability_closed_loop_precheck_20261019"
RULES = {"rho_1p0": 1., "rho_0p8": .8, "rho_0p7": .7, "rho_0p6": .6, "rho_ee": 0.}
SEEDS = list(range(542000, 542005))
WORKERS = 5


def executed_frequency(requested, metrics, agent_min):
    floor = max(float(agent_min), float(metrics["rho_action_min"]))
    if not (0 <= requested <= 1 and 0 < floor <= 1):
        raise ValueError("Invalid requested frequency or physical action floor")
    return max(float(requested), floor), floor


def install_rule(low, requested):
    original = low.select_frequency

    def select(*args, **kwargs):
        decision = dict(original(*args, **kwargs))
        metrics = kwargs["safety_metrics"]
        rho, floor = executed_frequency(requested, metrics, low.rho_min)
        decision.update(rho=rho, action_rho=rho, action_u=rho,
                        actor_update_mask=False, forced_max_frequency=False,
                        rho_floor_binding=float(requested < floor - 1e-9),
                        policy_mean=rho, log_prob=0., entropy=0.)
        return decision

    low.select_frequency = select


def scenario():
    from tools import eval_all_baselines as e
    config = read(ROOT / "configs/v3_sm_74_new_eval_50wf.json")
    config["generator"].update(dataset_scale="large", num_workflows=50)
    return e.apply_deadline_type(config, "loose")


def check_sources(plan):
    for path, expected in plan["source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen source changed: {path}")


def network_sha(agent):
    h = hashlib.sha256()
    modules = [agent.high_agent.agent_R.q_eval, agent.high_agent.agent_R.q_target,
               agent.low_agent.actor, agent.low_agent.critic_1, agent.low_agent.critic_2]
    for module in modules:
        for key, value in sorted(module.state_dict().items()):
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare():
    from tools import eval_all_baselines as e
    if OUT.exists():
        raise FileExistsError("Precheck already exists; no restart, overwrite or resume")
    old = read(ROOT / "results/reliability_diagnostic_20261015/frozen_plan.json")
    model = old["run"]
    for key in ("checkpoint", "config"):
        if sha(model[key]) != model[key + "_sha256"]:
            raise ValueError("Original model/config changed")
    payload = e.torch.load(model["checkpoint"], map_location="cpu", weights_only=False)
    train = payload["config"]["training"]
    if int(train["eval_seed"]) != SEEDS[0] or int(train["eval_episodes"]) < len(SEEDS):
        raise ValueError("Seeds are not the first five original validation seeds")
    config = scenario()
    assert (config["resources"]["num_edge"], config["resources"]["num_cloud"]) == (56, 18)
    assert config["simulation"]["workflow_reliability_requirement"] == .9
    assert config["simulation"]["dvfs_action_floor"] == "energy_efficient"
    assert config["simulation"]["enable_backup_duplication"] is False
    with redirect_stdout(io.StringIO()):
        resources = e.build_resources(config, 42)
        samples = {str(s): identity(e.build_workflows(config, resources, s)) for s in SEEDS}
    paths = {Path(model["checkpoint"]), Path(model["config"]),
             ROOT / "configs/resources_edge_cloud_dvfs_v3.json", Path(__file__),
             ROOT / "docs/RELIABILITY_CLOSED_LOOP_PRECHECK_20261019.md",
             ROOT / "tests/test_reliability_closed_loop_precheck.py",
             ROOT / "results/reliability_diagnostic_20261015/frozen_plan.json"}
    paths.update((ROOT / "src").rglob("*.py"))
    paths.update((ROOT / "data/sec_job_reference").glob("*.xml"))
    paths.update(ROOT / "tools" / p for p in ["hierarchical_common.py", "eval_all_baselines.py",
                 "train_hierarchical_d3qn.py", "train_reliability_aware_hierarchical_d3qn.py",
                 "diagnose_reliability.py", "eval_alibaba_sl111.py"])
    # Guard existing results/manuscript against accidental writes by this independent precheck.
    paths.update([ROOT / "paper/manuscript/main2.tex",
                  ROOT / "results/legacy_train_new_eval_20261009/SCENARIO_MATRIX.md",
                  Path(model["source"]) / "comparison_rows.jsonl"])
    plan = dict(model=model, seeds=SEEDS, rules=RULES, workers=WORKERS, torch_threads=1,
                training=False, scale="large", deadline="loose", servers=74,
                num_workflows=50, arrival_rate=.03, resource_seed=42,
                model_initialization_seed=742000, samples=samples,
                seed_role="first_five_original_validation_seeds_not_new_holdout",
                intervention="replace_low_actions_only; high_weights_frozen; placement_and_queues_recomputed",
                deadline_handling="unchanged; rho_safe_is_not_a_hard_action_clip",
                source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)})
    OUT.mkdir()
    (OUT / "cells").mkdir()
    write(OUT / "frozen_plan.json", plan)
    print("[precheck-prepared] validation_seeds=5 rules=5 batches=25", flush=True)
    return plan


def run_cell(item):
    from tools import eval_all_baselines as e
    from src.device_utils import configure_torch_runtime
    rule, seed = item
    destination = OUT / "cells" / f"{rule}_{seed}.json"
    if destination.exists():
        raise FileExistsError("Cell already exists; no repeat evaluation")
    plan = read(OUT / "frozen_plan.json")
    configure_torch_runtime(1)
    start = time.monotonic()
    with redirect_stdout(io.StringIO()):
        config = scenario()
        resources = e.build_resources(config, 42)
        workflows = e.build_workflows(config, resources, seed)
        sample = identity(workflows)
        if sample != plan["samples"][str(seed)]:
            raise ValueError("Unpaired workflows/arrivals/deadlines")
        _, _, agent, objective = e.load_d3qn(plan["model"]["checkpoint"], 742000)
        before = network_sha(agent)
        install_rule(agent.low_agent, RULES[rule])
        env = e.build_env(workflows, resources, dict(config["simulation"]), seed)
        with e.torch.no_grad():
            result = e.collect_d3qn_episode(env, agent, objective,
                e.HighRolloutBuffer(), e.LowRolloutBuffer(), training=False,
                high_deterministic=True, low_deterministic=True,
                fixed_low_rho_one=False, high_algorithm="d3qn")
        if network_sha(agent) != before:
            raise ValueError("Frozen neural weights/buffers changed")
    metrics = e.json_ready(e.add_normalized_metrics(result["metrics"], 50))
    workflows_out = result["workflow_results"]
    events = result["event_log"]
    if len(workflows_out) != 50 or len(events) != sample["tasks"] or not env.done:
        raise ValueError("Incomplete rollout")
    if len({w["workflow_id"] for w in workflows_out}) != 50:
        raise ValueError("Duplicate workflow results")
    servers = {int(s["id"]): s for s in env.servers}
    task_rows = []
    for event in events:
        if event.get("backup_enabled") or event.get("replica_role") != "primary":
            raise ValueError("Unexpected backup execution")
        server = servers[int(event["server_id"])]
        expected = max(RULES[rule], float(server["rho_energy_efficient"]),
                       float(server["min_rho"]), float(agent.low_agent.rho_min))
        if not math.isclose(float(event["rho"]), expected, abs_tol=1e-10):
            raise ValueError("Executed frequency differs from frozen rule/floor")
        task_rows.append({k: event[k] for k in ("workflow_id", "task_id", "server_id",
                          "rho", "exec_time", "energy", "task_reliability")})
    workflow_rows = [{k: w[k] for k in ("workflow_id", "arrival_time", "response_time",
                     "relative_deadline", "deadline_satisfied", "reliability",
                     "reliability_req", "reliability_met", "total_energy")}
                     for w in workflows_out]
    assert all(w["reliability_req"] == .9 for w in workflow_rows)
    dsr = statistics.mean(bool(w["deadline_satisfied"]) for w in workflow_rows)
    rel = statistics.mean(float(w["reliability"]) >= .9 for w in workflow_rows)
    if not math.isclose(dsr, metrics["deadline_success_rate"], abs_tol=1e-12):
        raise ValueError("Deadline metric mismatch")
    if not math.isclose(rel, metrics["reliability_success_rate"], abs_tol=1e-12):
        raise ValueError("Reliability metric mismatch")
    row = dict(rule=rule, eval_seed=seed, metrics=metrics, sample=sample,
               workflows=workflow_rows, tasks=task_rows, weights_unchanged=True,
               network_sha256=before, elapsed_seconds=time.monotonic()-start)
    write(destination, row)
    return dict(rule=rule, seed=seed, energy_j=metrics["total_energy"],
                esp_met=dsr >= .9-1e-9, rel_met=rel >= 1-1e-9,
                workflow_rel_percent=100*rel, seconds=row["elapsed_seconds"])


def summarize(data):
    expected = {(r, s) for r in RULES for s in SEEDS}
    keys = [(r["rule"], r["eval_seed"]) for r in data]
    if len(keys) != 25 or len(set(keys)) != 25 or set(keys) != expected:
        raise ValueError("Summary requires all 25 unique predeclared batches")
    groups = []
    baseline = {r["eval_seed"]: r for r in data if r["rule"] == "rho_1p0"}
    for rule in RULES:
        group = [r for r in data if r["rule"] == rule]
        energies = [r["metrics"]["total_energy"] for r in group]
        workflows = [w for r in group for w in r["workflows"]]
        ratios = [energies[i] / baseline[r["eval_seed"]]["metrics"]["total_energy"]
                  for i, r in enumerate(group)]
        conflict = [r["eval_seed"] for r in group
                    if r["metrics"]["deadline_success_rate"] >= .9-1e-9
                    and r["metrics"]["reliability_success_rate"] < 1-1e-9
                    and r["metrics"]["total_energy"] < baseline[r["eval_seed"]]["metrics"]["total_energy"]]
        groups.append(dict(rule=rule, batches=5, energy_mean_j=statistics.mean(energies),
            energy_sample_sd_j=statistics.stdev(energies),
            esp_percent=100*statistics.mean(r["metrics"]["deadline_success_rate"] >= .9-1e-9 for r in group),
            rel_percent=100*statistics.mean(r["metrics"]["reliability_success_rate"] >= 1-1e-9 for r in group),
            workflow_rel_percent=100*statistics.mean(w["reliability"] >= .9 for w in workflows),
            min_workflow_R=min(w["reliability"] for w in workflows),
            mean_energy_ratio_to_rho1=statistics.mean(ratios),
            simultaneous_ddl_met_rel_failed_energy_saved_seeds=conflict))
    return groups


def report():
    plan = read(OUT / "frozen_plan.json")
    check_sources(plan)
    data = [read(OUT / "cells" / f"{r}_{s}.json") for r in RULES for s in SEEDS]
    for row in data:
        assert row["sample"] == plan["samples"][str(row["eval_seed"])], "Sample identity changed"
    groups = summarize(data)
    write(OUT / "summary.json", dict(all_complete=True, batches=25, groups=groups,
          frozen_sources_unchanged=True, training=False, scope="exploratory_validation_not_causal_ablation"))
    lines = ["# Reliability / Deadline closed-loop precheck", "",
             "74 servers; Large/Loose; 50 workflows; validation seeds 542000–542004; no training.",
             "Frozen upper weights, recomputed placements and queues; original fault rates and action floor.",
             "Rel is the percentage of batches where every workflow has R >= 0.9, not physical R = 1.", "",
             "| Rule | Energy J (mean ± sample SD) | ESP % | Rel % | Workflow attainment % | Minimum R | Conflict seeds |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for g in groups:
        lines.append(f"| {g['rule']} | {g['energy_mean_j']:.2f} ± {g['energy_sample_sd_j']:.2f} | "
            f"{g['esp_percent']:g} | {g['rel_percent']:g} | {g['workflow_rel_percent']:g} | "
            f"{g['min_workflow_R']:.6f} | {g['simultaneous_ddl_met_rel_failed_energy_saved_seeds']} |")
    lines += ["", "Conflict requires lower energy than rho=1 on the SAME seed, ESP threshold met, and reliability failure.",
              "Exploratory validation evidence only; cannot predict trained Reliability-off or prove a learning-term effect.",
              "No new samples, no failed rows removed, no automatic training or further sweep."]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write(OUT / "completion.json", dict(batches=25, complete=True, source_hashes_verified=True))
    print("[precheck-complete] batches=25 output_dir="+str(OUT), flush=True)
    print(json.dumps(groups, ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "report"))
    args = parser.parse_args()
    if args.action == "report":
        report()
        return
    plan = prepare()
    check_sources(plan)
    jobs = [(r, s) for r in RULES for s in SEEDS]
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(run_cell, item) for item in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            print(f"[precheck-progress] {i}/25 "+json.dumps(future.result()), flush=True)
    report()


if __name__ == "__main__":
    main()
