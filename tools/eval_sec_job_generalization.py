"""Frozen S37 evaluation across workload/resource sizes; no learning or model resizing."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import statistics
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import eval_all_baselines as evaluation

CASES = tuple((name + str(edge + cloud), scale, edge, cloud)
              for name, scale in (("S", "small"), ("M", "middle"), ("L", "large"))
              for edge, cloud in ((20, 6), (28, 9)))


def save_outputs(out, rows, workflow_rows):
    summaries = []
    for case, _, _, _ in CASES:
        group = [r for r in rows if r["case"] == case]
        if not group:
            continue
        energy = [r["energy_J"] for r in group]
        summaries.append({"case": case, "seeds_completed": len(group),
                          "energy_mean_J": statistics.mean(energy),
                          "energy_sample_sd_J": statistics.stdev(energy) if len(group) > 1 else 0.0,
                          "ESP90": evaluation.satisfaction_statistics(
                              (r["on_time_count"] / r["workflows"] for r in group),
                              thresholds=(0.9,))["esp"]["0.90"],
                          "reliability_ESP100": sum(r["reliability_rate"] >= 1 - 1e-9
                                                     for r in group) / len(group)})
    for filename, data in (("seed_results.csv", rows), ("workflow_results.csv", workflow_rows),
                           ("summary.csv", summaries)):
        with (out / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    lines = ["# Frozen S37: adapted sec-job protocol, Medium deadline", "",
             "One Small-trained S37 ep1060 checkpoint; no retraining or checkpoint reselection.",
             "50 workflows/seed, arrival rate 0.03/s, shared FCFS reference, DRLEC physical model.",
             "5-seed defaults are exploratory, not formal acceptance; all fixed seeds retained.",
             "53-server cases are unsupported by the unchanged 37-slot network, not scored as failures.",
             "Energy: mean +/- sample SD in J. ESP@90: fraction of seeds with at least 45/50 on time.",
             "", "| Case | Seeds completed | Energy (J) | ESP@90 | Reliability ESP@100 |",
             "|---|---:|---:|---:|---:|"]
    lines += [f"| {r['case']} | {r['seeds_completed']} | {r['energy_mean_J']:.2f} +/- "
              f"{r['energy_sample_sd_J']:.2f} | {r['ESP90']:.0%} | {r['reliability_ESP100']:.0%} |"
              for r in summaries]
    lines += ["", "Generation, references and arrival rate differ from the old experiment.",
              "Do not merge these results into old paper tables or claim an algorithmic improvement.",
              "See experiment_manifest.json, seed_results.csv and workflow_results.csv for provenance."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/v3_small_28edge_9cloud_sec_job_fcfs.json")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=242000)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.num_seeds <= 0:
        raise ValueError("num-seeds must be positive")
    out, checkpoint = Path(args.output_dir).resolve(), Path(args.checkpoint).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh output directory; existing results are never overwritten")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload, config, agent, objective = evaluation.load_d3qn(checkpoint, args.seed_start)
    assert payload["episode"] == 1060 and agent.high_agent.max_servers == 37
    protocol = evaluation.load_json(args.config)
    assert protocol["generator"]["profile"] == "sec_job_fcfs_v1"
    assert protocol["generator"]["num_workflows"] == 50
    config["generator"] = copy.deepcopy(protocol["generator"])
    config["simulation"]["deadline_type"] = "medium"
    evaluation.torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"checkpoint": str(checkpoint), "sha256": digest, "episode": payload["episode"],
                "max_servers": 37, "evaluation_config": config, "cases": CASES,
                "seeds": list(range(args.seed_start, args.seed_start + args.num_seeds)),
                "resource_seed": config["training"]["seed"], "training": False,
                "selection": "fixed consecutive diagnostic seeds; no filtering",
                "unsupported_cases": ["S53", "M53", "L53"]}
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    rows, workflow_rows, signatures = [], [], {}
    for case, scale, edge, cloud in CASES:
        scenario = copy.deepcopy(config)
        scenario["generator"]["dataset_scale"] = scale
        scenario["resources"].update(num_edge=edge, num_cloud=cloud)
        with redirect_stdout(io.StringIO()):
            resources = evaluation.build_resources(scenario, config["training"]["seed"])
        assert len(resources["servers"]) == edge + cloud <= agent.high_agent.max_servers
        for seed in manifest["seeds"]:
            started = time.perf_counter()
            workflows = evaluation.build_workflows(scenario, resources, seed)
            signature = tuple((w.arrival_time, w.deadline_multiplier,
                               tuple((t.id, t.workload_cycles, t.parents) for t in w.tasks),
                               tuple((e.src, e.dst, e.data_size_mb) for e in w.edges)) for w in workflows)
            assert signature == signatures.setdefault((scale, seed), signature)
            env = evaluation.build_env(workflows, resources, scenario["simulation"], seed)
            with redirect_stdout(io.StringIO()):
                result = evaluation.collect_d3qn_episode(
                    env, agent, objective, evaluation.HighRolloutBuffer(), evaluation.LowRolloutBuffer(),
                    training=False, high_deterministic=True, low_deterministic=True,
                    fixed_low_rho_one=False, high_algorithm="d3qn")
            metrics, finished = result["metrics"], result["workflow_results"]
            assert len(finished) == metrics["completed_workflow_count"] == 50
            count = sum(bool(w["deadline_satisfied"]) for w in finished)
            assert abs(count / 50 - metrics["deadline_success_rate"]) < 1e-9
            assert metrics["lambda_deadline"] == 1.4350000000000351
            rows.append({"case": case, "seed": seed, "workflows": 50,
                         "on_time_count": count, "energy_J": metrics["total_energy"],
                         "reliability_rate": metrics["reliability_success_rate"],
                         "queue_s": metrics["mean_high_queue_delay_seconds"],
                         "rho_mean": metrics["rho_mean"], "seconds": time.perf_counter() - started})
            for w in finished:
                workflow_rows.append({"case": case, "seed": seed, **{k: w[k] for k in (
                    "workflow_id", "arrival_time", "relative_deadline", "reference_makespan",
                    "deadline_multiplier", "response_time", "deadline_satisfied", "reliability_met")}})
            save_outputs(out, rows, workflow_rows)
            print(f"[generalization] {case} seed={seed} on_time={count}/50 "
                  f"E={metrics['total_energy']:.2f}J seconds={rows[-1]['seconds']:.1f}", flush=True)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    print(f"Completed {len(rows)} rollouts; output_dir={out}", flush=True)


if __name__ == "__main__":
    main()
