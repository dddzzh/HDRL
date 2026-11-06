"""Select saved SS checkpoints: all 40 validation seeds feasible, then energy minimum."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import eval_all_baselines as evaluation

SEEDS = tuple(range(542000, 542020)) + tuple(range(642000, 642020))
# Previously failing VALIDATION seeds first. This changes cost, not the all-pass rule.
ORDER = (542004, 542006, 542007, 542014, 642019) + tuple(
    seed for seed in SEEDS if seed not in (542004, 542006, 542007, 542014, 642019))


def passed(row):
    return (row["deadline_success_rate"] >= .90 - 1e-9
            and row["reliability_success_rate"] >= 1.0 - 1e-9)


def candidate_status(rows):
    seeds = [row["eval_seed"] for row in rows]
    if len(seeds) != len(set(seeds)) or not set(seeds).issubset(SEEDS):
        raise ValueError("Duplicate or unexpected validation seeds")
    if any(not passed(row) for row in rows):
        return "rejected"
    return "feasible" if set(seeds) == set(SEEDS) else "pending"


def save_report(out, candidates, rows, finished=False):
    points = []
    for candidate in candidates:
        group = [r for r in rows if r["checkpoint_sha256"] == candidate["sha256"]]
        status = candidate_status(group)
        point = {**candidate, "status": status, "evaluations": len(group),
                 "failure_seeds": [r["eval_seed"] for r in group if not passed(r)]}
        if status == "feasible":
            point["energy_mean_J"] = evaluation.describe([r["total_energy"] for r in group])["mean"]
            point["energy_sample_sd_J"] = evaluation.describe([r["total_energy"] for r in group])["std"]
        points.append(point)
    feasible = [p for p in points if p["status"] == "feasible"]
    winner = min(feasible, key=lambda p: (p["energy_mean_J"], p["episode"])) if feasible else None
    result = {"complete": finished, "selected": winner if finished else None,
              "best_so_far": winner, "candidates": points}
    (out / "selection.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# SS checkpoints：约束满足后最低能耗", "",
             "固定Small＋37台＋Medium Deadline；两块542000–542019、642000–642019各20种子。",
             "每个种子至少45/50条按时且可靠性全部达标；两块ESP@90必须均100%。",
             "某种子失败即筛除该候选，不把不完整评估的通过比例当作完整ESP。",
             "只比较通过全部40种子的候选能耗。无可行点不选替代模型、不放宽阈值。",
             "未读取1042000测试块或跨规模结果。没有改模型、λ、环境、频率或种子。", "",
             "状态：" + ("筛选完成" if finished else "筛选进行中；尚未锁定模型"), "",
             "| Episode | 状态 | 已评估种子 | 失败种子 | 可行能耗（J，均值 ± SD） |",
             "|---:|---|---:|---|---:|"]
    for p in points:
        energy = f"{p['energy_mean_J']:.2f} ± {p['energy_sample_sd_J']:.2f}" if p["status"] == "feasible" else "—"
        lines.append(f"| {p['episode']} | {p['status']} | {p['evaluations']}/40 | {p['failure_seeds']} | {energy} |")
    if finished:
        lines += ["", f"最终选定：ep{winner['episode']}，平均能耗{winner['energy_mean_J']:.2f} J。" if winner
                  else "最终结论：本次所有已保存候选均未通过，selected=null；没有可行checkpoint。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run, out = evaluation.resolve_path(args.run_dir), evaluation.resolve_path(args.output_dir)
    config_path = evaluation.resolve_path(args.config)
    config = evaluation.load_json(config_path)
    assert config["generator"]["dataset_scale"] == "small"
    assert config["generator"]["profile"] == "sec_job_fcfs_v1"
    assert config["generator"]["num_workflows"] == 50
    assert config["simulation"]["deadline_type"] == "medium"
    assert config["resources"]["num_edge"] == 28 and config["resources"]["num_cloud"] == 9
    paths = sorted(p for p in run.glob("checkpoint_ep*.pt") if "_high" not in p.name and "_low" not in p.name)
    paths += [run / "best_d3qn_checkpoint.pt"]
    candidates = []
    for path in paths:
        payload = evaluation.torch.load(path, map_location="cpu", weights_only=False)
        candidates.append({"path": str(path), "episode": int(payload["episode"]),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if len({p["episode"] for p in candidates}) != len(candidates):
        raise ValueError("Duplicate checkpoint episodes; resolve candidates explicitly")
    manifest = {"candidates": candidates, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "seeds": SEEDS, "evaluation_order": ORDER,
                "threshold": .90, "ESP_target": 1.0, "reliability_target": 1.0,
                "duplicate_final_latest": "ep1200 covered by periodic full checkpoint"}
    manifest = json.loads(json.dumps(manifest))
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Candidate/config/protocol changed; use a fresh output directory")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    rows_path = out / "comparison_rows.jsonl"
    rows = [json.loads(s) for s in rows_path.read_text(encoding="utf-8").splitlines()] if rows_path.exists() else []
    # Reuse the completed ep1176 audit, never its independent-test results.
    best = candidates[-1]
    provenance_path = out.parent / "checkpoint_provenance.json"
    if not any(r["checkpoint_sha256"] == best["sha256"] for r in rows) and provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        assert provenance["checkpoint_sha256"] == best["sha256"]
        assert provenance["config_sha256"] == manifest["config_sha256"]
        for seed in (542000, 642000):
            audit = out.parent / "source_validation" / f"block_{seed}" / "comparison_rows.jsonl"
            source_rows = [json.loads(s) for s in audit.read_text(encoding="utf-8").splitlines()]
            assert [r["eval_seed"] for r in source_rows] == list(range(seed, seed + 20))
            rows += [{**r, "checkpoint_sha256": best["sha256"], "checkpoint_episode": best["episode"]} for r in source_rows]
    evaluation.write_rows(out, rows)
    save_report(out, candidates, rows)
    evaluation.torch.set_num_threads(1)
    with redirect_stdout(io.StringIO()):
        resources = evaluation.build_resources(config, int(config["training"]["seed"]))
    for candidate in reversed(candidates):
        group = [r for r in rows if r["checkpoint_sha256"] == candidate["sha256"]]
        if candidate_status(group) != "pending":
            continue
        _, _, agent, objective = evaluation.load_d3qn(candidate["path"], 542000)
        seen = {r["eval_seed"] for r in group}
        for seed in ORDER:
            if seed in seen:
                continue
            with redirect_stdout(io.StringIO()):
                metrics = evaluation.run_d3qn_family(config, resources, agent, objective, seed, "d3qn")
            assert int(metrics["completed_workflow_count"]) == 50
            row = {"checkpoint_episode": candidate["episode"], "checkpoint_sha256": candidate["sha256"],
                   "eval_seed": seed, **metrics}
            rows.append(row)
            evaluation.write_rows(out, rows)
            save_report(out, candidates, rows)
            print(f"[candidate] ep={candidate['episode']} seed={seed} pass={passed(row)} "
                  f"E={metrics['total_energy']:.2f}J", flush=True)
            if not passed(row):
                break
        assert hashlib.sha256(Path(candidate["path"]).read_bytes()).hexdigest() == candidate["sha256"]
    winner = save_report(out, candidates, rows, finished=True)
    if winner:
        shutil.copy2(winner["path"], out / "selected_checkpoint.pt")
        assert hashlib.sha256((out / "selected_checkpoint.pt").read_bytes()).hexdigest() == winner["sha256"]
    print(f"[complete] selected_episode={winner['episode'] if winner else None}", flush=True)


if __name__ == "__main__":
    main()
