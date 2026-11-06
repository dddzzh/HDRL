"""Diagnostic, target-validation-assisted SS selection; never reads test results."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEED_START = 1142000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(rows, scales, seeds):
    expected = {(scale, "tight", seed) for scale in scales for seed in seeds}
    keys = [(r["scale"], r["deadline_type"], r["eval_seed"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Missing, duplicate or unexpected scenario/seed results")
    if any(r["num_workflows"] != 50 or r["completed_workflow_count"] != 50
           or r["mode"] != "d3qn_sac" for r in rows):
        raise ValueError("Expected complete 50-workflow D3QN+SAC rollouts")
    groups = {}
    for scale in scales:
        group = [r for r in rows if r["scale"] == scale]
        groups[scale] = {
            "n": len(group),
            "esp90": sum(r["deadline_success_rate"] >= .9 - 1e-9 for r in group) / len(group),
            "rel_esp": sum(r["reliability_success_rate"] >= 1 - 1e-9 for r in group) / len(group),
            "energy_J": statistics.mean(r["total_energy"] for r in group),
            "energy_sd_J": statistics.stdev(r["total_energy"] for r in group),
        }
    return {"groups": groups, "min_esp90": min(g["esp90"] for g in groups.values()),
            "min_rel_esp": min(g["rel_esp"] for g in groups.values()),
            "worst_seed_sr": min(r["deadline_success_rate"] for r in rows),
            "energy_J": statistics.mean(r["total_energy"] for r in rows)}


def rank(point):
    feasible = point["min_esp90"] >= 1 - 1e-9 and point["min_rel_esp"] >= 1 - 1e-9
    return (not point["small_validation_feasible"], point["min_rel_esp"] < 1 - 1e-9,
            -point["min_esp90"], 0 if feasible else -point["worst_seed_sr"],
            point["energy_J"], point["episode"])


def write_report(out, points, phase, complete=False):
    ranked = sorted(points, key=rank)
    result = {"phase": phase, "complete": complete, "points": ranked,
              "selected": ranked[0] if complete and ranked else None,
              "selected_feasible": (bool(ranked[0]["small_validation_feasible"]
                                         and ranked[0]["min_esp90"] == 1
                                         and ranked[0]["min_rel_esp"] == 1)
                                    if complete and ranked else None)}
    temporary = out / "selection.tmp"
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(out / "selection.json")
    lines = ["# SS checkpoint泛化验证筛选", "",
             "50工作流；37台服务器；新sec-job/FCFS协议；Tight Deadline。",
             "原始模型不变，742000及1042000测试块不参与筛选。",
             "这是目标分布验证参与选点，不是完全未见Large的零样本选点协议。",
             "初筛：全部ep500–1200，每20代一份，共36份；1142000–1142002三个Large验证种子。",
             "复评：初筛中Small原验证达标的前三名，Medium/Tight和Large/Tight各20种子1142000–1142019。",
             "排序：Small原验证可行优先、可靠性全达标优先、最小ESP最高；",
             "不可行并列以最差种子按时率辅助排序，可行并列按均值能耗最低。",
             "前三名复评不保证找出全部36份在完整20种子上的全局最优；未复评的模型不宣称完整ESP。",
             "不改正式best模型、不删换种子、不掩盖无可行点。", "",
             f"阶段：{phase}；已完成候选：{len(points)}；完整结束：{complete}", "",
             "| ep | 原Small验证达标 | min ESP@90 | min可靠性ESP | 能耗均值（J） |",
             "|---:|---|---:|---:|---:|"]
    for p in ranked:
        lines.append(f"| {p['episode']} | {p['small_validation_feasible']} | {p['min_esp90']:.1%} | "
                     f"{p['min_rel_esp']:.1%} | {p['energy_J']:.2f} |")
    if complete and ranked:
        winner = ranked[0]
        feasible = winner["small_validation_feasible"] and winner["min_esp90"] == 1 and winner["min_rel_esp"] == 1
        lines += ["", f"诊断选定ep{winner['episode']}，完整可行：{feasible}。",
                  f"模型：`{winner['path']}`；SHA256：`{winner['sha256']}`。",
                  "不自动替换正式best；最终独立测试需另行预先固定未查看的种子块。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(candidate, out, config, phase, scales, count):
    case = out / phase / f"ep{candidate['episode']:06d}"
    if digest(candidate["path"]) != candidate["sha256"]:
        raise ValueError("Source checkpoint changed")
    command = [sys.executable, "-u", str(ROOT / "tools/eval_all_baselines.py"),
               "--config", str(config), "--d3qn-checkpoint", candidate["path"],
               "--modes", "d3qn_sac", "--scales", ",".join(scales), "--deadline-types", "tight",
               "--seed-start", str(SEED_START), "--num-seeds", str(count),
               "--workers", "3" if phase == "refine" else "1", "--output-dir", str(case)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUTF8": "1"}
    with (case / "evaluation.log").open("a", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest = json.loads((case / "experiment_manifest.json").read_text(encoding="utf-8"))
    if manifest["d3qn_checkpoint_episode"] != candidate["episode"]:
        raise ValueError("Filename/payload episode mismatch")
    rows = [json.loads(s) for s in (case / "comparison_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    point = {**candidate, **summarize(rows, scales, range(SEED_START, SEED_START + count))}
    if digest(candidate["path"]) != candidate["sha256"]:
        raise ValueError("Source checkpoint changed during evaluation")
    return point


def stage(candidates, out, config, phase, scales, count, workers):
    points = []
    write_report(out, points, phase)
    # Prepare directories before threads to avoid concurrent parent creation.
    (out / phase).mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        (out / phase / f"ep{candidate['episode']:06d}").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, c, out, config, phase, scales, count) for c in candidates]
        for future in as_completed(futures):
            try:
                point = future.result()
            except Exception as error:
                print(f"[error] {phase}: {type(error).__name__}: {error}", flush=True)
                for pending in futures:
                    pending.cancel()
                raise
            points.append(point)
            write_report(out, points, phase)
            print(f"[{phase}] ep={point['episode']} done={len(points)}/{len(candidates)} "
                  f"min_ESP90={point['min_esp90']:.3f} E={point['energy_J']:.2f}J", flush=True)
    (out / f"{phase}_points.json").write_text(json.dumps(sorted(points, key=rank), indent=2), encoding="utf-8")
    return sorted(points, key=rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run, source, out = (Path(p).resolve() for p in (args.run_dir, args.config, args.output_dir))
    config = json.loads(source.read_text(encoding="utf-8"))
    if config["generator"]["num_workflows"] != 50 or config["generator"]["profile"] != "sec_job_fcfs_v1":
        raise ValueError("This protocol requires50-workflow sec_job_fcfs_v1")
    history = [json.loads(s) for s in (run / "eval_history.jsonl").read_text(encoding="utf-8").splitlines()]
    candidates = []
    for path in sorted(run.glob("checkpoint_ep*.pt")):
        number = path.stem.removeprefix("checkpoint_ep")
        if not number.isdigit() or not 500 <= int(number) <= 1200:
            continue
        blocks = [r for r in history if r["episode"] == int(number)]
        own_ok = (len(blocks) == 2 and {r["eval_seed"] for r in blocks} == {542000, 642000}
                  and all(r["deadline_esp_0p90"] >= 1 - 1e-9 and r["reliability_esp_1p00"] >= 1 - 1e-9 for r in blocks))
        candidates.append({"episode": int(number), "path": str(path), "sha256": digest(path),
                           "small_validation_feasible": own_ok})
    if [c["episode"] for c in candidates] != list(range(500, 1201, 20)):
        raise ValueError("Expected all36 saved full checkpoints")
    out.mkdir(parents=True, exist_ok=True)
    protocol = {"candidates": candidates, "source_config_sha256": digest(source),
                "history_sha256": digest(run / "eval_history.jsonl"), "seed_start": SEED_START,
                "screen_seeds": 3, "refine_seeds": 20, "num_workflows": 50,
                "screen_scales": ["large"], "refine_scales": ["medium", "large"], "shortlist": 3}
    manifest = out / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Resume protocol changed; use a fresh output directory")
    manifest.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    snapshot = out / "eval_config.json"
    if snapshot.exists() and digest(snapshot) != digest(source):
        raise ValueError("Evaluation config snapshot changed")
    if not snapshot.exists():
        shutil.copy2(source, snapshot)
    screen = stage(candidates, out, snapshot, "screen", ["large"], 3, 8)
    shortlist = [c for c in screen if c["small_validation_feasible"]][:3]
    if not shortlist:
        raise ValueError("No candidate passed original Small validation")
    refine = stage(shortlist, out, snapshot, "refine", ["medium", "large"], 20, 3)
    write_report(out, refine, "refine", complete=True)
    print(f"[complete] diagnostic_selected_ep={refine[0]['episode']}", flush=True)


if __name__ == "__main__":
    main()
