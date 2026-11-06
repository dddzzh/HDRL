"""Prepare, run, and report the explicitly conditional success-seed panel."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.evaluation_seed_schedule import validate_seed_schedule

ORIGINAL = ROOT / "results/legacy_train_new_eval_20261009"
OUTPUT = ROOT / "results/success_seed_screening_20261013"
SCALES = ["small", "medium", "large"]
DEADLINES = ["loose", "medium", "tight"]
MODES = ["d3qn_sac", "dqn_edge", "dream_dqn", "oo_actor_critic"]
LABELS = dict(zip(MODES, ["D3QN+SAC", "DQN-Edge", "DReaM-DQN", "OOAC"]))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_at(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def passed(row):
    return row["deadline_success_rate"] >= .9 - 1e-9 and row["reliability_success_rate"] >= 1 - 1e-9


def write_json_once(path, data):
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"Existing plan differs: {path}")
    else:
        path.write_text(payload, encoding="utf-8")


def prepare():
    OUTPUT.mkdir(exist_ok=True)
    candidates = {}
    candidate_hashes = {}
    for name in ("ss_37_dream_large_candidates", "ss_37_dream_medium_tight_candidates",
                 "ss_37_dream_large_tight_extra_candidates"):
        path = OUTPUT / name / "comparison_rows.jsonl"
        candidate_hashes[str(path)] = digest(path)
        for row in rows_at(path):
            if passed(row):
                key = (row["scale"], row["deadline_type"])
                candidates.setdefault(key, []).append(row["eval_seed"])
    for key, seeds in candidates.items():
        if len(seeds) != len(set(seeds)):
            raise ValueError("Duplicate candidate seed")
        seeds.sort()
    historical = read_json(ORIGINAL / "baselines/provenance.json")
    schedules, mappings, runs, original_hashes = {}, {}, [], {}
    for resource, count in (("ss", 37), ("sm", 74), ("sl", 111)):
        schedule = {s: {d: list(range(742000, 742020)) for d in DEADLINES} for s in SCALES}
        bad = {(s, d): set() for s in SCALES for d in DEADLINES}
        for mode in MODES:
            if resource == "ss" and mode != "d3qn_sac":
                source = ORIGINAL / "baselines" / mode
                p = next(m for m in historical["models"] if m["mode"] == mode)
                checkpoint = Path(p["checkpoint"])
                train_config = Path(p["training_config"])
                expected_cp = p["checkpoint_sha256"]
                expected_train = p["training_config_sha256"]
                expected_eval = historical["evaluation_config_sha256"]
                episode = p["saved_episode"]
            else:
                source = ORIGINAL / (f"{resource}_{count}" + ("" if mode == "d3qn_sac" else f"_{mode}"))
                p = read_json(source / "checkpoint_provenance.json")
                checkpoint = Path(p["source_checkpoint"])
                expected_cp = p["checkpoint_sha256"]
                expected_eval = p["config_sha256"]
                episode = p.get("global_episode", p["episode"])
                if mode == "d3qn_sac":
                    # The existing D3QN loader reads its config from this payload.
                    train_config, expected_train = checkpoint, expected_cp
                else:
                    train_config = Path(p["training_config"])
                    expected_train = p["training_config_sha256"]
            config = ROOT / f"configs/v3_{resource}_{count}_new_eval_50wf.json"
            for path, expected in ((checkpoint, expected_cp), (train_config, expected_train), (config, expected_eval)):
                if digest(path) != expected:
                    raise ValueError(f"Source changed: {path}")
            old_rows = rows_at(source / "comparison_rows.jsonl")
            expected_keys = {(s, d, n) for s in SCALES for d in DEADLINES for n in range(742000, 742020)}
            keys = [(r["scale"], r["deadline_type"], r["eval_seed"]) for r in old_rows]
            if len(keys) != 180 or set(keys) != expected_keys:
                raise ValueError(f"Original panel not complete: {source}")
            for r in old_rows:
                if r["mode"] != mode or r["num_workflows"] != 50:
                    raise ValueError("Original protocol differs")
                if not passed(r):
                    bad[r["scale"], r["deadline_type"]].add(r["eval_seed"])
            for path in source.glob("comparison_rows.*"):
                original_hashes[str(path)] = digest(path)
            runs.append(dict(id=f"{resource}_{count}_{mode}", resource=resource, servers=count, mode=mode,
                             source=str(source), checkpoint=str(checkpoint), checkpoint_sha256=expected_cp,
                             training_config=str(train_config), training_config_sha256=expected_train,
                             training_config_storage="checkpoint.config" if mode == "d3qn_sac" else "json",
                             config=str(config), config_sha256=expected_eval, source_episode=episode))
        mapping = []
        for (scale, deadline), failed in bad.items():
            pool = candidates.get((scale, deadline), [])
            if len(pool) < len(failed):
                raise ValueError(f"Insufficient candidates: {resource}/{scale}/{deadline}")
            replacements = dict(zip(sorted(failed), pool[:len(failed)]))
            schedule[scale][deadline] = [replacements.get(n, n) for n in range(742000, 742020)]
            for old, new in replacements.items():
                mapping.append(dict(scale=scale, deadline_type=deadline, old_seed=old, new_seed=new,
                                    candidate_basis="SS37 DReaM passed; all four models pending verification"))
        validate_seed_schedule(schedule, SCALES, DEADLINES, 20)
        path = OUTPUT / f"seed_schedule_{resource}_{count}.json"
        write_json_once(path, schedule)
        schedules[resource] = dict(path=str(path), sha256=digest(path), data=schedule)
        mappings[resource] = mapping
    plan = dict(selection_is_conditional=True, model_initialization_seed=742000,
                num_workflows=50, num_seeds=20, scales=SCALES, deadline_types=DEADLINES,
                schedules=schedules, mappings=mappings, runs=runs,
                candidate_source_hashes=candidate_hashes, original_file_sha256=original_hashes)
    write_json_once(OUTPUT / "panel_plan.json", plan)
    print(f"Prepared {len(runs)} fixed model runs; replacements by resource:",
          {r: len(m) for r, m in mappings.items()})
    return plan


def check_sources(plan, run=None):
    pairs = dict(plan["original_file_sha256"])
    pairs.update(plan["candidate_source_hashes"])
    selected = [run] if run else plan["runs"]
    for r in selected:
        pairs.update({r["checkpoint"]: r["checkpoint_sha256"],
                      r["training_config"]: r["training_config_sha256"],
                      r["config"]: r["config_sha256"]})
    pairs.update({s["path"]: s["sha256"] for s in plan["schedules"].values()})
    for path, expected in pairs.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Source or schedule changed: {path}")


def validate_rows(rows, run, schedule, complete=False):
    expected = [(s, d, seed) for s in SCALES for d in DEADLINES for seed in schedule[s][d]]
    keys = [(r["scale"], r["deadline_type"], r["eval_seed"]) for r in rows]
    if keys != expected[:len(keys)] or len(keys) > 180 or (complete and len(keys) != 180):
        raise ValueError("Panel records differ from the explicit ordered seed schedule")
    for r in rows:
        metrics = [r[k] for k in ("total_energy", "deadline_success_rate", "reliability_success_rate")]
        if (r["mode"] != run["mode"] or r["num_workflows"] != 50
                or not all(math.isfinite(v) for v in metrics)
                or metrics[0] <= 0 or not all(0 <= v <= 1 for v in metrics[1:])):
            raise ValueError("Invalid panel metrics or protocol")


def run_one(run_id):
    from tools import eval_all_baselines as evaluation
    from unittest.mock import patch
    plan = read_json(OUTPUT / "panel_plan.json")
    run = next(r for r in plan["runs"] if r["id"] == run_id)
    check_sources(plan, run)
    schedule = plan["schedules"][run["resource"]]
    out = OUTPUT / "paired_panel" / run_id
    if (out / "comparison_rows.jsonl").exists():
        raise ValueError("Do not restart or merge an existing panel run")
    out.mkdir(parents=True, exist_ok=True)
    write_json_once(out / "run_provenance.json",
                    dict(run=run, schedule_sha256=schedule["sha256"], conditional=True,
                         model_initialization_seed=742000, workers=1))
    prefix = {"d3qn_sac": "d3qn", "dream_dqn": "dream", "dqn_edge": "dqn-edge", "oo_actor_critic": "oo"}[run["mode"]]
    argv = ["eval_all_baselines", "--config", run["config"], f"--{prefix}-checkpoint", run["checkpoint"],
            "--modes", run["mode"], "--scales", ",".join(SCALES),
            "--deadline-types", ",".join(DEADLINES), "--seed-start", "742000",
            "--num-seeds", "20", "--seed-schedule", schedule["path"], "--workers", "1",
            "--output-dir", str(out)]
    if run["mode"] != "d3qn_sac":
        argv += [f"--{prefix}-config", run["training_config"]]
    with patch.object(sys, "argv", argv):
        args = evaluation.parse_args()
    evaluation.torch.set_num_threads(1)
    original_writer = evaluation.write_rows

    def checked_writer(directory, rows):
        validate_rows(rows, run, schedule["data"])
        original_writer(directory, rows)

    evaluation.write_rows = checked_writer
    try:
        _, rows, _ = evaluation.evaluate(args)
        validate_rows(rows, run, schedule["data"], complete=True)
        check_sources(plan, run)
        failures = [dict(scale=r["scale"], deadline_type=r["deadline_type"], eval_seed=r["eval_seed"])
                    for r in rows if not passed(r)]
        write_json_once(out / "completion.json", dict(records=180, source_hashes_verified=True, failures=failures))
        print(f"[panel-complete] id={run_id} records=180 failures={len(failures)}", flush=True)
    finally:
        evaluation.write_rows = original_writer


def report():
    from tools.eval_v3_scenarios import refresh, result_cells
    plan = read_json(OUTPUT / "panel_plan.json")
    check_sources(plan)
    lines = ["## 成功种子筛选后的完整配对评估（条件样本，非总体成功率）", "",
             "原始未筛选数据全部保留在下方原始表与CSV中。新名单按原始失败取并集，同场景四法共用20个唯一真实种子。",
             "原成功位置保留，失败位置使用真实新seed；所有模型从初始化seed742000开始完整重跑，DQN-Edge保持跨序列EMA，未拼接旧行。",
             "部分进度不是最终结果；新序列仍失败时保留实际失败，不把它改成100%。", "",
             "[真实seed名单、旧→新映射及12模型身份](../success_seed_screening_20261013/panel_plan.json)。", ""]
    all_complete, failures = True, []
    for resource, count in (("ss", 37), ("sm", 74), ("sl", 111)):
        lines += [f"### 新{resource.upper()}配对评估：{count}台服务器", "",
                  "| 场景 | Deadline | 方法 | 能耗（J，均值±样本SD） | ESP@90 | 可靠性达标 | 状态 |",
                  "|---|---|---|---:|---:|---:|---|"]
        data = {}
        for run in (r for r in plan["runs"] if r["resource"] == resource):
            out = OUTPUT / "paired_panel" / run["id"]
            path = out / "comparison_rows.jsonl"
            rows = rows_at(path) if path.exists() else []
            validate_rows(rows, run, plan["schedules"][resource]["data"])
            done = (out / "completion.json").exists()
            all_complete &= done
            if done:
                validate_rows(rows, run, plan["schedules"][resource]["data"], complete=True)
                failures.extend(dict(run=run["id"], scale=r["scale"], deadline_type=r["deadline_type"],
                                     eval_seed=r["eval_seed"]) for r in rows if not passed(r))
            data[run["mode"]] = rows
        for scale in SCALES:
            for deadline in DEADLINES:
                for mode in MODES:
                    group = [r for r in data[mode] if r["scale"] == scale and r["deadline_type"] == deadline]
                    cells = result_cells(group, 20)
                    lines.append(f"| {scale[0].upper() + resource[1].upper()} | {deadline.title()} | {LABELS[mode]} | {cells} |")
        lines += [""]
    lines += [f"整体完整核验：{'完成' if all_complete else '未完成'}；已完成模型中的未通过记录：{len(failures)}。", "",
              "## 原始未筛选评估结果（保留与恢复入口）", ""]
    (ORIGINAL / "SUCCESS_SEED_REPLACEMENT.md").write_text("\n".join(lines), encoding="utf-8")
    refresh(ORIGINAL)
    print(json.dumps(dict(all_complete=all_complete, failures=failures), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prepare", action="store_true")
    choice.add_argument("--run")
    choice.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        run_one(args.run)
    else:
        report()


if __name__ == "__main__":
    main()
