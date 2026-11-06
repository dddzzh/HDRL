"""Reuse the existing evaluator and refresh a separate V3 scenario summary."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import eval_all_baselines as evaluation

BASELINES = {"dqn_edge": ("DQN-Edge", "dqn_edge"),
             "dream_dqn": ("DReaM-DQN", "dream"),
             "oo_actor_critic": ("OOAC", "oo")}


def validate_results(rows, provenance, num_workflows, *, complete=False):
    expected = {(s, d, n) for s in provenance["scales"].split(",")
                for d in provenance["deadline_types"].split(",")
                for n in range(provenance["seed_start"], provenance["seed_start"] + provenance["num_seeds"])}
    keys = [(r["scale"], r["deadline_type"], r["eval_seed"]) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected or (complete and set(keys) != expected):
        raise ValueError("Missing, duplicate or unexpected evaluation rows")
    for row in rows:
        if row["mode"] != provenance.get("mode", "d3qn_sac") or row["num_workflows"] != num_workflows:
            raise ValueError("Wrong model or workflow count")
        values = [float(row[k]) for k in ("total_energy", "deadline_success_rate", "reliability_success_rate")]
        if not all(math.isfinite(v) for v in values) or values[0] <= 0 or not all(0 <= v <= 1 for v in values[1:]):
            raise ValueError("Invalid evaluation metrics")


def result_cells(rows, expected):
    if not rows:
        return "— | — | — | 待评估"
    seeds = [r["eval_seed"] for r in rows]
    if len(seeds) != len(set(seeds)) or len(rows) > expected:
        raise ValueError("Duplicate seeds or more results than requested")
    energy = evaluation.describe([r["total_energy"] for r in rows])
    sd = f"{energy['std']:.2f}" if len(rows) > 1 else "—"
    ddl = sum(r["deadline_success_rate"] >= .9 - 1e-9 for r in rows)
    rel = sum(r["reliability_success_rate"] >= 1 - 1e-9 for r in rows)
    status = "完成" if len(rows) == expected else f"进行中 {len(rows)}/{expected}，非最终结果"
    return (f"{energy['mean']:.2f} ± {sd} | {ddl / len(rows):.0%} ({ddl}/{len(rows)}) | "
            f"{rel / len(rows):.0%} ({rel}/{len(rows)}) | {status}")


def scenario_results_complete(root):
    paths = list(root.glob("*/checkpoint_provenance.json"))
    for path in paths:
        provenance = json.loads(path.read_text(encoding="utf-8"))
        raw = path.parent / "comparison_rows.jsonl"
        rows = [json.loads(s) for s in raw.read_text(encoding="utf-8").splitlines()] if raw.exists() else []
        validate_results(rows, provenance, 50)
        expected = provenance["num_seeds"] * len(provenance["scales"].split(",")) * len(provenance["deadline_types"].split(","))
        if len(rows) != expected:
            return False
    return bool(paths)


def refresh(root):
    lines = ["# 固定模型泛化：场景汇总", "",
             "每个模型训练于 Small 工作流 + Medium Deadline；泛化时服务器数量不变。",
             "能耗：J，均值 ± 样本标准差。ESP@90：每个种子至少45/50工作流按时完成的比例。",
             "可靠性：每个种子所有工作流达到可靠性要求0.9的比例，不是物理可靠性等于1。",
             "checkpoint筛选协议以来源run的训练配置和验证日志为准；此表只评估固定模型，不重选checkpoint。",
             "测试后不重选模型、不替换失败种子。不同模型或生成协议的结果不得混作同一行。", ""]
    replacement_report = root / "SUCCESS_SEED_REPLACEMENT.md"
    if replacement_report.exists():
        lines += replacement_report.read_text(encoding="utf-8").splitlines() + [""]
    alibaba_report = root / "ALIBABA_SL111_100WF.md"
    if alibaba_report.exists():
        lines += alibaba_report.read_text(encoding="utf-8").splitlines() + [""]
    if (root / "PROTOCOL.md").exists():
        lines += ["训练与评估协议、模型身份及数据来源见[本次评估协议](PROTOCOL.md)。", ""]
    if (root / "S37_BASELINE_GENERALIZATION.md").exists():
        lines += ["37台历史对比模型的完整九档结果已收录在本文末尾；原独立表仍保留：[四方法配对对比汇总](S37_BASELINE_GENERALIZATION.md)。", "",
                  "注意：37台模型的 MS/LS 泛化结果不等于74台 SM模型的 MM/LM结果，不跨资源规模混用。", ""]
    for model, count in (("ss", 37), ("sm", 74), ("sl", 111)):
        out = root / f"{model}_{count}"
        lines += [f"## {model.upper()} 模型：{count}台服务器", ""]
        provenance_path = out / "checkpoint_provenance.json"
        rows_path = out / "comparison_rows.jsonl"
        if provenance_path.exists():
            p = json.loads(provenance_path.read_text(encoding="utf-8"))
            lines += [f"固定模型：ep{p.get('global_episode', p['episode'])}，来源 `{p['source_checkpoint']}`。",
                      f"SHA256：`{p['checkpoint_sha256']}`。",
                      f"测试种子：{p['seed_start']}–{p['seed_start'] + p['num_seeds'] - 1}。",
                      f"逐种子原始数据：[CSV]({model}_{count}/comparison_rows.csv)。", ""]
        rows = [json.loads(s) for s in rows_path.read_text(encoding="utf-8").splitlines()] if rows_path.exists() else []
        expected = p["num_seeds"] if provenance_path.exists() else 20
        lines += ["| 场景 | 工作流 | Deadline | 能耗（J） | ESP@90 | 可靠性达标 | 状态 |",
                  "|---|---|---|---:|---:|---:|---|"]
        for name, scale in (("Small", "small"), ("Medium", "medium"), ("Large", "large")):
            case = name[0] + model[1].upper()
            for regime in ("loose", "medium", "tight"):
                group = [r for r in rows if r["scale"] == scale and r["deadline_type"] == regime]
                lines.append(f"| {case} | {name} | {regime.title()} | {result_cells(group, expected)} |")
        lines += [""]
        comparisons = []
        for mode, (label, _) in BASELINES.items():
            directory = root / f"{model}_{count}_{mode}"
            path = directory / "checkpoint_provenance.json"
            if not path.exists():
                continue
            bp = json.loads(path.read_text(encoding="utf-8"))
            if not provenance_path.exists() or any(bp[k] != p[k] for k in
                    ("seed_start", "num_seeds", "scales", "deadline_types", "config_sha256")):
                raise ValueError("Baseline and ours evaluation protocols differ")
            raw_path = directory / "comparison_rows.jsonl"
            raw = [json.loads(s) for s in raw_path.read_text(encoding="utf-8").splitlines()] if raw_path.exists() else []
            validate_results(raw, bp, 50)
            comparisons.append((label, raw))
            lines += [f"{label}：ep{bp['episode']}，来源 `{bp['source_checkpoint']}`；",
                      f"SHA256：`{bp['checkpoint_sha256']}`；[逐种子CSV]({directory.name}/comparison_rows.csv)。", ""]
        if comparisons:
            lines += [f"### {model.upper()} 配对对比（相同服务器、生成协议及测试种子）", "",
                      "| 场景 | Deadline | 方法 | 能耗（J） | ESP@90 | 可靠性达标 | 状态 |",
                      "|---|---|---|---:|---:|---:|---|"]
            for name, scale in (("Small", "small"), ("Medium", "medium"), ("Large", "large")):
                for regime in ("loose", "medium", "tight"):
                    for label, data in [("D3QN+SAC", rows)] + comparisons:
                        group = [r for r in data if r["scale"] == scale and r["deadline_type"] == regime]
                        lines.append(f"| {name[0] + model[1].upper()} | {regime.title()} | {label} | {result_cells(group, expected)} |")
            lines += [""]
    baseline_report = root / "S37_BASELINE_GENERALIZATION.md"
    if baseline_report.exists():
        lines += ["## 37台服务器：本文与三个对比算法的完整配对结果", "",
                  *baseline_report.read_text(encoding="utf-8").splitlines()[1:], ""]
    path = root / "SCENARIO_MATRIX.md"
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def main():
    args = evaluation.parse_args()
    if args.modes not in {"d3qn_sac", *BASELINES} or not args.output_dir or not args.deadline_types:
        raise ValueError("Specify one supported model, --deadline-types and --output-dir")
    out = evaluation.resolve_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "d3qn" if args.modes == "d3qn_sac" else BASELINES[args.modes][1]
    checkpoint = evaluation.resolve_path(getattr(args, prefix + "_checkpoint"))
    payload = evaluation.torch.load(checkpoint, map_location="cpu", weights_only=False)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    provenance = {"episode": payload["episode"], "source_checkpoint": str(checkpoint),
                  "checkpoint_sha256": digest, "seed_start": args.seed_start,
                  "num_seeds": args.num_seeds, "scales": args.scales,
                  "deadline_types": args.deadline_types,
                  "config_sha256": hashlib.sha256(evaluation.resolve_path(args.config).read_bytes()).hexdigest()}
    offset = int(payload.get("config", {}).get("training", {}).get("episode_index_offset", 0))
    if offset:
        provenance["global_episode"] = int(payload["episode"]) + offset
    if args.modes != "d3qn_sac":
        training_path = evaluation.resolve_path(getattr(args, prefix + "_config"))
        provenance.update(mode=args.modes, training_config=str(training_path),
                          training_config_sha256=hashlib.sha256(training_path.read_bytes()).hexdigest())
    path = out / "checkpoint_provenance.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != provenance:
        raise ValueError("Resume provenance differs; use a fresh output directory")
    if not path.exists() and (out / "comparison_rows.jsonl").exists():
        raise ValueError("Existing results have no verified provenance")
    config = evaluation.load_json(args.config)
    num_workflows = int(config["generator"]["num_workflows"])
    raw_path = out / "comparison_rows.jsonl"
    if raw_path.exists():
        validate_results([json.loads(s) for s in raw_path.read_text(encoding="utf-8").splitlines()], provenance, num_workflows)
    path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    refresh(out.parent)
    original_write = evaluation.write_rows

    def write_and_refresh(directory, rows):
        validate_results(rows, provenance, num_workflows)
        original_write(directory, rows)
        refresh(out.parent)

    evaluation.write_rows = write_and_refresh
    evaluation.torch.set_num_threads(1)
    try:
        evaluation.evaluate(args)
        validate_results([json.loads(s) for s in raw_path.read_text(encoding="utf-8").splitlines()],
                         provenance, num_workflows, complete=True)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
        assert hashlib.sha256(evaluation.resolve_path(args.config).read_bytes()).hexdigest() == provenance["config_sha256"]
    finally:
        evaluation.write_rows = original_write
        refresh(out.parent)


if __name__ == "__main__":
    main()
