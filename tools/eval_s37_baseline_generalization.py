"""Evaluate three frozen S37 baselines using the existing evaluator, one process each."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import eval_all_baselines as evaluation
from tools.eval_v3_scenarios import result_cells, refresh, scenario_results_complete

OUT = ROOT / "results/legacy_train_new_eval_20261009"
CONFIG = ROOT / "configs/v3_ss_37_new_eval_50wf.json"
SPECS = (
    ("dqn_edge", "DQN-Edge", "dqn-edge", "dqn_edge_small_28edge_9cloud", "best_checkpoint.pt", 249),
    ("dream_dqn", "DReaM-DQN", "dream", "dream_dqn_small_28edge_9cloud", "checkpoint_ep0250.pt", 249),
    ("oo_actor_critic", "OOAC", "oo", "oo_actor_critic_small_28edge_9cloud", "best_oo_actor_critic_checkpoint.pt", 900),
)
SCALES = ("small", "medium", "large")
REGIMES = ("loose", "medium", "tight")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_rows(rows, mode, *, complete=True):
    expected = {(scale, ddl, seed) for scale in SCALES for ddl in REGIMES for seed in range(742000, 742020)}
    keys = [(r["scale"], r["deadline_type"], r["eval_seed"]) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected or (complete and set(keys) != expected):
        raise ValueError("Missing or duplicate seed/condition results")
    for row in rows:
        if row["mode"] != mode or row["num_workflows"] != 50:
            raise ValueError("Wrong model or workflow count")
        for field in ("total_energy", "deadline_success_rate", "reliability_success_rate"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Nonfinite metric: {field}")
        if row["total_energy"] <= 0 or not 0 <= row["deadline_success_rate"] <= 1 or not 0 <= row["reliability_success_rate"] <= 1:
            raise ValueError("Invalid evaluation metrics")


def read_rows(directory):
    path = directory / "comparison_rows.jsonl"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    # An active evaluator may still be appending the final line.
    lines = text.splitlines() if text.endswith("\n") else text.splitlines()[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def write_report(completed, failed):
    ours = read_rows(OUT / "ss_37")
    validate_rows(ours, "d3qn_sac")
    models = [("D3QN+SAC", ours, "完成")]
    for mode, label, *_ in SPECS:
        data = read_rows(OUT / "baselines" / mode)
        if data:
            validate_rows(data, mode, complete=mode in completed)
        models.append((label, data, "评估失败，见日志" if mode in failed else "评估中"))
    lines = ["# S37固定模型：旧协议训练 → 新协议泛化对比", "",
             "所有方法固定37台服务器，50工作流/种子，测试种子742000–742019。",
             "能耗为J、均值 ± 样本标准差；ESP@90要求每个种子至少45/50按时完成。",
             "可靠性达标为所有工作流满足0.9要求的种子比例。固定模型，不重训、不重选、不替换失败种子。",
             "DReaM使用历史验证块选出的ep250回退模型；当时未达到约束，不冒充可行最优模型。",
             "模型来源与SHA256：[provenance.json](baselines/provenance.json)。", "",
             "| 场景 | Deadline | 方法 | 能耗（J） | ESP@90 | 可靠性达标 | 状态 |",
             "|---|---|---|---:|---:|---:|---|"]
    for case, scale in zip(("SS", "MS", "LS"), SCALES):
        for ddl in REGIMES:
            for label, data, status in models:
                group = [r for r in data if r["scale"] == scale and r["deadline_type"] == ddl]
                cells = result_cells(group, 20) if len(group) == 20 else f"— | — | — | {status} {len(group)}/20"
                lines.append(f"| {case} | {ddl.title()} | {label} | {cells} |")
    lines += ["", "未达标方法的低能耗不能直接解释为约束满足下的节能优势；重复使用该测试块不构成新的独立确认。"]
    path = OUT / "S37_BASELINE_GENERALIZATION.md"
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    evaluation.torch.set_num_threads(1)
    config = evaluation.load_json(CONFIG)
    assert config["generator"]["profile"] == "sec_job_fcfs_v1" and config["generator"]["num_workflows"] == 50
    with redirect_stdout(io.StringIO()):
        resources = evaluation.build_resources(config, 42)
    assert len(resources["servers"]) == 37
    provenance = {"evaluation_config": str(CONFIG), "evaluation_config_sha256": digest(CONFIG), "models": []}
    jobs = []
    for mode, label, flag, run, filename, episode in SPECS:
        source = ROOT / "results" / run / "20260924_160451"
        checkpoint, training_config = source / filename, source / "resolved_config.json"
        training = evaluation.load_json(training_config)
        assert training["generator"]["arrival_rate"] == .1 and training["generator"]["num_workflows"] == 20
        assert training["generator"]["arrival_process"] == "cai_2025_exponential"
        assert training["generator"].get("profile", "cai_2025_heft") == "cai_2025_heft"
        assert training["generator"]["dataset_scale"] == "small" and training["simulation"]["deadline_type"] == "medium"
        assert training["resources"]["num_edge"] == 28 and training["resources"]["num_cloud"] == 9
        payload = evaluation.torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["episode"] == episode, (label, payload["episode"])
        del payload
        item = dict(mode=mode, label=label, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                    saved_episode=episode, training_config=str(training_config), training_config_sha256=digest(training_config))
        provenance["models"].append(item)
        directory = OUT / "baselines" / mode
        directory.mkdir(parents=True, exist_ok=True)
        jobs.append((mode, checkpoint, directory, [sys.executable, "-u", str(ROOT / "tools/eval_all_baselines.py"),
                     "--config", str(CONFIG), f"--{flag}-config", str(training_config),
                     f"--{flag}-checkpoint", str(checkpoint), "--modes", mode,
                     "--scales", "small,medium,large", "--deadline-types", "loose,medium,tight",
                     "--seed-start", "742000", "--num-seeds", "20", "--output-dir", str(directory)]))
    path = OUT / "baselines/provenance.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != provenance:
        raise ValueError("Evaluation provenance changed; do not reuse these outputs")
    path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    # Reuse loaders and rollouts to check all three frozen state/action encoders before the full sweep.
    smoke = copy.deepcopy(config)
    smoke["generator"]["num_workflows"] = 2
    with redirect_stdout(io.StringIO()):
        for mode, checkpoint, _, _ in jobs:
            training = evaluation.load_json(checkpoint.parent / "resolved_config.json")
            if mode == "dqn_edge":
                _, runtime = evaluation.load_dqn_edge(training, checkpoint, resources, 742000)
                metrics = evaluation.run_dqn_edge(smoke, resources, runtime, 742000)
                del runtime
            elif mode == "dream_dqn":
                _, scheduler = evaluation.load_dream(training, checkpoint, resources, 742000)
                metrics = evaluation.run_dream(smoke, resources, scheduler, 742000)
                del scheduler
            else:
                _, encoder, agent, objective, weight, _ = evaluation.load_oo(training, checkpoint, resources, 742000)
                metrics = evaluation.run_oo(smoke, resources, encoder, agent, objective, weight, 742000)
                del agent
            assert math.isfinite(metrics["total_energy"]) and metrics["total_energy"] > 0
    print("Compatibility rollouts passed for all three S37 baselines.", flush=True)

    def run(job):
        mode, checkpoint, directory, command = job
        with (directory / "evaluation.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        assert digest(checkpoint) == next(m["checkpoint_sha256"] for m in provenance["models"] if m["mode"] == mode)
        if result.returncode:
            return mode, False
        validate_rows(read_rows(directory), mode)
        return mode, True

    completed, failed = set(), set()
    write_report(completed, failed)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run, job): job[0] for job in jobs}
        for future in as_completed(futures):
            mode = futures[future]
            try:
                _, success = future.result()
            except Exception as error:
                print(f"Failed {mode}: {error}", flush=True)
                success = False
            (completed if success else failed).add(mode)
            write_report(completed, failed)
            print(f"{mode}: {'complete' if success else 'failed'}; completed={len(completed)}/3", flush=True)
    if failed:
        raise RuntimeError(f"Incomplete baseline sweep: {sorted(failed)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--refresh-only"]:
        while True:
            write_report(set(), set())
            refresh(OUT)
            if (all(len(read_rows(OUT / "baselines" / mode)) == 180 for mode, *_ in SPECS)
                    and scenario_results_complete(OUT)):
                break
            time.sleep(2)
    else:
        main()
