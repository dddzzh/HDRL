"""Separate Full-model budget/deadline-reference pilots; original files unchanged."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Production configures dependency paths/CPU threads before importing learners.
from tools import train_hierarchical_d3qn as production
from tools import train_coupled_hierarchical_d3qn_sac as coupled
from tools.eval_alibaba_sl111 import digest, identity
from src.datasets.shared_heft_deadlines import assign_shared_heft_deadlines

BASE_CONFIG = ROOT / "configs/v3_sm_74_legacy_train.json"
MODES = ("isolated_heft", "shared_heft")
OUTPUT_ROOT = "results/deadline_reference_sm74_budget001"


def validate_config(config):
    """Freeze this two-factor pilot; reject accidental training/protocol changes."""
    expected = production.load_json(BASE_CONFIG)
    mode = config.get("deadline_reference_experiment", {}).get("training_reference")
    if mode not in MODES:
        raise ValueError(f"training_reference must be one of {MODES}")
    expected["agent"]["low"]["lagrangian"]["deadline_budget"] = 0.01
    expected["training"]["checkpoint_dir"] = f"{OUTPUT_ROOT}/{mode}"
    expected["deadline_reference_experiment"] = {
        "training_reference": mode,
        "validation_reference": "unchanged_legacy_isolated_heft",
        "reference_rho": 1.0,
    }
    if config != expected:
        raise ValueError("Pilot config must equal the legacy SM config except budget, output and experiment metadata.")
    production.validate_training_config(config["training"])
    return mode


@contextmanager
def installed_reference(mode, input_sink=None):
    """Process-local training hook; serial and spawned validation stay original."""
    if mode not in MODES:
        raise ValueError(f"Unknown reference: {mode}")
    original_build = production.build_workflows
    original_evaluate = production.evaluate_agent

    def build_training(config, resources, seed, smoke_test=False):
        workflows = original_build(config, resources, seed, smoke_test=smoke_test)
        before = identity(workflows)
        deadlines = []
        if mode == "shared_heft":
            workflows, deadlines = assign_shared_heft_deadlines(
                workflows, resources, config["simulation"], seed,
            )
        after = identity(workflows)
        if before["physical_sha256"] != after["physical_sha256"]:
            raise RuntimeError("Changing reference changed physical workflow inputs.")
        if input_sink is not None:
            input_sink({"seed": int(seed), "training_reference": mode,
                        **after, "legacy_deadlines_sha256": before["deadlines_sha256"],
                        "deadline_reference_rows": deadlines})
        return workflows

    def evaluate_unchanged(*args, **kwargs):
        # Spawn workers import unchanged production code. Serial validation must
        # likewise bypass this hook, regardless of DRLEC_EVAL_WORKERS overrides.
        production.build_workflows = original_build
        try:
            return original_evaluate(*args, **kwargs)
        finally:
            production.build_workflows = build_training

    production.build_workflows = build_training
    production.evaluate_agent = evaluate_unchanged
    try:
        yield
    finally:
        production.build_workflows = original_build
        production.evaluate_agent = original_evaluate


def run_training(config):
    config = copy.deepcopy(config)
    mode = validate_config(config)
    training = config["training"]
    training["run_name"] = f"seed42_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
    output = ROOT / training["checkpoint_dir"] / training["run_name"]
    output.mkdir(parents=True, exist_ok=False)
    # Snapshot code/config identities without modifying any old experiment.
    sources = sorted({p for folder in ("src", "tools") for p in (ROOT / folder).rglob("*.py")})
    sources += [BASE_CONFIG, ROOT / "configs/resources_edge_cloud_dvfs_v3.json"]
    sources += sorted((ROOT / "configs").glob("diagnostic_sm74_*_budget001.json"))
    manifest = {
        "status": "running", "config": config,
        "training_reference": mode, "validation_reference": "unchanged_legacy_isolated_heft",
        "test_reference": "unchanged_sec_job_fcfs_v1; separate evaluation, not run here",
        "episode_workload_seeds": list(range(43, 1243)),
        "validation_seed_starts": production.checkpoint_validation_seeds(training, 100042),
        "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
        "feedback": "unchanged coupled Full attribution and reliability feedback",
    }
    manifest_path = output / "deadline_reference_manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    save_manifest()
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[deadline-reference] mode={mode} budget=0.01 episodes=1200 seed=42 output_dir={output}", flush=True)
    try:
        with (output / "training_inputs.jsonl").open("x", encoding="utf-8") as handle:
            def record(row):
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()

            with installed_reference(mode, record):
                result = coupled.run_training(config)
    except BaseException as error:
        manifest.update(status="interrupted_or_failed", error=f"{type(error).__name__}: {error}")
        save_manifest()
        raise
    manifest.update(status="completed", validation_feasible=result["best_metrics"] is not None)
    save_manifest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check", action="store_true", help="Validate only; no training or run directory.")
    args = parser.parse_args()
    config = production.load_json(args.config)
    mode = validate_config(config)
    if args.check:
        print(f"valid=True mode={mode} episodes=1200 seed=42 training_started=False")
        return
    result = run_training(config)
    print(f"output_dir={result['output_dir']}", flush=True)


if __name__ == "__main__":
    main()
