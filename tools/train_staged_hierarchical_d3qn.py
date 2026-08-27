from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CPU_THREADS = os.environ.get("DRLEC_CPU_THREADS", "1")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = _CPU_THREADS
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.hierarchical_common import load_json
from tools.train_hierarchical_d3qn import run_training


EXPECTED_PHASES = (
    ("heft", False, False),
    ("d3qn", False, True),
    ("d3qn", True, False),
    ("d3qn", False, False),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Four-stage HEFT/SAC -> D3QN/frozen-SAC -> frozen-D3QN/SAC -> "
            "joint D3QN/SAC curriculum training."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/hierarchical_d3qn_sac_staged.json",
        help="Staged-training configuration file.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one episode per phase with one evaluation episode.",
    )
    return parser.parse_args()


def _resolve_project_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _deep_update(target, updates):
    for key, value in dict(updates or {}).items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _validate_staged_config(staged_config):
    phases = list(staged_config.get("phases", []))
    if len(phases) != 4:
        raise ValueError("staged training requires exactly four phases.")
    for index, (phase, expected) in enumerate(zip(phases, EXPECTED_PHASES), start=1):
        high_algorithm = str(phase.get("high_algorithm", "")).lower()
        freeze_high = bool(phase.get("freeze_high", False))
        freeze_low = bool(phase.get("freeze_low", False))
        if (high_algorithm, freeze_high, freeze_low) != expected:
            raise ValueError(
                f"phase {index} must use high_algorithm={expected[0]!r}, "
                f"freeze_high={expected[1]}, freeze_low={expected[2]}; got "
                f"{high_algorithm!r}, {freeze_high}, {freeze_low}."
            )
        if int(phase.get("episodes", 0)) <= 0:
            raise ValueError(f"phase {index} episodes must be positive.")
        for key in ("high_learning_rate", "low_learning_rate"):
            if key in phase and float(phase[key]) <= 0.0:
                raise ValueError(f"phase {index} {key} must be positive.")
        for key in ("high_updates_per_episode", "low_updates_per_episode"):
            if key in phase and int(phase[key]) <= 0:
                raise ValueError(f"phase {index} {key} must be positive.")
    handoff = str(staged_config.get("handoff_checkpoint", "best")).lower()
    if handoff not in {"best", "final"}:
        raise ValueError("handoff_checkpoint must be 'best' or 'final'.")
    checkpoint_deadline_target = float(
        staged_config.get("checkpoint_deadline_target", 0.99)
    )
    if not 0.0 <= checkpoint_deadline_target <= 1.0:
        raise ValueError("checkpoint_deadline_target must be in [0, 1].")
    return phases


def _phase_config(
    base_config,
    staged_config,
    phase,
    run_name,
    global_offset,
    smoke_test,
    *,
    phase_index,
    high_active_episode_offset,
    low_active_episode_offset,
):
    config = copy.deepcopy(base_config)
    training = config.setdefault("training", {})
    agent = config.setdefault("agent", {})
    high = agent.setdefault("high", {})
    low = agent.setdefault("low", {})

    phase_name = str(phase["name"])
    training["checkpoint_dir"] = str(
        staged_config.get(
            "checkpoint_dir", "results/hierarchical_d3qn_sac_staged"
        )
    )
    training["run_name"] = f"{run_name}/{phase_name}"
    training["phase_name"] = phase_name
    training["phase_index"] = int(phase_index)
    training["phase_count"] = 4
    training["episode_index_offset"] = int(global_offset)
    training["high_active_episode_offset"] = int(
        high_active_episode_offset
    )
    training["low_active_episode_offset"] = int(
        low_active_episode_offset
    )
    training["checkpoint_deadline_target"] = float(
        staged_config.get("checkpoint_deadline_target", 0.99)
    )
    training["total_episodes"] = 1 if smoke_test else int(phase["episodes"])
    if "eval_interval" in phase:
        training["eval_interval"] = int(phase["eval_interval"])
    if "eval_episodes" in phase:
        training["eval_episodes"] = int(phase["eval_episodes"])
    if smoke_test:
        training["eval_interval"] = 1
        training["eval_episodes"] = 1
        training["save_every"] = 1

    agent["high_algorithm"] = str(phase["high_algorithm"]).lower()
    agent["low_algorithm"] = "sac"
    low["algorithm"] = "sac"
    if "high_learning_rate" in phase:
        high["learning_rate"] = float(phase["high_learning_rate"])
    if "low_learning_rate" in phase:
        low["learning_rate"] = float(phase["low_learning_rate"])
    if "high_updates_per_episode" in phase:
        high["training_per_episode"] = int(phase["high_updates_per_episode"])
    if "low_updates_per_episode" in phase:
        low["training_per_episode"] = int(phase["low_updates_per_episode"])
    return config


def _select_handoff(output_dir, high_algorithm, preference):
    output_dir = Path(output_dir)
    preferred = (
        output_dir / f"best_{high_algorithm}_checkpoint.pt"
        if preference == "best"
        else output_dir / "final_checkpoint.pt"
    )
    if preferred.is_file():
        return preferred
    fallback = output_dir / "final_checkpoint.pt"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"No handoff checkpoint found in {output_dir}.")


def _write_manifest(path, payload):
    def json_default(value):
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "item"):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )


def _summarize_best_metrics(metrics):
    metrics = dict(metrics or {})
    keys = (
        "total_energy",
        "deadline_success_rate",
        "reliability_success_rate",
        "rho_mean",
        "average_workflow_response_time",
        "global_makespan",
        "mean_high_reward",
        "mean_low_reward",
        "lambda_deadline",
        "low_deadline_lambda",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def run_staged_training(staged_config, *, run_name=None, seed=None, smoke_test=False):
    phases = _validate_staged_config(staged_config)
    base_config_path = _resolve_project_path(
        staged_config.get("base_config", "configs/hierarchical_d3qn_sac.json")
    )
    base_config = load_json(base_config_path)
    _deep_update(base_config, staged_config.get("base_overrides", {}))
    if bool(
        base_config.get("agent", {})
        .get("shared_deadline_lagrangian", {})
        .get("enabled", False)
    ):
        raise ValueError(
            "Four-stage freeze/alternating training requires separate deadline "
            "Lagrange multipliers; set agent.shared_deadline_lagrangian.enabled=false "
            "in base_overrides."
        )
    base_seed = int(
        seed
        if seed is not None
        else base_config.get("training", {}).get("seed", 42)
    )
    run_name = str(run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    checkpoint_root = _resolve_project_path(
        staged_config.get(
            "checkpoint_dir", "results/hierarchical_d3qn_sac_staged"
        )
    )
    root_output_dir = checkpoint_root / run_name
    root_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root_output_dir / "staged_training_manifest.json"
    manifest = {
        "framework": "heft_sac_then_d3qn_then_alternating_then_joint_v1",
        "base_config": str(base_config_path),
        "base_overrides": copy.deepcopy(
            staged_config.get("base_overrides", {})
        ),
        "seed": base_seed,
        "run_name": run_name,
        "status": "running",
        "handoff_checkpoint": str(
            staged_config.get("handoff_checkpoint", "best")
        ).lower(),
        "checkpoint_selection": "deadline_feasible_then_min_energy",
        "checkpoint_deadline_target": float(
            staged_config.get("checkpoint_deadline_target", 0.99)
        ),
        "phases": [],
    }
    _write_manifest(manifest_path, manifest)

    handoff_checkpoint = None
    global_offset = 0
    high_active_episode_offset = 0
    low_active_episode_offset = 0
    preference = manifest["handoff_checkpoint"]
    for index, phase in enumerate(phases, start=1):
        phase_name = str(phase["name"])
        phase_episodes = 1 if smoke_test else int(phase["episodes"])
        phase_config = _phase_config(
            base_config,
            staged_config,
            phase,
            run_name,
            global_offset,
            smoke_test,
            phase_index=index,
            high_active_episode_offset=high_active_episode_offset,
            low_active_episode_offset=low_active_episode_offset,
        )
        phase_record = {
            "index": index,
            "name": phase_name,
            "status": "running",
            "global_episode_start": global_offset + 1,
            "global_episode_end": global_offset + phase_episodes,
            "high_algorithm": str(phase["high_algorithm"]).lower(),
            "freeze_high": bool(phase.get("freeze_high", False)),
            "freeze_low": bool(phase.get("freeze_low", False)),
            "high_active_episode_offset": int(high_active_episode_offset),
            "low_active_episode_offset": int(low_active_episode_offset),
            "input_checkpoint": (
                None if handoff_checkpoint is None else str(handoff_checkpoint)
            ),
        }
        manifest["phases"].append(phase_record)
        _write_manifest(manifest_path, manifest)
        print(
            "[stage] "
            f"start={index}/4 name={phase_name} "
            f"global_episodes={global_offset + 1}-{global_offset + phase_episodes} "
            f"high={phase_record['high_algorithm']} "
            f"freeze_high={phase_record['freeze_high']} "
            f"freeze_low={phase_record['freeze_low']}"
        )

        try:
            result = run_training(
                phase_config,
                smoke_test=smoke_test,
                episodes_override=phase_episodes,
                seed_override=base_seed,
                low_algorithm_override="sac",
                high_algorithm_override=phase_record["high_algorithm"],
                freeze_high=phase_record["freeze_high"],
                freeze_low=phase_record["freeze_low"],
                resume_checkpoint=handoff_checkpoint,
                epsilon_start_override=phase.get("epsilon_start"),
                episode_index_offset=global_offset,
            )
            phase_output_dir = Path(result["output_dir"])
            handoff_checkpoint = _select_handoff(
                phase_output_dir,
                phase_record["high_algorithm"],
                preference,
            )
            phase_record.update(
                {
                    "status": "complete",
                    "output_dir": str(phase_output_dir),
                    "handoff_checkpoint": str(handoff_checkpoint),
                    "best_metrics": _summarize_best_metrics(
                        result.get("best_metrics")
                    ),
                }
            )
            del result
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        except Exception as exc:
            phase_record["status"] = "failed"
            phase_record["error"] = f"{type(exc).__name__}: {exc}"
            manifest["status"] = "failed"
            _write_manifest(manifest_path, manifest)
            raise

        global_offset += phase_episodes
        if (
            phase_record["high_algorithm"] == "d3qn"
            and not phase_record["freeze_high"]
        ):
            high_active_episode_offset += phase_episodes
        if not phase_record["freeze_low"]:
            low_active_episode_offset += phase_episodes
        _write_manifest(manifest_path, manifest)
        print(
            "[stage] "
            f"complete={index}/4 name={phase_name} "
            f"handoff={handoff_checkpoint}"
        )

    manifest["status"] = "complete"
    manifest["total_episodes"] = global_offset
    manifest["final_checkpoint"] = str(handoff_checkpoint)
    _write_manifest(manifest_path, manifest)
    return {
        "output_dir": str(root_output_dir),
        "manifest": str(manifest_path),
        "final_checkpoint": str(handoff_checkpoint),
    }


def main():
    args = parse_args()
    staged_config = load_json(args.config)
    result = run_staged_training(
        staged_config,
        run_name=args.run_name,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )
    print(f"output_dir={result['output_dir']}")
    print(f"final_checkpoint={result['final_checkpoint']}")


if __name__ == "__main__":
    main()
