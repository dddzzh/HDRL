from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SITE_PACKAGES = PROJECT_ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.D3QN.dual_head_constrained import DualHeadHierarchicalD3QNAgent
from tools import train_hierarchical_d3qn as common_trainer
from tools.hierarchical_common import load_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train standalone dual-head constrained D3QN at fixed rho=1."
    )
    parser.add_argument(
        "--config", default="configs/hierarchical_dual_head_d3qn.json"
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    return parser.parse_args()


def _print_dual_head_metrics(label, row):
    common_trainer._ORIGINAL_PRINT_D3QN_METRICS_BLOCK(label, row)
    if "high_energy_loss" not in row:
        return
    print(
        "  dual_head     "
        f"E_loss={float(row.get('high_energy_loss', 0.0)):.6g} "
        f"D_loss={float(row.get('high_deadline_loss', 0.0)):.6g} "
        f"Q_E={float(row.get('high_energy_q_mean', 0.0)):.6g} "
        f"Q_D={float(row.get('high_deadline_q_mean', 0.0)):.6g} "
        f"TD_E={float(row.get('high_energy_td_abs_mean', 0.0)):.6g} "
        f"TD_D={float(row.get('high_deadline_td_abs_mean', 0.0)):.6g} "
        f"positive={float(row.get('high_deadline_positive_replay_fraction', 0.0)):.3f} "
        f"lambda={float(row.get('high_deadline_lambda_for_action', row.get('lambda_deadline', 0.0))):.6g} "
        f"std_E={float(row.get('dual_energy_action_std_mean', 0.0)):.6g} "
        f"std_D={float(row.get('dual_deadline_action_std_mean', 0.0)):.6g} "
        f"lambda_stdD_over_stdE={float(row.get('dual_deadline_to_energy_scale_mean', 0.0)):.6g} "
        f"dense_D={int(float(row.get('high_dense_deadline_cost_enabled', 0.0)) > 0.5)} "
        f"cross_D={int(float(row.get('high_deadline_head_crossing_count', 0.0)))} "
        f"cross_beta={float(row.get('high_deadline_crossing_penalty', 0.0)):.6g} "
        f"miss_beta={float(row.get('high_deadline_miss_penalty', 0.0)):.6g}"
    )


def main():
    args = parse_args()
    config = load_json(args.config)
    common_trainer.HierarchicalD3QNAgent = DualHeadHierarchicalD3QNAgent
    if not hasattr(common_trainer, "_ORIGINAL_PRINT_D3QN_METRICS_BLOCK"):
        common_trainer._ORIGINAL_PRINT_D3QN_METRICS_BLOCK = (
            common_trainer.print_d3qn_metrics_block
        )
    common_trainer.print_d3qn_metrics_block = _print_dual_head_metrics
    result = common_trainer.run_training(
        config,
        smoke_test=args.smoke_test,
        episodes_override=args.episodes,
        seed_override=args.seed,
        fixed_low_rho_one=True,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(f"output_dir={result['output_dir']}")


if __name__ == "__main__":
    main()
