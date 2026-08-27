from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.D3QN.dual_head_feasible import FeasibleRiskHierarchicalD3QNAgent
from tools import train_hierarchical_d3qn as common_trainer
from tools.hierarchical_common import load_json


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train fixed-rho=1 energy-D3QN plus workflow Deadline-risk "
            "feasibility head."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/hierarchical_dual_head_feasible_d3qn.json",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    return parser.parse_args()


def _print_feasible_risk_metrics(label, row):
    common_trainer._ORIGINAL_PRINT_D3QN_METRICS_BLOCK(label, row)
    if "high_energy_loss" in row or "high_deadline_loss" in row:
        print(
            "  risk_head    "
            f"E_loss={float(row.get('high_energy_loss', 0.0)):.6g} "
            f"D_bce={float(row.get('high_deadline_loss', 0.0)):.6g} "
            f"Q_E={float(row.get('high_energy_q_mean', 0.0)):.6g} "
            f"p_mean={float(row.get('high_risk_probability_mean', 0.0)):.6g} "
            f"y_mean={float(row.get('high_risk_label_mean', 0.0)):.6g} "
            f"brier={float(row.get('high_risk_brier_score', 0.0)):.6g} "
            f"ece={float(row.get('high_risk_ece', 0.0)):.6g} "
            f"recall={float(row.get('high_risk_positive_recall', 0.0)):.3f}"
        )
    if "risk_selected_mean" in row or "high_risk_threshold" in row:
        print(
            "  feasible     "
            f"tau={float(row.get('high_risk_threshold', row.get('risk_threshold_mean', 0.0))):.6g} "
            f"gate={int(float(row.get('high_risk_gate_active', row.get('risk_gate_active_rate', 0.0))) > 0.5)} "
            f"wf={int(float(row.get('high_risk_replay_workflows', 0.0)))} "
            f"risk_selected={float(row.get('risk_selected_mean', 0.0)):.6g} "
            f"feasible_n={float(row.get('feasible_action_count_mean', 0.0)):.3f} "
            f"fallback={float(row.get('no_feasible_action_rate', 0.0)):.3f} "
            f"rollout_brier={float(row.get('risk_rollout_brier_score', 0.0)):.6g} "
            f"rollout_ece={float(row.get('risk_rollout_ece', 0.0)):.6g}"
        )


def main():
    args = parse_args()
    config = load_json(args.config)
    common_trainer.HierarchicalD3QNAgent = FeasibleRiskHierarchicalD3QNAgent
    if not hasattr(common_trainer, "_ORIGINAL_PRINT_D3QN_METRICS_BLOCK"):
        common_trainer._ORIGINAL_PRINT_D3QN_METRICS_BLOCK = (
            common_trainer.print_d3qn_metrics_block
        )
    common_trainer.print_d3qn_metrics_block = _print_feasible_risk_metrics
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
