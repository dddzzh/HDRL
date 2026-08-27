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
from tools.train_dual_head_d3qn import _print_dual_head_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a dual-head D3QN with a SAC lower-level policy."
    )
    parser.add_argument(
        "--config",
        default="configs/hierarchical_dual_head_d3qn_sac_freeze_high.json",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--high-checkpoint", default=None)
    parser.add_argument(
        "--joint",
        action="store_true",
        help=(
            "Jointly update the upper D3QN and lower SAC. Without this flag, "
            "the historical frozen-upper behavior is preserved."
        ),
    )
    return parser.parse_args()


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
        episodes_override=args.episodes,
        seed_override=args.seed,
        fixed_low_rho_one=False,
        freeze_high=not args.joint,
        high_checkpoint=args.high_checkpoint,
    )
    print(f"output_dir={result['output_dir']}")


if __name__ == "__main__":
    main()
