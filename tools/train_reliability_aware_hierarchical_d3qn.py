"""Real workflow trainer for Reliability-aware High D3QN + unchanged SAC.

This is a thin standalone entry point over the production trainer. Environment
creation, workflow collection, HighRolloutBuffer, SAC replay/update, deadline
dual update, evaluation and history writing all come from
``train_hierarchical_d3qn``. Only the high-level learner is replaced.
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCAL_SITE_PACKAGES = ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

from src.D3QN.dual_head_constrained import DualHeadHierarchicalD3QNAgent
from src.D3QN.reliability_aware_high import ReliabilityAwareHighLevelD3QNAgent
from src.D3QN.utils import as_torch_rng_state
from tools import train_hierarchical_d3qn as production
import torch


class ReliabilityAwareHierarchicalD3QNAgent(DualHeadHierarchicalD3QNAgent):
    """Production hierarchical agent with only its high learner replaced."""

    CHECKPOINT_FORMAT = "hierarchical_reliability_aware_d3qn_v1"

    def __init__(self, config=None, seed=42):
        super().__init__(config=config, seed=seed)
        config = dict(config or {})
        shared = dict(config.get("shared", {}))
        high_cfg = {**shared, **dict(config.get("high", {}))}
        self.high_agent = ReliabilityAwareHighLevelD3QNAgent(high_cfg, seed=seed)
        # The low SAC instance created by the production constructor remains
        # untouched, including state/action/critic dimensions and optimizers.
        if self.shared_deadline_lagrangian_enabled:
            self.high_agent.deadline_lagrangian = self.shared_deadline_lagrangian
            self.high_agent.deadline_lagrangian_enabled = True
        self._validate_encoder_config()
        self._validate_shared_deadline_lagrangian()

    def save_checkpoint(self, path, **kwargs):
        """Persist all three high-level heads and both constraint multipliers."""
        super().save_checkpoint(path, **kwargs)
        reliability_state = self.high_agent.reliability_lagrangian.state_dict()
        paths = (
            Path(path),
            self._component_checkpoint_path(Path(path), "high"),
        )
        for checkpoint_path in paths:
            payload = torch.load(
                str(checkpoint_path), map_location="cpu", weights_only=False
            )
            payload["reliability_lagrangian"] = copy.deepcopy(reliability_state)
            torch.save(payload, str(checkpoint_path))

    def load_checkpoint(self, path):
        """Load legacy primary or production dual-head full checkpoints.

        The reliability head is new, so dual-head checkpoints initialize only
        their energy/deadline heads; the reliability head starts from its
        configured initialization while SAC is restored exactly.
        """
        payload = torch.load(str(path), map_location=self.high_agent.agent_R.device,
                             weights_only=False)
        fmt = payload.get("checkpoint_format")
        if fmt == self.CHECKPOINT_FORMAT:
            restored = super().load_checkpoint(path)
            reliability_state = restored.get("reliability_lagrangian")
            if reliability_state is not None:
                self.high_agent.reliability_lagrangian.load_state_dict(
                    reliability_state
                )
            return restored
        if fmt != "hierarchical_dual_head_d3qn_v1":
            raise RuntimeError(
                "Reliability-aware training requires a complete dual-head or "
                "reliability-aware checkpoint; "
                f"got checkpoint_format={fmt!r}."
            )
        saved = payload["dual_head_state_dict"]
        learner = self.high_agent.agent_R
        learner.energy_q_eval.load_state_dict(saved["energy_q_eval"])
        learner.energy_q_target.load_state_dict(saved["energy_q_target"])
        learner.deadline_q_eval.load_state_dict(saved["deadline_q_eval"])
        learner.deadline_q_target.load_state_dict(saved["deadline_q_target"])
        learner.energy_optimizer.load_state_dict(saved["energy_optimizer"])
        learner.deadline_optimizer.load_state_dict(saved["deadline_optimizer"])
        learner.epsilon = float(saved.get("epsilon", learner.epsilon))
        if self.low_agent is not None and payload.get("low_model_state_dict") is not None:
            self.low_agent.load_state_dict(copy.deepcopy(payload["low_model_state_dict"]))
        self._load_lagrangian_checkpoint_state(payload)
        if payload.get("numpy_random_state") is not None:
            import numpy as np
            np.random.set_state(payload["numpy_random_state"])
        if payload.get("torch_random_state") is not None:
            torch.random.set_rng_state(as_torch_rng_state(payload["torch_random_state"]))
        if torch.cuda.is_available() and payload.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all([
                as_torch_rng_state(state)
                for state in payload["cuda_random_state"]
            ])
        return payload

    def load_high_checkpoint(self, path, *, freeze=False):
        payload = self.load_checkpoint(path)
        if freeze:
            learner = self.high_agent.agent_R
            for network in (
                learner.energy_q_eval,
                learner.energy_q_target,
                learner.deadline_q_eval,
                learner.deadline_q_target,
                learner.reliability_q_eval,
                learner.reliability_q_target,
            ):
                network.eval()
                for parameter in network.parameters():
                    parameter.requires_grad_(False)
        return payload


def run_training(config, **kwargs):
    original = production.HierarchicalD3QNAgent
    production.HierarchicalD3QNAgent = ReliabilityAwareHierarchicalD3QNAgent
    try:
        return production.run_training(config, **kwargs)
    finally:
        production.HierarchicalD3QNAgent = original


def main():
    args = production.parse_args()
    config = production.load_json(args.config)
    result = run_training(
        config,
        smoke_test=args.smoke_test,
        episodes_override=args.episodes,
        seed_override=args.seed,
        fixed_low_rho_one=args.fixed_low_rho_one,
        low_algorithm_override=args.low_algorithm,
        high_algorithm_override=args.high_algorithm,
        freeze_high=args.freeze_high,
        freeze_low=args.freeze_low,
        high_checkpoint=args.high_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
        high_learning_rate_override=args.high_learning_rate,
        high_updates_override=args.high_updates_per_episode,
        epsilon_start_override=args.epsilon_start,
    )
    print(f"output_dir={result['output_dir']}")


if __name__ == "__main__":
    main()
