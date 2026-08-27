"""Trainer for the coupled (option-aware) hierarchical D3QN + DVFS SAC.

Two switches, each independently ablatable, layered on top of the unchanged
three-head trainer ``train_reliability_aware_hierarchical_d3qn``:

``--physical-low-reward`` / ``agent.physical_low_reward``
    Replace the low SAC's replay z-scored energy/time trade-off with a fixed
    physical scale (:mod:`src.sac.physical_dvfs_sac`).  The upper level,
    replay layout, networks and Lagrangian controllers are untouched.

``--dvfs-aware-high-state`` / ``agent.dvfs_aware_high_state``
    Widen the upper state from 12 to 15 features per server so the upper
    policy can see each server's DVFS operating point
    (:mod:`src.D3QN.dvfs_aware_state`).  This changes ``high_state_dim`` and
    therefore requires training the upper level from scratch.

Everything else -- environment, workflow generation, deadlines, reliability
requirement, server resources, comparison algorithms, rollout collection,
evaluation and checkpoint selection -- is reused verbatim.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCAL_SITE_PACKAGES = ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir() and str(LOCAL_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

from src.D3QN.dvfs_aware_state import DVFSAwareHighStateEncoder
from src.D3QN.reliability_aware_high import ReliabilityAwareHighLevelD3QNAgent
from src.sac.physical_dvfs_sac import PhysicalDVFSLowSACAgent
from tools import train_hierarchical_d3qn as production
from tools.train_reliability_aware_hierarchical_d3qn import (
    ReliabilityAwareHierarchicalD3QNAgent,
)

SHARED_LOW_STATE_MODES = {"shared_high_selected_server", "action_conditioned_ctde"}

# ``DeadlineLagrangianController.load_state_dict`` restores ``dual_learning_rate``,
# ``deadline_budget``, ``lambda_max``, ``warmup_episodes`` and ``window_workflows``
# from the checkpoint, so a warm-started run silently keeps the *checkpoint's*
# dual hyper-parameters and ignores the config.  That is why every staged
# restart in this project inherited ``dual_learning_rate = 0.001`` and the dual
# loop never closed (measured d(lambda)/d(episode) = 6e-5, i.e. a ~10^4-episode
# time constant).  The multiplier value itself is deliberately NOT reset, so a
# warm start still continues from the lambda it had reached.
LAGRANGIAN_HYPERPARAMETERS = (
    ("dual_learning_rate", "dual_lr", float),
    ("deadline_budget", "budget", float),
    ("lambda_max", "lambda_max", float),
    ("warmup_episodes", "warmup_episodes", int),
    ("window_workflows", "window_workflows", int),
)

# The reliability controllers use a different key prefix but suffer from the
# same frozen-dual problem: with the shipped ``reliability_dual_learning_rate``
# of 0.001 the multiplier cannot react.  That is harmless while the lower level
# sits at rho = 1 (workflow reliability 0.937 against a 0.9 requirement), but
# real DVFS makes reliability the *binding* constraint: the failure rate model
# lambda(f) = lambda_max * 10^(d (f_max - f)/(f_max - f_min)) combined with the
# longer execution time multiplies task risk by 1.4x at rho = 0.90, 1.7x at
# rho = 0.85 and 3.1x at rho = 0.70, so the feasible frequency floor is
# rho ~ 0.85 -- far above the energy-optimal rho* of 0.40-0.48.
RELIABILITY_LAGRANGIAN_HYPERPARAMETERS = (
    ("reliability_dual_learning_rate", "dual_lr", float),
    ("reliability_budget", "budget", float),
    ("reliability_lambda_max", "lambda_max", float),
    ("reliability_warmup_episodes", "warmup_episodes", int),
    ("reliability_window_workflows", "window_workflows", int),
)


def _lagrangian_hyperparameter_drift(controller, block, label,
                                     spec=LAGRANGIAN_HYPERPARAMETERS):
    """Return every hyper-parameter whose live value disagrees with the config."""
    if controller is None or not block:
        return []
    drifted = []
    for config_key, attribute, cast in spec:
        if config_key not in block:
            continue
        wanted = cast(block[config_key])
        current = cast(getattr(controller, attribute))
        if current != wanted:
            drifted.append(f"{label}.{config_key} live={current} config={wanted}")
    return drifted


class ConfiguredLagrangianMixin:
    """Verify that every dual controller kept its *configured* tuning.

    The controllers themselves now restore state only (see
    ``DeadlineLagrangianController.load_state_dict``), so this is a guard rather
    than a fix: if a checkpoint ever manages to overwrite a configured
    hyper-parameter again, training stops here instead of silently running with
    a dual loop that cannot close.
    """

    def _reapply_configured_lagrangian(self):
        config = dict(getattr(self, "config", {}) or {})
        shared = dict(config.get("shared", {}))
        low_config = dict(config.get("low", {}))
        low_agent = getattr(self, "low_agent", None)
        targets = [
            (
                getattr(self.high_agent, "deadline_lagrangian", None),
                dict({**shared, **dict(config.get("high", {}))}.get("lagrangian", {})),
                "high",
                LAGRANGIAN_HYPERPARAMETERS,
            ),
            (
                getattr(low_agent, "deadline_lagrangian", None),
                dict(low_config.get("lagrangian", {})),
                "low",
                LAGRANGIAN_HYPERPARAMETERS,
            ),
            (
                getattr(self, "shared_deadline_lagrangian", None),
                dict(config.get("lagrangian", {})),
                "shared",
                LAGRANGIAN_HYPERPARAMETERS,
            ),
            (
                getattr(low_agent, "reliability_lagrangian", None),
                dict(low_config.get("reliability_lagrangian", {})),
                "low_reliability",
                RELIABILITY_LAGRANGIAN_HYPERPARAMETERS,
            ),
        ]
        drifted = []
        for controller, block, label, spec in targets:
            drifted.extend(
                _lagrangian_hyperparameter_drift(controller, block, label, spec)
            )
        if drifted:
            raise RuntimeError(
                "A checkpoint overwrote configured Lagrangian hyper-parameters: "
                + "; ".join(drifted)
            )
        return drifted

    def load_checkpoint(self, path):
        payload = super().load_checkpoint(path)
        self._reapply_configured_lagrangian()
        return payload

    def load_high_checkpoint(self, path, *, freeze=False):
        payload = super().load_high_checkpoint(path, freeze=freeze)
        self._reapply_configured_lagrangian()
        return payload


class DVFSAwareReliabilityHighLevelD3QNAgent(ReliabilityAwareHighLevelD3QNAgent):
    """Three-head upper learner on the 15-feature-per-server DVFS state."""

    @staticmethod
    def build_state_encoder(state_cfg):
        return DVFSAwareHighStateEncoder(state_cfg)


class ConfiguredLagrangianReliabilityAgent(
    ConfiguredLagrangianMixin, ReliabilityAwareHierarchicalD3QNAgent
):
    """S1: unchanged learner, but the configured dual hyper-parameters win."""


class PhysicalLowRewardHierarchicalAgent(
    ConfiguredLagrangianMixin, ReliabilityAwareHierarchicalD3QNAgent
):
    """S2: unchanged upper level, physically scaled low SAC reward.

    The checkpoint format is deliberately inherited so this variant can still
    warm-start from an existing three-head or dual-head upper checkpoint.
    """

    @staticmethod
    def build_low_sac_agent(low_cfg, seed):
        return PhysicalDVFSLowSACAgent(low_cfg, seed=seed)


class CoupledDVFSHierarchicalAgent(PhysicalLowRewardHierarchicalAgent):
    """S2 + S3: DVFS-aware upper state and physically scaled low reward."""

    CHECKPOINT_FORMAT = "hierarchical_coupled_dvfs_d3qn_v1"

    def __init__(self, config=None, seed=42):
        super().__init__(config=config, seed=seed)
        config = dict(config or {})
        shared = dict(config.get("shared", {}))
        high_cfg = {**shared, **dict(config.get("high", {}))}
        self.high_agent = DVFSAwareReliabilityHighLevelD3QNAgent(high_cfg, seed=seed)
        if self.shared_deadline_lagrangian_enabled:
            self.high_agent.deadline_lagrangian = self.shared_deadline_lagrangian
            self.high_agent.deadline_lagrangian_enabled = True
        low_state_mode = str(
            dict(config.get("low", {})).get("state_mode", "local_9")
        ).strip().lower()
        if low_state_mode in SHARED_LOW_STATE_MODES:
            # Those modes derive their dimensions from the upper encoder that
            # was built before this replacement, so they would silently use the
            # 347-D layout.  Fail loudly rather than train a mismatched critic.
            raise ValueError(
                "dvfs_aware_high_state is only wired for low state_mode "
                f"'local_9'; got {low_state_mode!r}."
            )
        self._validate_encoder_config()
        self._validate_shared_deadline_lagrangian()


def dvfs_delay_share_misses(result):
    """Per-workflow low-level Deadline responsibility as a DVFS delay share.

    The production signal is ``_low_causal_deadline_misses``: a workflow counts
    against the lower level only when it missed *and* some task crossed the
    workflow Deadline while its rho=1 reference did not.  That reference is
    recomputed at decision time against the *already inflated* queue, so once
    DVFS has congested a server every later miss looks like an upper-level
    placement error.  Measured on the published SAC run: the system missed 35%
    of workflows at episode 81 while the low dual cost was 0.0035, and
    ``lambda_low`` moved 10.0000 -> 10.0185 over 600 episodes.  The dual is
    therefore structurally unable to regulate rho.

    This replacement charges the lower level the fraction of each missed
    workflow's overrun that its own DVFS stretching accounts for::

        delay_w  = sum over the workflow's tasks of exec_time * (1 - rho)
        overrun_w = response_time_w - D_w
        cost_w   = miss_w * clip(delay_w / overrun_w, 0, 1)

    ``exec_time * (1 - rho)`` is exact for ``T(rho) = T(1) / rho``.  A workflow
    that ran everything at rho=1 still contributes zero, and a workflow whose
    overrun is fully explained by DVFS contributes one, so the dual keeps
    constraining a proper workflow-miss probability in [0, 1].
    """
    workflow_results = result["workflow_results"]
    delay_by_workflow = {}
    for event in result["event_log"]:
        if event.get("replica_role") != "primary":
            continue
        rho = float(event.get("rho", 1.0))
        if rho >= 1.0 - 1e-9:
            continue
        workflow_id = str(event.get("workflow_id"))
        delay_by_workflow[workflow_id] = delay_by_workflow.get(workflow_id, 0.0) + (
            float(event.get("exec_time", 0.0)) * (1.0 - rho)
        )
    costs = []
    for row in workflow_results:
        workflow_id = str(row["workflow_id"])
        response = float(row.get("response_time") or 0.0)
        deadline = float(row.get("relative_deadline") or 0.0)
        overrun = response - deadline
        if overrun <= 1e-9:
            costs.append(0.0)
            continue
        share = delay_by_workflow.get(workflow_id, 0.0) / overrun
        costs.append(float(min(1.0, max(0.0, share))))
    return costs


def reliability_violation_indicators(result):
    """Binary per-workflow reliability violations for the low-level dual.

    Two earlier signals were tried and measured to fail:

    * the production *projected* violation stays near zero while the projection
      still credits every unscheduled task with
      ``remaining_task_reliability_estimate`` (0.999), so rho reached 0.83 at a
      58% reliability success rate with nothing for the multiplier to react to;
    * the mean *risk-budget share* spent by DVFS is dense and two-sided, but a
      mean-share target cannot deliver per-workflow feasibility -- at a budget
      of 0.35 the success rate settled around 0.50-0.65 because individual
      workflows start from different baseline reliabilities.

    The acceptance criterion is a per-workflow success rate of 100%, so the dual
    observes exactly that: one indicator per completed workflow, constrained to
    a small budget.  The *reward* keeps the dense
    ``reliability_budget_share_spent`` surrogate, so SAC still gets a gradient
    for every rho < 1 while the multiplier tracks the true constraint.
    """
    return [
        float(row.get("reliability_met") is False)
        for row in result["workflow_results"]
    ]


def install_low_dual_attribution(mode):
    """Wrap ``collect_d3qn_episode`` to change only the low duals' cost signals."""
    original = production.collect_d3qn_episode

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        result["low_deadline_dual_costs"] = dvfs_delay_share_misses(result)
        indicators = reliability_violation_indicators(result)
        if len(indicators) == len(result["low_reliability_dual_costs"]):
            result["low_reliability_dual_costs"] = indicators
        return result

    if mode == "dvfs_delay_share":
        production.collect_d3qn_episode = wrapped
        return original
    return None


def select_agent_class(config):
    agent_cfg = dict(config.get("agent", {}))
    dvfs_state = bool(agent_cfg.get("dvfs_aware_high_state", False))
    physical_low = bool(agent_cfg.get("physical_low_reward", False))
    if dvfs_state and not physical_low:
        raise ValueError(
            "dvfs_aware_high_state assumes the lower level actually reaches the "
            "advertised operating point; enable physical_low_reward as well."
        )
    if dvfs_state:
        return CoupledDVFSHierarchicalAgent
    if physical_low:
        return PhysicalLowRewardHierarchicalAgent
    if bool(agent_cfg.get("enforce_config_lagrangian", False)):
        return ConfiguredLagrangianReliabilityAgent
    return ReliabilityAwareHierarchicalD3QNAgent


def run_training(config, **kwargs):
    agent_class = select_agent_class(config)
    attribution = str(
        dict(config.get("agent", {})).get("low_dual_attribution", "counterfactual")
    ).strip().lower()
    if attribution not in {"counterfactual", "dvfs_delay_share"}:
        raise ValueError(
            "agent.low_dual_attribution must be 'counterfactual' or "
            f"'dvfs_delay_share'; got {attribution!r}."
        )
    original_agent = production.HierarchicalD3QNAgent
    original_collect = install_low_dual_attribution(attribution)
    production.HierarchicalD3QNAgent = agent_class
    try:
        return production.run_training(config, **kwargs)
    finally:
        production.HierarchicalD3QNAgent = original_agent
        if original_collect is not None:
            production.collect_d3qn_episode = original_collect


def main():
    parser_args = production.parse_args()
    config = production.load_json(parser_args.config)
    agent_class = select_agent_class(config)
    print(f"[coupled] agent_class={agent_class.__name__}", flush=True)
    result = run_training(
        copy.deepcopy(config),
        smoke_test=parser_args.smoke_test,
        episodes_override=parser_args.episodes,
        seed_override=parser_args.seed,
        fixed_low_rho_one=parser_args.fixed_low_rho_one,
        low_algorithm_override=parser_args.low_algorithm,
        high_algorithm_override=parser_args.high_algorithm,
        freeze_high=parser_args.freeze_high,
        freeze_low=parser_args.freeze_low,
        high_checkpoint=parser_args.high_checkpoint,
        resume_checkpoint=parser_args.resume_checkpoint,
        high_learning_rate_override=parser_args.high_learning_rate,
        high_updates_override=parser_args.high_updates_per_episode,
        epsilon_start_override=parser_args.epsilon_start,
    )
    print(f"output_dir={result['output_dir']}")


if __name__ == "__main__":
    main()
