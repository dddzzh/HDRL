"""Independent A/slack-time and B/lower-dual routes; legacy entry points unchanged."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_coupled_hierarchical_d3qn_sac as coupled
from src.sac.experimental_deadline_routes import (
    DEADLINE_ROUTES, DENSE_RISK_COST_MODE, DenseRiskLowSACAgent,
)

production = coupled.production
FORMAT = "hierarchical_deadline_routes_d3qn_v1"
_ORIGINAL_COLLECT = production.collect_d3qn_episode


def validate_agent_config(config):
    route = config.get("deadline_route")
    low = config.get("low", {})
    high = config.get("high", {})
    if route not in DEADLINE_ROUTES or low.get("deadline_route") != route:
        raise ValueError("Both agent and agent.low must identify the same experimental deadline_route.")
    if config.get("physical_low_reward", False) or config.get("dvfs_aware_high_state", False):
        raise ValueError("Deadline routes keep the legacy base reward/state; do not combine other variants.")
    if (config.get("low_algorithm") != "sac" or config.get("high_algorithm") != "d3qn"
            or low.get("action_mode") != "server_affine"):
        raise ValueError("Deadline routes require three-head D3QN and corrected server_affine SAC.")
    if low.get("reliability_cost_mode") != DENSE_RISK_COST_MODE:
        raise ValueError("Deadline routes require the dense reliability budget-share cost.")
    if (high.get("reliability_dual_mode") != "two_sided_miss_rate"
            or not low.get("reliability_lagrangian", {}).get("enabled", False)):
        raise ValueError("Both reliability controllers must remain enabled and two-sided.")
    if (float(high.get("reliability_dual_budget", -1)) != .005
            or float(low["reliability_lagrangian"].get("reliability_budget", -1)) != .005):
        raise ValueError("The two experimental reliability budgets must both be 0.005.")
    return route


class DeadlineRouteHierarchicalAgent(coupled.ConfiguredLagrangianReliabilityAgent):
    CHECKPOINT_FORMAT = FORMAT

    def __init__(self, config=None, seed=42):
        values = copy.deepcopy(dict(config or {}))
        validate_agent_config(values)
        super().__init__(values, seed=seed)

    @staticmethod
    def build_low_sac_agent(low_cfg, seed):
        return DenseRiskLowSACAgent(low_cfg, seed=seed)

    def _check_checkpoint(self, path, *, require_low):
        import torch
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        saved = payload.get("config", {})
        saved_agent = saved.get("agent", saved)
        if (payload.get("checkpoint_format") != FORMAT
                or saved_agent.get("deadline_route") != self.config["deadline_route"]):
            raise ValueError("Checkpoint is legacy or belongs to the other Deadline route; use a matching new run.")
        if require_low and payload.get("low_model_state_dict") is None:
            raise ValueError("Full resume requires a full route checkpoint, not a high-only component.")

    def load_checkpoint(self, path):
        self._check_checkpoint(path, require_low=True)
        return super().load_checkpoint(path)

    def load_high_checkpoint(self, path, *, freeze=False):
        self._check_checkpoint(path, require_low=False)
        return super().load_high_checkpoint(path, freeze=freeze)


def collect_route_episode(*args, **kwargs):
    """Keep reward credit unchanged; feed true terminal misses to the duals."""
    kwargs["low_dual_attribution"] = "workflow_counterfactual"
    result = _ORIGINAL_COLLECT(*args, **kwargs)
    deadline_misses = []
    reliability_misses = []
    for row in result["workflow_results"]:
        if row.get("deadline_satisfied") is None or row.get("reliability_met") is None:
            raise ValueError("Route duals require complete, labeled workflow results.")
        response, deadline = float(row["response_time"]), float(row["relative_deadline"])
        actual_r, required_r = float(row["reliability"]), float(row["reliability_req"])
        if not all(math.isfinite(value) for value in (response, deadline, actual_r, required_r)) or deadline <= 0:
            raise ValueError("Invalid terminal workflow constraint values.")
        missed = response > deadline + 1e-8
        rel_missed = actual_r < required_r
        if missed != (not bool(row["deadline_satisfied"])) or rel_missed != (not bool(row["reliability_met"])):
            raise ValueError("Terminal workflow flags disagree with the actual constraints.")
        deadline_misses.append(float(missed))
        reliability_misses.append(float(rel_missed))
    if not deadline_misses:
        raise ValueError("No completed workflows for the route duals.")
    result["low_deadline_dual_costs"] = deadline_misses
    result["low_reliability_dual_costs"] = reliability_misses
    return result


def load_route_checkpoint(path, seed=742000, device="cpu"):
    """Explicit loader for future evaluation; old evaluators reject this format."""
    import torch
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") != FORMAT:
        raise ValueError("Not an experimental Deadline-route checkpoint.")
    config = copy.deepcopy(payload["config"])
    agent_config = copy.deepcopy(config.get("agent", config))
    for block in ("shared", "high", "low"):
        agent_config.setdefault(block, {})["device"] = device
    agent = DeadlineRouteHierarchicalAgent(agent_config, seed=seed)
    agent.load_checkpoint(path)
    agent.high_agent.epsilon = 0.
    from src.D3QN.reward import HierarchicalObjective
    objective = HierarchicalObjective(config.get("objective", {}))
    objective.load_state_dict(payload.get("objective_state", {}))
    return payload, agent, objective


def run_training(config, **kwargs):
    validate_agent_config(config["agent"])
    if kwargs.get("high_algorithm_override") not in (None, "d3qn") or kwargs.get("low_algorithm_override") not in (None, "sac"):
        raise ValueError("Do not change the algorithms in a Deadline-route experiment.")
    if kwargs.get("fixed_low_rho_one", False):
        raise ValueError("The two routes must train their own frequency policy, not force rho=1.")
    old_agent, old_collect = production.HierarchicalD3QNAgent, production.collect_d3qn_episode
    production.HierarchicalD3QNAgent = DeadlineRouteHierarchicalAgent
    production.collect_d3qn_episode = collect_route_episode
    try:
        return production.run_training(copy.deepcopy(config), **kwargs)
    finally:
        production.HierarchicalD3QNAgent = old_agent
        production.collect_d3qn_episode = old_collect


def main():
    args = production.parse_args()
    config = production.load_json(args.config)
    print(f"[deadline-route] route={config['agent']['deadline_route']} format={FORMAT} dense_reliability=True", flush=True)
    result = run_training(config, smoke_test=args.smoke_test, episodes_override=args.episodes, seed_override=args.seed,
                fixed_low_rho_one=args.fixed_low_rho_one, high_algorithm_override=args.high_algorithm,
                low_algorithm_override=args.low_algorithm, freeze_high=args.freeze_high, freeze_low=args.freeze_low,
                high_checkpoint=args.high_checkpoint, resume_checkpoint=args.resume_checkpoint,
                high_learning_rate_override=args.high_learning_rate,
                high_updates_override=args.high_updates_per_episode, epsilon_start_override=args.epsilon_start)
    print(f"output_dir={result['output_dir']}")


if __name__ == "__main__":
    # Parallel validation imports the agent by module name in spawned workers.
    from tools.train_deadline_routes_d3qn_sac import main as canonical_main
    canonical_main()
