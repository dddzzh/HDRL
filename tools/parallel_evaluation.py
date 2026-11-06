"""Spawn independent seed rollouts from one frozen snapshot; preserve seed order."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import copy
import importlib
import io
import multiprocessing
import os
from pathlib import Path
from tempfile import TemporaryDirectory

_runtime = None


def _initialize(snapshot, class_spec, agent_config, config, resources, flags):
    global _runtime
    os.environ["DRLEC_EVAL_WORKERS"] = "1"
    from src.device_utils import configure_torch_runtime
    from src.D3QN.reward import HierarchicalObjective
    from tools import train_hierarchical_d3qn as production
    configure_torch_runtime(1)
    cls = getattr(importlib.import_module(class_spec[0]), class_spec[1])
    agent = cls(agent_config, seed=42)
    payload = agent.load_checkpoint(snapshot)
    objective = HierarchicalObjective(config.get("objective", {}))
    objective.load_state_dict(payload.get("objective_state", {}))
    agent.high_agent.epsilon = 0.0
    attribution = config.get("agent", {}).get("low_dual_attribution")
    if attribution in {"workflow_counterfactual", "dvfs_delay_share"}:
        from tools.train_coupled_hierarchical_d3qn_sac import install_low_dual_attribution
        install_low_dual_attribution(attribution)
    _runtime = (production, config, resources, agent, objective, flags)


def _evaluate_seed(seed):
    from src.ppo.hierarchical_rollout import HighRolloutBuffer, LowRolloutBuffer
    production, config, resources, agent, objective, flags = _runtime
    smoke, rho_one, high_algorithm = flags
    with redirect_stdout(io.StringIO()):
        workflows = production.build_workflows(config, resources, seed, smoke_test=smoke)
        env = production.build_env(workflows, resources, config.get("simulation", {}), seed)
        result = production.collect_d3qn_episode(
            env, agent, objective, HighRolloutBuffer(), LowRolloutBuffer(), training=False,
            high_deterministic=True, low_deterministic=True,
            fixed_low_rho_one=rho_one, high_algorithm=high_algorithm)
    return result["metrics"]


def evaluate_parallel(config, resources, agent, objective, seed, count, workers,
                      smoke_test=False, fixed_low_rho_one=False, high_algorithm="d3qn"):
    from tools.hierarchical_common import attach_satisfaction_metrics, average_metrics
    if int(count) < 1 or int(workers) < 1:
        raise ValueError("Evaluation count and workers must be positive")
    print(f"[eval-parallel] seeds={seed}-{int(seed) + int(count) - 1} "
          f"workers={min(int(workers), int(count))} frozen_model=True", flush=True)
    agent_config = copy.deepcopy(agent.config)
    # Match the parent's neural device; do not silently switch CPU/GPU numerics.
    agent_config.setdefault("shared", {})["device"] = str(agent.high_agent.agent_R.device)
    class_spec = (type(agent).__module__, type(agent).__name__)
    if class_spec[0] == "__main__":
        class_spec = ("tools.train_coupled_hierarchical_d3qn_sac", class_spec[1])
    with TemporaryDirectory(prefix="drlec_eval_") as directory:
        snapshot = Path(directory) / "frozen.pt"
        agent.save_checkpoint(snapshot, objective_state=objective.state_dict(),
                              config=config, include_replay=False)
        with ProcessPoolExecutor(
                max_workers=min(int(workers), int(count)),
                mp_context=multiprocessing.get_context("spawn"), initializer=_initialize,
                initargs=(str(snapshot), class_spec, agent_config, config, resources,
                          (smoke_test, fixed_low_rho_one, high_algorithm))) as pool:
            rows = list(pool.map(_evaluate_seed, range(int(seed), int(seed) + int(count))))
    return attach_satisfaction_metrics(average_metrics(rows), rows)
