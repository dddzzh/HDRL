"""In-memory route checks; no environment rollout or experiment training."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_deadline_routes_d3qn_sac as routes
from tools import train_coupled_hierarchical_d3qn_sac as coupled
from src.sac.experimental_deadline_routes import BatchFailureLowDeadlineController
from src.sac.hierarchical_low_sac import REWARD_COMPONENT_NAMES, CONSTRAINT_FLAG_NAMES
from src.D3QN.reliability_aware_high import ReliabilityLagrangian
import numpy as np
import torch

FILES = {
    "a_slack": "configs/deadline_route_a_slack_sm74_sac_affine_20261106.json",
    "b_dual": "configs/deadline_route_b_dual_sm74_sac_affine_20261106.json",
}


def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def tiny_config(route):
    config = read(FILES[route])
    config["agent"]["shared"]["state"] = {"max_servers": 2, "max_ready_tasks": 2}
    for block in ("shared", "high", "low"):
        config["agent"][block]["device"] = "cpu"
    for block in ("high", "low"):
        config["agent"][block].update(hidden_dim=8, hidden_layers=1, replay_size=8,
                                     batch_size=2, training_per_episode=1)
    return config


def agent(route):
    return routes.DeadlineRouteHierarchicalAgent(tiny_config(route)["agent"], seed=42)


class DeadlineRouteChecks(unittest.TestCase):
    def test_configuration_preserves_environment_and_upper_deadline(self):
        base = read("configs/sac_action_fix_sm74_20261105.json")
        for name, filename in FILES.items():
            config = read(filename)
            for key in ("simulation", "resources", "generator", "objective"):
                self.assertEqual(config[key], base[key])
            self.assertEqual(config["agent"]["high"]["lagrangian"], base["agent"]["high"]["lagrangian"])
            training = copy.deepcopy(base["training"])
            training["checkpoint_dir"] = f"results/deadline_routes_sm74_sac_affine_20261106/{name}"
            self.assertEqual(config["training"], training)
            self.assertEqual(routes.validate_agent_config(config["agent"]), name)
            self.assertEqual(config["agent"]["low"]["action_mode"], "server_affine")

    def test_route_a_stays_zero_without_fixed_penalty_fallback(self):
        low = agent("a_slack").low_agent
        self.assertIsNotNone(low.deadline_lagrangian)
        for episode in range(200):
            low.update_deadline_lagrangian([1.] * 20, episode)
            self.assertEqual(low.deadline_lambda, 0.)
        values = torch.tensor([[0., 1., 0., 0., 0.]])
        stats = {"mean": np.zeros(5), "std": np.ones(5)}
        cost = torch.tensor([3.])
        reward = low._compose_rewards(values, torch.zeros((1, 2)), stats, cost, torch.zeros(1))
        torch.testing.assert_close(reward, torch.tensor([-1.]))

    def test_route_b_uses_batch_failure_and_boundary_is_inclusive(self):
        cfg = copy.deepcopy(read(FILES["b_dual"])["agent"]["low"]["lagrangian"])
        cfg.update(warmup_episodes=0, window_episodes=10)
        controller = BatchFailureLowDeadlineController.from_config(cfg)
        for episode in range(10):
            row = controller.update([0.] * 18 + [1.] * 2, episode)
        self.assertEqual(row["low_deadline_episode_failed"], 0.)  # DSR=.90 passes.
        self.assertEqual(row["low_deadline_constraint_value"], 0.)
        self.assertLess(controller.value, 1.)
        for episode in range(10, 20):
            row = controller.update([0.] * 17 + [1.] * 3, episode)
        self.assertEqual(row["low_deadline_constraint_value"], 1.)
        self.assertGreater(controller.value, 1.)
        for episode in range(20, 220):
            controller.update([0.] * 20, episode)
        self.assertEqual(controller.value, 0.)

    def test_route_b_has_no_fixed_time_term(self):
        low = agent("b_dual").low_agent
        values = torch.tensor([[0., 1., 0., 0., 0.]])
        stats = {"mean": np.zeros(5), "std": np.ones(5)}
        torch.testing.assert_close(low._base_reward(values, stats), torch.zeros(1))
        reward = low._compose_rewards(values, torch.zeros((1, 2)), stats, torch.ones(1), torch.zeros(1))
        torch.testing.assert_close(reward, torch.tensor([-1.]))

    def test_dense_risk_replay_keeps_energy_units_and_affine_action(self):
        for name in FILES:
            low = agent(name).low_agent
            state = np.zeros(low.low_state_dim, dtype=np.float32)
            state[-1] = .4
            components = {key: 0. for key in (*REWARD_COMPONENT_NAMES, *CONSTRAINT_FLAG_NAMES)}
            components.update(energy_saving=7., energy_saving_ratio=.1,
                              reliability_budget_share_spent=.35)
            low.remember_transition(state, .7, components, np.zeros_like(state), True)
            self.assertAlmostEqual(float(low.replay.reward_components[0, 4]), .35, places=6)
            self.assertEqual(float(low.replay.reward_components[0, 0]), 7.)
            self.assertAlmostEqual(float(low.replay.actions[0, 0]), .7, places=6)
            self.assertEqual(components["reliability_violation"], 0.)
            for bad in (-1., float("nan")):
                invalid = {**components, "reliability_budget_share_spent": bad}
                with self.assertRaises(ValueError):
                    low.remember_transition(state, .7, invalid, np.zeros_like(state), True)
            self.assertEqual(low.replay.size, 1)

    def test_reliability_controllers_are_two_sided_and_legacy_is_unchanged(self):
        for name in FILES:
            value = agent(name)
            value.high_agent.update_reliability_lagrangian([0.] * 20)
            value.low_agent.update_reliability_lagrangian([0.] * 20, 0)
            self.assertAlmostEqual(value.high_agent.lambda_reliability, .9975)
            self.assertAlmostEqual(value.low_agent.reliability_lambda, .9975)
            value.high_agent.update_reliability_lagrangian([1.] * 20)
            value.low_agent.update_reliability_lagrangian([1.] * 20, 1)
            self.assertGreater(value.high_agent.lambda_reliability, .9975)
            self.assertGreater(value.low_agent.reliability_lambda, .9975)
        legacy = ReliabilityLagrangian(lambda_init=1., dual_lr=.001)
        legacy.update(0.)
        self.assertEqual(legacy.lambda_reliability, 1.)

    def test_network_initialization_same_across_routes(self):
        first, second = agent("a_slack"), agent("b_dual")
        for name in ("energy_q_eval", "deadline_q_eval", "reliability_q_eval"):
            left = getattr(first.high_agent.agent_R, name).state_dict()
            right = getattr(second.high_agent.agent_R, name).state_dict()
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        for name in ("actor", "critic_1", "critic_2"):
            left, right = getattr(first.low_agent, name).state_dict(), getattr(second.low_agent, name).state_dict()
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)

    def test_collector_uses_true_misses_not_continuous_credit(self):
        workflows = [dict(response_time=12., relative_deadline=10., deadline_satisfied=False,
                          reliability=.94, reliability_req=.95, reliability_met=False),
                     dict(response_time=8., relative_deadline=10., deadline_satisfied=True,
                          reliability=.98, reliability_req=.95, reliability_met=True)]
        source = dict(workflow_results=workflows, low_deadline_dual_costs=[.001, 0.],
                      low_reliability_dual_costs=[0., 0.], untouched_reward=[.002, .003])
        with patch.object(routes, "_ORIGINAL_COLLECT", return_value=copy.deepcopy(source)) as original:
            result = routes.collect_route_episode(training=False)
        self.assertEqual(original.call_args.kwargs["low_dual_attribution"], "workflow_counterfactual")
        self.assertEqual(result["low_deadline_dual_costs"], [1., 0.])
        self.assertEqual(result["low_reliability_dual_costs"], [1., 0.])
        self.assertEqual(result["untouched_reward"], source["untouched_reward"])

    def test_full_checkpoint_roundtrip_and_canonical_worker_loader(self):
        for name in FILES:
            config = tiny_config(name)
            first = agent(name)
            first.high_agent.update_reliability_lagrangian([1.] * 20)
            first.low_agent.update_reliability_lagrangian([1.] * 20, 100)
            for episode in range(12):
                first.low_agent.update_deadline_lagrangian([1.] * 20, episode + 100)
            with tempfile.TemporaryDirectory(prefix="drlec_route_check_") as folder:
                checkpoint = Path(folder)/"full.pt"
                first.save_checkpoint(checkpoint, config=config, episode=12)
                payload, second, _ = routes.load_route_checkpoint(checkpoint)
                self.assertEqual(payload["checkpoint_format"], routes.FORMAT)
                self.assertEqual(first.low_agent.deadline_lagrangian.state_dict(), second.low_agent.deadline_lagrangian.state_dict())
                self.assertEqual(first.high_agent.reliability_lagrangian.state_dict(), second.high_agent.reliability_lagrangian.state_dict())
                self.assertEqual(first.low_agent.reliability_lagrangian.state_dict(), second.low_agent.reliability_lagrangian.state_dict())
                other = agent("b_dual" if name == "a_slack" else "a_slack")
                with self.assertRaises(ValueError):
                    other.load_checkpoint(checkpoint)
                code = ("from tools.train_deadline_routes_d3qn_sac import load_route_checkpoint; "
                        "import sys,torch; torch.set_num_threads(1); "
                        "p,a,o=load_route_checkpoint(sys.argv[1]); "
                        "assert type(a).__module__=='tools.train_deadline_routes_d3qn_sac'; print('worker-load-ok')")
                result = subprocess.run([sys.executable, "-B", "-c", code, str(checkpoint)], cwd=ROOT,
                                        capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("worker-load-ok", result.stdout)

    def test_missing_or_mixed_replay_modes_rejected(self):
        for name in FILES:
            low = agent(name).low_agent
            payload = copy.deepcopy(low.state_dict())
            del payload["reliability_cost_mode"]
            with self.assertRaises(ValueError):
                low.load_state_dict(payload)
            payload = copy.deepcopy(low.state_dict())
            payload["deadline_route"] = "b_dual" if name == "a_slack" else "a_slack"
            with self.assertRaises(ValueError):
                low.load_state_dict(payload)

    def test_training_dispatch_restores_legacy_globals(self):
        original_agent = routes.production.HierarchicalD3QNAgent
        original_collect = routes.production.collect_d3qn_episode
        def fake(*args, **kwargs):
            self.assertIs(routes.production.HierarchicalD3QNAgent, routes.DeadlineRouteHierarchicalAgent)
            self.assertIs(routes.production.collect_d3qn_episode, routes.collect_route_episode)
            return {"output_dir": "not-run"}
        with patch.object(routes.production, "run_training", side_effect=fake):
            self.assertEqual(routes.run_training(tiny_config("b_dual"))["output_dir"], "not-run")
        self.assertIs(routes.production.HierarchicalD3QNAgent, original_agent)
        self.assertIs(routes.production.collect_d3qn_episode, original_collect)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
