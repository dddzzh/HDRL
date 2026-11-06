"""Small in-memory SAC regression checks; no experiment/checkpoint writes.

Run: python -B tools/check_sac_action_consistency.py
"""
from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOCAL_SITE_PACKAGES = ROOT / "tmp" / "py310_site_packages"
if sys.version_info[:2] == (3, 10) and LOCAL_SITE_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_SITE_PACKAGES))

import numpy as np
import torch

from src.sac.hierarchical_low_sac import (
    CONSTRAINT_FLAG_NAMES,
    REWARD_COMPONENT_NAMES,
    LowLevelDVFSContinuousSACAgent,
    LowSACActor,
)


def make_agent(action_mode="server_affine", state_mode="local_9", **extra):
    return LowLevelDVFSContinuousSACAgent({
        "device": "cpu", "action_mode": action_mode, "state_mode": state_mode,
        "state": {"max_servers": 2}, "rho_min": 0.2,
        "hidden_dim": 8, "hidden_layers": 1, "replay_size": 8,
        "batch_size": 2, "training_per_episode": 1,
        "shared_high_state_dim": 7, "shared_high_server_feature_dim": 12,
        "shared_high_global_feature_dim": 2, **extra,
    }, seed=42)


def observation():
    return SimpleNamespace(
        vector=np.zeros(7, dtype=np.float32),
        ready_tasks=[{"workload_cycles": 1e8, "relative_deadline": 10.0}],
        servers=[{"max_freq_hz": 1e9, "static_power": 10., "dynamic_power": 50.}],
        server_features=np.zeros((2, 12), dtype=np.float32),
        server_mask=np.array([True, False]),
        global_features=np.zeros(2, dtype=np.float32),
    )


def decision(agent, floor, deterministic=True):
    return agent.select_frequency(
        observation(), 0, 0, deterministic=deterministic,
        safety_metrics={"rho_hardware_min": floor, "rho_safe": 0.95},
    )


def fill_replay(agent, all_terminal=False):
    first, second = decision(agent, 0.6), decision(agent, 0.8)
    rewards = {name: 0.0 for name in (*REWARD_COMPONENT_NAMES, *CONSTRAINT_FLAG_NAMES)}
    rewards["energy_saving"] = 0.1
    for current, nxt, done in ((first, second, all_terminal), (second, second, True)):
        agent.remember_transition(
            current["state"], current["rho"], rewards,
            np.zeros_like(nxt["state"]) if done else nxt["state"], done,
            critic_state=current["critic_state"],
            next_critic_state=np.zeros_like(nxt["critic_state"]) if done else nxt["critic_state"],
        )
    return first, second


class SACActionChecks(unittest.TestCase):
    def test_affine_bounds_and_jacobian(self):
        actor = LowSACActor(2, hidden_dim=8, hidden_layers=1, action_low=0.2, action_floor_index=1)
        legacy = LowSACActor(2, hidden_dim=8, hidden_layers=1, action_low=0.2)
        legacy.load_state_dict(actor.state_dict())
        states = torch.tensor([[0., .2], [0., .6], [0., .9]])
        torch.manual_seed(7)
        raw, raw_logp, _, _ = legacy.sample(states)
        torch.manual_seed(7)
        rho, logp, _, _ = actor.sample(states)
        floors = states[:, 1:2]
        expected = floors + (1 - floors) * ((raw - .2) / .8)
        torch.testing.assert_close(rho, expected)
        torch.testing.assert_close(logp, raw_logp + math.log(.8) - torch.log(1 - floors[:, 0]))
        self.assertTrue(bool(((rho >= floors) & (rho <= 1)).all()))
        (rho.sum() + logp.sum()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_executed_replayed_and_sampled_action_equal(self):
        agent = make_agent()
        for floor in (.2, .6, .723456789, .99):
            chosen = decision(agent, floor)
            sampled = agent.actor.sample(torch.from_numpy(chosen["state"]).unsqueeze(0), True)[0].item()
            self.assertEqual(sampled, chosen["rho"])
            self.assertEqual(chosen["rho"], chosen["action_rho"])
            self.assertEqual(chosen["action_u"], chosen["rho"])
            self.assertEqual(chosen["rho_floor_binding"], 0.)
            self.assertGreaterEqual(chosen["rho"], floor)
        fill_replay(agent)
        np.testing.assert_array_equal(agent.replay.actions[:2, 0],
                                      [decision(agent, .6)["rho"], decision(agent, .8)["rho"]])

    def test_all_state_modes_and_training_q_inputs(self):
        for state_mode in ("local_9", "shared_high_selected_server", "action_conditioned_ctde"):
            with self.subTest(state_mode=state_mode):
                agent = make_agent(state_mode=state_mode)
                fill_replay(agent)
                calls = []
                def check_q_inputs(module, inputs):
                    states, actions = inputs
                    floors = states[:, agent.actor_state_dim - 1]
                    self.assertTrue(bool((actions[:, 0] >= floors).all()))
                    self.assertTrue(bool((actions[:, 0] <= 1.).all()))
                    calls.append((module, floors.detach().clone()))
                hooks = [net.register_forward_pre_hook(check_q_inputs) for net in
                         (agent.critic_1, agent.critic_2, agent.target_critic_1, agent.target_critic_2)]
                try:
                    metrics = agent.learn()
                finally:
                    for hook in hooks:
                        hook.remove()
                self.assertTrue(metrics and all(np.isfinite(v) for v in metrics.values()))
                target_floors = [f for net, f in calls if net is agent.target_critic_1]
                self.assertEqual(len(target_floors), 1)
                torch.testing.assert_close(target_floors[0], torch.tensor([.8]))
                self.assertEqual(len(calls), 6)  # two target, two replay, two actor Q calls

    def test_terminal_zero_state_never_sampled(self):
        agent = make_agent()
        fill_replay(agent, all_terminal=True)
        def forbid_target(module, inputs):
            self.fail("Terminal-only batch must not evaluate a target critic.")
        handle = agent.target_critic_1.register_forward_pre_hook(forbid_target)
        try:
            self.assertTrue(all(np.isfinite(v) for v in agent.learn().values()))
        finally:
            handle.remove()

    def test_singleton_frequency_has_no_entropy_or_actor_update(self):
        agent = make_agent()
        chosen = decision(agent, 1.)
        self.assertEqual(chosen["rho"], 1.)
        self.assertEqual(chosen["log_prob"], 0.)
        self.assertFalse(chosen["actor_update_mask"])
        rho, logp, _, _ = agent.actor.sample(torch.from_numpy(chosen["state"]).unsqueeze(0))
        self.assertEqual(rho.item(), 1.)
        self.assertEqual(logp.item(), 0.)
        self.assertFalse(decision(agent, .99999901)["actor_update_mask"])

    def test_invalid_floor_rejected(self):
        agent = make_agent()
        for floor in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                decision(agent, floor)
        with self.assertRaises(ValueError):
            agent.build_low_state(observation(), 0, 0)

    def test_invalid_replay_rejected_before_insertion(self):
        agent = make_agent()
        current = decision(agent, .6)
        rewards = {k: 0. for k in (*REWARD_COMPONENT_NAMES, *CONSTRAINT_FLAG_NAMES)}
        with self.assertRaises(ValueError):
            agent.remember_transition(current["state"], .3, rewards, current["state"], False)
        with self.assertRaises(ValueError):
            agent.remember_transition(current["state"], .8, rewards, np.zeros_like(current["state"]), False)
        self.assertEqual(agent.replay.size, 0)

    def test_affine_checkpoint_and_replay_roundtrip(self):
        agent = make_agent()
        fill_replay(agent)
        agent.learn()
        restored = make_agent()
        restored.load_state_dict(copy.deepcopy(agent.state_dict(include_replay=True)))
        self.assertEqual(decision(agent, .6)["rho"], decision(restored, .6)["rho"])
        np.testing.assert_array_equal(agent.replay.states, restored.replay.states)
        np.testing.assert_array_equal(agent.replay.actions, restored.replay.actions)
        self.assertTrue(all(np.isfinite(v) for v in restored.learn().values()))

    def test_checkpoint_modes_cannot_be_mixed(self):
        old, new = make_agent("legacy_clip"), make_agent()
        old_payload = old.state_dict()
        old_payload.pop("action_mode")  # pre-fix checkpoints have no mode tag
        with self.assertRaisesRegex(ValueError, "action mode mismatch"):
            new.load_state_dict(old_payload)
        with self.assertRaisesRegex(ValueError, "action mode mismatch"):
            old.load_state_dict(new.state_dict())
        old.load_state_dict(old_payload)

    def test_legacy_sampling_formula_unchanged(self):
        agent = make_agent("legacy_clip")
        chosen = decision(agent, .8)
        self.assertEqual(agent.actor_state_dim, 11)
        states = torch.from_numpy(chosen["state"]).unsqueeze(0)
        mean, log_std = agent.actor(states)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        torch.manual_seed(23)
        latent = distribution.rsample()
        squashed = latent.tanh()
        expected = .2 + .8 * .5 * (squashed + 1)
        logp = distribution.log_prob(latent) - torch.log(1 - squashed.pow(2) + 1e-8)
        logp += np.log(2.) - np.log(.8)
        torch.manual_seed(23)
        actual, actual_logp, _, _ = agent.actor.sample(states)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_logp, logp.sum(-1), rtol=0, atol=0)
        self.assertEqual(chosen["rho"], max(.8, agent.actor.sample(states, True)[0].item()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
