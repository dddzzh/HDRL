import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.baselines.oo_actor_critic import OOActorCriticAgent


class OOActorCriticTests(unittest.TestCase):
    def _agent(self, *, num_servers=3, rho_levels=(0.2, 0.6, 1.0), **overrides):
        config = {
            "device": "cpu",
            "hidden_dim": 16,
            "hidden_layers": 2,
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "target_update_interval": 3,
            "num_candidates": num_servers * len(rho_levels),
            "top_s": num_servers * len(rho_levels),
            "bit_flip_probability": 0.2,
            "replay_size": 64,
            "batch_size": 4,
            "training_per_episode": 2,
        }
        config.update(overrides)
        return OOActorCriticAgent(
            state_dim=7,
            num_servers=num_servers,
            rho_levels=rho_levels,
            config=config,
            seed=7,
        )

    def test_binary_joint_action_round_trip_and_invalid_code(self):
        agent = self._agent()
        for server_index in range(agent.num_servers):
            for frequency_index in range(len(agent.rho_levels)):
                bits = agent.encode_action(server_index, frequency_index)
                self.assertEqual(bits.shape, (agent.action_dim,))
                self.assertTrue(np.isin(bits, (0.0, 1.0)).all())
                self.assertEqual(
                    agent.decode_action(bits), (server_index, frequency_index)
                )

        invalid_server = np.concatenate(
            (np.ones(agent.server_bits, dtype=np.float32), np.zeros(agent.frequency_bits))
        )
        self.assertIsNone(agent.decode_action(invalid_server))

    def test_oo_precise_stage_selects_best_valid_joint_action(self):
        agent = self._agent(num_servers=2, rho_levels=(0.2, 1.0))
        for parameter in agent.critic.parameters():
            parameter.data.zero_()
        state = np.linspace(-0.5, 0.5, agent.state_dim, dtype=np.float32)
        valid_mask = np.asarray([True, True], dtype=bool)

        def exact_score(server_index, frequency_index):
            return 10.0 if (server_index, frequency_index) == (1, 0) else -1.0

        decision = agent.select_action(
            state, valid_mask, exact_score, deterministic=True
        )
        self.assertEqual(decision["server_index"], 1)
        self.assertEqual(decision["frequency_index"], 0)
        self.assertEqual(decision["candidate_count"], 4)
        self.assertEqual(decision["top_s_count"], 4)
        self.assertEqual(agent.decode_action(decision["action_bits"]), (1, 0))

    def test_actor_critic_update_and_checkpoint_are_finite(self):
        agent = self._agent()
        rng = np.random.default_rng(11)
        for index in range(8):
            server_index = index % agent.num_servers
            frequency_index = index % len(agent.rho_levels)
            agent.replay.store(
                rng.normal(size=agent.state_dim).astype(np.float32),
                agent.encode_action(server_index, frequency_index),
                float(index) / 10.0,
                rng.normal(size=agent.state_dim).astype(np.float32),
                index == 7,
            )
        metrics = agent.update()
        self.assertEqual(metrics["updates"], 2.0)
        for key, value in metrics.items():
            self.assertTrue(math.isfinite(float(value)), key)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oo.pt"
            agent.save_checkpoint(path, episode=3, include_replay=True)
            restored = self._agent()
            payload = restored.load_checkpoint(path, load_replay=True)
            self.assertEqual(payload["episode"], 3)
            self.assertEqual(restored.replay.size, agent.replay.size)
            for original, loaded in zip(
                agent.actor.parameters(), restored.actor.parameters()
            ):
                np.testing.assert_allclose(
                    original.detach().cpu().numpy(),
                    loaded.detach().cpu().numpy(),
                )


if __name__ == "__main__":
    unittest.main()
