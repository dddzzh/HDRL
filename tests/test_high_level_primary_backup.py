import unittest

import numpy as np

from src.ppo.dual_agent import (
    HierarchicalPPOAgent,
    HighLevelPPOAgent,
    _HighLevelNetwork,
    _masked_softmax,
)
from src.ppo.rollout_buffer import PPORolloutBuffer

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def sample_state(server_count=3, masked_index=None):
    servers = []
    for index in range(server_count):
        server = {
            "id": index,
            "type": "edge",
            "layer": "edge",
            "max_freq_hz": 2.0e9 + index * 1.0e8,
            "static_power": 20.0,
            "dynamic_power": 60.0,
            "bandwidth_mbps": 1000.0,
            "execution_available_time": 0.0,
            "execution_queue_load": 0.0,
            "execution_queue_count": 0,
            "failure_rate": 1e-6,
            "cpu_utilization": 0.0,
        }
        if masked_index is not None and index == masked_index:
            continue
        servers.append(server)
    return {
        "time": 0.0,
        "ready_tasks": [
            {
                "workflow_id": 0,
                "task_id": 0,
                "workload_cycles": 1.0e9,
                "absolute_deadline": 100.0,
                "workflow_task_count": 1,
                "reliability": 0.99,
            }
        ],
        "servers": servers,
    }


@unittest.skipIf(torch is None, "PyTorch is required")
class HighLevelPrimaryBackupTests(unittest.TestCase):
    def make_agent(self, max_servers=3):
        return HighLevelPPOAgent(
            {
                "device": "cpu",
                "hidden_dim": 16,
                "hidden_layers": 1,
                "state": {
                    "max_ready_tasks": 4,
                    "max_servers": max_servers,
                    "time_horizon": 1000.0,
                },
                "ppo_epochs": 1,
                "batch_size": 8,
            },
            seed=123,
        )

    def test_network_output_shapes(self):
        network = _HighLevelNetwork(input_dim=7, max_servers=4, hidden_dim=8, hidden_layers=1)
        outputs = network(torch.zeros((5, 7), dtype=torch.float32))
        self.assertEqual(outputs["primary_server_logits"].shape, (5, 4))
        self.assertEqual(outputs["backup_server_logits"].shape, (5, 4))

    def test_random_sampling_primary_backup_differ(self):
        agent = self.make_agent()
        state = sample_state()
        for _ in range(50):
            decision = agent.select_assignment(state, deterministic=False)
            self.assertNotEqual(
                decision["primary_server_index"],
                decision["backup_server_index"],
            )

    def test_server_mask_applies_to_primary_and_backup(self):
        agent = self.make_agent(max_servers=4)
        state = sample_state(server_count=3)
        for _ in range(30):
            decision = agent.select_assignment(state, deterministic=False)
            self.assertLess(decision["primary_server_index"], 3)
            self.assertLess(decision["backup_server_index"], 3)

    def test_single_valid_server_raises(self):
        agent = self.make_agent(max_servers=3)
        with self.assertRaisesRegex(
            ValueError,
            "requires at least two valid servers",
        ):
            agent.select_assignment(sample_state(server_count=1), deterministic=True)

    def test_deterministic_uses_separate_valid_argmax(self):
        agent = self.make_agent(max_servers=3)
        with torch.no_grad():
            for param in agent.network.parameters():
                param.zero_()
            agent.network.primary_server_actor.bias.copy_(torch.tensor([0.1, 0.5, -0.2]))
            agent.network.backup_server_actor.bias.copy_(torch.tensor([0.2, 0.4, 0.9]))
        decision = agent.select_assignment(sample_state(), deterministic=True)
        self.assertEqual(decision["primary_server_index"], 1)
        self.assertEqual(decision["backup_server_index"], 2)

    def test_joint_log_prob_matches_evaluate_actions(self):
        agent = self.make_agent()
        state = sample_state()
        decision = agent.select_assignment(state, deterministic=True)
        states = torch.as_tensor(np.asarray([decision["state"]]), dtype=torch.float32)
        actions = torch.as_tensor(
            [[decision["primary_server_index"], decision["backup_server_index"]]],
            dtype=torch.long,
        )
        masks = {
            "server_mask": torch.as_tensor(np.asarray([decision["server_mask"]]), dtype=torch.bool)
        }
        log_probs, _, _, _, entropy = agent.evaluate_actions(states, actions, masks)
        self.assertAlmostEqual(float(log_probs.item()), decision["log_prob"], places=6)
        self.assertAlmostEqual(
            decision["entropy"],
            decision["primary_entropy"] + decision["backup_entropy"],
            places=6,
        )
        self.assertAlmostEqual(float(entropy.item()), decision["entropy"], places=6)

    def test_evaluate_actions_rejects_invalid_actions(self):
        agent = self.make_agent()
        decision = agent.select_assignment(sample_state(), deterministic=True)
        states = torch.as_tensor(np.asarray([decision["state"]]), dtype=torch.float32)
        masks = {
            "server_mask": torch.as_tensor(np.asarray([decision["server_mask"]]), dtype=torch.bool)
        }
        with self.assertRaisesRegex(ValueError, "must differ"):
            agent.evaluate_actions(states, torch.as_tensor([[0, 0]]), masks)

    def test_primary_and_backup_actors_receive_gradients(self):
        agent = self.make_agent()
        decision = agent.select_assignment(sample_state(), deterministic=True)
        states = torch.as_tensor(
            np.asarray([decision["state"], decision["state"]]),
            dtype=torch.float32,
        )
        actions = torch.as_tensor([[0, 1], [1, 2]], dtype=torch.long)
        masks = {
            "server_mask": torch.as_tensor(
                np.asarray([decision["server_mask"], decision["server_mask"]]),
                dtype=torch.bool,
            )
        }
        log_probs, _, _, _, entropy = agent.evaluate_actions(states, actions, masks)
        loss = -(log_probs + 0.01 * entropy).mean()
        agent.optimizer.zero_grad()
        loss.backward()
        primary_grad = agent.network.primary_server_actor.weight.grad
        backup_grad = agent.network.backup_server_actor.weight.grad
        self.assertIsNotNone(primary_grad)
        self.assertIsNotNone(backup_grad)
        self.assertTrue(bool(torch.isfinite(primary_grad).all().item()))
        self.assertTrue(bool(torch.isfinite(backup_grad).all().item()))
        self.assertGreater(float(primary_grad.abs().sum().item()), 0.0)
        self.assertGreater(float(backup_grad.abs().sum().item()), 0.0)

    def test_buffer_action_shape_is_two_columns(self):
        buffer = PPORolloutBuffer()
        buffer.add(
            state=np.zeros(4, dtype=np.float32),
            action=[1, 2],
            log_prob=-1.0,
            value=0.0,
            reward=0.0,
            done=False,
            masks={"server_mask": np.array([True, True, True])},
        )
        buffer.add(
            state=np.zeros(4, dtype=np.float32),
            action=(2, 0),
            log_prob=-1.0,
            value=0.0,
            reward=0.0,
            done=True,
            masks={"server_mask": np.array([True, True, True])},
        )
        self.assertEqual(np.asarray(buffer.actions).shape, (2, 2))

    def test_hierarchical_env_action_uses_primary_and_backup(self):
        agent = HierarchicalPPOAgent(
            {
                "shared": {
                    "device": "cpu",
                    "hidden_dim": 16,
                    "hidden_layers": 1,
                    "state": {
                        "max_ready_tasks": 4,
                        "max_servers": 3,
                        "time_horizon": 1000.0,
                    },
                },
                "low": {
                    "hidden_dim": 16,
                    "hidden_layers": 1,
                },
            },
            seed=123,
        )
        high, low, action = agent.select_action(sample_state(), deterministic=True)
        self.assertEqual(action["server_id"], high["primary_server_id"])
        self.assertEqual(action["backup_server_id"], high["backup_server_id"])
        self.assertTrue(action["backup"])
        self.assertIn("rho", action)
        self.assertIn("rho", low)

    def test_heuristic_backup_methods_removed(self):
        self.assertFalse(hasattr(HierarchicalPPOAgent, "_select_backup_server_id"))
        self.assertFalse(hasattr(HighLevelPPOAgent, "_select_backup_server_index"))


if __name__ == "__main__":
    unittest.main()
