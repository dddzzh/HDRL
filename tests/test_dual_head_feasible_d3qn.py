import unittest

import numpy as np
import torch
from torch import nn

from src.D3QN.dual_head_feasible import (
    FeasibleRiskDualHeadD3QN,
    FeasibleRiskHighLevelD3QNAgent,
    WorkflowRiskReplay,
)
from src.ppo.hierarchical_rollout import HighRolloutBuffer
from tools.hierarchical_common import _workflow_balanced_risk_calibration


class _ConstantNetwork(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", torch.as_tensor(values, dtype=torch.float32))

    def forward(self, states):
        return self.values.unsqueeze(0).expand(states.shape[0], -1)


def _add_transition(buffer, *, workflow_id, partial, deadline, action=0):
    buffer.add(
        state=np.asarray([partial, deadline], dtype=np.float32),
        primary_action=action,
        backup_action=None,
        old_log_prob=0.0,
        reward_value=0.0,
        deadline_value=0.0,
        reliability_value=0.0,
        reward=0.0,
        deadline_cost=0.0,
        reliability_cost=0.0,
        deadline_limit=deadline,
        reliability_limit=1.0,
        partial_makespan=partial,
        partial_log_risk=0.0,
        workflow_id=workflow_id,
        workflow_done=False,
        episode_done=False,
        next_state=None,
        primary_mask=np.asarray([True, True]),
        backup_pair_mask=np.zeros((2, 2), dtype=bool),
        info={},
    )


class FeasibleRiskD3QNTests(unittest.TestCase):
    def _learner(self, **overrides):
        args = dict(
            state_dim=2,
            action_dim=3,
            hidden_dim=8,
            hidden_layers=1,
            replay_size=100,
            batch_size=4,
            risk_min_workflows=2,
            risk_gate_min_completed_workflows=2,
            risk_anneal_start_episode=1,
            risk_anneal_end_episode=3,
            device="cpu",
            seed=7,
        )
        args.update(overrides)
        return FeasibleRiskDualHeadD3QN(**args)

    def test_terminal_label_is_shared_without_delta_c_allocation(self):
        buffer = HighRolloutBuffer()
        _add_transition(buffer, workflow_id="hit", partial=1.0, deadline=3.0)
        _add_transition(buffer, workflow_id="hit", partial=2.0, deadline=3.0)
        _add_transition(buffer, workflow_id="miss", partial=1.0, deadline=3.0)
        _add_transition(buffer, workflow_id="miss", partial=4.0, deadline=3.0)
        labels = FeasibleRiskHighLevelD3QNAgent.workflow_terminal_labels(buffer)
        self.assertEqual(labels, {"hit": 0.0, "miss": 1.0})

    def test_workflow_replay_samples_workflows_then_steps(self):
        replay = WorkflowRiskReplay(
            state_dim=2,
            max_transitions=100,
            batch_size=1000,
            positive_fraction=0.3,
            min_workflows=2,
            seed=3,
        )
        replay.add_workflow(np.zeros((1, 2)), np.zeros(1), 1.0)
        replay.add_workflow(np.ones((20, 2)), np.ones(20), 0.0)
        batch = replay.sample()
        self.assertAlmostEqual(float(np.mean(batch["labels"])), 0.3, places=2)
        corrected = float(np.average(batch["labels"], weights=batch["weights"]))
        self.assertAlmostEqual(corrected, 0.5, places=2)

    def test_feasible_set_prefers_energy_then_falls_back_to_minimum_risk(self):
        learner = self._learner(risk_threshold_start=0.2, risk_threshold_end=0.1)
        learner.risk_replay.completed_count = 2
        energy = np.asarray([0.5, 0.9, 0.8])
        mask = np.asarray([True, True, True])
        action, feasible, fallback = learner._greedy_action_numpy(
            energy, np.asarray([0.1, 0.3, 0.15]), mask
        )
        self.assertEqual(action, 2)
        self.assertTrue(np.array_equal(feasible, [True, False, True]))
        self.assertFalse(fallback)
        action, feasible, fallback = learner._greedy_action_numpy(
            energy, np.asarray([0.4, 0.3, 0.5]), mask
        )
        self.assertEqual(action, 1)
        self.assertFalse(feasible.any())
        self.assertTrue(fallback)

    def test_energy_target_uses_the_same_feasible_selector(self):
        learner = self._learner(risk_threshold_start=0.2, risk_threshold_end=0.1)
        learner.risk_replay.completed_count = 2
        learner.energy_q_eval = _ConstantNetwork([0.5, 0.9, 0.8])
        # Sigmoid logits give risks approximately [0.1, 0.3, 0.15].
        learner.risk_logit_net = _ConstantNetwork(
            [-2.1972246, -0.8472979, -1.734601]
        )
        actions = learner._next_actions(
            torch.zeros((2, 2)),
            torch.as_tensor([[True, True, True], [True, False, True]]),
        )
        self.assertTrue(torch.equal(actions.cpu(), torch.as_tensor([2, 2])))

    def test_threshold_curriculum(self):
        learner = self._learner(risk_threshold_start=0.2, risk_threshold_end=0.01)
        learner.episodes_seen = 1
        self.assertAlmostEqual(learner.current_risk_threshold, 0.2)
        learner.episodes_seen = 2
        self.assertAlmostEqual(learner.current_risk_threshold, 0.105)
        learner.episodes_seen = 3
        self.assertAlmostEqual(learner.current_risk_threshold, 0.01)

    def test_rollout_calibration_weights_workflows_equally(self):
        buffer = HighRolloutBuffer()
        _add_transition(buffer, workflow_id="miss", partial=4.0, deadline=3.0)
        for _ in range(4):
            _add_transition(buffer, workflow_id="hit", partial=2.0, deadline=3.0)
        decisions = [{"risk_selected": 0.5} for _ in range(buffer.size)]
        metrics = _workflow_balanced_risk_calibration(buffer, decisions)
        # Equal workflow weight gives mean label 0.5 and Brier 0.25 even
        # though the successful workflow contains four times as many tasks.
        self.assertAlmostEqual(metrics["risk_rollout_brier_score"], 0.25)
        self.assertAlmostEqual(metrics["risk_rollout_ece"], 0.0)


if __name__ == "__main__":
    unittest.main()
