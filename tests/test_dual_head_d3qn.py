import unittest

import numpy as np

from src.D3QN.dual_head_constrained import (
    DualHeadConstrainedD3QN,
    DualHeadHighLevelD3QNAgent,
)
from src.ppo.hierarchical_rollout import HighRolloutBuffer


def _add_transition(buffer, *, workflow_id, increment, partial, deadline):
    buffer.add(
        state=np.zeros(3, dtype=np.float32),
        primary_action=0,
        backup_action=None,
        old_log_prob=0.0,
        reward_value=0.0,
        deadline_value=0.0,
        reliability_value=0.0,
        reward=0.0,
        deadline_cost=increment,
        reliability_cost=0.0,
        deadline_limit=deadline,
        reliability_limit=1.0,
        partial_makespan=partial,
        partial_log_risk=0.0,
        workflow_id=workflow_id,
        workflow_done=False,
        episode_done=False,
        next_state=None,
        primary_mask=np.ones(2, dtype=bool),
        backup_pair_mask=np.zeros((2, 2), dtype=bool),
        info={"high_base_energy_reward": 0.1},
    )


class DualHeadD3QNTests(unittest.TestCase):
    def test_binary_workflow_credit_sums_to_one_only_for_missed_workflow(self):
        buffer = HighRolloutBuffer()
        _add_transition(buffer, workflow_id="hit", increment=2.0, partial=2.0, deadline=5.0)
        _add_transition(buffer, workflow_id="hit", increment=2.0, partial=4.0, deadline=5.0)
        buffer.mark_workflow_done("hit")
        _add_transition(buffer, workflow_id="miss", increment=1.0, partial=1.0, deadline=4.0)
        _add_transition(buffer, workflow_id="miss", increment=4.0, partial=5.0, deadline=4.0)
        buffer.mark_workflow_done("miss")

        credits, misses = DualHeadHighLevelD3QNAgent.workflow_binary_deadline_credits(buffer)

        self.assertTrue(np.isclose(credits[:2].sum(), 0.0))
        self.assertTrue(np.isclose(credits[2:].sum(), 1.0))
        self.assertTrue(np.allclose(credits[2:], [0.2, 0.8]))
        self.assertEqual(misses, {"hit": 0.0, "miss": 1.0})

    def test_dense_deadline_cost_sums_to_completion_over_deadline(self):
        buffer = HighRolloutBuffer()
        _add_transition(buffer, workflow_id="hit", increment=2.0, partial=2.0, deadline=5.0)
        _add_transition(buffer, workflow_id="hit", increment=2.0, partial=4.0, deadline=5.0)
        buffer.mark_workflow_done("hit")
        _add_transition(buffer, workflow_id="miss", increment=1.0, partial=1.0, deadline=4.0)
        _add_transition(buffer, workflow_id="miss", increment=4.0, partial=5.0, deadline=4.0)
        buffer.mark_workflow_done("miss")

        costs, misses = DualHeadHighLevelD3QNAgent.workflow_dense_deadline_costs(
            buffer
        )

        self.assertTrue(np.allclose(costs[:2], [0.4, 0.4]))
        self.assertTrue(np.isclose(costs[:2].sum(), 4.0 / 5.0))
        self.assertTrue(np.allclose(costs[2:], [0.25, 1.0]))
        self.assertTrue(np.isclose(costs[2:].sum(), 5.0 / 4.0))
        self.assertEqual(misses, {"hit": 0.0, "miss": 1.0})

    def test_dense_crossing_cost_adds_one_only_at_first_deadline_crossing(self):
        buffer = HighRolloutBuffer()
        _add_transition(buffer, workflow_id="hit", increment=2.0, partial=2.0, deadline=5.0)
        _add_transition(buffer, workflow_id="hit", increment=2.0, partial=4.0, deadline=5.0)
        buffer.mark_workflow_done("hit")
        _add_transition(buffer, workflow_id="miss", increment=1.0, partial=1.0, deadline=4.0)
        _add_transition(buffer, workflow_id="miss", increment=4.0, partial=5.0, deadline=4.0)
        buffer.mark_workflow_done("miss")

        costs, misses, crossing_count = (
            DualHeadHighLevelD3QNAgent.workflow_dense_crossing_deadline_costs(
                buffer, crossing_penalty=1.0
            )
        )

        self.assertTrue(np.allclose(costs[:2], [0.4, 0.4]))
        self.assertTrue(np.allclose(costs[2:], [0.25, 2.0]))
        self.assertTrue(np.isclose(costs[:2].sum(), 4.0 / 5.0))
        self.assertTrue(np.isclose(costs[2:].sum(), 5.0 / 4.0 + 1.0))
        self.assertEqual(misses, {"hit": 0.0, "miss": 1.0})
        self.assertEqual(crossing_count, 1)

    def test_dense_miss_credit_distributes_one_over_all_makespan_increments(self):
        buffer = HighRolloutBuffer()
        _add_transition(buffer, workflow_id="hit", increment=2.0, partial=2.0, deadline=5.0)
        _add_transition(buffer, workflow_id="hit", increment=2.0, partial=4.0, deadline=5.0)
        buffer.mark_workflow_done("hit")
        _add_transition(buffer, workflow_id="miss", increment=1.0, partial=1.0, deadline=4.0)
        _add_transition(buffer, workflow_id="miss", increment=4.0, partial=5.0, deadline=4.0)
        buffer.mark_workflow_done("miss")

        costs, misses = (
            DualHeadHighLevelD3QNAgent.workflow_dense_miss_credit_deadline_costs(
                buffer, miss_penalty=1.0
            )
        )

        self.assertTrue(np.allclose(costs[:2], [0.4, 0.4]))
        self.assertTrue(np.allclose(costs[2:], [0.25 + 0.2, 1.0 + 0.8]))
        self.assertTrue(np.isclose(costs[:2].sum(), 4.0 / 5.0))
        self.assertTrue(np.isclose(costs[2:].sum(), 5.0 / 4.0 + 1.0))
        self.assertEqual(misses, {"hit": 0.0, "miss": 1.0})

    def test_combined_action_uses_energy_minus_lambda_deadline(self):
        learner = DualHeadConstrainedD3QN(
            state_dim=2,
            action_dim=2,
            hidden_dim=4,
            hidden_layers=1,
            replay_size=8,
            batch_size=2,
            device="cpu",
        )
        learner.action_q_values = lambda _state: (
            np.asarray([1.0, 0.8]),
            np.asarray([0.5, 0.0]),
        )
        combined, energy, deadline = learner.combined_action_q_values(
            np.zeros(2), deadline_lambda=1.0
        )
        self.assertTrue(np.allclose(combined, [0.5, 0.8]))
        self.assertEqual(
            learner.choose_action_from_values(
                combined, np.ones(2, dtype=bool), is_train=False
            ),
            1,
        )
        self.assertTrue(np.allclose(energy, [1.0, 0.8]))
        self.assertTrue(np.allclose(deadline, [0.5, 0.0]))


if __name__ == "__main__":
    unittest.main()
