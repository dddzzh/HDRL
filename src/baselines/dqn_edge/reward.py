from __future__ import annotations

import numpy as np

from src.D3QN.reward import HierarchicalObjective
from src.shared.hierarchical_state_reward import SHARED_UPPER_REWARD_VERSION


EPS = 1e-8
COMMON_REWARD_SOURCE = SHARED_UPPER_REWARD_VERSION


class DQNEdgePaperReward:
    """Eq. 28, split into rewards for the two objective Q-networks."""

    def __init__(self):
        self.time_scale = self.energy_scale = 1.0

    def reset_episode(self):
        self.time_scale = self.energy_scale = 1.0

    def _attention_target(self, time_cost, energy_cost):
        time_cost = max(float(time_cost), 0.0)
        energy_cost = max(float(energy_cost), 0.0)
        self.time_scale = 0.99 * self.time_scale + 0.01 * max(time_cost, 1e-8)
        self.energy_scale = 0.99 * self.energy_scale + 0.01 * max(energy_cost, 1e-8)
        pressure = np.asarray(
            [time_cost / self.time_scale, energy_cost / self.energy_scale],
            np.float64,
        )
        shifted = pressure - pressure.max()
        return (np.exp(shifted) / np.exp(shifted).sum()).astype(np.float32)

    def compute(self, task, workflow_state, info, **unused):
        del unused
        time_cost, energy_cost = self._paper_costs(task, workflow_state, info)
        attention_target = self._attention_target(time_cost, energy_cost)
        return -time_cost, -energy_cost, attention_target, {
            "time_cost": time_cost, "energy_cost": energy_cost,
            "source": "wen_2026_dqn_edge_equation_28",
        }

    @staticmethod
    def _paper_costs(task, workflow_state, info):
        plan = info.get("event", info)
        finish = float(plan.get("final_aft", plan.get("aft", plan.get("finish_time", 0.0))))
        arrival = float(workflow_state["arrival_time"])
        energy = float(plan.get("total_energy", plan.get("energy", 0.0)))
        if energy == 0.0:
            energy = float(workflow_state.get("task_energy", {}).get(int(task["task_id"]), 0.0))
        return max(0.0, finish - arrival), max(0.0, energy)


class DQNEdgeNormalizedPaperReward(DQNEdgePaperReward):
    """Eq. 28 with action-independent, dimensionless reward scales."""

    def compute(
        self,
        task,
        workflow_state,
        info,
        *,
        env=None,
        state=None,
        action=None,
        env_reward=None,
    ):
        del action, env_reward
        time_cost, energy_cost = self._paper_costs(task, workflow_state, info)
        if env is not None:
            deadline = float(env.workflow_relative_deadline(workflow_state))
        else:
            deadline = float(
                workflow_state.get(
                    "relative_deadline",
                    task.get("relative_deadline", 0.0),
                )
            )
        if deadline <= EPS:
            raise ValueError("Normalized DQN-Edge reward requires a positive workflow Deadline.")
        context = (state or {}).get("decision_context", {})
        energy_scale = float(context.get("task_energy_scale", 0.0))
        if energy_scale <= EPS:
            raise ValueError("Normalized DQN-Edge reward requires task_energy_scale.")
        normalized_time_cost = time_cost / deadline
        normalized_energy_cost = energy_cost / energy_scale
        attention_target = self._attention_target(
            normalized_time_cost,
            normalized_energy_cost,
        )
        return (
            -float(normalized_time_cost),
            -float(normalized_energy_cost),
            attention_target,
            {
                "time_cost": float(normalized_time_cost),
                "energy_cost": float(normalized_energy_cost),
                "raw_time_cost": float(time_cost),
                "raw_energy_cost": float(energy_cost),
                "deadline_scale": float(deadline),
                "energy_scale": float(energy_scale),
                "source": "wen_2026_dqn_edge_equation_28_normalized",
            },
        )


class DQNEdgeCommonReward(DQNEdgePaperReward):
    """Normalized time/energy objectives for DQN-Edge's two Q-networks.

    The reward components remain separate: the time Q-network receives the
    normalized incremental Deadline objective, the energy Q-network receives
    normalized placement energy saving, and DQN-Edge's attention network alone
    learns their fusion weights.
    """

    def __init__(self, objective_config=None):
        super().__init__()
        values = dict(objective_config or {})
        values.setdefault("reward_mode", "lagrangian_deadline")
        values.setdefault("reward_profile", "ep740_lagrangian_v1")
        self.objective = HierarchicalObjective(values)

    def reset_episode(self):
        super().reset_episode()
        self.objective.reset_episode()

    def split_feedback(self, feedback):
        components = {
            "version": "dqn_edge_ep740_upper_components_v1",
            "time_reward": -float(feedback["high_deadline_constraint_cost"]),
            "energy_reward": float(feedback["high_base_energy_reward"]),
            "deadline_constraint_cost": float(
                feedback["high_deadline_constraint_cost"]
            ),
            "energy_reference": float(feedback["task_energy_scale"]),
            "energy_actual": float(feedback["primary_energy_actual"]),
            "workflow_task_count": float(feedback["workflow_task_count"]),
            "source_reward_mode": str(feedback.get("high_reward_mode", "")),
        }
        return (
            float(components["time_reward"]),
            float(components["energy_reward"]),
            components,
        )

    def compute(
        self,
        task,
        workflow_state,
        info,
        *,
        env=None,
        state=None,
        action=None,
        env_reward=None,
    ):
        if env is None or state is None or action is None:
            raise ValueError("Common DQN-Edge reward requires env, state, and action.")
        context = state.get("decision_context", {})
        servers = tuple(state.get("servers", ()))
        server_id = int(action["server_id"])
        server_index = next(
            (index for index, server in enumerate(servers) if int(server["id"]) == server_id),
            None,
        )
        if server_index is None:
            raise ValueError(f"DQN-Edge selected unknown server_id={server_id}.")
        decision = {
            "workflow_id": str(action["workflow_id"]),
            "task_id": int(action["task_id"]),
            "primary_server_id": server_id,
            "primary_server_index": int(server_index),
            "primary_server_mask": context.get("primary_server_mask"),
            "backup_pair_mask": context.get("backup_pair_mask"),
        }
        feedback = self.objective.compute_step_feedback(
            env,
            state,
            decision,
            {"rho": 1.0, "dvfs_index": None},
            action,
            env_reward,
            info,
        )
        time_reward, energy_reward, components = self.split_feedback(feedback)
        attention_target = self._attention_target(
            -time_reward,
            abs(energy_reward),
        )
        return time_reward, energy_reward, attention_target, {
            "time_cost": float(-time_reward),
            "energy_cost": float(abs(energy_reward)),
            "source": COMMON_REWARD_SOURCE,
            "components": components,
            "feedback": feedback,
            "uses_lagrangian": False,
            "reward_profile": str(getattr(self.objective, "reward_profile", "")),
        }
