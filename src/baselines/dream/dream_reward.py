from __future__ import annotations

from abc import ABC, abstractmethod

from src.shared.hierarchical_state_reward import (
    FIXED_LAMBDA_UPPER_REWARD_VERSION,
    compose_fixed_lambda_upper_reward,
    low_reliability_counterfactual_cost,
)


class DReaMRewardFunction(ABC):
    @abstractmethod
    def compute_reward(self, env, task, workflow, decision, transition_info): ...


class ExistingEnvironmentReward(DReaMRewardFunction):
    """Explicit adapter to the existing environment's reward; never the paper reward."""

    def compute_reward(self, env, task, workflow, decision, transition_info):
        del env, task, workflow, decision
        if "environment_reward" not in transition_info:
            raise RuntimeError("DReaM reward function did not receive environment_reward.")
        reward = float(transition_info["environment_reward"])
        return reward, {"source": "existing_edge_cloud_environment"}


class SharedWorkflowReward(DReaMRewardFunction):
    """Use the common fixed-lambda upper reward on DReaM's action."""

    def __init__(self, objective, fixed_reliability_lambda=0.0):
        self.objective = objective
        self.fixed_reliability_lambda = float(fixed_reliability_lambda)
        if self.fixed_reliability_lambda < 0.0:
            raise ValueError("DReaM fixed reliability lambda must be non-negative.")

    def reset_episode(self):
        self.objective.reset_episode()

    def compute_reward(self, env, task, workflow, decision, transition_info):
        del task, workflow
        if self.objective.reward_mode != "deadline_incremental_only":
            raise ValueError(
                "DReaM shared upper reward requires objective.reward_mode="
                "'deadline_incremental_only'."
            )
        required = ("pre_state", "env_action", "environment_reward", "environment_info")
        missing = [key for key in required if key not in transition_info]
        if missing:
            raise RuntimeError(f"DReaM shared reward missing transition fields: {missing}.")
        pre_state = transition_info["pre_state"]
        context = pre_state["decision_context"]
        decision_payload = {
            "workflow_id": transition_info["env_action"]["workflow_id"],
            "task_id": transition_info["env_action"]["task_id"],
            "primary_server_id": decision.server_id,
            "primary_server_index": decision.server_id,
            "primary_server_mask": context["primary_server_mask"],
            "backup_pair_mask": context["backup_pair_mask"],
        }
        feedback = self.objective.compute_step_feedback(
            env, pre_state, decision_payload,
            {"rho": decision.rho, "dvfs_index": decision.dvfs_index},
            transition_info["env_action"], transition_info["environment_reward"],
            transition_info["environment_info"],
        )

        components = compose_fixed_lambda_upper_reward(
            normalized_makespan_increment=float(
                feedback["normalized_makespan_increment"]
            ),
            deadline_violation_delta=float(
                feedback["high_deadline_violation_delta"]
            ),
            energy_scale=float(feedback["task_energy_scale"]),
            energy_actual=float(feedback["primary_energy_actual"]),
            workflow_task_count=float(feedback["workflow_task_count"]),
            fixed_deadline_lambda=self.objective.fixed_deadline_lambda,
            fixed_deadline_miss_penalty=(
                getattr(
                    self.objective,
                    "deadline_incremental_violation_weight",
                    0.0,
                )
            ),
        )
        reliability = {
            "reference_reliability_violation": 0.0,
            "actual_reliability_violation": 0.0,
            "reliability_violation_delta": 0.0,
            "reliability_crossed": 0.0,
        }
        reference = transition_info.get("rho1_reference_metrics")
        event = dict(transition_info["environment_info"].get("event", {}))
        if env is not None and reference is not None:
            workflow_state = env.workflow_by_id[
                str(transition_info["env_action"]["workflow_id"])
            ]
            reliability = low_reliability_counterfactual_cost(
                env,
                workflow_state,
                int(transition_info["env_action"]["task_id"]),
                float(reference["final_reliability"]),
                float(event.get(
                    "final_reliability", reference["final_reliability"]
                )),
            )
        reliability_penalty = (
            self.fixed_reliability_lambda
            * reliability["reliability_violation_delta"]
        )
        components = {
            **components,
            **reliability,
            "fixed_reliability_lambda": self.fixed_reliability_lambda,
            "low_reliability_penalty": float(reliability_penalty),
            "reward": float(components["reward"] - reliability_penalty),
        }
        feedback.update(
            {
                "high_reward": float(components["reward"]),
                "high_reward_base": float(components["deadline_reward"]),
                "high_reward_mode": components["version"],
                "high_makespan_reward": float(components["deadline_reward"]),
                "high_makespan_reward_component": float(
                    components["deadline_reward"]
                ),
                "high_time_reward_raw": float(components["deadline_reward"]),
                "high_energy_reward_raw": float(
                    components["energy_saving_ratio"]
                ),
                "high_energy_reward": float(components["energy_reward"]),
                "high_energy_reward_component": float(
                    components["energy_reward"]
                ),
                "high_low_reward_bonus": float(components["energy_reward"]),
                "high_energy_saving_joules": float(components["energy_saving"]),
                # Keep the incremental C/D cost and the explicit unit-jump
                # miss cost visible as separate diagnostics.
                "high_deadline_penalty": float(
                    components["deadline_miss_cost"]
                ),
                "fixed_deadline_cost": float(components["deadline_cost"]),
                "task_energy_scale": float(components["energy_scale"]),
            }
        )
        return float(components["reward"]), {
            "source": (
                components["version"]
                if self.fixed_reliability_lambda == 0.0
                else f"{components['version']}+low_reliability_v1"
            ),
            "components": components,
            "feedback": feedback,
        }
