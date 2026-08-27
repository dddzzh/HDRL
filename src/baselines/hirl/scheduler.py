from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.hierarchical_state_reward import (
    low_reliability_counterfactual_cost,
)


@dataclass(frozen=True)
class HIRLDecision:
    server_id: int
    server_index: int
    rho: float
    load: float
    epsilon: float


class HIRLScheduler:
    def __init__(self, agent, state_builder):
        self.agent = agent
        self.state_builder = state_builder

    def select_action(self, env, training=True):
        state = env.get_state()
        task = state["ready_tasks"][0]
        workflow = state["workflows"][task["workflow_id"]]
        context = state["decision_context"]
        vector = self.state_builder.build_state(env, task, workflow, context)
        server_count = len(state["servers"])
        mask = np.zeros(self.agent.num_servers, dtype=bool)
        mask[:server_count] = True
        load = self.load_metric(state)
        policy = self.agent.select_action(vector, mask, load, training=training)
        index = policy["server_index"]
        server_id = int(state["servers"][index]["id"])
        decision = HIRLDecision(
            server_id=server_id,
            server_index=index,
            rho=float(policy["rho"]),
            load=load,
            epsilon=float(policy["epsilon"]),
        )
        env_action = {
            "workflow_id": task["workflow_id"],
            "task_id": task["task_id"],
            "server_id": server_id,
            "rho": decision.rho,
        }
        return policy, decision, env_action

    def rewards(self, env, pre_state, task, event, rho1_reference):
        cfg = self.agent.config
        deadline = max(float(event.get("relative_deadline") or task.get("relative_deadline") or 1.0), 1e-8)
        latency = max(float(event.get("task_total_time_primary", 0.0)), 0.0) / deadline
        context = pre_state.get("decision_context", {})
        energy_scale = max(float(context.get("task_energy_scale", 1.0)), 1e-8)
        energy = max(float(event.get("energy", 0.0)), 0.0) / energy_scale
        deadline_failed = not bool(event.get("deadline_satisfied", True))
        reliability_required = float(task.get("reliability", 0.0) or 0.0)
        reliability_failed = float(event.get("final_reliability", 1.0)) < reliability_required
        failure = deadline_failed or reliability_failed
        priority = float(task.get("priority", task.get("priority_level", 1.0)) or 1.0)
        success = 0.0 if failure else cfg.completion_reward * priority
        workflow_state = env.workflow_by_id[str(task["workflow_id"])]
        reliability = low_reliability_counterfactual_cost(
            env,
            workflow_state,
            int(task["task_id"]),
            float(rho1_reference["final_reliability"]),
            float(event.get(
                "final_reliability", rho1_reference["final_reliability"]
            )),
        )
        reliability_penalty = (
            cfg.fixed_reliability_lambda
            * reliability["reliability_violation_delta"]
        )
        # In this adapted HIRL implementation TD3 selects rho and receives the
        # variable named upper_reward. Charge only that frequency controller
        # for reliability damage relative to the same-server rho=1 reference.
        upper = success - cfg.energy_weight * energy - reliability_penalty
        lower = -(
            cfg.latency_weight * priority * latency
            + (1.0 - cfg.latency_weight) * energy
        ) - cfg.failure_penalty * float(failure)
        return upper, lower, failure, {
            "normalized_latency": latency,
            "normalized_energy": energy,
            "deadline_failed": deadline_failed,
            "reliability_failed": reliability_failed,
            **reliability,
            "low_reliability_penalty": float(reliability_penalty),
        }

    @staticmethod
    def load_metric(state):
        servers = state.get("servers", [])
        if servers:
            queue = float(np.mean([np.tanh(max(float(row.get("queue_load", 0.0)), 0.0)) for row in servers]))
            utilization = float(np.mean([np.clip(float(row.get("cpu_utilization", 0.0)), 0.0, 1.0) for row in servers]))
        else:
            queue = utilization = 0.0
        ready = min(len(state.get("ready_tasks", [])) / 32.0, 1.0)
        return float(np.clip(0.6 * queue + 0.2 * utilization + 0.2 * ready, 0.0, 1.0))
