from __future__ import annotations

from dataclasses import replace

from src.simulator.resource import map_discrete_rho_to_server_range

from .dream_action import DREAM_RHO_LEVELS, decode_action


class DReaMScheduler:
    def __init__(self, agent, state_builder=None, reward_function=None, enable_fault_tolerance=False):
        if state_builder is None: raise RuntimeError("DReaM state builder is not connected.")
        if reward_function is None: raise RuntimeError("DReaM reward function is not connected.")
        self.agent, self.state_builder, self.reward_function = agent, state_builder, reward_function
        self.enable_fault_tolerance = bool(enable_fault_tolerance)

    def reset(self): pass

    def select_action(self, env, training=True):
        state = env.get_state(); task = state["ready_tasks"][0]
        workflow = state["workflows"][task["workflow_id"]]
        context = env.build_hierarchical_decision_context(task["workflow_id"], task["task_id"])
        vector = self.state_builder.build_state(env, task, workflow, context)
        action_id = self.agent.select_action(vector, training=training)
        decision = decode_action(action_id, len(env.servers), self.enable_fault_tolerance)
        server = next(
            row for row in env.servers if int(row["id"]) == decision.server_id
        )
        decision = replace(
            decision,
            rho=map_discrete_rho_to_server_range(
                server, DREAM_RHO_LEVELS, decision.dvfs_index
            ),
        )
        if decision.ft_index is not None:
            raise RuntimeError("Adapted DReaM fault-tolerance execution is intentionally not connected.")
        action = {"workflow_id": task["workflow_id"], "task_id": task["task_id"],
                  "server_id": decision.server_id, "rho": decision.rho}
        return vector, action_id, decision, action
