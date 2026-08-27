from __future__ import annotations


class DQNEdgeScheduler:
    def __init__(self, agent, state_builder, reward_function, rank_critical_beta=1.0):
        self.agent, self.state_builder, self.reward_function = agent, state_builder, reward_function
        self.rank_critical_beta = float(rank_critical_beta)

    def select_action(self, env, training=True):
        task = self.state_builder.select_ranked_ready_task(env, self.rank_critical_beta)
        workflow = env.get_state()["workflows"][task["workflow_id"]]
        workflow_state = env.workflow_by_id[task["workflow_id"]]
        vector = self.state_builder.build_state(env, task, workflow)
        context = self.state_builder.context(env)
        server_id, weights = self.agent.select_action(vector, context, training=training)
        action = {"workflow_id": task["workflow_id"], "task_id": task["task_id"],
                  "server_id": server_id, "rho": 1.0}
        return vector, context, server_id, weights, task, workflow_state, action
