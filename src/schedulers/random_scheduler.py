import random

from src.schedulers.base_scheduler import BaseScheduler


class RandomScheduler(BaseScheduler):
    def reset(self):
        self.rng = random.Random(self.seed)

    def select_action(self, state):
        if not state["ready_tasks"]:
            raise ValueError("Random scheduler received empty ready task queue")
        if not state["servers"]:
            raise ValueError("Random scheduler received empty server list")

        if not hasattr(self, "rng"):
            self.reset()

        task = self.rng.choice(state["ready_tasks"])
        server = self.rng.choice(state["servers"])
        rho_levels = state.get("rho_levels") or [self.params.get("rho", 1.0)]
        rho = float(self.rng.choice(rho_levels))
        return {
            "workflow_id": task["workflow_id"],
            "task_id": task["task_id"],
            "server_id": server["id"],
            "rho": rho,
            "backup": False,
            "backup_server_id": None,
        }
