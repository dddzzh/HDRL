from src.schedulers.base_scheduler import BaseScheduler
from src.simulator.resource import get_actual_freq_hz


class FastestServerScheduler(BaseScheduler):
    def select_action(self, state):
        if not state["ready_tasks"]:
            raise ValueError("Fastest scheduler received empty ready task queue")
        if not state["servers"]:
            raise ValueError("Fastest scheduler received empty server list")

        task = sorted(
            state["ready_tasks"],
            key=lambda item: (item["arrival_time"], item["workflow_id"], item["task_id"]),
        )[0]
        server = sorted(
            state["servers"],
            key=lambda item: (-get_actual_freq_hz(item, 1.0), item["id"]),
        )[0]
        return {
            "workflow_id": task["workflow_id"],
            "task_id": task["task_id"],
            "server_id": server["id"],
            "rho": 1.0,
            "backup": False,
            "backup_server_id": None,
        }
