from src.schedulers.base_scheduler import BaseScheduler


class FCFSScheduler(BaseScheduler):
    def select_action(self, state):
        if not state["ready_tasks"]:
            raise ValueError("FCFS scheduler received empty ready task queue")

        task = sorted(
            state["ready_tasks"],
            key=lambda item: (item["arrival_time"], item["workflow_id"], item["task_id"]),
        )[0]
        server = sorted(
            state["servers"],
            key=lambda item: (item["available_time"], item["id"]),
        )[0]
        rho = float(self.params.get("rho", 1.0))
        return {
            "workflow_id": task["workflow_id"],
            "task_id": task["task_id"],
            "server_id": server["id"],
            "rho": rho,
            "backup": False,
            "backup_server_id": None,
        }
