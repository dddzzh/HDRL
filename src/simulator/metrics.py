def compute_summary(workflow_results, event_log, servers):
    total_energy = sum(float(event.get("energy", 0.0)) for event in event_log)
    finished_workflows = [
        workflow for workflow in workflow_results if workflow.get("finish_time") is not None
    ]
    logical_task_exec_times = []
    logical_task_waiting_times = []
    logical_task_communication_times = []
    logical_task_total_times = []
    for workflow in workflow_results:
        for task in workflow.get("task_plans", []):
            if task.get("logical_task_exec_time") is not None:
                logical_task_exec_times.append(float(task["logical_task_exec_time"]))
            if task.get("logical_task_waiting_time") is not None:
                logical_task_waiting_times.append(float(task["logical_task_waiting_time"]))
            if task.get("logical_task_communication_time") is not None:
                logical_task_communication_times.append(float(task["logical_task_communication_time"]))
            if task.get("logical_task_total_time") is not None:
                logical_task_total_times.append(float(task["logical_task_total_time"]))
    backup_events = [
        event
        for event in event_log
        if event.get("replica_role") == "backup" or event.get("queue_type") == "backup"
    ]
    backup_energy = sum(float(event.get("energy", 0.0)) for event in backup_events)

    absolute_deadline_rows = [
        workflow
        for workflow in finished_workflows
        if workflow.get("absolute_deadline") is not None
    ]
    if absolute_deadline_rows:
        deadline_values = [
            bool(workflow.get("deadline_satisfied"))
            for workflow in absolute_deadline_rows
        ]
        deadline_violation_sum = sum(
            float(workflow.get("deadline_violation", 0.0) or 0.0)
            for workflow in absolute_deadline_rows
        )
    else:
        deadline_values = []
        deadline_violation_sum = 0.0
    deadline_total_count = len(deadline_values)
    deadline_success_count = sum(1 for value in deadline_values if value)
    deadline_violation_count = (
        deadline_total_count - deadline_success_count
        if deadline_total_count
        else 0
    )
    reliability_values = [
        workflow.get("reliability_met")
        for workflow in workflow_results
        if workflow.get("reliability_req") not in (None, "")
        and workflow.get("reliability") is not None
    ]
    reliability_total_count = len(reliability_values)
    reliability_success_count = sum(1 for value in reliability_values if value)
    reliability_violation_count = (
        reliability_total_count - reliability_success_count
        if reliability_total_count
        else 0
    )

    busy_by_server = {int(server["id"]): 0.0 for server in servers}
    for event in event_log:
        server_id = int(event["server_id"])
        busy_by_server[server_id] = busy_by_server.get(server_id, 0.0) + float(event["exec_time"])

    horizon = 0.0
    if event_log:
        horizon = max(float(event["finish_time"]) for event in event_log)
    avg_server_utilization = 0.0
    if servers and horizon > 0:
        avg_server_utilization = sum(
            busy_by_server.get(int(server["id"]), 0.0) / horizon for server in servers
        ) / len(servers)

    return {
        "total_energy": total_energy,
        "avg_workflow_makespan": _avg(finished_workflows, "makespan"),
        "max_workflow_makespan": _max(finished_workflows, "makespan"),
        "avg_workflow_compute_span": _avg(finished_workflows, "compute_span"),
        "max_workflow_compute_span": _max(finished_workflows, "compute_span"),
        "avg_workflow_response_time": _avg(finished_workflows, "response_time"),
        "max_workflow_response_time": _max(finished_workflows, "response_time"),
        "avg_workflow_waiting_time": _avg(finished_workflows, "waiting_time"),
        "max_workflow_waiting_time": _max(finished_workflows, "waiting_time"),
        "total_finished_workflows": len(finished_workflows),
        "deadline_success_count": deadline_success_count,
        "deadline_violation_count": deadline_violation_count,
        "deadline_violation_sum": deadline_violation_sum,
        "deadline_total_count": deadline_total_count,
        "deadline_success_rate": (
            deadline_success_count / deadline_total_count
            if deadline_values
            else None
        ),
        "avg_workflow_reliability": _avg(finished_workflows, "reliability"),
        "reliability_success_count": reliability_success_count,
        "reliability_violation_count": reliability_violation_count,
        "reliability_total_count": reliability_total_count,
        "reliability_success_rate": (
            reliability_success_count / reliability_total_count
            if reliability_values
            else None
        ),
        "avg_task_execution_time": (
            sum(logical_task_exec_times) / len(logical_task_exec_times)
            if logical_task_exec_times
            else 0.0
        ),
        "avg_task_waiting_time": (
            sum(logical_task_waiting_times) / len(logical_task_waiting_times)
            if logical_task_waiting_times
            else 0.0
        ),
        "max_task_waiting_time": (
            max(logical_task_waiting_times) if logical_task_waiting_times else 0.0
        ),
        "avg_task_communication_time": (
            sum(logical_task_communication_times) / len(logical_task_communication_times)
            if logical_task_communication_times
            else 0.0
        ),
        "max_task_communication_time": (
            max(logical_task_communication_times) if logical_task_communication_times else 0.0
        ),
        "avg_task_total_time": (
            sum(logical_task_total_times) / len(logical_task_total_times)
            if logical_task_total_times
            else 0.0
        ),
        "max_task_total_time": max(logical_task_total_times) if logical_task_total_times else 0.0,
        "avg_server_utilization": avg_server_utilization,
        "backup_replica_count": len(backup_events),
        "backup_energy": backup_energy,
    }


def _avg(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return sum(values) / len(values) if values else 0.0


def _max(rows, field):
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return max(values) if values else 0.0
