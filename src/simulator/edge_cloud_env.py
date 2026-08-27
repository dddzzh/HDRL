import copy
import math
import random

import numpy as np

from src.simulator.event_queue import EventQueue, EventType
from src.schedulers.heft import (
    HEFTScheduler,
    calculate_upward_ranks,
    reference_heft_makespan,
)
from src.simulator.deadline import cai_2025_deadline_fields, validate_deadline_type
from src.simulator.metrics import compute_summary
from src.simulator.network import get_link_bandwidth
from src.simulator.resource import (
    clone_runtime_resources,
    compute_energy,
    compute_execution_time,
    compute_transfer_time,
)
from src.simulator.reliability import task_reliability_from_server, workflow_reliability
from src.simulator.sub_deadline import distribute_existing_workflow_deadline
from src.simulator.workflow import Task, Workflow, create_workflow_state, is_task_ready


RELIABILITY_EPS = 1e-12


class EdgeCloudWorkflowEnv:
    def __init__(self, workflows, resources, sim_config, seed=42):
        self.initial_workflows = copy.deepcopy(workflows)
        self.initial_resources = copy.deepcopy(resources)
        self.sim_config = sim_config or {}
        self.seed = seed
        self.rng = random.Random(seed)
        self.in_run = False
        self.event_queue = EventQueue()
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)
        self.ready_task_order = self._ready_task_order()
        self.reference_makespan_cache = {}
        self.resources = clone_runtime_resources(self.initial_resources)
        self.servers = self.resources["servers"]
        network = self.resources.get("network", {})
        self.link_bandwidth_mbps = dict(network.get("link_bandwidth_mbps", {}))
        self.bandwidth_mbps = float(network.get("bandwidth_mbps", 100.0))
        self.reliability_exponent = float(self.sim_config.get("reliability_exponent", 1.0))
        self.enable_backup_duplication = bool(self.sim_config.get("enable_backup_duplication", False))
        self.workflow_states = [
            create_workflow_state(copy.deepcopy(workflow))
            for workflow in self.initial_workflows
        ]
        for state in self.workflow_states:
            self._initialize_deadline_fields(state)
            self._initialize_sub_deadline_fields(state)
        self.workflow_upward_ranks = {}
        if self.ready_task_order == "upward_rank" or bool(
            self.sim_config.get("compute_upward_rank_feature", False)
        ):
            self.workflow_upward_ranks = {
                state["workflow_id"]: calculate_upward_ranks(
                    state["workflow"],
                    self.servers,
                    self.link_bandwidth_mbps,
                    rho=float(self.sim_config.get("upward_rank_rho", 1.0)),
                )
                for state in self.workflow_states
            }
        self.workflow_by_id = {
            state["workflow_id"]: state for state in self.workflow_states
        }
        self.current_time = 0.0
        self.done = False
        self.event_log = []
        self.trace_count = 0
        self._pending_finish_events = 0
        self.server_real_timeline = {int(server["id"]): [] for server in self.servers}
        self._server_busy_completed = {int(server["id"]): 0.0 for server in self.servers}
        self._server_busy_cursor = {int(server["id"]): 0 for server in self.servers}
        self.server_history_ema_alpha = float(self.sim_config.get("server_history_ema_alpha", 0.9))
        self.server_history_ema_alpha = min(max(self.server_history_ema_alpha, 0.0), 0.999)
        self.server_history_stats = {
            int(server["id"]): {
                "wait_time_ema": 0.0,
                "queued_processing_time_ema": 0.0,
                "queue_count_ema": 0.0,
                "recent_primary_assignments": 0.0,
                "recent_backup_assignments": 0.0,
            }
            for server in self.servers
        }

        self.event_queue = EventQueue()
        for state in self.workflow_states:
            self.event_queue.push(
                state["arrival_time"],
                EventType.WORKFLOW_ARRIVAL,
                workflow_id=state["workflow_id"],
            )
        self._process_events_through(self.current_time)
        return self.get_state()

    def get_state(self):
        ready_tasks = self._get_ready_tasks()
        state = {
            "time": self.current_time,
            "ready_task_order": self.ready_task_order,
            "ready_tasks": ready_tasks,
            "workflows": {
                state["workflow_id"]: state["workflow"]
                for state in self.workflow_states
            },
            "link_bandwidth_mbps": dict(self.link_bandwidth_mbps),
            "servers": [
                {
                    "id": server["id"],
                    "server_id": server.get("server_id", server["id"]),
                    "available_time": self._server_execution_available_time(server),
                    "execution_available_time": self._server_execution_available_time(server),
                    "backup_available_time": self._server_backup_available_time(server),
                    "max_freq_hz": server["max_freq_hz"],
                    "max_freq_ghz": server.get("max_freq_ghz", server["max_freq_hz"] / 1e9),
                    "type": server["type"],
                    "server_type": server.get("server_type", server["type"]),
                    "layer": server.get("layer", server["type"]),
                    "static_power": server["static_power"],
                    "dynamic_power": server["dynamic_power"],
                    "active_power_reference": server.get(
                        "active_power_reference",
                        server["static_power"] + server["dynamic_power"],
                    ),
                    "bandwidth_mbps": self._server_bandwidth(server),
                    "queue_load": self._server_queue_load(server),
                    "queue_count": self._server_queue_count(server),
                    "execution_queue_load": self._server_queue_load(server, queue_type="execution"),
                    "execution_queue_count": self._server_queue_count(server, queue_type="execution"),
                    "backup_queue_load": self._server_queue_load(server, queue_type="backup"),
                    "backup_queue_count": self._server_queue_count(server, queue_type="backup"),
                    "failure_rate": server.get("failure_rate", 0.0),
                    "min_rho": server.get("min_rho", 0.1),
                    "cpu_utilization": self._server_utilization(server),
                }
                for server in self.servers
            ],
        }
        if ready_tasks:
            current_task = ready_tasks[0]
            state["decision_context"] = self.build_hierarchical_decision_context(
                current_task["workflow_id"],
                current_task["task_id"],
            )
        return state

    def step(self, action):
        workflow_id = str(action.get("workflow_id"))
        task_id = int(action.get("task_id"))
        server_id = int(action.get("server_id"))
        backup_server_id = action.get("backup_server_id")
        primary_server = self._get_server(server_id)
        primary_rho = self._normalize_rho(action.get("rho"), primary_server)

        workflow_state = self.workflow_by_id.get(workflow_id)
        if workflow_state is None:
            raise ValueError(f"Unknown workflow_id in action: {workflow_id}")
        if not workflow_state["released"]:
            raise ValueError(f"Workflow is not released yet: {workflow_id}")
        if not is_task_ready(workflow_state, task_id):
            raise ValueError(f"Task is not ready: workflow={workflow_id}, task={task_id}")

        task = self._get_task(workflow_state["workflow"], task_id)
        primary_plan = self.estimate_replica_plan(
            task=task,
            server=primary_server,
            rho=primary_rho,
            workflow_state=workflow_state,
            queue_type="execution",
            replica_role="primary",
        )
        need_backup, trigger_reason = self.check_backup_trigger(
            task,
            primary_plan,
            workflow_state,
        )
        backup_plan = None
        if need_backup:
            backup_plan = self.select_backup_plan(
                task,
                primary_plan,
                workflow_state,
                backup_server_id=backup_server_id,
            )

        final_plan = self.combine_primary_backup(
            primary_plan,
            backup_plan,
            trigger_reason,
            task=task,
            workflow_state=workflow_state,
        )
        event = self.commit_task_plan(final_plan, workflow_state)

        self._update_done()
        return self.get_state(), -float(final_plan["total_energy"]), self.done, {"event": event}

    def estimate_action_metrics_ref_rho1(self, action):
        workflow_id = str(action.get("workflow_id"))
        task_id = int(action.get("task_id"))
        server_id = int(action.get("server_id"))
        backup_server_id = action.get("backup_server_id")
        primary_server = self._get_server(server_id)

        workflow_state = self.workflow_by_id.get(workflow_id)
        if workflow_state is None:
            raise ValueError(f"Unknown workflow_id in action: {workflow_id}")
        if not workflow_state["released"]:
            raise ValueError(f"Workflow is not released yet: {workflow_id}")
        if not is_task_ready(workflow_state, task_id):
            raise ValueError(f"Task is not ready: workflow={workflow_id}, task={task_id}")

        task = self._get_task(workflow_state["workflow"], task_id)
        primary_plan = self.estimate_replica_plan(
            task=task,
            server=primary_server,
            rho=1.0,
            workflow_state=workflow_state,
            queue_type="execution",
            replica_role="primary",
        )
        need_backup, trigger_reason = self.check_backup_trigger(
            task,
            primary_plan,
            workflow_state,
        )
        backup_plan = None
        if need_backup:
            backup_plan = self.select_backup_plan(
                task,
                primary_plan,
                workflow_state,
                backup_server_id=backup_server_id,
            )
        final_plan = self.combine_primary_backup(
            primary_plan,
            backup_plan,
            trigger_reason,
            task=task,
            workflow_state=workflow_state,
        )
        finish_times = [
            float(value)
            for value in workflow_state.get("task_finish_times", {}).values()
            if value is not None
        ]
        workflow_arrival_time = float(workflow_state["arrival_time"])
        workflow_partial_makespan_before = (
            max(0.0, max(finish_times) - workflow_arrival_time)
            if finish_times
            else 0.0
        )
        return {
            "total_energy": float(final_plan["total_energy"]),
            "primary_aft": float(final_plan["primary_plan"]["aft"]),
            "final_aft": float(final_plan["final_aft"]),
            "final_reliability": float(final_plan["final_reliability"]),
            "workflow_arrival_time": workflow_arrival_time,
            "workflow_partial_makespan_before": float(
                workflow_partial_makespan_before
            ),
        }

    def estimate_action_energy_ref(self, action):
        return float(self.estimate_action_metrics_ref_rho1(action)["total_energy"])

    def estimate_primary_energy_ref_rho1(self, action):
        workflow_id = str(action.get("workflow_id"))
        task_id = int(action.get("task_id"))
        server_id = int(action.get("server_id"))
        workflow_state = self.workflow_by_id.get(workflow_id)
        if workflow_state is None:
            raise ValueError(f"Unknown workflow_id in action: {workflow_id}")
        task = self._get_task(workflow_state["workflow"], task_id)
        server = self._get_server(server_id)
        primary_plan = self.estimate_replica_plan(
            task=task,
            server=server,
            rho=1.0,
            workflow_state=workflow_state,
            queue_type="execution",
            replica_role="primary",
        )
        return float(primary_plan["energy"])

    def compute_low_safety_metrics(self, workflow_id, task_id, server_id):
        workflow_id = str(workflow_id)
        task_id = int(task_id)
        server_id = int(server_id)
        workflow_state = self.workflow_by_id[workflow_id]
        task = self._get_task(workflow_state["workflow"], task_id)
        server = self._get_server(server_id)
        rho_hardware_min = float(server.get("min_rho", self.sim_config.get("rho_min", 0.1)))
        rho_hardware_min = min(max(rho_hardware_min, 0.0), 1.0)
        plan_rho1 = self.estimate_replica_plan(
            task=task,
            server=server,
            rho=1.0,
            workflow_state=workflow_state,
            queue_type="execution",
            replica_role="primary",
        )
        workflow_absolute_deadline = self.workflow_absolute_deadline(workflow_state)
        workflow_relative_deadline = self.workflow_relative_deadline(workflow_state)
        task_sub_deadline_relative = None
        task_sub_deadline_absolute = None
        if bool(workflow_state.get("sub_deadline_enabled", False)):
            (
                task_sub_deadline_relative,
                task_sub_deadline_absolute,
            ) = self.task_sub_deadline(workflow_state, task_id)
        final_download_time = self._estimate_final_download_time_if_exit(workflow_state, task_id, server)
        task_latest_finish_time = float(workflow_absolute_deadline) - float(final_download_time)
        margin = float(self.sim_config.get("low_deadline_safety_margin_ratio", 0.05))
        safe_execution_budget = (1.0 - margin) * (
            float(task_latest_finish_time) - float(plan_rho1["ast"])
        )
        execution_time_rho_1 = float(plan_rho1["exec_time"])
        tolerance = float(self.sim_config.get("low_rho_feasibility_tolerance", 1e-8))
        if safe_execution_budget <= 0.0:
            rho_deadline_required = float("inf")
            low_action_feasible = False
            rho_safe = 1.0
        else:
            rho_deadline_required = execution_time_rho_1 / max(safe_execution_budget, 1e-12)
            low_action_feasible = rho_deadline_required <= 1.0 + tolerance
            rho_safe = 1.0 if not low_action_feasible else max(rho_hardware_min, rho_deadline_required)
            rho_safe = min(max(rho_safe, rho_hardware_min), 1.0)
        return {
            "workflow_id": workflow_id,
            "task_id": task_id,
            "server_id": server_id,
            "current_time": float(self.current_time),
            "workflow_arrival_time": float(workflow_state["arrival_time"]),
            "workflow_absolute_deadline": float(workflow_absolute_deadline),
            "workflow_relative_deadline": float(workflow_relative_deadline),
            "workflow_task_count": float(
                self._workflow_task_count(workflow_state["workflow"])
            ),
            "task_sub_deadline_relative": task_sub_deadline_relative,
            "task_sub_deadline_absolute": task_sub_deadline_absolute,
            "estimated_ast": float(plan_rho1["ast"]),
            "execution_time_rho_1": execution_time_rho_1,
            "task_reliability_rho_1": float(plan_rho1["reliability"]),
            "task_latest_finish_time": float(task_latest_finish_time),
            "safe_execution_budget": float(safe_execution_budget),
            "rho_hardware_min": float(rho_hardware_min),
            "rho_deadline_required": float(rho_deadline_required),
            "rho_safe": float(rho_safe),
            "low_action_feasible": bool(low_action_feasible),
        }

    def build_hierarchical_decision_context(self, workflow_id, task_id) -> dict:
        """构建分层 PPO 共用的安全决策元数据，不改变网络状态向量。"""
        workflow_id = str(workflow_id)
        task_id = int(task_id)
        server_count = len(self.servers)
        if server_count <= 0:
            raise RuntimeError("Cannot build decision context without servers.")

        primary_server_mask = np.zeros(server_count, dtype=bool)
        server_safety_metrics = {}
        server_energy_refs = []

        for index, server in enumerate(self.servers):
            server_id = int(server["id"])
            metrics = self.compute_low_safety_metrics(workflow_id, task_id, server_id)
            low_action_feasible = bool(metrics["low_action_feasible"])
            # Structural mask only: every real server is selectable. Deadline
            # feasibility remains diagnostic/reward information for SAC/D3QN.
            primary_server_mask[index] = True
            primary_energy_ref_rho1 = float(
                self.estimate_primary_energy_ref_rho1(
                    {
                        "workflow_id": workflow_id,
                        "task_id": task_id,
                        "server_id": server_id,
                        "rho": 1.0,
                    }
                )
            )
            if primary_energy_ref_rho1 > 0.0 and math.isfinite(primary_energy_ref_rho1):
                server_energy_refs.append(primary_energy_ref_rho1)
            rho_deadline_required = metrics.get("rho_deadline_required")
            server_safety_metrics[str(server_id)] = {
                "server_id": server_id,
                "rho_safe": float(metrics["rho_safe"]),
                "rho_deadline": (
                    None
                    if rho_deadline_required is None or not math.isfinite(float(rho_deadline_required))
                    else float(rho_deadline_required)
                ),
                "rho_reliability": None,
                "low_action_feasible": low_action_feasible,
                "estimated_ast": float(metrics["estimated_ast"]),
                "execution_time_rho_1": float(metrics["execution_time_rho_1"]),
                "task_reliability_rho_1": float(metrics["task_reliability_rho_1"]),
                "safe_execution_budget": float(metrics["safe_execution_budget"]),
                "task_latest_finish_time": float(metrics["task_latest_finish_time"]),
                "current_time": float(metrics["current_time"]),
                "workflow_relative_deadline": float(metrics["workflow_relative_deadline"]),
                "workflow_task_count": float(metrics["workflow_task_count"]),
                "workflow_absolute_deadline": float(metrics["workflow_absolute_deadline"]),
                "task_sub_deadline_relative": metrics.get(
                    "task_sub_deadline_relative"
                ),
                "task_sub_deadline_absolute": metrics.get(
                    "task_sub_deadline_absolute"
                ),
                "primary_energy_ref_rho1": primary_energy_ref_rho1,
                **self._server_history_metrics(server_id),
            }

        if not server_energy_refs:
            raise RuntimeError(
                f"No positive primary energy reference for workflow={workflow_id}, task={task_id}."
            )
        task_energy_scale = float(
            np.median(np.asarray(server_energy_refs, dtype=np.float64))
        )

        backup_pair_mask = np.zeros((server_count, server_count), dtype=bool)
        if self.enable_backup_duplication:
            for primary_index in range(server_count):
                for backup_index in range(server_count):
                    backup_pair_mask[primary_index, backup_index] = (
                        bool(primary_server_mask[primary_index])
                        and backup_index != primary_index
                    )

        return {
            "workflow_id": workflow_id,
            "task_id": task_id,
            "primary_server_mask": primary_server_mask,
            "backup_pair_mask": backup_pair_mask,
            "backup_enabled": bool(self.enable_backup_duplication),
            "server_safety_metrics": server_safety_metrics,
            "task_energy_scale": task_energy_scale,
            "primary_server_mask_mode": "structural_only",
            "shield_relaxed": False,
            "shield_relaxation_reason": None,
        }

    def _estimate_final_download_time_if_exit(self, workflow_state, task_id, server):
        # The scheduling boundary ends when the exit task finishes. IoT result
        # delivery is outside the model, so no final download consumes slack.
        del workflow_state, task_id, server
        return 0.0

    def _child_task_ids(self, workflow, task_id):
        task_id = int(task_id)
        if isinstance(workflow, Workflow):
            return [int(edge.dst) for edge in workflow.edges if int(edge.src) == task_id]
        return [int(edge["dst"]) for edge in workflow.get("edges", []) if int(edge["src"]) == task_id]

    def run(self, scheduler):
        self.in_run = True
        self.reset()
        scheduler.reset()
        max_steps = sum(self._workflow_task_count(state["workflow"]) for state in self.workflow_states) + 1000
        steps = 0

        while not self.done and steps < max_steps:
            state = self.get_state()
            if not state["ready_tasks"]:
                if not self._advance_to_next_decision_point():
                    self._update_done()
                    if not self.done:
                        raise RuntimeError(
                            "Simulation stalled with unfinished workflows and no future events"
                        )
                    break
                continue

            if isinstance(scheduler, HEFTScheduler):
                action = scheduler.select_action(
                    state,
                    context_builder=self.build_hierarchical_decision_context,
                )
            else:
                action = scheduler.select_action(state)
            self._trace_schedule_decision(state, action)
            self.step(action)
            steps += 1

        if steps >= max_steps and not self.done:
            raise RuntimeError(f"Simulation exceeded max steps: {max_steps}")

        workflow_results = self._build_workflow_results()
        summary = compute_summary(workflow_results, self.event_log, self.servers)
        return {
            "summary": summary,
            "workflow_results": workflow_results,
            "event_log": list(self.event_log),
        }

    def estimate_replica_plan(
        self,
        task,
        server,
        rho,
        workflow_state,
        queue_type="execution",
        replica_role="primary",
        primary_plan=None,
    ):
        task_id = int(task["id"])
        server_id = int(server["id"])
        replica_role = str(replica_role)
        if replica_role not in {"primary", "backup"}:
            raise ValueError(f"Unknown replica_role: {replica_role}")
        if replica_role == "backup" and primary_plan is None:
            raise ValueError("primary_plan is required when estimating a backup replica")

        ready_info = self.compute_replica_ready_times(
            workflow_state,
            task_id,
            server_id,
            replica_role=replica_role,
        )
        dependency_ready_time = float(ready_info["dependency_ready_time"])
        data_ready_time = float(ready_info["data_ready_time"])
        transfer_time = float(ready_info["transfer_time"])
        upload_time = float(ready_info["upload_time"])
        ready_queue_waiting_time = max(0.0, self.current_time - data_ready_time)
        execution_available_before = self._server_execution_available_time(server)
        backup_available_before = self._server_backup_available_time(server)
        available_time = self._server_queue_available_time(server, queue_type)
        if replica_role == "backup":
            base_ready_time = max(
                self.current_time,
                workflow_state["arrival_time"],
                float(primary_plan["ast"]),
                data_ready_time,
            )
        else:
            base_ready_time = max(
                self.current_time,
                workflow_state["arrival_time"],
                data_ready_time,
            )
        ast = max(base_ready_time, available_time)
        task_communication_time = max(0.0, data_ready_time - dependency_ready_time)
        task_waiting_time = max(0.0, ast - data_ready_time)
        server_queue_waiting_time = task_waiting_time
        exec_time = compute_execution_time(task["workload_cycles"], server, rho)
        aft = ast + exec_time
        task_total_time = task_waiting_time + task_communication_time + exec_time
        energy = compute_energy(task["workload_cycles"], server, rho)
        reliability, failure_rate = task_reliability_from_server(
            server,
            rho,
            exec_time,
            exponent_default=self.reliability_exponent,
        )
        plan = {
            "task_id": task_id,
            "workflow_id": workflow_state["workflow_id"],
            "server_id": server_id,
            "server_type": server.get("server_type", server["type"]),
            "layer": server.get("layer", server["type"]),
            "rho": float(rho),
            "frequency": float(rho) * float(server["max_freq_hz"]),
            "current_time": float(self.current_time),
            "workflow_arrival_time": float(workflow_state["arrival_time"]),
            "replica_role": replica_role,
            "dependency_ready_time": dependency_ready_time,
            "task_ready_time": dependency_ready_time,
            "ready_queue_waiting_time": ready_queue_waiting_time,
            "ready_time": data_ready_time,
            "data_ready_time": data_ready_time,
            "base_ready_time": base_ready_time,
            "task_communication_time": task_communication_time,
            "task_waiting_time": task_waiting_time,
            "task_total_time": task_total_time,
            "server_execution_available_time_before": execution_available_before,
            "server_backup_available_time_before": backup_available_before,
            "available_time": available_time,
            "ast": ast,
            "aft": aft,
            "duration": aft - float(workflow_state["arrival_time"]),
            "exec_time": exec_time,
            "upload_time": upload_time,
            "transfer_time": transfer_time,
            "server_queue_waiting_time": server_queue_waiting_time,
            "energy": energy,
            "reliability": reliability,
            "base_failure_rate": float(server.get("failure_rate", 0.0)),
            "failure_rate": failure_rate,
            "min_freq_hz": float(server.get("min_freq_hz", 0.0)),
            "max_freq_hz": float(server["max_freq_hz"]),
            "failure_rate_exponent": float(
                server.get("failure_rate_exponent", self.reliability_exponent)
            ),
            "queue_type": queue_type,
            "backup_path_uses_primary_fallback": bool(
                ready_info.get("backup_path_uses_primary_fallback", False)
            ),
        }
        if replica_role == "backup":
            plan["primary_ast"] = float(primary_plan["ast"])
        self._assert_replica_time_plan(plan)
        return plan

    def check_backup_trigger(self, task, primary_plan, workflow_state):
        if not self.enable_backup_duplication:
            return False, "none"

        deadline_risk = self.compute_task_deadline_violation(
            task,
            workflow_state,
            primary_plan["aft"],
        ) > 0.0
        reliability_risk = self._workflow_reliability_risk(
            workflow_state,
            int(task["id"]),
            primary_plan["reliability"],
        )

        if deadline_risk and reliability_risk:
            return True, "both"
        if deadline_risk:
            return True, "deadline"
        if reliability_risk:
            return True, "reliability"
        return False, "none"

    def select_backup_plan(self, task, primary_plan, workflow_state, backup_server_id=None):
        if backup_server_id is None:
            return None
        primary_server_id = int(primary_plan["server_id"])
        backup_server_id = int(backup_server_id)
        if backup_server_id == primary_server_id:
            return None
        backup_server = self._get_server(backup_server_id)
        require_deadline_feasible = bool(
            self.sim_config.get("require_backup_deadline_feasible", True)
        )
        candidates = []

        for rho in self._backup_rho_candidates(backup_server, primary_plan["rho"]):
            backup_plan = self.estimate_replica_plan(
                task=task,
                server=backup_server,
                rho=rho,
                workflow_state=workflow_state,
                queue_type="backup",
                replica_role="backup",
                primary_plan=primary_plan,
            )
            combined_reliability = self._combined_task_reliability(
                primary_plan["reliability"],
                backup_plan["reliability"],
            )
            reliability_improved = combined_reliability > primary_plan["reliability"]
            backup_deadline_violation = self.compute_task_deadline_violation(
                task,
                workflow_state,
                backup_plan["aft"],
            )
            backup_deadline_feasible = backup_deadline_violation <= 0.0
            if not reliability_improved:
                continue
            if require_deadline_feasible and not backup_deadline_feasible:
                continue
            backup_plan["combined_reliability"] = combined_reliability
            backup_plan["backup_deadline_violation"] = backup_deadline_violation
            backup_plan["backup_deadline_satisfied"] = backup_deadline_feasible
            candidates.append(
                (
                    backup_deadline_violation,
                    -combined_reliability,
                    backup_plan["energy"],
                    backup_plan["aft"],
                    backup_plan,
                )
            )

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[:4])
        return candidates[0][4]

    def combine_primary_backup(
        self,
        primary_plan,
        backup_plan,
        trigger_reason="none",
        task=None,
        workflow_state=None,
    ):
        final_aft = primary_plan["aft"]
        logical_aft = primary_plan["aft"]
        final_server_id = primary_plan["server_id"]
        logical_source_server = primary_plan["server_id"]
        final_rho = primary_plan["rho"]
        normal_aft = primary_plan["aft"]
        normal_source_server = primary_plan["server_id"]
        if backup_plan is None:
            failure_aft = None
            failure_source_server = None
            final_reliability = primary_plan["reliability"]
            total_energy = primary_plan["energy"]
            backup_enabled = False
        else:
            failure_aft = backup_plan["aft"]
            failure_source_server = backup_plan["server_id"]
            final_reliability = self._combined_task_reliability(
                primary_plan["reliability"],
                backup_plan["reliability"],
            )
            total_energy = primary_plan["energy"] + backup_plan["energy"]
            backup_enabled = True

        absolute_deadline = None
        relative_deadline = None
        deadline_satisfied = None
        deadline_violation = 0.0
        primary_deadline_violation = 0.0
        backup_deadline_violation = None
        backup_deadline_satisfied = None
        if task is not None and workflow_state is not None:
            absolute_deadline = self.workflow_absolute_deadline(workflow_state)
            relative_deadline = self.workflow_relative_deadline(workflow_state)
            deadline_satisfied = self.is_task_deadline_satisfied(task, workflow_state, logical_aft)
            deadline_violation = self.compute_task_deadline_violation(
                task,
                workflow_state,
                logical_aft,
            )
            primary_deadline_violation = deadline_violation
            if backup_plan is not None:
                backup_deadline_violation = self.compute_task_deadline_violation(
                    task,
                    workflow_state,
                    backup_plan["aft"],
                )
                backup_deadline_satisfied = backup_deadline_violation <= 0.0

        return {
            "workflow_id": primary_plan["workflow_id"],
            "task_id": primary_plan["task_id"],
            "primary_plan": primary_plan,
            "backup_plan": backup_plan,
            "backup_enabled": backup_enabled,
            "backup_trigger_reason": trigger_reason,
            "final_aft": final_aft,
            "final_reliability": final_reliability,
            "final_server_id": final_server_id,
            "logical_aft": logical_aft,
            "logical_source_server": logical_source_server,
            "final_rho": final_rho,
            "total_energy": total_energy,
            "reliability_gain": final_reliability - primary_plan["reliability"],
            "winning_replica_role": "primary",
            "normal_aft": normal_aft,
            "failure_aft": failure_aft,
            "normal_source_server": normal_source_server,
            "failure_source_server": failure_source_server,
            "absolute_deadline": absolute_deadline,
            "relative_deadline": relative_deadline,
            "deadline_satisfied": deadline_satisfied,
            "deadline_violation": deadline_violation,
            "primary_deadline_violation": primary_deadline_violation,
            "backup_deadline_violation": backup_deadline_violation,
            "backup_deadline_satisfied": backup_deadline_satisfied,
        }

    def commit_task_plan(self, final_plan, workflow_state):
        task_id = int(final_plan["task_id"])
        primary_plan = final_plan["primary_plan"]
        backup_plan = final_plan["backup_plan"]

        primary_server = self._get_server(primary_plan["server_id"])
        self.update_server_timeline(primary_server, primary_plan, "primary")
        self._set_server_execution_available_time(primary_server, primary_plan["aft"])
        if backup_plan is not None:
            backup_server = self._get_server(backup_plan["server_id"])
            self.update_server_timeline(backup_server, backup_plan, "backup")
            self._set_server_backup_available_time(backup_server, backup_plan["aft"])
            self._set_server_execution_available_time(backup_server, backup_plan["aft"])
        self._update_server_history_stats(primary_plan, backup_plan)

        workflow_state["task_status"][task_id] = "running"
        workflow_state["task_start_times"][task_id] = primary_plan["ast"]
        workflow_state["task_finish_times"][task_id] = primary_plan["aft"]
        workflow_state["task_server"][task_id] = primary_plan["server_id"]
        workflow_state["task_rho"][task_id] = primary_plan["rho"]
        workflow_state["task_energy"][task_id] = final_plan["total_energy"]
        workflow_state["task_reliability"][task_id] = final_plan["final_reliability"]
        workflow_state.setdefault("task_primary_finish_times", {})[task_id] = primary_plan["aft"]
        workflow_state.setdefault("task_primary_server", {})[task_id] = primary_plan["server_id"]
        workflow_state.setdefault("task_backup_finish_times", {})[task_id] = (
            backup_plan["aft"] if backup_plan else None
        )
        workflow_state.setdefault("task_backup_server", {})[task_id] = (
            backup_plan["server_id"] if backup_plan else None
        )
        workflow_state["task_primary_plan"][task_id] = dict(primary_plan)
        workflow_state["task_backup_plan"][task_id] = dict(backup_plan) if backup_plan else None
        workflow_state["task_final_plan"][task_id] = dict(final_plan)
        workflow_state["energy_used"] = (
            float(workflow_state.get("energy_used", 0.0)) + final_plan["total_energy"]
        )
        workflow_state["reliability_product"] = workflow_reliability(
            workflow_state["task_reliability"].values()
        )

        if primary_plan["upload_time"] > 0 and workflow_state.get("upload_time") is None:
            workflow_state["upload_time"] = primary_plan["upload_time"]

        primary_event = self._replica_event(primary_plan, final_plan, "primary")
        self.event_log.append(primary_event)
        if backup_plan is not None:
            self.event_log.append(self._replica_event(backup_plan, final_plan, "backup"))

        final_event = self._final_task_event(final_plan)
        self._trace_schedule_result(final_event)
        self.event_queue.push(
            primary_plan["aft"],
            EventType.TASK_FINISH,
            workflow_id=final_plan["workflow_id"],
            task_id=task_id,
        )
        self._pending_finish_events += 1
        return final_event

    def _advance_to_next_decision_point(self):
        while not self._get_ready_tasks():
            if self.event_queue.empty():
                return False
            next_time = self.event_queue.peek().time
            self.current_time = next_time
            self._process_events_through(self.current_time)
        return True

    def _process_events_through(self, through_time: float):
        while True:
            next_event = self.event_queue.peek()
            if next_event is None or next_event.time > through_time:
                break
            self._process_next_event()

    def _process_next_event(self):
        event = self.event_queue.pop()
        if event is None:
            return
        self.current_time = event.time
        if event.event_type == EventType.WORKFLOW_ARRIVAL:
            self._handle_workflow_arrival(event.payload["workflow_id"])
        elif event.event_type == EventType.TASK_FINISH:
            self._handle_task_finish(
                event.payload["workflow_id"],
                int(event.payload["task_id"]),
            )

    def _handle_workflow_arrival(self, workflow_id: str):
        state = self.workflow_by_id.get(workflow_id)
        if state is None or state["released"]:
            return
        state["released"] = True
        self._trace(
            "ARRIVAL "
            f"time={self.current_time:.6f} "
            f"workflow={workflow_id} "
            f"arrival_time={state['arrival_time']:.6f} "
            f"tasks={self._workflow_task_count(state['workflow'])}"
        )

    def _handle_task_finish(self, workflow_id: str, task_id: int):
        workflow_state = self.workflow_by_id.get(workflow_id)
        if workflow_state is None:
            return

        status = workflow_state["task_status"].get(task_id)
        if status != "running":
            return

        workflow_state["task_status"][task_id] = "finished"
        workflow_state["task_finish_times"][task_id] = self.current_time
        self._pending_finish_events = max(0, self._pending_finish_events - 1)
        self._update_workflow_completion(workflow_state)
        self._trace(
            "FINISH "
            f"time={self.current_time:.6f} "
            f"workflow={workflow_id} "
            f"task={task_id}"
        )

    def _get_ready_tasks(self):
        ready_tasks = []
        for state in self.workflow_states:
            if not state["released"] or state["completed"]:
                continue
            workflow = state["workflow"]
            if isinstance(workflow, Workflow):
                task_items = ((task.id, task) for task in workflow.tasks)
            else:
                task_items = ((int(task["id"]), task) for task in workflow["tasks"])
            for task_id, task in sorted(task_items, key=lambda item: item[0]):
                if is_task_ready(state, task_id):
                    workload_cycles = (
                        task.workload_cycles if isinstance(task, Task) else task["workload_cycles"]
                    )
                    task_payload = {"id": task_id, "workload_cycles": workload_cycles}
                    task_ready_time = self.compute_task_ready_time(task_payload, state)
                    absolute_deadline = self.workflow_absolute_deadline(state)
                    meta = self._workflow_task_metadata(state, workflow)
                    sub_deadline_relative = None
                    sub_deadline_absolute = None
                    if bool(state.get("sub_deadline_enabled", False)):
                        (
                            sub_deadline_relative,
                            sub_deadline_absolute,
                        ) = self.task_sub_deadline(state, task_id)
                    ready_tasks.append(
                        {
                            "workflow_id": state["workflow_id"],
                            "task_id": task_id,
                            "arrival_time": state["arrival_time"],
                            "workload_cycles": workload_cycles,
                            "absolute_deadline": absolute_deadline,
                            "relative_deadline": self.workflow_relative_deadline(state),
                            "sub_deadline_relative": sub_deadline_relative,
                            "sub_deadline_absolute": sub_deadline_absolute,
                            "task_ready_time": task_ready_time,
                            "ready_queue_waiting_time": max(0.0, self.current_time - task_ready_time),
                            "workflow_task_count": self._workflow_task_count(workflow),
                            "energy_budget": meta["energy_budget"],
                            "reliability": meta["reliability"],
                            "workflow_energy_used": meta["workflow_energy_used"],
                            "workflow_reliability": meta["workflow_reliability"],
                            "upward_rank": float(
                                self.workflow_upward_ranks.get(
                                    state["workflow_id"], {}
                                ).get(task_id, 0.0)
                            ),
                        }
                    )
        # Task dispatch is an environment rule shared by every placement
        # algorithm.  The policy only chooses a server (and, where applicable,
        # a frequency) for ready_tasks[0].
        if self.ready_task_order == "upward_rank":
            ready_tasks.sort(
                key=lambda item: (
                    -float(item["upward_rank"]),
                    item["arrival_time"],
                    item["workflow_id"],
                    item["task_id"],
                )
            )
        else:
            ready_tasks.sort(
                key=lambda item: (
                    float(item.get("workload_cycles", 0.0)),
                    item["arrival_time"],
                    item["workflow_id"],
                    item["task_id"],
                )
            )
        return ready_tasks

    def _ready_task_order(self):
        aliases = {
            "upward_rank": "upward_rank",
            "uprank": "upward_rank",
            "rank_u": "upward_rank",
            "smallest_workload": "smallest_workload",
            "sjf": "smallest_workload",
        }
        configured = str(
            self.sim_config.get("ready_task_order", "smallest_workload")
        ).strip().lower()
        if configured not in aliases:
            raise ValueError(
                f"invalid ready_task_order={configured!r}; expected one of "
                f"{sorted(aliases)}"
            )
        return aliases[configured]

    def _update_done(self):
        if self._pending_finish_events > 0:
            self.done = False
            return
        if self.event_queue.empty() and all(state["completed"] for state in self.workflow_states):
            self.done = True
            return
        if self.event_queue.empty():
            self.done = all(state["completed"] for state in self.workflow_states)
            return
        self.done = False

    def _update_workflow_completion(self, workflow_state):
        if all(status == "finished" for status in workflow_state["task_status"].values()):
            workflow_state["completed"] = True
            self._finalize_egress(workflow_state)

    def _finalize_egress(self, workflow_state):
        if workflow_state.get("egress_applied"):
            return

        compute_finish = max(workflow_state["task_finish_times"].values())
        workflow_state["download_time"] = 0.0
        workflow_state["egress_finish_time"] = compute_finish
        workflow_state["egress_applied"] = True
        self._trace(
            "EGRESS "
            f"workflow={workflow_state['workflow_id']} "
            "download=0.000000 "
            f"finish={workflow_state['egress_finish_time']:.6f}"
        )

    def _link_bandwidth(self, src_layer: str, dst_layer: str) -> float:
        return get_link_bandwidth(self.link_bandwidth_mbps, src_layer, dst_layer)

    def _initialize_deadline_fields(self, workflow_state):
        workflow = workflow_state["workflow"]
        existing_deadline = self._workflow_field(workflow, "absolute_deadline")

        if existing_deadline is not None:
            absolute_deadline = float(existing_deadline)
            relative_deadline = self._workflow_field(workflow, "relative_deadline")
            if relative_deadline is None:
                relative_deadline = absolute_deadline - float(workflow_state["arrival_time"])
            reference_makespan = self._workflow_field(workflow, "reference_makespan")
            if reference_makespan is None:
                reference_makespan = reference_heft_makespan(
                    workflow,
                    self.servers,
                    self.link_bandwidth_mbps,
                    rho=float(self.sim_config.get("deadline_heft_rho", 1.0)),
                    cache=self.reference_makespan_cache,
                )
            deadline_multiplier = self._workflow_field(workflow, "deadline_multiplier")
            deadline_type = self._workflow_field(workflow, "deadline_type") or self._deadline_type()
            deadline_info = {
                "deadline_type": deadline_type,
                "reference_makespan": float(reference_makespan),
                "deadline_multiplier": None if deadline_multiplier is None else float(deadline_multiplier),
                "relative_deadline": float(relative_deadline),
                "absolute_deadline": absolute_deadline,
            }
        else:
            deadline_info = cai_2025_deadline_fields(
                workflow,
                float(workflow_state["arrival_time"]),
                self.servers,
                deadline_type=self._deadline_type(),
                rng=self.rng,
                link_bandwidth_mbps=self.link_bandwidth_mbps,
                rho=float(self.sim_config.get("deadline_heft_rho", 1.0)),
                cache=self.reference_makespan_cache,
            )

        self._set_deadline_fields(workflow, deadline_info)
        self._set_workflow_state_deadline_fields(workflow_state, deadline_info)
        self._trace_deadline_generation(workflow_state, deadline_info)

    def _initialize_sub_deadline_fields(self, workflow_state):
        config = dict(self.sim_config.get("sub_deadline", {}))
        if not bool(config.get("enabled", False)):
            workflow_state["sub_deadline_enabled"] = False
            workflow_state["task_sub_deadlines_relative"] = {}
            workflow_state["task_sub_deadlines_absolute"] = {}
            return
        method = str(config.get("method", "heft_aft_scaled_v1")).strip().lower()
        if method != "heft_aft_scaled_v1":
            raise ValueError(
                "simulation.sub_deadline.method must be 'heft_aft_scaled_v1'"
            )
        stats, details = distribute_existing_workflow_deadline(
            workflow_state["workflow"],
            self.servers,
            link_bandwidth_mbps=self.link_bandwidth_mbps,
            rho=float(self.sim_config.get("deadline_heft_rho", 1.0)),
            return_details=True,
            cache=self.reference_makespan_cache,
        )
        if details is None:
            raise RuntimeError("sub-Deadline details were unexpectedly omitted")
        workflow_state["sub_deadline_enabled"] = True
        workflow_state["sub_deadline_method"] = stats.method
        workflow_state["sub_deadline_stats"] = stats
        workflow_state["task_sub_deadlines_relative"] = dict(
            details.relative_by_task
        )
        workflow_state["task_sub_deadlines_absolute"] = dict(
            details.absolute_by_task
        )

    def task_sub_deadline(self, workflow_state, task_id):
        task_id = int(task_id)
        relative = workflow_state.get("task_sub_deadlines_relative", {}).get(
            task_id
        )
        absolute = workflow_state.get("task_sub_deadlines_absolute", {}).get(
            task_id
        )
        if relative is None or absolute is None:
            raise KeyError(
                "sub-Deadline is unavailable for "
                f"workflow={workflow_state['workflow_id']}, task={task_id}"
            )
        return float(relative), float(absolute)

    def _deadline_type(self):
        return validate_deadline_type(self.sim_config.get("deadline_type", "medium"))

    def _workflow_field(self, workflow, field):
        if isinstance(workflow, Workflow):
            return getattr(workflow, field, None)
        return workflow.get(field)

    def _set_deadline_fields(self, workflow, deadline_info):
        absolute_deadline = float(deadline_info["absolute_deadline"])
        reference_makespan = float(deadline_info["reference_makespan"])
        relative_deadline = float(deadline_info["relative_deadline"])
        deadline_multiplier = deadline_info.get("deadline_multiplier")
        if isinstance(workflow, Workflow):
            workflow.reference_makespan = reference_makespan
            workflow.relative_deadline = relative_deadline
            workflow.absolute_deadline = absolute_deadline
            workflow.deadline_type = str(deadline_info["deadline_type"])
            if deadline_multiplier is not None:
                workflow.deadline_multiplier = float(deadline_multiplier)
            return
        workflow["reference_makespan"] = reference_makespan
        workflow["relative_deadline"] = relative_deadline
        workflow["absolute_deadline"] = absolute_deadline
        workflow["deadline_type"] = str(deadline_info["deadline_type"])
        if deadline_multiplier is not None:
            workflow["deadline_multiplier"] = float(deadline_multiplier)

    def _set_workflow_state_deadline_fields(self, workflow_state, deadline_info):
        deadline_multiplier = deadline_info.get("deadline_multiplier")
        reference_makespan = float(deadline_info["reference_makespan"])
        relative_deadline = float(deadline_info["relative_deadline"])
        absolute_deadline = float(deadline_info["absolute_deadline"])
        if reference_makespan <= 0:
            raise ValueError(f"reference_makespan must be positive, got {reference_makespan}")
        if relative_deadline <= 0:
            raise ValueError(f"relative_deadline must be positive, got {relative_deadline}")
        if absolute_deadline < float(workflow_state["arrival_time"]):
            raise ValueError("absolute_deadline must be >= arrival_time")
        if deadline_multiplier is not None and float(deadline_multiplier) not in {2.0, 3.0}:
            raise ValueError(f"deadline_multiplier must be 2.0 or 3.0, got {deadline_multiplier}")
        workflow_state["reference_makespan"] = float(deadline_info["reference_makespan"])
        workflow_state["deadline_multiplier"] = (
            None if deadline_multiplier is None else float(deadline_multiplier)
        )
        workflow_state["relative_deadline"] = float(deadline_info["relative_deadline"])
        workflow_state["absolute_deadline"] = float(deadline_info["absolute_deadline"])
        workflow_state["deadline_type"] = str(deadline_info["deadline_type"])

    def _trace_deadline_generation(self, workflow_state, deadline_info):
        if not bool(self.sim_config.get("log_deadline_generation", self.sim_config.get("trace_deadline_generation", False))):
            return
        workflow = workflow_state["workflow"]
        workflow_type = (
            getattr(workflow, "workflow_type", None)
            if isinstance(workflow, Workflow)
            else workflow.get("workflow_type", workflow.get("type", "unknown"))
        )
        multiplier = deadline_info.get("deadline_multiplier")
        multiplier_text = "n/a" if multiplier is None else f"{float(multiplier):.1f}"
        print(
            "DEADLINE "
            f"workflow_id={workflow_state['workflow_id']} "
            f"workflow_type={workflow_type or 'unknown'} "
            f"arrival_time={float(workflow_state['arrival_time']):.6f} "
            f"reference_makespan={float(deadline_info['reference_makespan']):.6f} "
            f"deadline_type={deadline_info['deadline_type']} "
            f"deadline_multiplier={multiplier_text} "
            f"relative_deadline={float(deadline_info['relative_deadline']):.6f} "
            f"absolute_deadline={float(deadline_info['absolute_deadline']):.6f}"
        )

    def workflow_absolute_deadline(self, workflow_state):
        if workflow_state.get("absolute_deadline") is not None:
            return float(workflow_state["absolute_deadline"])
        workflow = workflow_state["workflow"]
        if isinstance(workflow, Workflow) and workflow.absolute_deadline is not None:
            return float(workflow.absolute_deadline)
        if not isinstance(workflow, Workflow) and workflow.get("absolute_deadline") is not None:
            return float(workflow["absolute_deadline"])
        self._initialize_deadline_fields(workflow_state)
        return float(workflow_state["absolute_deadline"])

    def workflow_relative_deadline(self, workflow_state):
        if workflow_state.get("relative_deadline") is not None:
            return float(workflow_state["relative_deadline"])
        return self.workflow_absolute_deadline(workflow_state) - float(workflow_state["arrival_time"])

    def _server_bandwidth(self, server):
        bandwidth = float(server.get("bandwidth_mbps", 0.0) or 0.0)
        if bandwidth > 0.0:
            return bandwidth
        return self._link_bandwidth("iot", server["type"])

    def _server_execution_available_time(self, server):
        return float(server.get("execution_available_time", server.get("available_time", 0.0)))

    def _set_server_execution_available_time(self, server, value):
        server["execution_available_time"] = float(value)
        # Backward-compatible alias for older schedulers and state encoders.
        server["available_time"] = server["execution_available_time"]

    def _server_backup_available_time(self, server):
        return float(server.get("backup_available_time", 0.0))

    def _set_server_backup_available_time(self, server, value):
        server["backup_available_time"] = float(value)

    def _server_queue_available_time(self, server, queue_type="execution"):
        if queue_type == "backup":
            return max(
                self._server_backup_available_time(server),
                self._server_execution_available_time(server),
            )
        return self._server_execution_available_time(server)

    def _server_queue_load(self, server, queue_type="execution"):
        return max(self._server_queue_available_time(server, queue_type) - self.current_time, 0.0)

    def _server_queue_count(self, server, queue_type="execution"):
        server_id = int(server["id"])
        rows = self.server_real_timeline.get(server_id, [])
        start_index = self._advance_server_busy_cursor(server_id)
        return sum(
            1
            for row in rows[start_index:]
            if row.get("queue_type", "execution") == queue_type
            and float(row["start"]) > self.current_time
        )

    def _server_busy_time(self, server):
        server_id = int(server["id"])
        rows = self.server_real_timeline.get(server_id, [])
        start_index = self._advance_server_busy_cursor(server_id)
        busy_time = self._server_busy_completed.get(server_id, 0.0)
        if start_index < len(rows):
            row = rows[start_index]
            start_time = float(row["start"])
            finish_time = float(row["finish"])
            if start_time < self.current_time < finish_time:
                busy_time += self.current_time - start_time
        return busy_time

    def _advance_server_busy_cursor(self, server_id):
        # current_time is monotonic inside one episode. Keep accumulated busy
        # time for completed intervals, and leave only active/future intervals
        # for get_state() to inspect.
        rows = self.server_real_timeline.get(int(server_id), [])
        cursor = self._server_busy_cursor.get(int(server_id), 0)
        completed = self._server_busy_completed.get(int(server_id), 0.0)
        while cursor < len(rows) and float(rows[cursor]["finish"]) <= self.current_time:
            completed += float(rows[cursor]["finish"]) - float(rows[cursor]["start"])
            cursor += 1
        self._server_busy_cursor[int(server_id)] = cursor
        self._server_busy_completed[int(server_id)] = completed
        return cursor

    def _server_utilization(self, server):
        observed_time = max(float(self.current_time), 1e-9)
        return min(1.0, max(0.0, self._server_busy_time(server) / observed_time))

    def _server_history_metrics(self, server_id):
        return dict(
            self.server_history_stats.get(
                int(server_id),
                {
                    "wait_time_ema": 0.0,
                    "queued_processing_time_ema": 0.0,
                    "queue_count_ema": 0.0,
                    "recent_primary_assignments": 0.0,
                    "recent_backup_assignments": 0.0,
                },
            )
        )

    def _update_server_history_stats(self, primary_plan, backup_plan):
        alpha = self.server_history_ema_alpha
        primary_server_id = int(primary_plan["server_id"])
        backup_server_id = None if backup_plan is None else int(backup_plan["server_id"])
        for server in self.servers:
            server_id = int(server["id"])
            stats = self.server_history_stats.setdefault(
                server_id,
                {
                    "wait_time_ema": 0.0,
                    "queued_processing_time_ema": 0.0,
                    "queue_count_ema": 0.0,
                    "recent_primary_assignments": 0.0,
                    "recent_backup_assignments": 0.0,
                },
            )
            wait_time = self._server_queue_load(server, queue_type="execution")
            queue_count = self._server_queue_count(server, queue_type="execution")
            stats["wait_time_ema"] = alpha * float(stats["wait_time_ema"]) + (1.0 - alpha) * float(wait_time)
            stats["queued_processing_time_ema"] = (
                alpha * float(stats["queued_processing_time_ema"])
                + (1.0 - alpha) * float(wait_time)
            )
            stats["queue_count_ema"] = alpha * float(stats["queue_count_ema"]) + (1.0 - alpha) * float(queue_count)
            stats["recent_primary_assignments"] = alpha * float(stats["recent_primary_assignments"]) + (
                1.0 if server_id == primary_server_id else 0.0
            )
            stats["recent_backup_assignments"] = alpha * float(stats["recent_backup_assignments"]) + (
                1.0 if backup_server_id is not None and server_id == backup_server_id else 0.0
            )

    def update_server_timeline(self, server, replica_plan, role):
        server_id = int(server["id"])
        self.server_real_timeline.setdefault(server_id, []).append(
            {
                "start": float(replica_plan["ast"]),
                "finish": float(replica_plan["aft"]),
                "task_id": int(replica_plan["task_id"]),
                "workflow_id": replica_plan["workflow_id"],
                "role": role,
                "queue_type": "backup" if role == "backup" else "execution",
            }
        )
        self.validate_no_overlap(server_id)

    def validate_no_overlap(self, server_id=None):
        timelines = (
            {int(server_id): self.server_real_timeline.get(int(server_id), [])}
            if server_id is not None
            else self.server_real_timeline
        )
        eps = 1e-9
        for sid, rows in timelines.items():
            ordered = sorted(rows, key=lambda item: (float(item["start"]), float(item["finish"])))
            for previous, current in zip(ordered, ordered[1:]):
                if float(current["start"]) + eps < float(previous["finish"]):
                    raise AssertionError(
                        "Overlapping real server timeline intervals: "
                        f"server={sid}, "
                        f"prev=({previous['workflow_id']}, t{previous['task_id']}, "
                        f"{previous['role']}, {previous['start']}->{previous['finish']}), "
                        f"curr=({current['workflow_id']}, t{current['task_id']}, "
                        f"{current['role']}, {current['start']}->{current['finish']})"
                    )
        return True

    def compute_task_ready_time(self, task, workflow_state):
        task_id = int(task["id"] if isinstance(task, dict) else task.id)
        parents = self._parent_task_ids(workflow_state["workflow"], task_id)
        if not parents:
            return float(workflow_state["arrival_time"])
        ready_times = []
        for parent_id in parents:
            parent_status = workflow_state["task_status"].get(parent_id)
            if parent_status != "finished":
                raise ValueError(
                    f"Dependency not finished: workflow={workflow_state['workflow_id']}, "
                    f"parent={parent_id}, status={parent_status}"
                )
            ready_times.append(float(workflow_state["task_finish_times"][parent_id]))
        return max(ready_times)

    def compute_data_ready_time(self, workflow_state, task_id, target_server_id):
        ready_info = self.compute_replica_ready_times(
            workflow_state,
            task_id,
            target_server_id,
            replica_role="primary",
        )
        return (
            ready_info["data_ready_time"],
            ready_info["transfer_time"],
            ready_info["upload_time"],
        )

    def compute_replica_ready_times(
        self,
        workflow_state,
        task_id,
        target_server_id,
        replica_role="primary",
    ):
        workflow = workflow_state["workflow"]
        parents = self._parent_task_ids(workflow, task_id)
        target_server = self._get_server(int(target_server_id))
        arrival_time = float(workflow_state["arrival_time"])
        fallback_used = False

        if isinstance(workflow, Workflow):
            edge_data_mb = workflow.edge_data_mb
        else:
            edge_data_mb = workflow["edge_data_mb"]

        if not parents:
            return {
                "dependency_ready_time": arrival_time,
                "data_ready_time": arrival_time,
                "transfer_time": 0.0,
                "upload_time": 0.0,
                "backup_path_uses_primary_fallback": False,
            }

        dependency_times = []
        data_ready_times = []
        max_transfer_time = 0.0
        for parent_id in parents:
            parent_status = workflow_state["task_status"].get(parent_id)
            if parent_status != "finished":
                raise ValueError(
                    f"Dependency not finished: workflow={workflow_state['workflow_id']}, "
                    f"parent={parent_id}, status={parent_status}"
                )

            parent_primary_plan = workflow_state["task_primary_plan"].get(parent_id)
            if parent_primary_plan is None:
                raise ValueError(
                    f"Missing parent primary plan: workflow={workflow_state['workflow_id']}, "
                    f"parent={parent_id}"
                )

            if replica_role == "backup":
                parent_backup_plan = workflow_state["task_backup_plan"].get(parent_id)
                if parent_backup_plan is not None:
                    source_aft = float(parent_backup_plan["aft"])
                    source_server_id = int(parent_backup_plan["server_id"])
                else:
                    source_aft = float(parent_primary_plan["aft"])
                    source_server_id = int(parent_primary_plan["server_id"])
                    fallback_used = True
            else:
                source_aft = float(parent_primary_plan["aft"])
                source_server_id = int(parent_primary_plan["server_id"])

            dependency_times.append(source_aft)
            if source_server_id == int(target_server_id):
                data_ready_times.append(source_aft)
                continue

            source_server = self._get_server(source_server_id)
            data_size_mb = edge_data_mb[(parent_id, task_id)]
            transfer_time = compute_transfer_time(
                data_size_mb,
                self._link_bandwidth(source_server["type"], target_server["type"]),
            )
            max_transfer_time = max(max_transfer_time, transfer_time)
            data_ready_times.append(source_aft + transfer_time)

        return {
            "dependency_ready_time": max(dependency_times),
            "data_ready_time": max(data_ready_times),
            "transfer_time": max_transfer_time,
            "upload_time": 0.0,
            "backup_path_uses_primary_fallback": fallback_used,
        }

    def _parent_task_ids(self, workflow, task_id):
        if isinstance(workflow, Workflow):
            return workflow.get_parents(task_id)
        return workflow["parents"].get(task_id, [])

    def compute_absolute_deadline(self, task, workflow_state):
        _ = task
        return self.workflow_absolute_deadline(workflow_state)

    def is_task_deadline_satisfied(self, task, workflow_state, logical_finish_time):
        return float(logical_finish_time) <= self.compute_absolute_deadline(task, workflow_state)

    def compute_task_deadline_violation(self, task, workflow_state, logical_finish_time):
        # Compatibility wrapper: deadline is workflow-level now. During a step,
        # this is only an intermediate estimate; final metrics judge the whole
        # workflow response time against absolute_deadline.
        _ = task
        return max(
            0.0,
            float(logical_finish_time) - self.compute_absolute_deadline(task, workflow_state),
        )

    def _assert_replica_time_plan(self, plan):
        eps = 1e-9
        role = plan.get("replica_role", plan.get("queue_type", "execution"))
        ast = float(plan["ast"])
        aft = float(plan["aft"])
        dependency_ready_time = float(plan.get("dependency_ready_time", plan["task_ready_time"]))
        data_ready_time = float(plan["data_ready_time"])
        task_communication_time = float(plan.get("task_communication_time", 0.0))
        task_waiting_time = float(plan.get("task_waiting_time", 0.0))
        exec_time = float(plan["exec_time"])
        task_total_time = float(plan.get("task_total_time", 0.0))
        checks = (
            ("current_time", float(plan["current_time"])),
            ("workflow_arrival_time", float(plan["workflow_arrival_time"])),
            ("data_ready_time", data_ready_time),
            ("available_time", float(plan["available_time"])),
        )
        for name, lower_bound in checks:
            if ast + eps < lower_bound:
                raise AssertionError(
                    f"{role} AST time violation for workflow={plan['workflow_id']} "
                    f"task={plan['task_id']} server={plan['server_id']}: "
                    f"ast={ast}, {name}={lower_bound}"
                )
        if role == "backup":
            primary_ast = float(plan.get("primary_ast", ast))
            if ast + eps < primary_ast:
                raise AssertionError(
                    f"backup AST before primary AST for workflow={plan['workflow_id']} "
                    f"task={plan['task_id']} server={plan['server_id']}: "
                    f"backup_ast={ast}, primary_ast={primary_ast}"
                )
        if data_ready_time + eps < dependency_ready_time:
            raise AssertionError(
                f"{role} data-ready time violation for workflow={plan['workflow_id']} "
                f"task={plan['task_id']} server={plan['server_id']}: "
                f"data_ready_time={data_ready_time}, dependency_ready_time={dependency_ready_time}"
            )
        expected_comm = max(0.0, data_ready_time - dependency_ready_time)
        if abs(task_communication_time - expected_comm) > eps:
            raise AssertionError(
                f"{role} communication time mismatch for workflow={plan['workflow_id']} "
                f"task={plan['task_id']} server={plan['server_id']}: "
                f"task_communication_time={task_communication_time}, expected={expected_comm}"
            )
        expected_wait = max(0.0, ast - data_ready_time)
        if abs(task_waiting_time - expected_wait) > eps:
            raise AssertionError(
                f"{role} task waiting time mismatch for workflow={plan['workflow_id']} "
                f"task={plan['task_id']} server={plan['server_id']}: "
                f"task_waiting_time={task_waiting_time}, expected={expected_wait}"
            )
        if abs((ast + exec_time) - aft) > eps:
            raise AssertionError(
                f"{role} AFT mismatch for workflow={plan['workflow_id']} "
                f"task={plan['task_id']} server={plan['server_id']}: "
                f"ast={ast}, exec_time={exec_time}, aft={aft}"
            )
        expected_total = task_waiting_time + task_communication_time + exec_time
        if abs(task_total_time - expected_total) > eps:
            raise AssertionError(
                f"{role} task total time mismatch for workflow={plan['workflow_id']} "
                f"task={plan['task_id']} server={plan['server_id']}: "
                f"task_total_time={task_total_time}, expected={expected_total}"
            )
        if abs(task_total_time - (aft - dependency_ready_time)) > eps:
            raise AssertionError(
                f"{role} ready-to-finish time mismatch for workflow={plan['workflow_id']} "
                f"task={plan['task_id']} server={plan['server_id']}: "
                f"task_total_time={task_total_time}, aft-dependency_ready={aft - dependency_ready_time}"
            )

    def _workflow_reliability_risk(self, workflow_state, task_id, candidate_task_reliability):
        reliability_req = self._workflow_reliability_requirement(workflow_state)
        if reliability_req <= 0.0:
            return False
        log_estimate = self._estimated_workflow_log_reliability(
            workflow_state,
            task_id,
            candidate_task_reliability,
        )
        return log_estimate < math.log(max(reliability_req, RELIABILITY_EPS))

    def _workflow_reliability_requirement(self, workflow_state):
        workflow = workflow_state["workflow"]
        static_meta = workflow_state.get("_static_meta")
        if static_meta is None:
            static_meta = self._compute_static_workflow_meta(workflow_state, workflow)
            workflow_state["_static_meta"] = static_meta
        return float(static_meta.get("reliability", self.sim_config.get("default_reliability", 0.0)))

    def _estimated_workflow_log_reliability(
        self,
        workflow_state,
        task_id,
        candidate_task_reliability,
    ):
        log_value = 0.0
        for scheduled_task_id, reliability in workflow_state.get("task_reliability", {}).items():
            if int(scheduled_task_id) == int(task_id):
                continue
            log_value += math.log(max(float(reliability), RELIABILITY_EPS))
        log_value += math.log(max(float(candidate_task_reliability), RELIABILITY_EPS))

        remaining_estimate = float(
            self.sim_config.get("remaining_task_reliability_estimate", 0.999)
        )
        for remaining_task_id in self._workflow_task_ids(workflow_state["workflow"]):
            if int(remaining_task_id) == int(task_id):
                continue
            if remaining_task_id in workflow_state.get("task_reliability", {}):
                continue
            log_value += math.log(max(remaining_estimate, RELIABILITY_EPS))
        return log_value

    def _workflow_task_ids(self, workflow):
        if isinstance(workflow, Workflow):
            return list(workflow.task_ids())
        return [int(task["id"]) for task in workflow["tasks"]]

    def _backup_rho_candidates(self, server, primary_rho):
        configured = self.sim_config.get("backup_rho_levels")
        if configured is None:
            configured = self.resources.get("rho_levels")
        if configured is None:
            configured = (primary_rho, 1.0)
        values = []
        for value in configured:
            rho = self._normalize_rho(value, server)
            if all(abs(rho - existing) > 1e-9 for existing in values):
                values.append(rho)
        return values or [self._normalize_rho(primary_rho, server)]

    def _combined_task_reliability(self, primary_reliability, backup_reliability):
        primary = min(1.0, max(0.0, float(primary_reliability)))
        backup = min(1.0, max(0.0, float(backup_reliability)))
        return min(1.0, max(0.0, 1.0 - (1.0 - primary) * (1.0 - backup)))

    def _replica_event(self, replica_plan, final_plan, replica_role):
        return {
            "workflow_id": replica_plan["workflow_id"],
            "task_id": replica_plan["task_id"],
            "server_id": replica_plan["server_id"],
            "rho": replica_plan["rho"],
            "start_time": replica_plan["ast"],
            "finish_time": replica_plan["aft"],
            "current_time": replica_plan["current_time"],
            "workflow_arrival_time": replica_plan["workflow_arrival_time"],
            "dependency_ready_time": replica_plan["dependency_ready_time"],
            "task_ready_time": replica_plan["task_ready_time"],
            "ready_queue_waiting_time": replica_plan["ready_queue_waiting_time"],
            "data_ready_time": replica_plan["data_ready_time"],
            "base_ready_time": replica_plan["base_ready_time"],
            "task_communication_time": replica_plan["task_communication_time"],
            "task_waiting_time": replica_plan["task_waiting_time"],
            "task_total_time": replica_plan["task_total_time"],
            "server_execution_available_time_before": replica_plan[
                "server_execution_available_time_before"
            ],
            "server_backup_available_time_before": replica_plan[
                "server_backup_available_time_before"
            ],
            "server_queue_waiting_time": replica_plan["server_queue_waiting_time"],
            "exec_time": replica_plan["exec_time"],
            "upload_time": replica_plan["upload_time"],
            "transfer_time": replica_plan["transfer_time"],
            "energy": replica_plan["energy"],
            "server_type": replica_plan["server_type"],
            "queue_type": replica_plan["queue_type"],
            "replica_role": replica_plan.get("replica_role", replica_role),
            "backup_path_uses_primary_fallback": replica_plan.get(
                "backup_path_uses_primary_fallback",
                False,
            ),
            "backup_enabled": final_plan["backup_enabled"],
            "backup_trigger_reason": final_plan["backup_trigger_reason"],
            "logical_finish_time": final_plan["final_aft"],
            "base_failure_rate": replica_plan["base_failure_rate"],
            "failure_rate": replica_plan["failure_rate"],
            "actual_freq_hz": replica_plan["frequency"],
            "min_freq_hz": replica_plan["min_freq_hz"],
            "max_freq_hz": replica_plan["max_freq_hz"],
            "failure_rate_exponent": replica_plan["failure_rate_exponent"],
            "task_reliability": replica_plan["reliability"],
        }

    def _final_task_event(self, final_plan):
        primary = final_plan["primary_plan"]
        backup = final_plan["backup_plan"]
        event = self._replica_event(primary, final_plan, "primary")
        event.update(
            {
                "finish_time": final_plan["final_aft"],
                "energy": final_plan["total_energy"],
                "task_reliability": final_plan["final_reliability"],
                "final_aft": final_plan["final_aft"],
                "logical_AFT": final_plan["logical_aft"],
                "logical_source_server": final_plan["logical_source_server"],
                "normal_aft": final_plan["normal_aft"],
                "failure_aft": final_plan["failure_aft"],
                "normal_source_server": final_plan["normal_source_server"],
                "failure_source_server": final_plan["failure_source_server"],
                "final_reliability": final_plan["final_reliability"],
                "reliability_gain": final_plan["reliability_gain"],
                "primary_server": primary["server_id"],
                "primary_rho": primary["rho"],
                "primary_frequency": primary["frequency"],
                "primary_ast": primary["ast"],
                "primary_aft": primary["aft"],
                "AST_primary": primary["ast"],
                "AFT_primary": primary["aft"],
                "primary_exec_time": primary["exec_time"],
                "execution_time_primary": primary["exec_time"],
                "dependency_ready_time_primary": primary["dependency_ready_time"],
                "data_ready_time_primary": primary["data_ready_time"],
                "base_ready_time_primary": primary["base_ready_time"],
                "task_communication_time_primary": primary["task_communication_time"],
                "task_waiting_time_primary": primary["task_waiting_time"],
                "task_total_time_primary": primary["task_total_time"],
                "server_queue_waiting_time_primary": primary["server_queue_waiting_time"],
                "server_execution_available_time_primary_before": primary[
                    "server_execution_available_time_before"
                ],
                "primary_reliability": primary["reliability"],
                "primary_energy": primary["energy"],
                "backup_server": backup["server_id"] if backup else None,
                "backup_rho": backup["rho"] if backup else None,
                "backup_frequency": backup["frequency"] if backup else None,
                "backup_ast": backup["ast"] if backup else None,
                "backup_aft": backup["aft"] if backup else None,
                "AST_backup": backup["ast"] if backup else None,
                "AFT_backup": backup["aft"] if backup else None,
                "backup_exec_time": backup["exec_time"] if backup else None,
                "execution_time_backup": backup["exec_time"] if backup else None,
                "dependency_ready_time_backup": (
                    backup["dependency_ready_time"] if backup else None
                ),
                "data_ready_time_backup": backup["data_ready_time"] if backup else None,
                "base_ready_time_backup": backup["base_ready_time"] if backup else None,
                "task_communication_time_backup": (
                    backup["task_communication_time"] if backup else None
                ),
                "task_waiting_time_backup": backup["task_waiting_time"] if backup else None,
                "task_total_time_backup": backup["task_total_time"] if backup else None,
                "server_queue_waiting_time_backup": (
                    backup["server_queue_waiting_time"] if backup else None
                ),
                "server_execution_available_time_backup_before": (
                    backup["server_execution_available_time_before"] if backup else None
                ),
                "server_backup_available_time_backup_before": (
                    backup["server_backup_available_time_before"] if backup else None
                ),
                "backup_reliability": backup["reliability"] if backup else None,
                "backup_energy": backup["energy"] if backup else 0.0,
                "backup_path_uses_primary_fallback": (
                    backup.get("backup_path_uses_primary_fallback", False) if backup else False
                ),
                "has_backup": final_plan["backup_enabled"],
                "backup_replica_count": 1 if backup else 0,
                "absolute_deadline": final_plan["absolute_deadline"],
                "relative_deadline": final_plan["relative_deadline"],
                "deadline_satisfied": final_plan["deadline_satisfied"],
                "deadline_violation": final_plan["deadline_violation"],
                "primary_deadline_violation": final_plan["primary_deadline_violation"],
                "backup_deadline_violation": final_plan["backup_deadline_violation"],
                "backup_deadline_satisfied": final_plan["backup_deadline_satisfied"],
            }
        )
        return event

    def _normalize_rho(self, rho, server=None):
        if rho is None:
            rho = self.sim_config.get("default_rho", 1.0)
        rho = float(rho)
        rho_min = float(
            server.get("min_rho", self.sim_config.get("rho_min", 0.1))
            if server is not None
            else self.sim_config.get("rho_min", 0.1)
        )
        return min(1.0, max(rho_min, rho))

    def _get_server(self, server_id):
        for server in self.servers:
            if int(server["id"]) == server_id:
                return server
        raise ValueError(f"Unknown server_id in action: {server_id}")

    def _get_task(self, workflow, task_id):
        if isinstance(workflow, Workflow):
            task = workflow.get_task(task_id)
            return {"id": task.id, "workload_cycles": task.workload_cycles}

        for task in workflow["tasks"]:
            if int(task["id"]) == task_id:
                return {"id": int(task["id"]), "workload_cycles": float(task["workload_cycles"])}
        raise ValueError(f"Unknown task_id in workflow {workflow['workflow_id']}: {task_id}")

    def _dependency_ready_time(self, workflow_state, task_id, target_server_id):
        return self.compute_replica_ready_times(
            workflow_state,
            task_id,
            target_server_id,
            replica_role="primary",
        )

    def _build_workflow_results(self):
        results = []
        for state in self.workflow_states:
            start_times = list(state["task_start_times"].values())
            finish_times = list(state["task_finish_times"].values())
            compute_finish = max(finish_times) if finish_times else None
            # egress_finish_time is set on completion; stays None for unfinished workflows
            finish_time = state["egress_finish_time"]
            first_start = min(start_times) if start_times else None
            response_time = (
                finish_time - state["arrival_time"] if finish_time is not None else None
            )
            makespan = response_time
            absolute_deadline = self.workflow_absolute_deadline(state)
            workflow_relative_deadline = self.workflow_relative_deadline(state)
            deadline_satisfied = (
                finish_time <= absolute_deadline if finish_time is not None else None
            )
            deadline_violation = (
                max(0.0, finish_time - absolute_deadline) if finish_time is not None else None
            )
            compute_span = (
                compute_finish - min(start_times)
                if compute_finish is not None and start_times
                else None
            )
            waiting_time = (
                first_start - state["arrival_time"] if first_start is not None else None
            )
            workflow = state["workflow"]
            reliability_req = (
                workflow.reliability if isinstance(workflow, Workflow) else workflow.get("reliability")
            )
            if reliability_req is None:
                reliability_req = state.get("_static_meta", {}).get("reliability")
            reliability = (
                float(state.get("reliability_product", 1.0))
                if finish_time is not None
                else None
            )
            reliability_met = None
            if reliability is not None and reliability_req is not None:
                reliability_met = reliability >= float(reliability_req)
            task_plans = self._workflow_task_plan_rows(state)
            self._assert_workflow_time_result(
                workflow_id=state["workflow_id"],
                response_time=response_time,
                task_plans=task_plans,
            )
            results.append(
                {
                    "workflow_id": state["workflow_id"],
                    "arrival_time": state["arrival_time"],
                    "finish_time": finish_time,
                    "compute_finish_time": compute_finish,
                    "absolute_deadline": absolute_deadline,
                    "relative_deadline": workflow_relative_deadline,
                    "reference_makespan": state.get("reference_makespan"),
                    "deadline_multiplier": state.get("deadline_multiplier"),
                    "deadline_type": state.get("deadline_type"),
                    "deadline_satisfied": deadline_satisfied,
                    "deadline_violation": deadline_violation,
                    "upload_time": state.get("upload_time"),
                    "download_time": state.get("download_time"),
                    "response_time": response_time,
                    "makespan": makespan,
                    "compute_span": compute_span,
                    "waiting_time": waiting_time,
                    "total_energy": sum(state["task_energy"].values()),
                    "reliability": reliability,
                    "reliability_req": reliability_req,
                    "reliability_met": reliability_met,
                    "task_plans": task_plans,
                }
            )
        return results

    def _assert_workflow_time_result(self, workflow_id, response_time, task_plans):
        if response_time is None:
            return
        eps = 1e-9
        for row in task_plans:
            logical_total = row.get("logical_task_total_time")
            logical_wait = row.get("logical_task_waiting_time")
            logical_comm = row.get("logical_task_communication_time")
            logical_exec = row.get("logical_task_exec_time")
            if logical_total is not None and float(response_time) + eps < float(logical_total):
                raise AssertionError(
                    f"workflow response time is shorter than logical task total time: "
                    f"workflow={workflow_id}, task={row.get('task_id')}, "
                    f"response_time={response_time}, logical_task_total_time={logical_total}"
                )
            if None not in (logical_total, logical_wait, logical_comm, logical_exec):
                expected_total = float(logical_wait) + float(logical_comm) + float(logical_exec)
                if abs(float(logical_total) - expected_total) > eps:
                    raise AssertionError(
                        f"logical task time components do not sum to total: "
                        f"workflow={workflow_id}, task={row.get('task_id')}, "
                        f"total={logical_total}, wait={logical_wait}, "
                        f"comm={logical_comm}, exec={logical_exec}, expected={expected_total}"
                    )

    def _workflow_task_plan_rows(self, workflow_state):
        rows = []
        for task_id in sorted(workflow_state.get("task_final_plan", {})):
            final_plan = workflow_state["task_final_plan"][task_id]
            primary = final_plan["primary_plan"]
            backup = final_plan["backup_plan"]
            winning_role = final_plan["winning_replica_role"]
            logical_plan = primary
            logical_task_total_time = primary["task_total_time"]
            rows.append(
                {
                    "task_id": int(task_id),
                    "workflow_id": workflow_state["workflow_id"],
                    "current_time": primary["current_time"],
                    "workflow_arrival_time": primary["workflow_arrival_time"],
                    "dependency_ready_time": primary["dependency_ready_time"],
                    "task_ready_time": primary["task_ready_time"],
                    "ready_queue_waiting_time": primary["ready_queue_waiting_time"],
                    "primary_server": primary["server_id"],
                    "primary_frequency": primary["frequency"],
                    "primary_rho": primary["rho"],
                    "primary_ast": primary["ast"],
                    "primary_aft": primary["aft"],
                    "primary_exec_time": primary["exec_time"],
                    "dependency_ready_time_primary": primary["dependency_ready_time"],
                    "data_ready_time_primary": primary["data_ready_time"],
                    "base_ready_time_primary": primary["base_ready_time"],
                    "task_communication_time_primary": primary["task_communication_time"],
                    "task_waiting_time_primary": primary["task_waiting_time"],
                    "task_total_time_primary": primary["task_total_time"],
                    "server_queue_waiting_time_primary": primary["server_queue_waiting_time"],
                    "server_execution_available_time_primary_before": primary[
                        "server_execution_available_time_before"
                    ],
                    "primary_reliability": primary["reliability"],
                    "primary_energy": primary["energy"],
                    "backup_enabled": final_plan["backup_enabled"],
                    "backup_trigger_reason": final_plan["backup_trigger_reason"],
                    "backup_server": backup["server_id"] if backup else None,
                    "backup_frequency": backup["frequency"] if backup else None,
                    "backup_rho": backup["rho"] if backup else None,
                    "backup_ast": backup["ast"] if backup else None,
                    "backup_aft": backup["aft"] if backup else None,
                    "backup_exec_time": backup["exec_time"] if backup else None,
                    "dependency_ready_time_backup": (
                        backup["dependency_ready_time"] if backup else None
                    ),
                    "data_ready_time_backup": backup["data_ready_time"] if backup else None,
                    "base_ready_time_backup": backup["base_ready_time"] if backup else None,
                    "task_communication_time_backup": (
                        backup["task_communication_time"] if backup else None
                    ),
                    "task_waiting_time_backup": backup["task_waiting_time"] if backup else None,
                    "task_total_time_backup": backup["task_total_time"] if backup else None,
                    "server_queue_waiting_time_backup": (
                        backup["server_queue_waiting_time"] if backup else None
                    ),
                    "server_execution_available_time_backup_before": (
                        backup["server_execution_available_time_before"] if backup else None
                    ),
                    "server_backup_available_time_backup_before": (
                        backup["server_backup_available_time_before"] if backup else None
                    ),
                    "backup_reliability": backup["reliability"] if backup else None,
                    "backup_energy": backup["energy"] if backup else 0.0,
                    "backup_path_uses_primary_fallback": (
                        backup.get("backup_path_uses_primary_fallback", False) if backup else False
                    ),
                    "final_aft": final_plan["final_aft"],
                    "logical_AFT": final_plan["logical_aft"],
                    "normal_aft": final_plan["normal_aft"],
                    "failure_aft": final_plan["failure_aft"],
                    "normal_source_server": final_plan["normal_source_server"],
                    "failure_source_server": final_plan["failure_source_server"],
                    "winning_replica_role": winning_role,
                    "logical_task_waiting_time": logical_plan["task_waiting_time"],
                    "logical_task_communication_time": logical_plan["task_communication_time"],
                    "logical_task_exec_time": logical_plan["exec_time"],
                    "logical_task_total_time": logical_task_total_time,
                    "logical_source_server": final_plan["logical_source_server"],
                    "final_server": final_plan["final_server_id"],
                    "final_reliability": final_plan["final_reliability"],
                    "total_energy": final_plan["total_energy"],
                    "reliability_gain": final_plan["reliability_gain"],
                    "absolute_deadline": final_plan["absolute_deadline"],
                    "relative_deadline": final_plan["relative_deadline"],
                    "deadline_satisfied": final_plan["deadline_satisfied"],
                    "deadline_violation": final_plan["deadline_violation"],
                    "primary_deadline_violation": final_plan["primary_deadline_violation"],
                    "backup_deadline_violation": final_plan["backup_deadline_violation"],
                    "backup_deadline_satisfied": final_plan["backup_deadline_satisfied"],
                }
            )
        return rows

    def _workflow_task_metadata(self, workflow_state, workflow):
        # static per-workflow constraints are constant across the episode -> cache once
        static_meta = workflow_state.get("_static_meta")
        if static_meta is None:
            static_meta = self._compute_static_workflow_meta(workflow_state, workflow)
            workflow_state["_static_meta"] = static_meta

        return {
            "energy_budget": static_meta["energy_budget"],
            "reliability": static_meta["reliability"],
            "workflow_energy_used": float(workflow_state.get("energy_used", 0.0)),
            "workflow_reliability": float(workflow_state.get("reliability_product", 1.0)),
        }

    def _compute_static_workflow_meta(self, workflow_state, workflow):
        arrival_time = float(workflow_state["arrival_time"])
        max_server_freq_hz = max(
            (float(server.get("max_freq_hz", 0.0)) for server in self.servers),
            default=1.0,
        )
        max_server_freq_hz = max(max_server_freq_hz, 1.0)

        if isinstance(workflow, Workflow):
            reliability = workflow.reliability
            energy_budget = workflow.energy_budget
            total_cycles = sum(task.workload_cycles for task in workflow.tasks)
        else:
            reliability = workflow.get("reliability")
            energy_budget = workflow.get("energy_budget")
            total_cycles = sum(float(task["workload_cycles"]) for task in workflow["tasks"])

        if energy_budget is None:
            min_makespan = total_cycles / max_server_freq_hz
            e_min = float(self.sim_config.get("energy_min_factor", 2.0)) * min_makespan
            e_max = float(self.sim_config.get("energy_max_factor", 20.0)) * min_makespan
            eta = float(self.sim_config.get("energy_eta", 0.5))
            energy_budget = e_min + eta * (e_max - e_min)
        if reliability is None:
            reliability = float(self.sim_config.get("default_reliability", 0.9))

        return {
            "energy_budget": float(energy_budget),
            "reliability": float(reliability),
        }

    def _workflow_task_count(self, workflow):
        if isinstance(workflow, Workflow):
            return len(workflow.tasks)
        return len(workflow["tasks"])

    def _trace_enabled(self):
        return self.in_run and bool(self.sim_config.get("trace_scheduling", False))

    def _trace(self, message):
        if not self._trace_enabled():
            return
        trace_limit = self.sim_config.get("trace_limit")
        if trace_limit is not None and self.trace_count >= int(trace_limit):
            return
        print(message)
        self.trace_count += 1

    def _trace_schedule_decision(self, state, action):
        ready_tasks = state["ready_tasks"]
        ready_preview = ", ".join(
            f"{task['workflow_id']}:t{task['task_id']}"
            for task in ready_tasks[:8]
        )
        if len(ready_tasks) > 8:
            ready_preview += f", ... +{len(ready_tasks) - 8}"
        self._trace(
            "DECISION "
            f"time={state['time']:.6f} "
            f"ready={len(ready_tasks)} "
            f"[{ready_preview}] -> "
            f"workflow={action['workflow_id']} "
            f"task={action['task_id']} "
            f"server={action['server_id']} "
            f"rho={float(action['rho']):.6f}"
        )

    def _trace_schedule_result(self, event):
        transfer_suffix = ""
        if float(event.get("upload_time", 0.0)) > 0:
            transfer_suffix += f" upload={event['upload_time']:.6f}"
        if float(event.get("transfer_time", 0.0)) > 0:
            transfer_suffix += f" transfer={event['transfer_time']:.6f}"
        backup_suffix = ""
        if event.get("backup_enabled"):
            backup_suffix = (
                f" backup=server{event.get('backup_server')} "
                f"backup_start={float(event.get('backup_ast')):.6f} "
                f"backup_finish={float(event.get('backup_aft')):.6f} "
                f"final_finish={float(event.get('final_aft')):.6f}"
            )
        self._trace(
            "EXECUTE "
            f"workflow={event['workflow_id']} "
            f"task={event['task_id']} "
            f"server={event['server_id']} "
            f"rho={event['rho']:.6f} "
            f"start={event['start_time']:.6f} "
            f"finish={event['finish_time']:.6f} "
            f"exec={event['exec_time']:.6f}"
            f"{transfer_suffix} "
            f"energy={event['energy']:.6f}"
            f"{backup_suffix}"
        )
