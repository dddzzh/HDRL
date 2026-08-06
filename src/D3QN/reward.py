from __future__ import annotations

import math


EPS = 1e-8


class HierarchicalObjective:
    def __init__(self, config=None):
        config = dict(config or {})
        self.time_reward_mode = str(
            config.get("time_reward_mode", "legacy_slack")
        ).strip().lower()
        if self.time_reward_mode not in {"legacy_slack", "incremental_cost"}:
            raise ValueError(
                "time_reward_mode must be 'legacy_slack' or 'incremental_cost'."
            )
        self.high_makespan_weight = float(config.get("high_makespan_weight", 0.7))
        self.high_energy_weight = float(
            config.get("high_energy_weight", config.get("high_low_reward_weight", 0.3))
        )
        # Compatibility alias for older checkpoints/config diagnostics.
        self.high_low_reward_weight = self.high_energy_weight
        constraint_penalty = dict(config.get("constraint_penalty", {}))
        self.constraint_crossing_weight = float(
            constraint_penalty.get(
                "crossing_weight",
                config.get("high_reliability_fixed_penalty", 5.0),
            )
        )
        self.constraint_severity_weight = float(
            constraint_penalty.get(
                "severity_weight",
                config.get("high_reliability_severity_weight", 2.0),
            )
        )
        # Compatibility aliases used by the unchanged OO baseline.
        self.high_reliability_fixed_penalty = self.constraint_crossing_weight
        self.high_reliability_severity_weight = self.constraint_severity_weight
        if self.high_makespan_weight < 0.0 or self.high_energy_weight < 0.0:
            raise ValueError("high reward weights must be non-negative.")
        if self.constraint_crossing_weight < 0.0 or self.constraint_severity_weight < 0.0:
            raise ValueError("high constraint penalty weights must be non-negative.")
        total_weight = self.high_makespan_weight + self.high_energy_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                "high_makespan_weight + high_energy_weight must equal 1.0."
            )
        self.reset_episode()

    def reset_episode(self):
        self.previous_partial_makespan = {}
        self.previous_partial_log_risk = {}
        self.deadline_cost_sums = {}
        self.high_makespan_component_sums = {}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return None

    def compute_step_feedback(self, env, state, high_decision, low_decision, env_action, env_reward, info) -> dict:
        event = dict((info or {}).get("event", {}))
        workflow_id = str(env_action["workflow_id"])
        task_id = int(env_action["task_id"])
        workflow_state = env.workflow_by_id[workflow_id]
        state_workflow_id = self._workflow_state_id(workflow_state)
        if state_workflow_id is not None and str(state_workflow_id) != workflow_id:
            raise AssertionError("workflow_state does not match env_action workflow_id.")
        deadline = max(float(env.workflow_relative_deadline(workflow_state)), EPS)

        previous_makespan = float(self.previous_partial_makespan.get(workflow_id, 0.0))
        current_makespan = self._current_partial_makespan(workflow_state)
        makespan_increment = max(0.0, current_makespan - previous_makespan)
        normalized_makespan_increment = makespan_increment / deadline
        task_timing = self._task_completion_timing(state, env_action, event)
        previous_slack_ratio = (deadline - previous_makespan) / deadline
        current_slack_ratio = (deadline - current_makespan) / deadline

        total_energy = float(event.get("energy", -float(env_reward) if env_reward is not None else 0.0))
        primary_energy_actual = float(event.get("primary_energy", total_energy))
        backup_energy_actual = float(event.get("backup_energy", 0.0))
        primary_energy_ref = float(env_action.get("primary_energy_ref_rho1", 0.0))
        if primary_energy_ref <= EPS:
            primary_energy_ref = float(env.estimate_primary_energy_ref_rho1(env_action))
        workflow_task_count = max(1.0, float(self._workflow_task_count(workflow_state)))

        # The incremental time cost telescopes to -C_w / D_w over the whole
        # workflow, so it must not be divided by N_w.  Only the per-task energy
        # term is divided by N_w to keep workflow returns comparable by size.
        if self.time_reward_mode == "incremental_cost":
            high_time_reward_raw = -normalized_makespan_increment
        else:
            high_time_reward_raw = current_slack_ratio / workflow_task_count
        high_makespan_reward_component = self.high_makespan_weight * high_time_reward_raw

        low_reward = primary_energy_ref - primary_energy_actual
        low_energy_saving_ratio = low_reward / max(primary_energy_ref, EPS)
        decision_context = (state or {}).get("decision_context", {})
        task_energy_scale = float(
            decision_context.get("task_energy_scale", primary_energy_ref)
            if isinstance(decision_context, dict)
            else primary_energy_ref
        )
        task_energy_scale = max(task_energy_scale, EPS)
        high_energy_reward_raw = (
            task_energy_scale - primary_energy_actual
        ) / task_energy_scale
        high_energy_reward_component = (
            self.high_energy_weight * high_energy_reward_raw / workflow_task_count
        )
        high_reward_base = high_makespan_reward_component + high_energy_reward_component

        previous_log_risk = float(self.previous_partial_log_risk.get(workflow_id, 0.0))
        current_log_risk = self._current_partial_log_risk(workflow_state)
        reliability_cost = max(0.0, current_log_risk - previous_log_risk)
        reliability_limit = self._allowed_log_risk(env, workflow_state)
        reliability_scale = max(reliability_limit, EPS)
        previous_reliability_violation = max(0.0, previous_log_risk - reliability_limit) / reliability_scale
        current_reliability_violation = max(0.0, current_log_risk - reliability_limit) / reliability_scale
        reliability_violation_delta = max(
            0.0, current_reliability_violation - previous_reliability_violation
        )
        reliability_crossed = float(
            current_reliability_violation > EPS and previous_reliability_violation <= EPS
        )
        high_reliability_penalty = (
            self.constraint_crossing_weight * reliability_crossed
            + self.constraint_severity_weight * reliability_violation_delta
        )
        high_reward = high_reward_base - high_reliability_penalty

        self.previous_partial_makespan[workflow_id] = current_makespan
        self.previous_partial_log_risk[workflow_id] = current_log_risk
        self.deadline_cost_sums[workflow_id] = float(self.deadline_cost_sums.get(workflow_id, 0.0)) + makespan_increment
        self.high_makespan_component_sums[workflow_id] = (
            float(self.high_makespan_component_sums.get(workflow_id, 0.0)) + high_makespan_reward_component
        )

        result = {
            "workflow_id": workflow_id,
            "task_id": task_id,
            "high_reward": high_reward,
            "high_reward_base": high_reward_base,
            "high_makespan_reward": high_makespan_reward_component,
            "high_makespan_reward_component": high_makespan_reward_component,
            "high_task_time_reward": high_makespan_reward_component,
            "high_task_time_reward_component": high_makespan_reward_component,
            "task_time_regret": task_timing["task_time_regret"],
            "task_complete_time_ratio": task_timing["task_complete_time_ratio"],
            "normalized_makespan_increment": normalized_makespan_increment,
            "previous_slack_ratio": previous_slack_ratio,
            "current_slack_ratio": current_slack_ratio,
            "high_slack_reward_raw": high_time_reward_raw,
            "high_time_reward_raw": high_time_reward_raw,
            "task_actual_time": task_timing["task_actual_time"],
            "task_fastest_time": task_timing["task_fastest_time"],
            "task_ready_time": task_timing["task_ready_time"],
            "task_finish_time": task_timing["task_finish_time"],
            "high_energy_reward_raw": high_energy_reward_raw,
            "high_energy_reward": high_energy_reward_component,
            "high_energy_reward_component": high_energy_reward_component,
            "task_energy_scale": task_energy_scale,
            # Compatibility field: it now contains the absolute-energy component.
            "high_low_reward_bonus": high_energy_reward_component,
            "low_energy_saving_ratio": low_energy_saving_ratio,
            "low_reward": low_reward,
            "deadline_cost": makespan_increment,
            "deadline_limit": deadline,
            "reliability_cost": reliability_cost,
            "reliability_limit": reliability_limit,
            "reliability_crossed": reliability_crossed,
            "reliability_violation": current_reliability_violation,
            "reliability_violation_delta": reliability_violation_delta,
            "high_reliability_penalty": high_reliability_penalty,
            "high_deadline_penalty": 0.0,
            "current_partial_makespan": current_makespan,
            "current_partial_makespan_seconds": current_makespan,
            "current_partial_log_risk": current_log_risk,
            "primary_energy_ref": primary_energy_ref,
            "workflow_task_count": workflow_task_count,
            "primary_energy_actual": primary_energy_actual,
            "backup_energy_actual": backup_energy_actual,
            "total_energy_actual": total_energy,
        }
        self._assert_feedback_finite(result)
        return result

    def compute_terminal_feedback(self, env, workflow_result) -> dict:
        workflow_id = str(workflow_result["workflow_id"])
        response_time = float(workflow_result["response_time"])
        deadline = max(float(workflow_result.get("relative_deadline", response_time)), EPS)
        previous_makespan = float(self.previous_partial_makespan.get(workflow_id, 0.0))
        terminal_increment = max(0.0, response_time - previous_makespan)
        deadline_cost_delta = terminal_increment
        normalized_terminal_increment = terminal_increment / deadline
        high_makespan_reward_delta = (
            -self.high_makespan_weight * normalized_terminal_increment
            if self.time_reward_mode == "incremental_cost"
            else 0.0
        )
        if self.time_reward_mode == "incremental_cost":
            deadline_crossed = float(response_time > deadline + EPS)
            deadline_violation = max(0.0, response_time - deadline) / deadline
            high_deadline_penalty = (
                self.constraint_crossing_weight * deadline_crossed
                + self.constraint_severity_weight * deadline_violation
            )
        else:
            # The unchanged OO/legacy configurations keep their historical
            # reward exactly; only the new D3QN mode enables this terminal P_D.
            deadline_crossed = 0.0
            deadline_violation = 0.0
            high_deadline_penalty = 0.0
        high_reward_delta = high_makespan_reward_delta - high_deadline_penalty
        self.previous_partial_makespan[workflow_id] = response_time
        self.deadline_cost_sums[workflow_id] = float(self.deadline_cost_sums.get(workflow_id, 0.0)) + deadline_cost_delta
        self.high_makespan_component_sums[workflow_id] = (
            float(self.high_makespan_component_sums.get(workflow_id, 0.0))
            + high_makespan_reward_delta
        )
        expected_deadline_cost = response_time
        actual_deadline_cost = float(self.deadline_cost_sums[workflow_id])
        if abs(actual_deadline_cost - expected_deadline_cost) > 1e-5:
            raise AssertionError(
                "Terminal deadline telescope failed: "
                f"workflow={workflow_id}, "
                f"sum={actual_deadline_cost}, "
                f"expected={expected_deadline_cost}, "
                f"deadline={deadline}"
            )
        result = {
            "workflow_id": workflow_id,
            "terminal_makespan_increment": terminal_increment,
            "normalized_terminal_makespan_increment": normalized_terminal_increment,
            "deadline_cost_delta": deadline_cost_delta,
            "high_reward_delta": high_reward_delta,
            "high_makespan_reward_delta": high_makespan_reward_delta,
            "terminal_high_makespan_reward": high_makespan_reward_delta,
            "high_deadline_crossed": deadline_crossed,
            "high_deadline_violation": deadline_violation,
            "high_deadline_penalty": high_deadline_penalty,
            "final_partial_makespan": response_time,
        }
        self._assert_feedback_finite(result)
        return result

    def _current_partial_makespan(self, workflow_state):
        finish_times = [float(value) for value in workflow_state.get("task_finish_times", {}).values() if value is not None]
        if not finish_times:
            return 0.0
        return max(0.0, max(finish_times) - float(workflow_state["arrival_time"]))

    def _task_completion_timing(self, state, env_action, event):
        context = (state or {}).get("decision_context", {})
        metrics_by_id = context.get("server_safety_metrics", {}) if isinstance(context, dict) else {}
        servers = list((state or {}).get("servers", []))

        selected_server_id = str(env_action.get("server_id"))
        selected_metrics = metrics_by_id.get(selected_server_id, {})
        task_ready_time = float(
            event.get(
                "task_ready_time",
                selected_metrics.get(
                    "task_ready_time",
                    selected_metrics.get("current_time", event.get("current_time", 0.0)),
                ),
            )
        )
        task_finish_time = float(
            event.get(
                "final_aft",
                event.get("logical_finish_time", event.get("finish_time", task_ready_time)),
            )
        )
        actual_time = max(0.0, task_finish_time - task_ready_time)
        fastest_time = self._fastest_complete_task_time(
            metrics_by_id,
            servers,
            task_ready_time,
        )
        if fastest_time <= EPS:
            fastest_time = max(actual_time, EPS)
        regret = max(0.0, actual_time - fastest_time) / max(fastest_time, EPS)
        complete_time_ratio = actual_time / max(fastest_time, EPS)
        return {
            "task_time_regret": regret,
            "task_complete_time_ratio": complete_time_ratio,
            "task_actual_time": actual_time,
            "task_fastest_time": fastest_time,
            "task_ready_time": task_ready_time,
            "task_finish_time": task_finish_time,
        }

    def _fastest_complete_task_time(self, metrics_by_id, servers, task_ready_time):
        candidates = []
        if servers:
            for server in servers:
                server_id = str(server.get("id"))
                metrics = metrics_by_id.get(server_id)
                if not metrics:
                    continue
                candidates.append(self._candidate_complete_time(metrics, task_ready_time))
        if not candidates:
            for metrics in metrics_by_id.values():
                candidates.append(self._candidate_complete_time(metrics, task_ready_time))
        finite = [value for value in candidates if value is not None and math.isfinite(float(value)) and float(value) > EPS]
        return min(finite) if finite else EPS

    def _candidate_complete_time(self, metrics, task_ready_time):
        if not isinstance(metrics, dict):
            return None
        if "estimated_finish_time" in metrics:
            finish_time = float(metrics["estimated_finish_time"])
        elif "estimated_ast" in metrics and "execution_time_rho_1" in metrics:
            finish_time = float(metrics["estimated_ast"]) + float(metrics["execution_time_rho_1"])
        else:
            return None
        return max(0.0, finish_time - float(task_ready_time))

    def _current_partial_log_risk(self, workflow_state):
        reliability = min(1.0, max(EPS, float(workflow_state.get("reliability_product", 1.0))))
        return -math.log(reliability)

    def _allowed_log_risk(self, env, workflow_state):
        reliability_req = float(env._workflow_reliability_requirement(workflow_state))
        reliability_req = min(1.0, max(EPS, reliability_req))
        return -math.log(reliability_req)

    def _workflow_task_count(self, workflow_state):
        workflow = workflow_state["workflow"]
        if hasattr(workflow, "tasks"):
            return len(workflow.tasks)
        return len(workflow.get("tasks", []))

    def _workflow_state_id(self, workflow_state):
        if workflow_state.get("workflow_id") is not None:
            return workflow_state["workflow_id"]
        workflow = workflow_state.get("workflow")
        if hasattr(workflow, "workflow_id"):
            return workflow.workflow_id
        if isinstance(workflow, dict):
            return workflow.get("workflow_id")
        return None

    def _assert_feedback_finite(self, values):
        for key, value in values.items():
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise FloatingPointError(f"Objective feedback {key} is not finite: {value}")
