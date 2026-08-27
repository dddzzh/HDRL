from __future__ import annotations

import copy
from collections import deque
from pathlib import Path

import numpy as np

from src.D3QN.hierarchical_d3qn_agent import HierarchicalD3QNAgent
from src.D3QN.utils import as_torch_rng_state
from src.coordination.c3c import (
    CounterfactualCrossLevelCritic,
    compute_c3c_step_components,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class C3CHierarchicalAgent:
    """Independent D3QN-SAC-C3C learner that leaves the legacy HCC path intact."""

    def __init__(self, config=None, seed=42):
        config = copy.deepcopy(config or {})
        self.config = copy.deepcopy(config)
        self.seed = int(seed)
        base_config = copy.deepcopy(config)
        base_config["coordination_critic"] = {"enabled": False}
        self.base_agent = HierarchicalD3QNAgent(base_config, seed=self.seed)
        if self.base_agent.low_algorithm != "sac" or self.base_agent.low_agent is None:
            raise ValueError("C3C requires a trainable continuous low SAC agent.")
        if self.base_agent.shared_deadline_lagrangian_enabled:
            raise ValueError(
                "C3C v1 uses separate high/low Deadline multipliers; shared lambda is not supported."
            )
        self.high_agent = self.base_agent.high_agent
        self.low_agent = self.base_agent.low_agent
        self.low_algorithm = "sac"
        self.use_low_agent = True
        self.shared_deadline_lagrangian_enabled = False
        self.shared_deadline_lagrangian = None
        # Compatibility with the legacy collector.  The old absolute HCC is
        # deliberately disabled; C3C is exposed through its own field.
        self.coordination_critic = None
        shared = dict(config.get("shared", {}))
        c3c_config = {**shared, **dict(config.get("c3c", {}))}
        if not bool(c3c_config.get("enabled", True)):
            raise ValueError("The independent C3C trainer requires c3c.enabled=true.")
        self.reward_profile = str(
            c3c_config.get("profile_name", "c3c_rho1_residual_v1")
        )
        self.c3c = CounterfactualCrossLevelCritic(
            high_state_dim=self.high_agent.encoder.high_state_dim,
            action_dim=self.high_agent.action_dim,
            config=c3c_config,
            seed=self.seed + 3001,
        )
        self.c3c.attach_low_actor(self.low_agent.actor)
        self.q_scale_alignment_enabled = bool(
            c3c_config.get("q_scale_alignment_enabled", False)
        )
        self.correction_q_std_clip = float(
            c3c_config.get("correction_q_std_clip", 0.5)
        )
        self.correction_q_std_floor = float(
            c3c_config.get("correction_q_std_floor", 1e-3)
        )
        self.correction_advantage_std_floor = float(
            c3c_config.get("correction_advantage_std_floor", 1e-4)
        )
        self.relative_to_d3qn_action = bool(
            c3c_config.get("relative_to_d3qn_action", False)
        )
        self.extra_deadline_risk_weight = float(
            c3c_config.get("extra_deadline_risk_weight", 0.0)
        )
        self.low_guidance_lambda_source = str(
            c3c_config.get("low_guidance_lambda_source", "high")
        ).strip().lower()
        self.constraint_gate_enabled = bool(
            c3c_config.get("constraint_gate_enabled", False)
        )
        self.constraint_gate_activate_success = float(
            c3c_config.get("constraint_gate_activate_success", 0.97)
        )
        self.constraint_gate_deactivate_success = float(
            c3c_config.get("constraint_gate_deactivate_success", 0.95)
        )
        self.constraint_gate_window_workflows = int(
            c3c_config.get("constraint_gate_window_workflows", 500)
        )
        self.constraint_gate_min_workflows = int(
            c3c_config.get(
                "constraint_gate_min_workflows",
                self.constraint_gate_window_workflows,
            )
        )
        self.constraint_gate_anneal_workflows = int(
            c3c_config.get("constraint_gate_anneal_workflows", 1000)
        )
        if self.correction_q_std_clip <= 0.0:
            raise ValueError("C3C correction_q_std_clip must be positive.")
        if self.correction_q_std_floor <= 0.0:
            raise ValueError("C3C correction_q_std_floor must be positive.")
        if self.correction_advantage_std_floor <= 0.0:
            raise ValueError(
                "C3C correction_advantage_std_floor must be positive."
            )
        if self.extra_deadline_risk_weight < 0.0:
            raise ValueError(
                "C3C extra_deadline_risk_weight must be non-negative."
            )
        if self.low_guidance_lambda_source not in {"high", "low"}:
            raise ValueError(
                "C3C low_guidance_lambda_source must be 'high' or 'low'."
            )
        if not (
            0.0
            <= self.constraint_gate_deactivate_success
            <= self.constraint_gate_activate_success
            <= 1.0
        ):
            raise ValueError(
                "C3C constraint gate thresholds must satisfy "
                "0 <= deactivate <= activate <= 1."
            )
        if self.constraint_gate_window_workflows <= 0:
            raise ValueError(
                "C3C constraint_gate_window_workflows must be positive."
            )
        if not (
            1
            <= self.constraint_gate_min_workflows
            <= self.constraint_gate_window_workflows
        ):
            raise ValueError(
                "C3C constraint_gate_min_workflows must be in "
                "[1, constraint_gate_window_workflows]."
            )
        if self.constraint_gate_anneal_workflows < 0:
            raise ValueError(
                "C3C constraint_gate_anneal_workflows must be non-negative."
            )
        self.constraint_gate_window = deque(
            maxlen=self.constraint_gate_window_workflows
        )
        self.constraint_gate_active = not self.constraint_gate_enabled
        self.constraint_gate_active_workflows = 0

    @property
    def deadline_lagrangian_mode(self):
        return "separate"

    def select_action(
        self,
        env_state,
        deterministic=False,
        low_deterministic=None,
        high_selector_mode="policy",
    ):
        if low_deterministic is None:
            low_deterministic = deterministic
        observation = self.high_agent.build_high_state(env_state)
        context = self.high_agent._require_context(env_state)
        primary_mask, _ = self.high_agent._context_masks(context)
        candidate_context = self.c3c.encoder.encode(
            observation, context, self.low_agent
        )
        # Only structural server existence is retained.  C3C does not add a
        # safety/deadline feasibility mask.
        valid_servers = np.asarray(primary_mask, dtype=bool) & np.asarray(
            candidate_context["valid_servers"], dtype=bool
        )
        candidate_context["valid_servers"] = valid_servers
        predicted_rhos = self.c3c.predict_rhos(
            candidate_context["base_features"],
            candidate_context["hardware_mins"],
            self.low_agent.actor,
            target=False,
        )
        base_q = self.high_agent.agent_R.action_q_values(observation.vector)
        combined_q = base_q.copy()
        diagnostics = {
            "coordination_enabled": True,
            "coordination_guidance_active": False,
            "coordination_beta": float(self.c3c.beta),
            "c3c_guidance_active": False,
        }
        guidance = None
        gate_factor = self.constraint_gate_factor
        diagnostics.update(
            {
                "c3c_constraint_gate_factor": float(gate_factor),
                "c3c_constraint_gate_active": float(
                    self.constraint_gate_active
                ),
            }
        )
        if self.c3c.guidance_active and gate_factor > 0.0:
            guidance = self.c3c.candidate_guidance(
                candidate_context["base_features"],
                predicted_rhos,
                self.high_agent.deadline_lambda,
            )
            correction, correction_diagnostics = self._coordinated_correction(
                base_q,
                guidance,
                valid_servers,
                gate_factor=gate_factor,
            )
            combined_q[valid_servers] = base_q[valid_servers] + correction[
                valid_servers
            ]
            base_best = int(
                np.flatnonzero(valid_servers)[np.argmax(base_q[valid_servers])]
            )
            coordinated_best = int(
                np.flatnonzero(valid_servers)[np.argmax(combined_q[valid_servers])]
            )
            diagnostics.update(
                {
                    "coordination_guidance_active": True,
                    "c3c_guidance_active": True,
                    "coordination_base_best_server_index": base_best,
                    "coordination_best_server_index": coordinated_best,
                    "coordination_changed_greedy_action": bool(
                        base_best != coordinated_best
                    ),
                    "c3c_uncertainty_mean": float(
                        np.mean(guidance["uncertainty"][valid_servers])
                    ),
                    "c3c_confidence_mean": float(
                        np.mean(guidance["confidence"][valid_servers])
                    ),
                    "c3c_beta_effective_mean": float(
                        np.mean(guidance["beta_effective"][valid_servers])
                    ),
                    "c3c_constraint_gate_factor": float(gate_factor),
                    **correction_diagnostics,
                }
            )
        high_decision = self.high_agent.select_assignment(
            env_state,
            deterministic=deterministic,
            selector_mode=high_selector_mode,
            action_values=combined_q,
            selection_diagnostics=diagnostics,
        )
        high_decision.update(
            {
                "c3c_candidate_bases": candidate_context["base_features"],
                "c3c_hardware_mins": candidate_context["hardware_mins"],
                "c3c_valid_servers": valid_servers,
                "c3c_predicted_rhos": predicted_rhos,
            }
        )
        selected_index = int(high_decision["primary_server_index"])
        if guidance is not None:
            high_decision.update(
                {
                    "c3c_selected_advantage": float(
                        guidance["advantage"][selected_index]
                    ),
                    "c3c_selected_correction": float(
                        correction[selected_index]
                    ),
                    "c3c_selected_uncertainty": float(
                        guidance["uncertainty"][selected_index]
                    ),
                    "c3c_selected_confidence": float(
                        guidance["confidence"][selected_index]
                    ),
                }
            )

        metrics_by_id = context.get("server_safety_metrics", {})
        safety = metrics_by_id.get(str(high_decision["primary_server_id"]))
        if safety is None:
            raise KeyError(
                f"Missing safety metrics for server {high_decision['primary_server_id']}."
            )
        low_decision = self.low_agent.select_frequency(
            high_decision["observation"],
            high_decision["task_index"],
            selected_index,
            deterministic=bool(low_deterministic),
            safety_metrics=safety,
        )
        env_action = {
            "workflow_id": high_decision["workflow_id"],
            "task_id": high_decision["task_id"],
            "server_id": high_decision["primary_server_id"],
            "rho": low_decision["rho"],
            "backup": False,
            "backup_server_id": None,
        }
        return high_decision, low_decision, env_action

    @property
    def constraint_gate_factor(self):
        if not self.constraint_gate_enabled:
            return 1.0
        if not self.constraint_gate_active:
            return 0.0
        if self.constraint_gate_anneal_workflows <= 0:
            return 1.0
        return float(
            min(
                1.0,
                self.constraint_gate_active_workflows
                / float(self.constraint_gate_anneal_workflows),
            )
        )

    def constraint_gate_status(self):
        count = len(self.constraint_gate_window)
        miss_rate = (
            float(np.mean(self.constraint_gate_window)) if count else 1.0
        )
        return {
            "c3c_constraint_gate_enabled": float(self.constraint_gate_enabled),
            "c3c_constraint_gate_active": float(self.constraint_gate_active),
            "c3c_constraint_gate_factor": float(self.constraint_gate_factor),
            "c3c_constraint_gate_workflow_count": float(count),
            "c3c_constraint_gate_miss_rate": float(miss_rate),
            "c3c_constraint_gate_success_rate": float(1.0 - miss_rate),
            "c3c_constraint_gate_active_workflows": float(
                self.constraint_gate_active_workflows
            ),
        }

    def update_constraint_gate(self, workflow_misses):
        misses = np.asarray(workflow_misses, dtype=np.float64)
        if misses.ndim != 1 or (
            misses.size
            and (
                not np.isfinite(misses).all()
                or np.any(misses < 0.0)
                or np.any(misses > 1.0)
            )
        ):
            raise ValueError(
                "C3C constraint gate expects finite binary workflow misses."
            )
        self.constraint_gate_window.extend(float(value) for value in misses)
        if not self.constraint_gate_enabled:
            return self.constraint_gate_status()
        count = len(self.constraint_gate_window)
        success = 1.0 - float(np.mean(self.constraint_gate_window))
        if self.constraint_gate_active:
            if (
                count >= self.constraint_gate_min_workflows
                and success < self.constraint_gate_deactivate_success
            ):
                self.constraint_gate_active = False
                self.constraint_gate_active_workflows = 0
            else:
                self.constraint_gate_active_workflows += int(misses.size)
        elif (
            count >= self.constraint_gate_min_workflows
            and success >= self.constraint_gate_activate_success
        ):
            self.constraint_gate_active = True
            self.constraint_gate_active_workflows = int(misses.size)
        return self.constraint_gate_status()

    def _coordinated_correction(
        self,
        base_q,
        guidance,
        valid_servers,
        *,
        gate_factor,
    ):
        valid = np.asarray(valid_servers, dtype=bool)
        raw = np.asarray(guidance["correction"], dtype=np.float64)
        correction = np.zeros_like(raw)
        if not np.any(valid):
            raise RuntimeError("C3C correction requires at least one valid server.")
        q_values = np.asarray(base_q, dtype=np.float64)[valid]
        advantages = np.asarray(guidance["advantage"], dtype=np.float64)
        deadlines = (
            np.asarray(guidance["deadline"], dtype=np.float64)
            if self.relative_to_d3qn_action
            else np.zeros_like(advantages)
        )
        base_best = int(
            np.flatnonzero(valid)[np.argmax(np.asarray(base_q)[valid])]
        )
        if self.relative_to_d3qn_action:
            extra_risk = np.maximum(0.0, deadlines - deadlines[base_best])
            advantages = (
                advantages
                - advantages[base_best]
                - self.extra_deadline_risk_weight
                * max(float(self.high_agent.deadline_lambda), 0.0)
                * extra_risk
            )
        advantage_values = advantages[valid]
        q_std = max(float(np.std(q_values)), self.correction_q_std_floor)
        advantage_std = float(
            np.sqrt(np.mean(np.square(advantage_values)))
            if self.relative_to_d3qn_action
            else np.std(advantage_values)
        )
        if self.q_scale_alignment_enabled:
            if advantage_std >= self.correction_advantage_std_floor:
                centered = (
                    advantages
                    if self.relative_to_d3qn_action
                    else advantages - float(np.mean(advantage_values))
                )
                aligned = (
                    np.asarray(guidance["beta_effective"], dtype=np.float64)
                    * centered
                    * q_std
                    / advantage_std
                )
                clip = self.correction_q_std_clip * q_std
                correction[valid] = np.clip(aligned[valid], -clip, clip)
        else:
            correction[valid] = (
                np.asarray(guidance["beta_effective"], dtype=np.float64)[valid]
                * advantages[valid]
                if self.relative_to_d3qn_action
                else raw[valid]
            )
        correction *= float(gate_factor)
        return correction, {
            "c3c_base_q_std": float(q_std),
            "c3c_advantage_std": float(advantage_std),
            "c3c_correction_abs_mean": float(
                np.mean(np.abs(correction[valid]))
            ),
            "c3c_correction_to_q_std": float(
                np.mean(np.abs(correction[valid])) / q_std
            ),
            "c3c_q_scale_alignment_active": float(
                self.q_scale_alignment_enabled
            ),
            "c3c_relative_to_d3qn_action": float(
                self.relative_to_d3qn_action
            ),
            "c3c_base_action_correction": float(correction[base_best]),
            "c3c_extra_deadline_risk_mean": float(
                np.mean(
                    np.maximum(0.0, deadlines[valid] - deadlines[base_best])
                )
            ),
        }

    def recompose_and_remember_episode(
        self,
        high_buffer,
        *,
        energy_weight=1.0,
        store=True,
    ):
        components = []
        deadline_lambda = float(self.high_agent.deadline_lambda)
        for index, info in enumerate(high_buffer.infos):
            row = compute_c3c_step_components(
                info,
                energy_weight=energy_weight,
                deadline_lambda=deadline_lambda,
            )
            components.append(row)
            info.update(
                {
                    "c3c_base_energy_reward": row["base_energy_reward"],
                    "c3c_residual_energy_reward": row[
                        "residual_energy_reward"
                    ],
                    "c3c_actual_energy_reward": row["actual_energy_reward"],
                    "c3c_base_deadline_cost": row["base_deadline_cost"],
                    "c3c_actual_deadline_cost": row["actual_deadline_cost"],
                    "c3c_residual_deadline_cost": row[
                        "residual_deadline_cost"
                    ],
                    "c3c_base_reward": row["base_reward"],
                    "c3c_residual_reward": row["residual_reward"],
                    "c3c_realized_reward": row["realized_reward"],
                    "c3c_decomposition_error": row["decomposition_error"],
                    "c3c_deadline_crossed_by_dvfs": row[
                        "deadline_crossed_by_dvfs"
                    ],
                    "c3c_actual_rho": float(
                        info.get("coordination_low_rho", 1.0)
                    ),
                    # Recompose the upper replay into the rho=1 placement
                    # baseline.  Legacy ep740 fields remain under their
                    # original names elsewhere in the info record.
                    "high_base_energy_reward": row["base_energy_reward"],
                    "high_base_energy_reward_local": row["base_energy_reward"],
                    "high_deadline_constraint_cost": row[
                        "base_deadline_cost"
                    ],
                    "high_training_reward": row["base_reward"],
                    "high_reward": row["base_reward"],
                    "high_reward_profile": self.reward_profile,
                }
            )
            high_buffer.rewards[index] = float(row["base_reward"])
        stored = self.c3c.remember_from_high_buffer(high_buffer) if store else 0
        if not components:
            return {"stored": float(stored)}
        result = {"stored": float(stored)}
        for key in components[0]:
            values = np.asarray([row[key] for row in components], dtype=np.float64)
            result[f"{key}_mean"] = float(values.mean())
            result[f"{key}_std"] = float(values.std())
            result[f"{key}_min"] = float(values.min())
            result[f"{key}_max"] = float(values.max())
            result[f"{key}_zero_rate"] = float(np.mean(np.abs(values) <= 1e-12))
            result[f"{key}_unique_ratio"] = float(
                np.unique(np.round(values, decimals=8)).size / values.size
            )
        result["decomposition_error_abs_max"] = float(
            max(abs(row["decomposition_error"]) for row in components)
        )
        decisions = [info.get("high_decision", {}) for info in high_buffer.infos]
        upward_rank_features = []
        for decision in decisions:
            candidate_bases = decision.get("c3c_candidate_bases")
            if candidate_bases is None:
                continue
            candidate_bases = np.asarray(candidate_bases, dtype=np.float64)
            if (
                candidate_bases.ndim == 2
                and candidate_bases.shape[0] > 0
                and candidate_bases.shape[1] >= self.c3c.layout.task_dim
            ):
                upward_rank_features.append(
                    float(candidate_bases[0, self.c3c.layout.task_dim - 1])
                )
        result["selection_upward_rank_feature_abs_mean"] = (
            float(np.mean(np.abs(upward_rank_features)))
            if upward_rank_features
            else 0.0
        )
        for source_key, metric_key in (
            ("c3c_constraint_gate_factor", "constraint_gate_factor"),
            ("c3c_base_q_std", "base_q_std"),
            ("c3c_advantage_std", "advantage_std"),
            ("c3c_correction_abs_mean", "correction_abs_mean"),
            ("c3c_correction_to_q_std", "correction_to_q_std"),
            ("c3c_base_action_correction", "base_action_correction"),
            ("c3c_extra_deadline_risk_mean", "extra_deadline_risk"),
            ("c3c_relative_to_d3qn_action", "relative_to_d3qn_action"),
        ):
            values = [
                float(decision[source_key])
                for decision in decisions
                if source_key in decision
            ]
            result[f"selection_{metric_key}_mean"] = (
                float(np.mean(values)) if values else 0.0
            )
        return result

    def update_c3c(self):
        return self.c3c.update_critics(
            self.high_agent,
            self.low_agent,
            self.high_agent.deadline_lambda,
            coordination_factor=self.constraint_gate_factor,
        )

    def update_high(self):
        return self.c3c.update_high(
            self.high_agent,
            self.low_agent,
            self.high_agent.deadline_lambda,
            coordination_factor=self.constraint_gate_factor,
        )

    def update_low_safety(self):
        deadline_lambda = (
            self.low_agent.deadline_lambda
            if self.low_guidance_lambda_source == "low"
            else self.high_agent.deadline_lambda
        )
        metrics = self.c3c.update_low_safety(
            self.low_agent, deadline_lambda
        )
        metrics["deadline_lambda"] = float(deadline_lambda)
        metrics["deadline_lambda_is_low"] = float(
            self.low_guidance_lambda_source == "low"
        )
        return metrics

    def update_shared_deadline_lagrangian(self, workflow_misses, episode):
        del workflow_misses, episode
        raise RuntimeError("C3C v1 does not use a shared Deadline lambda.")

    def save_checkpoint(
        self,
        path,
        *,
        objective_state=None,
        episode=0,
        config=None,
        include_replay=False,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required to save C3C checkpoints.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_format": "c3c_hierarchical_d3qn_sac_v1",
            "high_q_eval_state_dict": self.high_agent.agent_R.q_eval.state_dict(),
            "high_q_target_state_dict": self.high_agent.agent_R.q_target.state_dict(),
            "high_optimizer_state_dict": self.high_agent.agent_R.optimizer.state_dict(),
            "high_epsilon": float(self.high_agent.epsilon),
            "high_deadline_lagrangian": (
                None
                if self.high_agent.deadline_lagrangian is None
                else self.high_agent.deadline_lagrangian.state_dict()
            ),
            "low_sac_state_dict": self.low_agent.state_dict(
                include_replay=include_replay
            ),
            "c3c_state_dict": self.c3c.state_dict(
                include_replay=include_replay
            ),
            "objective_state": dict(objective_state or {}),
            "constraint_gate_state": {
                "window": list(self.constraint_gate_window),
                "active": bool(self.constraint_gate_active),
                "active_workflows": int(
                    self.constraint_gate_active_workflows
                ),
            },
            "episode": int(episode),
            "config": copy.deepcopy(config if config is not None else self.config),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        torch.save(payload, str(path))

    def load_checkpoint(self, path):
        if torch is None:
            raise RuntimeError("PyTorch is required to load C3C checkpoints.")
        payload = torch.load(
            str(path), map_location=self.high_agent.agent_R.device, weights_only=False
        )
        if payload.get("checkpoint_format") != "c3c_hierarchical_d3qn_sac_v1":
            raise RuntimeError("Checkpoint is not an independent C3C checkpoint.")
        self.high_agent.agent_R.q_eval.load_state_dict(
            payload["high_q_eval_state_dict"]
        )
        self.high_agent.agent_R.q_target.load_state_dict(
            payload["high_q_target_state_dict"]
        )
        self.high_agent.agent_R.optimizer.load_state_dict(
            payload["high_optimizer_state_dict"]
        )
        self.high_agent.epsilon = float(payload["high_epsilon"])
        if (
            self.high_agent.deadline_lagrangian is not None
            and payload.get("high_deadline_lagrangian") is not None
        ):
            self.high_agent.deadline_lagrangian.load_state_dict(
                payload["high_deadline_lagrangian"]
            )
        self.low_agent.load_state_dict(payload["low_sac_state_dict"])
        self.c3c.load_state_dict(payload["c3c_state_dict"])
        gate_state = dict(payload.get("constraint_gate_state") or {})
        restored_window = np.asarray(
            gate_state.get("window", []), dtype=np.float64
        )
        if restored_window.ndim != 1 or (
            restored_window.size
            and (
                not np.isfinite(restored_window).all()
                or np.any(restored_window < 0.0)
                or np.any(restored_window > 1.0)
            )
        ):
            raise ValueError("C3C checkpoint contains an invalid gate window.")
        self.constraint_gate_window = deque(
            (
                float(value)
                for value in restored_window[
                    -self.constraint_gate_window_workflows :
                ]
            ),
            maxlen=self.constraint_gate_window_workflows,
        )
        self.constraint_gate_active = bool(
            gate_state.get("active", not self.constraint_gate_enabled)
        )
        self.constraint_gate_active_workflows = max(
            0, int(gate_state.get("active_workflows", 0))
        )
        if payload.get("numpy_random_state") is not None:
            np.random.set_state(payload["numpy_random_state"])
        if payload.get("torch_random_state") is not None:
            torch.random.set_rng_state(
                as_torch_rng_state(payload["torch_random_state"])
            )
        if torch.cuda.is_available() and payload.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(
                [as_torch_rng_state(row) for row in payload["cuda_random_state"]]
            )
        return payload
