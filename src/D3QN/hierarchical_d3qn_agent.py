from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from src.D3QN.D3QN import D3QN
from src.D3QN.deadline_lagrangian import DeadlineLagrangianController
from src.D3QN.state import HierarchicalStateEncoder
from src.D3QN.utils import as_torch_rng_state
from src.coordination.hcc import HierarchicalCoordinationCritic
from src.ppo.hierarchical_agent import LowLevelDVFSPPOAgent
from src.sac.deadline_lagrangian import LowDeadlineLagrangianController
from src.sac.hierarchical_low_sac import LowLevelDVFSContinuousSACAgent

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class HighLevelD3QNAgent:
    @staticmethod
    def build_state_encoder(state_cfg):
        """Factory hook so variants can widen the upper state.

        The production agent returns the unchanged 347-D encoder.
        """
        return HierarchicalStateEncoder(state_cfg)

    def __init__(self, config=None, seed=42):
        config = dict(config or {})
        self.config = copy.deepcopy(config)
        self.seed = int(seed)
        np.random.seed(self.seed)
        if torch is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        state_cfg = config.get("state", config)
        self.encoder = self.build_state_encoder(state_cfg)
        self.max_servers = int(self.encoder.max_servers)
        self.action_mode = str(config.get("action_mode", "primary_only")).lower()
        if self.action_mode != "primary_only":
            raise ValueError("High-level D3QN supports only action_mode='primary_only'.")
        self.backup_selection = str(
            config.get("backup_selection", "reliability_then_finish_time")
        ).lower()
        if self.backup_selection not in {"disabled", "reliability_then_finish_time"}:
            raise ValueError(
                "High-level D3QN backup_selection must be 'disabled' or "
                "'reliability_then_finish_time'."
            )
        self.action_dim = self.max_servers
        self.training_per_episode = int(config.get("training_per_episode", 10))
        reward_config = dict(config.get("reward", {}))
        self.reward_mode = str(reward_config.get("mode", "scalar")).strip().lower()
        if self.reward_mode not in {"scalar", "replay_zscore"}:
            raise ValueError("High D3QN reward.mode must be 'scalar' or 'replay_zscore'.")
        self.reward_std_floor = float(reward_config.get("std_floor", 0.05))
        self.lagrangian_energy_clip = float(
            reward_config.get("component_clip", 3.0)
        )
        self.lagrangian_energy_normalization = str(
            reward_config.get(
                "lagrangian_energy_normalization", "replay_zscore"
            )
        ).strip().lower()
        self.reward_weights = (
            float(reward_config.get("workflow_slack_weight", 1.0)),
            float(reward_config.get("energy_saving_weight", 1.0)),
        )
        self.deadline_violation_weight = float(
            reward_config.get("deadline_violation_weight", 2.0)
        )
        self.deadline_cost_reward_weight = float(config.get("deadline_cost_reward_weight", 0.0))
        lagrangian_config = dict(config.get("lagrangian", {}))
        lagrangian_requested = bool(
            lagrangian_config.get("enabled", False)
        )
        objective_reward_mode = config.get("objective_reward_mode")
        if objective_reward_mode is None and lagrangian_requested:
            # Evaluation utilities may construct the agent directly from the
            # raw agent.high block.  An explicit mode always wins; otherwise
            # an enabled Lagrangian block identifies the new checkpoint mode.
            objective_reward_mode = "lagrangian_deadline"
        self.objective_reward_mode = str(
            objective_reward_mode or ""
        ).strip().lower()
        self.deadline_lagrangian_enabled = (
            self.objective_reward_mode == "lagrangian_deadline"
            and lagrangian_requested
        )
        if self.objective_reward_mode == "lagrangian_deadline":
            if not lagrangian_requested:
                raise ValueError(
                    "objective.reward_mode='lagrangian_deadline' requires "
                    "high.lagrangian.enabled=true."
                )
            self.deadline_lagrangian = DeadlineLagrangianController.from_config(
                lagrangian_config
            )
        else:
            self.deadline_lagrangian = None
        self.agent_R = D3QN(
            state_dim=self.encoder.high_state_dim,
            action_dim=self.action_dim,
            alpha=float(config.get("learning_rate", config.get("alpha", 3e-4))),
            hidden_dim=int(config.get("hidden_dim", config.get("fc1_dim", 256))),
            hidden_layers=int(config.get("hidden_layers", 2)),
            gamma=float(config.get("gamma", config.get("reward_gamma", 0.99))),
            tau=float(config.get("tau", 0.005)),
            epsilon=float(config.get("epsilon", 1.0)),
            eps_end=float(config.get("epsilon_min", config.get("eps_end", 0.05))),
            eps_dec=float(config.get("epsilon_decay", config.get("eps_dec", 2e-4))),
            max_size=int(config.get("replay_size", config.get("max_size", 100000))),
            batch_size=int(config.get("batch_size", 256)),
            max_grad_norm=float(config.get("max_grad_norm", 1.0)),
            reward_mode=self.reward_mode,
            reward_weights=self.reward_weights,
            deadline_violation_weight=self.deadline_violation_weight,
            reward_std_floor=self.reward_std_floor,
            lagrangian_energy_clip=self.lagrangian_energy_clip,
            lagrangian_energy_normalization=(
                self.lagrangian_energy_normalization
            ),
            device=config.get("device", "auto"),
            seed=self.seed,
        )

    @property
    def epsilon(self):
        return self.agent_R.epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.agent_R.epsilon = float(value)

    @property
    def replay(self):
        return self.agent_R.memory

    def build_high_state(self, env_state):
        return self.encoder.encode_high_state(env_state)

    def select_assignment(
        self,
        env_state,
        deterministic=False,
        selector_mode="policy",
        action_values=None,
        selection_diagnostics=None,
    ):
        if selector_mode != "policy":
            raise ValueError("Only selector_mode='policy' is supported for D3QN training.")
        observation = self.build_high_state(env_state)
        context = self._require_context(env_state)
        primary_mask, backup_pair_mask = self._context_masks(context)
        base_q_values = self.agent_R.action_q_values(observation.vector)
        candidate_values = (
            base_q_values
            if action_values is None
            else np.asarray(action_values, dtype=np.float64)
        )
        action = self.agent_R.choose_action_from_values(
            candidate_values, primary_mask, isTrain=not bool(deterministic)
        )
        primary_idx = self.decode_action(action)
        backup_enabled = bool(context.get("backup_enabled", True))
        if self.backup_selection == "disabled" or not backup_enabled:
            backup_idx = None
            effective_backup_mask = np.zeros(self.max_servers, dtype=bool)
            backup_shield_relaxed = False
        else:
            backup_idx, effective_backup_mask, backup_shield_relaxed = self._select_backup_index(
                observation,
                context,
                primary_idx,
                backup_pair_mask,
            )
        q_value = float(base_q_values[action])
        decision = self._decision_payload(
            observation,
            context,
            primary_mask,
            backup_pair_mask,
            primary_idx,
            backup_idx,
            action,
            q_value,
            selector_mode,
            effective_backup_mask,
            backup_shield_relaxed,
        )
        decision.update(dict(selection_diagnostics or {}))
        return decision

    def remember_from_high_buffer(self, buffer):
        if buffer.size == 0:
            return {"stored": 0.0, "deadline_reward_penalty_total": 0.0}
        stored = 0
        deadline_reward_penalty_total = 0.0
        zero_state = np.zeros(self.encoder.high_state_dim, dtype=np.float32)
        zero_mask = np.zeros(self.action_dim, dtype=bool)
        for index in range(buffer.size):
            action = self.encode_action(buffer.primary_actions[index])
            action_mask = np.asarray(buffer.primary_masks[index], dtype=bool).copy()
            done = index == buffer.size - 1
            if done:
                next_state = zero_state
                next_mask = zero_mask
            else:
                next_state = buffer.states[index + 1]
                next_mask = np.asarray(buffer.primary_masks[index + 1], dtype=bool).copy()
            info = buffer.infos[index]
            base_energy_reward = None
            deadline_constraint_cost = 0.0
            if self.deadline_lagrangian_enabled:
                base_energy_reward = float(info["high_base_energy_reward"])
                # Workflow-terminal causal credit is patched into the info
                # stream after completion.  It sums to one binary miss per
                # violated workflow and is therefore aligned with the dual.
                deadline_constraint_cost = float(
                    info["high_deadline_constraint_cost"]
                )
                current_lambda = self.deadline_lagrangian.value
                replay_reward = (
                    base_energy_reward
                    - current_lambda * deadline_constraint_cost
                )
                reward_components = None
                deadline_crossed = float(info["high_deadline_crossed"])
                deadline_violation = float(info["high_deadline_violation"])
                deadline_reward_penalty = 0.0
            elif self.reward_mode == "replay_zscore":
                reward_components = (
                    float(info["high_workflow_slack_ratio"]),
                    float(info["high_energy_saving_joules"]),
                )
                deadline_crossed = float(info["high_deadline_crossed"])
                deadline_violation = float(info["high_deadline_violation"])
                deadline_reward_penalty = 0.0
                replay_reward = float(buffer.rewards[index])
            else:
                deadline_limit = max(float(buffer.deadline_limits[index]), 1e-8)
                normalized_deadline_cost = float(buffer.deadline_costs[index]) / deadline_limit
                deadline_reward_penalty = self.deadline_cost_reward_weight * normalized_deadline_cost
                replay_reward = float(buffer.rewards[index]) - deadline_reward_penalty
                reward_components = None
                deadline_crossed = 0.0
                deadline_violation = 0.0
                base_energy_reward = None
                deadline_constraint_cost = 0.0
            self.agent_R.remember(
                buffer.states[index],
                action,
                replay_reward,
                next_state,
                done,
                action_mask,
                next_mask,
                reward_components=reward_components,
                deadline_crossed=deadline_crossed,
                deadline_violation=deadline_violation,
                base_energy_reward=(
                    base_energy_reward
                    if self.deadline_lagrangian_enabled
                    else None
                ),
                deadline_constraint_cost=deadline_constraint_cost,
            )
            stored += 1
            deadline_reward_penalty_total += deadline_reward_penalty
        return {
            "stored": float(stored),
            "deadline_reward_penalty_total": float(deadline_reward_penalty_total),
        }

    def update(self, buffer=None):
        remember_metrics = self.remember_from_high_buffer(buffer) if buffer is not None else {
            "stored": 0.0,
            "deadline_reward_penalty_total": 0.0,
        }
        totals = {
            "replay_size": float(self.replay.size),
            "epsilon": float(self.epsilon),
            "stored_transitions": float(remember_metrics["stored"]),
            "deadline_reward_penalty_total": float(remember_metrics["deadline_reward_penalty_total"]),
            "deadline_cost_reward_weight": float(self.deadline_cost_reward_weight),
        }
        updates = 0
        reward_statistics = (
            self.replay.reward_statistics(self.reward_std_floor)
            if self.reward_mode == "replay_zscore"
            else None
        )
        for _ in range(self.training_per_episode):
            metrics = self.agent_R.learn(
                reward_statistics,
                deadline_lambda=(
                    self.deadline_lagrangian.value
                    if self.deadline_lagrangian_enabled
                    else None
                ),
            )
            if not metrics:
                continue
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            updates += 1
        if updates > 0:
            for key in list(totals.keys()):
                if key in {
                    "replay_size",
                    "epsilon",
                    "stored_transitions",
                    "deadline_reward_penalty_total",
                    "deadline_cost_reward_weight",
                }:
                    continue
                totals[key] /= updates
        totals["updates"] = float(updates)
        totals["epsilon"] = float(self.epsilon)
        return totals

    @property
    def deadline_lambda(self):
        if self.deadline_lagrangian is None:
            return 0.0
        return self.deadline_lagrangian.value

    def update_deadline_lagrangian(self, workflow_usage_ratios, episode):
        if self.deadline_lagrangian is None:
            return {}
        return self.deadline_lagrangian.update(
            workflow_usage_ratios, episode=episode
        )

    def encode_action(self, primary_idx, backup_idx=None):
        del backup_idx
        primary_idx = int(primary_idx)
        if not 0 <= primary_idx < self.max_servers:
            raise ValueError(f"Primary server index out of range: {primary_idx}.")
        return primary_idx

    def decode_action(self, action):
        action = int(action)
        if not 0 <= action < self.max_servers:
            raise ValueError(f"Primary server action out of range: {action}.")
        return action

    def _context_masks(self, context):
        primary_raw = np.asarray(context["primary_server_mask"], dtype=bool)
        pair_raw = np.asarray(context["backup_pair_mask"], dtype=bool)
        if primary_raw.ndim != 1:
            raise ValueError("decision_context primary_server_mask must be one-dimensional.")
        if pair_raw.shape != (primary_raw.shape[0], primary_raw.shape[0]):
            raise ValueError("decision_context backup_pair_mask must be [server_count, server_count].")
        if primary_raw.shape[0] > self.max_servers:
            raise ValueError("server count exceeds encoder max_servers.")
        primary_mask = np.zeros(self.max_servers, dtype=bool)
        backup_pair_mask = np.zeros((self.max_servers, self.max_servers), dtype=bool)
        # False is used only for padded, non-existent server indices. All real
        # servers remain selectable regardless of deadline feasibility.
        primary_mask[: primary_raw.shape[0]] = True
        backup_pair_mask[: pair_raw.shape[0], : pair_raw.shape[1]] = pair_raw
        return primary_mask, backup_pair_mask

    def _select_backup_index(
        self,
        observation,
        context,
        primary_idx,
        backup_pair_mask,
    ):
        effective_mask = np.asarray(backup_pair_mask[int(primary_idx)], dtype=bool).copy()
        effective_mask[int(primary_idx)] = False
        backup_shield_relaxed = False
        if int(effective_mask.sum()) <= 0:
            effective_mask = np.zeros(self.max_servers, dtype=bool)
            effective_mask[: len(observation.servers)] = True
            effective_mask[int(primary_idx)] = False
            backup_shield_relaxed = True
        candidates = np.flatnonzero(effective_mask)
        if candidates.size <= 0:
            raise ValueError("No backup server is available after excluding the primary server.")
        metrics_by_id = context.get("server_safety_metrics", {})

        def backup_rank(index):
            server_id = int(observation.servers[int(index)]["id"])
            metrics = metrics_by_id.get(str(server_id), {})
            reliability = self._finite_metric(
                metrics.get("task_reliability_rho_1"), default=0.0
            )
            estimated_finish_time = metrics.get("estimated_finish_time")
            if estimated_finish_time is None:
                estimated_finish_time = self._finite_metric(
                    metrics.get("estimated_ast"), default=float("inf")
                ) + self._finite_metric(
                    metrics.get("execution_time_rho_1"), default=float("inf")
                )
            finish_time = self._finite_metric(
                estimated_finish_time, default=float("inf")
            )
            energy = self._finite_metric(
                metrics.get("primary_energy_ref_rho1"), default=float("inf")
            )
            return (-reliability, finish_time, energy, server_id)

        backup_idx = min((int(index) for index in candidates), key=backup_rank)
        return backup_idx, effective_mask, backup_shield_relaxed

    def _decision_payload(
        self,
        observation,
        context,
        primary_mask,
        backup_pair_mask,
        primary_idx,
        backup_idx,
        action,
        q_value,
        selector_mode,
        effective_backup_mask,
        backup_shield_relaxed,
    ):
        if primary_idx >= len(observation.servers):
            raise ValueError("Selected server index exceeds observed server count.")
        if backup_idx is not None and backup_idx >= len(observation.servers):
            raise ValueError("Selected backup server index exceeds observed server count.")
        task = observation.ready_tasks[0]
        primary_server = observation.servers[primary_idx]
        backup_server = None if backup_idx is None else observation.servers[backup_idx]
        return {
            "observation": observation,
            "state": observation.vector,
            "task_index": 0,
            "primary_server_index": int(primary_idx),
            "backup_server_index": None if backup_idx is None else int(backup_idx),
            "server_index": int(primary_idx),
            "primary_action": int(action),
            "workflow_id": str(task["workflow_id"]),
            "task_id": int(task["task_id"]),
            "primary_server_id": int(primary_server["id"]),
            "backup_server_id": None if backup_server is None else int(backup_server["id"]),
            "server_id": int(primary_server["id"]),
            "primary_server_mask": primary_mask.copy(),
            "backup_pair_mask": backup_pair_mask.copy(),
            "effective_backup_mask": np.asarray(effective_backup_mask, dtype=bool).copy(),
            "log_prob": 0.0,
            "value": float(q_value),
            "deadline_value": 0.0,
            "reliability_value": 0.0,
            "entropy": 0.0,
            "action_mask_mode": "structural_only",
            "shield_relaxed": False,
            "shield_relaxation_reason": None,
            "backup_shield_relaxed": bool(backup_shield_relaxed),
            "backup_selector": self.backup_selection,
            "selector_mode": selector_mode,
        }

    @staticmethod
    def _finite_metric(value, default):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(default)
        return value if np.isfinite(value) else float(default)

    def _require_context(self, env_state):
        context = env_state.get("decision_context")
        if context is None:
            raise KeyError("env_state must contain decision_context for hierarchical D3QN.")
        return context


class HierarchicalD3QNAgent:
    @staticmethod
    def build_low_sac_agent(low_cfg, seed):
        """Factory hook so variants can swap the low SAC reward scaling."""
        return LowLevelDVFSContinuousSACAgent(low_cfg, seed=seed)

    def __init__(self, config=None, seed=42):
        config = dict(config or {})
        self.config = copy.deepcopy(config)
        self.seed = int(seed)
        shared = dict(config.get("shared", {}))
        high_cfg = {**shared, **dict(config.get("high", {}))}
        low_cfg = {**shared, **dict(config.get("low", {}))}
        self.use_low_agent = bool(config.get("use_low_agent", True))
        self.low_algorithm = str(config.get("low_algorithm", low_cfg.get("algorithm", "ppo"))).lower()
        shared_lagrangian_config = dict(
            config.get("shared_deadline_lagrangian", {})
        )
        self.shared_deadline_lagrangian_enabled = bool(
            shared_lagrangian_config.get("enabled", False)
        )
        if self.shared_deadline_lagrangian_enabled:
            if str(high_cfg.get("objective_reward_mode", "")).strip().lower() != "lagrangian_deadline":
                raise ValueError(
                    "shared_deadline_lagrangian.enabled=true requires "
                    "objective.reward_mode='lagrangian_deadline'."
                )
            if self.use_low_agent and self.low_algorithm != "sac":
                raise ValueError(
                    "shared Deadline Lagrangian supports the low SAC agent only."
                )
            if self.use_low_agent and str(
                dict(low_cfg.get("reward", {})).get("mode", "")
            ).strip().lower() != "lagrangian_deadline":
                raise ValueError(
                    "shared_deadline_lagrangian.enabled=true requires "
                    "low.reward.mode='lagrangian_deadline'."
                )
            # Constructors still validate their local Lagrangian modes. The
            # shared block is the single source of numerical hyperparameters.
            shared_overrides = {
                **shared_lagrangian_config,
                "enabled": True,
            }
            high_cfg["lagrangian"] = {
                **dict(high_cfg.get("lagrangian", {})),
                **shared_overrides,
            }
            low_cfg["lagrangian"] = {
                **dict(low_cfg.get("lagrangian", {})),
                **shared_overrides,
            }
        self.high_agent = HighLevelD3QNAgent(high_cfg, seed=self.seed)
        # The optional shared-state SAC reuses the exact high observation
        # vector and appends only the selected-server identity.  Inject the
        # authoritative D3QN state dimension because the legacy PPO encoder
        # used inside the low agent has a different historical high state.
        low_cfg["shared_high_state_dim"] = int(
            self.high_agent.encoder.high_state_dim
        )
        low_cfg["shared_high_server_feature_dim"] = int(
            self.high_agent.encoder.server_feature_dim
        )
        low_cfg["shared_high_global_feature_dim"] = int(
            self.high_agent.encoder.global_feature_dim
        )
        if not self.use_low_agent:
            self.low_agent = None
        elif self.low_algorithm == "sac":
            self.low_agent = self.build_low_sac_agent(low_cfg, self.seed)
        elif self.low_algorithm == "ppo":
            self.low_agent = LowLevelDVFSPPOAgent(low_cfg, seed=self.seed)
        else:
            raise ValueError(f"Unsupported D3QN low_algorithm: {self.low_algorithm}")
        self.shared_deadline_lagrangian = None
        if self.shared_deadline_lagrangian_enabled:
            self.shared_deadline_lagrangian = (
                LowDeadlineLagrangianController.from_config(
                    shared_lagrangian_config
                )
            )
            self.high_agent.deadline_lagrangian = (
                self.shared_deadline_lagrangian
            )
            self.high_agent.deadline_lagrangian_enabled = True
            if self.low_agent is not None:
                self.low_agent.deadline_lagrangian = (
                    self.shared_deadline_lagrangian
                )
        self._validate_encoder_config()
        self._validate_shared_deadline_lagrangian()
        coordination_cfg = {
            **shared,
            **dict(config.get("coordination_critic", {})),
        }
        self.coordination_critic_enabled = bool(
            coordination_cfg.get("enabled", False)
        )
        if self.coordination_critic_enabled:
            if self.low_agent is None or self.low_algorithm != "sac":
                raise ValueError(
                    "coordination_critic.enabled=true requires the low SAC agent."
                )
            self.coordination_critic = HierarchicalCoordinationCritic(
                state_dim=self.high_agent.encoder.high_state_dim,
                action_dim=self.high_agent.action_dim,
                config=coordination_cfg,
                seed=self.seed + 1009,
            )
        else:
            self.coordination_critic = None

    def _validate_encoder_config(self):
        if self.low_agent is None:
            return
        for field in ("max_ready_tasks", "max_servers"):
            if getattr(self.high_agent.encoder, field) != getattr(self.low_agent.encoder, field):
                raise ValueError(f"High/low encoder config mismatch for {field}.")

    def _validate_shared_deadline_lagrangian(self):
        if not self.shared_deadline_lagrangian_enabled:
            return
        if self.high_agent.deadline_lagrangian is not self.shared_deadline_lagrangian:
            raise RuntimeError("High agent is not bound to the shared Deadline lambda.")
        if (
            self.low_agent is not None
            and self.low_agent.deadline_lagrangian
            is not self.shared_deadline_lagrangian
        ):
            raise RuntimeError("Low agent is not bound to the shared Deadline lambda.")

    @property
    def deadline_lagrangian_mode(self):
        if self.shared_deadline_lagrangian_enabled:
            return "shared"
        return "separate"

    def update_shared_deadline_lagrangian(self, workflow_misses, episode):
        if self.shared_deadline_lagrangian is None:
            return {}
        low_metrics = self.shared_deadline_lagrangian.update(
            workflow_misses, episode=episode
        )
        before = float(low_metrics["low_deadline_lambda_before_update"])
        after = float(low_metrics["low_deadline_lambda_after_update"])
        budget = float(low_metrics["low_deadline_constraint_budget"])
        batch_mean = float(
            low_metrics["low_deadline_constraint_batch_mean"]
        )
        return {
            **low_metrics,
            "deadline_lagrangian_mode": "shared",
            "shared_deadline_lambda": after,
            "shared_deadline_lambda_before_update": before,
            "shared_deadline_lambda_after_update": after,
            "shared_deadline_constraint_budget": budget,
            "shared_deadline_success_target": 1.0 - budget,
            "shared_deadline_constraint_mean": float(
                low_metrics["low_deadline_constraint_mean"]
            ),
            "shared_deadline_constraint_excess": float(
                low_metrics["low_deadline_constraint_excess"]
            ),
            "shared_deadline_constraint_workflow_count": int(
                low_metrics["low_deadline_constraint_workflow_count"]
            ),
            "shared_deadline_constraint_batch_mean": batch_mean,
            "shared_deadline_constraint_batch_workflow_count": int(
                low_metrics["low_deadline_constraint_batch_workflow_count"]
            ),
            "shared_deadline_dual_updated": float(
                low_metrics["low_deadline_dual_updated"]
            ),
            "shared_deadline_dual_warmup_active": float(
                low_metrics["low_deadline_dual_warmup_active"]
            ),
            # Backward-compatible high-level names keep existing plots and
            # metric readers working while both agents use the same value.
            "lambda_deadline": after,
            "lambda_deadline_before_update": before,
            "lambda_deadline_after_update": after,
            "deadline_constraint_budget": budget,
            "deadline_success_target": 1.0 - budget,
            "deadline_miss_rate": batch_mean,
            "deadline_excess": batch_mean - budget,
            "completed_workflow_count": int(
                low_metrics["low_deadline_constraint_batch_workflow_count"]
            ),
            "deadline_dual_updated": float(
                low_metrics["low_deadline_dual_updated"]
            ),
            "deadline_dual_warmup_active": float(
                low_metrics["low_deadline_dual_warmup_active"]
            ),
        }

    @staticmethod
    def _standardize_candidates(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size <= 1:
            return np.zeros_like(values)
        return (values - values.mean()) / max(float(values.std()), 1e-6)

    def _predict_candidate_rhos(self, observation, context, candidate_indices):
        metrics_by_id = context.get("server_safety_metrics", {})
        low_states = []
        hardware_mins = []
        for server_index in candidate_indices:
            server_id = int(observation.servers[int(server_index)]["id"])
            safety_metrics = metrics_by_id.get(str(server_id))
            if safety_metrics is None:
                raise KeyError(f"Missing safety metrics for server {server_id}.")
            low_states.append(
                self.low_agent.build_low_state(
                    observation,
                    0,
                    int(server_index),
                    safety_metrics=safety_metrics,
                )
            )
            hardware_mins.append(
                min(
                    1.0,
                    max(
                        self.low_agent.rho_min,
                        float(
                            safety_metrics.get(
                                "rho_hardware_min", self.low_agent.rho_min
                            )
                        ),
                    ),
                )
            )
        state_tensor = torch.as_tensor(
            np.asarray(low_states, dtype=np.float32),
            dtype=torch.float32,
            device=self.low_agent.device,
        )
        with torch.no_grad():
            actions, _, _, _ = self.low_agent.actor.sample(
                state_tensor, deterministic=True
            )
        rhos = actions.detach().cpu().numpy().reshape(-1).astype(np.float64)
        return np.clip(
            np.maximum(rhos, np.asarray(hardware_mins, dtype=np.float64)),
            self.low_agent.rho_min,
            1.0,
        )

    def _select_coordinated_assignment(
        self, env_state, deterministic=False, high_selector_mode="policy"
    ):
        critic = self.coordination_critic
        if critic is None or not critic.guidance_active:
            return self.high_agent.select_assignment(
                env_state,
                deterministic=deterministic,
                selector_mode=high_selector_mode,
                selection_diagnostics={
                    "coordination_enabled": bool(critic is not None),
                    "coordination_guidance_active": False,
                    "coordination_beta": 0.0 if critic is None else critic.beta,
                },
            )

        observation = self.high_agent.build_high_state(env_state)
        context = self.high_agent._require_context(env_state)
        primary_mask, _ = self.high_agent._context_masks(context)
        candidate_indices = np.flatnonzero(primary_mask)
        candidate_indices = candidate_indices[
            candidate_indices < len(observation.servers)
        ]
        if candidate_indices.size <= 0:
            raise ValueError("Coordination critic has no real server candidate.")
        predicted_rhos = self._predict_candidate_rhos(
            observation, context, candidate_indices
        )
        base_q = self.high_agent.agent_R.action_q_values(observation.vector)
        coordination = critic.candidate_coordination_values(
            observation.vector,
            candidate_indices,
            predicted_rhos,
            self.high_agent.deadline_lambda,
        )
        beta = critic.beta
        combined_values = base_q.copy()
        combined_values[candidate_indices] = (
            (1.0 - beta)
            * self._standardize_candidates(base_q[candidate_indices])
            + beta
            * self._standardize_candidates(coordination["values"])
        )
        base_best = int(candidate_indices[np.argmax(base_q[candidate_indices])])
        coordinated_best = int(
            candidate_indices[np.argmax(combined_values[candidate_indices])]
        )
        selected = self.high_agent.select_assignment(
            env_state,
            deterministic=deterministic,
            selector_mode=high_selector_mode,
            action_values=combined_values,
            selection_diagnostics={
                "coordination_enabled": True,
                "coordination_guidance_active": True,
                "coordination_beta": float(beta),
                "coordination_base_best_server_index": base_best,
                "coordination_best_server_index": coordinated_best,
                "coordination_changed_greedy_action": bool(
                    base_best != coordinated_best
                ),
            },
        )
        selected_index = int(selected["primary_server_index"])
        candidate_position = int(
            np.flatnonzero(candidate_indices == selected_index)[0]
        )
        selected.update(
            {
                "coordination_predicted_rho": float(
                    predicted_rhos[candidate_position]
                ),
                "coordination_selected_value": float(
                    coordination["values"][candidate_position]
                ),
                "coordination_selected_energy_cost": float(
                    coordination["energy_costs"][candidate_position]
                ),
                "coordination_selected_deadline_cost": float(
                    coordination["deadline_costs"][candidate_position]
                ),
            }
        )
        return selected

    def select_action(self, env_state, deterministic=False, low_deterministic=None, high_selector_mode="policy"):
        if low_deterministic is None:
            low_deterministic = deterministic
        high_decision = self._select_coordinated_assignment(
            env_state,
            deterministic=deterministic,
            high_selector_mode=high_selector_mode,
        )
        if self.low_agent is None:
            low_decision = {
                "state": np.zeros(1, dtype=np.float32),
                "action_rho": 1.0,
                "action_u": 1.0,
                "rho": 1.0,
                "log_prob": 0.0,
                "value": 0.0,
                "rho_alpha": 0.0,
                "rho_beta": 0.0,
                "entropy": 0.0,
                "rho_safe": 1.0,
                "low_action_feasible": True,
                "has_action_span": False,
                "actor_update_mask": False,
                "forced_max_frequency": False,
                "fixed_rho_one": True,
            }
        else:
            context = env_state["decision_context"]
            metrics_by_id = context.get("server_safety_metrics", {})
            safety_metrics = metrics_by_id.get(str(high_decision["primary_server_id"]))
            if safety_metrics is None:
                raise KeyError(f"Missing safety metrics for server {high_decision['primary_server_id']}.")
            low_decision = self.low_agent.select_frequency(
                high_decision["observation"],
                high_decision["task_index"],
                high_decision["primary_server_index"],
                deterministic=bool(low_deterministic),
                safety_metrics=safety_metrics,
            )
        env_action = {
            "workflow_id": high_decision["workflow_id"],
            "task_id": high_decision["task_id"],
            "server_id": high_decision["primary_server_id"],
            "rho": low_decision["rho"],
            "backup": high_decision["backup_server_id"] is not None,
            "backup_server_id": high_decision["backup_server_id"],
        }
        return high_decision, low_decision, env_action

    def save_checkpoint(self, path, *, objective_state=None, episode=0, training_phase=None, config=None, include_replay=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_format": "hierarchical_d3qn_primary_v3",
            "high_action_mode": self.high_agent.action_mode,
            "backup_selection": self.high_agent.backup_selection,
            "high_q_eval_state_dict": self.high_agent.agent_R.q_eval.state_dict(),
            "high_q_target_state_dict": self.high_agent.agent_R.q_target.state_dict(),
            "high_optimizer_state_dict": self.high_agent.agent_R.optimizer.state_dict(),
            "high_replay_state_dict": self.high_agent.replay.state_dict() if include_replay else None,
            "high_replay_saved": bool(include_replay),
            "high_epsilon": float(self.high_agent.epsilon),
            "deadline_lagrangian_mode": self.deadline_lagrangian_mode,
            "shared_deadline_lagrangian": self._shared_lagrangian_state_dict(),
            "deadline_lagrangian": (
                None
                if self.shared_deadline_lagrangian_enabled
                or self.high_agent.deadline_lagrangian is None
                else self.high_agent.deadline_lagrangian.state_dict()
            ),
            "use_low_agent": bool(self.use_low_agent),
            "low_algorithm": self.low_algorithm,
            "low_model_state_dict": self._low_model_state_dict(include_replay=include_replay),
            "coordination_critic_state_dict": (
                None
                if self.coordination_critic is None
                else self.coordination_critic.state_dict(
                    include_replay=include_replay
                )
            ),
            "low_optimizer_state_dict": None if self.low_agent is None or self.low_algorithm == "sac" else self.low_agent.optimizer.state_dict(),
            "objective_state": dict(objective_state or {}),
            "episode": int(episode),
            "training_phase": training_phase,
            "config": copy.deepcopy(config if config is not None else self.config),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(payload, str(path))
        self._save_component_checkpoints(
            path,
            objective_state=objective_state,
            episode=episode,
            training_phase=training_phase,
            config=config,
        )

    def _save_component_checkpoints(self, path, *, objective_state=None, episode=0, training_phase=None, config=None):
        common = {
            "episode": int(episode),
            "training_phase": training_phase,
            "config": copy.deepcopy(config if config is not None else self.config),
            "objective_state": dict(objective_state or {}),
            "deadline_lagrangian_mode": self.deadline_lagrangian_mode,
            "shared_deadline_lagrangian": self._shared_lagrangian_state_dict(),
        }
        high_payload = {
            "checkpoint_format": "hierarchical_d3qn_high_primary_v2",
            "high_action_mode": self.high_agent.action_mode,
            "backup_selection": self.high_agent.backup_selection,
            "high_q_eval_state_dict": self.high_agent.agent_R.q_eval.state_dict(),
            "high_q_target_state_dict": self.high_agent.agent_R.q_target.state_dict(),
            "high_optimizer_state_dict": self.high_agent.agent_R.optimizer.state_dict(),
            "high_epsilon": float(self.high_agent.epsilon),
            "high_action_dim": int(self.high_agent.action_dim),
            "deadline_lagrangian": (
                None
                if self.shared_deadline_lagrangian_enabled
                or self.high_agent.deadline_lagrangian is None
                else self.high_agent.deadline_lagrangian.state_dict()
            ),
            **common,
        }
        torch.save(high_payload, str(self._component_checkpoint_path(path, "high")))
        if self.coordination_critic is not None:
            coordination_payload = {
                "checkpoint_format": "hierarchical_coordination_critic_v1",
                "coordination_critic_state_dict": (
                    self.coordination_critic.state_dict(include_replay=False)
                ),
                **common,
            }
            torch.save(
                coordination_payload,
                str(self._component_checkpoint_path(path, "coordination")),
            )
        if self.low_agent is None:
            return
        if self.low_algorithm == "sac":
            low_payload = {
                "checkpoint_format": "hierarchical_d3qn_low_sac_v1",
                "low_algorithm": "sac",
                "low_sac_state_dict": self._low_model_state_dict(
                    include_replay=False
                ),
                **common,
            }
            torch.save(low_payload, str(self._component_checkpoint_path(path, "low")))
            return
        low_payload = {
            "checkpoint_format": "hierarchical_d3qn_low_ppo_v1",
            "low_algorithm": "ppo",
            "low_model_state_dict": self.low_agent.network.state_dict(),
            "low_optimizer_state_dict": self.low_agent.optimizer.state_dict(),
            "low_has_deadline_cost_critic": True,
            **common,
        }
        torch.save(low_payload, str(self._component_checkpoint_path(path, "low")))

    @staticmethod
    def _component_checkpoint_path(path, component):
        path = Path(path)
        name = path.name
        if name.endswith("_checkpoint.pt"):
            return path.with_name(name.replace("_checkpoint.pt", f"_{component}.pt"))
        if name.endswith(".pt"):
            return path.with_name(f"{path.stem}_{component}.pt")
        return path.with_name(f"{name}_{component}.pt")

    def load_checkpoint(self, path):
        payload = torch.load(str(path), map_location=self.high_agent.agent_R.device, weights_only=False)
        if payload.get("checkpoint_format") != "hierarchical_d3qn_primary_v3":
            raise RuntimeError(
                "Checkpoint is incompatible with primary-only D3QN. "
                "Start a fresh run instead of loading a joint primary-backup checkpoint."
            )
        self.high_agent.agent_R.q_eval.load_state_dict(payload["high_q_eval_state_dict"])
        self.high_agent.agent_R.q_target.load_state_dict(payload["high_q_target_state_dict"])
        self.high_agent.agent_R.optimizer.load_state_dict(payload["high_optimizer_state_dict"])
        if payload.get("high_replay_state_dict") is not None:
            try:
                high_replay_state = payload["high_replay_state_dict"]
                if self.high_agent.deadline_lagrangian_enabled and (
                    "base_energy_rewards" not in high_replay_state
                    or "deadline_constraint_costs" not in high_replay_state
                ):
                    raise ValueError(
                        "Old scalar-reward replay cannot resume Lagrangian "
                        "training; load network weights with a fresh replay."
                    )
                self.high_agent.replay.load_state_dict(high_replay_state)
            except ValueError:
                # Evaluation only needs network weights; replay shape can differ after config changes.
                if bool(payload.get("strict_replay_load", False)):
                    raise
        self.high_agent.epsilon = float(payload.get("high_epsilon", self.high_agent.epsilon))
        if self.low_agent is not None and payload.get("low_model_state_dict") is not None:
            if self.low_algorithm == "sac":
                low_state = copy.deepcopy(payload["low_model_state_dict"])
                if self.shared_deadline_lagrangian_enabled:
                    # A legacy checkpoint may contain a separate low lambda.
                    # Never let it overwrite the configured shared controller.
                    low_state["deadline_lagrangian"] = None
                self.low_agent.load_state_dict(low_state)
            else:
                self.low_agent.network.load_state_dict(payload["low_model_state_dict"])
                self.low_agent.optimizer.load_state_dict(payload["low_optimizer_state_dict"])
        self._load_lagrangian_checkpoint_state(payload)
        if (
            self.coordination_critic is not None
            and payload.get("coordination_critic_state_dict") is not None
        ):
            self.coordination_critic.load_state_dict(
                payload["coordination_critic_state_dict"]
            )
        if payload.get("numpy_random_state") is not None:
            np.random.set_state(payload["numpy_random_state"])
        if payload.get("torch_random_state") is not None:
            torch.random.set_rng_state(as_torch_rng_state(payload["torch_random_state"]))
        if torch.cuda.is_available() and payload.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all([as_torch_rng_state(state) for state in payload["cuda_random_state"]])
        return payload

    def load_high_checkpoint(self, path, *, freeze=False):
        """Load only the D3QN server-selection policy.

        The low-level agent, replay buffers, optimizers, and random-number states are
        intentionally left untouched so a fresh SAC policy can be trained under a
        fixed, deterministic upper-level policy.
        """
        payload = torch.load(
            str(path), map_location=self.high_agent.agent_R.device, weights_only=False
        )
        checkpoint_format = payload.get("checkpoint_format")
        supported = {
            "hierarchical_d3qn_primary_v3",
            "hierarchical_d3qn_high_primary_v2",
        }
        if checkpoint_format not in supported:
            raise RuntimeError(
                "High-level initialization requires a primary-only D3QN full/high "
                f"checkpoint; got checkpoint_format={checkpoint_format!r}."
            )
        checkpoint_action_dim = payload.get("high_action_dim")
        if (
            checkpoint_action_dim is not None
            and int(checkpoint_action_dim) != int(self.high_agent.action_dim)
        ):
            raise RuntimeError(
                "High checkpoint action dimension mismatch: "
                f"checkpoint={int(checkpoint_action_dim)}, "
                f"current={int(self.high_agent.action_dim)}."
            )
        try:
            self.high_agent.agent_R.q_eval.load_state_dict(
                payload["high_q_eval_state_dict"]
            )
            self.high_agent.agent_R.q_target.load_state_dict(
                payload.get(
                    "high_q_target_state_dict", payload["high_q_eval_state_dict"]
                )
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "High checkpoint network/state dimension is incompatible with the "
                "current high-level state encoder."
            ) from exc
        self.high_agent.epsilon = float(
            payload.get("high_epsilon", self.high_agent.epsilon)
        )
        self._load_lagrangian_checkpoint_state(payload)
        if freeze:
            for network in (
                self.high_agent.agent_R.q_eval,
                self.high_agent.agent_R.q_target,
            ):
                network.eval()
                for parameter in network.parameters():
                    parameter.requires_grad_(False)
        return payload

    def _low_model_state_dict(self, include_replay=False):
        if self.low_agent is None:
            return None
        if self.low_algorithm == "sac":
            payload = self.low_agent.state_dict(include_replay=include_replay)
            if self.shared_deadline_lagrangian_enabled:
                payload["deadline_lagrangian"] = None
            return payload
        return self.low_agent.network.state_dict()

    def _shared_lagrangian_state_dict(self):
        if self.shared_deadline_lagrangian is None:
            return None
        return self.shared_deadline_lagrangian.state_dict()

    def _load_lagrangian_checkpoint_state(self, payload):
        if self.shared_deadline_lagrangian_enabled:
            shared_state = payload.get("shared_deadline_lagrangian")
            if shared_state is not None:
                self.shared_deadline_lagrangian.load_state_dict(shared_state)
            # Separate-lambda checkpoints intentionally keep the new shared
            # controller at its configured initial value.
            self._validate_shared_deadline_lagrangian()
            return
        if (
            self.high_agent.deadline_lagrangian is not None
            and payload.get("deadline_lagrangian") is not None
        ):
            self.high_agent.deadline_lagrangian.load_state_dict(
                payload["deadline_lagrangian"]
            )
