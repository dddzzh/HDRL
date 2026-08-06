from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from src.D3QN.D3QN import D3QN
from src.D3QN.state import HierarchicalStateEncoder
from src.D3QN.utils import as_torch_rng_state
from src.ppo.hierarchical_agent import LowLevelDVFSPPOAgent
from src.sac.hierarchical_low_sac import LowLevelDVFSContinuousSACAgent

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class HighLevelD3QNAgent:
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
        self.encoder = HierarchicalStateEncoder(state_cfg)
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
        self.deadline_cost_reward_weight = float(config.get("deadline_cost_reward_weight", 0.0))
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

    def select_assignment(self, env_state, deterministic=False, selector_mode="policy"):
        if selector_mode != "policy":
            raise ValueError("Only selector_mode='policy' is supported for D3QN training.")
        observation = self.build_high_state(env_state)
        context = self._require_context(env_state)
        primary_mask, backup_pair_mask = self._context_masks(context)
        action = self.agent_R.choose_action(
            observation.vector,
            primary_mask,
            isTrain=not bool(deterministic),
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
        q_value = self.agent_R.action_q_value(observation.vector, action)
        return self._decision_payload(
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
            deadline_limit = max(float(buffer.deadline_limits[index]), 1e-8)
            normalized_deadline_cost = float(buffer.deadline_costs[index]) / deadline_limit
            deadline_reward_penalty = self.deadline_cost_reward_weight * normalized_deadline_cost
            replay_reward = float(buffer.rewards[index]) - deadline_reward_penalty
            self.agent_R.remember(
                buffer.states[index],
                action,
                replay_reward,
                next_state,
                done,
                action_mask,
                next_mask,
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
        for _ in range(self.training_per_episode):
            metrics = self.agent_R.learn()
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
        # Do not filter real servers by deadline/reliability. False entries are
        # reserved only for padded, non-existent server indices.
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
    def __init__(self, config=None, seed=42):
        config = dict(config or {})
        self.config = copy.deepcopy(config)
        self.seed = int(seed)
        shared = dict(config.get("shared", {}))
        high_cfg = {**shared, **dict(config.get("high", {}))}
        low_cfg = {**shared, **dict(config.get("low", {}))}
        self.use_low_agent = bool(config.get("use_low_agent", True))
        self.low_algorithm = str(config.get("low_algorithm", low_cfg.get("algorithm", "ppo"))).lower()
        self.high_agent = HighLevelD3QNAgent(high_cfg, seed=self.seed)
        if not self.use_low_agent:
            self.low_agent = None
        elif self.low_algorithm == "sac":
            self.low_agent = LowLevelDVFSContinuousSACAgent(low_cfg, seed=self.seed)
        elif self.low_algorithm == "ppo":
            self.low_agent = LowLevelDVFSPPOAgent(low_cfg, seed=self.seed)
        else:
            raise ValueError(f"Unsupported D3QN low_algorithm: {self.low_algorithm}")
        self._validate_encoder_config()

    def _validate_encoder_config(self):
        if self.low_agent is None:
            return
        for field in ("max_ready_tasks", "max_servers"):
            if getattr(self.high_agent.encoder, field) != getattr(self.low_agent.encoder, field):
                raise ValueError(f"High/low encoder config mismatch for {field}.")

    def select_action(self, env_state, deterministic=False, low_deterministic=None, high_selector_mode="policy"):
        if low_deterministic is None:
            low_deterministic = deterministic
        high_decision = self.high_agent.select_assignment(
            env_state,
            deterministic=deterministic,
            selector_mode=high_selector_mode,
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
            "use_low_agent": bool(self.use_low_agent),
            "low_algorithm": self.low_algorithm,
            "low_model_state_dict": self._low_model_state_dict(include_replay=include_replay),
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
            **common,
        }
        torch.save(high_payload, str(self._component_checkpoint_path(path, "high")))
        if self.low_agent is None:
            return
        if self.low_algorithm == "sac":
            low_payload = {
                "checkpoint_format": "hierarchical_d3qn_low_sac_v1",
                "low_algorithm": "sac",
                "low_sac_state_dict": self.low_agent.state_dict(include_replay=False),
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
                self.high_agent.replay.load_state_dict(payload["high_replay_state_dict"])
            except ValueError:
                # Evaluation only needs network weights; replay shape can differ after config changes.
                if bool(payload.get("strict_replay_load", False)):
                    raise
        self.high_agent.epsilon = float(payload.get("high_epsilon", self.high_agent.epsilon))
        if self.low_agent is not None and payload.get("low_model_state_dict") is not None:
            if self.low_algorithm == "sac":
                self.low_agent.load_state_dict(payload["low_model_state_dict"])
            else:
                self.low_agent.network.load_state_dict(payload["low_model_state_dict"])
                self.low_agent.optimizer.load_state_dict(payload["low_optimizer_state_dict"])
        if payload.get("numpy_random_state") is not None:
            np.random.set_state(payload["numpy_random_state"])
        if payload.get("torch_random_state") is not None:
            torch.random.set_rng_state(as_torch_rng_state(payload["torch_random_state"]))
        if torch.cuda.is_available() and payload.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all([as_torch_rng_state(state) for state in payload["cuda_random_state"]])
        return payload

    def _low_model_state_dict(self, include_replay=False):
        if self.low_agent is None:
            return None
        if self.low_algorithm == "sac":
            return self.low_agent.state_dict(include_replay=include_replay)
        return self.low_agent.network.state_dict()
