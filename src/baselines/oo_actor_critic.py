from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from src.a2c.a2c import _require_torch
from src.device_utils import resolve_agent_device
from src.shared.hierarchical_state_reward import (
    SharedHierarchicalReward,
    compose_fixed_lambda_upper_reward,
    compose_upper_deadline_energy_reward,
    compute_dvfs_reward_components,
    low_reliability_counterfactual_cost,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = object
    F = None


EPS = 1e-8


def _build_mlp(input_dim: int, hidden_dim: int, hidden_layers: int):
    layers = []
    current_dim = int(input_dim)
    for _ in range(int(hidden_layers)):
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
        current_dim = int(hidden_dim)
    return nn.Sequential(*layers), current_dim


def _bit_width(size: int) -> int:
    if int(size) <= 0:
        raise ValueError("Categorical action size must be positive.")
    return max(1, int(math.ceil(math.log2(int(size)))))


def _integer_to_bits(value: int, width: int) -> np.ndarray:
    value = int(value)
    width = int(width)
    if value < 0 or value >= 2**width:
        raise ValueError(f"Value {value} cannot be represented with {width} bits.")
    return np.fromiter(
        (1.0 if character == "1" else 0.0 for character in f"{value:0{width}b}"),
        dtype=np.float32,
        count=width,
    )


def _bits_to_integer(bits: Sequence[float]) -> int:
    result = 0
    for value in bits:
        result = (result << 1) | int(float(value) >= 0.5)
    return int(result)


class OOReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size=50000, batch_size=32):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        self.mem_count = 0
        self.states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.deadline_rewards = np.zeros(self.max_size, dtype=np.float32)
        self.energy_savings = np.zeros(self.max_size, dtype=np.float32)
        self.slack_time_costs = np.zeros(self.max_size, dtype=np.float32)
        self.deadline_violation_deltas = np.zeros(self.max_size, dtype=np.float32)
        self.reliability_violation_deltas = np.zeros(
            self.max_size, dtype=np.float32
        )
        self.next_states = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=bool)

    @property
    def size(self):
        return min(self.mem_count, self.max_size)

    def ready(self):
        return self.size >= self.batch_size

    def store(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        *,
        deadline_reward=None,
        energy_saving=0.0,
        slack_time_cost=0.0,
        deadline_violation_delta=0.0,
        reliability_violation_delta=0.0,
    ):
        index = self.mem_count % self.max_size
        self.states[index] = np.asarray(state, dtype=np.float32)
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (self.action_dim,):
            raise ValueError(
                f"OO action bits must have shape {(self.action_dim,)}, got {action_array.shape}."
            )
        if not np.isin(action_array, (0.0, 1.0)).all():
            raise ValueError("OO replay actions must be binary.")
        self.actions[index] = action_array
        self.rewards[index] = float(reward)
        self.deadline_rewards[index] = float(
            reward if deadline_reward is None else deadline_reward
        )
        self.energy_savings[index] = float(energy_saving)
        self.slack_time_costs[index] = float(slack_time_cost)
        self.deadline_violation_deltas[index] = float(deadline_violation_delta)
        self.reliability_violation_deltas[index] = float(
            reliability_violation_delta
        )
        self.next_states[index] = np.asarray(next_state, dtype=np.float32)
        self.dones[index] = bool(done)
        self.mem_count += 1

    def sample(self, rng):
        if not self.ready():
            return None
        indices = rng.choice(self.size, self.batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.deadline_rewards[indices],
            self.energy_savings[indices],
            self.slack_time_costs[indices],
            self.deadline_violation_deltas[indices],
            self.reliability_violation_deltas[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def energy_statistics(self, std_floor=0.05):
        return self.component_statistics(self.energy_savings, std_floor)

    def slack_time_statistics(self, std_floor=0.005):
        return self.component_statistics(self.slack_time_costs, std_floor)

    def component_statistics(self, source, std_floor):
        values = source[: self.size].astype(np.float64, copy=False)
        if values.size < 2:
            return {
                "count": int(values.size),
                "mean": 0.0,
                "std": max(float(std_floor), EPS),
            }
        return {
            "count": int(values.size),
            "mean": float(values.mean(dtype=np.float64)),
            "std": float(max(values.std(dtype=np.float64), float(std_floor))),
        }

    def state_dict(self):
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_size": self.max_size,
            "batch_size": self.batch_size,
            "mem_count": self.mem_count,
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "deadline_rewards": self.deadline_rewards,
            "energy_savings": self.energy_savings,
            "slack_time_costs": self.slack_time_costs,
            "deadline_violation_deltas": self.deadline_violation_deltas,
            "reliability_violation_deltas": self.reliability_violation_deltas,
            "next_states": self.next_states,
            "dones": self.dones,
        }

    def load_state_dict(self, payload):
        if int(payload["state_dim"]) != self.state_dim:
            raise ValueError("OO replay state dimension mismatch.")
        if int(payload["action_dim"]) != self.action_dim:
            raise ValueError("OO replay action dimension mismatch.")
        self.mem_count = int(payload["mem_count"])
        for name in ("states", "actions", "rewards", "next_states", "dones"):
            target = getattr(self, name)
            source = np.asarray(payload[name])
            if target.shape != source.shape:
                raise ValueError(f"OO replay {name} shape mismatch: {source.shape} != {target.shape}.")
            target[:] = source
        for name, fallback in (
            ("deadline_rewards", self.rewards),
            ("energy_savings", None),
            ("slack_time_costs", None),
            ("deadline_violation_deltas", None),
            ("reliability_violation_deltas", None),
        ):
            target = getattr(self, name)
            if name not in payload:
                target[:] = fallback if fallback is not None else 0.0
                continue
            source = np.asarray(payload[name])
            if target.shape != source.shape:
                raise ValueError(
                    f"OO replay {name} shape mismatch: {source.shape} != {target.shape}."
                )
            target[:] = source


class BitActor(nn.Module):
    """Actor from paper Eq. (23): Bernoulli probabilities for action bits."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim=256, hidden_layers=2):
        super().__init__()
        self.backbone, output_dim = _build_mlp(state_dim, hidden_dim, hidden_layers)
        self.output = nn.Linear(output_dim, action_dim)

    def forward(self, states):
        return torch.sigmoid(self.output(self.backbone(states)))


class StateActionCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim=256, hidden_layers=2):
        super().__init__()
        self.backbone, output_dim = _build_mlp(
            state_dim + action_dim, hidden_dim, hidden_layers
        )
        self.output = nn.Linear(output_dim, 1)

    def forward(self, states, actions):
        features = torch.cat((states, actions), dim=-1)
        return self.output(self.backbone(features)).squeeze(-1)


class OOActorCriticAgent:
    """Paper-adapted actor-critic with ordinal-optimization action search.

    The paper defines a binary joint action ``[server index, VM level]``. Its
    Eq. (23) trains the actor with binary cross entropy against the selected
    action, rather than the deterministic-policy-gradient loss used by a
    standard DDPG implementation. This class follows the published equations.
    """

    def __init__(self, state_dim: int, num_servers: int, rho_levels, config=None, seed=42):
        _require_torch()
        config = dict(config or {})
        self.config = copy.deepcopy(config)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.state_dim = int(state_dim)
        self.num_servers = int(num_servers)
        self.rho_levels = tuple(float(value) for value in rho_levels)
        if self.num_servers <= 0:
            raise ValueError("OO comparison requires at least one server.")
        if not self.rho_levels:
            raise ValueError("OO comparison requires at least one frequency level.")
        if any(not 0.0 < value <= 1.0 for value in self.rho_levels):
            raise ValueError("OO rho levels must be in (0, 1].")
        if tuple(sorted(set(self.rho_levels))) != self.rho_levels:
            raise ValueError("OO rho levels must be sorted and unique.")

        self.server_bits = _bit_width(self.num_servers)
        self.frequency_bits = _bit_width(len(self.rho_levels))
        self.action_dim = self.server_bits + self.frequency_bits
        self.num_candidates = int(config.get("num_candidates", 100))
        self.top_s = int(config.get("top_s", 10))
        # The paper says "random bit perturbations" but does not report a
        # probability. Keep this reproducibility choice explicit in config.
        self.bit_flip_probability = float(config.get("bit_flip_probability", 0.2))
        self.random_warmup_transitions = int(
            config.get("random_warmup_transitions", 0)
        )
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.target_update_interval = int(config.get("target_update_interval", 3))
        self.learning_rate = float(config.get("learning_rate", 5e-4))
        self.training_per_episode = int(config.get("training_per_episode", 10))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.reward_mode = str(
            config.get("reward_mode", "legacy_deadline_energy")
        ).strip().lower()
        if self.reward_mode not in {
            "legacy_deadline_energy",
            "shared_hierarchical_v1",
            "shared_upper_same_server_energy_v2",
            "fixed_lambda_upper_v1",
            "hierarchical_fixed_lambda_v2",
        }:
            raise ValueError(f"Unsupported OO reward_mode: {self.reward_mode!r}.")
        self.shared_reward = (
            SharedHierarchicalReward(config)
            if self.reward_mode in {
                "shared_hierarchical_v1",
                "shared_upper_same_server_energy_v2",
                "hierarchical_fixed_lambda_v2",
            }
            else None
        )
        self.energy_reward_weight = float(
            config.get(
                "energy_reward_weight",
                0.0
                if self.shared_reward is None
                else self.shared_reward.joint_low_reward_weight,
            )
        )
        self.energy_std_floor = float(
            config.get(
                "energy_std_floor",
                0.05 if self.shared_reward is None else self.shared_reward.energy_std_floor,
            )
        )
        self.slack_time_std_floor = float(
            config.get(
                "slack_time_std_floor",
                0.005
                if self.shared_reward is None
                else self.shared_reward.slack_time_std_floor,
            )
        )
        if self.energy_reward_weight < 0.0:
            raise ValueError("OO energy_reward_weight must be non-negative.")
        if self.energy_std_floor <= 0.0:
            raise ValueError("OO energy_std_floor must be positive.")
        if self.slack_time_std_floor <= 0.0:
            raise ValueError("OO slack_time_std_floor must be positive.")
        self._checkpoint_energy_statistics = None
        self._checkpoint_reward_statistics = None
        if self.num_candidates <= 0 or self.top_s <= 0:
            raise ValueError("OO num_candidates and top_s must be positive.")
        if self.random_warmup_transitions < 0:
            raise ValueError("OO random_warmup_transitions must be non-negative.")
        if self.top_s > self.num_candidates:
            raise ValueError("OO top_s cannot exceed num_candidates.")
        if not 0.0 < self.bit_flip_probability < 1.0:
            raise ValueError("OO bit_flip_probability must be in (0, 1).")
        if self.target_update_interval <= 0:
            raise ValueError("OO target_update_interval must be positive.")

        hidden_dim = int(config.get("hidden_dim", 256))
        hidden_layers = int(config.get("hidden_layers", 2))
        self.device = resolve_agent_device(config.get("device", "auto"))
        self.actor = BitActor(
            self.state_dim, self.action_dim, hidden_dim, hidden_layers
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.critic = StateActionCritic(
            self.state_dim, self.action_dim, hidden_dim, hidden_layers
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.replay = OOReplayBuffer(
            self.state_dim,
            self.action_dim,
            max_size=int(config.get("replay_size", 50000)),
            batch_size=int(config.get("batch_size", 32)),
        )
        self.update_steps = 0

        all_bits = []
        all_pairs = []
        for server_index in range(self.num_servers):
            for frequency_index in range(len(self.rho_levels)):
                all_pairs.append((server_index, frequency_index))
                all_bits.append(self.encode_action(server_index, frequency_index))
        self._all_pairs = np.asarray(all_pairs, dtype=np.int64)
        self._all_bits = np.asarray(all_bits, dtype=np.float32)

    def encode_action(self, server_index: int, frequency_index: int) -> np.ndarray:
        server_index = int(server_index)
        frequency_index = int(frequency_index)
        if not 0 <= server_index < self.num_servers:
            raise ValueError(f"OO server index out of range: {server_index}.")
        if not 0 <= frequency_index < len(self.rho_levels):
            raise ValueError(f"OO frequency index out of range: {frequency_index}.")
        return np.concatenate(
            (
                _integer_to_bits(server_index, self.server_bits),
                _integer_to_bits(frequency_index, self.frequency_bits),
            )
        ).astype(np.float32)

    def decode_action(self, action_bits) -> tuple[int, int] | None:
        bits = np.asarray(action_bits, dtype=np.float32)
        if bits.shape != (self.action_dim,):
            raise ValueError(f"OO action bits must have shape {(self.action_dim,)}.")
        server_index = _bits_to_integer(bits[: self.server_bits])
        frequency_index = _bits_to_integer(bits[self.server_bits :])
        if server_index >= self.num_servers or frequency_index >= len(self.rho_levels):
            return None
        return int(server_index), int(frequency_index)

    def select_action(
        self,
        state,
        valid_server_mask,
        exact_score_fn: Callable[[int, int], float],
        *,
        deterministic=False,
    ):
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.shape != (self.state_dim,):
            raise ValueError(f"OO state must have shape {(self.state_dim,)}, got {state_array.shape}.")
        valid_mask = np.asarray(valid_server_mask, dtype=bool)
        if valid_mask.shape != (self.num_servers,):
            raise ValueError(
                f"OO valid_server_mask must have shape {(self.num_servers,)}, got {valid_mask.shape}."
            )
        if not valid_mask.any():
            raise ValueError("OO action selection has no structurally valid server.")

        state_tensor = torch.as_tensor(
            state_array, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            probabilities = self.actor(state_tensor).squeeze(0).cpu().numpy()
        preliminary_bits = (probabilities >= 0.5).astype(np.float32)
        if (
            not deterministic
            and self.replay.size < self.random_warmup_transitions
        ):
            valid_indices = np.flatnonzero(
                valid_mask[self._all_pairs[:, 0]]
            )
            selected_index = int(self.rng.choice(valid_indices))
            server_index, frequency_index = self._all_pairs[selected_index]
            selected_bits = self._all_bits[selected_index].copy()
            action_tensor = torch.as_tensor(
                selected_bits, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                q_value = float(self.critic(state_tensor, action_tensor).item())
            exact_score = float(
                exact_score_fn(int(server_index), int(frequency_index))
            )
            return {
                "state": state_array,
                "action_bits": selected_bits,
                "server_index": int(server_index),
                "frequency_index": int(frequency_index),
                "rho": float(self.rho_levels[int(frequency_index)]),
                "q_value": q_value,
                "exact_score": exact_score,
                "actor_probabilities": probabilities.astype(np.float32),
                "preliminary_action_bits": preliminary_bits,
                "preliminary_projected_action": self._all_pairs[
                    selected_index
                ].copy(),
                "candidate_count": int(valid_indices.size),
                "top_s_count": 0,
                "selected_hamming_distance": int(
                    np.abs(selected_bits - preliminary_bits).sum()
                ),
                "random_warmup": True,
            }
        candidate_indices, projected_index = self._candidate_indices(
            preliminary_bits,
            probabilities,
            valid_mask,
            deterministic=bool(deterministic),
        )
        candidate_bits = self._all_bits[candidate_indices]
        states_tensor = state_tensor.expand(candidate_bits.shape[0], -1)
        actions_tensor = torch.as_tensor(
            candidate_bits, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            q_values = self.critic(states_tensor, actions_tensor).cpu().numpy()
        top_count = min(self.top_s, candidate_indices.size)
        top_positions = np.argsort(-q_values, kind="stable")[:top_count]

        finalists = []
        for position in top_positions:
            global_index = int(candidate_indices[int(position)])
            server_index, frequency_index = self._all_pairs[global_index]
            exact_score = float(exact_score_fn(int(server_index), int(frequency_index)))
            if not math.isfinite(exact_score):
                raise FloatingPointError("OO exact candidate score is NaN or Inf.")
            finalists.append(
                (
                    exact_score,
                    float(q_values[int(position)]),
                    float(self.rho_levels[int(frequency_index)]),
                    -int(global_index),
                    int(global_index),
                )
            )
        _, q_value, _, _, selected_index = max(finalists)
        server_index, frequency_index = self._all_pairs[selected_index]
        selected_bits = self._all_bits[selected_index].copy()
        selected_score = max(finalists)[0]
        return {
            "state": state_array,
            "action_bits": selected_bits,
            "server_index": int(server_index),
            "frequency_index": int(frequency_index),
            "rho": float(self.rho_levels[int(frequency_index)]),
            "q_value": float(q_value),
            "exact_score": float(selected_score),
            "actor_probabilities": probabilities.astype(np.float32),
            "preliminary_action_bits": preliminary_bits,
            "preliminary_projected_action": self._all_pairs[projected_index].copy(),
            "candidate_count": int(candidate_indices.size),
            "top_s_count": int(top_count),
            "selected_hamming_distance": int(
                np.abs(selected_bits - preliminary_bits).sum()
            ),
            "random_warmup": False,
        }

    def _candidate_indices(
        self,
        preliminary_bits,
        probabilities,
        valid_server_mask,
        *,
        deterministic,
    ):
        valid_indices = np.flatnonzero(valid_server_mask[self._all_pairs[:, 0]])
        valid_bits = self._all_bits[valid_indices]
        hamming = np.abs(valid_bits - preliminary_bits[None, :]).sum(axis=1)
        probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1.0 - EPS)
        log_likelihood = (
            valid_bits * np.log(probabilities)[None, :]
            + (1.0 - valid_bits) * np.log(1.0 - probabilities)[None, :]
        ).sum(axis=1)
        projection_order = np.lexsort((valid_indices, -log_likelihood, hamming))
        projected_index = int(valid_indices[int(projection_order[0])])
        target_count = min(self.num_candidates, valid_indices.size)
        if target_count == valid_indices.size:
            return valid_indices.astype(np.int64), projected_index
        if deterministic:
            chosen = valid_indices[projection_order[:target_count]]
            return chosen.astype(np.int64), projected_index

        chosen = {projected_index}
        attempts = 0
        max_attempts = max(100, target_count * 50)
        while len(chosen) < target_count and attempts < max_attempts:
            flip_mask = self.rng.random(self.action_dim) < self.bit_flip_probability
            if not bool(flip_mask.any()):
                flip_mask[int(self.rng.integers(self.action_dim))] = True
            perturbed = np.logical_xor(preliminary_bits.astype(bool), flip_mask).astype(
                np.float32
            )
            pair = self.decode_action(perturbed)
            if pair is not None:
                server_index, frequency_index = pair
                if bool(valid_server_mask[server_index]):
                    chosen.add(server_index * len(self.rho_levels) + frequency_index)
            attempts += 1

        if len(chosen) < target_count:
            remaining = np.asarray(
                [index for index in valid_indices if int(index) not in chosen],
                dtype=np.int64,
            )
            remaining_bits = self._all_bits[remaining]
            distances = np.abs(remaining_bits - preliminary_bits[None, :]).sum(axis=1)
            p = self.bit_flip_probability
            weights = np.power(p, distances) * np.power(1.0 - p, self.action_dim - distances)
            weights = np.maximum(weights, EPS)
            weights = weights / weights.sum()
            needed = min(target_count - len(chosen), remaining.size)
            sampled = self.rng.choice(remaining, size=needed, replace=False, p=weights)
            chosen.update(int(index) for index in sampled)
        return np.asarray(sorted(chosen), dtype=np.int64), projected_index

    def energy_statistics(self):
        statistics = self.replay.energy_statistics(self.energy_std_floor)
        if statistics["count"] >= 2:
            return statistics
        if self._checkpoint_energy_statistics is not None:
            return dict(self._checkpoint_energy_statistics)
        return statistics

    def energy_zscore(self, energy_saving, statistics=None):
        statistics = dict(statistics or self.energy_statistics())
        if int(statistics.get("count", 0)) < 2:
            return 0.0
        return (
            float(energy_saving) - float(statistics["mean"])
        ) / max(float(statistics["std"]), self.energy_std_floor)

    def reward_statistics(self):
        statistics = {
            "energy": self.replay.energy_statistics(self.energy_std_floor),
            "slack_time": self.replay.slack_time_statistics(
                self.slack_time_std_floor
            ),
        }
        if (
            statistics["energy"]["count"] >= 2
            and statistics["slack_time"]["count"] >= 2
        ):
            return statistics
        if self._checkpoint_reward_statistics is not None:
            result = copy.deepcopy(self._checkpoint_reward_statistics)
            for name in ("energy", "slack_time"):
                if statistics[name]["count"] >= 2:
                    result[name] = statistics[name]
            return result
        return statistics

    def compose_reward(
        self,
        deadline_reward,
        energy_saving,
        statistics=None,
        *,
        slack_time_cost=0.0,
        deadline_violation_delta=0.0,
        reliability_violation_delta=0.0,
    ):
        if self.reward_mode == "shared_upper_same_server_energy_v2":
            rewards = deadline_rewards.astype(np.float32, copy=True)
            statistics = self.energy_statistics()
            slack_statistics = self.reward_statistics()["slack_time"]
        elif self.shared_reward is not None:
            result = self.shared_reward.compose(
                deadline_reward,
                energy_saving=energy_saving,
                slack_time_cost=slack_time_cost,
                deadline_violation_delta=deadline_violation_delta,
                reliability_violation_delta=reliability_violation_delta,
                statistics=statistics or self.reward_statistics(),
            )
            return {**result, "deadline_reward": float(deadline_reward)}
        energy_zscore = self.energy_zscore(energy_saving, statistics)
        energy_reward = self.energy_reward_weight * energy_zscore
        return {
            "reward": float(deadline_reward) + float(energy_reward),
            "deadline_reward": float(deadline_reward),
            "energy_saving": float(energy_saving),
            "energy_zscore": float(energy_zscore),
            "energy_reward": float(energy_reward),
        }

    def remember_from_high_buffer(self, high_buffer, deadline_cost_reward_weight=1.0):
        if high_buffer.size == 0:
            return {"stored_transitions": 0.0, "deadline_reward_penalty_total": 0.0}
        zero_state = np.zeros(self.state_dim, dtype=np.float32)
        stored = 0
        penalty_total = 0.0
        for index in range(high_buffer.size):
            info = high_buffer.infos[index]
            action_bits = np.asarray(info["oo_action_bits"], dtype=np.float32)
            if "oo_energy_saving_joules" in info:
                deadline_penalty = 0.0
                replay_reward = float(high_buffer.rewards[index])
                if info.get("oo_reward_mode") in {
                    "shared_upper_same_server_energy_v2",
                    "fixed_lambda_upper_v1",
                }:
                    # This mode has no replay-dependent normalization, so the
                    # executed full scalar reward is already the training target.
                    deadline_reward = replay_reward
                    slack_time_cost = 0.0
                    deadline_violation_delta = 0.0
                    reliability_violation_delta = 0.0
                elif info.get("oo_reward_mode") in {
                    "shared_hierarchical_v1",
                    "hierarchical_fixed_lambda_v2",
                }:
                    deadline_reward = replay_reward - float(
                        info.get("oo_weighted_low_reward", 0.0)
                    )
                    slack_time_cost = float(info.get("oo_slack_time_cost", 0.0))
                    deadline_violation_delta = float(
                        info.get("oo_deadline_violation_delta", 0.0)
                    )
                    reliability_violation_delta = float(
                        info.get("oo_reliability_violation_delta", 0.0)
                    )
                else:
                    deadline_reward = replay_reward - float(
                        info.get("oo_energy_reward", 0.0)
                    )
                    slack_time_cost = 0.0
                    deadline_violation_delta = 0.0
                    reliability_violation_delta = 0.0
                energy_saving = float(info["oo_energy_saving_joules"])
            else:
                deadline_limit = max(float(high_buffer.deadline_limits[index]), EPS)
                deadline_penalty = (
                    float(deadline_cost_reward_weight)
                    * float(high_buffer.deadline_costs[index])
                    / deadline_limit
                )
                replay_reward = float(high_buffer.rewards[index]) - deadline_penalty
                deadline_reward = replay_reward
                energy_saving = 0.0
                slack_time_cost = 0.0
                deadline_violation_delta = 0.0
                reliability_violation_delta = 0.0
            done = index == high_buffer.size - 1
            next_state = zero_state if done else high_buffer.states[index + 1]
            self.replay.store(
                high_buffer.states[index],
                action_bits,
                replay_reward,
                next_state,
                done,
                deadline_reward=deadline_reward,
                energy_saving=energy_saving,
                slack_time_cost=slack_time_cost,
                deadline_violation_delta=deadline_violation_delta,
                reliability_violation_delta=reliability_violation_delta,
            )
            penalty_total += deadline_penalty
            stored += 1
        energy_statistics = self.energy_statistics()
        reward_statistics = self.reward_statistics()
        return {
            "stored_transitions": float(stored),
            "deadline_reward_penalty_total": float(penalty_total),
            "energy_replay_mean": float(energy_statistics["mean"]),
            "energy_replay_std": float(energy_statistics["std"]),
            "slack_time_replay_mean": float(
                reward_statistics["slack_time"]["mean"]
            ),
            "slack_time_replay_std": float(
                reward_statistics["slack_time"]["std"]
            ),
        }

    def update(self):
        totals = {"replay_size": float(self.replay.size)}
        updates = 0
        for _ in range(self.training_per_episode):
            metrics = self.learn()
            if not metrics:
                continue
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            updates += 1
        if updates:
            for key in tuple(totals):
                if key != "replay_size":
                    totals[key] /= updates
        totals["updates"] = float(updates)
        totals["replay_size"] = float(self.replay.size)
        return totals

    def learn(self):
        sample = self.replay.sample(self.rng)
        if sample is None:
            return {}
        (
            states,
            actions,
            deadline_rewards,
            energy_savings,
            slack_time_costs,
            deadline_violation_deltas,
            reliability_violation_deltas,
            next_states,
            dones,
        ) = sample
        if self.reward_mode == "fixed_lambda_upper_v1":
            # The full, already-scaled scalar reward is stored in the
            # deadline_reward replay field for this non-normalized mode.
            rewards = deadline_rewards.astype(np.float32, copy=True)
            statistics = self.energy_statistics()
            slack_statistics = {"mean": 0.0, "std": self.slack_time_std_floor}
        elif self.shared_reward is not None:
            reward_statistics = self.reward_statistics()
            rewards = np.asarray(
                [
                    self.shared_reward.compose(
                        deadline_reward,
                        energy_saving=energy_saving,
                        slack_time_cost=slack_time_cost,
                        deadline_violation_delta=deadline_violation_delta,
                        reliability_violation_delta=reliability_violation_delta,
                        statistics=reward_statistics,
                    )["reward"]
                    for deadline_reward, energy_saving, slack_time_cost, deadline_violation_delta, reliability_violation_delta in zip(
                        deadline_rewards,
                        energy_savings,
                        slack_time_costs,
                        deadline_violation_deltas,
                        reliability_violation_deltas,
                    )
                ],
                dtype=np.float32,
            )
            statistics = reward_statistics["energy"]
            slack_statistics = reward_statistics["slack_time"]
        else:
            statistics = self.energy_statistics()
            if int(statistics["count"]) >= 2:
                normalized_energy = (
                    energy_savings.astype(np.float64) - float(statistics["mean"])
                ) / max(float(statistics["std"]), self.energy_std_floor)
            else:
                normalized_energy = np.zeros_like(energy_savings, dtype=np.float64)
            rewards = (
                deadline_rewards.astype(np.float64)
                + self.energy_reward_weight * normalized_energy
            ).astype(np.float32)
            slack_statistics = {"mean": 0.0, "std": self.slack_time_std_floor}
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            next_actions = self.target_actor(next_states_t)
            next_q = self.target_critic(next_states_t, next_actions)
            targets = rewards_t + self.gamma * (~dones_t).float() * next_q
        q_values = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(q_values, targets)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm
        )
        self.critic_optimizer.step()

        actor_probabilities = self.actor(states_t)
        actor_loss = F.binary_cross_entropy(actor_probabilities, actions_t)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.max_grad_norm
        )
        self.actor_optimizer.step()

        self.update_steps += 1
        target_updated = self.update_steps % self.target_update_interval == 0
        if target_updated:
            self.soft_update_targets()
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "q_mean": float(q_values.detach().mean().item()),
            "target_mean": float(targets.detach().mean().item()),
            "td_abs_mean": float((targets - q_values.detach()).abs().mean().item()),
            "actor_grad_norm_before_clip": float(actor_grad),
            "critic_grad_norm_before_clip": float(critic_grad),
            "actor_bit_mean": float(actor_probabilities.detach().mean().item()),
            "energy_replay_mean": float(statistics["mean"]),
            "energy_replay_std": float(statistics["std"]),
            "slack_time_replay_mean": float(slack_statistics["mean"]),
            "slack_time_replay_std": float(slack_statistics["std"]),
            "target_updated": float(target_updated),
        }

    def soft_update_targets(self, tau=None):
        tau = self.tau if tau is None else float(tau)
        with torch.no_grad():
            for target, source in zip(self.target_actor.parameters(), self.actor.parameters()):
                target.mul_(1.0 - tau).add_(source, alpha=tau)
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.mul_(1.0 - tau).add_(source, alpha=tau)

    def save_checkpoint(
        self,
        path,
        *,
        objective_state=None,
        episode=0,
        config=None,
        include_replay=False,
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_format": "adapted_oo_actor_critic_v1",
            "actor_state_dict": self.actor.state_dict(),
            "target_actor_state_dict": self.target_actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "target_critic_state_dict": self.target_critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "replay_state_dict": self.replay.state_dict() if include_replay else None,
            "replay_saved": bool(include_replay),
            "update_steps": int(self.update_steps),
            "state_dim": self.state_dim,
            "num_servers": self.num_servers,
            "rho_levels": self.rho_levels,
            "server_bits": self.server_bits,
            "frequency_bits": self.frequency_bits,
            "reward_mode": self.reward_mode,
            "objective_state": dict(objective_state or {}),
            "energy_reward_statistics": self.energy_statistics(),
            "shared_reward_statistics": self.reward_statistics(),
            "episode": int(episode),
            "config": copy.deepcopy(config if config is not None else self.config),
            "numpy_rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "torch_random_state": torch.random.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path, *, load_replay=False):
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload.get("checkpoint_format") != "adapted_oo_actor_critic_v1":
            raise ValueError("Unsupported OO checkpoint format.")
        if int(payload["state_dim"]) != self.state_dim:
            raise ValueError("OO checkpoint state dimension mismatch.")
        if int(payload["num_servers"]) != self.num_servers:
            raise ValueError("OO checkpoint server count mismatch.")
        if tuple(float(value) for value in payload["rho_levels"]) != self.rho_levels:
            raise ValueError("OO checkpoint rho levels mismatch.")
        saved_reward_mode = str(
            payload.get("reward_mode", "legacy_deadline_energy")
        ).strip().lower()
        if saved_reward_mode != self.reward_mode:
            raise ValueError(
                "OO checkpoint reward mode mismatch: "
                f"checkpoint={saved_reward_mode}, current={self.reward_mode}."
            )
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.target_actor.load_state_dict(payload["target_actor_state_dict"])
        self.critic.load_state_dict(payload["critic_state_dict"])
        self.target_critic.load_state_dict(payload["target_critic_state_dict"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        self.update_steps = int(payload.get("update_steps", 0))
        if load_replay and payload.get("replay_state_dict") is not None:
            self.replay.load_state_dict(payload["replay_state_dict"])
        saved_energy_statistics = payload.get("energy_reward_statistics")
        if saved_energy_statistics is not None:
            self._checkpoint_energy_statistics = {
                "count": int(saved_energy_statistics.get("count", 0)),
                "mean": float(saved_energy_statistics.get("mean", 0.0)),
                "std": max(
                    float(saved_energy_statistics.get("std", self.energy_std_floor)),
                    self.energy_std_floor,
                ),
            }
        saved_shared_statistics = payload.get("shared_reward_statistics")
        if saved_shared_statistics is not None:
            self._checkpoint_reward_statistics = copy.deepcopy(
                saved_shared_statistics
            )
        if payload.get("numpy_rng_state") is not None:
            self.rng.bit_generator.state = payload["numpy_rng_state"]
        return payload


def evaluate_upper_reward_candidate(
    env,
    objective,
    state: Mapping,
    *,
    server_id: int,
    rho: float,
    deadline_cost_reward_weight=1.0,
    energy_reward_weight=0.0,
    energy_statistics=None,
    shared_reward=None,
    shared_reward_mode=None,
    reward_statistics=None,
):
    """Side-effect-free exact OO score using the current upper-level reward.

    The current mode mirrors the shared incremental Deadline reward and adds the
    OO action's replay-normalized absolute energy saving. Legacy checkpoints keep
    their historical weighted reward and extra normalized Deadline cost. This is
    used only for the paper's Top-s precise choosing stage; the executed
    transition remains the source of truth written to replay.
    """

    ready_tasks = tuple(state.get("ready_tasks", ()))
    if not ready_tasks:
        raise ValueError("Cannot evaluate an OO action without a ready task.")
    task_row = ready_tasks[0]
    workflow_id = str(task_row["workflow_id"])
    task_id = int(task_row["task_id"])
    workflow_state = env.workflow_by_id[workflow_id]
    task = env._get_task(workflow_state["workflow"], task_id)
    server = env._get_server(int(server_id))
    normalized_rho = env._normalize_rho(float(rho), server)
    primary_plan = env.estimate_replica_plan(
        task=task,
        server=server,
        rho=normalized_rho,
        workflow_state=workflow_state,
        queue_type="execution",
        replica_role="primary",
    )

    deadline = max(float(env.workflow_relative_deadline(workflow_state)), EPS)
    workflow_task_count = max(1.0, float(objective._workflow_task_count(workflow_state)))
    previous_makespan = float(objective.previous_partial_makespan.get(workflow_id, 0.0))
    finish_times = [
        float(value)
        for value in workflow_state.get("task_finish_times", {}).values()
        if value is not None
    ]
    finish_times.append(float(primary_plan["aft"]))
    current_makespan = max(0.0, max(finish_times) - float(workflow_state["arrival_time"]))
    makespan_increment = max(0.0, current_makespan - previous_makespan)
    current_slack_ratio = (deadline - current_makespan) / deadline
    context = state.get("decision_context", {})
    task_energy_scale = max(
        float(context.get("task_energy_scale", primary_plan["energy"])), EPS
    )

    if objective.reward_mode == "deadline_incremental_only":
        normalized_makespan_increment = makespan_increment / deadline
        previous_violation = objective._deadline_violation_with_unit_jump(
            previous_makespan, deadline
        )
        current_violation = objective._deadline_violation_with_unit_jump(
            current_makespan, deadline
        )
        violation_delta = max(0.0, current_violation - previous_violation)
        deadline_penalty = (
            float(objective.deadline_incremental_violation_weight)
            * violation_delta
        )
        deadline_reward = -normalized_makespan_increment - deadline_penalty
        if shared_reward_mode == "hierarchical_fixed_lambda_v2":
            reference_plan = (
                primary_plan
                if abs(float(normalized_rho) - 1.0) <= EPS
                else env.estimate_replica_plan(
                    task=task,
                    server=server,
                    rho=1.0,
                    workflow_state=workflow_state,
                    queue_type="execution",
                    replica_role="primary",
                )
            )
            high_components = compose_fixed_lambda_upper_reward(
                normalized_makespan_increment=normalized_makespan_increment,
                deadline_violation_delta=violation_delta,
                energy_scale=task_energy_scale,
                # The upper placement term evaluates the chosen server at
                # rho=1, exactly as D3QN does before lower-level DVFS.
                energy_actual=float(reference_plan["energy"]),
                workflow_task_count=workflow_task_count,
                fixed_deadline_lambda=objective.fixed_deadline_lambda,
                fixed_deadline_miss_penalty=(
                    objective.deadline_incremental_violation_weight
                ),
            )
            dvfs_components = compute_dvfs_reward_components(
                deadline=deadline,
                partial_makespan_before=previous_makespan,
                workflow_arrival_time=float(workflow_state["arrival_time"]),
                reference_aft=float(reference_plan["aft"]),
                actual_aft=float(primary_plan["aft"]),
                reference_energy=float(reference_plan["energy"]),
                actual_energy=float(primary_plan["energy"]),
                slack_floor_ratio=shared_reward.slack_floor_ratio,
            )
            reliability_components = (
                low_reliability_counterfactual_cost(
                    env,
                    workflow_state,
                    task_id,
                    reference_plan["reliability"],
                    primary_plan["reliability"],
                )
                if shared_reward.low_reliability_violation_weight > 0.0
                else {
                    "reference_reliability_violation": 0.0,
                    "actual_reliability_violation": 0.0,
                    "reliability_violation_delta": 0.0,
                    "reliability_crossed": 0.0,
                }
            )
            composed = shared_reward.compose(
                float(high_components["reward"]),
                energy_saving=float(dvfs_components["energy_saving"]),
                slack_time_cost=float(dvfs_components["slack_time_cost"]),
                deadline_violation_delta=float(
                    dvfs_components["deadline_constraint_cost"]
                ),
                reliability_violation_delta=float(
                    reliability_components["reliability_violation_delta"]
                ),
                statistics=reward_statistics,
            )
            return {
                **dvfs_components,
                **reliability_components,
                "replay_reward": float(composed["reward"]),
                "high_reward": float(composed["reward"]),
                "deadline_reward": float(high_components["deadline_reward"]),
                "hierarchical_high_reward": float(high_components["reward"]),
                "deadline_penalty": float(high_components["deadline_cost"]),
                "current_makespan": float(current_makespan),
                "makespan_increment": float(makespan_increment),
                "normalized_makespan_increment": float(
                    normalized_makespan_increment
                ),
                "deadline_violation": float(current_violation),
                "deadline_violation_delta": float(
                    dvfs_components["deadline_constraint_cost"]
                ),
                "energy_reference": float(reference_plan["energy"]),
                "energy_saving": float(dvfs_components["energy_saving"]),
                "energy_saving_ratio": float(
                    dvfs_components["energy_saving_ratio"]
                ),
                "energy_reward": float(composed["energy_reward"]),
                "placement_energy_reference": float(task_energy_scale),
                "placement_energy_actual_rho1": float(reference_plan["energy"]),
                "placement_energy_reward": float(
                    high_components["energy_reward"]
                ),
                "rho": float(normalized_rho),
                "energy": float(primary_plan["energy"]),
                "aft": float(primary_plan["aft"]),
                **{
                    f"joint_{key}": value
                    for key, value in composed.items()
                    if key != "reward"
                },
            }
        if shared_reward_mode == "fixed_lambda_upper_v1":
            composed = compose_fixed_lambda_upper_reward(
                normalized_makespan_increment=normalized_makespan_increment,
                deadline_violation_delta=violation_delta,
                energy_scale=task_energy_scale,
                energy_actual=float(primary_plan["energy"]),
                workflow_task_count=workflow_task_count,
                fixed_deadline_lambda=objective.fixed_deadline_lambda,
                fixed_deadline_miss_penalty=(
                    objective.deadline_incremental_violation_weight
                ),
            )
            return {
                "replay_reward": float(composed["reward"]),
                "high_reward": float(composed["reward"]),
                "deadline_reward": float(composed["deadline_reward"]),
                "deadline_penalty": float(composed["deadline_cost"]),
                "current_makespan": float(current_makespan),
                "makespan_increment": float(makespan_increment),
                "normalized_makespan_increment": float(
                    normalized_makespan_increment
                ),
                "deadline_violation": float(current_violation),
                "deadline_violation_delta": float(violation_delta),
                "energy_reference": float(composed["energy_scale"]),
                "energy_saving": float(composed["energy_saving"]),
                "energy_saving_ratio": float(
                    composed["energy_saving_ratio"]
                ),
                "energy_reward": float(composed["energy_reward"]),
                "fixed_deadline_lambda": float(
                    composed["fixed_deadline_lambda"]
                ),
                "slack_time_cost": 0.0,
                "rho": float(normalized_rho),
                "energy": float(primary_plan["energy"]),
                "aft": float(primary_plan["aft"]),
            }
        if shared_reward is not None:
            reference_plan = (
                primary_plan
                if abs(float(normalized_rho) - 1.0) <= EPS
                else env.estimate_replica_plan(
                    task=task,
                    server=server,
                    rho=1.0,
                    workflow_state=workflow_state,
                    queue_type="execution",
                    replica_role="primary",
                )
            )
            components = compute_dvfs_reward_components(
                deadline=deadline,
                partial_makespan_before=previous_makespan,
                workflow_arrival_time=float(workflow_state["arrival_time"]),
                reference_aft=float(reference_plan["aft"]),
                actual_aft=float(primary_plan["aft"]),
                reference_energy=float(reference_plan["energy"]),
                actual_energy=float(primary_plan["energy"]),
                slack_floor_ratio=shared_reward.slack_floor_ratio,
            )
            reliability_components = (
                low_reliability_counterfactual_cost(
                    env,
                    workflow_state,
                    task_id,
                    reference_plan["reliability"],
                    primary_plan["reliability"],
                )
                if shared_reward.low_reliability_violation_weight > 0.0
                else {
                    "reference_reliability_violation": 0.0,
                    "actual_reliability_violation": 0.0,
                    "reliability_violation_delta": 0.0,
                    "reliability_crossed": 0.0,
                }
            )
            if shared_reward_mode == "shared_upper_same_server_energy_v2":
                composed = compose_upper_deadline_energy_reward(
                    normalized_makespan_increment=normalized_makespan_increment,
                    deadline_violation_delta=violation_delta,
                    energy_reference=float(reference_plan["energy"]),
                    energy_actual=float(primary_plan["energy"]),
                    workflow_task_count=workflow_task_count,
                    deadline_violation_weight=(
                        objective.deadline_incremental_violation_weight
                    ),
                    energy_weight=objective.high_energy_weight,
                )
                return {
                    **components,
                    "replay_reward": float(composed["reward"]),
                    "high_reward": float(composed["reward"]),
                    "deadline_reward": float(
                        composed["time_reward"] - composed["deadline_penalty"]
                    ),
                    "deadline_penalty": float(composed["deadline_penalty"]),
                    "current_makespan": float(current_makespan),
                    "makespan_increment": float(makespan_increment),
                    "normalized_makespan_increment": float(
                        normalized_makespan_increment
                    ),
                    "deadline_violation": float(current_violation),
                    "deadline_violation_delta": float(violation_delta),
                    "energy_reference": float(composed["energy_reference"]),
                    "energy_saving": float(composed["energy_saving"]),
                    "energy_saving_ratio": float(
                        composed["energy_saving_ratio"]
                    ),
                    "energy_reward": float(composed["energy_reward"]),
                    "rho": float(normalized_rho),
                    "energy": float(primary_plan["energy"]),
                    "aft": float(primary_plan["aft"]),
                }
            composed = shared_reward.compose(
                deadline_reward,
                energy_saving=components["energy_saving"],
                slack_time_cost=components["slack_time_cost"],
                deadline_violation_delta=components["deadline_violation"],
                reliability_violation_delta=(
                    reliability_components["reliability_violation_delta"]
                ),
                statistics=reward_statistics,
            )
            return {
                **components,
                **reliability_components,
                "replay_reward": float(composed["reward"]),
                "high_reward": float(composed["reward"]),
                "deadline_reward": float(deadline_reward),
                "deadline_penalty": float(deadline_penalty),
                "current_makespan": float(current_makespan),
                "makespan_increment": float(makespan_increment),
                "normalized_makespan_increment": float(
                    normalized_makespan_increment
                ),
                "deadline_violation": float(current_violation),
                "deadline_violation_delta": float(
                    components["deadline_violation"]
                ),
                "rho": float(normalized_rho),
                "energy": float(primary_plan["energy"]),
                "aft": float(primary_plan["aft"]),
                **{
                    f"joint_{key}": value
                    for key, value in composed.items()
                    if key not in {"reward", "high_reward"}
                },
            }
        energy_saving = task_energy_scale - float(primary_plan["energy"])
        statistics = dict(energy_statistics or {})
        if int(statistics.get("count", 0)) >= 2:
            energy_zscore = (
                energy_saving - float(statistics.get("mean", 0.0))
            ) / max(float(statistics.get("std", 0.05)), EPS)
        else:
            energy_zscore = 0.0
        energy_reward = float(energy_reward_weight) * energy_zscore
        replay_reward = deadline_reward + energy_reward
        return {
            "replay_reward": float(replay_reward),
            "high_reward": float(replay_reward),
            "deadline_reward": float(deadline_reward),
            "deadline_penalty": float(deadline_penalty),
            "current_makespan": float(current_makespan),
            "makespan_increment": float(makespan_increment),
            "normalized_makespan_increment": float(normalized_makespan_increment),
            "deadline_violation": float(current_violation),
            "deadline_violation_delta": float(violation_delta),
            "energy_reference": float(task_energy_scale),
            "energy_saving": float(energy_saving),
            "energy_zscore": float(energy_zscore),
            "energy_reward": float(energy_reward),
            "rho": float(normalized_rho),
            "energy": float(primary_plan["energy"]),
            "aft": float(primary_plan["aft"]),
        }

    high_time_reward = (
        float(objective.high_makespan_weight)
        * current_slack_ratio
        / workflow_task_count
    )

    high_energy_reward_raw = (
        task_energy_scale - float(primary_plan["energy"])
    ) / task_energy_scale
    high_energy_reward = (
        float(objective.high_energy_weight)
        * high_energy_reward_raw
        / workflow_task_count
    )

    previous_log_risk = float(objective.previous_partial_log_risk.get(workflow_id, 0.0))
    existing_reliabilities = [
        min(1.0, max(EPS, float(value)))
        for value in workflow_state.get("task_reliability", {}).values()
    ]
    existing_reliabilities.append(
        min(1.0, max(EPS, float(primary_plan["reliability"])))
    )
    reliability_product = float(math.prod(existing_reliabilities))
    current_log_risk = -math.log(min(1.0, max(EPS, reliability_product)))
    reliability_limit = float(objective._allowed_log_risk(env, workflow_state))
    reliability_scale = max(reliability_limit, EPS)
    previous_violation = max(0.0, previous_log_risk - reliability_limit) / reliability_scale
    current_violation = max(0.0, current_log_risk - reliability_limit) / reliability_scale
    violation_delta = max(0.0, current_violation - previous_violation)
    reliability_crossed = float(current_violation > EPS and previous_violation <= EPS)
    reliability_penalty = (
        float(objective.high_reliability_fixed_penalty) * reliability_crossed
        + float(objective.high_reliability_severity_weight) * violation_delta
    )
    high_reward = high_time_reward + high_energy_reward - reliability_penalty
    deadline_penalty = (
        float(deadline_cost_reward_weight) * makespan_increment / deadline
    )
    replay_reward = high_reward - deadline_penalty
    values = {
        "replay_reward": float(replay_reward),
        "high_reward": float(high_reward),
        "deadline_penalty": float(deadline_penalty),
        "current_makespan": float(current_makespan),
        "makespan_increment": float(makespan_increment),
        "high_time_reward": float(high_time_reward),
        "high_energy_reward": float(high_energy_reward),
        "reliability_penalty": float(reliability_penalty),
        "rho": float(normalized_rho),
        "energy": float(primary_plan["energy"]),
        "aft": float(primary_plan["aft"]),
        "reliability": float(primary_plan["reliability"]),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("OO candidate evaluation produced NaN or Inf.")
    return values
