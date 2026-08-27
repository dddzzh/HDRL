from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HIRLConfig:
    hidden_dim: int = 256
    gamma: float = 0.99
    ddqn_learning_rate: float = 1e-4
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 5e-4
    ddqn_replay_size: int = 600_000
    td3_replay_size: int = 5_000
    batch_size: int = 256
    target_update_frequency: int = 1_000
    tau: float = 0.001
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    load_noise_alpha: float = 0.5
    load_learning_rate_alpha: float = 0.5
    epsilon_min: float = 0.01
    epsilon_max: float = 0.3
    epsilon_beta: float = 2.0
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    failure_priority: float = 1.0
    completion_reward: float = 1.0
    energy_weight: float = 0.5
    latency_weight: float = 0.5
    failure_penalty: float = 1.0
    fixed_reliability_lambda: float = 1.0
    min_rho: float = 0.2
    max_rho: float = 1.0
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "auto"

    @classmethod
    def from_mapping(cls, values=None):
        values = dict(values or {})
        known = cls.__dataclass_fields__
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"Unknown HIRL configuration fields: {unknown}.")
        result = cls(**values)
        if result.min_rho <= 0.0 or result.max_rho <= result.min_rho:
            raise ValueError("HIRL requires 0 < min_rho < max_rho.")
        if result.batch_size <= 0 or result.hidden_dim <= 0:
            raise ValueError("HIRL batch_size and hidden_dim must be positive.")
        if result.load_learning_rate_alpha < 0.0:
            raise ValueError("HIRL load_learning_rate_alpha must be non-negative.")
        if result.fixed_reliability_lambda < 0.0:
            raise ValueError("HIRL fixed reliability lambda must be non-negative.")
        return result

    def as_dict(self):
        return asdict(self)
