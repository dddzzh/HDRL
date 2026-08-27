from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class DQNEdgeConfig:
    """Paper defaults use Table 4 when the paper contradicts Algorithm 1."""

    hidden_dim: int = 64
    hidden_layers: int = 2
    attention_hidden_dim: int = 32
    learning_rate: float = 5e-6
    attention_learning_rate: float = 5e-6
    gamma: float = 0.9
    batch_size: int = 128
    replay_capacity: int = 10_000
    epsilon_start: float = 0.3
    epsilon_min: float = 0.01
    epsilon_decay_per_step: float = 1e-5
    target_update_frequency: int = 100
    training_episodes: int = 500
    rank_critical_beta: float = 1.0
    attention_ema_alpha: float = 0.1
    max_grad_norm: float = 10.0
    seed: int = 42
    device: str = "cpu"

    @classmethod
    def from_mapping(cls, values=None):
        values = values or {}
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"Unknown DQN-Edge config fields: {unknown}")
        result = cls(**values)
        if result.hidden_layers != 2:
            raise ValueError("This reproduction implements the paper's two-hidden-layer network.")
        if not 0.0 < result.attention_ema_alpha <= 1.0:
            raise ValueError("attention_ema_alpha must be in (0, 1].")
        return result

    def as_dict(self):
        return asdict(self)
