from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DReaMConfig:
    """Paper hyperparameters, with edge-cloud/implementation switches."""

    hidden_dim: int = 128
    gamma: float = 0.3
    learning_rate: float = 0.05
    replay_capacity: int = 16800
    batch_size: int = 1000
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.9999
    target_update_frequency: int = 50
    training_per_episode: int = 1
    training_episodes: int = 100000
    max_grad_norm: float = 1.0
    huber_delta: float = 1.0
    max_abs_q: float = 1000.0
    max_abs_parameter: float = 1000.0
    enable_fault_tolerance: bool = False
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        if self.hidden_dim <= 0 or self.replay_capacity <= 0 or self.batch_size <= 1:
            raise ValueError("DReaM dimensions/capacities must be positive; batch_size must exceed one.")
        if not 0.0 <= self.gamma <= 1.0 or self.learning_rate <= 0.0:
            raise ValueError("Invalid DReaM gamma or learning rate.")
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ValueError("Invalid DReaM epsilon range.")
        if (
            not 0.0 < self.epsilon_decay <= 1.0
            or self.target_update_frequency <= 0
            or self.training_per_episode <= 0
        ):
            raise ValueError("Invalid DReaM decay/update frequency.")
        if self.max_grad_norm <= 0.0 or self.huber_delta <= 0.0:
            raise ValueError("DReaM gradient norm and Huber delta must be positive.")
        if self.max_abs_q <= 0.0 or self.max_abs_parameter <= 0.0:
            raise ValueError("DReaM divergence thresholds must be positive.")

    @classmethod
    def from_mapping(cls, values=None):
        values = dict(values or {})
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in known})

    def as_dict(self):
        return asdict(self)
