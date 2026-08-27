from __future__ import annotations

from dataclasses import dataclass

from .dream_fault_tolerance import FaultToleranceTechnique


DREAM_RHO_LEVELS = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class DReaMDecision:
    server_id: int
    dvfs_index: int
    rho: float
    ft_index: int | None = None
    ft_technique: FaultToleranceTechnique | None = None

    def as_dict(self):
        return {
            "server_id": self.server_id, "dvfs_index": self.dvfs_index,
            "rho": self.rho, "ft_index": self.ft_index,
            "ft_technique": self.ft_technique,
        }


def action_dimension(num_servers: int, enable_fault_tolerance: bool = False) -> int:
    if int(num_servers) <= 0:
        raise ValueError("num_servers must be positive.")
    return int(num_servers) * len(DREAM_RHO_LEVELS) * (3 if enable_fault_tolerance else 1)


def encode_action(server_id, dvfs_index, num_servers, enable_fault_tolerance=False, ft_index=None):
    server_id, dvfs_index = int(server_id), int(dvfs_index)
    if not 0 <= server_id < int(num_servers):
        raise ValueError("Invalid DReaM server_id.")
    if not 0 <= dvfs_index < len(DREAM_RHO_LEVELS):
        raise ValueError("Invalid DReaM dvfs_index.")
    base = server_id * len(DREAM_RHO_LEVELS) + dvfs_index
    if not enable_fault_tolerance:
        if ft_index is not None:
            raise ValueError("ft_index must be None while fault tolerance is disabled.")
        return base
    if ft_index is None or not 0 <= int(ft_index) < 3:
        raise ValueError("FT-enabled DReaM requires ft_index in [0, 2].")
    return base * 3 + int(ft_index)


def decode_action(action_id, num_servers, enable_fault_tolerance=False):
    action_id = int(action_id)
    dim = action_dimension(num_servers, enable_fault_tolerance)
    if not 0 <= action_id < dim:
        raise ValueError(f"DReaM action_id {action_id} outside [0, {dim}).")
    ft_index = action_id % 3 if enable_fault_tolerance else None
    base = action_id // 3 if enable_fault_tolerance else action_id
    server_id, dvfs_index = divmod(base, len(DREAM_RHO_LEVELS))
    rho = DREAM_RHO_LEVELS[dvfs_index]
    return DReaMDecision(
        server_id, dvfs_index, rho, ft_index,
        FaultToleranceTechnique(ft_index) if ft_index is not None else None,
    )
