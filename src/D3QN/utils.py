from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def as_torch_rng_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype=torch.uint8)
    return torch.as_tensor(value, dtype=torch.uint8)
