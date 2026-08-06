from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# Simulation, data generation, metrics, and numpy state encoding stay on CPU.
SIMULATION_DEVICE = "cpu"


def resolve_agent_device(device: str | None = None):
    """Device used by neural-network agents and optimizers."""
    return resolve_device(device)


def resolve_device(device: str | None = None):
    """Resolve agent training device.

    - ``cpu``: run agent networks on CPU.
    - ``cuda`` / ``cuda:0``: run agent networks on GPU.
    - ``auto``: use CUDA when available, otherwise CPU.
    """
    _require_torch()
    device_name = str(device).lower() if device is not None else None
    if device_name in {"gpu"}:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device is None or str(device).lower() in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{requested}' but CUDA is not available. "
            "Install a CUDA build of PyTorch or use --device cpu."
        )
    return torch.device(requested)


def describe_device(device) -> str:
    _require_torch()
    if device.type != "cuda":
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return f"cuda ({torch.cuda.get_device_name(index)})"


def describe_runtime_devices(agent_device) -> str:
    return (
        f"simulation={SIMULATION_DEVICE} (env / metrics / state encoding), "
        f"agent={describe_device(agent_device)} (neural-network agents)"
    )


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required. Install it with: pip install torch")
