"""MAPPO checkpoint helpers with explicit, versioned RNG state handling."""

import random
from typing import Any, Dict, Optional

import numpy as np
import torch


MAPPO_RNG_STATE_VERSION = 1


def capture_rng_state() -> Dict[str, Any]:
    """Capture every process-global RNG used by MAPPO training."""
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "version": MAPPO_RNG_STATE_VERSION,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda_all": cuda_state,
    }


def restore_rng_state(state: Dict[str, Any]) -> Dict[str, bool]:
    """Restore a state produced by :func:`capture_rng_state`.

    CUDA state is restored only when CUDA is available in the current process.
    Missing required CPU-side fields are treated as an invalid checkpoint rather
    than silently producing a non-reproducible continuation.
    """
    if not isinstance(state, dict):
        raise TypeError("MAPPO RNG state must be a dictionary.")
    version = int(state.get("version", -1))
    if version != MAPPO_RNG_STATE_VERSION:
        raise ValueError(
            f"Unsupported MAPPO RNG state version {version}; "
            f"expected {MAPPO_RNG_STATE_VERSION}."
        )
    missing = [key for key in ("python", "numpy", "torch_cpu") if key not in state]
    if missing:
        raise KeyError(f"MAPPO RNG state is missing required fields: {missing}")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())

    cuda_saved = state.get("torch_cuda_all", None)
    cuda_restored = False
    if cuda_saved is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_saved])
        cuda_restored = True
    return {
        "python": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda_all": cuda_restored,
    }


def checkpoint_rng_state(checkpoint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a supported MAPPO RNG payload, or ``None`` for legacy checkpoints."""
    state = checkpoint.get("rng_state", None)
    return state if isinstance(state, dict) else None


def torch_load_checkpoint(path: str, map_location=None):
    """Load a trusted local checkpoint across old and new PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
