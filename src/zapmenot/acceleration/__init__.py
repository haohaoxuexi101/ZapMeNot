"""Acceleration backend registry and helpers for ZapMeNot.

This module exposes a small plugin system that allows the numerical hot
loops in the point-kernel solver to be executed by interchangeable
backends.  The default backend uses NumPy which mirrors the historical
behaviour.  Optional backends such as the Taichi GPU implementation can be
registered to provide accelerated execution without forcing callers to
sprinkle conditional logic across the code base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BackendInfo:
    """Descriptive information about a registered backend."""

    name: str
    description: str
    uses_gpu: bool


class BaseAccelerationBackend:
    """Abstract base class for acceleration backends."""

    name: str = "base"
    description: str = ""
    uses_gpu: bool = False

    def is_available(self) -> bool:
        """Return ``True`` when the backend can be used."""

        return True

    # ---- Geometry helpers -------------------------------------------------
    def compute_ray_geometry(
        self, starts: np.ndarray, end: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (lengths, directions, inverse_directions, signs)."""

        raise NotImplementedError

    # ---- Exposure accumulation -------------------------------------------
    def accumulate_exposure(
        self,
        weights: np.ndarray,
        total_distance: np.ndarray,
        total_mfp: np.ndarray,
        photon_energy: float,
        photon_yield: float,
        conversion_factor: float,
        dose_coeff: float,
        buildup: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Return (total_flux, uncollided_exposure, collided_exposure)."""

        raise NotImplementedError


_BACKEND_REGISTRY: Dict[str, BaseAccelerationBackend] = {}


def register_backend(backend: BaseAccelerationBackend) -> None:
    """Register a backend implementation."""

    if backend.name in _BACKEND_REGISTRY:
        raise ValueError(f"Backend '{backend.name}' already registered")
    _BACKEND_REGISTRY[backend.name] = backend


def get_backend(
    name: Optional[str] = None,
    prefer_gpu: bool = False,
    fallback_to_default: bool = True,
) -> BaseAccelerationBackend:
    """Retrieve a backend by name or preference."""

    if name is not None:
        backend = _BACKEND_REGISTRY.get(name)
        if backend is None:
            raise KeyError(f"Backend '{name}' is not registered")
        if not backend.is_available():
            raise RuntimeError(f"Backend '{name}' is not available")
        return backend

    gpu_candidates = [b for b in _BACKEND_REGISTRY.values() if b.uses_gpu and b.is_available()]
    cpu_candidates = [b for b in _BACKEND_REGISTRY.values() if (not b.uses_gpu) and b.is_available()]

    if prefer_gpu and gpu_candidates:
        return gpu_candidates[0]
    if cpu_candidates:
        return cpu_candidates[0]
    if gpu_candidates:
        return gpu_candidates[0]

    if not fallback_to_default:
        raise RuntimeError("No acceleration backend is currently available")

    raise RuntimeError("No acceleration backend is registered")


def list_backends(include_unavailable: bool = False) -> Iterable[BackendInfo]:
    """Yield information about the registered backends."""

    for backend in _BACKEND_REGISTRY.values():
        if include_unavailable or backend.is_available():
            yield BackendInfo(backend.name, backend.description, backend.uses_gpu)


def has_backend(name: str) -> bool:
    """Return ``True`` if a backend with *name* exists."""

    backend = _BACKEND_REGISTRY.get(name)
    return bool(backend and backend.is_available())


def _ensure_numpy_registered() -> None:
    """Import the NumPy backend lazily to avoid circular imports."""

    if "numpy" not in _BACKEND_REGISTRY:
        from . import numpy_backend  # noqa: F401  # side-effect registration


_ensure_numpy_registered()

try:  # pragma: no cover - optional dependency
    from . import taichi_backend  # noqa: F401  # side-effect registration
except Exception:  # noqa: BLE001 - we intentionally swallow import failures
    pass
