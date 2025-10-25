"""Compatibility shims for Taichi acceleration helpers."""

from __future__ import annotations

import numpy as np

from .acceleration import get_backend, has_backend


def is_available() -> bool:
    """Return ``True`` when the Taichi backend can be used."""

    try:
        return has_backend("taichi")
    except Exception:  # pragma: no cover - defensive
        return False


def _get_backend():
    try:
        backend = get_backend("taichi")
    except (KeyError, RuntimeError) as exc:  # pragma: no cover - guard rails
        raise RuntimeError("Taichi acceleration was requested but is unavailable") from exc
    if not backend.is_available():  # pragma: no cover - sanity check
        raise RuntimeError("Taichi acceleration was requested but is unavailable")
    return backend


def compute_exposure(
    weights: np.ndarray,
    total_distance: np.ndarray,
    total_mfp: np.ndarray,
    photon_energy: float,
    photon_yield: float,
    conversion_factor: float,
    dose_coeff: float,
    buildup: np.ndarray,
):
    """Compute exposure values via the Taichi backend."""

    backend = _get_backend()
    return backend.accumulate_exposure(
        weights,
        total_distance,
        total_mfp,
        photon_energy,
        photon_yield,
        conversion_factor,
        dose_coeff,
        buildup,
    )


def compute_ray_geometry(
    starts: np.ndarray,
    end_point: np.ndarray,
):
    """Compute ray geometry via the Taichi backend."""

    backend = _get_backend()
    return backend.compute_ray_geometry(starts, end_point)
