"""Reference NumPy backend for ZapMeNot acceleration hooks."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from . import BaseAccelerationBackend, register_backend


class NumpyBackend(BaseAccelerationBackend):
    """Pure NumPy backend that mirrors the legacy implementation."""

    name = "numpy"
    description = "NumPy reference backend"
    uses_gpu = False

    def compute_ray_geometry(
        self, starts: np.ndarray, end: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        diff = end - starts
        lengths = np.linalg.norm(diff, axis=1)
        directions = np.zeros_like(diff)
        nonzero = lengths > 0
        if np.any(nonzero):
            directions[nonzero] = diff[nonzero] / lengths[nonzero][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse_directions = 1.0 / directions
        inverse_directions[~np.isfinite(inverse_directions)] = math.inf
        signs = (inverse_directions < 0).astype(np.int32)
        return lengths, directions, inverse_directions, signs

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
        inv_r2 = 1.0 / (4.0 * math.pi * np.power(total_distance, 2))
        attenuation = np.exp(-total_mfp)
        uncollided_flux = photon_yield * weights * attenuation * photon_energy * inv_r2
        total_flux = float(np.sum(uncollided_flux))
        uncollided_exposure = (
            uncollided_flux * conversion_factor * dose_coeff * 1000.0 * 3600.0
        )
        total_uncollided = float(np.sum(uncollided_exposure))
        total_collided = float(np.sum(uncollided_exposure * buildup))
        return total_flux, total_uncollided, total_collided


register_backend(NumpyBackend())
