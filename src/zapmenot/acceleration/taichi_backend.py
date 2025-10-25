"""Taichi accelerated backend for ZapMeNot."""

from __future__ import annotations

import importlib
import math
from typing import Tuple

import numpy as np

from . import BaseAccelerationBackend, register_backend

_taichi_spec = importlib.util.find_spec("taichi")
if _taichi_spec is not None:  # pragma: no cover - guarded import
    import taichi as ti  # type: ignore
else:  # pragma: no cover - guarded import
    ti = None  # type: ignore


class TaichiBackend(BaseAccelerationBackend):
    """Acceleration backend implemented using Taichi kernels."""

    name = "taichi"
    description = "Taichi parallel backend"
    uses_gpu = True

    def __init__(self) -> None:
        self._initialised = False
        self._arch = None
        self.uses_gpu = True

    def is_available(self) -> bool:
        return ti is not None

    def _ensure_initialised(self) -> None:
        if not self.is_available():
            raise RuntimeError("Taichi is not available in this environment")
        if self._initialised:
            return
        # Prefer GPU execution when available, but gracefully fall back to CPU.
        # Taichi selects an appropriate backend for the requested architecture.
        preferred_architectures = [ti.gpu, ti.cpu]
        for arch in preferred_architectures:
            try:
                ti.init(arch=arch, default_fp=ti.f64, default_ip=ti.i32)
                self._arch = arch
                self.uses_gpu = arch != ti.cpu
                break
            except Exception:  # pragma: no cover - backend probing
                continue
        if self._arch is None:  # pragma: no cover - safety net
            ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32)
            self._arch = ti.cpu
            self.uses_gpu = False
        self._initialised = True

    def compute_ray_geometry(
        self, starts: np.ndarray, end: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_initialised()
        n = int(starts.shape[0])
        lengths = np.zeros(n, dtype=np.float64)
        directions = np.zeros((n, 3), dtype=np.float64)
        inverse_directions = np.zeros((n, 3), dtype=np.float64)
        signs = np.zeros((n, 3), dtype=np.int32)
        if n == 0:
            return lengths, directions, inverse_directions, signs
        starts_np = np.ascontiguousarray(starts, dtype=np.float64)
        end_np = np.ascontiguousarray(end, dtype=np.float64)
        _ray_geometry_kernel(
            starts_np,
            end_np,
            lengths,
            directions,
            inverse_directions,
            signs,
            n,
        )
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
        self._ensure_initialised()
        n = int(weights.shape[0])
        if n == 0:
            return 0.0, 0.0, 0.0
        weights_np = np.ascontiguousarray(weights, dtype=np.float64)
        total_distance_np = np.ascontiguousarray(total_distance, dtype=np.float64)
        total_mfp_np = np.ascontiguousarray(total_mfp, dtype=np.float64)
        buildup_np = np.ascontiguousarray(buildup, dtype=np.float64)
        result = np.zeros(3, dtype=np.float64)
        _exposure_kernel(
            weights_np,
            total_distance_np,
            total_mfp_np,
            buildup_np,
            result,
            n,
            float(photon_energy),
            float(photon_yield),
            float(conversion_factor),
            float(dose_coeff),
        )
        return float(result[0]), float(result[1]), float(result[2])


if _taichi_spec is not None:  # pragma: no cover - compiled conditionally

    @ti.kernel  # type: ignore[misc]
    def _exposure_kernel(
        weights: ti.types.ndarray(dtype=ti.f64, ndim=1),
        total_distance: ti.types.ndarray(dtype=ti.f64, ndim=1),
        total_mfp: ti.types.ndarray(dtype=ti.f64, ndim=1),
        buildup: ti.types.ndarray(dtype=ti.f64, ndim=1),
        result: ti.types.ndarray(dtype=ti.f64, ndim=1),
        n: ti.i32,
        photon_energy: ti.f64,
        photon_yield: ti.f64,
        conversion_factor: ti.f64,
        dose_coeff: ti.f64,
    ) -> None:
        total_flux = ti.cast(0.0, ti.f64)
        total_uncollided = ti.cast(0.0, ti.f64)
        total_collided = ti.cast(0.0, ti.f64)
        for i in range(n):
            inv_r2 = 1.0 / (4.0 * math.pi * total_distance[i] * total_distance[i])
            attenuation = ti.exp(-total_mfp[i])
            uncollided_flux = (
                photon_yield
                * weights[i]
                * attenuation
                * photon_energy
                * inv_r2
            )
            total_flux += uncollided_flux
            exposure = (
                uncollided_flux
                * conversion_factor
                * dose_coeff
                * 1000.0
                * 3600.0
            )
            total_uncollided += exposure
            total_collided += exposure * buildup[i]
        result[0] = total_flux
        result[1] = total_uncollided
        result[2] = total_collided


if _taichi_spec is not None:  # pragma: no cover - compiled conditionally

    @ti.kernel  # type: ignore[misc]
    def _ray_geometry_kernel(
        starts: ti.types.ndarray(dtype=ti.f64, ndim=2),
        end_point: ti.types.ndarray(dtype=ti.f64, ndim=1),
        lengths: ti.types.ndarray(dtype=ti.f64, ndim=1),
        directions: ti.types.ndarray(dtype=ti.f64, ndim=2),
        inverse_directions: ti.types.ndarray(dtype=ti.f64, ndim=2),
        signs: ti.types.ndarray(dtype=ti.i32, ndim=2),
        n: ti.i32,
    ) -> None:
        for i in range(n):
            dx = end_point[0] - starts[i, 0]
            dy = end_point[1] - starts[i, 1]
            dz = end_point[2] - starts[i, 2]
            length = ti.sqrt(dx * dx + dy * dy + dz * dz)
            lengths[i] = length
            if length == 0.0:
                directions[i, 0] = 0.0
                directions[i, 1] = 0.0
                directions[i, 2] = 0.0
                inverse_directions[i, 0] = math.inf
                inverse_directions[i, 1] = math.inf
                inverse_directions[i, 2] = math.inf
                signs[i, 0] = 0
                signs[i, 1] = 0
                signs[i, 2] = 0
            else:
                inv_length = 1.0 / length
                dir_x = dx * inv_length
                dir_y = dy * inv_length
                dir_z = dz * inv_length
                directions[i, 0] = dir_x
                directions[i, 1] = dir_y
                directions[i, 2] = dir_z
                inv_dir_x = 1.0 / dir_x if dir_x != 0 else math.inf
                inv_dir_y = 1.0 / dir_y if dir_y != 0 else math.inf
                inv_dir_z = 1.0 / dir_z if dir_z != 0 else math.inf
                inverse_directions[i, 0] = inv_dir_x
                inverse_directions[i, 1] = inv_dir_y
                inverse_directions[i, 2] = inv_dir_z
                signs[i, 0] = 1 if inv_dir_x < 0 else 0
                signs[i, 1] = 1 if inv_dir_y < 0 else 0
                signs[i, 2] = 1 if inv_dir_z < 0 else 0


register_backend(TaichiBackend())
