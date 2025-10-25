# Taichi GPU Optimisation Guide

This guide summarises the hotspots in ZapMeNot's shielding solver that are
amenable to Taichi acceleration and describes how the new backend system
activates GPU parallelism.

## Overview of Accelerated Components

| Module | Hot Loop | Description | Accelerator Hook |
| --- | --- | --- | --- |
| `src/zapmenot/model.py` | Exposure accumulation over quadrature points and photons | Builds mean-free-path totals, applies buildup factors, and aggregates flux and exposure contributions. | Delegates to `BaseAccelerationBackend.accumulate_exposure` so Taichi can parallelise per-source-point computations. |
| `src/zapmenot/ray.py` | Ray construction for detector-to-source pairs | Computes ray length, unit direction, inverse direction, and sign flags for each source point. | Delegates to `BaseAccelerationBackend.compute_ray_geometry` which maps to a Taichi kernel for GPU execution. |
| `src/zapmenot/shield.py` | Geometric intersection routines | Analytical intersections are evaluated per-ray per-shield; these rely on vector operations but remain numerically stable on CPU. | Remain on CPU; results feed the accelerated exposure kernel. |
| `src/zapmenot/source.py` | Quadrature generation | Generates deterministic quadrature grids. | Remains on CPU because data generation is not compute-bound. |

## Enabling GPU Execution

1. **Backend Selection:** `Model` instances default to `get_backend(prefer_gpu=True)` which selects the Taichi backend when the runtime is available. Users can override the backend by calling `Model.set_acceleration_backend()` or the compatibility helper `Model.enable_taichi_acceleration()`.

2. **Ray Geometry:** `ray.build_rays()` now accepts either an explicit backend or the legacy `use_taichi` flag. The Taichi backend executes `_ray_geometry_kernel` where each thread computes the geometry attributes for a source point independently.

3. **Exposure Aggregation:** `BaseAccelerationBackend.accumulate_exposure()` batches the per-point exposure contributions. In the Taichi backend the `_exposure_kernel` computes inverse-square fall-off, attenuation, and buildup adjustments in parallel on the GPU.

4. **Data Transfer:** Inputs are converted to contiguous `float64` buffers before being passed to Taichi kernels, preserving numerical parity with the NumPy reference implementation.

## Integrating New Accelerated Paths

To offload additional loops:

1. Identify a NumPy routine that performs a bulk operation across source points or shields.
2. Add an abstract method to `BaseAccelerationBackend` describing the operation's inputs and outputs.
3. Implement the NumPy version in `numpy_backend.py` to maintain correctness.
4. Implement the Taichi kernel in `taichi_backend.py`, ensuring kernel signatures accept `ti.types.ndarray` with matching dtypes.
5. Call the new backend method from the high-level module (e.g., `model.py` or `shield.py`).

## Runtime Diagnostics

* `zapmenot.acceleration.list_backends()` exposes the registered backends and whether they utilise the GPU.
* `Model.is_taichi_acceleration_enabled()` reports whether the Taichi backend is active.
* Exceptions raised by `ray.build_rays(..., use_taichi=True)` clearly signal when Taichi is requested but unavailable.

Refer to `docs/point_kernel_parallel_manual.md` for the theoretical motivation behind the kernels and the parallel programming model assumptions.
