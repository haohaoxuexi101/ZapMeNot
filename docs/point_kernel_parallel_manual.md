# Point-Kernel Parallel Acceleration Manual

This manual explains the theory behind ZapMeNot's point-kernel solver and how
parallel execution improves runtime without sacrificing numerical fidelity.

## 1. Point-Kernel Fundamentals

A point-kernel shielding calculation estimates dose at a detector by summing
contributions from discrete source points. Each contribution depends on:

1. **Geometric attenuation** via inverse-square law (`1 / (4π r²)`).
2. **Material attenuation** governed by the exponential `exp(-Σ μ_i d_i)` where
   `μ_i` is the macroscopic attenuation coefficient for shield `i` and `d_i` is
   the path length through that shield.
3. **Buildup factors** that account for scattered photons.
4. **Conversion coefficients** translating flux to exposure or dose.

The classical solver iterates over every source point for every photon energy,
performing identical arithmetic on independent data. This embarrassingly
parallel structure is ideal for GPU acceleration.

## 2. Parallel Decomposition

### 2.1 Ray Geometry Kernel

* **Inputs:** `starts[N,3]`, `end[3]`
* **Outputs:** `lengths[N]`, `directions[N,3]`, `inverse_directions[N,3]`,
  `signs[N,3]`
* **Parallel Strategy:** Each thread computes the vector difference between a
  source point and the detector, normalises it, and records per-axis metadata.
  Zero-length rays are handled separately to avoid division by zero.

### 2.2 Exposure Accumulation Kernel

* **Inputs:**
  * `weights[N]` – quadrature weights
  * `total_distance[N]` – ray lengths
  * `total_mfp[N]` – total mean-free-path equivalents
  * `buildup[N]` – buildup factors (per-point or scalar broadcast)
  * Scalars for photon energy, photon yield, conversion factor, and dose
    coefficient
* **Outputs:**
  * `total_flux` – aggregate uncollided energy flux
  * `total_uncollided_exposure`
  * `total_collided_exposure`
* **Parallel Strategy:** Each thread evaluates the inverse-square fall-off and
  attenuation for one source point. Exposures are scaled and accumulated via
  thread-local variables that Taichi reduces deterministically into the output
  vector.

## 3. Numerical Considerations

* **Precision:** All kernels operate in double precision (`float64`) to match
  the NumPy backend and reduce sensitivity to large attenuation exponents.
* **Stability:** Division-by-zero is avoided by branching on zero-length rays
  and substituting `inf` for reciprocal directions.
* **Determinism:** Accumulations use deterministic reductions, ensuring GPU and
  CPU paths produce reproducible results.

## 4. Extending the Kernels

1. **Identify** independent work items (e.g., additional per-point physics
   corrections).
2. **Design** kernel inputs/outputs as plain arrays to minimise host-device
   synchronisation.
3. **Implement** the NumPy reference path first; then port the loop to a Taichi
   kernel with identical arithmetic.
4. **Validate** with regression tests comparing CPU and GPU backends using
   representative geometries and attenuation regimes.

## 5. Performance Tuning Checklist

- Use contiguous NumPy arrays to avoid copies when transferring to Taichi.
- Batch multiple photons where possible to amortise kernel launches.
- Prefer vectorised shield intersection functions on the CPU before invoking
  GPU kernels.
- Profile with Taichi's built-in profiler (`ti.profiler_print()`) to identify
  bottlenecks when extending acceleration.

By adhering to this workflow, developers can confidently expand ZapMeNot's
parallel coverage while maintaining the validated physics of the point-kernel
model.
