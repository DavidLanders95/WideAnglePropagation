# Wide-Angle Propagation

Numerical propagation tools for electron-wave simulations, with examples for
multislice, angular-spectrum propagation, wave-propagation multislice (WPM),
and second-order Klein-Gordon propagation.

## What Is Included

- `wide_angle_propagation/propagation_methods.py`: public propagation kernels
  and simulation loops.
- `wide_angle_propagation/notebook_utils.py`: beam/amplitude utilities, plotting
  helpers, and compact result-file helpers used by the notebooks.
- `wide_angle_propagation/ptychography_1d.py`: focused helpers for masked
  pixelwise and physics-informed lattice-site reconstruction, together with
  truth-free calibrated count objectives, non-pickled result persistence, and
  selected-scan side-view caches for one-dimensional glancing-incidence scans.
- `wide_angle_propagation/ptychography_support_contract_1d.py`: immutable,
  digest-bound TARGET/NUISANCE/fixed/below-budget site classification that
  prevents illuminated exterior material from being silently treated as
  known pristine or exposed as recovered structure.
- `wide_angle_propagation/ptychography_atomic_validation_1d.py`: direct
  Kirkland Si voxel quadrature, exact subpixel template caching, and
  provenance-bound Lobato/Kirkland numerical comparisons that deliberately
  make no experimental-validity claim.
- `wide_angle_propagation/ptychography_alignment_1d.py`: truth-isolated,
  coarse-to-fine global alignment, complete-slab candidate rebuilding,
  training/validation data isolation, validation-equivalence summaries, and
  digest-revalidated evidence archives and aligned reconstruction entry points.
- `wide_angle_propagation/ptychography_diagnostics_1d.py`: truth-free,
  dose-scaled local sensitivity diagnostics with explicitly conservative
  interpretation.
- `wide_angle_propagation/ptychography_observability_1d.py`: gauge-free dense
  reference calculations and a bounded, selected-site matrix-free
  Poisson-Fisher/PCG diagnostic with explicit or calibration-bound low-rank
  scan/probe/detector nuisance projection.
- `wide_angle_propagation/ptychography_stochastic_observability_1d.py`:
  callback-based Gaussian screening of all physical-output covariance and
  Fisher-null leakage with simultaneous Monte Carlo bounds and fail-closed
  solver/resource accounting.
- `wide_angle_propagation/ptychography_benchmarks_1d.py`: reproducible,
  truth-isolated detector and forward-model mismatch sweeps with sourced
  acceptance criteria and digest-bound reports.
- `wide_angle_propagation/ptychography_ensemble_1d.py`: multistart basin,
  ambiguity, compact persistence, and trust-gate summaries.
- `wide_angle_propagation/ptychography_workflow_1d.py`: concise experiment,
  reconstruction-comparison, plotting, contract-bound TARGET-only per-update
  GIF (streaming through FFmpeg when available), and interactive-viewer API
  used by the glancing-incidence ptychography notebook.
- `scripts/benchmark_ptychography_1d.py`: synchronized CPU/GPU performance
  harness for the reusable prepared lattice-site runtime.
- `tests/`: regression and behavior tests for the propagation methods.
- `notebooks/figure_generation/01_axel_lubk_verification.ipynb`: Au [100]
  beam-amplitude verification against the full KG ODE reference.
- `notebooks/figure_generation/02_converge_probe_si.ipynb`: Si CBED
  comparison notebook.
- `notebooks/figure_generation/03_convergent_probe_au.ipynb`: Au CBED
  comparison notebook.
- `notebooks/figure_generation/04_wpm_binning_diagnostics.ipynb`: WPM binning
  diagnostic figure notebook.

See `notebooks/README.md` for notebook run notes and output locations.

## Installation

From a local checkout:

```bash
python -m pip install -e ".[dev]"
```

GPU-enabled notebook workflows also require a working CuPy/JAX/abTEM
installation compatible with your CUDA runtime.

The optional glancing-incidence reconstruction workflow uses Optax:

```bash
python -m pip install -e ".[dev,ptychography]"
```

The current scientific limitations, validation gates, and implementation
sequence for the glancing ptychography prototype are tracked in
[`docs/ptychography_robustness.md`](docs/ptychography_robustness.md).

## Ptychography Performance Benchmark

Run a small CPU harness check with:

```bash
python scripts/benchmark_ptychography_1d.py \
  --quick --device cpu --updates 2 --starts 1 \
  --output benchmark_quick.json
```

Omit `--quick` to use the exact default geometry from the maintained
ptychography notebook. A reference GPU run is, for example:

```bash
python scripts/benchmark_ptychography_1d.py \
  --device gpu --precision float64 --updates 500 --starts 5 \
  --output benchmark_notebook_gpu.json
```

The JSON separates one-time geometry construction, data simulation, and eager
prepared-runtime compilation from every optimizer start. All timed device
results are synchronized, update-rate scopes are recorded explicitly, and JAX
preallocation is disabled before JAX is imported. The `--quick` result is an
installation/harness diagnostic and is not comparable with notebook-geometry
performance.

## Minimal Usage

```python
import jax.numpy as jnp

from wide_angle_propagation import (
    angular_spectrum_propagation_kernel,
    simulate_fresnel_as,
)

energy = 300e3
sampling = (0.1, 0.1)
slice_thickness = 2.0
potential = jnp.zeros((4, 64, 64))
probe = jnp.ones((64, 64), dtype=jnp.complex128)

kernel = angular_spectrum_propagation_kernel(
    64, 64, sampling, z=slice_thickness, energy=energy
)
exit_wave, diffraction_pattern, wavefronts = simulate_fresnel_as(
    potential, probe, kernel, slice_thickness, energy
)
```

## Tests

```bash
python scripts/check_static.py
python scripts/check_static.py --enforce-clean-notebooks
pytest tests/test_multislice_method_basics.py
pytest
```

`scripts/check_static.py` validates package syntax, exported names, and the
three maintained notebooks without importing the GPU/scientific runtime stack.
Use `--enforce-clean-notebooks` when you want saved notebook outputs to fail
the check. The pytest suite requires JAX, and some integration tests and
notebooks also require GPU dependencies (`cupy`, JAX with the appropriate
backend, and abTEM data generation).

## Notes

Figure-generating notebooks live in `notebooks/figure_generation/`. They save
generated figures and compact `.npz` results under `notebooks/cbed/results/`,
`notebooks/verification/figures/`, and `Paper/figures/` when the corresponding
save flags are enabled. Standalone LaTeX/TikZ figure sources live under
`Paper/figure_sources/`. Old exploratory notebooks and generated byproducts
that are not part of the maintained paper workflow are collected under
`archive/delete_candidates_2026-07-08/` for review before deletion.
