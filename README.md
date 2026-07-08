# Wide-Angle Propagation

Numerical propagation tools for electron-wave simulations, with examples for
multislice, angular-spectrum propagation, wave-propagation multislice (WPM),
and second-order Klein-Gordon propagation.

## What Is Included

- `wide_angle_propagation/propagation_methods.py`: public propagation kernels
  and simulation loops.
- `wide_angle_propagation/notebook_utils.py`: beam/amplitude utilities, plotting
  helpers, and compact result-file helpers used by the notebooks.
- `tests/`: regression and behavior tests for the propagation methods.
- `notebooks/verification/01_axel_lubk_verification.ipynb`: Au [100]
  beam-amplitude verification against the full KG ODE reference.
- `notebooks/cbed/02_converge_probe_si.ipynb`: Si CBED comparison notebook.
- `notebooks/cbed/03_convergent_probe_au.ipynb`: Au CBED comparison notebook.

See `notebooks/README.md` for notebook run notes and output locations.

## Installation

From a local checkout:

```bash
python -m pip install -e ".[dev]"
```

GPU-enabled notebook workflows also require a working CuPy/JAX/abTEM
installation compatible with your CUDA runtime.

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

The notebooks save generated figures and compact `.npz` results under
`notebooks/cbed/results/` and `Paper/figures/` when the corresponding save
flags are enabled.
