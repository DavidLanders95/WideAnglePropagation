# Wide-angle multislice propagation for electron microscopy

This repository contains the implementation and numerical workflows accompanying
the paper *Wide-Angle Multislice Propagation for Electron Microscopy: Comparison
of Fresnel, Angular-Spectrum, and Wave-Propagation Methods* by David Landers and
Jean-Luc Rouvière.

The code compares four scalar electron-wave propagation models:

- conventional Fresnel multislice (F-MS);
- angular-spectrum multislice (AS-MS), which replaces the paraxial free-space
  propagator with the exact spherical propagator;
- wave-propagation multislice (WP-MS), which adds a local-medium correction; and
- direct integration of the second-order Klein--Gordon equation as a numerical
  reference.

The reusable implementations are in `wide_angle_propagation/`, while the four
notebooks define the paper benchmarks and figure-generation code.

## Repository contents

| Path | Purpose |
| --- | --- |
| `wide_angle_propagation/propagation_methods.py` | Propagation kernels, multislice methods, and Klein--Gordon ODE reference |
| `wide_angle_propagation/notebook_utils.py` | Plotting, result-file, and notebook workflow helpers |
| `notebooks/01_axel_lubk_verification.ipynb` | Au [100] plane-wave benchmark against the ODE reference |
| `notebooks/02_converge_probe_si.ipynb` | Si [111]/[110] convergent-probe CBED comparison |
| `notebooks/03_convergent_probe_au.ipynb` | Au [100] convergent-probe thickness series |
| `notebooks/04_wpm_binning_diagnostics.ipynb` | WP-MS refractive-index binning diagnostics |
| `notebooks/verification/results/` | Archived numerical reference used by the verification notebook |
| `Paper/` | Manuscript source, compiled paper, and publication figures |
| `tests/` | Dependency-light numerical and public-API tests |

## Quick start: library and tests

Python 3.10 or later is required. The following creates an isolated CPU
environment suitable for importing the library and running the test suite:

```bash
git clone https://github.com/DavidLanders95/WideAnglePropagation.git
cd WideAnglePropagation
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

The core routines can then be imported directly:

```python
from wide_angle_propagation import (
    angular_spectrum_propagation_kernel,
    fresnel_propagation_kernel,
    simulate_fresnel_as,
    simulate_kg_ode_full,
    simulate_wpm,
)
```

Lengths and real-space sampling are expressed in ångströms, electron kinetic
energies in electronvolts, and electrostatic potentials in volts. A sampling
tuple follows array order, `(dy, dx)`.

## Running the paper notebooks

The publication-scale notebooks use JAX and abTEM with CuPy arrays. An NVIDIA
GPU is strongly recommended; the full CBED calculations are memory intensive
and can take substantially longer than the unit tests. Install a JAX and CuPy
build that matches the CUDA version on the host, then install this repository's
notebook dependencies. For a CUDA 12 system, one possible setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "jax[cuda12]" cupy-cuda12x
python -m pip install -e ".[notebooks]"
jupyter lab
```

If the machine uses another CUDA release, follow the JAX and CuPy installation
instructions for that release instead of using the two CUDA 12 packages above.
Confirm that the accelerator is visible before starting a full calculation:

```bash
python -c "import jax; print(jax.devices())"
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

Open Jupyter from the repository root so the notebooks can resolve the package,
archived results, and `Paper/figures/` consistently. Each notebook has a runtime
control cell near the top. In particular, inspect `DEBUG_MODE`,
`LOAD_EXISTING_RESULTS`, `RECOMPUTE_SIMULATION`, and `SAVE_RESULTS_NPZ` before
running all cells. The Si notebook provides a smaller `DEBUG_MODE` workflow;
the publication settings should only be used after the debug run succeeds.

The repository includes the compact Au ODE benchmark used by notebook 01. The
large CBED result archives are not part of the source release and are regenerated
by notebooks 02 and 03. Generated figures are written below `Paper/figures/`.

## Reproducing the reported results

Run the notebooks in numerical order. Notebook 01 validates the propagation
implementations and the ODE reference, notebooks 02 and 03 generate the Si and
Au CBED comparisons, and notebook 04 examines the WP-MS binning approximation.
The notebooks record their numerical controls alongside saved results. Exact
wall times and peak memory depend strongly on the GPU, grid size, number of
slices, and WP-MS bin count.

For a quick installation check, use `pytest -q`; it exercises the analytic
limits and public API without requiring CuPy, abTEM, or a GPU. Passing these
tests verifies the implementation, but does not rerun the publication-scale
simulations.

## Citation

If you use this software, cite the accompanying paper and the archived Zenodo
release. The release DOI should be added here after the Zenodo record is
published. Machine-readable author and repository metadata are provided in
`CITATION.cff`.

## License

The software is released under the [MIT License](LICENSE).
