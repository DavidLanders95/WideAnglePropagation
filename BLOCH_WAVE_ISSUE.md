# Issue: Extract, test, and validate the Bloch wave eigenvalue solver

## Context

This repo implements wave propagation methods (Fresnel multislice, Angular Spectrum, WPM) for electron microscopy simulation. We also have a **Bloch wave dynamical diffraction solver** that currently lives *only* inside a Jupyter notebook (`wpm/au_axel_lubk_verification.ipynb`, cell ~22). It needs to be extracted into the package, properly tested, and used to generate publication figures.

The Bloch solver computes beam amplitudes vs crystal thickness by:
1. Building a 3D reciprocal-lattice beam basis {(h,k,l)} for an FCC crystal
2. Assembling the dynamical structure matrix (diagonal = excitation errors, off-diagonal = Fourier coefficients of the potential using Lobato scattering factors)
3. Solving the eigenvalue problem on GPU via `cupy.linalg.eigh`
4. Propagating through thickness to get diffracted beam amplitudes

The reference we compare against is **Rother & Scheerschmidt (2009)** — their Figure 3 shows Klein-Gordon ODE solutions for Au at 300 keV, which is the "ground truth" our methods must match.

## Environment

```bash
source /nobackup/dl277493/temgym_core/bin/activate
pip install -e .  # installs wide_angle_propagation package
```

Key dependencies: `jax`, `jaxlib` (GPU), `cupy`, `numpy`, `scipy`, `ase`, `abtem`, `matplotlib`.

**GPU required** for the eigenvalue solve (CuPy). Tests that need GPU should be marked with `@pytest.mark.gpu`.

Run tests with: `pytest tests/ -v`

## Task 1: Extract the Bloch solver into `wide_angle_propagation/bloch.py`

Extract `solve_bloch_wave_gpu` and its helper `_structure_factor` from the notebook into a new module `wide_angle_propagation/bloch.py`.

### Current function location
- **Notebook**: `wpm/au_axel_lubk_verification.ipynb`, the cell containing `def solve_bloch_wave_gpu(`
- The function uses these imports: `numpy`, `cupy`, `scipy.special.factorial`, and Lobato scattering factor coefficients (hardcoded inside the function)

### Function signature to preserve
```python
def solve_bloch_wave_gpu(
    g_max_zolz, g_max_holz, l_max, n_beams_max,
    *, atoms, wavelength, x,
    paper_00=None, paper_028=None,
    include_eigensystem=False,
    include_structure_samples=False,
) -> dict:
```

### Requirements
- Move the function verbatim first, then clean up imports
- Add it to `wide_angle_propagation/__init__.py` exports
- The notebook cell should then `from wide_angle_propagation.bloch import solve_bloch_wave_gpu`
- Do NOT change the physics/math — only reorganize code

## Task 2: Write tests in `tests/test_bloch.py`

Create `tests/test_bloch.py` with these test cases:

### 2a. Structure factor tests (no GPU needed)
- For FCC with 4 atoms at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5):
  - Even-parity reflections (h+k+l all even, e.g. (2,0,0), (1,1,1)): structure factor = 4.0
  - Mixed-parity reflections (e.g. (1,0,0), (2,1,0)): structure factor = 0.0
- Test `_structure_factor` directly (may need to make it public or test via the solver)

### 2b. Unitarity / current conservation (GPU)
- Run the solver for Au FCC, 300 keV, with reasonable parameters (e.g. `g_max_zolz=10, g_max_holz=15, l_max=8, n_beams_max=5000`)
- The eigenvector matrix from `eigh` should satisfy unitarity: `max|I - C†C| < 1e-10`
- Request `include_eigensystem=True` and verify

### 2c. Thickness=0 gives unit amplitude for central beam
- At zero thickness, the [0,0] beam amplitude should be ~1.0 (incident beam)
- `result['amp_00_coh'][0]` should be `pytest.approx(1.0, abs=0.01)`

### 2d. RMSE against reference data (GPU, regression test)
- Use the Klein-Gordon ODE reference data from the notebook (the raw string data for `raw_data_Au_Beam_0_0_Klein_Gordon_MS` and `raw_data_Au_Beam_0_28_Klein_Gordon_FWD`)
- Extract it into a fixture or a small data file in `tests/`
- With best parameters (`g_max_zolz=15, g_max_holz=25, l_max=10, n_beams_max=12000`):
  - `result['rmse_00']` should be < 0.05
  - `result['rmse_028']` should be < 0.05
  - `result['rmse_avg']` should be < 0.03
- This is a regression test — it validates the solver hasn't broken, not that it matches perfectly

### 2e. Beam count consistency
- `result['n_beams'] == result['n_zolz'] + result['n_holz']`
- With small parameters, n_beams should be < n_beams_max

### Test markers
```python
import pytest
gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy/GPU not available")
```

## Task 3: Generate Figure 2 as a standalone script

Create `scripts/generate_figure2.py` (or `wpm/generate_figure2.py`) that:

1. Sets up Au FCC crystal (a=4.08 Å, 300 keV, ase `bulk("Au", "fcc", a=4.08, cubic=True)`)
2. Runs the Bloch solver with best parameters: `g_max_zolz=15, g_max_holz=25, l_max=10, n_beams_max=12000`
3. Runs Fresnel multislice and Angular Spectrum multislice using `simulate_fresnel_as` from `wide_angle_propagation.propagation` (follow the pattern in the notebook — bootstrap over 5 lattice parameter samples with 95% CI)
4. Runs WPM using `simulate_wpm` from `wide_angle_propagation.propagation`
5. Plots a 2-panel figure:
   - **Left panel**: [0,0] beam amplitude vs thickness (unit cells 0-25)
   - **Right panel**: [0,28] beam amplitude vs thickness
   - Lines: Klein-Gordon ODE (green), Klein-Gordon MS from paper (green dashed), Fresnel MS (blue), Angular Spectrum MS (blue dashed), WPM (red), Bloch wave (orange or black)
   - 95% confidence interval bands for bootstrap methods
6. Saves to `Paper/Au_beam_amplitudes.pdf` (this is what the LaTeX `\includegraphics` references)

### Reference data
The Klein-Gordon ODE solutions are hardcoded as CSV strings in the notebook. They were digitized from Rother & Scheerschmidt (2009), Figure 3. Extract them into `wide_angle_propagation/reference_data.py` or a JSON/CSV file.

### Crystal and simulation parameters
```python
a = 4.08  # Å, Au lattice parameter
energy = 300e3  # eV
gpts = (128, 128, 2)  # coarse grid
n_cells = range(0, 26)  # 0 to 25 unit cells thickness
n_slices = 2  # slices per unit cell
```

## Acceptance criteria

- [ ] `wide_angle_propagation/bloch.py` exists with `solve_bloch_wave_gpu` and `_structure_factor`
- [ ] `from wide_angle_propagation.bloch import solve_bloch_wave_gpu` works
- [ ] `pytest tests/test_bloch.py -v` passes (skip GPU tests gracefully if no GPU)
- [ ] Structure factor test passes without GPU
- [ ] RMSE regression test: `rmse_avg < 0.03` for best parameters
- [ ] Figure generation script runs end-to-end and produces PDF
- [ ] The figure shows all methods (Fresnel MS, AS MS, WPM, Bloch, paper ODE reference)
- [ ] No changes to the physics/math in the solver — only code reorganization

## Files to create/modify

| Action | File |
|--------|------|
| Create | `wide_angle_propagation/bloch.py` |
| Create | `tests/test_bloch.py` |
| Create | `scripts/generate_figure2.py` |
| Modify | `wide_angle_propagation/__init__.py` (add bloch exports) |
| Modify | `wpm/au_axel_lubk_verification.ipynb` (import from package instead of inline) |
| Optionally create | `wide_angle_propagation/reference_data.py` or `data/reference_au_300kev.json` |

## Important physics notes

- The solver uses **Lobato parametrization** for electron scattering factors (5 Gaussians + 5 Lorentzians), coefficients hardcoded for Au (Z=79)
- Excitation error: $s_g = \frac{K_0^2 - |\mathbf{K}_0 + \mathbf{g}|^2}{2K_0}$ where $K_0 = 1/\lambda$
- Structure matrix element: $A_{g,g'} = \frac{2m_e e}{\\hbar^2} \frac{V_{g-g'}}{\\Omega}$ with relativistic correction
- The `[0,28]` reflection at 135 mrad is what distinguishes WPM from paraxial methods — it's the key physics result
- FCC selection rules: only reflections where h,k,l are all even or all odd have nonzero structure factors
