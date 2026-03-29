# Experiment: Fresnel vs WPM Divergence Threshold Study

## Objective

Determine at what sample thicknesses and beam conditions the **Wave Propagation Method (WPM)** begins to produce significantly different diffraction patterns compared to the traditional **Fresnel (paraxial) propagation** method. The Angular Spectrum method is included as a non-paraxial reference.

The experiment systematically sweeps **energy**, **thickness**, **probe convergence angle**, **material**, and **sample geometry** (bulk vs edge) to map the divergence landscape.

---

## Parameter Grid

| Parameter        | Values                                          | Count |
|------------------|------------------------------------------------|-------|
| **Energy**       | 30, 60, 100, 150, 200 keV                      | 5     |
| **Thickness**    | 1, 2, 5, 10, 20, 50, 75, 100, 150 nm           | 9     |
| **Material**     | Au (FCC, Z=79), Si (diamond, Z=14)              | 2     |
| **Geometry**     | bulk (full periodic), edge (half vacuum)         | 2     |
| **Semiangle**    | 10, 20, 40, 80 mrad                             | 4     |

**Total simulation runs**: 5 × 9 × 2 × 2 × 4 = **720**
**Unique potential builds**: 9 × 2 × 2 = **36** (potential is independent of energy and semiangle)

---

## Implementation

All code should be implemented in a single self-contained Python script: **`wpm/experiment_sweep.py`**

### Environment Setup

```python
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import abtem
from ase.build import bulk, surface
import pandas as pd
import json
from pathlib import Path
import sys
import time

parent = Path(__file__).resolve().parent.parent
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))

from wide_angle_propagation import (
    energy2wavelength,
    fresnel_propagation_kernel,
    angular_spectrum_propagation_kernel,
    simulate_fresnel_as,
    simulate_wpm,
)

abtem.config.set({"device": "cpu"})
abtem.config.set({"precision": "float64"})
```

### 1. Sample Construction Functions

#### `build_bulk_sample(material, thickness_nm, lateral_size_ang=100)`

Returns `(atoms, slice_thickness_ang)`.

**Gold (Au)**:
```python
au_bulk = bulk('Au', 'fcc', a=4.078)
atoms = surface(au_bulk, (1, 1, 0), layers=2)
atoms = abtem.orthogonalize_cell(atoms)
z_periodicity = 4.078 / np.sqrt(2)
atoms.cell[2, 2] = z_periodicity
atoms.pbc = [True, True, True]

nx = int(np.ceil(lateral_size_ang / atoms.cell.lengths()[0]))
ny = int(np.ceil(lateral_size_ang / atoms.cell.lengths()[1]))
nz = int(np.ceil(thickness_ang / atoms.cell.lengths()[2]))
sample = atoms * (nx, ny, nz)
slice_thickness = 2.0  # Angstrom
```

**Silicon (Si)**:
```python
si_bulk = bulk('Si', crystalstructure='diamond', a=5.431)
atoms = surface(si_bulk, (1, 1, 1), layers=3, periodic=True)
atoms = abtem.orthogonalize_cell(atoms)

nx = int(np.ceil(lateral_size_ang / atoms.cell.lengths()[0]))
ny = int(np.ceil(lateral_size_ang / atoms.cell.lengths()[1]))
nz = int(np.ceil(thickness_ang / atoms.cell.lengths()[2]))
sample = atoms * (nx, ny, nz)
slice_thickness = 2.0  # Angstrom
```

Convert thickness: `thickness_ang = thickness_nm * 10.0`

#### `build_potential(atoms, slice_thickness, sampling=0.1)`

```python
potential = abtem.Potential(
    atoms,
    sampling=sampling,
    slice_thickness=slice_thickness,
    parametrization='lobato'
)
potential_array = potential.build(lazy=False).array / slice_thickness
return potential_array, potential
```

The division by `slice_thickness` converts from integrated potential to mean potential per slice, matching the convention used by `simulate_fresnel_as` and `simulate_wpm`.

#### `build_edge_potential(potential_array)`

Create a vacuum-material edge by zeroing the left half of the potential:

```python
potential_edge = potential_array.copy()
nx = potential_edge.shape[2]
potential_edge[:, :, :nx // 2] = 0.0
return potential_edge
```

This creates a sharp boundary at the center column: vacuum on the left, material on the right. The probe should be centered at the boundary to maximize edge-scattering effects.

### 2. Probe Construction

For each `(energy, semiangle)` pair, build a fresh probe matched to the potential grid:

```python
probe = abtem.Probe(
    energy=energy_eV,
    semiangle_cutoff=semiangle_mrad,
    defocus=0,
)
probe.grid.match(potential)  # Match grid to the potential object
probe_array = probe.build(lazy=False).array
sampling = (probe.grid.sampling[0], probe.grid.sampling[1])
```

**For edge geometry**: center the probe at the vacuum-material boundary. This means the probe position should be at `(ny//2, nx//2)` where `nx//2` is the edge column. Since `abtem.Probe` builds centered by default and the edge is at `nx//2`, the default centered probe is already at the boundary.

### 3. Three-Way Comparison

For each simulation run, execute all three propagation methods:

```python
wavelength = energy2wavelength(energy_eV)
gpts = probe_array.shape

# --- Method 1: Fresnel (paraxial) ---
fresnel_kernel = fresnel_propagation_kernel(
    gpts[0], gpts[1], sampling, z=slice_thickness, energy=energy_eV
)
ew_fresnel, dp_fresnel, _ = simulate_fresnel_as(
    potential_array, probe_array, fresnel_kernel, slice_thickness, energy_eV
)

# --- Method 2: Angular Spectrum (exact vacuum propagation) ---
as_kernel = angular_spectrum_propagation_kernel(
    gpts[0], gpts[1], sampling, z=slice_thickness, energy=energy_eV
)
ew_as, dp_as, _ = simulate_fresnel_as(
    potential_array, probe_array, as_kernel, slice_thickness, energy_eV
)

# --- Method 3: WPM (wide-angle corrected) ---
ew_wpm, dp_wpm, _ = simulate_wpm(
    potential_array, probe_array, slice_thickness, energy_eV, sampling,
    n_bins=128, power_spacing=2.0
)
```

**Important**: `simulate_fresnel_as` accepts any pre-computed propagation kernel — both Fresnel and Angular Spectrum kernels work with the same function. WPM uses its own `simulate_wpm` function which builds the kernel internally.

**Memory optimization**: The third return value (wavefronts stack) is discarded (`_`) to avoid storing all intermediate slices. Only retain wavefronts for selected diagnostic runs (see Section 7).

### 4. Quantitative Metrics

Compute these metrics for each method pair (Fresnel vs WPM, AS vs WPM, Fresnel vs AS):

```python
def compute_metrics(ew_a, ew_b, dp_a, dp_b, sampling, wavelength):
    """
    Compare two simulation outputs. Returns dict of scalar metrics.
    
    Parameters:
        ew_a, ew_b: exit wave arrays (complex, shape [ny, nx])
        dp_a, dp_b: diffraction pattern arrays (real, shape [ny, nx])
        sampling: (dx, dy) in Angstrom
        wavelength: in Angstrom
    """
    metrics = {}
    
    # --- Diffraction pattern metrics ---
    I_a = np.asarray(dp_a, dtype=np.float64)
    I_b = np.asarray(dp_b, dtype=np.float64)
    
    # Normalized RMSE of diffraction intensity
    denom = I_a.max() if I_a.max() > 0 else 1.0
    metrics['dp_nrmse'] = float(np.sqrt(np.mean((I_b - I_a) ** 2)) / denom)
    
    # Peak intensity ratio
    peak_a = I_a.max() if I_a.max() > 0 else 1.0
    peak_b = I_b.max() if I_b.max() > 0 else 1.0
    metrics['dp_peak_ratio'] = float(peak_b / peak_a)
    
    # Pearson correlation of flattened diffraction patterns
    corr = np.corrcoef(I_a.ravel(), I_b.ravel())[0, 1]
    metrics['dp_pearson'] = float(corr)
    
    # R-factor (crystallography-style): sum|sqrt(I_b) - sqrt(I_a)| / sum|sqrt(I_a)|
    sqrt_a = np.sqrt(np.maximum(I_a, 0))
    sqrt_b = np.sqrt(np.maximum(I_b, 0))
    denom_r = sqrt_a.sum() if sqrt_a.sum() > 0 else 1.0
    metrics['dp_r_factor'] = float(np.sum(np.abs(sqrt_b - sqrt_a)) / denom_r)
    
    # High-angle divergence: NRMSE outside 50 mrad
    try:
        ny, nx = I_a.shape
        fx = np.fft.fftfreq(nx, float(sampling[0]))
        fy = np.fft.fftfreq(ny, float(sampling[1]))
        # For fftshifted patterns, reconstruct angle from pixel position
        cx, cy = nx // 2, ny // 2
        Y, X = np.mgrid[:ny, :nx]
        angle_map = np.sqrt(
            ((X - cx) / (nx * float(sampling[0])) * float(wavelength) * 1000) ** 2 +
            ((Y - cy) / (ny * float(sampling[1])) * float(wavelength) * 1000) ** 2
        )
        high_mask = angle_map > 50.0  # mrad
        if high_mask.sum() > 0:
            denom_ha = I_a[high_mask].max() if I_a[high_mask].max() > 0 else 1.0
            metrics['dp_nrmse_high_angle'] = float(
                np.sqrt(np.mean((I_b[high_mask] - I_a[high_mask]) ** 2)) / denom_ha
            )
        else:
            metrics['dp_nrmse_high_angle'] = 0.0
    except Exception:
        metrics['dp_nrmse_high_angle'] = float('nan')
    
    # --- Exit wave metrics ---
    ew_a_np = np.asarray(ew_a)
    ew_b_np = np.asarray(ew_b)
    
    # Exit wave amplitude NRMSE
    amp_a = np.abs(ew_a_np)
    amp_b = np.abs(ew_b_np)
    denom_ew = amp_a.max() if amp_a.max() > 0 else 1.0
    metrics['ew_amp_nrmse'] = float(np.sqrt(np.mean((amp_b - amp_a) ** 2)) / denom_ew)
    
    # Exit wave phase RMSE (wrapped difference)
    phase_diff = np.angle(ew_b_np * np.conj(ew_a_np))  # wrapped phase difference
    metrics['ew_phase_rmse'] = float(np.sqrt(np.mean(phase_diff ** 2)))
    
    # Intensity conservation check
    metrics['total_intensity_a'] = float(np.sum(np.abs(ew_a_np) ** 2))
    metrics['total_intensity_b'] = float(np.sum(np.abs(ew_b_np) ** 2))
    
    return metrics
```

**Metric definitions:**

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `dp_nrmse` | Normalized RMSE of diffraction pattern intensity | 0 = identical, higher = more divergent |
| `dp_peak_ratio` | Ratio of peak diffraction intensities | 1.0 = identical peak |
| `dp_pearson` | Pearson correlation of diffraction patterns | 1.0 = perfectly correlated |
| `dp_r_factor` | Crystallographic R-factor on √I | 0 = identical, standard quality metric |
| `dp_nrmse_high_angle` | NRMSE only for angles > 50 mrad | Isolates wide-angle scattering differences |
| `ew_amp_nrmse` | Exit wave amplitude NRMSE | 0 = identical exit waves |
| `ew_phase_rmse` | Wrapped phase difference RMSE (radians) | 0 = identical phases |
| `total_intensity_a/b` | Sum of |ψ|² for each method | Should be approximately equal (conservation) |

### 5. Main Experiment Loop

```python
ENERGIES = [30e3, 60e3, 100e3, 150e3, 200e3]           # eV
THICKNESSES_NM = [1, 2, 5, 10, 20, 50, 75, 100, 150]   # nm
MATERIALS = ['Au', 'Si']
GEOMETRIES = ['bulk', 'edge']
SEMIANGLES = [10.0, 20.0, 40.0, 80.0]                   # mrad

SAMPLING = 0.1       # Angstrom
LATERAL_SIZE = 100   # Angstrom
N_BINS = 128
POWER_SPACING = 2.0

OUTPUT_DIR = Path("wpm/experiment_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUTPUT_DIR / "sweep_results.csv"
```

**Loop structure** — outer loop builds potential once, inner loop varies beam parameters:

```python
all_results = []

for material in MATERIALS:
    for thickness_nm in THICKNESSES_NM:
        thickness_ang = thickness_nm * 10.0
        slice_thickness = 2.0  # Angstrom

        # Build sample and potential (once per material+thickness)
        atoms, _ = build_bulk_sample(material, thickness_nm, LATERAL_SIZE)
        potential_array, potential_obj = build_potential(atoms, slice_thickness, SAMPLING)

        for geometry in GEOMETRIES:
            if geometry == 'edge':
                pot = build_edge_potential(potential_array)
            else:
                pot = potential_array

            for energy_eV in ENERGIES:
                wavelength = float(energy2wavelength(energy_eV))

                for semiangle in SEMIANGLES:
                    t0 = time.time()

                    # Build probe
                    probe = abtem.Probe(
                        energy=energy_eV,
                        semiangle_cutoff=semiangle,
                        defocus=0,
                    )
                    probe.grid.match(potential_obj)
                    probe_array = probe.build(lazy=False).array
                    sampling = (float(probe.grid.sampling[0]),
                                float(probe.grid.sampling[1]))
                    gpts = probe_array.shape

                    # Run three methods
                    fresnel_H = fresnel_propagation_kernel(
                        gpts[0], gpts[1], sampling,
                        z=slice_thickness, energy=energy_eV
                    )
                    as_H = angular_spectrum_propagation_kernel(
                        gpts[0], gpts[1], sampling,
                        z=slice_thickness, energy=energy_eV
                    )

                    ew_f, dp_f, _ = simulate_fresnel_as(
                        pot, probe_array, fresnel_H, slice_thickness, energy_eV
                    )
                    ew_as, dp_as, _ = simulate_fresnel_as(
                        pot, probe_array, as_H, slice_thickness, energy_eV
                    )
                    ew_wpm, dp_wpm, _ = simulate_wpm(
                        pot, probe_array, slice_thickness, energy_eV, sampling,
                        n_bins=N_BINS, power_spacing=POWER_SPACING
                    )

                    # Compute metrics for all three pairs
                    m_fw = compute_metrics(ew_f, ew_wpm, dp_f, dp_wpm,
                                           sampling, wavelength)
                    m_fa = compute_metrics(ew_f, ew_as, dp_f, dp_as,
                                           sampling, wavelength)
                    m_aw = compute_metrics(ew_as, ew_wpm, dp_as, dp_wpm,
                                           sampling, wavelength)

                    elapsed = time.time() - t0

                    row = {
                        'material': material,
                        'thickness_nm': thickness_nm,
                        'geometry': geometry,
                        'energy_keV': energy_eV / 1e3,
                        'semiangle_mrad': semiangle,
                        'elapsed_s': elapsed,
                    }
                    # Prefix each metric with the comparison pair
                    for k, v in m_fw.items():
                        row[f'fresnel_vs_wpm_{k}'] = v
                    for k, v in m_fa.items():
                        row[f'fresnel_vs_as_{k}'] = v
                    for k, v in m_aw.items():
                        row[f'as_vs_wpm_{k}'] = v

                    all_results.append(row)

                    print(f"[{len(all_results)}/720] "
                          f"{material} {geometry} {thickness_nm}nm "
                          f"{energy_eV/1e3:.0f}keV {semiangle}mrad "
                          f"NRMSE(F-W)={m_fw['dp_nrmse']:.4f} "
                          f"R(F-W)={m_fw['dp_r_factor']:.4f} "
                          f"({elapsed:.1f}s)")

            # Checkpoint after each (material, thickness, geometry) batch
            df = pd.DataFrame(all_results)
            df.to_csv(RESULTS_CSV, index=False)
            print(f"  -> Checkpointed {len(all_results)} results to {RESULTS_CSV}")

# Final save
df = pd.DataFrame(all_results)
df.to_csv(RESULTS_CSV, index=False)
print(f"\nDone. {len(all_results)} results saved to {RESULTS_CSV}")
```

### 6. Visualization & Analysis

After the sweep completes, generate summary visualizations. This can be a separate script (`wpm/experiment_plots.py`) or appended to the sweep script.

#### 6a. Divergence Heatmaps

For each `(material, geometry)` combination and each semiangle, create a heatmap of `dp_nrmse` with thickness on the y-axis and energy on the x-axis:

```python
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

fig, axes = plt.subplots(len(MATERIALS), len(GEOMETRIES),
                          figsize=(12, 8), squeeze=False)

for i, material in enumerate(MATERIALS):
    for j, geometry in enumerate(GEOMETRIES):
        sub = df[(df.material == material) & (df.geometry == geometry)
                 & (df.semiangle_mrad == 20.0)]  # Pick a reference semiangle
        pivot = sub.pivot(index='thickness_nm', columns='energy_keV',
                          values='fresnel_vs_wpm_dp_nrmse')
        ax = axes[i, j]
        im = ax.imshow(pivot.values, aspect='auto',
                        norm=LogNorm(vmin=1e-4, vmax=1.0),
                        cmap='inferno', origin='lower')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{e:.0f}" for e in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{t}" for t in pivot.index])
        ax.set_xlabel("Energy (keV)")
        ax.set_ylabel("Thickness (nm)")
        ax.set_title(f"{material} {geometry}")
        fig.colorbar(im, ax=ax, label="NRMSE (Fresnel vs WPM)")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "heatmap_nrmse_fresnel_vs_wpm.pdf", dpi=150)
```

Generate similar heatmaps for:
- `fresnel_vs_wpm_dp_r_factor` (R-factor)
- `fresnel_vs_wpm_dp_nrmse_high_angle` (high-angle region only)
- `as_vs_wpm_dp_nrmse` (AS vs WPM — should be smaller than Fresnel vs WPM)

#### 6b. Semiangle Dependence Line Plots

For each material, plot NRMSE vs thickness with semiangle as the line color:

```python
fig, axes = plt.subplots(1, len(MATERIALS), figsize=(14, 5))

for i, material in enumerate(MATERIALS):
    ax = axes[i]
    for semi in SEMIANGLES:
        sub = df[(df.material == material) & (df.geometry == 'bulk')
                 & (df.semiangle_mrad == semi) & (df.energy_keV == 100)]
        ax.semilogy(sub.thickness_nm, sub.fresnel_vs_wpm_dp_nrmse,
                     'o-', label=f'{semi} mrad')
    ax.set_xlabel("Thickness (nm)")
    ax.set_ylabel("NRMSE (Fresnel vs WPM)")
    ax.set_title(f"{material} — 100 keV, bulk")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "semiangle_dependence.pdf", dpi=150)
```

#### 6c. Edge vs Bulk Comparison

```python
fig, axes = plt.subplots(1, len(MATERIALS), figsize=(14, 5))

for i, material in enumerate(MATERIALS):
    ax = axes[i]
    for geom in GEOMETRIES:
        sub = df[(df.material == material) & (df.geometry == geom)
                 & (df.semiangle_mrad == 20.0) & (df.energy_keV == 100)]
        ax.semilogy(sub.thickness_nm, sub.fresnel_vs_wpm_dp_nrmse,
                     'o-', label=f'{geom}')
    ax.set_xlabel("Thickness (nm)")
    ax.set_ylabel("NRMSE (Fresnel vs WPM)")
    ax.set_title(f"{material} — 100 keV, 20 mrad")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "edge_vs_bulk.pdf", dpi=150)
```

#### 6d. Divergence Threshold Table

Find the thickness where NRMSE first exceeds 5% and 10% for each condition:

```python
thresholds = []
for material in MATERIALS:
    for geometry in GEOMETRIES:
        for energy in ENERGIES:
            for semi in SEMIANGLES:
                sub = df[(df.material == material) & (df.geometry == geometry)
                         & (df.energy_keV == energy/1e3)
                         & (df.semiangle_mrad == semi)].sort_values('thickness_nm')
                nrmse = sub.fresnel_vs_wpm_dp_nrmse.values
                thick = sub.thickness_nm.values

                t5 = thick[nrmse > 0.05][0] if (nrmse > 0.05).any() else '>150'
                t10 = thick[nrmse > 0.10][0] if (nrmse > 0.10).any() else '>150'

                thresholds.append({
                    'material': material, 'geometry': geometry,
                    'energy_keV': energy/1e3, 'semiangle_mrad': semi,
                    'threshold_5pct_nm': t5, 'threshold_10pct_nm': t10,
                })

thresh_df = pd.DataFrame(thresholds)
thresh_df.to_csv(OUTPUT_DIR / "divergence_thresholds.csv", index=False)
print(thresh_df.to_string())
```

### 7. Diagnostic Runs (Optional)

For a small subset of "interesting" runs (e.g., where NRMSE first crosses 5%), re-run with wavefronts enabled to produce slice-by-slice comparison plots:

```python
# Identify interesting runs from the threshold table
interesting = thresh_df[thresh_df.threshold_5pct_nm != '>150']

for _, row in interesting.iterrows():
    # Re-run with wavefronts stored
    # ... (same as main loop but keep the third return value)
    # Save slice-by-slice intensity/phase comparison plots
    # following the pattern from wpm/thick_gold_structure.ipynb
```

This produces the detailed cross-section plots (intensity + phase per slice) that are already implemented in existing notebooks.

---

## Verification Checklist

Run these checks before launching the full sweep:

1. **Smoke test**: Run a single case `(Au, bulk, 5nm, 100keV, 20mrad)`. Confirm:
   - All three methods return finite, non-NaN arrays
   - Total intensity is conserved within 1% for each method
   - Metrics are finite and in expected ranges
   - Runtime is reasonable (estimate total time from this)

2. **Vacuum sanity**: Set `potential_array = np.zeros_like(potential_array)` for one run.
   - All three methods should produce identical results (`dp_nrmse ≈ 0`, `dp_pearson ≈ 1.0`)
   - This validates the comparison pipeline itself

3. **Thin-sample check**: At 1 nm thickness, all methods should agree closely (`dp_nrmse < 0.01`) for all energies.
   - If they don't, there may be a normalization or convention mismatch

4. **Edge geometry check**: Print `potential_array[:, :, nx//4]` (vacuum region) and `potential_array[:, :, 3*nx//4]` (material region). The vacuum side should be all zeros, the material side should have nonzero values.

5. **Monotonicity sanity**: For any fixed `(material, energy, semiangle)`, NRMSE should generally increase with thickness. If it oscillates wildly, investigate numerical issues.

---

## Expected Outcomes

Based on theory and preliminary observations:

1. **Fresnel vs WPM divergence increases with**:
   - Increasing thickness (more cumulative error from paraxial approximation)
   - Increasing semiangle (wider angles violate the Fresnel small-angle assumption)
   - Higher atomic number Z (stronger scattering potentials → larger angular deflections)
   - Lower energy (shorter wavelength → Fresnel approximation valid over narrower angular range relative to the wavelength)

2. **Edge geometry should amplify divergence** because:
   - The vacuum-material boundary creates sharp lateral potential gradients
   - Large-angle scattering off the edge is precisely the regime where WPM corrects Fresnel
   - Fresnel cannot properly handle the angular spectrum redistribution at sharp boundaries

3. **Angular Spectrum should be intermediate**: AS is exact for vacuum propagation but approximate for material propagation (uses the same phase-grating approximation), so AS vs WPM differences should be smaller than Fresnel vs WPM. AS vs Fresnel differences isolate the kinematic (propagator) error from the dynamic (material interaction) error.

---

## Output Files

All outputs go to `wpm/experiment_results/`:

| File | Description |
|------|-------------|
| `sweep_results.csv` | Full results table (720 rows × ~30 columns) |
| `divergence_thresholds.csv` | Thickness where NRMSE crosses 5% and 10% |
| `heatmap_nrmse_fresnel_vs_wpm.pdf` | Main result: NRMSE heatmaps by material/geometry |
| `heatmap_rfactor_fresnel_vs_wpm.pdf` | R-factor heatmaps |
| `heatmap_high_angle.pdf` | High-angle (>50 mrad) NRMSE heatmaps |
| `semiangle_dependence.pdf` | NRMSE vs thickness, parameterized by semiangle |
| `edge_vs_bulk.pdf` | Edge vs bulk geometry comparison |

---

## Practical Notes

- **Runtime estimate**: Each simulation at 0.1 Å sampling on a 1000×1000 grid with 750 slices (150 nm / 2 Å) takes ~minutes on GPU. The full 720-run sweep may take **12-48 hours** depending on hardware. Consider:
  - Starting with a **coarse sweep**: 3 energies × 5 thicknesses × 2 materials × 1 geometry × 2 semiangles = 60 runs (~1-4 hours)
  - Extending to the full grid only after verifying the coarse results are sensible
  
- **Memory**: A single wavefronts stack for 150 nm at 0.1 Å sampling ≈ 750 slices × 1000 × 1000 × 16 bytes ≈ 12 GB. **Discard wavefronts** (`_`) in the main loop; only keep exit wave and diffraction pattern.

- **Checkpointing**: Results are saved to CSV after each `(material, thickness, geometry)` batch. If the script crashes, it can be restarted from where it left off by checking which rows already exist in the CSV and skipping them.

- **Restart logic** (implement in the main loop):
  ```python
  if RESULTS_CSV.exists():
      existing = pd.read_csv(RESULTS_CSV)
      done_keys = set(zip(existing.material, existing.thickness_nm,
                          existing.geometry, existing.energy_keV,
                          existing.semiangle_mrad))
  else:
      done_keys = set()
  
  # In the loop:
  key = (material, thickness_nm, geometry, energy_eV/1e3, semiangle)
  if key in done_keys:
      continue
  ```

- **JAX compilation**: The first run at each grid size will be slower due to JIT compilation. Subsequent runs with the same grid size will be fast. Since the grid size only changes with (material, thickness), most of the inner loop benefits from cached compilation. Expect a warmup penalty of ~30-60s per unique grid size.

---

## Reference Files

| File | Purpose |
|------|---------|
| `wide_angle_propagation/propagation.py` | All propagation functions: `simulate_fresnel_as`, `simulate_wpm`, `fresnel_propagation_kernel`, `angular_spectrum_propagation_kernel`, `energy2wavelength`, `electron_refractive_index` |
| `wide_angle_propagation/__init__.py` | Public API (re-exports from `propagation.py`) |
| `wpm/thick_gold_structure.ipynb` | Gold sample construction template, Fresnel vs WPM comparison and plotting patterns |
| `wpm/thick_silicon.ipynb` | Silicon sample construction template with frozen phonons |
| `wpm/CBED_Si_111.ipynb` | Silicon (111) CBED with GPU-accelerated abtem |
| `wpm/wpm_adaptive_comparison.ipynb` | WPM binning diagnostics: bin maps, weight visualization, error analysis |
| `wpm/cbed.ipynb` | Five-method comparison (Fresnel, AS, WPM, multislice, Lippmann-Schwinger) |

---

## Key API Details

### `simulate_fresnel_as(potential, probe, prop_kernel, slice_thickness, energy)`
- `potential`: `(N, ny, nx)` — N slices of mean electrostatic potential (V/Å → already divided by slice_thickness)
- `probe`: `(ny, nx)` — complex initial wavefront
- `prop_kernel`: `(ny, nx)` — pre-computed Fresnel or Angular Spectrum kernel
- `slice_thickness`: float — slice thickness in Ångstöms
- `energy`: float — beam energy in eV
- **Returns**: `(exit_wave, diffraction_pattern, wavefronts_stack)`

### `simulate_wpm(potential, probe, slice_thickness, energy, sampling, n_bins=128, power_spacing=2.0)`
- `potential`: same as above
- `probe`: same
- `slice_thickness`: float in Å
- `energy`: float in eV
- `sampling`: `(dx, dy)` tuple in Å
- `n_bins`: number of refractive index bins (default 128)
- `power_spacing`: polynomial bin spacing exponent (default 2.0)
- **Returns**: `(exit_wave, diffraction_pattern, wavefronts_stack)`

### `fresnel_propagation_kernel(n, m, ps, z, energy)`
- `n, m`: grid dimensions
- `ps`: `(dx, dy)` sampling in Å
- `z`: propagation distance (slice thickness) in Å
- `energy`: beam energy in eV
- **Returns**: `(n, m)` complex kernel array

### `angular_spectrum_propagation_kernel(n, m, ps, z, energy)`
- Same signature as fresnel. Uses exact `kz = sqrt(k² - kx² - ky²)` instead of paraxial approximation.

### `energy2wavelength(energy)`
- Input: energy in eV
- Output: relativistic de Broglie wavelength in Å
