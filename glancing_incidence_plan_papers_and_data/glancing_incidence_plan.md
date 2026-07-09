# Project plan: differentiable sideview glancing-incidence electron scattering

## Core objective

Build a differentiable forward-modeling framework for a simplified **sideview glancing-incidence SEM geometry**.

Coordinate convention:

```math
x = \text{vertical coordinate}, \qquad z = \text{horizontal propagation coordinate}.
```

The example geometry is:

```text
upright input line        tilted atomic sample        upright output line
left side          --->   glancing interaction  --->  right side
```

The field is modeled as:

```math
\psi(x,z),
```

so the sideview problem is treated as a **2D wave problem represented on 1D lines**. In this geometry, direct line-to-line propagation should use **cylindrical sideview wave behavior**, with asymptotic amplitude scaling like \(1/\sqrt{R}\), rather than 3D spherical \(1/R\) behavior.

The implementation should be differentiable in JAX with respect to:

```text
atom strengths
atom positions
sample tilt
beam tilt
projected potential
slice potential
```

## Source material for this plan

This folder contains the implementation notes and references that should be used when the corresponding method is implemented:

```text
glancing_incidence_plan_papers_and_data/
  titled_plane_baseline.ipynb
  bidirectional_wpm_paper.pdf
  bidirectional_wpm_thesis.pdf
```

Use `titled_plane_baseline.ipynb` for the direct tilted-plane propagation prototype. That notebook uses a Rayleigh-Sommerfeld style propagation integral and Simpson quadrature to validate propagation between tilted planes. The sideview implementation should adapt the same idea to 1D source and detector lines.

Use `bidirectional_wpm_paper.pdf` and `bidirectional_wpm_thesis.pdf` when implementing the bidirectional WPM update. The plan below identifies the state variables and tests, but the actual interface-coupling formula should be taken from those references rather than invented from the high-level description here.

## Methods to implement

The project will compare four forward models:

```text
1. Fresnel baseline
2. Single-slice cylindrical projected-potential model
3. Bidirectional WPM
4. Bidirectional Pade square-root BPM / multislice
```

The fourth method is the main advanced method. It should explicitly carry forward and backward waves.

## Method 1: Fresnel baseline

### Purpose

The Fresnel model is the fast paraxial baseline. It should use the same sideview \(x,z\) convention as the other methods, but it should keep the paraxial free-space kernel. This gives a control case for quantifying how much wide-angle and bidirectional physics matters.

The first implementation should be a slice-based 1D Fresnel multislice baseline.

### API

```python
simulate_glancing_fresnel_baseline_1d(
    input_wave,
    potential_slices,
    dx,
    dz,
    energy,
    *,
    input_tilt=0.0,
    return_diagnostics=True,
)
```

### Pipeline

```text
1. Build the input wave on the upright left edge, optionally with a phase ramp for beam tilt.
2. Propagate to the sample using a 1D Fresnel propagator - and apply a phase grating for each potential slice.
4. Continue propagating until the upright output edge.
5. Return output wave, intensity, and diagnostics.
```

### Core helpers

```python
get_frequencies_1d
fresnel_propagation_kernel_1d
```

### Tests

```text
vacuum propagation preserves a plane wave
Gaussian beam propagates in the expected direction
finite JAX gradients through potential strength
```

## Method 2: Single-slice cylindrical projected-potential model

### Purpose

This is 1D single slice allegory.

```text
input line
  -> cylindrical propagation to tilted sample line
  -> projected-potential interaction
  -> cylindrical propagation to output line
```

Use `glancing_incidence_plan_papers_and_data/titled_plane_baseline.ipynb` as the closest prototype. The notebook is 3D and plane-based, whereas this model should be 2D and line-based, but the same direct propagation-integral structure applies.

### API

```python
simulate_single_slice_cylindrical_1d(
    input_wave,
    input_line,
    sample_line,
    output_line,
    projected_potential,
    energy,
    *,
    quadrature="trapezoid",
    green_kernel="cylindrical_asymptotic",
    steering="specular",
    return_diagnostics=True,
)
```

### Cylindrical kernel

For two sideview points \(\mathbf{r}_a=(x_a,z_a)\) and \(\mathbf{r}_b=(x_b,z_b)\),

```math
R = |\mathbf{r}_b-\mathbf{r}_a|.
```

Use a differentiable high-frequency cylindrical asymptotic kernel:

```math
G_{\mathrm{cyl}}(R) \propto \frac{\exp(i k R - i\pi/4)}{\sqrt{R}}.
```

Then the propagated field is approximated as:

```math
\psi_b(s_b) = \int G_{\mathrm{cyl}}(R(s_b,s_a))\psi_a(s_a)\,ds_a.
```

In discretized form:

```math
\psi_b[i] = \sum_j G_{\mathrm{cyl}}(R_{ij})\psi_a[j]\Delta s_a.
```

The first implementation can use trapezoidal weights. Simpson weights are a reasonable follow-up if the tilted-plane notebook shows a clear accuracy benefit for the same grid size.

### Projected-potential interaction

Use a projected phase grating:

```math
\psi_s(s) \leftarrow \psi_s(s)\exp\left[i\sigma V_{\mathrm{proj}}(s)\right].
```

Or, using refractive-index form:

```math
\psi_s(s) \leftarrow \psi_s(s)
\exp\left[
i \frac{2\pi}{\lambda}
(n(s)-1)\ell_{\mathrm{int}}(s)
\right].
```

### Core helpers

```python
cylindrical_green_asymptotic_1d
line_to_line_cylindrical_propagate_1d
phase_grating_1d_from_projected_potential
simulate_single_slice_cylindrical_1d
```

### Tests

```text
cylindrical propagation has expected 1/sqrt(R) amplitude trend
parallel-line case agrees with Fresnel propagation in a controlled regime
Gaussian beam lands near expected output coordinate
projected atom potential creates expected phase and intensity perturbation
finite gradients through atom strength, atom position, sample tilt, and beam tilt
```

We are going to have to be careful here too - because if there are a grid of atoms in depth, we will need to find a way 
to have the beam see some kind of summed depth potential, so it is still a single slice interaction. 

## Method 3: Bidirectional WPM

### Purpose

The bidirectional WPM model explicitly carries right-going and left-going fields:

```math
\psi^+(x,z), \qquad \psi^-(x,z).
```

It is used to model glancing reflection and two-way wave exchange in a slice-based way. The implementation should follow the bidirectional WPM references rather than treating the coupling rule as an ad hoc numerical boundary condition.

Relevant references:

```text
glancing_incidence_plan_papers_and_data/bidirectional_wpm_paper.pdf
glancing_incidence_plan_papers_and_data/bidirectional_wpm_thesis.pdf
```

### API

```python
simulate_bidirectional_wpm_1d(
    potential_slices,
    input_wave,
    dx,
    dz,
    energy,
    *,
    n_bins=128,
    n_sweeps=4,
    boundary="outgoing",
    return_diagnostics=True,
)
```

### State

```python
psi_plus   # right-going / transmitted component
psi_minus  # left-going / reflected component
```

### Algorithm

```text
1. Convert each potential slice to a refractive-index or wave-number profile.
2. Initialize psi_plus from the input beam.
3. Initialize psi_minus from the right boundary condition, usually zero.
4. Sweep psi_plus forward through the slices.
5. Sweep psi_minus backward through the slices.
6. Couple psi_plus and psi_minus at slice-to-slice changes.
7. Repeat for a fixed number of sweeps.
8. Return transmitted wave, reflected wave, and diagnostics.
```

### Diagnostics

```text
transmitted power
reflected power
forward/backward residual per sweep
norm drift
specular peak position
runtime
```

### Core helpers

```python
wpm_step_adaptive_1d
interface_coupling_wpm_1d
bidirectional_wpm_sweep_1d
simulate_bidirectional_wpm_1d
```

### Tests

```text
vacuum gives near-zero reflected power
uniform medium gives near-zero artificial reflection
single potential/interface gives nonzero reflection with correct sign
weak potential agrees with Fresnel and single-slice cylindrical model
strong glancing slab produces reflected component
fixed sweep count remains differentiable
```

## Method 4: Bidirectional Pade square-root BPM / multislice

### Purpose

This is the final advanced method.

It should solve the sideview propagation problem using a **bidirectional square-root propagation operator** approximated by fixed-order **Pade rational functions**. 
Use jax.grad where possible to enable automatic differentiation to calculate the Pade functions. 
It should explicitly maintain forward and backward components:

```math
\Psi_j =
\begin{bmatrix}
\psi_j^+ \\
\psi_j^-
\end{bmatrix}.
```

This method is intended to be:

```text
wide-angle
bidirectional
stable at glancing incidence
compatible with JAX autodiff
suitable for low-energy electron scattering examples
```

### API

```python
simulate_bidirectional_pade_bpm_1d(
    potential_slices,
    input_wave,
    dx,
    dz,
    energy,
    *,
    pade_order=(1, 1),
    n0_mode="slice_mean",
    evanescent="damp",
    boundary="pml",
    scattering_update="s_matrix",
    n_sweeps=4,
    return_diagnostics=True,
)
```

### Sideview operator

Use the refractive-index Helmholtz form used in `Paper/main.tex`. In the sideview 1D transverse case, the forward square-root branch is

```math
\partial_z\psi
=
2\pi i
\left[
\sqrt{
k_0^2 n_j^2(x)
+
\frac{1}{4\pi^2}D_x
}
-
k_0
\right]\psi,
```

where:

```text
D_x = 1D transverse Laplacian
k_0 = incident electron spatial frequency in cycles per unit length
n_j(x) = refractive index profile of slice j
```

For a fixed reference index \(n_{0,j}\), rewrite the square-root term as

```math
K_{z,j}
=
k_0 n_{0,j}
\sqrt{1 + X_j},
```

with

```math
X_j
=
\frac{n_j^2(x)-n_{0,j}^2}{n_{0,j}^2}
+
\frac{D_x}{(2\pi k_0 n_{0,j})^2}.
```

The simplest initial choice is \(n_{0,j}=1\), which gives

```math
X_j
=
\left[n_j^2(x)-1\right]
+
\frac{D_x}{(2\pi k_0)^2}.
```

This is equivalent to the previous potential-based expression when the corrected Schrodinger index is used, because `Paper/main.tex` defines

```math
n_\mathrm{S}^2(x)
=
1+\frac{\sigma}{\pi k_0}V(x).
```

The Pade approximation should therefore be applied to the dimensionless operator \(X_j\):

```math
\sqrt{1+X_j}
\approx
R_{m,n}(X_j)
=
\frac{P_m(X_j)}{Q_n(X_j)}.
```

The effective longitudinal propagation operator is then

```math
K_{z,j}
=
k_0 n_{0,j} R_{m,n}(X_j).
```

Forward and backward propagation through a slice use opposite signs relative to this longitudinal operator:

```math
P_j^+ \approx \exp(+i2\pi dz K_{z,j}),
\qquad
P_j^- \approx \exp(-i2\pi dz K_{z,j}).
```

### Fixed-order rational step

Start with low-order Pade steps and validate each order before adding the next:

```text
[1/1] first
[2/2] second
[4/4] after validation
```

Use fixed linear solves of the form:

```math
(I - bX)\psi_{\text{next}} = (I + aX)\psi_{\text{current}}.
```

This is good for JAX because the computational graph has fixed structure.

The first version should use dense linear solves so the mathematics is explicit. After the tests pass, replace the dense solve with a tridiagonal or banded solve if the operator structure supports it.

### Bidirectional state update

Use either explicit forward/backward sweeps:

```python
psi_plus = forward_pade_step(...)
psi_minus = backward_pade_step(...)
psi_plus, psi_minus = interface_scattering_update(...)
```

or a 2x2 block scattering update:

```math
\begin{bmatrix}
\psi_{j+1}^+ \\
\psi_j^-
\end{bmatrix}
=
S_j
\begin{bmatrix}
\psi_j^+ \\
\psi_{j+1}^-
\end{bmatrix}.
```

### Core helpers

```python
pade_sqrt_coefficients
build_sideview_operator_x_1d
apply_pade_rational_1d
pade_forward_step_1d
pade_backward_step_1d
interface_scattering_matrix_1d
bidirectional_pade_sweep_1d
simulate_bidirectional_pade_bpm_1d
```

### Tests

```text
vacuum gives no reflected wave
uniform medium gives no artificial reflection
single interface gives physically reasonable reflection
small-angle weak potential agrees with Fresnel
wide-angle vacuum agrees better with angular spectrum than Fresnel
[2/2] improves over [1/1]
evanescent modes are damped or filtered, not amplified
finite gradients through potential, beam tilt, and sample geometry
strong glancing slab gives better behavior than Fresnel
```

## Geometry module

Add a sideview geometry abstraction.

### `Line1D`

```python
@dataclass
class Line1D:
    r0: Array          # shape (2,), center in (x,z)
    tangent: Array     # shape (2,), unit direction along line
    normal: Array      # shape (2,), unit normal
    coords: Array      # shape (n,), coordinate along line

    def points(self):
        return self.r0[None, :] + self.coords[:, None] * self.tangent[None, :]
```

### Geometry helpers

```python
normalize
rotation_2d
line_from_angle
reflect_direction
phase_ramp_for_direction
make_tilted_gaussian_beam_1d
```

### Canonical layout

```text
input line:
  upright line on the left

sample line:
  tilted line near the center

output line:
  upright line on the right
```

### Beam direction

Use:

```math
\hat{k}_{\mathrm{in}} = (\sin\theta,\cos\theta),
```

where \(\theta\) controls beam tilt in the \(x\)-\(z\) plane.

### Specular direction

Use:

```math
\hat{k}_{\mathrm{out}}
=
\hat{k}_{\mathrm{in}}
-
2(\hat{k}_{\mathrm{in}}\cdot \hat{n})\hat{n}.
```

## Synthetic samples

### Sample A: smooth tilted slab

Use a smooth potential layer:

```math
U(x,z)
=
U_0
\exp\left[
-\frac{d_\perp(x,z)^2}{2\sigma_\perp^2}
\right],
```

where \(d_\perp\) is distance from the tilted sample line.

Purpose:

```text
clean glancing/specular reflection test
```

### Sample B: Gaussian atoms on a tilted line

Use:

```math
U(x,z)
=
\sum_i A_i
\exp\left[
-\frac{(x-x_i)^2+(z-z_i)^2}{2\sigma_i^2}
\right].
```

Suggested defaults:

```text
8-20 atoms
spacing: 1-3 Angstrom
sigma: 0.1-0.3 Angstrom
energies: 1, 5, 10, 20, 30 keV
glancing angles: 1 deg, 2 deg, 5 deg, 10 deg, 20 deg
grid sizes: 256, 512, 1024
```

Purpose:

```text
diffraction
roughness
atom-position gradients
potential reconstruction experiments later
```

## Repository layout

```text
wide_angle_propagation/
  propagation_methods.py
  sideview_geometry.py
  notebook_utils.py

notebooks/
  sideview_glancing_incidence_example.ipynb
  sideview_glancing_benchmark.ipynb

tests/
  test_sideview_geometry.py
  test_sideview_propagators_1d.py
  test_single_slice_cylindrical_1d.py
  test_bidirectional_wpm_1d.py
  test_bidirectional_pade_bpm_1d.py
  test_glancing_method_agreement.py
  test_glancing_differentiability.py

Paper/
  figures/
    sideview_geometry.*
    projected_atomic_sample.*
    method_comparison.*
    reflected_transmitted_power.*
    speed_accuracy_benchmark.*
```

## Public functions to add

```python
# Geometry
Line1D
normalize
rotation_2d
line_from_angle
reflect_direction
phase_ramp_for_direction
make_tilted_gaussian_beam_1d

# 1D propagation
get_frequencies_1d
fresnel_propagation_kernel_1d
angular_spectrum_propagation_kernel_1d
fourier_propagate_1d
diffraction_intensity_1d

# Potentials
schrodinger_refractive_index_1d
klein_gordon_refractive_index_1d
make_gaussian_atom_potential_sideview_1d
project_atoms_to_sample_line_1d
project_potential_to_sample_line_1d
phase_grating_1d_from_projected_potential

# Single-slice cylindrical model
cylindrical_green_asymptotic_1d
line_to_line_cylindrical_propagate_1d
simulate_single_slice_cylindrical_1d

# Fresnel baseline
simulate_glancing_fresnel_baseline_1d

# Bidirectional WPM
wpm_step_adaptive_1d
interface_coupling_wpm_1d
bidirectional_wpm_sweep_1d
simulate_bidirectional_wpm_1d

# Bidirectional Pade BPM
pade_sqrt_coefficients
build_sideview_operator_x_1d
apply_pade_rational_1d
pade_forward_step_1d
pade_backward_step_1d
interface_scattering_matrix_1d
bidirectional_pade_sweep_1d
simulate_bidirectional_pade_bpm_1d
```

Update `__all__` so these functions are exported by the package.

## Benchmark metrics

For each method, measure:

```text
JAX compile time
steady-state runtime
output intensity
phase-aligned complex field error
reflected power
transmitted power
specular peak coordinate
norm drift
memory use if easy
finite-gradient status
```

Use `block_until_ready()` for JAX timing.

Suggested benchmark sweep:

```text
energies:
  1, 5, 10, 20, 30 keV

optional high-energy controls:
  100, 300 keV

glancing angles:
  1 deg, 2 deg, 5 deg, 10 deg, 20 deg

grid sizes:
  256, 512, 1024
```

## Open implementation decisions

These decisions should be settled before coding the advanced methods:

```text
1. Units:
   Decide whether all internal lengths are Angstrom, nm, or meters, and keep the choice consistent with the existing electron-wavelength utilities.

2. Potential convention:
   Use refractive index as the internal convention for WPM and Pade BPM and Fresnel Methods. Add explicit conversion helpers from electrostatic potential to either the corrected Schrodinger index or the Klein-Gordon index, following `Paper/main.tex`. 

3. Boundary handling:
   Start with outgoing or simple absorbing boundaries for WPM and Pade BPM. Add PML only after vacuum and uniform-medium reflection tests pass.

4. Reference comparisons:
   Use angular spectrum propagation for small-angle free-space checks, the cylindrical direct integral for line-to-line checks, and WPM/Pade self-consistency for reflection benchmarks until a higher-fidelity reference is available.

5. Differentiability policy:
   Keep loop counts, Pade order, and boundary mode static. Gradients should be tested with respect to continuous parameters only: potential values, atom positions, atom strengths, beam tilt, and sample tilt.
```

## Implementation phases

### Phase 0: branch and scaffolding

```bash
git checkout -b glancing-bidirectional-pade-sideview
```

Deliverables:

```text
sideview geometry module
empty test files
sideview example notebook shell
benchmark notebook shell
```

Acceptance:

```text
package imports cleanly
existing tests pass
new tests are discoverable
```

### Phase 1: sideview geometry and 1D free-space propagation

Implement:

```python
Line1D
normalize
rotation_2d
line_from_angle
reflect_direction
phase_ramp_for_direction
get_frequencies_1d
fresnel_propagation_kernel_1d
angular_spectrum_propagation_kernel_1d
fourier_propagate_1d
diffraction_intensity_1d
```

Acceptance:

```text
line geometry correct
reflection law correct
vacuum angular-spectrum propagation correct
Fresnel agrees with angular spectrum in small-angle limit
```

### Phase 2: single-slice cylindrical model

Implement:

```python
cylindrical_green_asymptotic_1d
line_to_line_cylindrical_propagate_1d
phase_grating_1d_from_projected_potential
simulate_single_slice_cylindrical_1d
```

Acceptance:

```text
Gaussian beam propagates to expected position
projected atom creates stable phase perturbation
parallel-line case agrees with angular spectrum in controlled regime
JAX gradients are finite
```

### Phase 3: Fresnel baseline

Implement:

```python
simulate_glancing_fresnel_baseline_1d
```

Acceptance:

```text
fastest method at the same grid size and slice count
stable in weak scattering
agrees with small-angle cylindrical model
shows expected degradation in strong glancing benchmarks
```

### Phase 4: bidirectional WPM

Implement:

```python
wpm_step_adaptive_1d
interface_coupling_wpm_1d
bidirectional_wpm_sweep_1d
simulate_bidirectional_wpm_1d
```

Acceptance:

```text
vacuum reflection near zero
uniform-medium reflection near zero
single-interface reflection nonzero
fixed sweep count differentiable
residual decreases with sweep count
```

### Phase 5: bidirectional Pade square-root BPM

Implement:

```python
pade_sqrt_coefficients
build_sideview_operator_x_1d
apply_pade_rational_1d
pade_forward_step_1d
pade_backward_step_1d
interface_scattering_matrix_1d
bidirectional_pade_sweep_1d
simulate_bidirectional_pade_bpm_1d
```

Start with:

```text
[1/1] Pade
dense linear solves
fixed sweep count
simple absorbing boundary
```

Then add:

```text
[2/2] Pade
banded or tridiagonal solves
PML-style boundary
```

Acceptance:

```text
vacuum and uniform-medium tests pass
single-interface reflection is physically reasonable
[2/2] improves over [1/1]
evanescent components do not blow up
JAX gradients are finite
strong glancing slab improves over Fresnel
```

### Phase 6: benchmark notebook and paper figures

Deliverables:

```text
sideview geometry figure
projected atom sample figure
four-method output comparison
reflected/transmitted power plot
runtime table
speed-accuracy plot
energy sweep
glancing-angle sweep
```

Methods in plots:

```text
Fresnel baseline
Single-slice cylindrical projected-potential model
Bidirectional WPM
Bidirectional Pade BPM
```

## Paper narrative

The paper should present a controlled differentiable benchmark for sideview glancing-incidence electron scattering.

The main story:

```text
Fresnel propagation is fast and useful as a baseline, but it is paraxial.

The single-slice cylindrical projected-potential model gives a direct geometric
sideview calculation for a tilted atomic sample.

Bidirectional WPM introduces explicit forward and backward wave components.

Bidirectional Pade square-root BPM is the main advanced method: it combines
wide-angle propagation, explicit bidirectionality, fixed-order rational
approximations, and JAX-compatible differentiability.
```

Recommended paper sections:

```text
1. Introduction
2. Sideview glancing-incidence geometry
3. Electron wavelength, potential, and projected phase interaction
4. Four differentiable forward models
5. Numerical implementation
6. Synthetic tilted-sample benchmarks
7. Runtime and accuracy comparison
8. Gradient and inverse-modeling checks
9. Discussion
10. Conclusion
```

## Copy-paste brief for implementation

```text
Implement a differentiable sideview glancing-incidence electron scattering benchmark
in the WideAnglePropagation repository.

Branch:
  glancing-bidirectional-pade-sideview

Primary methods:
  1. Fresnel baseline
  2. Single-slice cylindrical projected-potential model
  3. Bidirectional WPM
  4. Bidirectional Pade square-root BPM / multislice

Main target file:
  wide_angle_propagation/propagation_methods.py

Optional geometry helper:
  wide_angle_propagation/sideview_geometry.py

Add geometry utilities:
  Line1D
  normalize
  rotation_2d
  line_from_angle
  reflect_direction
  phase_ramp_for_direction
  make_tilted_gaussian_beam_1d

Add 1D propagation helpers:
  get_frequencies_1d
  fresnel_propagation_kernel_1d
  angular_spectrum_propagation_kernel_1d
  fourier_propagate_1d
  diffraction_intensity_1d

Add potential helpers:
  make_gaussian_atom_potential_sideview_1d
  project_atoms_to_sample_line_1d
  project_potential_to_sample_line_1d
  phase_grating_1d_from_projected_potential

Add single-slice cylindrical model:
  cylindrical_green_asymptotic_1d
  line_to_line_cylindrical_propagate_1d
  simulate_single_slice_cylindrical_1d

Add Fresnel baseline:
  simulate_glancing_fresnel_baseline_1d

Add bidirectional WPM:
  wpm_step_adaptive_1d
  interface_coupling_wpm_1d
  bidirectional_wpm_sweep_1d
  simulate_bidirectional_wpm_1d

Add bidirectional Pade square-root BPM:
  pade_sqrt_coefficients
  build_sideview_operator_x_1d
  apply_pade_rational_1d
  pade_forward_step_1d
  pade_backward_step_1d
  interface_scattering_matrix_1d
  bidirectional_pade_sweep_1d
  simulate_bidirectional_pade_bpm_1d

The bidirectional methods must explicitly maintain:
  psi_plus
  psi_minus

The Pade method should use fixed order:
  start with [1/1]
  then [2/2]
  then [4/4] after validation

Tests:
  vacuum propagation
  uniform medium
  single-interface reflection
  small-angle Fresnel agreement
  cylindrical single-slice sanity check
  bidirectional WPM reflection check
  bidirectional Pade order convergence
  evanescent damping / no blow-up
  finite jax.grad checks
  glancing slab method comparison

Benchmarks:
  energies: 1, 5, 10, 20, 30 keV
  optional controls: 100, 300 keV
  glancing angles: 1 deg, 2 deg, 5 deg, 10 deg, 20 deg
  grid sizes: 256, 512, 1024
  report compile time separately from steady-state runtime
  use block_until_ready() for timings

Expected method behavior:
  Fresnel is fastest but paraxial.
  Single-slice cylindrical is geometry-faithful but thin/projected.
  Bidirectional WPM captures forward/backward wave exchange.
  Bidirectional Pade BPM is the main advanced differentiable wide-angle model.
```

## Bottom line

The implementation plan is:

```text
Fresnel baseline
Single-slice cylindrical projected-potential model
Bidirectional WPM
Bidirectional Pade square-root BPM / multislice
```

The main technical target is the fourth method: a **bidirectional Pade square-root BPM** that uses explicit forward and backward fields, fixed-order rational propagation steps, stable boundary handling, and JAX-compatible differentiable operations.
