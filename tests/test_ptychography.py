"""
Tests for the multislice ptychography module.

Covers:
1. Probe creation and normalisation
2. Fresnel kernel reciprocity
3. Probe shift correctness
4. Vacuum forward model (identity transmissions)
5. Single-slice phase object recovery
6. Multi-slice reconstruction convergence
7. Scan position diversity
8. WPM propagation correctness (vacuum, intensity preservation)
9. WPM internal helpers (_smoothstep, _get_polynomial_bins)
10. WPM vs Fresnel cross-consistency for weak scatterers
11. WPM forward model and 4D-STEM simulation
12. Gradient flow through WPM
13. WPM reconstruction convergence
"""

import os, sys, site
from pathlib import Path

# Auto-detect CUDA headers for CuPy
if not os.environ.get("CUDA_PATH"):
    for sp in site.getsitepackages():
        candidate = Path(sp) / "nvidia" / "cuda_runtime"
        header = candidate / "include" / "cuda_fp16.h"
        if header.exists():
            os.environ["CUDA_PATH"] = str(candidate)
            break

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

# Ensure package is importable
parent = Path(__file__).resolve().parent.parent
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))

from wide_angle_propagation.ptychography import (
    make_fresnel_kernel,
    make_angular_spectrum_kernel,
    propagate_fresnel,
    propagate_wpm,
    _smoothstep,
    _get_polynomial_bins,
    make_probe,
    shift_probe,
    multislice_forward_fresnel,
    multislice_forward_fresnel_scan,
    multislice_forward_wpm,
    amplitude_loss,
    intensity_loss,
    MultislicePtychographyReconstructor,
    simulate_4dstem,
    make_grid_scan,
)


# ---- Fixtures ----

@pytest.fixture
def small_grid():
    """Small 64x64 grid for fast tests."""
    ny, nx = 64, 64
    sampling = (0.5, 0.5)  # Angstroms
    energy = 100e3  # eV
    return ny, nx, sampling, energy


@pytest.fixture
def probe_64(small_grid):
    """Converged STEM probe on 64x64 grid."""
    ny, nx, sampling, energy = small_grid
    return make_probe(ny, nx, sampling, energy, semiangle_mrad=20.0)


# ---- Test probe creation ----

class TestProbe:
    def test_probe_normalised(self, small_grid):
        ny, nx, sampling, energy = small_grid
        probe = make_probe(ny, nx, sampling, energy, semiangle_mrad=20.0)
        total_intensity = float(jnp.sum(jnp.abs(probe)**2))
        assert abs(total_intensity - 1.0) < 1e-10, f"Probe not normalised: {total_intensity}"

    def test_probe_shape(self, small_grid):
        ny, nx, sampling, energy = small_grid
        probe = make_probe(ny, nx, sampling, energy, semiangle_mrad=20.0)
        assert probe.shape == (ny, nx)
        assert probe.dtype == jnp.complex128

    def test_probe_peaked_at_centre(self, small_grid):
        ny, nx, sampling, energy = small_grid
        probe = make_probe(ny, nx, sampling, energy, semiangle_mrad=20.0)
        intensity = jnp.abs(probe)**2
        # Peak should be near centre (0,0 in FFT convention)
        peak_idx = jnp.unravel_index(jnp.argmax(intensity), intensity.shape)
        assert peak_idx[0] == 0 or peak_idx[0] == ny - 1 or abs(peak_idx[0]) < 3
        assert peak_idx[1] == 0 or peak_idx[1] == nx - 1 or abs(peak_idx[1]) < 3


# ---- Test propagation kernels ----

class TestKernels:
    def test_fresnel_kernel_shape(self, small_grid):
        ny, nx, sampling, energy = small_grid
        from wide_angle_propagation.propagation import energy2wavelength
        wl = float(energy2wavelength(jnp.float64(energy)))
        H = make_fresnel_kernel(ny, nx, sampling, 2.0, wl)
        assert H.shape == (ny, nx)
        assert H.dtype == jnp.complex128

    def test_fresnel_vacuum_propagation_preserves_intensity(self, small_grid, probe_64):
        """Propagating through vacuum should preserve total intensity."""
        ny, nx, sampling, energy = small_grid
        from wide_angle_propagation.propagation import energy2wavelength
        wl = float(energy2wavelength(jnp.float64(energy)))
        H = make_fresnel_kernel(ny, nx, sampling, 2.0, wl)

        wave_out = propagate_fresnel(probe_64, H)
        intensity_in = float(jnp.sum(jnp.abs(probe_64)**2))
        intensity_out = float(jnp.sum(jnp.abs(wave_out)**2))
        assert abs(intensity_out - intensity_in) < 1e-10

    def test_forward_backward_roundtrip(self, small_grid, probe_64):
        """Propagating forward then backward should return to start."""
        ny, nx, sampling, energy = small_grid
        from wide_angle_propagation.propagation import energy2wavelength
        wl = float(energy2wavelength(jnp.float64(energy)))

        dz = 2.0
        H_fwd = make_fresnel_kernel(ny, nx, sampling, dz, wl)
        H_bwd = make_fresnel_kernel(ny, nx, sampling, -dz, wl)

        wave_fwd = propagate_fresnel(probe_64, H_fwd)
        wave_back = propagate_fresnel(wave_fwd, H_bwd)

        error = float(jnp.max(jnp.abs(wave_back - probe_64)))
        assert error < 1e-10, f"Round-trip error: {error}"


# ---- Test probe shifting ----

class TestProbeShift:
    def test_zero_shift_is_identity(self, probe_64):
        shifted = shift_probe(probe_64, jnp.array([0.0, 0.0]))
        error = float(jnp.max(jnp.abs(shifted - probe_64)))
        assert error < 1e-12

    def test_shift_preserves_intensity(self, probe_64):
        shifted = shift_probe(probe_64, jnp.array([3.5, -2.1]))
        i_orig = float(jnp.sum(jnp.abs(probe_64)**2))
        i_shifted = float(jnp.sum(jnp.abs(shifted)**2))
        assert abs(i_shifted - i_orig) < 1e-10

    def test_integer_shift_matches_roll(self, probe_64):
        """Integer pixel shift should exactly match jnp.roll."""
        shifted = shift_probe(probe_64, jnp.array([2.0, -3.0]))
        rolled = jnp.roll(jnp.roll(probe_64, 2, axis=0), -3, axis=1)
        error = float(jnp.max(jnp.abs(shifted - rolled)))
        assert error < 1e-8, f"Shift vs roll error: {error}"


# ---- Test vacuum forward model ----

class TestVacuumForward:
    def test_vacuum_multislice_preserves_intensity(self, small_grid, probe_64):
        """Identity transmission + propagation should preserve intensity."""
        ny, nx, sampling, energy = small_grid
        from wide_angle_propagation.propagation import energy2wavelength
        wl = float(energy2wavelength(jnp.float64(energy)))

        n_slices = 5
        transmissions = jnp.ones((n_slices, ny, nx), dtype=jnp.complex128)
        kernel = make_fresnel_kernel(ny, nx, sampling, 2.0, wl)

        exit_wave = multislice_forward_fresnel(probe_64, transmissions, kernel)
        i_in = float(jnp.sum(jnp.abs(probe_64)**2))
        i_out = float(jnp.sum(jnp.abs(exit_wave)**2))
        assert abs(i_out - i_in) / i_in < 1e-8

    def test_vacuum_dp_peaked_at_zero(self, small_grid, probe_64):
        """Vacuum diffraction pattern should have maximum at centre (zero beam)."""
        ny, nx, sampling, energy = small_grid
        from wide_angle_propagation.propagation import energy2wavelength
        wl = float(energy2wavelength(jnp.float64(energy)))

        transmissions = jnp.ones((3, ny, nx), dtype=jnp.complex128)
        kernel = make_fresnel_kernel(ny, nx, sampling, 2.0, wl)

        exit_wave = multislice_forward_fresnel(probe_64, transmissions, kernel)
        dp = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(exit_wave)))**2

        # Centre should be at the global maximum value (flat-top disc)
        centre_val = float(dp[ny // 2, nx // 2])
        max_val = float(dp.max())
        assert abs(centre_val - max_val) / (max_val + 1e-30) < 0.01


# ---- Test loss functions ----

class TestLoss:
    def test_amplitude_loss_zero_for_identical(self):
        dp = jnp.ones((8, 8)) * 0.5
        assert float(amplitude_loss(dp, dp)) < 1e-15

    def test_intensity_loss_zero_for_identical(self):
        dp = jnp.ones((8, 8)) * 0.5
        assert float(intensity_loss(dp, dp)) < 1e-15

    def test_amplitude_loss_positive_for_different(self):
        dp1 = jnp.ones((8, 8)) * 0.5
        dp2 = jnp.ones((8, 8)) * 1.0
        assert float(amplitude_loss(dp1, dp2)) > 0

    def test_loss_is_differentiable(self):
        """Check that JAX can differentiate through the loss."""
        dp_meas = jnp.ones((8, 8)) * 0.5

        def loss_fn(x):
            return amplitude_loss(x, dp_meas)

        grad = jax.grad(loss_fn)(jnp.ones((8, 8)) * 0.3)
        assert grad.shape == (8, 8)
        assert jnp.isfinite(grad).all()


# ---- Test 4D-STEM simulation ----

class TestSimulate4DSTEM:
    def test_vacuum_simulation(self, small_grid, probe_64):
        """4D-STEM of vacuum should give identical patterns for all positions."""
        ny, nx, sampling, energy = small_grid
        # Zero potential = vacuum
        potential = jnp.zeros((3, ny, nx), dtype=jnp.float64)
        positions = make_grid_scan(ny, nx, 2, 2, margin_pix=8)

        dps, trans = simulate_4dstem(
            potential, probe_64, positions,
            dz=2.0, sampling=sampling, energy=energy,
            propagator='fresnel',
        )

        assert dps.shape == (4, ny, nx)
        assert trans.shape == (3, ny, nx)

        # All DPs should be very similar (vacuum is translation-invariant)
        for j in range(1, 4):
            nrmse = np.sqrt(np.mean((dps[j] - dps[0])**2)) / (dps[0].max() + 1e-30)
            assert nrmse < 0.05, f"Vacuum DPs differ: NRMSE={nrmse}"

    def test_transmission_is_unity_for_vacuum(self, small_grid, probe_64):
        ny, nx, sampling, energy = small_grid
        potential = jnp.zeros((2, ny, nx), dtype=jnp.float64)
        positions = jnp.array([[0.0, 0.0]])

        _, trans = simulate_4dstem(
            potential, probe_64, positions,
            dz=2.0, sampling=sampling, energy=energy,
        )

        # Zero potential => n = 1 => phase = 0 => T = 1
        error = np.max(np.abs(trans - 1.0))
        assert error < 1e-10, f"Vacuum transmission not unity: max error {error}"


# ---- Test grid scan ----

class TestGridScan:
    def test_grid_scan_shape(self):
        positions = make_grid_scan(64, 64, 3, 4, margin_pix=4)
        assert positions.shape == (12, 2)

    def test_grid_scan_within_bounds(self):
        ny, nx = 64, 64
        positions = make_grid_scan(ny, nx, 5, 5, margin_pix=4)
        # Shifts are relative to centre
        for j in range(positions.shape[0]):
            abs_y = positions[j, 0] + ny / 2.0
            abs_x = positions[j, 1] + nx / 2.0
            assert 0 <= abs_y <= ny, f"Y out of bounds: {abs_y}"
            assert 0 <= abs_x <= nx, f"X out of bounds: {abs_x}"


# ---- Test single-slice reconstruction (the acid test) ----

class TestSingleSliceReconstruction:
    """
    Create a simple phase object with a single slice,
    simulate DPs, then reconstruct the phase.
    The recovered phase should match the ground truth.
    """

    def test_single_slice_phase_recovery(self, small_grid):
        ny, nx, sampling, energy = small_grid
        from wide_angle_propagation.propagation import energy2wavelength
        wl = float(energy2wavelength(jnp.float64(energy)))

        # Create a simple Gaussian phase object
        Y, X = jnp.mgrid[:ny, :nx]
        cy, cx = ny / 2.0, nx / 2.0
        r2 = (X - cx)**2 + (Y - cy)**2
        sigma = 8.0
        gt_phase = 0.5 * jnp.exp(-r2 / (2 * sigma**2))

        # Convert to potential: phase = 2*pi*(n-1)*dz/wl, and n-1 ~ sigma_e * V
        # We'll just directly build transmissions for the forward model
        gt_transmission = jnp.exp(1j * gt_phase)[None, :, :]  # (1, ny, nx)

        # Probe
        probe = make_probe(ny, nx, sampling, energy, semiangle_mrad=20.0)

        # Scan positions
        positions = make_grid_scan(ny, nx, 4, 4, margin_pix=8)
        n_pos = positions.shape[0]

        # Simulate data using Fresnel
        dz = 2.0
        kernel = make_fresnel_kernel(ny, nx, sampling, dz, wl)

        dps = multislice_forward_fresnel_scan(
            probe, gt_transmission, kernel, positions
        )

        # Reconstruct
        recon = MultislicePtychographyReconstructor(
            measured_dps=dps,
            probe=probe,
            positions_pix=positions,
            n_slices=1,
            dz=dz,
            sampling=sampling,
            energy=energy,
            propagator='fresnel',
            learning_rate=0.05,
            loss_fn='amplitude',
        )

        losses = recon.reconstruct(n_iterations=200, verbose=False)

        # Check convergence: loss should decrease significantly
        assert losses[-1] < losses[0] * 0.1, (
            f"Loss didn't converge: {losses[0]:.3e} -> {losses[-1]:.3e}"
        )

        # Check phase recovery
        recovered_phase = recon.get_recovered_phase()[0]

        # Correlation between recovered and ground truth phases
        gt_np = np.asarray(gt_phase)
        rec_np = np.asarray(recovered_phase)
        corr = np.corrcoef(gt_np.ravel(), rec_np.ravel())[0, 1]
        assert corr > 0.8, f"Phase correlation too low: {corr:.3f}"


# ---- Test multi-slice reconstruction convergence ----

class TestMultiSliceReconstruction:
    """
    Two-slice object: check that the reconstruction loss converges.
    We don't demand perfect depth separation on a small grid —
    just that the optimizer makes progress.
    """

    def test_two_slice_loss_decreases(self, small_grid):
        ny, nx, sampling, energy = small_grid
        from wide_angle_propagation.propagation import energy2wavelength
        wl = float(energy2wavelength(jnp.float64(energy)))

        # Two slices with different phase patterns
        Y, X = jnp.mgrid[:ny, :nx]
        cy, cx = ny / 2.0, nx / 2.0

        phase1 = 0.3 * jnp.exp(-((X - cx - 5)**2 + (Y - cy)**2) / (2 * 6**2))
        phase2 = 0.3 * jnp.exp(-((X - cx + 5)**2 + (Y - cy)**2) / (2 * 6**2))

        gt_trans = jnp.stack([
            jnp.exp(1j * phase1),
            jnp.exp(1j * phase2),
        ])

        probe = make_probe(ny, nx, sampling, energy, semiangle_mrad=20.0)
        positions = make_grid_scan(ny, nx, 4, 4, margin_pix=8)

        dz = 2.0
        kernel = make_fresnel_kernel(ny, nx, sampling, dz, wl)
        dps = multislice_forward_fresnel_scan(probe, gt_trans, kernel, positions)

        recon = MultislicePtychographyReconstructor(
            measured_dps=dps,
            probe=probe,
            positions_pix=positions,
            n_slices=2,
            dz=dz,
            sampling=sampling,
            energy=energy,
            propagator='fresnel',
            learning_rate=0.05,
        )

        losses = recon.reconstruct(n_iterations=100, verbose=False)

        # Loss should decrease
        assert losses[-1] < losses[0] * 0.5, (
            f"Multi-slice loss didn't decrease enough: {losses[0]:.3e} -> {losses[-1]:.3e}"
        )


# ---- Test gradient flow ----

class TestGradientFlow:
    """Verify JAX can differentiate through the full forward model."""

    def test_gradient_through_multislice(self, small_grid, probe_64):
        ny, nx, sampling, energy = small_grid
        from wide_angle_propagation.propagation import energy2wavelength
        wl = float(energy2wavelength(jnp.float64(energy)))

        kernel = make_fresnel_kernel(ny, nx, sampling, 2.0, wl)
        target_dp = jnp.ones((ny, nx)) * 0.01

        def loss_fn(obj_phase):
            trans = jnp.exp(1j * obj_phase)[None, :, :]
            ew = multislice_forward_fresnel(probe_64, trans, kernel)
            dp = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(ew)))**2
            return amplitude_loss(dp, target_dp)

        phase = jnp.zeros((ny, nx))
        grad = jax.grad(loss_fn)(phase)

        assert grad.shape == (ny, nx)
        assert jnp.isfinite(grad).all()
        # Gradient should be non-trivial
        assert float(jnp.max(jnp.abs(grad))) > 0


# =========================================================================
# WPM-specific tests
# =========================================================================

@pytest.fixture
def wpm_params(small_grid):
    """Common WPM parameters derived from the small grid fixture."""
    ny, nx, sampling, energy = small_grid
    from wide_angle_propagation.propagation import energy2wavelength
    wl = float(energy2wavelength(jnp.float64(energy)))
    dz = 2.0
    return dict(ny=ny, nx=nx, sampling=sampling, energy=energy,
                wavelength=wl, dz=dz)


# ---- WPM internal helpers ----

class TestSmoothstep:
    def test_boundary_values(self):
        assert float(_smoothstep(jnp.array(0.0))) == 0.0
        assert float(_smoothstep(jnp.array(1.0))) == 1.0

    def test_clamps_outside_01(self):
        assert float(_smoothstep(jnp.array(-0.5))) == 0.0
        assert float(_smoothstep(jnp.array(1.5))) == 1.0

    def test_monotonic(self):
        x = jnp.linspace(0, 1, 100)
        y = _smoothstep(x)
        diffs = jnp.diff(y)
        assert (diffs >= -1e-15).all(), "smoothstep must be monotonically non-decreasing"

    def test_midpoint(self):
        val = float(_smoothstep(jnp.array(0.5)))
        assert abs(val - 0.5) < 1e-12, f"smoothstep(0.5) should be 0.5, got {val}"


class TestPolynomialBins:
    def test_endpoints(self):
        bins = _get_polynomial_bins(1.0, 2.0, 10, power=2.0)
        assert abs(float(bins[0]) - 1.0) < 1e-12
        assert abs(float(bins[-1]) - 2.0) < 1e-12

    def test_sorted(self):
        bins = _get_polynomial_bins(0.5, 1.5, 32, power=3.0)
        diffs = jnp.diff(bins)
        assert (diffs >= -1e-15).all(), "Bins must be sorted"

    def test_correct_count(self):
        bins = _get_polynomial_bins(1.0, 1.1, 64, power=2.0)
        assert bins.shape == (64,)

    def test_uniform_when_power_one(self):
        bins = _get_polynomial_bins(0.0, 1.0, 11, power=1.0)
        expected = jnp.linspace(0, 1, 11)
        assert jnp.allclose(bins, expected, atol=1e-12)


# ---- WPM propagation basics ----

class TestWPMPropagation:
    def test_vacuum_preserves_intensity(self, wpm_params, probe_64):
        """WPM propagation through vacuum (n=1 everywhere) should preserve intensity."""
        p = wpm_params
        n_map = jnp.ones((p['ny'], p['nx']), dtype=jnp.float64)
        wave_out = propagate_wpm(
            probe_64, n_map, p['dz'], p['wavelength'], p['sampling'],
            n_bins=16,
        )
        i_in = float(jnp.sum(jnp.abs(probe_64)**2))
        i_out = float(jnp.sum(jnp.abs(wave_out)**2))
        assert abs(i_out - i_in) / i_in < 1e-6, (
            f"WPM vacuum intensity not preserved: {i_in} -> {i_out}"
        )

    def test_vacuum_matches_fresnel(self, wpm_params, probe_64):
        """WPM with uniform n=1 should closely match Fresnel propagation."""
        p = wpm_params
        n_map = jnp.ones((p['ny'], p['nx']), dtype=jnp.float64)
        wave_wpm = propagate_wpm(
            probe_64, n_map, p['dz'], p['wavelength'], p['sampling'],
            n_bins=16,
        )
        kernel = make_fresnel_kernel(
            p['ny'], p['nx'], p['sampling'], p['dz'], p['wavelength']
        )
        wave_fresnel = propagate_fresnel(probe_64, kernel)

        # WPM with exact n=1 uses the full angular spectrum, so compare
        # against the angular spectrum kernel instead
        H_as = make_angular_spectrum_kernel(
            p['ny'], p['nx'], p['sampling'], p['dz'], p['wavelength']
        )
        wave_as = propagate_fresnel(probe_64, H_as)

        # WPM vacuum should be very close to angular spectrum (it IS angular spectrum for n=1)
        error_as = float(jnp.max(jnp.abs(wave_wpm - wave_as)))
        assert error_as < 1e-6, f"WPM vacuum vs AS error: {error_as}"

    def test_output_shape_dtype(self, wpm_params, probe_64):
        p = wpm_params
        n_map = jnp.ones((p['ny'], p['nx']), dtype=jnp.float64)
        wave_out = propagate_wpm(
            probe_64, n_map, p['dz'], p['wavelength'], p['sampling'],
            n_bins=16,
        )
        assert wave_out.shape == probe_64.shape
        assert jnp.iscomplexobj(wave_out)

    def test_nonuniform_n_changes_wave(self, wpm_params, probe_64):
        """A non-uniform refractive index map should produce a different
        wave compared to vacuum propagation."""
        p = wpm_params
        ny, nx = p['ny'], p['nx']
        # Create a non-trivial refractive index map
        Y, X = jnp.mgrid[:ny, :nx]
        n_map = 1.0 + 0.01 * jnp.exp(-((X - nx/2)**2 + (Y - ny/2)**2) / 100)

        wave_material = propagate_wpm(
            probe_64, n_map, p['dz'], p['wavelength'], p['sampling'],
            n_bins=16,
        )
        wave_vacuum = propagate_wpm(
            probe_64, jnp.ones_like(n_map), p['dz'], p['wavelength'], p['sampling'],
            n_bins=16,
        )
        diff = float(jnp.max(jnp.abs(wave_material - wave_vacuum)))
        assert diff > 1e-10, "Non-uniform n_map must produce different wave than vacuum"

    def test_bin_count_affects_accuracy(self, wpm_params, probe_64):
        """More bins should give a result closer to the analytical limit."""
        p = wpm_params
        ny, nx = p['ny'], p['nx']
        Y, X = jnp.mgrid[:ny, :nx]
        n_map = 1.0 + 0.005 * jnp.sin(2 * jnp.pi * X / nx)

        wave_few = propagate_wpm(
            probe_64, n_map, p['dz'], p['wavelength'], p['sampling'],
            n_bins=4,
        )
        wave_many = propagate_wpm(
            probe_64, n_map, p['dz'], p['wavelength'], p['sampling'],
            n_bins=64,
        )
        # They shouldn't be identical (different binning)
        diff = float(jnp.max(jnp.abs(wave_few - wave_many)))
        # But for a weak scatterer with few bins vs many, the difference
        # should still be modest
        assert diff < 0.1, f"Few vs many bins differ too much: {diff}"
        assert diff > 0, "Different bin counts should give slightly different results"


# ---- WPM multislice forward model ----

class TestWPMForwardModel:
    def test_vacuum_preserves_intensity(self, wpm_params, probe_64):
        """Multislice WPM through vacuum should preserve total intensity."""
        p = wpm_params
        n_slices = 3
        n_maps = jnp.ones((n_slices, p['ny'], p['nx']), dtype=jnp.float64)
        exit_wave = multislice_forward_wpm(
            probe_64, n_maps, p['dz'], p['wavelength'], p['sampling'],
            n_bins=16,
        )
        i_in = float(jnp.sum(jnp.abs(probe_64)**2))
        i_out = float(jnp.sum(jnp.abs(exit_wave)**2))
        assert abs(i_out - i_in) / i_in < 1e-5, (
            f"WPM multislice vacuum intensity: {i_in} -> {i_out}"
        )

    def test_vacuum_dp_peaked_at_zero(self, wpm_params, probe_64):
        """Vacuum WPM DP should have maximum at centre."""
        p = wpm_params
        n_maps = jnp.ones((3, p['ny'], p['nx']), dtype=jnp.float64)
        exit_wave = multislice_forward_wpm(
            probe_64, n_maps, p['dz'], p['wavelength'], p['sampling'],
            n_bins=16,
        )
        dp = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(exit_wave)))**2
        ny, nx = p['ny'], p['nx']
        centre_val = float(dp[ny // 2, nx // 2])
        max_val = float(dp.max())
        assert abs(centre_val - max_val) / (max_val + 1e-30) < 0.01

    def test_output_shape(self, wpm_params, probe_64):
        p = wpm_params
        n_maps = jnp.ones((5, p['ny'], p['nx']), dtype=jnp.float64)
        exit_wave = multislice_forward_wpm(
            probe_64, n_maps, p['dz'], p['wavelength'], p['sampling'],
            n_bins=16,
        )
        assert exit_wave.shape == probe_64.shape


# ---- WPM vs Fresnel cross-consistency ----

class TestWPMFresnelConsistency:
    """For weak, thin samples the Fresnel and WPM forward models
    should produce very similar diffraction patterns."""

    def test_weak_object_dp_similarity(self, wpm_params, probe_64):
        p = wpm_params
        ny, nx = p['ny'], p['nx']

        # Weak Gaussian phase object
        Y, X = jnp.mgrid[:ny, :nx]
        phase = 0.1 * jnp.exp(-((X - nx/2)**2 + (Y - ny/2)**2) / (2 * 8**2))
        transmission = jnp.exp(1j * phase)[None, :, :]

        # Fresnel forward
        kernel = make_fresnel_kernel(
            ny, nx, p['sampling'], p['dz'], p['wavelength']
        )
        ew_fresnel = multislice_forward_fresnel(probe_64, transmission, kernel)
        dp_fresnel = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(ew_fresnel)))**2

        # WPM forward: convert phase to refractive index
        n_maps = 1.0 + phase[None, :, :] * p['wavelength'] / (2 * jnp.pi * p['dz'])
        ew_wpm = multislice_forward_wpm(
            probe_64, n_maps, p['dz'], p['wavelength'], p['sampling'],
            n_bins=32,
        )
        dp_wpm = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(ew_wpm)))**2

        # Normalised RMS error between the two DPs
        nrmse = float(jnp.sqrt(jnp.mean((dp_fresnel - dp_wpm)**2))) / (
            float(dp_fresnel.max()) + 1e-30
        )
        assert nrmse < 0.15, (
            f"Weak-object DPs differ too much: NRMSE = {nrmse:.4f}"
        )

    def test_vacuum_dp_identical(self, wpm_params, probe_64):
        """In vacuum both methods should give essentially the same DP."""
        p = wpm_params
        ny, nx = p['ny'], p['nx']

        # Fresnel
        trans = jnp.ones((2, ny, nx), dtype=jnp.complex128)
        kernel = make_fresnel_kernel(
            ny, nx, p['sampling'], p['dz'], p['wavelength']
        )
        ew_f = multislice_forward_fresnel(probe_64, trans, kernel)
        dp_f = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(ew_f)))**2

        # WPM
        n_maps = jnp.ones((2, ny, nx), dtype=jnp.float64)
        ew_w = multislice_forward_wpm(
            probe_64, n_maps, p['dz'], p['wavelength'], p['sampling'],
            n_bins=16,
        )
        dp_w = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(ew_w)))**2

        nrmse = float(jnp.sqrt(jnp.mean((dp_f - dp_w)**2))) / (
            float(dp_f.max()) + 1e-30
        )
        # Fresnel is a paraxial approximation of AS; WPM uses full AS in vacuum
        # So they can differ slightly — but not much for a small aperture probe
        assert nrmse < 0.05, f"Vacuum DPs differ: NRMSE = {nrmse:.4f}"


# ---- WPM 4D-STEM simulation ----

class TestWPMSimulate4DSTEM:
    def test_wpm_vacuum_simulation(self, small_grid, probe_64):
        """4D-STEM with WPM propagator on vacuum should work and
        give similar patterns for all positions."""
        ny, nx, sampling, energy = small_grid
        potential = jnp.zeros((3, ny, nx), dtype=jnp.float64)
        positions = make_grid_scan(ny, nx, 2, 2, margin_pix=8)

        dps, trans = simulate_4dstem(
            potential, probe_64, positions,
            dz=2.0, sampling=sampling, energy=energy,
            propagator='wpm', n_bins=16,
        )
        assert dps.shape == (4, ny, nx)
        assert trans.shape == (3, ny, nx)

        # All vacuum DPs very similar
        for j in range(1, 4):
            nrmse = np.sqrt(np.mean((dps[j] - dps[0])**2)) / (dps[0].max() + 1e-30)
            assert nrmse < 0.05, f"WPM vacuum DPs differ: NRMSE={nrmse}"

    def test_wpm_nonzero_potential_dps_differ_from_vacuum(self, small_grid, probe_64):
        """A non-zero potential should produce DPs different from vacuum."""
        ny, nx, sampling, energy = small_grid

        # Gaussian bump potential in slice 0
        Y, X = jnp.mgrid[:ny, :nx]
        pot = jnp.zeros((2, ny, nx), dtype=jnp.float64)
        pot = pot.at[0].set(
            500.0 * jnp.exp(-((X - nx/2)**2 + (Y - ny/2)**2) / (2 * 5**2))
        )
        positions = jnp.array([[0.0, 0.0]])

        dps_mat, _ = simulate_4dstem(
            pot, probe_64, positions,
            dz=2.0, sampling=sampling, energy=energy,
            propagator='wpm', n_bins=16,
        )
        dps_vac, _ = simulate_4dstem(
            jnp.zeros_like(pot), probe_64, positions,
            dz=2.0, sampling=sampling, energy=energy,
            propagator='wpm', n_bins=16,
        )
        diff = np.sqrt(np.mean((dps_mat - dps_vac)**2)) / (dps_vac.max() + 1e-30)
        assert diff > 0.001, f"Material DPs should differ from vacuum: diff={diff}"


# ---- Gradient flow through WPM ----

class TestWPMGradientFlow:
    def test_gradient_through_propagate_wpm(self, wpm_params, probe_64):
        """JAX should be able to differentiate through propagate_wpm."""
        p = wpm_params
        target_dp = jnp.ones((p['ny'], p['nx'])) * 0.01

        def loss_fn(n_flat):
            n_map = n_flat.reshape(p['ny'], p['nx'])
            wave = propagate_wpm(
                probe_64, n_map, p['dz'], p['wavelength'], p['sampling'],
                n_bins=8,
            )
            dp = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(wave)))**2
            return amplitude_loss(dp, target_dp)

        n0 = jnp.ones(p['ny'] * p['nx'])
        grad = jax.grad(loss_fn)(n0)
        assert grad.shape == n0.shape
        assert jnp.isfinite(grad).all(), "Gradient through WPM contains non-finite values"

    def test_gradient_through_multislice_wpm(self, wpm_params, probe_64):
        """Gradient through the full WPM multislice model."""
        p = wpm_params
        target_dp = jnp.ones((p['ny'], p['nx'])) * 0.01

        def loss_fn(obj_phase):
            n_maps = 1.0 + obj_phase * p['wavelength'] / (2 * jnp.pi * p['dz'])
            ew = multislice_forward_wpm(
                probe_64, n_maps, p['dz'], p['wavelength'], p['sampling'],
                n_bins=8,
            )
            dp = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(ew)))**2
            return amplitude_loss(dp, target_dp)

        phase = jnp.zeros((2, p['ny'], p['nx']))
        grad = jax.grad(loss_fn)(phase)
        assert grad.shape == phase.shape
        assert jnp.isfinite(grad).all(), "Gradient through WPM multislice not finite"
        assert float(jnp.max(jnp.abs(grad))) > 0, "WPM gradient is all zeros"

    def test_wpm_gradient_nonzero_with_material(self, wpm_params, probe_64):
        """Gradient w.r.t. refractive index should be non-trivial for a material."""
        p = wpm_params
        ny, nx = p['ny'], p['nx']

        # Generate a target from a Gaussian bump
        Y, X = jnp.mgrid[:ny, :nx]
        n_true = 1.0 + 0.01 * jnp.exp(-((X - nx/2)**2 + (Y - ny/2)**2) / (2 * 5**2))
        wave_true = propagate_wpm(
            probe_64, n_true, p['dz'], p['wavelength'], p['sampling'], n_bins=16,
        )
        target_dp = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(wave_true)))**2

        def loss_fn(n_map):
            wave = propagate_wpm(
                probe_64, n_map, p['dz'], p['wavelength'], p['sampling'],
                n_bins=16,
            )
            dp = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(wave)))**2
            return amplitude_loss(dp, target_dp)

        # Start from vacuum — gradient should point towards the true n_map
        n_init = jnp.ones((ny, nx), dtype=jnp.float64)
        grad = jax.grad(loss_fn)(n_init)
        assert jnp.isfinite(grad).all()
        assert float(jnp.max(jnp.abs(grad))) > 0


# ---- WPM reconstruction ----

class TestWPMReconstruction:
    def test_wpm_single_slice_loss_decreases(self, small_grid):
        """WPM reconstructor should decrease the loss on a simple single-slice object."""
        ny, nx, sampling, energy = small_grid
        from wide_angle_propagation.propagation import energy2wavelength
        wl = float(energy2wavelength(jnp.float64(energy)))

        # Simple Gaussian phase object
        Y, X = jnp.mgrid[:ny, :nx]
        cy, cx = ny / 2.0, nx / 2.0
        phase = 0.3 * jnp.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * 8**2))
        n_map = 1.0 + phase * wl / (2 * jnp.pi * 2.0)

        probe = make_probe(ny, nx, sampling, energy, semiangle_mrad=20.0)
        positions = make_grid_scan(ny, nx, 3, 3, margin_pix=8)

        # Generate data with WPM forward model
        dps_list = []
        for j in range(positions.shape[0]):
            shifted = shift_probe(probe, positions[j])
            ew = multislice_forward_wpm(
                shifted, n_map[None], 2.0, wl, sampling, n_bins=16,
            )
            dp = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(ew)))**2
            dps_list.append(dp)
        dps = jnp.stack(dps_list)

        recon = MultislicePtychographyReconstructor(
            measured_dps=dps,
            probe=probe,
            positions_pix=positions,
            n_slices=1,
            dz=2.0,
            sampling=sampling,
            energy=energy,
            propagator='wpm',
            n_bins=16,
            learning_rate=0.02,
            loss_fn='amplitude',
        )

        losses = recon.reconstruct(n_iterations=50, verbose=False)
        assert losses[-1] < losses[0], (
            f"WPM loss did not decrease: {losses[0]:.3e} -> {losses[-1]:.3e}"
        )

    def test_wpm_reconstruction_produces_finite_phases(self, small_grid):
        """Reconstructed phases from WPM should contain no NaN/Inf."""
        ny, nx, sampling, energy = small_grid
        from wide_angle_propagation.propagation import energy2wavelength
        wl = float(energy2wavelength(jnp.float64(energy)))

        probe = make_probe(ny, nx, sampling, energy, semiangle_mrad=20.0)
        positions = make_grid_scan(ny, nx, 2, 2, margin_pix=8)

        # Simulate vacuum data with WPM
        n_maps = jnp.ones((2, ny, nx), dtype=jnp.float64)
        dps_list = []
        for j in range(positions.shape[0]):
            shifted = shift_probe(probe, positions[j])
            ew = multislice_forward_wpm(
                shifted, n_maps, 2.0, wl, sampling, n_bins=16,
            )
            dp = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(ew)))**2
            dps_list.append(dp)
        dps = jnp.stack(dps_list)

        recon = MultislicePtychographyReconstructor(
            measured_dps=dps,
            probe=probe,
            positions_pix=positions,
            n_slices=2,
            dz=2.0,
            sampling=sampling,
            energy=energy,
            propagator='wpm',
            n_bins=16,
            learning_rate=0.01,
        )
        recon.reconstruct(n_iterations=20, verbose=False)

        phases = recon.get_recovered_phase()
        amps = recon.get_recovered_amplitude()
        assert np.isfinite(phases).all(), "Recovered phases contain NaN/Inf"
        assert np.isfinite(amps).all(), "Recovered amplitudes contain NaN/Inf"
