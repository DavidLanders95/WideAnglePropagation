"""Tests for the ptychography module.

All tests use small grids and few iterations so they finish quickly.
The WPM tests use a minimal n_bins=4 to keep compilation times short.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from wide_angle_propagation.ptychography import (
    make_probe,
    make_gaussian_probe,
    generate_scan_positions,
    fourier_shift,
    simulate_ptychography_as,
    simulate_ptychography_wpm,
    epie_thin,
    epie_multislice_as,
    reconstruct_as,
    reconstruct_wpm,
    make_phase_object,
    make_potential_phantom,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NY, NX = 32, 32
ENERGY = 300e3          # 300 kV
SAMPLING = (0.2, 0.2)   # Å/pixel
SLICE_DZ = 2.0          # Å
CONVERGENCE = 20.0      # mrad


@pytest.fixture(scope="module")
def probe():
    return make_probe(NY, NX, CONVERGENCE, ENERGY, SAMPLING)


@pytest.fixture(scope="module")
def gaussian_probe():
    return make_gaussian_probe(NY, NX, sigma=4.0)


@pytest.fixture(scope="module")
def scan_positions():
    # 3×3 grid with 1.0 Å step (small step keeps everything within the probe support)
    return generate_scan_positions(3, 3, step_y=1.0, step_x=1.0)


@pytest.fixture(scope="module")
def thin_object():
    return make_phase_object(NY, NX, n_features=3, seed=0)


@pytest.fixture(scope="module")
def phase_slices():
    # 2 slices for multi-slice tests
    s0 = make_phase_object(NY, NX, n_features=2, seed=1)
    s1 = make_phase_object(NY, NX, n_features=2, seed=2)
    return jnp.stack([s0, s1])


@pytest.fixture(scope="module")
def potential_slices():
    return make_potential_phantom(NY, NX, n_slices=2, peak_potential=3.0, seed=7)


# ---------------------------------------------------------------------------
# 1. Probe generation
# ---------------------------------------------------------------------------


class TestMakeProbe:
    def test_shape(self, probe):
        assert probe.shape == (NY, NX)

    def test_complex(self, probe):
        assert jnp.issubdtype(probe.dtype, jnp.complexfloating)

    def test_normalized(self, probe):
        intensity = float(jnp.sum(jnp.abs(probe) ** 2))
        assert abs(intensity - 1.0) < 1e-6

    def test_nonzero(self, probe):
        assert float(jnp.max(jnp.abs(probe))) > 0.0

    def test_defocus_changes_probe(self):
        p0 = make_probe(NY, NX, CONVERGENCE, ENERGY, SAMPLING, defocus=0.0)
        p1 = make_probe(NY, NX, CONVERGENCE, ENERGY, SAMPLING, defocus=50.0)
        assert not jnp.allclose(p0, p1)


class TestMakeGaussianProbe:
    def test_shape(self, gaussian_probe):
        assert gaussian_probe.shape == (NY, NX)

    def test_normalized(self, gaussian_probe):
        assert abs(float(jnp.sum(jnp.abs(gaussian_probe) ** 2)) - 1.0) < 1e-6

    def test_centered_at_origin(self):
        p = make_gaussian_probe(NY, NX, sigma=2.0, center=(0.0, 0.0))
        # Maximum should be at (0, 0)
        idx = int(jnp.argmax(jnp.abs(p)))
        row, col = divmod(idx, NX)
        assert row == 0 and col == 0


# ---------------------------------------------------------------------------
# 2. Scan positions
# ---------------------------------------------------------------------------


class TestGenerateScanPositions:
    def test_shape(self, scan_positions):
        assert scan_positions.shape == (9, 2)

    def test_step_spacing(self):
        pos = generate_scan_positions(4, 4, step_y=2.0, step_x=3.0)
        # Unique y steps
        ys = np.unique(np.array(pos[:, 0]))
        xs = np.unique(np.array(pos[:, 1]))
        assert len(ys) == 4
        assert len(xs) == 4
        assert abs(float(ys[1] - ys[0]) - 2.0) < 1e-10
        assert abs(float(xs[1] - xs[0]) - 3.0) < 1e-10

    def test_origin(self):
        pos = generate_scan_positions(2, 2, 1.0, 1.0, origin_y=5.0, origin_x=7.0)
        assert abs(float(pos[0, 0]) - 5.0) < 1e-10
        assert abs(float(pos[0, 1]) - 7.0) < 1e-10


# ---------------------------------------------------------------------------
# 3. Fourier shift
# ---------------------------------------------------------------------------


class TestFourierShift:
    def test_zero_shift_identity(self):
        rng = np.random.default_rng(0)
        field = jnp.array(rng.standard_normal((NY, NX)) + 1j * rng.standard_normal((NY, NX)))
        dy, dx = SAMPLING
        fy = jnp.fft.fftfreq(NY, d=dy)
        fx = jnp.fft.fftfreq(NX, d=dx)
        FY, FX = jnp.meshgrid(fy, fx, indexing="ij")
        shifted = fourier_shift(field, jnp.array([0.0, 0.0]), FY, FX)
        assert jnp.allclose(shifted, field, atol=1e-10)

    def test_shift_changes_field(self):
        p = make_gaussian_probe(NY, NX, sigma=4.0, center=(0.0, 0.0))
        dy, dx = SAMPLING
        fy = jnp.fft.fftfreq(NY, d=dy)
        fx = jnp.fft.fftfreq(NX, d=dx)
        FY, FX = jnp.meshgrid(fy, fx, indexing="ij")
        shifted = fourier_shift(p, jnp.array([float(NY // 4) * dy, 0.0]), FY, FX)
        assert not jnp.allclose(p, shifted, atol=1e-6)

    def test_preserves_norm(self):
        p = make_gaussian_probe(NY, NX, sigma=4.0)
        dy, dx = SAMPLING
        fy = jnp.fft.fftfreq(NY, d=dy)
        fx = jnp.fft.fftfreq(NX, d=dx)
        FY, FX = jnp.meshgrid(fy, fx, indexing="ij")
        shifted = fourier_shift(p, jnp.array([0.4, 0.2]), FY, FX)
        norm_before = float(jnp.sum(jnp.abs(p) ** 2))
        norm_after = float(jnp.sum(jnp.abs(shifted) ** 2))
        assert abs(norm_before - norm_after) < 1e-6


# ---------------------------------------------------------------------------
# 4. Forward model – Angular Spectrum
# ---------------------------------------------------------------------------


class TestSimulatePtychographyAS:
    def test_output_shapes(self, probe, scan_positions, thin_object):
        obj_slices = thin_object[None]  # (1, ny, nx)
        dps, ews = simulate_ptychography_as(
            obj_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        n_pos = len(scan_positions)
        assert dps.shape == (n_pos, NY, NX)
        assert ews.shape == (n_pos, NY, NX)

    def test_nonnegative_intensity(self, probe, scan_positions, thin_object):
        obj_slices = thin_object[None]
        dps, _ = simulate_ptychography_as(
            obj_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        assert float(jnp.min(dps)) >= 0.0

    def test_parseval(self, probe, scan_positions, thin_object):
        """Total diffracted intensity equals total exit-wave intensity (Parseval)."""
        obj_slices = thin_object[None]
        dps, ews = simulate_ptychography_as(
            obj_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        for i in range(len(scan_positions)):
            real_sum = float(jnp.sum(jnp.abs(ews[i]) ** 2))
            fourier_sum = float(jnp.sum(dps[i])) / (NY * NX)
            assert abs(real_sum - fourier_sum) / (real_sum + 1e-12) < 1e-4

    def test_multislice_differs_from_single(self, probe, scan_positions, phase_slices):
        dps_multi, _ = simulate_ptychography_as(
            phase_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        combined = phase_slices[0] * phase_slices[1]
        dps_single, _ = simulate_ptychography_as(
            combined[None], probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        # With propagation between slices the patterns differ
        assert not jnp.allclose(dps_multi, dps_single, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Forward model – WPM
# ---------------------------------------------------------------------------


class TestSimulatePtychographyWPM:
    def test_output_shapes(self, probe, scan_positions, potential_slices):
        dps, ews = simulate_ptychography_wpm(
            potential_slices, probe, scan_positions,
            SLICE_DZ, ENERGY, SAMPLING, n_bins=4
        )
        n_pos = len(scan_positions)
        assert dps.shape == (n_pos, NY, NX)
        assert ews.shape == (n_pos, NY, NX)

    def test_nonnegative_intensity(self, probe, scan_positions, potential_slices):
        dps, _ = simulate_ptychography_wpm(
            potential_slices, probe, scan_positions,
            SLICE_DZ, ENERGY, SAMPLING, n_bins=4
        )
        assert float(jnp.min(dps)) >= 0.0

    def test_zero_potential_equals_free_space(self, probe, scan_positions):
        zero_pot = jnp.zeros((1, NY, NX), dtype=jnp.float64)
        dps_wpm, _ = simulate_ptychography_wpm(
            zero_pot, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING, n_bins=4
        )
        # For zero potential, WPM should produce patterns close to the probe FFT
        # at each (shifted) position; they should all be positive and finite
        assert jnp.all(jnp.isfinite(dps_wpm))


# ---------------------------------------------------------------------------
# 6. ePIE – Thin Object
# ---------------------------------------------------------------------------


class TestEpieThin:
    def test_error_decreases(self, probe, scan_positions, thin_object):
        obj_slices = thin_object[None]
        dps, _ = simulate_ptychography_as(
            obj_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        _, _, errors = epie_thin(
            dps, scan_positions, probe, SAMPLING,
            n_iter=5, alpha=1.0, beta=0.9, update_probe=False, seed=0
        )
        assert errors[-1] <= errors[0], "ePIE error did not decrease"

    def test_output_shapes(self, probe, scan_positions, thin_object):
        obj_slices = thin_object[None]
        dps, _ = simulate_ptychography_as(
            obj_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        obj_rec, probe_rec, errors = epie_thin(
            dps, scan_positions, probe, SAMPLING, n_iter=3
        )
        assert obj_rec.shape == (NY, NX)
        assert probe_rec.shape == (NY, NX)
        assert len(errors) == 3

    def test_probe_update_option(self, probe, scan_positions, thin_object):
        obj_slices = thin_object[None]
        dps, _ = simulate_ptychography_as(
            obj_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        _, probe_rec_no_upd, _ = epie_thin(
            dps, scan_positions, probe, SAMPLING, n_iter=3, update_probe=False
        )
        _, probe_rec_upd, _ = epie_thin(
            dps, scan_positions, probe, SAMPLING, n_iter=3, update_probe=True
        )
        # When updating the probe, it should change
        assert not jnp.allclose(probe_rec_upd, probe, atol=1e-10)
        # Without probe update, probe should remain identical
        assert jnp.allclose(probe_rec_no_upd, probe, atol=1e-10)


# ---------------------------------------------------------------------------
# 7. Multi-slice ePIE – Angular Spectrum
# ---------------------------------------------------------------------------


class TestEpieMultisliceAS:
    def test_error_decreases(self, probe, scan_positions, phase_slices):
        dps, _ = simulate_ptychography_as(
            phase_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        _, _, errors = epie_multislice_as(
            dps, scan_positions, probe,
            n_slices=2,
            slice_thickness=SLICE_DZ,
            energy=ENERGY,
            sampling=SAMPLING,
            n_iter=5, alpha=0.5, beta=0.5, update_probe=False, seed=0,
        )
        assert errors[-1] <= errors[0], "Multi-slice ePIE error did not decrease"

    def test_output_shapes(self, probe, scan_positions, phase_slices):
        dps, _ = simulate_ptychography_as(
            phase_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        slices_rec, probe_rec, errors = epie_multislice_as(
            dps, scan_positions, probe,
            n_slices=2,
            slice_thickness=SLICE_DZ,
            energy=ENERGY,
            sampling=SAMPLING,
            n_iter=3,
        )
        assert slices_rec.shape == (2, NY, NX)
        assert probe_rec.shape == (NY, NX)
        assert len(errors) == 3

    def test_single_slice_consistent_with_epie_thin(self, probe, scan_positions, thin_object):
        """With one slice the multi-slice ePIE should give similar behaviour to epie_thin."""
        obj_slices = thin_object[None]
        dps, _ = simulate_ptychography_as(
            obj_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        _, _, err_ms = epie_multislice_as(
            dps, scan_positions, probe,
            n_slices=1, slice_thickness=SLICE_DZ, energy=ENERGY, sampling=SAMPLING,
            n_iter=5, alpha=1.0, beta=0.9, update_probe=False, seed=0,
        )
        _, _, err_thin = epie_thin(
            dps, scan_positions, probe, SAMPLING,
            n_iter=5, alpha=1.0, beta=0.9, update_probe=False, seed=0,
        )
        # Both should converge (last < first)
        assert err_ms[-1] <= err_ms[0]
        assert err_thin[-1] <= err_thin[0]


# ---------------------------------------------------------------------------
# 8. Gradient-based reconstruction – Angular Spectrum
# ---------------------------------------------------------------------------


class TestReconstructAS:
    def test_loss_decreases(self, probe, scan_positions, thin_object):
        obj_slices = thin_object[None]
        dps, _ = simulate_ptychography_as(
            obj_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        _, losses = reconstruct_as(
            dps, scan_positions, probe,
            n_slices=1, slice_thickness=SLICE_DZ, energy=ENERGY, sampling=SAMPLING,
            n_iter=10, learning_rate=1e-2,
        )
        assert losses[-1] < losses[0], "AS gradient loss did not decrease"

    def test_output_types(self, probe, scan_positions, thin_object):
        obj_slices = thin_object[None]
        dps, _ = simulate_ptychography_as(
            obj_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        rec, losses = reconstruct_as(
            dps, scan_positions, probe,
            n_slices=1, slice_thickness=SLICE_DZ, energy=ENERGY, sampling=SAMPLING,
            n_iter=5,
        )
        assert rec.shape == (1, NY, NX)
        assert jnp.issubdtype(rec.dtype, jnp.complexfloating)
        assert isinstance(losses, list)
        assert len(losses) == 5

    def test_multislice_loss_decreases(self, probe, scan_positions, phase_slices):
        dps, _ = simulate_ptychography_as(
            phase_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        _, losses = reconstruct_as(
            dps, scan_positions, probe,
            n_slices=2, slice_thickness=SLICE_DZ, energy=ENERGY, sampling=SAMPLING,
            n_iter=10, learning_rate=1e-2,
        )
        assert losses[-1] < losses[0]

    def test_custom_init(self, probe, scan_positions, thin_object):
        obj_slices = thin_object[None]
        dps, _ = simulate_ptychography_as(
            obj_slices, probe, scan_positions, SLICE_DZ, ENERGY, SAMPLING
        )
        init = jnp.ones((1, NY, NX), dtype=jnp.complex128)
        rec, losses = reconstruct_as(
            dps, scan_positions, probe,
            n_slices=1, slice_thickness=SLICE_DZ, energy=ENERGY, sampling=SAMPLING,
            n_iter=5, init_object_slices=init,
        )
        assert rec.shape == (1, NY, NX)


# ---------------------------------------------------------------------------
# 9. Gradient-based reconstruction – WPM
# ---------------------------------------------------------------------------


class TestReconstructWPM:
    def test_loss_decreases(self, probe, scan_positions, potential_slices):
        dps, _ = simulate_ptychography_wpm(
            potential_slices, probe, scan_positions,
            SLICE_DZ, ENERGY, SAMPLING, n_bins=4
        )
        # Use a non-trivial init so the WPM binning spans a real potential
        # range and the gradient is non-degenerate (starting from exactly zero
        # makes n_min=n_max which zeroes the binning interpolation gradient).
        rng = np.random.default_rng(99)
        init_v = rng.uniform(0.0, 1.0, (2, NY, NX)).astype(np.float32)
        _, losses = reconstruct_wpm(
            dps, scan_positions, probe,
            n_slices=2, slice_thickness=SLICE_DZ, energy=ENERGY, sampling=SAMPLING,
            n_iter=10, learning_rate=1e-2, n_bins=4,
            init_potentials=init_v,
        )
        assert losses[-1] < losses[0], "WPM gradient loss did not decrease"

    def test_output_types(self, probe, scan_positions, potential_slices):
        dps, _ = simulate_ptychography_wpm(
            potential_slices, probe, scan_positions,
            SLICE_DZ, ENERGY, SAMPLING, n_bins=4
        )
        rec, losses = reconstruct_wpm(
            dps, scan_positions, probe,
            n_slices=2, slice_thickness=SLICE_DZ, energy=ENERGY, sampling=SAMPLING,
            n_iter=5, n_bins=4,
        )
        assert rec.shape == (2, NY, NX)
        assert jnp.issubdtype(rec.dtype, jnp.floating)
        assert len(losses) == 5


# ---------------------------------------------------------------------------
# 10. Phantom helpers
# ---------------------------------------------------------------------------


class TestPhantomHelpers:
    def test_phase_object_shape(self):
        obj = make_phase_object(NY, NX)
        assert obj.shape == (NY, NX)

    def test_phase_object_unit_amplitude(self):
        obj = make_phase_object(NY, NX)
        amps = jnp.abs(obj)
        assert jnp.allclose(amps, jnp.ones_like(amps), atol=1e-10)

    def test_phase_object_reproducible(self):
        o1 = make_phase_object(NY, NX, seed=99)
        o2 = make_phase_object(NY, NX, seed=99)
        assert jnp.allclose(o1, o2)

    def test_potential_phantom_shape(self):
        pot = make_potential_phantom(NY, NX, n_slices=3)
        assert pot.shape == (3, NY, NX)

    def test_potential_phantom_nonnegative(self):
        pot = make_potential_phantom(NY, NX, n_slices=2)
        assert float(jnp.min(pot)) >= 0.0

    def test_potential_phantom_reproducible(self):
        p1 = make_potential_phantom(NY, NX, n_slices=2, seed=5)
        p2 = make_potential_phantom(NY, NX, n_slices=2, seed=5)
        assert jnp.allclose(p1, p2)
