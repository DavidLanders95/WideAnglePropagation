"""Tests for the ptychography module.

These are lightweight tests that verify the module's core building blocks
work correctly without running a full reconstruction (which would be slow).
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from wide_angle_propagation.ptychography import (
    make_probe,
    generate_scan_positions,
    forward_model,
    normalised_mse,
    pearson_correlation,
    make_simple_sample,
    move_probe,
    amplitude_loss,
    reconstruct,
)
from wide_angle_propagation.propagation import (
    fresnel_propagation_kernel,
    angular_spectrum_propagation_kernel,
    energy2wavelength,
)

ENERGY = 200e3          # 200 keV
GPTS = (32, 32)
SAMPLING = (0.5, 0.5)  # Å
SEMI_ANGLE = 20.0      # mrad
SLICE_DZ = 2.0          # Å


# ============================================================================
# Probe
# ============================================================================

class TestMakeProbe:
    def test_shape(self):
        probe = make_probe(GPTS, SAMPLING, ENERGY, SEMI_ANGLE)
        assert probe.shape == GPTS

    def test_normalised(self):
        probe = make_probe(GPTS, SAMPLING, ENERGY, SEMI_ANGLE)
        assert jnp.isclose(jnp.sum(jnp.abs(probe) ** 2), 1.0, atol=1e-6)

    def test_complex(self):
        probe = make_probe(GPTS, SAMPLING, ENERGY, SEMI_ANGLE)
        assert jnp.iscomplexobj(probe)


# ============================================================================
# Scan positions
# ============================================================================

class TestScanPositions:
    def test_number_of_positions(self):
        pos = generate_scan_positions(GPTS, SAMPLING, n_positions=16)
        assert pos.shape == (16, 2)  # 4×4 grid

    def test_positions_within_bounds(self):
        pos = generate_scan_positions(GPTS, SAMPLING, n_positions=9)
        assert np.all(pos[:, 0] >= 0) and np.all(pos[:, 0] < GPTS[0])
        assert np.all(pos[:, 1] >= 0) and np.all(pos[:, 1] < GPTS[1])


# ============================================================================
# Simple sample
# ============================================================================

class TestMakeSimpleSample:
    def test_single_slice(self):
        pot = make_simple_sample(GPTS, SAMPLING, 0.2, SLICE_DZ, ENERGY)
        assert pot.shape[0] == 1  # 0.2 nm = 2 Å → 1 slice

    def test_multiple_slices(self):
        pot = make_simple_sample(GPTS, SAMPLING, 5.0, SLICE_DZ, ENERGY)
        assert pot.shape[0] == 25  # 50 Å / 2 Å = 25 slices

    def test_has_features(self):
        pot = make_simple_sample(GPTS, SAMPLING, 1.0, SLICE_DZ, ENERGY)
        # Should have non-zero values where features are
        assert float(jnp.max(pot)) > 0.0

    def test_thick_sample(self):
        pot = make_simple_sample(GPTS, SAMPLING, 50.0, SLICE_DZ, ENERGY)
        assert pot.shape[0] == 250  # 500 Å / 2 Å


# ============================================================================
# Forward model
# ============================================================================

class TestForwardModel:
    @pytest.fixture
    def setup(self):
        pot = make_simple_sample(GPTS, SAMPLING, 1.0, SLICE_DZ, ENERGY)
        probe = make_probe(GPTS, SAMPLING, ENERGY, SEMI_ANGLE)
        pos = generate_scan_positions(GPTS, SAMPLING, n_positions=4)
        return pot, probe, pos

    def test_fresnel_output_shape(self, setup):
        pot, probe, pos = setup
        pk = fresnel_propagation_kernel(
            GPTS[0], GPTS[1], SAMPLING, SLICE_DZ, ENERGY,
        )
        dps = forward_model(
            pot, probe, pos, "fresnel", SLICE_DZ, ENERGY, SAMPLING,
            prop_kernel=pk,
        )
        assert dps.shape == (4, GPTS[0], GPTS[1])

    def test_angular_spectrum_output_shape(self, setup):
        pot, probe, pos = setup
        pk = angular_spectrum_propagation_kernel(
            GPTS[0], GPTS[1], SAMPLING, SLICE_DZ, ENERGY,
        )
        dps = forward_model(
            pot, probe, pos, "angular_spectrum", SLICE_DZ, ENERGY, SAMPLING,
            prop_kernel=pk,
        )
        assert dps.shape == (4, GPTS[0], GPTS[1])

    def test_wpm_output_shape(self, setup):
        pot, probe, pos = setup
        dps = forward_model(
            pot, probe, pos, "wpm", SLICE_DZ, ENERGY, SAMPLING,
            n_bins=32, power_spacing=2.0,
        )
        assert dps.shape == (4, GPTS[0], GPTS[1])

    def test_diffraction_non_negative(self, setup):
        pot, probe, pos = setup
        pk = fresnel_propagation_kernel(
            GPTS[0], GPTS[1], SAMPLING, SLICE_DZ, ENERGY,
        )
        dps = forward_model(
            pot, probe, pos, "fresnel", SLICE_DZ, ENERGY, SAMPLING,
            prop_kernel=pk,
        )
        assert jnp.all(dps >= 0)


# ============================================================================
# Loss & metrics
# ============================================================================

class TestMetrics:
    def test_nmse_identical(self):
        a = jnp.ones((2, 8, 8))
        assert normalised_mse(a, a) == pytest.approx(0.0, abs=1e-10)

    def test_nmse_different(self):
        a = jnp.ones((2, 8, 8))
        b = jnp.zeros((2, 8, 8))
        assert normalised_mse(a, b) > 0

    def test_pearson_identical(self):
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert pearson_correlation(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_amplitude_loss_zero(self):
        dp = jnp.ones((8, 8))
        assert float(amplitude_loss(dp, dp)) == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Reconstruction (tiny smoke test)
# ============================================================================

class TestReconstruction:
    def test_loss_decreases(self):
        """Verify that the loss actually decreases during reconstruction."""
        gpts = (16, 16)
        sampling = (0.5, 0.5)
        pot = make_simple_sample(gpts, sampling, 0.2, SLICE_DZ, ENERGY)
        probe = make_probe(gpts, sampling, ENERGY, SEMI_ANGLE)
        pos = generate_scan_positions(gpts, sampling, n_positions=4)

        pk = fresnel_propagation_kernel(
            gpts[0], gpts[1], sampling, SLICE_DZ, ENERGY,
        )
        measured = forward_model(
            pot, probe, pos, "fresnel", SLICE_DZ, ENERGY, sampling,
            prop_kernel=pk,
        )

        recon, losses = reconstruct(
            measured, pos, "fresnel", gpts,
            n_slices=1, slice_thickness=SLICE_DZ,
            energy=ENERGY, sampling=sampling,
            semi_angle_mrad=SEMI_ANGLE,
            n_iterations=10, learning_rate=0.05,
            verbose=False,
        )

        assert recon.shape == pot.shape
        # Loss should decrease from first to last iteration
        assert losses[-1] < losses[0]
