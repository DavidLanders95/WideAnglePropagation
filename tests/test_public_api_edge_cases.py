"""Focused public API behavior tests for small edge cases."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ase")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (
    angular_spectrum_propagation_kernel,
    simulate_fresnel_as,
    simulate_wpm,
    wpm_step_adaptive,
)
from wide_angle_propagation.notebook_utils import (
    simulate_fresnel_as_exit_only,
    simulate_wpm_exit_only,
)


def test_simulate_fresnel_as_handles_empty_potential_stack():
    probe = jnp.ones((4, 4), dtype=jnp.complex128)
    potential = jnp.empty((0, 4, 4), dtype=jnp.float64)
    identity_kernel = jnp.ones((4, 4), dtype=jnp.complex128)

    exit_wave, diffraction_pattern, wavefronts = simulate_fresnel_as(
        potential,
        probe,
        identity_kernel,
        slice_thickness=1.0,
        energy=200e3,
    )

    np.testing.assert_allclose(np.asarray(exit_wave), np.asarray(probe))
    np.testing.assert_allclose(
        np.asarray(diffraction_pattern),
        np.abs(np.fft.fftshift(np.fft.fft2(np.asarray(probe)))) ** 2,
    )
    assert wavefronts.shape == (0, 4, 4)
    assert wavefronts.dtype == probe.dtype


def test_simulate_wpm_handles_empty_potential_stack():
    probe = jnp.ones((4, 4), dtype=jnp.complex128)
    potential = jnp.empty((0, 4, 4), dtype=jnp.float64)

    exit_wave, diffraction_pattern, wavefronts = simulate_wpm(
        potential,
        probe,
        slice_thickness=1.0,
        energy=200e3,
        sampling=(0.1, 0.1),
    )

    np.testing.assert_allclose(np.asarray(exit_wave), np.asarray(probe))
    np.testing.assert_allclose(
        np.asarray(diffraction_pattern),
        np.abs(np.fft.fftshift(np.fft.fft2(np.asarray(probe)))) ** 2,
    )
    assert wavefronts.shape == (0, 4, 4)
    assert wavefronts.dtype == probe.dtype


def test_wpm_step_adaptive_rejects_invalid_bin_settings():
    wave = jnp.ones((4, 4), dtype=jnp.complex128)
    refractive_index = jnp.ones((4, 4), dtype=jnp.float64)

    with pytest.raises(ValueError, match="n_bins"):
        wpm_step_adaptive(wave, refractive_index, 1.0, 200e3, (0.1, 0.1), n_bins=1)

    with pytest.raises(ValueError, match="power_spacing"):
        wpm_step_adaptive(
            wave,
            refractive_index,
            1.0,
            200e3,
            (0.1, 0.1),
            power_spacing=0.0,
        )

    with pytest.raises(ValueError, match="bin_batch_size"):
        wpm_step_adaptive(
            wave,
            refractive_index,
            1.0,
            200e3,
            (0.1, 0.1),
            bin_batch_size=0,
        )


def test_wpm_step_adaptive_preserves_singleton_spatial_dimension():
    wave = jnp.ones((5, 1), dtype=jnp.complex128)
    refractive_index = jnp.linspace(1.0, 1.001, 5)[:, None]

    full, _, _, _ = wpm_step_adaptive(
        wave,
        refractive_index,
        0.2,
        200e3,
        (0.1, 0.1),
        n_bins=5,
    )
    batched, _, _, _ = wpm_step_adaptive(
        wave,
        refractive_index,
        0.2,
        200e3,
        (0.1, 0.1),
        n_bins=5,
        bin_batch_size=2,
    )

    assert full.shape == wave.shape
    assert batched.shape == wave.shape
    np.testing.assert_allclose(
        np.asarray(batched), np.asarray(full), rtol=1e-13, atol=1e-13
    )


def test_scan_exit_only_helpers_match_full_solvers():
    probe = jnp.ones((8, 8), dtype=jnp.complex128)
    y, x = jnp.mgrid[:8, :8]
    one_slice = 10.0 + jnp.cos(2.0 * jnp.pi * x / 8.0)
    potential = jnp.stack([one_slice, 0.5 * one_slice])
    sampling = (0.2, 0.2)
    thickness = 0.5
    energy = 200e3
    kernel = angular_spectrum_propagation_kernel(
        8, 8, sampling, z=thickness, energy=energy
    )

    expected_as, _, _ = simulate_fresnel_as(
        potential, probe, kernel, thickness, energy
    )
    actual_as = simulate_fresnel_as_exit_only(
        potential, probe, kernel, thickness, energy
    )
    np.testing.assert_allclose(
        np.asarray(actual_as), np.asarray(expected_as), rtol=1e-13, atol=1e-13
    )

    expected_wpm, _, _ = simulate_wpm(
        potential, probe, thickness, energy, sampling, n_bins=8
    )
    actual_wpm = simulate_wpm_exit_only(
        potential, probe, thickness, energy, sampling, n_bins=8
    )
    np.testing.assert_allclose(
        np.asarray(actual_wpm), np.asarray(expected_wpm), rtol=1e-13, atol=1e-13
    )

    batched_wpm = simulate_wpm_exit_only(
        potential,
        probe,
        thickness,
        energy,
        sampling,
        n_bins=8,
        bin_batch_size=3,
    )
    np.testing.assert_allclose(
        np.asarray(batched_wpm), np.asarray(expected_wpm), rtol=1e-13, atol=1e-13
    )


def test_scan_exit_only_helpers_promote_complex64_carry_when_needed():
    probe = jnp.ones((4, 4), dtype=jnp.complex64)
    potential = jnp.ones((2, 4, 4), dtype=jnp.float64)
    sampling = (0.2, 0.2)
    thickness = 0.5
    energy = 200e3
    kernel = angular_spectrum_propagation_kernel(
        4, 4, sampling, z=thickness, energy=energy
    )

    propagated_as = simulate_fresnel_as_exit_only(
        potential, probe, kernel, thickness, energy
    )
    propagated_wpm = simulate_wpm_exit_only(
        potential,
        probe,
        thickness,
        energy,
        sampling,
        n_bins=5,
        bin_batch_size=2,
    )

    assert propagated_as.shape == probe.shape
    assert propagated_wpm.shape == probe.shape
    assert propagated_as.dtype == jnp.complex128
    assert propagated_wpm.dtype == jnp.complex128
