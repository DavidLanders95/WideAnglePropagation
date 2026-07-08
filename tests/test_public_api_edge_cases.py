"""Focused public API behavior tests for small edge cases."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ase")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (
    simulate_fresnel_as,
    simulate_wpm,
    wpm_step_adaptive,
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
