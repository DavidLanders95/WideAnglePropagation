import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ase")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (
    angular_spectrum_propagation_kernel_1d,
    diffraction_intensity_1d,
    fourier_propagate_1d,
    fresnel_propagation_kernel_1d,
    get_frequencies_1d,
    simulate_glancing_angular_spectrum_1d,
    simulate_glancing_fresnel_baseline_1d,
)


ENERGY = 200e3
N = 64
DX = 0.2
DZ = 1.0


def test_get_frequencies_1d_matches_numpy():
    np.testing.assert_allclose(
        np.asarray(get_frequencies_1d(8, 0.25)),
        np.fft.fftfreq(8, 0.25),
    )


def test_vacuum_plane_wave_preserved_by_1d_kernels():
    wave = jnp.ones(N, dtype=jnp.complex128)
    for kernel in [
        fresnel_propagation_kernel_1d(N, DX, DZ, ENERGY),
        angular_spectrum_propagation_kernel_1d(N, DX, DZ, ENERGY),
    ]:
        propagated = fourier_propagate_1d(wave, kernel)
        assert np.allclose(np.abs(np.asarray(propagated)), 1.0, atol=1e-10)


def test_fresnel_and_angular_spectrum_agree_for_small_angle_frequency():
    freq_bin = 1
    coords = jnp.arange(N)
    wave = jnp.exp(2j * jnp.pi * freq_bin * coords / N)

    fresnel = fourier_propagate_1d(
        wave,
        fresnel_propagation_kernel_1d(N, DX, 0.1, ENERGY),
    )
    angular_spectrum = fourier_propagate_1d(
        wave,
        angular_spectrum_propagation_kernel_1d(N, DX, 0.1, ENERGY),
    )
    phase_aligned = fresnel * jnp.vdot(fresnel, angular_spectrum) / jnp.abs(jnp.vdot(fresnel, angular_spectrum))
    rel_err = jnp.linalg.norm(phase_aligned - angular_spectrum) / jnp.linalg.norm(angular_spectrum)
    assert rel_err < 1e-5


def test_fresnel_baseline_gradient_through_potential_strength():
    wave = jnp.ones(N, dtype=jnp.complex128)
    base = jnp.exp(-0.5 * (jnp.linspace(-2.0, 2.0, N) / 0.5) ** 2)

    def objective(strength):
        potential = (strength * base)[None, :]
        exit_wave, _, _ = simulate_glancing_fresnel_baseline_1d(
            wave,
            potential,
            DX,
            DZ,
            ENERGY,
        )
        return jnp.real(exit_wave[3])

    grad = jax.grad(objective)(2.0)
    assert np.isfinite(np.asarray(grad))


def test_glancing_angular_spectrum_matches_direct_split_step():
    wave = jnp.exp(-0.5 * (jnp.linspace(-2.0, 2.0, N) / 0.5) ** 2).astype(
        jnp.complex128
    )
    potential = jnp.zeros((2, N), dtype=jnp.float64)

    propagated, _, diagnostics = simulate_glancing_angular_spectrum_1d(
        wave,
        potential,
        DX,
        DZ,
        ENERGY,
    )
    kernel = angular_spectrum_propagation_kernel_1d(N, DX, DZ, ENERGY)
    expected = fourier_propagate_1d(fourier_propagate_1d(wave, kernel), kernel)

    np.testing.assert_allclose(np.asarray(propagated), np.asarray(expected), atol=1e-10)
    assert diagnostics["wavefronts"].shape == potential.shape


def test_diffraction_intensity_1d_shape_and_nonnegative():
    wave = jnp.ones(N, dtype=jnp.complex128)
    intensity = diffraction_intensity_1d(wave)
    assert intensity.shape == (N,)
    assert np.all(np.asarray(intensity) >= 0.0)
