import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ase")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (
    angular_spectrum_propagation_kernel_1d,
    fourier_propagate_1d,
    simulate_bidirectional_pade_bpm_1d,
    simulate_bidirectional_wpm_1d,
    simulate_glancing_fresnel_baseline_1d,
)


ENERGY = 200e3
N = 16
DX = 0.4
DZ = 0.2


def _phase_aligned_error(reference, candidate):
    overlap = jnp.vdot(candidate, reference)
    candidate_aligned = candidate * overlap / jnp.abs(overlap)
    return jnp.linalg.norm(candidate_aligned - reference) / jnp.linalg.norm(reference)


def test_vacuum_fresnel_agrees_with_angular_spectrum_at_small_distance():
    wave = jnp.exp(-0.5 * (jnp.linspace(-1.0, 1.0, N) / 0.3) ** 2).astype(jnp.complex128)
    potential = jnp.zeros((1, N))

    fresnel_wave, _, _ = simulate_glancing_fresnel_baseline_1d(
        wave,
        potential,
        DX,
        DZ,
        ENERGY,
    )
    reference = fourier_propagate_1d(
        wave,
        angular_spectrum_propagation_kernel_1d(N, DX, DZ, ENERGY),
    )

    assert _phase_aligned_error(reference, fresnel_wave) < 1e-4


def test_bidirectional_methods_have_no_vacuum_reflection():
    wave = jnp.ones(N, dtype=jnp.complex128)
    potential = jnp.zeros((2, N))

    _, wpm_reflected, _, wpm_diag = simulate_bidirectional_wpm_1d(
        potential,
        wave,
        DX,
        DZ,
        ENERGY,
        n_bins=8,
        n_sweeps=2,
    )
    _, pade_reflected, _, pade_diag = simulate_bidirectional_pade_bpm_1d(
        potential,
        wave,
        DX,
        DZ,
        ENERGY,
        pade_order=(1, 1),
        n_sweeps=2,
    )

    np.testing.assert_allclose(np.asarray(wpm_reflected), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(pade_reflected), 0.0, atol=1e-12)
    assert float(wpm_diag["reflected_power"]) < 1e-20
    assert float(pade_diag["reflected_power"]) < 1e-20
