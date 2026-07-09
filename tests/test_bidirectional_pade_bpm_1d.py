import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ase")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (
    apply_pade_rational_1d,
    build_sideview_operator_x_1d,
    pade_sqrt_coefficients,
    simulate_bidirectional_pade_bpm_1d,
)


ENERGY = 100e3
N = 12
DX = 0.3
DZ = 0.5


def _eval_rational(coeffs, x):
    p, q = coeffs
    numerator = sum(float(c) * x**i for i, c in enumerate(np.asarray(p)))
    denominator = sum(float(c) * x**i for i, c in enumerate(np.asarray(q)))
    return numerator / denominator


def test_pade_sqrt_coefficients_match_known_1_1_values():
    p, q = pade_sqrt_coefficients((1, 1))
    np.testing.assert_allclose(np.asarray(p), [1.0, 0.75])
    np.testing.assert_allclose(np.asarray(q), [1.0, 0.25])


def test_higher_order_pade_improves_scalar_sqrt_approximation():
    x = 0.8
    exact = np.sqrt(1.0 + x)
    err_11 = abs(_eval_rational(pade_sqrt_coefficients((1, 1)), x) - exact)
    err_22 = abs(_eval_rational(pade_sqrt_coefficients((2, 2)), x) - exact)
    assert err_22 < err_11


def test_apply_pade_rational_identity_operator():
    operator = jnp.zeros((N, N), dtype=jnp.complex128)
    wave = jnp.linspace(0.0, 1.0, N).astype(jnp.complex128)
    out = apply_pade_rational_1d(operator, wave, pade_order=(1, 1))
    np.testing.assert_allclose(np.asarray(out), np.asarray(wave), atol=1e-12)


def test_build_sideview_operator_shape_and_finite_values():
    potential = jnp.zeros(N)
    operator, n0 = build_sideview_operator_x_1d(potential, DX, ENERGY)
    assert operator.shape == (N, N)
    assert np.isfinite(np.asarray(n0))
    assert np.all(np.isfinite(np.asarray(operator)))


def test_bidirectional_pade_vacuum_has_zero_reflection():
    wave = jnp.ones(N, dtype=jnp.complex128)
    potential = jnp.zeros((2, N))
    transmitted, reflected, intensity, diagnostics = simulate_bidirectional_pade_bpm_1d(
        potential,
        wave,
        DX,
        DZ,
        ENERGY,
        pade_order=(1, 1),
        n_sweeps=3,
    )

    assert transmitted.shape == wave.shape
    assert intensity.shape == wave.shape
    assert diagnostics["residual_per_sweep"].shape == (3,)
    np.testing.assert_allclose(np.asarray(reflected), 0.0, atol=1e-12)
    assert float(diagnostics["reflected_power"]) < 1e-20


def test_bidirectional_pade_interface_reflection_and_gradient_are_finite():
    wave = jnp.ones(N, dtype=jnp.complex128)
    profile = jnp.exp(-0.5 * (jnp.linspace(-1.0, 1.0, N) / 0.4) ** 2)

    def objective(strength):
        potential = jnp.stack([jnp.zeros(N), strength * profile])
        transmitted, reflected, _, diagnostics = simulate_bidirectional_pade_bpm_1d(
            potential,
            wave,
            DX,
            DZ,
            ENERGY,
            pade_order=(1, 1),
            n_sweeps=2,
        )
        return jnp.real(transmitted[0]) + diagnostics["reflected_power"] + jnp.real(reflected[0])

    value = objective(5.0)
    grad = jax.grad(objective)(5.0)
    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(grad))
