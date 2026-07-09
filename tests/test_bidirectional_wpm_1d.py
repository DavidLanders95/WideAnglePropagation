import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ase")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (
    interface_coupling_wpm_1d,
    simulate_bidirectional_wpm_1d,
    wpm_step_adaptive_1d,
)


ENERGY = 100e3
N = 32
DX = 0.25
DZ = 1.0


def test_wpm_step_adaptive_1d_preserves_vacuum_plane_wave_amplitude():
    wave = jnp.ones(N, dtype=jnp.complex128)
    n_map = jnp.ones(N)
    propagated, weights, idx_l, refs = wpm_step_adaptive_1d(
        wave,
        n_map,
        DZ,
        ENERGY,
        DX,
        n_bins=8,
    )
    assert propagated.shape == wave.shape
    assert weights.shape == wave.shape
    assert idx_l.shape == wave.shape
    assert refs.shape == (8,)
    np.testing.assert_allclose(np.abs(np.asarray(propagated)), 1.0, atol=1e-10)


def test_interface_coupling_wpm_1d_reflects_index_step():
    wave = jnp.ones(N, dtype=jnp.complex128)
    _, reflected, info = interface_coupling_wpm_1d(
        wave,
        jnp.ones(N),
        1.05 * jnp.ones(N),
        DX,
        ENERGY,
    )
    assert jnp.linalg.norm(reflected) > 0.0
    assert info["r"].shape == wave.shape


def test_bidirectional_wpm_vacuum_reflection_is_zero():
    wave = jnp.ones(N, dtype=jnp.complex128)
    potential = jnp.zeros((3, N))
    transmitted, reflected, intensity, diagnostics = simulate_bidirectional_wpm_1d(
        potential,
        wave,
        DX,
        DZ,
        ENERGY,
        n_bins=8,
        n_sweeps=3,
    )
    assert transmitted.shape == wave.shape
    assert intensity.shape == wave.shape
    assert diagnostics["residual_per_sweep"].shape == (3,)
    assert float(diagnostics["reflected_power"]) < 1e-20
    np.testing.assert_allclose(np.asarray(reflected), 0.0, atol=1e-12)


def test_bidirectional_wpm_gradient_through_potential():
    wave = jnp.ones(N, dtype=jnp.complex128)
    profile = jnp.exp(-0.5 * (jnp.linspace(-1.0, 1.0, N) / 0.3) ** 2)

    def objective(strength):
        potential = jnp.stack([jnp.zeros(N), strength * profile])
        transmitted, reflected, _, _ = simulate_bidirectional_wpm_1d(
            potential,
            wave,
            DX,
            DZ,
            ENERGY,
            n_bins=8,
            n_sweeps=2,
        )
        return jnp.real(transmitted[0]) + jnp.sum(jnp.abs(reflected) ** 2)

    grad = jax.grad(objective)(5.0)
    assert np.isfinite(np.asarray(grad))
