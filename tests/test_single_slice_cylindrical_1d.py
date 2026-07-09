import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ase")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (
    cylindrical_green_asymptotic_1d,
    project_atoms_to_sample_line_1d,
    simulate_single_slice_cylindrical_1d,
)
from wide_angle_propagation.sideview_geometry import line_from_angle


ENERGY = 100e3


def _lines(n=48, dx=0.1):
    coords = (jnp.arange(n) - n // 2) * dx
    input_line = line_from_angle(jnp.array([0.0, -5.0]), 0.0, coords)
    sample_line = line_from_angle(jnp.array([0.0, 0.0]), 0.1, coords)
    output_line = line_from_angle(jnp.array([0.0, 5.0]), 0.0, coords)
    return coords, input_line, sample_line, output_line


def test_cylindrical_kernel_has_inverse_sqrt_amplitude_trend():
    r1 = 10.0
    r2 = 40.0
    g1 = cylindrical_green_asymptotic_1d(r1, ENERGY)
    g2 = cylindrical_green_asymptotic_1d(r2, ENERGY)
    ratio = jnp.abs(g1) / jnp.abs(g2)
    np.testing.assert_allclose(np.asarray(ratio), np.sqrt(r2 / r1), rtol=1e-12)


def test_single_slice_cylindrical_returns_finite_output():
    coords, input_line, sample_line, output_line = _lines()
    input_wave = jnp.exp(-0.5 * (coords / 0.7) ** 2).astype(jnp.complex128)
    projected = jnp.zeros_like(coords)

    output_wave, intensity, diagnostics = simulate_single_slice_cylindrical_1d(
        input_wave,
        input_line,
        sample_line,
        output_line,
        projected,
        ENERGY,
    )

    assert output_wave.shape == coords.shape
    assert intensity.shape == coords.shape
    assert diagnostics["sample_wave"].shape == coords.shape
    assert np.all(np.isfinite(np.asarray(output_wave)))


def test_projected_atom_potential_gradient_through_strength_and_position():
    coords, _, sample_line, _ = _lines(n=32)
    atom_positions = jnp.array([[0.1, 0.0], [0.6, 0.1]])
    sigmas = jnp.array([0.2, 0.25])

    def objective(strength, x_shift):
        shifted = atom_positions.at[0, 0].set(atom_positions[0, 0] + x_shift)
        projected = project_atoms_to_sample_line_1d(
            sample_line,
            shifted,
            jnp.array([strength, 0.8]),
            sigmas,
        )
        return jnp.sum(projected * coords)

    grads = jax.grad(objective, argnums=(0, 1))(1.2, 0.05)
    assert all(np.isfinite(np.asarray(g)) for g in grads)
