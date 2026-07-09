import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ase")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (
    project_atoms_to_sample_line_1d,
    simulate_glancing_fresnel_baseline_1d,
    simulate_single_slice_cylindrical_1d,
)
from wide_angle_propagation.sideview_geometry import line_from_angle, make_tilted_gaussian_beam_1d


ENERGY = 100e3
N = 24
DX = 0.15
DZ = 0.5


def test_gradients_through_beam_tilt_and_slice_potential():
    coords = (jnp.arange(N) - N // 2) * DX
    profile = jnp.exp(-0.5 * (coords / 0.4) ** 2)

    def objective(beam_tilt, strength):
        wave = make_tilted_gaussian_beam_1d(coords, ENERGY, waist=0.5, tilt=beam_tilt)
        potential = (strength * profile)[None, :]
        exit_wave, _, _ = simulate_glancing_fresnel_baseline_1d(
            wave,
            potential,
            DX,
            DZ,
            ENERGY,
        )
        return jnp.real(exit_wave[2])

    grads = jax.grad(objective, argnums=(0, 1))(0.02, 3.0)
    assert all(np.isfinite(np.asarray(g)) for g in grads)


def test_gradients_through_sample_tilt_and_atom_position():
    coords = (jnp.arange(N) - N // 2) * DX
    input_line = line_from_angle(jnp.array([0.0, -3.0]), 0.0, coords)
    output_line = line_from_angle(jnp.array([0.0, 3.0]), 0.0, coords)
    atom_positions = jnp.array([[0.05, 0.0], [0.4, 0.0]])
    sigmas = jnp.array([0.18, 0.22])
    input_wave = jnp.exp(-0.5 * (coords / 0.5) ** 2).astype(jnp.complex128)

    def objective(sample_tilt, atom_x):
        sample_line = line_from_angle(jnp.array([0.0, 0.0]), sample_tilt, coords)
        shifted_atoms = atom_positions.at[0, 0].set(atom_x)
        projected = project_atoms_to_sample_line_1d(
            sample_line,
            shifted_atoms,
            jnp.array([1.0, 0.7]),
            sigmas,
        )
        output_wave, _, _ = simulate_single_slice_cylindrical_1d(
            input_wave,
            input_line,
            sample_line,
            output_line,
            projected,
            ENERGY,
        )
        return jnp.real(output_wave[N // 2])

    grads = jax.grad(objective, argnums=(0, 1))(0.1, 0.05)
    assert all(np.isfinite(np.asarray(g)) for g in grads)
