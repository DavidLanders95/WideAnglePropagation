import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ase")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (
    angular_spectrum_propagation_kernel_1d,
    build_atom_aligned_screen_partition_1d,
    fourier_propagate_1d,
    project_potential_to_screens_1d,
    simulate_glancing_angular_spectrum_1d,
    simulate_projected_as_screens_1d,
)


ENERGY = 200e3
DX = 0.2


def test_two_sub_screens_straddle_planes_and_integrate_exactly():
    axial = np.arange(0.0, 4.0, 0.25)
    planes = np.array([1.0, 3.0])
    axial_weight = np.exp(-0.5 * ((axial - 1.0) / 0.3) ** 2)
    axial_weight += np.exp(-0.5 * ((axial - 3.0) / 0.3) ** 2)
    reference = axial_weight[:, None] * np.ones((1, 8))
    partition = build_atom_aligned_screen_partition_1d(
        axial,
        planes,
        sub_screens_per_plane=2,
        reference_potential=reference,
    )

    assert partition.n_screens == 4
    assert partition.screen_positions[0] < planes[0]
    assert partition.screen_positions[1] >= planes[0]
    assert partition.screen_positions[2] < planes[1]
    assert partition.screen_positions[3] >= planes[1]

    potential = jnp.asarray(reference)
    projected = project_potential_to_screens_1d(potential, partition)
    np.testing.assert_allclose(
        np.asarray(projected).sum(axis=0),
        np.asarray(potential).sum(axis=0) * partition.fine_axial_sampling,
        rtol=1e-12,
        atol=1e-12,
    )


def test_three_sub_screens_keep_central_screens_on_atomic_planes():
    axial = np.arange(0.0, 4.0, 0.125)
    planes = np.array([1.0, 3.0])
    reference = np.exp(-0.5 * ((axial[:, None] - planes[0]) / 0.25) ** 2)
    reference += np.exp(-0.5 * ((axial[:, None] - planes[1]) / 0.25) ** 2)
    partition = build_atom_aligned_screen_partition_1d(
        axial,
        planes,
        sub_screens_per_plane=3,
        reference_potential=reference,
    )

    assert partition.n_screens == 6
    np.testing.assert_allclose(partition.screen_positions[[1, 4]], planes)
    projected = project_potential_to_screens_1d(jnp.asarray(reference), partition)
    np.testing.assert_allclose(
        np.asarray(projected).sum(axis=0),
        reference.sum(axis=0) * partition.fine_axial_sampling,
        rtol=1e-12,
        atol=1e-12,
    )


def test_projection_preserves_two_transverse_dimensions():
    axial = np.arange(6) * 0.2
    partition = build_atom_aligned_screen_partition_1d(
        axial,
        np.array([0.2, 0.8]),
        sub_screens_per_plane=3,
    )
    potential = jnp.arange(6 * 3 * 4, dtype=jnp.float32).reshape(6, 3, 4)
    projected = project_potential_to_screens_1d(potential, partition)

    assert projected.shape == (partition.n_screens, 3, 4)
    np.testing.assert_allclose(
        np.asarray(projected).sum(axis=0),
        np.asarray(potential).sum(axis=0) * partition.fine_axial_sampling,
        rtol=1e-6,
        atol=1e-6,
    )


def test_one_screen_per_fine_plane_matches_existing_as_multislice():
    n_s, n_u = 5, 32
    ds = 0.3
    axial = np.arange(n_s) * ds
    transverse = np.linspace(-2.0, 2.0, n_u)
    wave = jnp.exp(-0.5 * (transverse / 0.7) ** 2).astype(jnp.complex128)
    potential = jnp.asarray(
        np.stack(
            [0.2 * (index + 1) * np.exp(-0.5 * (transverse / 0.4) ** 2)
             for index in range(n_s)]
        )
    )
    partition = build_atom_aligned_screen_partition_1d(
        axial,
        axial,
        sub_screens_per_plane=1,
        reference_potential=potential,
    )
    projected = project_potential_to_screens_1d(potential, partition)
    screened, _ = simulate_projected_as_screens_1d(
        wave,
        projected,
        partition.screen_positions,
        DX,
        ENERGY,
        domain_start=partition.domain_start,
        domain_end=partition.domain_end,
        return_diagnostics=False,
    )
    reference, _, _ = simulate_glancing_angular_spectrum_1d(
        wave,
        potential,
        DX,
        ds,
        ENERGY,
    )
    np.testing.assert_allclose(
        np.asarray(screened),
        np.asarray(reference),
        rtol=1e-11,
        atol=1e-11,
    )


def test_nonuniform_vacuum_screens_equal_one_full_as_step():
    n_u = 32
    total_length = 3.0
    wave = jnp.exp(-0.5 * (jnp.linspace(-2.0, 2.0, n_u) / 0.6) ** 2).astype(
        jnp.complex128
    )
    positions = np.array([0.2, 0.9, 2.2])
    projected = jnp.zeros((len(positions), n_u), dtype=jnp.float64)
    screened, _ = simulate_projected_as_screens_1d(
        wave,
        projected,
        positions,
        DX,
        ENERGY,
        domain_start=0.0,
        domain_end=total_length,
        return_diagnostics=False,
    )
    expected = fourier_propagate_1d(
        wave,
        angular_spectrum_propagation_kernel_1d(
            n_u, DX, total_length, ENERGY
        ),
    )
    np.testing.assert_allclose(
        np.asarray(screened),
        np.asarray(expected),
        rtol=1e-11,
        atol=1e-11,
    )


def test_screen_projection_and_propagation_are_differentiable():
    n_s, n_u = 8, 16
    axial = np.arange(n_s) * 0.25
    partition = build_atom_aligned_screen_partition_1d(
        axial,
        np.array([0.5, 1.5]),
        sub_screens_per_plane=2,
    )
    profile = jnp.exp(-0.5 * (jnp.linspace(-1.0, 1.0, n_u) / 0.3) ** 2)
    wave = jnp.ones(n_u, dtype=jnp.complex128)

    def objective(strength):
        potential = strength * jnp.ones((n_s, 1)) * profile[None, :]
        projected = project_potential_to_screens_1d(potential, partition)
        exit_wave, _ = simulate_projected_as_screens_1d(
            wave,
            projected,
            partition.screen_positions,
            DX,
            ENERGY,
            domain_start=partition.domain_start,
            domain_end=partition.domain_end,
            return_diagnostics=False,
        )
        return jnp.real(exit_wave[2])

    assert np.isfinite(np.asarray(jax.grad(objective)(0.5)))


def test_complex64_wave_remains_complex64():
    wave = jnp.ones(16, dtype=jnp.complex64)
    axial = np.arange(4) * 0.25
    partition = build_atom_aligned_screen_partition_1d(
        axial, np.array([0.25, 0.75]), sub_screens_per_plane=1
    )
    projected = project_potential_to_screens_1d(
        jnp.zeros((4, 16), dtype=jnp.float32), partition
    )
    assert projected.dtype == jnp.float32
    exit_wave, intensity = simulate_projected_as_screens_1d(
        wave,
        projected,
        partition.screen_positions,
        np.float32(DX),
        np.float32(ENERGY),
        domain_start=partition.domain_start,
        domain_end=partition.domain_end,
        return_diagnostics=False,
    )
    assert exit_wave.dtype == jnp.complex64
    assert intensity.dtype == jnp.float32
