"""Focused tests for the minimal free-atom ptychography model."""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (  # noqa: E402
    fresnel_propagation_kernel_1d,
)
from wide_angle_propagation.ptychography_1d import (  # noqa: E402
    simulate_glancing_scan_1d,
)
from wide_angle_propagation.ptychography_atoms_1d import (  # noqa: E402
    FreeAtomModel1D,
    free_atom_repulsion_1d,
    make_atom_template_1d,
    make_si_atom_template_1d,
    reconstruct_free_atoms_1d,
    render_free_atoms_1d,
    render_species_mixture_atoms_1d,
    uniform_atom_candidates_1d,
)


def _gaussian_model():
    sampling = 0.25
    coordinates_s = jnp.arange(41) * sampling
    coordinates_u = (jnp.arange(41) - 20) * sampling
    relative = (jnp.arange(9) - 4) * sampling
    grid_s, grid_u = jnp.meshgrid(relative, relative, indexing="ij")
    template = jnp.exp(-0.5 * (grid_s**2 + grid_u**2) / 0.2**2)
    bounds = jnp.asarray([[2.0, 8.0], [-3.0, 3.0]])
    candidates = uniform_atom_candidates_1d(bounds, (3, 2))
    return FreeAtomModel1D(
        coordinates_s,
        coordinates_u,
        template,
        bounds,
        candidates,
    )


def test_zero_unit_and_linear_occupancy_rendering():
    model = _gaussian_model()
    position = jnp.asarray([[5.0, 0.0]])
    zero = render_free_atoms_1d(model, position, jnp.zeros(1))
    unit = render_free_atoms_1d(model, position, jnp.ones(1))
    partial = render_free_atoms_1d(model, position, jnp.asarray([0.35]))

    np.testing.assert_array_equal(np.asarray(zero), 0.0)
    center_s = int(np.argmin(np.abs(np.asarray(model.axial_coordinates) - 5.0)))
    center_u = int(np.argmin(np.abs(np.asarray(model.transverse_coordinates))))
    np.testing.assert_allclose(
        np.asarray(unit)[center_s - 4 : center_s + 5, center_u - 4 : center_u + 5],
        np.asarray(model.atom_template),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(partial, 0.35 * unit, rtol=2e-6, atol=2e-7)


def test_species_mixture_interpolates_fixed_templates():
    model = _gaussian_model()
    position = jnp.asarray([[5.0, 0.0]])
    templates = jnp.stack([model.atom_template, 2.0 * model.atom_template])
    mixture = render_species_mixture_atoms_1d(
        model,
        templates,
        position,
        jnp.ones(1),
        jnp.asarray([[0.25, 0.75]]),
    )
    reference = 1.75 * render_free_atoms_1d(model, position, jnp.ones(1))
    np.testing.assert_allclose(mixture, reference, rtol=2e-6, atol=2e-7)

    with pytest.raises(ValueError, match="sum to one"):
        render_species_mixture_atoms_1d(
            model, templates, position, jnp.ones(1), jnp.asarray([[0.2, 0.2]])
        )


def test_generic_si_template_matches_convenience_wrapper():
    pytest.importorskip("abtem")
    pytest.importorskip("ase")
    np.testing.assert_allclose(
        make_atom_template_1d("Si", 0.5, 0.5),
        make_si_atom_template_1d(0.5, 0.5),
    )


def test_position_and_occupancy_gradients_match_finite_differences():
    model = _gaussian_model()
    position = jnp.asarray([[5.13, 0.17]])
    occupancy = jnp.asarray([0.63])
    weights = jnp.linspace(0.2, 1.3, 41 * 41).reshape(41, 41)

    def objective(values):
        return jnp.sum(render_free_atoms_1d(model, values[:2][None], values[2:]) * weights)

    values = jnp.asarray([position[0, 0], position[0, 1], occupancy[0]])
    automatic = jax.grad(objective)(values)
    step = 1e-4
    finite = []
    for index in range(3):
        offset = jnp.zeros(3).at[index].set(step)
        finite.append((objective(values + offset) - objective(values - offset)) / (2 * step))
    np.testing.assert_allclose(automatic, jnp.asarray(finite), rtol=3e-3, atol=2e-4)


def test_local_renderer_matches_dense_renderer_and_keeps_fixed_background():
    dense_model = _gaussian_model()
    fixed = jnp.full((41, 41), 0.25)
    positions = dense_model.initial_positions + jnp.asarray([0.11, -0.08])
    occupancies = jnp.linspace(0.1, 0.9, positions.shape[0])
    dense_with_background = FreeAtomModel1D(
        dense_model.axial_coordinates,
        dense_model.transverse_coordinates,
        dense_model.atom_template,
        dense_model.candidate_bounds,
        dense_model.initial_positions,
        fixed_potential=fixed,
    )
    local_model = FreeAtomModel1D(
        dense_model.axial_coordinates,
        dense_model.transverse_coordinates,
        dense_model.atom_template,
        dense_model.candidate_bounds,
        dense_model.initial_positions,
        fixed_potential=fixed,
        maximum_displacement_A=0.5,
    )

    dense = render_free_atoms_1d(dense_with_background, positions, occupancies)
    local = render_free_atoms_1d(local_model, positions, occupancies)
    np.testing.assert_allclose(local, dense, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        jax.grad(lambda value: render_free_atoms_1d(local_model, value, occupancies).sum())(
            positions
        ),
        jax.grad(
            lambda value: render_free_atoms_1d(
                dense_with_background, value, occupancies
            ).sum()
        )(positions),
        rtol=3e-5,
        atol=3e-5,
    )


def test_repulsion_ignores_dormant_and_separated_candidates():
    near = jnp.asarray([[0.0, 0.0], [1.0, 0.0]])
    far = jnp.asarray([[0.0, 0.0], [2.1, 0.0]])
    assert float(free_atom_repulsion_1d(near, jnp.ones(2))) > 0.0
    assert float(free_atom_repulsion_1d(near, jnp.asarray([1.0, 0.0]))) == 0.0
    assert float(free_atom_repulsion_1d(far, jnp.ones(2))) == 0.0


def _three_atom_problem():
    energy = 5e3
    sampling = 0.25
    n_s = n_u = 48
    coordinates_s = jnp.arange(n_s) * sampling
    coordinates_u = (jnp.arange(n_u) - n_u // 2) * sampling
    bounds = jnp.asarray([[2.0, 10.0], [-2.5, 2.5]])
    model = FreeAtomModel1D(
        coordinates_s,
        coordinates_u,
        make_si_atom_template_1d(sampling, sampling),
        bounds,
        uniform_atom_candidates_1d(bounds, (3, 3)),
    )
    truth_positions = jnp.asarray([[3.25, -1.15], [5.95, 0.15], [8.35, 1.2]])
    truth = render_free_atoms_1d(model, truth_positions, jnp.ones(3))
    kernel = fresnel_propagation_kernel_1d(n_u, sampling, sampling, energy)
    axial_starts = np.arange(0, 25, 3, dtype=np.int32)
    probe_centres = np.asarray([-1.25, 0.0, 1.25])
    starts = jnp.asarray(np.repeat(axial_starts, probe_centres.size))
    centres = np.tile(probe_centres, axial_starts.size)
    probes = jnp.stack(
        [
            jnp.exp(-0.5 * ((coordinates_u - centre) / 1.0) ** 2)
            * jnp.exp(0.1j * coordinates_u)
            for centre in centres
        ]
    )
    measured = simulate_glancing_scan_1d(
        truth, probes, starts, 24, kernel, sampling, energy
    )
    return model, truth_positions, probes, starts, kernel, measured, sampling, energy


def test_fast_three_atom_reconstruction_reduces_loss_and_localizes_atoms():
    model, truth, probes, starts, kernel, measured, sampling, energy = (
        _three_atom_problem()
    )
    validation = np.arange(0, len(starts), 5)
    result = reconstruct_free_atoms_1d(
        model,
        probes,
        starts,
        24,
        kernel,
        sampling,
        energy,
        measured,
        validation_indices=validation,
        updates=800,
        occupancy_only_updates=50,
        minibatch_size=6,
        validation_interval=20,
    )
    active = np.asarray(result.occupancies) >= 0.5
    assert np.min(np.asarray(result.validation_loss_history)[1:]) < np.asarray(
        result.validation_loss_history
    )[0]
    assert int(np.sum(active)) == 3
    distances = np.linalg.norm(
        np.asarray(result.positions)[active, None] - np.asarray(truth)[None], axis=-1
    )
    rows, columns = linear_sum_assignment(distances)
    assert np.sqrt(np.mean(distances[rows, columns] ** 2)) < 0.25
    assert np.all(np.asarray(result.position_history) >= np.asarray(model.candidate_bounds)[:, 0])
    assert np.all(np.asarray(result.position_history) <= np.asarray(model.candidate_bounds)[:, 1])
    assert np.all(np.asarray(result.occupancy_history) >= 0.0)
    assert np.all(np.asarray(result.occupancy_history) <= 1.0)


def _nine_atom_problem():
    energy = 5e3
    sampling = 0.25
    n_s, n_u = 96, 64
    coordinates_s = jnp.arange(n_s) * sampling
    coordinates_u = (jnp.arange(n_u) - n_u // 2) * sampling
    bounds = jnp.asarray([[6.0, 18.0], [-4.0, 4.0]])
    model = FreeAtomModel1D(
        coordinates_s,
        coordinates_u,
        make_si_atom_template_1d(sampling, sampling),
        bounds,
        uniform_atom_candidates_1d(bounds, (8, 3)),
        metadata={"species": "Si", "candidate_source": "uniform beam region"},
    )
    spacing = 2.35
    row_height = spacing * np.sqrt(3.0) / 2.0
    truth_positions = []
    for row in range(3):
        row_positions = 8.0 + (row % 2) * spacing / 2.0 + np.arange(4) * spacing
        for column, axial in enumerate(row_positions):
            if (row == 1 and column == 1) or (row == 2 and column in (0, 3)):
                continue
            index = len(truth_positions)
            truth_positions.append(
                [
                    axial + 0.16 * np.sin(0.9 * (index + 1)),
                    -2.0 + row * row_height + 0.12 * np.cos(0.7 * (index + 1)),
                ]
            )
    truth_positions = jnp.asarray(truth_positions)
    truth_potential = render_free_atoms_1d(model, truth_positions, jnp.ones(9))
    kernel = fresnel_propagation_kernel_1d(n_u, sampling, sampling, energy)
    axial_starts = np.arange(0, 49, 3, dtype=np.int32)
    probe_centres = np.asarray([-2.0, 0.0, 2.0])
    starts = jnp.asarray(np.repeat(axial_starts, probe_centres.size))
    centres = np.tile(probe_centres, axial_starts.size)
    probes = jnp.stack(
        [
            jnp.exp(-0.5 * ((coordinates_u - centre) / 1.2) ** 2)
            * jnp.exp(0.12j * coordinates_u)
            for centre in centres
        ]
    )
    measured = simulate_glancing_scan_1d(
        truth_potential, probes, starts, 48, kernel, sampling, energy
    )
    return model, truth_positions, probes, starts, kernel, measured, sampling, energy


@pytest.mark.slow
def test_nine_atom_gate_recovers_structure_without_truth_initialization():
    model, truth, probes, starts, kernel, measured, sampling, energy = (
        _nine_atom_problem()
    )
    validation = np.arange(0, len(starts), 5)
    result = reconstruct_free_atoms_1d(
        model,
        probes,
        starts,
        48,
        kernel,
        sampling,
        energy,
        measured,
        validation_indices=validation,
        updates=1000,
        occupancy_only_updates=200,
        minibatch_size=9,
        validation_interval=20,
    )
    active = np.asarray(result.occupancies) >= 0.5
    recovered = np.asarray(result.positions)[active]
    assert recovered.shape == (9, 2)
    distances = np.linalg.norm(recovered[:, None] - np.asarray(truth)[None], axis=-1)
    rows, columns = linear_sum_assignment(distances)
    position_rmse = np.sqrt(np.mean(distances[rows, columns] ** 2))
    assert position_rmse <= 0.25
    assert np.min(pdist(recovered)) >= 1.8
    assert np.nanmin(np.asarray(result.validation_loss_history)) < 1e-3

    bounds = np.asarray(model.candidate_bounds)
    cutoff = 4.0
    coordinates_s = np.asarray(model.axial_coordinates)
    coordinates_u = np.asarray(model.transverse_coordinates)
    pixel_mask = (
        (coordinates_s[:, None] >= bounds[0, 0] - cutoff)
        & (coordinates_s[:, None] <= bounds[0, 1] + cutoff)
        & (coordinates_u[None, :] >= bounds[1, 0] - cutoff)
        & (coordinates_u[None, :] <= bounds[1, 1] + cutoff)
    )
    assert int(np.sum(pixel_mask)) / result.metadata["n_specimen_parameters"] >= 50.0
