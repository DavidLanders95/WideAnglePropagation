"""Tests for sparse crystalline-host defect ptychography."""

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (  # noqa: E402
    fresnel_propagation_kernel_1d,
)
from wide_angle_propagation.ptychography_1d import simulate_glancing_scan_1d  # noqa: E402
from wide_angle_propagation.ptychography_crystal_1d import (  # noqa: E402
    CrystallineDefectReconstruction1D,
    build_diamond_neighbor_graph_1d,
    keating_lattice_energy_1d,
    load_crystalline_defect_reconstruction_1d,
    make_crystalline_defect_model_1d,
    make_crystalline_host_model_1d,
    reconstruct_crystalline_defects_1d,
    reconstruct_crystalline_host_1d,
    render_crystalline_defects_1d,
    render_crystalline_host_1d,
    save_crystalline_defect_reconstruction_1d,
    transform_crystalline_host_1d,
)


def _tetrahedron(bond_length=2.35):
    directions = np.asarray(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float
    ) / np.sqrt(3.0)
    return jnp.asarray(np.vstack([np.zeros(3), bond_length * directions]))


def test_sparse_graph_and_keating_invariances():
    positions = _tetrahedron()
    bonds, angles = build_diamond_neighbor_graph_1d(positions, bond_cutoff_A=2.5)
    assert bonds.shape == (4, 2)
    assert angles.shape == (6, 3)
    occupancies = jnp.ones(5)
    probabilities = jax.nn.one_hot(jnp.zeros(5, dtype=jnp.int32), 2)
    lengths = jnp.asarray([[2.35, 2.40], [2.40, 2.45]])

    ideal = keating_lattice_energy_1d(
        positions, occupancies, probabilities, bonds, angles, lengths
    )
    translated = keating_lattice_energy_1d(
        positions + jnp.asarray([4.0, -2.0, 1.0]),
        occupancies,
        probabilities,
        bonds,
        angles,
        lengths,
    )
    theta = 0.37
    rotation = jnp.asarray(
        [[jnp.cos(theta), -jnp.sin(theta), 0.0],
         [jnp.sin(theta), jnp.cos(theta), 0.0],
         [0.0, 0.0, 1.0]]
    )
    rotated = keating_lattice_energy_1d(
        positions @ rotation.T, occupancies, probabilities, bonds, angles, lengths
    )
    perturbed = positions.at[1, 0].add(0.25)
    strained = keating_lattice_energy_1d(
        perturbed, occupancies, probabilities, bonds, angles, lengths
    )

    assert float(ideal) < 1e-12
    np.testing.assert_allclose(translated, ideal, atol=1e-12)
    np.testing.assert_allclose(rotated, ideal, atol=1e-12)
    assert float(strained) > float(ideal) + 1e-5


def test_vacancy_gates_its_elastic_terms():
    positions = _tetrahedron()
    bonds, angles = build_diamond_neighbor_graph_1d(positions, bond_cutoff_A=2.5)
    probabilities = jax.nn.one_hot(jnp.zeros(5, dtype=jnp.int32), 2)
    lengths = jnp.asarray([[2.35, 2.40], [2.40, 2.45]])
    vacancy = jnp.ones(5).at[1].set(0.0)
    displaced_vacancy = positions.at[1].add(jnp.asarray([3.0, 1.0, -2.0]))
    reference = keating_lattice_energy_1d(
        positions, vacancy, probabilities, bonds, angles, lengths
    )
    actual = keating_lattice_energy_1d(
        displaced_vacancy, vacancy, probabilities, bonds, angles, lengths
    )
    np.testing.assert_allclose(actual, reference, atol=1e-12)


def test_neighbor_graph_storage_scales_linearly_for_a_chain():
    positions = np.zeros((1000, 3), dtype=float)
    positions[:, 0] = np.arange(1000)
    bonds, angles = build_diamond_neighbor_graph_1d(
        positions, bond_cutoff_A=1.01
    )
    assert bonds.shape[0] == 999
    assert angles.shape[0] == 998
    assert bonds.size + angles.size < 6 * len(positions)


def test_host_transform_is_exactly_frozen_outside_illumination_support():
    reference = jnp.asarray(
        [[0.0, 2.0, 0.0], [2.0, 3.0, -1.0], [4.0, 4.0, -2.0]]
    )
    weights = jnp.asarray([1.0, 0.5, 0.0])
    fully_transformed = transform_crystalline_host_1d(
        reference,
        jnp.asarray([0.7, -0.3]),
        jnp.asarray([[0.01, 0.02], [-0.01, -0.02]]),
        jnp.asarray(0.05),
        jnp.asarray([[0.1, 0.2], [-0.2, 0.1], [0.3, -0.1]]),
    )
    masked = transform_crystalline_host_1d(
        reference,
        jnp.asarray([0.7, -0.3]),
        jnp.asarray([[0.01, 0.02], [-0.01, -0.02]]),
        jnp.asarray(0.05),
        jnp.asarray([[0.1, 0.2], [-0.2, 0.1], [0.3, -0.1]]),
        weights,
    )

    np.testing.assert_allclose(masked[0], fully_transformed[0], atol=1e-12)
    np.testing.assert_allclose(masked[2], reference[2], atol=1e-12)
    expected_middle = reference[1] + 0.5 * (fully_transformed[1] - reference[1])
    np.testing.assert_allclose(masked[1], expected_middle, atol=1e-12)
    np.testing.assert_allclose(masked[:, 1], reference[:, 1], atol=1e-12)


def _small_model():
    sampling = 0.25
    s = jnp.arange(49) * sampling
    u = (jnp.arange(49) - 24) * sampling
    relative = (jnp.arange(9) - 4) * sampling
    grid_s, grid_u = jnp.meshgrid(relative, relative, indexing="ij")
    template = jnp.exp(-0.5 * (grid_s**2 + grid_u**2) / 0.22**2)
    templates = jnp.stack([template, 1.7 * template])
    host = jnp.asarray([[3.0, 0.0, -0.5], [5.35, 0.0, -0.5], [7.7, 0.0, -0.5]])
    adatoms = jnp.asarray([[5.0, 1.0], [7.0, 1.0]])
    return make_crystalline_defect_model_1d(
        s,
        u,
        templates,
        ("Si", "Ge"),
        host,
        jnp.asarray([[1.0, 10.0], [-2.0, 2.0]]),
        adatoms,
        jnp.asarray([[1.0, 10.0], [0.0, 2.0]]),
        species_bond_lengths_A=jnp.asarray([[2.35, 2.40], [2.40, 2.45]]),
        bond_cutoff_A=2.6,
        host_maximum_displacement_A=1.0,
        adatom_maximum_displacement_A=0.75,
    )


def test_model_validates_and_stores_site_update_weights():
    base = _small_model()
    weights = np.asarray([1.0, 0.25, 0.0])
    model = make_crystalline_host_model_1d(
        base.axial_coordinates,
        base.transverse_coordinates,
        base.species_templates[0],
        base.host_reference_positions_3d,
        jnp.asarray([[1.0, 10.0], [-2.0, 2.0]]),
        host_update_weights=weights,
        bond_cutoff_A=2.6,
    )
    np.testing.assert_allclose(model.host_update_weights, weights)

    with pytest.raises(ValueError, match="one value per host site"):
        make_crystalline_host_model_1d(
            base.axial_coordinates,
            base.transverse_coordinates,
            base.species_templates[0],
            base.host_reference_positions_3d,
            jnp.asarray([[1.0, 10.0], [-2.0, 2.0]]),
            host_update_weights=[1.0, 0.0],
            bond_cutoff_A=2.6,
        )


def test_mixed_rendering_and_all_defect_gradients_are_finite():
    model = _small_model()
    n_host = model.host_reference_positions_3d.shape[0]
    host_species_logits = jnp.zeros((n_host, 2))
    adatom_species_logits = jnp.zeros((2, 2))
    values = {
        # The final host moves just beyond the specimen grid while its template
        # still overlaps it. Boundary templates must remain renderable.
        "translation": jnp.asarray([5.0, -0.05]),
        "strain": jnp.asarray([[0.005, 0.0], [0.0, -0.004]]),
        "rotation": jnp.asarray(0.01),
        "displacements": jnp.zeros((n_host, 2)),
        "host_occupancies": jnp.asarray([1.0, 0.8, 1.0]),
        "host_species_logits": host_species_logits,
        "adatom_positions": model.adatom_initial_positions,
        "adatom_occupancies": jnp.asarray([0.4, 0.1]),
        "adatom_species_logits": adatom_species_logits,
    }

    def objective(parameters):
        host_positions = transform_crystalline_host_1d(
            model.host_reference_positions_3d,
            parameters["translation"],
            parameters["strain"],
            parameters["rotation"],
            parameters["displacements"],
        )
        potential = render_crystalline_defects_1d(
            model,
            host_positions,
            parameters["host_occupancies"],
            jax.nn.softmax(parameters["host_species_logits"], axis=-1),
            parameters["adatom_positions"],
            parameters["adatom_occupancies"],
            jax.nn.softmax(parameters["adatom_species_logits"], axis=-1),
        )
        weights = jnp.linspace(0.2, 1.3, potential.size).reshape(potential.shape)
        return jnp.sum(potential * weights)

    potential_value = objective(values)
    gradients = jax.grad(objective)(values)
    assert float(potential_value) > 0.0
    for gradient in jax.tree.leaves(gradients):
        assert np.all(np.isfinite(np.asarray(gradient)))


def test_compact_joint_solver_reduces_held_out_loss():
    model = _small_model()
    host_positions = model.host_reference_positions_3d
    host_occupancies = jnp.ones(3)
    host_species = jax.nn.one_hot(jnp.asarray([0, 1, 0]), 2)
    adatom_positions = model.adatom_initial_positions
    adatom_occupancies = jnp.asarray([1.0, 0.0])
    adatom_species = jax.nn.one_hot(jnp.asarray([0, 0]), 2)
    truth = render_crystalline_defects_1d(
        model,
        host_positions,
        host_occupancies,
        host_species,
        adatom_positions,
        adatom_occupancies,
        adatom_species,
    )
    energy = 5e3
    sampling = 0.25
    n_u = truth.shape[1]
    u = model.transverse_coordinates
    probe_centers = np.linspace(-1.0, 1.0, 9)
    probes = jnp.stack(
        [jnp.exp(-0.5 * ((u - center) / 0.9) ** 2) for center in probe_centers]
    ).astype(jnp.complex128)
    starts = jnp.zeros(len(probes), dtype=jnp.int32)
    kernel = fresnel_propagation_kernel_1d(n_u, sampling, sampling, energy)
    measured = simulate_glancing_scan_1d(
        truth, probes, starts, truth.shape[0], kernel, sampling, energy
    )
    result = reconstruct_crystalline_defects_1d(
        model,
        probes,
        starts,
        truth.shape[0],
        kernel,
        sampling,
        energy,
        measured,
        validation_indices=[0, 4, 8],
        updates=80,
        stage_global_end=5,
        stage_host_end=15,
        stage_defect_end=25,
        minibatch_size=3,
        validation_interval=10,
        evaluation_batch_size=3,
        keating_weight=0.0,
        host_occupancy_weight=0.0,
        substitution_weight=0.0,
        adatom_weight=0.0,
        displacement_weight=0.0,
        binary_weight=0.0,
        entropy_weight=0.0,
        repulsion_weight=0.0,
        seed=3,
    )
    history = np.asarray(result.validation_loss_history)
    assert np.nanmin(history[1:]) < history[0]
    assert result.host_species_probabilities.shape == (3, 2)
    assert result.adatom_species_probabilities.shape == (2, 2)


def test_pristine_host_wrapper_fixes_species_occupancy_and_empty_adatoms():
    defect_model = _small_model()
    model = make_crystalline_host_model_1d(
        defect_model.axial_coordinates,
        defect_model.transverse_coordinates,
        defect_model.species_templates[0],
        defect_model.host_reference_positions_3d,
        jnp.asarray([[1.0, 10.0], [-2.0, 2.0]]),
        equilibrium_bond_length_A=2.35,
        host_update_weights=jnp.asarray([1.0, 0.0, 0.0]),
        bond_cutoff_A=2.6,
    )
    truth_positions = transform_crystalline_host_1d(
        model.host_reference_positions_3d,
        jnp.asarray([0.2, -0.1]),
        jnp.zeros((2, 2)),
        jnp.asarray(0.0),
        jnp.zeros((3, 2)),
        model.host_update_weights,
    )
    truth = render_crystalline_host_1d(
        model, truth_positions
    )
    probes = jnp.stack(
        [jnp.exp(-0.5 * ((model.transverse_coordinates - center) / 0.9) ** 2)
         for center in np.linspace(-1.0, 1.0, 7)]
    ).astype(jnp.complex128)
    starts = jnp.zeros(len(probes), dtype=jnp.int32)
    kernel = fresnel_propagation_kernel_1d(truth.shape[1], 0.25, 0.25, 5e3)
    measured = simulate_glancing_scan_1d(
        truth, probes, starts, truth.shape[0], kernel, 0.25, 5e3
    )
    result = reconstruct_crystalline_host_1d(
        model,
        probes,
        starts,
        truth.shape[0],
        kernel,
        0.25,
        5e3,
        measured,
        validation_indices=[0, 3, 6],
        updates=4,
        stage_global_end=4,
        stage_host_end=4,
        stage_defect_end=4,
        minibatch_size=2,
        validation_interval=2,
        keating_weight=0.0,
        enable_host_displacements=False,
    )
    np.testing.assert_allclose(result.host_occupancies, 1.0)
    np.testing.assert_allclose(result.host_species_probabilities, 1.0)
    assert result.host_species_probabilities.shape == (3, 1)
    assert result.adatom_positions.shape == (0, 2)
    assert result.adatom_occupancies.shape == (0,)
    np.testing.assert_allclose(
        result.host_positions_3d[1:],
        model.host_reference_positions_3d[1:],
        atol=0.0,
    )


def _isolated_defect_result(*, host_occupancies, host_labels, adatom_occupancies,
                            enable_host_occupancies, enable_substitutions,
                            enable_adatoms, updates=260):
    model = _small_model()
    host_species = jax.nn.one_hot(jnp.asarray(host_labels), 2)
    adatom_species = jax.nn.one_hot(jnp.asarray([0, 0]), 2)
    truth = render_crystalline_defects_1d(
        model,
        model.host_reference_positions_3d,
        jnp.asarray(host_occupancies),
        host_species,
        model.adatom_initial_positions,
        jnp.asarray(adatom_occupancies),
        adatom_species,
    )
    u = model.transverse_coordinates
    probes = jnp.stack(
        [jnp.exp(-0.5 * ((u - center) / 0.9) ** 2)
         for center in np.linspace(-1.0, 1.0, 13)]
    ).astype(jnp.complex128)
    starts = jnp.zeros(len(probes), dtype=jnp.int32)
    kernel = fresnel_propagation_kernel_1d(truth.shape[1], 0.25, 0.25, 5e3)
    measured = simulate_glancing_scan_1d(
        truth, probes, starts, truth.shape[0], kernel, 0.25, 5e3
    )
    return reconstruct_crystalline_defects_1d(
        model,
        probes,
        starts,
        truth.shape[0],
        kernel,
        0.25,
        5e3,
        measured,
        validation_indices=[0, 6, 12],
        updates=updates,
        stage_global_end=0,
        stage_host_end=0,
        stage_defect_end=40,
        minibatch_size=5,
        validation_interval=20,
        evaluation_batch_size=3,
        keating_weight=0.0,
        host_occupancy_weight=0.0,
        substitution_weight=0.0,
        adatom_weight=0.0,
        displacement_weight=0.0,
        binary_weight=1e-3,
        entropy_weight=1e-3,
        repulsion_weight=0.0,
        initial_host_occupancy=1.0,
        initial_host_si_probability=0.95,
        enable_global_transform=False,
        enable_host_displacements=False,
        enable_host_occupancies=enable_host_occupancies,
        enable_substitutions=enable_substitutions,
        enable_adatoms=enable_adatoms,
        seed=2,
    )


def test_isolated_ge_substitution_is_localized():
    result = _isolated_defect_result(
        host_occupancies=[1.0, 1.0, 1.0],
        host_labels=[0, 1, 0],
        adatom_occupancies=[0.0, 0.0],
        enable_host_occupancies=False,
        enable_substitutions=True,
        enable_adatoms=False,
        updates=320,
    )
    ge_probability = np.asarray(result.host_species_probabilities)[:, 1]
    assert int(np.argmax(ge_probability)) == 1
    assert ge_probability[1] > 0.6


def test_isolated_vacancy_is_localized():
    result = _isolated_defect_result(
        host_occupancies=[1.0, 0.0, 1.0],
        host_labels=[0, 0, 0],
        adatom_occupancies=[0.0, 0.0],
        enable_host_occupancies=True,
        enable_substitutions=False,
        enable_adatoms=False,
    )
    occupancy = np.asarray(result.host_occupancies)
    assert int(np.argmin(occupancy)) == 1
    assert occupancy[1] + 0.2 < min(occupancy[0], occupancy[2])


def test_isolated_si_adatom_is_localized_without_false_candidate():
    result = _isolated_defect_result(
        host_occupancies=[1.0, 1.0, 1.0],
        host_labels=[0, 0, 0],
        adatom_occupancies=[1.0, 0.0],
        enable_host_occupancies=False,
        enable_substitutions=False,
        enable_adatoms=True,
    )
    occupancy = np.asarray(result.adatom_occupancies)
    assert occupancy[0] > occupancy[1]
    assert occupancy[0] > 0.3


def test_pristine_truth_does_not_create_confident_defects():
    result = _isolated_defect_result(
        host_occupancies=[1.0, 1.0, 1.0],
        host_labels=[0, 0, 0],
        adatom_occupancies=[0.0, 0.0],
        enable_host_occupancies=False,
        enable_substitutions=True,
        enable_adatoms=True,
    )
    assert float(jnp.max(result.host_species_probabilities[:, 1])) < 0.5
    assert float(jnp.max(result.adatom_occupancies)) < 0.5


def test_crystalline_result_round_trip(tmp_path):
    array = jnp.asarray([1.0, 2.0])
    result = CrystallineDefectReconstruction1D(
        host_positions_3d=jnp.ones((2, 3)),
        host_occupancies=array,
        host_species_probabilities=jnp.ones((2, 2)) / 2,
        adatom_positions=jnp.ones((1, 2)),
        adatom_occupancies=jnp.ones(1),
        adatom_species_probabilities=jnp.ones((1, 2)) / 2,
        translation=jnp.zeros(2),
        strain=jnp.zeros((2, 2)),
        rotation_rad=jnp.asarray(0.0),
        potential=jnp.ones((3, 4)),
        predicted_intensities=jnp.ones((2, 4)),
        measured_intensities=jnp.ones((2, 4)),
        update_history=jnp.asarray([0, 2]),
        elapsed_time_history=array,
        training_loss_history=array,
        validation_loss_history=array,
        translation_history=jnp.zeros((2, 2)),
        strain_history=jnp.zeros((2, 2, 2)),
        rotation_history=jnp.zeros(2),
        host_displacement_history=jnp.zeros((2, 2, 2)),
        host_occupancy_history=jnp.ones((2, 2)),
        host_species_probability_history=jnp.ones((2, 2, 2)) / 2,
        adatom_position_history=jnp.ones((2, 1, 2)),
        adatom_occupancy_history=jnp.ones((2, 1)),
        adatom_species_probability_history=jnp.ones((2, 1, 2)) / 2,
        best_update=2,
        metadata={"species_names": ["Si", "Ge"]},
    )
    path = tmp_path / "crystal_result.npz"
    save_crystalline_defect_reconstruction_1d(path, result)
    loaded = load_crystalline_defect_reconstruction_1d(path)
    np.testing.assert_allclose(loaded.host_positions_3d, result.host_positions_3d)
    np.testing.assert_allclose(loaded.adatom_species_probabilities, result.adatom_species_probabilities)
    assert loaded.best_update == 2
    assert loaded.metadata == result.metadata
