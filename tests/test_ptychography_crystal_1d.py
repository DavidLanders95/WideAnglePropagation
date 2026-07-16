"""Tests for four-parameter JAX crystalline-host registration."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest


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
from wide_angle_propagation import ptychography_crystal_1d as crystal_module  # noqa: E402
from wide_angle_propagation.ptychography_crystal_1d import (  # noqa: E402
    CrystallineHostModel1D,
    CrystallineRegistrationParameters1D,
    CrystallineRegistrationResult1D,
    _balanced_amplitude_loss_1d,
    make_crystalline_host_model_1d,
    register_crystalline_host_1d,
    render_crystalline_host_1d,
    transform_crystalline_host_1d,
)


def _small_model():
    sampling = 0.25
    s = jnp.arange(65) * sampling
    u = (jnp.arange(65) - 32) * sampling
    relative = (jnp.arange(9) - 4) * sampling
    grid_s, grid_u = jnp.meshgrid(relative, relative, indexing="ij")
    template = jnp.exp(-0.5 * (grid_s**2 + grid_u**2) / 0.28**2)
    host = jnp.asarray(
        [
            [2.5, 0.0, -1.5],
            [4.7, 1.0, 0.4],
            [7.2, -0.5, -2.0],
            [9.8, 2.0, 1.2],
            [12.5, -1.5, -0.4],
            [14.0, 0.3, 2.0],
        ]
    )
    return make_crystalline_host_model_1d(
        s,
        u,
        template,
        host,
        axial_period_A=4.0,
        metadata={"host": "asymmetric test crystal"},
    )


def test_model_is_a_true_lean_host_type_and_validates_inputs():
    model = _small_model()
    assert isinstance(model, CrystallineHostModel1D)
    assert model.reference_positions_3d.shape == (6, 3)
    assert model.axial_period_A == 4.0

    with pytest.raises(ValueError, match="uniformly increasing"):
        make_crystalline_host_model_1d(
            [0.0, 0.5, 1.1],
            model.transverse_coordinates,
            model.atom_template,
            model.reference_positions_3d,
            axial_period_A=4.0,
        )
    with pytest.raises(ValueError, match="inside the specimen grid"):
        make_crystalline_host_model_1d(
            model.axial_coordinates,
            model.transverse_coordinates,
            model.atom_template,
            model.reference_positions_3d.at[0, 0].set(-2.0),
            axial_period_A=4.0,
        )
    with pytest.raises(ValueError, match="positive and finite"):
        make_crystalline_host_model_1d(
            model.axial_coordinates,
            model.transverse_coordinates,
            model.atom_template,
            model.reference_positions_3d,
            axial_period_A=0.0,
        )


def test_transform_identity_known_values_and_latent_y_invariance():
    reference = jnp.asarray(
        [[0.0, 2.0, -1.0], [2.0, 3.0, 0.0], [4.0, 4.0, 1.0]]
    )
    identity = transform_crystalline_host_1d(reference, 0.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(identity, reference, atol=1e-12)

    transformed = transform_crystalline_host_1d(
        reference,
        axial_phase_A=0.7,
        surface_offset_A=-0.2,
        rotation_rad=jnp.pi / 2,
        axial_strain=0.1,
    )
    projected = np.asarray(reference)[:, [0, 2]]
    center = projected.mean(axis=0)
    relative = projected - center
    relative[:, 0] *= 1.1
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    expected_projected = relative @ rotation.T + center + [0.7, -0.2]
    np.testing.assert_allclose(transformed[:, [0, 2]], expected_projected, atol=1e-12)
    np.testing.assert_allclose(transformed[:, 1], reference[:, 1], atol=0.0)


def test_render_is_jittable_and_has_finite_four_parameter_gradients():
    model = _small_model()

    def objective(values):
        positions = transform_crystalline_host_1d(
            model.reference_positions_3d,
            values[0],
            values[1],
            values[2],
            values[3],
        )
        potential = render_crystalline_host_1d(model, positions)
        grid_s, grid_u = jnp.meshgrid(
            model.axial_coordinates,
            model.transverse_coordinates,
            indexing="ij",
        )
        weights = jnp.sin(0.17 * grid_s + 0.31 * grid_u)
        return jnp.sum(potential * weights)

    values = jnp.asarray([0.2, -0.1, 0.005, 0.002])
    value, gradients = jax.jit(jax.value_and_grad(objective))(values)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradients)))
    assert np.all(np.abs(np.asarray(gradients)) > 1e-8)


def test_balanced_loss_normalizes_bands_independently_and_ignores_padding():
    measured = jnp.asarray([[4.0, 1.0, 9.0, 16.0], [1.0, 4.0, 4.0, 1.0]])
    predicted = jnp.asarray([[1.0, 4.0, 9.0, 25.0], [4.0, 1.0, 9.0, 1.0]])
    scan_weights = jnp.ones(2)
    reflected = jnp.asarray([False, False, True, True])
    actual = _balanced_amplitude_loss_1d(
        predicted,
        measured,
        scan_weights,
        reflected,
        whole_detector_weight=0.5,
    )
    amplitude_error = (np.sqrt(np.asarray(predicted)) - np.sqrt(np.asarray(measured))) ** 2
    all_loss = amplitude_error.sum() / np.asarray(measured).sum()
    reflected_loss = amplitude_error[:, 2:].sum() / np.asarray(measured)[:, 2:].sum()
    np.testing.assert_allclose(actual, 0.5 * (all_loss + reflected_loss), rtol=1e-10)

    padded = _balanced_amplitude_loss_1d(
        jnp.pad(predicted, ((0, 1), (0, 0))),
        jnp.pad(measured, ((0, 1), (0, 0))),
        jnp.asarray([1.0, 1.0, 0.0]),
        reflected,
        whole_detector_weight=0.5,
    )
    np.testing.assert_allclose(padded, actual, atol=1e-12)


def _registration_problem(*, truth_parameters=None):
    model = _small_model()
    truth = truth_parameters or CrystallineRegistrationParameters1D(
        axial_phase_A=0.35,
        surface_offset_A=-0.16,
        rotation_rad=np.deg2rad(0.28),
        axial_strain=-0.004,
    )
    truth_positions = transform_crystalline_host_1d(
        model.reference_positions_3d,
        truth.axial_phase_A,
        truth.surface_offset_A,
        truth.rotation_rad,
        truth.axial_strain,
    )
    truth_potential = render_crystalline_host_1d(model, truth_positions)
    u = model.transverse_coordinates
    centers = np.linspace(-3.0, 3.0, 9)
    probes = jnp.stack(
        [
            jnp.exp(-0.5 * ((u - center) / 1.2) ** 2)
            * jnp.exp(1j * (0.04 + 0.01 * index) * u)
            for index, center in enumerate(centers)
        ]
    ).astype(jnp.complex128)
    starts = jnp.zeros(len(probes), dtype=jnp.int32)
    energy = 5e3
    sampling = 0.25
    kernel = fresnel_propagation_kernel_1d(
        truth_potential.shape[1], sampling, sampling, energy
    )
    measured = simulate_glancing_scan_1d(
        truth_potential,
        probes,
        starts,
        truth_potential.shape[0],
        kernel,
        sampling,
        energy,
    )
    detector_angles = jnp.linspace(-40.0, 40.0, truth_potential.shape[1])
    return model, truth, probes, starts, kernel, measured, detector_angles


def test_padded_batch_objective_matches_one_unpadded_batch():
    model, _, probes, starts, kernel, measured, detector_angles = (
        _registration_problem(
            truth_parameters=CrystallineRegistrationParameters1D()
        )
    )
    common = (
        model,
        probes,
        starts,
        measured.shape[1],
        kernel,
        0.25,
        5e3,
        measured,
        detector_angles,
    )
    settings = dict(
        reflected_angle_bounds_mrad=(0.0, 35.0),
        specular_angle_bounds_mrad=(10.0, 30.0),
        phase_grid_points=5,
        updates=1,
    )
    padded = register_crystalline_host_1d(
        *common, batch_size=4, **settings
    )
    unpadded = register_crystalline_host_1d(
        *common, batch_size=len(probes), **settings
    )
    np.testing.assert_allclose(
        padded.phase_grid_objective,
        unpadded.phase_grid_objective,
        rtol=2e-12,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        padded.initial_objective, unpadded.initial_objective, atol=2e-14
    )


def test_phase_grid_selects_the_correct_periodic_basin():
    model, truth, probes, starts, kernel, measured, detector_angles = (
        _registration_problem()
    )
    initial = CrystallineRegistrationParameters1D(
        axial_phase_A=-0.9,
        surface_offset_A=truth.surface_offset_A,
        rotation_rad=truth.rotation_rad,
        axial_strain=truth.axial_strain,
    )
    result = register_crystalline_host_1d(
        model,
        probes,
        starts,
        measured.shape[1],
        kernel,
        0.25,
        5e3,
        measured,
        detector_angles,
        initial_parameters=initial,
        reflected_angle_bounds_mrad=(0.0, 35.0),
        specular_angle_bounds_mrad=(10.0, 30.0),
        batch_size=4,
        phase_grid_points=17,
        updates=1,
    )
    grid_spacing = model.axial_period_A / 17
    assert (
        abs(
            float(result.optimization_start_parameters.axial_phase_A)
            - truth.axial_phase_A
        )
        <= grid_spacing / 2
    )


def test_adam_deterministically_recovers_known_four_parameter_transform():
    model, truth, probes, starts, kernel, measured, detector_angles = (
        _registration_problem()
    )
    initial = CrystallineRegistrationParameters1D(
        axial_phase_A=-0.9,
        surface_offset_A=-0.05,
        rotation_rad=np.deg2rad(0.15),
        axial_strain=-0.002,
    )
    common = (
        model,
        probes,
        starts,
        measured.shape[1],
        kernel,
        0.25,
        5e3,
        measured,
        detector_angles,
    )
    settings = dict(
        initial_parameters=initial,
        reflected_angle_bounds_mrad=(0.0, 35.0),
        specular_angle_bounds_mrad=(10.0, 30.0),
        batch_size=4,
        phase_grid_points=17,
        updates=200,
        learning_rate_start=5e-2,
        learning_rate_end=1e-3,
    )
    result = register_crystalline_host_1d(*common, **settings)
    repeated = register_crystalline_host_1d(*common, **settings)
    np.testing.assert_array_equal(
        result.objective_history, repeated.objective_history
    )
    np.testing.assert_array_equal(
        result.parameter_history, repeated.parameter_history
    )
    assert isinstance(result, CrystallineRegistrationResult1D)
    assert result.metadata["n_parameters"] == 4
    assert result.objective_history.shape == (201,)
    assert result.parameter_history.shape == (201, 4)
    assert bool(result.converged)
    assert float(result.objective_history[-1]) < 0.2 * float(result.initial_objective)
    assert abs(float(result.parameters.axial_phase_A) - truth.axial_phase_A) < 0.12
    assert abs(float(result.parameters.surface_offset_A) - truth.surface_offset_A) < 0.12
    assert abs(float(result.parameters.rotation_rad) - truth.rotation_rad) < np.deg2rad(0.12)
    assert abs(float(result.parameters.axial_strain) - truth.axial_strain) < 0.002
    scales = np.asarray([model.axial_period_A / 2, 1.0, np.deg2rad(1.0), 0.02])
    final_values = np.asarray(
        [
            result.parameters.axial_phase_A,
            result.parameters.surface_offset_A,
            result.parameters.rotation_rad,
            result.parameters.axial_strain,
        ]
    )
    assert np.all(np.abs(final_values) <= scales)
    recovered = np.asarray(result.host_positions_3d)
    expected = np.asarray(
        transform_crystalline_host_1d(
            model.reference_positions_3d,
            truth.axial_phase_A,
            truth.surface_offset_A,
            truth.rotation_rad,
            truth.axial_strain,
        )
    )
    assert np.sqrt(np.mean((recovered[:, [0, 2]] - expected[:, [0, 2]]) ** 2)) < 0.12


def test_registration_rejects_invalid_bands_nonfinite_data_and_bounds():
    model, _, probes, starts, kernel, measured, detector_angles = _registration_problem(
        truth_parameters=CrystallineRegistrationParameters1D()
    )
    common = (
        model,
        probes,
        starts,
        measured.shape[1],
        kernel,
        0.25,
        5e3,
        measured,
        detector_angles,
    )
    with pytest.raises(ValueError, match="contains no pixels"):
        register_crystalline_host_1d(
            *common,
            reflected_angle_bounds_mrad=(100.0, 120.0),
            updates=1,
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        register_crystalline_host_1d(
            *common[:7],
            measured.at[0, 0].set(jnp.nan),
            detector_angles,
            updates=1,
        )
    with pytest.raises(ValueError, match="finite and real"):
        register_crystalline_host_1d(
            *common[:-1],
            detector_angles.at[0].set(jnp.nan),
            updates=1,
        )
    with pytest.raises(ValueError, match="input_probe must be finite"):
        register_crystalline_host_1d(
            common[0],
            probes.at[0, 0].set(jnp.nan),
            *common[2:],
            updates=1,
        )
    with pytest.raises(TypeError, match="must contain integers"):
        register_crystalline_host_1d(
            common[0], common[1], starts.astype(float), *common[3:], updates=1
        )
    with pytest.raises(ValueError, match="outside fit bounds"):
        register_crystalline_host_1d(
            *common,
            initial_parameters=CrystallineRegistrationParameters1D(
                surface_offset_A=1.2
            ),
            updates=1,
        )


def test_public_api_has_no_defect_persistence_or_host_aliases():
    expected = {
        "CrystallineHostModel1D",
        "CrystallineRegistrationParameters1D",
        "CrystallineRegistrationResult1D",
        "make_crystalline_host_model_1d",
        "register_crystalline_host_1d",
        "render_crystalline_host_1d",
        "transform_crystalline_host_1d",
    }
    assert set(crystal_module.__all__) == expected
    removed = {
        "CrystallineDefectModel1D",
        "CrystallineDefectReconstruction1D",
        "build_diamond_neighbor_graph_1d",
        "keating_lattice_energy_1d",
        "reconstruct_crystalline_defects_1d",
        "reconstruct_crystalline_host_1d",
        "save_crystalline_defect_reconstruction_1d",
        "load_crystalline_defect_reconstruction_1d",
    }
    assert all(not hasattr(crystal_module, name) for name in removed)
    source = Path(crystal_module.__file__).read_text(encoding="utf-8")
    assert "from scipy" not in source
    assert "import scipy" not in source


def test_registration_notebook_is_clean_python_without_deleted_workflows():
    notebook_path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "sideview_glancing_ptychography_1d.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert not cell.get("outputs")
        source = "".join(cell["source"])
        source_without_magics = "\n".join(
            line
            for line in source.splitlines()
            if not line.lstrip().startswith("%")
        )
        ast.parse(source_without_magics, filename=f"notebook cell {index}")
        code.append(source)
    joined = "\n".join(code)
    removed_workflow_names = {
        "CrystallineDefect",
        "keating_lattice_energy_1d",
        "reconstruct_crystalline_host_1d",
        "save_crystalline_defect_reconstruction_1d",
        "load_crystalline_defect_reconstruction_1d",
        "validation_indices",
        "training_indices",
        "host_update_weights",
        "local_displacement",
        "cKDTree",
        "FuncAnimation",
        "dataset_path",
        "reconstruction_path",
    }
    assert all(name not in joined for name in removed_workflow_names)
    assert "scipy" not in joined.lower()
    assert ".gif" not in joined.lower()
