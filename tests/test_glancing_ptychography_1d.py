"""Focused tests for direct-potential 1D glancing ptychography."""

from dataclasses import replace

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

import wide_angle_propagation.ptychography_1d as ptychography_1d_module  # noqa: E402

from wide_angle_propagation.propagation_methods import (  # noqa: E402
    fresnel_propagation_kernel_1d,
    phase_grating_1d_from_projected_potential,
    simulate_glancing_fresnel_baseline_1d,
)
from wide_angle_propagation.ptychography_1d import (  # noqa: E402
    beam_path_reconstruction_region_1d,
    ConvergenceOptions1D,
    decompose_lattice_site_displacement_controls_1d,
    decompose_lattice_site_similarity_controls_1d,
    GlancingScan1D,
    LatticeOptimizationOptions1D,
    LatticeSiteModel1D,
    LatticeSiteReconstruction1D,
    PreparedLatticeSiteReconstruction1D,
    PotentialReconstruction1D,
    PtychographyMeasurement1D,
    PtychographyObjective1D,
    load_glancing_scan_1d,
    load_glancing_sideview_cache_1d,
    load_lattice_site_reconstruction_1d,
    load_potential_reconstruction_1d,
    lattice_site_displacements_1d,
    normalized_amplitude_loss_1d,
    prepare_lattice_site_reconstruction_1d,
    ptychography_expected_signal_electrons_1d,
    ptychography_objective_loss_1d,
    reconstruct_lattice_site_potential_1d,
    reconstruct_potential_1d,
    render_lattice_site_potential_1d,
    render_lattice_site_potential_from_displacements_1d,
    run_prepared_lattice_site_reconstruction_1d,
    save_glancing_scan_1d,
    save_glancing_sideview_cache_1d,
    save_lattice_site_reconstruction_1d,
    save_potential_reconstruction_1d,
    simulate_glancing_scan_1d,
    simulate_glancing_sideview_cache_1d,
    validate_ptychography_measurement_1d,
)


ENERGY = 30e3
N_U = 32
DU = 0.25
DS = 0.5


def _probe(n_u=N_U, du=DU):
    u = (jnp.arange(n_u) - n_u // 2) * du
    return jnp.exp(-0.5 * ((u + 0.15) / 0.7) ** 2) * jnp.exp(0.35j * u)


def _potential(n_s, n_u=N_U):
    u = (jnp.arange(n_u) - n_u // 2) * DU
    slice_strength = 1.0 + 0.17 * jnp.arange(n_s)
    transverse_profile = 90.0 * jnp.exp(-0.5 * ((u - 0.2) / 0.8) ** 2)
    return slice_strength[:, None] * transverse_profile[None, :]


def _kernel(n_u=N_U, du=DU, ds=DS):
    return fresnel_propagation_kernel_1d(n_u, du, ds, ENERGY)


def _small_lattice_model(maximum_displacement=0.5):
    shape = (8, 12)
    patch = np.array(
        [
            [0.0, 0.2, 0.0],
            [0.3, 2.0, 0.3],
            [0.0, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    starts = np.array([[2, 3], [4, 7]], dtype=np.int32)
    reference = np.full(shape, 0.05, dtype=np.float64)
    for start in starts:
        reference[
            start[0] : start[0] + patch.shape[0],
            start[1] : start[1] + patch.shape[1],
        ] += patch
    sites = np.column_stack(
        [
            (starts[:, 0] + 1) * 0.4,
            (starts[:, 1] + 1) * 0.3,
        ]
    )
    return LatticeSiteModel1D(
        reference_potential=jnp.asarray(reference),
        site_coordinates=jnp.asarray(sites),
        site_patches=jnp.asarray(np.stack([patch, patch])),
        patch_starts=jnp.asarray(starts),
        control_coordinates_s=jnp.array([0.0, (shape[0] - 1) * 0.4]),
        control_coordinates_u=jnp.array([0.0, (shape[1] - 1) * 0.3]),
        axial_sampling=0.4,
        transverse_sampling=0.3,
        maximum_displacement=maximum_displacement,
        metadata={"species": "Si"},
    )


@pytest.fixture(scope="module")
def prepared_small_lattice_problem():
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    model = _small_lattice_model(maximum_displacement=0.0)
    u = (jnp.arange(12) - 6) * 0.3
    base_probe = jnp.exp(-0.5 * ((u + 0.1) / 0.65) ** 2) * jnp.exp(0.25j * u)
    probes = jnp.stack([jnp.roll(base_probe, index - 2) for index in range(5)])
    starts = jnp.arange(5)
    kernel = fresnel_propagation_kernel_1d(12, 0.3, 0.4, ENERGY)
    target = render_lattice_site_potential_1d(
        model,
        jnp.array([0.65, 0.0]),
        jnp.zeros((2, 2, 2)),
    )
    measured = simulate_glancing_scan_1d(
        target, probes, starts, 4, kernel, 0.4, ENERGY
    )
    return prepare_lattice_site_reconstruction_1d(
        model,
        probes,
        starts,
        4,
        kernel,
        0.4,
        ENERGY,
        measured,
        validation_indices=[3],
        audit_indices=[4],
        potential_max=10.0,
        minibatch_size=2,
        evaluation_batch_size=3,
        rematerialize=False,
    )


@pytest.fixture(scope="module")
def prepared_small_poisson_problem(prepared_small_lattice_problem):
    source = prepared_small_lattice_problem
    objective = PtychographyObjective1D(
        kind="poisson_deviance",
        electrons_per_pattern=2_000.0,
        minimum_expected_electrons=1e-6,
    )
    signal = ptychography_expected_signal_electrons_1d(
        source.measured_intensities, source.probe_rows, objective
    )
    valid = jnp.ones(signal.shape, dtype=bool).at[0, 0].set(False)
    measurement = PtychographyMeasurement1D(
        calibrated_signal_electrons=signal.at[0, 0].set(jnp.nan),
        observed_total_electrons=(signal + 0.2).at[0, 0].set(-1e200),
        valid_mask=valid,
        calibrated_dark_electrons_per_pixel=0.2,
        calibrated_read_noise_std_electrons=0.0,
        calibration_id="matched_noiseless_poisson",
        metadata={"source": "focused_test"},
    )
    return prepare_lattice_site_reconstruction_1d(
        source.model,
        source.input_probe,
        source.window_starts,
        source.window_length,
        source.propagation_kernel,
        source.slice_thickness,
        source.energy,
        measurement=measurement,
        objective=objective,
        scan_coordinates=source.scan_coordinates,
        detector_angles=source.detector_angles,
        validation_indices=np.asarray(source.validation_indices),
        audit_indices=np.asarray(source.audit_indices),
        excluded_indices=np.asarray(source.excluded_indices),
        potential_max=source.potential_max,
        minibatch_size=source.minibatch_size,
        evaluation_batch_size=source.evaluation_batch_size,
        gradient_clip=source.gradient_clip,
        rematerialize=source.rematerialize,
    )


@pytest.fixture(scope="module")
def prepared_small_poisson_heldout_changed(prepared_small_poisson_problem):
    source = prepared_small_poisson_problem
    measurement = source.measurement
    assert measurement is not None and source.objective is not None
    held_out = jnp.asarray(
        np.concatenate(
            [
                np.asarray(source.validation_indices),
                np.asarray(source.audit_indices),
            ]
        )
    )
    changed = replace(
        measurement,
        calibrated_signal_electrons=(
            measurement.calibrated_signal_electrons.at[held_out].add(500.0)
        ),
        observed_total_electrons=(
            measurement.observed_total_electrons.at[held_out].add(500.0)
        ),
    )
    return _reprepare_calibrated(source, changed, source.objective)


@pytest.fixture(scope="module")
def small_poisson_reconstruction(prepared_small_poisson_problem):
    return _prepared_run(
        prepared_small_poisson_problem,
        updates=20,
        validation_interval=2,
        checkpoint_interval=1,
    )


def _prepared_run(prepared, **overrides):
    options = {
        "initial_vacancy_fractions": jnp.array([0.1, 0.0]),
        "learning_rate_start": 0.04,
        "learning_rate_end": 0.01,
        "updates": 3,
        "validation_interval": 1,
        "seed": 7,
    }
    options.update(overrides)
    return run_prepared_lattice_site_reconstruction_1d(prepared, **options)


def _reprepare_calibrated(prepared, measurement, objective):
    return prepare_lattice_site_reconstruction_1d(
        prepared.model,
        prepared.input_probe,
        prepared.window_starts,
        prepared.window_length,
        prepared.propagation_kernel,
        prepared.slice_thickness,
        prepared.energy,
        measurement=measurement,
        objective=objective,
        scan_coordinates=prepared.scan_coordinates,
        detector_angles=prepared.detector_angles,
        validation_indices=np.asarray(prepared.validation_indices),
        audit_indices=np.asarray(prepared.audit_indices),
        excluded_indices=np.asarray(prepared.excluded_indices),
        potential_max=prepared.potential_max,
        minibatch_size=prepared.minibatch_size,
        evaluation_batch_size=prepared.evaluation_batch_size,
        gradient_clip=prepared.gradient_clip,
        epsilon=prepared.epsilon,
        rematerialize=prepared.rematerialize,
    )


def _assert_reconstruction_trajectory_equal(first, second):
    for name in (
        "potential",
        "initial_potential",
        "vacancy_fractions",
        "displacement_controls",
        "rigid_displacement",
        "predicted_intensities",
        "update_history",
        "training_loss_history",
        "validation_loss_history",
        "gradient_norm_history",
        "normalized_step_history",
        "active_bound_fraction_history",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    np.testing.assert_array_equal(
        first.optimization_stage_history, second.optimization_stage_history
    )
    assert first.best_update == second.best_update
    assert first.completed_updates == second.completed_updates
    assert first.stop_reason == second.stop_reason
    assert first.audit_loss == second.audit_loss


def test_prepared_lattice_wrapper_matches_explicit_prepare_and_run(
    prepared_small_lattice_problem,
):
    prepared = prepared_small_lattice_problem
    assert isinstance(prepared, PreparedLatticeSiteReconstruction1D)
    assert prepared.preparation_time_s > 0.0
    assert len(prepared.reconstruction_problem_id) == 64
    assert prepared.reconstructor_id
    explicit = _prepared_run(prepared)
    wrapped = reconstruct_lattice_site_potential_1d(
        prepared.model,
        prepared.input_probe,
        prepared.window_starts,
        prepared.window_length,
        prepared.propagation_kernel,
        prepared.slice_thickness,
        prepared.energy,
        prepared.measured_intensities,
        initial_vacancy_fractions=jnp.array([0.1, 0.0]),
        scan_coordinates=prepared.scan_coordinates,
        detector_angles=prepared.detector_angles,
        validation_indices=np.asarray(prepared.validation_indices),
        audit_indices=np.asarray(prepared.audit_indices),
        excluded_indices=np.asarray(prepared.excluded_indices),
        potential_max=prepared.potential_max,
        learning_rate_start=0.04,
        learning_rate_end=0.01,
        updates=3,
        minibatch_size=prepared.minibatch_size,
        validation_interval=1,
        evaluation_batch_size=prepared.evaluation_batch_size,
        gradient_clip=prepared.gradient_clip,
        epsilon=prepared.epsilon,
        rematerialize=prepared.rematerialize,
        seed=7,
    )
    _assert_reconstruction_trajectory_equal(explicit, wrapped)
    assert explicit.metadata["prepared_api_version"] == 3
    assert explicit.metadata["preparation_time_s"] == prepared.preparation_time_s
    assert (
        explicit.metadata["reconstruction_problem_id"]
        == prepared.reconstruction_problem_id
        == wrapped.metadata["reconstruction_problem_id"]
    )
    assert explicit.metadata["reconstructor_id"] == prepared.reconstructor_id
    assert (
        explicit.metadata["elapsed_time_history_scope"]
        == "run_only_excludes_preparation"
    )


def test_prepared_runs_are_deterministic_without_state_leakage(
    prepared_small_lattice_problem,
):
    prepared = prepared_small_lattice_problem
    first_a = _prepared_run(prepared)
    second_a = _prepared_run(prepared)
    _assert_reconstruction_trajectory_equal(first_a, second_a)

    _prepared_run(
        prepared,
        initial_vacancy_fractions=jnp.array([0.25, 0.2]),
        seed=19,
    )
    third_a = _prepared_run(prepared)
    _assert_reconstruction_trajectory_equal(first_a, third_a)


def test_prepared_checkpoint_collection_does_not_change_optimization(
    prepared_small_lattice_problem,
):
    prepared = prepared_small_lattice_problem
    without_checkpoints = _prepared_run(prepared, checkpoint_interval=None)
    with_checkpoints = _prepared_run(prepared, checkpoint_interval=1)
    _assert_reconstruction_trajectory_equal(without_checkpoints, with_checkpoints)
    assert without_checkpoints.checkpoint_updates.size == 0
    np.testing.assert_array_equal(with_checkpoints.checkpoint_updates, [0, 1, 2, 3])


def test_training_diagnostic_subset_preserves_validation_selected_trajectory(
    prepared_small_lattice_problem,
):
    prepared = prepared_small_lattice_problem
    full = _prepared_run(prepared, training_diagnostic_scan_count=None)
    subset = _prepared_run(prepared, training_diagnostic_scan_count=1)

    for name in (
        "potential",
        "vacancy_fractions",
        "displacement_controls",
        "rigid_displacement",
        "predicted_intensities",
        "validation_loss_history",
        "vacancy_fraction_history",
        "displacement_control_history",
        "rigid_displacement_history",
    ):
        np.testing.assert_array_equal(getattr(subset, name), getattr(full, name))
    assert subset.best_update == full.best_update
    assert subset.completed_updates == full.completed_updates
    assert subset.stop_reason == full.stop_reason
    assert not np.array_equal(
        subset.training_loss_history,
        full.training_loss_history,
    )
    assert subset.metadata["training_loss_history_scope"] == (
        "fixed_geometry_stratified_training_subset"
    )
    assert len(subset.metadata["training_diagnostic_indices"]) == 1
    assert subset.metadata["final_full_training_loss"] == pytest.approx(
        full.metadata["final_full_training_loss"], rel=0.0, abs=0.0
    )
    timings = subset.metadata["optimization_phase_timings_s"]
    assert all(np.isfinite(value) and value >= 0.0 for value in timings.values())
    assert (
        subset.metadata["optimization_phase_classified_time_s"]
        + subset.metadata["optimization_phase_unclassified_time_s"]
    ) == pytest.approx(subset.metadata["optimization_time_s"], rel=1e-12)
    assert subset.metadata["training_diagnostic_scan_evaluations"] == len(
        subset.update_history
    )


def test_training_diagnostic_selection_is_geometry_only_and_falls_back_without_validation():
    training = np.asarray([4, 0, 2, 1], dtype=np.int64)
    coordinates = np.asarray([0.0, 1.0, 2.0, 30.0, 4.0, 50.0])
    selected, metadata = (
        ptychography_1d_module._geometry_stratified_training_diagnostic_indices_1d(
            training,
            coordinates,
            2,
            validation_available=True,
        )
    )
    np.testing.assert_array_equal(selected, [1, 4])
    changed_nontraining_coordinates = coordinates.copy()
    changed_nontraining_coordinates[[3, 5]] = [-1e6, 1e6]
    repeated, repeated_metadata = (
        ptychography_1d_module._geometry_stratified_training_diagnostic_indices_1d(
            training,
            changed_nontraining_coordinates,
            2,
            validation_available=True,
        )
    )
    np.testing.assert_array_equal(repeated, selected)
    assert repeated_metadata["selection_sha256"] == metadata["selection_sha256"]

    fallback, fallback_metadata = (
        ptychography_1d_module._geometry_stratified_training_diagnostic_indices_1d(
            training,
            coordinates,
            1,
            validation_available=False,
        )
    )
    np.testing.assert_array_equal(fallback, training)
    assert fallback_metadata["uses_full_training_partition"] is True
    assert fallback_metadata["fallback_reason"] == (
        "full_training_is_authoritative_without_validation"
    )


def test_prepared_fixed_evaluation_batch_pads_and_crops_short_tail(
    prepared_small_lattice_problem,
):
    prepared = prepared_small_lattice_problem
    assert prepared.measured_intensities.shape[0] % prepared.evaluation_batch_size != 0
    result = _prepared_run(prepared)
    direct = simulate_glancing_scan_1d(
        result.potential,
        prepared.probe_rows,
        prepared.window_starts,
        prepared.window_length,
        prepared.propagation_kernel,
        prepared.slice_thickness,
        prepared.energy,
        rematerialize=prepared.rematerialize,
    )
    assert result.predicted_intensities.shape == prepared.measured_intensities.shape
    np.testing.assert_allclose(
        result.predicted_intensities,
        direct,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.all(np.isfinite(result.validation_loss_history))
    assert np.isfinite(result.audit_loss)


def test_prepared_bad_initialization_fails_before_executable_call(
    prepared_small_lattice_problem,
):
    with pytest.raises(ValueError, match="must have shape"):
        run_prepared_lattice_site_reconstruction_1d(
            prepared_small_lattice_problem,
            initial_vacancy_fractions=jnp.zeros(3),
            updates=1,
        )


def test_prepared_static_contract_rejects_replaced_data_and_scalars(
    prepared_small_lattice_problem,
):
    prepared = prepared_small_lattice_problem
    changed_measurements = prepared.measured_intensities.at[0, 0].add(1.0)
    for changed in (
        replace(prepared, measured_intensities=changed_measurements),
        replace(prepared, epsilon=2.0 * prepared.epsilon),
        replace(prepared, _static_contract=None),
    ):
        with pytest.raises(ValueError, match="static contract"):
            run_prepared_lattice_site_reconstruction_1d(changed, updates=1)


def test_repreparing_changed_data_changes_problem_id(
    prepared_small_lattice_problem,
):
    prepared = prepared_small_lattice_problem
    changed_measurements = prepared.measured_intensities.at[-1, 0].add(1.0)
    changed = prepare_lattice_site_reconstruction_1d(
        prepared.model,
        prepared.input_probe,
        prepared.window_starts,
        prepared.window_length,
        prepared.propagation_kernel,
        prepared.slice_thickness,
        prepared.energy,
        changed_measurements,
        scan_coordinates=prepared.scan_coordinates,
        detector_angles=prepared.detector_angles,
        validation_indices=np.asarray(prepared.validation_indices),
        audit_indices=np.asarray(prepared.audit_indices),
        excluded_indices=np.asarray(prepared.excluded_indices),
        potential_max=prepared.potential_max,
        minibatch_size=prepared.minibatch_size,
        evaluation_batch_size=prepared.evaluation_batch_size,
        gradient_clip=prepared.gradient_clip,
        epsilon=prepared.epsilon,
        rematerialize=prepared.rematerialize,
    )
    assert changed.reconstruction_problem_id != prepared.reconstruction_problem_id


def test_prepared_detector_mask_is_static_and_changes_problem_identity(
    prepared_small_lattice_problem,
):
    prepared = prepared_small_lattice_problem
    valid = jnp.ones(prepared.measured_intensities.shape, dtype=bool)
    valid = valid.at[0, 0].set(False)
    with pytest.raises(ValueError, match="static contract"):
        run_prepared_lattice_site_reconstruction_1d(
            replace(prepared, detector_valid_mask=valid), updates=1
        )

    changed = prepare_lattice_site_reconstruction_1d(
        prepared.model,
        prepared.input_probe,
        prepared.window_starts,
        prepared.window_length,
        prepared.propagation_kernel,
        prepared.slice_thickness,
        prepared.energy,
        prepared.measured_intensities,
        detector_valid_mask=valid,
        scan_coordinates=prepared.scan_coordinates,
        detector_angles=prepared.detector_angles,
        validation_indices=np.asarray(prepared.validation_indices),
        audit_indices=np.asarray(prepared.audit_indices),
        excluded_indices=np.asarray(prepared.excluded_indices),
        potential_max=prepared.potential_max,
        minibatch_size=prepared.minibatch_size,
        evaluation_batch_size=prepared.evaluation_batch_size,
        gradient_clip=prepared.gradient_clip,
        epsilon=prepared.epsilon,
        rematerialize=prepared.rematerialize,
    )
    assert changed.reconstruction_problem_id != prepared.reconstruction_problem_id
    assert changed.objective_id == changed.metadata["objective_id"]
    assert changed.metadata["measurement_contract"] == (
        "masked_nonnegative_intensity"
    )
    assert changed.metadata["poisson_count_likelihood_supported"] is False
    assert changed.metadata["read_noise_likelihood_supported"] is False
    np.testing.assert_array_equal(changed.detector_valid_mask, valid)
    result = _prepared_run(changed, updates=1, validation_interval=1)
    np.testing.assert_array_equal(result.detector_valid_mask, valid)
    assert result.metadata["objective_id"] == changed.objective_id
    assert result.metadata["measurement_contract"] == (
        "masked_nonnegative_intensity"
    )


def test_detector_mask_partition_with_no_valid_scan_fails_closed(
    prepared_small_lattice_problem,
):
    prepared = prepared_small_lattice_problem
    valid = jnp.ones(prepared.measured_intensities.shape, dtype=bool)
    valid = valid.at[int(prepared.validation_indices[0]), :].set(False)
    with pytest.raises(ValueError, match="validation scan"):
        prepare_lattice_site_reconstruction_1d(
            prepared.model,
            prepared.input_probe,
            prepared.window_starts,
            prepared.window_length,
            prepared.propagation_kernel,
            prepared.slice_thickness,
            prepared.energy,
            prepared.measured_intensities,
            detector_valid_mask=valid,
            validation_indices=np.asarray(prepared.validation_indices),
            audit_indices=np.asarray(prepared.audit_indices),
            excluded_indices=np.asarray(prepared.excluded_indices),
            potential_max=prepared.potential_max,
            minibatch_size=prepared.minibatch_size,
            evaluation_batch_size=prepared.evaluation_batch_size,
            rematerialize=prepared.rematerialize,
        )


def test_prepared_trajectory_ignores_only_masked_measurements(
    prepared_small_lattice_problem,
):
    original = prepared_small_lattice_problem
    valid = jnp.ones(original.measured_intensities.shape, dtype=bool)
    valid = valid.at[0, 0].set(False)

    def prepare(measured):
        return prepare_lattice_site_reconstruction_1d(
            original.model,
            original.input_probe,
            original.window_starts,
            original.window_length,
            original.propagation_kernel,
            original.slice_thickness,
            original.energy,
            measured,
            detector_valid_mask=valid,
            scan_coordinates=original.scan_coordinates,
            detector_angles=original.detector_angles,
            validation_indices=np.asarray(original.validation_indices),
            audit_indices=np.asarray(original.audit_indices),
            excluded_indices=np.asarray(original.excluded_indices),
            potential_max=original.potential_max,
            minibatch_size=original.minibatch_size,
            evaluation_batch_size=original.evaluation_batch_size,
            gradient_clip=original.gradient_clip,
            epsilon=original.epsilon,
            rematerialize=original.rematerialize,
        )

    baseline = _prepared_run(prepare(original.measured_intensities))
    masked_changed = _prepared_run(
        prepare(original.measured_intensities.at[0, 0].set(jnp.nan))
    )
    _assert_reconstruction_trajectory_equal(baseline, masked_changed)

    valid_changed = _prepared_run(
        prepare(original.measured_intensities.at[1, 1].add(100.0))
    )
    assert not np.array_equal(
        valid_changed.training_loss_history, baseline.training_loss_history
    )


def test_calibrated_prepared_static_contract_and_problem_hash_bind_inputs(
    prepared_small_poisson_problem,
    prepared_small_poisson_heldout_changed,
):
    prepared = prepared_small_poisson_problem
    assert prepared.measurement is not None and prepared.objective is not None
    for changed in (
        replace(
            prepared,
            measurement=replace(
                prepared.measurement,
                calibration_id="replacement_without_reprepare",
            ),
        ),
        replace(
            prepared,
            objective=replace(prepared.objective, relative_signal_scale=0.9),
        ),
    ):
        with pytest.raises(ValueError, match="static contract"):
            run_prepared_lattice_site_reconstruction_1d(changed, updates=1)
    assert (
        prepared_small_poisson_heldout_changed.reconstruction_problem_id
        != prepared.reconstruction_problem_id
    )
    changed_objective = replace(
        prepared.objective,
        electrons_per_pattern=(
            1.1 * jnp.asarray(prepared.objective.electrons_per_pattern)
        ),
    )
    reprepared = _reprepare_calibrated(
        prepared, prepared.measurement, changed_objective
    )
    assert reprepared.reconstruction_problem_id != prepared.reconstruction_problem_id


def test_calibrated_prepared_training_ignores_heldout_measurement_values(
    prepared_small_poisson_problem,
    prepared_small_poisson_heldout_changed,
):
    options = {
        "updates": 3,
        "validation_interval": 1,
        "checkpoint_interval": 1,
        "seed": 13,
    }
    baseline = _prepared_run(prepared_small_poisson_problem, **options)
    changed = _prepared_run(prepared_small_poisson_heldout_changed, **options)
    np.testing.assert_array_equal(
        baseline.vacancy_fraction_history, changed.vacancy_fraction_history
    )
    np.testing.assert_array_equal(
        baseline.displacement_control_history,
        changed.displacement_control_history,
    )
    np.testing.assert_array_equal(
        baseline.rigid_displacement_history, changed.rigid_displacement_history
    )
    np.testing.assert_array_equal(
        baseline.training_loss_history, changed.training_loss_history
    )
    assert not np.array_equal(
        baseline.validation_loss_history, changed.validation_loss_history
    )


def test_tiny_prepared_poisson_vacancy_loss_reduces(
    prepared_small_poisson_problem,
    small_poisson_reconstruction,
):
    result = small_poisson_reconstruction
    assert float(result.training_loss_history[-1]) < float(
        result.training_loss_history[0]
    )
    assert result.predicted_signal_electrons is not None
    assert result.measurement is not None and result.objective is not None
    np.testing.assert_array_equal(
        result.detector_valid_mask, result.measurement.valid_mask
    )
    assert result.metadata["objective_kind"] == "poisson_deviance"
    assert result.metadata["relative_signal_scale_fitted"] is False
    assert result.metadata["structural_trust_from_measurement_objective"] is False


def test_calibrated_and_legacy_measurements_are_mutually_exclusive(
    prepared_small_poisson_problem,
):
    prepared = prepared_small_poisson_problem
    assert prepared.measurement is not None and prepared.objective is not None
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_lattice_site_reconstruction_1d(
            prepared.model,
            prepared.input_probe,
            prepared.window_starts,
            prepared.window_length,
            prepared.propagation_kernel,
            prepared.slice_thickness,
            prepared.energy,
            prepared.measured_intensities,
            measurement=prepared.measurement,
            objective=prepared.objective,
            potential_max=prepared.potential_max,
            rematerialize=False,
        )


def test_calibrated_prepare_rejects_zero_norm_probe_row(
    prepared_small_poisson_problem,
):
    prepared = prepared_small_poisson_problem
    assert prepared.measurement is not None and prepared.objective is not None
    probes = prepared.probe_rows.at[0].set(0.0)
    with pytest.raises(ValueError, match="positive norm"):
        prepare_lattice_site_reconstruction_1d(
            prepared.model,
            probes,
            prepared.window_starts,
            prepared.window_length,
            prepared.propagation_kernel,
            prepared.slice_thickness,
            prepared.energy,
            measurement=prepared.measurement,
            objective=prepared.objective,
            validation_indices=np.asarray(prepared.validation_indices),
            audit_indices=np.asarray(prepared.audit_indices),
            excluded_indices=np.asarray(prepared.excluded_indices),
            potential_max=prepared.potential_max,
            rematerialize=False,
        )


def test_scan_validates_starts_and_returns_full_detector():
    potential = _potential(7)
    intensities = simulate_glancing_scan_1d(
        potential,
        _probe(),
        jnp.array([0, 2, 4], dtype=jnp.int32),
        3,
        _kernel(),
        DS,
        ENERGY,
    )
    assert intensities.shape == (3, N_U)
    assert np.all(np.asarray(intensities) >= 0.0)

    with pytest.raises(ValueError):
        simulate_glancing_scan_1d(
            potential, _probe(), jnp.array([-1]), 3, _kernel(), DS, ENERGY
        )
    with pytest.raises(ValueError):
        simulate_glancing_scan_1d(
            potential, _probe(), jnp.array([5]), 3, _kernel(), DS, ENERGY
        )
    with pytest.raises((TypeError, ValueError)):
        simulate_glancing_scan_1d(
            potential, _probe(), jnp.array([0.0]), 3, _kernel(), DS, ENERGY
        )


def test_masked_amplitude_loss_matches_manual_value_and_gradient():
    epsilon = 1e-10
    predicted = jnp.asarray([1.0, -7.0, 9.0, 1e200], dtype=jnp.float64)
    measured = jnp.asarray([4.0, np.nan, 16.0, -3.0], dtype=jnp.float64)
    valid = jnp.asarray([True, False, True, False])

    loss_function = lambda values: normalized_amplitude_loss_1d(
        values,
        measured,
        epsilon=epsilon,
        detector_valid_mask=valid,
    )
    actual_loss = loss_function(predicted)
    actual_gradient = jax.grad(loss_function)(predicted)

    selected_prediction = np.asarray(predicted)[[0, 2]]
    selected_measurement = np.asarray(measured)[[0, 2]]
    denominator = np.sum(selected_measurement)
    difference = np.sqrt(selected_prediction + epsilon) - np.sqrt(
        selected_measurement + epsilon
    )
    expected_loss = np.sum(difference**2) / denominator
    expected_gradient = np.zeros(4)
    expected_gradient[[0, 2]] = difference / (
        np.sqrt(selected_prediction + epsilon) * denominator
    )
    assert float(actual_loss) == pytest.approx(expected_loss, rel=1e-13)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-13)


def test_masked_amplitude_loss_ignores_masked_values_but_uses_valid_values():
    predicted = jnp.asarray([1.0, 4.0, 9.0, 16.0])
    measured = jnp.asarray([1.5, 3.0, 8.0, 15.0])
    valid = jnp.asarray([True, False, True, False])
    baseline = normalized_amplitude_loss_1d(
        predicted, measured, detector_valid_mask=valid
    )
    masked_changed = normalized_amplitude_loss_1d(
        predicted.at[1].set(1e200).at[3].set(-1.0),
        measured.at[1].set(np.nan).at[3].set(1e200),
        detector_valid_mask=valid,
    )
    valid_changed = normalized_amplitude_loss_1d(
        predicted.at[0].add(0.5), measured, detector_valid_mask=valid
    )
    np.testing.assert_array_equal(masked_changed, baseline)
    assert not np.isclose(float(valid_changed), float(baseline))

    with pytest.raises(ValueError, match="at least one observation"):
        normalized_amplitude_loss_1d(
            predicted, measured, detector_valid_mask=jnp.zeros(4, dtype=bool)
        )


def test_poisson_deviance_matches_manual_zero_count_and_finite_difference():
    predicted = jnp.asarray([[0.2, 1.1, 4.0]], dtype=jnp.float64)
    probes = jnp.ones((1, 3), dtype=jnp.complex128)
    valid = jnp.asarray([[True, True, False]])
    measurement = PtychographyMeasurement1D(
        calibrated_signal_electrons=jnp.asarray([[-0.3, 1.7, jnp.nan]]),
        observed_total_electrons=jnp.asarray([[0.0, 2.0, -1e200]]),
        valid_mask=valid,
        calibrated_dark_electrons_per_pixel=0.3,
        calibrated_read_noise_std_electrons=0.0,
        calibration_id="manual_poisson",
    )
    objective = PtychographyObjective1D(
        kind="poisson_deviance",
        electrons_per_pattern=9.0,
        minimum_expected_electrons=1e-6,
    )
    actual = ptychography_objective_loss_1d(
        predicted, probes, measurement, objective
    )
    mean = np.asarray([0.5, 1.4])
    observed = np.asarray([0.0, 2.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(
            observed > 0.0, observed * np.log(observed / mean), 0.0
        )
    expected = np.mean(2.0 * (mean - observed + log_term))
    assert float(actual) == pytest.approx(expected, rel=1e-13)

    objective_at = lambda value: ptychography_objective_loss_1d(
        predicted.at[0, 1].set(value), probes, measurement, objective
    )
    step = 1e-5
    finite_difference = (
        objective_at(predicted[0, 1] + step)
        - objective_at(predicted[0, 1] - step)
    ) / (2.0 * step)
    gradient = jax.grad(objective_at)(predicted[0, 1])
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-8)


def test_poisson_gaussian_nll_accepts_negative_signal_and_matches_manual_gradient():
    predicted = jnp.asarray([[0.4, 1.2, 3.0]], dtype=jnp.float64)
    probes = jnp.ones((1, 3), dtype=jnp.complex128)
    valid = jnp.asarray([[True, True, False]])
    measurement = PtychographyMeasurement1D(
        calibrated_signal_electrons=jnp.asarray([[-1.5, 2.3, jnp.nan]]),
        observed_total_electrons=jnp.asarray([[0.5, 4.3, -1e200]]),
        valid_mask=valid,
        calibrated_dark_electrons_per_pixel=2.0,
        calibrated_read_noise_std_electrons=1.5,
        calibration_id="manual_read_noise",
    )
    objective = PtychographyObjective1D(
        kind="poisson_gaussian_nll",
        electrons_per_pattern=9.0,
        minimum_expected_electrons=0.25,
    )
    actual = ptychography_objective_loss_1d(
        predicted, probes, measurement, objective
    )
    signal = np.asarray([0.4, 1.2])
    observed = np.asarray([-1.5, 2.3])
    variance = signal + 2.0 + 1.5**2
    expected = np.mean(
        0.5
        * (
            (observed - signal) ** 2 / variance
            + np.log(variance / 0.25)
        )
    )
    assert float(actual) == pytest.approx(expected, rel=1e-13)

    objective_at = lambda value: ptychography_objective_loss_1d(
        predicted.at[0, 0].set(value), probes, measurement, objective
    )
    step = 1e-5
    finite_difference = (
        objective_at(predicted[0, 0] + step)
        - objective_at(predicted[0, 0] - step)
    ) / (2.0 * step)
    gradient = jax.grad(objective_at)(predicted[0, 0])
    np.testing.assert_allclose(gradient, finite_difference, rtol=2e-8)

    sentinel_changed = PtychographyMeasurement1D(
        calibrated_signal_electrons=jnp.asarray([[-1.5, 2.3, 1e200]]),
        observed_total_electrons=jnp.asarray([[0.5, 4.3, jnp.nan]]),
        valid_mask=valid,
        calibrated_dark_electrons_per_pixel=jnp.asarray(
            [[2.0, 2.0, jnp.nan]]
        ),
        calibrated_read_noise_std_electrons=jnp.asarray(
            [[1.5, 1.5, -1e200]]
        ),
        calibration_id="manual_read_noise",
    )
    np.testing.assert_array_equal(
        ptychography_objective_loss_1d(
            predicted, probes, sentinel_changed, objective
        ),
        actual,
    )


def test_fft_count_conversion_conserves_declared_electrons_per_pattern():
    probes = jnp.asarray(
        [
            [1.0, 0.5j, -0.2, 0.1j],
            [0.3, -0.7j, 1.2, 0.4j],
        ],
        dtype=jnp.complex128,
    )
    intensities = jnp.abs(jnp.fft.fftshift(jnp.fft.fft(probes), axes=-1)) ** 2
    objective = PtychographyObjective1D(
        kind="poisson_deviance",
        electrons_per_pattern=jnp.asarray([120.0, 350.0]),
    )
    signal = ptychography_expected_signal_electrons_1d(
        intensities, probes, objective
    )
    np.testing.assert_allclose(
        np.sum(signal, axis=1), [120.0, 350.0], rtol=2e-14
    )


def test_poisson_deviance_rejects_declared_read_noise():
    measurement = PtychographyMeasurement1D(
        calibrated_signal_electrons=jnp.ones((1, 2)),
        observed_total_electrons=jnp.ones((1, 2)),
        valid_mask=jnp.ones((1, 2), dtype=bool),
        calibrated_dark_electrons_per_pixel=0.0,
        calibrated_read_noise_std_electrons=0.1,
        calibration_id="not_ideal_poisson",
    )
    objective = PtychographyObjective1D("poisson_deviance", 10.0)
    with pytest.raises(ValueError, match="zero read noise"):
        ptychography_objective_loss_1d(
            jnp.ones((1, 2)), jnp.ones((1, 2)), measurement, objective
        )
    negative_total = replace(
        measurement,
        calibrated_signal_electrons=jnp.asarray([[-0.1, 1.0]]),
        observed_total_electrons=jnp.asarray([[-0.1, 1.0]]),
        calibrated_read_noise_std_electrons=0.0,
    )
    with pytest.raises(ValueError, match="non-negative observed total"):
        ptychography_objective_loss_1d(
            jnp.ones((1, 2)), jnp.ones((1, 2)), negative_total, objective
        )


def test_calibrated_measurement_rejects_inconsistent_dark_subtraction():
    inconsistent = PtychographyMeasurement1D(
        calibrated_signal_electrons=jnp.asarray([[1.0, jnp.nan]]),
        observed_total_electrons=jnp.asarray([[1.25, -1e200]]),
        valid_mask=jnp.asarray([[True, False]]),
        calibrated_dark_electrons_per_pixel=0.1,
        calibrated_read_noise_std_electrons=0.0,
        calibration_id="inconsistent-dark-subtraction",
    )
    with pytest.raises(ValueError, match="minus calibrated dark"):
        validate_ptychography_measurement_1d(inconsistent)


def test_batched_scan_matches_serial_fresnel_baseline():
    potential = _potential(8)
    starts = np.array([0, 2, 5])
    window_length = 3
    batched = simulate_glancing_scan_1d(
        potential,
        _probe(),
        starts,
        window_length,
        _kernel(),
        DS,
        ENERGY,
        rematerialize=True,
    )
    serial = []
    for start in starts:
        _, intensity, _ = simulate_glancing_fresnel_baseline_1d(
            _probe(),
            potential[start : start + window_length],
            DU,
            DS,
            ENERGY,
        )
        serial.append(intensity)
    np.testing.assert_allclose(
        np.asarray(batched), np.asarray(jnp.stack(serial)), rtol=2e-12, atol=2e-12
    )


def test_fftshift_never_reorders_scans():
    potential = _potential(4)
    starts = np.array([0, 3, 1])
    unit_kernel = jnp.ones(N_U, dtype=jnp.complex128)
    actual = simulate_glancing_scan_1d(
        potential, _probe(), starts, 1, unit_kernel, DS, ENERGY
    )
    unshifted = []
    for start in starts:
        exit_wave = _probe() * phase_grating_1d_from_projected_potential(
            potential[start] * DS, ENERGY
        )
        unshifted.append(jnp.abs(jnp.fft.fft(exit_wave)) ** 2)
    expected = jnp.fft.fftshift(jnp.stack(unshifted), axes=-1)
    assert not np.allclose(np.asarray(expected[0]), np.asarray(expected[1]))
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-12)


def test_beam_path_region_tracks_scan_windows_and_overlap_counts():
    u = jnp.linspace(-1.0, 1.0, 9)
    mask, coverage = beam_path_reconstruction_region_1d(
        8,
        u,
        jnp.array([0, 2], dtype=jnp.int32),
        5,
        1.0,
        0.0,
        0.2,
        -1.0,
        radius_waists=1.0,
    )

    assert mask.shape == coverage.shape == (8, 9)
    center = int(np.argmin(np.abs(np.asarray(u))))
    np.testing.assert_array_equal(
        np.asarray(coverage[:, center]), np.array([1, 1, 2, 2, 2, 1, 1, 0])
    )
    assert np.all(np.asarray(mask[:, np.asarray(u) > 0.0]) == 0)
    assert np.all(np.asarray(mask[:7, center]))
    assert not bool(np.asarray(mask[7, center]))


def test_beam_path_region_follows_tilted_centreline():
    u = jnp.arange(-2.0, 2.01, 0.25)
    mask, coverage = beam_path_reconstruction_region_1d(
        5,
        u,
        jnp.array([0], dtype=jnp.int32),
        5,
        1.0,
        -np.arctan(0.5),
        0.01,
        -2.0,
        radius_waists=1.0,
    )

    # The midpoint convention places the centreline at u=-0.25 for s=3.
    u_index = int(np.flatnonzero(np.isclose(np.asarray(u), -0.25))[0])
    assert int(np.asarray(coverage[3, u_index])) == 1
    assert bool(np.asarray(mask[3, u_index]))


def test_masked_potential_has_zero_gradient_outside_mask():
    n_s, n_u = 5, 12
    mask = jnp.zeros((n_s, n_u), dtype=bool).at[1:4, 4:8].set(True)
    candidate = jnp.linspace(10.0, 100.0, n_s * n_u).reshape(n_s, n_u)
    probe = _probe(n_u=n_u, du=0.3)
    kernel = _kernel(n_u=n_u, du=0.3, ds=0.4)

    def objective(full_candidate):
        potential = jnp.where(mask, full_candidate, 0.0)
        intensity = simulate_glancing_scan_1d(
            potential, probe, jnp.array([0, 1]), 4, kernel, 0.4, ENERGY
        )
        return jnp.sum(jnp.sqrt(intensity + 1e-12))

    gradient = jax.grad(objective)(candidate)
    assert np.all(np.isfinite(np.asarray(gradient[mask])))
    np.testing.assert_array_equal(np.asarray(gradient[~mask]), 0.0)


def test_lattice_renderer_identity_and_unit_vacancy():
    model = _small_lattice_model()
    vacancies = jnp.zeros(2)
    controls = jnp.zeros((2, 2, 2))
    identity = render_lattice_site_potential_1d(model, vacancies, controls)
    np.testing.assert_allclose(identity, model.reference_potential, atol=1e-14)

    one_vacancy = render_lattice_site_potential_1d(
        model, vacancies.at[0].set(1.0), controls
    )
    expected = np.asarray(model.reference_potential).copy()
    start = np.asarray(model.patch_starts[0])
    patch = np.asarray(model.site_patches[0])
    expected[
        start[0] : start[0] + patch.shape[0],
        start[1] : start[1] + patch.shape[1],
    ] -= patch
    np.testing.assert_allclose(one_vacancy, expected, atol=1e-14)

    controls = controls.at[0, :, 0].set(0.08)
    controls = controls.at[1, :, 0].set(-0.04)
    site_displacements = lattice_site_displacements_1d(
        model.site_coordinates,
        controls,
        model.control_coordinates_s,
        model.control_coordinates_u,
    )
    controlled = render_lattice_site_potential_1d(model, vacancies, controls)
    independent = render_lattice_site_potential_from_displacements_1d(
        model, vacancies, site_displacements
    )
    np.testing.assert_allclose(independent, controlled, atol=1e-12)


def test_lattice_renderer_positive_displacement_moves_patch_to_positive_axis():
    model = _small_lattice_model()
    vacancies = jnp.zeros(2)
    displacements = jnp.zeros((2, 2)).at[0, 0].set(model.axial_sampling)
    rendered = render_lattice_site_potential_from_displacements_1d(
        model, vacancies, displacements
    )

    expected = np.asarray(model.reference_potential).copy()
    start = np.asarray(model.patch_starts[0])
    patch = np.asarray(model.site_patches[0])
    shifted = np.zeros_like(patch)
    shifted[1:] = patch[:-1]
    region = (
        slice(start[0], start[0] + patch.shape[0]),
        slice(start[1], start[1] + patch.shape[1]),
    )
    expected[region] += shifted - patch
    np.testing.assert_allclose(rendered, expected, rtol=0.0, atol=2e-14)


def test_lattice_renderer_is_smooth_at_zero_displacement():
    model = _small_lattice_model()
    vacancies = jnp.array([0.13, 0.07])
    base_displacements = jnp.zeros((2, 2))
    grid = jnp.arange(np.prod(model.reference_potential.shape), dtype=jnp.float64)
    weights = (jnp.sin(0.37 * grid) + 0.013 * grid).reshape(
        model.reference_potential.shape
    )

    def objective(first_site_displacement):
        displacements = base_displacements.at[0].set(first_site_displacement)
        potential = render_lattice_site_potential_from_displacements_1d(
            model, vacancies, displacements
        )
        return jnp.sum(weights * potential)

    zero = jnp.zeros(2)
    automatic = jax.grad(objective)(zero)
    # Keys convolution is C1 but not C2 at integer shifts, so the centred
    # difference converges linearly at this knot.
    step = 1e-7
    finite_difference = jnp.stack(
        [
            (
                objective(zero.at[component].set(step))
                - objective(zero.at[component].set(-step))
            )
            / (2.0 * step)
            for component in range(2)
        ]
    )
    np.testing.assert_allclose(
        automatic, finite_difference, rtol=1e-6, atol=2e-7
    )

    continuity_step = 1e-7
    gradient_before = jax.grad(objective)(jnp.array([-continuity_step, 0.0]))
    gradient_after = jax.grad(objective)(jnp.array([continuity_step, 0.0]))
    np.testing.assert_allclose(
        gradient_before, gradient_after, rtol=2e-5, atol=2e-5
    )


def test_rigid_residual_decomposition_preserves_motion_and_removes_gauge():
    model = _small_lattice_model()
    controls = jnp.array(
        [
            [[0.10, -0.04], [0.16, 0.03]],
            [[-0.02, 0.08], [0.05, -0.01]],
        ]
    )
    initial_rigid = jnp.array([0.03, -0.02])
    rigid, residual = decompose_lattice_site_displacement_controls_1d(
        model.site_coordinates,
        controls,
        model.control_coordinates_s,
        model.control_coordinates_u,
        rigid_displacement=initial_rigid,
    )
    original_sites = initial_rigid + lattice_site_displacements_1d(
        model.site_coordinates,
        controls,
        model.control_coordinates_s,
        model.control_coordinates_u,
    )
    residual_sites = lattice_site_displacements_1d(
        model.site_coordinates,
        residual,
        model.control_coordinates_s,
        model.control_coordinates_u,
    )
    np.testing.assert_allclose(rigid + residual_sites, original_sites, atol=1e-12)
    np.testing.assert_allclose(np.mean(residual_sites, axis=0), 0.0, atol=1e-12)
    vacancies = jnp.array([0.2, 0.0])
    legacy = render_lattice_site_potential_1d(
        model, vacancies, controls + initial_rigid
    )
    decomposed = render_lattice_site_potential_1d(
        model, vacancies, residual + rigid
    )
    np.testing.assert_allclose(decomposed, legacy, atol=1e-12)


def test_similarity_control_projection_removes_alignment_modes_and_preserves_shear():
    controls_s = jnp.asarray([-1.0, 0.0, 1.0])
    controls_u = jnp.asarray([-1.0, 0.0, 1.0])
    grid_s, grid_u = jnp.meshgrid(controls_s, controls_u, indexing="ij")
    sites = jnp.stack([grid_s.ravel(), grid_u.ravel()], axis=1)
    rotation = jnp.stack([-grid_u, grid_s], axis=-1)
    dilation = jnp.stack([grid_s, grid_u], axis=-1)
    translation = jnp.broadcast_to(jnp.asarray([0.2, -0.1]), rotation.shape)
    controls = translation + 0.07 * rotation + 0.04 * dilation

    similarity, residual = decompose_lattice_site_similarity_controls_1d(
        sites,
        controls,
        controls_s,
        controls_u,
    )
    np.testing.assert_allclose(similarity, controls, atol=2e-12)
    np.testing.assert_allclose(residual, 0.0, atol=2e-12)

    shear = jnp.stack([0.05 * grid_s, -0.05 * grid_u], axis=-1)
    shear_similarity, shear_residual = (
        decompose_lattice_site_similarity_controls_1d(
            sites,
            shear,
            controls_s,
            controls_u,
        )
    )
    np.testing.assert_allclose(shear_similarity, 0.0, atol=2e-12)
    np.testing.assert_allclose(shear_residual, shear, atol=2e-12)

    local = jnp.zeros_like(shear).at[1, 1, 0].set(0.2)
    _, local_residual = decompose_lattice_site_similarity_controls_1d(
        sites,
        local,
        controls_s,
        controls_u,
    )
    repeated_similarity, repeated_residual = (
        decompose_lattice_site_similarity_controls_1d(
            sites,
            local_residual,
            controls_s,
            controls_u,
        )
    )
    assert float(jnp.linalg.norm(local_residual)) > 0.0
    np.testing.assert_allclose(repeated_similarity, 0.0, atol=2e-12)
    np.testing.assert_allclose(repeated_residual, local_residual, atol=2e-12)

    gradient = jax.grad(
        lambda values: jnp.sum(
            decompose_lattice_site_similarity_controls_1d(
                sites,
                values,
                controls_s,
                controls_u,
            )[1]
            ** 2
        )
    )(local)
    assert np.all(np.isfinite(gradient))


def test_prepared_similarity_gauge_cannot_refit_global_alignment():
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    model = _small_lattice_model()
    n_u = model.reference_potential.shape[1]
    probe = _probe(n_u=n_u, du=0.3)
    starts = jnp.arange(3)
    kernel = _kernel(n_u=n_u, du=0.3, ds=0.4)
    measured = simulate_glancing_scan_1d(
        model.reference_potential,
        probe,
        starts,
        4,
        kernel,
        0.4,
        ENERGY,
    )
    translated = jnp.broadcast_to(jnp.asarray([0.08, -0.04]), (2, 2, 2))
    result = reconstruct_lattice_site_potential_1d(
        model,
        probe,
        starts,
        4,
        kernel,
        0.4,
        ENERGY,
        measured,
        initial_displacement_controls=translated,
        similarity_residual_gauge=True,
        validation_indices=[2],
        potential_max=10.0,
        updates=1,
        minibatch_size=1,
        validation_interval=1,
        evaluation_batch_size=2,
        rematerialize=False,
    )
    assert result.metadata["displacement_gauge"] == (
        "translation_rotation_isotropic_dilation"
    )
    assert result.metadata["similarity_residual_gauge"] is True
    assert result.metadata["n_residual_control_dof"] == 4
    np.testing.assert_allclose(result.initial_displacement_controls, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.rigid_displacement, 0.0, atol=1e-12)

    with pytest.raises(ValueError, match="cannot be combined"):
        prepare_lattice_site_reconstruction_1d(
            model,
            probe,
            starts,
            4,
            kernel,
            0.4,
            ENERGY,
            measured,
            separate_rigid_registration=True,
            similarity_residual_gauge=True,
            potential_max=10.0,
            rematerialize=False,
        )


def test_lattice_renderer_vacancy_and_displacement_gradients_match_difference():
    model = _small_lattice_model()
    weights = jnp.linspace(0.2, 1.7, np.prod(model.reference_potential.shape)).reshape(
        model.reference_potential.shape
    )
    controls = jnp.zeros((2, 2, 2)).at[..., 0].set(0.07)
    vacancies = jnp.array([0.15, 0.05])

    def objective(vacancy_values, control_values):
        potential = render_lattice_site_potential_1d(
            model, vacancy_values, control_values
        )
        return jnp.sum(weights * potential)

    vacancy_gradient, control_gradient = jax.grad(objective, argnums=(0, 1))(
        vacancies, controls
    )
    step = 1e-5
    vacancy_difference = (
        objective(vacancies.at[0].add(step), controls)
        - objective(vacancies.at[0].add(-step), controls)
    ) / (2 * step)
    control_difference = (
        objective(vacancies, controls.at[0, 0, 0].add(step))
        - objective(vacancies, controls.at[0, 0, 0].add(-step))
    ) / (2 * step)
    np.testing.assert_allclose(
        vacancy_gradient[0], vacancy_difference, rtol=2e-7, atol=2e-9
    )
    np.testing.assert_allclose(
        control_gradient[0, 0, 0], control_difference, rtol=2e-5, atol=2e-8
    )


def test_direct_reconstruction_retains_fixed_exterior():
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    n_s, n_u = 5, 12
    mask = jnp.zeros((n_s, n_u), dtype=bool).at[1:4, 4:8].set(True)
    fixed = jnp.full((n_s, n_u), 17.0)
    initial = fixed.at[mask].set(30.0)
    starts = jnp.array([0, 1])
    probe = _probe(n_u=n_u, du=0.3)
    kernel = _kernel(n_u=n_u, du=0.3, ds=0.4)
    measured = simulate_glancing_scan_1d(initial, probe, starts, 4, kernel, 0.4, ENERGY)
    result = reconstruct_potential_1d(
        initial,
        mask,
        probe,
        starts,
        4,
        kernel,
        0.4,
        ENERGY,
        measured,
        fixed_potential=fixed,
        potential_scale=30.0,
        potential_max=50.0,
        audit_indices=[1],
        updates=1,
        minibatch_size=2,
        validation_interval=1,
        rematerialize=False,
    )
    np.testing.assert_array_equal(np.asarray(result.potential)[~np.asarray(mask)], 17.0)
    assert result.metadata["uses_fixed_potential"] is True
    assert result.metadata["training_indices"] == [0]
    assert result.metadata["audit_indices"] == [1]
    assert result.audit_loss == pytest.approx(0.0, abs=1e-12)


def test_pixel_reconstruction_trajectory_ignores_only_masked_measurements():
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    n_s, n_u = 5, 12
    specimen_mask = jnp.zeros((n_s, n_u), dtype=bool).at[1:4, 4:8].set(True)
    target = jnp.zeros((n_s, n_u)).at[specimen_mask].set(28.0)
    initial = jnp.zeros((n_s, n_u)).at[specimen_mask].set(18.0)
    starts = jnp.asarray([0, 1])
    probe = _probe(n_u=n_u, du=0.3)
    kernel = _kernel(n_u=n_u, du=0.3, ds=0.4)
    measured = simulate_glancing_scan_1d(
        target, probe, starts, 4, kernel, 0.4, ENERGY
    )
    detector_mask = jnp.ones(measured.shape, dtype=bool).at[0, 0].set(False)

    def run(values):
        return reconstruct_potential_1d(
            initial,
            specimen_mask,
            probe,
            starts,
            4,
            kernel,
            0.4,
            ENERGY,
            values,
            detector_valid_mask=detector_mask,
            validation_indices=[1],
            potential_scale=20.0,
            potential_max=50.0,
            learning_rate_start=0.02,
            learning_rate_end=0.01,
            updates=2,
            minibatch_size=1,
            validation_interval=1,
            evaluation_batch_size=2,
            rematerialize=False,
            seed=11,
        )

    baseline = run(measured)
    masked_changed = run(measured.at[0, 0].set(1e200))
    for name in (
        "potential",
        "predicted_intensities",
        "update_history",
        "training_loss_history",
        "validation_loss_history",
    ):
        np.testing.assert_array_equal(
            getattr(masked_changed, name), getattr(baseline, name)
        )
    assert masked_changed.best_update == baseline.best_update

    valid_changed = run(measured.at[0, 1].add(100.0))
    assert not np.array_equal(
        valid_changed.training_loss_history, baseline.training_loss_history
    )
    assert baseline.metadata["measurement_contract"] == (
        "masked_nonnegative_intensity"
    )
    assert baseline.metadata["poisson_count_likelihood_supported"] is False
    assert baseline.metadata["read_noise_likelihood_supported"] is False


def test_direct_potential_gradient_matches_finite_difference():
    n_s = 4
    u = (jnp.arange(N_U) - N_U // 2) * DU
    profile = jnp.exp(-0.5 * ((u - 0.1) / 0.8) ** 2)
    weights = jnp.linspace(0.2, 1.3, N_U) ** 2

    def objective(value):
        potential = jnp.zeros((n_s, N_U)).at[2].set(value * profile)
        intensity = simulate_glancing_scan_1d(
            potential,
            _probe(),
            jnp.array([0]),
            n_s,
            _kernel(),
            DS,
            ENERGY,
            rematerialize=True,
        )[0]
        return jnp.sum(weights * intensity) / jnp.sum(intensity)

    x0 = jnp.asarray(220.0)
    automatic = jax.grad(objective)(x0)
    step = 1e-2
    finite = (objective(x0 + step) - objective(x0 - step)) / (2 * step)
    assert np.isfinite(np.asarray(automatic))
    np.testing.assert_allclose(
        np.asarray(automatic), np.asarray(finite), rtol=2e-4, atol=2e-8
    )


def test_sideview_cache_matches_batch_detector_and_downsamples_intensity():
    potential = _potential(8)
    starts = jnp.array([0, 2, 4], dtype=jnp.int32)
    u = (jnp.arange(N_U) - N_U // 2) * DU
    cache = simulate_glancing_sideview_cache_1d(
        potential,
        _probe(),
        starts,
        4,
        _kernel(),
        DS,
        ENERGY,
        jnp.array([0, 2]),
        transverse_coordinates=u,
        axial_stride=2,
        transverse_stride=2,
    )
    expected = simulate_glancing_scan_1d(
        potential, _probe(), starts, 4, _kernel(), DS, ENERGY
    )
    np.testing.assert_allclose(
        np.asarray(cache.detector_intensities),
        np.asarray(expected[jnp.asarray([0, 2])]),
        rtol=2e-6,
        atol=2e-6,
    )
    assert cache.sideview_wavefields.shape == (2, 2, N_U // 2)
    assert cache.sideview_wavefields.dtype == jnp.complex64
    assert cache.sideview_intensities.dtype == jnp.float32
    full_mean_power = np.sum(np.asarray(cache.sideview_intensities[0])) * 4
    assert full_mean_power > 0.0


def test_scan_cache_and_potential_result_round_trip_without_pickle(tmp_path):
    detector_valid_mask = jnp.asarray(
        [
            [True, False, True, True],
            [True, True, False, True],
            [False, True, True, True],
        ]
    )
    scan = GlancingScan1D(
        intensities=jnp.arange(12, dtype=jnp.float64).reshape(3, 4),
        window_starts=jnp.array([0, 2, 4]),
        scan_coordinates=jnp.array([1.0, 2.0, 3.0]),
        detector_angles=jnp.linspace(-2.0, 2.0, 4),
        metadata={"energy_eV": 30_000.0},
        detector_valid_mask=detector_valid_mask,
    )
    scan_path = tmp_path / "scan.npz"
    save_glancing_scan_1d(scan_path, scan)
    with np.load(scan_path, allow_pickle=False) as raw:
        assert raw["metadata_json"].dtype.kind in {"U", "S"}
    loaded_scan = load_glancing_scan_1d(scan_path)
    np.testing.assert_allclose(loaded_scan.intensities, scan.intensities)
    np.testing.assert_array_equal(
        loaded_scan.detector_valid_mask, detector_valid_mask
    )
    with np.load(scan_path, allow_pickle=False) as archive:
        scan_payload = {name: np.asarray(archive[name]) for name in archive.files}
    for suffix, malformed_mask, message in (
        ("shape", np.ones((2, 2), dtype=bool), "intensity shape"),
        ("dtype", np.ones((3, 4), dtype=np.uint8), "Boolean dtype"),
    ):
        malformed_path = tmp_path / f"scan_bad_mask_{suffix}.npz"
        np.savez_compressed(
            malformed_path,
            **{**scan_payload, "detector_valid_mask": malformed_mask},
        )
        with pytest.raises((TypeError, ValueError), match=message):
            load_glancing_scan_1d(malformed_path)

    potential = _potential(6, n_u=8)
    starts = jnp.array([0, 2])
    probe = _probe(n_u=8, du=0.4)
    kernel = _kernel(n_u=8, du=0.4, ds=0.5)
    cache = simulate_glancing_sideview_cache_1d(
        potential,
        probe,
        starts,
        4,
        kernel,
        DS,
        ENERGY,
        jnp.array([0, 1]),
        axial_stride=2,
        transverse_stride=2,
        metadata={"model": "truth"},
    )
    cache_path = tmp_path / "sideviews.npz"
    save_glancing_sideview_cache_1d(cache_path, cache)
    loaded_cache = load_glancing_sideview_cache_1d(cache_path)
    np.testing.assert_allclose(
        loaded_cache.sideview_wavefields, cache.sideview_wavefields
    )
    assert loaded_cache.metadata == cache.metadata

    mask = jnp.zeros((3, 4), dtype=bool).at[:, 1:3].set(True)
    result = PotentialReconstruction1D(
        potential=jnp.arange(12, dtype=jnp.float64).reshape(3, 4),
        initial_potential=jnp.ones((3, 4)),
        reconstruction_mask=mask,
        axial_coordinates=jnp.arange(3, dtype=jnp.float64),
        transverse_coordinates=jnp.arange(4, dtype=jnp.float64),
        predicted_intensities=scan.intensities,
        measured_intensities=scan.intensities,
        window_starts=scan.window_starts,
        scan_coordinates=scan.scan_coordinates,
        detector_angles=scan.detector_angles,
        update_history=jnp.array([0, 10]),
        training_loss_history=jnp.array([1.0, 0.1]),
        validation_loss_history=jnp.array([1.1, 0.2]),
        best_update=10,
        audit_loss=0.4,
        metadata={"n_unknown_pixels": 6},
        detector_valid_mask=detector_valid_mask,
    )
    result_path = tmp_path / "result.npz"
    save_potential_reconstruction_1d(result_path, result)
    loaded_result = load_potential_reconstruction_1d(result_path)
    np.testing.assert_allclose(loaded_result.potential, result.potential)
    np.testing.assert_array_equal(
        loaded_result.reconstruction_mask, result.reconstruction_mask
    )
    assert loaded_result.best_update == result.best_update
    assert loaded_result.audit_loss == pytest.approx(0.4)
    np.testing.assert_array_equal(
        loaded_result.detector_valid_mask, detector_valid_mask
    )

    lattice_model = _small_lattice_model()
    lattice_result = LatticeSiteReconstruction1D(
        potential=lattice_model.reference_potential,
        initial_potential=lattice_model.reference_potential,
        vacancy_fractions=jnp.array([0.9, 0.1]),
        initial_vacancy_fractions=jnp.zeros(2),
        displacement_controls=jnp.zeros((2, 2, 2)).at[..., 0].set(0.03),
        initial_displacement_controls=jnp.zeros((2, 2, 2)),
        site_coordinates=lattice_model.site_coordinates,
        displaced_site_coordinates=lattice_model.site_coordinates
        + jnp.array([0.03, 0.0]),
        control_coordinates_s=lattice_model.control_coordinates_s,
        control_coordinates_u=lattice_model.control_coordinates_u,
        predicted_intensities=scan.intensities,
        measured_intensities=scan.intensities,
        window_starts=scan.window_starts,
        scan_coordinates=scan.scan_coordinates,
        detector_angles=scan.detector_angles,
        update_history=jnp.array([0, 10]),
        elapsed_time_history=jnp.array([0.5, 1.5]),
        training_loss_history=jnp.array([1.0, 0.1]),
        validation_loss_history=jnp.array([1.1, 0.2]),
        best_update=10,
        completed_updates=10,
        converged=True,
        stop_reason="plateau",
        audit_loss=0.3,
        gradient_norm_history=jnp.array([0.5, 0.01]),
        normalized_step_history=jnp.array([0.2, 1e-5]),
        active_bound_fraction_history=jnp.array([0.5, 0.25]),
        rigid_displacement=jnp.array([0.03, -0.01]),
        initial_rigid_displacement=jnp.array([0.01, 0.0]),
        rigid_displacement_history=jnp.array([[0.01, 0.0], [0.03, -0.01]]),
        optimization_stage_history=np.asarray(["initial", "joint"]),
        checkpoint_updates=jnp.array([0, 10]),
        vacancy_fraction_history=jnp.array([[0.0, 0.0], [0.9, 0.1]]),
        displacement_control_history=jnp.stack(
            [jnp.zeros((2, 2, 2)), jnp.zeros((2, 2, 2)).at[..., 0].set(0.03)]
        ),
        metadata={"species": "Si"},
        detector_valid_mask=detector_valid_mask,
    )
    lattice_path = tmp_path / "lattice_result.npz"
    save_lattice_site_reconstruction_1d(lattice_path, lattice_result)
    loaded_lattice = load_lattice_site_reconstruction_1d(lattice_path)
    np.testing.assert_allclose(
        loaded_lattice.vacancy_fractions, lattice_result.vacancy_fractions
    )
    np.testing.assert_allclose(
        loaded_lattice.displacement_controls,
        lattice_result.displacement_controls,
    )
    np.testing.assert_allclose(
        loaded_lattice.vacancy_fraction_history,
        lattice_result.vacancy_fraction_history,
    )
    np.testing.assert_allclose(
        loaded_lattice.displacement_control_history,
        lattice_result.displacement_control_history,
    )
    assert loaded_lattice.completed_updates == 10
    assert loaded_lattice.converged is True
    assert loaded_lattice.stop_reason == "plateau"
    assert loaded_lattice.audit_loss == pytest.approx(0.3)
    np.testing.assert_array_equal(
        loaded_lattice.detector_valid_mask, detector_valid_mask
    )
    np.testing.assert_allclose(
        loaded_lattice.rigid_displacement, lattice_result.rigid_displacement
    )
    np.testing.assert_allclose(
        loaded_lattice.rigid_displacement_history,
        lattice_result.rigid_displacement_history,
    )
    np.testing.assert_array_equal(
        loaded_lattice.optimization_stage_history,
        lattice_result.optimization_stage_history,
    )
    assert loaded_lattice.metadata == {"species": "Si"}


def test_calibrated_lattice_result_round_trip_without_pickle(
    tmp_path,
    small_poisson_reconstruction,
):
    path = tmp_path / "calibrated_lattice_result.npz"
    save_lattice_site_reconstruction_1d(path, small_poisson_reconstruction)
    with np.load(path, allow_pickle=False) as archive:
        assert bool(archive["measurement_present"].item())
        assert bool(archive["objective_present"].item())
        assert bool(archive["predicted_signal_electrons_present"].item())
    loaded = load_lattice_site_reconstruction_1d(path)
    assert loaded.measurement is not None and loaded.objective is not None
    np.testing.assert_allclose(
        loaded.predicted_signal_electrons,
        small_poisson_reconstruction.predicted_signal_electrons,
    )
    np.testing.assert_array_equal(
        loaded.measurement.valid_mask,
        small_poisson_reconstruction.measurement.valid_mask,
    )
    np.testing.assert_allclose(
        loaded.measurement.observed_total_electrons,
        small_poisson_reconstruction.measurement.observed_total_electrons,
    )
    np.testing.assert_allclose(
        loaded.objective.electrons_per_pattern,
        small_poisson_reconstruction.objective.electrons_per_pattern,
    )
    assert loaded.metadata["likelihood_interpretation"] == (
        small_poisson_reconstruction.metadata["likelihood_interpretation"]
    )


def test_prepared_gaussian_approximation_runs_and_persists_label(
    tmp_path,
    prepared_small_poisson_problem,
):
    prepared = prepared_small_poisson_problem
    assert prepared.measurement is not None and prepared.objective is not None
    measurement = replace(
        prepared.measurement,
        calibrated_signal_electrons=(
            prepared.measurement.calibrated_signal_electrons.at[0, 1].set(-0.5)
        ),
        observed_total_electrons=(
            prepared.measurement.observed_total_electrons.at[0, 1].set(-0.3)
        ),
        calibrated_read_noise_std_electrons=1.25,
        calibration_id="declared_read_noise",
    )
    objective = replace(prepared.objective, kind="poisson_gaussian_nll")
    gaussian_prepared = _reprepare_calibrated(
        prepared, measurement, objective
    )
    result = _prepared_run(
        gaussian_prepared, updates=1, validation_interval=1
    )
    assert np.all(np.isfinite(result.training_loss_history))
    assert result.metadata["measurement_contract"] == (
        "heteroscedastic_poisson_gaussian_approximation"
    )
    assert "not the exact Poisson-Gaussian convolution" in result.metadata[
        "likelihood_interpretation"
    ]
    assert result.metadata["structural_trust_from_measurement_objective"] is False

    path = tmp_path / "gaussian_approximation.npz"
    save_lattice_site_reconstruction_1d(path, result)
    loaded = load_lattice_site_reconstruction_1d(path)
    assert loaded.objective is not None
    assert loaded.objective.kind == "poisson_gaussian_nll"
    assert loaded.metadata["likelihood_interpretation"] == (
        result.metadata["likelihood_interpretation"]
    )


def test_reconstruction_rejects_phase_wrapping_bound():
    n_s, n_u = 5, 12
    initial = jnp.ones((n_s, n_u))
    mask = jnp.ones_like(initial, dtype=bool)
    measured = jnp.ones((2, n_u))
    with pytest.raises(ValueError, match="phase bound"):
        reconstruct_potential_1d(
            initial,
            mask,
            _probe(n_u=n_u, du=0.3),
            jnp.array([0, 1]),
            4,
            _kernel(n_u=n_u, du=0.3, ds=0.4),
            0.4,
            ENERGY,
            measured,
            potential_scale=1.0,
            potential_max=1e9,
            updates=1,
        )


@pytest.mark.parametrize(
    ("initial_parameters", "exception", "message"),
    [
        (
            {"initial_vacancy_fractions": jnp.array([1.2, 0.0])},
            ValueError,
            r"\[0, 1\]",
        ),
        (
            {"initial_vacancy_fractions": jnp.array([jnp.nan, 0.0])},
            ValueError,
            "finite",
        ),
        (
            {
                "initial_vacancy_fractions": jnp.array(
                    [0.0 + 0.1j, 0.0]
                )
            },
            TypeError,
            "must be real",
        ),
        (
            {
                "initial_displacement_controls": jnp.zeros(
                    (2, 2, 2), dtype=jnp.complex128
                )
            },
            TypeError,
            "must be real",
        ),
        (
            {
                "initial_rigid_displacement": jnp.array(
                    [0.0 + 0.1j, 0.0]
                )
            },
            TypeError,
            "must be real",
        ),
    ],
)
def test_lattice_reconstruction_rejects_invalid_initial_parameters(
    initial_parameters, exception, message
):
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    model = _small_lattice_model()
    n_u = model.reference_potential.shape[1]
    parameters = {
        "initial_displacement_controls": jnp.zeros((2, 2, 2)),
        **initial_parameters,
    }
    with pytest.raises(exception, match=message):
        reconstruct_lattice_site_potential_1d(
            model,
            _probe(n_u=n_u, du=0.3),
            jnp.array([0]),
            4,
            _kernel(n_u=n_u, du=0.3, ds=0.4),
            0.4,
            ENERGY,
            jnp.ones((1, n_u)),
            updates=1,
            **parameters,
        )


def test_tiny_lattice_vacancy_reconstruction_recovers_site_fraction():
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    model = _small_lattice_model(maximum_displacement=0.0)
    u = (jnp.arange(12) - 6) * 0.3
    probe = jnp.exp(-0.5 * ((u + 0.1) / 0.65) ** 2) * jnp.exp(0.25j * u)
    kernel = fresnel_propagation_kernel_1d(12, 0.3, 0.4, ENERGY)
    starts = jnp.arange(5)
    controls = jnp.zeros((2, 2, 2))
    target_vacancies = jnp.array([0.85, 0.0])
    target = render_lattice_site_potential_1d(model, target_vacancies, controls)
    measured = simulate_glancing_scan_1d(target, probe, starts, 4, kernel, 0.4, ENERGY)
    initial_prediction = simulate_glancing_scan_1d(
        model.reference_potential, probe, starts, 4, kernel, 0.4, ENERGY
    )
    initial_loss = normalized_amplitude_loss_1d(initial_prediction, measured)

    result = reconstruct_lattice_site_potential_1d(
        model,
        probe,
        starts,
        4,
        kernel,
        0.4,
        ENERGY,
        measured,
        potential_max=10.0,
        learning_rate_start=0.1,
        learning_rate_end=1e-3,
        updates=100,
        minibatch_size=5,
        validation_interval=20,
        evaluation_batch_size=5,
        rematerialize=False,
        checkpoint_interval=25,
    )
    recovered_loss = normalized_amplitude_loss_1d(
        result.predicted_intensities, measured
    )
    assert float(recovered_loss) < 1e-4 * float(initial_loss)
    np.testing.assert_allclose(result.vacancy_fractions, target_vacancies, atol=4e-3)
    np.testing.assert_array_equal(result.checkpoint_updates, [0, 25, 50, 75, 100])
    assert result.vacancy_fraction_history.shape == (5, 2)
    assert result.displacement_control_history.shape == (5, 2, 2, 2)
    assert result.completed_updates == 100
    assert result.converged is False
    assert result.stop_reason == "maximum_updates"
    assert result.gradient_norm_history.shape == result.update_history.shape
    assert result.normalized_step_history.shape == result.update_history.shape


def test_lattice_reconstruction_reports_target_loss_stopping():
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    model = _small_lattice_model(maximum_displacement=0.0)
    u = (jnp.arange(12) - 6) * 0.3
    probe = jnp.exp(-0.5 * ((u + 0.1) / 0.65) ** 2)
    kernel = fresnel_propagation_kernel_1d(12, 0.3, 0.4, ENERGY)
    starts = jnp.arange(3)
    measured = simulate_glancing_scan_1d(
        model.reference_potential, probe, starts, 4, kernel, 0.4, ENERGY
    )
    result = reconstruct_lattice_site_potential_1d(
        model,
        probe,
        starts,
        4,
        kernel,
        0.4,
        ENERGY,
        measured,
        potential_max=10.0,
        updates=10,
        minibatch_size=3,
        validation_interval=1,
        evaluation_batch_size=3,
        rematerialize=False,
        audit_indices=[2],
        convergence=ConvergenceOptions1D(min_updates=1, target_loss=1e-12),
    )
    assert result.converged is True
    assert result.stop_reason == "target_loss"
    assert result.completed_updates == 1
    assert result.audit_loss == pytest.approx(0.0, abs=1e-12)
    assert 2 not in result.metadata["training_indices"]


def test_tiny_staged_budget_preserves_joint_updates():
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    model = _small_lattice_model()
    n_u = model.reference_potential.shape[1]
    probe = _probe(n_u=n_u, du=0.3)
    kernel = _kernel(n_u=n_u, du=0.3, ds=0.4)
    starts = jnp.arange(3)
    measured = simulate_glancing_scan_1d(
        model.reference_potential, probe, starts, 4, kernel, 0.4, ENERGY
    )

    result = reconstruct_lattice_site_potential_1d(
        model,
        probe,
        starts,
        4,
        kernel,
        0.4,
        ENERGY,
        measured,
        separate_rigid_registration=True,
        maximum_rigid_displacement=0.15,
        maximum_residual_displacement=0.35,
        potential_max=10.0,
        updates=3,
        minibatch_size=3,
        validation_interval=1,
        evaluation_batch_size=3,
        rematerialize=False,
        optimization=LatticeOptimizationOptions1D(
            mode="staged",
            rigid_stage_fraction=0.33,
            vacancy_stage_fraction=0.33,
            residual_stage_fraction=0.33,
        ),
    )

    np.testing.assert_array_equal(
        result.optimization_stage_history,
        ["initial", "joint", "joint", "joint"],
    )
    assert result.metadata["optimization_stage_boundaries"] == {
        "site_translation_end": 0,
        "vacancy_end": 0,
        "residual_end": 0,
        "joint_end": 3,
    }


def test_tiny_lattice_strain_reconstruction_recovers_site_displacements():
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    model = _small_lattice_model(maximum_displacement=0.5)
    u = (jnp.arange(12) - 6) * 0.3
    base_probe = jnp.exp(-0.5 * ((u + 0.1) / 0.65) ** 2) * jnp.exp(0.25j * u)
    probes = jnp.stack([jnp.roll(base_probe, index - 2) for index in range(5)])
    kernel = fresnel_propagation_kernel_1d(12, 0.3, 0.4, ENERGY)
    starts = jnp.arange(5)
    target_controls = jnp.zeros((2, 2, 2))
    target_controls = target_controls.at[0, :, 1].set(0.12)
    target_controls = target_controls.at[1, :, 1].set(-0.12)
    target = render_lattice_site_potential_1d(model, jnp.zeros(2), target_controls)
    measured = simulate_glancing_scan_1d(target, probes, starts, 4, kernel, 0.4, ENERGY)

    result = reconstruct_lattice_site_potential_1d(
        model,
        probes,
        starts,
        4,
        kernel,
        0.4,
        ENERGY,
        measured,
        potential_max=10.0,
        learning_rate_start=0.05,
        learning_rate_end=5e-4,
        updates=300,
        minibatch_size=5,
        validation_interval=50,
        evaluation_batch_size=5,
        rematerialize=False,
    )
    site_s_fraction = np.asarray(model.site_coordinates[:, 0]) / float(
        model.control_coordinates_s[-1]
    )
    expected_u_displacement = 0.12 - 0.24 * site_s_fraction
    recovered_displacement = np.asarray(
        result.displaced_site_coordinates - result.site_coordinates
    )
    np.testing.assert_allclose(
        recovered_displacement[:, 1], expected_u_displacement, atol=1e-4
    )
    np.testing.assert_allclose(recovered_displacement[:, 0], 0.0, atol=1e-3)
    # Smooth displacement gradients can leave a tiny transient occupancy
    # compensation, but it must remain orders of magnitude below the 0.5
    # vacancy decision threshold.
    np.testing.assert_allclose(result.vacancy_fractions, 0.0, atol=1e-3)


def test_site_translation_is_recovered_separately_from_residual_motion():
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    model = _small_lattice_model(maximum_displacement=0.5)
    u = (jnp.arange(12) - 6) * 0.3
    base_probe = jnp.exp(-0.5 * ((u + 0.1) / 0.65) ** 2) * jnp.exp(0.25j * u)
    probes = jnp.stack([jnp.roll(base_probe, index - 2) for index in range(5)])
    kernel = fresnel_propagation_kernel_1d(12, 0.3, 0.4, ENERGY)
    starts = jnp.arange(5)
    target_rigid = jnp.array([0.08, -0.06])
    target_controls = jnp.broadcast_to(target_rigid, (2, 2, 2))
    target = render_lattice_site_potential_1d(
        model, jnp.zeros(2), target_controls
    )
    measured = simulate_glancing_scan_1d(
        target, probes, starts, 4, kernel, 0.4, ENERGY
    )
    result = reconstruct_lattice_site_potential_1d(
        model,
        probes,
        starts,
        4,
        kernel,
        0.4,
        ENERGY,
        measured,
        potential_max=10.0,
        separate_rigid_registration=True,
        maximum_rigid_displacement=0.15,
        maximum_residual_displacement=0.35,
        learning_rate_start=0.05,
        learning_rate_end=5e-4,
        updates=200,
        minibatch_size=5,
        validation_interval=40,
        evaluation_batch_size=5,
        rematerialize=False,
        optimization=LatticeOptimizationOptions1D(
            mode="staged",
            rigid_stage_fraction=0.4,
            vacancy_stage_fraction=0.0,
            residual_stage_fraction=0.0,
        ),
    )
    residual_sites = lattice_site_displacements_1d(
        result.site_coordinates,
        result.displacement_controls,
        result.control_coordinates_s,
        result.control_coordinates_u,
    )
    np.testing.assert_allclose(result.rigid_displacement, target_rigid, atol=1.5e-2)
    np.testing.assert_allclose(np.mean(residual_sites, axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(residual_sites, 0.0, atol=2e-3)
    np.testing.assert_array_equal(
        result.optimization_stage_history,
        ["initial", "site_translation", "site_translation", "joint", "joint", "joint"],
    )


def test_tiny_direct_potential_reconstruction_reduces_loss_and_recovers_shape():
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    n_s, n_u = 7, 24
    du, ds = 0.3, 0.4
    u = (jnp.arange(n_u) - n_u // 2) * du
    probe = jnp.exp(-0.5 * ((u + 0.1) / 0.65) ** 2) * jnp.exp(0.25j * u)
    kernel = fresnel_propagation_kernel_1d(n_u, du, ds, ENERGY)
    starts = jnp.arange(5)
    mask = jnp.zeros((n_s, n_u), dtype=bool)
    mask = mask.at[1:6, 9:15].set(True)
    s_profile = jnp.exp(-0.5 * ((jnp.arange(n_s) - 3.0) / 1.1) ** 2)
    u_profile = jnp.exp(-0.5 * ((u - 0.15) / 0.55) ** 2)
    target = 650.0 * s_profile[:, None] * u_profile[None, :] * mask
    initial = 60.0 * mask
    measured = simulate_glancing_scan_1d(target, probe, starts, 3, kernel, ds, ENERGY)
    initial_prediction = simulate_glancing_scan_1d(
        initial, probe, starts, 3, kernel, ds, ENERGY
    )
    initial_loss = normalized_amplitude_loss_1d(initial_prediction, measured)

    result = reconstruct_potential_1d(
        initial,
        mask,
        probe,
        starts,
        3,
        kernel,
        ds,
        ENERGY,
        measured,
        transverse_coordinates=u,
        potential_scale=500.0,
        potential_max=900.0,
        learning_rate_start=4e-2,
        learning_rate_end=5e-4,
        updates=300,
        minibatch_size=5,
        validation_interval=20,
        evaluation_batch_size=5,
        rematerialize=False,
        seed=4,
    )
    recovered_loss = normalized_amplitude_loss_1d(
        result.predicted_intensities, measured
    )
    correlation = np.corrcoef(
        np.asarray(result.potential)[np.asarray(mask)],
        np.asarray(target)[np.asarray(mask)],
    )[0, 1]
    assert float(recovered_loss) < 0.05 * float(initial_loss)
    assert correlation > 0.9
