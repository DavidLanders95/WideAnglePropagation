"""Marginal Fisher observability and gauge-free displacement tests."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (  # noqa: E402
    fresnel_propagation_kernel_1d,
)
from wide_angle_propagation.ptychography_1d import (  # noqa: E402
    LatticeSiteModel1D,
    LatticeSiteReconstruction1D,
    PtychographyMeasurement1D,
    PtychographyObjective1D,
    prepare_lattice_site_reconstruction_1d,
    ptychography_expected_signal_electrons_1d,
    render_lattice_site_potential_1d,
    simulate_glancing_scan_1d,
)
from wide_angle_propagation.ptychography_diagnostics_1d import (  # noqa: E402
    PoissonCountingModel1D,
)
from wide_angle_propagation import (  # noqa: E402
    ptychography_observability_1d as observability,
)
from wide_angle_propagation.ptychography_observability_1d import (  # noqa: E402
    MatrixFreeObservabilityOptions1D,
    PreparedNuisanceOptions1D,
    SiteObservabilityOptions1D,
    WhitenedNuisanceProfile1D,
    estimate_lattice_site_observability_1d,
    estimate_prepared_lattice_site_observability_matrix_free_1d,
    estimate_prepared_lattice_site_observability_stochastic_1d,
    lattice_displacement_basis_1d,
    load_lattice_site_observability_1d,
    marginal_covariance_from_jacobian_1d,
    pcg_solve_observability_1d,
    poisson_counting_model_from_prepared_1d,
    prepared_whitened_nuisance_profile_1d,
    save_lattice_site_observability_1d,
)
from wide_angle_propagation.ptychography_stochastic_observability_1d import (  # noqa: E402
    StochasticFisherScreeningOptions1D,
)


ENERGY = 30e3


def _problem():
    shape = (9, 10)
    patch = np.asarray(
        [[0.0, 0.3, 0.0], [0.1, 2.0, 0.5], [0.0, 0.2, 0.0]],
        dtype=float,
    )
    patch_starts = np.asarray([[1, 2], [5, 6]], dtype=np.int32)
    reference = np.full(shape, 0.02, dtype=float)
    for start in patch_starts:
        reference[
            start[0] : start[0] + 3,
            start[1] : start[1] + 3,
        ] += patch
    sites = np.column_stack(
        [(patch_starts[:, 0] + 1) * 0.4, (patch_starts[:, 1] + 1) * 0.3]
    )
    model = LatticeSiteModel1D(
        reference_potential=jnp.asarray(reference),
        site_coordinates=jnp.asarray(sites),
        site_patches=jnp.asarray(np.stack([patch, patch])),
        patch_starts=jnp.asarray(patch_starts),
        control_coordinates_s=jnp.asarray([0.0, 3.2]),
        control_coordinates_u=jnp.asarray([0.0, 2.7]),
        axial_sampling=0.4,
        transverse_sampling=0.3,
        maximum_displacement=0.5,
    )
    starts = jnp.asarray([0, 1, 4, 5])
    n_u = shape[1]
    u = (jnp.arange(n_u) - n_u // 2) * 0.3
    base_probe = jnp.exp(-0.5 * ((u + 0.1) / 0.7) ** 2) * jnp.exp(0.2j * u)
    probes = jnp.stack([jnp.roll(base_probe, shift) for shift in (-1, 0, 1, 2)])
    result = LatticeSiteReconstruction1D(
        potential=reference,
        initial_potential=reference,
        vacancy_fractions=np.asarray([0.05, 0.05]),
        initial_vacancy_fractions=np.zeros(2),
        displacement_controls=np.zeros((2, 2, 2)),
        initial_displacement_controls=np.zeros((2, 2, 2)),
        site_coordinates=sites,
        displaced_site_coordinates=sites,
        control_coordinates_s=np.asarray([0.0, 3.2]),
        control_coordinates_u=np.asarray([0.0, 2.7]),
        predicted_intensities=np.zeros((4, n_u)),
        measured_intensities=np.zeros((4, n_u)),
        window_starts=starts,
        scan_coordinates=np.arange(4.0),
        detector_angles=np.arange(n_u),
        update_history=np.asarray([0]),
        elapsed_time_history=np.asarray([0.0]),
        training_loss_history=np.asarray([0.0]),
        validation_loss_history=np.asarray([0.0]),
        best_update=0,
        rigid_displacement=np.zeros(2),
        metadata={
            "best_metric": 0.0,
            "training_indices": [0, 2],
            "audit_indices": [1, 3],
        },
    )
    kernel = fresnel_propagation_kernel_1d(n_u, 0.3, 0.4, ENERGY)
    return model, result, probes, starts, kernel


@pytest.fixture(scope="module")
def prepared_observability_problem():
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    model, result, probes, starts, kernel = _problem()
    measured = simulate_glancing_scan_1d(
        model.reference_potential,
        probes,
        starts,
        4,
        kernel,
        0.4,
        ENERGY,
        rematerialize=False,
    )
    detector_mask = np.ones(measured.shape, dtype=bool)
    detector_mask[:, 0] = False
    detector_mask[0, 1] = False
    prepared = prepare_lattice_site_reconstruction_1d(
        model,
        probes,
        starts,
        4,
        kernel,
        0.4,
        ENERGY,
        measured,
        detector_valid_mask=detector_mask,
        audit_indices=[1, 3],
        potential_max=10.0,
        minibatch_size=2,
        evaluation_batch_size=3,
        rematerialize=False,
    )
    vacancies = jnp.asarray([0.05, 0.05])
    controls = jnp.zeros((2, 2, 2), dtype=prepared.model.reference_potential.dtype)
    potential = render_lattice_site_potential_1d(
        prepared.model, vacancies, controls
    )
    prediction = simulate_glancing_scan_1d(
        potential,
        prepared.probe_rows,
        prepared.window_starts,
        prepared.window_length,
        prepared.propagation_kernel,
        prepared.slice_thickness,
        prepared.energy,
        rematerialize=prepared.rematerialize,
    )
    reconstruction = replace(
        result,
        potential=potential,
        initial_potential=prepared.model.reference_potential,
        vacancy_fractions=vacancies,
        displacement_controls=controls,
        site_coordinates=prepared.model.site_coordinates,
        displaced_site_coordinates=prepared.model.site_coordinates,
        control_coordinates_s=prepared.model.control_coordinates_s,
        control_coordinates_u=prepared.model.control_coordinates_u,
        predicted_intensities=prediction,
        measured_intensities=prepared.measured_intensities,
        detector_valid_mask=prepared.detector_valid_mask,
        window_starts=prepared.window_starts,
        scan_coordinates=prepared.scan_coordinates,
        detector_angles=prepared.detector_angles,
        metadata={
            "reconstruction_problem_id": prepared.reconstruction_problem_id,
            "reconstructor_id": prepared.reconstructor_id,
            "objective_id": prepared.objective_id,
            "training_indices": np.asarray(prepared.training_indices).tolist(),
            "validation_indices": [],
            "audit_indices": np.asarray(prepared.audit_indices).tolist(),
            "excluded_indices": [],
            "seed": 0,
        },
    )
    return prepared, reconstruction


@pytest.fixture(scope="module")
def prepared_poisson_observability_problem(prepared_observability_problem):
    source, reconstruction = prepared_observability_problem
    objective = PtychographyObjective1D(
        kind="poisson_deviance",
        electrons_per_pattern=1_000_000.0,
        minimum_expected_electrons=1e-7,
        relative_signal_scale=0.75,
    )
    signal = ptychography_expected_signal_electrons_1d(
        source.measured_intensities,
        source.probe_rows,
        objective,
    )
    measurement = PtychographyMeasurement1D(
        calibrated_signal_electrons=signal,
        observed_total_electrons=signal + 0.25,
        valid_mask=source.detector_valid_mask,
        calibrated_dark_electrons_per_pixel=0.25,
        calibrated_read_noise_std_electrons=0.0,
        calibration_id="prepared-poisson-observability-v1",
    )
    prepared = prepare_lattice_site_reconstruction_1d(
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
        audit_indices=np.asarray(source.audit_indices),
        potential_max=source.potential_max,
        minibatch_size=source.minibatch_size,
        evaluation_batch_size=source.evaluation_batch_size,
        rematerialize=source.rematerialize,
    )
    prediction = simulate_glancing_scan_1d(
        reconstruction.potential,
        prepared.probe_rows,
        prepared.window_starts,
        prepared.window_length,
        prepared.propagation_kernel,
        prepared.slice_thickness,
        prepared.energy,
        rematerialize=prepared.rematerialize,
    )
    predicted_signal = ptychography_expected_signal_electrons_1d(
        prediction,
        prepared.probe_rows,
        prepared.objective,
    )
    reconstruction = replace(
        reconstruction,
        predicted_intensities=prediction,
        predicted_signal_electrons=predicted_signal,
        measured_intensities=prepared.measured_intensities,
        detector_valid_mask=prepared.detector_valid_mask,
        measurement=prepared.measurement,
        objective=prepared.objective,
        metadata={
            **dict(reconstruction.metadata),
            "reconstruction_problem_id": prepared.reconstruction_problem_id,
            "reconstructor_id": prepared.reconstructor_id,
            "objective_id": prepared.objective_id,
            "training_indices": np.asarray(prepared.training_indices).tolist(),
            "validation_indices": np.asarray(
                prepared.validation_indices
            ).tolist(),
            "audit_indices": np.asarray(prepared.audit_indices).tolist(),
            "excluded_indices": np.asarray(prepared.excluded_indices).tolist(),
        },
    )
    return prepared, reconstruction


def test_marginal_covariance_marks_correlated_null_directions_unestimable():
    covariance, estimable, rank = marginal_covariance_from_jacobian_1d(
        [[1.0, 1.0]], np.eye(2)
    )
    assert rank == 1
    np.testing.assert_array_equal(estimable, [False, False])
    assert np.isinf(np.diag(covariance)).all()

    covariance, estimable, _ = marginal_covariance_from_jacobian_1d(
        [[1.0, 0.0]], [[0.0, 1e-12]]
    )
    np.testing.assert_array_equal(estimable, [False])
    assert np.isinf(covariance[0, 0])

    covariance, estimable, rank = marginal_covariance_from_jacobian_1d(
        np.diag([2.0, 4.0]), np.eye(2)
    )
    assert rank == 2
    np.testing.assert_array_equal(estimable, [True, True])
    np.testing.assert_allclose(covariance, np.diag([0.25, 0.0625]))

    for output_scale in (1.0, 1e-6, 1e-12):
        covariance, estimable, rank = marginal_covariance_from_jacobian_1d(
            [[1.0, 0.0]], [[0.0, output_scale]]
        )
        assert rank == 1
        assert not estimable[0]
        assert np.isinf(covariance[0, 0])

    covariance, estimable, _ = marginal_covariance_from_jacobian_1d(
        [[1.0, 0.0]], [[1.0, 1e-8]], rank_rtol=1e-9
    )
    assert not estimable[0]
    assert np.isinf(covariance[0, 0])


def test_displacement_basis_removes_mean_gauge_and_preserves_site_motion():
    model, *_ = _problem()
    basis = lattice_displacement_basis_1d(model)
    interpolation = np.asarray(basis.interpolation_matrix)
    controls = np.asarray(basis.control_basis)
    sites = np.asarray(basis.site_basis)
    np.testing.assert_allclose(interpolation @ controls, sites, atol=1e-11)
    np.testing.assert_allclose(np.mean(sites, axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.mean(sites**2, axis=0)), 1.0)
    assert basis.relative_reconstruction_error < 1e-10


def test_displacement_basis_is_dtype_aware_and_supports_zero_residual_rank():
    model, *_ = _problem()
    float32_model = replace(
        model,
        reference_potential=jnp.asarray(model.reference_potential, dtype=jnp.float32),
        site_coordinates=jnp.asarray(model.site_coordinates, dtype=jnp.float32),
        site_patches=jnp.asarray(model.site_patches, dtype=jnp.float32),
        control_coordinates_s=jnp.asarray(
            model.control_coordinates_s, dtype=jnp.float32
        ),
        control_coordinates_u=jnp.asarray(
            model.control_coordinates_u, dtype=jnp.float32
        ),
    )
    float32_basis = lattice_displacement_basis_1d(float32_model)
    assert float32_basis.numerical_rank == 1
    np.testing.assert_allclose(
        np.asarray(float32_basis.interpolation_matrix)
        @ np.asarray(float32_basis.control_basis),
        np.asarray(float32_basis.site_basis),
        atol=2e-6,
    )

    one_site_model = replace(
        model,
        site_coordinates=model.site_coordinates[:1],
        site_patches=model.site_patches[:1],
        patch_starts=model.patch_starts[:1],
    )
    zero_rank = lattice_displacement_basis_1d(one_site_model)
    assert zero_rank.numerical_rank == 0
    assert zero_rank.control_basis.shape == (4, 0)
    assert zero_rank.site_basis.shape == (1, 0)
    assert zero_rank.relative_reconstruction_error == 0.0


def test_prepared_nuisance_constructor_is_calibration_bound_and_analytic(
    prepared_poisson_observability_problem,
):
    prepared, reconstruction = prepared_poisson_observability_problem
    options = PreparedNuisanceOptions1D(
        include_scan_origin_shift=False,
        include_probe_transverse_shift=False,
        include_probe_tilt=False,
        include_probe_log_width=False,
        include_detector_frequency_offset=False,
        include_detector_log_gain=True,
        include_detector_dark_offset=True,
    )
    profile = prepared_whitened_nuisance_profile_1d(
        prepared,
        reconstruction,
        options=options,
    )
    repeated = prepared_whitened_nuisance_profile_1d(
        prepared,
        reconstruction,
        options=options,
    )
    assert profile.parameter_names == (
        "detector_log_signal_gain",
        "detector_dark_offset_electrons",
    )
    assert profile.profile_id == repeated.profile_id
    assert profile.metadata["calibration_id"] == (
        prepared.measurement.calibration_id
    )
    assert profile.metadata["nuisance_scope_complete"] is False
    assert profile.metadata["nuisance_prior"] == (
        "unconstrained_local_profile_span"
    )
    coverage = profile.metadata["coverage"]
    assert coverage["scan_geometry"][
        "common_relative_axial_origin_shift"
    ] is False
    assert not any(coverage["probe"].values())
    assert coverage["detector_calibration"] == {
        "common_reciprocal_frequency_offset": False,
        "common_log_signal_gain": True,
        "common_dark_offset": True,
    }
    assert (
        "scan_geometry.common_relative_axial_origin_shift"
        in profile.metadata["missing_nuisance_scopes"]
    )
    tangent = np.asarray(profile.tangent_matrix)
    signal = np.asarray(reconstruction.predicted_signal_electrons)
    dark = np.asarray(
        prepared.measurement.calibrated_dark_electrons_per_pixel
    )
    valid = np.asarray(prepared.detector_valid_mask)
    mean = signal + dark
    expected_gain = np.where(valid, signal / np.sqrt(mean), 0.0)
    expected_dark = np.where(valid, 1.0 / np.sqrt(mean), 0.0)
    np.testing.assert_allclose(tangent[..., 0], expected_gain, rtol=2e-10)
    np.testing.assert_allclose(tangent[..., 1], expected_dark, rtol=2e-10)
    assert np.all(tangent[~valid] == 0.0)

    report = estimate_prepared_lattice_site_observability_matrix_free_1d(
        prepared,
        reconstruction,
        nuisance_profile=profile,
        site_indices=[0],
        options=_matrix_free_options(),
    )
    assert report.metadata["generated_nuisance_profile"] is True
    assert report.metadata["nuisance_profile_id"] == profile.profile_id
    assert report.metadata["represented_nuisance_coverage"] == coverage

    wrong_calibration = replace(
        profile,
        metadata={**dict(profile.metadata), "calibration_id": "wrong"},
    )
    with pytest.raises(ValueError, match="calibration_id"):
        estimate_prepared_lattice_site_observability_matrix_free_1d(
            prepared,
            reconstruction,
            nuisance_profile=wrong_calibration,
            site_indices=[0],
            options=_matrix_free_options(),
        )

    wrong_identifier = replace(profile, profile_id=profile.profile_id + "0")
    with pytest.raises(ValueError, match="identifier does not authenticate"):
        estimate_prepared_lattice_site_observability_matrix_free_1d(
            prepared,
            reconstruction,
            nuisance_profile=wrong_identifier,
            site_indices=[0],
            options=_matrix_free_options(),
        )

    changed_vacancies = np.asarray(reconstruction.vacancy_fractions).copy()
    changed_vacancies[0] += 0.01
    total_controls = np.asarray(reconstruction.displacement_controls) + np.asarray(
        reconstruction.rigid_displacement
    )[None, None, :]
    changed_potential = render_lattice_site_potential_1d(
        prepared.model,
        changed_vacancies,
        total_controls,
    )
    changed_reconstruction = replace(
        reconstruction,
        vacancy_fractions=changed_vacancies,
        potential=changed_potential,
    )
    with pytest.raises(ValueError, match="reconstruction_state_sha256"):
        estimate_prepared_lattice_site_observability_matrix_free_1d(
            prepared,
            changed_reconstruction,
            nuisance_profile=profile,
            site_indices=[0],
            options=_matrix_free_options(),
        )

    complete_low_rank_profile = prepared_whitened_nuisance_profile_1d(
        prepared,
        reconstruction,
    )
    assert complete_low_rank_profile.parameter_names == (
        "scan_origin_shift_A",
        "probe_transverse_shift_A",
        "probe_tilt_rad",
        "probe_log_width",
        "detector_frequency_offset_inverse_A",
        "detector_log_signal_gain",
        "detector_dark_offset_electrons",
    )
    complete_tangent = np.asarray(complete_low_rank_profile.tangent_matrix)
    assert complete_tangent.shape == (*signal.shape, 7)
    assert np.all(np.isfinite(complete_tangent))
    assert np.all(complete_tangent[~valid] == 0.0)
    assert np.all(np.linalg.norm(complete_tangent.reshape(-1, 7), axis=0) > 0.0)
    assert all(
        all(group.values())
        for group in complete_low_rank_profile.metadata["coverage"].values()
    )


def test_physical_output_jvp_and_vjp_match_dense_adjoint(
    prepared_observability_problem,
):
    prepared, reconstruction = prepared_observability_problem
    parameterization = observability._gauge_free_specimen_parameterization_1d(
        prepared.model,
        reconstruction,
        _matrix_free_options(),
    )
    dense = parameterization.physical_output_jacobian()
    parameter_direction = np.linspace(
        -0.7,
        0.9,
        parameterization.n_parameter,
    )
    output_cotangent = np.linspace(
        0.8,
        -0.4,
        parameterization.n_physical_output,
    )
    np.testing.assert_allclose(
        parameterization.physical_output_jvp(parameter_direction),
        dense @ parameter_direction,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        parameterization.physical_output_vjp(output_cotangent),
        dense.T @ output_cotangent,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        parameterization.physical_output_row_norm_squared(),
        np.sum(dense**2, axis=1),
        atol=1e-13,
    )
    assert np.dot(
        parameterization.physical_output_jvp(parameter_direction),
        output_cotangent,
    ) == pytest.approx(
        np.dot(
            parameter_direction,
            parameterization.physical_output_vjp(output_cotangent),
        )
    )


def test_prepared_stochastic_adapter_screens_all_sites_fail_closed(
    prepared_poisson_observability_problem,
):
    prepared, reconstruction = prepared_poisson_observability_problem
    report = estimate_prepared_lattice_site_observability_stochastic_1d(
        prepared,
        reconstruction,
        operator_options=_matrix_free_options(exhaustive=False),
        screening_options=StochasticFisherScreeningOptions1D(
            covariance_probe_count=2,
            null_probe_count=2,
            random_seed=17,
            maximum_iterations=64,
            relative_residual_tolerance=1e-8,
            operator_check_vectors=1,
            maximum_pcg_solves=4,
            maximum_total_pcg_iterations=256,
            maximum_fisher_matvec_calls=1024,
        ),
    )
    assert report.structurally_trusted is False
    assert report.suitable_for_trust_gate is False
    assert report.metadata["screening_only"] is True
    assert report.metadata["all_site_count"] == len(
        prepared.model.site_coordinates
    )
    assert report.fit.screening.factor_covariance_verified is True
    assert report.fit.screening.numerically_valid
    assert report.fit.screening.physical_marginal_variance.shape == (6,)
    assert len(report.fit.screening.covariance_blocks) == 2
    assert report.audit is not None
    assert report.audit.screening.factor_covariance_verified is True
    assert report.audit.screening.numerically_valid


def test_dense_observability_profiles_specimen_parameters_but_fails_open_nuisances(
    tmp_path,
):
    model, result, probes, starts, kernel = _problem()
    report = estimate_lattice_site_observability_1d(
        model,
        result,
        probes,
        starts,
        4,
        kernel,
        0.4,
        ENERGY,
        PoissonCountingModel1D(
            electrons_per_pattern=1e6,
            calibrated=True,
            calibration_id="synthetic-dose",
        ),
        options=SiteObservabilityOptions1D(
            dense_max_parameters=32,
            rematerialize=False,
        ),
    )
    assert report.fit.solver_verified
    assert report.audit is not None and report.audit.solver_verified
    assert report.fit.metadata["n_parameters"] == 6
    assert report.ideal_poisson_information
    assert report.calibrated_noise
    assert not report.nuisance_scope_complete
    assert not report.suitable_for_trust_gate
    assert report.metadata["active_site_translation_scope"] == (
        "variable_sites_relative_to_fixed_reference"
    )

    path = tmp_path / "observability.npz"
    save_lattice_site_observability_1d(path, report)
    with np.load(path, allow_pickle=False) as data:
        assert all(array.dtype != object for array in data.values())
    loaded = load_lattice_site_observability_1d(path)
    np.testing.assert_allclose(loaded.site_coordinates, report.site_coordinates)
    np.testing.assert_allclose(
        loaded.fit.vacancy_standard_error,
        report.fit.vacancy_standard_error,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        loaded.audit.displacement_covariance_A2,
        report.audit.displacement_covariance_A2,
        equal_nan=True,
    )
    np.testing.assert_array_equal(loaded.site_observable, report.site_observable)
    assert loaded.fit.metadata == report.fit.metadata
    assert loaded.audit.metadata == report.audit.metadata
    assert loaded.metadata["loaded_archive_fail_closed"] is True
    for key, value in report.metadata.items():
        assert loaded.metadata[key] == value


def test_one_site_observability_profiles_vacancy_and_translation_only():
    model, result, probes, starts, kernel = _problem()
    one_site_model = replace(
        model,
        site_coordinates=model.site_coordinates[:1],
        site_patches=model.site_patches[:1],
        patch_starts=model.patch_starts[:1],
    )
    one_site_result = replace(
        result,
        vacancy_fractions=result.vacancy_fractions[:1],
        initial_vacancy_fractions=result.initial_vacancy_fractions[:1],
        site_coordinates=result.site_coordinates[:1],
        displaced_site_coordinates=result.displaced_site_coordinates[:1],
    )
    report = estimate_lattice_site_observability_1d(
        one_site_model,
        one_site_result,
        probes,
        starts,
        4,
        kernel,
        0.4,
        ENERGY,
        PoissonCountingModel1D(electrons_per_pattern=1e5),
        options=SiteObservabilityOptions1D(
            dense_max_parameters=8,
            rematerialize=False,
        ),
    )
    assert report.fit.metadata["n_parameters"] == 3
    assert report.metadata["displacement_basis_rank"] == 0


def test_observability_pcg_converges_for_spd_and_reports_singular_breakdown():
    matrix = np.asarray([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.asarray([1.0, 2.0])
    solved = pcg_solve_observability_1d(
        lambda value: matrix @ value,
        rhs,
        relative_residual_tolerance=1e-12,
    )
    assert solved.converged
    assert solved.stop_reason == "converged"
    np.testing.assert_allclose(solved.solution, np.linalg.solve(matrix, rhs))
    np.testing.assert_allclose(
        solved.residual_norm,
        np.linalg.norm(rhs - matrix @ np.asarray(solved.solution)),
    )
    assert solved.residual_norm_history[0] == pytest.approx(np.linalg.norm(rhs))

    singular = np.diag([1.0, 0.0])
    failed = pcg_solve_observability_1d(
        lambda value: singular @ value,
        np.asarray([0.0, 1.0]),
    )
    assert not failed.converged
    assert failed.breakdown
    assert failed.stop_reason == "zero_curvature_breakdown"
    assert failed.relative_residual == pytest.approx(1.0)


def _matrix_free_options(**changes):
    values = {
        "scan_batch_size": 1,
        "maximum_iterations": 32,
        "relative_residual_tolerance": 2e-9,
        "stagnation_iterations": 8,
        "operator_check_vectors": 1,
        "exhaustive": True,
        "exhaustive_max_parameters": 16,
        "exhaustive_relative_tolerance": 2e-6,
        "maximum_selected_sites": 2,
    }
    values.update(changes)
    return MatrixFreeObservabilityOptions1D(**values)


def test_prepared_poisson_counting_contract_is_derived_and_fail_closed(
    prepared_poisson_observability_problem,
):
    prepared, reconstruction = prepared_poisson_observability_problem
    canonical = poisson_counting_model_from_prepared_1d(prepared)
    assert canonical == PoissonCountingModel1D(
        electrons_per_pattern=750_000.0,
        background_electrons_per_pixel=0.25,
        minimum_expected_electrons=1e-7,
        calibrated=True,
        calibration_id="prepared-poisson-observability-v1",
    )

    report = estimate_prepared_lattice_site_observability_matrix_free_1d(
        prepared,
        reconstruction,
        site_indices=[0],
        options=_matrix_free_options(),
    )
    assert report.ideal_poisson_information
    assert not report.calibrated_noise
    assert report.metadata["prepared_counting_contract_bound"] is True
    assert report.metadata["counting_contract_scope"] == (
        "derived_from_prepared_poisson_objective"
    )
    assert report.metadata["declared_counting_model_calibrated"] is True
    assert report.metadata["typed_calibration_evidence_supplied"] is False

    matching = estimate_prepared_lattice_site_observability_matrix_free_1d(
        prepared,
        reconstruction,
        canonical,
        site_indices=[0],
        options=_matrix_free_options(),
    )
    assert matching.metadata["counting_contract_scope"] == (
        "caller_verified_against_prepared_poisson_objective"
    )
    assert matching.metadata["counting_calibration_sha256"] != (
        report.metadata["counting_calibration_sha256"]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("electrons_per_pattern", 750_001.0),
        ("background_electrons_per_pixel", 0.251),
        ("minimum_expected_electrons", 2e-7),
        ("calibrated", False),
        ("calibration_id", "different-calibration"),
    ],
)
def test_prepared_poisson_rejects_every_conflicting_count_field(
    prepared_poisson_observability_problem,
    field,
    value,
):
    prepared, reconstruction = prepared_poisson_observability_problem
    canonical = poisson_counting_model_from_prepared_1d(prepared)
    conflicting = replace(canonical, **{field: value})
    with pytest.raises(ValueError, match=field):
        estimate_prepared_lattice_site_observability_matrix_free_1d(
            prepared,
            reconstruction,
            conflicting,
            site_indices=[0],
            options=_matrix_free_options(),
        )


def test_prepared_poisson_contract_rejects_unrepresentable_objectives(
    prepared_poisson_observability_problem,
):
    prepared, _ = prepared_poisson_observability_problem
    assert prepared.objective is not None and prepared.measurement is not None

    gaussian = replace(
        prepared,
        objective=replace(prepared.objective, kind="poisson_gaussian_nll"),
    )
    with pytest.raises(ValueError, match="Gaussian/read-noise"):
        observability._canonical_prepared_poisson_counting_model_1d(gaussian)

    varying_dose = replace(
        prepared,
        objective=replace(
            prepared.objective,
            electrons_per_pattern=jnp.asarray(
                [1_000_000.0, 1_000_001.0, 1_000_000.0, 1_000_000.0]
            ),
        ),
    )
    with pytest.raises(ValueError, match="nonconstant per-scan effective dose"):
        observability._canonical_prepared_poisson_counting_model_1d(varying_dose)

    dark = np.asarray(
        prepared.measurement.calibrated_dark_electrons_per_pixel
    ).copy()
    valid = np.asarray(prepared.detector_valid_mask)
    changed_pixel = tuple(np.argwhere(valid)[0])
    dark[changed_pixel] += 0.01
    varying_dark = replace(
        prepared,
        measurement=replace(
            prepared.measurement,
            calibrated_dark_electrons_per_pixel=dark,
        ),
    )
    with pytest.raises(ValueError, match="nonconstant valid-pixel calibrated dark"):
        observability._canonical_prepared_poisson_counting_model_1d(varying_dark)

    read_noise = np.zeros_like(dark)
    read_noise[changed_pixel] = 0.01
    noisy = replace(
        prepared,
        measurement=replace(
            prepared.measurement,
            calibrated_read_noise_std_electrons=read_noise,
        ),
    )
    with pytest.raises(ValueError, match="exactly zero declared read noise"):
        observability._canonical_prepared_poisson_counting_model_1d(noisy)

    invalid = tuple(np.argwhere(~valid)[0])
    dark[changed_pixel] = 0.25
    dark[invalid] = 1e200
    invalid_only = replace(
        prepared,
        measurement=replace(
            prepared.measurement,
            calibrated_dark_electrons_per_pixel=dark,
        ),
    )
    assert (
        observability._canonical_prepared_poisson_counting_model_1d(invalid_only)
        == poisson_counting_model_from_prepared_1d(prepared)
    )


def test_matrix_free_rejects_reconstruction_renderer_state_mismatch(
    prepared_observability_problem,
):
    prepared, reconstruction = prepared_observability_problem
    counting = PoissonCountingModel1D(electrons_per_pattern=1e5)

    with pytest.raises(ValueError, match="external hypothetical counting_model"):
        estimate_prepared_lattice_site_observability_matrix_free_1d(
            prepared,
            reconstruction,
            site_indices=[0],
            options=_matrix_free_options(),
        )

    changed_potential = np.asarray(reconstruction.potential).copy()
    changed_potential[0, 0] += 0.01
    with pytest.raises(ValueError, match="potential is inconsistent"):
        estimate_prepared_lattice_site_observability_matrix_free_1d(
            prepared,
            replace(reconstruction, potential=changed_potential),
            counting,
            site_indices=[0],
            options=_matrix_free_options(),
        )

    changed_coordinates = np.asarray(
        reconstruction.displaced_site_coordinates
    ).copy()
    changed_coordinates[0, 0] += 0.01
    with pytest.raises(ValueError, match="displaced_site_coordinates"):
        estimate_prepared_lattice_site_observability_matrix_free_1d(
            prepared,
            replace(
                reconstruction,
                displaced_site_coordinates=changed_coordinates,
            ),
            counting,
            site_indices=[0],
            options=_matrix_free_options(),
        )

    changed_controls = np.asarray(reconstruction.displacement_controls).copy()
    changed_controls[0, 0, 1] += 0.01
    with pytest.raises(ValueError, match="displaced_site_coordinates"):
        estimate_prepared_lattice_site_observability_matrix_free_1d(
            prepared,
            replace(reconstruction, displacement_controls=changed_controls),
            counting,
            site_indices=[0],
            options=_matrix_free_options(),
        )


def test_matrix_free_projected_fisher_matches_dense_exhaustive_oracle(
    prepared_observability_problem,
):
    prepared, reconstruction = prepared_observability_problem
    counting = PoissonCountingModel1D(
        electrons_per_pattern=1e6,
        calibrated=True,
        calibration_id="synthetic-counts-v1",
    )
    report = estimate_prepared_lattice_site_observability_matrix_free_1d(
        prepared,
        reconstruction,
        counting,
        site_indices=[0],
        options=_matrix_free_options(),
    )
    dense = estimate_lattice_site_observability_1d(
        prepared.model,
        reconstruction,
        prepared.input_probe,
        prepared.window_starts,
        prepared.window_length,
        prepared.propagation_kernel,
        prepared.slice_thickness,
        prepared.energy,
        counting,
        detector_mask=prepared.detector_valid_mask,
        options=SiteObservabilityOptions1D(
            dense_max_parameters=16,
            rematerialize=prepared.rematerialize,
        ),
    )
    for matrix_free_split, dense_split in (
        (report.fit, dense.fit),
        (report.audit, dense.audit),
    ):
        exhaustive = matrix_free_split.metadata["exhaustive"]
        assert exhaustive["passed"]
        assert exhaustive["physical_estimability_mismatch_count"] == 0
        assert exhaustive["operator_relative_error"] < 2e-6
        np.testing.assert_allclose(
            matrix_free_split.vacancy_standard_error[0],
            dense_split.vacancy_standard_error[0],
            rtol=3e-5,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            matrix_free_split.displacement_covariance_A2[0],
            dense_split.displacement_covariance_A2[0],
            rtol=3e-5,
            atol=1e-9,
        )
        assert matrix_free_split.metadata["operator_checks_passed"]
        assert matrix_free_split.metadata["projector_checks_passed"]
    assert not report.nuisance_scope_complete
    assert not report.suitable_for_trust_gate
    assert not report.calibrated_noise
    assert report.metadata["counting_contract_scope"] == (
        "external_hypothetical_legacy_amplitude"
    )
    assert report.metadata["prepared_counting_contract_bound"] is False
    assert report.metadata["declared_counting_model_calibrated"] is True
    assert report.metadata["reconstruction_problem_id"] == (
        prepared.reconstruction_problem_id
    )
    assert report.metadata["reconstructor_id"] == prepared.reconstructor_id
    for key in (
        "reconstruction_state_sha256",
        "detector_mask_sha256",
        "counting_calibration_sha256",
        "nuisance_profile_sha256",
    ):
        assert len(report.metadata[key]) == 64


def test_matrix_free_nuisance_projection_is_exact_and_provenance_bound(
    prepared_observability_problem,
):
    prepared, reconstruction = prepared_observability_problem
    tangent = np.ones((*prepared.measured_intensities.shape, 2), dtype=float)
    tangent[..., 1] = 2.0
    profile = WhitenedNuisanceProfile1D(
        tangent_matrix=tangent,
        parameter_names=("gain", "duplicate_gain"),
        profile_id="deliberately-rank-deficient-gain-v1",
        metadata={"calibration": "unit-test", "version": 1},
    )
    report = estimate_prepared_lattice_site_observability_matrix_free_1d(
        prepared,
        reconstruction,
        PoissonCountingModel1D(electrons_per_pattern=1e6),
        nuisance_profile=profile,
        site_indices=[0],
        options=_matrix_free_options(),
    )
    for split in (report.fit, report.audit):
        assert split.metadata["nuisance_rank"] == 1
        assert split.metadata["projector_checks_passed"]
        assert split.metadata["exhaustive"]["passed"]
    assert report.metadata["nuisance_profile_id"] == profile.profile_id
    assert report.metadata["nuisance_profile_metadata"] == profile.metadata

    changed_profile = replace(profile, metadata={"calibration": "changed"})
    changed = estimate_prepared_lattice_site_observability_matrix_free_1d(
        prepared,
        reconstruction,
        PoissonCountingModel1D(electrons_per_pattern=1e6),
        nuisance_profile=changed_profile,
        site_indices=[0],
        options=_matrix_free_options(),
    )
    assert changed.metadata["nuisance_profile_sha256"] != (
        report.metadata["nuisance_profile_sha256"]
    )


def test_matrix_free_bindings_caps_failures_and_reload_are_fail_closed(
    prepared_observability_problem,
    tmp_path,
):
    prepared, reconstruction = prepared_observability_problem
    counting = PoissonCountingModel1D(electrons_per_pattern=1e5)
    with pytest.raises(ValueError, match="maximum_selected_sites"):
        estimate_prepared_lattice_site_observability_matrix_free_1d(
            prepared,
            reconstruction,
            counting,
            options=_matrix_free_options(maximum_selected_sites=1),
        )
    changed_metadata = {
        **dict(reconstruction.metadata),
        "reconstruction_problem_id": "0" * 64,
    }
    with pytest.raises(ValueError, match="reconstruction_problem_id"):
        estimate_prepared_lattice_site_observability_matrix_free_1d(
            prepared,
            replace(reconstruction, metadata=changed_metadata),
            counting,
            site_indices=[0],
            options=_matrix_free_options(),
        )
    wrong_mask = np.asarray(prepared.detector_valid_mask).copy()
    wrong_mask[0, 1] = True
    with pytest.raises(ValueError, match="detector_valid_mask"):
        estimate_prepared_lattice_site_observability_matrix_free_1d(
            prepared,
            replace(reconstruction, detector_valid_mask=wrong_mask),
            counting,
            site_indices=[0],
            options=_matrix_free_options(),
        )

    forced_failure = estimate_prepared_lattice_site_observability_matrix_free_1d(
        prepared,
        reconstruction,
        counting,
        site_indices=[0],
        options=_matrix_free_options(
            maximum_iterations=1,
            relative_residual_tolerance=1e-12,
            exhaustive=False,
        ),
    )
    assert not forced_failure.fit.solver_verified
    assert not all(forced_failure.fit.metadata["pcg"]["converged"])

    fabricated = replace(
        forced_failure,
        calibrated_noise=True,
        nuisance_scope_complete=True,
        suitable_for_trust_gate=True,
    )
    path = tmp_path / "matrix_free_observability.npz"
    save_lattice_site_observability_1d(path, fabricated)
    loaded = load_lattice_site_observability_1d(path)
    assert not loaded.calibrated_noise
    assert not loaded.nuisance_scope_complete
    assert not loaded.suitable_for_trust_gate
    assert loaded.metadata["loaded_archive_fail_closed"] is True


def test_default_jax_precision_keeps_interaction_and_scan_finite():
    script = textwrap.dedent(
        """
        import json
        import jax
        import jax.numpy as jnp
        import numpy as np

        from wide_angle_propagation.propagation_methods import (
            fresnel_propagation_kernel_1d,
            interaction_constant,
        )
        from wide_angle_propagation.ptychography_1d import simulate_glancing_scan_1d

        assert not jax.config.jax_enable_x64
        energy = 30e3
        n_detector = 8
        probe = jnp.exp(-0.5 * ((jnp.arange(n_detector) - 3.5) / 1.5) ** 2)
        kernel = fresnel_propagation_kernel_1d(n_detector, 0.3, 0.4, energy)
        intensity = simulate_glancing_scan_1d(
            jnp.zeros((3, n_detector), dtype=jnp.float32),
            probe,
            jnp.asarray([0], dtype=jnp.int32),
            3,
            kernel,
            0.4,
            energy,
        )
        sigma = interaction_constant(energy)
        parseval_ratio = (
            jnp.sum(intensity) / (n_detector * jnp.sum(jnp.abs(probe) ** 2))
        )
        print(json.dumps({
            "sigma": float(sigma),
            "sigma_finite": bool(jnp.isfinite(sigma)),
            "intensity_finite": bool(jnp.all(jnp.isfinite(intensity))),
            "parseval_ratio": float(parseval_ratio),
        }))
        """
    )
    environment = dict(os.environ)
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["sigma_finite"]
    assert payload["intensity_finite"]
    assert payload["sigma"] == pytest.approx(0.001543269916, rel=2e-6)
    assert payload["parseval_ratio"] == pytest.approx(1.0, rel=2e-6)
