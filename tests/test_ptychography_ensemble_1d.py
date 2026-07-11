"""Truth-free consensus tests for lattice-site multistart reconstructions."""

from dataclasses import replace

import numpy as np
import pytest


pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from wide_angle_propagation.ptychography_benchmarks_1d import (  # noqa: E402
    BenchmarkCriteria1D,
    BenchmarkCriterion1D,
    DetectorPerturbation1D,
    ForwardModelInputs1D,
    ForwardModelMismatch1D,
    ReconstructionBenchmarkOutput1D,
    SyntheticBenchmarkScenario1D,
    evaluate_residual_calibration_evidence_1d,
    generate_detector_measurement_1d,
    run_synthetic_benchmark_sweep_1d,
)
from wide_angle_propagation.propagation_methods import (  # noqa: E402
    fresnel_propagation_kernel_1d,
)
from wide_angle_propagation.ptychography_1d import (  # noqa: E402
    LatticeSiteModel1D,
    LatticeSiteReconstruction1D,
    prepare_lattice_site_reconstruction_1d,
    render_lattice_site_potential_1d,
    simulate_glancing_scan_1d,
)
from wide_angle_propagation.ptychography_diagnostics_1d import (  # noqa: E402
    LatticeSiteSensitivityScreen1D,
)
from wide_angle_propagation.ptychography_observability_1d import (  # noqa: E402
    LatticeSiteObservability1D,
    SiteObservabilitySplit1D,
)
from wide_angle_propagation.ptychography_ensemble_1d import (  # noqa: E402
    MultistartOptions1D,
    PreparedMultistartResult1D,
    PreparedMultistartRunOptions1D,
    load_lattice_site_ensemble_1d,
    multistart_site_translation_offsets_1d,
    run_prepared_lattice_site_multistart_1d,
    save_lattice_site_ensemble_1d,
    summarize_lattice_site_ensemble_1d,
)


_PROBLEM_ID = "ensemble-test-reconstruction-problem-v1"
_RECONSTRUCTOR_ID = "ensemble-test-reconstructor-v1"
_GENERATOR_ID = "independent-ensemble-test-generator-v1"


@pytest.fixture(scope="module")
def prepared_multistart_problems():
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    energy = 30e3
    shape = (8, 12)
    patch = np.asarray(
        [
            [0.0, 0.2, 0.0],
            [0.3, 2.0, 0.3],
            [0.0, 0.2, 0.0],
        ],
        dtype=np.float32,
    )
    patch_starts = np.asarray([[2, 3], [4, 7]], dtype=np.int32)
    reference = np.full(shape, 0.05, dtype=np.float32)
    for start in patch_starts:
        reference[
            start[0] : start[0] + patch.shape[0],
            start[1] : start[1] + patch.shape[1],
        ] += patch
    sites = np.column_stack(
        [
            (patch_starts[:, 0] + 1) * 0.4,
            (patch_starts[:, 1] + 1) * 0.3,
        ]
    ).astype(np.float32)
    model = LatticeSiteModel1D(
        reference_potential=jnp.asarray(reference),
        site_coordinates=jnp.asarray(sites),
        site_patches=jnp.asarray(np.stack([patch, patch])),
        patch_starts=jnp.asarray(patch_starts),
        control_coordinates_s=jnp.asarray([0.0, (shape[0] - 1) * 0.4]),
        control_coordinates_u=jnp.asarray([0.0, (shape[1] - 1) * 0.3]),
        axial_sampling=0.4,
        transverse_sampling=0.3,
        maximum_displacement=0.2,
        metadata={"species": "Si"},
    )
    u = (jnp.arange(shape[1]) - shape[1] // 2) * 0.3
    base_probe = jnp.exp(-0.5 * ((u + 0.1) / 0.65) ** 2) * jnp.exp(0.25j * u)
    probes = jnp.stack([jnp.roll(base_probe, index - 2) for index in range(5)])
    starts = jnp.arange(5)
    kernel = fresnel_propagation_kernel_1d(shape[1], 0.3, 0.4, energy)
    target = render_lattice_site_potential_1d(
        model,
        jnp.asarray([0.65, 0.0]),
        jnp.zeros((2, 2, 2)),
    )
    measured = simulate_glancing_scan_1d(
        target, probes, starts, 4, kernel, 0.4, energy
    )
    common = dict(
        model=model,
        input_probe=probes,
        window_starts=starts,
        window_length=4,
        propagation_kernel=kernel,
        slice_thickness=0.4,
        energy=energy,
        measured_intensities=measured,
        validation_indices=[3],
        audit_indices=[4],
        potential_max=10.0,
        minibatch_size=2,
        evaluation_batch_size=3,
        rematerialize=False,
    )
    legacy = prepare_lattice_site_reconstruction_1d(**common)
    separate = prepare_lattice_site_reconstruction_1d(
        **common,
        separate_rigid_registration=True,
        maximum_rigid_displacement=0.08,
        maximum_residual_displacement=0.1,
    )
    return {"legacy": legacy, "separate": separate}


def _prepared_runner_options(*, base_seed=7, initial_vacancies=None):
    return PreparedMultistartRunOptions1D(
        ensemble_options=MultistartOptions1D(
            n_starts=3,
            base_seed=base_seed,
            initial_translation_half_width_A=(0.025, 0.015),
            relative_loss_tolerance=1e6,
            absolute_loss_tolerance=1e-5,
            minimum_accepted_starts=1,
            minimum_accepted_fraction=1 / 3,
        ),
        initial_vacancy_fractions=initial_vacancies,
        learning_rate_start=0.03,
        learning_rate_end=0.01,
        updates=2,
        validation_interval=1,
        representative_checkpoint_interval=1,
    )


def _assert_run_numerics_equal(first, second):
    for name in (
        "potential",
        "initial_potential",
        "vacancy_fractions",
        "initial_vacancy_fractions",
        "displacement_controls",
        "initial_displacement_controls",
        "rigid_displacement",
        "initial_rigid_displacement",
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
    assert first.converged == second.converged
    assert first.stop_reason == second.stop_reason
    assert first.audit_loss == second.audit_loss


def _result(
    vacancies,
    residual,
    rigid,
    *,
    loss,
    converged=True,
    bound_fraction=0.0,
    seed=None,
    audit_loss=1.0,
    reconstruction_problem_id=_PROBLEM_ID,
    reconstructor_id=_RECONSTRUCTOR_ID,
    material_scope_complete=False,
    material_scope_fully_parameterized=None,
):
    vacancies = np.asarray(vacancies, dtype=float)
    residual = np.asarray(residual, dtype=float)
    rigid = np.asarray(rigid, dtype=float)
    sites = np.stack(
        [np.arange(len(vacancies), dtype=float), np.zeros(len(vacancies))],
        axis=1,
    )
    controls = np.zeros((2, 2, 2), dtype=float)
    fully_parameterized = (
        bool(material_scope_complete)
        if material_scope_fully_parameterized is None
        else bool(material_scope_fully_parameterized)
    )
    return LatticeSiteReconstruction1D(
        potential=np.zeros((2, 2)),
        initial_potential=np.zeros((2, 2)),
        vacancy_fractions=vacancies,
        initial_vacancy_fractions=np.zeros_like(vacancies),
        displacement_controls=controls,
        initial_displacement_controls=controls,
        site_coordinates=sites,
        displaced_site_coordinates=sites + residual + rigid,
        control_coordinates_s=np.asarray([0.0, 1.0]),
        control_coordinates_u=np.asarray([0.0, 1.0]),
        predicted_intensities=np.zeros((3, 2)),
        measured_intensities=np.zeros((3, 2)),
        window_starts=np.asarray([0, 0, 0]),
        scan_coordinates=np.asarray([0.0, 1.0, 2.0]),
        detector_angles=np.asarray([0.0, 1.0]),
        update_history=np.asarray([0, 1]),
        elapsed_time_history=np.asarray([0.0, 1.0]),
        training_loss_history=np.asarray([loss, loss]),
        validation_loss_history=np.asarray([loss, loss]),
        best_update=1,
        completed_updates=1,
        converged=converged,
        stop_reason="plateau" if converged else "maximum_updates",
        audit_loss=audit_loss,
        rigid_displacement=rigid,
        metadata={
            "best_metric": float(loss),
            "audit_metric": float(audit_loss),
            "training_indices": [0],
            "validation_indices": [2],
            "audit_indices": [1],
            "excluded_indices": [],
            "reconstruction_problem_id": reconstruction_problem_id,
            "reconstructor_id": reconstructor_id,
            "best_total_displacement_bound_fraction": float(bound_fraction),
            "material_scope_fully_parameterized": fully_parameterized,
            **({} if seed is None else {"seed": int(seed)}),
        },
        site_role_codes=(
            np.full(len(vacancies), 1, dtype=np.int8)
            if material_scope_complete
            else np.empty(0, dtype=np.int8)
        ),
        support_contract_id=("a" * 64 if material_scope_complete else None),
        material_scope_complete=material_scope_complete,
        material_scope_fully_parameterized=fully_parameterized,
    )


def _sensitivity(mask):
    mask = np.asarray(mask, dtype=bool)
    n_site = len(mask)
    sites = np.stack(
        [np.arange(n_site, dtype=float), np.zeros(n_site)], axis=1
    )
    return LatticeSiteSensitivityScreen1D(
        site_coordinates=sites,
        fisher_blocks=np.zeros((n_site, 3, 3)),
        fisher_diagonal_relative_error=np.zeros((n_site, 3)),
        vacancy_standard_error_lower_bound=np.ones(n_site),
        displacement_standard_error_lower_bound_A=np.ones((n_site, 2)),
        vacancy_sensitive=mask,
        displacement_sensitive=np.broadcast_to(mask[:, None], (n_site, 2)),
        displacement_applicable=np.ones(n_site, dtype=bool),
        site_sensitive=mask,
        scan_indices=np.asarray([0]),
    )


def _observability(mask, *, reconstruction_problem_id=_PROBLEM_ID):
    mask = np.asarray(mask, dtype=bool)
    n_site = len(mask)
    sites = np.stack(
        [np.arange(n_site, dtype=float), np.zeros(n_site)], axis=1
    )

    def split(indices):
        return SiteObservabilitySplit1D(
            scan_indices=np.asarray(indices),
            vacancy_standard_error=np.full(n_site, 0.01),
            vacancy_z_to_decision_boundary=np.full(n_site, 10.0),
            displacement_covariance_A2=np.broadcast_to(
                np.eye(2)[None] * 1e-4, (n_site, 2, 2)
            ),
            displacement_confidence_radius_A=np.full(n_site, 0.02),
            vacancy_information_adequate=mask,
            displacement_information_adequate=mask,
            site_observable=mask,
            physical_output_estimable=np.ones(3 * n_site, dtype=bool),
            solver_verified=True,
            effective_rank=3 * n_site,
        )
    return LatticeSiteObservability1D(
        site_coordinates=sites,
        fit=split([0]),
        audit=split([1]),
        vacancy_information_adequate=mask,
        displacement_information_adequate=mask,
        site_observable=mask,
        ideal_poisson_information=True,
        calibrated_noise=True,
        nuisance_scope_complete=True,
        suitable_for_trust_gate=True,
        metadata={"reconstruction_problem_id": reconstruction_problem_id},
    )


def _residual_evidence(
    *,
    reconstruction_problem_id=_PROBLEM_ID,
    held_out_scan_indices=(1,),
    upper_bound=100.0,
):
    expected = np.full((len(held_out_scan_indices), 4), 40.0)
    measurement = generate_detector_measurement_1d(
        expected,
        DetectorPerturbation1D(calibration_id="ensemble-held-out-detector-v1"),
        seed=21,
    )
    criteria = BenchmarkCriteria1D(
        criteria_id="ensemble-held-out-residual-policy-v1",
        criteria=(
            BenchmarkCriterion1D(
                criterion_id="ensemble-held-out-residual-bias",
                metric_name="residual.standardized_mean_abs",
                threshold_source="test:ensemble-held-out-residual-policy-v1",
                upper_bound=upper_bound,
            ),
        ),
    )
    return evaluate_residual_calibration_evidence_1d(
        measurement,
        expected,
        criteria=criteria,
        held_out_scan_indices=held_out_scan_indices,
        reconstruction_problem_id=reconstruction_problem_id,
    )


def _mismatch_report(
    *,
    reconstructor_id=_RECONSTRUCTOR_ID,
    generator_id=_GENERATOR_ID,
    non_nominal=True,
    include_truth_criterion=True,
    structural_estimate=0.0,
):
    nominal = ForwardModelInputs1D(
        probe=np.asarray([0.0, 1.0, 0.0], dtype=np.complex128),
        probe_sampling_A=1.0,
        scan_coordinates_A=np.asarray([-1.0, 0.0, 1.0]),
        detector_angles_rad=np.asarray([-0.01, 0.01]),
        energy_eV=30_000.0,
    )

    def expected_signal(inputs):
        scans = np.asarray(inputs.scan_coordinates_A)[:, None]
        angles = np.asarray(inputs.detector_angles_rad)[None, :]
        return 40.0 + 0.1 * scans + angles

    def reconstruct(measurement, _inputs):
        return ReconstructionBenchmarkOutput1D(
            predicted_signal_electrons=np.maximum(
                measurement.calibrated_signal_electrons, 0.0
            ),
            estimated_parameters={
                "structure": np.asarray([structural_estimate], dtype=float)
            },
            metadata={"reconstructor_id": reconstructor_id},
        )

    criterion = (
        BenchmarkCriterion1D(
            criterion_id="ensemble-structural-accuracy",
            metric_name="truth.structure.rmse",
            threshold_source="test:ensemble-mismatch-policy-v1",
            upper_bound=0.1,
        )
        if include_truth_criterion
        else BenchmarkCriterion1D(
            criterion_id="ensemble-residual-bias",
            metric_name="residual.standardized_mean_abs",
            threshold_source="test:ensemble-mismatch-policy-v1",
            upper_bound=100.0,
        )
    )
    detector = (
        DetectorPerturbation1D(
            read_noise_std_electrons=0.25,
            calibrated_read_noise_std_electrons=0.25,
            calibration_id="ensemble-non-nominal-detector-v1",
        )
        if non_nominal
        else DetectorPerturbation1D()
    )
    scenario = SyntheticBenchmarkScenario1D(
        scenario_id="ensemble-mismatch-scenario-v1",
        seed=9,
        detector=detector,
        forward_mismatch=(
            ForwardModelMismatch1D(probe_amplitude_scale=1.01)
            if non_nominal
            else ForwardModelMismatch1D()
        ),
    )
    return run_synthetic_benchmark_sweep_1d(
        nominal,
        {"structure": np.asarray([0.0])},
        (scenario,),
        expected_signal,
        reconstruct,
        criteria=BenchmarkCriteria1D(
            criteria_id="ensemble-mismatch-policy-v1",
            criteria=(criterion,),
        ),
        benchmark_id="ensemble-mismatch-benchmark-v1",
        truth_id="ensemble-structural-truth-v1",
        generator_id=generator_id,
        reconstructor_id=reconstructor_id,
    )


def test_ensemble_retains_divergent_low_loss_basin_and_marks_ambiguity():
    residual = np.zeros((2, 2))
    results = [
        _result([0.9, 0.0], residual, [0.0, 0.0], loss=1.0),
        _result([0.0, 0.9], residual, [0.0, 0.0], loss=1.02),
        _result([0.9, 0.0], residual, [0.0, 0.0], loss=2.0),
    ]
    ensemble = summarize_lattice_site_ensemble_1d(
        results,
        options=MultistartOptions1D(
            n_starts=3,
            relative_loss_tolerance=0.05,
            minimum_accepted_starts=2,
            minimum_accepted_fraction=2 / 3,
        ),
    )
    np.testing.assert_array_equal(ensemble.accepted_mask, [True, True, False])
    np.testing.assert_allclose(
        ensemble.consensus.vacancy_call_frequency, [0.5, 0.5]
    )
    np.testing.assert_array_equal(ensemble.consensus.vacancy_state, [-1, -1])
    assert ensemble.representative_index in {0, 1}
    assert ensemble.trust_flags["dominant_low_loss_basin"] is False
    assert ensemble.optimizer_stable is False
    assert ensemble.structurally_trusted is False
    assert not np.any(ensemble.consensus.site_trusted)


def test_multistart_offsets_are_zero_first_deterministic_and_antithetic():
    options = MultistartOptions1D(
        n_starts=6,
        base_seed=4,
        initial_translation_half_width_A=(0.1, 0.2),
    )
    first = multistart_site_translation_offsets_1d(options)
    second = multistart_site_translation_offsets_1d(options)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[0], 0.0)
    np.testing.assert_allclose(first[1], -first[2])
    np.testing.assert_allclose(first[3], -first[4])
    assert np.all(np.abs(first) <= (0.1, 0.2))


def test_prepared_multistart_reuses_checkpointed_medoid_trajectory(
    prepared_multistart_problems,
):
    options = _prepared_runner_options()
    for mode in ("legacy", "separate"):
        result = run_prepared_lattice_site_multistart_1d(
            prepared_multistart_problems[mode],
            options=options,
        )
        assert isinstance(result, PreparedMultistartResult1D)
        assert len(result.screening_results) == 3
        np.testing.assert_array_equal(result.seeds, [7, 8, 9])
        np.testing.assert_array_equal(result.initial_site_translations_A[0], 0.0)
        np.testing.assert_allclose(
            result.initial_site_translations_A[1],
            -result.initial_site_translations_A[2],
        )
        representative_index = result.ensemble.representative_index
        assert (
            result.screening_results[representative_index]
            is result.representative_result
        )
        np.testing.assert_array_equal(
            result.representative_result.checkpoint_updates, [0, 1, 2]
        )
        for index, run in enumerate(result.screening_results):
            if index == representative_index:
                continue
            assert run.checkpoint_updates.size == 0
            assert (
                run.metadata["checkpoint_history_status"]
                == "discarded_nonrepresentative"
            )
        assert result.representative_trajectory_reused is True
        _assert_run_numerics_equal(
            result.representative_screening_result,
            result.representative_result,
        )
        assert result.registration_scope == "active_sites_relative_to_fixed_reference"
        assert (
            result.representative_result.metadata["registration_scope"]
            == result.registration_scope
        )
        assert (
            result.representative_result.metadata[
                "registration_is_global_experimental_alignment"
            ]
            is False
        )
        assert result.ensemble.structurally_trusted is False
        assert not np.any(result.ensemble.consensus.site_trusted)

        for start, offset in zip(
            result.screening_results, result.initial_site_translations_A
        ):
            if mode == "separate":
                np.testing.assert_allclose(start.initial_rigid_displacement, offset)
                np.testing.assert_allclose(start.initial_displacement_controls, 0.0)
            else:
                np.testing.assert_allclose(start.initial_rigid_displacement, 0.0)
                np.testing.assert_allclose(
                    start.initial_displacement_controls,
                    np.broadcast_to(offset, (2, 2, 2)),
                )


def test_prepared_multistart_is_deterministic_across_intervening_run_a_b_a(
    prepared_multistart_problems,
):
    prepared = prepared_multistart_problems["legacy"]
    options_a = _prepared_runner_options(base_seed=11)
    options_b = _prepared_runner_options(
        base_seed=29,
        initial_vacancies=np.asarray([0.2, 0.1], dtype=np.float32),
    )

    first_a = run_prepared_lattice_site_multistart_1d(
        prepared, options=options_a
    )
    run_prepared_lattice_site_multistart_1d(prepared, options=options_b)
    second_a = run_prepared_lattice_site_multistart_1d(
        prepared, options=options_a
    )

    np.testing.assert_array_equal(
        first_a.initial_site_translations_A,
        second_a.initial_site_translations_A,
    )
    np.testing.assert_array_equal(first_a.seeds, second_a.seeds)
    np.testing.assert_array_equal(
        first_a.ensemble.accepted_mask, second_a.ensemble.accepted_mask
    )
    assert (
        first_a.ensemble.representative_index
        == second_a.ensemble.representative_index
    )
    for first, second in zip(
        first_a.screening_results, second_a.screening_results
    ):
        _assert_run_numerics_equal(first, second)
    _assert_run_numerics_equal(
        first_a.representative_result, second_a.representative_result
    )


def test_prepared_multistart_audit_data_cannot_change_selection(
    prepared_multistart_problems,
):
    prepared = prepared_multistart_problems["legacy"]
    altered_measurements = np.asarray(prepared.measured_intensities).copy()
    audit_indices = np.asarray(prepared.audit_indices, dtype=int)
    altered_measurements[audit_indices] = 100.0 + 10.0 * altered_measurements[
        audit_indices
    ]
    altered_audit = prepare_lattice_site_reconstruction_1d(
        model=prepared.model,
        input_probe=prepared.input_probe,
        window_starts=prepared.window_starts,
        window_length=prepared.window_length,
        propagation_kernel=prepared.propagation_kernel,
        slice_thickness=prepared.slice_thickness,
        energy=prepared.energy,
        measured_intensities=jnp.asarray(altered_measurements),
        separate_rigid_registration=prepared.separate_rigid_registration,
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
        **(
            {
                "maximum_rigid_displacement": (
                    prepared.maximum_rigid_displacement
                ),
                "maximum_residual_displacement": (
                    prepared.maximum_residual_displacement
                ),
            }
            if prepared.separate_rigid_registration
            else {}
        ),
    )
    options = _prepared_runner_options(base_seed=17)

    original = run_prepared_lattice_site_multistart_1d(
        prepared, options=options
    )
    changed = run_prepared_lattice_site_multistart_1d(
        altered_audit, options=options
    )

    np.testing.assert_array_equal(
        original.ensemble.accepted_mask, changed.ensemble.accepted_mask
    )
    assert (
        original.ensemble.representative_index
        == changed.ensemble.representative_index
    )
    np.testing.assert_array_equal(
        [run.metadata["best_metric"] for run in original.screening_results],
        [run.metadata["best_metric"] for run in changed.screening_results],
    )
    for first, second in zip(
        original.screening_results, changed.screening_results
    ):
        for name in (
            "potential",
            "vacancy_fractions",
            "displacement_controls",
            "rigid_displacement",
        ):
            np.testing.assert_array_equal(
                getattr(first, name), getattr(second, name)
            )
    assert not np.allclose(
        [run.audit_loss for run in original.screening_results],
        [run.audit_loss for run in changed.screening_results],
    )


def test_prepared_multistart_rejects_translations_outside_prepared_bounds(
    prepared_multistart_problems,
):
    legacy_options = replace(
        _prepared_runner_options(),
        ensemble_options=replace(
            _prepared_runner_options().ensemble_options,
            initial_translation_half_width_A=(0.21, 0.0),
        ),
    )
    with pytest.raises(ValueError, match="constant-control bound"):
        run_prepared_lattice_site_multistart_1d(
            prepared_multistart_problems["legacy"],
            options=legacy_options,
        )

    separate_options = replace(
        _prepared_runner_options(),
        ensemble_options=replace(
            _prepared_runner_options().ensemble_options,
            initial_translation_half_width_A=(0.081, 0.0),
        ),
    )
    with pytest.raises(ValueError, match="rigid-registration bound"):
        run_prepared_lattice_site_multistart_1d(
            prepared_multistart_problems["separate"],
            options=separate_options,
        )


def test_prepared_multistart_requires_validation_for_selection(
    prepared_multistart_problems,
):
    without_validation = replace(
        prepared_multistart_problems["legacy"],
        validation_indices=jnp.empty(0, dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="non-empty validation split"):
        run_prepared_lattice_site_multistart_1d(
            without_validation,
            options=_prepared_runner_options(),
        )


def test_local_sensitivity_cannot_replace_marginalized_observability():
    results = [
        _result([0.91, 0.04], [[0.0, 0.0], [0.01, 0.0]], [0.01, 0.0], loss=1.0),
        _result([0.90, 0.05], [[0.0, 0.0], [0.00, 0.0]], [0.00, 0.0], loss=1.0),
        _result([0.89, 0.06], [[0.0, 0.0], [-0.01, 0.0]], [-0.01, 0.0], loss=1.0),
    ]
    ensemble = summarize_lattice_site_ensemble_1d(
        results,
        options=MultistartOptions1D(n_starts=3, minimum_accepted_starts=3),
        sensitivity_screen=_sensitivity([True, True]),
    )
    np.testing.assert_array_equal(ensemble.consensus.vacancy_state, [1, 0])
    assert np.isnan(ensemble.consensus.residual_displacement_median[0]).all()
    assert np.all(ensemble.consensus.sensitive)
    assert not np.any(ensemble.consensus.observable)
    assert not np.any(ensemble.consensus.site_trusted)
    assert ensemble.optimizer_stable is True
    assert ensemble.trust_flags["local_sensitivity_available"] is True
    assert ensemble.trust_flags["observability_available"] is False
    assert ensemble.structurally_trusted is False


def test_site_trust_is_false_when_global_optimizer_checks_fail():
    result = _result(
        [0.92, 0.03],
        [[0.0, 0.0], [0.0, 0.0]],
        [0.0, 0.0],
        loss=1.0,
        converged=False,
        bound_fraction=0.9,
    )
    ensemble = summarize_lattice_site_ensemble_1d(
        [result],
        options=MultistartOptions1D(
            n_starts=1,
            minimum_accepted_starts=1,
            minimum_accepted_fraction=1.0,
        ),
        observability_reports=[_observability([True, True])],
        residual_calibration_evidence=_residual_evidence(),
        mismatch_benchmark_report=_mismatch_report(),
    )

    assert ensemble.optimizer_stable is False
    assert not np.any(ensemble.consensus.site_trusted)
    assert ensemble.structurally_trusted is False


def test_typed_marginalized_observability_can_unlock_trust_gate():
    results = [
        _result(
            [0.91, 0.04],
            [[0.0, 0.0], [0.01, 0.0]],
            [0.01, 0.0],
            loss=1.0,
            material_scope_complete=True,
        ),
        _result(
            [0.90, 0.05],
            [[0.0, 0.0], [0.00, 0.0]],
            [0.00, 0.0],
            loss=1.0,
            material_scope_complete=True,
        ),
        _result(
            [0.89, 0.06],
            [[0.0, 0.0], [-0.01, 0.0]],
            [-0.01, 0.0],
            loss=1.0,
            material_scope_complete=True,
        ),
    ]
    ensemble = summarize_lattice_site_ensemble_1d(
        results,
        options=MultistartOptions1D(n_starts=3, minimum_accepted_starts=3),
        observability_reports=[_observability([True, True]) for _ in results],
        residual_calibration_evidence=_residual_evidence(),
        mismatch_benchmark_report=_mismatch_report(),
    )

    assert np.all(ensemble.consensus.sensitive)
    assert np.all(ensemble.consensus.observable)
    assert np.all(ensemble.consensus.site_trusted)
    assert ensemble.trust_flags["observability_noise_calibrated"] is True
    assert ensemble.trust_flags["observability_nuisance_scope_complete"] is True
    assert ensemble.trust_flags["observability_solver_verified"] is True
    assert ensemble.trust_flags["observability_problem_ids_verified"] is True
    assert (
        ensemble.trust_flags["residual_calibration_evidence_passed"] is True
    )
    assert (
        ensemble.trust_flags["mismatch_benchmark_independent_forward"] is True
    )
    assert ensemble.optimizer_stable is True
    assert ensemble.structurally_trusted is True


def test_naked_trust_booleans_cannot_unlock_the_evidence_gate():
    result = _result(
        [0.91, 0.04],
        [[0.0, 0.0], [0.0, 0.0]],
        [0.0, 0.0],
        loss=1.0,
    )
    with pytest.raises(TypeError, match="ResidualCalibrationEvidence1D"):
        summarize_lattice_site_ensemble_1d(
            [result],
            options=MultistartOptions1D(
                n_starts=1,
                minimum_accepted_starts=1,
                minimum_accepted_fraction=1.0,
            ),
            residual_calibration_evidence=True,
        )
    with pytest.raises(TypeError, match="SyntheticBenchmarkReport1D"):
        summarize_lattice_site_ensemble_1d(
            [result],
            options=MultistartOptions1D(
                n_starts=1,
                minimum_accepted_starts=1,
                minimum_accepted_fraction=1.0,
            ),
            mismatch_benchmark_report=True,
        )


@pytest.mark.parametrize(
    "evidence",
    [
        pytest.param(
            _residual_evidence(held_out_scan_indices=(0,)),
            id="wrong-held-out-indices",
        ),
        pytest.param(
            _residual_evidence(reconstruction_problem_id="other-problem-v1"),
            id="wrong-problem-id",
        ),
    ],
)
def test_residual_evidence_must_match_persisted_audit_and_problem(evidence):
    result = _result(
        [0.91, 0.04],
        [[0.0, 0.0], [0.0, 0.0]],
        [0.0, 0.0],
        loss=1.0,
    )
    with pytest.raises(ValueError, match="held-out indices|reconstruction_problem_id"):
        summarize_lattice_site_ensemble_1d(
            [result],
            options=MultistartOptions1D(
                n_starts=1,
                minimum_accepted_starts=1,
                minimum_accepted_fraction=1.0,
            ),
            residual_calibration_evidence=evidence,
        )


def test_failed_residual_evidence_cannot_unlock_structural_trust():
    result = _result(
        [0.91, 0.04],
        [[0.0, 0.0], [0.0, 0.0]],
        [0.0, 0.0],
        loss=1.0,
    )
    failed_evidence = _residual_evidence(upper_bound=-1.0)
    assert failed_evidence.passed is False
    ensemble = summarize_lattice_site_ensemble_1d(
        [result],
        options=MultistartOptions1D(
            n_starts=1,
            minimum_accepted_starts=1,
            minimum_accepted_fraction=1.0,
        ),
        observability_reports=[_observability([True, True])],
        residual_calibration_evidence=failed_evidence,
        mismatch_benchmark_report=_mismatch_report(),
    )
    assert ensemble.trust_flags["residual_calibration_evidence_passed"] is False
    assert ensemble.structurally_trusted is False
    assert not np.any(ensemble.consensus.site_trusted)


def test_evidence_ids_must_match_every_optimizer_start():
    results = [
        _result(
            [0.91, 0.04],
            [[0.0, 0.0], [0.0, 0.0]],
            [0.0, 0.0],
            loss=1.0,
        ),
        _result(
            [0.90, 0.05],
            [[0.0, 0.0], [0.0, 0.0]],
            [0.0, 0.0],
            loss=1.0,
            reconstructor_id="different-start-reconstructor-v1",
        ),
    ]
    with pytest.raises(ValueError, match="share one metadata 'reconstructor_id'"):
        summarize_lattice_site_ensemble_1d(
            results,
            options=MultistartOptions1D(
                n_starts=2,
                minimum_accepted_starts=2,
                minimum_accepted_fraction=1.0,
            ),
            mismatch_benchmark_report=_mismatch_report(),
        )

    with pytest.raises(ValueError, match="reconstructor_id does not match"):
        summarize_lattice_site_ensemble_1d(
            [results[0]],
            options=MultistartOptions1D(
                n_starts=1,
                minimum_accepted_starts=1,
                minimum_accepted_fraction=1.0,
            ),
            mismatch_benchmark_report=_mismatch_report(
                reconstructor_id="different-report-reconstructor-v1"
            ),
        )


def test_observability_report_must_match_reconstruction_problem_id():
    result = _result(
        [0.91, 0.04],
        [[0.0, 0.0], [0.0, 0.0]],
        [0.0, 0.0],
        loss=1.0,
    )
    with pytest.raises(
        ValueError,
        match="observability report reconstruction_problem_id",
    ):
        summarize_lattice_site_ensemble_1d(
            [result],
            options=MultistartOptions1D(
                n_starts=1,
                minimum_accepted_starts=1,
                minimum_accepted_fraction=1.0,
            ),
            observability_reports=[
                _observability(
                    [True, True],
                    reconstruction_problem_id="different-observability-problem-v1",
                )
            ],
        )


@pytest.mark.parametrize(
    ("report", "flag"),
    [
        pytest.param(
            _mismatch_report(generator_id=_RECONSTRUCTOR_ID),
            "mismatch_benchmark_independent_forward",
            id="shared-generator-and-reconstructor",
        ),
        pytest.param(
            _mismatch_report(non_nominal=False),
            "mismatch_benchmark_non_nominal_scenario_present",
            id="nominal-only-sweep",
        ),
        pytest.param(
            _mismatch_report(include_truth_criterion=False),
            "mismatch_benchmark_truth_structural_criterion_present",
            id="no-structural-truth-criterion",
        ),
        pytest.param(
            _mismatch_report(structural_estimate=1.0),
            "mismatch_benchmark_sourced_criteria_passed",
            id="failed-sourced-criterion",
        ),
    ],
)
def test_mismatch_report_must_be_relevant_and_independent_for_trust(report, flag):
    result = _result(
        [0.91, 0.04],
        [[0.0, 0.0], [0.0, 0.0]],
        [0.0, 0.0],
        loss=1.0,
    )
    ensemble = summarize_lattice_site_ensemble_1d(
        [result],
        options=MultistartOptions1D(
            n_starts=1,
            minimum_accepted_starts=1,
            minimum_accepted_fraction=1.0,
        ),
        observability_reports=[_observability([True, True])],
        residual_calibration_evidence=_residual_evidence(),
        mismatch_benchmark_report=report,
    )
    assert ensemble.trust_flags[flag] is False
    assert ensemble.structurally_trusted is False
    assert not np.any(ensemble.consensus.site_trusted)


def test_ensemble_rejects_mismatched_site_order():
    first = _result([0.9, 0.0], np.zeros((2, 2)), [0.0, 0.0], loss=1.0)
    second = _result([0.9, 0.0], np.zeros((2, 2)), [0.0, 0.0], loss=1.0)
    second = LatticeSiteReconstruction1D(
        **{
            **second.__dict__,
            "site_coordinates": np.asarray(second.site_coordinates)[::-1],
        }
    )
    with pytest.raises(ValueError, match="ordered site coordinates"):
        summarize_lattice_site_ensemble_1d([first, second])


def test_evidence_site_matching_tolerates_dtype_rounding_but_preserves_order():
    result = _result(
        [0.91, 0.04],
        [[0.0, 0.0], [0.01, 0.0]],
        [0.0, 0.0],
        loss=1.0,
    )
    sites = np.asarray(
        [[0.123456789, 0.234567891], [1.345678912, 0.456789123]],
        dtype=np.float64,
    )
    total_displacement = (
        np.asarray(result.displaced_site_coordinates)
        - np.asarray(result.site_coordinates)
    )
    result = replace(
        result,
        site_coordinates=sites,
        displaced_site_coordinates=sites + total_displacement,
    )
    report = replace(
        _observability([True, True]),
        site_coordinates=sites.astype(np.float32),
    )
    screen = replace(
        _sensitivity([True, True]),
        site_coordinates=sites.astype(np.float32),
    )

    ensemble = summarize_lattice_site_ensemble_1d(
        [result],
        options=MultistartOptions1D(
            n_starts=1,
            minimum_accepted_starts=1,
            minimum_accepted_fraction=1.0,
        ),
        sensitivity_screen=screen,
        observability_reports=[report],
    )

    assert ensemble.trust_flags["observability_available"] is True
    assert np.all(ensemble.consensus.sensitive)


def test_held_out_audit_losses_never_select_starts_or_representative():
    residual = np.zeros((2, 2))
    first_results = [
        _result([0.9, 0.0], residual, [0.0, 0.0], loss=1.0, audit_loss=100.0),
        _result([0.0, 0.9], residual, [0.0, 0.0], loss=1.02, audit_loss=1.0),
        _result([0.0, 0.0], residual, [0.0, 0.0], loss=2.0, audit_loss=0.0),
    ]
    second_results = [
        _result([0.9, 0.0], residual, [0.0, 0.0], loss=1.0, audit_loss=0.0),
        _result([0.0, 0.9], residual, [0.0, 0.0], loss=1.02, audit_loss=100.0),
        _result([0.0, 0.0], residual, [0.0, 0.0], loss=2.0, audit_loss=1.0),
    ]
    options = MultistartOptions1D(
        n_starts=3,
        relative_loss_tolerance=0.05,
        minimum_accepted_starts=2,
        minimum_accepted_fraction=2 / 3,
    )
    first = summarize_lattice_site_ensemble_1d(
        first_results,
        options=options,
    )
    second = summarize_lattice_site_ensemble_1d(
        second_results,
        options=options,
    )

    np.testing.assert_array_equal(first.accepted_mask, [True, True, False])
    np.testing.assert_array_equal(second.accepted_mask, first.accepted_mask)
    assert second.representative_index == first.representative_index
    np.testing.assert_allclose(
        [run.loss for run in first.runs], [1.0, 1.02, 2.0]
    )
    np.testing.assert_allclose(
        [run.audit_loss for run in first.runs], [100.0, 1.0, 0.0]
    )


def test_ensemble_save_load_round_trip_without_pickle(tmp_path):
    results = [
        _result(
            [0.91, 0.04],
            [[0.0, 0.0], [0.01, 0.0]],
            [0.01, 0.0],
            loss=1.0,
            seed=7,
        ),
        _result(
            [0.89, 0.06],
            [[0.0, 0.0], [-0.01, 0.0]],
            [-0.01, 0.0],
            loss=1.01,
        ),
    ]
    ensemble = summarize_lattice_site_ensemble_1d(
        results,
        options=MultistartOptions1D(
            n_starts=2,
            minimum_accepted_starts=2,
            minimum_accepted_fraction=1.0,
        ),
        sensitivity_screen=_sensitivity([True, False]),
    )
    path = tmp_path / "ensemble.npz"

    save_lattice_site_ensemble_1d(path, ensemble)
    with np.load(path, allow_pickle=False) as stored:
        assert int(stored["schema_version"].item()) == 5
        assert all(value.dtype != object for value in stored.values())
    loaded = load_lattice_site_ensemble_1d(path)

    np.testing.assert_array_equal(loaded.accepted_mask, ensemble.accepted_mask)
    assert loaded.accepted_loss_cutoff == ensemble.accepted_loss_cutoff
    assert loaded.representative_index == ensemble.representative_index
    assert loaded.rigid_radial_q90_A == ensemble.rigid_radial_q90_A
    assert loaded.optimizer_stable is ensemble.optimizer_stable
    assert loaded.structurally_trusted is False
    np.testing.assert_allclose(loaded.rigid_median, ensemble.rigid_median)
    np.testing.assert_allclose(loaded.rigid_q05, ensemble.rigid_q05)
    np.testing.assert_allclose(loaded.rigid_q95, ensemble.rigid_q95)
    expected_flags = {
        name: None if value is None else bool(value)
        for name, value in ensemble.trust_flags.items()
    }
    expected_flags["archive_typed_evidence_persisted"] = False
    expected_flags["archive_structural_trust_reverified"] = False
    assert loaded.trust_flags == expected_flags
    np.testing.assert_array_equal(loaded.site_coordinates, ensemble.site_coordinates)
    assert loaded.options == ensemble.options
    assert loaded.scan_partition.n_scans == ensemble.scan_partition.n_scans
    for name in (
        "training_indices",
        "validation_indices",
        "audit_indices",
        "excluded_indices",
    ):
        np.testing.assert_array_equal(
            getattr(loaded.scan_partition, name),
            getattr(ensemble.scan_partition, name),
        )
    assert loaded.evidence_provenance.source == "loaded_compact_archive_v5"
    assert loaded.evidence_provenance.sensitivity_screen_supplied is True
    assert loaded.evidence_provenance.observability_report_count == 0
    assert loaded.evidence_provenance.typed_evidence_persisted is False
    assert (
        loaded.evidence_provenance.structural_trust_reverified_after_load
        is False
    )

    assert len(loaded.runs) == len(ensemble.runs)
    for actual, expected in zip(loaded.runs, ensemble.runs):
        assert actual.loss == expected.loss
        assert actual.converged is expected.converged
        assert actual.bound_fraction == expected.bound_fraction
        assert actual.seed == expected.seed
        assert actual.audit_loss == expected.audit_loss
        np.testing.assert_allclose(
            actual.vacancy_fractions, expected.vacancy_fractions
        )
        np.testing.assert_allclose(
            actual.residual_site_displacements,
            expected.residual_site_displacements,
        )
        np.testing.assert_allclose(
            actual.rigid_displacement, expected.rigid_displacement
        )

    for name in (
        "vacancy_median",
        "vacancy_q05",
        "vacancy_q95",
        "vacancy_call_frequency",
        "vacancy_state",
        "residual_displacement_median",
        "residual_displacement_q05",
        "residual_displacement_q95",
        "residual_displacement_radial_q90_A",
        "optimizer_agreement",
        "sensitive",
        "observable",
        "site_trusted",
    ):
        np.testing.assert_allclose(
            getattr(loaded.consensus, name),
            getattr(ensemble.consensus, name),
            equal_nan=True,
        )


def test_loaded_compact_archive_fails_structural_and_site_trust_closed(tmp_path):
    results = [
        _result(
            [0.91, 0.04],
            [[0.0, 0.0], [0.01, 0.0]],
            [0.01, 0.0],
            loss=1.0,
            material_scope_complete=True,
        ),
        _result(
            [0.90, 0.05],
            [[0.0, 0.0], [0.00, 0.0]],
            [0.00, 0.0],
            loss=1.0,
            material_scope_complete=True,
        ),
        _result(
            [0.89, 0.06],
            [[0.0, 0.0], [-0.01, 0.0]],
            [-0.01, 0.0],
            loss=1.0,
            material_scope_complete=True,
        ),
    ]
    ensemble = summarize_lattice_site_ensemble_1d(
        results,
        options=MultistartOptions1D(n_starts=3, minimum_accepted_starts=3),
        observability_reports=[_observability([True, True]) for _ in results],
        residual_calibration_evidence=_residual_evidence(),
        mismatch_benchmark_report=_mismatch_report(),
    )
    assert ensemble.structurally_trusted is True
    assert np.all(ensemble.consensus.site_trusted)

    path = tmp_path / "trusted_summary.npz"
    save_lattice_site_ensemble_1d(path, ensemble)
    loaded = load_lattice_site_ensemble_1d(path)

    assert loaded.structurally_trusted is False
    assert not np.any(loaded.consensus.site_trusted)
    assert loaded.optimizer_stable is True
    assert loaded.evidence_provenance.structurally_trusted_at_summary is True
    assert loaded.evidence_provenance.trusted_site_count_at_summary == 2
    assert loaded.evidence_provenance.observability_report_count == 3
    assert (
        loaded.evidence_provenance.observability_problem_ids_verified_at_summary
        is True
    )
    assert loaded.evidence_provenance.residual_calibration_evidence_supplied is True
    assert loaded.evidence_provenance.residual_calibration_passed_at_summary is True
    assert loaded.evidence_provenance.mismatch_benchmark_report_supplied is True
    assert loaded.evidence_provenance.mismatch_benchmark_passed_at_summary is True
    assert (
        loaded.evidence_provenance.common_reconstruction_problem_id
        == _PROBLEM_ID
    )
    assert loaded.evidence_provenance.common_reconstructor_id == _RECONSTRUCTOR_ID
    assert (
        loaded.evidence_provenance.mismatch_generator_id == _GENERATOR_ID
    )
    assert (
        loaded.evidence_provenance.mismatch_independent_forward_at_summary
        is True
    )
    assert loaded.trust_flags["archive_typed_evidence_persisted"] is False
    assert loaded.trust_flags["archive_structural_trust_reverified"] is False
