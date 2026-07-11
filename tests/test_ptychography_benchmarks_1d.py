"""Synthetic detector and forward-mismatch benchmark tests."""

from dataclasses import fields, replace
import hashlib
import json

import numpy as np
import pytest

from wide_angle_propagation.ptychography_benchmarks_1d import (
    BenchmarkCriteria1D,
    BenchmarkCriterion1D,
    DetectorMeasurement1D,
    DetectorPerturbation1D,
    ForwardModelInputs1D,
    ForwardModelMismatch1D,
    ReconstructionBenchmarkOutput1D,
    ResidualCalibrationEvidence1D,
    SyntheticBenchmarkScenario1D,
    ThresholdEvaluation1D,
    apply_forward_model_mismatch_1d,
    evaluate_residual_calibration_evidence_1d,
    generate_detector_measurement_1d,
    load_residual_calibration_evidence_1d,
    load_synthetic_benchmark_report_1d,
    ptychography_measurement_from_detector_1d,
    residual_calibration_report_1d,
    run_synthetic_benchmark_sweep_1d,
    save_residual_calibration_evidence_1d,
    save_synthetic_benchmark_report_1d,
    validate_benchmark_criteria_1d,
    validate_residual_calibration_evidence_1d,
    validate_synthetic_benchmark_report_1d,
)


def _nominal_inputs():
    return ForwardModelInputs1D(
        probe=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.complex128),
        probe_sampling_A=1.0,
        scan_coordinates_A=np.asarray([-1.0, 0.0, 1.0, 2.0]),
        detector_angles_rad=np.linspace(-0.02, 0.02, 5),
        energy_eV=30_000.0,
        incidence_angle_rad=0.08,
        potential=np.arange(12, dtype=float).reshape(3, 4),
        template=np.asarray([[0.0, 1.0], [2.0, 0.0]]),
        template_sampling_A=(1.0, 1.0),
    )


def test_forward_perturbations_are_reproducible_and_cover_physical_inputs():
    nominal = _nominal_inputs()
    mismatch = ForwardModelMismatch1D(
        probe_amplitude_scale=2.0,
        probe_position_offset_A=1.0,
        scan_position_offset_A=0.2,
        scan_jitter_std_A=0.05,
        incidence_angle_offset_rad=0.01,
        detector_angle_offset_rad=-0.003,
        energy_scale=0.99,
        potential_scale=1.1,
        template_scale=0.8,
    )
    first = apply_forward_model_mismatch_1d(nominal, mismatch, seed=17)
    repeated = apply_forward_model_mismatch_1d(nominal, mismatch, seed=17)
    different = apply_forward_model_mismatch_1d(nominal, mismatch, seed=18)

    np.testing.assert_allclose(first.probe, [0.0, 0.0, 2.0, 0.0])
    np.testing.assert_array_equal(first.scan_coordinates_A, repeated.scan_coordinates_A)
    assert not np.array_equal(first.scan_coordinates_A, different.scan_coordinates_A)
    np.testing.assert_allclose(
        first.detector_angles_rad,
        nominal.detector_angles_rad - 0.003,
    )
    assert first.incidence_angle_rad == pytest.approx(0.09)
    assert first.energy_eV == pytest.approx(29_700.0)
    np.testing.assert_allclose(first.potential, 1.1 * nominal.potential)
    np.testing.assert_allclose(first.template, 0.8 * nominal.template)


def test_template_shape_and_cutoff_mismatch_are_explicit_and_physical():
    nominal = replace(
        _nominal_inputs(),
        template=np.ones((5, 5)),
        template_sampling_A=(1.0, 1.0),
    )
    truncated = apply_forward_model_mismatch_1d(
        nominal,
        ForwardModelMismatch1D(template_cutoff_A=1.1),
        seed=1,
    )
    assert np.count_nonzero(truncated.template) == 5
    assert truncated.template[2, 2] == 1.0
    assert truncated.template[0, 0] == 0.0

    delta = np.zeros((5, 5))
    delta[2, 2] = 1.0
    broadened = apply_forward_model_mismatch_1d(
        replace(nominal, template=delta),
        ForwardModelMismatch1D(template_width_scale=2.0),
        seed=1,
    )
    assert np.count_nonzero(broadened.template) > 1
    assert broadened.template[2, 2] == 1.0


def test_a_nontrivial_missing_input_perturbation_is_not_silently_ignored():
    nominal = replace(
        _nominal_inputs(), template=None, template_sampling_A=None
    )
    with pytest.raises(ValueError, match="template_scale requires"):
        apply_forward_model_mismatch_1d(
            nominal,
            ForwardModelMismatch1D(template_scale=1.1),
            seed=0,
        )


def test_detector_generation_handles_gain_dark_masking_and_saturation():
    expected = np.asarray(
        [[1.0, 2.0, 100.0, 4.0], [2.0, 3.0, 100.0, 5.0]]
    )
    detector = DetectorPerturbation1D(
        gain_adu_per_electron=2.0,
        calibrated_gain_adu_per_electron=4.0,
        dark_electrons_per_pixel=3.0,
        calibrated_dark_electrons_per_pixel=1.0,
        saturation_electrons=20.0,
        masked_detector_indices=(1,),
        calibration_id="synthetic_gain_mismatch_v1",
    )
    first = generate_detector_measurement_1d(expected, detector, seed=5)
    repeated = generate_detector_measurement_1d(expected, detector, seed=5)

    np.testing.assert_array_equal(first.raw_adu, repeated.raw_adu)
    np.testing.assert_array_equal(first.valid_mask, repeated.valid_mask)
    np.testing.assert_allclose(
        first.calibrated_signal_electrons,
        first.raw_adu / 4.0 - 1.0,
    )
    assert np.all(first.masked_mask[:, 1])
    assert np.all(first.saturated_mask[:, 2])
    assert np.all(first.raw_adu[:, 2] == 40.0)
    assert not np.any(first.valid_mask[:, 1:3])
    assert first.calibration_id == "synthetic_gain_mismatch_v1"


def test_detector_rejects_a_scenario_with_no_usable_measurement():
    with pytest.raises(ValueError, match="no valid pixels"):
        generate_detector_measurement_1d(
            np.ones((2, 3)),
            DetectorPerturbation1D(masked_detector_indices=(0, 1, 2)),
            seed=1,
        )


def test_detector_adapter_exposes_only_calibrated_truth_free_measurement():
    raw_adu = np.asarray(
        [[4.0, 12.0, 20.0], [8.0, 16.0, 24.0]],
        dtype=np.float64,
    )
    calibrated_gain = 4.0
    calibrated_dark = 1.25
    calibrated_signal = raw_adu / calibrated_gain - calibrated_dark
    saturated = np.asarray(
        [[False, False, True], [False, False, False]],
        dtype=bool,
    )
    masked = np.asarray(
        [[False, False, False], [False, False, True]],
        dtype=bool,
    )
    valid = ~(saturated | masked)
    measurement = DetectorMeasurement1D(
        raw_adu=raw_adu,
        calibrated_signal_electrons=calibrated_signal,
        valid_mask=valid,
        saturated_mask=saturated,
        masked_mask=masked,
        calibrated_gain_adu_per_electron=calibrated_gain,
        calibrated_dark_electrons_per_pixel=calibrated_dark,
        calibrated_read_noise_std_electrons=0.375,
        calibration_id="declared-calibration-v19",
        detector_seed=1_987_654_321,
    )

    converted = ptychography_measurement_from_detector_1d(measurement)

    assert converted.__class__.__name__ == "PtychographyMeasurement1D"
    np.testing.assert_array_equal(
        converted.calibrated_signal_electrons,
        calibrated_signal,
    )
    np.testing.assert_array_equal(
        converted.observed_total_electrons,
        raw_adu / calibrated_gain,
    )
    np.testing.assert_array_equal(converted.valid_mask, valid)
    assert converted.valid_mask[0, 0]
    assert converted.calibrated_signal_electrons[0, 0] < 0.0
    assert converted.calibrated_dark_electrons_per_pixel == calibrated_dark
    assert converted.calibrated_read_noise_std_electrons == 0.375
    assert converted.calibration_id == "declared-calibration-v19"
    assert dict(converted.metadata) == {
        "source_type": "DetectorMeasurement1D",
        "adapter_schema": (
            "wide_angle_propagation."
            "ptychography_measurement_from_detector_1d:v1"
        ),
        "total_observation_semantics": "calibrated_electron_equivalent",
        "integer_count_contract": False,
    }

    expected_fields = {
        "calibrated_signal_electrons",
        "observed_total_electrons",
        "valid_mask",
        "calibrated_dark_electrons_per_pixel",
        "calibrated_read_noise_std_electrons",
        "calibration_id",
        "metadata",
    }
    assert {item.name for item in fields(converted)} == expected_fields
    forbidden = {
        "raw_adu",
        "saturated_mask",
        "masked_mask",
        "detector_seed",
        "scenario",
        "truth",
        "perturbed_inputs",
        "detection_efficiency",
        "gain_adu_per_electron",
        "true_gain_adu_per_electron",
        "true_dark_electrons_per_pixel",
        "true_read_noise_std_electrons",
    }
    assert all(not hasattr(converted, name) for name in forbidden)
    assert forbidden.isdisjoint(converted.metadata)
    assert not np.shares_memory(
        np.asarray(converted.calibrated_signal_electrons),
        calibrated_signal,
    )
    assert not np.shares_memory(np.asarray(converted.valid_mask), valid)


def test_detector_adapter_validates_measurement_before_projection():
    expected = np.full((2, 3), 5.0)
    measurement = generate_detector_measurement_1d(
        expected,
        DetectorPerturbation1D(calibration_id="adapter-validation-v1"),
        seed=5,
    )
    inconsistent = replace(
        measurement,
        valid_mask=np.zeros_like(measurement.valid_mask, dtype=bool),
    )
    with pytest.raises(ValueError, match="masks are internally inconsistent"):
        ptychography_measurement_from_detector_1d(inconsistent)

    inconsistent_calibration = replace(
        measurement,
        calibrated_signal_electrons=(
            np.asarray(measurement.calibrated_signal_electrons)
            + np.asarray(measurement.valid_mask, dtype=float)
        ),
    )
    with pytest.raises(ValueError, match="inconsistent with raw ADU"):
        ptychography_measurement_from_detector_1d(inconsistent_calibration)


def test_residual_calibration_matches_a_large_calibrated_poisson_sample():
    expected = np.full((2000, 10), 100.0)
    measurement = generate_detector_measurement_1d(
        expected,
        DetectorPerturbation1D(
            dark_electrons_per_pixel=2.0,
            calibrated_dark_electrons_per_pixel=2.0,
            calibration_id="exact_synthetic_counts",
        ),
        seed=52,
    )
    report = residual_calibration_report_1d(measurement, expected)

    assert abs(report.standardized_residual_mean) < 0.03
    assert report.standardized_residual_std == pytest.approx(1.0, abs=0.03)
    assert report.coverage_1sigma == pytest.approx(0.6827, abs=0.025)
    assert report.coverage_2sigma == pytest.approx(0.9545, abs=0.015)
    assert report.poisson_deviance_per_valid_pixel == pytest.approx(
        1.0, abs=0.04
    )
    assert report.poisson_deviance_model == (
        "poisson_deviance_on_electron_equivalents_under_declared_calibration"
    )
    assert report.calibration_id == "exact_synthetic_counts"


def test_read_noise_uses_standardized_coverage_without_claiming_poisson_deviance():
    expected = np.full((20, 5), 10.0)
    detector = DetectorPerturbation1D(
        read_noise_std_electrons=2.0,
        calibrated_read_noise_std_electrons=2.0,
        calibration_id="read_noise_v1",
    )
    measurement = generate_detector_measurement_1d(expected, detector, seed=9)
    report = residual_calibration_report_1d(measurement, expected)
    assert report.poisson_deviance_per_valid_pixel is None
    assert report.poisson_deviance_model == (
        "not_applicable_poisson_plus_read_noise"
    )
    assert np.isfinite(report.standardized_residual_std)


def _residual_evidence_criteria(*, upper_bound=5.0):
    return BenchmarkCriteria1D(
        criteria_id="held-out-residual-policy-v1",
        criteria=(
            BenchmarkCriterion1D(
                criterion_id="held-out-residual-bias",
                metric_name="residual.standardized_mean_abs",
                threshold_source="user:held-out-residual-policy-v1",
                upper_bound=upper_bound,
            ),
            BenchmarkCriterion1D(
                criterion_id="held-out-residual-scale",
                metric_name="residual.standardized_std_error",
                threshold_source="user:held-out-residual-policy-v1",
                upper_bound=upper_bound,
            ),
        ),
        metadata={"split": "held-out only"},
    )


def _residual_evidence(*, criteria=None):
    expected = np.full((4, 8), 40.0)
    measurement = generate_detector_measurement_1d(
        expected,
        DetectorPerturbation1D(calibration_id="detector-calibration-v7"),
        seed=12,
    )
    evidence = evaluate_residual_calibration_evidence_1d(
        measurement,
        expected,
        criteria=(
            _residual_evidence_criteria()
            if criteria is None
            else criteria
        ),
        held_out_scan_indices=np.asarray([2, 7, 11, 16], dtype=np.int64),
        reconstruction_problem_id="sha256:inverse-problem-and-result-v4",
    )
    return evidence, measurement, expected


def test_residual_evidence_binds_held_out_data_policy_and_problem_identity():
    evidence, measurement, prediction = _residual_evidence()

    assert evidence.passed
    assert evidence.held_out_scan_indices == (2, 7, 11, 16)
    assert evidence.measurement_shape == (4, 8)
    assert evidence.calibration_id == measurement.calibration_id
    assert evidence.residual_calibration.calibration_id == measurement.calibration_id
    assert len(evidence.measurement_sha256) == 64
    assert len(evidence.prediction_sha256) == 64
    assert all(
        evaluation.scenario_id == evidence.reconstruction_problem_id
        for evaluation in evidence.threshold_evaluations
    )
    assert "passed" not in {item.name for item in fields(evidence)}
    validate_residual_calibration_evidence_1d(evidence)

    changed = evaluate_residual_calibration_evidence_1d(
        measurement,
        prediction + 1.0,
        criteria=evidence.criteria,
        held_out_scan_indices=evidence.held_out_scan_indices,
        reconstruction_problem_id=evidence.reconstruction_problem_id,
    )
    assert changed.prediction_sha256 != evidence.prediction_sha256
    assert changed.measurement_sha256 == evidence.measurement_sha256

    repeated_with_new_counts = generate_detector_measurement_1d(
        prediction,
        DetectorPerturbation1D(calibration_id="detector-calibration-v7"),
        seed=13,
    )
    changed_measurement = evaluate_residual_calibration_evidence_1d(
        repeated_with_new_counts,
        prediction,
        criteria=evidence.criteria,
        held_out_scan_indices=evidence.held_out_scan_indices,
        reconstruction_problem_id=evidence.reconstruction_problem_id,
    )
    assert changed_measurement.measurement_sha256 != evidence.measurement_sha256
    assert changed_measurement.prediction_sha256 == evidence.prediction_sha256


def test_residual_evidence_rejects_nonresidual_or_unavailable_criteria():
    nonresidual = BenchmarkCriteria1D(
        criteria_id="invalid-truth-policy",
        criteria=(
            BenchmarkCriterion1D(
                criterion_id="truth-error",
                metric_name="truth.vacancy.rmse",
                threshold_source="user:invalid-for-residuals",
                upper_bound=0.1,
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"only residual\.\* criteria"):
        _residual_evidence(criteria=nonresidual)

    expected = np.full((3, 5), 20.0)
    measurement = generate_detector_measurement_1d(
        expected,
        DetectorPerturbation1D(
            read_noise_std_electrons=1.0,
            calibrated_read_noise_std_electrons=1.0,
        ),
        seed=3,
    )
    unavailable = BenchmarkCriteria1D(
        criteria_id="unavailable-poisson-policy",
        criteria=(
            BenchmarkCriterion1D(
                criterion_id="poisson-deviance",
                metric_name="residual.poisson_deviance_per_valid_pixel",
                threshold_source="user:poisson-only",
                upper_bound=2.0,
            ),
        ),
    )
    with pytest.raises(ValueError, match="is unavailable"):
        evaluate_residual_calibration_evidence_1d(
            measurement,
            expected,
            criteria=unavailable,
            held_out_scan_indices=(1, 3, 5),
            reconstruction_problem_id="read-noise-problem-v1",
        )


@pytest.mark.parametrize(
    ("indices", "problem_id", "message"),
    [
        ((), "problem-v1", "must not be empty"),
        ((1, 1, 2, 3), "problem-v1", "must be unique"),
        ((1, 2), "problem-v1", "leading dimension"),
        ((1, 2, 3, 4), " ", "must not be empty"),
    ],
)
def test_residual_evidence_requires_explicit_held_out_provenance(
    indices, problem_id, message
):
    expected = np.full((4, 5), 10.0)
    measurement = generate_detector_measurement_1d(
        expected, DetectorPerturbation1D(), seed=2
    )
    with pytest.raises((TypeError, ValueError), match=message):
        evaluate_residual_calibration_evidence_1d(
            measurement,
            expected,
            criteria=_residual_evidence_criteria(),
            held_out_scan_indices=indices,
            reconstruction_problem_id=problem_id,
        )


def test_residual_evidence_validation_rederives_thresholds_and_calibration():
    evidence, _, _ = _residual_evidence()
    changed_evaluation = replace(
        evidence.threshold_evaluations[0],
        observed_value=evidence.threshold_evaluations[0].observed_value + 0.1,
    )
    with pytest.raises(ValueError, match="evaluations differ"):
        validate_residual_calibration_evidence_1d(
            replace(
                evidence,
                threshold_evaluations=(changed_evaluation,)
                + evidence.threshold_evaluations[1:],
            )
        )
    with pytest.raises(ValueError, match="identifiers differ"):
        validate_residual_calibration_evidence_1d(
            replace(evidence, calibration_id="different-calibration")
        )


def test_residual_evidence_round_trip_is_nonpickled_and_tamper_evident(
    tmp_path,
):
    evidence, _, _ = _residual_evidence()
    path = tmp_path / "held_out_residual_evidence.npz"
    save_residual_calibration_evidence_1d(path, evidence)

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema_version",
            "payload_json",
            "payload_sha256",
        }
        assert all(array.dtype != object for array in archive.values())
        payload_text = str(archive["payload_json"].item())
        digest = str(archive["payload_sha256"].item())
    payload = json.loads(payload_text)
    assert "passed" not in payload
    assert payload["held_out_scan_indices"] == [2, 7, 11, 16]
    assert payload["criteria"]["criteria"][0]["threshold_source"].startswith(
        "user:"
    )

    loaded = load_residual_calibration_evidence_1d(path)
    assert loaded == evidence
    assert loaded.passed == evidence.passed

    digest_tamper = tmp_path / "residual_digest_tamper.npz"
    np.savez_compressed(
        digest_tamper,
        schema_version=np.asarray(1, dtype=np.int64),
        payload_json=np.asarray(payload_text.replace("problem-and-result", "PROBLEM")),
        payload_sha256=np.asarray(digest),
    )
    with pytest.raises(ValueError, match="digest does not match"):
        load_residual_calibration_evidence_1d(digest_tamper)

    structural_tamper = tmp_path / "residual_structural_tamper.npz"
    payload["threshold_evaluations"][0]["observed_value"] += 0.25
    tampered_text = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    np.savez_compressed(
        structural_tamper,
        schema_version=np.asarray(1, dtype=np.int64),
        payload_json=np.asarray(tampered_text),
        payload_sha256=np.asarray(
            hashlib.sha256(tampered_text.encode("utf-8")).hexdigest()
        ),
    )
    with pytest.raises(ValueError, match="evaluations differ"):
        load_residual_calibration_evidence_1d(structural_tamper)

    extra_field = tmp_path / "residual_extra_field.npz"
    extra_payload = json.loads(payload_text)
    extra_payload["assignable_passed"] = True
    extra_text = json.dumps(
        extra_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    np.savez_compressed(
        extra_field,
        schema_version=np.asarray(1, dtype=np.int64),
        payload_json=np.asarray(extra_text),
        payload_sha256=np.asarray(
            hashlib.sha256(extra_text.encode("utf-8")).hexdigest()
        ),
    )
    with pytest.raises(ValueError, match="invalid fields"):
        load_residual_calibration_evidence_1d(extra_field)

    wrong_schema = tmp_path / "residual_wrong_schema.npz"
    np.savez_compressed(
        wrong_schema,
        schema_version=np.asarray(2, dtype=np.int64),
        payload_json=np.asarray(payload_text),
        payload_sha256=np.asarray(digest),
    )
    with pytest.raises(ValueError, match="unsupported"):
        load_residual_calibration_evidence_1d(wrong_schema)


def _criteria(*, truth_rmse=0.2):
    return BenchmarkCriteria1D(
        criteria_id="user_acceptance_v3",
        criteria=(
            BenchmarkCriterion1D(
                criterion_id="theta_accuracy",
                metric_name="truth.theta.rmse",
                threshold_source="user:vacancy-strain-study-v3",
                upper_bound=truth_rmse,
            ),
            BenchmarkCriterion1D(
                criterion_id="residual_bias",
                metric_name="residual.standardized_mean_abs",
                threshold_source="user:vacancy-strain-study-v3",
                upper_bound=2.0,
            ),
            BenchmarkCriterion1D(
                criterion_id="usable_detector_fraction",
                metric_name="data.valid_fraction",
                threshold_source="user:vacancy-strain-study-v3",
                lower_bound=0.8,
            ),
        ),
        metadata={"owner": "synthetic benchmark test"},
    )


def _benchmark_report(criteria=None):
    nominal = _nominal_inputs()
    forward_seen = []
    reconstruction_seen = []

    def forward(inputs):
        forward_seen.append(np.asarray(inputs.scan_coordinates_A).copy())
        scans = np.asarray(inputs.scan_coordinates_A)[:, None]
        angles = np.asarray(inputs.detector_angles_rad)[None, :]
        return 40.0 + 0.5 * scans + 10.0 * angles

    def reconstruct(measurement, inputs):
        reconstruction_seen.append(inputs)
        prediction = np.maximum(measurement.calibrated_signal_electrons, 0.0)
        return ReconstructionBenchmarkOutput1D(
            predicted_signal_electrons=prediction,
            estimated_parameters={"theta": np.asarray([1.05])},
            metadata={"optimizer": "test estimator", "iterations": 3},
        )

    scenarios = (
        SyntheticBenchmarkScenario1D(
            scenario_id="nominal_poisson",
            seed=7,
            metadata={"family": "detector"},
        ),
        SyntheticBenchmarkScenario1D(
            scenario_id="jitter_read_mask",
            seed=8,
            detector=DetectorPerturbation1D(
                read_noise_std_electrons=0.5,
                calibrated_read_noise_std_electrons=0.5,
                masked_detector_indices=(4,),
                calibration_id="synthetic_read_mask_v1",
            ),
            forward_mismatch=ForwardModelMismatch1D(
                probe_amplitude_scale=1.02,
                probe_position_offset_A=0.05,
                scan_jitter_std_A=0.03,
                detector_angle_offset_rad=1e-3,
                energy_scale=0.995,
                potential_scale=1.01,
                template_scale=0.98,
            ),
            metadata={"family": "combined"},
        ),
    )
    report = run_synthetic_benchmark_sweep_1d(
        nominal,
        {"theta": np.asarray([1.0])},
        scenarios,
        forward,
        reconstruct,
        criteria=_criteria() if criteria is None else criteria,
        benchmark_id="tiny_truth_aware_sweep_v1",
        truth_id="tiny_truth_v1",
        generator_id="test_forward_v1",
        reconstructor_id="test_inverse_v1",
        metadata={"purpose": "unit test"},
    )
    return report, nominal, forward_seen, reconstruction_seen


def test_sweep_keeps_perturbed_truth_inputs_out_of_reconstruction_callback():
    report, nominal, forward_seen, reconstruction_seen = _benchmark_report()

    assert report.passed
    assert all(item.passed for item in report.scenarios)
    assert len(forward_seen) == 2
    np.testing.assert_array_equal(forward_seen[0], nominal.scan_coordinates_A)
    assert not np.array_equal(forward_seen[1], nominal.scan_coordinates_A)
    assert all(item is nominal for item in reconstruction_seen)
    assert report.scenarios[1].metrics["data.valid_fraction"] == 0.8
    assert report.scenarios[1].scenario.forward_mismatch.scan_jitter_std_A == 0.03
    assert len(report.worst_case_evaluations) == 3
    assert report.truth_metric_id == "truth_parameter_error_metrics_1d:v1"
    assert len(report.scenarios[0].estimated_parameters_sha256) == 64
    assert all(
        gate.criterion.threshold_source == "user:vacancy-strain-study-v3"
        for gate in report.worst_case_evaluations
    )


def test_acceptance_is_derived_from_explicit_threshold_evidence():
    criterion = BenchmarkCriterion1D(
        criterion_id="accuracy",
        metric_name="truth.rmse",
        threshold_source="user:explicit-test",
        upper_bound=0.5,
    )
    evaluation = ThresholdEvaluation1D(
        criterion=criterion,
        observed_value=0.4,
        scenario_id="test",
    )
    assert evaluation.passed
    assert not replace(
        evaluation,
        criterion=replace(criterion, upper_bound=0.3),
    ).passed

    strict_report, _, _, _ = _benchmark_report(criteria=_criteria(truth_rmse=0.01))
    assert not strict_report.passed
    assert not all(item.passed for item in strict_report.scenarios)
    worst = strict_report.worst_case_evaluations[0]
    assert worst.observed_value == pytest.approx(0.05)
    assert worst.criterion.upper_bound == 0.01


@pytest.mark.parametrize(
    ("criteria", "message"),
    [
        (
            BenchmarkCriteria1D(criteria_id="empty", criteria=()),
            "non-empty tuple",
        ),
        (
            BenchmarkCriteria1D(
                criteria_id="missing-source",
                criteria=(
                    BenchmarkCriterion1D(
                        criterion_id="x",
                        metric_name="truth.x.rmse",
                        threshold_source=" ",
                        upper_bound=1.0,
                    ),
                ),
            ),
            "must not be empty",
        ),
        (
            BenchmarkCriteria1D(
                criteria_id="unbounded",
                criteria=(
                    BenchmarkCriterion1D(
                        criterion_id="x",
                        metric_name="truth.x.rmse",
                        threshold_source="user:test",
                    ),
                ),
            ),
            "requires at least one bound",
        ),
    ],
)
def test_thresholds_must_be_explicit_and_sourced(criteria, message):
    with pytest.raises((TypeError, ValueError), match=message):
        validate_benchmark_criteria_1d(criteria)


def test_unavailable_diagnostic_cannot_be_silently_treated_as_passing():
    criteria = BenchmarkCriteria1D(
        criteria_id="requires-poisson",
        criteria=(
            BenchmarkCriterion1D(
                criterion_id="poisson-deviance",
                metric_name="residual.poisson_deviance_per_valid_pixel",
                threshold_source="user:test",
                upper_bound=2.0,
            ),
        ),
    )
    with pytest.raises(ValueError, match="is unavailable"):
        _benchmark_report(criteria=criteria)


def test_custom_truth_metric_requires_an_explicit_evaluator_identifier():
    nominal = _nominal_inputs()

    def forward(inputs):
        return np.full(
            (
                np.asarray(inputs.scan_coordinates_A).size,
                np.asarray(inputs.detector_angles_rad).size,
            ),
            10.0,
        )

    def reconstruct(measurement, inputs):
        del inputs
        return ReconstructionBenchmarkOutput1D(
            predicted_signal_electrons=np.full(
                np.asarray(measurement.raw_adu).shape, 10.0
            ),
            estimated_parameters={"theta": np.asarray([1.0])},
        )

    with pytest.raises(ValueError, match="truth_metric_id is required"):
        run_synthetic_benchmark_sweep_1d(
            nominal,
            {"theta": np.asarray([1.0])},
            (SyntheticBenchmarkScenario1D("nominal", seed=1),),
            forward,
            reconstruct,
            criteria=_criteria(),
            benchmark_id="custom-metric-test",
            truth_id="truth-v1",
            generator_id="forward-v1",
            reconstructor_id="inverse-v1",
            truth_metric_callback=lambda truth, estimates: {
                "theta.rmse": 0.0
            },
        )


def test_report_round_trip_is_non_pickled_and_schema_validated(tmp_path):
    report, _, _, _ = _benchmark_report()
    path = tmp_path / "benchmark.npz"
    save_synthetic_benchmark_report_1d(path, report)

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema_version",
            "payload_json",
            "payload_sha256",
        }
        assert all(array.dtype != object for array in archive.values())
    loaded = load_synthetic_benchmark_report_1d(path)
    assert loaded == report
    assert loaded.passed == report.passed
    validate_synthetic_benchmark_report_1d(loaded)

    with np.load(path, allow_pickle=False) as archive:
        payload = str(archive["payload_json"].item())
        digest = str(archive["payload_sha256"].item())
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(
        tampered,
        schema_version=np.asarray(1, dtype=np.int64),
        payload_json=np.asarray(payload.replace("unit test", "unit TEST")),
        payload_sha256=np.asarray(digest),
    )
    with pytest.raises(ValueError, match="digest does not match"):
        load_synthetic_benchmark_report_1d(tampered)

    wrong_schema = tmp_path / "wrong_schema.npz"
    actual_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    np.savez_compressed(
        wrong_schema,
        schema_version=np.asarray(99, dtype=np.int64),
        payload_json=np.asarray(payload),
        payload_sha256=np.asarray(actual_digest),
    )
    with pytest.raises(ValueError, match="unsupported"):
        load_synthetic_benchmark_report_1d(wrong_schema)

    extra_field = tmp_path / "extra_field.npz"
    payload_object = json.loads(payload)
    payload_object["unversioned_claim"] = True
    extra_payload = json.dumps(
        payload_object,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    np.savez_compressed(
        extra_field,
        schema_version=np.asarray(1, dtype=np.int64),
        payload_json=np.asarray(extra_payload),
        payload_sha256=np.asarray(
            hashlib.sha256(extra_payload.encode("utf-8")).hexdigest()
        ),
    )
    with pytest.raises(ValueError, match="invalid fields"):
        load_synthetic_benchmark_report_1d(extra_field)


def test_report_validation_detects_metric_gate_tampering():
    report, _, _, _ = _benchmark_report()
    first = report.scenarios[0]
    changed_metrics = dict(first.metrics)
    changed_metrics["truth.theta.rmse"] = 0.9
    tampered = replace(
        report,
        scenarios=(replace(first, metrics=changed_metrics),) + report.scenarios[1:],
    )
    with pytest.raises(ValueError, match="threshold value differs"):
        validate_synthetic_benchmark_report_1d(tampered)


def test_report_json_contains_scenarios_thresholds_seeds_and_no_pass_field(tmp_path):
    report, _, _, _ = _benchmark_report()
    path = tmp_path / "benchmark.npz"
    save_synthetic_benchmark_report_1d(path, report)
    with np.load(path, allow_pickle=False) as archive:
        payload = json.loads(str(archive["payload_json"].item()))

    assert payload["criteria"]["criteria"][0]["threshold_source"].startswith(
        "user:"
    )
    assert payload["scenarios"][1]["scenario"]["forward_mismatch"][
        "scan_jitter_std_A"
    ] == 0.03
    assert payload["scenarios"][0]["detector_seed"] != payload["scenarios"][0][
        "mismatch_seed"
    ]
    assert "passed" not in json.dumps(payload)
