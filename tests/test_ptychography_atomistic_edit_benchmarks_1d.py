"""AE-3 truth-isolation and fail-closed benchmark-contract tests."""

from dataclasses import fields, replace
import hashlib

import numpy as np
import pytest

from wide_angle_propagation.ptychography_atomistic_edit_benchmarks_1d import (
    AE3_ABLATION_CATALOG_1D,
    AE3_BLIND_CASE_CATALOG_1D,
    ActiveEditMultistartEvidence1D,
    AtomisticEditAblationArm1D,
    AtomisticEditAblationStatus1D,
    AtomisticEditBlindAcceptancePolicy1D,
    AtomisticEditBlindAuditCounts1D,
    AtomisticEditBlindCase1D,
    AtomisticEditBlindCaseRole1D,
    AtomisticEditBlindPrivateTruth1D,
    AtomisticEditBlindPublicProblem1D,
    AtomisticEditBlindReconstruction1D,
    AtomisticEditReconstructionContract1D,
    ObservabilityEvidence1D,
    PhysicalAdmissibilityMetrics1D,
    ResolutionAwareMassMeasure1D,
    atomistic_edit_public_problem_digest_1d,
    atomistic_edit_public_problem_schema_digest_1d,
    resolution_aware_mass_transport_metrics_1d,
    run_atomistic_edit_blind_benchmarks_1d,
    validate_atomistic_edit_blind_benchmark_report_1d,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _contract(*, model: str = "shared-model") -> AtomisticEditReconstructionContract1D:
    return AtomisticEditReconstructionContract1D(
        model_sha256=_digest(model),
        options_sha256=_digest("object-free-options"),
        prior_sha256=_digest("frozen-level1-prior"),
        selection_rule_sha256=_digest("frozen-lambda-selection"),
        nuisance_scope_sha256=_digest("small-common-nuisance"),
        observability_rule_sha256=_digest("independent-resolution-rule"),
        fitted_spatial_dimension=2,
    )


def _public_problem(
    *,
    contract: AtomisticEditReconstructionContract1D | None = None,
) -> AtomisticEditBlindPublicProblem1D:
    observed = np.asarray([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    return AtomisticEditBlindPublicProblem1D(
        selection_observed_total_electrons=observed,
        selection_valid_mask=np.ones_like(observed, dtype=bool),
        audit_prediction_shape=(1, 3),
        contract=_contract() if contract is None else contract,
        public_arrays={
            "probe_rows": np.eye(3),
            "scan_positions_A": np.asarray([-1.0, 0.0, 1.0]),
        },
        public_scalars={"energy_eV": 30_000.0, "window_length": 3},
    )


def _empty() -> ResolutionAwareMassMeasure1D:
    return ResolutionAwareMassMeasure1D.empty(2)


def _measure(*positions, masses=None) -> ResolutionAwareMassMeasure1D:
    if masses is None:
        masses = np.ones(len(positions))
    return ResolutionAwareMassMeasure1D(
        positions_A=np.asarray(positions, dtype=float).reshape(-1, 2),
        masses_host_equivalent=np.asarray(masses, dtype=float),
    )


def _truth(role: AtomisticEditBlindCaseRole1D, *, shift: float = 0.0):
    empty = _empty()
    if role is AtomisticEditBlindCaseRole1D.PRISTINE_HOST:
        return AtomisticEditBlindPrivateTruth1D(empty, empty)
    if role is AtomisticEditBlindCaseRole1D.ONE_VACANCY:
        return AtomisticEditBlindPrivateTruth1D(
            empty, _measure((1.0 + shift, 2.0))
        )
    if role is AtomisticEditBlindCaseRole1D.ONE_OFF_LATTICE_ADDITION:
        return AtomisticEditBlindPrivateTruth1D(
            _measure((1.3 + shift, 2.7)), empty
        )
    if role is AtomisticEditBlindCaseRole1D.ONE_SUBSTITUTION:
        return AtomisticEditBlindPrivateTruth1D(
            _measure((1.0 + shift, 2.0), masses=(0.8,)),
            _measure((1.0 + shift, 2.0)),
            host_kernel_id="Si-host",
            generating_addition_kernel_id="Ge-independent-direct-quadrature",
            generating_element="Ge",
        )
    if role is AtomisticEditBlindCaseRole1D.IRREGULAR_FINITE_CLUSTER:
        return AtomisticEditBlindPrivateTruth1D(
            _measure((0.2 + shift, 1.1), (1.7 + shift, 2.4), masses=(0.7, 1.2)),
            empty,
        )
    if role is AtomisticEditBlindCaseRole1D.METASTABLE_DEFECT:
        return AtomisticEditBlindPrivateTruth1D(
            empty, empty, host_deformation_rms_A=0.15 + shift
        )
    if role is AtomisticEditBlindCaseRole1D.NUISANCE_ONLY_MISMATCH:
        return AtomisticEditBlindPrivateTruth1D(
            empty, empty, mismatch_cause="private_probe_coherence_mismatch"
        )
    if role is AtomisticEditBlindCaseRole1D.AXIALLY_UNRESOLVED_ADDITION:
        return AtomisticEditBlindPrivateTruth1D(
            _measure((2.2 + shift, 1.8)),
            empty,
            axial_depth_uncertainty_A=1.2,
            slice_thickness_A=0.5,
        )
    raise AssertionError(role)


def _cases(
    events: list[str] | None = None,
    *,
    label_prefix: str = "private-v1",
    truth_shift: float = 0.0,
    audit_shift: float = 0.0,
    contract: AtomisticEditReconstructionContract1D | None = None,
):
    problem = _public_problem(contract=contract)
    result = []
    for role in AE3_BLIND_CASE_CATALOG_1D:
        def audit_factory(role=role):
            if events is not None:
                events.append(f"audit:{role.value}")
            return AtomisticEditBlindAuditCounts1D(
                observed_total_electrons=np.asarray(
                    [[5.0 + audit_shift, 4.0, 3.0]]
                ),
                valid_mask=np.ones((1, 3), dtype=bool),
            )

        def truth_factory(role=role):
            if events is not None:
                events.append(f"truth:{role.value}")
            return _truth(role, shift=truth_shift)

        result.append(
            AtomisticEditBlindCase1D(
                role=role,
                private_case_label=f"{label_prefix}:{role.value}",
                public_problem=problem,
                private_audit_factory=audit_factory,
                private_truth_factory=truth_factory,
            )
        )
    return tuple(result)


def _reconstruction(
    problem: AtomisticEditBlindPublicProblem1D,
    *,
    observability: ObservabilityEvidence1D | None = None,
) -> AtomisticEditBlindReconstruction1D:
    return AtomisticEditBlindReconstruction1D(
        predicted_selection_total_electrons=(
            problem.selection_observed_total_electrons
        ),
        predicted_audit_total_electrons=np.asarray([[5.0, 4.0, 3.0]]),
        additions=_empty(),
        removals=_empty(),
        deformation_parameter_count=12,
        fitted_spatial_dimension=2,
        maximum_dormant_kkt_violation=0.0,
        recovered_host_deformation_rms_A=0.0,
        multistart=ActiveEditMultistartEvidence1D(
            validation_count_deviances=(1.0, 1.0),
            total_addition_masses=(0.0, 0.0),
            total_removal_masses=(0.0, 0.0),
            support_distance_to_medoid_resolution_units=(0.0, 0.0),
            selected_start_index=0,
            ambiguity_disposition="identifiable",
        ),
        physical_metrics=PhysicalAdmissibilityMetrics1D(
            hard_core_overlap_mass=0.0,
            host_deformation_roughness=0.0,
        ),
        observability=observability,
        # Nuisance, depth, and archive evidence are intentionally absent here.
    )


def _policy() -> AtomisticEditBlindAcceptancePolicy1D:
    return AtomisticEditBlindAcceptancePolicy1D(
        threshold_source="frozen-test-policy:v1"
    )


def test_catalog_is_exactly_the_required_eight_cases_and_three_arms():
    assert tuple(item.value for item in AE3_BLIND_CASE_CATALOG_1D) == (
        "pristine_host",
        "one_vacancy",
        "one_off_lattice_interstitial_or_adatom",
        "one_substitution_different_truth_kernel",
        "irregular_finite_added_cluster",
        "data_supported_strained_or_metastable_defect",
        "probe_scan_or_coherence_mismatch_no_defect",
        "axially_unresolved_addition",
    )
    assert tuple(item.value for item in AE3_ABLATION_CATALOG_1D) == (
        "a0_count_likelihood_plus_edit_penalty",
        "a1_plus_hard_core_and_host_elasticity",
        "a2_plus_material_energy_envelope",
    )
    assert {item.value for item in AtomisticEditAblationStatus1D} == {
        "completed",
        "failed",
        "blocked_not_run",
    }


def test_private_truth_and_label_changes_cannot_taint_callback_input():
    first_events: list[str] = []
    second_events: list[str] = []
    first_inputs: list[tuple[str, str, tuple[str, ...]]] = []
    second_inputs: list[tuple[str, str, tuple[str, ...]]] = []

    forbidden = {
        "role",
        "case_id",
        "seed",
        "truth",
        "private_audit_factory",
        "private_case_label",
        "generating_element",
        "generating_coordinates",
        "mismatch_cause",
        "object_metadata",
    }

    def callback(log, events):
        def run(problem):
            events.append("callback")
            names = tuple(item.name for item in fields(problem))
            assert forbidden.isdisjoint(names)
            assert forbidden.isdisjoint(problem.public_arrays)
            assert forbidden.isdisjoint(problem.public_scalars)
            assert all(not value.flags.writeable for value in problem.public_arrays.values())
            log.append(
                (
                    atomistic_edit_public_problem_digest_1d(problem),
                    atomistic_edit_public_problem_schema_digest_1d(problem),
                    names,
                )
            )
            return _reconstruction(problem)

        return run

    callbacks_first = {
        AtomisticEditAblationArm1D.COUNT_AND_EDIT: callback(
            first_inputs, first_events
        ),
        AtomisticEditAblationArm1D.LEVEL1_PHYSICAL: callback(
            first_inputs, first_events
        ),
    }
    callbacks_second = {
        AtomisticEditAblationArm1D.COUNT_AND_EDIT: callback(
            second_inputs, second_events
        ),
        AtomisticEditAblationArm1D.LEVEL1_PHYSICAL: callback(
            second_inputs, second_events
        ),
    }
    first = run_atomistic_edit_blind_benchmarks_1d(
        _cases(first_events, label_prefix="secret-label-a", truth_shift=0.0),
        callbacks_first,
        _policy(),
    )
    second = run_atomistic_edit_blind_benchmarks_1d(
        _cases(
            second_events,
            label_prefix="entirely-different",
            truth_shift=0.03,
            audit_shift=0.7,
        ),
        callbacks_second,
        _policy(),
    )

    assert first_inputs == second_inputs
    assert len(first_inputs) == 16
    assert first_events[:16] == ["callback"] * 16
    assert second_events[:16] == ["callback"] * 16
    assert all(
        event.startswith(("audit:", "truth:")) for event in first_events[16:]
    )
    assert all(
        event.startswith(("audit:", "truth:")) for event in second_events[16:]
    )
    assert {item.public_problem_sha256 for item in first.case_reports} == {
        item.public_problem_sha256 for item in second.case_reports
    }
    first_level1 = next(
        item
        for item in first.case_reports
        if item.case_role is AtomisticEditBlindCaseRole1D.PRISTINE_HOST
        and item.ablation is AtomisticEditAblationArm1D.LEVEL1_PHYSICAL
    )
    second_level1 = next(
        item
        for item in second.case_reports
        if item.case_role is AtomisticEditBlindCaseRole1D.PRISTINE_HOST
        and item.ablation is AtomisticEditAblationArm1D.LEVEL1_PHYSICAL
    )
    assert first_level1.held_out_count_metrics.poisson_deviance == pytest.approx(0.0)
    assert second_level1.held_out_count_metrics.poisson_deviance > 0.0


def test_public_problem_rejects_metadata_side_channels_and_is_immutable():
    base = _public_problem()
    with pytest.raises(ValueError, match="private 'truth'"):
        replace(base, public_arrays={"truth_positions_A": np.zeros((1, 2))})
    with pytest.raises(ValueError, match="private 'seed'"):
        replace(base, public_scalars={"simulation_seed": 7})
    with pytest.raises(TypeError, match="number or Boolean"):
        replace(base, public_scalars={"nominal_mode": "one_vacancy"})

    assert not base.selection_observed_total_electrons.flags.writeable
    assert not base.selection_valid_mask.flags.writeable
    with pytest.raises(ValueError):
        base.selection_observed_total_electrons[0, 0] = 99.0
    with pytest.raises(TypeError):
        base.public_arrays["new"] = np.zeros(1)


def test_runner_requires_each_role_once_and_identical_public_contract_schema():
    cases = list(_cases())
    callbacks = {
        AtomisticEditAblationArm1D.COUNT_AND_EDIT: _reconstruction,
        AtomisticEditAblationArm1D.LEVEL1_PHYSICAL: _reconstruction,
    }
    with pytest.raises(ValueError, match="exactly eight"):
        run_atomistic_edit_blind_benchmarks_1d(cases[:-1], callbacks, _policy())

    cases[-1] = replace(cases[-1], role=cases[0].role)
    with pytest.raises(ValueError, match="each required role exactly once"):
        run_atomistic_edit_blind_benchmarks_1d(cases, callbacks, _policy())

    cases = list(_cases())
    changed_problem = _public_problem(contract=_contract(model="different-model"))
    cases[-1] = replace(cases[-1], public_problem=changed_problem)
    with pytest.raises(ValueError, match="identical model, options, prior"):
        run_atomistic_edit_blind_benchmarks_1d(cases, callbacks, _policy())

    cases = list(_cases())
    changed_shape = replace(
        cases[-1].public_problem,
        public_arrays={
            "probe_rows": np.eye(4),
            "scan_positions_A": np.asarray([-1.0, 0.0, 1.0]),
        },
    )
    cases[-1] = replace(cases[-1], public_problem=changed_shape)
    with pytest.raises(ValueError, match="identical public schema"):
        run_atomistic_edit_blind_benchmarks_1d(cases, callbacks, _policy())


def test_energy_ablation_is_present_but_unconditionally_blocked_in_v1():
    callbacks = {
        AtomisticEditAblationArm1D.COUNT_AND_EDIT: _reconstruction,
        AtomisticEditAblationArm1D.LEVEL1_PHYSICAL: _reconstruction,
    }
    report = run_atomistic_edit_blind_benchmarks_1d(
        _cases(), callbacks, _policy()
    )
    energy = [
        item
        for item in report.case_reports
        if item.ablation is AtomisticEditAblationArm1D.MATERIAL_ENERGY_ENVELOPE
    ]
    assert len(energy) == 8
    assert all(
        item.status is AtomisticEditAblationStatus1D.BLOCKED_NOT_RUN
        for item in energy
    )
    assert all(item.reconstruction is None for item in energy)
    assert all(item.failure_stage == "chemistry_validation_gate" for item in energy)
    assert all("surfaces, defects, strain" in item.diagnostic for item in energy)
    validate_atomistic_edit_blind_benchmark_report_1d(report)

    with pytest.raises(ValueError, match="energy arm is blocked"):
        run_atomistic_edit_blind_benchmarks_1d(
            _cases(),
            {
                **callbacks,
                AtomisticEditAblationArm1D.MATERIAL_ENERGY_ENVELOPE: _reconstruction,
            },
            _policy(),
        )


def test_report_fails_closed_when_nuisance_depth_or_observability_is_missing():
    report = run_atomistic_edit_blind_benchmarks_1d(
        _cases(),
        {
            AtomisticEditAblationArm1D.COUNT_AND_EDIT: _reconstruction,
            AtomisticEditAblationArm1D.LEVEL1_PHYSICAL: _reconstruction,
        },
        _policy(),
    )
    gate_by_id = {gate.gate_id: gate for gate in report.gates}
    nuisance = gate_by_id[
        "probe_scan_or_coherence_mismatch_no_defect.level1.nuisance_attribution"
    ]
    observability = gate_by_id[
        "one_vacancy.level1.observability_evidence"
    ]
    depth = gate_by_id[
        "axially_unresolved_addition.level1.depth_uncertainty_reporting"
    ]
    feature = gate_by_id[
        "axially_unresolved_addition.level1.no_subresponse_axial_feature"
    ]
    assert nuisance.measured_value is None and not nuisance.passed
    assert observability.measured_value is None and not observability.passed
    assert depth.measured_value is None and not depth.passed
    assert feature.measured_value is None and not feature.passed
    assert not report.accepted
    assert nuisance.gate_id in report.failed_gate_ids
    assert "accepted" not in {item.name for item in fields(report)}
    assert "passed" not in {item.name for item in fields(nuisance)}


def test_active_count_is_derived_from_sparse_state_not_a_capacity_or_boolean():
    problem = _public_problem()
    reconstruction = AtomisticEditBlindReconstruction1D(
        predicted_selection_total_electrons=(
            problem.selection_observed_total_electrons
        ),
        predicted_audit_total_electrons=np.asarray([[5.0, 4.0, 3.0]]),
        additions=_measure((0.1, 0.2), (0.3, 0.4)),
        removals=_measure((0.5, 0.6)),
        deformation_parameter_count=11,
        fitted_spatial_dimension=2,
        maximum_dormant_kkt_violation=0.0,
        recovered_host_deformation_rms_A=0.0,
        multistart=ActiveEditMultistartEvidence1D(
            (1.0, 1.0),
            (2.0, 2.0),
            (1.0, 1.0),
            (0.0, 0.0),
            0,
            "identifiable",
        ),
        physical_metrics=PhysicalAdmissibilityMetrics1D(0.0, 0.0),
    )
    assert reconstruction.active_parameter_count == 11 + 1 + 3 * 2
    assert "active_parameter_count" not in {
        item.name for item in fields(reconstruction)
    }


def test_resolution_aware_unbalanced_transport_is_permutation_invariant():
    truth = _measure((0.0, 0.0), (2.0, 0.0), masses=(0.75, 1.25))
    permuted = _measure((2.0, 0.0), (0.0, 0.0), masses=(1.25, 0.75))
    exact = resolution_aware_mass_transport_metrics_1d(
        truth, permuted, resolution_A=(1.0, 2.0)
    )
    assert exact.normalized_transport_cost == pytest.approx(0.0, abs=1e-12)
    assert exact.relative_total_mass_error == pytest.approx(0.0)
    assert exact.matched_mass == pytest.approx(2.0)

    displaced = _measure((0.5, 0.0), (2.0, 1.0), masses=(0.75, 1.25))
    shifted = resolution_aware_mass_transport_metrics_1d(
        truth, displaced, resolution_A=(1.0, 2.0)
    )
    assert shifted.normalized_transport_cost > 0.0
    assert shifted.normalized_transport_cost < 1.0
    assert shifted.resolution_normalized_rms_displacement == pytest.approx(0.5)


def test_report_retains_held_out_metrics_and_resolution_aware_transport_inputs():
    observability = ObservabilityEvidence1D(
        observability_rule_sha256=_contract().observability_rule_sha256,
        resolution_A=(1.0, 1.0),
    )

    def callback(problem):
        return _reconstruction(problem, observability=observability)

    report = run_atomistic_edit_blind_benchmarks_1d(
        _cases(),
        {
            AtomisticEditAblationArm1D.COUNT_AND_EDIT: callback,
            AtomisticEditAblationArm1D.LEVEL1_PHYSICAL: callback,
        },
        _policy(),
    )
    vacancy = next(
        item
        for item in report.case_reports
        if item.case_role is AtomisticEditBlindCaseRole1D.ONE_VACANCY
        and item.ablation is AtomisticEditAblationArm1D.LEVEL1_PHYSICAL
    )
    assert vacancy.held_out_count_metrics is not None
    assert vacancy.held_out_count_metrics.poisson_deviance == pytest.approx(0.0)
    assert vacancy.active_parameter_count == 12
    assert vacancy.audit_truth_removals.centre_count == 1
    assert vacancy.transport_resolution_A == (1.0, 1.0)
    assert vacancy.removal_transport is not None
    assert vacancy.removal_transport.truth_total_mass == pytest.approx(1.0)
    assert vacancy.removal_transport.estimate_total_mass == pytest.approx(0.0)
    assert vacancy.removal_transport.normalized_transport_cost == pytest.approx(1.0)
