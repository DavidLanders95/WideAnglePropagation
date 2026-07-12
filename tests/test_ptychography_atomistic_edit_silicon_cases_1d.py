"""Focused CPU gates for the physical AE-3 silicon case/adapter layer."""

from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.ptychography_atomistic_edit_1d import (  # noqa: E402
    AtomisticEditOptions1D,
)
from wide_angle_propagation.ptychography_atomistic_edit_benchmarks_1d import (  # noqa: E402
    AE3_BLIND_CASE_CATALOG_1D,
    AtomisticEditAblationArm1D,
    AtomisticEditBlindCaseRole1D,
    atomistic_edit_public_problem_schema_digest_1d,
)
from wide_angle_propagation.ptychography_atomistic_edit_silicon_cases_1d import (  # noqa: E402
    make_atomistic_edit_blind_count_selection_contract_1d,
    make_silicon_atomistic_edit_blind_cases_1d,
    make_silicon_atomistic_edit_reconstruction_callback_1d,
    make_silicon_atomistic_edit_reconstruction_callbacks_1d,
)
from wide_angle_propagation.ptychography_atomistic_edit_solver_1d import (  # noqa: E402
    AtomisticEditSolverOptions1D,
)
from wide_angle_propagation.ptychography_atomistic_truth_1d import (  # noqa: E402
    render_direct_atomic_template_1d,
)
from wide_angle_propagation.ptychography_workflow_1d import (  # noqa: E402
    SiliconGlancingConfig1D,
    build_atomistic_edit_discovery_support_1d,
    build_silicon_glancing_experiment_1d,
)
import wide_angle_propagation.ptychography_atomistic_edit_silicon_cases_1d as silicon_cases  # noqa: E402,E501


@pytest.fixture(scope="module")
def silicon_suite():
    experiment = build_silicon_glancing_experiment_1d(
        SiliconGlancingConfig1D(
            beam_waist_A=1.5,
            slab_depth_A=8.0,
            vacuum_above_A=8.0,
            vacuum_below_A=10.0,
            window_length_A=30.0,
            scan_start_A=10.0,
            scan_stop_A=20.0,
            n_scans=6,
            defect_center_s_A=15.0,
            defect_width_sites=2,
            validation_stride=3,
            audit_fraction=0.17,
            audit_blocks=1,
            atomic_template_cutoff_A=None,
            cutoff_check_A=10.0,
            maximum_displacement_A=0.5,
            displacement_control_spacing_A=10.0,
            displacement_control_spacing_u_A=3.0,
        )
    )
    discovery = build_atomistic_edit_discovery_support_1d(
        experiment, surface_envelope_A=(-3.0, 4.0)
    )
    options = AtomisticEditOptions1D(
        max_host_removals=3,
        max_extra_centres=3,
        max_scattering_equivalent_per_centre=2.0,
        minimum_separation_A=1.5,
        expected_rms_host_strain=0.1,
        # Very large penalties make the adapter smoke test an intentionally
        # cheap empty-edit numerical path; they do not enter truth generation.
        edit_penalty_path=(1e12, 1e11),
        discovery_support=discovery,
        enable_material_energy_envelope=False,
    )
    count_contract = make_atomistic_edit_blind_count_selection_contract_1d(
        experiment,
        electrons_per_pattern=2_000.0,
        calibration_id="compact-six-scan-ae3-counts-v1",
        poisson_sample=False,
    )
    cases = make_silicon_atomistic_edit_blind_cases_1d(
        experiment,
        options,
        count_contract,
        private_seeds=tuple(range(101, 109)),
    )
    return experiment, options, count_contract, cases


def _case_map(cases):
    return {case.role: case for case in cases}


def _off_grid(position, experiment):
    index = np.asarray(
        [
            (position[0] - float(experiment.axial_coordinates[0]))
            / experiment.axial_sampling,
            (position[1] - float(experiment.transverse_coordinates[0]))
            / experiment.transverse_sampling,
        ]
    )
    return bool(np.any(np.abs(index - np.rint(index)) > 0.05))


def test_factory_has_exact_catalog_and_identical_truth_free_public_schema(
    silicon_suite,
):
    _, _, count_contract, cases = silicon_suite
    assert tuple(case.role for case in cases) == AE3_BLIND_CASE_CATALOG_1D
    assert len(cases) == 8
    assert len({case.public_problem.contract for case in cases}) == 1
    assert len(
        {
            atomistic_edit_public_problem_schema_digest_1d(case.public_problem)
            for case in cases
        }
    ) == 1
    expected_selection_rows = len(count_contract.training_indices) + len(
        count_contract.validation_indices
    )
    for case in cases:
        problem = case.public_problem
        assert problem.selection_observed_total_electrons.shape[0] == (
            expected_selection_rows
        )
        assert problem.audit_prediction_shape[0] == len(count_contract.audit_indices)
        assert not problem.public_arrays
        assert not problem.public_scalars
        assert problem.selection_observed_total_electrons.flags.writeable is False
    assert set(np.asarray(count_contract.selection_indices)).isdisjoint(
        set(np.asarray(count_contract.audit_indices))
    )


def test_eight_truth_roles_are_physical_and_counts_have_nonzero_deltas(
    silicon_suite,
):
    experiment, _, _, cases = silicon_suite
    by_role = _case_map(cases)
    truth = {role: case.private_truth_factory() for role, case in by_role.items()}

    pristine = truth[AtomisticEditBlindCaseRole1D.PRISTINE_HOST]
    assert pristine.additions.centre_count == 0
    assert pristine.removals.centre_count == 0
    assert pristine.host_deformation_rms_A == 0.0

    vacancy = truth[AtomisticEditBlindCaseRole1D.ONE_VACANCY]
    assert vacancy.removals.centre_count == 1
    host_sites = np.asarray(experiment.lattice_model.site_coordinates)
    assert np.min(
        np.linalg.norm(
            host_sites - np.asarray(vacancy.removals.positions_A)[0], axis=1
        )
    ) < 1e-12

    off_lattice = truth[AtomisticEditBlindCaseRole1D.ONE_OFF_LATTICE_ADDITION]
    assert off_lattice.additions.centre_count == 1
    assert _off_grid(off_lattice.additions.positions_A[0], experiment)

    substitution = truth[AtomisticEditBlindCaseRole1D.ONE_SUBSTITUTION]
    assert substitution.generating_element == "Ge"
    assert substitution.generating_addition_kernel_id != substitution.host_kernel_id
    np.testing.assert_allclose(
        substitution.additions.positions_A,
        substitution.removals.positions_A,
        rtol=0.0,
        atol=1e-12,
    )

    cluster = truth[AtomisticEditBlindCaseRole1D.IRREGULAR_FINITE_CLUSTER]
    assert cluster.additions.centre_count == 3
    assert all(_off_grid(position, experiment) for position in cluster.additions.positions_A)
    positions = np.asarray(cluster.additions.positions_A)
    pair_distances = sorted(
        np.linalg.norm(positions[i] - positions[j])
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    )
    assert np.ptp(pair_distances) > 1e-3

    metastable = truth[AtomisticEditBlindCaseRole1D.METASTABLE_DEFECT]
    assert metastable.host_deformation_rms_A > 0.0
    assert metastable.additions.centre_count == metastable.removals.centre_count == 0

    nuisance = truth[AtomisticEditBlindCaseRole1D.NUISANCE_ONLY_MISMATCH]
    assert nuisance.mismatch_cause is not None
    assert nuisance.additions.centre_count == nuisance.removals.centre_count == 0
    assert nuisance.host_deformation_rms_A == 0.0

    unresolved = truth[AtomisticEditBlindCaseRole1D.AXIALLY_UNRESOLVED_ADDITION]
    assert unresolved.additions.centre_count == 1
    assert unresolved.axial_depth_uncertainty_A > unresolved.slice_thickness_A
    assert _off_grid(unresolved.additions.positions_A[0], experiment)

    pristine_counts = np.asarray(
        by_role[
            AtomisticEditBlindCaseRole1D.PRISTINE_HOST
        ].public_problem.selection_observed_total_electrons
    )
    for role in AE3_BLIND_CASE_CATALOG_1D[1:]:
        delta = np.linalg.norm(
            np.asarray(by_role[role].public_problem.selection_observed_total_electrons)
            - pristine_counts
        )
        assert delta > 1e-8, f"{role.value} produced no measurable physical delta"


def test_substitution_uses_a_different_direct_ge_kernel(silicon_suite):
    experiment, _, _, cases = silicon_suite
    substitution = _case_map(cases)[
        AtomisticEditBlindCaseRole1D.ONE_SUBSTITUTION
    ].private_truth_factory()
    position = np.asarray(substitution.additions.positions_A)[0]
    index = np.asarray(
        [
            (position[0] - float(experiment.axial_coordinates[0]))
            / experiment.axial_sampling,
            (position[1] - float(experiment.transverse_coordinates[0]))
            / experiment.transverse_sampling,
        ]
    )
    anchor = np.floor(index + 0.5).astype(int)
    offset = (
        float(position[0] - np.asarray(experiment.axial_coordinates)[anchor[0]]),
        float(position[1] - np.asarray(experiment.transverse_coordinates)[anchor[1]]),
    )
    render_arguments = {
        "sampling_s_A": experiment.axial_sampling,
        "sampling_u_A": experiment.transverse_sampling,
        "options": experiment.independent_kirkland_template.options,
        "fractional_offset_A": offset,
    }
    silicon = render_direct_atomic_template_1d("Si", **render_arguments)
    germanium = render_direct_atomic_template_1d("Ge", **render_arguments)
    assert silicon.template_id != germanium.template_id
    assert not np.allclose(silicon.values, germanium.values)
    assert germanium.integrated_scattering != pytest.approx(
        silicon.integrated_scattering, rel=1e-3
    )


def test_private_seed_changes_truth_but_not_schema_and_audit_is_lazy(
    silicon_suite, monkeypatch
):
    experiment, options, count_contract, original_cases = silicon_suite
    streams = []
    integration_methods = []
    original_observed_counts = silicon_cases._observed_counts
    original_accumulate = silicon_cases.accumulate_weighted_atomic_potential_1d

    def observed_counts_spy(*args, stream, **kwargs):
        streams.append(stream)
        return original_observed_counts(*args, stream=stream, **kwargs)

    def accumulate_spy(*args, numerical_options=None, **kwargs):
        integration_methods.append(numerical_options.integration_method)
        return original_accumulate(
            *args, numerical_options=numerical_options, **kwargs
        )

    monkeypatch.setattr(silicon_cases, "_observed_counts", observed_counts_spy)
    monkeypatch.setattr(
        silicon_cases, "accumulate_weighted_atomic_potential_1d", accumulate_spy
    )
    changed_cases = make_silicon_atomistic_edit_blind_cases_1d(
        experiment,
        options,
        count_contract,
        private_seeds=tuple(range(301, 309)),
    )
    assert streams == [1] * 8
    assert integration_methods
    assert set(integration_methods) == {"adaptive_factorized_cubature"}
    changed_cases[0].private_truth_factory()
    assert streams == [1] * 8
    first_audit = changed_cases[0].private_audit_factory()
    assert streams == [1] * 8 + [2]
    second_audit = changed_cases[0].private_audit_factory()
    np.testing.assert_array_equal(
        first_audit.observed_total_electrons,
        second_audit.observed_total_electrons,
    )

    original_schema = {
        atomistic_edit_public_problem_schema_digest_1d(case.public_problem)
        for case in original_cases
    }
    changed_schema = {
        atomistic_edit_public_problem_schema_digest_1d(case.public_problem)
        for case in changed_cases
    }
    assert original_schema == changed_schema
    assert original_cases[0].public_problem.contract == (
        changed_cases[0].public_problem.contract
    )
    original_position = _case_map(original_cases)[
        AtomisticEditBlindCaseRole1D.ONE_OFF_LATTICE_ADDITION
    ].private_truth_factory().additions.positions_A
    changed_position = _case_map(changed_cases)[
        AtomisticEditBlindCaseRole1D.ONE_OFF_LATTICE_ADDITION
    ].private_truth_factory().additions.positions_A
    assert not np.allclose(original_position, changed_position)


def test_solver_adapter_uses_selection_only_and_fails_closed_on_evidence(
    silicon_suite,
):
    experiment, options, count_contract, cases = silicon_suite
    solver_options = AtomisticEditSolverOptions1D(
        maximum_active_set_iterations=1,
        joint_refinement_updates=0,
        polish_updates=0,
        debias_updates=0,
        proposal_grid_kkt_tolerance=1e30,
        active_projected_gradient_tolerance=1e30,
        debias_projected_gradient_tolerance=1e30,
        seed=77,
    )
    callback = make_silicon_atomistic_edit_reconstruction_callback_1d(
        experiment,
        options,
        count_contract,
        ablation=AtomisticEditAblationArm1D.LEVEL1_PHYSICAL,
        solver_options=solver_options,
        number_of_starts=1,
        initial_host_control_std_A=0.0,
        show_progress=False,
    )
    pristine_problem = _case_map(cases)[
        AtomisticEditBlindCaseRole1D.PRISTINE_HOST
    ].public_problem
    result = callback(pristine_problem)
    assert result.predicted_selection_total_electrons.shape == (
        pristine_problem.selection_observed_total_electrons.shape
    )
    assert result.predicted_audit_total_electrons.shape == (
        pristine_problem.audit_prediction_shape
    )
    assert result.observability is None
    assert result.nuisance_attribution is None
    assert result.archive_evidence is None
    assert result.multistart.ambiguity_disposition == "not_assessed"
    assert result.active_parameter_count == result.deformation_parameter_count

    callbacks = make_silicon_atomistic_edit_reconstruction_callbacks_1d(
        experiment,
        options,
        count_contract,
        solver_options=solver_options,
        number_of_starts=1,
    )
    assert set(callbacks) == {
        AtomisticEditAblationArm1D.COUNT_AND_EDIT,
        AtomisticEditAblationArm1D.LEVEL1_PHYSICAL,
    }
    with pytest.raises(ValueError, match="blocked_not_run"):
        make_silicon_atomistic_edit_reconstruction_callback_1d(
            experiment,
            options,
            count_contract,
            ablation=AtomisticEditAblationArm1D.MATERIAL_ENERGY_ENVELOPE,
            solver_options=solver_options,
        )


def test_count_contract_is_immutable_and_rejects_partition_drift(silicon_suite):
    experiment, _, count_contract, _ = silicon_suite
    assert count_contract.detector_valid_mask.flags.writeable is False
    assert count_contract.electrons_per_pattern.flags.writeable is False
    with pytest.raises(ValueError, match="disagrees with the experiment partition"):
        make_silicon_atomistic_edit_blind_cases_1d(
            experiment,
            # The option value is irrelevant: binding must reject the drift
            # before any private truth is supplied.
            silicon_suite[1],
            replace(
                count_contract,
                training_indices=np.asarray(count_contract.validation_indices),
                validation_indices=np.asarray(count_contract.training_indices),
            ),
            private_seeds=tuple(range(401, 409)),
        )
