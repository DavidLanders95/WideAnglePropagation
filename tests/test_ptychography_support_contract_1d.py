"""Focused tests for geometry-bound lattice material-support contracts."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from wide_angle_propagation.ptychography_support_contract_1d import (
    LatticeSiteParameterCounts1D,
    LatticeSiteRole1D,
    classify_lattice_site_support_1d,
    lattice_site_support_contract_id_1d,
    validate_lattice_site_support_contract_1d,
)


def _geometry():
    target = np.zeros((6, 7), dtype=bool)
    target[2:4, 2:4] = True
    forward = np.zeros_like(target)
    forward[1:5, 1:6] = True
    coordinates = np.asarray(
        [
            [2.0, 2.0],
            [1.0, 0.0],
            [5.0, 6.0],
            [0.0, 5.0],
        ]
    )
    centers = np.asarray([[2, 2], [1, 0], [5, 6], [0, 5]])
    # Site 1 has its center outside the forward mask, but its complete padded
    # footprint reaches a forward pixel. Site 3 is an explicitly fixed site
    # whose footprint also reaches the forward mask.
    starts = np.asarray([[2, 2], [1, 0], [5, 6], [0, 5]])
    shapes = np.asarray([[1, 1], [2, 3], [1, 1], [2, 2]])
    return coordinates, centers, starts, shapes, target, forward


def _contract(**overrides):
    coordinates, centers, starts, shapes, target, forward = _geometry()
    arguments = {
        "all_site_coordinates": coordinates,
        "site_center_indices": centers,
        "site_patch_starts": starts,
        "site_patch_shapes": shapes,
        "target_pixel_mask": target,
        "forward_pixel_mask": forward,
        "known_fixed_site_mask": np.asarray([False, False, False, True]),
        "fixed_material_provenance_id": "synthetic-pristine-exterior-v1",
        "excluded_probe_power": 1e-5,
        "atomic_template_cutoff_A": 6.0,
        "maximum_displacement_A": 0.5,
        "displacement_control_shape": (2, 3, 2),
        "removed_displacement_dof": 2,
        "registration_parameter_count": 2,
        "maximum_nuisance_sites": 8,
        "maximum_specimen_parameters": 64,
    }
    arguments.update(overrides)
    return classify_lattice_site_support_1d(**arguments)


def _readonly(value, dtype):
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _with_current_digest(contract, **changes):
    changed = replace(contract, **changes)
    return replace(
        changed,
        contract_id=lattice_site_support_contract_id_1d(changed),
    )


def test_patch_footprints_classify_roles_and_exact_parameter_counts():
    contract = _contract()
    np.testing.assert_array_equal(
        contract.site_role_codes,
        [
            LatticeSiteRole1D.TARGET,
            LatticeSiteRole1D.NUISANCE,
            LatticeSiteRole1D.BELOW_INTERACTION_BUDGET,
            LatticeSiteRole1D.FIXED_KNOWN,
        ],
    )
    np.testing.assert_array_equal(contract.modeled_site_indices, [0, 1])
    np.testing.assert_array_equal(contract.target_site_indices, [0])
    np.testing.assert_array_equal(contract.nuisance_site_indices, [1])
    assert contract.forward_relevant_mask[1]
    assert not contract.forward_pixel_mask[1, 0]
    # The nuisance patch reaches into the chosen target pixels, but its center
    # remains outside and must not become reportable structure.
    assert contract.nuisance_influence_mask[2, 2]
    assert contract.target_influence_mask[2, 2]
    assert contract.strict_requirements_satisfied

    counts = contract.parameter_counts
    assert counts == LatticeSiteParameterCounts1D(
        target_vacancy_parameters=1,
        nuisance_vacancy_parameters=1,
        displacement_control_parameters=12,
        removed_displacement_dof=2,
        residual_displacement_control_dof=10,
        registration_parameters=2,
        total_specimen_parameters=14,
    )
    assert dict(contract.parameter_count_metadata) == {
        "target_sites": 1,
        "nuisance_sites": 1,
        "fixed_known_sites": 1,
        "below_interaction_budget_sites": 1,
        "unresolved_sites": 0,
        "modeled_sites": 2,
        "target_vacancy_parameters": 1,
        "nuisance_vacancy_parameters": 1,
        "displacement_control_parameters": 12,
        "removed_displacement_dof": 2,
        "residual_displacement_control_dof": 10,
        "registration_parameters": 2,
        "total_specimen_parameters": 14,
    }


def test_contract_is_frozen_and_owns_read_only_canonical_arrays():
    coordinates, centers, starts, shapes, target, forward = _geometry()
    contract = _contract()
    coordinates[:] = -100.0
    centers[:] = -100
    starts[:] = -100
    shapes[:] = 100
    target[:] = False
    forward[:] = False

    np.testing.assert_array_equal(contract.all_site_coordinates[0], [2.0, 2.0])
    assert contract.target_pixel_mask[2, 2]
    for value in (
        contract.all_site_coordinates,
        contract.site_center_indices,
        contract.site_patch_starts,
        contract.site_patch_shapes,
        contract.target_pixel_mask,
        contract.forward_pixel_mask,
        contract.site_role_codes,
        contract.modeled_site_indices,
    ):
        assert value.flags.c_contiguous
        assert not value.flags.writeable
    with pytest.raises(ValueError):
        contract.site_role_codes[0] = LatticeSiteRole1D.NUISANCE
    with pytest.raises(FrozenInstanceError):
        contract.maximum_nuisance_sites = 4


def test_digest_is_deterministic_and_binds_arrays_options_counts_and_provenance():
    first = _contract()
    repeated = _contract()
    assert first.contract_id == repeated.contract_id
    assert first.contract_id == lattice_site_support_contract_id_1d(first)
    assert len(first.contract_id) == 64

    coordinates, centers, starts, shapes, target, forward = _geometry()
    coordinates = coordinates.copy()
    coordinates[0, 0] += 0.125
    changed_coordinate = _contract(all_site_coordinates=coordinates)
    changed_provenance = _contract(
        fixed_material_provenance_id="another-provenance"
    )
    changed_budget = _contract(maximum_nuisance_sites=9)
    changed_controls = _contract(displacement_control_shape=(2, 4, 2))
    changed_mask = target.copy()
    changed_mask[2, 4] = True
    changed_forward = forward.copy()
    changed_forward[2, 4] = True
    changed_geometry = _contract(
        target_pixel_mask=changed_mask,
        forward_pixel_mask=changed_forward,
    )
    identifiers = {
        first.contract_id,
        changed_coordinate.contract_id,
        changed_provenance.contract_id,
        changed_budget.contract_id,
        changed_controls.contract_id,
        changed_geometry.contract_id,
    }
    assert len(identifiers) == 6

    stale = replace(first, maximum_nuisance_sites=9)
    with pytest.raises(ValueError, match="contract_id"):
        validate_lattice_site_support_contract_1d(stale)


def test_strict_mode_rejects_unresolved_sites_and_reports_indices():
    with pytest.raises(ValueError, match="UNRESOLVED"):
        _contract(
            exterior_policy="leave_unresolved",
            known_fixed_site_mask=np.zeros(4, dtype=bool),
            fixed_material_provenance_id=None,
        )
    exploratory = _contract(
        exterior_policy="leave_unresolved",
        known_fixed_site_mask=np.zeros(4, dtype=bool),
        fixed_material_provenance_id=None,
        strict=False,
    )
    np.testing.assert_array_equal(
        np.flatnonzero(
            exploratory.site_role_codes == LatticeSiteRole1D.UNRESOLVED
        ),
        [1, 3],
    )
    assert not exploratory.strict_requirements_satisfied
    with pytest.raises(ValueError, match="UNRESOLVED"):
        validate_lattice_site_support_contract_1d(exploratory, strict=True)


def test_strict_mode_requires_fixed_provenance_without_treating_it_as_proof():
    with pytest.raises(ValueError, match="provenance"):
        _contract(fixed_material_provenance_id=None)
    exploratory = _contract(
        fixed_material_provenance_id=None,
        strict=False,
    )
    assert not exploratory.strict_requirements_satisfied
    assert exploratory.site_role_codes[3] == LatticeSiteRole1D.FIXED_KNOWN


def test_strict_mode_rejects_an_empty_reportable_target():
    empty_target = np.zeros((6, 7), dtype=bool)
    with pytest.raises(ValueError, match="at least one TARGET"):
        _contract(target_pixel_mask=empty_target)
    exploratory = _contract(target_pixel_mask=empty_target, strict=False)
    assert exploratory.parameter_counts.target_vacancy_parameters == 0
    assert not exploratory.strict_requirements_satisfied


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"maximum_nuisance_sites": 0}, "nuisance-site budget exceeded"),
        ({"maximum_specimen_parameters": 13}, "parameter budget exceeded"),
    ],
)
def test_resource_budgets_fail_without_silent_truncation(overrides, message):
    with pytest.raises(ValueError, match=message):
        _contract(**overrides)
    exploratory = _contract(strict=False, **overrides)
    np.testing.assert_array_equal(exploratory.modeled_site_indices, [0, 1])
    assert exploratory.parameter_counts.total_specimen_parameters == 14
    assert not exploratory.strict_requirements_satisfied


def test_input_validation_rejects_nonboolean_or_non_nested_masks():
    coordinates, centers, starts, shapes, target, forward = _geometry()
    with pytest.raises(TypeError, match="Boolean"):
        _contract(target_pixel_mask=target.astype(np.int8))
    bad_target = target.copy()
    bad_target[0, 0] = True
    with pytest.raises(ValueError, match="subset"):
        _contract(target_pixel_mask=bad_target)
    with pytest.raises(ValueError, match="positive"):
        bad_shapes = shapes.copy()
        bad_shapes[0, 0] = 0
        classify_lattice_site_support_1d(
            coordinates,
            centers,
            starts,
            bad_shapes,
            target,
            forward,
            excluded_probe_power=1e-5,
            atomic_template_cutoff_A=6.0,
            maximum_displacement_A=0.5,
        )


def test_semantic_validation_rejects_role_and_count_fabrication_even_if_rehashed():
    contract = _contract()
    roles = np.asarray(contract.site_role_codes).copy()
    roles[1] = LatticeSiteRole1D.BELOW_INTERACTION_BUDGET
    fabricated_role = _with_current_digest(
        contract,
        site_role_codes=_readonly(roles, np.int8),
    )
    with pytest.raises(ValueError, match="forward-relevant"):
        validate_lattice_site_support_contract_1d(fabricated_role)

    fabricated_counts = _with_current_digest(
        contract,
        parameter_counts=replace(
            contract.parameter_counts,
            total_specimen_parameters=13,
        ),
    )
    with pytest.raises(ValueError, match="parameter counts"):
        validate_lattice_site_support_contract_1d(fabricated_counts)


def test_clipped_boundary_footprints_are_conservative_and_never_wrap():
    target = np.zeros((3, 4), dtype=bool)
    target[0, 0] = True
    forward = target.copy()
    forward[2, 3] = True
    contract = classify_lattice_site_support_1d(
        [[0.0, 0.0], [2.0, 3.0], [-2.0, -2.0]],
        [[0, 0], [2, 3], [-4, -4]],
        [[-2, -2], [2, 3], [-5, -5]],
        [[3, 3], [3, 3], [2, 2]],
        target,
        forward,
        excluded_probe_power=1e-6,
        atomic_template_cutoff_A=4.0,
        maximum_displacement_A=0.5,
        maximum_nuisance_sites=4,
        maximum_specimen_parameters=8,
    )
    np.testing.assert_array_equal(
        contract.site_role_codes,
        [
            LatticeSiteRole1D.TARGET,
            LatticeSiteRole1D.NUISANCE,
            LatticeSiteRole1D.BELOW_INTERACTION_BUDGET,
        ],
    )
    assert contract.target_influence_mask[0, 0]
    assert contract.nuisance_influence_mask[2, 3]
    assert np.count_nonzero(contract.target_influence_mask) == 1
    assert np.count_nonzero(contract.nuisance_influence_mask) == 1
