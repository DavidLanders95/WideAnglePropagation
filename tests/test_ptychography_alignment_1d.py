"""Truth-isolation and determinism tests for alignment search foundations."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from wide_angle_propagation.ptychography_1d import GlancingScan1D
from wide_angle_propagation.ptychography_alignment_1d import (
    AlignmentCandidateScore1D,
    AlignmentInitializationOptions1D,
    alignment_candidate_catalog_id_1d,
    build_alignment_selection_data_1d,
    canonical_axial_phase_fraction_1d,
    generate_silicon_alignment_candidates_1d,
    geometry_stratified_training_subset_1d,
    refine_silicon_alignment_candidates_1d,
    select_alignment_candidate_1d,
)


def _options(**changes):
    values = {
        "candidates_per_termination": 4,
        "training_screen_scan_count": 2,
        "lattice_scale_bounds": (0.99, 1.01),
        "in_plane_rotation_bounds_rad": (-0.01, 0.02),
        "seed": 7,
    }
    values.update(changes)
    return AlignmentInitializationOptions1D(**values)


def _scan(intensities=None, *, metadata=None, detector_valid_mask=None):
    values = (
        np.arange(32, dtype=float).reshape(8, 4) + 1.0
        if intensities is None
        else np.asarray(intensities)
    )
    return GlancingScan1D(
        intensities=values,
        window_starts=np.arange(8, dtype=np.int32),
        scan_coordinates=np.asarray([4.0, 0.0, 3.0, 1.0, 5.0, 2.0, 6.0, 7.0]),
        detector_angles=np.linspace(-2.0, 2.0, 4),
        metadata={} if metadata is None else metadata,
        detector_valid_mask=detector_valid_mask,
    )


def _selection_data(scan=None):
    return build_alignment_selection_data_1d(
        _scan() if scan is None else scan,
        training_indices=[0, 1, 2, 3],
        validation_indices=[4, 5],
        audit_indices=[6],
        guard_indices=[7],
        training_screen_scan_count=2,
    )


def test_sobol_catalog_is_deterministic_balanced_unique_and_bounded():
    options = _options()
    first = generate_silicon_alignment_candidates_1d(options=options)
    second = generate_silicon_alignment_candidates_1d(options=options)
    assert first == second
    assert len(first) == 8
    assert len({candidate.candidate_id for candidate in first}) == len(first)
    counts = {
        termination: sum(candidate.termination_id == termination for candidate in first)
        for termination in ("si_termination_0", "si_termination_1")
    }
    assert counts == {"si_termination_0": 4, "si_termination_1": 4}
    assert all(0.0 <= candidate.axial_phase_fraction < 1.0 for candidate in first)
    assert all(
        options.in_plane_rotation_bounds_rad[0]
        <= candidate.in_plane_rotation_rad
        <= options.in_plane_rotation_bounds_rad[1]
        for candidate in first
    )
    assert all(
        options.lattice_scale_bounds[0]
        <= candidate.lattice_scale
        <= options.lattice_scale_bounds[1]
        for candidate in first
    )
    assert generate_silicon_alignment_candidates_1d(
        options=_options(seed=8)
    ) != first
    assert alignment_candidate_catalog_id_1d(first, options=options) == (
        alignment_candidate_catalog_id_1d(second, options=options)
    )


def test_phase_is_canonical_modulo_one_and_catalog_rejects_invalid_options():
    assert canonical_axial_phase_fraction_1d(1.25) == pytest.approx(0.25)
    assert canonical_axial_phase_fraction_1d(-0.25) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="power of two"):
        generate_silicon_alignment_candidates_1d(
            options=_options(candidates_per_termination=3)
        )
    with pytest.raises(ValueError, match="unique"):
        generate_silicon_alignment_candidates_1d(
            ("same", "same"), options=_options()
        )


def test_fine_stencil_is_deterministic_bounded_and_parent_bound():
    options = _options(
        fine_phase_step_fraction=0.1,
        fine_rotation_step_rad=0.002,
        fine_log_scale_step=1e-3,
    )
    parents = generate_silicon_alignment_candidates_1d(options=options)[:2]
    first = refine_silicon_alignment_candidates_1d(parents, options=options)
    second = refine_silicon_alignment_candidates_1d(parents, options=options)
    assert first == second
    assert first
    assert len({candidate.candidate_id for candidate in first}) == len(first)
    parent_ids = {parent.candidate_id for parent in parents}
    assert all(candidate.parent_candidate_id in parent_ids for candidate in first)
    assert all(candidate.refinement_level == 1 for candidate in first)
    assert all(0.0 <= candidate.axial_phase_fraction < 1.0 for candidate in first)
    assert all(
        options.in_plane_rotation_bounds_rad[0]
        <= candidate.in_plane_rotation_rad
        <= options.in_plane_rotation_bounds_rad[1]
        for candidate in first
    )
    assert all(
        options.lattice_scale_bounds[0]
        <= candidate.lattice_scale
        <= options.lattice_scale_bounds[1]
        for candidate in first
    )


def test_geometry_subset_uses_only_training_coordinates():
    training = np.asarray([4, 0, 2, 1], dtype=np.int64)
    coordinates = np.asarray([0.0, 1.0, 2.0, 30.0, 4.0, 50.0])
    selected = geometry_stratified_training_subset_1d(training, coordinates, 2)
    np.testing.assert_array_equal(selected, [1, 4])
    changed = coordinates.copy()
    changed[[3, 5]] = [-1e9, 1e9]
    np.testing.assert_array_equal(
        geometry_stratified_training_subset_1d(training, changed, 2),
        selected,
    )


def test_selection_boundary_omits_audit_and_guard_values_and_ignores_metadata():
    baseline = _selection_data(_scan(metadata={"dataset_case": "secret_truth_case"}))
    assert baseline.metadata["audit_observations_stored"] is False
    assert baseline.metadata["scan_metadata_used"] is False
    assert set(np.asarray(baseline.source_scan_indices)).isdisjoint({6, 7})
    assert not np.asarray(baseline.intensities).flags.writeable

    changed = np.asarray(_scan().intensities).copy()
    changed[6] = np.nan
    changed[7] = 1e200
    changed_data = _selection_data(_scan(changed, metadata={"truth": "ignored"}))
    assert changed_data.selection_data_id == baseline.selection_data_id
    np.testing.assert_array_equal(changed_data.intensities, baseline.intensities)

    truth_container = SimpleNamespace(
        scan=_scan(),
        truth_potential=np.ones((2, 2)),
    )
    with pytest.raises(TypeError, match="truth-free GlancingScan1D"):
        _selection_data(truth_container)


def test_selection_data_digest_responds_only_to_copied_observations():
    baseline_scan = _scan()
    baseline = _selection_data(baseline_scan)
    changed_training = np.asarray(baseline_scan.intensities).copy()
    changed_training[3, 0] += 1.0
    assert _selection_data(_scan(changed_training)).selection_data_id != (
        baseline.selection_data_id
    )
    changed_validation = np.asarray(baseline_scan.intensities).copy()
    changed_validation[4, 0] += 1.0
    assert _selection_data(_scan(changed_validation)).selection_data_id != (
        baseline.selection_data_id
    )

    valid = np.ones_like(baseline_scan.intensities, dtype=bool)
    valid[6] = False
    sentinel = np.asarray(baseline_scan.intensities).copy()
    sentinel[6] = np.nan
    masked_audit = _selection_data(_scan(sentinel, detector_valid_mask=valid))
    all_valid = _selection_data(
        _scan(baseline_scan.intensities, detector_valid_mask=np.ones_like(valid))
    )
    assert masked_audit.selection_data_id == all_valid.selection_data_id
    np.testing.assert_array_equal(masked_audit.intensities, all_valid.intensities)


def _score(candidate, per_scan, *, training_loss=0.5, model_suffix="model"):
    per_scan = np.asarray(per_scan, dtype=float)
    return AlignmentCandidateScore1D(
        candidate=candidate,
        training_screen_loss=training_loss,
        validation_loss=float(np.mean(per_scan)),
        validation_loss_per_scan=per_scan,
        candidate_model_id=f"{model_suffix}-{candidate.candidate_id}",
    )


def test_paired_validation_selection_preserves_equivalence_and_never_claims_trust():
    options = _options(
        validation_absolute_band=1e-12,
        validation_relative_band=1e-3,
        validation_equivalence_z=1.96,
    )
    candidates = generate_silicon_alignment_candidates_1d(options=options)[:3]
    scores = (
        _score(candidates[0], [1.0, 1.0, 1.0]),
        _score(candidates[1], [1.0005, 1.0004, 1.0006]),
        _score(candidates[2], [1.1, 1.1, 1.1]),
    )
    catalog_id = alignment_candidate_catalog_id_1d(
        generate_silicon_alignment_candidates_1d(options=options),
        options=options,
    )
    summary = select_alignment_candidate_1d(
        scores,
        selection_data_id="selection-data-v1",
        candidate_catalog_id=catalog_id,
        options=options,
    )
    assert summary.minimum_loss_candidate_id == candidates[0].candidate_id
    assert summary.selected_candidate_id == candidates[0].candidate_id
    assert set(summary.equivalent_candidate_ids) == {
        candidates[0].candidate_id,
        candidates[1].candidate_id,
    }
    assert summary.unique_selection is False
    assert summary.structurally_trusted is False
    assert summary.metadata["audit_used_for_selection"] is False
    assert len(summary.alignment_selection_id) == 64

    tampered_mean = replace(scores[0], validation_loss=2.0)
    with pytest.raises(ValueError, match="must equal mean"):
        select_alignment_candidate_1d(
            (tampered_mean,),
            selection_data_id="selection-data-v1",
            candidate_catalog_id=catalog_id,
            options=options,
        )
