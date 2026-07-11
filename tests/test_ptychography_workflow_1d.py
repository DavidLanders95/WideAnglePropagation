"""Integration tests for the compact glancing-ptychography workflow API."""

from dataclasses import replace

import matplotlib
import numpy as np
import pytest


matplotlib.use("Agg")
jax = pytest.importorskip("jax")
pytest.importorskip("abtem")
pytest.importorskip("optax")
PILImage = pytest.importorskip("PIL.Image")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.ptychography_workflow_1d import (  # noqa: E402
    GlancingDataset1D,
    ReconstructionOptions1D,
    SiliconGlancingConfig1D,
    build_silicon_alignment_prior_1d,
    build_silicon_alignment_problem_1d,
    build_silicon_glancing_experiment_1d,
    gaussian_interaction_region_1d,
    plot_experiment_overview_1d,
    plot_lattice_reconstruction_1d,
    plot_lattice_sensitivity_screen_1d,
    plot_reconstruction_comparison_1d,
    reconstruct_experiment_1d,
    reconstruct_lattice_multistart_experiment_1d,
    reconstruction_metrics_1d,
    save_experiment_results_1d,
    save_lattice_reconstruction_gif_1d,
    screen_lattice_reconstruction_sensitivity_1d,
    simulate_experiment_1d,
    stratified_scan_partition_1d,
)
from wide_angle_propagation.ptychography_diagnostics_1d import (  # noqa: E402
    PoissonCountingModel1D,
    SensitivityScreenOptions1D,
)
from wide_angle_propagation.ptychography_1d import (  # noqa: E402
    GlancingScan1D,
    LatticeSiteReconstruction1D,
    PtychographyMeasurement1D,
    PtychographyObjective1D,
    ptychography_expected_signal_electrons_1d,
    render_lattice_site_potential_1d,
    simulate_glancing_scan_1d,
)
from wide_angle_propagation.ptychography_alignment_1d import (  # noqa: E402
    AlignmentInitializationOptions1D,
    generate_silicon_alignment_candidates_1d,
    initialize_silicon_alignment_1d,
    load_silicon_alignment_initialization_1d,
    prepare_aligned_lattice_site_reconstruction_1d,
    rebuild_silicon_alignment_candidate_1d,
    save_silicon_alignment_initialization_1d,
)
from wide_angle_propagation.ptychography_ensemble_1d import (  # noqa: E402
    MultistartOptions1D,
)
import wide_angle_propagation.ptychography_workflow_1d as workflow  # noqa: E402


def test_gaussian_interaction_region_uses_coordinates_angle_and_overlap():
    s_A = np.arange(7.0)
    u_A = np.arange(-4.0, 2.0)
    region = gaussian_interaction_region_1d(
        s_A,
        u_A,
        np.asarray([2.0, 4.0]),
        beam_waist_A=1.0,
        beam_tilt_rad=-np.pi / 4.0,
        slab_bottom_A=-4.0,
        intensity_threshold=np.exp(-4.0),
        minimum_scan_coverage=2,
    )
    assert region.radius_A == pytest.approx(2.0)
    assert float(region.peak_relative_intensity[3, 3]) == pytest.approx(1.0)
    assert int(region.scan_coverage[3, 3]) == 2
    assert bool(region.reconstruction_mask[3, 3])
    assert np.all(
        np.asarray(region.reconstruction_mask) <= np.asarray(region.forward_mask)
    )
    assert region.metadata["mutable_support"] == (
        "nominal_geometry_training_scans_only"
    )


def test_mutable_interaction_region_uses_only_nominal_training_beams():
    region = gaussian_interaction_region_1d(
        np.asarray([1.0, 3.0, 5.0]),
        np.asarray([-1.0]),
        np.asarray([0.0, 4.0, 8.0]),
        beam_waist_A=1.0,
        beam_tilt_rad=-np.pi / 4.0,
        slab_bottom_A=-2.0,
        intensity_threshold=np.exp(-1.0),
        minimum_scan_coverage=1,
        beam_position_uncertainty_A=2.0,
        mutable_scan_indices=np.asarray([0, 2]),
    )

    # The middle pixel is reached only after uncertainty expansion. The final
    # pixel lies exactly on held-out scan 1, but outside the nominal support of
    # either mutable scan. Neither becomes an independently mutable pixel.
    assert bool(region.forward_mask[1, 0])
    assert not bool(region.nominal_forward_mask[1, 0])
    assert bool(region.nominal_forward_mask[2, 0])
    assert int(region.scan_coverage[2, 0]) == 0
    assert not bool(region.reconstruction_mask[1, 0])
    assert not bool(region.reconstruction_mask[2, 0])
    assert bool(region.reconstruction_mask[0, 0])
    assert region.radius_A > region.nominal_radius_A
    assert not region.metadata["geometry_uncertainty_expands_mutable_support"]


def test_angle_uncertainty_uses_spatially_local_ray_envelope():
    s_A = np.linspace(0.0, 40.0, 41)
    u_A = np.linspace(-20.0, 0.0, 21)
    scans_A = np.asarray([10.0, 20.0, 30.0])
    tilt = -np.deg2rad(20.0)
    angle_bound = np.deg2rad(4.0)
    region = gaussian_interaction_region_1d(
        s_A,
        u_A,
        scans_A,
        beam_waist_A=1.5,
        beam_tilt_rad=tilt,
        slab_bottom_A=-20.0,
        excluded_probe_power=1e-3,
        minimum_scan_coverage=1,
        beam_position_uncertainty_A=0.5,
        beam_angle_uncertainty_rad=angle_bound,
    )
    nominal = gaussian_interaction_region_1d(
        s_A,
        u_A,
        scans_A,
        beam_waist_A=1.5,
        beam_tilt_rad=tilt,
        slab_bottom_A=-20.0,
        excluded_probe_power=1e-3,
        minimum_scan_coverage=1,
    )
    np.testing.assert_array_equal(
        region.reconstruction_mask,
        nominal.reconstruction_mask,
    )

    grid_s, grid_u = np.meshgrid(s_A, u_A, indexing="ij")
    support_radius = region.nominal_radius_A + 0.5
    brute_force = np.zeros((len(s_A), len(u_A)), dtype=bool)
    for bounded_tilt in np.linspace(
        tilt - angle_bound,
        tilt + angle_bound,
        2001,
    ):
        for landing_A in scans_A:
            perpendicular_distance = np.abs(
                grid_u * np.cos(bounded_tilt)
                - (grid_s - landing_A) * np.sin(bounded_tilt)
            )
            brute_force |= perpendicular_distance <= support_radius
    np.testing.assert_array_equal(region.forward_mask, brute_force)

    maximum_path_A = max(
        abs(s_A[0] - scans_A[-1]),
        abs(s_A[-1] - scans_A[0]),
    )
    old_global_radius = (
        region.nominal_radius_A
        + 0.5
        + maximum_path_A * np.tan(angle_bound)
    )
    nominal_distance = np.min(
        [
            np.abs(
                grid_u * np.cos(tilt)
                - (grid_s - landing_A) * np.sin(tilt)
            )
            for landing_A in scans_A
        ],
        axis=0,
    )
    old_global_mask = nominal_distance <= old_global_radius
    assert np.all(np.asarray(region.forward_mask) <= old_global_mask)
    assert np.count_nonzero(region.forward_mask) < np.count_nonzero(
        old_global_mask
    )
    assert region.metadata["uncertainty_expansion"] == (
        "spatially_local_bounded_ray_envelope"
    )

    with pytest.raises(ValueError, match="surface-parallel"):
        gaussian_interaction_region_1d(
            s_A,
            u_A,
            scans_A,
            beam_waist_A=1.5,
            beam_tilt_rad=tilt,
            slab_bottom_A=-20.0,
            beam_angle_uncertainty_rad=abs(tilt),
        )


def test_stratified_scan_partition_is_disjoint_distributed_and_guarded():
    partition = stratified_scan_partition_1d(
        30,
        validation_stride=5,
        audit_fraction=0.2,
        audit_blocks=3,
        audit_guard_scans=1,
    )
    groups = [
        np.asarray(partition.training_indices),
        np.asarray(partition.validation_indices),
        np.asarray(partition.audit_indices),
        np.asarray(partition.guard_indices),
    ]
    for index, first in enumerate(groups):
        for second in groups[index + 1 :]:
            assert not np.intersect1d(first, second).size
    np.testing.assert_array_equal(
        np.sort(np.concatenate(groups)), np.arange(30)
    )
    assert len(partition.audit_indices) == 6
    assert np.ptp(np.asarray(partition.audit_indices)) >= 15
    assert partition.metadata["audit_blocks_used"] == 3


def test_stratified_partition_reports_touching_blocks_as_one_effective_block():
    partition = stratified_scan_partition_1d(
        4,
        validation_stride=2,
        audit_fraction=0.49,
        audit_blocks=2,
    )
    np.testing.assert_array_equal(partition.audit_indices, np.asarray([1, 2]))
    assert partition.metadata["audit_blocks_placed"] == 2
    assert partition.metadata["audit_blocks_used"] == 1
    assert partition.metadata["audit_block_bounds_stop_exclusive"] == [(1, 3)]


def test_exterior_material_policy_fails_closed_without_provenance():
    with pytest.raises(ValueError, match="requires.*provenance"):
        build_silicon_glancing_experiment_1d(
            SiliconGlancingConfig1D(
                exterior_material_policy="known_fixed",
            )
        )
    with pytest.raises(ValueError, match="deprecated"):
        build_silicon_glancing_experiment_1d(
            SiliconGlancingConfig1D(
                fixed_exterior_assumption="known_pristine",
            )
        )


@pytest.fixture(scope="module")
def tiny_experiment():
    config = SiliconGlancingConfig1D(
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
        atomic_template_cutoff_A=None,
        cutoff_check_A=10.0,
        maximum_displacement_A=0.5,
        displacement_control_spacing_A=10.0,
        displacement_control_spacing_u_A=3.0,
    )
    return build_silicon_glancing_experiment_1d(config)


def _gif_dataset(experiment, case="strained_surface_defects"):
    detector_shape = (
        len(experiment.window_starts),
        len(experiment.detector_angles),
    )
    scan = GlancingScan1D(
        intensities=np.zeros(detector_shape),
        window_starts=np.asarray(experiment.window_starts),
        scan_coordinates=np.asarray(experiment.scan_coordinates),
        detector_angles=np.asarray(experiment.detector_angles),
        detector_valid_mask=np.ones(detector_shape, dtype=bool),
    )
    return GlancingDataset1D(
        case=case,
        potential=np.asarray(experiment.truth_potentials[case]),
        scan=scan,
        truth_vacancy_fractions=np.asarray(
            experiment.truth_vacancy_fractions[case]
        ),
        truth_displacement_controls=np.asarray(
            experiment.truth_displacement_controls[case]
        ),
        truth_rigid_displacement=np.asarray(
            experiment.truth_rigid_displacements[case]
        ),
        zero_exterior_amplitude_nrmse=0.0,
        template_cutoff_amplitude_nrmse=0.0,
        template_cutoff_max_scan_amplitude_nrmse=0.0,
        template_stress_worst_scan_amplitude_nrmse=0.0,
        template_certified_worst_amplitude_nrmse=0.0,
        kirkland_alternative_amplitude_nrmse=0.0,
        kirkland_alternative_max_scan_amplitude_nrmse=0.0,
    )


def _gif_result(experiment, *, checkpoint_updates=(0,), completed_updates=None):
    model = experiment.lattice_model
    sites = np.asarray(model.site_coordinates)
    n_site = len(sites)
    controls = np.zeros(
        (
            len(model.control_coordinates_s),
            len(model.control_coordinates_u),
            2,
        )
    )
    vacancies = np.zeros(n_site)
    rigid = np.zeros(2)
    potential = np.asarray(
        render_lattice_site_potential_1d(
            model,
            vacancies,
            controls + rigid,
        )
    )
    updates = np.asarray(checkpoint_updates, dtype=np.int32)
    if completed_updates is None:
        completed_updates = int(updates[-1])
    roles = np.asarray(experiment.support_contract.site_role_codes)[
        np.asarray(experiment.support_contract.modeled_site_indices)
    ]
    detector_shape = (
        len(experiment.window_starts),
        len(experiment.detector_angles),
    )
    return LatticeSiteReconstruction1D(
        potential=potential,
        initial_potential=potential,
        vacancy_fractions=vacancies,
        initial_vacancy_fractions=vacancies,
        displacement_controls=controls,
        initial_displacement_controls=controls,
        site_coordinates=sites,
        displaced_site_coordinates=sites,
        control_coordinates_s=np.asarray(model.control_coordinates_s),
        control_coordinates_u=np.asarray(model.control_coordinates_u),
        predicted_intensities=np.zeros(detector_shape),
        measured_intensities=np.zeros(detector_shape),
        window_starts=np.asarray(experiment.window_starts),
        scan_coordinates=np.asarray(experiment.scan_coordinates),
        detector_angles=np.asarray(experiment.detector_angles),
        update_history=np.asarray([0], dtype=np.int32),
        elapsed_time_history=np.asarray([0.0]),
        training_loss_history=np.asarray([1.0]),
        validation_loss_history=np.asarray([1.0]),
        best_update=0,
        completed_updates=completed_updates,
        rigid_displacement=rigid,
        rigid_displacement_history=np.repeat(
            rigid[None, :], len(updates), axis=0
        ),
        checkpoint_updates=updates,
        vacancy_fraction_history=np.repeat(
            vacancies[None, :], len(updates), axis=0
        ),
        displacement_control_history=np.repeat(
            controls[None, ...], len(updates), axis=0
        ),
        metadata={"best_metric": 1.0, "checkpoint_interval": 1},
        site_role_codes=roles,
        support_contract_id=experiment.support_contract.contract_id,
        material_scope_complete=True,
        material_scope_fully_parameterized=True,
    )


def _skip_gif_encoding(monkeypatch):
    from matplotlib.animation import Animation

    def fake_save(animation, *args, **kwargs):
        animation._draw_was_started = True

    monkeypatch.setattr(Animation, "save", fake_save)


@pytest.mark.parametrize("mismatch", ["support_contract", "site_roles"])
def test_gif_rejects_result_model_reporting_scope_mismatch(
    tiny_experiment,
    tmp_path,
    monkeypatch,
    mismatch,
):
    experiment = tiny_experiment
    dataset = _gif_dataset(experiment)
    result = _gif_result(experiment)
    if mismatch == "support_contract":
        result = replace(result, support_contract_id="b" * 64)
    else:
        roles = np.asarray(result.site_role_codes).copy()
        target = np.flatnonzero(
            roles == int(workflow.LatticeSiteRole1D.TARGET)
        )[0]
        nuisance = np.flatnonzero(
            roles == int(workflow.LatticeSiteRole1D.NUISANCE)
        )[0]
        roles[target], roles[nuisance] = roles[nuisance], roles[target]
        result = replace(result, site_role_codes=roles)
    _skip_gif_encoding(monkeypatch)

    with pytest.raises(ValueError, match="support[- ]contract|site roles|TARGET"):
        save_lattice_reconstruction_gif_1d(
            tmp_path / f"{mismatch}.gif",
            experiment,
            dataset,
            result,
        )


def test_gif_rejects_truncated_every_update_checkpoint_history(
    tiny_experiment,
    tmp_path,
    monkeypatch,
):
    experiment = tiny_experiment
    dataset = _gif_dataset(experiment)
    result = _gif_result(
        experiment,
        checkpoint_updates=(0, 1),
        completed_updates=2,
    )
    _skip_gif_encoding(monkeypatch)

    with pytest.raises(
        ValueError,
        match="checkpoint.*completed|complete.*checkpoint|truncated",
    ):
        save_lattice_reconstruction_gif_1d(
            tmp_path / "truncated.gif",
            experiment,
            dataset,
            result,
        )


def test_gif_intersects_explicit_mask_with_computed_target_support(
    tiny_experiment,
    tmp_path,
    monkeypatch,
):
    experiment = tiny_experiment
    dataset = _gif_dataset(experiment)
    result = _gif_result(experiment)
    explicit = np.ones_like(
        np.asarray(experiment.target_lattice_influence_mask),
        dtype=bool,
    )
    target_pixels = np.argwhere(experiment.target_lattice_influence_mask)
    explicit[tuple(target_pixels[len(target_pixels) // 2])] = False
    expected = explicit & np.asarray(experiment.target_lattice_influence_mask)
    captured = {}
    original_view = workflow._update_region_view

    def capture_view(selected_experiment, *, mask=None, margin_A=1.0):
        captured["mask"] = np.asarray(mask).copy()
        return original_view(
            selected_experiment,
            mask=mask,
            margin_A=margin_A,
        )

    monkeypatch.setattr(workflow, "_update_region_view", capture_view)
    _skip_gif_encoding(monkeypatch)

    save_lattice_reconstruction_gif_1d(
        tmp_path / "intersected.gif",
        experiment,
        dataset,
        result,
        lattice_influence_mask=explicit,
    )

    np.testing.assert_array_equal(captured["mask"], expected)


def test_different_coordinate_gif_uses_target_truth_or_rejects_explicitly(
    tiny_experiment,
    tmp_path,
    monkeypatch,
):
    experiment = tiny_experiment
    dataset = _gif_dataset(experiment)
    result = _gif_result(experiment)
    shift = np.asarray([0.125, 0.0])
    shifted_sites = np.asarray(experiment.lattice_model.site_coordinates) + shift
    shifted_model = replace(
        experiment.lattice_model,
        site_coordinates=shifted_sites,
    )
    result = replace(
        result,
        site_coordinates=shifted_sites,
        displaced_site_coordinates=shifted_sites,
    )
    captured_images = []
    from matplotlib.axes import Axes

    original_imshow = Axes.imshow

    def capture_imshow(axis, values, *args, **kwargs):
        captured_images.append(np.asarray(values).copy())
        return original_imshow(axis, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", capture_imshow)
    _skip_gif_encoding(monkeypatch)
    target_model, target_sites = workflow._target_only_lattice_model_1d(
        experiment,
        experiment.lattice_model,
        result=_gif_result(experiment),
    )
    expected_truth = np.asarray(
        render_lattice_site_potential_1d(
            target_model,
            np.asarray(dataset.truth_vacancy_fractions)[target_sites],
            dataset.truth_displacement_controls
            + dataset.truth_rigid_displacement,
        )
    )
    target_support = np.asarray(experiment.target_lattice_influence_mask)
    slices, _ = workflow._update_region_view(
        experiment,
        mask=target_support,
    )
    expected_display = np.where(
        target_support[slices],
        expected_truth[slices],
        np.nan,
    ).T

    try:
        save_lattice_reconstruction_gif_1d(
            tmp_path / "different_coordinates.gif",
            experiment,
            dataset,
            result,
            lattice_model=shifted_model,
            lattice_influence_mask=np.ones_like(target_support),
        )
    except ValueError as error:
        message = str(error).lower()
        assert any(
            term in message
            for term in ("truth", "target", "alignment", "support contract")
        )
    else:
        assert captured_images
        np.testing.assert_allclose(
            captured_images[0],
            expected_display,
            equal_nan=True,
        )


def test_alignment_candidates_rebuild_complete_slab_without_truth(tiny_experiment):
    prior = build_silicon_alignment_prior_1d(tiny_experiment)
    truth_changed = replace(
        tiny_experiment,
        truth_potentials={"fabricated": np.full_like(tiny_experiment.pristine_potential, 7.0)},
        truth_vacancy_fractions={"fabricated": np.asarray([1.0])},
        truth_displacement_controls={"fabricated": np.asarray([9.0])},
    )
    repeated_prior = build_silicon_alignment_prior_1d(truth_changed)
    assert repeated_prior.prior_id == prior.prior_id
    assert prior.metadata["structurally_trusted"] is False

    options = AlignmentInitializationOptions1D(
        candidates_per_termination=2,
        training_screen_scan_count=2,
        lattice_scale_bounds=(1.0, 1.0),
        in_plane_rotation_bounds_rad=(0.0, 0.0),
        seed=3,
    )
    candidates = generate_silicon_alignment_candidates_1d(
        prior.termination_ids,
        options=options,
    )
    first = rebuild_silicon_alignment_candidate_1d(prior, candidates[0])
    alternate_termination = rebuild_silicon_alignment_candidate_1d(
        prior, candidates[1]
    )
    assert first.candidate_model_id != alternate_termination.candidate_model_id
    assert first.metadata["uses_defect_truth"] is False
    assert first.lattice_model.metadata["fixed_exterior_rebuilt"] is True
    exterior = ~np.asarray(prior.reconstruction_mask)
    assert np.any(
        np.asarray(first.lattice_model.reference_potential)[exterior]
        != np.asarray(alternate_termination.lattice_model.reference_potential)[
            exterior
        ]
    )

    model = first.lattice_model
    rendered = render_lattice_site_potential_1d(
        model,
        np.zeros(len(model.site_coordinates)),
        np.zeros(
            (
                len(model.control_coordinates_s),
                len(model.control_coordinates_u),
                2,
            )
        ),
    )
    np.testing.assert_allclose(rendered, model.reference_potential, atol=1e-11)


def test_truth_free_alignment_initialization_uses_training_then_validation(
    tiny_experiment,
    tmp_path,
):
    dataset = simulate_experiment_1d(tiny_experiment, "vacancy", batch_size=2)
    problem = build_silicon_alignment_problem_1d(tiny_experiment)
    assert problem.metadata["contains_observations"] is False
    options = AlignmentInitializationOptions1D(
        candidates_per_termination=2,
        training_screen_scan_count=1,
        validation_shortlist_size=2,
        refinement_rounds=0,
        lattice_scale_bounds=(1.0, 1.0),
        in_plane_rotation_bounds_rad=(0.0, 0.0),
        seed=5,
    )
    initialized = initialize_silicon_alignment_1d(
        problem,
        dataset.scan,
        options=options,
    )
    assert initialized.structurally_trusted is False
    assert len(initialized.candidate_scores) == 2
    assert initialized.metadata["candidate_catalog_size"] == 4
    assert initialized.metadata["audit_used_for_selection"] is False
    assert initialized.selection_summary.metadata["audit_used_for_selection"] is False
    assert set(np.asarray(initialized.training_screen_indices)).issubset(
        set(np.asarray(tiny_experiment.training_indices))
    )
    assert set(np.asarray(initialized.validation_indices)) == set(
        np.asarray(tiny_experiment.validation_indices)
    )
    assert initialized.selected_model.lattice_model.metadata[
        "reference_rebuild_scope"
    ] == "complete_finite_slab_all_atoms"
    prepared = prepare_aligned_lattice_site_reconstruction_1d(
        initialized,
        problem,
        dataset.scan,
        minibatch_size=1,
        evaluation_batch_size=2,
        rematerialize=False,
    )
    assert prepared.similarity_residual_gauge is True
    assert prepared.separate_rigid_registration is False
    assert prepared.metadata["displacement_gauge"] == (
        "translation_rotation_isotropic_dilation"
    )
    assert prepared.model.metadata["alignment_selection_id"] == (
        initialized.selection_summary.alignment_selection_id
    )

    archive_path = tmp_path / "alignment-initialization.npz"
    save_silicon_alignment_initialization_1d(archive_path, initialized)
    with np.load(archive_path, allow_pickle=False) as archive:
        assert archive["schema_version"].item() == 1
        assert archive["archive_digest"].shape == ()
        assert archive["catalog_candidate_id"].dtype.kind == "U"
    loaded = load_silicon_alignment_initialization_1d(
        archive_path,
        problem,
        dataset.scan,
    )
    assert loaded.selection_summary.alignment_selection_id == (
        initialized.selection_summary.alignment_selection_id
    )
    assert loaded.selected_model.candidate_model_id == (
        initialized.selected_model.candidate_model_id
    )
    assert tuple(
        candidate.candidate_id for candidate in loaded.candidate_catalog
    ) == tuple(
        candidate.candidate_id for candidate in initialized.candidate_catalog
    )

    with np.load(archive_path, allow_pickle=False) as archive:
        corrupted_payload = {
            name: np.asarray(archive[name]) for name in archive.files
        }
    corrupted_payload["score_validation_loss"] = np.asarray(
        corrupted_payload["score_validation_loss"]
    ).copy()
    corrupted_payload["score_validation_loss"][0] += 1e-3
    corrupted_path = tmp_path / "alignment-corrupted.npz"
    np.savez_compressed(corrupted_path, **corrupted_payload)
    with pytest.raises(ValueError, match="archive digest"):
        load_silicon_alignment_initialization_1d(
            corrupted_path,
            problem,
            dataset.scan,
        )

    changed_intensities = np.asarray(dataset.scan.intensities).copy()
    changed_intensities[int(initialized.training_screen_indices[0]), 0] += 1e-3
    changed_scan = replace(dataset.scan, intensities=changed_intensities)
    with pytest.raises(ValueError, match="raw scan"):
        load_silicon_alignment_initialization_1d(
            archive_path,
            problem,
            changed_scan,
        )


def test_atomic_parameterization_comparison_is_same_grid_and_fail_closed(
    tiny_experiment,
):
    experiment = tiny_experiment
    candidate = experiment.independent_kirkland_template
    comparison = experiment.lobato_kirkland_template_comparison
    summary = experiment.summary["atomic template parameterization diagnostic"]

    assert candidate.values.shape == tuple(
        np.asarray(experiment.lattice_model.site_patches).shape[1:]
    )
    assert candidate.fractional_offset_A == (0.0, 0.0)
    assert not candidate.values.flags.writeable
    assert candidate.trust_claim is False
    assert comparison.trust_claim is False
    assert comparison.candidate_template_sha256 == candidate.template_sha256
    assert np.isfinite(comparison.raw_relative_l2)
    assert np.isfinite(comparison.scale_adjusted_shape_relative_l2)
    assert comparison.raw_relative_l2 > 0.0
    assert summary["candidate_template_sha256"] == candidate.template_sha256
    assert summary["comparison_sha256"] == comparison.comparison_sha256
    assert summary["diagnostic"] == (
        "direct_Kirkland_vs_production_Lobato_same_template_grid"
    )
    assert summary["trust_claim"] is False
    assert summary["has_acceptance_threshold"] is False
    assert summary["used_for_cutoff_certification"] is False
    assert "Kirkland-versus-Lobato" in " ".join(summary["limitations"])
    assert not any(
        "kirkland" in name and "potential" in name
        for name in vars(experiment)
    )


def test_truth_free_alignment_recovers_an_exact_catalogued_pose(tiny_experiment):
    problem = build_silicon_alignment_problem_1d(tiny_experiment)
    options = AlignmentInitializationOptions1D(
        candidates_per_termination=2,
        training_screen_scan_count=2,
        validation_shortlist_size=4,
        refinement_rounds=0,
        lattice_scale_bounds=(0.995, 1.005),
        in_plane_rotation_bounds_rad=(-0.004, 0.004),
        seed=11,
    )
    candidates = generate_silicon_alignment_candidates_1d(
        problem.prior.termination_ids,
        options=options,
    )
    generating_candidate = candidates[-1]
    generating_model = rebuild_silicon_alignment_candidate_1d(
        problem.prior,
        generating_candidate,
    )
    intensities = simulate_glancing_scan_1d(
        generating_model.lattice_model.reference_potential,
        problem.input_probes,
        problem.window_starts,
        problem.window_length,
        problem.propagation_kernel,
        problem.slice_thickness_A,
        problem.energy_eV,
        rematerialize=False,
    )
    observed_scan = GlancingScan1D(
        intensities=np.asarray(intensities),
        window_starts=problem.window_starts,
        scan_coordinates=problem.scan_coordinates,
        detector_angles=problem.detector_angles,
        metadata={"truth_not_consumed_by_initializer": generating_candidate.candidate_id},
    )

    initialized = initialize_silicon_alignment_1d(
        problem,
        observed_scan,
        options=options,
    )

    assert initialized.selection_summary.selected_candidate_id == (
        generating_candidate.candidate_id
    )
    assert initialized.selected_model.candidate_model_id == (
        generating_model.candidate_model_id
    )
    selected_score = next(
        score
        for score in initialized.candidate_scores
        if score.candidate.candidate_id == generating_candidate.candidate_id
    )
    assert selected_score.training_screen_loss < 1e-20
    assert selected_score.validation_loss < 1e-20
    assert initialized.metadata["audit_used_for_selection"] is False


def test_compact_workflow_builds_simulates_and_reconstructs(tiny_experiment, tmp_path):
    experiment = tiny_experiment
    assert experiment.config.exterior_material_policy == "parameterize_uncertain"
    assert experiment.support_contract.strict_requirements_satisfied
    assert experiment.lattice_model.support_contract is experiment.support_contract
    assert experiment.summary["material scope complete"] is True
    assert experiment.summary["target Si sites"] > 0
    assert experiment.summary["nuisance Si sites"] > 0
    np.testing.assert_array_equal(
        experiment.modeled_target_site_mask
        | experiment.modeled_nuisance_site_mask,
        np.ones(len(experiment.variable_sites), dtype=bool),
    )
    np.testing.assert_array_equal(
        experiment.target_sites,
        np.asarray(experiment.variable_sites)[
            np.asarray(experiment.modeled_target_site_mask)
        ],
    )
    assert np.any(
        np.asarray(experiment.nuisance_lattice_influence_mask)
        & np.asarray(experiment.target_lattice_influence_mask)
    )
    assert set(experiment.truth_potentials) == {
        "vacancy",
        "vacancy_plus_strain",
        "strained_surface_defects",
    }
    assert set(experiment.cutoff_check_potentials) == set(
        experiment.truth_potentials
    )
    assert experiment.summary["simple vacancy sites"] == 2
    assert experiment.summary["complex surface-defect sites"] > 2
    assert (
        experiment.summary["lattice parameters"] < experiment.summary["pixel unknowns"]
    )
    assert float(np.min(np.asarray(experiment.truth_potentials["vacancy"]))) >= -1e-12
    hard_controls = np.asarray(
        experiment.truth_displacement_controls["strained_surface_defects"]
    )
    hard_controls = hard_controls + np.asarray(
        experiment.truth_rigid_displacements["strained_surface_defects"]
    )
    assert np.max(np.abs(hard_controls[..., 0])) == pytest.approx(0.35)
    assert np.max(np.abs(hard_controls[..., 1])) == pytest.approx(0.20)
    support = np.asarray(experiment.reconstruction_mask)
    lattice_influence = np.asarray(experiment.lattice_influence_mask)
    site_selection = np.asarray(experiment.site_selection_mask)
    interaction = experiment.interaction_region
    forward = np.asarray(interaction.forward_mask)
    coverage = np.asarray(interaction.scan_coverage)
    peak_intensity = np.asarray(interaction.peak_relative_intensity)
    assert experiment.config.update_region == "auto"
    np.testing.assert_array_equal(site_selection, interaction.reconstruction_mask)
    np.testing.assert_array_equal(support, site_selection)
    assert np.all(forward[site_selection])
    assert np.all(
        coverage[site_selection] >= interaction.minimum_scan_coverage
    )
    assert np.all(
        peak_intensity[site_selection] >= interaction.intensity_threshold
    )
    for potential in experiment.truth_potentials.values():
        delta = np.asarray(potential - experiment.pristine_potential)
        np.testing.assert_allclose(delta[~lattice_influence], 0.0, atol=1e-12)
    assert np.all(
        np.asarray(interaction.scan_coverage)[site_selection]
        >= interaction.minimum_scan_coverage
    )
    assert interaction.metadata["mutable_scan_indices"] == np.asarray(
        experiment.training_indices
    ).tolist()
    assert experiment.template_certification.cutoff_A < 8.0
    assert (
        experiment.template_certification.relative_tail_l2
        <= experiment.template_certification.tolerance
    )
    assert not np.intersect1d(
        experiment.validation_indices, experiment.audit_indices
    ).size
    partitions = (
        experiment.training_indices,
        experiment.validation_indices,
        experiment.audit_indices,
        experiment.guard_indices,
    )
    np.testing.assert_array_equal(
        np.sort(np.concatenate([np.asarray(values) for values in partitions])),
        np.arange(experiment.config.n_scans),
    )
    assert np.all(np.asarray(experiment.audit_site_scan_coverage) >= 0)
    assert experiment.audit_site_scan_coverage_metadata[
        "uses_measured_diffraction_values"
    ] is False
    assert "not evidence" in experiment.audit_site_scan_coverage_metadata[
        "interpretation"
    ]

    dataset = simulate_experiment_1d(
        experiment, "strained_surface_defects", batch_size=2
    )
    assert dataset.intensities.shape == (
        experiment.config.n_scans,
        len(experiment.transverse_coordinates),
    )
    assert dataset.template_cutoff_amplitude_nrmse < 1e-4
    assert dataset.template_cutoff_max_scan_amplitude_nrmse < 1e-4
    assert dataset.template_stress_worst_scan_amplitude_nrmse < 1e-4
    assert dataset.template_certified_worst_amplitude_nrmse < 1e-4
    assert np.isfinite(dataset.kirkland_alternative_amplitude_nrmse)
    assert np.isfinite(dataset.kirkland_alternative_max_scan_amplitude_nrmse)
    assert dataset.kirkland_alternative_amplitude_nrmse > 0.0
    assert (
        dataset.kirkland_alternative_max_scan_amplitude_nrmse
        >= dataset.kirkland_alternative_amplitude_nrmse
    )
    assert (
        dataset.kirkland_alternative_amplitude_nrmse
        > experiment.config.atomic_template_amplitude_tolerance
    )
    assert dataset.scan.metadata["kirkland_alternative_amplitude_nrmse"] == (
        dataset.kirkland_alternative_amplitude_nrmse
    )
    assert dataset.scan.metadata[
        "kirkland_alternative_max_scan_amplitude_nrmse"
    ] == dataset.kirkland_alternative_max_scan_amplitude_nrmse
    assert dataset.scan.metadata["kirkland_alternative_trust_claim"] is False
    assert dataset.scan.metadata[
        "kirkland_alternative_has_acceptance_threshold"
    ] is False
    assert dataset.scan.metadata[
        "kirkland_alternative_used_for_cutoff_certification"
    ] is False
    assert "not an independent end-to-end" in dataset.scan.metadata[
        "kirkland_alternative_shared_components_limitation"
    ]
    assert dataset.scan.metadata[
        "atomic_template_parameterization_diagnostic"
    ]["comparison_sha256"] == (
        experiment.lobato_kirkland_template_comparison.comparison_sha256
    )
    assert dataset.scan.metadata["template_amplitude_check_scope"] == (
        "whole_finite_slab_case_specific_and_maximum_displacement_all_scans"
    )
    assert dataset.scan.metadata["audit_construction"] == (
        "geometry_only_stratified_contiguous_blocks"
    )
    detector_valid_mask = np.ones(dataset.intensities.shape, dtype=bool)
    detector_valid_mask[0, 0] = False
    measured = np.asarray(dataset.intensities).copy()
    measured[0, 0] = np.nan
    dataset = replace(
        dataset,
        scan=replace(
            dataset.scan,
            intensities=measured,
            detector_valid_mask=detector_valid_mask,
        ),
    )

    results = reconstruct_experiment_1d(
        experiment,
        dataset,
        methods=("lattice_sites",),
        options=ReconstructionOptions1D(
            lattice_updates=2,
            minibatch_size=2,
            validation_interval_lattice=1,
            evaluation_batch_size=2,
            rematerialize=False,
            initial_site_offset_A=(0.10, -0.05),
            initial_control_noise_A=0.0,
            lattice_checkpoint_interval=1,
        ),
    )
    assert tuple(results) == ("lattice sites",)
    lattice_result = results["lattice sites"]
    assert lattice_result.material_scope_complete
    assert lattice_result.support_contract_id == experiment.support_contract.contract_id
    np.testing.assert_array_equal(
        lattice_result.target_site_mask,
        experiment.modeled_target_site_mask,
    )
    np.testing.assert_array_equal(
        lattice_result.nuisance_site_mask,
        experiment.modeled_nuisance_site_mask,
    )
    initial_controls = np.asarray(
        results["lattice sites"].initial_displacement_controls
    )
    np.testing.assert_allclose(initial_controls, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        results["lattice sites"].initial_rigid_displacement,
        (0.10, -0.05),
        atol=1e-12,
    )
    assert (
        results["lattice sites"].metadata["registration_scope"]
        == "variable_sites_relative_to_fixed_reference"
    )
    np.testing.assert_array_equal(
        results["lattice sites"].detector_valid_mask,
        detector_valid_mask,
    )
    assert results["lattice sites"].metadata["measurement_contract"] == (
        "masked_nonnegative_intensity"
    )
    np.testing.assert_array_equal(
        results["lattice sites"].metadata["training_indices"],
        experiment.training_indices,
    )
    np.testing.assert_array_equal(
        results["lattice sites"].checkpoint_updates,
        np.arange(results["lattice sites"].completed_updates + 1),
    )
    metrics = reconstruction_metrics_1d(experiment, dataset, results)
    assert np.isfinite(metrics["lattice sites"]["held-out audit loss"])
    assert (
        metrics["lattice sites"]["specimen parameters"]
        == experiment.summary["lattice parameters"]
    )
    assert metrics["lattice sites"]["potential metric scope"] == (
        "target_sites_only_nuisance_reset_to_pristine"
    )
    assert metrics["lattice sites"]["nuisance vacancy parameters"] == (
        experiment.summary["nuisance Si sites"]
    )
    sensitivity = screen_lattice_reconstruction_sensitivity_1d(
        experiment,
        results["lattice sites"],
        PoissonCountingModel1D(electrons_per_pattern=1e4),
        options=SensitivityScreenOptions1D(
            hutchinson_probes=2,
            probe_batch_size=1,
            evaluation_batch_size=1,
            rematerialize=False,
        ),
    )
    np.testing.assert_array_equal(
        sensitivity.scan_indices, experiment.audit_indices
    )
    assert sensitivity.site_sensitive.shape == (
        experiment.summary["variable Si sites"],
    )

    paths = save_experiment_results_1d(tmp_path, dataset, results)
    assert all(path.exists() for path in paths.values())
    gif_path = save_lattice_reconstruction_gif_1d(
        tmp_path / "reconstruction.gif",
        experiment,
        dataset,
        results["lattice sites"],
        fps=2,
        dpi=40,
    )
    assert gif_path.read_bytes().startswith(b"GIF")
    with PILImage.open(gif_path) as gif:
        assert gif.n_frames == len(results["lattice sites"].checkpoint_updates)

    selected_initial = replace(
        results["lattice sites"],
        potential=results["lattice sites"].initial_potential,
        best_update=0,
    )
    selected_gif_path = save_lattice_reconstruction_gif_1d(
        tmp_path / "reconstruction_selected_initial.gif",
        experiment,
        dataset,
        selected_initial,
        fps=2,
        dpi=40,
    )
    with PILImage.open(selected_gif_path) as gif:
        assert gif.n_frames == len(selected_initial.checkpoint_updates) + 1

    with pytest.raises(TypeError, match="fps must be an integer"):
        save_lattice_reconstruction_gif_1d(
            tmp_path / "invalid_fps.gif",
            experiment,
            dataset,
            results["lattice sites"],
            fps=2.5,
        )
    mismatched_selected = replace(
        results["lattice sites"],
        potential=np.zeros_like(results["lattice sites"].potential),
    )
    with pytest.raises(ValueError, match="best checkpoint does not reproduce"):
        save_lattice_reconstruction_gif_1d(
            tmp_path / "mismatched_selected.gif",
            experiment,
            dataset,
            mismatched_selected,
            fps=2,
            dpi=40,
        )

    overview = plot_experiment_overview_1d(experiment, dataset)
    comparison = plot_reconstruction_comparison_1d(experiment, dataset, results)
    lattice_figures = plot_lattice_reconstruction_1d(
        experiment, results["lattice sites"]
    )
    sensitivity_figure = plot_lattice_sensitivity_screen_1d(sensitivity)
    assert overview is not None
    assert len(comparison) == 2
    assert len(lattice_figures) == 2
    assert sensitivity_figure is not None
    comparison_limits = comparison[0].axes[0].get_xlim()
    comparison_width = comparison_limits[1] - comparison_limits[0]
    influenced_s = np.asarray(experiment.axial_coordinates)[
        np.any(lattice_influence, axis=1)
    ]
    assert comparison_limits[0] <= influenced_s[0]
    assert comparison_limits[1] >= influenced_s[-1]
    full_width = float(
        experiment.axial_coordinates[-1] - experiment.axial_coordinates[0]
    )
    assert comparison_width <= full_width


def test_template_reference_rebuilds_fixed_atoms_outside_variable_influence(
    tiny_experiment,
):
    deliberately_short = replace(
        tiny_experiment.config,
        atomic_template_cutoff_A=3.0,
        atomic_template_tolerance=2e-3,
        atomic_template_amplitude_tolerance=1.0,
    )
    experiment = build_silicon_glancing_experiment_1d(deliberately_short)
    compact, reference = experiment.template_stress_potential_pairs[
        "maximum_positive_diagonal_displacement"
    ]
    difference = np.asarray(reference) - np.asarray(compact)
    outside_variable_influence = ~np.asarray(experiment.lattice_influence_mask)

    assert np.linalg.norm(difference) > 0.0
    assert np.linalg.norm(difference[outside_variable_influence]) > 0.0

    strict_experiment = replace(
        experiment,
        config=replace(
            experiment.config,
            atomic_template_amplitude_tolerance=1e-12,
        ),
    )
    with pytest.raises(RuntimeError, match="whole-slab forward amplitude"):
        simulate_experiment_1d(
            strict_experiment,
            "strained_surface_defects",
            batch_size=2,
        )


def test_high_level_multistart_reuses_problem_and_checkpoints_medoid(
    tiny_experiment,
):
    dataset = simulate_experiment_1d(
        tiny_experiment,
        "vacancy",
        batch_size=2,
    )
    detector_valid_mask = np.ones(dataset.intensities.shape, dtype=bool)
    detector_valid_mask[0, 0] = False
    objective = PtychographyObjective1D(
        kind="poisson_deviance",
        electrons_per_pattern=1_000.0,
    )
    signal = ptychography_expected_signal_electrons_1d(
        dataset.intensities,
        tiny_experiment.input_probes,
        objective,
    )
    calibrated_measurement = PtychographyMeasurement1D(
        calibrated_signal_electrons=signal.at[0, 0].set(np.nan),
        observed_total_electrons=(signal + 0.1).at[0, 0].set(-1e200),
        valid_mask=detector_valid_mask,
        calibrated_dark_electrons_per_pixel=0.1,
        calibrated_read_noise_std_electrons=0.0,
        calibration_id="workflow_calibrated_counts",
    )
    measured = np.asarray(dataset.intensities).copy()
    measured[0, 0] = np.nan
    dataset = replace(
        dataset,
        scan=replace(
            dataset.scan,
            intensities=measured,
            detector_valid_mask=detector_valid_mask,
        ),
    )
    result = reconstruct_lattice_multistart_experiment_1d(
        tiny_experiment,
        dataset,
        options=ReconstructionOptions1D(
            lattice_updates=2,
            minibatch_size=2,
            validation_interval_lattice=1,
            evaluation_batch_size=2,
            rematerialize=False,
            progress=False,
            lattice_checkpoint_interval=1,
        ),
        multistart_options=MultistartOptions1D(
            n_starts=3,
            base_seed=5,
            initial_translation_half_width_A=(0.05, 0.05),
            minimum_accepted_starts=1,
        ),
        measurement=calibrated_measurement,
        objective=objective,
    )

    assert len(result.screening_results) == 3
    assert result.representative_trajectory_reused is True
    assert (
        result.representative_result
        is result.screening_results[result.ensemble.representative_index]
    )
    np.testing.assert_array_equal(
        result.representative_result.checkpoint_updates,
        [0, 1, 2],
    )
    problem_ids = {
        run.metadata["reconstruction_problem_id"]
        for run in result.screening_results
    }
    assert problem_ids == {
        result.representative_result.metadata["reconstruction_problem_id"]
    }
    assert result.representative_result.metadata["reconstructor_id"] == (
        "wide_angle_propagation.lattice_site_prepared:v1"
    )
    np.testing.assert_array_equal(
        result.representative_result.detector_valid_mask,
        detector_valid_mask,
    )
    assert result.representative_result.predicted_signal_electrons is not None
    assert result.representative_result.metadata["objective_kind"] == (
        "poisson_deviance"
    )
    assert result.registration_scope == "active_sites_relative_to_fixed_reference"


def test_compact_workflow_rejects_unknown_case_and_method(tiny_experiment):
    with pytest.raises(ValueError, match="case"):
        simulate_experiment_1d(tiny_experiment, "unknown", batch_size=2)

    dataset = simulate_experiment_1d(tiny_experiment, "vacancy", batch_size=2)
    with pytest.raises(ValueError, match="unknown reconstruction methods"):
        reconstruct_experiment_1d(tiny_experiment, dataset, methods=("not-a-method",))


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("window_starts", "window_starts"),
        ("scan_coordinates", "scan_coordinates"),
        ("detector_angles", "detector_angles"),
        ("intensities", "intensities shape"),
        ("split_metadata", "metadata\\['audit_indices'\\]"),
    ],
)
def test_reconstruct_experiment_rejects_incompatible_scan_geometry(
    tiny_experiment, field, message
):
    scan = simulate_experiment_1d(
        tiny_experiment, "vacancy", batch_size=2
    ).scan
    if field == "window_starts":
        values = np.asarray(scan.window_starts).copy()
        values[0] += 1
        bad_scan = replace(scan, window_starts=values)
    elif field == "scan_coordinates":
        values = np.asarray(scan.scan_coordinates).copy()
        values[0] += 0.1
        bad_scan = replace(scan, scan_coordinates=values)
    elif field == "detector_angles":
        values = np.asarray(scan.detector_angles).copy()
        values[0] += 0.1
        bad_scan = replace(scan, detector_angles=values)
    elif field == "intensities":
        bad_scan = replace(scan, intensities=np.asarray(scan.intensities)[:-1])
    else:
        metadata = dict(scan.metadata)
        metadata["audit_indices"] = metadata["training_indices"]
        bad_scan = replace(scan, metadata=metadata)
    with pytest.raises(ValueError, match=message):
        reconstruct_experiment_1d(tiny_experiment, bad_scan, methods=())


def test_reconstruct_experiment_accepts_semantically_identical_float32_geometry(
    tiny_experiment,
):
    scan = simulate_experiment_1d(
        tiny_experiment, "vacancy", batch_size=2
    ).scan
    compatible_scan = replace(
        scan,
        scan_coordinates=np.asarray(scan.scan_coordinates, dtype=np.float32),
        detector_angles=np.asarray(scan.detector_angles, dtype=np.float32),
    )

    assert reconstruct_experiment_1d(
        tiny_experiment, compatible_scan, methods=()
    ) == {}
