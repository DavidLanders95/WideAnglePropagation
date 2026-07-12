"""Workflow integration gates for geometry-derived atomistic-edit support."""

from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.ptychography_1d import (  # noqa: E402
    LatticeSiteModel1D,
)
from wide_angle_propagation.ptychography_atomistic_edit_1d import (  # noqa: E402
    AtomisticEditOptions1D,
    empty_atomistic_edit_state_1d,
    make_atomistic_edit_discovery_support_1d,
    render_atomistic_edit_potential_1d,
)
from wide_angle_propagation.ptychography_support_contract_1d import (  # noqa: E402
    classify_lattice_site_support_1d,
)
from wide_angle_propagation.ptychography_workflow_1d import (  # noqa: E402
    AtomicTemplateCertification1D,
    SiliconGlancingConfig1D,
    SiliconGlancingExperiment1D,
    build_atomistic_edit_discovery_support_1d,
    build_atomistic_edit_model_1d,
    gaussian_interaction_region_1d,
)


SHAPE = (21, 21)
S_A = np.arange(SHAPE[0], dtype=np.float64)
U_A = np.arange(-10.0, 11.0, dtype=np.float64)
SCANS_A = np.asarray([4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
TRAINING = np.asarray([1, 2, 3, 4], dtype=np.int32)
SURFACE_ENVELOPE_A = (-3.0, 3.0)
HOST_CENTRES = np.asarray([[10, 8], [14, 8]], dtype=np.int32)
HOST_STARTS = HOST_CENTRES - 2


def _host_patch() -> np.ndarray:
    patch = np.zeros((5, 5), dtype=np.float64)
    patch[1:4, 1:4] = np.asarray(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
    )
    return patch


def _compact_host():
    patch = _host_patch()
    patches = np.stack([patch, patch])
    reference = np.full(SHAPE, 0.05, dtype=np.float64)
    for start, site_patch in zip(HOST_STARTS, patches):
        start_s, start_u = start
        reference[start_s : start_s + 5, start_u : start_u + 5] += site_patch
    coordinates = np.column_stack(
        [S_A[HOST_CENTRES[:, 0]], U_A[HOST_CENTRES[:, 1]]]
    )
    target_pixels = np.zeros(SHAPE, dtype=bool)
    target_pixels[tuple(HOST_CENTRES[0])] = True
    support = classify_lattice_site_support_1d(
        coordinates,
        HOST_CENTRES,
        HOST_STARTS,
        np.full((2, 2), 5, dtype=np.int32),
        target_pixels,
        np.ones(SHAPE, dtype=bool),
        excluded_probe_power=0.05,
        atomic_template_cutoff_A=2.0,
        maximum_displacement_A=0.5,
        displacement_control_shape=(2, 2, 2),
        maximum_nuisance_sites=4,
        maximum_specimen_parameters=64,
        strict=True,
    )
    model = LatticeSiteModel1D(
        reference_potential=jnp.asarray(reference),
        site_coordinates=jnp.asarray(coordinates),
        site_patches=jnp.asarray(patches),
        patch_starts=jnp.asarray(HOST_STARTS),
        control_coordinates_s=jnp.asarray([S_A[0], S_A[-1]]),
        control_coordinates_u=jnp.asarray([U_A[0], U_A[-1]]),
        axial_sampling=1.0,
        transverse_sampling=1.0,
        maximum_displacement=0.5,
        metadata={"species": "Si", "parameterization": "compact-Lobato"},
        support_contract=support,
    )
    return model, support, reference, coordinates


def _region(
    config,
    *,
    slab_top_A,
    scans=SCANS_A,
    mutable_scan_indices=TRAINING,
    position_uncertainty_A=None,
    angle_uncertainty_deg=None,
):
    return gaussian_interaction_region_1d(
        S_A,
        U_A,
        scans,
        beam_waist_A=config.beam_waist_A,
        beam_tilt_rad=-np.deg2rad(config.glancing_angle_deg),
        slab_bottom_A=-3.0,
        slab_top_A=slab_top_A,
        excluded_probe_power=config.interaction_excluded_probe_power,
        intensity_threshold=config.interaction_intensity_threshold,
        minimum_scan_coverage=config.minimum_scan_coverage,
        beam_position_uncertainty_A=(
            config.beam_position_uncertainty_A
            if position_uncertainty_A is None
            else position_uncertainty_A
        ),
        beam_angle_uncertainty_rad=np.deg2rad(
            config.beam_angle_uncertainty_deg
            if angle_uncertainty_deg is None
            else angle_uncertainty_deg
        ),
        mutable_scan_indices=mutable_scan_indices,
    )


@pytest.fixture(scope="module")
def compact_silicon_experiment():
    host_model, support, reference, coordinates = _compact_host()
    config = SiliconGlancingConfig1D(
        energy_eV=30_000.0,
        glancing_angle_deg=20.0,
        beam_waist_A=1.2,
        slab_depth_A=3.0,
        vacuum_above_A=10.0,
        vacuum_below_A=7.0,
        window_length_A=10.0,
        sampling_u_A=1.0,
        sampling_s_A=1.0,
        scan_start_A=float(SCANS_A[0]),
        scan_stop_A=float(SCANS_A[-1]),
        n_scans=len(SCANS_A),
        validation_stride=3,
        interaction_excluded_probe_power=0.05,
        minimum_scan_coverage=1,
        beam_position_uncertainty_A=0.5,
        beam_angle_uncertainty_deg=2.0,
        atomic_template_cutoff_A=2.0,
        maximum_displacement_A=0.5,
        displacement_control_spacing_A=20.0,
        displacement_control_spacing_u_A=20.0,
    )
    interaction_region = _region(config, slab_top_A=0.0)
    target_site_mask = np.asarray([True, False])
    nuisance_site_mask = ~target_site_mask
    return SiliconGlancingExperiment1D(
        config=config,
        pristine_potential=jnp.asarray(reference),
        lattice_model=host_model,
        template_certification=AtomicTemplateCertification1D(
            cutoff_A=2.0,
            reference_cutoff_A=4.0,
            relative_tail_l2=0.0,
            tolerance=1e-8,
            candidate_errors={"2": 0.0},
        ),
        independent_kirkland_template=SimpleNamespace(
            options=SimpleNamespace(projection_width_A=5.0)
        ),
        lobato_kirkland_template_comparison=None,
        support_contract=support,
        interaction_region=interaction_region,
        truth_potentials={},
        truth_vacancy_fractions={},
        truth_displacement_controls={},
        truth_rigid_displacements={},
        defect_site_indices={},
        all_site_coordinates=jnp.asarray(coordinates),
        variable_sites=jnp.asarray(coordinates),
        target_sites=jnp.asarray(coordinates[target_site_mask]),
        modeled_target_site_mask=jnp.asarray(target_site_mask),
        modeled_nuisance_site_mask=jnp.asarray(nuisance_site_mask),
        site_selection_mask=jnp.asarray([True, True]),
        reconstruction_mask=jnp.asarray(support.target_pixel_mask),
        lattice_influence_mask=jnp.asarray(
            support.target_influence_mask | support.nuisance_influence_mask
        ),
        target_lattice_influence_mask=jnp.asarray(
            support.target_influence_mask
        ),
        nuisance_lattice_influence_mask=jnp.asarray(
            support.nuisance_influence_mask
        ),
        beam_path_scan_coverage=interaction_region.scan_coverage,
        input_probes=jnp.zeros((len(SCANS_A), SHAPE[1]), dtype=jnp.complex128),
        propagation_kernel=jnp.ones(SHAPE[1], dtype=jnp.complex128),
        window_starts=jnp.zeros(len(SCANS_A), dtype=jnp.int32),
        window_length=1,
        scan_coordinates=jnp.asarray(SCANS_A),
        axial_coordinates=jnp.asarray(S_A),
        transverse_coordinates=jnp.asarray(U_A),
        detector_angles=jnp.arange(SHAPE[1], dtype=jnp.float64),
        training_indices=jnp.asarray(TRAINING),
        validation_indices=jnp.asarray([0], dtype=jnp.int32),
        audit_indices=jnp.asarray([5], dtype=jnp.int32),
        guard_indices=jnp.empty((0,), dtype=jnp.int32),
        audit_site_scan_coverage=jnp.zeros(2, dtype=jnp.int32),
        audit_site_scan_coverage_metadata={},
        cutoff_check_potentials={},
        template_stress_potential_pairs={},
        axial_sampling=1.0,
        transverse_sampling=1.0,
        summary={"fixture": "compact silicon, no synthetic object metadata"},
    )


@pytest.fixture(scope="module")
def discovery_support(compact_silicon_experiment):
    return build_atomistic_edit_discovery_support_1d(
        compact_silicon_experiment,
        surface_envelope_A=SURFACE_ENVELOPE_A,
    )


def _options(discovery):
    return AtomisticEditOptions1D(
        max_host_removals=2,
        max_extra_centres=3,
        max_scattering_equivalent_per_centre=2.0,
        minimum_separation_A=2.0,
        expected_rms_host_strain=0.1,
        edit_penalty_path=(1.0, 0.5),
        discovery_support=discovery,
        enable_material_energy_envelope=False,
    )


@pytest.fixture(scope="module")
def compact_edit_model(compact_silicon_experiment, discovery_support):
    return build_atomistic_edit_model_1d(
        compact_silicon_experiment, _options(discovery_support)
    )


def _complete_representative_footprint(experiment):
    patches = np.asarray(experiment.lattice_model.site_patches)
    starts = np.asarray(experiment.lattice_model.patch_starts)
    sites = np.asarray(experiment.lattice_model.site_coordinates)
    representative = int(np.argmax(np.sum(patches, axis=(1, 2))))
    site_index = np.asarray(
        [
            (
                sites[representative, 0]
                - float(experiment.axial_coordinates[0])
            )
            / experiment.axial_sampling,
            (
                sites[representative, 1]
                - float(experiment.transverse_coordinates[0])
            )
            / experiment.transverse_sampling,
        ]
    )
    centre = site_index - starts[representative]
    start_offset = np.floor(-centre + 0.5).astype(np.int64)
    rows, columns = np.indices(SHAPE)
    return (
        (rows + start_offset[0] >= 0)
        & (rows + start_offset[0] + patches.shape[1] <= SHAPE[0])
        & (columns + start_offset[1] >= 0)
        & (columns + start_offset[1] + patches.shape[2] <= SHAPE[1])
    )


def test_surface_adjacent_vacuum_target_is_geometry_only(
    compact_silicon_experiment, discovery_support
):
    vacuum_columns = U_A > 0.0
    assert np.any(np.asarray(discovery_support.target_mask)[:, vacuum_columns])
    assert compact_silicon_experiment.truth_potentials == {}
    assert compact_silicon_experiment.defect_site_indices == {}

    truth_tainted_copy = replace(
        compact_silicon_experiment,
        truth_potentials={"hidden_object": np.full(SHAPE, 123.0)},
        defect_site_indices={"hidden_object": np.asarray([0, 1])},
    )
    rebuilt = build_atomistic_edit_discovery_support_1d(
        truth_tainted_copy,
        surface_envelope_A=SURFACE_ENVELOPE_A,
    )
    assert rebuilt.contract_id == discovery_support.contract_id
    np.testing.assert_array_equal(rebuilt.target_mask, discovery_support.target_mask)
    np.testing.assert_array_equal(
        rebuilt.nuisance_mask, discovery_support.nuisance_mask
    )


def test_held_out_and_uncertainty_only_support_is_nuisance(
    compact_silicon_experiment, discovery_support
):
    config = compact_silicon_experiment.config
    raw = _region(config, slab_top_A=SURFACE_ENVELOPE_A[1])
    training_only = _region(
        config,
        slab_top_A=SURFACE_ENVELOPE_A[1],
        scans=SCANS_A[TRAINING],
        mutable_scan_indices=np.arange(len(TRAINING), dtype=np.int32),
        position_uncertainty_A=0.0,
        angle_uncertainty_deg=0.0,
    )
    complete = _complete_representative_footprint(compact_silicon_experiment)
    held_out_or_uncertainty_only = (
        np.asarray(raw.forward_mask)
        & ~np.asarray(training_only.forward_mask)
        & complete
    )
    assert np.any(held_out_or_uncertainty_only)
    assert np.all(
        np.asarray(discovery_support.nuisance_mask)[
            held_out_or_uncertainty_only
        ]
    )
    expected_nuisance = (
        np.asarray(raw.forward_mask)
        & ~np.asarray(raw.reconstruction_mask)
        & complete
    )
    np.testing.assert_array_equal(
        discovery_support.nuisance_mask, expected_nuisance
    )


def test_discovery_masks_are_disjoint_digest_bound_and_finite_grid_contracted(
    compact_silicon_experiment, discovery_support
):
    target = np.asarray(discovery_support.target_mask)
    nuisance = np.asarray(discovery_support.nuisance_mask)
    complete = _complete_representative_footprint(compact_silicon_experiment)
    assert np.any(target)
    assert np.any(nuisance)
    assert not np.any(target & nuisance)
    assert np.all(complete[target | nuisance])
    assert discovery_support.target_mask.flags.writeable is False
    assert discovery_support.nuisance_mask.flags.writeable is False
    assert discovery_support.metadata[
        "finite_grid_boundary_contracted_for_full_kernel"
    ] is True

    raw = _region(
        compact_silicon_experiment.config,
        slab_top_A=SURFACE_ENVELOPE_A[1],
    )
    assert np.any(
        (np.asarray(raw.forward_mask) | np.asarray(raw.reconstruction_mask))
        & ~complete
    )
    rebuilt = build_atomistic_edit_discovery_support_1d(
        compact_silicon_experiment,
        surface_envelope_A=SURFACE_ENVELOPE_A,
    )
    changed_envelope = build_atomistic_edit_discovery_support_1d(
        compact_silicon_experiment,
        surface_envelope_A=(-3.0, 2.0),
    )
    assert rebuilt.contract_id == discovery_support.contract_id
    assert changed_envelope.contract_id != discovery_support.contract_id


def test_workflow_model_zero_edit_is_exactly_pristine(
    compact_silicon_experiment, compact_edit_model
):
    state = empty_atomistic_edit_state_1d(compact_edit_model)
    np.testing.assert_array_equal(
        render_atomistic_edit_potential_1d(compact_edit_model, state),
        compact_silicon_experiment.pristine_potential,
    )


def test_workflow_unit_addition_integrates_like_representative_host(
    compact_silicon_experiment, compact_edit_model, discovery_support
):
    state = empty_atomistic_edit_state_1d(compact_edit_model)
    anchor = np.argwhere(np.asarray(discovery_support.target_mask))[0]
    anchors = np.asarray(state.extra_anchor_indices).copy()
    anchors[0] = anchor
    masses = np.zeros(compact_edit_model.options.max_extra_centres)
    masses[0] = 1.0
    active = np.zeros(compact_edit_model.options.max_extra_centres, dtype=bool)
    active[0] = True
    state = replace(
        state,
        extra_anchor_indices=jnp.asarray(anchors),
        extra_scattering_equivalents=jnp.asarray(masses),
        extra_active=jnp.asarray(active),
    )
    rendered = np.asarray(
        render_atomistic_edit_potential_1d(compact_edit_model, state)
    )
    pristine = np.asarray(compact_silicon_experiment.pristine_potential)
    integrated_addition = (
        np.sum(rendered - pristine)
        * compact_silicon_experiment.axial_sampling
        * compact_silicon_experiment.transverse_sampling
    )
    host_integrals = np.sum(
        np.asarray(compact_silicon_experiment.lattice_model.site_patches),
        axis=(1, 2),
    )
    host_integrals *= (
        compact_silicon_experiment.axial_sampling
        * compact_silicon_experiment.transverse_sampling
    )
    representative_host = float(np.max(host_integrals))
    assert compact_edit_model.addition_kernel.host_equivalent_integrated_scattering == (
        pytest.approx(representative_host, abs=1e-15)
    )
    assert integrated_addition == pytest.approx(representative_host, abs=3e-14)


def test_workflow_model_rejects_uncontracted_raw_edge_discovery(
    compact_silicon_experiment,
):
    target = np.zeros(SHAPE, dtype=bool)
    target[0, np.flatnonzero(U_A == 0.0)[0]] = True
    raw_edge = make_atomistic_edit_discovery_support_1d(
        S_A,
        U_A,
        target,
        np.zeros(SHAPE, dtype=bool),
        surface_envelope_A=SURFACE_ENVELOPE_A,
        geometry_source_id="deliberately-uncontracted-edge:v1",
        excluded_probe_power=0.05,
    )
    with pytest.raises(ValueError, match="boundary|footprint|padding"):
        build_atomistic_edit_model_1d(
            compact_silicon_experiment, _options(raw_edge)
        )


def test_public_options_are_only_the_object_agnostic_physics_contract():
    assert {field.name for field in fields(AtomisticEditOptions1D)} == {
        "max_host_removals",
        "max_extra_centres",
        "max_scattering_equivalent_per_centre",
        "minimum_separation_A",
        "expected_rms_host_strain",
        "edit_penalty_path",
        "discovery_support",
        "enable_material_energy_envelope",
    }
