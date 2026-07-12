"""CPU gates for the maintained atomistic-edit workflow boundary."""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from dataclasses import replace
from types import MappingProxyType

import matplotlib
import numpy as np
import pytest


matplotlib.use("Agg")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from tests.test_ptychography_atomistic_edit_workflow_1d import (  # noqa: E402
    compact_silicon_experiment as _compact_experiment_fixture,
)
from wide_angle_propagation.ptychography_1d import (  # noqa: E402
    GlancingScan1D,
    LatticeSiteModel1D,
    PtychographyObjective1D,
    lattice_site_displacements_1d,
    render_lattice_site_potential_from_displacements_1d,
)
from wide_angle_propagation.ptychography_atomistic_edit_1d import (  # noqa: E402
    AtomisticEditOptions1D,
    empty_atomistic_edit_state_1d,
    render_atomistic_edit_potential_1d,
)
from wide_angle_propagation.ptychography_workflow_1d import (  # noqa: E402
    _atomistic_target_view_arrays_1d,
    _target_addition_influence_mask_1d,
    _target_only_atomistic_edit_state_1d,
    build_atomistic_edit_discovery_support_1d,
    plot_atomistic_edit_reconstruction_1d,
    prepare_atomistic_edit_experiment_1d,
    synthetic_noiseless_poisson_measurement_1d,
)


SURFACE_ENVELOPE_A = (-3.0, 3.0)


@pytest.fixture(scope="module")
def adapter_experiment():
    # Reuse the renderer-focused compact host, but supply a real nonzero probe
    # and a nonempty guard partition for this workflow boundary.
    experiment = _compact_experiment_fixture.__wrapped__()
    u_A = np.asarray(experiment.transverse_coordinates, dtype=float)
    base = np.exp(-0.5 * (u_A / 3.0) ** 2) * np.exp(0.1j * u_A)
    probes = np.stack([np.roll(base, shift) for shift in range(6)])
    return replace(
        experiment,
        input_probes=jnp.asarray(probes),
        training_indices=jnp.asarray([1, 2, 3], dtype=jnp.int32),
        validation_indices=jnp.asarray([0], dtype=jnp.int32),
        audit_indices=jnp.asarray([5], dtype=jnp.int32),
        guard_indices=jnp.asarray([4], dtype=jnp.int32),
    )


@pytest.fixture(scope="module")
def adapter_scan(adapter_experiment):
    probes = np.asarray(adapter_experiment.input_probes)
    intensities = np.abs(
        np.fft.fftshift(np.fft.fft(probes, axis=-1), axes=-1)
    ) ** 2
    return GlancingScan1D(
        intensities=jnp.asarray(intensities),
        window_starts=adapter_experiment.window_starts,
        scan_coordinates=adapter_experiment.scan_coordinates,
        detector_angles=adapter_experiment.detector_angles,
        metadata={"synthetic_truth_secret": "must_not_cross_adapter"},
    )


@pytest.fixture(scope="module")
def objective(adapter_experiment):
    return PtychographyObjective1D(
        kind="poisson_deviance",
        electrons_per_pattern=jnp.arange(
            1000.0,
            1000.0 + len(adapter_experiment.window_starts),
        ),
        minimum_expected_electrons=1e-9,
        relative_signal_scale=1.0,
    )


@pytest.fixture(scope="module")
def measurement(adapter_experiment, adapter_scan, objective):
    return synthetic_noiseless_poisson_measurement_1d(
        adapter_experiment,
        adapter_scan,
        objective,
        detector_valid_mask=np.ones(
            np.asarray(adapter_scan.intensities).shape, dtype=bool
        ),
    )


@pytest.fixture(scope="module")
def edit_options(adapter_experiment):
    discovery = build_atomistic_edit_discovery_support_1d(
        adapter_experiment,
        surface_envelope_A=SURFACE_ENVELOPE_A,
    )
    return AtomisticEditOptions1D(
        max_host_removals=2,
        max_extra_centres=3,
        max_scattering_equivalent_per_centre=2.0,
        minimum_separation_A=2.0,
        expected_rms_host_strain=0.1,
        edit_penalty_path=(1.0, 0.5),
        discovery_support=discovery,
    )


@pytest.fixture(scope="module")
def prepared(adapter_experiment, measurement, objective, edit_options):
    return prepare_atomistic_edit_experiment_1d(
        adapter_experiment,
        measurement,
        objective,
        edit_options,
        surface_envelope_A=SURFACE_ENVELOPE_A,
    )


def _mixed_target_nuisance_state(prepared):
    model = prepared.model
    state = empty_atomistic_edit_state_1d(model)
    target_anchor = np.argwhere(
        model.options.discovery_support.target_mask
    )[0]
    nuisance_anchor = np.argwhere(
        model.options.discovery_support.nuisance_mask
    )[0]
    return replace(
        state,
        host_removal_indices=jnp.asarray([0, 1], dtype=jnp.int32),
        host_removal_fractions=jnp.asarray([0.4, 0.9]),
        host_removal_active=jnp.asarray([True, True]),
        extra_anchor_indices=jnp.asarray(
            [target_anchor, nuisance_anchor, target_anchor], dtype=jnp.int32
        ),
        extra_position_offsets_A=jnp.zeros((3, 2)),
        extra_scattering_equivalents=jnp.asarray([0.2, 0.7, 0.0]),
        extra_active=jnp.asarray([True, True, False]),
    )


def test_synthetic_fft_conversion_is_calibrated_noiseless_and_truth_free(
    adapter_experiment,
    adapter_scan,
    objective,
    measurement,
):
    assert jax.default_backend() == "cpu"
    np.testing.assert_allclose(
        np.sum(np.asarray(measurement.calibrated_signal_electrons), axis=1),
        np.asarray(objective.electrons_per_pattern),
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_array_equal(
        measurement.calibrated_signal_electrons,
        measurement.observed_total_electrons,
    )
    assert np.all(np.asarray(measurement.calibrated_dark_electrons_per_pixel) == 0)
    assert np.all(
        np.asarray(measurement.calibrated_read_noise_std_electrons) == 0
    )
    assert measurement.metadata["synthetic_only"] is True
    assert measurement.metadata["random_counts_drawn"] is False
    assert measurement.metadata["relative_signal_scale_fitted"] is False
    assert measurement.metadata["truth_fields_accepted"] is False
    assert not any("secret" in key for key in measurement.metadata)

    valid = np.ones(np.asarray(adapter_scan.intensities).shape, dtype=bool)
    valid[0, 0] = False
    intensities = np.asarray(adapter_scan.intensities).copy()
    intensities[0, 0] = np.nan
    masked_scan = replace(adapter_scan, intensities=intensities)
    masked = synthetic_noiseless_poisson_measurement_1d(
        adapter_experiment,
        masked_scan,
        objective,
        detector_valid_mask=valid,
    )
    assert bool(np.asarray(masked.valid_mask)[0, 0]) is False
    assert float(np.asarray(masked.calibrated_signal_electrons)[0, 0]) == 0.0


def test_preparation_is_truth_independent_and_preserves_exact_partition(
    adapter_experiment,
    measurement,
    objective,
    edit_options,
    prepared,
):
    poisoned = replace(
        adapter_experiment,
        truth_potentials={"private": np.full((2, 2), np.nan)},
        truth_vacancy_fractions={"private": np.asarray([123.0])},
        truth_displacement_controls={"private": np.asarray([456.0])},
        truth_rigid_displacements={"private": np.asarray([789.0])},
        defect_site_indices={"private": np.asarray([999])},
    )
    independently_prepared = prepare_atomistic_edit_experiment_1d(
        poisoned,
        measurement,
        objective,
        edit_options,
        surface_envelope_A=SURFACE_ENVELOPE_A,
    )
    assert independently_prepared.reconstruction_problem_id == (
        prepared.reconstruction_problem_id
    )
    assert independently_prepared.model.model_id == prepared.model.model_id
    assert independently_prepared.metadata["truth_fields_read"] is False
    np.testing.assert_array_equal(prepared.training_indices, [1, 2, 3])
    np.testing.assert_array_equal(prepared.validation_indices, [0])
    np.testing.assert_array_equal(prepared.audit_indices, [5])
    np.testing.assert_array_equal(prepared.excluded_indices, [4])


def test_target_view_resets_nuisance_edits_and_preserves_fixed_exterior(
    adapter_experiment,
    prepared,
):
    state = _mixed_target_nuisance_state(prepared)
    display_state = _target_only_atomistic_edit_state_1d(prepared.model, state)
    np.testing.assert_array_equal(
        display_state.host_removal_active, [True, False]
    )
    np.testing.assert_array_equal(display_state.extra_active, [True, False, False])
    reportable = np.asarray(
        prepared.model.host_model.support_contract.target_influence_mask
    ) | _target_addition_influence_mask_1d(prepared.model)
    displayed_full_grid = np.asarray(
        render_atomistic_edit_potential_1d(prepared.model, display_state)
    )
    np.testing.assert_array_equal(
        displayed_full_grid[~reportable],
        np.asarray(prepared.model.host_model.reference_potential)[~reportable],
    )

    recovered, truth, returned_mask = _atomistic_target_view_arrays_1d(
        adapter_experiment,
        prepared,
        state,
        truth_state=empty_atomistic_edit_state_1d(prepared.model),
    )
    np.testing.assert_array_equal(returned_mask, reportable)
    np.testing.assert_allclose(
        recovered[reportable], displayed_full_grid[reportable]
    )
    assert np.all(np.isnan(recovered[~reportable]))
    assert truth is not None and np.all(np.isnan(truth[~reportable]))


def test_target_view_excludes_nuisance_displacement_on_overlapping_footprints(
    adapter_experiment,
    prepared,
):
    model = prepared.model
    host_model = model.host_model
    state = empty_atomistic_edit_state_1d(model)
    controls = np.zeros_like(np.asarray(state.host_displacement_controls))
    controls[0, :, 0] = -0.2
    controls[-1, :, 0] = 0.4
    controls[..., 1] = 0.25
    state = replace(state, host_displacement_controls=jnp.asarray(controls))

    target = np.asarray(adapter_experiment.modeled_target_site_mask, dtype=bool)
    target_model = LatticeSiteModel1D(
        reference_potential=host_model.reference_potential,
        site_coordinates=host_model.site_coordinates[target],
        site_patches=host_model.site_patches[target],
        patch_starts=host_model.patch_starts[target],
        control_coordinates_s=host_model.control_coordinates_s,
        control_coordinates_u=host_model.control_coordinates_u,
        axial_sampling=host_model.axial_sampling,
        transverse_sampling=host_model.transverse_sampling,
        maximum_displacement=host_model.maximum_displacement,
    )
    site_displacements = lattice_site_displacements_1d(
        host_model.site_coordinates,
        state.host_displacement_controls,
        host_model.control_coordinates_s,
        host_model.control_coordinates_u,
    )
    expected_target_host = np.asarray(
        render_lattice_site_potential_from_displacements_1d(
            target_model,
            np.zeros(np.count_nonzero(target)),
            site_displacements[target],
        )
    )
    full_deformed_host = np.asarray(
        render_atomistic_edit_potential_1d(model, state)
    )
    target_influence = np.asarray(
        host_model.support_contract.target_influence_mask, dtype=bool
    )
    nuisance_influence = np.asarray(
        host_model.support_contract.nuisance_influence_mask, dtype=bool
    )
    overlap = target_influence & nuisance_influence
    assert np.any(overlap)
    assert np.max(
        np.abs(full_deformed_host[overlap] - expected_target_host[overlap])
    ) > 1e-6

    recovered, _, reportable = _atomistic_target_view_arrays_1d(
        adapter_experiment,
        prepared,
        state,
    )
    np.testing.assert_allclose(
        recovered[reportable],
        expected_target_host[reportable],
        rtol=2e-14,
        atol=2e-14,
    )


def test_target_plot_masks_everything_outside_reportable_influence(
    adapter_experiment,
    prepared,
):
    state = _mixed_target_nuisance_state(prepared)
    figure = plot_atomistic_edit_reconstruction_1d(
        adapter_experiment,
        prepared,
        state,
        truth_potential=np.asarray(
            prepared.model.host_model.reference_potential
        ),
    )
    image_axes = [axis for axis in figure.axes if axis.images]
    assert len(image_axes) == 2
    assert [axis.get_title() for axis in image_axes] == [
        "Private full truth (TARGET support crop)",
        "Reference + TARGET edit/deformation deltas",
    ]
    reportable = np.asarray(
        prepared.model.host_model.support_contract.target_influence_mask
    ) | _target_addition_influence_mask_1d(prepared.model)
    for axis in image_axes:
        plotted = axis.images[0].get_array()
        np.testing.assert_array_equal(
            np.ma.getmaskarray(plotted),
            (~reportable).T,
        )


def test_surface_support_and_target_label_mismatches_fail_closed(
    adapter_experiment,
    measurement,
    objective,
    edit_options,
    prepared,
):
    wrong_discovery = build_atomistic_edit_discovery_support_1d(
        adapter_experiment,
        surface_envelope_A=(-2.0, 3.0),
    )
    with pytest.raises(ValueError, match="does not match the geometry-derived"):
        prepare_atomistic_edit_experiment_1d(
            adapter_experiment,
            measurement,
            objective,
            replace(edit_options, discovery_support=wrong_discovery),
            surface_envelope_A=SURFACE_ENVELOPE_A,
        )

    bad_metadata = MappingProxyType(
        {**dict(prepared.metadata), "atomistic_edit_model_id": "wrong-model"}
    )
    mismatched = replace(prepared, metadata=bad_metadata)
    with pytest.raises(ValueError, match="TARGET-labelled output is forbidden"):
        plot_atomistic_edit_reconstruction_1d(
            adapter_experiment,
            mismatched,
            empty_atomistic_edit_state_1d(prepared.model),
        )
