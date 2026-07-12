"""Focused gates for the narrow silicon atomistic-edit user facade."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

from collections.abc import Iterator, Mapping
from dataclasses import fields, replace
import importlib

import matplotlib
import numpy as np
import pytest
import wide_angle_propagation as package


matplotlib.use("Agg")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from tests.test_ptychography_atomistic_edit_workflow_1d import (  # noqa: E402
    compact_silicon_experiment as _compact_experiment_fixture,
)
from wide_angle_propagation.ptychography_1d import (  # noqa: E402
    GlancingScan1D,
    PtychographyObjective1D,
)
from wide_angle_propagation.ptychography_atomistic_edit_io_1d import (  # noqa: E402
    make_atomistic_edit_reconstruction_bundle_1d,
    save_atomistic_edit_reconstruction_bundle_1d,
)
from wide_angle_propagation.ptychography_atomistic_workflow_1d import (  # noqa: E402
    SiliconAtomisticEditConfig1D,
    SiliconAtomisticEditRun1D,
    load_silicon_atomistic_edit_run_1d,
    plot_silicon_atomistic_edit_run_1d,
    reconstruct_silicon_atomistic_edits_1d,
    save_silicon_atomistic_edit_run_1d,
    summarize_silicon_atomistic_edit_run_1d,
)
from wide_angle_propagation.ptychography_workflow_1d import (  # noqa: E402
    synthetic_noiseless_poisson_measurement_1d,
)


class _ForbiddenTruthMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError("the facade read a private truth field")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("the facade iterated over private truth fields")

    def __len__(self) -> int:
        raise AssertionError("the facade measured a private truth field")


@pytest.fixture(scope="module")
def facade_experiment():
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
def facade_objective(facade_experiment):
    return PtychographyObjective1D(
        kind="poisson_deviance",
        electrons_per_pattern=jnp.full(
            len(facade_experiment.window_starts), 2_000.0
        ),
        minimum_expected_electrons=1e-9,
        relative_signal_scale=1.0,
    )


@pytest.fixture(scope="module")
def facade_measurement(facade_experiment, facade_objective):
    probes = np.asarray(facade_experiment.input_probes)
    intensities = np.abs(
        np.fft.fftshift(np.fft.fft(probes, axis=-1), axes=-1)
    ) ** 2
    scan = GlancingScan1D(
        intensities=jnp.asarray(intensities),
        window_starts=facade_experiment.window_starts,
        scan_coordinates=facade_experiment.scan_coordinates,
        detector_angles=facade_experiment.detector_angles,
        metadata={"private_truth_note": "must not cross the facade"},
    )
    return synthetic_noiseless_poisson_measurement_1d(
        facade_experiment,
        scan,
        facade_objective,
        detector_valid_mask=np.ones_like(intensities, dtype=bool),
        calibration_id="compact-facade-noiseless-counts:v1",
    )


@pytest.fixture(scope="module")
def facade_config():
    return SiliconAtomisticEditConfig1D(
        edit_penalty_path=(1e12,),
        max_host_removals=2,
        max_extra_centres=2,
        max_scattering_equivalent_per_centre=2.0,
        minimum_separation_A=2.0,
        expected_rms_host_strain=0.1,
        vacuum_discovery_band_A=3.0,
        maximum_active_set_iterations=1,
        joint_refinement_updates=0,
        polish_updates=0,
        debias_updates=0,
        show_progress=False,
        evaluate_audit=False,
    )


@pytest.fixture(scope="module")
def facade_run(
    facade_experiment,
    facade_measurement,
    facade_objective,
    facade_config,
):
    private = _ForbiddenTruthMapping()
    poisoned = replace(
        facade_experiment,
        truth_potentials=private,
        truth_vacancy_fractions=private,
        truth_displacement_controls=private,
        truth_rigid_displacements=private,
        defect_site_indices=private,
    )
    events = []
    run = reconstruct_silicon_atomistic_edits_1d(
        poisoned,
        facade_measurement,
        facade_objective,
        config=facade_config,
        progress_callback=events.append,
    )
    return run, events


def test_config_is_small_object_free_and_requires_a_calibrated_path():
    expected_fields = {
        "edit_penalty_path",
        "max_host_removals",
        "max_extra_centres",
        "max_scattering_equivalent_per_centre",
        "minimum_separation_A",
        "expected_rms_host_strain",
        "vacuum_discovery_band_A",
        "maximum_active_set_iterations",
        "joint_refinement_updates",
        "polish_updates",
        "debias_updates",
        "training_scan_batch_size",
        "seed",
        "show_progress",
        "evaluate_audit",
    }
    assert {item.name for item in fields(SiliconAtomisticEditConfig1D)} == (
        expected_fields
    )
    with pytest.raises(TypeError, match="edit_penalty_path"):
        SiliconAtomisticEditConfig1D()
    with pytest.raises(ValueError, match="must not be empty"):
        SiliconAtomisticEditConfig1D(edit_penalty_path=())
    with pytest.raises(ValueError, match="strictly decreasing"):
        SiliconAtomisticEditConfig1D(edit_penalty_path=(1.0, 1.0))
    with pytest.raises(ValueError, match="capacity"):
        SiliconAtomisticEditConfig1D(
            edit_penalty_path=(1.0,),
            max_host_removals=0,
            max_extra_centres=0,
        )
    with pytest.raises(ValueError, match="must be positive"):
        SiliconAtomisticEditConfig1D(
            edit_penalty_path=(1.0,),
            maximum_active_set_iterations=0,
        )
    normalized = SiliconAtomisticEditConfig1D(edit_penalty_path=[2.0, 1.0])
    assert normalized.edit_penalty_path == (2.0, 1.0)
    assert normalized.training_scan_batch_size == 32
    with pytest.raises(ValueError, match="training_scan_batch_size"):
        SiliconAtomisticEditConfig1D(
            edit_penalty_path=(1.0,), training_scan_batch_size=0
        )


def test_package_root_presents_the_facade_not_legacy_inverse_entry_points():
    expected = {
        "SiliconAtomisticEditConfig1D",
        "SiliconAtomisticEditRun1D",
        "load_silicon_atomistic_edit_run_1d",
        "plot_silicon_atomistic_edit_run_1d",
        "reconstruct_silicon_atomistic_edits_1d",
        "save_silicon_atomistic_edit_run_1d",
        "summarize_silicon_atomistic_edit_run_1d",
    }
    assert expected <= set(package.__all__)
    legacy = {
        "reconstruct_experiment_1d",
        "reconstruct_potential_1d",
        "reconstruct_pixel_potential_1d",
        "reconstruct_lattice_site_potential_1d",
        "run_prepared_lattice_site_reconstruction_1d",
    }
    assert legacy.isdisjoint(package.__all__)
    core = importlib.import_module("wide_angle_propagation.ptychography_1d")
    workflow = importlib.import_module(
        "wide_angle_propagation.ptychography_workflow_1d"
    )
    assert legacy.isdisjoint(core.__all__)
    assert legacy.isdisjoint(workflow.__all__)


def test_facade_run_is_truth_free_derives_support_and_forwards_progress(
    facade_run,
):
    run, events = facade_run
    assert isinstance(run, SiliconAtomisticEditRun1D)
    assert run.prepared.metadata["truth_fields_read"] is False
    assert run.surface_envelope_A == (-3.0, 3.0)
    assert run.prepared.model.options.edit_penalty_path == (1e12,)
    assert run.prepared.model.options.max_host_removals == 2
    assert run.prepared.model.options.max_extra_centres == 2
    phases = [event.phase for event in events]
    assert phases[0] == "initial"
    assert "lambda_complete" in phases
    assert phases[-1] == "debias"
    assert run.result.prepared_problem_id == run.prepared.reconstruction_problem_id
    assert run.result.metadata["audit_evaluated"] is False
    assert run.solver_options.training_scan_batch_size == 32
    assert run.result.metadata["training_scan_batch_size"] == 32


def test_plot_helper_keeps_authenticated_target_view(
    facade_experiment,
    facade_run,
):
    run, _ = facade_run
    figure = plot_silicon_atomistic_edit_run_1d(
        facade_experiment,
        run,
        truth_potential=np.asarray(run.prepared.model.host_model.reference_potential),
    )
    image_axes = [axis for axis in figure.axes if axis.images]
    assert len(image_axes) == 2
    assert [axis.get_title() for axis in image_axes] == [
        "Private full truth (TARGET support crop)",
        "Reference + TARGET edit/deformation deltas",
    ]
    summary = summarize_silicon_atomistic_edit_run_1d(
        facade_experiment, run
    )
    assert summary["active_parameter_count"] == run.result.active_parameter_count
    assert summary["edit_counts_by_role"] == {
        "target": {
            "host_removals": 0,
            "extra_centres": 0,
            "total_active_edits": 0,
        },
        "nuisance": {
            "host_removals": 0,
            "extra_centres": 0,
            "total_active_edits": 0,
        },
    }
    target_deformation = summary["target_site_displacements"]
    assert len(target_deformation["host_site_indices"]) == 1
    np.testing.assert_allclose(target_deformation["vectors_A"], 0.0)
    target_strain = summary["derived_target_strain"]
    assert target_strain["units"] == "dimensionless"
    np.testing.assert_allclose(target_strain["strain_tensor"], 0.0)


def test_save_load_helper_replays_the_same_prepared_result(
    tmp_path,
    facade_run,
):
    run, _ = facade_run
    path = tmp_path / "silicon_atomistic_edit_run.npz"
    save_silicon_atomistic_edit_run_1d(
        path,
        run,
        provenance={"test_scope": "compact_facade_replay"},
    )
    loaded = load_silicon_atomistic_edit_run_1d(path)
    assert len(loaded.archive_id) == 64
    assert loaded.prepared.reconstruction_problem_id == (
        run.prepared.reconstruction_problem_id
    )
    assert loaded.result.prepared_problem_id == run.result.prepared_problem_id
    assert loaded.result.active_parameter_count == run.result.active_parameter_count
    np.testing.assert_array_equal(
        loaded.result.debiased_state.host_removal_active,
        run.result.debiased_state.host_removal_active,
    )
    np.testing.assert_array_equal(
        loaded.result.debiased_state.extra_active,
        run.result.debiased_state.extra_active,
    )

    generic_path = tmp_path / "generic_ae2_bundle.npz"
    generic = make_atomistic_edit_reconstruction_bundle_1d(
        run.prepared,
        run.result,
        solver_options=run.solver_options,
        provenance={"workflow": "lower_level_generic_ae2"},
    )
    save_atomistic_edit_reconstruction_bundle_1d(generic_path, generic)
    with pytest.raises(ValueError, match="not created by the silicon AE facade"):
        load_silicon_atomistic_edit_run_1d(generic_path)
