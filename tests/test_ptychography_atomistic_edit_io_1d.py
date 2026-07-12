"""Authenticated AE-2 archive and independent replay tests."""

from dataclasses import replace
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("optax", reason="the ptychography extra is not installed")

from tests.atomistic_edit_test_helpers import (  # noqa: E402
    CompactAtomisticEditModelSpec1D,
    CompactPreparedProblemSpec1D,
    make_compact_atomistic_edit_model_1d,
    make_compact_prepared_atomistic_edit_problem_1d,
)
from wide_angle_propagation.ptychography_atomistic_edit_1d import (  # noqa: E402
    render_atomistic_edit_potential_1d,
)
import wide_angle_propagation.ptychography_atomistic_edit_io_1d as io_module  # noqa: E402
from wide_angle_propagation.ptychography_atomistic_edit_io_1d import (  # noqa: E402
    _archive_digest,
    load_atomistic_edit_reconstruction_bundle_1d,
    make_atomistic_edit_reconstruction_bundle_1d,
    save_atomistic_edit_reconstruction_bundle_1d,
)
from wide_angle_propagation.ptychography_atomistic_edit_solver_1d import (  # noqa: E402
    AtomisticEditSolverOptions1D,
    run_prepared_atomistic_edit_reconstruction_1d,
)


SHAPE = (9, 9)
HOST_CENTRE = np.asarray([[4, 6]], dtype=np.int32)


def _model():
    return make_compact_atomistic_edit_model_1d(
        CompactAtomisticEditModelSpec1D(
            shape=SHAPE,
            host_centres=HOST_CENTRE,
            target_discovery_centres=((2, 2),),
            nuisance_discovery_centres=((6, 2),),
            edit_penalty_path=(1e6,),
            max_host_removals=1,
            max_extra_centres=1,
            deformation_parameter_count=8,
            fixture_id="ae2-archive-test",
            reference_background=0.01,
            maximum_displacement_A=0.2,
        ),
    )


def _prepared(model):
    return make_compact_prepared_atomistic_edit_problem_1d(
        model,
        CompactPreparedProblemSpec1D(
            window_starts=(0, 1, 0),
            window_length=8,
            probe_shifts=(-1, 0, 1),
            validation_indices=(1,),
            audit_indices=(2,),
            electrons_per_pattern=100_000.0,
            fixture_id="ae2-archive-test",
        ),
    )


@pytest.fixture(scope="module")
def archived_case(tmp_path_factory):
    assert jax.default_backend() == "cpu"
    prepared = _prepared(_model())
    options = AtomisticEditSolverOptions1D(
        maximum_active_set_iterations=1,
        joint_refinement_updates=0,
        polish_updates=0,
        debias_updates=0,
        proposal_grid_kkt_tolerance=1e-8,
        active_projected_gradient_tolerance=1e6,
        debias_projected_gradient_tolerance=1e6,
        training_scan_batch_size=1,
        seed=23,
    )
    result = run_prepared_atomistic_edit_reconstruction_1d(
        prepared, options=options
    )
    assert result.converged
    bundle = make_atomistic_edit_reconstruction_bundle_1d(
        prepared,
        result,
        solver_options=options,
        provenance={"blind_case": "pristine-test-fixture"},
    )
    path = tmp_path_factory.mktemp("ae2-archive") / "result.npz"
    save_atomistic_edit_reconstruction_bundle_1d(path, bundle)
    return path, prepared, result, options


def _payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.array(archive[name], copy=True, order="C")
            for name in archive.files
        }


def _write(path: Path, payload: dict[str, np.ndarray], *, reseal: bool) -> None:
    if reseal:
        body = {
            name: value
            for name, value in payload.items()
            if name != "archive_sha256"
        }
        payload["archive_sha256"] = np.asarray(_archive_digest(body))
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)


def _assert_state_equal(first, second):
    for name in (
        "host_removal_indices",
        "host_removal_fractions",
        "host_removal_active",
        "extra_anchor_indices",
        "extra_position_offsets_A",
        "extra_scattering_equivalents",
        "extra_active",
        "host_displacement_controls",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_roundtrip_reconstructs_exact_problem_path_states_and_selected_fit(
    archived_case,
):
    path, prepared, result, options = archived_case
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
    loaded = load_atomistic_edit_reconstruction_bundle_1d(path)
    assert loaded.prepared.reconstruction_problem_id == (
        prepared.reconstruction_problem_id
    )
    assert loaded.prepared.model.model_id == prepared.model.model_id
    assert loaded.solver_options == options
    assert len(loaded.archive_id) == 64
    assert loaded.provenance["jax_default_backend"] == "cpu"
    assert loaded.provenance["caller_metadata"] == {
        "blind_case": "pristine-test-fixture"
    }
    replayed = loaded.reconstruction
    assert replayed.metadata == result.metadata
    assert replayed.selected_path_index == result.selected_path_index
    assert replayed.active_parameter_count == result.active_parameter_count
    _assert_state_equal(replayed.penalized_state, result.penalized_state)
    _assert_state_equal(replayed.debiased_state, result.debiased_state)
    _assert_state_equal(replayed.path_points[0].state, result.path_points[0].state)
    assert replayed.path_points[0].training_objective.total_objective == (
        result.path_points[0].training_objective.total_objective
    )
    np.testing.assert_array_equal(
        render_atomistic_edit_potential_1d(
            loaded.prepared.model, replayed.debiased_state
        ),
        render_atomistic_edit_potential_1d(
            prepared.model, result.debiased_state
        ),
    )


def test_archive_replay_uses_the_recorded_exact_scan_batching(
    archived_case, monkeypatch
):
    path, _, _, options = archived_case
    proposal_batch_sizes = []
    gradient_batch_sizes = []
    original_proposals = io_module.atomistic_edit_proposal_scores_1d
    original_gradients = io_module._objective_value_and_gradients

    def proposals(*args, **kwargs):
        proposal_batch_sizes.append(kwargs.get("training_scan_batch_size"))
        return original_proposals(*args, **kwargs)

    def gradients(*args, **kwargs):
        gradient_batch_sizes.append(kwargs.get("training_scan_batch_size"))
        return original_gradients(*args, **kwargs)

    monkeypatch.setattr(
        io_module, "atomistic_edit_proposal_scores_1d", proposals
    )
    monkeypatch.setattr(
        io_module, "_objective_value_and_gradients", gradients
    )
    load_atomistic_edit_reconstruction_bundle_1d(path)

    assert proposal_batch_sizes
    assert gradient_batch_sizes
    assert set(proposal_batch_sizes) == {options.training_scan_batch_size}
    assert set(gradient_batch_sizes) == {options.training_scan_batch_size}


def test_archive_rejects_checksum_tampering_and_extra_fields(
    archived_case, tmp_path
):
    source, _, _, _ = archived_case
    tampered = _payload(source)
    signal = tampered["prepared_measurement_signal"].copy()
    signal.flat[0] += 1.0
    tampered["prepared_measurement_signal"] = signal
    tampered_path = tmp_path / "tampered.npz"
    _write(tampered_path, tampered, reseal=False)
    with pytest.raises(ValueError, match="SHA-256"):
        load_atomistic_edit_reconstruction_bundle_1d(tampered_path)

    extra = _payload(source)
    extra["not_in_schema"] = np.asarray(1, dtype=np.int64)
    extra_path = tmp_path / "extra.npz"
    _write(extra_path, extra, reseal=False)
    with pytest.raises(ValueError, match="fields differ from schema"):
        load_atomistic_edit_reconstruction_bundle_1d(extra_path)


def test_archive_rejects_resealed_objective_and_problem_identity_changes(
    archived_case, tmp_path
):
    source, _, _, _ = archived_case
    objective_payload = _payload(source)
    result_fields = json.loads(str(objective_payload["result_json"].item()))
    result_fields["path_points"][0]["training_objective"][
        "count_deviance"
    ] += 1.0
    objective_payload["result_json"] = np.asarray(
        json.dumps(
            result_fields,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    objective_path = tmp_path / "resealed-objective.npz"
    _write(objective_path, objective_payload, reseal=True)
    with pytest.raises(ValueError, match="not numerically reproducible"):
        load_atomistic_edit_reconstruction_bundle_1d(objective_path)

    identity_payload = _payload(source)
    prepared_fields = json.loads(str(identity_payload["prepared_json"].item()))
    prepared_fields["reconstruction_problem_id"] = "0" * 64
    identity_payload["prepared_json"] = np.asarray(
        json.dumps(
            prepared_fields,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    identity_path = tmp_path / "resealed-identity.npz"
    _write(identity_path, identity_payload, reseal=True)
    with pytest.raises(ValueError, match="does not replay"):
        load_atomistic_edit_reconstruction_bundle_1d(identity_path)


def test_bundle_requires_the_exact_solver_options(archived_case):
    _, prepared, result, options = archived_case
    with pytest.raises(ValueError, match="metadata|tolerance"):
        make_atomistic_edit_reconstruction_bundle_1d(
            prepared,
            result,
            solver_options=replace(options, seed=options.seed + 1),
        )


def test_archive_retains_workflow_appended_prepared_provenance(
    archived_case, tmp_path
):
    _, prepared, result, options = archived_case
    enriched = replace(
        prepared,
        metadata={
            **dict(prepared.metadata),
            "workflow_adapter": "test_truth_free_adapter:v1",
            "experiment_geometry_id": "c" * 64,
            "truth_fields_read": False,
        },
    )
    bundle = make_atomistic_edit_reconstruction_bundle_1d(
        enriched,
        result,
        solver_options=options,
    )
    path = tmp_path / "workflow-metadata.npz"
    save_atomistic_edit_reconstruction_bundle_1d(path, bundle)
    loaded = load_atomistic_edit_reconstruction_bundle_1d(path)
    assert dict(loaded.prepared.metadata) == dict(enriched.metadata)
