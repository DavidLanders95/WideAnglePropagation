"""Authenticated persistence gates for AE-1 atomistic-edit snapshots."""

from dataclasses import replace
import json

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation import (  # noqa: E402
    ptychography_atomistic_edit_1d as atomistic_module,
)
from tests.atomistic_edit_test_helpers import (  # noqa: E402
    CompactAtomisticEditModelSpec1D,
    make_compact_atomistic_edit_model_1d,
)
from wide_angle_propagation.ptychography_atomistic_edit_1d import (  # noqa: E402
    atomistic_edit_active_parameter_count_1d,
    atomistic_edit_prior_components_1d,
    atomistic_edit_state_is_admissible_1d,
    empty_atomistic_edit_state_1d,
    load_atomistic_edit_snapshot_1d,
    make_atomistic_edit_snapshot_1d,
    render_atomistic_edit_potential_1d,
    save_atomistic_edit_snapshot_1d,
    validate_atomistic_edit_snapshot_1d,
)


SHAPE = (13, 13)
HOST_CENTRES = np.asarray([[6, 6], [6, 9]], dtype=np.int32)
TARGET_ANCHOR = (3, 3)
NUISANCE_ANCHOR = (9, 8)

EXPECTED_ARCHIVE_FIELDS = {
    "schema_version",
    "archive_contract",
    "host_reference_potential",
    "host_site_coordinates",
    "host_site_patches",
    "host_patch_starts",
    "host_control_coordinates_s",
    "host_control_coordinates_u",
    "host_axial_sampling",
    "host_transverse_sampling",
    "host_maximum_displacement",
    "host_metadata_json",
    "host_support_all_site_coordinates",
    "host_support_site_center_indices",
    "host_support_site_patch_starts",
    "host_support_site_patch_shapes",
    "host_support_target_pixel_mask",
    "host_support_forward_pixel_mask",
    "host_support_target_center_mask",
    "host_support_forward_relevant_mask",
    "host_support_site_role_codes",
    "host_support_modeled_site_indices",
    "host_support_target_influence_mask",
    "host_support_nuisance_influence_mask",
    "host_support_json",
    "discovery_axial_coordinates_A",
    "discovery_transverse_coordinates_A",
    "discovery_target_mask",
    "discovery_nuisance_mask",
    "discovery_json",
    "addition_kernel_values",
    "addition_kernel_centre_index",
    "addition_kernel_json",
    "edit_support_target_discovery_mask",
    "edit_support_nuisance_discovery_mask",
    "edit_support_addition_influence_mask",
    "edit_support_total_influence_mask",
    "edit_support_json",
    "model_axial_coordinates_A",
    "model_transverse_coordinates_A",
    "model_host_hard_core_pairs",
    "model_json",
    "state_host_removal_indices",
    "state_host_removal_fractions",
    "state_host_removal_active",
    "state_extra_anchor_indices",
    "state_extra_position_offsets_A",
    "state_extra_scattering_equivalents",
    "state_extra_active",
    "state_host_displacement_controls",
    "rendered_potential",
    "snapshot_json",
    "archive_sha256",
}

EXPECTED_JSON_FIELDS = {
    "host_support_json": {
        "schema_version",
        "classification_contract",
        "exterior_policy",
        "excluded_probe_power",
        "atomic_template_cutoff_A",
        "maximum_displacement_A",
        "fixed_material_provenance_id",
        "displacement_control_shape",
        "removed_displacement_dof",
        "registration_parameter_count",
        "maximum_nuisance_sites",
        "maximum_specimen_parameters",
        "parameter_counts",
        "contract_id",
    },
    "discovery_json": {
        "surface_envelope_A",
        "geometry_source_id",
        "excluded_probe_power",
        "contract_id",
        "metadata",
    },
    "addition_kernel_json": {
        "axial_sampling_A",
        "transverse_sampling_A",
        "host_equivalent_integrated_scattering",
        "parameterization_id",
        "cutoff_A",
        "projection_width_A",
        "boundary_mass_fraction",
        "normalization_tolerance",
        "kernel_id",
        "metadata",
    },
    "edit_support_json": {
        "schema_version",
        "host_support_contract_id",
        "discovery_contract_id",
        "kernel_id",
        "maximum_host_removals",
        "maximum_extra_centres",
        "maximum_scattering_equivalent_per_centre",
        "minimum_separation_A",
        "expected_rms_host_strain",
        "spatial_dimension",
        "deformation_parameter_count",
        "elastic_model_id",
        "hard_core_policy_id",
        "contract_id",
        "edit_penalty_path",
        "enable_material_energy_envelope",
    },
    "model_json": {"deformation_parameter_count", "model_id", "metadata"},
    "snapshot_json": {
        "active_parameter_count",
        "selected_edit_penalty",
        "edit_penalty_rule_id",
        "data_objective_value",
        "data_objective_id",
        "prior_components",
        "total_objective_value",
        "kkt_status",
        "capacity_status",
        "converged",
        "metadata",
        "snapshot_id",
    },
}

STATE_FIELDS = (
    "host_removal_indices",
    "host_removal_fractions",
    "host_removal_active",
    "extra_anchor_indices",
    "extra_position_offsets_A",
    "extra_scattering_equivalents",
    "extra_active",
    "host_displacement_controls",
)


def _compact_model():
    return make_compact_atomistic_edit_model_1d(
        CompactAtomisticEditModelSpec1D(
            shape=SHAPE,
            host_centres=HOST_CENTRES,
            target_discovery_centres=(TARGET_ANCHOR,),
            nuisance_discovery_centres=(NUISANCE_ANCHOR,),
            edit_penalty_path=(1.0, 0.5, 0.25),
            max_host_removals=2,
            max_extra_centres=3,
            deformation_parameter_count=6,
            fixture_id="ae1-persistence-test",
            reference_background=0.05,
            maximum_displacement_A=0.5,
        )
    )


@pytest.fixture(scope="module")
def compact_model():
    return _compact_model()


@pytest.fixture(scope="module")
def active_state(compact_model):
    state = empty_atomistic_edit_state_1d(compact_model)
    controls = np.zeros((2, 2, 2), dtype=np.float64)
    controls[1, :, 0] = 0.05
    controls[:, 1, 1] = 0.02
    return replace(
        state,
        host_removal_indices=jnp.asarray([0, 1]),
        host_removal_fractions=jnp.asarray([0.2, 0.4]),
        host_removal_active=jnp.asarray([True, True]),
        extra_anchor_indices=jnp.asarray(
            [TARGET_ANCHOR, NUISANCE_ANCHOR, TARGET_ANCHOR]
        ),
        extra_position_offsets_A=jnp.asarray(
            [[0.2, -0.1], [-0.25, 0.2], [0.0, 0.0]]
        ),
        extra_scattering_equivalents=jnp.asarray([0.35, 0.65, 0.0]),
        extra_active=jnp.asarray([True, True, False]),
        host_displacement_controls=jnp.asarray(controls),
    )


@pytest.fixture(scope="module")
def snapshot(compact_model, active_state):
    return make_atomistic_edit_snapshot_1d(
        compact_model,
        active_state,
        selected_edit_penalty=0.5,
        edit_penalty_rule_id="held-out-count-path:v1",
        data_objective_value=12.25,
        data_objective_id="calibrated-count-deviance:v1",
        metadata={"case": "compact-persistence", "seed": 7},
    )


@pytest.fixture(scope="module")
def archive_path(tmp_path_factory, snapshot):
    path = tmp_path_factory.mktemp("atomistic-edit-archive") / "snapshot.npz"
    save_atomistic_edit_snapshot_1d(path, snapshot)
    return path


def _read_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {
            name: np.array(archive[name], copy=True, order="C")
            for name in archive.files
        }


def _write_archive(path, payload, *, reseal):
    values = {name: np.asarray(value) for name, value in payload.items()}
    if reseal:
        unsigned = {
            name: value
            for name, value in values.items()
            if name != "archive_sha256"
        }
        values["archive_sha256"] = np.asarray(
            atomistic_module._archive_digest(unsigned)
        )
    np.savez_compressed(path, **values)


def _replace_json(payload, field, transform):
    decoded = json.loads(str(np.asarray(payload[field]).item()))
    transform(decoded)
    payload[field] = np.asarray(
        json.dumps(decoded, allow_nan=False, sort_keys=True, separators=(",", ":"))
    )


def test_snapshot_roundtrip_preserves_complete_rerenderable_contract(
    snapshot, archive_path
):
    loaded = load_atomistic_edit_snapshot_1d(archive_path)
    assert loaded.snapshot_id == snapshot.snapshot_id
    assert loaded.model.model_id == snapshot.model.model_id
    assert loaded.model.support_contract.contract_id == (
        snapshot.model.support_contract.contract_id
    )
    assert loaded.model.host_model.support_contract.contract_id == (
        snapshot.model.host_model.support_contract.contract_id
    )
    assert loaded.model.options.discovery_support.contract_id == (
        snapshot.model.options.discovery_support.contract_id
    )
    assert loaded.model.addition_kernel.kernel_id == (
        snapshot.model.addition_kernel.kernel_id
    )
    assert loaded.model.options.edit_penalty_path == (
        snapshot.model.options.edit_penalty_path
    )
    assert loaded.model.options.max_host_removals == 2
    assert loaded.model.options.max_extra_centres == 3
    assert dict(loaded.model.metadata) == dict(snapshot.model.metadata)
    assert dict(loaded.metadata) == dict(snapshot.metadata)

    for name in STATE_FIELDS:
        np.testing.assert_array_equal(
            getattr(loaded.state, name), getattr(snapshot.state, name)
        )
    for name in (
        "target_discovery_mask",
        "nuisance_discovery_mask",
        "addition_influence_mask",
        "total_influence_mask",
    ):
        np.testing.assert_array_equal(
            getattr(loaded.model.support_contract, name),
            getattr(snapshot.model.support_contract, name),
        )
    np.testing.assert_array_equal(
        loaded.model.host_model.reference_potential,
        snapshot.model.host_model.reference_potential,
    )
    np.testing.assert_array_equal(
        loaded.model.host_model.site_patches,
        snapshot.model.host_model.site_patches,
    )
    np.testing.assert_array_equal(
        loaded.model.addition_kernel.unit_integrated_values,
        snapshot.model.addition_kernel.unit_integrated_values,
    )
    np.testing.assert_array_equal(
        loaded.rendered_potential,
        render_atomistic_edit_potential_1d(loaded.model, loaded.state),
    )
    assert validate_atomistic_edit_snapshot_1d(loaded) is loaded


def test_roundtrip_preserves_prior_objective_and_active_count(snapshot, archive_path):
    loaded = load_atomistic_edit_snapshot_1d(archive_path)
    expected_prior = atomistic_edit_prior_components_1d(
        loaded.model, loaded.state, loaded.selected_edit_penalty
    )
    for name in (
        "edit_mass",
        "weighted_edit_penalty",
        "elastic_penalty",
        "hard_core_penalty",
        "total_prior",
    ):
        assert getattr(loaded.prior_components, name) == getattr(
            expected_prior, name
        )
        assert getattr(loaded.prior_components, name) == getattr(
            snapshot.prior_components, name
        )
    assert loaded.data_objective_value == snapshot.data_objective_value
    assert loaded.data_objective_id == snapshot.data_objective_id
    assert loaded.total_objective_value == (
        loaded.data_objective_value + loaded.prior_components.total_prior
    )
    assert loaded.active_parameter_count == (
        atomistic_edit_active_parameter_count_1d(loaded.model, loaded.state)
    )
    assert loaded.active_parameter_count == 6 + 2 + 3 * 2


def test_snapshot_identity_is_canonical_under_sparse_slot_permutations(
    compact_model, active_state, snapshot
):
    extra_order = jnp.asarray([1, 0, 2])
    permuted_state = replace(
        active_state,
        host_removal_indices=active_state.host_removal_indices[::-1],
        host_removal_fractions=active_state.host_removal_fractions[::-1],
        host_removal_active=active_state.host_removal_active[::-1],
        extra_anchor_indices=active_state.extra_anchor_indices[extra_order],
        extra_position_offsets_A=active_state.extra_position_offsets_A[extra_order],
        extra_scattering_equivalents=(
            active_state.extra_scattering_equivalents[extra_order]
        ),
        extra_active=active_state.extra_active[extra_order],
    )
    permuted = make_atomistic_edit_snapshot_1d(
        compact_model,
        permuted_state,
        selected_edit_penalty=0.5,
        edit_penalty_rule_id="held-out-count-path:v1",
        data_objective_value=12.25,
        data_objective_id="calibrated-count-deviance:v1",
        metadata={"case": "compact-persistence", "seed": 7},
    )
    assert permuted.snapshot_id == snapshot.snapshot_id
    for name in STATE_FIELDS:
        np.testing.assert_array_equal(
            getattr(permuted.state, name), getattr(snapshot.state, name)
        )

    dormant_values = replace(
        active_state,
        extra_anchor_indices=active_state.extra_anchor_indices.at[2].set(
            jnp.asarray(HOST_CENTRES[0])
        ),
        extra_position_offsets_A=(
            active_state.extra_position_offsets_A.at[2].set(
                jnp.asarray([0.4, -0.4])
            )
        ),
        extra_scattering_equivalents=(
            active_state.extra_scattering_equivalents.at[2].set(1.7)
        ),
    )
    dormant = make_atomistic_edit_snapshot_1d(
        compact_model,
        dormant_values,
        selected_edit_penalty=0.5,
        edit_penalty_rule_id="held-out-count-path:v1",
        data_objective_value=12.25,
        data_objective_id="calibrated-count-deviance:v1",
        metadata={"case": "compact-persistence", "seed": 7},
    )
    assert dormant.snapshot_id == snapshot.snapshot_id
    for name in STATE_FIELDS:
        np.testing.assert_array_equal(
            getattr(dormant.state, name), getattr(snapshot.state, name)
        )

    one_removal_state = replace(
        active_state,
        host_removal_fractions=active_state.host_removal_fractions.at[1].set(0.0),
        host_removal_active=active_state.host_removal_active.at[1].set(False),
    )
    one_removal = make_atomistic_edit_snapshot_1d(
        compact_model,
        one_removal_state,
        selected_edit_penalty=0.5,
        edit_penalty_rule_id="held-out-count-path:v1",
        data_objective_value=12.25,
        data_objective_id="calibrated-count-deviance:v1",
        metadata={"case": "one-removal-dormant-slot"},
    )
    dirty_removal = replace(
        one_removal_state,
        host_removal_fractions=(
            one_removal_state.host_removal_fractions.at[1].set(0.9)
        ),
    )
    canonical_dirty_removal = make_atomistic_edit_snapshot_1d(
        compact_model,
        dirty_removal,
        selected_edit_penalty=0.5,
        edit_penalty_rule_id="held-out-count-path:v1",
        data_objective_value=12.25,
        data_objective_id="calibrated-count-deviance:v1",
        metadata={"case": "one-removal-dormant-slot"},
    )
    assert canonical_dirty_removal.snapshot_id == one_removal.snapshot_id
    for name in STATE_FIELDS:
        np.testing.assert_array_equal(
            getattr(canonical_dirty_removal.state, name),
            getattr(one_removal.state, name),
        )


def test_archive_is_non_pickled_and_has_exact_schema(archive_path):
    with np.load(archive_path, allow_pickle=False) as archive:
        assert set(archive.files) == EXPECTED_ARCHIVE_FIELDS
        assert all(archive[name].dtype != object for name in archive.files)
        assert archive["schema_version"].dtype == np.dtype(np.int64)
        assert archive["schema_version"].shape == ()
        assert archive["schema_version"].item() == 1
        assert archive["archive_sha256"].dtype.kind == "U"
        assert len(str(archive["archive_sha256"].item())) == 64
        for field, expected in EXPECTED_JSON_FIELDS.items():
            decoded = json.loads(str(archive[field].item()))
            assert set(decoded) == expected
        prior = json.loads(str(archive["snapshot_json"].item()))[
            "prior_components"
        ]
        assert set(prior) == {
            "edit_mass",
            "weighted_edit_penalty",
            "elastic_penalty",
            "hard_core_penalty",
            "total_prior",
        }


def test_archive_digest_and_exact_field_set_reject_tampering(
    archive_path, tmp_path
):
    payload = _read_archive(archive_path)
    rendered = payload["rendered_potential"].copy()
    rendered[0, 0] += 1e-3
    payload["rendered_potential"] = rendered
    bad_digest = tmp_path / "bad-digest.npz"
    _write_archive(bad_digest, payload, reseal=False)
    with pytest.raises(ValueError, match="SHA-256 verification"):
        load_atomistic_edit_snapshot_1d(bad_digest)

    payload = _read_archive(archive_path)
    payload["unexpected_pickle_free_field"] = np.asarray(1, dtype=np.int64)
    bad_schema = tmp_path / "bad-schema.npz"
    _write_archive(bad_schema, payload, reseal=False)
    with pytest.raises(ValueError, match="fields differ from schema"):
        load_atomistic_edit_snapshot_1d(bad_schema)


def _tamper_resealed_payload(payload, layer):
    if layer == "model":
        _replace_json(
            payload,
            "model_json",
            lambda value: value.update(model_id="0" * 64),
        )
    elif layer == "support":
        mask = payload["edit_support_total_influence_mask"].copy()
        mask[0, 0] = ~mask[0, 0]
        payload["edit_support_total_influence_mask"] = mask
    elif layer == "kernel":
        values = payload["addition_kernel_values"].copy()
        values[2, 2] -= 0.01
        values[2, 1] += 0.01
        payload["addition_kernel_values"] = values
    elif layer == "state":
        masses = payload["state_extra_scattering_equivalents"].copy()
        masses[0] += 0.01
        payload["state_extra_scattering_equivalents"] = masses
    elif layer == "rendered":
        rendered = payload["rendered_potential"].copy()
        rendered[5, 5] += 0.01
        payload["rendered_potential"] = rendered
    elif layer == "objective":
        def change_objective(value):
            value["data_objective_value"] += 1.0
            value["total_objective_value"] += 1.0

        _replace_json(payload, "snapshot_json", change_objective)
    else:  # pragma: no cover - keeps the parameterized table exhaustive
        raise AssertionError(f"unknown tampering layer {layer}")


@pytest.mark.parametrize(
    ("layer", "message"),
    [
        ("model", "model_id"),
        ("support", "atomistic support"),
        ("kernel", "kernel_id"),
        ("state", "potential does not exactly rerender"),
        ("rendered", "potential does not exactly rerender"),
        ("objective", "snapshot_id"),
    ],
)
def test_resealed_layer_tampering_is_independently_rejected(
    archive_path, tmp_path, layer, message
):
    payload = _read_archive(archive_path)
    _tamper_resealed_payload(payload, layer)
    path = tmp_path / f"tampered-{layer}.npz"
    _write_archive(path, payload, reseal=True)
    with pytest.raises(ValueError, match=message):
        load_atomistic_edit_snapshot_1d(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"selected_edit_penalty": 0.75}, "exact member"),
        ({"selected_edit_penalty": 0.0}, "positive"),
        ({"edit_penalty_rule_id": ""}, "non-empty"),
        ({"edit_penalty_rule_id": "   "}, "non-empty"),
        ({"data_objective_value": np.nan}, "finite"),
        ({"data_objective_id": ""}, "non-empty"),
    ],
)
def test_snapshot_creation_validates_path_penalty_rule_and_objective_identity(
    compact_model, active_state, changes, message
):
    arguments = {
        "selected_edit_penalty": 0.5,
        "edit_penalty_rule_id": "held-out-count-path:v1",
        "data_objective_value": 12.25,
        "data_objective_id": "calibrated-count-deviance:v1",
    }
    arguments.update(changes)
    with pytest.raises((TypeError, ValueError), match=message):
        make_atomistic_edit_snapshot_1d(compact_model, active_state, **arguments)


def test_snapshot_rejects_hard_core_inadmissible_state(compact_model):
    state = empty_atomistic_edit_state_1d(compact_model)
    anchors = np.asarray(state.extra_anchor_indices).copy()
    anchors[0] = HOST_CENTRES[0]
    masses = np.zeros(compact_model.options.max_extra_centres)
    masses[0] = 1.0
    active = np.zeros(compact_model.options.max_extra_centres, dtype=bool)
    active[0] = True
    inadmissible = replace(
        state,
        extra_anchor_indices=jnp.asarray(anchors),
        extra_scattering_equivalents=jnp.asarray(masses),
        extra_active=jnp.asarray(active),
    )
    assert not atomistic_edit_state_is_admissible_1d(compact_model, inadmissible)
    with pytest.raises(ValueError, match="hard admissibility"):
        make_atomistic_edit_snapshot_1d(
            compact_model,
            inadmissible,
            selected_edit_penalty=0.5,
            edit_penalty_rule_id="held-out-count-path:v1",
            data_objective_value=12.25,
            data_objective_id="calibrated-count-deviance:v1",
        )


def test_ae1_status_is_fixed_fail_closed_in_memory_and_archive(
    snapshot, archive_path, tmp_path
):
    assert snapshot.kkt_status == "not_evaluated_ae1"
    assert snapshot.capacity_status == "not_evaluated_ae1"
    assert snapshot.converged is False
    for changes, message in (
        ({"kkt_status": "satisfied"}, "cannot claim a KKT evaluation"),
        ({"capacity_status": "within_capacity"}, "capacity assessment"),
        ({"converged": True}, "optimizer convergence"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_atomistic_edit_snapshot_1d(replace(snapshot, **changes))

    for field, value, message in (
        ("kkt_status", "satisfied", "cannot claim a KKT evaluation"),
        ("capacity_status", "within_capacity", "capacity assessment"),
        ("converged", True, "optimizer convergence"),
    ):
        payload = _read_archive(archive_path)
        _replace_json(
            payload,
            "snapshot_json",
            lambda decoded, field=field, value=value: decoded.update(
                {field: value}
            ),
        )
        path = tmp_path / f"false-{field}.npz"
        _write_archive(path, payload, reseal=True)
        with pytest.raises(ValueError, match=message):
            load_atomistic_edit_snapshot_1d(path)
