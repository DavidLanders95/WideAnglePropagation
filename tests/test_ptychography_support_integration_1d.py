"""Integration tests for material-support contracts in lattice reconstruction."""

from dataclasses import replace
import json

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax", reason="the ptychography extra is not installed")
jax.config.update("jax_enable_x64", True)

import wide_angle_propagation.ptychography_1d as ptychography_1d_module  # noqa: E402

from wide_angle_propagation.ptychography_1d import (  # noqa: E402
    LatticeSiteModel1D,
    load_lattice_site_reconstruction_1d,
    prepare_lattice_site_reconstruction_1d,
    run_prepared_lattice_site_reconstruction_1d,
    save_lattice_site_reconstruction_1d,
)
from wide_angle_propagation.ptychography_ensemble_1d import (  # noqa: E402
    MultistartOptions1D,
    summarize_lattice_site_ensemble_1d,
)
from wide_angle_propagation.ptychography_support_contract_1d import (  # noqa: E402
    LatticeSiteRole1D,
    classify_lattice_site_support_1d,
)


ENERGY_EV = 30_000.0
POTENTIAL_SHAPE = (5, 8)
AXIAL_SAMPLING_A = 0.4
TRANSVERSE_SAMPLING_A = 0.3


def _site_geometry():
    centers = np.asarray([[1, 2], [3, 5]], dtype=np.int64)
    starts = np.asarray([[0, 1], [2, 4]], dtype=np.int64)
    shapes = np.full((2, 2), 3, dtype=np.int64)
    coordinates = centers * np.asarray(
        [AXIAL_SAMPLING_A, TRANSVERSE_SAMPLING_A]
    )
    patch = np.asarray(
        [
            [0.0, 0.1, 0.0],
            [0.1, 0.9, 0.1],
            [0.0, 0.1, 0.0],
        ],
        dtype=np.float64,
    )
    patches = np.stack([patch, patch])
    reference = np.full(POTENTIAL_SHAPE, 0.02, dtype=np.float64)
    for start, site_patch in zip(starts, patches):
        start_s, start_u = start
        reference[
            start_s : start_s + site_patch.shape[0],
            start_u : start_u + site_patch.shape[1],
        ] += site_patch
    return coordinates, centers, starts, shapes, patches, reference


def _support_contract(*, fixed_exterior: bool = False):
    coordinates, centers, starts, shapes, _, _ = _site_geometry()
    target = np.zeros(POTENTIAL_SHAPE, dtype=bool)
    target[tuple(centers[0])] = True
    forward = np.ones(POTENTIAL_SHAPE, dtype=bool)
    known_fixed = np.asarray([False, fixed_exterior], dtype=bool)
    return classify_lattice_site_support_1d(
        coordinates,
        centers,
        starts,
        shapes,
        target,
        forward,
        known_fixed_site_mask=known_fixed,
        fixed_material_provenance_id=(
            "matched-pristine-exterior:v1" if fixed_exterior else None
        ),
        excluded_probe_power=1e-5,
        atomic_template_cutoff_A=2.0,
        maximum_displacement_A=0.0,
        displacement_control_shape=(2, 2, 2),
        maximum_nuisance_sites=4,
        maximum_specimen_parameters=32,
        strict=True,
    )


def _model(contract=None):
    coordinates, _, starts, _, patches, reference = _site_geometry()
    modeled = (
        np.arange(len(coordinates), dtype=np.int64)
        if contract is None
        else np.asarray(contract.modeled_site_indices)
    )
    return LatticeSiteModel1D(
        reference_potential=jnp.asarray(reference),
        site_coordinates=jnp.asarray(coordinates[modeled]),
        site_patches=jnp.asarray(patches[modeled]),
        patch_starts=jnp.asarray(starts[modeled]),
        control_coordinates_s=jnp.asarray(
            [0.0, (POTENTIAL_SHAPE[0] - 1) * AXIAL_SAMPLING_A]
        ),
        control_coordinates_u=jnp.asarray(
            [0.0, (POTENTIAL_SHAPE[1] - 1) * TRANSVERSE_SAMPLING_A]
        ),
        axial_sampling=AXIAL_SAMPLING_A,
        transverse_sampling=TRANSVERSE_SAMPLING_A,
        maximum_displacement=0.0,
        metadata={"species": "Si", "fixture": "support-integration"},
        support_contract=contract,
    )


def _prepare(model, *, strict=True):
    u = (jnp.arange(POTENTIAL_SHAPE[1]) - 3.5) * TRANSVERSE_SAMPLING_A
    probe = jnp.exp(-0.5 * (u / 0.6) ** 2) * jnp.exp(0.2j * u)
    return prepare_lattice_site_reconstruction_1d(
        model,
        probe,
        jnp.asarray([0, 1]),
        4,
        jnp.ones(POTENTIAL_SHAPE[1], dtype=jnp.complex128),
        AXIAL_SAMPLING_A,
        ENERGY_EV,
        jnp.ones((2, POTENTIAL_SHAPE[1]), dtype=jnp.float64),
        validation_indices=[1],
        potential_max=4.0,
        minibatch_size=1,
        evaluation_batch_size=1,
        rematerialize=False,
        require_complete_material_scope=strict,
    )


@pytest.fixture(scope="module")
def nuisance_contract():
    return _support_contract(fixed_exterior=False)


@pytest.fixture(scope="module")
def strict_prepared(nuisance_contract):
    return _prepare(_model(nuisance_contract))


@pytest.fixture(scope="module")
def strict_result(strict_prepared):
    return run_prepared_lattice_site_reconstruction_1d(
        strict_prepared,
        initial_vacancy_fractions=jnp.asarray([0.0, 0.0]),
        initial_displacement_controls=jnp.zeros((2, 2, 2)),
        learning_rate_start=0.01,
        learning_rate_end=0.01,
        updates=1,
        validation_interval=1,
        checkpoint_interval=1,
        seed=3,
    )


def test_strict_preparation_rejects_legacy_model_without_support_contract():
    with pytest.raises(
        ValueError,
        match="strict material-scope preparation requires.*SupportContract",
    ):
        _prepare(_model(), strict=True)


def test_valid_contract_binds_problem_id_scope_and_modeled_role_order(
    strict_prepared,
    nuisance_contract,
):
    expected_roles = np.asarray(
        [LatticeSiteRole1D.TARGET, LatticeSiteRole1D.NUISANCE],
        dtype=np.int8,
    )
    modeled_roles = np.asarray(nuisance_contract.site_role_codes)[
        np.asarray(nuisance_contract.modeled_site_indices)
    ]

    np.testing.assert_array_equal(modeled_roles, expected_roles)
    assert strict_prepared.model.support_contract is nuisance_contract
    assert strict_prepared.metadata["support_contract_id"] == (
        nuisance_contract.contract_id
    )
    assert strict_prepared.metadata["material_scope_complete"] is True
    assert strict_prepared.metadata["material_scope_fully_parameterized"] is True
    assert strict_prepared.metadata["support_contract_required"] is True
    assert strict_prepared.metadata["n_target_sites"] == 1
    assert strict_prepared.metadata["n_nuisance_sites"] == 1
    assert strict_prepared.metadata["reconstruction_problem_id"] == (
        strict_prepared.reconstruction_problem_id
    )
    assert len(strict_prepared.reconstruction_problem_id) == 64


def test_known_fixed_versus_nuisance_contract_changes_problem_id(
    strict_prepared,
    nuisance_contract,
):
    fixed_contract = _support_contract(fixed_exterior=True)
    fixed_prepared = _prepare(_model(fixed_contract))

    assert fixed_contract.contract_id != nuisance_contract.contract_id
    assert fixed_prepared.reconstruction_problem_id != (
        strict_prepared.reconstruction_problem_id
    )
    assert fixed_prepared.metadata["n_target_sites"] == 1
    assert fixed_prepared.metadata["n_nuisance_sites"] == 0
    assert fixed_prepared.metadata["material_scope_complete"] is True
    assert fixed_prepared.metadata["material_scope_fully_parameterized"] is False
    assert fixed_prepared.metadata["fixed_material_provenance_verified"] is False


@pytest.mark.parametrize("mismatch", ["coordinate", "patch"])
def test_model_contract_coordinate_or_patch_mismatch_rejects(
    nuisance_contract,
    mismatch,
):
    model = _model(nuisance_contract)
    if mismatch == "coordinate":
        model = replace(
            model,
            site_coordinates=model.site_coordinates.at[0, 0].add(0.01),
        )
        message = "site_coordinates do not match"
    else:
        model = replace(
            model,
            patch_starts=model.patch_starts.at[0, 0].add(1),
        )
        message = "patch_starts do not match"

    with pytest.raises(ValueError, match=message):
        _prepare(model)


def test_tiny_run_carries_roles_and_target_nuisance_masks(
    strict_result,
    strict_prepared,
):
    expected_roles = np.asarray(
        [LatticeSiteRole1D.TARGET, LatticeSiteRole1D.NUISANCE],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(strict_result.site_role_codes, expected_roles)
    np.testing.assert_array_equal(strict_result.target_site_mask, [True, False])
    np.testing.assert_array_equal(strict_result.nuisance_site_mask, [False, True])
    assert strict_result.support_contract_id == (
        strict_prepared.model.support_contract.contract_id
    )
    assert strict_result.material_scope_complete is True
    assert strict_result.material_scope_fully_parameterized is True
    assert strict_result.metadata["material_scope_fully_parameterized"] is True
    assert strict_result.metadata["structural_reporting_scope"] == (
        "target_sites_only"
    )
    assert strict_result.metadata["n_target_vacancy_parameters"] == 1
    assert strict_result.metadata["n_nuisance_vacancy_parameters"] == 1


def _rewrite_npz(source, destination, mutation):
    with np.load(source, allow_pickle=False) as archive:
        payload = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    mutation(payload)
    np.savez(destination, **payload)


def test_save_load_preserves_support_fields_and_rejects_tampering(
    tmp_path,
    strict_result,
):
    path = tmp_path / "support_result.npz"
    save_lattice_site_reconstruction_1d(path, strict_result)

    loaded = load_lattice_site_reconstruction_1d(path)
    np.testing.assert_array_equal(
        loaded.site_role_codes, strict_result.site_role_codes
    )
    np.testing.assert_array_equal(loaded.target_site_mask, [True, False])
    np.testing.assert_array_equal(loaded.nuisance_site_mask, [False, True])
    assert loaded.support_contract_id == strict_result.support_contract_id
    assert loaded.material_scope_complete is True
    assert loaded.material_scope_fully_parameterized is True
    assert loaded.metadata["material_scope_fully_parameterized"] is True

    role_path = tmp_path / "tampered_role.npz"

    def swap_roles(payload):
        payload["site_role_codes"] = payload["site_role_codes"][::-1].copy()

    _rewrite_npz(path, role_path, swap_roles)
    with pytest.raises(ValueError, match="support_evidence_id does not match"):
        load_lattice_site_reconstruction_1d(role_path)

    evidence_path = tmp_path / "tampered_evidence.npz"

    def replace_evidence(payload):
        payload["support_evidence_id"] = np.asarray("0" * 64)

    _rewrite_npz(path, evidence_path, replace_evidence)
    with pytest.raises(ValueError, match="support_evidence_id does not match"):
        load_lattice_site_reconstruction_1d(evidence_path)


def test_scope_field_or_metadata_tampering_is_rejected(
    tmp_path,
    strict_result,
):
    path = tmp_path / "authenticated_scope_result.npz"
    save_lattice_site_reconstruction_1d(path, strict_result)

    field_path = tmp_path / "tampered_typed_scope.npz"

    def clear_typed_scope(payload):
        payload["material_scope_fully_parameterized"] = np.asarray(False)

    _rewrite_npz(path, field_path, clear_typed_scope)
    with pytest.raises(ValueError, match="support_evidence_id does not match"):
        load_lattice_site_reconstruction_1d(field_path)

    metadata_path = tmp_path / "tampered_scope_metadata.npz"

    def clear_metadata_scope(payload):
        metadata = json.loads(str(payload["metadata_json"].item()))
        metadata["material_scope_fully_parameterized"] = False
        payload["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True)
        )

    _rewrite_npz(path, metadata_path, clear_metadata_scope)
    with pytest.raises(
        ValueError,
        match="metadata material_scope_fully_parameterized.*disagrees",
    ):
        load_lattice_site_reconstruction_1d(metadata_path)


def test_legacy_v1_unbound_scope_metadata_cannot_gain_trust(
    tmp_path,
    strict_result,
):
    current_path = tmp_path / "current_scope_result.npz"
    legacy_path = tmp_path / "legacy_v1_scope_result.npz"
    save_lattice_site_reconstruction_1d(current_path, strict_result)

    def downgrade_to_v1_with_unbound_positive_metadata(payload):
        payload.pop("material_scope_fully_parameterized")
        metadata = json.loads(str(payload["metadata_json"].item()))
        metadata["material_scope_fully_parameterized"] = True
        payload["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True)
        )
        payload["support_evidence_id"] = np.asarray(
            ptychography_1d_module._lattice_result_support_evidence_id_1d(
                payload["site_coordinates"],
                payload["site_role_codes"],
                str(payload["support_contract_id"].item()),
                bool(payload["material_scope_complete"].item()),
            )
        )

    _rewrite_npz(
        current_path,
        legacy_path,
        downgrade_to_v1_with_unbound_positive_metadata,
    )
    loaded = load_lattice_site_reconstruction_1d(legacy_path)

    assert loaded.material_scope_complete is True
    assert loaded.material_scope_fully_parameterized is False
    assert loaded.metadata["material_scope_fully_parameterized"] is False
    assert loaded.metadata["legacy_material_scope_metadata_was_unbound"] is True

    ensemble = summarize_lattice_site_ensemble_1d(
        [loaded],
        options=MultistartOptions1D(
            n_starts=1,
            minimum_accepted_starts=1,
            minimum_accepted_fraction=1.0,
        ),
    )
    assert ensemble.trust_flags["material_scope_complete"] is False
    assert ensemble.structurally_trusted is False
