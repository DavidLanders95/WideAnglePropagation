"""Focused role-boundary tests for lattice-site ensemble summaries."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest


pytest.importorskip("jax")

from wide_angle_propagation.ptychography_1d import (
    LatticeSiteReconstruction1D,
)
from wide_angle_propagation.ptychography_ensemble_1d import (
    MultistartOptions1D,
    summarize_lattice_site_ensemble_1d,
)
from wide_angle_propagation.ptychography_support_contract_1d import (
    LatticeSiteRole1D,
)


_SUPPORT_ID = "a" * 64
_OTHER_SUPPORT_ID = "b" * 64
_ROLES = np.asarray(
    [LatticeSiteRole1D.TARGET, LatticeSiteRole1D.NUISANCE],
    dtype=np.int8,
)


def _options() -> MultistartOptions1D:
    return MultistartOptions1D(
        n_starts=3,
        minimum_accepted_starts=3,
        minimum_accepted_fraction=1.0,
        relative_loss_tolerance=0.0,
        absolute_loss_tolerance=0.0,
    )


def _result(
    target_vacancy: float,
    nuisance_vacancy: float,
    *,
    target_residual: tuple[float, float] = (0.0, 0.0),
    nuisance_residual: tuple[float, float] = (0.0, 0.0),
    roles: np.ndarray | None = _ROLES,
    support_id: str | None = _SUPPORT_ID,
    material_scope_complete: bool = True,
    material_scope_fully_parameterized: bool | None = None,
) -> LatticeSiteReconstruction1D:
    sites = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    residual = np.asarray([target_residual, nuisance_residual], dtype=float)
    vacancies = np.asarray([target_vacancy, nuisance_vacancy], dtype=float)
    controls = np.zeros((2, 2, 2), dtype=float)
    role_codes = (
        np.empty(0, dtype=np.int8)
        if roles is None
        else np.asarray(roles, dtype=np.int8)
    )
    fully_parameterized = (
        bool(material_scope_complete)
        if material_scope_fully_parameterized is None
        else bool(material_scope_fully_parameterized)
    )
    return LatticeSiteReconstruction1D(
        potential=np.zeros((2, 2)),
        initial_potential=np.zeros((2, 2)),
        vacancy_fractions=vacancies,
        initial_vacancy_fractions=np.zeros_like(vacancies),
        displacement_controls=controls,
        initial_displacement_controls=controls,
        site_coordinates=sites,
        displaced_site_coordinates=sites + residual,
        control_coordinates_s=np.asarray([0.0, 1.0]),
        control_coordinates_u=np.asarray([0.0, 1.0]),
        predicted_intensities=np.zeros((3, 2)),
        measured_intensities=np.zeros((3, 2)),
        window_starts=np.zeros(3, dtype=np.int32),
        scan_coordinates=np.arange(3, dtype=float),
        detector_angles=np.asarray([0.0, 1.0]),
        update_history=np.asarray([0], dtype=np.int32),
        elapsed_time_history=np.asarray([0.0]),
        training_loss_history=np.asarray([1.0]),
        validation_loss_history=np.asarray([1.0]),
        best_update=0,
        completed_updates=1,
        converged=True,
        stop_reason="plateau",
        rigid_displacement=np.zeros(2),
        site_role_codes=role_codes,
        support_contract_id=support_id,
        material_scope_complete=material_scope_complete,
        material_scope_fully_parameterized=fully_parameterized,
        metadata={
            "best_metric": 1.0,
            "best_total_displacement_bound_fraction": 0.0,
            "material_scope_fully_parameterized": fully_parameterized,
        },
    )


def _role_aware_starts(
    nuisance_vacancies: tuple[float, float, float],
    nuisance_residuals: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ],
) -> list[LatticeSiteReconstruction1D]:
    target_vacancies = (0.10, 0.12, 0.14)
    return [
        _result(
            target_vacancy,
            nuisance_vacancy,
            nuisance_residual=nuisance_residual,
        )
        for target_vacancy, nuisance_vacancy, nuisance_residual in zip(
            target_vacancies,
            nuisance_vacancies,
            nuisance_residuals,
            strict=True,
        )
    ]


def test_nuisance_sites_are_excluded_from_consensus_and_trust() -> None:
    starts = _role_aware_starts(
        (0.0, 0.5, 1.0),
        ((-20.0, 15.0), (0.0, 0.0), (30.0, -25.0)),
    )

    ensemble = summarize_lattice_site_ensemble_1d(starts, options=_options())
    consensus = ensemble.consensus

    assert consensus.vacancy_state[0] == 0
    assert consensus.optimizer_agreement[0]
    assert np.isfinite(consensus.vacancy_median[0])
    assert ensemble.trust_flags["vacancy_consensus"] is True
    assert ensemble.trust_flags["residual_strain_consensus"] is True

    nuisance_index = 1
    for values in (
        consensus.vacancy_median,
        consensus.vacancy_q05,
        consensus.vacancy_q95,
        consensus.vacancy_call_frequency,
        consensus.residual_displacement_radial_q90_A,
    ):
        assert np.isnan(values[nuisance_index])
    for values in (
        consensus.residual_displacement_median,
        consensus.residual_displacement_q05,
        consensus.residual_displacement_q95,
    ):
        assert np.isnan(values[nuisance_index]).all()
    assert consensus.vacancy_state[nuisance_index] == -1
    for values in (
        consensus.optimizer_agreement,
        consensus.sensitive,
        consensus.observable,
        consensus.site_trusted,
    ):
        assert not values[nuisance_index]


def test_nuisance_only_changes_do_not_change_target_medoid_or_basin() -> None:
    reference = _role_aware_starts(
        (0.2, 0.2, 0.2),
        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    )
    nuisance_perturbed = _role_aware_starts(
        (1.0, 0.0, 1.0),
        ((100.0, -100.0), (-75.0, 60.0), (250.0, 300.0)),
    )

    first = summarize_lattice_site_ensemble_1d(reference, options=_options())
    second = summarize_lattice_site_ensemble_1d(
        nuisance_perturbed,
        options=_options(),
    )

    assert first.representative_index == second.representative_index == 1
    assert first.trust_flags["dominant_low_loss_basin"] is True
    assert second.trust_flags["dominant_low_loss_basin"] is True
    assert first.optimizer_stable is second.optimizer_stable is True
    for name in (
        "vacancy_median",
        "vacancy_q05",
        "vacancy_q95",
        "vacancy_call_frequency",
        "vacancy_state",
        "optimizer_agreement",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(first.consensus, name))[0],
            np.asarray(getattr(second.consensus, name))[0],
        )


def test_ensemble_rejects_role_or_support_contract_mismatch() -> None:
    starts = _role_aware_starts(
        (0.0, 0.0, 0.0),
        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    )

    mismatched_roles = list(starts)
    mismatched_roles[1] = replace(
        mismatched_roles[1],
        site_role_codes=np.asarray(
            [LatticeSiteRole1D.NUISANCE, LatticeSiteRole1D.TARGET],
            dtype=np.int8,
        ),
    )
    with pytest.raises(ValueError, match="identical ordered site roles"):
        summarize_lattice_site_ensemble_1d(
            mismatched_roles,
            options=_options(),
        )

    mismatched_support = list(starts)
    mismatched_support[1] = replace(
        mismatched_support[1],
        support_contract_id=_OTHER_SUPPORT_ID,
    )
    with pytest.raises(ValueError, match="same support contract"):
        summarize_lattice_site_ensemble_1d(
            mismatched_support,
            options=_options(),
        )


def test_legacy_roleless_results_fail_closed_on_material_scope() -> None:
    legacy = [
        _result(
            target_vacancy,
            0.0,
            roles=None,
            support_id=None,
            material_scope_complete=False,
        )
        for target_vacancy in (0.10, 0.12, 0.14)
    ]

    ensemble = summarize_lattice_site_ensemble_1d(legacy, options=_options())

    assert ensemble.trust_flags["material_scope_complete"] is False
    assert ensemble.structurally_trusted is False
    assert not np.any(ensemble.consensus.site_trusted)

    unsupported_completeness_claim = [
        replace(
            result,
            support_contract_id=_SUPPORT_ID,
            material_scope_complete=True,
        )
        for result in legacy
    ]
    with pytest.raises(
        ValueError,
        match="support contract|material-scope completeness",
    ):
        summarize_lattice_site_ensemble_1d(
            unsupported_completeness_claim,
            options=_options(),
        )
