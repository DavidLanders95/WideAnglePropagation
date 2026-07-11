"""Focused tests for fail-closed stochastic Fisher screening."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.stats import chi2

from wide_angle_propagation.ptychography_stochastic_observability_1d import (
    StochasticFisherScreeningOptions1D,
    StochasticPhysicalBlock1D,
    estimate_stochastic_fisher_screening_1d,
)


def _diagonal_screen(seed=13):
    fisher_diagonal = np.asarray([4.0, 1.0, 0.0])
    detector_factor = np.asarray([2.0, 1.0, 0.0])
    physical = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, -0.5, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    options = StochasticFisherScreeningOptions1D(
        covariance_probe_count=256,
        null_probe_count=256,
        random_seed=seed,
        simultaneous_confidence=0.95,
        maximum_iterations=8,
        relative_residual_tolerance=1e-12,
        maximum_pcg_solves=512,
        maximum_total_pcg_iterations=4096,
        maximum_fisher_matvec_calls=16_384,
    )
    result = estimate_stochastic_fisher_screening_1d(
        parameter_count=3,
        detector_probe_shape=(3,),
        detector_vjp=lambda probe: detector_factor * probe,
        fisher_matvec=lambda direction: fisher_diagonal * direction,
        physical_jvp=lambda direction: physical @ direction,
        physical_covariance_blocks=(
            StochasticPhysicalBlock1D(
                name="mixed",
                row_indices=(0, 2),
                column_indices=(1, 3),
            ),
        ),
        options=options,
    )
    return result, physical


def test_diagonal_fisher_recovers_covariance_and_null_leakage_with_bounds():
    result, physical = _diagonal_screen()
    expected_covariance = physical @ np.diag([0.25, 1.0, 0.0]) @ physical.T
    expected_leakage = np.diag(
        physical @ np.diag([0.0, 0.0, 1.0]) @ physical.T
    )

    assert result.numerically_valid
    assert result.operator_checks.passed
    assert result.factor_covariance_verified is False
    assert result.structurally_trusted is False
    assert result.suitable_for_trust_gate is False
    with pytest.raises(FrozenInstanceError):
        result.random_seed = 2

    np.testing.assert_allclose(
        result.physical_marginal_variance,
        np.diag(expected_covariance),
        rtol=0.18,
        atol=0.03,
    )
    np.testing.assert_array_less(
        result.physical_marginal_variance_lower - 1e-15,
        np.diag(expected_covariance) + 1e-15,
    )
    np.testing.assert_array_less(
        np.diag(expected_covariance) - 1e-15,
        result.physical_marginal_variance_upper + 1e-15,
    )
    np.testing.assert_allclose(
        result.physical_null_leakage,
        expected_leakage,
        rtol=0.18,
        atol=0.03,
    )
    np.testing.assert_array_less(
        result.physical_null_leakage_lower - 1e-15,
        expected_leakage + 1e-15,
    )
    np.testing.assert_array_less(
        expected_leakage - 1e-15,
        result.physical_null_leakage_upper + 1e-15,
    )
    assert result.global_nullity_estimate == pytest.approx(1.0, rel=0.18)
    assert result.global_nullity_rank_interval_valid
    assert 1 in result.global_nullity_rank_confidence_set

    block = result.covariance_blocks[0]
    np.testing.assert_array_equal(block.row_indices, [0, 2])
    np.testing.assert_array_equal(block.column_indices, [1, 3])
    np.testing.assert_allclose(
        block.covariance,
        expected_covariance[np.ix_([0, 2], [1, 3])],
        rtol=0.2,
        atol=0.06,
    )
    assert np.all(np.isfinite(block.monte_carlo_standard_error))
    assert not result.physical_marginal_variance.flags.writeable


def test_exact_chi_square_formula_deterministic_streams_and_pcg_accounting():
    first, _ = _diagonal_screen(seed=29)
    second, _ = _diagonal_screen(seed=29)
    np.testing.assert_array_equal(
        first.physical_marginal_variance,
        second.physical_marginal_variance,
    )
    np.testing.assert_array_equal(
        first.physical_null_leakage,
        second.physical_null_leakage,
    )
    assert first.covariance_stream_seed == second.covariance_stream_seed
    assert first.null_stream_seed == second.null_stream_seed

    degrees = first.covariance_probe_count
    summed_squares = first.physical_marginal_variance * degrees
    expected_lower = summed_squares / chi2.ppf(
        1.0 - first.per_quantity_error_probability / 2.0,
        degrees,
    )
    expected_upper = summed_squares / chi2.ppf(
        first.per_quantity_error_probability / 2.0,
        degrees,
    )
    np.testing.assert_allclose(
        first.physical_marginal_variance_lower, expected_lower
    )
    np.testing.assert_allclose(
        first.physical_marginal_variance_upper, expected_upper
    )
    assert len(first.covariance_solver_diagnostics) == 256
    assert len(first.null_solver_diagnostics) == 256
    assert first.budget.attempted_pcg_solves == 512
    assert first.budget.converged_pcg_solves == 512
    assert first.budget.all_budget_checks_passed
    assert first.budget.actual_pcg_iterations == sum(
        diagnostic.iterations
        for diagnostic in (
            first.covariance_solver_diagnostics
            + first.null_solver_diagnostics
        )
    )


def test_unsuccessful_pcg_censors_every_numerical_claim():
    fisher = np.asarray([[1.0, 0.0], [0.0, 4.0]])
    factor = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    result = estimate_stochastic_fisher_screening_1d(
        parameter_count=2,
        detector_probe_shape=2,
        detector_vjp=lambda probe: factor @ probe,
        fisher_matvec=lambda direction: fisher @ direction,
        physical_jvp=lambda direction: direction,
        physical_covariance_blocks=(
            StochasticPhysicalBlock1D("all", (0, 1), (0, 1)),
        ),
        options=StochasticFisherScreeningOptions1D(
            covariance_probe_count=2,
            null_probe_count=2,
            maximum_iterations=1,
            relative_residual_tolerance=1e-14,
            maximum_pcg_solves=4,
            maximum_total_pcg_iterations=4,
            maximum_fisher_matvec_calls=32,
        ),
    )

    assert not result.numerically_valid
    assert result.budget.converged_pcg_solves < 4
    assert len(result.covariance_solver_diagnostics) == 2
    assert len(result.null_solver_diagnostics) == 2
    assert np.all(np.isnan(result.physical_marginal_variance))
    assert np.all(result.physical_marginal_variance_lower == 0.0)
    assert np.all(np.isinf(result.physical_marginal_variance_upper))
    assert np.all(np.isnan(result.physical_null_leakage))
    assert np.all(np.isinf(result.physical_null_leakage_mcse))
    assert np.all(np.isnan(result.covariance_blocks[0].covariance))
    assert result.global_nullity_rank_confidence_set == (0, 1, 2)
    assert not result.global_nullity_rank_interval_valid
    assert result.structurally_trusted is False


def test_operator_contract_and_hard_solve_budget_fail_before_sampling():
    with pytest.raises(ValueError, match="contract checks failed"):
        estimate_stochastic_fisher_screening_1d(
            parameter_count=2,
            detector_probe_shape=2,
            detector_vjp=lambda probe: probe,
            fisher_matvec=lambda direction: np.asarray(
                [[1.0, 1.0], [0.0, 1.0]]
            )
            @ direction,
            physical_jvp=lambda direction: direction,
            options=StochasticFisherScreeningOptions1D(
                covariance_probe_count=2,
                null_probe_count=2,
                maximum_pcg_solves=4,
                maximum_total_pcg_iterations=1024,
                maximum_fisher_matvec_calls=4096,
            ),
        )

    calls = {"fisher": 0}

    def fisher(direction):
        calls["fisher"] += 1
        return direction

    with pytest.raises(ValueError, match="maximum_pcg_solves"):
        estimate_stochastic_fisher_screening_1d(
            parameter_count=2,
            detector_probe_shape=2,
            detector_vjp=lambda probe: probe,
            fisher_matvec=fisher,
            physical_jvp=lambda direction: direction,
            options=StochasticFisherScreeningOptions1D(
                covariance_probe_count=2,
                null_probe_count=2,
                maximum_pcg_solves=3,
                maximum_total_pcg_iterations=1024,
                maximum_fisher_matvec_calls=4096,
            ),
        )
    assert calls["fisher"] == 0

    with pytest.raises(ValueError, match="identity preconditioner"):
        estimate_stochastic_fisher_screening_1d(
            parameter_count=2,
            detector_probe_shape=2,
            detector_vjp=lambda probe: probe,
            fisher_matvec=lambda direction: direction,
            physical_jvp=lambda direction: direction,
            preconditioner_diagonal=np.asarray([1.0, 2.0]),
            options=StochasticFisherScreeningOptions1D(
                covariance_probe_count=2,
                null_probe_count=2,
                maximum_pcg_solves=4,
                maximum_total_pcg_iterations=1024,
                maximum_fisher_matvec_calls=4096,
            ),
        )
