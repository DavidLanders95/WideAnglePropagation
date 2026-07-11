"""Fail-closed stochastic Fisher screening from matrix-free linear callbacks.

This module is intentionally independent of the prepared ptychography API.  It
implements two Gaussian-probe identities for a positive-semidefinite Fisher
operator ``F`` and a physical-output Jacobian ``B``:

``x = F^+ J^T P xi`` gives ``E[(B x) (B x)^T] = B F^+ B^T``, while
``n = w - F^+ F w`` gives ``E[(B n)**2]`` as physical-output leakage into the
parameter null space.  The caller supplies the factor action ``J^T P xi`` as
``detector_vjp``; no detector Jacobian is materialized.

The report is screening evidence only.  Its result type exposes immutable
``False`` properties for structural trust, irrespective of numerical success.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any, Callable, Sequence

import numpy as np

from .ptychography_observability_1d import (
    PCGSolveDiagnostics1D,
    pcg_solve_observability_1d,
)


__all__ = [
    "StochasticFisherBudget1D",
    "StochasticFisherOperatorChecks1D",
    "StochasticFisherScreeningOptions1D",
    "StochasticFisherScreeningResult1D",
    "StochasticPhysicalBlock1D",
    "StochasticPhysicalBlockEstimate1D",
    "estimate_stochastic_fisher_screening_1d",
]


Array = Any


@dataclass(frozen=True)
class StochasticPhysicalBlock1D:
    """A requested cross-covariance block of physical-output indices."""

    name: str
    row_indices: Sequence[int]
    column_indices: Sequence[int]


@dataclass(frozen=True)
class StochasticPhysicalBlockEstimate1D:
    """Monte Carlo mean and empirical MCSE for one covariance block."""

    name: str
    row_indices: Array
    column_indices: Array
    covariance: Array
    monte_carlo_standard_error: Array


@dataclass(frozen=True)
class StochasticFisherScreeningOptions1D:
    """Sampling, PCG, operator-check, and hard resource budgets."""

    covariance_probe_count: int = 64
    null_probe_count: int = 64
    random_seed: int = 0
    simultaneous_confidence: float = 0.95
    maximum_iterations: int = 256
    relative_residual_tolerance: float = 1e-7
    absolute_residual_tolerance: float = 0.0
    stagnation_iterations: int = 12
    stagnation_relative_improvement: float = 1e-3
    curvature_tolerance: float = 1e-12
    operator_check_vectors: int = 3
    symmetry_tolerance: float = 1e-8
    psd_tolerance: float = 1e-10
    linearity_tolerance: float = 1e-8
    zero_tolerance: float = 1e-10
    maximum_pcg_solves: int = 256
    maximum_total_pcg_iterations: int = 65_536
    maximum_fisher_matvec_calls: int = 131_072


@dataclass(frozen=True)
class StochasticFisherOperatorChecks1D:
    """Deterministic checks required before stochastic estimates are used."""

    zero_operator_norm: float
    maximum_symmetry_error: float
    minimum_normalized_curvature: float
    maximum_fisher_linearity_error: float
    detector_vjp_linearity_error: float
    physical_jvp_zero_norm: float
    physical_jvp_linearity_error: float
    passed: bool


@dataclass(frozen=True)
class StochasticFisherBudget1D:
    """Configured worst cases and observed solver work."""

    requested_pcg_solves: int
    maximum_pcg_solves: int
    configured_worst_case_pcg_iterations: int
    maximum_total_pcg_iterations: int
    configured_worst_case_fisher_matvec_calls: int
    maximum_fisher_matvec_calls: int
    attempted_pcg_solves: int
    converged_pcg_solves: int
    actual_pcg_iterations: int
    actual_fisher_matvec_calls: int
    detector_vjp_calls: int
    physical_jvp_calls: int
    all_budget_checks_passed: bool


@dataclass(frozen=True)
class StochasticFisherScreeningResult1D:
    """Stochastic numerical screen which is never structural-trust evidence."""

    physical_marginal_variance: Array
    physical_marginal_variance_lower: Array
    physical_marginal_variance_upper: Array
    physical_marginal_variance_mcse: Array
    covariance_blocks: tuple[StochasticPhysicalBlockEstimate1D, ...]
    physical_null_leakage: Array
    physical_null_leakage_lower: Array
    physical_null_leakage_upper: Array
    physical_null_leakage_mcse: Array
    global_nullity_estimate: float
    global_nullity_mcse: float
    global_nullity_rank_lower: int
    global_nullity_rank_upper: int
    global_nullity_rank_confidence_set: tuple[int, ...]
    global_nullity_rank_interval_valid: bool
    covariance_solver_diagnostics: tuple[PCGSolveDiagnostics1D, ...]
    null_solver_diagnostics: tuple[PCGSolveDiagnostics1D, ...]
    operator_checks: StochasticFisherOperatorChecks1D
    budget: StochasticFisherBudget1D
    covariance_probe_count: int
    null_probe_count: int
    simultaneous_confidence: float
    per_quantity_error_probability: float
    random_seed: int
    check_stream_seed: int
    covariance_stream_seed: int
    null_stream_seed: int
    numerically_valid: bool
    factor_covariance_verified: bool
    bound_method: str

    @property
    def structurally_trusted(self) -> bool:
        """Stochastic screening is never sufficient for structural trust."""
        return False

    @property
    def suitable_for_trust_gate(self) -> bool:
        """Return ``False`` by construction, including for successful runs."""
        return False


def _positive_index(value: Any, name: str) -> int:
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _validated_options(
    options: StochasticFisherScreeningOptions1D | None,
) -> StochasticFisherScreeningOptions1D:
    options = StochasticFisherScreeningOptions1D() if options is None else options
    if not isinstance(options, StochasticFisherScreeningOptions1D):
        raise TypeError(
            "options must be a StochasticFisherScreeningOptions1D instance or None"
        )
    for name in (
        "covariance_probe_count",
        "null_probe_count",
        "maximum_iterations",
        "stagnation_iterations",
        "operator_check_vectors",
        "maximum_pcg_solves",
        "maximum_total_pcg_iterations",
        "maximum_fisher_matvec_calls",
    ):
        _positive_index(getattr(options, name), f"options.{name}")
    for name in ("covariance_probe_count", "null_probe_count"):
        if operator.index(getattr(options, name)) < 2:
            raise ValueError(f"options.{name} must be at least two")
    try:
        seed = operator.index(options.random_seed)
    except TypeError as error:
        raise TypeError("options.random_seed must be an integer") from error
    if seed < 0 or seed >= 2**64:
        raise ValueError("options.random_seed must lie in [0, 2**64)")
    confidence = float(options.simultaneous_confidence)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("options.simultaneous_confidence must lie in (0, 1)")
    positive = (
        "relative_residual_tolerance",
        "curvature_tolerance",
        "symmetry_tolerance",
        "psd_tolerance",
        "linearity_tolerance",
        "zero_tolerance",
    )
    for name in positive:
        value = float(getattr(options, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"options.{name} must be finite and positive")
    absolute = float(options.absolute_residual_tolerance)
    if not np.isfinite(absolute) or absolute < 0.0:
        raise ValueError(
            "options.absolute_residual_tolerance must be finite and non-negative"
        )
    improvement = float(options.stagnation_relative_improvement)
    if not np.isfinite(improvement) or not 0.0 < improvement < 1.0:
        raise ValueError(
            "options.stagnation_relative_improvement must lie strictly in (0, 1)"
        )
    return options


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _floating_vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != (size,) or not np.issubdtype(result.dtype, np.floating):
        raise TypeError(f"{name} must return a floating vector with shape ({size},)")
    result = np.asarray(result, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} returned non-finite values")
    return result


def _relative_error(residual: Any, scale: float) -> float:
    return float(np.linalg.norm(np.asarray(residual))) / max(float(scale), 1.0)


def _validate_blocks(
    blocks: Sequence[StochasticPhysicalBlock1D],
    n_output: int,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    validated = []
    names: set[str] = set()
    for block in tuple(blocks):
        if not isinstance(block, StochasticPhysicalBlock1D):
            raise TypeError(
                "physical_covariance_blocks must contain "
                "StochasticPhysicalBlock1D instances"
            )
        if not isinstance(block.name, str) or not block.name:
            raise ValueError("physical covariance block names must be non-empty")
        if block.name in names:
            raise ValueError("physical covariance block names must be unique")
        names.add(block.name)
        rows = np.asarray(block.row_indices)
        columns = np.asarray(block.column_indices)
        for indices, role in ((rows, "row"), (columns, "column")):
            if indices.ndim != 1 or indices.size == 0:
                raise ValueError(f"block {block.name!r} {role} indices must be 1D")
            if not np.issubdtype(indices.dtype, np.integer):
                raise TypeError(f"block {block.name!r} {role} indices must be integers")
            if np.any(indices < 0) or np.any(indices >= n_output):
                raise ValueError(f"block {block.name!r} {role} index is out of range")
            if np.unique(indices).size != indices.size:
                raise ValueError(f"block {block.name!r} {role} indices repeat")
        validated.append(
            (
                block.name,
                np.asarray(rows, dtype=np.int64),
                np.asarray(columns, dtype=np.int64),
            )
        )
    return tuple(validated)


def _chi_square_variance_bounds(
    samples: np.ndarray,
    error_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.stats import chi2

    n_probe = samples.shape[0]
    squares = np.square(samples)
    sums = np.sum(squares, axis=0)
    estimates = sums / n_probe
    mcse = np.std(squares, axis=0, ddof=1) / np.sqrt(n_probe)
    lower_quantile = float(chi2.ppf(error_probability / 2.0, n_probe))
    upper_quantile = float(chi2.ppf(1.0 - error_probability / 2.0, n_probe))
    lower = sums / upper_quantile
    upper = sums / lower_quantile
    return estimates, lower, upper, mcse


def _solver_accepted(
    diagnostic: PCGSolveDiagnostics1D,
    relative_tolerance: float,
) -> bool:
    return bool(
        diagnostic.converged
        and not diagnostic.breakdown
        and not diagnostic.stagnated
        and np.isfinite(diagnostic.residual_norm)
        and np.isfinite(diagnostic.relative_residual)
        and diagnostic.relative_residual <= relative_tolerance
    )


def estimate_stochastic_fisher_screening_1d(
    *,
    parameter_count: int,
    detector_probe_shape: Sequence[int] | int,
    detector_vjp: Callable[[Array], Array],
    fisher_matvec: Callable[[Array], Array],
    physical_jvp: Callable[[Array], Array],
    physical_covariance_blocks: Sequence[StochasticPhysicalBlock1D] = (),
    preconditioner_diagonal: Array | None = None,
    options: StochasticFisherScreeningOptions1D | None = None,
) -> StochasticFisherScreeningResult1D:
    """Estimate physical covariance and null leakage with Gaussian probes.

    ``detector_vjp(xi)`` must have covariance ``F`` for a standard-normal
    detector probe ``xi``.  For a projected detector Jacobian this callback is
    ``J.T @ P @ xi``.  ``physical_jvp(direction)`` applies the physical-output
    Jacobian ``B``.  The stochastic contract is checked for shape, finiteness,
    and linearity, but its covariance identity remains a caller responsibility;
    consequently this function can only produce screening evidence.

    Conditional chi-square marginal intervals are Bonferroni-adjusted for
    simultaneous coverage. Their statistical interpretation assumes that
    ``detector_vjp(xi)`` has covariance ``F`` and that accepted PCG solves are
    accurate enough to act linearly. This generic callback API checks neither
    assertion exactly, so ``factor_covariance_verified`` is always false and
    the result remains screening-only. Any unsuccessful solve censors every
    numerical estimate to ``NaN`` and ``[0, inf]``.
    """
    options = _validated_options(options)
    n_parameter = _positive_index(parameter_count, "parameter_count")
    if isinstance(detector_probe_shape, (int, np.integer)):
        detector_shape = (_positive_index(detector_probe_shape, "detector_probe_shape"),)
    else:
        try:
            detector_shape = tuple(
                _positive_index(value, "detector_probe_shape entry")
                for value in detector_probe_shape
            )
        except TypeError as error:
            raise TypeError(
                "detector_probe_shape must be an integer or a sequence of integers"
            ) from error
        if not detector_shape:
            raise ValueError("detector_probe_shape cannot be empty")
    for callback, name in (
        (detector_vjp, "detector_vjp"),
        (fisher_matvec, "fisher_matvec"),
        (physical_jvp, "physical_jvp"),
    ):
        if not callable(callback):
            raise TypeError(f"{name} must be callable")

    preconditioner = np.ones(n_parameter, dtype=float)
    if preconditioner_diagonal is not None:
        preconditioner = np.asarray(preconditioner_diagonal)
        if (
            preconditioner.shape != (n_parameter,)
            or not np.issubdtype(preconditioner.dtype, np.floating)
        ):
            raise TypeError(
                "preconditioner_diagonal must be a floating parameter vector"
            )
        preconditioner = np.asarray(preconditioner, dtype=float)
        if np.any(~np.isfinite(preconditioner)) or np.any(preconditioner <= 0.0):
            raise ValueError(
                "preconditioner_diagonal must contain finite positive values"
            )
        if not np.array_equal(preconditioner, np.ones_like(preconditioner)):
            raise ValueError(
                "stochastic null projection requires the identity preconditioner"
            )

    n_covariance = operator.index(options.covariance_probe_count)
    n_null = operator.index(options.null_probe_count)
    n_solves = n_covariance + n_null
    n_checks = operator.index(options.operator_check_vectors)
    max_iterations = operator.index(options.maximum_iterations)
    worst_iterations = n_solves * max_iterations
    operator_matvec_calls = 1 + 3 * n_checks
    worst_matvec_calls = (
        operator_matvec_calls
        + n_null
        + n_solves * (1 + 2 * max_iterations)
    )
    if n_solves > operator.index(options.maximum_pcg_solves):
        raise ValueError("requested Gaussian probes exceed maximum_pcg_solves")
    if worst_iterations > operator.index(options.maximum_total_pcg_iterations):
        raise ValueError(
            "configured worst-case PCG iterations exceed the hard iteration budget"
        )
    if worst_matvec_calls > operator.index(options.maximum_fisher_matvec_calls):
        raise ValueError(
            "configured worst-case Fisher calls exceed the hard matvec budget"
        )

    counts = {"fisher": 0, "detector": 0, "physical": 0}

    def apply_fisher(direction: Any) -> np.ndarray:
        counts["fisher"] += 1
        if counts["fisher"] > operator.index(options.maximum_fisher_matvec_calls):
            raise RuntimeError("Fisher matvec budget was exceeded")
        vector = _floating_vector(direction, n_parameter, "Fisher direction")
        return _floating_vector(
            fisher_matvec(vector), n_parameter, "fisher_matvec"
        )

    def apply_detector(probe: Any) -> np.ndarray:
        counts["detector"] += 1
        probe_array = np.asarray(probe)
        if (
            probe_array.shape != detector_shape
            or not np.issubdtype(probe_array.dtype, np.floating)
            or not np.all(np.isfinite(probe_array))
        ):
            raise TypeError(
                "detector probes must be finite floating arrays with detector_probe_shape"
            )
        return _floating_vector(
            detector_vjp(np.asarray(probe_array, dtype=float)),
            n_parameter,
            "detector_vjp",
        )

    zero_parameter = np.zeros(n_parameter, dtype=float)
    zero_fisher = apply_fisher(zero_parameter)
    counts["physical"] += 1
    physical_zero = np.asarray(physical_jvp(zero_parameter))
    if physical_zero.ndim != 1 or not np.issubdtype(
        physical_zero.dtype, np.floating
    ):
        raise TypeError("physical_jvp must return a one-dimensional floating array")
    physical_zero = np.asarray(physical_zero, dtype=float)
    if physical_zero.size < 1 or not np.all(np.isfinite(physical_zero)):
        raise ValueError("physical_jvp must return a finite non-empty vector")
    n_output = physical_zero.size

    def apply_physical(direction: Any) -> np.ndarray:
        counts["physical"] += 1
        vector = _floating_vector(direction, n_parameter, "physical direction")
        return _floating_vector(
            physical_jvp(vector), n_output, "physical_jvp"
        )

    blocks = _validate_blocks(physical_covariance_blocks, n_output)
    seed_sequence = np.random.SeedSequence(operator.index(options.random_seed))
    check_seed, covariance_seed, null_seed = seed_sequence.spawn(3)
    seed_ids = tuple(
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in (check_seed, covariance_seed, null_seed)
    )
    check_rng = np.random.default_rng(check_seed)

    symmetry_errors = []
    curvatures = []
    fisher_linearity_errors = []
    physical_linearity_errors = []
    for _ in range(n_checks):
        first = check_rng.standard_normal(n_parameter)
        second = check_rng.standard_normal(n_parameter)
        first /= max(float(np.linalg.norm(first)), np.finfo(float).tiny)
        second /= max(float(np.linalg.norm(second)), np.finfo(float).tiny)
        first_image = apply_fisher(first)
        second_image = apply_fisher(second)
        sum_image = apply_fisher(first + second)
        symmetry_errors.append(
            abs(float(np.dot(first, second_image) - np.dot(first_image, second)))
            / max(
                float(np.linalg.norm(first) * np.linalg.norm(second_image))
                + float(np.linalg.norm(first_image) * np.linalg.norm(second)),
                1.0,
            )
        )
        for vector, image in ((first, first_image), (second, second_image)):
            curvatures.append(
                float(np.dot(vector, image))
                / max(float(np.linalg.norm(vector) * np.linalg.norm(image)), 1.0)
            )
        fisher_linearity_errors.append(
            _relative_error(
                sum_image - first_image - second_image,
                np.linalg.norm(first_image) + np.linalg.norm(second_image),
            )
        )
        first_physical = apply_physical(first)
        second_physical = apply_physical(second)
        sum_physical = apply_physical(first + second)
        physical_linearity_errors.append(
            _relative_error(
                sum_physical - first_physical - second_physical,
                np.linalg.norm(first_physical) + np.linalg.norm(second_physical),
            )
        )

    first_detector = check_rng.standard_normal(detector_shape)
    second_detector = check_rng.standard_normal(detector_shape)
    first_detector_image = apply_detector(first_detector)
    second_detector_image = apply_detector(second_detector)
    detector_sum_image = apply_detector(first_detector + second_detector)
    detector_linearity_error = _relative_error(
        detector_sum_image - first_detector_image - second_detector_image,
        np.linalg.norm(first_detector_image) + np.linalg.norm(second_detector_image),
    )
    zero_operator_norm = float(np.linalg.norm(zero_fisher))
    physical_zero_norm = float(np.linalg.norm(physical_zero))
    maximum_symmetry_error = max(symmetry_errors, default=0.0)
    minimum_curvature = min(curvatures, default=0.0)
    maximum_fisher_linearity = max(fisher_linearity_errors, default=0.0)
    maximum_physical_linearity = max(physical_linearity_errors, default=0.0)
    checks_passed = bool(
        zero_operator_norm <= float(options.zero_tolerance)
        and physical_zero_norm <= float(options.zero_tolerance)
        and maximum_symmetry_error <= float(options.symmetry_tolerance)
        and minimum_curvature >= -float(options.psd_tolerance)
        and maximum_fisher_linearity <= float(options.linearity_tolerance)
        and maximum_physical_linearity <= float(options.linearity_tolerance)
        and detector_linearity_error <= float(options.linearity_tolerance)
    )
    operator_checks = StochasticFisherOperatorChecks1D(
        zero_operator_norm=zero_operator_norm,
        maximum_symmetry_error=maximum_symmetry_error,
        minimum_normalized_curvature=minimum_curvature,
        maximum_fisher_linearity_error=maximum_fisher_linearity,
        detector_vjp_linearity_error=detector_linearity_error,
        physical_jvp_zero_norm=physical_zero_norm,
        physical_jvp_linearity_error=maximum_physical_linearity,
        passed=checks_passed,
    )
    if not checks_passed:
        raise ValueError(
            "Fisher, detector_vjp, or physical_jvp contract checks failed"
        )

    effective_relative_tolerance = max(
        float(options.relative_residual_tolerance),
        50.0 * float(np.finfo(float).eps),
    )

    def solve(right_hand_side: np.ndarray) -> PCGSolveDiagnostics1D:
        return pcg_solve_observability_1d(
            apply_fisher,
            right_hand_side,
            preconditioner_diagonal=preconditioner,
            maximum_iterations=max_iterations,
            relative_residual_tolerance=effective_relative_tolerance,
            absolute_residual_tolerance=options.absolute_residual_tolerance,
            stagnation_iterations=options.stagnation_iterations,
            stagnation_relative_improvement=(
                options.stagnation_relative_improvement
            ),
            curvature_tolerance=options.curvature_tolerance,
        )

    covariance_rng = np.random.default_rng(covariance_seed)
    covariance_samples = np.empty((n_covariance, n_output), dtype=float)
    covariance_diagnostics = []
    for probe_index in range(n_covariance):
        probe = covariance_rng.standard_normal(detector_shape)
        diagnostic = solve(apply_detector(probe))
        covariance_diagnostics.append(diagnostic)
        covariance_samples[probe_index] = apply_physical(diagnostic.solution)

    null_rng = np.random.default_rng(null_seed)
    null_samples = np.empty((n_null, n_output), dtype=float)
    global_null_samples = np.empty(n_null, dtype=float)
    null_diagnostics = []
    for probe_index in range(n_null):
        parameter_probe = null_rng.standard_normal(n_parameter)
        diagnostic = solve(apply_fisher(parameter_probe))
        null_diagnostics.append(diagnostic)
        null_direction = parameter_probe - np.asarray(diagnostic.solution)
        null_samples[probe_index] = apply_physical(null_direction)
        global_null_samples[probe_index] = float(np.dot(null_direction, null_direction))

    covariance_diagnostics_tuple = tuple(covariance_diagnostics)
    null_diagnostics_tuple = tuple(null_diagnostics)
    all_diagnostics = covariance_diagnostics_tuple + null_diagnostics_tuple
    numerically_valid = bool(
        all(
            _solver_accepted(diagnostic, effective_relative_tolerance)
            for diagnostic in all_diagnostics
        )
    )
    actual_iterations = int(sum(item.iterations for item in all_diagnostics))
    converged_solves = int(
        sum(
            _solver_accepted(item, effective_relative_tolerance)
            for item in all_diagnostics
        )
    )
    budget_passed = bool(
        len(all_diagnostics) <= operator.index(options.maximum_pcg_solves)
        and actual_iterations
        <= operator.index(options.maximum_total_pcg_iterations)
        and counts["fisher"]
        <= operator.index(options.maximum_fisher_matvec_calls)
    )
    numerically_valid = bool(numerically_valid and budget_passed)
    budget = StochasticFisherBudget1D(
        requested_pcg_solves=n_solves,
        maximum_pcg_solves=operator.index(options.maximum_pcg_solves),
        configured_worst_case_pcg_iterations=worst_iterations,
        maximum_total_pcg_iterations=operator.index(
            options.maximum_total_pcg_iterations
        ),
        configured_worst_case_fisher_matvec_calls=worst_matvec_calls,
        maximum_fisher_matvec_calls=operator.index(
            options.maximum_fisher_matvec_calls
        ),
        attempted_pcg_solves=len(all_diagnostics),
        converged_pcg_solves=converged_solves,
        actual_pcg_iterations=actual_iterations,
        actual_fisher_matvec_calls=counts["fisher"],
        detector_vjp_calls=counts["detector"],
        physical_jvp_calls=counts["physical"],
        all_budget_checks_passed=budget_passed,
    )

    n_bounded_quantities = 2 * n_output + 1
    per_quantity_error = (
        1.0 - float(options.simultaneous_confidence)
    ) / n_bounded_quantities
    block_estimates = []
    if numerically_valid:
        marginal, marginal_lower, marginal_upper, marginal_mcse = (
            _chi_square_variance_bounds(
                covariance_samples, per_quantity_error
            )
        )
        leakage, leakage_lower, leakage_upper, leakage_mcse = (
            _chi_square_variance_bounds(null_samples, per_quantity_error)
        )
        for name, rows, columns in blocks:
            products = (
                covariance_samples[:, rows, None]
                * covariance_samples[:, None, columns]
            )
            block_estimates.append(
                StochasticPhysicalBlockEstimate1D(
                    name=name,
                    row_indices=_readonly(rows, dtype=np.int64),
                    column_indices=_readonly(columns, dtype=np.int64),
                    covariance=_readonly(np.mean(products, axis=0)),
                    monte_carlo_standard_error=_readonly(
                        np.std(products, axis=0, ddof=1) / np.sqrt(n_covariance)
                    ),
                )
            )
        global_estimate = float(np.mean(global_null_samples))
        global_mcse = float(
            np.std(global_null_samples, ddof=1) / np.sqrt(n_null)
        )
        from scipy.stats import chi2

        total_null_norm = float(np.sum(global_null_samples))
        accepted_ranks = []
        zero_threshold = float(options.zero_tolerance) ** 2 * n_null
        if total_null_norm <= zero_threshold:
            accepted_ranks.append(0)
        for null_rank in range(1, n_parameter + 1):
            degrees = n_null * null_rank
            lower_quantile = float(
                chi2.ppf(per_quantity_error / 2.0, degrees)
            )
            upper_quantile = float(
                chi2.ppf(1.0 - per_quantity_error / 2.0, degrees)
            )
            if lower_quantile <= total_null_norm <= upper_quantile:
                accepted_ranks.append(null_rank)
        rank_interval_valid = bool(accepted_ranks)
        if not accepted_ranks:
            accepted_ranks = list(range(n_parameter + 1))
        rank_lower = min(accepted_ranks)
        rank_upper = max(accepted_ranks)
    else:
        marginal = np.full(n_output, np.nan)
        marginal_lower = np.zeros(n_output)
        marginal_upper = np.full(n_output, np.inf)
        marginal_mcse = np.full(n_output, np.inf)
        leakage = np.full(n_output, np.nan)
        leakage_lower = np.zeros(n_output)
        leakage_upper = np.full(n_output, np.inf)
        leakage_mcse = np.full(n_output, np.inf)
        for name, rows, columns in blocks:
            shape = (rows.size, columns.size)
            block_estimates.append(
                StochasticPhysicalBlockEstimate1D(
                    name=name,
                    row_indices=_readonly(rows, dtype=np.int64),
                    column_indices=_readonly(columns, dtype=np.int64),
                    covariance=_readonly(np.full(shape, np.nan)),
                    monte_carlo_standard_error=_readonly(np.full(shape, np.inf)),
                )
            )
        global_estimate = float("nan")
        global_mcse = float("inf")
        rank_lower = 0
        rank_upper = n_parameter
        accepted_ranks = list(range(n_parameter + 1))
        rank_interval_valid = False

    return StochasticFisherScreeningResult1D(
        physical_marginal_variance=_readonly(marginal),
        physical_marginal_variance_lower=_readonly(marginal_lower),
        physical_marginal_variance_upper=_readonly(marginal_upper),
        physical_marginal_variance_mcse=_readonly(marginal_mcse),
        covariance_blocks=tuple(block_estimates),
        physical_null_leakage=_readonly(leakage),
        physical_null_leakage_lower=_readonly(leakage_lower),
        physical_null_leakage_upper=_readonly(leakage_upper),
        physical_null_leakage_mcse=_readonly(leakage_mcse),
        global_nullity_estimate=global_estimate,
        global_nullity_mcse=global_mcse,
        global_nullity_rank_lower=rank_lower,
        global_nullity_rank_upper=rank_upper,
        global_nullity_rank_confidence_set=tuple(accepted_ranks),
        global_nullity_rank_interval_valid=rank_interval_valid,
        covariance_solver_diagnostics=covariance_diagnostics_tuple,
        null_solver_diagnostics=null_diagnostics_tuple,
        operator_checks=operator_checks,
        budget=budget,
        covariance_probe_count=n_covariance,
        null_probe_count=n_null,
        simultaneous_confidence=float(options.simultaneous_confidence),
        per_quantity_error_probability=per_quantity_error,
        random_seed=operator.index(options.random_seed),
        check_stream_seed=seed_ids[0],
        covariance_stream_seed=seed_ids[1],
        null_stream_seed=seed_ids[2],
        numerically_valid=numerically_valid,
        factor_covariance_verified=False,
        bound_method=(
            "conditional chi-square marginal pivots with Bonferroni "
            "simultaneous coverage; empirical product MCSE for cross blocks"
        ),
    )
