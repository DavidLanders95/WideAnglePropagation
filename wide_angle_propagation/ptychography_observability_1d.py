"""Marginalized local plug-in Poisson-Fisher diagnostics for small problems.

The dense implementation is the correctness reference for the prepared,
matrix-free phase-1 solver in this module. Both profile correlations between
vacancies, smooth displacement, and active-site translation, and explicitly
test whether reported physical quantities lie in the Fisher row space.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import operator
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .propagation_methods import energy2wavelength
from .ptychography_1d import (
    LatticeSiteModel1D,
    LatticeSiteReconstruction1D,
    PreparedLatticeSiteReconstruction1D,
    _validate_prepared_static_contract_1d,
    lattice_site_displacements_1d,
    render_lattice_site_potential_1d,
    simulate_glancing_scan_1d,
)
from .ptychography_diagnostics_1d import (
    PoissonCountingModel1D,
    _coordinate_tolerances,
    _floating_host_array,
    _validated_reconstruction_site_state_1d,
    validate_poisson_counting_model_1d,
)
from .ptychography_support_contract_1d import LatticeSiteRole1D


__all__ = [
    "LatticeDisplacementBasis1D",
    "LatticeSiteObservability1D",
    "MatrixFreeObservabilityOptions1D",
    "PCGSolveDiagnostics1D",
    "PreparedStochasticObservability1D",
    "PreparedStochasticObservabilitySplit1D",
    "PreparedNuisanceOptions1D",
    "SiteObservabilityOptions1D",
    "SiteObservabilitySplit1D",
    "WhitenedNuisanceProfile1D",
    "estimate_lattice_site_observability_1d",
    "estimate_prepared_lattice_site_observability_matrix_free_1d",
    "estimate_prepared_lattice_site_observability_stochastic_1d",
    "lattice_displacement_basis_1d",
    "load_lattice_site_observability_1d",
    "marginal_covariance_from_jacobian_1d",
    "pcg_solve_observability_1d",
    "poisson_counting_model_from_prepared_1d",
    "prepared_whitened_nuisance_profile_1d",
    "save_lattice_site_observability_1d",
]


Array = Any


@dataclass(frozen=True)
class LatticeDisplacementBasis1D:
    """Gauge-free controls with coefficients measured in RMS site motion."""

    control_basis: Array
    site_basis: Array
    interpolation_matrix: Array
    singular_values: Array
    numerical_rank: int
    relative_reconstruction_error: float


@dataclass(frozen=True)
class SiteObservabilityOptions1D:
    """Dense Fisher rank and physical reporting thresholds."""

    dense_max_parameters: int = 512
    vacancy_threshold: float = 0.5
    vacancy_margin: float = 0.1
    minimum_vacancy_z: float = 3.0
    displacement_confidence: float = 0.95
    maximum_displacement_radius_A: float = 0.05
    vacancy_parameter_scale: float = 0.1
    displacement_parameter_scale_A: float = 0.05
    control_basis_rtol: float = 1e-10
    fisher_rank_rtol: float = 1e-9
    rematerialize: bool = True


@dataclass(frozen=True)
class MatrixFreeObservabilityOptions1D:
    """Exact matrix-free Fisher/PCG policy for selected lattice sites.

    This phase-1 solver avoids materializing the detector Jacobian.  It still
    requires one linear solve per requested physical output, so callers should
    select a scientifically motivated subset for notebook-scale problems.
    """

    scan_batch_size: int = 4
    maximum_iterations: int = 256
    relative_residual_tolerance: float = 1e-7
    absolute_residual_tolerance: float = 0.0
    stagnation_iterations: int = 12
    stagnation_relative_improvement: float = 1e-3
    curvature_tolerance: float = 1e-12
    operator_check_vectors: int = 2
    symmetry_tolerance: float = 1e-7
    psd_tolerance: float = 1e-10
    projector_tolerance: float = 1e-8
    nuisance_rank_rtol: float = 1e-10
    maximum_nuisance_columns: int = 32
    maximum_selected_sites: int = 32
    exhaustive: bool = False
    exhaustive_max_parameters: int = 64
    exhaustive_relative_tolerance: float = 2e-6
    fisher_rank_rtol: float = 1e-9
    vacancy_threshold: float = 0.5
    vacancy_margin: float = 0.1
    minimum_vacancy_z: float = 3.0
    displacement_confidence: float = 0.95
    maximum_displacement_radius_A: float = 0.05
    vacancy_parameter_scale: float = 0.1
    displacement_parameter_scale_A: float = 0.05
    control_basis_rtol: float = 1e-10


@dataclass(frozen=True)
class WhitenedNuisanceProfile1D:
    """Explicit detector-space tangent in ``2*sqrt(expected counts)`` units."""

    tangent_matrix: Array
    parameter_names: tuple[str, ...] = ()
    profile_id: str = "explicit_whitened_nuisance_tangent"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedNuisanceOptions1D:
    """Bounded low-rank nuisance directions derived from a prepared problem.

    These directions profile local calibration uncertainty; they do not assert
    that the complete experimental nuisance space has been enumerated.
    """

    include_scan_origin_shift: bool = True
    include_probe_transverse_shift: bool = True
    include_probe_tilt: bool = True
    include_probe_log_width: bool = True
    include_detector_frequency_offset: bool = True
    include_detector_log_gain: bool = True
    include_detector_dark_offset: bool = True


_PREPARED_NUISANCE_CONTRACT_1D = "prepared_poisson_nuisance_autodiff:v2"
_PREPARED_NUISANCE_OPTION_NAME_PAIRS_1D = (
    ("include_scan_origin_shift", "scan_origin_shift_A"),
    ("include_probe_transverse_shift", "probe_transverse_shift_A"),
    ("include_probe_tilt", "probe_tilt_rad"),
    ("include_probe_log_width", "probe_log_width"),
    (
        "include_detector_frequency_offset",
        "detector_frequency_offset_inverse_A",
    ),
    ("include_detector_log_gain", "detector_log_signal_gain"),
    ("include_detector_dark_offset", "detector_dark_offset_electrons"),
)


@dataclass(frozen=True)
class PCGSolveDiagnostics1D:
    """Zero-start PCG solution with recomputed true-residual diagnostics."""

    solution: Array
    converged: bool
    iterations: int
    stop_reason: str
    residual_norm: float
    relative_residual: float
    residual_norm_history: Array
    curvature_history: Array
    breakdown: bool
    stagnated: bool


@dataclass(frozen=True)
class SiteObservabilitySplit1D:
    """Marginal uncertainty and site decisions for one scan subset."""

    scan_indices: Array
    vacancy_standard_error: Array
    vacancy_z_to_decision_boundary: Array
    displacement_covariance_A2: Array
    displacement_confidence_radius_A: Array
    vacancy_information_adequate: Array
    displacement_information_adequate: Array
    site_observable: Array
    physical_output_estimable: Array
    solver_verified: bool
    effective_rank: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LatticeSiteObservability1D:
    """Fit/audit observability with explicit provenance and trust suitability."""

    site_coordinates: Array
    fit: SiteObservabilitySplit1D
    audit: SiteObservabilitySplit1D | None
    vacancy_information_adequate: Array
    displacement_information_adequate: Array
    site_observable: Array
    ideal_poisson_information: bool
    calibrated_noise: bool
    nuisance_scope_complete: bool
    suitable_for_trust_gate: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedStochasticObservabilitySplit1D:
    """One fit/audit all-site stochastic screen and operator provenance."""

    scan_indices: Array
    screening: Any
    operator_checks_passed: bool
    projector_checks_passed: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedStochasticObservability1D:
    """Prepared all-site screen which can only nominate exact follow-up sites."""

    site_coordinates: Array
    fit: PreparedStochasticObservabilitySplit1D
    audit: PreparedStochasticObservabilitySplit1D | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def structurally_trusted(self) -> bool:
        return False

    @property
    def suitable_for_trust_gate(self) -> bool:
        return False


def _validated_options(options: SiteObservabilityOptions1D) -> None:
    if not isinstance(options, SiteObservabilityOptions1D):
        raise TypeError("options must be a SiteObservabilityOptions1D instance")
    if operator.index(options.dense_max_parameters) < 1:
        raise ValueError("dense_max_parameters must be positive")
    for name in (
        "minimum_vacancy_z",
        "maximum_displacement_radius_A",
        "vacancy_parameter_scale",
        "displacement_parameter_scale_A",
        "control_basis_rtol",
        "fisher_rank_rtol",
    ):
        value = float(getattr(options, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"options.{name} must be finite and positive")
    threshold = float(options.vacancy_threshold)
    margin = float(options.vacancy_margin)
    if not 0.0 < threshold < 1.0 or not 0.0 <= margin < min(
        threshold, 1.0 - threshold
    ):
        raise ValueError("vacancy threshold and margin are incompatible")
    confidence = float(options.displacement_confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("displacement_confidence must lie strictly in (0, 1)")
    if not isinstance(options.rematerialize, (bool, np.bool_)):
        raise TypeError("options.rematerialize must be a boolean")


def _validated_matrix_free_options(
    options: MatrixFreeObservabilityOptions1D,
) -> None:
    if not isinstance(options, MatrixFreeObservabilityOptions1D):
        raise TypeError(
            "options must be a MatrixFreeObservabilityOptions1D instance"
        )
    for name, minimum in (
        ("scan_batch_size", 1),
        ("maximum_iterations", 1),
        ("stagnation_iterations", 1),
        ("operator_check_vectors", 1),
        ("maximum_nuisance_columns", 0),
        ("maximum_selected_sites", 1),
        ("exhaustive_max_parameters", 1),
    ):
        if operator.index(getattr(options, name)) < minimum:
            raise ValueError(f"options.{name} must be at least {minimum}")
    for name in (
        "relative_residual_tolerance",
        "curvature_tolerance",
        "symmetry_tolerance",
        "psd_tolerance",
        "projector_tolerance",
        "nuisance_rank_rtol",
        "exhaustive_relative_tolerance",
        "fisher_rank_rtol",
        "minimum_vacancy_z",
        "maximum_displacement_radius_A",
        "vacancy_parameter_scale",
        "displacement_parameter_scale_A",
        "control_basis_rtol",
    ):
        value = float(getattr(options, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"options.{name} must be finite and positive")
    absolute_tolerance = float(options.absolute_residual_tolerance)
    if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError(
            "options.absolute_residual_tolerance must be finite and non-negative"
        )
    stagnation_improvement = float(options.stagnation_relative_improvement)
    if not 0.0 < stagnation_improvement < 1.0:
        raise ValueError(
            "options.stagnation_relative_improvement must lie in (0, 1)"
        )
    threshold = float(options.vacancy_threshold)
    margin = float(options.vacancy_margin)
    if not 0.0 < threshold < 1.0 or not 0.0 <= margin < min(
        threshold, 1.0 - threshold
    ):
        raise ValueError("vacancy threshold and margin are incompatible")
    confidence = float(options.displacement_confidence)
    if not 0.0 < confidence < 1.0:
        raise ValueError("displacement_confidence must lie strictly in (0, 1)")
    if not isinstance(options.exhaustive, (bool, np.bool_)):
        raise TypeError("options.exhaustive must be a boolean")


def _validated_prepared_nuisance_options(
    options: PreparedNuisanceOptions1D | None,
) -> PreparedNuisanceOptions1D:
    options = PreparedNuisanceOptions1D() if options is None else options
    if not isinstance(options, PreparedNuisanceOptions1D):
        raise TypeError(
            "options must be a PreparedNuisanceOptions1D instance or None"
        )
    names = tuple(
        name for name, _ in _PREPARED_NUISANCE_OPTION_NAME_PAIRS_1D
    )
    for name in names:
        if not isinstance(getattr(options, name), (bool, np.bool_)):
            raise TypeError(f"options.{name} must be a boolean")
    if not any(bool(getattr(options, name)) for name in names):
        raise ValueError("at least one prepared nuisance direction is required")
    return options


def pcg_solve_observability_1d(
    matvec: Any,
    right_hand_side: Any,
    *,
    preconditioner_diagonal: Any | None = None,
    maximum_iterations: int = 256,
    relative_residual_tolerance: float = 1e-7,
    absolute_residual_tolerance: float = 0.0,
    stagnation_iterations: int = 12,
    stagnation_relative_improvement: float = 1e-3,
    curvature_tolerance: float = 1e-12,
) -> PCGSolveDiagnostics1D:
    """Solve a symmetric positive-semidefinite system with audited PCG.

    The initial solution is exactly zero.  The residual is recomputed as
    ``b - A(x)`` after every update rather than inferred only from the CG
    recurrence.  Non-positive curvature, numerical failure, and lack of
    residual improvement are distinct terminal states.
    """
    if not callable(matvec):
        raise TypeError("matvec must be callable")
    rhs = np.asarray(right_hand_side)
    if rhs.ndim != 1 or not np.issubdtype(rhs.dtype, np.floating):
        raise TypeError("right_hand_side must be a one-dimensional floating array")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("right_hand_side must contain only finite values")
    n_parameter = rhs.size
    if preconditioner_diagonal is None:
        diagonal = np.ones(n_parameter, dtype=rhs.dtype)
    else:
        diagonal = np.asarray(preconditioner_diagonal, dtype=rhs.dtype)
        if diagonal.shape != rhs.shape:
            raise ValueError("preconditioner_diagonal must match right_hand_side")
        if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
            raise ValueError(
                "preconditioner_diagonal must contain finite positive values"
            )
    max_iterations = operator.index(maximum_iterations)
    stagnation_window = operator.index(stagnation_iterations)
    if max_iterations < 1 or stagnation_window < 1:
        raise ValueError("iteration counts must be positive")
    relative_tolerance = float(relative_residual_tolerance)
    absolute_tolerance = float(absolute_residual_tolerance)
    stagnation_improvement = float(stagnation_relative_improvement)
    curvature_relative_tolerance = float(curvature_tolerance)
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("relative_residual_tolerance must be finite and positive")
    if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError(
            "absolute_residual_tolerance must be finite and non-negative"
        )
    if not 0.0 < stagnation_improvement < 1.0:
        raise ValueError("stagnation_relative_improvement must lie in (0, 1)")
    if (
        not np.isfinite(curvature_relative_tolerance)
        or curvature_relative_tolerance <= 0.0
    ):
        raise ValueError("curvature_tolerance must be finite and positive")

    def applied(value: np.ndarray) -> np.ndarray:
        result = np.asarray(matvec(value), dtype=rhs.dtype)
        if result.shape != rhs.shape:
            raise ValueError("matvec returned an array with incompatible shape")
        return result

    solution = np.zeros_like(rhs)
    true_residual = rhs - applied(solution)
    rhs_norm = float(np.linalg.norm(rhs))
    residual_norm = float(np.linalg.norm(true_residual))
    residual_history = [residual_norm]
    curvature_history: list[float] = []
    target = max(absolute_tolerance, relative_tolerance * rhs_norm)
    if not np.isfinite(residual_norm):
        return PCGSolveDiagnostics1D(
            solution=solution,
            converged=False,
            iterations=0,
            stop_reason="nonfinite_initial_residual",
            residual_norm=residual_norm,
            relative_residual=float("inf"),
            residual_norm_history=np.asarray(residual_history),
            curvature_history=np.empty(0),
            breakdown=True,
            stagnated=False,
        )
    if residual_norm <= target:
        relative = 0.0 if rhs_norm == 0.0 else residual_norm / rhs_norm
        return PCGSolveDiagnostics1D(
            solution=solution,
            converged=True,
            iterations=0,
            stop_reason="zero_rhs" if rhs_norm == 0.0 else "converged",
            residual_norm=residual_norm,
            relative_residual=relative,
            residual_norm_history=np.asarray(residual_history),
            curvature_history=np.empty(0),
            breakdown=False,
            stagnated=False,
        )

    preconditioned = true_residual / diagonal
    rho = float(np.dot(true_residual, preconditioned))
    direction = preconditioned.copy()
    stop_reason = "maximum_iterations"
    breakdown = False
    stagnated = False
    completed = 0
    if not np.isfinite(rho) or rho <= 0.0:
        stop_reason = "preconditioner_breakdown"
        breakdown = True
    else:
        for iteration in range(1, max_iterations + 1):
            applied_direction = applied(direction)
            curvature = float(np.dot(direction, applied_direction))
            curvature_history.append(curvature)
            curvature_scale = float(
                np.linalg.norm(direction) * np.linalg.norm(applied_direction)
            )
            if (
                not np.isfinite(curvature)
                or not np.isfinite(curvature_scale)
                or curvature_scale == 0.0
                or curvature
                <= curvature_relative_tolerance * curvature_scale
            ):
                stop_reason = (
                    "negative_curvature_breakdown"
                    if curvature < 0.0
                    else "zero_curvature_breakdown"
                )
                breakdown = True
                completed = iteration - 1
                break
            step = rho / curvature
            if not np.isfinite(step):
                stop_reason = "nonfinite_step_breakdown"
                breakdown = True
                completed = iteration - 1
                break
            solution = solution + step * direction
            true_residual = rhs - applied(solution)
            residual_norm = float(np.linalg.norm(true_residual))
            residual_history.append(residual_norm)
            completed = iteration
            if not np.isfinite(residual_norm):
                stop_reason = "nonfinite_residual_breakdown"
                breakdown = True
                break
            if residual_norm <= target:
                stop_reason = "converged"
                break
            if len(residual_history) > stagnation_window:
                earlier_best = min(residual_history[:-stagnation_window])
                recent_best = min(residual_history[-stagnation_window:])
                if recent_best >= (1.0 - stagnation_improvement) * earlier_best:
                    stop_reason = "stagnation"
                    stagnated = True
                    break
            preconditioned = true_residual / diagonal
            new_rho = float(np.dot(true_residual, preconditioned))
            if not np.isfinite(new_rho) or new_rho <= 0.0:
                stop_reason = "preconditioner_breakdown"
                breakdown = True
                break
            direction = preconditioned + (new_rho / rho) * direction
            rho = new_rho
    relative = residual_norm / rhs_norm if rhs_norm else 0.0
    return PCGSolveDiagnostics1D(
        solution=solution,
        converged=bool(residual_norm <= target and not breakdown),
        iterations=completed,
        stop_reason=stop_reason,
        residual_norm=residual_norm,
        relative_residual=relative,
        residual_norm_history=np.asarray(residual_history),
        curvature_history=np.asarray(curvature_history),
        breakdown=breakdown,
        stagnated=stagnated,
    )


def lattice_displacement_basis_1d(
    model: LatticeSiteModel1D,
    *,
    rtol: float = 1e-10,
) -> LatticeDisplacementBasis1D:
    """Construct the SVD basis of zero-mean site motions reachable by controls."""
    sites = jnp.asarray(model.site_coordinates)
    controls_s = jnp.asarray(model.control_coordinates_s)
    controls_u = jnp.asarray(model.control_coordinates_u)
    n_site = int(sites.shape[0])
    control_shape = (int(controls_s.size), int(controls_u.size), 2)
    n_control = control_shape[0] * control_shape[1]
    tolerance = float(rtol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rtol must be finite and positive")
    sites_host = _floating_host_array("model.site_coordinates", sites, 2)
    dtype_epsilon = float(np.finfo(sites_host.dtype).eps)
    effective_tolerance = max(tolerance, 8.0 * dtype_epsilon)

    def interpolate_component(flat_controls: Array) -> Array:
        controls = jnp.zeros(control_shape, dtype=sites.dtype)
        controls = controls.at[..., 0].set(
            flat_controls.reshape(control_shape[:2])
        )
        return lattice_site_displacements_1d(
            sites, controls, controls_s, controls_u
        )[:, 0]

    interpolation = np.asarray(
        jax.jacfwd(interpolate_component)(jnp.zeros(n_control, dtype=sites.dtype))
    )
    centering = np.eye(n_site) - np.ones((n_site, n_site)) / n_site
    centered = centering @ interpolation
    left, singular_values, right_transpose = np.linalg.svd(
        centered, full_matrices=False
    )
    scale = singular_values[0] if singular_values.size else 0.0
    interpolation_scale = max(float(np.linalg.norm(interpolation, ord=2)), 1.0)
    if scale <= effective_tolerance * interpolation_scale:
        rank = 0
    else:
        rank = int(
            np.count_nonzero(singular_values > effective_tolerance * scale)
        )
    root_n = np.sqrt(n_site)
    if rank:
        site_basis = root_n * left[:, :rank]
        control_basis = (
            root_n
            * right_transpose[:rank].T
            / singular_values[:rank][None, :]
        )
        # ``P W`` fixes the site-mean gauge, while its right singular vectors can
        # still contain a constant-control component. Subtract that component in
        # control space; bilinear interpolation maps an all-ones control exactly
        # to an all-ones site displacement.
        site_motion = interpolation @ control_basis
        control_basis -= np.ones((n_control, 1)) * np.mean(
            site_motion, axis=0, keepdims=True
        )
        reconstructed = interpolation @ control_basis
        error = np.linalg.norm(reconstructed - site_basis) / np.linalg.norm(
            site_basis
        )
        if not np.isfinite(error) or error > 10.0 * effective_tolerance:
            raise RuntimeError(
                "gauge-free displacement basis does not preserve interpolated motion"
            )
    else:
        site_basis = np.empty((n_site, 0), dtype=interpolation.dtype)
        control_basis = np.empty((n_control, 0), dtype=interpolation.dtype)
        error = 0.0
    return LatticeDisplacementBasis1D(
        control_basis=jnp.asarray(control_basis),
        site_basis=jnp.asarray(site_basis),
        interpolation_matrix=jnp.asarray(interpolation),
        singular_values=jnp.asarray(singular_values[:rank]),
        numerical_rank=rank,
        relative_reconstruction_error=float(error),
    )


def marginal_covariance_from_jacobian_1d(
    jacobian: Any,
    physical_output_jacobian: Any,
    *,
    rank_rtol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return profiled covariance, estimability mask, and numerical rank.

    Non-estimable rows receive infinite diagonal variance. This avoids the
    misleading finite minimum-norm variance produced by blindly applying a
    pseudoinverse to a singular Fisher matrix.
    """
    detector_jacobian = np.asarray(jacobian, dtype=float)
    output_jacobian = np.asarray(physical_output_jacobian, dtype=float)
    if detector_jacobian.ndim != 2 or output_jacobian.ndim != 2:
        raise ValueError("jacobians must be two-dimensional")
    if detector_jacobian.shape[1] != output_jacobian.shape[1]:
        raise ValueError("jacobians must have the same parameter dimension")
    if np.any(~np.isfinite(detector_jacobian)) or np.any(
        ~np.isfinite(output_jacobian)
    ):
        raise ValueError("jacobians must contain only finite values")
    tolerance = float(rank_rtol)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("rank_rtol must be finite and positive")

    _, singular_values, right_transpose = np.linalg.svd(
        detector_jacobian, full_matrices=False
    )
    scale = singular_values[0] if singular_values.size else 0.0
    rank = int(np.count_nonzero(singular_values > tolerance * scale))
    if rank:
        row_basis = right_transpose[:rank].T
        projected = output_jacobian @ row_basis
        reconstructed_rows = projected @ row_basis.T
        residual_norm = np.linalg.norm(
            output_jacobian - reconstructed_rows, axis=1
        )
        row_norm = np.linalg.norm(output_jacobian, axis=1)
        zero_rows = row_norm == 0.0
        estimable = zero_rows | (residual_norm <= tolerance * row_norm)
        transformed = projected / singular_values[:rank][None, :]
        covariance = transformed @ transformed.T
    else:
        estimable = np.zeros(output_jacobian.shape[0], dtype=bool)
        covariance = np.zeros(
            (output_jacobian.shape[0], output_jacobian.shape[0]), dtype=float
        )
    invalid = ~estimable
    covariance[invalid, :] = np.nan
    covariance[:, invalid] = np.nan
    covariance[invalid, invalid] = np.inf
    return covariance, estimable, rank


def _indices(
    values: Sequence[int] | None,
    *,
    n_scan: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    indices = np.asarray(values)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(f"{name} must be a one-dimensional integer sequence")
    indices = indices.astype(np.int32, copy=False)
    if (
        not indices.size
        or np.unique(indices).size != indices.size
        or np.any(indices < 0)
        or np.any(indices >= n_scan)
    ):
        raise ValueError(f"{name} must contain unique valid scan indices")
    return indices


def estimate_lattice_site_observability_1d(
    model: LatticeSiteModel1D,
    reconstruction: LatticeSiteReconstruction1D,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    counting_model: PoissonCountingModel1D,
    *,
    fit_indices: Sequence[int] | None = None,
    audit_indices: Sequence[int] | None = None,
    detector_mask: Any | None = None,
    options: SiteObservabilityOptions1D | None = None,
) -> LatticeSiteObservability1D:
    """Compute dense local plug-in marginalized Fisher diagnostics."""
    options = SiteObservabilityOptions1D() if options is None else options
    _validated_options(options)
    validate_poisson_counting_model_1d(counting_model)
    dose = float(counting_model.electrons_per_pattern)
    background = float(counting_model.background_electrons_per_pixel)
    floor = float(counting_model.minimum_expected_electrons)

    vacancies_device, sites, displaced_device = (
        _validated_reconstruction_site_state_1d(model, reconstruction)
    )
    result_sites = np.asarray(sites)
    vacancies = np.asarray(vacancies_device, dtype=float)
    total_displacement = np.asarray(displaced_device) - result_sites
    rigid_host = _floating_host_array(
        "reconstruction.rigid_displacement",
        reconstruction.rigid_displacement,
        1,
    )
    if rigid_host.shape != (2,):
        raise ValueError("reconstruction.rigid_displacement must have shape (2,)")
    rigid = np.asarray(rigid_host, dtype=float)
    residual = total_displacement - rigid
    residual -= np.mean(residual, axis=0, keepdims=True)
    n_site = len(vacancies)
    basis = lattice_displacement_basis_1d(
        model, rtol=options.control_basis_rtol
    )
    site_basis = np.asarray(basis.site_basis)
    control_basis = jnp.asarray(basis.control_basis)
    coefficients = np.column_stack(
        [
            site_basis.T @ residual[:, component] / n_site
            for component in range(2)
        ]
    )
    represented = site_basis @ coefficients
    representation_error = np.linalg.norm(represented - residual) / max(
        np.linalg.norm(residual), 1.0
    )
    coordinate_dtype = np.asarray(model.site_coordinates).dtype
    representation_tolerance = max(
        float(options.control_basis_rtol),
        8.0 * float(np.finfo(coordinate_dtype).eps),
    )
    if (
        not np.isfinite(representation_error)
        or representation_error > 20.0 * representation_tolerance
    ):
        raise ValueError(
            "reconstructed residual displacement is outside the control-field basis"
        )

    vacancy_scale = float(options.vacancy_parameter_scale)
    displacement_scale = float(options.displacement_parameter_scale_A)
    rank = basis.numerical_rank
    x0 = np.concatenate(
        [
            vacancies / vacancy_scale,
            coefficients[:, 0] / displacement_scale,
            coefficients[:, 1] / displacement_scale,
            rigid / displacement_scale,
        ]
    )
    if x0.size > options.dense_max_parameters:
        raise ValueError(
            f"dense observability has {x0.size} parameters, above "
            f"dense_max_parameters={options.dense_max_parameters}; use the "
            "prepared matrix-free observability entry point"
        )
    x0_device = jnp.asarray(x0, dtype=model.reference_potential.dtype)

    starts = jnp.asarray(window_starts)
    probes = jnp.asarray(input_probe)
    kernel = jnp.asarray(propagation_kernel)
    if starts.ndim != 1 or not jnp.issubdtype(starts.dtype, jnp.integer):
        raise TypeError("window_starts must be a one-dimensional integer array")
    n_scan = int(starts.shape[0])
    n_detector = int(model.reference_potential.shape[1])
    if probes.ndim == 1:
        if probes.shape[0] != n_detector:
            raise ValueError("input_probe must have detector length")
        probe_rows = jnp.broadcast_to(probes, (n_scan, n_detector))
    elif probes.shape == (n_scan, n_detector):
        probe_rows = probes
    else:
        raise ValueError("input_probe must be 1D or have one row per scan")
    if kernel.shape != (n_detector,):
        raise ValueError("propagation_kernel must have detector length")
    probes_host = np.asarray(probe_rows)
    kernel_host = np.asarray(kernel)
    if not np.issubdtype(probes_host.dtype, np.inexact):
        raise TypeError("input_probe must use a floating or complex dtype")
    if not np.issubdtype(kernel_host.dtype, np.inexact):
        raise TypeError("propagation_kernel must use a floating or complex dtype")
    if not np.all(np.isfinite(probes_host)):
        raise ValueError("input_probe must contain only finite values")
    if not np.all(np.isfinite(kernel_host)):
        raise ValueError("propagation_kernel must contain only finite values")
    incident_norm_host = n_detector * np.sum(
        np.abs(probes_host) ** 2, axis=1
    )
    if np.any(~np.isfinite(incident_norm_host)) or np.any(incident_norm_host <= 0.0):
        raise ValueError("every input probe must have finite positive incident norm")
    if fit_indices is None:
        fit_indices = reconstruction.metadata.get("training_indices")
    if audit_indices is None:
        audit_indices = reconstruction.metadata.get("audit_indices")
    fit = _indices(fit_indices, n_scan=n_scan, name="fit_indices")
    if fit is None:
        raise ValueError("fit_indices are required and cannot fall back to all scans")
    audit = _indices(audit_indices, n_scan=n_scan, name="audit_indices")
    if audit is not None and np.intersect1d(fit, audit).size:
        raise ValueError("fit_indices and audit_indices must be disjoint")

    if detector_mask is None:
        mask = jnp.ones((n_scan, n_detector), dtype=bool)
    else:
        mask_host = np.asarray(detector_mask)
        if mask_host.shape == (n_detector,):
            mask_host = np.broadcast_to(mask_host, (n_scan, n_detector))
        if mask_host.shape != (n_scan, n_detector) or mask_host.dtype != bool:
            raise TypeError("detector_mask has incompatible shape or dtype")
        mask = jnp.asarray(mask_host)

    def decode(parameters: Array) -> tuple[Array, Array, Array, Array]:
        offset = 0
        vacancy_values = parameters[offset : offset + n_site] * vacancy_scale
        offset += n_site
        axial_coefficients = parameters[offset : offset + rank] * displacement_scale
        offset += rank
        transverse_coefficients = (
            parameters[offset : offset + rank] * displacement_scale
        )
        offset += rank
        rigid_values = parameters[offset : offset + 2] * displacement_scale
        controls_flat = jnp.stack(
            [
                control_basis @ axial_coefficients,
                control_basis @ transverse_coefficients,
            ],
            axis=1,
        )
        controls = controls_flat.reshape(
            len(model.control_coordinates_s),
            len(model.control_coordinates_u),
            2,
        )
        residual_sites = jnp.stack(
            [
                jnp.asarray(basis.site_basis) @ axial_coefficients,
                jnp.asarray(basis.site_basis) @ transverse_coefficients,
            ],
            axis=1,
        )
        return vacancy_values, controls, rigid_values, residual_sites

    def physical_outputs(parameters: Array) -> Array:
        vacancy_values, _, rigid_values, residual_sites = decode(parameters)
        total_sites = residual_sites + rigid_values
        return jnp.concatenate([vacancy_values, total_sites.reshape(-1)])

    output_jacobian = np.asarray(jax.jacfwd(physical_outputs)(x0_device))

    def observable(parameters: Array, indices: Array) -> Array:
        vacancy_values, controls, rigid_values, _ = decode(parameters)
        potential = render_lattice_site_potential_1d(
            model, vacancy_values, controls + rigid_values
        )
        batch_probes = probe_rows[indices]
        intensities = simulate_glancing_scan_1d(
            potential,
            batch_probes,
            starts[indices],
            window_length,
            kernel,
            slice_thickness,
            energy,
            rematerialize=options.rematerialize,
        )
        incident_norm = n_detector * jnp.sum(
            jnp.abs(batch_probes) ** 2, axis=1, keepdims=True
        )
        signal = dose * intensities / incident_norm
        expected = jnp.maximum(signal + background, floor)
        return jnp.where(mask[indices], 2.0 * jnp.sqrt(expected), 0.0).reshape(-1)

    from scipy.stats import chi2

    chi_square_radius = float(
        chi2.ppf(options.displacement_confidence, df=2)
    )

    def evaluate_split(indices: np.ndarray, role: str) -> SiteObservabilitySplit1D:
        jacobian = np.asarray(
            jax.jacfwd(lambda values: observable(values, jnp.asarray(indices)))(
                x0_device
            )
        )
        covariance, estimable, effective_rank = (
            marginal_covariance_from_jacobian_1d(
                jacobian,
                output_jacobian,
                rank_rtol=options.fisher_rank_rtol,
            )
        )
        vacancy_variance = np.diag(covariance)[:n_site]
        vacancy_error = np.sqrt(vacancy_variance)
        vacancy_error[~estimable[:n_site]] = np.inf
        occupied_boundary = options.vacancy_threshold - options.vacancy_margin
        vacant_boundary = options.vacancy_threshold + options.vacancy_margin
        occupied = vacancies < occupied_boundary
        vacant = vacancies > vacant_boundary
        distance_to_decision = np.where(
            occupied,
            occupied_boundary - vacancies,
            np.where(vacant, vacancies - vacant_boundary, 0.0),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            vacancy_z = distance_to_decision / vacancy_error
        vacancy_z[~np.isfinite(vacancy_z)] = 0.0
        displacement_covariance = np.full((n_site, 2, 2), np.inf)
        displacement_radius = np.full(n_site, np.inf)
        displacement_estimable = np.zeros(n_site, dtype=bool)
        for site_index in range(n_site):
            output_indices = n_site + np.asarray(
                [2 * site_index, 2 * site_index + 1]
            )
            if np.all(estimable[output_indices]):
                block = covariance[np.ix_(output_indices, output_indices)]
                if np.all(np.isfinite(block)):
                    block = 0.5 * (block + block.T)
                    eigenvalues = np.linalg.eigvalsh(block)
                    if np.min(eigenvalues) >= -1e-10 * max(
                        np.max(eigenvalues), 1.0
                    ):
                        displacement_covariance[site_index] = block
                        displacement_radius[site_index] = np.sqrt(
                            chi_square_radius * max(float(np.max(eigenvalues)), 0.0)
                        )
                        displacement_estimable[site_index] = True
        vacancy_adequate = estimable[:n_site] & (occupied | vacant) & (
            vacancy_z >= options.minimum_vacancy_z
        )
        displacement_adequate = displacement_estimable & (
            displacement_radius <= options.maximum_displacement_radius_A
        )
        site_observable = vacancy_adequate & (
            vacant | (occupied & displacement_adequate)
        )
        finite_block = covariance[np.ix_(estimable, estimable)]
        if finite_block.size:
            symmetry_error = np.linalg.norm(finite_block - finite_block.T) / max(
                np.linalg.norm(finite_block), 1.0
            )
            minimum_covariance_eigenvalue = float(
                np.min(np.linalg.eigvalsh(0.5 * (finite_block + finite_block.T)))
            )
        else:
            symmetry_error = float("inf")
            minimum_covariance_eigenvalue = float("-inf")
        solver_verified = bool(
            effective_rank > 0
            and np.all(np.isfinite(jacobian))
            and symmetry_error <= 100.0 * np.finfo(float).eps
            and minimum_covariance_eigenvalue
            >= -1e-10 * max(float(np.linalg.norm(finite_block, ord=2)), 1.0)
        )
        return SiteObservabilitySplit1D(
            scan_indices=jnp.asarray(indices),
            vacancy_standard_error=jnp.asarray(vacancy_error),
            vacancy_z_to_decision_boundary=jnp.asarray(vacancy_z),
            displacement_covariance_A2=jnp.asarray(displacement_covariance),
            displacement_confidence_radius_A=jnp.asarray(displacement_radius),
            vacancy_information_adequate=jnp.asarray(vacancy_adequate),
            displacement_information_adequate=jnp.asarray(
                displacement_adequate
            ),
            site_observable=jnp.asarray(site_observable),
            physical_output_estimable=jnp.asarray(estimable),
            solver_verified=solver_verified,
            effective_rank=effective_rank,
            metadata={
                "role": role,
                "method": "dense_svd",
                "n_observations": int(jacobian.shape[0]),
                "n_parameters": int(jacobian.shape[1]),
                "fisher_rank_rtol": float(options.fisher_rank_rtol),
                "covariance_symmetry_error": symmetry_error,
                "minimum_covariance_eigenvalue": minimum_covariance_eigenvalue,
            },
        )

    fit_report = evaluate_split(fit, "fit")
    audit_report = evaluate_split(audit, "audit") if audit is not None else None
    if audit_report is None:
        combined_vacancy = np.zeros(n_site, dtype=bool)
        combined_displacement = np.zeros(n_site, dtype=bool)
        combined_sites = np.zeros(n_site, dtype=bool)
    else:
        combined_vacancy = np.asarray(
            fit_report.vacancy_information_adequate
        ) & np.asarray(audit_report.vacancy_information_adequate)
        combined_displacement = np.asarray(
            fit_report.displacement_information_adequate
        ) & np.asarray(audit_report.displacement_information_adequate)
        combined_sites = np.asarray(fit_report.site_observable) & np.asarray(
            audit_report.site_observable
        )
    calibrated = bool(counting_model.calibrated)
    # The current dense reference profiles every specimen parameter but still
    # holds probe, scan, detector, and fixed-exterior nuisance quantities fixed.
    # This fact is derived from the implemented parameterization, not accepted
    # as a caller-supplied Boolean assertion.
    nuisance_scope_complete = False
    suitable = bool(
        audit_report is not None
        and calibrated
        and nuisance_scope_complete
        and fit_report.solver_verified
        and audit_report.solver_verified
    )
    return LatticeSiteObservability1D(
        site_coordinates=sites,
        fit=fit_report,
        audit=audit_report,
        vacancy_information_adequate=jnp.asarray(combined_vacancy),
        displacement_information_adequate=jnp.asarray(combined_displacement),
        site_observable=jnp.asarray(combined_sites),
        ideal_poisson_information=True,
        calibrated_noise=calibrated,
        nuisance_scope_complete=bool(nuisance_scope_complete),
        suitable_for_trust_gate=suitable,
        metadata={
            "method": "dense_profiled_local_plugin_ideal_poisson_fisher",
            "fisher_evaluation": "local_plugin_at_reconstructed_structure",
            "uncertainty_interpretation": (
                "interior_asymptotic_approximation_not_boundary_calibration"
            ),
            "parameterization": "zero_mean_rms_site_displacement_svd_basis",
            "n_parameters": int(x0.size),
            "displacement_basis_rank": rank,
            "displacement_basis_relative_error": (
                basis.relative_reconstruction_error
            ),
            "active_site_translation_scope": (
                "variable_sites_relative_to_fixed_reference"
            ),
            "calibration_id": counting_model.calibration_id,
            "model_conditional": True,
        },
    )


def _digest_arrays_and_metadata_1d(
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        if value.dtype.hasobject:
            raise TypeError(f"cannot hash object-valued array {name!r}")
        header = json.dumps(
            {"name": name, "dtype": value.dtype.str, "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = np.ascontiguousarray(value).tobytes(order="C")
        for chunk in (header, payload):
            digest.update(len(chunk).to_bytes(8, "big"))
            digest.update(chunk)
    encoded = json.dumps(
        dict(metadata),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest()


def _canonical_json_metadata_1d(
    name: str,
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in metadata):
        raise TypeError(f"{name} keys must be strings")

    def converted(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"{name} contains a non-JSON value {value!r}")

    try:
        encoded = json.dumps(
            dict(metadata),
            allow_nan=False,
            default=converted,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be finite and strictly JSON-serializable"
        ) from exc
    return json.loads(encoded)


@dataclass(frozen=True)
class _GaugeFreeSpecimenParameterization1D:
    model: LatticeSiteModel1D
    basis: LatticeDisplacementBasis1D
    x0: Array
    vacancies: Array
    vacancy_scale: float
    displacement_scale: float
    n_site: int
    rank: int

    @property
    def n_parameter(self) -> int:
        return int(self.x0.size)

    @property
    def n_physical_output(self) -> int:
        return 3 * self.n_site

    def decode(self, parameters: Array) -> tuple[Array, Array, Array, Array]:
        offset = 0
        vacancies = parameters[: self.n_site] * self.vacancy_scale
        offset += self.n_site
        axial = parameters[offset : offset + self.rank] * self.displacement_scale
        offset += self.rank
        transverse = (
            parameters[offset : offset + self.rank] * self.displacement_scale
        )
        offset += self.rank
        translation = parameters[offset : offset + 2] * self.displacement_scale
        control_basis = jnp.asarray(self.basis.control_basis)
        flat_controls = jnp.stack(
            [control_basis @ axial, control_basis @ transverse], axis=1
        )
        controls = flat_controls.reshape(
            len(self.model.control_coordinates_s),
            len(self.model.control_coordinates_u),
            2,
        )
        site_basis = jnp.asarray(self.basis.site_basis)
        residual_sites = jnp.stack(
            [site_basis @ axial, site_basis @ transverse], axis=1
        )
        return vacancies, controls, translation, residual_sites

    def physical_output_rhs(self, output_index: int) -> np.ndarray:
        index = operator.index(output_index)
        if index < 0 or index >= self.n_physical_output:
            raise IndexError("physical output index is out of range")
        result = np.zeros(self.n_parameter, dtype=np.asarray(self.x0).dtype)
        if index < self.n_site:
            result[index] = self.vacancy_scale
            return result
        displacement_index = index - self.n_site
        site_index, component = divmod(displacement_index, 2)
        basis_row = np.asarray(self.basis.site_basis)[site_index]
        coefficient_offset = self.n_site + component * self.rank
        result[coefficient_offset : coefficient_offset + self.rank] = (
            self.displacement_scale * basis_row
        )
        result[self.n_site + 2 * self.rank + component] = self.displacement_scale
        return result

    def physical_output_jvp(self, parameter_direction: Any) -> np.ndarray:
        """Apply the physical-output Jacobian without materializing it."""
        direction = np.asarray(parameter_direction)
        if direction.shape != (self.n_parameter,) or not np.issubdtype(
            direction.dtype, np.floating
        ):
            raise TypeError(
                "parameter_direction must be a floating parameter vector"
            )
        direction = direction.astype(
            np.result_type(direction.dtype, np.asarray(self.x0).dtype),
            copy=False,
        )
        offset = self.n_site
        axial = direction[offset : offset + self.rank]
        offset += self.rank
        transverse = direction[offset : offset + self.rank]
        offset += self.rank
        translation = direction[offset : offset + 2]
        site_basis = np.asarray(self.basis.site_basis)
        displacement = np.column_stack(
            [site_basis @ axial, site_basis @ transverse]
        )
        displacement += translation[None, :]
        return np.concatenate(
            [
                self.vacancy_scale * direction[: self.n_site],
                (self.displacement_scale * displacement).reshape(-1),
            ]
        )

    def physical_output_vjp(self, output_cotangent: Any) -> np.ndarray:
        """Apply the transpose physical-output Jacobian matrix-free."""
        cotangent = np.asarray(output_cotangent)
        if cotangent.shape != (self.n_physical_output,) or not np.issubdtype(
            cotangent.dtype, np.floating
        ):
            raise TypeError(
                "output_cotangent must be a floating physical-output vector"
            )
        cotangent = cotangent.astype(
            np.result_type(cotangent.dtype, np.asarray(self.x0).dtype),
            copy=False,
        )
        result = np.zeros(self.n_parameter, dtype=cotangent.dtype)
        result[: self.n_site] = self.vacancy_scale * cotangent[: self.n_site]
        displacement = cotangent[self.n_site :].reshape(self.n_site, 2)
        site_basis = np.asarray(self.basis.site_basis)
        coefficient_offset = self.n_site
        result[coefficient_offset : coefficient_offset + self.rank] = (
            self.displacement_scale * site_basis.T @ displacement[:, 0]
        )
        coefficient_offset += self.rank
        result[coefficient_offset : coefficient_offset + self.rank] = (
            self.displacement_scale * site_basis.T @ displacement[:, 1]
        )
        result[self.n_site + 2 * self.rank :] = (
            self.displacement_scale * np.sum(displacement, axis=0)
        )
        return result

    def physical_output_row_norm_squared(self) -> np.ndarray:
        """Return ``diag(B B.T)`` for the physical-output Jacobian ``B``."""
        vacancy_norms = np.full(
            self.n_site,
            self.vacancy_scale**2,
            dtype=np.asarray(self.x0).dtype,
        )
        basis_norms = np.sum(
            np.asarray(self.basis.site_basis) ** 2,
            axis=1,
        )
        displacement_norms = (
            self.displacement_scale**2 * (basis_norms + 1.0)
        )
        return np.concatenate(
            [
                vacancy_norms,
                np.repeat(displacement_norms, 2),
            ]
        )

    def physical_output_jacobian(self) -> np.ndarray:
        return np.stack(
            [
                self.physical_output_rhs(index)
                for index in range(self.n_physical_output)
            ]
        )


def _gauge_free_specimen_parameterization_1d(
    model: LatticeSiteModel1D,
    reconstruction: LatticeSiteReconstruction1D,
    options: MatrixFreeObservabilityOptions1D,
) -> _GaugeFreeSpecimenParameterization1D:
    vacancies_device, sites_device, displaced_device = (
        _validated_reconstruction_site_state_1d(model, reconstruction)
    )
    sites = np.asarray(sites_device)
    vacancies = np.asarray(vacancies_device)
    total_displacement = np.asarray(displaced_device) - sites
    translation = np.mean(total_displacement, axis=0)
    residual = total_displacement - translation
    basis = lattice_displacement_basis_1d(
        model, rtol=options.control_basis_rtol
    )
    site_basis = np.asarray(basis.site_basis)
    n_site = len(sites)
    coefficients = np.column_stack(
        [
            site_basis.T @ residual[:, component] / n_site
            for component in range(2)
        ]
    )
    represented = site_basis @ coefficients
    representation_error = np.linalg.norm(represented - residual) / max(
        np.linalg.norm(residual), 1.0
    )
    coordinate_dtype = sites.dtype
    tolerance = max(
        float(options.control_basis_rtol),
        8.0 * float(np.finfo(coordinate_dtype).eps),
    )
    if (
        not np.isfinite(representation_error)
        or representation_error > 20.0 * tolerance
    ):
        raise ValueError(
            "reconstructed displacement is outside the prepared control-field basis"
        )
    vacancy_scale = float(options.vacancy_parameter_scale)
    displacement_scale = float(options.displacement_parameter_scale_A)
    x0 = np.concatenate(
        [
            vacancies / vacancy_scale,
            coefficients[:, 0] / displacement_scale,
            coefficients[:, 1] / displacement_scale,
            translation / displacement_scale,
        ]
    )
    return _GaugeFreeSpecimenParameterization1D(
        model=model,
        basis=basis,
        x0=jnp.asarray(x0, dtype=model.reference_potential.dtype),
        vacancies=jnp.asarray(vacancies, dtype=model.reference_potential.dtype),
        vacancy_scale=vacancy_scale,
        displacement_scale=displacement_scale,
        n_site=n_site,
        rank=basis.numerical_rank,
    )


def _validated_selected_sites_1d(
    site_indices: Sequence[int] | None,
    n_site: int,
    maximum_selected_sites: int,
) -> np.ndarray:
    if site_indices is None:
        indices = np.arange(n_site, dtype=np.int32)
        if indices.size > maximum_selected_sites:
            raise ValueError(
                "all candidate sites exceed options.maximum_selected_sites; "
                "provide an explicit bounded site_indices subset"
            )
        return indices
    indices = np.asarray(site_indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("site_indices must be a one-dimensional integer sequence")
    indices = indices.astype(np.int32, copy=False)
    if (
        not indices.size
        or np.unique(indices).size != indices.size
        or np.any(indices < 0)
        or np.any(indices >= n_site)
    ):
        raise ValueError("site_indices must contain unique valid site indices")
    if indices.size > maximum_selected_sites:
        raise ValueError("site_indices exceed options.maximum_selected_sites")
    return indices


def _validated_nuisance_tangent_1d(
    profile: WhitenedNuisanceProfile1D | Any | None,
    *,
    n_scan: int,
    n_detector: int,
    maximum_columns: int,
) -> tuple[np.ndarray, tuple[str, ...], str, Mapping[str, Any]]:
    if profile is None:
        tangent = np.empty((n_scan, n_detector, 0), dtype=float)
        names: tuple[str, ...] = ()
        profile_id = "none"
        metadata: Mapping[str, Any] = {}
    else:
        if isinstance(profile, WhitenedNuisanceProfile1D):
            tangent_value = profile.tangent_matrix
            names = tuple(profile.parameter_names)
            profile_id = profile.profile_id
            metadata = _canonical_json_metadata_1d(
                "nuisance_profile.metadata", profile.metadata
            )
        else:
            tangent_value = profile
            names = ()
            profile_id = "anonymous_explicit_whitened_tangent"
            metadata = {}
        tangent = np.asarray(tangent_value)
        if tangent.ndim == 2 and tangent.shape[0] == n_scan * n_detector:
            tangent = tangent.reshape(n_scan, n_detector, tangent.shape[1])
        if tangent.ndim != 3 or tangent.shape[:2] != (n_scan, n_detector):
            raise ValueError(
                "whitened nuisance tangent must have shape "
                "(n_scan, n_detector, n_nuisance)"
            )
        if not np.issubdtype(tangent.dtype, np.floating):
            raise TypeError("whitened nuisance tangent must use a floating dtype")
        if np.any(~np.isfinite(tangent)):
            raise ValueError("whitened nuisance tangent must be finite")
    n_column = tangent.shape[2]
    if n_column > maximum_columns:
        raise ValueError(
            f"nuisance tangent has {n_column} columns, above the configured maximum"
        )
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("nuisance profile_id must be a non-empty string")
    if names:
        if len(names) != n_column or len(set(names)) != len(names):
            raise ValueError(
                "nuisance parameter_names must be unique and match the columns"
            )
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("nuisance parameter names must be non-empty strings")
    else:
        names = tuple(f"nuisance_{index}" for index in range(n_column))
    return tangent, names, profile_id, metadata


def _orthonormal_nuisance_basis_1d(
    tangent: np.ndarray,
    detector_mask: np.ndarray,
    indices: np.ndarray,
    *,
    rank_rtol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if tangent.shape[2] == 0:
        return (
            np.empty((len(indices) * tangent.shape[1], 0), dtype=float),
            np.empty(0),
        )
    selected = np.asarray(tangent[indices], dtype=float).reshape(-1, tangent.shape[2])
    selected = selected * detector_mask[indices].reshape(-1, 1)
    left, singular_values, _ = np.linalg.svd(selected, full_matrices=False)
    scale = singular_values[0] if singular_values.size else 0.0
    rank = int(np.count_nonzero(singular_values > rank_rtol * scale))
    return left[:, :rank], singular_values[:rank]


def _projector_checks_1d(basis: np.ndarray) -> dict[str, float]:
    if basis.shape[1] == 0:
        return {
            "orthogonality_error": 0.0,
            "idempotence_error": 0.0,
            "symmetry_error": 0.0,
            "annihilation_error": 0.0,
        }
    identity = np.eye(basis.shape[1])
    orthogonality = np.linalg.norm(basis.T @ basis - identity) / max(
        np.linalg.norm(identity), 1.0
    )
    row = np.arange(1, basis.shape[0] + 1, dtype=float)
    first = np.sin(row)
    second = np.cos(np.sqrt(2.0) * row)

    def projected(value: np.ndarray) -> np.ndarray:
        return value - basis @ (basis.T @ value)

    projected_first = projected(first)
    idempotence = np.linalg.norm(projected(projected_first) - projected_first) / max(
        np.linalg.norm(projected_first), 1.0
    )
    symmetry = abs(
        float(np.dot(first, projected(second)) - np.dot(projected(first), second))
    ) / max(np.linalg.norm(first) * np.linalg.norm(second), 1.0)
    annihilation = np.linalg.norm(basis.T @ projected_first) / max(
        np.linalg.norm(first), 1.0
    )
    return {
        "orthogonality_error": float(orthogonality),
        "idempotence_error": float(idempotence),
        "symmetry_error": float(symmetry),
        "annihilation_error": float(annihilation),
    }


def _pcg_metadata_1d(
    output_indices: Sequence[int],
    diagnostics: Sequence[PCGSolveDiagnostics1D],
) -> dict[str, Any]:
    return {
        "physical_output_indices": [int(value) for value in output_indices],
        "converged": [bool(value.converged) for value in diagnostics],
        "iterations": [int(value.iterations) for value in diagnostics],
        "stop_reasons": [value.stop_reason for value in diagnostics],
        "relative_true_residuals": [
            float(value.relative_residual) for value in diagnostics
        ],
        "breakdown": [bool(value.breakdown) for value in diagnostics],
        "stagnated": [bool(value.stagnated) for value in diagnostics],
        "residual_norm_histories": [
            np.asarray(value.residual_norm_history).tolist() for value in diagnostics
        ],
        "curvature_histories": [
            np.asarray(value.curvature_history).tolist() for value in diagnostics
        ],
    }


def _canonical_prepared_poisson_counting_model_1d(
    prepared: PreparedLatticeSiteReconstruction1D,
) -> PoissonCountingModel1D:
    """Derive the scalar ideal-Poisson model represented by a prepared loss.

    The current matrix-free operator has scalar dose and background fields.
    It must therefore fail closed rather than average a per-scan dose or a
    detector-dependent dark calibration that it cannot represent exactly.
    """
    measurement = prepared.measurement
    objective = prepared.objective
    if measurement is None or objective is None:
        if measurement is not None or objective is not None:
            raise ValueError(
                "prepared measurement and objective must either both be present "
                "or both be absent"
            )
        raise ValueError(
            "prepared reconstruction has no calibrated Poisson objective; "
            "supply an external hypothetical counting_model"
        )
    if objective.kind != "poisson_deviance":
        raise ValueError(
            "ideal-Poisson observability requires objective.kind="
            "'poisson_deviance'; Gaussian/read-noise objectives require a "
            "separate objective-aware Fisher operator"
        )

    shape = tuple(int(value) for value in prepared.measured_intensities.shape)
    valid = np.asarray(measurement.valid_mask)
    if valid.dtype != np.bool_ or valid.shape != shape:
        raise ValueError(
            "prepared calibrated measurement must have a matching boolean valid mask"
        )
    if prepared.detector_valid_mask is None or not np.array_equal(
        np.asarray(prepared.detector_valid_mask, dtype=bool), valid
    ):
        raise ValueError(
            "prepared detector_valid_mask does not match measurement.valid_mask"
        )
    if not np.any(valid):
        raise ValueError("prepared calibrated measurement has no valid pixels")

    read_noise = np.asarray(
        measurement.calibrated_read_noise_std_electrons, dtype=float
    )
    if read_noise.shape != shape:
        raise ValueError(
            "prepared calibrated read-noise array has incompatible shape"
        )
    selected_read_noise = read_noise[valid]
    if np.any(~np.isfinite(selected_read_noise)) or np.any(
        selected_read_noise != 0.0
    ):
        raise ValueError(
            "ideal-Poisson observability requires exactly zero declared read "
            "noise at every valid pixel"
        )

    dose = np.asarray(objective.electrons_per_pattern, dtype=float)
    if dose.ndim == 0:
        dose = np.full(shape[0], float(dose), dtype=float)
    if dose.shape != (shape[0],):
        raise ValueError(
            "prepared objective dose must be scalar or have one value per scan"
        )
    effective_dose = dose * float(objective.relative_signal_scale)
    if np.any(~np.isfinite(effective_dose)) or np.any(effective_dose <= 0.0):
        raise ValueError("prepared effective Poisson dose must be finite and positive")
    if not np.array_equal(
        effective_dose,
        np.full(effective_dose.shape, effective_dose[0], dtype=float),
    ):
        raise ValueError(
            "the scalar ideal-Poisson operator cannot represent nonconstant "
            "per-scan effective dose"
        )

    dark = np.asarray(
        measurement.calibrated_dark_electrons_per_pixel, dtype=float
    )
    if dark.shape != shape:
        raise ValueError("prepared calibrated dark array has incompatible shape")
    selected_dark = dark[valid]
    if np.any(~np.isfinite(selected_dark)) or np.any(selected_dark < 0.0):
        raise ValueError(
            "prepared calibrated dark must be finite and non-negative at valid pixels"
        )
    if not np.array_equal(
        selected_dark,
        np.full(selected_dark.shape, selected_dark[0], dtype=float),
    ):
        raise ValueError(
            "the scalar ideal-Poisson operator cannot represent nonconstant "
            "valid-pixel calibrated dark"
        )

    model = PoissonCountingModel1D(
        electrons_per_pattern=float(effective_dose[0]),
        background_electrons_per_pixel=float(selected_dark[0]),
        minimum_expected_electrons=float(objective.minimum_expected_electrons),
        calibrated=True,
        calibration_id=measurement.calibration_id,
    )
    validate_poisson_counting_model_1d(model)
    return model


def poisson_counting_model_from_prepared_1d(
    prepared: PreparedLatticeSiteReconstruction1D,
) -> PoissonCountingModel1D:
    """Return the exact scalar Poisson model encoded by a prepared objective.

    This helper rejects amplitude, Gaussian/read-noise, per-scan-dose, and
    spatially varying-dark problems that the scalar ideal-Poisson operator
    cannot represent without changing their scientific contract.
    """
    if not isinstance(prepared, PreparedLatticeSiteReconstruction1D):
        raise TypeError(
            "prepared must be a PreparedLatticeSiteReconstruction1D instance"
        )
    _validate_prepared_static_contract_1d(prepared)
    return _canonical_prepared_poisson_counting_model_1d(prepared)


def _counting_model_mismatches_1d(
    supplied: PoissonCountingModel1D,
    canonical: PoissonCountingModel1D,
) -> list[str]:
    fields = (
        "electrons_per_pattern",
        "background_electrons_per_pixel",
        "minimum_expected_electrons",
        "calibrated",
        "calibration_id",
    )
    return [
        name
        for name in fields
        if getattr(supplied, name) != getattr(canonical, name)
    ]


def _validate_reconstruction_renderer_state_1d(
    model: LatticeSiteModel1D,
    reconstruction: LatticeSiteReconstruction1D,
) -> None:
    """Prove that stored coordinates and potential describe one rendered state."""
    vacancies, sites, displaced = _validated_reconstruction_site_state_1d(
        model, reconstruction
    )
    controls = _floating_host_array(
        "reconstruction.displacement_controls",
        reconstruction.displacement_controls,
        3,
    )
    rigid = _floating_host_array(
        "reconstruction.rigid_displacement",
        reconstruction.rigid_displacement,
        1,
    )
    control_s = _floating_host_array(
        "reconstruction.control_coordinates_s",
        reconstruction.control_coordinates_s,
        1,
    )
    control_u = _floating_host_array(
        "reconstruction.control_coordinates_u",
        reconstruction.control_coordinates_u,
        1,
    )
    model_control_s = _floating_host_array(
        "model.control_coordinates_s", model.control_coordinates_s, 1
    )
    model_control_u = _floating_host_array(
        "model.control_coordinates_u", model.control_coordinates_u, 1
    )
    if controls.shape != (len(model_control_s), len(model_control_u), 2):
        raise ValueError(
            "reconstruction displacement_controls do not match the prepared model"
        )
    if rigid.shape != (2,):
        raise ValueError("reconstruction rigid_displacement must have shape (2,)")
    for name, actual, expected in (
        ("control_coordinates_s", control_s, model_control_s),
        ("control_coordinates_u", control_u, model_control_u),
    ):
        if actual.shape != expected.shape:
            raise ValueError(f"reconstruction {name} do not match the prepared model")
        rtol, atol = _coordinate_tolerances(actual, expected)
        if not np.allclose(actual, expected, rtol=rtol, atol=atol):
            raise ValueError(f"reconstruction {name} do not match the prepared model")

    total_controls = controls + rigid[None, None, :]
    rendered_displacements = np.asarray(
        lattice_site_displacements_1d(
            sites,
            total_controls,
            model.control_coordinates_s,
            model.control_coordinates_u,
        )
    )
    expected_displaced = np.asarray(sites) + rendered_displacements
    displaced_host = np.asarray(displaced)
    rtol, atol = _coordinate_tolerances(expected_displaced, displaced_host)
    if not np.allclose(expected_displaced, displaced_host, rtol=rtol, atol=atol):
        raise ValueError(
            "reconstruction displaced_site_coordinates are inconsistent with "
            "displacement_controls and rigid_displacement"
        )

    stored_potential = _floating_host_array(
        "reconstruction.potential", reconstruction.potential, 2
    )
    rendered_potential = np.asarray(
        render_lattice_site_potential_1d(model, vacancies, total_controls)
    )
    if stored_potential.shape != rendered_potential.shape:
        raise ValueError(
            "reconstruction potential shape does not match the prepared renderer"
        )
    rtol, atol = _coordinate_tolerances(stored_potential, rendered_potential)
    if not np.allclose(
        stored_potential, rendered_potential, rtol=rtol, atol=atol
    ):
        raise ValueError(
            "reconstruction potential is inconsistent with its lattice-site "
            "parameters"
        )


def _prepared_nuisance_state_digest_1d(
    prepared: PreparedLatticeSiteReconstruction1D,
    reconstruction: LatticeSiteReconstruction1D,
) -> str:
    return _digest_arrays_and_metadata_1d(
        {
            "potential": reconstruction.potential,
            "vacancy_fractions": reconstruction.vacancy_fractions,
            "displacement_controls": reconstruction.displacement_controls,
            "rigid_displacement": reconstruction.rigid_displacement,
            "site_coordinates": reconstruction.site_coordinates,
            "displaced_site_coordinates": (
                reconstruction.displaced_site_coordinates
            ),
        },
        {
            "reconstruction_problem_id": prepared.reconstruction_problem_id,
            "best_update": int(reconstruction.best_update),
            "completed_updates": int(reconstruction.completed_updates),
            "seed": reconstruction.metadata.get("seed"),
        },
    )


def _prepared_nuisance_mask_digest_1d(detector_mask: np.ndarray) -> str:
    return _digest_arrays_and_metadata_1d(
        {"detector_valid_mask": detector_mask},
        {"mode": "prepared_explicit"},
    )


def _prepared_nuisance_counting_digest_1d(
    counting_model: PoissonCountingModel1D,
) -> str:
    return _digest_arrays_and_metadata_1d(
        {},
        {
            "electrons_per_pattern": float(
                counting_model.electrons_per_pattern
            ),
            "background_electrons_per_pixel": float(
                counting_model.background_electrons_per_pixel
            ),
            "minimum_expected_electrons": float(
                counting_model.minimum_expected_electrons
            ),
            "calibrated": bool(counting_model.calibrated),
            "calibration_id": counting_model.calibration_id,
        },
    )


def _prepared_nuisance_coverage_1d(
    options: PreparedNuisanceOptions1D,
) -> tuple[dict[str, dict[str, bool]], list[str]]:
    coverage = {
        "scan_geometry": {
            "common_relative_axial_origin_shift": bool(
                options.include_scan_origin_shift
            ),
        },
        "probe": {
            "common_transverse_shift": bool(
                options.include_probe_transverse_shift
            ),
            "common_tilt": bool(options.include_probe_tilt),
            "common_log_width": bool(options.include_probe_log_width),
        },
        "detector_calibration": {
            "common_reciprocal_frequency_offset": bool(
                options.include_detector_frequency_offset
            ),
            "common_log_signal_gain": bool(
                options.include_detector_log_gain
            ),
            "common_dark_offset": bool(
                options.include_detector_dark_offset
            ),
        },
    }
    missing = [
        "per_scan_position_jitter",
        "partial_coherence",
        "probe_defocus_and_higher_aberrations",
        "detector_nonlinearity_point_spread_and_frequency_scale",
        "fixed_exterior_material",
        "forward_model_discrepancy",
    ]
    for group, directions in coverage.items():
        for direction, represented in directions.items():
            if not represented:
                missing.append(f"{group}.{direction}")
    return coverage, missing


def _linearized_probe_log_width_1d(
    rows: Array,
    transverse_coordinates: Array,
    transverse_sampling: float,
    log_width: Array,
) -> Array:
    """Apply a symmetric local dilation without differentiating at interp knots.

    For ``f(exp(-a) * u)``, the derivative at ``a=0`` is ``-u*f'(u)``.
    A centered difference of the zero-extended sampled probe defines that
    derivative symmetrically and avoids JAX's arbitrary one-sided derivative
    of piecewise-linear interpolation exactly at every source knot.
    """
    padded = jnp.pad(rows, ((0, 0), (1, 1)))
    derivative = (padded[:, 2:] - padded[:, :-2]) / (
        2.0 * transverse_sampling
    )
    direction = -transverse_coordinates[None, :] * derivative
    return rows + log_width * direction


def prepared_whitened_nuisance_profile_1d(
    prepared: PreparedLatticeSiteReconstruction1D,
    reconstruction: LatticeSiteReconstruction1D,
    *,
    options: PreparedNuisanceOptions1D | None = None,
) -> WhitenedNuisanceProfile1D:
    """Construct calibration-bound low-rank nuisance tangents by autodiff.

    The returned columns use the same ``2*sqrt(expected electrons)`` observable
    as the ideal-Poisson Fisher operator. A global scan-origin shift translates
    the complete rendered specimen, not only its active sites. Probe shift,
    tilt, and width preserve incident norm; detector gain, dark offset, and
    reciprocal-frequency offset act after propagation. The profile remains a
    bounded local nuisance model rather than a claim of complete coverage.
    """
    options = _validated_prepared_nuisance_options(options)
    if not isinstance(prepared, PreparedLatticeSiteReconstruction1D):
        raise TypeError(
            "prepared must be a PreparedLatticeSiteReconstruction1D instance"
        )
    _validate_prepared_static_contract_1d(prepared)
    if not isinstance(reconstruction, LatticeSiteReconstruction1D):
        raise TypeError(
            "reconstruction must be a LatticeSiteReconstruction1D instance"
        )
    if not isinstance(reconstruction.metadata, Mapping):
        raise TypeError("reconstruction.metadata must be a mapping")
    for name, expected in (
        ("reconstruction_problem_id", prepared.reconstruction_problem_id),
        ("reconstructor_id", prepared.reconstructor_id),
        ("objective_id", prepared.objective_id),
    ):
        if reconstruction.metadata.get(name) != expected:
            raise ValueError(
                f"reconstruction metadata {name!r} does not match the prepared problem"
            )
    if prepared.similarity_residual_gauge:
        raise ValueError(
            "prepared nuisance profiling does not yet implement the aligned "
            "translation/rotation/dilation residual gauge"
        )
    _validate_reconstruction_renderer_state_1d(
        prepared.model,
        reconstruction,
    )
    counting_model = _canonical_prepared_poisson_counting_model_1d(prepared)
    names = []
    for option_name, parameter_name in _PREPARED_NUISANCE_OPTION_NAME_PAIRS_1D:
        if bool(getattr(options, option_name)):
            names.append(parameter_name)
    parameter_names = tuple(names)
    parameter_indices = {
        name: index for index, name in enumerate(parameter_names)
    }

    potential = jnp.asarray(reconstruction.potential)
    probe_rows = jnp.asarray(prepared.probe_rows)
    starts = jnp.asarray(prepared.window_starts)
    kernel = jnp.asarray(prepared.propagation_kernel)
    n_scan, n_detector = prepared.measured_intensities.shape
    detector_mask = (
        np.ones((n_scan, n_detector), dtype=bool)
        if prepared.detector_valid_mask is None
        else np.asarray(prepared.detector_valid_mask, dtype=bool)
    )
    mask_device = jnp.asarray(detector_mask)
    real_dtype = potential.dtype
    transverse_sampling = float(prepared.model.transverse_sampling)
    axial_sampling = float(prepared.model.axial_sampling)
    if not np.isfinite(transverse_sampling) or transverse_sampling <= 0.0:
        raise ValueError("prepared transverse sampling must be finite and positive")
    if not np.isfinite(axial_sampling) or axial_sampling <= 0.0:
        raise ValueError("prepared axial sampling must be finite and positive")
    detector_frequency_sampling = 1.0 / (
        n_detector * transverse_sampling
    )

    transverse_coordinates = (
        jnp.arange(n_detector, dtype=real_dtype)
        - 0.5 * (n_detector - 1)
    ) * transverse_sampling
    probe_frequencies = jnp.fft.fftfreq(
        n_detector,
        d=transverse_sampling,
    ).astype(real_dtype)
    detector_frequencies = jnp.fft.fftfreq(
        n_detector,
        d=detector_frequency_sampling,
    ).astype(real_dtype)
    wavelength = jnp.asarray(
        energy2wavelength(prepared.energy),
        dtype=real_dtype,
    )
    target_probe_norm = jnp.sum(jnp.abs(probe_rows) ** 2, axis=1)
    padded_s = potential.shape[0] * 2
    pad_before = (padded_s - potential.shape[0]) // 2
    pad_after = padded_s - potential.shape[0] - pad_before
    potential_frequencies = jnp.fft.fftfreq(
        padded_s,
        d=axial_sampling,
    ).astype(real_dtype)

    def parameter(parameters: Array, name: str) -> Array:
        index = parameter_indices.get(name)
        return (
            jnp.asarray(0.0, dtype=real_dtype)
            if index is None
            else parameters[index]
        )

    def shifted_potential(parameters: Array) -> Array:
        if not options.include_scan_origin_shift:
            return potential
        shift_A = parameter(parameters, "scan_origin_shift_A")
        padded = jnp.pad(potential, ((pad_before, pad_after), (0, 0)))
        phase = jnp.exp(
            -2j * jnp.pi * potential_frequencies * shift_A
        )[:, None]
        shifted = jnp.fft.ifft(
            jnp.fft.fft(padded, axis=0) * phase,
            axis=0,
        ).real
        return shifted[pad_before : pad_before + potential.shape[0]]

    def transformed_probes(parameters: Array) -> Array:
        rows = probe_rows
        if options.include_probe_log_width:
            rows = _linearized_probe_log_width_1d(
                rows,
                transverse_coordinates,
                transverse_sampling,
                parameter(parameters, "probe_log_width"),
            )
        if options.include_probe_transverse_shift:
            shift_A = parameter(parameters, "probe_transverse_shift_A")
            phase = jnp.exp(
                -2j * jnp.pi * probe_frequencies * shift_A
            )
            rows = jnp.fft.ifft(jnp.fft.fft(rows, axis=1) * phase, axis=1)
        if options.include_probe_tilt:
            tilt = parameter(parameters, "probe_tilt_rad")
            rows = rows * jnp.exp(
                2j
                * jnp.pi
                * transverse_coordinates
                * jnp.sin(tilt)
                / wavelength
            )
        current_norm = jnp.sum(jnp.abs(rows) ** 2, axis=1)
        safe_norm = jnp.maximum(current_norm, jnp.finfo(real_dtype).tiny)
        return rows * jnp.sqrt(target_probe_norm / safe_norm)[:, None]

    dose = jnp.asarray(counting_model.electrons_per_pattern, dtype=real_dtype)
    background = jnp.asarray(
        counting_model.background_electrons_per_pixel,
        dtype=real_dtype,
    )
    floor = jnp.asarray(
        counting_model.minimum_expected_electrons,
        dtype=real_dtype,
    )

    def nuisance_observable(parameters: Array) -> Array:
        probes = transformed_probes(parameters)
        intensities = simulate_glancing_scan_1d(
            shifted_potential(parameters),
            probes,
            starts,
            prepared.window_length,
            kernel,
            prepared.slice_thickness,
            prepared.energy,
            rematerialize=prepared.rematerialize,
        )
        incident_norm = n_detector * jnp.sum(
            jnp.abs(probes) ** 2,
            axis=1,
            keepdims=True,
        )
        signal = dose * intensities / incident_norm
        if options.include_detector_frequency_offset:
            frequency_offset = parameter(
                parameters,
                "detector_frequency_offset_inverse_A",
            )
            detector_phase = jnp.exp(
                -2j * jnp.pi * detector_frequencies * frequency_offset
            )
            signal = jnp.fft.ifft(
                jnp.fft.fft(signal, axis=1) * detector_phase,
                axis=1,
            ).real
        log_gain = parameter(parameters, "detector_log_signal_gain")
        dark_offset = parameter(parameters, "detector_dark_offset_electrons")
        expected = jnp.maximum(
            jnp.exp(log_gain) * signal + background + dark_offset,
            floor,
        )
        return jnp.where(
            mask_device,
            2.0 * jnp.sqrt(expected),
            0.0,
        ).reshape(-1)

    zero = jnp.zeros(len(parameter_names), dtype=real_dtype)

    @jax.jit
    def directional_tangent(direction: Array) -> Array:
        return jax.jvp(nuisance_observable, (zero,), (direction,))[1]

    identity = np.eye(len(parameter_names), dtype=np.asarray(zero).dtype)
    tangent_columns = [
        np.asarray(
            jax.block_until_ready(
                directional_tangent(jnp.asarray(direction, dtype=real_dtype))
            )
        )
        for direction in identity
    ]
    tangent = np.stack(tangent_columns, axis=-1).reshape(
        n_scan,
        n_detector,
        len(parameter_names),
    )
    if np.any(~np.isfinite(tangent)):
        raise FloatingPointError("prepared nuisance tangent is non-finite")
    if np.any(tangent[~detector_mask] != 0.0):
        raise RuntimeError("masked detector pixels have nonzero nuisance tangent")
    options_metadata = {
        name: bool(getattr(options, name))
        for name, _ in _PREPARED_NUISANCE_OPTION_NAME_PAIRS_1D
    }
    reconstruction_state_digest = _prepared_nuisance_state_digest_1d(
        prepared, reconstruction
    )
    detector_mask_digest = _prepared_nuisance_mask_digest_1d(detector_mask)
    counting_digest = _prepared_nuisance_counting_digest_1d(counting_model)
    coverage, missing_nuisance_scopes = _prepared_nuisance_coverage_1d(
        options
    )
    metadata = {
        "constructor_contract": _PREPARED_NUISANCE_CONTRACT_1D,
        "reconstruction_problem_id": prepared.reconstruction_problem_id,
        "reconstructor_id": prepared.reconstructor_id,
        "objective_id": prepared.objective_id,
        "calibration_id": counting_model.calibration_id,
        "counting_contract_sha256": counting_digest,
        "detector_mask_sha256": detector_mask_digest,
        "reconstruction_state_sha256": reconstruction_state_digest,
        "parameter_names": list(parameter_names),
        "options": options_metadata,
        "observable": "two_sqrt_expected_electrons",
        "derivative_method": "jax_jvp_at_reconstructed_state",
        "probe_log_width_linearization": (
            "zero_extended_centered_spatial_difference"
            if options.include_probe_log_width
            else "not_included"
        ),
        "complete_specimen_shift": bool(options.include_scan_origin_shift),
        "probe_incident_norm_preserved": True,
        "nuisance_scope_complete": False,
        "coverage": coverage,
        "missing_nuisance_scopes": missing_nuisance_scopes,
        "nuisance_prior": "unconstrained_local_profile_span",
        "tangent_dtype": str(tangent.dtype),
        "jax_backend": jax.default_backend(),
        "jax_devices": sorted(
            str(device) for device in prepared.model.reference_potential.devices()
        ),
        "training_indices": np.asarray(
            prepared.training_indices, dtype=int
        ).tolist(),
        "validation_indices": np.asarray(
            prepared.validation_indices, dtype=int
        ).tolist(),
        "audit_indices": np.asarray(
            prepared.audit_indices, dtype=int
        ).tolist(),
        "excluded_indices": np.asarray(
            prepared.excluded_indices, dtype=int
        ).tolist(),
    }
    profile_digest = _digest_arrays_and_metadata_1d(
        {"whitened_tangent": tangent},
        metadata,
    )
    return WhitenedNuisanceProfile1D(
        tangent_matrix=jnp.asarray(tangent),
        parameter_names=parameter_names,
        profile_id=f"prepared-poisson-nuisance-{profile_digest}",
        metadata=metadata,
    )


def _validate_generated_prepared_nuisance_profile_1d(
    *,
    tangent: np.ndarray,
    parameter_names: tuple[str, ...],
    profile_id: str,
    metadata: Mapping[str, Any],
    prepared: PreparedLatticeSiteReconstruction1D,
    reconstruction: LatticeSiteReconstruction1D,
    counting_model: PoissonCountingModel1D,
    detector_mask: np.ndarray,
) -> tuple[list[str], Mapping[str, Any]]:
    contract = metadata.get("constructor_contract")
    if contract != _PREPARED_NUISANCE_CONTRACT_1D:
        raise ValueError("generated nuisance profile contract is unsupported")
    required = {
        "reconstruction_problem_id",
        "reconstructor_id",
        "objective_id",
        "calibration_id",
        "counting_contract_sha256",
        "detector_mask_sha256",
        "reconstruction_state_sha256",
        "parameter_names",
        "options",
        "observable",
        "derivative_method",
        "probe_log_width_linearization",
        "complete_specimen_shift",
        "probe_incident_norm_preserved",
        "nuisance_scope_complete",
        "coverage",
        "missing_nuisance_scopes",
        "nuisance_prior",
        "tangent_dtype",
        "training_indices",
        "validation_indices",
        "audit_indices",
        "excluded_indices",
    }
    missing_keys = sorted(required.difference(metadata))
    if missing_keys:
        raise ValueError(
            "generated nuisance metadata omits required field(s): "
            + ", ".join(missing_keys)
        )

    exact_values = {
        "reconstruction_problem_id": prepared.reconstruction_problem_id,
        "reconstructor_id": prepared.reconstructor_id,
        "objective_id": prepared.objective_id,
        "calibration_id": counting_model.calibration_id,
        "counting_contract_sha256": (
            _prepared_nuisance_counting_digest_1d(counting_model)
        ),
        "detector_mask_sha256": _prepared_nuisance_mask_digest_1d(
            detector_mask
        ),
        "reconstruction_state_sha256": _prepared_nuisance_state_digest_1d(
            prepared, reconstruction
        ),
        "parameter_names": list(parameter_names),
        "observable": "two_sqrt_expected_electrons",
        "derivative_method": "jax_jvp_at_reconstructed_state",
        "probe_incident_norm_preserved": True,
        "nuisance_scope_complete": False,
        "nuisance_prior": "unconstrained_local_profile_span",
        "tangent_dtype": str(tangent.dtype),
        "training_indices": np.asarray(
            prepared.training_indices, dtype=int
        ).tolist(),
        "validation_indices": np.asarray(
            prepared.validation_indices, dtype=int
        ).tolist(),
        "audit_indices": np.asarray(
            prepared.audit_indices, dtype=int
        ).tolist(),
        "excluded_indices": np.asarray(
            prepared.excluded_indices, dtype=int
        ).tolist(),
    }
    for name, expected in exact_values.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"generated nuisance profile {name!r} does not match the "
                "prepared reconstruction"
            )
    if metadata.get("nuisance_scope_complete") is not False:
        raise ValueError(
            "generated nuisance profile 'nuisance_scope_complete' must be false"
        )
    if metadata.get("probe_incident_norm_preserved") is not True:
        raise ValueError(
            "generated nuisance profile must preserve the declared probe norm"
        )

    options_value = metadata.get("options")
    if not isinstance(options_value, Mapping):
        raise ValueError("generated nuisance profile options must be a mapping")
    option_names = {
        name for name, _ in _PREPARED_NUISANCE_OPTION_NAME_PAIRS_1D
    }
    if set(options_value) != option_names or any(
        not isinstance(options_value[name], bool) for name in option_names
    ):
        raise ValueError(
            "generated nuisance profile options are not the canonical boolean set"
        )
    nuisance_options = _validated_prepared_nuisance_options(
        PreparedNuisanceOptions1D(**dict(options_value))
    )
    if metadata.get("complete_specimen_shift") is not bool(
        nuisance_options.include_scan_origin_shift
    ):
        raise ValueError(
            "generated nuisance complete-specimen-shift flag does not match "
            "its options"
        )
    expected_width_linearization = (
        "zero_extended_centered_spatial_difference"
        if nuisance_options.include_probe_log_width
        else "not_included"
    )
    if metadata.get("probe_log_width_linearization") != (
        expected_width_linearization
    ):
        raise ValueError(
            "generated nuisance probe-width linearization does not match its "
            "options"
        )
    expected_names = tuple(
        parameter_name
        for option_name, parameter_name in (
            _PREPARED_NUISANCE_OPTION_NAME_PAIRS_1D
        )
        if bool(getattr(nuisance_options, option_name))
    )
    if parameter_names != expected_names:
        raise ValueError(
            "generated nuisance parameter names do not match its options"
        )
    expected_coverage, expected_missing = _prepared_nuisance_coverage_1d(
        nuisance_options
    )
    if metadata.get("coverage") != expected_coverage:
        raise ValueError(
            "generated nuisance profile coverage does not match its options"
        )
    coverage_value = metadata.get("coverage")
    if any(
        not isinstance(value, bool)
        for directions in coverage_value.values()
        for value in directions.values()
    ):
        raise ValueError(
            "generated nuisance profile coverage flags must be booleans"
        )
    if metadata.get("missing_nuisance_scopes") != expected_missing:
        raise ValueError(
            "generated nuisance profile missing-scope list does not match its "
            "options"
        )

    expected_digest = _digest_arrays_and_metadata_1d(
        {"whitened_tangent": tangent}, metadata
    )
    if profile_id != f"prepared-poisson-nuisance-{expected_digest}":
        raise ValueError(
            "generated nuisance profile identifier does not authenticate its "
            "tangent and metadata"
        )
    return expected_missing, expected_coverage


def estimate_prepared_lattice_site_observability_matrix_free_1d(
    prepared: PreparedLatticeSiteReconstruction1D,
    reconstruction: LatticeSiteReconstruction1D,
    counting_model: PoissonCountingModel1D | None = None,
    *,
    nuisance_profile: WhitenedNuisanceProfile1D | Any | None = None,
    site_indices: Sequence[int] | None = None,
    preconditioner_diagonal: Any | None = None,
    options: MatrixFreeObservabilityOptions1D | None = None,
    _stochastic_options: Any | None = None,
) -> LatticeSiteObservability1D | PreparedStochasticObservability1D:
    """Estimate selected-site covariance using a projected Fisher operator.

    The detector Jacobian is never materialized unless ``options.exhaustive``
    is explicitly enabled for a tiny problem.  The optional nuisance tangent
    must already be whitened in the same ``2*sqrt(expected counts)`` observable
    used by the ideal-Poisson Fisher calculation.

    A calibrated prepared Poisson objective is the source of truth for dose,
    dark background, numerical floor, and calibration identity.  Its canonical
    scalar count model is derived automatically; an optional caller-supplied
    model must match it exactly.  Legacy amplitude problems instead require an
    explicit external model and are labelled as hypothetical count analyses.

    This phase-1 implementation profiles only the supplied low-rank tangent.
    Probe, scan, detector-calibration, and exterior-material nuisance scopes are
    therefore incomplete by construction; returned reports are never suitable
    for a structural trust gate.
    """
    options = (
        MatrixFreeObservabilityOptions1D() if options is None else options
    )
    _validated_matrix_free_options(options)
    stochastic_mode = _stochastic_options is not None
    if stochastic_mode and options.exhaustive:
        raise ValueError("stochastic all-site screening cannot be exhaustive")
    if stochastic_mode and site_indices is not None:
        raise ValueError("stochastic all-site screening does not accept site_indices")
    if stochastic_mode and preconditioner_diagonal is not None:
        raise ValueError(
            "stochastic all-site screening requires the identity preconditioner"
        )
    if not isinstance(prepared, PreparedLatticeSiteReconstruction1D):
        raise TypeError(
            "prepared must be a PreparedLatticeSiteReconstruction1D instance"
        )
    _validate_prepared_static_contract_1d(prepared)
    if not isinstance(reconstruction, LatticeSiteReconstruction1D):
        raise TypeError(
            "reconstruction must be a LatticeSiteReconstruction1D instance"
        )
    if not isinstance(reconstruction.metadata, Mapping):
        raise TypeError("reconstruction.metadata must be a mapping")
    for name, expected in (
        ("reconstruction_problem_id", prepared.reconstruction_problem_id),
        ("reconstructor_id", prepared.reconstructor_id),
        ("objective_id", prepared.objective_id),
    ):
        if reconstruction.metadata.get(name) != expected:
            raise ValueError(
                f"reconstruction metadata {name!r} does not match the prepared problem"
            )
    if prepared.similarity_residual_gauge:
        raise ValueError(
            "matrix-free observability does not yet implement the aligned "
            "translation/rotation/dilation residual gauge"
        )
    model = prepared.model
    _validate_reconstruction_renderer_state_1d(model, reconstruction)

    has_calibrated_problem = (
        prepared.measurement is not None or prepared.objective is not None
    )
    if has_calibrated_problem:
        canonical_counting_model = _canonical_prepared_poisson_counting_model_1d(
            prepared
        )
        if counting_model is None:
            counting_contract_scope = "derived_from_prepared_poisson_objective"
        else:
            validate_poisson_counting_model_1d(counting_model)
            mismatches = _counting_model_mismatches_1d(
                counting_model, canonical_counting_model
            )
            if mismatches:
                raise ValueError(
                    "counting_model conflicts with the prepared Poisson objective "
                    f"in field(s): {', '.join(mismatches)}"
                )
            counting_contract_scope = (
                "caller_verified_against_prepared_poisson_objective"
            )
        counting_model = canonical_counting_model
        prepared_counting_contract_bound = True
    else:
        if counting_model is None:
            raise ValueError(
                "legacy amplitude prepared problems require an external "
                "hypothetical counting_model"
            )
        validate_poisson_counting_model_1d(counting_model)
        counting_contract_scope = "external_hypothetical_legacy_amplitude"
        prepared_counting_contract_bound = False

    parameterization = _gauge_free_specimen_parameterization_1d(
        model, reconstruction, options
    )
    result_site_roles = np.asarray(reconstruction.site_role_codes)
    if result_site_roles.size:
        if result_site_roles.shape != (parameterization.n_site,):
            raise ValueError(
                "reconstruction site_role_codes do not match the prepared model"
            )
        reportable_sites = np.flatnonzero(
            result_site_roles == int(LatticeSiteRole1D.TARGET)
        ).astype(np.int32)
        nuisance_sites = np.flatnonzero(
            result_site_roles == int(LatticeSiteRole1D.NUISANCE)
        ).astype(np.int32)
        if reportable_sites.size + nuisance_sites.size != parameterization.n_site:
            raise ValueError(
                "prepared observability requires TARGET/NUISANCE modeled-site roles"
            )
    else:
        reportable_sites = np.arange(parameterization.n_site, dtype=np.int32)
        nuisance_sites = np.empty(0, dtype=np.int32)
    if stochastic_mode:
        selected_sites = reportable_sites
    else:
        maximum_selected_sites = operator.index(options.maximum_selected_sites)
        if result_site_roles.size and site_indices is None:
            selected_sites = reportable_sites
            if selected_sites.size > maximum_selected_sites:
                raise ValueError(
                    "all TARGET sites exceed options.maximum_selected_sites; "
                    "provide an explicit bounded site_indices subset"
                )
        else:
            selected_sites = _validated_selected_sites_1d(
                site_indices,
                parameterization.n_site,
                maximum_selected_sites,
            )
        selected_nuisance = np.intersect1d(
            selected_sites, nuisance_sites, assume_unique=True
        )
        if selected_nuisance.size:
            raise ValueError(
                "nuisance sites cannot be selected as structural observability "
                f"outputs: {selected_nuisance.tolist()}"
            )
    n_scan, n_detector = prepared.measured_intensities.shape
    if prepared.detector_valid_mask is None:
        detector_mask = np.ones((n_scan, n_detector), dtype=bool)
        detector_mask_mode = "all_valid_implicit"
    else:
        detector_mask = np.asarray(prepared.detector_valid_mask, dtype=bool)
        detector_mask_mode = "prepared_explicit"
    reconstruction_mask = getattr(reconstruction, "detector_valid_mask", None)
    if (reconstruction_mask is None) != (prepared.detector_valid_mask is None):
        raise ValueError(
            "reconstruction detector_valid_mask does not match the prepared problem"
        )
    if reconstruction_mask is not None and not np.array_equal(
        np.asarray(reconstruction_mask, dtype=bool), detector_mask
    ):
        raise ValueError(
            "reconstruction detector_valid_mask does not match the prepared problem"
        )

    tangent, nuisance_names, nuisance_profile_id, nuisance_metadata = (
        _validated_nuisance_tangent_1d(
            nuisance_profile,
            n_scan=n_scan,
            n_detector=n_detector,
            maximum_columns=options.maximum_nuisance_columns,
        )
    )
    constructor_contract = nuisance_metadata.get("constructor_contract")
    generated_nuisance_profile = isinstance(
        constructor_contract, str
    ) and constructor_contract.startswith("prepared_poisson_nuisance_autodiff:")
    if generated_nuisance_profile:
        (
            missing_nuisance_scopes,
            represented_nuisance_coverage,
        ) = _validate_generated_prepared_nuisance_profile_1d(
            tangent=tangent,
            parameter_names=nuisance_names,
            profile_id=nuisance_profile_id,
            metadata=nuisance_metadata,
            prepared=prepared,
            reconstruction=reconstruction,
            counting_model=counting_model,
            detector_mask=detector_mask,
        )
    else:
        missing_nuisance_scopes = [
            "probe",
            "scan_geometry",
            "detector_calibration",
            "fixed_exterior_material",
        ]
        represented_nuisance_coverage = {}
    material_scope_complete = bool(
        prepared.metadata.get("material_scope_complete", False)
        and prepared.metadata.get("material_scope_fully_parameterized", False)
        and reconstruction.material_scope_complete
        and reconstruction.material_scope_fully_parameterized
        and reconstruction.support_contract_id
        == prepared.metadata.get("support_contract_id")
    )
    if material_scope_complete:
        missing_nuisance_scopes = [
            scope
            for scope in missing_nuisance_scopes
            if scope != "fixed_exterior_material"
        ]
        represented_nuisance_coverage = {
            **dict(represented_nuisance_coverage),
            "material_support": {
                "target_sites": int(reportable_sites.size),
                "nuisance_sites_profiled": int(nuisance_sites.size),
                "support_contract_id": reconstruction.support_contract_id,
            },
        }
    nuisance_digest = _digest_arrays_and_metadata_1d(
        {"whitened_tangent": tangent},
        {
            "profile_id": nuisance_profile_id,
            "parameter_names": list(nuisance_names),
            "rank_rtol": float(options.nuisance_rank_rtol),
            "profile_metadata": nuisance_metadata,
        },
    )
    detector_mask_digest = _digest_arrays_and_metadata_1d(
        {"detector_valid_mask": detector_mask},
        {"mode": detector_mask_mode},
    )
    counting_digest = _digest_arrays_and_metadata_1d(
        {},
        {
            "electrons_per_pattern": float(counting_model.electrons_per_pattern),
            "background_electrons_per_pixel": float(
                counting_model.background_electrons_per_pixel
            ),
            "minimum_expected_electrons": float(
                counting_model.minimum_expected_electrons
            ),
            "calibrated": bool(counting_model.calibrated),
            "calibration_id": counting_model.calibration_id,
            "counting_contract_scope": counting_contract_scope,
            "prepared_counting_contract_bound": (
                prepared_counting_contract_bound
            ),
        },
    )
    reconstruction_state_hash = _digest_arrays_and_metadata_1d(
        {
            "potential": reconstruction.potential,
            "vacancy_fractions": reconstruction.vacancy_fractions,
            "displacement_controls": reconstruction.displacement_controls,
            "rigid_displacement": reconstruction.rigid_displacement,
            "site_coordinates": reconstruction.site_coordinates,
            "displaced_site_coordinates": reconstruction.displaced_site_coordinates,
        },
        {
            "reconstruction_problem_id": prepared.reconstruction_problem_id,
            "best_update": int(reconstruction.best_update),
            "completed_updates": int(reconstruction.completed_updates),
            "seed": reconstruction.metadata.get("seed"),
        },
    )
    if preconditioner_diagonal is None:
        preconditioner = np.ones(parameterization.n_parameter, dtype=float)
        preconditioner_mode = "identity"
    else:
        preconditioner = np.asarray(preconditioner_diagonal, dtype=float)
        if preconditioner.shape != (parameterization.n_parameter,):
            raise ValueError(
                "preconditioner_diagonal must have one value per specimen parameter"
            )
        if np.any(~np.isfinite(preconditioner)) or np.any(preconditioner <= 0.0):
            raise ValueError(
                "preconditioner_diagonal must contain finite positive values"
            )
        preconditioner_mode = "explicit_diagonal"
    preconditioner_digest = _digest_arrays_and_metadata_1d(
        {"preconditioner_diagonal": preconditioner},
        {"mode": preconditioner_mode},
    )

    dose = float(counting_model.electrons_per_pattern)
    background = float(counting_model.background_electrons_per_pixel)
    floor = float(counting_model.minimum_expected_electrons)
    probe_rows = jnp.asarray(prepared.probe_rows)
    starts = jnp.asarray(prepared.window_starts)
    kernel = jnp.asarray(prepared.propagation_kernel)
    mask_device = jnp.asarray(detector_mask)
    x0 = jnp.asarray(parameterization.x0)

    def observable_batch(
        parameters: Array,
        batch_indices: Array,
        valid_scans: Array,
    ) -> Array:
        vacancies, controls, translation, _ = parameterization.decode(parameters)
        potential = render_lattice_site_potential_1d(
            model, vacancies, controls + translation
        )
        batch_probes = probe_rows[batch_indices]
        intensities = simulate_glancing_scan_1d(
            potential,
            batch_probes,
            starts[batch_indices],
            prepared.window_length,
            kernel,
            prepared.slice_thickness,
            prepared.energy,
            rematerialize=prepared.rematerialize,
        )
        incident_norm = n_detector * jnp.sum(
            jnp.abs(batch_probes) ** 2, axis=1, keepdims=True
        )
        expected = jnp.maximum(
            dose * intensities / incident_norm + background,
            floor,
        )
        valid = mask_device[batch_indices] & valid_scans[:, None]
        return jnp.where(valid, 2.0 * jnp.sqrt(expected), 0.0).reshape(-1)

    @jax.jit
    def jvp_batch(
        direction: Array,
        batch_indices: Array,
        valid_scans: Array,
    ) -> Array:
        return jax.jvp(
            lambda values: observable_batch(values, batch_indices, valid_scans),
            (x0,),
            (direction,),
        )[1]

    @jax.jit
    def vjp_batch(
        cotangent: Array,
        batch_indices: Array,
        valid_scans: Array,
    ) -> Array:
        _, pullback = jax.vjp(
            lambda values: observable_batch(values, batch_indices, valid_scans),
            x0,
        )
        return pullback(cotangent)[0]

    scan_batch_size = operator.index(options.scan_batch_size)

    def batch_descriptors(indices: np.ndarray) -> list[tuple[Array, Array, int, int]]:
        descriptors: list[tuple[Array, Array, int, int]] = []
        row_start = 0
        for begin in range(0, len(indices), scan_batch_size):
            actual_indices = indices[begin : begin + scan_batch_size]
            actual_scans = len(actual_indices)
            padded = np.pad(
                actual_indices,
                (0, scan_batch_size - actual_scans),
                mode="edge",
            )
            valid = np.arange(scan_batch_size) < actual_scans
            actual_rows = actual_scans * n_detector
            descriptors.append(
                (
                    jnp.asarray(padded, dtype=jnp.int32),
                    jnp.asarray(valid),
                    row_start,
                    actual_rows,
                )
            )
            row_start += actual_rows
        return descriptors

    parameter_dtype = np.asarray(x0).dtype
    effective_relative_tolerance = max(
        float(options.relative_residual_tolerance),
        50.0 * float(np.finfo(parameter_dtype).eps),
    )

    from scipy.stats import chi2

    chi_square_radius = float(
        chi2.ppf(options.displacement_confidence, df=2)
    )

    def evaluate_split(indices: np.ndarray, role: str) -> SiteObservabilitySplit1D:
        nuisance_basis, nuisance_singular_values = (
            _orthonormal_nuisance_basis_1d(
                tangent,
                detector_mask,
                indices,
                rank_rtol=float(options.nuisance_rank_rtol),
            )
        )
        projector_checks = _projector_checks_1d(nuisance_basis)
        descriptors = batch_descriptors(indices)
        nuisance_rank = nuisance_basis.shape[1]
        nuisance_batches = []
        for _, _, row_start, actual_rows in descriptors:
            padded = np.zeros(
                (scan_batch_size * n_detector, nuisance_rank), dtype=float
            )
            padded[:actual_rows] = nuisance_basis[
                row_start : row_start + actual_rows
            ]
            nuisance_batches.append(jnp.asarray(padded, dtype=x0.dtype))

        def fisher_matvec(direction: Any) -> np.ndarray:
            vector = jnp.asarray(direction, dtype=x0.dtype)
            if vector.shape != x0.shape:
                raise ValueError("Fisher direction has incompatible shape")
            directional_observables = []
            nuisance_coefficient = jnp.zeros((nuisance_rank,), dtype=x0.dtype)
            for descriptor, nuisance_batch in zip(
                descriptors, nuisance_batches
            ):
                batch_indices, valid, _, _ = descriptor
                directional = jvp_batch(vector, batch_indices, valid)
                directional_observables.append(directional)
                nuisance_coefficient = (
                    nuisance_coefficient + nuisance_batch.T @ directional
                )
            result = jnp.zeros_like(x0)
            for descriptor, nuisance_batch, directional in zip(
                descriptors, nuisance_batches, directional_observables
            ):
                batch_indices, valid, _, _ = descriptor
                projected = directional - nuisance_batch @ nuisance_coefficient
                result = result + vjp_batch(projected, batch_indices, valid)
            return np.asarray(jax.block_until_ready(result), dtype=float)

        symmetry_errors = []
        normalized_curvatures = []
        parameter_axis = np.arange(1, parameterization.n_parameter + 1, dtype=float)
        for check_index in range(operator.index(options.operator_check_vectors)):
            first = np.sin((check_index + 1.0) * parameter_axis)
            second = np.cos(np.sqrt(check_index + 2.0) * parameter_axis)
            first_image = fisher_matvec(first)
            second_image = fisher_matvec(second)
            symmetry_errors.append(
                abs(
                    float(
                        np.dot(first, second_image)
                        - np.dot(first_image, second)
                    )
                )
                / max(
                    np.linalg.norm(first) * np.linalg.norm(second_image)
                    + np.linalg.norm(first_image) * np.linalg.norm(second),
                    1.0,
                )
            )
            for vector, image in (
                (first, first_image),
                (second, second_image),
            ):
                normalized_curvatures.append(
                    float(np.dot(vector, image))
                    / max(np.linalg.norm(vector) * np.linalg.norm(image), 1e-300)
                )
        zero_operator_error = float(
            np.linalg.norm(fisher_matvec(np.zeros(parameterization.n_parameter)))
        )
        maximum_symmetry_error = max(symmetry_errors, default=0.0)
        minimum_normalized_curvature = min(normalized_curvatures, default=0.0)
        operator_checks_passed = bool(
            zero_operator_error <= float(options.symmetry_tolerance)
            and maximum_symmetry_error <= float(options.symmetry_tolerance)
            and minimum_normalized_curvature >= -float(options.psd_tolerance)
        )
        projector_checks_passed = bool(
            max(projector_checks.values(), default=0.0)
            <= float(options.projector_tolerance)
        )

        if stochastic_mode:
            from .ptychography_stochastic_observability_1d import (
                StochasticFisherScreeningOptions1D,
                StochasticPhysicalBlock1D,
                estimate_stochastic_fisher_screening_1d,
            )

            if not isinstance(
                _stochastic_options, StochasticFisherScreeningOptions1D
            ):
                raise TypeError(
                    "stochastic options must be a "
                    "StochasticFisherScreeningOptions1D instance"
                )

            def factorized_detector_vjp(probe: Any) -> np.ndarray:
                probe_array = np.asarray(probe, dtype=float)
                expected_shape = (len(indices), n_detector)
                if probe_array.shape != expected_shape:
                    raise ValueError(
                        "stochastic detector probe has an incompatible shape"
                    )
                flattened = probe_array.reshape(-1)
                coefficient = nuisance_basis.T @ flattened
                projected = flattened - nuisance_basis @ coefficient
                result = jnp.zeros_like(x0)
                for descriptor in descriptors:
                    batch_indices, valid, row_start, actual_rows = descriptor
                    padded = np.zeros(
                        scan_batch_size * n_detector,
                        dtype=parameter_dtype,
                    )
                    padded[:actual_rows] = projected[
                        row_start : row_start + actual_rows
                    ]
                    result = result + vjp_batch(
                        jnp.asarray(padded, dtype=x0.dtype),
                        batch_indices,
                        valid,
                    )
                return np.asarray(
                    jax.block_until_ready(result),
                    dtype=float,
                )

            stochastic_output_indices = np.concatenate(
                [
                    selected_sites,
                    np.asarray(
                        [
                            parameterization.n_site + 2 * int(site_index) + component
                            for site_index in selected_sites
                            for component in (0, 1)
                        ],
                        dtype=np.int32,
                    ),
                ]
            )

            def reportable_physical_jvp(direction: Any) -> np.ndarray:
                return np.asarray(
                    parameterization.physical_output_jvp(direction)
                )[stochastic_output_indices]

            displacement_blocks = tuple(
                StochasticPhysicalBlock1D(
                    name=f"site_{site_index}_displacement",
                    row_indices=(
                        selected_sites.size + 2 * output_site_index,
                        selected_sites.size + 2 * output_site_index + 1,
                    ),
                    column_indices=(
                        selected_sites.size + 2 * output_site_index,
                        selected_sites.size + 2 * output_site_index + 1,
                    ),
                )
                for output_site_index, site_index in enumerate(selected_sites)
            )
            role_code = 0 if role == "fit" else 1
            role_seed = int(
                np.random.SeedSequence(
                    [_stochastic_options.random_seed, role_code]
                ).generate_state(1, dtype=np.uint64)[0]
            )
            role_options = replace(
                _stochastic_options,
                random_seed=role_seed,
            )
            screening = estimate_stochastic_fisher_screening_1d(
                parameter_count=parameterization.n_parameter,
                detector_probe_shape=(len(indices), n_detector),
                detector_vjp=factorized_detector_vjp,
                fisher_matvec=fisher_matvec,
                physical_jvp=reportable_physical_jvp,
                physical_covariance_blocks=displacement_blocks,
                options=role_options,
            )
            screening = replace(
                screening,
                factor_covariance_verified=True,
            )
            return PreparedStochasticObservabilitySplit1D(
                scan_indices=jnp.asarray(indices),
                screening=screening,
                operator_checks_passed=operator_checks_passed,
                projector_checks_passed=projector_checks_passed,
                metadata={
                    "role": role,
                    "method": (
                        "factorized_gaussian_all_site_projected_fisher"
                    ),
                    "n_parameters": parameterization.n_parameter,
                    "n_physical_outputs": (
                        stochastic_output_indices.size
                    ),
                    "physical_output_scope": "TARGET_sites_only",
                    "nuisance_sites_profiled_in_fisher": int(
                        nuisance_sites.size
                    ),
                    "nuisance_rank": nuisance_rank,
                    "nuisance_singular_values": (
                        nuisance_singular_values.tolist()
                    ),
                    "projector_checks": projector_checks,
                    "factor_covariance_constructed_from_same_jacobian": True,
                    "exact_selected_site_followup_required": True,
                },
            )

        output_indices = []
        for site_index in selected_sites:
            output_indices.extend(
                [
                    int(site_index),
                    parameterization.n_site + 2 * int(site_index),
                    parameterization.n_site + 2 * int(site_index) + 1,
                ]
            )
        right_hand_sides = [
            parameterization.physical_output_rhs(index)
            for index in output_indices
        ]
        diagnostics = [
            pcg_solve_observability_1d(
                fisher_matvec,
                rhs,
                preconditioner_diagonal=preconditioner,
                maximum_iterations=options.maximum_iterations,
                relative_residual_tolerance=effective_relative_tolerance,
                absolute_residual_tolerance=options.absolute_residual_tolerance,
                stagnation_iterations=options.stagnation_iterations,
                stagnation_relative_improvement=(
                    options.stagnation_relative_improvement
                ),
                curvature_tolerance=options.curvature_tolerance,
            )
            for rhs in right_hand_sides
        ]
        physical_estimable = np.zeros(
            parameterization.n_physical_output, dtype=bool
        )
        solutions: dict[int, np.ndarray] = {}
        for output_index, diagnostic in zip(output_indices, diagnostics):
            estimable = bool(
                diagnostic.converged
                and diagnostic.relative_residual
                <= effective_relative_tolerance
            )
            physical_estimable[output_index] = estimable
            if estimable:
                solutions[output_index] = np.asarray(diagnostic.solution, dtype=float)

        vacancy_error = np.full(parameterization.n_site, np.inf)
        displacement_covariance = np.full(
            (parameterization.n_site, 2, 2), np.inf
        )
        displacement_radius = np.full(parameterization.n_site, np.inf)
        for site_index in selected_sites:
            vacancy_output = int(site_index)
            if physical_estimable[vacancy_output]:
                rhs = parameterization.physical_output_rhs(vacancy_output)
                variance = float(np.dot(rhs, solutions[vacancy_output]))
                if variance >= -1e-10 * max(abs(variance), 1.0):
                    vacancy_error[site_index] = np.sqrt(max(variance, 0.0))
                else:
                    physical_estimable[vacancy_output] = False
            displacement_outputs = [
                parameterization.n_site + 2 * int(site_index),
                parameterization.n_site + 2 * int(site_index) + 1,
            ]
            if np.all(physical_estimable[displacement_outputs]):
                block = np.empty((2, 2), dtype=float)
                for row, row_output in enumerate(displacement_outputs):
                    row_rhs = parameterization.physical_output_rhs(row_output)
                    for column, column_output in enumerate(displacement_outputs):
                        block[row, column] = np.dot(
                            row_rhs, solutions[column_output]
                        )
                block = 0.5 * (block + block.T)
                eigenvalues = np.linalg.eigvalsh(block)
                if np.min(eigenvalues) >= -1e-10 * max(
                    np.max(np.abs(eigenvalues)), 1.0
                ):
                    displacement_covariance[site_index] = block
                    displacement_radius[site_index] = np.sqrt(
                        chi_square_radius * max(float(np.max(eigenvalues)), 0.0)
                    )
                else:
                    physical_estimable[displacement_outputs] = False

        exhaustive_metadata: dict[str, Any] = {
            "enabled": bool(options.exhaustive)
        }
        effective_rank = -1
        exhaustive_passed = not options.exhaustive
        if options.exhaustive:
            if parameterization.n_parameter > options.exhaustive_max_parameters:
                raise ValueError(
                    "exhaustive observability parameter count exceeds "
                    "options.exhaustive_max_parameters"
                )
            jacobian_parts = []
            for descriptor in descriptors:
                batch_indices, valid, _, actual_rows = descriptor
                jacobian = np.asarray(
                    jax.jacfwd(
                        lambda values: observable_batch(
                            values, batch_indices, valid
                        )
                    )(x0)
                )
                jacobian_parts.append(jacobian[:actual_rows])
            jacobian = np.concatenate(jacobian_parts, axis=0)
            projected_jacobian = jacobian - nuisance_basis @ (
                nuisance_basis.T @ jacobian
            )
            output_jacobian = parameterization.physical_output_jacobian()
            dense_covariance, dense_estimable, effective_rank = (
                marginal_covariance_from_jacobian_1d(
                    projected_jacobian,
                    output_jacobian,
                    rank_rtol=options.fisher_rank_rtol,
                )
            )
            operator_matrix = np.column_stack(
                [
                    fisher_matvec(
                        np.eye(parameterization.n_parameter, dtype=float)[:, column]
                    )
                    for column in range(parameterization.n_parameter)
                ]
            )
            dense_fisher = projected_jacobian.T @ projected_jacobian
            operator_relative_error = np.linalg.norm(
                operator_matrix - dense_fisher
            ) / max(np.linalg.norm(dense_fisher), 1.0)
            mismatch_count = int(
                np.count_nonzero(
                    physical_estimable[output_indices]
                    != dense_estimable[output_indices]
                )
            )
            covariance_errors = []
            covariance_scale = 1.0
            for row_output in output_indices:
                if not (
                    physical_estimable[row_output]
                    and dense_estimable[row_output]
                ):
                    continue
                row_rhs = parameterization.physical_output_rhs(row_output)
                for column_output in output_indices:
                    if not (
                        physical_estimable[column_output]
                        and dense_estimable[column_output]
                    ):
                        continue
                    pcg_value = float(
                        np.dot(row_rhs, solutions[column_output])
                    )
                    dense_value = float(
                        dense_covariance[row_output, column_output]
                    )
                    covariance_errors.append(abs(pcg_value - dense_value))
                    covariance_scale = max(covariance_scale, abs(dense_value))
            maximum_covariance_error = max(covariance_errors, default=0.0)
            exhaustive_passed = bool(
                mismatch_count == 0
                and operator_relative_error
                <= float(options.exhaustive_relative_tolerance)
                and maximum_covariance_error
                <= float(options.exhaustive_relative_tolerance) * covariance_scale
            )
            exhaustive_metadata = {
                "enabled": True,
                "effective_rank": effective_rank,
                "operator_relative_error": float(operator_relative_error),
                "physical_estimability_mismatch_count": mismatch_count,
                "maximum_covariance_absolute_error": float(
                    maximum_covariance_error
                ),
                "covariance_comparison_scale": float(covariance_scale),
                "passed": exhaustive_passed,
            }

        vacancies = np.asarray(parameterization.vacancies, dtype=float)
        occupied_boundary = options.vacancy_threshold - options.vacancy_margin
        vacant_boundary = options.vacancy_threshold + options.vacancy_margin
        occupied = vacancies < occupied_boundary
        vacant = vacancies > vacant_boundary
        distance_to_decision = np.where(
            occupied,
            occupied_boundary - vacancies,
            np.where(vacant, vacancies - vacant_boundary, 0.0),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            vacancy_z = distance_to_decision / vacancy_error
        vacancy_z[~np.isfinite(vacancy_z)] = 0.0
        vacancy_adequate = (
            physical_estimable[: parameterization.n_site]
            & (occupied | vacant)
            & (vacancy_z >= options.minimum_vacancy_z)
        )
        displacement_estimable = np.asarray(
            [
                np.all(
                    physical_estimable[
                        parameterization.n_site
                        + np.asarray([2 * site, 2 * site + 1])
                    ]
                )
                for site in range(parameterization.n_site)
            ]
        )
        displacement_adequate = displacement_estimable & (
            displacement_radius <= options.maximum_displacement_radius_A
        )
        site_observable = vacancy_adequate & (
            vacant | (occupied & displacement_adequate)
        )
        all_pcg_converged = all(value.converged for value in diagnostics)
        preconditioner_verifiable = bool(
            preconditioner_mode == "identity"
            or (options.exhaustive and exhaustive_passed)
        )
        solver_verified = bool(
            operator_checks_passed
            and projector_checks_passed
            and exhaustive_passed
            and all_pcg_converged
            and preconditioner_verifiable
        )
        return SiteObservabilitySplit1D(
            scan_indices=jnp.asarray(indices),
            vacancy_standard_error=jnp.asarray(vacancy_error),
            vacancy_z_to_decision_boundary=jnp.asarray(vacancy_z),
            displacement_covariance_A2=jnp.asarray(displacement_covariance),
            displacement_confidence_radius_A=jnp.asarray(displacement_radius),
            vacancy_information_adequate=jnp.asarray(vacancy_adequate),
            displacement_information_adequate=jnp.asarray(
                displacement_adequate
            ),
            site_observable=jnp.asarray(site_observable),
            physical_output_estimable=jnp.asarray(physical_estimable),
            solver_verified=solver_verified,
            effective_rank=effective_rank,
            metadata={
                "role": role,
                "method": "matrix_free_projected_fisher_pcg",
                "n_observations": int(len(indices) * n_detector),
                "n_valid_observations": int(
                    np.count_nonzero(detector_mask[indices])
                ),
                "n_parameters": parameterization.n_parameter,
                "selected_site_indices": selected_sites.tolist(),
                "nuisance_rank": nuisance_rank,
                "nuisance_singular_values": nuisance_singular_values.tolist(),
                "projector_checks": projector_checks,
                "projector_checks_passed": projector_checks_passed,
                "operator_zero_error": zero_operator_error,
                "maximum_operator_symmetry_error": maximum_symmetry_error,
                "minimum_normalized_operator_curvature": (
                    minimum_normalized_curvature
                ),
                "operator_checks_passed": operator_checks_passed,
                "pcg": _pcg_metadata_1d(output_indices, diagnostics),
                "exhaustive": exhaustive_metadata,
                "preconditioner_verifiable": preconditioner_verifiable,
            },
        )

    fit_indices = np.asarray(prepared.training_indices, dtype=np.int32)
    audit_indices = np.asarray(prepared.audit_indices, dtype=np.int32)
    fit_report = evaluate_split(fit_indices, "fit")
    audit_report = (
        evaluate_split(audit_indices, "audit") if audit_indices.size else None
    )
    if stochastic_mode:
        return PreparedStochasticObservability1D(
            site_coordinates=jnp.asarray(model.site_coordinates)[reportable_sites],
            fit=fit_report,
            audit=audit_report,
            metadata={
                "method": (
                    "prepared_factorized_gaussian_all_site_fisher_screen"
                ),
                "screening_only": True,
                "exact_selected_site_followup_required": True,
                "reconstruction_problem_id": prepared.reconstruction_problem_id,
                "reconstructor_id": prepared.reconstructor_id,
                "objective_id": prepared.objective_id,
                "reconstruction_state_sha256": reconstruction_state_hash,
                "detector_mask_sha256": detector_mask_digest,
                "counting_calibration_sha256": counting_digest,
                "calibration_id": counting_model.calibration_id,
                "counting_contract_scope": counting_contract_scope,
                "prepared_counting_contract_bound": (
                    prepared_counting_contract_bound
                ),
                "calibrated_noise": False,
                "nuisance_scope_complete": False,
                "missing_nuisance_scopes": missing_nuisance_scopes,
                "nuisance_profile_sha256": nuisance_digest,
                "nuisance_profile_id": nuisance_profile_id,
                "nuisance_parameter_names": list(nuisance_names),
                "training_indices": fit_indices.tolist(),
                "validation_indices": np.asarray(
                    prepared.validation_indices, dtype=int
                ).tolist(),
                "audit_indices": audit_indices.tolist(),
                "excluded_indices": np.asarray(
                    prepared.excluded_indices, dtype=int
                ).tolist(),
                "all_site_count": parameterization.n_site,
                "reportable_target_site_count": int(reportable_sites.size),
                "profiled_nuisance_site_count": int(nuisance_sites.size),
                "material_scope_complete": material_scope_complete,
                "support_contract_id": reconstruction.support_contract_id,
                "factor_covariance_verified_by_prepared_adapter": True,
                "suitable_for_trust_gate": False,
            },
        )
    if audit_report is None:
        combined_vacancy = np.zeros(parameterization.n_site, dtype=bool)
        combined_displacement = np.zeros(parameterization.n_site, dtype=bool)
        combined_sites = np.zeros(parameterization.n_site, dtype=bool)
    else:
        combined_vacancy = np.asarray(
            fit_report.vacancy_information_adequate
        ) & np.asarray(audit_report.vacancy_information_adequate)
        combined_displacement = np.asarray(
            fit_report.displacement_information_adequate
        ) & np.asarray(audit_report.displacement_information_adequate)
        combined_sites = np.asarray(fit_report.site_observable) & np.asarray(
            audit_report.site_observable
        )
    solver_policy = {
        "zero_start": True,
        "true_residual_recomputed_every_iteration": True,
        "scan_batch_size": scan_batch_size,
        "maximum_iterations": int(options.maximum_iterations),
        "requested_relative_residual_tolerance": float(
            options.relative_residual_tolerance
        ),
        "effective_relative_residual_tolerance": effective_relative_tolerance,
        "absolute_residual_tolerance": float(
            options.absolute_residual_tolerance
        ),
        "stagnation_iterations": int(options.stagnation_iterations),
        "stagnation_relative_improvement": float(
            options.stagnation_relative_improvement
        ),
        "curvature_tolerance": float(options.curvature_tolerance),
        "preconditioner_mode": preconditioner_mode,
        "preconditioner_digest": preconditioner_digest,
    }
    return LatticeSiteObservability1D(
        site_coordinates=jnp.asarray(model.site_coordinates),
        fit=fit_report,
        audit=audit_report,
        vacancy_information_adequate=jnp.asarray(combined_vacancy),
        displacement_information_adequate=jnp.asarray(combined_displacement),
        site_observable=jnp.asarray(combined_sites),
        ideal_poisson_information=True,
        # A declared calibration flag and identifier are provenance, not typed
        # residual-calibration evidence.  This phase-1 entry point accepts no
        # such evidence and must therefore remain fail closed on this gate.
        calibrated_noise=False,
        nuisance_scope_complete=False,
        suitable_for_trust_gate=False,
        metadata={
            "method": "matrix_free_projected_local_plugin_ideal_poisson_fisher",
            "phase": 1,
            "fisher_evaluation": "local_plugin_at_reconstructed_structure",
            "uncertainty_interpretation": (
                "interior_asymptotic_approximation_not_boundary_calibration"
            ),
            "parameterization": "zero_mean_rms_site_displacement_svd_basis",
            "n_parameters": parameterization.n_parameter,
            "displacement_basis_rank": parameterization.rank,
            "reconstruction_problem_id": prepared.reconstruction_problem_id,
            "reconstructor_id": prepared.reconstructor_id,
            "objective_id": prepared.objective_id,
            "reconstruction_state_sha256": reconstruction_state_hash,
            "detector_mask_sha256": detector_mask_digest,
            "detector_mask_mode": detector_mask_mode,
            "counting_calibration_sha256": counting_digest,
            "calibration_id": counting_model.calibration_id,
            "counting_contract_scope": counting_contract_scope,
            "prepared_counting_contract_bound": (
                prepared_counting_contract_bound
            ),
            "declared_counting_model_calibrated": bool(
                counting_model.calibrated
            ),
            "typed_calibration_evidence_supplied": False,
            "nuisance_profile_sha256": nuisance_digest,
            "nuisance_profile_id": nuisance_profile_id,
            "nuisance_parameter_names": list(nuisance_names),
            "nuisance_profile_metadata": dict(nuisance_metadata),
            "generated_nuisance_profile": generated_nuisance_profile,
            "represented_nuisance_coverage": represented_nuisance_coverage,
            "training_indices": fit_indices.tolist(),
            "validation_indices": np.asarray(
                prepared.validation_indices, dtype=int
            ).tolist(),
            "audit_indices": audit_indices.tolist(),
            "excluded_indices": np.asarray(
                prepared.excluded_indices, dtype=int
            ).tolist(),
            "selected_site_indices": selected_sites.tolist(),
            "structural_reporting_site_indices": reportable_sites.tolist(),
            "profiled_nuisance_site_indices": nuisance_sites.tolist(),
            "material_scope_complete": material_scope_complete,
            "support_contract_id": reconstruction.support_contract_id,
            "potential_dtype": str(model.reference_potential.dtype),
            "jax_backend": jax.default_backend(),
            "jax_devices": sorted(
                str(device) for device in model.reference_potential.devices()
            ),
            "solver_policy": solver_policy,
            "nuisance_scope_complete_derived": False,
            "missing_nuisance_scopes": missing_nuisance_scopes,
            "model_conditional": True,
            "loaded_archive_fail_closed": False,
        },
    )


def estimate_prepared_lattice_site_observability_stochastic_1d(
    prepared: PreparedLatticeSiteReconstruction1D,
    reconstruction: LatticeSiteReconstruction1D,
    counting_model: PoissonCountingModel1D | None = None,
    *,
    nuisance_profile: WhitenedNuisanceProfile1D | Any | None = None,
    operator_options: MatrixFreeObservabilityOptions1D | None = None,
    screening_options: Any | None = None,
) -> PreparedStochasticObservability1D:
    """Screen every site with factorized Gaussian covariance/null probes.

    This adapter constructs the detector factor from the same prepared
    Jacobian and nuisance projector as the exact selected-site Fisher method.
    It therefore verifies the factor-covariance identity by construction while
    retaining the stochastic core's simultaneous bounds and hard budgets.
    Positive results only nominate sites for exact selected-site follow-up and
    can never satisfy the structural-trust gate.
    """
    from .ptychography_stochastic_observability_1d import (
        StochasticFisherScreeningOptions1D,
    )

    resolved_screening_options = (
        StochasticFisherScreeningOptions1D()
        if screening_options is None
        else screening_options
    )
    result = estimate_prepared_lattice_site_observability_matrix_free_1d(
        prepared,
        reconstruction,
        counting_model,
        nuisance_profile=nuisance_profile,
        site_indices=None,
        preconditioner_diagonal=None,
        options=operator_options,
        _stochastic_options=resolved_screening_options,
    )
    if not isinstance(result, PreparedStochasticObservability1D):
        raise RuntimeError("stochastic prepared observability returned wrong type")
    return result


def _split_storage(prefix: str, split: SiteObservabilitySplit1D) -> dict[str, Any]:
    return {
        f"{prefix}_scan_indices": np.asarray(split.scan_indices),
        f"{prefix}_vacancy_standard_error": np.asarray(
            split.vacancy_standard_error
        ),
        f"{prefix}_vacancy_z_to_decision_boundary": np.asarray(
            split.vacancy_z_to_decision_boundary
        ),
        f"{prefix}_displacement_covariance_A2": np.asarray(
            split.displacement_covariance_A2
        ),
        f"{prefix}_displacement_confidence_radius_A": np.asarray(
            split.displacement_confidence_radius_A
        ),
        f"{prefix}_vacancy_information_adequate": np.asarray(
            split.vacancy_information_adequate
        ),
        f"{prefix}_displacement_information_adequate": np.asarray(
            split.displacement_information_adequate
        ),
        f"{prefix}_site_observable": np.asarray(split.site_observable),
        f"{prefix}_physical_output_estimable": np.asarray(
            split.physical_output_estimable
        ),
        f"{prefix}_solver_verified": np.asarray(split.solver_verified),
        f"{prefix}_effective_rank": np.asarray(split.effective_rank, dtype=np.int64),
        f"{prefix}_metadata_json": np.asarray(
            json.dumps(
                dict(split.metadata),
                default=lambda value: np.asarray(value).tolist(),
                sort_keys=True,
            )
        ),
    }


def save_lattice_site_observability_1d(
    path: str | Path,
    report: LatticeSiteObservability1D,
) -> None:
    """Save a dense or matrix-free observability report without pickle."""
    if not isinstance(report, LatticeSiteObservability1D):
        raise TypeError("report must be a LatticeSiteObservability1D")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "site_coordinates": np.asarray(report.site_coordinates),
        "audit_present": np.asarray(report.audit is not None),
        "vacancy_information_adequate": np.asarray(
            report.vacancy_information_adequate
        ),
        "displacement_information_adequate": np.asarray(
            report.displacement_information_adequate
        ),
        "site_observable": np.asarray(report.site_observable),
        "ideal_poisson_information": np.asarray(
            report.ideal_poisson_information
        ),
        "calibrated_noise": np.asarray(report.calibrated_noise),
        "nuisance_scope_complete": np.asarray(report.nuisance_scope_complete),
        "suitable_for_trust_gate": np.asarray(report.suitable_for_trust_gate),
        "metadata_json": np.asarray(
            json.dumps(
                dict(report.metadata),
                default=lambda value: np.asarray(value).tolist(),
                sort_keys=True,
            )
        ),
        **_split_storage("fit", report.fit),
    }
    if report.audit is not None:
        arrays.update(_split_storage("audit", report.audit))
    np.savez_compressed(destination, **arrays)


def _load_split(data: Any, prefix: str) -> SiteObservabilitySplit1D:
    return SiteObservabilitySplit1D(
        scan_indices=jnp.asarray(data[f"{prefix}_scan_indices"]),
        vacancy_standard_error=jnp.asarray(
            data[f"{prefix}_vacancy_standard_error"]
        ),
        vacancy_z_to_decision_boundary=jnp.asarray(
            data[f"{prefix}_vacancy_z_to_decision_boundary"]
        ),
        displacement_covariance_A2=jnp.asarray(
            data[f"{prefix}_displacement_covariance_A2"]
        ),
        displacement_confidence_radius_A=jnp.asarray(
            data[f"{prefix}_displacement_confidence_radius_A"]
        ),
        vacancy_information_adequate=jnp.asarray(
            data[f"{prefix}_vacancy_information_adequate"]
        ),
        displacement_information_adequate=jnp.asarray(
            data[f"{prefix}_displacement_information_adequate"]
        ),
        site_observable=jnp.asarray(data[f"{prefix}_site_observable"]),
        physical_output_estimable=jnp.asarray(
            data[f"{prefix}_physical_output_estimable"]
        ),
        solver_verified=bool(data[f"{prefix}_solver_verified"].item()),
        effective_rank=int(data[f"{prefix}_effective_rank"].item()),
        metadata=json.loads(str(data[f"{prefix}_metadata_json"].item())),
    )


def load_lattice_site_observability_1d(
    path: str | Path,
) -> LatticeSiteObservability1D:
    """Load a report written by the matching non-pickled save helper."""
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema_version"].item()) != 1:
            raise ValueError("unsupported observability schema version")
        fit = _load_split(data, "fit")
        audit = _load_split(data, "audit") if data["audit_present"].item() else None
        metadata = json.loads(str(data["metadata_json"].item()))
        metadata["loaded_archive_fail_closed"] = True
        return LatticeSiteObservability1D(
            site_coordinates=jnp.asarray(data["site_coordinates"]),
            fit=fit,
            audit=audit,
            vacancy_information_adequate=jnp.asarray(
                data["vacancy_information_adequate"]
            ),
            displacement_information_adequate=jnp.asarray(
                data["displacement_information_adequate"]
            ),
            site_observable=jnp.asarray(data["site_observable"]),
            ideal_poisson_information=bool(
                data["ideal_poisson_information"].item()
            ),
            # Schema v1 carries no typed calibration evidence.  A persisted
            # caller-controlled boolean must never become evidence on reload.
            calibrated_noise=False,
            nuisance_scope_complete=False,
            suitable_for_trust_gate=False,
            metadata=metadata,
        )
