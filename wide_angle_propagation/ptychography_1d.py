"""Differentiable 1D glancing-incidence propagation and compatibility helpers.

The propagation coordinate is ``s`` and the single transverse/detector
coordinate is ``u``.  A scan translates a fixed-length axial window through a
global two-dimensional electrostatic potential ``V(s, u)``. The maintained
user-facing inverse route is the sparse atomistic-edit facade in
``ptychography_atomistic_workflow_1d``. Pixel and dense lattice-site routines
remain here for compatibility and controlled baseline tests; they are not
re-exported by the package root.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import operator
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpy as np

from .propagation_methods import energy2wavelength, interaction_constant
from .ptychography_support_contract_1d import (
    LatticeSiteRole1D,
    LatticeSiteSupportContract1D,
    validate_lattice_site_support_contract_1d,
)


__all__ = [
    "beam_path_reconstruction_region_1d",
    "decompose_lattice_site_displacement_controls_1d",
    "decompose_lattice_site_similarity_controls_1d",
    "GlancingScan1D",
    "GlancingSideviewCache1D",
    "LatticeSiteModel1D",
    "PtychographyMeasurement1D",
    "PtychographyObjective1D",
    "load_glancing_scan_1d",
    "load_glancing_sideview_cache_1d",
    "lattice_site_displacements_1d",
    "normalized_amplitude_loss_1d",
    "ptychography_expected_signal_electrons_1d",
    "ptychography_objective_from_signal_electrons_1d",
    "ptychography_objective_loss_1d",
    "render_lattice_site_potential_1d",
    "render_lattice_site_potential_from_displacements_1d",
    "save_glancing_scan_1d",
    "save_glancing_sideview_cache_1d",
    "simulate_glancing_scan_1d",
    "simulate_glancing_sideview_cache_1d",
    "validate_ptychography_measurement_1d",
    "validate_ptychography_objective_1d",
]


Array = Any


_LATTICE_SITE_RECONSTRUCTOR_ID = (
    "wide_angle_propagation.lattice_site_prepared:v1"
)
_NORMALIZED_AMPLITUDE_OBJECTIVE_ID = (
    "wide_angle_propagation.normalized_amplitude_loss_1d:valid_mask_v1"
)


def _measurement_contract_1d(detector_valid_mask: Array | None) -> str:
    return (
        "masked_nonnegative_intensity"
        if detector_valid_mask is not None
        else "legacy_unmasked_intensity"
    )


@dataclass(frozen=True)
class ConvergenceOptions1D:
    """Stopping criteria for a reconstruction with a finite update budget."""

    min_updates: int = 200
    patience_evaluations: int = 8
    relative_min_delta: float = 1e-4
    normalized_step_tolerance: float = 1e-4
    target_loss: float | None = None


@dataclass(frozen=True)
class LatticeOptimizationOptions1D:
    """Parameter-group learning rates and optional staged update schedule."""

    mode: str = "joint"
    rigid_stage_fraction: float = 0.15
    vacancy_stage_fraction: float = 0.25
    residual_stage_fraction: float = 0.25
    vacancy_learning_rate_scale: float = 1.0
    residual_learning_rate_scale: float = 0.5
    rigid_learning_rate_scale: float = 1.0


@dataclass(frozen=True)
class PtychographyMeasurement1D:
    """Truth-free calibrated detector values supplied to an inverse problem.

    Both observed arrays are retained because the Poisson deviance uses total
    calibrated electron-equivalent observations, whereas the read-noise
    approximation uses the calibrated signal after declared dark subtraction.
    The two arrays must obey that declared subtraction on every valid pixel.
    The type contains no synthetic truth or uncalibrated detector parameters.
    """

    calibrated_signal_electrons: Array
    observed_total_electrons: Array
    valid_mask: Array
    calibrated_dark_electrons_per_pixel: Array
    calibrated_read_noise_std_electrons: Array
    calibration_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PtychographyObjective1D:
    """Fixed calibrated detector objective; no global scale is fitted."""

    kind: Literal["poisson_deviance", "poisson_gaussian_nll"]
    electrons_per_pattern: Array
    minimum_expected_electrons: float = 1e-9
    relative_signal_scale: float = 1.0


@dataclass(frozen=True)
class GlancingScan1D:
    """A simulated scan and the coordinates needed to interpret it."""

    intensities: Array
    window_starts: Array
    scan_coordinates: Array
    detector_angles: Array
    metadata: Mapping[str, Any] = field(default_factory=dict)
    detector_valid_mask: Array | None = None


@dataclass(frozen=True)
class GlancingSideviewCache1D:
    """Downsampled internal fields and full exit/detector waves for selected scans."""

    scan_indices: Array
    window_starts: Array
    scan_coordinates: Array
    local_s_coordinates: Array
    sideview_u_coordinates: Array
    transverse_coordinates: Array
    sideview_wavefields: Array
    sideview_intensities: Array
    exit_waves: Array
    detector_waves: Array
    detector_intensities: Array
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PotentialReconstruction1D:
    """Best direct-potential estimate and its optimization diagnostics."""

    potential: Array
    initial_potential: Array
    reconstruction_mask: Array
    axial_coordinates: Array
    transverse_coordinates: Array
    predicted_intensities: Array
    measured_intensities: Array
    window_starts: Array
    scan_coordinates: Array
    detector_angles: Array
    update_history: Array
    training_loss_history: Array
    validation_loss_history: Array
    best_update: int
    audit_loss: float = float("nan")
    elapsed_time_history: Array = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    detector_valid_mask: Array | None = None


@dataclass(frozen=True)
class LatticeSiteModel1D:
    """Known reference potential and compact variable-site atom templates.

    Coordinates and displacements use ``(s, u)`` ordering.  Each site patch is
    stored on the specimen grid before displacement, and ``patch_starts`` gives
    the corresponding upper-left grid index in the full potential.  The patch
    must include enough zero padding to accommodate ``maximum_displacement``.
    """

    reference_potential: Array
    site_coordinates: Array
    site_patches: Array
    patch_starts: Array
    control_coordinates_s: Array
    control_coordinates_u: Array
    axial_sampling: Any
    transverse_sampling: Any
    maximum_displacement: Any = 0.5
    metadata: Mapping[str, Any] = field(default_factory=dict)
    support_contract: LatticeSiteSupportContract1D | None = None


@dataclass(frozen=True)
class LatticeSiteReconstruction1D:
    """Best lattice-site estimate and its optimization diagnostics."""

    potential: Array
    initial_potential: Array
    vacancy_fractions: Array
    initial_vacancy_fractions: Array
    displacement_controls: Array
    initial_displacement_controls: Array
    site_coordinates: Array
    displaced_site_coordinates: Array
    control_coordinates_s: Array
    control_coordinates_u: Array
    predicted_intensities: Array
    measured_intensities: Array
    window_starts: Array
    scan_coordinates: Array
    detector_angles: Array
    update_history: Array
    elapsed_time_history: Array
    training_loss_history: Array
    validation_loss_history: Array
    best_update: int
    completed_updates: int = 0
    converged: bool = False
    stop_reason: str = "maximum_updates"
    audit_loss: float = float("nan")
    gradient_norm_history: Array = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    normalized_step_history: Array = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    active_bound_fraction_history: Array = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    rigid_displacement: Array = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )
    initial_rigid_displacement: Array = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )
    rigid_displacement_history: Array = field(
        default_factory=lambda: np.empty((0, 2), dtype=float)
    )
    optimization_stage_history: Array = field(
        default_factory=lambda: np.empty(0, dtype="U16")
    )
    checkpoint_updates: Array = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    vacancy_fraction_history: Array = field(
        default_factory=lambda: np.empty((0, 0), dtype=float)
    )
    displacement_control_history: Array = field(
        default_factory=lambda: np.empty((0, 0, 0, 2), dtype=float)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    detector_valid_mask: Array | None = None
    predicted_signal_electrons: Array | None = None
    measurement: PtychographyMeasurement1D | None = None
    objective: PtychographyObjective1D | None = None
    site_role_codes: Array = field(
        default_factory=lambda: np.empty(0, dtype=np.int8)
    )
    support_contract_id: str | None = None
    material_scope_complete: bool = False
    material_scope_fully_parameterized: bool = False

    @property
    def target_site_mask(self) -> np.ndarray:
        """Sites eligible for structural reporting under the support contract."""
        roles = np.asarray(self.site_role_codes)
        if roles.size == 0:
            return np.zeros(np.asarray(self.site_coordinates).shape[0], dtype=bool)
        return roles == int(LatticeSiteRole1D.TARGET)

    @property
    def nuisance_site_mask(self) -> np.ndarray:
        """Modeled sites that must not be presented as recovered structure."""
        roles = np.asarray(self.site_role_codes)
        if roles.size == 0:
            return np.zeros(np.asarray(self.site_coordinates).shape[0], dtype=bool)
        return roles == int(LatticeSiteRole1D.NUISANCE)


@dataclass(frozen=True, eq=False)
class PreparedLatticeSiteReconstruction1D:
    """Validated, shape-specialized lattice reconstruction problem.

    Preparation transfers the fixed data and eagerly compiles the renderer,
    minibatch update, and fixed-size prediction executable.  A prepared
    problem can therefore be reused for independent initializations without
    retracing the forward model.  Optimizer state and random-number state are
    deliberately created by :func:`run_prepared_lattice_site_reconstruction_1d`
    and are never stored here.  Static fields are identity-bound to the
    compiled closures; prepare a new object whenever data or geometry changes.
    ``reconstruction_problem_id`` is the portable SHA-256 of that exact problem.
    """

    model: LatticeSiteModel1D
    input_probe: Array
    probe_rows: Array
    window_starts: Array
    window_length: int
    propagation_kernel: Array
    slice_thickness: Any
    energy: Any
    measured_intensities: Array
    measurement: PtychographyMeasurement1D | None
    objective: PtychographyObjective1D | None
    detector_valid_mask: Array | None
    scan_coordinates: Array
    detector_angles: Array
    training_indices: Array
    validation_indices: Array
    audit_indices: Array
    excluded_indices: Array
    potential_max: float
    maximum_phase_per_slice: float
    separate_rigid_registration: bool
    similarity_residual_gauge: bool
    maximum_rigid_displacement: float
    maximum_residual_displacement: float
    control_scale: float
    minibatch_size: int
    evaluation_batch_size: int
    gradient_clip: float
    epsilon: float
    rematerialize: bool
    objective_id: str
    reconstruction_problem_id: str
    reconstructor_id: str
    preparation_time_s: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _assemble: Any = field(default=None, repr=False, compare=False)
    _train_step: Any = field(default=None, repr=False, compare=False)
    _predict_batch: Any = field(default=None, repr=False, compare=False)
    _optimizer: Any = field(default=None, repr=False, compare=False)
    _static_contract: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, eq=False)
class _PreparedLatticeSiteStaticContract1D:
    """Process-local identity contract for captured compiled constants."""

    identity_fields: tuple[tuple[str, Any], ...]
    scalar_fields: tuple[tuple[str, tuple[Any, ...]], ...]


def _array(name: str, value: Any, ndim: int) -> Array:
    array = jnp.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {array.shape}")
    return array


def _concrete_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, jax.core.Tracer):
        return None
    try:
        return np.asarray(value)
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        return None


def _sha256_chunk(digest: Any, label: str, payload: bytes) -> None:
    """Add one unambiguous labelled byte string to a SHA-256 digest."""
    encoded_label = label.encode("utf-8")
    digest.update(len(encoded_label).to_bytes(8, "big"))
    digest.update(encoded_label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _reconstruction_problem_id_1d(
    *,
    arrays: Mapping[str, Any],
    options: Mapping[str, Any],
) -> str:
    """Hash the exact numerical inverse problem, including dtype and shape."""
    digest = hashlib.sha256()
    _sha256_chunk(
        digest,
        "contract",
        b"wide_angle_propagation.lattice_site_problem:v1",
    )
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise TypeError(f"cannot hash object-valued array {name!r}")
        dtype_string = array.dtype.str
        shape = list(array.shape)
        array = np.ascontiguousarray(array)
        header = json.dumps(
            {
                "dtype": dtype_string,
                "shape": shape,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _sha256_chunk(digest, f"array:{name}:header", header)
        _sha256_chunk(digest, f"array:{name}:data", array.tobytes(order="C"))
    options_json = json.dumps(
        dict(options),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _sha256_chunk(digest, "options", options_json)
    return digest.hexdigest()


def _array_sha256_1d(value: Any) -> str:
    """Hash one array with an unambiguous dtype-and-shape header."""
    array = np.asarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    _sha256_chunk(digest, "header", header)
    _sha256_chunk(
        digest,
        "data",
        np.ascontiguousarray(array).tobytes(order="C"),
    )
    return digest.hexdigest()


_TRAINING_DIAGNOSTIC_SELECTION_CONTRACT_1D = (
    "scan_coordinate_quantile_midpoints_with_index_tie_break:v1"
)


def _geometry_stratified_training_diagnostic_indices_1d(
    training_indices: Any,
    scan_coordinates: Any,
    requested_scan_count: int | None,
    *,
    validation_available: bool,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Choose deterministic training diagnostics without inspecting data values.

    When validation is available, evenly spaced rank midpoints in sorted scan
    coordinate provide a compact geometry-stratified diagnostic.  Acquisition
    indices break exact coordinate ties.  Without validation, the complete
    training partition is retained because its loss is authoritative for best
    checkpoint selection and convergence decisions.
    """
    training = np.asarray(training_indices, dtype=np.int64)
    coordinates = np.asarray(scan_coordinates)
    if training.ndim != 1 or training.size == 0:
        raise ValueError("training_indices must be a non-empty one-dimensional array")
    if coordinates.ndim != 1:
        raise ValueError("scan_coordinates must be one-dimensional")
    if np.any(training < 0) or np.any(training >= coordinates.size):
        raise ValueError("training_indices are outside scan_coordinates")
    if np.unique(training).size != training.size:
        raise ValueError("training_indices must be unique")
    if not np.all(np.isfinite(coordinates[training])):
        raise ValueError("training scan coordinates must be finite")
    if requested_scan_count is None:
        requested = None
    else:
        if isinstance(requested_scan_count, (bool, np.bool_)):
            raise TypeError("training_diagnostic_scan_count must be an integer")
        requested = _integer(
            "training_diagnostic_scan_count", requested_scan_count
        )

    fallback_reason: str | None = None
    if not validation_available:
        selected = training.copy()
        fallback_reason = "full_training_is_authoritative_without_validation"
    elif requested is None:
        selected = training.copy()
        fallback_reason = "prepared_runner_legacy_full_training_default"
    elif requested >= training.size:
        selected = training.copy()
        fallback_reason = "requested_count_not_smaller_than_training_partition"
    else:
        coordinate_values = coordinates[training]
        order = np.lexsort((training, coordinate_values))
        ordered_training = training[order]
        rank_midpoints = np.floor(
            (np.arange(requested, dtype=np.float64) + 0.5)
            * ordered_training.size
            / requested
        ).astype(np.int64)
        selected = ordered_training[rank_midpoints]
        if np.unique(selected).size != requested:
            raise RuntimeError(
                "geometry-stratified diagnostic selection produced duplicate scans"
            )

    selected = np.ascontiguousarray(selected, dtype=np.int64)
    selection_digest = _reconstruction_problem_id_1d(
        arrays={
            "training_indices": training,
            "training_scan_coordinates": coordinates[training],
            "selected_indices": selected,
        },
        options={
            "contract": _TRAINING_DIAGNOSTIC_SELECTION_CONTRACT_1D,
            "requested_scan_count": requested,
            "validation_available": bool(validation_available),
        },
    )
    return selected, {
        "construction": _TRAINING_DIAGNOSTIC_SELECTION_CONTRACT_1D,
        "requested_scan_count": requested,
        "resolved_scan_count": int(selected.size),
        "uses_full_training_partition": bool(
            np.array_equal(selected, training)
        ),
        "fallback_reason": fallback_reason,
        "selection_sha256": selection_digest,
    }


_PREPARED_IDENTITY_FIELDS_1D = (
    "model",
    "input_probe",
    "probe_rows",
    "window_starts",
    "propagation_kernel",
    "slice_thickness",
    "energy",
    "measured_intensities",
    "measurement",
    "objective",
    "detector_valid_mask",
    "scan_coordinates",
    "detector_angles",
    "training_indices",
    "validation_indices",
    "audit_indices",
    "excluded_indices",
    "metadata",
    "_assemble",
    "_train_step",
    "_predict_batch",
    "_optimizer",
)


_PREPARED_SCALAR_FIELDS_1D = (
    "window_length",
    "potential_max",
    "maximum_phase_per_slice",
    "separate_rigid_registration",
    "similarity_residual_gauge",
    "maximum_rigid_displacement",
    "maximum_residual_displacement",
    "control_scale",
    "minibatch_size",
    "evaluation_batch_size",
    "gradient_clip",
    "epsilon",
    "rematerialize",
    "objective_id",
    "reconstruction_problem_id",
    "reconstructor_id",
    "preparation_time_s",
)


def _prepared_scalar_token_1d(value: Any) -> tuple[Any, ...]:
    if isinstance(value, str):
        return ("str", value)
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.hasobject:
        raise TypeError("prepared scalar contract values must be scalar")
    array = np.ascontiguousarray(array)
    return (array.dtype.str, array.tobytes(order="C"))


def _make_prepared_static_contract_1d(
    prepared: PreparedLatticeSiteReconstruction1D,
) -> _PreparedLatticeSiteStaticContract1D:
    return _PreparedLatticeSiteStaticContract1D(
        identity_fields=tuple(
            (name, getattr(prepared, name))
            for name in _PREPARED_IDENTITY_FIELDS_1D
        ),
        scalar_fields=tuple(
            (name, _prepared_scalar_token_1d(getattr(prepared, name)))
            for name in _PREPARED_SCALAR_FIELDS_1D
        ),
    )


def _validate_prepared_static_contract_1d(
    prepared: PreparedLatticeSiteReconstruction1D,
) -> None:
    contract = prepared._static_contract
    if not isinstance(contract, _PreparedLatticeSiteStaticContract1D):
        raise ValueError(
            "prepared reconstruction has no valid process-local static contract; "
            "construct it with prepare_lattice_site_reconstruction_1d"
        )
    for name, expected in contract.identity_fields:
        if getattr(prepared, name) is not expected:
            raise ValueError(
                "prepared reconstruction static contract mismatch for "
                f"{name!r}; prepare a new inverse problem after changing static data"
            )
    for name, expected in contract.scalar_fields:
        if _prepared_scalar_token_1d(getattr(prepared, name)) != expected:
            raise ValueError(
                "prepared reconstruction static contract mismatch for "
                f"{name!r}; prepare a new inverse problem after changing static data"
            )
    problem_id = prepared.reconstruction_problem_id
    if len(problem_id) != 64 or any(
        character not in "0123456789abcdef" for character in problem_id
    ):
        raise ValueError("prepared reconstruction_problem_id is not a SHA-256 digest")


def _positive_scalar(name: str, value: Any, *, allow_zero: bool = False) -> None:
    concrete = _concrete_numpy(value)
    if concrete is None:
        return
    if concrete.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if np.iscomplexobj(concrete):
        raise TypeError(f"{name} must be real")
    scalar = float(concrete)
    valid = np.isfinite(scalar) and (scalar >= 0.0 if allow_zero else scalar > 0.0)
    if not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {relation} scalar, got {scalar!r}")


def _integer(name: str, value: Any, *, minimum: int = 1) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result}")
    return result


def _validate_convergence_options(
    options: ConvergenceOptions1D | None,
) -> ConvergenceOptions1D:
    options = ConvergenceOptions1D() if options is None else options
    if not isinstance(options, ConvergenceOptions1D):
        raise TypeError("convergence must be a ConvergenceOptions1D instance or None")
    _integer("convergence.min_updates", options.min_updates, minimum=0)
    _integer("convergence.patience_evaluations", options.patience_evaluations)
    _positive_scalar(
        "convergence.relative_min_delta",
        options.relative_min_delta,
        allow_zero=True,
    )
    _positive_scalar(
        "convergence.normalized_step_tolerance",
        options.normalized_step_tolerance,
        allow_zero=True,
    )
    if options.target_loss is not None:
        _positive_scalar(
            "convergence.target_loss", options.target_loss, allow_zero=True
        )
    return options


def _validate_lattice_optimization_options(
    options: LatticeOptimizationOptions1D | None,
) -> LatticeOptimizationOptions1D:
    options = LatticeOptimizationOptions1D() if options is None else options
    if not isinstance(options, LatticeOptimizationOptions1D):
        raise TypeError(
            "optimization must be a LatticeOptimizationOptions1D instance or None"
        )
    if options.mode not in {"joint", "staged"}:
        raise ValueError("optimization.mode must be 'joint' or 'staged'")
    fractions = np.asarray(
        [
            options.rigid_stage_fraction,
            options.vacancy_stage_fraction,
            options.residual_stage_fraction,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(fractions)) or np.any(fractions < 0.0):
        raise ValueError("optimization stage fractions must be finite and non-negative")
    if float(np.sum(fractions)) >= 1.0:
        raise ValueError("optimization stage fractions must sum to less than one")
    for name, value in (
        ("vacancy_learning_rate_scale", options.vacancy_learning_rate_scale),
        ("residual_learning_rate_scale", options.residual_learning_rate_scale),
        ("rigid_learning_rate_scale", options.rigid_learning_rate_scale),
    ):
        _positive_scalar(f"optimization.{name}", value)
    return options


def _validate_window_starts(
    window_starts: Any,
    *,
    n_s: int,
    window_length: int,
) -> Array:
    starts = _array("window_starts", window_starts, 1)
    if not jnp.issubdtype(starts.dtype, jnp.integer):
        raise TypeError("window_starts must contain integers")
    if starts.shape[0] == 0:
        raise ValueError("window_starts must contain at least one scan")
    concrete = _concrete_numpy(starts)
    if concrete is not None and (
        np.any(concrete < 0) or np.any(concrete + window_length > n_s)
    ):
        raise ValueError(
            f"every window start must satisfy 0 <= start <= {n_s - window_length}"
        )
    return starts


def _scan_partition_indices_1d(
    n_scan: int,
    *,
    validation_indices: Sequence[int],
    audit_indices: Sequence[int],
    excluded_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate disjoint train, validation, audit, and unused scan partitions."""

    def validated(name: str, values: Sequence[int]) -> np.ndarray:
        indices = np.asarray(values)
        if indices.ndim != 1 or (
            indices.size and not np.issubdtype(indices.dtype, np.integer)
        ):
            raise TypeError(f"{name} must be a one-dimensional integer sequence")
        indices = indices.astype(np.int64, copy=False)
        if (
            np.unique(indices).size != indices.size
            or np.any(indices < 0)
            or np.any(indices >= n_scan)
        ):
            raise ValueError(f"{name} must contain unique valid scan indices")
        return indices

    validation = validated("validation_indices", validation_indices)
    audit = validated("audit_indices", audit_indices)
    excluded = validated("excluded_indices", excluded_indices)
    partitions = {
        "validation_indices": validation,
        "audit_indices": audit,
        "excluded_indices": excluded,
    }
    names = tuple(partitions)
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1 :]:
            if np.intersect1d(
                partitions[first_name], partitions[second_name]
            ).size:
                raise ValueError(f"{first_name} and {second_name} must be disjoint")
    held_out = np.concatenate([validation, audit, excluded])
    training = np.setdiff1d(
        np.arange(n_scan, dtype=np.int64), held_out, assume_unique=True
    )
    if training.size == 0:
        raise ValueError("at least one scan must remain for training")
    return training, validation, audit, excluded


def _pixel_spacing(coordinates: Array) -> Array:
    if coordinates.shape[0] < 2:
        raise ValueError("coordinates must contain at least two points")
    concrete = _concrete_numpy(coordinates)
    if concrete is not None:
        differences = np.diff(concrete.astype(float, copy=False))
        if not np.all(np.isfinite(concrete)) or np.any(differences == 0.0):
            raise ValueError("coordinates must contain finite, distinct neighbors")
    return jnp.median(jnp.abs(jnp.diff(coordinates)))


def _validate_progress(progress: bool, description: str) -> None:
    if not isinstance(progress, (bool, np.bool_)):
        raise TypeError("progress must be a boolean")
    if not isinstance(description, str):
        raise TypeError("progress_description must be a string")


def _update_iterator(
    updates: int,
    *,
    progress: bool,
    description: str,
):
    """Return an update iterator, optionally backed by a notebook-safe TQDM bar."""
    _validate_progress(progress, description)
    if not progress:
        return range(1, updates + 1)
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:  # pragma: no cover - optional display dependency
        raise ImportError(
            "progress=True requires tqdm; install the notebook or dev extra"
        ) from exc
    return tqdm(
        range(1, updates + 1),
        total=updates,
        desc=description,
        unit="update",
        dynamic_ncols=True,
    )


def _multislice_step(
    wave: Array,
    potential_slice: Array,
    transfer: Array,
    sigma_dz: Array,
) -> Array:
    wave = wave * jnp.exp(1j * sigma_dz * potential_slice)
    return jnp.fft.ifft(jnp.fft.fft(wave, axis=-1) * transfer, axis=-1)


def simulate_glancing_scan_1d(
    global_potential: Any,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    *,
    rematerialize: bool = False,
) -> Array:
    """Return full fftshifted intensities for scan-specific probes and windows.

    The wave is propagated with :func:`jax.lax.scan`; FFTs act only on the last
    axis and no internal wavefront stack is retained. ``input_probe`` may be a
    single probe shared by all scans or a two-dimensional array containing one
    probe per scan.
    """
    potential = _array("global_potential", global_potential, 2)
    probe = jnp.asarray(input_probe)
    if probe.ndim not in (1, 2):
        raise ValueError("input_probe must be one- or two-dimensional")
    kernel = _array("propagation_kernel", propagation_kernel, 1)
    length = _integer("window_length", window_length)
    n_s, n_u = potential.shape
    if length > n_s:
        raise ValueError("window_length cannot exceed global_potential.shape[0]")
    if probe.shape[-1] != n_u or kernel.shape[0] != n_u:
        raise ValueError("input_probe and propagation_kernel must have length n_u")
    starts = _validate_window_starts(
        window_starts,
        n_s=n_s,
        window_length=length,
    )
    _positive_scalar("slice_thickness", slice_thickness)
    _positive_scalar("energy", energy)
    if not isinstance(rematerialize, (bool, np.bool_)):
        raise TypeError("rematerialize must be a boolean")

    if probe.ndim == 1:
        probes = jnp.broadcast_to(probe, (starts.shape[0], n_u))
    elif probe.shape[0] == starts.shape[0]:
        probes = probe
    else:
        raise ValueError("two-dimensional input_probe must have one row per scan")
    complex_dtype = jnp.result_type(probes.dtype, kernel.dtype, jnp.complex64)
    probes = probes.astype(complex_dtype)
    transfer = kernel.astype(complex_dtype)
    sigma_dz = interaction_constant(energy) * slice_thickness

    def step(wave: Array, potential_slice: Array) -> tuple[Array, None]:
        return _multislice_step(wave, potential_slice, transfer, sigma_dz), None

    scan_step = jax.checkpoint(step) if rematerialize else step

    def run_window(start: Array, initial_wave: Array) -> Array:
        slices = jax.lax.dynamic_slice_in_dim(potential, start, length, axis=0)
        exit_wave, _ = jax.lax.scan(scan_step, initial_wave, slices)
        return exit_wave

    exit_waves = jax.vmap(run_window)(starts, probes)
    detector_waves = jnp.fft.fftshift(jnp.fft.fft(exit_waves, axis=-1), axes=-1)
    return jnp.abs(detector_waves) ** 2


def _detector_valid_mask_1d(
    detector_valid_mask: Any | None,
    shape: tuple[int, ...],
) -> Array | None:
    """Return a validated Boolean detector mask without inventing validity."""
    if detector_valid_mask is None:
        return None
    valid = jnp.asarray(detector_valid_mask)
    if valid.shape != shape:
        raise ValueError(
            "detector_valid_mask must match the intensity shape, got "
            f"{valid.shape} and {shape}"
        )
    if valid.dtype != jnp.bool_:
        raise TypeError("detector_valid_mask must have Boolean dtype")
    valid_host = _concrete_numpy(valid)
    if valid_host is not None and not np.any(valid_host):
        raise ValueError("detector_valid_mask must select at least one observation")
    return valid


def _validate_masked_intensities_1d(
    name: str,
    intensities: Array,
    detector_valid_mask: Array | None,
) -> None:
    """Validate only observations that enter the objective."""
    host = _concrete_numpy(intensities)
    if host is None:
        return
    selected = host if detector_valid_mask is None else host[np.asarray(detector_valid_mask)]
    if not np.all(np.isfinite(selected)) or np.any(selected < 0.0):
        qualifier = "" if detector_valid_mask is None else " at valid detector pixels"
        raise ValueError(f"{name} must be finite and non-negative{qualifier}")


def _validate_detector_mask_partitions_1d(
    detector_valid_mask: Array | None,
    *,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    audit_indices: np.ndarray,
) -> None:
    """Require every fitted or assessed scan to contain an observable pixel."""
    if detector_valid_mask is None:
        return
    valid_per_scan = np.any(np.asarray(detector_valid_mask), axis=1)
    for name, indices in (
        ("training", training_indices),
        ("validation", validation_indices),
        ("audit", audit_indices),
    ):
        invalid_scans = indices[~valid_per_scan[indices]]
        if invalid_scans.size:
            raise ValueError(
                "detector_valid_mask leaves no valid observations for "
                f"{name} scan(s) {invalid_scans.tolist()}"
            )


def _calibration_array_1d(
    name: str,
    value: Any,
    shape: tuple[int, int],
    valid_mask: Array,
) -> Array:
    array = jnp.asarray(value)
    if jnp.iscomplexobj(array) or not jnp.issubdtype(array.dtype, jnp.number):
        raise TypeError(f"{name} must be a real numeric scalar or array")
    if array.ndim == 0:
        array = jnp.broadcast_to(array, shape)
    elif array.shape == (shape[1],):
        array = jnp.broadcast_to(array[None, :], shape)
    elif array.shape != shape:
        raise ValueError(
            f"{name} must be scalar or broadcast over measurement shape {shape}"
        )
    host = _concrete_numpy(array)
    if host is not None:
        selected = host[np.asarray(valid_mask)]
        if not np.all(np.isfinite(selected)) or np.any(selected < 0.0):
            raise ValueError(
                f"{name} must be finite and non-negative at valid pixels"
            )
    return array


def _validated_ptychography_measurement_1d(
    measurement: PtychographyMeasurement1D,
) -> PtychographyMeasurement1D:
    if not isinstance(measurement, PtychographyMeasurement1D):
        raise TypeError("measurement must be a PtychographyMeasurement1D")
    signal = _array(
        "measurement.calibrated_signal_electrons",
        measurement.calibrated_signal_electrons,
        2,
    )
    total = _array(
        "measurement.observed_total_electrons",
        measurement.observed_total_electrons,
        2,
    )
    if signal.shape != total.shape:
        raise ValueError("measurement observed arrays must have identical shapes")
    if jnp.iscomplexobj(signal) or jnp.iscomplexobj(total):
        raise TypeError("measurement observed arrays must be real")
    valid = _detector_valid_mask_1d(measurement.valid_mask, signal.shape)
    assert valid is not None
    for name, value in (
        ("measurement.calibrated_signal_electrons", signal),
        ("measurement.observed_total_electrons", total),
    ):
        host = _concrete_numpy(value)
        if host is not None and not np.all(
            np.isfinite(host[np.asarray(valid)])
        ):
            raise ValueError(f"{name} must be finite at valid pixels")
    dark = _calibration_array_1d(
        "measurement.calibrated_dark_electrons_per_pixel",
        measurement.calibrated_dark_electrons_per_pixel,
        signal.shape,
        valid,
    )
    read_noise = _calibration_array_1d(
        "measurement.calibrated_read_noise_std_electrons",
        measurement.calibrated_read_noise_std_electrons,
        signal.shape,
        valid,
    )
    signal_host = np.asarray(signal)[np.asarray(valid)]
    total_host = np.asarray(total)[np.asarray(valid)]
    dark_host = np.asarray(dark)[np.asarray(valid)]
    consistency_dtype = np.result_type(
        signal_host.dtype,
        total_host.dtype,
        dark_host.dtype,
        np.float32,
    )
    epsilon = np.finfo(consistency_dtype).eps
    consistency_scale = max(
        1.0,
        float(np.max(np.abs(signal_host))),
        float(np.max(np.abs(total_host))),
        float(np.max(np.abs(dark_host))),
    )
    if not np.allclose(
        signal_host,
        total_host - dark_host,
        rtol=64.0 * epsilon,
        atol=64.0 * epsilon * consistency_scale,
    ):
        raise ValueError(
            "measurement.calibrated_signal_electrons must equal "
            "observed_total_electrons minus calibrated dark at valid pixels"
        )
    if not isinstance(measurement.calibration_id, str) or not (
        measurement.calibration_id.strip()
    ):
        raise ValueError("measurement.calibration_id must be a non-empty string")
    if not isinstance(measurement.metadata, Mapping):
        raise TypeError("measurement.metadata must be a mapping")
    return PtychographyMeasurement1D(
        calibrated_signal_electrons=signal,
        observed_total_electrons=total,
        valid_mask=valid,
        calibrated_dark_electrons_per_pixel=dark,
        calibrated_read_noise_std_electrons=read_noise,
        calibration_id=measurement.calibration_id,
        metadata=MappingProxyType(dict(measurement.metadata)),
    )


def validate_ptychography_measurement_1d(
    measurement: PtychographyMeasurement1D,
) -> None:
    """Validate truth-free observations, calibration arrays, and mask."""
    _validated_ptychography_measurement_1d(measurement)


def _validated_ptychography_objective_1d(
    objective: PtychographyObjective1D,
    *,
    n_scans: int | None,
) -> tuple[PtychographyObjective1D, Array]:
    if not isinstance(objective, PtychographyObjective1D):
        raise TypeError("objective must be a PtychographyObjective1D")
    if objective.kind not in {"poisson_deviance", "poisson_gaussian_nll"}:
        raise ValueError(
            "objective.kind must be 'poisson_deviance' or "
            "'poisson_gaussian_nll'"
        )
    dose_host = np.asarray(objective.electrons_per_pattern)
    if (
        np.iscomplexobj(dose_host)
        or np.issubdtype(dose_host.dtype, np.bool_)
        or not np.issubdtype(dose_host.dtype, np.number)
    ):
        raise TypeError("objective.electrons_per_pattern must be real numeric")
    if dose_host.ndim == 0:
        if n_scans is None:
            dose_host = dose_host.reshape(1)
        else:
            dose_host = np.full(n_scans, dose_host.item(), dtype=dose_host.dtype)
    elif dose_host.ndim != 1:
        raise ValueError(
            "objective.electrons_per_pattern must be scalar or one-dimensional"
        )
    if n_scans is not None and dose_host.shape != (n_scans,):
        raise ValueError(
            "objective.electrons_per_pattern must be scalar or have one value "
            "per scan"
        )
    if dose_host.size == 0 or np.any(~np.isfinite(dose_host)) or np.any(
        dose_host <= 0.0
    ):
        raise ValueError(
            "objective.electrons_per_pattern must contain finite positive values"
        )
    _positive_scalar(
        "objective.minimum_expected_electrons",
        objective.minimum_expected_electrons,
    )
    _positive_scalar(
        "objective.relative_signal_scale", objective.relative_signal_scale
    )
    dose = jnp.asarray(dose_host)
    return (
        PtychographyObjective1D(
            kind=objective.kind,
            electrons_per_pattern=dose,
            minimum_expected_electrons=float(
                np.asarray(objective.minimum_expected_electrons)
            ),
            relative_signal_scale=float(
                np.asarray(objective.relative_signal_scale)
            ),
        ),
        dose,
    )


def validate_ptychography_objective_1d(
    objective: PtychographyObjective1D,
    *,
    n_scans: int | None = None,
) -> None:
    """Validate an objective and its fixed scalar or per-scan dose."""
    resolved_n_scans = (
        None if n_scans is None else _integer("n_scans", n_scans)
    )
    _validated_ptychography_objective_1d(
        objective, n_scans=resolved_n_scans
    )


def _expected_signal_electrons_1d(
    intensities: Array,
    probe_rows: Array,
    dose_per_scan: Array,
    relative_signal_scale: float,
) -> Array:
    n_detector = intensities.shape[1]
    incident_norm = n_detector * jnp.sum(
        jnp.abs(probe_rows) ** 2, axis=1, keepdims=True
    )
    return (
        relative_signal_scale
        * dose_per_scan[:, None]
        * intensities
        / incident_norm
    )


def ptychography_expected_signal_electrons_1d(
    predicted_intensities: Any,
    probe_rows: Any,
    objective: PtychographyObjective1D,
) -> Array:
    """Convert FFT intensities to calibrated signal electrons exactly."""
    intensities = _array("predicted_intensities", predicted_intensities, 2)
    probes = jnp.asarray(probe_rows)
    if probes.ndim == 1:
        probes = jnp.broadcast_to(probes, intensities.shape)
    if probes.shape != intensities.shape:
        raise ValueError("probe_rows must match predicted_intensities.shape")
    for name, value in (
        ("predicted_intensities", intensities),
        ("probe_rows", probes),
    ):
        host = _concrete_numpy(value)
        if host is not None and not np.all(np.isfinite(host)):
            raise ValueError(f"{name} must contain only finite values")
    intensity_host = _concrete_numpy(intensities)
    if intensity_host is not None and np.any(intensity_host < 0.0):
        raise ValueError("predicted_intensities must be non-negative")
    probe_host = _concrete_numpy(probes)
    if probe_host is not None:
        norms = intensities.shape[1] * np.sum(np.abs(probe_host) ** 2, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("every probe row must have finite positive norm")
    objective, dose = _validated_ptychography_objective_1d(
        objective, n_scans=int(intensities.shape[0])
    )
    return _expected_signal_electrons_1d(
        intensities, probes, dose, objective.relative_signal_scale
    )


def _ptychography_objective_from_signal_1d(
    predicted_signal: Array,
    measurement: PtychographyMeasurement1D,
    objective: PtychographyObjective1D,
) -> Array:
    valid = measurement.valid_mask
    signal = jnp.where(valid, predicted_signal, 0.0)
    dark = jnp.where(
        valid, measurement.calibrated_dark_electrons_per_pixel, 0.0
    )
    mean_total = jnp.maximum(
        signal + dark, objective.minimum_expected_electrons
    )
    if objective.kind == "poisson_deviance":
        observed = jnp.where(valid, measurement.observed_total_electrons, 0.0)
        ratio = jnp.where(observed > 0.0, observed / mean_total, 1.0)
        log_term = jnp.where(observed > 0.0, observed * jnp.log(ratio), 0.0)
        loss_terms = 2.0 * (mean_total - observed + log_term)
    else:
        observed = jnp.where(
            valid, measurement.calibrated_signal_electrons, 0.0
        )
        read_variance = jnp.where(
            valid,
            measurement.calibrated_read_noise_std_electrons**2,
            0.0,
        )
        variance = mean_total + read_variance
        loss_terms = 0.5 * (
            (observed - signal) ** 2 / variance
            + jnp.log(variance / objective.minimum_expected_electrons)
        )
    return jnp.sum(jnp.where(valid, loss_terms, 0.0)) / jnp.count_nonzero(valid)


def ptychography_objective_from_signal_electrons_1d(
    predicted_signal_electrons: Any,
    measurement: PtychographyMeasurement1D,
    objective: PtychographyObjective1D,
) -> Array:
    """Evaluate the calibrated detector objective from expected signal counts.

    This is the shared, JAX-differentiable count-loss boundary for specimen
    parameterizations that already converted their forward intensities into
    signal electrons.  It deliberately fits no detector scale, background, or
    calibration field.  Callers that introduce bounded nuisance parameters
    must transform the prediction explicitly and bind that transformation into
    their own problem contract.
    """
    measurement, objective, _ = _validated_measurement_objective_pair_1d(
        measurement, objective
    )
    predicted = _array(
        "predicted_signal_electrons", predicted_signal_electrons, 2
    )
    if predicted.shape != measurement.calibrated_signal_electrons.shape:
        raise ValueError(
            "predicted_signal_electrons must match the measurement shape"
        )
    host = _concrete_numpy(predicted)
    if host is not None and (
        np.any(~np.isfinite(host)) or np.any(host < 0.0)
    ):
        raise ValueError(
            "predicted_signal_electrons must be finite and non-negative"
        )
    return _ptychography_objective_from_signal_1d(
        predicted, measurement, objective
    )


def _validated_measurement_objective_pair_1d(
    measurement: PtychographyMeasurement1D,
    objective: PtychographyObjective1D,
) -> tuple[PtychographyMeasurement1D, PtychographyObjective1D, Array]:
    measurement = _validated_ptychography_measurement_1d(measurement)
    objective, dose = _validated_ptychography_objective_1d(
        objective,
        n_scans=int(measurement.calibrated_signal_electrons.shape[0]),
    )
    valid = np.asarray(measurement.valid_mask)
    if objective.kind == "poisson_deviance":
        read_noise = np.asarray(
            measurement.calibrated_read_noise_std_electrons
        )[valid]
        if np.any(read_noise != 0.0):
            raise ValueError("poisson_deviance requires declared zero read noise")
        observed_total = np.asarray(measurement.observed_total_electrons)[valid]
        if np.any(observed_total < 0.0):
            raise ValueError(
                "poisson_deviance requires non-negative observed total counts"
            )
    else:
        read_noise = np.asarray(
            measurement.calibrated_read_noise_std_electrons
        )[valid]
        if np.any(read_noise <= 0.0):
            raise ValueError(
                "poisson_gaussian_nll requires positive declared read noise"
            )
    return measurement, objective, dose


def ptychography_objective_loss_1d(
    predicted_intensities: Any,
    probe_rows: Any,
    measurement: PtychographyMeasurement1D,
    objective: PtychographyObjective1D,
) -> Array:
    """Evaluate a fixed calibrated objective without fitting detector scale."""
    measurement, objective, dose = _validated_measurement_objective_pair_1d(
        measurement, objective
    )
    intensities = _array("predicted_intensities", predicted_intensities, 2)
    probes = jnp.asarray(probe_rows)
    if probes.ndim == 1:
        probes = jnp.broadcast_to(probes, intensities.shape)
    if intensities.shape != measurement.calibrated_signal_electrons.shape:
        raise ValueError("prediction and measurement shapes must match")
    if probes.shape != intensities.shape:
        raise ValueError("probe_rows must match predicted_intensities.shape")
    host = _concrete_numpy(intensities)
    if host is not None and (
        np.any(~np.isfinite(host)) or np.any(host < 0.0)
    ):
        raise ValueError("predicted_intensities must be finite and non-negative")
    predicted_signal = _expected_signal_electrons_1d(
        intensities, probes, dose, objective.relative_signal_scale
    )
    return _ptychography_objective_from_signal_1d(
        predicted_signal, measurement, objective
    )


def normalized_amplitude_loss_1d(
    predicted_intensities: Any,
    measured_intensities: Any,
    *,
    epsilon: Any = 1e-12,
    detector_valid_mask: Any | None = None,
) -> Array:
    """Return normalized amplitude error over explicitly valid observations.

    ``None`` retains the original all-valid numerical path.  When a Boolean
    mask is supplied, invalid values are replaced before either square root,
    so a saturation sentinel, negative value, or non-finite value outside the
    valid set cannot contaminate the objective or its gradient.
    """
    predicted = jnp.asarray(predicted_intensities)
    measured = jnp.asarray(measured_intensities)
    if predicted.shape != measured.shape:
        raise ValueError(
            "predicted_intensities and measured_intensities must have identical "
            f"shapes, got {predicted.shape} and {measured.shape}"
        )
    if predicted.ndim == 0:
        raise ValueError("intensity arrays must have at least one dimension")
    if jnp.iscomplexobj(predicted) or jnp.iscomplexobj(measured):
        raise TypeError("intensity arrays must be real")
    _positive_scalar("epsilon", epsilon)
    valid = _detector_valid_mask_1d(detector_valid_mask, predicted.shape)
    _validate_masked_intensities_1d("predicted_intensities", predicted, valid)
    _validate_masked_intensities_1d("measured_intensities", measured, valid)
    if valid is not None:
        predicted = jnp.where(valid, predicted, jnp.zeros((), predicted.dtype))
        measured = jnp.where(valid, measured, jnp.zeros((), measured.dtype))
    amplitude_error = (
        jnp.sqrt(predicted + epsilon) - jnp.sqrt(measured + epsilon)
    ) ** 2
    return jnp.sum(amplitude_error) / jnp.maximum(jnp.sum(measured), epsilon)


def beam_path_reconstruction_region_1d(
    n_global_s: int,
    transverse_coordinates: Any,
    window_starts: Any,
    window_length: int,
    axial_sampling: Any,
    beam_tilt: Any,
    beam_waist: Any,
    slab_bottom: Any,
    *,
    slab_top: Any = 0.0,
    radius_waists: Any = 3.0,
    minimum_scan_coverage: int = 1,
) -> tuple[Array, Array]:
    """Return a geometric beam-path mask and per-pixel scan coverage count.

    Scan ``j`` crosses ``u=0`` at the midpoint of its local propagation
    window.  Its centreline is

    ``u_j(s) = tan(beam_tilt) * (s - s_cross_j)``.

    A material pixel belongs to the reconstruction region when it lies within
    ``radius_waists * beam_waist`` perpendicular distance of at least
    ``minimum_scan_coverage`` centrelines while those scans are inside their
    local windows.  Potential values inside the returned mask remain mutually
    independent; only their finite geometric support is prescribed.
    """
    n_s = _integer("n_global_s", n_global_s)
    length = _integer("window_length", window_length)
    if length > n_s:
        raise ValueError("window_length cannot exceed n_global_s")
    coordinates_u = _array("transverse_coordinates", transverse_coordinates, 1)
    if jnp.iscomplexobj(coordinates_u):
        raise TypeError("transverse_coordinates must be real")
    starts = _validate_window_starts(window_starts, n_s=n_s, window_length=length)
    _positive_scalar("axial_sampling", axial_sampling)
    _positive_scalar("beam_waist", beam_waist)
    _positive_scalar("radius_waists", radius_waists)
    coverage_required = _integer("minimum_scan_coverage", minimum_scan_coverage)
    tilt_host = np.asarray(beam_tilt)
    if tilt_host.ndim != 0 or np.iscomplexobj(tilt_host) or not np.isfinite(tilt_host):
        raise ValueError("beam_tilt must be a finite real scalar")
    bottom = float(np.asarray(slab_bottom))
    top = float(np.asarray(slab_top))
    if not np.isfinite(bottom) or not np.isfinite(top) or bottom >= top:
        raise ValueError("slab_bottom and slab_top must be finite with bottom < top")

    ds = float(np.asarray(axial_sampling))
    tilt = float(tilt_host)
    radius = float(np.asarray(radius_waists)) * float(np.asarray(beam_waist))
    local_s = np.arange(length, dtype=float) * ds
    local_midpoint = 0.5 * length * ds
    center_u = np.tan(tilt) * (local_s - local_midpoint)
    perpendicular_scale = abs(np.cos(tilt))
    u_host = np.asarray(coordinates_u, dtype=float)
    material = (u_host >= bottom) & (u_host <= top)
    local_distance = np.abs(u_host[None, :] - center_u[:, None]) * perpendicular_scale
    local_region = (local_distance <= radius) & material[None, :]

    coverage = np.zeros((n_s, u_host.size), dtype=np.int32)
    for start in np.asarray(starts, dtype=np.int64):
        coverage[int(start) : int(start) + length] += local_region
    mask = material[None, :] & (coverage >= coverage_required)
    return jnp.asarray(mask), jnp.asarray(coverage)


def _block_average_2d(array: Array, stride_s: int, stride_u: int) -> Array:
    n_s = (array.shape[-2] // stride_s) * stride_s
    n_u = (array.shape[-1] // stride_u) * stride_u
    trimmed = array[..., :n_s, :n_u]
    shape = (*trimmed.shape[:-2], n_s // stride_s, stride_s, n_u // stride_u, stride_u)
    return trimmed.reshape(shape).mean(axis=(-3, -1))


def _block_average_1d(array: Array, stride: int) -> Array:
    n = (array.shape[0] // stride) * stride
    return array[:n].reshape(n // stride, stride).mean(axis=1)


def simulate_glancing_sideview_cache_1d(
    global_potential: Any,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    scan_indices: Any,
    *,
    transverse_coordinates: Any | None = None,
    scan_coordinates: Any | None = None,
    axial_stride: int = 8,
    transverse_stride: int = 2,
    metadata: Mapping[str, Any] | None = None,
) -> GlancingSideviewCache1D:
    """Generate a compact diagnostic cache for selected scan positions only.

    ``input_probe`` can be shared across scans or supplied as one probe per scan.
    """
    potential = _array("global_potential", global_potential, 2)
    probe = jnp.asarray(input_probe)
    if probe.ndim not in (1, 2):
        raise ValueError("input_probe must be one- or two-dimensional")
    kernel = _array("propagation_kernel", propagation_kernel, 1)
    length = _integer("window_length", window_length)
    stride_s = _integer("axial_stride", axial_stride)
    stride_u = _integer("transverse_stride", transverse_stride)
    n_s, n_u = potential.shape
    if length > n_s:
        raise ValueError("window_length cannot exceed global_potential.shape[0]")
    if probe.shape[-1] != n_u or kernel.shape[0] != n_u:
        raise ValueError("input_probe and propagation_kernel must have length n_u")
    starts = _validate_window_starts(window_starts, n_s=n_s, window_length=length)
    indices = _array("scan_indices", scan_indices, 1)
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise TypeError("scan_indices must contain integers")
    indices_host = np.asarray(indices, dtype=np.int64)
    if (
        indices_host.size == 0
        or np.any(indices_host < 0)
        or np.any(indices_host >= starts.shape[0])
    ):
        raise ValueError("scan_indices must contain valid scan positions")
    if np.unique(indices_host).size != indices_host.size:
        raise ValueError("scan_indices must be unique")
    _positive_scalar("slice_thickness", slice_thickness)
    _positive_scalar("energy", energy)

    if transverse_coordinates is None:
        coordinates_u = jnp.arange(n_u, dtype=jnp.float32)
    else:
        coordinates_u = _array("transverse_coordinates", transverse_coordinates, 1)
        if coordinates_u.shape[0] != n_u:
            raise ValueError("transverse_coordinates must have length n_u")
    if scan_coordinates is None:
        coordinates_scan = (starts + length / 2) * slice_thickness
    else:
        coordinates_scan = _array("scan_coordinates", scan_coordinates, 1)
        if coordinates_scan.shape[0] != starts.shape[0]:
            raise ValueError("scan_coordinates must have length n_scan")

    if probe.ndim == 1:
        probes = jnp.broadcast_to(probe, (starts.shape[0], n_u))
    elif probe.shape[0] == starts.shape[0]:
        probes = probe
    else:
        raise ValueError("two-dimensional input_probe must have one row per scan")
    transfer = kernel.astype(jnp.result_type(probes, kernel, jnp.complex64))
    sigma_dz = interaction_constant(energy) * slice_thickness

    def run_window(start: Array, initial_wave: Array) -> tuple[Array, Array]:
        slices = jax.lax.dynamic_slice_in_dim(potential, start, length, axis=0)

        def step(wave: Array, potential_slice: Array) -> tuple[Array, Array]:
            wave = _multislice_step(wave, potential_slice, transfer, sigma_dz)
            return wave, wave

        return jax.lax.scan(step, initial_wave, slices)

    run_window_jit = jax.jit(run_window)
    sideview_fields = []
    sideview_intensities = []
    exit_waves = []
    detector_waves = []
    detector_intensities = []
    for index in indices_host:
        exit_wave, wavefields = run_window_jit(
            starts[int(index)], probes[int(index)].astype(transfer.dtype)
        )
        detector_wave = jnp.fft.fftshift(jnp.fft.fft(exit_wave))
        sideview_fields.append(
            _block_average_2d(wavefields, stride_s, stride_u).astype(jnp.complex64)
        )
        sideview_intensities.append(
            _block_average_2d(
                jnp.abs(wavefields) ** 2,
                stride_s,
                stride_u,
            ).astype(jnp.float32)
        )
        exit_waves.append(exit_wave)
        detector_waves.append(detector_wave)
        detector_intensities.append(jnp.abs(detector_wave) ** 2)

    sideview_fields = jnp.stack(sideview_fields)
    sideview_intensities = jnp.stack(sideview_intensities)
    local_s = _block_average_1d(
        jnp.arange(length, dtype=jnp.result_type(slice_thickness, jnp.float32))
        * slice_thickness,
        stride_s,
    )
    sideview_u = _block_average_1d(coordinates_u, stride_u)
    selected_starts = starts[indices]
    cache_metadata = {
        "axial_stride": stride_s,
        "transverse_stride": stride_u,
        "original_sideview_shape": [length, n_u],
        "stored_sideview_shape": list(sideview_fields.shape[-2:]),
        "complex_dtype": "complex64",
        "intensity_dtype": "float32",
        "downsampling": "complex and intensity block averages computed separately",
        **dict(metadata or {}),
    }
    return GlancingSideviewCache1D(
        scan_indices=indices,
        window_starts=selected_starts,
        scan_coordinates=coordinates_scan[indices],
        local_s_coordinates=local_s,
        sideview_u_coordinates=sideview_u,
        transverse_coordinates=coordinates_u,
        sideview_wavefields=sideview_fields,
        sideview_intensities=sideview_intensities,
        exit_waves=jnp.stack(exit_waves).astype(jnp.complex64),
        detector_waves=jnp.stack(detector_waves).astype(jnp.complex64),
        detector_intensities=jnp.stack(detector_intensities).astype(jnp.float32),
        metadata=cache_metadata,
    )


def _scatter_masked_values(
    normalized_values: Array,
    flat_indices: Array,
    shape: tuple[int, int],
    potential_scale: Array,
) -> Array:
    flat = jnp.zeros((shape[0] * shape[1],), dtype=normalized_values.dtype)
    flat = flat.at[flat_indices].set(normalized_values * potential_scale)
    return flat.reshape(shape)


def _validate_lattice_site_model_1d(
    model: LatticeSiteModel1D,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    reference = _array("model.reference_potential", model.reference_potential, 2)
    sites = _array("model.site_coordinates", model.site_coordinates, 2)
    patches = _array("model.site_patches", model.site_patches, 3)
    starts = _array("model.patch_starts", model.patch_starts, 2)
    controls_s = _array("model.control_coordinates_s", model.control_coordinates_s, 1)
    controls_u = _array("model.control_coordinates_u", model.control_coordinates_u, 1)
    if sites.shape[1:] != (2,):
        raise ValueError("model.site_coordinates must have shape (n_site, 2)")
    if starts.shape != sites.shape:
        raise ValueError("model.patch_starts must have shape (n_site, 2)")
    if patches.shape[0] != sites.shape[0]:
        raise ValueError("model.site_patches must have one patch per site")
    if sites.shape[0] == 0:
        raise ValueError("model must contain at least one variable site")
    if patches.shape[1] < 2 or patches.shape[2] < 2:
        raise ValueError("site patches must contain at least two samples per axis")
    if not jnp.issubdtype(starts.dtype, jnp.integer):
        raise TypeError("model.patch_starts must contain integers")
    if controls_s.shape[0] == 0 or controls_u.shape[0] == 0:
        raise ValueError("control-coordinate arrays must not be empty")
    if any(jnp.iscomplexobj(value) for value in (reference, sites, patches)):
        raise TypeError("lattice-site model arrays must be real")
    _positive_scalar("model.axial_sampling", model.axial_sampling)
    _positive_scalar("model.transverse_sampling", model.transverse_sampling)
    _positive_scalar(
        "model.maximum_displacement", model.maximum_displacement, allow_zero=True
    )

    for name, value in (
        ("model.reference_potential", reference),
        ("model.site_coordinates", sites),
        ("model.site_patches", patches),
        ("model.control_coordinates_s", controls_s),
        ("model.control_coordinates_u", controls_u),
    ):
        concrete = _concrete_numpy(value)
        if concrete is not None and not np.all(np.isfinite(concrete)):
            raise ValueError(f"{name} must contain only finite values")
    for name, coordinates in (
        ("model.control_coordinates_s", controls_s),
        ("model.control_coordinates_u", controls_u),
    ):
        concrete = _concrete_numpy(coordinates)
        if concrete is not None and concrete.size > 1:
            differences = np.diff(concrete.astype(float, copy=False))
            if np.any(differences <= 0.0) or not np.allclose(
                differences, differences[0], rtol=1e-6, atol=1e-12
            ):
                raise ValueError(f"{name} must be uniformly increasing")
    return reference, sites, patches, starts, controls_s, controls_u


def _coordinate_indices(values: Array, coordinates: Array) -> Array:
    if coordinates.shape[0] == 1:
        return jnp.zeros_like(values)
    return (
        (values - coordinates[0])
        / (coordinates[-1] - coordinates[0])
        * (coordinates.shape[0] - 1)
    )


def lattice_site_displacements_1d(
    site_coordinates: Array,
    displacement_controls: Array,
    control_coordinates_s: Array,
    control_coordinates_u: Array,
) -> Array:
    """Interpolate ``(s, u)`` control displacements at lattice sites."""
    site_s_indices = _coordinate_indices(site_coordinates[:, 0], control_coordinates_s)
    site_u_indices = _coordinate_indices(site_coordinates[:, 1], control_coordinates_u)
    sample_coordinates = jnp.stack([site_s_indices, site_u_indices])
    components = [
        map_coordinates(
            displacement_controls[..., component],
            sample_coordinates,
            order=1,
            mode="nearest",
        )
        for component in range(2)
    ]
    return jnp.stack(components, axis=-1)


def decompose_lattice_site_displacement_controls_1d(
    site_coordinates: Any,
    displacement_controls: Any,
    control_coordinates_s: Any,
    control_coordinates_u: Any,
    *,
    rigid_displacement: Any | None = None,
) -> tuple[Array, Array]:
    """Split total controls into rigid and zero-site-mean residual motion."""
    sites = _array("site_coordinates", site_coordinates, 2)
    controls = _array("displacement_controls", displacement_controls, 3)
    controls_s = _array("control_coordinates_s", control_coordinates_s, 1)
    controls_u = _array("control_coordinates_u", control_coordinates_u, 1)
    if sites.shape[1:] != (2,):
        raise ValueError("site_coordinates must have shape (n_site, 2)")
    if controls.shape != (len(controls_s), len(controls_u), 2):
        raise ValueError("displacement_controls have incompatible shape")
    if rigid_displacement is None:
        rigid = jnp.zeros((2,), dtype=controls.dtype)
    else:
        rigid = _array("rigid_displacement", rigid_displacement, 1)
        if rigid.shape != (2,):
            raise ValueError("rigid_displacement must have shape (2,)")
    site_displacements = lattice_site_displacements_1d(
        sites, controls, controls_s, controls_u
    )
    mean_displacement = jnp.mean(site_displacements, axis=0)
    return rigid + mean_displacement, controls - mean_displacement


def decompose_lattice_site_similarity_controls_1d(
    site_coordinates: Any,
    displacement_controls: Any,
    control_coordinates_s: Any,
    control_coordinates_u: Any,
    *,
    site_weights: Any | None = None,
) -> tuple[Array, Array]:
    """Split controls into global similarity modes and a residual field.

    The removed control-space subspace contains two translations, in-section
    rotation, and isotropic dilation.  Modes are fitted to interpolated site
    motion with optional non-negative geometry weights.  This gauge is useful
    after a complete-slab alignment search: residual strain cannot silently
    undo the selected origin, rotation, or lattice scale, while shear,
    anisotropic strain, and local defects remain available.

    Returns ``(similarity_controls, residual_controls)``.  The operation is a
    differentiable linear projection with respect to ``displacement_controls``.
    """
    sites = _array("site_coordinates", site_coordinates, 2)
    controls = _array("displacement_controls", displacement_controls, 3)
    controls_s = _array("control_coordinates_s", control_coordinates_s, 1)
    controls_u = _array("control_coordinates_u", control_coordinates_u, 1)
    if sites.shape[1:] != (2,) or sites.shape[0] < 2:
        raise ValueError("site_coordinates must have shape (n_site, 2), n_site >= 2")
    if not len(controls_s) or not len(controls_u):
        raise ValueError("control coordinate arrays must not be empty")
    expected_shape = (len(controls_s), len(controls_u), 2)
    if controls.shape != expected_shape:
        raise ValueError(
            f"displacement_controls must have shape {expected_shape}"
        )
    if any(
        jnp.iscomplexobj(value)
        for value in (sites, controls, controls_s, controls_u)
    ):
        raise TypeError("similarity-gauge coordinates and controls must be real")
    if site_weights is None:
        weights = jnp.ones((sites.shape[0],), dtype=controls.dtype)
    else:
        weights = _array("site_weights", site_weights, 1)
        if weights.shape != (sites.shape[0],):
            raise ValueError("site_weights must have one value per site")
        if jnp.iscomplexobj(weights):
            raise TypeError("site_weights must be real")
        weights = weights.astype(controls.dtype)

    for name, value in (
        ("site_coordinates", sites),
        ("displacement_controls", controls),
        ("control_coordinates_s", controls_s),
        ("control_coordinates_u", controls_u),
        ("site_weights", weights),
    ):
        host = _concrete_numpy(value)
        if host is not None and not np.all(np.isfinite(host)):
            raise ValueError(f"{name} must contain only finite values")
    for name, coordinates in (
        ("control_coordinates_s", controls_s),
        ("control_coordinates_u", controls_u),
    ):
        host = _concrete_numpy(coordinates)
        if host is not None and host.size > 1:
            differences = np.diff(host.astype(float, copy=False))
            if np.any(differences <= 0.0) or not np.allclose(
                differences,
                differences[0],
                rtol=1e-6,
                atol=1e-12,
            ):
                raise ValueError(f"{name} must be uniformly increasing")
    weights_host = _concrete_numpy(weights)
    if weights_host is not None and (
        np.any(weights_host < 0.0) or not np.any(weights_host > 0.0)
    ):
        raise ValueError("site_weights must be non-negative with positive sum")

    weight_sum = jnp.sum(weights)
    centroid = jnp.sum(weights[:, None] * sites, axis=0) / weight_sum
    centered_sites = sites - centroid
    rms_radius = jnp.sqrt(
        jnp.sum(weights * jnp.sum(centered_sites**2, axis=1)) / weight_sum
    )
    radius_host = _concrete_numpy(rms_radius)
    if radius_host is not None and (
        not np.isfinite(radius_host) or float(radius_host) <= 0.0
    ):
        raise ValueError("weighted site geometry must have positive spatial extent")

    grid_s, grid_u = jnp.meshgrid(controls_s, controls_u, indexing="ij")
    centered_grid_s = (grid_s - centroid[0]) / rms_radius
    centered_grid_u = (grid_u - centroid[1]) / rms_radius
    zeros = jnp.zeros_like(grid_s)
    ones = jnp.ones_like(grid_s)
    mode_controls = jnp.stack(
        [
            jnp.stack([ones, zeros], axis=-1),
            jnp.stack([zeros, ones], axis=-1),
            jnp.stack([-centered_grid_u, centered_grid_s], axis=-1),
            jnp.stack([centered_grid_s, centered_grid_u], axis=-1),
        ],
        axis=0,
    ).astype(controls.dtype)
    mode_site_displacements = jnp.stack(
        [
            lattice_site_displacements_1d(
                sites,
                mode_controls[index],
                controls_s,
                controls_u,
            )
            for index in range(4)
        ],
        axis=1,
    )
    site_displacements = lattice_site_displacements_1d(
        sites,
        controls,
        controls_s,
        controls_u,
    )
    gram = jnp.einsum(
        "imc,i,inc->mn",
        mode_site_displacements,
        weights,
        mode_site_displacements,
    )
    rhs = jnp.einsum(
        "imc,i,ic->m",
        mode_site_displacements,
        weights,
        site_displacements,
    )
    gram_host = _concrete_numpy(gram)
    if gram_host is not None:
        checked_gram = np.asarray(gram_host, dtype=np.float64)
        singular_values = np.linalg.svd(checked_gram, compute_uv=False)
        gram_epsilon = np.finfo(np.asarray(gram_host).dtype).eps
        tolerance = (
            max(checked_gram.shape)
            * gram_epsilon
            * singular_values[0]
        )
        if singular_values[-1] <= tolerance:
            raise ValueError(
                "site geometry does not independently constrain translation, "
                "rotation, and isotropic dilation"
            )
    coefficients = jnp.linalg.solve(gram, rhs)
    similarity_controls = jnp.einsum(
        "m,msuc->suc",
        coefficients,
        mode_controls,
    )
    return similarity_controls, controls - similarity_controls


def render_lattice_site_potential_1d(
    model: LatticeSiteModel1D,
    vacancy_fractions: Any,
    displacement_controls: Any,
) -> Array:
    """Render a known lattice with variable vacancies and smooth displacements.

    ``vacancy_fractions`` contains one value in ``[0, 1]`` per variable site.
    ``displacement_controls`` has shape ``(n_control_s, n_control_u, 2)`` and
    stores physical displacements in Angstrom in ``(s, u)`` order.  Bilinear
    interpolation transfers the control displacements to the lattice sites.
    """
    reference, sites, patches, starts, controls_s, controls_u = (
        _validate_lattice_site_model_1d(model)
    )
    vacancies = _array("vacancy_fractions", vacancy_fractions, 1)
    controls = _array("displacement_controls", displacement_controls, 3)
    if vacancies.shape[0] != sites.shape[0]:
        raise ValueError("vacancy_fractions must have one value per site")
    expected_controls = (controls_s.shape[0], controls_u.shape[0], 2)
    if controls.shape != expected_controls:
        raise ValueError(
            f"displacement_controls must have shape {expected_controls}, "
            f"got {controls.shape}"
        )
    if jnp.iscomplexobj(vacancies) or jnp.iscomplexobj(controls):
        raise TypeError("vacancy fractions and displacement controls must be real")
    vacancy_host = _concrete_numpy(vacancies)
    if vacancy_host is not None and (
        not np.all(np.isfinite(vacancy_host))
        or np.any(vacancy_host < 0.0)
        or np.any(vacancy_host > 1.0)
    ):
        raise ValueError("vacancy_fractions must contain finite values in [0, 1]")
    controls_host = _concrete_numpy(controls)
    maximum_displacement = float(np.asarray(model.maximum_displacement))
    if controls_host is not None and (
        not np.all(np.isfinite(controls_host))
        or np.any(np.abs(controls_host) > maximum_displacement)
    ):
        raise ValueError("displacement_controls exceed model.maximum_displacement")

    displacements = lattice_site_displacements_1d(
        sites, controls, controls_s, controls_u
    )
    return _render_lattice_site_arrays_1d(
        reference,
        patches,
        starts,
        vacancies,
        displacements,
        axial_sampling=model.axial_sampling,
        transverse_sampling=model.transverse_sampling,
        maximum_displacement=maximum_displacement,
    )


def render_lattice_site_potential_from_displacements_1d(
    model: LatticeSiteModel1D,
    vacancy_fractions: Any,
    site_displacements: Any,
) -> Array:
    """Render vacancies and independent physical site displacements.

    This lower-level parameterization is useful for local sensitivity and
    identifiability diagnostics. Reconstruction should normally use
    :func:`render_lattice_site_potential_1d`, which constrains the displacement
    field through smooth controls.
    """
    reference, sites, patches, starts, _, _ = _validate_lattice_site_model_1d(
        model
    )
    vacancies = _array("vacancy_fractions", vacancy_fractions, 1)
    displacements = _array("site_displacements", site_displacements, 2)
    if vacancies.shape != (sites.shape[0],):
        raise ValueError("vacancy_fractions must have one value per site")
    if displacements.shape != sites.shape:
        raise ValueError("site_displacements must have shape (n_site, 2)")
    if jnp.iscomplexobj(vacancies) or jnp.iscomplexobj(displacements):
        raise TypeError("vacancy fractions and site displacements must be real")
    vacancy_host = _concrete_numpy(vacancies)
    if vacancy_host is not None and (
        not np.all(np.isfinite(vacancy_host))
        or np.any(vacancy_host < 0.0)
        or np.any(vacancy_host > 1.0)
    ):
        raise ValueError("vacancy_fractions must contain finite values in [0, 1]")
    displacement_host = _concrete_numpy(displacements)
    maximum_displacement = float(np.asarray(model.maximum_displacement))
    if displacement_host is not None and (
        not np.all(np.isfinite(displacement_host))
        or np.any(np.abs(displacement_host) > maximum_displacement)
    ):
        raise ValueError("site_displacements exceed model.maximum_displacement")
    return _render_lattice_site_arrays_1d(
        reference,
        patches,
        starts,
        vacancies,
        displacements,
        axial_sampling=model.axial_sampling,
        transverse_sampling=model.transverse_sampling,
        maximum_displacement=maximum_displacement,
    )


def _render_lattice_site_arrays_1d(
    reference: Array,
    patches: Array,
    starts: Array,
    vacancies: Array,
    displacements: Array,
    *,
    axial_sampling: Any,
    transverse_sampling: Any,
    maximum_displacement: Any,
) -> Array:
    """Render already validated physical site parameters.

    Atomic patches are translated with separable Keys cubic convolution rather
    than piecewise-linear image interpolation.  The latter has a one-sided
    derivative whenever a displacement is an integer number of pixels,
    including at the pristine zero-displacement initialization.  The cubic
    kernel is interpolating and continuously differentiable at those knots.
    Samples beyond the compact patch are zero, while the model contract
    requires the patch itself to include the maximum physical displacement.
    """
    shifted_patches = _shift_lattice_site_patches_1d(
        patches,
        displacements,
        axial_sampling=axial_sampling,
        transverse_sampling=transverse_sampling,
        maximum_displacement=maximum_displacement,
    )
    patch_delta = ((1.0 - vacancies[:, None, None]) * shifted_patches - patches).astype(
        reference.dtype
    )

    offsets_s = jnp.arange(patches.shape[1], dtype=starts.dtype)
    offsets_u = jnp.arange(patches.shape[2], dtype=starts.dtype)
    rows = starts[:, 0, None, None] + offsets_s[None, :, None]
    columns = starts[:, 1, None, None] + offsets_u[None, None, :]
    rows = jnp.broadcast_to(rows, patch_delta.shape)
    columns = jnp.broadcast_to(columns, patch_delta.shape)
    valid = (
        (rows >= 0)
        & (rows < reference.shape[0])
        & (columns >= 0)
        & (columns < reference.shape[1])
    )
    flat_indices = jnp.clip(rows, 0, reference.shape[0] - 1) * reference.shape[
        1
    ] + jnp.clip(columns, 0, reference.shape[1] - 1)
    flat = reference.reshape(-1)
    flat = flat.at[flat_indices.reshape(-1)].add(
        jnp.where(valid, patch_delta, 0.0).reshape(-1)
    )
    return flat.reshape(reference.shape)


def _shift_lattice_site_patches_1d(
    patches: Array,
    displacements: Array,
    *,
    axial_sampling: Any,
    transverse_sampling: Any,
    maximum_displacement: Any,
) -> Array:
    """Smoothly translate compact patches with an interpolating cubic kernel.

    The Keys kernel uses ``a=-1/2`` (cubic convolution/Catmull--Rom), has four
    samples of support along each axis, and is ``C1`` at integer shifts.  A
    positive displacement moves patch content toward increasing array index,
    matching ``map_coordinates(grid - displacement / sampling)``.  Indices
    outside the already maximum-displacement-padded patch evaluate to zero;
    they never wrap to the opposite edge.
    """
    maximum = float(np.asarray(maximum_displacement))
    if maximum == 0.0:
        return patches

    work_dtype = jnp.result_type(patches.dtype, displacements.dtype, jnp.float32)
    shifted = patches.astype(work_dtype)
    displacement_values = displacements.astype(work_dtype)
    shift_s = displacement_values[:, 0] / jnp.asarray(
        axial_sampling, dtype=work_dtype
    )
    shift_u = displacement_values[:, 1] / jnp.asarray(
        transverse_sampling, dtype=work_dtype
    )

    shifted = jax.vmap(
        lambda patch, shift: _shift_patch_axis_keys_cubic_1d(
            patch, shift, axis=0
        )
    )(shifted, shift_s)
    return jax.vmap(
        lambda patch, shift: _shift_patch_axis_keys_cubic_1d(
            patch, shift, axis=1
        )
    )(shifted, shift_u)


def _keys_cubic_kernel_1d(distance: Array) -> Array:
    """Evaluate the interpolating Keys kernel with cubic parameter ``-1/2``."""
    parameter = -0.5
    absolute = jnp.abs(distance)
    inner = (
        (parameter + 2.0) * absolute - (parameter + 3.0)
    ) * absolute**2 + 1.0
    outer = (
        ((parameter * absolute - 5.0 * parameter) * absolute + 8.0 * parameter)
        * absolute
        - 4.0 * parameter
    )
    return jnp.where(
        absolute < 1.0,
        inner,
        jnp.where(absolute < 2.0, outer, 0.0),
    )


def _shift_patch_axis_keys_cubic_1d(
    patch: Array,
    shift_pixels: Array,
    *,
    axis: int,
) -> Array:
    """Translate one patch along one axis using four zero-extended samples."""
    base_offset = jnp.floor(-shift_pixels).astype(jnp.int32)
    offsets = base_offset + jnp.arange(-1, 3, dtype=jnp.int32)
    weights = _keys_cubic_kernel_1d(-shift_pixels - offsets)
    axis_length = patch.shape[axis]
    targets = jnp.arange(axis_length, dtype=jnp.int32)
    indices = targets[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < axis_length)
    samples = jnp.take(patch, jnp.clip(indices, 0, axis_length - 1), axis=axis)
    if axis == 0:
        return jnp.sum(
            jnp.where(valid[:, :, None], samples, 0.0)
            * weights[None, :, None],
            axis=1,
        )
    return jnp.sum(
        jnp.where(valid[None, :, :], samples, 0.0)
        * weights[None, None, :],
        axis=2,
    )


def reconstruct_potential_1d(
    initial_potential: Any,
    reconstruction_mask: Any,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    measured_intensities: Any,
    *,
    detector_valid_mask: Any | None = None,
    axial_coordinates: Any | None = None,
    transverse_coordinates: Any | None = None,
    scan_coordinates: Any | None = None,
    detector_angles: Any | None = None,
    validation_indices: Sequence[int] = (),
    audit_indices: Sequence[int] = (),
    excluded_indices: Sequence[int] = (),
    fixed_potential: Any | None = None,
    potential_scale: Any | None = None,
    potential_max: Any | None = None,
    learning_rate_start: Any = 1e-2,
    learning_rate_end: Any = 1e-4,
    updates: int = 4000,
    minibatch_size: int = 5,
    validation_interval: int = 100,
    evaluation_batch_size: int = 10,
    gradient_clip: Any = 1.0,
    epsilon: Any = 1e-12,
    rematerialize: bool = True,
    seed: int = 0,
    progress: bool = False,
    progress_description: str = "pixel reconstruction",
) -> PotentialReconstruction1D:
    """Recover non-negative pixels while retaining an optional fixed exterior.

    Values inside ``reconstruction_mask`` are initialized from
    ``initial_potential`` and optimized independently.  Values outside the mask
    are zero unless ``fixed_potential`` is supplied, in which case its exterior
    values remain in every forward simulation.
    """
    try:
        import optax
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "reconstruct_potential_1d requires Optax; install the 'ptychography' extra"
        ) from exc

    initial = _array("initial_potential", initial_potential, 2)
    mask = _array("reconstruction_mask", reconstruction_mask, 2).astype(bool)
    if mask.shape != initial.shape:
        raise ValueError("reconstruction_mask must match initial_potential.shape")
    initial_host = np.asarray(initial)
    mask_host = np.asarray(mask)
    if not np.any(mask_host):
        raise ValueError("reconstruction_mask must select at least one pixel")
    if np.iscomplexobj(initial_host) or not np.all(np.isfinite(initial_host)):
        raise ValueError("initial_potential must be finite and real")
    if np.any(initial_host[mask_host] < 0.0):
        raise ValueError("initial_potential must be non-negative inside the mask")
    if fixed_potential is None:
        fixed = jnp.zeros_like(initial)
        fixed_host = np.zeros_like(initial_host)
    else:
        fixed = _array("fixed_potential", fixed_potential, 2)
        if fixed.shape != initial.shape:
            raise ValueError("fixed_potential must match initial_potential.shape")
        fixed_host = np.asarray(fixed)
        if (
            np.iscomplexobj(fixed_host)
            or not np.all(np.isfinite(fixed_host))
            or np.any(fixed_host[~mask_host] < 0.0)
        ):
            raise ValueError(
                "fixed_potential must be finite, real, and non-negative outside "
                "the reconstruction mask"
            )

    probe = jnp.asarray(input_probe)
    if probe.ndim not in (1, 2):
        raise ValueError("input_probe must be one- or two-dimensional")
    kernel = _array("propagation_kernel", propagation_kernel, 1)
    measured = _array("measured_intensities", measured_intensities, 2)
    n_s, n_u = initial.shape
    length = _integer("window_length", window_length)
    starts = _validate_window_starts(window_starts, n_s=n_s, window_length=length)
    n_scan = starts.shape[0]
    if probe.shape[-1] != n_u or kernel.shape[0] != n_u:
        raise ValueError("input_probe and propagation_kernel must have length n_u")
    if probe.ndim == 2 and probe.shape[0] != n_scan:
        raise ValueError("two-dimensional input_probe must have one row per scan")
    if measured.shape != (n_scan, n_u):
        raise ValueError(f"measured_intensities must have shape {(n_scan, n_u)}")
    valid_mask = _detector_valid_mask_1d(
        detector_valid_mask, (n_scan, n_u)
    )
    _validate_masked_intensities_1d(
        "measured_intensities", measured, valid_mask
    )

    n_updates = _integer("updates", updates)
    batch_size = _integer("minibatch_size", minibatch_size)
    metric_interval = _integer("validation_interval", validation_interval)
    eval_batch_size = _integer("evaluation_batch_size", evaluation_batch_size)
    seed_value = operator.index(seed)
    _positive_scalar("slice_thickness", slice_thickness)
    _positive_scalar("energy", energy)
    _positive_scalar("learning_rate_start", learning_rate_start)
    _positive_scalar("learning_rate_end", learning_rate_end)
    _positive_scalar("gradient_clip", gradient_clip)
    _positive_scalar("epsilon", epsilon)
    if float(np.asarray(learning_rate_end)) > float(np.asarray(learning_rate_start)):
        raise ValueError("learning_rate_end must not exceed learning_rate_start")
    if not isinstance(rematerialize, (bool, np.bool_)):
        raise TypeError("rematerialize must be a boolean")
    _validate_progress(progress, progress_description)

    positive_initial = initial_host[mask_host & (initial_host > 0.0)]
    if potential_scale is None:
        resolved_scale = (
            float(np.mean(positive_initial)) if positive_initial.size else 1.0
        )
    else:
        resolved_scale = float(np.asarray(potential_scale))
    _positive_scalar("potential_scale", resolved_scale)
    if potential_max is None:
        resolved_max = 1.25 * max(
            float(np.max(initial_host[mask_host])), resolved_scale
        )
    else:
        resolved_max = float(np.asarray(potential_max))
    _positive_scalar("potential_max", resolved_max)
    if np.any(initial_host[mask_host] > resolved_max):
        raise ValueError("initial_potential exceeds potential_max inside the mask")
    fixed_exterior_max = (
        float(np.max(fixed_host[~mask_host])) if np.any(~mask_host) else 0.0
    )
    maximum_modeled_potential = max(resolved_max, fixed_exterior_max)
    max_phase = (
        float(np.asarray(interaction_constant(energy)))
        * float(np.asarray(slice_thickness))
        * maximum_modeled_potential
    )
    if max_phase >= np.pi:
        raise ValueError(
            "the optimized or fixed potential violates the per-slice phase bound: "
            f"sigma * slice_thickness * max_potential = {max_phase:.6g} >= pi"
        )

    training_host, validation_host, audit_host, excluded_host = (
        _scan_partition_indices_1d(
            n_scan,
            validation_indices=validation_indices,
            audit_indices=audit_indices,
            excluded_indices=excluded_indices,
        )
    )
    _validate_detector_mask_partitions_1d(
        valid_mask,
        training_indices=training_host,
        validation_indices=validation_host,
        audit_indices=audit_host,
    )

    flat_indices_host = np.flatnonzero(mask_host).astype(np.int32)
    flat_indices = jnp.asarray(flat_indices_host)
    scale = jnp.asarray(resolved_scale, dtype=jnp.result_type(initial, jnp.float32))
    upper_normalized = jnp.asarray(resolved_max / resolved_scale, dtype=scale.dtype)
    values = jnp.asarray(initial_host.reshape(-1)[flat_indices_host] / resolved_scale)

    if axial_coordinates is None:
        coordinates_s = jnp.arange(n_s, dtype=scale.dtype) * slice_thickness
    else:
        coordinates_s = _array("axial_coordinates", axial_coordinates, 1)
        if coordinates_s.shape[0] != n_s:
            raise ValueError("axial_coordinates must have length n_s")
    if transverse_coordinates is None:
        coordinates_u = jnp.arange(n_u, dtype=scale.dtype)
    else:
        coordinates_u = _array("transverse_coordinates", transverse_coordinates, 1)
        if coordinates_u.shape[0] != n_u:
            raise ValueError("transverse_coordinates must have length n_u")
    if scan_coordinates is None:
        coordinates_scan = coordinates_s[starts + length // 2]
    else:
        coordinates_scan = _array("scan_coordinates", scan_coordinates, 1)
        if coordinates_scan.shape[0] != n_scan:
            raise ValueError("scan_coordinates must have length n_scan")
    if detector_angles is None:
        du = _pixel_spacing(coordinates_u)
        frequencies = jnp.fft.fftshift(jnp.fft.fftfreq(n_u, du))
        detector_theta = 1e3 * jnp.arcsin(
            jnp.clip(energy2wavelength(energy) * frequencies, -1.0, 1.0)
        )
    else:
        detector_theta = _array("detector_angles", detector_angles, 1)
        if detector_theta.shape[0] != n_u:
            raise ValueError("detector_angles must have length n_u")

    fixed_flat = fixed.reshape(-1)

    def assemble(normalized_values: Array) -> Array:
        flat = fixed_flat.at[flat_indices].set(normalized_values * scale)
        return flat.reshape((n_s, n_u))

    probe_rows = jnp.broadcast_to(probe, (n_scan, n_u)) if probe.ndim == 1 else probe

    def batch_loss(
        normalized_values: Array,
        batch_starts: Array,
        batch_probes: Array,
        batch_measured: Array,
        batch_valid_mask: Array | None,
    ) -> Array:
        prediction = simulate_glancing_scan_1d(
            assemble(normalized_values),
            batch_probes,
            batch_starts,
            length,
            kernel,
            slice_thickness,
            energy,
            rematerialize=rematerialize,
        )
        return normalized_amplitude_loss_1d(
            prediction,
            batch_measured,
            epsilon=epsilon,
            detector_valid_mask=batch_valid_mask,
        )

    batch_value_and_grad = jax.jit(jax.value_and_grad(batch_loss))
    predict_batch = jax.jit(
        lambda normalized_values, batch_starts, batch_probes: simulate_glancing_scan_1d(
            assemble(normalized_values),
            batch_probes,
            batch_starts,
            length,
            kernel,
            slice_thickness,
            energy,
            rematerialize=rematerialize,
        )
    )

    alpha = float(np.asarray(learning_rate_end)) / float(
        np.asarray(learning_rate_start)
    )
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate_start,
        decay_steps=max(n_updates, 1),
        alpha=alpha,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip), optax.adam(schedule)
    )
    optimizer_state = optimizer.init(values)
    rng = np.random.default_rng(seed_value)

    def predict_indices(normalized_values: Array, indices: np.ndarray) -> Array:
        predictions = []
        for begin in range(0, len(indices), eval_batch_size):
            batch_indices = indices[begin : begin + eval_batch_size]
            predictions.append(
                predict_batch(
                    normalized_values,
                    starts[jnp.asarray(batch_indices)],
                    probe_rows[jnp.asarray(batch_indices)],
                )
            )
        return jnp.concatenate(predictions, axis=0)

    def evaluate(normalized_values: Array, indices: np.ndarray) -> float:
        prediction = predict_indices(normalized_values, indices)
        return float(
            np.asarray(
                normalized_amplitude_loss_1d(
                    prediction,
                    measured[jnp.asarray(indices)],
                    epsilon=epsilon,
                    detector_valid_mask=(
                        None
                        if valid_mask is None
                        else valid_mask[jnp.asarray(indices)]
                    ),
                )
            )
        )

    update_history: list[int] = []
    elapsed_history: list[float] = []
    training_history: list[float] = []
    validation_history: list[float] = []
    optimization_start = perf_counter()

    def record(update: int, normalized_values: Array) -> tuple[float, float]:
        training_loss = evaluate(normalized_values, training_host)
        validation_loss = (
            evaluate(normalized_values, validation_host)
            if validation_host.size
            else float("nan")
        )
        update_history.append(update)
        elapsed_history.append(perf_counter() - optimization_start)
        training_history.append(training_loss)
        validation_history.append(validation_loss)
        return training_loss, validation_loss

    training_loss, validation_loss = record(0, values)
    best_metric = validation_loss if validation_host.size else training_loss
    best_values = values
    best_update = 0

    for update in _update_iterator(
        n_updates,
        progress=progress,
        description=progress_description,
    ):
        batch_indices = rng.choice(
            training_host,
            size=batch_size,
            replace=training_host.size < batch_size,
        )
        _, gradient = batch_value_and_grad(
            values,
            starts[jnp.asarray(batch_indices)],
            probe_rows[jnp.asarray(batch_indices)],
            measured[jnp.asarray(batch_indices)],
            (
                None
                if valid_mask is None
                else valid_mask[jnp.asarray(batch_indices)]
            ),
        )
        parameter_updates, optimizer_state = optimizer.update(
            gradient,
            optimizer_state,
            values,
        )
        values = optax.apply_updates(values, parameter_updates)
        values = jnp.clip(values, 0.0, upper_normalized)

        if update % metric_interval == 0 or update == n_updates:
            training_loss, validation_loss = record(update, values)
            metric = validation_loss if validation_host.size else training_loss
            if np.isfinite(metric) and metric < best_metric:
                best_metric = metric
                best_values = values
                best_update = update

    best_potential = assemble(best_values)
    initial_global = assemble(
        jnp.asarray(initial_host.reshape(-1)[flat_indices_host] / resolved_scale)
    )
    all_indices = np.arange(n_scan, dtype=np.int64)
    predicted = predict_indices(best_values, all_indices)
    audit_loss = (
        float(
            np.asarray(
                normalized_amplitude_loss_1d(
                    predicted[jnp.asarray(audit_host)],
                    measured[jnp.asarray(audit_host)],
                    epsilon=epsilon,
                    detector_valid_mask=(
                        None
                        if valid_mask is None
                        else valid_mask[jnp.asarray(audit_host)]
                    ),
                )
            )
        )
        if audit_host.size
        else float("nan")
    )
    metadata = {
        "energy_eV": float(np.asarray(energy)),
        "slice_thickness_A": float(np.asarray(slice_thickness)),
        "potential_scale_V": resolved_scale,
        "potential_max_V": resolved_max,
        "maximum_phase_per_slice_rad": max_phase,
        "updates": n_updates,
        "minibatch_size": batch_size,
        "validation_interval": metric_interval,
        "evaluation_batch_size": eval_batch_size,
        "learning_rate_start": float(np.asarray(learning_rate_start)),
        "learning_rate_end": float(np.asarray(learning_rate_end)),
        "gradient_clip": float(np.asarray(gradient_clip)),
        "seed": int(seed_value),
        "training_indices": training_host.tolist(),
        "validation_indices": validation_host.tolist(),
        "audit_indices": audit_host.tolist(),
        "excluded_indices": excluded_host.tolist(),
        "audit_metric": audit_loss,
        "n_unknown_pixels": int(flat_indices_host.size),
        "uses_fixed_potential": fixed_potential is not None,
        "fixed_exterior_max_V": fixed_exterior_max,
        "best_metric": best_metric,
        "detector_angle_unit": "mrad",
        "objective_id": _NORMALIZED_AMPLITUDE_OBJECTIVE_ID,
        "measurement_contract": _measurement_contract_1d(valid_mask),
        "poisson_count_likelihood_supported": False,
        "read_noise_likelihood_supported": False,
        "detector_valid_mask_present": valid_mask is not None,
        "detector_valid_mask_sha256": (
            _array_sha256_1d(valid_mask) if valid_mask is not None else None
        ),
        "n_valid_detector_observations": (
            int(np.count_nonzero(np.asarray(valid_mask)))
            if valid_mask is not None
            else int(measured.size)
        ),
    }
    return PotentialReconstruction1D(
        potential=best_potential,
        initial_potential=initial_global,
        reconstruction_mask=mask,
        axial_coordinates=coordinates_s,
        transverse_coordinates=coordinates_u,
        predicted_intensities=predicted,
        measured_intensities=measured,
        window_starts=starts,
        scan_coordinates=coordinates_scan,
        detector_angles=detector_theta,
        update_history=jnp.asarray(update_history),
        elapsed_time_history=jnp.asarray(elapsed_history),
        training_loss_history=jnp.asarray(training_history),
        validation_loss_history=jnp.asarray(validation_history),
        best_update=best_update,
        audit_loss=audit_loss,
        metadata=metadata,
        detector_valid_mask=valid_mask,
    )


def prepare_lattice_site_reconstruction_1d(
    model: LatticeSiteModel1D,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    measured_intensities: Any | None = None,
    *,
    measurement: PtychographyMeasurement1D | None = None,
    objective: PtychographyObjective1D | None = None,
    detector_valid_mask: Any | None = None,
    separate_rigid_registration: bool = False,
    similarity_residual_gauge: bool = False,
    maximum_rigid_displacement: Any | None = None,
    maximum_residual_displacement: Any | None = None,
    scan_coordinates: Any | None = None,
    detector_angles: Any | None = None,
    validation_indices: Sequence[int] = (),
    audit_indices: Sequence[int] = (),
    excluded_indices: Sequence[int] = (),
    potential_max: Any | None = None,
    minibatch_size: int = 5,
    evaluation_batch_size: int = 10,
    gradient_clip: Any = 1.0,
    epsilon: Any = 1e-12,
    rematerialize: bool = True,
    require_complete_material_scope: bool = False,
) -> PreparedLatticeSiteReconstruction1D:
    """Validate and eagerly compile a fixed lattice inverse problem.

    The complete known reference specimen remains present.  Only the occupancy
    and position of the sites in ``model`` are changed, so fixed atoms continue
    to contribute to every forward simulation.  Minibatch and evaluation batch
    sizes are part of the prepared shape contract; run-specific schedules and
    initial values remain independent inputs to the prepared executable.
    """
    preparation_start = perf_counter()
    try:
        import optax
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "prepare_lattice_site_reconstruction_1d requires Optax; install "
            "the 'ptychography' extra"
        ) from exc

    if not isinstance(model, LatticeSiteModel1D):
        raise TypeError("model must be a LatticeSiteModel1D instance")
    if not isinstance(require_complete_material_scope, (bool, np.bool_)):
        raise TypeError("require_complete_material_scope must be a boolean")
    require_complete_material_scope = bool(require_complete_material_scope)
    reference, sites, patches, patch_starts, controls_s, controls_u = (
        _validate_lattice_site_model_1d(model)
    )
    support_contract = model.support_contract
    if support_contract is None:
        if require_complete_material_scope:
            raise ValueError(
                "strict material-scope preparation requires a "
                "LatticeSiteSupportContract1D"
            )
        modeled_site_roles = np.empty(0, dtype=np.int8)
        material_scope_complete = False
        material_scope_fully_parameterized = False
    else:
        support_contract = validate_lattice_site_support_contract_1d(
            support_contract,
            strict=require_complete_material_scope,
        )
        modeled_indices = np.asarray(support_contract.modeled_site_indices)
        expected_sites = np.asarray(support_contract.all_site_coordinates)[
            modeled_indices
        ]
        expected_starts = np.asarray(support_contract.site_patch_starts)[
            modeled_indices
        ]
        expected_shapes = np.asarray(support_contract.site_patch_shapes)[
            modeled_indices
        ]
        actual_patch_shape = np.broadcast_to(
            np.asarray(patches.shape[1:], dtype=np.int64),
            expected_shapes.shape,
        )
        if not np.array_equal(np.asarray(sites), expected_sites):
            raise ValueError(
                "model.site_coordinates do not match support-contract modeled sites"
            )
        if not np.array_equal(np.asarray(patch_starts), expected_starts):
            raise ValueError(
                "model.patch_starts do not match the support contract"
            )
        if not np.array_equal(actual_patch_shape, expected_shapes):
            raise ValueError(
                "model.site_patches do not match support-contract patch shapes"
            )
        if tuple(reference.shape) != tuple(support_contract.target_pixel_mask.shape):
            raise ValueError(
                "model.reference_potential does not match support-contract masks"
            )
        if not np.isclose(
            float(np.asarray(model.maximum_displacement)),
            support_contract.maximum_displacement_A,
            rtol=0.0,
            atol=8.0 * np.finfo(float).eps,
        ):
            raise ValueError(
                "model.maximum_displacement does not match the support contract"
            )
        modeled_site_roles = np.asarray(support_contract.site_role_codes)[
            modeled_indices
        ].astype(np.int8, copy=True)
        valid_modeled_roles = np.isin(
            modeled_site_roles,
            [int(LatticeSiteRole1D.TARGET), int(LatticeSiteRole1D.NUISANCE)],
        )
        if not np.all(valid_modeled_roles):
            raise ValueError(
                "support-contract modeled sites must be TARGET or NUISANCE"
            )
        material_scope_complete = bool(
            support_contract.strict_requirements_satisfied
        )
        material_scope_fully_parameterized = bool(
            material_scope_complete
            and not np.any(
                np.asarray(support_contract.site_role_codes)
                == int(LatticeSiteRole1D.FIXED_KNOWN)
            )
        )
    model = LatticeSiteModel1D(
        reference_potential=reference,
        site_coordinates=sites,
        site_patches=patches,
        patch_starts=patch_starts,
        control_coordinates_s=controls_s,
        control_coordinates_u=controls_u,
        axial_sampling=jnp.asarray(model.axial_sampling),
        transverse_sampling=jnp.asarray(model.transverse_sampling),
        maximum_displacement=jnp.asarray(model.maximum_displacement),
        metadata=MappingProxyType(dict(model.metadata)),
        support_contract=support_contract,
    )
    n_s, n_u = reference.shape
    n_site = sites.shape[0]
    control_shape = (controls_s.shape[0], controls_u.shape[0], 2)
    if not isinstance(separate_rigid_registration, (bool, np.bool_)):
        raise TypeError("separate_rigid_registration must be a boolean")
    separate_rigid_registration = bool(separate_rigid_registration)
    if not isinstance(similarity_residual_gauge, (bool, np.bool_)):
        raise TypeError("similarity_residual_gauge must be a boolean")
    similarity_residual_gauge = bool(similarity_residual_gauge)
    if separate_rigid_registration and similarity_residual_gauge:
        raise ValueError(
            "similarity_residual_gauge cannot be combined with active-site "
            "rigid registration"
        )

    probe = jnp.asarray(input_probe)
    if probe.ndim not in (1, 2):
        raise ValueError("input_probe must be one- or two-dimensional")
    kernel = _array("propagation_kernel", propagation_kernel, 1)
    length = _integer("window_length", window_length)
    starts = _validate_window_starts(window_starts, n_s=n_s, window_length=length)
    n_scan = starts.shape[0]
    if probe.shape[-1] != n_u or kernel.shape[0] != n_u:
        raise ValueError("input_probe and propagation_kernel must have length n_u")
    if probe.ndim == 2 and probe.shape[0] != n_scan:
        raise ValueError("two-dimensional input_probe must have one row per scan")
    if measurement is None:
        if measured_intensities is None:
            raise ValueError(
                "measured_intensities is required when measurement is not supplied"
            )
        if objective is not None:
            raise ValueError("objective requires measurement")
        measured = _array("measured_intensities", measured_intensities, 2)
        if measured.shape != (n_scan, n_u):
            raise ValueError(
                f"measured_intensities must have shape {(n_scan, n_u)}"
            )
        valid_mask = _detector_valid_mask_1d(
            detector_valid_mask, (n_scan, n_u)
        )
        _validate_masked_intensities_1d(
            "measured_intensities", measured, valid_mask
        )
        resolved_measurement = None
        resolved_objective = None
        dose_per_scan = None
        resolved_objective_id = _NORMALIZED_AMPLITUDE_OBJECTIVE_ID
        measurement_contract = _measurement_contract_1d(valid_mask)
    else:
        if measured_intensities is not None:
            raise ValueError(
                "measurement and measured_intensities are mutually exclusive"
            )
        if detector_valid_mask is not None:
            raise ValueError(
                "measurement.valid_mask and detector_valid_mask are mutually exclusive"
            )
        if objective is None:
            raise ValueError("objective is required with measurement")
        (
            resolved_measurement,
            resolved_objective,
            dose_per_scan,
        ) = _validated_measurement_objective_pair_1d(measurement, objective)
        if resolved_measurement.calibrated_signal_electrons.shape != (
            n_scan,
            n_u,
        ):
            raise ValueError(
                "measurement arrays must match the scan and detector shape "
                f"{(n_scan, n_u)}"
            )
        measured = resolved_measurement.calibrated_signal_electrons
        valid_mask = resolved_measurement.valid_mask
        resolved_objective_id = (
            "wide_angle_propagation.ptychography_objective_1d:"
            f"{resolved_objective.kind}:v1"
        )
        measurement_contract = (
            "nonnegative_total_electron_equivalent_poisson_deviance"
            if resolved_objective.kind == "poisson_deviance"
            else "heteroscedastic_poisson_gaussian_approximation"
        )
    for name, value in (
        ("input_probe", probe),
        ("propagation_kernel", kernel),
    ):
        if not np.all(np.isfinite(np.asarray(value))):
            raise ValueError(f"{name} must contain only finite values")

    batch_size = _integer("minibatch_size", minibatch_size)
    eval_batch_size = _integer("evaluation_batch_size", evaluation_batch_size)
    _positive_scalar("slice_thickness", slice_thickness)
    _positive_scalar("energy", energy)
    _positive_scalar("gradient_clip", gradient_clip)
    _positive_scalar("epsilon", epsilon)
    if not isinstance(rematerialize, (bool, np.bool_)):
        raise TypeError("rematerialize must be a boolean")
    slice_thickness = jnp.asarray(slice_thickness)
    energy = jnp.asarray(energy)
    gradient_clip = float(np.asarray(gradient_clip))
    epsilon = float(np.asarray(epsilon))
    rematerialize = bool(rematerialize)

    reference_max = float(np.max(np.asarray(reference)))
    if potential_max is None:
        resolved_max = 2.0 * max(reference_max, 1.0)
    else:
        resolved_max = float(np.asarray(potential_max))
    _positive_scalar("potential_max", resolved_max)
    if reference_max > resolved_max:
        raise ValueError("model.reference_potential exceeds potential_max")
    max_phase = (
        float(np.asarray(interaction_constant(energy)))
        * float(np.asarray(slice_thickness))
        * resolved_max
    )
    if not np.isfinite(max_phase):
        raise ValueError("the per-slice phase bound is not finite")
    if max_phase >= np.pi:
        raise ValueError(
            "potential_max violates the per-slice phase bound: "
            f"sigma * slice_thickness * potential_max = {max_phase:.6g} >= pi"
        )

    training_host, validation_host, audit_host, excluded_host = (
        _scan_partition_indices_1d(
            n_scan,
            validation_indices=validation_indices,
            audit_indices=audit_indices,
            excluded_indices=excluded_indices,
        )
    )
    _validate_detector_mask_partitions_1d(
        valid_mask,
        training_indices=training_host,
        validation_indices=validation_host,
        audit_indices=audit_host,
    )

    if scan_coordinates is None:
        coordinates_scan = (starts + length / 2) * slice_thickness
    else:
        coordinates_scan = _array("scan_coordinates", scan_coordinates, 1)
        if coordinates_scan.shape[0] != n_scan:
            raise ValueError("scan_coordinates must have length n_scan")
    if detector_angles is None:
        frequencies = jnp.fft.fftshift(jnp.fft.fftfreq(n_u, model.transverse_sampling))
        detector_theta = 1e3 * jnp.arcsin(
            jnp.clip(energy2wavelength(energy) * frequencies, -1.0, 1.0)
        )
    else:
        detector_theta = _array("detector_angles", detector_angles, 1)
        if detector_theta.shape[0] != n_u:
            raise ValueError("detector_angles must have length n_u")
    for name, value in (
        ("scan_coordinates", coordinates_scan),
        ("detector_angles", detector_theta),
    ):
        value_host = np.asarray(value)
        if np.iscomplexobj(value_host) or not np.all(np.isfinite(value_host)):
            raise ValueError(f"{name} must contain only finite real values")

    maximum_displacement = float(np.asarray(model.maximum_displacement))
    if separate_rigid_registration:
        rigid_limit = (
            min(0.15, maximum_displacement)
            if maximum_rigid_displacement is None
            else float(np.asarray(maximum_rigid_displacement))
        )
        residual_limit = (
            maximum_displacement - rigid_limit
            if maximum_residual_displacement is None
            else float(np.asarray(maximum_residual_displacement))
        )
        _positive_scalar(
            "maximum_rigid_displacement", rigid_limit, allow_zero=True
        )
        _positive_scalar(
            "maximum_residual_displacement", residual_limit, allow_zero=True
        )
        if rigid_limit + residual_limit > maximum_displacement + 1e-12:
            raise ValueError(
                "maximum rigid plus residual displacement exceeds "
                "model.maximum_displacement"
            )
        control_scale = 0.5 * residual_limit
    else:
        if maximum_rigid_displacement is not None:
            raise ValueError(
                "maximum_rigid_displacement requires separate_rigid_registration=True"
            )
        if maximum_residual_displacement is not None:
            raise ValueError(
                "maximum_residual_displacement requires "
                "separate_rigid_registration=True"
            )
        rigid_limit = 0.0
        residual_limit = maximum_displacement
        control_scale = residual_limit

    def physical_residual_controls(values: Mapping[str, Array]) -> Array:
        residual = values["controls"] * control_scale
        if separate_rigid_registration:
            _, residual = decompose_lattice_site_displacement_controls_1d(
                sites,
                residual,
                controls_s,
                controls_u,
            )
        elif similarity_residual_gauge:
            _, residual = decompose_lattice_site_similarity_controls_1d(
                sites,
                residual,
                controls_s,
                controls_u,
            )
        return residual

    def physical_rigid_displacement(values: Mapping[str, Array]) -> Array:
        return values["rigid"] * rigid_limit

    def physical_controls(values: Mapping[str, Array]) -> Array:
        return physical_residual_controls(values) + physical_rigid_displacement(
            values
        )

    def assemble(values: Mapping[str, Array]) -> Array:
        return render_lattice_site_potential_1d(
            model, values["vacancies"], physical_controls(values)
        )

    probe_rows = jnp.broadcast_to(probe, (n_scan, n_u)) if probe.ndim == 1 else probe
    if resolved_measurement is not None:
        probe_rows_host = np.asarray(probe_rows)
        incident_norm = n_u * np.sum(np.abs(probe_rows_host) ** 2, axis=1)
        if np.any(~np.isfinite(incident_norm)) or np.any(incident_norm <= 0.0):
            raise ValueError(
                "every probe row must have finite positive norm for count conversion"
            )

    def batch_loss(
        values: Mapping[str, Array],
        batch_indices: Array,
    ) -> Array:
        prediction = simulate_glancing_scan_1d(
            assemble(values),
            probe_rows[batch_indices],
            starts[batch_indices],
            length,
            kernel,
            slice_thickness,
            energy,
            rematerialize=rematerialize,
        )
        if resolved_measurement is None:
            return normalized_amplitude_loss_1d(
                prediction,
                measured[batch_indices],
                epsilon=epsilon,
                detector_valid_mask=(
                    None if valid_mask is None else valid_mask[batch_indices]
                ),
            )
        assert resolved_objective is not None and dose_per_scan is not None
        predicted_signal = _expected_signal_electrons_1d(
            prediction,
            probe_rows[batch_indices],
            dose_per_scan[batch_indices],
            resolved_objective.relative_signal_scale,
        )
        batch_measurement = PtychographyMeasurement1D(
            calibrated_signal_electrons=(
                resolved_measurement.calibrated_signal_electrons[batch_indices]
            ),
            observed_total_electrons=(
                resolved_measurement.observed_total_electrons[batch_indices]
            ),
            valid_mask=resolved_measurement.valid_mask[batch_indices],
            calibrated_dark_electrons_per_pixel=(
                resolved_measurement.calibrated_dark_electrons_per_pixel[
                    batch_indices
                ]
            ),
            calibrated_read_noise_std_electrons=(
                resolved_measurement.calibrated_read_noise_std_electrons[
                    batch_indices
                ]
            ),
            calibration_id=resolved_measurement.calibration_id,
        )
        return _ptychography_objective_from_signal_1d(
            predicted_signal, batch_measurement, resolved_objective
        )

    batch_value_and_grad = jax.value_and_grad(batch_loss)

    def predict_batch(
        potential: Array, batch_indices: Array
    ) -> Array:
        return simulate_glancing_scan_1d(
            potential,
            probe_rows[batch_indices],
            starts[batch_indices],
            length,
            kernel,
            slice_thickness,
            energy,
            rematerialize=rematerialize,
        )

    optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip), optax.adam(1.0)
    )

    def train_step(
        values: Mapping[str, Array],
        state: Any,
        batch_indices: Array,
        learning_rates: Mapping[str, Array],
        active_groups: Mapping[str, Array],
    ) -> tuple[Mapping[str, Array], Any, Array, Mapping[str, Array]]:
        loss, gradient = batch_value_and_grad(values, batch_indices)
        gradient = {
            key: gradient[key] * active_groups[key] for key in gradient
        }
        parameter_updates, state = optimizer.update(gradient, state, values)
        parameter_updates = {
            key: parameter_updates[key]
            * learning_rates[key]
            * active_groups[key]
            for key in parameter_updates
        }
        values = optax.apply_updates(values, parameter_updates)
        values = {
            "vacancies": jnp.clip(values["vacancies"], 0.0, 1.0),
            "controls": jnp.clip(values["controls"], -1.0, 1.0),
            "rigid": jnp.clip(values["rigid"], -1.0, 1.0),
        }
        return values, state, loss, gradient

    # Compile against canonical, fixed-shape examples.  No optimizer update is
    # executed here, and neither optimizer state nor random state is retained.
    sample_parameters = {
        "vacancies": jnp.zeros((n_site,), dtype=reference.dtype),
        "controls": jnp.zeros(control_shape, dtype=reference.dtype),
        "rigid": jnp.zeros((2,), dtype=reference.dtype),
    }
    sample_optimizer_state = optimizer.init(sample_parameters)
    training_example = np.resize(training_host, batch_size)
    evaluation_example = np.resize(training_host, eval_batch_size)
    sample_learning_rates = {
        key: jnp.asarray(0.0, dtype=reference.dtype) for key in sample_parameters
    }
    sample_active_groups = {
        key: jnp.asarray(1.0, dtype=reference.dtype) for key in sample_parameters
    }
    assemble_compiled = jax.jit(assemble).lower(sample_parameters).compile()
    sample_potential = assemble_compiled(sample_parameters)
    train_step_compiled = jax.jit(train_step).lower(
        sample_parameters,
        sample_optimizer_state,
        jnp.asarray(training_example),
        sample_learning_rates,
        sample_active_groups,
    ).compile()
    predict_batch_compiled = jax.jit(predict_batch).lower(
        sample_potential,
        jnp.asarray(evaluation_example),
    ).compile()
    # Execute and synchronize each executable once so loading constants and
    # transferring closed-over arrays cannot leak into the first run timer.
    # The dummy optimizer output is discarded and all learning rates are zero.
    sample_train_output = train_step_compiled(
        sample_parameters,
        sample_optimizer_state,
        jnp.asarray(training_example),
        sample_learning_rates,
        sample_active_groups,
    )
    sample_prediction = predict_batch_compiled(
        sample_potential,
        jnp.asarray(evaluation_example),
    )
    jax.block_until_ready(
        (sample_potential, sample_train_output, sample_prediction)
    )
    training_indices_array = jnp.asarray(training_host, dtype=jnp.int32)
    validation_indices_array = jnp.asarray(validation_host, dtype=jnp.int32)
    audit_indices_array = jnp.asarray(audit_host, dtype=jnp.int32)
    excluded_indices_array = jnp.asarray(excluded_host, dtype=jnp.int32)
    problem_arrays = {
        "model.reference_potential": model.reference_potential,
        "model.site_coordinates": model.site_coordinates,
        "model.site_patches": model.site_patches,
        "model.patch_starts": model.patch_starts,
        "model.control_coordinates_s": model.control_coordinates_s,
        "model.control_coordinates_u": model.control_coordinates_u,
        "model.axial_sampling": model.axial_sampling,
        "model.transverse_sampling": model.transverse_sampling,
        "model.maximum_displacement": model.maximum_displacement,
        "input_probe": probe,
        "probe_rows": probe_rows,
        "window_starts": starts,
        "propagation_kernel": kernel,
        "slice_thickness": slice_thickness,
        "energy": energy,
        "measured_intensities": measured,
        "scan_coordinates": coordinates_scan,
        "detector_angles": detector_theta,
        "training_indices": training_indices_array,
        "validation_indices": validation_indices_array,
        "audit_indices": audit_indices_array,
        "excluded_indices": excluded_indices_array,
    }
    if support_contract is not None:
        problem_arrays.update(
            {
                "support.all_site_coordinates": (
                    support_contract.all_site_coordinates
                ),
                "support.site_center_indices": (
                    support_contract.site_center_indices
                ),
                "support.site_patch_starts": (
                    support_contract.site_patch_starts
                ),
                "support.site_patch_shapes": (
                    support_contract.site_patch_shapes
                ),
                "support.target_pixel_mask": (
                    support_contract.target_pixel_mask
                ),
                "support.forward_pixel_mask": (
                    support_contract.forward_pixel_mask
                ),
                "support.site_role_codes": support_contract.site_role_codes,
                "support.modeled_site_indices": (
                    support_contract.modeled_site_indices
                ),
                "support.target_influence_mask": (
                    support_contract.target_influence_mask
                ),
                "support.nuisance_influence_mask": (
                    support_contract.nuisance_influence_mask
                ),
            }
        )
    if valid_mask is not None:
        problem_arrays["detector_valid_mask"] = valid_mask
    if resolved_measurement is not None:
        assert resolved_objective is not None and dose_per_scan is not None
        problem_arrays.update(
            {
                "measurement.calibrated_signal_electrons": (
                    resolved_measurement.calibrated_signal_electrons
                ),
                "measurement.observed_total_electrons": (
                    resolved_measurement.observed_total_electrons
                ),
                "measurement.calibrated_dark_electrons_per_pixel": (
                    resolved_measurement.calibrated_dark_electrons_per_pixel
                ),
                "measurement.calibrated_read_noise_std_electrons": (
                    resolved_measurement.calibrated_read_noise_std_electrons
                ),
                "objective.electrons_per_pattern": dose_per_scan,
            }
        )
    objective_options = {
        "objective_id": resolved_objective_id,
        "measurement_contract": measurement_contract,
    }
    if resolved_measurement is not None:
        assert resolved_objective is not None
        objective_options.update(
            {
                "calibration_id": resolved_measurement.calibration_id,
                "objective_kind": resolved_objective.kind,
                "minimum_expected_electrons": (
                    resolved_objective.minimum_expected_electrons
                ),
                "relative_signal_scale": (
                    resolved_objective.relative_signal_scale
                ),
                "relative_signal_scale_fitted": False,
            }
        )
    reconstruction_problem_id = _reconstruction_problem_id_1d(
        arrays=problem_arrays,
        options={
            "reconstructor_id": _LATTICE_SITE_RECONSTRUCTOR_ID,
            **objective_options,
            "detector_valid_mask_mode": (
                "explicit_boolean" if valid_mask is not None else "none_all_valid"
            ),
            "window_length": length,
            "potential_max": resolved_max,
            "separate_rigid_registration": separate_rigid_registration,
            "similarity_residual_gauge": similarity_residual_gauge,
            "maximum_rigid_displacement": rigid_limit,
            "maximum_residual_displacement": residual_limit,
            "control_scale": control_scale,
            "minibatch_size": batch_size,
            "evaluation_batch_size": eval_batch_size,
            "gradient_clip": gradient_clip,
            "epsilon": epsilon,
            "rematerialize": rematerialize,
            "require_complete_material_scope": (
                require_complete_material_scope
            ),
            "material_scope_complete": material_scope_complete,
            "material_scope_fully_parameterized": (
                material_scope_fully_parameterized
            ),
            "support_contract_id": (
                support_contract.contract_id
                if support_contract is not None
                else None
            ),
        },
    )
    preparation_time = perf_counter() - preparation_start
    metadata = MappingProxyType(
        {
            "prepared_api_version": 3,
            "reconstruction_problem_id": reconstruction_problem_id,
            "reconstructor_id": _LATTICE_SITE_RECONSTRUCTOR_ID,
            "jax_backend": jax.default_backend(),
            "jax_devices": sorted(str(device) for device in reference.devices()),
            "potential_dtype": str(reference.dtype),
            "probe_dtype": str(probe.dtype),
            "compiled_minibatch_size": batch_size,
            "compiled_evaluation_batch_size": eval_batch_size,
            "preparation_time_s": preparation_time,
            "similarity_residual_gauge": similarity_residual_gauge,
            "displacement_gauge": (
                "equal_candidate_site_mean"
                if separate_rigid_registration
                else (
                    "translation_rotation_isotropic_dilation"
                    if similarity_residual_gauge
                    else "legacy_total_controls"
                )
            ),
            "objective_id": resolved_objective_id,
            "measurement_contract": measurement_contract,
            "objective_kind": (
                resolved_objective.kind
                if resolved_objective is not None
                else "normalized_amplitude"
            ),
            "poisson_count_likelihood_supported": (
                False
            ),
            "poisson_deviance_supported": (
                resolved_objective is not None
                and resolved_objective.kind == "poisson_deviance"
            ),
            "integer_count_contract_enforced": False,
            "read_noise_likelihood_supported": (
                resolved_objective is not None
                and resolved_objective.kind == "poisson_gaussian_nll"
            ),
            "likelihood_interpretation": (
                "heteroscedastic Gaussian approximation, not the exact "
                "Poisson-Gaussian convolution"
                if resolved_objective is not None
                and resolved_objective.kind == "poisson_gaussian_nll"
                else (
                    "Poisson deviance for calibrated non-negative "
                    "electron-equivalent totals; not an exact integer-count "
                    "likelihood without an independent raw-count contract"
                    if resolved_objective is not None
                    else "legacy normalized-amplitude loss"
                )
            ),
            "gaussian_log_variance_reference": (
                "minimum_expected_electrons; changes only an additive constant"
                if resolved_objective is not None
                and resolved_objective.kind == "poisson_gaussian_nll"
                else None
            ),
            "calibration_id": (
                resolved_measurement.calibration_id
                if resolved_measurement is not None
                else None
            ),
            "calibration_benchmark_required_for_trust": True,
            "structural_trust_from_measurement_objective": False,
            "material_scope_complete": material_scope_complete,
            "support_contract_id": (
                support_contract.contract_id
                if support_contract is not None
                else None
            ),
            "support_contract_required": require_complete_material_scope,
            "n_target_sites": int(
                np.count_nonzero(
                    modeled_site_roles == int(LatticeSiteRole1D.TARGET)
                )
            ),
            "n_nuisance_sites": int(
                np.count_nonzero(
                    modeled_site_roles == int(LatticeSiteRole1D.NUISANCE)
                )
            ),
            "n_fixed_known_sites": int(
                0
                if support_contract is None
                else np.count_nonzero(
                    np.asarray(support_contract.site_role_codes)
                    == int(LatticeSiteRole1D.FIXED_KNOWN)
                )
            ),
            "n_below_interaction_budget_sites": int(
                0
                if support_contract is None
                else np.count_nonzero(
                    np.asarray(support_contract.site_role_codes)
                    == int(LatticeSiteRole1D.BELOW_INTERACTION_BUDGET)
                )
            ),
            "fixed_material_provenance_verified": False,
            "material_scope_fully_parameterized": (
                material_scope_fully_parameterized
            ),
            "relative_signal_scale_fitted": False,
            "relative_signal_scale": (
                resolved_objective.relative_signal_scale
                if resolved_objective is not None
                else None
            ),
            "detector_valid_mask_present": valid_mask is not None,
            "detector_valid_mask_sha256": (
                _array_sha256_1d(valid_mask) if valid_mask is not None else None
            ),
            "n_valid_detector_observations": (
                int(np.count_nonzero(np.asarray(valid_mask)))
                if valid_mask is not None
                else int(measured.size)
            ),
        }
    )
    prepared = PreparedLatticeSiteReconstruction1D(
        model=model,
        input_probe=probe,
        probe_rows=probe_rows,
        window_starts=starts,
        window_length=length,
        propagation_kernel=kernel,
        slice_thickness=slice_thickness,
        energy=energy,
        measured_intensities=measured,
        measurement=resolved_measurement,
        objective=resolved_objective,
        detector_valid_mask=valid_mask,
        scan_coordinates=coordinates_scan,
        detector_angles=detector_theta,
        training_indices=training_indices_array,
        validation_indices=validation_indices_array,
        audit_indices=audit_indices_array,
        excluded_indices=excluded_indices_array,
        potential_max=resolved_max,
        maximum_phase_per_slice=max_phase,
        separate_rigid_registration=separate_rigid_registration,
        similarity_residual_gauge=similarity_residual_gauge,
        maximum_rigid_displacement=rigid_limit,
        maximum_residual_displacement=residual_limit,
        control_scale=control_scale,
        minibatch_size=batch_size,
        evaluation_batch_size=eval_batch_size,
        gradient_clip=gradient_clip,
        epsilon=epsilon,
        rematerialize=rematerialize,
        objective_id=resolved_objective_id,
        reconstruction_problem_id=reconstruction_problem_id,
        reconstructor_id=_LATTICE_SITE_RECONSTRUCTOR_ID,
        preparation_time_s=preparation_time,
        metadata=metadata,
        _assemble=assemble_compiled,
        _train_step=train_step_compiled,
        _predict_batch=predict_batch_compiled,
        _optimizer=optimizer,
    )
    return replace(
        prepared,
        _static_contract=_make_prepared_static_contract_1d(prepared),
    )


def run_prepared_lattice_site_reconstruction_1d(
    prepared: PreparedLatticeSiteReconstruction1D,
    *,
    initial_vacancy_fractions: Any | None = None,
    initial_displacement_controls: Any | None = None,
    initial_rigid_displacement: Any | None = None,
    learning_rate_start: Any = 2e-2,
    learning_rate_end: Any = 2e-4,
    updates: int = 500,
    validation_interval: int = 25,
    training_diagnostic_scan_count: int | None = None,
    seed: int = 0,
    progress: bool = False,
    progress_description: str = "lattice-site reconstruction",
    checkpoint_interval: int | None = None,
    convergence: ConvergenceOptions1D | None = None,
    optimization: LatticeOptimizationOptions1D | None = None,
) -> LatticeSiteReconstruction1D:
    """Run one independent initialization of a prepared inverse problem."""
    run_start = perf_counter()
    if not isinstance(prepared, PreparedLatticeSiteReconstruction1D):
        raise TypeError(
            "prepared must be a PreparedLatticeSiteReconstruction1D instance"
        )
    _validate_prepared_static_contract_1d(prepared)
    try:
        import optax
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "run_prepared_lattice_site_reconstruction_1d requires Optax; "
            "install the 'ptychography' extra"
        ) from exc

    model = prepared.model
    reference, sites, _, _, controls_s, controls_u = _validate_lattice_site_model_1d(
        model
    )
    n_site = sites.shape[0]
    support_contract = model.support_contract
    if support_contract is None:
        modeled_site_roles = np.empty(0, dtype=np.int8)
        support_contract_id = None
        material_scope_complete = False
        material_scope_fully_parameterized = False
    else:
        support_contract = validate_lattice_site_support_contract_1d(
            support_contract,
            strict=False,
        )
        modeled_site_roles = np.asarray(support_contract.site_role_codes)[
            np.asarray(support_contract.modeled_site_indices)
        ].astype(np.int8, copy=True)
        support_contract_id = support_contract.contract_id
        material_scope_complete = bool(
            support_contract.strict_requirements_satisfied
        )
        material_scope_fully_parameterized = bool(
            material_scope_complete
            and not np.any(
                np.asarray(support_contract.site_role_codes)
                == int(LatticeSiteRole1D.FIXED_KNOWN)
            )
        )
    control_shape = (controls_s.shape[0], controls_u.shape[0], 2)
    probe = prepared.input_probe
    probe_rows = prepared.probe_rows
    starts = prepared.window_starts
    slice_thickness = prepared.slice_thickness
    energy = prepared.energy
    measured = prepared.measured_intensities
    measurement = prepared.measurement
    objective = prepared.objective
    dose_per_scan = (
        None if objective is None else jnp.asarray(objective.electrons_per_pattern)
    )
    valid_mask = prepared.detector_valid_mask
    coordinates_scan = prepared.scan_coordinates
    detector_theta = prepared.detector_angles
    n_scan, n_u = measured.shape
    training_host = np.asarray(prepared.training_indices, dtype=np.int64)
    validation_host = np.asarray(prepared.validation_indices, dtype=np.int64)
    audit_host = np.asarray(prepared.audit_indices, dtype=np.int64)
    excluded_host = np.asarray(prepared.excluded_indices, dtype=np.int64)
    training_diagnostic_host, training_diagnostic_selection = (
        _geometry_stratified_training_diagnostic_indices_1d(
            training_host,
            coordinates_scan,
            training_diagnostic_scan_count,
            validation_available=bool(validation_host.size),
        )
    )
    phase_timings_s = {
        "optimizer_chunks": 0.0,
        "rendering": 0.0,
        "training_diagnostics": 0.0,
        "validation": 0.0,
        "final_prediction": 0.0,
        "final_full_training_reduction": 0.0,
        "audit_reduction": 0.0,
    }
    initialization_rendering_time_s = 0.0
    optimizer_chunk_count = 0
    resolved_max = prepared.potential_max
    max_phase = prepared.maximum_phase_per_slice
    separate_rigid_registration = prepared.separate_rigid_registration
    similarity_residual_gauge = prepared.similarity_residual_gauge
    rigid_limit = prepared.maximum_rigid_displacement
    residual_limit = prepared.maximum_residual_displacement
    control_scale = prepared.control_scale
    maximum_displacement = float(np.asarray(model.maximum_displacement))
    batch_size = prepared.minibatch_size
    eval_batch_size = prepared.evaluation_batch_size
    gradient_clip = prepared.gradient_clip
    epsilon = prepared.epsilon

    n_updates = _integer("updates", updates)
    metric_interval = _integer("validation_interval", validation_interval)
    seed_value = operator.index(seed)
    _positive_scalar("learning_rate_start", learning_rate_start)
    _positive_scalar("learning_rate_end", learning_rate_end)
    if float(np.asarray(learning_rate_end)) > float(np.asarray(learning_rate_start)):
        raise ValueError("learning_rate_end must not exceed learning_rate_start")
    _validate_progress(progress, progress_description)
    convergence_options = _validate_convergence_options(convergence)
    optimization_options = _validate_lattice_optimization_options(optimization)
    if checkpoint_interval is None:
        checkpoint_stride = 0
    else:
        checkpoint_stride = _integer("checkpoint_interval", checkpoint_interval)

    if initial_vacancy_fractions is None:
        initial_vacancies = jnp.zeros((n_site,), dtype=reference.dtype)
    else:
        initial_vacancies = _array(
            "initial_vacancy_fractions", initial_vacancy_fractions, 1
        )
    if initial_displacement_controls is None:
        initial_controls = jnp.zeros(control_shape, dtype=reference.dtype)
    else:
        initial_controls = _array(
            "initial_displacement_controls", initial_displacement_controls, 3
        )
    if initial_rigid_displacement is None:
        initial_rigid = jnp.zeros((2,), dtype=reference.dtype)
    else:
        initial_rigid = _array(
            "initial_rigid_displacement", initial_rigid_displacement, 1
        )
    if initial_vacancies.shape != (n_site,):
        raise ValueError(f"initial_vacancy_fractions must have shape {(n_site,)}")
    if initial_controls.shape != control_shape:
        raise ValueError(
            f"initial_displacement_controls must have shape {control_shape}"
        )
    if initial_rigid.shape != (2,):
        raise ValueError("initial_rigid_displacement must have shape (2,)")
    for name, value in (
        ("initial_vacancy_fractions", initial_vacancies),
        ("initial_displacement_controls", initial_controls),
        ("initial_rigid_displacement", initial_rigid),
    ):
        if jnp.iscomplexobj(value):
            raise TypeError(f"{name} must be real")
        value_host = np.asarray(value)
        if not np.all(np.isfinite(value_host)):
            raise ValueError(f"{name} must contain only finite values")
    initial_vacancies_host = np.asarray(initial_vacancies)
    if np.any(initial_vacancies_host < 0.0) or np.any(
        initial_vacancies_host > 1.0
    ):
        raise ValueError("initial_vacancy_fractions must lie in [0, 1]")

    safe_control_scale = control_scale if control_scale > 0.0 else 1.0
    safe_rigid_scale = rigid_limit if rigid_limit > 0.0 else 1.0
    if separate_rigid_registration:
        initial_rigid, initial_controls = (
            decompose_lattice_site_displacement_controls_1d(
                sites,
                initial_controls,
                controls_s,
                controls_u,
                rigid_displacement=initial_rigid,
            )
        )
    elif similarity_residual_gauge:
        _, initial_controls = decompose_lattice_site_similarity_controls_1d(
            sites,
            initial_controls,
            controls_s,
            controls_u,
        )
    initial_controls_host = np.asarray(initial_controls)
    initial_rigid_host = np.asarray(initial_rigid)
    if not np.all(np.isfinite(initial_controls_host)) or np.any(
        np.abs(initial_controls_host) > control_scale + 1e-12
    ):
        raise ValueError(
            "initial_displacement_controls exceed the raw residual-control bound"
        )
    if not np.all(np.isfinite(initial_rigid_host)) or np.any(
        np.abs(initial_rigid_host) > rigid_limit + 1e-12
    ):
        raise ValueError("initial_rigid_displacement exceeds its bound")
    parameters = {
        "vacancies": initial_vacancies.astype(reference.dtype),
        "controls": (initial_controls / safe_control_scale).astype(reference.dtype),
        "rigid": (initial_rigid / safe_rigid_scale).astype(reference.dtype),
    }

    def physical_residual_controls(values: Mapping[str, Array]) -> Array:
        residual = values["controls"] * control_scale
        if separate_rigid_registration:
            _, residual = decompose_lattice_site_displacement_controls_1d(
                sites,
                residual,
                controls_s,
                controls_u,
            )
        elif similarity_residual_gauge:
            _, residual = decompose_lattice_site_similarity_controls_1d(
                sites,
                residual,
                controls_s,
                controls_u,
            )
        return residual

    def physical_rigid_displacement(values: Mapping[str, Array]) -> Array:
        return values["rigid"] * rigid_limit

    def physical_controls(values: Mapping[str, Array]) -> Array:
        return physical_residual_controls(values) + physical_rigid_displacement(
            values
        )

    initial_residual_controls = physical_residual_controls(parameters)
    # All user-provided initial arrays and bounds have been checked before the
    # first prepared executable is invoked.
    phase_start = perf_counter()
    initial_potential = prepared._assemble(parameters)
    initial_potential_host = np.asarray(initial_potential)
    initialization_rendering_time_s = perf_counter() - phase_start
    if not np.all(np.isfinite(initial_potential_host)):
        raise ValueError("the initial lattice-site potential is not finite")
    if float(np.max(initial_potential_host)) > resolved_max:
        raise ValueError("the initial lattice-site potential exceeds potential_max")

    alpha = float(np.asarray(learning_rate_end)) / float(
        np.asarray(learning_rate_start)
    )
    base_learning_rate = optax.cosine_decay_schedule(
        init_value=learning_rate_start,
        decay_steps=max(n_updates, 1),
        alpha=alpha,
    )
    optimizer_state = prepared._optimizer.init(parameters)
    rng = np.random.default_rng(seed_value)
    assemble_jit = prepared._assemble
    train_step = prepared._train_step
    predict_batch = prepared._predict_batch

    if optimization_options.mode == "staged":
        # Flooring each allocation preserves at least one joint update because
        # the validated stage fractions sum to less than one.  Inactive groups
        # do not consume an otherwise useful optimization stage.
        rigid_stage_updates = (
            int(np.floor(n_updates * optimization_options.rigid_stage_fraction))
            if separate_rigid_registration and rigid_limit > 0.0
            else 0
        )
        vacancy_stage_updates = int(
            np.floor(n_updates * optimization_options.vacancy_stage_fraction)
        )
        residual_stage_updates = (
            int(
                np.floor(
                    n_updates * optimization_options.residual_stage_fraction
                )
            )
            if residual_limit > 0.0
            else 0
        )
    else:
        rigid_stage_updates = vacancy_stage_updates = residual_stage_updates = 0
    rigid_stage_end = rigid_stage_updates
    vacancy_stage_end = rigid_stage_end + vacancy_stage_updates
    residual_stage_end = vacancy_stage_end + residual_stage_updates

    def stage_and_groups(update: int) -> tuple[str, Mapping[str, Array]]:
        if update <= rigid_stage_end and rigid_stage_updates:
            stage = "site_translation"
            active = (
                False,
                not separate_rigid_registration,
                separate_rigid_registration,
            )
        elif update <= vacancy_stage_end and vacancy_stage_updates:
            stage = "vacancies"
            active = (True, False, False)
        elif update <= residual_stage_end and residual_stage_updates:
            stage = "residual"
            active = (False, True, False)
        else:
            stage = "joint"
            active = (True, True, separate_rigid_registration)
        return stage, {
            "vacancies": jnp.asarray(active[0], dtype=reference.dtype),
            "controls": jnp.asarray(active[1], dtype=reference.dtype),
            "rigid": jnp.asarray(active[2], dtype=reference.dtype),
        }

    def group_learning_rates(update: int) -> Mapping[str, Array]:
        rate = base_learning_rate(update - 1)
        return {
            "vacancies": jnp.asarray(
                rate * optimization_options.vacancy_learning_rate_scale,
                dtype=reference.dtype,
            ),
            "controls": jnp.asarray(
                rate * optimization_options.residual_learning_rate_scale,
                dtype=reference.dtype,
            ),
            "rigid": jnp.asarray(
                rate * optimization_options.rigid_learning_rate_scale,
                dtype=reference.dtype,
            ),
        }

    def predict_indices(potential: Array, indices: np.ndarray) -> Array:
        predictions = []
        for begin in range(0, len(indices), eval_batch_size):
            batch_indices = indices[begin : begin + eval_batch_size]
            actual_size = batch_indices.size
            if actual_size < eval_batch_size:
                batch_indices = np.pad(
                    batch_indices,
                    (0, eval_batch_size - actual_size),
                    mode="edge",
                )
            batch_prediction = predict_batch(
                potential,
                jnp.asarray(batch_indices),
            )
            predictions.append(batch_prediction[:actual_size])
        return jnp.concatenate(predictions, axis=0)

    def loss_from_prediction(
        prediction: Array, indices: np.ndarray
    ) -> float:
        if measurement is not None:
            assert objective is not None and dose_per_scan is not None
            device_indices = jnp.asarray(indices)
            predicted_signal = _expected_signal_electrons_1d(
                prediction,
                probe_rows[device_indices],
                dose_per_scan[device_indices],
                objective.relative_signal_scale,
            )
            selected_measurement = PtychographyMeasurement1D(
                calibrated_signal_electrons=(
                    measurement.calibrated_signal_electrons[device_indices]
                ),
                observed_total_electrons=(
                    measurement.observed_total_electrons[device_indices]
                ),
                valid_mask=measurement.valid_mask[device_indices],
                calibrated_dark_electrons_per_pixel=(
                    measurement.calibrated_dark_electrons_per_pixel[
                        device_indices
                    ]
                ),
                calibrated_read_noise_std_electrons=(
                    measurement.calibrated_read_noise_std_electrons[
                        device_indices
                    ]
                ),
                calibration_id=measurement.calibration_id,
            )
            return float(
                np.asarray(
                    _ptychography_objective_from_signal_1d(
                        predicted_signal, selected_measurement, objective
                    )
                )
            )
        return float(
            np.asarray(
                normalized_amplitude_loss_1d(
                    prediction,
                    measured[jnp.asarray(indices)],
                    epsilon=epsilon,
                    detector_valid_mask=(
                        None
                        if valid_mask is None
                        else valid_mask[jnp.asarray(indices)]
                    ),
                )
            )
        )

    def evaluate(potential: Array, indices: np.ndarray) -> float:
        return loss_from_prediction(predict_indices(potential, indices), indices)

    update_history: list[int] = []
    elapsed_history: list[float] = []
    training_history: list[float] = []
    validation_history: list[float] = []
    gradient_norm_history: list[float] = []
    normalized_step_history: list[float] = []
    active_bound_fraction_history: list[float] = []
    optimization_stage_history: list[str] = []
    optimization_start = perf_counter()

    def active_bound_fraction(values: Mapping[str, Array]) -> float:
        tolerance = 1e-6
        active_vacancies = (values["vacancies"] <= tolerance) | (
            values["vacancies"] >= 1.0 - tolerance
        )
        active_controls = jnp.abs(values["controls"]) >= 1.0 - tolerance
        count = jnp.count_nonzero(active_vacancies) + jnp.count_nonzero(
            active_controls
        )
        total = values["vacancies"].size + values["controls"].size
        if separate_rigid_registration:
            active_rigid = jnp.abs(values["rigid"]) >= 1.0 - tolerance
            count = count + jnp.count_nonzero(active_rigid)
            total += values["rigid"].size
        return float(np.asarray(count / total))

    def record(
        update: int,
        values: Mapping[str, Array],
        *,
        gradient_norm: float,
        normalized_step: float,
        stage: str,
    ) -> tuple[float, float]:
        phase_start = perf_counter()
        potential = assemble_jit(values)
        jax.block_until_ready(potential)
        phase_timings_s["rendering"] += perf_counter() - phase_start

        phase_start = perf_counter()
        training_loss = evaluate(potential, training_diagnostic_host)
        phase_timings_s["training_diagnostics"] += perf_counter() - phase_start

        phase_start = perf_counter()
        validation_loss = (
            evaluate(potential, validation_host)
            if validation_host.size
            else float("nan")
        )
        phase_timings_s["validation"] += perf_counter() - phase_start
        if not np.isfinite(training_loss) or (
            validation_host.size and not np.isfinite(validation_loss)
        ):
            raise FloatingPointError(
                f"non-finite reconstruction loss at update {update}"
            )
        update_history.append(update)
        elapsed_history.append(perf_counter() - optimization_start)
        training_history.append(training_loss)
        validation_history.append(validation_loss)
        gradient_norm_history.append(gradient_norm)
        normalized_step_history.append(normalized_step)
        active_bound_fraction_history.append(active_bound_fraction(values))
        optimization_stage_history.append(stage)
        return training_loss, validation_loss

    training_loss, validation_loss = record(
        0,
        parameters,
        gradient_norm=float("nan"),
        normalized_step=float("nan"),
        stage="initial",
    )
    best_metric = validation_loss if validation_host.size else training_loss
    best_parameters = parameters
    best_update = 0
    meaningful_best_metric = best_metric
    stale_evaluations = 0
    last_evaluated_parameters = parameters
    completed_updates = 0
    converged = False
    stop_reason = "maximum_updates"
    last_evaluated_stage = "initial"
    checkpoint_updates: list[int] = []
    vacancy_checkpoints: list[Array] = []
    control_checkpoints: list[Array] = []
    rigid_checkpoints: list[Array] = []

    def checkpoint(update: int, values: Mapping[str, Array]) -> None:
        checkpoint_updates.append(update)
        vacancy_checkpoints.append(values["vacancies"])
        control_checkpoints.append(physical_residual_controls(values))
        rigid_checkpoints.append(physical_rigid_displacement(values))

    if checkpoint_stride:
        checkpoint(0, parameters)

    optimizer_chunk_start = perf_counter()
    for update in _update_iterator(
        n_updates,
        progress=progress,
        description=progress_description,
    ):
        stage, active_groups = stage_and_groups(update)
        batch_indices = rng.choice(
            training_host,
            size=batch_size,
            replace=training_host.size < batch_size,
        )
        parameters, optimizer_state, _, gradient = train_step(
            parameters,
            optimizer_state,
            jnp.asarray(batch_indices),
            group_learning_rates(update),
            active_groups,
        )
        completed_updates = update
        if checkpoint_stride and (
            update % checkpoint_stride == 0 or update == n_updates
        ):
            checkpoint(update, parameters)

        if update % metric_interval == 0 or update == n_updates:
            gradient_norm = float(np.asarray(optax.global_norm(gradient)))
            squared_step = sum(
                jnp.sum((parameters[key] - last_evaluated_parameters[key]) ** 2)
                for key in parameters
            )
            n_parameters_total = sum(value.size for value in parameters.values())
            normalized_step = float(
                np.asarray(jnp.sqrt(squared_step / n_parameters_total))
            )
            phase_timings_s["optimizer_chunks"] += (
                perf_counter() - optimizer_chunk_start
            )
            optimizer_chunk_count += 1
            if not np.isfinite(gradient_norm) or not np.isfinite(normalized_step):
                raise FloatingPointError(
                    f"non-finite optimization diagnostic at update {update}"
                )
            training_loss, validation_loss = record(
                update,
                parameters,
                gradient_norm=gradient_norm,
                normalized_step=normalized_step,
                stage=stage,
            )
            metric = validation_loss if validation_host.size else training_loss
            if metric < best_metric:
                best_metric = metric
                best_parameters = parameters
                best_update = update
            relative_delta = convergence_options.relative_min_delta
            if stage != last_evaluated_stage:
                meaningful_best_metric = metric
                stale_evaluations = 0
                last_evaluated_stage = stage
            else:
                meaningful_threshold = meaningful_best_metric * (
                    1.0 - relative_delta
                )
                if metric < meaningful_threshold:
                    meaningful_best_metric = metric
                    stale_evaluations = 0
                else:
                    stale_evaluations += 1
            last_evaluated_parameters = parameters

            minimum_updates_reached = (
                update >= convergence_options.min_updates and stage == "joint"
            )
            if (
                minimum_updates_reached
                and convergence_options.target_loss is not None
                and metric <= convergence_options.target_loss
            ):
                converged = True
                stop_reason = "target_loss"
                if checkpoint_stride and checkpoint_updates[-1] != update:
                    checkpoint(update, parameters)
                break
            optimizer_chunk_start = perf_counter()
            if (
                minimum_updates_reached
                and stale_evaluations
                >= convergence_options.patience_evaluations
                and normalized_step
                <= convergence_options.normalized_step_tolerance
            ):
                converged = True
                stop_reason = "plateau"
                if checkpoint_stride and checkpoint_updates[-1] != update:
                    checkpoint(update, parameters)
                break

    best_residual_controls = physical_residual_controls(best_parameters)
    best_rigid = physical_rigid_displacement(best_parameters)
    best_controls = best_residual_controls + best_rigid
    phase_start = perf_counter()
    best_potential = assemble_jit(best_parameters)
    jax.block_until_ready(best_potential)
    phase_timings_s["rendering"] += perf_counter() - phase_start
    all_indices = np.arange(n_scan, dtype=np.int64)
    phase_start = perf_counter()
    predicted = predict_indices(best_potential, all_indices)
    predicted_signal = (
        _expected_signal_electrons_1d(
            predicted,
            probe_rows,
            dose_per_scan,
            objective.relative_signal_scale,
        )
        if objective is not None and dose_per_scan is not None
        else None
    )
    prediction_readiness = [predicted]
    if predicted_signal is not None:
        prediction_readiness.append(predicted_signal)
    jax.block_until_ready(tuple(prediction_readiness))
    phase_timings_s["final_prediction"] += perf_counter() - phase_start

    phase_start = perf_counter()
    final_full_training_loss = loss_from_prediction(
        predicted[jnp.asarray(training_host)], training_host
    )
    phase_timings_s["final_full_training_reduction"] += (
        perf_counter() - phase_start
    )
    phase_start = perf_counter()
    audit_loss = (
        loss_from_prediction(
            predicted[jnp.asarray(audit_host)], audit_host
        )
        if audit_host.size
        else float("nan")
    )
    phase_timings_s["audit_reduction"] += perf_counter() - phase_start
    phase_start = perf_counter()
    site_displacements = lattice_site_displacements_1d(
        sites, best_controls, controls_s, controls_u
    )
    jax.block_until_ready(site_displacements)
    phase_timings_s["final_parameter_diagnostics"] = (
        perf_counter() - phase_start
    )
    best_total_displacement_bound_fraction = (
        float(
            np.asarray(
                jnp.mean(
                    jnp.any(
                        jnp.abs(site_displacements)
                        >= maximum_displacement - 1e-6,
                        axis=1,
                    )
                )
            )
        )
        if maximum_displacement > 0.0
        else 0.0
    )
    n_control_parameters = int(np.prod(control_shape))
    n_registration_parameters = 2 if separate_rigid_registration else 0
    n_removed_similarity_dof = 4 if similarity_residual_gauge else 0
    n_residual_control_dof = n_control_parameters - (
        2 if separate_rigid_registration else n_removed_similarity_dof
    )
    best_control_bound_fraction = float(
        np.asarray(jnp.mean(jnp.abs(best_parameters["controls"]) >= 1.0 - 1e-6))
    )
    best_vacancy_bound_fraction = float(
        np.asarray(
            jnp.mean(
                (best_parameters["vacancies"] <= 1e-6)
                | (best_parameters["vacancies"] >= 1.0 - 1e-6)
            )
        )
    )
    optimization_time = perf_counter() - optimization_start
    classified_phase_time = float(sum(phase_timings_s.values()))
    unclassified_phase_time = max(
        optimization_time - classified_phase_time,
        0.0,
    )
    run_time = perf_counter() - run_start
    metadata = {
        **dict(model.metadata),
        **dict(prepared.metadata),
        "result_schema_version": 4,
        "reconstruction_problem_id": prepared.reconstruction_problem_id,
        "reconstructor_id": prepared.reconstructor_id,
        "jax_backend": jax.default_backend(),
        "jax_devices": sorted(str(device) for device in reference.devices()),
        "potential_dtype": str(reference.dtype),
        "probe_dtype": str(probe.dtype),
        "energy_eV": float(np.asarray(energy)),
        "slice_thickness_A": float(np.asarray(slice_thickness)),
        "potential_max_V": resolved_max,
        "maximum_phase_per_slice_rad": max_phase,
        "maximum_displacement_A": float(np.asarray(model.maximum_displacement)),
        "separate_rigid_registration": bool(separate_rigid_registration),
        "similarity_residual_gauge": bool(similarity_residual_gauge),
        "displacement_gauge": (
            "equal_candidate_site_mean"
            if separate_rigid_registration
            else (
                "translation_rotation_isotropic_dilation"
                if similarity_residual_gauge
                else "legacy_total_controls"
            )
        ),
        "registration_scope": "variable_sites_relative_to_fixed_reference",
        "maximum_rigid_displacement_A": rigid_limit,
        "maximum_residual_displacement_A": residual_limit,
        "updates": n_updates,
        "minibatch_size": batch_size,
        "validation_interval": metric_interval,
        "evaluation_batch_size": eval_batch_size,
        "learning_rate_start": float(np.asarray(learning_rate_start)),
        "learning_rate_end": float(np.asarray(learning_rate_end)),
        "optimization_mode": optimization_options.mode,
        "optimization_stage_boundaries": {
            "site_translation_end": rigid_stage_end,
            "vacancy_end": vacancy_stage_end,
            "residual_end": residual_stage_end,
            "joint_end": n_updates,
        },
        "vacancy_learning_rate_scale": (
            optimization_options.vacancy_learning_rate_scale
        ),
        "residual_learning_rate_scale": (
            optimization_options.residual_learning_rate_scale
        ),
        "rigid_learning_rate_scale": optimization_options.rigid_learning_rate_scale,
        "gradient_clip": float(np.asarray(gradient_clip)),
        "seed": int(seed_value),
        "training_indices": training_host.tolist(),
        "training_diagnostic_indices": training_diagnostic_host.tolist(),
        "training_diagnostic_selection": dict(training_diagnostic_selection),
        "training_loss_history_scope": (
            "full_training_partition"
            if training_diagnostic_selection["uses_full_training_partition"]
            else "fixed_geometry_stratified_training_subset"
        ),
        "final_full_training_loss": final_full_training_loss,
        "validation_indices": validation_host.tolist(),
        "audit_indices": audit_host.tolist(),
        "excluded_indices": excluded_host.tolist(),
        "audit_metric": audit_loss,
        "n_variable_sites": int(n_site),
        "n_vacancy_parameters": int(n_site),
        "n_target_vacancy_parameters": int(
            np.count_nonzero(
                modeled_site_roles == int(LatticeSiteRole1D.TARGET)
            )
        ),
        "n_nuisance_vacancy_parameters": int(
            np.count_nonzero(
                modeled_site_roles == int(LatticeSiteRole1D.NUISANCE)
            )
        ),
        "material_scope_complete": material_scope_complete,
        "material_scope_fully_parameterized": (
            material_scope_fully_parameterized
        ),
        "support_contract_id": support_contract_id,
        "structural_reporting_scope": (
            "target_sites_only"
            if support_contract_id is not None
            else "none_legacy_model_has_no_support_contract"
        ),
        "n_displacement_control_parameters": n_control_parameters,
        "n_residual_control_dof": n_residual_control_dof,
        "n_registration_parameters": n_registration_parameters,
        "n_specimen_parameters": (
            int(n_site) + n_residual_control_dof + n_registration_parameters
        ),
        "checkpoint_interval": checkpoint_stride or None,
        "completed_updates": completed_updates,
        "converged": converged,
        "stop_reason": stop_reason,
        "convergence_min_updates": convergence_options.min_updates,
        "convergence_patience_evaluations": (
            convergence_options.patience_evaluations
        ),
        "convergence_relative_min_delta": convergence_options.relative_min_delta,
        "convergence_normalized_step_tolerance": (
            convergence_options.normalized_step_tolerance
        ),
        "convergence_target_loss": convergence_options.target_loss,
        "best_control_bound_fraction": best_control_bound_fraction,
        "best_vacancy_bound_fraction": best_vacancy_bound_fraction,
        "best_total_displacement_bound_fraction": (
            best_total_displacement_bound_fraction
        ),
        "best_metric": best_metric,
        "elapsed_time_history_scope": "run_only_excludes_preparation",
        "initialization_rendering_time_s": initialization_rendering_time_s,
        "optimization_time_s": optimization_time,
        "optimization_phase_timings_s": dict(phase_timings_s),
        "optimization_phase_classified_time_s": classified_phase_time,
        "optimization_phase_unclassified_time_s": unclassified_phase_time,
        "optimizer_chunk_count": optimizer_chunk_count,
        "training_diagnostic_scan_evaluations": int(
            len(update_history) * training_diagnostic_host.size
        ),
        "validation_scan_evaluations": int(
            len(update_history) * validation_host.size
        ),
        "final_prediction_scan_evaluations": int(n_scan),
        "training_diagnostic_batch_evaluations": int(
            len(update_history)
            * int(np.ceil(training_diagnostic_host.size / eval_batch_size))
        ),
        "validation_batch_evaluations": int(
            len(update_history)
            * (
                int(np.ceil(validation_host.size / eval_batch_size))
                if validation_host.size
                else 0
            )
        ),
        "timing_synchronization_scope": (
            "synchronized_at_existing_metric_boundaries_and_final_outputs"
        ),
        "run_time_s": run_time,
        "run_time_scope": "full_run_excludes_preparation",
        "detector_angle_unit": "mrad",
    }
    return LatticeSiteReconstruction1D(
        potential=best_potential,
        initial_potential=initial_potential,
        vacancy_fractions=best_parameters["vacancies"],
        initial_vacancy_fractions=initial_vacancies,
        displacement_controls=best_residual_controls,
        initial_displacement_controls=initial_residual_controls,
        site_coordinates=sites,
        displaced_site_coordinates=sites + site_displacements,
        control_coordinates_s=controls_s,
        control_coordinates_u=controls_u,
        predicted_intensities=predicted,
        measured_intensities=measured,
        window_starts=starts,
        scan_coordinates=coordinates_scan,
        detector_angles=detector_theta,
        update_history=jnp.asarray(update_history),
        elapsed_time_history=jnp.asarray(elapsed_history),
        training_loss_history=jnp.asarray(training_history),
        validation_loss_history=jnp.asarray(validation_history),
        best_update=best_update,
        completed_updates=completed_updates,
        converged=converged,
        stop_reason=stop_reason,
        audit_loss=audit_loss,
        gradient_norm_history=jnp.asarray(gradient_norm_history),
        normalized_step_history=jnp.asarray(normalized_step_history),
        active_bound_fraction_history=jnp.asarray(
            active_bound_fraction_history
        ),
        rigid_displacement=best_rigid,
        initial_rigid_displacement=initial_rigid,
        rigid_displacement_history=(
            jnp.stack(rigid_checkpoints)
            if rigid_checkpoints
            else jnp.empty((0, 2), dtype=reference.dtype)
        ),
        optimization_stage_history=np.asarray(optimization_stage_history),
        checkpoint_updates=jnp.asarray(checkpoint_updates, dtype=jnp.int32),
        vacancy_fraction_history=(
            jnp.stack(vacancy_checkpoints)
            if vacancy_checkpoints
            else jnp.empty((0, n_site), dtype=reference.dtype)
        ),
        displacement_control_history=(
            jnp.stack(control_checkpoints)
            if control_checkpoints
            else jnp.empty((0, *control_shape), dtype=reference.dtype)
        ),
        metadata=metadata,
        detector_valid_mask=valid_mask,
        predicted_signal_electrons=predicted_signal,
        measurement=measurement,
        objective=objective,
        site_role_codes=jnp.asarray(modeled_site_roles, dtype=jnp.int8),
        support_contract_id=support_contract_id,
        material_scope_complete=material_scope_complete,
        material_scope_fully_parameterized=(
            material_scope_fully_parameterized
        ),
    )


def reconstruct_lattice_site_potential_1d(
    model: LatticeSiteModel1D,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    measured_intensities: Any | None = None,
    *,
    measurement: PtychographyMeasurement1D | None = None,
    objective: PtychographyObjective1D | None = None,
    detector_valid_mask: Any | None = None,
    initial_vacancy_fractions: Any | None = None,
    initial_displacement_controls: Any | None = None,
    initial_rigid_displacement: Any | None = None,
    separate_rigid_registration: bool = False,
    similarity_residual_gauge: bool = False,
    maximum_rigid_displacement: Any | None = None,
    maximum_residual_displacement: Any | None = None,
    scan_coordinates: Any | None = None,
    detector_angles: Any | None = None,
    validation_indices: Sequence[int] = (),
    audit_indices: Sequence[int] = (),
    excluded_indices: Sequence[int] = (),
    potential_max: Any | None = None,
    learning_rate_start: Any = 2e-2,
    learning_rate_end: Any = 2e-4,
    updates: int = 500,
    minibatch_size: int = 5,
    validation_interval: int = 25,
    training_diagnostic_scan_count: int | None = None,
    evaluation_batch_size: int = 10,
    gradient_clip: Any = 1.0,
    epsilon: Any = 1e-12,
    rematerialize: bool = True,
    require_complete_material_scope: bool = False,
    seed: int = 0,
    progress: bool = False,
    progress_description: str = "lattice-site reconstruction",
    checkpoint_interval: int | None = None,
    convergence: ConvergenceOptions1D | None = None,
    optimization: LatticeOptimizationOptions1D | None = None,
) -> LatticeSiteReconstruction1D:
    """Prepare and run one lattice-site reconstruction.

    This compatibility entry point preserves the original API.  Call
    :func:`prepare_lattice_site_reconstruction_1d` once and
    :func:`run_prepared_lattice_site_reconstruction_1d` repeatedly when
    comparing multiple initializations of the same measured scan.
    """
    prepared = prepare_lattice_site_reconstruction_1d(
        model,
        input_probe,
        window_starts,
        window_length,
        propagation_kernel,
        slice_thickness,
        energy,
        measured_intensities,
        measurement=measurement,
        objective=objective,
        detector_valid_mask=detector_valid_mask,
        separate_rigid_registration=separate_rigid_registration,
        similarity_residual_gauge=similarity_residual_gauge,
        maximum_rigid_displacement=maximum_rigid_displacement,
        maximum_residual_displacement=maximum_residual_displacement,
        scan_coordinates=scan_coordinates,
        detector_angles=detector_angles,
        validation_indices=validation_indices,
        audit_indices=audit_indices,
        excluded_indices=excluded_indices,
        potential_max=potential_max,
        minibatch_size=minibatch_size,
        evaluation_batch_size=evaluation_batch_size,
        gradient_clip=gradient_clip,
        epsilon=epsilon,
        rematerialize=rematerialize,
        require_complete_material_scope=require_complete_material_scope,
    )
    return run_prepared_lattice_site_reconstruction_1d(
        prepared,
        initial_vacancy_fractions=initial_vacancy_fractions,
        initial_displacement_controls=initial_displacement_controls,
        initial_rigid_displacement=initial_rigid_displacement,
        learning_rate_start=learning_rate_start,
        learning_rate_end=learning_rate_end,
        updates=updates,
        validation_interval=validation_interval,
        training_diagnostic_scan_count=training_diagnostic_scan_count,
        seed=seed,
        progress=progress,
        progress_description=progress_description,
        checkpoint_interval=checkpoint_interval,
        convergence=convergence,
        optimization=optimization,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        array = np.asarray(value)
    except Exception as exc:  # pragma: no cover - input-specific error path
        raise TypeError(f"metadata value {value!r} is not JSON serializable") from exc
    return array.item() if array.ndim == 0 else array.tolist()


def _metadata_json(metadata: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(dict(metadata), default=_json_default, sort_keys=True))


def _save_npz(path: str | Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def _load_detector_valid_mask_1d(
    archive: Any,
    measured_shape: tuple[int, ...],
) -> Array | None:
    """Load and validate the optional non-pickled detector-mask payload."""
    if "detector_valid_mask_present" not in archive.files:
        return None
    marker = np.asarray(archive["detector_valid_mask_present"])
    if marker.shape != () or marker.dtype != np.bool_:
        raise ValueError(
            "detector_valid_mask_present must be a scalar Boolean"
        )
    if not bool(marker.item()):
        return None
    if "detector_valid_mask" not in archive.files:
        raise ValueError(
            "archive marks detector_valid_mask present but omits its payload"
        )
    return _detector_valid_mask_1d(
        jnp.asarray(archive["detector_valid_mask"]), measured_shape
    )


def _archive_boolean_marker_1d(archive: Any, name: str) -> bool:
    if name not in archive.files:
        return False
    marker = np.asarray(archive[name])
    if marker.shape != () or marker.dtype != np.bool_:
        raise ValueError(f"{name} must be a scalar Boolean")
    return bool(marker.item())


def _load_ptychography_measurement_1d(
    archive: Any,
) -> PtychographyMeasurement1D | None:
    if not _archive_boolean_marker_1d(archive, "measurement_present"):
        return None
    required = {
        "measurement_calibrated_signal_electrons",
        "measurement_observed_total_electrons",
        "measurement_valid_mask",
        "measurement_calibrated_dark_electrons_per_pixel",
        "measurement_calibrated_read_noise_std_electrons",
        "measurement_calibration_id",
        "measurement_metadata_json",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        raise ValueError("measurement archive payload is incomplete: " + ", ".join(missing))
    measurement = PtychographyMeasurement1D(
        calibrated_signal_electrons=jnp.asarray(
            archive["measurement_calibrated_signal_electrons"]
        ),
        observed_total_electrons=jnp.asarray(
            archive["measurement_observed_total_electrons"]
        ),
        valid_mask=jnp.asarray(archive["measurement_valid_mask"]),
        calibrated_dark_electrons_per_pixel=jnp.asarray(
            archive["measurement_calibrated_dark_electrons_per_pixel"]
        ),
        calibrated_read_noise_std_electrons=jnp.asarray(
            archive["measurement_calibrated_read_noise_std_electrons"]
        ),
        calibration_id=str(archive["measurement_calibration_id"].item()),
        metadata=json.loads(str(archive["measurement_metadata_json"].item())),
    )
    return _validated_ptychography_measurement_1d(measurement)


def _load_ptychography_objective_1d(
    archive: Any,
    *,
    n_scans: int,
) -> PtychographyObjective1D | None:
    if not _archive_boolean_marker_1d(archive, "objective_present"):
        return None
    required = {
        "objective_kind",
        "objective_electrons_per_pattern",
        "objective_minimum_expected_electrons",
        "objective_relative_signal_scale",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        raise ValueError("objective archive payload is incomplete: " + ", ".join(missing))
    objective, _ = _validated_ptychography_objective_1d(
        PtychographyObjective1D(
            kind=str(archive["objective_kind"].item()),
            electrons_per_pattern=jnp.asarray(
                archive["objective_electrons_per_pattern"]
            ),
            minimum_expected_electrons=float(
                archive["objective_minimum_expected_electrons"].item()
            ),
            relative_signal_scale=float(
                archive["objective_relative_signal_scale"].item()
            ),
        ),
        n_scans=n_scans,
    )
    return objective


def save_glancing_scan_1d(path: str | Path, scan: GlancingScan1D) -> None:
    """Save a scan with non-pickled JSON metadata."""
    _save_npz(
        path,
        intensities=np.asarray(scan.intensities),
        window_starts=np.asarray(scan.window_starts),
        scan_coordinates=np.asarray(scan.scan_coordinates),
        detector_angles=np.asarray(scan.detector_angles),
        detector_valid_mask=(
            np.asarray(scan.detector_valid_mask, dtype=bool)
            if scan.detector_valid_mask is not None
            else np.empty(0, dtype=bool)
        ),
        detector_valid_mask_present=np.asarray(
            scan.detector_valid_mask is not None
        ),
        metadata_json=_metadata_json(scan.metadata),
    )


def load_glancing_scan_1d(path: str | Path) -> GlancingScan1D:
    """Load a scan written by :func:`save_glancing_scan_1d`."""
    with np.load(path, allow_pickle=False) as data:
        intensities = jnp.asarray(data["intensities"])
        return GlancingScan1D(
            intensities=intensities,
            window_starts=jnp.asarray(data["window_starts"]),
            scan_coordinates=jnp.asarray(data["scan_coordinates"]),
            detector_angles=jnp.asarray(data["detector_angles"]),
            metadata=json.loads(str(data["metadata_json"].item())),
            detector_valid_mask=_load_detector_valid_mask_1d(
                data, intensities.shape
            ),
        )


def save_glancing_sideview_cache_1d(
    path: str | Path,
    cache: GlancingSideviewCache1D,
) -> None:
    """Save a compact selected-scan side-view cache."""
    _save_npz(
        path,
        scan_indices=np.asarray(cache.scan_indices),
        window_starts=np.asarray(cache.window_starts),
        scan_coordinates=np.asarray(cache.scan_coordinates),
        local_s_coordinates=np.asarray(cache.local_s_coordinates),
        sideview_u_coordinates=np.asarray(cache.sideview_u_coordinates),
        transverse_coordinates=np.asarray(cache.transverse_coordinates),
        sideview_wavefields=np.asarray(cache.sideview_wavefields),
        sideview_intensities=np.asarray(cache.sideview_intensities),
        exit_waves=np.asarray(cache.exit_waves),
        detector_waves=np.asarray(cache.detector_waves),
        detector_intensities=np.asarray(cache.detector_intensities),
        metadata_json=_metadata_json(cache.metadata),
    )


def load_glancing_sideview_cache_1d(path: str | Path) -> GlancingSideviewCache1D:
    """Load a cache written by :func:`save_glancing_sideview_cache_1d`."""
    with np.load(path, allow_pickle=False) as data:
        return GlancingSideviewCache1D(
            scan_indices=jnp.asarray(data["scan_indices"]),
            window_starts=jnp.asarray(data["window_starts"]),
            scan_coordinates=jnp.asarray(data["scan_coordinates"]),
            local_s_coordinates=jnp.asarray(data["local_s_coordinates"]),
            sideview_u_coordinates=jnp.asarray(data["sideview_u_coordinates"]),
            transverse_coordinates=jnp.asarray(data["transverse_coordinates"]),
            sideview_wavefields=jnp.asarray(data["sideview_wavefields"]),
            sideview_intensities=jnp.asarray(data["sideview_intensities"]),
            exit_waves=jnp.asarray(data["exit_waves"]),
            detector_waves=jnp.asarray(data["detector_waves"]),
            detector_intensities=jnp.asarray(data["detector_intensities"]),
            metadata=json.loads(str(data["metadata_json"].item())),
        )


def save_potential_reconstruction_1d(
    path: str | Path,
    result: PotentialReconstruction1D,
) -> None:
    """Save a direct-potential reconstruction with JSON metadata."""
    _save_npz(
        path,
        potential=np.asarray(result.potential),
        initial_potential=np.asarray(result.initial_potential),
        reconstruction_mask=np.asarray(result.reconstruction_mask),
        axial_coordinates=np.asarray(result.axial_coordinates),
        transverse_coordinates=np.asarray(result.transverse_coordinates),
        predicted_intensities=np.asarray(result.predicted_intensities),
        measured_intensities=np.asarray(result.measured_intensities),
        window_starts=np.asarray(result.window_starts),
        scan_coordinates=np.asarray(result.scan_coordinates),
        detector_angles=np.asarray(result.detector_angles),
        detector_valid_mask=(
            np.asarray(result.detector_valid_mask, dtype=bool)
            if result.detector_valid_mask is not None
            else np.empty(0, dtype=bool)
        ),
        detector_valid_mask_present=np.asarray(
            result.detector_valid_mask is not None
        ),
        update_history=np.asarray(result.update_history),
        elapsed_time_history=np.asarray(result.elapsed_time_history),
        training_loss_history=np.asarray(result.training_loss_history),
        validation_loss_history=np.asarray(result.validation_loss_history),
        best_update=np.asarray(result.best_update, dtype=np.int64),
        audit_loss=np.asarray(result.audit_loss),
        metadata_json=_metadata_json(result.metadata),
    )


def load_potential_reconstruction_1d(path: str | Path) -> PotentialReconstruction1D:
    """Load a result written by :func:`save_potential_reconstruction_1d`."""
    with np.load(path, allow_pickle=False) as data:
        measured = jnp.asarray(data["measured_intensities"])
        return PotentialReconstruction1D(
            potential=jnp.asarray(data["potential"]),
            initial_potential=jnp.asarray(data["initial_potential"]),
            reconstruction_mask=jnp.asarray(data["reconstruction_mask"]),
            axial_coordinates=jnp.asarray(data["axial_coordinates"]),
            transverse_coordinates=jnp.asarray(data["transverse_coordinates"]),
            predicted_intensities=jnp.asarray(data["predicted_intensities"]),
            measured_intensities=measured,
            window_starts=jnp.asarray(data["window_starts"]),
            scan_coordinates=jnp.asarray(data["scan_coordinates"]),
            detector_angles=jnp.asarray(data["detector_angles"]),
            update_history=jnp.asarray(data["update_history"]),
            elapsed_time_history=jnp.asarray(
                data["elapsed_time_history"]
                if "elapsed_time_history" in data.files
                else np.zeros_like(data["update_history"], dtype=float)
            ),
            training_loss_history=jnp.asarray(data["training_loss_history"]),
            validation_loss_history=jnp.asarray(data["validation_loss_history"]),
            best_update=int(data["best_update"].item()),
            audit_loss=float(
                data["audit_loss"].item()
                if "audit_loss" in data.files
                else np.nan
            ),
            metadata=json.loads(str(data["metadata_json"].item())),
            detector_valid_mask=_load_detector_valid_mask_1d(
                data, measured.shape
            ),
        )


def _lattice_result_support_evidence_id_1d(
    site_coordinates: Any,
    site_role_codes: Any,
    support_contract_id: str | None,
    material_scope_complete: bool,
    material_scope_fully_parameterized: bool | None = None,
) -> str:
    """Validate and hash the reportable/nuisance result partition.

    ``None`` for ``material_scope_fully_parameterized`` verifies the legacy
    version-1 evidence contract. New results always pass an explicit Boolean
    and use version 2, which binds trust eligibility into the archive digest.
    """
    sites = np.asarray(site_coordinates)
    roles = np.asarray(site_role_codes)
    if roles.size == 0:
        if roles.shape != (0,):
            raise ValueError("legacy site_role_codes must be an empty vector")
        if (
            support_contract_id is not None
            or material_scope_complete
            or material_scope_fully_parameterized not in (None, False)
        ):
            raise ValueError(
                "material-scope claims require site roles and a support contract"
            )
        return ""
    if roles.shape != (sites.shape[0],) or not np.issubdtype(
        roles.dtype, np.integer
    ):
        raise ValueError("site_role_codes must contain one integer per site")
    roles = roles.astype(np.int8, copy=False)
    allowed = np.asarray(
        [int(LatticeSiteRole1D.TARGET), int(LatticeSiteRole1D.NUISANCE)],
        dtype=np.int8,
    )
    if np.any(~np.isin(roles, allowed)):
        raise ValueError(
            "reconstruction site roles must be TARGET or NUISANCE"
        )
    if not isinstance(support_contract_id, str) or (
        len(support_contract_id) != 64
        or any(character not in "0123456789abcdef" for character in support_contract_id)
    ):
        raise ValueError("support_contract_id must be a lowercase SHA-256 digest")
    if not isinstance(material_scope_complete, (bool, np.bool_)):
        raise TypeError("material_scope_complete must be a boolean")
    if material_scope_fully_parameterized is not None and not isinstance(
        material_scope_fully_parameterized, (bool, np.bool_)
    ):
        raise TypeError("material_scope_fully_parameterized must be a boolean")
    if material_scope_fully_parameterized and not material_scope_complete:
        raise ValueError(
            "fully parameterized material scope requires complete material scope"
        )
    legacy_evidence = material_scope_fully_parameterized is None
    options: dict[str, Any] = {
        "contract": (
            "lattice_result_support_partition:v1"
            if legacy_evidence
            else "lattice_result_support_partition:v2"
        ),
        "support_contract_id": support_contract_id,
        "material_scope_complete": bool(material_scope_complete),
    }
    if not legacy_evidence:
        options["material_scope_fully_parameterized"] = bool(
            material_scope_fully_parameterized
        )
    return _reconstruction_problem_id_1d(
        arrays={"site_coordinates": sites, "site_role_codes": roles},
        options=options,
    )


def save_lattice_site_reconstruction_1d(
    path: str | Path,
    result: LatticeSiteReconstruction1D,
) -> None:
    """Save a lattice-site reconstruction without pickled objects."""
    support_evidence_id = _lattice_result_support_evidence_id_1d(
        result.site_coordinates,
        result.site_role_codes,
        result.support_contract_id,
        result.material_scope_complete,
        result.material_scope_fully_parameterized,
    )
    metadata_scope = result.metadata.get("material_scope_fully_parameterized")
    if metadata_scope is not None and (
        not isinstance(metadata_scope, (bool, np.bool_))
        or bool(metadata_scope) != bool(result.material_scope_fully_parameterized)
    ):
        raise ValueError(
            "metadata material_scope_fully_parameterized disagrees with the "
            "typed reconstruction field"
        )
    _save_npz(
        path,
        potential=np.asarray(result.potential),
        initial_potential=np.asarray(result.initial_potential),
        vacancy_fractions=np.asarray(result.vacancy_fractions),
        initial_vacancy_fractions=np.asarray(result.initial_vacancy_fractions),
        displacement_controls=np.asarray(result.displacement_controls),
        initial_displacement_controls=np.asarray(result.initial_displacement_controls),
        site_coordinates=np.asarray(result.site_coordinates),
        site_role_codes=np.asarray(result.site_role_codes, dtype=np.int8),
        support_contract_id=np.asarray(result.support_contract_id or ""),
        material_scope_complete=np.asarray(result.material_scope_complete),
        material_scope_fully_parameterized=np.asarray(
            result.material_scope_fully_parameterized
        ),
        support_evidence_id=np.asarray(support_evidence_id),
        displaced_site_coordinates=np.asarray(result.displaced_site_coordinates),
        control_coordinates_s=np.asarray(result.control_coordinates_s),
        control_coordinates_u=np.asarray(result.control_coordinates_u),
        predicted_intensities=np.asarray(result.predicted_intensities),
        predicted_signal_electrons=(
            np.asarray(result.predicted_signal_electrons)
            if result.predicted_signal_electrons is not None
            else np.empty(0, dtype=float)
        ),
        predicted_signal_electrons_present=np.asarray(
            result.predicted_signal_electrons is not None
        ),
        measured_intensities=np.asarray(result.measured_intensities),
        window_starts=np.asarray(result.window_starts),
        scan_coordinates=np.asarray(result.scan_coordinates),
        detector_angles=np.asarray(result.detector_angles),
        detector_valid_mask=(
            np.asarray(result.detector_valid_mask, dtype=bool)
            if result.detector_valid_mask is not None
            else np.empty(0, dtype=bool)
        ),
        detector_valid_mask_present=np.asarray(
            result.detector_valid_mask is not None
        ),
        measurement_present=np.asarray(result.measurement is not None),
        measurement_calibrated_signal_electrons=(
            np.asarray(result.measurement.calibrated_signal_electrons)
            if result.measurement is not None
            else np.empty(0, dtype=float)
        ),
        measurement_observed_total_electrons=(
            np.asarray(result.measurement.observed_total_electrons)
            if result.measurement is not None
            else np.empty(0, dtype=float)
        ),
        measurement_valid_mask=(
            np.asarray(result.measurement.valid_mask)
            if result.measurement is not None
            else np.empty(0, dtype=bool)
        ),
        measurement_calibrated_dark_electrons_per_pixel=(
            np.asarray(
                result.measurement.calibrated_dark_electrons_per_pixel
            )
            if result.measurement is not None
            else np.empty(0, dtype=float)
        ),
        measurement_calibrated_read_noise_std_electrons=(
            np.asarray(
                result.measurement.calibrated_read_noise_std_electrons
            )
            if result.measurement is not None
            else np.empty(0, dtype=float)
        ),
        measurement_calibration_id=np.asarray(
            result.measurement.calibration_id
            if result.measurement is not None
            else ""
        ),
        measurement_metadata_json=_metadata_json(
            result.measurement.metadata
            if result.measurement is not None
            else {}
        ),
        objective_present=np.asarray(result.objective is not None),
        objective_kind=np.asarray(
            result.objective.kind if result.objective is not None else ""
        ),
        objective_electrons_per_pattern=(
            np.asarray(result.objective.electrons_per_pattern)
            if result.objective is not None
            else np.empty(0, dtype=float)
        ),
        objective_minimum_expected_electrons=np.asarray(
            result.objective.minimum_expected_electrons
            if result.objective is not None
            else np.nan
        ),
        objective_relative_signal_scale=np.asarray(
            result.objective.relative_signal_scale
            if result.objective is not None
            else np.nan
        ),
        update_history=np.asarray(result.update_history),
        elapsed_time_history=np.asarray(result.elapsed_time_history),
        training_loss_history=np.asarray(result.training_loss_history),
        validation_loss_history=np.asarray(result.validation_loss_history),
        best_update=np.asarray(result.best_update, dtype=np.int64),
        completed_updates=np.asarray(result.completed_updates, dtype=np.int64),
        converged=np.asarray(result.converged),
        stop_reason=np.asarray(result.stop_reason),
        audit_loss=np.asarray(result.audit_loss),
        gradient_norm_history=np.asarray(result.gradient_norm_history),
        normalized_step_history=np.asarray(result.normalized_step_history),
        active_bound_fraction_history=np.asarray(
            result.active_bound_fraction_history
        ),
        rigid_displacement=np.asarray(result.rigid_displacement),
        initial_rigid_displacement=np.asarray(result.initial_rigid_displacement),
        rigid_displacement_history=np.asarray(result.rigid_displacement_history),
        optimization_stage_history=np.asarray(result.optimization_stage_history),
        checkpoint_updates=np.asarray(result.checkpoint_updates),
        vacancy_fraction_history=np.asarray(result.vacancy_fraction_history),
        displacement_control_history=np.asarray(
            result.displacement_control_history
        ),
        metadata_json=_metadata_json(result.metadata),
    )


def load_lattice_site_reconstruction_1d(
    path: str | Path,
) -> LatticeSiteReconstruction1D:
    """Load a result written by :func:`save_lattice_site_reconstruction_1d`."""
    with np.load(path, allow_pickle=False) as data:
        measured = jnp.asarray(data["measured_intensities"])
        measurement = _load_ptychography_measurement_1d(data)
        objective = _load_ptychography_objective_1d(
            data, n_scans=int(measured.shape[0])
        )
        if (measurement is None) != (objective is None):
            raise ValueError(
                "measurement and objective must either both be present or absent"
            )
        if measurement is not None and objective is not None:
            measurement, objective, _ = _validated_measurement_objective_pair_1d(
                measurement, objective
            )
        detector_valid_mask = _load_detector_valid_mask_1d(
            data, measured.shape
        )
        if measurement is not None and (
            detector_valid_mask is None
            or not np.array_equal(
                np.asarray(detector_valid_mask),
                np.asarray(measurement.valid_mask),
            )
        ):
            raise ValueError(
                "saved detector_valid_mask must match measurement.valid_mask"
            )
        predicted_signal = None
        if _archive_boolean_marker_1d(
            data, "predicted_signal_electrons_present"
        ):
            if "predicted_signal_electrons" not in data.files:
                raise ValueError("predicted signal archive payload is missing")
            predicted_signal = jnp.asarray(data["predicted_signal_electrons"])
            if predicted_signal.shape != measured.shape:
                raise ValueError(
                    "predicted_signal_electrons must match measurement shape"
                )
            predicted_signal_host = np.asarray(predicted_signal)
            if np.any(~np.isfinite(predicted_signal_host)) or np.any(
                predicted_signal_host < 0.0
            ):
                raise ValueError(
                    "predicted_signal_electrons must be finite and non-negative"
                )
        if measurement is not None and predicted_signal is None:
            raise ValueError("calibrated measurement archive omits predicted signal")
        support_fields_v1 = {
            "site_role_codes",
            "support_contract_id",
            "material_scope_complete",
            "support_evidence_id",
        }
        support_field_v2 = "material_scope_fully_parameterized"
        present_support_fields = support_fields_v1.intersection(data.files)
        if present_support_fields and present_support_fields != support_fields_v1:
            missing = sorted(support_fields_v1 - present_support_fields)
            raise ValueError(
                "lattice result support evidence is incomplete; missing "
                f"{missing}"
            )
        if support_field_v2 in data.files and not present_support_fields:
            raise ValueError(
                "fully parameterized material evidence requires the complete "
                "support partition"
            )
        if present_support_fields:
            site_role_codes = np.asarray(data["site_role_codes"])
            contract_text = str(data["support_contract_id"].item())
            support_contract_id = contract_text or None
            material_scope_complete = _archive_boolean_marker_1d(
                data, "material_scope_complete"
            )
            has_v2_support_evidence = support_field_v2 in data.files
            material_scope_fully_parameterized = (
                _archive_boolean_marker_1d(data, support_field_v2)
                if has_v2_support_evidence
                else False
            )
            expected_support_evidence = _lattice_result_support_evidence_id_1d(
                data["site_coordinates"],
                site_role_codes,
                support_contract_id,
                material_scope_complete,
                (
                    material_scope_fully_parameterized
                    if has_v2_support_evidence
                    else None
                ),
            )
            archived_support_evidence = str(data["support_evidence_id"].item())
            if archived_support_evidence != expected_support_evidence:
                raise ValueError(
                    "support_evidence_id does not match lattice result support fields"
                )
        else:
            site_role_codes = np.empty(0, dtype=np.int8)
            support_contract_id = None
            material_scope_complete = False
            material_scope_fully_parameterized = False
        metadata = json.loads(str(data["metadata_json"].item()))
        metadata_scope = metadata.get("material_scope_fully_parameterized")
        if support_field_v2 in data.files:
            if metadata_scope is not None and (
                not isinstance(metadata_scope, bool)
                or metadata_scope != material_scope_fully_parameterized
            ):
                raise ValueError(
                    "saved metadata material_scope_fully_parameterized "
                    "disagrees with support evidence"
                )
        else:
            # Version-1 metadata was not covered by support_evidence_id. Preserve
            # no unverified positive claim when loading that legacy format.
            metadata = {
                **metadata,
                "material_scope_fully_parameterized": False,
                "legacy_material_scope_metadata_was_unbound": bool(
                    metadata_scope is True
                ),
            }
        return LatticeSiteReconstruction1D(
            potential=jnp.asarray(data["potential"]),
            initial_potential=jnp.asarray(data["initial_potential"]),
            vacancy_fractions=jnp.asarray(data["vacancy_fractions"]),
            initial_vacancy_fractions=jnp.asarray(data["initial_vacancy_fractions"]),
            displacement_controls=jnp.asarray(data["displacement_controls"]),
            initial_displacement_controls=jnp.asarray(
                data["initial_displacement_controls"]
            ),
            site_coordinates=jnp.asarray(data["site_coordinates"]),
            displaced_site_coordinates=jnp.asarray(data["displaced_site_coordinates"]),
            control_coordinates_s=jnp.asarray(data["control_coordinates_s"]),
            control_coordinates_u=jnp.asarray(data["control_coordinates_u"]),
            predicted_intensities=jnp.asarray(data["predicted_intensities"]),
            measured_intensities=measured,
            window_starts=jnp.asarray(data["window_starts"]),
            scan_coordinates=jnp.asarray(data["scan_coordinates"]),
            detector_angles=jnp.asarray(data["detector_angles"]),
            update_history=jnp.asarray(data["update_history"]),
            elapsed_time_history=jnp.asarray(data["elapsed_time_history"]),
            training_loss_history=jnp.asarray(data["training_loss_history"]),
            validation_loss_history=jnp.asarray(data["validation_loss_history"]),
            best_update=int(data["best_update"].item()),
            completed_updates=int(
                data["completed_updates"].item()
                if "completed_updates" in data.files
                else data["update_history"][-1]
            ),
            converged=bool(
                data["converged"].item() if "converged" in data.files else False
            ),
            stop_reason=(
                str(data["stop_reason"].item())
                if "stop_reason" in data.files
                else "legacy_unknown"
            ),
            audit_loss=float(
                data["audit_loss"].item()
                if "audit_loss" in data.files
                else np.nan
            ),
            gradient_norm_history=jnp.asarray(
                data["gradient_norm_history"]
                if "gradient_norm_history" in data.files
                else np.full_like(data["update_history"], np.nan, dtype=float)
            ),
            normalized_step_history=jnp.asarray(
                data["normalized_step_history"]
                if "normalized_step_history" in data.files
                else np.full_like(data["update_history"], np.nan, dtype=float)
            ),
            active_bound_fraction_history=jnp.asarray(
                data["active_bound_fraction_history"]
                if "active_bound_fraction_history" in data.files
                else np.full_like(data["update_history"], np.nan, dtype=float)
            ),
            rigid_displacement=jnp.asarray(
                data["rigid_displacement"]
                if "rigid_displacement" in data.files
                else np.zeros(2, dtype=float)
            ),
            initial_rigid_displacement=jnp.asarray(
                data["initial_rigid_displacement"]
                if "initial_rigid_displacement" in data.files
                else np.zeros(2, dtype=float)
            ),
            rigid_displacement_history=jnp.asarray(
                data["rigid_displacement_history"]
                if "rigid_displacement_history" in data.files
                else np.empty((0, 2), dtype=float)
            ),
            optimization_stage_history=np.asarray(
                data["optimization_stage_history"]
                if "optimization_stage_history" in data.files
                else np.empty(0, dtype="U16")
            ),
            checkpoint_updates=jnp.asarray(
                data["checkpoint_updates"]
                if "checkpoint_updates" in data.files
                else np.empty(0, dtype=np.int32)
            ),
            vacancy_fraction_history=jnp.asarray(
                data["vacancy_fraction_history"]
                if "vacancy_fraction_history" in data.files
                else np.empty((0, data["vacancy_fractions"].shape[0]))
            ),
            displacement_control_history=jnp.asarray(
                data["displacement_control_history"]
                if "displacement_control_history" in data.files
                else np.empty((0, *data["displacement_controls"].shape))
            ),
            metadata=metadata,
            detector_valid_mask=detector_valid_mask,
            predicted_signal_electrons=predicted_signal,
            measurement=measurement,
            objective=objective,
            site_role_codes=jnp.asarray(site_role_codes, dtype=jnp.int8),
            support_contract_id=support_contract_id,
            material_scope_complete=material_scope_complete,
            material_scope_fully_parameterized=(
                material_scope_fully_parameterized
            ),
        )
