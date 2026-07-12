"""AE-2 active-set reconstruction for sparse side-view atomistic edits.

This module consumes the AE-1 state and renderer without changing their
contract.  It implements a bounded, truth-free active-set method for a
calibrated Poisson-deviance acquisition.  Birth directions are certified only
on the declared discovery pixel grid; continuous positions are refined after
birth, but the first certificate deliberately does not claim continuous-space
KKT completeness.

The maintained specimen is two dimensional in ``(s, u)``.  No nuisance image,
synthetic truth, chemistry label, or material-specific energy envelope is
accepted by this API.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import operator
import threading
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from .ptychography_1d import (
    PtychographyMeasurement1D,
    PtychographyObjective1D,
    lattice_site_displacements_1d,
    simulate_glancing_scan_1d,
    validate_ptychography_measurement_1d,
    validate_ptychography_objective_1d,
)
from .ptychography_atomistic_edit_1d import (
    AtomisticEditModel1D,
    AtomisticEditState1D,
    _dense_host_removals,
    atomistic_edit_active_parameter_count_1d,
    atomistic_edit_addition_positions_1d,
    atomistic_edit_prior_components_1d,
    atomistic_edit_state_is_admissible_1d,
    atomistic_edit_state_is_within_discovery_support_1d,
    empty_atomistic_edit_state_1d,
    render_atomistic_edit_potential_1d,
    validate_atomistic_edit_state_1d,
)
from .ptychography_support_contract_1d import (
    LatticeSiteSupportContract1D,
)


__all__ = [
    "AtomisticEditAblation1D",
    "AtomisticEditGridKKTCertificate1D",
    "AtomisticEditLambdaPathPoint1D",
    "AtomisticEditMultistartReconstruction1D",
    "AtomisticEditObjectiveComponents1D",
    "AtomisticEditProgressEvent1D",
    "AtomisticEditProposalScores1D",
    "AtomisticEditReconstruction1D",
    "AtomisticEditSolverOptions1D",
    "PreparedAtomisticEditReconstruction1D",
    "atomistic_edit_objective_components_1d",
    "atomistic_edit_proposal_scores_1d",
    "prepare_atomistic_edit_reconstruction_1d",
    "run_prepared_atomistic_edit_reconstruction_1d",
    "run_prepared_atomistic_edit_multistart_reconstruction_1d",
]


Array = Any
AtomisticEditAblation1D = Literal["edit_only", "level1_physical"]
AtomisticEditProgressPhase1D = Literal[
    "initial",
    "refinement",
    "birth",
    "polish",
    "lambda_complete",
    "debias",
]
_RECONSTRUCTOR_ID = "wide_angle_propagation.atomistic_edit_active_set_1d:v1"
_PROPOSAL_CERTIFICATE = "full_training_proposal_grid_kkt:v1"
_SELECTION_RULE = "validation_largest_lambda_within_frozen_tolerance:v1"
_DEBIAS_RULE = "support_and_position_fixed_no_edit_penalty:v1"
_HOST_ADJOINT_SITE_BATCH_SIZE = 4
_SPATIAL_NEIGHBOR_QUERY_BATCH_SIZE = 32_768


@dataclass(frozen=True)
class AtomisticEditSolverOptions1D:
    """Numerical policy for the fixed, model-owned regularization path.

    ``training_scan_batch_size`` bounds the scan axis of reverse-mode graphs.
    Batches are accumulated deterministically into the exact full-training
    objective and gradient; ``None`` retains the original single graph.
    """

    ablation: AtomisticEditAblation1D = "level1_physical"
    maximum_active_set_iterations: int = 16
    joint_refinement_updates: int = 40
    polish_updates: int = 40
    debias_updates: int = 60
    learning_rate: float = 2e-2
    polish_learning_rate: float = 5e-3
    debias_learning_rate: float = 5e-3
    gradient_clip: float = 1.0
    birth_removal_fraction: float = 0.1
    birth_scattering_equivalent: float = 0.1
    pruning_threshold: float = 1e-4
    proposal_grid_kkt_tolerance: float = 1e-6
    active_projected_gradient_tolerance: float = 1e-5
    debias_projected_gradient_tolerance: float = 1e-5
    duplicate_merge_resolution_A: float = 1e-6
    validation_relative_tolerance: float = 0.01
    validation_absolute_tolerance: float = 1e-10
    maximum_backtracking_steps: int = 8
    training_scan_batch_size: int | None = None
    seed: int = 0


@dataclass(frozen=True, eq=False)
class AtomisticEditProgressEvent1D:
    """Truth-free structural checkpoint emitted at meaningful solver steps.

    These events describe active-set changes and terminal refinement stages,
    not every internal Adam sub-update.  The state arrays are immutable JAX
    snapshots, so a plotting callback cannot mutate the running optimizer.
    """

    phase: AtomisticEditProgressPhase1D
    path_index: int
    active_set_iteration: int
    edit_penalty: float
    state: AtomisticEditState1D
    detail: str = ""


AtomisticEditProgressCallback1D = Callable[[AtomisticEditProgressEvent1D], None]


@dataclass(frozen=True, eq=False)
class PreparedAtomisticEditReconstruction1D:
    """Validated truth-free Poisson acquisition and fixed forward geometry."""

    model: AtomisticEditModel1D
    probe_rows: Array
    window_starts: Array
    window_length: int
    propagation_kernel: Array
    slice_thickness_A: float
    energy_eV: float
    measurement: PtychographyMeasurement1D
    objective: PtychographyObjective1D
    training_indices: Array
    validation_indices: Array
    audit_indices: Array
    excluded_indices: Array
    reconstruction_problem_id: str
    reconstructor_id: str = _RECONSTRUCTOR_ID
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class AtomisticEditObjectiveComponents1D:
    """Separately reproducible count and Level-1 objective components."""

    count_deviance: Array
    edit_mass: Array
    weighted_edit_penalty: Array
    elastic_penalty: Array
    hard_core_penalty: Array
    total_objective: Array
    edit_penalty: float
    ablation: AtomisticEditAblation1D
    scan_indices: Array


@dataclass(frozen=True, eq=False)
class AtomisticEditProposalScores1D:
    """Full-training adjoint/KKT scores in host-equivalent edit units."""

    addition_data_derivative_grid: Array
    addition_hard_core_derivative_grid: Array
    addition_violation_grid: Array
    host_removal_data_derivative: Array
    host_removal_hard_core_derivative: Array
    host_removal_violation: Array
    paired_replacement_violation: Array
    paired_replacement_anchor_indices: Array
    paired_replacement_scattering_equivalent: Array
    best_kind: str
    best_index: tuple[int, int] | int | None
    best_violation: float
    edit_penalty: float
    training_indices: Array
    score_units: str = "objective_change_per_host_equivalent_edit_mass"
    certificate_scope: str = _PROPOSAL_CERTIFICATE


@dataclass(frozen=True, eq=False)
class AtomisticEditGridKKTCertificate1D:
    """Proposal-grid dormant-direction and active projected-gradient audit."""

    edit_penalty: float
    maximum_addition_violation: float
    maximum_host_removal_violation: float
    maximum_paired_replacement_violation: float
    maximum_dormant_violation: float
    active_projected_gradient_norm: float
    proposal_tolerance: float
    active_gradient_tolerance: float
    proposal_grid_satisfied: bool
    active_projected_gradient_satisfied: bool
    satisfied: bool
    continuous_birth_kkt_evaluated: bool = False
    certificate_scope: str = _PROPOSAL_CERTIFICATE


@dataclass(frozen=True, eq=False)
class AtomisticEditLambdaPathPoint1D:
    """One solved or explicitly failed point on the frozen homotopy."""

    edit_penalty: float
    state: AtomisticEditState1D
    training_objective: AtomisticEditObjectiveComponents1D
    validation_count_deviance: float
    kkt: AtomisticEditGridKKTCertificate1D
    active_set_iterations: int
    optimizer_reset_count: int
    births: tuple[str, ...]
    pruned_host_removals: int
    pruned_extra_centres: int
    merged_extra_centres: int
    duplicate_status: str
    capacity_status: str
    stop_reason: str
    converged: bool


@dataclass(frozen=True, eq=False)
class AtomisticEditReconstruction1D:
    """Validation-selected penalized support and fixed-support debiased fit."""

    prepared_problem_id: str
    reconstructor_id: str
    penalized_state: AtomisticEditState1D
    debiased_state: AtomisticEditState1D
    selected_edit_penalty: float
    selected_path_index: int
    path_points: tuple[AtomisticEditLambdaPathPoint1D, ...]
    penalized_training_objective: AtomisticEditObjectiveComponents1D
    debiased_training_objective: AtomisticEditObjectiveComponents1D
    penalized_validation_count_deviance: float
    debiased_validation_count_deviance: float
    debiased_audit_count_deviance: float | None
    selected_kkt: AtomisticEditGridKKTCertificate1D
    debias_projected_gradient_norm: float
    debias_projected_gradient_tolerance: float
    debias_converged: bool
    active_parameter_count: int
    capacity_exhausted: bool
    converged: bool
    stop_reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class AtomisticEditMultistartReconstruction1D:
    """Validation-selected deterministic starts and ambiguity declaration."""

    candidates: tuple[AtomisticEditReconstruction1D, ...]
    selected_result: AtomisticEditReconstruction1D
    selected_start_index: int
    validation_eligible_start_indices: tuple[int, ...]
    ambiguous_start_indices: tuple[int, ...]
    structurally_ambiguous: bool
    start_seeds: tuple[int, ...]
    initial_host_control_rms_A: tuple[float, ...]
    numerically_converged: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _index(name: str, value: Any, *, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    lower = 0 if allow_zero else 1
    if result < lower:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")
    return int(result)


def _finite(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    array = np.asarray(value)
    if array.shape != () or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validated_solver_options(
    options: AtomisticEditSolverOptions1D | None,
) -> AtomisticEditSolverOptions1D:
    result = AtomisticEditSolverOptions1D() if options is None else options
    if not isinstance(result, AtomisticEditSolverOptions1D):
        raise TypeError("options must be AtomisticEditSolverOptions1D or None")
    if result.ablation not in {"edit_only", "level1_physical"}:
        raise ValueError("options.ablation must be 'edit_only' or 'level1_physical'")
    for name in (
        "maximum_active_set_iterations",
        "joint_refinement_updates",
        "polish_updates",
        "debias_updates",
        "maximum_backtracking_steps",
    ):
        _index(
            name,
            getattr(result, name),
            allow_zero=name != "maximum_active_set_iterations",
        )
    if result.training_scan_batch_size is not None:
        _index("training_scan_batch_size", result.training_scan_batch_size)
    for name in (
        "learning_rate",
        "polish_learning_rate",
        "debias_learning_rate",
        "gradient_clip",
        "birth_removal_fraction",
        "birth_scattering_equivalent",
        "proposal_grid_kkt_tolerance",
        "active_projected_gradient_tolerance",
        "debias_projected_gradient_tolerance",
        "duplicate_merge_resolution_A",
    ):
        _finite(name, getattr(result, name), positive=True)
    for name in (
        "pruning_threshold",
        "validation_relative_tolerance",
        "validation_absolute_tolerance",
    ):
        _finite(name, getattr(result, name), nonnegative=True)
    if result.birth_removal_fraction > 1.0:
        raise ValueError("birth_removal_fraction must not exceed one")
    try:
        seed = operator.index(result.seed)
    except TypeError as error:
        raise TypeError("seed must be an integer") from error
    if seed < 0 or seed >= 2**64:
        raise ValueError("seed must lie in [0, 2**64)")
    return result


def _hash_problem(arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        header = json.dumps(
            {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        for payload in (header, array.tobytes(order="C")):
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    encoded = json.dumps(
        dict(metadata), allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    return digest.hexdigest()


def _validated_partition(
    n_scan: int,
    validation_indices: Sequence[int],
    audit_indices: Sequence[int],
    excluded_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = {}
    for name, raw in (
        ("validation_indices", validation_indices),
        ("audit_indices", audit_indices),
        ("excluded_indices", excluded_indices),
    ):
        array = np.asarray(raw)
        if array.ndim != 1 or (
            array.size and not np.issubdtype(array.dtype, np.integer)
        ):
            raise TypeError(f"{name} must be a one-dimensional integer sequence")
        array = np.asarray(array, dtype=np.int32)
        if np.any(array < 0) or np.any(array >= n_scan):
            raise ValueError(f"{name} contains an out-of-range scan")
        if np.unique(array).size != array.size:
            raise ValueError(f"{name} must not repeat scans")
        values[name] = np.sort(array)
    held = list(values.values())
    for first in range(len(held)):
        for second in range(first + 1, len(held)):
            if np.intersect1d(held[first], held[second]).size:
                raise ValueError(
                    "validation, audit, and excluded scans must be disjoint"
                )
    training = np.setdiff1d(
        np.arange(n_scan, dtype=np.int32),
        np.concatenate(held) if held else np.empty(0, dtype=np.int32),
        assume_unique=True,
    )
    if not training.size:
        raise ValueError("the prepared acquisition requires training scans")
    if not values["validation_indices"].size:
        raise ValueError("the frozen lambda path requires validation scans")
    return (
        training,
        values["validation_indices"],
        values["audit_indices"],
        values["excluded_indices"],
    )


def prepare_atomistic_edit_reconstruction_1d(
    model: AtomisticEditModel1D,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness_A: Any,
    energy_eV: Any,
    measurement: PtychographyMeasurement1D,
    objective: PtychographyObjective1D,
    *,
    validation_indices: Sequence[int],
    audit_indices: Sequence[int] = (),
    excluded_indices: Sequence[int] = (),
) -> PreparedAtomisticEditReconstruction1D:
    """Bind an AE-1 model to a calibrated Poisson-deviance acquisition."""
    if not isinstance(model, AtomisticEditModel1D):
        raise TypeError("model must be an AtomisticEditModel1D")
    if model.options.enable_material_energy_envelope:
        raise ValueError("the AE-2 solver does not accept an energy envelope")
    validate_ptychography_measurement_1d(measurement)
    validate_ptychography_objective_1d(objective)
    if objective.kind != "poisson_deviance":
        raise ValueError("AE-2 requires objective.kind='poisson_deviance'")
    valid = np.asarray(measurement.valid_mask, dtype=bool)
    read_noise = np.asarray(
        measurement.calibrated_read_noise_std_electrons
    )
    if np.any(read_noise[valid] != 0.0):
        raise ValueError("Poisson-deviance AE-2 requires declared zero read noise")
    observed = np.asarray(measurement.observed_total_electrons)
    if observed.ndim != 2:
        raise ValueError("measurement arrays must have shape (scan, detector)")
    n_scan, n_detector = observed.shape
    probe = jnp.asarray(input_probe)
    if probe.ndim == 1:
        if probe.shape != (n_detector,):
            raise ValueError("one-dimensional input_probe has wrong detector length")
        probe_rows = jnp.broadcast_to(probe, (n_scan, n_detector))
    elif probe.shape == (n_scan, n_detector):
        probe_rows = probe
    else:
        raise ValueError("input_probe must have shape (detector,) or (scan, detector)")
    starts = np.asarray(window_starts)
    if starts.shape != (n_scan,) or not np.issubdtype(starts.dtype, np.integer):
        raise TypeError("window_starts must contain one integer per scan")
    starts = np.asarray(starts, dtype=np.int32)
    length = _index("window_length", window_length)
    kernel = jnp.asarray(propagation_kernel)
    if kernel.shape != (n_detector,) or not jnp.iscomplexobj(kernel):
        raise TypeError("propagation_kernel must be a complex detector-length vector")
    slice_A = _finite("slice_thickness_A", slice_thickness_A, positive=True)
    energy = _finite("energy_eV", energy_eV, positive=True)
    training, validation, audit, excluded = _validated_partition(
        n_scan, validation_indices, audit_indices, excluded_indices
    )
    full_control_count = int(
        np.prod(
            np.asarray(
                empty_atomistic_edit_state_1d(model).host_displacement_controls
            ).shape
        )
    )
    if model.deformation_parameter_count != full_control_count:
        raise ValueError(
            "AE-2 cannot optimize an unauthenticated deformation subspace: "
            "model.deformation_parameter_count must equal the stored control size"
        )
    metadata = {
        "schema": "prepared_atomistic_edit_reconstruction_1d:v1",
        "model_id": model.model_id,
        "objective_kind": objective.kind,
        "calibration_id": measurement.calibration_id,
        "truth_inputs_accepted": False,
        "nuisance_image_present": False,
        "energy_envelope_present": False,
        "birth_certificate_scope": _PROPOSAL_CERTIFICATE,
        "count_interpretation": (
            "Poisson deviance on calibrated non-negative electron-equivalent totals"
        ),
    }
    problem_id = _hash_problem(
        {
            "probe_rows": probe_rows,
            "window_starts": starts,
            "propagation_kernel": kernel,
            "measurement_signal": measurement.calibrated_signal_electrons,
            "measurement_total": measurement.observed_total_electrons,
            "measurement_valid": measurement.valid_mask,
            "measurement_dark": measurement.calibrated_dark_electrons_per_pixel,
            "measurement_read_noise": (
                measurement.calibrated_read_noise_std_electrons
            ),
            "objective_dose": objective.electrons_per_pattern,
            "training_indices": training,
            "validation_indices": validation,
            "audit_indices": audit,
            "excluded_indices": excluded,
        },
        {
            **metadata,
            "window_length": length,
            "slice_thickness_A": slice_A,
            "energy_eV": energy,
            "minimum_expected_electrons": float(
                objective.minimum_expected_electrons
            ),
            "relative_signal_scale": float(objective.relative_signal_scale),
            "edit_penalty_path": list(model.options.edit_penalty_path),
        },
    )
    return PreparedAtomisticEditReconstruction1D(
        model=model,
        probe_rows=probe_rows,
        window_starts=jnp.asarray(starts),
        window_length=length,
        propagation_kernel=kernel,
        slice_thickness_A=slice_A,
        energy_eV=energy,
        measurement=measurement,
        objective=objective,
        training_indices=jnp.asarray(training),
        validation_indices=jnp.asarray(validation),
        audit_indices=jnp.asarray(audit),
        excluded_indices=jnp.asarray(excluded),
        reconstruction_problem_id=problem_id,
        metadata=MappingProxyType(metadata),
    )


def _count_deviance_sum_from_potential(
    prepared: PreparedAtomisticEditReconstruction1D,
    potential: Array,
    indices: Array,
) -> tuple[Array, Array]:
    """Return the unnormalized Poisson deviance and valid-pixel count."""
    index = jnp.asarray(indices, dtype=jnp.int32)
    probes = jnp.asarray(prepared.probe_rows)[index]
    prediction = simulate_glancing_scan_1d(
        potential,
        probes,
        jnp.asarray(prepared.window_starts)[index],
        prepared.window_length,
        prepared.propagation_kernel,
        prepared.slice_thickness_A,
        prepared.energy_eV,
        rematerialize=False,
    )
    # Preparation has already validated the zero-read-noise Poisson contract.
    # Keep this inner objective entirely in JAX so a fixed active-set
    # refinement can be compiled once; re-entering the public validation layer
    # here would attempt NumPy conversion of closed-over JIT tracers.
    measurement_shape = jnp.shape(
        prepared.measurement.observed_total_electrons
    )
    n_detector = prediction.shape[1]
    incident_norm = n_detector * jnp.sum(
        jnp.abs(probes) ** 2, axis=1, keepdims=True
    )
    dose = jnp.asarray(prepared.objective.electrons_per_pattern)
    if dose.ndim == 0:
        selected_dose = jnp.broadcast_to(dose, (index.shape[0],))
    else:
        selected_dose = jnp.broadcast_to(dose, (measurement_shape[0],))[index]
    predicted_signal = (
        jnp.asarray(prepared.objective.relative_signal_scale)
        * selected_dose[:, None]
        * prediction
        / incident_norm
    )

    def selected(value: Any) -> Array:
        return jnp.broadcast_to(jnp.asarray(value), measurement_shape)[index]

    valid = selected(prepared.measurement.valid_mask).astype(bool)
    observed = selected(prepared.measurement.observed_total_electrons)
    predicted_signal = jnp.where(valid, predicted_signal, 0.0)
    dark = jnp.where(
        valid,
        selected(prepared.measurement.calibrated_dark_electrons_per_pixel),
        0.0,
    )
    mean_total = jnp.maximum(
        predicted_signal + dark,
        jnp.asarray(prepared.objective.minimum_expected_electrons),
    )
    safe_observed = jnp.where(valid, observed, 0.0)
    ratio = jnp.where(
        safe_observed > 0.0,
        safe_observed / mean_total,
        1.0,
    )
    log_term = jnp.where(
        safe_observed > 0.0,
        safe_observed * jnp.log(ratio),
        0.0,
    )
    terms = 2.0 * (mean_total - safe_observed + log_term)
    return jnp.sum(jnp.where(valid, terms, 0.0)), jnp.count_nonzero(valid)


def _count_loss_from_potential(
    prepared: PreparedAtomisticEditReconstruction1D,
    potential: Array,
    indices: Array,
) -> Array:
    deviance_sum, valid_count = _count_deviance_sum_from_potential(
        prepared, potential, indices
    )
    return deviance_sum / valid_count


def _scan_batches(
    indices: Array,
    batch_size: int | None,
) -> tuple[np.ndarray, ...]:
    """Return stable contiguous scan batches without changing scan weights."""
    index = np.asarray(indices, dtype=np.int32)
    if index.ndim != 1 or not index.size:
        raise ValueError("indices must contain at least one scan")
    if batch_size is None or batch_size >= index.size:
        return (index,)
    resolved = _index("training_scan_batch_size", batch_size)
    return tuple(
        np.ascontiguousarray(index[start : start + resolved])
        for start in range(0, index.size, resolved)
    )


def _valid_observation_count(
    prepared: PreparedAtomisticEditReconstruction1D,
    indices: Array,
) -> int:
    measurement_shape = np.shape(
        prepared.measurement.observed_total_electrons
    )
    valid = np.broadcast_to(
        np.asarray(prepared.measurement.valid_mask, dtype=bool),
        measurement_shape,
    )
    count = int(np.count_nonzero(valid[np.asarray(indices, dtype=np.int32)]))
    if count <= 0:
        raise ValueError("selected scans contain no valid detector pixels")
    return count


def _count_loss(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    indices: Array,
) -> Array:
    return _count_loss_from_potential(
        prepared,
        render_atomistic_edit_potential_1d(prepared.model, state),
        indices,
    )


def atomistic_edit_objective_components_1d(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    edit_penalty: Any,
    *,
    scan_indices: Sequence[int] | Array | None = None,
    ablation: AtomisticEditAblation1D = "level1_physical",
) -> AtomisticEditObjectiveComponents1D:
    """Evaluate count, edit, elasticity, and hard-core terms separately."""
    if not isinstance(prepared, PreparedAtomisticEditReconstruction1D):
        raise TypeError("prepared must be PreparedAtomisticEditReconstruction1D")
    validate_atomistic_edit_state_1d(prepared.model, state)
    penalty = _finite("edit_penalty", edit_penalty, nonnegative=True)
    if ablation not in {"edit_only", "level1_physical"}:
        raise ValueError("ablation must be 'edit_only' or 'level1_physical'")
    indices = (
        prepared.training_indices
        if scan_indices is None
        else jnp.asarray(scan_indices, dtype=jnp.int32)
    )
    if jnp.asarray(indices).ndim != 1 or jnp.asarray(indices).size == 0:
        raise ValueError("scan_indices must contain at least one scan")
    count = _count_loss(prepared, state, indices)
    raw = atomistic_edit_prior_components_1d(
        prepared.model,
        state,
        penalty if penalty > 0.0 else 1.0,
    )
    edit_mass = raw.edit_mass
    weighted = jnp.asarray(penalty, dtype=jnp.asarray(edit_mass).dtype) * edit_mass
    elastic = raw.elastic_penalty
    hard_core = raw.hard_core_penalty
    physical = elastic + hard_core if ablation == "level1_physical" else 0.0
    return AtomisticEditObjectiveComponents1D(
        count_deviance=count,
        edit_mass=edit_mass,
        weighted_edit_penalty=weighted,
        elastic_penalty=elastic,
        hard_core_penalty=hard_core,
        total_objective=count + weighted + physical,
        edit_penalty=penalty,
        ablation=ablation,
        scan_indices=jnp.asarray(indices, dtype=jnp.int32),
    )


def _numpy_objective_components(
    components: AtomisticEditObjectiveComponents1D,
) -> AtomisticEditObjectiveComponents1D:
    return AtomisticEditObjectiveComponents1D(
        count_deviance=float(np.asarray(components.count_deviance)),
        edit_mass=float(np.asarray(components.edit_mass)),
        weighted_edit_penalty=float(
            np.asarray(components.weighted_edit_penalty)
        ),
        elastic_penalty=float(np.asarray(components.elastic_penalty)),
        hard_core_penalty=float(np.asarray(components.hard_core_penalty)),
        total_objective=float(np.asarray(components.total_objective)),
        edit_penalty=components.edit_penalty,
        ablation=components.ablation,
        scan_indices=np.asarray(components.scan_indices, dtype=np.int32),
    )


def _count_loss_value(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    indices: Array,
    scan_batch_size: int | None,
) -> float:
    """Evaluate mean deviance with bounded peak scan memory when requested."""
    batches = _scan_batches(indices, scan_batch_size)
    if len(batches) == 1:
        return float(
            np.asarray(
                jax.block_until_ready(_count_loss(prepared, state, batches[0]))
            )
        )
    potential = render_atomistic_edit_potential_1d(prepared.model, state)
    deviance_sum = 0.0
    for batch in batches:
        value, _ = _count_deviance_sum_from_potential(
            prepared, potential, batch
        )
        deviance_sum += float(
            np.asarray(jax.block_until_ready(value))
        )
    return deviance_sum / _valid_observation_count(prepared, indices)


def _objective_components_value(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    edit_penalty: float,
    *,
    scan_indices: Array,
    ablation: AtomisticEditAblation1D,
    scan_batch_size: int | None,
) -> AtomisticEditObjectiveComponents1D:
    batches = _scan_batches(scan_indices, scan_batch_size)
    if len(batches) == 1:
        return _numpy_objective_components(
            atomistic_edit_objective_components_1d(
                prepared,
                state,
                edit_penalty,
                scan_indices=batches[0],
                ablation=ablation,
            )
        )
    count = _count_loss_value(
        prepared, state, scan_indices, scan_batch_size
    )
    prior = atomistic_edit_prior_components_1d(
        prepared.model,
        state,
        edit_penalty if edit_penalty > 0.0 else 1.0,
    )
    edit_mass = float(np.asarray(prior.edit_mass))
    weighted = edit_penalty * edit_mass
    elastic = float(np.asarray(prior.elastic_penalty))
    hard_core = float(np.asarray(prior.hard_core_penalty))
    physical = elastic + hard_core if ablation == "level1_physical" else 0.0
    return AtomisticEditObjectiveComponents1D(
        count_deviance=count,
        edit_mass=edit_mass,
        weighted_edit_penalty=weighted,
        elastic_penalty=elastic,
        hard_core_penalty=hard_core,
        total_objective=count + weighted + physical,
        edit_penalty=edit_penalty,
        ablation=ablation,
        scan_indices=np.asarray(scan_indices, dtype=np.int32),
    )


def _dense_host_removals_numpy(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
) -> np.ndarray:
    result = np.zeros(len(model.host_model.site_coordinates), dtype=float)
    indices = np.asarray(state.host_removal_indices, dtype=np.int64)
    fractions = np.asarray(state.host_removal_fractions, dtype=float)
    active = np.asarray(state.host_removal_active, dtype=bool)
    np.add.at(result, indices[active], fractions[active])
    return result


def _host_geometry(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support = model.host_model.support_contract
    if not isinstance(support, LatticeSiteSupportContract1D):
        raise ValueError("atomistic-edit host support contract is missing")
    all_sites = np.asarray(support.all_site_coordinates, dtype=float)
    modeled = np.asarray(support.modeled_site_indices, dtype=np.int64)
    displacements = np.asarray(
        lattice_site_displacements_1d(
            model.host_model.site_coordinates,
            state.host_displacement_controls,
            model.host_model.control_coordinates_s,
            model.host_model.control_coordinates_u,
        )
    )
    displaced = all_sites.copy()
    displaced[modeled] += displacements
    dense = _dense_host_removals_numpy(model, state)
    all_removals = np.zeros(len(all_sites), dtype=float)
    all_removals[modeled] = dense
    return displaced, 1.0 - all_removals, modeled, dense


def _hard_core_phi_numpy(distance: np.ndarray, minimum: float) -> np.ndarray:
    onset = 1.1 * minimum
    numerator = np.maximum(onset - distance, 0.0)
    gap = np.maximum(distance - minimum, 1e-6 * minimum)
    return np.where(distance < onset, (numerator / gap) ** 2, 0.0)


def _weighted_hard_core_neighbor_sums(
    query_positions: np.ndarray,
    source_positions: np.ndarray,
    source_weights: np.ndarray,
    minimum: float,
) -> np.ndarray:
    """Sum finite-range hard-core terms without forming all point pairs."""
    from scipy.spatial import cKDTree

    queries = np.asarray(query_positions, dtype=float).reshape(-1, 2)
    sources = np.asarray(source_positions, dtype=float).reshape(-1, 2)
    weights = np.asarray(source_weights, dtype=float).reshape(-1)
    if len(sources) != len(weights):
        raise ValueError(
            "source_positions and source_weights must have equal length"
        )
    result = np.zeros(len(queries), dtype=float)
    if not len(queries) or not len(sources):
        return result

    source_tree = cKDTree(sources)
    radius = 1.1 * float(minimum)
    for start in range(0, len(queries), _SPATIAL_NEIGHBOR_QUERY_BATCH_SIZE):
        stop = min(start + _SPATIAL_NEIGHBOR_QUERY_BATCH_SIZE, len(queries))
        distances = cKDTree(queries[start:stop]).sparse_distance_matrix(
            source_tree,
            radius,
            output_type="coo_matrix",
        )
        if distances.nnz:
            contributions = weights[distances.col] * _hard_core_phi_numpy(
                distances.data, minimum
            )
            result[start:stop] = np.bincount(
                distances.row,
                weights=contributions,
                minlength=stop - start,
            )
    return result


def _minimum_separation_admissible_mask(
    query_positions: np.ndarray,
    obstacle_positions: np.ndarray,
    minimum: float,
) -> np.ndarray:
    """Test exact nearest-neighbour clearance in bounded query batches."""
    from scipy.spatial import cKDTree

    queries = np.asarray(query_positions, dtype=float).reshape(-1, 2)
    obstacles = np.asarray(obstacle_positions, dtype=float).reshape(-1, 2)
    result = np.ones(len(queries), dtype=bool)
    if not len(queries) or not len(obstacles):
        return result

    obstacle_tree = cKDTree(obstacles)
    for start in range(0, len(queries), _SPATIAL_NEIGHBOR_QUERY_BATCH_SIZE):
        stop = min(start + _SPATIAL_NEIGHBOR_QUERY_BATCH_SIZE, len(queries))
        nearest, _ = obstacle_tree.query(queries[start:stop], k=1)
        result[start:stop] = nearest >= minimum
    return result


def _hard_core_directional_derivatives(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
    candidate_anchors: np.ndarray,
    paired_anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hosts, occupancy, modeled, _ = _host_geometry(model, state)
    minimum = float(model.options.minimum_separation_A)
    maximum_mass = float(model.options.max_scattering_equivalent_per_centre)
    extra_active = np.asarray(state.extra_active, dtype=bool) & (
        np.asarray(state.extra_scattering_equivalents, dtype=float) > 0.0
    )
    extra_positions = np.asarray(
        atomistic_edit_addition_positions_1d(model, state), dtype=float
    )[extra_active]
    extra_normalized_mass = (
        np.asarray(state.extra_scattering_equivalents, dtype=float)[extra_active]
        / maximum_mass
    )

    removal_derivative_all = np.zeros(len(hosts), dtype=float)
    pairs = np.asarray(model.host_hard_core_pairs, dtype=np.int64)
    if pairs.size:
        distances = np.linalg.norm(
            hosts[pairs[:, 0]] - hosts[pairs[:, 1]], axis=1
        )
        phi = _hard_core_phi_numpy(distances, minimum)
        np.add.at(
            removal_derivative_all,
            pairs[:, 0],
            -occupancy[pairs[:, 1]] * phi,
        )
        np.add.at(
            removal_derivative_all,
            pairs[:, 1],
            -occupancy[pairs[:, 0]] * phi,
        )
    if extra_positions.size:
        removal_derivative_all -= _weighted_hard_core_neighbor_sums(
            hosts,
            extra_positions,
            extra_normalized_mass,
            minimum,
        )

    s_A = np.asarray(model.axial_coordinates_A, dtype=float)
    u_A = np.asarray(model.transverse_coordinates_A, dtype=float)
    positions = np.column_stack(
        [s_A[candidate_anchors[:, 0]], u_A[candidate_anchors[:, 1]]]
    )
    addition_derivative = _weighted_hard_core_neighbor_sums(
        positions,
        hosts,
        occupancy,
        minimum,
    )
    if extra_positions.size:
        addition_derivative += _weighted_hard_core_neighbor_sums(
            positions,
            extra_positions,
            extra_normalized_mass,
            minimum,
        )
    addition_derivative /= maximum_mass

    paired_derivative = np.full(len(modeled), np.inf, dtype=float)
    valid_paired = np.all(paired_anchors >= 0, axis=1)
    if np.any(valid_paired):
        valid_local = np.flatnonzero(valid_paired)
        valid_anchors = paired_anchors[valid_paired]
        paired_positions = np.column_stack(
            [s_A[valid_anchors[:, 0]], u_A[valid_anchors[:, 1]]]
        )
        paired_values = _weighted_hard_core_neighbor_sums(
            paired_positions,
            hosts,
            occupancy,
            minimum,
        )
        paired_all_indices = modeled[valid_paired]
        own_distances = np.linalg.norm(
            hosts[paired_all_indices] - paired_positions, axis=1
        )
        paired_values -= occupancy[paired_all_indices] * _hard_core_phi_numpy(
            own_distances, minimum
        )
        if extra_positions.size:
            paired_values += _weighted_hard_core_neighbor_sums(
                paired_positions,
                extra_positions,
                extra_normalized_mass,
                minimum,
            )
        paired_derivative[valid_local] = (
            removal_derivative_all[paired_all_indices]
            + paired_values / maximum_mass
        )
    return (
        addition_derivative,
        removal_derivative_all[modeled],
        paired_derivative,
    )


def _shift_axis_numpy(
    patch: np.ndarray,
    shift_pixels: float,
    *,
    axis: int,
) -> np.ndarray:
    length = patch.shape[axis]
    targets = np.arange(length)
    source = targets.astype(float) - shift_pixels
    lower = np.floor(source).astype(int)
    fraction = source - lower
    upper = lower + 1
    lower_valid = (lower >= 0) & (lower < length)
    upper_valid = (upper >= 0) & (upper < length)
    lower_samples = np.take(patch, np.clip(lower, 0, length - 1), axis=axis)
    upper_samples = np.take(patch, np.clip(upper, 0, length - 1), axis=axis)
    if axis == 0:
        return (
            np.where(lower_valid[:, None], lower_samples, 0.0)
            * (1.0 - fraction)[:, None]
            + np.where(upper_valid[:, None], upper_samples, 0.0)
            * fraction[:, None]
        )
    return (
        np.where(lower_valid[None, :], lower_samples, 0.0)
        * (1.0 - fraction)[None, :]
        + np.where(upper_valid[None, :], upper_samples, 0.0)
        * fraction[None, :]
    )


def _addition_data_derivative_grid(
    model: AtomisticEditModel1D,
    potential_adjoint: np.ndarray,
) -> np.ndarray:
    from scipy.signal import correlate

    kernel = np.asarray(model.addition_kernel.unit_integrated_values, dtype=float)
    centre = np.asarray(model.addition_kernel.centre_index, dtype=float)
    start_offset = np.floor(-centre + 0.5).astype(int)
    base_shift = -(start_offset.astype(float) + centre)
    shifted = _shift_axis_numpy(kernel, float(base_shift[0]), axis=0)
    shifted = _shift_axis_numpy(shifted, float(base_shift[1]), axis=1)
    correlation = correlate(
        np.asarray(potential_adjoint, dtype=float),
        shifted,
        mode="valid",
        method="fft",
    )
    result = np.full(potential_adjoint.shape, np.nan, dtype=float)
    anchors = np.argwhere(model.options.discovery_support.discovery_mask)
    starts = anchors + start_offset[None, :]
    result[anchors[:, 0], anchors[:, 1]] = correlation[
        starts[:, 0], starts[:, 1]
    ] * float(model.addition_kernel.host_equivalent_integrated_scattering)
    return result


def _paired_replacement_anchors(model: AtomisticEditModel1D) -> np.ndarray:
    sites = np.asarray(model.host_model.site_coordinates, dtype=float)
    s_A = np.asarray(model.axial_coordinates_A, dtype=float)
    u_A = np.asarray(model.transverse_coordinates_A, dtype=float)
    anchors = np.column_stack(
        [
            np.rint((sites[:, 0] - s_A[0]) / (s_A[1] - s_A[0])),
            np.rint((sites[:, 1] - u_A[0]) / (u_A[1] - u_A[0])),
        ]
    ).astype(np.int32)
    shape = np.asarray(model.options.discovery_support.discovery_mask).shape
    inside = (
        (anchors[:, 0] >= 0)
        & (anchors[:, 0] < shape[0])
        & (anchors[:, 1] >= 0)
        & (anchors[:, 1] < shape[1])
    )
    discovery = np.asarray(model.options.discovery_support.discovery_mask)
    valid = np.zeros(len(anchors), dtype=bool)
    valid[inside] = discovery[
        anchors[inside, 0], anchors[inside, 1]
    ]
    anchors[~valid] = -1
    return anchors


def _addition_admissible_mask(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
    anchors: np.ndarray,
    *,
    excluded_host_local_index: int | None = None,
) -> np.ndarray:
    if not len(anchors):
        return np.empty(0, dtype=bool)
    hosts, occupancy, modeled, _ = _host_geometry(model, state)
    occupied = occupancy > 1e-12
    if excluded_host_local_index is not None:
        occupied[int(modeled[excluded_host_local_index])] = False
    active_extra = np.asarray(state.extra_active, dtype=bool) & (
        np.asarray(state.extra_scattering_equivalents, dtype=float) > 0.0
    )
    extras = np.asarray(atomistic_edit_addition_positions_1d(model, state))[
        active_extra
    ]
    s_A = np.asarray(model.axial_coordinates_A, dtype=float)
    u_A = np.asarray(model.transverse_coordinates_A, dtype=float)
    positions = np.column_stack(
        [s_A[anchors[:, 0]], u_A[anchors[:, 1]]]
    )
    minimum = float(model.options.minimum_separation_A) - 1e-12
    obstacles = hosts[occupied]
    if extras.size:
        obstacles = np.concatenate([obstacles, extras], axis=0)
    return _minimum_separation_admissible_mask(positions, obstacles, minimum)


def _paired_addition_admissible_mask(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
    anchors: np.ndarray,
) -> np.ndarray:
    """Check all replacement anchors while excluding each replaced host."""
    from scipy.spatial import cKDTree

    result = np.zeros(len(anchors), dtype=bool)
    valid = np.all(anchors >= 0, axis=1)
    if not np.any(valid):
        return result

    hosts, occupancy, modeled, _ = _host_geometry(model, state)
    s_A = np.asarray(model.axial_coordinates_A, dtype=float)
    u_A = np.asarray(model.transverse_coordinates_A, dtype=float)
    valid_anchors = anchors[valid]
    positions = np.column_stack(
        [s_A[valid_anchors[:, 0]], u_A[valid_anchors[:, 1]]]
    )
    own_all_indices = modeled[valid]
    minimum = float(model.options.minimum_separation_A) - 1e-12
    clear = np.ones(len(positions), dtype=bool)

    occupied_indices = np.flatnonzero(occupancy > 1e-12)
    if len(occupied_indices):
        neighbor_count = min(2, len(occupied_indices))
        distances, neighbor_indices = cKDTree(hosts[occupied_indices]).query(
            positions, k=neighbor_count
        )
        if neighbor_count == 1:
            distances = distances[:, None]
            neighbor_indices = neighbor_indices[:, None]
        neighbor_all_indices = occupied_indices[neighbor_indices]
        distances = np.where(
            neighbor_all_indices == own_all_indices[:, None],
            np.inf,
            distances,
        )
        clear &= np.min(distances, axis=1) >= minimum

    active_extra = np.asarray(state.extra_active, dtype=bool) & (
        np.asarray(state.extra_scattering_equivalents, dtype=float) > 0.0
    )
    extras = np.asarray(atomistic_edit_addition_positions_1d(model, state))[
        active_extra
    ]
    if extras.size:
        clear &= _minimum_separation_admissible_mask(
            positions, extras, minimum
        )
    result[valid] = clear
    return result


def atomistic_edit_proposal_scores_1d(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    edit_penalty: Any,
    *,
    ablation: AtomisticEditAblation1D = "level1_physical",
    training_scan_batch_size: int | None = None,
) -> AtomisticEditProposalScores1D:
    """Score every dormant grid addition, host removal, and replacement.

    The data derivatives are obtained from the exact full-training potential
    adjoint. When ``training_scan_batch_size`` is set, that adjoint is summed
    deterministically over bounded scan batches rather than materialized in
    one reverse-mode graph. Addition scores are its correlation with the exact
    zero-offset AE-1 kernel. Host-removal scores use the transpose of the
    differentiable host renderer. Both directions are therefore measured per
    host-equivalent edit mass before the common ``lambda_edit`` term is applied.
    """
    if not isinstance(prepared, PreparedAtomisticEditReconstruction1D):
        raise TypeError("prepared must be PreparedAtomisticEditReconstruction1D")
    validate_atomistic_edit_state_1d(prepared.model, state)
    penalty = _finite("edit_penalty", edit_penalty, positive=True)
    if penalty not in prepared.model.options.edit_penalty_path:
        raise ValueError("edit_penalty must belong to the frozen model path")
    if ablation not in {"edit_only", "level1_physical"}:
        raise ValueError("ablation must be 'edit_only' or 'level1_physical'")
    if training_scan_batch_size is not None:
        _index("training_scan_batch_size", training_scan_batch_size)
    model = prepared.model
    rendered = render_atomistic_edit_potential_1d(model, state)
    potential_adjoint = _full_training_potential_adjoint(
        prepared,
        rendered,
        training_scan_batch_size=training_scan_batch_size,
    )
    potential_adjoint_host = np.asarray(
        jax.block_until_ready(potential_adjoint), dtype=float
    )

    addition_data = _addition_data_derivative_grid(
        model, potential_adjoint_host
    )
    removal_data = np.asarray(
        _host_site_parameter_adjoints(
            prepared, state, potential_adjoint
        )[0],
        dtype=float,
    )

    discovery = np.asarray(model.options.discovery_support.discovery_mask)
    candidate_anchors = np.argwhere(discovery).astype(np.int32)
    paired_anchors = _paired_replacement_anchors(model)
    if ablation == "level1_physical":
        addition_hard_values, removal_hard, pair_hard = (
            _hard_core_directional_derivatives(
                model,
                state,
                candidate_anchors,
                paired_anchors,
            )
        )
    else:
        addition_hard_values = np.zeros(len(candidate_anchors), dtype=float)
        removal_hard = np.zeros(len(removal_data), dtype=float)
        pair_hard = np.zeros(len(removal_data), dtype=float)

    addition_hard_grid = np.full(addition_data.shape, np.nan, dtype=float)
    addition_hard_grid[candidate_anchors[:, 0], candidate_anchors[:, 1]] = (
        addition_hard_values
    )
    addition_violation = np.full(addition_data.shape, -np.inf, dtype=float)
    candidate_violation = -(
        addition_data[candidate_anchors[:, 0], candidate_anchors[:, 1]]
        + addition_hard_values
        + penalty
    )
    if ablation == "level1_physical":
        candidate_admissible = _addition_admissible_mask(
            model, state, candidate_anchors
        )
        candidate_violation[~candidate_admissible] = -np.inf
    active_anchors = np.asarray(state.extra_anchor_indices)[
        np.asarray(state.extra_active, dtype=bool)
    ]
    if active_anchors.size:
        shape = discovery.shape
        candidate_keys = np.ravel_multi_index(candidate_anchors.T, shape)
        active_keys = np.ravel_multi_index(active_anchors.T, shape)
        candidate_violation[np.isin(candidate_keys, active_keys)] = -np.inf
    addition_violation[
        candidate_anchors[:, 0], candidate_anchors[:, 1]
    ] = candidate_violation

    removal_violation = -(removal_data + removal_hard + penalty)
    active_removal_indices = set(
        np.asarray(state.host_removal_indices)[
            np.asarray(state.host_removal_active, dtype=bool)
        ].tolist()
    )
    for index in active_removal_indices:
        removal_violation[int(index)] = -np.inf

    paired_violation = np.full(len(removal_data), -np.inf, dtype=float)
    paired_mass = np.zeros(len(removal_data), dtype=float)
    replacement_masses = np.unique(
        np.asarray(
            [
                min(
                    0.1,
                    model.options.max_scattering_equivalent_per_centre,
                ),
                min(
                    1.0,
                    model.options.max_scattering_equivalent_per_centre,
                ),
                model.options.max_scattering_equivalent_per_centre,
            ],
            dtype=float,
        )
    )
    paired_allowed = np.all(paired_anchors >= 0, axis=1)
    if active_removal_indices:
        paired_allowed[list(active_removal_indices)] = False
    if active_anchors.size and np.any(paired_allowed):
        allowed_indices = np.flatnonzero(paired_allowed)
        allowed_keys = np.ravel_multi_index(
            paired_anchors[allowed_indices].T, discovery.shape
        )
        active_keys = np.ravel_multi_index(active_anchors.T, discovery.shape)
        paired_allowed[allowed_indices[np.isin(allowed_keys, active_keys)]] = False
    if ablation == "level1_physical":
        paired_allowed &= _paired_addition_admissible_mask(
            model, state, paired_anchors
        )
    paired_indices = np.flatnonzero(paired_allowed)
    if len(paired_indices):
        anchors = paired_anchors[paired_indices]
        addition_direction = addition_data[anchors[:, 0], anchors[:, 1]]
        pair_addition_hard = (
            pair_hard[paired_indices] - removal_hard[paired_indices]
        )
        changes = (
            removal_data[paired_indices, None]
            + removal_hard[paired_indices, None]
            + replacement_masses[None, :]
            * (addition_direction + pair_addition_hard)[:, None]
            + penalty * (1.0 + replacement_masses[None, :])
        )
        best_mass_indices = np.argmin(changes, axis=1)
        paired_violation[paired_indices] = -changes[
            np.arange(len(paired_indices)), best_mass_indices
        ]
        paired_mass[paired_indices] = replacement_masses[best_mass_indices]

    choices: list[tuple[float, int, str, tuple[int, int] | int]] = []
    if np.any(np.isfinite(addition_violation)):
        flat_index = int(np.nanargmax(addition_violation))
        anchor = tuple(
            int(value) for value in np.unravel_index(flat_index, discovery.shape)
        )
        choices.append((float(addition_violation[anchor]), 0, "addition", anchor))
    if np.any(np.isfinite(removal_violation)):
        index = int(np.argmax(removal_violation))
        choices.append(
            (float(removal_violation[index]), 1, "host_removal", index)
        )
    if np.any(np.isfinite(paired_violation)):
        index = int(np.argmax(paired_violation))
        choices.append(
            (float(paired_violation[index]), 2, "paired_replacement", index)
        )
    if choices:
        # The fixed kind/index ordering makes exact ties deterministic. The
        # configured seed is recorded by the run and does not perturb scores.
        best_value, _, best_kind, best_index = max(
            choices, key=lambda item: (item[0], -item[1])
        )
    else:
        best_value, best_kind, best_index = -np.inf, "none", None
    return AtomisticEditProposalScores1D(
        addition_data_derivative_grid=addition_data,
        addition_hard_core_derivative_grid=addition_hard_grid,
        addition_violation_grid=addition_violation,
        host_removal_data_derivative=removal_data,
        host_removal_hard_core_derivative=removal_hard,
        host_removal_violation=removal_violation,
        paired_replacement_violation=paired_violation,
        paired_replacement_anchor_indices=paired_anchors,
        paired_replacement_scattering_equivalent=paired_mass,
        best_kind=best_kind,
        best_index=best_index,
        best_violation=float(best_value),
        edit_penalty=penalty,
        training_indices=np.asarray(prepared.training_indices, dtype=np.int32),
    )


def _finite_maximum(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else -np.inf


def _state_parameters(state: AtomisticEditState1D) -> dict[str, Array]:
    return {
        "host_removal_fractions": jnp.asarray(state.host_removal_fractions),
        "extra_position_offsets_A": jnp.asarray(state.extra_position_offsets_A),
        "extra_scattering_equivalents": jnp.asarray(
            state.extra_scattering_equivalents
        ),
        "host_displacement_controls": jnp.asarray(
            state.host_displacement_controls
        ),
    }


def _state_structure(state: AtomisticEditState1D) -> dict[str, Array]:
    """Return fixed-capacity discrete state as dynamic executable inputs.

    Host indices, addition anchors, and their active masks change at births,
    pruning, merging, and re-anchoring.  Passing them as arrays (rather than
    closing over a state object) lets one fixed-shape XLA executable serve
    every active set in a prepared reconstruction.
    """
    return {
        "host_removal_indices": jnp.asarray(state.host_removal_indices),
        "host_removal_active": jnp.asarray(state.host_removal_active),
        "extra_anchor_indices": jnp.asarray(state.extra_anchor_indices),
        "extra_active": jnp.asarray(state.extra_active),
    }


def _state_with_parameters(
    template: AtomisticEditState1D,
    parameters: Mapping[str, Array],
) -> AtomisticEditState1D:
    return replace(template, **dict(parameters))


def _state_from_structure_and_parameters(
    structure: Mapping[str, Array],
    parameters: Mapping[str, Array],
) -> AtomisticEditState1D:
    """Construct a traceable AE-1 state from two fixed-shape pytrees."""
    return AtomisticEditState1D(
        host_removal_indices=structure["host_removal_indices"],
        host_removal_fractions=parameters["host_removal_fractions"],
        host_removal_active=structure["host_removal_active"],
        extra_anchor_indices=structure["extra_anchor_indices"],
        extra_position_offsets_A=parameters["extra_position_offsets_A"],
        extra_scattering_equivalents=parameters[
            "extra_scattering_equivalents"
        ],
        extra_active=structure["extra_active"],
        host_displacement_controls=parameters[
            "host_displacement_controls"
        ],
    )


def _project_parameters(
    model: AtomisticEditModel1D,
    template: AtomisticEditState1D,
    parameters: Mapping[str, Array],
) -> dict[str, Array]:
    removal_active = jnp.asarray(template.host_removal_active)
    extra_active = jnp.asarray(template.extra_active)
    half_pixel = jnp.asarray(
        [
            0.5 * model.addition_kernel.axial_sampling_A,
            0.5 * model.addition_kernel.transverse_sampling_A,
        ]
    )
    maximum_displacement = jnp.asarray(model.host_model.maximum_displacement)
    return {
        "host_removal_fractions": jnp.where(
            removal_active,
            jnp.clip(parameters["host_removal_fractions"], 0.0, 1.0),
            0.0,
        ),
        "extra_position_offsets_A": jnp.where(
            extra_active[:, None],
            jnp.clip(
                parameters["extra_position_offsets_A"],
                -half_pixel[None, :],
                half_pixel[None, :],
            ),
            0.0,
        ),
        "extra_scattering_equivalents": jnp.where(
            extra_active,
            jnp.clip(
                parameters["extra_scattering_equivalents"],
                0.0,
                model.options.max_scattering_equivalent_per_centre,
            ),
            0.0,
        ),
        "host_displacement_controls": jnp.clip(
            parameters["host_displacement_controls"],
            -maximum_displacement,
            maximum_displacement,
        ),
    }


def _masked_gradients(
    template: AtomisticEditState1D,
    gradients: Mapping[str, Array],
    *,
    freeze_positions: bool,
) -> dict[str, Array]:
    result = {
        "host_removal_fractions": gradients["host_removal_fractions"]
        * jnp.asarray(template.host_removal_active),
        "extra_position_offsets_A": gradients["extra_position_offsets_A"]
        * jnp.asarray(template.extra_active)[:, None],
        "extra_scattering_equivalents": gradients[
            "extra_scattering_equivalents"
        ]
        * jnp.asarray(template.extra_active),
        "host_displacement_controls": gradients["host_displacement_controls"],
    }
    if freeze_positions:
        result["extra_position_offsets_A"] = jnp.zeros_like(
            result["extra_position_offsets_A"]
        )
    return result


def _projected_gradient_norm(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
    gradients: Mapping[str, Any],
    *,
    freeze_positions: bool = False,
) -> float:
    maximum_mass = float(model.options.max_scattering_equivalent_per_centre)
    half_pixel = np.asarray(
        [
            0.5 * model.addition_kernel.axial_sampling_A,
            0.5 * model.addition_kernel.transverse_sampling_A,
        ]
    )
    maximum_displacement = float(np.asarray(model.host_model.maximum_displacement))

    def projected(
        values: np.ndarray,
        gradient: np.ndarray,
        lower: Any,
        upper: Any,
        active: np.ndarray,
    ) -> np.ndarray:
        lower_array = np.broadcast_to(np.asarray(lower), values.shape)
        upper_array = np.broadcast_to(np.asarray(upper), values.shape)
        result = gradient.copy()
        result = np.where(
            values <= lower_array + 1e-12,
            np.minimum(result, 0.0),
            result,
        )
        result = np.where(
            values >= upper_array - 1e-12,
            np.maximum(result, 0.0),
            result,
        )
        return np.where(active, result, 0.0)

    parts = []
    removal_active = np.asarray(state.host_removal_active, dtype=bool)
    extra_active = np.asarray(state.extra_active, dtype=bool)
    parts.append(
        projected(
            np.asarray(state.host_removal_fractions, dtype=float),
            np.asarray(gradients["host_removal_fractions"], dtype=float),
            0.0,
            1.0,
            removal_active,
        ).reshape(-1)
    )
    parts.append(
        projected(
            np.asarray(state.extra_scattering_equivalents, dtype=float),
            np.asarray(gradients["extra_scattering_equivalents"], dtype=float),
            0.0,
            maximum_mass,
            extra_active,
        ).reshape(-1)
    )
    if not freeze_positions:
        parts.append(
            projected(
                np.asarray(state.extra_position_offsets_A, dtype=float),
                np.asarray(
                    gradients["extra_position_offsets_A"], dtype=float
                ),
                -half_pixel,
                half_pixel,
                extra_active[:, None],
            ).reshape(-1)
        )
    controls = np.asarray(state.host_displacement_controls, dtype=float)
    parts.append(
        projected(
            controls,
            np.asarray(gradients["host_displacement_controls"], dtype=float),
            -maximum_displacement,
            maximum_displacement,
            np.ones_like(controls, dtype=bool),
        ).reshape(-1)
    )
    combined = np.concatenate(parts)
    return float(np.max(np.abs(combined))) if combined.size else 0.0


@dataclass(eq=False)
class _CompiledObjectiveValueAndGradient1D:
    """One prepared-problem executable plus trace/reuse diagnostics."""

    function: Any
    trace_counter: list[int]
    lookup_count: int = 1


@dataclass(eq=False)
class _CompiledPotentialValueAndAdjoint1D:
    """Reusable unnormalized scan-batch count value and potential adjoint."""

    function: Any
    trace_counter: list[int]


@dataclass(frozen=True, eq=False)
class _HostAdjointGeometry1D:
    """CPU-resident immutable compact host geometry reused across updates."""

    patches: np.ndarray
    patch_starts: np.ndarray
    axial_sampling: float
    transverse_sampling: float


_COMPILED_OBJECTIVE_CACHE: weakref.WeakKeyDictionary[
    PreparedAtomisticEditReconstruction1D,
    dict[AtomisticEditAblation1D, _CompiledObjectiveValueAndGradient1D],
] = weakref.WeakKeyDictionary()
_COMPILED_POTENTIAL_ADJOINT_CACHE: weakref.WeakKeyDictionary[
    PreparedAtomisticEditReconstruction1D,
    _CompiledPotentialValueAndAdjoint1D,
] = weakref.WeakKeyDictionary()
_HOST_ADJOINT_GEOMETRY_CACHE: weakref.WeakKeyDictionary[
    PreparedAtomisticEditReconstruction1D,
    _HostAdjointGeometry1D,
] = weakref.WeakKeyDictionary()
_COMPILED_OBJECTIVE_CACHE_LOCK = threading.RLock()


def _objective_from_dynamic_state(
    prepared: PreparedAtomisticEditReconstruction1D,
    parameters: Mapping[str, Array],
    structure: Mapping[str, Array],
    edit_penalty: Array,
    ablation: AtomisticEditAblation1D,
) -> Array:
    """Evaluate the unchanged AE-2 objective with dynamic topology and λ.

    The public component evaluator validates a Python scalar penalty.  The
    compiled path instead validates λ before entry and forms the same weighted
    edit-mass term here, allowing λ to remain a scalar device argument rather
    than a compilation constant.
    """
    candidate = _state_from_structure_and_parameters(structure, parameters)
    count = _count_loss(prepared, candidate, prepared.training_indices)
    return count + _prior_objective_from_dynamic_state(
        prepared,
        parameters,
        structure,
        edit_penalty,
        ablation,
    )


def _prior_objective_from_dynamic_state(
    prepared: PreparedAtomisticEditReconstruction1D,
    parameters: Mapping[str, Array],
    structure: Mapping[str, Array],
    edit_penalty: Array,
    ablation: AtomisticEditAblation1D,
) -> Array:
    """Return the edit and physical prior without evaluating any scans."""
    candidate = _state_from_structure_and_parameters(structure, parameters)
    prior = atomistic_edit_prior_components_1d(
        prepared.model,
        candidate,
        1.0,
    )
    weighted_edit = jnp.asarray(
        edit_penalty, dtype=jnp.asarray(prior.edit_mass).dtype
    ) * prior.edit_mass
    physical = (
        prior.elastic_penalty + prior.hard_core_penalty
        if ablation == "level1_physical"
        else jnp.asarray(0.0, dtype=jnp.asarray(prior.edit_mass).dtype)
    )
    return weighted_edit + physical


def _make_compiled_objective_value_and_gradient(
    prepared: PreparedAtomisticEditReconstruction1D,
    ablation: AtomisticEditAblation1D,
) -> _CompiledObjectiveValueAndGradient1D:
    # Hold only a weak Python reference in the cached function.  The traced
    # executable owns its device constants, while a discarded prepared
    # problem can still leave the WeakKeyDictionary and release the wrapper.
    prepared_reference = weakref.ref(prepared)
    trace_counter = [0]

    def objective(
        parameters: Mapping[str, Array],
        structure: Mapping[str, Array],
        edit_penalty: Array,
    ) -> Array:
        # This Python side effect runs only when JAX traces a new input
        # signature.  It provides a robust regression diagnostic without a
        # machine-dependent wall-time threshold.
        trace_counter[0] += 1
        current = prepared_reference()
        if current is None:  # pragma: no cover - impossible for a live call
            raise RuntimeError("prepared AE-2 problem was released during tracing")
        return _objective_from_dynamic_state(
            current,
            parameters,
            structure,
            edit_penalty,
            ablation,
        )

    return _CompiledObjectiveValueAndGradient1D(
        function=jax.jit(jax.value_and_grad(objective, argnums=0)),
        trace_counter=trace_counter,
    )


def _compiled_objective_value_and_gradient(
    prepared: PreparedAtomisticEditReconstruction1D,
    ablation: AtomisticEditAblation1D,
) -> _CompiledObjectiveValueAndGradient1D:
    """Return the reusable fixed-shape executable for a prepared problem."""
    with _COMPILED_OBJECTIVE_CACHE_LOCK:
        by_ablation = _COMPILED_OBJECTIVE_CACHE.get(prepared)
        if by_ablation is None:
            by_ablation = {}
            _COMPILED_OBJECTIVE_CACHE[prepared] = by_ablation
        result = by_ablation.get(ablation)
        if result is None:
            result = _make_compiled_objective_value_and_gradient(
                prepared, ablation
            )
            by_ablation[ablation] = result
        else:
            result.lookup_count += 1
        return result


def _compiled_potential_adjoint(
    prepared: PreparedAtomisticEditReconstruction1D,
) -> _CompiledPotentialValueAndAdjoint1D:
    with _COMPILED_OBJECTIVE_CACHE_LOCK:
        result = _COMPILED_POTENTIAL_ADJOINT_CACHE.get(prepared)
        if result is not None:
            return result
        prepared_reference = weakref.ref(prepared)
        trace_counter = [0]

        def count_sum(potential: Array, scan_indices: Array) -> Array:
            trace_counter[0] += 1
            current = prepared_reference()
            if current is None:  # pragma: no cover - impossible for a live call
                raise RuntimeError(
                    "prepared AE-2 problem was released during tracing"
                )
            return _count_deviance_sum_from_potential(
                current, potential, scan_indices
            )[0]

        result = _CompiledPotentialValueAndAdjoint1D(
            function=jax.jit(jax.value_and_grad(count_sum, argnums=0)),
            trace_counter=trace_counter,
        )
        _COMPILED_POTENTIAL_ADJOINT_CACHE[prepared] = result
        return result


def _keys_cubic_kernel_numpy(distance: np.ndarray) -> np.ndarray:
    """Evaluate the established Keys ``a=-1/2`` kernel in NumPy."""
    parameter = np.asarray(-0.5, dtype=distance.dtype)
    absolute = np.abs(distance)
    inner = (
        (parameter + 2.0) * absolute - (parameter + 3.0)
    ) * absolute**2 + 1.0
    outer = (
        ((parameter * absolute - 5.0 * parameter) * absolute + 8.0 * parameter)
        * absolute
        - 4.0 * parameter
    )
    return np.where(
        absolute < 1.0,
        inner,
        np.where(absolute < 2.0, outer, 0.0),
    )


def _keys_cubic_kernel_derivative_numpy(distance: np.ndarray) -> np.ndarray:
    """Differentiate the established Keys kernel with respect to distance."""
    parameter = np.asarray(-0.5, dtype=distance.dtype)
    absolute = np.abs(distance)
    inner = (
        3.0 * (parameter + 2.0) * absolute**2
        - 2.0 * (parameter + 3.0) * absolute
    )
    outer = (
        3.0 * parameter * absolute**2
        - 10.0 * parameter * absolute
        + 8.0 * parameter
    )
    radial = np.where(
        absolute < 1.0,
        inner,
        np.where(absolute < 2.0, outer, 0.0),
    )
    return radial * np.sign(distance)


def _shift_patch_axis_keys_cubic_numpy_1d(
    patches: np.ndarray,
    shifts: np.ndarray,
    *,
    axis: int,
    derivative: bool = False,
) -> np.ndarray:
    """Shift a bounded patch batch, or differentiate it, without XLA."""
    if patches.ndim != 3 or shifts.shape != (len(patches),):
        raise ValueError("patches/shifts have incompatible local-adjoint shapes")
    work_dtype = np.result_type(patches.dtype, shifts.dtype, np.float32)
    work = np.asarray(patches, dtype=work_dtype)
    shift_values = np.asarray(shifts, dtype=work_dtype)
    base = np.floor(-shift_values).astype(np.int64)
    offsets = base[:, None] + np.arange(-1, 3, dtype=np.int64)[None, :]
    distances = -shift_values[:, None] - offsets.astype(work_dtype)
    weights = (
        -_keys_cubic_kernel_derivative_numpy(distances)
        if derivative
        else _keys_cubic_kernel_numpy(distances)
    )
    batch = np.arange(len(work), dtype=np.int64)[:, None, None, None]
    if axis == 0:
        targets = np.arange(work.shape[1], dtype=np.int64)
        indices = targets[None, :, None] + offsets[:, None, :]
        valid = (indices >= 0) & (indices < work.shape[1])
        columns = np.arange(work.shape[2], dtype=np.int64)[None, None, None, :]
        samples = work[
            batch,
            np.clip(indices, 0, work.shape[1] - 1)[:, :, :, None],
            columns,
        ]
        return np.sum(
            np.where(valid[:, :, :, None], samples, 0.0)
            * weights[:, None, :, None],
            axis=2,
        )
    if axis == 1:
        targets = np.arange(work.shape[2], dtype=np.int64)
        indices = targets[None, :, None] + offsets[:, None, :]
        valid = (indices >= 0) & (indices < work.shape[2])
        rows = np.arange(work.shape[1], dtype=np.int64)[None, :, None, None]
        samples = work[
            batch,
            rows,
            np.clip(indices, 0, work.shape[2] - 1)[:, None, :, :],
        ]
        return np.sum(
            np.where(valid[:, None, :, :], samples, 0.0)
            * weights[:, None, None, :],
            axis=3,
        )
    raise ValueError("axis must be zero or one")


def _host_patch_adjoints_numpy(
    vacancies: np.ndarray,
    site_displacements: np.ndarray,
    patches: np.ndarray,
    cotangent_patches: np.ndarray,
    *,
    axial_sampling: float,
    transverse_sampling: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Contract local host patches and their analytic shift derivatives."""
    shift_s = site_displacements[:, 0] / axial_sampling
    shift_u = site_displacements[:, 1] / transverse_sampling
    shifted_s = _shift_patch_axis_keys_cubic_numpy_1d(
        patches, shift_s, axis=0
    )
    shifted = _shift_patch_axis_keys_cubic_numpy_1d(
        shifted_s, shift_u, axis=1
    )
    derivative_s_stage = _shift_patch_axis_keys_cubic_numpy_1d(
        patches, shift_s, axis=0, derivative=True
    )
    derivative_s = _shift_patch_axis_keys_cubic_numpy_1d(
        derivative_s_stage, shift_u, axis=1
    )
    derivative_u = _shift_patch_axis_keys_cubic_numpy_1d(
        shifted_s, shift_u, axis=1, derivative=True
    )
    vacancy_adjoint = -np.sum(cotangent_patches * shifted, axis=(1, 2))
    occupancy = 1.0 - vacancies
    displacement_adjoint = np.column_stack(
        [
            occupancy
            * np.sum(cotangent_patches * derivative_s, axis=(1, 2))
            / axial_sampling,
            occupancy
            * np.sum(cotangent_patches * derivative_u, axis=(1, 2))
            / transverse_sampling,
        ]
    )
    return vacancy_adjoint, displacement_adjoint


def _cotangent_site_patches_numpy(
    potential_adjoint: np.ndarray,
    patch_starts: np.ndarray,
    patch_shape: tuple[int, int],
) -> np.ndarray:
    """Gather the transpose-scatter cotangent for compact site patches."""
    offsets_s = np.arange(patch_shape[0], dtype=np.int64)
    offsets_u = np.arange(patch_shape[1], dtype=np.int64)
    rows = patch_starts[:, 0, None, None] + offsets_s[None, :, None]
    columns = patch_starts[:, 1, None, None] + offsets_u[None, None, :]
    rows = np.broadcast_to(rows, (len(patch_starts), *patch_shape))
    columns = np.broadcast_to(columns, (len(patch_starts), *patch_shape))
    valid = (
        (rows >= 0)
        & (rows < potential_adjoint.shape[0])
        & (columns >= 0)
        & (columns < potential_adjoint.shape[1])
    )
    gathered = potential_adjoint[
        np.clip(rows, 0, potential_adjoint.shape[0] - 1),
        np.clip(columns, 0, potential_adjoint.shape[1] - 1),
    ]
    return np.where(valid, gathered, 0.0)


def _host_adjoint_geometry(
    prepared: PreparedAtomisticEditReconstruction1D,
) -> _HostAdjointGeometry1D:
    """Materialize compact host patches on CPU once per prepared problem."""
    with _COMPILED_OBJECTIVE_CACHE_LOCK:
        result = _HOST_ADJOINT_GEOMETRY_CACHE.get(prepared)
        if result is not None:
            return result
        host = prepared.model.host_model
        patches = np.ascontiguousarray(
            jax.device_get(jnp.asarray(host.site_patches))
        )
        starts = np.ascontiguousarray(
            np.asarray(host.patch_starts, dtype=np.int64)
        )
        patches.setflags(write=False)
        starts.setflags(write=False)
        result = _HostAdjointGeometry1D(
            patches=patches,
            patch_starts=starts,
            axial_sampling=float(np.asarray(host.axial_sampling)),
            transverse_sampling=float(np.asarray(host.transverse_sampling)),
        )
        _HOST_ADJOINT_GEOMETRY_CACHE[prepared] = result
        return result


def _host_site_parameter_adjoints(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    potential_adjoint: Array,
) -> tuple[Array, Array]:
    """Return dense vacancy and per-site displacement adjoints in chunks."""
    host = prepared.model.host_model
    sites = jnp.asarray(host.site_coordinates)
    controls = jnp.asarray(state.host_displacement_controls)
    displacements = lattice_site_displacements_1d(
        sites,
        controls,
        host.control_coordinates_s,
        host.control_coordinates_u,
    )
    vacancies = _dense_host_removals(prepared.model, state)
    vacancy_values = np.asarray(
        jax.device_get(jax.block_until_ready(vacancies))
    )
    displacement_values = np.asarray(
        jax.device_get(jax.block_until_ready(displacements))
    )
    geometry = _host_adjoint_geometry(prepared)
    patches = geometry.patches
    starts = geometry.patch_starts
    cotangent = np.asarray(
        jax.device_get(jax.block_until_ready(potential_adjoint))
    )
    patch_shape = (patches.shape[1], patches.shape[2])
    vacancy_adjoint = np.empty(len(patches), dtype=np.dtype(patches.dtype))
    displacement_adjoint = np.empty(
        (len(patches), 2), dtype=np.dtype(displacements.dtype)
    )
    for start in range(0, len(patches), _HOST_ADJOINT_SITE_BATCH_SIZE):
        stop = min(start + _HOST_ADJOINT_SITE_BATCH_SIZE, len(patches))
        patch_chunk = patches[start:stop]
        cotangent_chunk = _cotangent_site_patches_numpy(
            cotangent,
            starts[start:stop],
            patch_shape,
        )
        vacancy_gradient, displacement_gradient = _host_patch_adjoints_numpy(
            vacancy_values[start:stop],
            displacement_values[start:stop],
            np.asarray(patch_chunk),
            cotangent_chunk,
            axial_sampling=geometry.axial_sampling,
            transverse_sampling=geometry.transverse_sampling,
        )
        vacancy_adjoint[start:stop] = vacancy_gradient
        displacement_adjoint[start:stop] = displacement_gradient
    return (
        jnp.asarray(vacancy_adjoint, dtype=vacancies.dtype),
        jnp.asarray(displacement_adjoint, dtype=displacements.dtype),
    )


def _control_axis_interpolation_numpy(
    values: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(coordinates) == 1:
        zeros = np.zeros(len(values), dtype=np.int64)
        return zeros, zeros, np.zeros(len(values), dtype=float)
    indices = (
        (values - coordinates[0])
        / (coordinates[-1] - coordinates[0])
        * (len(coordinates) - 1)
    )
    indices = np.clip(indices, 0.0, len(coordinates) - 1.0)
    lower = np.floor(indices).astype(np.int64)
    upper = np.minimum(lower + 1, len(coordinates) - 1)
    return lower, upper, indices - lower


def _control_adjoint_from_site_displacements_numpy(
    host: Any,
    site_displacement_adjoint: Array,
) -> Array:
    """Apply the exact bilinear/nearest control transpose without JAX VJP."""
    sites = np.asarray(host.site_coordinates, dtype=float)
    coordinates_s = np.asarray(host.control_coordinates_s, dtype=float)
    coordinates_u = np.asarray(host.control_coordinates_u, dtype=float)
    site_adjoint = np.asarray(
        jax.device_get(jax.block_until_ready(site_displacement_adjoint))
    )
    lower_s, upper_s, fraction_s = _control_axis_interpolation_numpy(
        sites[:, 0], coordinates_s
    )
    lower_u, upper_u, fraction_u = _control_axis_interpolation_numpy(
        sites[:, 1], coordinates_u
    )
    result = np.zeros(
        (len(coordinates_s), len(coordinates_u), 2),
        dtype=site_adjoint.dtype,
    )
    corners = (
        (lower_s, lower_u, (1.0 - fraction_s) * (1.0 - fraction_u)),
        (upper_s, lower_u, fraction_s * (1.0 - fraction_u)),
        (lower_s, upper_u, (1.0 - fraction_s) * fraction_u),
        (upper_s, upper_u, fraction_s * fraction_u),
    )
    for indices_s, indices_u, weight in corners:
        for component in range(2):
            np.add.at(
                result[..., component],
                (indices_s, indices_u),
                weight * site_adjoint[:, component],
            )
    return jnp.asarray(
        result,
        dtype=jnp.asarray(site_displacement_adjoint).dtype,
    )


def _shift_patch_axis_linear_numpy_1d(
    patches: np.ndarray,
    shifts: np.ndarray,
    *,
    axis: int,
    derivative: bool = False,
) -> np.ndarray:
    """Apply the exact zero-extended linear extra-centre shift in NumPy."""
    work_dtype = np.result_type(patches.dtype, shifts.dtype, np.float32)
    work = np.asarray(patches, dtype=work_dtype)
    shift_values = np.asarray(shifts, dtype=work_dtype)
    batch = np.arange(len(work), dtype=np.int64)[:, None, None]
    if axis == 0:
        targets = np.arange(work.shape[1], dtype=work_dtype)[None, :]
        source = targets - shift_values[:, None]
        lower = np.floor(source).astype(np.int64)
        fraction = source - lower.astype(work_dtype)
        upper = lower + 1
        columns = np.arange(work.shape[2], dtype=np.int64)[None, None, :]

        def samples(indices: np.ndarray) -> np.ndarray:
            valid = (indices >= 0) & (indices < work.shape[1])
            selected = work[
                batch,
                np.clip(indices, 0, work.shape[1] - 1)[:, :, None],
                columns,
            ]
            return np.where(valid[:, :, None], selected, 0.0)

        lower_samples = samples(lower)
        upper_samples = samples(upper)
        if derivative:
            return lower_samples - upper_samples
        return (
            lower_samples * (1.0 - fraction)[:, :, None]
            + upper_samples * fraction[:, :, None]
        )
    if axis == 1:
        targets = np.arange(work.shape[2], dtype=work_dtype)[None, :]
        source = targets - shift_values[:, None]
        lower = np.floor(source).astype(np.int64)
        fraction = source - lower.astype(work_dtype)
        upper = lower + 1
        rows = np.arange(work.shape[1], dtype=np.int64)[None, :, None]

        def samples(indices: np.ndarray) -> np.ndarray:
            valid = (indices >= 0) & (indices < work.shape[2])
            selected = work[
                batch,
                rows,
                np.clip(indices, 0, work.shape[2] - 1)[:, None, :],
            ]
            return np.where(valid[:, None, :], selected, 0.0)

        lower_samples = samples(lower)
        upper_samples = samples(upper)
        if derivative:
            return lower_samples - upper_samples
        return (
            lower_samples * (1.0 - fraction)[:, None, :]
            + upper_samples * fraction[:, None, :]
        )
    raise ValueError("axis must be zero or one")


def _extra_parameter_data_gradients_numpy(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    parameters: Mapping[str, Array],
    potential_adjoint: Array,
) -> Mapping[str, Array]:
    """Pull back bounded addition patches without a full-grid scatter VJP."""
    gradients = {
        name: jnp.zeros_like(value) for name, value in parameters.items()
    }
    offsets_A = np.asarray(state.extra_position_offsets_A)
    if not len(offsets_A):
        return gradients
    model = prepared.model
    kernel = np.asarray(model.addition_kernel.unit_integrated_values)
    centre = np.asarray(model.addition_kernel.centre_index, dtype=float)
    anchors = np.asarray(state.extra_anchor_indices, dtype=np.int64)
    start_offset = np.floor(-centre + 0.5).astype(np.int64)
    starts = anchors + start_offset[None, :]
    cotangent_patches = _cotangent_site_patches_numpy(
        np.asarray(jax.device_get(jax.block_until_ready(potential_adjoint))),
        starts,
        kernel.shape,
    )
    sampling = np.asarray(
        [
            model.addition_kernel.axial_sampling_A,
            model.addition_kernel.transverse_sampling_A,
        ],
        dtype=offsets_A.dtype,
    )
    base_shift = -(start_offset.astype(offsets_A.dtype) + centre)
    shifts = base_shift[None, :] + offsets_A / sampling[None, :]
    patches = np.broadcast_to(kernel, (len(offsets_A), *kernel.shape))
    shifted_s = _shift_patch_axis_linear_numpy_1d(
        patches, shifts[:, 0], axis=0
    )
    shifted = _shift_patch_axis_linear_numpy_1d(
        shifted_s, shifts[:, 1], axis=1
    )
    derivative_s_stage = _shift_patch_axis_linear_numpy_1d(
        patches, shifts[:, 0], axis=0, derivative=True
    )
    derivative_s = _shift_patch_axis_linear_numpy_1d(
        derivative_s_stage, shifts[:, 1], axis=1
    )
    derivative_u = _shift_patch_axis_linear_numpy_1d(
        shifted_s, shifts[:, 1], axis=1, derivative=True
    )
    active = np.asarray(state.extra_active, dtype=bool)
    masses = np.asarray(state.extra_scattering_equivalents)
    scale = float(
        model.addition_kernel.host_equivalent_integrated_scattering
    )
    mass_adjoint = active * scale * np.sum(
        cotangent_patches * shifted, axis=(1, 2)
    )
    offset_adjoint = np.column_stack(
        [
            active
            * masses
            * scale
            * np.sum(cotangent_patches * derivative_s, axis=(1, 2))
            / sampling[0],
            active
            * masses
            * scale
            * np.sum(cotangent_patches * derivative_u, axis=(1, 2))
            / sampling[1],
        ]
    )
    gradients["extra_scattering_equivalents"] = jnp.asarray(
        mass_adjoint,
        dtype=jnp.asarray(parameters["extra_scattering_equivalents"]).dtype,
    )
    gradients["extra_position_offsets_A"] = jnp.asarray(
        offset_adjoint,
        dtype=jnp.asarray(parameters["extra_position_offsets_A"]).dtype,
    )
    return gradients


def _parameter_data_gradients_from_potential_adjoint(
    prepared: PreparedAtomisticEditReconstruction1D,
    parameters: Mapping[str, Array],
    structure: Mapping[str, Array],
    potential_adjoint: Array,
) -> Mapping[str, Array]:
    """Apply the atom renderer transpose without a monolithic renderer VJP."""
    state = _state_from_structure_and_parameters(structure, parameters)
    vacancy_adjoint, site_displacement_adjoint = (
        _host_site_parameter_adjoints(
            prepared, state, potential_adjoint
        )
    )
    host = prepared.model.host_model
    control_adjoint = _control_adjoint_from_site_displacements_numpy(
        host, site_displacement_adjoint
    )
    removal_adjoint = jnp.where(
        jnp.asarray(state.host_removal_active),
        vacancy_adjoint[jnp.asarray(state.host_removal_indices)],
        jnp.zeros((), dtype=vacancy_adjoint.dtype),
    )

    gradients = dict(
        _extra_parameter_data_gradients_numpy(
            prepared,
            state,
            parameters,
            potential_adjoint,
        )
    )
    gradients["host_removal_fractions"] = (
        gradients["host_removal_fractions"] + removal_adjoint
    )
    gradients["host_displacement_controls"] = (
        gradients["host_displacement_controls"] + control_adjoint
    )
    return gradients


def _full_training_count_value_and_potential_adjoint(
    prepared: PreparedAtomisticEditReconstruction1D,
    potential: Array,
    *,
    training_scan_batch_size: int | None,
) -> tuple[Array, Array]:
    """Return exact mean deviance and its adjoint with bounded scan memory."""
    batches = _scan_batches(
        prepared.training_indices, training_scan_batch_size
    )
    if len(batches) == 1:
        return jax.value_and_grad(
            lambda current: _count_loss_from_potential(
                prepared, current, prepared.training_indices
            )
        )(potential)
    total_valid = _valid_observation_count(
        prepared, prepared.training_indices
    )
    executable = _compiled_potential_adjoint(prepared).function
    value_total = 0.0
    accumulated = np.zeros_like(np.asarray(potential))
    for batch in batches:
        value, gradient = executable(
            potential, jnp.asarray(batch, dtype=jnp.int32)
        )
        value = jax.device_get(jax.block_until_ready(value))
        gradient = jax.device_get(jax.block_until_ready(gradient))
        value_total += float(np.asarray(value))
        accumulated += np.asarray(gradient)
    dtype = jnp.asarray(potential).dtype
    return (
        jnp.asarray(value_total / total_valid, dtype=dtype),
        jnp.asarray(accumulated / total_valid, dtype=dtype),
    )


def _full_training_potential_adjoint(
    prepared: PreparedAtomisticEditReconstruction1D,
    potential: Array,
    *,
    training_scan_batch_size: int | None,
) -> Array:
    return _full_training_count_value_and_potential_adjoint(
        prepared,
        potential,
        training_scan_batch_size=training_scan_batch_size,
    )[1]


def _compiled_objective_cache_info(
    prepared: PreparedAtomisticEditReconstruction1D,
    ablation: AtomisticEditAblation1D,
) -> Mapping[str, int]:
    """Return private trace diagnostics used by the performance regression."""
    executable = _compiled_objective_value_and_gradient(prepared, ablation)
    return MappingProxyType(
        {
            "lookup_count": int(executable.lookup_count),
            "trace_count": int(executable.trace_counter[0]),
        }
    )


def _clear_compiled_objective_cache() -> None:
    """Clear only the private AE-2 wrapper cache (primarily for tests)."""
    with _COMPILED_OBJECTIVE_CACHE_LOCK:
        _COMPILED_OBJECTIVE_CACHE.clear()
        _COMPILED_POTENTIAL_ADJOINT_CACHE.clear()
        _HOST_ADJOINT_GEOMETRY_CACHE.clear()


def _batched_objective_value_and_gradients_from_parameters(
    prepared: PreparedAtomisticEditReconstruction1D,
    parameters: Mapping[str, Array],
    structure: Mapping[str, Array],
    penalty: Array,
    ablation: AtomisticEditAblation1D,
    training_scan_batch_size: int,
) -> tuple[Array, Mapping[str, Array]]:
    """Apply the exact factorized chain rule without a fused scan/renderer JIT.

    Only the count derivative with respect to the rendered potential is scan
    batched. Host patch adjoints are evaluated in bounded site chunks, their
    displacement adjoints are pulled back through the small control
    interpolation once, additions are pulled back separately, and the prior
    is differentiated once. This avoids stochastic gradients and every
    monolithic all-sites renderer VJP.
    """

    candidate = _state_from_structure_and_parameters(structure, parameters)
    potential = render_atomistic_edit_potential_1d(
        prepared.model, candidate
    )
    count_value, potential_adjoint = (
        _full_training_count_value_and_potential_adjoint(
            prepared,
            potential,
            training_scan_batch_size=training_scan_batch_size,
        )
    )
    data_gradients = _parameter_data_gradients_from_potential_adjoint(
        prepared,
        parameters,
        structure,
        potential_adjoint,
    )
    prior_value, prior_gradients = jax.value_and_grad(
        lambda values: _prior_objective_from_dynamic_state(
            prepared,
            values,
            structure,
            penalty,
            ablation,
        )
    )(parameters)
    gradients = jax.tree_util.tree_map(
        lambda data, prior: data + prior,
        data_gradients,
        prior_gradients,
    )
    return count_value + prior_value, gradients


def _reference_objective_value_and_gradients(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    edit_penalty: float,
    ablation: AtomisticEditAblation1D,
) -> tuple[Array, Mapping[str, Array]]:
    """Eager reference for executable-equivalence tests and diagnostics."""
    parameters = _state_parameters(state)
    structure = _state_structure(state)
    penalty = jnp.asarray(
        edit_penalty,
        dtype=jnp.asarray(state.host_removal_fractions).dtype,
    )
    return jax.value_and_grad(
        lambda values: _objective_from_dynamic_state(
            prepared,
            values,
            structure,
            penalty,
            ablation,
        )
    )(parameters)


def _objective_value_and_gradients(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    edit_penalty: float,
    ablation: AtomisticEditAblation1D,
    *,
    training_scan_batch_size: int | None = None,
) -> tuple[Array, Mapping[str, Array]]:
    parameters = _state_parameters(state)
    structure = _state_structure(state)
    penalty = jnp.asarray(
        edit_penalty,
        dtype=jnp.asarray(state.host_removal_fractions).dtype,
    )
    if (
        training_scan_batch_size is not None
        and training_scan_batch_size < int(np.asarray(prepared.training_indices).size)
    ):
        return _batched_objective_value_and_gradients_from_parameters(
            prepared,
            parameters,
            structure,
            penalty,
            ablation,
            training_scan_batch_size,
        )
    executable = _compiled_objective_value_and_gradient(prepared, ablation)
    return executable.function(parameters, structure, penalty)


def _interpolate_parameters(
    first: Mapping[str, Array],
    second: Mapping[str, Array],
    fraction: float,
) -> dict[str, Array]:
    return {
        name: first[name] + fraction * (second[name] - first[name])
        for name in first
    }


def _refine_state(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    edit_penalty: float,
    *,
    ablation: AtomisticEditAblation1D,
    updates: int,
    learning_rate: float,
    gradient_clip: float,
    maximum_backtracking_steps: int,
    freeze_positions: bool = False,
    training_scan_batch_size: int | None = None,
) -> tuple[AtomisticEditState1D, int]:
    """Run one freshly initialized projected Adam refinement."""
    if updates == 0:
        return state, 0
    try:
        import optax
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("AE-2 refinement requires Optax") from error
    optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.adam(learning_rate),
    )
    parameters = _project_parameters(
        prepared.model, state, _state_parameters(state)
    )
    optimizer_state = optimizer.init(parameters)
    structure = _state_structure(state)
    penalty = jnp.asarray(
        edit_penalty,
        dtype=jnp.asarray(state.host_removal_fractions).dtype,
    )
    # Fixed capacities make every active-set topology the same collection of
    # array shapes.  Reuse the prepared-problem executable across births,
    # pruning, re-anchoring, homotopy λ values, polish, and debiasing.
    use_scan_batches = (
        training_scan_batch_size is not None
        and training_scan_batch_size < int(np.asarray(prepared.training_indices).size)
    )
    value_and_grad = (
        None
        if use_scan_batches
        else _compiled_objective_value_and_gradient(prepared, ablation).function
    )
    accepted_updates = 0
    for _ in range(updates):
        if use_scan_batches:
            value, gradients = (
                _batched_objective_value_and_gradients_from_parameters(
                    prepared,
                    parameters,
                    structure,
                    penalty,
                    ablation,
                    training_scan_batch_size,
                )
            )
        else:
            value, gradients = value_and_grad(parameters, structure, penalty)
        if not np.isfinite(float(np.asarray(jax.block_until_ready(value)))):
            raise FloatingPointError("atomistic-edit objective became non-finite")
        gradients = _masked_gradients(
            state, gradients, freeze_positions=freeze_positions
        )
        updates_tree, next_optimizer_state = optimizer.update(
            gradients, optimizer_state, parameters
        )
        proposed = optax.apply_updates(parameters, updates_tree)
        proposed = _project_parameters(prepared.model, state, proposed)
        selected = None
        for backtrack in range(maximum_backtracking_steps + 1):
            fraction = 0.5**backtrack
            trial = _project_parameters(
                prepared.model,
                state,
                _interpolate_parameters(parameters, proposed, fraction),
            )
            trial_state = _state_with_parameters(state, trial)
            if ablation == "edit_only":
                admissible = atomistic_edit_state_is_within_discovery_support_1d(
                    prepared.model, trial_state
                )
            else:
                admissible = atomistic_edit_state_is_admissible_1d(
                    prepared.model, trial_state
                )
            if admissible:
                selected = trial
                break
        if selected is None:
            # A rejected constrained step does not carry Adam momentum into
            # later attempts; this is an explicit optimizer reset.
            optimizer_state = optimizer.init(parameters)
            continue
        parameters = selected
        optimizer_state = next_optimizer_state
        accepted_updates += 1
    result = _state_with_parameters(state, parameters)
    validate_atomistic_edit_state_1d(prepared.model, result)
    if ablation == "level1_physical" and not atomistic_edit_state_is_admissible_1d(
        prepared.model, result
    ):
        raise RuntimeError(
            "projected Level-1 refinement returned an inadmissible state"
        )
    return result, accepted_updates


def _reanchor_state(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
) -> AtomisticEditState1D:
    anchors = np.asarray(state.extra_anchor_indices, dtype=np.int32).copy()
    offsets = np.asarray(state.extra_position_offsets_A, dtype=float).copy()
    active = np.asarray(state.extra_active, dtype=bool)
    positions = np.asarray(atomistic_edit_addition_positions_1d(model, state))
    s_A = np.asarray(model.axial_coordinates_A, dtype=float)
    u_A = np.asarray(model.transverse_coordinates_A, dtype=float)
    ds = s_A[1] - s_A[0]
    du = u_A[1] - u_A[0]
    discovery = np.asarray(model.options.discovery_support.discovery_mask)
    changed = False
    for index in np.flatnonzero(active):
        candidate = np.asarray(
            [
                int(np.rint((positions[index, 0] - s_A[0]) / ds)),
                int(np.rint((positions[index, 1] - u_A[0]) / du)),
            ],
            dtype=np.int32,
        )
        if (
            np.any(candidate < 0)
            or candidate[0] >= discovery.shape[0]
            or candidate[1] >= discovery.shape[1]
            or not discovery[candidate[0], candidate[1]]
        ):
            continue
        candidate_offset = positions[index] - np.asarray(
            [s_A[candidate[0]], u_A[candidate[1]]]
        )
        if np.all(np.abs(candidate_offset) <= 0.5 * np.asarray([ds, du]) + 1e-12):
            changed |= not np.array_equal(candidate, anchors[index])
            anchors[index] = candidate
            offsets[index] = candidate_offset
    if not changed:
        return state
    result = replace(
        state,
        extra_anchor_indices=jnp.asarray(anchors),
        extra_position_offsets_A=jnp.asarray(
            offsets, dtype=jnp.asarray(state.extra_position_offsets_A).dtype
        ),
    )
    validate_atomistic_edit_state_1d(model, result)
    return result


def _prune_state(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
    threshold: float,
    *,
    ablation: AtomisticEditAblation1D,
) -> tuple[AtomisticEditState1D, int, int]:
    removal_active = np.asarray(state.host_removal_active, dtype=bool).copy()
    extra_active = np.asarray(state.extra_active, dtype=bool).copy()
    removal_pruned = removal_active & (
        np.asarray(state.host_removal_fractions, dtype=float) <= threshold
    )
    extra_pruned = extra_active & (
        np.asarray(state.extra_scattering_equivalents, dtype=float) <= threshold
    )
    removal_active[removal_pruned] = False
    extra_active[extra_pruned] = False
    result = replace(
        state,
        host_removal_active=jnp.asarray(removal_active),
        host_removal_fractions=jnp.where(
            jnp.asarray(removal_active), state.host_removal_fractions, 0.0
        ),
        extra_active=jnp.asarray(extra_active),
        extra_scattering_equivalents=jnp.where(
            jnp.asarray(extra_active), state.extra_scattering_equivalents, 0.0
        ),
        extra_position_offsets_A=jnp.where(
            jnp.asarray(extra_active)[:, None],
            state.extra_position_offsets_A,
            0.0,
        ),
    )
    if ablation == "level1_physical" and not atomistic_edit_state_is_admissible_1d(
        model, result
    ):
        return state, 0, 0
    validate_atomistic_edit_state_1d(model, result)
    return result, int(np.count_nonzero(removal_pruned)), int(
        np.count_nonzero(extra_pruned)
    )


def _merge_duplicate_additions(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
    resolution_A: float,
    *,
    ablation: AtomisticEditAblation1D,
) -> tuple[AtomisticEditState1D, int, bool]:
    """Merge only active additions inside the declared numerical resolution."""
    active_slots = np.flatnonzero(np.asarray(state.extra_active, dtype=bool))
    if active_slots.size < 2:
        return state, 0, False
    positions = np.asarray(atomistic_edit_addition_positions_1d(model, state))
    parent = np.arange(active_slots.size)

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for first in range(active_slots.size):
        for second in range(first + 1, active_slots.size):
            distance = np.linalg.norm(
                positions[active_slots[first]] - positions[active_slots[second]]
            )
            if distance <= resolution_A:
                first_root = root(first)
                second_root = root(second)
                parent[second_root] = first_root
    groups: dict[int, list[int]] = {}
    for local_index, slot in enumerate(active_slots):
        groups.setdefault(root(local_index), []).append(int(slot))
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    if not duplicate_groups:
        return state, 0, False

    anchors = np.asarray(state.extra_anchor_indices, dtype=np.int32).copy()
    offsets = np.asarray(state.extra_position_offsets_A, dtype=float).copy()
    masses = np.asarray(
        state.extra_scattering_equivalents, dtype=float
    ).copy()
    active = np.asarray(state.extra_active, dtype=bool).copy()
    s_A = np.asarray(model.axial_coordinates_A, dtype=float)
    u_A = np.asarray(model.transverse_coordinates_A, dtype=float)
    half_pixel = 0.5 * np.asarray(
        [
            model.addition_kernel.axial_sampling_A,
            model.addition_kernel.transverse_sampling_A,
        ]
    )
    discovery = np.asarray(model.options.discovery_support.discovery_mask)
    maximum_mass = float(model.options.max_scattering_equivalent_per_centre)
    merged = 0
    for group in duplicate_groups:
        group_mass = masses[group]
        total_mass = float(np.sum(group_mass))
        if total_mass > maximum_mass + 1e-12:
            return state, 0, True
        if total_mass > 0.0:
            centre = np.sum(
                group_mass[:, None] * positions[group], axis=0
            ) / total_mass
        else:
            centre = np.mean(positions[group], axis=0)
        anchor = np.asarray(
            [
                int(np.argmin(np.abs(s_A - centre[0]))),
                int(np.argmin(np.abs(u_A - centre[1]))),
            ],
            dtype=np.int32,
        )
        offset = centre - np.asarray([s_A[anchor[0]], u_A[anchor[1]]])
        if not discovery[anchor[0], anchor[1]] or np.any(
            np.abs(offset) > half_pixel + 1e-12
        ):
            return state, 0, True
        survivor = min(group)
        anchors[survivor] = anchor
        offsets[survivor] = offset
        masses[survivor] = total_mass
        for slot in group:
            if slot == survivor:
                continue
            active[slot] = False
            offsets[slot] = 0.0
            masses[slot] = 0.0
            merged += 1
    result = replace(
        state,
        extra_anchor_indices=jnp.asarray(anchors),
        extra_position_offsets_A=jnp.asarray(
            offsets, dtype=jnp.asarray(state.extra_position_offsets_A).dtype
        ),
        extra_scattering_equivalents=jnp.asarray(
            masses,
            dtype=jnp.asarray(state.extra_scattering_equivalents).dtype,
        ),
        extra_active=jnp.asarray(active),
    )
    validate_atomistic_edit_state_1d(model, result)
    if ablation == "level1_physical" and not atomistic_edit_state_is_admissible_1d(
        model, result
    ):
        return state, 0, True
    return result, merged, False


def _first_inactive(mask: Any) -> int | None:
    indices = np.flatnonzero(~np.asarray(mask, dtype=bool))
    return int(indices[0]) if indices.size else None


def _capacity_status(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
) -> str:
    saturated = []
    if np.count_nonzero(np.asarray(state.host_removal_active)) >= int(
        model.options.max_host_removals
    ):
        saturated.append("host_removals")
    if np.count_nonzero(np.asarray(state.extra_active)) >= int(
        model.options.max_extra_centres
    ):
        saturated.append("extra_centres")
    return (
        "saturated_resource_bound:" + "+".join(saturated)
        if saturated
        else "capacity_available"
    )


def _birth_from_proposal(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    scores: AtomisticEditProposalScores1D,
    options: AtomisticEditSolverOptions1D,
) -> tuple[AtomisticEditState1D, str, bool]:
    model = prepared.model
    kind = scores.best_kind
    if kind == "none" or scores.best_index is None:
        return state, "none", False
    removal_slot = _first_inactive(state.host_removal_active)
    extra_slot = _first_inactive(state.extra_active)
    needs_removal = kind in {"host_removal", "paired_replacement"}
    needs_extra = kind in {"addition", "paired_replacement"}
    if (needs_removal and removal_slot is None) or (
        needs_extra and extra_slot is None
    ):
        return state, f"capacity_exhausted:{kind}", True

    result = state
    if needs_removal:
        assert removal_slot is not None and isinstance(scores.best_index, int)
        indices = np.asarray(result.host_removal_indices, dtype=np.int32).copy()
        fractions = np.asarray(
            result.host_removal_fractions, dtype=float
        ).copy()
        active = np.asarray(result.host_removal_active, dtype=bool).copy()
        indices[removal_slot] = int(scores.best_index)
        fractions[removal_slot] = (
            1.0
            if kind == "paired_replacement"
            else options.birth_removal_fraction
        )
        active[removal_slot] = True
        result = replace(
            result,
            host_removal_indices=jnp.asarray(indices),
            host_removal_fractions=jnp.asarray(
                fractions,
                dtype=jnp.asarray(state.host_removal_fractions).dtype,
            ),
            host_removal_active=jnp.asarray(active),
        )
    if needs_extra:
        assert extra_slot is not None
        if kind == "paired_replacement":
            assert isinstance(scores.best_index, int)
            anchor = np.asarray(
                scores.paired_replacement_anchor_indices[scores.best_index],
                dtype=np.int32,
            )
        else:
            assert isinstance(scores.best_index, tuple)
            anchor = np.asarray(scores.best_index, dtype=np.int32)
        anchors = np.asarray(result.extra_anchor_indices, dtype=np.int32).copy()
        offsets = np.asarray(
            result.extra_position_offsets_A, dtype=float
        ).copy()
        masses = np.asarray(
            result.extra_scattering_equivalents, dtype=float
        ).copy()
        active = np.asarray(result.extra_active, dtype=bool).copy()
        anchors[extra_slot] = anchor
        offsets[extra_slot] = 0.0
        masses[extra_slot] = (
            float(
                scores.paired_replacement_scattering_equivalent[
                    int(scores.best_index)
                ]
            )
            if kind == "paired_replacement"
            else options.birth_scattering_equivalent
        )
        active[extra_slot] = True
        result = replace(
            result,
            extra_anchor_indices=jnp.asarray(anchors),
            extra_position_offsets_A=jnp.asarray(
                offsets,
                dtype=jnp.asarray(state.extra_position_offsets_A).dtype,
            ),
            extra_scattering_equivalents=jnp.asarray(
                masses,
                dtype=jnp.asarray(state.extra_scattering_equivalents).dtype,
            ),
            extra_active=jnp.asarray(active),
        )
    validate_atomistic_edit_state_1d(model, result)
    if (
        options.ablation == "level1_physical"
        and not atomistic_edit_state_is_admissible_1d(model, result)
    ):
        raise RuntimeError(
            "the selected proposal disagrees with the Level-1 admissibility screen"
        )
    return result, kind, False


def _kkt_certificate(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    edit_penalty: float,
    options: AtomisticEditSolverOptions1D,
    scores: AtomisticEditProposalScores1D | None = None,
) -> AtomisticEditGridKKTCertificate1D:
    if scores is None:
        scores = atomistic_edit_proposal_scores_1d(
            prepared,
            state,
            edit_penalty,
            ablation=options.ablation,
            training_scan_batch_size=options.training_scan_batch_size,
        )
    _, gradients = _objective_value_and_gradients(
        prepared,
        state,
        edit_penalty,
        options.ablation,
        training_scan_batch_size=options.training_scan_batch_size,
    )
    active_norm = _projected_gradient_norm(prepared.model, state, gradients)
    maximum_addition = _finite_maximum(scores.addition_violation_grid)
    maximum_removal = _finite_maximum(scores.host_removal_violation)
    maximum_pair = _finite_maximum(scores.paired_replacement_violation)
    maximum = max(maximum_addition, maximum_removal, maximum_pair)
    proposal_ok = maximum <= options.proposal_grid_kkt_tolerance
    active_ok = active_norm <= options.active_projected_gradient_tolerance
    return AtomisticEditGridKKTCertificate1D(
        edit_penalty=edit_penalty,
        maximum_addition_violation=maximum_addition,
        maximum_host_removal_violation=maximum_removal,
        maximum_paired_replacement_violation=maximum_pair,
        maximum_dormant_violation=maximum,
        active_projected_gradient_norm=active_norm,
        proposal_tolerance=options.proposal_grid_kkt_tolerance,
        active_gradient_tolerance=options.active_projected_gradient_tolerance,
        proposal_grid_satisfied=proposal_ok,
        active_projected_gradient_satisfied=active_ok,
        satisfied=bool(proposal_ok and active_ok),
    )


class _NullProgress:
    def set_description(self, *_: Any, **__: Any) -> None:
        return None

    def set_postfix(self, *_: Any, **__: Any) -> None:
        return None

    def update(self, *_: Any, **__: Any) -> None:
        return None

    def close(self) -> None:
        return None


def _make_progress(enabled: bool, total: int) -> Any:
    if not enabled:
        return _NullProgress()
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _NullProgress()
    return tqdm(total=total, unit="active-set step", dynamic_ncols=True)


def _progress_state_snapshot(state: AtomisticEditState1D) -> AtomisticEditState1D:
    """Detach one immutable state for a user callback."""
    return replace(
        state,
        **{
            name: jnp.array(getattr(state, name), copy=True)
            for name in (
                "host_removal_indices",
                "host_removal_fractions",
                "host_removal_active",
                "extra_anchor_indices",
                "extra_position_offsets_A",
                "extra_scattering_equivalents",
                "extra_active",
                "host_displacement_controls",
            )
        },
    )


def _emit_progress_event(
    callback: AtomisticEditProgressCallback1D | None,
    *,
    phase: AtomisticEditProgressPhase1D,
    path_index: int,
    active_set_iteration: int,
    edit_penalty: float,
    state: AtomisticEditState1D,
    detail: str = "",
) -> None:
    if callback is None:
        return
    callback(
        AtomisticEditProgressEvent1D(
            phase=phase,
            path_index=path_index,
            active_set_iteration=active_set_iteration,
            edit_penalty=float(edit_penalty),
            state=_progress_state_snapshot(state),
            detail=detail,
        )
    )


def _solve_path_point(
    prepared: PreparedAtomisticEditReconstruction1D,
    initial_state: AtomisticEditState1D,
    edit_penalty: float,
    options: AtomisticEditSolverOptions1D,
    progress: Any | None = None,
    *,
    path_index: int,
    progress_callback: AtomisticEditProgressCallback1D | None = None,
) -> AtomisticEditLambdaPathPoint1D:
    state = initial_state
    births: list[str] = []
    optimizer_resets = 0
    pruned_removals = 0
    pruned_extras = 0
    merged_extras = 0
    duplicate_status = "resolved_or_absent"
    capacity_status = "capacity_available"
    stop_reason = "maximum_active_set_iterations"
    certificate: AtomisticEditGridKKTCertificate1D | None = None
    completed_iterations = 0

    for iteration in range(options.maximum_active_set_iterations):
        completed_iterations = iteration + 1
        if progress is not None:
            progress.set_description(
                f"AE-2 lambda={edit_penalty:.3g} iteration={iteration + 1}"
            )
            progress.update(1)
        state, _ = _refine_state(
            prepared,
            state,
            edit_penalty,
            ablation=options.ablation,
            updates=options.joint_refinement_updates,
            learning_rate=options.learning_rate,
            gradient_clip=options.gradient_clip,
            maximum_backtracking_steps=options.maximum_backtracking_steps,
            training_scan_batch_size=options.training_scan_batch_size,
        )
        if options.joint_refinement_updates:
            optimizer_resets += 1
        state = _reanchor_state(prepared.model, state)
        state, removed_count, extra_count = _prune_state(
            prepared.model,
            state,
            options.pruning_threshold,
            ablation=options.ablation,
        )
        pruned_removals += removed_count
        pruned_extras += extra_count
        state, merge_count, unresolved_duplicates = _merge_duplicate_additions(
            prepared.model,
            state,
            options.duplicate_merge_resolution_A,
            ablation=options.ablation,
        )
        merged_extras += merge_count
        if unresolved_duplicates:
            duplicate_status = "unresolved_duplicate_merge_fail_closed"
            stop_reason = duplicate_status
            if progress is not None:
                progress.set_postfix(event="duplicate-fail-closed")
            break

        _emit_progress_event(
            progress_callback,
            phase="refinement",
            path_index=path_index,
            active_set_iteration=iteration + 1,
            edit_penalty=edit_penalty,
            state=state,
            detail="joint_refinement_prune_merge",
        )

        scores = atomistic_edit_proposal_scores_1d(
            prepared,
            state,
            edit_penalty,
            ablation=options.ablation,
            training_scan_batch_size=options.training_scan_batch_size,
        )
        if scores.best_violation > options.proposal_grid_kkt_tolerance:
            state, event, exhausted = _birth_from_proposal(
                prepared, state, scores, options
            )
            births.append(event)
            _emit_progress_event(
                progress_callback,
                phase="birth",
                path_index=path_index,
                active_set_iteration=iteration + 1,
                edit_penalty=edit_penalty,
                state=state,
                detail=event,
            )
            if progress is not None:
                progress.set_postfix(
                    event=event, violation=f"{scores.best_violation:.3g}"
                )
            if exhausted:
                capacity_status = "exhausted_with_violating_direction"
                stop_reason = "capacity_exhausted_before_grid_kkt"
                certificate = _kkt_certificate(
                    prepared, state, edit_penalty, options, scores
                )
                break
            # The next refinement constructs a fresh optimizer after this
            # discrete active-set change.
            continue

        state, _ = _refine_state(
            prepared,
            state,
            edit_penalty,
            ablation=options.ablation,
            updates=options.polish_updates,
            learning_rate=options.polish_learning_rate,
            gradient_clip=options.gradient_clip,
            maximum_backtracking_steps=options.maximum_backtracking_steps,
            training_scan_batch_size=options.training_scan_batch_size,
        )
        if options.polish_updates:
            optimizer_resets += 1
        state = _reanchor_state(prepared.model, state)
        state, removed_count, extra_count = _prune_state(
            prepared.model,
            state,
            options.pruning_threshold,
            ablation=options.ablation,
        )
        pruned_removals += removed_count
        pruned_extras += extra_count
        state, merge_count, unresolved_duplicates = _merge_duplicate_additions(
            prepared.model,
            state,
            options.duplicate_merge_resolution_A,
            ablation=options.ablation,
        )
        merged_extras += merge_count
        if unresolved_duplicates:
            duplicate_status = "unresolved_duplicate_merge_fail_closed"
            stop_reason = duplicate_status
            if progress is not None:
                progress.set_postfix(event="duplicate-fail-closed")
            break
        _emit_progress_event(
            progress_callback,
            phase="polish",
            path_index=path_index,
            active_set_iteration=iteration + 1,
            edit_penalty=edit_penalty,
            state=state,
            detail="projected_polish_prune_merge",
        )
        rescored = atomistic_edit_proposal_scores_1d(
            prepared,
            state,
            edit_penalty,
            ablation=options.ablation,
            training_scan_batch_size=options.training_scan_batch_size,
        )
        certificate = _kkt_certificate(
            prepared, state, edit_penalty, options, rescored
        )
        if progress is not None:
            progress.set_postfix(
                event="grid-kkt",
                dormant=f"{certificate.maximum_dormant_violation:.3g}",
                active=f"{certificate.active_projected_gradient_norm:.3g}",
            )
        if certificate.satisfied:
            capacity_status = _capacity_status(prepared.model, state)
            stop_reason = (
                "proposal_grid_kkt_and_projected_polish_satisfied"
                if capacity_status == "capacity_available"
                else "capacity_saturated_after_grid_kkt"
            )
            break
        if (
            rescored.best_violation > options.proposal_grid_kkt_tolerance
        ):
            state, event, exhausted = _birth_from_proposal(
                prepared, state, rescored, options
            )
            births.append(event)
            _emit_progress_event(
                progress_callback,
                phase="birth",
                path_index=path_index,
                active_set_iteration=iteration + 1,
                edit_penalty=edit_penalty,
                state=state,
                detail=event,
            )
            if exhausted:
                capacity_status = "exhausted_with_violating_direction"
                stop_reason = "capacity_exhausted_after_projected_polish"
                break
            continue
        stop_reason = "active_projected_gradient_not_converged"

    if certificate is None or stop_reason == "maximum_active_set_iterations":
        scores = atomistic_edit_proposal_scores_1d(
            prepared,
            state,
            edit_penalty,
            ablation=options.ablation,
            training_scan_batch_size=options.training_scan_batch_size,
        )
        certificate = _kkt_certificate(
            prepared, state, edit_penalty, options, scores
        )
    if capacity_status == "capacity_available":
        capacity_status = _capacity_status(prepared.model, state)
    converged = bool(
        capacity_status == "capacity_available"
        and duplicate_status == "resolved_or_absent"
        and certificate.satisfied
    )
    training = _objective_components_value(
        prepared,
        state,
        edit_penalty,
        scan_indices=prepared.training_indices,
        ablation=options.ablation,
        scan_batch_size=options.training_scan_batch_size,
    )
    validation = _count_loss_value(
        prepared,
        state,
        prepared.validation_indices,
        options.training_scan_batch_size,
    )
    _emit_progress_event(
        progress_callback,
        phase="lambda_complete",
        path_index=path_index,
        active_set_iteration=completed_iterations,
        edit_penalty=edit_penalty,
        state=state,
        detail=stop_reason,
    )
    return AtomisticEditLambdaPathPoint1D(
        edit_penalty=edit_penalty,
        state=state,
        training_objective=training,
        validation_count_deviance=validation,
        kkt=certificate,
        active_set_iterations=completed_iterations,
        optimizer_reset_count=optimizer_resets,
        births=tuple(births),
        pruned_host_removals=pruned_removals,
        pruned_extra_centres=pruned_extras,
        merged_extra_centres=merged_extras,
        duplicate_status=duplicate_status,
        capacity_status=capacity_status,
        stop_reason=stop_reason,
        converged=converged,
    )


def _validation_selected_index(
    path_points: Sequence[AtomisticEditLambdaPathPoint1D],
    options: AtomisticEditSolverOptions1D,
) -> int:
    if not path_points:
        raise ValueError("the regularization path produced no points")
    validation = np.asarray(
        [point.validation_count_deviance for point in path_points], dtype=float
    )
    if np.any(~np.isfinite(validation)):
        raise FloatingPointError("validation count deviance is non-finite")
    best = float(np.min(validation))
    tolerance = options.validation_absolute_tolerance + (
        options.validation_relative_tolerance * max(abs(best), 1e-15)
    )
    eligible = np.flatnonzero(validation <= best + tolerance)
    # The path is strictly decreasing, so the first eligible index is the
    # largest regularization strength satisfying the frozen rule.
    return int(eligible[0])


def _validate_truth_free_initial_state(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
) -> None:
    validate_atomistic_edit_state_1d(model, state)
    empty = empty_atomistic_edit_state_1d(model)
    for name in (
        "host_removal_indices",
        "host_removal_fractions",
        "host_removal_active",
        "extra_anchor_indices",
        "extra_position_offsets_A",
        "extra_scattering_equivalents",
        "extra_active",
    ):
        if not np.array_equal(
            np.asarray(getattr(state, name)), np.asarray(getattr(empty, name))
        ):
            raise ValueError(
                "initial_state must contain empty edits; only bounded host "
                "displacement controls may be initialized"
            )


def run_prepared_atomistic_edit_reconstruction_1d(
    prepared: PreparedAtomisticEditReconstruction1D,
    *,
    initial_state: AtomisticEditState1D | None = None,
    options: AtomisticEditSolverOptions1D | None = None,
    show_progress: bool = False,
    evaluate_audit: bool = True,
    progress_callback: AtomisticEditProgressCallback1D | None = None,
) -> AtomisticEditReconstruction1D:
    """Run homotopy selection/debias and optionally emit structural events."""
    if not isinstance(prepared, PreparedAtomisticEditReconstruction1D):
        raise TypeError("prepared must be PreparedAtomisticEditReconstruction1D")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress must be Boolean")
    if not isinstance(evaluate_audit, (bool, np.bool_)):
        raise TypeError("evaluate_audit must be Boolean")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    options = _validated_solver_options(options)
    if (
        options.birth_scattering_equivalent
        > prepared.model.options.max_scattering_equivalent_per_centre
    ):
        raise ValueError(
            "birth_scattering_equivalent exceeds the model's one-centre bound"
        )
    state = (
        empty_atomistic_edit_state_1d(prepared.model)
        if initial_state is None
        else initial_state
    )
    _validate_truth_free_initial_state(prepared.model, state)
    if (
        options.ablation == "level1_physical"
        and not atomistic_edit_state_is_admissible_1d(prepared.model, state)
    ):
        raise ValueError("initial_state is not Level-1 physically admissible")
    path = tuple(float(value) for value in prepared.model.options.edit_penalty_path)
    if any(left <= right for left, right in zip(path, path[1:])):
        raise ValueError("the model edit_penalty_path is not strictly decreasing")

    _emit_progress_event(
        progress_callback,
        phase="initial",
        path_index=-1,
        active_set_iteration=0,
        edit_penalty=path[0],
        state=state,
        detail="empty_edit_initialization",
    )

    points: list[AtomisticEditLambdaPathPoint1D] = []
    warm_state = state
    path_complete = True
    progress = _make_progress(
        bool(show_progress),
        len(path) * options.maximum_active_set_iterations,
    )
    try:
        for path_index, penalty in enumerate(path):
            point = _solve_path_point(
                prepared,
                warm_state,
                penalty,
                options,
                progress,
                path_index=path_index,
                progress_callback=progress_callback,
            )
            points.append(point)
            warm_state = point.state
            if not point.converged:
                path_complete = False
                break
    finally:
        progress.close()
    selected_index = _validation_selected_index(points, options)
    selected = points[selected_index]
    penalized_state = selected.state

    debiased_state, _ = _refine_state(
        prepared,
        penalized_state,
        0.0,
        ablation=options.ablation,
        updates=options.debias_updates,
        learning_rate=options.debias_learning_rate,
        gradient_clip=options.gradient_clip,
        maximum_backtracking_steps=options.maximum_backtracking_steps,
        freeze_positions=True,
        training_scan_batch_size=options.training_scan_batch_size,
    )
    _emit_progress_event(
        progress_callback,
        phase="debias",
        path_index=selected_index,
        active_set_iteration=selected.active_set_iterations,
        edit_penalty=0.0,
        state=debiased_state,
        detail="support_and_position_fixed",
    )
    _, debias_gradients = _objective_value_and_gradients(
        prepared,
        debiased_state,
        0.0,
        options.ablation,
        training_scan_batch_size=options.training_scan_batch_size,
    )
    debias_projected_gradient_norm = _projected_gradient_norm(
        prepared.model,
        debiased_state,
        debias_gradients,
        freeze_positions=True,
    )
    debias_converged = bool(
        debias_projected_gradient_norm
        <= options.debias_projected_gradient_tolerance
    )
    # No pruning or re-anchoring is allowed after support/position selection.
    for name in (
        "host_removal_active",
        "extra_active",
        "extra_anchor_indices",
        "extra_position_offsets_A",
    ):
        if not np.array_equal(
            np.asarray(getattr(debiased_state, name)),
            np.asarray(getattr(penalized_state, name)),
        ):
            raise RuntimeError(
                "support/position-fixed debiasing changed " + name
            )

    debiased_training = _objective_components_value(
        prepared,
        debiased_state,
        0.0,
        scan_indices=prepared.training_indices,
        ablation=options.ablation,
        scan_batch_size=options.training_scan_batch_size,
    )
    debiased_validation = _count_loss_value(
        prepared,
        debiased_state,
        prepared.validation_indices,
        options.training_scan_batch_size,
    )
    audit_was_evaluated = bool(
        evaluate_audit and np.asarray(prepared.audit_indices).size
    )
    audit = (
        _count_loss_value(
            prepared,
            debiased_state,
            prepared.audit_indices,
            options.training_scan_batch_size,
        )
        if audit_was_evaluated
        else None
    )
    capacity_exhausted = any(
        point.capacity_status != "capacity_available" for point in points
    )
    converged = bool(
        path_complete
        and len(points) == len(path)
        and all(point.converged for point in points)
        and not capacity_exhausted
        and debias_converged
    )
    if capacity_exhausted:
        stop_reason = "capacity_bound_fail_closed"
    elif any(
        point.duplicate_status != "resolved_or_absent" for point in points
    ):
        stop_reason = "duplicate_merge_fail_closed"
    elif not path_complete:
        stop_reason = "regularization_path_incomplete"
    elif not debias_converged:
        stop_reason = "debias_projected_gradient_not_converged"
    else:
        stop_reason = "frozen_path_solved_and_validation_selected"
    training_scan_count = int(np.asarray(prepared.training_indices).size)
    scan_batching_active = bool(
        options.training_scan_batch_size is not None
        and options.training_scan_batch_size < training_scan_count
    )
    metadata = MappingProxyType(
        {
            "schema": "atomistic_edit_reconstruction_1d:v1",
            "ablation": options.ablation,
            "seed": int(options.seed),
            "seed_used_for_single_start": False,
            "truth_inputs_used": False,
            "nuisance_image_used": False,
            "energy_envelope_used": False,
            "penalty_path": list(path),
            "penalty_path_strictly_decreasing": True,
            "selection_rule": _SELECTION_RULE,
            "selection_uses_validation_only": True,
            "audit_used_for_selection": False,
            "audit_evaluated": audit_was_evaluated,
            "debias_rule": _DEBIAS_RULE,
            "debias_projected_gradient_certified": debias_converged,
            "duplicate_merge_resolution_A": options.duplicate_merge_resolution_A,
            "duplicate_merge_enabled": True,
            "proposal_certificate_scope": _PROPOSAL_CERTIFICATE,
            "continuous_birth_kkt_evaluated": False,
            "optimizer_reset_at_every_refinement": True,
            "progress_requested": bool(show_progress),
            "full_training_proposal_scores": True,
            "full_training_projected_polish": True,
            "training_scan_batch_size": options.training_scan_batch_size,
            "effective_training_scan_batch_size": (
                options.training_scan_batch_size
                if scan_batching_active
                else training_scan_count
            ),
            "training_gradient_accumulation": (
                "deterministic_exact_scan_batch_sum"
                if scan_batching_active
                else "single_full_training_graph"
            ),
            "scan_batch_normalization": "global_valid_detector_pixel_count",
            "active_parameter_count_formula": "P_deformation + K_minus + 3*K_plus",
        }
    )
    return AtomisticEditReconstruction1D(
        prepared_problem_id=prepared.reconstruction_problem_id,
        reconstructor_id=prepared.reconstructor_id,
        penalized_state=penalized_state,
        debiased_state=debiased_state,
        selected_edit_penalty=selected.edit_penalty,
        selected_path_index=selected_index,
        path_points=tuple(points),
        penalized_training_objective=selected.training_objective,
        debiased_training_objective=debiased_training,
        penalized_validation_count_deviance=selected.validation_count_deviance,
        debiased_validation_count_deviance=debiased_validation,
        debiased_audit_count_deviance=audit,
        selected_kkt=selected.kkt,
        debias_projected_gradient_norm=debias_projected_gradient_norm,
        debias_projected_gradient_tolerance=(
            options.debias_projected_gradient_tolerance
        ),
        debias_converged=debias_converged,
        active_parameter_count=atomistic_edit_active_parameter_count_1d(
            prepared.model, debiased_state
        ),
        capacity_exhausted=capacity_exhausted,
        converged=converged,
        stop_reason=stop_reason,
        metadata=metadata,
    )


def _states_agree_within_declared_resolution(
    model: AtomisticEditModel1D,
    first: AtomisticEditState1D,
    second: AtomisticEditState1D,
    *,
    position_tolerance_A: float,
    amplitude_tolerance: float,
) -> bool:
    first_removals = _dense_host_removals_numpy(model, first)
    second_removals = _dense_host_removals_numpy(model, second)
    if not np.allclose(
        first_removals,
        second_removals,
        rtol=0.0,
        atol=amplitude_tolerance,
    ):
        return False

    def additions(state: AtomisticEditState1D) -> tuple[np.ndarray, np.ndarray]:
        active = np.asarray(state.extra_active, dtype=bool)
        positions = np.asarray(
            atomistic_edit_addition_positions_1d(model, state), dtype=float
        )[active]
        masses = np.asarray(
            state.extra_scattering_equivalents, dtype=float
        )[active]
        if positions.size:
            order = np.lexsort((positions[:, 1], positions[:, 0]))
            positions = positions[order]
            masses = masses[order]
        return positions, masses

    first_positions, first_masses = additions(first)
    second_positions, second_masses = additions(second)
    if first_positions.shape != second_positions.shape:
        return False
    if not np.allclose(
        first_positions,
        second_positions,
        rtol=0.0,
        atol=position_tolerance_A,
    ) or not np.allclose(
        first_masses,
        second_masses,
        rtol=0.0,
        atol=amplitude_tolerance,
    ):
        return False
    return bool(
        np.allclose(
            np.asarray(first.host_displacement_controls),
            np.asarray(second.host_displacement_controls),
            rtol=0.0,
            atol=position_tolerance_A,
        )
    )


def run_prepared_atomistic_edit_multistart_reconstruction_1d(
    prepared: PreparedAtomisticEditReconstruction1D,
    *,
    number_of_starts: int,
    initial_host_control_std_A: Any,
    options: AtomisticEditSolverOptions1D | None = None,
    show_progress: bool = False,
) -> AtomisticEditMultistartReconstruction1D:
    """Run deterministic empty-edit starts and select using validation only.

    The first start is the exact zero-control reference. Remaining starts use
    clipped zero-mean Gaussian host controls generated from the declared seed.
    No active edit, synthetic truth, audit count, or object descriptor enters
    initialization or selection.
    """
    if not isinstance(prepared, PreparedAtomisticEditReconstruction1D):
        raise TypeError("prepared must be PreparedAtomisticEditReconstruction1D")
    count = _index("number_of_starts", number_of_starts)
    standard_deviation = _finite(
        "initial_host_control_std_A",
        initial_host_control_std_A,
        nonnegative=True,
    )
    solver_options = _validated_solver_options(options)
    maximum_displacement = float(
        np.asarray(prepared.model.host_model.maximum_displacement)
    )
    if standard_deviation > maximum_displacement:
        raise ValueError(
            "initial_host_control_std_A must not exceed the host displacement bound"
        )
    seed_sequence = np.random.SeedSequence(solver_options.seed)
    child_sequences = seed_sequence.spawn(count)
    empty = empty_atomistic_edit_state_1d(prepared.model)
    control_shape = np.asarray(empty.host_displacement_controls).shape
    control_dtype = jnp.asarray(empty.host_displacement_controls).dtype
    candidates = []
    start_seeds = []
    initial_rms = []
    for start_index, child_sequence in enumerate(child_sequences):
        start_seed = int(
            child_sequence.generate_state(1, dtype=np.uint64)[0]
        )
        start_seeds.append(start_seed)
        if start_index == 0 or standard_deviation == 0.0:
            controls = np.zeros(control_shape, dtype=float)
        else:
            controls = np.random.default_rng(child_sequence).normal(
                loc=0.0,
                scale=standard_deviation,
                size=control_shape,
            )
            controls = np.clip(
                controls, -maximum_displacement, maximum_displacement
            )
        initial_rms.append(float(np.sqrt(np.mean(controls**2))))
        initial_state = replace(
            empty,
            host_displacement_controls=jnp.asarray(
                controls, dtype=control_dtype
            ),
        )
        candidates.append(
            run_prepared_atomistic_edit_reconstruction_1d(
                prepared,
                initial_state=initial_state,
                options=replace(solver_options, seed=start_seed),
                show_progress=show_progress,
                evaluate_audit=False,
            )
        )

    validation = np.asarray(
        [candidate.penalized_validation_count_deviance for candidate in candidates],
        dtype=float,
    )
    if np.any(~np.isfinite(validation)):
        raise FloatingPointError("a multistart validation deviance is non-finite")
    selected_index = int(np.argmin(validation))
    best = float(validation[selected_index])
    tolerance = solver_options.validation_absolute_tolerance + (
        solver_options.validation_relative_tolerance * max(abs(best), 1e-15)
    )
    eligible = tuple(
        int(index) for index in np.flatnonzero(validation <= best + tolerance)
    )
    selected_state = candidates[selected_index].debiased_state
    ambiguous = tuple(
        index
        for index in eligible
        if index != selected_index
        and not _states_agree_within_declared_resolution(
            prepared.model,
            selected_state,
            candidates[index].debiased_state,
            position_tolerance_A=solver_options.duplicate_merge_resolution_A,
            amplitude_tolerance=max(solver_options.pruning_threshold, 1e-12),
        )
    )
    selected_result = candidates[selected_index]
    selected_audit_was_evaluated = bool(
        np.asarray(prepared.audit_indices).size
    )
    selected_audit = (
        float(
            np.asarray(
                _count_loss(
                    prepared,
                    selected_result.debiased_state,
                    prepared.audit_indices,
                )
            )
        )
        if selected_audit_was_evaluated
        else None
    )
    selected_result = replace(
        selected_result,
        debiased_audit_count_deviance=selected_audit,
        metadata=MappingProxyType(
            {
                **dict(selected_result.metadata),
                "audit_evaluated": selected_audit_was_evaluated,
                "audit_evaluated_after_multistart_selection": (
                    selected_audit_was_evaluated
                ),
            }
        ),
    )
    metadata = MappingProxyType(
        {
            "schema": "atomistic_edit_multistart_reconstruction_1d:v1",
            "seed": int(solver_options.seed),
            "deterministic_seed_used": True,
            "number_of_starts": count,
            "first_start_is_zero_control_reference": True,
            "remaining_starts_are_seeded_host_controls": True,
            "initial_edits_are_empty": True,
            "truth_inputs_used": False,
            "selection_uses_validation_only": True,
            "audit_used_for_selection": False,
            "candidate_audits_evaluated": False,
            "selected_audit_evaluated_after_selection": (
                selected_audit_was_evaluated
            ),
            "ambiguity_compares_debiased_states": True,
            "ambiguity_position_tolerance_A": (
                solver_options.duplicate_merge_resolution_A
            ),
        }
    )
    return AtomisticEditMultistartReconstruction1D(
        candidates=tuple(candidates),
        selected_result=selected_result,
        selected_start_index=selected_index,
        validation_eligible_start_indices=eligible,
        ambiguous_start_indices=ambiguous,
        structurally_ambiguous=bool(ambiguous),
        start_seeds=tuple(start_seeds),
        initial_host_control_rms_A=tuple(initial_rms),
        numerically_converged=bool(selected_result.converged),
        metadata=metadata,
    )
