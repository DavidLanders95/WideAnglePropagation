"""Authenticated replay archives for one completed AE-2 reconstruction.

The outer archive binds the calibrated acquisition, fixed forward problem,
regularization path, every path state and certificate, and the selected
penalized/debiased result.  The atomistic specimen model is carried by a
nested, independently authenticated AE-1 snapshot rather than by a second
model serialization implementation.

Loading is deliberately expensive: it reconstructs the public prepared
problem, rerenders the fitted specimens, and replays every reported objective,
validation/audit count loss, proposal-grid KKT certificate, projected-gradient
norm, active count, and solver status.  A valid SHA-256 alone is therefore not
enough to bless resealed but semantically inconsistent fields.

Only :class:`AtomisticEditReconstruction1D` (one start) is supported here.
The multistart aggregate remains a separate future archive contract; each of
its candidate single-start results can already be archived independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from importlib import metadata as importlib_metadata
import json
import operator
import os
from pathlib import Path
import platform
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .ptychography_1d import (
    PtychographyMeasurement1D,
    PtychographyObjective1D,
)
from .ptychography_atomistic_edit_1d import (
    AtomisticEditModel1D,
    AtomisticEditState1D,
    atomistic_edit_active_parameter_count_1d,
    empty_atomistic_edit_state_1d,
    load_atomistic_edit_snapshot_1d,
    make_atomistic_edit_snapshot_1d,
    render_atomistic_edit_potential_1d,
    save_atomistic_edit_snapshot_1d,
    validate_atomistic_edit_state_1d,
)
from .ptychography_atomistic_edit_solver_1d import (
    AtomisticEditGridKKTCertificate1D,
    AtomisticEditLambdaPathPoint1D,
    AtomisticEditObjectiveComponents1D,
    AtomisticEditReconstruction1D,
    AtomisticEditSolverOptions1D,
    PreparedAtomisticEditReconstruction1D,
    _objective_value_and_gradients,
    atomistic_edit_objective_components_1d,
    atomistic_edit_proposal_scores_1d,
    prepare_atomistic_edit_reconstruction_1d,
)


__all__ = [
    "AtomisticEditReconstructionBundle1D",
    "load_atomistic_edit_reconstruction_bundle_1d",
    "make_atomistic_edit_reconstruction_bundle_1d",
    "save_atomistic_edit_reconstruction_bundle_1d",
    "validate_atomistic_edit_reconstruction_bundle_1d",
]


_SCHEMA_VERSION = 1
_ARCHIVE_CONTRACT = "atomistic_edit_reconstruction_bundle_1d:v1"
_MODEL_ANCHOR_OBJECTIVE_ID = "ae2_model_anchor_no_data_claim:v1"
_CERTIFICATE_SCOPE = "full_training_proposal_grid_kkt:v1"
_SELECTION_RULE = "validation_largest_lambda_within_frozen_tolerance:v1"
_DEBIAS_RULE = "support_and_position_fixed_no_edit_penalty:v1"

_STATE_FIELDS = (
    "host_removal_indices",
    "host_removal_fractions",
    "host_removal_active",
    "extra_anchor_indices",
    "extra_position_offsets_A",
    "extra_scattering_equivalents",
    "extra_active",
    "host_displacement_controls",
)

_PREPARED_JSON_FIELDS = {
    "window_length",
    "slice_thickness_A",
    "energy_eV",
    "measurement_calibration_id",
    "measurement_metadata",
    "objective_kind",
    "objective_minimum_expected_electrons",
    "objective_relative_signal_scale",
    "reconstruction_problem_id",
    "reconstructor_id",
    "model_id",
    "metadata",
}

_OBJECTIVE_JSON_FIELDS = {
    "count_deviance",
    "edit_mass",
    "weighted_edit_penalty",
    "elastic_penalty",
    "hard_core_penalty",
    "total_objective",
    "edit_penalty",
    "ablation",
}

_KKT_JSON_FIELDS = {
    "edit_penalty",
    "maximum_addition_violation",
    "maximum_host_removal_violation",
    "maximum_paired_replacement_violation",
    "maximum_dormant_violation",
    "active_projected_gradient_norm",
    "proposal_tolerance",
    "active_gradient_tolerance",
    "proposal_grid_satisfied",
    "active_projected_gradient_satisfied",
    "satisfied",
    "continuous_birth_kkt_evaluated",
    "certificate_scope",
}

_PATH_POINT_JSON_FIELDS = {
    "edit_penalty",
    "training_objective",
    "validation_count_deviance",
    "kkt",
    "active_set_iterations",
    "optimizer_reset_count",
    "births",
    "pruned_host_removals",
    "pruned_extra_centres",
    "merged_extra_centres",
    "duplicate_status",
    "capacity_status",
    "stop_reason",
    "converged",
}

_RESULT_JSON_FIELDS = {
    "prepared_problem_id",
    "reconstructor_id",
    "selected_edit_penalty",
    "selected_path_index",
    "path_points",
    "penalized_training_objective",
    "debiased_training_objective",
    "penalized_validation_count_deviance",
    "debiased_validation_count_deviance",
    "debiased_audit_count_deviance",
    "selected_kkt",
    "debias_projected_gradient_norm",
    "debias_projected_gradient_tolerance",
    "debias_converged",
    "active_parameter_count",
    "capacity_exhausted",
    "converged",
    "stop_reason",
    "metadata",
}

_SOLVER_OPTIONS_FIELDS = set(AtomisticEditSolverOptions1D.__dataclass_fields__)

_RESULT_METADATA_FIELDS = {
    "schema",
    "ablation",
    "seed",
    "seed_used_for_single_start",
    "truth_inputs_used",
    "nuisance_image_used",
    "energy_envelope_used",
    "penalty_path",
    "penalty_path_strictly_decreasing",
    "selection_rule",
    "selection_uses_validation_only",
    "audit_used_for_selection",
    "audit_evaluated",
    "debias_rule",
    "debias_projected_gradient_certified",
    "duplicate_merge_resolution_A",
    "duplicate_merge_enabled",
    "proposal_certificate_scope",
    "continuous_birth_kkt_evaluated",
    "optimizer_reset_at_every_refinement",
    "progress_requested",
    "full_training_proposal_scores",
    "full_training_projected_polish",
    "training_scan_batch_size",
    "effective_training_scan_batch_size",
    "training_gradient_accumulation",
    "scan_batch_normalization",
    "active_parameter_count_formula",
}

_PROVENANCE_FIELDS = {
    "schema",
    "python_version",
    "numpy_version",
    "jax_version",
    "package_version",
    "platform",
    "jax_default_backend",
    "jax_enable_x64",
    "devices",
    "array_dtypes",
    "caller_metadata",
}

_ARCHIVE_FIELDS = {
    "schema_version",
    "archive_contract",
    "ae1_model_snapshot_npz",
    "prepared_probe_rows",
    "prepared_window_starts",
    "prepared_propagation_kernel",
    "prepared_measurement_signal",
    "prepared_measurement_total",
    "prepared_measurement_valid",
    "prepared_measurement_dark",
    "prepared_measurement_read_noise",
    "prepared_objective_dose",
    "prepared_training_indices",
    "prepared_validation_indices",
    "prepared_audit_indices",
    "prepared_excluded_indices",
    "prepared_json",
    "solver_options_json",
    "result_json",
    "path_training_scan_indices",
    "penalized_training_scan_indices",
    "debiased_training_scan_indices",
    "provenance_json",
}
for _prefix in ("penalized_state", "debiased_state"):
    _ARCHIVE_FIELDS.update(f"{_prefix}_{name}" for name in _STATE_FIELDS)
_ARCHIVE_FIELDS.update(f"path_state_{name}" for name in _STATE_FIELDS)


@dataclass(frozen=True, eq=False)
class AtomisticEditReconstructionBundle1D:
    """Independently replayable prepared problem and one AE-2 result."""

    prepared: PreparedAtomisticEditReconstruction1D
    reconstruction: AtomisticEditReconstruction1D
    solver_options: AtomisticEditSolverOptions1D
    provenance: Mapping[str, Any] = field(default_factory=dict)
    archive_id: str = ""


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            dict(value),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise TypeError("archive metadata must be finite and JSON serializable") from error


def _json_mapping(
    value: Any,
    *,
    name: str,
    expected_fields: set[str] | None = None,
) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind != "U":
            raise ValueError(f"{name} must be scalar Unicode JSON")
        value = str(array.item())
    if not isinstance(value, str):
        raise TypeError(f"{name} must be JSON text")

    def reject_constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON constant {token}")

    try:
        decoded = json.loads(value, parse_constant=reject_constant)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"{name} is invalid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must decode to an object")
    if expected_fields is not None and set(decoded) != expected_fields:
        raise ValueError(
            f"{name} fields differ from schema: "
            f"missing={sorted(expected_fields - set(decoded))}, "
            f"extra={sorted(set(decoded) - expected_fields)}"
        )
    return decoded


def _boolean(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be Boolean")
    return value


def _integer(name: str, value: Any, *, nonnegative: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(result)


def _number(
    name: str,
    value: Any,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number")
    array = np.asarray(value)
    if array.shape != () or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _metadata(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    # Round-trip now so tuples and NumPy objects cannot hide in memory-only data.
    decoded = _json_mapping(_canonical_json(value), name=name)
    return MappingProxyType(decoded)


def _state_arrays(prefix: str, state: AtomisticEditState1D) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_{name}": np.asarray(getattr(state, name))
        for name in _STATE_FIELDS
    }


def _stacked_state_arrays(
    states: Sequence[AtomisticEditState1D],
) -> dict[str, np.ndarray]:
    if not states:
        raise ValueError("an AE-2 archive requires at least one path state")
    return {
        f"path_state_{name}": np.stack(
            [np.asarray(getattr(state, name)) for state in states], axis=0
        )
        for name in _STATE_FIELDS
    }


def _state_from_payload(
    payload: Mapping[str, np.ndarray],
    prefix: str,
    template: AtomisticEditState1D,
    *,
    path_index: int | None = None,
    path_length: int | None = None,
) -> AtomisticEditState1D:
    values: dict[str, Any] = {}
    for name in _STATE_FIELDS:
        key = f"{prefix}_{name}"
        array = np.asarray(payload[key])
        expected = np.asarray(getattr(template, name))
        if path_index is None:
            expected_shape = expected.shape
        else:
            if path_length is None:
                raise RuntimeError("internal path length is missing")
            expected_shape = (path_length, *expected.shape)
        if array.shape != expected_shape or array.dtype != expected.dtype:
            raise ValueError(
                f"archive field {key} must have shape {expected_shape} "
                f"and dtype {expected.dtype}"
            )
        selected = array if path_index is None else array[path_index]
        values[name] = jnp.asarray(selected)
    state = AtomisticEditState1D(**values)
    return state


def _objective_json(components: AtomisticEditObjectiveComponents1D) -> dict[str, Any]:
    return {
        "count_deviance": float(np.asarray(components.count_deviance)),
        "edit_mass": float(np.asarray(components.edit_mass)),
        "weighted_edit_penalty": float(
            np.asarray(components.weighted_edit_penalty)
        ),
        "elastic_penalty": float(np.asarray(components.elastic_penalty)),
        "hard_core_penalty": float(np.asarray(components.hard_core_penalty)),
        "total_objective": float(np.asarray(components.total_objective)),
        "edit_penalty": float(components.edit_penalty),
        "ablation": components.ablation,
    }


def _objective_from_json(
    fields: Any,
    scan_indices: Any,
    *,
    name: str,
) -> AtomisticEditObjectiveComponents1D:
    if not isinstance(fields, dict) or set(fields) != _OBJECTIVE_JSON_FIELDS:
        raise ValueError(f"{name} has the wrong objective schema")
    ablation = fields["ablation"]
    if ablation not in {"edit_only", "level1_physical"}:
        raise ValueError(f"{name}.ablation is unsupported")
    return AtomisticEditObjectiveComponents1D(
        count_deviance=_number(f"{name}.count_deviance", fields["count_deviance"]),
        edit_mass=_number(f"{name}.edit_mass", fields["edit_mass"], nonnegative=True),
        weighted_edit_penalty=_number(
            f"{name}.weighted_edit_penalty",
            fields["weighted_edit_penalty"],
            nonnegative=True,
        ),
        elastic_penalty=_number(
            f"{name}.elastic_penalty", fields["elastic_penalty"], nonnegative=True
        ),
        hard_core_penalty=_number(
            f"{name}.hard_core_penalty",
            fields["hard_core_penalty"],
            nonnegative=True,
        ),
        total_objective=_number(
            f"{name}.total_objective", fields["total_objective"]
        ),
        edit_penalty=_number(
            f"{name}.edit_penalty", fields["edit_penalty"], nonnegative=True
        ),
        ablation=ablation,
        scan_indices=np.asarray(scan_indices, dtype=np.int32),
    )


def _kkt_json(certificate: AtomisticEditGridKKTCertificate1D) -> dict[str, Any]:
    def value(number: Any) -> float | str:
        result = float(number)
        if np.isnan(result) or np.isposinf(result):
            raise ValueError("KKT certificates cannot contain NaN or positive infinity")
        return "negative_infinity" if np.isneginf(result) else result

    return {
        "edit_penalty": float(certificate.edit_penalty),
        "maximum_addition_violation": value(
            certificate.maximum_addition_violation
        ),
        "maximum_host_removal_violation": value(
            certificate.maximum_host_removal_violation
        ),
        "maximum_paired_replacement_violation": value(
            certificate.maximum_paired_replacement_violation
        ),
        "maximum_dormant_violation": value(
            certificate.maximum_dormant_violation
        ),
        "active_projected_gradient_norm": float(
            certificate.active_projected_gradient_norm
        ),
        "proposal_tolerance": float(certificate.proposal_tolerance),
        "active_gradient_tolerance": float(
            certificate.active_gradient_tolerance
        ),
        "proposal_grid_satisfied": bool(certificate.proposal_grid_satisfied),
        "active_projected_gradient_satisfied": bool(
            certificate.active_projected_gradient_satisfied
        ),
        "satisfied": bool(certificate.satisfied),
        "continuous_birth_kkt_evaluated": bool(
            certificate.continuous_birth_kkt_evaluated
        ),
        "certificate_scope": certificate.certificate_scope,
    }


def _kkt_from_json(fields: Any, *, name: str) -> AtomisticEditGridKKTCertificate1D:
    if not isinstance(fields, dict) or set(fields) != _KKT_JSON_FIELDS:
        raise ValueError(f"{name} has the wrong KKT schema")
    def violation(field_name: str) -> float:
        value = fields[field_name]
        if value == "negative_infinity":
            return float("-inf")
        return _number(f"{name}.{field_name}", value)

    return AtomisticEditGridKKTCertificate1D(
        edit_penalty=_number(f"{name}.edit_penalty", fields["edit_penalty"], positive=True),
        maximum_addition_violation=violation("maximum_addition_violation"),
        maximum_host_removal_violation=violation(
            "maximum_host_removal_violation"
        ),
        maximum_paired_replacement_violation=violation(
            "maximum_paired_replacement_violation"
        ),
        maximum_dormant_violation=violation("maximum_dormant_violation"),
        active_projected_gradient_norm=_number(
            f"{name}.active_projected_gradient_norm",
            fields["active_projected_gradient_norm"],
            nonnegative=True,
        ),
        proposal_tolerance=_number(
            f"{name}.proposal_tolerance", fields["proposal_tolerance"], positive=True
        ),
        active_gradient_tolerance=_number(
            f"{name}.active_gradient_tolerance",
            fields["active_gradient_tolerance"],
            positive=True,
        ),
        proposal_grid_satisfied=_boolean(
            f"{name}.proposal_grid_satisfied", fields["proposal_grid_satisfied"]
        ),
        active_projected_gradient_satisfied=_boolean(
            f"{name}.active_projected_gradient_satisfied",
            fields["active_projected_gradient_satisfied"],
        ),
        satisfied=_boolean(f"{name}.satisfied", fields["satisfied"]),
        continuous_birth_kkt_evaluated=_boolean(
            f"{name}.continuous_birth_kkt_evaluated",
            fields["continuous_birth_kkt_evaluated"],
        ),
        certificate_scope=_identifier(
            f"{name}.certificate_scope", fields["certificate_scope"]
        ),
    )


def _path_point_json(point: AtomisticEditLambdaPathPoint1D) -> dict[str, Any]:
    return {
        "edit_penalty": float(point.edit_penalty),
        "training_objective": _objective_json(point.training_objective),
        "validation_count_deviance": float(point.validation_count_deviance),
        "kkt": _kkt_json(point.kkt),
        "active_set_iterations": int(point.active_set_iterations),
        "optimizer_reset_count": int(point.optimizer_reset_count),
        "births": list(point.births),
        "pruned_host_removals": int(point.pruned_host_removals),
        "pruned_extra_centres": int(point.pruned_extra_centres),
        "merged_extra_centres": int(point.merged_extra_centres),
        "duplicate_status": point.duplicate_status,
        "capacity_status": point.capacity_status,
        "stop_reason": point.stop_reason,
        "converged": bool(point.converged),
    }


def _result_json(result: AtomisticEditReconstruction1D) -> str:
    return _canonical_json(
        {
            "prepared_problem_id": result.prepared_problem_id,
            "reconstructor_id": result.reconstructor_id,
            "selected_edit_penalty": float(result.selected_edit_penalty),
            "selected_path_index": int(result.selected_path_index),
            "path_points": [_path_point_json(point) for point in result.path_points],
            "penalized_training_objective": _objective_json(
                result.penalized_training_objective
            ),
            "debiased_training_objective": _objective_json(
                result.debiased_training_objective
            ),
            "penalized_validation_count_deviance": float(
                result.penalized_validation_count_deviance
            ),
            "debiased_validation_count_deviance": float(
                result.debiased_validation_count_deviance
            ),
            "debiased_audit_count_deviance": (
                None
                if result.debiased_audit_count_deviance is None
                else float(result.debiased_audit_count_deviance)
            ),
            "selected_kkt": _kkt_json(result.selected_kkt),
            "debias_projected_gradient_norm": float(
                result.debias_projected_gradient_norm
            ),
            "debias_projected_gradient_tolerance": float(
                result.debias_projected_gradient_tolerance
            ),
            "debias_converged": bool(result.debias_converged),
            "active_parameter_count": int(result.active_parameter_count),
            "capacity_exhausted": bool(result.capacity_exhausted),
            "converged": bool(result.converged),
            "stop_reason": result.stop_reason,
            "metadata": dict(result.metadata),
        }
    )


def _solver_options_json(options: AtomisticEditSolverOptions1D) -> str:
    return _canonical_json(
        {name: getattr(options, name) for name in sorted(_SOLVER_OPTIONS_FIELDS)}
    )


def _solver_options_from_json(value: Any) -> AtomisticEditSolverOptions1D:
    fields = _json_mapping(
        value,
        name="solver_options_json",
        expected_fields=_SOLVER_OPTIONS_FIELDS,
    )
    defaults = AtomisticEditSolverOptions1D()
    values: dict[str, Any] = {}
    integer_fields = {
        "maximum_active_set_iterations",
        "joint_refinement_updates",
        "polish_updates",
        "debias_updates",
        "maximum_backtracking_steps",
        "seed",
    }
    for name in sorted(_SOLVER_OPTIONS_FIELDS):
        if name == "ablation":
            value_field = fields[name]
            if value_field not in {"edit_only", "level1_physical"}:
                raise ValueError("solver_options_json.ablation is unsupported")
            values[name] = value_field
        elif name == "training_scan_batch_size":
            values[name] = (
                None
                if fields[name] is None
                else _integer(
                    "solver_options_json.training_scan_batch_size",
                    fields[name],
                    nonnegative=False,
                )
            )
        elif name in integer_fields:
            values[name] = _integer(
                f"solver_options_json.{name}", fields[name], nonnegative=True
            )
        else:
            values[name] = _number(
                f"solver_options_json.{name}", fields[name], nonnegative=True
            )
        # A schema check against the dataclass default also catches accidental
        # future Boolean/string fields that need an explicit archive policy.
        if isinstance(getattr(defaults, name), bool):
            raise RuntimeError(f"unhandled Boolean solver option {name}")
    options = AtomisticEditSolverOptions1D(**values)
    if options.maximum_active_set_iterations <= 0:
        raise ValueError("maximum_active_set_iterations must be positive")
    if (
        options.training_scan_batch_size is not None
        and options.training_scan_batch_size <= 0
    ):
        raise ValueError("training_scan_batch_size must be positive when set")
    positive_fields = {
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
    }
    if any(getattr(options, name) <= 0.0 for name in positive_fields):
        raise ValueError("positive solver options must remain positive")
    if options.birth_removal_fraction > 1.0 or options.seed >= 2**64:
        raise ValueError("solver options lie outside their public bounds")
    return options


def _prepared_json(prepared: PreparedAtomisticEditReconstruction1D) -> str:
    return _canonical_json(
        {
            "window_length": int(prepared.window_length),
            "slice_thickness_A": float(prepared.slice_thickness_A),
            "energy_eV": float(prepared.energy_eV),
            "measurement_calibration_id": prepared.measurement.calibration_id,
            "measurement_metadata": dict(prepared.measurement.metadata),
            "objective_kind": prepared.objective.kind,
            "objective_minimum_expected_electrons": float(
                prepared.objective.minimum_expected_electrons
            ),
            "objective_relative_signal_scale": float(
                prepared.objective.relative_signal_scale
            ),
            "reconstruction_problem_id": prepared.reconstruction_problem_id,
            "reconstructor_id": prepared.reconstructor_id,
            "model_id": prepared.model.model_id,
            "metadata": dict(prepared.metadata),
        }
    )


def _package_version() -> str:
    for distribution in ("wide-angle-propagation", "WideAnglePropagation"):
        try:
            return importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            continue
    return "source-tree-uninstalled"


def _make_provenance(
    prepared: PreparedAtomisticEditReconstruction1D,
    result: AtomisticEditReconstruction1D,
    caller_metadata: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    devices = []
    for device in jax.devices():
        devices.append(
            {
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "id": int(device.id),
            }
        )
    state = result.debiased_state
    array_dtypes = {
        "probe_rows": str(np.asarray(prepared.probe_rows).dtype),
        "propagation_kernel": str(np.asarray(prepared.propagation_kernel).dtype),
        "measurement_signal": str(
            np.asarray(prepared.measurement.calibrated_signal_electrons).dtype
        ),
        **{
            f"state_{name}": str(np.asarray(getattr(state, name)).dtype)
            for name in _STATE_FIELDS
        },
    }
    value = {
        "schema": "atomistic_edit_reconstruction_provenance_1d:v1",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "package_version": _package_version(),
        "platform": platform.platform(),
        "jax_default_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "devices": devices,
        "array_dtypes": array_dtypes,
        "caller_metadata": dict(caller_metadata or {}),
    }
    return _metadata("provenance", value)


def _validate_provenance(value: Mapping[str, Any]) -> Mapping[str, Any]:
    fields = _json_mapping(
        _canonical_json(value),
        name="provenance_json",
        expected_fields=_PROVENANCE_FIELDS,
    )
    if fields["schema"] != "atomistic_edit_reconstruction_provenance_1d:v1":
        raise ValueError("unsupported provenance schema")
    for name in (
        "python_version",
        "numpy_version",
        "jax_version",
        "package_version",
        "platform",
        "jax_default_backend",
    ):
        if not isinstance(fields[name], str) or not fields[name]:
            raise ValueError(f"provenance.{name} must be non-empty text")
    _boolean("provenance.jax_enable_x64", fields["jax_enable_x64"])
    if not isinstance(fields["devices"], list) or not fields["devices"]:
        raise ValueError("provenance.devices must be a non-empty list")
    for index, device in enumerate(fields["devices"]):
        if not isinstance(device, dict) or set(device) != {
            "platform",
            "device_kind",
            "id",
        }:
            raise ValueError(f"provenance.devices[{index}] has the wrong schema")
        if not isinstance(device["platform"], str) or not isinstance(
            device["device_kind"], str
        ):
            raise TypeError("provenance device names must be text")
        _integer(f"provenance.devices[{index}].id", device["id"], nonnegative=True)
    if not isinstance(fields["array_dtypes"], dict) or not fields["array_dtypes"]:
        raise ValueError("provenance.array_dtypes must be a non-empty object")
    if not all(
        isinstance(name, str) and isinstance(dtype, str) and dtype
        for name, dtype in fields["array_dtypes"].items()
    ):
        raise TypeError("provenance array dtype entries must be text")
    if not isinstance(fields["caller_metadata"], dict):
        raise TypeError("provenance.caller_metadata must be an object")
    return MappingProxyType(fields)


def _nested_ae1_snapshot_bytes(model: AtomisticEditModel1D) -> np.ndarray:
    state = empty_atomistic_edit_state_1d(model)
    snapshot = make_atomistic_edit_snapshot_1d(
        model,
        state,
        selected_edit_penalty=float(model.options.edit_penalty_path[0]),
        edit_penalty_rule_id=_SELECTION_RULE,
        data_objective_value=0.0,
        data_objective_id=_MODEL_ANCHOR_OBJECTIVE_ID,
        metadata={
            "schema": "ae2_archive_model_anchor:v1",
            "purpose": "authenticated_model_transport_not_solver_evidence",
        },
    )
    with tempfile.TemporaryDirectory(prefix="ae2-model-snapshot-") as directory:
        path = Path(directory) / "model.npz"
        save_atomistic_edit_snapshot_1d(path, snapshot)
        return np.frombuffer(path.read_bytes(), dtype=np.uint8).copy()


def _load_nested_ae1_snapshot(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype != np.uint8 or array.size == 0:
        raise ValueError("ae1_model_snapshot_npz must be a non-empty uint8 vector")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", prefix="ae2-model-snapshot-", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(array.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        snapshot = load_atomistic_edit_snapshot_1d(temporary_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    if snapshot.data_objective_id != _MODEL_ANCHOR_OBJECTIVE_ID:
        raise ValueError("nested AE-1 snapshot is not an AE-2 model anchor")
    expected_metadata = {
        "schema": "ae2_archive_model_anchor:v1",
        "purpose": "authenticated_model_transport_not_solver_evidence",
    }
    if dict(snapshot.metadata) != expected_metadata:
        raise ValueError("nested AE-1 model-anchor metadata is inconsistent")
    empty = empty_atomistic_edit_state_1d(snapshot.model)
    for name in _STATE_FIELDS:
        if not np.array_equal(
            np.asarray(getattr(snapshot.state, name)),
            np.asarray(getattr(empty, name)),
        ):
            raise ValueError("nested AE-1 model anchor does not contain an empty state")
    return snapshot


def _archive_digest(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload):
        array = np.ascontiguousarray(np.asarray(payload[name]))
        header = _canonical_json(
            {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)}
        ).encode("utf-8")
        for value in (header, array.tobytes(order="C")):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    contract = _canonical_json(
        {"contract": _ARCHIVE_CONTRACT, "schema_version": _SCHEMA_VERSION}
    ).encode("utf-8")
    digest.update(len(contract).to_bytes(8, "big"))
    digest.update(contract)
    return digest.hexdigest()


def _same_array(name: str, first: Any, second: Any) -> None:
    left = np.asarray(first)
    right = np.asarray(second)
    if (
        left.shape != right.shape
        or left.dtype != right.dtype
        or not np.array_equal(left, right)
    ):
        raise ValueError(f"{name} is not reproduced exactly")


def _same_state(name: str, first: AtomisticEditState1D, second: AtomisticEditState1D) -> None:
    for field_name in _STATE_FIELDS:
        _same_array(
            f"{name}.{field_name}",
            getattr(first, field_name),
            getattr(second, field_name),
        )


def _numeric_tolerance(values: Sequence[Any]) -> tuple[float, float]:
    dtypes = [np.asarray(value).dtype for value in values]
    real_dtypes = [
        np.empty((), dtype=dtype).real.dtype
        for dtype in dtypes
        if np.issubdtype(dtype, np.number)
    ]
    if not real_dtypes:
        return 0.0, 0.0
    dtype = np.result_type(*real_dtypes)
    epsilon = np.finfo(dtype if dtype.kind == "f" else np.float64).eps
    return float(256.0 * epsilon), float(256.0 * epsilon)


def _same_number(name: str, stored: Any, recomputed: Any, *, dtype_sources: Sequence[Any]) -> None:
    left = float(stored)
    right = float(np.asarray(recomputed))
    if np.isneginf(left) and np.isneginf(right):
        return
    if not np.isfinite(left) or not np.isfinite(right):
        raise ValueError(f"{name} contains an unsupported non-finite value")
    rtol, epsilon = _numeric_tolerance(dtype_sources)
    atol = epsilon * max(1.0, abs(left), abs(right))
    if not np.isclose(left, right, rtol=rtol, atol=atol):
        raise ValueError(f"{name} is not numerically reproducible")


def _same_gradient_norm(
    name: str,
    stored: Any,
    recomputed: Any,
    *,
    dtype_sources: Sequence[Any],
) -> None:
    """Compare replayed reverse-mode reductions with a small roundoff floor."""
    left = float(stored)
    right = float(np.asarray(recomputed))
    if not np.isfinite(left) or not np.isfinite(right):
        raise ValueError(f"{name} contains an unsupported non-finite value")
    base_rtol, base_epsilon = _numeric_tolerance(dtype_sources)
    # Batched FFT adjoints can change their final reduction order across XLA
    # executions.  Keep the allowance far below the solver's declared KKT
    # tolerances while avoiding a false archive failure at float64 roundoff.
    rtol = max(base_rtol, 1e-10)
    atol = max(
        base_epsilon * max(1.0, abs(left), abs(right)),
        1e-10 * max(1.0, abs(left), abs(right)),
    )
    if not np.isclose(left, right, rtol=rtol, atol=atol):
        raise ValueError(
            f"{name} is not numerically reproducible: "
            f"stored={left:.17g}, replayed={right:.17g}"
        )


def _same_objective(
    name: str,
    stored: AtomisticEditObjectiveComponents1D,
    recomputed: AtomisticEditObjectiveComponents1D,
    prepared: PreparedAtomisticEditReconstruction1D,
) -> None:
    if stored.ablation != recomputed.ablation:
        raise ValueError(f"{name}.ablation is inconsistent")
    if stored.edit_penalty != recomputed.edit_penalty:
        raise ValueError(f"{name}.edit_penalty is inconsistent")
    _same_array(f"{name}.scan_indices", stored.scan_indices, recomputed.scan_indices)
    sources = (
        prepared.probe_rows,
        prepared.propagation_kernel,
        prepared.measurement.calibrated_signal_electrons,
    )
    for field_name in (
        "count_deviance",
        "edit_mass",
        "weighted_edit_penalty",
        "elastic_penalty",
        "hard_core_penalty",
        "total_objective",
    ):
        _same_number(
            f"{name}.{field_name}",
            getattr(stored, field_name),
            getattr(recomputed, field_name),
            dtype_sources=sources,
        )


def _finite_maximum(value: Any) -> float:
    array = np.asarray(value, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else -np.inf


def _projected_gradient_norm(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    edit_penalty: float,
    ablation: str,
    *,
    freeze_positions: bool,
    training_scan_batch_size: int | None,
) -> float:
    _, gradients = _objective_value_and_gradients(
        prepared,
        state,
        edit_penalty,
        ablation,
        training_scan_batch_size=training_scan_batch_size,
    )
    model = prepared.model
    maximum_mass = float(model.options.max_scattering_equivalent_per_centre)
    half_pixel = np.asarray(
        [
            0.5 * model.addition_kernel.axial_sampling_A,
            0.5 * model.addition_kernel.transverse_sampling_A,
        ]
    )
    maximum_displacement = float(np.asarray(model.host_model.maximum_displacement))

    def projected(
        values: Any,
        gradient: Any,
        lower: Any,
        upper: Any,
        active: Any,
    ) -> np.ndarray:
        values_array = np.asarray(values, dtype=float)
        gradient_array = np.asarray(gradient, dtype=float)
        lower_array = np.broadcast_to(np.asarray(lower), values_array.shape)
        upper_array = np.broadcast_to(np.asarray(upper), values_array.shape)
        result = np.where(
            values_array <= lower_array + 1e-12,
            np.minimum(gradient_array, 0.0),
            gradient_array,
        )
        result = np.where(
            values_array >= upper_array - 1e-12,
            np.maximum(result, 0.0),
            result,
        )
        return np.where(np.broadcast_to(active, values_array.shape), result, 0.0)

    removal_active = np.asarray(state.host_removal_active, dtype=bool)
    extra_active = np.asarray(state.extra_active, dtype=bool)
    parts = [
        projected(
            state.host_removal_fractions,
            gradients["host_removal_fractions"],
            0.0,
            1.0,
            removal_active,
        ).reshape(-1),
        projected(
            state.extra_scattering_equivalents,
            gradients["extra_scattering_equivalents"],
            0.0,
            maximum_mass,
            extra_active,
        ).reshape(-1),
    ]
    if not freeze_positions:
        parts.append(
            projected(
                state.extra_position_offsets_A,
                gradients["extra_position_offsets_A"],
                -half_pixel,
                half_pixel,
                extra_active[:, None],
            ).reshape(-1)
        )
    controls = np.asarray(state.host_displacement_controls)
    parts.append(
        projected(
            controls,
            gradients["host_displacement_controls"],
            -maximum_displacement,
            maximum_displacement,
            np.ones_like(controls, dtype=bool),
        ).reshape(-1)
    )
    combined = np.concatenate(parts)
    return float(np.max(np.abs(combined))) if combined.size else 0.0


def _capacity_status(model: AtomisticEditModel1D, state: AtomisticEditState1D) -> str:
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


def _validate_kkt(
    prepared: PreparedAtomisticEditReconstruction1D,
    state: AtomisticEditState1D,
    certificate: AtomisticEditGridKKTCertificate1D,
    options: AtomisticEditSolverOptions1D,
    *,
    name: str,
) -> None:
    if certificate.certificate_scope != _CERTIFICATE_SCOPE:
        raise ValueError(f"{name} overclaims its certificate scope")
    if certificate.continuous_birth_kkt_evaluated:
        raise ValueError(f"{name} cannot claim continuous-birth KKT evaluation")
    if certificate.proposal_tolerance != options.proposal_grid_kkt_tolerance:
        raise ValueError(f"{name}.proposal_tolerance differs from solver options")
    if certificate.active_gradient_tolerance != (
        options.active_projected_gradient_tolerance
    ):
        raise ValueError(f"{name}.active_gradient_tolerance differs from options")
    scores = atomistic_edit_proposal_scores_1d(
        prepared,
        state,
        certificate.edit_penalty,
        ablation=options.ablation,
        training_scan_batch_size=options.training_scan_batch_size,
    )
    maxima = {
        "maximum_addition_violation": _finite_maximum(
            scores.addition_violation_grid
        ),
        "maximum_host_removal_violation": _finite_maximum(
            scores.host_removal_violation
        ),
        "maximum_paired_replacement_violation": _finite_maximum(
            scores.paired_replacement_violation
        ),
    }
    sources = (prepared.probe_rows, prepared.propagation_kernel)
    for field_name, recomputed in maxima.items():
        _same_number(
            f"{name}.{field_name}",
            getattr(certificate, field_name),
            recomputed,
            dtype_sources=sources,
        )
    maximum = max(maxima.values())
    _same_number(
        f"{name}.maximum_dormant_violation",
        certificate.maximum_dormant_violation,
        maximum,
        dtype_sources=sources,
    )
    active_norm = _projected_gradient_norm(
        prepared,
        state,
        certificate.edit_penalty,
        options.ablation,
        freeze_positions=False,
        training_scan_batch_size=options.training_scan_batch_size,
    )
    _same_gradient_norm(
        f"{name}.active_projected_gradient_norm",
        certificate.active_projected_gradient_norm,
        active_norm,
        dtype_sources=sources,
    )
    proposal_ok = maximum <= certificate.proposal_tolerance
    active_ok = active_norm <= certificate.active_gradient_tolerance
    if certificate.proposal_grid_satisfied != proposal_ok:
        raise ValueError(f"{name}.proposal_grid_satisfied is inconsistent")
    if certificate.active_projected_gradient_satisfied != active_ok:
        raise ValueError(
            f"{name}.active_projected_gradient_satisfied is inconsistent"
        )
    if certificate.satisfied != bool(proposal_ok and active_ok):
        raise ValueError(f"{name}.satisfied is inconsistent")


def _validate_result_metadata(
    result: AtomisticEditReconstruction1D,
    prepared: PreparedAtomisticEditReconstruction1D,
    options: AtomisticEditSolverOptions1D,
) -> None:
    metadata = dict(result.metadata)
    if set(metadata) != _RESULT_METADATA_FIELDS:
        raise ValueError("single-start result metadata differs from its exact schema")
    training_scan_count = int(np.asarray(prepared.training_indices).size)
    scan_batching_active = bool(
        options.training_scan_batch_size is not None
        and options.training_scan_batch_size < training_scan_count
    )
    expected = {
        "schema": "atomistic_edit_reconstruction_1d:v1",
        "ablation": options.ablation,
        "seed": int(options.seed),
        "seed_used_for_single_start": False,
        "truth_inputs_used": False,
        "nuisance_image_used": False,
        "energy_envelope_used": False,
        "penalty_path": list(prepared.model.options.edit_penalty_path),
        "penalty_path_strictly_decreasing": True,
        "selection_rule": _SELECTION_RULE,
        "selection_uses_validation_only": True,
        "audit_used_for_selection": False,
        "audit_evaluated": result.debiased_audit_count_deviance is not None,
        "debias_rule": _DEBIAS_RULE,
        "debias_projected_gradient_certified": result.debias_converged,
        "duplicate_merge_resolution_A": options.duplicate_merge_resolution_A,
        "duplicate_merge_enabled": True,
        "proposal_certificate_scope": _CERTIFICATE_SCOPE,
        "continuous_birth_kkt_evaluated": False,
        "optimizer_reset_at_every_refinement": True,
        "progress_requested": metadata["progress_requested"],
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
    if type(metadata["progress_requested"]) is not bool:
        raise TypeError("result metadata progress_requested must be Boolean")
    if metadata != expected:
        raise ValueError("single-start result metadata is semantically inconsistent")


def _reprepare(prepared: PreparedAtomisticEditReconstruction1D) -> PreparedAtomisticEditReconstruction1D:
    canonical_measurement_metadata = dict(
        _metadata(
            "prepared.measurement.metadata",
            dict(prepared.measurement.metadata),
        )
    )
    if dict(prepared.measurement.metadata) != canonical_measurement_metadata:
        raise ValueError(
            "prepared measurement metadata must use JSON-native value types"
        )
    reconstructed = prepare_atomistic_edit_reconstruction_1d(
        prepared.model,
        prepared.probe_rows,
        prepared.window_starts,
        prepared.window_length,
        prepared.propagation_kernel,
        prepared.slice_thickness_A,
        prepared.energy_eV,
        prepared.measurement,
        prepared.objective,
        validation_indices=np.asarray(prepared.validation_indices),
        audit_indices=np.asarray(prepared.audit_indices),
        excluded_indices=np.asarray(prepared.excluded_indices),
    )
    for name in (
        "probe_rows",
        "window_starts",
        "propagation_kernel",
        "training_indices",
        "validation_indices",
        "audit_indices",
        "excluded_indices",
    ):
        _same_array(f"prepared.{name}", getattr(prepared, name), getattr(reconstructed, name))
    if prepared.reconstruction_problem_id != reconstructed.reconstruction_problem_id:
        raise ValueError("prepared reconstruction_problem_id is not reproducible")
    if prepared.reconstructor_id != reconstructed.reconstructor_id:
        raise ValueError("prepared reconstructor_id is not reproducible")
    stored_metadata = dict(prepared.metadata)
    constructor_metadata = dict(reconstructed.metadata)
    if any(
        key not in stored_metadata or stored_metadata[key] != value
        for key, value in constructor_metadata.items()
    ):
        raise ValueError(
            "prepared metadata does not retain its constructor-authentic fields"
        )
    # A package workflow may append digest-bound geometry/support provenance
    # after core preparation.  Those fields are descriptive and covered by the
    # outer archive digest; retain them while requiring the complete core
    # constructor contract above.
    return replace(
        reconstructed,
        metadata=MappingProxyType(stored_metadata),
    )


def _validation_selected_index(
    points: Sequence[AtomisticEditLambdaPathPoint1D],
    options: AtomisticEditSolverOptions1D,
) -> int:
    validation = np.asarray(
        [point.validation_count_deviance for point in points], dtype=float
    )
    best = float(np.min(validation))
    tolerance = options.validation_absolute_tolerance + (
        options.validation_relative_tolerance * max(abs(best), 1e-15)
    )
    return int(np.flatnonzero(validation <= best + tolerance)[0])


def validate_atomistic_edit_reconstruction_bundle_1d(
    bundle: AtomisticEditReconstructionBundle1D,
) -> AtomisticEditReconstructionBundle1D:
    """Replay and validate all scientific claims in a single-start bundle."""
    if not isinstance(bundle, AtomisticEditReconstructionBundle1D):
        raise TypeError("bundle must be AtomisticEditReconstructionBundle1D")
    prepared = bundle.prepared
    result = bundle.reconstruction
    options = bundle.solver_options
    if not isinstance(prepared, PreparedAtomisticEditReconstruction1D):
        raise TypeError("bundle.prepared has the wrong type")
    if not isinstance(result, AtomisticEditReconstruction1D):
        raise TypeError("bundle.reconstruction must be a single-start AE-2 result")
    if not isinstance(options, AtomisticEditSolverOptions1D):
        raise TypeError("bundle.solver_options has the wrong type")
    # Round-trip through the strict solver-options JSON validator.
    options = _solver_options_from_json(_solver_options_json(options))
    _reprepare(prepared)
    provenance = _validate_provenance(bundle.provenance)

    if result.prepared_problem_id != prepared.reconstruction_problem_id:
        raise ValueError("result prepared_problem_id differs from the prepared problem")
    if result.reconstructor_id != prepared.reconstructor_id:
        raise ValueError("result reconstructor_id differs from the prepared problem")
    _validate_result_metadata(result, prepared, options)
    if not result.path_points:
        raise ValueError("result.path_points must not be empty")
    path = tuple(float(value) for value in prepared.model.options.edit_penalty_path)
    point_penalties = tuple(float(point.edit_penalty) for point in result.path_points)
    if point_penalties != path[: len(point_penalties)]:
        raise ValueError("saved lambda path is not the frozen decreasing model path")
    if any(left <= right for left, right in zip(point_penalties, point_penalties[1:])):
        raise ValueError("saved lambda path is not strictly decreasing")
    if any(not point.converged for point in result.path_points[:-1]):
        raise ValueError("a path point appears after a failed warm-start point")

    ablation = options.ablation
    for index, point in enumerate(result.path_points):
        validate_atomistic_edit_state_1d(prepared.model, point.state)
        rendered = np.asarray(
            render_atomistic_edit_potential_1d(prepared.model, point.state)
        )
        if np.iscomplexobj(rendered) or np.any(~np.isfinite(rendered)):
            raise ValueError(f"path_points[{index}] does not rerender a finite potential")
        if point.training_objective.edit_penalty != point.edit_penalty:
            raise ValueError(f"path_points[{index}] objective penalty is inconsistent")
        if point.kkt.edit_penalty != point.edit_penalty:
            raise ValueError(f"path_points[{index}] KKT penalty is inconsistent")
        recomputed = atomistic_edit_objective_components_1d(
            prepared,
            point.state,
            point.edit_penalty,
            scan_indices=prepared.training_indices,
            ablation=ablation,
        )
        _same_objective(
            f"path_points[{index}].training_objective",
            point.training_objective,
            recomputed,
            prepared,
        )
        validation = atomistic_edit_objective_components_1d(
            prepared,
            point.state,
            0.0,
            scan_indices=prepared.validation_indices,
            ablation=ablation,
        ).count_deviance
        _same_number(
            f"path_points[{index}].validation_count_deviance",
            point.validation_count_deviance,
            validation,
            dtype_sources=(prepared.probe_rows, prepared.propagation_kernel),
        )
        _validate_kkt(
            prepared,
            point.state,
            point.kkt,
            options,
            name=f"path_points[{index}].kkt",
        )
        if point.active_set_iterations <= 0 or (
            point.active_set_iterations > options.maximum_active_set_iterations
        ):
            raise ValueError(f"path_points[{index}] active-set iteration count is invalid")
        for name in (
            "optimizer_reset_count",
            "pruned_host_removals",
            "pruned_extra_centres",
            "merged_extra_centres",
        ):
            if _integer(
                f"path_points[{index}].{name}", getattr(point, name), nonnegative=True
            ) != getattr(point, name):
                raise ValueError(f"path_points[{index}].{name} is invalid")
        if not isinstance(point.births, tuple) or not all(
            isinstance(value, str) and value for value in point.births
        ):
            raise TypeError(f"path_points[{index}].births must be non-empty strings")
        if point.duplicate_status not in {
            "resolved_or_absent",
            "unresolved_duplicate_merge_fail_closed",
        }:
            raise ValueError(f"path_points[{index}] duplicate status is unsupported")
        direct_capacity = _capacity_status(prepared.model, point.state)
        if point.capacity_status == "capacity_available":
            if direct_capacity != "capacity_available":
                raise ValueError(f"path_points[{index}] hides capacity saturation")
        elif point.capacity_status.startswith("saturated_resource_bound:"):
            if point.capacity_status != direct_capacity:
                raise ValueError(f"path_points[{index}] capacity status is inconsistent")
        elif point.capacity_status == "exhausted_with_violating_direction":
            if direct_capacity == "capacity_available" or point.kkt.satisfied:
                raise ValueError(f"path_points[{index}] exhaustion claim is inconsistent")
        else:
            raise ValueError(f"path_points[{index}] capacity status is unsupported")
        expected_converged = bool(
            point.capacity_status == "capacity_available"
            and point.duplicate_status == "resolved_or_absent"
            and point.kkt.satisfied
        )
        if point.converged != expected_converged:
            raise ValueError(f"path_points[{index}].converged is inconsistent")

    selected_index = _validation_selected_index(result.path_points, options)
    if result.selected_path_index != selected_index:
        raise ValueError("selected_path_index violates the frozen validation rule")
    selected = result.path_points[selected_index]
    if result.selected_edit_penalty != selected.edit_penalty:
        raise ValueError("selected_edit_penalty differs from the selected path point")
    _same_state("penalized_state", result.penalized_state, selected.state)
    _same_objective(
        "penalized_training_objective",
        result.penalized_training_objective,
        selected.training_objective,
        prepared,
    )
    _same_number(
        "penalized_validation_count_deviance",
        result.penalized_validation_count_deviance,
        selected.validation_count_deviance,
        dtype_sources=(prepared.probe_rows,),
    )
    if _kkt_json(result.selected_kkt) != _kkt_json(selected.kkt):
        raise ValueError("selected_kkt differs from the selected path certificate")

    validate_atomistic_edit_state_1d(prepared.model, result.debiased_state)
    for name in (
        "host_removal_active",
        "extra_active",
        "extra_anchor_indices",
        "extra_position_offsets_A",
    ):
        _same_array(
            f"fixed-support debias {name}",
            getattr(result.debiased_state, name),
            getattr(result.penalized_state, name),
        )
    for name, state in (
        ("penalized_state", result.penalized_state),
        ("debiased_state", result.debiased_state),
    ):
        potential = np.asarray(render_atomistic_edit_potential_1d(prepared.model, state))
        if np.iscomplexobj(potential) or np.any(~np.isfinite(potential)):
            raise ValueError(f"{name} does not rerender a finite real potential")

    debiased_objective = atomistic_edit_objective_components_1d(
        prepared,
        result.debiased_state,
        0.0,
        scan_indices=prepared.training_indices,
        ablation=ablation,
    )
    _same_objective(
        "debiased_training_objective",
        result.debiased_training_objective,
        debiased_objective,
        prepared,
    )
    debiased_validation = atomistic_edit_objective_components_1d(
        prepared,
        result.debiased_state,
        0.0,
        scan_indices=prepared.validation_indices,
        ablation=ablation,
    ).count_deviance
    _same_number(
        "debiased_validation_count_deviance",
        result.debiased_validation_count_deviance,
        debiased_validation,
        dtype_sources=(prepared.probe_rows, prepared.propagation_kernel),
    )
    if result.debiased_audit_count_deviance is None:
        if dict(result.metadata)["audit_evaluated"]:
            raise ValueError("audit metadata claims an absent audit evaluation")
    else:
        if np.asarray(prepared.audit_indices).size == 0:
            raise ValueError("an audit loss is stored without audit scans")
        audit = atomistic_edit_objective_components_1d(
            prepared,
            result.debiased_state,
            0.0,
            scan_indices=prepared.audit_indices,
            ablation=ablation,
        ).count_deviance
        _same_number(
            "debiased_audit_count_deviance",
            result.debiased_audit_count_deviance,
            audit,
            dtype_sources=(prepared.probe_rows, prepared.propagation_kernel),
        )

    debias_norm = _projected_gradient_norm(
        prepared,
        result.debiased_state,
        0.0,
        ablation,
        freeze_positions=True,
        training_scan_batch_size=options.training_scan_batch_size,
    )
    _same_gradient_norm(
        "debias_projected_gradient_norm",
        result.debias_projected_gradient_norm,
        debias_norm,
        dtype_sources=(prepared.probe_rows, prepared.propagation_kernel),
    )
    if result.debias_projected_gradient_tolerance != (
        options.debias_projected_gradient_tolerance
    ):
        raise ValueError("debias projected-gradient tolerance differs from options")
    if result.debias_converged != bool(
        debias_norm <= result.debias_projected_gradient_tolerance
    ):
        raise ValueError("debias_converged is inconsistent")
    active_count = atomistic_edit_active_parameter_count_1d(
        prepared.model, result.debiased_state
    )
    if result.active_parameter_count != active_count:
        raise ValueError("active_parameter_count is inconsistent")
    capacity_exhausted = any(
        point.capacity_status != "capacity_available" for point in result.path_points
    )
    if result.capacity_exhausted != capacity_exhausted:
        raise ValueError("capacity_exhausted is inconsistent")
    path_complete = len(result.path_points) == len(path) and all(
        point.converged for point in result.path_points
    )
    expected_converged = bool(
        path_complete and not capacity_exhausted and result.debias_converged
    )
    if result.converged != expected_converged:
        raise ValueError("result.converged is inconsistent")
    if capacity_exhausted:
        expected_stop = "capacity_bound_fail_closed"
    elif any(
        point.duplicate_status != "resolved_or_absent"
        for point in result.path_points
    ):
        expected_stop = "duplicate_merge_fail_closed"
    elif not path_complete:
        expected_stop = "regularization_path_incomplete"
    elif not result.debias_converged:
        expected_stop = "debias_projected_gradient_not_converged"
    else:
        expected_stop = "frozen_path_solved_and_validation_selected"
    if result.stop_reason != expected_stop:
        raise ValueError("result.stop_reason is inconsistent")

    archive_id = bundle.archive_id
    if archive_id:
        if len(archive_id) != 64 or any(
            character not in "0123456789abcdef" for character in archive_id
        ):
            raise ValueError("archive_id must be a lowercase SHA-256 digest")
    return replace(bundle, solver_options=options, provenance=provenance)


def make_atomistic_edit_reconstruction_bundle_1d(
    prepared: PreparedAtomisticEditReconstruction1D,
    reconstruction: AtomisticEditReconstruction1D,
    *,
    solver_options: AtomisticEditSolverOptions1D | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> AtomisticEditReconstructionBundle1D:
    """Create and fully replay a single-start AE-2 archive bundle.

    ``solver_options`` defaults to the public solver defaults.  A result made
    with custom options must pass those exact options; mismatches fail closed.
    Caller provenance is descriptive only and never becomes a trust Boolean.
    """
    options = AtomisticEditSolverOptions1D() if solver_options is None else solver_options
    bundle = AtomisticEditReconstructionBundle1D(
        prepared=prepared,
        reconstruction=reconstruction,
        solver_options=options,
        provenance=_make_provenance(prepared, reconstruction, provenance),
    )
    return validate_atomistic_edit_reconstruction_bundle_1d(bundle)


def _archive_payload(bundle: AtomisticEditReconstructionBundle1D) -> dict[str, np.ndarray]:
    bundle = validate_atomistic_edit_reconstruction_bundle_1d(bundle)
    prepared = bundle.prepared
    result = bundle.reconstruction
    path_points = result.path_points
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(_SCHEMA_VERSION, dtype=np.int64),
        "archive_contract": np.asarray(_ARCHIVE_CONTRACT),
        "ae1_model_snapshot_npz": _nested_ae1_snapshot_bytes(prepared.model),
        "prepared_probe_rows": np.asarray(prepared.probe_rows),
        "prepared_window_starts": np.asarray(prepared.window_starts),
        "prepared_propagation_kernel": np.asarray(prepared.propagation_kernel),
        "prepared_measurement_signal": np.asarray(
            prepared.measurement.calibrated_signal_electrons
        ),
        "prepared_measurement_total": np.asarray(
            prepared.measurement.observed_total_electrons
        ),
        "prepared_measurement_valid": np.asarray(prepared.measurement.valid_mask),
        "prepared_measurement_dark": np.asarray(
            prepared.measurement.calibrated_dark_electrons_per_pixel
        ),
        "prepared_measurement_read_noise": np.asarray(
            prepared.measurement.calibrated_read_noise_std_electrons
        ),
        "prepared_objective_dose": np.asarray(
            prepared.objective.electrons_per_pattern
        ),
        "prepared_training_indices": np.asarray(prepared.training_indices),
        "prepared_validation_indices": np.asarray(prepared.validation_indices),
        "prepared_audit_indices": np.asarray(prepared.audit_indices),
        "prepared_excluded_indices": np.asarray(prepared.excluded_indices),
        "prepared_json": np.asarray(_prepared_json(prepared)),
        "solver_options_json": np.asarray(_solver_options_json(bundle.solver_options)),
        "result_json": np.asarray(_result_json(result)),
        "path_training_scan_indices": np.stack(
            [np.asarray(point.training_objective.scan_indices) for point in path_points]
        ),
        "penalized_training_scan_indices": np.asarray(
            result.penalized_training_objective.scan_indices
        ),
        "debiased_training_scan_indices": np.asarray(
            result.debiased_training_objective.scan_indices
        ),
        "provenance_json": np.asarray(_canonical_json(bundle.provenance)),
        **_state_arrays("penalized_state", result.penalized_state),
        **_state_arrays("debiased_state", result.debiased_state),
        **_stacked_state_arrays([point.state for point in path_points]),
    }
    if set(payload) != _ARCHIVE_FIELDS:
        raise RuntimeError("internal AE-2 archive schema is incomplete")
    return payload


def save_atomistic_edit_reconstruction_bundle_1d(
    path: str | Path,
    bundle: AtomisticEditReconstructionBundle1D,
) -> None:
    """Atomically write a non-pickled, SHA-256-bound AE-2 archive."""
    payload = _archive_payload(bundle)
    archive = {**payload, "archive_sha256": np.asarray(_archive_digest(payload))}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            suffix=".npz",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **archive)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_scalar(
    payload: Mapping[str, np.ndarray],
    name: str,
    *,
    dtype: Any | None = None,
    unicode: bool = False,
) -> Any:
    array = np.asarray(payload[name])
    if array.shape != ():
        raise ValueError(f"archive field {name} must be scalar")
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise ValueError(f"archive field {name} has the wrong dtype")
    if unicode and array.dtype.kind != "U":
        raise ValueError(f"archive field {name} must be scalar Unicode")
    return array.item()


def _load_result(
    fields: Mapping[str, Any],
    payload: Mapping[str, np.ndarray],
    model: AtomisticEditModel1D,
) -> AtomisticEditReconstruction1D:
    path_fields = fields["path_points"]
    if not isinstance(path_fields, list) or not path_fields:
        raise ValueError("result_json.path_points must be a non-empty list")
    path_length = len(path_fields)
    empty = empty_atomistic_edit_state_1d(model)
    path_scan_indices = np.asarray(payload["path_training_scan_indices"])
    expected_scan_shape = (
        path_length,
        np.asarray(payload["prepared_training_indices"]).size,
    )
    if path_scan_indices.shape != expected_scan_shape or path_scan_indices.dtype != np.int32:
        raise ValueError(
            "path_training_scan_indices has the wrong shape or dtype"
        )
    points = []
    for index, point_fields in enumerate(path_fields):
        if not isinstance(point_fields, dict) or set(point_fields) != _PATH_POINT_JSON_FIELDS:
            raise ValueError(f"result_json.path_points[{index}] has the wrong schema")
        births = point_fields["births"]
        if not isinstance(births, list) or not all(
            isinstance(value, str) and value for value in births
        ):
            raise TypeError(f"result_json.path_points[{index}].births is invalid")
        state = _state_from_payload(
            payload,
            "path_state",
            empty,
            path_index=index,
            path_length=path_length,
        )
        points.append(
            AtomisticEditLambdaPathPoint1D(
                edit_penalty=_number(
                    f"path_points[{index}].edit_penalty",
                    point_fields["edit_penalty"],
                    positive=True,
                ),
                state=state,
                training_objective=_objective_from_json(
                    point_fields["training_objective"],
                    path_scan_indices[index],
                    name=f"path_points[{index}].training_objective",
                ),
                validation_count_deviance=_number(
                    f"path_points[{index}].validation_count_deviance",
                    point_fields["validation_count_deviance"],
                ),
                kkt=_kkt_from_json(
                    point_fields["kkt"], name=f"path_points[{index}].kkt"
                ),
                active_set_iterations=_integer(
                    f"path_points[{index}].active_set_iterations",
                    point_fields["active_set_iterations"],
                    nonnegative=True,
                ),
                optimizer_reset_count=_integer(
                    f"path_points[{index}].optimizer_reset_count",
                    point_fields["optimizer_reset_count"],
                    nonnegative=True,
                ),
                births=tuple(births),
                pruned_host_removals=_integer(
                    f"path_points[{index}].pruned_host_removals",
                    point_fields["pruned_host_removals"],
                    nonnegative=True,
                ),
                pruned_extra_centres=_integer(
                    f"path_points[{index}].pruned_extra_centres",
                    point_fields["pruned_extra_centres"],
                    nonnegative=True,
                ),
                merged_extra_centres=_integer(
                    f"path_points[{index}].merged_extra_centres",
                    point_fields["merged_extra_centres"],
                    nonnegative=True,
                ),
                duplicate_status=_identifier(
                    f"path_points[{index}].duplicate_status",
                    point_fields["duplicate_status"],
                ),
                capacity_status=_identifier(
                    f"path_points[{index}].capacity_status",
                    point_fields["capacity_status"],
                ),
                stop_reason=_identifier(
                    f"path_points[{index}].stop_reason", point_fields["stop_reason"]
                ),
                converged=_boolean(
                    f"path_points[{index}].converged", point_fields["converged"]
                ),
            )
        )

    penalized_scan_indices = np.asarray(payload["penalized_training_scan_indices"])
    debiased_scan_indices = np.asarray(payload["debiased_training_scan_indices"])
    training_shape = np.asarray(payload["prepared_training_indices"]).shape
    for name, array in (
        ("penalized_training_scan_indices", penalized_scan_indices),
        ("debiased_training_scan_indices", debiased_scan_indices),
    ):
        if array.shape != training_shape or array.dtype != np.int32:
            raise ValueError(f"{name} has the wrong shape or dtype")
    audit_value = fields["debiased_audit_count_deviance"]
    if audit_value is not None:
        audit_value = _number("debiased_audit_count_deviance", audit_value)
    return AtomisticEditReconstruction1D(
        prepared_problem_id=_identifier(
            "result_json.prepared_problem_id", fields["prepared_problem_id"]
        ),
        reconstructor_id=_identifier(
            "result_json.reconstructor_id", fields["reconstructor_id"]
        ),
        penalized_state=_state_from_payload(payload, "penalized_state", empty),
        debiased_state=_state_from_payload(payload, "debiased_state", empty),
        selected_edit_penalty=_number(
            "result_json.selected_edit_penalty",
            fields["selected_edit_penalty"],
            positive=True,
        ),
        selected_path_index=_integer(
            "result_json.selected_path_index",
            fields["selected_path_index"],
            nonnegative=True,
        ),
        path_points=tuple(points),
        penalized_training_objective=_objective_from_json(
            fields["penalized_training_objective"],
            penalized_scan_indices,
            name="penalized_training_objective",
        ),
        debiased_training_objective=_objective_from_json(
            fields["debiased_training_objective"],
            debiased_scan_indices,
            name="debiased_training_objective",
        ),
        penalized_validation_count_deviance=_number(
            "result_json.penalized_validation_count_deviance",
            fields["penalized_validation_count_deviance"],
        ),
        debiased_validation_count_deviance=_number(
            "result_json.debiased_validation_count_deviance",
            fields["debiased_validation_count_deviance"],
        ),
        debiased_audit_count_deviance=audit_value,
        selected_kkt=_kkt_from_json(fields["selected_kkt"], name="selected_kkt"),
        debias_projected_gradient_norm=_number(
            "result_json.debias_projected_gradient_norm",
            fields["debias_projected_gradient_norm"],
            nonnegative=True,
        ),
        debias_projected_gradient_tolerance=_number(
            "result_json.debias_projected_gradient_tolerance",
            fields["debias_projected_gradient_tolerance"],
            positive=True,
        ),
        debias_converged=_boolean(
            "result_json.debias_converged", fields["debias_converged"]
        ),
        active_parameter_count=_integer(
            "result_json.active_parameter_count",
            fields["active_parameter_count"],
            nonnegative=True,
        ),
        capacity_exhausted=_boolean(
            "result_json.capacity_exhausted", fields["capacity_exhausted"]
        ),
        converged=_boolean("result_json.converged", fields["converged"]),
        stop_reason=_identifier("result_json.stop_reason", fields["stop_reason"]),
        metadata=_metadata("result_json.metadata", fields["metadata"]),
    )


def load_atomistic_edit_reconstruction_bundle_1d(
    path: str | Path,
) -> AtomisticEditReconstructionBundle1D:
    """Load, authenticate, reconstruct, rerender, and replay an AE-2 archive."""
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            expected = _ARCHIVE_FIELDS | {"archive_sha256"}
            actual = set(archive.files)
            if actual != expected:
                raise ValueError(
                    "AE-2 archive fields differ from schema: "
                    f"missing={sorted(expected - actual)}, "
                    f"extra={sorted(actual - expected)}"
                )
            payload = {
                name: np.array(archive[name], copy=True, order="C")
                for name in _ARCHIVE_FIELDS
            }
            stored_digest = np.array(archive["archive_sha256"], copy=True)
    except (OSError, EOFError, KeyError) as error:
        raise ValueError("AE-2 reconstruction archive is unreadable") from error
    if _load_scalar(payload, "schema_version", dtype=np.int64) != _SCHEMA_VERSION:
        raise ValueError("unsupported AE-2 reconstruction archive schema")
    if _load_scalar(payload, "archive_contract", unicode=True) != _ARCHIVE_CONTRACT:
        raise ValueError("unsupported AE-2 reconstruction archive contract")
    if stored_digest.shape != () or stored_digest.dtype.kind != "U":
        raise ValueError("archive_sha256 must be scalar Unicode")
    digest = str(stored_digest.item())
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or digest != _archive_digest(payload)
    ):
        raise ValueError("AE-2 reconstruction archive SHA-256 verification failed")

    snapshot = _load_nested_ae1_snapshot(payload["ae1_model_snapshot_npz"])
    prepared_fields = _json_mapping(
        payload["prepared_json"],
        name="prepared_json",
        expected_fields=_PREPARED_JSON_FIELDS,
    )
    if prepared_fields["model_id"] != snapshot.model.model_id:
        raise ValueError("prepared model_id differs from the authenticated AE-1 model")
    measurement = PtychographyMeasurement1D(
        calibrated_signal_electrons=jnp.asarray(payload["prepared_measurement_signal"]),
        observed_total_electrons=jnp.asarray(payload["prepared_measurement_total"]),
        valid_mask=jnp.asarray(payload["prepared_measurement_valid"]),
        calibrated_dark_electrons_per_pixel=jnp.asarray(
            payload["prepared_measurement_dark"]
        ),
        calibrated_read_noise_std_electrons=jnp.asarray(
            payload["prepared_measurement_read_noise"]
        ),
        calibration_id=_identifier(
            "prepared_json.measurement_calibration_id",
            prepared_fields["measurement_calibration_id"],
        ),
        metadata=_metadata(
            "prepared_json.measurement_metadata",
            prepared_fields["measurement_metadata"],
        ),
    )
    objective_kind = prepared_fields["objective_kind"]
    if objective_kind != "poisson_deviance":
        raise ValueError("AE-2 archive objective must be poisson_deviance")
    objective = PtychographyObjective1D(
        kind=objective_kind,
        electrons_per_pattern=jnp.asarray(payload["prepared_objective_dose"]),
        minimum_expected_electrons=_number(
            "prepared_json.objective_minimum_expected_electrons",
            prepared_fields["objective_minimum_expected_electrons"],
            positive=True,
        ),
        relative_signal_scale=_number(
            "prepared_json.objective_relative_signal_scale",
            prepared_fields["objective_relative_signal_scale"],
            positive=True,
        ),
    )
    for name in (
        "prepared_window_starts",
        "prepared_training_indices",
        "prepared_validation_indices",
        "prepared_audit_indices",
        "prepared_excluded_indices",
    ):
        array = np.asarray(payload[name])
        if array.ndim != 1 or array.dtype != np.int32:
            raise ValueError(f"{name} must be a one-dimensional int32 array")
    if np.asarray(payload["prepared_measurement_valid"]).dtype != bool:
        raise ValueError("prepared_measurement_valid must have Boolean dtype")
    prepared = prepare_atomistic_edit_reconstruction_1d(
        snapshot.model,
        payload["prepared_probe_rows"],
        payload["prepared_window_starts"],
        _integer(
            "prepared_json.window_length",
            prepared_fields["window_length"],
            nonnegative=True,
        ),
        payload["prepared_propagation_kernel"],
        _number(
            "prepared_json.slice_thickness_A",
            prepared_fields["slice_thickness_A"],
            positive=True,
        ),
        _number(
            "prepared_json.energy_eV", prepared_fields["energy_eV"], positive=True
        ),
        measurement,
        objective,
        validation_indices=payload["prepared_validation_indices"],
        audit_indices=payload["prepared_audit_indices"],
        excluded_indices=payload["prepared_excluded_indices"],
    )
    for name, archived in (
        ("training_indices", payload["prepared_training_indices"]),
        ("validation_indices", payload["prepared_validation_indices"]),
        ("audit_indices", payload["prepared_audit_indices"]),
        ("excluded_indices", payload["prepared_excluded_indices"]),
    ):
        _same_array(f"prepared.{name}", getattr(prepared, name), archived)
    if prepared.reconstruction_problem_id != _identifier(
        "prepared_json.reconstruction_problem_id",
        prepared_fields["reconstruction_problem_id"],
    ):
        raise ValueError("prepared reconstruction_problem_id does not replay")
    if prepared.reconstructor_id != _identifier(
        "prepared_json.reconstructor_id", prepared_fields["reconstructor_id"]
    ):
        raise ValueError("prepared reconstructor_id does not replay")
    archived_prepared_metadata = dict(
        _metadata("prepared_json.metadata", prepared_fields["metadata"])
    )
    constructor_prepared_metadata = dict(prepared.metadata)
    if any(
        key not in archived_prepared_metadata
        or archived_prepared_metadata[key] != value
        for key, value in constructor_prepared_metadata.items()
    ):
        raise ValueError(
            "prepared metadata does not retain its constructor-authentic fields"
        )
    prepared = replace(
        prepared,
        metadata=MappingProxyType(archived_prepared_metadata),
    )

    options = _solver_options_from_json(payload["solver_options_json"])
    result_fields = _json_mapping(
        payload["result_json"],
        name="result_json",
        expected_fields=_RESULT_JSON_FIELDS,
    )
    result = _load_result(result_fields, payload, prepared.model)
    provenance = _validate_provenance(
        _json_mapping(
            payload["provenance_json"],
            name="provenance_json",
            expected_fields=_PROVENANCE_FIELDS,
        )
    )
    bundle = AtomisticEditReconstructionBundle1D(
        prepared=prepared,
        reconstruction=result,
        solver_options=options,
        provenance=provenance,
        archive_id=digest,
    )
    return validate_atomistic_edit_reconstruction_bundle_1d(bundle)
