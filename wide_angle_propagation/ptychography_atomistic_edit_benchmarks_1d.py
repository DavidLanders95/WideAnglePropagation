"""Truth-isolated AE-3 benchmark contracts for sparse atomistic edits.

This module is deliberately a *contract and assessment* layer.  It neither
generates specimens nor simulates diffraction.  A case factory supplies an
immutable, case-agnostic public count problem and a lazy private-truth
callback.  Reconstruction callbacks see only the public problem; every truth
callback is evaluated after all reconstruction calls have returned.

The maintained v1 benchmark always records the material-energy ablation as
``blocked_not_run``.  There is no Boolean escape hatch for enabling it: a
future implementation must introduce a typed chemistry-validation contract
covering surfaces, defects, strain, and cross-species environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import operator
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np


__all__ = [
    "AE3_ABLATION_CATALOG_1D",
    "AE3_BLIND_CASE_CATALOG_1D",
    "ActiveEditMultistartEvidence1D",
    "AtomisticEditAblationArm1D",
    "AtomisticEditAblationStatus1D",
    "AtomisticEditBlindAcceptancePolicy1D",
    "AtomisticEditBlindAuditCounts1D",
    "AtomisticEditBlindBenchmarkReport1D",
    "AtomisticEditBlindCase1D",
    "AtomisticEditBlindCaseRole1D",
    "AtomisticEditBlindPrivateTruth1D",
    "AtomisticEditBlindPublicProblem1D",
    "AtomisticEditBlindReconstruction1D",
    "AtomisticEditCaseAblationReport1D",
    "AtomisticEditReconstructionContract1D",
    "DerivedAtomisticEditGate1D",
    "HeldOutCountMetrics1D",
    "NuisanceAttributionEvidence1D",
    "ObservabilityEvidence1D",
    "PhysicalAdmissibilityMetrics1D",
    "ReconstructionArchiveEvidence1D",
    "ResolutionAwareMassMeasure1D",
    "ResolutionAwareMassTransportMetrics1D",
    "atomistic_edit_public_problem_digest_1d",
    "atomistic_edit_public_problem_schema_digest_1d",
    "resolution_aware_mass_transport_metrics_1d",
    "run_atomistic_edit_blind_benchmarks_1d",
    "validate_atomistic_edit_blind_benchmark_report_1d",
    "validate_atomistic_edit_blind_public_problem_1d",
]


Array = Any
BlindReconstructionCallback1D = Callable[
    ["AtomisticEditBlindPublicProblem1D"],
    "AtomisticEditBlindReconstruction1D",
]


class AtomisticEditBlindCaseRole1D(str, Enum):
    """The eight mandatory private AE-3 truth roles."""

    PRISTINE_HOST = "pristine_host"
    ONE_VACANCY = "one_vacancy"
    ONE_OFF_LATTICE_ADDITION = "one_off_lattice_interstitial_or_adatom"
    ONE_SUBSTITUTION = "one_substitution_different_truth_kernel"
    IRREGULAR_FINITE_CLUSTER = "irregular_finite_added_cluster"
    METASTABLE_DEFECT = "data_supported_strained_or_metastable_defect"
    NUISANCE_ONLY_MISMATCH = "probe_scan_or_coherence_mismatch_no_defect"
    AXIALLY_UNRESOLVED_ADDITION = "axially_unresolved_addition"


class AtomisticEditAblationArm1D(str, Enum):
    """The three frozen AE-3 ablation arms."""

    COUNT_AND_EDIT = "a0_count_likelihood_plus_edit_penalty"
    LEVEL1_PHYSICAL = "a1_plus_hard_core_and_host_elasticity"
    MATERIAL_ENERGY_ENVELOPE = "a2_plus_material_energy_envelope"


class AtomisticEditAblationStatus1D(str, Enum):
    """Execution status; this is not an acceptance Boolean."""

    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED_NOT_RUN = "blocked_not_run"


AE3_BLIND_CASE_CATALOG_1D = tuple(AtomisticEditBlindCaseRole1D)
AE3_ABLATION_CATALOG_1D = tuple(AtomisticEditAblationArm1D)

_PUBLIC_SCHEMA_ID = (
    "wide_angle_propagation.atomistic_edit_blind_public_problem_1d:v1"
)
_REPORT_SCHEMA_ID = (
    "wide_angle_propagation.atomistic_edit_blind_benchmark_report_1d:v1"
)
_ENERGY_BLOCK_REASON = (
    "blocked_not_run: v1 has no typed chemistry-validation gate covering "
    "surfaces, defects, strain, and cross-species environments"
)
_FORBIDDEN_PUBLIC_KEY_FRAGMENTS = (
    "case",
    "truth",
    "seed",
    "label",
    "object",
    "defect",
    "mismatch",
    "cause",
    "generating",
    "element",
    "particle",
    "cluster",
    "vacancy",
    "interstitial",
    "adatom",
    "substitution",
    "metastable",
    "unresolved",
    "private",
)


def _readonly_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _finite(name: str, value: Any, *, nonnegative: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _index(name: str, value: Any, *, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < (0 if allow_zero else 1):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")
    return int(result)


def _sha256_text(name: str, value: Any) -> str:
    result = str(value)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return result


def _identifier(name: str, value: Any) -> str:
    result = str(value)
    if not result or result.strip() != result:
        raise ValueError(f"{name} must be a non-empty stripped identifier")
    return result


def _check_public_key(name: str) -> str:
    key = _identifier("public payload key", name)
    normalized = "".join(character for character in key.lower() if character.isalnum())
    for fragment in _FORBIDDEN_PUBLIC_KEY_FRAGMENTS:
        compact = "".join(c for c in fragment if c.isalnum())
        if compact in normalized:
            raise ValueError(
                f"public payload key {key!r} may disclose private {fragment!r} metadata"
            )
    return key


def _hash_array(hasher: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(json.dumps(array.shape).encode("ascii"))
    hasher.update(array.view(np.uint8).tobytes())


@dataclass(frozen=True)
class AtomisticEditReconstructionContract1D:
    """Opaque, case-invariant reconstruction/prior/selection identities."""

    model_sha256: str
    options_sha256: str
    prior_sha256: str
    selection_rule_sha256: str
    nuisance_scope_sha256: str
    observability_rule_sha256: str
    fitted_spatial_dimension: int = 2
    schema_id: str = _PUBLIC_SCHEMA_ID

    def __post_init__(self) -> None:
        for name in (
            "model_sha256",
            "options_sha256",
            "prior_sha256",
            "selection_rule_sha256",
            "nuisance_scope_sha256",
            "observability_rule_sha256",
        ):
            object.__setattr__(self, name, _sha256_text(name, getattr(self, name)))
        object.__setattr__(
            self,
            "fitted_spatial_dimension",
            _index("fitted_spatial_dimension", self.fitted_spatial_dimension),
        )
        if self.schema_id != _PUBLIC_SCHEMA_ID:
            raise ValueError(f"schema_id must equal {_PUBLIC_SCHEMA_ID!r}")


@dataclass(frozen=True, eq=False)
class AtomisticEditBlindPublicProblem1D:
    """The only value passed through the reconstruction callback boundary.

    ``public_arrays`` may contain nominal geometry and prepared numerical
    operands.  Keys carrying case labels, seeds, truth, generating chemistry,
    defect coordinates, or mismatch causes are rejected.  ``public_scalars``
    accepts finite numbers and Booleans only, so arbitrary descriptive strings
    cannot become a covert object-metadata channel.
    """

    selection_observed_total_electrons: Array
    selection_valid_mask: Array
    audit_prediction_shape: tuple[int, ...]
    contract: AtomisticEditReconstructionContract1D
    public_arrays: Mapping[str, Array] = field(default_factory=dict)
    public_scalars: Mapping[str, float | int | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        observed = _readonly_array(
            self.selection_observed_total_electrons, dtype=np.float64
        )
        valid = _readonly_array(self.selection_valid_mask, dtype=bool)
        if observed.ndim < 2:
            raise ValueError(
                "selection_observed_total_electrons must have a leading scan axis"
            )
        if valid.shape != observed.shape:
            raise ValueError(
                "selection_valid_mask must match selection_observed_total_electrons"
            )
        if not np.all(np.isfinite(observed)) or np.any(observed < 0.0):
            raise ValueError(
                "selection_observed_total_electrons must be finite and non-negative"
            )
        if not np.any(valid):
            raise ValueError("the selection-visible split has no valid counts")
        audit_shape = tuple(
            _index("audit_prediction_shape", item) for item in self.audit_prediction_shape
        )
        if len(audit_shape) < 2:
            raise ValueError("audit_prediction_shape must include scan and detector axes")
        if not isinstance(self.contract, AtomisticEditReconstructionContract1D):
            raise TypeError("contract must be AtomisticEditReconstructionContract1D")

        arrays: dict[str, np.ndarray] = {}
        for raw_key, raw_value in sorted(self.public_arrays.items()):
            key = _check_public_key(raw_key)
            value = _readonly_array(raw_value)
            if value.dtype.kind not in "biufc":
                raise TypeError(f"public_arrays[{key!r}] must be numeric or Boolean")
            if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                raise ValueError(f"public_arrays[{key!r}] must be finite")
            arrays[key] = value

        scalars: dict[str, float | int | bool] = {}
        for raw_key, raw_value in sorted(self.public_scalars.items()):
            key = _check_public_key(raw_key)
            if isinstance(raw_value, (bool, np.bool_)):
                scalars[key] = bool(raw_value)
            elif isinstance(raw_value, (int, np.integer)):
                scalars[key] = int(raw_value)
            elif isinstance(raw_value, (float, np.floating)):
                scalars[key] = _finite(f"public_scalars[{key!r}]", raw_value)
            else:
                raise TypeError(
                    f"public_scalars[{key!r}] must be a number or Boolean"
                )

        object.__setattr__(self, "selection_observed_total_electrons", observed)
        object.__setattr__(self, "selection_valid_mask", valid)
        object.__setattr__(self, "audit_prediction_shape", audit_shape)
        object.__setattr__(self, "public_arrays", MappingProxyType(arrays))
        object.__setattr__(self, "public_scalars", MappingProxyType(scalars))


def validate_atomistic_edit_blind_public_problem_1d(
    problem: AtomisticEditBlindPublicProblem1D,
) -> None:
    """Re-run the public-boundary validation without evaluating private truth."""

    if not isinstance(problem, AtomisticEditBlindPublicProblem1D):
        raise TypeError("problem must be AtomisticEditBlindPublicProblem1D")
    # Frozen dataclasses can still be forged through object.__setattr__.
    AtomisticEditBlindPublicProblem1D(
        selection_observed_total_electrons=problem.selection_observed_total_electrons,
        selection_valid_mask=problem.selection_valid_mask,
        audit_prediction_shape=problem.audit_prediction_shape,
        contract=problem.contract,
        public_arrays=problem.public_arrays,
        public_scalars=problem.public_scalars,
    )


def atomistic_edit_public_problem_digest_1d(
    problem: AtomisticEditBlindPublicProblem1D,
) -> str:
    """Hash exactly the callback-visible values, never case or truth metadata."""

    validate_atomistic_edit_blind_public_problem_1d(problem)
    hasher = hashlib.sha256()
    hasher.update(problem.contract.schema_id.encode("utf-8"))
    for name in (
        "model_sha256",
        "options_sha256",
        "prior_sha256",
        "selection_rule_sha256",
        "nuisance_scope_sha256",
        "observability_rule_sha256",
    ):
        hasher.update(getattr(problem.contract, name).encode("ascii"))
    hasher.update(str(problem.contract.fitted_spatial_dimension).encode("ascii"))
    _hash_array(
        hasher,
        "selection_observed_total_electrons",
        problem.selection_observed_total_electrons,
    )
    _hash_array(hasher, "selection_valid_mask", problem.selection_valid_mask)
    hasher.update(json.dumps(problem.audit_prediction_shape).encode("ascii"))
    for key, value in problem.public_arrays.items():
        _hash_array(hasher, f"public_arrays:{key}", value)
    hasher.update(
        json.dumps(dict(problem.public_scalars), sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    )
    return hasher.hexdigest()


def atomistic_edit_public_problem_schema_digest_1d(
    problem: AtomisticEditBlindPublicProblem1D,
) -> str:
    """Hash callback-visible names, shapes and dtypes but not numeric values."""

    validate_atomistic_edit_blind_public_problem_1d(problem)
    schema = {
        "schema_id": problem.contract.schema_id,
        "contract": {
            "fitted_spatial_dimension": problem.contract.fitted_spatial_dimension,
            # These identities must be identical across cases, not merely typed alike.
            "model_sha256": problem.contract.model_sha256,
            "options_sha256": problem.contract.options_sha256,
            "prior_sha256": problem.contract.prior_sha256,
            "selection_rule_sha256": problem.contract.selection_rule_sha256,
            "nuisance_scope_sha256": problem.contract.nuisance_scope_sha256,
            "observability_rule_sha256": problem.contract.observability_rule_sha256,
        },
        "observed": {
            "shape": np.asarray(problem.selection_observed_total_electrons).shape,
            "dtype": str(np.asarray(problem.selection_observed_total_electrons).dtype),
        },
        "valid": {
            "shape": np.asarray(problem.selection_valid_mask).shape,
            "dtype": str(np.asarray(problem.selection_valid_mask).dtype),
        },
        "audit_prediction_shape": problem.audit_prediction_shape,
        "arrays": {
            key: {"shape": value.shape, "dtype": str(value.dtype)}
            for key, value in problem.public_arrays.items()
        },
        "scalars": {
            key: type(value).__name__ for key, value in problem.public_scalars.items()
        },
    }
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, eq=False)
class ResolutionAwareMassMeasure1D:
    """A finite positive measure in host-equivalent scattering units."""

    positions_A: Array
    masses_host_equivalent: Array

    def __post_init__(self) -> None:
        positions = _readonly_array(self.positions_A, dtype=np.float64)
        masses = _readonly_array(self.masses_host_equivalent, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] < 1:
            raise ValueError("positions_A must have shape (centres, dimension)")
        if masses.shape != (positions.shape[0],):
            raise ValueError("masses_host_equivalent must have one value per centre")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions_A must be finite")
        if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
            raise ValueError("active masses_host_equivalent must be finite and positive")
        object.__setattr__(self, "positions_A", positions)
        object.__setattr__(self, "masses_host_equivalent", masses)

    @classmethod
    def empty(cls, dimension: int = 2) -> "ResolutionAwareMassMeasure1D":
        dimension = _index("dimension", dimension)
        return cls(np.empty((0, dimension)), np.empty((0,)))

    @property
    def dimension(self) -> int:
        return int(np.asarray(self.positions_A).shape[1])

    @property
    def centre_count(self) -> int:
        return int(np.asarray(self.positions_A).shape[0])

    @property
    def total_mass(self) -> float:
        return float(np.sum(np.asarray(self.masses_host_equivalent)))


@dataclass(frozen=True)
class AtomisticEditBlindPrivateTruth1D:
    """Private audit truth, returned lazily only after reconstruction finishes."""

    additions: ResolutionAwareMassMeasure1D
    removals: ResolutionAwareMassMeasure1D
    host_deformation_rms_A: float = 0.0
    host_kernel_id: str = "host_kernel"
    generating_addition_kernel_id: str | None = None
    generating_element: str | None = None
    mismatch_cause: str | None = None
    axial_depth_uncertainty_A: float | None = None
    slice_thickness_A: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.additions, ResolutionAwareMassMeasure1D):
            raise TypeError("additions must be ResolutionAwareMassMeasure1D")
        if not isinstance(self.removals, ResolutionAwareMassMeasure1D):
            raise TypeError("removals must be ResolutionAwareMassMeasure1D")
        if self.additions.dimension != self.removals.dimension:
            raise ValueError("addition and removal truth dimensions must agree")
        object.__setattr__(
            self,
            "host_deformation_rms_A",
            _finite(
                "host_deformation_rms_A",
                self.host_deformation_rms_A,
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "host_kernel_id",
            _identifier("host_kernel_id", self.host_kernel_id),
        )
        if self.generating_addition_kernel_id is not None:
            object.__setattr__(
                self,
                "generating_addition_kernel_id",
                _identifier(
                    "generating_addition_kernel_id",
                    self.generating_addition_kernel_id,
                ),
            )
        if self.generating_element is not None:
            object.__setattr__(
                self,
                "generating_element",
                _identifier("generating_element", self.generating_element),
            )
        if self.mismatch_cause is not None:
            object.__setattr__(
                self,
                "mismatch_cause",
                _identifier("mismatch_cause", self.mismatch_cause),
            )
        for name in ("axial_depth_uncertainty_A", "slice_thickness_A"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive(name, value))


@dataclass(frozen=True, eq=False)
class AtomisticEditBlindAuditCounts1D:
    """Private held-out counts and validity, evaluated after reconstruction."""

    observed_total_electrons: Array
    valid_mask: Array

    def __post_init__(self) -> None:
        observed = _readonly_array(self.observed_total_electrons, dtype=np.float64)
        valid = _readonly_array(self.valid_mask, dtype=bool)
        if observed.ndim < 2:
            raise ValueError("private audit counts must include scan and detector axes")
        if valid.shape != observed.shape:
            raise ValueError("private audit valid_mask must match its counts")
        if not np.all(np.isfinite(observed)) or np.any(observed < 0.0):
            raise ValueError("private audit counts must be finite and non-negative")
        if not np.any(valid):
            raise ValueError("private audit split has no valid count observations")
        object.__setattr__(self, "observed_total_electrons", observed)
        object.__setattr__(self, "valid_mask", valid)


@dataclass(frozen=True)
class AtomisticEditBlindCase1D:
    """A public problem paired with private role metadata and lazy truth."""

    role: AtomisticEditBlindCaseRole1D
    private_case_label: str
    public_problem: AtomisticEditBlindPublicProblem1D
    private_audit_factory: Callable[[], AtomisticEditBlindAuditCounts1D]
    private_truth_factory: Callable[[], AtomisticEditBlindPrivateTruth1D]

    def __post_init__(self) -> None:
        if not isinstance(self.role, AtomisticEditBlindCaseRole1D):
            raise TypeError("role must be AtomisticEditBlindCaseRole1D")
        object.__setattr__(
            self,
            "private_case_label",
            _identifier("private_case_label", self.private_case_label),
        )
        validate_atomistic_edit_blind_public_problem_1d(self.public_problem)
        if not callable(self.private_audit_factory):
            raise TypeError("private_audit_factory must be callable")
        if not callable(self.private_truth_factory):
            raise TypeError("private_truth_factory must be callable")


@dataclass(frozen=True)
class ObservabilityEvidence1D:
    """Acquisition-bound spatial resolution and depth-reporting evidence."""

    observability_rule_sha256: str
    resolution_A: tuple[float, ...]
    depth_axis: int | None = None
    depth_response_A: float | None = None
    reported_minimum_axial_feature_width_A: float | None = None
    reported_axial_uncertainty_A: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observability_rule_sha256",
            _sha256_text("observability_rule_sha256", self.observability_rule_sha256),
        )
        resolution = tuple(_positive("resolution_A", item) for item in self.resolution_A)
        if not resolution:
            raise ValueError("resolution_A must be non-empty")
        object.__setattr__(self, "resolution_A", resolution)
        if self.depth_axis is not None:
            axis = _index("depth_axis", self.depth_axis, allow_zero=True)
            if axis >= len(resolution):
                raise ValueError("depth_axis is outside resolution_A")
            object.__setattr__(self, "depth_axis", axis)
        for name in (
            "depth_response_A",
            "reported_minimum_axial_feature_width_A",
            "reported_axial_uncertainty_A",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive(name, value))


@dataclass(frozen=True)
class NuisanceAttributionEvidence1D:
    """Truth-free profile comparison for the declared small nuisance block."""

    nuisance_scope_sha256: str
    profiled_parameter_count: int
    profiled_no_edit_held_out_deviance_per_pixel: float
    structural_edit_held_out_deviance_per_pixel: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "nuisance_scope_sha256",
            _sha256_text("nuisance_scope_sha256", self.nuisance_scope_sha256),
        )
        object.__setattr__(
            self,
            "profiled_parameter_count",
            _index("profiled_parameter_count", self.profiled_parameter_count),
        )
        for name in (
            "profiled_no_edit_held_out_deviance_per_pixel",
            "structural_edit_held_out_deviance_per_pixel",
        ):
            object.__setattr__(
                self, name, _finite(name, getattr(self, name), nonnegative=True)
            )


class _AmbiguityDisposition1D(str, Enum):
    IDENTIFIABLE = "identifiable"
    AMBIGUOUS = "ambiguous"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True)
class ActiveEditMultistartEvidence1D:
    """Numerical per-start quantities used to derive agreement or ambiguity."""

    validation_count_deviances: tuple[float, ...]
    total_addition_masses: tuple[float, ...]
    total_removal_masses: tuple[float, ...]
    support_distance_to_medoid_resolution_units: tuple[float, ...]
    selected_start_index: int
    ambiguity_disposition: str

    def __post_init__(self) -> None:
        count = len(self.validation_count_deviances)
        if count < 1:
            raise ValueError("multistart evidence must contain at least one start")
        for name in (
            "total_addition_masses",
            "total_removal_masses",
            "support_distance_to_medoid_resolution_units",
        ):
            if len(getattr(self, name)) != count:
                raise ValueError(f"{name} must have one value per start")
        for name in (
            "validation_count_deviances",
            "total_addition_masses",
            "total_removal_masses",
            "support_distance_to_medoid_resolution_units",
        ):
            values = tuple(
                _finite(name, item, nonnegative=True) for item in getattr(self, name)
            )
            object.__setattr__(self, name, values)
        selected = _index("selected_start_index", self.selected_start_index, allow_zero=True)
        if selected >= count:
            raise ValueError("selected_start_index is outside the starts")
        object.__setattr__(self, "selected_start_index", selected)
        try:
            disposition = _AmbiguityDisposition1D(self.ambiguity_disposition)
        except ValueError as error:
            raise ValueError(
                "ambiguity_disposition must be identifiable, ambiguous, or not_assessed"
            ) from error
        object.__setattr__(self, "ambiguity_disposition", disposition.value)


@dataclass(frozen=True)
class PhysicalAdmissibilityMetrics1D:
    """Measured overlap and high-frequency host-distortion diagnostics."""

    hard_core_overlap_mass: float
    host_deformation_roughness: float

    def __post_init__(self) -> None:
        for name in ("hard_core_overlap_mass", "host_deformation_roughness"):
            object.__setattr__(
                self, name, _finite(name, getattr(self, name), nonnegative=True)
            )


@dataclass(frozen=True)
class ReconstructionArchiveEvidence1D:
    """Numeric replay errors from the non-pickled reconstruction archive."""

    archive_sha256: str
    rerender_max_abs_error: float
    objective_component_max_abs_error: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "archive_sha256", _sha256_text("archive_sha256", self.archive_sha256)
        )
        for name in ("rerender_max_abs_error", "objective_component_max_abs_error"):
            object.__setattr__(
                self, name, _finite(name, getattr(self, name), nonnegative=True)
            )


@dataclass(frozen=True, eq=False)
class AtomisticEditBlindReconstruction1D:
    """Truth-free reconstruction output consumed by the benchmark assessor."""

    predicted_selection_total_electrons: Array
    predicted_audit_total_electrons: Array
    additions: ResolutionAwareMassMeasure1D
    removals: ResolutionAwareMassMeasure1D
    deformation_parameter_count: int
    fitted_spatial_dimension: int
    maximum_dormant_kkt_violation: float
    recovered_host_deformation_rms_A: float
    multistart: ActiveEditMultistartEvidence1D
    physical_metrics: PhysicalAdmissibilityMetrics1D
    observability: ObservabilityEvidence1D | None = None
    nuisance_attribution: NuisanceAttributionEvidence1D | None = None
    archive_evidence: ReconstructionArchiveEvidence1D | None = None

    def __post_init__(self) -> None:
        predicted_selection = _readonly_array(
            self.predicted_selection_total_electrons, dtype=np.float64
        )
        predicted_audit = _readonly_array(
            self.predicted_audit_total_electrons, dtype=np.float64
        )
        for name, predicted in (
            ("predicted_selection_total_electrons", predicted_selection),
            ("predicted_audit_total_electrons", predicted_audit),
        ):
            if not np.all(np.isfinite(predicted)) or np.any(predicted < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, predicted)
        if not isinstance(self.additions, ResolutionAwareMassMeasure1D):
            raise TypeError("additions must be ResolutionAwareMassMeasure1D")
        if not isinstance(self.removals, ResolutionAwareMassMeasure1D):
            raise TypeError("removals must be ResolutionAwareMassMeasure1D")
        fitted_dimension = _index(
            "fitted_spatial_dimension", self.fitted_spatial_dimension
        )
        if (
            self.additions.dimension != fitted_dimension
            or self.removals.dimension != fitted_dimension
        ):
            raise ValueError("edit measure dimensions must equal fitted_spatial_dimension")
        object.__setattr__(self, "fitted_spatial_dimension", fitted_dimension)
        object.__setattr__(
            self,
            "deformation_parameter_count",
            _index(
                "deformation_parameter_count",
                self.deformation_parameter_count,
                allow_zero=True,
            ),
        )
        for name in (
            "maximum_dormant_kkt_violation",
            "recovered_host_deformation_rms_A",
        ):
            object.__setattr__(
                self, name, _finite(name, getattr(self, name), nonnegative=True)
            )
        if not isinstance(self.multistart, ActiveEditMultistartEvidence1D):
            raise TypeError("multistart must be ActiveEditMultistartEvidence1D")
        if not isinstance(self.physical_metrics, PhysicalAdmissibilityMetrics1D):
            raise TypeError("physical_metrics must be PhysicalAdmissibilityMetrics1D")
        if self.observability is not None and not isinstance(
            self.observability, ObservabilityEvidence1D
        ):
            raise TypeError("observability must be ObservabilityEvidence1D or None")
        if self.nuisance_attribution is not None and not isinstance(
            self.nuisance_attribution, NuisanceAttributionEvidence1D
        ):
            raise TypeError(
                "nuisance_attribution must be NuisanceAttributionEvidence1D or None"
            )
        if self.archive_evidence is not None and not isinstance(
            self.archive_evidence, ReconstructionArchiveEvidence1D
        ):
            raise TypeError(
                "archive_evidence must be ReconstructionArchiveEvidence1D or None"
            )

    @property
    def active_parameter_count(self) -> int:
        """Exact active count: P_deformation + K_minus + (D + 1) K_plus."""

        return int(
            self.deformation_parameter_count
            + self.removals.centre_count
            + (self.fitted_spatial_dimension + 1) * self.additions.centre_count
        )


@dataclass(frozen=True)
class HeldOutCountMetrics1D:
    valid_count: int
    poisson_deviance: float
    poisson_deviance_per_valid_pixel: float
    root_mean_square_error_electrons: float
    mean_absolute_error_electrons: float
    observed_total_electrons: float
    predicted_total_electrons: float


@dataclass(frozen=True)
class ResolutionAwareMassTransportMetrics1D:
    truth_total_mass: float
    estimate_total_mass: float
    matched_mass: float
    unmatched_truth_mass: float
    unmatched_estimate_mass: float
    relative_total_mass_error: float
    normalized_transport_cost: float
    resolution_normalized_rms_displacement: float | None


def resolution_aware_mass_transport_metrics_1d(
    truth: ResolutionAwareMassMeasure1D,
    estimate: ResolutionAwareMassMeasure1D,
    resolution_A: Sequence[float],
) -> ResolutionAwareMassTransportMetrics1D:
    """Compute symmetric unbalanced transport in resolution-normalized space.

    Moving one mass unit costs its Euclidean displacement in resolution units;
    deleting or creating one unit costs one.  Consequently, transport beyond
    two resolution units is naturally replaced by one deletion and one birth.
    The finite linear program is used only for audit metrics, not inference.
    """

    if not isinstance(truth, ResolutionAwareMassMeasure1D) or not isinstance(
        estimate, ResolutionAwareMassMeasure1D
    ):
        raise TypeError("truth and estimate must be ResolutionAwareMassMeasure1D")
    if truth.dimension != estimate.dimension:
        raise ValueError("truth and estimate dimensions must agree")
    resolution = np.asarray(tuple(resolution_A), dtype=np.float64)
    if (
        resolution.shape != (truth.dimension,)
        or not np.all(np.isfinite(resolution))
        or np.any(resolution <= 0.0)
    ):
        raise ValueError("resolution_A must contain one positive value per dimension")

    truth_mass = np.asarray(truth.masses_host_equivalent, dtype=np.float64)
    estimate_mass = np.asarray(estimate.masses_host_equivalent, dtype=np.float64)
    truth_total = float(np.sum(truth_mass))
    estimate_total = float(np.sum(estimate_mass))
    scale = max(truth_total, estimate_total, np.finfo(np.float64).eps)
    relative_mass_error = abs(estimate_total - truth_total) / max(
        truth_total, np.finfo(np.float64).eps
    )

    if truth.centre_count == 0 and estimate.centre_count == 0:
        return ResolutionAwareMassTransportMetrics1D(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None
        )
    if truth.centre_count == 0:
        return ResolutionAwareMassTransportMetrics1D(
            0.0,
            estimate_total,
            0.0,
            0.0,
            estimate_total,
            float("inf"),
            1.0,
            None,
        )
    if estimate.centre_count == 0:
        return ResolutionAwareMassTransportMetrics1D(
            truth_total,
            0.0,
            0.0,
            truth_total,
            0.0,
            1.0,
            1.0,
            None,
        )

    from scipy.optimize import linprog

    delta = (
        np.asarray(truth.positions_A)[:, None, :]
        - np.asarray(estimate.positions_A)[None, :, :]
    ) / resolution[None, None, :]
    distance = np.linalg.norm(delta, axis=-1)
    n_truth, n_estimate = distance.shape
    transport_count = n_truth * n_estimate
    objective = np.concatenate(
        (distance.ravel(), np.ones(n_truth), np.ones(n_estimate))
    )
    equality = np.zeros((n_truth + n_estimate, objective.size), dtype=np.float64)
    for i in range(n_truth):
        equality[i, i * n_estimate : (i + 1) * n_estimate] = 1.0
        equality[i, transport_count + i] = 1.0
    for j in range(n_estimate):
        equality[n_truth + j, j:transport_count:n_estimate] = 1.0
        equality[n_truth + j, transport_count + n_truth + j] = 1.0
    rhs = np.concatenate((truth_mass, estimate_mass))
    solution = linprog(
        objective,
        A_eq=equality,
        b_eq=rhs,
        bounds=(0.0, None),
        method="highs",
    )
    if not solution.success:
        raise RuntimeError(f"mass-transport audit failed: {solution.message}")
    transport = solution.x[:transport_count].reshape(n_truth, n_estimate)
    matched = float(np.sum(transport))
    unmatched_truth = float(np.sum(solution.x[transport_count : transport_count + n_truth]))
    unmatched_estimate = float(np.sum(solution.x[transport_count + n_truth :]))
    normalized_cost = float(solution.fun / scale)
    rms = (
        float(np.sqrt(np.sum(transport * distance**2) / matched))
        if matched > np.finfo(np.float64).eps
        else None
    )
    return ResolutionAwareMassTransportMetrics1D(
        truth_total_mass=truth_total,
        estimate_total_mass=estimate_total,
        matched_mass=matched,
        unmatched_truth_mass=unmatched_truth,
        unmatched_estimate_mass=unmatched_estimate,
        relative_total_mass_error=float(relative_mass_error),
        normalized_transport_cost=normalized_cost,
        resolution_normalized_rms_displacement=rms,
    )


@dataclass(frozen=True)
class AtomisticEditBlindAcceptancePolicy1D:
    """Frozen, sourced thresholds used to derive every acceptance gate."""

    threshold_source: str
    maximum_held_out_deviance_per_valid_pixel: float = 5.0
    empty_edit_mass_tolerance: float = 1e-3
    maximum_normalized_transport_cost: float = 0.5
    maximum_relative_mass_error: float = 0.25
    maximum_dormant_kkt_violation: float = 1e-5
    minimum_multistart_count: int = 2
    validation_equivalence_relative_tolerance: float = 0.01
    validation_equivalence_absolute_tolerance: float = 1e-10
    maximum_equivalent_support_distance_resolution_units: float = 1.0
    maximum_equivalent_mass_spread: float = 0.25
    dense_active_parameter_limit: int = 6_040
    archive_rerender_tolerance: float = 1e-10
    archive_objective_tolerance: float = 1e-10
    maximum_level1_relative_count_degradation: float = 0.02
    maximum_level1_absolute_count_degradation: float = 1e-8
    minimum_physical_metric_improvement: float = 1e-8
    nuisance_profile_relative_tolerance: float = 0.01
    nuisance_profile_absolute_tolerance: float = 1e-8
    maximum_metastable_relative_deformation_error: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "threshold_source", _identifier("threshold_source", self.threshold_source)
        )
        for name in (
            "maximum_held_out_deviance_per_valid_pixel",
            "empty_edit_mass_tolerance",
            "maximum_normalized_transport_cost",
            "maximum_relative_mass_error",
            "maximum_dormant_kkt_violation",
            "validation_equivalence_relative_tolerance",
            "validation_equivalence_absolute_tolerance",
            "maximum_equivalent_support_distance_resolution_units",
            "maximum_equivalent_mass_spread",
            "archive_rerender_tolerance",
            "archive_objective_tolerance",
            "maximum_level1_relative_count_degradation",
            "maximum_level1_absolute_count_degradation",
            "minimum_physical_metric_improvement",
            "nuisance_profile_relative_tolerance",
            "nuisance_profile_absolute_tolerance",
            "maximum_metastable_relative_deformation_error",
        ):
            object.__setattr__(
                self, name, _finite(name, getattr(self, name), nonnegative=True)
            )
        object.__setattr__(
            self,
            "minimum_multistart_count",
            _index("minimum_multistart_count", self.minimum_multistart_count),
        )
        limit = _index("dense_active_parameter_limit", self.dense_active_parameter_limit)
        if limit != 6_040:
            raise ValueError("dense_active_parameter_limit is frozen at 6040 for AE-3 v1")
        object.__setattr__(self, "dense_active_parameter_limit", limit)


@dataclass(frozen=True)
class DerivedAtomisticEditGate1D:
    """A measured interval gate; :attr:`passed` is derived, never stored."""

    gate_id: str
    threshold_source: str
    measured_value: float | None
    lower_bound: float | None = None
    upper_bound: float | None = None
    evidence_kind: str = "numeric"
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_id", _identifier("gate_id", self.gate_id))
        object.__setattr__(
            self, "threshold_source", _identifier("threshold_source", self.threshold_source)
        )
        if self.measured_value is not None:
            value = float(self.measured_value)
            if math.isnan(value):
                raise ValueError("measured_value must not be NaN")
            object.__setattr__(self, "measured_value", value)
        for name in ("lower_bound", "upper_bound"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(name, value))
        if self.lower_bound is None and self.upper_bound is None:
            raise ValueError("a derived gate needs at least one numeric bound")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise ValueError("lower_bound must not exceed upper_bound")
        object.__setattr__(self, "evidence_kind", _identifier("evidence_kind", self.evidence_kind))

    @property
    def passed(self) -> bool:
        if self.measured_value is None or not np.isfinite(self.measured_value):
            return False
        if self.lower_bound is not None and self.measured_value < self.lower_bound:
            return False
        if self.upper_bound is not None and self.measured_value > self.upper_bound:
            return False
        return True


@dataclass(frozen=True, eq=False)
class AtomisticEditCaseAblationReport1D:
    case_role: AtomisticEditBlindCaseRole1D
    private_case_label: str
    ablation: AtomisticEditAblationArm1D
    status: AtomisticEditAblationStatus1D
    public_problem_sha256: str
    public_schema_sha256: str
    reconstruction: AtomisticEditBlindReconstruction1D | None
    held_out_count_metrics: HeldOutCountMetrics1D | None
    audit_truth_additions: ResolutionAwareMassMeasure1D | None
    audit_truth_removals: ResolutionAwareMassMeasure1D | None
    transport_resolution_A: tuple[float, ...] | None
    addition_transport: ResolutionAwareMassTransportMetrics1D | None
    removal_transport: ResolutionAwareMassTransportMetrics1D | None
    failure_stage: str | None = None
    diagnostic: str | None = None

    @property
    def active_parameter_count(self) -> int | None:
        if self.reconstruction is None:
            return None
        return self.reconstruction.active_parameter_count


@dataclass(frozen=True, eq=False)
class AtomisticEditBlindBenchmarkReport1D:
    """Complete eight-case/three-arm report with derived acceptance only."""

    case_reports: tuple[AtomisticEditCaseAblationReport1D, ...]
    gates: tuple[DerivedAtomisticEditGate1D, ...]
    policy: AtomisticEditBlindAcceptancePolicy1D
    public_schema_sha256: str
    report_schema_id: str = _REPORT_SCHEMA_ID

    @property
    def accepted(self) -> bool:
        return (
            self.report_schema_id == _REPORT_SCHEMA_ID
            and len(self.case_reports)
            == len(AE3_BLIND_CASE_CATALOG_1D) * len(AE3_ABLATION_CATALOG_1D)
            and bool(self.gates)
            and all(gate.passed for gate in self.gates)
        )

    @property
    def failed_gate_ids(self) -> tuple[str, ...]:
        return tuple(gate.gate_id for gate in self.gates if not gate.passed)


def _held_out_count_metrics(
    audit: AtomisticEditBlindAuditCounts1D,
    reconstruction: AtomisticEditBlindReconstruction1D,
) -> HeldOutCountMetrics1D:
    observed = np.asarray(audit.observed_total_electrons)
    predicted = np.asarray(reconstruction.predicted_audit_total_electrons)
    if predicted.shape != observed.shape:
        raise ValueError(
            "predicted_audit_total_electrons must match the private audit count shape"
        )
    valid = np.asarray(audit.valid_mask)
    y = observed[valid]
    mu = predicted[valid]
    if y.size == 0:
        raise ValueError("the held-out split has no valid observations")
    safe_mu = np.maximum(mu, np.finfo(np.float64).tiny)
    term = safe_mu - y
    positive = y > 0.0
    term[positive] += y[positive] * np.log(y[positive] / safe_mu[positive])
    deviance = float(2.0 * np.sum(term))
    residual = mu - y
    return HeldOutCountMetrics1D(
        valid_count=int(y.size),
        poisson_deviance=deviance,
        poisson_deviance_per_valid_pixel=deviance / float(y.size),
        root_mean_square_error_electrons=float(np.sqrt(np.mean(residual**2))),
        mean_absolute_error_electrons=float(np.mean(np.abs(residual))),
        observed_total_electrons=float(np.sum(y)),
        predicted_total_electrons=float(np.sum(mu)),
    )


def _validate_private_truth_for_role(
    role: AtomisticEditBlindCaseRole1D,
    truth: AtomisticEditBlindPrivateTruth1D,
) -> None:
    if not isinstance(truth, AtomisticEditBlindPrivateTruth1D):
        raise TypeError("private_truth_factory must return AtomisticEditBlindPrivateTruth1D")
    add_count = truth.additions.centre_count
    remove_count = truth.removals.centre_count
    if role is AtomisticEditBlindCaseRole1D.PRISTINE_HOST:
        if add_count or remove_count or truth.host_deformation_rms_A != 0.0:
            raise ValueError("pristine truth must contain no edits or deformation")
    elif role is AtomisticEditBlindCaseRole1D.ONE_VACANCY:
        if remove_count != 1 or add_count:
            raise ValueError("one-vacancy truth must contain exactly one removal")
    elif role is AtomisticEditBlindCaseRole1D.ONE_OFF_LATTICE_ADDITION:
        if add_count != 1 or remove_count:
            raise ValueError("off-lattice truth must contain exactly one addition")
    elif role is AtomisticEditBlindCaseRole1D.ONE_SUBSTITUTION:
        if add_count != 1 or remove_count != 1:
            raise ValueError("substitution truth must contain one addition and removal")
        if (
            truth.generating_addition_kernel_id is None
            or truth.generating_addition_kernel_id == truth.host_kernel_id
        ):
            raise ValueError("substitution truth must use a different generating kernel")
    elif role is AtomisticEditBlindCaseRole1D.IRREGULAR_FINITE_CLUSTER:
        if add_count < 2:
            raise ValueError("irregular-cluster truth must contain multiple additions")
    elif role is AtomisticEditBlindCaseRole1D.METASTABLE_DEFECT:
        if truth.host_deformation_rms_A <= 0.0:
            raise ValueError("metastable truth must contain a non-zero host deformation")
    elif role is AtomisticEditBlindCaseRole1D.NUISANCE_ONLY_MISMATCH:
        if add_count or remove_count or truth.host_deformation_rms_A != 0.0:
            raise ValueError("nuisance-only truth must contain no structural defect")
        if truth.mismatch_cause is None:
            raise ValueError("nuisance-only private truth must declare its mismatch cause")
    elif role is AtomisticEditBlindCaseRole1D.AXIALLY_UNRESOLVED_ADDITION:
        if add_count != 1:
            raise ValueError("axially unresolved truth must contain one addition")
        if truth.axial_depth_uncertainty_A is None or truth.slice_thickness_A is None:
            raise ValueError("axially unresolved truth needs depth uncertainty and slice thickness")
        if truth.axial_depth_uncertainty_A <= truth.slice_thickness_A:
            raise ValueError("truth depth uncertainty must exceed one slice")


def _clone_public_problem(
    problem: AtomisticEditBlindPublicProblem1D,
) -> AtomisticEditBlindPublicProblem1D:
    """Detach callback identity from the private case wrapper."""

    return AtomisticEditBlindPublicProblem1D(
        selection_observed_total_electrons=problem.selection_observed_total_electrons,
        selection_valid_mask=problem.selection_valid_mask,
        audit_prediction_shape=problem.audit_prediction_shape,
        contract=problem.contract,
        public_arrays=problem.public_arrays,
        public_scalars=problem.public_scalars,
    )


def _gate(
    gate_id: str,
    policy: AtomisticEditBlindAcceptancePolicy1D,
    measured: float | None,
    *,
    lower: float | None = None,
    upper: float | None = None,
    evidence_kind: str = "numeric",
    detail: str = "",
) -> DerivedAtomisticEditGate1D:
    return DerivedAtomisticEditGate1D(
        gate_id=gate_id,
        threshold_source=policy.threshold_source,
        measured_value=measured,
        lower_bound=lower,
        upper_bound=upper,
        evidence_kind=evidence_kind,
        detail=detail,
    )


def _multistart_gate_value(
    evidence: ActiveEditMultistartEvidence1D,
    policy: AtomisticEditBlindAcceptancePolicy1D,
) -> float:
    losses = np.asarray(evidence.validation_count_deviances)
    best = float(np.min(losses))
    equivalent = losses <= (
        best
        + policy.validation_equivalence_absolute_tolerance
        + policy.validation_equivalence_relative_tolerance * max(abs(best), 1.0)
    )
    indices = np.flatnonzero(equivalent)
    if indices.size < policy.minimum_multistart_count:
        return 0.0
    support_distance = np.asarray(
        evidence.support_distance_to_medoid_resolution_units
    )[indices]
    addition = np.asarray(evidence.total_addition_masses)[indices]
    removal = np.asarray(evidence.total_removal_masses)[indices]
    disagreement = bool(
        np.max(support_distance)
        > policy.maximum_equivalent_support_distance_resolution_units
        or np.ptp(addition) > policy.maximum_equivalent_mass_spread
        or np.ptp(removal) > policy.maximum_equivalent_mass_spread
    )
    if disagreement:
        return float(evidence.ambiguity_disposition == _AmbiguityDisposition1D.AMBIGUOUS.value)
    return float(
        evidence.ambiguity_disposition == _AmbiguityDisposition1D.IDENTIFIABLE.value
    )


def _case_gates(
    role: AtomisticEditBlindCaseRole1D,
    truth: AtomisticEditBlindPrivateTruth1D | None,
    report: AtomisticEditCaseAblationReport1D,
    problem: AtomisticEditBlindPublicProblem1D,
    policy: AtomisticEditBlindAcceptancePolicy1D,
) -> list[DerivedAtomisticEditGate1D]:
    prefix = f"{role.value}.level1"
    reconstruction = report.reconstruction
    metrics = report.held_out_count_metrics
    completed = report.status is AtomisticEditAblationStatus1D.COMPLETED
    gates = [
        _gate(
            f"{prefix}.execution",
            policy,
            1.0 if completed else None,
            lower=1.0,
            evidence_kind="callback_execution",
        ),
        _gate(
            f"{prefix}.held_out_count",
            policy,
            metrics.poisson_deviance_per_valid_pixel if metrics is not None else None,
            upper=policy.maximum_held_out_deviance_per_valid_pixel,
            evidence_kind="held_out_calibrated_counts",
        ),
        _gate(
            f"{prefix}.proposal_grid_kkt",
            policy,
            reconstruction.maximum_dormant_kkt_violation
            if reconstruction is not None
            else None,
            upper=policy.maximum_dormant_kkt_violation,
            evidence_kind="full_training_proposal_grid_kkt",
        ),
        _gate(
            f"{prefix}.active_parameter_count",
            policy,
            float(reconstruction.active_parameter_count)
            if reconstruction is not None
            else None,
            upper=float(policy.dense_active_parameter_limit - 1),
            evidence_kind="exact_active_count",
        ),
        _gate(
            f"{prefix}.multistart_or_ambiguity",
            policy,
            _multistart_gate_value(reconstruction.multistart, policy)
            if reconstruction is not None
            else None,
            lower=1.0,
            evidence_kind="validation_equivalent_multistart",
        ),
        _gate(
            f"{prefix}.archive_rerender",
            policy,
            reconstruction.archive_evidence.rerender_max_abs_error
            if reconstruction is not None and reconstruction.archive_evidence is not None
            else None,
            upper=policy.archive_rerender_tolerance,
            evidence_kind="archive_replay",
            detail="missing archive evidence fails closed",
        ),
        _gate(
            f"{prefix}.archive_objective_components",
            policy,
            reconstruction.archive_evidence.objective_component_max_abs_error
            if reconstruction is not None and reconstruction.archive_evidence is not None
            else None,
            upper=policy.archive_objective_tolerance,
            evidence_kind="archive_replay",
            detail="missing objective replay evidence fails closed",
        ),
    ]
    if reconstruction is None or truth is None:
        return gates

    edit_mass = reconstruction.additions.total_mass + reconstruction.removals.total_mass
    if role in (
        AtomisticEditBlindCaseRole1D.PRISTINE_HOST,
        AtomisticEditBlindCaseRole1D.NUISANCE_ONLY_MISMATCH,
    ):
        gates.append(
            _gate(
                f"{prefix}.empty_stable_edit_set",
                policy,
                edit_mass,
                upper=policy.empty_edit_mass_tolerance,
                evidence_kind="selected_edit_mass",
            )
        )

    structural_roles = {
        AtomisticEditBlindCaseRole1D.ONE_VACANCY,
        AtomisticEditBlindCaseRole1D.ONE_OFF_LATTICE_ADDITION,
        AtomisticEditBlindCaseRole1D.ONE_SUBSTITUTION,
        AtomisticEditBlindCaseRole1D.IRREGULAR_FINITE_CLUSTER,
        AtomisticEditBlindCaseRole1D.METASTABLE_DEFECT,
        AtomisticEditBlindCaseRole1D.AXIALLY_UNRESOLVED_ADDITION,
    }
    if role in structural_roles:
        observability = reconstruction.observability
        valid_observability = bool(
            observability is not None
            and observability.observability_rule_sha256
            == problem.contract.observability_rule_sha256
            and len(observability.resolution_A)
            == reconstruction.fitted_spatial_dimension
        )
        gates.append(
            _gate(
                f"{prefix}.observability_evidence",
                policy,
                1.0 if valid_observability else None,
                lower=1.0,
                evidence_kind="typed_acquisition_bound_observability",
                detail="missing or unbound observability evidence fails closed",
            )
        )
        if report.addition_transport is not None:
            gates.extend(
                (
                    _gate(
                        f"{prefix}.addition_transport",
                        policy,
                        report.addition_transport.normalized_transport_cost,
                        upper=policy.maximum_normalized_transport_cost,
                        evidence_kind="resolution_aware_unbalanced_mass_transport",
                    ),
                    _gate(
                        f"{prefix}.addition_mass",
                        policy,
                        report.addition_transport.relative_total_mass_error,
                        upper=policy.maximum_relative_mass_error,
                        evidence_kind="host_equivalent_integrated_scattering",
                    ),
                )
            )
        elif truth.additions.centre_count:
            gates.append(
                _gate(
                    f"{prefix}.addition_transport",
                    policy,
                    None,
                    upper=policy.maximum_normalized_transport_cost,
                    evidence_kind="resolution_aware_unbalanced_mass_transport",
                    detail="missing observability resolution prevents transport audit",
                )
            )
        if report.removal_transport is not None:
            gates.extend(
                (
                    _gate(
                        f"{prefix}.removal_transport",
                        policy,
                        report.removal_transport.normalized_transport_cost,
                        upper=policy.maximum_normalized_transport_cost,
                        evidence_kind="resolution_aware_unbalanced_mass_transport",
                    ),
                    _gate(
                        f"{prefix}.removal_mass",
                        policy,
                        report.removal_transport.relative_total_mass_error,
                        upper=policy.maximum_relative_mass_error,
                        evidence_kind="host_equivalent_integrated_scattering",
                    ),
                )
            )
        elif truth.removals.centre_count:
            gates.append(
                _gate(
                    f"{prefix}.removal_transport",
                    policy,
                    None,
                    upper=policy.maximum_normalized_transport_cost,
                    evidence_kind="resolution_aware_unbalanced_mass_transport",
                    detail="missing observability resolution prevents transport audit",
                )
            )

    if role is AtomisticEditBlindCaseRole1D.NUISANCE_ONLY_MISMATCH:
        evidence = reconstruction.nuisance_attribution
        nuisance_value: float | None = None
        if (
            evidence is not None
            and evidence.nuisance_scope_sha256 == problem.contract.nuisance_scope_sha256
        ):
            allowed = (
                evidence.structural_edit_held_out_deviance_per_pixel
                * (1.0 + policy.nuisance_profile_relative_tolerance)
                + policy.nuisance_profile_absolute_tolerance
            )
            nuisance_value = (
                evidence.profiled_no_edit_held_out_deviance_per_pixel - allowed
            )
        gates.append(
            _gate(
                f"{prefix}.nuisance_attribution",
                policy,
                nuisance_value,
                upper=0.0,
                evidence_kind="typed_profiled_nuisance",
                detail="missing or unbound nuisance evidence fails closed",
            )
        )

    if role is AtomisticEditBlindCaseRole1D.METASTABLE_DEFECT:
        denominator = max(truth.host_deformation_rms_A, np.finfo(np.float64).eps)
        relative_error = abs(
            reconstruction.recovered_host_deformation_rms_A
            - truth.host_deformation_rms_A
        ) / denominator
        gates.append(
            _gate(
                f"{prefix}.metastable_deformation_recovery",
                policy,
                relative_error,
                upper=policy.maximum_metastable_relative_deformation_error,
                evidence_kind="private_truth_after_reconstruction",
            )
        )

    if role is AtomisticEditBlindCaseRole1D.AXIALLY_UNRESOLVED_ADDITION:
        evidence = reconstruction.observability
        depth_value: float | None = None
        feature_value: float | None = None
        if (
            evidence is not None
            and evidence.depth_axis is not None
            and evidence.depth_response_A is not None
            and evidence.reported_axial_uncertainty_A is not None
        ):
            depth_value = (
                evidence.reported_axial_uncertainty_A - evidence.depth_response_A
            )
            if evidence.reported_minimum_axial_feature_width_A is not None:
                feature_value = (
                    evidence.reported_minimum_axial_feature_width_A
                    - evidence.depth_response_A
                )
        gates.extend(
            (
                _gate(
                    f"{prefix}.depth_uncertainty_reporting",
                    policy,
                    depth_value,
                    lower=0.0,
                    evidence_kind="independent_depth_response",
                    detail="missing depth-response evidence fails closed",
                ),
                _gate(
                    f"{prefix}.no_subresponse_axial_feature",
                    policy,
                    feature_value,
                    lower=0.0,
                    evidence_kind="independent_depth_response",
                    detail="missing reported feature width fails closed",
                ),
            )
        )
    return gates


def _validate_case_catalog(
    cases: Sequence[AtomisticEditBlindCase1D],
) -> tuple[AtomisticEditBlindCase1D, ...]:
    result = tuple(cases)
    if len(result) != len(AE3_BLIND_CASE_CATALOG_1D):
        raise ValueError("AE-3 requires exactly eight blind cases")
    if any(not isinstance(case, AtomisticEditBlindCase1D) for case in result):
        raise TypeError("cases must contain AtomisticEditBlindCase1D values")
    roles = tuple(case.role for case in result)
    if set(roles) != set(AE3_BLIND_CASE_CATALOG_1D) or len(set(roles)) != len(roles):
        raise ValueError("AE-3 cases must contain each required role exactly once")
    labels = tuple(case.private_case_label for case in result)
    if len(set(labels)) != len(labels):
        raise ValueError("private_case_label values must be unique")
    contracts = {case.public_problem.contract for case in result}
    if len(contracts) != 1:
        raise ValueError(
            "all AE-3 cases must use identical model, options, prior, nuisance, "
            "observability, and selection-rule contracts"
        )
    schema_digests = {
        atomistic_edit_public_problem_schema_digest_1d(case.public_problem)
        for case in result
    }
    if len(schema_digests) != 1:
        raise ValueError("all AE-3 callback inputs must have an identical public schema")
    return result


def run_atomistic_edit_blind_benchmarks_1d(
    cases: Sequence[AtomisticEditBlindCase1D],
    reconstruction_callbacks: Mapping[
        AtomisticEditAblationArm1D, BlindReconstructionCallback1D
    ],
    policy: AtomisticEditBlindAcceptancePolicy1D,
) -> AtomisticEditBlindBenchmarkReport1D:
    """Run both Level-1 arms blindly, then evaluate private truth and gates.

    The energy arm cannot be supplied in v1 and is always emitted as
    ``blocked_not_run``.  Calls are ordered by callback-visible problem digest,
    not private role, and receive detached public-problem instances.  Python
    cannot prevent a deliberately malicious callback from using global state,
    but this API provides it no object reference carrying private case data.
    """

    if not isinstance(policy, AtomisticEditBlindAcceptancePolicy1D):
        raise TypeError("policy must be AtomisticEditBlindAcceptancePolicy1D")
    cases_tuple = _validate_case_catalog(cases)
    callbacks = dict(reconstruction_callbacks)
    required = {
        AtomisticEditAblationArm1D.COUNT_AND_EDIT,
        AtomisticEditAblationArm1D.LEVEL1_PHYSICAL,
    }
    if set(callbacks) != required:
        raise ValueError(
            "v1 requires exactly count-and-edit and Level-1 callbacks; "
            "the energy arm is blocked"
        )
    if any(not callable(callback) for callback in callbacks.values()):
        raise TypeError("all reconstruction_callbacks must be callable")

    ordered = sorted(
        cases_tuple,
        key=lambda case: atomistic_edit_public_problem_digest_1d(case.public_problem),
    )
    callback_results: dict[
        tuple[AtomisticEditBlindCaseRole1D, AtomisticEditAblationArm1D],
        tuple[AtomisticEditBlindReconstruction1D | None, str | None],
    ] = {}
    # Phase one: neither private audit counts nor private truth are touched here.
    for ablation in (
        AtomisticEditAblationArm1D.COUNT_AND_EDIT,
        AtomisticEditAblationArm1D.LEVEL1_PHYSICAL,
    ):
        callback = callbacks[ablation]
        for case in ordered:
            try:
                output = callback(_clone_public_problem(case.public_problem))
                if not isinstance(output, AtomisticEditBlindReconstruction1D):
                    raise TypeError(
                        "reconstruction callback must return "
                        "AtomisticEditBlindReconstruction1D"
                    )
                if (
                    output.fitted_spatial_dimension
                    != case.public_problem.contract.fitted_spatial_dimension
                ):
                    raise ValueError(
                        "reconstruction fitted dimension disagrees with the public contract"
                    )
                if output.predicted_selection_total_electrons.shape != (
                    case.public_problem.selection_observed_total_electrons.shape
                ):
                    raise ValueError(
                        "predicted_selection_total_electrons must match the "
                        "selection-visible count shape"
                    )
                if output.predicted_audit_total_electrons.shape != (
                    case.public_problem.audit_prediction_shape
                ):
                    raise ValueError(
                        "predicted_audit_total_electrons must match the declared "
                        "audit prediction shape"
                    )
                callback_results[(case.role, ablation)] = (output, None)
            except Exception as error:  # report a failed blind case; do not reveal truth
                callback_results[(case.role, ablation)] = (
                    None,
                    f"{type(error).__name__}: {error}",
                )

    # Phase two: all inverse callbacks have returned before audit data or truth
    # are materialized.
    private_audit: dict[
        AtomisticEditBlindCaseRole1D,
        tuple[AtomisticEditBlindAuditCounts1D | None, str | None],
    ] = {}
    private_truth: dict[
        AtomisticEditBlindCaseRole1D,
        tuple[AtomisticEditBlindPrivateTruth1D | None, str | None],
    ] = {}
    for case in cases_tuple:
        try:
            audit = case.private_audit_factory()
            if not isinstance(audit, AtomisticEditBlindAuditCounts1D):
                raise TypeError(
                    "private_audit_factory must return AtomisticEditBlindAuditCounts1D"
                )
            if audit.observed_total_electrons.shape != (
                case.public_problem.audit_prediction_shape
            ):
                raise ValueError(
                    "private audit shape must match the public audit prediction shape"
                )
            private_audit[case.role] = (audit, None)
        except Exception as error:
            private_audit[case.role] = (
                None,
                f"{type(error).__name__}: {error}",
            )
        try:
            truth = case.private_truth_factory()
            _validate_private_truth_for_role(case.role, truth)
            private_truth[case.role] = (truth, None)
        except Exception as error:
            private_truth[case.role] = (None, f"{type(error).__name__}: {error}")

    reports: list[AtomisticEditCaseAblationReport1D] = []
    report_lookup: dict[
        tuple[AtomisticEditBlindCaseRole1D, AtomisticEditAblationArm1D],
        AtomisticEditCaseAblationReport1D,
    ] = {}
    for case in cases_tuple:
        problem_digest = atomistic_edit_public_problem_digest_1d(case.public_problem)
        schema_digest = atomistic_edit_public_problem_schema_digest_1d(case.public_problem)
        audit, audit_error = private_audit[case.role]
        truth, truth_error = private_truth[case.role]
        for ablation in (
            AtomisticEditAblationArm1D.COUNT_AND_EDIT,
            AtomisticEditAblationArm1D.LEVEL1_PHYSICAL,
        ):
            output, callback_error = callback_results[(case.role, ablation)]
            metrics: HeldOutCountMetrics1D | None = None
            addition_transport: ResolutionAwareMassTransportMetrics1D | None = None
            removal_transport: ResolutionAwareMassTransportMetrics1D | None = None
            failure_stage: str | None = None
            diagnostic: str | None = None
            if callback_error is not None:
                status = AtomisticEditAblationStatus1D.FAILED
                failure_stage = "reconstruction_callback"
                diagnostic = callback_error
            elif audit_error is not None:
                status = AtomisticEditAblationStatus1D.FAILED
                failure_stage = "private_held_out_audit"
                diagnostic = audit_error
            elif truth_error is not None:
                status = AtomisticEditAblationStatus1D.FAILED
                failure_stage = "private_truth_audit"
                diagnostic = truth_error
            else:
                assert output is not None and audit is not None and truth is not None
                try:
                    metrics = _held_out_count_metrics(audit, output)
                    observability = output.observability
                    if (
                        observability is not None
                        and observability.observability_rule_sha256
                        == case.public_problem.contract.observability_rule_sha256
                        and len(observability.resolution_A)
                        == output.fitted_spatial_dimension
                    ):
                        if truth.additions.centre_count or output.additions.centre_count:
                            addition_transport = resolution_aware_mass_transport_metrics_1d(
                                truth.additions,
                                output.additions,
                                observability.resolution_A,
                            )
                        if truth.removals.centre_count or output.removals.centre_count:
                            removal_transport = resolution_aware_mass_transport_metrics_1d(
                                truth.removals,
                                output.removals,
                                observability.resolution_A,
                            )
                    status = AtomisticEditAblationStatus1D.COMPLETED
                except Exception as error:
                    status = AtomisticEditAblationStatus1D.FAILED
                    failure_stage = "post_reconstruction_audit"
                    diagnostic = f"{type(error).__name__}: {error}"
            report = AtomisticEditCaseAblationReport1D(
                case_role=case.role,
                private_case_label=case.private_case_label,
                ablation=ablation,
                status=status,
                public_problem_sha256=problem_digest,
                public_schema_sha256=schema_digest,
                reconstruction=output,
                held_out_count_metrics=metrics,
                audit_truth_additions=truth.additions if truth is not None else None,
                audit_truth_removals=truth.removals if truth is not None else None,
                transport_resolution_A=(
                    output.observability.resolution_A
                    if output is not None
                    and output.observability is not None
                    and output.observability.observability_rule_sha256
                    == case.public_problem.contract.observability_rule_sha256
                    else None
                ),
                addition_transport=addition_transport,
                removal_transport=removal_transport,
                failure_stage=failure_stage,
                diagnostic=diagnostic,
            )
            reports.append(report)
            report_lookup[(case.role, ablation)] = report
        energy_report = AtomisticEditCaseAblationReport1D(
            case_role=case.role,
            private_case_label=case.private_case_label,
            ablation=AtomisticEditAblationArm1D.MATERIAL_ENERGY_ENVELOPE,
            status=AtomisticEditAblationStatus1D.BLOCKED_NOT_RUN,
            public_problem_sha256=problem_digest,
            public_schema_sha256=schema_digest,
            reconstruction=None,
            held_out_count_metrics=None,
            audit_truth_additions=truth.additions if truth is not None else None,
            audit_truth_removals=truth.removals if truth is not None else None,
            transport_resolution_A=None,
            addition_transport=None,
            removal_transport=None,
            failure_stage="chemistry_validation_gate",
            diagnostic=_ENERGY_BLOCK_REASON,
        )
        reports.append(energy_report)
        report_lookup[
            (case.role, AtomisticEditAblationArm1D.MATERIAL_ENERGY_ENVELOPE)
        ] = energy_report

    gates: list[DerivedAtomisticEditGate1D] = []
    for case in cases_tuple:
        truth, _ = private_truth[case.role]
        level1 = report_lookup[
            (case.role, AtomisticEditAblationArm1D.LEVEL1_PHYSICAL)
        ]
        gates.extend(
            _case_gates(case.role, truth, level1, case.public_problem, policy)
        )

    # Derive the physical-prior ablation from measured Level-0/Level-1 outputs.
    count_excesses: list[float] = []
    level0_physical = 0.0
    level1_physical = 0.0
    complete_pairs = True
    for role in AE3_BLIND_CASE_CATALOG_1D:
        baseline = report_lookup[(role, AtomisticEditAblationArm1D.COUNT_AND_EDIT)]
        physical = report_lookup[(role, AtomisticEditAblationArm1D.LEVEL1_PHYSICAL)]
        if (
            baseline.status is not AtomisticEditAblationStatus1D.COMPLETED
            or physical.status is not AtomisticEditAblationStatus1D.COMPLETED
            or baseline.held_out_count_metrics is None
            or physical.held_out_count_metrics is None
            or baseline.reconstruction is None
            or physical.reconstruction is None
        ):
            complete_pairs = False
            continue
        base_count = baseline.held_out_count_metrics.poisson_deviance_per_valid_pixel
        physical_count = physical.held_out_count_metrics.poisson_deviance_per_valid_pixel
        allowed = (
            base_count * (1.0 + policy.maximum_level1_relative_count_degradation)
            + policy.maximum_level1_absolute_count_degradation
        )
        count_excesses.append(physical_count - allowed)
        level0_physical += (
            baseline.reconstruction.physical_metrics.hard_core_overlap_mass
            + baseline.reconstruction.physical_metrics.host_deformation_roughness
        )
        level1_physical += (
            physical.reconstruction.physical_metrics.hard_core_overlap_mass
            + physical.reconstruction.physical_metrics.host_deformation_roughness
        )
    gates.extend(
        (
            _gate(
                "ablation.level1_held_out_count_non_degradation",
                policy,
                max(count_excesses) if complete_pairs and count_excesses else None,
                upper=0.0,
                evidence_kind="paired_held_out_count_deviance",
            ),
            _gate(
                "ablation.level1_physical_non_worsening",
                policy,
                level1_physical - level0_physical if complete_pairs else None,
                upper=0.0,
                evidence_kind="paired_overlap_and_deformation_metrics",
            ),
            _gate(
                "ablation.level1_physical_exclusion",
                policy,
                level0_physical - level1_physical if complete_pairs else None,
                lower=policy.minimum_physical_metric_improvement,
                evidence_kind="paired_overlap_and_deformation_metrics",
                detail="no measured physical improvement cannot validate the prior",
            ),
        )
    )

    schema_digest = atomistic_edit_public_problem_schema_digest_1d(
        cases_tuple[0].public_problem
    )
    result = AtomisticEditBlindBenchmarkReport1D(
        case_reports=tuple(reports),
        gates=tuple(gates),
        policy=policy,
        public_schema_sha256=schema_digest,
    )
    validate_atomistic_edit_blind_benchmark_report_1d(result)
    return result


def validate_atomistic_edit_blind_benchmark_report_1d(
    report: AtomisticEditBlindBenchmarkReport1D,
) -> None:
    """Validate catalog completeness and the fail-closed v1 energy status."""

    if not isinstance(report, AtomisticEditBlindBenchmarkReport1D):
        raise TypeError("report must be AtomisticEditBlindBenchmarkReport1D")
    if report.report_schema_id != _REPORT_SCHEMA_ID:
        raise ValueError("unknown atomistic-edit blind benchmark report schema")
    expected = {
        (role, ablation)
        for role in AE3_BLIND_CASE_CATALOG_1D
        for ablation in AE3_ABLATION_CATALOG_1D
    }
    actual = {(item.case_role, item.ablation) for item in report.case_reports}
    if actual != expected or len(report.case_reports) != len(expected):
        raise ValueError("report must contain every AE-3 case/ablation pair exactly once")
    schemas = {item.public_schema_sha256 for item in report.case_reports}
    if schemas != {report.public_schema_sha256}:
        raise ValueError("case reports do not share the declared public schema")
    for item in report.case_reports:
        if item.ablation is AtomisticEditAblationArm1D.MATERIAL_ENERGY_ENVELOPE:
            if (
                item.status is not AtomisticEditAblationStatus1D.BLOCKED_NOT_RUN
                or item.reconstruction is not None
                or item.failure_stage != "chemistry_validation_gate"
            ):
                raise ValueError("v1 material-energy ablation must be blocked_not_run")
    if not report.gates or len({gate.gate_id for gate in report.gates}) != len(
        report.gates
    ):
        raise ValueError("derived gate identifiers must be non-empty and unique")
