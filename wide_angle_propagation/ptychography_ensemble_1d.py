"""Truth-free multistart consensus diagnostics for lattice ptychography."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .ptychography_benchmarks_1d import (
    DetectorPerturbation1D,
    ForwardModelMismatch1D,
    ResidualCalibrationEvidence1D,
    SyntheticBenchmarkReport1D,
    validate_residual_calibration_evidence_1d,
    validate_synthetic_benchmark_report_1d,
)
from .ptychography_1d import (
    ConvergenceOptions1D,
    decompose_lattice_site_displacement_controls_1d,
    LatticeOptimizationOptions1D,
    LatticeSiteReconstruction1D,
    PreparedLatticeSiteReconstruction1D,
    run_prepared_lattice_site_reconstruction_1d,
)
from .ptychography_diagnostics_1d import LatticeSiteSensitivityScreen1D
from .ptychography_observability_1d import LatticeSiteObservability1D
from .ptychography_support_contract_1d import LatticeSiteRole1D


__all__ = [
    "EnsembleEvidenceProvenance1D",
    "EnsembleScanPartition1D",
    "LatticeSiteEnsemble1D",
    "LatticeSiteRunSummary1D",
    "MultistartOptions1D",
    "PreparedMultistartResult1D",
    "PreparedMultistartRunOptions1D",
    "SitewiseConsensus1D",
    "multistart_site_translation_offsets_1d",
    "load_lattice_site_ensemble_1d",
    "save_lattice_site_ensemble_1d",
    "run_prepared_lattice_site_multistart_1d",
    "summarize_lattice_site_ensemble_1d",
]


Array = Any


@dataclass(frozen=True)
class MultistartOptions1D:
    """Loss filtering and agreement criteria for independent optimizer starts."""

    n_starts: int = 8
    base_seed: int = 0
    initial_translation_half_width_A: tuple[float, float] = (0.15, 0.15)
    relative_loss_tolerance: float = 0.05
    absolute_loss_tolerance: float = 1e-8
    minimum_accepted_starts: int = 3
    minimum_accepted_fraction: float = 0.5
    vacancy_threshold: float = 0.5
    vacancy_margin: float = 0.1
    agreement_fraction: float = 0.8
    maximum_displacement_spread_A: float = 0.05
    maximum_rigid_spread_A: float = 0.05
    minimum_converged_fraction: float = 0.8
    maximum_bound_fraction: float = 0.05


@dataclass(frozen=True, eq=False)
class PreparedMultistartRunOptions1D:
    """Run controls for a reusable prepared multistart reconstruction.

    ``ensemble_options`` controls deterministic translations and consensus
    filtering.  The remaining fields are passed unchanged to every prepared
    optimizer run. Screening runs collect compact parameter checkpoints.
    After validation-only medoid selection, non-selected checkpoint histories
    are discarded while the selected screening trajectory is reused directly
    for visualization.
    """

    ensemble_options: MultistartOptions1D = field(
        default_factory=MultistartOptions1D
    )
    initial_vacancy_fractions: Array | None = None
    initial_displacement_controls: Array | None = None
    initial_rigid_displacement: Array | None = None
    learning_rate_start: float = 2e-2
    learning_rate_end: float = 2e-4
    updates: int = 500
    validation_interval: int = 25
    training_diagnostic_scan_count: int | None = None
    convergence: ConvergenceOptions1D | None = None
    optimization: LatticeOptimizationOptions1D | None = None
    representative_checkpoint_interval: int = 1
    progress: bool = False
    progress_description: str = "lattice-site multistart"


@dataclass(frozen=True, eq=False)
class PreparedMultistartResult1D:
    """Screening runs and their directly reused validation-selected medoid.

    The registration parameter moves only the candidate sites represented by
    the lattice model relative to the fixed reference potential.  It is not a
    global specimen/probe/detector registration estimate.
    """

    screening_results: tuple[LatticeSiteReconstruction1D, ...]
    ensemble: LatticeSiteEnsemble1D
    representative_result: LatticeSiteReconstruction1D
    initial_site_translations_A: Array
    seeds: Array
    options: PreparedMultistartRunOptions1D
    representative_trajectory_reused: bool
    registration_scope: str = "active_sites_relative_to_fixed_reference"

    @property
    def representative_screening_result(self) -> LatticeSiteReconstruction1D:
        """Return the actual screening run selected as the ensemble medoid."""
        return self.screening_results[self.ensemble.representative_index]


@dataclass(frozen=True)
class LatticeSiteRunSummary1D:
    """Compact parameter and numerical summary for one optimizer start."""

    # ``loss`` is always the pre-audit selection metric (normally validation
    # loss). Audit data must never change the accepted set or representative.
    loss: float
    converged: bool
    bound_fraction: float
    vacancy_fractions: Array
    residual_site_displacements: Array
    rigid_displacement: Array
    seed: int | None = None
    audit_loss: float = float("nan")


@dataclass(frozen=True)
class SitewiseConsensus1D:
    """Equal-weight intervals and calls across accepted low-loss starts.

    ``vacancy_state`` is ``1`` for vacancy, ``0`` for occupied, and ``-1`` for
    ambiguous. ``vacancy_call_frequency`` is an optimizer-start frequency, not
    a posterior probability.
    """

    vacancy_median: Array
    vacancy_q05: Array
    vacancy_q95: Array
    vacancy_call_frequency: Array
    vacancy_state: Array
    residual_displacement_median: Array
    residual_displacement_q05: Array
    residual_displacement_q95: Array
    residual_displacement_radial_q90_A: Array
    optimizer_agreement: Array
    sensitive: Array
    observable: Array
    site_trusted: Array


@dataclass(frozen=True)
class EnsembleScanPartition1D:
    """Ordered scan partition shared by every optimizer start."""

    n_scans: int
    training_indices: Array
    validation_indices: Array
    audit_indices: Array
    excluded_indices: Array


@dataclass(frozen=True)
class EnsembleEvidenceProvenance1D:
    """Provenance record for evidence used to make structural-trust calls.

    Compact ensemble archives do not contain the detector data or the full
    sensitivity, observability, residual-calibration, and mismatch reports.
    Consequently, an archive records which evidence was supplied to the live
    summary but never treats those claims as independently reverified on load.
    """

    source: str = "live_summary"
    sensitivity_screen_supplied: bool = False
    observability_report_count: int = 0
    observability_problem_ids_verified_at_summary: bool | None = None
    residual_calibration_evidence_supplied: bool = False
    residual_calibration_passed_at_summary: bool | None = None
    mismatch_benchmark_report_supplied: bool = False
    mismatch_benchmark_passed_at_summary: bool | None = None
    common_reconstruction_problem_id: str | None = None
    common_reconstructor_id: str | None = None
    mismatch_benchmark_id: str | None = None
    mismatch_generator_id: str | None = None
    mismatch_non_nominal_scenario_present_at_summary: bool | None = None
    mismatch_truth_structural_criterion_present_at_summary: bool | None = None
    mismatch_independent_forward_at_summary: bool | None = None
    structurally_trusted_at_summary: bool = False
    trusted_site_count_at_summary: int = 0
    typed_evidence_persisted: bool = False
    structural_trust_reverified_after_load: bool = False


@dataclass(frozen=True)
class LatticeSiteEnsemble1D:
    """Low-loss ensemble, a real medoid representative, and trust diagnostics."""

    runs: tuple[LatticeSiteRunSummary1D, ...]
    accepted_mask: Array
    accepted_loss_cutoff: float
    representative_index: int
    consensus: SitewiseConsensus1D
    rigid_median: Array
    rigid_q05: Array
    rigid_q95: Array
    rigid_radial_q90_A: float
    trust_flags: Mapping[str, bool | None] = field(default_factory=dict)
    optimizer_stable: bool = False
    structurally_trusted: bool = False
    site_coordinates: Array | None = None
    options: MultistartOptions1D | None = None
    scan_partition: EnsembleScanPartition1D | None = None
    evidence_provenance: EnsembleEvidenceProvenance1D = field(
        default_factory=EnsembleEvidenceProvenance1D
    )


def _validate_options(options: MultistartOptions1D) -> None:
    if not isinstance(options.n_starts, (int, np.integer)) or isinstance(
        options.n_starts, (bool, np.bool_)
    ):
        raise TypeError("n_starts must be an integer")
    if options.n_starts < 1:
        raise ValueError("n_starts must be positive")
    if not isinstance(options.base_seed, (int, np.integer)) or isinstance(
        options.base_seed, (bool, np.bool_)
    ):
        raise TypeError("base_seed must be an integer")
    if options.base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    if not isinstance(
        options.minimum_accepted_starts, (int, np.integer)
    ) or isinstance(options.minimum_accepted_starts, (bool, np.bool_)):
        raise TypeError("minimum_accepted_starts must be an integer")
    if options.minimum_accepted_starts < 1:
        raise ValueError("minimum_accepted_starts must be positive")
    if options.minimum_accepted_starts > options.n_starts:
        raise ValueError("minimum_accepted_starts cannot exceed n_starts")
    for name in (
        "relative_loss_tolerance",
        "absolute_loss_tolerance",
        "vacancy_margin",
        "maximum_displacement_spread_A",
        "maximum_rigid_spread_A",
        "maximum_bound_fraction",
    ):
        value = float(getattr(options, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    for name in (
        "minimum_accepted_fraction",
        "vacancy_threshold",
        "agreement_fraction",
        "minimum_converged_fraction",
    ):
        value = float(getattr(options, name))
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if options.vacancy_margin >= min(
        options.vacancy_threshold, 1.0 - options.vacancy_threshold
    ):
        raise ValueError("vacancy_margin leaves no occupied or vacancy interval")
    half_width = np.asarray(options.initial_translation_half_width_A, dtype=float)
    if (
        half_width.shape != (2,)
        or np.any(~np.isfinite(half_width))
        or np.any(half_width < 0.0)
    ):
        raise ValueError(
            "initial_translation_half_width_A must contain two non-negative values"
        )


def _positive_integer(name: str, value: Any) -> int:
    if not isinstance(value, (int, np.integer)) or isinstance(
        value, (bool, np.bool_)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _validated_prepared_run_options(
    options: PreparedMultistartRunOptions1D | None,
) -> PreparedMultistartRunOptions1D:
    options = PreparedMultistartRunOptions1D() if options is None else options
    if not isinstance(options, PreparedMultistartRunOptions1D):
        raise TypeError(
            "options must be a PreparedMultistartRunOptions1D instance or None"
        )
    if not isinstance(options.ensemble_options, MultistartOptions1D):
        raise TypeError("ensemble_options must be a MultistartOptions1D instance")
    _validate_options(options.ensemble_options)
    _positive_integer("updates", options.updates)
    _positive_integer("validation_interval", options.validation_interval)
    if options.training_diagnostic_scan_count is not None:
        _positive_integer(
            "training_diagnostic_scan_count",
            options.training_diagnostic_scan_count,
        )
    _positive_integer(
        "representative_checkpoint_interval",
        options.representative_checkpoint_interval,
    )
    learning_rate_start = float(options.learning_rate_start)
    learning_rate_end = float(options.learning_rate_end)
    if (
        not np.isfinite(learning_rate_start)
        or learning_rate_start <= 0.0
        or not np.isfinite(learning_rate_end)
        or learning_rate_end <= 0.0
    ):
        raise ValueError("learning rates must be finite and positive")
    if learning_rate_end > learning_rate_start:
        raise ValueError("learning_rate_end must not exceed learning_rate_start")
    if options.convergence is not None and not isinstance(
        options.convergence, ConvergenceOptions1D
    ):
        raise TypeError("convergence must be a ConvergenceOptions1D instance or None")
    if options.optimization is not None and not isinstance(
        options.optimization, LatticeOptimizationOptions1D
    ):
        raise TypeError(
            "optimization must be a LatticeOptimizationOptions1D instance or None"
        )
    if not isinstance(options.progress, (bool, np.bool_)):
        raise TypeError("progress must be a boolean")
    if not isinstance(options.progress_description, str):
        raise TypeError("progress_description must be a string")
    if not options.progress_description.strip():
        raise ValueError("progress_description must not be empty")
    return options


_PARTITION_NAMES = (
    "training_indices",
    "validation_indices",
    "audit_indices",
    "excluded_indices",
)


def _validated_scan_partition(
    partition: EnsembleScanPartition1D,
) -> EnsembleScanPartition1D:
    if not isinstance(partition.n_scans, (int, np.integer)):
        raise TypeError("scan-partition n_scans must be an integer")
    n_scans = int(partition.n_scans)
    if n_scans < 1:
        raise ValueError("scan-partition n_scans must be positive")
    normalized: dict[str, np.ndarray] = {}
    for name in _PARTITION_NAMES:
        values = np.asarray(getattr(partition, name))
        if values.ndim != 1 or (
            values.size and not np.issubdtype(values.dtype, np.integer)
        ):
            raise ValueError(f"scan partition {name} must be a 1D integer array")
        values = values.astype(np.int64, copy=True)
        if (
            np.unique(values).size != values.size
            or np.any(values < 0)
            or np.any(values >= n_scans)
        ):
            raise ValueError(f"scan partition {name} contains invalid indices")
        normalized[name] = values
    if not normalized["training_indices"].size:
        raise ValueError("scan partition must contain at least one training scan")
    if not normalized["validation_indices"].size:
        raise ValueError("scan partition must contain at least one validation scan")
    for first_index, first_name in enumerate(_PARTITION_NAMES):
        for second_name in _PARTITION_NAMES[first_index + 1 :]:
            if np.intersect1d(
                normalized[first_name], normalized[second_name]
            ).size:
                raise ValueError("scan partitions are not disjoint")
    combined = np.concatenate([normalized[name] for name in _PARTITION_NAMES])
    if not np.array_equal(np.sort(combined), np.arange(n_scans)):
        raise ValueError("scan partitions do not cover every scan exactly once")
    return EnsembleScanPartition1D(n_scans=n_scans, **normalized)


def _scan_partition_from_results(
    results: Sequence[LatticeSiteReconstruction1D],
) -> EnsembleScanPartition1D | None:
    reference: EnsembleScanPartition1D | None = None
    missing_all = False
    for result in results:
        present = tuple(name in result.metadata for name in _PARTITION_NAMES)
        if not any(present):
            missing_all = True
            if reference is not None:
                raise ValueError("optimizer starts have inconsistent scan provenance")
            continue
        if not all(present) or missing_all:
            raise ValueError(
                "scan provenance must contain all four partition index arrays"
            )
        measured = np.asarray(result.measured_intensities)
        if measured.ndim < 1:
            raise ValueError("measured_intensities must have a scan dimension")
        candidate = _validated_scan_partition(
            EnsembleScanPartition1D(
                n_scans=int(measured.shape[0]),
                **{
                    name: np.asarray(result.metadata[name])
                    for name in _PARTITION_NAMES
                },
            )
        )
        if reference is None:
            reference = candidate
        elif candidate.n_scans != reference.n_scans or any(
            not np.array_equal(
                getattr(candidate, name), getattr(reference, name)
            )
            for name in _PARTITION_NAMES
        ):
            raise ValueError("optimizer starts use different scan partitions")
    return reference


def _options_json(options: MultistartOptions1D) -> str:
    _validate_options(options)
    payload = {
        "n_starts": int(options.n_starts),
        "base_seed": int(options.base_seed),
        "initial_translation_half_width_A": [
            float(value) for value in options.initial_translation_half_width_A
        ],
        "relative_loss_tolerance": float(options.relative_loss_tolerance),
        "absolute_loss_tolerance": float(options.absolute_loss_tolerance),
        "minimum_accepted_starts": int(options.minimum_accepted_starts),
        "minimum_accepted_fraction": float(options.minimum_accepted_fraction),
        "vacancy_threshold": float(options.vacancy_threshold),
        "vacancy_margin": float(options.vacancy_margin),
        "agreement_fraction": float(options.agreement_fraction),
        "maximum_displacement_spread_A": float(
            options.maximum_displacement_spread_A
        ),
        "maximum_rigid_spread_A": float(options.maximum_rigid_spread_A),
        "minimum_converged_fraction": float(options.minimum_converged_fraction),
        "maximum_bound_fraction": float(options.maximum_bound_fraction),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _options_from_json(serialized: str) -> MultistartOptions1D:
    try:
        payload = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid multistart options metadata") from error
    if not isinstance(payload, dict):
        raise ValueError("multistart options metadata must be a JSON object")
    expected = {
        "n_starts",
        "base_seed",
        "initial_translation_half_width_A",
        "relative_loss_tolerance",
        "absolute_loss_tolerance",
        "minimum_accepted_starts",
        "minimum_accepted_fraction",
        "vacancy_threshold",
        "vacancy_margin",
        "agreement_fraction",
        "maximum_displacement_spread_A",
        "maximum_rigid_spread_A",
        "minimum_converged_fraction",
        "maximum_bound_fraction",
    }
    if set(payload) != expected:
        raise ValueError("multistart options metadata has missing or unknown fields")
    half_width = payload["initial_translation_half_width_A"]
    if not isinstance(half_width, list) or len(half_width) != 2:
        raise ValueError(
            "initial_translation_half_width_A metadata must contain two values"
        )
    options = MultistartOptions1D(
        **{
            **payload,
            "initial_translation_half_width_A": tuple(half_width),
        }
    )
    _validate_options(options)
    return options


def _optional_bool(value: bool | None, *, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be bool or None")
    return bool(value)


def _optional_nonempty_string(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _shared_result_metadata_identifier(
    results: Sequence[LatticeSiteReconstruction1D],
    name: str,
) -> str:
    """Return a required nonempty identifier shared by every optimizer run."""
    identifiers: list[str] = []
    for index, result in enumerate(results):
        if not isinstance(result.metadata, Mapping):
            raise TypeError(f"optimizer start {index} metadata must be a mapping")
        value = result.metadata.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"every optimizer start must provide a nonempty metadata {name!r}"
            )
        identifiers.append(value)
    if len(set(identifiers)) != 1:
        raise ValueError(
            f"optimizer starts must share one metadata {name!r} identifier"
        )
    return identifiers[0]


def _validated_residual_evidence_for_results(
    evidence: ResidualCalibrationEvidence1D | None,
    results: Sequence[LatticeSiteReconstruction1D],
    scan_partition: EnsembleScanPartition1D | None,
) -> tuple[bool | None, str | None]:
    """Validate held-out residual evidence and bind it to the inverse problem."""
    if evidence is None:
        return None, None
    if not isinstance(evidence, ResidualCalibrationEvidence1D):
        raise TypeError(
            "residual_calibration_evidence must be a "
            "ResidualCalibrationEvidence1D instance or None"
        )
    validate_residual_calibration_evidence_1d(evidence)
    if scan_partition is None or not scan_partition.audit_indices.size:
        raise ValueError(
            "residual calibration evidence requires a persisted non-empty audit "
            "partition"
        )
    expected_indices = tuple(
        int(value) for value in np.asarray(scan_partition.audit_indices)
    )
    if evidence.held_out_scan_indices != expected_indices:
        raise ValueError(
            "residual calibration evidence held-out indices must exactly match "
            "the persisted audit indices"
        )
    problem_id = _shared_result_metadata_identifier(
        results, "reconstruction_problem_id"
    )
    if evidence.reconstruction_problem_id != problem_id:
        raise ValueError(
            "residual calibration evidence reconstruction_problem_id does not "
            "match the optimizer starts"
        )
    return bool(evidence.passed), problem_id


_NOMINAL_DETECTOR = DetectorPerturbation1D()
_NOMINAL_FORWARD_MODEL = ForwardModelMismatch1D()


def _scenario_is_genuinely_non_nominal(scenario: Any) -> bool:
    """Exclude identifier-only changes from mismatch coverage claims."""
    detector = scenario.detector
    detector_non_nominal = any(
        getattr(detector, name) != getattr(_NOMINAL_DETECTOR, name)
        for name in DetectorPerturbation1D.__dataclass_fields__
        if name != "calibration_id"
    )
    forward_non_nominal = any(
        getattr(scenario.forward_mismatch, name)
        != getattr(_NOMINAL_FORWARD_MODEL, name)
        for name in ForwardModelMismatch1D.__dataclass_fields__
    )
    return bool(detector_non_nominal or forward_non_nominal)


def _validated_mismatch_report_for_results(
    report: SyntheticBenchmarkReport1D | None,
    results: Sequence[LatticeSiteReconstruction1D],
) -> tuple[
    bool | None,
    bool | None,
    bool | None,
    bool | None,
    str | None,
]:
    """Validate robustness evidence and bind its algorithm identity to runs."""
    if report is None:
        return None, None, None, None, None
    if not isinstance(report, SyntheticBenchmarkReport1D):
        raise TypeError(
            "mismatch_benchmark_report must be a "
            "SyntheticBenchmarkReport1D instance or None"
        )
    validate_synthetic_benchmark_report_1d(report)
    reconstructor_id = _shared_result_metadata_identifier(
        results, "reconstructor_id"
    )
    if report.reconstructor_id != reconstructor_id:
        raise ValueError(
            "mismatch benchmark reconstructor_id does not match the optimizer "
            "starts"
        )
    has_non_nominal_scenario = any(
        _scenario_is_genuinely_non_nominal(item.scenario)
        for item in report.scenarios
    )
    has_truth_structural_criterion = any(
        criterion.metric_name.startswith("truth.")
        for criterion in report.criteria.criteria
    )
    independent_forward = report.generator_id != report.reconstructor_id
    return (
        bool(report.passed),
        bool(has_non_nominal_scenario),
        bool(has_truth_structural_criterion),
        bool(independent_forward),
        reconstructor_id,
    )


def _validated_trust_flags(
    values: Mapping[str, bool | None],
) -> dict[str, bool | None]:
    if not isinstance(values, Mapping):
        raise TypeError("trust_flags must be a mapping")
    normalized: dict[str, bool | None] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise ValueError("trust-flag names must be non-empty strings")
        normalized[name] = _optional_bool(value, name=f"trust flag {name}")
    return normalized


def _trust_flags_from_json(serialized: str) -> dict[str, bool | None]:
    try:
        payload = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("invalid trust-flags metadata") from error
    if not isinstance(payload, dict):
        raise ValueError("archived trust flags must be a JSON object")
    return _validated_trust_flags(payload)


def multistart_site_translation_offsets_1d(
    options: MultistartOptions1D | None = None,
) -> np.ndarray:
    """Return deterministic zero-first, antithetic active-site translations."""
    options = MultistartOptions1D() if options is None else options
    if not isinstance(options, MultistartOptions1D):
        raise TypeError("options must be a MultistartOptions1D instance or None")
    _validate_options(options)
    half_width = np.asarray(options.initial_translation_half_width_A, dtype=float)
    offsets = [np.zeros(2, dtype=float)]
    rng = np.random.default_rng(options.base_seed)
    while len(offsets) < options.n_starts:
        candidate = rng.uniform(-half_width, half_width)
        offsets.append(candidate)
        if len(offsets) < options.n_starts:
            offsets.append(-candidate)
    return np.asarray(offsets)


_ACTIVE_SITE_REGISTRATION_SCOPE = "active_sites_relative_to_fixed_reference"


def _finite_real_initial_array(
    name: str,
    value: Any | None,
    *,
    shape: tuple[int, ...],
    default: float = 0.0,
    dtype: np.dtype[Any],
) -> np.ndarray:
    array = (
        np.full(shape, default, dtype=dtype)
        if value is None
        else np.asarray(value)
    )
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real")
    array = np.asarray(array, dtype=dtype)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _prepared_multistart_initializations(
    prepared: PreparedLatticeSiteReconstruction1D,
    options: PreparedMultistartRunOptions1D,
    offsets: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Build and preflight every active-site translation initialization."""
    model = prepared.model
    reference = np.asarray(model.reference_potential)
    dtype = reference.dtype
    sites = np.asarray(model.site_coordinates)
    controls_s = np.asarray(model.control_coordinates_s)
    controls_u = np.asarray(model.control_coordinates_u)
    control_shape = (controls_s.size, controls_u.size, 2)
    vacancies = _finite_real_initial_array(
        "initial_vacancy_fractions",
        options.initial_vacancy_fractions,
        shape=(sites.shape[0],),
        dtype=dtype,
    )
    if np.any((vacancies < 0.0) | (vacancies > 1.0)):
        raise ValueError("initial_vacancy_fractions must lie in [0, 1]")
    controls = _finite_real_initial_array(
        "initial_displacement_controls",
        options.initial_displacement_controls,
        shape=control_shape,
        dtype=dtype,
    )
    rigid = _finite_real_initial_array(
        "initial_rigid_displacement",
        options.initial_rigid_displacement,
        shape=(2,),
        dtype=dtype,
    )
    half_width = np.asarray(
        options.ensemble_options.initial_translation_half_width_A,
        dtype=float,
    )
    tolerance = 32.0 * max(
        np.finfo(dtype).eps if np.issubdtype(dtype, np.inexact) else 0.0,
        np.finfo(float).eps,
    )

    if prepared.separate_rigid_registration:
        canonical_rigid, canonical_controls = (
            decompose_lattice_site_displacement_controls_1d(
                model.site_coordinates,
                controls,
                model.control_coordinates_s,
                model.control_coordinates_u,
                rigid_displacement=rigid,
            )
        )
        canonical_rigid = np.asarray(canonical_rigid, dtype=dtype)
        canonical_controls = np.asarray(canonical_controls, dtype=dtype)
        rigid_limit = float(prepared.maximum_rigid_displacement)
        control_limit = float(prepared.control_scale)
        if np.any(np.abs(canonical_controls) > control_limit + tolerance):
            raise ValueError(
                "initial residual controls exceed the prepared active-site "
                "residual bound"
            )
        if np.any(np.abs(canonical_rigid) + half_width > rigid_limit + tolerance):
            raise ValueError(
                "initial translation half-width exceeds the prepared active-site "
                "rigid-registration bound"
            )
        return tuple(
            (
                vacancies.copy(),
                canonical_controls.copy(),
                np.asarray(canonical_rigid + offset, dtype=dtype),
            )
            for offset in offsets
        )

    control_limit = float(prepared.control_scale)
    if np.any(np.abs(rigid) > tolerance):
        raise ValueError(
            "initial_rigid_displacement requires a prepared problem with "
            "separate_rigid_registration=True"
        )
    if np.any(
        np.abs(controls) + half_width.reshape((1, 1, 2))
        > control_limit + tolerance
    ):
        raise ValueError(
            "initial translation half-width exceeds the prepared active-site "
            "constant-control bound"
        )
    return tuple(
        (
            vacancies.copy(),
            np.asarray(controls + offset.reshape((1, 1, 2)), dtype=dtype),
            np.zeros(2, dtype=dtype),
        )
        for offset in offsets
    )


def _label_multistart_result(
    result: LatticeSiteReconstruction1D,
    *,
    start_index: int,
    initial_translation: np.ndarray,
) -> LatticeSiteReconstruction1D:
    metadata = {
        **dict(result.metadata),
        "registration_scope": _ACTIVE_SITE_REGISTRATION_SCOPE,
        "registration_is_global_experimental_alignment": False,
        "multistart_selection_metric": "validation_loss",
        "held_out_audit_role": "post_selection_evaluation_only",
        "multistart_start_index": int(start_index),
        "multistart_initial_site_translation_A": np.asarray(
            initial_translation, dtype=float
        ).tolist(),
    }
    return replace(result, metadata=metadata)


def _strip_nonrepresentative_checkpoints(
    result: LatticeSiteReconstruction1D,
) -> LatticeSiteReconstruction1D:
    """Discard compact parameter histories after medoid selection."""
    metadata = {
        **dict(result.metadata),
        "screening_checkpoint_interval": result.metadata.get(
            "checkpoint_interval"
        ),
        "checkpoint_interval": None,
        "checkpoint_history_status": "discarded_nonrepresentative",
    }
    return replace(
        result,
        checkpoint_updates=result.checkpoint_updates[:0],
        vacancy_fraction_history=result.vacancy_fraction_history[:0],
        displacement_control_history=result.displacement_control_history[:0],
        rigid_displacement_history=result.rigid_displacement_history[:0],
        metadata=metadata,
    )


def run_prepared_lattice_site_multistart_1d(
    prepared: PreparedLatticeSiteReconstruction1D,
    *,
    options: PreparedMultistartRunOptions1D | None = None,
) -> PreparedMultistartResult1D:
    """Run prepared starts and directly reuse their real medoid trajectory.

    Every screening start receives a fresh optimizer state and RNG state from
    :func:`run_prepared_lattice_site_reconstruction_1d`.  Candidate acceptance
    and medoid selection use validation losses only.  Held-out audit losses are
    evaluated and retained by the low-level runs, but are strictly post-
    selection diagnostics. Compact parameter checkpoints are collected for
    every candidate so the selected trajectory can be reused exactly; histories
    from non-selected candidates are discarded after selection. This avoids a
    second GPU trajectory, whose atomic scatter operations need not be bitwise
    reproducible. This runner reports optimizer repeatability; it does not
    manufacture sensitivity, observability, calibration, or mismatch evidence
    and therefore cannot by itself mark a structure as trusted.
    """
    if not isinstance(prepared, PreparedLatticeSiteReconstruction1D):
        raise TypeError(
            "prepared must be a PreparedLatticeSiteReconstruction1D instance"
        )
    options = _validated_prepared_run_options(options)
    if np.asarray(prepared.validation_indices).size == 0:
        raise ValueError(
            "prepared multistart selection requires a non-empty validation split"
        )
    ensemble_options = options.ensemble_options
    offsets = multistart_site_translation_offsets_1d(ensemble_options)
    initializations = _prepared_multistart_initializations(
        prepared, options, offsets
    )
    maximum_seed = int(ensemble_options.base_seed) + ensemble_options.n_starts - 1
    if maximum_seed > np.iinfo(np.int64).max:
        raise ValueError("base_seed is too large for deterministic start seeds")
    seeds = np.arange(
        int(ensemble_options.base_seed),
        maximum_seed + 1,
        dtype=np.int64,
    )

    common_arguments = {
        "learning_rate_start": options.learning_rate_start,
        "learning_rate_end": options.learning_rate_end,
        "updates": options.updates,
        "validation_interval": options.validation_interval,
        "training_diagnostic_scan_count": (
            options.training_diagnostic_scan_count
        ),
        "progress": bool(options.progress),
        "convergence": options.convergence,
        "optimization": options.optimization,
    }
    screening_results: list[LatticeSiteReconstruction1D] = []
    for start_index, (initialization, offset, seed) in enumerate(
        zip(initializations, offsets, seeds)
    ):
        vacancies, controls, rigid = initialization
        result = run_prepared_lattice_site_reconstruction_1d(
            prepared,
            initial_vacancy_fractions=vacancies,
            initial_displacement_controls=controls,
            initial_rigid_displacement=rigid,
            seed=int(seed),
            checkpoint_interval=options.representative_checkpoint_interval,
            progress_description=(
                f"{options.progress_description} screening "
                f"{start_index + 1}/{ensemble_options.n_starts}"
            ),
            **common_arguments,
        )
        screening_results.append(
            _label_multistart_result(
                result,
                start_index=start_index,
                initial_translation=offset,
            )
        )

    screening_tuple = tuple(screening_results)
    # No trust-evidence arguments are supplied here.  Audit loss is persisted
    # by each run but the summarizer's accepted set and medoid use best_metric,
    # which is validation loss because a validation partition is required.
    ensemble = summarize_lattice_site_ensemble_1d(
        screening_tuple,
        options=ensemble_options,
    )
    representative_index = ensemble.representative_index
    representative = screening_tuple[representative_index]
    returned_screening = tuple(
        result
        if index == representative_index
        else _strip_nonrepresentative_checkpoints(result)
        for index, result in enumerate(screening_tuple)
    )
    if returned_screening[representative_index] is not representative:
        raise RuntimeError(
            "selected representative trajectory was not reused by identity"
        )
    return PreparedMultistartResult1D(
        screening_results=returned_screening,
        ensemble=ensemble,
        representative_result=representative,
        initial_site_translations_A=offsets.copy(),
        seeds=seeds.copy(),
        options=options,
        representative_trajectory_reused=True,
        registration_scope=_ACTIVE_SITE_REGISTRATION_SCOPE,
    )


def _ordered_site_coordinates_match(first: Any, second: Any) -> bool:
    first_raw = np.asarray(first)
    second_raw = np.asarray(second)
    if first_raw.shape != second_raw.shape:
        return False
    tolerance = 8.0 * max(
        np.finfo(first_raw.dtype).eps
        if np.issubdtype(first_raw.dtype, np.inexact)
        else 0.0,
        np.finfo(second_raw.dtype).eps
        if np.issubdtype(second_raw.dtype, np.inexact)
        else 0.0,
    )
    return bool(
        np.allclose(
            np.asarray(first_raw, dtype=float),
            np.asarray(second_raw, dtype=float),
            rtol=tolerance,
            atol=tolerance,
        )
    )


def _run_summary(
    result: LatticeSiteReconstruction1D,
    *,
    loss: float,
    audit_loss: float,
) -> LatticeSiteRunSummary1D:
    sites = np.asarray(result.site_coordinates)
    total = np.asarray(result.displaced_site_coordinates) - sites
    rigid = np.asarray(result.rigid_displacement, dtype=float)
    if rigid.shape != (2,):
        raise ValueError("every rigid displacement must have shape (2,)")
    residual = total - rigid
    bound_fraction = float(
        result.metadata.get("best_total_displacement_bound_fraction", np.nan)
    )
    seed_value = result.metadata.get("seed")
    return LatticeSiteRunSummary1D(
        loss=float(loss),
        converged=bool(result.converged),
        bound_fraction=bound_fraction,
        vacancy_fractions=np.asarray(result.vacancy_fractions, dtype=float),
        residual_site_displacements=np.asarray(residual, dtype=float),
        rigid_displacement=rigid,
        seed=None if seed_value is None else int(seed_value),
        audit_loss=float(audit_loss),
    )


def _medoid_index(
    vacancies: np.ndarray,
    residual: np.ndarray,
    rigid: np.ndarray,
    *,
    vacancy_scale: float,
    displacement_scale: float,
    rigid_scale: float,
) -> tuple[int, np.ndarray]:
    vacancy_delta = (vacancies[:, None] - vacancies[None, :]) / vacancy_scale
    residual_delta = (residual[:, None] - residual[None, :]) / displacement_scale
    rigid_delta = (rigid[:, None] - rigid[None, :]) / rigid_scale
    distance = np.sqrt(
        np.mean(vacancy_delta**2, axis=2)
        + np.mean(residual_delta**2, axis=(2, 3))
        + np.mean(rigid_delta**2, axis=2)
    )
    return int(np.argmin(np.sum(distance, axis=1))), distance


def summarize_lattice_site_ensemble_1d(
    results: Sequence[LatticeSiteReconstruction1D],
    *,
    options: MultistartOptions1D | None = None,
    sensitivity_screen: LatticeSiteSensitivityScreen1D | None = None,
    observability_reports: Sequence[LatticeSiteObservability1D] | None = None,
    residual_calibration_evidence: ResidualCalibrationEvidence1D | None = None,
    mismatch_benchmark_report: SyntheticBenchmarkReport1D | None = None,
) -> LatticeSiteEnsemble1D:
    """Summarize repeatability and provenance-bound structural evidence.

    Validation losses alone select the accepted starts and representative.
    Held-out residual evidence and truth-aware mismatch benchmarks are only
    consulted after selection and can therefore gate trust but never select a
    reconstruction.
    """
    options = MultistartOptions1D() if options is None else options
    if not isinstance(options, MultistartOptions1D):
        raise TypeError("options must be a MultistartOptions1D instance or None")
    _validate_options(options)
    if not results:
        raise ValueError("results must contain at least one successful start")
    scan_partition = _scan_partition_from_results(results)
    residual_calibration_passed, residual_problem_id = (
        _validated_residual_evidence_for_results(
            residual_calibration_evidence,
            results,
            scan_partition,
        )
    )
    (
        mismatch_benchmark_passed,
        mismatch_has_non_nominal_scenario,
        mismatch_has_truth_structural_criterion,
        mismatch_independent_forward,
        mismatch_reconstructor_id,
    ) = _validated_mismatch_report_for_results(
        mismatch_benchmark_report,
        results,
    )

    first_sites = np.asarray(results[0].site_coordinates, dtype=float)
    if (
        first_sites.ndim != 2
        or first_sites.shape[1:] != (2,)
        or np.any(~np.isfinite(first_sites))
    ):
        raise ValueError("site coordinates must be a finite (n_site, 2) array")
    n_site = len(first_sites)
    first_roles = np.asarray(results[0].site_role_codes)
    if first_roles.size:
        if first_roles.shape != (n_site,):
            raise ValueError("site_role_codes must contain one role per site")
        reportable_sites = first_roles == int(LatticeSiteRole1D.TARGET)
        nuisance_sites = first_roles == int(LatticeSiteRole1D.NUISANCE)
        if not np.all(reportable_sites | nuisance_sites) or not np.any(
            reportable_sites
        ):
            raise ValueError(
                "ensemble modeled-site roles must be TARGET/NUISANCE with a target"
            )
        support_contract_id = results[0].support_contract_id
        material_partition_complete = bool(results[0].material_scope_complete)
        fully_parameterized_scope = bool(
            results[0].material_scope_fully_parameterized
        )
        if fully_parameterized_scope and not material_partition_complete:
            raise ValueError(
                "fully parameterized material scope requires a complete "
                "material partition"
            )
        material_scope_complete = bool(
            material_partition_complete
            and fully_parameterized_scope
        )
        if material_partition_complete and (
            not isinstance(support_contract_id, str)
            or len(support_contract_id) != 64
        ):
            raise ValueError(
                "complete material scope requires a support-contract digest"
            )
    else:
        reportable_sites = np.ones(n_site, dtype=bool)
        nuisance_sites = np.zeros(n_site, dtype=bool)
        support_contract_id = None
        material_partition_complete = False
        fully_parameterized_scope = False
        material_scope_complete = False
    for result_index, result in enumerate(results):
        vacancies = np.asarray(result.vacancy_fractions, dtype=float)
        displaced = np.asarray(result.displaced_site_coordinates, dtype=float)
        rigid = np.asarray(result.rigid_displacement, dtype=float)
        if (
            vacancies.shape != (n_site,)
            or np.any(~np.isfinite(vacancies))
            or np.any((vacancies < 0.0) | (vacancies > 1.0))
        ):
            raise ValueError(
                "every result must contain finite vacancy fractions in [0, 1]"
            )
        if displaced.shape != (n_site, 2) or np.any(~np.isfinite(displaced)):
            raise ValueError(
                "every result must contain finite displaced site coordinates"
            )
        if rigid.shape != (2,) or np.any(~np.isfinite(rigid)):
            raise ValueError("every result must contain a finite rigid displacement")
        result_roles = np.asarray(result.site_role_codes)
        if not np.array_equal(result_roles, first_roles):
            raise ValueError("all starts must use identical ordered site roles")
        if result.support_contract_id != support_contract_id:
            raise ValueError("all starts must use the same support contract")
        if bool(result.material_scope_complete) != material_partition_complete:
            raise ValueError("all starts must agree on material-scope completeness")
        if (
            bool(result.material_scope_fully_parameterized)
            != fully_parameterized_scope
        ):
            raise ValueError(
                "all starts must agree on fully parameterized material scope"
            )
        if result_index == 0:
            continue
        if not _ordered_site_coordinates_match(
            result.site_coordinates, results[0].site_coordinates
        ):
            raise ValueError("all starts must use identical ordered site coordinates")
    losses = np.asarray(
        [float(result.metadata["best_metric"]) for result in results]
    )
    if np.any(~np.isfinite(losses)) or np.any(losses < 0.0):
        raise ValueError("selection losses must be finite and non-negative")

    stored_audit = np.asarray([float(result.audit_loss) for result in results])
    held_out_audit_available = bool(np.all(np.isfinite(stored_audit)))
    audit = (
        stored_audit
        if held_out_audit_available
        else np.full(len(results), np.nan, dtype=float)
    )
    if held_out_audit_available:
        if scan_partition is None or not scan_partition.audit_indices.size:
            raise ValueError(
                "finite held-out audit loss requires a non-empty persisted audit "
                "partition"
            )
        for result, audit_value in zip(results, audit):
            metadata_audit = float(result.metadata.get("audit_metric", np.nan))
            if not np.isfinite(metadata_audit) or not np.isclose(
                metadata_audit, audit_value, rtol=1e-12, atol=1e-15
            ):
                raise ValueError("result audit_loss and audit_metric disagree")

    summaries = tuple(
        _run_summary(result, loss=loss, audit_loss=audit_loss)
        for result, loss, audit_loss in zip(results, losses, audit)
    )
    best_loss = float(np.min(losses))
    cutoff = best_loss + max(
        options.absolute_loss_tolerance,
        options.relative_loss_tolerance * max(best_loss, np.finfo(float).eps),
    )
    accepted_mask = losses <= cutoff
    accepted_indices = np.flatnonzero(accepted_mask)
    accepted = [summaries[index] for index in accepted_indices]
    vacancies = np.stack([run.vacancy_fractions for run in accepted])
    residual = np.stack([run.residual_site_displacements for run in accepted])
    rigid = np.stack([run.rigid_displacement for run in accepted])
    if vacancies.shape[1:] != (n_site,) or residual.shape[1:] != (n_site, 2):
        raise ValueError("run parameter arrays do not match the site coordinates")

    vacancy_q05, vacancy_median, vacancy_q95 = np.quantile(
        vacancies, (0.05, 0.5, 0.95), axis=0
    )
    vacancy_calls = vacancies >= options.vacancy_threshold
    call_frequency = np.mean(vacancy_calls, axis=0)
    vacancy_state = np.full(n_site, -1, dtype=np.int8)
    vacancy_state[
        vacancy_q95 < options.vacancy_threshold - options.vacancy_margin
    ] = 0
    vacancy_state[
        vacancy_q05 > options.vacancy_threshold + options.vacancy_margin
    ] = 1

    residual_q05, residual_median, residual_q95 = np.quantile(
        residual, (0.05, 0.5, 0.95), axis=0
    )
    radial_spread = np.quantile(
        np.linalg.norm(residual - residual_median[None, ...], axis=-1),
        0.9,
        axis=0,
    )
    call_agreement = np.maximum(call_frequency, 1.0 - call_frequency)
    optimizer_agreement = (
        (vacancy_state != -1)
        & (call_agreement >= options.agreement_fraction)
        & (
            (vacancy_state == 1)
            | (radial_spread <= options.maximum_displacement_spread_A)
        )
    )
    vacancy_q05[~reportable_sites] = np.nan
    vacancy_median[~reportable_sites] = np.nan
    vacancy_q95[~reportable_sites] = np.nan
    call_frequency[~reportable_sites] = np.nan
    vacancy_state[~reportable_sites] = -1
    optimizer_agreement[~reportable_sites] = False
    residual_q05[~reportable_sites] = np.nan
    residual_median[~reportable_sites] = np.nan
    residual_q95[~reportable_sites] = np.nan
    radial_spread[~reportable_sites] = np.nan
    confidently_vacant = vacancy_state == 1
    residual_q05[confidently_vacant] = np.nan
    residual_median[confidently_vacant] = np.nan
    residual_q95[confidently_vacant] = np.nan
    radial_spread[confidently_vacant] = np.nan

    if observability_reports is None:
        observable = np.zeros(n_site, dtype=bool)
        observability_available = False
        observability_noise_calibrated = False
        observability_nuisance_complete = False
        observability_solver_verified = False
        observability_problem_ids_verified = False
        observability_problem_id = None
    else:
        reports = tuple(observability_reports)
        if len(reports) != len(results):
            raise ValueError(
                "observability_reports must have one report per optimizer start"
            )
        observability_problem_id = _shared_result_metadata_identifier(
            results, "reconstruction_problem_id"
        )
        report_partitions_verified = True
        for report, result in zip(reports, results):
            if not isinstance(report, LatticeSiteObservability1D):
                raise TypeError(
                    "observability_reports must contain "
                    "LatticeSiteObservability1D instances"
                )
            if not _ordered_site_coordinates_match(
                report.site_coordinates, results[0].site_coordinates
            ):
                raise ValueError(
                    "observability reports and ensemble must use identical sites"
                )
            if not isinstance(report.metadata, Mapping):
                raise TypeError("observability report metadata must be a mapping")
            report_problem_id = report.metadata.get("reconstruction_problem_id")
            if (
                not isinstance(report_problem_id, str)
                or not report_problem_id.strip()
            ):
                raise ValueError(
                    "every observability report must provide a nonempty metadata "
                    "'reconstruction_problem_id'"
                )
            if report_problem_id != result.metadata["reconstruction_problem_id"]:
                raise ValueError(
                    "observability report reconstruction_problem_id does not "
                    "match its optimizer start"
                )
            expected_fit = np.asarray(
                result.metadata.get("training_indices", []), dtype=np.int64
            )
            expected_audit = np.asarray(
                result.metadata.get("audit_indices", []), dtype=np.int64
            )
            actual_fit = np.asarray(report.fit.scan_indices, dtype=np.int64)
            actual_audit = (
                np.asarray(report.audit.scan_indices, dtype=np.int64)
                if report.audit is not None
                else np.empty(0, dtype=np.int64)
            )
            report_partitions_verified &= bool(
                expected_fit.size
                and expected_audit.size
                and np.array_equal(actual_fit, expected_fit)
                and np.array_equal(actual_audit, expected_audit)
                and not np.intersect1d(actual_fit, actual_audit).size
                and report.ideal_poisson_information
            )
        observability_problem_ids_verified = True
        accepted_reports = [reports[index] for index in accepted_indices]
        observable = np.logical_and.reduce(
            [
                np.asarray(report.site_observable, dtype=bool)
                for report in accepted_reports
            ]
        )
        observability_noise_calibrated = all(
            report.calibrated_noise for report in accepted_reports
        )
        observability_nuisance_complete = all(
            report.nuisance_scope_complete for report in accepted_reports
        )
        observability_solver_verified = all(
            report.fit.solver_verified
            and report.audit is not None
            and report.audit.solver_verified
            for report in accepted_reports
        )
        observability_available = all(
            report.suitable_for_trust_gate for report in accepted_reports
        ) and report_partitions_verified and observability_problem_ids_verified
    if sensitivity_screen is None:
        # A suitable marginalized observability report already contains the
        # local-information requirement.
        sensitive = observable.copy()
        sensitivity_available = observability_available
    else:
        if not isinstance(sensitivity_screen, LatticeSiteSensitivityScreen1D):
            raise TypeError(
                "sensitivity_screen must be a LatticeSiteSensitivityScreen1D"
            )
        if not _ordered_site_coordinates_match(
            sensitivity_screen.site_coordinates, results[0].site_coordinates
        ):
            raise ValueError(
                "sensitivity screen and ensemble must use identical ordered sites"
            )
        sensitive = np.asarray(sensitivity_screen.site_sensitive, dtype=bool)
        if sensitive.shape != (n_site,):
            raise ValueError("sensitivity screen must have one value per site")
        sensitivity_available = True
    observable = np.asarray(observable, dtype=bool).copy()
    sensitive = np.asarray(sensitive, dtype=bool).copy()
    observable[~reportable_sites] = False
    sensitive[~reportable_sites] = False
    residual_evidence_available = residual_calibration_evidence is not None
    mismatch_report_available = mismatch_benchmark_report is not None
    mismatch_benchmark_qualified = bool(
        mismatch_benchmark_passed is True
        and mismatch_has_non_nominal_scenario is True
        and mismatch_has_truth_structural_criterion is True
        and mismatch_independent_forward is True
    )
    site_evidence_passed = (
        optimizer_agreement
        & sensitive
        & observable
        & observability_available
        & held_out_audit_available
        & residual_evidence_available
        & (residual_calibration_passed is True)
        & mismatch_report_available
        & mismatch_benchmark_qualified
    )
    site_evidence_passed &= reportable_sites

    rigid_q05, rigid_median, rigid_q95 = np.quantile(
        rigid, (0.05, 0.5, 0.95), axis=0
    )
    rigid_radial_q90 = float(
        np.quantile(np.linalg.norm(rigid - rigid_median, axis=1), 0.9)
    )
    local_medoid, pairwise_distance = _medoid_index(
        vacancies[:, reportable_sites],
        residual[:, reportable_sites],
        rigid,
        vacancy_scale=max(options.vacancy_margin, 1e-6),
        displacement_scale=max(options.maximum_displacement_spread_A, 1e-6),
        rigid_scale=max(options.maximum_rigid_spread_A, 1e-6),
    )
    representative_index = int(accepted_indices[local_medoid])

    enough_successful = len(results) >= options.n_starts
    enough_accepted = (
        len(accepted) >= options.minimum_accepted_starts
        and len(accepted) / options.n_starts >= options.minimum_accepted_fraction
    )
    converged_fraction = float(np.mean([run.converged for run in accepted]))
    bound_fractions = np.asarray([run.bound_fraction for run in accepted])
    bounds_ok = bool(
        np.all(np.isfinite(bound_fractions))
        and np.all(bound_fractions <= options.maximum_bound_fraction)
    )
    registration_consensus = rigid_radial_q90 <= options.maximum_rigid_spread_A
    vacancy_consensus = (
        float(np.mean(optimizer_agreement[reportable_sites]))
        >= options.agreement_fraction
    )
    occupied = reportable_sites & (vacancy_state == 0)
    residual_consensus = bool(
        np.all(radial_spread[occupied] <= options.maximum_displacement_spread_A)
    )
    dominant_basin = bool(np.all(pairwise_distance[local_medoid] <= 1.0))
    flags: dict[str, bool | None] = {
        "enough_successful_starts": enough_successful,
        "enough_low_loss_starts": enough_accepted,
        "dominant_low_loss_basin": dominant_basin,
        "numerical_convergence_fraction": (
            converged_fraction >= options.minimum_converged_fraction
        ),
        "parameter_bounds_ok": bounds_ok,
        "registration_consensus": registration_consensus,
        "vacancy_consensus": vacancy_consensus,
        "residual_strain_consensus": residual_consensus,
        "held_out_audit_available": held_out_audit_available,
        "local_sensitivity_available": sensitivity_available,
        "local_sensitivity_passed": (
            bool(np.all(sensitive[reportable_sites]))
            if sensitivity_available
            else None
        ),
        "material_scope_complete": material_scope_complete,
        "target_only_structural_reporting": True,
        "observability_available": observability_available,
        "observability_noise_calibrated": observability_noise_calibrated,
        "observability_nuisance_scope_complete": (
            observability_nuisance_complete
        ),
        "observability_solver_verified": observability_solver_verified,
        "observability_problem_ids_verified": (
            observability_problem_ids_verified
        ),
        "residual_calibration_evidence_available": residual_evidence_available,
        "residual_calibration_audit_indices_verified": (
            True if residual_evidence_available else None
        ),
        "residual_calibration_problem_id_verified": (
            True if residual_evidence_available else None
        ),
        "residual_calibration_evidence_passed": residual_calibration_passed,
        # Backward-readable derived aliases; callers can no longer inject these
        # values directly into the summarizer.
        "residual_calibration_passed": residual_calibration_passed,
        "mismatch_benchmark_report_available": mismatch_report_available,
        "mismatch_benchmark_sourced_criteria_passed": (
            mismatch_benchmark_passed
        ),
        "mismatch_benchmark_non_nominal_scenario_present": (
            mismatch_has_non_nominal_scenario
        ),
        "mismatch_benchmark_truth_structural_criterion_present": (
            mismatch_has_truth_structural_criterion
        ),
        "mismatch_benchmark_reconstructor_id_verified": (
            True if mismatch_report_available else None
        ),
        "mismatch_benchmark_independent_forward": mismatch_independent_forward,
        "relevant_mismatch_benchmark_passed": (
            mismatch_benchmark_qualified if mismatch_report_available else None
        ),
    }
    optimizer_stable = all(
        flags[name] is True
        for name in (
            "enough_successful_starts",
            "enough_low_loss_starts",
            "dominant_low_loss_basin",
            "numerical_convergence_fraction",
            "parameter_bounds_ok",
            "registration_consensus",
            "vacancy_consensus",
            "residual_strain_consensus",
        )
    )
    # A site is not labelled trusted when the ensemble as a whole failed its
    # numerical/basin checks, even if its local evidence happens to agree.
    site_trusted = site_evidence_passed & optimizer_stable
    structurally_trusted = optimizer_stable and all(
        flags[name] is True
        for name in (
            "held_out_audit_available",
            "material_scope_complete",
            "local_sensitivity_available",
            "local_sensitivity_passed",
            "observability_available",
            "observability_noise_calibrated",
            "observability_nuisance_scope_complete",
            "observability_solver_verified",
            "observability_problem_ids_verified",
            "residual_calibration_evidence_available",
            "residual_calibration_audit_indices_verified",
            "residual_calibration_problem_id_verified",
            "residual_calibration_evidence_passed",
            "mismatch_benchmark_report_available",
            "mismatch_benchmark_sourced_criteria_passed",
            "mismatch_benchmark_non_nominal_scenario_present",
            "mismatch_benchmark_truth_structural_criterion_present",
            "mismatch_benchmark_reconstructor_id_verified",
            "mismatch_benchmark_independent_forward",
            "relevant_mismatch_benchmark_passed",
        )
    ) and bool(np.all(site_trusted[reportable_sites]))
    consensus = SitewiseConsensus1D(
        vacancy_median=vacancy_median,
        vacancy_q05=vacancy_q05,
        vacancy_q95=vacancy_q95,
        vacancy_call_frequency=call_frequency,
        vacancy_state=vacancy_state,
        residual_displacement_median=residual_median,
        residual_displacement_q05=residual_q05,
        residual_displacement_q95=residual_q95,
        residual_displacement_radial_q90_A=radial_spread,
        optimizer_agreement=optimizer_agreement,
        sensitive=sensitive,
        observable=observable,
        site_trusted=site_trusted,
    )
    evidence_provenance = EnsembleEvidenceProvenance1D(
        source="live_summary",
        sensitivity_screen_supplied=sensitivity_screen is not None,
        observability_report_count=(
            0 if observability_reports is None else len(observability_reports)
        ),
        observability_problem_ids_verified_at_summary=(
            observability_problem_ids_verified
            if observability_reports is not None
            else None
        ),
        residual_calibration_evidence_supplied=residual_evidence_available,
        residual_calibration_passed_at_summary=residual_calibration_passed,
        mismatch_benchmark_report_supplied=mismatch_report_available,
        mismatch_benchmark_passed_at_summary=mismatch_benchmark_passed,
        common_reconstruction_problem_id=(
            residual_problem_id or observability_problem_id
        ),
        common_reconstructor_id=mismatch_reconstructor_id,
        mismatch_benchmark_id=(
            mismatch_benchmark_report.benchmark_id
            if mismatch_benchmark_report is not None
            else None
        ),
        mismatch_generator_id=(
            mismatch_benchmark_report.generator_id
            if mismatch_benchmark_report is not None
            else None
        ),
        mismatch_non_nominal_scenario_present_at_summary=(
            mismatch_has_non_nominal_scenario
        ),
        mismatch_truth_structural_criterion_present_at_summary=(
            mismatch_has_truth_structural_criterion
        ),
        mismatch_independent_forward_at_summary=mismatch_independent_forward,
        structurally_trusted_at_summary=structurally_trusted,
        trusted_site_count_at_summary=int(np.count_nonzero(site_trusted)),
        typed_evidence_persisted=False,
        structural_trust_reverified_after_load=False,
    )
    return LatticeSiteEnsemble1D(
        runs=summaries,
        accepted_mask=accepted_mask,
        accepted_loss_cutoff=cutoff,
        representative_index=representative_index,
        consensus=consensus,
        rigid_median=rigid_median,
        rigid_q05=rigid_q05,
        rigid_q95=rigid_q95,
        rigid_radial_q90_A=rigid_radial_q90,
        trust_flags=flags,
        optimizer_stable=optimizer_stable,
        structurally_trusted=structurally_trusted,
        site_coordinates=first_sites.copy(),
        options=options,
        scan_partition=scan_partition,
        evidence_provenance=evidence_provenance,
    )


def save_lattice_site_ensemble_1d(
    path: str | Path,
    ensemble: LatticeSiteEnsemble1D,
) -> None:
    """Save a site-mappable compact ensemble without pickle.

    The compact archive records evidence provenance but not the full evidence
    objects or detector data. :func:`load_lattice_site_ensemble_1d` therefore
    preserves numerical consensus while failing structural trust closed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runs = ensemble.runs
    if not runs:
        raise ValueError("an ensemble archive must contain at least one run")
    if ensemble.site_coordinates is None:
        raise ValueError("site_coordinates are required for a mappable archive")
    site_coordinates = np.asarray(ensemble.site_coordinates)
    if (
        site_coordinates.ndim != 2
        or site_coordinates.shape[1:] != (2,)
        or np.any(~np.isfinite(site_coordinates))
    ):
        raise ValueError("site_coordinates must be a finite (n_site, 2) array")
    n_site = len(site_coordinates)
    if ensemble.options is None:
        raise ValueError("MultistartOptions1D are required for a reproducible archive")
    if not isinstance(ensemble.options, MultistartOptions1D):
        raise TypeError("ensemble.options must be a MultistartOptions1D instance")
    options_json = _options_json(ensemble.options)
    if len(runs) > ensemble.options.n_starts:
        raise ValueError("archive contains more runs than the configured n_starts")
    if ensemble.scan_partition is None:
        raise ValueError("an exhaustive scan partition is required for the archive")
    scan_partition = _validated_scan_partition(ensemble.scan_partition)
    accepted_mask = np.asarray(ensemble.accepted_mask)
    if accepted_mask.shape != (len(runs),):
        raise ValueError("accepted_mask must contain one value per run")
    if not 0 <= int(ensemble.representative_index) < len(runs):
        raise ValueError("representative_index is outside the saved runs")
    if not bool(accepted_mask[int(ensemble.representative_index)]):
        raise ValueError("the representative run must be in the accepted set")
    for run in runs:
        if np.asarray(run.vacancy_fractions).shape != (n_site,):
            raise ValueError("every run must contain one vacancy value per site")
        if np.asarray(run.residual_site_displacements).shape != (n_site, 2):
            raise ValueError("every run must contain one displacement per site")
        if np.asarray(run.rigid_displacement).shape != (2,):
            raise ValueError("every run rigid displacement must have shape (2,)")
    if not isinstance(
        ensemble.evidence_provenance, EnsembleEvidenceProvenance1D
    ):
        raise TypeError(
            "ensemble.evidence_provenance must be an "
            "EnsembleEvidenceProvenance1D instance"
        )
    provenance = ensemble.evidence_provenance
    if not isinstance(provenance.source, str) or not provenance.source:
        raise ValueError("evidence provenance source must be a non-empty string")
    if (
        not isinstance(provenance.observability_report_count, (int, np.integer))
        or isinstance(provenance.observability_report_count, (bool, np.bool_))
        or provenance.observability_report_count < 0
    ):
        raise ValueError("observability_report_count must be a non-negative integer")
    residual_claim = _optional_bool(
        provenance.residual_calibration_passed_at_summary,
        name="residual_calibration_passed_at_summary",
    )
    mismatch_claim = _optional_bool(
        provenance.mismatch_benchmark_passed_at_summary,
        name="mismatch_benchmark_passed_at_summary",
    )
    observability_problem_claim = _optional_bool(
        provenance.observability_problem_ids_verified_at_summary,
        name="observability_problem_ids_verified_at_summary",
    )
    mismatch_non_nominal_claim = _optional_bool(
        provenance.mismatch_non_nominal_scenario_present_at_summary,
        name="mismatch_non_nominal_scenario_present_at_summary",
    )
    mismatch_truth_claim = _optional_bool(
        provenance.mismatch_truth_structural_criterion_present_at_summary,
        name="mismatch_truth_structural_criterion_present_at_summary",
    )
    mismatch_independent_claim = _optional_bool(
        provenance.mismatch_independent_forward_at_summary,
        name="mismatch_independent_forward_at_summary",
    )
    for name in (
        "sensitivity_screen_supplied",
        "residual_calibration_evidence_supplied",
        "mismatch_benchmark_report_supplied",
        "structurally_trusted_at_summary",
        "typed_evidence_persisted",
        "structural_trust_reverified_after_load",
    ):
        if not isinstance(getattr(provenance, name), (bool, np.bool_)):
            raise TypeError(f"evidence provenance {name} must be boolean")
    problem_id = _optional_nonempty_string(
        provenance.common_reconstruction_problem_id,
        name="common_reconstruction_problem_id",
    )
    reconstructor_id = _optional_nonempty_string(
        provenance.common_reconstructor_id,
        name="common_reconstructor_id",
    )
    benchmark_id = _optional_nonempty_string(
        provenance.mismatch_benchmark_id,
        name="mismatch_benchmark_id",
    )
    generator_id = _optional_nonempty_string(
        provenance.mismatch_generator_id,
        name="mismatch_generator_id",
    )
    residual_supplied = bool(provenance.residual_calibration_evidence_supplied)
    mismatch_supplied = bool(provenance.mismatch_benchmark_report_supplied)
    if residual_supplied:
        if residual_claim is None or problem_id is None:
            raise ValueError(
                "supplied residual evidence requires its pass claim and problem id"
            )
    elif residual_claim is not None:
        raise ValueError("residual evidence pass claim requires supplied evidence")
    if mismatch_supplied:
        if any(
            value is None
            for value in (
                mismatch_claim,
                mismatch_non_nominal_claim,
                mismatch_truth_claim,
                mismatch_independent_claim,
                reconstructor_id,
                benchmark_id,
                generator_id,
            )
        ):
            raise ValueError(
                "supplied mismatch report requires complete typed provenance"
            )
    elif any(
        value is not None
        for value in (
            mismatch_claim,
            mismatch_non_nominal_claim,
            mismatch_truth_claim,
            mismatch_independent_claim,
            reconstructor_id,
            benchmark_id,
            generator_id,
        )
    ):
        raise ValueError("mismatch provenance claims require a supplied report")
    if provenance.observability_report_count:
        if observability_problem_claim is None or problem_id is None:
            raise ValueError(
                "observability reports require a verified problem-id claim"
            )
    elif observability_problem_claim is not None:
        raise ValueError(
            "observability problem-id claim requires observability reports"
        )
    if provenance.typed_evidence_persisted:
        raise ValueError("schema 5 cannot persist complete typed evidence")
    evidence_json = json.dumps(
        {
            "source": str(provenance.source),
            "sensitivity_screen_supplied": bool(
                provenance.sensitivity_screen_supplied
            ),
            "observability_report_count": int(
                provenance.observability_report_count
            ),
            "observability_problem_ids_verified_at_summary": (
                observability_problem_claim
            ),
            "residual_calibration_evidence_supplied": residual_supplied,
            "residual_calibration_passed_at_summary": (
                residual_claim
            ),
            "mismatch_benchmark_report_supplied": mismatch_supplied,
            "mismatch_benchmark_passed_at_summary": (
                mismatch_claim
            ),
            "common_reconstruction_problem_id": problem_id,
            "common_reconstructor_id": reconstructor_id,
            "mismatch_benchmark_id": benchmark_id,
            "mismatch_generator_id": generator_id,
            "mismatch_non_nominal_scenario_present_at_summary": (
                mismatch_non_nominal_claim
            ),
            "mismatch_truth_structural_criterion_present_at_summary": (
                mismatch_truth_claim
            ),
            "mismatch_independent_forward_at_summary": (
                mismatch_independent_claim
            ),
            "structurally_trusted_at_summary": bool(
                ensemble.structurally_trusted
            ),
            "trusted_site_count_at_summary": int(
                np.count_nonzero(ensemble.consensus.site_trusted)
            ),
            # Reserved for a future archive that embeds all typed reports.
            "typed_evidence_persisted": False,
            "structural_trust_reverified_after_load": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    trust_flags = _validated_trust_flags(ensemble.trust_flags)
    np.savez_compressed(
        path,
        schema_version=np.asarray(5, dtype=np.int64),
        site_coordinates=site_coordinates,
        multistart_options_json=np.asarray(options_json),
        scan_n_scans=np.asarray(scan_partition.n_scans, dtype=np.int64),
        scan_training_indices=np.asarray(
            scan_partition.training_indices, dtype=np.int64
        ),
        scan_validation_indices=np.asarray(
            scan_partition.validation_indices, dtype=np.int64
        ),
        scan_audit_indices=np.asarray(scan_partition.audit_indices, dtype=np.int64),
        scan_excluded_indices=np.asarray(
            scan_partition.excluded_indices, dtype=np.int64
        ),
        evidence_provenance_json=np.asarray(evidence_json),
        run_losses=np.asarray([run.loss for run in runs]),
        run_audit_losses=np.asarray([run.audit_loss for run in runs]),
        run_converged=np.asarray([run.converged for run in runs]),
        run_bound_fractions=np.asarray([run.bound_fraction for run in runs]),
        run_vacancy_fractions=np.stack([run.vacancy_fractions for run in runs]),
        run_residual_site_displacements=np.stack(
            [run.residual_site_displacements for run in runs]
        ),
        run_rigid_displacements=np.stack([run.rigid_displacement for run in runs]),
        run_seeds=np.asarray(
            [-1 if run.seed is None else run.seed for run in runs], dtype=np.int64
        ),
        accepted_mask=accepted_mask,
        accepted_loss_cutoff=np.asarray(ensemble.accepted_loss_cutoff),
        representative_index=np.asarray(ensemble.representative_index, dtype=np.int64),
        vacancy_median=np.asarray(ensemble.consensus.vacancy_median),
        vacancy_q05=np.asarray(ensemble.consensus.vacancy_q05),
        vacancy_q95=np.asarray(ensemble.consensus.vacancy_q95),
        vacancy_call_frequency=np.asarray(
            ensemble.consensus.vacancy_call_frequency
        ),
        vacancy_state=np.asarray(ensemble.consensus.vacancy_state),
        residual_displacement_median=np.asarray(
            ensemble.consensus.residual_displacement_median
        ),
        residual_displacement_q05=np.asarray(
            ensemble.consensus.residual_displacement_q05
        ),
        residual_displacement_q95=np.asarray(
            ensemble.consensus.residual_displacement_q95
        ),
        residual_displacement_radial_q90_A=np.asarray(
            ensemble.consensus.residual_displacement_radial_q90_A
        ),
        optimizer_agreement=np.asarray(ensemble.consensus.optimizer_agreement),
        sensitive=np.asarray(ensemble.consensus.sensitive),
        observable=np.asarray(ensemble.consensus.observable),
        site_trusted_at_summary=np.asarray(ensemble.consensus.site_trusted),
        rigid_median=np.asarray(ensemble.rigid_median),
        rigid_q05=np.asarray(ensemble.rigid_q05),
        rigid_q95=np.asarray(ensemble.rigid_q95),
        rigid_radial_q90_A=np.asarray(ensemble.rigid_radial_q90_A),
        trust_flags_at_summary_json=np.asarray(
            json.dumps(trust_flags, sort_keys=True)
        ),
        optimizer_stable=np.asarray(ensemble.optimizer_stable),
        structurally_trusted_at_summary=np.asarray(
            ensemble.structurally_trusted
        ),
    )


def load_lattice_site_ensemble_1d(path: str | Path) -> LatticeSiteEnsemble1D:
    """Load a compact ensemble, failing unarchived evidence-based trust closed."""
    with np.load(path, allow_pickle=False) as data:
        schema_version = int(data["schema_version"].item())
        if schema_version not in (1, 2, 3, 4, 5):
            raise ValueError("unsupported lattice-site ensemble schema version")
        seeds = np.asarray(data["run_seeds"])
        if (
            seeds.ndim != 1
            or not len(seeds)
            or not np.issubdtype(seeds.dtype, np.integer)
        ):
            raise ValueError("ensemble archive must contain at least one run")
        n_run = len(seeds)
        run_losses = np.asarray(data["run_losses"])
        run_converged = np.asarray(data["run_converged"])
        run_bounds = np.asarray(data["run_bound_fractions"])
        run_audit = (
            np.asarray(data["run_audit_losses"])
            if schema_version >= 2
            else np.full(n_run, np.nan)
        )
        if any(
            values.shape != (n_run,)
            for values in (run_losses, run_converged, run_bounds, run_audit)
        ):
            raise ValueError("invalid archived per-run scalar arrays")
        if np.any(~np.isfinite(run_losses)) or np.any(run_losses < 0.0):
            raise ValueError(
                "archived selection losses must be finite and non-negative"
            )
        if np.any(~np.isfinite(run_audit) & ~np.isnan(run_audit)):
            raise ValueError("archived audit losses must be finite or NaN")
        run_vacancies = np.asarray(data["run_vacancy_fractions"])
        run_residual = np.asarray(data["run_residual_site_displacements"])
        run_rigid = np.asarray(data["run_rigid_displacements"])
        if run_vacancies.ndim != 2 or run_vacancies.shape[0] != n_run:
            raise ValueError("invalid archived vacancy-fraction array")
        n_site = run_vacancies.shape[1]
        if run_residual.shape != (n_run, n_site, 2):
            raise ValueError("invalid archived residual-displacement array")
        if run_rigid.shape != (n_run, 2):
            raise ValueError("invalid archived rigid-displacement array")
        if (
            np.any(~np.isfinite(run_vacancies))
            or np.any((run_vacancies < 0.0) | (run_vacancies > 1.0))
            or np.any(~np.isfinite(run_residual))
            or np.any(~np.isfinite(run_rigid))
        ):
            raise ValueError("archived run parameters are not finite and physical")
        runs = tuple(
            LatticeSiteRunSummary1D(
                loss=float(run_losses[index]),
                converged=bool(run_converged[index]),
                bound_fraction=float(run_bounds[index]),
                vacancy_fractions=run_vacancies[index],
                residual_site_displacements=run_residual[index],
                rigid_displacement=run_rigid[index],
                seed=None if seeds[index] < 0 else int(seeds[index]),
                audit_loss=float(run_audit[index]),
            )
            for index in range(n_run)
        )
        archived_site_trusted = np.asarray(
            data[
                "site_trusted_at_summary"
                if schema_version >= 4
                else "site_trusted"
            ],
            dtype=bool,
        )
        if archived_site_trusted.shape != (n_site,):
            raise ValueError("invalid archived site-trust array")
        consensus = SitewiseConsensus1D(
            vacancy_median=np.asarray(data["vacancy_median"]),
            vacancy_q05=np.asarray(data["vacancy_q05"]),
            vacancy_q95=np.asarray(data["vacancy_q95"]),
            vacancy_call_frequency=np.asarray(data["vacancy_call_frequency"]),
            vacancy_state=np.asarray(data["vacancy_state"]),
            residual_displacement_median=np.asarray(
                data["residual_displacement_median"]
            ),
            residual_displacement_q05=np.asarray(
                data["residual_displacement_q05"]
            ),
            residual_displacement_q95=np.asarray(
                data["residual_displacement_q95"]
            ),
            residual_displacement_radial_q90_A=np.asarray(
                data["residual_displacement_radial_q90_A"]
            ),
            optimizer_agreement=np.asarray(data["optimizer_agreement"]),
            sensitive=(
                np.asarray(data["sensitive"])
                if schema_version >= 3
                else np.zeros_like(data["observable"], dtype=bool)
            ),
            observable=np.asarray(data["observable"]),
            # The compact archive does not contain the reports needed to
            # independently reverify this claim.
            site_trusted=np.zeros(n_site, dtype=bool),
        )
        for name in (
            "vacancy_median",
            "vacancy_q05",
            "vacancy_q95",
            "vacancy_call_frequency",
            "vacancy_state",
            "residual_displacement_radial_q90_A",
            "optimizer_agreement",
            "sensitive",
            "observable",
            "site_trusted",
        ):
            if np.asarray(getattr(consensus, name)).shape != (n_site,):
                raise ValueError(f"invalid archived consensus array {name}")
        for name in (
            "residual_displacement_median",
            "residual_displacement_q05",
            "residual_displacement_q95",
        ):
            if np.asarray(getattr(consensus, name)).shape != (n_site, 2):
                raise ValueError(f"invalid archived consensus array {name}")
        accepted_mask = np.asarray(data["accepted_mask"], dtype=bool)
        representative_index = int(data["representative_index"].item())
        if accepted_mask.shape != (n_run,):
            raise ValueError("invalid archived accepted-mask array")
        if not 0 <= representative_index < n_run:
            raise ValueError("archived representative_index is outside the runs")
        if not accepted_mask[representative_index]:
            raise ValueError("archived representative is not an accepted run")

        if schema_version >= 4:
            site_coordinates = np.asarray(data["site_coordinates"])
            if (
                site_coordinates.shape != (n_site, 2)
                or np.any(~np.isfinite(site_coordinates))
            ):
                raise ValueError("invalid archived ordered site coordinates")
            options = _options_from_json(
                str(data["multistart_options_json"].item())
            )
            if n_run > options.n_starts:
                raise ValueError(
                    "archive contains more runs than its multistart options permit"
                )
            scan_partition = _validated_scan_partition(
                EnsembleScanPartition1D(
                    n_scans=int(data["scan_n_scans"].item()),
                    training_indices=np.asarray(data["scan_training_indices"]),
                    validation_indices=np.asarray(data["scan_validation_indices"]),
                    audit_indices=np.asarray(data["scan_audit_indices"]),
                    excluded_indices=np.asarray(data["scan_excluded_indices"]),
                )
            )
            try:
                evidence_payload = json.loads(
                    str(data["evidence_provenance_json"].item())
                )
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError("invalid evidence provenance metadata") from error
            legacy_evidence_fields = {
                "source",
                "sensitivity_screen_supplied",
                "observability_report_count",
                "residual_calibration_passed_at_summary",
                "mismatch_benchmark_passed_at_summary",
                "structurally_trusted_at_summary",
                "trusted_site_count_at_summary",
                "typed_evidence_persisted",
                "structural_trust_reverified_after_load",
            }
            typed_evidence_fields = legacy_evidence_fields | {
                "observability_problem_ids_verified_at_summary",
                "residual_calibration_evidence_supplied",
                "mismatch_benchmark_report_supplied",
                "common_reconstruction_problem_id",
                "common_reconstructor_id",
                "mismatch_benchmark_id",
                "mismatch_generator_id",
                "mismatch_non_nominal_scenario_present_at_summary",
                "mismatch_truth_structural_criterion_present_at_summary",
                "mismatch_independent_forward_at_summary",
            }
            evidence_fields = (
                typed_evidence_fields
                if schema_version >= 5
                else legacy_evidence_fields
            )
            if not isinstance(evidence_payload, dict) or set(
                evidence_payload
            ) != evidence_fields:
                raise ValueError(
                    "evidence provenance metadata has missing or unknown fields"
                )
            report_count = evidence_payload["observability_report_count"]
            trusted_count = evidence_payload["trusted_site_count_at_summary"]
            if (
                not isinstance(report_count, int)
                or isinstance(report_count, bool)
                or report_count < 0
                or report_count not in (0, n_run)
                or not isinstance(trusted_count, int)
                or isinstance(trusted_count, bool)
                or trusted_count != int(np.count_nonzero(archived_site_trusted))
            ):
                raise ValueError("invalid evidence provenance counts")
            for name in (
                "sensitivity_screen_supplied",
                "structurally_trusted_at_summary",
                "typed_evidence_persisted",
                "structural_trust_reverified_after_load",
            ):
                if not isinstance(evidence_payload[name], bool):
                    raise ValueError(f"evidence provenance {name} must be boolean")
            for name in (
                "residual_calibration_passed_at_summary",
                "mismatch_benchmark_passed_at_summary",
            ):
                if evidence_payload[name] is not None and not isinstance(
                    evidence_payload[name], bool
                ):
                    raise ValueError(
                        f"evidence provenance {name} must be boolean or null"
                    )
            if schema_version >= 5:
                for name in (
                    "residual_calibration_evidence_supplied",
                    "mismatch_benchmark_report_supplied",
                ):
                    if not isinstance(evidence_payload[name], bool):
                        raise ValueError(
                            f"evidence provenance {name} must be boolean"
                        )
                for name in (
                    "observability_problem_ids_verified_at_summary",
                    "mismatch_non_nominal_scenario_present_at_summary",
                    "mismatch_truth_structural_criterion_present_at_summary",
                    "mismatch_independent_forward_at_summary",
                ):
                    if evidence_payload[name] is not None and not isinstance(
                        evidence_payload[name], bool
                    ):
                        raise ValueError(
                            f"evidence provenance {name} must be boolean or null"
                        )
                for name in (
                    "common_reconstruction_problem_id",
                    "common_reconstructor_id",
                    "mismatch_benchmark_id",
                    "mismatch_generator_id",
                ):
                    value = evidence_payload[name]
                    if value is not None and (
                        not isinstance(value, str) or not value.strip()
                    ):
                        raise ValueError(
                            f"evidence provenance {name} must be nonempty or null"
                        )
            if (
                not isinstance(evidence_payload["source"], str)
                or not evidence_payload["source"]
            ):
                raise ValueError("evidence provenance source must be non-empty")
            if (
                evidence_payload["typed_evidence_persisted"]
                or evidence_payload["structural_trust_reverified_after_load"]
            ):
                raise ValueError(
                    "compact ensemble schema cannot contain reverified embedded "
                    "typed evidence"
                )
            archived_structural_trust = bool(
                data["structurally_trusted_at_summary"].item()
            )
            if (
                evidence_payload["structurally_trusted_at_summary"]
                != archived_structural_trust
                or (archived_structural_trust and trusted_count != n_site)
            ):
                raise ValueError("archived structural-trust provenance disagrees")
            if schema_version >= 5:
                residual_supplied = evidence_payload[
                    "residual_calibration_evidence_supplied"
                ]
                mismatch_supplied = evidence_payload[
                    "mismatch_benchmark_report_supplied"
                ]
                problem_id = evidence_payload[
                    "common_reconstruction_problem_id"
                ]
                mismatch_values = (
                    evidence_payload["mismatch_benchmark_passed_at_summary"],
                    evidence_payload[
                        "mismatch_non_nominal_scenario_present_at_summary"
                    ],
                    evidence_payload[
                        "mismatch_truth_structural_criterion_present_at_summary"
                    ],
                    evidence_payload[
                        "mismatch_independent_forward_at_summary"
                    ],
                    evidence_payload["common_reconstructor_id"],
                    evidence_payload["mismatch_benchmark_id"],
                    evidence_payload["mismatch_generator_id"],
                )
                if residual_supplied:
                    if (
                        evidence_payload[
                            "residual_calibration_passed_at_summary"
                        ]
                        is None
                        or problem_id is None
                    ):
                        raise ValueError(
                            "archived residual evidence provenance is incomplete"
                        )
                elif evidence_payload[
                    "residual_calibration_passed_at_summary"
                ] is not None:
                    raise ValueError(
                        "archived residual pass claim has no typed evidence"
                    )
                if mismatch_supplied:
                    if any(value is None for value in mismatch_values):
                        raise ValueError(
                            "archived mismatch evidence provenance is incomplete"
                        )
                elif any(value is not None for value in mismatch_values):
                    raise ValueError(
                        "archived mismatch claims have no typed report"
                    )
                observability_claim = evidence_payload[
                    "observability_problem_ids_verified_at_summary"
                ]
                if report_count:
                    if observability_claim is None or problem_id is None:
                        raise ValueError(
                            "archived observability provenance is incomplete"
                        )
                elif observability_claim is not None:
                    raise ValueError(
                        "archived observability claim has no reports"
                    )
                if archived_structural_trust and not all(
                    value is True
                    for value in (
                        residual_supplied,
                        evidence_payload[
                            "residual_calibration_passed_at_summary"
                        ],
                        mismatch_supplied,
                        *mismatch_values[:4],
                        observability_claim,
                    )
                ):
                    raise ValueError(
                        "archived structural trust lacks qualifying typed evidence"
                    )
            evidence_provenance = EnsembleEvidenceProvenance1D(
                source=f"loaded_compact_archive_v{schema_version}",
                sensitivity_screen_supplied=evidence_payload[
                    "sensitivity_screen_supplied"
                ],
                observability_report_count=report_count,
                observability_problem_ids_verified_at_summary=(
                    evidence_payload[
                        "observability_problem_ids_verified_at_summary"
                    ]
                    if schema_version >= 5
                    else None
                ),
                residual_calibration_evidence_supplied=(
                    evidence_payload["residual_calibration_evidence_supplied"]
                    if schema_version >= 5
                    else False
                ),
                residual_calibration_passed_at_summary=evidence_payload[
                    "residual_calibration_passed_at_summary"
                ],
                mismatch_benchmark_report_supplied=(
                    evidence_payload["mismatch_benchmark_report_supplied"]
                    if schema_version >= 5
                    else False
                ),
                mismatch_benchmark_passed_at_summary=evidence_payload[
                    "mismatch_benchmark_passed_at_summary"
                ],
                common_reconstruction_problem_id=(
                    evidence_payload["common_reconstruction_problem_id"]
                    if schema_version >= 5
                    else None
                ),
                common_reconstructor_id=(
                    evidence_payload["common_reconstructor_id"]
                    if schema_version >= 5
                    else None
                ),
                mismatch_benchmark_id=(
                    evidence_payload["mismatch_benchmark_id"]
                    if schema_version >= 5
                    else None
                ),
                mismatch_generator_id=(
                    evidence_payload["mismatch_generator_id"]
                    if schema_version >= 5
                    else None
                ),
                mismatch_non_nominal_scenario_present_at_summary=(
                    evidence_payload[
                        "mismatch_non_nominal_scenario_present_at_summary"
                    ]
                    if schema_version >= 5
                    else None
                ),
                mismatch_truth_structural_criterion_present_at_summary=(
                    evidence_payload[
                        "mismatch_truth_structural_criterion_present_at_summary"
                    ]
                    if schema_version >= 5
                    else None
                ),
                mismatch_independent_forward_at_summary=(
                    evidence_payload[
                        "mismatch_independent_forward_at_summary"
                    ]
                    if schema_version >= 5
                    else None
                ),
                structurally_trusted_at_summary=evidence_payload[
                    "structurally_trusted_at_summary"
                ],
                trusted_site_count_at_summary=trusted_count,
                typed_evidence_persisted=False,
                structural_trust_reverified_after_load=False,
            )
            trust_flags = _trust_flags_from_json(
                str(data["trust_flags_at_summary_json"].item())
            )
        else:
            site_coordinates = None
            options = None
            scan_partition = None
            evidence_provenance = EnsembleEvidenceProvenance1D(
                source=f"loaded_legacy_archive_v{schema_version}",
                structurally_trusted_at_summary=bool(
                    data["structurally_trusted"].item()
                ),
                trusted_site_count_at_summary=int(
                    np.count_nonzero(archived_site_trusted)
                ),
            )
            trust_flags = _trust_flags_from_json(
                str(data["trust_flags_json"].item())
            )
        trust_flags["archive_typed_evidence_persisted"] = False
        trust_flags["archive_structural_trust_reverified"] = False
        return LatticeSiteEnsemble1D(
            runs=runs,
            accepted_mask=accepted_mask,
            accepted_loss_cutoff=float(data["accepted_loss_cutoff"].item()),
            representative_index=representative_index,
            consensus=consensus,
            rigid_median=np.asarray(data["rigid_median"]),
            rigid_q05=np.asarray(data["rigid_q05"]),
            rigid_q95=np.asarray(data["rigid_q95"]),
            rigid_radial_q90_A=float(data["rigid_radial_q90_A"].item()),
            trust_flags=trust_flags,
            optimizer_stable=bool(data["optimizer_stable"].item()),
            structurally_trusted=False,
            site_coordinates=site_coordinates,
            options=options,
            scan_partition=scan_partition,
            evidence_provenance=evidence_provenance,
        )
