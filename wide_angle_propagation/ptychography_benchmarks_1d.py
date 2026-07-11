"""Reproducible synthetic robustness benchmarks for 1D ptychography.

This module deliberately separates three roles:

* a forward callback receives perturbed *true* inputs and generates noiseless
  expected detector electrons;
* a detector model turns those expectations into calibrated measurements;
* a reconstruction callback receives only the measurement and the unperturbed
  nominal inputs.

Consequently, the benchmark harness does not disclose the simulated mismatch
or truth to the inverse method.  Truth is used only after reconstruction to
compute user-defined accuracy metrics.

The reports do not contain a freely assignable pass/fail boolean.  Acceptance
is always derived from a measured value and a named threshold carrying an
explicit source.  These synthetic tests are evidence about the enumerated
scenarios, not a certificate for untested experimental mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from .ptychography_1d import PtychographyMeasurement1D


__all__ = [
    "BenchmarkCriteria1D",
    "BenchmarkCriterion1D",
    "DetectorMeasurement1D",
    "DetectorPerturbation1D",
    "ForwardModelInputs1D",
    "ForwardModelMismatch1D",
    "ReconstructionBenchmarkOutput1D",
    "ResidualCalibrationEvidence1D",
    "ResidualCalibrationReport1D",
    "ScenarioBenchmarkReport1D",
    "SyntheticBenchmarkReport1D",
    "SyntheticBenchmarkScenario1D",
    "ThresholdEvaluation1D",
    "apply_forward_model_mismatch_1d",
    "generate_detector_measurement_1d",
    "evaluate_residual_calibration_evidence_1d",
    "load_residual_calibration_evidence_1d",
    "load_synthetic_benchmark_report_1d",
    "ptychography_measurement_from_detector_1d",
    "residual_calibration_report_1d",
    "run_synthetic_benchmark_sweep_1d",
    "save_residual_calibration_evidence_1d",
    "save_synthetic_benchmark_report_1d",
    "truth_parameter_error_metrics_1d",
    "validate_benchmark_criteria_1d",
    "validate_detector_perturbation_1d",
    "validate_forward_model_inputs_1d",
    "validate_forward_model_mismatch_1d",
    "validate_residual_calibration_evidence_1d",
    "validate_synthetic_benchmark_report_1d",
]


Array = Any
TruthMetricCallback = Callable[
    [Mapping[str, Array], Mapping[str, Array]], Mapping[str, float]
]
ExpectedSignalCallback = Callable[["ForwardModelInputs1D"], Array]
ReconstructionCallback = Callable[
    ["DetectorMeasurement1D", "ForwardModelInputs1D"],
    "ReconstructionBenchmarkOutput1D",
]


@dataclass(frozen=True)
class DetectorPerturbation1D:
    """True detector response and the calibration used by reconstruction.

    Signal and dark current are Poisson distributed in electron units.  Read
    noise is added in electron units before conversion to ADU.  Values prefixed
    by ``calibrated_`` are the quantities used to convert the raw image and to
    form residual variances; setting them differently from the true values
    creates a controlled calibration mismatch.
    """

    detection_efficiency: float = 1.0
    gain_adu_per_electron: float = 1.0
    calibrated_gain_adu_per_electron: float = 1.0
    dark_electrons_per_pixel: float = 0.0
    calibrated_dark_electrons_per_pixel: float = 0.0
    read_noise_std_electrons: float = 0.0
    calibrated_read_noise_std_electrons: float = 0.0
    saturation_electrons: float | None = None
    masked_detector_indices: tuple[int, ...] = ()
    calibration_id: str = "synthetic_nominal"


@dataclass(frozen=True)
class ForwardModelMismatch1D:
    """Controlled deviations of the data-generating model from the nominal one."""

    probe_amplitude_scale: float = 1.0
    probe_position_offset_A: float = 0.0
    scan_position_offset_A: float = 0.0
    scan_jitter_std_A: float = 0.0
    incidence_angle_offset_rad: float = 0.0
    detector_angle_offset_rad: float = 0.0
    energy_scale: float = 1.0
    potential_scale: float = 1.0
    template_scale: float = 1.0
    template_width_scale: float = 1.0
    template_cutoff_A: float | None = None


@dataclass(frozen=True)
class SyntheticBenchmarkScenario1D:
    """One reproducible detector and forward-model perturbation scenario."""

    scenario_id: str
    seed: int
    detector: DetectorPerturbation1D = field(
        default_factory=DetectorPerturbation1D
    )
    forward_mismatch: ForwardModelMismatch1D = field(
        default_factory=ForwardModelMismatch1D
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ForwardModelInputs1D:
    """Small generic input bundle used at the benchmark callback boundary."""

    probe: Array
    probe_sampling_A: float
    scan_coordinates_A: Array
    detector_angles_rad: Array
    energy_eV: float
    incidence_angle_rad: float | None = None
    potential: Array | None = None
    template: Array | None = None
    template_sampling_A: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DetectorMeasurement1D:
    """Calibrated detector data exposed to a reconstruction callback.

    This type intentionally omits the true detector parameters.  ``valid_mask``
    excludes explicitly masked and saturated pixels.  A reconstruction must
    use that mask rather than interpret clipped values as observations.
    """

    raw_adu: Array
    calibrated_signal_electrons: Array
    valid_mask: Array
    saturated_mask: Array
    masked_mask: Array
    calibrated_gain_adu_per_electron: float
    calibrated_dark_electrons_per_pixel: float
    calibrated_read_noise_std_electrons: float
    calibration_id: str
    detector_seed: int


@dataclass(frozen=True)
class ReconstructionBenchmarkOutput1D:
    """Minimal truth-free output required from a reconstruction callback.

    ``predicted_signal_electrons`` must be expressed on the calibrated signal
    scale of ``DetectorMeasurement1D`` (after the declared dark subtraction).
    """

    predicted_signal_electrons: Array
    estimated_parameters: Mapping[str, Array]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResidualCalibrationReport1D:
    """Truth-free residual calibration diagnostics on valid detector pixels."""

    valid_pixel_count: int
    standardized_residual_mean: float
    standardized_residual_std: float
    standardized_residual_rms: float
    standardized_residual_q05: float
    standardized_residual_q50: float
    standardized_residual_q95: float
    coverage_1sigma: float
    coverage_2sigma: float
    coverage_1sigma_error: float
    coverage_2sigma_error: float
    poisson_deviance_per_valid_pixel: float | None
    poisson_deviance_model: str
    standardized_variance_model: str
    calibration_id: str


@dataclass(frozen=True)
class ResidualCalibrationEvidence1D:
    """Held-out residual evidence bound to data, policy, and inverse problem.

    ``measurement_sha256`` covers all measurement arrays and the declared
    numeric detector calibration used by the residual calculation.
    ``prediction_sha256`` covers the predicted signal evaluated against those
    measurements.  The held-out scan indices identify the leading rows of both
    arrays; they are evidence provenance and are never used for reconstruction
    selection by this module.

    Acceptance is deliberately not a stored field.  It is derived from the
    complete, sourced threshold evaluations through :attr:`passed`.
    """

    residual_calibration: ResidualCalibrationReport1D
    criteria: BenchmarkCriteria1D
    threshold_evaluations: tuple[ThresholdEvaluation1D, ...]
    held_out_scan_indices: tuple[int, ...]
    measurement_shape: tuple[int, int]
    calibration_id: str
    measurement_sha256: str
    prediction_sha256: str
    reconstruction_problem_id: str
    minimum_variance_electrons2: float
    evaluator_id: str

    @property
    def passed(self) -> bool:
        """Return the conjunction of the residual-only sourced evaluations."""
        return bool(self.threshold_evaluations) and all(
            evaluation.passed for evaluation in self.threshold_evaluations
        )


@dataclass(frozen=True)
class BenchmarkCriterion1D:
    """A named acceptance interval with explicit threshold provenance."""

    criterion_id: str
    metric_name: str
    threshold_source: str
    lower_bound: float | None = None
    upper_bound: float | None = None


@dataclass(frozen=True)
class BenchmarkCriteria1D:
    """User-selected acceptance policy for a benchmark sweep."""

    criteria_id: str
    criteria: tuple[BenchmarkCriterion1D, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThresholdEvaluation1D:
    """One observed metric tied to its complete acceptance criterion."""

    criterion: BenchmarkCriterion1D
    observed_value: float
    scenario_id: str

    @property
    def passed(self) -> bool:
        """Derive acceptance from evidence and sourced bounds."""
        lower_ok = (
            self.criterion.lower_bound is None
            or self.observed_value >= self.criterion.lower_bound
        )
        upper_ok = (
            self.criterion.upper_bound is None
            or self.observed_value <= self.criterion.upper_bound
        )
        return bool(lower_ok and upper_ok)


@dataclass(frozen=True)
class ScenarioBenchmarkReport1D:
    """Metrics and reproducibility evidence for one synthetic scenario."""

    scenario: SyntheticBenchmarkScenario1D
    metrics: Mapping[str, float]
    residual_calibration: ResidualCalibrationReport1D
    threshold_evaluations: tuple[ThresholdEvaluation1D, ...]
    mismatch_seed: int
    detector_seed: int
    measurement_shape: tuple[int, ...]
    valid_pixel_count: int
    masked_pixel_count: int
    saturated_pixel_count: int
    generated_signal_sha256: str
    measurement_sha256: str
    prediction_sha256: str
    estimated_parameters_sha256: str
    perturbed_inputs_sha256: str
    reconstruction_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return the conjunction of the stored, sourced evaluations."""
        return bool(self.threshold_evaluations) and all(
            evaluation.passed for evaluation in self.threshold_evaluations
        )


@dataclass(frozen=True)
class SyntheticBenchmarkReport1D:
    """Truth-aware results for an explicitly enumerated robustness sweep."""

    benchmark_id: str
    truth_id: str
    generator_id: str
    reconstructor_id: str
    truth_metric_id: str
    criteria: BenchmarkCriteria1D
    scenarios: tuple[ScenarioBenchmarkReport1D, ...]
    worst_case_evaluations: tuple[ThresholdEvaluation1D, ...]
    nominal_inputs_sha256: str
    truth_sha256: str
    rng_algorithm: str
    numpy_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Derive the aggregate gate from worst-case metric evaluations."""
        return bool(self.worst_case_evaluations) and all(
            evaluation.passed for evaluation in self.worst_case_evaluations
        )


_EXPECTED_COVERAGE_1SIGMA = 0.6826894921370859
_EXPECTED_COVERAGE_2SIGMA = 0.9544997361036416
_RESIDUAL_EVIDENCE_EVALUATOR_ID = "residual_calibration_evidence_1d:v1"
_RESIDUAL_EVIDENCE_SCHEMA_VERSION = 1
_RESIDUAL_EVIDENCE_ARCHIVE_KEYS = frozenset(
    {"schema_version", "payload_json", "payload_sha256"}
)
_REPORT_SCHEMA_VERSION = 1
_RNG_ALGORITHM = "numpy.default_rng.PCG64;SeedSequence.spawn(2)"
_REPORT_ARCHIVE_KEYS = frozenset(
    {"schema_version", "payload_json", "payload_sha256"}
)


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _finite_scalar(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or np.issubdtype(array.dtype, np.bool_)
        or np.iscomplexobj(array)
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise TypeError(f"{name} must be a real numeric scalar")
    resolved = float(array)
    if not np.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    if positive and resolved <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and resolved < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


def _integer_seed(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    resolved = int(value)
    if resolved < 0 or resolved > np.iinfo(np.uint64).max:
        raise ValueError(f"{name} must lie in the uint64 range")
    return resolved


def _numeric_array(
    name: str,
    value: Any,
    *,
    ndim: int | None = None,
    nonnegative: bool = False,
    allow_complex: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must be a numeric array")
    if np.iscomplexobj(array) and not allow_complex:
        raise TypeError(f"{name} must be real-valued")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _json_safe(value: Any, *, path: str = "metadata") -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return int(value)
    if isinstance(value, (float, np.floating)):
        resolved = float(value)
        if not np.isfinite(resolved):
            raise ValueError(f"{path} contains a non-finite value")
        return resolved
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), path=path)
    if isinstance(value, (tuple, list)):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            result[key] = _json_safe(item, path=f"{path}.{key}")
        return result
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def _array_digest(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _mapping_digest(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        _nonempty_string("mapping key", name)
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(_array_digest(values[name])))
    return digest.hexdigest()


def _inputs_digest(inputs: ForwardModelInputs1D) -> str:
    arrays: list[Any] = [
        inputs.probe,
        inputs.scan_coordinates_A,
        inputs.detector_angles_rad,
        np.asarray(
            [inputs.probe_sampling_A, inputs.energy_eV], dtype=np.float64
        ),
        np.asarray(
            [
                np.nan
                if inputs.incidence_angle_rad is None
                else inputs.incidence_angle_rad
            ],
            dtype=np.float64,
        ),
    ]
    for optional in (inputs.potential, inputs.template):
        if optional is None:
            arrays.append(np.asarray([], dtype=np.float64))
        else:
            arrays.append(optional)
    arrays.append(
        np.asarray(
            []
            if inputs.template_sampling_A is None
            else inputs.template_sampling_A,
            dtype=np.float64,
        )
    )
    return _array_digest(*arrays)


def validate_detector_perturbation_1d(
    detector: DetectorPerturbation1D,
) -> None:
    """Validate a detector perturbation without silently repairing it."""
    if not isinstance(detector, DetectorPerturbation1D):
        raise TypeError("detector must be a DetectorPerturbation1D")
    efficiency = _finite_scalar(
        "detector.detection_efficiency",
        detector.detection_efficiency,
        positive=True,
    )
    if efficiency > 1.0:
        raise ValueError("detector.detection_efficiency must not exceed one")
    for name in (
        "gain_adu_per_electron",
        "calibrated_gain_adu_per_electron",
    ):
        _finite_scalar(
            f"detector.{name}", getattr(detector, name), positive=True
        )
    for name in (
        "dark_electrons_per_pixel",
        "calibrated_dark_electrons_per_pixel",
        "read_noise_std_electrons",
        "calibrated_read_noise_std_electrons",
    ):
        _finite_scalar(
            f"detector.{name}", getattr(detector, name), nonnegative=True
        )
    if detector.saturation_electrons is not None:
        _finite_scalar(
            "detector.saturation_electrons",
            detector.saturation_electrons,
            positive=True,
        )
    indices = detector.masked_detector_indices
    if not isinstance(indices, tuple):
        raise TypeError("detector.masked_detector_indices must be a tuple")
    normalized = []
    for index in indices:
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("masked detector indices must be integers")
        if int(index) < 0:
            raise ValueError("masked detector indices must be non-negative")
        normalized.append(int(index))
    if len(set(normalized)) != len(normalized):
        raise ValueError("masked detector indices must be unique")
    _nonempty_string("detector.calibration_id", detector.calibration_id)


def validate_forward_model_mismatch_1d(
    mismatch: ForwardModelMismatch1D,
) -> None:
    """Validate the finite, physically signed mismatch controls."""
    if not isinstance(mismatch, ForwardModelMismatch1D):
        raise TypeError("mismatch must be a ForwardModelMismatch1D")
    for name in (
        "probe_amplitude_scale",
        "energy_scale",
        "potential_scale",
        "template_scale",
        "template_width_scale",
    ):
        _finite_scalar(name, getattr(mismatch, name), positive=True)
    if mismatch.template_cutoff_A is not None:
        _finite_scalar(
            "template_cutoff_A", mismatch.template_cutoff_A, positive=True
        )
    _finite_scalar(
        "scan_jitter_std_A", mismatch.scan_jitter_std_A, nonnegative=True
    )
    for name in (
        "probe_position_offset_A",
        "scan_position_offset_A",
        "incidence_angle_offset_rad",
        "detector_angle_offset_rad",
    ):
        _finite_scalar(name, getattr(mismatch, name))


def validate_forward_model_inputs_1d(inputs: ForwardModelInputs1D) -> None:
    """Validate nominal or perturbed callback inputs."""
    if not isinstance(inputs, ForwardModelInputs1D):
        raise TypeError("inputs must be a ForwardModelInputs1D")
    probe = _numeric_array(
        "inputs.probe", inputs.probe, ndim=1, allow_complex=True
    )
    scans = _numeric_array(
        "inputs.scan_coordinates_A", inputs.scan_coordinates_A, ndim=1
    )
    angles = _numeric_array(
        "inputs.detector_angles_rad", inputs.detector_angles_rad, ndim=1
    )
    if probe.size < 2:
        raise ValueError("inputs.probe must contain at least two samples")
    if not scans.size or not angles.size:
        raise ValueError("scan coordinates and detector angles must not be empty")
    _finite_scalar(
        "inputs.probe_sampling_A", inputs.probe_sampling_A, positive=True
    )
    _finite_scalar("inputs.energy_eV", inputs.energy_eV, positive=True)
    if inputs.incidence_angle_rad is not None:
        _finite_scalar(
            "inputs.incidence_angle_rad", inputs.incidence_angle_rad
        )
    for name in ("potential", "template"):
        value = getattr(inputs, name)
        if value is not None:
            array = _numeric_array(f"inputs.{name}", value)
            if array.size == 0 or array.ndim == 0:
                raise ValueError(
                    f"inputs.{name} must be a non-scalar, non-empty array"
                )
    sampling = inputs.template_sampling_A
    if sampling is not None:
        if inputs.template is None:
            raise ValueError("template_sampling_A requires a template")
        if not isinstance(sampling, tuple):
            raise TypeError("template_sampling_A must be a tuple")
        template_ndim = np.asarray(inputs.template).ndim
        if len(sampling) != template_ndim:
            raise ValueError(
                "template_sampling_A must contain one value per template axis"
            )
        for value in sampling:
            _finite_scalar("template_sampling_A value", value, positive=True)


def _shift_probe_with_zero_boundary(
    probe: np.ndarray,
    sampling_A: float,
    offset_A: float,
) -> np.ndarray:
    coordinate = np.arange(probe.size, dtype=float) * sampling_A
    source_coordinate = coordinate - offset_A
    if np.iscomplexobj(probe):
        return np.interp(
            source_coordinate,
            coordinate,
            np.real(probe),
            left=0.0,
            right=0.0,
        ) + 1j * np.interp(
            source_coordinate,
            coordinate,
            np.imag(probe),
            left=0.0,
            right=0.0,
        )
    return np.interp(
        source_coordinate,
        coordinate,
        probe,
        left=0.0,
        right=0.0,
    )


def _rescale_array_about_center(
    array: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Linearly resample every array axis about its geometric center."""
    result = np.asarray(array, dtype=float)
    if scale == 1.0:
        return result.copy()
    for axis, length in enumerate(result.shape):
        coordinate = np.arange(length, dtype=float)
        center = 0.5 * (length - 1)
        source = center + (coordinate - center) / scale
        moved = np.moveaxis(result, axis, -1)
        flat = moved.reshape(-1, length)
        interpolated = np.stack(
            [
                np.interp(
                    source,
                    coordinate,
                    row,
                    left=0.0,
                    right=0.0,
                )
                for row in flat
            ]
        )
        result = np.moveaxis(interpolated.reshape(moved.shape), -1, axis)
    return result


def _apply_template_cutoff(
    template: np.ndarray,
    sampling_A: tuple[float, ...],
    cutoff_A: float,
) -> np.ndarray:
    radius_squared = np.zeros(template.shape, dtype=float)
    for axis, (length, sampling) in enumerate(
        zip(template.shape, sampling_A, strict=True)
    ):
        coordinate = (np.arange(length) - 0.5 * (length - 1)) * sampling
        shape = [1] * template.ndim
        shape[axis] = length
        radius_squared += coordinate.reshape(shape) ** 2
    return np.where(radius_squared <= cutoff_A**2, template, 0.0)


def apply_forward_model_mismatch_1d(
    nominal_inputs: ForwardModelInputs1D,
    mismatch: ForwardModelMismatch1D,
    *,
    seed: int,
) -> ForwardModelInputs1D:
    """Create reproducible data-generating inputs from nominal inputs.

    Positive probe offsets move the sampled probe toward increasing array
    coordinate and use zero boundary values.  Scan jitter is independent
    Gaussian jitter.  Scaling a missing potential or template is rejected when
    the requested scale differs from one, avoiding a silently ineffective
    benchmark scenario.
    """
    validate_forward_model_inputs_1d(nominal_inputs)
    validate_forward_model_mismatch_1d(mismatch)
    resolved_seed = _integer_seed("seed", seed)
    if (
        nominal_inputs.incidence_angle_rad is None
        and mismatch.incidence_angle_offset_rad != 0.0
    ):
        raise ValueError(
            "incidence_angle_offset_rad requires a nominal incidence angle"
        )
    if nominal_inputs.potential is None and mismatch.potential_scale != 1.0:
        raise ValueError("potential_scale requires a nominal potential")
    if nominal_inputs.template is None and mismatch.template_scale != 1.0:
        raise ValueError("template_scale requires a nominal template")
    if (
        nominal_inputs.template is None
        and mismatch.template_width_scale != 1.0
    ):
        raise ValueError("template_width_scale requires a nominal template")
    if mismatch.template_cutoff_A is not None:
        if nominal_inputs.template is None:
            raise ValueError("template_cutoff_A requires a nominal template")
        if nominal_inputs.template_sampling_A is None:
            raise ValueError(
                "template_cutoff_A requires nominal template sampling"
            )

    rng = np.random.default_rng(resolved_seed)
    probe = _shift_probe_with_zero_boundary(
        np.asarray(nominal_inputs.probe),
        float(nominal_inputs.probe_sampling_A),
        float(mismatch.probe_position_offset_A),
    ) * float(mismatch.probe_amplitude_scale)
    scans = np.asarray(nominal_inputs.scan_coordinates_A, dtype=float).copy()
    scans += float(mismatch.scan_position_offset_A)
    if mismatch.scan_jitter_std_A:
        scans += rng.normal(
            0.0,
            float(mismatch.scan_jitter_std_A),
            size=scans.shape,
        )
    angles = np.asarray(nominal_inputs.detector_angles_rad, dtype=float)
    angles = angles + float(mismatch.detector_angle_offset_rad)
    incidence = nominal_inputs.incidence_angle_rad
    if incidence is not None:
        incidence = float(incidence) + float(mismatch.incidence_angle_offset_rad)
    potential = (
        None
        if nominal_inputs.potential is None
        else np.asarray(nominal_inputs.potential) * mismatch.potential_scale
    )
    template = None
    if nominal_inputs.template is not None:
        template = _rescale_array_about_center(
            np.asarray(nominal_inputs.template),
            float(mismatch.template_width_scale),
        )
        template *= mismatch.template_scale
        if mismatch.template_cutoff_A is not None:
            template = _apply_template_cutoff(
                template,
                nominal_inputs.template_sampling_A,
                float(mismatch.template_cutoff_A),
            )
    perturbed = ForwardModelInputs1D(
        probe=probe,
        probe_sampling_A=float(nominal_inputs.probe_sampling_A),
        scan_coordinates_A=scans,
        detector_angles_rad=angles,
        energy_eV=float(nominal_inputs.energy_eV) * mismatch.energy_scale,
        incidence_angle_rad=incidence,
        potential=potential,
        template=template,
        template_sampling_A=nominal_inputs.template_sampling_A,
    )
    validate_forward_model_inputs_1d(perturbed)
    return perturbed


def generate_detector_measurement_1d(
    expected_signal_electrons: Array,
    detector: DetectorPerturbation1D,
    *,
    seed: int,
) -> DetectorMeasurement1D:
    """Draw calibrated Poisson/counting-detector data reproducibly.

    The input is the expected signal before detection efficiency and excludes
    detector dark current.  The returned valid mask is false at user-masked and
    saturated pixels.  Mask indices address the final (detector) array axis.
    """
    validate_detector_perturbation_1d(detector)
    resolved_seed = _integer_seed("seed", seed)
    expected = _numeric_array(
        "expected_signal_electrons",
        expected_signal_electrons,
        nonnegative=True,
    ).astype(float, copy=False)
    if expected.ndim < 1 or expected.size == 0:
        raise ValueError("expected_signal_electrons must be a non-empty array")
    n_detector = expected.shape[-1]
    indices = np.asarray(detector.masked_detector_indices, dtype=np.int64)
    if indices.size and np.any(indices >= n_detector):
        raise ValueError("masked detector index lies outside the detector axis")

    rng = np.random.default_rng(resolved_seed)
    poisson_mean = (
        detector.detection_efficiency * expected
        + detector.dark_electrons_per_pixel
    )
    photoelectrons = rng.poisson(poisson_mean).astype(float)
    if detector.read_noise_std_electrons:
        analogue_electrons = photoelectrons + rng.normal(
            0.0,
            detector.read_noise_std_electrons,
            size=expected.shape,
        )
    else:
        analogue_electrons = photoelectrons

    saturated = np.zeros(expected.shape, dtype=bool)
    if detector.saturation_electrons is not None:
        saturated = analogue_electrons >= detector.saturation_electrons
        analogue_electrons = np.minimum(
            analogue_electrons, detector.saturation_electrons
        )
    masked = np.zeros(expected.shape, dtype=bool)
    if indices.size:
        masked[..., indices] = True
    valid = ~(masked | saturated)
    if not np.any(valid):
        raise ValueError("detector perturbation leaves no valid pixels")

    raw_adu = analogue_electrons * detector.gain_adu_per_electron
    calibrated_signal = (
        raw_adu / detector.calibrated_gain_adu_per_electron
        - detector.calibrated_dark_electrons_per_pixel
    )
    return DetectorMeasurement1D(
        raw_adu=raw_adu,
        calibrated_signal_electrons=calibrated_signal,
        valid_mask=valid,
        saturated_mask=saturated,
        masked_mask=masked,
        calibrated_gain_adu_per_electron=float(
            detector.calibrated_gain_adu_per_electron
        ),
        calibrated_dark_electrons_per_pixel=float(
            detector.calibrated_dark_electrons_per_pixel
        ),
        calibrated_read_noise_std_electrons=float(
            detector.calibrated_read_noise_std_electrons
        ),
        calibration_id=detector.calibration_id,
        detector_seed=resolved_seed,
    )


def _validate_measurement(measurement: DetectorMeasurement1D) -> None:
    if not isinstance(measurement, DetectorMeasurement1D):
        raise TypeError("measurement must be a DetectorMeasurement1D")
    raw = _numeric_array("measurement.raw_adu", measurement.raw_adu)
    calibrated = _numeric_array(
        "measurement.calibrated_signal_electrons",
        measurement.calibrated_signal_electrons,
    )
    if raw.shape != calibrated.shape or raw.size == 0:
        raise ValueError("measurement arrays must have one common non-empty shape")
    masks = []
    for name in ("valid_mask", "saturated_mask", "masked_mask"):
        mask = np.asarray(getattr(measurement, name))
        if mask.dtype != np.bool_ or mask.shape != raw.shape:
            raise ValueError(f"measurement.{name} must be a matching bool array")
        masks.append(mask)
    valid, saturated, masked = masks
    if np.any(valid & (saturated | masked)):
        raise ValueError("measurement valid mask includes excluded pixels")
    if not np.array_equal(valid, ~(saturated | masked)):
        raise ValueError("measurement masks are internally inconsistent")
    _finite_scalar(
        "measurement.calibrated_gain_adu_per_electron",
        measurement.calibrated_gain_adu_per_electron,
        positive=True,
    )
    _finite_scalar(
        "measurement.calibrated_dark_electrons_per_pixel",
        measurement.calibrated_dark_electrons_per_pixel,
        nonnegative=True,
    )
    _finite_scalar(
        "measurement.calibrated_read_noise_std_electrons",
        measurement.calibrated_read_noise_std_electrons,
        nonnegative=True,
    )
    _nonempty_string("measurement.calibration_id", measurement.calibration_id)
    _integer_seed("measurement.detector_seed", measurement.detector_seed)
    expected_calibrated = (
        raw / float(measurement.calibrated_gain_adu_per_electron)
        - float(measurement.calibrated_dark_electrons_per_pixel)
    )
    dtype = np.result_type(raw.dtype, calibrated.dtype, np.float32)
    epsilon = np.finfo(dtype).eps
    scale = max(
        1.0,
        float(np.max(np.abs(calibrated[valid]))),
        float(np.max(np.abs(expected_calibrated[valid]))),
    )
    if not np.allclose(
        calibrated[valid],
        expected_calibrated[valid],
        rtol=64.0 * epsilon,
        atol=64.0 * epsilon * scale,
    ):
        raise ValueError(
            "measurement calibrated signal is inconsistent with raw ADU, "
            "declared gain, and declared dark on valid pixels"
        )


def ptychography_measurement_from_detector_1d(
    measurement: DetectorMeasurement1D,
) -> PtychographyMeasurement1D:
    """Expose only calibrated, truth-free detector data to reconstruction.

    The benchmark detector object contains raw ADU and a reproducibility seed
    in addition to its calibrated observations.  Neither is copied as
    provenance: raw ADU is used only to derive observed electrons under the
    *declared calibration*, and the simulator seed has no place in the inverse
    problem.  In particular, no true detector perturbation, benchmark
    scenario, perturbed forward input, or reconstruction truth is accepted by
    this adapter.
    """
    from .ptychography_1d import PtychographyMeasurement1D

    _validate_measurement(measurement)
    calibrated_signal = np.array(
        measurement.calibrated_signal_electrons,
        copy=True,
    )
    observed_total = np.asarray(measurement.raw_adu, dtype=float) / float(
        measurement.calibrated_gain_adu_per_electron
    )
    valid_mask = np.array(measurement.valid_mask, dtype=bool, copy=True)
    return PtychographyMeasurement1D(
        calibrated_signal_electrons=calibrated_signal,
        observed_total_electrons=np.array(observed_total, copy=True),
        valid_mask=valid_mask,
        calibrated_dark_electrons_per_pixel=float(
            measurement.calibrated_dark_electrons_per_pixel
        ),
        calibrated_read_noise_std_electrons=float(
            measurement.calibrated_read_noise_std_electrons
        ),
        calibration_id=measurement.calibration_id,
        metadata={
            "source_type": "DetectorMeasurement1D",
            "adapter_schema": (
                "wide_angle_propagation."
                "ptychography_measurement_from_detector_1d:v1"
            ),
            "total_observation_semantics": "calibrated_electron_equivalent",
            "integer_count_contract": False,
        },
    )


def _measurement_digest_1d(measurement: DetectorMeasurement1D) -> str:
    """Digest every array and numeric calibration affecting residuals."""
    return _array_digest(
        measurement.raw_adu,
        measurement.calibrated_signal_electrons,
        measurement.valid_mask,
        measurement.saturated_mask,
        measurement.masked_mask,
        np.asarray(
            [
                measurement.calibrated_gain_adu_per_electron,
                measurement.calibrated_dark_electrons_per_pixel,
                measurement.calibrated_read_noise_std_electrons,
            ],
            dtype=np.float64,
        ),
        np.asarray(measurement.detector_seed, dtype=np.uint64),
    )


def residual_calibration_report_1d(
    measurement: DetectorMeasurement1D,
    predicted_signal_electrons: Array,
    *,
    minimum_variance_electrons2: float = 1e-9,
) -> ResidualCalibrationReport1D:
    """Compute standardized-residual coverage and optional Poisson deviance.

    Poisson deviance is reported only when the calibration model declares zero
    read noise.  Standardized residuals use the calibrated Poisson-plus-read
    variance and remain available for all detector scenarios.  At least two
    valid pixels are required so the sample standard deviation is meaningful.
    """
    _validate_measurement(measurement)
    prediction = _numeric_array(
        "predicted_signal_electrons",
        predicted_signal_electrons,
        nonnegative=True,
    ).astype(float, copy=False)
    observed = np.asarray(measurement.calibrated_signal_electrons, dtype=float)
    if prediction.shape != observed.shape:
        raise ValueError("prediction and measurement shapes differ")
    floor = _finite_scalar(
        "minimum_variance_electrons2",
        minimum_variance_electrons2,
        positive=True,
    )
    valid = np.asarray(measurement.valid_mask)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 2:
        raise ValueError("at least two valid detector pixels are required")
    predicted_total = (
        prediction + measurement.calibrated_dark_electrons_per_pixel
    )
    variance = np.maximum(predicted_total, floor) + (
        measurement.calibrated_read_noise_std_electrons**2
    )
    residual = (observed - prediction) / np.sqrt(variance)
    selected = residual[valid]
    coverage_1 = float(np.mean(np.abs(selected) <= 1.0))
    coverage_2 = float(np.mean(np.abs(selected) <= 2.0))

    deviance: float | None
    deviance_model: str
    if measurement.calibrated_read_noise_std_electrons == 0.0:
        observed_total = (
            np.asarray(measurement.raw_adu, dtype=float)
            / measurement.calibrated_gain_adu_per_electron
        )[valid]
        mean_total = predicted_total[valid]
        mean_total = np.maximum(mean_total, floor)
        if np.any(observed_total < 0.0):
            deviance = None
            deviance_model = "not_applicable_negative_calibrated_counts"
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_term = np.where(
                    observed_total > 0.0,
                    observed_total * np.log(observed_total / mean_total),
                    0.0,
                )
            deviance = float(
                np.sum(2.0 * (log_term - (observed_total - mean_total)))
                / valid_count
            )
            deviance_model = (
                "poisson_deviance_on_electron_equivalents_under_"
                "declared_calibration"
            )
    else:
        deviance = None
        deviance_model = "not_applicable_poisson_plus_read_noise"

    quantiles = np.quantile(selected, [0.05, 0.5, 0.95])
    return ResidualCalibrationReport1D(
        valid_pixel_count=valid_count,
        standardized_residual_mean=float(np.mean(selected)),
        standardized_residual_std=float(np.std(selected, ddof=1)),
        standardized_residual_rms=float(np.sqrt(np.mean(selected**2))),
        standardized_residual_q05=float(quantiles[0]),
        standardized_residual_q50=float(quantiles[1]),
        standardized_residual_q95=float(quantiles[2]),
        coverage_1sigma=coverage_1,
        coverage_2sigma=coverage_2,
        coverage_1sigma_error=abs(coverage_1 - _EXPECTED_COVERAGE_1SIGMA),
        coverage_2sigma_error=abs(coverage_2 - _EXPECTED_COVERAGE_2SIGMA),
        poisson_deviance_per_valid_pixel=deviance,
        poisson_deviance_model=deviance_model,
        standardized_variance_model=(
            "declared_poisson_plus_gaussian_read_noise"
        ),
        calibration_id=measurement.calibration_id,
    )


def truth_parameter_error_metrics_1d(
    truth: Mapping[str, Array],
    estimates: Mapping[str, Array],
) -> Mapping[str, float]:
    """Compute RMSE, MAE, and maximum error for matching named arrays."""
    if not isinstance(truth, Mapping) or not isinstance(estimates, Mapping):
        raise TypeError("truth and estimates must be mappings")
    metrics: dict[str, float] = {}
    for name in sorted(truth):
        _nonempty_string("truth parameter name", name)
        if name not in estimates:
            continue
        true_array = _numeric_array(f"truth[{name!r}]", truth[name])
        estimate = _numeric_array(f"estimates[{name!r}]", estimates[name])
        if estimate.shape != true_array.shape:
            raise ValueError(f"estimate {name!r} has the wrong shape")
        difference = np.asarray(estimate, dtype=float) - np.asarray(
            true_array, dtype=float
        )
        metrics[f"{name}.rmse"] = float(np.sqrt(np.mean(difference**2)))
        metrics[f"{name}.mae"] = float(np.mean(np.abs(difference)))
        metrics[f"{name}.maximum_absolute_error"] = float(
            np.max(np.abs(difference))
        )
    if not metrics:
        raise ValueError("truth and estimates have no matching parameter names")
    return metrics


def validate_benchmark_criteria_1d(criteria: BenchmarkCriteria1D) -> None:
    """Require explicit, finite, sourced acceptance thresholds."""
    if not isinstance(criteria, BenchmarkCriteria1D):
        raise TypeError("criteria must be a BenchmarkCriteria1D")
    _nonempty_string("criteria.criteria_id", criteria.criteria_id)
    if not isinstance(criteria.criteria, tuple) or not criteria.criteria:
        raise ValueError("criteria.criteria must be a non-empty tuple")
    identifiers: set[str] = set()
    metric_names: set[str] = set()
    for criterion in criteria.criteria:
        if not isinstance(criterion, BenchmarkCriterion1D):
            raise TypeError("every criterion must be a BenchmarkCriterion1D")
        identifier = _nonempty_string(
            "criterion.criterion_id", criterion.criterion_id
        )
        metric = _nonempty_string("criterion.metric_name", criterion.metric_name)
        _nonempty_string(
            "criterion.threshold_source", criterion.threshold_source
        )
        if identifier in identifiers:
            raise ValueError("criterion identifiers must be unique")
        if metric in metric_names:
            raise ValueError("each metric may have only one criterion")
        identifiers.add(identifier)
        metric_names.add(metric)
        if criterion.lower_bound is None and criterion.upper_bound is None:
            raise ValueError("each criterion requires at least one bound")
        lower = (
            None
            if criterion.lower_bound is None
            else _finite_scalar("criterion.lower_bound", criterion.lower_bound)
        )
        upper = (
            None
            if criterion.upper_bound is None
            else _finite_scalar("criterion.upper_bound", criterion.upper_bound)
        )
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("criterion lower bound exceeds upper bound")
    _json_safe(criteria.metadata, path="criteria.metadata")


def _validate_scenario(scenario: SyntheticBenchmarkScenario1D) -> None:
    if not isinstance(scenario, SyntheticBenchmarkScenario1D):
        raise TypeError("scenarios must contain SyntheticBenchmarkScenario1D")
    _nonempty_string("scenario.scenario_id", scenario.scenario_id)
    _integer_seed("scenario.seed", scenario.seed)
    validate_detector_perturbation_1d(scenario.detector)
    validate_forward_model_mismatch_1d(scenario.forward_mismatch)
    _json_safe(scenario.metadata, path="scenario.metadata")


def _finite_metric_mapping(
    name: str, metrics: Mapping[str, Any]
) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized: dict[str, float] = {}
    for metric_name, value in metrics.items():
        _nonempty_string(f"{name} metric name", metric_name)
        if metric_name in normalized:
            raise ValueError(f"{name} contains a duplicate metric")
        normalized[metric_name] = _finite_scalar(
            f"{name}[{metric_name!r}]", value
        )
    return normalized


def _threshold_evaluations(
    scenario_id: str,
    metrics: Mapping[str, float],
    criteria: BenchmarkCriteria1D,
) -> tuple[ThresholdEvaluation1D, ...]:
    evaluations = []
    for criterion in criteria.criteria:
        if criterion.metric_name not in metrics:
            raise ValueError(
                f"criterion metric {criterion.metric_name!r} is unavailable "
                f"for scenario {scenario_id!r}"
            )
        evaluations.append(
            ThresholdEvaluation1D(
                criterion=criterion,
                observed_value=float(metrics[criterion.metric_name]),
                scenario_id=scenario_id,
            )
        )
    return tuple(evaluations)


def _worst_evaluation(
    evaluations: Sequence[ThresholdEvaluation1D],
) -> ThresholdEvaluation1D:
    criterion = evaluations[0].criterion
    lower = criterion.lower_bound
    upper = criterion.upper_bound
    values = np.asarray([item.observed_value for item in evaluations])
    if lower is None:
        index = int(np.argmax(values))
    elif upper is None:
        index = int(np.argmin(values))
    else:
        scale = max(float(upper - lower), abs(lower), abs(upper), 1.0)
        scores = np.maximum((lower - values) / scale, (values - upper) / scale)
        index = int(np.argmax(scores))
    return evaluations[index]


def _spawn_uint64_seeds(seed: int) -> tuple[int, int]:
    children = np.random.SeedSequence(seed).spawn(2)
    return tuple(
        int(child.generate_state(1, dtype=np.uint64)[0]) for child in children
    )


def _residual_metrics(
    report: ResidualCalibrationReport1D,
) -> dict[str, float]:
    metrics = {
        "residual.standardized_mean": report.standardized_residual_mean,
        "residual.standardized_mean_abs": abs(
            report.standardized_residual_mean
        ),
        "residual.standardized_std": report.standardized_residual_std,
        "residual.standardized_std_error": abs(
            report.standardized_residual_std - 1.0
        ),
        "residual.standardized_rms": report.standardized_residual_rms,
        "residual.coverage_1sigma": report.coverage_1sigma,
        "residual.coverage_2sigma": report.coverage_2sigma,
        "residual.coverage_1sigma_error": report.coverage_1sigma_error,
        "residual.coverage_2sigma_error": report.coverage_2sigma_error,
    }
    if report.poisson_deviance_per_valid_pixel is not None:
        metrics["residual.poisson_deviance_per_valid_pixel"] = (
            report.poisson_deviance_per_valid_pixel
        )
    return metrics


def run_synthetic_benchmark_sweep_1d(
    nominal_inputs: ForwardModelInputs1D,
    truth_parameters: Mapping[str, Array],
    scenarios: Sequence[SyntheticBenchmarkScenario1D],
    expected_signal_callback: ExpectedSignalCallback,
    reconstruction_callback: ReconstructionCallback,
    *,
    criteria: BenchmarkCriteria1D,
    benchmark_id: str,
    truth_id: str,
    generator_id: str,
    reconstructor_id: str,
    truth_metric_callback: TruthMetricCallback = truth_parameter_error_metrics_1d,
    truth_metric_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> SyntheticBenchmarkReport1D:
    """Run a truth-isolated detector and forward-mismatch benchmark sweep.

    ``expected_signal_callback`` receives perturbed true inputs.
    ``reconstruction_callback`` receives the calibrated measurement and the
    original nominal inputs only.  Truth is passed solely to
    ``truth_metric_callback`` after reconstruction.
    """
    validate_forward_model_inputs_1d(nominal_inputs)
    validate_benchmark_criteria_1d(criteria)
    for name, value in (
        ("benchmark_id", benchmark_id),
        ("truth_id", truth_id),
        ("generator_id", generator_id),
        ("reconstructor_id", reconstructor_id),
    ):
        _nonempty_string(name, value)
    if not callable(expected_signal_callback):
        raise TypeError("expected_signal_callback must be callable")
    if not callable(reconstruction_callback):
        raise TypeError("reconstruction_callback must be callable")
    if not callable(truth_metric_callback):
        raise TypeError("truth_metric_callback must be callable")
    if truth_metric_id is None:
        if truth_metric_callback is not truth_parameter_error_metrics_1d:
            raise ValueError(
                "truth_metric_id is required for a custom truth metric callback"
            )
        truth_metric_id = "truth_parameter_error_metrics_1d:v1"
    _nonempty_string("truth_metric_id", truth_metric_id)
    if not isinstance(truth_parameters, Mapping) or not truth_parameters:
        raise ValueError("truth_parameters must be a non-empty mapping")
    for name, value in truth_parameters.items():
        _nonempty_string("truth parameter name", name)
        _numeric_array(f"truth_parameters[{name!r}]", value)
    scenario_tuple = tuple(scenarios)
    if not scenario_tuple:
        raise ValueError("at least one benchmark scenario is required")
    scenario_ids: set[str] = set()
    for scenario in scenario_tuple:
        _validate_scenario(scenario)
        if scenario.scenario_id in scenario_ids:
            raise ValueError("scenario identifiers must be unique")
        scenario_ids.add(scenario.scenario_id)
    report_metadata = {} if metadata is None else _json_safe(metadata)

    n_scan = np.asarray(nominal_inputs.scan_coordinates_A).size
    n_detector = np.asarray(nominal_inputs.detector_angles_rad).size
    scenario_reports: list[ScenarioBenchmarkReport1D] = []
    for scenario in scenario_tuple:
        mismatch_seed, detector_seed = _spawn_uint64_seeds(scenario.seed)
        perturbed_inputs = apply_forward_model_mismatch_1d(
            nominal_inputs,
            scenario.forward_mismatch,
            seed=mismatch_seed,
        )
        expected = _numeric_array(
            "expected_signal_callback output",
            expected_signal_callback(perturbed_inputs),
            nonnegative=True,
        ).astype(float, copy=False)
        expected_shape = (n_scan, n_detector)
        if expected.shape != expected_shape:
            raise ValueError(
                "expected_signal_callback must return shape "
                f"{expected_shape}, received {expected.shape}"
            )
        measurement = generate_detector_measurement_1d(
            expected,
            scenario.detector,
            seed=detector_seed,
        )
        output = reconstruction_callback(measurement, nominal_inputs)
        if not isinstance(output, ReconstructionBenchmarkOutput1D):
            raise TypeError(
                "reconstruction_callback must return "
                "ReconstructionBenchmarkOutput1D"
            )
        prediction = _numeric_array(
            "reconstruction predicted_signal_electrons",
            output.predicted_signal_electrons,
            nonnegative=True,
        ).astype(float, copy=False)
        if prediction.shape != expected_shape:
            raise ValueError("reconstruction prediction has the wrong shape")
        if not isinstance(output.estimated_parameters, Mapping):
            raise TypeError("estimated_parameters must be a mapping")
        if not output.estimated_parameters:
            raise ValueError("estimated_parameters must not be empty")
        for parameter_name, value in output.estimated_parameters.items():
            _nonempty_string("estimated parameter name", parameter_name)
            _numeric_array(f"estimated_parameters[{parameter_name!r}]", value)
        reconstruction_metadata = _json_safe(
            output.metadata, path="reconstruction metadata"
        )

        residual = residual_calibration_report_1d(measurement, prediction)
        truth_metrics = _finite_metric_mapping(
            "truth_metric_callback output",
            truth_metric_callback(
                truth_parameters, output.estimated_parameters
            ),
        )
        metrics = _residual_metrics(residual)
        for name, value in truth_metrics.items():
            namespaced = f"truth.{name}"
            if namespaced in metrics:
                raise ValueError("truth metric collides with built-in metric")
            metrics[namespaced] = value
        total_pixels = expected.size
        valid_count = int(np.count_nonzero(measurement.valid_mask))
        masked_count = int(np.count_nonzero(measurement.masked_mask))
        saturated_count = int(np.count_nonzero(measurement.saturated_mask))
        metrics.update(
            {
                "data.valid_fraction": valid_count / total_pixels,
                "data.masked_fraction": masked_count / total_pixels,
                "data.saturated_fraction": saturated_count / total_pixels,
            }
        )
        evaluations = _threshold_evaluations(
            scenario.scenario_id, metrics, criteria
        )
        scenario_reports.append(
            ScenarioBenchmarkReport1D(
                scenario=scenario,
                metrics=metrics,
                residual_calibration=residual,
                threshold_evaluations=evaluations,
                mismatch_seed=mismatch_seed,
                detector_seed=detector_seed,
                measurement_shape=tuple(expected.shape),
                valid_pixel_count=valid_count,
                masked_pixel_count=masked_count,
                saturated_pixel_count=saturated_count,
                generated_signal_sha256=_array_digest(expected),
                measurement_sha256=_array_digest(
                    measurement.raw_adu,
                    measurement.calibrated_signal_electrons,
                    measurement.valid_mask,
                    measurement.masked_mask,
                    measurement.saturated_mask,
                ),
                prediction_sha256=_array_digest(prediction),
                estimated_parameters_sha256=_mapping_digest(
                    output.estimated_parameters
                ),
                perturbed_inputs_sha256=_inputs_digest(perturbed_inputs),
                reconstruction_metadata=reconstruction_metadata,
            )
        )

    worst_case = tuple(
        _worst_evaluation(
            [
                report.threshold_evaluations[index]
                for report in scenario_reports
            ]
        )
        for index in range(len(criteria.criteria))
    )
    report = SyntheticBenchmarkReport1D(
        benchmark_id=benchmark_id,
        truth_id=truth_id,
        generator_id=generator_id,
        reconstructor_id=reconstructor_id,
        truth_metric_id=truth_metric_id,
        criteria=criteria,
        scenarios=tuple(scenario_reports),
        worst_case_evaluations=worst_case,
        nominal_inputs_sha256=_inputs_digest(nominal_inputs),
        truth_sha256=_mapping_digest(truth_parameters),
        rng_algorithm=_RNG_ALGORITHM,
        numpy_version=np.__version__,
        metadata=report_metadata,
    )
    validate_synthetic_benchmark_report_1d(report)
    return report


def _validate_residual_report(report: ResidualCalibrationReport1D) -> None:
    if not isinstance(report, ResidualCalibrationReport1D):
        raise TypeError("residual report has the wrong type")
    if (
        isinstance(report.valid_pixel_count, (bool, np.bool_))
        or not isinstance(report.valid_pixel_count, (int, np.integer))
        or report.valid_pixel_count < 2
    ):
        raise ValueError("residual valid_pixel_count must be at least two")
    for name in (
        "standardized_residual_mean",
        "standardized_residual_std",
        "standardized_residual_rms",
        "standardized_residual_q05",
        "standardized_residual_q50",
        "standardized_residual_q95",
        "coverage_1sigma",
        "coverage_2sigma",
        "coverage_1sigma_error",
        "coverage_2sigma_error",
    ):
        _finite_scalar(f"residual.{name}", getattr(report, name))
    for name in ("coverage_1sigma", "coverage_2sigma"):
        value = float(getattr(report, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"residual.{name} must lie in [0, 1]")
    if report.coverage_2sigma < report.coverage_1sigma:
        raise ValueError("residual 2-sigma coverage cannot be below 1-sigma")
    if report.coverage_1sigma_error != abs(
        report.coverage_1sigma - _EXPECTED_COVERAGE_1SIGMA
    ):
        raise ValueError("residual 1-sigma coverage error is inconsistent")
    if report.coverage_2sigma_error != abs(
        report.coverage_2sigma - _EXPECTED_COVERAGE_2SIGMA
    ):
        raise ValueError("residual 2-sigma coverage error is inconsistent")
    if not (
        report.standardized_residual_q05
        <= report.standardized_residual_q50
        <= report.standardized_residual_q95
    ):
        raise ValueError("residual quantiles are not ordered")
    if report.standardized_residual_std < 0.0:
        raise ValueError("residual standard deviation must be non-negative")
    if report.standardized_residual_rms < 0.0:
        raise ValueError("residual RMS must be non-negative")
    if report.poisson_deviance_per_valid_pixel is not None:
        _finite_scalar(
            "residual.poisson_deviance_per_valid_pixel",
            report.poisson_deviance_per_valid_pixel,
            nonnegative=True,
        )
        if report.poisson_deviance_model.startswith("not_applicable"):
            raise ValueError("an inapplicable residual model cannot have deviance")
    elif not report.poisson_deviance_model.startswith("not_applicable"):
        raise ValueError("an applicable Poisson model must contain deviance")
    _nonempty_string(
        "residual.poisson_deviance_model", report.poisson_deviance_model
    )
    if report.standardized_variance_model != (
        "declared_poisson_plus_gaussian_read_noise"
    ):
        raise ValueError("residual standardized variance model is unsupported")
    _nonempty_string("residual.calibration_id", report.calibration_id)


def _validate_sha256(name: str, value: Any) -> None:
    text = _nonempty_string(name, value)
    if len(text) != 64:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    try:
        bytes.fromhex(text)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest") from error


def _held_out_scan_indices_1d(
    indices: Array,
) -> tuple[int, ...]:
    if isinstance(indices, (str, bytes)):
        raise TypeError("held_out_scan_indices must be a sequence of integers")
    array = np.asarray(indices)
    if array.ndim != 1:
        raise ValueError("held_out_scan_indices must be one-dimensional")
    normalized: list[int] = []
    for index in array.tolist():
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("held-out scan indices must be integers")
        resolved = int(index)
        if resolved < 0:
            raise ValueError("held-out scan indices must be non-negative")
        normalized.append(resolved)
    if not normalized:
        raise ValueError("held_out_scan_indices must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("held-out scan indices must be unique")
    return tuple(normalized)


def _validate_residual_criteria_1d(
    criteria: BenchmarkCriteria1D,
    residual_metrics: Mapping[str, float] | None = None,
) -> None:
    validate_benchmark_criteria_1d(criteria)
    for criterion in criteria.criteria:
        if not criterion.metric_name.startswith("residual."):
            raise ValueError(
                "residual-calibration evidence accepts only residual.* criteria"
            )
        if (
            residual_metrics is not None
            and criterion.metric_name not in residual_metrics
        ):
            raise ValueError(
                f"residual criterion metric {criterion.metric_name!r} "
                "is unavailable for this detector calibration"
            )


def validate_residual_calibration_evidence_1d(
    evidence: ResidualCalibrationEvidence1D,
) -> None:
    """Validate residual evidence and rederive every acceptance evaluation."""
    if not isinstance(evidence, ResidualCalibrationEvidence1D):
        raise TypeError("evidence must be a ResidualCalibrationEvidence1D")
    _validate_residual_report(evidence.residual_calibration)
    residual_metrics = _residual_metrics(evidence.residual_calibration)
    _validate_residual_criteria_1d(evidence.criteria, residual_metrics)

    if not isinstance(evidence.held_out_scan_indices, tuple):
        raise TypeError("evidence.held_out_scan_indices must be a tuple")
    indices = _held_out_scan_indices_1d(evidence.held_out_scan_indices)
    shape = evidence.measurement_shape
    if (
        not isinstance(shape, tuple)
        or len(shape) != 2
        or any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or int(size) < 1
            for size in shape
        )
    ):
        raise ValueError(
            "evidence.measurement_shape must contain two positive integers"
        )
    if len(indices) != int(shape[0]):
        raise ValueError(
            "held-out scan count must equal the measurement leading dimension"
        )
    if evidence.residual_calibration.valid_pixel_count > int(np.prod(shape)):
        raise ValueError("residual valid-pixel count exceeds measurement size")

    calibration_id = _nonempty_string(
        "evidence.calibration_id", evidence.calibration_id
    )
    if calibration_id != evidence.residual_calibration.calibration_id:
        raise ValueError("evidence and residual calibration identifiers differ")
    _validate_sha256(
        "evidence.measurement_sha256", evidence.measurement_sha256
    )
    _validate_sha256(
        "evidence.prediction_sha256", evidence.prediction_sha256
    )
    problem_id = _nonempty_string(
        "evidence.reconstruction_problem_id",
        evidence.reconstruction_problem_id,
    )
    _finite_scalar(
        "evidence.minimum_variance_electrons2",
        evidence.minimum_variance_electrons2,
        positive=True,
    )
    if evidence.evaluator_id != _RESIDUAL_EVIDENCE_EVALUATOR_ID:
        raise ValueError("residual-calibration evidence evaluator is unsupported")

    expected = _threshold_evaluations(
        problem_id,
        residual_metrics,
        evidence.criteria,
    )
    if not isinstance(evidence.threshold_evaluations, tuple):
        raise TypeError("evidence.threshold_evaluations must be a tuple")
    if evidence.threshold_evaluations != expected:
        raise ValueError(
            "residual threshold evaluations differ from the report and criteria"
        )


def evaluate_residual_calibration_evidence_1d(
    measurement: DetectorMeasurement1D,
    predicted_signal_electrons: Array,
    *,
    criteria: BenchmarkCriteria1D,
    held_out_scan_indices: Array,
    reconstruction_problem_id: str,
    minimum_variance_electrons2: float = 1e-9,
) -> ResidualCalibrationEvidence1D:
    """Evaluate residual calibration on explicitly identified held-out scans.

    The caller is responsible for supplying scans that were excluded from all
    reconstruction fitting and model selection.  This function validates and
    records that split identity, but cannot infer training provenance from a
    detector array alone.
    """
    _validate_measurement(measurement)
    observed = np.asarray(measurement.calibrated_signal_electrons)
    if observed.ndim != 2:
        raise ValueError("held-out detector measurements must be two-dimensional")
    prediction = _numeric_array(
        "predicted_signal_electrons",
        predicted_signal_electrons,
        nonnegative=True,
    ).astype(float, copy=False)
    if prediction.shape != observed.shape:
        raise ValueError("prediction and held-out measurement shapes differ")
    indices = _held_out_scan_indices_1d(held_out_scan_indices)
    if len(indices) != observed.shape[0]:
        raise ValueError(
            "held-out scan count must equal the measurement leading dimension"
        )
    problem_id = _nonempty_string(
        "reconstruction_problem_id", reconstruction_problem_id
    )
    floor = _finite_scalar(
        "minimum_variance_electrons2",
        minimum_variance_electrons2,
        positive=True,
    )
    report = residual_calibration_report_1d(
        measurement,
        prediction,
        minimum_variance_electrons2=floor,
    )
    residual_metrics = _residual_metrics(report)
    _validate_residual_criteria_1d(criteria, residual_metrics)
    evidence = ResidualCalibrationEvidence1D(
        residual_calibration=report,
        criteria=criteria,
        threshold_evaluations=_threshold_evaluations(
            problem_id,
            residual_metrics,
            criteria,
        ),
        held_out_scan_indices=indices,
        measurement_shape=tuple(int(size) for size in observed.shape),
        calibration_id=measurement.calibration_id,
        measurement_sha256=_measurement_digest_1d(measurement),
        prediction_sha256=_array_digest(prediction),
        reconstruction_problem_id=problem_id,
        minimum_variance_electrons2=floor,
        evaluator_id=_RESIDUAL_EVIDENCE_EVALUATOR_ID,
    )
    validate_residual_calibration_evidence_1d(evidence)
    return evidence


def validate_synthetic_benchmark_report_1d(
    report: SyntheticBenchmarkReport1D,
) -> None:
    """Validate internal report consistency and provenance-bearing gates."""
    if not isinstance(report, SyntheticBenchmarkReport1D):
        raise TypeError("report must be a SyntheticBenchmarkReport1D")
    for name in (
        "benchmark_id",
        "truth_id",
        "generator_id",
        "reconstructor_id",
        "truth_metric_id",
    ):
        _nonempty_string(f"report.{name}", getattr(report, name))
    validate_benchmark_criteria_1d(report.criteria)
    if not isinstance(report.scenarios, tuple) or not report.scenarios:
        raise ValueError("report.scenarios must be a non-empty tuple")
    scenario_ids: set[str] = set()
    expected_evaluations: dict[str, list[ThresholdEvaluation1D]] = {
        criterion.criterion_id: [] for criterion in report.criteria.criteria
    }
    for scenario_report in report.scenarios:
        if not isinstance(scenario_report, ScenarioBenchmarkReport1D):
            raise TypeError("report contains an invalid scenario report")
        _validate_scenario(scenario_report.scenario)
        scenario_id = scenario_report.scenario.scenario_id
        if scenario_id in scenario_ids:
            raise ValueError("report scenario identifiers must be unique")
        scenario_ids.add(scenario_id)
        metrics = _finite_metric_mapping("scenario metrics", scenario_report.metrics)
        _validate_residual_report(scenario_report.residual_calibration)
        residual_metrics = _residual_metrics(scenario_report.residual_calibration)
        for name, expected_value in residual_metrics.items():
            if name not in metrics or metrics[name] != expected_value:
                raise ValueError(
                    f"scenario metric {name!r} differs from its residual report"
                )
        shape = scenario_report.measurement_shape
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
                for value in shape
            )
        ):
            raise ValueError("scenario measurement_shape must contain two sizes")
        total = int(np.prod(shape))
        for name in (
            "valid_pixel_count",
            "masked_pixel_count",
            "saturated_pixel_count",
        ):
            count = getattr(scenario_report, name)
            if (
                isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer))
                or not 0 <= count <= total
            ):
                raise ValueError(f"scenario {name} is invalid")
        if scenario_report.valid_pixel_count != (
            scenario_report.residual_calibration.valid_pixel_count
        ):
            raise ValueError("scenario and residual valid-pixel counts differ")
        accounted_without_saturation = (
            scenario_report.valid_pixel_count + scenario_report.masked_pixel_count
        )
        if accounted_without_saturation > total:
            raise ValueError("scenario pixel counts are inconsistent")
        if scenario_report.valid_pixel_count < (
            total
            - scenario_report.masked_pixel_count
            - scenario_report.saturated_pixel_count
        ):
            raise ValueError("scenario pixel counts are inconsistent")
        expected_masked_count = (
            shape[0]
            * len(scenario_report.scenario.detector.masked_detector_indices)
        )
        if scenario_report.masked_pixel_count != expected_masked_count:
            raise ValueError("scenario masked-pixel count differs from its mask")
        expected_data_metrics = {
            "data.valid_fraction": scenario_report.valid_pixel_count / total,
            "data.masked_fraction": scenario_report.masked_pixel_count / total,
            "data.saturated_fraction": (
                scenario_report.saturated_pixel_count / total
            ),
        }
        for name, expected_value in expected_data_metrics.items():
            if name not in metrics or metrics[name] != expected_value:
                raise ValueError(
                    f"scenario metric {name!r} differs from its pixel count"
                )
        _integer_seed("scenario mismatch_seed", scenario_report.mismatch_seed)
        _integer_seed("scenario detector_seed", scenario_report.detector_seed)
        expected_mismatch_seed, expected_detector_seed = _spawn_uint64_seeds(
            scenario_report.scenario.seed
        )
        if (
            scenario_report.mismatch_seed != expected_mismatch_seed
            or scenario_report.detector_seed != expected_detector_seed
        ):
            raise ValueError("scenario random streams do not match its seed")
        for name in (
            "generated_signal_sha256",
            "measurement_sha256",
            "prediction_sha256",
            "estimated_parameters_sha256",
            "perturbed_inputs_sha256",
        ):
            _validate_sha256(name, getattr(scenario_report, name))
        _json_safe(
            scenario_report.reconstruction_metadata,
            path="scenario reconstruction metadata",
        )
        expected = _threshold_evaluations(
            scenario_id, metrics, report.criteria
        )
        if len(expected) != len(scenario_report.threshold_evaluations):
            raise ValueError("scenario threshold evaluation count differs")
        for actual, derived in zip(
            scenario_report.threshold_evaluations, expected, strict=True
        ):
            if not isinstance(actual, ThresholdEvaluation1D):
                raise TypeError("scenario threshold has the wrong type")
            if actual.criterion != derived.criterion:
                raise ValueError("scenario threshold criterion differs")
            if actual.scenario_id != scenario_id:
                raise ValueError("scenario threshold has the wrong scenario id")
            if actual.observed_value != derived.observed_value:
                raise ValueError("scenario threshold value differs from metric")
            expected_evaluations[actual.criterion.criterion_id].append(actual)

    if len(report.worst_case_evaluations) != len(report.criteria.criteria):
        raise ValueError("worst-case threshold evaluation count differs")
    for criterion, stored in zip(
        report.criteria.criteria,
        report.worst_case_evaluations,
        strict=True,
    ):
        if not isinstance(stored, ThresholdEvaluation1D):
            raise TypeError("worst-case threshold has the wrong type")
        derived = _worst_evaluation(expected_evaluations[criterion.criterion_id])
        if stored != derived:
            raise ValueError("stored worst-case threshold evaluation is inconsistent")
    _validate_sha256("report.nominal_inputs_sha256", report.nominal_inputs_sha256)
    _validate_sha256("report.truth_sha256", report.truth_sha256)
    if report.rng_algorithm != _RNG_ALGORITHM:
        raise ValueError("report RNG algorithm is unsupported")
    _nonempty_string("report.numpy_version", report.numpy_version)
    _json_safe(report.metadata, path="report.metadata")


def _criterion_payload(criterion: BenchmarkCriterion1D) -> dict[str, Any]:
    return {
        "criterion_id": criterion.criterion_id,
        "metric_name": criterion.metric_name,
        "threshold_source": criterion.threshold_source,
        "lower_bound": criterion.lower_bound,
        "upper_bound": criterion.upper_bound,
    }


def _scenario_payload(scenario: SyntheticBenchmarkScenario1D) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "detector": {
            "detection_efficiency": scenario.detector.detection_efficiency,
            "gain_adu_per_electron": scenario.detector.gain_adu_per_electron,
            "calibrated_gain_adu_per_electron": (
                scenario.detector.calibrated_gain_adu_per_electron
            ),
            "dark_electrons_per_pixel": (
                scenario.detector.dark_electrons_per_pixel
            ),
            "calibrated_dark_electrons_per_pixel": (
                scenario.detector.calibrated_dark_electrons_per_pixel
            ),
            "read_noise_std_electrons": (
                scenario.detector.read_noise_std_electrons
            ),
            "calibrated_read_noise_std_electrons": (
                scenario.detector.calibrated_read_noise_std_electrons
            ),
            "saturation_electrons": scenario.detector.saturation_electrons,
            "masked_detector_indices": list(
                scenario.detector.masked_detector_indices
            ),
            "calibration_id": scenario.detector.calibration_id,
        },
        "forward_mismatch": {
            name: getattr(scenario.forward_mismatch, name)
            for name in ForwardModelMismatch1D.__dataclass_fields__
        },
        "metadata": _json_safe(scenario.metadata),
    }


def _residual_payload(report: ResidualCalibrationReport1D) -> dict[str, Any]:
    return {
        name: getattr(report, name)
        for name in ResidualCalibrationReport1D.__dataclass_fields__
    }


def _evaluation_payload(evaluation: ThresholdEvaluation1D) -> dict[str, Any]:
    return {
        "criterion": _criterion_payload(evaluation.criterion),
        "observed_value": evaluation.observed_value,
        "scenario_id": evaluation.scenario_id,
    }


def _report_payload(report: SyntheticBenchmarkReport1D) -> dict[str, Any]:
    return {
        "benchmark_id": report.benchmark_id,
        "truth_id": report.truth_id,
        "generator_id": report.generator_id,
        "reconstructor_id": report.reconstructor_id,
        "truth_metric_id": report.truth_metric_id,
        "criteria": {
            "criteria_id": report.criteria.criteria_id,
            "criteria": [
                _criterion_payload(criterion)
                for criterion in report.criteria.criteria
            ],
            "metadata": _json_safe(report.criteria.metadata),
        },
        "scenarios": [
            {
                "scenario": _scenario_payload(item.scenario),
                "metrics": dict(item.metrics),
                "residual_calibration": _residual_payload(
                    item.residual_calibration
                ),
                "threshold_evaluations": [
                    _evaluation_payload(evaluation)
                    for evaluation in item.threshold_evaluations
                ],
                "mismatch_seed": item.mismatch_seed,
                "detector_seed": item.detector_seed,
                "measurement_shape": list(item.measurement_shape),
                "valid_pixel_count": item.valid_pixel_count,
                "masked_pixel_count": item.masked_pixel_count,
                "saturated_pixel_count": item.saturated_pixel_count,
                "generated_signal_sha256": item.generated_signal_sha256,
                "measurement_sha256": item.measurement_sha256,
                "prediction_sha256": item.prediction_sha256,
                "estimated_parameters_sha256": (
                    item.estimated_parameters_sha256
                ),
                "perturbed_inputs_sha256": item.perturbed_inputs_sha256,
                "reconstruction_metadata": _json_safe(
                    item.reconstruction_metadata
                ),
            }
            for item in report.scenarios
        ],
        "worst_case_evaluations": [
            _evaluation_payload(evaluation)
            for evaluation in report.worst_case_evaluations
        ],
        "nominal_inputs_sha256": report.nominal_inputs_sha256,
        "truth_sha256": report.truth_sha256,
        "rng_algorithm": report.rng_algorithm,
        "numpy_version": report.numpy_version,
        "metadata": _json_safe(report.metadata),
    }


def save_synthetic_benchmark_report_1d(
    path: str | Path,
    report: SyntheticBenchmarkReport1D,
) -> None:
    """Save a benchmark report as schema-versioned, non-pickled JSON-in-NPZ."""
    validate_synthetic_benchmark_report_1d(report)
    payload_text = json.dumps(
        _report_payload(report),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    payload_digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        schema_version=np.asarray(_REPORT_SCHEMA_VERSION, dtype=np.int64),
        payload_json=np.asarray(payload_text),
        payload_sha256=np.asarray(payload_digest),
    )


def _criterion_from_payload(payload: Mapping[str, Any]) -> BenchmarkCriterion1D:
    _require_mapping_keys(
        payload,
        {
            "criterion_id",
            "metric_name",
            "threshold_source",
            "lower_bound",
            "upper_bound",
        },
        "criterion",
    )
    return BenchmarkCriterion1D(
        criterion_id=payload["criterion_id"],
        metric_name=payload["metric_name"],
        threshold_source=payload["threshold_source"],
        lower_bound=payload["lower_bound"],
        upper_bound=payload["upper_bound"],
    )


def _scenario_from_payload(payload: Mapping[str, Any]) -> SyntheticBenchmarkScenario1D:
    _require_mapping_keys(
        payload,
        {
            "scenario_id",
            "seed",
            "detector",
            "forward_mismatch",
            "metadata",
        },
        "scenario",
    )
    _require_mapping_keys(
        payload["detector"],
        set(DetectorPerturbation1D.__dataclass_fields__),
        "scenario detector",
    )
    _require_mapping_keys(
        payload["forward_mismatch"],
        set(ForwardModelMismatch1D.__dataclass_fields__),
        "scenario forward mismatch",
    )
    detector_payload = dict(payload["detector"])
    detector_payload["masked_detector_indices"] = tuple(
        detector_payload["masked_detector_indices"]
    )
    return SyntheticBenchmarkScenario1D(
        scenario_id=payload["scenario_id"],
        seed=payload["seed"],
        detector=DetectorPerturbation1D(**detector_payload),
        forward_mismatch=ForwardModelMismatch1D(**payload["forward_mismatch"]),
        metadata=payload["metadata"],
    )


def _evaluation_from_payload(payload: Mapping[str, Any]) -> ThresholdEvaluation1D:
    _require_mapping_keys(
        payload,
        {"criterion", "observed_value", "scenario_id"},
        "threshold evaluation",
    )
    return ThresholdEvaluation1D(
        criterion=_criterion_from_payload(payload["criterion"]),
        observed_value=payload["observed_value"],
        scenario_id=payload["scenario_id"],
    )


def _require_mapping_keys(
    payload: Any,
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{context} has invalid fields; missing={missing}, extra={extra}"
        )


def _report_from_payload(payload: Mapping[str, Any]) -> SyntheticBenchmarkReport1D:
    _require_mapping_keys(
        payload,
        {
            "benchmark_id",
            "truth_id",
            "generator_id",
            "reconstructor_id",
            "truth_metric_id",
            "criteria",
            "scenarios",
            "worst_case_evaluations",
            "nominal_inputs_sha256",
            "truth_sha256",
            "rng_algorithm",
            "numpy_version",
            "metadata",
        },
        "benchmark payload",
    )
    criteria_payload = payload["criteria"]
    _require_mapping_keys(
        criteria_payload,
        {"criteria_id", "criteria", "metadata"},
        "benchmark criteria",
    )
    criteria = BenchmarkCriteria1D(
        criteria_id=criteria_payload["criteria_id"],
        criteria=tuple(
            _criterion_from_payload(item)
            for item in criteria_payload["criteria"]
        ),
        metadata=criteria_payload["metadata"],
    )
    scenario_reports = []
    for item in payload["scenarios"]:
        _require_mapping_keys(
            item,
            {
                "scenario",
                "metrics",
                "residual_calibration",
                "threshold_evaluations",
                "mismatch_seed",
                "detector_seed",
                "measurement_shape",
                "valid_pixel_count",
                "masked_pixel_count",
                "saturated_pixel_count",
                "generated_signal_sha256",
                "measurement_sha256",
                "prediction_sha256",
                "estimated_parameters_sha256",
                "perturbed_inputs_sha256",
                "reconstruction_metadata",
            },
            "scenario report",
        )
        _require_mapping_keys(
            item["residual_calibration"],
            set(ResidualCalibrationReport1D.__dataclass_fields__),
            "residual calibration",
        )
        residual = ResidualCalibrationReport1D(**item["residual_calibration"])
        scenario_reports.append(
            ScenarioBenchmarkReport1D(
                scenario=_scenario_from_payload(item["scenario"]),
                metrics=item["metrics"],
                residual_calibration=residual,
                threshold_evaluations=tuple(
                    _evaluation_from_payload(evaluation)
                    for evaluation in item["threshold_evaluations"]
                ),
                mismatch_seed=item["mismatch_seed"],
                detector_seed=item["detector_seed"],
                measurement_shape=tuple(item["measurement_shape"]),
                valid_pixel_count=item["valid_pixel_count"],
                masked_pixel_count=item["masked_pixel_count"],
                saturated_pixel_count=item["saturated_pixel_count"],
                generated_signal_sha256=item["generated_signal_sha256"],
                measurement_sha256=item["measurement_sha256"],
                prediction_sha256=item["prediction_sha256"],
                estimated_parameters_sha256=item[
                    "estimated_parameters_sha256"
                ],
                perturbed_inputs_sha256=item["perturbed_inputs_sha256"],
                reconstruction_metadata=item["reconstruction_metadata"],
            )
        )
    return SyntheticBenchmarkReport1D(
        benchmark_id=payload["benchmark_id"],
        truth_id=payload["truth_id"],
        generator_id=payload["generator_id"],
        reconstructor_id=payload["reconstructor_id"],
        truth_metric_id=payload["truth_metric_id"],
        criteria=criteria,
        scenarios=tuple(scenario_reports),
        worst_case_evaluations=tuple(
            _evaluation_from_payload(item)
            for item in payload["worst_case_evaluations"]
        ),
        nominal_inputs_sha256=payload["nominal_inputs_sha256"],
        truth_sha256=payload["truth_sha256"],
        rng_algorithm=payload["rng_algorithm"],
        numpy_version=payload["numpy_version"],
        metadata=payload["metadata"],
    )


def load_synthetic_benchmark_report_1d(
    path: str | Path,
) -> SyntheticBenchmarkReport1D:
    """Load and fully validate a report written by the matching save helper."""
    with np.load(path, allow_pickle=False) as archive:
        if frozenset(archive.files) != _REPORT_ARCHIVE_KEYS:
            raise ValueError("benchmark archive has unexpected or missing fields")
        schema = archive["schema_version"]
        if schema.shape != () or not np.issubdtype(schema.dtype, np.integer):
            raise ValueError("benchmark schema version must be a scalar integer")
        if int(schema.item()) != _REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported synthetic-benchmark schema version")
        payload_array = archive["payload_json"]
        digest_array = archive["payload_sha256"]
        if payload_array.shape != () or payload_array.dtype.kind != "U":
            raise ValueError("benchmark payload must be a scalar Unicode string")
        if digest_array.shape != () or digest_array.dtype.kind != "U":
            raise ValueError("benchmark digest must be a scalar Unicode string")
        payload_text = str(payload_array.item())
        stored_digest = str(digest_array.item())
    actual_digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    if stored_digest != actual_digest:
        raise ValueError("benchmark payload digest does not match")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ValueError("benchmark payload is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("benchmark payload must be a JSON object")
    try:
        report = _report_from_payload(payload)
    except (KeyError, TypeError) as error:
        raise ValueError("benchmark payload does not match the schema") from error
    validate_synthetic_benchmark_report_1d(report)
    return report


def _residual_evidence_payload(
    evidence: ResidualCalibrationEvidence1D,
) -> dict[str, Any]:
    return {
        "residual_calibration": _residual_payload(
            evidence.residual_calibration
        ),
        "criteria": {
            "criteria_id": evidence.criteria.criteria_id,
            "criteria": [
                _criterion_payload(criterion)
                for criterion in evidence.criteria.criteria
            ],
            "metadata": _json_safe(evidence.criteria.metadata),
        },
        "threshold_evaluations": [
            _evaluation_payload(evaluation)
            for evaluation in evidence.threshold_evaluations
        ],
        "held_out_scan_indices": list(evidence.held_out_scan_indices),
        "measurement_shape": list(evidence.measurement_shape),
        "calibration_id": evidence.calibration_id,
        "measurement_sha256": evidence.measurement_sha256,
        "prediction_sha256": evidence.prediction_sha256,
        "reconstruction_problem_id": evidence.reconstruction_problem_id,
        "minimum_variance_electrons2": (
            evidence.minimum_variance_electrons2
        ),
        "evaluator_id": evidence.evaluator_id,
    }


def save_residual_calibration_evidence_1d(
    path: str | Path,
    evidence: ResidualCalibrationEvidence1D,
) -> None:
    """Save held-out residual evidence as canonical JSON in a safe NPZ."""
    validate_residual_calibration_evidence_1d(evidence)
    payload_text = json.dumps(
        _residual_evidence_payload(evidence),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    payload_digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        schema_version=np.asarray(
            _RESIDUAL_EVIDENCE_SCHEMA_VERSION, dtype=np.int64
        ),
        payload_json=np.asarray(payload_text),
        payload_sha256=np.asarray(payload_digest),
    )


def _residual_evidence_from_payload(
    payload: Mapping[str, Any],
) -> ResidualCalibrationEvidence1D:
    _require_mapping_keys(
        payload,
        {
            "residual_calibration",
            "criteria",
            "threshold_evaluations",
            "held_out_scan_indices",
            "measurement_shape",
            "calibration_id",
            "measurement_sha256",
            "prediction_sha256",
            "reconstruction_problem_id",
            "minimum_variance_electrons2",
            "evaluator_id",
        },
        "residual-calibration evidence",
    )
    residual_payload = payload["residual_calibration"]
    _require_mapping_keys(
        residual_payload,
        set(ResidualCalibrationReport1D.__dataclass_fields__),
        "residual-calibration report",
    )
    criteria_payload = payload["criteria"]
    _require_mapping_keys(
        criteria_payload,
        {"criteria_id", "criteria", "metadata"},
        "residual-calibration criteria",
    )
    criteria_items = criteria_payload["criteria"]
    if not isinstance(criteria_items, list):
        raise ValueError("residual-calibration criteria must be a JSON array")
    evaluations_payload = payload["threshold_evaluations"]
    if not isinstance(evaluations_payload, list):
        raise ValueError("residual threshold evaluations must be a JSON array")
    indices = payload["held_out_scan_indices"]
    shape = payload["measurement_shape"]
    if not isinstance(indices, list):
        raise ValueError("held-out scan indices must be a JSON array")
    if not isinstance(shape, list):
        raise ValueError("measurement shape must be a JSON array")
    return ResidualCalibrationEvidence1D(
        residual_calibration=ResidualCalibrationReport1D(**residual_payload),
        criteria=BenchmarkCriteria1D(
            criteria_id=criteria_payload["criteria_id"],
            criteria=tuple(
                _criterion_from_payload(item) for item in criteria_items
            ),
            metadata=criteria_payload["metadata"],
        ),
        threshold_evaluations=tuple(
            _evaluation_from_payload(item) for item in evaluations_payload
        ),
        held_out_scan_indices=tuple(indices),
        measurement_shape=tuple(shape),
        calibration_id=payload["calibration_id"],
        measurement_sha256=payload["measurement_sha256"],
        prediction_sha256=payload["prediction_sha256"],
        reconstruction_problem_id=payload["reconstruction_problem_id"],
        minimum_variance_electrons2=payload[
            "minimum_variance_electrons2"
        ],
        evaluator_id=payload["evaluator_id"],
    )


def load_residual_calibration_evidence_1d(
    path: str | Path,
) -> ResidualCalibrationEvidence1D:
    """Load residual evidence after strict archive, schema, and digest checks."""
    with np.load(path, allow_pickle=False) as archive:
        if frozenset(archive.files) != _RESIDUAL_EVIDENCE_ARCHIVE_KEYS:
            raise ValueError(
                "residual-evidence archive has unexpected or missing fields"
            )
        schema = archive["schema_version"]
        if schema.shape != () or not np.issubdtype(schema.dtype, np.integer):
            raise ValueError(
                "residual-evidence schema version must be a scalar integer"
            )
        if int(schema.item()) != _RESIDUAL_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unsupported residual-evidence schema version")
        payload_array = archive["payload_json"]
        digest_array = archive["payload_sha256"]
        if payload_array.shape != () or payload_array.dtype.kind != "U":
            raise ValueError(
                "residual-evidence payload must be a scalar Unicode string"
            )
        if digest_array.shape != () or digest_array.dtype.kind != "U":
            raise ValueError(
                "residual-evidence digest must be a scalar Unicode string"
            )
        payload_text = str(payload_array.item())
        stored_digest = str(digest_array.item())
    _validate_sha256("residual-evidence payload digest", stored_digest)
    actual_digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    if stored_digest != actual_digest:
        raise ValueError("residual-evidence payload digest does not match")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ValueError("residual-evidence payload is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("residual-evidence payload must be a JSON object")
    try:
        evidence = _residual_evidence_from_payload(payload)
    except (KeyError, TypeError) as error:
        raise ValueError(
            "residual-evidence payload does not match the schema"
        ) from error
    validate_residual_calibration_evidence_1d(evidence)
    return evidence
