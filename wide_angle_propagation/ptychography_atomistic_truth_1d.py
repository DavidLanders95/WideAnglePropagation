r"""Truth-isolated direct atomic quadrature for blind atomistic-edit tests.

This module evaluates Kirkland atomic potentials directly for arbitrary element
symbols, continuous side-view positions, and non-negative per-centre weights.
It does not use the production Lobato host renderer, abTEM's potential builder,
projection-integral machinery, or image interpolation.  It is intended for
blind synthetic edit deltas and numerical quadrature studies.

Propagation is still shared with the reconstruction workflow, and an analytic
independent-atom parameterization is not experimental or first-principles
validation.  Every result therefore remains explicitly fail-closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .ptychography_atomic_validation_1d import (
    AtomicTemplateQuadratureOptions1D,
    render_si_atomic_template_1d,
)


__all__ = [
    "AdaptiveAtomicCubatureConvergenceReport1D",
    "AtomicQuadratureConvergenceReport1D",
    "DirectAtomicNumericalOptions1D",
    "DirectAtomicTemplate1D",
    "WeightedAtomicPotentialGrid1D",
    "accumulate_weighted_atomic_potential_1d",
    "render_direct_atomic_template_1d",
    "sweep_adaptive_atomic_cubature_convergence_1d",
    "sweep_atomic_quadrature_convergence_1d",
]


_ELEMENT_PATTERN = re.compile(r"^[A-Z][a-z]?$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_FAIL_CLOSED_REASON = (
    "direct Kirkland edit generator is numerically independent of the Lobato "
    "template builder but shares independent-atom assumptions and downstream "
    "propagation; it is not experimental or first-principles validation"
)


@dataclass(frozen=True)
class DirectAtomicNumericalOptions1D:
    """Numerical policy for the generic direct atomic truth renderer.

    ``tensor_product`` exactly preserves the original renderer.  The optional
    ``adaptive_factorized_cubature`` uses the analytic Gaussian/Laplace form of
    the Kirkland basis to integrate every finite voxel at once.  Rectangular
    integrals factor into error-function differences, leaving one adaptive
    vector-valued integral.  This treats the singular and near-core voxels
    without sampling ``r=0`` and yields a whole-template L2 error estimate.
    """

    integration_method: Literal[
        "tensor_product", "adaptive_factorized_cubature"
    ] = "tensor_product"
    adaptive_relative_tolerance: float = 1e-7
    adaptive_absolute_l2_tolerance: float = 1e-9
    adaptive_quadrature_rule: Literal["gk15", "gk21"] = "gk21"
    adaptive_max_intervals: int = 4096
    adaptive_max_evaluations: int = 500_000
    adaptive_error_safety_factor: float = 4.0

    def __post_init__(self) -> None:
        if self.integration_method not in {
            "tensor_product",
            "adaptive_factorized_cubature",
        }:
            raise ValueError(
                "integration_method must be 'tensor_product' or "
                "'adaptive_factorized_cubature'"
            )
        for name in (
            "adaptive_relative_tolerance",
            "adaptive_absolute_l2_tolerance",
            "adaptive_error_safety_factor",
        ):
            object.__setattr__(self, name, _positive(name, getattr(self, name)))
        if self.adaptive_error_safety_factor < 1.0:
            raise ValueError("adaptive_error_safety_factor must be at least one")
        if self.adaptive_quadrature_rule not in {"gk15", "gk21"}:
            raise ValueError("adaptive_quadrature_rule must be 'gk15' or 'gk21'")
        for name in (
            "adaptive_max_intervals",
            "adaptive_max_evaluations",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be an integer")
            try:
                converted = int(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise TypeError(f"{name} must be an integer") from error
            if converted != value or converted < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, converted)


_LIMITATIONS = (
    "independent neutral-atom parameterization",
    "finite centred out-of-plane projection width",
    "compact square cutoff",
    "no bonding, charge transfer, thermal motion, or ionization",
    "shared downstream multislice propagation in current blind benchmarks",
)


@dataclass(frozen=True, eq=False)
class DirectAtomicTemplate1D:
    """Immutable exact-subpixel template for one atomic element."""

    element: str
    values: np.ndarray
    unit_integrated_values: np.ndarray
    integrated_scattering: float
    sampling_s_A: float
    sampling_u_A: float
    half_shape: tuple[int, int]
    fractional_offset_A: tuple[float, float]
    options: AtomicTemplateQuadratureOptions1D
    template_id: str
    numerical_options: DirectAtomicNumericalOptions1D = field(
        default_factory=DirectAtomicNumericalOptions1D
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trust_claim: bool = False
    trust_reason: str = _FAIL_CLOSED_REASON
    limitations: tuple[str, ...] = _LIMITATIONS

    def __post_init__(self) -> None:
        _validate_direct_atomic_template(self)


@dataclass(frozen=True, eq=False)
class WeightedAtomicPotentialGrid1D:
    """Finite-grid sum of weighted, explicitly positioned atomic centres."""

    values: np.ndarray
    s_coordinates_A: np.ndarray
    u_coordinates_A: np.ndarray
    site_coordinates_A: np.ndarray
    elements: tuple[str, ...]
    scattering_weights: np.ndarray
    template_ids: tuple[str, ...]
    options: AtomicTemplateQuadratureOptions1D
    require_full_kernel_support: bool
    grid_id: str
    numerical_options: DirectAtomicNumericalOptions1D = field(
        default_factory=DirectAtomicNumericalOptions1D
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trust_claim: bool = False
    trust_reason: str = _FAIL_CLOSED_REASON
    limitations: tuple[str, ...] = _LIMITATIONS

    def __post_init__(self) -> None:
        _validate_weighted_atomic_grid(self)


@dataclass(frozen=True, eq=False)
class AtomicQuadratureConvergenceReport1D:
    """Tensor-order sweep against the highest declared reference order.

    ``passed`` certifies :attr:`candidate_order_pair` against
    :attr:`reference_order_pair`; it does not certify ``base_options`` unless
    the caller explicitly made the candidate pair equal those base orders.
    """

    elements: tuple[str, ...]
    fractional_offsets_A: np.ndarray
    order_pairs: np.ndarray
    maximum_relative_l2_by_order: np.ndarray
    maximum_relative_integral_error_by_order: np.ndarray
    relative_l2_tolerance: float
    relative_integral_tolerance: float
    passed: bool
    reference_order_pair: tuple[int, int]
    sampling_s_A: float
    sampling_u_A: float
    base_options: AtomicTemplateQuadratureOptions1D
    report_id: str
    numerical_options: DirectAtomicNumericalOptions1D = field(
        default_factory=DirectAtomicNumericalOptions1D
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trust_claim: bool = False
    trust_reason: str = _FAIL_CLOSED_REASON
    limitations: tuple[str, ...] = _LIMITATIONS

    def __post_init__(self) -> None:
        _validate_quadrature_report(self)

    @property
    def candidate_order_pair(self) -> tuple[int, int]:
        """Return the authenticated penultimate order used by ``passed``."""

        pairs = np.asarray(self.order_pairs)
        return int(pairs[-2, 0]), int(pairs[-2, 1])


@dataclass(frozen=True, eq=False)
class AdaptiveAtomicCubatureConvergenceReport1D:
    """Tolerance sweep against the tightest factorized-cubature result."""

    elements: tuple[str, ...]
    fractional_offsets_A: np.ndarray
    tolerance_levels: np.ndarray
    maximum_relative_l2_by_level: np.ndarray
    maximum_relative_integral_error_by_level: np.ndarray
    maximum_reported_template_l2_error_by_level: np.ndarray
    maximum_function_evaluations_by_level: np.ndarray
    relative_l2_tolerance: float
    relative_integral_tolerance: float
    passed: bool
    sampling_s_A: float
    sampling_u_A: float
    base_options: AtomicTemplateQuadratureOptions1D
    base_numerical_options: DirectAtomicNumericalOptions1D
    report_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trust_claim: bool = False
    trust_reason: str = _FAIL_CLOSED_REASON
    limitations: tuple[str, ...] = _LIMITATIONS

    def __post_init__(self) -> None:
        _validate_adaptive_cubature_report(self)


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _positive(name: str, value: Any, *, allow_zero: bool = False) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or np.iscomplexobj(array)
        or isinstance(value, (bool, np.bool_))
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(array)
    if not np.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


_DEFAULT_NUMERICAL_OPTIONS = DirectAtomicNumericalOptions1D()


def _element(value: Any) -> str:
    if not isinstance(value, str) or _ELEMENT_PATTERN.fullmatch(value) is None:
        raise ValueError("element symbols must use standard one/two-letter spelling")
    return value


def _uniform_axis(name: str, value: Any) -> tuple[np.ndarray, float]:
    axis = np.asarray(value)
    if (
        axis.ndim != 1
        or axis.size < 2
        or np.iscomplexobj(axis)
        or np.any(~np.isfinite(axis))
    ):
        raise ValueError(f"{name} must be a finite real 1D array of length >= 2")
    converted = axis.astype(float, copy=False)
    differences = np.diff(converted)
    spacing = float((converted[-1] - converted[0]) / (converted.size - 1))
    tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(converted))), abs(spacing))
    )
    if np.any(differences <= 0.0) or not np.allclose(
        differences, spacing, rtol=0.0, atol=tolerance
    ):
        raise ValueError(f"{name} must be uniformly increasing")
    return _readonly(axis, dtype=np.float64), spacing


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def _jsonable(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    mapping = dict(value or {})
    try:
        payload = json.dumps(
            _plain_json(mapping), sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise TypeError("metadata must be JSON serializable") from error
    copied = json.loads(payload)
    return _freeze_json(copied)


@lru_cache(maxsize=1)
def _adaptive_runtime_provenance() -> Mapping[str, str]:
    return MappingProxyType(
        {
            "renderer_api": "factorized-kirkland-finite-voxel-cubature:v1",
            "abtem_version": importlib.metadata.version("abtem"),
            "scipy_version": importlib.metadata.version("scipy"),
            "numpy_version": np.__version__,
        }
    )


def _numerical_options(
    value: DirectAtomicNumericalOptions1D | None,
) -> DirectAtomicNumericalOptions1D:
    if value is None:
        return _DEFAULT_NUMERICAL_OPTIONS
    if not isinstance(value, DirectAtomicNumericalOptions1D):
        raise TypeError(
            "numerical_options must be DirectAtomicNumericalOptions1D or None"
        )
    return value


def _validate_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _digest(arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    result = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        result.update(name.encode("utf-8"))
        result.update(str(array.dtype).encode("ascii"))
        result.update(json.dumps(list(array.shape)).encode("ascii"))
        result.update(array.tobytes(order="C"))
    result.update(
        json.dumps(
            _plain_json(metadata),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return result.hexdigest()


def _support_pixels(cutoff_A: float, sampling_A: float) -> int:
    """Ceil physical support without a roundoff-only extra pixel."""
    ratio = cutoff_A / sampling_A
    nearest = round(ratio)
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(ratio))
    if abs(ratio - nearest) <= tolerance:
        return int(nearest)
    return int(math.ceil(ratio))


def _half_shape(
    sampling_s_A: float,
    sampling_u_A: float,
    cutoff_A: float,
    value: Sequence[int] | None,
) -> tuple[int, int]:
    if value is None:
        return (
            _support_pixels(cutoff_A, sampling_s_A),
            _support_pixels(cutoff_A, sampling_u_A),
        )
    if isinstance(value, (str, bytes)):
        raise TypeError("half_shape must contain two positive integers")
    try:
        entries = tuple(value)
    except TypeError as error:
        raise TypeError("half_shape must contain two positive integers") from error
    if len(entries) != 2:
        raise ValueError("half_shape must contain two positive integers")
    if any(isinstance(item, (bool, np.bool_)) for item in entries) or any(
        not isinstance(item, (int, np.integer)) for item in entries
    ):
        raise TypeError("half_shape must contain two positive integers")
    try:
        half_s, half_u = (int(entries[0]), int(entries[1]))
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("half_shape must contain two positive integers") from error
    if half_s < 1 or half_u < 1:
        raise ValueError("half_shape must contain two positive integers")
    if half_s * sampling_s_A + 1e-12 < cutoff_A:
        raise ValueError("half_shape does not reach cutoff_A on the s axis")
    if half_u * sampling_u_A + 1e-12 < cutoff_A:
        raise ValueError("half_shape does not reach cutoff_A on the u axis")
    return half_s, half_u


def _fractional_offset(
    value: Sequence[float], *, sampling_s_A: float, sampling_u_A: float
) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError("fractional_offset_A must contain two real values")
    try:
        entries = tuple(value)
    except TypeError as error:
        raise TypeError(
            "fractional_offset_A must contain two real values"
        ) from error
    if len(entries) != 2:
        raise ValueError("fractional_offset_A must contain two real values")
    if any(isinstance(item, (bool, np.bool_)) for item in entries):
        raise TypeError("fractional_offset_A must contain two real values")
    try:
        offset_s, offset_u = (float(entries[0]), float(entries[1]))
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(
            "fractional_offset_A must contain two real values"
        ) from error
    if not np.isfinite(offset_s) or not np.isfinite(offset_u):
        raise ValueError("fractional offsets must be finite")
    if not (-0.5 * sampling_s_A <= offset_s < 0.5 * sampling_s_A):
        raise ValueError("fractional s offset must lie in [-sampling/2, sampling/2)")
    if not (-0.5 * sampling_u_A <= offset_u < 0.5 * sampling_u_A):
        raise ValueError("fractional u offset must lie in [-sampling/2, sampling/2)")
    return offset_s, offset_u


def _real_array(
    name: str,
    value: Any,
    *,
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    source = np.asarray(value)
    if (
        np.issubdtype(source.dtype, np.bool_)
        or not np.issubdtype(source.dtype, np.number)
        or np.iscomplexobj(source)
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if source.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if shape is not None:
        if len(shape) != ndim:
            raise RuntimeError("internal shape specification is invalid")
        for actual, expected in zip(source.shape, shape, strict=True):
            if expected is not None and actual != expected:
                raise ValueError(f"{name} has shape {source.shape}, expected {shape}")
    result = _readonly(source, dtype=np.float64)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _template_identity(
    *,
    element: str,
    sampling_s_A: float,
    sampling_u_A: float,
    half_shape: tuple[int, int],
    fractional_offset_A: tuple[float, float],
    options: AtomicTemplateQuadratureOptions1D,
    numerical_options: DirectAtomicNumericalOptions1D,
    integrated_scattering: float,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "direct_atomic_template_1d:v2",
        "element": element,
        "sampling_s_A": sampling_s_A,
        "sampling_u_A": sampling_u_A,
        "half_shape": list(half_shape),
        "fractional_offset_A": list(fractional_offset_A),
        "options": asdict(options),
        "numerical_options": asdict(numerical_options),
        "integrated_scattering": integrated_scattering,
        "metadata": _plain_json(metadata),
    }


def _grid_identity(
    *,
    elements: tuple[str, ...],
    template_ids: tuple[str, ...],
    options: AtomicTemplateQuadratureOptions1D,
    numerical_options: DirectAtomicNumericalOptions1D,
    require_full_kernel_support: bool,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "weighted_direct_atomic_grid_1d:v2",
        "elements": list(elements),
        "template_ids": list(template_ids),
        "options": asdict(options),
        "numerical_options": asdict(numerical_options),
        "require_full_kernel_support": require_full_kernel_support,
        "metadata": _plain_json(metadata),
    }


def _report_identity(
    *,
    elements: tuple[str, ...],
    sampling_s_A: float,
    sampling_u_A: float,
    base_options: AtomicTemplateQuadratureOptions1D,
    numerical_options: DirectAtomicNumericalOptions1D,
    relative_l2_tolerance: float,
    relative_integral_tolerance: float,
    passed: bool,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "atomic_quadrature_convergence_1d:v2",
        "elements": list(elements),
        "sampling_s_A": sampling_s_A,
        "sampling_u_A": sampling_u_A,
        "base_options": asdict(base_options),
        "numerical_options": asdict(numerical_options),
        "relative_l2_tolerance": relative_l2_tolerance,
        "relative_integral_tolerance": relative_integral_tolerance,
        "passed": passed,
        "metadata": _plain_json(metadata),
    }


def _adaptive_report_identity(
    *,
    elements: tuple[str, ...],
    sampling_s_A: float,
    sampling_u_A: float,
    base_options: AtomicTemplateQuadratureOptions1D,
    base_numerical_options: DirectAtomicNumericalOptions1D,
    relative_l2_tolerance: float,
    relative_integral_tolerance: float,
    passed: bool,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "adaptive_atomic_cubature_convergence_1d:v1",
        "elements": list(elements),
        "sampling_s_A": sampling_s_A,
        "sampling_u_A": sampling_u_A,
        "base_options": asdict(base_options),
        "base_numerical_options": asdict(base_numerical_options),
        "relative_l2_tolerance": relative_l2_tolerance,
        "relative_integral_tolerance": relative_integral_tolerance,
        "passed": passed,
        "metadata": _plain_json(metadata),
    }


def _validate_fail_closed_result(value: Any, *, kind: str) -> None:
    if value.trust_claim is not False:
        raise ValueError(f"{kind} must fail closed")
    if value.trust_reason != _FAIL_CLOSED_REASON:
        raise ValueError("trust_reason must use the module fail-closed policy")
    if tuple(value.limitations) != _LIMITATIONS:
        raise ValueError("limitations must retain the complete module limitations")
    object.__setattr__(value, "limitations", _LIMITATIONS)


def _validate_direct_atomic_template(value: DirectAtomicTemplate1D) -> None:
    symbol = _element(value.element)
    if not isinstance(value.options, AtomicTemplateQuadratureOptions1D):
        raise TypeError("options must be AtomicTemplateQuadratureOptions1D")
    numerical_options = _numerical_options(value.numerical_options)
    ds = _positive("sampling_s_A", value.sampling_s_A)
    du = _positive("sampling_u_A", value.sampling_u_A)
    half_shape = _half_shape(ds, du, value.options.cutoff_A, value.half_shape)
    offset = _fractional_offset(
        value.fractional_offset_A, sampling_s_A=ds, sampling_u_A=du
    )
    shape = (2 * half_shape[0] + 1, 2 * half_shape[1] + 1)
    values = _real_array("values", value.values, ndim=2, shape=shape)
    unit = _real_array(
        "unit_integrated_values", value.unit_integrated_values, ndim=2, shape=shape
    )
    if np.any(values < 0.0) or np.any(unit < 0.0):
        raise ValueError("atomic template values must be non-negative")
    integral = _positive("integrated_scattering", value.integrated_scattering)
    expected_integral = float(np.sum(values, dtype=np.float64) * ds * du)
    if not np.isclose(integral, expected_integral, rtol=5e-14, atol=0.0):
        raise ValueError("integrated_scattering does not match values")
    if not np.allclose(unit, values / integral, rtol=5e-14, atol=0.0):
        raise ValueError("unit_integrated_values do not match values")
    metadata = _jsonable(value.metadata)
    _validate_fail_closed_result(value, kind="direct atomic template")
    digest = _validate_digest(value.template_id, "template_id")
    identity = _template_identity(
        element=symbol,
        sampling_s_A=ds,
        sampling_u_A=du,
        half_shape=half_shape,
        fractional_offset_A=offset,
        options=value.options,
        numerical_options=numerical_options,
        integrated_scattering=integral,
        metadata=metadata,
    )
    expected_digest = _digest(
        {"values": values, "unit_integrated_values": unit}, identity
    )
    if digest != expected_digest:
        raise ValueError("template_id does not match the template content")
    object.__setattr__(value, "element", symbol)
    object.__setattr__(value, "sampling_s_A", ds)
    object.__setattr__(value, "sampling_u_A", du)
    object.__setattr__(value, "half_shape", half_shape)
    object.__setattr__(value, "fractional_offset_A", offset)
    object.__setattr__(value, "values", values)
    object.__setattr__(value, "unit_integrated_values", unit)
    object.__setattr__(value, "integrated_scattering", integral)
    object.__setattr__(value, "numerical_options", numerical_options)
    object.__setattr__(value, "metadata", metadata)


def _validate_weighted_atomic_grid(value: WeightedAtomicPotentialGrid1D) -> None:
    if not isinstance(value.options, AtomicTemplateQuadratureOptions1D):
        raise TypeError("options must be AtomicTemplateQuadratureOptions1D")
    numerical_options = _numerical_options(value.numerical_options)
    if not isinstance(value.require_full_kernel_support, (bool, np.bool_)):
        raise TypeError("require_full_kernel_support must be Boolean")
    require_full_support = bool(value.require_full_kernel_support)
    s_axis, _ = _uniform_axis("s_coordinates_A", value.s_coordinates_A)
    u_axis, _ = _uniform_axis("u_coordinates_A", value.u_coordinates_A)
    values = _real_array(
        "values", value.values, ndim=2, shape=(len(s_axis), len(u_axis))
    )
    if np.any(values < 0.0):
        raise ValueError("weighted atomic potential must be non-negative")
    sites = _real_array("site_coordinates_A", value.site_coordinates_A, ndim=2)
    if sites.shape[1:] != (2,):
        raise ValueError("site_coordinates_A must have shape (n_site, 2)")
    weights = _real_array(
        "scattering_weights",
        value.scattering_weights,
        ndim=1,
        shape=(len(sites),),
    )
    if np.any(weights < 0.0):
        raise ValueError("scattering_weights must be non-negative")
    if isinstance(value.elements, (str, bytes)) or len(value.elements) != len(sites):
        raise ValueError("elements must contain one symbol per site")
    symbols = tuple(_element(symbol) for symbol in value.elements)
    if len(value.template_ids) != len(sites):
        raise ValueError("template_ids must contain one digest per site")
    template_ids = tuple(
        _validate_digest(digest, "template_ids entry") for digest in value.template_ids
    )
    metadata = _jsonable(value.metadata)
    _validate_fail_closed_result(value, kind="weighted atomic grid")
    digest = _validate_digest(value.grid_id, "grid_id")
    identity = _grid_identity(
        elements=symbols,
        template_ids=template_ids,
        options=value.options,
        numerical_options=numerical_options,
        require_full_kernel_support=require_full_support,
        metadata=metadata,
    )
    expected_digest = _digest(
        {
            "values": values,
            "s_coordinates_A": s_axis,
            "u_coordinates_A": u_axis,
            "site_coordinates_A": sites,
            "scattering_weights": weights,
        },
        identity,
    )
    if digest != expected_digest:
        raise ValueError("grid_id does not match the grid content")
    object.__setattr__(value, "values", values)
    object.__setattr__(value, "s_coordinates_A", s_axis)
    object.__setattr__(value, "u_coordinates_A", u_axis)
    object.__setattr__(value, "site_coordinates_A", sites)
    object.__setattr__(value, "scattering_weights", weights)
    object.__setattr__(value, "elements", symbols)
    object.__setattr__(value, "template_ids", template_ids)
    object.__setattr__(value, "numerical_options", numerical_options)
    object.__setattr__(value, "require_full_kernel_support", require_full_support)
    object.__setattr__(value, "metadata", metadata)


def _validate_quadrature_report(value: AtomicQuadratureConvergenceReport1D) -> None:
    if isinstance(value.elements, (str, bytes)) or not value.elements:
        raise ValueError("elements must contain at least one symbol")
    symbols = tuple(_element(symbol) for symbol in value.elements)
    ds = _positive("sampling_s_A", value.sampling_s_A)
    du = _positive("sampling_u_A", value.sampling_u_A)
    if not isinstance(value.base_options, AtomicTemplateQuadratureOptions1D):
        raise TypeError("base_options must be AtomicTemplateQuadratureOptions1D")
    numerical_options = _numerical_options(value.numerical_options)
    offsets = _real_array("fractional_offsets_A", value.fractional_offsets_A, ndim=2)
    if offsets.shape[1:] != (2,) or len(offsets) < 1:
        raise ValueError("fractional_offsets_A must have shape (n_offset, 2)")
    for offset in offsets:
        _fractional_offset(offset, sampling_s_A=ds, sampling_u_A=du)
    pairs_source = np.asarray(value.order_pairs)
    if (
        pairs_source.ndim != 2
        or pairs_source.shape[1:] != (2,)
        or not np.issubdtype(pairs_source.dtype, np.integer)
        or np.issubdtype(pairs_source.dtype, np.bool_)
    ):
        raise TypeError("order_pairs must have integer shape (n_order, 2)")
    if len(pairs_source) < 2:
        raise ValueError("at least two quadrature order pairs are required")
    if np.any(pairs_source < 2) or np.any(pairs_source % 2):
        raise ValueError("every pixel/projection order must be even and at least 2")
    if np.any(np.diff(pairs_source[:, 0]) < 0) or np.any(
        np.diff(pairs_source[:, 1]) < 0
    ):
        raise ValueError("order_pairs must be componentwise non-decreasing")
    if np.any(pairs_source > np.iinfo(np.int32).max):
        raise ValueError("quadrature orders exceed the supported integer range")
    pairs = _readonly(pairs_source, dtype=np.int32)
    maximum_l2 = _real_array(
        "maximum_relative_l2_by_order",
        value.maximum_relative_l2_by_order,
        ndim=1,
        shape=(len(pairs),),
    )
    maximum_integral = _real_array(
        "maximum_relative_integral_error_by_order",
        value.maximum_relative_integral_error_by_order,
        ndim=1,
        shape=(len(pairs),),
    )
    if np.any(maximum_l2 < 0.0) or np.any(maximum_integral < 0.0):
        raise ValueError("quadrature errors must be non-negative")
    l2_tolerance = _positive("relative_l2_tolerance", value.relative_l2_tolerance)
    integral_tolerance = _positive(
        "relative_integral_tolerance", value.relative_integral_tolerance
    )
    if not isinstance(value.passed, (bool, np.bool_)):
        raise TypeError("passed must be Boolean")
    passed = bool(value.passed)
    expected_passed = bool(
        maximum_l2[-2] <= l2_tolerance
        and maximum_integral[-2] <= integral_tolerance
    )
    if passed != expected_passed:
        raise ValueError("passed does not match the declared convergence criterion")
    reference_pair = tuple(int(entry) for entry in value.reference_order_pair)
    if reference_pair != (int(pairs[-1, 0]), int(pairs[-1, 1])):
        raise ValueError("reference_order_pair must equal the highest declared order")
    metadata = _jsonable(value.metadata)
    _validate_fail_closed_result(value, kind="quadrature convergence report")
    digest = _validate_digest(value.report_id, "report_id")
    identity = _report_identity(
        elements=symbols,
        sampling_s_A=ds,
        sampling_u_A=du,
        base_options=value.base_options,
        numerical_options=numerical_options,
        relative_l2_tolerance=l2_tolerance,
        relative_integral_tolerance=integral_tolerance,
        passed=passed,
        metadata=metadata,
    )
    expected_digest = _digest(
        {
            "fractional_offsets_A": offsets,
            "order_pairs": pairs,
            "maximum_relative_l2_by_order": maximum_l2,
            "maximum_relative_integral_error_by_order": maximum_integral,
        },
        identity,
    )
    if digest != expected_digest:
        raise ValueError("report_id does not match the convergence evidence")
    object.__setattr__(value, "elements", symbols)
    object.__setattr__(value, "sampling_s_A", ds)
    object.__setattr__(value, "sampling_u_A", du)
    object.__setattr__(value, "fractional_offsets_A", offsets)
    object.__setattr__(value, "order_pairs", pairs)
    object.__setattr__(value, "maximum_relative_l2_by_order", maximum_l2)
    object.__setattr__(
        value, "maximum_relative_integral_error_by_order", maximum_integral
    )
    object.__setattr__(value, "relative_l2_tolerance", l2_tolerance)
    object.__setattr__(value, "relative_integral_tolerance", integral_tolerance)
    object.__setattr__(value, "passed", passed)
    object.__setattr__(value, "reference_order_pair", reference_pair)
    object.__setattr__(value, "numerical_options", numerical_options)
    object.__setattr__(value, "metadata", metadata)


def _validate_adaptive_cubature_report(
    value: AdaptiveAtomicCubatureConvergenceReport1D,
) -> None:
    if isinstance(value.elements, (str, bytes)) or not value.elements:
        raise ValueError("elements must contain at least one symbol")
    symbols = tuple(_element(symbol) for symbol in value.elements)
    ds = _positive("sampling_s_A", value.sampling_s_A)
    du = _positive("sampling_u_A", value.sampling_u_A)
    if not isinstance(value.base_options, AtomicTemplateQuadratureOptions1D):
        raise TypeError("base_options must be AtomicTemplateQuadratureOptions1D")
    numerical = _numerical_options(value.base_numerical_options)
    if numerical.integration_method != "adaptive_factorized_cubature":
        raise ValueError(
            "base_numerical_options must select adaptive_factorized_cubature"
        )
    offsets = _real_array(
        "fractional_offsets_A", value.fractional_offsets_A, ndim=2
    )
    if offsets.shape[1:] != (2,) or len(offsets) < 1:
        raise ValueError("fractional_offsets_A must have shape (n_offset, 2)")
    for offset in offsets:
        _fractional_offset(offset, sampling_s_A=ds, sampling_u_A=du)
    levels = _real_array("tolerance_levels", value.tolerance_levels, ndim=2)
    if levels.shape[1:] != (2,) or len(levels) < 2 or np.any(levels <= 0.0):
        raise ValueError("tolerance_levels must have positive shape (n_level, 2)")
    differences = np.diff(levels, axis=0)
    if np.any(differences > 0.0) or np.any(
        np.all(differences == 0.0, axis=1)
    ):
        raise ValueError(
            "tolerance_levels must tighten componentwise at every level"
        )
    count = len(levels)
    maximum_l2 = _real_array(
        "maximum_relative_l2_by_level",
        value.maximum_relative_l2_by_level,
        ndim=1,
        shape=(count,),
    )
    maximum_integral = _real_array(
        "maximum_relative_integral_error_by_level",
        value.maximum_relative_integral_error_by_level,
        ndim=1,
        shape=(count,),
    )
    reported_error = _real_array(
        "maximum_reported_template_l2_error_by_level",
        value.maximum_reported_template_l2_error_by_level,
        ndim=1,
        shape=(count,),
    )
    evaluations_source = np.asarray(value.maximum_function_evaluations_by_level)
    if (
        evaluations_source.shape != (count,)
        or not np.issubdtype(evaluations_source.dtype, np.integer)
        or np.issubdtype(evaluations_source.dtype, np.bool_)
        or np.any(evaluations_source <= 0)
    ):
        raise TypeError(
            "maximum_function_evaluations_by_level must contain positive integers"
        )
    evaluations = _readonly(evaluations_source, dtype=np.int64)
    if (
        np.any(maximum_l2 < 0.0)
        or np.any(maximum_integral < 0.0)
        or np.any(reported_error < 0.0)
        or maximum_l2[-1] != 0.0
        or maximum_integral[-1] != 0.0
    ):
        raise ValueError("adaptive convergence errors are invalid")
    l2_tolerance = _positive("relative_l2_tolerance", value.relative_l2_tolerance)
    integral_tolerance = _positive(
        "relative_integral_tolerance", value.relative_integral_tolerance
    )
    expected_passed = bool(
        np.max(maximum_l2[:-1]) <= l2_tolerance
        and np.max(maximum_integral[:-1]) <= integral_tolerance
    )
    if not isinstance(value.passed, (bool, np.bool_)) or bool(value.passed) != expected_passed:
        raise ValueError("passed does not match the adaptive convergence criterion")
    metadata = _jsonable(value.metadata)
    _validate_fail_closed_result(value, kind="adaptive cubature convergence report")
    digest = _validate_digest(value.report_id, "report_id")
    identity = _adaptive_report_identity(
        elements=symbols,
        sampling_s_A=ds,
        sampling_u_A=du,
        base_options=value.base_options,
        base_numerical_options=numerical,
        relative_l2_tolerance=l2_tolerance,
        relative_integral_tolerance=integral_tolerance,
        passed=expected_passed,
        metadata=metadata,
    )
    expected_digest = _digest(
        {
            "fractional_offsets_A": offsets,
            "tolerance_levels": levels,
            "maximum_relative_l2_by_level": maximum_l2,
            "maximum_relative_integral_error_by_level": maximum_integral,
            "maximum_reported_template_l2_error_by_level": reported_error,
            "maximum_function_evaluations_by_level": evaluations,
        },
        identity,
    )
    if digest != expected_digest:
        raise ValueError("report_id does not match adaptive convergence evidence")
    object.__setattr__(value, "elements", symbols)
    object.__setattr__(value, "fractional_offsets_A", offsets)
    object.__setattr__(value, "tolerance_levels", levels)
    object.__setattr__(value, "maximum_relative_l2_by_level", maximum_l2)
    object.__setattr__(
        value, "maximum_relative_integral_error_by_level", maximum_integral
    )
    object.__setattr__(
        value, "maximum_reported_template_l2_error_by_level", reported_error
    )
    object.__setattr__(
        value, "maximum_function_evaluations_by_level", evaluations
    )
    object.__setattr__(value, "relative_l2_tolerance", l2_tolerance)
    object.__setattr__(value, "relative_integral_tolerance", integral_tolerance)
    object.__setattr__(value, "passed", expected_passed)
    object.__setattr__(value, "sampling_s_A", ds)
    object.__setattr__(value, "sampling_u_A", du)
    object.__setattr__(value, "base_numerical_options", numerical)
    object.__setattr__(value, "metadata", metadata)


def _factorized_interval_gaussian_integral(
    rate_A2: float, lower_A: np.ndarray, upper_A: np.ndarray
) -> np.ndarray:
    from scipy.special import erf

    root = math.sqrt(rate_A2)
    return (
        math.sqrt(math.pi)
        / (2.0 * root)
        * (erf(root * upper_A) - erf(root * lower_A))
    )


def _factorized_kirkland_template(
    element: str,
    options: AtomicTemplateQuadratureOptions1D,
    numerical_options: DirectAtomicNumericalOptions1D,
    sampling_s_A: float,
    sampling_u_A: float,
    half_s: int,
    half_u: int,
    offset_s_A: float,
    offset_u_A: float,
) -> tuple[
    np.ndarray, tuple[int, int, int, float, float, int, int, str]
]:
    """Integrate the complete finite Kirkland template by one vector cubature."""

    from abtem.parametrizations import KirklandParametrization
    from scipy.integrate import quad_vec
    from scipy.special import erf

    parameterization = KirklandParametrization()
    try:
        parameters = np.asarray(
            parameterization.scaled_parameters(element, "potential"),
            dtype=np.float64,
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"no Kirkland potential is available for {element!r}") from error
    if parameters.shape != (4, 3) or np.any(~np.isfinite(parameters)) or np.any(
        parameters <= 0.0
    ):
        raise ValueError("Kirkland potential parameters are invalid")
    s_centres = np.arange(-half_s, half_s + 1, dtype=np.float64) * sampling_s_A
    u_centres = np.arange(-half_u, half_u + 1, dtype=np.float64) * sampling_u_A
    s_lower = s_centres - 0.5 * sampling_s_A - offset_s_A
    s_upper = s_centres + 0.5 * sampling_s_A - offset_s_A
    u_lower = u_centres - 0.5 * sampling_u_A - offset_u_A
    u_upper = u_centres + 0.5 * sampling_u_A - offset_u_A
    z_lower = np.asarray([-0.5 * options.projection_width_A])
    z_upper = np.asarray([0.5 * options.projection_width_A])
    shape = (len(s_centres), len(u_centres))
    voxel_volume = (
        sampling_s_A * sampling_u_A * options.projection_width_A
    )

    gaussian_integral = np.zeros(shape, dtype=np.float64)
    for amplitude, rate in zip(parameters[2], parameters[3], strict=True):
        s_factor = _factorized_interval_gaussian_integral(rate, s_lower, s_upper)
        u_factor = _factorized_interval_gaussian_integral(rate, u_lower, u_upper)
        z_factor = float(
            _factorized_interval_gaussian_integral(rate, z_lower, z_upper)[0]
        )
        gaussian_integral += amplitude * np.outer(s_factor, u_factor) * z_factor

    yukawa_amplitudes = np.asarray(parameters[0], dtype=np.float64)
    yukawa_rates = np.asarray(parameters[1], dtype=np.float64)

    def endpoint_delta(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        return np.sign(upper) - np.sign(lower)

    endpoint_outer = (
        endpoint_delta(s_lower, s_upper)[:, None]
        * endpoint_delta(u_lower, u_upper)[None, :]
        * float(endpoint_delta(z_lower, z_upper)[0])
    )

    def integrand(x_value: float) -> np.ndarray:
        x = float(x_value)
        if x <= np.finfo(np.float64).tiny:
            return np.zeros(shape, dtype=np.float64)
        if x >= 1.0:
            return (
                math.pi
                / 8.0
                * float(np.sum(yukawa_amplitudes))
                * endpoint_outer
            ).reshape(-1)
        t = x / (1.0 - x)
        root_t = math.sqrt(t)
        s_delta = erf(root_t * s_upper) - erf(root_t * s_lower)
        u_delta = erf(root_t * u_upper) - erf(root_t * u_lower)
        z_delta = float(erf(root_t * z_upper[0]) - erf(root_t * z_lower[0]))
        outer = np.outer(s_delta, u_delta) * z_delta
        result = np.zeros(shape, dtype=np.float64)
        common_log = math.log(math.pi / 8.0) - 2.0 * math.log(x)
        for amplitude, rate in zip(
            yukawa_amplitudes, yukawa_rates, strict=True
        ):
            exponent = common_log - rate**2 * (1.0 - x) / (4.0 * x)
            if exponent > math.log(np.finfo(np.float64).tiny):
                result += amplitude * math.exp(exponent) * outer
        return result.reshape(-1)

    safety = numerical_options.adaptive_error_safety_factor
    yukawa_integral, raw_error, info = quad_vec(
        integrand,
        0.0,
        1.0,
        epsabs=numerical_options.adaptive_absolute_l2_tolerance / safety,
        epsrel=numerical_options.adaptive_relative_tolerance / safety,
        norm="2",
        quadrature=numerical_options.adaptive_quadrature_rule,
        limit=numerical_options.adaptive_max_intervals,
        full_output=True,
    )
    yukawa_integral = np.asarray(yukawa_integral, dtype=np.float64).reshape(shape)
    estimated_integral_l2_error = safety * float(raw_error)
    integral_tolerance = max(
        numerical_options.adaptive_absolute_l2_tolerance,
        numerical_options.adaptive_relative_tolerance
        * float(np.linalg.norm(yukawa_integral)),
    )
    if not bool(info.success):
        raise RuntimeError(
            "adaptive factorized Kirkland cubature did not converge: "
            f"{info.message}"
        )
    if int(info.neval) > numerical_options.adaptive_max_evaluations:
        raise RuntimeError(
            "adaptive factorized Kirkland cubature exceeded "
            "adaptive_max_evaluations"
        )
    if estimated_integral_l2_error > integral_tolerance:
        raise RuntimeError(
            "adaptive factorized Kirkland cubature error estimate exceeds its "
            "authenticated whole-template tolerance"
        )
    total_integral = yukawa_integral + gaussian_integral
    values = total_integral / voxel_volume
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError(
            "adaptive factorized Kirkland cubature produced invalid values"
        )
    diagnostic = (
        int(values.size),
        int(info.neval),
        int(len(info.intervals)),
        estimated_integral_l2_error / voxel_volume,
        integral_tolerance / voxel_volume,
        3,
        int(info.status),
        str(info.message),
    )
    return values, diagnostic


@lru_cache(maxsize=256)
def _direct_template_payload(
    element: str,
    options: AtomicTemplateQuadratureOptions1D,
    numerical_options: DirectAtomicNumericalOptions1D,
    sampling_s_A: float,
    sampling_u_A: float,
    half_s: int,
    half_u: int,
    offset_s_A: float,
    offset_u_A: float,
) -> tuple[
    bytes, tuple[int, int, int, float, float, int, int, str]
]:
    from abtem.parametrizations import KirklandParametrization

    if (
        numerical_options.integration_method
        == "adaptive_factorized_cubature"
    ):
        values, diagnostic = _factorized_kirkland_template(
            element,
            options,
            numerical_options,
            sampling_s_A,
            sampling_u_A,
            half_s,
            half_u,
            offset_s_A,
            offset_u_A,
        )
        return values.tobytes(order="C"), diagnostic

    # Tensor-product mode is the compatibility diagnostic for the established
    # direct Si renderer.  Reuse that renderer for silicon so both APIs also
    # share its cache entry: abTEM's global precision setting is mutable, and
    # two otherwise identical independent caches can capture different
    # parameter dtypes when another workflow changes that setting mid-process.
    # Blind positive-edit truth uses the factorized policy above and therefore
    # does not take this compatibility branch.
    if element == "Si":
        established = render_si_atomic_template_1d(
            sampling_s_A=sampling_s_A,
            sampling_u_A=sampling_u_A,
            options=options,
            half_shape=(half_s, half_u),
            fractional_offset_A=(offset_s_A, offset_u_A),
        )
        diagnostic = (
            0,
            0,
            0,
            0.0,
            0.0,
            0,
            -1,
            "adaptive factorized cubature not evaluated",
        )
        return np.asarray(established.values).tobytes(order="C"), diagnostic

    pixel_nodes, pixel_weights = np.polynomial.legendre.leggauss(
        options.pixel_quadrature_order
    )
    z_nodes, z_weights = np.polynomial.legendre.leggauss(
        options.projection_quadrature_order
    )
    s_centres = np.arange(-half_s, half_s + 1, dtype=np.float64) * sampling_s_A
    u_centres = np.arange(-half_u, half_u + 1, dtype=np.float64) * sampling_u_A
    s_samples = (
        s_centres[:, None]
        + 0.5 * sampling_s_A * pixel_nodes[None, :]
        - offset_s_A
    )
    u_samples = (
        u_centres[:, None]
        + 0.5 * sampling_u_A * pixel_nodes[None, :]
        - offset_u_A
    )
    z_samples = 0.5 * options.projection_width_A * z_nodes
    try:
        radial_potential = KirklandParametrization().potential(element)
    except (KeyError, ValueError) as error:
        raise ValueError(f"no Kirkland potential is available for {element!r}") from error
    tensor_weights = (
        pixel_weights[:, None, None]
        * pixel_weights[None, :, None]
        * z_weights[None, None, :]
        / 8.0
    )
    values = np.empty((s_centres.size, u_centres.size), dtype=np.float64)
    for s_index, samples_s in enumerate(s_samples):
        radius = np.sqrt(
            samples_s[:, None, None, None] ** 2
            + u_samples[None, :, :, None] ** 2
            + z_samples[None, None, None, :] ** 2
        )
        evaluated = np.asarray(radial_potential(radius), dtype=np.float64)
        values[s_index] = np.sum(
            evaluated * tensor_weights[:, None, :, :], axis=(0, 2, 3)
        )
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError(
            "direct atomic quadrature produced non-finite or negative values"
        )
    diagnostic = (
        0,
        0,
        0,
        0.0,
        0.0,
        0,
        -1,
        "adaptive factorized cubature not evaluated",
    )
    return values.tobytes(order="C"), diagnostic


def render_direct_atomic_template_1d(
    element: str,
    *,
    sampling_s_A: float,
    sampling_u_A: float,
    options: AtomicTemplateQuadratureOptions1D,
    numerical_options: DirectAtomicNumericalOptions1D | None = None,
    half_shape: Sequence[int] | None = None,
    fractional_offset_A: Sequence[float] = (0.0, 0.0),
    metadata: Mapping[str, Any] | None = None,
) -> DirectAtomicTemplate1D:
    """Render one exact-subpixel atomic template under an explicit policy."""
    symbol = _element(element)
    if not isinstance(options, AtomicTemplateQuadratureOptions1D):
        raise TypeError("options must be AtomicTemplateQuadratureOptions1D")
    numerical = _numerical_options(numerical_options)
    ds = _positive("sampling_s_A", sampling_s_A)
    du = _positive("sampling_u_A", sampling_u_A)
    half_s, half_u = _half_shape(ds, du, options.cutoff_A, half_shape)
    offset = _fractional_offset(
        fractional_offset_A, sampling_s_A=ds, sampling_u_A=du
    )
    if numerical.integration_method == "tensor_product":
        evaluations = (
            (2 * half_s + 1)
            * (2 * half_u + 1)
            * options.pixel_quadrature_order**2
            * options.projection_quadrature_order
        )
        if evaluations > options.maximum_quadrature_evaluations:
            raise ValueError(
                "requested direct template exceeds maximum_quadrature_evaluations"
            )
    elif numerical.adaptive_max_evaluations > options.maximum_quadrature_evaluations:
        raise ValueError(
            "adaptive_max_evaluations exceeds the atomic template's authenticated "
            "maximum_quadrature_evaluations"
        )
    payload, diagnostic = _direct_template_payload(
        symbol,
        options,
        numerical,
        ds,
        du,
        half_s,
        half_u,
        offset[0],
        offset[1],
    )
    values = np.frombuffer(payload, dtype=np.float64).reshape(
        2 * half_s + 1, 2 * half_u + 1
    )
    integral = float(np.sum(values, dtype=np.float64) * ds * du)
    if not np.isfinite(integral) or integral <= 0.0:
        raise FloatingPointError("direct atomic template has invalid integral")
    unit = np.asarray(values) / integral
    metadata_values = dict(metadata or {})
    numerical_metadata = {
        "direct_atomic_numerical_options": asdict(numerical),
        "adaptive_factorized_cubature": {
            "evaluated": bool(
                numerical.integration_method
                == "adaptive_factorized_cubature"
            ),
            "converged": bool(
                numerical.integration_method
                == "adaptive_factorized_cubature"
            ),
            "finite_voxel_count": int(diagnostic[0]),
            "function_evaluations": int(diagnostic[1]),
            "subinterval_count": int(diagnostic[2]),
            "estimated_template_l2_error": float(diagnostic[3]),
            "template_l2_tolerance": float(diagnostic[4]),
            "analytic_gaussian_term_count": int(diagnostic[5]),
            "scipy_status": int(diagnostic[6]),
            "scipy_message": str(diagnostic[7]),
            "runtime_provenance": (
                dict(_adaptive_runtime_provenance())
                if numerical.integration_method
                == "adaptive_factorized_cubature"
                else {}
            ),
            "status": (
                "target_precision_reached"
                if numerical.integration_method
                == "adaptive_factorized_cubature"
                else "not_evaluated_tensor_product"
            ),
        },
    }
    for name, expected in numerical_metadata.items():
        if name in metadata_values and metadata_values[name] != expected:
            raise ValueError(f"metadata field {name!r} is reserved by the renderer")
        metadata_values[name] = expected
    metadata = _jsonable(metadata_values)
    identity = _template_identity(
        element=symbol,
        sampling_s_A=ds,
        sampling_u_A=du,
        half_shape=(half_s, half_u),
        fractional_offset_A=offset,
        options=options,
        numerical_options=numerical,
        integrated_scattering=integral,
        metadata=metadata,
    )
    template_id = _digest(
        {"values": values, "unit_integrated_values": unit}, identity
    )
    return DirectAtomicTemplate1D(
        element=symbol,
        values=_readonly(values),
        unit_integrated_values=_readonly(unit),
        integrated_scattering=integral,
        sampling_s_A=ds,
        sampling_u_A=du,
        half_shape=(half_s, half_u),
        fractional_offset_A=offset,
        options=options,
        template_id=template_id,
        numerical_options=numerical,
        metadata=metadata,
    )


def accumulate_weighted_atomic_potential_1d(
    site_coordinates_A: Any,
    elements: Sequence[str],
    scattering_weights: Any,
    *,
    s_coordinates_A: Any,
    u_coordinates_A: Any,
    options: AtomicTemplateQuadratureOptions1D,
    numerical_options: DirectAtomicNumericalOptions1D | None = None,
    require_full_kernel_support: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> WeightedAtomicPotentialGrid1D:
    """Accumulate positive weighted centres without interpolation or objects."""
    if not isinstance(options, AtomicTemplateQuadratureOptions1D):
        raise TypeError("options must be AtomicTemplateQuadratureOptions1D")
    numerical = _numerical_options(numerical_options)
    sites = _real_array("site_coordinates_A", site_coordinates_A, ndim=2)
    if sites.shape[1:] != (2,):
        raise ValueError("site_coordinates_A must have finite shape (n_site, 2)")
    weights = _real_array(
        "scattering_weights",
        scattering_weights,
        ndim=1,
        shape=(len(sites),),
    )
    if np.any(weights < 0.0):
        raise ValueError("scattering_weights must be finite and non-negative")
    if isinstance(elements, (str, bytes)) or len(elements) != len(sites):
        raise ValueError("elements must contain one symbol per site")
    symbols = tuple(_element(value) for value in elements)
    canonical_order = sorted(
        range(len(sites)),
        key=lambda index: (
            symbols[index],
            float(sites[index, 0]),
            float(sites[index, 1]),
            float(weights[index]),
        ),
    )
    if canonical_order:
        canonical_indices = np.asarray(canonical_order, dtype=np.int64)
        sites = _readonly(np.asarray(sites)[canonical_indices], dtype=np.float64)
        weights = _readonly(np.asarray(weights)[canonical_indices], dtype=np.float64)
        symbols = tuple(symbols[index] for index in canonical_order)
    s_axis, ds = _uniform_axis("s_coordinates_A", s_coordinates_A)
    u_axis, du = _uniform_axis("u_coordinates_A", u_coordinates_A)
    if not isinstance(require_full_kernel_support, (bool, np.bool_)):
        raise TypeError("require_full_kernel_support must be Boolean")
    result = np.zeros((len(s_axis), len(u_axis)), dtype=np.float64)
    template_ids: list[str] = []
    clipped_kernel_count = 0
    for site, symbol, weight in zip(sites, symbols, weights, strict=True):
        continuous_index = np.asarray(
            [(site[0] - s_axis[0]) / ds, (site[1] - u_axis[0]) / du]
        )
        anchor = np.floor(continuous_index + 0.5).astype(np.int64)
        if np.any(anchor < 0) or anchor[0] >= len(s_axis) or anchor[1] >= len(u_axis):
            raise ValueError("atomic site centre lies outside the finite grid")
        anchor_position = np.asarray(
            [s_axis[0] + anchor[0] * ds, u_axis[0] + anchor[1] * du]
        )
        offset_array = np.asarray(site) - anchor_position
        if offset_array[0] >= 0.5 * ds:
            anchor[0] += 1
            offset_array[0] -= ds
        elif offset_array[0] < -0.5 * ds:
            anchor[0] -= 1
            offset_array[0] += ds
        if offset_array[1] >= 0.5 * du:
            anchor[1] += 1
            offset_array[1] -= du
        elif offset_array[1] < -0.5 * du:
            anchor[1] -= 1
            offset_array[1] += du
        if np.any(anchor < 0) or anchor[0] >= len(s_axis) or anchor[1] >= len(u_axis):
            raise ValueError("atomic site centre lies outside the finite grid")
        offset = (float(offset_array[0]), float(offset_array[1]))
        template = render_direct_atomic_template_1d(
            symbol,
            sampling_s_A=ds,
            sampling_u_A=du,
            options=options,
            numerical_options=numerical,
            fractional_offset_A=offset,
        )
        template_ids.append(template.template_id)
        start = anchor - np.asarray(template.half_shape, dtype=np.int64)
        stop = start + np.asarray(template.values.shape, dtype=np.int64)
        fully_supported = bool(
            np.all(start >= 0)
            and stop[0] <= len(s_axis)
            and stop[1] <= len(u_axis)
        )
        if not fully_supported:
            clipped_kernel_count += 1
        if require_full_kernel_support and not fully_supported:
            raise ValueError(
                "an atomic kernel would be clipped by the finite grid"
            )
        source_start = np.maximum(-start, 0)
        source_stop = np.asarray(template.values.shape) - np.maximum(
            stop - np.asarray(result.shape), 0
        )
        destination_start = np.maximum(start, 0)
        destination_stop = destination_start + (source_stop - source_start)
        result[
            destination_start[0] : destination_stop[0],
            destination_start[1] : destination_stop[1],
        ] += float(weight) * template.values[
            source_start[0] : source_stop[0],
            source_start[1] : source_stop[1],
        ]
    metadata_values = dict(metadata or {})
    policy_metadata = {
        "kernel_support_policy": (
            "full_support_required"
            if require_full_kernel_support
            else "finite_grid_clipping_allowed_fail_closed"
        ),
        "clipped_kernel_count": clipped_kernel_count,
        "direct_atomic_numerical_options": asdict(numerical),
        "adaptive_runtime_provenance": (
            dict(_adaptive_runtime_provenance())
            if numerical.integration_method == "adaptive_factorized_cubature"
            else {}
        ),
    }
    for name, expected in policy_metadata.items():
        if name in metadata_values and metadata_values[name] != expected:
            raise ValueError(f"metadata field {name!r} is reserved by the renderer")
        metadata_values[name] = expected
    metadata = _jsonable(metadata_values)
    identity = _grid_identity(
        elements=symbols,
        template_ids=tuple(template_ids),
        options=options,
        numerical_options=numerical,
        require_full_kernel_support=bool(require_full_kernel_support),
        metadata=metadata,
    )
    grid_id = _digest(
        {
            "values": result,
            "s_coordinates_A": s_axis,
            "u_coordinates_A": u_axis,
            "site_coordinates_A": sites,
            "scattering_weights": weights,
        },
        identity,
    )
    return WeightedAtomicPotentialGrid1D(
        values=_readonly(result),
        s_coordinates_A=s_axis,
        u_coordinates_A=u_axis,
        site_coordinates_A=_readonly(sites, dtype=np.float64),
        elements=symbols,
        scattering_weights=_readonly(weights, dtype=np.float64),
        template_ids=tuple(template_ids),
        options=options,
        require_full_kernel_support=bool(require_full_kernel_support),
        grid_id=grid_id,
        numerical_options=numerical,
        metadata=metadata,
    )


def sweep_adaptive_atomic_cubature_convergence_1d(
    elements: Sequence[str],
    *,
    sampling_s_A: float,
    sampling_u_A: float,
    base_options: AtomicTemplateQuadratureOptions1D,
    tolerance_levels: Sequence[tuple[float, float]],
    fractional_offsets_A: Sequence[tuple[float, float]] = ((0.0, 0.0),),
    base_numerical_options: DirectAtomicNumericalOptions1D | None = None,
    relative_l2_tolerance: float = 1e-4,
    relative_integral_tolerance: float = 1e-4,
    metadata: Mapping[str, Any] | None = None,
) -> AdaptiveAtomicCubatureConvergenceReport1D:
    """Sweep adaptive tolerances against the tightest declared reference."""

    if not isinstance(base_options, AtomicTemplateQuadratureOptions1D):
        raise TypeError("base_options must be AtomicTemplateQuadratureOptions1D")
    numerical = (
        DirectAtomicNumericalOptions1D(
            integration_method="adaptive_factorized_cubature"
        )
        if base_numerical_options is None
        else _numerical_options(base_numerical_options)
    )
    if numerical.integration_method != "adaptive_factorized_cubature":
        raise ValueError(
            "base_numerical_options must select adaptive_factorized_cubature"
        )
    if isinstance(elements, (str, bytes)):
        raise TypeError("elements must be a sequence of element symbols")
    symbols = tuple(_element(value) for value in elements)
    if not symbols:
        raise ValueError("elements must not be empty")
    ds = _positive("sampling_s_A", sampling_s_A)
    du = _positive("sampling_u_A", sampling_u_A)
    levels = _real_array("tolerance_levels", tolerance_levels, ndim=2)
    if levels.shape[1:] != (2,) or len(levels) < 2 or np.any(levels <= 0.0):
        raise ValueError("tolerance_levels must have positive shape (n_level, 2)")
    differences = np.diff(levels, axis=0)
    if np.any(differences > 0.0) or np.any(
        np.all(differences == 0.0, axis=1)
    ):
        raise ValueError(
            "tolerance_levels must tighten componentwise at every level"
        )
    offsets = _real_array(
        "fractional_offsets_A", fractional_offsets_A, ndim=2
    )
    if offsets.shape[1:] != (2,) or not len(offsets):
        raise ValueError("fractional_offsets_A must have shape (n_offset, 2)")
    for offset in offsets:
        _fractional_offset(offset, sampling_s_A=ds, sampling_u_A=du)
    l2_tolerance = _positive("relative_l2_tolerance", relative_l2_tolerance)
    integral_tolerance = _positive(
        "relative_integral_tolerance", relative_integral_tolerance
    )
    templates: list[list[np.ndarray]] = []
    integrals: list[list[float]] = []
    reported_errors: list[float] = []
    maximum_evaluations: list[int] = []
    for relative_tolerance, absolute_tolerance in levels:
        level_options = replace(
            numerical,
            adaptive_relative_tolerance=float(relative_tolerance),
            adaptive_absolute_l2_tolerance=float(absolute_tolerance),
        )
        level_templates = []
        level_integrals = []
        level_reported_errors = []
        level_evaluations = []
        for symbol in symbols:
            for offset in offsets:
                template = render_direct_atomic_template_1d(
                    symbol,
                    sampling_s_A=ds,
                    sampling_u_A=du,
                    options=base_options,
                    numerical_options=level_options,
                    fractional_offset_A=tuple(offset),
                )
                evidence = template.metadata["adaptive_factorized_cubature"]
                if not evidence["converged"]:
                    raise RuntimeError(
                        "adaptive template returned without converged cubature evidence"
                    )
                level_templates.append(np.asarray(template.values))
                level_integrals.append(template.integrated_scattering)
                level_reported_errors.append(
                    float(evidence["estimated_template_l2_error"])
                )
                level_evaluations.append(int(evidence["function_evaluations"]))
        templates.append(level_templates)
        integrals.append(level_integrals)
        reported_errors.append(max(level_reported_errors))
        maximum_evaluations.append(max(level_evaluations))
    reference_templates = templates[-1]
    reference_integrals = np.asarray(integrals[-1], dtype=np.float64)
    maximum_l2 = []
    maximum_integral = []
    for level_templates, level_integrals in zip(templates, integrals, strict=True):
        maximum_l2.append(
            max(
                float(np.linalg.norm(value - reference) / np.linalg.norm(reference))
                for value, reference in zip(
                    level_templates, reference_templates, strict=True
                )
            )
        )
        maximum_integral.append(
            float(
                np.max(
                    np.abs(np.asarray(level_integrals) - reference_integrals)
                    / np.abs(reference_integrals)
                )
            )
        )
    maximum_l2_array = np.asarray(maximum_l2, dtype=np.float64)
    maximum_integral_array = np.asarray(maximum_integral, dtype=np.float64)
    reported_error_array = np.asarray(reported_errors, dtype=np.float64)
    evaluation_array = np.asarray(maximum_evaluations, dtype=np.int64)
    passed = bool(
        np.max(maximum_l2_array[:-1]) <= l2_tolerance
        and np.max(maximum_integral_array[:-1]) <= integral_tolerance
    )
    metadata_values = dict(metadata or {})
    reserved = {
        "comparison_reference": "tightest_declared_adaptive_tolerance",
        "tensor_order_diagnostic_is_separate": True,
        "whole_finite_template_vector_cubature": True,
        "adaptive_runtime_provenance": dict(_adaptive_runtime_provenance()),
    }
    for name, expected in reserved.items():
        if name in metadata_values and metadata_values[name] != expected:
            raise ValueError(f"metadata field {name!r} is reserved by the sweep")
        metadata_values[name] = expected
    metadata_result = _jsonable(metadata_values)
    identity = _adaptive_report_identity(
        elements=symbols,
        sampling_s_A=ds,
        sampling_u_A=du,
        base_options=base_options,
        base_numerical_options=numerical,
        relative_l2_tolerance=l2_tolerance,
        relative_integral_tolerance=integral_tolerance,
        passed=passed,
        metadata=metadata_result,
    )
    report_id = _digest(
        {
            "fractional_offsets_A": offsets,
            "tolerance_levels": levels,
            "maximum_relative_l2_by_level": maximum_l2_array,
            "maximum_relative_integral_error_by_level": maximum_integral_array,
            "maximum_reported_template_l2_error_by_level": reported_error_array,
            "maximum_function_evaluations_by_level": evaluation_array,
        },
        identity,
    )
    return AdaptiveAtomicCubatureConvergenceReport1D(
        elements=symbols,
        fractional_offsets_A=_readonly(offsets),
        tolerance_levels=_readonly(levels),
        maximum_relative_l2_by_level=_readonly(maximum_l2_array),
        maximum_relative_integral_error_by_level=_readonly(
            maximum_integral_array
        ),
        maximum_reported_template_l2_error_by_level=_readonly(
            reported_error_array
        ),
        maximum_function_evaluations_by_level=_readonly(
            evaluation_array, dtype=np.int64
        ),
        relative_l2_tolerance=l2_tolerance,
        relative_integral_tolerance=integral_tolerance,
        passed=passed,
        sampling_s_A=ds,
        sampling_u_A=du,
        base_options=base_options,
        base_numerical_options=numerical,
        report_id=report_id,
        metadata=metadata_result,
    )


def sweep_atomic_quadrature_convergence_1d(
    elements: Sequence[str],
    *,
    sampling_s_A: float,
    sampling_u_A: float,
    base_options: AtomicTemplateQuadratureOptions1D,
    numerical_options: DirectAtomicNumericalOptions1D | None = None,
    order_pairs: Sequence[tuple[int, int]],
    fractional_offsets_A: Sequence[tuple[float, float]] = ((0.0, 0.0),),
    relative_l2_tolerance: float = 1e-4,
    relative_integral_tolerance: float = 1e-4,
    metadata: Mapping[str, Any] | None = None,
) -> AtomicQuadratureConvergenceReport1D:
    """Sweep direct-quadrature orders at fixed physical sampling and cutoff."""
    if not isinstance(base_options, AtomicTemplateQuadratureOptions1D):
        raise TypeError("base_options must be AtomicTemplateQuadratureOptions1D")
    numerical = _numerical_options(numerical_options)
    if numerical.integration_method != "tensor_product":
        raise ValueError(
            "order_pairs are only meaningful for integration_method='tensor_product'; "
            "use sweep_adaptive_atomic_cubature_convergence_1d for adaptive "
            "tolerance evidence"
        )
    if isinstance(elements, (str, bytes)):
        raise TypeError("elements must be a sequence of element symbols")
    symbols = tuple(_element(value) for value in elements)
    if not symbols:
        raise ValueError("elements must not be empty")
    ds = _positive("sampling_s_A", sampling_s_A)
    du = _positive("sampling_u_A", sampling_u_A)
    pairs = np.asarray(order_pairs)
    if pairs.ndim != 2 or pairs.shape[1:] != (2,) or not np.issubdtype(
        pairs.dtype, np.integer
    ):
        raise TypeError("order_pairs must have integer shape (n_order, 2)")
    if len(pairs) < 2:
        raise ValueError("at least two quadrature order pairs are required")
    if np.any(pairs < 2) or np.any(pairs % 2):
        raise ValueError("every pixel/projection order must be even and at least 2")
    if np.any(np.diff(pairs[:, 0]) < 0) or np.any(np.diff(pairs[:, 1]) < 0):
        raise ValueError("order_pairs must be componentwise non-decreasing")
    if np.any(pairs > np.iinfo(np.int32).max):
        raise ValueError("quadrature orders exceed the supported integer range")
    pairs = _readonly(pairs, dtype=np.int32)
    offsets = _real_array(
        "fractional_offsets_A", fractional_offsets_A, ndim=2
    )
    if offsets.ndim != 2 or offsets.shape[1:] != (2,) or not len(offsets):
        raise ValueError("fractional_offsets_A must have shape (n_offset, 2)")
    for offset in offsets:
        _fractional_offset(offset, sampling_s_A=ds, sampling_u_A=du)
    l2_tolerance = _positive("relative_l2_tolerance", relative_l2_tolerance)
    integral_tolerance = _positive(
        "relative_integral_tolerance", relative_integral_tolerance
    )
    templates: list[list[np.ndarray]] = []
    integrals: list[list[float]] = []
    for pixel_order, projection_order in pairs:
        options = replace(
            base_options,
            pixel_quadrature_order=int(pixel_order),
            projection_quadrature_order=int(projection_order),
        )
        order_templates: list[np.ndarray] = []
        order_integrals: list[float] = []
        for symbol in symbols:
            for offset in offsets:
                template = render_direct_atomic_template_1d(
                    symbol,
                    sampling_s_A=ds,
                    sampling_u_A=du,
                    options=options,
                    numerical_options=numerical,
                    fractional_offset_A=tuple(offset),
                )
                order_templates.append(np.asarray(template.values))
                order_integrals.append(template.integrated_scattering)
        templates.append(order_templates)
        integrals.append(order_integrals)
    reference_templates = templates[-1]
    reference_integrals = np.asarray(integrals[-1])
    maximum_l2 = []
    maximum_integral = []
    for order_templates, order_integrals in zip(templates, integrals, strict=True):
        errors_l2 = [
            float(np.linalg.norm(value - reference) / np.linalg.norm(reference))
            for value, reference in zip(
                order_templates, reference_templates, strict=True
            )
        ]
        errors_integral = np.abs(
            np.asarray(order_integrals) - reference_integrals
        ) / np.abs(reference_integrals)
        maximum_l2.append(max(errors_l2))
        maximum_integral.append(float(np.max(errors_integral)))
    maximum_l2_array = np.asarray(maximum_l2, dtype=np.float64)
    maximum_integral_array = np.asarray(maximum_integral, dtype=np.float64)
    # The highest order is the numerical reference. Acceptance asks whether the
    # immediately preceding declared order is already within tolerance.
    passed = bool(
        maximum_l2_array[-2] <= l2_tolerance
        and maximum_integral_array[-2] <= integral_tolerance
    )
    metadata = _jsonable(metadata)
    identity = _report_identity(
        elements=symbols,
        sampling_s_A=ds,
        sampling_u_A=du,
        base_options=base_options,
        numerical_options=numerical,
        relative_l2_tolerance=l2_tolerance,
        relative_integral_tolerance=integral_tolerance,
        passed=passed,
        metadata=metadata,
    )
    report_id = _digest(
        {
            "fractional_offsets_A": offsets,
            "order_pairs": pairs,
            "maximum_relative_l2_by_order": maximum_l2_array,
            "maximum_relative_integral_error_by_order": maximum_integral_array,
        },
        identity,
    )
    return AtomicQuadratureConvergenceReport1D(
        elements=symbols,
        fractional_offsets_A=_readonly(offsets),
        order_pairs=_readonly(pairs, dtype=np.int32),
        maximum_relative_l2_by_order=_readonly(maximum_l2_array),
        maximum_relative_integral_error_by_order=_readonly(
            maximum_integral_array
        ),
        relative_l2_tolerance=l2_tolerance,
        relative_integral_tolerance=integral_tolerance,
        passed=passed,
        reference_order_pair=(int(pairs[-1, 0]), int(pairs[-1, 1])),
        sampling_s_A=ds,
        sampling_u_A=du,
        base_options=base_options,
        report_id=report_id,
        numerical_options=numerical,
        metadata=metadata,
    )
