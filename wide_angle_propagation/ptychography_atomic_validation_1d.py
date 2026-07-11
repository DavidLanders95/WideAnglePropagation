r"""Independent numerical validation of projected silicon atom templates.

This module deliberately does not use :class:`abtem.Potential`, abTEM's
projection-integral machinery, or image interpolation.  Instead, it evaluates
``KirklandParametrization().potential("Si")`` directly and integrates that
radial three-dimensional potential over every output pixel and a finite
out-of-plane interval with tensor Gauss--Legendre quadrature.

The result is useful for detecting discretization, normalization, centering,
and interpolation mistakes in a production Lobato atomic-template renderer.
Kirkland is a different analytic independent-atom-model parameterization, so
the comparison is not a pure implementation equivalence test: disagreement
can be physical parameterization mismatch, numerical mismatch, or both.  It is
also *not* experimental evidence for either parameterization, and all result
objects therefore carry a fail-closed false trust claim.

Arrays use ``(s, u)`` order.  Grid coordinates denote pixel centres and output
values are voxel averages,

.. math::

   \bar V_{ij} = (\Delta s\,\Delta u\,w)^{-1}
      \int_{s_i-\Delta s/2}^{s_i+\Delta s/2}
      \int_{u_j-\Delta u/2}^{u_j+\Delta u/2}
      \int_{-w/2}^{w/2} V(\sqrt{s^2+u^2+z^2})\,dz\,du\,ds.

Even quadrature orders are required so that no tensor node samples the
Coulomb singularity at the origin.  The singularity is integrable, but direct
point evaluation at exactly zero is not finite.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


__all__ = [
    "AtomicTemplateComparison1D",
    "AtomicTemplateQuadratureOptions1D",
    "FiniteSiPotentialGrid1D",
    "IndependentSiAtomicTemplate1D",
    "accumulate_si_atomic_potential_1d",
    "atomic_template_cache_info_1d",
    "clear_atomic_template_cache_1d",
    "compare_si_atomic_template_1d",
    "render_si_atomic_template_1d",
]


_API_VERSION = "independent-si-atomic-template-validation-1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TRUST_REASON = (
    "fail-closed diagnostic only: direct Kirkland quadrature uses a different "
    "analytic IAM parameterization from the production Lobato template, so it "
    "is neither an implementation-equivalence certificate nor experimental "
    "physical validation"
)
_LIMITATIONS = (
    "neutral, isolated silicon atoms only",
    "finite centred out-of-plane integration omits atoms outside that width",
    "square compact patches truncate the nonzero atomic-potential tail",
    "the central Coulomb singularity can require an explicit order-convergence study",
    "Kirkland-versus-Lobato differences conflate parameterization and numerics",
    "no thermal motion, bonding, charge transfer, or detector-model validation",
)


def _finite_positive(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _strict_even_order(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be an integer") from error
    if result != value:
        raise TypeError(f"{name} must be an integer")
    if result < 2 or result % 2:
        raise ValueError(
            f"{name} must be an even integer of at least 2; even orders avoid "
            "sampling the radial singularity at exactly zero"
        )
    return result


def _strict_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be an integer") from error
    if result != value:
        raise TypeError(f"{name} must be an integer")
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _canonical_float(value: float) -> float:
    """Return a stable float key, normalizing negative zero only."""
    result = float(value)
    return 0.0 if result == 0.0 else result


def _ceil_support_pixels(cutoff_A: float, sampling_A: float) -> int:
    """Ceil a physical support without adding a pixel at a roundoff-only tie."""
    ratio = cutoff_A / sampling_A
    nearest = round(ratio)
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(ratio))
    if abs(ratio - nearest) <= tolerance:
        return int(nearest)
    return int(math.ceil(ratio))


def _readonly_float_array(
    value: Any,
    *,
    name: str,
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if (
        np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if shape is not None:
        if len(shape) != ndim:
            raise RuntimeError("internal shape specification is invalid")
        for actual, expected in zip(array.shape, shape, strict=True):
            if expected is not None and actual != expected:
                raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _validate_digest(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash_chunks(chunks: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for label, payload in chunks:
        encoded_label = label.encode("utf-8")
        digest.update(len(encoded_label).to_bytes(8, "big"))
        digest.update(encoded_label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _array_chunks(name: str, value: np.ndarray) -> tuple[tuple[str, bytes], ...]:
    array = np.ascontiguousarray(value)
    header = _json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C"}
    )
    return (
        (f"{name}:header", header),
        (f"{name}:data", array.tobytes(order="C")),
    )


def _validated_provenance(
    value: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("provenance must be a sequence of string pairs")
    result: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("each provenance entry must be a two-string tuple")
        key, entry = item
        if not isinstance(key, str) or not key:
            raise ValueError("provenance keys must be non-empty strings")
        if not isinstance(entry, str) or not entry:
            raise ValueError("provenance values must be non-empty strings")
        result.append((key, entry))
    if result != sorted(result):
        raise ValueError("provenance entries must be sorted by key")
    if len({key for key, _ in result}) != len(result):
        raise ValueError("provenance keys must be unique")
    return tuple(result)


def _canonical_provenance_input(
    value: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    if isinstance(value, Mapping):
        entries = tuple(sorted(value.items()))
    else:
        entries = tuple(value)
        if entries != tuple(sorted(entries)):
            entries = tuple(sorted(entries))
    return _validated_provenance(entries)


def _dependency_version() -> str:
    try:
        return importlib.metadata.version("abtem")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _base_provenance() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                "abtem_version": _dependency_version(),
                "api_version": _API_VERSION,
                "atomic_function": (
                    "abtem.parametrizations.KirklandParametrization()."
                    "potential('Si')"
                ),
                "element": "Si",
                "integration": "direct tensor Gauss-Legendre voxel average",
                "output_axis_order": "(s,u)",
                "output_quantity": "finite-width voxel-averaged electrostatic potential",
                "prohibited_builders": (
                    "abtem.Potential, QuadratureProjectionIntegrals, and image "
                    "shift/interpolation are not called"
                ),
                "support": "square local patch with finite half-width",
                "trust_policy": _TRUST_REASON,
            }.items()
        )
    )


@dataclass(frozen=True)
class AtomicTemplateQuadratureOptions1D:
    """Numerical choices for direct finite-width silicon integration.

    ``cutoff_A`` is a compact-patch half-width, not evidence that the omitted
    tail is negligible.  A separate cutoff-convergence study is required for
    each sampling and forward geometry.
    """

    projection_width_A: float
    cutoff_A: float = 8.0
    pixel_quadrature_order: int = 2
    projection_quadrature_order: int = 24
    maximum_quadrature_evaluations: int = 50_000_000

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "projection_width_A",
            _finite_positive(self.projection_width_A, "projection_width_A"),
        )
        object.__setattr__(self, "cutoff_A", _finite_positive(self.cutoff_A, "cutoff_A"))
        object.__setattr__(
            self,
            "pixel_quadrature_order",
            _strict_even_order(self.pixel_quadrature_order, "pixel_quadrature_order"),
        )
        object.__setattr__(
            self,
            "projection_quadrature_order",
            _strict_even_order(
                self.projection_quadrature_order,
                "projection_quadrature_order",
            ),
        )
        object.__setattr__(
            self,
            "maximum_quadrature_evaluations",
            _strict_positive_integer(
                self.maximum_quadrature_evaluations,
                "maximum_quadrature_evaluations",
            ),
        )

    @property
    def options_sha256(self) -> str:
        return _hash_chunks(
            (
                ("api_version", _API_VERSION.encode("utf-8")),
                ("options", _json_bytes(asdict(self))),
            )
        )


def _template_digest(
    *,
    values: np.ndarray,
    sampling_s_A: float,
    sampling_u_A: float,
    half_shape: tuple[int, int],
    fractional_offset_A: tuple[float, float],
    options: AtomicTemplateQuadratureOptions1D,
    provenance: tuple[tuple[str, str], ...],
) -> str:
    chunks: list[tuple[str, bytes]] = [
        ("api_version", _API_VERSION.encode("utf-8")),
        ("kind", b"independent_si_atomic_template"),
        (
            "geometry",
            _json_bytes(
                {
                    "sampling_s_A": sampling_s_A,
                    "sampling_u_A": sampling_u_A,
                    "half_shape": list(half_shape),
                    "fractional_offset_A": list(fractional_offset_A),
                }
            ),
        ),
        ("options", _json_bytes(asdict(options))),
        ("provenance", _json_bytes(provenance)),
    ]
    chunks.extend(_array_chunks("values", values))
    return _hash_chunks(chunks)


@dataclass(frozen=True)
class IndependentSiAtomicTemplate1D:
    """Immutable result of direct silicon atomic-potential integration."""

    values: np.ndarray
    sampling_s_A: float
    sampling_u_A: float
    half_shape: tuple[int, int]
    fractional_offset_A: tuple[float, float]
    options: AtomicTemplateQuadratureOptions1D
    template_sha256: str
    provenance: tuple[tuple[str, str], ...]
    trust_claim: bool = False
    trust_reason: str = _TRUST_REASON
    limitations: tuple[str, ...] = field(default=_LIMITATIONS)

    def __post_init__(self) -> None:
        ds = _finite_positive(self.sampling_s_A, "sampling_s_A")
        du = _finite_positive(self.sampling_u_A, "sampling_u_A")
        object.__setattr__(self, "sampling_s_A", ds)
        object.__setattr__(self, "sampling_u_A", du)
        if (
            not isinstance(self.half_shape, tuple)
            or len(self.half_shape) != 2
            or any(isinstance(value, (bool, np.bool_)) for value in self.half_shape)
            or any(not isinstance(value, (int, np.integer)) for value in self.half_shape)
            or any(int(value) < 0 for value in self.half_shape)
        ):
            raise ValueError("half_shape must contain two non-negative integers")
        half_shape = tuple(int(value) for value in self.half_shape)
        object.__setattr__(self, "half_shape", half_shape)
        values = _readonly_float_array(
            self.values,
            name="values",
            ndim=2,
            shape=(2 * half_shape[0] + 1, 2 * half_shape[1] + 1),
        )
        if np.any(values < 0.0):
            raise ValueError("values must be non-negative for neutral silicon")
        object.__setattr__(self, "values", values)
        offsets = _validate_fractional_offset(
            self.fractional_offset_A,
            sampling_s_A=ds,
            sampling_u_A=du,
        )
        object.__setattr__(self, "fractional_offset_A", offsets)
        if not isinstance(self.options, AtomicTemplateQuadratureOptions1D):
            raise TypeError("options must be AtomicTemplateQuadratureOptions1D")
        provenance = _validated_provenance(self.provenance)
        object.__setattr__(self, "provenance", provenance)
        if self.trust_claim is not False:
            raise ValueError("independent template validation must fail closed")
        if self.trust_reason != _TRUST_REASON:
            raise ValueError("trust_reason must use the module fail-closed policy")
        if tuple(self.limitations) != _LIMITATIONS:
            raise ValueError("limitations must retain the complete module limitations")
        object.__setattr__(self, "limitations", _LIMITATIONS)
        digest = _validate_digest(self.template_sha256, "template_sha256")
        expected = _template_digest(
            values=values,
            sampling_s_A=ds,
            sampling_u_A=du,
            half_shape=half_shape,
            fractional_offset_A=offsets,
            options=self.options,
            provenance=provenance,
        )
        if digest != expected:
            raise ValueError("template_sha256 does not match the template content")


def _validate_fractional_offset(
    value: Sequence[float],
    *,
    sampling_s_A: float,
    sampling_u_A: float,
) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("fractional_offset_A must contain two real values")
    offsets: list[float] = []
    for entry, sampling, name in zip(
        value,
        (sampling_s_A, sampling_u_A),
        ("s", "u"),
        strict=True,
    ):
        try:
            offset = _canonical_float(float(entry))
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("fractional_offset_A entries must be real scalars") from error
        if not np.isfinite(offset):
            raise ValueError("fractional_offset_A entries must be finite")
        half = 0.5 * sampling
        if offset < -half or not offset < half:
            raise ValueError(
                f"fractional {name} offset must be in [-sampling/2, sampling/2)"
            )
        offsets.append(offset)
    return offsets[0], offsets[1]


def _validate_half_shape(value: Sequence[int]) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("half_shape must contain two integers")
    result: list[int] = []
    for entry in value:
        if isinstance(entry, (bool, np.bool_)):
            raise TypeError("half_shape entries must be integers")
        try:
            integer = int(entry)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("half_shape entries must be integers") from error
        if integer != entry:
            raise TypeError("half_shape entries must be integers")
        if integer < 0:
            raise ValueError("half_shape entries must be non-negative")
        result.append(integer)
    return result[0], result[1]


@lru_cache(maxsize=512)
def _render_template_bytes_cached(
    options: AtomicTemplateQuadratureOptions1D,
    sampling_s_A: float,
    sampling_u_A: float,
    half_s: int,
    half_u: int,
    offset_s_A: float,
    offset_u_A: float,
) -> bytes:
    """Compute a template cache entry without returning a mutable array."""
    # Importing the parametrization lazily keeps this validation helper optional.
    from abtem.parametrizations import KirklandParametrization

    pixel_nodes, pixel_weights = np.polynomial.legendre.leggauss(
        options.pixel_quadrature_order
    )
    z_nodes, z_weights = np.polynomial.legendre.leggauss(
        options.projection_quadrature_order
    )
    s_centres = np.arange(-half_s, half_s + 1, dtype=np.float64) * sampling_s_A
    u_centres = np.arange(-half_u, half_u + 1, dtype=np.float64) * sampling_u_A
    s = (
        s_centres[:, None]
        + 0.5 * sampling_s_A * pixel_nodes[None, :]
        - offset_s_A
    )
    u = (
        u_centres[:, None]
        + 0.5 * sampling_u_A * pixel_nodes[None, :]
        - offset_u_A
    )
    z = 0.5 * options.projection_width_A * z_nodes
    radial_potential = KirklandParametrization().potential("Si")

    # The mapped-Jacobian factors cancel when the integral is divided by the
    # voxel volume, leaving the product-weight sum divided by 2**3.
    weights = (
        pixel_weights[:, None, None]
        * pixel_weights[None, :, None]
        * z_weights[None, None, :]
        / 8.0
    )
    values = np.empty((s_centres.size, u_centres.size), dtype=np.float64)
    for s_index, s_samples in enumerate(s):
        radius = np.sqrt(
            s_samples[:, None, None, None] ** 2
            + u[None, :, :, None] ** 2
            + z[None, None, None, :] ** 2
        )
        evaluated = np.asarray(radial_potential(radius), dtype=np.float64)
        values[s_index] = np.sum(evaluated * weights[:, None, :, :], axis=(0, 2, 3))
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError(
            "direct Kirkland quadrature produced non-finite or negative values"
        )
    return values.tobytes(order="C")


def render_si_atomic_template_1d(
    *,
    sampling_s_A: float,
    sampling_u_A: float,
    options: AtomicTemplateQuadratureOptions1D,
    half_shape: Sequence[int] | None = None,
    fractional_offset_A: Sequence[float] = (0.0, 0.0),
) -> IndependentSiAtomicTemplate1D:
    """Directly render a compact, exactly subpixel-centred Si template.

    Fractional offsets are evaluated in the radial potential itself; no shifted
    image or interpolated template is used.  Exact floating-point offsets form
    part of the bounded LRU cache key.  ``half_shape=None`` chooses
    ``ceil(cutoff_A / sampling)`` independently on both axes.
    """
    if not isinstance(options, AtomicTemplateQuadratureOptions1D):
        raise TypeError("options must be AtomicTemplateQuadratureOptions1D")
    ds = _finite_positive(sampling_s_A, "sampling_s_A")
    du = _finite_positive(sampling_u_A, "sampling_u_A")
    if half_shape is None:
        half_s = _ceil_support_pixels(options.cutoff_A, ds)
        half_u = _ceil_support_pixels(options.cutoff_A, du)
    else:
        half_s, half_u = _validate_half_shape(half_shape)
        if half_s * ds + np.finfo(float).eps < options.cutoff_A:
            raise ValueError("half_shape does not reach cutoff_A on the s axis")
        if half_u * du + np.finfo(float).eps < options.cutoff_A:
            raise ValueError("half_shape does not reach cutoff_A on the u axis")
    evaluations = (
        (2 * half_s + 1)
        * (2 * half_u + 1)
        * options.pixel_quadrature_order**2
        * options.projection_quadrature_order
    )
    if evaluations > options.maximum_quadrature_evaluations:
        raise ValueError(
            "requested template exceeds maximum_quadrature_evaluations "
            f"({evaluations} > {options.maximum_quadrature_evaluations}); increase "
            "the cap only after reviewing memory and runtime"
        )
    offset_s, offset_u = _validate_fractional_offset(
        fractional_offset_A,
        sampling_s_A=ds,
        sampling_u_A=du,
    )
    payload = _render_template_bytes_cached(
        options,
        ds,
        du,
        half_s,
        half_u,
        offset_s,
        offset_u,
    )
    values = np.frombuffer(payload, dtype=np.float64).reshape(
        2 * half_s + 1,
        2 * half_u + 1,
    )
    provenance = _base_provenance()
    digest = _template_digest(
        values=values,
        sampling_s_A=ds,
        sampling_u_A=du,
        half_shape=(half_s, half_u),
        fractional_offset_A=(offset_s, offset_u),
        options=options,
        provenance=provenance,
    )
    return IndependentSiAtomicTemplate1D(
        values=values,
        sampling_s_A=ds,
        sampling_u_A=du,
        half_shape=(half_s, half_u),
        fractional_offset_A=(offset_s, offset_u),
        options=options,
        template_sha256=digest,
        provenance=provenance,
    )


def atomic_template_cache_info_1d() -> Any:
    """Return the standard immutable ``functools`` cache information tuple."""
    return _render_template_bytes_cached.cache_info()


def clear_atomic_template_cache_1d() -> None:
    """Clear cached quadrature templates (primarily useful in controlled tests)."""
    _render_template_bytes_cached.cache_clear()


def _validate_uniform_axis(value: Any, name: str) -> tuple[np.ndarray, float]:
    axis = _readonly_float_array(value, name=name, ndim=1)
    if axis.size < 2:
        raise ValueError(f"{name} must contain at least two pixel centres")
    differences = np.diff(axis)
    if np.any(differences <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    spacing = float((axis[-1] - axis[0]) / (axis.size - 1))
    tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(axis))), abs(spacing))
    )
    if not np.allclose(differences, spacing, rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} must be uniformly sampled")
    return axis, spacing


def _grid_digest(
    *,
    values: np.ndarray,
    s_coordinates_A: np.ndarray,
    u_coordinates_A: np.ndarray,
    site_coordinates_A: np.ndarray,
    fractional_offsets_A: np.ndarray,
    template_sha256: tuple[str, ...],
    options: AtomicTemplateQuadratureOptions1D,
    provenance: tuple[tuple[str, str], ...],
) -> str:
    chunks: list[tuple[str, bytes]] = [
        ("api_version", _API_VERSION.encode("utf-8")),
        ("kind", b"finite_si_potential_grid"),
        ("options", _json_bytes(asdict(options))),
        ("template_sha256", _json_bytes(template_sha256)),
        ("provenance", _json_bytes(provenance)),
    ]
    for name, array in (
        ("values", values),
        ("s_coordinates_A", s_coordinates_A),
        ("u_coordinates_A", u_coordinates_A),
        ("site_coordinates_A", site_coordinates_A),
        ("fractional_offsets_A", fractional_offsets_A),
    ):
        chunks.extend(_array_chunks(name, array))
    return _hash_chunks(chunks)


@dataclass(frozen=True)
class FiniteSiPotentialGrid1D:
    """Immutable finite-grid accumulation from explicit Si site coordinates."""

    values: np.ndarray
    s_coordinates_A: np.ndarray
    u_coordinates_A: np.ndarray
    site_coordinates_A: np.ndarray
    fractional_offsets_A: np.ndarray
    template_sha256: tuple[str, ...]
    options: AtomicTemplateQuadratureOptions1D
    grid_sha256: str
    provenance: tuple[tuple[str, str], ...]
    trust_claim: bool = False
    trust_reason: str = _TRUST_REASON
    limitations: tuple[str, ...] = field(default=_LIMITATIONS)

    def __post_init__(self) -> None:
        s_axis, _ = _validate_uniform_axis(self.s_coordinates_A, "s_coordinates_A")
        u_axis, _ = _validate_uniform_axis(self.u_coordinates_A, "u_coordinates_A")
        values = _readonly_float_array(
            self.values,
            name="values",
            ndim=2,
            shape=(s_axis.size, u_axis.size),
        )
        if np.any(values < 0.0):
            raise ValueError("values must be non-negative for neutral silicon")
        sites = _readonly_float_array(
            self.site_coordinates_A,
            name="site_coordinates_A",
            ndim=2,
            shape=(None, 2),
        )
        offsets = _readonly_float_array(
            self.fractional_offsets_A,
            name="fractional_offsets_A",
            ndim=2,
            shape=(None, 2),
        )
        if offsets.shape[0] != len(self.template_sha256):
            raise ValueError(
                "fractional_offsets_A and template_sha256 must have equal lengths"
            )
        template_digests = tuple(
            _validate_digest(value, "template_sha256 entry")
            for value in self.template_sha256
        )
        if len(set(template_digests)) != len(template_digests):
            raise ValueError("template_sha256 entries must be unique")
        if not isinstance(self.options, AtomicTemplateQuadratureOptions1D):
            raise TypeError("options must be AtomicTemplateQuadratureOptions1D")
        provenance = _validated_provenance(self.provenance)
        if self.trust_claim is not False:
            raise ValueError("finite-grid validation must fail closed")
        if self.trust_reason != _TRUST_REASON:
            raise ValueError("trust_reason must use the module fail-closed policy")
        if tuple(self.limitations) != _LIMITATIONS:
            raise ValueError("limitations must retain the complete module limitations")
        object.__setattr__(self, "limitations", _LIMITATIONS)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "s_coordinates_A", s_axis)
        object.__setattr__(self, "u_coordinates_A", u_axis)
        object.__setattr__(self, "site_coordinates_A", sites)
        object.__setattr__(self, "fractional_offsets_A", offsets)
        object.__setattr__(self, "template_sha256", template_digests)
        object.__setattr__(self, "provenance", provenance)
        digest = _validate_digest(self.grid_sha256, "grid_sha256")
        expected = _grid_digest(
            values=values,
            s_coordinates_A=s_axis,
            u_coordinates_A=u_axis,
            site_coordinates_A=sites,
            fractional_offsets_A=offsets,
            template_sha256=template_digests,
            options=self.options,
            provenance=provenance,
        )
        if digest != expected:
            raise ValueError("grid_sha256 does not match the finite-grid content")


def accumulate_si_atomic_potential_1d(
    site_coordinates_A: Any,
    *,
    s_coordinates_A: Any,
    u_coordinates_A: Any,
    options: AtomicTemplateQuadratureOptions1D,
) -> FiniteSiPotentialGrid1D:
    """Accumulate exact-subpixel local templates from explicit ``(s,u)`` sites.

    Sites may lie outside the finite grid; only the overlap of their compact
    patch with the grid is accumulated.  No implicit periodic images or hidden
    atoms are introduced.  Repeated fractional offsets reuse direct-quadrature
    cache entries, while every returned grid records the unique templates used.
    """
    if not isinstance(options, AtomicTemplateQuadratureOptions1D):
        raise TypeError("options must be AtomicTemplateQuadratureOptions1D")
    s_axis, ds = _validate_uniform_axis(s_coordinates_A, "s_coordinates_A")
    u_axis, du = _validate_uniform_axis(u_coordinates_A, "u_coordinates_A")
    sites = _readonly_float_array(
        site_coordinates_A,
        name="site_coordinates_A",
        ndim=2,
        shape=(None, 2),
    )
    half_s = _ceil_support_pixels(options.cutoff_A, ds)
    half_u = _ceil_support_pixels(options.cutoff_A, du)
    values = np.zeros((s_axis.size, u_axis.size), dtype=np.float64)
    templates: dict[tuple[float, float], IndependentSiAtomicTemplate1D] = {}

    for site_s_A, site_u_A in sites:
        pixel_s = (float(site_s_A) - float(s_axis[0])) / ds
        pixel_u = (float(site_u_A) - float(u_axis[0])) / du
        centre_s = int(math.floor(pixel_s + 0.5))
        centre_u = int(math.floor(pixel_u + 0.5))
        offset_s = _canonical_float(float(site_s_A) - (float(s_axis[0]) + centre_s * ds))
        offset_u = _canonical_float(float(site_u_A) - (float(u_axis[0]) + centre_u * du))
        # Roundoff at a nominal +half-pixel tie belongs to the upper centre.
        if offset_s >= 0.5 * ds:
            centre_s += 1
            offset_s = _canonical_float(offset_s - ds)
        elif offset_s < -0.5 * ds:
            centre_s -= 1
            offset_s = _canonical_float(offset_s + ds)
        if offset_u >= 0.5 * du:
            centre_u += 1
            offset_u = _canonical_float(offset_u - du)
        elif offset_u < -0.5 * du:
            centre_u -= 1
            offset_u = _canonical_float(offset_u + du)
        key = (offset_s, offset_u)
        template = templates.get(key)
        if template is None:
            template = render_si_atomic_template_1d(
                sampling_s_A=ds,
                sampling_u_A=du,
                options=options,
                half_shape=(half_s, half_u),
                fractional_offset_A=key,
            )
            templates[key] = template

        grid_s_start = max(0, centre_s - half_s)
        grid_s_stop = min(s_axis.size, centre_s + half_s + 1)
        grid_u_start = max(0, centre_u - half_u)
        grid_u_stop = min(u_axis.size, centre_u + half_u + 1)
        if grid_s_start >= grid_s_stop or grid_u_start >= grid_u_stop:
            continue
        template_s_start = grid_s_start - (centre_s - half_s)
        template_u_start = grid_u_start - (centre_u - half_u)
        template_s_stop = template_s_start + (grid_s_stop - grid_s_start)
        template_u_stop = template_u_start + (grid_u_stop - grid_u_start)
        values[grid_s_start:grid_s_stop, grid_u_start:grid_u_stop] += template.values[
            template_s_start:template_s_stop,
            template_u_start:template_u_stop,
        ]

    ordered_templates = sorted(templates.items())
    if ordered_templates:
        offsets = np.asarray([key for key, _ in ordered_templates], dtype=np.float64)
    else:
        offsets = np.empty((0, 2), dtype=np.float64)
    template_digests = tuple(
        template.template_sha256 for _, template in ordered_templates
    )
    provenance = _base_provenance()
    digest = _grid_digest(
        values=values,
        s_coordinates_A=s_axis,
        u_coordinates_A=u_axis,
        site_coordinates_A=sites,
        fractional_offsets_A=offsets,
        template_sha256=template_digests,
        options=options,
        provenance=provenance,
    )
    return FiniteSiPotentialGrid1D(
        values=values,
        s_coordinates_A=s_axis,
        u_coordinates_A=u_axis,
        site_coordinates_A=sites,
        fractional_offsets_A=offsets,
        template_sha256=template_digests,
        options=options,
        grid_sha256=digest,
        provenance=provenance,
    )


def _comparison_digest(
    *,
    candidate_template_sha256: str,
    reference_template_sha256: str,
    metrics: Mapping[str, float],
    reference_provenance: tuple[tuple[str, str], ...],
) -> str:
    return _hash_chunks(
        (
            ("api_version", _API_VERSION.encode("utf-8")),
            ("kind", b"si_atomic_template_comparison"),
            ("candidate_template_sha256", candidate_template_sha256.encode("ascii")),
            ("reference_template_sha256", reference_template_sha256.encode("ascii")),
            ("metrics", _json_bytes(dict(metrics))),
            ("reference_provenance", _json_bytes(reference_provenance)),
        )
    )


@dataclass(frozen=True)
class AtomicTemplateComparison1D:
    """Immutable numerical comparison, never a physical-validity certificate."""

    raw_relative_l2: float
    scale_adjusted_shape_relative_l2: float
    optimal_candidate_scale: float
    peak_ratio: float
    integral_ratio: float
    candidate_template_sha256: str
    reference_template_sha256: str
    comparison_sha256: str
    reference_provenance: tuple[tuple[str, str], ...]
    trust_claim: bool = False
    trust_reason: str = _TRUST_REASON
    limitations: tuple[str, ...] = field(default=_LIMITATIONS)

    def __post_init__(self) -> None:
        metrics = {
            "raw_relative_l2": self.raw_relative_l2,
            "scale_adjusted_shape_relative_l2": self.scale_adjusted_shape_relative_l2,
            "optimal_candidate_scale": self.optimal_candidate_scale,
            "peak_ratio": self.peak_ratio,
            "integral_ratio": self.integral_ratio,
        }
        for name, value in metrics.items():
            try:
                converted = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise TypeError(f"{name} must be a real scalar") from error
            if not np.isfinite(converted):
                raise ValueError(f"{name} must be finite")
            if converted < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, converted)
            metrics[name] = converted
        candidate_digest = _validate_digest(
            self.candidate_template_sha256,
            "candidate_template_sha256",
        )
        reference_digest = _validate_digest(
            self.reference_template_sha256,
            "reference_template_sha256",
        )
        provenance = _validated_provenance(self.reference_provenance)
        object.__setattr__(self, "reference_provenance", provenance)
        if self.trust_claim is not False:
            raise ValueError("template comparison must fail closed")
        if self.trust_reason != _TRUST_REASON:
            raise ValueError("trust_reason must use the module fail-closed policy")
        if tuple(self.limitations) != _LIMITATIONS:
            raise ValueError("limitations must retain the complete module limitations")
        object.__setattr__(self, "limitations", _LIMITATIONS)
        digest = _validate_digest(self.comparison_sha256, "comparison_sha256")
        expected = _comparison_digest(
            candidate_template_sha256=candidate_digest,
            reference_template_sha256=reference_digest,
            metrics=metrics,
            reference_provenance=provenance,
        )
        if digest != expected:
            raise ValueError("comparison_sha256 does not match comparison content")


def compare_si_atomic_template_1d(
    candidate: IndependentSiAtomicTemplate1D,
    reference_template: Any,
    *,
    reference_provenance: Mapping[str, str] | Sequence[tuple[str, str]],
) -> AtomicTemplateComparison1D:
    """Compare direct quadrature to a supplied same-grid reference template.

    The scale-adjusted metric minimizes ``||a * candidate - reference||`` over
    a single non-negative scalar ``a``.  Peak and integral ratios retain the
    raw normalization mismatch.  Reference provenance is mandatory so an
    anonymous array cannot accidentally acquire even a numerical trust claim.
    """
    if not isinstance(candidate, IndependentSiAtomicTemplate1D):
        raise TypeError("candidate must be IndependentSiAtomicTemplate1D")
    reference = _readonly_float_array(
        reference_template,
        name="reference_template",
        ndim=2,
        shape=candidate.values.shape,
    )
    provenance = _canonical_provenance_input(reference_provenance)
    if not provenance:
        raise ValueError("reference_provenance must not be empty")
    candidate_values = candidate.values
    reference_norm = float(np.linalg.norm(reference))
    candidate_squared_norm = float(np.vdot(candidate_values, candidate_values).real)
    if reference_norm <= np.finfo(np.float64).tiny:
        raise ValueError("reference_template must have nonzero L2 norm")
    if np.any(reference < 0.0):
        raise ValueError("reference_template must be non-negative for neutral silicon")
    if candidate_squared_norm <= np.finfo(np.float64).tiny:
        raise ValueError("candidate template must have nonzero L2 norm")
    reference_peak = float(np.max(reference))
    reference_integral = float(np.sum(reference, dtype=np.float64))
    candidate_peak = float(np.max(candidate_values))
    candidate_integral = float(np.sum(candidate_values, dtype=np.float64))
    if reference_peak <= 0.0:
        raise ValueError("reference_template must have a positive peak")
    if reference_integral <= 0.0:
        raise ValueError("reference_template must have a positive integral")
    dot = float(np.vdot(candidate_values, reference).real)
    optimal_scale = max(0.0, dot / candidate_squared_norm)
    raw_relative_l2 = float(
        np.linalg.norm(candidate_values - reference) / reference_norm
    )
    shape_relative_l2 = float(
        np.linalg.norm(optimal_scale * candidate_values - reference) / reference_norm
    )
    metrics = {
        "raw_relative_l2": raw_relative_l2,
        "scale_adjusted_shape_relative_l2": shape_relative_l2,
        "optimal_candidate_scale": optimal_scale,
        "peak_ratio": candidate_peak / reference_peak,
        "integral_ratio": candidate_integral / reference_integral,
    }
    reference_digest = _hash_chunks(_array_chunks("reference_template", reference))
    digest = _comparison_digest(
        candidate_template_sha256=candidate.template_sha256,
        reference_template_sha256=reference_digest,
        metrics=metrics,
        reference_provenance=provenance,
    )
    return AtomicTemplateComparison1D(
        **metrics,
        candidate_template_sha256=candidate.template_sha256,
        reference_template_sha256=reference_digest,
        comparison_sha256=digest,
        reference_provenance=provenance,
    )
