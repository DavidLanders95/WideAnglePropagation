"""Geometry-bound material-support contracts for lattice ptychography.

This module separates sites that may be reported as specimen structure from
sites that are present only to absorb uncertain illuminated exterior material.
It is intentionally NumPy-only and does not depend on the renderer or JAX.

The classification is conservative with respect to the supplied masks: the
complete rectangular footprint of every already displacement-padded atomic
patch is tested, including zero-valued edge samples that a translated cubic
interpolant could make nonzero.  A footprint outside ``forward_pixel_mask`` is
only below the *declared geometric interaction budget*.  This is not a
detector-space error bound or an observability certificate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from enum import IntEnum
import hashlib
import json
import math
import operator
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np


__all__ = [
    "LatticeSiteParameterCounts1D",
    "LatticeSiteRole1D",
    "LatticeSiteSupportContract1D",
    "classify_lattice_site_support_1d",
    "lattice_site_support_contract_id_1d",
    "validate_lattice_site_support_contract_1d",
]


_SUPPORT_SCHEMA_VERSION_1D = 1
_CLASSIFICATION_CONTRACT_1D = (
    "site_center_target_complete_padded_patch_forward_roles:v1"
)
_EXTERIOR_POLICIES_1D = (
    "parameterize_uncertain",
    "leave_unresolved",
)


class LatticeSiteRole1D(IntEnum):
    """Role of one known lattice site in a reconstruction support contract."""

    TARGET = 1
    NUISANCE = 2
    FIXED_KNOWN = 3
    BELOW_INTERACTION_BUDGET = 4
    UNRESOLVED = 5


@dataclass(frozen=True)
class LatticeSiteParameterCounts1D:
    """Exact specimen parameter counts implied by one site classification."""

    target_vacancy_parameters: int
    nuisance_vacancy_parameters: int
    displacement_control_parameters: int
    removed_displacement_dof: int
    residual_displacement_control_dof: int
    registration_parameters: int
    total_specimen_parameters: int


@dataclass(frozen=True, eq=False)
class LatticeSiteSupportContract1D:
    """Immutable, digest-bound classification of every known lattice site.

    ``TARGET`` and ``NUISANCE`` sites, in their original all-site order, are
    listed by ``modeled_site_indices`` and are intended to form the numerical
    :class:`~wide_angle_propagation.ptychography_1d.LatticeSiteModel1D`.
    Nuisance values are forward-model degrees of freedom, not reportable
    structural results.

    A strict contract resolves every forward-relevant site and stays within
    both declared resource budgets. ``FIXED_KNOWN`` is an explicit scientific
    assertion and therefore requires a provenance identifier. The identifier
    records the assertion; it cannot prove that the assertion is true.
    """

    schema_version: int
    classification_contract: str
    all_site_coordinates: np.ndarray
    site_center_indices: np.ndarray
    site_patch_starts: np.ndarray
    site_patch_shapes: np.ndarray
    target_pixel_mask: np.ndarray
    forward_pixel_mask: np.ndarray
    target_center_mask: np.ndarray
    forward_relevant_mask: np.ndarray
    site_role_codes: np.ndarray
    modeled_site_indices: np.ndarray
    target_influence_mask: np.ndarray
    nuisance_influence_mask: np.ndarray
    exterior_policy: str
    excluded_probe_power: float
    atomic_template_cutoff_A: float
    maximum_displacement_A: float
    fixed_material_provenance_id: str | None
    displacement_control_shape: tuple[int, ...]
    removed_displacement_dof: int
    registration_parameter_count: int
    maximum_nuisance_sites: int
    maximum_specimen_parameters: int
    parameter_counts: LatticeSiteParameterCounts1D
    contract_id: str

    @property
    def target_site_indices(self) -> np.ndarray:
        """Read-only all-site indices eligible for structural reporting."""
        return _readonly_array(
            np.flatnonzero(
                self.site_role_codes == int(LatticeSiteRole1D.TARGET)
            ),
            dtype=np.int64,
        )

    @property
    def nuisance_site_indices(self) -> np.ndarray:
        """Read-only all-site indices modeled only as nuisance material."""
        return _readonly_array(
            np.flatnonzero(
                self.site_role_codes == int(LatticeSiteRole1D.NUISANCE)
            ),
            dtype=np.int64,
        )

    @property
    def strict_requirements_satisfied(self) -> bool:
        """Whether the derived strict provenance and budget gates pass."""
        roles = np.asarray(self.site_role_codes)
        unresolved = np.count_nonzero(
            roles == int(LatticeSiteRole1D.UNRESOLVED)
        )
        fixed = np.count_nonzero(
            roles == int(LatticeSiteRole1D.FIXED_KNOWN)
        )
        provenance_present = (
            fixed == 0
            or (
                isinstance(self.fixed_material_provenance_id, str)
                and bool(self.fixed_material_provenance_id.strip())
            )
        )
        return bool(
            self.parameter_counts.target_vacancy_parameters > 0
            and unresolved == 0
            and provenance_present
            and self.parameter_counts.nuisance_vacancy_parameters
            <= self.maximum_nuisance_sites
            and self.parameter_counts.total_specimen_parameters
            <= self.maximum_specimen_parameters
        )

    @property
    def parameter_count_metadata(self) -> Mapping[str, int]:
        """Read-only exact counts for logging before compilation."""
        roles = np.asarray(self.site_role_codes)
        role_counts = {
            "target_sites": int(
                np.count_nonzero(roles == int(LatticeSiteRole1D.TARGET))
            ),
            "nuisance_sites": int(
                np.count_nonzero(roles == int(LatticeSiteRole1D.NUISANCE))
            ),
            "fixed_known_sites": int(
                np.count_nonzero(roles == int(LatticeSiteRole1D.FIXED_KNOWN))
            ),
            "below_interaction_budget_sites": int(
                np.count_nonzero(
                    roles
                    == int(LatticeSiteRole1D.BELOW_INTERACTION_BUDGET)
                )
            ),
            "unresolved_sites": int(
                np.count_nonzero(roles == int(LatticeSiteRole1D.UNRESOLVED))
            ),
            "modeled_sites": int(len(self.modeled_site_indices)),
        }
        return MappingProxyType({**role_counts, **asdict(self.parameter_counts)})


def _readonly_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _boolean_array(name: str, value: Any, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype != np.bool_:
        raise TypeError(f"{name} must be a {ndim}D Boolean array")
    return _readonly_array(array, dtype=bool)


def _integer_array(name: str, value: Any, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be a {ndim}D integer array")
    return _readonly_array(array, dtype=np.int64)


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(result)


def _finite_scalar(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    allow_zero: bool = False,
) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and not (result >= 0.0 if allow_zero else result > 0.0):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")
    return result


def _validated_control_shape(value: Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise TypeError("displacement_control_shape must be an integer sequence")
    try:
        shape = tuple(_nonnegative_integer("control dimension", item) for item in value)
    except TypeError as error:
        raise TypeError(
            "displacement_control_shape must be an integer sequence"
        ) from error
    if not shape:
        return ()
    if len(shape) != 3 or shape[-1] != 2 or any(item < 1 for item in shape):
        raise ValueError(
            "displacement_control_shape must be empty/None or have shape "
            "(n_control_s, n_control_u, 2) with positive dimensions"
        )
    return shape


def _target_centers_in_mask(
    centers: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    selected = np.zeros(centers.shape[0], dtype=bool)
    n_s, n_u = target_mask.shape
    in_grid = (
        (centers[:, 0] >= 0)
        & (centers[:, 0] < n_s)
        & (centers[:, 1] >= 0)
        & (centers[:, 1] < n_u)
    )
    selected[in_grid] = target_mask[
        centers[in_grid, 0],
        centers[in_grid, 1],
    ]
    return selected


def _forward_footprint_mask(
    starts: np.ndarray,
    shapes: np.ndarray,
    forward_mask: np.ndarray,
) -> np.ndarray:
    n_site = starts.shape[0]
    forward_overlap = np.zeros(n_site, dtype=bool)
    n_s, n_u = forward_mask.shape
    for index, (start, shape) in enumerate(zip(starts, shapes)):
        start_s, start_u = (int(item) for item in start)
        shape_s, shape_u = (int(item) for item in shape)
        stop_s = start_s + shape_s
        stop_u = start_u + shape_u
        clipped_start_s = max(start_s, 0)
        clipped_start_u = max(start_u, 0)
        clipped_stop_s = min(stop_s, n_s)
        clipped_stop_u = min(stop_u, n_u)
        if (
            clipped_start_s >= clipped_stop_s
            or clipped_start_u >= clipped_stop_u
        ):
            continue
        footprint = np.s_[
            clipped_start_s:clipped_stop_s,
            clipped_start_u:clipped_stop_u,
        ]
        forward_overlap[index] = bool(np.any(forward_mask[footprint]))
    return forward_overlap


def _influence_mask(
    starts: np.ndarray,
    shapes: np.ndarray,
    selected_sites: np.ndarray,
    potential_shape: tuple[int, int],
) -> np.ndarray:
    influence = np.zeros(potential_shape, dtype=bool)
    n_s, n_u = potential_shape
    for index in np.flatnonzero(selected_sites):
        start_s, start_u = (int(item) for item in starts[index])
        shape_s, shape_u = (int(item) for item in shapes[index])
        stop_s = start_s + shape_s
        stop_u = start_u + shape_u
        clipped_start_s = max(start_s, 0)
        clipped_start_u = max(start_u, 0)
        clipped_stop_s = min(stop_s, n_s)
        clipped_stop_u = min(stop_u, n_u)
        if (
            clipped_start_s < clipped_stop_s
            and clipped_start_u < clipped_stop_u
        ):
            influence[
                clipped_start_s:clipped_stop_s,
                clipped_start_u:clipped_stop_u,
            ] = True
    return influence


def _parameter_counts(
    roles: np.ndarray,
    control_shape: tuple[int, ...],
    removed_displacement_dof: int,
    registration_parameter_count: int,
) -> LatticeSiteParameterCounts1D:
    target = int(np.count_nonzero(roles == int(LatticeSiteRole1D.TARGET)))
    nuisance = int(
        np.count_nonzero(roles == int(LatticeSiteRole1D.NUISANCE))
    )
    control_parameters = math.prod(control_shape) if control_shape else 0
    if removed_displacement_dof > control_parameters:
        raise ValueError(
            "removed_displacement_dof exceeds displacement-control parameters"
        )
    residual = control_parameters - removed_displacement_dof
    return LatticeSiteParameterCounts1D(
        target_vacancy_parameters=target,
        nuisance_vacancy_parameters=nuisance,
        displacement_control_parameters=control_parameters,
        removed_displacement_dof=removed_displacement_dof,
        residual_displacement_control_dof=residual,
        registration_parameters=registration_parameter_count,
        total_specimen_parameters=(target + nuisance + residual + registration_parameter_count),
    )


def _sha256_chunk(digest: Any, label: str, payload: bytes) -> None:
    encoded_label = label.encode("utf-8")
    digest.update(len(encoded_label).to_bytes(8, "big"))
    digest.update(encoded_label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _support_contract_digest(contract: LatticeSiteSupportContract1D) -> str:
    """Hash every dataclass field except the digest that is being computed."""
    digest = hashlib.sha256()
    for item in fields(contract):
        name = item.name
        if name == "contract_id":
            continue
        value = getattr(contract, name)
        if isinstance(value, np.ndarray):
            header = json.dumps(
                {"dtype": value.dtype.str, "shape": list(value.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            _sha256_chunk(digest, f"{name}:array_header", header)
            _sha256_chunk(
                digest,
                f"{name}:array_data",
                np.ascontiguousarray(value).tobytes(order="C"),
            )
        elif isinstance(value, LatticeSiteParameterCounts1D):
            payload = asdict(value)
            _sha256_chunk(
                digest,
                f"{name}:dataclass",
                json.dumps(
                    payload,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
        elif value is None or isinstance(value, (str, int, float, bool, tuple)):
            _sha256_chunk(
                digest,
                f"{name}:scalar",
                json.dumps(
                    {"type": type(value).__name__, "value": value},
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
        else:  # pragma: no cover - guards future contract fields
            raise TypeError(
                f"unsupported support-contract field {name!r}: "
                f"{type(value).__name__}"
            )
    return digest.hexdigest()


def lattice_site_support_contract_id_1d(
    contract: LatticeSiteSupportContract1D,
) -> str:
    """Recompute the deterministic SHA-256 of every semantic contract field."""
    if not isinstance(contract, LatticeSiteSupportContract1D):
        raise TypeError("contract must be a LatticeSiteSupportContract1D")
    return _support_contract_digest(contract)


def classify_lattice_site_support_1d(
    all_site_coordinates: Any,
    site_center_indices: Any,
    site_patch_starts: Any,
    site_patch_shapes: Any,
    target_pixel_mask: Any,
    forward_pixel_mask: Any,
    *,
    exterior_policy: Literal[
        "parameterize_uncertain", "leave_unresolved"
    ] = "parameterize_uncertain",
    known_fixed_site_mask: Any | None = None,
    fixed_material_provenance_id: str | None = None,
    excluded_probe_power: Any,
    atomic_template_cutoff_A: Any,
    maximum_displacement_A: Any,
    displacement_control_shape: Sequence[int] | None = None,
    removed_displacement_dof: int = 0,
    registration_parameter_count: int = 0,
    maximum_nuisance_sites: int = 4096,
    maximum_specimen_parameters: int = 8192,
    strict: bool = True,
) -> LatticeSiteSupportContract1D:
    """Classify known sites from complete padded patch footprints.

    Only a site whose explicit in-grid center lies in ``target_pixel_mask`` is
    reportable. A padded patch touching the chosen region cannot silently make
    its site a target. Complete patch footprints determine forward relevance;
    non-target forward-relevant sites are fixed only when named by
    ``known_fixed_site_mask`` and otherwise become nuisance or unresolved.

    The function never drops sites to meet a resource budget. In strict mode it
    raises with the exact over-budget count instead.
    """
    if not isinstance(strict, (bool, np.bool_)):
        raise TypeError("strict must be a boolean")
    coordinates = np.asarray(all_site_coordinates)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1:] != (2,)
        or not np.issubdtype(coordinates.dtype, np.number)
        or np.iscomplexobj(coordinates)
        or np.any(~np.isfinite(coordinates))
    ):
        raise ValueError(
            "all_site_coordinates must be a finite real array of shape (n_site, 2)"
        )
    coordinates = _readonly_array(coordinates, dtype=np.float64)
    if coordinates.shape[0] == 0:
        raise ValueError("all_site_coordinates must contain at least one site")
    centers = _integer_array("site_center_indices", site_center_indices, 2)
    starts = _integer_array("site_patch_starts", site_patch_starts, 2)
    shapes = _integer_array("site_patch_shapes", site_patch_shapes, 2)
    if (
        centers.shape != coordinates.shape
        or starts.shape != coordinates.shape
        or shapes.shape != coordinates.shape
    ):
        raise ValueError(
            "site centers, patch starts, and patch shapes must each have "
            "shape (n_site, 2)"
        )
    if np.any(shapes <= 0):
        raise ValueError("site_patch_shapes must contain positive dimensions")
    centers_inside = all(
        int(start_s) <= int(center_s) < int(start_s) + int(shape_s)
        and int(start_u) <= int(center_u) < int(start_u) + int(shape_u)
        for (center_s, center_u), (start_s, start_u), (shape_s, shape_u)
        in zip(centers, starts, shapes)
    )
    if not centers_inside:
        raise ValueError("every site center must lie inside its patch footprint")
    target_mask = _boolean_array("target_pixel_mask", target_pixel_mask, 2)
    forward_mask = _boolean_array("forward_pixel_mask", forward_pixel_mask, 2)
    if target_mask.shape != forward_mask.shape or not target_mask.size:
        raise ValueError("target and forward pixel masks must have one non-empty shape")
    if np.any(target_mask & ~forward_mask):
        raise ValueError("target_pixel_mask must be a subset of forward_pixel_mask")
    if exterior_policy not in _EXTERIOR_POLICIES_1D:
        raise ValueError(
            "exterior_policy must be 'parameterize_uncertain' or "
            "'leave_unresolved'"
        )
    if known_fixed_site_mask is None:
        known_fixed = np.zeros(len(coordinates), dtype=bool)
    else:
        known_fixed = np.asarray(known_fixed_site_mask)
        if known_fixed.dtype != np.bool_ or known_fixed.shape != (len(coordinates),):
            raise TypeError(
                "known_fixed_site_mask must be a Boolean vector with one value per site"
            )
        known_fixed = known_fixed.copy()
    if fixed_material_provenance_id is not None and (
        not isinstance(fixed_material_provenance_id, str)
        or not fixed_material_provenance_id.strip()
    ):
        raise ValueError(
            "fixed_material_provenance_id must be a non-empty string or None"
        )
    omitted_power = _finite_scalar("excluded_probe_power", excluded_probe_power)
    if not 0.0 < omitted_power < 1.0:
        raise ValueError("excluded_probe_power must lie strictly in (0, 1)")
    cutoff = _finite_scalar(
        "atomic_template_cutoff_A", atomic_template_cutoff_A, positive=True
    )
    displacement = _finite_scalar(
        "maximum_displacement_A",
        maximum_displacement_A,
        positive=True,
        allow_zero=True,
    )
    control_shape = _validated_control_shape(displacement_control_shape)
    removed_dof = _nonnegative_integer(
        "removed_displacement_dof", removed_displacement_dof
    )
    registration_count = _nonnegative_integer(
        "registration_parameter_count", registration_parameter_count
    )
    nuisance_budget = _nonnegative_integer(
        "maximum_nuisance_sites", maximum_nuisance_sites
    )
    parameter_budget = _nonnegative_integer(
        "maximum_specimen_parameters", maximum_specimen_parameters
    )

    target_centers = _target_centers_in_mask(centers, target_mask)
    forward_relevant = _forward_footprint_mask(starts, shapes, forward_mask)
    roles = np.full(
        len(coordinates),
        int(LatticeSiteRole1D.BELOW_INTERACTION_BUDGET),
        dtype=np.int8,
    )
    roles[target_centers] = int(LatticeSiteRole1D.TARGET)
    fixed_exterior = known_fixed & ~target_centers
    roles[fixed_exterior] = int(LatticeSiteRole1D.FIXED_KNOWN)
    uncertain_forward = forward_relevant & ~target_centers & ~fixed_exterior
    uncertain_role = (
        LatticeSiteRole1D.NUISANCE
        if exterior_policy == "parameterize_uncertain"
        else LatticeSiteRole1D.UNRESOLVED
    )
    roles[uncertain_forward] = int(uncertain_role)
    modeled = np.flatnonzero(
        (roles == int(LatticeSiteRole1D.TARGET))
        | (roles == int(LatticeSiteRole1D.NUISANCE))
    ).astype(np.int64, copy=False)
    target_sites = roles == int(LatticeSiteRole1D.TARGET)
    nuisance_sites = roles == int(LatticeSiteRole1D.NUISANCE)
    target_influence = _influence_mask(
        starts, shapes, target_sites, target_mask.shape
    )
    nuisance_influence = _influence_mask(
        starts, shapes, nuisance_sites, target_mask.shape
    )
    counts = _parameter_counts(
        roles,
        control_shape,
        removed_dof,
        registration_count,
    )
    provisional = LatticeSiteSupportContract1D(
        schema_version=_SUPPORT_SCHEMA_VERSION_1D,
        classification_contract=_CLASSIFICATION_CONTRACT_1D,
        all_site_coordinates=coordinates,
        site_center_indices=centers,
        site_patch_starts=starts,
        site_patch_shapes=shapes,
        target_pixel_mask=target_mask,
        forward_pixel_mask=forward_mask,
        target_center_mask=_readonly_array(target_centers, dtype=bool),
        forward_relevant_mask=_readonly_array(forward_relevant, dtype=bool),
        site_role_codes=_readonly_array(roles, dtype=np.int8),
        modeled_site_indices=_readonly_array(modeled, dtype=np.int64),
        target_influence_mask=_readonly_array(target_influence, dtype=bool),
        nuisance_influence_mask=_readonly_array(nuisance_influence, dtype=bool),
        exterior_policy=exterior_policy,
        excluded_probe_power=omitted_power,
        atomic_template_cutoff_A=cutoff,
        maximum_displacement_A=displacement,
        fixed_material_provenance_id=fixed_material_provenance_id,
        displacement_control_shape=control_shape,
        removed_displacement_dof=removed_dof,
        registration_parameter_count=registration_count,
        maximum_nuisance_sites=nuisance_budget,
        maximum_specimen_parameters=parameter_budget,
        parameter_counts=counts,
        contract_id="",
    )
    contract = replace(
        provisional,
        contract_id=_support_contract_digest(provisional),
    )
    return validate_lattice_site_support_contract_1d(
        contract,
        strict=bool(strict),
    )


def _validate_canonical_array(
    name: str,
    value: Any,
    *,
    dtype: np.dtype[Any],
    ndim: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"contract {name} must be a NumPy array")
    if value.dtype != dtype or value.ndim != ndim or not value.flags.c_contiguous:
        raise TypeError(
            f"contract {name} must be a C-contiguous {dtype} {ndim}D array"
        )
    if value.flags.writeable:
        raise ValueError(f"contract {name} must be read-only")
    return value


def validate_lattice_site_support_contract_1d(
    contract: LatticeSiteSupportContract1D,
    *,
    strict: bool = True,
) -> LatticeSiteSupportContract1D:
    """Validate numerical, semantic, provenance, budget, and digest invariants."""
    if not isinstance(contract, LatticeSiteSupportContract1D):
        raise TypeError("contract must be a LatticeSiteSupportContract1D")
    if not isinstance(strict, (bool, np.bool_)):
        raise TypeError("strict must be a boolean")
    if contract.schema_version != _SUPPORT_SCHEMA_VERSION_1D:
        raise ValueError("unsupported lattice-site support schema version")
    if contract.classification_contract != _CLASSIFICATION_CONTRACT_1D:
        raise ValueError("unsupported lattice-site classification contract")
    coordinates = _validate_canonical_array(
        "all_site_coordinates",
        contract.all_site_coordinates,
        dtype=np.dtype(np.float64),
        ndim=2,
    )
    centers = _validate_canonical_array(
        "site_center_indices",
        contract.site_center_indices,
        dtype=np.dtype(np.int64),
        ndim=2,
    )
    starts = _validate_canonical_array(
        "site_patch_starts",
        contract.site_patch_starts,
        dtype=np.dtype(np.int64),
        ndim=2,
    )
    shapes = _validate_canonical_array(
        "site_patch_shapes",
        contract.site_patch_shapes,
        dtype=np.dtype(np.int64),
        ndim=2,
    )
    target_mask = _validate_canonical_array(
        "target_pixel_mask",
        contract.target_pixel_mask,
        dtype=np.dtype(bool),
        ndim=2,
    )
    forward_mask = _validate_canonical_array(
        "forward_pixel_mask",
        contract.forward_pixel_mask,
        dtype=np.dtype(bool),
        ndim=2,
    )
    target_centers = _validate_canonical_array(
        "target_center_mask",
        contract.target_center_mask,
        dtype=np.dtype(bool),
        ndim=1,
    )
    forward_relevant = _validate_canonical_array(
        "forward_relevant_mask",
        contract.forward_relevant_mask,
        dtype=np.dtype(bool),
        ndim=1,
    )
    roles = _validate_canonical_array(
        "site_role_codes",
        contract.site_role_codes,
        dtype=np.dtype(np.int8),
        ndim=1,
    )
    modeled = _validate_canonical_array(
        "modeled_site_indices",
        contract.modeled_site_indices,
        dtype=np.dtype(np.int64),
        ndim=1,
    )
    target_influence = _validate_canonical_array(
        "target_influence_mask",
        contract.target_influence_mask,
        dtype=np.dtype(bool),
        ndim=2,
    )
    nuisance_influence = _validate_canonical_array(
        "nuisance_influence_mask",
        contract.nuisance_influence_mask,
        dtype=np.dtype(bool),
        ndim=2,
    )
    n_site = coordinates.shape[0]
    if coordinates.shape != (n_site, 2) or not n_site:
        raise ValueError("contract all_site_coordinates must have shape (n_site, 2)")
    if np.any(~np.isfinite(coordinates)):
        raise ValueError("contract site coordinates must be finite")
    if (
        centers.shape != (n_site, 2)
        or starts.shape != (n_site, 2)
        or shapes.shape != (n_site, 2)
    ):
        raise ValueError(
            "contract center and patch arrays must have shape (n_site, 2)"
        )
    if np.any(shapes <= 0):
        raise ValueError("contract patch shapes must be positive")
    centers_inside = all(
        int(start_s) <= int(center_s) < int(start_s) + int(shape_s)
        and int(start_u) <= int(center_u) < int(start_u) + int(shape_u)
        for (center_s, center_u), (start_s, start_u), (shape_s, shape_u)
        in zip(centers, starts, shapes)
    )
    if not centers_inside:
        raise ValueError("contract site centers must lie inside patch footprints")
    if target_mask.shape != forward_mask.shape or not target_mask.size:
        raise ValueError("contract pixel masks must have one non-empty shape")
    if (
        target_influence.shape != target_mask.shape
        or nuisance_influence.shape != target_mask.shape
    ):
        raise ValueError("contract influence masks must match the pixel masks")
    if np.any(target_mask & ~forward_mask):
        raise ValueError("contract target mask must be a subset of its forward mask")
    for name, value in (
        ("target_center_mask", target_centers),
        ("forward_relevant_mask", forward_relevant),
        ("site_role_codes", roles),
    ):
        if value.shape != (n_site,):
            raise ValueError(f"contract {name} must have one value per site")
    valid_roles = np.asarray([int(role) for role in LatticeSiteRole1D], dtype=np.int8)
    if np.any(~np.isin(roles, valid_roles)):
        raise ValueError("contract contains an unknown lattice-site role code")
    if contract.exterior_policy not in _EXTERIOR_POLICIES_1D:
        raise ValueError("contract exterior policy is unsupported")
    recomputed_target = _target_centers_in_mask(centers, target_mask)
    recomputed_forward = _forward_footprint_mask(
        starts, shapes, forward_mask
    )
    if not np.array_equal(target_centers, recomputed_target):
        raise ValueError("target-center mask is inconsistent with site centers")
    if not np.array_equal(forward_relevant, recomputed_forward):
        raise ValueError("forward-relevant mask is inconsistent with patches")

    target_role = roles == int(LatticeSiteRole1D.TARGET)
    nuisance_role = roles == int(LatticeSiteRole1D.NUISANCE)
    fixed_role = roles == int(LatticeSiteRole1D.FIXED_KNOWN)
    below_role = roles == int(LatticeSiteRole1D.BELOW_INTERACTION_BUDGET)
    unresolved_role = roles == int(LatticeSiteRole1D.UNRESOLVED)
    if not np.array_equal(target_role, target_centers):
        raise ValueError("TARGET roles must exactly match target center selection")
    if np.any(nuisance_role & (~forward_relevant | target_centers)):
        raise ValueError("NUISANCE roles must be non-target and forward relevant")
    if np.any(unresolved_role & (~forward_relevant | target_centers)):
        raise ValueError("UNRESOLVED roles must be non-target and forward relevant")
    if np.any(below_role & forward_relevant):
        raise ValueError("forward-relevant sites cannot be below interaction budget")
    exterior_forward = forward_relevant & ~target_centers & ~fixed_role
    expected_uncertain = (
        nuisance_role
        if contract.exterior_policy == "parameterize_uncertain"
        else unresolved_role
    )
    if not np.array_equal(expected_uncertain, exterior_forward):
        raise ValueError("site roles are inconsistent with the exterior policy")
    exterior_below = ~forward_relevant & ~fixed_role
    if not np.array_equal(below_role, exterior_below):
        raise ValueError("non-forward exterior roles are inconsistent")
    expected_modeled = np.flatnonzero(target_role | nuisance_role).astype(np.int64)
    if not np.array_equal(modeled, expected_modeled):
        raise ValueError("modeled_site_indices do not match TARGET and NUISANCE roles")
    expected_target_influence = _influence_mask(
        starts, shapes, target_role, target_mask.shape
    )
    expected_nuisance_influence = _influence_mask(
        starts, shapes, nuisance_role, target_mask.shape
    )
    if not np.array_equal(target_influence, expected_target_influence):
        raise ValueError("target influence mask is inconsistent with site roles")
    if not np.array_equal(nuisance_influence, expected_nuisance_influence):
        raise ValueError("nuisance influence mask is inconsistent with site roles")

    omitted_power = _finite_scalar(
        "contract.excluded_probe_power", contract.excluded_probe_power
    )
    if not 0.0 < omitted_power < 1.0:
        raise ValueError("contract excluded probe power must lie in (0, 1)")
    _finite_scalar(
        "contract.atomic_template_cutoff_A",
        contract.atomic_template_cutoff_A,
        positive=True,
    )
    _finite_scalar(
        "contract.maximum_displacement_A",
        contract.maximum_displacement_A,
        positive=True,
        allow_zero=True,
    )
    control_shape = _validated_control_shape(contract.displacement_control_shape)
    removed_dof = _nonnegative_integer(
        "contract.removed_displacement_dof",
        contract.removed_displacement_dof,
    )
    registration_count = _nonnegative_integer(
        "contract.registration_parameter_count",
        contract.registration_parameter_count,
    )
    nuisance_budget = _nonnegative_integer(
        "contract.maximum_nuisance_sites",
        contract.maximum_nuisance_sites,
    )
    parameter_budget = _nonnegative_integer(
        "contract.maximum_specimen_parameters",
        contract.maximum_specimen_parameters,
    )
    if not isinstance(contract.parameter_counts, LatticeSiteParameterCounts1D):
        raise TypeError(
            "contract.parameter_counts must be a LatticeSiteParameterCounts1D"
        )
    for count_field in fields(contract.parameter_counts):
        _nonnegative_integer(
            f"contract.parameter_counts.{count_field.name}",
            getattr(contract.parameter_counts, count_field.name),
        )
    expected_counts = _parameter_counts(
        roles, control_shape, removed_dof, registration_count
    )
    if contract.parameter_counts != expected_counts:
        raise ValueError("contract parameter counts are inconsistent with site roles")
    if contract.fixed_material_provenance_id is not None and (
        not isinstance(contract.fixed_material_provenance_id, str)
        or not contract.fixed_material_provenance_id.strip()
    ):
        raise ValueError(
            "fixed_material_provenance_id must be a non-empty string or None"
        )
    if (
        strict
        and fixed_role.any()
        and contract.fixed_material_provenance_id is None
    ):
        raise ValueError(
            "FIXED_KNOWN sites require fixed_material_provenance_id"
        )
    if strict and not target_role.any():
        raise ValueError("strict support contract requires at least one TARGET site")
    if strict and unresolved_role.any():
        indices = np.flatnonzero(unresolved_role)
        raise ValueError(
            f"{indices.size} forward-relevant site(s) remain UNRESOLVED; "
            f"first indices: {indices[:8].tolist()}"
        )
    if strict and expected_counts.nuisance_vacancy_parameters > nuisance_budget:
        raise ValueError(
            "nuisance-site budget exceeded: "
            f"{expected_counts.nuisance_vacancy_parameters} > {nuisance_budget}; "
            "no sites were truncated"
        )
    if strict and expected_counts.total_specimen_parameters > parameter_budget:
        raise ValueError(
            "specimen-parameter budget exceeded: "
            f"{expected_counts.total_specimen_parameters} > {parameter_budget}; "
            "no parameters were truncated"
        )
    expected_digest = _support_contract_digest(contract)
    if contract.contract_id != expected_digest:
        raise ValueError("contract_id does not match the support contract fields")
    return contract
