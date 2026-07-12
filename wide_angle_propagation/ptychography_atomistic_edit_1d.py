"""Sparse, object-agnostic atomistic edits for side-view ptychography.

This module implements the AE-1 state and renderer described in
``docs/ptychography_robustness.md``.  A known deformable host is retained from
the lattice-site model.  Sparse host removals and positive, continuous
off-lattice scattering centres are represented with fixed-capacity arrays and
explicit active masks.  Capacity is only a compilation/resource bound; all
reported parameter counts use the active state.

The maintained problem is two dimensional in ``(s, u)``.  It must not be
presented as a three-dimensional atom-localization method.  The optional
material-specific energy envelope is deliberately not implemented here: the
roadmap requires a separate chemistry/ablation gate before it may be enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import operator
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .ptychography_1d import (
    LatticeSiteModel1D,
    lattice_site_displacements_1d,
    render_lattice_site_potential_1d,
)
from .ptychography_support_contract_1d import (
    LatticeSiteParameterCounts1D,
    LatticeSiteRole1D,
    LatticeSiteSupportContract1D,
    validate_lattice_site_support_contract_1d,
)


__all__ = [
    "AtomisticEditDiscoverySupport1D",
    "AtomisticEditKernel1D",
    "AtomisticEditModel1D",
    "AtomisticEditOptions1D",
    "AtomisticEditPriorComponents1D",
    "AtomisticEditSnapshot1D",
    "AtomisticEditState1D",
    "AtomisticEditSupportContract1D",
    "atomistic_edit_active_parameter_count_1d",
    "atomistic_edit_addition_positions_1d",
    "atomistic_edit_addition_roles_1d",
    "atomistic_edit_prior_components_1d",
    "atomistic_edit_state_is_admissible_1d",
    "atomistic_edit_state_is_within_discovery_support_1d",
    "empty_atomistic_edit_state_1d",
    "load_atomistic_edit_snapshot_1d",
    "make_atomistic_edit_discovery_support_1d",
    "make_atomistic_edit_kernel_1d",
    "make_atomistic_edit_model_1d",
    "make_atomistic_edit_snapshot_1d",
    "render_atomistic_edit_potential_1d",
    "save_atomistic_edit_snapshot_1d",
    "validate_atomistic_edit_snapshot_1d",
    "validate_atomistic_edit_state_1d",
]


Array = Any
_SUPPORT_SCHEMA_VERSION = 1
_SIDEVIEW_SPATIAL_DIMENSION = 2
_ELASTIC_MODEL_ID = "symmetric_small_strain_equal_weight:v1"
_HARD_CORE_POLICY_ID = "occupancy_weighted_reciprocal_gap:onset_1.1:v1"
_SNAPSHOT_ARCHIVE_SCHEMA_VERSION = 1
_SNAPSHOT_ARCHIVE_CONTRACT = "atomistic_edit_snapshot_1d:authenticated_npz:v1"
_AE1_KKT_STATUS = "not_evaluated_ae1"
_AE1_CAPACITY_STATUS = "not_evaluated_ae1"


@dataclass(frozen=True, eq=False)
class AtomisticEditDiscoverySupport1D:
    """Geometry-derived centre-admissibility masks with reporting roles.

    ``target_mask`` is eligible for structural reporting. ``nuisance_mask`` is
    searched and fitted because it can affect the measured wave, but edits
    born there must never be presented as recovered specimen structure.
    Together they may include a declared surface-adjacent vacuum band; unlike
    the lattice-site mask, discovery is not implicitly clipped to known host
    material.
    """

    axial_coordinates_A: np.ndarray
    transverse_coordinates_A: np.ndarray
    target_mask: np.ndarray
    nuisance_mask: np.ndarray
    surface_envelope_A: tuple[float, float]
    geometry_source_id: str
    excluded_probe_power: float
    contract_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def discovery_mask(self) -> np.ndarray:
        result = np.asarray(self.target_mask) | np.asarray(self.nuisance_mask)
        result = np.array(result, copy=True)
        result.setflags(write=False)
        return result


@dataclass(frozen=True, eq=False)
class AtomisticEditKernel1D:
    """Positive, unit-integrated addition kernel and host-equivalent scale."""

    unit_integrated_values: np.ndarray
    centre_index: np.ndarray
    axial_sampling_A: float
    transverse_sampling_A: float
    host_equivalent_integrated_scattering: float
    parameterization_id: str
    cutoff_A: float
    projection_width_A: float
    boundary_mass_fraction: float
    normalization_tolerance: float
    kernel_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AtomisticEditOptions1D:
    """Physical/resource contract for the object-agnostic edit method.

    No field encodes object existence, number, centre, radius, shape, phase, or
    chemistry.  ``max_*`` values are static compilation capacities only.
    """

    max_host_removals: int
    max_extra_centres: int
    max_scattering_equivalent_per_centre: float
    minimum_separation_A: float
    expected_rms_host_strain: float
    edit_penalty_path: tuple[float, ...]
    discovery_support: AtomisticEditDiscoverySupport1D
    enable_material_energy_envelope: bool = False


@dataclass(frozen=True, eq=False)
class AtomisticEditSupportContract1D:
    """Digest-bound host, discovery, kernel, capacity, and physics contract."""

    schema_version: int
    host_support_contract_id: str
    discovery_contract_id: str
    kernel_id: str
    target_discovery_mask: np.ndarray
    nuisance_discovery_mask: np.ndarray
    addition_influence_mask: np.ndarray
    total_influence_mask: np.ndarray
    maximum_host_removals: int
    maximum_extra_centres: int
    maximum_scattering_equivalent_per_centre: float
    minimum_separation_A: float
    expected_rms_host_strain: float
    spatial_dimension: int
    deformation_parameter_count: int
    elastic_model_id: str
    hard_core_policy_id: str
    contract_id: str

    @property
    def strict_geometry_satisfied(self) -> bool:
        target = np.asarray(self.target_discovery_mask)
        nuisance = np.asarray(self.nuisance_discovery_mask)
        return bool(
            self.schema_version == _SUPPORT_SCHEMA_VERSION
            and target.dtype == np.bool_
            and nuisance.dtype == np.bool_
            and target.shape == nuisance.shape
            and np.any(target)
            and not np.any(target & nuisance)
            and self.maximum_host_removals >= 0
            and self.maximum_extra_centres >= 0
            and self.maximum_host_removals + self.maximum_extra_centres > 0
        )


@dataclass(frozen=True, eq=False)
class AtomisticEditModel1D:
    """Finite deformable host plus one generic positive addition kernel."""

    host_model: LatticeSiteModel1D
    axial_coordinates_A: Array
    transverse_coordinates_A: Array
    addition_kernel: AtomisticEditKernel1D
    options: AtomisticEditOptions1D
    support_contract: AtomisticEditSupportContract1D
    host_hard_core_pairs: Array
    deformation_parameter_count: int
    model_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class AtomisticEditState1D:
    """Fixed-capacity sparse state with continuous anchored additions.

    Addition positions are ``coordinate[anchor] + offset``.  Offsets are
    continuously optimized within half a specimen pixel.  Re-anchoring, when
    needed, is a discrete active-set operation outside the differentiated
    renderer; JAX therefore never differentiates through ``floor`` or nearest
    pixel selection.
    """

    host_removal_indices: Array
    host_removal_fractions: Array
    host_removal_active: Array
    extra_anchor_indices: Array
    extra_position_offsets_A: Array
    extra_scattering_equivalents: Array
    extra_active: Array
    host_displacement_controls: Array


@dataclass(frozen=True, eq=False)
class AtomisticEditPriorComponents1D:
    """Separately reported level-1 physical prior components."""

    edit_mass: Array
    weighted_edit_penalty: Array
    elastic_penalty: Array
    hard_core_penalty: Array
    total_prior: Array


@dataclass(frozen=True, eq=False)
class AtomisticEditSnapshot1D:
    """Authenticated AE-1 state that can independently rerender its specimen.

    This is deliberately not an optimizer result.  AE-1 has no active-set or
    KKT solver, so the status fields are fixed to fail-closed placeholders and
    ``converged`` must remain false.  ``data_objective_value`` is supplied by
    the caller together with a non-empty objective identifier; the physical
    prior components are recomputed from the archived model and state.
    """

    model: AtomisticEditModel1D
    state: AtomisticEditState1D
    rendered_potential: Array
    active_parameter_count: int
    selected_edit_penalty: float
    edit_penalty_rule_id: str
    data_objective_value: float
    data_objective_id: str
    prior_components: AtomisticEditPriorComponents1D
    total_objective_value: float
    kkt_status: str
    capacity_status: str
    converged: bool
    metadata: Mapping[str, Any]
    snapshot_id: str


def _readonly_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _jsonable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    mapping = dict(value or {})
    try:
        json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise TypeError("metadata must be JSON serializable") from error
    return MappingProxyType(mapping)


def _hash_arrays_and_metadata(
    arrays: Mapping[str, Any], metadata: Mapping[str, Any]
) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    digest.update(
        json.dumps(
            dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _positive_float(name: str, value: Any, *, allow_zero: bool = False) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real scalar")
    result = float(array)
    if not np.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


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


def _uniform_coordinates(name: str, value: Any) -> tuple[np.ndarray, float]:
    coordinates = np.asarray(value)
    if (
        coordinates.ndim != 1
        or coordinates.size < 2
        or np.iscomplexobj(coordinates)
        or np.any(~np.isfinite(coordinates))
    ):
        raise ValueError(f"{name} must be a finite real 1D array of length >= 2")
    differences = np.diff(coordinates.astype(float, copy=False))
    if np.issubdtype(coordinates.dtype, np.inexact):
        epsilon = min(
            float(np.finfo(coordinates.dtype).eps),
            float(np.finfo(np.float32).eps),
        )
    else:
        epsilon = float(np.finfo(float).eps)
    coordinate_scale = max(
        1.0,
        float(np.max(np.abs(coordinates))),
        float(abs(differences[0])),
    )
    relative_tolerance = max(1e-8, 8.0 * epsilon)
    absolute_tolerance = max(1e-12, 8.0 * epsilon * coordinate_scale)
    if np.any(differences <= 0.0) or not np.allclose(
        differences,
        differences[0],
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    ):
        raise ValueError(f"{name} must be uniformly increasing")
    return _readonly_array(coordinates), float(differences[0])


def _strictly_decreasing_penalty_path(value: Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("edit_penalty_path must be a sequence of positive values")
    try:
        path = tuple(_positive_float("edit penalty", item) for item in value)
    except TypeError as error:
        raise TypeError(
            "edit_penalty_path must be a sequence of positive values"
        ) from error
    if not path:
        raise ValueError("edit_penalty_path must not be empty")
    if any(left <= right for left, right in zip(path, path[1:])):
        raise ValueError("edit_penalty_path must be strictly decreasing")
    return path


def make_atomistic_edit_discovery_support_1d(
    axial_coordinates_A: Any,
    transverse_coordinates_A: Any,
    target_mask: Any,
    nuisance_mask: Any,
    *,
    surface_envelope_A: Sequence[float],
    geometry_source_id: str,
    excluded_probe_power: float,
    metadata: Mapping[str, Any] | None = None,
) -> AtomisticEditDiscoverySupport1D:
    """Create a typed TARGET/NUISANCE discovery-volume contract.

    The masks must be produced without object metadata.  They may include
    vacuum above the nominal host surface as long as it lies inside the
    declared broad surface envelope.
    """
    s_A, _ = _uniform_coordinates("axial_coordinates_A", axial_coordinates_A)
    u_A, _ = _uniform_coordinates(
        "transverse_coordinates_A", transverse_coordinates_A
    )
    target = np.asarray(target_mask)
    nuisance = np.asarray(nuisance_mask)
    expected_shape = (s_A.size, u_A.size)
    for name, mask in (("target_mask", target), ("nuisance_mask", nuisance)):
        if mask.shape != expected_shape or mask.dtype != np.bool_:
            raise TypeError(f"{name} must be a Boolean array of shape {expected_shape}")
    if not np.any(target):
        raise ValueError("target_mask must contain at least one discovery pixel")
    if np.any(target & nuisance):
        raise ValueError("target and nuisance discovery masks must be disjoint")
    if isinstance(surface_envelope_A, (str, bytes)):
        raise TypeError("surface_envelope_A must contain bottom and top bounds")
    try:
        bottom, top = tuple(float(item) for item in surface_envelope_A)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "surface_envelope_A must contain bottom and top bounds"
        ) from error
    if not np.isfinite(bottom) or not np.isfinite(top) or bottom >= top:
        raise ValueError("surface_envelope_A must be finite with bottom < top")
    selected_u = np.any(target | nuisance, axis=0)
    if np.any((u_A[selected_u] < bottom) | (u_A[selected_u] > top)):
        raise ValueError("discovery masks extend outside surface_envelope_A")
    if not isinstance(geometry_source_id, str) or not geometry_source_id.strip():
        raise ValueError("geometry_source_id must be a non-empty string")
    omitted_power = _positive_float("excluded_probe_power", excluded_probe_power)
    if omitted_power >= 1.0:
        raise ValueError("excluded_probe_power must lie strictly below 1")
    metadata = _jsonable_mapping(metadata)
    contract_metadata = {
        "schema": "atomistic_edit_discovery_support_1d:v1",
        "surface_envelope_A": [bottom, top],
        "geometry_source_id": geometry_source_id,
        "excluded_probe_power": omitted_power,
        "metadata": dict(metadata),
    }
    contract_id = _hash_arrays_and_metadata(
        {
            "axial_coordinates_A": s_A,
            "transverse_coordinates_A": u_A,
            "target_mask": target,
            "nuisance_mask": nuisance,
        },
        contract_metadata,
    )
    return AtomisticEditDiscoverySupport1D(
        axial_coordinates_A=s_A,
        transverse_coordinates_A=u_A,
        target_mask=_readonly_array(target, dtype=bool),
        nuisance_mask=_readonly_array(nuisance, dtype=bool),
        surface_envelope_A=(bottom, top),
        geometry_source_id=geometry_source_id,
        excluded_probe_power=omitted_power,
        contract_id=contract_id,
        metadata=metadata,
    )


def make_atomistic_edit_kernel_1d(
    values: Any,
    *,
    axial_sampling_A: float,
    transverse_sampling_A: float,
    host_equivalent_integrated_scattering: float,
    centre_index: Sequence[float] | None = None,
    parameterization_id: str,
    cutoff_A: float,
    projection_width_A: float,
    maximum_boundary_mass_fraction: float = 1e-6,
    normalization_tolerance: float = 1e-10,
    metadata: Mapping[str, Any] | None = None,
) -> AtomisticEditKernel1D:
    """Normalize one positive atom-like kernel and bind its physical scale."""
    kernel = np.asarray(values)
    if (
        kernel.ndim != 2
        or min(kernel.shape) < 3
        or np.iscomplexobj(kernel)
        or np.any(~np.isfinite(kernel))
    ):
        raise ValueError("values must be a finite real 2D kernel of shape >= (3, 3)")
    if np.any(kernel < 0.0) or not np.any(kernel > 0.0):
        raise ValueError("the added-scattering kernel must be non-negative and nonzero")
    ds = _positive_float("axial_sampling_A", axial_sampling_A)
    du = _positive_float("transverse_sampling_A", transverse_sampling_A)
    host_integral = _positive_float(
        "host_equivalent_integrated_scattering",
        host_equivalent_integrated_scattering,
    )
    cutoff = _positive_float("cutoff_A", cutoff_A)
    projection_width = _positive_float("projection_width_A", projection_width_A)
    boundary_limit = _positive_float(
        "maximum_boundary_mass_fraction",
        maximum_boundary_mass_fraction,
        allow_zero=True,
    )
    tolerance = _positive_float("normalization_tolerance", normalization_tolerance)
    if centre_index is None:
        centre = 0.5 * (np.asarray(kernel.shape, dtype=float) - 1.0)
    else:
        centre = np.asarray(tuple(centre_index), dtype=float)
    if centre.shape != (2,) or np.any(~np.isfinite(centre)):
        raise ValueError("centre_index must contain two finite coordinates")
    if np.any(centre < 0.0) or np.any(centre > np.asarray(kernel.shape) - 1.0):
        raise ValueError("centre_index must lie inside the kernel")
    if not isinstance(parameterization_id, str) or not parameterization_id.strip():
        raise ValueError("parameterization_id must be a non-empty string")
    integral = float(np.sum(kernel, dtype=np.float64) * ds * du)
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError("kernel integral must be finite and positive")
    unit_kernel = np.asarray(kernel, dtype=np.float64) / integral
    normalized_integral = float(np.sum(unit_kernel, dtype=np.float64) * ds * du)
    if not np.isclose(normalized_integral, 1.0, rtol=0.0, atol=tolerance):
        raise RuntimeError("unit-integrated kernel normalization failed")
    boundary = np.zeros(kernel.shape, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    boundary_mass_fraction = float(
        np.sum(unit_kernel[boundary], dtype=np.float64) * ds * du
    )
    if boundary_mass_fraction > boundary_limit:
        raise ValueError(
            "kernel boundary mass exceeds maximum_boundary_mass_fraction; "
            "increase the physically certified cutoff/padding"
        )
    metadata = _jsonable_mapping(metadata)
    identity_metadata = {
        "schema": "atomistic_edit_unit_kernel_1d:v1",
        "axial_sampling_A": ds,
        "transverse_sampling_A": du,
        "host_equivalent_integrated_scattering": host_integral,
        "parameterization_id": parameterization_id,
        "cutoff_A": cutoff,
        "projection_width_A": projection_width,
        "boundary_mass_fraction": boundary_mass_fraction,
        "normalization_tolerance": tolerance,
        "metadata": dict(metadata),
    }
    kernel_id = _hash_arrays_and_metadata(
        {"unit_integrated_values": unit_kernel, "centre_index": centre},
        identity_metadata,
    )
    return AtomisticEditKernel1D(
        unit_integrated_values=_readonly_array(unit_kernel),
        centre_index=_readonly_array(centre),
        axial_sampling_A=ds,
        transverse_sampling_A=du,
        host_equivalent_integrated_scattering=host_integral,
        parameterization_id=parameterization_id,
        cutoff_A=cutoff,
        projection_width_A=projection_width,
        boundary_mass_fraction=boundary_mass_fraction,
        normalization_tolerance=tolerance,
        kernel_id=kernel_id,
        metadata=metadata,
    )


def _rectangular_influence_mask(
    anchors: np.ndarray,
    *,
    start_offset: tuple[int, int],
    patch_shape: tuple[int, int],
) -> np.ndarray:
    """Return the union of identical anchor-relative rectangular footprints."""
    n_s, n_u = anchors.shape
    min_s, min_u = start_offset
    max_s = min_s + patch_shape[0] - 1
    max_u = min_u + patch_shape[1] - 1
    source_low_s = np.clip(np.arange(n_s) - max_s, 0, n_s)
    source_high_s = np.clip(np.arange(n_s) - min_s + 1, 0, n_s)
    source_low_u = np.clip(np.arange(n_u) - max_u, 0, n_u)
    source_high_u = np.clip(np.arange(n_u) - min_u + 1, 0, n_u)
    prefix = np.pad(anchors.astype(np.int64), ((1, 0), (1, 0)))
    prefix = np.cumsum(np.cumsum(prefix, axis=0), axis=1)
    high_high = prefix[np.ix_(source_high_s, source_high_u)]
    low_high = prefix[np.ix_(source_low_s, source_high_u)]
    high_low = prefix[np.ix_(source_high_s, source_low_u)]
    low_low = prefix[np.ix_(source_low_s, source_low_u)]
    return (high_high - low_high - high_low + low_low) > 0


def _host_hard_core_pairs(
    all_sites: np.ndarray,
    *,
    minimum_separation_A: float,
    maximum_displacement_A: float,
) -> np.ndarray:
    from scipy.spatial import cKDTree

    search_radius = 1.1 * minimum_separation_A + 2.0 * maximum_displacement_A
    pairs = cKDTree(all_sites).query_pairs(search_radius, output_type="ndarray")
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32)


def make_atomistic_edit_model_1d(
    host_model: LatticeSiteModel1D,
    axial_coordinates_A: Any,
    transverse_coordinates_A: Any,
    addition_kernel: AtomisticEditKernel1D,
    options: AtomisticEditOptions1D,
    *,
    deformation_parameter_count: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> AtomisticEditModel1D:
    """Bind a finite host, discovery volume, kernel, and physical edit policy."""
    if not isinstance(host_model, LatticeSiteModel1D):
        raise TypeError("host_model must be a LatticeSiteModel1D")
    if host_model.support_contract is None:
        raise ValueError("atomistic edits require a host material-support contract")
    host_support = validate_lattice_site_support_contract_1d(
        host_model.support_contract, strict=True
    )
    if not isinstance(addition_kernel, AtomisticEditKernel1D):
        raise TypeError("addition_kernel must be an AtomisticEditKernel1D")
    if not isinstance(options, AtomisticEditOptions1D):
        raise TypeError("options must be an AtomisticEditOptions1D")
    if not isinstance(options.discovery_support, AtomisticEditDiscoverySupport1D):
        raise TypeError(
            "options.discovery_support must be an "
            "AtomisticEditDiscoverySupport1D"
        )
    if not isinstance(options.enable_material_energy_envelope, (bool, np.bool_)):
        raise TypeError("enable_material_energy_envelope must be a boolean")
    if options.enable_material_energy_envelope:
        raise NotImplementedError(
            "the material-specific energy envelope is blocked until the "
            "chemistry validation and frozen ablation gate passes"
        )
    maximum_removals = _nonnegative_integer(
        "max_host_removals", options.max_host_removals
    )
    maximum_extras = _nonnegative_integer(
        "max_extra_centres", options.max_extra_centres
    )
    if maximum_removals + maximum_extras == 0:
        raise ValueError("at least one atomistic-edit capacity must be positive")
    maximum_mass = _positive_float(
        "max_scattering_equivalent_per_centre",
        options.max_scattering_equivalent_per_centre,
    )
    minimum_separation = _positive_float(
        "minimum_separation_A", options.minimum_separation_A
    )
    expected_strain = _positive_float(
        "expected_rms_host_strain", options.expected_rms_host_strain
    )
    penalty_path = _strictly_decreasing_penalty_path(options.edit_penalty_path)
    resolved_options = AtomisticEditOptions1D(
        max_host_removals=maximum_removals,
        max_extra_centres=maximum_extras,
        max_scattering_equivalent_per_centre=maximum_mass,
        minimum_separation_A=minimum_separation,
        expected_rms_host_strain=expected_strain,
        edit_penalty_path=penalty_path,
        discovery_support=options.discovery_support,
        enable_material_energy_envelope=False,
    )
    s_A, ds = _uniform_coordinates("axial_coordinates_A", axial_coordinates_A)
    u_A, du = _uniform_coordinates(
        "transverse_coordinates_A", transverse_coordinates_A
    )
    reference_shape = np.asarray(host_model.reference_potential).shape
    if reference_shape != (s_A.size, u_A.size):
        raise ValueError("coordinate axes must match host reference_potential")
    discovery = options.discovery_support
    if not np.array_equal(s_A, discovery.axial_coordinates_A) or not np.array_equal(
        u_A, discovery.transverse_coordinates_A
    ):
        raise ValueError("discovery-support coordinates do not match the host grid")
    if not np.isclose(ds, addition_kernel.axial_sampling_A) or not np.isclose(
        du, addition_kernel.transverse_sampling_A
    ):
        raise ValueError("addition-kernel sampling does not match the host grid")
    controls_shape = (
        len(host_model.control_coordinates_s),
        len(host_model.control_coordinates_u),
        2,
    )
    full_deformation_count = int(np.prod(controls_shape))
    if deformation_parameter_count is None:
        resolved_deformation_count = full_deformation_count
    else:
        resolved_deformation_count = _nonnegative_integer(
            "deformation_parameter_count", deformation_parameter_count
        )
        if resolved_deformation_count > full_deformation_count:
            raise ValueError(
                "deformation_parameter_count cannot exceed the control size"
            )
    centre = np.asarray(addition_kernel.centre_index)
    start_offset = tuple(
        np.floor(-centre + 0.5).astype(np.int64).tolist()
    )
    discovery_mask = np.asarray(discovery.discovery_mask)
    anchor_rows, anchor_columns = np.indices(reference_shape)
    full_kernel_footprint = (
        (anchor_rows + start_offset[0] >= 0)
        & (
            anchor_rows
            + start_offset[0]
            + addition_kernel.unit_integrated_values.shape[0]
            <= reference_shape[0]
        )
        & (anchor_columns + start_offset[1] >= 0)
        & (
            anchor_columns
            + start_offset[1]
            + addition_kernel.unit_integrated_values.shape[1]
            <= reference_shape[1]
        )
    )
    if np.any(discovery_mask & ~full_kernel_footprint):
        raise ValueError(
            "discovery support contains anchors whose complete addition-kernel "
            "footprint would leave the specimen grid; pad the grid or contract "
            "the object-free discovery volume"
        )
    addition_influence = _rectangular_influence_mask(
        discovery_mask,
        start_offset=start_offset,
        patch_shape=tuple(addition_kernel.unit_integrated_values.shape),
    )
    host_influence = np.asarray(host_support.target_influence_mask) | np.asarray(
        host_support.nuisance_influence_mask
    )
    total_influence = host_influence | addition_influence
    contract_metadata = {
        "schema": "atomistic_edit_support_contract_1d:v1",
        "host_support_contract_id": host_support.contract_id,
        "discovery_contract_id": discovery.contract_id,
        "kernel_id": addition_kernel.kernel_id,
        "maximum_host_removals": maximum_removals,
        "maximum_extra_centres": maximum_extras,
        "maximum_scattering_equivalent_per_centre": maximum_mass,
        "minimum_separation_A": minimum_separation,
        "expected_rms_host_strain": expected_strain,
        "spatial_dimension": _SIDEVIEW_SPATIAL_DIMENSION,
        "deformation_parameter_count": resolved_deformation_count,
        "elastic_model_id": _ELASTIC_MODEL_ID,
        "hard_core_policy_id": _HARD_CORE_POLICY_ID,
        "edit_penalty_path": list(penalty_path),
    }
    contract_id = _hash_arrays_and_metadata(
        {
            "target_discovery_mask": discovery.target_mask,
            "nuisance_discovery_mask": discovery.nuisance_mask,
            "addition_influence_mask": addition_influence,
            "total_influence_mask": total_influence,
        },
        contract_metadata,
    )
    support_contract = AtomisticEditSupportContract1D(
        schema_version=_SUPPORT_SCHEMA_VERSION,
        host_support_contract_id=host_support.contract_id,
        discovery_contract_id=discovery.contract_id,
        kernel_id=addition_kernel.kernel_id,
        target_discovery_mask=_readonly_array(discovery.target_mask, dtype=bool),
        nuisance_discovery_mask=_readonly_array(discovery.nuisance_mask, dtype=bool),
        addition_influence_mask=_readonly_array(addition_influence, dtype=bool),
        total_influence_mask=_readonly_array(total_influence, dtype=bool),
        maximum_host_removals=maximum_removals,
        maximum_extra_centres=maximum_extras,
        maximum_scattering_equivalent_per_centre=maximum_mass,
        minimum_separation_A=minimum_separation,
        expected_rms_host_strain=expected_strain,
        spatial_dimension=_SIDEVIEW_SPATIAL_DIMENSION,
        deformation_parameter_count=resolved_deformation_count,
        elastic_model_id=_ELASTIC_MODEL_ID,
        hard_core_policy_id=_HARD_CORE_POLICY_ID,
        contract_id=contract_id,
    )
    all_sites = np.asarray(host_support.all_site_coordinates, dtype=float)
    hard_core_pairs = _host_hard_core_pairs(
        all_sites,
        minimum_separation_A=minimum_separation,
        maximum_displacement_A=float(np.asarray(host_model.maximum_displacement)),
    )
    metadata = _jsonable_mapping(metadata)
    model_metadata = {
        "schema": "atomistic_edit_model_1d:v1",
        "host_support_contract_id": host_support.contract_id,
        "atomistic_edit_support_contract_id": contract_id,
        "kernel_id": addition_kernel.kernel_id,
        "metadata": dict(metadata),
    }
    model_id = _hash_arrays_and_metadata(
        {
            "host_reference_potential": host_model.reference_potential,
            "host_site_coordinates": host_model.site_coordinates,
            "host_site_patches": host_model.site_patches,
            "host_patch_starts": host_model.patch_starts,
            "host_control_coordinates_s": host_model.control_coordinates_s,
            "host_control_coordinates_u": host_model.control_coordinates_u,
            "unit_integrated_added_kernel": addition_kernel.unit_integrated_values,
            "host_hard_core_pairs": hard_core_pairs,
        },
        model_metadata,
    )
    return AtomisticEditModel1D(
        host_model=host_model,
        axial_coordinates_A=jnp.asarray(s_A),
        transverse_coordinates_A=jnp.asarray(u_A),
        addition_kernel=addition_kernel,
        options=resolved_options,
        support_contract=support_contract,
        host_hard_core_pairs=jnp.asarray(hard_core_pairs, dtype=jnp.int32),
        deformation_parameter_count=resolved_deformation_count,
        model_id=model_id,
        metadata=metadata,
    )


def empty_atomistic_edit_state_1d(model: AtomisticEditModel1D) -> AtomisticEditState1D:
    """Return the pristine, zero-deformation, empty-edit initialization."""
    if not isinstance(model, AtomisticEditModel1D):
        raise TypeError("model must be an AtomisticEditModel1D")
    dtype = jnp.asarray(model.host_model.reference_potential).dtype
    n_minus = model.options.max_host_removals
    n_plus = model.options.max_extra_centres
    first_anchor = np.argwhere(
        np.asarray(model.options.discovery_support.discovery_mask)
    )[0]
    anchors = np.broadcast_to(first_anchor, (n_plus, 2)).copy()
    controls_shape = (
        len(model.host_model.control_coordinates_s),
        len(model.host_model.control_coordinates_u),
        2,
    )
    return AtomisticEditState1D(
        host_removal_indices=jnp.zeros((n_minus,), dtype=jnp.int32),
        host_removal_fractions=jnp.zeros((n_minus,), dtype=dtype),
        host_removal_active=jnp.zeros((n_minus,), dtype=bool),
        extra_anchor_indices=jnp.asarray(anchors, dtype=jnp.int32),
        extra_position_offsets_A=jnp.zeros((n_plus, 2), dtype=dtype),
        extra_scattering_equivalents=jnp.zeros((n_plus,), dtype=dtype),
        extra_active=jnp.zeros((n_plus,), dtype=bool),
        host_displacement_controls=jnp.zeros(controls_shape, dtype=dtype),
    )


def _concrete_array(value: Any) -> np.ndarray | None:
    try:
        return np.asarray(value)
    except (TypeError, jax.errors.TracerArrayConversionError):
        return None


class _ContinuousAdditionSupportError(ValueError):
    """An active continuous centre crossed the hard discovery boundary."""


def _validate_continuous_addition_support(
    model: AtomisticEditModel1D,
    anchors: np.ndarray,
    offsets_A: np.ndarray,
    active: np.ndarray,
) -> None:
    """Require every active continuous centre to remain in discovery support.

    The Boolean discovery mask is sampled on the specimen grid.  A subpixel
    centre is admitted only when every grid vertex needed to bracket its
    displacement from the anchor is in the TARGET/NUISANCE union.  This
    conservative interpolation-cell rule prevents a boundary anchor from
    moving into an unmodelled region while still allowing motion toward an
    admitted neighbouring anchor.  The physical surface envelope is checked
    directly at the resulting continuous transverse coordinate.
    """
    active_slots = np.flatnonzero(np.asarray(active, dtype=bool))
    if not active_slots.size:
        return
    discovery = np.asarray(model.options.discovery_support.discovery_mask)
    sampling = np.asarray(
        [
            model.addition_kernel.axial_sampling_A,
            model.addition_kernel.transverse_sampling_A,
        ],
        dtype=float,
    )
    axes = (
        np.asarray(model.axial_coordinates_A, dtype=float),
        np.asarray(model.transverse_coordinates_A, dtype=float),
    )
    active_anchors = np.asarray(anchors, dtype=np.int64)[active_slots]
    active_offsets = np.asarray(offsets_A, dtype=float)[active_slots]
    positions = np.column_stack(
        [
            axes[0][active_anchors[:, 0]],
            axes[1][active_anchors[:, 1]],
        ]
    ) + active_offsets
    bottom_A, top_A = model.options.discovery_support.surface_envelope_A
    envelope_tolerance_A = 32.0 * np.finfo(float).eps * max(
        1.0, abs(bottom_A), abs(top_A)
    )
    if np.any(positions[:, 1] < bottom_A - envelope_tolerance_A) or np.any(
        positions[:, 1] > top_A + envelope_tolerance_A
    ):
        raise _ContinuousAdditionSupportError(
            "active continuous extra centres leave the declared surface envelope"
        )

    directions = np.sign(active_offsets / sampling[None, :]).astype(np.int64)
    for anchor, direction in zip(active_anchors, directions, strict=True):
        rows = (int(anchor[0]),)
        columns = (int(anchor[1]),)
        if direction[0]:
            rows += (int(anchor[0] + direction[0]),)
        if direction[1]:
            columns += (int(anchor[1] + direction[1]),)
        if (
            min(rows) < 0
            or max(rows) >= discovery.shape[0]
            or min(columns) < 0
            or max(columns) >= discovery.shape[1]
            or not np.all(discovery[np.ix_(rows, columns)])
        ):
            raise _ContinuousAdditionSupportError(
                "active continuous extra centres leave TARGET/NUISANCE "
                "discovery support"
            )


def validate_atomistic_edit_state_1d(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
) -> None:
    """Validate capacities, bounds, anchors, and active-index uniqueness."""
    if not isinstance(model, AtomisticEditModel1D):
        raise TypeError("model must be an AtomisticEditModel1D")
    if not isinstance(state, AtomisticEditState1D):
        raise TypeError("state must be an AtomisticEditState1D")
    n_minus = model.options.max_host_removals
    n_plus = model.options.max_extra_centres
    expected_controls = (
        len(model.host_model.control_coordinates_s),
        len(model.host_model.control_coordinates_u),
        2,
    )
    arrays = {
        "host_removal_indices": (state.host_removal_indices, (n_minus,)),
        "host_removal_fractions": (state.host_removal_fractions, (n_minus,)),
        "host_removal_active": (state.host_removal_active, (n_minus,)),
        "extra_anchor_indices": (state.extra_anchor_indices, (n_plus, 2)),
        "extra_position_offsets_A": (
            state.extra_position_offsets_A,
            (n_plus, 2),
        ),
        "extra_scattering_equivalents": (
            state.extra_scattering_equivalents,
            (n_plus,),
        ),
        "extra_active": (state.extra_active, (n_plus,)),
        "host_displacement_controls": (
            state.host_displacement_controls,
            expected_controls,
        ),
    }
    for name, (value, shape) in arrays.items():
        if tuple(jnp.shape(value)) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    if not jnp.issubdtype(jnp.asarray(state.host_removal_indices).dtype, jnp.integer):
        raise TypeError("host_removal_indices must contain integers")
    if not jnp.issubdtype(jnp.asarray(state.extra_anchor_indices).dtype, jnp.integer):
        raise TypeError("extra_anchor_indices must contain integers")
    if jnp.asarray(state.host_removal_active).dtype != jnp.bool_:
        raise TypeError("host_removal_active must be Boolean")
    if jnp.asarray(state.extra_active).dtype != jnp.bool_:
        raise TypeError("extra_active must be Boolean")
    for name in (
        "host_removal_fractions",
        "extra_position_offsets_A",
        "extra_scattering_equivalents",
        "host_displacement_controls",
    ):
        if jnp.iscomplexobj(jnp.asarray(getattr(state, name))):
            raise TypeError(f"{name} must be real")
    concrete = {name: _concrete_array(value) for name, (value, _) in arrays.items()}
    if any(
        value is not None
        and value.dtype != np.bool_
        and not np.issubdtype(value.dtype, np.integer)
        and np.any(~np.isfinite(value))
        for value in concrete.values()
    ):
        raise ValueError("atomistic-edit state arrays must be finite")
    indices = concrete["host_removal_indices"]
    removal_active = concrete["host_removal_active"]
    removals = concrete["host_removal_fractions"]
    anchors = concrete["extra_anchor_indices"]
    offsets = concrete["extra_position_offsets_A"]
    masses = concrete["extra_scattering_equivalents"]
    extra_active = concrete["extra_active"]
    controls = concrete["host_displacement_controls"]
    if indices is not None:
        n_host = len(model.host_model.site_coordinates)
        if np.any(indices < 0) or np.any(indices >= n_host):
            raise ValueError("host_removal_indices lie outside the host model")
        if removal_active is not None:
            active_indices = indices[removal_active]
            if np.unique(active_indices).size != active_indices.size:
                raise ValueError("active host-removal indices must be unique")
    if removals is not None and (
        np.any(removals < 0.0) or np.any(removals > 1.0)
    ):
        raise ValueError("host_removal_fractions must lie in [0, 1]")
    if masses is not None and (
        np.any(masses < 0.0)
        or np.any(
            masses > model.options.max_scattering_equivalent_per_centre
        )
    ):
        raise ValueError("extra scattering equivalents exceed their physical range")
    if anchors is not None:
        shape = np.asarray(model.support_contract.target_discovery_mask).shape
        if (
            np.any(anchors[:, 0] < 0)
            or np.any(anchors[:, 0] >= shape[0])
            or np.any(anchors[:, 1] < 0)
            or np.any(anchors[:, 1] >= shape[1])
        ):
            raise ValueError("extra anchors lie outside the specimen grid")
        if extra_active is not None:
            discovery = np.asarray(model.options.discovery_support.discovery_mask)
            active_anchors = anchors[extra_active]
            if active_anchors.size and not np.all(
                discovery[active_anchors[:, 0], active_anchors[:, 1]]
            ):
                raise ValueError("active extra anchors lie outside discovery support")
    if offsets is not None:
        half_pixel = 0.5 * np.asarray(
            [
                model.addition_kernel.axial_sampling_A,
                model.addition_kernel.transverse_sampling_A,
            ]
        )
        if np.any(np.abs(offsets) > half_pixel + 1e-12):
            raise ValueError(
                "extra position offsets must stay within half a specimen pixel; "
                "re-anchor the active centre before further refinement"
            )
    if (
        anchors is not None
        and offsets is not None
        and extra_active is not None
    ):
        _validate_continuous_addition_support(
            model,
            anchors,
            offsets,
            extra_active,
        )
    if controls is not None and np.any(
        np.abs(controls) > float(np.asarray(model.host_model.maximum_displacement))
    ):
        raise ValueError("host displacement controls exceed the host-model bound")


def atomistic_edit_addition_positions_1d(
    model: AtomisticEditModel1D, state: AtomisticEditState1D
) -> Array:
    """Return continuous ``(s, u)`` positions for every addition slot."""
    validate_atomistic_edit_state_1d(model, state)
    anchors = jnp.asarray(state.extra_anchor_indices)
    s_A = jnp.asarray(model.axial_coordinates_A)
    u_A = jnp.asarray(model.transverse_coordinates_A)
    anchor_positions = jnp.stack(
        [s_A[anchors[:, 0]], u_A[anchors[:, 1]]], axis=-1
    )
    return anchor_positions + jnp.asarray(state.extra_position_offsets_A)


def atomistic_edit_addition_roles_1d(
    model: AtomisticEditModel1D, state: AtomisticEditState1D
) -> Array:
    """Return TARGET/NUISANCE role codes for active anchors, zero if dormant."""
    validate_atomistic_edit_state_1d(model, state)
    anchors = jnp.asarray(state.extra_anchor_indices)
    active = jnp.asarray(state.extra_active)
    target = jnp.asarray(model.support_contract.target_discovery_mask)[
        anchors[:, 0], anchors[:, 1]
    ]
    role = jnp.where(
        target,
        int(LatticeSiteRole1D.TARGET),
        int(LatticeSiteRole1D.NUISANCE),
    )
    return jnp.where(active, role, 0).astype(jnp.int8)


def _dense_host_removals(
    model: AtomisticEditModel1D, state: AtomisticEditState1D
) -> Array:
    n_host = len(model.host_model.site_coordinates)
    dtype = jnp.asarray(state.host_removal_fractions).dtype
    values = jnp.where(
        jnp.asarray(state.host_removal_active),
        jnp.asarray(state.host_removal_fractions),
        jnp.zeros((), dtype=dtype),
    )
    return jnp.zeros((n_host,), dtype=dtype).at[
        jnp.asarray(state.host_removal_indices)
    ].add(values)


def _shift_patch_axis_linear(
    patch: Array, shift_pixels: Array, *, axis: int
) -> Array:
    """Positivity-preserving zero-extended linear translation."""
    length = patch.shape[axis]
    targets = jnp.arange(length, dtype=jnp.int32)
    source = targets.astype(shift_pixels.dtype) - shift_pixels
    lower = jnp.floor(source).astype(jnp.int32)
    fraction = source - lower.astype(source.dtype)
    upper = lower + 1
    lower_valid = (lower >= 0) & (lower < length)
    upper_valid = (upper >= 0) & (upper < length)
    lower_samples = jnp.take(patch, jnp.clip(lower, 0, length - 1), axis=axis)
    upper_samples = jnp.take(patch, jnp.clip(upper, 0, length - 1), axis=axis)
    if axis == 0:
        return (
            jnp.where(lower_valid[:, None], lower_samples, 0.0)
            * (1.0 - fraction)[:, None]
            + jnp.where(upper_valid[:, None], upper_samples, 0.0)
            * fraction[:, None]
        )
    return (
        jnp.where(lower_valid[None, :], lower_samples, 0.0)
        * (1.0 - fraction)[None, :]
        + jnp.where(upper_valid[None, :], upper_samples, 0.0)
        * fraction[None, :]
    )


def _render_extra_centres(
    model: AtomisticEditModel1D, state: AtomisticEditState1D
) -> Array:
    reference = jnp.asarray(model.host_model.reference_potential)
    kernel = jnp.asarray(model.addition_kernel.unit_integrated_values).astype(
        reference.dtype
    )
    centre = jnp.asarray(model.addition_kernel.centre_index, dtype=reference.dtype)
    anchors = jnp.asarray(state.extra_anchor_indices)
    offsets_A = jnp.asarray(state.extra_position_offsets_A, dtype=reference.dtype)
    start_offset = jnp.floor(-centre + 0.5).astype(jnp.int32)
    starts = anchors + start_offset[None, :]
    base_shift = -(start_offset.astype(reference.dtype) + centre)
    sampling = jnp.asarray(
        [
            model.addition_kernel.axial_sampling_A,
            model.addition_kernel.transverse_sampling_A,
        ],
        dtype=reference.dtype,
    )
    shifts = base_shift[None, :] + offsets_A / sampling[None, :]

    def shift_one(shift: Array) -> Array:
        shifted = _shift_patch_axis_linear(kernel, shift[0], axis=0)
        return _shift_patch_axis_linear(shifted, shift[1], axis=1)

    shifted = jax.vmap(shift_one)(shifts)
    active_mass = jnp.where(
        jnp.asarray(state.extra_active),
        jnp.asarray(state.extra_scattering_equivalents, dtype=reference.dtype),
        0.0,
    )
    scaled = shifted * active_mass[:, None, None] * jnp.asarray(
        model.addition_kernel.host_equivalent_integrated_scattering,
        dtype=reference.dtype,
    )
    offsets_s = jnp.arange(kernel.shape[0], dtype=jnp.int32)
    offsets_u = jnp.arange(kernel.shape[1], dtype=jnp.int32)
    rows = starts[:, 0, None, None] + offsets_s[None, :, None]
    columns = starts[:, 1, None, None] + offsets_u[None, None, :]
    rows = jnp.broadcast_to(rows, scaled.shape)
    columns = jnp.broadcast_to(columns, scaled.shape)
    valid = (
        (rows >= 0)
        & (rows < reference.shape[0])
        & (columns >= 0)
        & (columns < reference.shape[1])
    )
    flat_indices = jnp.clip(rows, 0, reference.shape[0] - 1) * reference.shape[
        1
    ] + jnp.clip(columns, 0, reference.shape[1] - 1)
    flat = jnp.zeros(reference.size, dtype=reference.dtype)
    flat = flat.at[flat_indices.reshape(-1)].add(
        jnp.where(valid, scaled, 0.0).reshape(-1)
    )
    return flat.reshape(reference.shape)


def render_atomistic_edit_potential_1d(
    model: AtomisticEditModel1D, state: AtomisticEditState1D
) -> Array:
    """Render the deformable host, sparse removals, and positive additions."""
    validate_atomistic_edit_state_1d(model, state)
    host = render_lattice_site_potential_1d(
        model.host_model,
        _dense_host_removals(model, state),
        state.host_displacement_controls,
    )
    return host + _render_extra_centres(model, state)


def atomistic_edit_active_parameter_count_1d(
    model: AtomisticEditModel1D, state: AtomisticEditState1D
) -> int:
    """Return ``P_deformation + K_- + 3 K_+`` for the side-view model."""
    validate_atomistic_edit_state_1d(model, state)
    removal_active = np.asarray(state.host_removal_active) & (
        np.asarray(state.host_removal_fractions) > 0.0
    )
    extra_active = np.asarray(state.extra_active) & (
        np.asarray(state.extra_scattering_equivalents) > 0.0
    )
    return int(
        model.deformation_parameter_count
        + np.count_nonzero(removal_active)
        + (_SIDEVIEW_SPATIAL_DIMENSION + 1) * np.count_nonzero(extra_active)
    )


def _symmetric_strain_penalty(model: AtomisticEditModel1D, controls: Array) -> Array:
    controls = jnp.asarray(controls)
    ds_control = (
        jnp.asarray(model.host_model.control_coordinates_s[1])
        - jnp.asarray(model.host_model.control_coordinates_s[0])
        if len(model.host_model.control_coordinates_s) > 1
        else jnp.asarray(1.0, dtype=controls.dtype)
    )
    du_control = (
        jnp.asarray(model.host_model.control_coordinates_u[1])
        - jnp.asarray(model.host_model.control_coordinates_u[0])
        if len(model.host_model.control_coordinates_u) > 1
        else jnp.asarray(1.0, dtype=controls.dtype)
    )
    d_us_ds = (
        jnp.diff(controls[..., 0], axis=0) / ds_control
        if controls.shape[0] > 1
        else jnp.zeros((0, controls.shape[1]), dtype=controls.dtype)
    )
    d_uu_ds = (
        jnp.diff(controls[..., 1], axis=0) / ds_control
        if controls.shape[0] > 1
        else jnp.zeros((0, controls.shape[1]), dtype=controls.dtype)
    )
    d_us_du = (
        jnp.diff(controls[..., 0], axis=1) / du_control
        if controls.shape[1] > 1
        else jnp.zeros((controls.shape[0], 0), dtype=controls.dtype)
    )
    d_uu_du = (
        jnp.diff(controls[..., 1], axis=1) / du_control
        if controls.shape[1] > 1
        else jnp.zeros((controls.shape[0], 0), dtype=controls.dtype)
    )
    terms = []
    if d_us_ds.size:
        terms.append(jnp.mean(d_us_ds**2))
    if d_uu_du.size:
        terms.append(jnp.mean(d_uu_du**2))
    if d_us_du.size and d_uu_ds.size:
        common_s = min(d_us_du.shape[0], d_uu_ds.shape[0])
        common_u = min(d_us_du.shape[1], d_uu_ds.shape[1])
        shear = 0.5 * (
            d_us_du[:common_s, :common_u]
            + d_uu_ds[:common_s, :common_u]
        )
        terms.append(2.0 * jnp.mean(shear**2))
    if not terms:
        return jnp.asarray(0.0, dtype=controls.dtype)
    strain_norm = sum(terms) / len(terms)
    sigma = jnp.asarray(
        model.options.expected_rms_host_strain, dtype=controls.dtype
    )
    return 0.5 * strain_norm / sigma**2


def _hard_core_phi(distance: Array, minimum_separation: Array) -> Array:
    onset = 1.1 * minimum_separation
    numerator = jnp.maximum(onset - distance, 0.0)
    numerical_gap = jnp.maximum(
        distance - minimum_separation,
        1e-6 * minimum_separation,
    )
    return jnp.where(distance < onset, (numerator / numerical_gap) ** 2, 0.0)


def _differentiable_pair_distance(
    difference: Array,
    minimum_separation: Array,
) -> Array:
    """Return a finite-gradient norm even for coincident dormant slots.

    ``jax.numpy.linalg.norm`` has an undefined derivative at an exactly zero
    vector.  Fixed-capacity dormant additions deliberately share an inert
    anchor, so tracing the occupancy-weighted hard-core term through that norm
    can otherwise produce ``0 * NaN`` gradients.  The tiny physical-scale
    floor is far below every admissibility tolerance; exact active overlaps
    are still rejected by the hard constraint before refinement.
    """
    squared = jnp.sum(jnp.asarray(difference) ** 2, axis=-1)
    floor = 1e-12 * jnp.asarray(minimum_separation, dtype=squared.dtype)
    return jnp.sqrt(squared + floor**2)


def _hard_core_penalty(model: AtomisticEditModel1D, state: AtomisticEditState1D) -> Array:
    support = model.host_model.support_contract
    assert isinstance(support, LatticeSiteSupportContract1D)
    all_sites = jnp.asarray(support.all_site_coordinates)
    modeled = jnp.asarray(support.modeled_site_indices)
    displacements = lattice_site_displacements_1d(
        jnp.asarray(model.host_model.site_coordinates),
        jnp.asarray(state.host_displacement_controls),
        jnp.asarray(model.host_model.control_coordinates_s),
        jnp.asarray(model.host_model.control_coordinates_u),
    )
    all_displacements = jnp.zeros_like(all_sites).at[modeled].set(displacements)
    displaced_hosts = all_sites + all_displacements
    dense_removals = _dense_host_removals(model, state)
    all_removals = jnp.zeros((all_sites.shape[0],), dtype=dense_removals.dtype).at[
        modeled
    ].set(dense_removals)
    host_occupancy = 1.0 - all_removals
    minimum = jnp.asarray(
        model.options.minimum_separation_A, dtype=displaced_hosts.dtype
    )
    pairs = jnp.asarray(model.host_hard_core_pairs)
    if pairs.shape[0]:
        pair_distance = _differentiable_pair_distance(
            displaced_hosts[pairs[:, 0]] - displaced_hosts[pairs[:, 1]],
            minimum,
        )
        pair_weight = host_occupancy[pairs[:, 0]] * host_occupancy[pairs[:, 1]]
        host_host = jnp.sum(pair_weight * _hard_core_phi(pair_distance, minimum))
    else:
        host_host = jnp.asarray(0.0, dtype=displaced_hosts.dtype)
    extra_positions = atomistic_edit_addition_positions_1d(model, state)
    active_mass = jnp.where(
        jnp.asarray(state.extra_active),
        jnp.asarray(state.extra_scattering_equivalents),
        0.0,
    )
    normalized_mass = (
        active_mass / model.options.max_scattering_equivalent_per_centre
    )
    if extra_positions.shape[0]:
        host_extra_distance = _differentiable_pair_distance(
            displaced_hosts[:, None, :] - extra_positions[None, :, :],
            minimum,
        )
        host_extra_weight = host_occupancy[:, None] * normalized_mass[None, :]
        host_extra = jnp.sum(
            host_extra_weight * _hard_core_phi(host_extra_distance, minimum)
        )
        difference = extra_positions[:, None, :] - extra_positions[None, :, :]
        extra_distance = _differentiable_pair_distance(difference, minimum)
        upper = jnp.triu(jnp.ones(extra_distance.shape, dtype=bool), k=1)
        extra_weight = normalized_mass[:, None] * normalized_mass[None, :]
        extra_extra = jnp.sum(
            jnp.where(
                upper,
                extra_weight * _hard_core_phi(extra_distance, minimum),
                0.0,
            )
        )
    else:
        host_extra = jnp.asarray(0.0, dtype=displaced_hosts.dtype)
        extra_extra = jnp.asarray(0.0, dtype=displaced_hosts.dtype)
    return host_host + host_extra + extra_extra


def atomistic_edit_prior_components_1d(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
    edit_penalty: Any,
) -> AtomisticEditPriorComponents1D:
    """Evaluate edit mass, weak elasticity, and occupancy-weighted hard core."""
    validate_atomistic_edit_state_1d(model, state)
    penalty = _positive_float("edit_penalty", edit_penalty)
    removal_mass = jnp.sum(
        jnp.where(
            jnp.asarray(state.host_removal_active),
            jnp.asarray(state.host_removal_fractions),
            0.0,
        )
    )
    addition_mass = jnp.sum(
        jnp.where(
            jnp.asarray(state.extra_active),
            jnp.asarray(state.extra_scattering_equivalents),
            0.0,
        )
    )
    edit_mass = removal_mass + addition_mass
    weighted = jnp.asarray(penalty, dtype=edit_mass.dtype) * edit_mass
    elastic = _symmetric_strain_penalty(
        model, state.host_displacement_controls
    )
    hard_core = _hard_core_penalty(model, state)
    return AtomisticEditPriorComponents1D(
        edit_mass=edit_mass,
        weighted_edit_penalty=weighted,
        elastic_penalty=elastic,
        hard_core_penalty=hard_core,
        total_prior=weighted + elastic + hard_core,
    )


def atomistic_edit_state_is_within_discovery_support_1d(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
) -> bool:
    """Return whether active continuous centres obey the hard geometry bound.

    Invalid state shapes, dtypes, capacities, or physical parameter ranges
    still raise through the public state validator.  Only a continuous-centre
    crossing of the TARGET/NUISANCE union or surface envelope returns false.
    """
    try:
        validate_atomistic_edit_state_1d(model, state)
    except _ContinuousAdditionSupportError:
        return False
    return True


def atomistic_edit_state_is_admissible_1d(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
    *,
    tolerance_A: float = 1e-9,
) -> bool:
    """Check the hard discovery boundary and minimum active-centre separation."""
    if not atomistic_edit_state_is_within_discovery_support_1d(model, state):
        return False
    tolerance = _positive_float("tolerance_A", tolerance_A, allow_zero=True)
    support = model.host_model.support_contract
    assert isinstance(support, LatticeSiteSupportContract1D)
    host_sites = np.asarray(support.all_site_coordinates, dtype=float)
    modeled = np.asarray(support.modeled_site_indices, dtype=np.int64)
    controls = np.asarray(state.host_displacement_controls, dtype=float)
    host_displacements = np.asarray(
        lattice_site_displacements_1d(
            np.asarray(model.host_model.site_coordinates),
            controls,
            np.asarray(model.host_model.control_coordinates_s),
            np.asarray(model.host_model.control_coordinates_u),
        )
    )
    displaced_hosts = host_sites.copy()
    displaced_hosts[modeled] += host_displacements
    removals = np.asarray(_dense_host_removals(model, state))
    all_removals = np.zeros(len(host_sites), dtype=float)
    all_removals[modeled] = removals
    occupied_hosts = all_removals < 1.0 - 1e-12
    active_extra = np.asarray(state.extra_active) & (
        np.asarray(state.extra_scattering_equivalents) > 0.0
    )
    extra_positions = np.asarray(atomistic_edit_addition_positions_1d(model, state))[
        active_extra
    ]
    minimum = model.options.minimum_separation_A - tolerance
    if extra_positions.size:
        host_distance = np.linalg.norm(
            displaced_hosts[occupied_hosts, None, :]
            - extra_positions[None, :, :],
            axis=2,
        )
        if np.any(host_distance < minimum):
            return False
        if len(extra_positions) > 1:
            difference = extra_positions[:, None, :] - extra_positions[None, :, :]
            distances = np.linalg.norm(difference, axis=2)
            upper = np.triu(np.ones(distances.shape, dtype=bool), k=1)
            if np.any(distances[upper] < minimum):
                return False
    return True


def _finite_float(name: str, value: Any) -> float:
    array = np.asarray(value)
    if array.shape != () or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonempty_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _canonical_json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_archive_mapping(
    value: Any,
    *,
    name: str,
    expected_fields: set[str] | None = None,
) -> dict[str, Any]:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind != "U":
        raise ValueError(f"{name} must be a scalar Unicode JSON value")
    serialized = str(array.item())

    def reject_constant(constant: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON constant {constant}")

    try:
        decoded = json.loads(serialized, parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid finite JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must encode a JSON object")
    if expected_fields is not None and set(decoded) != expected_fields:
        missing = sorted(expected_fields - set(decoded))
        extra = sorted(set(decoded) - expected_fields)
        raise ValueError(
            f"{name} fields differ from schema: missing={missing}, extra={extra}"
        )
    if serialized != _canonical_json_text(decoded):
        raise ValueError(f"{name} must use canonical JSON serialization")
    return decoded


def _require_exact_array(name: str, actual: Any, expected: Any) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if (
        actual_array.dtype != expected_array.dtype
        or actual_array.shape != expected_array.shape
        or not np.array_equal(actual_array, expected_array)
    ):
        raise ValueError(f"{name} does not match its authenticated value")


def _validated_discovery_identity(
    discovery: AtomisticEditDiscoverySupport1D,
) -> AtomisticEditDiscoverySupport1D:
    if not isinstance(discovery, AtomisticEditDiscoverySupport1D):
        raise TypeError("discovery support has the wrong type")
    rebuilt = make_atomistic_edit_discovery_support_1d(
        discovery.axial_coordinates_A,
        discovery.transverse_coordinates_A,
        discovery.target_mask,
        discovery.nuisance_mask,
        surface_envelope_A=discovery.surface_envelope_A,
        geometry_source_id=discovery.geometry_source_id,
        excluded_probe_power=discovery.excluded_probe_power,
        metadata=discovery.metadata,
    )
    if rebuilt.contract_id != discovery.contract_id:
        raise ValueError("discovery contract_id does not match its fields")
    return rebuilt


def _validated_kernel_identity(
    kernel: AtomisticEditKernel1D,
) -> AtomisticEditKernel1D:
    if not isinstance(kernel, AtomisticEditKernel1D):
        raise TypeError("addition kernel has the wrong type")
    values = np.asarray(kernel.unit_integrated_values)
    centre = np.asarray(kernel.centre_index)
    if (
        values.dtype != np.float64
        or values.ndim != 2
        or min(values.shape) < 3
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
        or not np.any(values > 0.0)
    ):
        raise ValueError(
            "unit_integrated_values must be a finite non-negative float64 2D kernel"
        )
    if (
        centre.dtype != np.float64
        or centre.shape != (2,)
        or np.any(~np.isfinite(centre))
        or np.any(centre < 0.0)
        or np.any(centre > np.asarray(values.shape) - 1.0)
    ):
        raise ValueError("kernel centre_index is invalid")
    ds = _positive_float("kernel.axial_sampling_A", kernel.axial_sampling_A)
    du = _positive_float(
        "kernel.transverse_sampling_A", kernel.transverse_sampling_A
    )
    host_integral = _positive_float(
        "kernel.host_equivalent_integrated_scattering",
        kernel.host_equivalent_integrated_scattering,
    )
    cutoff = _positive_float("kernel.cutoff_A", kernel.cutoff_A)
    projection_width = _positive_float(
        "kernel.projection_width_A", kernel.projection_width_A
    )
    boundary_fraction = _positive_float(
        "kernel.boundary_mass_fraction",
        kernel.boundary_mass_fraction,
        allow_zero=True,
    )
    tolerance = _positive_float(
        "kernel.normalization_tolerance", kernel.normalization_tolerance
    )
    parameterization_id = _nonempty_identifier(
        "kernel.parameterization_id", kernel.parameterization_id
    )
    normalized_integral = float(np.sum(values, dtype=np.float64) * ds * du)
    if not np.isclose(normalized_integral, 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("addition kernel is not unit integrated")
    boundary = np.zeros(values.shape, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    recomputed_boundary = float(
        np.sum(values[boundary], dtype=np.float64) * ds * du
    )
    boundary_tolerance = max(tolerance, 16.0 * np.finfo(float).eps)
    if not np.isclose(
        recomputed_boundary,
        boundary_fraction,
        rtol=0.0,
        atol=boundary_tolerance,
    ):
        raise ValueError("kernel boundary_mass_fraction does not match its values")
    metadata = _jsonable_mapping(kernel.metadata)
    identity_metadata = {
        "schema": "atomistic_edit_unit_kernel_1d:v1",
        "axial_sampling_A": ds,
        "transverse_sampling_A": du,
        "host_equivalent_integrated_scattering": host_integral,
        "parameterization_id": parameterization_id,
        "cutoff_A": cutoff,
        "projection_width_A": projection_width,
        "boundary_mass_fraction": boundary_fraction,
        "normalization_tolerance": tolerance,
        "metadata": dict(metadata),
    }
    expected_id = _hash_arrays_and_metadata(
        {"unit_integrated_values": values, "centre_index": centre},
        identity_metadata,
    )
    if kernel.kernel_id != expected_id:
        raise ValueError("kernel_id does not match the normalized kernel fields")
    return AtomisticEditKernel1D(
        unit_integrated_values=_readonly_array(values, dtype=np.float64),
        centre_index=_readonly_array(centre, dtype=np.float64),
        axial_sampling_A=ds,
        transverse_sampling_A=du,
        host_equivalent_integrated_scattering=host_integral,
        parameterization_id=parameterization_id,
        cutoff_A=cutoff,
        projection_width_A=projection_width,
        boundary_mass_fraction=boundary_fraction,
        normalization_tolerance=tolerance,
        kernel_id=kernel.kernel_id,
        metadata=metadata,
    )


_EDIT_SUPPORT_ARRAY_FIELDS = (
    "target_discovery_mask",
    "nuisance_discovery_mask",
    "addition_influence_mask",
    "total_influence_mask",
)


def _require_same_edit_support(
    actual: AtomisticEditSupportContract1D,
    expected: AtomisticEditSupportContract1D,
) -> None:
    if not isinstance(expected, AtomisticEditSupportContract1D):
        raise TypeError("atomistic support contract has the wrong type")
    for name in _EDIT_SUPPORT_ARRAY_FIELDS:
        _require_exact_array(
            f"atomistic support {name}",
            getattr(actual, name),
            getattr(expected, name),
        )
    for name in (
        "schema_version",
        "host_support_contract_id",
        "discovery_contract_id",
        "kernel_id",
        "maximum_host_removals",
        "maximum_extra_centres",
        "maximum_scattering_equivalent_per_centre",
        "minimum_separation_A",
        "expected_rms_host_strain",
        "spatial_dimension",
        "deformation_parameter_count",
        "elastic_model_id",
        "hard_core_policy_id",
        "contract_id",
    ):
        if getattr(actual, name) != getattr(expected, name):
            raise ValueError(
                f"atomistic support {name} does not match its derived value"
            )
    if not expected.strict_geometry_satisfied:
        raise ValueError("atomistic support contract fails strict geometry")


def _validated_model_identity(model: AtomisticEditModel1D) -> None:
    if not isinstance(model, AtomisticEditModel1D):
        raise TypeError("model must be an AtomisticEditModel1D")
    if model.host_model.support_contract is None:
        raise ValueError("atomistic host model omits its material-support contract")
    validate_lattice_site_support_contract_1d(
        model.host_model.support_contract,
        strict=True,
    )
    discovery = _validated_discovery_identity(model.options.discovery_support)
    kernel = _validated_kernel_identity(model.addition_kernel)
    if model.support_contract.host_support_contract_id != (
        model.host_model.support_contract.contract_id
    ):
        raise ValueError("atomistic and host support-contract IDs disagree")
    if model.support_contract.discovery_contract_id != discovery.contract_id:
        raise ValueError("atomistic support uses the wrong discovery contract")
    if model.support_contract.kernel_id != kernel.kernel_id:
        raise ValueError("atomistic support uses the wrong addition kernel")
    rebuilt = make_atomistic_edit_model_1d(
        model.host_model,
        model.axial_coordinates_A,
        model.transverse_coordinates_A,
        kernel,
        AtomisticEditOptions1D(
            max_host_removals=model.options.max_host_removals,
            max_extra_centres=model.options.max_extra_centres,
            max_scattering_equivalent_per_centre=(
                model.options.max_scattering_equivalent_per_centre
            ),
            minimum_separation_A=model.options.minimum_separation_A,
            expected_rms_host_strain=model.options.expected_rms_host_strain,
            edit_penalty_path=model.options.edit_penalty_path,
            discovery_support=discovery,
            enable_material_energy_envelope=False,
        ),
        deformation_parameter_count=model.deformation_parameter_count,
        metadata=model.metadata,
    )
    _require_same_edit_support(rebuilt.support_contract, model.support_contract)
    _require_exact_array(
        "model host_hard_core_pairs",
        rebuilt.host_hard_core_pairs,
        model.host_hard_core_pairs,
    )
    _require_exact_array(
        "model axial_coordinates_A",
        rebuilt.axial_coordinates_A,
        model.axial_coordinates_A,
    )
    _require_exact_array(
        "model transverse_coordinates_A",
        rebuilt.transverse_coordinates_A,
        model.transverse_coordinates_A,
    )
    if rebuilt.model_id != model.model_id:
        raise ValueError("model_id does not match the atomistic model fields")
    if dict(rebuilt.metadata) != dict(model.metadata):
        raise ValueError("atomistic model metadata are not canonical")


def _canonical_state(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
) -> AtomisticEditState1D:
    """Return one fixed-capacity representation of a physical sparse state.

    Sparse slot order is an implementation detail and must not change a saved
    specimen identity.  Positive active removals are ordered by host index;
    positive active additions are ordered lexicographically by their anchored
    continuous coordinates.  Zero-amplitude and explicitly inactive slots are
    physically dormant, so they are reset to deterministic inert values after
    the active prefix.  Host deformation controls are not sparse and retain
    their original ordering.
    """
    removal_indices = np.asarray(state.host_removal_indices)
    removal_fractions = np.asarray(state.host_removal_fractions)
    removal_active = np.asarray(state.host_removal_active, dtype=bool)
    physical_removal = removal_active & (removal_fractions > 0.0)
    removal_slots = np.flatnonzero(physical_removal)
    if removal_slots.size:
        removal_slots = removal_slots[
            np.argsort(removal_indices[removal_slots], kind="stable")
        ]
    canonical_removal_indices = np.zeros_like(removal_indices)
    canonical_removal_fractions = np.zeros_like(removal_fractions)
    canonical_removal_active = np.zeros_like(removal_active)
    n_removal = removal_slots.size
    canonical_removal_indices[:n_removal] = removal_indices[removal_slots]
    canonical_removal_fractions[:n_removal] = removal_fractions[removal_slots]
    canonical_removal_active[:n_removal] = True

    extra_anchors = np.asarray(state.extra_anchor_indices)
    extra_offsets = np.asarray(state.extra_position_offsets_A)
    extra_masses = np.asarray(state.extra_scattering_equivalents)
    extra_active = np.asarray(state.extra_active, dtype=bool)
    physical_extra = extra_active & (extra_masses > 0.0)
    extra_slots = np.flatnonzero(physical_extra)
    if extra_slots.size:
        continuous_s = (
            np.asarray(model.axial_coordinates_A)[extra_anchors[extra_slots, 0]]
            + extra_offsets[extra_slots, 0]
        )
        continuous_u = (
            np.asarray(model.transverse_coordinates_A)[
                extra_anchors[extra_slots, 1]
            ]
            + extra_offsets[extra_slots, 1]
        )
        order = np.lexsort(
            (
                extra_masses[extra_slots],
                extra_offsets[extra_slots, 1],
                extra_offsets[extra_slots, 0],
                extra_anchors[extra_slots, 1],
                extra_anchors[extra_slots, 0],
                continuous_u,
                continuous_s,
            )
        )
        extra_slots = extra_slots[order]
    first_anchor = np.argwhere(
        np.asarray(model.options.discovery_support.discovery_mask)
    )[0]
    canonical_extra_anchors = np.broadcast_to(
        first_anchor, extra_anchors.shape
    ).copy()
    canonical_extra_offsets = np.zeros_like(extra_offsets)
    canonical_extra_masses = np.zeros_like(extra_masses)
    canonical_extra_active = np.zeros_like(extra_active)
    n_extra = extra_slots.size
    canonical_extra_anchors[:n_extra] = extra_anchors[extra_slots]
    canonical_extra_offsets[:n_extra] = extra_offsets[extra_slots]
    canonical_extra_masses[:n_extra] = extra_masses[extra_slots]
    canonical_extra_active[:n_extra] = True

    return AtomisticEditState1D(
        host_removal_indices=jnp.asarray(canonical_removal_indices),
        host_removal_fractions=jnp.asarray(canonical_removal_fractions),
        host_removal_active=jnp.asarray(canonical_removal_active),
        extra_anchor_indices=jnp.asarray(canonical_extra_anchors),
        extra_position_offsets_A=jnp.asarray(canonical_extra_offsets),
        extra_scattering_equivalents=jnp.asarray(canonical_extra_masses),
        extra_active=jnp.asarray(canonical_extra_active),
        host_displacement_controls=jnp.asarray(
            state.host_displacement_controls
        ),
    )


def _prior_float_components(
    components: AtomisticEditPriorComponents1D,
) -> AtomisticEditPriorComponents1D:
    if not isinstance(components, AtomisticEditPriorComponents1D):
        raise TypeError("prior_components has the wrong type")
    values = {
        name: _finite_float(
            f"prior_components.{name}", getattr(components, name)
        )
        for name in (
            "edit_mass",
            "weighted_edit_penalty",
            "elastic_penalty",
            "hard_core_penalty",
            "total_prior",
        )
    }
    expected_total = (
        values["weighted_edit_penalty"]
        + values["elastic_penalty"]
        + values["hard_core_penalty"]
    )
    if values["total_prior"] != expected_total:
        raise ValueError("total_prior does not equal its reported components")
    return AtomisticEditPriorComponents1D(**values)


def _snapshot_digest(snapshot: AtomisticEditSnapshot1D) -> str:
    state = snapshot.state
    prior = snapshot.prior_components
    return _hash_arrays_and_metadata(
        {
            "rendered_potential": snapshot.rendered_potential,
            "host_removal_indices": state.host_removal_indices,
            "host_removal_fractions": state.host_removal_fractions,
            "host_removal_active": state.host_removal_active,
            "extra_anchor_indices": state.extra_anchor_indices,
            "extra_position_offsets_A": state.extra_position_offsets_A,
            "extra_scattering_equivalents": (
                state.extra_scattering_equivalents
            ),
            "extra_active": state.extra_active,
            "host_displacement_controls": state.host_displacement_controls,
        },
        {
            "schema": "atomistic_edit_snapshot_1d:v1",
            "model_id": snapshot.model.model_id,
            "host_support_contract_id": (
                snapshot.model.host_model.support_contract.contract_id
            ),
            "atomistic_support_contract_id": (
                snapshot.model.support_contract.contract_id
            ),
            "discovery_contract_id": (
                snapshot.model.options.discovery_support.contract_id
            ),
            "kernel_id": snapshot.model.addition_kernel.kernel_id,
            "active_parameter_count": snapshot.active_parameter_count,
            "selected_edit_penalty": snapshot.selected_edit_penalty,
            "edit_penalty_rule_id": snapshot.edit_penalty_rule_id,
            "data_objective_value": snapshot.data_objective_value,
            "data_objective_id": snapshot.data_objective_id,
            "prior_components": {
                "edit_mass": prior.edit_mass,
                "weighted_edit_penalty": prior.weighted_edit_penalty,
                "elastic_penalty": prior.elastic_penalty,
                "hard_core_penalty": prior.hard_core_penalty,
                "total_prior": prior.total_prior,
            },
            "total_objective_value": snapshot.total_objective_value,
            "kkt_status": snapshot.kkt_status,
            "capacity_status": snapshot.capacity_status,
            "converged": snapshot.converged,
            "host_model_metadata": dict(snapshot.model.host_model.metadata),
            "discovery_metadata": dict(
                snapshot.model.options.discovery_support.metadata
            ),
            "kernel_metadata": dict(snapshot.model.addition_kernel.metadata),
            "model_metadata": dict(snapshot.model.metadata),
            "snapshot_metadata": dict(snapshot.metadata),
        },
    )


def make_atomistic_edit_snapshot_1d(
    model: AtomisticEditModel1D,
    state: AtomisticEditState1D,
    *,
    selected_edit_penalty: Any,
    edit_penalty_rule_id: str,
    data_objective_value: Any,
    data_objective_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> AtomisticEditSnapshot1D:
    """Create a fail-closed, rerenderable AE-1 snapshot.

    The selected penalty must be one member of the model's frozen path.  Data
    objective evaluation remains outside AE-1, so its finite value and typed
    identifier are recorded but never treated as KKT or convergence evidence.
    """
    _validated_model_identity(model)
    validate_atomistic_edit_state_1d(model, state)
    canonical_state = _canonical_state(model, state)
    validate_atomistic_edit_state_1d(model, canonical_state)
    if not atomistic_edit_state_is_admissible_1d(model, canonical_state):
        raise ValueError("atomistic-edit state violates a hard admissibility rule")
    penalty = _positive_float("selected_edit_penalty", selected_edit_penalty)
    if penalty not in model.options.edit_penalty_path:
        raise ValueError(
            "selected_edit_penalty must be an exact member of edit_penalty_path"
        )
    rule_id = _nonempty_identifier(
        "edit_penalty_rule_id", edit_penalty_rule_id
    )
    objective_value = _finite_float(
        "data_objective_value", data_objective_value
    )
    objective_id = _nonempty_identifier("data_objective_id", data_objective_id)
    metadata_value = _jsonable_mapping(metadata)
    rendered = jnp.asarray(
        render_atomistic_edit_potential_1d(model, canonical_state)
    )
    rendered_host = np.asarray(rendered)
    if np.iscomplexobj(rendered_host) or np.any(~np.isfinite(rendered_host)):
        raise ValueError("rendered atomistic-edit potential must be finite and real")
    active_count = atomistic_edit_active_parameter_count_1d(
        model, canonical_state
    )
    prior = _prior_float_components(
        atomistic_edit_prior_components_1d(model, canonical_state, penalty)
    )
    total_objective = objective_value + float(prior.total_prior)
    if not np.isfinite(total_objective):
        raise ValueError("total objective value is not finite")
    provisional = AtomisticEditSnapshot1D(
        model=model,
        state=canonical_state,
        rendered_potential=rendered,
        active_parameter_count=active_count,
        selected_edit_penalty=penalty,
        edit_penalty_rule_id=rule_id,
        data_objective_value=objective_value,
        data_objective_id=objective_id,
        prior_components=prior,
        total_objective_value=total_objective,
        kkt_status=_AE1_KKT_STATUS,
        capacity_status=_AE1_CAPACITY_STATUS,
        converged=False,
        metadata=metadata_value,
        snapshot_id="",
    )
    snapshot = replace(provisional, snapshot_id=_snapshot_digest(provisional))
    return validate_atomistic_edit_snapshot_1d(snapshot)


def validate_atomistic_edit_snapshot_1d(
    snapshot: AtomisticEditSnapshot1D,
) -> AtomisticEditSnapshot1D:
    """Reverify renderer, active count, priors, IDs, and fail-closed status."""
    if not isinstance(snapshot, AtomisticEditSnapshot1D):
        raise TypeError("snapshot must be an AtomisticEditSnapshot1D")
    _validated_model_identity(snapshot.model)
    validate_atomistic_edit_state_1d(snapshot.model, snapshot.state)
    canonical_state = _canonical_state(snapshot.model, snapshot.state)
    for name in (
        "host_removal_indices",
        "host_removal_fractions",
        "host_removal_active",
        "extra_anchor_indices",
        "extra_position_offsets_A",
        "extra_scattering_equivalents",
        "extra_active",
        "host_displacement_controls",
    ):
        _require_exact_array(
            f"snapshot canonical state {name}",
            getattr(snapshot.state, name),
            getattr(canonical_state, name),
        )
    if not atomistic_edit_state_is_admissible_1d(snapshot.model, snapshot.state):
        raise ValueError("snapshot state violates a hard admissibility rule")
    penalty = _positive_float(
        "snapshot.selected_edit_penalty", snapshot.selected_edit_penalty
    )
    if penalty not in snapshot.model.options.edit_penalty_path:
        raise ValueError("snapshot penalty is not on the frozen penalty path")
    _nonempty_identifier(
        "snapshot.edit_penalty_rule_id", snapshot.edit_penalty_rule_id
    )
    data_value = _finite_float(
        "snapshot.data_objective_value", snapshot.data_objective_value
    )
    _nonempty_identifier("snapshot.data_objective_id", snapshot.data_objective_id)
    metadata = _jsonable_mapping(snapshot.metadata)
    if dict(metadata) != dict(snapshot.metadata):
        raise ValueError("snapshot metadata are not canonical")
    if snapshot.kkt_status != _AE1_KKT_STATUS:
        raise ValueError("AE-1 snapshots cannot claim a KKT evaluation")
    if snapshot.capacity_status != _AE1_CAPACITY_STATUS:
        raise ValueError("AE-1 snapshots cannot claim a capacity assessment")
    if not isinstance(snapshot.converged, (bool, np.bool_)):
        raise TypeError("snapshot.converged must be Boolean")
    if bool(snapshot.converged):
        raise ValueError("AE-1 snapshots cannot claim optimizer convergence")
    expected_count = atomistic_edit_active_parameter_count_1d(
        snapshot.model, snapshot.state
    )
    if (
        isinstance(snapshot.active_parameter_count, (bool, np.bool_))
        or operator.index(snapshot.active_parameter_count) != expected_count
    ):
        raise ValueError("snapshot active_parameter_count is inconsistent")
    rendered = np.asarray(
        render_atomistic_edit_potential_1d(snapshot.model, snapshot.state)
    )
    stored_rendered = np.asarray(snapshot.rendered_potential)
    if (
        stored_rendered.shape != rendered.shape
        or stored_rendered.dtype != rendered.dtype
        or not np.array_equal(stored_rendered, rendered)
    ):
        raise ValueError("snapshot potential does not exactly rerender")
    stored_prior = _prior_float_components(snapshot.prior_components)
    recomputed_prior = _prior_float_components(
        atomistic_edit_prior_components_1d(
            snapshot.model,
            snapshot.state,
            penalty,
        )
    )
    for name in (
        "edit_mass",
        "weighted_edit_penalty",
        "elastic_penalty",
        "hard_core_penalty",
        "total_prior",
    ):
        if getattr(stored_prior, name) != getattr(recomputed_prior, name):
            raise ValueError(
                f"snapshot prior component {name} is not reproducible"
            )
    expected_total = data_value + float(stored_prior.total_prior)
    if _finite_float(
        "snapshot.total_objective_value", snapshot.total_objective_value
    ) != expected_total:
        raise ValueError("snapshot total objective does not equal its components")
    if not isinstance(snapshot.snapshot_id, str) or len(snapshot.snapshot_id) != 64:
        raise ValueError("snapshot_id must be a SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in snapshot.snapshot_id):
        raise ValueError("snapshot_id must be a lowercase SHA-256 hex digest")
    if snapshot.snapshot_id != _snapshot_digest(
        replace(snapshot, snapshot_id="")
    ):
        raise ValueError("snapshot_id does not match the snapshot fields")
    return snapshot


_HOST_SUPPORT_JSON_FIELDS = {
    "schema_version",
    "classification_contract",
    "exterior_policy",
    "excluded_probe_power",
    "atomic_template_cutoff_A",
    "maximum_displacement_A",
    "fixed_material_provenance_id",
    "displacement_control_shape",
    "removed_displacement_dof",
    "registration_parameter_count",
    "maximum_nuisance_sites",
    "maximum_specimen_parameters",
    "parameter_counts",
    "contract_id",
}
_DISCOVERY_JSON_FIELDS = {
    "surface_envelope_A",
    "geometry_source_id",
    "excluded_probe_power",
    "contract_id",
    "metadata",
}
_KERNEL_JSON_FIELDS = {
    "axial_sampling_A",
    "transverse_sampling_A",
    "host_equivalent_integrated_scattering",
    "parameterization_id",
    "cutoff_A",
    "projection_width_A",
    "boundary_mass_fraction",
    "normalization_tolerance",
    "kernel_id",
    "metadata",
}
_EDIT_SUPPORT_JSON_FIELDS = {
    "schema_version",
    "host_support_contract_id",
    "discovery_contract_id",
    "kernel_id",
    "maximum_host_removals",
    "maximum_extra_centres",
    "maximum_scattering_equivalent_per_centre",
    "minimum_separation_A",
    "expected_rms_host_strain",
    "spatial_dimension",
    "deformation_parameter_count",
    "elastic_model_id",
    "hard_core_policy_id",
    "contract_id",
    "edit_penalty_path",
    "enable_material_energy_envelope",
}
_MODEL_JSON_FIELDS = {"deformation_parameter_count", "model_id", "metadata"}
_SNAPSHOT_JSON_FIELDS = {
    "active_parameter_count",
    "selected_edit_penalty",
    "edit_penalty_rule_id",
    "data_objective_value",
    "data_objective_id",
    "prior_components",
    "total_objective_value",
    "kkt_status",
    "capacity_status",
    "converged",
    "metadata",
    "snapshot_id",
}
_PRIOR_JSON_FIELDS = {
    "edit_mass",
    "weighted_edit_penalty",
    "elastic_penalty",
    "hard_core_penalty",
    "total_prior",
}

_SNAPSHOT_ARCHIVE_FIELDS = {
    "schema_version",
    "archive_contract",
    "host_reference_potential",
    "host_site_coordinates",
    "host_site_patches",
    "host_patch_starts",
    "host_control_coordinates_s",
    "host_control_coordinates_u",
    "host_axial_sampling",
    "host_transverse_sampling",
    "host_maximum_displacement",
    "host_metadata_json",
    "host_support_all_site_coordinates",
    "host_support_site_center_indices",
    "host_support_site_patch_starts",
    "host_support_site_patch_shapes",
    "host_support_target_pixel_mask",
    "host_support_forward_pixel_mask",
    "host_support_target_center_mask",
    "host_support_forward_relevant_mask",
    "host_support_site_role_codes",
    "host_support_modeled_site_indices",
    "host_support_target_influence_mask",
    "host_support_nuisance_influence_mask",
    "host_support_json",
    "discovery_axial_coordinates_A",
    "discovery_transverse_coordinates_A",
    "discovery_target_mask",
    "discovery_nuisance_mask",
    "discovery_json",
    "addition_kernel_values",
    "addition_kernel_centre_index",
    "addition_kernel_json",
    "edit_support_target_discovery_mask",
    "edit_support_nuisance_discovery_mask",
    "edit_support_addition_influence_mask",
    "edit_support_total_influence_mask",
    "edit_support_json",
    "model_axial_coordinates_A",
    "model_transverse_coordinates_A",
    "model_host_hard_core_pairs",
    "model_json",
    "state_host_removal_indices",
    "state_host_removal_fractions",
    "state_host_removal_active",
    "state_extra_anchor_indices",
    "state_extra_position_offsets_A",
    "state_extra_scattering_equivalents",
    "state_extra_active",
    "state_host_displacement_controls",
    "rendered_potential",
    "snapshot_json",
}


def _host_support_json(contract: LatticeSiteSupportContract1D) -> str:
    counts = contract.parameter_counts
    return _canonical_json_text(
        {
            "schema_version": int(contract.schema_version),
            "classification_contract": contract.classification_contract,
            "exterior_policy": contract.exterior_policy,
            "excluded_probe_power": float(contract.excluded_probe_power),
            "atomic_template_cutoff_A": float(
                contract.atomic_template_cutoff_A
            ),
            "maximum_displacement_A": float(contract.maximum_displacement_A),
            "fixed_material_provenance_id": (
                contract.fixed_material_provenance_id
            ),
            "displacement_control_shape": [
                int(value) for value in contract.displacement_control_shape
            ],
            "removed_displacement_dof": int(contract.removed_displacement_dof),
            "registration_parameter_count": int(
                contract.registration_parameter_count
            ),
            "maximum_nuisance_sites": int(contract.maximum_nuisance_sites),
            "maximum_specimen_parameters": int(
                contract.maximum_specimen_parameters
            ),
            "parameter_counts": {
                "target_vacancy_parameters": int(
                    counts.target_vacancy_parameters
                ),
                "nuisance_vacancy_parameters": int(
                    counts.nuisance_vacancy_parameters
                ),
                "displacement_control_parameters": int(
                    counts.displacement_control_parameters
                ),
                "removed_displacement_dof": int(
                    counts.removed_displacement_dof
                ),
                "residual_displacement_control_dof": int(
                    counts.residual_displacement_control_dof
                ),
                "registration_parameters": int(counts.registration_parameters),
                "total_specimen_parameters": int(
                    counts.total_specimen_parameters
                ),
            },
            "contract_id": contract.contract_id,
        }
    )


def _snapshot_archive_payload(
    snapshot: AtomisticEditSnapshot1D,
) -> dict[str, np.ndarray]:
    validate_atomistic_edit_snapshot_1d(snapshot)
    model = snapshot.model
    host = model.host_model
    host_support = host.support_contract
    assert isinstance(host_support, LatticeSiteSupportContract1D)
    discovery = model.options.discovery_support
    kernel = model.addition_kernel
    edit_support = model.support_contract
    state = snapshot.state
    prior = snapshot.prior_components
    discovery_json = _canonical_json_text(
        {
            "surface_envelope_A": [
                float(value) for value in discovery.surface_envelope_A
            ],
            "geometry_source_id": discovery.geometry_source_id,
            "excluded_probe_power": float(discovery.excluded_probe_power),
            "contract_id": discovery.contract_id,
            "metadata": dict(discovery.metadata),
        }
    )
    kernel_json = _canonical_json_text(
        {
            "axial_sampling_A": float(kernel.axial_sampling_A),
            "transverse_sampling_A": float(kernel.transverse_sampling_A),
            "host_equivalent_integrated_scattering": float(
                kernel.host_equivalent_integrated_scattering
            ),
            "parameterization_id": kernel.parameterization_id,
            "cutoff_A": float(kernel.cutoff_A),
            "projection_width_A": float(kernel.projection_width_A),
            "boundary_mass_fraction": float(kernel.boundary_mass_fraction),
            "normalization_tolerance": float(kernel.normalization_tolerance),
            "kernel_id": kernel.kernel_id,
            "metadata": dict(kernel.metadata),
        }
    )
    edit_support_json = _canonical_json_text(
        {
            "schema_version": int(edit_support.schema_version),
            "host_support_contract_id": edit_support.host_support_contract_id,
            "discovery_contract_id": edit_support.discovery_contract_id,
            "kernel_id": edit_support.kernel_id,
            "maximum_host_removals": int(edit_support.maximum_host_removals),
            "maximum_extra_centres": int(edit_support.maximum_extra_centres),
            "maximum_scattering_equivalent_per_centre": float(
                edit_support.maximum_scattering_equivalent_per_centre
            ),
            "minimum_separation_A": float(edit_support.minimum_separation_A),
            "expected_rms_host_strain": float(
                edit_support.expected_rms_host_strain
            ),
            "spatial_dimension": int(edit_support.spatial_dimension),
            "deformation_parameter_count": int(
                edit_support.deformation_parameter_count
            ),
            "elastic_model_id": edit_support.elastic_model_id,
            "hard_core_policy_id": edit_support.hard_core_policy_id,
            "contract_id": edit_support.contract_id,
            "edit_penalty_path": [
                float(value) for value in model.options.edit_penalty_path
            ],
            "enable_material_energy_envelope": bool(
                model.options.enable_material_energy_envelope
            ),
        }
    )
    snapshot_json = _canonical_json_text(
        {
            "active_parameter_count": int(snapshot.active_parameter_count),
            "selected_edit_penalty": float(snapshot.selected_edit_penalty),
            "edit_penalty_rule_id": snapshot.edit_penalty_rule_id,
            "data_objective_value": float(snapshot.data_objective_value),
            "data_objective_id": snapshot.data_objective_id,
            "prior_components": {
                "edit_mass": float(prior.edit_mass),
                "weighted_edit_penalty": float(prior.weighted_edit_penalty),
                "elastic_penalty": float(prior.elastic_penalty),
                "hard_core_penalty": float(prior.hard_core_penalty),
                "total_prior": float(prior.total_prior),
            },
            "total_objective_value": float(snapshot.total_objective_value),
            "kkt_status": snapshot.kkt_status,
            "capacity_status": snapshot.capacity_status,
            "converged": bool(snapshot.converged),
            "metadata": dict(snapshot.metadata),
            "snapshot_id": snapshot.snapshot_id,
        }
    )
    return {
        "schema_version": np.asarray(
            _SNAPSHOT_ARCHIVE_SCHEMA_VERSION, dtype=np.int64
        ),
        "archive_contract": np.asarray(_SNAPSHOT_ARCHIVE_CONTRACT),
        "host_reference_potential": np.asarray(host.reference_potential),
        "host_site_coordinates": np.asarray(host.site_coordinates),
        "host_site_patches": np.asarray(host.site_patches),
        "host_patch_starts": np.asarray(host.patch_starts),
        "host_control_coordinates_s": np.asarray(host.control_coordinates_s),
        "host_control_coordinates_u": np.asarray(host.control_coordinates_u),
        "host_axial_sampling": np.asarray(host.axial_sampling),
        "host_transverse_sampling": np.asarray(host.transverse_sampling),
        "host_maximum_displacement": np.asarray(host.maximum_displacement),
        "host_metadata_json": np.asarray(
            _canonical_json_text(dict(host.metadata))
        ),
        "host_support_all_site_coordinates": np.asarray(
            host_support.all_site_coordinates
        ),
        "host_support_site_center_indices": np.asarray(
            host_support.site_center_indices
        ),
        "host_support_site_patch_starts": np.asarray(
            host_support.site_patch_starts
        ),
        "host_support_site_patch_shapes": np.asarray(
            host_support.site_patch_shapes
        ),
        "host_support_target_pixel_mask": np.asarray(
            host_support.target_pixel_mask
        ),
        "host_support_forward_pixel_mask": np.asarray(
            host_support.forward_pixel_mask
        ),
        "host_support_target_center_mask": np.asarray(
            host_support.target_center_mask
        ),
        "host_support_forward_relevant_mask": np.asarray(
            host_support.forward_relevant_mask
        ),
        "host_support_site_role_codes": np.asarray(
            host_support.site_role_codes
        ),
        "host_support_modeled_site_indices": np.asarray(
            host_support.modeled_site_indices
        ),
        "host_support_target_influence_mask": np.asarray(
            host_support.target_influence_mask
        ),
        "host_support_nuisance_influence_mask": np.asarray(
            host_support.nuisance_influence_mask
        ),
        "host_support_json": np.asarray(_host_support_json(host_support)),
        "discovery_axial_coordinates_A": np.asarray(
            discovery.axial_coordinates_A
        ),
        "discovery_transverse_coordinates_A": np.asarray(
            discovery.transverse_coordinates_A
        ),
        "discovery_target_mask": np.asarray(discovery.target_mask),
        "discovery_nuisance_mask": np.asarray(discovery.nuisance_mask),
        "discovery_json": np.asarray(discovery_json),
        "addition_kernel_values": np.asarray(kernel.unit_integrated_values),
        "addition_kernel_centre_index": np.asarray(kernel.centre_index),
        "addition_kernel_json": np.asarray(kernel_json),
        "edit_support_target_discovery_mask": np.asarray(
            edit_support.target_discovery_mask
        ),
        "edit_support_nuisance_discovery_mask": np.asarray(
            edit_support.nuisance_discovery_mask
        ),
        "edit_support_addition_influence_mask": np.asarray(
            edit_support.addition_influence_mask
        ),
        "edit_support_total_influence_mask": np.asarray(
            edit_support.total_influence_mask
        ),
        "edit_support_json": np.asarray(edit_support_json),
        "model_axial_coordinates_A": np.asarray(model.axial_coordinates_A),
        "model_transverse_coordinates_A": np.asarray(
            model.transverse_coordinates_A
        ),
        "model_host_hard_core_pairs": np.asarray(model.host_hard_core_pairs),
        "model_json": np.asarray(
            _canonical_json_text(
                {
                    "deformation_parameter_count": int(
                        model.deformation_parameter_count
                    ),
                    "model_id": model.model_id,
                    "metadata": dict(model.metadata),
                }
            )
        ),
        "state_host_removal_indices": np.asarray(
            state.host_removal_indices
        ),
        "state_host_removal_fractions": np.asarray(
            state.host_removal_fractions
        ),
        "state_host_removal_active": np.asarray(state.host_removal_active),
        "state_extra_anchor_indices": np.asarray(state.extra_anchor_indices),
        "state_extra_position_offsets_A": np.asarray(
            state.extra_position_offsets_A
        ),
        "state_extra_scattering_equivalents": np.asarray(
            state.extra_scattering_equivalents
        ),
        "state_extra_active": np.asarray(state.extra_active),
        "state_host_displacement_controls": np.asarray(
            state.host_displacement_controls
        ),
        "rendered_potential": np.asarray(snapshot.rendered_potential),
        "snapshot_json": np.asarray(snapshot_json),
    }


def _archive_digest(payload: Mapping[str, Any]) -> str:
    return _hash_arrays_and_metadata(
        payload,
        {
            "contract": _SNAPSHOT_ARCHIVE_CONTRACT,
            "schema_version": _SNAPSHOT_ARCHIVE_SCHEMA_VERSION,
        },
    )


def save_atomistic_edit_snapshot_1d(
    path: str | Path,
    snapshot: AtomisticEditSnapshot1D,
) -> None:
    """Atomically save a complete non-pickled snapshot with SHA-256 evidence."""
    payload = _snapshot_archive_payload(snapshot)
    if set(payload) != _SNAPSHOT_ARCHIVE_FIELDS:
        raise RuntimeError("internal atomistic-edit archive schema is incomplete")
    archive_payload = {
        **payload,
        "archive_sha256": np.asarray(_archive_digest(payload)),
    }
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
            np.savez_compressed(handle, **archive_payload)
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


def _archive_scalar(
    payload: Mapping[str, np.ndarray],
    name: str,
    *,
    dtype: np.dtype[Any] | None = None,
    unicode: bool = False,
) -> Any:
    array = np.asarray(payload[name])
    if array.shape != ():
        raise ValueError(f"archive field {name} must be scalar")
    if dtype is not None and array.dtype != dtype:
        raise ValueError(
            f"archive field {name} must have dtype {np.dtype(dtype)}"
        )
    if unicode and array.dtype.kind != "U":
        raise ValueError(f"archive field {name} must be scalar Unicode")
    return array.item()


def _archive_typed_array(
    payload: Mapping[str, np.ndarray],
    name: str,
    *,
    dtype: Any,
) -> np.ndarray:
    array = np.asarray(payload[name])
    expected_dtype = np.dtype(dtype)
    if array.dtype != expected_dtype:
        raise ValueError(
            f"archive field {name} must have dtype {expected_dtype}"
        )
    return _readonly_array(array)


def _json_integer(name: str, value: Any, *, nonnegative: bool = True) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a JSON integer")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _json_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a JSON number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _json_boolean(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON Boolean")
    return value


def _json_metadata(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return _jsonable_mapping(value)


def _json_number_pair(name: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two JSON numbers")
    return (
        _json_number(f"{name}[0]", value[0]),
        _json_number(f"{name}[1]", value[1]),
    )


def _load_host_support_contract_1d(
    payload: Mapping[str, np.ndarray],
) -> LatticeSiteSupportContract1D:
    fields = _json_archive_mapping(
        payload["host_support_json"],
        name="host_support_json",
        expected_fields=_HOST_SUPPORT_JSON_FIELDS,
    )
    counts_value = fields["parameter_counts"]
    count_fields = {
        "target_vacancy_parameters",
        "nuisance_vacancy_parameters",
        "displacement_control_parameters",
        "removed_displacement_dof",
        "residual_displacement_control_dof",
        "registration_parameters",
        "total_specimen_parameters",
    }
    if not isinstance(counts_value, dict) or set(counts_value) != count_fields:
        raise ValueError(
            "host_support_json.parameter_counts has the wrong schema"
        )
    counts = LatticeSiteParameterCounts1D(
        **{
            name: _json_integer(
                f"host_support_json.parameter_counts.{name}",
                counts_value[name],
            )
            for name in sorted(count_fields)
        }
    )
    control_shape_value = fields["displacement_control_shape"]
    if not isinstance(control_shape_value, list):
        raise ValueError(
            "host_support_json.displacement_control_shape must be a JSON array"
        )
    control_shape = tuple(
        _json_integer(
            f"host_support_json.displacement_control_shape[{index}]", item
        )
        for index, item in enumerate(control_shape_value)
    )
    provenance = fields["fixed_material_provenance_id"]
    if provenance is not None and not isinstance(provenance, str):
        raise ValueError(
            "host_support_json.fixed_material_provenance_id must be a string "
            "or null"
        )
    contract = LatticeSiteSupportContract1D(
        schema_version=_json_integer(
            "host_support_json.schema_version", fields["schema_version"]
        ),
        classification_contract=_nonempty_identifier(
            "host_support_json.classification_contract",
            fields["classification_contract"],
        ),
        all_site_coordinates=_archive_typed_array(
            payload,
            "host_support_all_site_coordinates",
            dtype=np.float64,
        ),
        site_center_indices=_archive_typed_array(
            payload, "host_support_site_center_indices", dtype=np.int64
        ),
        site_patch_starts=_archive_typed_array(
            payload, "host_support_site_patch_starts", dtype=np.int64
        ),
        site_patch_shapes=_archive_typed_array(
            payload, "host_support_site_patch_shapes", dtype=np.int64
        ),
        target_pixel_mask=_archive_typed_array(
            payload, "host_support_target_pixel_mask", dtype=bool
        ),
        forward_pixel_mask=_archive_typed_array(
            payload, "host_support_forward_pixel_mask", dtype=bool
        ),
        target_center_mask=_archive_typed_array(
            payload, "host_support_target_center_mask", dtype=bool
        ),
        forward_relevant_mask=_archive_typed_array(
            payload, "host_support_forward_relevant_mask", dtype=bool
        ),
        site_role_codes=_archive_typed_array(
            payload, "host_support_site_role_codes", dtype=np.int8
        ),
        modeled_site_indices=_archive_typed_array(
            payload, "host_support_modeled_site_indices", dtype=np.int64
        ),
        target_influence_mask=_archive_typed_array(
            payload, "host_support_target_influence_mask", dtype=bool
        ),
        nuisance_influence_mask=_archive_typed_array(
            payload, "host_support_nuisance_influence_mask", dtype=bool
        ),
        exterior_policy=_nonempty_identifier(
            "host_support_json.exterior_policy", fields["exterior_policy"]
        ),
        excluded_probe_power=_json_number(
            "host_support_json.excluded_probe_power",
            fields["excluded_probe_power"],
        ),
        atomic_template_cutoff_A=_json_number(
            "host_support_json.atomic_template_cutoff_A",
            fields["atomic_template_cutoff_A"],
        ),
        maximum_displacement_A=_json_number(
            "host_support_json.maximum_displacement_A",
            fields["maximum_displacement_A"],
        ),
        fixed_material_provenance_id=provenance,
        displacement_control_shape=control_shape,
        removed_displacement_dof=_json_integer(
            "host_support_json.removed_displacement_dof",
            fields["removed_displacement_dof"],
        ),
        registration_parameter_count=_json_integer(
            "host_support_json.registration_parameter_count",
            fields["registration_parameter_count"],
        ),
        maximum_nuisance_sites=_json_integer(
            "host_support_json.maximum_nuisance_sites",
            fields["maximum_nuisance_sites"],
        ),
        maximum_specimen_parameters=_json_integer(
            "host_support_json.maximum_specimen_parameters",
            fields["maximum_specimen_parameters"],
        ),
        parameter_counts=counts,
        contract_id=_nonempty_identifier(
            "host_support_json.contract_id", fields["contract_id"]
        ),
    )
    return validate_lattice_site_support_contract_1d(contract, strict=True)


def _load_host_model_1d(
    payload: Mapping[str, np.ndarray],
    support: LatticeSiteSupportContract1D,
) -> LatticeSiteModel1D:
    metadata = _json_archive_mapping(
        payload["host_metadata_json"], name="host_metadata_json"
    )
    return LatticeSiteModel1D(
        reference_potential=jnp.asarray(payload["host_reference_potential"]),
        site_coordinates=jnp.asarray(payload["host_site_coordinates"]),
        site_patches=jnp.asarray(payload["host_site_patches"]),
        patch_starts=jnp.asarray(payload["host_patch_starts"]),
        control_coordinates_s=jnp.asarray(
            payload["host_control_coordinates_s"]
        ),
        control_coordinates_u=jnp.asarray(
            payload["host_control_coordinates_u"]
        ),
        axial_sampling=jnp.asarray(payload["host_axial_sampling"]),
        transverse_sampling=jnp.asarray(
            payload["host_transverse_sampling"]
        ),
        maximum_displacement=jnp.asarray(
            payload["host_maximum_displacement"]
        ),
        metadata=_jsonable_mapping(metadata),
        support_contract=support,
    )


def _load_discovery_support_1d(
    payload: Mapping[str, np.ndarray],
) -> AtomisticEditDiscoverySupport1D:
    fields = _json_archive_mapping(
        payload["discovery_json"],
        name="discovery_json",
        expected_fields=_DISCOVERY_JSON_FIELDS,
    )
    discovery = make_atomistic_edit_discovery_support_1d(
        payload["discovery_axial_coordinates_A"],
        payload["discovery_transverse_coordinates_A"],
        payload["discovery_target_mask"],
        payload["discovery_nuisance_mask"],
        surface_envelope_A=_json_number_pair(
            "discovery_json.surface_envelope_A",
            fields["surface_envelope_A"],
        ),
        geometry_source_id=_nonempty_identifier(
            "discovery_json.geometry_source_id",
            fields["geometry_source_id"],
        ),
        excluded_probe_power=_json_number(
            "discovery_json.excluded_probe_power",
            fields["excluded_probe_power"],
        ),
        metadata=_json_metadata(
            "discovery_json.metadata", fields["metadata"]
        ),
    )
    if discovery.contract_id != _nonempty_identifier(
        "discovery_json.contract_id", fields["contract_id"]
    ):
        raise ValueError("discovery contract_id does not match its fields")
    return discovery


def _load_addition_kernel_1d(
    payload: Mapping[str, np.ndarray],
) -> AtomisticEditKernel1D:
    fields = _json_archive_mapping(
        payload["addition_kernel_json"],
        name="addition_kernel_json",
        expected_fields=_KERNEL_JSON_FIELDS,
    )
    kernel = AtomisticEditKernel1D(
        unit_integrated_values=_archive_typed_array(
            payload, "addition_kernel_values", dtype=np.float64
        ),
        centre_index=_archive_typed_array(
            payload, "addition_kernel_centre_index", dtype=np.float64
        ),
        axial_sampling_A=_json_number(
            "addition_kernel_json.axial_sampling_A",
            fields["axial_sampling_A"],
        ),
        transverse_sampling_A=_json_number(
            "addition_kernel_json.transverse_sampling_A",
            fields["transverse_sampling_A"],
        ),
        host_equivalent_integrated_scattering=_json_number(
            "addition_kernel_json.host_equivalent_integrated_scattering",
            fields["host_equivalent_integrated_scattering"],
        ),
        parameterization_id=_nonempty_identifier(
            "addition_kernel_json.parameterization_id",
            fields["parameterization_id"],
        ),
        cutoff_A=_json_number(
            "addition_kernel_json.cutoff_A", fields["cutoff_A"]
        ),
        projection_width_A=_json_number(
            "addition_kernel_json.projection_width_A",
            fields["projection_width_A"],
        ),
        boundary_mass_fraction=_json_number(
            "addition_kernel_json.boundary_mass_fraction",
            fields["boundary_mass_fraction"],
        ),
        normalization_tolerance=_json_number(
            "addition_kernel_json.normalization_tolerance",
            fields["normalization_tolerance"],
        ),
        kernel_id=_nonempty_identifier(
            "addition_kernel_json.kernel_id", fields["kernel_id"]
        ),
        metadata=_json_metadata(
            "addition_kernel_json.metadata", fields["metadata"]
        ),
    )
    return _validated_kernel_identity(kernel)


def _load_edit_options_1d(
    fields: Mapping[str, Any],
    discovery: AtomisticEditDiscoverySupport1D,
) -> AtomisticEditOptions1D:
    path_value = fields["edit_penalty_path"]
    if not isinstance(path_value, list):
        raise ValueError("edit_support_json.edit_penalty_path must be a JSON array")
    penalty_path = tuple(
        _json_number(f"edit_support_json.edit_penalty_path[{index}]", item)
        for index, item in enumerate(path_value)
    )
    return AtomisticEditOptions1D(
        max_host_removals=_json_integer(
            "edit_support_json.maximum_host_removals",
            fields["maximum_host_removals"],
        ),
        max_extra_centres=_json_integer(
            "edit_support_json.maximum_extra_centres",
            fields["maximum_extra_centres"],
        ),
        max_scattering_equivalent_per_centre=_json_number(
            "edit_support_json.maximum_scattering_equivalent_per_centre",
            fields["maximum_scattering_equivalent_per_centre"],
        ),
        minimum_separation_A=_json_number(
            "edit_support_json.minimum_separation_A",
            fields["minimum_separation_A"],
        ),
        expected_rms_host_strain=_json_number(
            "edit_support_json.expected_rms_host_strain",
            fields["expected_rms_host_strain"],
        ),
        edit_penalty_path=penalty_path,
        discovery_support=discovery,
        enable_material_energy_envelope=_json_boolean(
            "edit_support_json.enable_material_energy_envelope",
            fields["enable_material_energy_envelope"],
        ),
    )


def _load_stored_edit_support_1d(
    payload: Mapping[str, np.ndarray],
    fields: Mapping[str, Any],
) -> AtomisticEditSupportContract1D:
    return AtomisticEditSupportContract1D(
        schema_version=_json_integer(
            "edit_support_json.schema_version", fields["schema_version"]
        ),
        host_support_contract_id=_nonempty_identifier(
            "edit_support_json.host_support_contract_id",
            fields["host_support_contract_id"],
        ),
        discovery_contract_id=_nonempty_identifier(
            "edit_support_json.discovery_contract_id",
            fields["discovery_contract_id"],
        ),
        kernel_id=_nonempty_identifier(
            "edit_support_json.kernel_id", fields["kernel_id"]
        ),
        target_discovery_mask=_archive_typed_array(
            payload, "edit_support_target_discovery_mask", dtype=bool
        ),
        nuisance_discovery_mask=_archive_typed_array(
            payload, "edit_support_nuisance_discovery_mask", dtype=bool
        ),
        addition_influence_mask=_archive_typed_array(
            payload, "edit_support_addition_influence_mask", dtype=bool
        ),
        total_influence_mask=_archive_typed_array(
            payload, "edit_support_total_influence_mask", dtype=bool
        ),
        maximum_host_removals=_json_integer(
            "edit_support_json.maximum_host_removals",
            fields["maximum_host_removals"],
        ),
        maximum_extra_centres=_json_integer(
            "edit_support_json.maximum_extra_centres",
            fields["maximum_extra_centres"],
        ),
        maximum_scattering_equivalent_per_centre=_json_number(
            "edit_support_json.maximum_scattering_equivalent_per_centre",
            fields["maximum_scattering_equivalent_per_centre"],
        ),
        minimum_separation_A=_json_number(
            "edit_support_json.minimum_separation_A",
            fields["minimum_separation_A"],
        ),
        expected_rms_host_strain=_json_number(
            "edit_support_json.expected_rms_host_strain",
            fields["expected_rms_host_strain"],
        ),
        spatial_dimension=_json_integer(
            "edit_support_json.spatial_dimension", fields["spatial_dimension"]
        ),
        deformation_parameter_count=_json_integer(
            "edit_support_json.deformation_parameter_count",
            fields["deformation_parameter_count"],
        ),
        elastic_model_id=_nonempty_identifier(
            "edit_support_json.elastic_model_id", fields["elastic_model_id"]
        ),
        hard_core_policy_id=_nonempty_identifier(
            "edit_support_json.hard_core_policy_id",
            fields["hard_core_policy_id"],
        ),
        contract_id=_nonempty_identifier(
            "edit_support_json.contract_id", fields["contract_id"]
        ),
    )


def _load_atomistic_edit_state_1d(
    payload: Mapping[str, np.ndarray],
) -> AtomisticEditState1D:
    return AtomisticEditState1D(
        host_removal_indices=jnp.asarray(
            payload["state_host_removal_indices"]
        ),
        host_removal_fractions=jnp.asarray(
            payload["state_host_removal_fractions"]
        ),
        host_removal_active=jnp.asarray(payload["state_host_removal_active"]),
        extra_anchor_indices=jnp.asarray(payload["state_extra_anchor_indices"]),
        extra_position_offsets_A=jnp.asarray(
            payload["state_extra_position_offsets_A"]
        ),
        extra_scattering_equivalents=jnp.asarray(
            payload["state_extra_scattering_equivalents"]
        ),
        extra_active=jnp.asarray(payload["state_extra_active"]),
        host_displacement_controls=jnp.asarray(
            payload["state_host_displacement_controls"]
        ),
    )


def load_atomistic_edit_snapshot_1d(
    path: str | Path,
) -> AtomisticEditSnapshot1D:
    """Load and independently revalidate an authenticated non-pickled snapshot."""
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            expected_fields = _SNAPSHOT_ARCHIVE_FIELDS | {"archive_sha256"}
            actual_fields = set(archive.files)
            if actual_fields != expected_fields:
                missing = sorted(expected_fields - actual_fields)
                extra = sorted(actual_fields - expected_fields)
                raise ValueError(
                    "atomistic-edit archive fields differ from schema: "
                    f"missing={missing}, extra={extra}"
                )
            payload = {
                name: np.array(archive[name], copy=True, order="C")
                for name in _SNAPSHOT_ARCHIVE_FIELDS
            }
            stored_archive_digest = np.array(
                archive["archive_sha256"], copy=True
            )
    except (OSError, EOFError, KeyError) as error:
        raise ValueError("atomistic-edit snapshot archive is unreadable") from error

    schema_version = _archive_scalar(
        payload, "schema_version", dtype=np.dtype(np.int64)
    )
    if schema_version != _SNAPSHOT_ARCHIVE_SCHEMA_VERSION:
        raise ValueError("unsupported atomistic-edit snapshot schema version")
    archive_contract = _archive_scalar(
        payload, "archive_contract", unicode=True
    )
    if archive_contract != _SNAPSHOT_ARCHIVE_CONTRACT:
        raise ValueError("unsupported atomistic-edit archive contract")
    digest_array = np.asarray(stored_archive_digest)
    if digest_array.shape != () or digest_array.dtype.kind != "U":
        raise ValueError("archive_sha256 must be a scalar Unicode digest")
    digest = str(digest_array.item())
    if (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or digest != _archive_digest(payload)
    ):
        raise ValueError("atomistic-edit archive SHA-256 verification failed")

    host_support = _load_host_support_contract_1d(payload)
    host_model = _load_host_model_1d(payload, host_support)
    discovery = _load_discovery_support_1d(payload)
    kernel = _load_addition_kernel_1d(payload)
    edit_fields = _json_archive_mapping(
        payload["edit_support_json"],
        name="edit_support_json",
        expected_fields=_EDIT_SUPPORT_JSON_FIELDS,
    )
    options = _load_edit_options_1d(edit_fields, discovery)
    model_fields = _json_archive_mapping(
        payload["model_json"],
        name="model_json",
        expected_fields=_MODEL_JSON_FIELDS,
    )
    model = make_atomistic_edit_model_1d(
        host_model,
        payload["model_axial_coordinates_A"],
        payload["model_transverse_coordinates_A"],
        kernel,
        options,
        deformation_parameter_count=_json_integer(
            "model_json.deformation_parameter_count",
            model_fields["deformation_parameter_count"],
        ),
        metadata=_json_metadata("model_json.metadata", model_fields["metadata"]),
    )
    stored_support = _load_stored_edit_support_1d(payload, edit_fields)
    _require_same_edit_support(model.support_contract, stored_support)
    _require_exact_array(
        "model axial_coordinates_A",
        model.axial_coordinates_A,
        payload["model_axial_coordinates_A"],
    )
    _require_exact_array(
        "model transverse_coordinates_A",
        model.transverse_coordinates_A,
        payload["model_transverse_coordinates_A"],
    )
    _require_exact_array(
        "model host_hard_core_pairs",
        model.host_hard_core_pairs,
        payload["model_host_hard_core_pairs"],
    )
    if model.model_id != _nonempty_identifier(
        "model_json.model_id", model_fields["model_id"]
    ):
        raise ValueError("model_id does not match the archived model")

    state = _load_atomistic_edit_state_1d(payload)
    validate_atomistic_edit_state_1d(model, state)
    snapshot_fields = _json_archive_mapping(
        payload["snapshot_json"],
        name="snapshot_json",
        expected_fields=_SNAPSHOT_JSON_FIELDS,
    )
    prior_fields = snapshot_fields["prior_components"]
    if not isinstance(prior_fields, dict) or set(prior_fields) != _PRIOR_JSON_FIELDS:
        raise ValueError("snapshot prior_components has the wrong schema")
    prior = AtomisticEditPriorComponents1D(
        **{
            name: _json_number(
                f"snapshot_json.prior_components.{name}", prior_fields[name]
            )
            for name in sorted(_PRIOR_JSON_FIELDS)
        }
    )
    snapshot = AtomisticEditSnapshot1D(
        model=model,
        state=state,
        rendered_potential=jnp.asarray(payload["rendered_potential"]),
        active_parameter_count=_json_integer(
            "snapshot_json.active_parameter_count",
            snapshot_fields["active_parameter_count"],
        ),
        selected_edit_penalty=_json_number(
            "snapshot_json.selected_edit_penalty",
            snapshot_fields["selected_edit_penalty"],
        ),
        edit_penalty_rule_id=_nonempty_identifier(
            "snapshot_json.edit_penalty_rule_id",
            snapshot_fields["edit_penalty_rule_id"],
        ),
        data_objective_value=_json_number(
            "snapshot_json.data_objective_value",
            snapshot_fields["data_objective_value"],
        ),
        data_objective_id=_nonempty_identifier(
            "snapshot_json.data_objective_id",
            snapshot_fields["data_objective_id"],
        ),
        prior_components=prior,
        total_objective_value=_json_number(
            "snapshot_json.total_objective_value",
            snapshot_fields["total_objective_value"],
        ),
        kkt_status=_nonempty_identifier(
            "snapshot_json.kkt_status", snapshot_fields["kkt_status"]
        ),
        capacity_status=_nonempty_identifier(
            "snapshot_json.capacity_status", snapshot_fields["capacity_status"]
        ),
        converged=_json_boolean(
            "snapshot_json.converged", snapshot_fields["converged"]
        ),
        metadata=_json_metadata(
            "snapshot_json.metadata", snapshot_fields["metadata"]
        ),
        snapshot_id=_nonempty_identifier(
            "snapshot_json.snapshot_id", snapshot_fields["snapshot_id"]
        ),
    )
    return validate_atomistic_edit_snapshot_1d(snapshot)
