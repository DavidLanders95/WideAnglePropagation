"""Truth-isolated complete-slab alignment for 1D silicon ptychography.

The module freezes a geometry-only coarse catalog and deterministic refinement
policy before copying diffraction values.  It rebuilds the complete finite Si
slab for every candidate, uses only a geometry-stratified training screen for
ranking, reserves validation for the shortlist, and reports validation
ambiguity without consulting audit or synthetic-truth fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import os
import operator
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import qmc
from scipy.ndimage import shift as shift_image

from .ptychography_1d import (
    ConvergenceOptions1D,
    GlancingScan1D,
    LatticeOptimizationOptions1D,
    LatticeSiteModel1D,
    LatticeSiteReconstruction1D,
    PreparedLatticeSiteReconstruction1D,
    PtychographyMeasurement1D,
    PtychographyObjective1D,
    normalized_amplitude_loss_1d,
    prepare_lattice_site_reconstruction_1d,
    render_lattice_site_potential_1d,
    simulate_glancing_scan_1d,
    run_prepared_lattice_site_reconstruction_1d,
)


__all__ = [
    "AlignmentCandidateScore1D",
    "AlignmentInitializationOptions1D",
    "AlignmentSelectionData1D",
    "AlignmentSelectionSummary1D",
    "SiliconAlignmentCandidate1D",
    "SiliconAlignmentModel1D",
    "SiliconAlignmentPrior1D",
    "SiliconAlignmentForwardProblem1D",
    "SiliconAlignmentInitialization1D",
    "alignment_candidate_catalog_id_1d",
    "build_alignment_selection_data_1d",
    "canonical_axial_phase_fraction_1d",
    "generate_silicon_alignment_candidates_1d",
    "geometry_stratified_training_subset_1d",
    "initialize_silicon_alignment_1d",
    "make_silicon_alignment_forward_problem_1d",
    "make_silicon_alignment_prior_1d",
    "load_silicon_alignment_initialization_1d",
    "prepare_aligned_lattice_site_reconstruction_1d",
    "rebuild_silicon_alignment_candidate_1d",
    "reconstruct_aligned_lattice_site_1d",
    "refine_silicon_alignment_candidates_1d",
    "save_silicon_alignment_initialization_1d",
    "select_alignment_candidate_1d",
]


Array = Any
_CANDIDATE_CONTRACT = "silicon_alignment_candidate:v1"
_CATALOG_CONTRACT = "silicon_alignment_candidate_catalog:v1"
_SELECTION_DATA_CONTRACT = "alignment_training_validation_data:v1"
_SELECTION_CONTRACT = "paired_validation_equivalence_selection:v1"


@dataclass(frozen=True)
class SiliconAlignmentCandidate1D:
    """One truth-free global lattice candidate.

    Axial phase is dimensionless and canonical modulo one projected repeat.
    Rotation and lattice scale describe a future complete-slab rebuild; they
    must never be applied only to the active sites of an existing reference.
    """

    termination_id: str
    axial_phase_fraction: float
    in_plane_rotation_rad: float
    lattice_scale: float
    refinement_level: int
    parent_candidate_id: str | None
    candidate_id: str


@dataclass(frozen=True)
class AlignmentInitializationOptions1D:
    """Geometry-only catalog and validation-equivalence policy."""

    candidates_per_termination: int = 16
    training_screen_scan_count: int = 32
    coarse_shortlist_size: int = 8
    validation_shortlist_size: int = 4
    refinement_rounds: int = 1
    fine_phase_step_fraction: float = 1.0 / 32.0
    fine_rotation_step_rad: float = np.deg2rad(0.025)
    fine_log_scale_step: float = 2.5e-4
    lattice_scale_bounds: tuple[float, float] = (0.98, 1.02)
    in_plane_rotation_bounds_rad: tuple[float, float] = (
        -np.deg2rad(0.25),
        np.deg2rad(0.25),
    )
    validation_absolute_band: float = 1e-10
    validation_relative_band: float = 1e-3
    validation_equivalence_z: float = 1.96
    seed: int = 0


@dataclass(frozen=True)
class AlignmentSelectionData1D:
    """Copied observations available to alignment selection.

    Rows contain the geometry-stratified training screen followed by complete
    validation.  Audit and guard observations are never stored in this type.
    Source indices are retained solely to map the copied rows back to geometry.
    """

    intensities: Array
    detector_valid_mask: Array | None
    window_starts: Array
    scan_coordinates: Array
    detector_angles: Array
    source_scan_indices: Array
    training_source_indices: Array
    validation_source_indices: Array
    training_local_indices: Array
    validation_local_indices: Array
    selection_data_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignmentCandidateScore1D:
    """Training-screen and per-scan validation scores for one candidate."""

    candidate: SiliconAlignmentCandidate1D
    training_screen_loss: float
    validation_loss: float
    validation_loss_per_scan: Array
    candidate_model_id: str


@dataclass(frozen=True)
class AlignmentSelectionSummary1D:
    """Validation-selected representative and its equivalent alternatives."""

    minimum_loss_candidate_id: str
    selected_candidate_id: str
    equivalent_candidate_ids: tuple[str, ...]
    unique_selection: bool
    candidate_catalog_id: str
    selection_data_id: str
    alignment_selection_id: str
    structurally_trusted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SiliconAlignmentPrior1D:
    """Truth-free material/grid inputs needed to rebuild complete Si slabs."""

    axial_coordinates: Array
    transverse_coordinates: Array
    reconstruction_mask: Array
    projected_si_template: Array
    template_half_shape: tuple[int, int]
    projected_basis_fractional_su: Array
    nominal_lattice_A: float
    slab_depth_A: float
    maximum_displacement_A: float
    displacement_control_spacing_s_A: float
    displacement_control_spacing_u_A: float
    termination_ids: tuple[str, ...]
    termination_offsets_fractional_u: Array
    prior_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SiliconAlignmentModel1D:
    """One complete-slab candidate and its active-site influence support."""

    candidate: SiliconAlignmentCandidate1D
    lattice_model: LatticeSiteModel1D
    all_site_coordinates: Array
    variable_site_coordinates: Array
    lattice_influence_mask: Array
    prior_id: str
    candidate_model_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SiliconAlignmentForwardProblem1D:
    """Truth-free known forward geometry used during alignment selection."""

    prior: SiliconAlignmentPrior1D
    input_probes: Array
    propagation_kernel: Array
    window_starts: Array
    window_length: int
    scan_coordinates: Array
    detector_angles: Array
    slice_thickness_A: float
    energy_eV: float
    training_indices: Array
    validation_indices: Array
    audit_indices: Array
    guard_indices: Array
    alignment_problem_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SiliconAlignmentInitialization1D:
    """Validation-selected complete-slab initialization and ambiguity set."""

    selected_model: SiliconAlignmentModel1D
    candidate_scores: tuple[AlignmentCandidateScore1D, ...]
    selection_summary: AlignmentSelectionSummary1D
    training_screen_indices: Array
    validation_indices: Array
    audit_indices: Array
    guard_indices: Array
    alignment_problem_id: str
    candidate_catalog: tuple[SiliconAlignmentCandidate1D, ...] = ()
    structurally_trusted: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(payload))).hexdigest()


def _digest_arrays(
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        if array.dtype.hasobject:
            raise TypeError(f"cannot hash object-valued array {name!r}")
        header = _canonical_json(
            {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)}
        )
        payload = np.ascontiguousarray(array).tobytes(order="C")
        for chunk in (header, payload):
            digest.update(len(chunk).to_bytes(8, "big"))
            digest.update(chunk)
    encoded_metadata = _canonical_json(dict(metadata))
    digest.update(len(encoded_metadata).to_bytes(8, "big"))
    digest.update(encoded_metadata)
    return digest.hexdigest()


def _readonly_copy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        resolved = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if resolved < 1:
        raise ValueError(f"{name} must be positive")
    return resolved


def _finite_scalar(name: str, value: Any) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or np.iscomplexobj(array)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"{name} must be a real scalar")
    resolved = float(array)
    if not np.isfinite(resolved):
        raise ValueError(f"{name} must be finite")
    return resolved


def _validated_options(
    options: AlignmentInitializationOptions1D | None,
) -> AlignmentInitializationOptions1D:
    options = AlignmentInitializationOptions1D() if options is None else options
    if not isinstance(options, AlignmentInitializationOptions1D):
        raise TypeError(
            "options must be an AlignmentInitializationOptions1D instance or None"
        )
    count = _positive_integer(
        "options.candidates_per_termination",
        options.candidates_per_termination,
    )
    if count & (count - 1):
        raise ValueError("candidates_per_termination must be a power of two")
    _positive_integer(
        "options.training_screen_scan_count",
        options.training_screen_scan_count,
    )
    _positive_integer(
        "options.coarse_shortlist_size",
        options.coarse_shortlist_size,
    )
    _positive_integer(
        "options.validation_shortlist_size",
        options.validation_shortlist_size,
    )
    if isinstance(options.refinement_rounds, (bool, np.bool_)):
        raise TypeError("options.refinement_rounds must be an integer")
    try:
        refinement_rounds = operator.index(options.refinement_rounds)
    except TypeError as error:
        raise TypeError("options.refinement_rounds must be an integer") from error
    if refinement_rounds < 0:
        raise ValueError("options.refinement_rounds must be non-negative")
    scale_bounds = np.asarray(options.lattice_scale_bounds, dtype=float)
    rotation_bounds = np.asarray(
        options.in_plane_rotation_bounds_rad,
        dtype=float,
    )
    for name, bounds in (
        ("lattice_scale_bounds", scale_bounds),
        ("in_plane_rotation_bounds_rad", rotation_bounds),
    ):
        if bounds.shape != (2,) or np.any(~np.isfinite(bounds)):
            raise ValueError(f"options.{name} must contain two finite values")
        if bounds[0] > bounds[1]:
            raise ValueError(f"options.{name} must be ordered")
    if scale_bounds[0] <= 0.0:
        raise ValueError("lattice_scale_bounds must be positive")
    for name, value in (
        ("validation_absolute_band", options.validation_absolute_band),
        ("validation_relative_band", options.validation_relative_band),
        ("validation_equivalence_z", options.validation_equivalence_z),
        ("fine_phase_step_fraction", options.fine_phase_step_fraction),
        ("fine_rotation_step_rad", options.fine_rotation_step_rad),
        ("fine_log_scale_step", options.fine_log_scale_step),
    ):
        if _finite_scalar(f"options.{name}", value) < 0.0:
            raise ValueError(f"options.{name} must be non-negative")
    if isinstance(options.seed, (bool, np.bool_)):
        raise TypeError("options.seed must be an integer")
    try:
        seed = operator.index(options.seed)
    except TypeError as error:
        raise TypeError("options.seed must be an integer") from error
    if seed < 0:
        raise ValueError("options.seed must be non-negative")
    return options


def _alignment_options_payload(
    options: AlignmentInitializationOptions1D,
) -> dict[str, Any]:
    options = _validated_options(options)
    return {
        "candidates_per_termination": options.candidates_per_termination,
        "training_screen_scan_count": options.training_screen_scan_count,
        "coarse_shortlist_size": options.coarse_shortlist_size,
        "validation_shortlist_size": options.validation_shortlist_size,
        "refinement_rounds": options.refinement_rounds,
        "fine_phase_step_fraction": options.fine_phase_step_fraction,
        "fine_rotation_step_rad": options.fine_rotation_step_rad,
        "fine_log_scale_step": options.fine_log_scale_step,
        "lattice_scale_bounds": list(options.lattice_scale_bounds),
        "in_plane_rotation_bounds_rad": list(
            options.in_plane_rotation_bounds_rad
        ),
        "validation_absolute_band": options.validation_absolute_band,
        "validation_relative_band": options.validation_relative_band,
        "validation_equivalence_z": options.validation_equivalence_z,
        "seed": int(options.seed),
    }


def _alignment_options_from_payload(
    payload: Any,
) -> AlignmentInitializationOptions1D:
    if not isinstance(payload, Mapping):
        raise ValueError("archived alignment options must be a JSON object")
    expected = set(_alignment_options_payload(AlignmentInitializationOptions1D()))
    if set(payload) != expected:
        raise ValueError("archived alignment options have an invalid field set")
    try:
        options = AlignmentInitializationOptions1D(
            candidates_per_termination=payload["candidates_per_termination"],
            training_screen_scan_count=payload["training_screen_scan_count"],
            coarse_shortlist_size=payload["coarse_shortlist_size"],
            validation_shortlist_size=payload["validation_shortlist_size"],
            refinement_rounds=payload["refinement_rounds"],
            fine_phase_step_fraction=payload["fine_phase_step_fraction"],
            fine_rotation_step_rad=payload["fine_rotation_step_rad"],
            fine_log_scale_step=payload["fine_log_scale_step"],
            lattice_scale_bounds=tuple(payload["lattice_scale_bounds"]),
            in_plane_rotation_bounds_rad=tuple(
                payload["in_plane_rotation_bounds_rad"]
            ),
            validation_absolute_band=payload["validation_absolute_band"],
            validation_relative_band=payload["validation_relative_band"],
            validation_equivalence_z=payload["validation_equivalence_z"],
            seed=payload["seed"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("archived alignment options are invalid") from error
    return _validated_options(options)


def canonical_axial_phase_fraction_1d(value: Any) -> float:
    """Return a finite axial phase in the canonical half-open interval [0, 1)."""
    phase = _finite_scalar("axial_phase_fraction", value)
    canonical = phase % 1.0
    return 0.0 if canonical == 1.0 else canonical


def _candidate_payload(
    *,
    termination_id: str,
    axial_phase_fraction: float,
    in_plane_rotation_rad: float,
    lattice_scale: float,
    refinement_level: int,
    parent_candidate_id: str | None,
) -> dict[str, Any]:
    return {
        "contract": _CANDIDATE_CONTRACT,
        "termination_id": termination_id,
        "axial_phase_fraction": axial_phase_fraction,
        "in_plane_rotation_rad": in_plane_rotation_rad,
        "lattice_scale": lattice_scale,
        "refinement_level": refinement_level,
        "parent_candidate_id": parent_candidate_id,
    }


def _validated_candidate(
    candidate: SiliconAlignmentCandidate1D,
) -> SiliconAlignmentCandidate1D:
    if not isinstance(candidate, SiliconAlignmentCandidate1D):
        raise TypeError("candidate must be a SiliconAlignmentCandidate1D")
    if not isinstance(candidate.termination_id, str) or not (
        candidate.termination_id.strip()
    ):
        raise ValueError("candidate.termination_id must be non-empty")
    phase = canonical_axial_phase_fraction_1d(candidate.axial_phase_fraction)
    if phase != candidate.axial_phase_fraction:
        raise ValueError("candidate axial phase must already be canonical in [0, 1)")
    rotation = _finite_scalar(
        "candidate.in_plane_rotation_rad",
        candidate.in_plane_rotation_rad,
    )
    scale = _finite_scalar("candidate.lattice_scale", candidate.lattice_scale)
    if scale <= 0.0:
        raise ValueError("candidate.lattice_scale must be positive")
    if isinstance(candidate.refinement_level, (bool, np.bool_)):
        raise TypeError("candidate.refinement_level must be an integer")
    try:
        refinement_level = operator.index(candidate.refinement_level)
    except TypeError as error:
        raise TypeError("candidate.refinement_level must be an integer") from error
    if refinement_level < 0:
        raise ValueError("candidate.refinement_level must be non-negative")
    if candidate.parent_candidate_id is not None and (
        not isinstance(candidate.parent_candidate_id, str)
        or not candidate.parent_candidate_id.strip()
    ):
        raise ValueError("candidate.parent_candidate_id must be non-empty or None")
    expected = _digest_payload(
        _candidate_payload(
            termination_id=candidate.termination_id,
            axial_phase_fraction=phase,
            in_plane_rotation_rad=rotation,
            lattice_scale=scale,
            refinement_level=refinement_level,
            parent_candidate_id=candidate.parent_candidate_id,
        )
    )
    if candidate.candidate_id != expected:
        raise ValueError("candidate_id does not match the canonical candidate")
    return candidate


def _make_alignment_candidate(
    *,
    termination_id: str,
    axial_phase_fraction: float,
    in_plane_rotation_rad: float,
    lattice_scale: float,
    refinement_level: int,
    parent_candidate_id: str | None,
) -> SiliconAlignmentCandidate1D:
    phase = canonical_axial_phase_fraction_1d(axial_phase_fraction)
    payload = _candidate_payload(
        termination_id=termination_id,
        axial_phase_fraction=phase,
        in_plane_rotation_rad=float(in_plane_rotation_rad),
        lattice_scale=float(lattice_scale),
        refinement_level=int(refinement_level),
        parent_candidate_id=parent_candidate_id,
    )
    return _validated_candidate(
        SiliconAlignmentCandidate1D(
            termination_id=termination_id,
            axial_phase_fraction=phase,
            in_plane_rotation_rad=float(in_plane_rotation_rad),
            lattice_scale=float(lattice_scale),
            refinement_level=int(refinement_level),
            parent_candidate_id=parent_candidate_id,
            candidate_id=_digest_payload(payload),
        )
    )


def generate_silicon_alignment_candidates_1d(
    termination_ids: Sequence[str] = ("si_termination_0", "si_termination_1"),
    *,
    options: AlignmentInitializationOptions1D | None = None,
) -> tuple[SiliconAlignmentCandidate1D, ...]:
    """Generate a deterministic, termination-balanced Sobol catalog."""
    options = _validated_options(options)
    terminations = tuple(termination_ids)
    if not terminations or any(
        not isinstance(value, str) or not value.strip() for value in terminations
    ):
        raise ValueError("termination_ids must contain non-empty strings")
    if len(set(terminations)) != len(terminations):
        raise ValueError("termination_ids must be unique")

    exponent = int(np.log2(options.candidates_per_termination))
    unit_points = qmc.Sobol(
        d=3,
        scramble=True,
        seed=int(options.seed),
    ).random_base2(exponent)
    scale_bounds = np.asarray(options.lattice_scale_bounds, dtype=float)
    log_scale = np.log(scale_bounds)
    rotation_bounds = np.asarray(options.in_plane_rotation_bounds_rad, dtype=float)
    candidates: list[SiliconAlignmentCandidate1D] = []
    for point_index, point in enumerate(unit_points):
        phase = canonical_axial_phase_fraction_1d(point[0])
        rotation = float(
            rotation_bounds[0]
            + point[1] * (rotation_bounds[1] - rotation_bounds[0])
        )
        scale = float(np.exp(log_scale[0] + point[2] * (log_scale[1] - log_scale[0])))
        for termination_id in terminations:
            candidate = _make_alignment_candidate(
                termination_id=termination_id,
                axial_phase_fraction=phase,
                in_plane_rotation_rad=rotation,
                lattice_scale=scale,
                refinement_level=0,
                parent_candidate_id=None,
            )
            candidates.append(candidate)
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise RuntimeError("Sobol alignment catalog contains duplicate candidates")
    return tuple(candidates)


def refine_silicon_alignment_candidates_1d(
    parents: Sequence[SiliconAlignmentCandidate1D],
    *,
    options: AlignmentInitializationOptions1D | None = None,
) -> tuple[SiliconAlignmentCandidate1D, ...]:
    """Return a deterministic bounded one-coordinate-at-a-time fine stencil."""
    options = _validated_options(options)
    resolved_parents = tuple(_validated_candidate(parent) for parent in parents)
    if not resolved_parents:
        raise ValueError("parents must not be empty")
    rotation_min, rotation_max = options.in_plane_rotation_bounds_rad
    log_scale_min, log_scale_max = np.log(options.lattice_scale_bounds)
    offsets = (
        (options.fine_phase_step_fraction, 0.0, 0.0),
        (-options.fine_phase_step_fraction, 0.0, 0.0),
        (0.0, options.fine_rotation_step_rad, 0.0),
        (0.0, -options.fine_rotation_step_rad, 0.0),
        (0.0, 0.0, options.fine_log_scale_step),
        (0.0, 0.0, -options.fine_log_scale_step),
    )
    refined: list[SiliconAlignmentCandidate1D] = []
    seen_physical = {
        (
            parent.termination_id,
            round(parent.axial_phase_fraction, 15),
            round(parent.in_plane_rotation_rad, 15),
            round(parent.lattice_scale, 15),
        )
        for parent in resolved_parents
    }
    for parent in resolved_parents:
        for phase_offset, rotation_offset, log_scale_offset in offsets:
            rotation = float(
                np.clip(
                    parent.in_plane_rotation_rad + rotation_offset,
                    rotation_min,
                    rotation_max,
                )
            )
            log_scale = float(
                np.clip(
                    np.log(parent.lattice_scale) + log_scale_offset,
                    log_scale_min,
                    log_scale_max,
                )
            )
            candidate = _make_alignment_candidate(
                termination_id=parent.termination_id,
                axial_phase_fraction=(
                    parent.axial_phase_fraction + phase_offset
                ),
                in_plane_rotation_rad=rotation,
                lattice_scale=float(np.exp(log_scale)),
                refinement_level=parent.refinement_level + 1,
                parent_candidate_id=parent.candidate_id,
            )
            physical_key = (
                candidate.termination_id,
                round(candidate.axial_phase_fraction, 15),
                round(candidate.in_plane_rotation_rad, 15),
                round(candidate.lattice_scale, 15),
            )
            if physical_key not in seen_physical:
                seen_physical.add(physical_key)
                refined.append(candidate)
    return tuple(refined)


def alignment_candidate_catalog_id_1d(
    candidates: Sequence[SiliconAlignmentCandidate1D],
    *,
    options: AlignmentInitializationOptions1D | None = None,
) -> str:
    """Hash an ordered candidate catalog and the policy that generated it."""
    options = _validated_options(options)
    resolved = tuple(_validated_candidate(candidate) for candidate in candidates)
    if not resolved:
        raise ValueError("candidates must not be empty")
    return _digest_payload(
        {
            "contract": _CATALOG_CONTRACT,
            "candidate_ids": [candidate.candidate_id for candidate in resolved],
            "options": _alignment_options_payload(options),
        }
    )


def _uniform_spacing(name: str, coordinates: np.ndarray) -> float:
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError(f"{name} must contain at least two coordinates")
    if np.iscomplexobj(coordinates) or np.any(~np.isfinite(coordinates)):
        raise ValueError(f"{name} must be finite and real")
    differences = np.diff(coordinates.astype(float, copy=False))
    if np.any(differences <= 0.0) or not np.allclose(
        differences,
        differences[0],
        rtol=1e-8,
        atol=1e-12,
    ):
        raise ValueError(f"{name} must be uniformly increasing")
    return float(differences[0])


def _silicon_alignment_prior_id(
    *,
    axial_coordinates: Any,
    transverse_coordinates: Any,
    reconstruction_mask: Any,
    projected_si_template: Any,
    projected_basis_fractional_su: Any,
    termination_offsets_fractional_u: Any,
    template_half_shape: tuple[int, int],
    nominal_lattice_A: float,
    slab_depth_A: float,
    maximum_displacement_A: float,
    displacement_control_spacing_s_A: float,
    displacement_control_spacing_u_A: float,
    termination_ids: tuple[str, ...],
) -> str:
    return _digest_arrays(
        {
            "axial_coordinates": axial_coordinates,
            "transverse_coordinates": transverse_coordinates,
            "reconstruction_mask": reconstruction_mask,
            "projected_si_template": projected_si_template,
            "projected_basis_fractional_su": projected_basis_fractional_su,
            "termination_offsets_fractional_u": (
                termination_offsets_fractional_u
            ),
        },
        {
            "contract": "silicon_alignment_prior:v1",
            "template_half_shape": list(template_half_shape),
            "nominal_lattice_A": nominal_lattice_A,
            "slab_depth_A": slab_depth_A,
            "maximum_displacement_A": maximum_displacement_A,
            "displacement_control_spacing_s_A": (
                displacement_control_spacing_s_A
            ),
            "displacement_control_spacing_u_A": (
                displacement_control_spacing_u_A
            ),
            "termination_ids": list(termination_ids),
        },
    )


def make_silicon_alignment_prior_1d(
    *,
    axial_coordinates: Any,
    transverse_coordinates: Any,
    reconstruction_mask: Any,
    projected_si_template: Any,
    template_half_shape: tuple[int, int],
    projected_basis_fractional_su: Any,
    nominal_lattice_A: Any,
    slab_depth_A: Any,
    maximum_displacement_A: Any,
    displacement_control_spacing_s_A: Any,
    displacement_control_spacing_u_A: Any,
    termination_ids: Sequence[str] = ("si_termination_0", "si_termination_1"),
    termination_offsets_fractional_u: Any = (0.0, 0.25),
    metadata: Mapping[str, Any] | None = None,
) -> SiliconAlignmentPrior1D:
    """Build a hash-bound prior containing no diffraction or defect truth."""
    s_coordinates = np.asarray(axial_coordinates)
    u_coordinates = np.asarray(transverse_coordinates)
    _uniform_spacing("axial_coordinates", s_coordinates)
    _uniform_spacing("transverse_coordinates", u_coordinates)
    mask = np.asarray(reconstruction_mask)
    if mask.dtype != np.bool_ or mask.shape != (
        s_coordinates.size,
        u_coordinates.size,
    ):
        raise ValueError("reconstruction_mask must be a matching Boolean array")
    if not np.any(mask):
        raise ValueError("reconstruction_mask must contain a mutable pixel")
    template = np.asarray(projected_si_template)
    if (
        template.ndim != 2
        or not np.issubdtype(template.dtype, np.floating)
        or np.iscomplexobj(template)
        or np.any(~np.isfinite(template))
    ):
        raise ValueError(
            "projected_si_template must be a finite floating-point 2D array"
        )
    if (
        not isinstance(template_half_shape, tuple)
        or len(template_half_shape) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
            for value in template_half_shape
        )
    ):
        raise ValueError("template_half_shape must contain two positive integers")
    half_shape = tuple(int(value) for value in template_half_shape)
    if template.shape != (2 * half_shape[0] + 1, 2 * half_shape[1] + 1):
        raise ValueError("projected_si_template shape and half-shape disagree")
    basis = np.asarray(projected_basis_fractional_su)
    if (
        basis.ndim != 2
        or basis.shape[1] != 2
        or not basis.size
        or np.iscomplexobj(basis)
        or np.any(~np.isfinite(basis))
    ):
        raise ValueError("projected_basis_fractional_su must have shape (n, 2)")
    terminations = tuple(termination_ids)
    offsets = np.asarray(termination_offsets_fractional_u)
    if (
        not terminations
        or len(set(terminations)) != len(terminations)
        or any(not isinstance(value, str) or not value.strip() for value in terminations)
    ):
        raise ValueError("termination_ids must contain unique non-empty strings")
    if offsets.shape != (len(terminations),) or np.iscomplexobj(offsets) or np.any(
        ~np.isfinite(offsets)
    ):
        raise ValueError("termination offsets must match termination_ids")
    physical_values = {
        "nominal_lattice_A": _finite_scalar(
            "nominal_lattice_A", nominal_lattice_A
        ),
        "slab_depth_A": _finite_scalar("slab_depth_A", slab_depth_A),
        "maximum_displacement_A": _finite_scalar(
            "maximum_displacement_A", maximum_displacement_A
        ),
        "displacement_control_spacing_s_A": _finite_scalar(
            "displacement_control_spacing_s_A",
            displacement_control_spacing_s_A,
        ),
        "displacement_control_spacing_u_A": _finite_scalar(
            "displacement_control_spacing_u_A",
            displacement_control_spacing_u_A,
        ),
    }
    positive_physical_values = (
        value
        for key, value in physical_values.items()
        if key != "maximum_displacement_A"
    )
    if any(value <= 0.0 for value in positive_physical_values):
        raise ValueError("lattice, slab, and control spacings must be positive")
    if physical_values["maximum_displacement_A"] < 0.0:
        raise ValueError("maximum_displacement_A must be non-negative")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    prior_id = _silicon_alignment_prior_id(
        axial_coordinates=s_coordinates,
        transverse_coordinates=u_coordinates,
        reconstruction_mask=mask,
        projected_si_template=template,
        projected_basis_fractional_su=basis,
        termination_offsets_fractional_u=offsets,
        template_half_shape=half_shape,
        termination_ids=terminations,
        **physical_values,
    )
    return SiliconAlignmentPrior1D(
        axial_coordinates=_readonly_copy(s_coordinates),
        transverse_coordinates=_readonly_copy(u_coordinates),
        reconstruction_mask=_readonly_copy(mask, dtype=bool),
        projected_si_template=_readonly_copy(template),
        template_half_shape=half_shape,
        projected_basis_fractional_su=_readonly_copy(basis),
        nominal_lattice_A=physical_values["nominal_lattice_A"],
        slab_depth_A=physical_values["slab_depth_A"],
        maximum_displacement_A=physical_values["maximum_displacement_A"],
        displacement_control_spacing_s_A=physical_values[
            "displacement_control_spacing_s_A"
        ],
        displacement_control_spacing_u_A=physical_values[
            "displacement_control_spacing_u_A"
        ],
        termination_ids=terminations,
        termination_offsets_fractional_u=_readonly_copy(offsets),
        prior_id=prior_id,
        metadata=MappingProxyType(
            {
                **({} if metadata is None else dict(metadata)),
                "scope": "truth_free_complete_slab_rebuild_prior",
                "metadata_affects_prior_id": False,
                "structurally_trusted": False,
            }
        ),
    )


def _validated_prior(prior: SiliconAlignmentPrior1D) -> SiliconAlignmentPrior1D:
    if not isinstance(prior, SiliconAlignmentPrior1D):
        raise TypeError("prior must be a SiliconAlignmentPrior1D")
    expected = _silicon_alignment_prior_id(
        axial_coordinates=prior.axial_coordinates,
        transverse_coordinates=prior.transverse_coordinates,
        reconstruction_mask=prior.reconstruction_mask,
        projected_si_template=prior.projected_si_template,
        projected_basis_fractional_su=prior.projected_basis_fractional_su,
        termination_offsets_fractional_u=prior.termination_offsets_fractional_u,
        template_half_shape=prior.template_half_shape,
        nominal_lattice_A=prior.nominal_lattice_A,
        slab_depth_A=prior.slab_depth_A,
        maximum_displacement_A=prior.maximum_displacement_A,
        displacement_control_spacing_s_A=prior.displacement_control_spacing_s_A,
        displacement_control_spacing_u_A=prior.displacement_control_spacing_u_A,
        termination_ids=prior.termination_ids,
    )
    if prior.prior_id != expected:
        raise ValueError("prior_id does not match the numerical alignment prior")
    return prior


def _alignment_control_axis(values: np.ndarray, spacing: float) -> np.ndarray:
    start = np.floor(np.min(values) / spacing) * spacing
    stop = np.ceil(np.max(values) / spacing) * spacing
    axis = np.arange(start, stop + 0.5 * spacing, spacing)
    return axis if axis.size > 1 else np.asarray([start, start + spacing])


def _candidate_site_coordinates(
    prior: SiliconAlignmentPrior1D,
    candidate: SiliconAlignmentCandidate1D,
) -> np.ndarray:
    lattice = prior.nominal_lattice_A * candidate.lattice_scale
    basis = np.asarray(prior.projected_basis_fractional_su) * lattice
    termination_index = prior.termination_ids.index(candidate.termination_id)
    termination_shift = (
        float(prior.termination_offsets_fractional_u[termination_index]) * lattice
    )
    phase_shift = candidate.axial_phase_fraction * 0.5 * lattice
    s_coordinates = np.asarray(prior.axial_coordinates)
    s_origin = float(s_coordinates[0])
    ds = _uniform_spacing("prior.axial_coordinates", s_coordinates)
    length = float(s_coordinates[-1] - s_origin + ds)
    depth = float(prior.slab_depth_A)
    rotation = float(candidate.in_plane_rotation_rad)
    extra = int(np.ceil(max(length, depth) * abs(np.sin(rotation)) / lattice)) + 4
    n_s_cells = int(np.ceil(length / lattice))
    n_u_cells = int(np.ceil(depth / lattice))
    cell_s = np.arange(-extra, n_s_cells + extra + 1)
    cell_u = np.arange(-n_u_cells - extra - 1, extra + 1)
    cells_s, cells_u, basis_index = np.meshgrid(
        cell_s,
        cell_u,
        np.arange(len(basis)),
        indexing="ij",
    )
    raw_s = (
        basis[basis_index, 0]
        + cells_s * lattice
        + phase_shift
    )
    raw_u = (
        basis[basis_index, 1]
        - np.max(basis[:, 1])
        + cells_u * lattice
        + termination_shift
    )
    cosine = np.cos(rotation)
    sine = np.sin(rotation)
    rotated_s = cosine * raw_s - sine * raw_u + s_origin
    rotated_u = sine * raw_s + cosine * raw_u
    selected = (
        (rotated_s >= s_origin)
        & (rotated_s < s_origin + length)
        & (rotated_u >= -depth)
        & (rotated_u <= 0.0)
    )
    sites = np.column_stack([rotated_s[selected], rotated_u[selected]])
    if not sites.size:
        raise ValueError("alignment candidate produces no sites in the finite slab")
    return np.unique(np.round(sites, decimals=10), axis=0)


def _alignment_patches_for_sites(
    sites: np.ndarray,
    template: np.ndarray,
    half_shape: tuple[int, int],
    s_coordinates: np.ndarray,
    u_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ds = _uniform_spacing("axial_coordinates", s_coordinates)
    du = _uniform_spacing("transverse_coordinates", u_coordinates)
    half_s, half_u = half_shape
    patches = []
    starts = []
    cache: dict[tuple[float, float], np.ndarray] = {}
    for site_s, site_u in sites:
        site_s_pixel = (site_s - s_coordinates[0]) / ds
        site_u_pixel = (site_u - u_coordinates[0]) / du
        center_s = int(np.rint(site_s_pixel))
        center_u = int(np.rint(site_u_pixel))
        shift = (site_s_pixel - center_s, site_u_pixel - center_u)
        key = tuple(float(np.round(value, 10)) for value in shift)
        if key not in cache:
            cache[key] = shift_image(
                template,
                shift=shift,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
        patches.append(cache[key].copy())
        starts.append((center_s - half_s, center_u - half_u))
    return np.asarray(patches), np.asarray(starts, dtype=np.int32)


def _alignment_reference_potential(
    sites: np.ndarray,
    template: np.ndarray,
    half_shape: tuple[int, int],
    s_coordinates: np.ndarray,
    u_coordinates: np.ndarray,
) -> np.ndarray:
    reference = np.zeros(
        (s_coordinates.size, u_coordinates.size),
        dtype=template.dtype,
    )
    patches, starts = _alignment_patches_for_sites(
        sites,
        template,
        half_shape,
        s_coordinates,
        u_coordinates,
    )
    for patch, (start_s, start_u) in zip(patches, starts):
        source_s_start = max(-int(start_s), 0)
        source_u_start = max(-int(start_u), 0)
        source_s_stop = min(patch.shape[0], len(s_coordinates) - int(start_s))
        source_u_stop = min(patch.shape[1], len(u_coordinates) - int(start_u))
        if source_s_start >= source_s_stop or source_u_start >= source_u_stop:
            continue
        target_s = slice(int(start_s) + source_s_start, int(start_s) + source_s_stop)
        target_u = slice(int(start_u) + source_u_start, int(start_u) + source_u_stop)
        reference[target_s, target_u] += patch[
            source_s_start:source_s_stop,
            source_u_start:source_u_stop,
        ]
    return reference


def _alignment_influence_mask(
    patches: np.ndarray,
    starts: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for patch, (start_s, start_u) in zip(patches, starts):
        local_s, local_u = np.indices(patch.shape)
        global_s = int(start_s) + local_s.ravel()
        global_u = int(start_u) + local_u.ravel()
        valid = (
            (global_s >= 0)
            & (global_s < shape[0])
            & (global_u >= 0)
            & (global_u < shape[1])
        )
        mask[global_s[valid], global_u[valid]] = True
    return mask


def rebuild_silicon_alignment_candidate_1d(
    prior: SiliconAlignmentPrior1D,
    candidate: SiliconAlignmentCandidate1D,
) -> SiliconAlignmentModel1D:
    """Rebuild every fixed and variable atom for one global candidate."""
    prior = _validated_prior(prior)
    candidate = _validated_candidate(candidate)
    if candidate.termination_id not in prior.termination_ids:
        raise ValueError("candidate termination is outside the alignment prior")
    s_coordinates = np.asarray(prior.axial_coordinates)
    u_coordinates = np.asarray(prior.transverse_coordinates)
    template = np.asarray(prior.projected_si_template)
    all_sites = _candidate_site_coordinates(prior, candidate)
    ds = _uniform_spacing("prior.axial_coordinates", s_coordinates)
    du = _uniform_spacing("prior.transverse_coordinates", u_coordinates)
    s_indices = np.rint((all_sites[:, 0] - s_coordinates[0]) / ds).astype(int)
    u_indices = np.rint((all_sites[:, 1] - u_coordinates[0]) / du).astype(int)
    in_grid = (
        (s_indices >= 0)
        & (s_indices < s_coordinates.size)
        & (u_indices >= 0)
        & (u_indices < u_coordinates.size)
    )
    mask = np.asarray(prior.reconstruction_mask)
    active = in_grid & mask[
        np.clip(s_indices, 0, s_coordinates.size - 1),
        np.clip(u_indices, 0, u_coordinates.size - 1),
    ]
    variable_sites = all_sites[active]
    if not variable_sites.size:
        raise ValueError("alignment candidate has no sites in the mutable support")
    reference = _alignment_reference_potential(
        all_sites,
        template,
        prior.template_half_shape,
        s_coordinates,
        u_coordinates,
    )
    patches, starts = _alignment_patches_for_sites(
        variable_sites,
        template,
        prior.template_half_shape,
        s_coordinates,
        u_coordinates,
    )
    control_s = _alignment_control_axis(
        variable_sites[:, 0],
        prior.displacement_control_spacing_s_A,
    )
    control_u = _alignment_control_axis(
        variable_sites[:, 1],
        prior.displacement_control_spacing_u_A,
    )
    influence = _alignment_influence_mask(patches, starts, reference.shape)
    model = LatticeSiteModel1D(
        reference_potential=reference,
        site_coordinates=variable_sites,
        site_patches=patches,
        patch_starts=starts,
        control_coordinates_s=control_s,
        control_coordinates_u=control_u,
        axial_sampling=ds,
        transverse_sampling=du,
        maximum_displacement=prior.maximum_displacement_A,
        metadata=MappingProxyType(
            {
                "species": "Si",
                "alignment_candidate_id": candidate.candidate_id,
                "alignment_prior_id": prior.prior_id,
                "reference_rebuild_scope": "complete_finite_slab_all_atoms",
                "fixed_exterior_rebuilt": True,
            }
        ),
    )
    zero_render = np.asarray(
        render_lattice_site_potential_1d(
            model,
            np.zeros(len(variable_sites), dtype=reference.dtype),
            np.zeros(
                (len(control_s), len(control_u), 2),
                dtype=reference.dtype,
            ),
        )
    )
    tolerance = 16.0 * np.finfo(reference.dtype).eps * max(
        float(np.max(np.abs(reference))),
        1.0,
    )
    if not np.allclose(
        zero_render,
        reference,
        rtol=16.0 * np.finfo(reference.dtype).eps,
        atol=tolerance,
    ):
        raise RuntimeError("zero defect/strain does not reproduce candidate reference")
    model_id = _digest_arrays(
        {
            "reference_potential": reference,
            "all_site_coordinates": all_sites,
            "variable_site_coordinates": variable_sites,
            "site_patches": patches,
            "patch_starts": starts,
            "control_coordinates_s": control_s,
            "control_coordinates_u": control_u,
            "lattice_influence_mask": influence,
        },
        {
            "contract": "silicon_alignment_complete_slab_model:v1",
            "prior_id": prior.prior_id,
            "candidate_id": candidate.candidate_id,
        },
    )
    return SiliconAlignmentModel1D(
        candidate=candidate,
        lattice_model=model,
        all_site_coordinates=_readonly_copy(all_sites),
        variable_site_coordinates=_readonly_copy(variable_sites),
        lattice_influence_mask=_readonly_copy(influence, dtype=bool),
        prior_id=prior.prior_id,
        candidate_model_id=model_id,
        metadata=MappingProxyType(
            {
                "scope": "complete_finite_slab_candidate_rebuild",
                "uses_diffraction_values": False,
                "uses_defect_truth": False,
                "structurally_trusted": False,
            }
        ),
    )


def _alignment_forward_problem_id(
    *,
    prior_id: str,
    input_probes: Any,
    propagation_kernel: Any,
    window_starts: Any,
    scan_coordinates: Any,
    detector_angles: Any,
    training_indices: Any,
    validation_indices: Any,
    audit_indices: Any,
    guard_indices: Any,
    window_length: int,
    slice_thickness_A: float,
    energy_eV: float,
) -> str:
    return _digest_arrays(
        {
            "input_probes": input_probes,
            "propagation_kernel": propagation_kernel,
            "window_starts": window_starts,
            "scan_coordinates": scan_coordinates,
            "detector_angles": detector_angles,
            "training_indices": training_indices,
            "validation_indices": validation_indices,
            "audit_indices": audit_indices,
            "guard_indices": guard_indices,
        },
        {
            "contract": "silicon_alignment_forward_problem:v1",
            "prior_id": prior_id,
            "window_length": window_length,
            "slice_thickness_A": slice_thickness_A,
            "energy_eV": energy_eV,
        },
    )


def make_silicon_alignment_forward_problem_1d(
    prior: SiliconAlignmentPrior1D,
    *,
    input_probes: Any,
    propagation_kernel: Any,
    window_starts: Any,
    window_length: int,
    scan_coordinates: Any,
    detector_angles: Any,
    slice_thickness_A: Any,
    energy_eV: Any,
    training_indices: Any,
    validation_indices: Any,
    audit_indices: Any = (),
    guard_indices: Any = (),
    metadata: Mapping[str, Any] | None = None,
) -> SiliconAlignmentForwardProblem1D:
    """Bind a complete-slab prior to known geometry without observations."""
    prior = _validated_prior(prior)
    s_coordinates = np.asarray(prior.axial_coordinates)
    n_s = s_coordinates.size
    n_u = len(prior.transverse_coordinates)
    starts = np.asarray(window_starts)
    if starts.ndim != 1 or not starts.size or not np.issubdtype(
        starts.dtype, np.integer
    ):
        raise TypeError("window_starts must be a non-empty integer vector")
    starts = starts.astype(np.int32, copy=False)
    if isinstance(window_length, (bool, np.bool_)):
        raise TypeError("window_length must be an integer")
    try:
        length = operator.index(window_length)
    except TypeError as error:
        raise TypeError("window_length must be an integer") from error
    if length < 1 or length > n_s or np.any(starts < 0) or np.any(
        starts + length > n_s
    ):
        raise ValueError("window length/starts are outside the prior potential")
    n_scans = starts.size
    coordinates = np.asarray(scan_coordinates)
    angles = np.asarray(detector_angles)
    if coordinates.shape != (n_scans,) or np.iscomplexobj(coordinates) or np.any(
        ~np.isfinite(coordinates)
    ):
        raise ValueError("scan_coordinates must contain one finite value per scan")
    if angles.shape != (n_u,) or np.iscomplexobj(angles) or np.any(
        ~np.isfinite(angles)
    ):
        raise ValueError("detector_angles must match the transverse grid")
    probes = np.asarray(input_probes)
    if probes.ndim == 1:
        if probes.shape != (n_u,):
            raise ValueError("shared input probe must match the transverse grid")
        probes = np.broadcast_to(probes, (n_scans, n_u)).copy()
    elif probes.shape != (n_scans, n_u):
        raise ValueError("input_probes must have one row per scan")
    if np.any(~np.isfinite(probes)):
        raise ValueError("input_probes must be finite")
    kernel = np.asarray(propagation_kernel)
    if kernel.shape != (n_u,) or np.any(~np.isfinite(kernel)):
        raise ValueError("propagation_kernel must match the transverse grid")
    partitions = {
        "training_indices": _validated_partition(
            "training_indices", training_indices, n_scans
        ),
        "validation_indices": _validated_partition(
            "validation_indices", validation_indices, n_scans
        ),
        "audit_indices": _validated_partition("audit_indices", audit_indices, n_scans),
        "guard_indices": _validated_partition("guard_indices", guard_indices, n_scans),
    }
    if not partitions["training_indices"].size or not partitions[
        "validation_indices"
    ].size:
        raise ValueError("alignment requires non-empty training and validation")
    names = tuple(partitions)
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1 :]:
            if np.intersect1d(
                partitions[first_name], partitions[second_name]
            ).size:
                raise ValueError(f"{first_name} and {second_name} must be disjoint")
    slice_thickness = _finite_scalar("slice_thickness_A", slice_thickness_A)
    energy = _finite_scalar("energy_eV", energy_eV)
    if slice_thickness <= 0.0 or energy <= 0.0:
        raise ValueError("slice_thickness_A and energy_eV must be positive")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    problem_id = _alignment_forward_problem_id(
        prior_id=prior.prior_id,
        input_probes=probes,
        propagation_kernel=kernel,
        window_starts=starts,
        scan_coordinates=coordinates,
        detector_angles=angles,
        window_length=length,
        slice_thickness_A=slice_thickness,
        energy_eV=energy,
        **partitions,
    )
    return SiliconAlignmentForwardProblem1D(
        prior=prior,
        input_probes=_readonly_copy(probes),
        propagation_kernel=_readonly_copy(kernel),
        window_starts=_readonly_copy(starts, dtype=np.int32),
        window_length=length,
        scan_coordinates=_readonly_copy(coordinates),
        detector_angles=_readonly_copy(angles),
        slice_thickness_A=slice_thickness,
        energy_eV=energy,
        training_indices=_readonly_copy(partitions["training_indices"]),
        validation_indices=_readonly_copy(partitions["validation_indices"]),
        audit_indices=_readonly_copy(partitions["audit_indices"]),
        guard_indices=_readonly_copy(partitions["guard_indices"]),
        alignment_problem_id=problem_id,
        metadata=MappingProxyType(
            {
                **({} if metadata is None else dict(metadata)),
                "scope": "truth_free_known_forward_geometry",
                "contains_observations": False,
                "structurally_trusted": False,
            }
        ),
    )


def _validated_forward_problem(
    problem: SiliconAlignmentForwardProblem1D,
) -> SiliconAlignmentForwardProblem1D:
    if not isinstance(problem, SiliconAlignmentForwardProblem1D):
        raise TypeError("problem must be a SiliconAlignmentForwardProblem1D")
    _validated_prior(problem.prior)
    expected = _alignment_forward_problem_id(
        prior_id=problem.prior.prior_id,
        input_probes=problem.input_probes,
        propagation_kernel=problem.propagation_kernel,
        window_starts=problem.window_starts,
        scan_coordinates=problem.scan_coordinates,
        detector_angles=problem.detector_angles,
        training_indices=problem.training_indices,
        validation_indices=problem.validation_indices,
        audit_indices=problem.audit_indices,
        guard_indices=problem.guard_indices,
        window_length=problem.window_length,
        slice_thickness_A=problem.slice_thickness_A,
        energy_eV=problem.energy_eV,
    )
    if problem.alignment_problem_id != expected:
        raise ValueError("alignment_problem_id does not match its numerical inputs")
    return problem


def _alignment_loss_per_scan(
    predicted: Any,
    measured: Any,
    valid_mask: Any | None,
) -> np.ndarray:
    predicted_array = np.asarray(predicted)
    measured_array = np.asarray(measured)
    if predicted_array.shape != measured_array.shape or predicted_array.ndim != 2:
        raise ValueError("alignment prediction and measurement shapes differ")
    losses = []
    for index in range(predicted_array.shape[0]):
        losses.append(
            float(
                np.asarray(
                    normalized_amplitude_loss_1d(
                        predicted_array[index],
                        measured_array[index],
                        detector_valid_mask=(
                            None if valid_mask is None else valid_mask[index]
                        ),
                    )
                )
            )
        )
    result = np.asarray(losses, dtype=float)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("alignment candidate produced non-finite loss")
    return result


def _bind_alignment_selection_to_model(
    model: SiliconAlignmentModel1D,
    summary: AlignmentSelectionSummary1D,
) -> SiliconAlignmentModel1D:
    if model.candidate.candidate_id != summary.selected_candidate_id:
        raise ValueError("selected model and alignment summary differ")
    lattice_model = replace(
        model.lattice_model,
        metadata=MappingProxyType(
            {
                **dict(model.lattice_model.metadata),
                "alignment_selection_id": summary.alignment_selection_id,
                "alignment_selection_unique": summary.unique_selection,
                "alignment_equivalent_candidate_ids": list(
                    summary.equivalent_candidate_ids
                ),
                "displacement_gauge_required": (
                    "translation_rotation_isotropic_dilation"
                ),
            }
        ),
    )
    return replace(
        model,
        lattice_model=lattice_model,
        metadata=MappingProxyType(
            {
                **dict(model.metadata),
                "alignment_selection_id": summary.alignment_selection_id,
                "alignment_selection_unique": summary.unique_selection,
            }
        ),
    )


def initialize_silicon_alignment_1d(
    problem: SiliconAlignmentForwardProblem1D,
    scan: GlancingScan1D,
    *,
    options: AlignmentInitializationOptions1D | None = None,
) -> SiliconAlignmentInitialization1D:
    """Screen pristine complete-slab candidates using training then validation.

    The coarse catalog and local refinement stencil are frozen before
    diffraction values are copied. Only a geometry-stratified training screen
    ranks coarse and fine candidates. Complete validation evaluates the final
    deterministic shortlist. Audit and guard observations never cross the
    selection boundary. Vacancy fractions and residual strain remain exactly
    zero during this initialization stage.
    """
    problem = _validated_forward_problem(problem)
    options = _validated_options(options)
    if type(scan) is not GlancingScan1D:
        raise TypeError("scan must be exactly a truth-free GlancingScan1D")
    for name, observed, expected in (
        ("window_starts", scan.window_starts, problem.window_starts),
        ("scan_coordinates", scan.scan_coordinates, problem.scan_coordinates),
        ("detector_angles", scan.detector_angles, problem.detector_angles),
    ):
        observed_array = np.asarray(observed)
        expected_array = np.asarray(expected)
        if observed_array.shape != expected_array.shape or not np.allclose(
            observed_array,
            expected_array,
            rtol=8.0 * np.finfo(float).eps,
            atol=8.0 * np.finfo(float).eps,
        ):
            raise ValueError(f"scan {name} do not match the alignment problem")
    coarse_candidates = generate_silicon_alignment_candidates_1d(
        problem.prior.termination_ids,
        options=options,
    )
    coarse_catalog_id = alignment_candidate_catalog_id_1d(
        coarse_candidates,
        options=options,
    )
    selection_data = build_alignment_selection_data_1d(
        scan,
        training_indices=problem.training_indices,
        validation_indices=problem.validation_indices,
        audit_indices=problem.audit_indices,
        guard_indices=problem.guard_indices,
        training_screen_scan_count=options.training_screen_scan_count,
    )
    training_source = np.asarray(selection_data.training_source_indices)
    training_local = np.asarray(selection_data.training_local_indices)
    models: dict[str, SiliconAlignmentModel1D] = {}
    training_losses: dict[str, float] = {}

    def screen_training(candidate: SiliconAlignmentCandidate1D) -> None:
        if candidate.candidate_id in training_losses:
            return
        model = rebuild_silicon_alignment_candidate_1d(problem.prior, candidate)
        models[candidate.candidate_id] = model
        predicted = simulate_glancing_scan_1d(
            model.lattice_model.reference_potential,
            np.asarray(problem.input_probes)[training_source],
            np.asarray(problem.window_starts)[training_source],
            problem.window_length,
            problem.propagation_kernel,
            problem.slice_thickness_A,
            problem.energy_eV,
            rematerialize=False,
        )
        mask = (
            None
            if selection_data.detector_valid_mask is None
            else np.asarray(selection_data.detector_valid_mask)[training_local]
        )
        per_scan = _alignment_loss_per_scan(
            predicted,
            np.asarray(selection_data.intensities)[training_local],
            mask,
        )
        training_losses[candidate.candidate_id] = float(np.mean(per_scan))

    all_candidates = list(coarse_candidates)
    for candidate in coarse_candidates:
        screen_training(candidate)
    frontier = sorted(
        coarse_candidates,
        key=lambda candidate: (
            training_losses[candidate.candidate_id],
            candidate.candidate_id,
        ),
    )[: min(options.coarse_shortlist_size, len(coarse_candidates))]
    for _ in range(options.refinement_rounds):
        refined = refine_silicon_alignment_candidates_1d(
            frontier,
            options=options,
        )
        known_ids = {candidate.candidate_id for candidate in all_candidates}
        new_candidates = tuple(
            candidate
            for candidate in refined
            if candidate.candidate_id not in known_ids
        )
        if not new_candidates:
            break
        for candidate in new_candidates:
            screen_training(candidate)
        all_candidates.extend(new_candidates)
        frontier = sorted(
            all_candidates,
            key=lambda candidate: (
                training_losses[candidate.candidate_id],
                candidate.candidate_id,
            ),
        )[: min(options.coarse_shortlist_size, len(all_candidates))]

    candidate_tuple = tuple(all_candidates)
    catalog_id = alignment_candidate_catalog_id_1d(
        candidate_tuple,
        options=options,
    )

    shortlist_count = min(options.validation_shortlist_size, len(candidate_tuple))
    shortlist = sorted(
        candidate_tuple,
        key=lambda candidate: (
            training_losses[candidate.candidate_id],
            candidate.candidate_id,
        ),
    )[:shortlist_count]
    validation_source = np.asarray(selection_data.validation_source_indices)
    validation_local = np.asarray(selection_data.validation_local_indices)
    scores = []
    for candidate in shortlist:
        model = models[candidate.candidate_id]
        predicted = simulate_glancing_scan_1d(
            model.lattice_model.reference_potential,
            np.asarray(problem.input_probes)[validation_source],
            np.asarray(problem.window_starts)[validation_source],
            problem.window_length,
            problem.propagation_kernel,
            problem.slice_thickness_A,
            problem.energy_eV,
            rematerialize=False,
        )
        mask = (
            None
            if selection_data.detector_valid_mask is None
            else np.asarray(selection_data.detector_valid_mask)[validation_local]
        )
        validation_per_scan = _alignment_loss_per_scan(
            predicted,
            np.asarray(selection_data.intensities)[validation_local],
            mask,
        )
        scores.append(
            AlignmentCandidateScore1D(
                candidate=candidate,
                training_screen_loss=training_losses[candidate.candidate_id],
                validation_loss=float(np.mean(validation_per_scan)),
                validation_loss_per_scan=_readonly_copy(validation_per_scan),
                candidate_model_id=model.candidate_model_id,
            )
        )
    score_tuple = tuple(scores)
    summary = select_alignment_candidate_1d(
        score_tuple,
        selection_data_id=selection_data.selection_data_id,
        candidate_catalog_id=catalog_id,
        options=options,
    )
    selected_model = _bind_alignment_selection_to_model(
        models[summary.selected_candidate_id],
        summary,
    )
    return SiliconAlignmentInitialization1D(
        selected_model=selected_model,
        candidate_scores=score_tuple,
        selection_summary=summary,
        training_screen_indices=_readonly_copy(training_source),
        validation_indices=_readonly_copy(problem.validation_indices),
        audit_indices=_readonly_copy(problem.audit_indices),
        guard_indices=_readonly_copy(problem.guard_indices),
        alignment_problem_id=problem.alignment_problem_id,
        candidate_catalog=candidate_tuple,
        structurally_trusted=False,
        metadata=MappingProxyType(
            {
                "scope": "pristine_complete_slab_coarse_to_fine_alignment_v1",
                "coarse_candidate_catalog_id": coarse_catalog_id,
                "coarse_candidate_count": len(coarse_candidates),
                "candidate_catalog_size": len(candidate_tuple),
                "refinement_rounds_completed": max(
                    (candidate.refinement_level for candidate in candidate_tuple),
                    default=0,
                ),
                "validation_shortlist_size": shortlist_count,
                "initialization_options": _alignment_options_payload(options),
                "vacancy_initialization": "all_occupied",
                "residual_strain_initialization": "zero",
                "probe_refinement": "not_implemented_fixed_calibrated_probe",
                "audit_used_for_selection": False,
                "structural_trust_reason": (
                    "alignment selection alone is not a structural validation"
                ),
            }
        ),
    )


def prepare_aligned_lattice_site_reconstruction_1d(
    initialization: SiliconAlignmentInitialization1D,
    problem: SiliconAlignmentForwardProblem1D,
    scan: GlancingScan1D,
    *,
    measurement: PtychographyMeasurement1D | None = None,
    objective: PtychographyObjective1D | None = None,
    potential_max: Any | None = None,
    minibatch_size: int = 5,
    evaluation_batch_size: int = 10,
    gradient_clip: Any = 1.0,
    epsilon: Any = 1e-12,
    rematerialize: bool = True,
) -> PreparedLatticeSiteReconstruction1D:
    """Prepare the validation-selected model with its global gauge frozen.

    Residual controls are projected off translations, rotation, and isotropic
    dilation, so local strain cannot undo the complete-slab alignment. Audit
    values remain present only in the prepared post-selection partition.
    """
    problem = _validated_forward_problem(problem)
    if not isinstance(initialization, SiliconAlignmentInitialization1D):
        raise TypeError(
            "initialization must be a SiliconAlignmentInitialization1D"
        )
    if initialization.alignment_problem_id != problem.alignment_problem_id:
        raise ValueError("initialization and alignment problem IDs differ")
    summary = initialization.selection_summary
    selected = initialization.selected_model
    if (
        selected.candidate.candidate_id != summary.selected_candidate_id
        or selected.prior_id != problem.prior.prior_id
        or selected.metadata.get("alignment_selection_id")
        != summary.alignment_selection_id
        or selected.lattice_model.metadata.get("alignment_selection_id")
        != summary.alignment_selection_id
    ):
        raise ValueError("selected model is inconsistent with alignment summary")
    if type(scan) is not GlancingScan1D:
        raise TypeError("scan must be exactly a truth-free GlancingScan1D")
    for name, observed, expected in (
        ("window_starts", scan.window_starts, problem.window_starts),
        ("scan_coordinates", scan.scan_coordinates, problem.scan_coordinates),
        ("detector_angles", scan.detector_angles, problem.detector_angles),
    ):
        observed_array = np.asarray(observed)
        expected_array = np.asarray(expected)
        if observed_array.shape != expected_array.shape or not np.allclose(
            observed_array,
            expected_array,
            rtol=8.0 * np.finfo(float).eps,
            atol=8.0 * np.finfo(float).eps,
        ):
            raise ValueError(f"scan {name} do not match the alignment problem")
    if measurement is None:
        measured_intensities = scan.intensities
        detector_valid_mask = scan.detector_valid_mask
    else:
        measured_intensities = None
        detector_valid_mask = None
        if objective is None:
            raise ValueError("objective is required with measurement")
    resolved_potential_max = (
        2.0
        * max(
            float(
                np.max(
                    np.asarray(selected.lattice_model.reference_potential)
                )
            ),
            1.0,
        )
        if potential_max is None
        else potential_max
    )
    return prepare_lattice_site_reconstruction_1d(
        selected.lattice_model,
        problem.input_probes,
        problem.window_starts,
        problem.window_length,
        problem.propagation_kernel,
        problem.slice_thickness_A,
        problem.energy_eV,
        measured_intensities,
        measurement=measurement,
        objective=objective,
        detector_valid_mask=detector_valid_mask,
        separate_rigid_registration=False,
        similarity_residual_gauge=True,
        scan_coordinates=problem.scan_coordinates,
        detector_angles=problem.detector_angles,
        validation_indices=problem.validation_indices,
        audit_indices=problem.audit_indices,
        excluded_indices=problem.guard_indices,
        potential_max=resolved_potential_max,
        minibatch_size=minibatch_size,
        evaluation_batch_size=evaluation_batch_size,
        gradient_clip=gradient_clip,
        epsilon=epsilon,
        rematerialize=rematerialize,
    )


def reconstruct_aligned_lattice_site_1d(
    initialization: SiliconAlignmentInitialization1D,
    problem: SiliconAlignmentForwardProblem1D,
    scan: GlancingScan1D,
    *,
    measurement: PtychographyMeasurement1D | None = None,
    objective: PtychographyObjective1D | None = None,
    updates: int = 500,
    minibatch_size: int = 5,
    evaluation_batch_size: int = 10,
    validation_interval: int = 25,
    training_diagnostic_scan_count: int | None = 32,
    learning_rate_start: Any = 2e-2,
    learning_rate_end: Any = 2e-4,
    checkpoint_interval: int | None = 1,
    convergence: ConvergenceOptions1D | None = None,
    optimization: LatticeOptimizationOptions1D | None = None,
    rematerialize: bool = True,
    seed: int = 0,
    progress: bool = True,
) -> LatticeSiteReconstruction1D:
    """Prepare and refine a selected complete-slab model with local parameters."""
    prepared = prepare_aligned_lattice_site_reconstruction_1d(
        initialization,
        problem,
        scan,
        measurement=measurement,
        objective=objective,
        minibatch_size=minibatch_size,
        evaluation_batch_size=evaluation_batch_size,
        rematerialize=rematerialize,
    )
    return run_prepared_lattice_site_reconstruction_1d(
        prepared,
        learning_rate_start=learning_rate_start,
        learning_rate_end=learning_rate_end,
        updates=updates,
        validation_interval=validation_interval,
        training_diagnostic_scan_count=training_diagnostic_scan_count,
        seed=seed,
        progress=progress,
        progress_description="aligned lattice-site reconstruction",
        checkpoint_interval=checkpoint_interval,
        convergence=convergence,
        optimization=(
            LatticeOptimizationOptions1D(mode="staged")
            if optimization is None
            else optimization
        ),
    )


_ALIGNMENT_ARCHIVE_CONTRACT = (
    "wide_angle_propagation.silicon_alignment_initialization:v1"
)
_ALIGNMENT_ARCHIVE_FIELDS = frozenset(
    {
        "schema_version",
        "alignment_problem_id",
        "initialization_metadata_json",
        "catalog_termination_id",
        "catalog_axial_phase_fraction",
        "catalog_in_plane_rotation_rad",
        "catalog_lattice_scale",
        "catalog_refinement_level",
        "catalog_parent_present",
        "catalog_parent_candidate_id",
        "catalog_candidate_id",
        "score_candidate_id",
        "score_training_screen_loss",
        "score_validation_loss",
        "score_validation_loss_per_scan",
        "score_candidate_model_id",
        "summary_minimum_loss_candidate_id",
        "summary_selected_candidate_id",
        "summary_equivalent_candidate_ids",
        "summary_unique_selection",
        "summary_candidate_catalog_id",
        "summary_selection_data_id",
        "summary_alignment_selection_id",
        "selected_candidate_model_id",
        "training_screen_indices",
        "validation_indices",
        "audit_indices",
        "guard_indices",
        "initialization_structurally_trusted",
        "options_json",
    }
)


def _alignment_summary_matches(
    observed: AlignmentSelectionSummary1D,
    expected: AlignmentSelectionSummary1D,
) -> bool:
    return all(
        (
            observed.minimum_loss_candidate_id
            == expected.minimum_loss_candidate_id,
            observed.selected_candidate_id == expected.selected_candidate_id,
            observed.equivalent_candidate_ids
            == expected.equivalent_candidate_ids,
            observed.unique_selection == expected.unique_selection,
            observed.candidate_catalog_id == expected.candidate_catalog_id,
            observed.selection_data_id == expected.selection_data_id,
            observed.alignment_selection_id == expected.alignment_selection_id,
            observed.structurally_trusted is False,
            expected.structurally_trusted is False,
        )
    )


def _validated_alignment_initialization_evidence(
    initialization: SiliconAlignmentInitialization1D,
) -> tuple[
    AlignmentInitializationOptions1D,
    tuple[SiliconAlignmentCandidate1D, ...],
    tuple[AlignmentCandidateScore1D, ...],
    AlignmentSelectionSummary1D,
]:
    if not isinstance(initialization, SiliconAlignmentInitialization1D):
        raise TypeError(
            "initialization must be a SiliconAlignmentInitialization1D"
        )
    if initialization.structurally_trusted:
        raise ValueError("alignment initialization cannot claim structural trust")
    if not isinstance(initialization.metadata, Mapping):
        raise TypeError("initialization.metadata must be a mapping")
    options = _alignment_options_from_payload(
        initialization.metadata.get("initialization_options")
    )
    catalog = tuple(
        _validated_candidate(candidate)
        for candidate in initialization.candidate_catalog
    )
    if not catalog:
        raise ValueError("alignment initialization omits its candidate catalog")
    catalog_ids = [candidate.candidate_id for candidate in catalog]
    if len(set(catalog_ids)) != len(catalog_ids):
        raise ValueError("alignment candidate catalog contains duplicate IDs")
    catalog_by_id = {candidate.candidate_id: candidate for candidate in catalog}
    for candidate in catalog:
        if candidate.refinement_level == 0:
            if candidate.parent_candidate_id is not None:
                raise ValueError("coarse alignment candidates cannot have parents")
            continue
        parent = catalog_by_id.get(candidate.parent_candidate_id)
        if (
            parent is None
            or parent.termination_id != candidate.termination_id
            or parent.refinement_level + 1 != candidate.refinement_level
        ):
            raise ValueError("fine alignment candidate has an invalid parent")
    scores = tuple(
        _validated_score(score) for score in initialization.candidate_scores
    )
    if not scores:
        raise ValueError("alignment initialization omits candidate scores")
    score_ids = [score.candidate.candidate_id for score in scores]
    if len(set(score_ids)) != len(score_ids) or not set(score_ids).issubset(
        catalog_by_id
    ):
        raise ValueError("alignment scores do not reference a unique catalog subset")
    for score in scores:
        if score.candidate != catalog_by_id[score.candidate.candidate_id]:
            raise ValueError("alignment score candidate differs from its catalog entry")
    summary = initialization.selection_summary
    expected_catalog_id = alignment_candidate_catalog_id_1d(
        catalog,
        options=options,
    )
    if summary.candidate_catalog_id != expected_catalog_id:
        raise ValueError("alignment summary and candidate catalog IDs differ")
    expected_summary = select_alignment_candidate_1d(
        scores,
        selection_data_id=summary.selection_data_id,
        candidate_catalog_id=expected_catalog_id,
        options=options,
    )
    if not _alignment_summary_matches(summary, expected_summary):
        raise ValueError("alignment summary does not reproduce from its scores")
    selected = initialization.selected_model
    selected_score = next(
        (
            score
            for score in scores
            if score.candidate.candidate_id == summary.selected_candidate_id
        ),
        None,
    )
    if (
        selected_score is None
        or selected.candidate != selected_score.candidate
        or selected.candidate_model_id != selected_score.candidate_model_id
        or selected.metadata.get("alignment_selection_id")
        != summary.alignment_selection_id
        or selected.lattice_model.metadata.get("alignment_selection_id")
        != summary.alignment_selection_id
    ):
        raise ValueError("selected alignment model is inconsistent with its evidence")
    if not isinstance(initialization.alignment_problem_id, str) or not (
        initialization.alignment_problem_id.strip()
    ):
        raise ValueError("alignment_problem_id must be a non-empty string")
    return options, catalog, scores, summary


def _alignment_archive_payload(
    initialization: SiliconAlignmentInitialization1D,
) -> dict[str, np.ndarray]:
    options, catalog, scores, summary = (
        _validated_alignment_initialization_evidence(initialization)
    )
    parents_present = np.asarray(
        [candidate.parent_candidate_id is not None for candidate in catalog],
        dtype=bool,
    )
    parent_ids = np.asarray(
        [candidate.parent_candidate_id or "" for candidate in catalog],
        dtype=np.str_,
    )
    metadata_json = json.dumps(
        dict(initialization.metadata),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "alignment_problem_id": np.asarray(initialization.alignment_problem_id),
        "initialization_metadata_json": np.asarray(metadata_json),
        "catalog_termination_id": np.asarray(
            [candidate.termination_id for candidate in catalog], dtype=np.str_
        ),
        "catalog_axial_phase_fraction": np.asarray(
            [candidate.axial_phase_fraction for candidate in catalog], dtype=float
        ),
        "catalog_in_plane_rotation_rad": np.asarray(
            [candidate.in_plane_rotation_rad for candidate in catalog], dtype=float
        ),
        "catalog_lattice_scale": np.asarray(
            [candidate.lattice_scale for candidate in catalog], dtype=float
        ),
        "catalog_refinement_level": np.asarray(
            [candidate.refinement_level for candidate in catalog], dtype=np.int64
        ),
        "catalog_parent_present": parents_present,
        "catalog_parent_candidate_id": parent_ids,
        "catalog_candidate_id": np.asarray(
            [candidate.candidate_id for candidate in catalog], dtype=np.str_
        ),
        "score_candidate_id": np.asarray(
            [score.candidate.candidate_id for score in scores], dtype=np.str_
        ),
        "score_training_screen_loss": np.asarray(
            [score.training_screen_loss for score in scores], dtype=float
        ),
        "score_validation_loss": np.asarray(
            [score.validation_loss for score in scores], dtype=float
        ),
        "score_validation_loss_per_scan": np.stack(
            [np.asarray(score.validation_loss_per_scan) for score in scores]
        ),
        "score_candidate_model_id": np.asarray(
            [score.candidate_model_id for score in scores], dtype=np.str_
        ),
        "summary_minimum_loss_candidate_id": np.asarray(
            summary.minimum_loss_candidate_id
        ),
        "summary_selected_candidate_id": np.asarray(
            summary.selected_candidate_id
        ),
        "summary_equivalent_candidate_ids": np.asarray(
            summary.equivalent_candidate_ids, dtype=np.str_
        ),
        "summary_unique_selection": np.asarray(summary.unique_selection),
        "summary_candidate_catalog_id": np.asarray(summary.candidate_catalog_id),
        "summary_selection_data_id": np.asarray(summary.selection_data_id),
        "summary_alignment_selection_id": np.asarray(
            summary.alignment_selection_id
        ),
        "selected_candidate_model_id": np.asarray(
            initialization.selected_model.candidate_model_id
        ),
        "training_screen_indices": np.asarray(
            initialization.training_screen_indices, dtype=np.int64
        ),
        "validation_indices": np.asarray(
            initialization.validation_indices, dtype=np.int64
        ),
        "audit_indices": np.asarray(
            initialization.audit_indices, dtype=np.int64
        ),
        "guard_indices": np.asarray(
            initialization.guard_indices, dtype=np.int64
        ),
        "initialization_structurally_trusted": np.asarray(False),
        "options_json": np.asarray(
            json.dumps(
                _alignment_options_payload(options),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    }


def _atomic_save_alignment_archive(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
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
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_silicon_alignment_initialization_1d(
    path: str | Path,
    initialization: SiliconAlignmentInitialization1D,
) -> None:
    """Atomically save non-pickled alignment evidence with a SHA-256 digest."""
    payload = _alignment_archive_payload(initialization)
    archive_digest = _digest_arrays(
        payload,
        {
            "contract": _ALIGNMENT_ARCHIVE_CONTRACT,
            "schema_version": 1,
        },
    )
    _atomic_save_alignment_archive(
        path,
        {**payload, "archive_digest": np.asarray(archive_digest)},
    )


def _catalog_from_alignment_archive(
    payload: Mapping[str, np.ndarray],
) -> tuple[SiliconAlignmentCandidate1D, ...]:
    candidate_ids = np.asarray(payload["catalog_candidate_id"])
    if candidate_ids.ndim != 1 or candidate_ids.dtype.kind != "U":
        raise ValueError("archived candidate IDs must be a Unicode vector")
    n_candidates = len(candidate_ids)
    fields = {
        name: np.asarray(payload[name])
        for name in (
            "catalog_termination_id",
            "catalog_axial_phase_fraction",
            "catalog_in_plane_rotation_rad",
            "catalog_lattice_scale",
            "catalog_refinement_level",
            "catalog_parent_present",
            "catalog_parent_candidate_id",
        )
    }
    if n_candidates < 1 or any(
        value.shape != (n_candidates,) for value in fields.values()
    ):
        raise ValueError("archived alignment catalog has invalid shapes")
    for name in (
        "catalog_termination_id",
        "catalog_parent_candidate_id",
    ):
        if fields[name].dtype.kind != "U":
            raise ValueError(f"archived {name} must use a Unicode dtype")
    for name in (
        "catalog_axial_phase_fraction",
        "catalog_in_plane_rotation_rad",
        "catalog_lattice_scale",
    ):
        if not np.issubdtype(fields[name].dtype, np.floating):
            raise ValueError(f"archived {name} must use a floating dtype")
    if not np.issubdtype(
        fields["catalog_refinement_level"].dtype,
        np.integer,
    ):
        raise ValueError("archived refinement levels must use an integer dtype")
    if fields["catalog_parent_present"].dtype != np.bool_:
        raise ValueError("archived parent markers must be Boolean")
    candidates = []
    for index in range(n_candidates):
        parent = (
            str(fields["catalog_parent_candidate_id"][index])
            if bool(fields["catalog_parent_present"][index])
            else None
        )
        candidate = _make_alignment_candidate(
            termination_id=str(fields["catalog_termination_id"][index]),
            axial_phase_fraction=float(
                fields["catalog_axial_phase_fraction"][index]
            ),
            in_plane_rotation_rad=float(
                fields["catalog_in_plane_rotation_rad"][index]
            ),
            lattice_scale=float(fields["catalog_lattice_scale"][index]),
            refinement_level=int(fields["catalog_refinement_level"][index]),
            parent_candidate_id=parent,
        )
        if candidate.candidate_id != str(candidate_ids[index]):
            raise ValueError("archived candidate ID does not match its parameters")
        candidates.append(candidate)
    return tuple(candidates)


def _alignment_archive_scalar(
    payload: Mapping[str, np.ndarray],
    name: str,
) -> Any:
    value = np.asarray(payload[name])
    if value.shape != ():
        raise ValueError(f"archived {name} must be scalar")
    return value.item()


def _alignment_archive_boolean_scalar(
    payload: Mapping[str, np.ndarray],
    name: str,
) -> bool:
    value = np.asarray(payload[name])
    if value.shape != () or value.dtype != np.bool_:
        raise ValueError(f"archived {name} must be a scalar Boolean")
    return bool(value.item())


def _alignment_archive_integer_scalar(
    payload: Mapping[str, np.ndarray],
    name: str,
) -> int:
    value = np.asarray(payload[name])
    if value.shape != () or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"archived {name} must be a scalar integer")
    return int(value.item())


def _scores_from_alignment_archive(
    payload: Mapping[str, np.ndarray],
    catalog: Sequence[SiliconAlignmentCandidate1D],
) -> tuple[AlignmentCandidateScore1D, ...]:
    score_ids = np.asarray(payload["score_candidate_id"])
    n_scores = len(score_ids)
    training_losses = np.asarray(payload["score_training_screen_loss"])
    validation_losses = np.asarray(payload["score_validation_loss"])
    validation_per_scan = np.asarray(
        payload["score_validation_loss_per_scan"]
    )
    model_ids = np.asarray(payload["score_candidate_model_id"])
    if (
        n_scores < 1
        or training_losses.shape != (n_scores,)
        or validation_losses.shape != (n_scores,)
        or model_ids.shape != (n_scores,)
        or validation_per_scan.ndim != 2
        or validation_per_scan.shape[0] != n_scores
        or validation_per_scan.shape[1] < 1
    ):
        raise ValueError("archived alignment scores have invalid shapes")
    if score_ids.dtype.kind != "U" or model_ids.dtype.kind != "U":
        raise ValueError("archived score IDs must use a Unicode dtype")
    for name, value in (
        ("score_training_screen_loss", training_losses),
        ("score_validation_loss", validation_losses),
        ("score_validation_loss_per_scan", validation_per_scan),
    ):
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"archived {name} must use a floating dtype")
    catalog_by_id = {
        candidate.candidate_id: candidate for candidate in catalog
    }
    scores = []
    for index, candidate_id_value in enumerate(score_ids):
        candidate_id = str(candidate_id_value)
        if candidate_id not in catalog_by_id:
            raise ValueError("archived score references an unknown candidate")
        scores.append(
            _validated_score(
                AlignmentCandidateScore1D(
                    candidate=catalog_by_id[candidate_id],
                    training_screen_loss=float(training_losses[index]),
                    validation_loss=float(validation_losses[index]),
                    validation_loss_per_scan=_readonly_copy(
                        validation_per_scan[index]
                    ),
                    candidate_model_id=str(model_ids[index]),
                )
            )
        )
    if len({score.candidate.candidate_id for score in scores}) != n_scores:
        raise ValueError("archived alignment scores contain duplicate candidates")
    return tuple(scores)


def _alignment_evidence_is_close(
    archived: Any,
    recomputed: Any,
) -> bool:
    first = np.asarray(archived)
    second = np.asarray(recomputed)
    if first.shape != second.shape:
        return False
    dtype = np.result_type(first.dtype, second.dtype, np.float32)
    tolerance = max(512.0 * np.finfo(dtype).eps, 5e-10)
    scale = max(
        float(np.max(np.abs(first))) if first.size else 0.0,
        float(np.max(np.abs(second))) if second.size else 0.0,
        1.0,
    )
    return bool(
        np.allclose(
            first,
            second,
            rtol=tolerance,
            atol=tolerance * scale,
        )
    )


def load_silicon_alignment_initialization_1d(
    path: str | Path,
    problem: SiliconAlignmentForwardProblem1D,
    scan: GlancingScan1D,
) -> SiliconAlignmentInitialization1D:
    """Load and reverify alignment evidence against its problem and raw scan.

    The archive is non-pickled and digest-bound. Loading also rebuilds every
    validation-shortlisted model and recomputes its training and validation
    losses, so a valid container alone cannot substitute for matching physics
    inputs or matching diffraction data.
    """
    problem = _validated_forward_problem(problem)
    if type(scan) is not GlancingScan1D:
        raise TypeError("scan must be exactly a truth-free GlancingScan1D")
    with np.load(path, allow_pickle=False) as archive:
        files = set(archive.files)
        if "archive_digest" not in files:
            raise ValueError("alignment archive omits archive_digest")
        payload = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "archive_digest"
        }
        archived_digest_array = np.asarray(archive["archive_digest"])
    archived_digest = (
        str(archived_digest_array.item())
        if archived_digest_array.shape == ()
        else ""
    )
    expected_digest = _digest_arrays(
        payload,
        {
            "contract": _ALIGNMENT_ARCHIVE_CONTRACT,
            "schema_version": 1,
        },
    )
    if archived_digest != expected_digest:
        raise ValueError("alignment archive digest does not match its payload")
    required = set(_ALIGNMENT_ARCHIVE_FIELDS)
    if set(payload) != required:
        missing = sorted(required - set(payload))
        extra = sorted(set(payload) - required)
        raise ValueError(
            "alignment archive fields differ from schema: "
            f"missing={missing}, extra={extra}"
        )
    if _alignment_archive_integer_scalar(payload, "schema_version") != 1:
        raise ValueError("unsupported alignment archive schema version")
    if _alignment_archive_boolean_scalar(
        payload, "initialization_structurally_trusted"
    ):
        raise ValueError("archived alignment initialization cannot be trusted")
    archived_problem_id = str(
        _alignment_archive_scalar(payload, "alignment_problem_id")
    )
    if archived_problem_id != problem.alignment_problem_id:
        raise ValueError("alignment archive and forward problem IDs differ")
    try:
        metadata = json.loads(
            str(_alignment_archive_scalar(payload, "initialization_metadata_json"))
        )
        options_payload = json.loads(
            str(_alignment_archive_scalar(payload, "options_json"))
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("alignment archive contains invalid JSON") from error
    if not isinstance(metadata, Mapping):
        raise ValueError("archived initialization metadata must be a JSON object")
    options = _alignment_options_from_payload(options_payload)
    if metadata.get("initialization_options") != _alignment_options_payload(
        options
    ):
        raise ValueError("archived metadata and alignment options differ")

    for name, observed, expected in (
        ("window_starts", scan.window_starts, problem.window_starts),
        ("scan_coordinates", scan.scan_coordinates, problem.scan_coordinates),
        ("detector_angles", scan.detector_angles, problem.detector_angles),
    ):
        observed_array = np.asarray(observed)
        expected_array = np.asarray(expected)
        if observed_array.shape != expected_array.shape or not np.allclose(
            observed_array,
            expected_array,
            rtol=8.0 * np.finfo(float).eps,
            atol=8.0 * np.finfo(float).eps,
        ):
            raise ValueError(f"scan {name} do not match the alignment problem")
    selection_data = build_alignment_selection_data_1d(
        scan,
        training_indices=problem.training_indices,
        validation_indices=problem.validation_indices,
        audit_indices=problem.audit_indices,
        guard_indices=problem.guard_indices,
        training_screen_scan_count=options.training_screen_scan_count,
    )
    archived_selection_data_id = str(
        _alignment_archive_scalar(payload, "summary_selection_data_id")
    )
    if selection_data.selection_data_id != archived_selection_data_id:
        raise ValueError("raw scan does not reproduce archived selection data")

    catalog = _catalog_from_alignment_archive(payload)
    catalog_id = alignment_candidate_catalog_id_1d(catalog, options=options)
    if catalog_id != str(
        _alignment_archive_scalar(payload, "summary_candidate_catalog_id")
    ):
        raise ValueError("archived candidate catalog ID is invalid")
    scores = _scores_from_alignment_archive(payload, catalog)
    recomputed_summary = select_alignment_candidate_1d(
        scores,
        selection_data_id=selection_data.selection_data_id,
        candidate_catalog_id=catalog_id,
        options=options,
    )
    summary_values_match = all(
        (
            recomputed_summary.minimum_loss_candidate_id
            == str(
                _alignment_archive_scalar(
                    payload, "summary_minimum_loss_candidate_id"
                )
            ),
            recomputed_summary.selected_candidate_id
            == str(
                _alignment_archive_scalar(
                    payload, "summary_selected_candidate_id"
                )
            ),
            recomputed_summary.equivalent_candidate_ids
            == tuple(
                str(value)
                for value in np.asarray(
                    payload["summary_equivalent_candidate_ids"]
                )
            ),
            recomputed_summary.unique_selection
            == _alignment_archive_boolean_scalar(
                payload, "summary_unique_selection"
            ),
            recomputed_summary.alignment_selection_id
            == str(
                _alignment_archive_scalar(
                    payload, "summary_alignment_selection_id"
                )
            ),
        )
    )
    if not summary_values_match:
        raise ValueError("archived alignment summary is not reproducible")

    training_source = np.asarray(selection_data.training_source_indices)
    training_local = np.asarray(selection_data.training_local_indices)
    validation_source = np.asarray(selection_data.validation_source_indices)
    validation_local = np.asarray(selection_data.validation_local_indices)
    selected_model: SiliconAlignmentModel1D | None = None
    for score in scores:
        model = rebuild_silicon_alignment_candidate_1d(
            problem.prior,
            score.candidate,
        )
        if model.candidate_model_id != score.candidate_model_id:
            raise ValueError("archived candidate model ID is not reproducible")
        training_prediction = simulate_glancing_scan_1d(
            model.lattice_model.reference_potential,
            np.asarray(problem.input_probes)[training_source],
            np.asarray(problem.window_starts)[training_source],
            problem.window_length,
            problem.propagation_kernel,
            problem.slice_thickness_A,
            problem.energy_eV,
            rematerialize=False,
        )
        validation_prediction = simulate_glancing_scan_1d(
            model.lattice_model.reference_potential,
            np.asarray(problem.input_probes)[validation_source],
            np.asarray(problem.window_starts)[validation_source],
            problem.window_length,
            problem.propagation_kernel,
            problem.slice_thickness_A,
            problem.energy_eV,
            rematerialize=False,
        )
        mask = selection_data.detector_valid_mask
        training_per_scan = _alignment_loss_per_scan(
            training_prediction,
            np.asarray(selection_data.intensities)[training_local],
            None if mask is None else np.asarray(mask)[training_local],
        )
        validation_per_scan = _alignment_loss_per_scan(
            validation_prediction,
            np.asarray(selection_data.intensities)[validation_local],
            None if mask is None else np.asarray(mask)[validation_local],
        )
        if not _alignment_evidence_is_close(
            score.training_screen_loss,
            np.mean(training_per_scan),
        ) or not _alignment_evidence_is_close(
            score.validation_loss_per_scan,
            validation_per_scan,
        ):
            raise ValueError("archived alignment losses are not reproducible")
        if score.candidate.candidate_id == recomputed_summary.selected_candidate_id:
            selected_model = model
    if selected_model is None or selected_model.candidate_model_id != str(
        _alignment_archive_scalar(payload, "selected_candidate_model_id")
    ):
        raise ValueError("archived selected model is not reproducible")

    n_scans = len(np.asarray(problem.window_starts))
    partitions = {
        name: _validated_partition(name, payload[name], n_scans)
        for name in (
            "training_screen_indices",
            "validation_indices",
            "audit_indices",
            "guard_indices",
        )
    }
    expected_partitions = {
        "training_screen_indices": np.asarray(
            selection_data.training_source_indices
        ),
        "validation_indices": np.asarray(problem.validation_indices),
        "audit_indices": np.asarray(problem.audit_indices),
        "guard_indices": np.asarray(problem.guard_indices),
    }
    if any(
        not np.array_equal(partitions[name], expected_partitions[name])
        for name in partitions
    ):
        raise ValueError("archived alignment partitions do not match the problem")
    selected_model = _bind_alignment_selection_to_model(
        selected_model,
        recomputed_summary,
    )
    return SiliconAlignmentInitialization1D(
        selected_model=selected_model,
        candidate_scores=scores,
        selection_summary=recomputed_summary,
        training_screen_indices=_readonly_copy(
            partitions["training_screen_indices"]
        ),
        validation_indices=_readonly_copy(partitions["validation_indices"]),
        audit_indices=_readonly_copy(partitions["audit_indices"]),
        guard_indices=_readonly_copy(partitions["guard_indices"]),
        alignment_problem_id=problem.alignment_problem_id,
        candidate_catalog=catalog,
        structurally_trusted=False,
        metadata=MappingProxyType(dict(metadata)),
    )


def geometry_stratified_training_subset_1d(
    training_indices: Any,
    scan_coordinates: Any,
    scan_count: int,
) -> np.ndarray:
    """Select coordinate-quantile midpoints using no diffraction values."""
    training = np.asarray(training_indices)
    coordinates = np.asarray(scan_coordinates)
    count = _positive_integer("scan_count", scan_count)
    if training.ndim != 1 or not training.size or not np.issubdtype(
        training.dtype, np.integer
    ):
        raise TypeError("training_indices must be a non-empty integer vector")
    training = training.astype(np.int64, copy=False)
    if np.unique(training).size != training.size:
        raise ValueError("training_indices must be unique")
    if coordinates.ndim != 1 or np.any(training < 0) or np.any(
        training >= coordinates.size
    ):
        raise ValueError("scan_coordinates and training_indices are incompatible")
    selected_coordinates = coordinates[training]
    if np.iscomplexobj(selected_coordinates) or np.any(
        ~np.isfinite(selected_coordinates)
    ):
        raise ValueError("training scan coordinates must be finite and real")
    if count >= training.size:
        return np.array(training, copy=True)
    ordered = training[np.lexsort((training, selected_coordinates))]
    ranks = np.floor(
        (np.arange(count, dtype=float) + 0.5) * ordered.size / count
    ).astype(np.int64)
    selected = np.ascontiguousarray(ordered[ranks], dtype=np.int64)
    if np.unique(selected).size != count:
        raise RuntimeError("geometry-stratified subset contains duplicate scans")
    return selected


def _validated_partition(
    name: str,
    values: Any,
    n_scans: int,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and not np.issubdtype(array.dtype, np.integer)):
        raise TypeError(f"{name} must be a one-dimensional integer sequence")
    array = array.astype(np.int64, copy=False)
    if np.unique(array).size != array.size or np.any(array < 0) or np.any(
        array >= n_scans
    ):
        raise ValueError(f"{name} must contain unique in-range indices")
    return array


def build_alignment_selection_data_1d(
    scan: GlancingScan1D,
    *,
    training_indices: Any,
    validation_indices: Any,
    audit_indices: Any = (),
    guard_indices: Any = (),
    training_screen_scan_count: int = 32,
) -> AlignmentSelectionData1D:
    """Copy only training-screen and validation rows across the search boundary."""
    if type(scan) is not GlancingScan1D:
        raise TypeError(
            "scan must be exactly a truth-free GlancingScan1D; truth-bearing "
            "dataset containers are not accepted"
        )
    intensities = np.asarray(scan.intensities)
    if intensities.ndim != 2 or not intensities.size or np.iscomplexobj(intensities):
        raise ValueError("scan.intensities must be a non-empty real 2D array")
    n_scans, n_detector = intensities.shape
    starts = np.asarray(scan.window_starts)
    coordinates = np.asarray(scan.scan_coordinates)
    angles = np.asarray(scan.detector_angles)
    if starts.shape != (n_scans,) or not np.issubdtype(starts.dtype, np.integer):
        raise ValueError("scan.window_starts must contain one integer per scan")
    if coordinates.shape != (n_scans,) or np.iscomplexobj(coordinates) or np.any(
        ~np.isfinite(coordinates)
    ):
        raise ValueError("scan.scan_coordinates must contain finite real values")
    if angles.shape != (n_detector,) or np.iscomplexobj(angles) or np.any(
        ~np.isfinite(angles)
    ):
        raise ValueError("scan.detector_angles must contain finite real values")

    partitions = {
        "training_indices": _validated_partition(
            "training_indices", training_indices, n_scans
        ),
        "validation_indices": _validated_partition(
            "validation_indices", validation_indices, n_scans
        ),
        "audit_indices": _validated_partition("audit_indices", audit_indices, n_scans),
        "guard_indices": _validated_partition("guard_indices", guard_indices, n_scans),
    }
    if not partitions["training_indices"].size or not partitions[
        "validation_indices"
    ].size:
        raise ValueError("alignment selection requires training and validation scans")
    names = tuple(partitions)
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1 :]:
            if np.intersect1d(
                partitions[first_name], partitions[second_name]
            ).size:
                raise ValueError(f"{first_name} and {second_name} must be disjoint")

    training_screen = geometry_stratified_training_subset_1d(
        partitions["training_indices"],
        coordinates,
        training_screen_scan_count,
    )
    validation = partitions["validation_indices"]
    source_indices = np.concatenate([training_screen, validation])
    selected_intensities = np.array(intensities[source_indices], copy=True)
    if scan.detector_valid_mask is None:
        selected_mask = None
        selected_values = selected_intensities
    else:
        complete_mask = np.asarray(scan.detector_valid_mask)
        if complete_mask.dtype != np.bool_ or complete_mask.shape != intensities.shape:
            raise ValueError("scan.detector_valid_mask must match intensities")
        selected_mask = np.array(complete_mask[source_indices], copy=True)
        if np.any(~np.any(selected_mask, axis=1)):
            raise ValueError("every selected scan must retain a valid detector pixel")
        selected_values = selected_intensities[selected_mask]
    if np.any(~np.isfinite(selected_values)) or np.any(selected_values < 0.0):
        raise ValueError("selected valid intensities must be finite and non-negative")

    n_training = training_screen.size
    training_local = np.arange(n_training, dtype=np.int64)
    validation_local = np.arange(n_training, source_indices.size, dtype=np.int64)
    arrays = {
        "intensities": selected_intensities,
        "window_starts": starts[source_indices],
        "scan_coordinates": coordinates[source_indices],
        "detector_angles": angles,
        "source_scan_indices": source_indices,
        "training_source_indices": training_screen,
        "validation_source_indices": validation,
        "training_local_indices": training_local,
        "validation_local_indices": validation_local,
    }
    if selected_mask is not None:
        arrays["detector_valid_mask"] = selected_mask
    selection_data_id = _digest_arrays(
        arrays,
        {
            "contract": _SELECTION_DATA_CONTRACT,
            "scope": "training_screen_plus_complete_validation_only",
        },
    )
    return AlignmentSelectionData1D(
        intensities=_readonly_copy(selected_intensities),
        detector_valid_mask=(
            None if selected_mask is None else _readonly_copy(selected_mask, dtype=bool)
        ),
        window_starts=_readonly_copy(starts[source_indices]),
        scan_coordinates=_readonly_copy(coordinates[source_indices]),
        detector_angles=_readonly_copy(angles),
        source_scan_indices=_readonly_copy(source_indices, dtype=np.int64),
        training_source_indices=_readonly_copy(training_screen, dtype=np.int64),
        validation_source_indices=_readonly_copy(validation, dtype=np.int64),
        training_local_indices=_readonly_copy(training_local, dtype=np.int64),
        validation_local_indices=_readonly_copy(validation_local, dtype=np.int64),
        selection_data_id=selection_data_id,
        metadata=MappingProxyType(
            {
                "scope": "foundation_only_training_and_validation",
                "audit_observations_stored": False,
                "guard_observations_stored": False,
                "scan_metadata_used": False,
                "structurally_trusted": False,
            }
        ),
    )


def _validated_score(score: AlignmentCandidateScore1D) -> AlignmentCandidateScore1D:
    if not isinstance(score, AlignmentCandidateScore1D):
        raise TypeError("scores must contain AlignmentCandidateScore1D instances")
    _validated_candidate(score.candidate)
    training_loss = _finite_scalar(
        "score.training_screen_loss", score.training_screen_loss
    )
    validation_loss = _finite_scalar("score.validation_loss", score.validation_loss)
    per_scan = np.asarray(score.validation_loss_per_scan)
    if per_scan.ndim != 1 or not per_scan.size or np.iscomplexobj(per_scan) or np.any(
        ~np.isfinite(per_scan)
    ):
        raise ValueError("validation_loss_per_scan must be a finite real vector")
    tolerance = 64.0 * np.finfo(np.result_type(per_scan.dtype, float)).eps
    if not np.isclose(
        validation_loss,
        float(np.mean(per_scan)),
        rtol=tolerance,
        atol=tolerance * max(1.0, float(np.max(np.abs(per_scan)))),
    ):
        raise ValueError("validation_loss must equal mean(validation_loss_per_scan)")
    if not isinstance(score.candidate_model_id, str) or not (
        score.candidate_model_id.strip()
    ):
        raise ValueError("candidate_model_id must be a non-empty string")
    del training_loss
    return score


def select_alignment_candidate_1d(
    scores: Sequence[AlignmentCandidateScore1D],
    *,
    selection_data_id: str,
    candidate_catalog_id: str,
    options: AlignmentInitializationOptions1D | None = None,
) -> AlignmentSelectionSummary1D:
    """Select by validation and preserve statistically equivalent candidates."""
    options = _validated_options(options)
    resolved = tuple(_validated_score(score) for score in scores)
    if not resolved:
        raise ValueError("scores must not be empty")
    if not isinstance(selection_data_id, str) or not selection_data_id.strip():
        raise ValueError("selection_data_id must be non-empty")
    if not isinstance(candidate_catalog_id, str) or not candidate_catalog_id.strip():
        raise ValueError("candidate_catalog_id must be non-empty")
    candidate_ids = [score.candidate.candidate_id for score in resolved]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("scores must contain unique candidate IDs")
    n_validation = len(np.asarray(resolved[0].validation_loss_per_scan))
    if any(
        len(np.asarray(score.validation_loss_per_scan)) != n_validation
        for score in resolved
    ):
        raise ValueError("all candidates must use the same validation scans")

    ordered = sorted(
        resolved,
        key=lambda score: (
            score.validation_loss,
            score.candidate.candidate_id,
        ),
    )
    best = ordered[0]
    best_per_scan = np.asarray(best.validation_loss_per_scan, dtype=float)
    equivalent: list[str] = []
    equivalence_evidence: dict[str, dict[str, float]] = {}
    for score in resolved:
        differences = np.asarray(score.validation_loss_per_scan, dtype=float) - best_per_scan
        mean_difference = float(np.mean(differences))
        standard_error = (
            float(np.std(differences, ddof=1) / np.sqrt(n_validation))
            if n_validation > 1
            else 0.0
        )
        band = max(
            float(options.validation_absolute_band),
            float(options.validation_relative_band)
            * max(abs(float(best.validation_loss)), 1.0),
            float(options.validation_equivalence_z) * standard_error,
        )
        if mean_difference <= band:
            equivalent.append(score.candidate.candidate_id)
        equivalence_evidence[score.candidate.candidate_id] = {
            "mean_paired_difference": mean_difference,
            "paired_standard_error": standard_error,
            "equivalence_band": band,
        }
    equivalent_ids = tuple(sorted(equivalent))
    selection_id = _digest_payload(
        {
            "contract": _SELECTION_CONTRACT,
            "selection_data_id": selection_data_id,
            "candidate_catalog_id": candidate_catalog_id,
            "ordered_scores": [
                {
                    "candidate_id": score.candidate.candidate_id,
                    "candidate_model_id": score.candidate_model_id,
                    "training_screen_loss": float(score.training_screen_loss),
                    "validation_loss": float(score.validation_loss),
                    "validation_loss_per_scan": np.asarray(
                        score.validation_loss_per_scan, dtype=float
                    ).tolist(),
                }
                for score in resolved
            ],
            "minimum_loss_candidate_id": best.candidate.candidate_id,
            "equivalent_candidate_ids": list(equivalent_ids),
            "equivalence_evidence": equivalence_evidence,
        }
    )
    return AlignmentSelectionSummary1D(
        minimum_loss_candidate_id=best.candidate.candidate_id,
        selected_candidate_id=best.candidate.candidate_id,
        equivalent_candidate_ids=equivalent_ids,
        unique_selection=len(equivalent_ids) == 1,
        candidate_catalog_id=candidate_catalog_id,
        selection_data_id=selection_data_id,
        alignment_selection_id=selection_id,
        structurally_trusted=False,
        metadata=MappingProxyType(
            {
                "scope": "paired_validation_alignment_selection",
                "selection_uses": "complete_validation_only_after_training_screen",
                "audit_used_for_selection": False,
                "equivalence_evidence": equivalence_evidence,
                "structural_trust_reason": (
                    "validation selection is not structural validation"
                ),
            }
        ),
    )
