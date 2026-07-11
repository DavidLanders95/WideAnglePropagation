"""Readable high-level workflow for the glancing ptychography notebook.

The numerical inverse methods remain in :mod:`ptychography_1d`.  This module
collects the experiment construction, matched synthetic data, comparison
baselines, and notebook visualization behind a small public API so the example
notebook can concentrate on the scientific sequence rather than bookkeeping.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace as dataclass_replace
import operator
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from .propagation_methods import (
    angular_spectrum_propagation_kernel_1d,
    energy2wavelength,
    interaction_constant,
)
from .ptychography_1d import (
    ConvergenceOptions1D,
    GlancingScan1D,
    GlancingSideviewCache1D,
    LatticeOptimizationOptions1D,
    LatticeSiteModel1D,
    LatticeSiteReconstruction1D,
    PotentialReconstruction1D,
    PtychographyMeasurement1D,
    PtychographyObjective1D,
    decompose_lattice_site_displacement_controls_1d,
    lattice_site_displacements_1d,
    prepare_lattice_site_reconstruction_1d,
    reconstruct_lattice_site_potential_1d,
    reconstruct_potential_1d,
    render_lattice_site_potential_1d,
    save_glancing_scan_1d,
    save_lattice_site_reconstruction_1d,
    save_potential_reconstruction_1d,
    simulate_glancing_scan_1d,
    simulate_glancing_sideview_cache_1d,
)
from .ptychography_atomic_validation_1d import (
    AtomicTemplateComparison1D,
    AtomicTemplateQuadratureOptions1D,
    IndependentSiAtomicTemplate1D,
    compare_si_atomic_template_1d,
    render_si_atomic_template_1d,
)
from .ptychography_ensemble_1d import (
    MultistartOptions1D,
    PreparedMultistartResult1D,
    PreparedMultistartRunOptions1D,
    run_prepared_lattice_site_multistart_1d,
)
from .ptychography_alignment_1d import (
    SiliconAlignmentForwardProblem1D,
    SiliconAlignmentPrior1D,
    make_silicon_alignment_forward_problem_1d,
    make_silicon_alignment_prior_1d,
)
from .ptychography_diagnostics_1d import (
    LatticeSiteSensitivityScreen1D,
    PoissonCountingModel1D,
    SensitivityScreenOptions1D,
    lattice_site_sensitivity_screen_1d,
)
from .ptychography_support_contract_1d import (
    LatticeSiteRole1D,
    LatticeSiteSupportContract1D,
    classify_lattice_site_support_1d,
    validate_lattice_site_support_contract_1d,
)
from .sideview_geometry import make_tilted_gaussian_beam_1d


__all__ = [
    "AtomicTemplateCertification1D",
    "GlancingDataset1D",
    "InteractionRegion1D",
    "ReconstructionOptions1D",
    "ScanPartition1D",
    "SiliconGlancingConfig1D",
    "SiliconGlancingExperiment1D",
    "build_silicon_glancing_experiment_1d",
    "build_silicon_alignment_prior_1d",
    "build_silicon_alignment_problem_1d",
    "gaussian_interaction_region_1d",
    "make_glancing_scan_viewer_1d",
    "plot_experiment_overview_1d",
    "plot_lattice_reconstruction_1d",
    "plot_lattice_sensitivity_screen_1d",
    "plot_reconstruction_comparison_1d",
    "reconstruct_experiment_1d",
    "reconstruct_lattice_multistart_experiment_1d",
    "reconstruction_metrics_1d",
    "save_experiment_results_1d",
    "save_lattice_reconstruction_gif_1d",
    "screen_lattice_reconstruction_sensitivity_1d",
    "simulate_experiment_1d",
    "stratified_scan_partition_1d",
]


Array = Any


@dataclass(frozen=True)
class AtomicTemplateCertification1D:
    """Numerical evidence that a compact atomic template contains its tails."""

    cutoff_A: float
    reference_cutoff_A: float
    relative_tail_l2: float
    tolerance: float
    candidate_errors: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class InteractionRegion1D:
    """Illuminated forward volume and stricter data-supported mutable volume."""

    forward_mask: Array
    nominal_forward_mask: Array
    reconstruction_mask: Array
    scan_coverage: Array
    forward_scan_coverage: Array
    peak_relative_intensity: Array
    intensity_threshold: float
    excluded_probe_power: float
    minimum_scan_coverage: int
    radius_A: float
    nominal_radius_A: float
    uncertainty_margin_A: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScanPartition1D:
    """Geometry-only train, validation, audit, and unused guard indices."""

    training_indices: Array
    validation_indices: Array
    audit_indices: Array
    guard_indices: Array
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SiliconGlancingConfig1D:
    """Physical and numerical choices for the matched silicon experiment."""

    energy_eV: float = 30_000.0
    glancing_angle_deg: float = 2.0
    beam_waist_A: float = 3.0
    slab_depth_A: float = 50.0
    vacuum_above_A: float = 100.0
    vacuum_below_A: float = 190.0
    window_length_A: float = 1_000.0
    sampling_u_A: float = 0.15
    sampling_s_A: float = 0.35
    si_lattice_A: float = 5.431
    scan_start_A: float = 400.0
    scan_stop_A: float = 600.0
    n_scans: int = 300
    defect_center_s_A: float = 500.0
    defect_width_sites: int = 10
    validation_stride: int = 10
    audit_fraction: float = 0.1
    audit_blocks: int = 3
    audit_guard_scans: int = 0
    beam_path_radius_waists: float = 3.0
    minimum_scan_coverage: int | None = None
    # ``auto`` derives the support from excluded probe power and scan overlap;
    # ``landing`` and ``beam_path`` retain the earlier manual policies.
    update_region: str = "auto"
    landing_radius_waists: float = 3.0
    landing_depth_A: float | None = None
    interaction_excluded_probe_power: float = 1e-5
    interaction_intensity_threshold: float | None = None
    beam_position_uncertainty_A: float = 0.0
    beam_angle_uncertainty_deg: float = 0.0
    atomic_template_cutoff_A: float | None = None
    atomic_template_tolerance: float = 1e-6
    atomic_template_amplitude_tolerance: float = 1e-4
    atomic_template_candidates_A: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 8.0)
    cutoff_check_A: float = 10.0
    exterior_material_policy: str = "parameterize_uncertain"
    fixed_exterior_provenance_id: str | None = None
    maximum_nuisance_sites: int = 4096
    maximum_specimen_parameters: int = 8192
    # Deprecated unsafe spelling. Supplying it now fails with migration advice.
    fixed_exterior_assumption: str | None = None
    maximum_displacement_A: float = 0.5
    displacement_control_spacing_A: float = 25.0
    displacement_control_spacing_u_A: float = 3.0


@dataclass(frozen=True)
class SiliconGlancingExperiment1D:
    """Complete known geometry and lattice model shared by both datasets."""

    config: SiliconGlancingConfig1D
    pristine_potential: Array
    lattice_model: LatticeSiteModel1D
    template_certification: AtomicTemplateCertification1D
    independent_kirkland_template: IndependentSiAtomicTemplate1D
    lobato_kirkland_template_comparison: AtomicTemplateComparison1D
    support_contract: LatticeSiteSupportContract1D
    interaction_region: InteractionRegion1D
    truth_potentials: Mapping[str, Array]
    truth_vacancy_fractions: Mapping[str, Array]
    truth_displacement_controls: Mapping[str, Array]
    truth_rigid_displacements: Mapping[str, Array]
    defect_site_indices: Mapping[str, Array]
    all_site_coordinates: Array
    variable_sites: Array
    target_sites: Array
    modeled_target_site_mask: Array
    modeled_nuisance_site_mask: Array
    site_selection_mask: Array
    reconstruction_mask: Array
    lattice_influence_mask: Array
    target_lattice_influence_mask: Array
    nuisance_lattice_influence_mask: Array
    beam_path_scan_coverage: Array
    input_probes: Array
    propagation_kernel: Array
    window_starts: Array
    window_length: int
    scan_coordinates: Array
    axial_coordinates: Array
    transverse_coordinates: Array
    detector_angles: Array
    training_indices: Array
    validation_indices: Array
    audit_indices: Array
    guard_indices: Array
    audit_site_scan_coverage: Array
    audit_site_scan_coverage_metadata: Mapping[str, Any]
    cutoff_check_potentials: Mapping[str, Array]
    template_stress_potential_pairs: Mapping[str, tuple[Array, Array]]
    axial_sampling: float
    transverse_sampling: float
    summary: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GlancingDataset1D:
    """One selected truth case, its diffraction data, and mismatch diagnostics."""

    case: str
    potential: Array
    scan: GlancingScan1D
    truth_vacancy_fractions: Array
    truth_displacement_controls: Array
    truth_rigid_displacement: Array
    zero_exterior_amplitude_nrmse: float
    template_cutoff_amplitude_nrmse: float
    template_cutoff_max_scan_amplitude_nrmse: float
    template_stress_worst_scan_amplitude_nrmse: float
    template_certified_worst_amplitude_nrmse: float
    kirkland_alternative_amplitude_nrmse: float
    kirkland_alternative_max_scan_amplitude_nrmse: float

    @property
    def intensities(self) -> Array:
        return self.scan.intensities


@dataclass(frozen=True)
class ReconstructionOptions1D:
    """Shared optimization controls for the three comparison methods."""

    pixel_updates: int = 4000
    lattice_updates: int = 500
    minibatch_size: int = 5
    validation_interval_pixels: int = 100
    validation_interval_lattice: int = 25
    training_diagnostic_scan_count: int | None = 32
    evaluation_batch_size: int = 10
    rematerialize: bool = True
    seed: int = 0
    progress: bool = True
    initial_site_offset_A: tuple[float, float] = (0.0, 0.0)
    initial_control_noise_A: float = 0.0
    separate_rigid_registration: bool = True
    maximum_rigid_displacement_A: float = 0.15
    maximum_residual_displacement_A: float = 0.35
    lattice_checkpoint_interval: int | None = None
    lattice_convergence: ConvergenceOptions1D = field(
        default_factory=ConvergenceOptions1D
    )
    lattice_optimization: LatticeOptimizationOptions1D = field(
        default_factory=lambda: LatticeOptimizationOptions1D(mode="staged")
    )


def stratified_scan_partition_1d(
    n_scans: int,
    *,
    validation_stride: int,
    audit_fraction: float = 0.1,
    audit_blocks: int = 3,
    audit_guard_scans: int = 0,
) -> ScanPartition1D:
    """Reserve distributed contiguous audit blocks using geometry only.

    The split is fixed before any diffraction values are inspected. Guard
    scans are unused by optimization or assessment; they can reduce direct
    overlap around audit blocks when the acquisition geometry permits it.
    """
    n_scan = operator.index(n_scans)
    stride = operator.index(validation_stride)
    n_blocks_requested = operator.index(audit_blocks)
    guard_width = operator.index(audit_guard_scans)
    fraction = float(audit_fraction)
    if n_scan < 3 or stride < 1:
        raise ValueError("n_scans must be at least three and validation_stride positive")
    if not np.isfinite(fraction) or not 0.0 < fraction < 0.5:
        raise ValueError("audit_fraction must lie strictly between zero and 0.5")
    if n_blocks_requested < 1 or guard_width < 0:
        raise ValueError(
            "audit_blocks must be positive and audit_guard_scans non-negative"
        )

    audit_count = max(1, int(round(fraction * n_scan)))
    if audit_count >= n_scan - 1:
        raise ValueError("audit_fraction must leave scans outside the audit set")
    n_blocks = min(n_blocks_requested, audit_count)
    block_lengths = np.full(n_blocks, audit_count // n_blocks, dtype=int)
    block_lengths[: audit_count % n_blocks] += 1
    centers = np.linspace(0.0, n_scan - 1.0, n_blocks + 2)[1:-1]
    audit_parts = []
    block_bounds = []
    for center, block_length in zip(centers, block_lengths):
        start = int(round(center - 0.5 * (block_length - 1)))
        start = min(max(start, 0), n_scan - int(block_length))
        stop = start + int(block_length)
        audit_parts.append(np.arange(start, stop, dtype=np.int32))
        block_bounds.append((start, stop))
    audit = np.unique(np.concatenate(audit_parts)).astype(np.int32)
    if audit.size != audit_count:
        raise RuntimeError(
            "audit blocks overlap; reduce audit_fraction or audit_blocks"
        )
    block_starts = np.r_[0, np.flatnonzero(np.diff(audit) > 1) + 1]
    block_stops = np.r_[block_starts[1:], audit.size]
    effective_block_bounds = [
        (int(audit[start]), int(audit[stop - 1]) + 1)
        for start, stop in zip(block_starts, block_stops)
    ]

    guard = []
    for start, stop in block_bounds:
        guard.extend(range(max(0, start - guard_width), start))
        guard.extend(range(stop, min(n_scan, stop + guard_width)))
    guard_indices = np.setdiff1d(
        np.unique(np.asarray(guard, dtype=np.int32)), audit, assume_unique=True
    ).astype(np.int32)
    validation = np.arange(0, n_scan, stride, dtype=np.int32)
    validation = np.setdiff1d(validation, audit, assume_unique=True)
    validation = np.setdiff1d(validation, guard_indices, assume_unique=True).astype(
        np.int32
    )
    held_out = np.concatenate([validation, audit, guard_indices])
    training = np.setdiff1d(
        np.arange(n_scan, dtype=np.int32), held_out, assume_unique=True
    ).astype(np.int32)
    if not validation.size or not training.size:
        raise ValueError(
            "scan partition leaves no validation or training scans; reduce audit "
            "fraction/guards or validation density"
        )
    return ScanPartition1D(
        training_indices=jnp.asarray(training),
        validation_indices=jnp.asarray(validation),
        audit_indices=jnp.asarray(audit),
        guard_indices=jnp.asarray(guard_indices),
        metadata={
            "construction": "geometry_only_stratified_contiguous_blocks",
            "audit_fraction_requested": fraction,
            "audit_blocks_requested": n_blocks_requested,
            "audit_blocks_placed": n_blocks,
            "audit_blocks_used": len(effective_block_bounds),
            "audit_block_bounds_stop_exclusive": effective_block_bounds,
            "audit_guard_scans": guard_width,
        },
    )


def _control_axis(values: np.ndarray, spacing: float) -> np.ndarray:
    start = np.floor(np.min(values) / spacing) * spacing
    stop = np.ceil(np.max(values) / spacing) * spacing
    axis = np.arange(start, stop + 0.5 * spacing, spacing)
    if axis.size == 1:
        axis = np.asarray([start, start + spacing])
    return axis


def _projected_si_sites(
    unit,
    *,
    lattice_A: float,
    length_A: float,
    depth_A: float,
) -> np.ndarray:
    projected_basis = unit.positions[:, [2, 0]]
    top_x_A = float(np.max(unit.positions[:, 0]))
    sites = []
    n_s_cells = int(np.ceil(length_A / lattice_A))
    n_u_cells = int(np.ceil(depth_A / lattice_A))
    for cell_s in range(-1, n_s_cells + 2):
        for cell_u in range(-n_u_cells - 2, 2):
            for basis_s_A, basis_x_A in projected_basis:
                site_s_A = float(basis_s_A + cell_s * lattice_A)
                site_u_A = float(basis_x_A - top_x_A + cell_u * lattice_A)
                if 0.0 <= site_s_A < length_A and -depth_A <= site_u_A <= 0.0:
                    sites.append((site_s_A, site_u_A))
    return np.unique(np.round(np.asarray(sites), decimals=10), axis=0)


def _projected_si_template(
    config: SiliconGlancingConfig1D,
    *,
    ds: float,
    du: float,
    cutoff_A: float,
) -> tuple[np.ndarray, tuple[int, int]]:
    import abtem
    from ase import Atoms

    padded_radius_A = cutoff_A + config.maximum_displacement_A
    half_s = int(np.ceil(padded_radius_A / ds))
    half_u = int(np.ceil(padded_radius_A / du))
    n_patch_s = 2 * half_s + 1
    n_patch_u = 2 * half_u + 1
    cell = np.diag([n_patch_u * du, config.si_lattice_A, n_patch_s * ds])
    isolated_si = Atoms(
        "Si",
        positions=[[half_u * du, 0.5 * config.si_lattice_A, half_s * ds]],
        cell=cell,
        pbc=[False, True, False],
    )
    builder = abtem.Potential(
        isolated_si,
        gpts=(n_patch_u, n_patch_s),
        slice_thickness=config.si_lattice_A,
        projection="finite",
        parametrization="lobato",
        plane="xz",
        periodic=False,
        device="cpu",
    )
    # abTEM stores the two projected axes as (u, s).
    template = np.asarray(builder.build(lazy=False).array)[0].T / config.si_lattice_A
    return template, (half_s, half_u)


def _certified_projected_si_template(
    config: SiliconGlancingConfig1D,
    *,
    ds: float,
    du: float,
) -> tuple[
    np.ndarray,
    tuple[int, int],
    AtomicTemplateCertification1D,
]:
    """Select or verify a cutoff against a larger isolated-atom template."""
    tolerance = float(config.atomic_template_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("atomic_template_tolerance must be finite and positive")
    amplitude_tolerance = float(config.atomic_template_amplitude_tolerance)
    if not np.isfinite(amplitude_tolerance) or amplitude_tolerance <= 0.0:
        raise ValueError(
            "atomic_template_amplitude_tolerance must be finite and positive"
        )
    candidates = np.asarray(config.atomic_template_candidates_A, dtype=float)
    if (
        candidates.ndim != 1
        or candidates.size == 0
        or np.any(~np.isfinite(candidates))
        or np.any(candidates <= 0.0)
        or np.any(np.diff(candidates) <= 0.0)
    ):
        raise ValueError(
            "atomic_template_candidates_A must be finite, positive, and increasing"
        )
    requested = config.atomic_template_cutoff_A
    if requested is not None and (not np.isfinite(requested) or requested <= 0.0):
        raise ValueError("atomic_template_cutoff_A must be positive or None")
    reference_cutoff = max(
        float(config.cutoff_check_A),
        float(candidates[-1]),
        0.0 if requested is None else float(requested),
    )
    if reference_cutoff <= 0.0:
        raise ValueError("cutoff_check_A must be positive")

    reference, (reference_half_s, reference_half_u) = _projected_si_template(
        config, ds=ds, du=du, cutoff_A=reference_cutoff
    )
    relative_norm = max(float(np.linalg.norm(reference)), np.finfo(float).tiny)
    grid_s = (np.arange(reference.shape[0]) - reference_half_s) * ds
    grid_u = (np.arange(reference.shape[1]) - reference_half_u) * du
    radius = np.sqrt(grid_s[:, None] ** 2 + grid_u[None, :] ** 2)

    def tail_error(cutoff_A: float) -> float:
        return float(np.linalg.norm(reference[radius > cutoff_A]) / relative_norm)

    candidate_errors = {
        f"{candidate:g}": tail_error(float(candidate)) for candidate in candidates
    }
    if requested is None:
        eligible = [
            float(candidate)
            for candidate in candidates
            if tail_error(float(candidate)) <= tolerance
        ]
        if not eligible:
            raise ValueError(
                "no atomic template candidate meets atomic_template_tolerance; "
                "extend atomic_template_candidates_A or cutoff_check_A"
            )
        selected = eligible[0]
    else:
        selected = float(requested)
    selected_error = tail_error(selected)
    if selected_error > tolerance:
        raise ValueError(
            f"atomic template cutoff {selected:g} A has relative tail L2 "
            f"{selected_error:.3g}, above tolerance {tolerance:.3g}"
        )

    half_s = int(np.ceil((selected + config.maximum_displacement_A) / ds))
    half_u = int(np.ceil((selected + config.maximum_displacement_A) / du))
    template = reference[
        reference_half_s - half_s : reference_half_s + half_s + 1,
        reference_half_u - half_u : reference_half_u + half_u + 1,
    ]
    expected_shape = (2 * half_s + 1, 2 * half_u + 1)
    if template.shape != expected_shape:
        raise ValueError("cutoff_check_A is too small to pad the selected template")
    certification = AtomicTemplateCertification1D(
        cutoff_A=selected,
        reference_cutoff_A=reference_cutoff,
        relative_tail_l2=selected_error,
        tolerance=tolerance,
        candidate_errors=candidate_errors,
    )
    return template, (half_s, half_u), certification


def _atomic_parameterization_diagnostic_metadata(
    template: IndependentSiAtomicTemplate1D,
    comparison: AtomicTemplateComparison1D,
) -> dict[str, Any]:
    """Return JSON-ready, explicitly non-certifying comparison evidence."""
    return {
        "diagnostic": "direct_Kirkland_vs_production_Lobato_same_template_grid",
        "candidate_parameterization": "Kirkland independent-atom model",
        "reference_parameterization": "Lobato independent-atom model",
        "raw_relative_l2": comparison.raw_relative_l2,
        "scale_adjusted_shape_relative_l2": (
            comparison.scale_adjusted_shape_relative_l2
        ),
        "optimal_candidate_scale": comparison.optimal_candidate_scale,
        "candidate_to_reference_peak_ratio": comparison.peak_ratio,
        "candidate_to_reference_integral_ratio": comparison.integral_ratio,
        "candidate_template_sha256": comparison.candidate_template_sha256,
        "reference_template_sha256": comparison.reference_template_sha256,
        "comparison_sha256": comparison.comparison_sha256,
        "quadrature_options_sha256": template.options.options_sha256,
        "trust_claim": False,
        "trust_reason": comparison.trust_reason,
        "has_acceptance_threshold": False,
        "used_for_cutoff_certification": False,
        "limitations": list(comparison.limitations),
    }


def _patches_for_sites(
    site_coordinates: np.ndarray,
    template: np.ndarray,
    half_shape: tuple[int, int],
    *,
    s_A: np.ndarray,
    u_A: np.ndarray,
    ds: float,
    du: float,
    material_u_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import shift as shift_image

    half_s, half_u = half_shape
    patches = []
    starts = []
    shifted_templates: dict[tuple[float, float], np.ndarray] = {}
    for site_s_A, site_u_A in np.asarray(site_coordinates):
        site_s_pixel = (site_s_A - s_A[0]) / ds
        site_u_pixel = (site_u_A - u_A[0]) / du
        center_s = int(np.rint(site_s_pixel))
        center_u = int(np.rint(site_u_pixel))
        fractional_shift = (site_s_pixel - center_s, site_u_pixel - center_u)
        key = tuple(float(np.round(value, 10)) for value in fractional_shift)
        if key not in shifted_templates:
            shifted_templates[key] = shift_image(
                template,
                shift=fractional_shift,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
        shifted = shifted_templates[key].copy()
        start_s = center_s - half_s
        start_u = center_u - half_u
        if material_u_mask is not None:
            global_u_indices = start_u + np.arange(template.shape[1])
            valid_u = (global_u_indices >= 0) & (global_u_indices < len(u_A))
            clipped_u = np.clip(global_u_indices, 0, len(u_A) - 1)
            valid_u &= material_u_mask[clipped_u]
            shifted[:, ~valid_u] = 0.0
        patches.append(shifted)
        starts.append((start_s, start_u))
    return np.asarray(patches), np.asarray(starts, dtype=np.int32)


def _finite_reference_potential(
    site_coordinates: np.ndarray,
    template: np.ndarray,
    half_shape: tuple[int, int],
    *,
    s_A: np.ndarray,
    u_A: np.ndarray,
    ds: float,
    du: float,
    batch_size: int = 512,
) -> np.ndarray:
    """Render a finite independent-atom slab without clipping vacuum tails."""
    reference = np.zeros((len(s_A), len(u_A)), dtype=template.dtype)
    for begin in range(0, len(site_coordinates), batch_size):
        patches, starts = _patches_for_sites(
            site_coordinates[begin : begin + batch_size],
            template,
            half_shape,
            s_A=s_A,
            u_A=u_A,
            ds=ds,
            du=du,
            material_u_mask=None,
        )
        for patch, (start_s, start_u) in zip(patches, starts):
            source_s_start = max(-int(start_s), 0)
            source_u_start = max(-int(start_u), 0)
            source_s_stop = min(patch.shape[0], len(s_A) - int(start_s))
            source_u_stop = min(patch.shape[1], len(u_A) - int(start_u))
            if source_s_start >= source_s_stop or source_u_start >= source_u_stop:
                continue
            target_s = slice(
                int(start_s) + source_s_start,
                int(start_s) + source_s_stop,
            )
            target_u = slice(
                int(start_u) + source_u_start,
                int(start_u) + source_u_stop,
            )
            reference[target_s, target_u] += patch[
                source_s_start:source_s_stop,
                source_u_start:source_u_stop,
            ]
    return reference


def _lattice_parameter_update_mask(
    patches: np.ndarray,
    patch_starts: np.ndarray,
    *,
    potential_shape: tuple[int, int],
    maximum_displacement_A: float,
    ds: float,
    du: float,
) -> np.ndarray:
    """Return every potential pixel reachable by the selected site parameters."""
    if maximum_displacement_A < 0.0 or ds <= 0.0 or du <= 0.0:
        raise ValueError("displacement bound and sampling must be non-negative/positive")
    mask = np.zeros(potential_shape, dtype=bool)
    for patch, (start_s, start_u) in zip(patches, patch_starts):
        # The differentiable renderer scatters the complete, already padded
        # patch. Four-sample Keys cubic translation can make an edge sample
        # nonzero even when that sample is exactly zero in the pristine patch,
        # so thresholding the pristine values would understate the true
        # influence region.
        local_s, local_u = np.indices(np.asarray(patch).shape)
        local_s = local_s.ravel()
        local_u = local_u.ravel()
        global_s = int(start_s) + local_s
        global_u = int(start_u) + local_u
        valid = (
            (global_s >= 0)
            & (global_s < potential_shape[0])
            & (global_u >= 0)
            & (global_u < potential_shape[1])
        )
        mask[global_s[valid], global_u[valid]] = True
    return mask


def _beam_path_region(
    config: SiliconGlancingConfig1D,
    s_A: np.ndarray,
    u_A: np.ndarray,
    scan_coordinates_A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    material = (u_A >= -config.slab_depth_A) & (u_A <= 0.0)
    coverage = np.zeros((len(s_A), len(u_A)), dtype=np.int32)
    tilt = -np.deg2rad(config.glancing_angle_deg)
    radius_A = config.beam_path_radius_waists * config.beam_waist_A
    for landing_A in scan_coordinates_A:
        centerline_A = np.tan(tilt) * (s_A - landing_A)
        distance_A = np.abs(u_A[None, :] - centerline_A[:, None]) * abs(np.cos(tilt))
        coverage += (distance_A <= radius_A) & material[None, :]
    minimum_coverage = (
        1 if config.minimum_scan_coverage is None else config.minimum_scan_coverage
    )
    mask = material[None, :] & (coverage >= minimum_coverage)
    return mask, coverage


def gaussian_interaction_region_1d(
    axial_coordinates_A: Array,
    transverse_coordinates_A: Array,
    scan_coordinates_A: Array,
    *,
    beam_waist_A: float,
    beam_tilt_rad: float,
    slab_bottom_A: float,
    slab_top_A: float = 0.0,
    excluded_probe_power: float = 1e-5,
    intensity_threshold: float | None = None,
    minimum_scan_coverage: int | None = None,
    beam_position_uncertainty_A: float = 0.0,
    beam_angle_uncertainty_rad: float = 0.0,
    mutable_scan_indices: Array | None = None,
) -> InteractionRegion1D:
    """Derive forward and trainable volumes from incident-beam geometry.

    The forward mask is the union of all beam paths after expanding them by
    the declared geometry uncertainty.  The reconstruction mask is deliberately
    stricter: it uses the nominal beam geometry and only the scans named by
    ``mutable_scan_indices``.  Thus validation, audit, or guard measurements
    cannot make an otherwise unsupported specimen value trainable.
    """
    s_A = np.asarray(axial_coordinates_A, dtype=float)
    u_A = np.asarray(transverse_coordinates_A, dtype=float)
    if s_A.ndim != 1 or u_A.ndim != 1 or not len(s_A) or not len(u_A):
        raise ValueError("axial and transverse coordinates must be non-empty 1D arrays")
    if np.any(~np.isfinite(s_A)) or np.any(~np.isfinite(u_A)):
        raise ValueError("axial and transverse coordinates must be finite")
    omitted_power = float(excluded_probe_power)
    if not np.isfinite(omitted_power) or not 0.0 < omitted_power < 1.0:
        raise ValueError("excluded_probe_power must lie strictly in (0, 1)")
    scans = np.asarray(scan_coordinates_A, dtype=float)
    if scans.ndim != 1 or scans.size < 2 or np.any(np.diff(scans) <= 0.0):
        raise ValueError("scan coordinates must be a strictly increasing 1D array")
    if mutable_scan_indices is None:
        mutable_indices = np.arange(scans.size, dtype=np.int32)
    else:
        mutable_indices = np.asarray(mutable_scan_indices)
        if mutable_indices.ndim != 1 or not mutable_indices.size:
            raise ValueError("mutable_scan_indices must be a non-empty 1D array")
        if not np.issubdtype(mutable_indices.dtype, np.integer):
            raise TypeError("mutable_scan_indices must contain integers")
        mutable_indices = mutable_indices.astype(np.int32, copy=False)
        if (
            np.any(mutable_indices < 0)
            or np.any(mutable_indices >= scans.size)
            or np.unique(mutable_indices).size != mutable_indices.size
        ):
            raise ValueError(
                "mutable_scan_indices must be unique and lie within the scan array"
            )
        mutable_indices = np.sort(mutable_indices)
    mutable_scans = scans[mutable_indices]
    if minimum_scan_coverage is None:
        minimum_coverage = min(
            mutable_scans.size,
            max(2, int(np.ceil(0.01 * mutable_scans.size))),
        )
    else:
        minimum_coverage = int(minimum_scan_coverage)
        if minimum_coverage < 1 or minimum_coverage > mutable_scans.size:
            raise ValueError(
                "minimum_scan_coverage must lie between 1 and the number of "
                "mutable scans"
            )

    waist_A = float(beam_waist_A)
    if not np.isfinite(waist_A) or waist_A <= 0.0:
        raise ValueError("beam_waist_A must be finite and positive")
    if intensity_threshold is None:
        from scipy.special import erfcinv

        gaussian_radius_A = waist_A * float(erfcinv(omitted_power))
        threshold = float(np.exp(-((gaussian_radius_A / waist_A) ** 2)))
    else:
        threshold = float(intensity_threshold)
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError(
                "interaction_intensity_threshold must lie strictly in (0, 1)"
            )
        gaussian_radius_A = waist_A * np.sqrt(-np.log(threshold))
        from scipy.special import erfc

        omitted_power = float(erfc(gaussian_radius_A / waist_A))
    position_uncertainty_A = float(beam_position_uncertainty_A)
    angle_uncertainty_rad = float(beam_angle_uncertainty_rad)
    if (
        not np.isfinite(position_uncertainty_A)
        or position_uncertainty_A < 0.0
        or not np.isfinite(angle_uncertainty_rad)
        or angle_uncertainty_rad < 0.0
    ):
        raise ValueError("beam position and angle uncertainties must be non-negative")
    maximum_path_A = max(
        abs(float(s_A[0]) - float(scans[-1])),
        abs(float(s_A[-1]) - float(scans[0])),
    )
    uncertainty_margin_A = position_uncertainty_A + maximum_path_A * np.tan(
        angle_uncertainty_rad
    )
    # ``radius_A`` is retained as a scalar worst-case summary for backwards
    # compatibility. The forward mask below uses a spatially local angular
    # envelope and does not apply this maximum margin everywhere.
    radius_A = gaussian_radius_A + uncertainty_margin_A
    tilt = float(beam_tilt_rad)
    if not np.isfinite(tilt):
        raise ValueError("beam_tilt_rad must be finite")
    if (
        angle_uncertainty_rad > 0.0
        and angle_uncertainty_rad >= abs(tilt) - 1e-12
    ):
        raise ValueError(
            "beam angle uncertainty reaches a surface-parallel ray; provide "
            "a tighter calibrated bound or a finite-aperture ray model"
        )
    sine = abs(np.sin(tilt))
    tangent = np.tan(tilt)
    bottom_A = float(slab_bottom_A)
    top_A = float(slab_top_A)
    if not np.isfinite(bottom_A) or not np.isfinite(top_A) or bottom_A >= top_A:
        raise ValueError("slab bounds must be finite with bottom below top")
    material = (u_A >= bottom_A) & (u_A <= top_A)

    def scan_geometry(
        selected_scans: np.ndarray, support_radius_A: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if sine < 1e-12:
            distance = np.broadcast_to(
                np.abs(u_A - top_A)[None, :], (len(s_A), len(u_A))
            )
            active = distance <= support_radius_A
            selected_coverage = np.where(
                active, selected_scans.size, 0
            ).astype(np.int32)
            return distance, selected_coverage
        equivalent_landing = s_A[:, None] - u_A[None, :] / tangent
        insertion = np.searchsorted(selected_scans, equivalent_landing)
        left = selected_scans[
            np.clip(insertion - 1, 0, selected_scans.size - 1)
        ]
        right = selected_scans[np.clip(insertion, 0, selected_scans.size - 1)]
        nearest = np.where(
            np.abs(equivalent_landing - left)
            <= np.abs(equivalent_landing - right),
            left,
            right,
        )
        distance = sine * np.abs(equivalent_landing - nearest)
        landing_half_width = support_radius_A / sine
        lower = np.searchsorted(
            selected_scans, equivalent_landing - landing_half_width
        )
        upper = np.searchsorted(
            selected_scans,
            equivalent_landing + landing_half_width,
            side="right",
        )
        return distance, (upper - lower).astype(np.int32)

    def uncertain_scan_coverage(selected_scans: np.ndarray) -> np.ndarray:
        """Count scans whose bounded ray family can reach each grid point."""
        support_radius_A = gaussian_radius_A + position_uncertainty_A
        if angle_uncertainty_rad == 0.0 and abs(np.sin(tilt)) < 1e-12:
            return scan_geometry(selected_scans, support_radius_A)[1]
        angle_bounds = (
            tilt - angle_uncertainty_rad,
            tilt + angle_uncertainty_rad,
        )
        lower_landing = np.full((len(s_A), len(u_A)), np.inf)
        upper_landing = np.full((len(s_A), len(u_A)), -np.inf)
        for bounded_tilt in angle_bounds:
            bounded_sine = abs(np.sin(bounded_tilt))
            bounded_tangent = np.tan(bounded_tilt)
            if bounded_sine < 1e-12 or abs(bounded_tangent) < 1e-12:
                raise ValueError(
                    "beam uncertainty contains an unsupported parallel ray"
                )
            equivalent_landing = (
                s_A[:, None] - u_A[None, :] / bounded_tangent
            )
            landing_half_width = support_radius_A / bounded_sine
            lower_landing = np.minimum(
                lower_landing,
                equivalent_landing - landing_half_width,
            )
            upper_landing = np.maximum(
                upper_landing,
                equivalent_landing + landing_half_width,
            )
        lower = np.searchsorted(selected_scans, lower_landing)
        upper = np.searchsorted(selected_scans, upper_landing, side="right")
        return (upper - lower).astype(np.int32)

    forward_coverage = uncertain_scan_coverage(scans)
    nominal_forward_distance, _ = scan_geometry(scans, gaussian_radius_A)
    mutable_distance, coverage = scan_geometry(
        mutable_scans, gaussian_radius_A
    )
    peak_intensity = np.exp(-((mutable_distance / waist_A) ** 2))
    peak_intensity = np.where(material[None, :], peak_intensity, 0.0)
    coverage = np.where(material[None, :], coverage, 0).astype(np.int32)
    forward_coverage = np.where(
        material[None, :], forward_coverage, 0
    ).astype(np.int32)
    forward_mask = material[None, :] & (forward_coverage > 0)
    nominal_forward_mask = material[None, :] & (
        nominal_forward_distance <= gaussian_radius_A
    )
    reconstruction_mask = (
        material[None, :]
        & (mutable_distance <= gaussian_radius_A)
        & (coverage >= minimum_coverage)
    )
    if not np.any(reconstruction_mask):
        raise ValueError(
            "the automatic interaction region is empty; the scan overlap is "
            "insufficient for minimum_scan_coverage"
        )
    return InteractionRegion1D(
        forward_mask=jnp.asarray(forward_mask),
        nominal_forward_mask=jnp.asarray(nominal_forward_mask),
        reconstruction_mask=jnp.asarray(reconstruction_mask),
        scan_coverage=jnp.asarray(coverage),
        forward_scan_coverage=jnp.asarray(forward_coverage),
        peak_relative_intensity=jnp.asarray(peak_intensity),
        intensity_threshold=threshold,
        excluded_probe_power=omitted_power,
        minimum_scan_coverage=minimum_coverage,
        radius_A=float(radius_A),
        nominal_radius_A=float(gaussian_radius_A),
        uncertainty_margin_A=float(uncertainty_margin_A),
        metadata={
            "construction": "Gaussian ray geometry",
            "intensity_definition": "peak relative incident intensity",
            "excluded_probe_power_definition": "two-sided 1D Gaussian intensity",
            "gaussian_radius_A": float(gaussian_radius_A),
            "nominal_mutable_radius_A": float(gaussian_radius_A),
            "uncertainty_expanded_forward_radius_A": float(radius_A),
            "uncertainty_expansion": "spatially_local_bounded_ray_envelope",
            "scalar_radius_role": "worst_case_summary_only",
            "angle_interval_deg": [
                float(np.rad2deg(tilt - angle_uncertainty_rad)),
                float(np.rad2deg(tilt + angle_uncertainty_rad)),
            ],
            "beam_tilt_deg": float(np.rad2deg(tilt)),
            "scan_coordinate_count": int(scans.size),
            "mutable_scan_count": int(mutable_scans.size),
            "mutable_scan_indices": mutable_indices.tolist(),
            "mutable_support": "nominal_geometry_training_scans_only",
            "forward_support": "all_scans_with_declared_geometry_uncertainty",
            "forward_support_role": (
                "fixed_forward_model_and_geometry_nuisance_support"
            ),
            "geometry_uncertainty_expands_mutable_support": False,
        },
    )


def _automatic_interaction_region(
    config: SiliconGlancingConfig1D,
    s_A: np.ndarray,
    u_A: np.ndarray,
    scan_coordinates_A: np.ndarray,
    mutable_scan_indices: np.ndarray,
) -> InteractionRegion1D:
    return gaussian_interaction_region_1d(
        s_A,
        u_A,
        scan_coordinates_A,
        beam_waist_A=config.beam_waist_A,
        beam_tilt_rad=-np.deg2rad(config.glancing_angle_deg),
        slab_bottom_A=-config.slab_depth_A,
        excluded_probe_power=config.interaction_excluded_probe_power,
        intensity_threshold=config.interaction_intensity_threshold,
        minimum_scan_coverage=config.minimum_scan_coverage,
        beam_position_uncertainty_A=config.beam_position_uncertainty_A,
        beam_angle_uncertainty_rad=np.deg2rad(
            config.beam_angle_uncertainty_deg
        ),
        mutable_scan_indices=mutable_scan_indices,
    )


def _landing_region(
    config: SiliconGlancingConfig1D,
    s_A: np.ndarray,
    u_A: np.ndarray,
    scan_coordinates_A: np.ndarray,
) -> np.ndarray:
    """Return a conservative shallow support around the scanned surface landings."""
    radius_A = config.landing_radius_waists * config.beam_waist_A
    depth_A = radius_A if config.landing_depth_A is None else config.landing_depth_A
    if radius_A <= 0.0 or depth_A <= 0.0:
        raise ValueError("landing-region radius and depth must be positive")
    nearest_landing_A = np.min(
        np.abs(s_A[:, None] - scan_coordinates_A[None, :]), axis=1
    )
    return (
        (nearest_landing_A[:, None] <= radius_A)
        & (u_A[None, :] <= 0.0)
        & (u_A[None, :] >= -depth_A)
    )


def _truth_controls(
    control_s_A: np.ndarray,
    control_u_A: np.ndarray,
    defect_center_s_A: float,
    maximum_displacement_A: float,
) -> Mapping[str, np.ndarray]:
    control_s, control_u = np.meshgrid(control_s_A, control_u_A, indexing="ij")
    envelope = np.exp(
        -0.5 * ((control_s - defect_center_s_A) / 75.0) ** 2
        - 0.5 * ((control_u + 15.0) / 20.0) ** 2
    )
    axial = envelope / max(float(np.max(np.abs(envelope))), 1e-12)
    transverse = ((control_s - defect_center_s_A) / 75.0) * envelope
    transverse /= max(float(np.max(np.abs(transverse))), 1e-12)
    strained = np.stack(
        [
            min(0.25, maximum_displacement_A) * axial,
            min(0.15, maximum_displacement_A) * transverse,
        ],
        axis=-1,
    )

    # A two-lobed, depth-dependent field is deliberately less compatible with
    # one affine deformation while remaining smooth on the control grid.
    left = np.exp(
        -0.5 * ((control_s - defect_center_s_A + 42.0) / 48.0) ** 2
        - 0.5 * ((control_u + 3.0) / 5.0) ** 2
    )
    right = np.exp(
        -0.5 * ((control_s - defect_center_s_A - 38.0) / 32.0) ** 2
        - 0.5 * ((control_u + 7.0) / 4.0) ** 2
    )
    depth_shear = ((control_u + 4.0) / 8.0) * np.exp(
        -0.5 * ((control_s - defect_center_s_A) / 65.0) ** 2
        - 0.5 * ((control_u + 5.0) / 7.0) ** 2
    )
    hard_axial = left - 0.8 * right + 0.35 * depth_shear
    hard_transverse = (
        ((control_s - defect_center_s_A + 42.0) / 48.0) * left
        - 0.7 * ((control_s - defect_center_s_A - 38.0) / 32.0) * right
        + 0.3 * depth_shear
    )
    hard_axial /= max(float(np.max(np.abs(hard_axial))), 1e-12)
    hard_transverse /= max(float(np.max(np.abs(hard_transverse))), 1e-12)
    hard = np.stack(
        [
            min(0.35, maximum_displacement_A) * hard_axial,
            min(0.20, maximum_displacement_A) * hard_transverse,
        ],
        axis=-1,
    )
    return {
        "vacancy": np.zeros_like(strained),
        "vacancy_plus_strain": strained,
        "strained_surface_defects": hard,
    }


def _surface_defect_truths(
    variable_sites: np.ndarray,
    *,
    center_s_A: float,
    simple_width_sites: int,
    eligible_site_mask: np.ndarray | None = None,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    """Build a simple terrace vacancy and an irregular multilayer surface pit."""

    def nearest(indices: np.ndarray, target: float, count: int) -> np.ndarray:
        if not len(indices) or count <= 0:
            return np.empty(0, dtype=int)
        order = np.argsort(np.abs(variable_sites[indices, 0] - target))
        return indices[order[: min(count, len(indices))]]

    if eligible_site_mask is None:
        eligible = np.ones(len(variable_sites), dtype=bool)
    else:
        eligible = np.asarray(eligible_site_mask)
        if eligible.dtype != np.bool_ or eligible.shape != (len(variable_sites),):
            raise TypeError(
                "eligible_site_mask must be a Boolean vector over variable sites"
            )
    if not np.any(eligible):
        raise ValueError("no structurally reportable sites are eligible for defects")
    layers = np.unique(
        np.round(variable_sites[eligible, 1], decimals=8)
    )[::-1]
    layer_indices = [
        np.flatnonzero(
            eligible & np.isclose(variable_sites[:, 1], layer, atol=1e-7)
        )
        for layer in layers[:3]
    ]
    if not layer_indices or len(layer_indices[0]) < simple_width_sites:
        raise ValueError("not enough variable top-layer sites for the defect")

    simple_sites = nearest(layer_indices[0], center_s_A, simple_width_sites)
    simple = np.zeros(len(variable_sites), dtype=float)
    simple[simple_sites] = 1.0

    top_width = max(simple_width_sites + 4, 6)
    central_widths = (
        top_width,
        max(round(0.6 * top_width), 2),
        max(round(0.3 * top_width), 1),
    )
    complex_groups = [
        nearest(indices, center_s_A, width)
        for indices, width in zip(layer_indices, central_widths)
    ]
    top_s = variable_sites[layer_indices[0], 0]
    available_half_span = max(
        min(center_s_A - float(np.min(top_s)), float(np.max(top_s)) - center_s_A),
        0.0,
    )
    satellite_offset = min(45.0, 0.55 * available_half_span)
    satellite_width = max(round(0.3 * top_width), 2)
    complex_groups.extend(
        [
            nearest(layer_indices[0], center_s_A - satellite_offset, satellite_width),
            nearest(layer_indices[0], center_s_A + satellite_offset, satellite_width),
        ]
    )
    complex_sites = np.unique(np.concatenate(complex_groups))
    complex_fractions = np.zeros(len(variable_sites), dtype=float)
    complex_fractions[complex_sites] = 1.0
    fractions = {
        "vacancy": simple,
        "vacancy_plus_strain": simple.copy(),
        "strained_surface_defects": complex_fractions,
    }
    indices = {
        case: np.flatnonzero(values >= 0.5) for case, values in fractions.items()
    }
    return fractions, indices


def build_silicon_glancing_experiment_1d(
    config: SiliconGlancingConfig1D | None = None,
) -> SiliconGlancingExperiment1D:
    """Construct the complete matched silicon geometry and lattice model."""
    import abtem
    from ase.build import bulk

    config = SiliconGlancingConfig1D() if config is None else config
    if config.n_scans < 2:
        raise ValueError("n_scans must be at least two")
    if config.defect_width_sites < 1:
        raise ValueError("defect_width_sites must be positive")
    if config.fixed_exterior_assumption is not None:
        raise ValueError(
            "fixed_exterior_assumption is deprecated because it silently asserted "
            "pristine material; use exterior_material_policy with an explicit "
            "provenance identifier when material is genuinely known"
        )
    if config.exterior_material_policy not in {
        "parameterize_uncertain",
        "known_fixed",
        "reject",
    }:
        raise ValueError(
            "exterior_material_policy must be 'parameterize_uncertain', "
            "'known_fixed', or 'reject'"
        )
    if (
        config.exterior_material_policy == "known_fixed"
        and not config.fixed_exterior_provenance_id
    ):
        raise ValueError(
            "known_fixed exterior material requires "
            "fixed_exterior_provenance_id"
        )
    abtem.config.set({"device": "cpu", "precision": "float64"})

    unit = bulk("Si", "diamond", a=config.si_lattice_A, cubic=True)
    unit.pbc = [True, True, True]
    projection_width_A = float(unit.cell.lengths()[1])
    potential_builder = abtem.Potential(
        unit,
        sampling=(config.sampling_u_A, config.sampling_s_A),
        slice_thickness=projection_width_A,
        projection="finite",
        parametrization="lobato",
        plane="xz",
        device="cpu",
    )
    du, ds = (float(value) for value in potential_builder.sampling)
    window_length = int(round(config.window_length_A / ds))
    n_u = int(
        np.ceil(
            (config.slab_depth_A + config.vacuum_above_A + config.vacuum_below_A) / du
        )
    )
    n_u += n_u % 2
    u_A = (np.arange(n_u) - n_u // 2) * du
    s_A = np.arange(window_length) * ds

    scan_coordinates = np.linspace(
        config.scan_start_A, config.scan_stop_A, config.n_scans
    )
    scan_partition = stratified_scan_partition_1d(
        config.n_scans,
        validation_stride=config.validation_stride,
        audit_fraction=config.audit_fraction,
        audit_blocks=config.audit_blocks,
        audit_guard_scans=config.audit_guard_scans,
    )
    training_indices = np.asarray(scan_partition.training_indices)
    validation_indices = np.asarray(scan_partition.validation_indices)
    audit_indices = np.asarray(scan_partition.audit_indices)
    guard_indices = np.asarray(scan_partition.guard_indices)
    training_scan_coordinates = scan_coordinates[training_indices]
    interaction_region = _automatic_interaction_region(
        config,
        s_A,
        u_A,
        scan_coordinates,
        training_indices,
    )
    coverage = np.asarray(interaction_region.scan_coverage)
    if config.update_region == "auto":
        reconstruction_mask = np.asarray(interaction_region.reconstruction_mask)
    elif config.update_region == "landing":
        reconstruction_mask = _landing_region(
            config, s_A, u_A, training_scan_coordinates
        )
    elif config.update_region == "beam_path":
        legacy_coverage = (
            1
            if config.minimum_scan_coverage is None
            else config.minimum_scan_coverage
        )
        legacy_config = dataclass_replace(
            config, minimum_scan_coverage=legacy_coverage
        )
        reconstruction_mask, _ = _beam_path_region(
            legacy_config, s_A, u_A, training_scan_coordinates
        )
    else:
        raise ValueError("update_region must be 'auto', 'landing', or 'beam_path'")
    site_selection_mask = np.asarray(reconstruction_mask, dtype=bool)
    all_sites = _projected_si_sites(
        unit,
        lattice_A=config.si_lattice_A,
        length_A=window_length * ds,
        depth_A=config.slab_depth_A,
    )
    site_s_indices = np.rint((all_sites[:, 0] - s_A[0]) / ds).astype(int)
    site_u_indices = np.rint((all_sites[:, 1] - u_A[0]) / du).astype(int)
    template, half_shape, template_certification = _certified_projected_si_template(
        config, ds=ds, du=du
    )
    site_center_indices = np.column_stack(
        [site_s_indices, site_u_indices]
    ).astype(np.int64, copy=False)
    site_patch_starts = site_center_indices - np.asarray(
        half_shape, dtype=np.int64
    )
    site_patch_shapes = np.broadcast_to(
        np.asarray(template.shape, dtype=np.int64),
        site_patch_starts.shape,
    ).copy()
    if config.exterior_material_policy == "known_fixed":
        contract_policy = "parameterize_uncertain"
        known_fixed_site_mask = np.ones(len(all_sites), dtype=bool)
    elif config.exterior_material_policy == "reject":
        contract_policy = "leave_unresolved"
        known_fixed_site_mask = None
    else:
        contract_policy = "parameterize_uncertain"
        known_fixed_site_mask = None
    support_arguments = {
        "all_site_coordinates": all_sites,
        "site_center_indices": site_center_indices,
        "site_patch_starts": site_patch_starts,
        "site_patch_shapes": site_patch_shapes,
        "target_pixel_mask": site_selection_mask,
        "forward_pixel_mask": np.asarray(interaction_region.forward_mask),
        "exterior_policy": contract_policy,
        "known_fixed_site_mask": known_fixed_site_mask,
        "fixed_material_provenance_id": (
            config.fixed_exterior_provenance_id
        ),
        "excluded_probe_power": interaction_region.excluded_probe_power,
        "atomic_template_cutoff_A": template_certification.cutoff_A,
        "maximum_displacement_A": config.maximum_displacement_A,
        "maximum_nuisance_sites": config.maximum_nuisance_sites,
        "maximum_specimen_parameters": config.maximum_specimen_parameters,
    }
    preliminary_support_contract = classify_lattice_site_support_1d(
        **support_arguments,
        strict=True,
    )
    modeled_indices = np.asarray(
        preliminary_support_contract.modeled_site_indices
    )
    variable_sites = all_sites[modeled_indices]
    if len(variable_sites) == 0:
        raise ValueError("the selected update region contains no silicon sites")
    control_s_A = _control_axis(
        variable_sites[:, 0], config.displacement_control_spacing_A
    )
    control_u_A = _control_axis(
        variable_sites[:, 1], config.displacement_control_spacing_u_A
    )
    support_contract = classify_lattice_site_support_1d(
        **support_arguments,
        displacement_control_shape=(len(control_s_A), len(control_u_A), 2),
        strict=True,
    )
    if not np.array_equal(
        modeled_indices, np.asarray(support_contract.modeled_site_indices)
    ):
        raise RuntimeError(
            "adding displacement-control counts changed the geometric site roles"
        )
    modeled_roles = np.asarray(support_contract.site_role_codes)[modeled_indices]
    modeled_target_site_mask = modeled_roles == int(LatticeSiteRole1D.TARGET)
    modeled_nuisance_site_mask = modeled_roles == int(
        LatticeSiteRole1D.NUISANCE
    )
    target_sites = variable_sites[modeled_target_site_mask]
    independent_kirkland_template = render_si_atomic_template_1d(
        sampling_s_A=ds,
        sampling_u_A=du,
        options=AtomicTemplateQuadratureOptions1D(
            projection_width_A=projection_width_A,
            cutoff_A=template_certification.cutoff_A,
        ),
        half_shape=half_shape,
    )
    lobato_kirkland_comparison = compare_si_atomic_template_1d(
        independent_kirkland_template,
        template,
        reference_provenance={
            "atomic_parameterization": "Lobato",
            "builder": "abtem.Potential finite projection",
            "normalization": "divided by finite projection width",
            "output_axis_order": "(s,u)",
            "projection_width_A": f"{projection_width_A:.17g}",
            "sampling_s_A": f"{ds:.17g}",
            "sampling_u_A": f"{du:.17g}",
        },
    )
    pristine = _finite_reference_potential(
        all_sites,
        template,
        half_shape,
        s_A=s_A,
        u_A=u_A,
        ds=ds,
        du=du,
    )
    patches, patch_starts = _patches_for_sites(
        variable_sites,
        template,
        half_shape,
        s_A=s_A,
        u_A=u_A,
        ds=ds,
        du=du,
        material_u_mask=None,
    )
    lattice_influence_mask = _lattice_parameter_update_mask(
        patches,
        patch_starts,
        potential_shape=pristine.shape,
        maximum_displacement_A=config.maximum_displacement_A,
        ds=ds,
        du=du,
    )
    contracted_influence = np.asarray(
        support_contract.target_influence_mask
        | support_contract.nuisance_influence_mask
    )
    if not np.array_equal(lattice_influence_mask, contracted_influence):
        raise RuntimeError(
            "renderer influence support disagrees with the material-support "
            "contract"
        )
    target_lattice_influence_mask = np.asarray(
        support_contract.target_influence_mask
    )
    nuisance_lattice_influence_mask = np.asarray(
        support_contract.nuisance_influence_mask
    )
    lattice_model = LatticeSiteModel1D(
        reference_potential=jnp.asarray(pristine),
        site_coordinates=jnp.asarray(variable_sites),
        site_patches=jnp.asarray(patches),
        patch_starts=jnp.asarray(patch_starts),
        control_coordinates_s=jnp.asarray(control_s_A),
        control_coordinates_u=jnp.asarray(control_u_A),
        axial_sampling=ds,
        transverse_sampling=du,
        maximum_displacement=config.maximum_displacement_A,
        metadata={
            "species": "Si",
            "atomic_potential": "Lobato finite projection",
            "atomic_template_cutoff_A": template_certification.cutoff_A,
            "atomic_template_relative_tail_l2": (
                template_certification.relative_tail_l2
            ),
            "displacement_control_spacing_s_A": config.displacement_control_spacing_A,
            "displacement_control_spacing_u_A": config.displacement_control_spacing_u_A,
            "update_region": config.update_region,
            "support_contract_id": support_contract.contract_id,
            "material_scope_complete": True,
            "structural_reporting_scope": "target_sites_only",
        },
        support_contract=support_contract,
    )

    truth_vacancies, defect_indices = _surface_defect_truths(
        variable_sites,
        center_s_A=config.defect_center_s_A,
        simple_width_sites=config.defect_width_sites,
        eligible_site_mask=modeled_target_site_mask,
    )
    truth_controls = _truth_controls(
        control_s_A,
        control_u_A,
        config.defect_center_s_A,
        config.maximum_displacement_A,
    )
    truth_rigid_displacements = {}
    residual_truth_controls = {}
    for case, controls in truth_controls.items():
        rigid, residual = decompose_lattice_site_displacement_controls_1d(
            jnp.asarray(variable_sites),
            jnp.asarray(controls),
            jnp.asarray(control_s_A),
            jnp.asarray(control_u_A),
        )
        truth_rigid_displacements[case] = rigid
        residual_truth_controls[case] = residual
    truth_controls = residual_truth_controls
    truth_vacancies = {
        case: jnp.asarray(value) for case, value in truth_vacancies.items()
    }
    truth_potentials = {
        case: render_lattice_site_potential_1d(
            lattice_model,
            truth_vacancies[case],
            controls + truth_rigid_displacements[case],
        )
        for case, controls in truth_controls.items()
    }

    larger_template, larger_half_shape = _projected_si_template(
        config,
        ds=ds,
        du=du,
        cutoff_A=template_certification.reference_cutoff_A,
    )
    larger_pristine = _finite_reference_potential(
        all_sites,
        larger_template,
        larger_half_shape,
        s_A=s_A,
        u_A=u_A,
        ds=ds,
        du=du,
    )
    larger_patches, larger_starts = _patches_for_sites(
        variable_sites,
        larger_template,
        larger_half_shape,
        s_A=s_A,
        u_A=u_A,
        ds=ds,
        du=du,
        material_u_mask=None,
    )
    cutoff_model = LatticeSiteModel1D(
        reference_potential=jnp.asarray(larger_pristine),
        site_coordinates=jnp.asarray(variable_sites),
        site_patches=jnp.asarray(larger_patches),
        patch_starts=jnp.asarray(larger_starts),
        control_coordinates_s=jnp.asarray(control_s_A),
        control_coordinates_u=jnp.asarray(control_u_A),
        axial_sampling=ds,
        transverse_sampling=du,
        maximum_displacement=config.maximum_displacement_A,
        metadata={
            "species": "Si",
            "atomic_potential": "Lobato finite projection",
            "atomic_template_cutoff_A": (
                template_certification.reference_cutoff_A
            ),
            "certification_role": "larger_cutoff_reference",
        },
    )
    cutoff_checks = {
        case: render_lattice_site_potential_1d(
            cutoff_model,
            truth_vacancies[case],
            controls + truth_rigid_displacements[case],
        )
        for case, controls in truth_controls.items()
    }
    stress_shape = (
        len(control_s_A),
        len(control_u_A),
        2,
    )
    template_stress_pairs = {}
    for name, displacement in (
        (
            "maximum_positive_diagonal_displacement",
            np.asarray(
                [config.maximum_displacement_A, config.maximum_displacement_A]
            ),
        ),
        (
            "maximum_negative_diagonal_displacement",
            np.asarray(
                [-config.maximum_displacement_A, -config.maximum_displacement_A]
            ),
        ),
    ):
        stress_controls = jnp.asarray(
            np.broadcast_to(displacement, stress_shape).copy()
        )
        compact_stress = render_lattice_site_potential_1d(
            lattice_model,
            jnp.zeros(len(variable_sites)),
            stress_controls,
        )
        reference_stress = render_lattice_site_potential_1d(
            cutoff_model,
            jnp.zeros(len(variable_sites)),
            stress_controls,
        )
        template_stress_pairs[name] = (compact_stress, reference_stress)

    tilt = -float(np.deg2rad(config.glancing_angle_deg))
    beam_centers = -scan_coordinates * np.tan(tilt)
    input_probes = jnp.stack(
        [
            make_tilted_gaussian_beam_1d(
                jnp.asarray(u_A),
                config.energy_eV,
                waist=config.beam_waist_A,
                center=float(center),
                tilt=tilt,
            )
            for center in beam_centers
        ]
    ).astype(jnp.complex128)
    wavelength_A = float(energy2wavelength(config.energy_eV))
    carrier_fraction = 2.0 * du * abs(np.sin(tilt)) / wavelength_A
    if carrier_fraction > 0.75:
        raise RuntimeError(
            f"tilt carrier uses {carrier_fraction:.3f} of Nyquist; reduce sampling_u_A"
        )
    detector_frequency = np.fft.fftshift(np.fft.fftfreq(n_u, d=du))
    detector_angles = 1e3 * np.arcsin(
        np.clip(wavelength_A * detector_frequency, -1.0, 1.0)
    )
    if abs(np.sin(tilt)) < 1e-12:
        site_scan_distance = np.broadcast_to(
            np.abs(variable_sites[:, 1, None]),
            (len(variable_sites), len(scan_coordinates)),
        )
    else:
        equivalent_site_landing = (
            variable_sites[:, 0] - variable_sites[:, 1] / np.tan(tilt)
        )
        site_scan_distance = abs(np.sin(tilt)) * np.abs(
            equivalent_site_landing[:, None] - scan_coordinates[None, :]
        )
    maximum_perpendicular_site_shift_A = config.maximum_displacement_A * (
        abs(np.sin(tilt)) + abs(np.cos(tilt))
    )
    possible_audit_radius_A = (
        interaction_region.nominal_radius_A + maximum_perpendicular_site_shift_A
    )
    audit_site_scan_coverage = np.sum(
        site_scan_distance[:, audit_indices] <= possible_audit_radius_A,
        axis=1,
    ).astype(np.int32)
    audit_site_scan_coverage_metadata = {
        "definition": (
            "count of held-out audit centrelines that could enter the nominal "
            "beam threshold for at least one allowed site displacement"
        ),
        "interpretation": (
            "conservative geometric possibility only; nonzero coverage is not "
            "evidence of information, sensitivity, or identifiability"
        ),
        "uses_measured_diffraction_values": False,
        "nominal_beam_radius_A": float(interaction_region.nominal_radius_A),
        "maximum_perpendicular_site_shift_A": float(
            maximum_perpendicular_site_shift_A
        ),
        "possible_overlap_radius_A": float(possible_audit_radius_A),
    }
    n_controls = int(np.prod(truth_controls["vacancy_plus_strain"].shape))
    n_parameters = len(variable_sites) + n_controls
    n_target_structural_parameters = int(
        np.count_nonzero(modeled_target_site_mask)
    ) + n_controls
    if n_parameters != support_contract.parameter_counts.total_specimen_parameters:
        raise RuntimeError(
            "workflow parameter count disagrees with the material-support contract"
        )
    summary = {
        "potential shape": pristine.shape,
        "sampling (ds, du) A": (ds, du),
        "scan count": config.n_scans,
        "training scans": len(training_indices),
        "validation scans": len(validation_indices),
        "audit scans": len(audit_indices),
        "audit blocks": scan_partition.metadata["audit_blocks_used"],
        "guard scans": len(guard_indices),
        "minimum possible audit geometric overlaps per variable site": int(
            np.min(audit_site_scan_coverage)
        ),
        "variable sites with possible audit geometric overlap": int(
            np.count_nonzero(audit_site_scan_coverage)
        ),
        "pixel unknowns": int(reconstruction_mask.sum()),
        "update region": config.update_region,
        "forward interaction pixels": int(
            np.count_nonzero(np.asarray(interaction_region.forward_mask))
        ),
        "illuminated pixels outside active-site support": int(
            np.count_nonzero(
                np.asarray(interaction_region.forward_mask) & ~site_selection_mask
            )
        ),
        "interaction intensity threshold": interaction_region.intensity_threshold,
        "excluded probe power": interaction_region.excluded_probe_power,
        "nominal mutable interaction radius (A)": (
            interaction_region.nominal_radius_A
        ),
        "uncertainty-expanded forward interaction radius (A)": (
            interaction_region.radius_A
        ),
        "geometry uncertainty margin (A)": (
            interaction_region.uncertainty_margin_A
        ),
        "minimum training-scan coverage": (
            interaction_region.minimum_scan_coverage
        ),
        "exterior material policy": config.exterior_material_policy,
        "fixed exterior provenance": config.fixed_exterior_provenance_id,
        "material support contract": support_contract.contract_id,
        "material scope complete": support_contract.strict_requirements_satisfied,
        "atomic template cutoff (A)": template_certification.cutoff_A,
        "atomic template relative tail L2": (
            template_certification.relative_tail_l2
        ),
        "atomic template parameterization diagnostic": (
            _atomic_parameterization_diagnostic_metadata(
                independent_kirkland_template,
                lobato_kirkland_comparison,
            )
        ),
        "target Si sites": int(np.count_nonzero(modeled_target_site_mask)),
        "nuisance Si sites": int(np.count_nonzero(modeled_nuisance_site_mask)),
        "modeled Si sites": len(variable_sites),
        "variable Si sites": len(variable_sites),
        "fixed-known Si sites": int(
            support_contract.parameter_count_metadata["fixed_known_sites"]
        ),
        "below-interaction-budget Si sites": int(
            support_contract.parameter_count_metadata[
                "below_interaction_budget_sites"
            ]
        ),
        "displacement controls": n_controls,
        "lattice parameters": n_parameters,
        "target structural parameters": n_target_structural_parameters,
        "pixel / lattice reduction": float(
            reconstruction_mask.sum() / max(n_parameters, 1)
        ),
        "pixel / target-structural reduction": float(
            reconstruction_mask.sum()
            / max(n_target_structural_parameters, 1)
        ),
        "simple vacancy sites": len(defect_indices["vacancy"]),
        "complex surface-defect sites": len(
            defect_indices["strained_surface_defects"]
        ),
    }
    return SiliconGlancingExperiment1D(
        config=config,
        pristine_potential=jnp.asarray(pristine),
        lattice_model=lattice_model,
        template_certification=template_certification,
        independent_kirkland_template=independent_kirkland_template,
        lobato_kirkland_template_comparison=lobato_kirkland_comparison,
        support_contract=support_contract,
        interaction_region=interaction_region,
        truth_potentials=truth_potentials,
        truth_vacancy_fractions=truth_vacancies,
        truth_displacement_controls=truth_controls,
        truth_rigid_displacements=truth_rigid_displacements,
        defect_site_indices={
            case: jnp.asarray(indices) for case, indices in defect_indices.items()
        },
        all_site_coordinates=jnp.asarray(all_sites),
        variable_sites=jnp.asarray(variable_sites),
        target_sites=jnp.asarray(target_sites),
        modeled_target_site_mask=jnp.asarray(modeled_target_site_mask),
        modeled_nuisance_site_mask=jnp.asarray(modeled_nuisance_site_mask),
        site_selection_mask=jnp.asarray(site_selection_mask),
        reconstruction_mask=jnp.asarray(reconstruction_mask),
        lattice_influence_mask=jnp.asarray(lattice_influence_mask),
        target_lattice_influence_mask=jnp.asarray(
            target_lattice_influence_mask
        ),
        nuisance_lattice_influence_mask=jnp.asarray(
            nuisance_lattice_influence_mask
        ),
        beam_path_scan_coverage=jnp.asarray(coverage),
        input_probes=input_probes,
        propagation_kernel=angular_spectrum_propagation_kernel_1d(
            n_u, du, ds, config.energy_eV
        ),
        window_starts=jnp.zeros(config.n_scans, dtype=jnp.int32),
        window_length=window_length,
        scan_coordinates=jnp.asarray(scan_coordinates),
        axial_coordinates=jnp.asarray(s_A),
        transverse_coordinates=jnp.asarray(u_A),
        detector_angles=jnp.asarray(detector_angles),
        training_indices=jnp.asarray(training_indices),
        validation_indices=jnp.asarray(validation_indices),
        audit_indices=jnp.asarray(audit_indices),
        guard_indices=jnp.asarray(guard_indices),
        audit_site_scan_coverage=jnp.asarray(audit_site_scan_coverage),
        audit_site_scan_coverage_metadata=audit_site_scan_coverage_metadata,
        cutoff_check_potentials=cutoff_checks,
        template_stress_potential_pairs=template_stress_pairs,
        axial_sampling=ds,
        transverse_sampling=du,
        summary=summary,
    )


def build_silicon_alignment_prior_1d(
    experiment: SiliconGlancingExperiment1D,
) -> SiliconAlignmentPrior1D:
    """Copy only material/grid assumptions into a complete-slab search prior.

    This adapter is convenient for the synthetic notebook, but it deliberately
    ignores every truth potential, defect label, vacancy fraction, strain
    control, defect center, and generating displacement.  Candidate phase,
    termination, rotation, and scale subsequently rebuild all slab atoms.
    """
    if not isinstance(experiment, SiliconGlancingExperiment1D):
        raise TypeError("experiment must be a SiliconGlancingExperiment1D")
    from ase.build import bulk

    config = experiment.config
    ds = float(experiment.axial_sampling)
    du = float(experiment.transverse_sampling)
    template, half_shape = _projected_si_template(
        config,
        ds=ds,
        du=du,
        cutoff_A=experiment.template_certification.cutoff_A,
    )
    unit = bulk("Si", "diamond", a=config.si_lattice_A, cubic=True)
    projected_basis_fractional = (
        np.asarray(unit.positions)[:, [2, 0]] / config.si_lattice_A
    )
    normal_phases = np.unique(
        np.round(
            np.mod(
                np.asarray(unit.positions)[:, 0] / config.si_lattice_A,
                0.5,
            ),
            decimals=12,
        )
    )
    termination_ids = tuple(
        f"si_termination_{index}" for index in range(len(normal_phases))
    )
    return make_silicon_alignment_prior_1d(
        axial_coordinates=experiment.axial_coordinates,
        transverse_coordinates=experiment.transverse_coordinates,
        reconstruction_mask=experiment.reconstruction_mask,
        projected_si_template=template,
        template_half_shape=half_shape,
        projected_basis_fractional_su=projected_basis_fractional,
        nominal_lattice_A=config.si_lattice_A,
        slab_depth_A=config.slab_depth_A,
        maximum_displacement_A=config.maximum_displacement_A,
        displacement_control_spacing_s_A=(
            config.displacement_control_spacing_A
        ),
        displacement_control_spacing_u_A=(
            config.displacement_control_spacing_u_A
        ),
        termination_ids=termination_ids,
        termination_offsets_fractional_u=normal_phases,
        metadata={
            "species": "Si",
            "atomic_potential": "Lobato finite projection",
            "atomic_template_cutoff_A": (
                experiment.template_certification.cutoff_A
            ),
            "source_adapter": "SiliconGlancingExperiment1D_truth_fields_ignored",
        },
    )


def build_silicon_alignment_problem_1d(
    experiment: SiliconGlancingExperiment1D,
) -> SiliconAlignmentForwardProblem1D:
    """Return truth-free material and forward geometry for global alignment."""
    prior = build_silicon_alignment_prior_1d(experiment)
    return make_silicon_alignment_forward_problem_1d(
        prior,
        input_probes=experiment.input_probes,
        propagation_kernel=experiment.propagation_kernel,
        window_starts=experiment.window_starts,
        window_length=experiment.window_length,
        scan_coordinates=experiment.scan_coordinates,
        detector_angles=experiment.detector_angles,
        slice_thickness_A=experiment.axial_sampling,
        energy_eV=experiment.config.energy_eV,
        training_indices=experiment.training_indices,
        validation_indices=experiment.validation_indices,
        audit_indices=experiment.audit_indices,
        guard_indices=experiment.guard_indices,
        metadata={
            "source_adapter": "SiliconGlancingExperiment1D_truth_fields_ignored",
        },
    )


def _simulate_in_batches(
    experiment: SiliconGlancingExperiment1D,
    potential: Array,
    *,
    batch_size: int,
) -> Array:
    chunks = []
    for begin in range(0, len(experiment.window_starts), batch_size):
        end = begin + batch_size
        chunks.append(
            simulate_glancing_scan_1d(
                potential,
                experiment.input_probes[begin:end],
                experiment.window_starts[begin:end],
                experiment.window_length,
                experiment.propagation_kernel,
                experiment.axial_sampling,
                experiment.config.energy_eV,
                rematerialize=False,
            )
        )
    return jnp.concatenate(chunks, axis=0)


_KIRKLAND_ALTERNATIVE_SHARED_COMPONENTS = (
    "Only the centred atomic template and IAM parameterization are changed. "
    "Finite-grid accumulation, scipy bilinear base-site shifts, the "
    "differentiable displacement renderer, and angular-spectrum propagation "
    "are shared with the production Lobato path; this is not an independent "
    "end-to-end forward-model validation."
)


def _render_kirkland_alternative_case_potential_1d(
    experiment: SiliconGlancingExperiment1D,
    case: str,
) -> Array:
    """Lazily render one selected truth case with the Kirkland template."""
    if case not in experiment.truth_vacancy_fractions:
        raise ValueError(f"unknown synthetic case {case!r}")
    template_result = experiment.independent_kirkland_template
    template = np.asarray(template_result.values)
    half_shape = template_result.half_shape
    s_A = np.asarray(experiment.axial_coordinates)
    u_A = np.asarray(experiment.transverse_coordinates)
    all_sites = np.asarray(experiment.all_site_coordinates)
    variable_sites = np.asarray(experiment.variable_sites)
    ds = float(experiment.axial_sampling)
    du = float(experiment.transverse_sampling)
    kirkland_pristine = _finite_reference_potential(
        all_sites,
        template,
        half_shape,
        s_A=s_A,
        u_A=u_A,
        ds=ds,
        du=du,
    )
    kirkland_patches, kirkland_starts = _patches_for_sites(
        variable_sites,
        template,
        half_shape,
        s_A=s_A,
        u_A=u_A,
        ds=ds,
        du=du,
        material_u_mask=None,
    )
    production_model = experiment.lattice_model
    kirkland_model = LatticeSiteModel1D(
        reference_potential=jnp.asarray(kirkland_pristine),
        site_coordinates=production_model.site_coordinates,
        site_patches=jnp.asarray(kirkland_patches),
        patch_starts=jnp.asarray(kirkland_starts),
        control_coordinates_s=production_model.control_coordinates_s,
        control_coordinates_u=production_model.control_coordinates_u,
        axial_sampling=ds,
        transverse_sampling=du,
        maximum_displacement=production_model.maximum_displacement,
        metadata={
            "species": "Si",
            "atomic_potential": "direct-quadrature Kirkland finite projection",
            "candidate_template_sha256": template_result.template_sha256,
            "trust_claim": False,
            "shared_components_limitation": (
                _KIRKLAND_ALTERNATIVE_SHARED_COMPONENTS
            ),
        },
    )
    controls = (
        experiment.truth_displacement_controls[case]
        + experiment.truth_rigid_displacements[case]
    )
    return render_lattice_site_potential_1d(
        kirkland_model,
        experiment.truth_vacancy_fractions[case],
        controls,
    )


def _amplitude_nrmse(predicted: Array, reference: Array) -> float:
    predicted_amplitude = np.sqrt(np.asarray(predicted) + 1e-12)
    reference_amplitude = np.sqrt(np.asarray(reference) + 1e-12)
    return float(
        np.linalg.norm(predicted_amplitude - reference_amplitude)
        / np.linalg.norm(reference_amplitude)
    )


def _amplitude_nrmse_per_scan(predicted: Array, reference: Array) -> np.ndarray:
    """Return one detector-amplitude NRMSE for each scan pattern."""
    predicted_amplitude = np.sqrt(np.asarray(predicted) + 1e-12)
    reference_amplitude = np.sqrt(np.asarray(reference) + 1e-12)
    if predicted_amplitude.shape != reference_amplitude.shape or (
        predicted_amplitude.ndim != 2
    ):
        raise ValueError("diffraction arrays must have matching (scan, detector) shape")
    numerator = np.linalg.norm(
        predicted_amplitude - reference_amplitude,
        axis=1,
    )
    denominator = np.linalg.norm(reference_amplitude, axis=1)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError("every reference diffraction pattern must have finite norm")
    return numerator / denominator


def simulate_experiment_1d(
    experiment: SiliconGlancingExperiment1D,
    case: str = "vacancy",
    *,
    batch_size: int = 10,
) -> GlancingDataset1D:
    """Simulate one truth case and evaluate descriptive mismatch diagnostics."""
    if case not in experiment.truth_potentials:
        raise ValueError(
            f"case must be one of {tuple(experiment.truth_potentials)}, got {case!r}"
        )
    potential = experiment.truth_potentials[case]
    measured = _simulate_in_batches(experiment, potential, batch_size=batch_size)
    kirkland_potential = _render_kirkland_alternative_case_potential_1d(
        experiment,
        case,
    )
    kirkland_intensities = _simulate_in_batches(
        experiment,
        kirkland_potential,
        batch_size=batch_size,
    )
    del kirkland_potential
    kirkland_alternative_nrmse = _amplitude_nrmse(
        kirkland_intensities,
        measured,
    )
    kirkland_alternative_scan_errors = _amplitude_nrmse_per_scan(
        kirkland_intensities,
        measured,
    )
    kirkland_alternative_max_scan_nrmse = float(
        np.max(kirkland_alternative_scan_errors)
    )
    del kirkland_intensities
    masked = jnp.where(experiment.reconstruction_mask, potential, 0.0)
    masked_intensities = _simulate_in_batches(experiment, masked, batch_size=batch_size)
    cutoff_intensities = _simulate_in_batches(
        experiment,
        experiment.cutoff_check_potentials[case],
        batch_size=batch_size,
    )
    zero_exterior_nrmse = _amplitude_nrmse(masked_intensities, measured)
    cutoff_nrmse = _amplitude_nrmse(cutoff_intensities, measured)
    cutoff_scan_errors = _amplitude_nrmse_per_scan(
        cutoff_intensities, measured
    )
    cutoff_max_scan_nrmse = float(np.max(cutoff_scan_errors))
    stress_scan_errors = {}
    for name, (compact_potential, reference_potential) in (
        experiment.template_stress_potential_pairs.items()
    ):
        compact_intensities = _simulate_in_batches(
            experiment, compact_potential, batch_size=batch_size
        )
        reference_intensities = _simulate_in_batches(
            experiment, reference_potential, batch_size=batch_size
        )
        stress_scan_errors[name] = float(
            np.max(
                _amplitude_nrmse_per_scan(
                    compact_intensities, reference_intensities
                )
            )
        )
    stress_worst_nrmse = max(stress_scan_errors.values(), default=0.0)
    certified_worst_nrmse = max(cutoff_max_scan_nrmse, stress_worst_nrmse)
    if certified_worst_nrmse > experiment.config.atomic_template_amplitude_tolerance:
        raise RuntimeError(
            "the certified atomic template exceeds the whole-slab forward "
            "amplitude error "
            f"budget: {certified_worst_nrmse:.3g} > "
            f"{experiment.config.atomic_template_amplitude_tolerance:.3g}"
        )
    scan = GlancingScan1D(
        intensities=measured,
        window_starts=experiment.window_starts,
        scan_coordinates=experiment.scan_coordinates,
        detector_angles=experiment.detector_angles,
        metadata={
            "energy_eV": experiment.config.energy_eV,
            "propagation_model": "Angular spectrum",
            "dataset_case": case,
            "zero_exterior_amplitude_nrmse": zero_exterior_nrmse,
            "template_cutoff_amplitude_nrmse": cutoff_nrmse,
            "template_cutoff_max_scan_amplitude_nrmse": (
                cutoff_max_scan_nrmse
            ),
            "template_stress_max_scan_amplitude_nrmse": stress_scan_errors,
            "template_stress_worst_scan_amplitude_nrmse": (
                stress_worst_nrmse
            ),
            "template_certified_worst_amplitude_nrmse": (
                certified_worst_nrmse
            ),
            "atomic_template_parameterization_diagnostic": (
                _atomic_parameterization_diagnostic_metadata(
                    experiment.independent_kirkland_template,
                    experiment.lobato_kirkland_template_comparison,
                )
            ),
            "kirkland_alternative_amplitude_nrmse": (
                kirkland_alternative_nrmse
            ),
            "kirkland_alternative_max_scan_amplitude_nrmse": (
                kirkland_alternative_max_scan_nrmse
            ),
            "kirkland_alternative_trust_claim": False,
            "kirkland_alternative_has_acceptance_threshold": False,
            "kirkland_alternative_used_for_cutoff_certification": False,
            "kirkland_alternative_shared_components_limitation": (
                _KIRKLAND_ALTERNATIVE_SHARED_COMPONENTS
            ),
            "template_amplitude_check_scope": (
                "whole_finite_slab_case_specific_and_maximum_displacement_all_scans"
            ),
            "template_reference_cutoff_A": (
                experiment.template_certification.reference_cutoff_A
            ),
            "support_contract_id": experiment.support_contract.contract_id,
            "material_scope_complete": (
                experiment.support_contract.strict_requirements_satisfied
            ),
            "structural_reporting_scope": "target_sites_only",
            "target_site_count": int(
                np.count_nonzero(experiment.modeled_target_site_mask)
            ),
            "nuisance_site_count": int(
                np.count_nonzero(experiment.modeled_nuisance_site_mask)
            ),
            "training_indices": np.asarray(experiment.training_indices).tolist(),
            "validation_indices": np.asarray(experiment.validation_indices).tolist(),
            "audit_indices": np.asarray(experiment.audit_indices).tolist(),
            "guard_indices": np.asarray(experiment.guard_indices).tolist(),
            "audit_construction": "geometry_only_stratified_contiguous_blocks",
        },
    )
    return GlancingDataset1D(
        case=case,
        potential=potential,
        scan=scan,
        truth_vacancy_fractions=experiment.truth_vacancy_fractions[case],
        truth_displacement_controls=experiment.truth_displacement_controls[case],
        truth_rigid_displacement=experiment.truth_rigid_displacements[case],
        zero_exterior_amplitude_nrmse=zero_exterior_nrmse,
        template_cutoff_amplitude_nrmse=cutoff_nrmse,
        template_cutoff_max_scan_amplitude_nrmse=cutoff_max_scan_nrmse,
        template_stress_worst_scan_amplitude_nrmse=stress_worst_nrmse,
        template_certified_worst_amplitude_nrmse=certified_worst_nrmse,
        kirkland_alternative_amplitude_nrmse=kirkland_alternative_nrmse,
        kirkland_alternative_max_scan_amplitude_nrmse=(
            kirkland_alternative_max_scan_nrmse
        ),
    )


def reconstruct_experiment_1d(
    experiment: SiliconGlancingExperiment1D,
    dataset: GlancingDataset1D | GlancingScan1D,
    *,
    methods: Sequence[str] = ("lattice_sites",),
    options: ReconstructionOptions1D | None = None,
) -> OrderedDict[str, PotentialReconstruction1D | LatticeSiteReconstruction1D]:
    """Reconstruct synthetic datasets or truth-free measured scan containers."""
    options = ReconstructionOptions1D() if options is None else options
    allowed = {"blind_pixels", "warm_pixels", "lattice_sites"}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"unknown reconstruction methods: {sorted(unknown)}")

    scan = dataset.scan if isinstance(dataset, GlancingDataset1D) else dataset
    if not isinstance(scan, GlancingScan1D):
        raise TypeError("dataset must be a GlancingDataset1D or GlancingScan1D")

    def require_same_array(
        name: str, observed: Array, expected: Array, *, exact: bool
    ) -> None:
        observed_array = np.asarray(observed)
        expected_array = np.asarray(expected)
        compatible = observed_array.shape == expected_array.shape
        if compatible:
            inexact_dtypes = [
                array.dtype
                for array in (observed_array, expected_array)
                if np.issubdtype(array.dtype, np.inexact)
            ]
            tolerance = (
                8.0 * max(np.finfo(dtype).eps for dtype in inexact_dtypes)
                if inexact_dtypes
                else 0.0
            )
            compatible = bool(
                np.array_equal(observed_array, expected_array)
                if exact
                else np.allclose(
                    observed_array,
                    expected_array,
                    rtol=tolerance,
                    atol=tolerance,
                    equal_nan=False,
                )
            )
        if not compatible:
            raise ValueError(
                f"scan {name} are incompatible with the supplied experiment"
            )

    require_same_array(
        "window_starts", scan.window_starts, experiment.window_starts, exact=True
    )
    require_same_array(
        "scan_coordinates",
        scan.scan_coordinates,
        experiment.scan_coordinates,
        exact=False,
    )
    require_same_array(
        "detector_angles",
        scan.detector_angles,
        experiment.detector_angles,
        exact=False,
    )
    measured_shape = np.asarray(scan.intensities).shape
    expected_measured_shape = (
        len(experiment.window_starts),
        len(experiment.detector_angles),
    )
    if measured_shape != expected_measured_shape:
        raise ValueError(
            "scan intensities shape is incompatible with the supplied experiment: "
            f"expected {expected_measured_shape}, received {measured_shape}"
        )

    split_names = (
        "training_indices",
        "validation_indices",
        "audit_indices",
        "guard_indices",
    )
    present_splits = [name for name in split_names if name in scan.metadata]
    if present_splits and len(present_splits) != len(split_names):
        missing = sorted(set(split_names) - set(present_splits))
        raise ValueError(
            "scan split metadata are incomplete; missing " + ", ".join(missing)
        )
    for name in present_splits:
        require_same_array(
            f"metadata[{name!r}]",
            scan.metadata[name],
            getattr(experiment, name),
            exact=True,
        )
    support_metadata_names = {
        "support_contract_id",
        "material_scope_complete",
        "structural_reporting_scope",
        "target_site_count",
        "nuisance_site_count",
    }
    present_support_metadata = support_metadata_names.intersection(scan.metadata)
    if present_support_metadata and present_support_metadata != support_metadata_names:
        missing = sorted(support_metadata_names - present_support_metadata)
        raise ValueError(
            "scan support-contract metadata are incomplete; missing "
            + ", ".join(missing)
        )
    if present_support_metadata:
        expected_support_metadata = {
            "support_contract_id": experiment.support_contract.contract_id,
            "material_scope_complete": True,
            "structural_reporting_scope": "target_sites_only",
            "target_site_count": int(
                np.count_nonzero(experiment.modeled_target_site_mask)
            ),
            "nuisance_site_count": int(
                np.count_nonzero(experiment.modeled_nuisance_site_mask)
            ),
        }
        for name, expected in expected_support_metadata.items():
            if scan.metadata[name] != expected:
                raise ValueError(
                    f"scan metadata[{name!r}] do not match the material-support "
                    "contract"
                )

    pristine = np.asarray(experiment.pristine_potential)
    positive = pristine[pristine > 0.0]
    potential_scale = float(np.mean(positive))
    potential_max = 2.0 * float(np.max(positive))
    phase_bound = (
        float(interaction_constant(experiment.config.energy_eV))
        * experiment.axial_sampling
        * potential_max
    )
    if phase_bound >= np.pi:
        raise RuntimeError(
            f"per-slice phase bound is {phase_bound:.3f} rad; refine sampling_s_A"
        )

    common = dict(
        reconstruction_mask=experiment.reconstruction_mask,
        input_probe=experiment.input_probes,
        window_starts=experiment.window_starts,
        window_length=experiment.window_length,
        propagation_kernel=experiment.propagation_kernel,
        slice_thickness=experiment.axial_sampling,
        energy=experiment.config.energy_eV,
        measured_intensities=dataset.intensities,
        detector_valid_mask=scan.detector_valid_mask,
        axial_coordinates=experiment.axial_coordinates,
        transverse_coordinates=experiment.transverse_coordinates,
        scan_coordinates=experiment.scan_coordinates,
        detector_angles=experiment.detector_angles,
        validation_indices=np.asarray(experiment.validation_indices),
        audit_indices=np.asarray(experiment.audit_indices),
        excluded_indices=np.asarray(experiment.guard_indices),
        potential_scale=potential_scale,
        potential_max=potential_max,
        updates=options.pixel_updates,
        minibatch_size=options.minibatch_size,
        validation_interval=options.validation_interval_pixels,
        evaluation_batch_size=options.evaluation_batch_size,
        rematerialize=options.rematerialize,
        seed=options.seed,
    )
    results = OrderedDict()
    if "blind_pixels" in methods:
        rng = np.random.default_rng(options.seed)
        weak = (
            0.05 * potential_scale * (1.0 + 0.01 * rng.standard_normal(pristine.shape))
        )
        initial = np.where(
            experiment.reconstruction_mask,
            np.clip(weak, 0.0, potential_max),
            0.0,
        )
        blind_result = reconstruct_potential_1d(
            initial_potential=initial,
            progress=options.progress,
            progress_description="blind pixel reconstruction",
            **common,
        )
        results["blind pixels"] = dataclass_replace(
            blind_result,
            metadata={
                **dict(blind_result.metadata),
                "exterior_material_policy": "zero_unmodeled_baseline",
                "material_scope_complete": False,
                "structurally_trusted": False,
            },
        )
    if "warm_pixels" in methods:
        warm_result = reconstruct_potential_1d(
            initial_potential=experiment.pristine_potential,
            fixed_potential=experiment.pristine_potential,
            progress=options.progress,
            progress_description="warm pixel reconstruction",
            **common,
        )
        results["pristine-initialized pixels"] = dataclass_replace(
            warm_result,
            metadata={
                **dict(warm_result.metadata),
                "exterior_material_policy": "assumed_pristine_fixed_baseline",
                "material_scope_complete": bool(
                    not np.any(experiment.modeled_nuisance_site_mask)
                ),
                "structurally_trusted": False,
            },
        )
    if "lattice_sites" in methods:
        offset = np.asarray(options.initial_site_offset_A, dtype=float)
        if offset.shape != (2,) or not np.all(np.isfinite(offset)):
            raise ValueError("initial_site_offset_A must contain two finite values")
        if not np.isfinite(options.initial_control_noise_A) or (
            options.initial_control_noise_A < 0.0
        ):
            raise ValueError("initial_control_noise_A must be finite and non-negative")
        control_shape = (
            len(experiment.lattice_model.control_coordinates_s),
            len(experiment.lattice_model.control_coordinates_u),
            2,
        )
        rng = np.random.default_rng(options.seed)
        initial_controls = options.initial_control_noise_A * rng.standard_normal(
            control_shape
        )
        initial_rigid = offset if options.separate_rigid_registration else np.zeros(2)
        if not options.separate_rigid_registration:
            initial_controls += np.broadcast_to(offset, control_shape)
        maximum_displacement = experiment.lattice_model.maximum_displacement
        control_bound = (
            0.5 * options.maximum_residual_displacement_A
            if options.separate_rigid_registration
            else maximum_displacement
        )
        initial_controls = np.clip(
            initial_controls, -control_bound, control_bound
        )
        results["lattice sites"] = reconstruct_lattice_site_potential_1d(
            experiment.lattice_model,
            experiment.input_probes,
            experiment.window_starts,
            experiment.window_length,
            experiment.propagation_kernel,
            experiment.axial_sampling,
            experiment.config.energy_eV,
            dataset.intensities,
            detector_valid_mask=scan.detector_valid_mask,
            initial_displacement_controls=initial_controls,
            initial_rigid_displacement=initial_rigid,
            separate_rigid_registration=options.separate_rigid_registration,
            maximum_rigid_displacement=(
                options.maximum_rigid_displacement_A
                if options.separate_rigid_registration
                else None
            ),
            maximum_residual_displacement=(
                options.maximum_residual_displacement_A
                if options.separate_rigid_registration
                else None
            ),
            scan_coordinates=experiment.scan_coordinates,
            detector_angles=experiment.detector_angles,
            validation_indices=np.asarray(experiment.validation_indices),
            audit_indices=np.asarray(experiment.audit_indices),
            excluded_indices=np.asarray(experiment.guard_indices),
            potential_max=potential_max,
            updates=options.lattice_updates,
            minibatch_size=options.minibatch_size,
            validation_interval=options.validation_interval_lattice,
            training_diagnostic_scan_count=(
                options.training_diagnostic_scan_count
            ),
            evaluation_batch_size=options.evaluation_batch_size,
            rematerialize=options.rematerialize,
            require_complete_material_scope=True,
            seed=options.seed,
            progress=options.progress,
            progress_description="lattice-site reconstruction",
            checkpoint_interval=options.lattice_checkpoint_interval,
            convergence=options.lattice_convergence,
            optimization=options.lattice_optimization,
        )
    return results


def reconstruct_lattice_multistart_experiment_1d(
    experiment: SiliconGlancingExperiment1D,
    dataset: GlancingDataset1D | GlancingScan1D,
    *,
    options: ReconstructionOptions1D | None = None,
    multistart_options: MultistartOptions1D | None = None,
    run_options: PreparedMultistartRunOptions1D | None = None,
    measurement: PtychographyMeasurement1D | None = None,
    objective: PtychographyObjective1D | None = None,
) -> PreparedMultistartResult1D:
    """Run validation-selected lattice starts through one prepared problem.

    The default starts vary only the translation of the active lattice sites
    relative to the fixed reference. They do not search global specimen
    height, orientation, scale, probe registration, or detector calibration.
    Screening runs collect compact checkpoints. The selected real medoid
    trajectory is reused directly for visualization, and non-selected
    histories are discarded after validation-only selection.
    """
    options = ReconstructionOptions1D() if options is None else options
    if not isinstance(options, ReconstructionOptions1D):
        raise TypeError("options must be a ReconstructionOptions1D instance or None")
    if run_options is not None and multistart_options is not None:
        raise ValueError(
            "multistart_options cannot be supplied together with run_options"
        )
    if measurement is None and objective is not None:
        raise ValueError("objective requires an explicit measurement")
    if measurement is not None and objective is None:
        raise ValueError("objective is required with an explicit measurement")
    # Reuse the same fail-closed scan/geometry/split checks as the comparison
    # workflow without launching any reconstruction method.
    reconstruct_experiment_1d(experiment, dataset, methods=(), options=options)
    scan = dataset.scan if isinstance(dataset, GlancingDataset1D) else dataset

    pristine = np.asarray(experiment.pristine_potential)
    positive = pristine[pristine > 0.0]
    potential_max = 2.0 * float(np.max(positive))
    prepared = prepare_lattice_site_reconstruction_1d(
        experiment.lattice_model,
        experiment.input_probes,
        experiment.window_starts,
        experiment.window_length,
        experiment.propagation_kernel,
        experiment.axial_sampling,
        experiment.config.energy_eV,
        None if measurement is not None else scan.intensities,
        measurement=measurement,
        objective=objective,
        detector_valid_mask=(
            None if measurement is not None else scan.detector_valid_mask
        ),
        separate_rigid_registration=options.separate_rigid_registration,
        maximum_rigid_displacement=(
            options.maximum_rigid_displacement_A
            if options.separate_rigid_registration
            else None
        ),
        maximum_residual_displacement=(
            options.maximum_residual_displacement_A
            if options.separate_rigid_registration
            else None
        ),
        scan_coordinates=experiment.scan_coordinates,
        detector_angles=experiment.detector_angles,
        validation_indices=np.asarray(experiment.validation_indices),
        audit_indices=np.asarray(experiment.audit_indices),
        excluded_indices=np.asarray(experiment.guard_indices),
        potential_max=potential_max,
        minibatch_size=options.minibatch_size,
        evaluation_batch_size=options.evaluation_batch_size,
        rematerialize=options.rematerialize,
        require_complete_material_scope=True,
    )

    if run_options is None:
        offset = np.asarray(options.initial_site_offset_A, dtype=float)
        if offset.shape != (2,) or np.any(~np.isfinite(offset)):
            raise ValueError("initial_site_offset_A must contain two finite values")
        noise = float(options.initial_control_noise_A)
        if not np.isfinite(noise) or noise < 0.0:
            raise ValueError(
                "initial_control_noise_A must be finite and non-negative"
            )
        control_shape = (
            len(experiment.lattice_model.control_coordinates_s),
            len(experiment.lattice_model.control_coordinates_u),
            2,
        )
        rng = np.random.default_rng(options.seed)
        initial_controls = noise * rng.standard_normal(control_shape)
        initial_rigid = (
            offset
            if options.separate_rigid_registration
            else np.zeros(2, dtype=float)
        )
        if not options.separate_rigid_registration:
            initial_controls += np.broadcast_to(offset, control_shape)
        ensemble_options = (
            MultistartOptions1D(base_seed=options.seed)
            if multistart_options is None
            else multistart_options
        )
        checkpoint_interval = (
            1
            if options.lattice_checkpoint_interval is None
            else options.lattice_checkpoint_interval
        )
        run_options = PreparedMultistartRunOptions1D(
            ensemble_options=ensemble_options,
            initial_displacement_controls=initial_controls,
            initial_rigid_displacement=initial_rigid,
            updates=options.lattice_updates,
            validation_interval=options.validation_interval_lattice,
            training_diagnostic_scan_count=(
                options.training_diagnostic_scan_count
            ),
            convergence=options.lattice_convergence,
            optimization=options.lattice_optimization,
            representative_checkpoint_interval=checkpoint_interval,
            progress=options.progress,
        )
    elif not isinstance(run_options, PreparedMultistartRunOptions1D):
        raise TypeError(
            "run_options must be a PreparedMultistartRunOptions1D instance or None"
        )
    return run_prepared_lattice_site_multistart_1d(
        prepared,
        options=run_options,
    )


def _target_only_lattice_model_1d(
    experiment: SiliconGlancingExperiment1D,
    model: LatticeSiteModel1D,
    *,
    result: LatticeSiteReconstruction1D | None = None,
) -> tuple[LatticeSiteModel1D, np.ndarray]:
    """Return a contract-bound model containing reportable TARGET sites only."""
    n_site = len(model.site_coordinates)
    if model.support_contract is None:
        raise ValueError(
            "TARGET-only structural visualization requires a material-support "
            "contract; unscoped lattice models may only be shown as labelled "
            "forward-model diagnostics"
        )
    contract = validate_lattice_site_support_contract_1d(
        model.support_contract,
        strict=True,
    )
    if contract.contract_id != experiment.support_contract.contract_id:
        raise ValueError(
            "lattice-model support contract does not match the experiment "
            "TARGET truth contract"
        )
    modeled = np.asarray(contract.modeled_site_indices)
    expected_sites = np.asarray(contract.all_site_coordinates)[modeled]
    expected_starts = np.asarray(contract.site_patch_starts)[modeled]
    expected_roles = np.asarray(contract.site_role_codes)[modeled].astype(
        np.int8,
        copy=False,
    )
    if n_site != modeled.size or not np.array_equal(
        np.asarray(model.site_coordinates), expected_sites
    ):
        raise ValueError(
            "lattice-model site coordinates do not match its support contract"
        )
    if not np.array_equal(np.asarray(model.patch_starts), expected_starts):
        raise ValueError("lattice-model patch starts do not match its support contract")
    expected_patch_shapes = np.asarray(contract.site_patch_shapes)[modeled]
    actual_patch_shapes = np.broadcast_to(
        np.asarray(np.asarray(model.site_patches).shape[1:], dtype=np.int64),
        expected_patch_shapes.shape,
    )
    if not np.array_equal(actual_patch_shapes, expected_patch_shapes):
        raise ValueError("lattice-model patches do not match its support contract")
    target = expected_roles == int(LatticeSiteRole1D.TARGET)
    if not np.array_equal(
        target,
        np.asarray(experiment.modeled_target_site_mask, dtype=bool),
    ):
        raise ValueError(
            "lattice-model TARGET roles do not match the experiment reporting mask"
        )
    fully_parameterized = bool(
        contract.strict_requirements_satisfied
        and not np.any(
            np.asarray(contract.site_role_codes)
            == int(LatticeSiteRole1D.FIXED_KNOWN)
        )
    )
    if result is not None:
        if not np.array_equal(
            np.asarray(result.site_coordinates), np.asarray(model.site_coordinates)
        ):
            raise ValueError("result and lattice model must use identical sites")
        if not np.array_equal(np.asarray(result.site_role_codes), expected_roles):
            raise ValueError(
                "result site roles do not match the lattice-model support contract"
            )
        if result.support_contract_id != contract.contract_id:
            raise ValueError(
                "result support-contract ID does not match the lattice model"
            )
        if bool(result.material_scope_complete) != bool(
            contract.strict_requirements_satisfied
        ):
            raise ValueError(
                "result material-scope completeness does not match the support "
                "contract"
            )
        if bool(result.material_scope_fully_parameterized) != fully_parameterized:
            raise ValueError(
                "result fully-parameterized material scope does not match the "
                "support contract"
            )
    if target.shape != (n_site,) or not np.any(target):
        raise ValueError("the lattice model has no reportable TARGET sites")
    return (
        LatticeSiteModel1D(
            reference_potential=model.reference_potential,
            site_coordinates=model.site_coordinates[target],
            site_patches=model.site_patches[target],
            patch_starts=model.patch_starts[target],
            control_coordinates_s=model.control_coordinates_s,
            control_coordinates_u=model.control_coordinates_u,
            axial_sampling=model.axial_sampling,
            transverse_sampling=model.transverse_sampling,
            maximum_displacement=model.maximum_displacement,
            metadata={
                **dict(model.metadata),
                "rendering_scope": "target_sites_only_nuisance_reset_to_pristine",
            },
        ),
        target,
    )


def _target_structural_potentials_1d(
    experiment: SiliconGlancingExperiment1D,
    dataset: GlancingDataset1D,
    result: LatticeSiteReconstruction1D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render truth/result with nuisance deltas reset to their pristine state."""
    target_model, target = _target_only_lattice_model_1d(
        experiment,
        experiment.lattice_model,
        result=result,
    )
    if not np.array_equal(
        np.asarray(result.site_coordinates),
        np.asarray(experiment.lattice_model.site_coordinates),
    ):
        raise ValueError("result and experiment lattice sites do not match")
    truth = render_lattice_site_potential_1d(
        target_model,
        np.asarray(dataset.truth_vacancy_fractions)[target],
        dataset.truth_displacement_controls + dataset.truth_rigid_displacement,
    )
    recovered = render_lattice_site_potential_1d(
        target_model,
        np.asarray(result.vacancy_fractions)[target],
        result.displacement_controls + result.rigid_displacement,
    )
    target_support = _lattice_parameter_update_mask(
        np.asarray(target_model.site_patches),
        np.asarray(target_model.patch_starts),
        potential_shape=np.asarray(target_model.reference_potential).shape,
        maximum_displacement_A=float(target_model.maximum_displacement),
        ds=float(target_model.axial_sampling),
        du=float(target_model.transverse_sampling),
    )
    return np.asarray(truth), np.asarray(recovered), target, target_support


def reconstruction_metrics_1d(
    experiment: SiliconGlancingExperiment1D,
    dataset: GlancingDataset1D,
    results: Mapping[str, PotentialReconstruction1D | LatticeSiteReconstruction1D],
) -> dict[str, dict[str, float | int | bool | str]]:
    """Return comparable loss, timing, potential, vacancy, and strain metrics."""
    full_truth = np.asarray(dataset.potential)
    metrics = {}
    for name, result in results.items():
        if isinstance(result, LatticeSiteReconstruction1D):
            truth, recovered, target_sites, mask = (
                _target_structural_potentials_1d(experiment, dataset, result)
            )
            potential_scope = "target_sites_only_nuisance_reset_to_pristine"
        else:
            truth = full_truth
            recovered = np.asarray(result.potential)
            mask = np.asarray(experiment.reconstruction_mask)
            target_sites = None
            potential_scope = "pixel_update_region"
        row: dict[str, float | int | bool | str] = {
            "best update": int(result.best_update),
            "best validation loss": float(result.metadata["best_metric"]),
            "held-out audit loss": float(result.audit_loss),
            "potential NRMSE": float(
                np.linalg.norm(recovered[mask] - truth[mask])
                / np.linalg.norm(truth[mask])
            ),
            "potential metric scope": potential_scope,
        }
        history_updates = np.asarray(result.update_history)
        matches = np.flatnonzero(history_updates == result.best_update)
        if matches.size:
            row["time to best (s)"] = float(
                np.asarray(result.elapsed_time_history)[matches[0]]
            )
        if isinstance(result, LatticeSiteReconstruction1D):
            row["completed updates"] = int(result.completed_updates)
            row["numerically converged"] = bool(result.converged)
            row["stop reason"] = result.stop_reason
            if len(result.active_bound_fraction_history):
                row["active bound fraction"] = float(
                    result.active_bound_fraction_history[-1]
                )
            row["control bound fraction"] = float(
                result.metadata["best_control_bound_fraction"]
            )
            assert target_sites is not None
            predicted = (
                np.asarray(result.vacancy_fractions)[target_sites] >= 0.5
            )
            actual = (
                np.asarray(dataset.truth_vacancy_fractions)[target_sites] >= 0.5
            )
            tp = np.count_nonzero(predicted & actual)
            fp = np.count_nonzero(predicted & ~actual)
            fn = np.count_nonzero(~predicted & actual)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            row["vacancy F1"] = float(
                2.0 * precision * recall / max(precision + recall, 1e-12)
            )
            true_residual_displacements = np.asarray(
                lattice_site_displacements_1d(
                    experiment.variable_sites,
                    dataset.truth_displacement_controls,
                    experiment.lattice_model.control_coordinates_s,
                    experiment.lattice_model.control_coordinates_u,
                )
            )
            true_displacements = (
                true_residual_displacements
                + np.asarray(dataset.truth_rigid_displacement)
            )
            recovered_displacements = np.asarray(
                result.displaced_site_coordinates - result.site_coordinates
            )[target_sites]
            true_residual_displacements = true_residual_displacements[
                target_sites
            ]
            true_displacements = true_displacements[target_sites]
            recovered_residual_displacements = (
                recovered_displacements - np.asarray(result.rigid_displacement)
            )
            row["displacement RMSE (A)"] = float(
                np.sqrt(np.mean((recovered_displacements - true_displacements) ** 2))
            )
            row["residual displacement RMSE (A)"] = float(
                np.sqrt(
                    np.mean(
                        (
                            recovered_residual_displacements
                            - true_residual_displacements
                        )
                        ** 2
                    )
                )
            )
            row["rigid displacement error (A)"] = float(
                np.linalg.norm(
                    np.asarray(result.rigid_displacement)
                    - np.asarray(dataset.truth_rigid_displacement)
                )
            )
            row["specimen parameters"] = int(result.metadata["n_specimen_parameters"])
            row["target vacancy parameters"] = int(
                result.metadata.get(
                    "n_target_vacancy_parameters",
                    np.count_nonzero(target_sites),
                )
            )
            row["nuisance vacancy parameters"] = int(
                result.metadata.get("n_nuisance_vacancy_parameters", 0)
            )
            row["reportable structural parameters"] = (
                row["specimen parameters"]
                - row["nuisance vacancy parameters"]
            )
        else:
            row["specimen parameters"] = int(result.metadata["n_unknown_pixels"])
        metrics[name] = row
    return metrics


def screen_lattice_reconstruction_sensitivity_1d(
    experiment: SiliconGlancingExperiment1D,
    result: LatticeSiteReconstruction1D,
    counting_model: PoissonCountingModel1D,
    *,
    scan_indices: Sequence[int] | None = None,
    detector_mask: Any | None = None,
    options: SensitivityScreenOptions1D | None = None,
) -> LatticeSiteSensitivityScreen1D:
    """Run the necessary local Fisher screen on audit geometry by default."""
    selected = (
        np.asarray(experiment.audit_indices)
        if scan_indices is None
        else scan_indices
    )
    screen = lattice_site_sensitivity_screen_1d(
        experiment.lattice_model,
        result,
        experiment.input_probes,
        experiment.window_starts,
        experiment.window_length,
        experiment.propagation_kernel,
        experiment.axial_sampling,
        experiment.config.energy_eV,
        counting_model,
        scan_indices=selected,
        detector_mask=detector_mask,
        options=options,
    )
    return dataclass_replace(
        screen,
        metadata={
            **dict(screen.metadata),
            "structural_reporting_scope": "target_sites_only",
            "structural_reporting_site_mask": np.asarray(
                experiment.modeled_target_site_mask, dtype=bool
            ).tolist(),
            "nuisance_sites_profiled_in_forward_model": int(
                np.count_nonzero(experiment.modeled_nuisance_site_mask)
            ),
        },
    )


def save_experiment_results_1d(
    directory: str | Path,
    dataset: GlancingDataset1D,
    results: Mapping[str, PotentialReconstruction1D | LatticeSiteReconstruction1D],
) -> dict[str, Path]:
    """Save the selected scan and compact reconstructions, returning their paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"scan": directory / f"scan_{dataset.case}.npz"}
    save_glancing_scan_1d(paths["scan"], dataset.scan)
    for name, result in results.items():
        key = name.replace(" ", "_").replace("-", "_")
        path = directory / f"{key}_{dataset.case}.npz"
        if isinstance(result, LatticeSiteReconstruction1D):
            save_lattice_site_reconstruction_1d(path, result)
        else:
            save_potential_reconstruction_1d(path, result)
        paths[name] = path
    return paths


def _update_region_view(
    experiment: SiliconGlancingExperiment1D,
    *,
    mask: Array | None = None,
    margin_A: float = 1.0,
) -> tuple[tuple[slice, slice], list[float]]:
    """Return a tight, slightly padded view around the mutable support."""
    selected_mask = np.asarray(
        experiment.reconstruction_mask if mask is None else mask,
        dtype=bool,
    )
    if selected_mask.shape != np.asarray(experiment.reconstruction_mask).shape:
        raise ValueError("view mask must match the potential shape")
    rows, columns = np.where(selected_mask)
    if not rows.size:
        raise ValueError("view mask must contain at least one pixel")
    pad_s = int(np.ceil(margin_A / experiment.axial_sampling))
    pad_u = int(np.ceil(margin_A / experiment.transverse_sampling))
    s_slice = slice(
        max(int(rows.min()) - pad_s, 0),
        min(int(rows.max()) + pad_s + 1, selected_mask.shape[0]),
    )
    u_slice = slice(
        max(int(columns.min()) - pad_u, 0),
        min(int(columns.max()) + pad_u + 1, selected_mask.shape[1]),
    )
    s_A = np.asarray(experiment.axial_coordinates)[s_slice]
    u_A = np.asarray(experiment.transverse_coordinates)[u_slice]
    return (s_slice, u_slice), [s_A[0], s_A[-1], u_A[0], u_A[-1]]


def save_lattice_reconstruction_gif_1d(
    path: str | Path,
    experiment: SiliconGlancingExperiment1D,
    dataset: GlancingDataset1D,
    result: LatticeSiteReconstruction1D,
    *,
    fps: int = 20,
    frame_stride: int = 1,
    dpi: int = 100,
    writer: str = "auto",
    lattice_model: LatticeSiteModel1D | None = None,
    lattice_influence_mask: Array | None = None,
) -> Path:
    """Render reportable TARGET checkpoints beside truth and save them as a GIF.

    Chronological frames contain initialization and the requested checkpoint
    cadence.  The best-validation checkpoint is retained when frames are
    subsampled.  If that selected reconstruction is not the last optimizer
    iterate, it is appended once as a clearly labelled final summary frame so
    the animation ends on the best-validation TARGET component. Nuisance-site
    vacancy and displacement deltas are reset to pristine in every displayed
    frame; they remain active in the fitted forward-model potential.
    ``writer="auto"`` prefers FFmpeg's streaming pipe so hundreds of frames do
    not accumulate in memory, and falls back to Pillow when FFmpeg is absent.
    """
    import jax
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.animation import writers as animation_writers

    if not isinstance(experiment, SiliconGlancingExperiment1D):
        raise TypeError("experiment must be a SiliconGlancingExperiment1D")
    if not isinstance(dataset, GlancingDataset1D):
        raise TypeError("dataset must be a GlancingDataset1D")
    if not isinstance(result, LatticeSiteReconstruction1D):
        raise TypeError("result must be a LatticeSiteReconstruction1D")
    resolved_integers: dict[str, int] = {}
    for name, value in (
        ("fps", fps),
        ("frame_stride", frame_stride),
        ("dpi", dpi),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be an integer")
        try:
            resolved = operator.index(value)
        except TypeError as error:
            raise TypeError(f"{name} must be an integer") from error
        if resolved < 1:
            raise ValueError(f"{name} must be positive")
        resolved_integers[name] = resolved
    fps = resolved_integers["fps"]
    frame_stride = resolved_integers["frame_stride"]
    dpi = resolved_integers["dpi"]
    if not isinstance(writer, str):
        raise TypeError("writer must be 'auto', 'ffmpeg', or 'pillow'")
    writer = writer.strip().lower()
    if writer not in {"auto", "ffmpeg", "pillow"}:
        raise ValueError("writer must be 'auto', 'ffmpeg', or 'pillow'")
    if writer == "auto":
        writer = (
            "ffmpeg" if animation_writers.is_available("ffmpeg") else "pillow"
        )
    if writer == "ffmpeg" and not animation_writers.is_available("ffmpeg"):
        raise RuntimeError(
            "FFmpeg is unavailable; install it or select writer='pillow'"
        )

    updates = np.asarray(result.checkpoint_updates)
    vacancies = np.asarray(result.vacancy_fraction_history)
    controls = np.asarray(result.displacement_control_history)
    rigid = np.asarray(result.rigid_displacement_history)
    model = experiment.lattice_model if lattice_model is None else lattice_model
    if not isinstance(model, LatticeSiteModel1D):
        raise TypeError("lattice_model must be a LatticeSiteModel1D or None")
    n_site = int(np.asarray(model.site_coordinates).shape[0])
    expected_control_shape = (
        len(model.control_coordinates_s),
        len(model.control_coordinates_u),
        2,
    )
    if updates.ndim != 1 or updates.size == 0:
        raise ValueError(
            "the reconstruction has no checkpoints; set "
            "ReconstructionOptions1D.lattice_checkpoint_interval"
        )
    if not np.issubdtype(updates.dtype, np.integer):
        raise TypeError("checkpoint updates must be integers")
    if updates[0] != 0 or np.any(np.diff(updates) <= 0):
        raise ValueError(
            "checkpoint updates must start at zero and be strictly increasing"
        )
    if int(updates[-1]) != int(result.completed_updates):
        raise ValueError(
            "checkpoint history is truncated or inconsistent: its final update "
            "must equal completed_updates"
        )
    checkpoint_interval = result.metadata.get("checkpoint_interval")
    if checkpoint_interval == 1 and not np.array_equal(
        updates,
        np.arange(int(result.completed_updates) + 1, dtype=updates.dtype),
    ):
        raise ValueError(
            "checkpoint_interval=1 requires a complete checkpoint for every "
            "optimizer update"
        )
    if not 0 <= int(result.best_update) <= int(result.completed_updates):
        raise ValueError("best_update must lie between zero and completed_updates")
    if vacancies.shape != (updates.size, n_site):
        raise ValueError(
            "vacancy checkpoint history must have shape "
            f"{(updates.size, n_site)}"
        )
    if controls.shape != (updates.size, *expected_control_shape):
        raise ValueError(
            "displacement checkpoint history must have shape "
            f"{(updates.size, *expected_control_shape)}"
        )
    if rigid.shape == (0, 2):
        rigid = np.zeros((updates.size, 2), dtype=controls.dtype)
    elif rigid.shape != (updates.size, 2):
        raise ValueError(
            "rigid displacement history must have shape "
            f"{(updates.size, 2)} or {(0, 2)}"
        )
    for name, values in (
        ("vacancy checkpoint history", vacancies),
        ("displacement checkpoint history", controls),
        ("rigid displacement history", rigid),
    ):
        if np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite and real")
    if np.any(vacancies < 0.0) or np.any(vacancies > 1.0):
        raise ValueError("vacancy checkpoint history must lie in [0, 1]")

    potential_shape = np.asarray(model.reference_potential).shape
    full_truth_potential = np.asarray(dataset.potential)
    full_selected_potential = np.asarray(result.potential)
    if full_truth_potential.shape != potential_shape:
        raise ValueError("dataset potential must match the lattice-model shape")
    if full_selected_potential.shape != potential_shape:
        raise ValueError("result potential must match the lattice-model shape")
    if not np.all(np.isfinite(full_truth_potential)):
        raise ValueError("dataset potential must be finite")
    if not np.all(np.isfinite(full_selected_potential)):
        raise ValueError("result potential must be finite")
    if not np.array_equal(
        np.asarray(result.site_coordinates), np.asarray(model.site_coordinates)
    ):
        raise ValueError("result and experiment must use identical lattice sites")
    if not np.array_equal(
        np.asarray(result.control_coordinates_s),
        np.asarray(model.control_coordinates_s),
    ) or not np.array_equal(
        np.asarray(result.control_coordinates_u),
        np.asarray(model.control_coordinates_u),
    ):
        raise ValueError(
            "result and experiment must use identical displacement controls"
        )
    target_model, target_sites = _target_only_lattice_model_1d(
        experiment,
        model,
        result=result,
    )
    dataset_contract_id = dataset.scan.metadata.get("support_contract_id")
    if (
        dataset_contract_id is not None
        and dataset_contract_id != experiment.support_contract.contract_id
    ):
        raise ValueError(
            "dataset support-contract ID does not match the experiment"
        )
    display_vacancies = vacancies[:, target_sites]
    result_target_potential = np.asarray(
        render_lattice_site_potential_1d(
            target_model,
            np.asarray(result.vacancy_fractions)[target_sites],
            result.displacement_controls + result.rigid_displacement,
        )
    )
    if np.asarray(dataset.truth_vacancy_fractions).shape != (n_site,):
        raise ValueError(
            "dataset truth vacancies do not match the contract-bound lattice "
            "sites"
        )
    expected_control_shape = (
        np.asarray(target_model.control_coordinates_s).size,
        np.asarray(target_model.control_coordinates_u).size,
        2,
    )
    if np.asarray(dataset.truth_displacement_controls).shape != expected_control_shape:
        raise ValueError(
            "dataset truth displacement controls do not match the lattice model"
        )
    if np.asarray(dataset.truth_rigid_displacement).shape != (2,):
        raise ValueError("dataset truth rigid displacement must have shape (2,)")
    truth_potential = np.asarray(
        render_lattice_site_potential_1d(
            target_model,
            np.asarray(dataset.truth_vacancy_fractions)[target_sites],
            dataset.truth_displacement_controls
            + dataset.truth_rigid_displacement,
        )
    )
    truth_title = "Ground truth (TARGET updates; nuisance pristine)"

    path = Path(path)
    if path.suffix.lower() != ".gif":
        raise ValueError("path must end in .gif")
    path.parent.mkdir(parents=True, exist_ok=True)

    computed_target_support = _lattice_parameter_update_mask(
        np.asarray(target_model.site_patches),
        np.asarray(target_model.patch_starts),
        potential_shape=potential_shape,
        maximum_displacement_A=float(target_model.maximum_displacement),
        ds=float(target_model.axial_sampling),
        du=float(target_model.transverse_sampling),
    )
    experiment_target_support = np.asarray(
        experiment.target_lattice_influence_mask,
        dtype=bool,
    )
    if not np.array_equal(computed_target_support, experiment_target_support):
        raise ValueError(
            "computed TARGET support does not match the experiment support "
            "contract"
        )
    lattice_support = computed_target_support
    if lattice_influence_mask is not None:
        lattice_support = np.asarray(lattice_influence_mask, dtype=bool)
        if lattice_support.shape != potential_shape:
            raise ValueError(
                "lattice influence support must match the potential shape"
            )
        lattice_support &= computed_target_support
    if lattice_support.shape != potential_shape:
        raise ValueError("lattice influence support must match the potential shape")
    slices, extent = _update_region_view(experiment, mask=lattice_support)
    s_slice, u_slice = slices
    support = lattice_support[slices]
    cropped_model = LatticeSiteModel1D(
        reference_potential=target_model.reference_potential[slices],
        site_coordinates=target_model.site_coordinates,
        site_patches=target_model.site_patches,
        patch_starts=target_model.patch_starts
        - jnp.asarray([s_slice.start, u_slice.start]),
        control_coordinates_s=target_model.control_coordinates_s,
        control_coordinates_u=target_model.control_coordinates_u,
        axial_sampling=target_model.axial_sampling,
        transverse_sampling=target_model.transverse_sampling,
        maximum_displacement=target_model.maximum_displacement,
        metadata=target_model.metadata,
    )
    render_frame = jax.jit(
        lambda vacancy, control: render_lattice_site_potential_1d(
            cropped_model, vacancy, control
        )
    )
    truth_values = truth_potential[slices][support]
    if not truth_values.size:
        raise ValueError("lattice influence support contains no truth pixels")
    truth = np.where(support, truth_potential[slices], np.nan)
    vmax = float(np.percentile(truth_values, 99.5))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(np.abs(truth_values)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    frame_indices = list(range(0, updates.size, frame_stride))
    if frame_indices[-1] != updates.size - 1:
        frame_indices.append(updates.size - 1)
    best_update = int(result.best_update)
    best_matches = np.flatnonzero(updates == best_update)
    if best_matches.size:
        best_frame_index = int(best_matches[0])
        full_checkpoint_potential = np.asarray(
            render_lattice_site_potential_1d(
                model,
                vacancies[best_frame_index],
                controls[best_frame_index] + rigid[best_frame_index],
            )
        )
        full_influence = _lattice_parameter_update_mask(
            np.asarray(model.site_patches),
            np.asarray(model.patch_starts),
            potential_shape=potential_shape,
            maximum_displacement_A=float(model.maximum_displacement),
            ds=float(model.axial_sampling),
            du=float(model.transverse_sampling),
        )
        full_scale = max(
            1.0,
            float(np.max(np.abs(full_selected_potential[full_influence]))),
        )
        full_tolerance = (
            256.0 * np.finfo(full_checkpoint_potential.dtype).eps * full_scale
        )
        if not np.allclose(
            full_checkpoint_potential[full_influence],
            full_selected_potential[full_influence],
            rtol=256.0 * np.finfo(full_checkpoint_potential.dtype).eps,
            atol=full_tolerance,
        ):
            raise ValueError(
                "the best checkpoint does not reproduce result.potential inside "
                "the complete modeled influence support"
            )
        selected_target_potential = np.asarray(
            render_lattice_site_potential_1d(
                target_model,
                display_vacancies[best_frame_index],
                controls[best_frame_index] + rigid[best_frame_index],
            )
        )
        if best_frame_index not in frame_indices:
            frame_indices.append(best_frame_index)
            frame_indices.sort()
    else:
        best_frame_index = None
        selected_target_potential = result_target_potential

    if best_frame_index is not None:
        rendered_best = np.asarray(
            render_frame(
                display_vacancies[best_frame_index],
                controls[best_frame_index] + rigid[best_frame_index],
            )
        )
        comparison_scale = max(
            1.0,
            float(np.max(np.abs(selected_target_potential[slices][support]))),
        )
        tolerance = 256.0 * np.finfo(rendered_best.dtype).eps * comparison_scale
        if not np.allclose(
            rendered_best[support],
            selected_target_potential[slices][support],
            rtol=256.0 * np.finfo(rendered_best.dtype).eps,
            atol=tolerance,
        ):
            raise ValueError(
                "the best checkpoint does not reproduce the selected TARGET-only "
                "structural potential"
            )
    frame_entries: list[tuple[str, int]] = [
        ("history", index) for index in frame_indices
    ]
    if best_frame_index is None or int(updates[frame_indices[-1]]) != best_update:
        frame_entries.append(("selected", -1))

    first = frame_entries[0][1]
    total_controls = controls[first] + rigid[first]
    reconstructed = np.asarray(
        render_frame(display_vacancies[first], total_controls)
    )
    reconstructed = np.where(support, reconstructed, np.nan)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("white")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    truth_image = axes[0].imshow(
        truth.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )
    reconstruction_image = axes[1].imshow(
        reconstructed.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )
    axes[0].set_title(truth_title)
    first_suffix = (
        " (best validation)" if int(updates[first]) == best_update else ""
    )
    update_title = axes[1].set_title(
        f"TARGET-only reconstruction: update {updates[first]}{first_suffix}"
    )
    for axis in axes:
        axis.set(xlabel="s (A)", ylabel="u (A)")
    figure.colorbar(truth_image, ax=axes, label="projected potential")

    def update(frame_number: int):
        frame_kind, frame_index = frame_entries[frame_number]
        if frame_kind == "selected":
            potential = selected_target_potential[slices]
            title = (
                "Selected TARGET-only reconstruction: "
                f"update {best_update} (best validation)"
            )
        else:
            potential = np.asarray(
                render_frame(
                    display_vacancies[frame_index],
                    controls[frame_index] + rigid[frame_index],
                )
            )
            suffix = (
                " (best validation)"
                if int(updates[frame_index]) == best_update
                else ""
            )
            title = (
                "TARGET-only reconstruction: "
                f"update {updates[frame_index]}{suffix}"
            )
        reconstruction_image.set_data(np.where(support, potential, np.nan).T)
        update_title.set_text(title)
        return reconstruction_image, update_title

    animation = FuncAnimation(
        figure,
        update,
        frames=range(len(frame_entries)),
        interval=1_000 / fps,
        blit=False,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.stem}.",
            suffix=".gif",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        gif_writer = (
            FFMpegWriter(fps=fps, codec="gif")
            if writer == "ffmpeg"
            else PillowWriter(fps=fps)
        )
        animation.save(temporary_path, writer=gif_writer, dpi=dpi)
        temporary_path.replace(path)
    finally:
        plt.close(figure)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def plot_experiment_overview_1d(
    experiment: SiliconGlancingExperiment1D,
    dataset: GlancingDataset1D,
):
    """Plot the truth potential, beam coverage, support, and diffraction data."""
    import matplotlib.pyplot as plt

    s_A = np.asarray(experiment.axial_coordinates)
    u_A = np.asarray(experiment.transverse_coordinates)
    extent = [s_A[0], s_A[-1], u_A[0], u_A[-1]]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), constrained_layout=True)
    axes[0].imshow(
        np.asarray(dataset.potential).T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    axes[0].set_title(f"Truth: {dataset.case.replace('_', ' ')}")
    coverage = axes[1].imshow(
        np.asarray(experiment.beam_path_scan_coverage).T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    fig.colorbar(coverage, ax=axes[1], label="scans")
    axes[1].set_title("Beam-path coverage")
    axes[2].imshow(
        np.asarray(experiment.reconstruction_mask).T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray_r",
    )
    axes[2].set_title("Conservative update region")
    detector = np.asarray(experiment.detector_angles)
    positive = detector > 0.0
    axes[3].imshow(
        np.log10(np.asarray(dataset.intensities)[:, positive] + 1e-30),
        origin="lower",
        aspect="auto",
        extent=[
            detector[positive][0],
            detector[positive][-1],
            float(experiment.scan_coordinates[0]),
            float(experiment.scan_coordinates[-1]),
        ],
        cmap="magma",
    )
    axes[3].set(
        title="Noiseless diffraction",
        xlabel="detector angle (mrad)",
        ylabel="scan coordinate (A)",
    )
    for axis in axes[:3]:
        axis.set(xlabel="s (A)", ylabel="u (A)")
    return fig


def plot_reconstruction_comparison_1d(
    experiment: SiliconGlancingExperiment1D,
    dataset: GlancingDataset1D,
    results: Mapping[str, PotentialReconstruction1D | LatticeSiteReconstruction1D],
):
    """Plot recovered potentials and convergence against updates and time."""
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("results must contain at least one reconstruction")
    result_supports = {
        name: np.asarray(
            experiment.target_lattice_influence_mask
            if isinstance(result, LatticeSiteReconstruction1D)
            else experiment.reconstruction_mask,
            dtype=bool,
        )
        for name, result in results.items()
    }
    shared_support = np.logical_or.reduce(tuple(result_supports.values()))
    slices, extent = _update_region_view(experiment, mask=shared_support)
    lattice_results = {
        name: result
        for name, result in results.items()
        if isinstance(result, LatticeSiteReconstruction1D)
    }
    pixel_results = {
        name: result
        for name, result in results.items()
        if not isinstance(result, LatticeSiteReconstruction1D)
    }
    target_structural_pairs = {
        name: _target_structural_potentials_1d(experiment, dataset, result)
        for name, result in lattice_results.items()
    }
    truth_images: list[tuple[str, Array, np.ndarray]] = []
    if pixel_results:
        truth_images.append(
            ("ground truth (full specimen)", dataset.potential, shared_support)
        )
    if lattice_results:
        first_target_truth = next(iter(target_structural_pairs.values()))[0]
        truth_images.append(
            (
                "ground truth (TARGET updates; nuisance pristine)",
                first_target_truth,
                np.asarray(experiment.target_lattice_influence_mask, dtype=bool),
            )
        )
    n_panels = len(truth_images) + len(results)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.5 * n_panels, 3.8),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]
    target_only_potentials = {
        name: values[1] for name, values in target_structural_pairs.items()
    }
    vmax = np.percentile(
        np.asarray(dataset.potential)[shared_support],
        99.5,
    )
    display_entries: list[tuple[str, Array, np.ndarray]] = list(truth_images)
    display_entries.extend(
        (
            name,
            target_only_potentials.get(name, result.potential),
            result_supports[name],
        )
        for name, result in results.items()
    )
    for axis, (name, potential, entry_support) in zip(axes, display_entries):
        support = entry_support[slices]
        visible = np.where(support, np.asarray(potential)[slices], np.nan)
        image = axis.imshow(
            visible.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        title = (
            f"{name} (TARGET only)"
            if name in target_only_potentials
            else name
        )
        axis.set(title=title, xlabel="s (A)", ylabel="u (A)")
        fig.colorbar(image, ax=axis)

    convergence, convergence_axes = plt.subplots(
        1, 2, figsize=(10, 3.5), constrained_layout=True
    )
    for name, result in results.items():
        convergence_axes[0].semilogy(
            result.update_history, result.validation_loss_history, label=name
        )
        convergence_axes[1].semilogy(
            result.elapsed_time_history,
            result.validation_loss_history,
            label=name,
        )
    convergence_axes[0].set(
        xlabel="optimizer update", ylabel="validation amplitude loss"
    )
    convergence_axes[1].set(
        xlabel="elapsed time (s)", ylabel="validation amplitude loss"
    )
    for axis in convergence_axes:
        axis.legend()
    return fig, convergence


def plot_lattice_reconstruction_1d(
    experiment: SiliconGlancingExperiment1D,
    result: LatticeSiteReconstruction1D,
):
    """Plot recovered vacancy fractions, displacements, and strain components."""
    import matplotlib.pyplot as plt

    _, target = _target_only_lattice_model_1d(
        experiment,
        experiment.lattice_model,
        result=result,
    )
    sites = np.asarray(result.site_coordinates)[target]
    vacancies = np.asarray(result.vacancy_fractions)[target]
    displacements = np.asarray(
        result.displaced_site_coordinates - result.site_coordinates
    )[target]
    _, view_extent = _update_region_view(
        experiment, mask=experiment.target_lattice_influence_mask
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    axes[0].scatter(sites[:, 0], vacancies, s=10, c=vacancies, cmap="magma")
    axes[0].set(
        xlabel="site s (A)",
        ylabel="vacancy fraction",
        xlim=view_extent[:2],
        title="TARGET-site vacancy fractions",
    )
    axes[1].quiver(
        sites[:, 0],
        sites[:, 1],
        displacements[:, 0],
        displacements[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    axes[1].set(
        xlabel="site s (A)",
        ylabel="site u (A)",
        title="Displacement field",
        xlim=view_extent[:2],
        ylim=view_extent[2:],
    )

    controls = np.asarray(result.displacement_controls)
    control_s = np.asarray(result.control_coordinates_s)
    control_u = np.asarray(result.control_coordinates_u)
    strain_ss = np.gradient(controls[..., 0], control_s, axis=0)
    strain_uu = np.gradient(controls[..., 1], control_u, axis=1)
    strain_su = 0.5 * (
        np.gradient(controls[..., 0], control_u, axis=1)
        + np.gradient(controls[..., 1], control_s, axis=0)
    )
    strain_fig, strain_axes = plt.subplots(
        1, 3, figsize=(12, 3.5), constrained_layout=True
    )
    limit = max(
        np.max(np.abs(strain_ss)),
        np.max(np.abs(strain_uu)),
        np.max(np.abs(strain_su)),
        1e-12,
    )
    extent = [control_s[0], control_s[-1], control_u[0], control_u[-1]]
    for axis, strain, title in zip(
        strain_axes,
        (strain_ss, strain_uu, strain_su),
        (r"$\epsilon_{ss}$", r"$\epsilon_{uu}$", r"$\epsilon_{su}$"),
    ):
        image = axis.imshow(
            strain.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_xlim(view_extent[:2])
        axis.set_ylim(view_extent[2:])
        axis.set(title=title, xlabel="s (A)", ylabel="u (A)")
        axis.set(xlim=view_extent[:2], ylim=view_extent[2:])
        strain_fig.colorbar(image, ax=axis, label="strain")
    return fig, strain_fig


def plot_lattice_sensitivity_screen_1d(
    screen: LatticeSiteSensitivityScreen1D,
):
    """Plot optimistic local uncertainty bounds and the necessary pass mask."""
    import matplotlib.pyplot as plt

    if not isinstance(screen, LatticeSiteSensitivityScreen1D):
        raise TypeError("screen must be a LatticeSiteSensitivityScreen1D")
    reportable = np.asarray(
        screen.metadata.get(
            "structural_reporting_site_mask",
            np.ones(len(screen.site_coordinates), dtype=bool),
        ),
        dtype=bool,
    )
    if reportable.shape != (len(screen.site_coordinates),) or not np.any(
        reportable
    ):
        raise ValueError("sensitivity reportable-site mask is invalid")
    sites = np.asarray(screen.site_coordinates)[reportable]
    vacancy_error = np.asarray(
        screen.vacancy_standard_error_lower_bound
    )[reportable]
    displacement_error = np.max(
        np.asarray(screen.displacement_standard_error_lower_bound_A)[reportable],
        axis=1,
    )
    sensitive = np.asarray(screen.site_sensitive)[reportable]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    for axis, values, title, label in (
        (
            axes[0],
            vacancy_error,
            "Conditional vacancy uncertainty",
            r"$\log_{10}$ lower-bound SE",
        ),
        (
            axes[1],
            displacement_error,
            "Conditional displacement uncertainty",
            r"$\log_{10}$ lower-bound SE (A)",
        ),
    ):
        log_values = np.full_like(values, np.nan, dtype=float)
        finite = np.isfinite(values) & (values > 0.0)
        log_values[finite] = np.log10(values[finite])
        points = axis.scatter(
            sites[:, 0], sites[:, 1], c=log_values, s=12, cmap="viridis"
        )
        figure.colorbar(points, ax=axis, label=label)
        axis.set(title=title, xlabel="s (A)", ylabel="u (A)")
    axes[2].scatter(
        sites[:, 0],
        sites[:, 1],
        c=sensitive.astype(int),
        s=12,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
    )
    axes[2].set(
        title="Necessary local sensitivity pass",
        xlabel="s (A)",
        ylabel="u (A)",
    )
    return figure


def make_glancing_scan_viewer_1d(
    experiment: SiliconGlancingExperiment1D,
    dataset: GlancingDataset1D,
    *,
    source_potentials: Mapping[str, Array] | None = None,
    scan_stride: int = 1,
    axial_stride: int = 8,
    transverse_stride: int = 2,
):
    """Return the interactive side-view, exit-wave, and detector diagnostic.

    The returned widget retains the original notebook behavior: a scan slider,
    optional access to uncached positions, a potential-source selector, the
    global specimen side view, and linked exit-wave and diffraction panels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    try:
        import ipywidgets as widgets
    except ImportError as exc:  # pragma: no cover - notebook dependency
        raise ImportError(
            "the interactive scan viewer requires ipywidgets and ipympl"
        ) from exc

    if scan_stride < 1:
        raise ValueError("scan_stride must be positive")
    if source_potentials is None:
        source_potentials = {"ground truth": dataset.potential}
    if not source_potentials:
        raise ValueError("source_potentials must not be empty")

    n_scan = len(experiment.window_starts)
    cached_indices = np.unique(
        np.concatenate(
            [
                np.arange(0, n_scan, scan_stride, dtype=np.int32),
                np.asarray([0, n_scan - 1], dtype=np.int32),
            ]
        )
    )
    caches = {
        name: simulate_glancing_sideview_cache_1d(
            potential,
            experiment.input_probes,
            experiment.window_starts,
            experiment.window_length,
            experiment.propagation_kernel,
            experiment.axial_sampling,
            experiment.config.energy_eV,
            jnp.asarray(cached_indices),
            transverse_coordinates=experiment.transverse_coordinates,
            scan_coordinates=experiment.scan_coordinates,
            axial_stride=axial_stride,
            transverse_stride=transverse_stride,
            metadata={"potential_source": name, "model": "Angular spectrum"},
        )
        for name, potential in source_potentials.items()
    }
    on_demand: OrderedDict[tuple[str, int], GlancingSideviewCache1D] = OrderedDict()

    def cache_row(cache: GlancingSideviewCache1D, scan_index: int) -> int | None:
        matches = np.flatnonzero(np.asarray(cache.scan_indices) == scan_index)
        return int(matches[0]) if matches.size else None

    def cache_for(source: str, scan_index: int) -> tuple[GlancingSideviewCache1D, int]:
        cache = caches[source]
        row = cache_row(cache, scan_index)
        if row is not None:
            return cache, row
        key = (source, scan_index)
        if key not in on_demand:
            on_demand[key] = simulate_glancing_sideview_cache_1d(
                source_potentials[source],
                experiment.input_probes,
                experiment.window_starts,
                experiment.window_length,
                experiment.propagation_kernel,
                experiment.axial_sampling,
                experiment.config.energy_eV,
                jnp.asarray([scan_index]),
                transverse_coordinates=experiment.transverse_coordinates,
                scan_coordinates=experiment.scan_coordinates,
                axial_stride=axial_stride,
                transverse_stride=transverse_stride,
            )
            while len(on_demand) > 4:
                on_demand.popitem(last=False)
        return on_demand[key], 0

    side_max = max(
        float(np.max(np.asarray(cache.sideview_intensities)))
        for cache in caches.values()
    )
    exit_max = max(
        float(np.max(np.abs(np.asarray(cache.exit_waves)) ** 2))
        for cache in caches.values()
    )
    detector_max = max(
        float(np.max(np.asarray(dataset.intensities))),
        *[
            float(np.max(np.asarray(cache.detector_intensities)))
            for cache in caches.values()
        ],
    )
    s_A = np.asarray(experiment.axial_coordinates)
    u_A = np.asarray(experiment.transverse_coordinates)
    detector_angles = np.asarray(experiment.detector_angles)
    positive_detector = detector_angles > 0.0
    tilt = -np.deg2rad(experiment.config.glancing_angle_deg)
    potential_extent = [s_A[0], s_A[-1], u_A[0], u_A[-1]]

    def draw(scan_index: int, source: str, figure=None):
        cache, row = cache_for(source, int(scan_index))
        potential = np.asarray(source_potentials[source])
        side_intensity = np.asarray(cache.sideview_intensities[row])
        side_field = np.asarray(cache.sideview_wavefields[row])
        exit_intensity = np.abs(np.asarray(cache.exit_waves[row])) ** 2
        detector_intensity = np.asarray(cache.detector_intensities[row])
        start = int(experiment.window_starts[scan_index])
        surface_s_A = float(experiment.scan_coordinates[scan_index])
        local_s_A = start * experiment.axial_sampling + np.asarray(
            cache.local_s_coordinates
        )
        centerline_u_A = np.tan(tilt) * (local_s_A - surface_s_A)
        side_u_A = np.asarray(cache.sideview_u_coordinates)
        fixed_side = np.full(
            (len(s_A), side_intensity.shape[1]), np.nan, dtype=side_intensity.dtype
        )
        for block, intensity_row in enumerate(side_intensity):
            block_start = start + block * axial_stride
            block_stop = min(block_start + axial_stride, len(s_A))
            fixed_side[block_start:block_stop] = intensity_row
        fixed_side = np.ma.masked_invalid(fixed_side)

        if figure is None:
            with plt.ioff():
                figure = plt.figure(figsize=(14, 6.5), constrained_layout=True)
        else:
            figure.clear()
            figure.set_constrained_layout(True)
        grid = figure.add_gridspec(2, 2, width_ratios=(2.25, 1.0))
        side = figure.add_subplot(grid[:, 0])
        exit_axis = figure.add_subplot(grid[0, 1])
        detector_axis = figure.add_subplot(grid[1, 1])

        peak = float(np.max(potential))
        if peak > 0.0:
            potential_cmap = plt.get_cmap("plasma").copy()
            potential_cmap.set_bad(alpha=0.0)
            side.imshow(
                np.ma.masked_where(potential.T <= 0.0, potential.T),
                origin="lower",
                aspect="auto",
                extent=potential_extent,
                cmap=potential_cmap,
                norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=peak),
                alpha=0.5,
                zorder=2,
            )
        intensity_cmap = plt.get_cmap("viridis").copy()
        intensity_cmap.set_bad(alpha=0.0)
        side_image = side.imshow(
            fixed_side.T,
            origin="lower",
            aspect="auto",
            extent=[s_A[0], s_A[-1], side_u_A[0], side_u_A[-1]],
            cmap=intensity_cmap,
            norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=max(side_max, 1e-30)),
            zorder=1,
        )
        side.plot(
            local_s_A,
            centerline_u_A,
            color="white",
            linestyle="--",
            linewidth=1.2,
            label="nominal beam centreline",
        )
        side.scatter(
            [surface_s_A],
            [0.0],
            s=45,
            color="cyan",
            edgecolor="black",
            linewidth=0.6,
            zorder=4,
            label="surface landing position",
        )
        side.axhline(0.0, color="cyan", linewidth=0.8, alpha=0.8)
        side.axhline(
            -experiment.config.slab_depth_A,
            color="cyan",
            linewidth=0.8,
            alpha=0.8,
        )
        side.set(
            xlim=(float(s_A[0]), float(s_A[-1])),
            ylim=(-75.0, 100.0),
            title=f"Scan {scan_index}: beam landing at s = {surface_s_A:.2f} A",
            xlabel="global specimen coordinate s (A)",
            ylabel="u (A)",
        )
        side.legend(loc="lower left")
        figure.colorbar(side_image, ax=side, label="intensity")

        vacuum = u_A > 0.0
        exit_axis.plot(u_A[vacuum], exit_intensity[vacuum])
        exit_axis.set(
            xlim=(0.0, float(u_A[-1])),
            ylim=(0.0, max(exit_max, 1e-30)),
            title="Exit-wave intensity",
            xlabel="u (A)",
            ylabel="intensity",
        )
        detector_axis.plot(
            detector_angles[positive_detector],
            detector_intensity[positive_detector] + 1e-30,
            label=source,
        )
        detector_axis.plot(
            detector_angles[positive_detector],
            np.asarray(dataset.intensities[scan_index])[positive_detector] + 1e-30,
            color="k",
            alpha=0.5,
            label="measured",
        )
        detector_axis.set(
            ylim=(0.0, max(detector_max, 1e-30)),
            title="Recorded far field",
            xlabel="detector angle (mrad)",
            ylabel="intensity",
        )
        detector_axis.legend()
        figure.suptitle(
            f"{source}; scan coordinate {surface_s_A:.2f} A; "
            f"stored phase range {np.ptp(np.angle(side_field)):.2f} rad"
        )
        return figure

    def scan_options(indices):
        return tuple(
            (
                f"{index:03d} — {float(experiment.scan_coordinates[index]):8.2f} A",
                int(index),
            )
            for index in indices
        )

    cached_options = scan_options(cached_indices)
    all_options = scan_options(range(n_scan))
    scan_selector = widgets.SelectionSlider(
        options=cached_options,
        value=int(cached_indices[0]),
        description="scan position",
        continuous_update=False,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="720px"),
    )
    include_uncached = widgets.Checkbox(
        value=False,
        description="include uncached positions (recompute on selection)",
        indent=False,
        style={"description_width": "initial"},
    )
    source_selector = widgets.ToggleButtons(
        options=tuple(source_potentials), description="potential"
    )

    def update_scan_options(change):
        previous = int(scan_selector.value)
        indices = range(n_scan) if change["new"] else cached_indices
        nearest = min(indices, key=lambda index: abs(index - previous))
        with scan_selector.hold_trait_notifications():
            scan_selector.options = all_options if change["new"] else cached_options
            scan_selector.value = nearest

    include_uncached.observe(update_scan_options, names="value")
    rendered = [draw(scan_selector.value, source_selector.value)]

    def render(_change=None):
        draw(
            scan_selector.value,
            source_selector.value,
            figure=rendered[0],
        )
        rendered[0].canvas.draw_idle()

    scan_selector.observe(render, names="value")
    source_selector.observe(render, names="value")
    controls = widgets.VBox([scan_selector, include_uncached, source_selector])
    return widgets.VBox([controls, rendered[0].canvas])
