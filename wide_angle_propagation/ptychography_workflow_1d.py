"""Readable high-level workflow for the glancing ptychography notebook.

The numerical inverse methods remain in :mod:`ptychography_1d`.  This module
collects the experiment construction, matched synthetic data, comparison
baselines, and notebook visualization behind a small public API so the example
notebook can concentrate on the scientific sequence rather than bookkeeping.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from .propagation_methods import (
    angular_spectrum_propagation_kernel_1d,
    energy2wavelength,
    interaction_constant,
)
from .ptychography_1d import (
    GlancingScan1D,
    GlancingSideviewCache1D,
    LatticeSiteModel1D,
    LatticeSiteReconstruction1D,
    PotentialReconstruction1D,
    lattice_site_displacements_1d,
    reconstruct_lattice_site_potential_1d,
    reconstruct_potential_1d,
    render_lattice_site_potential_1d,
    save_glancing_scan_1d,
    save_lattice_site_reconstruction_1d,
    save_potential_reconstruction_1d,
    simulate_glancing_scan_1d,
    simulate_glancing_sideview_cache_1d,
)
from .sideview_geometry import make_tilted_gaussian_beam_1d


__all__ = [
    "GlancingDataset1D",
    "ReconstructionOptions1D",
    "SiliconGlancingConfig1D",
    "SiliconGlancingExperiment1D",
    "build_silicon_glancing_experiment_1d",
    "make_glancing_scan_viewer_1d",
    "plot_experiment_overview_1d",
    "plot_lattice_reconstruction_1d",
    "plot_reconstruction_comparison_1d",
    "reconstruct_experiment_1d",
    "reconstruction_metrics_1d",
    "save_experiment_results_1d",
    "simulate_experiment_1d",
]


Array = Any


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
    beam_path_radius_waists: float = 3.0
    minimum_scan_coverage: int = 1
    # ``landing`` updates only a shallow strip around the scan landing range;
    # ``beam_path`` retains the earlier tube following every nominal ray.
    update_region: str = "landing"
    landing_radius_waists: float = 3.0
    landing_depth_A: float | None = None
    atomic_template_cutoff_A: float = 8.0
    cutoff_check_A: float = 10.0
    maximum_displacement_A: float = 0.5
    displacement_control_spacing_A: float = 25.0


@dataclass(frozen=True)
class SiliconGlancingExperiment1D:
    """Complete known geometry and lattice model shared by both datasets."""

    config: SiliconGlancingConfig1D
    pristine_potential: Array
    lattice_model: LatticeSiteModel1D
    truth_potentials: Mapping[str, Array]
    truth_vacancy_fractions: Array
    truth_displacement_controls: Mapping[str, Array]
    vacancy_site_indices: Array
    variable_sites: Array
    reconstruction_mask: Array
    beam_path_scan_coverage: Array
    input_probes: Array
    propagation_kernel: Array
    window_starts: Array
    window_length: int
    scan_coordinates: Array
    axial_coordinates: Array
    transverse_coordinates: Array
    detector_angles: Array
    validation_indices: Array
    cutoff_check_potential: Array
    axial_sampling: float
    transverse_sampling: float
    summary: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GlancingDataset1D:
    """One selected truth case, its diffraction data, and mismatch diagnostics."""

    case: str
    potential: Array
    scan: GlancingScan1D
    truth_displacement_controls: Array
    zero_exterior_amplitude_nrmse: float
    template_cutoff_amplitude_nrmse: float

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
    evaluation_batch_size: int = 10
    rematerialize: bool = True
    seed: int = 0
    progress: bool = True


def _tile_to_shape(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    repeats = tuple(
        int(np.ceil(target / size)) for target, size in zip(shape, array.shape)
    )
    slices = tuple(slice(0, target) for target in shape)
    return np.tile(array, repeats)[slices]


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


def _patches_for_sites(
    site_coordinates: np.ndarray,
    template: np.ndarray,
    half_shape: tuple[int, int],
    *,
    s_A: np.ndarray,
    u_A: np.ndarray,
    ds: float,
    du: float,
    material_u_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import shift as shift_image

    half_s, half_u = half_shape
    patches = []
    starts = []
    for site_s_A, site_u_A in np.asarray(site_coordinates):
        site_s_pixel = (site_s_A - s_A[0]) / ds
        site_u_pixel = (site_u_A - u_A[0]) / du
        center_s = int(np.rint(site_s_pixel))
        center_u = int(np.rint(site_u_pixel))
        shifted = shift_image(
            template,
            shift=(site_s_pixel - center_s, site_u_pixel - center_u),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        start_s = center_s - half_s
        start_u = center_u - half_u
        global_u_indices = start_u + np.arange(template.shape[1])
        valid_u = (global_u_indices >= 0) & (global_u_indices < len(u_A))
        clipped_u = np.clip(global_u_indices, 0, len(u_A) - 1)
        valid_u &= material_u_mask[clipped_u]
        shifted[:, ~valid_u] = 0.0
        patches.append(shifted)
        starts.append((start_s, start_u))
    return np.asarray(patches), np.asarray(starts, dtype=np.int32)


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
    mask = material[None, :] & (coverage >= config.minimum_scan_coverage)
    return mask, coverage


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
) -> tuple[np.ndarray, np.ndarray]:
    control_s, control_u = np.meshgrid(control_s_A, control_u_A, indexing="ij")
    envelope = np.exp(
        -0.5 * ((control_s - defect_center_s_A) / 75.0) ** 2
        - 0.5 * ((control_u + 15.0) / 20.0) ** 2
    )
    axial = envelope / max(float(np.max(np.abs(envelope))), 1e-12)
    transverse = ((control_s - defect_center_s_A) / 75.0) * envelope
    transverse /= max(float(np.max(np.abs(transverse))), 1e-12)
    strained = np.stack([0.25 * axial, 0.15 * transverse], axis=-1)
    return np.zeros_like(strained), strained


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
    abtem.config.set({"device": "cpu", "precision": "float64"})

    unit = bulk("Si", "diamond", a=config.si_lattice_A, cubic=True)
    unit.pbc = [True, True, True]
    potential_builder = abtem.Potential(
        unit,
        sampling=(config.sampling_u_A, config.sampling_s_A),
        slice_thickness=float(unit.cell.lengths()[1]),
        projection="finite",
        parametrization="lobato",
        plane="xz",
        device="cpu",
    )
    unit_projected = np.asarray(potential_builder.build(lazy=False).array)[0]
    unit_potential = unit_projected / float(unit.cell.lengths()[1])
    top_indices = np.flatnonzero(
        np.isclose(unit.positions[:, 0], np.max(unit.positions[:, 0]))
    )
    vacancy_unit = unit.copy()
    del vacancy_unit[top_indices]
    vacancy_projected = np.asarray(
        abtem.Potential(
            vacancy_unit,
            sampling=(config.sampling_u_A, config.sampling_s_A),
            slice_thickness=float(unit.cell.lengths()[1]),
            projection="finite",
            parametrization="lobato",
            plane="xz",
            device="cpu",
        )
        .build(lazy=False)
        .array
    )[0]
    missing_top = (unit_projected - vacancy_projected) / float(unit.cell.lengths()[1])

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
    surface_index = np.flatnonzero(u_A <= 0.0)[-1]
    top_layer_index = int(np.argmax(np.sum(missing_top, axis=1)))
    roll_u = (surface_index - top_layer_index) % unit_potential.shape[0]
    bulk_potential = np.roll(
        _tile_to_shape(unit_potential, (n_u, window_length)), roll_u, axis=0
    ).T
    material_u = (u_A >= -config.slab_depth_A) & (u_A <= 0.0)
    pristine = bulk_potential * material_u[None, :]

    scan_coordinates = np.linspace(
        config.scan_start_A, config.scan_stop_A, config.n_scans
    )
    beam_path_mask, coverage = _beam_path_region(config, s_A, u_A, scan_coordinates)
    if config.update_region == "landing":
        reconstruction_mask = _landing_region(config, s_A, u_A, scan_coordinates)
    elif config.update_region == "beam_path":
        reconstruction_mask = beam_path_mask
    else:
        raise ValueError("update_region must be 'landing' or 'beam_path'")
    all_sites = _projected_si_sites(
        unit,
        lattice_A=config.si_lattice_A,
        length_A=window_length * ds,
        depth_A=config.slab_depth_A,
    )
    site_s_indices = np.rint((all_sites[:, 0] - s_A[0]) / ds).astype(int)
    site_u_indices = np.rint((all_sites[:, 1] - u_A[0]) / du).astype(int)
    in_grid = (
        (site_s_indices >= 0)
        & (site_s_indices < len(s_A))
        & (site_u_indices >= 0)
        & (site_u_indices < len(u_A))
    )
    variable_sites = all_sites[
        in_grid
        & reconstruction_mask[
            np.clip(site_s_indices, 0, len(s_A) - 1),
            np.clip(site_u_indices, 0, len(u_A) - 1),
        ]
    ]
    if len(variable_sites) == 0:
        raise ValueError("the selected update region contains no silicon sites")

    control_s_A = _control_axis(
        variable_sites[:, 0], config.displacement_control_spacing_A
    )
    control_u_A = _control_axis(
        variable_sites[:, 1], config.displacement_control_spacing_A
    )
    template, half_shape = _projected_si_template(
        config, ds=ds, du=du, cutoff_A=config.atomic_template_cutoff_A
    )
    patches, patch_starts = _patches_for_sites(
        variable_sites,
        template,
        half_shape,
        s_A=s_A,
        u_A=u_A,
        ds=ds,
        du=du,
        material_u_mask=material_u,
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
            "atomic_template_cutoff_A": config.atomic_template_cutoff_A,
            "displacement_control_spacing_A": config.displacement_control_spacing_A,
            "update_region": config.update_region,
        },
    )

    top_sites = np.flatnonzero(np.isclose(variable_sites[:, 1], 0.0, atol=0.51 * du))
    if len(top_sites) < config.defect_width_sites:
        raise ValueError("not enough illuminated top-layer sites for the defect")
    vacancy_sites = top_sites[
        np.argsort(np.abs(variable_sites[top_sites, 0] - config.defect_center_s_A))[
            : config.defect_width_sites
        ]
    ]
    vacancy_fractions = np.zeros(len(variable_sites), dtype=float)
    vacancy_fractions[vacancy_sites] = 1.0
    zero_controls, strained_controls = _truth_controls(
        control_s_A, control_u_A, config.defect_center_s_A
    )
    truth_controls = {
        "vacancy": jnp.asarray(zero_controls),
        "vacancy_plus_strain": jnp.asarray(strained_controls),
    }
    truth_potentials = {
        case: render_lattice_site_potential_1d(
            lattice_model, jnp.asarray(vacancy_fractions), controls
        )
        for case, controls in truth_controls.items()
    }

    larger_template, larger_half_shape = _projected_si_template(
        config, ds=ds, du=du, cutoff_A=config.cutoff_check_A
    )
    larger_patches, larger_starts = _patches_for_sites(
        variable_sites[vacancy_sites],
        larger_template,
        larger_half_shape,
        s_A=s_A,
        u_A=u_A,
        ds=ds,
        du=du,
        material_u_mask=material_u,
    )
    cutoff_model = LatticeSiteModel1D(
        reference_potential=jnp.asarray(pristine),
        site_coordinates=jnp.asarray(variable_sites[vacancy_sites]),
        site_patches=jnp.asarray(larger_patches),
        patch_starts=jnp.asarray(larger_starts),
        control_coordinates_s=jnp.asarray([control_s_A[0], control_s_A[-1]]),
        control_coordinates_u=jnp.asarray([control_u_A[0], control_u_A[-1]]),
        axial_sampling=ds,
        transverse_sampling=du,
        maximum_displacement=0.0,
    )
    cutoff_check = render_lattice_site_potential_1d(
        cutoff_model,
        jnp.ones(len(vacancy_sites)),
        jnp.zeros((2, 2, 2)),
    )

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
    validation_indices = np.arange(
        0, config.n_scans, config.validation_stride, dtype=np.int32
    )
    n_controls = int(np.prod(strained_controls.shape))
    n_parameters = len(variable_sites) + n_controls
    summary = {
        "potential shape": pristine.shape,
        "sampling (ds, du) A": (ds, du),
        "scan count": config.n_scans,
        "pixel unknowns": int(reconstruction_mask.sum()),
        "update region": config.update_region,
        "landing radius (A)": config.landing_radius_waists * config.beam_waist_A,
        "landing depth (A)": (
            config.landing_radius_waists * config.beam_waist_A
            if config.landing_depth_A is None
            else config.landing_depth_A
        ),
        "variable Si sites": len(variable_sites),
        "displacement controls": n_controls,
        "lattice parameters": n_parameters,
        "pixel / lattice reduction": float(
            reconstruction_mask.sum() / max(n_parameters, 1)
        ),
        "explicit vacancy sites": len(vacancy_sites),
    }
    return SiliconGlancingExperiment1D(
        config=config,
        pristine_potential=jnp.asarray(pristine),
        lattice_model=lattice_model,
        truth_potentials=truth_potentials,
        truth_vacancy_fractions=jnp.asarray(vacancy_fractions),
        truth_displacement_controls=truth_controls,
        vacancy_site_indices=jnp.asarray(vacancy_sites),
        variable_sites=jnp.asarray(variable_sites),
        reconstruction_mask=jnp.asarray(reconstruction_mask),
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
        validation_indices=jnp.asarray(validation_indices),
        cutoff_check_potential=cutoff_check,
        axial_sampling=ds,
        transverse_sampling=du,
        summary=summary,
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


def _amplitude_nrmse(predicted: Array, reference: Array) -> float:
    predicted_amplitude = np.sqrt(np.asarray(predicted) + 1e-12)
    reference_amplitude = np.sqrt(np.asarray(reference) + 1e-12)
    return float(
        np.linalg.norm(predicted_amplitude - reference_amplitude)
        / np.linalg.norm(reference_amplitude)
    )


def simulate_experiment_1d(
    experiment: SiliconGlancingExperiment1D,
    case: str = "vacancy",
    *,
    batch_size: int = 10,
) -> GlancingDataset1D:
    """Simulate one truth case and evaluate the two model-mismatch diagnostics."""
    if case not in experiment.truth_potentials:
        raise ValueError(
            f"case must be one of {tuple(experiment.truth_potentials)}, got {case!r}"
        )
    potential = experiment.truth_potentials[case]
    measured = _simulate_in_batches(experiment, potential, batch_size=batch_size)
    masked = jnp.where(experiment.reconstruction_mask, potential, 0.0)
    masked_intensities = _simulate_in_batches(experiment, masked, batch_size=batch_size)
    cutoff_intensities = _simulate_in_batches(
        experiment, experiment.cutoff_check_potential, batch_size=batch_size
    )
    vacancy_intensities = (
        measured
        if case == "vacancy"
        else _simulate_in_batches(
            experiment,
            experiment.truth_potentials["vacancy"],
            batch_size=batch_size,
        )
    )
    zero_exterior_nrmse = _amplitude_nrmse(masked_intensities, measured)
    cutoff_nrmse = _amplitude_nrmse(cutoff_intensities, vacancy_intensities)
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
            "validation_indices": np.asarray(experiment.validation_indices).tolist(),
        },
    )
    return GlancingDataset1D(
        case=case,
        potential=potential,
        scan=scan,
        truth_displacement_controls=experiment.truth_displacement_controls[case],
        zero_exterior_amplitude_nrmse=zero_exterior_nrmse,
        template_cutoff_amplitude_nrmse=cutoff_nrmse,
    )


def reconstruct_experiment_1d(
    experiment: SiliconGlancingExperiment1D,
    dataset: GlancingDataset1D,
    *,
    methods: Sequence[str] = ("lattice_sites",),
    options: ReconstructionOptions1D | None = None,
) -> OrderedDict[str, PotentialReconstruction1D | LatticeSiteReconstruction1D]:
    """Run selected baselines using concise, named method identifiers."""
    options = ReconstructionOptions1D() if options is None else options
    allowed = {"blind_pixels", "warm_pixels", "lattice_sites"}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"unknown reconstruction methods: {sorted(unknown)}")

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
        axial_coordinates=experiment.axial_coordinates,
        transverse_coordinates=experiment.transverse_coordinates,
        scan_coordinates=experiment.scan_coordinates,
        detector_angles=experiment.detector_angles,
        validation_indices=np.asarray(experiment.validation_indices),
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
        results["blind pixels"] = reconstruct_potential_1d(
            initial_potential=initial,
            progress=options.progress,
            progress_description="blind pixel reconstruction",
            **common,
        )
    if "warm_pixels" in methods:
        results["pristine-initialized pixels"] = reconstruct_potential_1d(
            initial_potential=experiment.pristine_potential,
            fixed_potential=experiment.pristine_potential,
            progress=options.progress,
            progress_description="warm pixel reconstruction",
            **common,
        )
    if "lattice_sites" in methods:
        results["lattice sites"] = reconstruct_lattice_site_potential_1d(
            experiment.lattice_model,
            experiment.input_probes,
            experiment.window_starts,
            experiment.window_length,
            experiment.propagation_kernel,
            experiment.axial_sampling,
            experiment.config.energy_eV,
            dataset.intensities,
            scan_coordinates=experiment.scan_coordinates,
            detector_angles=experiment.detector_angles,
            validation_indices=np.asarray(experiment.validation_indices),
            potential_max=potential_max,
            updates=options.lattice_updates,
            minibatch_size=options.minibatch_size,
            validation_interval=options.validation_interval_lattice,
            evaluation_batch_size=options.evaluation_batch_size,
            rematerialize=options.rematerialize,
            seed=options.seed,
            progress=options.progress,
            progress_description="lattice-site reconstruction",
        )
    return results


def reconstruction_metrics_1d(
    experiment: SiliconGlancingExperiment1D,
    dataset: GlancingDataset1D,
    results: Mapping[str, PotentialReconstruction1D | LatticeSiteReconstruction1D],
) -> dict[str, dict[str, float | int]]:
    """Return comparable loss, timing, potential, vacancy, and strain metrics."""
    truth = np.asarray(dataset.potential)
    mask = np.asarray(experiment.reconstruction_mask)
    metrics = {}
    for name, result in results.items():
        recovered = np.asarray(result.potential)
        row: dict[str, float | int] = {
            "best update": int(result.best_update),
            "best validation loss": float(result.metadata["best_metric"]),
            "potential NRMSE": float(
                np.linalg.norm(recovered[mask] - truth[mask])
                / np.linalg.norm(truth[mask])
            ),
        }
        history_updates = np.asarray(result.update_history)
        matches = np.flatnonzero(history_updates == result.best_update)
        if matches.size:
            row["time to best (s)"] = float(
                np.asarray(result.elapsed_time_history)[matches[0]]
            )
        if isinstance(result, LatticeSiteReconstruction1D):
            predicted = np.asarray(result.vacancy_fractions) >= 0.5
            actual = np.asarray(experiment.truth_vacancy_fractions) >= 0.5
            tp = np.count_nonzero(predicted & actual)
            fp = np.count_nonzero(predicted & ~actual)
            fn = np.count_nonzero(~predicted & actual)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            row["vacancy F1"] = float(
                2.0 * precision * recall / max(precision + recall, 1e-12)
            )
            true_displacements = np.asarray(
                lattice_site_displacements_1d(
                    experiment.variable_sites,
                    dataset.truth_displacement_controls,
                    experiment.lattice_model.control_coordinates_s,
                    experiment.lattice_model.control_coordinates_u,
                )
            )
            recovered_displacements = np.asarray(
                result.displaced_site_coordinates - result.site_coordinates
            )
            row["displacement RMSE (A)"] = float(
                np.sqrt(np.mean((recovered_displacements - true_displacements) ** 2))
            )
            row["specimen parameters"] = int(result.metadata["n_specimen_parameters"])
        else:
            row["specimen parameters"] = int(result.metadata["n_unknown_pixels"])
        metrics[name] = row
    return metrics


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
    s_A = np.asarray(experiment.axial_coordinates)
    u_A = np.asarray(experiment.transverse_coordinates)
    extent = [s_A[0], s_A[-1], u_A[0], u_A[-1]]
    fig, axes = plt.subplots(
        1,
        len(results) + 1,
        figsize=(4.5 * (len(results) + 1), 3.8),
        constrained_layout=True,
    )
    images = [("ground truth", dataset.potential), *results.items()]
    vmax = np.percentile(
        np.asarray(dataset.potential)[np.asarray(experiment.reconstruction_mask)],
        99.5,
    )
    for axis, (name, value) in zip(axes, images):
        potential = value if name == "ground truth" else value.potential
        image = axis.imshow(
            np.asarray(potential).T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        axis.set(title=name, xlabel="s (A)", ylabel="u (A)")
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

    sites = np.asarray(experiment.variable_sites)
    vacancies = np.asarray(result.vacancy_fractions)
    displacements = np.asarray(
        result.displaced_site_coordinates - result.site_coordinates
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    axes[0].scatter(sites[:, 0], vacancies, s=10, c=vacancies, cmap="magma")
    axes[0].set(xlabel="site s (A)", ylabel="vacancy fraction")
    axes[1].quiver(
        sites[:, 0],
        sites[:, 1],
        displacements[:, 0],
        displacements[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    axes[1].set(xlabel="site s (A)", ylabel="site u (A)", title="Displacement field")

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
        axis.set(title=title, xlabel="s (A)", ylabel="u (A)")
        strain_fig.colorbar(image, ax=axis, label="strain")
    return fig, strain_fig


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
