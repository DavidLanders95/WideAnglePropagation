"""Shared compact fixtures for focused atomistic-edit unit tests.

These builders deliberately create tiny numerical contracts, not benchmark
specimens.  Test modules retain their own case-specific constants and truth;
the helpers only remove repeated finite-host, kernel, support, and calibrated
count boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from wide_angle_propagation.propagation_methods import (
    fresnel_propagation_kernel_1d,
)
from wide_angle_propagation import ptychography_1d as ptychography
from wide_angle_propagation import ptychography_atomistic_edit_1d as atomistic
from wide_angle_propagation import ptychography_atomistic_edit_solver_1d as solver
from wide_angle_propagation.ptychography_support_contract_1d import (
    classify_lattice_site_support_1d,
)


@dataclass(frozen=True)
class CompactAtomisticEditModelSpec1D:
    """Small finite-host inputs that genuinely vary across focused tests."""

    shape: tuple[int, int]
    host_centres: Sequence[Sequence[int]]
    target_discovery_centres: Sequence[Sequence[int]]
    nuisance_discovery_centres: Sequence[Sequence[int]]
    edit_penalty_path: tuple[float, ...]
    max_host_removals: int
    max_extra_centres: int
    deformation_parameter_count: int
    fixture_id: str
    reference_background: float
    maximum_displacement_A: float


def _mark_neighbourhood(
    mask: np.ndarray,
    centre: Sequence[int],
    radius: int,
) -> None:
    row, column = (int(value) for value in centre)
    mask[
        max(0, row - radius) : min(mask.shape[0], row + radius + 1),
        max(0, column - radius) : min(mask.shape[1], column + radius + 1),
    ] = True


def make_compact_atomistic_edit_model_1d(
    spec: CompactAtomisticEditModelSpec1D,
):
    """Build the common positive kernel, finite host, and discovery contract."""

    raw_kernel = np.zeros((5, 5), dtype=np.float64)
    raw_kernel[1:4, 1:4] = np.asarray(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
    )
    kernel = atomistic.make_atomistic_edit_kernel_1d(
        raw_kernel,
        axial_sampling_A=1.0,
        transverse_sampling_A=1.0,
        host_equivalent_integrated_scattering=5.0,
        centre_index=(2.0, 2.0),
        parameterization_id=f"{spec.fixture_id}-host-equivalent:v1",
        cutoff_A=2.0,
        projection_width_A=5.0,
    )
    centres = np.asarray(spec.host_centres, dtype=np.int32)
    if centres.ndim != 2 or centres.shape[1:] != (2,) or not len(centres):
        raise ValueError("host_centres must have non-empty shape (n_host, 2)")
    starts = centres - 2
    patch = 5.0 * np.asarray(kernel.unit_integrated_values)
    patches = np.broadcast_to(patch, (len(centres), *patch.shape)).copy()
    reference = np.full(spec.shape, spec.reference_background, dtype=np.float64)
    for start, site_patch in zip(starts, patches, strict=True):
        row, column = (int(value) for value in start)
        reference[row : row + 5, column : column + 5] += site_patch

    target_host_indices = np.asarray([0], dtype=np.int64)
    target_pixels = np.zeros(spec.shape, dtype=bool)
    target_pixels[
        centres[target_host_indices, 0], centres[target_host_indices, 1]
    ] = True
    support = classify_lattice_site_support_1d(
        centres.astype(np.float64),
        centres,
        starts,
        np.full((len(centres), 2), 5, dtype=np.int32),
        target_pixels,
        np.ones(spec.shape, dtype=bool),
        excluded_probe_power=1e-6,
        atomic_template_cutoff_A=2.0,
        maximum_displacement_A=spec.maximum_displacement_A,
        displacement_control_shape=(2, 2, 2),
        maximum_nuisance_sites=max(4, len(centres)),
        maximum_specimen_parameters=128,
        strict=True,
    )
    axes = tuple(np.arange(length, dtype=np.float64) for length in spec.shape)
    host = ptychography.LatticeSiteModel1D(
        reference_potential=jnp.asarray(reference),
        site_coordinates=jnp.asarray(centres, dtype=jnp.float64),
        site_patches=jnp.asarray(patches),
        patch_starts=jnp.asarray(starts),
        control_coordinates_s=jnp.asarray([axes[0][0], axes[0][-1]]),
        control_coordinates_u=jnp.asarray([axes[1][0], axes[1][-1]]),
        axial_sampling=1.0,
        transverse_sampling=1.0,
        maximum_displacement=spec.maximum_displacement_A,
        metadata={"fixture": spec.fixture_id, "species": "Si"},
        support_contract=support,
    )

    target = np.zeros(spec.shape, dtype=bool)
    nuisance = np.zeros(spec.shape, dtype=bool)
    for centre in spec.target_discovery_centres:
        _mark_neighbourhood(target, centre, 1)
    for centre in centres[target_host_indices]:
        _mark_neighbourhood(target, centre, 1)
    nuisance_host_indices = np.setdiff1d(
        np.arange(len(centres)), target_host_indices
    )
    for centre in spec.nuisance_discovery_centres:
        _mark_neighbourhood(nuisance, centre, 1)
    for centre in centres[nuisance_host_indices]:
        _mark_neighbourhood(nuisance, centre, 1)
    rows, columns = np.indices(spec.shape)
    full_kernel_support = (
        (rows >= 2)
        & (rows < spec.shape[0] - 2)
        & (columns >= 2)
        & (columns < spec.shape[1] - 2)
    )
    target &= full_kernel_support
    nuisance &= full_kernel_support
    if np.any(target & nuisance):
        raise ValueError("compact TARGET and NUISANCE discovery regions overlap")
    discovery = atomistic.make_atomistic_edit_discovery_support_1d(
        axes[0],
        axes[1],
        target,
        nuisance,
        surface_envelope_A=(float(axes[1][0]), float(axes[1][-1])),
        geometry_source_id=f"{spec.fixture_id}-object-free-geometry:v1",
        excluded_probe_power=1e-6,
        metadata={"source": "shared compact test geometry"},
    )
    return atomistic.make_atomistic_edit_model_1d(
        host,
        axes[0],
        axes[1],
        kernel,
        atomistic.AtomisticEditOptions1D(
            max_host_removals=spec.max_host_removals,
            max_extra_centres=spec.max_extra_centres,
            max_scattering_equivalent_per_centre=2.0,
            minimum_separation_A=2.0,
            expected_rms_host_strain=0.1,
            edit_penalty_path=spec.edit_penalty_path,
            discovery_support=discovery,
        ),
        deformation_parameter_count=spec.deformation_parameter_count,
        metadata={"fixture": spec.fixture_id},
    )


@dataclass(frozen=True)
class CompactPreparedProblemSpec1D:
    """Common tiny calibrated-count forward problem configuration."""

    window_starts: tuple[int, ...]
    window_length: int
    probe_shifts: tuple[int, ...]
    validation_indices: tuple[int, ...]
    audit_indices: tuple[int, ...]
    electrons_per_pattern: float
    fixture_id: str
    objective_kind: str = "poisson_deviance"
    audit_count_scale: float = 1.0


def make_compact_prepared_atomistic_edit_problem_1d(
    model,
    spec: CompactPreparedProblemSpec1D,
    *,
    truth_state: atomistic.AtomisticEditState1D | None = None,
) -> solver.PreparedAtomisticEditReconstruction1D:
    """Simulate and prepare the shared compact calibrated-count acquisition."""

    detector_length = int(np.asarray(model.host_model.reference_potential).shape[1])
    transverse = jnp.arange(detector_length, dtype=jnp.float64)
    transverse = transverse - 0.5 * (detector_length - 1)
    base = jnp.exp(-0.5 * (transverse / 2.0) ** 2) * jnp.exp(
        0.13j * transverse
    )
    probes = jnp.stack([jnp.roll(base, shift) for shift in spec.probe_shifts])
    starts = jnp.asarray(spec.window_starts, dtype=jnp.int32)
    if probes.shape[0] != starts.size:
        raise ValueError("probe_shifts and window_starts must have equal length")
    propagation = fresnel_propagation_kernel_1d(
        detector_length, 1.0, 1.0, 30_000.0
    )
    state = (
        atomistic.empty_atomistic_edit_state_1d(model)
        if truth_state is None
        else truth_state
    )
    intensities = ptychography.simulate_glancing_scan_1d(
        atomistic.render_atomistic_edit_potential_1d(model, state),
        probes,
        starts,
        spec.window_length,
        propagation,
        1.0,
        30_000.0,
        rematerialize=False,
    )
    objective = ptychography.PtychographyObjective1D(
        kind=spec.objective_kind,
        electrons_per_pattern=spec.electrons_per_pattern,
        minimum_expected_electrons=1e-9,
    )
    signal = ptychography.ptychography_expected_signal_electrons_1d(
        intensities,
        probes,
        replace(objective, kind="poisson_deviance"),
    )
    if spec.audit_count_scale != 1.0:
        signal = signal.at[jnp.asarray(spec.audit_indices)].multiply(
            spec.audit_count_scale
        )
    measurement = ptychography.PtychographyMeasurement1D(
        calibrated_signal_electrons=signal,
        observed_total_electrons=signal,
        valid_mask=jnp.ones_like(signal, dtype=bool),
        calibrated_dark_electrons_per_pixel=jnp.zeros_like(signal),
        calibrated_read_noise_std_electrons=jnp.zeros_like(signal),
        calibration_id=f"{spec.fixture_id}-calibrated-counts:v1",
        metadata={"fixture": spec.fixture_id},
    )
    return solver.prepare_atomistic_edit_reconstruction_1d(
        model,
        probes,
        starts,
        spec.window_length,
        propagation,
        1.0,
        30_000.0,
        measurement,
        objective,
        validation_indices=spec.validation_indices,
        audit_indices=spec.audit_indices,
    )
