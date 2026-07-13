"""Crystalline host plus sparse substitution/adatom ptychography in 1D.

The diffraction forward model remains two-dimensional in ``(s, u)``.  Elastic
regularization is evaluated on latent three-dimensional host coordinates
``(s, y, u)`` so diamond-Si bond lengths and tetrahedral angles are retained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import operator
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from .propagation_methods import energy2wavelength
from .ptychography_1d import normalized_amplitude_loss_1d, simulate_glancing_scan_1d
from .ptychography_atoms_1d import (
    FreeAtomModel1D,
    render_species_mixture_atoms_1d,
)


__all__ = [
    "CrystallineDefectModel1D",
    "CrystallineDefectReconstruction1D",
    "CrystallineHostModel1D",
    "CrystallineHostReconstruction1D",
    "build_diamond_neighbor_graph_1d",
    "keating_lattice_energy_1d",
    "load_crystalline_defect_reconstruction_1d",
    "make_crystalline_defect_model_1d",
    "make_crystalline_host_model_1d",
    "reconstruct_crystalline_defects_1d",
    "reconstruct_crystalline_host_1d",
    "render_crystalline_defects_1d",
    "render_crystalline_host_1d",
    "save_crystalline_defect_reconstruction_1d",
    "transform_crystalline_host_1d",
]


Array = Any


@dataclass(frozen=True)
class CrystallineDefectModel1D:
    """A latent 3D crystalline host and a separate projected adatom pool.

    Host coordinates use column order ``(s, y, u)``.  Species templates have
    shape ``(n_species, template_s, template_u)``.  Bonds contain pairs and
    angles contain ``(center, neighbor_1, neighbor_2)`` triples.
    """

    axial_coordinates: Array
    transverse_coordinates: Array
    species_templates: Array
    species_names: tuple[str, ...]
    host_reference_positions_3d: Array
    host_bonds: Array
    host_angles: Array
    host_bounds: Array
    adatom_initial_positions: Array
    adatom_bounds: Array
    adatom_host_pairs: Array
    adatom_pairs: Array
    species_bond_lengths_A: Array
    host_update_weights: Array | None = None
    defect_core_bounds: Array | None = None
    fixed_potential: Array | None = None
    host_maximum_displacement_A: float = 4.0
    adatom_maximum_displacement_A: float = 3.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrystallineDefectReconstruction1D:
    """Best hybrid defect estimate and checkpoint histories."""

    host_positions_3d: Array
    host_occupancies: Array
    host_species_probabilities: Array
    adatom_positions: Array
    adatom_occupancies: Array
    adatom_species_probabilities: Array
    translation: Array
    strain: Array
    rotation_rad: Array
    potential: Array
    predicted_intensities: Array
    measured_intensities: Array
    update_history: Array
    elapsed_time_history: Array
    training_loss_history: Array
    validation_loss_history: Array
    translation_history: Array
    strain_history: Array
    rotation_history: Array
    host_displacement_history: Array
    host_occupancy_history: Array
    host_species_probability_history: Array
    adatom_position_history: Array
    adatom_occupancy_history: Array
    adatom_species_probability_history: Array
    best_update: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


# Pristine-host reconstruction uses the same storage layout while fixing all
# occupancy, substitution, and off-lattice variables.  Public aliases keep the
# notebook vocabulary focused on the simpler physical model.
CrystallineHostModel1D = CrystallineDefectModel1D
CrystallineHostReconstruction1D = CrystallineDefectReconstruction1D


def _array(name: str, value: Any, ndim: int) -> Array:
    result = jnp.asarray(value)
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {result.shape}")
    return result


def _host_projection(reference_positions_3d: Array) -> Array:
    return jnp.stack(
        [reference_positions_3d[:, 0], reference_positions_3d[:, 2]], axis=-1
    )


def build_diamond_neighbor_graph_1d(
    reference_positions_3d: Any,
    *,
    bond_cutoff_A: float = 2.65,
) -> tuple[Array, Array]:
    """Build sparse bond pairs and tetrahedral angle triples.

    The graph construction is host-side and uses a KD tree.  It therefore
    scales with the number of physical neighbors rather than forming an
    ``n_atom`` squared distance matrix.
    """
    positions = np.asarray(reference_positions_3d, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("reference_positions_3d must have shape (n_host, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("reference_positions_3d must be finite")
    if not np.isfinite(bond_cutoff_A) or bond_cutoff_A <= 0.0:
        raise ValueError("bond_cutoff_A must be positive")
    pairs = np.asarray(
        sorted(cKDTree(positions).query_pairs(float(bond_cutoff_A))), dtype=np.int32
    )
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int32)
    neighbors: list[list[int]] = [[] for _ in range(len(positions))]
    for first, second in pairs:
        neighbors[int(first)].append(int(second))
        neighbors[int(second)].append(int(first))
    angles = []
    for center, local in enumerate(neighbors):
        for first_index in range(len(local)):
            for second_index in range(first_index + 1, len(local)):
                angles.append((center, local[first_index], local[second_index]))
    angle_array = np.asarray(angles, dtype=np.int32)
    if angle_array.size == 0:
        angle_array = np.empty((0, 3), dtype=np.int32)
    return jnp.asarray(pairs), jnp.asarray(angle_array)


def _possible_cross_pairs(
    first: np.ndarray,
    second: np.ndarray,
    radius: float,
) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.empty((0, 2), dtype=np.int32)
    tree = cKDTree(second)
    pairs = [
        (first_index, second_index)
        for first_index, point in enumerate(first)
        for second_index in tree.query_ball_point(point, radius)
    ]
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


def _possible_self_pairs(positions: np.ndarray, radius: float) -> np.ndarray:
    if len(positions) < 2:
        return np.empty((0, 2), dtype=np.int32)
    pairs = sorted(cKDTree(positions).query_pairs(radius))
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


def make_crystalline_defect_model_1d(
    axial_coordinates: Any,
    transverse_coordinates: Any,
    species_templates: Any,
    species_names: Sequence[str],
    host_reference_positions_3d: Any,
    host_bounds: Any,
    adatom_initial_positions: Any,
    adatom_bounds: Any,
    *,
    species_bond_lengths_A: Any | None = None,
    host_update_weights: Any | None = None,
    defect_core_bounds: Any | None = None,
    fixed_potential: Any | None = None,
    bond_cutoff_A: float = 2.65,
    host_maximum_displacement_A: float = 4.0,
    adatom_maximum_displacement_A: float = 3.0,
    repulsion_neighbor_radius_A: float = 7.0,
    metadata: Mapping[str, Any] | None = None,
) -> CrystallineDefectModel1D:
    """Construct a validated model and all static sparse neighbor lists."""
    coordinates_s = np.asarray(axial_coordinates, dtype=float)
    coordinates_u = np.asarray(transverse_coordinates, dtype=float)
    templates = np.asarray(species_templates)
    names = tuple(str(name) for name in species_names)
    host = np.asarray(host_reference_positions_3d, dtype=float)
    host_bounds_array = np.asarray(host_bounds, dtype=float)
    adatoms = np.asarray(adatom_initial_positions, dtype=float)
    adatom_bounds_array = np.asarray(adatom_bounds, dtype=float)
    if coordinates_s.ndim != 1 or coordinates_u.ndim != 1:
        raise ValueError("specimen coordinates must be one-dimensional")
    if templates.ndim != 3 or templates.shape[0] != len(names) or len(names) < 1:
        raise ValueError("species templates and names do not match")
    if len(set(names)) != len(names):
        raise ValueError("species names must be unique")
    if host.ndim != 2 or host.shape[1] != 3 or host.shape[0] == 0:
        raise ValueError("host_reference_positions_3d must have shape (n, 3)")
    if adatoms.ndim != 2 or adatoms.shape[1] != 2:
        raise ValueError("adatom_initial_positions must have shape (n, 2)")
    if host_bounds_array.shape != (2, 2) or adatom_bounds_array.shape != (2, 2):
        raise ValueError("host and adatom bounds must have shape (2, 2)")
    if np.any(host_bounds_array[:, 1] <= host_bounds_array[:, 0]) or np.any(
        adatom_bounds_array[:, 1] <= adatom_bounds_array[:, 0]
    ):
        raise ValueError("candidate bounds must have positive width")
    projected_host = host[:, [0, 2]]
    if host_update_weights is None:
        update_weights = np.ones(host.shape[0], dtype=float)
    else:
        update_weights = np.asarray(host_update_weights, dtype=float)
    if update_weights.shape != (host.shape[0],):
        raise ValueError("host_update_weights must have one value per host site")
    if not np.all(np.isfinite(update_weights)) or np.any(
        (update_weights < 0.0) | (update_weights > 1.0)
    ):
        raise ValueError("host_update_weights must be finite and lie in [0, 1]")
    if np.any(projected_host < host_bounds_array[:, 0]) or np.any(
        projected_host > host_bounds_array[:, 1]
    ):
        raise ValueError("projected host sites must lie inside host_bounds")
    if len(adatoms) and (
        np.any(adatoms < adatom_bounds_array[:, 0])
        or np.any(adatoms > adatom_bounds_array[:, 1])
    ):
        raise ValueError("adatom seeds must lie inside adatom_bounds")
    for name, value in (
        ("host_maximum_displacement_A", host_maximum_displacement_A),
        ("adatom_maximum_displacement_A", adatom_maximum_displacement_A),
        ("repulsion_neighbor_radius_A", repulsion_neighbor_radius_A),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if species_bond_lengths_A is None:
        bond_lengths = np.full((len(names), len(names)), 2.3517, dtype=float)
    else:
        bond_lengths = np.asarray(species_bond_lengths_A, dtype=float)
    if bond_lengths.shape != (len(names), len(names)) or not np.allclose(
        bond_lengths, bond_lengths.T
    ):
        raise ValueError("species_bond_lengths_A must be a symmetric species matrix")
    if defect_core_bounds is None:
        core_bounds = host_bounds_array.copy()
        core_bounds[1, 1] = max(core_bounds[1, 1], adatom_bounds_array[1, 1])
    else:
        core_bounds = np.asarray(defect_core_bounds, dtype=float)
    if core_bounds.shape != (2, 2) or np.any(core_bounds[:, 1] <= core_bounds[:, 0]):
        raise ValueError("defect_core_bounds must have shape (2, 2) and positive width")
    bonds, angles = build_diamond_neighbor_graph_1d(
        host, bond_cutoff_A=bond_cutoff_A
    )
    possible_radius = (
        repulsion_neighbor_radius_A
        + host_maximum_displacement_A
        + adatom_maximum_displacement_A
    )
    cross = _possible_cross_pairs(adatoms, projected_host, possible_radius)
    adatom_pairs = _possible_self_pairs(
        adatoms, repulsion_neighbor_radius_A + 2 * adatom_maximum_displacement_A
    )
    return CrystallineDefectModel1D(
        axial_coordinates=jnp.asarray(coordinates_s),
        transverse_coordinates=jnp.asarray(coordinates_u),
        species_templates=jnp.asarray(templates),
        species_names=names,
        host_reference_positions_3d=jnp.asarray(host),
        host_bonds=bonds,
        host_angles=angles,
        host_bounds=jnp.asarray(host_bounds_array),
        adatom_initial_positions=jnp.asarray(adatoms),
        adatom_bounds=jnp.asarray(adatom_bounds_array),
        adatom_host_pairs=jnp.asarray(cross),
        adatom_pairs=jnp.asarray(adatom_pairs),
        species_bond_lengths_A=jnp.asarray(bond_lengths),
        host_update_weights=jnp.asarray(update_weights),
        defect_core_bounds=jnp.asarray(core_bounds),
        fixed_potential=None if fixed_potential is None else jnp.asarray(fixed_potential),
        host_maximum_displacement_A=float(host_maximum_displacement_A),
        adatom_maximum_displacement_A=float(adatom_maximum_displacement_A),
        metadata=dict(metadata or {}),
    )


def make_crystalline_host_model_1d(
    axial_coordinates: Any,
    transverse_coordinates: Any,
    atom_template: Any,
    host_reference_positions_3d: Any,
    host_bounds: Any,
    *,
    species_name: str = "Si",
    equilibrium_bond_length_A: float = 2.3517,
    host_update_weights: Any | None = None,
    fixed_potential: Any | None = None,
    bond_cutoff_A: float = 2.65,
    host_maximum_displacement_A: float = 4.0,
    metadata: Mapping[str, Any] | None = None,
) -> CrystallineHostModel1D:
    """Build a single-species crystalline host with no defect variables.

    The returned model still carries empty adatom arrays so it remains
    persistence-compatible with the more general defect reconstruction.
    """
    coordinates_s = np.asarray(axial_coordinates, dtype=float)
    coordinates_u = np.asarray(transverse_coordinates, dtype=float)
    template = np.asarray(atom_template)
    if template.ndim != 2:
        raise ValueError("atom_template must be two-dimensional")
    if not isinstance(species_name, str) or not species_name.strip():
        raise ValueError("species_name must be a non-empty string")
    if not np.isfinite(equilibrium_bond_length_A) or equilibrium_bond_length_A <= 0.0:
        raise ValueError("equilibrium_bond_length_A must be positive")
    if coordinates_s.ndim != 1 or coordinates_u.ndim != 1:
        raise ValueError("specimen coordinates must be one-dimensional")
    specimen_bounds = np.asarray(
        [
            [coordinates_s[0], coordinates_s[-1]],
            [coordinates_u[0], coordinates_u[-1]],
        ],
        dtype=float,
    )
    return make_crystalline_defect_model_1d(
        coordinates_s,
        coordinates_u,
        template[None, ...],
        (species_name.strip(),),
        host_reference_positions_3d,
        host_bounds,
        np.empty((0, 2), dtype=float),
        specimen_bounds,
        species_bond_lengths_A=np.asarray([[equilibrium_bond_length_A]]),
        host_update_weights=host_update_weights,
        defect_core_bounds=host_bounds,
        fixed_potential=fixed_potential,
        bond_cutoff_A=bond_cutoff_A,
        host_maximum_displacement_A=host_maximum_displacement_A,
        metadata=metadata,
    )


def transform_crystalline_host_1d(
    reference_positions_3d: Any,
    translation: Any,
    strain: Any,
    rotation_rad: Any,
    local_displacements: Any,
    update_weights: Any | None = None,
) -> Array:
    """Apply an in-plane transform only where site update weights permit it.

    A weight of zero leaves the reference site exactly fixed, while a weight
    of one applies the full affine and local transform.  Intermediate values
    provide a smooth boundary around an illuminated reconstruction volume.
    """
    reference = _array("reference_positions_3d", reference_positions_3d, 2)
    translation_array = _array("translation", translation, 1)
    strain_array = _array("strain", strain, 2)
    displacements = _array("local_displacements", local_displacements, 2)
    if reference.shape[1] != 3 or translation_array.shape != (2,):
        raise ValueError("reference and translation shapes are invalid")
    if strain_array.shape != (2, 2) or displacements.shape != (reference.shape[0], 2):
        raise ValueError("strain or displacement shape is invalid")
    weights = (
        jnp.ones((reference.shape[0],), dtype=reference.dtype)
        if update_weights is None
        else _array("update_weights", update_weights, 1)
    )
    if weights.shape != (reference.shape[0],):
        raise ValueError("update_weights must have one value per host site")
    projected = _host_projection(reference)
    center = jnp.mean(projected, axis=0)
    cosine = jnp.cos(rotation_rad)
    sine = jnp.sin(rotation_rad)
    rotation = jnp.asarray([[cosine, -sine], [sine, cosine]])
    deformation = rotation @ (jnp.eye(2, dtype=projected.dtype) + strain_array)
    transformed = (projected - center) @ deformation.T + center
    candidate = transformed + translation_array + displacements
    transformed = projected + weights[:, None] * (candidate - projected)
    return reference.at[:, 0].set(transformed[:, 0]).at[:, 2].set(transformed[:, 1])


def keating_lattice_energy_1d(
    positions_3d: Any,
    occupancies: Any,
    species_probabilities: Any,
    bonds: Any,
    angles: Any,
    species_bond_lengths_A: Any,
    *,
    stretch_weight: float = 1.0,
    bend_weight: float = 1.0,
) -> Array:
    """Return occupancy-gated sparse bond-stretch and tetrahedral-angle energy."""
    positions = _array("positions_3d", positions_3d, 2)
    occupancy = _array("occupancies", occupancies, 1)
    probabilities = _array("species_probabilities", species_probabilities, 2)
    bond_indices = _array("bonds", bonds, 2).astype(jnp.int32)
    angle_indices = _array("angles", angles, 2).astype(jnp.int32)
    lengths = _array("species_bond_lengths_A", species_bond_lengths_A, 2)
    if positions.shape != (occupancy.shape[0], 3):
        raise ValueError("position and occupancy shapes do not match")
    if probabilities.shape[0] != positions.shape[0]:
        raise ValueError("species probabilities must have one row per host site")
    if lengths.shape != (probabilities.shape[1], probabilities.shape[1]):
        raise ValueError("bond-length matrix does not match the species count")
    if bond_indices.shape[1:] != (2,) or angle_indices.shape[1:] != (3,):
        raise ValueError("bonds and angles must have shapes (n,2) and (m,3)")
    if bond_indices.shape[0]:
        first, second = bond_indices[:, 0], bond_indices[:, 1]
        vectors = positions[second] - positions[first]
        squared_distance = jnp.sum(vectors**2, axis=-1)
        pair_probabilities = (
            probabilities[first, :, None] * probabilities[second, None, :]
        )
        target = jnp.sum(pair_probabilities * lengths[None, :, :], axis=(1, 2))
        bond_gate = occupancy[first] * occupancy[second]
        stretch = jnp.sum(
            bond_gate * ((squared_distance - target**2) / target**2) ** 2
        ) / jnp.maximum(jnp.sum(bond_gate), 1.0)
    else:
        stretch = jnp.asarray(0.0, dtype=positions.dtype)
    if angle_indices.shape[0]:
        center, first, second = (
            angle_indices[:, 0],
            angle_indices[:, 1],
            angle_indices[:, 2],
        )
        vector_first = positions[first] - positions[center]
        vector_second = positions[second] - positions[center]
        cosine = jnp.sum(vector_first * vector_second, axis=-1) / jnp.maximum(
            jnp.linalg.norm(vector_first, axis=-1)
            * jnp.linalg.norm(vector_second, axis=-1),
            1e-8,
        )
        angle_gate = occupancy[center] * occupancy[first] * occupancy[second]
        bend = jnp.sum(angle_gate * (cosine + 1.0 / 3.0) ** 2) / jnp.maximum(
            jnp.sum(angle_gate), 1.0
        )
    else:
        bend = jnp.asarray(0.0, dtype=positions.dtype)
    return stretch_weight * stretch + bend_weight * bend


def _base_model(
    model: CrystallineDefectModel1D,
    initial_positions: Array,
    bounds: Array,
    maximum_displacement_A: float,
) -> FreeAtomModel1D:
    # A globally translated lattice may place a boundary atom just outside the
    # reconstructed grid while its finite template still overlaps the grid.
    # The local renderer only uses these positions as stencil anchors, so clip
    # the anchors to the grid and retain the physical positions passed to the
    # renderer itself.
    anchor_positions = jnp.clip(
        initial_positions,
        jnp.asarray(bounds)[:, 0],
        jnp.asarray(bounds)[:, 1],
    )
    return FreeAtomModel1D(
        model.axial_coordinates,
        model.transverse_coordinates,
        model.species_templates[0],
        bounds,
        anchor_positions,
        fixed_potential=None,
        maximum_displacement_A=maximum_displacement_A,
        metadata=model.metadata,
    )


def _render_bounds(model: CrystallineDefectModel1D) -> Array:
    return jnp.asarray(
        [
            [model.axial_coordinates[0], model.axial_coordinates[-1]],
            [model.transverse_coordinates[0], model.transverse_coordinates[-1]],
        ]
    )


def render_crystalline_defects_1d(
    model: CrystallineDefectModel1D,
    host_positions_3d: Any,
    host_occupancies: Any,
    host_species_probabilities: Any,
    adatom_positions: Any,
    adatom_occupancies: Any,
    adatom_species_probabilities: Any,
) -> Array:
    """Render host and off-lattice atoms as fixed-template species mixtures."""
    host_positions = _array("host_positions_3d", host_positions_3d, 2)
    projected_host = _host_projection(host_positions)
    host_model = _base_model(
        model,
        projected_host,
        _render_bounds(model),
        min(model.host_maximum_displacement_A, 0.5),
    )
    host_potential = render_species_mixture_atoms_1d(
        host_model,
        model.species_templates,
        projected_host,
        host_occupancies,
        host_species_probabilities,
    )
    adatom_positions_array = _array("adatom_positions", adatom_positions, 2)
    if adatom_positions_array.shape[0]:
        adatom_model = _base_model(
            model,
            adatom_positions_array,
            _render_bounds(model),
            min(model.adatom_maximum_displacement_A, 0.5),
        )
        adatom_potential = render_species_mixture_atoms_1d(
            adatom_model,
            model.species_templates,
            adatom_positions_array,
            adatom_occupancies,
            adatom_species_probabilities,
        )
    else:
        adatom_potential = jnp.zeros_like(host_potential)
    fixed = (
        jnp.zeros_like(host_potential)
        if model.fixed_potential is None
        else jnp.asarray(model.fixed_potential, dtype=host_potential.dtype)
    )
    return fixed + host_potential + adatom_potential


def render_crystalline_host_1d(
    model: CrystallineHostModel1D,
    host_positions_3d: Any,
) -> Array:
    """Render a fully occupied single-species crystalline host."""
    if len(model.species_names) != 1:
        raise ValueError("pristine host rendering requires exactly one species")
    positions = _array("host_positions_3d", host_positions_3d, 2)
    n_host = positions.shape[0]
    dtype = jnp.result_type(positions, model.species_templates)
    return render_crystalline_defects_1d(
        model,
        positions,
        jnp.ones((n_host,), dtype=dtype),
        jnp.ones((n_host, 1), dtype=dtype),
        jnp.empty((0, 2), dtype=dtype),
        jnp.empty((0,), dtype=dtype),
        jnp.empty((0, 1), dtype=dtype),
    )


def _sparse_repulsion(
    first_positions: Array,
    first_occupancies: Array,
    second_positions: Array,
    second_occupancies: Array,
    pairs: Array,
    minimum_distance_A: float,
) -> Array:
    if pairs.shape[0] == 0:
        return jnp.asarray(0.0, dtype=first_positions.dtype)
    first_indices, second_indices = pairs[:, 0], pairs[:, 1]
    distances = jnp.linalg.norm(
        first_positions[first_indices] - second_positions[second_indices], axis=-1
    )
    gate = first_occupancies[first_indices] * second_occupancies[second_indices]
    return jnp.sum(gate * jax.nn.relu(minimum_distance_A - distances) ** 2) / jnp.maximum(
        jnp.sum(gate), 1.0
    )


def reconstruct_crystalline_defects_1d(
    model: CrystallineDefectModel1D,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    measured_intensities: Any,
    *,
    validation_indices: Sequence[int] = (),
    updates: int = 1800,
    stage_global_end: int = 300,
    stage_host_end: int = 900,
    stage_defect_end: int = 1400,
    minibatch_size: int = 4,
    validation_interval: int = 25,
    evaluation_batch_size: int = 5,
    learning_rate_start: float = 1e-2,
    learning_rate_end: float = 5e-4,
    keating_weight: float = 5e-2,
    host_occupancy_weight: float = 2e-4,
    substitution_weight: float = 5e-5,
    adatom_weight: float = 5e-5,
    displacement_weight: float = 1e-3,
    binary_weight: float = 2e-4,
    entropy_weight: float = 1e-4,
    repulsion_weight: float = 5e-2,
    buffer_defect_multiplier: float = 10.0,
    initial_host_occupancy: float = 0.98,
    initial_adatom_occupancy: float = 0.01,
    initial_host_si_probability: float = 0.99,
    initial_adatom_si_probability: float = 0.9,
    translation_limit_A: float = 3.0,
    strain_limit: float = 0.03,
    rotation_limit_deg: float = 1.0,
    host_displacement_limit_A: float = 0.75,
    repulsion_minimum_distance_A: float = 1.8,
    enable_global_transform: bool = True,
    enable_host_displacements: bool = True,
    enable_host_occupancies: bool = True,
    enable_substitutions: bool = True,
    enable_adatoms: bool = True,
    seed: int = 0,
    progress: bool = False,
    progress_description: str = "crystalline defect reconstruction",
) -> CrystallineDefectReconstruction1D:
    """Jointly fit a deformable host, substitutions, and sparse adatoms."""
    try:
        import optax
    except ImportError as exc:  # pragma: no cover
        raise ImportError("reconstruct_crystalline_defects_1d requires Optax") from exc
    n_updates = operator.index(updates)
    global_end = operator.index(stage_global_end)
    host_end = operator.index(stage_host_end)
    defect_end = operator.index(stage_defect_end)
    if not (0 <= global_end <= host_end <= defect_end <= n_updates):
        raise ValueError("optimization stages must be ordered within updates")
    starts = _array("window_starts", window_starts, 1).astype(jnp.int32)
    measured = _array("measured_intensities", measured_intensities, 2)
    probe = jnp.asarray(input_probe)
    kernel = _array("propagation_kernel", propagation_kernel, 1)
    n_scan, n_u = measured.shape
    if starts.shape[0] != n_scan or probe.shape[-1] != n_u or kernel.shape[0] != n_u:
        raise ValueError("scan, probe, kernel, and measurements do not match")
    probe_rows = jnp.broadcast_to(probe, (n_scan, n_u)) if probe.ndim == 1 else probe
    if probe_rows.shape != (n_scan, n_u):
        raise ValueError("input_probe must be shared or contain one row per scan")
    validation = np.asarray(validation_indices, dtype=np.int32)
    if validation.ndim != 1 or np.unique(validation).size != validation.size or np.any(
        (validation < 0) | (validation >= n_scan)
    ):
        raise ValueError("validation_indices must be unique valid scan indices")
    training = np.setdiff1d(np.arange(n_scan, dtype=np.int32), validation)
    if training.size == 0:
        raise ValueError("at least one training scan is required")
    n_host = model.host_reference_positions_3d.shape[0]
    n_adatom = model.adatom_initial_positions.shape[0]
    n_species = len(model.species_names)
    if n_species < 1:
        raise ValueError("at least one host species is required")
    if enable_substitutions and n_species < 2:
        raise ValueError("substitution fitting requires at least two candidate species")
    enable_flags = (
        enable_global_transform,
        enable_host_displacements,
        enable_host_occupancies,
        enable_substitutions,
        enable_adatoms,
    )
    if not all(isinstance(value, (bool, np.bool_)) for value in enable_flags):
        raise TypeError("defect enable flags must be boolean")
    if not np.isfinite(buffer_defect_multiplier) or buffer_defect_multiplier < 1.0:
        raise ValueError("buffer_defect_multiplier must be finite and at least one")
    if n_species > 1:
        for name, probability in (
            ("initial_host_si_probability", initial_host_si_probability),
            ("initial_adatom_si_probability", initial_adatom_si_probability),
        ):
            if not np.isfinite(probability) or not 0.0 < probability < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
    dtype = jnp.result_type(model.host_reference_positions_3d, jnp.float32)
    host_update_weights = (
        jnp.ones((n_host,), dtype=dtype)
        if model.host_update_weights is None
        else jnp.asarray(model.host_update_weights, dtype=dtype)
    )
    if host_update_weights.shape != (n_host,):
        raise ValueError("model host_update_weights must have one value per host site")
    if n_species == 1:
        host_logits = jnp.zeros((n_host, 1), dtype=dtype)
        adatom_logits = jnp.zeros((n_adatom, 1), dtype=dtype)
    else:
        host_other = (1.0 - initial_host_si_probability) / (n_species - 1)
        adatom_other = (1.0 - initial_adatom_si_probability) / (n_species - 1)
        host_logits = jnp.full(
            (n_host, n_species), jnp.log(host_other), dtype=dtype
        ).at[:, 0].set(jnp.log(initial_host_si_probability))
        adatom_logits = jnp.full(
            (n_adatom, n_species), jnp.log(adatom_other), dtype=dtype
        ).at[:, 0].set(jnp.log(initial_adatom_si_probability))
    if not enable_substitutions:
        host_logits = jnp.full((n_host, n_species), -12.0, dtype=dtype).at[:, 0].set(12.0)
    resolved_initial_adatom_occupancy = (
        initial_adatom_occupancy if enable_adatoms else 0.0
    )
    resolved_initial_host_occupancy = (
        initial_host_occupancy if enable_host_occupancies else 1.0
    )
    parameters = {
        "translation": jnp.zeros(2, dtype=dtype),
        "strain": jnp.zeros((2, 2), dtype=dtype),
        "rotation": jnp.asarray(0.0, dtype=dtype),
        "host_displacements": jnp.zeros((n_host, 2), dtype=dtype),
        "host_occupancies": jnp.full(
            (n_host,), resolved_initial_host_occupancy, dtype=dtype
        ),
        "host_species_logits": host_logits,
        "adatom_positions": jnp.asarray(model.adatom_initial_positions, dtype=dtype),
        "adatom_occupancies": jnp.full(
            (n_adatom,), resolved_initial_adatom_occupancy, dtype=dtype
        ),
        "adatom_species_logits": adatom_logits,
    }

    def decode(values):
        host_positions = transform_crystalline_host_1d(
            model.host_reference_positions_3d,
            values["translation"],
            values["strain"],
            values["rotation"],
            values["host_displacements"],
            host_update_weights,
        )
        return (
            host_positions,
            jax.nn.softmax(values["host_species_logits"], axis=-1),
            jax.nn.softmax(values["adatom_species_logits"], axis=-1),
        )

    def objective(values, batch_indices, elastic_scale, discrete_scale):
        host_positions, host_species, adatom_species = decode(values)
        potential = render_crystalline_defects_1d(
            model,
            host_positions,
            values["host_occupancies"],
            host_species,
            values["adatom_positions"],
            values["adatom_occupancies"],
            adatom_species,
        )
        prediction = simulate_glancing_scan_1d(
            potential,
            probe_rows[batch_indices],
            starts[batch_indices],
            window_length,
            kernel,
            slice_thickness,
            energy,
        )
        data = jnp.sqrt(
            normalized_amplitude_loss_1d(prediction, measured[batch_indices]) + 1e-16
        )
        keating = keating_lattice_energy_1d(
            host_positions,
            values["host_occupancies"] * host_update_weights,
            host_species,
            model.host_bonds,
            model.host_angles,
            model.species_bond_lengths_A,
        )
        host_projection = _host_projection(host_positions)
        cross_repulsion = _sparse_repulsion(
            values["adatom_positions"],
            values["adatom_occupancies"],
            host_projection,
            values["host_occupancies"],
            model.adatom_host_pairs,
            repulsion_minimum_distance_A,
        )
        self_repulsion = _sparse_repulsion(
            values["adatom_positions"],
            values["adatom_occupancies"],
            values["adatom_positions"],
            values["adatom_occupancies"],
            model.adatom_pairs,
            repulsion_minimum_distance_A,
        )
        host_binary = jnp.mean(
            values["host_occupancies"] * (1.0 - values["host_occupancies"])
        )
        adatom_binary = (
            jnp.mean(values["adatom_occupancies"] * (1.0 - values["adatom_occupancies"]))
            if n_adatom
            else 0.0
        )
        host_entropy = -jnp.mean(jnp.sum(host_species * jnp.log(host_species + 1e-12), axis=-1))
        adatom_entropy = (
            -jnp.mean(jnp.sum(adatom_species * jnp.log(adatom_species + 1e-12), axis=-1))
            if n_adatom
            else 0.0
        )
        core_bounds = jnp.asarray(model.defect_core_bounds)
        host_in_core = (
            (host_projection[:, 0] >= core_bounds[0, 0])
            & (host_projection[:, 0] <= core_bounds[0, 1])
            & (host_projection[:, 1] >= core_bounds[1, 0])
            & (host_projection[:, 1] <= core_bounds[1, 1])
        )
        host_defect_weight = jnp.where(host_in_core, 1.0, buffer_defect_multiplier)
        adatom_in_core = (
            (values["adatom_positions"][:, 0] >= core_bounds[0, 0])
            & (values["adatom_positions"][:, 0] <= core_bounds[0, 1])
            & (values["adatom_positions"][:, 1] >= core_bounds[1, 0])
            & (values["adatom_positions"][:, 1] <= core_bounds[1, 1])
        )
        adatom_defect_weight = jnp.where(
            adatom_in_core, 1.0, buffer_defect_multiplier
        )
        return (
            data
            + elastic_scale * keating_weight * keating
            + host_occupancy_weight * jnp.mean((values["host_occupancies"] - 1.0) ** 2)
            + substitution_weight * jnp.sum(
                host_defect_weight * values["host_occupancies"] * (1.0 - host_species[:, 0])
            )
            + adatom_weight * (
                jnp.sum(adatom_defect_weight * values["adatom_occupancies"])
                if n_adatom else 0.0
            )
            + displacement_weight * jnp.sum(
                host_update_weights[:, None] * values["host_displacements"] ** 2
            ) / jnp.maximum(2.0 * jnp.sum(host_update_weights), 1.0)
            + discrete_scale * binary_weight * (host_binary + adatom_binary)
            + discrete_scale * entropy_weight * (host_entropy + adatom_entropy)
            + repulsion_weight * (cross_repulsion + self_repulsion)
        )

    schedule = optax.cosine_decay_schedule(
        learning_rate_start, n_updates, alpha=learning_rate_end / learning_rate_start
    )
    def scaled_schedule(factor):
        return lambda count: factor * schedule(count)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {
                "translation": optax.adam(scaled_schedule(1.0)),
                "affine": optax.adam(scaled_schedule(1e-3)),
                "host_position": optax.adam(scaled_schedule(0.2)),
                "occupancy": optax.adam(scaled_schedule(1.0)),
                "species": optax.adam(scaled_schedule(1.0)),
                "adatom_position": optax.adam(scaled_schedule(0.5)),
            },
            {
                "translation": "translation",
                "strain": "affine",
                "rotation": "affine",
                "host_displacements": "host_position",
                "host_occupancies": "occupancy",
                "host_species_logits": "species",
                "adatom_positions": "adatom_position",
                "adatom_occupancies": "occupancy",
                "adatom_species_logits": "species",
            },
        ),
    )
    state = optimizer.init(parameters)

    @jax.jit
    def update_step(values, optimizer_state, batch_indices, stage, elastic_scale, discrete_scale):
        loss, gradients = jax.value_and_grad(objective)(
            values, batch_indices, elastic_scale, discrete_scale
        )
        global_stage = (stage >= 0) & enable_global_transform
        host_stage = (stage >= 1) & enable_host_displacements
        occupancy_stage = enable_host_occupancies
        defect_stage = stage >= 2
        substitution_stage = defect_stage & enable_substitutions
        adatom_stage = defect_stage & enable_adatoms
        gradients = {
            "translation": jnp.where(global_stage, gradients["translation"], 0.0),
            "strain": jnp.where(global_stage, gradients["strain"], 0.0),
            "rotation": jnp.where(global_stage, gradients["rotation"], 0.0),
            "host_displacements": jnp.where(
                host_stage,
                gradients["host_displacements"] * host_update_weights[:, None],
                0.0,
            ),
            "host_occupancies": jnp.where(
                occupancy_stage,
                gradients["host_occupancies"] * host_update_weights,
                0.0,
            ),
            "host_species_logits": jnp.where(
                substitution_stage,
                gradients["host_species_logits"] * host_update_weights[:, None],
                0.0,
            ),
            "adatom_positions": jnp.where(adatom_stage, gradients["adatom_positions"], 0.0),
            "adatom_occupancies": jnp.where(adatom_stage, gradients["adatom_occupancies"], 0.0),
            "adatom_species_logits": jnp.where(adatom_stage, gradients["adatom_species_logits"], 0.0),
        }
        updates_value, optimizer_state = optimizer.update(gradients, optimizer_state, values)
        values = optax.apply_updates(values, updates_value)
        host_lower = jnp.asarray(model.host_bounds)[:, 0]
        host_upper = jnp.asarray(model.host_bounds)[:, 1]
        adatom_lower = jnp.maximum(
            jnp.asarray(model.adatom_bounds)[:, 0],
            model.adatom_initial_positions - model.adatom_maximum_displacement_A,
        )
        adatom_upper = jnp.minimum(
            jnp.asarray(model.adatom_bounds)[:, 1],
            model.adatom_initial_positions + model.adatom_maximum_displacement_A,
        )
        del host_lower, host_upper  # host positions are bounded through transform components.
        values = {
            **values,
            "translation": jnp.clip(values["translation"], -translation_limit_A, translation_limit_A),
            "strain": jnp.clip(values["strain"], -strain_limit, strain_limit),
            "rotation": jnp.clip(
                values["rotation"], -jnp.deg2rad(rotation_limit_deg), jnp.deg2rad(rotation_limit_deg)
            ),
            "host_displacements": jnp.clip(
                values["host_displacements"], -host_displacement_limit_A, host_displacement_limit_A
            ),
            "host_occupancies": jnp.clip(values["host_occupancies"], 0.0, 1.0),
            "adatom_positions": jnp.clip(values["adatom_positions"], adatom_lower, adatom_upper),
            "adatom_occupancies": jnp.clip(values["adatom_occupancies"], 0.0, 1.0),
        }
        return values, optimizer_state, loss

    @jax.jit
    def predict(values, indices):
        host_positions, host_species, adatom_species = decode(values)
        potential = render_crystalline_defects_1d(
            model,
            host_positions,
            values["host_occupancies"],
            host_species,
            values["adatom_positions"],
            values["adatom_occupancies"],
            adatom_species,
        )
        return simulate_glancing_scan_1d(
            potential,
            probe_rows[indices],
            starts[indices],
            window_length,
            kernel,
            slice_thickness,
            energy,
        )

    def evaluate(values, indices: np.ndarray) -> float:
        numerator = 0.0
        denominator = 0.0
        for start_index in range(0, len(indices), evaluation_batch_size):
            chosen = indices[start_index : start_index + evaluation_batch_size]
            predicted = predict(values, jnp.asarray(chosen))
            measured_batch = measured[chosen]
            numerator += float(
                jnp.sum((jnp.sqrt(predicted + 1e-12) - jnp.sqrt(measured_batch + 1e-12)) ** 2)
            )
            denominator += float(jnp.sum(measured_batch))
        return float(np.sqrt(numerator / max(denominator, 1e-12)))

    rng = np.random.default_rng(operator.index(seed))
    iterator = range(1, n_updates + 1)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc=progress_description, unit="update", dynamic_ncols=True)
    update_history: list[int] = []
    elapsed_history: list[float] = []
    training_history: list[float] = []
    validation_history: list[float] = []
    snapshots: dict[str, list[np.ndarray]] = {
        "translation": [], "strain": [], "rotation": [], "host_displacements": [],
        "host_occupancies": [], "host_species": [], "adatom_positions": [],
        "adatom_occupancies": [], "adatom_species": [],
    }
    best_values = parameters
    best_metric = np.inf
    best_update = 0
    start_time = perf_counter()

    def record(update, values):
        nonlocal best_values, best_metric, best_update
        training_loss = evaluate(values, training)
        validation_loss = evaluate(values, validation) if validation.size else np.nan
        metric = validation_loss if validation.size else training_loss
        host_positions, host_species, adatom_species = decode(values)
        del host_positions
        update_history.append(update)
        elapsed_history.append(perf_counter() - start_time)
        training_history.append(training_loss)
        validation_history.append(validation_loss)
        snapshots["translation"].append(np.asarray(values["translation"]))
        snapshots["strain"].append(np.asarray(values["strain"]))
        snapshots["rotation"].append(np.asarray(values["rotation"]))
        snapshots["host_displacements"].append(np.asarray(values["host_displacements"]))
        snapshots["host_occupancies"].append(np.asarray(values["host_occupancies"]))
        snapshots["host_species"].append(np.asarray(host_species))
        snapshots["adatom_positions"].append(np.asarray(values["adatom_positions"]))
        snapshots["adatom_occupancies"].append(np.asarray(values["adatom_occupancies"]))
        snapshots["adatom_species"].append(np.asarray(adatom_species))
        if metric < best_metric:
            best_metric = metric
            best_update = update
            best_values = {name: value for name, value in values.items()}

    record(0, parameters)
    for update in iterator:
        chosen = rng.choice(training, size=min(minibatch_size, training.size), replace=False)
        stage = 0 if update <= global_end else 1 if update <= host_end else 2 if update <= defect_end else 3
        final_fraction = max(update - defect_end, 0) / max(n_updates - defect_end, 1)
        elastic_scale = 1.0 - 0.75 * final_fraction
        discrete_scale = final_fraction
        parameters, state, _ = update_step(
            parameters,
            state,
            jnp.asarray(chosen),
            jnp.asarray(stage),
            jnp.asarray(elastic_scale),
            jnp.asarray(discrete_scale),
        )
        if update % validation_interval == 0 or update == n_updates:
            record(update, parameters)

    host_positions, host_species, adatom_species = decode(best_values)
    potential = render_crystalline_defects_1d(
        model,
        host_positions,
        best_values["host_occupancies"],
        host_species,
        best_values["adatom_positions"],
        best_values["adatom_occupancies"],
        adatom_species,
    )
    predicted = predict(best_values, jnp.arange(n_scan))
    metadata = {
        **dict(model.metadata),
        "species_names": list(model.species_names),
        "updates": n_updates,
        "stages": [global_end, host_end, defect_end, n_updates],
        "training_indices": training.tolist(),
        "validation_indices": validation.tolist(),
        "n_host_sites": int(n_host),
        "n_adatom_candidates": int(n_adatom),
        "n_bonds": int(model.host_bonds.shape[0]),
        "n_angles": int(model.host_angles.shape[0]),
        "effective_update_sites": float(jnp.sum(host_update_weights)),
        "fully_frozen_sites": int(jnp.sum(host_update_weights == 0.0)),
        "best_metric": best_metric,
        "seed": operator.index(seed),
        "enable_substitutions": bool(enable_substitutions),
        "enable_adatoms": bool(enable_adatoms),
        "enable_global_transform": bool(enable_global_transform),
        "enable_host_displacements": bool(enable_host_displacements),
        "enable_host_occupancies": bool(enable_host_occupancies),
        "buffer_defect_multiplier": float(buffer_defect_multiplier),
        "detector_angles_mrad": np.asarray(
            1e3 * jnp.arcsin(jnp.clip(
                energy2wavelength(energy)
                * jnp.fft.fftshift(jnp.fft.fftfreq(n_u, model.transverse_coordinates[1] - model.transverse_coordinates[0])),
                -1.0, 1.0,
            ))
        ).tolist(),
    }
    return CrystallineDefectReconstruction1D(
        host_positions_3d=host_positions,
        host_occupancies=best_values["host_occupancies"],
        host_species_probabilities=host_species,
        adatom_positions=best_values["adatom_positions"],
        adatom_occupancies=best_values["adatom_occupancies"],
        adatom_species_probabilities=adatom_species,
        translation=best_values["translation"],
        strain=best_values["strain"],
        rotation_rad=best_values["rotation"],
        potential=potential,
        predicted_intensities=predicted,
        measured_intensities=measured,
        update_history=jnp.asarray(update_history),
        elapsed_time_history=jnp.asarray(elapsed_history),
        training_loss_history=jnp.asarray(training_history),
        validation_loss_history=jnp.asarray(validation_history),
        translation_history=jnp.asarray(snapshots["translation"]),
        strain_history=jnp.asarray(snapshots["strain"]),
        rotation_history=jnp.asarray(snapshots["rotation"]),
        host_displacement_history=jnp.asarray(snapshots["host_displacements"]),
        host_occupancy_history=jnp.asarray(snapshots["host_occupancies"]),
        host_species_probability_history=jnp.asarray(snapshots["host_species"]),
        adatom_position_history=jnp.asarray(snapshots["adatom_positions"]),
        adatom_occupancy_history=jnp.asarray(snapshots["adatom_occupancies"]),
        adatom_species_probability_history=jnp.asarray(snapshots["adatom_species"]),
        best_update=best_update,
        metadata=metadata,
    )


def reconstruct_crystalline_host_1d(
    model: CrystallineHostModel1D,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    measured_intensities: Any,
    **kwargs: Any,
) -> CrystallineHostReconstruction1D:
    """Fit a pristine single-species host under Keating regularization.

    The specimen volume fixes the set of occupied host sites.  This wrapper
    removes substitution, vacancy, and adatom degrees of freedom while keeping
    the global lattice transform and optional local elastic displacements.
    """
    if len(model.species_names) != 1:
        raise ValueError("pristine host reconstruction requires exactly one species")
    controlled = {
        "enable_host_occupancies",
        "enable_substitutions",
        "enable_adatoms",
        "initial_host_occupancy",
    }
    overlap = controlled.intersection(kwargs)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise TypeError(f"pristine host reconstruction controls {names}")
    return reconstruct_crystalline_defects_1d(
        model,
        input_probe,
        window_starts,
        window_length,
        propagation_kernel,
        slice_thickness,
        energy,
        measured_intensities,
        initial_host_occupancy=1.0,
        enable_host_occupancies=False,
        enable_substitutions=False,
        enable_adatoms=False,
        **kwargs,
    )


def _json_default(value: Any) -> Any:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def save_crystalline_defect_reconstruction_1d(
    path: str | Path,
    result: CrystallineDefectReconstruction1D,
) -> None:
    """Save a reconstruction without pickle-backed object arrays."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.asarray(getattr(result, name))
        for name in result.__dataclass_fields__
        if name != "metadata"
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(dict(result.metadata), default=_json_default, sort_keys=True)
    )
    np.savez_compressed(destination, **arrays)


def load_crystalline_defect_reconstruction_1d(
    path: str | Path,
) -> CrystallineDefectReconstruction1D:
    """Load a result written by :func:`save_crystalline_defect_reconstruction_1d`."""
    with np.load(path, allow_pickle=False) as data:
        values = {
            name: (int(data[name]) if name == "best_update" else jnp.asarray(data[name]))
            for name in CrystallineDefectReconstruction1D.__dataclass_fields__
            if name != "metadata"
        }
        values["metadata"] = json.loads(str(data["metadata_json"].item()))
    return CrystallineDefectReconstruction1D(**values)
