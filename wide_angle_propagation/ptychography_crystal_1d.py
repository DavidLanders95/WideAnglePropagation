"""Full-slab sparse crystal-edit ptychography in one transverse dimension.

The specimen is a registered single-species crystal with projected host
displacements, discrete host removals, and a fixed-capacity set of off-lattice
atoms.  Diffraction updates, a weak linearized Keating proximal step, and a
temporary signed-pixel proposal step remain separate throughout the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import operator
from time import perf_counter
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .ptychography_1d import simulate_glancing_scan_1d
from .propagation_methods import energy2wavelength


__all__ = [
    "CrystalModel1D",
    "CrystalState1D",
    "CrystalReconstruction1D",
    "make_crystal_model_1d",
    "make_si_atom_template_1d",
    "render_crystal_1d",
    "reconstruct_crystal_1d",
]


Array = Any


@dataclass(frozen=True)
class CrystalModel1D:
    """A complete crystalline host and its training-defined mutable wedge."""

    axial_coordinates: Array
    transverse_coordinates: Array
    atom_template: Array
    reference_positions_3d: Array
    host_mobility: Array
    full_mobility_mask: Array
    scratch_mask: Array
    insertion_anchors_3d: Array
    bond_indices: Array
    bond_vectors_3d: Array
    angle_indices: Array
    angle_vectors_3d: Array
    axial_period_A: float
    latent_period_A: float
    slab_bounds_A: tuple[float, float]
    max_host_removals: int = 4
    max_extra_atoms: int = 4
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrystalState1D:
    """One fixed-capacity physical crystal state."""

    registration: Array
    host_displacements: Array
    removed_host_mask: Array
    extra_positions_3d: Array
    extra_active_mask: Array


@dataclass(frozen=True)
class CrystalReconstruction1D:
    """Final crystal estimate, proposal evidence, and compact event history."""

    state: CrystalState1D
    potential: Array
    predicted_intensities: Array
    measured_intensities: Array
    detector_angles_mrad: Array
    training_indices: Array
    selection_indices: Array
    audit_indices: Array
    target_nrmse: Array
    training_nrmse: Array
    selection_nrmse: Array
    audit_nrmse: Array
    termination_reason: str
    registration_history: Array
    registration_loss_history: Array
    event_stages: tuple[str, ...]
    event_updates: Array
    host_displacement_history: Array
    removed_host_history: Array
    extra_position_history: Array
    extra_active_history: Array
    training_nrmse_history: Array
    selection_nrmse_history: Array
    scratch_event_indices: Array
    scratch_residual_history: Array
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _array(name: str, value: Any, ndim: int) -> Array:
    result = jnp.asarray(value)
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {result.shape}")
    return result


def _integer(name: str, value: Any, *, minimum: int = 1) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_positive(name: str, value: Any, *, allow_zero: bool = False) -> float:
    result = float(value)
    valid = np.isfinite(result) and (result >= 0.0 if allow_zero else result > 0.0)
    if not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}")
    return result


def _uniform_coordinates(name: str, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a one-dimensional finite grid")
    differences = np.diff(array)
    if np.any(differences <= 0.0) or not np.allclose(
        differences, differences[0], rtol=5e-4, atol=1e-8
    ):
        raise ValueError(f"{name} must be uniformly increasing")
    return array


def make_si_atom_template_1d(
    axial_sampling: float,
    transverse_sampling: float,
    *,
    cutoff_A: float = 4.0,
    projection_width_A: float = 5.431,
) -> Array:
    """Generate one centred finite-projection Lobato silicon template."""
    try:
        import abtem
        from ase import Atoms
    except ImportError as exc:  # pragma: no cover - optional scientific stack
        raise ImportError("make_si_atom_template_1d requires abTEM and ASE") from exc
    ds = _finite_positive("axial_sampling", axial_sampling)
    du = _finite_positive("transverse_sampling", transverse_sampling)
    cutoff = _finite_positive("cutoff_A", cutoff_A)
    width = _finite_positive("projection_width_A", projection_width_A)
    half_s = int(np.ceil(cutoff / ds))
    half_u = int(np.ceil(cutoff / du))
    n_s, n_u = 2 * half_s + 1, 2 * half_u + 1
    atom = Atoms(
        "Si",
        positions=[[half_u * du, 0.5 * width, half_s * ds]],
        cell=np.diag([n_u * du, width, n_s * ds]),
        pbc=[False, True, False],
    )
    builder = abtem.Potential(
        atom,
        gpts=(n_u, n_s),
        slice_thickness=width,
        projection="finite",
        parametrization="lobato",
        plane="xz",
        periodic=False,
        device="cpu",
    )
    return jnp.asarray(np.asarray(builder.build(lazy=False).array)[0].T / width)


def _smooth_mobility(distance: np.ndarray, inner: float, outer: float) -> np.ndarray:
    fraction = np.clip((outer - distance) / (outer - inner), 0.0, 1.0)
    return fraction * fraction * (3.0 - 2.0 * fraction)


def _ray_distance(
    axial: np.ndarray,
    transverse: np.ndarray,
    landings: np.ndarray,
    beam_tilt_rad: float,
) -> np.ndarray:
    """Distance to the nearest post-landing ray on a broadcastable grid."""
    result = np.full(np.broadcast_shapes(axial.shape, transverse.shape), np.inf)
    tangent = np.tan(float(beam_tilt_rad))
    for landing in landings:
        post_landing = axial >= landing
        ray_u = (axial - landing) * tangent
        candidate = np.where(post_landing, np.abs(transverse - ray_u), np.inf)
        result = np.minimum(result, candidate)
    return result


def _periodic_diamond_graph(
    positions_3d: np.ndarray,
    latent_period_A: float,
    bond_cutoff_A: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build nearest-neighbour bonds and angles with periodic latent ``y``."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - required scientific stack
        raise ImportError("crystal graph construction requires SciPy") from exc
    positions = np.asarray(positions_3d, dtype=float).copy()
    period = float(latent_period_A)
    positions[:, 1] = np.mod(positions[:, 1], period)
    tiled = np.concatenate(
        [
            positions - np.asarray([0.0, period, 0.0]),
            positions,
            positions + np.asarray([0.0, period, 0.0]),
        ],
        axis=0,
    )
    sources = np.tile(np.arange(len(positions), dtype=np.int32), 3)
    tree = cKDTree(tiled)
    bonds: dict[tuple[int, int], np.ndarray] = {}
    neighbors: list[list[tuple[int, np.ndarray]]] = [[] for _ in positions]
    for first, point in enumerate(positions):
        local: dict[int, np.ndarray] = {}
        for tiled_index in tree.query_ball_point(point, bond_cutoff_A):
            second = int(sources[tiled_index])
            delta = tiled[tiled_index] - point
            distance = float(np.linalg.norm(delta))
            if distance < 1e-8 or distance > bond_cutoff_A:
                continue
            if second not in local or distance < np.linalg.norm(local[second]):
                local[second] = delta
        neighbors[first] = sorted(local.items(), key=lambda item: item[0])
        for second, delta in neighbors[first]:
            if first < second:
                bonds[(first, second)] = delta
            elif second < first and (second, first) not in bonds:
                bonds[(second, first)] = -delta
    bond_keys = sorted(bonds)
    bond_indices = np.asarray(bond_keys, dtype=np.int32).reshape(-1, 2)
    bond_vectors = np.asarray([bonds[key] for key in bond_keys], dtype=float).reshape(-1, 3)
    angle_indices: list[tuple[int, int, int]] = []
    angle_vectors: list[tuple[np.ndarray, np.ndarray]] = []
    for center, local in enumerate(neighbors):
        for (first, vector_first), (second, vector_second) in itertools.combinations(
            local, 2
        ):
            angle_indices.append((center, first, second))
            angle_vectors.append((vector_first, vector_second))
    return (
        bond_indices,
        bond_vectors,
        np.asarray(angle_indices, dtype=np.int32).reshape(-1, 3),
        np.asarray(angle_vectors, dtype=float).reshape(-1, 2, 3),
    )


def make_crystal_model_1d(
    axial_coordinates: Any,
    transverse_coordinates: Any,
    atom_template: Any,
    reference_positions_3d: Any,
    scan_coordinates_A: Any,
    training_indices: Sequence[int],
    *,
    beam_tilt_rad: float,
    airy_first_zero_A: float,
    slab_bounds_A: tuple[float, float],
    axial_period_A: float,
    latent_period_A: float,
    insertion_grid_spacing_A: float = 0.75,
    insertion_vacuum_A: float = 4.0,
    bond_cutoff_A: float = 2.65,
    max_host_removals: int = 4,
    max_extra_atoms: int = 4,
    metadata: Mapping[str, Any] | None = None,
) -> CrystalModel1D:
    """Build the complete host, training-ray wedge, and sparse Keating graph."""
    coordinates_s = _uniform_coordinates("axial_coordinates", axial_coordinates)
    coordinates_u = _uniform_coordinates("transverse_coordinates", transverse_coordinates)
    template = np.asarray(atom_template)
    positions = np.asarray(reference_positions_3d, dtype=float)
    scans = np.asarray(scan_coordinates_A, dtype=float)
    training = np.asarray(training_indices)
    if template.ndim != 2 or min(template.shape) < 3 or any(
        size % 2 == 0 for size in template.shape
    ):
        raise ValueError("atom_template must be a real odd-sized two-dimensional array")
    if np.iscomplexobj(template) or not np.all(np.isfinite(template)):
        raise ValueError("atom_template must be finite and real")
    if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) == 0:
        raise ValueError("reference_positions_3d must have shape (n_host, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("reference_positions_3d must be finite")
    if scans.ndim != 1 or not np.all(np.isfinite(scans)) or scans.size == 0:
        raise ValueError("scan_coordinates_A must be a non-empty finite vector")
    if training.ndim != 1 or not np.issubdtype(training.dtype, np.integer):
        raise TypeError("training_indices must be a one-dimensional integer sequence")
    training = training.astype(np.int32, copy=False)
    if (
        training.size == 0
        or np.unique(training).size != training.size
        or np.any(training < 0)
        or np.any(training >= scans.size)
    ):
        raise ValueError("training_indices must contain unique valid scan indices")
    projected = positions[:, [0, 2]]
    grid_bounds = np.asarray(
        [[coordinates_s[0], coordinates_s[-1]], [coordinates_u[0], coordinates_u[-1]]]
    )
    if np.any(projected < grid_bounds[:, 0]) or np.any(projected > grid_bounds[:, 1]):
        raise ValueError("all projected host positions must lie inside the specimen grid")
    bottom, top = (float(value) for value in slab_bounds_A)
    if not np.isfinite(bottom) or not np.isfinite(top) or bottom >= top:
        raise ValueError("slab_bounds_A must be finite with bottom < top")
    airy = _finite_positive("airy_first_zero_A", airy_first_zero_A)
    axial_period = _finite_positive("axial_period_A", axial_period_A)
    latent_period = _finite_positive("latent_period_A", latent_period_A)
    spacing = _finite_positive("insertion_grid_spacing_A", insertion_grid_spacing_A)
    vacuum = _finite_positive("insertion_vacuum_A", insertion_vacuum_A, allow_zero=True)
    cutoff = _finite_positive("bond_cutoff_A", bond_cutoff_A)
    removals = _integer("max_host_removals", max_host_removals)
    additions = _integer("max_extra_atoms", max_extra_atoms)
    train_landings = scans[training]
    inner_radius, outer_radius = 2.5 * airy, 4.0 * airy
    host_distance = _ray_distance(
        positions[:, 0], positions[:, 2], train_landings, beam_tilt_rad
    )
    mobility = _smooth_mobility(host_distance, inner_radius, outer_radius)
    full_mobility = mobility >= 1.0 - 1e-12

    grid_s = coordinates_s[:, None]
    grid_u = coordinates_u[None, :]
    pixel_distance = _ray_distance(grid_s, grid_u, train_landings, beam_tilt_rad)
    template_halo = max(
        (template.shape[0] // 2) * np.diff(coordinates_s)[0],
        (template.shape[1] // 2) * np.diff(coordinates_u)[0],
    )
    scratch_distance = np.maximum(pixel_distance - template_halo, 0.0)
    scratch_mask = _smooth_mobility(scratch_distance, inner_radius, outer_radius)
    scratch_mask *= (
        (grid_u >= bottom - template_halo)
        & (grid_u <= top + vacuum + template_halo)
    )

    anchor_s = np.arange(float(np.min(train_landings)), coordinates_s[-1] + spacing, spacing)
    anchor_u = np.arange(bottom, top + vacuum + 0.5 * spacing, spacing)
    anchor_grid_s, anchor_grid_u = np.meshgrid(anchor_s, anchor_u, indexing="ij")
    anchor_distance = _ray_distance(
        anchor_grid_s, anchor_grid_u, train_landings, beam_tilt_rad
    )
    selected = anchor_distance <= inner_radius
    anchors = np.stack(
        [
            anchor_grid_s[selected],
            np.full(np.count_nonzero(selected), 0.25 * latent_period),
            anchor_grid_u[selected],
        ],
        axis=1,
    )
    bonds, bond_vectors, angles, angle_vectors = _periodic_diamond_graph(
        positions, latent_period, cutoff
    )
    return CrystalModel1D(
        axial_coordinates=jnp.asarray(coordinates_s),
        transverse_coordinates=jnp.asarray(coordinates_u),
        atom_template=jnp.asarray(template),
        reference_positions_3d=jnp.asarray(positions),
        host_mobility=jnp.asarray(mobility),
        full_mobility_mask=jnp.asarray(full_mobility),
        scratch_mask=jnp.asarray(scratch_mask),
        insertion_anchors_3d=jnp.asarray(anchors),
        bond_indices=jnp.asarray(bonds),
        bond_vectors_3d=jnp.asarray(bond_vectors),
        angle_indices=jnp.asarray(angles),
        angle_vectors_3d=jnp.asarray(angle_vectors),
        axial_period_A=axial_period,
        latent_period_A=latent_period,
        slab_bounds_A=(bottom, top),
        max_host_removals=removals,
        max_extra_atoms=additions,
        metadata={
            "beam_tilt_rad": float(beam_tilt_rad),
            "airy_first_zero_A": airy,
            "mobility_inner_radius_A": inner_radius,
            "mobility_outer_radius_A": outer_radius,
            "training_indices": training.tolist(),
            **dict(metadata or {}),
        },
    )


def _apply_registration(reference_positions_3d: Array, parameters: Array) -> Array:
    reference = jnp.asarray(reference_positions_3d)
    projected = reference[:, [0, 2]]
    center = jnp.mean(projected, axis=0)
    relative = projected - center
    strained_s = relative[:, 0] * (1.0 + parameters[3])
    cosine, sine = jnp.cos(parameters[2]), jnp.sin(parameters[2])
    transformed = jnp.stack(
        [
            cosine * strained_s - sine * relative[:, 1],
            sine * strained_s + cosine * relative[:, 1],
        ],
        axis=1,
    )
    transformed = transformed + center + parameters[:2]
    return reference.at[:, 0].set(transformed[:, 0]).at[:, 2].set(transformed[:, 1])


def _state_positions(model: CrystalModel1D, state: CrystalState1D) -> tuple[Array, Array]:
    registered = _apply_registration(model.reference_positions_3d, state.registration)
    displaced = registered.at[:, 0].add(state.host_displacements[:, 0])
    displaced = displaced.at[:, 2].add(state.host_displacements[:, 1])
    return displaced, jnp.asarray(state.extra_positions_3d)


def _cubic_weights(fraction: Array) -> Array:
    return jnp.stack(
        [
            -fraction * (1.0 - fraction) * (2.0 - fraction) / 6.0,
            (1.0 + fraction) * (1.0 - fraction) * (2.0 - fraction) / 2.0,
            (1.0 + fraction) * fraction * (2.0 - fraction) / 2.0,
            -(1.0 + fraction) * fraction * (1.0 - fraction) / 6.0,
        ]
    )


def _splat_weighted_sites(
    model: CrystalModel1D,
    positions_3d: Array,
    site_weights: Array,
) -> Array:
    coordinates_s = jnp.asarray(model.axial_coordinates)
    coordinates_u = jnp.asarray(model.transverse_coordinates)
    positions = jnp.asarray(positions_3d)
    weights = jnp.asarray(site_weights)
    projected = positions[:, [0, 2]]
    ds, du = coordinates_s[1] - coordinates_s[0], coordinates_u[1] - coordinates_u[0]
    output_shape = (coordinates_s.shape[0], coordinates_u.shape[0])
    fractional = jnp.stack(
        [
            (projected[:, 0] - coordinates_s[0]) / ds,
            (projected[:, 1] - coordinates_u[0]) / du,
        ],
        axis=1,
    )
    lower = jnp.floor(fractional).astype(jnp.int32)
    fractions = fractional - lower
    weights_s = _cubic_weights(fractions[:, 0])
    weights_u = _cubic_weights(fractions[:, 1])
    corner_weights = weights_s[:, None, :] * weights_u[None, :, :]
    corner_weights = corner_weights * weights[None, None, :]
    offsets = jnp.arange(-1, 3, dtype=jnp.int32)
    offset_s, offset_u = jnp.meshgrid(offsets, offsets, indexing="ij")
    corners = lower[None, None, :, :] + jnp.stack([offset_s, offset_u], axis=-1)[
        :, :, None, :
    ]
    shape_array = jnp.asarray(output_shape, dtype=jnp.int32)
    valid = jnp.all((corners >= 0) & (corners < shape_array), axis=-1)
    clipped = jnp.clip(corners, 0, shape_array - 1)
    result = jnp.zeros(output_shape, dtype=jnp.result_type(model.atom_template, weights))
    return result.at[clipped[..., 0].reshape(-1), clipped[..., 1].reshape(-1)].add(
        jnp.where(valid, corner_weights, 0.0).reshape(-1)
    )


def _template_frequency(model: CrystalModel1D) -> tuple[Array, tuple[int, int]]:
    output_shape = (
        model.axial_coordinates.shape[0],
        model.transverse_coordinates.shape[0],
    )
    template_shape = model.atom_template.shape
    full_shape = (
        output_shape[0] + template_shape[0] - 1,
        output_shape[1] + template_shape[1] - 1,
    )
    return jnp.fft.rfftn(model.atom_template, full_shape, axes=(0, 1)), full_shape


def _convolve_site_grid(
    model: CrystalModel1D,
    site_grid: Array,
    template_frequency: Array,
    full_shape: tuple[int, int],
) -> Array:
    full = jnp.fft.irfftn(
        jnp.fft.rfftn(site_grid, full_shape, axes=(0, 1)) * template_frequency,
        full_shape,
        axes=(0, 1),
    )
    start_s = (model.atom_template.shape[0] - 1) // 2
    start_u = (model.atom_template.shape[1] - 1) // 2
    return full[
        start_s : start_s + site_grid.shape[0],
        start_u : start_u + site_grid.shape[1],
    ]


def _render_with_frequency(
    model: CrystalModel1D,
    state: CrystalState1D,
    template_frequency: Array,
    full_shape: tuple[int, int],
) -> Array:
    host_positions, extra_positions = _state_positions(model, state)
    host_weights = 1.0 - jnp.asarray(state.removed_host_mask, dtype=model.atom_template.dtype)
    extra_weights = jnp.asarray(state.extra_active_mask, dtype=model.atom_template.dtype)
    host_grid = _splat_weighted_sites(model, host_positions, host_weights)
    extra_grid = _splat_weighted_sites(model, extra_positions, extra_weights)
    return _convolve_site_grid(
        model, host_grid + extra_grid, template_frequency, full_shape
    )


def render_crystal_1d(model: CrystalModel1D, state: CrystalState1D) -> Array:
    """Render one physical crystal state on the complete specimen grid."""
    if not isinstance(model, CrystalModel1D) or not isinstance(state, CrystalState1D):
        raise TypeError("model and state must be crystal model/state instances")
    registration = _array("state.registration", state.registration, 1)
    displacement = _array("state.host_displacements", state.host_displacements, 2)
    removed = _array("state.removed_host_mask", state.removed_host_mask, 1)
    extras = _array("state.extra_positions_3d", state.extra_positions_3d, 2)
    active = _array("state.extra_active_mask", state.extra_active_mask, 1)
    n_host = model.reference_positions_3d.shape[0]
    if registration.shape != (4,) or displacement.shape != (n_host, 2):
        raise ValueError("registration or host-displacement shape does not match model")
    if removed.shape != (n_host,):
        raise ValueError("removed_host_mask must contain one value per host")
    if extras.shape != (model.max_extra_atoms, 3) or active.shape != (
        model.max_extra_atoms,
    ):
        raise ValueError("extra-atom state must match the model capacity")
    frequency, full_shape = _template_frequency(model)
    return _render_with_frequency(model, state, frequency, full_shape)


def _minimum_image_y(delta_y: Array, period: float) -> Array:
    return jnp.mod(delta_y + 0.5 * period, period) - 0.5 * period


def _keating_quadratic_1d(
    model: CrystalModel1D,
    host_displacements: Array,
    removed_host_mask: Array,
) -> Array:
    """Linearized Si Keating energy without constructing a dense Hessian."""
    displacement = jnp.asarray(host_displacements)
    displacement_3d = jnp.stack(
        [displacement[:, 0], jnp.zeros(displacement.shape[0]), displacement[:, 1]],
        axis=1,
    )
    active = ~jnp.asarray(removed_host_mask, dtype=bool)
    bond_indices = jnp.asarray(model.bond_indices, dtype=jnp.int32)
    bond_vectors = jnp.asarray(model.bond_vectors_3d)
    first, second = bond_indices[:, 0], bond_indices[:, 1]
    bond_delta = displacement_3d[second] - displacement_3d[first]
    bond_linear = 2.0 * jnp.sum(bond_vectors * bond_delta, axis=1)
    bond_active = active[first] & active[second]

    angle_indices = jnp.asarray(model.angle_indices, dtype=jnp.int32)
    angle_vectors = jnp.asarray(model.angle_vectors_3d)
    center, neighbor_first, neighbor_second = (
        angle_indices[:, 0],
        angle_indices[:, 1],
        angle_indices[:, 2],
    )
    delta_first = displacement_3d[neighbor_first] - displacement_3d[center]
    delta_second = displacement_3d[neighbor_second] - displacement_3d[center]
    angle_linear = jnp.sum(angle_vectors[:, 1] * delta_first, axis=1) + jnp.sum(
        angle_vectors[:, 0] * delta_second, axis=1
    )
    angle_active = active[center] & active[neighbor_first] & active[neighbor_second]

    bond_length = float(model.latent_period_A) * np.sqrt(3.0) / 4.0
    alpha = 2.965
    beta = 0.285 * alpha
    stretch = (3.0 * alpha / (16.0 * bond_length**2)) * jnp.sum(
        jnp.where(bond_active, bond_linear**2, 0.0)
    )
    bend = (3.0 * beta / (8.0 * bond_length**2)) * jnp.sum(
        jnp.where(angle_active, angle_linear**2, 0.0)
    )
    n_terms = jnp.maximum(
        jnp.sum(bond_active.astype(displacement.dtype))
        + jnp.sum(angle_active.astype(displacement.dtype)),
        1.0,
    )
    return (stretch + bend) / n_terms


def _hard_core_penalty_1d(model: CrystalModel1D, state: CrystalState1D) -> Array:
    host_positions, extra_positions = _state_positions(model, state)
    active_host = ~jnp.asarray(state.removed_host_mask, dtype=bool)
    active_extra = jnp.asarray(state.extra_active_mask, dtype=bool)
    bond_indices = jnp.asarray(model.bond_indices, dtype=jnp.int32)
    first, second = bond_indices[:, 0], bond_indices[:, 1]
    host_delta = host_positions[second] - host_positions[first]
    host_delta = host_delta.at[:, 1].set(
        _minimum_image_y(host_delta[:, 1], model.latent_period_A)
    )
    def safe_distance(delta: Array) -> Array:
        return jnp.sqrt(jnp.sum(delta**2, axis=-1) + 1e-12)

    host_distance = safe_distance(host_delta)
    host_mask = active_host[first] & active_host[second]

    cross_delta = extra_positions[:, None, :] - host_positions[None, :, :]
    cross_delta = cross_delta.at[:, :, 1].set(
        _minimum_image_y(cross_delta[:, :, 1], model.latent_period_A)
    )
    cross_distance = safe_distance(cross_delta)
    cross_mask = active_extra[:, None] & active_host[None, :]

    extra_delta = extra_positions[:, None, :] - extra_positions[None, :, :]
    extra_delta = extra_delta.at[:, :, 1].set(
        _minimum_image_y(extra_delta[:, :, 1], model.latent_period_A)
    )
    extra_distance = safe_distance(extra_delta)
    extra_mask = active_extra[:, None] & active_extra[None, :]
    extra_mask &= jnp.triu(jnp.ones_like(extra_mask, dtype=bool), k=1)

    def violation(distance: Array, mask: Array) -> Array:
        scaled = jax.nn.relu(1.8 - distance) / 0.05
        return jnp.sum(jnp.where(mask, scaled**2, 0.0))

    count = jnp.maximum(
        jnp.sum(host_mask) + jnp.sum(cross_mask) + jnp.sum(extra_mask), 1
    )
    return (
        violation(host_distance, host_mask)
        + violation(cross_distance, cross_mask)
        + violation(extra_distance, extra_mask)
    ) / count


def _balanced_amplitude_loss(
    predicted: Array,
    measured: Array,
    reflected_mask: Array,
    whole_detector_weight: float,
    epsilon: float = 1e-12,
) -> Array:
    amplitude_error = (jnp.sqrt(predicted + epsilon) - jnp.sqrt(measured + epsilon)) ** 2
    all_loss = jnp.sum(amplitude_error) / jnp.maximum(jnp.sum(measured), epsilon)
    reflected = jnp.asarray(reflected_mask, dtype=predicted.dtype)
    reflected_loss = jnp.sum(amplitude_error * reflected[None, :]) / jnp.maximum(
        jnp.sum(measured * reflected[None, :]), epsilon
    )
    return whole_detector_weight * all_loss + (1.0 - whole_detector_weight) * reflected_loss


def _state_from_values(
    registration: Array,
    values: Mapping[str, Array],
    removed_host_mask: Array,
    extra_active_mask: Array,
    latent_y_A: float,
) -> CrystalState1D:
    positions_su = values["extra_positions_su"]
    extras = jnp.stack(
        [
            positions_su[:, 0],
            jnp.full(positions_su.shape[0], latent_y_A, dtype=positions_su.dtype),
            positions_su[:, 1],
        ],
        axis=1,
    )
    return CrystalState1D(
        registration=registration,
        host_displacements=values["host_displacements"],
        removed_host_mask=removed_host_mask,
        extra_positions_3d=extras,
        extra_active_mask=extra_active_mask,
    )


def _partition_indices(name: str, values: Sequence[int], n_scan: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be a one-dimensional integer sequence")
    array = array.astype(np.int32, copy=False)
    if (
        array.size == 0
        or np.unique(array).size != array.size
        or np.any(array < 0)
        or np.any(array >= n_scan)
    ):
        raise ValueError(f"{name} must contain unique valid scan indices")
    return array


def _empty_state(model: CrystalModel1D, registration: Array) -> CrystalState1D:
    dtype = model.reference_positions_3d.dtype
    extras = jnp.zeros((model.max_extra_atoms, 3), dtype=dtype)
    extras = extras.at[:, 1].set(0.25 * model.latent_period_A)
    return CrystalState1D(
        registration=jnp.asarray(registration, dtype=dtype),
        host_displacements=jnp.zeros((model.reference_positions_3d.shape[0], 2), dtype=dtype),
        removed_host_mask=jnp.zeros(model.reference_positions_3d.shape[0], dtype=bool),
        extra_positions_3d=extras,
        extra_active_mask=jnp.zeros(model.max_extra_atoms, dtype=bool),
    )


def _matched_filter_residual_1d(
    model: CrystalModel1D,
    residual: Any,
) -> np.ndarray:
    """Correlate a signed scratch field with one silicon atom template."""
    try:
        from scipy.signal import fftconvolve
    except ImportError as exc:  # pragma: no cover - required scientific stack
        raise ImportError("temporary residual scoring requires SciPy") from exc
    residual_array = np.asarray(residual, dtype=np.float32)
    expected_shape = (
        model.axial_coordinates.shape[0],
        model.transverse_coordinates.shape[0],
    )
    if residual_array.shape != expected_shape:
        raise ValueError(f"residual must have shape {expected_shape}")
    template = np.asarray(model.atom_template, dtype=np.float32)
    normalizer = max(float(np.sum(template**2)), 1e-12)
    return fftconvolve(
        residual_array, template[::-1, ::-1], mode="same"
    ) / normalizer


def _sample_score_map_1d(
    model: CrystalModel1D,
    score_map: Any,
    positions_su: Any,
) -> np.ndarray:
    scores = np.asarray(score_map)
    positions = np.asarray(positions_su)
    coordinates_s = np.asarray(model.axial_coordinates)
    coordinates_u = np.asarray(model.transverse_coordinates)
    ds = float(coordinates_s[1] - coordinates_s[0])
    du = float(coordinates_u[1] - coordinates_u[0])
    indices_s = np.rint((positions[:, 0] - coordinates_s[0]) / ds).astype(int)
    indices_u = np.rint((positions[:, 1] - coordinates_u[0]) / du).astype(int)
    indices_s = np.clip(indices_s, 0, scores.shape[0] - 1)
    indices_u = np.clip(indices_u, 0, scores.shape[1] - 1)
    return scores[indices_s, indices_u]


def _rank_topology_proposals_1d(
    model: CrystalModel1D,
    state: CrystalState1D,
    score_map: Any,
) -> list[tuple[str, int, float, np.ndarray]]:
    """Return at most two negative host and two non-maximal positive peaks."""
    host_positions = np.asarray(_state_positions(model, state)[0])
    removed = np.asarray(state.removed_host_mask)
    eligible_removals = np.asarray(model.full_mobility_mask) & ~removed
    removal_scores = -_sample_score_map_1d(
        model, score_map, host_positions[:, [0, 2]]
    )
    proposals: list[tuple[str, int, float, np.ndarray]] = []
    if int(np.sum(removed)) < model.max_host_removals:
        for index in np.argsort(removal_scores)[::-1]:
            if eligible_removals[index] and removal_scores[index] > 0.0:
                proposals.append(
                    (
                        "remove",
                        int(index),
                        float(removal_scores[index]),
                        host_positions[index, [0, 2]],
                    )
                )
            if sum(item[0] == "remove" for item in proposals) == 2:
                break

    active_extra = np.asarray(state.extra_positions_3d)[
        np.asarray(state.extra_active_mask)
    ]
    if int(np.sum(np.asarray(state.extra_active_mask))) >= model.max_extra_atoms:
        return proposals
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - required scientific stack
        raise ImportError("insertion proposal scoring requires SciPy") from exc
    anchors = np.asarray(model.insertion_anchors_3d)
    active_host = host_positions[~removed]
    tiled_host = np.concatenate(
        [
            active_host - np.asarray([0.0, model.latent_period_A, 0.0]),
            active_host,
            active_host + np.asarray([0.0, model.latent_period_A, 0.0]),
        ]
    )
    nearest_distance, _ = cKDTree(tiled_host).query(anchors, k=1)
    allowed = nearest_distance >= 1.8
    if len(active_extra):
        extra_distance, _ = cKDTree(active_extra).query(anchors, k=1)
        allowed &= extra_distance >= 1.8
    addition_scores = _sample_score_map_1d(model, score_map, anchors[:, [0, 2]])
    chosen_positions: list[np.ndarray] = []
    for anchor_index in np.argsort(addition_scores)[::-1]:
        position_su = anchors[anchor_index, [0, 2]]
        separated = all(
            np.linalg.norm(position_su - previous) >= 1.8
            for previous in chosen_positions
        )
        if allowed[anchor_index] and addition_scores[anchor_index] > 0.0 and separated:
            proposals.append(
                (
                    "add",
                    int(anchor_index),
                    float(addition_scores[anchor_index]),
                    position_su,
                )
            )
            chosen_positions.append(position_su)
        if len(chosen_positions) == 2:
            break
    return proposals


def _proximal_keating_step(
    model: CrystalModel1D,
    host_displacements: Array,
    removed_host_mask: Array,
    *,
    sigma_A: float,
    strength: float,
    cg_iterations: int,
) -> Array:
    from jax.scipy.sparse.linalg import cg

    displacement = jnp.asarray(host_displacements)
    sigma_squared = jnp.asarray(sigma_A**2, dtype=displacement.dtype)

    def quadratic(values: Array) -> Array:
        return 0.5 * _keating_quadratic_1d(model, values, removed_host_mask)

    def matrix_vector(values: Array) -> Array:
        hessian_values = jax.grad(quadratic)(values)
        return values + strength * hessian_values / sigma_squared

    solution, _ = cg(
        matrix_vector,
        displacement,
        x0=displacement,
        tol=0.0,
        atol=0.0,
        maxiter=cg_iterations,
    )
    bounds = 0.5 * jnp.asarray(model.host_mobility)[:, None]
    return jnp.clip(solution, -bounds, bounds)


def _backtracked_proximal_keating_step(
    model: CrystalModel1D,
    host_displacements: Array,
    removed_host_mask: Array,
    host_update_mask: Array,
    evaluate_training: Any,
    current_training_nrmse: float,
    *,
    sigma_A: float,
    initial_strength: float,
    cg_iterations: int,
) -> tuple[Array, float, float, tuple[tuple[float, float], ...]]:
    """Try at most three halved mechanics strengths and retain only a safe step."""
    displacement = jnp.asarray(host_displacements)
    update_mask = jnp.asarray(host_update_mask, dtype=bool)
    strength = float(initial_strength)
    trials: list[tuple[float, float]] = []
    for _ in range(3):
        candidate = _proximal_keating_step(
            model,
            displacement,
            removed_host_mask,
            sigma_A=sigma_A,
            strength=strength,
            cg_iterations=cg_iterations,
        )
        candidate = jnp.where(update_mask[:, None], candidate, displacement)
        candidate_training = float(evaluate_training(candidate))
        trials.append((strength, candidate_training))
        if candidate_training <= 1.005 * current_training_nrmse:
            return candidate, candidate_training, strength, tuple(trials)
        strength *= 0.5
    return displacement, float(current_training_nrmse), 0.0, tuple(trials)


def reconstruct_crystal_1d(
    model: CrystalModel1D,
    input_probes: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    measured_intensities: Any,
    detector_angles_mrad: Any,
    *,
    training_indices: Sequence[int],
    selection_indices: Sequence[int],
    audit_indices: Sequence[int],
    target_nrmse: float,
    initial_registration: Any | None = None,
    reflected_angle_bounds_mrad: tuple[float, float] = (0.0, 80.0),
    whole_detector_weight: float = 0.5,
    registration_phase_points: int = 25,
    registration_updates: int = 50,
    initial_cycles: int = 6,
    accepted_cycles: int = 4,
    screening_cycles: int = 2,
    final_cycles: int = 8,
    data_updates_per_cycle: int = 10,
    data_batch_size: int = 3,
    data_learning_rate_A: float = 1e-2,
    hard_core_weight: float = 1e-2,
    mechanics_sigma_A: float = 0.15,
    mechanics_strength: float = 0.1,
    mechanics_cg_iterations: int = 8,
    scratch_updates: int = 5,
    scratch_learning_rate: float = 5e-2,
    scratch_stride: tuple[int, int] = (5, 2),
    max_active_iterations: int = 8,
    minimum_selection_improvement: float = 1e-5,
    progress: bool = False,
    progress_description: str = "crystal ptychography",
) -> CrystalReconstruction1D:
    """Register and reconstruct a full crystal with sparse discrete edits."""
    try:
        import optax
    except ImportError as exc:  # pragma: no cover - optional optimizer
        raise ImportError("reconstruct_crystal_1d requires Optax") from exc
    if not isinstance(model, CrystalModel1D):
        raise TypeError("model must be CrystalModel1D")
    reconstruction_start_time = perf_counter()
    if not isinstance(progress, (bool, np.bool_)) or not isinstance(
        progress_description, str
    ):
        raise TypeError("progress must be boolean and its description must be text")
    probes = jnp.asarray(input_probes)
    measured = _array("measured_intensities", measured_intensities, 2)
    starts = _array("window_starts", window_starts, 1)
    kernel = _array("propagation_kernel", propagation_kernel, 1)
    detector_angles = _array("detector_angles_mrad", detector_angles_mrad, 1)
    n_scan, n_u = measured.shape
    if probes.ndim == 1:
        probes = jnp.broadcast_to(probes, (n_scan, n_u))
    if probes.shape != (n_scan, n_u):
        raise ValueError("input_probes must have one transverse row per scan")
    if starts.shape != (n_scan,) or kernel.shape != (n_u,) or detector_angles.shape != (
        n_u,
    ):
        raise ValueError("scan, kernel, detector, and measurement shapes do not match")
    length = _integer("window_length", window_length)
    if length != model.axial_coordinates.shape[0]:
        raise ValueError("the crystal workflow requires the complete specimen window")
    if np.any(np.asarray(starts) != 0):
        raise ValueError("the full-slab crystal workflow requires zero window starts")
    if not np.all(np.isfinite(np.asarray(measured))) or np.any(np.asarray(measured) < 0):
        raise ValueError("measured_intensities must be finite and non-negative")
    training = _partition_indices("training_indices", training_indices, n_scan)
    selection = _partition_indices("selection_indices", selection_indices, n_scan)
    audit = _partition_indices("audit_indices", audit_indices, n_scan)
    if set(training) & set(selection) or set(training) & set(audit) or set(selection) & set(audit):
        raise ValueError("training, selection, and audit partitions must be disjoint")
    target = _finite_positive("target_nrmse", target_nrmse)
    whole_weight = float(whole_detector_weight)
    if not 0.0 <= whole_weight <= 1.0:
        raise ValueError("whole_detector_weight must lie in [0, 1]")
    reflected_bounds = np.asarray(reflected_angle_bounds_mrad, dtype=float)
    if reflected_bounds.shape != (2,) or reflected_bounds[1] <= reflected_bounds[0]:
        raise ValueError("reflected_angle_bounds_mrad must be an increasing pair")
    reflected_mask = (detector_angles >= reflected_bounds[0]) & (
        detector_angles <= reflected_bounds[1]
    )
    if not bool(jnp.any(reflected_mask)):
        raise ValueError("the reflected-angle band contains no detector samples")
    phase_points = _integer("registration_phase_points", registration_phase_points, minimum=3)
    registration_steps = _integer("registration_updates", registration_updates)
    initial_cycle_count = _integer("initial_cycles", initial_cycles, minimum=0)
    accepted_cycle_count = _integer("accepted_cycles", accepted_cycles, minimum=0)
    screening_cycle_count = _integer("screening_cycles", screening_cycles, minimum=0)
    final_cycle_count = _integer("final_cycles", final_cycles, minimum=0)
    updates_per_cycle = _integer("data_updates_per_cycle", data_updates_per_cycle)
    batch_size = _integer("data_batch_size", data_batch_size)
    cg_iterations = _integer("mechanics_cg_iterations", mechanics_cg_iterations)
    scratch_step_count = _integer("scratch_updates", scratch_updates, minimum=0)
    active_iterations = _integer("max_active_iterations", max_active_iterations)
    stride_s = _integer("scratch_stride[0]", scratch_stride[0])
    stride_u = _integer("scratch_stride[1]", scratch_stride[1])
    learning_rate = _finite_positive("data_learning_rate_A", data_learning_rate_A)
    hard_core = _finite_positive("hard_core_weight", hard_core_weight, allow_zero=True)
    mechanics_sigma = _finite_positive("mechanics_sigma_A", mechanics_sigma_A)
    mechanics_step = _finite_positive("mechanics_strength", mechanics_strength, allow_zero=True)
    scratch_rate = _finite_positive("scratch_learning_rate", scratch_learning_rate)
    minimum_improvement = _finite_positive(
        "minimum_selection_improvement", minimum_selection_improvement, allow_zero=True
    )
    if initial_registration is None:
        registration_initial = jnp.zeros(4, dtype=model.reference_positions_3d.dtype)
    else:
        registration_initial = _array("initial_registration", initial_registration, 1)
        if registration_initial.shape != (4,):
            raise ValueError("initial_registration must have shape (4,)")

    template_frequency, convolution_shape = _template_frequency(model)
    extra_latent_y = 0.25 * model.latent_period_A
    empty_removed = jnp.zeros(model.reference_positions_3d.shape[0], dtype=bool)
    empty_extra_active = jnp.zeros(model.max_extra_atoms, dtype=bool)
    empty_values = {
        "host_displacements": jnp.zeros(
            (model.reference_positions_3d.shape[0], 2),
            dtype=model.reference_positions_3d.dtype,
        ),
        "extra_positions_su": jnp.zeros(
            (model.max_extra_atoms, 2), dtype=model.reference_positions_3d.dtype
        ),
    }

    def potential_from_arrays(
        registration: Array,
        values: Mapping[str, Array],
        removed_mask: Array,
        extra_active_mask: Array,
    ) -> Array:
        state = _state_from_values(
            registration,
            values,
            removed_mask,
            extra_active_mask,
            extra_latent_y,
        )
        return _render_with_frequency(
            model, state, template_frequency, convolution_shape
        )

    def predict_arrays(
        registration: Array,
        values: Mapping[str, Array],
        removed_mask: Array,
        extra_active_mask: Array,
        indices: Array,
    ) -> Array:
        potential = potential_from_arrays(
            registration, values, removed_mask, extra_active_mask
        )
        return simulate_glancing_scan_1d(
            potential,
            probes[indices],
            starts[indices],
            length,
            kernel,
            slice_thickness,
            energy,
            rematerialize=True,
        )

    predict_jit = jax.jit(predict_arrays)

    def data_loss(
        registration: Array,
        values: Mapping[str, Array],
        removed_mask: Array,
        extra_active_mask: Array,
        indices: Array,
    ) -> Array:
        predicted = predict_arrays(
            registration, values, removed_mask, extra_active_mask, indices
        )
        loss = _balanced_amplitude_loss(
            predicted,
            measured[indices],
            reflected_mask,
            whole_weight,
        )
        state = _state_from_values(
            registration,
            values,
            removed_mask,
            extra_active_mask,
            extra_latent_y,
        )
        return loss + hard_core * _hard_core_penalty_1d(model, state)

    def evaluate(
        registration: Array,
        values: Mapping[str, Array],
        removed_mask: Array,
        extra_active_mask: Array,
        indices: np.ndarray,
    ) -> float:
        predicted = predict_jit(
            registration,
            values,
            removed_mask,
            extra_active_mask,
            jnp.asarray(indices),
        )
        loss = _balanced_amplitude_loss(
            predicted,
            measured[jnp.asarray(indices)],
            reflected_mask,
            whole_weight,
        )
        return float(np.sqrt(max(float(loss), 0.0)))

    registration_scales = jnp.asarray(
        [model.axial_period_A / 2.0, 1.0, np.deg2rad(1.0), 0.02],
        dtype=model.reference_positions_3d.dtype,
    )
    initial_normalized = registration_initial / registration_scales

    training_jax = jnp.asarray(training)

    def registration_loss(normalized: Array) -> Array:
        return data_loss(
            normalized * registration_scales,
            empty_values,
            empty_removed,
            empty_extra_active,
            training_jax,
        )

    registration_objective_jit = jax.jit(registration_loss)
    registration_value_and_grad = jax.jit(jax.value_and_grad(registration_loss))
    phase_grid = np.linspace(
        -0.5 * model.axial_period_A,
        0.5 * model.axial_period_A,
        phase_points,
        endpoint=False,
    )
    if progress:
        from tqdm.auto import tqdm

        phase_iterator = tqdm(
            phase_grid,
            desc=f"{progress_description}: phase search",
            unit="phase",
            dynamic_ncols=True,
        )
    else:
        phase_iterator = phase_grid
    phase_losses = []
    for phase in phase_iterator:
        candidate = registration_initial.at[0].set(phase)
        phase_losses.append(
            evaluate(
                candidate,
                empty_values,
                empty_removed,
                empty_extra_active,
                training,
            )
        )
    selected_phase = float(phase_grid[int(np.argmin(phase_losses))])
    registration_normalized = initial_normalized.at[0].set(
        selected_phase / registration_scales[0]
    )
    registration_schedule = optax.cosine_decay_schedule(
        5e-2,
        registration_steps,
        alpha=1e-3 / 5e-2,
    )
    registration_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(registration_schedule)
    )
    registration_optimizer_state = registration_optimizer.init(
        registration_normalized
    )
    registration_history = [np.asarray(registration_normalized * registration_scales)]
    registration_loss_history = [
        evaluate(
            registration_normalized * registration_scales,
            empty_values,
            empty_removed,
            empty_extra_active,
            training,
        )
    ]

    def fixed_batch(partition: np.ndarray, index: int) -> np.ndarray:
        offsets = (np.arange(batch_size) + index * batch_size) % len(partition)
        return partition[offsets]

    if progress:
        registration_iterator = tqdm(
            range(registration_steps),
            desc=f"{progress_description}: registration",
            unit="update",
            dynamic_ncols=True,
        )
    else:
        registration_iterator = range(registration_steps)
    for update in registration_iterator:
        value, gradient = registration_value_and_grad(registration_normalized)
        parameter_updates, registration_optimizer_state = registration_optimizer.update(
            gradient, registration_optimizer_state, registration_normalized
        )
        registration_normalized = optax.apply_updates(
            registration_normalized, parameter_updates
        )
        registration_normalized = jnp.clip(registration_normalized, -1.0, 1.0)
        registration_history.append(
            np.asarray(registration_normalized * registration_scales)
        )
        registration_loss_history.append(float(np.sqrt(max(float(value), 0.0))))
    if progress:
        registration_rephase_iterator = tqdm(
            phase_grid,
            desc=f"{progress_description}: registration rephase",
            unit="phase",
            leave=False,
            dynamic_ncols=True,
        )
    else:
        registration_rephase_iterator = phase_grid
    registration_rephase_losses = []
    for phase in registration_rephase_iterator:
        candidate = registration_normalized.at[0].set(
            phase / registration_scales[0]
        )
        registration_rephase_losses.append(
            float(registration_objective_jit(candidate))
        )
    registration_normalized = registration_normalized.at[0].set(
        phase_grid[int(np.argmin(registration_rephase_losses))]
        / registration_scales[0]
    )
    registration_history.append(
        np.asarray(registration_normalized * registration_scales)
    )
    registration_loss_history.append(
        float(np.sqrt(max(min(registration_rephase_losses), 0.0)))
    )
    registration = registration_normalized * registration_scales
    current_state = _empty_state(model, registration)

    event_stages: list[str] = []
    event_updates: list[int] = []
    host_history: list[np.ndarray] = []
    removed_history: list[np.ndarray] = []
    extra_position_history: list[np.ndarray] = []
    extra_active_history: list[np.ndarray] = []
    training_history: list[float] = []
    selection_history: list[float] = []
    scratch_event_indices: list[int] = []
    scratch_history: list[np.ndarray] = []
    proposal_evidence: list[dict[str, Any]] = []
    mechanics_evidence: list[dict[str, Any]] = []
    event_counter = 0
    batch_counter = 0

    def values_from_state(state: CrystalState1D) -> dict[str, Array]:
        return {
            "host_displacements": jnp.asarray(state.host_displacements),
            "extra_positions_su": jnp.asarray(state.extra_positions_3d)[:, [0, 2]],
        }

    def state_from_values(
        values: Mapping[str, Array],
        removed_mask: Array,
        extra_active_mask: Array,
    ) -> CrystalState1D:
        return _state_from_values(
            registration,
            values,
            removed_mask,
            extra_active_mask,
            extra_latent_y,
        )

    def record_event(
        stage: str,
        state: CrystalState1D,
        training_value: float,
        selection_value: float = np.nan,
    ) -> None:
        nonlocal event_counter
        event_stages.append(stage)
        event_updates.append(event_counter)
        event_counter += 1
        host_history.append(np.asarray(state.host_displacements, dtype=np.float32))
        removed_history.append(np.asarray(state.removed_host_mask, dtype=bool))
        extra_position_history.append(
            np.asarray(state.extra_positions_3d, dtype=np.float32)
        )
        extra_active_history.append(np.asarray(state.extra_active_mask, dtype=bool))
        training_history.append(float(training_value))
        selection_history.append(float(selection_value))

    current_values = values_from_state(current_state)
    initial_training_nrmse = evaluate(
        registration,
        current_values,
        current_state.removed_host_mask,
        current_state.extra_active_mask,
        training,
    )
    initial_selection_nrmse = evaluate(
        registration,
        current_values,
        current_state.removed_host_mask,
        current_state.extra_active_mask,
        selection,
    )
    record_event(
        "registered",
        current_state,
        initial_training_nrmse,
        initial_selection_nrmse,
    )

    physical_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(learning_rate)
    )

    def physical_objective(
        values: Mapping[str, Array],
        removed_mask: Array,
        extra_active_mask: Array,
        batch: Array,
    ) -> Array:
        return data_loss(
            registration,
            values,
            removed_mask,
            extra_active_mask,
            batch,
        )

    physical_value_and_grad = jax.jit(jax.value_and_grad(physical_objective))
    host_bounds = 0.5 * jnp.asarray(model.host_mobility)[:, None]
    extra_lower = jnp.asarray(
        [model.axial_coordinates[0], model.slab_bounds_A[0]],
        dtype=model.reference_positions_3d.dtype,
    )
    extra_upper = jnp.asarray(
        [model.axial_coordinates[-1], model.slab_bounds_A[1] + 4.0],
        dtype=model.reference_positions_3d.dtype,
    )

    def physical_update(
        values: Mapping[str, Array],
        optimizer_state: Any,
        removed_mask: Array,
        extra_active_mask: Array,
        batch: Array,
        host_update_mask: Array,
        extra_update_mask: Array,
    ):
        loss, gradients = physical_value_and_grad(
            values, removed_mask, extra_active_mask, batch
        )
        gradients = {
            "host_displacements": gradients["host_displacements"]
            * host_update_mask[:, None],
            "extra_positions_su": gradients["extra_positions_su"]
            * extra_update_mask[:, None],
        }
        updates, optimizer_state = physical_optimizer.update(
            gradients, optimizer_state, values
        )
        updated = optax.apply_updates(values, updates)
        updated = {
            "host_displacements": jnp.clip(
                updated["host_displacements"], -host_bounds, host_bounds
            ),
            "extra_positions_su": jnp.clip(
                updated["extra_positions_su"], extra_lower, extra_upper
            ),
        }
        return updated, optimizer_state, loss

    physical_update_jit = jax.jit(physical_update)

    def run_physical_cycles(
        state: CrystalState1D,
        cycles: int,
        stage_prefix: str,
        *,
        local_center_su: np.ndarray | None = None,
        retain_events: bool,
    ) -> CrystalState1D:
        nonlocal batch_counter
        if cycles == 0:
            return state
        values = values_from_state(state)
        removed_mask = jnp.asarray(state.removed_host_mask)
        extra_active_mask = jnp.asarray(state.extra_active_mask)
        host_update_mask = np.asarray(model.host_mobility) > 0.0
        if local_center_su is not None:
            host_positions, _ = _state_positions(model, state)
            projected = np.asarray(host_positions)[:, [0, 2]]
            host_update_mask &= (
                np.linalg.norm(projected - local_center_su[None, :], axis=1) <= 6.0
            )
        host_update_mask &= ~np.asarray(removed_mask)
        host_update_mask_jax = jnp.asarray(host_update_mask)
        extra_update_mask = np.asarray(extra_active_mask).copy()
        if local_center_su is not None:
            extra_positions_su = np.asarray(values["extra_positions_su"])
            extra_update_mask &= (
                np.linalg.norm(
                    extra_positions_su - local_center_su[None, :], axis=1
                )
                <= 6.0
            )
        extra_update_mask_jax = jnp.asarray(extra_update_mask)
        optimizer_state = physical_optimizer.init(values)
        if progress and retain_events:
            iterator = tqdm(
                range(cycles),
                desc=f"{progress_description}: {stage_prefix}",
                unit="cycle",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            iterator = range(cycles)
        for _ in iterator:
            for _ in range(updates_per_cycle):
                batch = jnp.asarray(fixed_batch(training, batch_counter))
                batch_counter += 1
                values, optimizer_state, loss = physical_update_jit(
                    values,
                    optimizer_state,
                    removed_mask,
                    extra_active_mask,
                    batch,
                    host_update_mask_jax,
                    extra_update_mask_jax,
                )
                if retain_events:
                    record_event(
                        f"{stage_prefix}: data",
                        state_from_values(values, removed_mask, extra_active_mask),
                        float(np.sqrt(max(float(loss), 0.0))),
                    )
            current = state_from_values(values, removed_mask, extra_active_mask)
            current_training = evaluate(
                registration,
                values,
                removed_mask,
                extra_active_mask,
                training,
            )
            current_selection = evaluate(
                registration,
                values,
                removed_mask,
                extra_active_mask,
                selection,
            )
            if retain_events:
                record_event(
                    f"{stage_prefix}: selection",
                    current,
                    current_training,
                    current_selection,
                )
            if mechanics_step > 0.0:
                def evaluate_candidate(candidate_displacement: Array) -> float:
                    candidate_values = {
                        **values,
                        "host_displacements": candidate_displacement,
                    }
                    return evaluate(
                        registration,
                        candidate_values,
                        removed_mask,
                        extra_active_mask,
                        training,
                    )
                accepted_displacement, accepted_training, accepted_strength, trials = (
                    _backtracked_proximal_keating_step(
                        model,
                        values["host_displacements"],
                        removed_mask,
                        host_update_mask_jax,
                        evaluate_candidate,
                        current_training,
                        sigma_A=mechanics_sigma,
                        initial_strength=mechanics_step,
                        cg_iterations=cg_iterations,
                    )
                )
                accepted = accepted_strength > 0.0
                values = {**values, "host_displacements": accepted_displacement}
                mechanics_evidence.append(
                    {
                        "stage": stage_prefix,
                        "accepted_strength": accepted_strength,
                        "trials": trials,
                    }
                )
                mechanics_state = state_from_values(
                    values, removed_mask, extra_active_mask
                )
                if retain_events:
                    record_event(
                        f"{stage_prefix}: mechanics" if accepted else f"{stage_prefix}: mechanics skipped",
                        mechanics_state,
                        accepted_training if accepted else current_training,
                        evaluate(
                            registration,
                            values,
                            removed_mask,
                            extra_active_mask,
                            selection,
                        ),
                    )
        return state_from_values(values, removed_mask, extra_active_mask)

    scratch_mask_flat = np.asarray(model.scratch_mask).reshape(-1)
    scratch_flat_indices = np.flatnonzero(scratch_mask_flat > 0.0).astype(
        np.int32
    )
    scratch_weights = jnp.asarray(scratch_mask_flat[scratch_flat_indices])
    scratch_scale = max(float(np.max(np.abs(np.asarray(model.atom_template)))), 1e-12)

    def run_scratch_residual(
        state: CrystalState1D,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Fit and return one temporary signed residual and matched-filter map."""
        base_potential = potential_from_arrays(
            state.registration,
            values_from_state(state),
            state.removed_host_mask,
            state.extra_active_mask,
        )
        flat_indices = jnp.asarray(scratch_flat_indices)
        residual_values = jnp.zeros(len(scratch_flat_indices), dtype=base_potential.dtype)

        def scratch_loss(values: Array, indices: Array) -> Array:
            residual_flat = jnp.zeros(base_potential.size, dtype=base_potential.dtype)
            residual_flat = residual_flat.at[flat_indices].set(
                values * scratch_weights * scratch_scale
            )
            residual = residual_flat.reshape(base_potential.shape)
            predicted = simulate_glancing_scan_1d(
                base_potential + residual,
                probes[indices],
                starts[indices],
                length,
                kernel,
                slice_thickness,
                energy,
                rematerialize=True,
            )
            return _balanced_amplitude_loss(
                predicted,
                measured[indices],
                reflected_mask,
                whole_weight,
            )

        scratch_value_and_grad = jax.jit(jax.value_and_grad(scratch_loss))
        scratch_optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(scratch_rate)
        )
        optimizer_state = scratch_optimizer.init(residual_values)
        final_loss = np.nan
        if progress:
            iterator = tqdm(
                range(scratch_step_count),
                desc=f"{progress_description}: temporary pixels",
                unit="update",
                leave=False,
                dynamic_ncols=True,
            )
        else:
            iterator = range(scratch_step_count)
        for update in iterator:
            batch = jnp.asarray(fixed_batch(training, update))
            loss, gradient = scratch_value_and_grad(residual_values, batch)
            updates, optimizer_state = scratch_optimizer.update(
                gradient, optimizer_state, residual_values
            )
            residual_values = optax.apply_updates(residual_values, updates)
            residual_values = jnp.clip(residual_values, -1.25, 1.25)
            final_loss = float(np.sqrt(max(float(loss), 0.0)))
        residual_flat = np.zeros(base_potential.size, dtype=np.float32)
        residual_flat[scratch_flat_indices] = (
            np.asarray(residual_values * scratch_weights, dtype=np.float32)
            * scratch_scale
        )
        residual = residual_flat.reshape(base_potential.shape)
        score_map = _matched_filter_residual_1d(model, residual)
        return residual, score_map, final_loss

    def apply_proposal(
        state: CrystalState1D,
        proposal: tuple[str, int, float, np.ndarray],
    ) -> CrystalState1D:
        kind, index, _, _ = proposal
        if kind == "remove":
            return CrystalState1D(
                registration=state.registration,
                host_displacements=state.host_displacements,
                removed_host_mask=jnp.asarray(state.removed_host_mask).at[index].set(True),
                extra_positions_3d=state.extra_positions_3d,
                extra_active_mask=state.extra_active_mask,
            )
        free_slots = np.flatnonzero(~np.asarray(state.extra_active_mask))
        if not len(free_slots):
            raise RuntimeError("no free extra-atom slot remains")
        slot = int(free_slots[0])
        anchor = jnp.asarray(model.insertion_anchors_3d[index])
        return CrystalState1D(
            registration=state.registration,
            host_displacements=state.host_displacements,
            removed_host_mask=state.removed_host_mask,
            extra_positions_3d=jnp.asarray(state.extra_positions_3d).at[slot].set(anchor),
            extra_active_mask=jnp.asarray(state.extra_active_mask).at[slot].set(True),
        )

    current_state = run_physical_cycles(
        current_state,
        initial_cycle_count,
        "initial host",
        retain_events=True,
    )
    current_selection = evaluate(
        current_state.registration,
        values_from_state(current_state),
        current_state.removed_host_mask,
        current_state.extra_active_mask,
        selection,
    )
    termination_reason = "target_reached" if current_selection <= target else "active_search"

    for active_iteration in range(active_iterations):
        if current_selection <= target:
            termination_reason = "target_reached"
            break
        residual, score_map, scratch_training = run_scratch_residual(current_state)
        scratch_event_indices.append(len(event_stages))
        scratch_history.append(residual[::stride_s, ::stride_u].astype(np.float32))
        record_event(
            f"topology {active_iteration + 1}: temporary pixels",
            current_state,
            scratch_training,
            current_selection,
        )
        proposals = _rank_topology_proposals_1d(model, current_state, score_map)
        if not proposals:
            termination_reason = "no_proposal"
            break
        screened: list[tuple[float, CrystalState1D, tuple[str, int, float, np.ndarray]]] = []
        saved_batch_counter = batch_counter
        for proposal in proposals:
            batch_counter = saved_batch_counter
            candidate = apply_proposal(current_state, proposal)
            candidate = run_physical_cycles(
                candidate,
                screening_cycle_count,
                "candidate screen",
                local_center_su=np.asarray(proposal[3]),
                retain_events=False,
            )
            candidate_values = values_from_state(candidate)
            candidate_selection = evaluate(
                candidate.registration,
                candidate_values,
                candidate.removed_host_mask,
                candidate.extra_active_mask,
                selection,
            )
            hard_core_value = float(_hard_core_penalty_1d(model, candidate))
            valid = np.isfinite(candidate_selection) and hard_core_value < 1e-6
            proposal_evidence.append(
                {
                    "iteration": active_iteration + 1,
                    "kind": proposal[0],
                    "index": proposal[1],
                    "scratch_score": proposal[2],
                    "selection_nrmse": candidate_selection,
                    "hard_core_penalty": hard_core_value,
                    "valid": bool(valid),
                }
            )
            if valid:
                screened.append((candidate_selection, candidate, proposal))
        batch_counter = saved_batch_counter
        if not screened:
            termination_reason = "no_valid_proposal"
            break
        screened.sort(key=lambda item: item[0])
        best_selection, best_state, best_proposal = screened[0]
        if current_selection - best_selection < minimum_improvement:
            termination_reason = "no_improving_edit"
            break
        current_state = best_state
        current_selection = best_selection
        current_training = evaluate(
            current_state.registration,
            values_from_state(current_state),
            current_state.removed_host_mask,
            current_state.extra_active_mask,
            training,
        )
        record_event(
            f"topology {active_iteration + 1}: accepted {best_proposal[0]}",
            current_state,
            current_training,
            current_selection,
        )
        current_state = run_physical_cycles(
            current_state,
            accepted_cycle_count,
            f"topology {active_iteration + 1} refinement",
            retain_events=True,
        )
        current_selection = evaluate(
            current_state.registration,
            values_from_state(current_state),
            current_state.removed_host_mask,
            current_state.extra_active_mask,
            selection,
        )
    else:
        if current_selection > target:
            termination_reason = "capacity_or_iteration_limit"

    if current_selection <= target:
        removed_indices = list(np.flatnonzero(np.asarray(current_state.removed_host_mask)))
        active_slots = list(np.flatnonzero(np.asarray(current_state.extra_active_mask)))
        for kind, index in [
            *(('remove', int(index)) for index in removed_indices),
            *(('add', int(index)) for index in active_slots),
        ]:
            if kind == "remove":
                trial = CrystalState1D(
                    registration=current_state.registration,
                    host_displacements=current_state.host_displacements,
                    removed_host_mask=jnp.asarray(current_state.removed_host_mask).at[index].set(False),
                    extra_positions_3d=current_state.extra_positions_3d,
                    extra_active_mask=current_state.extra_active_mask,
                )
                center = np.asarray(_state_positions(model, current_state)[0])[index, [0, 2]]
            else:
                trial = CrystalState1D(
                    registration=current_state.registration,
                    host_displacements=current_state.host_displacements,
                    removed_host_mask=current_state.removed_host_mask,
                    extra_positions_3d=current_state.extra_positions_3d,
                    extra_active_mask=jnp.asarray(current_state.extra_active_mask).at[index].set(False),
                )
                center = np.asarray(current_state.extra_positions_3d)[index, [0, 2]]
            saved_batch_counter = batch_counter
            trial = run_physical_cycles(
                trial,
                screening_cycle_count,
                "prune screen",
                local_center_su=center,
                retain_events=False,
            )
            batch_counter = saved_batch_counter
            trial_selection = evaluate(
                trial.registration,
                values_from_state(trial),
                trial.removed_host_mask,
                trial.extra_active_mask,
                selection,
            )
            if trial_selection <= target:
                current_state = trial
                current_selection = trial_selection
                record_event(
                    f"pruned {kind}",
                    current_state,
                    evaluate(
                        current_state.registration,
                        values_from_state(current_state),
                        current_state.removed_host_mask,
                        current_state.extra_active_mask,
                        training,
                    ),
                    current_selection,
                )

    current_state = run_physical_cycles(
        current_state,
        final_cycle_count,
        "final polish",
        retain_events=True,
    )
    final_values = values_from_state(current_state)
    final_training = evaluate(
        current_state.registration,
        final_values,
        current_state.removed_host_mask,
        current_state.extra_active_mask,
        training,
    )
    final_selection = evaluate(
        current_state.registration,
        final_values,
        current_state.removed_host_mask,
        current_state.extra_active_mask,
        selection,
    )
    final_audit = evaluate(
        current_state.registration,
        final_values,
        current_state.removed_host_mask,
        current_state.extra_active_mask,
        audit,
    )
    if final_selection <= target:
        termination_reason = "target_reached"
    record_event(
        "final audit",
        current_state,
        final_training,
        final_selection,
    )
    final_potential = potential_from_arrays(
        current_state.registration,
        final_values,
        current_state.removed_host_mask,
        current_state.extra_active_mask,
    )
    all_indices = jnp.arange(n_scan, dtype=jnp.int32)
    final_predictions = predict_jit(
        current_state.registration,
        final_values,
        current_state.removed_host_mask,
        current_state.extra_active_mask,
        all_indices,
    )
    if scratch_history:
        scratch_residual_array = jnp.asarray(np.stack(scratch_history))
    else:
        scratch_residual_array = jnp.zeros(
            (
                0,
                len(np.arange(0, model.axial_coordinates.shape[0], stride_s)),
                len(np.arange(0, model.transverse_coordinates.shape[0], stride_u)),
            ),
            dtype=jnp.float32,
        )
    metadata = {
        **dict(model.metadata),
        "n_host_sites": int(model.reference_positions_3d.shape[0]),
        "n_mobile_host_sites": int(np.count_nonzero(np.asarray(model.host_mobility) > 0.0)),
        "n_full_mobility_host_sites": int(np.count_nonzero(np.asarray(model.full_mobility_mask))),
        "n_insertion_anchors": int(model.insertion_anchors_3d.shape[0]),
        "n_bonds": int(model.bond_indices.shape[0]),
        "n_angles": int(model.angle_indices.shape[0]),
        "data_updates_per_cycle": updates_per_cycle,
        "registration_updates": registration_steps,
        "mechanics_sigma_A": mechanics_sigma,
        "mechanics_strength": mechanics_step,
        "scratch_updates": scratch_step_count,
        "pixel_residual_retained": False,
        "proposal_evidence": proposal_evidence,
        "mechanics_evidence": mechanics_evidence,
        "elapsed_seconds": perf_counter() - reconstruction_start_time,
    }
    return CrystalReconstruction1D(
        state=current_state,
        potential=final_potential,
        predicted_intensities=final_predictions,
        measured_intensities=measured,
        detector_angles_mrad=detector_angles,
        training_indices=jnp.asarray(training),
        selection_indices=jnp.asarray(selection),
        audit_indices=jnp.asarray(audit),
        target_nrmse=jnp.asarray(target),
        training_nrmse=jnp.asarray(final_training),
        selection_nrmse=jnp.asarray(final_selection),
        audit_nrmse=jnp.asarray(final_audit),
        termination_reason=termination_reason,
        registration_history=jnp.asarray(np.stack(registration_history)),
        registration_loss_history=jnp.asarray(registration_loss_history),
        event_stages=tuple(event_stages),
        event_updates=jnp.asarray(event_updates),
        host_displacement_history=jnp.asarray(np.stack(host_history)),
        removed_host_history=jnp.asarray(np.stack(removed_history)),
        extra_position_history=jnp.asarray(np.stack(extra_position_history)),
        extra_active_history=jnp.asarray(np.stack(extra_active_history)),
        training_nrmse_history=jnp.asarray(training_history),
        selection_nrmse_history=jnp.asarray(selection_history),
        scratch_event_indices=jnp.asarray(scratch_event_indices, dtype=jnp.int32),
        scratch_residual_history=scratch_residual_array,
        metadata=metadata,
    )
