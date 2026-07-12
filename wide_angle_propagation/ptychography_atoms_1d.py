"""Free-position known-species atoms for the 1D-transverse ptychography model.

The reconstruction knows the atomic species and the illuminated search region,
but it does not receive a lattice, particle shape, atom count, or truth sites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from time import perf_counter
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpy as np

from .propagation_methods import energy2wavelength
from .ptychography_1d import normalized_amplitude_loss_1d, simulate_glancing_scan_1d


__all__ = [
    "FreeAtomModel1D",
    "FreeAtomReconstruction1D",
    "free_atom_cohesion_1d",
    "free_atom_repulsion_1d",
    "make_atom_template_1d",
    "make_si_atom_template_1d",
    "reconstruct_free_atoms_1d",
    "render_free_atoms_1d",
    "render_species_mixture_atoms_1d",
    "uniform_atom_candidates_1d",
]


Array = Any


@dataclass(frozen=True)
class FreeAtomModel1D:
    """A known atomic template with freely movable candidate positions.

    ``candidate_bounds`` has shape ``(2, 2)`` and stores ``[[s_min, s_max],
    [u_min, u_max]]`` in Angstrom. The template is centred and sampled like the
    specimen grid. ``fixed_potential`` optionally retains known material outside
    the free-atom region. Setting ``maximum_displacement_A`` bounds each atom
    around its uniform seed and enables the memory-bounded local renderer.
    """

    axial_coordinates: Array
    transverse_coordinates: Array
    atom_template: Array
    candidate_bounds: Array
    initial_positions: Array
    fixed_potential: Array | None = None
    maximum_displacement_A: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FreeAtomReconstruction1D:
    """Best free-position known-species estimate and optimization history."""

    positions: Array
    occupancies: Array
    initial_positions: Array
    initial_occupancies: Array
    potential: Array
    predicted_intensities: Array
    measured_intensities: Array
    update_history: Array
    elapsed_time_history: Array
    training_loss_history: Array
    validation_loss_history: Array
    snapshot_updates: Array
    position_history: Array
    occupancy_history: Array
    best_update: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _as_array(name: str, value: Any, ndim: int) -> Array:
    array = jnp.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {array.shape}")
    return array


def _concrete_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, jax.core.Tracer):
        return None
    try:
        return np.asarray(value)
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        return None


def _validate_uniform_spacing(name: str, coordinates: Array) -> None:
    concrete = _concrete_numpy(coordinates)
    if concrete is None:
        return
    host = np.asarray(concrete, dtype=float)
    if host.size < 2 or not np.all(np.isfinite(host)):
        raise ValueError(f"{name} must contain at least two finite values")
    differences = np.diff(host)
    if np.any(differences <= 0.0) or not np.allclose(
        differences, differences[0], rtol=1e-6, atol=1e-12
    ):
        raise ValueError(f"{name} must be uniformly increasing")


def _model_arrays(
    model: FreeAtomModel1D,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    coordinates_s = _as_array("model.axial_coordinates", model.axial_coordinates, 1)
    coordinates_u = _as_array(
        "model.transverse_coordinates", model.transverse_coordinates, 1
    )
    template = _as_array("model.atom_template", model.atom_template, 2)
    bounds = _as_array("model.candidate_bounds", model.candidate_bounds, 2)
    initial_positions = _as_array(
        "model.initial_positions", model.initial_positions, 2
    )
    if bounds.shape != (2, 2):
        raise ValueError("model.candidate_bounds must have shape (2, 2)")
    if initial_positions.shape[1:] != (2,) or initial_positions.shape[0] == 0:
        raise ValueError("model.initial_positions must have shape (n_candidate, 2)")
    if min(template.shape) < 3:
        raise ValueError("model.atom_template must have at least three samples per axis")
    _validate_uniform_spacing("model.axial_coordinates", coordinates_s)
    _validate_uniform_spacing("model.transverse_coordinates", coordinates_u)
    ds = coordinates_s[1] - coordinates_s[0]
    du = coordinates_u[1] - coordinates_u[0]
    template_host = _concrete_numpy(template)
    bounds_host = _concrete_numpy(bounds)
    positions_host = _concrete_numpy(initial_positions)
    coordinates_s_host = _concrete_numpy(coordinates_s)
    coordinates_u_host = _concrete_numpy(coordinates_u)
    concrete_values = (
        template_host,
        bounds_host,
        positions_host,
        coordinates_s_host,
        coordinates_u_host,
    )
    if all(value is not None for value in concrete_values):
        assert template_host is not None
        assert bounds_host is not None
        assert positions_host is not None
        assert coordinates_s_host is not None
        assert coordinates_u_host is not None
        bounds_host = np.asarray(bounds_host, dtype=float)
        positions_host = np.asarray(positions_host, dtype=float)
        if (
            np.iscomplexobj(template_host)
            or not np.all(np.isfinite(template_host))
            or not np.all(np.isfinite(bounds_host))
            or not np.all(np.isfinite(positions_host))
        ):
            raise ValueError("free-atom model arrays must be finite and real")
        if np.any(bounds_host[:, 1] <= bounds_host[:, 0]):
            raise ValueError("each candidate bound must have positive width")
        grid_limits = np.asarray(
            [
                [coordinates_s_host[0], coordinates_s_host[-1]],
                [coordinates_u_host[0], coordinates_u_host[-1]],
            ],
            dtype=float,
        )
        if np.any(bounds_host[:, 0] < grid_limits[:, 0]) or np.any(
            bounds_host[:, 1] > grid_limits[:, 1]
        ):
            raise ValueError("candidate bounds must lie inside the specimen grid")
        if np.any(positions_host < bounds_host[:, 0]) or np.any(
            positions_host > bounds_host[:, 1]
        ):
            raise ValueError("initial candidate positions must lie inside the bounds")
    if model.maximum_displacement_A is not None:
        maximum_displacement = np.asarray(model.maximum_displacement_A)
        if (
            maximum_displacement.ndim != 0
            or not np.isfinite(maximum_displacement)
            or float(maximum_displacement) <= 0.0
        ):
            raise ValueError("model.maximum_displacement_A must be positive or None")
    return (
        coordinates_s,
        coordinates_u,
        template,
        bounds,
        initial_positions,
        ds,
        du,
    )


def _fixed_potential(model: FreeAtomModel1D, shape: tuple[int, int], dtype) -> Array:
    if model.fixed_potential is None:
        return jnp.zeros(shape, dtype=dtype)
    fixed = _as_array("model.fixed_potential", model.fixed_potential, 2)
    if fixed.shape != shape:
        raise ValueError("model.fixed_potential must match the specimen grid")
    fixed_host = _concrete_numpy(fixed)
    if fixed_host is not None and (
        np.iscomplexobj(fixed_host) or not np.all(np.isfinite(fixed_host))
    ):
        raise ValueError("model.fixed_potential must be finite and real")
    return fixed.astype(dtype)


def _candidate_position_limits(
    model: FreeAtomModel1D,
    bounds: Array,
    initial_positions: Array,
) -> tuple[Array, Array]:
    lower = jnp.broadcast_to(bounds[:, 0], initial_positions.shape)
    upper = jnp.broadcast_to(bounds[:, 1], initial_positions.shape)
    if model.maximum_displacement_A is not None:
        displacement = jnp.asarray(
            model.maximum_displacement_A, dtype=initial_positions.dtype
        )
        lower = jnp.maximum(lower, initial_positions - displacement)
        upper = jnp.minimum(upper, initial_positions + displacement)
    return lower, upper


def uniform_atom_candidates_1d(
    candidate_bounds: Any,
    shape: tuple[int, int] = (6, 4),
) -> Array:
    """Place a uniform grid of free candidates inside a rectangular region."""
    bounds = np.asarray(candidate_bounds, dtype=float)
    if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError("candidate_bounds must be a finite array with shape (2, 2)")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("each candidate bound must have positive width")
    if len(shape) != 2:
        raise ValueError("shape must contain axial and transverse counts")
    n_s, n_u = (operator.index(value) for value in shape)
    if n_s < 1 or n_u < 1:
        raise ValueError("candidate-grid counts must be positive")
    step = (bounds[:, 1] - bounds[:, 0]) / np.asarray([n_s, n_u])
    positions_s = bounds[0, 0] + (np.arange(n_s) + 0.5) * step[0]
    positions_u = bounds[1, 0] + (np.arange(n_u) + 0.5) * step[1]
    grid_s, grid_u = np.meshgrid(positions_s, positions_u, indexing="ij")
    return jnp.asarray(np.stack([grid_s.ravel(), grid_u.ravel()], axis=-1))


def make_atom_template_1d(
    element: str,
    axial_sampling: float,
    transverse_sampling: float,
    *,
    cutoff_A: float = 4.0,
    projection_width_A: float = 5.43,
) -> Array:
    """Generate a centred finite-projection Lobato template for one element."""
    try:
        import abtem
        from ase import Atoms
    except ImportError as exc:  # pragma: no cover - optional scientific dependency
        raise ImportError("make_atom_template_1d requires abTEM and ASE") from exc
    if not isinstance(element, str) or not element.strip():
        raise ValueError("element must be a non-empty chemical symbol")
    ds = float(axial_sampling)
    du = float(transverse_sampling)
    cutoff = float(cutoff_A)
    width = float(projection_width_A)
    if not all(np.isfinite(value) and value > 0.0 for value in (ds, du, cutoff, width)):
        raise ValueError("sampling, cutoff, and projection width must be positive")
    half_s = int(np.ceil(cutoff / ds))
    half_u = int(np.ceil(cutoff / du))
    n_s = 2 * half_s + 1
    n_u = 2 * half_u + 1
    cell = np.diag([n_u * du, width, n_s * ds])
    atom = Atoms(
        element.strip(),
        positions=[[half_u * du, 0.5 * width, half_s * ds]],
        cell=cell,
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


def make_si_atom_template_1d(
    axial_sampling: float,
    transverse_sampling: float,
    *,
    cutoff_A: float = 4.0,
    projection_width_A: float = 5.43,
) -> Array:
    """Generate a silicon template (backward-compatible convenience wrapper)."""
    return make_atom_template_1d(
        "Si",
        axial_sampling,
        transverse_sampling,
        cutoff_A=cutoff_A,
        projection_width_A=projection_width_A,
    )


def render_free_atoms_1d(
    model: FreeAtomModel1D,
    positions: Any,
    occupancies: Any,
) -> Array:
    """Render continuously positioned atoms on the specimen potential grid."""
    coordinates_s, coordinates_u, template, _, initial_positions, ds, du = (
        _model_arrays(model)
    )
    atom_positions = _as_array("positions", positions, 2)
    atom_occupancies = _as_array("occupancies", occupancies, 1)
    if (
        atom_positions.shape[1:] != (2,)
        or atom_positions.shape[0] != atom_occupancies.shape[0]
    ):
        raise ValueError("positions and occupancies must have shapes (n, 2) and (n,)")
    render_dtype = jnp.result_type(
        template.dtype, atom_positions.dtype, atom_occupancies.dtype
    )
    template = template.astype(render_dtype)
    output_shape = (coordinates_s.shape[0], coordinates_u.shape[0])
    fixed = _fixed_potential(model, output_shape, render_dtype)
    center_s = (template.shape[0] - 1) / 2.0
    center_u = (template.shape[1] - 1) / 2.0

    if model.maximum_displacement_A is None:
        grid_s, grid_u = jnp.meshgrid(coordinates_s, coordinates_u, indexing="ij")

        def render_one(position: Array) -> Array:
            sample_s = center_s + (grid_s - position[0]) / ds
            sample_u = center_u + (grid_u - position[1]) / du
            return map_coordinates(
                template,
                jnp.stack([sample_s, sample_u]),
                order=1,
                mode="constant",
                cval=0.0,
            )

        rendered = jax.vmap(render_one)(atom_positions)
        return fixed + jnp.sum(atom_occupancies[:, None, None] * rendered, axis=0)

    if atom_positions.shape[0] != initial_positions.shape[0]:
        raise ValueError(
            "local rendering requires one position per model.initial_positions row"
        )
    sampling_s_host = np.asarray(model.axial_coordinates, dtype=float)
    sampling_u_host = np.asarray(model.transverse_coordinates, dtype=float)
    ds_host = float(sampling_s_host[1] - sampling_s_host[0])
    du_host = float(sampling_u_host[1] - sampling_u_host[0])
    displacement = float(model.maximum_displacement_A)
    padding_s = int(np.ceil(displacement / ds_host)) + 1
    padding_u = int(np.ceil(displacement / du_host)) + 1
    output_half_s = (template.shape[0] - 1) // 2 + padding_s
    output_half_u = (template.shape[1] - 1) // 2 + padding_u
    offsets_s = jnp.arange(-output_half_s, output_half_s + 1, dtype=jnp.int32)
    offsets_u = jnp.arange(-output_half_u, output_half_u + 1, dtype=jnp.int32)
    anchor_s = jnp.rint(
        (initial_positions[:, 0] - coordinates_s[0]) / ds
    ).astype(jnp.int32)
    anchor_u = jnp.rint(
        (initial_positions[:, 1] - coordinates_u[0]) / du
    ).astype(jnp.int32)

    def render_local(position: Array, centre_s: Array, centre_u: Array):
        rows = centre_s + offsets_s[:, None]
        columns = centre_u + offsets_u[None, :]
        valid = (
            (rows >= 0)
            & (rows < output_shape[0])
            & (columns >= 0)
            & (columns < output_shape[1])
        )
        clipped_rows = jnp.clip(rows, 0, output_shape[0] - 1)
        clipped_columns = jnp.clip(columns, 0, output_shape[1] - 1)
        physical_s = coordinates_s[clipped_rows]
        physical_u = coordinates_u[clipped_columns]
        sample_s = center_s + (physical_s - position[0]) / ds
        sample_u = center_u + (physical_u - position[1]) / du
        values = map_coordinates(
            template,
            jnp.stack(
                [
                    jnp.broadcast_to(sample_s, valid.shape),
                    jnp.broadcast_to(sample_u, valid.shape),
                ]
            ),
            order=1,
            mode="constant",
            cval=0.0,
        )
        flat_indices = clipped_rows * output_shape[1] + clipped_columns
        return flat_indices, jnp.where(valid, values, 0.0)

    flat_indices, local_values = jax.vmap(render_local)(
        atom_positions, anchor_s, anchor_u
    )
    flat = fixed.reshape(-1)
    flat = flat.at[flat_indices.reshape(-1)].add(
        (atom_occupancies[:, None, None] * local_values).reshape(-1)
    )
    return flat.reshape(output_shape)


def render_species_mixture_atoms_1d(
    model: FreeAtomModel1D,
    atom_templates: Any,
    positions: Any,
    occupancies: Any,
    species_probabilities: Any,
) -> Array:
    """Render atoms as differentiable mixtures of fixed species templates.

    ``atom_templates`` has shape ``(n_species, template_s, template_u)`` and
    ``species_probabilities`` has shape ``(n_atoms, n_species)``. Each
    probability row must be non-negative and sum to one. This models a soft
    categorical species choice; it does not interpret the mixture as a
    fractional atomic number.
    """
    templates = _as_array("atom_templates", atom_templates, 3)
    atom_positions = _as_array("positions", positions, 2)
    atom_occupancies = _as_array("occupancies", occupancies, 1)
    probabilities = _as_array("species_probabilities", species_probabilities, 2)
    if templates.shape[1:] != jnp.asarray(model.atom_template).shape:
        raise ValueError("all species templates must match model.atom_template shape")
    if atom_positions.shape != (atom_occupancies.shape[0], 2):
        raise ValueError("positions and occupancies must have shapes (n, 2) and (n,)")
    if probabilities.shape != (atom_positions.shape[0], templates.shape[0]):
        raise ValueError("species_probabilities must have shape (n_atoms, n_species)")
    probabilities_host = _concrete_numpy(probabilities)
    if probabilities_host is not None and (
        np.any(probabilities_host < 0.0)
        or not np.allclose(np.sum(probabilities_host, axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("species probabilities must be non-negative and sum to one")

    species_potentials = []
    for species_index in range(templates.shape[0]):
        species_model = FreeAtomModel1D(
            model.axial_coordinates,
            model.transverse_coordinates,
            templates[species_index],
            model.candidate_bounds,
            model.initial_positions,
            fixed_potential=None,
            maximum_displacement_A=model.maximum_displacement_A,
            metadata=model.metadata,
        )
        species_potentials.append(
            render_free_atoms_1d(
                species_model,
                atom_positions,
                atom_occupancies * probabilities[:, species_index],
            )
        )
    mixture = jnp.sum(jnp.stack(species_potentials), axis=0)
    return mixture + _fixed_potential(model, mixture.shape, mixture.dtype)


def free_atom_repulsion_1d(
    positions: Any,
    occupancies: Any,
    *,
    minimum_distance_A: float = 1.8,
    transition_A: float = 0.15,
) -> Array:
    """Occupancy-weighted penalty for atom pairs closer than the hard core."""
    atom_positions = _as_array("positions", positions, 2)
    atom_occupancies = _as_array("occupancies", occupancies, 1)
    n_atoms = atom_positions.shape[0]
    delta = atom_positions[:, None, :] - atom_positions[None, :, :]
    distance = jnp.sqrt(jnp.sum(delta**2, axis=-1) + 1e-12)
    overlap = jax.nn.relu((minimum_distance_A - distance) / transition_A) ** 2
    pair_weight = atom_occupancies[:, None] * atom_occupancies[None, :]
    upper = jnp.triu(jnp.ones((n_atoms, n_atoms), dtype=bool), k=1)
    return jnp.sum(jnp.where(upper, pair_weight * overlap, 0.0)) / max(n_atoms, 1)


def free_atom_cohesion_1d(
    positions: Any,
    occupancies: Any,
    *,
    preferred_distance_A: float = 2.35,
    cutoff_A: float = 4.0,
    softening_A: float = 0.25,
) -> Array:
    """Weak softened Lennard-Jones-like coupling for an optional ablation."""
    atom_positions = _as_array("positions", positions, 2)
    atom_occupancies = _as_array("occupancies", occupancies, 1)
    n_atoms = atom_positions.shape[0]
    delta = atom_positions[:, None, :] - atom_positions[None, :, :]
    distance = jnp.sqrt(jnp.sum(delta**2, axis=-1) + 1e-12)
    effective_distance = jnp.sqrt(distance**2 + softening_A**2)
    preferred_effective = jnp.sqrt(preferred_distance_A**2 + softening_A**2)
    sigma = preferred_effective / 2.0 ** (1.0 / 6.0)
    ratio = sigma / effective_distance
    lennard_jones = 4.0 * (ratio**12 - ratio**6)
    cutoff_effective = jnp.sqrt(cutoff_A**2 + softening_A**2)
    cutoff_ratio = sigma / cutoff_effective
    shifted = lennard_jones - 4.0 * (cutoff_ratio**12 - cutoff_ratio**6)
    taper_width = min(0.5, cutoff_A / 2.0)
    taper_x = jnp.clip((cutoff_A - distance) / taper_width, 0.0, 1.0)
    taper = taper_x**2 * (3.0 - 2.0 * taper_x)
    pair_weight = atom_occupancies[:, None] * atom_occupancies[None, :]
    upper = jnp.triu(jnp.ones((n_atoms, n_atoms), dtype=bool), k=1)
    return jnp.sum(jnp.where(upper, pair_weight * taper * shifted, 0.0)) / max(
        n_atoms, 1
    )


def reconstruct_free_atoms_1d(
    model: FreeAtomModel1D,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    measured_intensities: Any,
    *,
    validation_indices: Sequence[int] = (),
    initial_occupancy: float = 0.1,
    mass_weight: float = 1e-3,
    repulsion_weight: float = 1e-2,
    cohesion_weight: float = 0.0,
    cohesion_start_update: int = 600,
    occupancy_learning_rate: float = 2e-2,
    position_learning_rate_A: float = 1e-2,
    updates: int = 1000,
    occupancy_only_updates: int = 200,
    minibatch_size: int = 5,
    validation_interval: int = 20,
    gradient_clip: float = 1.0,
    seed: int = 0,
    progress: bool = False,
    progress_description: str = "free-atom reconstruction",
) -> FreeAtomReconstruction1D:
    """Fit positions and occupancies of a fixed known-species template."""
    try:
        import optax
    except ImportError as exc:  # pragma: no cover - optional optimizer dependency
        raise ImportError("reconstruct_free_atoms_1d requires Optax") from exc
    coordinates_s, coordinates_u, _, bounds, initial_positions, _, du = _model_arrays(
        model
    )
    position_lower, position_upper = _candidate_position_limits(
        model, bounds, initial_positions
    )
    probe = jnp.asarray(input_probe)
    kernel = _as_array("propagation_kernel", propagation_kernel, 1)
    starts = _as_array("window_starts", window_starts, 1)
    measured = _as_array("measured_intensities", measured_intensities, 2)
    length = operator.index(window_length)
    n_updates = operator.index(updates)
    frozen_updates = operator.index(occupancy_only_updates)
    interval = operator.index(validation_interval)
    batch_size = operator.index(minibatch_size)
    seed_value = operator.index(seed)
    n_scan = starts.shape[0]
    n_u = coordinates_u.shape[0]
    if length < 1 or n_updates < 1 or interval < 1 or batch_size < 1:
        raise ValueError("length, updates, minibatch size, and interval must be positive")
    if frozen_updates < 0 or frozen_updates > n_updates:
        raise ValueError("occupancy_only_updates must lie between zero and updates")
    if probe.ndim not in (1, 2) or probe.shape[-1] != n_u:
        raise ValueError("input_probe must end with the transverse grid length")
    if probe.ndim == 2 and probe.shape[0] != n_scan:
        raise ValueError("a scan-dependent probe must have one row per scan")
    if kernel.shape != (n_u,) or measured.shape != (n_scan, n_u):
        raise ValueError("kernel or measured-intensity shape does not match the scan")
    if np.any(np.asarray(starts) < 0) or np.any(
        np.asarray(starts) + length > coordinates_s.shape[0]
    ):
        raise ValueError("window_starts must select valid specimen windows")
    for name, value in (
        ("initial_occupancy", initial_occupancy),
        ("mass_weight", mass_weight),
        ("repulsion_weight", repulsion_weight),
        ("cohesion_weight", cohesion_weight),
        ("occupancy_learning_rate", occupancy_learning_rate),
        ("position_learning_rate_A", position_learning_rate_A),
        ("gradient_clip", gradient_clip),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if initial_occupancy > 1.0:
        raise ValueError("initial_occupancy must not exceed one")
    validation = np.asarray(validation_indices)
    if validation.ndim != 1 or (
        validation.size and not np.issubdtype(validation.dtype, np.integer)
    ):
        raise TypeError("validation_indices must be a one-dimensional integer sequence")
    validation = validation.astype(np.int32, copy=False)
    if np.unique(validation).size != validation.size or np.any(validation < 0) or np.any(
        validation >= n_scan
    ):
        raise ValueError("validation_indices must be unique valid scan indices")
    training = np.setdiff1d(np.arange(n_scan, dtype=np.int32), validation)
    if training.size == 0:
        raise ValueError("at least one scan must remain for training")
    probe_rows = jnp.broadcast_to(probe, (n_scan, n_u)) if probe.ndim == 1 else probe
    parameters = {
        "positions": initial_positions,
        "occupancies": jnp.full(
            (initial_positions.shape[0],), initial_occupancy, dtype=initial_positions.dtype
        ),
    }
    initial_parameters = parameters

    def objective(
        values: Mapping[str, Array],
        batch_indices: Array,
        current_cohesion_weight: Array,
    ) -> Array:
        potential = render_free_atoms_1d(
            model, values["positions"], values["occupancies"]
        )
        prediction = simulate_glancing_scan_1d(
            potential,
            probe_rows[batch_indices],
            starts[batch_indices],
            length,
            kernel,
            slice_thickness,
            energy,
        )
        data_loss = jnp.sqrt(
            normalized_amplitude_loss_1d(prediction, measured[batch_indices])
            + 1e-16
        )
        mass = jnp.mean(values["occupancies"])
        repulsion = free_atom_repulsion_1d(
            values["positions"], values["occupancies"]
        )
        cohesion = free_atom_cohesion_1d(
            values["positions"], values["occupancies"]
        )
        return (
            data_loss
            + mass_weight * mass
            + repulsion_weight * repulsion
            + current_cohesion_weight * cohesion
        )

    occupancy_schedule = optax.cosine_decay_schedule(
        occupancy_learning_rate, n_updates, alpha=0.1
    )
    position_schedule = optax.cosine_decay_schedule(
        position_learning_rate_A, n_updates, alpha=0.1
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.multi_transform(
            {
                "positions": optax.adam(position_schedule),
                "occupancies": optax.adam(occupancy_schedule),
            },
            {"positions": "positions", "occupancies": "occupancies"},
        ),
    )
    optimizer_state = optimizer.init(parameters)

    @jax.jit
    def update_step(values, state, batch_indices, current_cohesion, move_positions):
        loss, gradients = jax.value_and_grad(objective)(
            values, batch_indices, current_cohesion
        )
        gradients = {
            "positions": jnp.where(move_positions, gradients["positions"], 0.0),
            "occupancies": gradients["occupancies"],
        }
        parameter_updates, state = optimizer.update(gradients, state, values)
        values = optax.apply_updates(values, parameter_updates)
        values = {
            "positions": jnp.clip(
                values["positions"], position_lower, position_upper
            ),
            "occupancies": jnp.clip(values["occupancies"], 0.0, 1.0),
        }
        return values, state, loss

    @jax.jit
    def predict(values, indices):
        return simulate_glancing_scan_1d(
            render_free_atoms_1d(model, values["positions"], values["occupancies"]),
            probe_rows[indices],
            starts[indices],
            length,
            kernel,
            slice_thickness,
            energy,
        )

    def evaluate(values, indices: np.ndarray) -> float:
        predicted = predict(values, jnp.asarray(indices))
        return float(
            jnp.sqrt(
                normalized_amplitude_loss_1d(predicted, measured[indices]) + 1e-16
            )
        )

    if not isinstance(progress, (bool, np.bool_)) or not isinstance(
        progress_description, str
    ):
        raise TypeError("progress must be boolean and its description must be text")
    iterator = range(1, n_updates + 1)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc=progress_description, unit="update", dynamic_ncols=True)
    rng = np.random.default_rng(seed_value)
    start_time = perf_counter()
    update_history: list[int] = []
    elapsed_history: list[float] = []
    training_history: list[float] = []
    validation_history: list[float] = []
    position_history: list[np.ndarray] = []
    occupancy_history: list[np.ndarray] = []
    best_values = parameters
    best_update = 0
    best_metric = np.inf

    def record(update: int, values: Mapping[str, Array]) -> None:
        nonlocal best_values, best_update, best_metric
        training_loss = evaluate(values, training)
        validation_loss = evaluate(values, validation) if validation.size else np.nan
        metric = validation_loss if validation.size else training_loss
        update_history.append(update)
        elapsed_history.append(perf_counter() - start_time)
        training_history.append(training_loss)
        validation_history.append(validation_loss)
        position_history.append(np.asarray(values["positions"]))
        occupancy_history.append(np.asarray(values["occupancies"]))
        if metric < best_metric:
            best_metric = metric
            best_update = update
            best_values = {key: value for key, value in values.items()}

    record(0, parameters)
    for update in iterator:
        chosen = rng.choice(training, size=min(batch_size, training.size), replace=False)
        if cohesion_weight > 0.0 and update >= cohesion_start_update:
            denominator = max(n_updates - cohesion_start_update, 1)
            current_cohesion = cohesion_weight * min(
                (update - cohesion_start_update) / denominator, 1.0
            )
        else:
            current_cohesion = 0.0
        parameters, optimizer_state, _ = update_step(
            parameters,
            optimizer_state,
            jnp.asarray(chosen),
            jnp.asarray(current_cohesion),
            jnp.asarray(update > frozen_updates),
        )
        if update % interval == 0 or update == n_updates:
            record(update, parameters)

    best_potential = render_free_atoms_1d(
        model, best_values["positions"], best_values["occupancies"]
    )
    all_indices = jnp.arange(n_scan)
    predicted = predict(best_values, all_indices)
    frequencies = jnp.fft.fftshift(jnp.fft.fftfreq(n_u, du))
    detector_angles = 1e3 * jnp.arcsin(
        jnp.clip(energy2wavelength(energy) * frequencies, -1.0, 1.0)
    )
    metadata = {
        **dict(model.metadata),
        "n_candidates": int(initial_positions.shape[0]),
        "n_specimen_parameters": int(3 * initial_positions.shape[0]),
        "training_indices": training.tolist(),
        "validation_indices": validation.tolist(),
        "mass_weight": mass_weight,
        "repulsion_weight": repulsion_weight,
        "cohesion_weight": cohesion_weight,
        "uses_fixed_potential": model.fixed_potential is not None,
        "maximum_displacement_A": model.maximum_displacement_A,
        "detector_angles_mrad": np.asarray(detector_angles).tolist(),
    }
    return FreeAtomReconstruction1D(
        positions=best_values["positions"],
        occupancies=best_values["occupancies"],
        initial_positions=initial_parameters["positions"],
        initial_occupancies=initial_parameters["occupancies"],
        potential=best_potential,
        predicted_intensities=predicted,
        measured_intensities=measured,
        update_history=jnp.asarray(update_history),
        elapsed_time_history=jnp.asarray(elapsed_history),
        training_loss_history=jnp.asarray(training_history),
        validation_loss_history=jnp.asarray(validation_history),
        snapshot_updates=jnp.asarray(update_history),
        position_history=jnp.asarray(position_history),
        occupancy_history=jnp.asarray(occupancy_history),
        best_update=best_update,
        metadata=metadata,
    )
