"""Four-parameter crystalline-host registration for glancing ptychography.

The diffraction model remains two-dimensional in ``(s, u)`` while host sites
retain latent three-dimensional coordinates ``(s, y, u)``.  Registration fits
only axial phase, surface-normal offset, in-plane rotation, and axial strain.
There are no per-site, occupancy, species, or off-lattice variables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .ptychography_1d import simulate_glancing_scan_1d
from .propagation_methods import interaction_constant


__all__ = [
    "CrystallineHostModel1D",
    "CrystallineRegistrationParameters1D",
    "CrystallineRegistrationResult1D",
    "make_crystalline_host_model_1d",
    "register_crystalline_host_1d",
    "render_crystalline_host_1d",
    "transform_crystalline_host_1d",
]


Array = Any


@dataclass(frozen=True)
class CrystallineHostModel1D:
    """A single-species crystalline host on a two-dimensional potential grid."""

    axial_coordinates: Array
    transverse_coordinates: Array
    atom_template: Array
    reference_positions_3d: Array
    axial_period_A: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CrystallineRegistrationParameters1D:
    """The four globally fitted crystalline-host parameters."""

    axial_phase_A: Any = 0.0
    surface_offset_A: Any = 0.0
    rotation_rad: Any = 0.0
    axial_strain: Any = 0.0


@dataclass(frozen=True)
class CrystallineRegistrationResult1D:
    """Registered host, predictions, metrics, and complete optimizer history."""

    initial_parameters: CrystallineRegistrationParameters1D
    optimization_start_parameters: CrystallineRegistrationParameters1D
    parameters: CrystallineRegistrationParameters1D
    host_positions_3d: Array
    potential: Array
    predicted_intensities: Array
    measured_intensities: Array
    detector_angles_mrad: Array
    reflected_detector_mask: Array
    specular_detector_mask: Array
    phase_grid_A: Array
    phase_grid_objective: Array
    objective_history: Array
    parameter_history: Array
    initial_objective: Array
    whole_detector_nrmse: Array
    reflected_nrmse: Array
    specular_nrmse: Array
    converged: Array
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _array(name: str, value: Any, ndim: int) -> Array:
    result = jnp.asarray(value)
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {result.shape}")
    return result


def _concrete_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, jax.core.Tracer):
        return None
    try:
        return np.asarray(value)
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        return None


def _validate_uniform_coordinates(name: str, coordinates: np.ndarray) -> None:
    if coordinates.size < 2 or not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} must contain at least two finite values")
    differences = np.diff(coordinates)
    if np.any(differences <= 0.0) or not np.allclose(
        differences, differences[0], rtol=5e-4, atol=1e-7
    ):
        raise ValueError(f"{name} must be uniformly increasing")


def make_crystalline_host_model_1d(
    axial_coordinates: Any,
    transverse_coordinates: Any,
    atom_template: Any,
    reference_positions_3d: Any,
    *,
    axial_period_A: float,
    metadata: Mapping[str, Any] | None = None,
) -> CrystallineHostModel1D:
    """Validate and build a fixed, single-species crystalline-host model."""
    coordinates_s = np.asarray(axial_coordinates, dtype=float)
    coordinates_u = np.asarray(transverse_coordinates, dtype=float)
    template = np.asarray(atom_template)
    positions = np.asarray(reference_positions_3d, dtype=float)
    if coordinates_s.ndim != 1 or coordinates_u.ndim != 1:
        raise ValueError("specimen coordinates must be one-dimensional")
    _validate_uniform_coordinates("axial_coordinates", coordinates_s)
    _validate_uniform_coordinates("transverse_coordinates", coordinates_u)
    if template.ndim != 2 or min(template.shape) < 3:
        raise ValueError("atom_template must be two-dimensional with at least 3 samples")
    if any(size % 2 == 0 for size in template.shape):
        raise ValueError("atom_template dimensions must be odd")
    if np.iscomplexobj(template) or not np.all(np.isfinite(template)):
        raise ValueError("atom_template must be finite and real")
    if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) == 0:
        raise ValueError("reference_positions_3d must have shape (n_host, 3)")
    if not np.all(np.isfinite(positions)):
        raise ValueError("reference_positions_3d must be finite")
    projected = positions[:, [0, 2]]
    grid_bounds = np.asarray(
        [
            [coordinates_s[0], coordinates_s[-1]],
            [coordinates_u[0], coordinates_u[-1]],
        ]
    )
    if np.any(projected < grid_bounds[:, 0]) or np.any(projected > grid_bounds[:, 1]):
        raise ValueError("reference host positions must lie inside the specimen grid")
    period = float(axial_period_A)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("axial_period_A must be positive and finite")
    return CrystallineHostModel1D(
        axial_coordinates=jnp.asarray(coordinates_s),
        transverse_coordinates=jnp.asarray(coordinates_u),
        atom_template=jnp.asarray(template),
        reference_positions_3d=jnp.asarray(positions),
        axial_period_A=period,
        metadata=dict(metadata or {}),
    )


def _parameters_to_array(parameters: CrystallineRegistrationParameters1D) -> Array:
    if not isinstance(parameters, CrystallineRegistrationParameters1D):
        raise TypeError("parameters must be CrystallineRegistrationParameters1D")
    values = jnp.asarray(
        [
            parameters.axial_phase_A,
            parameters.surface_offset_A,
            parameters.rotation_rad,
            parameters.axial_strain,
        ]
    )
    if values.shape != (4,):
        raise ValueError("registration parameters must be scalar values")
    concrete = _concrete_numpy(values)
    if concrete is not None and not np.all(np.isfinite(concrete)):
        raise ValueError("registration parameters must be finite")
    return values


def _array_to_parameters(values: Any) -> CrystallineRegistrationParameters1D:
    array = _array("parameter vector", values, 1)
    if array.shape != (4,):
        raise ValueError("parameter vector must have shape (4,)")
    return CrystallineRegistrationParameters1D(
        axial_phase_A=array[0],
        surface_offset_A=array[1],
        rotation_rad=array[2],
        axial_strain=array[3],
    )


def transform_crystalline_host_1d(
    reference_positions_3d: Any,
    axial_phase_A: Any,
    surface_offset_A: Any,
    rotation_rad: Any,
    axial_strain: Any,
) -> Array:
    """Apply axial strain, rotation, and translation to every host site.

    Strain and rotation act in the projected ``(s, u)`` plane about the mean
    projected host position. The latent ``y`` coordinate is retained exactly.
    """
    reference = _array("reference_positions_3d", reference_positions_3d, 2)
    if reference.shape[1:] != (3,) or reference.shape[0] == 0:
        raise ValueError("reference_positions_3d must have shape (n_host, 3)")
    scalar_values = jnp.asarray(
        [axial_phase_A, surface_offset_A, rotation_rad, axial_strain]
    )
    if scalar_values.shape != (4,):
        raise ValueError("registration transform parameters must be scalar")
    concrete_reference = _concrete_numpy(reference)
    concrete_scalars = _concrete_numpy(scalar_values)
    if concrete_reference is not None and not np.all(np.isfinite(concrete_reference)):
        raise ValueError("reference_positions_3d must be finite")
    if concrete_scalars is not None and not np.all(np.isfinite(concrete_scalars)):
        raise ValueError("registration transform parameters must be finite")
    projected = reference[:, [0, 2]]
    center = jnp.mean(projected, axis=0)
    relative = projected - center
    strained = relative.at[:, 0].multiply(1.0 + scalar_values[3])
    cosine = jnp.cos(scalar_values[2])
    sine = jnp.sin(scalar_values[2])
    transformed_s = cosine * strained[:, 0] - sine * strained[:, 1]
    transformed_u = sine * strained[:, 0] + cosine * strained[:, 1]
    transformed = jnp.stack([transformed_s, transformed_u], axis=1) + center
    transformed = transformed + scalar_values[:2]
    return reference.at[:, 0].set(transformed[:, 0]).at[:, 2].set(
        transformed[:, 1]
    )


def _same_fft_convolution_2d(image: Array, kernel: Array) -> Array:
    """Convolve two real JAX arrays and crop the odd-kernel ``same`` result."""
    full_shape = (
        image.shape[0] + kernel.shape[0] - 1,
        image.shape[1] + kernel.shape[1] - 1,
    )
    image_frequency = jnp.fft.rfftn(image, full_shape, axes=(0, 1))
    kernel_frequency = jnp.fft.rfftn(kernel, full_shape, axes=(0, 1))
    full = jnp.fft.irfftn(
        image_frequency * kernel_frequency, full_shape, axes=(0, 1)
    )
    start_s = (kernel.shape[0] - 1) // 2
    start_u = (kernel.shape[1] - 1) // 2
    return full[
        start_s : start_s + image.shape[0],
        start_u : start_u + image.shape[1],
    ]


def render_crystalline_host_1d(
    model: CrystallineHostModel1D,
    host_positions_3d: Any,
) -> Array:
    """Render the host by splatting sites and convolving their shared template."""
    positions = _array("host_positions_3d", host_positions_3d, 2)
    if positions.shape != model.reference_positions_3d.shape:
        raise ValueError("host_positions_3d must match the model reference shape")
    concrete_positions = _concrete_numpy(positions)
    if concrete_positions is not None and not np.all(np.isfinite(concrete_positions)):
        raise ValueError("host_positions_3d must be finite")
    site_grid = _splat_crystalline_sites_1d(model, positions)
    template = jnp.asarray(model.atom_template)
    return _same_fft_convolution_2d(site_grid, template)


def _splat_crystalline_sites_1d(
    model: CrystallineHostModel1D,
    positions: Array,
) -> Array:
    """Deposit projected sites onto the specimen grid with cubic weights."""
    coordinates_s = jnp.asarray(model.axial_coordinates)
    coordinates_u = jnp.asarray(model.transverse_coordinates)
    template = jnp.asarray(model.atom_template)
    projected = positions[:, [0, 2]]
    ds = coordinates_s[1] - coordinates_s[0]
    du = coordinates_u[1] - coordinates_u[0]
    output_shape = (coordinates_s.shape[0], coordinates_u.shape[0])
    fractional_indices = jnp.stack(
        [
            (projected[:, 0] - coordinates_s[0]) / ds,
            (projected[:, 1] - coordinates_u[0]) / du,
        ],
        axis=1,
    )
    lower_indices = jnp.floor(fractional_indices).astype(jnp.int32)
    fractions = fractional_indices - lower_indices
    fraction_s = fractions[:, 0]
    fraction_u = fractions[:, 1]

    def cubic_weights(fraction: Array) -> Array:
        return jnp.stack(
            [
                -fraction * (1.0 - fraction) * (2.0 - fraction) / 6.0,
                (1.0 + fraction)
                * (1.0 - fraction)
                * (2.0 - fraction)
                / 2.0,
                (1.0 + fraction) * fraction * (2.0 - fraction) / 2.0,
                -(1.0 + fraction)
                * fraction
                * (1.0 - fraction)
                / 6.0,
            ]
        )

    weights_s = cubic_weights(fraction_s)
    weights_u = cubic_weights(fraction_u)
    corner_weights = (weights_s[:, None, :] * weights_u[None, :, :]).astype(
        template.dtype
    )
    offsets = jnp.arange(-1, 3, dtype=jnp.int32)
    offsets_s, offsets_u = jnp.meshgrid(offsets, offsets, indexing="ij")
    corner_offsets = jnp.stack([offsets_s, offsets_u], axis=-1)
    corner_indices = (
        lower_indices[None, None, :, :] + corner_offsets[:, :, None, :]
    )
    shape_array = jnp.asarray(output_shape, dtype=jnp.int32)
    valid = jnp.all(
        (corner_indices >= 0) & (corner_indices < shape_array), axis=-1
    )
    clipped = jnp.clip(corner_indices, 0, shape_array - 1)
    site_grid = jnp.zeros(output_shape, dtype=template.dtype)
    site_grid = site_grid.at[
        clipped[..., 0].reshape(-1), clipped[..., 1].reshape(-1)
    ].add(jnp.where(valid, corner_weights, 0.0).reshape(-1))
    return site_grid


def _render_with_template_frequency_1d(
    model: CrystallineHostModel1D,
    host_positions_3d: Array,
    template_frequency: Array,
) -> Array:
    """Render using a precomputed FFT of the fixed atom template."""
    site_grid = _splat_crystalline_sites_1d(model, host_positions_3d)
    output_shape = site_grid.shape
    template_shape = model.atom_template.shape
    full_shape = (
        output_shape[0] + template_shape[0] - 1,
        output_shape[1] + template_shape[1] - 1,
    )
    full = jnp.fft.irfftn(
        jnp.fft.rfftn(site_grid, full_shape, axes=(0, 1)) * template_frequency,
        full_shape,
        axes=(0, 1),
    )
    start_s = (template_shape[0] - 1) // 2
    start_u = (template_shape[1] - 1) // 2
    return full[
        start_s : start_s + output_shape[0],
        start_u : start_u + output_shape[1],
    ]


def _normalized_amplitude_numerator(
    predicted: Array,
    measured: Array,
    scan_weights: Array,
    detector_mask: Array,
    epsilon: Array,
) -> Array:
    errors = (
        jnp.sqrt(predicted + epsilon) - jnp.sqrt(measured + epsilon)
    ) ** 2
    weights = scan_weights[:, None] * detector_mask[None, :]
    return jnp.sum(weights * errors)


def _balanced_amplitude_loss_1d(
    predicted: Any,
    measured: Any,
    scan_weights: Any,
    reflected_detector_mask: Any,
    *,
    whole_detector_weight: Any = 0.5,
    epsilon: Any = 1e-12,
) -> Array:
    """Return independently normalized whole/reflected squared amplitude loss."""
    predicted_array = _array("predicted", predicted, 2)
    measured_array = _array("measured", measured, 2)
    weights = _array("scan_weights", scan_weights, 1)
    reflected = _array("reflected_detector_mask", reflected_detector_mask, 1)
    if predicted_array.shape != measured_array.shape:
        raise ValueError("predicted and measured intensities must have equal shapes")
    if weights.shape != (predicted_array.shape[0],):
        raise ValueError("scan_weights must contain one value per scan")
    if reflected.shape != (predicted_array.shape[1],):
        raise ValueError("reflected_detector_mask must contain one value per pixel")
    reflected = reflected.astype(predicted_array.dtype)
    detector_all = jnp.ones_like(reflected)
    epsilon_array = jnp.asarray(epsilon, dtype=predicted_array.dtype)
    all_denominator = jnp.sum(weights[:, None] * measured_array)
    reflected_denominator = jnp.sum(
        weights[:, None] * reflected[None, :] * measured_array
    )
    all_loss = _normalized_amplitude_numerator(
        predicted_array,
        measured_array,
        weights,
        detector_all,
        epsilon_array,
    ) / jnp.maximum(all_denominator, epsilon_array)
    reflected_loss = _normalized_amplitude_numerator(
        predicted_array,
        measured_array,
        weights,
        reflected,
        epsilon_array,
    ) / jnp.maximum(reflected_denominator, epsilon_array)
    resolved_weight = jnp.asarray(whole_detector_weight, dtype=predicted_array.dtype)
    return resolved_weight * all_loss + (1.0 - resolved_weight) * reflected_loss


def _pad_scan_rows(array: Array, padded_size: int) -> Array:
    padding = padded_size - array.shape[0]
    return jnp.pad(array, [(0, padding)] + [(0, 0)] * (array.ndim - 1))


def _simulate_full_domain_batch_1d(
    global_potential: Array,
    input_probes: Array,
    propagation_kernel: Array,
    slice_thickness: Any,
    energy: Any,
    *,
    rematerialize: bool,
) -> Array:
    """Propagate a batch through the complete domain in one scan.

    This is the common registration case: every probe starts at zero and the
    propagation window is the complete potential.  Carrying all probe waves
    in one scan lets XLA issue batched FFTs instead of compiling one scan per
    padded batch.  The generic public simulator remains the fallback for
    arbitrary windows.
    """
    potential = _array("global_potential", global_potential, 2)
    probes = _array("input_probes", input_probes, 2)
    kernel = _array("propagation_kernel", propagation_kernel, 1)
    complex_dtype = jnp.result_type(probes.dtype, kernel.dtype, jnp.complex64)
    wave = probes.astype(complex_dtype)
    transfer = kernel.astype(complex_dtype)
    sigma_dz = interaction_constant(energy) * slice_thickness

    def step(current_wave: Array, potential_slice: Array):
        phase = jnp.exp(1j * sigma_dz * potential_slice)
        current_wave = current_wave * phase[None, :]
        propagated = jnp.fft.ifft(
            jnp.fft.fft(current_wave, axis=-1) * transfer[None, :], axis=-1
        )
        return propagated, None

    scan_step = jax.checkpoint(step) if rematerialize else step
    exit_waves, _ = jax.lax.scan(scan_step, wave, potential)
    detector_waves = jnp.fft.fftshift(
        jnp.fft.fft(exit_waves, axis=-1), axes=-1
    )
    return jnp.abs(detector_waves) ** 2


def _amplitude_numerator_from_amplitudes(
    predicted: Array,
    measured_amplitudes: Array,
    scan_weights: Array,
    detector_mask: Array,
    epsilon: Array,
) -> Array:
    """Squared amplitude numerator when measured amplitudes are precomputed."""
    errors = (jnp.sqrt(predicted + epsilon) - measured_amplitudes) ** 2
    weights = scan_weights[:, None] * detector_mask[None, :]
    return jnp.sum(weights * errors)


def _parameter_scales(model: CrystallineHostModel1D, dtype) -> Array:
    return jnp.asarray(
        [
            model.axial_period_A / 2.0,
            1.0,
            jnp.deg2rad(1.0),
            0.02,
        ],
        dtype=dtype,
    )


def register_crystalline_host_1d(
    model: CrystallineHostModel1D,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    measured_intensities: Any,
    detector_angles_mrad: Any,
    *,
    initial_parameters: CrystallineRegistrationParameters1D | None = None,
    reflected_angle_bounds_mrad: tuple[float, float] = (0.0, 80.0),
    specular_angle_bounds_mrad: tuple[float, float] = (25.0, 45.0),
    whole_detector_weight: float = 0.5,
    batch_size: int = 5,
    phase_grid_points: int = 25,
    updates: int = 200,
    learning_rate_start: float = 5e-2,
    learning_rate_end: float = 1e-3,
    gradient_clip: float = 1.0,
) -> CrystallineRegistrationResult1D:
    """Register a crystalline host using a compiled phase search and Adam fit."""
    try:
        import optax
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("register_crystalline_host_1d requires Optax") from exc
    from tqdm.auto import tqdm

    measured = _array("measured_intensities", measured_intensities, 2)
    detector_angles = _array("detector_angles_mrad", detector_angles_mrad, 1)
    if jnp.issubdtype(detector_angles.dtype, jnp.complexfloating):
        raise ValueError("detector_angles_mrad must be finite and real")
    starts = _array("window_starts", window_starts, 1)
    if not jnp.issubdtype(starts.dtype, jnp.integer):
        raise TypeError("window_starts must contain integers")
    starts = starts.astype(jnp.int32)
    kernel = _array("propagation_kernel", propagation_kernel, 1)
    probe = jnp.asarray(input_probe)
    if probe.ndim not in (1, 2):
        raise ValueError("input_probe must be one- or two-dimensional")
    n_scan, n_u = measured.shape
    if detector_angles.shape != (n_u,) or kernel.shape != (n_u,):
        raise ValueError("detector angles, kernel, and measurements do not match")
    if starts.shape != (n_scan,) or probe.shape[-1] != n_u:
        raise ValueError("scan starts, probes, and measurements do not match")
    probe_rows = jnp.broadcast_to(probe, (n_scan, n_u)) if probe.ndim == 1 else probe
    if probe_rows.shape != (n_scan, n_u):
        raise ValueError("two-dimensional input_probe must have one row per scan")
    resolved_window_length = operator.index(window_length)
    resolved_batch_size = operator.index(batch_size)
    resolved_grid_points = operator.index(phase_grid_points)
    resolved_updates = operator.index(updates)
    if resolved_window_length < 1:
        raise ValueError("window_length must be positive")
    if resolved_window_length > model.axial_coordinates.shape[0]:
        raise ValueError("window_length cannot exceed the specimen grid")
    if resolved_batch_size < 1:
        raise ValueError("batch_size must be positive")
    if resolved_grid_points < 3:
        raise ValueError("phase_grid_points must be at least three")
    if resolved_updates < 1:
        raise ValueError("updates must be positive")
    for name, value in (
        ("slice_thickness", slice_thickness),
        ("energy", energy),
        ("learning_rate_start", learning_rate_start),
        ("learning_rate_end", learning_rate_end),
        ("gradient_clip", gradient_clip),
    ):
        concrete = float(value)
        if not np.isfinite(concrete) or concrete <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    resolved_whole_weight = float(whole_detector_weight)
    if not np.isfinite(resolved_whole_weight) or not 0.0 <= resolved_whole_weight <= 1.0:
        raise ValueError("whole_detector_weight must lie in [0, 1]")
    reflected_bounds = np.asarray(reflected_angle_bounds_mrad, dtype=float)
    specular_bounds = np.asarray(specular_angle_bounds_mrad, dtype=float)
    if (
        reflected_bounds.shape != (2,)
        or not np.all(np.isfinite(reflected_bounds))
        or reflected_bounds[1] <= reflected_bounds[0]
    ):
        raise ValueError("reflected_angle_bounds_mrad must be increasing")
    if (
        specular_bounds.shape != (2,)
        or not np.all(np.isfinite(specular_bounds))
        or specular_bounds[1] <= specular_bounds[0]
    ):
        raise ValueError("specular_angle_bounds_mrad must be increasing")
    reflected_mask = (detector_angles > reflected_bounds[0]) & (
        detector_angles < reflected_bounds[1]
    )
    specular_mask = (detector_angles > specular_bounds[0]) & (
        detector_angles < specular_bounds[1]
    )
    reflected_host = _concrete_numpy(reflected_mask)
    specular_host = _concrete_numpy(specular_mask)
    measured_host = _concrete_numpy(measured)
    detector_host = _concrete_numpy(detector_angles)
    starts_host = _concrete_numpy(starts)
    probes_host = _concrete_numpy(probe_rows)
    kernel_host = _concrete_numpy(kernel)
    if detector_host is not None and (
        np.iscomplexobj(detector_host) or not np.all(np.isfinite(detector_host))
    ):
        raise ValueError("detector_angles_mrad must be finite and real")
    if starts_host is not None and (
        np.any(starts_host < 0)
        or np.any(starts_host + resolved_window_length > model.axial_coordinates.shape[0])
    ):
        raise ValueError("window_starts place a scan outside the specimen grid")
    if probes_host is not None and not np.all(np.isfinite(probes_host)):
        raise ValueError("input_probe must be finite")
    if kernel_host is not None and not np.all(np.isfinite(kernel_host)):
        raise ValueError("propagation_kernel must be finite")
    if reflected_host is not None and not np.any(reflected_host):
        raise ValueError("reflected detector band contains no pixels")
    if specular_host is not None and not np.any(specular_host):
        raise ValueError("specular detector band contains no pixels")
    if measured_host is not None and (
        np.iscomplexobj(measured_host)
        or not np.all(np.isfinite(measured_host))
        or np.any(measured_host < 0.0)
    ):
        raise ValueError("measured_intensities must be finite and non-negative")

    initial = initial_parameters or CrystallineRegistrationParameters1D()
    initial_physical = _parameters_to_array(initial).astype(
        jnp.result_type(measured, model.reference_positions_3d, jnp.float32)
    )
    scales = _parameter_scales(model, initial_physical.dtype)
    initial_normalized = initial_physical / scales
    initial_host = _concrete_numpy(initial_normalized)
    if initial_host is not None and np.any(np.abs(initial_host) > 1.0):
        raise ValueError("initial registration parameters lie outside fit bounds")

    padded_scan_count = (
        (n_scan + resolved_batch_size - 1) // resolved_batch_size
    ) * resolved_batch_size
    n_batch = padded_scan_count // resolved_batch_size
    padded_probes = _pad_scan_rows(probe_rows, padded_scan_count).reshape(
        n_batch, resolved_batch_size, n_u
    )
    padded_starts = _pad_scan_rows(starts, padded_scan_count).reshape(
        n_batch, resolved_batch_size
    )
    padded_measured = _pad_scan_rows(measured, padded_scan_count).reshape(
        n_batch, resolved_batch_size, n_u
    )
    padded_weights = jnp.pad(
        jnp.ones((n_scan,), dtype=measured.dtype),
        (0, padded_scan_count - n_scan),
    ).reshape(n_batch, resolved_batch_size)
    epsilon = jnp.asarray(1e-12, dtype=measured.dtype)
    padded_measured_amplitudes = jnp.sqrt(padded_measured + epsilon)
    reflected_numeric = reflected_mask.astype(measured.dtype)
    all_denominator = jnp.sum(measured)
    reflected_denominator = jnp.sum(measured * reflected_numeric[None, :])
    if measured_host is not None:
        if float(all_denominator) <= 0.0 or float(reflected_denominator) <= 0.0:
            raise ValueError("measured intensity must be positive in both loss bands")

    def decode(normalized_parameters: Array) -> Array:
        return normalized_parameters * scales

    template_shape = model.atom_template.shape
    render_full_shape = (
        model.axial_coordinates.shape[0] + template_shape[0] - 1,
        model.transverse_coordinates.shape[0] + template_shape[1] - 1,
    )
    template_frequency = jnp.fft.rfftn(
        jnp.asarray(model.atom_template), render_full_shape, axes=(0, 1)
    )

    def potential_from_normalized(normalized_parameters: Array) -> Array:
        physical = decode(normalized_parameters)
        positions = transform_crystalline_host_1d(
            model.reference_positions_3d,
            physical[0],
            physical[1],
            physical[2],
            physical[3],
        )
        return _render_with_template_frequency_1d(
            model, positions, template_frequency
        )

    flat_probes = padded_probes.reshape(padded_scan_count, n_u)
    flat_starts = padded_starts.reshape(padded_scan_count)
    full_domain_batch = bool(
        starts_host is not None
        and resolved_window_length == model.axial_coordinates.shape[0]
        and np.all(starts_host == 0)
    )

    def predict_flat(potential: Array) -> Array:
        if full_domain_batch:
            return _simulate_full_domain_batch_1d(
                potential,
                flat_probes,
                kernel,
                slice_thickness,
                energy,
                # ``predict_flat`` is wrapped in one checkpoint below.  A
                # second checkpoint around every slice would needlessly
                # rematerialize the same batched scan and is substantially
                # slower for the full-domain case.
                rematerialize=False,
            )
        return simulate_glancing_scan_1d(
            potential,
            flat_probes,
            flat_starts,
            resolved_window_length,
            kernel,
            slice_thickness,
            energy,
            rematerialize=True,
        )

    checkpointed_predict_flat = jax.checkpoint(predict_flat)

    def objective(normalized_parameters: Array) -> Array:
        potential = potential_from_normalized(normalized_parameters)
        predicted_batches = checkpointed_predict_flat(potential).reshape(
            n_batch, resolved_batch_size, n_u
        )

        def accumulate(carry, batch):
            predicted_batch, measured_amplitudes_batch, scan_weights = batch
            all_numerator, reflected_numerator = carry
            all_numerator = all_numerator + _amplitude_numerator_from_amplitudes(
                predicted_batch,
                measured_amplitudes_batch,
                scan_weights,
                jnp.ones((n_u,), dtype=measured.dtype),
                epsilon,
            )
            reflected_numerator = (
                reflected_numerator
                + _amplitude_numerator_from_amplitudes(
                    predicted_batch,
                    measured_amplitudes_batch,
                    scan_weights,
                    reflected_numeric,
                    epsilon,
                )
            )
            return (all_numerator, reflected_numerator), None

        (all_numerator, reflected_numerator), _ = jax.lax.scan(
            accumulate,
            (jnp.asarray(0.0, measured.dtype), jnp.asarray(0.0, measured.dtype)),
            (predicted_batches, padded_measured_amplitudes, padded_weights),
        )
        all_loss = all_numerator / jnp.maximum(all_denominator, epsilon)
        reflected_loss = reflected_numerator / jnp.maximum(
            reflected_denominator, epsilon
        )
        return (
            resolved_whole_weight * all_loss
            + (1.0 - resolved_whole_weight) * reflected_loss
        )

    objective_jit = jax.jit(objective)
    value_and_grad = jax.jit(jax.value_and_grad(objective))
    phase_grid = (
        jnp.arange(resolved_grid_points, dtype=initial_physical.dtype)
        - resolved_grid_points // 2
    ) * (model.axial_period_A / resolved_grid_points)

    def evaluate_phase(phase_A):
        candidate = initial_normalized.at[0].set(phase_A / scales[0])
        return objective_jit(candidate)

    # Both phases are compiled JAX operations, so keep the bar at the stage
    # level rather than introducing host callbacks into the CUDA scan.
    progress_bar = tqdm(
        total=2,
        desc="crystal registration",
        unit="stage",
        dynamic_ncols=True,
    )
    phase_grid_objective = jax.lax.map(evaluate_phase, phase_grid)
    progress_bar.update(1)
    selected_phase = phase_grid[jnp.argmin(phase_grid_objective)]
    optimization_start = initial_normalized.at[0].set(selected_phase / scales[0])
    initial_objective = objective_jit(initial_normalized)

    schedule = optax.cosine_decay_schedule(
        learning_rate_start,
        resolved_updates,
        alpha=learning_rate_end / learning_rate_start,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.adam(schedule),
    )
    optimizer_state = optimizer.init(optimization_start)

    def update_step(carry, _):
        normalized_parameters, state = carry
        value, gradients = value_and_grad(normalized_parameters)
        parameter_updates, state = optimizer.update(
            gradients, state, normalized_parameters
        )
        updated = optax.apply_updates(normalized_parameters, parameter_updates)
        updated = jnp.clip(updated, -1.0, 1.0)
        return (updated, state), (value, normalized_parameters)

    @jax.jit
    def run_updates(normalized_parameters, state):
        return jax.lax.scan(
            update_step,
            (normalized_parameters, state),
            xs=None,
            length=resolved_updates,
        )

    try:
        (final_normalized, _), (recorded_objective, recorded_normalized) = run_updates(
            optimization_start, optimizer_state
        )
        progress_bar.update(1)
    finally:
        progress_bar.close()
    final_objective = objective_jit(final_normalized)
    objective_history = jnp.concatenate(
        [recorded_objective, final_objective[None]], axis=0
    )
    normalized_history = jnp.concatenate(
        [recorded_normalized, final_normalized[None, :]], axis=0
    )
    parameter_history = normalized_history * scales[None, :]
    final_physical = decode(final_normalized)
    final_positions = transform_crystalline_host_1d(
        model.reference_positions_3d,
        final_physical[0],
        final_physical[1],
        final_physical[2],
        final_physical[3],
    )
    final_potential = _render_with_template_frequency_1d(
        model, final_positions, template_frequency
    )

    predicted = jax.jit(predict_flat)(final_potential)[:n_scan]
    scan_weights = jnp.ones((n_scan,), dtype=measured.dtype)
    whole_loss = _balanced_amplitude_loss_1d(
        predicted,
        measured,
        scan_weights,
        jnp.ones((n_u,), dtype=bool),
        whole_detector_weight=1.0,
    )
    reflected_loss = _balanced_amplitude_loss_1d(
        predicted,
        measured,
        scan_weights,
        reflected_mask,
        whole_detector_weight=0.0,
    )
    specular_loss = _balanced_amplitude_loss_1d(
        predicted,
        measured,
        scan_weights,
        specular_mask,
        whole_detector_weight=0.0,
    )
    optimization_start_physical = decode(optimization_start)
    metadata = {
        **dict(model.metadata),
        "parameter_names": [
            "axial_phase_A",
            "surface_offset_A",
            "rotation_rad",
            "axial_strain",
        ],
        "n_parameters": 4,
        "n_scans": int(n_scan),
        "batch_size": resolved_batch_size,
        "phase_grid_points": resolved_grid_points,
        "updates": resolved_updates,
        "learning_rate_start": float(learning_rate_start),
        "learning_rate_end": float(learning_rate_end),
        "whole_detector_weight": resolved_whole_weight,
        "reflected_angle_bounds_mrad": reflected_bounds.tolist(),
        "specular_angle_bounds_mrad": specular_bounds.tolist(),
    }
    return CrystallineRegistrationResult1D(
        initial_parameters=_array_to_parameters(initial_physical),
        optimization_start_parameters=_array_to_parameters(
            optimization_start_physical
        ),
        parameters=_array_to_parameters(final_physical),
        host_positions_3d=final_positions,
        potential=final_potential,
        predicted_intensities=predicted,
        measured_intensities=measured,
        detector_angles_mrad=detector_angles,
        reflected_detector_mask=reflected_mask,
        specular_detector_mask=specular_mask,
        phase_grid_A=phase_grid,
        phase_grid_objective=phase_grid_objective,
        objective_history=objective_history,
        parameter_history=parameter_history,
        initial_objective=initial_objective,
        whole_detector_nrmse=jnp.sqrt(whole_loss),
        reflected_nrmse=jnp.sqrt(reflected_loss),
        specular_nrmse=jnp.sqrt(specular_loss),
        converged=jnp.isfinite(final_objective) & (final_objective < initial_objective),
        metadata=metadata,
    )
