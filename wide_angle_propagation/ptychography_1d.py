"""Differentiable 1D glancing-incidence ptychography helpers.

The propagation coordinate is ``s`` and the single transverse/detector
coordinate is ``u``.  A scan translates a fixed-length axial window through a
global two-dimensional electrostatic potential ``V(s, u)``.  Reconstruction is
deliberately limited to independent real-potential pixels inside a finite
geometric region around the scanned beam paths; there is no height or surface
parameterization.
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

from .propagation_methods import energy2wavelength, interaction_constant


__all__ = [
    "beam_path_reconstruction_region_1d",
    "GlancingScan1D",
    "GlancingSideviewCache1D",
    "PotentialReconstruction1D",
    "load_glancing_scan_1d",
    "load_glancing_sideview_cache_1d",
    "load_potential_reconstruction_1d",
    "normalized_amplitude_loss_1d",
    "reconstruct_potential_1d",
    "save_glancing_scan_1d",
    "save_glancing_sideview_cache_1d",
    "save_potential_reconstruction_1d",
    "simulate_glancing_scan_1d",
    "simulate_glancing_sideview_cache_1d",
]


Array = Any


@dataclass(frozen=True)
class GlancingScan1D:
    """A simulated scan and the coordinates needed to interpret it."""

    intensities: Array
    window_starts: Array
    scan_coordinates: Array
    detector_angles: Array
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GlancingSideviewCache1D:
    """Downsampled internal fields and full exit/detector waves for selected scans."""

    scan_indices: Array
    window_starts: Array
    scan_coordinates: Array
    local_s_coordinates: Array
    sideview_u_coordinates: Array
    transverse_coordinates: Array
    sideview_wavefields: Array
    sideview_intensities: Array
    exit_waves: Array
    detector_waves: Array
    detector_intensities: Array
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PotentialReconstruction1D:
    """Best direct-potential estimate and its optimization diagnostics."""

    potential: Array
    initial_potential: Array
    reconstruction_mask: Array
    axial_coordinates: Array
    transverse_coordinates: Array
    predicted_intensities: Array
    measured_intensities: Array
    window_starts: Array
    scan_coordinates: Array
    detector_angles: Array
    update_history: Array
    training_loss_history: Array
    validation_loss_history: Array
    best_update: int
    elapsed_time_history: Array = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _array(name: str, value: Any, ndim: int) -> Array:
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


def _positive_scalar(name: str, value: Any, *, allow_zero: bool = False) -> None:
    concrete = _concrete_numpy(value)
    if concrete is None:
        return
    if concrete.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    if np.iscomplexobj(concrete):
        raise TypeError(f"{name} must be real")
    scalar = float(concrete)
    valid = np.isfinite(scalar) and (scalar >= 0.0 if allow_zero else scalar > 0.0)
    if not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {relation} scalar, got {scalar!r}")


def _integer(name: str, value: Any, *, minimum: int = 1) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result}")
    return result


def _validate_window_starts(
    window_starts: Any,
    *,
    n_s: int,
    window_length: int,
) -> Array:
    starts = _array("window_starts", window_starts, 1)
    if not jnp.issubdtype(starts.dtype, jnp.integer):
        raise TypeError("window_starts must contain integers")
    if starts.shape[0] == 0:
        raise ValueError("window_starts must contain at least one scan")
    concrete = _concrete_numpy(starts)
    if concrete is not None and (
        np.any(concrete < 0) or np.any(concrete + window_length > n_s)
    ):
        raise ValueError(
            "every window start must satisfy "
            f"0 <= start <= {n_s - window_length}"
        )
    return starts


def _pixel_spacing(coordinates: Array) -> Array:
    if coordinates.shape[0] < 2:
        raise ValueError("coordinates must contain at least two points")
    concrete = _concrete_numpy(coordinates)
    if concrete is not None:
        differences = np.diff(concrete.astype(float, copy=False))
        if not np.all(np.isfinite(concrete)) or np.any(differences == 0.0):
            raise ValueError("coordinates must contain finite, distinct neighbors")
    return jnp.median(jnp.abs(jnp.diff(coordinates)))


def _validate_progress(progress: bool, description: str) -> None:
    if not isinstance(progress, (bool, np.bool_)):
        raise TypeError("progress must be a boolean")
    if not isinstance(description, str):
        raise TypeError("progress_description must be a string")


def _update_iterator(
    updates: int,
    *,
    progress: bool,
    description: str,
):
    """Return an update iterator, optionally backed by notebook-safe TQDM."""
    _validate_progress(progress, description)
    if not progress:
        return range(1, updates + 1)
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:  # pragma: no cover - optional display dependency
        raise ImportError("progress=True requires tqdm") from exc
    return tqdm(
        range(1, updates + 1),
        total=updates,
        desc=description,
        unit="update",
        dynamic_ncols=True,
    )


def _multislice_step(
    wave: Array,
    potential_slice: Array,
    transfer: Array,
    sigma_dz: Array,
) -> Array:
    wave = wave * jnp.exp(1j * sigma_dz * potential_slice)
    return jnp.fft.ifft(jnp.fft.fft(wave, axis=-1) * transfer, axis=-1)


def simulate_glancing_scan_1d(
    global_potential: Any,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    *,
    rematerialize: bool = False,
) -> Array:
    """Return full fftshifted intensities for scan-specific probes and windows.

    The wave is propagated with :func:`jax.lax.scan`; FFTs act only on the last
    axis and no internal wavefront stack is retained. ``input_probe`` may be a
    single probe shared by all scans or a two-dimensional array containing one
    probe per scan.
    """
    potential = _array("global_potential", global_potential, 2)
    probe = jnp.asarray(input_probe)
    if probe.ndim not in (1, 2):
        raise ValueError("input_probe must be one- or two-dimensional")
    kernel = _array("propagation_kernel", propagation_kernel, 1)
    length = _integer("window_length", window_length)
    n_s, n_u = potential.shape
    if length > n_s:
        raise ValueError("window_length cannot exceed global_potential.shape[0]")
    if probe.shape[-1] != n_u or kernel.shape[0] != n_u:
        raise ValueError("input_probe and propagation_kernel must have length n_u")
    starts = _validate_window_starts(
        window_starts,
        n_s=n_s,
        window_length=length,
    )
    _positive_scalar("slice_thickness", slice_thickness)
    _positive_scalar("energy", energy)
    if not isinstance(rematerialize, (bool, np.bool_)):
        raise TypeError("rematerialize must be a boolean")

    if probe.ndim == 1:
        probes = jnp.broadcast_to(probe, (starts.shape[0], n_u))
    elif probe.shape[0] == starts.shape[0]:
        probes = probe
    else:
        raise ValueError("two-dimensional input_probe must have one row per scan")
    complex_dtype = jnp.result_type(probes.dtype, kernel.dtype, jnp.complex64)
    probes = probes.astype(complex_dtype)
    transfer = kernel.astype(complex_dtype)
    sigma_dz = interaction_constant(energy) * slice_thickness

    def step(wave: Array, potential_slice: Array) -> tuple[Array, None]:
        return _multislice_step(wave, potential_slice, transfer, sigma_dz), None

    scan_step = jax.checkpoint(step) if rematerialize else step

    def run_window(start: Array, initial_wave: Array) -> Array:
        slices = jax.lax.dynamic_slice_in_dim(potential, start, length, axis=0)
        exit_wave, _ = jax.lax.scan(scan_step, initial_wave, slices)
        return exit_wave

    exit_waves = jax.vmap(run_window)(starts, probes)
    detector_waves = jnp.fft.fftshift(jnp.fft.fft(exit_waves, axis=-1), axes=-1)
    return jnp.abs(detector_waves) ** 2


def normalized_amplitude_loss_1d(
    predicted_intensities: Any,
    measured_intensities: Any,
    *,
    epsilon: Any = 1e-12,
) -> Array:
    """Return the normalized squared error between predicted amplitudes."""
    predicted = jnp.asarray(predicted_intensities)
    measured = jnp.asarray(measured_intensities)
    if predicted.shape != measured.shape:
        raise ValueError(
            "predicted_intensities and measured_intensities must have identical "
            f"shapes, got {predicted.shape} and {measured.shape}"
        )
    if predicted.ndim == 0:
        raise ValueError("intensity arrays must have at least one dimension")
    if jnp.iscomplexobj(predicted) or jnp.iscomplexobj(measured):
        raise TypeError("intensity arrays must be real")
    _positive_scalar("epsilon", epsilon)
    predicted_host = _concrete_numpy(predicted)
    measured_host = _concrete_numpy(measured)
    if predicted_host is not None and np.any(predicted_host < 0.0):
        raise ValueError("predicted_intensities must be non-negative")
    if measured_host is not None and np.any(measured_host < 0.0):
        raise ValueError("measured_intensities must be non-negative")
    amplitude_error = (
        jnp.sqrt(predicted + epsilon) - jnp.sqrt(measured + epsilon)
    ) ** 2
    return jnp.sum(amplitude_error) / jnp.maximum(jnp.sum(measured), epsilon)


def beam_path_reconstruction_region_1d(
    n_global_s: int,
    transverse_coordinates: Any,
    window_starts: Any,
    window_length: int,
    axial_sampling: Any,
    beam_tilt: Any,
    beam_waist: Any,
    slab_bottom: Any,
    *,
    slab_top: Any = 0.0,
    radius_waists: Any = 3.0,
    minimum_scan_coverage: int = 1,
) -> tuple[Array, Array]:
    """Return a geometric beam-path mask and per-pixel scan coverage count.

    Scan ``j`` crosses ``u=0`` at the midpoint of its local propagation
    window.  Its centreline is

    ``u_j(s) = tan(beam_tilt) * (s - s_cross_j)``.

    A material pixel belongs to the reconstruction region when it lies within
    ``radius_waists * beam_waist`` perpendicular distance of at least
    ``minimum_scan_coverage`` centrelines while those scans are inside their
    local windows.  Potential values inside the returned mask remain mutually
    independent; only their finite geometric support is prescribed.
    """
    n_s = _integer("n_global_s", n_global_s)
    length = _integer("window_length", window_length)
    if length > n_s:
        raise ValueError("window_length cannot exceed n_global_s")
    coordinates_u = _array("transverse_coordinates", transverse_coordinates, 1)
    if jnp.iscomplexobj(coordinates_u):
        raise TypeError("transverse_coordinates must be real")
    starts = _validate_window_starts(window_starts, n_s=n_s, window_length=length)
    _positive_scalar("axial_sampling", axial_sampling)
    _positive_scalar("beam_waist", beam_waist)
    _positive_scalar("radius_waists", radius_waists)
    coverage_required = _integer("minimum_scan_coverage", minimum_scan_coverage)
    tilt_host = np.asarray(beam_tilt)
    if tilt_host.ndim != 0 or np.iscomplexobj(tilt_host) or not np.isfinite(tilt_host):
        raise ValueError("beam_tilt must be a finite real scalar")
    bottom = float(np.asarray(slab_bottom))
    top = float(np.asarray(slab_top))
    if not np.isfinite(bottom) or not np.isfinite(top) or bottom >= top:
        raise ValueError("slab_bottom and slab_top must be finite with bottom < top")

    ds = float(np.asarray(axial_sampling))
    tilt = float(tilt_host)
    radius = float(np.asarray(radius_waists)) * float(np.asarray(beam_waist))
    local_s = np.arange(length, dtype=float) * ds
    local_midpoint = 0.5 * length * ds
    center_u = np.tan(tilt) * (local_s - local_midpoint)
    perpendicular_scale = abs(np.cos(tilt))
    u_host = np.asarray(coordinates_u, dtype=float)
    material = (u_host >= bottom) & (u_host <= top)
    local_distance = (
        np.abs(u_host[None, :] - center_u[:, None]) * perpendicular_scale
    )
    local_region = (local_distance <= radius) & material[None, :]

    coverage = np.zeros((n_s, u_host.size), dtype=np.int32)
    for start in np.asarray(starts, dtype=np.int64):
        coverage[int(start) : int(start) + length] += local_region
    mask = material[None, :] & (coverage >= coverage_required)
    return jnp.asarray(mask), jnp.asarray(coverage)


def _block_average_2d(array: Array, stride_s: int, stride_u: int) -> Array:
    n_s = (array.shape[-2] // stride_s) * stride_s
    n_u = (array.shape[-1] // stride_u) * stride_u
    trimmed = array[..., :n_s, :n_u]
    shape = (*trimmed.shape[:-2], n_s // stride_s, stride_s, n_u // stride_u, stride_u)
    return trimmed.reshape(shape).mean(axis=(-3, -1))


def _block_average_1d(array: Array, stride: int) -> Array:
    n = (array.shape[0] // stride) * stride
    return array[:n].reshape(n // stride, stride).mean(axis=1)


def simulate_glancing_sideview_cache_1d(
    global_potential: Any,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    scan_indices: Any,
    *,
    transverse_coordinates: Any | None = None,
    scan_coordinates: Any | None = None,
    axial_stride: int = 8,
    transverse_stride: int = 2,
    metadata: Mapping[str, Any] | None = None,
) -> GlancingSideviewCache1D:
    """Generate a compact diagnostic cache for selected scan positions only.

    ``input_probe`` can be shared across scans or supplied as one probe per scan.
    """
    potential = _array("global_potential", global_potential, 2)
    probe = jnp.asarray(input_probe)
    if probe.ndim not in (1, 2):
        raise ValueError("input_probe must be one- or two-dimensional")
    kernel = _array("propagation_kernel", propagation_kernel, 1)
    length = _integer("window_length", window_length)
    stride_s = _integer("axial_stride", axial_stride)
    stride_u = _integer("transverse_stride", transverse_stride)
    n_s, n_u = potential.shape
    if length > n_s:
        raise ValueError("window_length cannot exceed global_potential.shape[0]")
    if probe.shape[-1] != n_u or kernel.shape[0] != n_u:
        raise ValueError("input_probe and propagation_kernel must have length n_u")
    starts = _validate_window_starts(window_starts, n_s=n_s, window_length=length)
    indices = _array("scan_indices", scan_indices, 1)
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise TypeError("scan_indices must contain integers")
    indices_host = np.asarray(indices, dtype=np.int64)
    if (
        indices_host.size == 0
        or np.any(indices_host < 0)
        or np.any(indices_host >= starts.shape[0])
    ):
        raise ValueError("scan_indices must contain valid scan positions")
    if np.unique(indices_host).size != indices_host.size:
        raise ValueError("scan_indices must be unique")
    _positive_scalar("slice_thickness", slice_thickness)
    _positive_scalar("energy", energy)

    if transverse_coordinates is None:
        coordinates_u = jnp.arange(n_u, dtype=jnp.float32)
    else:
        coordinates_u = _array("transverse_coordinates", transverse_coordinates, 1)
        if coordinates_u.shape[0] != n_u:
            raise ValueError("transverse_coordinates must have length n_u")
    if scan_coordinates is None:
        coordinates_scan = (starts + length / 2) * slice_thickness
    else:
        coordinates_scan = _array("scan_coordinates", scan_coordinates, 1)
        if coordinates_scan.shape[0] != starts.shape[0]:
            raise ValueError("scan_coordinates must have length n_scan")

    if probe.ndim == 1:
        probes = jnp.broadcast_to(probe, (starts.shape[0], n_u))
    elif probe.shape[0] == starts.shape[0]:
        probes = probe
    else:
        raise ValueError("two-dimensional input_probe must have one row per scan")
    transfer = kernel.astype(jnp.result_type(probes, kernel, jnp.complex64))
    sigma_dz = interaction_constant(energy) * slice_thickness

    def run_window(start: Array, initial_wave: Array) -> tuple[Array, Array]:
        slices = jax.lax.dynamic_slice_in_dim(potential, start, length, axis=0)

        def step(wave: Array, potential_slice: Array) -> tuple[Array, Array]:
            wave = _multislice_step(wave, potential_slice, transfer, sigma_dz)
            return wave, wave

        return jax.lax.scan(step, initial_wave, slices)

    run_window_jit = jax.jit(run_window)
    sideview_fields = []
    sideview_intensities = []
    exit_waves = []
    detector_waves = []
    detector_intensities = []
    for index in indices_host:
        exit_wave, wavefields = run_window_jit(
            starts[int(index)], probes[int(index)].astype(transfer.dtype)
        )
        detector_wave = jnp.fft.fftshift(jnp.fft.fft(exit_wave))
        sideview_fields.append(
            _block_average_2d(wavefields, stride_s, stride_u).astype(jnp.complex64)
        )
        sideview_intensities.append(
            _block_average_2d(
                jnp.abs(wavefields) ** 2,
                stride_s,
                stride_u,
            ).astype(jnp.float32)
        )
        exit_waves.append(exit_wave)
        detector_waves.append(detector_wave)
        detector_intensities.append(jnp.abs(detector_wave) ** 2)

    sideview_fields = jnp.stack(sideview_fields)
    sideview_intensities = jnp.stack(sideview_intensities)
    local_s = _block_average_1d(
        jnp.arange(length, dtype=jnp.result_type(slice_thickness, jnp.float32))
        * slice_thickness,
        stride_s,
    )
    sideview_u = _block_average_1d(coordinates_u, stride_u)
    selected_starts = starts[indices]
    cache_metadata = {
        "axial_stride": stride_s,
        "transverse_stride": stride_u,
        "original_sideview_shape": [length, n_u],
        "stored_sideview_shape": list(sideview_fields.shape[-2:]),
        "complex_dtype": "complex64",
        "intensity_dtype": "float32",
        "downsampling": "complex and intensity block averages computed separately",
        **dict(metadata or {}),
    }
    return GlancingSideviewCache1D(
        scan_indices=indices,
        window_starts=selected_starts,
        scan_coordinates=coordinates_scan[indices],
        local_s_coordinates=local_s,
        sideview_u_coordinates=sideview_u,
        transverse_coordinates=coordinates_u,
        sideview_wavefields=sideview_fields,
        sideview_intensities=sideview_intensities,
        exit_waves=jnp.stack(exit_waves).astype(jnp.complex64),
        detector_waves=jnp.stack(detector_waves).astype(jnp.complex64),
        detector_intensities=jnp.stack(detector_intensities).astype(jnp.float32),
        metadata=cache_metadata,
    )


def _scatter_masked_values(
    normalized_values: Array,
    flat_indices: Array,
    shape: tuple[int, int],
    potential_scale: Array,
) -> Array:
    flat = jnp.zeros((shape[0] * shape[1],), dtype=normalized_values.dtype)
    flat = flat.at[flat_indices].set(normalized_values * potential_scale)
    return flat.reshape(shape)


def reconstruct_potential_1d(
    initial_potential: Any,
    reconstruction_mask: Any,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    measured_intensities: Any,
    *,
    axial_coordinates: Any | None = None,
    transverse_coordinates: Any | None = None,
    scan_coordinates: Any | None = None,
    detector_angles: Any | None = None,
    validation_indices: Sequence[int] = (),
    fixed_potential: Any | None = None,
    potential_scale: Any | None = None,
    potential_max: Any | None = None,
    learning_rate_start: Any = 1e-2,
    learning_rate_end: Any = 1e-4,
    updates: int = 4000,
    minibatch_size: int = 5,
    validation_interval: int = 100,
    evaluation_batch_size: int = 10,
    gradient_clip: Any = 1.0,
    epsilon: Any = 1e-12,
    rematerialize: bool = True,
    seed: int = 0,
    progress: bool = False,
    progress_description: str = "pixel reconstruction",
) -> PotentialReconstruction1D:
    """Recover non-negative pixels while retaining an optional fixed exterior.

    Values inside ``reconstruction_mask`` are initialized from
    ``initial_potential`` and optimized independently. Values outside the mask
    are zero unless ``fixed_potential`` is supplied, in which case its exterior
    values remain fixed in every forward simulation.
    """
    try:
        import optax
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "reconstruct_potential_1d requires Optax; install the "
            "'ptychography' extra"
        ) from exc

    initial = _array("initial_potential", initial_potential, 2)
    mask = _array("reconstruction_mask", reconstruction_mask, 2).astype(bool)
    if mask.shape != initial.shape:
        raise ValueError("reconstruction_mask must match initial_potential.shape")
    initial_host = np.asarray(initial)
    mask_host = np.asarray(mask)
    if not np.any(mask_host):
        raise ValueError("reconstruction_mask must select at least one pixel")
    if np.iscomplexobj(initial_host) or not np.all(np.isfinite(initial_host)):
        raise ValueError("initial_potential must be finite and real")
    if np.any(initial_host[mask_host] < 0.0):
        raise ValueError("initial_potential must be non-negative inside the mask")
    if fixed_potential is None:
        fixed = jnp.zeros_like(initial)
        fixed_host = np.zeros_like(initial_host)
    else:
        fixed = _array("fixed_potential", fixed_potential, 2)
        if fixed.shape != initial.shape:
            raise ValueError("fixed_potential must match initial_potential.shape")
        fixed_host = np.asarray(fixed)
        if (
            np.iscomplexobj(fixed_host)
            or not np.all(np.isfinite(fixed_host))
            or np.any(fixed_host[~mask_host] < 0.0)
        ):
            raise ValueError(
                "fixed_potential must be finite, real, and non-negative outside "
                "the reconstruction mask"
            )

    probe = jnp.asarray(input_probe)
    if probe.ndim not in (1, 2):
        raise ValueError("input_probe must be one- or two-dimensional")
    kernel = _array("propagation_kernel", propagation_kernel, 1)
    measured = _array("measured_intensities", measured_intensities, 2)
    n_s, n_u = initial.shape
    length = _integer("window_length", window_length)
    starts = _validate_window_starts(window_starts, n_s=n_s, window_length=length)
    n_scan = starts.shape[0]
    if probe.shape[-1] != n_u or kernel.shape[0] != n_u:
        raise ValueError("input_probe and propagation_kernel must have length n_u")
    if probe.ndim == 2 and probe.shape[0] != n_scan:
        raise ValueError("two-dimensional input_probe must have one row per scan")
    if measured.shape != (n_scan, n_u):
        raise ValueError(f"measured_intensities must have shape {(n_scan, n_u)}")
    measured_host = np.asarray(measured)
    if not np.all(np.isfinite(measured_host)) or np.any(measured_host < 0.0):
        raise ValueError("measured_intensities must be finite and non-negative")

    n_updates = _integer("updates", updates)
    batch_size = _integer("minibatch_size", minibatch_size)
    metric_interval = _integer("validation_interval", validation_interval)
    eval_batch_size = _integer("evaluation_batch_size", evaluation_batch_size)
    seed_value = operator.index(seed)
    _positive_scalar("slice_thickness", slice_thickness)
    _positive_scalar("energy", energy)
    _positive_scalar("learning_rate_start", learning_rate_start)
    _positive_scalar("learning_rate_end", learning_rate_end)
    _positive_scalar("gradient_clip", gradient_clip)
    _positive_scalar("epsilon", epsilon)
    if float(np.asarray(learning_rate_end)) > float(np.asarray(learning_rate_start)):
        raise ValueError("learning_rate_end must not exceed learning_rate_start")
    if not isinstance(rematerialize, (bool, np.bool_)):
        raise TypeError("rematerialize must be a boolean")
    _validate_progress(progress, progress_description)

    positive_initial = initial_host[mask_host & (initial_host > 0.0)]
    if potential_scale is None:
        resolved_scale = float(np.mean(positive_initial)) if positive_initial.size else 1.0
    else:
        resolved_scale = float(np.asarray(potential_scale))
    _positive_scalar("potential_scale", resolved_scale)
    if potential_max is None:
        resolved_max = 1.25 * max(float(np.max(initial_host[mask_host])), resolved_scale)
    else:
        resolved_max = float(np.asarray(potential_max))
    _positive_scalar("potential_max", resolved_max)
    if np.any(initial_host[mask_host] > resolved_max):
        raise ValueError("initial_potential exceeds potential_max inside the mask")
    fixed_exterior_max = (
        float(np.max(fixed_host[~mask_host])) if np.any(~mask_host) else 0.0
    )
    maximum_modeled_potential = max(resolved_max, fixed_exterior_max)
    max_phase = (
        float(np.asarray(interaction_constant(energy)))
        * float(np.asarray(slice_thickness))
        * maximum_modeled_potential
    )
    if max_phase >= np.pi:
        raise ValueError(
            "the optimized or fixed potential violates the per-slice phase bound: "
            f"sigma * slice_thickness * max_potential = {max_phase:.6g} >= pi"
        )

    validation_host = np.asarray(validation_indices)
    if validation_host.ndim != 1 or (
        validation_host.size and not np.issubdtype(validation_host.dtype, np.integer)
    ):
        raise TypeError("validation_indices must be a one-dimensional integer sequence")
    validation_host = validation_host.astype(np.int64, copy=False)
    if (
        np.unique(validation_host).size != validation_host.size
        or np.any(validation_host < 0)
        or np.any(validation_host >= n_scan)
    ):
        raise ValueError("validation_indices must be unique valid scan indices")
    training_host = np.setdiff1d(np.arange(n_scan), validation_host, assume_unique=True)
    if training_host.size == 0:
        raise ValueError("at least one scan must remain for training")

    flat_indices_host = np.flatnonzero(mask_host).astype(np.int32)
    flat_indices = jnp.asarray(flat_indices_host)
    scale = jnp.asarray(resolved_scale, dtype=jnp.result_type(initial, jnp.float32))
    upper_normalized = jnp.asarray(resolved_max / resolved_scale, dtype=scale.dtype)
    values = jnp.asarray(initial_host.reshape(-1)[flat_indices_host] / resolved_scale)

    if axial_coordinates is None:
        coordinates_s = jnp.arange(n_s, dtype=scale.dtype) * slice_thickness
    else:
        coordinates_s = _array("axial_coordinates", axial_coordinates, 1)
        if coordinates_s.shape[0] != n_s:
            raise ValueError("axial_coordinates must have length n_s")
    if transverse_coordinates is None:
        coordinates_u = jnp.arange(n_u, dtype=scale.dtype)
    else:
        coordinates_u = _array("transverse_coordinates", transverse_coordinates, 1)
        if coordinates_u.shape[0] != n_u:
            raise ValueError("transverse_coordinates must have length n_u")
    if scan_coordinates is None:
        coordinates_scan = coordinates_s[starts + length // 2]
    else:
        coordinates_scan = _array("scan_coordinates", scan_coordinates, 1)
        if coordinates_scan.shape[0] != n_scan:
            raise ValueError("scan_coordinates must have length n_scan")
    if detector_angles is None:
        du = _pixel_spacing(coordinates_u)
        frequencies = jnp.fft.fftshift(jnp.fft.fftfreq(n_u, du))
        detector_theta = 1e3 * jnp.arcsin(
            jnp.clip(energy2wavelength(energy) * frequencies, -1.0, 1.0)
        )
    else:
        detector_theta = _array("detector_angles", detector_angles, 1)
        if detector_theta.shape[0] != n_u:
            raise ValueError("detector_angles must have length n_u")

    fixed_flat = fixed.reshape(-1)

    def assemble(normalized_values: Array) -> Array:
        flat = fixed_flat.at[flat_indices].set(normalized_values * scale)
        return flat.reshape((n_s, n_u))

    probe_rows = (
        jnp.broadcast_to(probe, (n_scan, n_u))
        if probe.ndim == 1
        else probe
    )

    def batch_loss(
        normalized_values: Array,
        batch_starts: Array,
        batch_probes: Array,
        batch_measured: Array,
    ) -> Array:
        prediction = simulate_glancing_scan_1d(
            assemble(normalized_values),
            batch_probes,
            batch_starts,
            length,
            kernel,
            slice_thickness,
            energy,
            rematerialize=rematerialize,
        )
        return normalized_amplitude_loss_1d(prediction, batch_measured, epsilon=epsilon)

    batch_value_and_grad = jax.jit(jax.value_and_grad(batch_loss))
    predict_batch = jax.jit(
        lambda normalized_values, batch_starts, batch_probes: simulate_glancing_scan_1d(
            assemble(normalized_values),
            batch_probes,
            batch_starts,
            length,
            kernel,
            slice_thickness,
            energy,
            rematerialize=rematerialize,
        )
    )

    alpha = float(np.asarray(learning_rate_end)) / float(np.asarray(learning_rate_start))
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate_start,
        decay_steps=max(n_updates, 1),
        alpha=alpha,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(gradient_clip), optax.adam(schedule))
    optimizer_state = optimizer.init(values)
    rng = np.random.default_rng(seed_value)

    def predict_indices(normalized_values: Array, indices: np.ndarray) -> Array:
        predictions = []
        for begin in range(0, len(indices), eval_batch_size):
            batch_indices = indices[begin : begin + eval_batch_size]
            predictions.append(
                predict_batch(
                    normalized_values,
                    starts[jnp.asarray(batch_indices)],
                    probe_rows[jnp.asarray(batch_indices)],
                )
            )
        return jnp.concatenate(predictions, axis=0)

    def evaluate(normalized_values: Array, indices: np.ndarray) -> float:
        prediction = predict_indices(normalized_values, indices)
        return float(
            np.asarray(
                normalized_amplitude_loss_1d(
                    prediction,
                    measured[jnp.asarray(indices)],
                    epsilon=epsilon,
                )
            )
        )

    update_history: list[int] = []
    elapsed_history: list[float] = []
    training_history: list[float] = []
    validation_history: list[float] = []
    optimization_start = perf_counter()

    def record(update: int, normalized_values: Array) -> tuple[float, float]:
        training_loss = evaluate(normalized_values, training_host)
        validation_loss = (
            evaluate(normalized_values, validation_host)
            if validation_host.size
            else float("nan")
        )
        update_history.append(update)
        elapsed_history.append(perf_counter() - optimization_start)
        training_history.append(training_loss)
        validation_history.append(validation_loss)
        return training_loss, validation_loss

    training_loss, validation_loss = record(0, values)
    best_metric = validation_loss if validation_host.size else training_loss
    best_values = values
    best_update = 0

    for update in _update_iterator(
        n_updates,
        progress=progress,
        description=progress_description,
    ):
        batch_indices = rng.choice(
            training_host,
            size=batch_size,
            replace=training_host.size < batch_size,
        )
        _, gradient = batch_value_and_grad(
            values,
            starts[jnp.asarray(batch_indices)],
            probe_rows[jnp.asarray(batch_indices)],
            measured[jnp.asarray(batch_indices)],
        )
        parameter_updates, optimizer_state = optimizer.update(
            gradient,
            optimizer_state,
            values,
        )
        values = optax.apply_updates(values, parameter_updates)
        values = jnp.clip(values, 0.0, upper_normalized)

        if update % metric_interval == 0 or update == n_updates:
            training_loss, validation_loss = record(update, values)
            metric = validation_loss if validation_host.size else training_loss
            if np.isfinite(metric) and metric < best_metric:
                best_metric = metric
                best_values = values
                best_update = update

    best_potential = assemble(best_values)
    initial_global = assemble(
        jnp.asarray(initial_host.reshape(-1)[flat_indices_host] / resolved_scale)
    )
    all_indices = np.arange(n_scan, dtype=np.int64)
    predicted = predict_indices(best_values, all_indices)
    metadata = {
        "energy_eV": float(np.asarray(energy)),
        "slice_thickness_A": float(np.asarray(slice_thickness)),
        "potential_scale_V": resolved_scale,
        "potential_max_V": resolved_max,
        "maximum_phase_per_slice_rad": max_phase,
        "updates": n_updates,
        "minibatch_size": batch_size,
        "validation_interval": metric_interval,
        "evaluation_batch_size": eval_batch_size,
        "learning_rate_start": float(np.asarray(learning_rate_start)),
        "learning_rate_end": float(np.asarray(learning_rate_end)),
        "gradient_clip": float(np.asarray(gradient_clip)),
        "training_indices": training_host.tolist(),
        "validation_indices": validation_host.tolist(),
        "n_unknown_pixels": int(flat_indices_host.size),
        "uses_fixed_potential": fixed_potential is not None,
        "fixed_exterior_max_V": fixed_exterior_max,
        "best_metric": best_metric,
        "detector_angle_unit": "mrad",
    }
    return PotentialReconstruction1D(
        potential=best_potential,
        initial_potential=initial_global,
        reconstruction_mask=mask,
        axial_coordinates=coordinates_s,
        transverse_coordinates=coordinates_u,
        predicted_intensities=predicted,
        measured_intensities=measured,
        window_starts=starts,
        scan_coordinates=coordinates_scan,
        detector_angles=detector_theta,
        update_history=jnp.asarray(update_history),
        elapsed_time_history=jnp.asarray(elapsed_history),
        training_loss_history=jnp.asarray(training_history),
        validation_loss_history=jnp.asarray(validation_history),
        best_update=best_update,
        metadata=metadata,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        array = np.asarray(value)
    except Exception as exc:  # pragma: no cover - input-specific error path
        raise TypeError(f"metadata value {value!r} is not JSON serializable") from exc
    return array.item() if array.ndim == 0 else array.tolist()


def _metadata_json(metadata: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(dict(metadata), default=_json_default, sort_keys=True))


def _save_npz(path: str | Path, **arrays: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def save_glancing_scan_1d(path: str | Path, scan: GlancingScan1D) -> None:
    """Save a scan with non-pickled JSON metadata."""
    _save_npz(
        path,
        intensities=np.asarray(scan.intensities),
        window_starts=np.asarray(scan.window_starts),
        scan_coordinates=np.asarray(scan.scan_coordinates),
        detector_angles=np.asarray(scan.detector_angles),
        metadata_json=_metadata_json(scan.metadata),
    )


def load_glancing_scan_1d(path: str | Path) -> GlancingScan1D:
    """Load a scan written by :func:`save_glancing_scan_1d`."""
    with np.load(path, allow_pickle=False) as data:
        return GlancingScan1D(
            intensities=jnp.asarray(data["intensities"]),
            window_starts=jnp.asarray(data["window_starts"]),
            scan_coordinates=jnp.asarray(data["scan_coordinates"]),
            detector_angles=jnp.asarray(data["detector_angles"]),
            metadata=json.loads(str(data["metadata_json"].item())),
        )


def save_glancing_sideview_cache_1d(
    path: str | Path,
    cache: GlancingSideviewCache1D,
) -> None:
    """Save a compact selected-scan side-view cache."""
    _save_npz(
        path,
        scan_indices=np.asarray(cache.scan_indices),
        window_starts=np.asarray(cache.window_starts),
        scan_coordinates=np.asarray(cache.scan_coordinates),
        local_s_coordinates=np.asarray(cache.local_s_coordinates),
        sideview_u_coordinates=np.asarray(cache.sideview_u_coordinates),
        transverse_coordinates=np.asarray(cache.transverse_coordinates),
        sideview_wavefields=np.asarray(cache.sideview_wavefields),
        sideview_intensities=np.asarray(cache.sideview_intensities),
        exit_waves=np.asarray(cache.exit_waves),
        detector_waves=np.asarray(cache.detector_waves),
        detector_intensities=np.asarray(cache.detector_intensities),
        metadata_json=_metadata_json(cache.metadata),
    )


def load_glancing_sideview_cache_1d(path: str | Path) -> GlancingSideviewCache1D:
    """Load a cache written by :func:`save_glancing_sideview_cache_1d`."""
    with np.load(path, allow_pickle=False) as data:
        return GlancingSideviewCache1D(
            scan_indices=jnp.asarray(data["scan_indices"]),
            window_starts=jnp.asarray(data["window_starts"]),
            scan_coordinates=jnp.asarray(data["scan_coordinates"]),
            local_s_coordinates=jnp.asarray(data["local_s_coordinates"]),
            sideview_u_coordinates=jnp.asarray(data["sideview_u_coordinates"]),
            transverse_coordinates=jnp.asarray(data["transverse_coordinates"]),
            sideview_wavefields=jnp.asarray(data["sideview_wavefields"]),
            sideview_intensities=jnp.asarray(data["sideview_intensities"]),
            exit_waves=jnp.asarray(data["exit_waves"]),
            detector_waves=jnp.asarray(data["detector_waves"]),
            detector_intensities=jnp.asarray(data["detector_intensities"]),
            metadata=json.loads(str(data["metadata_json"].item())),
        )


def save_potential_reconstruction_1d(
    path: str | Path,
    result: PotentialReconstruction1D,
) -> None:
    """Save a direct-potential reconstruction with JSON metadata."""
    _save_npz(
        path,
        potential=np.asarray(result.potential),
        initial_potential=np.asarray(result.initial_potential),
        reconstruction_mask=np.asarray(result.reconstruction_mask),
        axial_coordinates=np.asarray(result.axial_coordinates),
        transverse_coordinates=np.asarray(result.transverse_coordinates),
        predicted_intensities=np.asarray(result.predicted_intensities),
        measured_intensities=np.asarray(result.measured_intensities),
        window_starts=np.asarray(result.window_starts),
        scan_coordinates=np.asarray(result.scan_coordinates),
        detector_angles=np.asarray(result.detector_angles),
        update_history=np.asarray(result.update_history),
        elapsed_time_history=np.asarray(result.elapsed_time_history),
        training_loss_history=np.asarray(result.training_loss_history),
        validation_loss_history=np.asarray(result.validation_loss_history),
        best_update=np.asarray(result.best_update, dtype=np.int64),
        metadata_json=_metadata_json(result.metadata),
    )


def load_potential_reconstruction_1d(path: str | Path) -> PotentialReconstruction1D:
    """Load a result written by :func:`save_potential_reconstruction_1d`."""
    with np.load(path, allow_pickle=False) as data:
        return PotentialReconstruction1D(
            potential=jnp.asarray(data["potential"]),
            initial_potential=jnp.asarray(data["initial_potential"]),
            reconstruction_mask=jnp.asarray(data["reconstruction_mask"]),
            axial_coordinates=jnp.asarray(data["axial_coordinates"]),
            transverse_coordinates=jnp.asarray(data["transverse_coordinates"]),
            predicted_intensities=jnp.asarray(data["predicted_intensities"]),
            measured_intensities=jnp.asarray(data["measured_intensities"]),
            window_starts=jnp.asarray(data["window_starts"]),
            scan_coordinates=jnp.asarray(data["scan_coordinates"]),
            detector_angles=jnp.asarray(data["detector_angles"]),
            update_history=jnp.asarray(data["update_history"]),
            elapsed_time_history=jnp.asarray(
                data["elapsed_time_history"]
                if "elapsed_time_history" in data.files
                else np.zeros_like(data["update_history"], dtype=float)
            ),
            training_loss_history=jnp.asarray(data["training_loss_history"]),
            validation_loss_history=jnp.asarray(data["validation_loss_history"]),
            best_update=int(data["best_update"].item()),
            metadata=json.loads(str(data["metadata_json"].item())),
        )
