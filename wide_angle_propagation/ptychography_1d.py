"""Differentiable forward primitives for 1D glancing-incidence ptychography."""

from __future__ import annotations

from dataclasses import dataclass, field
import operator
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .propagation_methods import energy2wavelength, interaction_constant


__all__ = [
    "GlancingSideviewCache1D",
    "normalized_amplitude_loss_1d",
    "simulate_glancing_scan_1d",
    "simulate_glancing_sideview_cache_1d",
]


Array = Any


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
