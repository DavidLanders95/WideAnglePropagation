"""Convenience helpers used by the example notebooks.

These functions handle plotting coordinates, compact result files, and small
notebook workflow utilities. They are kept separate from the core propagation
kernels so the numerical implementation remains easier to read.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .propagation_methods import (
    Sampling,
    _slice_phase_grating,
    electron_refractive_index,
    fourier_propagate,
    wpm_step_adaptive,
)

__all__ = [
    "angle_crop_slices",
    "assert_scattering_support",
    "beam_amplitude_normalized",
    "beam_for_angle_mrad",
    "cbed_amplitude",
    "curve_rmse",
    "diffraction_angle_axes_mrad",
    "diffraction_pattern_numpy",
    "global_phase_removed",
    "grid_from_pixel_size",
    "load_cbed_results",
    "make_kirkland_probe",
    "masked_angle_crop",
    "method_key",
    "objective_lens_transfer_function",
    "radial_integrated_profiles",
    "relative_rmse",
    "resolve_paper_figures_dir",
    "resolve_repo_root",
    "save_cbed_results",
    "shared_percentile_limits",
    "simulate_fresnel_as_exit_only",
    "simulate_wpm_exit_only",
    "validate_cbed_results",
]


def resolve_repo_root(start: str | Path | None = None) -> Path:
    """Return the repository root containing notebooks and package sources."""
    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()
    for base in (current, *current.parents):
        if (base / "notebooks").exists() and (base / "wide_angle_propagation").exists():
            return base
    return current


def resolve_paper_figures_dir(start: str | Path | None = None) -> Path:
    """Return ``Paper/figures``, creating it when the paper directory exists."""
    current = Path.cwd() if start is None else Path(start)
    current = current.resolve()
    for base in (current, *current.parents):
        candidate = base / "Paper" / "figures"
        if candidate.parent.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    raise FileNotFoundError("Could not locate Paper/figures from the current directory")


def beam_amplitude_normalized(psi_xy, h: int, k: int, use_fftshift: bool = True) -> float:
    """Return normalized Fourier amplitude for beam index ``(h, k)``."""
    ny, nx = psi_xy.shape
    coefficients = np.fft.fft2(psi_xy) / (nx * ny)
    if use_fftshift:
        coefficients = np.fft.fftshift(coefficients)
        cy, cx = ny // 2, nx // 2
        return float(np.abs(coefficients[cy + k, cx + h]))
    return float(np.abs(coefficients[k % ny, h % nx]))


def grid_from_pixel_size(atoms, pixel_size_y: float, pixel_size_x: float):
    """Return nearest integer grid and exact sampling for an ASE cell."""
    cell = atoms.get_cell()
    length_y = float(cell[0, 0])
    length_x = float(cell[1, 1])
    ny = max(1, int(np.rint(length_y / float(pixel_size_y))))
    nx = max(1, int(np.rint(length_x / float(pixel_size_x))))
    return (ny, nx), (length_y / ny, length_x / nx)


def beam_for_angle_mrad(shape, sampling, wavelength, target_angle_mrad, axis="y"):
    """Return the nearest FFT beam index for a target scattering angle."""
    ny, nx = shape
    dy, dx = sampling
    if axis == "y":
        n, d = ny, dy
    elif axis == "x":
        n, d = nx, dx
    else:
        raise ValueError("axis must be 'x' or 'y'")

    target_frequency = float(target_angle_mrad) / (float(wavelength) * 1000.0)
    index = int(np.rint(target_frequency * n * d))
    max_offset = (n - 1) // 2
    h, k = (index, 0) if axis == "x" else (0, index)
    return {
        "h": h,
        "k": k,
        "index": index,
        "axis": axis,
        "visible": abs(index) <= max_offset,
        "actual_mrad": float(wavelength) * (index / (n * d)) * 1000.0,
        "max_visible_mrad": float(wavelength) * (max_offset / (n * d)) * 1000.0,
        "target_mrad": float(target_angle_mrad),
    }


def curve_rmse(a, b) -> float:
    """Return root-mean-square error between two curves."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def relative_rmse(values, reference) -> float:
    """Return RMSE normalized by the RMS of ``reference``."""
    values = np.asarray(values)
    reference = np.asarray(reference)
    numerator = np.sqrt(np.mean((values - reference) ** 2))
    denominator = max(np.sqrt(np.mean(reference**2)), 1e-30)
    return float(numerator / denominator)


def simulate_fresnel_as_exit_only(potential, wave, propagation_kernel_, slice_thickness, energy):
    """Run Fresnel/AS multislice propagation and return only the exit wave."""
    if potential.shape[0] == 0:
        return wave
    wave = jnp.asarray(
        wave,
        dtype=jnp.result_type(wave, propagation_kernel_, potential, 1j),
    )

    def step(wavefront, potential_slice):
        wavefront = wavefront * _slice_phase_grating(potential_slice, slice_thickness, energy)
        wavefront = fourier_propagate(wavefront, propagation_kernel_)
        return wavefront, None

    return jax.lax.scan(step, wave, potential)[0]


def simulate_wpm_exit_only(
    potential,
    wave,
    slice_thickness,
    energy,
    sampling: Sampling,
    n_bins: int = 128,
    power_spacing: float = 2.0,
    bin_batch_size: int | None = None,
):
    """Run WPM multislice propagation and return only the exit wave."""
    if potential.shape[0] == 0:
        return wave
    target_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    wave = jnp.asarray(wave, dtype=target_dtype)

    def step(wavefront, potential_slice):
        refractive_index = electron_refractive_index(potential_slice, energy)
        wavefront, _, _, _ = wpm_step_adaptive(
            wavefront,
            refractive_index,
            slice_thickness,
            energy,
            sampling,
            n_bins=n_bins,
            power_spacing=power_spacing,
            bin_batch_size=bin_batch_size,
        )
        return wavefront, None

    return jax.lax.scan(step, wave, potential)[0]


def diffraction_angle_axes_mrad(ny: int, nx: int, sampling, wavelength):
    """Return fftshifted scattering-angle axes in mrad."""
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=sampling[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=sampling[1]))
    theta_y = 1e3 * np.arcsin(np.clip(float(wavelength) * fy, -1.0, 1.0))
    theta_x = 1e3 * np.arcsin(np.clip(float(wavelength) * fx, -1.0, 1.0))
    return theta_y, theta_x


def assert_scattering_support(sampling, wavelength, max_angle_mrad) -> float:
    """Validate that sampling supports the requested scattering angle."""
    max_sampling = max(float(sampling[0]), float(sampling[1]))
    nyquist_mrad = 1e3 * np.arcsin(
        np.clip(float(wavelength) / (2.0 * max_sampling), 0.0, 1.0)
    )
    if nyquist_mrad < max_angle_mrad:
        raise ValueError(
            f"Grid supports {nyquist_mrad:.1f} mrad, below requested "
            f"{max_angle_mrad:.1f} mrad. Reduce the real-space sampling."
        )
    return float(nyquist_mrad)


def angle_crop_slices(theta_y, theta_x, cutoff_mrad):
    """Return row/column slices inside a square angular crop."""
    y_idx = np.where(np.abs(theta_y) <= cutoff_mrad)[0]
    x_idx = np.where(np.abs(theta_x) <= cutoff_mrad)[0]
    if y_idx.size == 0 or x_idx.size == 0:
        raise ValueError(f"No pixels inside +/- {cutoff_mrad} mrad crop")
    return slice(y_idx[0], y_idx[-1] + 1), slice(x_idx[0], x_idx[-1] + 1)


def masked_angle_crop(pattern, theta_y, theta_x, cutoff_mrad):
    """Return an angular crop masked to a circular cutoff plus imshow extent."""
    y_slice, x_slice = angle_crop_slices(theta_y, theta_x, cutoff_mrad)
    crop = np.asarray(pattern)[y_slice, x_slice].astype(np.float64)
    tx_grid, ty_grid = np.meshgrid(theta_x[x_slice], theta_y[y_slice])
    radius = np.sqrt(tx_grid**2 + ty_grid**2)
    crop = np.where(radius <= cutoff_mrad, crop, np.nan)
    extent = [
        theta_x[x_slice.start],
        theta_x[x_slice.stop - 1],
        theta_y[y_slice.start],
        theta_y[y_slice.stop - 1],
    ]
    return crop, extent


def radial_integrated_profiles(
    patterns_by_method,
    theta_y,
    theta_x,
    max_angle_mrad,
    bin_width_mrad,
):
    """Return annular sums and normalized radial profiles for CBED patterns."""
    theta_x_grid, theta_y_grid = np.meshgrid(theta_x, theta_y)
    radius = np.sqrt(theta_x_grid**2 + theta_y_grid**2)
    valid = radius <= max_angle_mrad
    bin_edges = np.arange(0.0, max_angle_mrad + bin_width_mrad, bin_width_mrad)
    if bin_edges[-1] < max_angle_mrad:
        bin_edges = np.append(bin_edges, max_angle_mrad)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    radial_sums = {}
    radial_profiles = {}
    for method, stack in patterns_by_method.items():
        method_sums = []
        method_profiles = []
        for pattern in stack:
            annular_sum, _ = np.histogram(
                radius[valid],
                bins=bin_edges,
                weights=np.asarray(pattern)[valid],
            )
            annular_sum = annular_sum.astype(np.float64)
            total = float(annular_sum.sum())
            profile = annular_sum / total if total > 0.0 else annular_sum
            method_sums.append(annular_sum)
            method_profiles.append(profile)
        radial_sums[method] = np.stack(method_sums)
        radial_profiles[method] = np.stack(method_profiles)
    return bin_edges, bin_centers, radial_sums, radial_profiles


def method_key(method_name: str) -> str:
    """Return a filesystem/NPZ-safe key for a method label."""
    return method_name.lower().replace(" ", "_").replace("-", "_")


def diffraction_pattern_numpy(exit_wave):
    """Return fftshifted far-field intensity as a NumPy array."""
    wave = np.asarray(exit_wave)
    return np.abs(np.fft.fftshift(np.fft.fft2(wave))) ** 2


def cbed_amplitude(pattern_intensity):
    """Return CBED amplitude from a non-negative diffraction intensity pattern."""
    return np.sqrt(np.maximum(np.asarray(pattern_intensity), 0.0))


def make_kirkland_probe(ny, nx, sampling, wavelength, semiangle_mrad, defocus, cs):
    """Build a Kirkland-style convergent probe on an unshifted FFT grid."""
    fy = np.fft.fftfreq(ny, d=sampling[0])
    fx = np.fft.fftfreq(nx, d=sampling[1])
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    spatial_frequency = np.sqrt(fx_grid**2 + fy_grid**2)
    alpha = np.arcsin(np.clip(float(wavelength) * spatial_frequency, 0.0, 1.0))

    aperture = alpha <= semiangle_mrad * 1e-3
    if not np.any(aperture):
        raise ValueError(f"No aperture pixels for {semiangle_mrad} mrad probe")

    aberration_length = 0.5 * defocus * alpha**2 + 0.25 * cs * alpha**4
    aberration_phase = np.exp(-1j * (2.0 * np.pi / float(wavelength)) * aberration_length)
    probe_fft = aperture.astype(np.complex128) * aberration_phase
    probe_fft /= np.linalg.norm(probe_fft)
    return jnp.array(np.fft.fftshift(np.fft.ifft2(probe_fft)), dtype=jnp.complex128)


def objective_lens_transfer_function(
    shape,
    sampling,
    wavelength,
    defocus,
    cs,
    aperture_mrad,
    focal_spread_nm,
):
    """Return a representative objective-lens transfer function."""
    ky = np.fft.fftfreq(shape[0], d=sampling[0])
    kx = np.fft.fftfreq(shape[1], d=sampling[1])
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    spatial_frequency_squared = kx_grid**2 + ky_grid**2
    alpha = float(wavelength) * np.sqrt(spatial_frequency_squared)

    chi = (
        np.pi * float(wavelength) * defocus * spatial_frequency_squared
        + 0.5 * np.pi * cs * float(wavelength) ** 3 * spatial_frequency_squared**2
    )
    aperture = alpha <= aperture_mrad * 1e-3
    focal_spread = focal_spread_nm * 10.0
    temporal_envelope = np.exp(
        -0.5 * (np.pi * float(wavelength) * focal_spread * spatial_frequency_squared) ** 2
    )
    return aperture * temporal_envelope * np.exp(-1.0j * chi)


def global_phase_removed(wave):
    """Remove arbitrary mean phase for phase visualization."""
    return wave * np.exp(-1.0j * np.angle(np.mean(wave)))


def shared_percentile_limits(arrays, lower=0.5, upper=99.5):
    """Return shared percentile color limits across a collection of arrays."""
    values = np.concatenate([np.asarray(array).ravel() for array in arrays])
    vmin, vmax = np.percentile(values, [lower, upper])
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def validate_cbed_results(results):
    """Validate shape and finite values in a CBED result dictionary."""
    n_thicknesses = len(results["actual_thicknesses_A"])
    for method in results["method_names"]:
        pattern_stack = results["patterns"][method]
        radial_stack = results["radial_profiles"][method]
        if pattern_stack.shape[0] != n_thicknesses:
            raise ValueError(
                f"{method}: expected {n_thicknesses} patterns, got {pattern_stack.shape[0]}"
            )
        if radial_stack.shape[0] != n_thicknesses:
            raise ValueError(
                f"{method}: expected {n_thicknesses} radial profiles, got {radial_stack.shape[0]}"
            )
        if not np.all(np.isfinite(pattern_stack)):
            raise ValueError(f"{method}: non-finite CBED pattern values")
        if not np.all(np.isfinite(radial_stack)):
            raise ValueError(f"{method}: non-finite radial profile values")
    return True


def save_cbed_results(path, cbed_results):
    """Save compact CBED result arrays to ``.npz``."""
    arrays = {
        "method_names": np.array(cbed_results["method_names"], dtype=object),
        "target_thicknesses_A": np.asarray(cbed_results["target_thicknesses_A"], dtype=float),
        "actual_thicknesses_A": np.asarray(cbed_results["actual_thicknesses_A"], dtype=float),
        "cell_repeats": np.asarray(cbed_results["cell_repeats"], dtype=int),
        "gpts": np.asarray(cbed_results["gpts"], dtype=int),
        "sampling_A": np.asarray(cbed_results["sampling_A"], dtype=float),
        "theta_y_mrad": np.asarray(cbed_results["theta_y_mrad"], dtype=float),
        "theta_x_mrad": np.asarray(cbed_results["theta_x_mrad"], dtype=float),
        "radial_bin_edges_mrad": np.asarray(cbed_results["radial_bin_edges_mrad"], dtype=float),
        "radial_bin_centers_mrad": np.asarray(
            cbed_results["radial_bin_centers_mrad"],
            dtype=float,
        ),
        "metadata": np.array([cbed_results["metadata"]], dtype=object),
    }
    for method in cbed_results["method_names"]:
        key = method_key(method)
        arrays[f"{key}_patterns"] = np.asarray(cbed_results["patterns"][method], dtype=np.float32)
        arrays[f"{key}_radial_sums"] = np.asarray(
            cbed_results["radial_sums"][method],
            dtype=np.float64,
        )
        arrays[f"{key}_radial_profiles"] = np.asarray(
            cbed_results["radial_profiles"][method],
            dtype=np.float64,
        )
        arrays[f"{key}_runtime_at_targets_s"] = np.asarray(
            cbed_results["runtime_at_targets_s"][method],
            dtype=float,
        )
        arrays[f"{key}_cell_runtime_s"] = np.asarray(
            cbed_results["cell_runtime_s"][method],
            dtype=float,
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_cbed_results(path):
    """Load compact CBED result arrays from ``.npz``."""
    with np.load(path, allow_pickle=True) as data:
        loaded_methods = [str(v) for v in data["method_names"]]
        loaded = {
            "method_names": loaded_methods,
            "target_thicknesses_A": data["target_thicknesses_A"].astype(float),
            "actual_thicknesses_A": data["actual_thicknesses_A"].astype(float),
            "cell_repeats": data["cell_repeats"].astype(int),
            "gpts": tuple(int(v) for v in data["gpts"]),
            "sampling_A": tuple(float(v) for v in data["sampling_A"]),
            "theta_y_mrad": data["theta_y_mrad"].astype(float),
            "theta_x_mrad": data["theta_x_mrad"].astype(float),
            "radial_bin_edges_mrad": data["radial_bin_edges_mrad"].astype(float),
            "radial_bin_centers_mrad": data["radial_bin_centers_mrad"].astype(float),
            "metadata": dict(data["metadata"][0]),
            "patterns": {},
            "radial_sums": {},
            "radial_profiles": {},
            "runtime_at_targets_s": {},
            "cell_runtime_s": {},
        }
        for method in loaded_methods:
            key = method_key(method)
            loaded["patterns"][method] = data[f"{key}_patterns"].copy()
            loaded["radial_sums"][method] = data[f"{key}_radial_sums"].copy()
            loaded["radial_profiles"][method] = data[f"{key}_radial_profiles"].copy()
            loaded["runtime_at_targets_s"][method] = data[
                f"{key}_runtime_at_targets_s"
            ].copy()
            loaded["cell_runtime_s"][method] = data[f"{key}_cell_runtime_s"].copy()
    return loaded
