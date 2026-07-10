"""Propagation kernels and simulation helpers used by the paper workflows.

The module is organized around the maintained comparison in the paper:

* electron optics conversions
* Fresnel and angular-spectrum propagation kernels
* adaptive-binned wave-propagation multislice
* full second-order Klein-Gordon ODE reference propagation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
from ase import units


Sampling = tuple[float, float]

__all__ = [
    "Sampling",
    "angular_spectrum_propagation_kernel",
    "angular_spectrum_propagation_kernel_1d",
    "apply_pade_rational_1d",
    "bidirectional_pade_sweep_1d",
    "bidirectional_wpm_sweep_1d",
    "build_sideview_operator_x_1d",
    "cylindrical_green_asymptotic_1d",
    "diffraction_intensity",
    "diffraction_intensity_1d",
    "electron_refractive_index",
    "electron_rest_energy",
    "energy2wavelength",
    "fourier_propagate_1d",
    "fourier_propagate",
    "fresnel_propagation_kernel_1d",
    "fresnel_propagation_kernel",
    "get_frequencies_1d",
    "interface_coupling_wpm_1d",
    "interface_scattering_matrix_1d",
    "klein_gordon_refractive_index_1d",
    "line_to_line_cylindrical_propagate_1d",
    "rayleigh_sommerfeld_propagate_1d",
    "make_gaussian_atom_potential_sideview_1d",
    "pade_backward_step_1d",
    "pade_forward_step_1d",
    "pade_sqrt_coefficients",
    "phase_grating_1d_from_projected_potential",
    "project_atoms_to_sample_line_1d",
    "project_potential_to_sample_line_1d",
    "schrodinger_refractive_index_1d",
    "simulate_fresnel_as",
    "simulate_fresnel_as_jit",
    "simulate_bidirectional_pade_bpm_1d",
    "simulate_bidirectional_wpm_1d",
    "simulate_glancing_angular_spectrum_1d",
    "simulate_glancing_fresnel_baseline_1d",
    "simulate_kg_ode_full",
    "simulate_single_slice_cylindrical_1d",
    "simulate_wpm",
    "simulate_wpm_jit",
    "wpm_step_adaptive",
    "wpm_step_adaptive_1d",
]


# =============================================================================
# 1. Physics Constants & Conversion Utilities
#    (Fundamental functions used by almost all other components)
# =============================================================================

def electron_rest_energy() -> float:
    """Return the electron rest energy E0 = m_e c^2 in eV."""
    m_e = units._me
    c = units._c
    eV = units._e
    return m_e * c**2 / eV


@jax.jit
def energy2wavelength(energy: float):
    """Return the relativistic de Broglie wavelength in Angstroms."""
    return (
        units._hplanck
        * units._c
        / jnp.sqrt(energy * (2 * units._me * units._c**2 / units._e + energy))
        / units._e
        * 1.0e10
    )


def electron_refractive_index(potential, energy):
    """Return electron refractive index from electrostatic potential in Volts."""
    E0 = electron_rest_energy()
    E = energy

    # Convert electrostatic potential (V) -> potential energy V (eV)
    # Electron charge is negative, V_potential_energy = -1 * V_electrostatic
    V = -potential
    EminusV = E - V

    numerator = 2 * EminusV * E0 + EminusV**2
    denominator = 2 * E * E0 + E**2

    return jnp.sqrt(numerator / denominator)


def interaction_constant(energy: float):
    """Return the relativistic TEM interaction constant in rad / (V Angstrom)."""
    wavelength_m = energy2wavelength(energy) * 1.0e-10
    gamma = 1.0 + energy / electron_rest_energy()
    sigma_m = (
        2.0
        * jnp.pi
        * gamma
        * units._me
        * units._e
        * wavelength_m
        / units._hplanck**2
    )
    return sigma_m * 1.0e-10


def schrodinger_refractive_index_1d(potential, energy):
    """Return the corrected Schrodinger electron refractive index.

    ``potential`` is an electrostatic potential in volts. Lengths are Angstroms,
    matching :func:`energy2wavelength`.
    """
    wavelength = energy2wavelength(energy)
    k0 = 1.0 / wavelength
    n_sq = 1.0 + interaction_constant(energy) * potential / (jnp.pi * k0)
    return jnp.sqrt(jnp.asarray(n_sq, dtype=jnp.complex128))


def klein_gordon_refractive_index_1d(potential, energy):
    """Return the Klein-Gordon electron refractive index for a 1D potential."""
    return electron_refractive_index(potential, energy)


# =============================================================================
# 2. Math Helpers & Grid Utilities
# =============================================================================

def get_frequencies(n: int, m: int, ps: Sampling):
    """Return FFT spatial-frequency grids for an ``(n, m)`` array.

    ``ps`` follows the row/column convention used throughout the notebooks:
    ``(dy, dx)``. The returned arrays have shape ``(n, m)``.
    """
    fy = jnp.fft.fftfreq(n, ps[0])
    fx = jnp.fft.fftfreq(m, ps[1])
    return jnp.meshgrid(fy, fx, indexing="ij")


def get_frequencies_1d(n: int, dx: float):
    """Return 1D FFT spatial frequencies in cycles per unit length."""
    return jnp.fft.fftfreq(n, dx)


def transverse_frequency_squared(shape: tuple[int, int], sampling: Sampling):
    """Return ``k_perp^2`` on the FFT grid in cycles per unit length."""
    fy, fx = get_frequencies(shape[0], shape[1], sampling)
    return fy**2 + fx**2


def diffraction_intensity(exit_wave):
    """Return fftshifted far-field intensity ``|FFT(exit_wave)|^2``."""
    detector_wave = jnp.fft.fftshift(jnp.fft.fft2(exit_wave))
    return jnp.abs(detector_wave) ** 2


def diffraction_intensity_1d(exit_wave):
    """Return fftshifted 1D far-field intensity ``|FFT(exit_wave)|^2``."""
    detector_wave = jnp.fft.fftshift(jnp.fft.fft(exit_wave))
    return jnp.abs(detector_wave) ** 2


def smoothstep(x):
    """Cubic smoothstep interpolation weight: ``3x^2 - 2x^3``."""
    x = jnp.clip(x, 0.0, 1.0)
    return 3 * x**2 - 2 * x**3


def get_polynomial_bins(n_min, n_max, n_bins, power=2.0):
    """Return refractive-index bins with optional polynomial high-end density."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if power <= 0:
        raise ValueError("power must be positive")

    t = jnp.linspace(0, 1, n_bins)
    t_warped = t**power
    return n_min + (n_max - n_min) * t_warped


# =============================================================================
# 3. Propagation Kernels
# =============================================================================

def fresnel_propagation_kernel(
    n: int,
    m: int,
    ps: Sampling,
    z: float,
    energy: float,
):
    """Return the paraxial Fresnel transfer function for distance ``z``.

    FFT frequencies are in cycles per unit length. This implementation retains
    the common vacuum carrier phase, which is a global phase relative to the
    envelope form used in the paper equations.
    """
    wavelength = energy2wavelength(energy)
    Fy, Fx = get_frequencies(n, m, ps)

    phase = jnp.exp(1j * (2 * jnp.pi / wavelength) * z)
    diffraction = jnp.exp(-1j * jnp.pi * wavelength * z * (Fx**2 + Fy**2))
    return phase * diffraction


def fresnel_propagation_kernel_1d(n: int, dx: float, z: float, energy: float):
    """Return the 1D paraxial Fresnel transfer function."""
    wavelength = energy2wavelength(energy)
    fx = get_frequencies_1d(n, dx)
    phase = jnp.exp(1j * (2.0 * jnp.pi / wavelength) * z)
    diffraction = jnp.exp(-1j * jnp.pi * wavelength * z * fx**2)
    return phase * diffraction


def angular_spectrum_propagation_kernel(
    n: int,
    m: int,
    ps: Sampling,
    z: float,
    energy: float,
):
    """Return the exact angular-spectrum transfer function for distance ``z``.

    FFT frequencies are in cycles per unit length. This implementation retains
    the common vacuum carrier phase, which is a global phase relative to the
    envelope form used in the paper equations.
    """
    wavelength = energy2wavelength(energy)
    Fy, Fx = get_frequencies(n, m, ps)
    kz_sq = (1 / wavelength) ** 2 - Fx**2 - Fy**2
    kz = jnp.sqrt(jnp.asarray(kz_sq, dtype=jnp.complex128))
    return jnp.exp(1j * 2 * jnp.pi * z * kz)


def angular_spectrum_propagation_kernel_1d(n: int, dx: float, z: float, energy: float):
    """Return the exact 1D angular-spectrum transfer function."""
    wavelength = energy2wavelength(energy)
    fx = get_frequencies_1d(n, dx)
    kz_sq = (1.0 / wavelength) ** 2 - fx**2
    kz = jnp.sqrt(jnp.asarray(kz_sq, dtype=jnp.complex128))
    return jnp.exp(1j * 2.0 * jnp.pi * z * kz)


def wpm_propagation_kernel(Ek, n_val, k0, k_perp2, dz):
    """Propagate one Fourier-space wave through one homogeneous WPM bin.

    ``k0`` and ``k_perp2`` are spatial frequencies in cycles per unit length.
    The factor ``2*pi`` is introduced only when converting the longitudinal
    spatial frequency to a phase in radians. As in the Fresnel and AS kernels,
    the common vacuum carrier is retained as a global phase.
    """
    kz = jnp.sqrt(jnp.asarray(n_val**2 * k0**2 - k_perp2, dtype=jnp.complex128))
    H = jnp.exp(1j * 2 * jnp.pi * dz * kz)
    return jnp.fft.ifft2(H * Ek)


wpm_propagation_kernel_vmap = jax.vmap(
    wpm_propagation_kernel,
    in_axes=(None, 0, None, None, None),
)


# =============================================================================
# 4. Propagation Logic (Steppers)
# =============================================================================

def fourier_propagate(field, transfer_function):
    """Propagate ``field`` by multiplying its FFT by ``transfer_function``."""
    return jnp.fft.ifft2(transfer_function * jnp.fft.fft2(field))


def fourier_propagate_1d(field, transfer_function):
    """Propagate a 1D field by multiplying its FFT by ``transfer_function``."""
    return jnp.fft.ifft(transfer_function * jnp.fft.fft(field))


def _stack_wavefronts_or_empty_1d(wavefronts, reference_wave):
    """Return saved 1D wavefronts with a stable empty-stack shape."""
    if wavefronts:
        return jnp.stack(wavefronts)
    return jnp.empty((0, reference_wave.shape[0]), dtype=reference_wave.dtype)


def _as_complex_wave_1d(wave):
    return jnp.asarray(wave, dtype=jnp.complex128)


def _trapz_weights(coords):
    coords = jnp.asarray(coords)
    n = coords.shape[0]
    if n < 2:
        return jnp.ones_like(coords)
    ds = jnp.diff(coords)
    first = ds[0] / 2.0
    last = ds[-1] / 2.0
    middle = (ds[:-1] + ds[1:]) / 2.0
    return jnp.concatenate([first[None], middle, last[None]])


def cylindrical_green_asymptotic_1d(R, energy, *, eps: float = 1e-12):
    """Unnormalised high-frequency 2D cylindrical Green-function shape.

    This helper is retained for source-integral experiments. It is not, by
    itself, a Rayleigh--Sommerfeld propagation kernel; use
    :func:`rayleigh_sommerfeld_propagate_1d` when the input is a boundary
    field.
    """
    wavelength = energy2wavelength(energy)
    k = 2.0 * jnp.pi / wavelength
    R_safe = jnp.maximum(jnp.asarray(R), eps)
    return jnp.exp(1j * (k * R_safe - jnp.pi / 4.0)) / jnp.sqrt(R_safe)


def line_to_line_cylindrical_propagate_1d(
    source_wave,
    source_line,
    target_line,
    energy,
    *,
    quadrature="trapezoid",
    green_kernel="cylindrical_asymptotic",
):
    """Propagate a 1D sideview field between two sampled lines."""
    if quadrature != "trapezoid":
        raise ValueError("Only trapezoid quadrature is currently implemented")
    if green_kernel != "cylindrical_asymptotic":
        raise ValueError("Only the cylindrical_asymptotic kernel is implemented")

    source_points = source_line.points()
    target_points = target_line.points()
    delta = target_points[:, None, :] - source_points[None, :, :]
    R = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
    weights = _trapz_weights(source_line.coords)
    kernel = cylindrical_green_asymptotic_1d(R, energy)
    return kernel @ (_as_complex_wave_1d(source_wave) * weights)


def _rayleigh_sommerfeld_asymptotic_kernel_1d(
    delta,
    source_normal,
    energy,
    *,
    eps: float = 1e-12,
):
    """Return the forward 2D Rayleigh--Sommerfeld kernel in the far field.

    For an outgoing scalar 2D Helmholtz wave, the field-only
    Rayleigh--Sommerfeld kernel on a planar boundary is the normal derivative
    of the Dirichlet image Green function. Its large-``k R`` limit is

    ``sqrt(k / (2 pi R)) exp(i (k R - pi/4)) (n_source . R_hat)``.

    The final factor is the obliquity factor. It rejects the backward
    half-space and, unlike a direct ``G U`` source integral, has the correct
    scalar-wave dimensions and normalization. The approximation is accurate
    when every retained source-target pair satisfies ``k R >> 1``.
    """
    delta = jnp.asarray(delta)
    source_normal = jnp.asarray(source_normal)
    R = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
    R_safe = jnp.maximum(R, eps)
    wavelength = energy2wavelength(energy)
    k = 2.0 * jnp.pi / wavelength
    obliquity = jnp.sum(delta * source_normal[None, None, :], axis=-1) / R_safe
    kernel = (
        jnp.sqrt(k / (2.0 * jnp.pi * R_safe))
        * jnp.exp(1j * (k * R_safe - jnp.pi / 4.0))
        * obliquity
    )
    return jnp.where(obliquity > 0.0, kernel, 0.0 + 0.0j)


def rayleigh_sommerfeld_propagate_1d(
    source_wave,
    source_line,
    target_line,
    energy,
    *,
    source_normal=None,
    quadrature="trapezoid",
):
    """Propagate a boundary field with a forward 2D RS asymptotic integral.

    ``source_wave`` is interpreted as the scalar field on ``source_line``,
    not as an arbitrary line-source density. ``source_normal`` selects the
    outgoing half-space; it defaults to ``source_line.normal`` and may be
    reversed when the physical field leaves the opposite side of the line.

    The implementation uses the high-frequency form of the 2D
    Rayleigh--Sommerfeld kernel. It should not be used for source-target
    separations comparable with an electron wavelength.
    """
    if quadrature != "trapezoid":
        raise ValueError("Only trapezoid quadrature is currently implemented")

    source_points = source_line.points()
    target_points = target_line.points()
    delta = target_points[:, None, :] - source_points[None, :, :]
    normal = source_line.normal if source_normal is None else jnp.asarray(source_normal)
    weights = _trapz_weights(source_line.coords)
    kernel = _rayleigh_sommerfeld_asymptotic_kernel_1d(delta, normal, energy)
    return kernel @ (_as_complex_wave_1d(source_wave) * weights)


def phase_grating_1d_from_projected_potential(projected_potential, energy):
    """Return a projected-potential phase grating ``exp(i sigma V_proj)``."""
    phase = interaction_constant(energy) * jnp.asarray(projected_potential)
    return jnp.exp(1j * phase)


def make_gaussian_atom_potential_sideview_1d(points, atom_positions, strengths, sigmas):
    """Evaluate a 2D sideview sum of Gaussian atom potentials on ``points``."""
    points = jnp.asarray(points)
    atom_positions = jnp.asarray(atom_positions)
    strengths = jnp.asarray(strengths)
    sigmas = jnp.asarray(sigmas)
    delta = points[:, None, :] - atom_positions[None, :, :]
    r2 = jnp.sum(delta * delta, axis=-1)
    return jnp.sum(strengths[None, :] * jnp.exp(-0.5 * r2 / sigmas[None, :] ** 2), axis=1)


def project_atoms_to_sample_line_1d(sample_line, atom_positions, strengths, sigmas):
    """Project Gaussian atoms onto the sampled sample line."""
    return make_gaussian_atom_potential_sideview_1d(
        sample_line.points(),
        atom_positions,
        strengths,
        sigmas,
    )


def project_potential_to_sample_line_1d(potential_fn, sample_line):
    """Sample an arbitrary sideview potential function on ``sample_line``."""
    return potential_fn(sample_line.points())


def simulate_single_slice_cylindrical_1d(
    input_wave,
    input_line,
    sample_line,
    output_line,
    projected_potential,
    energy,
    *,
    quadrature="trapezoid",
    green_kernel="cylindrical_asymptotic",
    steering="specular",
    input_normal=None,
    sample_normal=None,
    return_diagnostics=True,
):
    """Single-slice sideview model using forward RS boundary propagation.

    The potential is a projected phase grating on ``sample_line``. The two
    propagations use the normalized, obliquity-weighted 2D
    Rayleigh--Sommerfeld kernel. ``input_normal`` and ``sample_normal`` can
    select the outward side of each boundary independently.

    ``green_kernel`` and ``steering`` are retained as compatibility arguments
    for older callers; only the forward RS kernel is used.
    """
    if green_kernel != "cylindrical_asymptotic":
        raise ValueError("Only the Rayleigh--Sommerfeld asymptotic kernel is implemented")

    sample_wave = rayleigh_sommerfeld_propagate_1d(
        input_wave,
        input_line,
        sample_line,
        energy,
        quadrature=quadrature,
        source_normal=input_normal,
    )
    grating = phase_grating_1d_from_projected_potential(projected_potential, energy)
    sample_wave_after = sample_wave * grating
    output_wave = rayleigh_sommerfeld_propagate_1d(
        sample_wave_after,
        sample_line,
        output_line,
        energy,
        quadrature=quadrature,
        source_normal=sample_normal,
    )
    intensity = jnp.abs(output_wave) ** 2
    if not return_diagnostics:
        return output_wave, intensity
    input_line_intensity = jnp.sum(
        jnp.abs(input_wave) ** 2 * _trapz_weights(input_line.coords)
    )
    output_line_intensity = jnp.sum(
        intensity * _trapz_weights(output_line.coords)
    )
    diagnostics = {
        "sample_wave": sample_wave,
        "sample_wave_after_grating": sample_wave_after,
        # The legacy names are retained, although these are line integrals of
        # scalar intensity rather than vector electromagnetic fluxes.
        "input_power": input_line_intensity,
        "output_power": output_line_intensity,
        "input_line_intensity": input_line_intensity,
        "output_line_intensity": output_line_intensity,
        "steering": steering,
    }
    return output_wave, intensity, diagnostics


def _simulate_glancing_split_step_1d(
    input_wave,
    potential_slices,
    dx,
    dz,
    energy,
    *,
    kernel_builder,
    input_tilt=0.0,
    return_diagnostics=True,
):
    """Run a slice-based 1D split-step method with a supplied vacuum kernel."""
    wave = _as_complex_wave_1d(input_wave)
    potential_slices = jnp.asarray(potential_slices)
    n = wave.shape[0]
    coords = (jnp.arange(n) - n // 2) * dx
    if input_tilt != 0.0:
        wavelength = energy2wavelength(energy)
        wave = wave * jnp.exp(1j * 2.0 * jnp.pi * coords * jnp.sin(input_tilt) / wavelength)

    kernel = kernel_builder(n, dx, dz, energy)
    wavefronts = []
    for potential_slice in potential_slices:
        wave = wave * phase_grating_1d_from_projected_potential(potential_slice * dz, energy)
        wave = fourier_propagate_1d(wave, kernel)
        wavefronts.append(wave)

    intensity = diffraction_intensity_1d(wave)
    if not return_diagnostics:
        return wave, intensity
    diagnostics = {
        "wavefronts": _stack_wavefronts_or_empty_1d(wavefronts, wave),
        "transmitted_power": jnp.sum(jnp.abs(wave) ** 2),
        "input_tilt": input_tilt,
    }
    return wave, intensity, diagnostics


def simulate_glancing_fresnel_baseline_1d(
    input_wave,
    potential_slices,
    dx,
    dz,
    energy,
    *,
    input_tilt=0.0,
    return_diagnostics=True,
):
    """Slice-based 1D Fresnel multislice baseline for sideview propagation."""
    return _simulate_glancing_split_step_1d(
        input_wave,
        potential_slices,
        dx,
        dz,
        energy,
        kernel_builder=fresnel_propagation_kernel_1d,
        input_tilt=input_tilt,
        return_diagnostics=return_diagnostics,
    )


def simulate_glancing_angular_spectrum_1d(
    input_wave,
    potential_slices,
    dx,
    dz,
    energy,
    *,
    input_tilt=0.0,
    return_diagnostics=True,
):
    """Slice-based 1D angular-spectrum multislice propagation."""
    return _simulate_glancing_split_step_1d(
        input_wave,
        potential_slices,
        dx,
        dz,
        energy,
        kernel_builder=angular_spectrum_propagation_kernel_1d,
        input_tilt=input_tilt,
        return_diagnostics=return_diagnostics,
    )


def wpm_step_adaptive_1d(
    wave,
    n_map,
    dz,
    energy,
    dx,
    n_bins: int = 128,
    power_spacing: float = 2.0,
):
    """Run one adaptive-binned 1D WPM propagation step."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if power_spacing <= 0:
        raise ValueError("power_spacing must be positive")

    wave = _as_complex_wave_1d(wave)
    n_map = jnp.asarray(n_map)
    wavelength = energy2wavelength(energy)
    k0 = 1.0 / wavelength
    fx = get_frequencies_1d(wave.shape[0], dx)
    Ek = jnp.fft.fft(wave)

    n_min, n_max = n_map.min(), n_map.max()
    n_refs = get_polynomial_bins(n_min, n_max, n_bins, power=power_spacing)

    def propagate_for_ref(n_val):
        kz = jnp.sqrt(jnp.asarray(n_val**2 * k0**2 - fx**2, dtype=jnp.complex128))
        H = jnp.exp(1j * 2.0 * jnp.pi * dz * kz)
        return jnp.fft.ifft(H * Ek)

    ref_fields = jax.vmap(propagate_for_ref)(n_refs)
    idx_R = jnp.searchsorted(n_refs, n_map)
    idx_R = jnp.clip(idx_R, 1, n_bins - 1)
    idx_L = idx_R - 1

    n_L = n_refs[idx_L]
    n_R = n_refs[idx_R]
    denom = n_R - n_L
    w_raw = (n_map - n_L) / jnp.where(denom == 0, 1.0, denom)
    w = smoothstep(w_raw)

    field_L = jnp.take_along_axis(ref_fields, idx_L[None, :], axis=0).squeeze()
    field_R = jnp.take_along_axis(ref_fields, idx_R[None, :], axis=0).squeeze()
    return (1.0 - w) * field_L + w * field_R, w, idx_L, n_refs


def _branch_kz(kz_sq):
    kz = jnp.sqrt(jnp.asarray(kz_sq, dtype=jnp.complex128))
    return jnp.where(jnp.imag(kz) < 0.0, -kz, kz)


def interface_coupling_wpm_1d(
    wave,
    n_left,
    n_right,
    dx,
    energy,
    *,
    polarization="TE",
    direction="forward",
):
    """Return scalar WPM transmitted/reflected fields for one interface.

    Coefficients follow the bidirectional WPM TE/TM scalar reductions from the
    bundled references. ``n_left`` and ``n_right`` may be profiles; their means
    define a homogeneous reference interface for this first implementation.
    """
    wave = _as_complex_wave_1d(wave)
    n_a = jnp.mean(jnp.asarray(n_left))
    n_b = jnp.mean(jnp.asarray(n_right))
    if direction == "backward":
        n_a, n_b = n_b, n_a

    wavelength = energy2wavelength(energy)
    k0_ang = 2.0 * jnp.pi / wavelength
    kx = 2.0 * jnp.pi * get_frequencies_1d(wave.shape[0], dx)
    kz_a = _branch_kz((n_a * k0_ang) ** 2 - kx**2)
    kz_b = _branch_kz((n_b * k0_ang) ** 2 - kx**2)

    denom_te = kz_a + kz_b
    if polarization.upper() == "TM":
        denom = n_b**2 * kz_a + n_a**2 * kz_b
        t = 2.0 * n_a * n_b * kz_a / jnp.where(denom == 0, 1.0, denom)
        r = (n_b**2 * kz_a - n_a**2 * kz_b) / jnp.where(denom == 0, 1.0, denom)
    else:
        t = 2.0 * kz_a / jnp.where(denom_te == 0, 1.0, denom_te)
        r = (kz_a - kz_b) / jnp.where(denom_te == 0, 1.0, denom_te)

    spectrum = jnp.fft.fft(wave)
    transmitted = jnp.fft.ifft(t * spectrum)
    reflected = jnp.fft.ifft(r * spectrum)
    return transmitted, reflected, {"t": t, "r": r, "n_a": n_a, "n_b": n_b}


def bidirectional_wpm_sweep_1d(
    potential_slices,
    input_wave,
    dx,
    dz,
    energy,
    *,
    n_bins=128,
    n_sweeps=4,
    boundary="outgoing",
):
    """Reference bidirectional 1D WPM sweep with iterative reflected storage."""
    n_maps = klein_gordon_refractive_index_1d(potential_slices, energy)
    input_wave = _as_complex_wave_1d(input_wave)
    n_slices = potential_slices.shape[0]
    zero = jnp.zeros_like(input_wave)
    sweep_count = max(int(n_sweeps), 1)
    backward_from_right = [zero for _ in range(n_slices + 1)]
    residuals = []

    def forward_pass(backward_boundary):
        wave_plus = input_wave
        forward_wavefronts = []
        plus_at_interfaces = [zero for _ in range(n_slices + 1)]
        for potential_index in range(n_slices):
            wave_plus, _, _, _ = wpm_step_adaptive_1d(
                wave_plus,
                n_maps[potential_index],
                dz,
                energy,
                dx,
                n_bins=n_bins,
            )
            forward_wavefronts.append(wave_plus)
            plus_at_interfaces[potential_index + 1] = wave_plus
            if potential_index + 1 < n_slices:
                n_left = n_maps[potential_index]
                n_right = n_maps[potential_index + 1]
                backward_incident = backward_boundary[potential_index + 1]
                transmitted_lr, reflected_lr, _ = interface_coupling_wpm_1d(
                    wave_plus,
                    n_left,
                    n_right,
                    dx,
                    energy,
                    direction="forward",
                )
                transmitted_rl, reflected_rl, _ = interface_coupling_wpm_1d(
                    backward_incident,
                    n_left,
                    n_right,
                    dx,
                    energy,
                    direction="backward",
                )
                wave_plus = transmitted_lr + reflected_rl
        return wave_plus, forward_wavefronts, plus_at_interfaces

    def backward_pass(plus_at_interfaces):
        new_backward = [zero for _ in range(n_slices + 1)]
        backward_wavefronts = [zero for _ in range(n_slices)]
        wave_minus = zero
        for potential_index in range(n_slices - 2, -1, -1):
            new_backward[potential_index + 1] = wave_minus
            n_left = n_maps[potential_index]
            n_right = n_maps[potential_index + 1]
            _, reflected_lr, _ = interface_coupling_wpm_1d(
                plus_at_interfaces[potential_index + 1],
                n_left,
                n_right,
                dx,
                energy,
                direction="forward",
            )
            transmitted_rl, reflected_rl, _ = interface_coupling_wpm_1d(
                wave_minus,
                n_left,
                n_right,
                dx,
                energy,
                direction="backward",
            )
            wave_minus, _, _, _ = wpm_step_adaptive_1d(
                reflected_lr + transmitted_rl,
                n_maps[potential_index],
                -dz,
                energy,
                dx,
                n_bins=n_bins,
            )
            new_backward[potential_index] = wave_minus
            backward_wavefronts[potential_index] = wave_minus
        return new_backward, backward_wavefronts

    for _ in range(sweep_count):
        _, _, plus_at_interfaces = forward_pass(backward_from_right)
        new_backward_from_right, backward_wavefronts = backward_pass(plus_at_interfaces)
        residual = sum(
            jnp.linalg.norm(new_backward_from_right[i] - backward_from_right[i])
            for i in range(n_slices + 1)
        ) / (jnp.linalg.norm(input_wave) + 1e-30)
        residuals.append(residual)
        backward_from_right = new_backward_from_right

    transmitted, forward_wavefronts, plus_at_interfaces = forward_pass(backward_from_right)
    final_backward_from_right, backward_wavefronts = backward_pass(plus_at_interfaces)
    reflected_wave = final_backward_from_right[0]

    diagnostics = {
        "wavefronts_plus": _stack_wavefronts_or_empty_1d(forward_wavefronts, input_wave),
        "wavefronts_minus": _stack_wavefronts_or_empty_1d(backward_wavefronts, input_wave),
        "residual_per_sweep": jnp.asarray(residuals),
        "boundary": boundary,
        "two_way": True,
    }
    return transmitted, reflected_wave, diagnostics


def _apply_scalar_interface_scattering_1d(plus_from_left, minus_from_right, n_left, n_right):
    S = interface_scattering_matrix_1d(n_left, n_right)
    plus_right = S[0, 0] * plus_from_left + S[0, 1] * minus_from_right
    minus_left = S[1, 0] * plus_from_left + S[1, 1] * minus_from_right
    return plus_right, minus_left


def _residual_between_boundary_lists(new_boundary, old_boundary, reference_wave):
    return sum(
        jnp.linalg.norm(new_boundary[i] - old_boundary[i])
        for i in range(len(new_boundary))
    ) / (jnp.linalg.norm(reference_wave) + 1e-30)


def simulate_bidirectional_wpm_1d(
    potential_slices,
    input_wave,
    dx,
    dz,
    energy,
    *,
    n_bins=128,
    n_sweeps=4,
    boundary="outgoing",
    return_diagnostics=True,
):
    """Simulate sideview 1D bidirectional WPM."""
    transmitted, reflected, diagnostics = bidirectional_wpm_sweep_1d(
        potential_slices,
        input_wave,
        dx,
        dz,
        energy,
        n_bins=n_bins,
        n_sweeps=n_sweeps,
        boundary=boundary,
    )
    intensity = diffraction_intensity_1d(transmitted)
    reflected_power = jnp.sum(jnp.abs(reflected) ** 2)
    transmitted_power = jnp.sum(jnp.abs(transmitted) ** 2)
    diagnostics = {
        **diagnostics,
        "transmitted_power": transmitted_power,
        "reflected_power": reflected_power,
        "norm_drift": transmitted_power + reflected_power - jnp.sum(jnp.abs(input_wave) ** 2),
    }
    if not return_diagnostics:
        return transmitted, reflected, intensity
    return transmitted, reflected, intensity, diagnostics


def _binomial_half_coefficients(order: int):
    coeffs = [1.0]
    c = 1.0
    for k in range(1, order + 1):
        c *= (0.5 - (k - 1)) / k
        coeffs.append(c)
    return jnp.asarray(coeffs, dtype=jnp.float64)


def pade_sqrt_coefficients(pade_order=(1, 1)):
    """Return numerator/denominator coefficients for ``sqrt(1+x)`` Pade."""
    m, n = pade_order
    if m < 0 or n < 0:
        raise ValueError("Pade orders must be non-negative")
    coeffs = _binomial_half_coefficients(m + n)
    if n == 0:
        return coeffs[: m + 1], jnp.ones((1,), dtype=coeffs.dtype)

    rows = []
    rhs = []
    for k in range(m + 1, m + n + 1):
        rows.append([coeffs[k - j] for j in range(1, n + 1)])
        rhs.append(-coeffs[k])
    q_tail = jnp.linalg.solve(jnp.asarray(rows), jnp.asarray(rhs))
    q = jnp.concatenate([jnp.ones((1,), dtype=q_tail.dtype), q_tail])

    p_terms = []
    for k in range(m + 1):
        acc = 0.0
        for j in range(min(k, n) + 1):
            acc = acc + q[j] * coeffs[k - j]
        p_terms.append(acc)
    p = jnp.asarray(p_terms, dtype=q.dtype)
    return p, q


def _periodic_laplacian_matrix_1d(n: int, dx: float):
    eye = jnp.eye(n, dtype=jnp.float64)
    return (jnp.roll(eye, 1, axis=0) - 2.0 * eye + jnp.roll(eye, -1, axis=0)) / dx**2


def _spectral_laplacian_matrix_1d(n: int, dx: float):
    eye = jnp.eye(n, dtype=jnp.complex128)
    frequencies = get_frequencies_1d(n, dx)
    symbol = -(2.0 * jnp.pi * frequencies) ** 2
    return jnp.fft.ifft(symbol[:, None] * jnp.fft.fft(eye, axis=0), axis=0)


def build_sideview_operator_x_1d(
    potential_or_index,
    dx,
    energy,
    *,
    n0_mode="slice_mean",
    transverse_operator="spectral",
):
    """Build the dense dimensionless sideview square-root operator ``X``."""
    n_profile = jnp.asarray(potential_or_index)
    if jnp.iscomplexobj(n_profile):
        refractive_index = n_profile
    else:
        refractive_index = klein_gordon_refractive_index_1d(n_profile, energy)

    if n0_mode == "slice_mean":
        n0 = jnp.mean(refractive_index)
    elif n0_mode == "vacuum":
        n0 = jnp.asarray(1.0, dtype=refractive_index.dtype)
    else:
        n0 = jnp.asarray(n0_mode, dtype=refractive_index.dtype)

    wavelength = energy2wavelength(energy)
    k0 = 1.0 / wavelength
    if transverse_operator == "spectral":
        lap = _spectral_laplacian_matrix_1d(refractive_index.shape[0], dx)
    elif transverse_operator == "finite_difference":
        lap = _periodic_laplacian_matrix_1d(refractive_index.shape[0], dx)
    else:
        raise ValueError("transverse_operator must be 'spectral' or 'finite_difference'")
    lap_term = lap / (2.0 * jnp.pi * k0 * n0) ** 2
    index_term = jnp.diag((refractive_index**2 - n0**2) / n0**2)
    return jnp.asarray(index_term + lap_term, dtype=jnp.complex128), n0


def _poly_matrix(coeffs, matrix):
    result = jnp.zeros_like(matrix, dtype=jnp.complex128)
    power = jnp.eye(matrix.shape[0], dtype=jnp.complex128)
    for coeff in coeffs:
        result = result + coeff * power
        power = power @ matrix
    return result


def apply_pade_rational_1d(operator_x, wave, *, pade_order=(1, 1)):
    """Apply the Pade rational approximation to ``sqrt(1+X)``."""
    p, q = pade_sqrt_coefficients(pade_order)
    operator_x = jnp.asarray(operator_x, dtype=jnp.complex128)
    P = _poly_matrix(p, operator_x)
    Q = _poly_matrix(q, operator_x)
    return jnp.linalg.solve(Q, P @ _as_complex_wave_1d(wave))


def _pade_sqrt_matrix(operator_x, pade_order):
    p, q = pade_sqrt_coefficients(pade_order)
    P = _poly_matrix(p, operator_x)
    Q = _poly_matrix(q, operator_x)
    return jnp.linalg.solve(Q, P)


def pade_forward_step_1d(
    wave,
    potential_slice,
    dx,
    dz,
    energy,
    *,
    pade_order=(1, 1),
    n0_mode="slice_mean",
    evanescent="damp",
    transverse_operator="spectral",
):
    """Propagate a forward wave through one dense Pade square-root BPM step."""
    X, n0 = build_sideview_operator_x_1d(
        potential_slice,
        dx,
        energy,
        n0_mode=n0_mode,
        transverse_operator=transverse_operator,
    )
    R = _pade_sqrt_matrix(X, pade_order)
    k0 = 1.0 / energy2wavelength(energy)
    generator = 1j * 2.0 * jnp.pi * dz * k0 * n0 * R
    H = jax.scipy.linalg.expm(generator)
    return H @ _as_complex_wave_1d(wave)


def pade_backward_step_1d(
    wave,
    potential_slice,
    dx,
    dz,
    energy,
    *,
    pade_order=(1, 1),
    n0_mode="slice_mean",
    evanescent="damp",
    transverse_operator="spectral",
):
    """Propagate a backward wave through one dense Pade square-root BPM step."""
    X, n0 = build_sideview_operator_x_1d(
        potential_slice,
        dx,
        energy,
        n0_mode=n0_mode,
        transverse_operator=transverse_operator,
    )
    R = _pade_sqrt_matrix(X, pade_order)
    k0 = 1.0 / energy2wavelength(energy)
    generator = -1j * 2.0 * jnp.pi * dz * k0 * n0 * R
    H = jax.scipy.linalg.expm(generator)
    return H @ _as_complex_wave_1d(wave)


def interface_scattering_matrix_1d(n_left, n_right):
    """Return scalar normal-incidence interface scattering coefficients."""
    n_left = jnp.mean(jnp.asarray(n_left))
    n_right = jnp.mean(jnp.asarray(n_right))
    denom = jnp.where(n_left + n_right == 0, 1.0, n_left + n_right)
    r_lr = (n_left - n_right) / denom
    r_rl = -r_lr
    t_lr = 2.0 * n_left / denom
    t_rl = 2.0 * n_right / denom
    return jnp.array([[t_lr, r_rl], [r_lr, t_rl]], dtype=jnp.complex128)


def bidirectional_pade_sweep_1d(
    potential_slices,
    input_wave,
    dx,
    dz,
    energy,
    *,
    pade_order=(1, 1),
    n0_mode="slice_mean",
    evanescent="damp",
    boundary="pml",
    scattering_update="s_matrix",
    n_sweeps=4,
    transverse_operator="spectral",
):
    """Reference bidirectional dense Pade BPM sweep with two-way iterations."""
    input_wave = _as_complex_wave_1d(input_wave)
    n_slices = potential_slices.shape[0]
    zero = jnp.zeros_like(input_wave)
    sweep_count = max(int(n_sweeps), 1)
    backward_from_right = [zero for _ in range(n_slices + 1)]
    residuals = []
    n_maps = klein_gordon_refractive_index_1d(potential_slices, energy)

    def forward_pass(backward_boundary):
        wave_plus = input_wave
        wavefronts_plus = []
        plus_at_interfaces = [zero for _ in range(n_slices + 1)]
        for j in range(n_slices):
            wave_plus = pade_forward_step_1d(
                wave_plus,
                potential_slices[j],
                dx,
                dz,
                energy,
                pade_order=pade_order,
                n0_mode=n0_mode,
                evanescent=evanescent,
                transverse_operator=transverse_operator,
            )
            wavefronts_plus.append(wave_plus)
            plus_at_interfaces[j + 1] = wave_plus
            if j + 1 < n_slices:
                wave_plus, _ = _apply_scalar_interface_scattering_1d(
                    wave_plus,
                    backward_boundary[j + 1],
                    n_maps[j],
                    n_maps[j + 1],
                )
        return wave_plus, wavefronts_plus, plus_at_interfaces

    def backward_pass(plus_at_interfaces):
        new_backward = [zero for _ in range(n_slices + 1)]
        wavefronts_minus = [zero for _ in range(n_slices)]
        wave_minus = zero
        for j in range(n_slices - 2, -1, -1):
            new_backward[j + 1] = wave_minus
            _, minus_left = _apply_scalar_interface_scattering_1d(
                plus_at_interfaces[j + 1],
                wave_minus,
                n_maps[j],
                n_maps[j + 1],
            )
            wave_minus = pade_backward_step_1d(
                minus_left,
                potential_slices[j],
                dx,
                dz,
                energy,
                pade_order=pade_order,
                n0_mode=n0_mode,
                evanescent=evanescent,
                transverse_operator=transverse_operator,
            )
            new_backward[j] = wave_minus
            wavefronts_minus[j] = wave_minus
        return new_backward, wavefronts_minus

    for _ in range(sweep_count):
        _, _, plus_at_interfaces = forward_pass(backward_from_right)
        new_backward_from_right, wavefronts_minus = backward_pass(plus_at_interfaces)
        residuals.append(
            _residual_between_boundary_lists(
                new_backward_from_right,
                backward_from_right,
                input_wave,
            )
        )
        backward_from_right = new_backward_from_right

    transmitted, wavefronts_plus, plus_at_interfaces = forward_pass(backward_from_right)
    final_backward_from_right, wavefronts_minus = backward_pass(plus_at_interfaces)
    reflected = final_backward_from_right[0]

    diagnostics = {
        "wavefronts_plus": _stack_wavefronts_or_empty_1d(wavefronts_plus, input_wave),
        "wavefronts_minus": _stack_wavefronts_or_empty_1d(wavefronts_minus, input_wave),
        "residual_per_sweep": jnp.asarray(residuals),
        "boundary": boundary,
        "scattering_update": scattering_update,
        "two_way": True,
    }
    return transmitted, reflected, diagnostics


def simulate_bidirectional_pade_bpm_1d(
    potential_slices,
    input_wave,
    dx,
    dz,
    energy,
    *,
    pade_order=(1, 1),
    n0_mode="slice_mean",
    evanescent="damp",
    boundary="pml",
    scattering_update="s_matrix",
    n_sweeps=4,
    transverse_operator="spectral",
    return_diagnostics=True,
):
    """Simulate sideview 1D bidirectional Pade square-root BPM."""
    transmitted, reflected, diagnostics = bidirectional_pade_sweep_1d(
        potential_slices,
        input_wave,
        dx,
        dz,
        energy,
        pade_order=pade_order,
        n0_mode=n0_mode,
        evanescent=evanescent,
        boundary=boundary,
        scattering_update=scattering_update,
        n_sweeps=n_sweeps,
        transverse_operator=transverse_operator,
    )
    intensity = diffraction_intensity_1d(transmitted)
    transmitted_power = jnp.sum(jnp.abs(transmitted) ** 2)
    reflected_power = jnp.sum(jnp.abs(reflected) ** 2)
    diagnostics = {
        **diagnostics,
        "transmitted_power": transmitted_power,
        "reflected_power": reflected_power,
        "norm_drift": transmitted_power + reflected_power - jnp.sum(jnp.abs(input_wave) ** 2),
    }
    if not return_diagnostics:
        return transmitted, reflected, intensity
    return transmitted, reflected, intensity, diagnostics


def wpm_step_adaptive(
    wave,
    n_map,
    dz,
    energy,
    ps: Sampling,
    n_bins: int = 256,
    power_spacing: float = 2.0,
):
    """Run one adaptive-binned WPM propagation step."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if power_spacing <= 0:
        raise ValueError("power_spacing must be positive")

    ny, nx = wave.shape
    wavelength = energy2wavelength(energy)
    k0 = 1 / wavelength
    k_perp2 = transverse_frequency_squared((ny, nx), ps)

    Ek = jnp.fft.fft2(wave)

    n_min, n_max = n_map.min(), n_map.max()
    n_refs = get_polynomial_bins(n_min, n_max, n_bins, power=power_spacing)

    ref_fields = wpm_propagation_kernel_vmap(Ek, n_refs, k0, k_perp2, dz)

    # Find the bin indices for every pixel
    idx_R = jnp.searchsorted(n_refs, n_map)
    idx_R = jnp.clip(idx_R, 1, n_bins - 1)
    idx_L = idx_R - 1

    n_L = n_refs[idx_L]
    n_R = n_refs[idx_R]

    denom = n_R - n_L
    w_raw = (n_map - n_L) / jnp.where(denom == 0, 1.0, denom)
    w = smoothstep(w_raw)

    field_L = jnp.take_along_axis(ref_fields, idx_L[None, ...], axis=0).squeeze()
    field_R = jnp.take_along_axis(ref_fields, idx_R[None, ...], axis=0).squeeze()

    new_wave = (1 - w) * field_L + w * field_R

    return new_wave, w, idx_L, n_refs


def _slice_phase_grating(potential_slice, slice_thickness, energy):
    """Return exact phase grating from the slice refractive index."""
    wavelength = energy2wavelength(energy)
    n_slice = electron_refractive_index(potential_slice, energy)
    phase = 2 * jnp.pi * (n_slice - 1) * slice_thickness / wavelength
    return jnp.exp(1j * phase)


def _stack_wavefronts_or_empty(wavefronts, reference_wave):
    """Return saved wavefronts with a stable empty-stack shape."""
    if wavefronts:
        return jnp.stack(wavefronts)
    return jnp.empty((0, *reference_wave.shape), dtype=reference_wave.dtype)

# =============================================================================
# 5. High-Level Simulation Functions
# =============================================================================

def simulate_fresnel_as(potential, probe, prop_kernel, slice_thickness, energy):
    """Simulate multislice propagation with a fixed Fourier transfer function.

    Parameters:
    - potential: (N, ny, nx) array of potential slices (V).
    - probe: (ny, nx) array of the initial wavefront.
    - prop_kernel: (ny, nx) propagator kernel (Fresnel or AS).
    - slice_thickness: thickness of each slice (Å).
    - energy: Beam energy (eV).

    Returns:
    - exit_wave
    - diffraction_pattern (intensity)
    - wavefronts (stacked array if requested, else None)
    """
    wavefront = probe
    wavefronts = []

    for potential_slice in potential:
        wavefront = wavefront * _slice_phase_grating(potential_slice, slice_thickness, energy)
        wavefront = fourier_propagate(wavefront, prop_kernel)
        wavefronts.append(wavefront)

    exit_wave = wavefront
    diffraction_pattern = diffraction_intensity(exit_wave)
    return exit_wave, diffraction_pattern, _stack_wavefronts_or_empty(wavefronts, probe)


simulate_fresnel_as_jit = jax.jit(simulate_fresnel_as)


def simulate_wpm(
    potential,
    probe,
    slice_thickness,
    energy,
    sampling: Sampling,
    n_bins: int = 128,
    power_spacing: float = 2.0,
):
    """Simulate multislice propagation with adaptive-binned WPM."""
    wavefront = probe
    wavefronts = []

    for potential_slice in potential:
        refractive_index = electron_refractive_index(potential_slice, energy)
        wavefront, _, _, _ = wpm_step_adaptive(
            wavefront,
            refractive_index,
            slice_thickness,
            energy,
            sampling,
            n_bins=n_bins,
            power_spacing=power_spacing,
        )
        wavefronts.append(wavefront)

    exit_wave = wavefront
    diffraction_pattern = diffraction_intensity(exit_wave)
    return exit_wave, diffraction_pattern, _stack_wavefronts_or_empty(wavefronts, probe)


simulate_wpm_jit = jax.jit(simulate_wpm, static_argnames=("n_bins", "power_spacing"))


def _kg_full_ode_rhs(t, y, args):
    """RHS of the full second-order KG system with slice-wise potential."""
    all_k0_sq_n_sq, k_perp_sq, slice_thickness = args

    slice_idx = jnp.clip(
        jnp.floor(t / slice_thickness).astype(jnp.int32),
        0, all_k0_sq_n_sq.shape[0] - 1,
    )
    k0_sq_n_sq = all_k0_sq_n_sq[slice_idx]

    psi = y[0] + 1j * y[1]
    phi = y[2] + 1j * y[3]

    psi_k = jnp.fft.fft2(psi)
    dphi = (2 * jnp.pi) ** 2 * (
        jnp.fft.ifft2(k_perp_sq * psi_k) - k0_sq_n_sq * psi
    )

    return jnp.stack([phi.real, phi.imag, dphi.real, dphi.imag])


def _kg_forward_vacuum_phi(psi, k0, k_perp_sq):
    """Return the exact forward vacuum derivative for an arbitrary probe."""
    psi_k = jnp.fft.fft2(psi)
    kz = jnp.sqrt(jnp.asarray(k0**2 - k_perp_sq, dtype=jnp.complex128))
    kz = jnp.where(jnp.imag(kz) < 0, -kz, kz)
    return jnp.fft.ifft2(1j * 2 * jnp.pi * kz * psi_k)


def _stack_refractive_index_squared(potential, energy, k0):
    """Return ``k0^2 n^2`` in cycles-squared units for every potential slice."""
    return jnp.stack([
        k0**2 * electron_refractive_index(potential_slice, energy) ** 2
        for potential_slice in potential
    ])


def simulate_kg_ode_full(
    potential,
    probe,
    slice_thickness,
    energy,
    sampling: Sampling,
    initial_phi=None,
    rtol=1e-8,
    atol=1e-10,
    save_wavefronts=True,
    solver_name="dopri8",
    max_steps=None,
):
    """Solve the full non-paraxial second-order KG equation.

    This restores the true KG ODE system,

        d²ψ/dz² + (∇²⊥ + (2πk₀)² n²) ψ = 0,

    rather than the slowly-varying-envelope approximation obtained by
    dropping u″ after ψ = u·exp(2πi·k₀·z).

    The potential is treated as piecewise constant over each slice, and the
    adaptive solver is clipped to the slice boundaries so it never steps
    across a discontinuity in n²(z).

    This is a second-order state-space solve. When chaining multiple calls,
    the returned ``exit_phi`` must be fed back as ``initial_phi`` for the
    next call. Re-using only ``exit_wave`` changes the physical state.

    Parameters
    ----------
    potential : array, shape (N_slices, ny, nx)
        Potential in Volts (average over each slice).
    probe : array, shape (ny, nx)
        Initial wavefront ψ(0).
    slice_thickness : float
        Thickness of each slice in Angstroms.
    energy : float
        Beam energy in eV.
    sampling : tuple of float
        Pixel sizes (dy, dx) in Angstroms.
    initial_phi : array, shape (ny, nx), optional
        Initial dψ/dz. If omitted, the exact forward vacuum derivative of the
        input probe is used. When chaining repeated calls, pass the previous
        ``exit_phi`` here to preserve the full KG state.
    rtol, atol : float
        Tolerances for the adaptive ODE solver.
    save_wavefronts : bool, optional
        If True (default), save ψ at every slice boundary. Set to False
        to reduce memory usage and solver overhead when only the exit
        wave is needed.
    solver_name : {"dopri8", "tsit5"}, optional
        ODE solver to use. "dopri8" (default) is the most robust and is
        used in tests. "tsit5" can be faster but may require a larger
        step budget.
    max_steps : int, optional
        Maximum internal adaptive steps. If omitted, an automatic default
        based on *solver_name* is used.

    Returns
    -------
    exit_wave : array, shape (ny, nx)
        ψ at the exit plane.
    exit_phi : array, shape (ny, nx)
        dψ/dz at the exit plane.
    diffraction_pattern : array, shape (ny, nx)
        |FFT(exit_wave)|².
    wavefronts : array, shape (N_slices, ny, nx) or None
        ψ at each slice boundary, or None if *save_wavefronts* is False.
    """
    import diffrax

    wavelength = energy2wavelength(energy)
    k0 = 1 / wavelength
    k_perp_sq = transverse_frequency_squared(probe.shape, sampling)

    N_slices = potential.shape[0]
    total_thickness = N_slices * slice_thickness
    all_k0_sq_n_sq = _stack_refractive_index_squared(potential, energy, k0)

    psi0 = jnp.asarray(probe, dtype=jnp.complex128)
    if initial_phi is None:
        phi0 = _kg_forward_vacuum_phi(psi0, k0, k_perp_sq)
    else:
        phi0 = jnp.asarray(initial_phi, dtype=jnp.complex128)

    y0 = jnp.stack([psi0.real, psi0.imag, phi0.real, phi0.imag])

    omega_bound_sq = (2 * jnp.pi) ** 2 * (
        jnp.max(k_perp_sq) + jnp.max(jnp.abs(all_k0_sq_n_sq))
    )
    omega_max = jnp.sqrt(jnp.maximum(omega_bound_sq, 1e-30))
    dtmax = 3.5 / omega_max

    solver_name = solver_name.lower()
    if solver_name == "tsit5":
        solver = diffrax.Tsit5()
        default_max_steps = N_slices * 8000
    else:
        solver = diffrax.Dopri8()
        default_max_steps = N_slices * 2000

    if max_steps is None:
        max_steps = default_max_steps

    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol, dtmax=dtmax)
    term = diffrax.ODETerm(_kg_full_ode_rhs)

    save_ts = jnp.arange(1, N_slices + 1) * slice_thickness
    jump_ts = save_ts[:-1]
    if jump_ts.size == 0:
        jump_ts = None

    stepsize_controller = diffrax.ClipStepSizeController(
        stepsize_controller,
        jump_ts=jump_ts,
    )

    if save_wavefronts:
        saveat = diffrax.SaveAt(ts=save_ts)
    else:
        saveat = diffrax.SaveAt(t1=True)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=total_thickness,
        dt0=jnp.minimum(slice_thickness / 2.0, dtmax),
        y0=y0,
        args=(all_k0_sq_n_sq, k_perp_sq, slice_thickness),
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        saveat=saveat,
    )

    if save_wavefronts:
        wavefronts = sol.ys[:, 0, :, :] + 1j * sol.ys[:, 1, :, :]
        y_final = sol.ys[-1]
    else:
        wavefronts = None
        y_final = sol.ys[0]

    exit_wave = y_final[0] + 1j * y_final[1]
    exit_phi = y_final[2] + 1j * y_final[3]

    diffraction_pattern = diffraction_intensity(exit_wave)

    return exit_wave, exit_phi, diffraction_pattern, wavefronts
