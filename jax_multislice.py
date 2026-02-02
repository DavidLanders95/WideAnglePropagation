import numpy as np
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from abtem.multislice import _generate_potential_configurations
from abtem.antialias import AntialiasAperture
from ase import units
from functools import partial

def move_probe(probe, new_pos):
    """
    Move the probe by a given shift.

    Args:
        probe: The probe to move.
        shift: The shift to apply to the probe.

    Returns:
        The probe after the shift.
    """
    current_pos_row = probe.shape[0]//2 #probe coordinates are in y, x
    current_pos_col = probe.shape[1]//2 #probe coordinates are in y, x
    new_pos_row = new_pos[0] #scan coordinates are given in x, y
    new_pos_col = new_pos[1] #scan coordinates are given in x, y

    # if probe.shape[0] % 2 == 1:
    #     current_pos_row += 1

    shift_to_row = new_pos_row - current_pos_row
    shift_to_col = new_pos_col - current_pos_col
    shift = jnp.array([shift_to_row, shift_to_col])

    return jnp.roll(probe, shift, axis=(0, 1))


def get_frequencies(n, m, ps):
    fx = jnp.fft.fftfreq(n, ps[0])
    fy = jnp.fft.fftfreq(m, ps[1])
    Fx, Fy = jnp.meshgrid(fx, fy, indexing='ij')
    return Fx, Fy


@jax.jit
def shift_kernel(x0, y0, Fx, Fy):
    return jnp.exp(-1j * 2 * jnp.pi * (Fx * x0 + Fy * y0))


def fresnel_propagation_kernel(n: int,
                               m: int,
                               ps: float,
                               z: float,
                               energy: float):

    wavelength = energy2wavelength(energy)
    Fx, Fy = get_frequencies(n, m, ps)

    H = jnp.exp(1j * (
        2 * jnp.pi / wavelength) * z) * jnp.exp(
        -1j * jnp.pi * wavelength * z * (Fx**2 + Fy**2))

    return H


def angular_spectrum_propagation_kernel(n: int,
                                        m: int,
                                        ps: float,
                                        z: float,
                                        energy: float):
    wavelength = energy2wavelength(energy)
    Fx, Fy = get_frequencies(n, m, ps)
    H = jnp.exp(1j * 2 * jnp.pi * z * jnp.sqrt(
        (1 / wavelength)**2 - Fx**2 - Fy**2))

    return H


def wpm_propagation_kernel(Ek, n_val, k0, k_perp2, dz):
    # Calculate the longitudinal wave vector component
    # kz = sqrt((n * k0)^2 - k_perp^2)
    kz = jnp.sqrt(jnp.array(n_val**2 * k0**2 - k_perp2, dtype=jnp.complex128))
    H = jnp.exp(1j * dz * kz)
    return jnp.fft.ifft2(H * Ek)


wpm_propagation_kernel_vmap = jax.vmap(wpm_propagation_kernel, in_axes=(None, 0, None, None, None))


def make_k_damping_ramp(ny, nx, ps_y, ps_x, wavelength,
                        theta_start=0.2, theta_end=0.24):
    fy = jnp.fft.fftfreq(ny, d=ps_y)
    fx = jnp.fft.fftfreq(nx, d=ps_x)
    ky = (2 * jnp.pi) * fy[:, None]
    kx = (2 * jnp.pi) * fx[None, :]
    k_perp = jnp.sqrt(kx**2 + ky**2)

    k0 = 2 * jnp.pi / wavelength
    k_start = k0 * jnp.sin(theta_start)
    k_end = k0 * jnp.sin(theta_end)

    t = (k_perp - k_start) / (k_end - k_start + 1e-30)
    t = jnp.clip(t, 0.0, 1.0)
    ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * t))  # smooth 0..1

    return ramp, k_perp


def wpm_propagation_kernel_damped(Ek, n_val, k0, k_perp2, dz, ramp,
                                  damp_at_end=1e-2):
    # Your usual kz
    kz = jnp.sqrt(jnp.array(n_val**2 * k0**2 - k_perp2, dtype=jnp.complex128))

    # Choose alpha_max so that at ramp=1 the per-step amplitude multiplier is damp_at_end
    alpha_max = -jnp.log(damp_at_end) / dz
    damp = jnp.exp(-dz * alpha_max * jnp.sqrt(ramp))

    H = jnp.exp(1j * dz * kz) * damp
    return jnp.fft.ifft2(H * Ek)


wpm_propagation_kernel_damped_vmap = jax.vmap(wpm_propagation_kernel_damped, in_axes=(None, 0, None, None, None, None))


def wpm_step(wave, n_map, dz, energy, ps):
    """
    WPM step *without* using unique(n). One ifft2 per pixel's refractive index.

    Args:
        wave: complex array (ny, nx)
        n_map: refractive index map (ny, nx)
        dz: propagation distance
        energy: beam energy in eV
        ps: pixel size (dy, dx)
        pow_edge: Power of the supergaussian for the boundary.
    """
    ny, nx = wave.shape
    wavelength = energy2wavelength(energy)
    k0 = 2 * jnp.pi / wavelength

    # Frequencies -> k_perp^2
    Fy, Fx = get_frequencies(ny, nx, ps)
    kx = 2 * jnp.pi * Fx
    ky = 2 * jnp.pi * Fy
    k_perp2 = kx**2 + ky**2

    # One FFT of the input wave
    Ek = jnp.fft.fft2(wave)

    # Flatten refractive index map to list of values
    n_flat = n_map.reshape(-1)

    # For each refractive index value, propagate the whole field
    # fields has shape (P, ny, nx)
    ramp, _ = make_k_damping_ramp(ny, nx, ps[0], ps[1], wavelength,
                                  theta_start=0.24, theta_end=0.30)
    fields = wpm_propagation_kernel_damped_vmap(Ek, n_flat, k0, k_perp2, dz, ramp)

    P = n_flat.size
    p_indices = jnp.arange(P)
    iy, ix = jnp.divmod(p_indices, nx)

    # For each pixel p, pick the value at (iy[p], ix[p]) from its field
    def pick_pixel(field, y, x):
        return field[y, x]

    new_wave_flat = jax.vmap(pick_pixel)(fields, iy, ix)   # (P,)

    # Reshape back to (ny, nx)
    new_wave = new_wave_flat.reshape(ny, nx)

    return new_wave


# --- 1. The Paper's Smoothstep Function ---
def smoothstep(x):
    """
    Implements the smoothstep function from Eq. (1) of the paper.
    p(z) = 3z^2 - 2z^3 for 0 < z < 1.
    This creates a differentiable, smooth transition between bins.
    """
    # Clamp x to [0, 1] to handle values strictly inside/outside bins
    x = jnp.clip(x, 0.0, 1.0)
    return 3 * x**2 - 2 * x**3


# --- 2. Adaptive/Polynomial Binning ---
def get_polynomial_bins(n_min, n_max, n_bins, power=2.0):
    """
    Creates bin edges that are concentrated at the high end (atoms).

    If power=1.0: Linear spacing (Standard).
    If power=2.0: Quadratic spacing. Bins are dense at high n, sparse at low n.

    This answers your request to have unique propagators for complex areas (atoms)
    while grouping the simple background.
    """
    # Linear spacing from 0 to 1
    t = jnp.linspace(0, 1, n_bins)

    # Apply polynomial warping
    # If we want density at high values: 1 - (1-t)^power
    # If we want density at low values: t^power
    # Assuming atoms have HIGHER index than background:
    t_warped = t**power
    return n_min + (n_max - n_min) * t_warped


def wpm_step_adaptive(wave, n_map, dz, energy, ps, n_bins=256, power_spacing=2.0):
    """
    WPM step with Polynomial Binning and Paper-inspired Smoothstep masking.

    Args:
        wave: (ny, nx) complex field
        n_map: (ny, nx) refractive index map
        n_bins: Number of FFTs to run (tractability parameter)
        power_spacing: 1.0 = equal bins. >1.0 = focus bins on high indices (atoms).
    """
    ny, nx = wave.shape
    wavelength = energy2wavelength(energy)
    k0 = 2 * jnp.pi / wavelength

    # Frequency Grid
    dy, dx = ps
    ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    Fx, Fy = jnp.meshgrid(kx, ky)
    k_perp2 = Fx**2 + Fy**2

    # 1. FFT Input
    Ek = jnp.fft.fft2(wave)

    # 2. Define Adaptive Bins
    n_min, n_max = n_map.min(), n_map.max()

    # Generate the reference "bin" values
    # shape: (n_bins,)
    n_refs = get_polynomial_bins(n_min, n_max, n_bins, power=power_spacing)

    # 3. Compute Propagators (Batch FFT)
    ref_fields = wpm_propagation_kernel_vmap(Ek, n_refs, k0, k_perp2, dz)

    idx_R = jnp.searchsorted(n_refs, n_map)
    idx_R = jnp.clip(idx_R, 1, n_bins - 1)  # Clamp to valid range
    idx_L = idx_R - 1

    # Gather the reference n values at left and right boundaries
    n_L = n_refs[idx_L]
    n_R = n_refs[idx_R]

    # Calculate fractional weight w within the bin
    # n_map = (1-w)*n_L + w*n_R  -->  w = (n_map - n_L) / (n_R - n_L)
    denom = n_R - n_L
    w_raw = (n_map - n_L) / jnp.where(denom == 0, 1.0, denom)

    w = smoothstep(w_raw)

    field_L = jnp.take_along_axis(ref_fields, idx_L[None, ...], axis=0).squeeze()
    field_R = jnp.take_along_axis(ref_fields, idx_R[None, ...], axis=0).squeeze()

    new_wave = (1 - w) * field_L + w * field_R

    return new_wave, w, idx_L, n_refs


@jax.jit
def Propagator(u, H):
    ufft = jnp.fft.fft2(u)
    return jnp.fft.ifft2(H * ufft)


@jax.jit
def transmission_function(potential, energy):
    sigma = energy2sigma(energy)
    return jnp.exp(1j * sigma * potential)


def electron_refractive_index(potential, energy):
    E0 = electron_rest_energy()
    E = energy

    # Convert electrostatic potential (V) -> potential energy V (eV)
    # Electron charge = -1, so potential energy = -phi
    V = -potential

    EminusV = E - V

    numerator = 2 * EminusV * E0 + EminusV**2
    denominator = 2 * E * E0 + E**2

    n = jnp.sqrt(numerator / denominator)

    return n


def electron_refractive_index_taylor(potential, energy):
    E0 = electron_rest_energy()
    E = energy

    # The interaction constant factor derived from Taylor expansion:
    # (E + E0) / (E * (E + 2*E0))
    interaction_factor = (E + E0) / (E * (E + 2 * E0))

    # n = 1 + sigma_factor * potential
    n = 1.0 + interaction_factor * potential

    return n


def get_abtem_transmit(potential, energy):
    t_functions = []
    for _, potential_configuration in _generate_potential_configurations(
        potential
    ):
        for potential_slice in potential_configuration.generate_slices():
            transmission_function = potential_slice.transmission_function(
                energy=energy
            )
            transmission_function = AntialiasAperture().bandlimit(
                transmission_function, in_place=False
            )
            t_functions.append(transmission_function.array)

    return np.concatenate(t_functions, axis=0)


@jax.jit
def relativistic_mass_correction(energy: float) -> float:
    return 1 + units._e * energy / (units._me * units._c**2)


@jax.jit
def energy2mass(energy: float) -> float:
    """
    Calculate relativistic mass from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Relativistic mass [kg]
    """
    return relativistic_mass_correction(energy) * units._me


@jax.jit
def energy2sigma(energy: float) -> float:
    """
    Calculate interaction parameter from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Interaction parameter [1 / (Å * eV)].
    """
    return (
        2 * jnp.pi * energy2mass(energy) * units.kg * units._e * units.C *
        energy2wavelength(energy) /
        (units._hplanck * units.s * units.J) ** 2
    )


@jax.jit
def energy2wavelength(energy: float) -> float:
    """
    Calculate relativistic de Broglie wavelength from energy.

    Parameters
    ----------
    energy: float
        Energy [eV].

    Returns
    -------
    float
        Relativistic de Broglie wavelength [Å].
    """
    return (
        units._hplanck
        * units._c
        / jnp.sqrt(energy * (2 * units._me * units._c**2 / units._e + energy))
        / units._e
        * 1.0e10
    )


def electron_rest_energy():
    """
    Return the electron rest energy E0 = m_e c^2 in eV.
    """
    m_e = units._me
    c = units._c
    eV = units._e

    return m_e * c**2 / eV


@jdc.pytree_dataclass
class ProbeParamsFixed:
    wavelength: jdc.Static[float]
    alpha: jnp.array
    phi: jnp.array
    aperture: jnp.array


@jdc.pytree_dataclass
class ProbeParamsVariable:
    defocus: float = 0.
    astigmatism: float = 0.
    astigmatism_angle: float = 0.
    Cs: float = 0.
    coma: float = 0.
    coma_angle: float = 0.
    trefoil: float = 0.
    trefoil_angle: float = 0.


@jax.jit
def make_probe_fft(pp: ProbeParamsVariable, fpp: ProbeParamsFixed):
    alpha = fpp.alpha
    phi = fpp.phi
    aperture = fpp.aperture

    aberrations = jnp.zeros(alpha.shape, dtype=jnp.float32)
    aberrations += ((1 / 2) * alpha**2 * pp.defocus)
    aberrations += ((1 / 2) * alpha**2 * pp.astigmatism * jnp.cos(2 * (phi - pp.astigmatism_angle)))
    aberrations += ((1 / 3) * alpha**3 * pp.coma * jnp.cos(phi - pp.coma_angle))
    aberrations += ((1 / 3) * alpha**3 * pp.trefoil * jnp.cos(3 * (phi - pp.trefoil_angle)))
    aberrations += ((1 / 4) * alpha**4 * pp.Cs)
    aberrations *= (2 * jnp.pi / fpp.wavelength)
    # aberrations = jnp.exp(-1j * aberrations)
    aberrations = jnp.cos(-aberrations) + 1.0j * jnp.sin(-aberrations)

    probe_fft = jnp.ones(alpha.shape, dtype=jnp.complex64)
    probe_fft *= aperture
    probe_fft *= aberrations
    probe_fft /= jnp.linalg.norm(probe_fft)
    return probe_fft


def simple_fwhm(y):

    half_max = np.max(y) / 2.0

    indices = np.where(y > half_max)[0]

    if len(indices) < 2:
        return 0.0

    return indices[-1] - indices[0]
