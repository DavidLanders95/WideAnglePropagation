"""
Ptychographic reconstruction module for multislice electron microscopy.

Implements gradient-based multislice ptychography using three propagation methods:
  - Fresnel (paraxial approximation)
  - Angular Spectrum (exact Helmholtz)
  - Wave Propagation Method (WPM, wide-angle)

The forward model propagates a focused probe through a stack of object slices
and computes the far-field diffraction intensity.  Reconstruction minimises the
amplitude-based loss between measured and predicted diffraction patterns using
JAX automatic differentiation and the Adam optimiser.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial

from .propagation import (
    energy2wavelength,
    energy2sigma,
    get_frequencies,
    fresnel_propagation_kernel,
    angular_spectrum_propagation_kernel,
    Propagator,
    wpm_step_adaptive,
    electron_refractive_index,
)

# ============================================================================
# Probe generation
# ============================================================================

def make_probe(
    gpts,
    sampling,
    energy,
    semi_angle_mrad,
    defocus=0.0,
):
    """Create a converging electron probe (STEM-like).

    Parameters
    ----------
    gpts : tuple of int
        Grid size ``(ny, nx)``.
    sampling : tuple of float
        Pixel size ``(dy, dx)`` in Ångströms.
    energy : float
        Beam energy in eV.
    semi_angle_mrad : float
        Probe-forming aperture semi-angle in mrad.
    defocus : float, optional
        Defocus in Ångströms (positive = underfocus).

    Returns
    -------
    probe : jnp.ndarray, shape *gpts*, complex128
        Normalised probe wave-function in real space.
    """
    ny, nx = gpts
    wavelength = float(energy2wavelength(energy))

    Fx, Fy = get_frequencies(ny, nx, sampling)
    alpha = wavelength * jnp.sqrt(Fx ** 2 + Fy ** 2)  # angle in rad
    aperture = (alpha <= semi_angle_mrad * 1e-3).astype(jnp.float64)

    # Aberration function (defocus only for simplicity)
    chi = jnp.pi * wavelength * defocus * (Fx ** 2 + Fy ** 2)
    probe_k = aperture * jnp.exp(-1j * chi)

    probe = jnp.fft.ifft2(probe_k)
    probe = probe / jnp.sqrt(jnp.sum(jnp.abs(probe) ** 2))
    return probe.astype(jnp.complex128)


# ============================================================================
# Scan-position helpers
# ============================================================================

def generate_scan_positions(
    gpts, sampling, probe_region_frac=0.5, n_positions=16
):
    """Return a grid of scan positions (in pixel coordinates).

    Positions are placed on a roughly square grid that covers the central
    *probe_region_frac* of the array in each dimension.

    Parameters
    ----------
    gpts : tuple of int
        (ny, nx) grid points.
    sampling : tuple of float
        (dy, dx) pixel sizes in Å.
    probe_region_frac : float
        Fraction of the field that the scan covers (default 0.5).
    n_positions : int
        Total number of positions (rounded down to a perfect square).

    Returns
    -------
    positions : ndarray, shape (N, 2), int
        Row/col pixel indices for each scan point.
    """
    ny, nx = gpts
    n_side = int(np.sqrt(n_positions))

    margin_y = int(ny * (1 - probe_region_frac) / 2)
    margin_x = int(nx * (1 - probe_region_frac) / 2)

    rows = np.linspace(margin_y, ny - margin_y - 1, n_side, dtype=int)
    cols = np.linspace(margin_x, nx - margin_x - 1, n_side, dtype=int)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=-1)


def move_probe(probe, new_pos):
    """Shift probe so its centre sits at *new_pos* (row, col) pixels."""
    cy, cx = probe.shape[0] // 2, probe.shape[1] // 2
    shift = jnp.array([new_pos[0] - cy, new_pos[1] - cx])
    return jnp.roll(probe, shift, axis=(0, 1))


# ============================================================================
# Forward models (differentiable)
# ============================================================================

def _forward_fresnel(object_slices, probe, prop_kernel, slice_thickness, energy):
    """Multislice forward model using Fresnel propagation."""
    wavelength = energy2wavelength(energy)
    wavefront = probe
    n_slices = object_slices.shape[0]

    for i in range(n_slices):
        n = electron_refractive_index(object_slices[i], energy)
        phase_shift = jnp.exp(
            1j * 2 * jnp.pi * (n - 1) * slice_thickness / wavelength
        )
        wavefront = wavefront * phase_shift
        wavefront = Propagator(wavefront, prop_kernel)

    dp = jnp.fft.fftshift(jnp.fft.fft2(wavefront))
    return jnp.abs(dp) ** 2, wavefront


def _forward_angular_spectrum(
    object_slices, probe, prop_kernel, slice_thickness, energy
):
    """Multislice forward model using angular-spectrum propagation."""
    # Identical structure – only the kernel differs.
    return _forward_fresnel(
        object_slices, probe, prop_kernel, slice_thickness, energy
    )


def _forward_wpm(
    object_slices, probe, slice_thickness, energy, sampling,
    n_bins=128, power_spacing=2.0,
):
    """Multislice forward model using the Wave Propagation Method."""
    wavefront = probe
    n_slices = object_slices.shape[0]

    for i in range(n_slices):
        n = electron_refractive_index(object_slices[i], energy)
        wavefront, _, _, _ = wpm_step_adaptive(
            wavefront, n, slice_thickness, energy, sampling,
            n_bins=n_bins, power_spacing=power_spacing,
        )

    dp = jnp.fft.fftshift(jnp.fft.fft2(wavefront))
    return jnp.abs(dp) ** 2, wavefront


# ============================================================================
# Unified forward function
# ============================================================================

def forward_model(
    object_slices,
    probe,
    positions,
    method,
    slice_thickness,
    energy,
    sampling,
    prop_kernel=None,
    n_bins=128,
    power_spacing=2.0,
):
    """Run the multislice forward model for all scan positions.

    Parameters
    ----------
    object_slices : jnp.ndarray, shape (n_slices, ny, nx)
        Electrostatic potential for each slice (Volts).
    probe : jnp.ndarray, shape (ny, nx)
        Probe wave-function.
    positions : array-like, shape (n_pos, 2)
        Scan positions in pixel coordinates (row, col).
    method : str
        ``"fresnel"``, ``"angular_spectrum"`` or ``"wpm"``.
    slice_thickness : float
        Slice thickness in Ångströms.
    energy : float
        Beam energy in eV.
    sampling : tuple of float
        Pixel sizes (dy, dx) in Å.
    prop_kernel : jnp.ndarray or None
        Precomputed propagation kernel (Fresnel / AS).  Ignored for WPM.
    n_bins : int
        Number of refractive-index bins (WPM only).
    power_spacing : float
        Bin spacing power (WPM only).

    Returns
    -------
    diffraction_patterns : jnp.ndarray, shape (n_pos, ny, nx)
        Predicted diffraction intensities.
    """
    dps = []
    for pos in positions:
        shifted = move_probe(probe, pos)
        if method in ("fresnel", "angular_spectrum"):
            dp, _ = _forward_fresnel(
                object_slices, shifted, prop_kernel,
                slice_thickness, energy,
            )
        elif method == "wpm":
            dp, _ = _forward_wpm(
                object_slices, shifted, slice_thickness, energy,
                sampling, n_bins=n_bins, power_spacing=power_spacing,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        dps.append(dp)
    return jnp.stack(dps)


# ============================================================================
# Loss and reconstruction
# ============================================================================

def amplitude_loss(predicted_dp, measured_dp):
    """Amplitude-based loss: MSE between sqrt(intensities).

    Uses a small epsilon inside the sqrt to keep the gradient finite.
    """
    _eps = 1e-12
    pred_amp = jnp.sqrt(predicted_dp + _eps)
    meas_amp = jnp.sqrt(jnp.maximum(measured_dp, 0.0) + _eps)
    return jnp.mean((pred_amp - meas_amp) ** 2)


def _reconstruction_loss(
    object_slices,
    probe,
    measured_dps,
    positions,
    method,
    slice_thickness,
    energy,
    sampling,
    prop_kernel,
    n_bins,
    power_spacing,
):
    """Total loss over all scan positions (for AD)."""
    total = 0.0
    n_pos = measured_dps.shape[0]
    for i in range(n_pos):
        shifted = move_probe(probe, positions[i])
        if method in ("fresnel", "angular_spectrum"):
            dp, _ = _forward_fresnel(
                object_slices, shifted, prop_kernel,
                slice_thickness, energy,
            )
        else:
            dp, _ = _forward_wpm(
                object_slices, shifted, slice_thickness, energy,
                sampling, n_bins=n_bins, power_spacing=power_spacing,
            )
        total = total + amplitude_loss(dp, measured_dps[i])
    return total / n_pos


def reconstruct(
    measured_dps,
    positions,
    method,
    gpts,
    n_slices,
    slice_thickness,
    energy,
    sampling,
    semi_angle_mrad=20.0,
    n_iterations=50,
    learning_rate=1e-2,
    n_bins=128,
    power_spacing=2.0,
    verbose=True,
):
    """Gradient-descent multislice ptychographic reconstruction.

    Parameters
    ----------
    measured_dps : jnp.ndarray, shape (n_pos, ny, nx)
        Measured diffraction-pattern intensities.
    positions : ndarray, shape (n_pos, 2)
        Scan positions in pixel coordinates.
    method : str
        ``"fresnel"``, ``"angular_spectrum"`` or ``"wpm"``.
    gpts : tuple of int
        Grid size ``(ny, nx)``.
    n_slices : int
        Number of object slices to reconstruct.
    slice_thickness : float
        Thickness per slice in Å.
    energy : float
        Beam energy in eV.
    sampling : tuple of float
        ``(dy, dx)`` pixel sizes in Å.
    semi_angle_mrad : float
        Probe semi-angle (mrad) for the initial probe guess.
    n_iterations : int
        Number of Adam iterations.
    learning_rate : float
        Adam step size.
    n_bins : int
        WPM bins.
    power_spacing : float
        WPM bin spacing power.
    verbose : bool
        Print loss every 10 iterations.

    Returns
    -------
    recon_potential : jnp.ndarray, shape (n_slices, ny, nx)
        Reconstructed potential slices.
    losses : list of float
        Loss at each iteration.
    """
    ny, nx = gpts

    # Pre-compute propagation kernel (Fresnel / AS)
    prop_kernel = None
    if method == "fresnel":
        prop_kernel = fresnel_propagation_kernel(
            ny, nx, sampling, slice_thickness, energy,
        )
    elif method == "angular_spectrum":
        prop_kernel = angular_spectrum_propagation_kernel(
            ny, nx, sampling, slice_thickness, energy,
        )

    # Initial guess: small random potential to break symmetry.
    # This is important for WPM where uniform potential gives degenerate
    # refractive-index bins and zero gradients.
    key = jax.random.PRNGKey(0)
    object_slices = 0.01 * jax.random.normal(key, (n_slices, ny, nx), dtype=jnp.float64)

    # Probe (fixed during reconstruction for simplicity)
    probe = make_probe(gpts, sampling, energy, semi_angle_mrad)

    # Convert positions to jax array
    positions_jnp = jnp.array(positions)

    # ---- Adam optimiser state ----
    m = jnp.zeros_like(object_slices)
    v = jnp.zeros_like(object_slices)
    b1, b2, eps = 0.9, 0.999, 1e-8

    loss_fn = jax.value_and_grad(
        lambda slices: _reconstruction_loss(
            slices, probe, measured_dps, positions_jnp,
            method, slice_thickness, energy, sampling,
            prop_kernel, n_bins, power_spacing,
        )
    )

    losses = []
    for it in range(n_iterations):
        loss_val, grad = loss_fn(object_slices)
        losses.append(float(loss_val))

        # Adam update
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        m_hat = m / (1 - b1 ** (it + 1))
        v_hat = v / (1 - b2 ** (it + 1))
        object_slices = object_slices - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

        if verbose and (it % 10 == 0 or it == n_iterations - 1):
            print(f"  [{method}] iter {it:4d}  loss = {loss_val:.6e}")

    return object_slices, losses


# ============================================================================
# Quality metrics
# ============================================================================

def normalised_mse(reference, reconstruction):
    """Normalised mean-squared error (lower is better)."""
    diff = jnp.abs(reference) - jnp.abs(reconstruction)
    return float(jnp.sum(diff ** 2) / jnp.sum(jnp.abs(reference) ** 2))


def pearson_correlation(reference, reconstruction):
    """Pearson correlation between flattened arrays (higher is better)."""
    a = jnp.abs(reference).ravel()
    b = jnp.abs(reconstruction).ravel()
    a_zm = a - a.mean()
    b_zm = b - b.mean()
    num = jnp.sum(a_zm * b_zm)
    den = jnp.sqrt(jnp.sum(a_zm ** 2) * jnp.sum(b_zm ** 2))
    return float(num / jnp.maximum(den, 1e-30))


# ============================================================================
# Simple sample generator
# ============================================================================

def make_simple_sample(gpts, sampling, thickness_nm, slice_thickness_A, energy):
    """Create a simple phase object for testing ptychographic reconstruction.

    Generates a sample with a few rectangular features of different potentials,
    similar to a cross-section through a nanostructured material.

    Parameters
    ----------
    gpts : tuple of int
        (ny, nx) grid size.
    sampling : tuple of float
        (dy, dx) pixel sizes in Å.
    thickness_nm : float
        Total sample thickness in nm.
    slice_thickness_A : float
        Thickness per slice in Å.
    energy : float
        Beam energy in eV.

    Returns
    -------
    potential_slices : jnp.ndarray, shape (n_slices, ny, nx)
        Electrostatic potential slices in Volts.
    """
    ny, nx = gpts
    total_thickness_A = thickness_nm * 10.0  # nm → Å
    n_slices = max(1, int(round(total_thickness_A / slice_thickness_A)))

    # 2-D potential pattern (Volts) – the same in each slice
    # (uniform through thickness, to keep things simple)
    pot_2d = np.zeros((ny, nx), dtype=np.float64)

    # Feature 1: central rectangle, Silicon-like potential (~10 V)
    cy, cx = ny // 2, nx // 2
    hy, hx = ny // 6, nx // 6
    pot_2d[cy - hy : cy + hy, cx - hx : cx + hx] = 10.0

    # Feature 2: off-centre smaller block, higher potential (~20 V)
    oy, ox = ny // 4, nx // 4
    sy, sx = ny // 10, nx // 10
    pot_2d[oy - sy : oy + sy, ox - sx : ox + sx] = 20.0

    # Feature 3: another off-centre block
    oy2 = ny - ny // 4
    ox2 = nx - nx // 4
    pot_2d[oy2 - sy : oy2 + sy, ox2 - sx : ox2 + sx] = 15.0

    potential_slices = jnp.broadcast_to(
        jnp.array(pot_2d)[None, :, :], (n_slices, ny, nx)
    ).copy()

    return potential_slices
