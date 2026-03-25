"""Ptychography module for thick-sample reconstruction.

Provides:
  - Probe generation: ``make_probe``, ``make_gaussian_probe``
  - Scan grid: ``generate_scan_positions``
  - Fourier-shift helper: ``fourier_shift``
  - Forward models:
      ``simulate_ptychography_as``  – Angular Spectrum propagation
      ``simulate_ptychography_wpm`` – Wave Propagation Method
  - Reconstruction algorithms:
      ``epie_thin``           – classic ePIE for thin single-layer objects
      ``epie_multislice_as``  – multi-slice ePIE with Angular Spectrum
      ``reconstruct_as``      – gradient-based multi-slice (Angular Spectrum, Adam)
      ``reconstruct_wpm``     – gradient-based multi-slice (WPM, Adam)
  - Test-phantom helpers: ``make_phase_object``, ``make_potential_phantom``

References
----------
Maiden & Rodenburg (2009) Ultramicroscopy 109 1256-1262  – ePIE
Maiden, Humphry & Rodenburg (2012) J. Opt. Soc. Am. A 29 1606-1614  – multi-slice ePIE
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .propagation import (
    energy2wavelength,
    get_frequencies,
    angular_spectrum_propagation_kernel,
    electron_refractive_index,
    wpm_step_adaptive,
)

# =============================================================================
# 1. Probe Generation
# =============================================================================


def make_probe(ny, nx, convergence_angle_mrad, energy, sampling, defocus=0.0):
    """Create a convergent electron probe from a circular aperture.

    Parameters
    ----------
    ny, nx : int
        Number of grid points (rows, columns).
    convergence_angle_mrad : float
        Convergence semi-angle in mrad.
    energy : float
        Beam energy in eV.
    sampling : array_like, shape (2,)
        Pixel size (dy, dx) in Ångströms.
    defocus : float, optional
        Defocus in Ångströms (positive = overfocus).  Default 0.

    Returns
    -------
    probe : complex128 ndarray, shape (ny, nx)
        Probe normalised to unit total intensity (sum |probe|² = 1).
    """
    wavelength = float(energy2wavelength(energy))
    dy, dx = float(sampling[0]), float(sampling[1])

    fy = jnp.fft.fftfreq(ny, d=dy)
    fx = jnp.fft.fftfreq(nx, d=dx)
    FY, FX = jnp.meshgrid(fy, fx, indexing="ij")

    k_perp = jnp.sqrt(FY ** 2 + FX ** 2)  # 1/Å
    theta_mrad = k_perp * wavelength * 1000.0

    aperture = (theta_mrad <= convergence_angle_mrad).astype(jnp.float64)
    chi = jnp.pi * wavelength * defocus * (FY ** 2 + FX ** 2)
    ctf = aperture * jnp.exp(-1j * chi)

    # Probe in real space; aperture is already centred at frequency (0,0) via
    # fftfreq, so a plain IFFT2 gives a probe centred at pixel (0,0), which is
    # the conventional FFT-compatible origin used throughout the module.
    probe = jnp.fft.ifft2(ctf)
    probe = probe / jnp.sqrt(jnp.sum(jnp.abs(probe) ** 2))
    return probe.astype(jnp.complex128)


def make_gaussian_probe(ny, nx, sigma, center=None):
    """Create a real-space Gaussian probe.

    Parameters
    ----------
    ny, nx : int
        Grid dimensions.
    sigma : float
        Gaussian half-width (pixels).
    center : tuple (cy, cx), optional
        Centre pixel.  Defaults to (0, 0) (FFT-compatible origin).

    Returns
    -------
    probe : complex128 ndarray, shape (ny, nx), unit total intensity.
    """
    cy, cx = (0.0, 0.0) if center is None else center
    y = jnp.arange(ny, dtype=jnp.float64) - cy
    x = jnp.arange(nx, dtype=jnp.float64) - cx
    Y, X = jnp.meshgrid(y, x, indexing="ij")
    probe = jnp.exp(-(Y ** 2 + X ** 2) / (2.0 * sigma ** 2))
    probe = probe / jnp.sqrt(jnp.sum(jnp.abs(probe) ** 2))
    return probe.astype(jnp.complex128)


# =============================================================================
# 2. Scan Positions
# =============================================================================


def generate_scan_positions(n_y, n_x, step_y, step_x, origin_y=0.0, origin_x=0.0):
    """Generate a regular 2-D scan grid.

    Parameters
    ----------
    n_y, n_x : int
        Number of scan positions along y and x.
    step_y, step_x : float
        Step size in Ångströms.
    origin_y, origin_x : float
        Starting position in Ångströms.

    Returns
    -------
    positions : float64 ndarray, shape (n_y * n_x, 2)
        Each row is a ``(y, x)`` shift in Ångströms.
    """
    ys = origin_y + jnp.arange(n_y, dtype=jnp.float64) * step_y
    xs = origin_x + jnp.arange(n_x, dtype=jnp.float64) * step_x
    Ys, Xs = jnp.meshgrid(ys, xs, indexing="ij")
    return jnp.stack([Ys.ravel(), Xs.ravel()], axis=-1)


# =============================================================================
# 3. Fourier Shift
# =============================================================================


def fourier_shift(field, shift_yx, FY, FX):
    """Shift a 2-D complex field using the Fourier shift theorem.

    The field is shifted by ``(shift_y, shift_x)`` Ångströms via

        ``F_shifted(k) = F(k) * exp(-2πi (FY * shift_y + FX * shift_x))``

    Parameters
    ----------
    field : complex ndarray, shape (ny, nx)
    shift_yx : array_like, shape (2,)
        ``(shift_y, shift_x)`` in Ångströms.
    FY : ndarray, shape (ny, nx)
        Spatial frequencies along *rows* (y-direction) in 1/Å.
    FX : ndarray, shape (ny, nx)
        Spatial frequencies along *columns* (x-direction) in 1/Å.

    Returns
    -------
    shifted : complex ndarray, shape (ny, nx)
    """
    sy, sx = shift_yx[0], shift_yx[1]
    phase = jnp.exp(-1j * 2.0 * jnp.pi * (FY * sy + FX * sx))
    return jnp.fft.ifft2(jnp.fft.fft2(field) * phase)


# =============================================================================
# 4. Forward Models
# =============================================================================


def simulate_ptychography_as(
    object_slices,
    probe,
    positions,
    slice_thickness,
    energy,
    sampling,
):
    """Simulate ptychographic diffraction patterns using Angular Spectrum propagation.

    Forward model for each scan position:

    1. Shift probe to position *p* via Fourier shift theorem.
    2. For each slice *k*: ``wave = wave * t_k``.
       If *k < N−1*: propagate with AS kernel.
    3. ``dp = |FFT2(wave)|²``

    The diffraction patterns are stored **without** ``fftshift``; apply
    ``jnp.fft.fftshift`` before display if a centred pattern is desired.

    Parameters
    ----------
    object_slices : complex ndarray, shape (n_slices, ny, nx)
        Complex transmission functions.  For a single thin object use
        ``object_slices = obj[None]``.
    probe : complex ndarray, shape (ny, nx)
    positions : ndarray, shape (n_pos, 2)
        Scan positions (y, x) in Ångströms.
    slice_thickness : float
        Slice spacing in Ångströms.
    energy : float
        Beam energy in eV.
    sampling : array_like, shape (2,)
        Pixel size (dy, dx) in Ångströms.

    Returns
    -------
    diffraction_patterns : float ndarray, shape (n_pos, ny, nx)
    exit_waves : complex ndarray, shape (n_pos, ny, nx)
    """
    ny, nx = probe.shape
    dy, dx = float(sampling[0]), float(sampling[1])
    fy = jnp.fft.fftfreq(ny, d=dy)
    fx = jnp.fft.fftfreq(nx, d=dx)
    FY, FX = jnp.meshgrid(fy, fx, indexing="ij")

    H_prop = angular_spectrum_propagation_kernel(ny, nx, sampling, slice_thickness, energy)
    n_slices = object_slices.shape[0]

    dps, exit_waves = [], []
    for pos in positions:
        wave = fourier_shift(probe, pos, FY, FX)
        for k in range(n_slices):
            wave = wave * object_slices[k]
            if k < n_slices - 1:
                wave = jnp.fft.ifft2(H_prop * jnp.fft.fft2(wave))
        dps.append(jnp.abs(jnp.fft.fft2(wave)) ** 2)
        exit_waves.append(wave)

    return jnp.stack(dps), jnp.stack(exit_waves)


def simulate_ptychography_wpm(
    potential_slices,
    probe,
    positions,
    slice_thickness,
    energy,
    sampling,
    n_bins=64,
):
    """Simulate ptychographic diffraction patterns using the Wave Propagation Method.

    Forward model for each scan position:

    1. Shift probe to position *p*.
    2. For each slice *k*: ``n_k = refractive_index(V_k)``;
       ``wave = WPM_step(wave, n_k, dz)``.
    3. ``dp = |FFT2(wave)|²``

    Parameters
    ----------
    potential_slices : float ndarray, shape (n_slices, ny, nx)
        Electrostatic potential in V for each slice.
    probe : complex ndarray, shape (ny, nx)
    positions : ndarray, shape (n_pos, 2)
        Scan positions (y, x) in Ångströms.
    slice_thickness : float
        Slice spacing in Ångströms.
    energy : float
        Beam energy in eV.
    sampling : array_like, shape (2,)
        Pixel size (dy, dx) in Ångströms.
    n_bins : int, optional
        Number of refractive-index bins for WPM approximation.

    Returns
    -------
    diffraction_patterns : float ndarray, shape (n_pos, ny, nx)
    exit_waves : complex ndarray, shape (n_pos, ny, nx)
    """
    ny, nx = probe.shape
    dy, dx = float(sampling[0]), float(sampling[1])
    fy = jnp.fft.fftfreq(ny, d=dy)
    fx = jnp.fft.fftfreq(nx, d=dx)
    FY, FX = jnp.meshgrid(fy, fx, indexing="ij")

    n_slices = potential_slices.shape[0]
    dps, exit_waves = [], []

    for pos in positions:
        wave = fourier_shift(probe, pos, FY, FX)
        for k in range(n_slices):
            n_map = electron_refractive_index(potential_slices[k], energy)
            wave, _, _, _ = wpm_step_adaptive(
                wave, n_map, slice_thickness, energy, sampling, n_bins=n_bins
            )
        dps.append(jnp.abs(jnp.fft.fft2(wave)) ** 2)
        exit_waves.append(wave)

    return jnp.stack(dps), jnp.stack(exit_waves)


# =============================================================================
# 5. ePIE – Thin Object
# =============================================================================


def epie_thin(
    diffraction_patterns,
    positions,
    probe_init,
    sampling,
    n_iter=50,
    alpha=1.0,
    beta=0.9,
    update_probe=True,
    seed=0,
):
    """Extended Ptychographical Iterative Engine (ePIE) for thin samples.

    Implements the algorithm from Maiden & Rodenburg (2009).

    Parameters
    ----------
    diffraction_patterns : float ndarray, shape (n_pos, ny, nx)
        Measured diffraction *intensities* (not amplitudes).
    positions : ndarray, shape (n_pos, 2)
        Scan positions (y, x) in Ångströms.
    probe_init : complex ndarray, shape (ny, nx)
        Initial probe estimate (normalised).
    sampling : array_like, shape (2,)
        Pixel size (dy, dx) in Ångströms.
    n_iter : int
        Number of full sweeps through all positions.
    alpha : float
        Object update step-size (0 < alpha ≤ 1).
    beta : float
        Probe update step-size (0 < beta ≤ 1).
    update_probe : bool
        If ``True`` the probe is updated jointly with the object.
    seed : int
        Random seed for position shuffling each iteration.

    Returns
    -------
    obj_rec : complex ndarray, shape (ny, nx)
        Reconstructed object transmission function.
    probe_rec : complex ndarray, shape (ny, nx)
        Reconstructed probe.
    errors : list of float
        Mean Fourier-space amplitude error per iteration.
    """
    ny, nx = probe_init.shape
    dy, dx = float(sampling[0]), float(sampling[1])
    fy = jnp.fft.fftfreq(ny, d=dy)
    fx = jnp.fft.fftfreq(nx, d=dx)
    FY, FX = jnp.meshgrid(fy, fx, indexing="ij")
    eps = 1e-10

    obj = jnp.ones((ny, nx), dtype=jnp.complex128)
    probe = jnp.array(probe_init, dtype=jnp.complex128)
    meas_amp = jnp.sqrt(jnp.maximum(diffraction_patterns, 0.0))
    n_pos = len(positions)
    errors = []
    rng = np.random.default_rng(seed)

    for it in range(n_iter):
        err = 0.0
        for idx in rng.permutation(n_pos):
            pos = positions[idx]
            ma = meas_amp[idx]

            probe_k = fourier_shift(probe, pos, FY, FX)
            exit_wave = probe_k * obj

            psi_f = jnp.fft.fft2(exit_wave)
            amp_c = jnp.abs(psi_f)
            psi_f_c = jnp.where(amp_c > eps, ma * psi_f / amp_c, psi_f)
            d_exit = jnp.fft.ifft2(psi_f_c) - exit_wave
            err += float(jnp.mean(jnp.abs(psi_f_c - psi_f) ** 2))

            # Object update
            pk_max = jnp.max(jnp.abs(probe_k) ** 2)
            obj = obj + alpha * jnp.conj(probe_k) / (pk_max + eps) * d_exit

            # Probe update
            if update_probe:
                ok_max = jnp.max(jnp.abs(obj) ** 2)
                dp = beta * jnp.conj(obj) / (ok_max + eps) * d_exit
                probe = probe + fourier_shift(dp, -pos, FY, FX)

        errors.append(err / n_pos)

    return obj, probe, errors


# =============================================================================
# 6. Multi-slice ePIE – Angular Spectrum
# =============================================================================


def epie_multislice_as(
    diffraction_patterns,
    positions,
    probe_init,
    n_slices,
    slice_thickness,
    energy,
    sampling,
    n_iter=50,
    alpha=1.0,
    beta=0.9,
    update_probe=True,
    seed=0,
):
    """Multi-slice ePIE with Angular Spectrum propagation.

    Implements the multi-slice extension of ePIE from Maiden et al. (2012).
    Suitable for thick samples where a single transmission-function model is
    insufficient.

    **Forward model per position:**

    * ``waves[k]`` = wavefront *entering* slice *k* (before multiplication by ``t_k``).
    * ``waves[0]`` = probe shifted to scan position.
    * After last slice: ``exit_wave = waves[n_slices-1] * t_{n_slices-1}``
      (no propagation after the final slice).

    **Backward update:**

    Starting from ``d_exit = exit_wave_constrained − exit_wave``, corrections
    are back-propagated through each transmission and propagation operator in
    reverse order (adjoint AS propagation = conjugate of H since H is unitary).

    Parameters
    ----------
    diffraction_patterns : float ndarray, shape (n_pos, ny, nx)
    positions : ndarray, shape (n_pos, 2)
    probe_init : complex ndarray, shape (ny, nx)
    n_slices : int
        Number of transmission slices to reconstruct.
    slice_thickness : float
        Slice spacing in Ångströms.
    energy : float
        Beam energy in eV.
    sampling : array_like, shape (2,)
        Pixel size (dy, dx) in Ångströms.
    n_iter, alpha, beta, update_probe, seed : see ``epie_thin``.

    Returns
    -------
    slices_rec : complex ndarray, shape (n_slices, ny, nx)
    probe_rec : complex ndarray, shape (ny, nx)
    errors : list of float
    """
    ny, nx = probe_init.shape
    dy, dx = float(sampling[0]), float(sampling[1])
    fy = jnp.fft.fftfreq(ny, d=dy)
    fx = jnp.fft.fftfreq(nx, d=dx)
    FY, FX = jnp.meshgrid(fy, fx, indexing="ij")

    H_prop = angular_spectrum_propagation_kernel(ny, nx, sampling, slice_thickness, energy)
    # Adjoint = inverse for a unitary AS propagator
    H_adj = jnp.conj(H_prop)
    eps = 1e-10

    slices = jnp.ones((n_slices, ny, nx), dtype=jnp.complex128)
    probe = jnp.array(probe_init, dtype=jnp.complex128)
    meas_amp = jnp.sqrt(jnp.maximum(diffraction_patterns, 0.0))
    n_pos = len(positions)
    errors = []
    rng = np.random.default_rng(seed)

    for it in range(n_iter):
        err = 0.0
        for idx in rng.permutation(n_pos):
            pos = positions[idx]
            ma = meas_amp[idx]
            probe_k = fourier_shift(probe, pos, FY, FX)

            # ---- Forward pass: record wavefronts entering each slice --------
            waves = []
            wave = probe_k
            for k in range(n_slices):
                waves.append(wave)           # waves[k] = input to slice k
                wave = wave * slices[k]       # apply transmission
                if k < n_slices - 1:
                    wave = jnp.fft.ifft2(H_prop * jnp.fft.fft2(wave))
            exit_wave = wave                  # after last transmission

            # ---- Fourier constraint -----------------------------------------
            psi_f = jnp.fft.fft2(exit_wave)
            amp_c = jnp.abs(psi_f)
            psi_f_c = jnp.where(amp_c > eps, ma * psi_f / amp_c, psi_f)
            exit_wave_c = jnp.fft.ifft2(psi_f_c)
            err += float(jnp.mean(jnp.abs(psi_f_c - psi_f) ** 2))

            # ---- Backward pass ----------------------------------------------
            # d is the correction at the *output* of transmission k
            d = exit_wave_c - exit_wave
            new_slices = list(slices)
            d_probe_k = None

            for k in range(n_slices - 1, -1, -1):
                psi_in = waves[k]   # input to slice k
                t_k = slices[k]

                # d is correction at the output of t_k.
                # For k < n_slices-1 we must first backpropagate through the
                # propagator that was applied after t_k in the forward pass.
                if k < n_slices - 1:
                    d_at_t_out = jnp.fft.ifft2(H_adj * jnp.fft.fft2(d))
                else:
                    d_at_t_out = d

                # Update transmission function
                pin_max = jnp.max(jnp.abs(psi_in) ** 2)
                new_slices[k] = t_k + alpha * jnp.conj(psi_in) / (pin_max + eps) * d_at_t_out

                # Backpropagate through t_k to get correction at waves[k]
                t_max = jnp.max(jnp.abs(t_k) ** 2)
                d_through_t = beta * jnp.conj(t_k) / (t_max + eps) * d_at_t_out

                if k > 0:
                    # Backpropagate through the propagator between slice k-1 and k
                    d = jnp.fft.ifft2(H_adj * jnp.fft.fft2(d_through_t))
                else:
                    d_probe_k = d_through_t

            slices = jnp.stack(new_slices)

            # Probe update (un-shift the correction back to the probe frame)
            if update_probe and d_probe_k is not None:
                t0_max = jnp.max(jnp.abs(new_slices[0]) ** 2)
                dp_shift = beta * jnp.conj(new_slices[0]) / (t0_max + eps) * d_probe_k
                probe = probe + fourier_shift(dp_shift, -pos, FY, FX)

        errors.append(err / n_pos)

    return slices, probe, errors


# =============================================================================
# 7. Gradient-based Reconstruction – Angular Spectrum
# =============================================================================


def reconstruct_as(
    measured_dps,
    positions,
    probe,
    n_slices,
    slice_thickness,
    energy,
    sampling,
    n_iter=200,
    learning_rate=1e-2,
    init_object_slices=None,
):
    """Gradient-based multi-slice ptychography with Angular Spectrum propagation.

    Minimises an amplitude-NLLS loss

        ``L = mean_pos mean_pixel ( |Ψ_pred| − √I_meas )²``

    using the Adam optimiser.  The object is parameterised as a complex
    transmission-function array, split into real and imaginary parts so that
    standard real-valued optimisers can be used.

    Parameters
    ----------
    measured_dps : float ndarray, shape (n_pos, ny, nx)
        Measured diffraction intensities.
    positions : ndarray, shape (n_pos, 2)
        Scan positions (y, x) in Ångströms.
    probe : complex ndarray, shape (ny, nx)
        Known (fixed) probe estimate.
    n_slices : int
    slice_thickness : float in Ångströms
    energy : float in eV
    sampling : (dy, dx) in Ångströms
    n_iter : int
    learning_rate : float
    init_object_slices : optional complex ndarray, shape (n_slices, ny, nx)
        Initial guess.  Defaults to all-ones (uniform transmission).

    Returns
    -------
    object_slices_rec : complex ndarray, shape (n_slices, ny, nx)
    losses : list of float
    """
    ny, nx = probe.shape
    dy, dx = float(sampling[0]), float(sampling[1])
    fy = jnp.fft.fftfreq(ny, d=dy)
    fx = jnp.fft.fftfreq(nx, d=dx)
    FY, FX = jnp.meshgrid(fy, fx, indexing="ij")

    H_prop = angular_spectrum_propagation_kernel(ny, nx, sampling, slice_thickness, energy)

    if init_object_slices is None:
        init_re = np.ones((n_slices, ny, nx), dtype=np.float64)
        init_im = np.zeros((n_slices, ny, nx), dtype=np.float64)
    else:
        init_re = np.real(np.asarray(init_object_slices))
        init_im = np.imag(np.asarray(init_object_slices))

    params = {
        "obj_re": jnp.array(init_re),
        "obj_im": jnp.array(init_im),
    }

    meas_amp = jnp.sqrt(jnp.maximum(jnp.array(measured_dps, dtype=jnp.float64), 0.0))
    probe_j = jnp.array(probe, dtype=jnp.complex128)
    pos_j = jnp.array(positions)
    H_j = jnp.array(H_prop)
    FY_j, FX_j = jnp.array(FY), jnp.array(FX)
    n_pos = len(positions)

    def _forward_pos(obj, pos):
        wave = fourier_shift(probe_j, pos, FY_j, FX_j)
        for k in range(n_slices):
            wave = wave * obj[k]
            if k < n_slices - 1:
                wave = jnp.fft.ifft2(H_j * jnp.fft.fft2(wave))
        return jnp.abs(jnp.fft.fft2(wave))

    def loss_fn(params):
        obj = params["obj_re"] + 1j * params["obj_im"]
        total = jnp.zeros(())
        for i in range(n_pos):
            amp_pred = _forward_pos(obj, pos_j[i])
            total = total + jnp.mean((amp_pred - meas_amp[i]) ** 2)
        return total / n_pos

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    losses = []
    for i in range(n_iter):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))

    object_slices_rec = params["obj_re"] + 1j * params["obj_im"]
    return object_slices_rec, losses


# =============================================================================
# 8. Gradient-based Reconstruction – WPM
# =============================================================================


def reconstruct_wpm(
    measured_dps,
    positions,
    probe,
    n_slices,
    slice_thickness,
    energy,
    sampling,
    n_iter=200,
    learning_rate=1e-2,
    n_bins=32,
    init_potentials=None,
):
    """Gradient-based multi-slice ptychography with the Wave Propagation Method.

    The object is parameterised as electrostatic potential slices (real-valued).
    At each gradient step the WPM step is used as the forward model; JAX
    autodiff propagates gradients back through the refractive-index calculation
    and the binned WPM approximation.

    Minimises the same amplitude-NLLS loss as ``reconstruct_as``.

    Parameters
    ----------
    measured_dps : float ndarray, shape (n_pos, ny, nx)
    positions : ndarray, shape (n_pos, 2)
    probe : complex ndarray, shape (ny, nx)
    n_slices : int
    slice_thickness : float in Ångströms
    energy : float in eV
    sampling : (dy, dx) in Ångströms
    n_iter : int
    learning_rate : float
    n_bins : int
        Refractive-index bins for WPM approximation.
    init_potentials : optional float ndarray, shape (n_slices, ny, nx)
        Initial potential guess.  Defaults to all zeros, but note that a
        zero-everywhere potential makes the WPM refractive-index range
        degenerate (n_min = n_max), which zeroes the binning-interpolation
        gradient.  Providing a non-trivial initial guess (e.g. small random
        values) is therefore recommended.

    Returns
    -------
    potential_slices_rec : float ndarray, shape (n_slices, ny, nx)
    losses : list of float
    """
    ny, nx = probe.shape
    dy, dx = float(sampling[0]), float(sampling[1])
    fy = jnp.fft.fftfreq(ny, d=dy)
    fx = jnp.fft.fftfreq(nx, d=dx)
    FY, FX = jnp.meshgrid(fy, fx, indexing="ij")

    if init_potentials is None:
        v_init = jnp.zeros((n_slices, ny, nx), dtype=jnp.float64)
    else:
        v_init = jnp.array(init_potentials, dtype=jnp.float64)

    meas_amp = jnp.sqrt(jnp.maximum(jnp.array(measured_dps, dtype=jnp.float64), 0.0))
    probe_j = jnp.array(probe, dtype=jnp.complex128)
    pos_j = jnp.array(positions)
    FY_j, FX_j = jnp.array(FY), jnp.array(FX)
    n_pos = len(positions)

    def _forward_pos(potential_slices, pos):
        wave = fourier_shift(probe_j, pos, FY_j, FX_j)
        for k in range(n_slices):
            n_map = electron_refractive_index(potential_slices[k], energy)
            wave, _, _, _ = wpm_step_adaptive(
                wave, n_map, slice_thickness, energy, sampling, n_bins=n_bins
            )
        return jnp.abs(jnp.fft.fft2(wave))

    def loss_fn(potential_slices):
        total = jnp.zeros(())
        for i in range(n_pos):
            amp_pred = _forward_pos(potential_slices, pos_j[i])
            total = total + jnp.mean((amp_pred - meas_amp[i]) ** 2)
        return total / n_pos

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(v_init)

    @jax.jit
    def step(v, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(v)
        updates, new_state = optimizer.update(grads, opt_state)
        new_v = optax.apply_updates(v, updates)
        return new_v, new_state, loss

    losses = []
    v = v_init
    for i in range(n_iter):
        v, opt_state, loss = step(v, opt_state)
        losses.append(float(loss))

    return v, losses


# =============================================================================
# 9. Test-Phantom Helpers
# =============================================================================


def make_phase_object(ny, nx, n_features=5, max_phase=np.pi / 4, seed=42):
    """Create a random phase-only test object.

    Returns a complex transmission function ``exp(i * φ(y, x))`` composed of
    randomly placed circular discs with random phases.

    Parameters
    ----------
    ny, nx : int
    n_features : int
        Number of phase discs.
    max_phase : float
        Maximum phase in radians.
    seed : int

    Returns
    -------
    obj : complex128 ndarray, shape (ny, nx)
    """
    rng = np.random.default_rng(seed)
    phase = np.zeros((ny, nx))
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    for _ in range(n_features):
        cy = rng.integers(ny // 4, 3 * ny // 4)
        cx = rng.integers(nx // 4, 3 * nx // 4)
        r = rng.integers(max(2, ny // 16), max(3, ny // 6))
        phi = rng.uniform(-float(max_phase), float(max_phase))
        phase[(y - cy) ** 2 + (x - cx) ** 2 <= r ** 2] += phi
    return jnp.array(np.exp(1j * phase), dtype=jnp.complex128)


def make_potential_phantom(ny, nx, n_slices, peak_potential=5.0, seed=42):
    """Create a multi-slice electrostatic potential phantom for testing.

    Each slice contains randomly placed Gaussian blobs representing atom
    columns.

    Parameters
    ----------
    ny, nx : int
    n_slices : int
    peak_potential : float
        Peak potential in V.
    seed : int

    Returns
    -------
    potentials : float64 ndarray, shape (n_slices, ny, nx)
    """
    rng = np.random.default_rng(seed)
    potentials = np.zeros((n_slices, ny, nx))
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    for s in range(n_slices):
        for _ in range(rng.integers(3, 8)):
            cy = rng.uniform(ny // 4, 3 * ny // 4)
            cx = rng.uniform(nx // 4, 3 * nx // 4)
            sigma = rng.uniform(2.0, max(3.0, ny // 16))
            V0 = rng.uniform(1.0, float(peak_potential))
            potentials[s] += V0 * np.exp(
                -((y - cy) ** 2 + (x - cx) ** 2) / (2.0 * sigma ** 2)
            )
    return jnp.array(potentials, dtype=jnp.float64)
