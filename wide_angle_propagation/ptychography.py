"""
Multislice Ptychography in JAX

Gradient-descent-based ptychographic reconstruction supporting both
Fresnel and WPM propagators for inter-slice propagation.

Forward model:
    For each probe position r_j:
        1. Shift probe to position r_j
        2. For each slice i = 0 .. N-1:
            a. Multiply wavefront by transmission function T_i
            b. Propagate by dz using chosen propagator
        3. Record |FFT(exit_wave)|^2 as diffraction pattern

Inverse:
    Minimise  L = sum_j || sqrt(I_meas_j) - |FFT(psi_exit_j)| ||^2
    with respect to {T_i} using Adam optimiser on the object
    transmission functions.
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


# =========================================================================
# Propagation kernels
# =========================================================================

def make_fresnel_kernel(ny, nx, sampling, dz, wavelength):
    """Precompute Fresnel free-space propagation kernel."""
    dy, dx = sampling
    fy = jnp.fft.fftfreq(ny, dy)
    fx = jnp.fft.fftfreq(nx, dx)
    Fx, Fy = jnp.meshgrid(fx, fy, indexing='ij')
    H = jnp.exp(1j * (2 * jnp.pi / wavelength) * dz) * jnp.exp(
        -1j * jnp.pi * wavelength * dz * (Fx**2 + Fy**2)
    )
    return H


def make_angular_spectrum_kernel(ny, nx, sampling, dz, wavelength):
    """Precompute Angular Spectrum propagation kernel."""
    dy, dx = sampling
    fy = jnp.fft.fftfreq(ny, dy)
    fx = jnp.fft.fftfreq(nx, dx)
    Fx, Fy = jnp.meshgrid(fx, fy, indexing='ij')
    kz = jnp.sqrt(jnp.array(
        (1 / wavelength)**2 - Fx**2 - Fy**2, dtype=jnp.complex128
    ))
    H = jnp.exp(1j * 2 * jnp.pi * dz * kz)
    return H


def propagate_fresnel(wave, kernel):
    """Propagate using precomputed Fresnel/AS kernel."""
    return jnp.fft.ifft2(kernel * jnp.fft.fft2(wave))


def _smoothstep(x):
    x = jnp.clip(x, 0.0, 1.0)
    return 3 * x**2 - 2 * x**3


def _get_polynomial_bins(n_min, n_max, n_bins, power=2.0):
    t = jnp.linspace(0, 1, n_bins)
    t_warped = t**power
    return n_min + (n_max - n_min) * t_warped


def propagate_wpm(wave, n_map, dz, wavelength, sampling, n_bins=64, power_spacing=2.0):
    """
    WPM propagation step with adaptive binning.
    
    Parameters
    ----------
    wave : (ny, nx) complex array
    n_map : (ny, nx) refractive index map
    dz : slice thickness in Angstroms
    wavelength : electron wavelength in Angstroms
    sampling : (dy, dx) pixel size in Angstroms
    n_bins : number of refractive index bins
    power_spacing : polynomial power for bin spacing
    
    Returns
    -------
    propagated wave : (ny, nx) complex array
    """
    ny, nx = wave.shape
    k0 = 2 * jnp.pi / wavelength

    dy, dx = sampling
    ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    Kx, Ky = jnp.meshgrid(kx, ky, indexing='ij')
    k_perp2 = Kx**2 + Ky**2

    Ek = jnp.fft.fft2(wave)

    n_min, n_max = n_map.min(), n_map.max()
    n_refs = _get_polynomial_bins(n_min, n_max, n_bins, power=power_spacing)

    # Batch propagation for all reference indices
    def _prop_single(n_val):
        kz = jnp.sqrt(jnp.array(n_val**2 * k0**2 - k_perp2, dtype=jnp.complex128))
        H = jnp.exp(1j * dz * kz)
        return jnp.fft.ifft2(H * Ek)

    ref_fields = jax.vmap(_prop_single)(n_refs)

    # Interpolate between bins
    idx_R = jnp.searchsorted(n_refs, n_map)
    idx_R = jnp.clip(idx_R, 1, n_bins - 1)
    idx_L = idx_R - 1

    n_L = n_refs[idx_L]
    n_R = n_refs[idx_R]

    denom = n_R - n_L
    w_raw = (n_map - n_L) / jnp.where(denom == 0, 1.0, denom)
    w = _smoothstep(w_raw)

    field_L = jnp.take_along_axis(ref_fields, idx_L[None, ...], axis=0).squeeze(axis=0)
    field_R = jnp.take_along_axis(ref_fields, idx_R[None, ...], axis=0).squeeze(axis=0)

    return (1 - w) * field_L + w * field_R


# =========================================================================
# Forward model
# =========================================================================

def make_probe(ny, nx, sampling, energy, semiangle_mrad,
               defocus=0.0, c3=0.0):
    """
    Create a converged STEM probe.
    
    Parameters
    ----------
    ny, nx : grid dimensions
    sampling : (dy, dx) in Angstroms
    energy : beam energy in eV
    semiangle_mrad : convergence semi-angle in mrad
    defocus : defocus in Angstroms (positive = underfocus)
    c3 : spherical aberration C3 in Angstroms
    
    Returns
    -------
    probe : (ny, nx) complex array, normalised so sum |probe|^2 = 1
    """
    from wide_angle_propagation.propagation import energy2wavelength
    wavelength = float(energy2wavelength(jnp.float64(energy)))

    dy, dx = sampling
    fy = jnp.fft.fftfreq(ny, dy)
    fx = jnp.fft.fftfreq(nx, dx)
    Fx, Fy = jnp.meshgrid(fx, fy, indexing='ij')
    freq2 = Fx**2 + Fy**2
    freq = jnp.sqrt(freq2)

    # Aperture: frequencies below semiangle
    max_freq = semiangle_mrad * 1e-3 / wavelength
    aperture = (freq <= max_freq).astype(jnp.float64)

    # Aberration function: chi = pi * lambda * df * f^2 + 0.5*pi*C3*lambda^3*f^4
    chi = (jnp.pi * wavelength * defocus * freq2
           + 0.5 * jnp.pi * c3 * wavelength**3 * freq2**2)

    probe_ft = aperture * jnp.exp(-1j * chi)
    probe = jnp.fft.ifft2(probe_ft)

    # Normalise
    norm = jnp.sqrt(jnp.sum(jnp.abs(probe)**2))
    probe = probe / jnp.where(norm > 0, norm, 1.0)

    return probe


def shift_probe(probe, shift_pixels):
    """
    Shift probe by (dy_pix, dx_pix) using Fourier shift theorem.
    shift_pixels: (2,) array of (row_shift, col_shift) in pixels.
    """
    ny, nx = probe.shape
    fy = jnp.fft.fftfreq(ny)
    fx = jnp.fft.fftfreq(nx)
    Fx, Fy = jnp.meshgrid(fx, fy, indexing='ij')

    shift_phase = jnp.exp(
        -2j * jnp.pi * (Fx * shift_pixels[0] + Fy * shift_pixels[1])
    )
    return jnp.fft.ifft2(shift_phase * jnp.fft.fft2(probe))


def multislice_forward_fresnel(probe, transmissions, kernel):
    """
    Multislice forward model with Fresnel/AS propagator.
    
    Parameters
    ----------
    probe : (ny, nx) complex - incident probe
    transmissions : (N, ny, nx) complex - transmission functions per slice
    kernel : (ny, nx) complex - precomputed propagation kernel
    
    Returns
    -------
    exit_wave : (ny, nx) complex
    """
    wave = probe
    N = transmissions.shape[0]
    for i in range(N):
        wave = wave * transmissions[i]
        wave = propagate_fresnel(wave, kernel)
    return wave


def multislice_forward_fresnel_scan(probe, transmissions, kernel, positions_pix):
    """
    Run multislice forward model for multiple scan positions.
    Uses jax.vmap for parallelism.
    
    Parameters
    ----------
    probe : (ny, nx) complex
    transmissions : (N, ny, nx) complex
    kernel : (ny, nx) complex
    positions_pix : (J, 2) float - probe shifts in pixels
    
    Returns
    -------
    diffraction_patterns : (J, ny, nx) float
    """
    def _single_position(pos):
        shifted = shift_probe(probe, pos)
        exit_wave = multislice_forward_fresnel(shifted, transmissions, kernel)
        ft = jnp.fft.fftshift(jnp.fft.fft2(exit_wave))
        return jnp.abs(ft)**2

    return jax.vmap(_single_position)(positions_pix)


def multislice_forward_wpm(probe, n_maps, dz, wavelength, sampling,
                           n_bins=64, power_spacing=2.0):
    """
    Multislice forward model with WPM propagator.
    
    Parameters
    ----------
    probe : (ny, nx) complex
    n_maps : (N, ny, nx) float - refractive index maps per slice
    dz : slice thickness (Angstroms)
    wavelength : electron wavelength (Angstroms)
    sampling : (dy, dx)
    
    Returns
    -------
    exit_wave : (ny, nx) complex
    """
    wave = probe
    N = n_maps.shape[0]
    for i in range(N):
        # For WPM: transmission is embedded in the propagation step
        # The refractive index map encodes both the phase grating and propagation
        wave = propagate_wpm(wave, n_maps[i], dz, wavelength, sampling,
                             n_bins=n_bins, power_spacing=power_spacing)
    return wave


# =========================================================================
# Loss functions
# =========================================================================

def amplitude_loss(predicted_dp, measured_dp):
    """
    Amplitude-based loss: || sqrt(I_pred) - sqrt(I_meas) ||^2
    
    This is standard in ptychography — matching amplitudes rather than
    intensities gives better gradient conditioning.
    Uses epsilon to avoid NaN gradients from sqrt(0).
    """
    eps = 1e-30
    amp_pred = jnp.sqrt(predicted_dp + eps)
    amp_meas = jnp.sqrt(measured_dp + eps)
    return jnp.mean((amp_pred - amp_meas)**2)


def intensity_loss(predicted_dp, measured_dp):
    """L2 loss on intensities: || I_pred - I_meas ||^2"""
    return jnp.mean((predicted_dp - measured_dp)**2)


# =========================================================================
# Reconstruction engine
# =========================================================================

class MultislicePtychographyReconstructor:
    """
    Gradient-descent multislice ptychography reconstruction.
    
    Reconstructs object transmission functions from 4D-STEM data.
    Supports both Fresnel and WPM propagators.
    Uses optax for optimisation (Adam by default, or any optax optimizer).
    
    Parameters
    ----------
    measured_dps : (J, ny, nx) float - measured diffraction patterns
    probe : (ny, nx) complex - known probe function
    positions_pix : (J, 2) float - scan positions in pixels
    n_slices : int - number of reconstruction slices
    dz : float - slice thickness (Angstroms)
    sampling : (dy, dx) - pixel size (Angstroms)
    energy : float - beam energy (eV)
    propagator : str - 'fresnel', 'angular_spectrum', or 'wpm'
    n_bins : int - WPM bins (only used if propagator='wpm')
    learning_rate : float - learning rate for default Adam optimizer
    optimizer : optax optimizer - if provided, overrides learning_rate
    """

    def __init__(
        self,
        measured_dps,
        probe,
        positions_pix,
        n_slices,
        dz,
        sampling,
        energy,
        propagator='fresnel',
        n_bins=64,
        power_spacing=2.0,
        learning_rate=1e-3,
        loss_fn='amplitude',
        optimizer=None,
    ):
        import optax
        from wide_angle_propagation.propagation import energy2wavelength

        self.measured_dps = jnp.asarray(measured_dps, dtype=jnp.float64)
        self.probe = jnp.asarray(probe, dtype=jnp.complex128)
        self.positions_pix = jnp.asarray(positions_pix, dtype=jnp.float64)
        self.n_slices = n_slices
        self.dz = float(dz)
        self.sampling = (float(sampling[0]), float(sampling[1]))
        self.energy = float(energy)
        self.propagator = propagator
        self.n_bins = n_bins
        self.power_spacing = power_spacing

        self.wavelength = float(energy2wavelength(jnp.float64(energy)))

        ny, nx = probe.shape
        self.ny = ny
        self.nx = nx

        if loss_fn == 'amplitude':
            self._loss_fn = amplitude_loss
        else:
            self._loss_fn = intensity_loss

        # Precompute Fresnel/AS kernel
        if propagator == 'fresnel':
            self.kernel = make_fresnel_kernel(ny, nx, sampling, dz, self.wavelength)
        elif propagator == 'angular_spectrum':
            self.kernel = make_angular_spectrum_kernel(ny, nx, sampling, dz, self.wavelength)
        else:
            self.kernel = None

        # Initialise object transmission functions as identity (vacuum)
        # We parameterise as (amplitude, phase) to ensure physicality
        self._obj_phase = jnp.zeros((n_slices, ny, nx), dtype=jnp.float64)
        self._obj_amp = jnp.ones((n_slices, ny, nx), dtype=jnp.float64)

        # Optax optimizer
        if optimizer is not None:
            self._optax = optimizer
        else:
            self._optax = optax.adam(learning_rate)

        self._opt_state = self._optax.init((self._obj_phase, self._obj_amp))

    def get_transmissions(self):
        """Convert (amplitude, phase) to complex transmission functions."""
        return self._obj_amp * jnp.exp(1j * self._obj_phase)

    def _forward_single(self, obj_phase, obj_amp, pos):
        """Forward model for a single scan position."""
        transmissions = obj_amp * jnp.exp(1j * obj_phase)
        shifted = shift_probe(self.probe, pos)

        if self.propagator in ('fresnel', 'angular_spectrum'):
            exit_wave = multislice_forward_fresnel(shifted, transmissions, self.kernel)
        elif self.propagator == 'wpm':
            # WPM reconstruction: convert phase to refractive index map
            # phase = 2*pi*(n-1)*dz/wavelength  =>  n = 1 + phase*wavelength/(2*pi*dz)
            n_maps = 1.0 + obj_phase * self.wavelength / (2 * jnp.pi * self.dz)
            wave = shifted
            N = n_maps.shape[0]
            for i in range(N):
                # Apply amplitude modulation before WPM step
                wave = wave * obj_amp[i]
                wave = propagate_wpm(
                    wave, n_maps[i], self.dz, self.wavelength, self.sampling,
                    n_bins=self.n_bins, power_spacing=self.power_spacing,
                )
            exit_wave = wave
        else:
            raise ValueError(f"Unknown propagator: {self.propagator}")

        ft = jnp.fft.fftshift(jnp.fft.fft2(exit_wave))
        return jnp.abs(ft)**2

    def _total_loss(self, obj_phase, obj_amp):
        """Total loss over all scan positions."""
        def _single_loss(pos, meas):
            pred = self._forward_single(obj_phase, obj_amp, pos)
            return self._loss_fn(pred, meas)

        losses = jax.vmap(_single_loss)(self.positions_pix, self.measured_dps)
        return jnp.mean(losses)

    def step(self):
        """Perform one optimisation step. Returns current loss."""
        loss, grads = jax.value_and_grad(
            self._total_loss, argnums=(0, 1)
        )(self._obj_phase, self._obj_amp)

        updates, self._opt_state = self._optax.update(
            grads, self._opt_state, (self._obj_phase, self._obj_amp)
        )
        new_phase, new_amp = jax.tree.map(
            lambda p, u: p + u,
            (self._obj_phase, self._obj_amp),
            updates,
        )

        # Clamp amplitude to [0, 2] for stability
        self._obj_phase = new_phase
        self._obj_amp = jnp.clip(new_amp, 0.0, 2.0)

        return float(loss)

    def reconstruct(self, n_iterations=100, verbose=True, callback=None):
        """
        Run reconstruction for n_iterations.
        
        Parameters
        ----------
        n_iterations : int
        verbose : bool - print loss every 10 iterations
        callback : optional callable(step, loss, reconstructor)
        
        Returns
        -------
        losses : list of float
        """
        losses = []
        for i in range(n_iterations):
            loss = self.step()
            losses.append(loss)
            if verbose and (i % 10 == 0 or i == n_iterations - 1):
                print(f"  iter {i:4d}  loss = {loss:.6e}")
            if callback is not None:
                callback(i, loss, self)
        return losses

    def get_recovered_phase(self):
        """Return the reconstructed phase maps (N, ny, nx)."""
        return np.asarray(self._obj_phase)

    def get_recovered_amplitude(self):
        """Return the reconstructed amplitude maps (N, ny, nx)."""
        return np.asarray(self._obj_amp)

    def get_recovered_transmission(self):
        """Return complex transmission function (N, ny, nx)."""
        return np.asarray(self.get_transmissions())


# =========================================================================
# Convenience: simulate 4D-STEM dataset
# =========================================================================

def simulate_4dstem(
    potential_slices,
    probe,
    positions_pix,
    dz,
    sampling,
    energy,
    propagator='fresnel',
    n_bins=64,
    power_spacing=2.0,
    add_noise=False,
    noise_counts=1e4,
):
    """
    Simulate a 4D-STEM dataset from a known potential.
    
    Parameters
    ----------
    potential_slices : (N, ny, nx) float - electrostatic potential per slice (V)
    probe : (ny, nx) complex
    positions_pix : (J, 2) float
    dz : float - slice thickness (Angstroms)
    sampling : (dy, dx)
    energy : float - beam energy (eV)
    propagator : 'fresnel' or 'wpm'
    add_noise : bool - add Poisson noise
    noise_counts : float - total counts per pattern for Poisson noise
    
    Returns
    -------
    dps : (J, ny, nx) float - diffraction patterns
    transmissions : (N, ny, nx) complex - ground truth transmission functions
    """
    from wide_angle_propagation.propagation import (
        energy2wavelength, electron_refractive_index
    )

    wavelength = float(energy2wavelength(jnp.float64(energy)))
    ny, nx = probe.shape

    # Build ground truth transmission functions
    transmissions = []
    n_maps = []
    for i in range(potential_slices.shape[0]):
        n = electron_refractive_index(potential_slices[i], energy)
        n_maps.append(n)
        phase = 2 * jnp.pi * (n - 1) * dz / wavelength
        t = jnp.exp(1j * phase)
        transmissions.append(t)
    transmissions = jnp.stack(transmissions)
    n_maps = jnp.stack(n_maps)

    if propagator in ('fresnel', 'angular_spectrum'):
        if propagator == 'fresnel':
            kernel = make_fresnel_kernel(ny, nx, sampling, dz, wavelength)
        else:
            kernel = make_angular_spectrum_kernel(ny, nx, sampling, dz, wavelength)
        dps = multislice_forward_fresnel_scan(
            probe, transmissions, kernel, positions_pix
        )
    elif propagator == 'wpm':
        # For WPM, we need refractive index maps, not transmission functions
        def _single_wpm(pos):
            shifted = shift_probe(probe, pos)
            exit_wave = multislice_forward_wpm(
                shifted, n_maps, dz, wavelength, sampling,
                n_bins=n_bins, power_spacing=power_spacing
            )
            ft = jnp.fft.fftshift(jnp.fft.fft2(exit_wave))
            return jnp.abs(ft)**2

        dps = jax.vmap(_single_wpm)(positions_pix)
    else:
        raise ValueError(f"Unknown propagator: {propagator}")

    if add_noise:
        # Poisson noise: scale to counts, then draw Poisson samples
        key = jax.random.PRNGKey(42)
        total_per_pattern = jnp.sum(dps, axis=(-2, -1), keepdims=True)
        scaled = dps / jnp.where(total_per_pattern > 0, total_per_pattern, 1.0) * noise_counts
        dps = jax.random.poisson(key, scaled).astype(jnp.float64)
        # Re-normalise
        dps = dps / noise_counts * total_per_pattern.squeeze((-2, -1))[..., None, None]

    return np.asarray(dps), np.asarray(transmissions)


# =========================================================================
# Grid scan generation
# =========================================================================

def make_grid_scan(ny, nx, n_scan_y, n_scan_x, margin_pix=4):
    """
    Generate a regular grid of scan positions (in pixels).
    
    Returns (J, 2) array of (row, col) shifts relative to grid centre.
    """
    y_positions = jnp.linspace(margin_pix, ny - margin_pix, n_scan_y)
    x_positions = jnp.linspace(margin_pix, nx - margin_pix, n_scan_x)

    # Centre so that (ny//2, nx//2) maps to zero shift
    cy, cx = ny / 2.0, nx / 2.0
    positions = []
    for y in y_positions:
        for x in x_positions:
            positions.append([float(y - cy), float(x - cx)])

    return jnp.array(positions, dtype=jnp.float64)
