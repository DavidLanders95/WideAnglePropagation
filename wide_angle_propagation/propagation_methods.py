import numpy as np
import functools
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from abtem.multislice import _generate_potential_configurations
from abtem.antialias import AntialiasAperture
from ase import units


# =============================================================================
# 1. Physics Constants & Conversion Utilities
#    (Fundamental functions used by almost all other components)
# =============================================================================

def electron_rest_energy():
    """
    Return the electron rest energy E0 = m_e c^2 in eV.
    """
    m_e = units._me
    c = units._c
    eV = units._e
    return m_e * c**2 / eV


@jax.jit
def relativistic_mass_correction(energy: float) -> float:
    return 1 + units._e * energy / (units._me * units._c**2)


@jax.jit
def energy2mass(energy: float) -> float:
    """
    Calculate relativistic mass from energy.
    Returns: Relativistic mass [kg]
    """
    return relativistic_mass_correction(energy) * units._me


@jax.jit
def energy2wavelength(energy: float) -> float:
    """
    Calculate relativistic de Broglie wavelength from energy.
    Returns: Relativistic de Broglie wavelength [Å].
    """
    return (
        units._hplanck
        * units._c
        / jnp.sqrt(energy * (2 * units._me * units._c**2 / units._e + energy))
        / units._e
        * 1.0e10
    )


@jax.jit
def energy2sigma(energy: float) -> float:
    """
    Calculate interaction parameter (sigma) from energy.
    Returns: Interaction parameter [1 / (Å * eV)].
    """
    return (
        2 * jnp.pi * energy2mass(energy) * units.kg * units._e * units.C *
        energy2wavelength(energy) /
        (units._hplanck * units.s * units.J) ** 2
    )


def electron_refractive_index(potential, energy):
    """
    Calculate refractive index n from electrostatic potential V and energy E.
    """
    E0 = electron_rest_energy()
    E = energy

    # Convert electrostatic potential (V) -> potential energy V (eV)
    # Electron charge is negative, V_potential_energy = -1 * V_electrostatic
    V = -potential
    EminusV = E - V

    numerator = 2 * EminusV * E0 + EminusV**2
    denominator = 2 * E * E0 + E**2

    return jnp.sqrt(numerator / denominator)


def electron_refractive_index_taylor(potential, energy):
    """
    Calculate refractive index using a Taylor expansion approximation.
    n approx 1 + sigma * V
    """
    E0 = electron_rest_energy()
    E = energy

    interaction_factor = (E + E0) / (E * (E + 2 * E0))
    n = 1.0 + interaction_factor * potential

    return n


# =============================================================================
# 2. Math Helpers & Grid Utilities
# =============================================================================

def get_frequencies(n, m, ps):
    """Generate frequency grids for FFT operations."""
    fx = jnp.fft.fftfreq(n, ps[0])
    fy = jnp.fft.fftfreq(m, ps[1])
    Fx, Fy = jnp.meshgrid(fx, fy, indexing='ij')
    return Fx, Fy


def smoothstep(x):
    """
    Implements the smoothstep function: p(z) = 3z^2 - 2z^3 for 0 < z < 1.
    Used for smooth masking between WPM bins.
    """
    x = jnp.clip(x, 0.0, 1.0)
    return 3 * x**2 - 2 * x**3


def get_polynomial_bins(n_min, n_max, n_bins, power=2.0):
    """
    Creates bin edges that are concentrated at the high end (atoms).
    power=1.0: Linear spacing.
    power=2.0: Quadratic spacing (dense at high n, sparse at low n).
    """
    t = jnp.linspace(0, 1, n_bins)
    t_warped = t**power
    return n_min + (n_max - n_min) * t_warped


# =============================================================================
# 3. Propagation Kernels
# =============================================================================

@jax.jit
def shift_kernel(x0, y0, Fx, Fy):
    return jnp.exp(-1j * 2 * jnp.pi * (Fx * x0 + Fy * y0))


def fresnel_propagation_kernel(n: int, m: int, ps: tuple[float, float], z: float, energy: float):
    wavelength = energy2wavelength(energy)
    Fx, Fy = get_frequencies(n, m, ps)

    H = jnp.exp(1j * (2 * jnp.pi / wavelength) * z) * jnp.exp(
        -1j * jnp.pi * wavelength * z * (Fx**2 + Fy**2))
    return H


def angular_spectrum_propagation_kernel(
    n: int, m: int, ps: tuple[float, float], z: float, energy: float
):
    wavelength = energy2wavelength(energy)
    Fx, Fy = get_frequencies(n, m, ps)
    # Ensure complex type to handle evanescent waves (negative values inside sqrt)
    kz = jnp.sqrt(jnp.array((1 / wavelength)**2 - Fx**2 - Fy**2, dtype=jnp.complex128))
    H = jnp.exp(1j * 2 * jnp.pi * z * kz)
    return H


def wpm_propagation_kernel(Ek, n_val, k0, k_perp2, dz):
    """
    Core kernel for Wave Propagation Method (WPM).
    Propagates a wave with a SINGLE homogeneous refractive index n_val.
    """
    kz = jnp.sqrt(jnp.array(n_val**2 * k0**2 - k_perp2, dtype=jnp.complex128))
    H = jnp.exp(1j * dz * kz)
    return jnp.fft.ifft2(H * Ek)


# Vmapped version for processing multiple refractive indices in parallel
wpm_propagation_kernel_vmap = jax.vmap(wpm_propagation_kernel, in_axes=(None, 0, None, None, None))


# =============================================================================
# 4. Propagation Logic (Steppers)
# =============================================================================

def Propagator(u, H):
    """Simple Fourier space multiplication."""
    ufft = jnp.fft.fft2(u)
    return jnp.fft.ifft2(H * ufft)


def wpm_step(wave, n_map, dz, energy, ps):
    """
    Naive WPM step: One FFT/Propagator per pixel.
    Very slow for large grids due to massive over-calculation.
    """
    ny, nx = wave.shape
    wavelength = energy2wavelength(energy)
    k0 = 2 * jnp.pi / wavelength

    # Frequencies -> k_perp^2
    Fy, Fx = get_frequencies(ny, nx, ps)
    kx = 2 * jnp.pi * Fx
    ky = 2 * jnp.pi * Fy
    k_perp2 = kx**2 + ky**2

    Ek = jnp.fft.fft2(wave)
    n_flat = n_map.reshape(-1)

    fields = wpm_propagation_kernel_vmap(Ek, n_flat, k0, k_perp2, dz)

    P = n_flat.size
    p_indices = jnp.arange(P)
    iy, ix = jnp.divmod(p_indices, nx)

    def pick_pixel(field, y, x):
        return field[y, x]

    new_wave_flat = jax.vmap(pick_pixel)(fields, iy, ix)
    return new_wave_flat.reshape(ny, nx)


def wpm_step_adaptive(wave, n_map, dz, energy, ps, n_bins=256, power_spacing=2.0):
    """
    Optimized WPM step using Adaptive Binning and Smoothstep interpolation.
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

    Ek = jnp.fft.fft2(wave)

    n_min, n_max = n_map.min(), n_map.max()
    n_refs = get_polynomial_bins(n_min, n_max, n_bins, power=power_spacing)

    # Compute Propagators (Batch FFT)
    ref_fields = wpm_propagation_kernel_vmap(Ek, n_refs, k0, k_perp2, dz)

    # Find the bin indices for every pixel
    idx_R = jnp.searchsorted(n_refs, n_map)
    idx_R = jnp.clip(idx_R, 1, n_bins - 1)
    idx_L = idx_R - 1

    n_L = n_refs[idx_L]
    n_R = n_refs[idx_R]

    # Calculate interpolation weight
    denom = n_R - n_L
    w_raw = (n_map - n_L) / jnp.where(denom == 0, 1.0, denom)
    w = smoothstep(w_raw)

    field_L = jnp.take_along_axis(ref_fields, idx_L[None, ...], axis=0).squeeze()
    field_R = jnp.take_along_axis(ref_fields, idx_R[None, ...], axis=0).squeeze()

    new_wave = (1 - w) * field_L + w * field_R

    return new_wave, w, idx_L, n_refs


# =============================================================================
# 5. Simulation & Probe Tools
# =============================================================================

def move_probe(probe, new_pos):
    """
    Move the probe by a given shift using array rolling.
    """
    current_pos_row = probe.shape[0] // 2
    current_pos_col = probe.shape[1] // 2
    new_pos_row = new_pos[0]
    new_pos_col = new_pos[1]

    shift_to_row = new_pos_row - current_pos_row
    shift_to_col = new_pos_col - current_pos_col
    shift = jnp.array([shift_to_row, shift_to_col])

    return jnp.roll(probe, shift, axis=(0, 1))


@jax.jit
def transmission_function(potential, energy):
    """Calculates transmission function of a slice."""
    sigma = energy2sigma(energy)
    return jnp.exp(1j * sigma * potential)


def get_abtem_transmit(potential, energy):
    """
    Interfacing with abtem to get transmission functions for multiple slices.
    """
    t_functions = []
    for _, potential_configuration in _generate_potential_configurations(
        potential
    ):
        for potential_slice in potential_configuration.generate_slices():
            t_func = potential_slice.transmission_function(energy=energy)
            t_func = AntialiasAperture().bandlimit(t_func, in_place=False)
            t_functions.append(t_func.array)

    return np.concatenate(t_functions, axis=0)

# =============================================================================
# 6. High-Level Simulation Functions
# =============================================================================

def simulate_fresnel_as(potential, probe, prop_kernel, slice_thickness, energy):
    """
    Simulate propagation using the Fresnel (or Angular Spectrum) method.

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
    wavelength = energy2wavelength(energy)
    wavefront = probe
    wavefronts = []

    # Iterating over the potential slices
    # To support JIT, we might prefer ensuring this loop is unrolled or scanned.
    # Standard Python loop works with JIT if loop bound is static (array shape).

    N = potential.shape[0]

    for i in range(N):
        # Calculate Refractive Index for slice
        n = electron_refractive_index(potential[i], energy)

        # Phase Grating
        phase_shift = jnp.exp(1j * 2 * jnp.pi * (n - 1) * slice_thickness / wavelength)
        wavefront = wavefront * phase_shift

        # Propagation
        wavefront = Propagator(wavefront, prop_kernel)

        wavefronts.append(wavefront)

    exit_wave = wavefront

    detector_wavefront = jnp.fft.fftshift(jnp.fft.fft2(exit_wave))
    diffraction_pattern = (
        jnp.square(detector_wavefront.real)
        + jnp.square(detector_wavefront.imag)
    )

    return exit_wave, diffraction_pattern, jnp.stack(wavefronts)


simulate_fresnel_as_jit = jax.jit(simulate_fresnel_as)


def simulate_wpm(potential, probe, slice_thickness, energy, sampling, n_bins=128, power_spacing=2.0):
    """
    Simulate propagation using the Wave Propagation Method (WPM).
    """
    wavefront = probe
    wavefronts = []

    N = potential.shape[0]

    for i in range(N):
        # Refractive Index
        n = electron_refractive_index(potential[i], energy)

        # WPM Step
        wavefront, _, _, _ = wpm_step_adaptive(
            wavefront, n, slice_thickness, energy, sampling,
            n_bins=n_bins, power_spacing=power_spacing
        )

        wavefronts.append(wavefront)

    exit_wave = wavefront

    detector_wavefront = jnp.fft.fftshift(jnp.fft.fft2(exit_wave))
    diffraction_pattern = (
        jnp.square(detector_wavefront.real)
        + jnp.square(detector_wavefront.imag)
    )

    return exit_wave, diffraction_pattern, jnp.stack(wavefronts)



simulate_wpm_jit = jax.jit(simulate_wpm, static_argnames=('n_bins', 'power_spacing'))


def max_angle_gpts(shape, sampling, wavelength, max_angle_mrad, parity="odd"):
    """
    Return the cropped grid size matching max_angle_mrad.
    """
    ny, nx = shape
    dx, dy = float(sampling[0]), float(sampling[1])
    max_freq = float(max_angle_mrad) / (float(wavelength) * 1000.0)
    dfx = 1.0 / (nx * dx)
    dfy = 1.0 / (ny * dy)

    # Calculate half-widths
    half_x = int(np.floor(max_freq / dfx))
    half_y = int(np.floor(max_freq / dfy))

    # Clamp
    half_x = max(0, min(half_x, nx // 2))
    half_y = max(0, min(half_y, ny // 2))

    if parity == "even":
        new_nx = 2 * half_x
        new_ny = 2 * half_y
    else:
        new_nx = 2 * half_x + 1
        new_ny = 2 * half_y + 1

    new_nx = max(1, min(new_nx, nx))
    new_ny = max(1, min(new_ny, ny))
    return new_ny, new_nx


def downsample_to_max_angle(pattern, sampling, wavelength, max_angle_mrad, parity="odd"):
    """
    Center-crop a fftshifted diffraction pattern to max_angle_mrad.
    """
    arr = np.asarray(pattern)
    ny, nx = arr.shape
    new_ny, new_nx = max_angle_gpts(arr.shape, sampling, wavelength, max_angle_mrad, parity=parity)

    if (new_ny, new_nx) == (ny, nx):
        return arr

    cy, cx = ny // 2, nx // 2
    y0 = cy - new_ny // 2
    x0 = cx - new_nx // 2

    return arr[y0:y0 + new_ny, x0:x0 + new_nx]



# =============================================================================
# 7. 1D Helpers for Notebook / Line-Propagation
# =============================================================================

def get_kx_1d(n: int, dx: float):
    """Return angular spatial frequencies kx for a 1D grid.

    Returns values in units of rad / unit_length (i.e., multiplied by 2*pi).
    """
    return 2 * jnp.pi * jnp.fft.fftfreq(n, d=dx)


def fresnel_kernel_1d(kx_sq, k0, dy):
    """Fresnel (paraxial) 1D transfer function H(kx) = exp(-i kx^2 dy / (2 k0))."""
    return jnp.exp(-1j * kx_sq * dy / (2 * k0))


def angular_spectrum_kernel_1d(kx_sq, k0, dy):
    """Angular spectrum 1D kernel H(kx) = exp(i dy * sqrt(k0^2 - kx^2))."""
    kz = jnp.sqrt(jnp.complex128(k0**2 - kx_sq))
    return jnp.exp(1j * dy * kz)


@jax.jit
def propagate_1d(psi, H):
    """Propagate a 1D field by multiplying its FFT by transfer function H."""
    psi_k = jnp.fft.fft(psi)
    return jnp.fft.ifft(psi_k * H)


@jax.jit
def wpm_kernel_1d(psi_k, n_val, k0, kx_sq, dy):
    """WPM exact 1D kernel for a single refractive index value.

    psi_k: 1D FFT of input field
    n_val: scalar refractive index
    k0: vacuum wavenumber
    kx_sq: array of kx^2 values
    dy: propagation step
    Returns: propagated field in spatial domain (ifft)
    """
    kz = jnp.sqrt(jnp.complex128((n_val * k0)**2 - kx_sq))
    H = jnp.exp(1j * dy * kz)
    return jnp.fft.ifft(psi_k * H)


# Vmap version for batching over reference refractive indices
wpm_kernel_1d_vmap = jax.vmap(wpm_kernel_1d, in_axes=(None, 0, None, None, None))


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
    dphi = jnp.fft.ifft2(k_perp_sq * psi_k) - k0_sq_n_sq * psi

    return jnp.stack([phi.real, phi.imag, dphi.real, dphi.imag])


def _kg_forward_vacuum_phi(psi, k0, k_perp_sq):
    """Return the exact forward vacuum derivative for an arbitrary probe."""
    psi_k = jnp.fft.fft2(psi)
    kz = jnp.sqrt(jnp.array(k0**2 - k_perp_sq, dtype=jnp.complex128))
    kz = jnp.where(jnp.imag(kz) < 0, -kz, kz)
    return jnp.fft.ifft2(1j * kz * psi_k)


def simulate_kg_ode_full(potential, probe, slice_thickness, energy,
                         sampling, initial_phi=None, rtol=1e-8,
                         atol=1e-10, save_wavefronts=True,
                         solver_name="dopri8", max_steps=None):
    """Solve the full non-paraxial second-order KG equation.

    This restores the true KG ODE system,

        d²ψ/dz² + (∇²⊥ + k₀² n²) ψ = 0,

    rather than the slowly-varying-envelope approximation obtained by
    dropping u″ after ψ = u·exp(i·k₀·z).

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
    k0 = 2 * jnp.pi / wavelength
    ny, nx = probe.shape

    dy, dx = sampling
    ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)
    kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    Kx, Ky = jnp.meshgrid(kx, ky)
    k_perp_sq = Kx**2 + Ky**2

    N_slices = potential.shape[0]
    total_thickness = N_slices * slice_thickness

    all_k0_sq_n_sq = jnp.stack([
        k0**2 * electron_refractive_index(potential[i], energy) ** 2
        for i in range(N_slices)
    ])

    psi0 = jnp.asarray(probe, dtype=jnp.complex128)
    if initial_phi is None:
        phi0 = _kg_forward_vacuum_phi(psi0, k0, k_perp_sq)
    else:
        phi0 = jnp.asarray(initial_phi, dtype=jnp.complex128)

    y0 = jnp.stack([psi0.real, psi0.imag, phi0.real, phi0.imag])

    omega_bound_sq = jnp.max(k_perp_sq) + jnp.max(jnp.abs(all_k0_sq_n_sq))
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

    stepsize_controller = diffrax.PIDController(
        rtol=rtol, atol=atol, dtmax=dtmax,
    )
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

    detector_wavefront = jnp.fft.fftshift(jnp.fft.fft2(exit_wave))
    diffraction_pattern = (
        jnp.square(detector_wavefront.real)
        + jnp.square(detector_wavefront.imag)
    )

    return exit_wave, exit_phi, diffraction_pattern, wavefronts


def _record_amplitudes_vectorized(states, beam_indices, gpts):
    """Convert (n_thickness, N_beams) states to {(h,k): amplitudes} dict."""
    ny, nx = gpts
    bi = np.asarray(beam_indices)
    abs_s = np.abs(np.asarray(states))
    h = np.where(bi[:, 1] <= nx // 2, bi[:, 1], bi[:, 1] - nx)
    k = np.where(bi[:, 0] <= ny // 2, bi[:, 0], bi[:, 0] - ny)
    return {(int(h[i]), int(k[i])): abs_s[:, i] for i in range(len(bi))}


def _lanczos_expsqrt(matvec_fn, v, dz, m):
    r"""Compute exp(i·dz·√M) @ v via Lanczos iteration for Hermitian M.

    Uses *m* Lanczos steps (each requiring one matvec) to build an m×m
    tridiagonal approximation T of M.  The matrix function is then applied
    exactly on T via eigendecomposition.

    No explicit reorthogonalization; m ≈ 50–100 is usually sufficient for
    the moderate spectral widths encountered in electron propagation.
    """
    norm_v = jnp.linalg.norm(v)
    q1 = v / jnp.maximum(norm_v, 1e-30)
    q0 = jnp.zeros_like(q1)

    def step(carry, _):
        q_prev, q_curr, beta_prev = carry
        w = matvec_fn(q_curr)
        w = w - beta_prev * q_prev
        alpha = jnp.real(jnp.vdot(q_curr, w))
        w = w - alpha * q_curr
        beta = jnp.linalg.norm(w).real
        q_next = w / jnp.maximum(beta, 1e-30)
        return (q_curr, q_next, beta), (alpha, beta, q_curr)

    _, (alphas, betas, Q) = jax.lax.scan(
        step, (q0, q1, jnp.float64(0.0)), None, length=m
    )
    # alphas: (m,)  betas: (m,)  Q: (m, N)

    T = jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1)
    evals, evecs = jnp.linalg.eigh(T)
    sqrt_ev = jnp.sqrt(evals.astype(jnp.complex128))
    sqrt_ev = jnp.where(jnp.imag(sqrt_ev) < 0, -sqrt_ev, sqrt_ev)
    f_ev = jnp.exp(1j * dz * sqrt_ev)
    e1 = jnp.zeros(m, dtype=jnp.complex128).at[0].set(1.0)
    fT_e1 = evecs @ (f_ev * (evecs.T @ e1))

    return norm_v * (Q.T @ fT_e1)


def beam_amplitudes_fwd_direct_allbeams(
    potential,
    slice_thickness,
    energy,
    sampling,
    n_cells_array,
    gpts,
    lanczos_m=100,
):
    """KG FWD using ALL beams via FFT-based matvec + Lanczos.

    Instead of forming or decomposing the N×N structure matrix (infeasible
    for N = ny*nx = 16 384), this exploits the Toeplitz structure of M:

        [M v]_g = k₀² Σ_{g'} U_{g-g'} v_{g'} − |g⊥|² v_g

    The convolution U*v is computed via FFT in O(N log N).  The matrix
    function exp(i·dz·√M) is applied via Lanczos iteration (m steps, each
    costing one O(N log N) matvec), making the per-slice cost O(m·N log N)
    instead of O(N³) for eigendecomposition.

    Parameters
    ----------
    potential : array, shape (N_slices, ny, nx)
        Potential slices for ONE unit cell, in Volts.
    slice_thickness : float
        Thickness of each slice in Angstroms.
    energy : float
        Beam energy in eV.
    sampling : tuple of float
        (dy, dx) pixel sizes in Angstroms.
    n_cells_array : array-like of int
        Number of unit cells at which to evaluate.
    gpts : tuple of int
        (ny, nx) grid size.
    lanczos_m : int
        Number of Lanczos iterations per slice (default 100).

    Returns
    -------
    amplitudes : dict mapping (h, k) -> array of shape (len(n_cells_array),)
    beam_indices : array of shape (N_beams, 2)
    exit_state : complex array of shape (N_beams,)
        Complex Fourier-space state at the maximum requested thickness.
        Real-space exit wave:
        ``ny*nx * np.fft.ifft2(exit_state.reshape(ny, nx))``.
    """
    ny, nx = gpts
    n_cells_array = np.asarray(n_cells_array)

    # All beams — full FFT grid
    iy_all, ix_all = np.mgrid[:ny, :nx]
    beam_indices = np.stack([iy_all.ravel(), ix_all.ravel()], axis=1)
    N_beams = len(beam_indices)

    wavelength = float(energy2wavelength(energy))
    k0 = 2 * np.pi / wavelength
    k0_sq = k0 ** 2

    # Transverse k_perp² grid (ny, nx)
    dy, dx = sampling
    fy = jnp.fft.fftfreq(ny, d=dy)
    fx = jnp.fft.fftfreq(nx, d=dx)
    Fx, Fy = jnp.meshgrid(fx, fy)
    k_perp_sq_grid = (2 * jnp.pi * Fy) ** 2 + (2 * jnp.pi * Fx) ** 2

    # Pre-compute n²(r) for all slices — small (n_slices, ny, nx) arrays
    potential = jnp.asarray(potential)
    n_sq_all = jax.vmap(
        lambda V: electron_refractive_index(V, energy) ** 2
    )(potential)  # (n_slices, ny, nx)

    # JIT'd: propagate state through one unit cell (all slices) using
    # FFT matvec + Lanczos for each slice.
    @functools.partial(jax.jit, static_argnums=(4, 5, 6))
    def _propagate_one_cell(
        state_flat,
        n_sq_slices,
        k0_sq_,
        k_perp_sq_grid_,
        ny_,
        nx_,
        m_,
    ):
        def _one_slice(s, n_sq):
            def matvec(v):
                v_grid = v.reshape(ny_, nx_)
                conv = jnp.fft.fft2(n_sq * jnp.fft.ifft2(v_grid))
                return (k0_sq_ * conv - k_perp_sq_grid_ * v_grid).ravel()

            return _lanczos_expsqrt(matvec, s, slice_thickness, m_), None

        s_out, _ = jax.lax.scan(_one_slice, state_flat, n_sq_slices)
        return s_out

    # Initial state: plane wave (beam 0,0 = 1)
    state = jnp.zeros(N_beams, dtype=jnp.complex128)
    state = state.at[0].set(1.0)  # beam (0,0) is index 0 in mgrid order

    max_cells = int(n_cells_array.max()) if len(n_cells_array) > 0 else 0
    requested = set(int(n) for n in n_cells_array)

    # Collect states at requested thicknesses
    state_list = []
    cell_indices = []

    if 0 in requested:
        state_list.append(np.array(state))
        cell_indices.append(0)

    for cell in range(1, max_cells + 1):
        state = _propagate_one_cell(
            state,
            n_sq_all,
            k0_sq,
            k_perp_sq_grid,
            ny,
            nx,
            lanczos_m,
        )
        if cell in requested:
            state_list.append(np.abs(np.asarray(state)))
            cell_indices.append(cell)

    # Record for all requested cells
    if state_list:
        # First entry may be complex (initial state), rest are already abs
        abs_states = []
        for s in state_list:
            abs_states.append(np.abs(s) if np.iscomplexobj(s) else s)
        abs_all = np.stack(abs_states)  # (n_requested, N_beams)
    else:
        abs_all = np.zeros((0, N_beams))

    # Complex exit state at max thickness (for real-space exit wave)
    exit_state = np.asarray(state)  # complex, shape (N_beams,)

    # Re-order to match n_cells_array ordering
    cell_to_row = {c: r for r, c in enumerate(cell_indices)}
    ordered = np.stack([abs_all[cell_to_row[int(n)]] for n in n_cells_array])

    return _record_amplitudes_vectorized(
        ordered,
        beam_indices,
        gpts,
    ), beam_indices, exit_state


def propagation_kernel(n: int, m: int, ps: tuple[float, float], z: float, energy: float):
    """Backward-compatible alias for Fresnel propagation kernel."""
    return fresnel_propagation_kernel(n, m, ps, z, energy)


@jax.jit
def FresnelPropagator(u, H):
    """Backward-compatible alias for Fourier-space propagation."""
    return Propagator(u, H)


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
    """Build probe in reciprocal space from aberration parameters."""
    alpha = fpp.alpha
    phi = fpp.phi
    aperture = fpp.aperture

    aberrations = jnp.zeros(alpha.shape, dtype=jnp.float32)
    aberrations += ((1 / 2) * alpha**2 * pp.defocus)
    aberrations += (
        (1 / 2)
        * alpha**2
        * pp.astigmatism
        * jnp.cos(2 * (phi - pp.astigmatism_angle))
    )
    aberrations += (
        (1 / 3)
        * alpha**3
        * pp.coma
        * jnp.cos(phi - pp.coma_angle)
    )
    aberrations += (
        (1 / 3)
        * alpha**3
        * pp.trefoil
        * jnp.cos(3 * (phi - pp.trefoil_angle))
    )
    aberrations += ((1 / 4) * alpha**4 * pp.Cs)
    aberrations *= (2 * jnp.pi / fpp.wavelength)
    aberrations = jnp.cos(-aberrations) + 1.0j * jnp.sin(-aberrations)

    probe_fft = jnp.ones(alpha.shape, dtype=jnp.complex64)
    probe_fft *= aperture
    probe_fft *= aberrations
    probe_fft /= jnp.linalg.norm(probe_fft)
    return probe_fft
