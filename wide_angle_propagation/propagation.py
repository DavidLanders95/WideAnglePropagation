import numpy as np
import jax
import jax.numpy as jnp
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


def fresnel_propagation_kernel(n: int, m: int, ps: float, z: float, energy: float):
    wavelength = energy2wavelength(energy)
    Fx, Fy = get_frequencies(n, m, ps)

    H = jnp.exp(1j * (2 * jnp.pi / wavelength) * z) * jnp.exp(
        -1j * jnp.pi * wavelength * z * (Fx**2 + Fy**2))
    return H


def angular_spectrum_propagation_kernel(
    n: int, m: int, ps: float, z: float, energy: float
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
    # kz = sqrt((n * k0)^2 - k_perp^2)
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
# 6b. Lippmann-Schwinger 2D Solver (Integral equation approach)
# =============================================================================

def _greens_2d_hankel(k0, r):
    """2D Hankel Green's function (isotropic in space)."""
    from scipy.special import hankel1
    return 0.25j * hankel1(0, k0 * r)


def _self_weight_cell_average(k0, hx, hy, oversamp=32):
    """Self-weight for grid point accounting for anisotropic spacing."""
    xs = (np.arange(oversamp) + 0.5) / oversamp * hx - 0.5 * hx
    ys = (np.arange(oversamp) + 0.5) / oversamp * hy - 0.5 * hy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    R = np.sqrt(X**2 + Y**2)
    G = _greens_2d_hankel(k0, R)
    return G.mean() * (hx * hy)


def build_lippmann_schwinger_kernel_2d(k0, hx, hy, Nx, Ny, pad_to_double=True, oversamp_self=32):
    """
    Build 2D Lippmann-Schwinger kernel with ANISOTROPIC spacing (hx != hy allowed).
    
    Args:
        k0: wave number
        hx: grid spacing in x-direction (Angstroms)
        hy: grid spacing in y-direction (Angstroms)
        Nx, Ny: number of grid points
        pad_to_double: if True, pad to 2x size for circular convolution
        oversamp_self: oversampling for self-weight calculation
    
    Returns:
        Khat: FFT of kernel (complex128 JAX array)
        embed_shape: (Ex, Ey) padded shape
    """
    Ex, Ey = (2*Nx, 2*Ny) if pad_to_double else (Nx, Ny)

    # Construct coordinates with anisotropic spacing
    x = (np.arange(Nx) - Nx//2) * hx
    y = (np.arange(Ny) - Ny//2) * hy
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(X**2 + Y**2)

    # Green's function kernel (isotropic in space, but anisotropic in sampling)
    K = (hx * hy) * _greens_2d_hankel(k0, R)
    K[Nx//2, Ny//2] = _self_weight_cell_average(k0, hx, hy, oversamp=oversamp_self)

    # Embed into padded domain
    Kemb = np.zeros((Ex, Ey), dtype=np.complex128)
    sx, sy = (Ex-Nx)//2, (Ey-Ny)//2
    Kemb[sx:sx+Nx, sy:sy+Ny] = K

    Kemb = np.fft.ifftshift(Kemb)
    Khat = np.fft.fftn(Kemb)

    return jnp.asarray(Khat), (Ex, Ey)


def apply_convolution(source, Khat, embed_shape, roi_shape):
    """Apply FFT-based convolution with Green's function."""
    Ex, Ey = embed_shape
    Nx, Ny = roi_shape
    sx, sy = (Ex-Nx)//2, (Ey-Ny)//2

    src = jnp.zeros((Ex, Ey), dtype=jnp.complex128)
    src = src.at[sx:sx+Nx, sy:sy+Ny].set(source)

    conv = jnp.fft.ifftn(Khat * jnp.fft.fftn(src))
    return conv[sx:sx+Nx, sy:sy+Ny]


def solve_scattering_2d_lippmann_schwinger(potential_grid, energy_ev, pixel_size_A, 
                                           pad_factor=2, absorb_frac=0.15, absorb_strength=4.0,
                                           gmres_tol=1e-4, gmres_restart=70):
    """
    Solves 2D scalar Helmholtz for a pixelized potential using the Lippmann-Schwinger 
    integral equation with JAX/FFT and GMRES.
    
    Args:
        potential_grid: 2D numpy array of the electrostatic potential (in Volts), shape (Nx, Ny)
        energy_ev: Energy of incident electron/particle in eV
        pixel_size_A: Size of one pixel in Angstroms
        pad_factor: How much to pad the grid to avoid periodic artifacts (2 = double size)
        absorb_frac: Fraction of grid on each side used for absorbing boundary
        absorb_strength: Strength of absorbing taper (larger = stronger damping)
        gmres_tol: GMRES convergence tolerance
        gmres_restart: GMRES restart parameter
    
    Returns:
        wave_final: 2D array of the total wave function (complex)
    """
    from ase import units
    
    # --- 1. Physics Constants & Setup ---
    m0 = 510998.95000  # Rest mass energy in eV
    hc = 12398.4193    # eV * Angstrom

    hbar = units._hplanck / (2*np.pi)
    m_ec2_ev = (units._me * units._c**2) / units._e

    gamma = 1.0 + energy_ev / m_ec2_ev
    m_eff = gamma * units._me

    # Converts V [V] into the Helmholtz source term (in 1/Å^2)
    V_scale = -(2.0 * m_eff * units._e / (hbar**2)) * 1e-20

    E = energy_ev
    wavelength = hc / jnp.sqrt(E * (E + 2 * m0))
    k0 = 2 * jnp.pi / wavelength
    
    # --- 2. Padding ---
    orig_shape = potential_grid.shape
    padded_shape = [s * pad_factor for s in orig_shape]
    pad_width = [( (p-o)//2, (p-o)-(p-o)//2 ) for p, o in zip(padded_shape, orig_shape)]
    V_padded = jnp.pad(jnp.array(potential_grid), pad_width, mode='constant')
    
    # --- 3. Build anisotropic Green's function kernel ---
    Khat, embed_shape = build_lippmann_schwinger_kernel_2d(
        k0=float(k0),
        hx=float(pixel_size_A),
        hy=float(pixel_size_A),
        Nx=padded_shape[0],
        Ny=padded_shape[1],
        pad_to_double=True,
        oversamp_self=32
    )

    # --- 4. matvec with convolution ---
    @jax.jit
    def matvec(psi_flat):
        psi = psi_flat.reshape(padded_shape)
        source = V_scale * V_padded * psi
        scattered = apply_convolution(source, Khat, embed_shape, padded_shape)
        return (psi - scattered).ravel()

    # --- 5. Incident Wave (plane wave) ---
    y_coords = jnp.arange(padded_shape[1]) * pixel_size_A
    y_coords -= y_coords.mean()
    psi_inc_line = jnp.exp(1j * k0 * y_coords)
    psi_inc = jnp.broadcast_to(psi_inc_line, padded_shape)

    # --- 6. Solve with GMRES ---
    from jax.scipy.sparse.linalg import gmres
    
    psi_solution_flat, info = gmres(
        matvec,
        psi_inc.ravel(),
        tol=gmres_tol,
        restart=gmres_restart
    )
    if info != 0:
        print(f"  Warning: GMRES did not converge (info={info})")
    psi_solution = psi_solution_flat.reshape(padded_shape)

    # --- 7. Unpad ---
    starts = [p[0] for p in pad_width]
    ends = [s + o for s, o in zip(starts, orig_shape)]
    psi_final = psi_solution[starts[0]:ends[0], starts[1]:ends[1]]
    
    return psi_final


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
