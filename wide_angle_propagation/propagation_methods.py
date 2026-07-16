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
from ase import units


Sampling = tuple[float, float]

__all__ = [
    "Sampling",
    "angular_spectrum_propagation_kernel",
    "diffraction_intensity",
    "electron_refractive_index",
    "electron_refractive_index_squared",
    "electron_rest_energy",
    "energy2wavelength",
    "fourier_propagate",
    "fresnel_propagation_kernel",
    "simulate_fresnel_as",
    "simulate_fresnel_as_jit",
    "simulate_kg_ode_full",
    "simulate_wpm",
    "simulate_wpm_jit",
    "wpm_step_adaptive",
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


def electron_refractive_index_squared(potential, energy):
    """Return the Klein--Gordon electron refractive index squared.

    ``potential`` is the electrostatic potential in volts and ``energy`` is the
    incident kinetic energy in electronvolts.  For an electron, a positive
    electrostatic potential raises the local kinetic energy from ``E`` to
    ``E + potential`` in this eV/volt convention.
    """
    E0 = electron_rest_energy()
    E = energy
    denominator = E * (E + 2.0 * E0)
    linear = 2.0 * (E + E0) * potential / denominator
    quadratic = potential**2 / denominator
    return 1.0 + linear + quadratic


def electron_refractive_index(potential, energy):
    """Return the Klein--Gordon electron refractive index."""
    return jnp.sqrt(electron_refractive_index_squared(potential, energy))


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


def transverse_frequency_squared(shape: tuple[int, int], sampling: Sampling):
    """Return ``k_perp^2`` on the FFT grid in cycles per unit length."""
    fy, fx = get_frequencies(shape[0], shape[1], sampling)
    return fy**2 + fx**2


def diffraction_intensity(exit_wave):
    """Return fftshifted far-field intensity ``|FFT(exit_wave)|^2``."""
    detector_wave = jnp.fft.fftshift(jnp.fft.fft2(exit_wave))
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


def wpm_step_adaptive(
    wave,
    n_map,
    dz,
    energy,
    ps: Sampling,
    n_bins: int = 256,
    power_spacing: float = 2.0,
    bin_batch_size: int | None = None,
):
    """Run one adaptive-binned WPM propagation step."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if power_spacing <= 0:
        raise ValueError("power_spacing must be positive")
    if bin_batch_size is not None and bin_batch_size < 1:
        raise ValueError("bin_batch_size must be positive when provided")

    ny, nx = wave.shape
    wavelength = energy2wavelength(energy)
    k0 = 1 / wavelength
    k_perp2 = transverse_frequency_squared((ny, nx), ps)

    Ek = jnp.fft.fft2(wave)

    n_min, n_max = n_map.min(), n_map.max()
    n_refs = get_polynomial_bins(n_min, n_max, n_bins, power=power_spacing)

    # Find the bin indices for every pixel
    idx_R = jnp.searchsorted(n_refs, n_map)
    idx_R = jnp.clip(idx_R, 1, n_bins - 1)
    idx_L = idx_R - 1

    n_L = n_refs[idx_L]
    n_R = n_refs[idx_R]

    denom = n_R - n_L
    w_raw = (n_map - n_L) / jnp.where(denom == 0, 1.0, denom)
    w = smoothstep(w_raw)

    if bin_batch_size is None or bin_batch_size >= n_bins:
        ref_fields = wpm_propagation_kernel_vmap(Ek, n_refs, k0, k_perp2, dz)
        field_L = jnp.take_along_axis(
            ref_fields, idx_L[None, ...], axis=0
        )[0]
        field_R = jnp.take_along_axis(
            ref_fields, idx_R[None, ...], axis=0
        )[0]
        new_wave = (1 - w) * field_L + w * field_R
    else:
        # Accumulate the two selected bin fields in small batches. This is
        # algebraically identical to gathering from the full propagated bank,
        # but bounds peak memory for large transverse grids.
        n_padded = (
            (n_bins + bin_batch_size - 1) // bin_batch_size
        ) * bin_batch_size
        n_refs_padded = jnp.pad(
            n_refs,
            (0, n_padded - n_bins),
            mode="edge",
        )
        reference_batches = n_refs_padded.reshape(-1, bin_batch_size)
        index_batches = jnp.arange(n_padded).reshape(-1, bin_batch_size)

        def accumulate_batch(accumulated_wave, batch):
            reference_batch, index_batch = batch
            batch_fields = wpm_propagation_kernel_vmap(
                Ek, reference_batch, k0, k_perp2, dz
            )
            batch_indices = index_batch[:, None, None]
            left_mask = batch_indices == idx_L[None, ...]
            right_mask = batch_indices == idx_R[None, ...]
            batch_weights = (
                left_mask * (1.0 - w)[None, ...]
                + right_mask * w[None, ...]
            )
            accumulated_wave = accumulated_wave + jnp.sum(
                batch_weights * batch_fields,
                axis=0,
            )
            return accumulated_wave, None

        accumulator_dtype = (
            jnp.complex128 if jax.config.x64_enabled else jnp.complex64
        )
        new_wave, _ = jax.lax.scan(
            accumulate_batch,
            jnp.zeros(wave.shape, dtype=accumulator_dtype),
            (reference_batches, index_batches),
        )

    return new_wave, w, idx_L, n_refs


def _slice_phase_grating(potential_slice, slice_thickness, energy):
    """Return the paraxial KG transmission function for one potential slice.

    Applying the slowly varying-envelope approximation to

        nabla^2 psi + (2*pi*k0)^2 n^2 psi = 0

    gives the interaction phase ``pi*k0*(n^2 - 1)*dz``.  This retains the
    quadratic-potential term in the Klein--Gordon refractive index.  Linearising
    ``n^2`` in the electrostatic potential recovers the conventional
    ``exp(i*sigma*V_projected)`` transmission function.
    """
    wavelength = energy2wavelength(energy)
    n_squared = electron_refractive_index_squared(potential_slice, energy)
    phase = jnp.pi * (n_squared - 1.0) * slice_thickness / wavelength
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
    bin_batch_size: int | None = None,
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
            bin_batch_size=bin_batch_size,
        )
        wavefronts.append(wavefront)

    exit_wave = wavefront
    diffraction_pattern = diffraction_intensity(exit_wave)
    return exit_wave, diffraction_pattern, _stack_wavefronts_or_empty(wavefronts, probe)


simulate_wpm_jit = jax.jit(
    simulate_wpm,
    static_argnames=("n_bins", "power_spacing", "bin_batch_size"),
)


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
        k0**2 * electron_refractive_index_squared(potential_slice, energy)
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
