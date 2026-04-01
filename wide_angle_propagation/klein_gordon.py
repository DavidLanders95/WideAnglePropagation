"""Lanczos-based forward Klein-Gordon propagation in beam space."""
import functools

import numpy as np
import jax
import jax.numpy as jnp

from .wpm import (
    electron_refractive_index,
    energy2wavelength,
)


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


def beam_amplitudes_fwd_direct_allbeams(potential, slice_thickness, energy,
                                        sampling, n_cells_array, gpts,
                                        lanczos_m=100):
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
        Real-space exit wave: ``ny*nx * np.fft.ifft2(exit_state.reshape(ny, nx))``.
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
    def _propagate_one_cell(state_flat, n_sq_slices, k0_sq_, k_perp_sq_grid_,
                            ny_, nx_, m_):
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
            state, n_sq_all, k0_sq, k_perp_sq_grid, ny, nx, lanczos_m
        )
        if cell in requested:
            state_list.append(np.abs(np.asarray(state)))
            cell_indices.append(cell)

    # Record for all requested cells
    if state_list:
        # First entry may be complex (initial state), rest are already abs
        abs_states = []
        for i, s in enumerate(state_list):
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
        ordered, beam_indices, gpts
    ), beam_indices, exit_state
