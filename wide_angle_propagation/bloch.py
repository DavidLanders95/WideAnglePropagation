"""Bloch wave / scattering-matrix eigendecomposition for periodic crystals.

For a crystal with period c along z, the one-cell transfer matrix S maps
the state [psi_g, phi_g] at z=0 to z=c (where phi = dpsi/dz).

Eigendecomposing S gives exact beam amplitudes at any integer number of
unit cells:  psi_g(N) = sum_j  c_j * lambda_j^N * v_j^(g).

The transfer matrix is built by composing per-slice transfer matrices
obtained from the beam-basis matrix exponential of the Klein-Gordon
structure matrix.
"""
import functools

import numpy as np
import jax
import jax.numpy as jnp

from .wpm import (
    electron_refractive_index,
    energy2wavelength,
    _build_structure_matrix,
    _build_transfer_matrix_slice,
    _build_ms_slice_transfer,
)


def select_beams(gpts, sampling, max_angle_mrad=None, max_beams=None, energy=None):
    """Select a subset of beams by angle cutoff or count, sorted by |k_perp|.

    Parameters
    ----------
    gpts : tuple of int
        (ny, nx) grid size.
    sampling : tuple of float
        (dy, dx) pixel sizes in Angstroms.
    max_angle_mrad : float, optional
        Maximum scattering angle in mrad. Requires energy.
    max_beams : int, optional
        Maximum number of beams to keep (sorted by |k_perp|).
    energy : float, optional
        Beam energy in eV (needed for angle calculation).

    Returns
    -------
    beam_indices : array of shape (N_beams, 2)
        (iy, ix) indices into the FFT grid.
    k_perp_sq : array of shape (N_beams,)
        |k_perp|^2 for each beam.
    """
    ny, nx = gpts
    dy, dx = sampling
    fy = np.fft.fftfreq(ny, d=dy)
    fx = np.fft.fftfreq(nx, d=dx)
    Fx, Fy = np.meshgrid(fx, fy)

    freq_sq = Fx**2 + Fy**2

    # Build all (iy, ix) pairs
    iy_all, ix_all = np.mgrid[:ny, :nx]
    indices = np.stack([iy_all.ravel(), ix_all.ravel()], axis=1)
    f_sq_flat = freq_sq.ravel()

    # Sort by frequency magnitude
    order = np.argsort(f_sq_flat)
    indices = indices[order]
    f_sq_flat = f_sq_flat[order]

    # Apply angle cutoff
    if max_angle_mrad is not None and energy is not None:
        wavelength = float(energy2wavelength(energy))
        max_freq = max_angle_mrad / (wavelength * 1000.0)
        mask = f_sq_flat <= max_freq**2
        indices = indices[mask]
        f_sq_flat = f_sq_flat[mask]

    # Apply beam count limit
    if max_beams is not None and len(indices) > max_beams:
        indices = indices[:max_beams]
        f_sq_flat = f_sq_flat[:max_beams]

    k_perp_sq = (2 * np.pi)**2 * f_sq_flat
    return indices, k_perp_sq


# ---------------------------------------------------------------------------
# JIT'd helpers: batched propagator construction, scan-based propagation,
# FFT-based matvec, Lanczos for matrix-function-vector products
# ---------------------------------------------------------------------------

def _prepare_beam_data(beam_indices, sampling, gpts):
    """Pre-compute reusable beam-grid quantities for structure matrices."""
    ny, nx = gpts
    dy, dx = sampling
    beam_indices = np.asarray(beam_indices)
    fy = jnp.fft.fftfreq(ny, d=dy)
    fx = jnp.fft.fftfreq(nx, d=dx)
    beam_iy = jnp.asarray(beam_indices[:, 0])
    beam_ix = jnp.asarray(beam_indices[:, 1])
    diy = (beam_iy[:, None] - beam_iy[None, :]) % ny
    dix = (beam_ix[:, None] - beam_ix[None, :]) % nx
    kp2 = (2 * jnp.pi * fy[beam_iy]) ** 2 + (2 * jnp.pi * fx[beam_ix]) ** 2
    return beam_iy, beam_ix, diy, dix, kp2


@jax.jit
def _compose_fwd_propagator_jit(U_all, k0_sq, diy, dix, k_perp_sq, dz):
    """Compose one-cell forward operator S = P_{n-1}…P_0 via lax.scan.

    Memory: O(N²) – only the running product S is kept, not all N_slice
    intermediate propagator matrices.
    """
    N = k_perp_sq.shape[0]
    S0 = jnp.eye(N, dtype=jnp.complex128)

    def body(S, U_slice):
        M = k0_sq * U_slice[diy, dix] - jnp.diag(k_perp_sq)
        M_herm = (M + M.conj().T) / 2
        evals, V = jnp.linalg.eigh(M_herm)
        sqrt_e = jnp.sqrt(evals.astype(jnp.complex128))
        sqrt_e = jnp.where(jnp.imag(sqrt_e) < 0, -sqrt_e, sqrt_e)
        D = jnp.exp(1j * dz * sqrt_e)
        return (V * D[None, :]) @ (V.conj().T @ S), None

    S, _ = jax.lax.scan(body, S0, U_all)
    return S


@functools.partial(jax.jit, static_argnums=(2,))
def _propagate_direct_jit(S, state0, max_cells):
    """Apply one-cell operator S repeatedly via lax.scan."""
    def step(state, _):
        s = S @ state
        return s, s
    _, all_states = jax.lax.scan(step, state0, None, length=max_cells)
    return all_states  # (max_cells, N_beams)


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


def build_scattering_matrix(potential, slice_thickness, energy, sampling,
                            beam_indices=None, max_beams=200,
                            max_angle_mrad=None, **kwargs):
    """Build the one-unit-cell transfer matrix by composing slice transfer matrices.

    For each slice with piecewise-constant n²(x,y), the 2nd-order KG equation
    in the beam basis has the exact transfer matrix T_i = expm(dz * A_i).
    The full one-cell transfer matrix is S = T_n * ... * T_1.

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
    beam_indices : array of shape (N_beams, 2), optional
        Pre-selected (iy, ix) beam indices. If None, auto-selected.
    max_beams : int
        Maximum beams if auto-selecting.
    max_angle_mrad : float, optional
        Angle cutoff for beam selection.

    Returns
    -------
    S : array, shape (2*N_beams, 2*N_beams)
        Transfer matrix mapping [psi_beams, phi_beams] through one unit cell.
    beam_indices : array of shape (N_beams, 2)
        The beam indices used.
    """
    ny, nx = potential.shape[1], potential.shape[2]
    gpts = (ny, nx)

    if beam_indices is None:
        beam_indices, _ = select_beams(
            gpts, sampling,
            max_angle_mrad=max_angle_mrad,
            max_beams=max_beams,
            energy=energy,
        )

    beam_indices = np.asarray(beam_indices)
    N_beams = len(beam_indices)

    wavelength = float(energy2wavelength(energy))
    k0 = 2 * np.pi / wavelength

    # Start with identity
    S = jnp.eye(2 * N_beams, dtype=jnp.complex128)

    for i in range(potential.shape[0]):
        n_sq = electron_refractive_index(potential[i], energy) ** 2
        M = _build_structure_matrix(n_sq, k0, beam_indices, sampling, gpts)
        T = _build_transfer_matrix_slice(M, slice_thickness, N_beams)
        S = T @ S

    return S, beam_indices


def beam_amplitudes_vs_thickness(S, beam_indices, n_cells_array, energy,
                                 gpts, sampling):
    """Compute beam amplitudes at multiple thicknesses via eigendecomposition.

    Parameters
    ----------
    S : array, shape (2*N_beams, 2*N_beams)
        Transfer matrix from build_scattering_matrix.
    beam_indices : array of shape (N_beams, 2)
        (iy, ix) beam indices.
    n_cells_array : array-like of int
        Number of unit cells at which to evaluate.
    energy : float
        Beam energy in eV.
    gpts : tuple of int
        (ny, nx) grid size.
    sampling : tuple of float
        (dy, dx) pixel sizes.

    Returns
    -------
    amplitudes : dict mapping (h, k) -> array of shape (len(n_cells_array),)
        Normalized beam amplitudes |C_g| at each thickness.
    """
    n_cells_array = np.asarray(n_cells_array)
    N_beams = len(beam_indices)
    ny, nx = gpts

    # Eigendecomposition (GPU via JAX)
    S_jax = jnp.asarray(S)
    eigenvalues, eigenvectors = jnp.linalg.eig(S_jax)

    # Initial condition: plane wave psi = 1/(ny*nx), phi = i*k0 * psi
    wavelength = float(energy2wavelength(energy))
    k0 = 2 * np.pi / wavelength

    # In Fourier space the plane wave has amplitude 1 at beam (0,0) and 0 elsewhere
    # psi_g(0) = delta_{g,0} (in normalized DFT coefficients)
    # phi_g(0) = i*k0 * delta_{g,0}
    initial_state = jnp.zeros(2 * N_beams, dtype=jnp.complex128)

    # Find the (0,0) beam index
    for b, (iy, ix) in enumerate(beam_indices):
        if iy == 0 and ix == 0:
            initial_state = initial_state.at[b].set(1.0)          # psi coefficient
            initial_state = initial_state.at[N_beams + b].set(1j * k0)  # phi coefficient
            break

    # Decompose initial state in eigenbasis: c = V^{-1} * initial_state
    coeffs = jnp.linalg.solve(eigenvectors, initial_state)

    # For each thickness N: state(N) = V @ diag(lambda^N) @ c
    amplitudes = {}
    center_y, center_x = ny // 2, nx // 2

    for n_cells in n_cells_array:
        if n_cells == 0:
            state = initial_state
        else:
            lambda_n = eigenvalues ** n_cells
            state = eigenvectors @ (coeffs * lambda_n)

        # Extract psi coefficients (first N_beams entries)
        psi_coeffs = state[:N_beams]

        for b, (iy, ix) in enumerate(beam_indices):
            # Convert FFT index to (h, k) beam label
            h = ix if ix <= nx // 2 else ix - nx
            k = iy if iy <= ny // 2 else iy - ny
            key = (h, k)
            if key not in amplitudes:
                amplitudes[key] = []
            amplitudes[key].append(float(jnp.abs(psi_coeffs[b])))

    return {k: np.array(v) for k, v in amplitudes.items()}


def build_scattering_matrix_fwd(potential, slice_thickness, energy, sampling,
                                beam_indices=None, max_beams=200,
                                max_angle_mrad=None, **kwargs):
    """Build forward-only one-unit-cell transfer matrix (N×N).

    Uses the forward-only KG propagator for each slice, composing them
    into a single N×N matrix (instead of 2N×2N for the full KG).

    Parameters are the same as build_scattering_matrix.

    Returns
    -------
    S_fwd : array, shape (N_beams, N_beams)
        Forward-only transfer matrix for one unit cell.
    beam_indices : array of shape (N_beams, 2)
        The beam indices used.
    """
    ny, nx = potential.shape[1], potential.shape[2]
    gpts = (ny, nx)

    if beam_indices is None:
        beam_indices, _ = select_beams(
            gpts, sampling,
            max_angle_mrad=max_angle_mrad,
            max_beams=max_beams,
            energy=energy,
        )

    beam_indices = np.asarray(beam_indices)
    N_beams = len(beam_indices)

    wavelength = float(energy2wavelength(energy))
    k0 = 2 * np.pi / wavelength

    # Pre-compute beam data and Fourier coefficients
    _, _, diy, dix, kp2 = _prepare_beam_data(beam_indices, sampling, gpts)
    potential = jnp.asarray(potential)
    n_sq_all = jax.vmap(
        lambda V: electron_refractive_index(V, energy) ** 2
    )(potential)
    U_all = jnp.fft.fft2(n_sq_all) / (ny * nx)

    # Compose S via JIT'd lax.scan
    S = _compose_fwd_propagator_jit(U_all, k0 ** 2, diy, dix, kp2, slice_thickness)

    return S, beam_indices


def beam_amplitudes_vs_thickness_fwd(S_fwd, beam_indices, n_cells_array,
                                     gpts):
    """Compute beam amplitudes at multiple thicknesses via forward-only eigendecomposition.

    Parameters
    ----------
    S_fwd : array, shape (N_beams, N_beams)
        Forward-only transfer matrix from build_scattering_matrix_fwd.
    beam_indices : array of shape (N_beams, 2)
        (iy, ix) beam indices.
    n_cells_array : array-like of int
        Number of unit cells at which to evaluate.
    gpts : tuple of int
        (ny, nx) grid size.

    Returns
    -------
    amplitudes : dict mapping (h, k) -> array of shape (len(n_cells_array),)
        Normalized beam amplitudes |C_g| at each thickness.
    """
    n_cells_array = np.asarray(n_cells_array)
    N_beams = len(beam_indices)
    ny, nx = gpts

    # Eigendecomposition (GPU via JAX)
    S_jax = jnp.asarray(S_fwd)
    eigenvalues, eigenvectors = jnp.linalg.eig(S_jax)

    # Initial condition: plane wave c_g(0) = delta_{g,0}
    initial_state = jnp.zeros(N_beams, dtype=jnp.complex128)
    for b, (iy, ix) in enumerate(beam_indices):
        if iy == 0 and ix == 0:
            initial_state = initial_state.at[b].set(1.0)
            break

    # Decompose: coeffs = V^{-1} @ c_0
    coeffs = jnp.linalg.solve(eigenvectors, initial_state)

    amplitudes = {}

    for n_cells in n_cells_array:
        if n_cells == 0:
            state = initial_state
        else:
            lambda_n = eigenvalues ** n_cells
            state = eigenvectors @ (coeffs * lambda_n)

        for b, (iy, ix) in enumerate(beam_indices):
            h = ix if ix <= nx // 2 else ix - nx
            k = iy if iy <= ny // 2 else iy - ny
            key = (h, k)
            if key not in amplitudes:
                amplitudes[key] = []
            amplitudes[key].append(float(jnp.abs(state[b])))

    return {k: np.array(v) for k, v in amplitudes.items()}


def beam_amplitudes_direct(potential, slice_thickness, energy, sampling,
                           n_cells_array, gpts, beam_indices=None,
                           max_beams=200, max_angle_mrad=None):
    """Full KG beam amplitudes via direct propagation (no eigendecomposition).

    Applies the per-cell transfer matrix S = T_{n} ... T_1 to the state
    vector [psi, phi] iteratively for each unit cell.  This avoids the
    numerically fragile eigendecomposition of the large 2N×2N scattering
    matrix and is more robust for large beam counts (>200).

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
    beam_indices : array, optional
        Pre-selected beam indices. Auto-selected if None.
    max_beams : int
        Maximum beams if auto-selecting.
    max_angle_mrad : float, optional
        Angle cutoff.

    Returns
    -------
    amplitudes : dict mapping (h, k) -> array of shape (len(n_cells_array),)
    beam_indices : array of shape (N_beams, 2)
    """
    ny, nx = gpts
    n_cells_array = np.asarray(n_cells_array)

    if beam_indices is None:
        beam_indices, _ = select_beams(
            gpts, sampling, max_angle_mrad=max_angle_mrad,
            max_beams=max_beams, energy=energy,
        )
    beam_indices = np.asarray(beam_indices)
    N_beams = len(beam_indices)

    wavelength = float(energy2wavelength(energy))
    k0 = 2 * np.pi / wavelength

    # Build per-slice transfer matrices (only need to do this once)
    slice_transfers = []
    for i in range(potential.shape[0]):
        n_sq = electron_refractive_index(potential[i], energy) ** 2
        M = _build_structure_matrix(n_sq, k0, beam_indices, sampling, gpts)
        T = _build_transfer_matrix_slice(M, slice_thickness, N_beams)
        slice_transfers.append(T)

    # Initial state: plane wave psi_g(0) = delta_{g,0}, phi_g(0) = i*k0*delta_{g,0}
    state = jnp.zeros(2 * N_beams, dtype=jnp.complex128)
    for b, (iy, ix) in enumerate(beam_indices):
        if iy == 0 and ix == 0:
            state = state.at[b].set(1.0)
            state = state.at[N_beams + b].set(1j * k0)
            break

    # Propagate cell by cell, recording amplitudes at requested thicknesses
    max_cells = int(n_cells_array.max()) if len(n_cells_array) > 0 else 0
    requested = set(int(n) for n in n_cells_array)
    amplitudes = {}

    def _record(state_vec):
        psi_coeffs = state_vec[:N_beams]
        for b_idx, (iy, ix) in enumerate(beam_indices):
            h = ix if ix <= nx // 2 else ix - nx
            k = iy if iy <= ny // 2 else iy - ny
            key = (h, k)
            if key not in amplitudes:
                amplitudes[key] = []
            amplitudes[key].append(float(jnp.abs(psi_coeffs[b_idx])))

    if 0 in requested:
        _record(state)

    for cell in range(1, max_cells + 1):
        for T in slice_transfers:
            state = T @ state
        if cell in requested:
            _record(state)

    return {k: np.array(v) for k, v in amplitudes.items()}, beam_indices


def beam_amplitudes_fwd_direct(potential, slice_thickness, energy, sampling,
                               n_cells_array, gpts, beam_indices=None,
                               max_beams=200, max_angle_mrad=None):
    """KG FWD beam amplitudes via direct propagation (no eigendecomposition).

    Applies the per-cell forward propagator P = P_{n} ... P_1 to the
    beam coefficient vector c iteratively.  More robust than the
    eigendecomposition-based approach for large beam counts.

    Parameters are the same as beam_amplitudes_direct.

    Returns
    -------
    amplitudes : dict mapping (h, k) -> array of shape (len(n_cells_array),)
    beam_indices : array of shape (N_beams, 2)
    """
    ny, nx = gpts
    n_cells_array = np.asarray(n_cells_array)

    if beam_indices is None:
        beam_indices, _ = select_beams(
            gpts, sampling, max_angle_mrad=max_angle_mrad,
            max_beams=max_beams, energy=energy,
        )
    beam_indices = np.asarray(beam_indices)
    N_beams = len(beam_indices)

    wavelength = float(energy2wavelength(energy))
    k0 = 2 * np.pi / wavelength

    # Pre-compute beam data and Fourier coefficients
    _, _, diy, dix, kp2 = _prepare_beam_data(beam_indices, sampling, gpts)
    potential = jnp.asarray(potential)
    n_sq_all = jax.vmap(
        lambda V: electron_refractive_index(V, energy) ** 2
    )(potential)
    U_all = jnp.fft.fft2(n_sq_all) / (ny * nx)

    # Compose one-cell forward operator via JIT'd lax.scan
    S = _compose_fwd_propagator_jit(U_all, k0 ** 2, diy, dix, kp2, slice_thickness)

    # Initial state: plane wave c_g(0) = delta_{g,0}
    state = jnp.zeros(N_beams, dtype=jnp.complex128)
    for b, (iy, ix) in enumerate(beam_indices):
        if iy == 0 and ix == 0:
            state = state.at[b].set(1.0)
            break

    max_cells = int(n_cells_array.max()) if len(n_cells_array) > 0 else 0

    # JIT'd cell propagation via lax.scan
    if max_cells > 0:
        all_states = _propagate_direct_jit(S, state, max_cells)
        # all_states[i] = state after (i+1) cells
        full_states = jnp.concatenate([state[None, :], all_states], axis=0)
    else:
        full_states = state[None, :]

    # Select requested thicknesses and record
    idx = np.array([int(n) for n in n_cells_array])
    selected = full_states[idx]
    return _record_amplitudes_vectorized(selected, beam_indices, gpts), beam_indices


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


def beam_amplitudes_ms_direct(potential, slice_thickness, energy, sampling,
                              n_cells_array, gpts, beam_indices=None,
                              max_beams=200, max_angle_mrad=None,
                              propagation='fresnel'):
    """Beam amplitudes via multislice in beam basis (split-operator).

    This applies the same physics as the real-space multislice but in the
    beam (Fourier) basis.  Each slice uses a transmission matrix (from the
    phase grating) and a diagonal propagation matrix, composed as P @ T.
    The state vector c is propagated directly through each cell.

    This method gives results identical to the real-space Fresnel or
    Angular-Spectrum multislice (up to beam truncation) because it uses
    the same split-operator convention.

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
    beam_indices : array, optional
        Pre-selected beam indices. Auto-selected if None.
    max_beams : int
        Maximum beams if auto-selecting.
    max_angle_mrad : float, optional
        Angle cutoff.
    propagation : str
        'fresnel' (default) or 'as' (angular spectrum).

    Returns
    -------
    amplitudes : dict mapping (h, k) -> array of shape (len(n_cells_array),)
    beam_indices : array of shape (N_beams, 2)
    """
    ny, nx = gpts
    n_cells_array = np.asarray(n_cells_array)

    if beam_indices is None:
        beam_indices, _ = select_beams(
            gpts, sampling, max_angle_mrad=max_angle_mrad,
            max_beams=max_beams, energy=energy,
        )
    beam_indices = np.asarray(beam_indices)
    N_beams = len(beam_indices)

    wavelength = float(energy2wavelength(energy))
    k0 = 2 * np.pi / wavelength

    # Build per-slice MS transfer matrices (only done once)
    slice_transfers = []
    for i in range(potential.shape[0]):
        n_slice = electron_refractive_index(potential[i], energy)
        S = _build_ms_slice_transfer(
            n_slice, k0, slice_thickness, beam_indices, sampling, gpts,
            propagation=propagation,
        )
        slice_transfers.append(S)

    # Initial state: plane wave c_g(0) = delta_{g,0}
    state = jnp.zeros(N_beams, dtype=jnp.complex128)
    for b, (iy, ix) in enumerate(beam_indices):
        if iy == 0 and ix == 0:
            state = state.at[b].set(1.0)
            break

    max_cells = int(n_cells_array.max()) if len(n_cells_array) > 0 else 0
    requested = set(int(n) for n in n_cells_array)
    amplitudes = {}

    def _record(state_vec):
        for b_idx, (iy, ix) in enumerate(beam_indices):
            h = ix if ix <= nx // 2 else ix - nx
            k = iy if iy <= ny // 2 else iy - ny
            key = (h, k)
            if key not in amplitudes:
                amplitudes[key] = []
            amplitudes[key].append(float(jnp.abs(state_vec[b_idx])))

    if 0 in requested:
        _record(state)

    for cell in range(1, max_cells + 1):
        for S in slice_transfers:
            state = S @ state
        if cell in requested:
            _record(state)

    return {k: np.array(v) for k, v in amplitudes.items()}, beam_indices
