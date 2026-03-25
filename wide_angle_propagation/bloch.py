"""
Bloch wave dynamical diffraction solver for electron microscopy simulations.

Implements the Bloch wave method for computing diffracted beam amplitudes
as a function of crystal thickness, using the Lobato parametrization for
electron scattering factors.

Reference: Rother & Scheerschmidt (2009), Ultramicroscopy 109, 154-160.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
_M0   = 9.1093837015e-31    # kg, electron rest mass
_E    = 1.602176634e-19     # C, elementary charge
_H    = 6.62607015e-34      # J·s, Planck constant
_C    = 2.99792458e8        # m/s, speed of light
_EPS0 = 8.8541878128e-12    # F/m, vacuum permittivity

# ---------------------------------------------------------------------------
# Hardcoded Lobato scattering-factor parameters for Au (Z=79)
# Lobato & Van Dyck (2014), Acta Cryst. A70, 490-505, Table 1
# Form: f(s²) = Σ a_i (2 + b_i s²) / (1 + b_i s²)²  with  s = |g|/2  [Å⁻¹]
# When using k² = |g|² directly, set b_k = b_s / 4.
# ---------------------------------------------------------------------------
_LOBATO_AU_A   = np.array([4.9655, 8.6537, 9.1870, 3.8743, 1.3787], dtype=np.float64)
_LOBATO_AU_B_S = np.array([2.3678, 5.1001, 12.9153, 0.3636, 29.8205], dtype=np.float64)


# ---------------------------------------------------------------------------
# Energy / wavelength helpers
# ---------------------------------------------------------------------------

def _energy2wavelength(energy_eV: float) -> float:
    """Relativistic de Broglie wavelength [Å] for an electron at energy_eV."""
    E_kin = energy_eV * _E
    p = np.sqrt(2.0 * _M0 * E_kin * (1.0 + E_kin / (2.0 * _M0 * _C ** 2)))
    return (_H / p) * 1e10          # m → Å


def _energy2sigma(energy_eV: float) -> float:
    """Relativistic interaction parameter σ [rad/(V·Å)]."""
    E_kin = energy_eV * _E
    lam   = _energy2wavelength(energy_eV) * 1e-10  # Å → m
    m_rel = _M0 * (1.0 + E_kin / (_M0 * _C ** 2))
    sigma = (2.0 * np.pi * m_rel * _E * lam) / (_H ** 2)
    return sigma * 1e-10            # rad/(V·m) → rad/(V·Å)


def _kappa() -> float:
    """κ = 4πε₀/e  [1/(V·Å)]."""
    return 4.0 * np.pi * _EPS0 / _E * 1e-10


def wavelength_to_energy_eV(wavelength_ang: float) -> float:
    """Convert electron wavelength [Å] to kinetic energy [eV].

    Solves the relativistic quadratic  p²/(2m₀) · (1 + p²/(2m₀·(m₀c²))) = T.
    """
    lam  = wavelength_ang * 1e-10           # Å → m
    m0c2 = _M0 * _C ** 2                    # J
    p    = _H / lam                         # kg·m/s
    # T² + 2·m₀c²·T - (pc)² = 0  →  T = m₀c²(sqrt(1+(p/(m₀c))²) - 1)
    T_J = m0c2 * (np.sqrt(1.0 + (p / (_M0 * _C)) ** 2) - 1.0)
    return T_J / _E                         # J → eV


# ---------------------------------------------------------------------------
# Lobato scattering factor (vectorised)
# ---------------------------------------------------------------------------

def _lobato_scattering_factor(k2: np.ndarray,
                               a: np.ndarray,
                               b_k: np.ndarray) -> np.ndarray:
    """Evaluate the Lobato electron scattering factor.

    Parameters
    ----------
    k2  : (...) array, |g|² [Å⁻²]
    a   : (5,)  Lobato a-parameters [Å]
    b_k : (5,)  b-parameters pre-scaled for k² convention (= b_paper / 4)

    Returns
    -------
    f : (...) scattering factor [Å]
    """
    k2e = k2[..., np.newaxis]       # (..., 1)
    return np.sum(a * (2.0 + b_k * k2e) / (1.0 + b_k * k2e) ** 2, axis=-1)


# ---------------------------------------------------------------------------
# Geometric (atomistic) structure factor
# ---------------------------------------------------------------------------

def _structure_factor(hkl: np.ndarray, frac_coords: np.ndarray) -> complex:
    """Compute the geometric structure factor F(h,k,l).

    .. math::
        F(hkl) = \\sum_{\\text{atoms}} \\exp\\!\\bigl(2\\pi i\\,(hx + ky + lz)\\bigr)

    Parameters
    ----------
    hkl        : (3,) array-like, Miller indices [h, k, l]
    frac_coords: (N_atoms, 3) fractional atomic coordinates

    Returns
    -------
    F : complex scalar
    """
    hkl = np.asarray(hkl, dtype=np.float64)
    frac_coords = np.asarray(frac_coords, dtype=np.float64)
    phase = 2.0 * np.pi * (frac_coords @ hkl)   # (N_atoms,)
    return complex(np.sum(np.exp(1j * phase)))


# ---------------------------------------------------------------------------
# Lobato parameter lookup
# ---------------------------------------------------------------------------

def _get_lobato_params(atoms) -> tuple:
    """Return (a_table, b_k_table) arrays with shape (Z_max+1, 5).

    Uses abTEM's LobatoParametrization when available (same source as the
    notebook).  Falls back to hardcoded Au (Z=79) parameters from
    Lobato & Van Dyck (2014) otherwise.
    """
    Zs   = atoms.get_atomic_numbers()
    Zmax = int(Zs.max())

    a_table   = np.zeros((Zmax + 1, 5), dtype=np.float64)
    b_k_table = np.zeros((Zmax + 1, 5), dtype=np.float64)

    try:
        from abtem.parametrizations import LobatoParametrization
        from ase.data import chemical_symbols as _csym
        lob = LobatoParametrization()
        for Z in np.unique(Zs):
            sym = _csym[int(Z)]
            p = lob.scaled_parameters(sym, "scattering_factor")
            a_table[int(Z)]   = np.asarray(p[0], dtype=np.float64)
            b_k_table[int(Z)] = np.asarray(p[1], dtype=np.float64)
        return a_table, b_k_table
    except ImportError:
        pass

    # Fallback: only Au hardcoded
    for Z in np.unique(Zs):
        if int(Z) == 79:
            a_table[79]   = _LOBATO_AU_A
            b_k_table[79] = _LOBATO_AU_B_S / 4.0   # s² → k² convention
        else:
            raise ValueError(
                f"Z={Z} is not Au.  Install abTEM for multi-element Lobato parameters."
            )
    return a_table, b_k_table


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_bloch_wave_gpu(
    g_max_zolz: float,
    g_max_holz: float,
    l_max: int,
    n_beams_max: int,
    *,
    atoms,
    wavelength: float,
    x,
    paper_00=None,
    paper_028=None,
    include_eigensystem: bool = False,
    include_structure_samples: bool = False,
) -> dict:
    """Bloch-wave dynamical diffraction solver.

    Builds a 3D reciprocal-lattice beam basis, assembles the dynamical
    structure matrix using Lobato electron scattering factors, solves the
    Hermitian eigenvalue problem (CuPy on GPU when available; numpy fallback),
    and propagates the wave field through a range of crystal thicknesses.

    Parameters
    ----------
    g_max_zolz : float
        Maximum |g| [Å⁻¹] for ZOLZ beams (l = 0).
    g_max_holz : float
        Maximum |g| [Å⁻¹] for HOLZ beams (|l| > 0).
    l_max : int
        Maximum |l| Laue-zone index to include.
    n_beams_max : int
        Hard cap on total number of beams selected.
    atoms : ase.Atoms
        Crystal structure providing cell, positions, and atomic numbers.
    wavelength : float
        Electron de Broglie wavelength [Å].
    x : array-like
        Thickness values in **unit cells** (e.g. ``range(0, 26)``).
        The cell thickness (``atoms.cell[2, 2]``) is used internally.
    paper_00 : array-like, optional
        Reference curve for the [0, 0] beam.  Shape ``(2, N)`` with
        ``paper_00[0]`` = x-axis (unit cells) and ``paper_00[1]`` = amplitude,
        or shape ``(N, 2)`` with columns (x, amp).
    paper_028 : array-like, optional
        Reference curve for the [0, 28] beam.  Same format as *paper_00*.
    include_eigensystem : bool
        If ``True``, add ``'evals'`` and ``'evecs'`` to the returned dict.
    include_structure_samples : bool
        If ``True``, add ``'structure_samples'`` (the F-grid) to the dict.

    Returns
    -------
    dict with at minimum these keys:

    amp_00_coh  : np.ndarray, shape (len(x),)
    amp_028_coh : np.ndarray, shape (len(x),)
    n_beams     : int
    n_zolz      : int
    n_holz      : int
    rmse_00     : float   (only when *paper_00* is given)
    rmse_028    : float   (only when *paper_028* is given)
    rmse_avg    : float   (only when both paper references are given)
    evals       : np.ndarray   (only when *include_eigensystem* is True)
    evecs       : np.ndarray   (only when *include_eigensystem* is True)
    structure_samples : np.ndarray  (only when *include_structure_samples* is True)
    """
    x = np.asarray(x, dtype=np.float64)

    # ------------------------------------------------------------------ #
    # 1. Crystal geometry                                                   #
    # ------------------------------------------------------------------ #
    cell    = np.array(atoms.get_cell(),          dtype=np.float64)   # (3,3) Å
    recip   = np.array(atoms.cell.reciprocal(),   dtype=np.float64)   # (3,3) Å⁻¹
    volume  = float(atoms.cell.volume)                                 # Å³

    frac    = np.array(atoms.get_scaled_positions(), dtype=np.float64) # (Na,3)
    Zs      = np.array(atoms.get_atomic_numbers(),   dtype=np.int32)   # (Na,)

    cell_z  = float(cell[2, 2])   # unit-cell thickness along z [Å]

    # ------------------------------------------------------------------ #
    # 2. Enumerate reciprocal-lattice beams                                #
    # ------------------------------------------------------------------ #
    dg      = np.array([np.linalg.norm(recip[i]) for i in range(3)])
    H_max   = int(np.ceil(max(g_max_zolz, g_max_holz) / max(dg[0], 1e-12)))
    K_max   = int(np.ceil(max(g_max_zolz, g_max_holz) / max(dg[1], 1e-12)))
    L_max   = min(l_max, int(np.ceil(g_max_holz / max(dg[2], 1e-12))))

    hkl_list, g_list = [], []
    for h in range(-H_max, H_max + 1):
        for k in range(-K_max, K_max + 1):
            for l_idx in range(-L_max, L_max + 1):
                gv    = np.array([h, k, l_idx], dtype=np.float64) @ recip
                g_len = np.linalg.norm(gv)
                if l_idx == 0:
                    if g_len <= g_max_zolz + 1e-12:
                        hkl_list.append((h, k, l_idx))
                        g_list.append(gv)
                else:
                    if abs(l_idx) <= l_max and g_len <= g_max_holz + 1e-12:
                        hkl_list.append((h, k, l_idx))
                        g_list.append(gv)

    if not hkl_list:
        raise ValueError("No beams selected – check g_max_zolz / g_max_holz.")

    hkl_arr = np.array(hkl_list, dtype=np.int32)    # (N_all, 3)
    g_arr   = np.array(g_list,   dtype=np.float64)  # (N_all, 3)

    # Ensure (0,0,0) is always first
    z_idx = np.where(np.all(hkl_arr == 0, axis=1))[0]
    if z_idx.size == 0:
        raise ValueError("(0,0,0) beam missing from beam list.")
    if z_idx[0] != 0:
        hkl_arr[[0, z_idx[0]]] = hkl_arr[[z_idx[0], 0]]
        g_arr  [[0, z_idx[0]]] = g_arr  [[z_idx[0], 0]]

    # Cap at n_beams_max (sort by |g|, always keep beam 0)
    g_norms = np.linalg.norm(g_arr, axis=1)
    if len(hkl_arr) > n_beams_max:
        order = np.argsort(g_norms)[:n_beams_max]
        if 0 not in order:
            order[0] = 0
        hkl_arr = hkl_arr[order]
        g_arr   = g_arr  [order]
        g_norms = g_norms[order]

    N      = len(hkl_arr)
    n_zolz = int(np.sum(hkl_arr[:, 2] == 0))
    n_holz = N - n_zolz

    # ------------------------------------------------------------------ #
    # 3. Excitation errors   s_g = -(g_z + |g|²/(2K₀))                   #
    # ------------------------------------------------------------------ #
    K0 = 1.0 / wavelength
    sg = -(g_arr[:, 2] + 0.5 * g_norms ** 2 / K0)   # (N,) Å⁻¹

    # ------------------------------------------------------------------ #
    # 4. Scattering-factor lookup tables                                   #
    # ------------------------------------------------------------------ #
    a_table, b_k_table = _get_lobato_params(atoms)

    # ------------------------------------------------------------------ #
    # 5. Structure-factor grid F[Δh, Δk, Δl] (vectorised)                 #
    #                                                                      #
    # A[i,j] = prefactor × F(g_i − g_j) / Ω                              #
    # F is computed on the full grid of difference vectors.               #
    # ------------------------------------------------------------------ #
    Hs = int(np.max(np.abs(hkl_arr[:, 0])))
    Ks = int(np.max(np.abs(hkl_arr[:, 1])))
    Ls = int(np.max(np.abs(hkl_arr[:, 2])))

    # Grid spans all possible differences Δh ∈ [−2Hs, 2Hs], etc.
    dh_vals = np.arange(-2 * Hs, 2 * Hs + 1, dtype=np.int32)
    dk_vals = np.arange(-2 * Ks, 2 * Ks + 1, dtype=np.int32)
    dl_vals = np.arange(-2 * Ls, 2 * Ls + 1, dtype=np.int32) if Ls > 0 else np.array([0], dtype=np.int32)

    dH, dK, dL = np.meshgrid(dh_vals, dk_vals, dl_vals, indexing="ij")
    dhkl_flat  = np.stack([dH.ravel(), dK.ravel(), dL.ravel()], axis=1)  # (M, 3)

    g_diff_flat = (dhkl_flat.astype(np.float64)) @ recip   # (M, 3) Å⁻¹
    k2_diff     = np.sum(g_diff_flat ** 2, axis=1)          # (M,)

    # Atomic scattering factors: (Na, M)
    f_e = np.zeros((len(Zs), len(k2_diff)), dtype=np.float64)
    for a_idx, Z in enumerate(Zs):
        f_e[a_idx] = _lobato_scattering_factor(k2_diff, a_table[Z], b_k_table[Z])

    # Phase factors: exp(2πi (Δh·xf + Δk·yf + Δl·zf)) — shape (M, Na)
    phase_arg = 2.0 * np.pi * (frac @ dhkl_flat.T)   # (Na, M)
    phases    = np.exp(1j * phase_arg)                 # (Na, M)

    # Structure factor / volume (units: Å⁻³·Å = Å⁻²)
    F_flat = np.sum(f_e * phases, axis=0) / volume     # (M,)
    F_grid = F_flat.reshape(dH.shape)                  # (2Hs*2+1, 2Ks*2+1, 2Ls*2+1 or 1)

    if include_structure_samples:
        structure_samples = F_grid.copy()

    # ------------------------------------------------------------------ #
    # 6. Assemble the structure (Bloch) matrix A                          #
    #    off-diagonal: A[i,j] = prefactor × F(g_i−g_j)                   #
    #    diagonal:     A[i,i] = 2 s_g / λ                                 #
    # ------------------------------------------------------------------ #
    energy_eV = wavelength_to_energy_eV(wavelength)
    sigma_v   = _energy2sigma(energy_eV)
    kappa_v   = _kappa()
    prefactor = sigma_v / (wavelength * np.pi * kappa_v)

    # Difference indices: (N, N) → index into F_grid
    dh_ij = (hkl_arr[:, 0:1] - hkl_arr[np.newaxis, :, 0]).astype(np.int32)   # (N, N)
    dk_ij = (hkl_arr[:, 1:2] - hkl_arr[np.newaxis, :, 1]).astype(np.int32)   # (N, N)
    dl_ij = (hkl_arr[:, 2:3] - hkl_arr[np.newaxis, :, 2]).astype(np.int32)   # (N, N)

    dh_idx = dh_ij + 2 * Hs                            # shift to non-negative
    dk_idx = dk_ij + 2 * Ks
    dl_idx = (dl_ij + 2 * Ls) if Ls > 0 else np.zeros_like(dl_ij)

    # Bounds-check (should always pass by construction)
    dh_idx = np.clip(dh_idx, 0, F_grid.shape[0] - 1)
    dk_idx = np.clip(dk_idx, 0, F_grid.shape[1] - 1)
    dl_idx = np.clip(dl_idx, 0, F_grid.shape[2] - 1)

    A = prefactor * F_grid[dh_idx, dk_idx, dl_idx]     # (N, N) complex128

    # Set diagonal to excitation errors
    np.fill_diagonal(A, (2.0 / wavelength) * sg.astype(A.real.dtype))

    # ------------------------------------------------------------------ #
    # 7. Eigenvalue solve: GPU (CuPy) with CPU fallback                   #
    # ------------------------------------------------------------------ #
    if HAS_CUPY:
        A_gpu             = cp.array(A)
        evals_gpu, evecs_gpu = cp.linalg.eigh(A_gpu)
        evals = cp.asnumpy(evals_gpu)
        evecs = cp.asnumpy(evecs_gpu)
    else:
        evals, evecs = np.linalg.eigh(A)

    # ------------------------------------------------------------------ #
    # 8. Propagate to beam amplitudes                                      #
    #    γ_j = (λ/2) · eval_j                                             #
    #    ψ₀  = [1, 0, …, 0]  (incident beam only)                         #
    #    α_j = (C†ψ₀)_j       (excitation coefficients)                   #
    #    C(z) = C · (α ⊙ e^{2πi γ z})                                     #
    # ------------------------------------------------------------------ #
    gamma = (wavelength / 2.0) * evals             # (N,)
    psi0  = np.zeros(N, dtype=np.complex128)
    psi0[0] = 1.0
    alpha = evecs.conj().T @ psi0                  # (N,)

    # Thickness in Å
    z_vals    = x * cell_z                         # (T,)

    # Phase matrix: (T, N)
    phase_mat = np.exp(2j * np.pi
                       * gamma[np.newaxis, :]
                       * z_vals[:, np.newaxis])

    # C(z): (T, N)  — evecs: (N, N), (alpha * phase_mat).T: (N, T)
    Ct = (evecs @ (alpha[np.newaxis, :] * phase_mat).T).T   # (T, N)

    amp_all = np.abs(Ct)                           # (T, N)

    # ------------------------------------------------------------------ #
    # 9. Extract amplitudes for (0,0,0) and (0,28,0) beams               #
    # ------------------------------------------------------------------ #
    amp_00_coh = amp_all[:, 0]

    mask_028 = (hkl_arr[:, 0] == 0) & (hkl_arr[:, 1] == 28) & (hkl_arr[:, 2] == 0)
    if np.any(mask_028):
        idx_028    = int(np.where(mask_028)[0][0])
        amp_028_coh = amp_all[:, idx_028]
    else:
        amp_028_coh = np.zeros(len(x))

    # ------------------------------------------------------------------ #
    # 10. RMSE against paper reference data                               #
    # ------------------------------------------------------------------ #
    result: dict = {
        "amp_00_coh":  amp_00_coh,
        "amp_028_coh": amp_028_coh,
        "n_beams":     N,
        "n_zolz":      n_zolz,
        "n_holz":      n_holz,
    }

    def _rmse(ref, sim_x, sim_amp):
        ref = np.asarray(ref)
        if ref.ndim == 2 and ref.shape[0] == 2:
            rx, ry = ref[0], ref[1]
        elif ref.ndim == 2 and ref.shape[1] == 2:
            rx, ry = ref[:, 0], ref[:, 1]
        else:
            raise ValueError("Reference data must have shape (2, N) or (N, 2).")
        f_ref = interp1d(rx, ry, kind="linear", fill_value="extrapolate")
        return float(np.sqrt(np.mean((sim_amp - f_ref(sim_x)) ** 2)))

    if paper_00 is not None:
        result["rmse_00"] = _rmse(paper_00, x, amp_00_coh)
    if paper_028 is not None:
        result["rmse_028"] = _rmse(paper_028, x, amp_028_coh)
    if paper_00 is not None and paper_028 is not None:
        result["rmse_avg"] = 0.5 * (result["rmse_00"] + result["rmse_028"])

    if include_eigensystem:
        result["evals"] = evals
        result["evecs"] = evecs

    if include_structure_samples:
        result["structure_samples"] = structure_samples

    return result
