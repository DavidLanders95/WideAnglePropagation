#!/usr/bin/env python3
"""
Generate Figure 2: Beam-amplitude vs crystal thickness for Au at 300 keV.

Compares:
  - Klein-Gordon ODE reference from Rother & Scheerschmidt (2009)
  - Fresnel multislice / Angular Spectrum multislice  (with 95% CI bootstrap)
  - Wave Propagation Method (WPM)
  - Bloch wave dynamical diffraction (this work)

Output: Paper/Au_beam_amplitudes.pdf

Usage:
    python scripts/generate_figure2.py

Environment:
    source /nobackup/dl277493/temgym_core/bin/activate
    pip install -e .
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Add package root to path so the script can be run from repo root
# ---------------------------------------------------------------------------
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ---------------------------------------------------------------------------
# Imports from the package
# ---------------------------------------------------------------------------
from ase.build import bulk

from wide_angle_propagation.bloch import (
    solve_bloch_wave_gpu,
    _energy2wavelength,
    HAS_CUPY,
)
from wide_angle_propagation.reference_data import (
    AU_300KV_BEAM_00_KG_MS,
    AU_300KV_BEAM_028_KG_FWD,
    AU_300KV_BEAM_028_KG_MS,
    get_interp_00_ms,
    get_interp_028_fwd,
    get_interp_028_ms,
)


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
ENERGY    = 300e3          # eV
A_AU      = 4.08           # Å, Au lattice parameter
N_CELLS   = range(0, 26)   # unit-cell thicknesses to simulate
N_BOOTSTRAP = 5            # bootstrap samples for MS / AS CI


def _make_atoms():
    atoms = bulk("Au", "fcc", a=A_AU, cubic=True)
    atoms.info["thermal_sigma"]   = 0.0
    atoms.arrays["thermal_sigma"] = np.zeros(len(atoms))
    return atoms


# ---------------------------------------------------------------------------
# Beam-amplitude extractor (mirrors the notebook helper)
# ---------------------------------------------------------------------------

def _beam_amplitude(psi_xy, h, k, use_fftshift=True):
    """Return normalised |C(h,k)| for a 2-D wave function."""
    import numpy as np
    Ny, Nx = psi_xy.shape
    C = np.fft.fft2(psi_xy) / (Nx * Ny)
    if use_fftshift:
        C  = np.fft.fftshift(C)
        cy = Ny // 2
        cx = Nx // 2
        return float(np.abs(C[cy + k, cx + h]))
    return float(np.abs(C[k % Ny, h % Nx]))


# ---------------------------------------------------------------------------
# Fresnel / Angular Spectrum multislice sweep  (bootstrap over n_lattice slices)
# ---------------------------------------------------------------------------

def _run_multislice_bootstrap(pot_array, probe_array, propagator,
                               slice_thickness, energy,
                               n_cells=N_CELLS, n_bootstrap=N_BOOTSTRAP,
                               sampling=None):
    """Return (mean_00, lo_00, hi_00, mean_028, lo_028, hi_028) over cells."""
    from wide_angle_propagation import simulate_fresnel_as

    all_00  = []
    all_028 = []

    for _ in range(n_bootstrap):
        ms_00, ms_028 = [], []
        current_wave = probe_array.copy()
        for i in N_CELLS:
            if i == 0:
                w = current_wave
            else:
                w, _, _ = simulate_fresnel_as(
                    pot_array, current_wave, propagator,
                    slice_thickness, energy,
                )
                current_wave = w
            ms_00.append(_beam_amplitude(w, 0,  0))
            ms_028.append(_beam_amplitude(w, 0, 28))
        all_00.append(ms_00)
        all_028.append(ms_028)

    all_00  = np.array(all_00)    # (n_bootstrap, T)
    all_028 = np.array(all_028)

    p5, p95 = 2.5, 97.5
    return (
        all_00.mean(axis=0),
        np.percentile(all_00, p5,  axis=0),
        np.percentile(all_00, p95, axis=0),
        all_028.mean(axis=0),
        np.percentile(all_028, p5,  axis=0),
        np.percentile(all_028, p95, axis=0),
    )


# ---------------------------------------------------------------------------
# WPM sweep
# ---------------------------------------------------------------------------

def _run_wpm(pot_array, probe_array, slice_thickness, energy, sampling,
             n_cells=N_CELLS):
    """Return (amp_00, amp_028) arrays over unit cells."""
    from wide_angle_propagation import simulate_wpm

    wpm_00, wpm_028 = [], []
    current_wave = probe_array.copy()
    for i in N_CELLS:
        if i == 0:
            w = current_wave
        else:
            w, _, _ = simulate_wpm(
                pot_array, current_wave, slice_thickness, energy, sampling,
            )
            current_wave = w
        wpm_00.append(_beam_amplitude(w, 0,  0))
        wpm_028.append(_beam_amplitude(w, 0, 28))
    return np.array(wpm_00), np.array(wpm_028)


# ---------------------------------------------------------------------------
# Bloch wave sweep
# ---------------------------------------------------------------------------

def _run_bloch(atoms, wavelength, n_cells=N_CELLS):
    """Return (amp_00, amp_028) from the Bloch wave solver."""
    result = solve_bloch_wave_gpu(
        g_max_zolz=15,
        g_max_holz=25,
        l_max=10,
        n_beams_max=12000,
        atoms=atoms,
        wavelength=wavelength,
        x=list(n_cells),
        paper_00=AU_300KV_BEAM_00_KG_MS,
        paper_028=AU_300KV_BEAM_028_KG_FWD,
    )
    print(f"  Bloch: n_beams={result['n_beams']}, n_zolz={result['n_zolz']}, "
          f"n_holz={result['n_holz']}")
    if "rmse_avg" in result:
        print(f"  Bloch RMSE: 00={result['rmse_00']:.4f}, "
              f"028={result['rmse_028']:.4f}, avg={result['rmse_avg']:.4f}")
    return result["amp_00_coh"], result["amp_028_coh"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import abtem

    abtem.config.set({"device": "cpu"})
    abtem.config.set({"precision": "float64"})

    atoms      = _make_atoms()
    energy     = ENERGY
    wavelength = _energy2wavelength(energy)

    print(f"Au FCC  a={A_AU} Å,  300 keV,  λ={wavelength:.5f} Å")
    print(f"HAS_CUPY = {HAS_CUPY}")

    # ---------- potential setup ----------
    gpts       = (128, 128)
    cell_z     = atoms.cell[2, 2]
    slice_thick = cell_z / 2

    potential = abtem.Potential(
        atoms,
        gpts=gpts,
        slice_thickness=slice_thick,
        projection="infinite",
        parametrization="lobato",
    )
    pot_array = potential.build(lazy=False).array / slice_thick   # (2, 128, 128)

    from wide_angle_propagation import get_abtem_transmit, fresnel_propagation_kernel

    transmit = get_abtem_transmit(potential, energy)

    probe = abtem.PlaneWave(energy=energy)
    probe = probe.build(gpts=gpts, extent=(A_AU, A_AU))
    probe_array = np.array(probe.array)

    sampling = probe.grid.sampling

    fresnel_prop = fresnel_propagation_kernel(
        gpts[0], gpts[1],
        sampling,
        z=slice_thick,
        energy=energy,
    )

    x = np.array(list(N_CELLS), dtype=np.float64)

    # ---------- Multislice ----------
    print("Running Fresnel/AS multislice…")
    ms_00_m, ms_00_lo, ms_00_hi, ms_028_m, ms_028_lo, ms_028_hi = (
        _run_multislice_bootstrap(
            pot_array, probe_array, fresnel_prop,
            slice_thick, energy, sampling=sampling,
        )
    )

    # ---------- WPM ----------
    print("Running WPM…")
    wpm_00, wpm_028 = _run_wpm(pot_array, probe_array, slice_thick, energy, sampling)

    # ---------- Bloch ----------
    print("Running Bloch wave solver…")
    bloch_00, bloch_028 = _run_bloch(atoms, wavelength)

    # ---------- Reference data ----------
    interp_00_ms   = get_interp_00_ms()
    interp_028_fwd = get_interp_028_fwd()
    interp_028_ms  = get_interp_028_ms()

    x_ref = np.linspace(0, 25, 500)
    ref_00_ms   = interp_00_ms(x_ref)
    ref_028_fwd = interp_028_fwd(x_ref)
    ref_028_ms  = interp_028_ms(x_ref)

    # ---------- Plot ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: [0,0] beam ---
    ax1.plot(x_ref, ref_00_ms, color="green", lw=2, label="Klein-Gordon ODE (Rother 2009)")
    ax1.fill_between(x, ms_00_lo, ms_00_hi, color="blue", alpha=0.2)
    ax1.plot(x, ms_00_m,  color="blue",   lw=2,  label="Fresnel / AS MS (mean)")
    ax1.plot(x, wpm_00,   color="red",    lw=2,  label="WPM")
    ax1.plot(x, bloch_00, color="darkorange", lw=2, label="Bloch wave (this work)")

    ax1.set_xlabel("Thickness (unit cells)")
    ax1.set_ylabel("Normalised amplitude |C(0,0)|")
    ax1.set_title("[0, 0] beam — Au, 300 keV")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.4)

    # --- Right panel: [0,28] beam ---
    ax2.plot(x_ref, ref_028_ms, color="green", lw=2,
             label="Klein-Gordon MS (Rother 2009)")
    ax2.plot(x_ref, ref_028_fwd, color="green", lw=2, ls="--",
             label="Klein-Gordon FWD (Rother 2009)")
    ax2.fill_between(x, ms_028_lo, ms_028_hi, color="blue", alpha=0.2)
    ax2.plot(x, ms_028_m,   color="blue",      lw=2, label="Fresnel / AS MS (mean)")
    ax2.plot(x, wpm_028,    color="red",        lw=2, label="WPM")
    ax2.plot(x, bloch_028,  color="darkorange", lw=2, label="Bloch wave (this work)")

    ax2.set_xlabel("Thickness (unit cells)")
    ax2.set_ylabel("Normalised amplitude |C(0,28)|")
    ax2.set_title("[0, 28] beam  (≈135 mrad) — Au, 300 keV")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    fig.suptitle(
        "Au FCC  a=4.08 Å,  E=300 keV  — beam-amplitude vs crystal thickness",
        fontsize=12,
    )
    plt.tight_layout()

    out_dir = os.path.join(_REPO, "Paper")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Au_beam_amplitudes.pdf")
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
