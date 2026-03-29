#!/usr/bin/env python
"""
X-Z cross-section comparison: WPM vs Fresnel through bulk samples.

Records the wavefield at every slice during propagation, then plots
the central x-z plane showing amplitude and phase evolution through
the sample for both Fresnel multislice and WPM.

Usage:
    python wpm/xz_cross_section.py               # GPU run
    python wpm/xz_cross_section.py --quick        # quick subset
    python wpm/xz_cross_section.py --backend cpu
"""

import os, sys, site
from pathlib import Path

# CUDA header auto-detection
if not os.environ.get("CUDA_PATH"):
    for sp in site.getsitepackages():
        candidate = Path(sp) / "nvidia" / "cuda_runtime"
        header = candidate / "include" / "cuda_fp16.h"
        if header.exists():
            os.environ["CUDA_PATH"] = str(candidate)
            break

BACKEND = "gpu"
for i, arg in enumerate(sys.argv[1:]):
    if arg == "--backend" and i + 1 < len(sys.argv[1:]):
        BACKEND = sys.argv[i + 2].lower()
    elif arg.startswith("--backend="):
        BACKEND = arg.split("=", 1)[1].lower()

if BACKEND == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

parent = Path(__file__).resolve().parent.parent
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))

import abtem
from ase.build import bulk, surface

from wide_angle_propagation.propagation import (
    energy2wavelength,
    electron_refractive_index,
    fresnel_propagation_kernel,
    Propagator,
    wpm_step_adaptive,
)

abtem.config.set({"device": "cpu"})  # build potentials on CPU to avoid GPU OOM
abtem.config.set({"precision": "float64"})

OUT_DIR = Path(__file__).resolve().parent / "experiment_results"
OUT_DIR.mkdir(exist_ok=True)


def _to_numpy(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def build_bulk_sample(material, thickness_nm, lateral_size_ang=100.0,
                      sampling_ang=0.1, slice_dz=2.0):
    """Build periodic bulk sample and return (potential_array_V, dz, pot_obj)."""
    thickness_ang = thickness_nm * 10.0

    if material == "Au":
        au_bulk = bulk("Au", "fcc", a=4.076)
        atoms = surface(au_bulk, (1, 1, 0), layers=2)
        atoms = abtem.orthogonalize_cell(atoms)
        z_period = 4.076 / np.sqrt(2)
        atoms.cell[2, 2] = z_period
        atoms.pbc = True
    elif material == "Si":
        si_bulk = bulk("Si", crystalstructure="diamond", a=5.431)
        atoms = surface(si_bulk, (1, 1, 1), layers=3, periodic=True)
        atoms = abtem.orthogonalize_cell(atoms)
    else:
        raise ValueError(f"Unknown material: {material}")

    cell = atoms.cell.lengths()
    nx = max(1, int(np.ceil(lateral_size_ang / cell[0])))
    ny = max(1, int(np.ceil(lateral_size_ang / cell[1])))
    nz = max(1, int(np.ceil(thickness_ang / cell[2])))
    sample = atoms * (nx, ny, nz)

    pot_obj = abtem.Potential(
        sample, sampling=sampling_ang, slice_thickness=slice_dz,
        parametrization="lobato",
    )
    pot_array = _to_numpy(pot_obj.build(lazy=False).array / slice_dz)
    return pot_array, slice_dz, pot_obj


# =========================================================================
# Propagation with slice-by-slice recording
# =========================================================================

def propagate_fresnel_record(potential, probe, dz, energy, sampling):
    """Fresnel MS — returns (n_slices+1, ny, nx) complex wavefield stack."""
    wavelength = energy2wavelength(energy)
    ny, nx = probe.shape
    H = fresnel_propagation_kernel(ny, nx, sampling, z=dz, energy=energy)
    wave = jnp.asarray(probe, dtype=jnp.complex128)
    pot = jnp.asarray(potential)

    slices = [np.array(wave)]  # initial probe
    for i in range(pot.shape[0]):
        n = electron_refractive_index(pot[i], energy)
        phase = jnp.exp(1j * 2 * jnp.pi * (n - 1) * dz / wavelength)
        wave = Propagator(wave * phase, H)
        slices.append(np.array(wave))
    return np.stack(slices, axis=0)


def propagate_wpm_record(potential, probe, dz, energy, sampling, n_bins=64):
    """WPM — returns (n_slices+1, ny, nx) complex wavefield stack."""
    wave = jnp.asarray(probe, dtype=jnp.complex128)
    pot = jnp.asarray(potential)

    slices = [np.array(wave)]
    for i in range(pot.shape[0]):
        n = electron_refractive_index(pot[i], energy)
        wave, _, _, _ = wpm_step_adaptive(
            wave, n, dz, energy, sampling, n_bins=n_bins, power_spacing=2.0,
        )
        slices.append(np.array(wave))
    return np.stack(slices, axis=0)


# =========================================================================
# Plotting
# =========================================================================

def plot_xz_cross_sections(wf_fresnel, wf_wpm, potential, dz, sampling,
                           energy, material, thickness_nm, semiangle,
                           save_path):
    """
    Plot x-z cross-sections through the centre of the sample.

    Layout (3 rows × 3 cols):
      Row 0: Fresnel amplitude | WPM amplitude | Amplitude difference
      Row 1: Fresnel phase     | WPM phase     | Phase difference
      Row 2: Potential x-z     | Amplitude line profiles at select depths | Phase line profiles
    """
    n_slices_plus1, ny, nx = wf_fresnel.shape
    n_slices = n_slices_plus1 - 1
    cy = ny // 2  # central row in y

    # Extract central x-z plane: shape (n_slices+1, nx)
    xz_f = wf_fresnel[:, cy, :]
    xz_w = wf_wpm[:, cy, :]

    amp_f = np.abs(xz_f)
    amp_w = np.abs(xz_w)
    phase_f = np.angle(xz_f)
    phase_w = np.angle(xz_w)

    # Axes
    dx = float(sampling[0])
    x_axis = np.arange(nx) * dx  # Å
    z_axis = np.arange(n_slices_plus1) * dz  # Å (slice 0 = entrance surface)
    z_nm = z_axis / 10.0
    x_nm = x_axis / 10.0

    # Also extract potential x-z slice
    pot_xz = potential[:, cy, :]  # (n_slices, nx)

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle(
        f"{material} bulk x-z cross-section | {thickness_nm:.0f} nm | "
        f"{energy/1e3:.0f} keV | {semiangle:.0f} mrad probe",
        fontsize=14, fontweight="bold",
    )

    # Shared colour scales
    amp_vmax = max(amp_f.max(), amp_w.max())
    amp_vmin = 0

    # --- Row 0: Amplitude ---
    im00 = axes[0, 0].imshow(
        amp_f.T, aspect="auto", origin="lower",
        extent=[z_nm[0], z_nm[-1], x_nm[0], x_nm[-1]],
        cmap="inferno", vmin=amp_vmin, vmax=amp_vmax,
    )
    axes[0, 0].set_title("Fresnel — Amplitude")
    axes[0, 0].set_ylabel("x (nm)")
    fig.colorbar(im00, ax=axes[0, 0], shrink=0.8)

    im01 = axes[0, 1].imshow(
        amp_w.T, aspect="auto", origin="lower",
        extent=[z_nm[0], z_nm[-1], x_nm[0], x_nm[-1]],
        cmap="inferno", vmin=amp_vmin, vmax=amp_vmax,
    )
    axes[0, 1].set_title("WPM — Amplitude")
    fig.colorbar(im01, ax=axes[0, 1], shrink=0.8)

    amp_diff = amp_w - amp_f
    adlim = max(abs(amp_diff.min()), abs(amp_diff.max()), 1e-12)
    im02 = axes[0, 2].imshow(
        amp_diff.T, aspect="auto", origin="lower",
        extent=[z_nm[0], z_nm[-1], x_nm[0], x_nm[-1]],
        cmap="RdBu_r", vmin=-adlim, vmax=adlim,
    )
    axes[0, 2].set_title("Amplitude Difference (WPM − Fresnel)")
    fig.colorbar(im02, ax=axes[0, 2], shrink=0.8)

    # --- Row 1: Phase ---
    im10 = axes[1, 0].imshow(
        phase_f.T, aspect="auto", origin="lower",
        extent=[z_nm[0], z_nm[-1], x_nm[0], x_nm[-1]],
        cmap="twilight", vmin=-np.pi, vmax=np.pi,
    )
    axes[1, 0].set_title("Fresnel — Phase")
    axes[1, 0].set_ylabel("x (nm)")
    fig.colorbar(im10, ax=axes[1, 0], shrink=0.8, label="rad")

    im11 = axes[1, 1].imshow(
        phase_w.T, aspect="auto", origin="lower",
        extent=[z_nm[0], z_nm[-1], x_nm[0], x_nm[-1]],
        cmap="twilight", vmin=-np.pi, vmax=np.pi,
    )
    axes[1, 1].set_title("WPM — Phase")
    fig.colorbar(im11, ax=axes[1, 1], shrink=0.8, label="rad")

    phase_diff = np.angle(np.exp(1j * (phase_w - phase_f)))  # wrapped difference
    im12 = axes[1, 2].imshow(
        phase_diff.T, aspect="auto", origin="lower",
        extent=[z_nm[0], z_nm[-1], x_nm[0], x_nm[-1]],
        cmap="RdBu_r", vmin=-np.pi, vmax=np.pi,
    )
    axes[1, 2].set_title("Phase Difference (WPM − Fresnel)")
    fig.colorbar(im12, ax=axes[1, 2], shrink=0.8, label="rad")

    # --- Row 2: Potential + line profiles ---
    # Potential x-z
    pot_z_nm = (np.arange(potential.shape[0]) + 0.5) * dz / 10.0
    im20 = axes[2, 0].imshow(
        pot_xz.T, aspect="auto", origin="lower",
        extent=[pot_z_nm[0] - dz / 20, pot_z_nm[-1] + dz / 20,
                x_nm[0], x_nm[-1]],
        cmap="viridis",
    )
    axes[2, 0].set_title("Potential x-z (V)")
    axes[2, 0].set_xlabel("z (nm)")
    axes[2, 0].set_ylabel("x (nm)")
    fig.colorbar(im20, ax=axes[2, 0], shrink=0.8, label="V")

    # Amplitude line profiles at selected depths
    ax_amp = axes[2, 1]
    depth_fractions = [0.25, 0.5, 0.75, 1.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(depth_fractions)))
    for frac, col in zip(depth_fractions, colors):
        idx = min(int(frac * n_slices), n_slices)
        z_val = idx * dz / 10.0
        ax_amp.plot(x_nm, amp_f[idx, :], '-', color=col, alpha=0.7,
                    label=f'Fresnel z={z_val:.1f}nm')
        ax_amp.plot(x_nm, amp_w[idx, :], '--', color=col, alpha=0.7,
                    label=f'WPM z={z_val:.1f}nm')
    ax_amp.set_xlabel("x (nm)")
    ax_amp.set_ylabel("Amplitude")
    ax_amp.set_title("Amplitude profiles at depth")
    ax_amp.legend(fontsize=6, ncol=2)
    ax_amp.grid(True, alpha=0.3)

    # Phase line profiles at selected depths
    ax_ph = axes[2, 2]
    for frac, col in zip(depth_fractions, colors):
        idx = min(int(frac * n_slices), n_slices)
        z_val = idx * dz / 10.0
        ax_ph.plot(x_nm, phase_f[idx, :], '-', color=col, alpha=0.7,
                   label=f'Fresnel z={z_val:.1f}nm')
        ax_ph.plot(x_nm, phase_w[idx, :], '--', color=col, alpha=0.7,
                   label=f'WPM z={z_val:.1f}nm')
    ax_ph.set_xlabel("x (nm)")
    ax_ph.set_ylabel("Phase (rad)")
    ax_ph.set_title("Phase profiles at depth")
    ax_ph.legend(fontsize=6, ncol=2)
    ax_ph.grid(True, alpha=0.3)

    for ax in axes.flat:
        if ax.get_xlabel() == "":
            ax.set_xlabel("z (nm)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# =========================================================================
# Main
# =========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backend", default="gpu")
    args = parser.parse_args()

    print("=" * 65)
    print("X-Z Cross-Section Comparison: WPM vs Fresnel")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 65)

    if args.quick:
        configs = [
            # (material, thickness_nm, energy_eV, semiangle_mrad)
            ("Au", 10.0, 100e3, 20.0),
            ("Si", 10.0, 100e3, 20.0),
        ]
    else:
        configs = [
            ("Au", 10.0, 100e3, 20.0),
            ("Au", 50.0, 100e3, 20.0),
            ("Au", 50.0, 100e3, 80.0),
            ("Si", 10.0, 100e3, 20.0),
            ("Si", 50.0, 100e3, 20.0),
            ("Si", 50.0, 100e3, 80.0),
        ]

    sampling_ang = 0.1
    lateral_size = 30.0  # keep small to fit all slices in memory
    n_bins = 64

    for material, t_nm, energy, semi in configs:
        label = f"{material}_{t_nm:.0f}nm_{energy/1e3:.0f}keV_{semi:.0f}mrad"
        print(f"\n--- {label} ---")

        t0 = time.time()
        pot, dz, pot_obj = build_bulk_sample(
            material, t_nm, lateral_size_ang=lateral_size,
            sampling_ang=sampling_ang, slice_dz=2.0,
        )
        print(f"  Potential: {pot.shape} ({pot.shape[0]} slices), "
              f"built in {time.time()-t0:.1f}s")

        # Build probe
        probe_obj = abtem.Probe(energy=energy, semiangle_cutoff=semi, defocus=0)
        probe_obj.grid.match(pot_obj)
        probe_array = _to_numpy(probe_obj.build(lazy=False).array)
        sampling = (float(probe_obj.grid.sampling[0]),
                    float(probe_obj.grid.sampling[1]))
        print(f"  Probe: {probe_array.shape}, sampling={sampling}")

        # Propagate and record all slices
        t0 = time.time()
        wf_fresnel = propagate_fresnel_record(
            pot, probe_array, dz, energy, sampling)
        print(f"  Fresnel: {time.time()-t0:.1f}s, "
              f"wavefield stack: {wf_fresnel.shape}")

        t0 = time.time()
        wf_wpm = propagate_wpm_record(
            pot, probe_array, dz, energy, sampling, n_bins=n_bins)
        print(f"  WPM: {time.time()-t0:.1f}s, "
              f"wavefield stack: {wf_wpm.shape}")

        # Plot
        save_path = OUT_DIR / f"xz_cross_section_{label}.png"
        plot_xz_cross_sections(
            wf_fresnel, wf_wpm, pot, dz, sampling,
            energy, material, t_nm, semi, save_path,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
