#!/usr/bin/env python
"""
Thick-sample WPM vs Fresnel experiment: 2×2 forward/reconstruction comparison.

For a fair comparison we generate ground-truth 4D-STEM datasets with BOTH
forward models (Fresnel and WPM), then reconstruct each dataset with BOTH
propagators.  This gives a 2×2 matrix:

    Forward / Recon   |  Fresnel-recon  |  WPM-recon
    -----------------+-----------------+-----------
    Fresnel-forward  |  F→F (matched)  |  F→W
    WPM-forward      |  W→F            |  W→W (matched)

If WPM provides an advantage for thick samples, we expect:
  - W→W to outperform W→F  (WPM data needs WPM physics to fit)
  - W→F to have higher residual than F→F  (model mismatch)
  - The gap between F→F and W→F reveals the physics missed by Fresnel.

The experiment is run on an abTEM Au sample at various thicknesses.

Usage:
    python wpm/thick_sample_experiment.py
    python wpm/thick_sample_experiment.py --quick    # smaller grid, fewer iters
"""

import os, sys, site
from pathlib import Path

# CUDA header auto-detection for CuPy
if not os.environ.get("CUDA_PATH"):
    for sp in site.getsitepackages():
        candidate = Path(sp) / "nvidia" / "cuda_runtime"
        header = candidate / "include" / "cuda_fp16.h"
        if header.exists():
            os.environ["CUDA_PATH"] = str(candidate)
            break

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import json
import argparse

parent = Path(__file__).resolve().parent.parent
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))

import abtem
from ase.build import bulk, surface

from wide_angle_propagation.propagation import (
    energy2wavelength,
    electron_refractive_index,
)
from wide_angle_propagation.ptychography import (
    make_probe,
    make_fresnel_kernel,
    multislice_forward_fresnel,
    multislice_forward_fresnel_scan,
    multislice_forward_wpm,
    propagate_wpm,
    shift_probe,
    amplitude_loss,
    MultislicePtychographyReconstructor,
    simulate_4dstem,
    make_grid_scan,
)

abtem.config.set({"device": "gpu"})
abtem.config.set({"precision": "float64"})

OUT_DIR = Path(__file__).resolve().parent / "experiment_results"
OUT_DIR.mkdir(exist_ok=True)


# =========================================================================
# Sample construction
# =========================================================================

def _to_numpy(arr):
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def build_au_sample(thickness_nm, lateral_size_ang=50.0, sampling_ang=0.2,
                    slice_dz=2.0):
    """Build a bulk Au sample and return potential slices + metadata."""
    thickness_ang = thickness_nm * 10.0
    au_bulk = bulk("Au", "fcc", a=4.078)
    atoms = surface(au_bulk, (1, 1, 0), layers=2)
    atoms = abtem.orthogonalize_cell(atoms)
    z_period = 4.078 / np.sqrt(2)
    atoms.cell[2, 2] = z_period
    atoms.pbc = [True, True, True]

    cell_lengths = atoms.cell.lengths()
    nrep_x = max(1, int(np.ceil(lateral_size_ang / cell_lengths[0])))
    nrep_y = max(1, int(np.ceil(lateral_size_ang / cell_lengths[1])))
    nrep_z = max(1, int(np.ceil(thickness_ang / cell_lengths[2])))
    sample = atoms * (nrep_x, nrep_y, nrep_z)

    potential_obj = abtem.Potential(
        sample, sampling=sampling_ang, slice_thickness=slice_dz,
        parametrization="lobato",
    )
    pot_array = _to_numpy(potential_obj.build(lazy=False).array / slice_dz)
    print(f"  Au sample: {thickness_nm} nm, {pot_array.shape[0]} slices, "
          f"grid {pot_array.shape[1]}×{pot_array.shape[2]}")
    return pot_array, slice_dz


def insert_vacancy(potential, defect_slice_idx):
    """Zero out potential in a small disc at the given depth."""
    pot = potential.copy()
    ny, nx = pot.shape[1], pot.shape[2]
    cy, cx = ny // 2, nx // 2
    radius = max(3, ny // 15)
    Y, X = np.mgrid[:ny, :nx]
    mask = ((X - cx)**2 + (Y - cy)**2) <= radius**2
    pot[defect_slice_idx][mask] = 0.0
    return pot


# =========================================================================
# Generate 4D-STEM data with a specific forward model
# =========================================================================

def generate_4dstem(potential, probe, positions, dz, sampling, energy,
                    propagator='fresnel', n_bins=32):
    """
    Generate ground-truth 4D-STEM data using the specified propagator.
    Returns (dps, transmissions_or_nmaps).
    """
    potential_jax = jnp.asarray(potential, dtype=jnp.float64)
    t0 = time.time()
    dps, trans = simulate_4dstem(
        potential_jax, probe, positions,
        dz=dz, sampling=sampling, energy=energy,
        propagator=propagator, n_bins=n_bins,
    )
    elapsed = time.time() - t0
    print(f"    {propagator} forward: {elapsed:.1f}s, "
          f"DP range [{dps.min():.2e}, {dps.max():.2e}]")
    return dps, trans


# =========================================================================
# Reconstruct
# =========================================================================

def reconstruct(dps, probe, positions, n_recon_slices, recon_dz, sampling,
                energy, propagator, n_iterations=300, lr=0.02, n_bins=32,
                optimizer=None, verbose=False):
    """
    Run ptychographic reconstruction. Returns (reconstructor, losses).
    """
    t0 = time.time()
    recon = MultislicePtychographyReconstructor(
        measured_dps=dps,
        probe=probe,
        positions_pix=positions,
        n_slices=n_recon_slices,
        dz=recon_dz,
        sampling=sampling,
        energy=energy,
        propagator=propagator,
        n_bins=n_bins,
        learning_rate=lr,
        loss_fn='amplitude',
        optimizer=optimizer,
    )
    losses = recon.reconstruct(n_iterations=n_iterations, verbose=verbose)
    elapsed = time.time() - t0
    print(f"    {propagator} recon ({n_iterations} iters): "
          f"{elapsed:.1f}s, loss {losses[0]:.3e} → {losses[-1]:.3e}")
    return recon, losses


# =========================================================================
# Plotting
# =========================================================================

def plot_2x2_comparison(results, potential, defect_potential, defect_idx,
                        n_total_slices, n_recon_slices, thickness_nm,
                        save_path):
    """
    Plot the 2×2 forward/recon matrix for one thickness.

    results: dict with keys like ('fresnel','fresnel'), ('fresnel','wpm'), etc.
             each value is (recon, losses) for (defect, reference).
    """
    forward_models = ['fresnel', 'wpm']
    recon_models = ['fresnel', 'wpm']

    expected_slice = min(
        int(defect_idx / n_total_slices * n_recon_slices),
        n_recon_slices - 1,
    )

    n_cols = max(n_recon_slices, 2)
    fig, axes = plt.subplots(6, n_cols + 1, figsize=(4 * (n_cols + 1), 28))
    fig.suptitle(
        f"Thick Sample Experiment: {thickness_nm} nm Au\n"
        f"Defect at slice {defect_idx}/{n_total_slices} "
        f"(expected recon slice {expected_slice}/{n_recon_slices})\n"
        f"2×2: forward model (rows) × reconstruction method (cols in ΔPhase)",
        fontsize=13, fontweight='bold',
    )

    # Row 0: Potential cross-sections
    pot_y = potential.shape[1] // 2
    ax = axes[0, 0]
    ax.imshow(potential[:, pot_y, :].T, aspect='auto', cmap='hot', origin='lower')
    ax.set_title("Reference potential\n(z vs x)", fontsize=9)
    ax.set_xlabel("Slice (z)"); ax.set_ylabel("x pixel")

    ax = axes[0, 1]
    ax.imshow(defect_potential[:, pot_y, :].T, aspect='auto', cmap='hot', origin='lower')
    ax.axvline(defect_idx, color='cyan', ls='--', lw=1.5)
    ax.set_title("Defect potential\n(z vs x)", fontsize=9)
    ax.set_xlabel("Slice (z)")
    for j in range(2, n_cols + 1):
        axes[0, j].axis('off')

    # Row 1-4: ΔPhase per (forward, recon) combo
    row_idx = 1
    all_variances = {}
    for fwd in forward_models:
        for rec in recon_models:
            key = (fwd, rec)
            recon_def, losses_def = results[key]['defect']
            recon_ref, losses_ref = results[key]['ref']
            phase_def = recon_def.get_recovered_phase()
            phase_ref = recon_ref.get_recovered_phase()
            delta_phase = phase_def - phase_ref

            vmax = max(abs(delta_phase.min()), abs(delta_phase.max()), 1e-10)
            variances = [float(np.var(delta_phase[s])) for s in range(n_recon_slices)]
            all_variances[key] = variances

            for s in range(n_recon_slices):
                ax = axes[row_idx, s]
                im = ax.imshow(delta_phase[s], cmap='RdBu_r', origin='lower',
                               vmin=-vmax, vmax=vmax)
                star = " ★" if s == expected_slice else ""
                ax.set_title(f"s{s}{star}", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(im, ax=ax, shrink=0.7)

            # Label in the last column
            ax_lab = axes[row_idx, n_cols]
            ax_lab.axis('off')
            ax_lab.text(0.1, 0.5,
                        f"Fwd={fwd}\nRec={rec}\n"
                        f"Loss: {losses_def[-1]:.2e}\n"
                        f"Var peak: s{np.argmax(variances)}",
                        fontsize=10, transform=ax_lab.transAxes,
                        verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            row_idx += 1

    # Row 5: Loss curves + depth localisation bar chart
    ax_loss = axes[5, 0]
    colors = {'fresnel': 'C0', 'wpm': 'C1'}
    linestyles = {'fresnel': '-', 'wpm': '--'}
    for fwd in forward_models:
        for rec in recon_models:
            key = (fwd, rec)
            _, losses = results[key]['defect']
            ax_loss.semilogy(
                losses,
                linestyle=linestyles[rec],
                color=colors[fwd],
                linewidth=1.5,
                label=f"{fwd}→{rec}",
            )
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Convergence (defect)', fontsize=9)
    ax_loss.legend(fontsize=7)
    ax_loss.grid(True, alpha=0.3)

    # Bar chart: variance per slice for all 4 combos
    ax_bar = axes[5, 1]
    x = np.arange(n_recon_slices)
    bar_width = 0.2
    combo_labels = []
    for i, (fwd, rec) in enumerate([(f, r) for f in forward_models for r in recon_models]):
        key = (fwd, rec)
        offset = (i - 1.5) * bar_width
        bars = ax_bar.bar(x + offset, all_variances[key], bar_width,
                          label=f"{fwd}→{rec}", alpha=0.7)
    ax_bar.axvline(expected_slice, color='red', ls='--', lw=2, label='Expected')
    ax_bar.set_xlabel('Recon Slice')
    ax_bar.set_ylabel('ΔPhase Variance')
    ax_bar.set_title('Depth Localisation', fontsize=9)
    ax_bar.legend(fontsize=6, ncol=2)
    ax_bar.set_xticks(x)

    # Summary table in remaining columns
    ax_tab = axes[5, 2]
    ax_tab.axis('off')
    rows = []
    for fwd in forward_models:
        for rec in recon_models:
            key = (fwd, rec)
            _, losses = results[key]['defect']
            vars_ = all_variances[key]
            peak = int(np.argmax(vars_))
            err = abs(peak - expected_slice)
            contrast = vars_[peak] / (np.mean(vars_) + 1e-30)
            rows.append([
                f"{fwd}→{rec}",
                f"{losses[-1]:.2e}",
                str(peak),
                str(err),
                f"{contrast:.1f}×",
            ])
    table = ax_tab.table(
        cellText=rows,
        colLabels=["Combo", "Final Loss", "Peak Slice", "Depth Err", "Contrast"],
        cellLoc='center', loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)

    for j in range(3, n_cols + 1):
        axes[5, j].axis('off')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")


def plot_dp_cross_sections(dps_fresnel, dps_wpm, thickness_nm, save_path):
    """
    Plot cross-sections through diffraction patterns from both forward models
    to visualise where WPM and Fresnel differ.
    """
    # Use first scan position
    dp_f = dps_fresnel[0]
    dp_w = dps_wpm[0]
    ny, nx = dp_f.shape
    cy, cx = ny // 2, nx // 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"DP Cross-Sections: Fresnel vs WPM Forward Model ({thickness_nm} nm Au)",
        fontsize=13, fontweight='bold',
    )

    # Log-scale DP images
    for ax, dp, label in [(axes[0, 0], dp_f, 'Fresnel'), (axes[0, 1], dp_w, 'WPM')]:
        im = ax.imshow(np.log10(dp + 1e-10), cmap='inferno', origin='lower')
        ax.set_title(f"{label} DP (log10)", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.axhline(cy, color='cyan', ls='--', lw=0.5, alpha=0.5)
        ax.axvline(cx, color='cyan', ls='--', lw=0.5, alpha=0.5)

    # Horizontal cross-section through centre
    ax = axes[1, 0]
    line_f = dp_f[cy, :]
    line_w = dp_w[cy, :]
    freq = np.arange(nx) - cx
    ax.semilogy(freq, line_f + 1e-15, label='Fresnel', linewidth=1.5)
    ax.semilogy(freq, line_w + 1e-15, label='WPM', linewidth=1.5)
    ax.set_xlabel('Pixel (from centre)')
    ax.set_ylabel('Intensity')
    ax.set_title('Horizontal cross-section', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Difference
    ax = axes[1, 1]
    diff = dp_w - dp_f
    rel_diff = diff / (dp_f + 1e-15)
    ax.plot(freq, rel_diff[cy, :], linewidth=1.5, color='C2')
    ax.set_xlabel('Pixel (from centre)')
    ax.set_ylabel('(WPM - Fresnel) / Fresnel')
    ax.set_title('Relative difference (horizontal)', fontsize=10)
    ax.axhline(0, color='k', ls='-', lw=0.5)
    ax.grid(True, alpha=0.3)

    # Summary stats
    rms_diff = float(np.sqrt(np.mean(diff**2)))
    nrmse = rms_diff / (float(dp_f.max()) + 1e-30)
    max_rel = float(np.max(np.abs(rel_diff[cy, :])))
    ax.text(0.02, 0.98,
            f"NRMSE = {nrmse:.4f}\nMax rel diff = {max_rel:.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =========================================================================
# Main experiment
# =========================================================================

def run_experiment(thickness_nm, energy, semiangle_mrad, n_recon_slices,
                   n_scan, n_iterations, defect_depth_frac, lr,
                   n_bins, quick):
    """Run the full 2×2 experiment for one thickness."""
    print(f"\n{'='*65}")
    print(f"  thickness={thickness_nm} nm, E={energy/1e3:.0f} keV, "
          f"semi={semiangle_mrad} mrad, {n_recon_slices} recon slices")
    print(f"{'='*65}")

    # 1. Build sample
    lateral = 30.0 if quick else 50.0
    samp_val = 0.2
    pot_ref, dz = build_au_sample(
        thickness_nm, lateral_size_ang=lateral,
        sampling_ang=samp_val, slice_dz=2.0,
    )
    n_total = pot_ref.shape[0]
    ny, nx = pot_ref.shape[1], pot_ref.shape[2]
    sampling = (samp_val, samp_val)

    # Insert defect
    defect_idx = int(defect_depth_frac * n_total)
    defect_idx = min(defect_idx, n_total - 1)
    pot_def = insert_vacancy(pot_ref, defect_idx)
    print(f"  Defect at slice {defect_idx}/{n_total}")

    # 2. Probe and scan positions
    probe = make_probe(ny, nx, sampling, energy, semiangle_mrad)
    positions = make_grid_scan(ny, nx, n_scan, n_scan,
                               margin_pix=max(4, ny // 10))
    print(f"  {positions.shape[0]} scan positions, grid {ny}×{nx}")

    recon_dz = thickness_nm * 10.0 / n_recon_slices

    # 3. Generate ground-truth DPs with BOTH forward models
    forward_dps = {}
    for fwd in ['fresnel', 'wpm']:
        print(f"\n  Forward model: {fwd}")
        for label, pot in [('defect', pot_def), ('ref', pot_ref)]:
            print(f"    Simulating {label}...")
            dps, _ = generate_4dstem(
                pot, probe, positions, dz, sampling, energy,
                propagator=fwd, n_bins=n_bins,
            )
            forward_dps[(fwd, label)] = dps

    # 4. Plot DP cross-sections (defect sample, position 0)
    plot_dp_cross_sections(
        forward_dps[('fresnel', 'defect')],
        forward_dps[('wpm', 'defect')],
        thickness_nm,
        OUT_DIR / f"dp_cross_section_{thickness_nm}nm.png",
    )

    # 5. 2×2 reconstruction
    results = {}
    for fwd in ['fresnel', 'wpm']:
        for rec in ['fresnel', 'wpm']:
            key = (fwd, rec)
            results[key] = {}
            for label in ['defect', 'ref']:
                tag = f"{fwd}→{rec} ({label})"
                print(f"\n  Reconstructing: {tag}")
                recon, losses = reconstruct(
                    forward_dps[(fwd, label)],
                    probe, positions, n_recon_slices, recon_dz,
                    sampling, energy,
                    propagator=rec,
                    n_iterations=n_iterations,
                    lr=lr,
                    n_bins=n_bins,
                    verbose=(label == 'defect' and rec == 'fresnel'),
                )
                results[key][label] = (recon, losses)

    # 6. Plot 2×2 comparison
    plot_2x2_comparison(
        results, pot_ref, pot_def, defect_idx,
        n_total, n_recon_slices, thickness_nm,
        OUT_DIR / f"thick_sample_2x2_{thickness_nm}nm.png",
    )

    # 7. Collect summary metrics
    summary = {}
    expected_slice = min(
        int(defect_idx / n_total * n_recon_slices),
        n_recon_slices - 1,
    )
    for fwd in ['fresnel', 'wpm']:
        for rec in ['fresnel', 'wpm']:
            key = (fwd, rec)
            recon_def, losses_def = results[key]['defect']
            recon_ref, losses_ref = results[key]['ref']
            delta = recon_def.get_recovered_phase() - recon_ref.get_recovered_phase()
            variances = [float(np.var(delta[s])) for s in range(n_recon_slices)]
            peak = int(np.argmax(variances))
            summary[f"{fwd}→{rec}"] = {
                "final_loss": float(losses_def[-1]),
                "peak_slice": peak,
                "expected_slice": expected_slice,
                "depth_error": abs(peak - expected_slice),
                "phase_contrast": variances[peak] / (np.mean(variances) + 1e-30),
                "variances": variances,
            }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Thick-sample WPM vs Fresnel 2×2 experiment")
    parser.add_argument("--quick", action="store_true",
                        help="Smaller grid, fewer iterations")
    args = parser.parse_args()

    print("=" * 65)
    print("Thick Sample Experiment: WPM vs Fresnel (2×2 comparison)")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 65)

    # Experiment parameters
    energy = 100e3
    semiangle = 20.0
    defect_frac = 0.5
    n_bins = 32

    if args.quick:
        configs = [
            # (thickness_nm, n_recon_slices, n_scan, n_iter, lr)
            (5.0,  3, 4, 100, 0.02),
        ]
    else:
        configs = [
            (5.0,  3, 6, 300, 0.02),
            (10.0, 5, 6, 300, 0.02),
            (20.0, 5, 6, 400, 0.02),
        ]

    all_summaries = {}
    for thickness, n_rec, n_scan, n_iter, lr in configs:
        summary = run_experiment(
            thickness_nm=thickness,
            energy=energy,
            semiangle_mrad=semiangle,
            n_recon_slices=n_rec,
            n_scan=n_scan,
            n_iterations=n_iter,
            defect_depth_frac=defect_frac,
            lr=lr,
            n_bins=n_bins,
            quick=args.quick,
        )
        all_summaries[f"{thickness}nm"] = summary

    # Print summary table
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"{'Thickness':<10} {'Combo':<16} {'Final Loss':<12} "
          f"{'Peak':<6} {'Expected':<10} {'Err':<5} {'Contrast':<10}")
    print("-" * 75)
    for thick_label, summary in all_summaries.items():
        for combo, metrics in summary.items():
            print(f"{thick_label:<10} {combo:<16} "
                  f"{metrics['final_loss']:<12.3e} "
                  f"{metrics['peak_slice']:<6} "
                  f"{metrics['expected_slice']:<10} "
                  f"{metrics['depth_error']:<5} "
                  f"{metrics['phase_contrast']:<10.1f}×")
        print()

    # Save results
    out_path = OUT_DIR / "thick_sample_2x2_results.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
