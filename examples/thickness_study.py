#!/usr/bin/env python
"""
Thickness study: ptychographic multislice reconstruction quality
================================================================

Explores how well multislice ptychography reconstructs a simple sample at
increasing thicknesses (from a single slice up to ~50 nm) using the three
propagation methods available in the library:

  1. **Fresnel** – paraxial split-step propagator
  2. **Angular Spectrum** – exact Helmholtz propagator
  3. **WPM** – Wave Propagation Method (wide-angle)

For each thickness and method the script:
  * Generates a simple test sample with known features
  * Simulates measured diffraction patterns (forward model + Poisson noise)
  * Performs gradient-based multislice ptychographic reconstruction
  * Reports normalised MSE and Pearson correlation of the recovered potential

Results are printed to stdout and saved as a Matplotlib figure
``thickness_study_results.png``.

Usage
-----
    python examples/thickness_study.py            # quick defaults
    python examples/thickness_study.py --full      # full parameter sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Ensure the package can be found when running from the repo root
sys.path.insert(0, ".")

from wide_angle_propagation.propagation import (
    energy2wavelength,
    fresnel_propagation_kernel,
    angular_spectrum_propagation_kernel,
)
from wide_angle_propagation.ptychography import (
    make_probe,
    generate_scan_positions,
    forward_model,
    reconstruct,
    normalised_mse,
    pearson_correlation,
    make_simple_sample,
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_ENERGY = 200e3          # 200 keV (typical TEM)
DEFAULT_GPTS = (64, 64)        # Grid size
DEFAULT_SAMPLING = (0.2, 0.2)  # Pixel size in Å
DEFAULT_SEMI_ANGLE = 20.0      # mrad
DEFAULT_SLICE_DZ = 2.0         # Å per slice

# Thicknesses to test (nm)
QUICK_THICKNESSES = [0.2, 1.0, 5.0, 10.0, 20.0, 50.0]
FULL_THICKNESSES = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]

METHODS = ["fresnel", "angular_spectrum", "wpm"]


def _add_poisson_noise(dps, total_counts=1e6, seed=42):
    """Add Poisson noise to diffraction patterns."""
    key = jax.random.PRNGKey(seed)
    scale = total_counts / jnp.sum(dps, axis=(-2, -1), keepdims=True).mean()
    scaled = dps * scale
    noisy = jax.random.poisson(key, scaled).astype(jnp.float64)
    return noisy / scale


# ============================================================================
# Main study
# ============================================================================

def run_thickness_study(
    thicknesses_nm: list[float],
    energy: float = DEFAULT_ENERGY,
    gpts: tuple[int, int] = DEFAULT_GPTS,
    sampling: tuple[float, float] = DEFAULT_SAMPLING,
    semi_angle_mrad: float = DEFAULT_SEMI_ANGLE,
    slice_dz: float = DEFAULT_SLICE_DZ,
    n_positions: int = 9,
    n_iterations: int = 40,
    learning_rate: float = 5e-2,
    n_bins: int = 64,
    power_spacing: float = 2.0,
    noise_counts: float = 1e8,
    verbose: bool = True,
):
    """Run the full thickness study.

    Returns
    -------
    results : dict
        ``{method: {thickness_nm: {nmse, pearson, time_s, loss_final}}}``
    """
    results: dict = {m: {} for m in METHODS}

    for thick in thicknesses_nm:
        # ---- Generate ground-truth sample ----
        gt_potential = make_simple_sample(
            gpts, sampling, thick, slice_dz, energy,
        )
        n_slices = gt_potential.shape[0]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Thickness = {thick:.1f} nm  ({n_slices} slice(s))")
            print(f"{'='*60}")

        # ---- Probe and scan positions ----
        probe = make_probe(gpts, sampling, energy, semi_angle_mrad)
        positions = generate_scan_positions(
            gpts, sampling, probe_region_frac=0.5, n_positions=n_positions,
        )

        for method in METHODS:
            if verbose:
                print(f"\n--- {method.upper()} ---")

            # Pre-compute propagation kernel
            prop_kernel = None
            if method == "fresnel":
                prop_kernel = fresnel_propagation_kernel(
                    gpts[0], gpts[1], sampling, slice_dz, energy,
                )
            elif method == "angular_spectrum":
                prop_kernel = angular_spectrum_propagation_kernel(
                    gpts[0], gpts[1], sampling, slice_dz, energy,
                )

            # -- Simulate measured data --
            measured_dps = forward_model(
                gt_potential, probe, positions, method,
                slice_dz, energy, sampling,
                prop_kernel=prop_kernel,
                n_bins=n_bins, power_spacing=power_spacing,
            )
            measured_dps = _add_poisson_noise(measured_dps, noise_counts)

            # -- Reconstruct --
            t0 = time.time()
            recon, losses = reconstruct(
                measured_dps, positions, method, gpts,
                n_slices=n_slices,
                slice_thickness=slice_dz,
                energy=energy,
                sampling=sampling,
                semi_angle_mrad=semi_angle_mrad,
                n_iterations=n_iterations,
                learning_rate=learning_rate,
                n_bins=n_bins,
                power_spacing=power_spacing,
                verbose=verbose,
            )
            elapsed = time.time() - t0

            # -- Metrics --
            nmse = normalised_mse(gt_potential, recon)
            corr = pearson_correlation(gt_potential, recon)

            results[method][thick] = {
                "nmse": nmse,
                "pearson": corr,
                "time_s": elapsed,
                "loss_final": losses[-1] if losses else float("nan"),
                "n_slices": n_slices,
            }

            if verbose:
                print(
                    f"  NMSE = {nmse:.4f}  |  Pearson r = {corr:.4f}  |  "
                    f"time = {elapsed:.1f}s"
                )

    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_results(results, save_path="thickness_study_results.png"):
    """Create a two-panel figure: NMSE and Pearson vs thickness."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    markers = {"fresnel": "o-", "angular_spectrum": "s--", "wpm": "D-."}
    colors = {"fresnel": "#1f77b4", "angular_spectrum": "#ff7f0e", "wpm": "#2ca02c"}
    labels = {"fresnel": "Fresnel", "angular_spectrum": "Angular Spectrum", "wpm": "WPM"}

    for method in METHODS:
        thicknesses = sorted(results[method].keys())
        nmses = [results[method][t]["nmse"] for t in thicknesses]
        pearsons = [results[method][t]["pearson"] for t in thicknesses]

        ax1.plot(
            thicknesses, nmses, markers[method],
            label=labels[method], color=colors[method], linewidth=1.5,
        )
        ax2.plot(
            thicknesses, pearsons, markers[method],
            label=labels[method], color=colors[method], linewidth=1.5,
        )

    ax1.set_xlabel("Sample thickness (nm)")
    ax1.set_ylabel("Normalised MSE")
    ax1.set_title("Reconstruction error vs thickness")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.set_xlabel("Sample thickness (nm)")
    ax2.set_ylabel("Pearson correlation")
    ax2.set_title("Reconstruction quality vs thickness")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\nFigure saved to {save_path}")
    return fig


def plot_slice_comparison(
    gt_potential, recon_dict, thickness_nm, save_path=None,
):
    """Show ground-truth vs reconstructed middle slice for each method."""
    n_methods = len(recon_dict)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 4))

    mid = gt_potential.shape[0] // 2
    gt_slice = np.array(gt_potential[mid])
    vmin, vmax = gt_slice.min(), gt_slice.max()

    axes[0].imshow(gt_slice, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground truth")
    axes[0].axis("off")

    for i, (method, recon) in enumerate(recon_dict.items()):
        r_slice = np.array(recon[mid])
        axes[i + 1].imshow(r_slice, cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i + 1].set_title(method)
        axes[i + 1].axis("off")

    fig.suptitle(f"Thickness = {thickness_nm:.1f} nm (slice {mid})", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ptychographic multislice reconstruction thickness study",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run the full set of thicknesses (slower).",
    )
    parser.add_argument(
        "--iters", type=int, default=40,
        help="Number of reconstruction iterations per run (default: 40).",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-2,
        help="Adam learning rate (default: 0.05).",
    )
    parser.add_argument(
        "--positions", type=int, default=9,
        help="Number of probe scan positions (default: 9).",
    )
    parser.add_argument(
        "--output", type=str, default="thickness_study_results.png",
        help="Path for output figure.",
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Path to save results as JSON.",
    )
    args = parser.parse_args()

    thicknesses = FULL_THICKNESSES if args.full else QUICK_THICKNESSES

    results = run_thickness_study(
        thicknesses_nm=thicknesses,
        n_iterations=args.iters,
        learning_rate=args.lr,
        n_positions=args.positions,
    )

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"{'Thickness (nm)':>15}"
    for m in METHODS:
        header += f"  {'NMSE':>10}  {'Pearson':>8}"
    print(header)
    print("-" * len(header))

    for thick in sorted(next(iter(results.values())).keys()):
        row = f"{thick:>15.1f}"
        for m in METHODS:
            r = results[m][thick]
            row += f"  {r['nmse']:>10.4f}  {r['pearson']:>8.4f}"
        print(row)

    # Save figure
    plot_results(results, save_path=args.output)

    # Optionally save JSON
    if args.json:
        # Convert keys to strings for JSON serialisation
        json_results = {}
        for m in results:
            json_results[m] = {
                str(k): v for k, v in results[m].items()
            }
        with open(args.json, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()
