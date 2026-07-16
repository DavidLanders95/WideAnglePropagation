"""Generate the Si [111] CBED paper figure from converged simulation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "Paper" / "figures" / "method_explainer_generations",
    )
    return parser.parse_args()


def radial_profiles(patterns, theta_mrad, bin_width=0.5, max_angle=300.0):
    valid = theta_mrad <= max_angle
    indices = np.floor(theta_mrad[valid] / bin_width).astype(int)
    n_bins = int(np.floor(max_angle / bin_width))
    centers = (np.arange(n_bins) + 0.5) * bin_width
    profiles = {}
    for name, pattern in patterns.items():
        sums = np.bincount(indices, weights=pattern[valid], minlength=n_bins)[:n_bins]
        profiles[name] = sums / max(float(np.sum(pattern)), 1.0e-30)
    return centers, profiles


def match_peaks(compare, reference, maximum_distance=4.0):
    reference_angles = []
    shifts = []
    if not len(compare):
        return np.asarray(reference_angles), np.asarray(shifts)
    for value in reference:
        index = int(np.argmin(np.abs(compare - value)))
        if abs(compare[index] - value) <= maximum_distance:
            reference_angles.append(value)
            shifts.append(compare[index] - value)
    return np.asarray(reference_angles), np.asarray(shifts)


def dispersion_error(angle_mrad, wavelength):
    sine_squared = np.sin(np.asarray(angle_mrad) * 1.0e-3) ** 2
    exact = np.sqrt(1.0 - sine_squared) - 1.0
    paraxial = -0.5 * sine_squared
    return np.abs(2.0 * np.pi * (paraxial - exact) / wavelength)


def main() -> None:
    args = parse_args()
    with np.load(args.results, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        method_names = [str(value) for value in data["method_names"]]
        reference_name = str(data["reference_method"])
        stored_patterns = data["final_patterns"].copy()
        norm_ratio = data["norm_ratio"].copy()
        amplitude_rrmse = data["amplitude_rrmse_300mrad"].copy()

    patterns = {
        name: stored_patterns[index]
        for index, name in enumerate(method_names)
    }
    display_names = {
        "F-MS": "F-MS",
        "AS-MS": "AS-MS",
        reference_name: "WP-MS",
    }
    selected = (reference_name, "AS-MS", "F-MS")

    shape = patterns[reference_name].shape
    sampling_y, sampling_x = metadata["sampling_A"]
    wavelength = metadata["wavelength_A"]
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0], d=sampling_y))
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1], d=sampling_x))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    theta_mrad = 1.0e3 * np.arcsin(
        np.clip(wavelength * np.sqrt(fx_grid**2 + fy_grid**2), 0.0, 1.0)
    )
    inside = theta_mrad <= 300.0
    y_indices, x_indices = np.nonzero(inside)
    y_slice = slice(y_indices.min(), y_indices.max() + 1)
    x_slice = slice(x_indices.min(), x_indices.max() + 1)
    extent = [fx[x_slice.start], fx[x_slice.stop - 1], fy[y_slice.start], fy[y_slice.stop - 1]]
    extent = [1.0e3 * np.arcsin(np.clip(wavelength * value, -1, 1)) for value in extent]

    pattern = patterns[reference_name][y_slice, x_slice]
    pattern_mask = inside[y_slice, x_slice]
    pattern = np.where(pattern_mask, pattern, np.nan)
    pattern = pattern / np.nanmax(pattern)
    pattern = np.where(np.isfinite(pattern), np.maximum(pattern, 1.0e-8), np.nan)

    centers_raw, profiles_raw = radial_profiles(
        {name: patterns[name] for name in selected}, theta_mrad
    )
    centers = np.arange(0.0, 300.0 + 0.1, 0.1)
    profiles = {
        name: np.interp(centers, centers_raw, values)
        for name, values in profiles_raw.items()
    }
    peak_mask = (centers >= 5.0) & (centers <= 300.0)
    peaks = {}
    for name, values in profiles.items():
        logarithm = np.log10(np.maximum(values, 1.0e-18))
        indices, _ = find_peaks(
            np.where(peak_mask, logarithm, -np.inf),
            prominence=2.0,
            distance=int(round(4.0 / 0.1)),
        )
        peaks[name] = centers[indices]

    wp_angles, wp_shifts = match_peaks(peaks[reference_name], peaks["AS-MS"])
    f_angles, f_shifts = match_peaks(peaks["F-MS"], peaks["AS-MS"])
    theory_at_peaks = dispersion_error(f_angles, wavelength)
    scale = (
        float(np.dot(theory_at_peaks, np.abs(f_shifts)) / np.dot(theory_at_peaks, theory_at_peaks))
        if len(f_angles) >= 2 and np.dot(theory_at_peaks, theory_at_peaks) > 0
        else 0.0
    )

    colours = {reference_name: "C2", "AS-MS": "C1", "F-MS": "C0"}
    linestyles = {reference_name: "-", "AS-MS": "--", "F-MS": ":"}
    figure = plt.figure(figsize=(13.5, 7.2))
    grid = GridSpec(2, 2, figure=figure, width_ratios=[1.15, 1.0], hspace=0.34, wspace=0.28)
    axis_a = figure.add_subplot(grid[:, 0])
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("black")
    image = axis_a.imshow(
        pattern,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=LogNorm(vmin=1.0e-8, vmax=1.0),
        interpolation="nearest",
    )
    axis_a.set_title("(a) WP-MS CBED pattern", loc="left", fontweight="bold")
    axis_a.set_xlabel(r"$\theta_x$ (mrad)")
    axis_a.set_ylabel(r"$\theta_y$ (mrad)")
    axis_a.set_aspect("equal")
    colourbar = figure.colorbar(image, ax=axis_a, fraction=0.046, pad=0.04)
    colourbar.set_label("Normalised CBED intensity (log)")

    axis_b = figure.add_subplot(grid[0, 1])
    for name in selected:
        axis_b.semilogy(
            centers,
            np.maximum(profiles[name], 1.0e-18),
            color=colours[name],
            linestyle=linestyles[name],
            linewidth=1.5,
            label=display_names[name],
        )
        values = np.interp(peaks[name], centers, profiles[name])
        axis_b.scatter(
            peaks[name], values, marker="o", s=25, facecolors="none", edgecolors=colours[name]
        )
    axis_b.set_title("(b) Radial profiles and detected peaks", loc="left", fontweight="bold")
    axis_b.set_xlim(0, 300)
    axis_b.set_ylim(1.0e-10, 1.0)
    axis_b.set_xlabel("Scattering angle (mrad)")
    axis_b.set_ylabel("Annular intensity / total intensity")
    axis_b.grid(True, alpha=0.25)
    axis_b.legend(frameon=False, fontsize=8)

    axis_c = figure.add_subplot(grid[1, 1])
    axis_c.scatter(wp_angles, wp_shifts, color="C2", s=28, label="WP-MS − AS-MS")
    axis_c.scatter(f_angles, f_shifts, color="C0", s=28, label="F-MS − AS-MS")
    if len(f_angles) >= 2:
        theory_angles = np.linspace(0.0, 300.0, 500)
        axis_c.plot(
            theory_angles,
            scale * dispersion_error(theory_angles, wavelength),
            "k--",
            linewidth=1.3,
            label="scaled |paraxial − exact dispersion|",
        )
    axis_c.axhline(0.0, color="0.5", linewidth=0.8)
    axis_c.set_title("(c) Peak displacement", loc="left", fontweight="bold")
    axis_c.set_xlim(0, 300)
    axis_c.set_xlabel("AS-MS peak angle (mrad)")
    axis_c.set_ylabel(r"Peak shift $\Delta\theta$ (mrad)")
    axis_c.grid(True, alpha=0.25)
    axis_c.legend(frameon=False, fontsize=8)

    figure.suptitle(
        f"Si [111] CBED — {metadata['actual_thickness_A']:.1f} Å at "
        f"{metadata['energy_eV']/1e3:.0f} keV; {metadata['probe_semiangle_mrad']:.0f} mrad probe",
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.09)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    thickness_tag = int(round(metadata["actual_thickness_A"]))
    base = args.output_dir / f"cbed_kirkland_si111_combined_300mrad_{thickness_tag}A"
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    figure.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved -> {base.with_suffix('.pdf')}")

    print("Final convergence diagnostics")
    for index, name in enumerate(method_names):
        print(
            f"{name:12s} amplitude_rRMSE_300={amplitude_rrmse[index, -1]:.6e} "
            f"norm={norm_ratio[index, -1]:.8f}"
        )


if __name__ == "__main__":
    main()
