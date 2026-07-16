"""Generate the two Au [100] CBED paper figures from convergence results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "Paper" / "figures" / "method_explainer_generations",
    )
    parser.add_argument("--evaluation-cutoff-mrad", type=float, default=300.0)
    parser.add_argument("--display-cutoff-mrad", type=float, default=50.0)
    return parser.parse_args()


def load_results(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "metadata": json.loads(str(data["metadata_json"])),
            "method_names": [str(value) for value in data["method_names"]],
            "reference_method": str(data["reference_method"]),
            "thickness_nm": data["thickness_nm"].copy(),
            "cutoffs_mrad": data["cutoffs_mrad"].copy(),
            "amplitude_rrmse": data["amplitude_rrmse"].copy(),
            "complex_error": data["phase_aligned_complex_error"].copy(),
            "norm_ratio": data["norm_ratio"].copy(),
            "runtime_s": data["runtime_s"].copy(),
            "final_patterns": data["final_patterns"].copy(),
        }


def angle_axes(shape, sampling, wavelength):
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0], d=sampling[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1], d=sampling[1]))
    return (
        1.0e3 * np.arcsin(np.clip(wavelength * fy, -1.0, 1.0)),
        1.0e3 * np.arcsin(np.clip(wavelength * fx, -1.0, 1.0)),
    )


def crop_indices(axis, cutoff):
    indices = np.flatnonzero(np.abs(axis) <= cutoff)
    if not len(indices):
        raise ValueError(f"No Fourier pixels fall inside {cutoff:g} mrad")
    return slice(indices[0], indices[-1] + 1)


def plot_thickness_diagnostics(results, output_dir, cutoff):
    methods = results["method_names"]
    thickness = results["thickness_nm"]
    cutoff_index = int(np.argmin(np.abs(results["cutoffs_mrad"] - cutoff)))

    colours = {"F-MS": "C0", "AS-MS": "C1"}
    fig, axis = plt.subplots(figsize=(6.4, 4.3))

    for name in ("F-MS", "AS-MS"):
        index = methods.index(name)
        axis.semilogy(
            thickness,
            results["amplitude_rrmse"][index, :, cutoff_index],
            label=name,
            color=colours[name],
            linewidth=1.8,
        )
    axis.set_xlabel("Thickness (nm)")
    axis.set_ylabel("Relative CBED-amplitude difference")
    axis.set_title("CBED-amplitude difference from WP-MS")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "cbed_error_vs_thickness.pdf"
    png_path = output_dir / "cbed_error_vs_thickness.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved -> {pdf_path}")
    plt.close(fig)


def plot_difference_maps(results, output_dir, display_cutoff):
    methods = results["method_names"]
    reference = results["reference_method"]
    metadata = results["metadata"]
    patterns = {
        name: results["final_patterns"][index]
        for index, name in enumerate(methods)
    }
    shape = patterns[reference].shape
    theta_y, theta_x = angle_axes(
        shape,
        metadata["sampling_A"],
        metadata["wavelength_A"],
    )
    y_slice = crop_indices(theta_y, display_cutoff)
    x_slice = crop_indices(theta_x, display_cutoff)
    extent = [
        theta_x[x_slice.start],
        theta_x[x_slice.stop - 1],
        theta_y[y_slice.start],
        theta_y[y_slice.stop - 1],
    ]

    amplitudes = {
        name: np.sqrt(np.maximum(pattern[y_slice, x_slice], 0.0))
        for name, pattern in patterns.items()
    }
    differences = {
        name: 100.0 * np.abs(amplitudes[name] - amplitudes[reference])
        / np.maximum(amplitudes[reference], 1.0e-12)
        for name in ("AS-MS", "F-MS")
    }
    amplitude_norm = Normalize(vmin=0.005, vmax=0.03, clip=True)
    difference_floor_pct = 0.1
    difference_norm = LogNorm(vmin=difference_floor_pct, vmax=800.0)
    difference_cmap = plt.get_cmap("turbo").copy()
    difference_cmap.set_under("white")
    difference_cmap.set_over("white")

    fig = plt.figure(figsize=(8.8, 6.6))
    grid = GridSpec(
        2,
        7,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.09],
        hspace=0.36,
        wspace=0.30,
    )
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[0, 4:6]),
        fig.add_subplot(grid[1, 1:3]),
        fig.add_subplot(grid[1, 3:5]),
    ]
    panels = [
        ("WP-MS reference", amplitudes[reference], "amplitude"),
        ("AS-MS", amplitudes["AS-MS"], "amplitude"),
        ("F-MS", amplitudes["F-MS"], "amplitude"),
        ("AS-MS error", differences["AS-MS"], "difference"),
        ("F-MS error", differences["F-MS"], "difference"),
    ]
    amplitude_image = difference_image = None
    for panel_index, (method, values, kind) in enumerate(panels):
        axis = axes[panel_index]
        if kind == "amplitude":
            amplitude_image = axis.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap="magma",
                norm=amplitude_norm,
            )
        else:
            difference_image = axis.imshow(
                np.maximum(values, difference_floor_pct),
                origin="lower",
                extent=extent,
                cmap=difference_cmap,
                norm=difference_norm,
            )
        axis.set_title(
            f"({chr(ord('a') + panel_index)}) {method}",
            fontsize=10,
            fontweight="bold",
        )
        axis.set_xlabel(r"$\theta_x$ (mrad)")
        axis.set_xticks([-50.0, 0.0, 50.0], ["-50", "0", "50"])
        if panel_index in (0, 3):
            axis.set_ylabel(r"$\theta_y$ (mrad)")
            axis.set_yticks([-50.0, 0.0, 50.0], ["-50", "0", "50"])
        else:
            axis.set_yticks([])
        axis.tick_params(labelsize=6, length=2, width=0.5)
        axis.set_aspect("equal")

    amplitude_cax = fig.add_subplot(grid[0, 6])
    amplitude_bar = fig.colorbar(amplitude_image, cax=amplitude_cax)
    amplitude_bar.set_label("CBED amplitude (a.u.)", fontsize=9, labelpad=4)
    amplitude_bar.set_ticks([0.005, 0.01, 0.02, 0.03])
    amplitude_bar.ax.set_yticklabels(["0.005", "0.01", "0.02", "0.03"])
    amplitude_bar.ax.tick_params(labelsize=8)
    difference_cax = fig.add_subplot(grid[1, 6])
    difference_bar = fig.colorbar(difference_image, cax=difference_cax)
    difference_bar.set_label("Amplitude error vs WP-MS (%; log scale)", fontsize=9, labelpad=4)
    difference_bar.set_ticks([0.1, 1.0, 10.0, 100.0, 800.0])
    difference_bar.ax.set_yticklabels(["0.1", "1", "10", "100", "800"])
    difference_bar.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.95)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "cbed_combined_amplitude_log_error_maps.pdf"
    png_path = output_dir / "cbed_combined_amplitude_log_error_maps.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"Saved -> {pdf_path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = load_results(args.results.resolve())
    plot_thickness_diagnostics(results, args.output_dir, args.evaluation_cutoff_mrad)
    plot_difference_maps(results, args.output_dir, args.display_cutoff_mrad)


if __name__ == "__main__":
    main()
