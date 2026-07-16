"""Check full-thickness WP-MS bin convergence for the Au ODE benchmark."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".10")

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import abtem
import cupy
import jax
import jax.numpy as jnp
import numpy as np
from ase.build import bulk

from wide_angle_propagation.notebook_utils import (
    beam_amplitude_normalized,
    simulate_wpm_exit_only,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=512)
    parser.add_argument("--slices-per-cell", type=int, default=256)
    parser.add_argument("--cells", type=int, default=100)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "notebooks" / "verification" / "results" / "au100_lobato_kg_ode_benchmark.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "notebooks" / "verification" / "results" / "au100_wpm_bin_convergence.npz",
    )
    return parser.parse_args()


def phase_aligned_relative_l2(values, reference) -> float:
    phase = np.angle(np.vdot(reference, values))
    return float(
        np.linalg.norm(values * np.exp(-1j * phase) - reference)
        / np.linalg.norm(reference)
    )


def main() -> None:
    args = parse_args()
    if args.bins < 2 or args.cells < 1:
        raise ValueError("bins must be at least two and cells must be positive")

    abtem.config.set({"device": "gpu", "precision": "float64"})
    jax.config.update("jax_enable_x64", True)

    lattice_constant = 4.08
    energy = 300.0e3
    slices_per_cell = args.slices_per_cell
    shape = (128, 128)
    slice_thickness = lattice_constant / slices_per_cell
    atoms = bulk("Au", "fcc", a=lattice_constant, cubic=True)
    atoms.pbc = [True, True, True]
    atoms.info["thermal_sigma"] = 0.0
    atoms.arrays["thermal_sigma"] = np.zeros(len(atoms))
    potential_abtem = abtem.Potential(
        atoms,
        gpts=shape,
        slice_thickness=slice_thickness,
        projection="finite",
        parametrization="lobato",
    )
    sampling = tuple(float(value) for value in potential_abtem.sampling)
    potential = jnp.asarray(
        cupy.asnumpy(potential_abtem.build(lazy=False).array) / slice_thickness,
        dtype=jnp.float64,
    )
    del potential_abtem
    cupy.get_default_memory_pool().free_all_blocks()

    wave = jnp.ones(shape, dtype=jnp.complex128)
    entrance_norm = float(jnp.sum(jnp.abs(wave) ** 2))
    run_cell = jax.jit(
        lambda pot, current: simulate_wpm_exit_only(
            pot,
            current,
            slice_thickness,
            energy,
            sampling,
            n_bins=args.bins,
            power_spacing=2.0,
        )
    )

    beam_00 = [beam_amplitude_normalized(np.asarray(wave), 0, 0)]
    beam_028 = [beam_amplitude_normalized(np.asarray(wave), 0, 28)]
    norm_ratio = [1.0]
    for cell_index in range(args.cells):
        wave = run_cell(potential, wave)
        jax.block_until_ready(wave)
        wave_numpy = np.asarray(wave)
        beam_00.append(beam_amplitude_normalized(wave_numpy, 0, 0))
        beam_028.append(beam_amplitude_normalized(wave_numpy, 0, 28))
        norm_ratio.append(float(np.sum(np.abs(wave_numpy) ** 2) / entrance_norm))
        if cell_index == 0 or (cell_index + 1) % 10 == 0:
            print(
                f"bins={args.bins}, cell={cell_index + 1:3d}/{args.cells}, "
                f"norm={norm_ratio[-1]:.8f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "bins": args.bins,
        "slices_per_cell": args.slices_per_cell,
        "n_cells": np.arange(args.cells + 1),
        "beam_00": np.asarray(beam_00),
        "beam_028": np.asarray(beam_028),
        "norm_ratio": np.asarray(norm_ratio),
        "exit_wave": np.asarray(wave),
    }

    if args.baseline.exists():
        with np.load(args.baseline, allow_pickle=False) as baseline:
            baseline_wave = baseline["exit_wpm"]
            output["baseline_bins"] = int(baseline["wpm_n_bins"])
            output["exit_phase_aligned_relative_l2_vs_baseline"] = (
                phase_aligned_relative_l2(output["exit_wave"], baseline_wave)
            )
            output["beam_00_rmse_vs_baseline"] = float(
                np.sqrt(np.mean((output["beam_00"] - baseline["wpm_00"][: args.cells + 1]) ** 2))
            )
            output["beam_028_rmse_vs_baseline"] = float(
                np.sqrt(np.mean((output["beam_028"] - baseline["wpm_target"][: args.cells + 1]) ** 2))
            )

    np.savez_compressed(args.output, **output)
    print(f"Saved -> {args.output}")
    for name in (
        "exit_phase_aligned_relative_l2_vs_baseline",
        "beam_00_rmse_vs_baseline",
        "beam_028_rmse_vs_baseline",
    ):
        if name in output:
            print(f"{name}={output[name]:.8e}")


if __name__ == "__main__":
    main()
