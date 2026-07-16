"""Run one Au ODE-benchmark slice discretisation for convergence checks."""

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
    simulate_fresnel_as_exit_only,
    simulate_wpm_exit_only,
)
from wide_angle_propagation.propagation_methods import (
    angular_spectrum_propagation_kernel,
    fresnel_propagation_kernel,
    simulate_kg_ode_full,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slices-per-cell", type=int, required=True)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--cells", type=int, default=100)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "notebooks" / "verification" / "results" / "au100_lobato_kg_ode_benchmark.npz",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def phase_aligned_relative_l2(values, reference) -> float:
    phase = np.angle(np.vdot(reference, values))
    return float(
        np.linalg.norm(values * np.exp(-1j * phase) - reference)
        / np.linalg.norm(reference)
    )


def main() -> None:
    args = parse_args()
    if args.slices_per_cell < 1 or args.bins < 2 or args.cells < 1:
        raise ValueError("slice, bin, and cell counts must be positive")
    if args.output is None:
        args.output = (
            ROOT
            / "notebooks"
            / "verification"
            / "results"
            / f"au100_ode_slice_convergence_s{args.slices_per_cell}.npz"
        )

    abtem.config.set({"device": "gpu", "precision": "float64"})
    jax.config.update("jax_enable_x64", True)

    lattice_constant = 4.08
    energy = 300.0e3
    shape = (128, 128)
    slice_thickness = lattice_constant / args.slices_per_cell
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

    fresnel_kernel = jnp.asarray(
        fresnel_propagation_kernel(
            *shape, sampling, z=slice_thickness, energy=energy
        )
    )
    angular_kernel = jnp.asarray(
        angular_spectrum_propagation_kernel(
            *shape, sampling, z=slice_thickness, energy=energy
        )
    )
    run_fresnel = jax.jit(
        lambda pot, wave: simulate_fresnel_as_exit_only(
            pot, wave, fresnel_kernel, slice_thickness, energy
        )
    )
    run_angular = jax.jit(
        lambda pot, wave: simulate_fresnel_as_exit_only(
            pot, wave, angular_kernel, slice_thickness, energy
        )
    )
    run_wpm = jax.jit(
        lambda pot, wave: simulate_wpm_exit_only(
            pot,
            wave,
            slice_thickness,
            energy,
            sampling,
            n_bins=args.bins,
            power_spacing=2.0,
        )
    )

    names = ("fresnel", "as", "wpm", "ode")
    waves = {name: jnp.ones(shape, dtype=jnp.complex128) for name in names}
    phi_ode = None
    beam_00 = {name: [1.0] for name in names}
    beam_028 = {name: [0.0] for name in names}

    for cell_index in range(args.cells):
        waves["fresnel"] = run_fresnel(potential, waves["fresnel"])
        waves["as"] = run_angular(potential, waves["as"])
        waves["wpm"] = run_wpm(potential, waves["wpm"])
        waves["ode"], phi_ode, _, _ = simulate_kg_ode_full(
            potential,
            waves["ode"],
            slice_thickness,
            energy,
            sampling,
            initial_phi=phi_ode,
            rtol=1.0e-8,
            atol=1.0e-10,
            save_wavefronts=False,
        )
        jax.block_until_ready(waves["ode"])

        for name in names:
            wave = np.asarray(waves[name])
            beam_00[name].append(beam_amplitude_normalized(wave, 0, 0))
            beam_028[name].append(beam_amplitude_normalized(wave, 0, 28))
        if cell_index == 0 or (cell_index + 1) % 10 == 0:
            print(
                f"slices={args.slices_per_cell}, cell={cell_index + 1:3d}/{args.cells}, "
                f"ODE [0,0]={beam_00['ode'][-1]:.6f}"
            )

    output = {
        "slices_per_cell": args.slices_per_cell,
        "slice_thickness_A": slice_thickness,
        "wpm_bins": args.bins,
        "n_cells": np.arange(args.cells + 1),
    }
    for name in names:
        output[f"{name}_beam_00"] = np.asarray(beam_00[name])
        output[f"{name}_beam_028"] = np.asarray(beam_028[name])
        output[f"{name}_exit_wave"] = np.asarray(waves[name])

    if args.baseline.exists():
        baseline_keys = {
            "fresnel": ("ms_00", "ms_target", "exit_fresnel"),
            "as": ("as_00", "as_target", "exit_as"),
            "wpm": ("wpm_00", "wpm_target", "exit_wpm"),
            "ode": ("ode_00", "ode_target", "exit_ode"),
        }
        with np.load(args.baseline, allow_pickle=False) as baseline:
            output["baseline_slices_per_cell"] = int(baseline["n_slices_per_cell"])
            for name, (key_00, key_028, key_wave) in baseline_keys.items():
                output[f"{name}_beam_00_rmse_vs_baseline"] = float(
                    np.sqrt(np.mean((output[f"{name}_beam_00"] - baseline[key_00][: args.cells + 1]) ** 2))
                )
                output[f"{name}_beam_028_rmse_vs_baseline"] = float(
                    np.sqrt(np.mean((output[f"{name}_beam_028"] - baseline[key_028][: args.cells + 1]) ** 2))
                )
                output[f"{name}_exit_phase_aligned_l2_vs_baseline"] = (
                    phase_aligned_relative_l2(output[f"{name}_exit_wave"], baseline[key_wave])
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(f"Saved -> {args.output}")
    for name in names:
        metric = f"{name}_exit_phase_aligned_l2_vs_baseline"
        if metric in output:
            print(f"{metric}={output[metric]:.8e}")


if __name__ == "__main__":
    main()
