"""Run the publication Au [100] CBED and WP-MS bin-convergence sweep.

The script records angle-limited amplitude differences, phase-aligned complex
exit-wave differences, discrete wave-norm drift, timings, and final CBED
patterns.  It deliberately does not renormalise any propagated wave.
"""

from __future__ import annotations

import os

# Set allocator controls before importing CuPy, JAX, or abTEM.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".70")

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter


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
    diffraction_pattern_numpy,
    make_kirkland_probe,
    simulate_fresnel_as_exit_only,
    simulate_wpm_exit_only,
)
from wide_angle_propagation.propagation_methods import (
    angular_spectrum_propagation_kernel,
    electron_rest_energy,
    energy2wavelength,
    fresnel_propagation_kernel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--slices-per-cell", type=int, default=64)
    parser.add_argument("--gpts", type=int, nargs=2, default=[2048, 2048])
    parser.add_argument("--target-thickness-nm", type=float, default=100.0)
    parser.add_argument("--cutoffs-mrad", type=float, nargs="+", default=[50, 100, 200, 300])
    parser.add_argument("--energy-kev", type=float, default=200.0)
    parser.add_argument("--probe-mrad", type=float, default=5.0)
    parser.add_argument("--lateral-repeats", type=int, nargs=2, default=[9, 9])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def git_revision() -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return revision + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def radial_masks(shape, sampling, wavelength, cutoffs_mrad):
    fy = np.fft.fftfreq(shape[0], d=sampling[0])
    fx = np.fft.fftfreq(shape[1], d=sampling[1])
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    sine = wavelength * np.sqrt(fx_grid**2 + fy_grid**2)
    propagating = sine <= 1.0
    theta_mrad = 1.0e3 * np.arcsin(np.clip(sine, 0.0, 1.0))
    masks = np.stack([
        propagating & (theta_mrad <= cutoff)
        for cutoff in cutoffs_mrad
    ])
    return jnp.asarray(masks), theta_mrad


@jax.jit
def comparison_metrics(wave, reference, reference_amplitude, masks, entrance_norm):
    """Return band-limited amplitude rRMSE, complex error, and norm ratio."""
    amplitude = jnp.abs(jnp.fft.fft2(wave))
    squared_difference = (amplitude - reference_amplitude) ** 2
    mask_values = masks.astype(amplitude.dtype)
    numerator = jnp.sum(mask_values * squared_difference[None, ...], axis=(1, 2))
    denominator = jnp.sum(
        mask_values * reference_amplitude[None, ...] ** 2,
        axis=(1, 2),
    )
    amplitude_rrmse = jnp.sqrt(numerator / jnp.maximum(denominator, 1.0e-30))

    relative_phase = jnp.angle(jnp.vdot(reference, wave))
    aligned_wave = wave * jnp.exp(-1j * relative_phase)
    complex_error = jnp.linalg.norm(aligned_wave - reference) / jnp.maximum(
        jnp.linalg.norm(reference), 1.0e-30
    )
    norm_ratio = jnp.sum(jnp.abs(wave) ** 2) / entrance_norm
    return amplitude_rrmse, complex_error, norm_ratio


def main() -> None:
    args = parse_args()
    bin_counts = sorted(set(args.bins))
    if not bin_counts or bin_counts[0] < 2:
        raise ValueError("Every WP-MS bin count must be at least two")
    if args.slices_per_cell < 1:
        raise ValueError("slices-per-cell must be positive")

    abtem.config.set({"device": "gpu", "precision": "float64"})
    jax.config.update("jax_enable_x64", True)

    lattice_constant = 4.08
    energy = args.energy_kev * 1.0e3
    wavelength = float(energy2wavelength(energy))
    nx_repeat, ny_repeat = args.lateral_repeats
    shape = tuple(args.gpts)
    slice_thickness = lattice_constant / args.slices_per_cell
    n_cells = int(np.ceil(10.0 * args.target_thickness_nm / lattice_constant))
    thickness_nm = np.arange(1, n_cells + 1) * lattice_constant / 10.0

    atoms = bulk("Au", "fcc", a=lattice_constant, cubic=True)
    atoms.pbc = [True, True, True]
    atoms.info["thermal_sigma"] = 0.0
    atoms.arrays["thermal_sigma"] = np.zeros(len(atoms))
    supercell = atoms * (nx_repeat, ny_repeat, 1)

    print(
        f"Building finite-projection Lobato potential: {shape[0]}x{shape[1]}x"
        f"{args.slices_per_cell}, a={lattice_constant:.2f} A"
    )
    potential_abtem = abtem.Potential(
        supercell,
        gpts=shape,
        slice_thickness=slice_thickness,
        projection="finite",
        parametrization="lobato",
    )
    sampling = tuple(float(value) for value in potential_abtem.sampling)
    potential_cpu = cupy.asnumpy(potential_abtem.build(lazy=False).array)
    expected_shape = (args.slices_per_cell, *shape)
    if potential_cpu.shape != expected_shape:
        raise RuntimeError(
            f"Potential shape {potential_cpu.shape} does not match {expected_shape}"
        )
    potential_min_v = float(np.min(potential_cpu)) / slice_thickness
    potential_max_v = float(np.max(potential_cpu)) / slice_thickness
    rest_energy = electron_rest_energy()
    refractive_index_extrema = []
    for potential_value in (potential_min_v, potential_max_v):
        n_squared = (
            1.0
            + 2.0 * (energy + rest_energy) * potential_value
            / (energy * (energy + 2.0 * rest_energy))
            + potential_value**2 / (energy * (energy + 2.0 * rest_energy))
        )
        refractive_index_extrema.append(float(np.sqrt(n_squared)))
    print(
        f"Potential range={potential_min_v:.6g}..{potential_max_v:.6g} V; "
        f"KG index range={refractive_index_extrema[0]:.8f}.."
        f"{refractive_index_extrema[1]:.8f}"
    )
    potential = jnp.asarray(potential_cpu / slice_thickness, dtype=jnp.float64)
    del potential_abtem, potential_cpu
    cupy.get_default_memory_pool().free_all_blocks()

    probe = make_kirkland_probe(
        shape[0],
        shape[1],
        sampling,
        wavelength,
        args.probe_mrad,
        defocus=0.0,
        cs=0.0,
    )
    probe = jnp.asarray(probe, dtype=jnp.complex128)
    entrance_norm = jnp.sum(jnp.abs(probe) ** 2)

    fresnel_kernel = fresnel_propagation_kernel(
        *shape, sampling, z=slice_thickness, energy=energy
    )
    angular_kernel = angular_spectrum_propagation_kernel(
        *shape, sampling, z=slice_thickness, energy=energy
    )
    fresnel_kernel = jnp.asarray(fresnel_kernel)
    angular_kernel = jnp.asarray(angular_kernel)

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
    run_wpm = {
        bins: jax.jit(
            lambda pot, wave, bins=bins: simulate_wpm_exit_only(
                pot,
                wave,
                slice_thickness,
                energy,
                sampling,
                n_bins=bins,
                power_spacing=2.0,
                bin_batch_size=4,
            )
        )
        for bins in bin_counts
    }

    primary_bins = bin_counts[-1]
    method_names = ["F-MS", "AS-MS", *[f"WP-MS-{bins}" for bins in bin_counts]]
    reference_name = f"WP-MS-{primary_bins}"
    waves = {name: probe for name in method_names}
    n_methods = len(method_names)
    n_cutoffs = len(args.cutoffs_mrad)
    amplitude_rrmse = np.empty((n_methods, n_cells, n_cutoffs), dtype=np.float64)
    complex_error = np.empty((n_methods, n_cells), dtype=np.float64)
    norm_ratio = np.empty((n_methods, n_cells), dtype=np.float64)
    runtime_s = np.empty((n_methods, n_cells), dtype=np.float64)
    masks, theta_mrad = radial_masks(shape, sampling, wavelength, args.cutoffs_mrad)

    print(
        f"Propagating {n_cells} cells to {thickness_nm[-1]:.3f} nm; "
        f"WP-MS bins={bin_counts}; sampling={sampling} A"
    )
    for cell_index in range(n_cells):
        runners = {
            "F-MS": run_fresnel,
            "AS-MS": run_angular,
            **{f"WP-MS-{bins}": run_wpm[bins] for bins in bin_counts},
        }
        for method_index, name in enumerate(method_names):
            start = perf_counter()
            waves[name] = runners[name](potential, waves[name])
            jax.block_until_ready(waves[name])
            runtime_s[method_index, cell_index] = perf_counter() - start

        reference = waves[reference_name]
        reference_amplitude = jnp.abs(jnp.fft.fft2(reference))
        for method_index, name in enumerate(method_names):
            band_error, field_error, power = comparison_metrics(
                waves[name], reference, reference_amplitude, masks, entrance_norm
            )
            amplitude_rrmse[method_index, cell_index] = np.asarray(band_error)
            complex_error[method_index, cell_index] = float(field_error)
            norm_ratio[method_index, cell_index] = float(power)

        if cell_index == 0 or (cell_index + 1) % 10 == 0 or cell_index + 1 == n_cells:
            as_index = method_names.index("AS-MS")
            print(
                f"cell {cell_index + 1:3d}/{n_cells}: "
                f"AS/WP rRMSE <= {args.cutoffs_mrad[-1]:g} mrad = "
                f"{amplitude_rrmse[as_index, cell_index, -1]:.5f}; "
                f"WP norm = {norm_ratio[method_names.index(reference_name), cell_index]:.6f}"
            )

    final_patterns = np.stack([
        diffraction_pattern_numpy(waves[name]).astype(np.float32)
        for name in method_names
    ])

    output_path = args.output
    if output_path is None:
        output_path = (
            ROOT
            / "notebooks"
            / "cbed"
            / "results"
            / f"au100_cbed_convergence_s{args.slices_per_cell}_g{shape[0]}.npz"
        )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "description": "Au [100] CBED propagation and WP-MS convergence",
        "energy_eV": energy,
        "wavelength_A": wavelength,
        "lattice_constant_A": lattice_constant,
        "target_thickness_nm": args.target_thickness_nm,
        "actual_thickness_nm": float(thickness_nm[-1]),
        "n_unit_cells": n_cells,
        "lateral_repeats": [nx_repeat, ny_repeat],
        "gpts": list(shape),
        "sampling_A": list(sampling),
        "slices_per_cell": args.slices_per_cell,
        "slice_thickness_A": slice_thickness,
        "potential_min_V": potential_min_v,
        "potential_max_V": potential_max_v,
        "refractive_index_min": refractive_index_extrema[0],
        "refractive_index_max": refractive_index_extrema[1],
        "probe_semiangle_mrad": args.probe_mrad,
        "potential_parametrization": "lobato",
        "potential_projection": "finite",
        "index_model": "kg",
        "ms_phase_convention": "paraxial_n2",
        "wpm_bin_counts": bin_counts,
        "wpm_power_spacing": 2.0,
        "reference_method": reference_name,
        "metric": "CBED-amplitude relative L2 difference up to each angular cutoff",
        "git_revision": git_revision(),
        "jax_version": jax.__version__,
        "abtem_version": abtem.__version__,
        "precision": "float64 propagation; float32 stored final patterns",
    }
    np.savez_compressed(
        output_path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        method_names=np.asarray(method_names),
        reference_method=np.asarray(reference_name),
        thickness_nm=thickness_nm,
        cutoffs_mrad=np.asarray(args.cutoffs_mrad, dtype=float),
        amplitude_rrmse=amplitude_rrmse,
        phase_aligned_complex_error=complex_error,
        norm_ratio=norm_ratio,
        runtime_s=runtime_s,
        final_patterns=final_patterns,
        theta_minmax_mrad=np.asarray([theta_mrad.min(), theta_mrad.max()]),
    )
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()
