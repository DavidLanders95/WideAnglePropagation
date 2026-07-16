"""Run the Si [111] CBED calculation with full-thickness WP-MS convergence."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".70")

import argparse
import json
import subprocess
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
    diffraction_pattern_numpy,
    make_kirkland_probe,
    simulate_fresnel_as_exit_only,
    simulate_wpm_exit_only,
)
from wide_angle_propagation.propagation_methods import (
    angular_spectrum_propagation_kernel,
    energy2wavelength,
    fresnel_propagation_kernel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--target-thickness-a", type=float, default=1000.0)
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


@jax.jit
def scalar_metrics(wave, reference, reference_amplitude, mask, entrance_norm):
    amplitude = jnp.abs(jnp.fft.fft2(wave))
    mask_values = mask.astype(amplitude.dtype)
    amplitude_rrmse = jnp.sqrt(
        jnp.sum(mask_values * (amplitude - reference_amplitude) ** 2)
        / jnp.maximum(jnp.sum(mask_values * reference_amplitude**2), 1.0e-30)
    )
    relative_phase = jnp.angle(jnp.vdot(reference, wave))
    aligned = wave * jnp.exp(-1j * relative_phase)
    complex_error = jnp.linalg.norm(aligned - reference) / jnp.maximum(
        jnp.linalg.norm(reference), 1.0e-30
    )
    norm_ratio = jnp.sum(jnp.abs(wave) ** 2) / entrance_norm
    return amplitude_rrmse, complex_error, norm_ratio


def main() -> None:
    args = parse_args()
    bins = sorted(set(args.bins))
    if not bins or bins[0] < 2:
        raise ValueError("WP-MS bin counts must be at least two")

    abtem.config.set({"device": "gpu", "precision": "float64"})
    jax.config.update("jax_enable_x64", True)

    energy = 100.0e3
    wavelength = float(energy2wavelength(energy))
    lattice_constant = 5.431
    slices_per_repeat = 32
    sampling_requested = 0.05
    max_angle_mrad = 300.0

    silicon = bulk("Si", "diamond", a=lattice_constant, cubic=True)
    silicon.rotate((0, 0, 1), (1, 1, 1), rotate_cell=True)
    silicon.rotate(45, "z", rotate_cell=True)
    unit_cell = abtem.orthogonalize_cell(silicon)
    unit_cell.pbc = [True, True, True]
    unit_cell.wrap()
    supercell = unit_cell * (12, 7, 1)
    repeat_thickness = float(unit_cell.cell[2, 2])
    slice_thickness = repeat_thickness / slices_per_repeat
    n_repeats = max(1, int(np.rint(args.target_thickness_a / repeat_thickness)))
    actual_thickness_a = n_repeats * repeat_thickness

    potential_abtem = abtem.Potential(
        supercell,
        sampling=sampling_requested,
        slice_thickness=slice_thickness,
        projection="finite",
        parametrization="lobato",
    )
    shape = tuple(int(value) for value in potential_abtem.gpts)
    sampling = tuple(float(value) for value in potential_abtem.sampling)
    print(
        f"Building Si [111] potential: shape={shape}, slices={slices_per_repeat}, "
        f"sampling={sampling} A"
    )
    potential_cpu = cupy.asnumpy(potential_abtem.build(lazy=False).array)
    expected_shape = (slices_per_repeat, *shape)
    if potential_cpu.shape != expected_shape:
        raise RuntimeError(
            f"Potential shape {potential_cpu.shape} does not match {expected_shape}"
        )
    potential = jnp.asarray(potential_cpu / slice_thickness, dtype=jnp.float64)
    del potential_abtem, potential_cpu
    cupy.get_default_memory_pool().free_all_blocks()

    probe = jnp.asarray(
        make_kirkland_probe(
            *shape,
            sampling,
            wavelength,
            semiangle_mrad=8.0,
            defocus=0.0,
            cs=0.0,
        ),
        dtype=jnp.complex128,
    )
    entrance_norm = jnp.sum(jnp.abs(probe) ** 2)
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
    run_wpm = {
        count: jax.jit(
            lambda pot, wave, count=count: simulate_wpm_exit_only(
                pot,
                wave,
                slice_thickness,
                energy,
                sampling,
                n_bins=count,
                power_spacing=2.0,
                bin_batch_size=4,
            )
        )
        for count in bins
    }

    fy = np.fft.fftfreq(shape[0], d=sampling[0])
    fx = np.fft.fftfreq(shape[1], d=sampling[1])
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    radial_sine = wavelength * np.sqrt(fx_grid**2 + fy_grid**2)
    mask_300 = jnp.asarray(radial_sine <= np.sin(max_angle_mrad * 1.0e-3))

    method_names = ["F-MS", "AS-MS", *[f"WP-MS-{count}" for count in bins]]
    primary_name = f"WP-MS-{bins[-1]}"
    waves = {name: probe for name in method_names}
    amplitude_rrmse = np.empty((len(method_names), n_repeats), dtype=np.float64)
    complex_error = np.empty((len(method_names), n_repeats), dtype=np.float64)
    norm_ratio = np.empty((len(method_names), n_repeats), dtype=np.float64)

    print(
        f"Propagating {n_repeats} repeats to {actual_thickness_a / 10.0:.3f} nm; "
        f"WP-MS bins={bins}"
    )
    for repeat_index in range(n_repeats):
        waves["F-MS"] = run_fresnel(potential, waves["F-MS"])
        waves["AS-MS"] = run_angular(potential, waves["AS-MS"])
        for count in bins:
            name = f"WP-MS-{count}"
            waves[name] = run_wpm[count](potential, waves[name])
        jax.block_until_ready(waves[primary_name])

        reference = waves[primary_name]
        reference_amplitude = jnp.abs(jnp.fft.fft2(reference))
        for method_index, name in enumerate(method_names):
            metrics = scalar_metrics(
                waves[name], reference, reference_amplitude, mask_300, entrance_norm
            )
            amplitude_rrmse[method_index, repeat_index] = float(metrics[0])
            complex_error[method_index, repeat_index] = float(metrics[1])
            norm_ratio[method_index, repeat_index] = float(metrics[2])
        if (
            repeat_index == 0
            or (repeat_index + 1) % 10 == 0
            or repeat_index + 1 == n_repeats
        ):
            print(
                f"repeat {repeat_index + 1:3d}/{n_repeats}: "
                f"AS/WP amplitude rRMSE={amplitude_rrmse[1, repeat_index]:.6f}; "
                f"WP norm={norm_ratio[method_names.index(primary_name), repeat_index]:.6f}"
            )

    final_patterns = np.stack(
        [
            diffraction_pattern_numpy(waves[name]).astype(np.float32)
            for name in method_names
        ]
    )
    metadata = {
        "description": "Si [111] CBED and full-thickness WP-MS convergence",
        "energy_eV": energy,
        "wavelength_A": wavelength,
        "lattice_constant_A": lattice_constant,
        "gpts": list(shape),
        "sampling_A": list(sampling),
        "lateral_repeats": [12, 7],
        "repeat_thickness_A": repeat_thickness,
        "slices_per_repeat": slices_per_repeat,
        "slice_thickness_A": slice_thickness,
        "n_repeats": n_repeats,
        "actual_thickness_A": actual_thickness_a,
        "probe_semiangle_mrad": 8.0,
        "analysis_cutoff_mrad": max_angle_mrad,
        "potential_parametrization": "lobato",
        "potential_projection": "finite",
        "index_model": "kg",
        "ms_phase_convention": "paraxial_n2",
        "wpm_bin_counts": bins,
        "wpm_power_spacing": 2.0,
        "reference_method": primary_name,
        "git_revision": git_revision(),
        "precision": "float64 propagation; float32 stored final patterns",
        "abtem_version": abtem.__version__,
        "jax_version": jax.__version__,
    }

    if args.output is None:
        args.output = (
            ROOT / "notebooks" / "cbed" / "results" / "si111_cbed_convergence.npz"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        method_names=np.asarray(method_names),
        reference_method=np.asarray(primary_name),
        repeat_index=np.arange(1, n_repeats + 1),
        thickness_A=np.arange(1, n_repeats + 1) * repeat_thickness,
        amplitude_rrmse_300mrad=amplitude_rrmse,
        phase_aligned_complex_error=complex_error,
        norm_ratio=norm_ratio,
        final_patterns=final_patterns,
    )
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
