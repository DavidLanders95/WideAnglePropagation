"""Benchmark the paper propagators on one representative Au [100] workload.

Potential generation, probe construction, and JAX compilation are deliberately
excluded from the reported wall times. Each timed call propagates the same
probe through one Au unit cell and is synchronized before the timer is stopped.
"""

from __future__ import annotations

import os

# Set allocator controls before importing GPU-backed scientific packages.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".70")

import argparse
from datetime import datetime, timezone
import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = (
    ROOT / "notebooks" / "cbed" / "results" / "au100_propagation_timing_g2048.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--bins", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--gpts", type=int, nargs=2, default=[2048, 2048])
    parser.add_argument("--slices-per-cell", type=int, default=64)
    parser.add_argument("--bin-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def summarize_timings(timings: dict[str, list[float]]) -> list[dict[str, float | str]]:
    """Return median, quartiles, and AS-MS-relative time for each method."""
    if "AS-MS" not in timings:
        raise ValueError("timings must contain an AS-MS baseline")

    arrays: dict[str, np.ndarray] = {}
    for name, values in timings.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must contain at least one one-dimensional timing")
        if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
            raise ValueError(f"{name} timings must be finite and positive")
        arrays[name] = array

    as_median = float(np.median(arrays["AS-MS"]))
    summary = []
    for name, values in arrays.items():
        q1, q3 = np.percentile(values, [25.0, 75.0])
        median = float(np.median(values))
        summary.append(
            {
                "method": name,
                "median_s": median,
                "q1_s": float(q1),
                "q3_s": float(q3),
                "relative_to_as": median / as_median,
            }
        )
    return summary


def make_result_payload(
    metadata: dict[str, Any], timings: dict[str, list[float]]
) -> dict[str, Any]:
    """Build the JSON-serializable benchmark result payload."""
    return {
        "metadata": metadata,
        "timings_s": {
            name: [float(value) for value in values]
            for name, values in timings.items()
        },
        "summary": summarize_timings(timings),
    }


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


def nvidia_smi_metadata() -> dict[str, str]:
    try:
        line = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
        name, memory_mib, driver = (part.strip() for part in line.split(",", 2))
        return {
            "nvidia_smi_name": name,
            "gpu_memory_MiB": memory_mib,
            "nvidia_driver_version": driver,
        }
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return {}


def validate_args(args: argparse.Namespace) -> list[int]:
    bins = sorted(set(args.bins))
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.warmups < 1:
        raise ValueError("warmups must be at least one to exclude compilation")
    if not bins or bins[0] < 2:
        raise ValueError("all WP-MS bin counts must be at least two")
    if len(args.gpts) != 2 or min(args.gpts) < 2:
        raise ValueError("gpts must contain two values of at least two")
    if args.slices_per_cell < 1:
        raise ValueError("slices-per-cell must be positive")
    if args.bin_batch_size < 1:
        raise ValueError("bin-batch-size must be positive")
    return bins


def main() -> None:
    args = parse_args()
    bins = validate_args(args)

    import abtem
    import cupy
    import jax
    import jax.numpy as jnp
    import jaxlib
    from ase.build import bulk

    from wide_angle_propagation.notebook_utils import (
        make_kirkland_probe,
        simulate_fresnel_as_exit_only,
        simulate_wpm_exit_only,
    )
    from wide_angle_propagation.propagation_methods import (
        angular_spectrum_propagation_kernel,
        energy2wavelength,
        fresnel_propagation_kernel,
    )

    abtem.config.set({"device": "gpu", "precision": "float64"})
    jax.config.update("jax_enable_x64", True)

    lattice_constant = 4.08
    energy = 200.0e3
    lateral_repeats = (9, 9)
    shape = tuple(args.gpts)
    slice_thickness = lattice_constant / args.slices_per_cell

    atoms = bulk("Au", "fcc", a=lattice_constant, cubic=True)
    atoms.pbc = [True, True, True]
    atoms.info["thermal_sigma"] = 0.0
    atoms.arrays["thermal_sigma"] = np.zeros(len(atoms))
    supercell = atoms * (*lateral_repeats, 1)

    print(
        f"Building Au [100] potential: shape={shape}, "
        f"slices={args.slices_per_cell}"
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
    potential = jnp.asarray(potential_cpu / slice_thickness, dtype=jnp.float64)
    del potential_abtem, potential_cpu
    cupy.get_default_memory_pool().free_all_blocks()

    wavelength = float(energy2wavelength(energy))
    probe = jnp.asarray(
        make_kirkland_probe(
            *shape,
            sampling,
            wavelength,
            semiangle_mrad=5.0,
            defocus=0.0,
            cs=0.0,
        ),
        dtype=jnp.complex128,
    )
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

    runners = {
        "F-MS": jax.jit(
            lambda pot, wave: simulate_fresnel_as_exit_only(
                pot, wave, fresnel_kernel, slice_thickness, energy
            )
        ),
        "AS-MS": jax.jit(
            lambda pot, wave: simulate_fresnel_as_exit_only(
                pot, wave, angular_kernel, slice_thickness, energy
            )
        ),
    }
    runners.update(
        {
            f"WP-MS-{count}": jax.jit(
                lambda pot, wave, count=count: simulate_wpm_exit_only(
                    pot,
                    wave,
                    slice_thickness,
                    energy,
                    sampling,
                    n_bins=count,
                    power_spacing=2.0,
                    bin_batch_size=args.bin_batch_size,
                )
            )
            for count in bins
        }
    )

    print(f"Warming up {len(runners)} methods ({args.warmups} call(s) each)")
    for name, runner in runners.items():
        for _ in range(args.warmups):
            jax.block_until_ready(runner(potential, probe))
        print(f"  ready: {name}")

    timings = {name: [] for name in runners}
    rng = random.Random(args.seed)
    method_names = list(runners)
    for repeat in range(args.repeats):
        order = method_names.copy()
        rng.shuffle(order)
        for name in order:
            start = perf_counter()
            result = runners[name](potential, probe)
            jax.block_until_ready(result)
            timings[name].append(perf_counter() - start)
        current = summarize_timings(timings)
        values = ", ".join(
            f"{row['method']}={row['median_s']:.4f} s" for row in current
        )
        print(f"repeat {repeat + 1:2d}/{args.repeats}: {values}")

    device = jax.devices()[0]
    metadata: dict[str, Any] = {
        "description": "Illustrative propagation timing on one Au [100] unit cell",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "energy_eV": energy,
        "lattice_constant_A": lattice_constant,
        "lateral_repeats": list(lateral_repeats),
        "gpts": list(shape),
        "sampling_A": list(sampling),
        "slices_per_cell": args.slices_per_cell,
        "slice_thickness_A": slice_thickness,
        "probe_semiangle_mrad": 5.0,
        "potential_parametrization": "lobato",
        "potential_projection": "finite",
        "precision": "float64",
        "wpm_bin_counts": bins,
        "wpm_power_spacing": 2.0,
        "wpm_bin_batch_size": args.bin_batch_size,
        "warmup_calls_per_method": args.warmups,
        "timed_calls_per_method": args.repeats,
        "timing_order_seed": args.seed,
        "timed_scope": (
            "propagation only; potential, probe, kernels, compilation, and "
            "analysis excluded"
        ),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "abtem_version": abtem.__version__,
        "jax_platform": device.platform,
        "jax_device_kind": device.device_kind,
    }
    metadata.update(nvidia_smi_metadata())
    payload = make_result_payload(metadata, timings)

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved -> {output_path}")
    for row in payload["summary"]:
        print(
            f"{row['method']:>11}: {row['median_s']:.6f} s "
            f"[{row['q1_s']:.6f}, {row['q3_s']:.6f}], "
            f"{row['relative_to_as']:.2f}x AS-MS"
        )


if __name__ == "__main__":
    main()
