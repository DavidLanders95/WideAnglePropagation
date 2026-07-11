#!/usr/bin/env python3
"""Benchmark the reusable 1D lattice-site ptychography runtime.

The default problem is exactly ``SiliconGlancingConfig1D()``--the geometry
used by the maintained side-view notebook.  ``--quick`` selects a much smaller
problem for installation checks and benchmark-harness development; it is not a
scientific performance reference.

JAX is imported only after the command line has been parsed and the allocator
environment has been configured.  This is intentional: XLA allocator policy
is read during backend initialization and cannot be changed reliably later.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "wide_angle_propagation.ptychography_performance_benchmark"
REPORT_SCHEMA_VERSION = 1


def _positive_integer(value: str) -> int:
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return integer


def _nonnegative_integer(value: str) -> int:
    integer = int(value)
    if integer < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return integer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Time the prepared lattice-site reconstruction on the maintained "
            "notebook geometry, or on a reduced --quick geometry."
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "use a reduced CPU-friendly geometry (not comparable with the "
            "notebook reference geometry)"
        ),
    )
    parser.add_argument(
        "--updates",
        type=_positive_integer,
        default=500,
        help="optimizer updates per start (default: 500)",
    )
    parser.add_argument(
        "--starts",
        type=_positive_integer,
        default=5,
        help="independent prepared-runtime starts (notebook default: 5)",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float64",
        help="requested JAX real precision (default: float64)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="JAX platform selection made before JAX import (default: auto)",
    )
    parser.add_argument(
        "--case",
        choices=("vacancy", "vacancy_plus_strain", "strained_surface_defects"),
        default="strained_surface_defects",
        help="synthetic specimen used to generate the diffraction data",
    )
    parser.add_argument(
        "--validation-interval",
        type=_positive_integer,
        default=25,
        help="updates between synchronized train/validation evaluations",
    )
    parser.add_argument(
        "--training-diagnostic-scans",
        type=_positive_integer,
        default=32,
        help=(
            "fixed geometry-stratified training scans used only for loss "
            "history when validation exists (default: 32)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_integer,
        default=0,
        help="base seed for antithetic active-site translations and minibatches",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="write JSON to PATH; omit it or use '-' to write JSON to stdout",
    )
    return parser


def _configure_environment(*, precision: str, device: str) -> dict[str, str]:
    """Set documented JAX/XLA environment controls before importing JAX."""
    if "jax" in sys.modules:
        raise RuntimeError(
            "JAX was imported before benchmark environment configuration; run "
            "this file as a fresh process"
        )
    # Documented JAX GPU-memory control.  Do not replace this with a post-import
    # config mutation: the allocator is normally initialized with the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_ENABLE_X64"] = "true" if precision == "float64" else "false"
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device == "gpu":
        # JAX's platform name is CUDA even though default_backend() reports GPU.
        os.environ["JAX_PLATFORMS"] = "cuda"
    return {
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ[
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ],
        "JAX_ENABLE_X64": os.environ["JAX_ENABLE_X64"],
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "<unset>"),
    }


def _quick_config(workflow: Any) -> Any:
    """Return a tiny geometry that still exercises the complete workflow."""
    return workflow.SiliconGlancingConfig1D(
        slab_depth_A=6.0,
        vacuum_above_A=9.0,
        vacuum_below_A=9.0,
        window_length_A=20.0,
        scan_start_A=6.0,
        scan_stop_A=14.0,
        n_scans=6,
        defect_center_s_A=10.0,
        defect_width_sites=1,
        validation_stride=3,
        audit_fraction=1.0 / 6.0,
        audit_blocks=1,
        atomic_template_cutoff_A=8.0,
        displacement_control_spacing_A=5.0,
        displacement_control_spacing_u_A=3.0,
    )


def _block_experiment(jax: Any, experiment: Any) -> None:
    arrays = [
        experiment.pristine_potential,
        experiment.input_probes,
        experiment.propagation_kernel,
        experiment.reconstruction_mask,
        experiment.lattice_influence_mask,
        experiment.lattice_model.reference_potential,
        experiment.lattice_model.site_patches,
        *experiment.truth_potentials.values(),
    ]
    jax.block_until_ready(tuple(arrays))


def _block_dataset(jax: Any, dataset: Any) -> None:
    jax.block_until_ready(
        (
            dataset.potential,
            dataset.intensities,
            dataset.truth_vacancy_fractions,
            dataset.truth_displacement_controls,
            dataset.truth_rigid_displacement,
        )
    )


def _block_prepared(jax: Any, prepared: Any) -> None:
    # Preparation already eagerly compiles and synchronizes its three
    # executables.  This additional fence makes the outer wall-time contract
    # explicit and protects the benchmark if implementation details change.
    jax.block_until_ready(
        (
            prepared.model.reference_potential,
            prepared.model.site_patches,
            prepared.input_probe,
            prepared.measured_intensities,
            prepared.propagation_kernel,
        )
    )


def _block_result(jax: Any, result: Any) -> None:
    jax.block_until_ready(
        (
            result.potential,
            result.vacancy_fractions,
            result.displacement_controls,
            result.rigid_displacement,
            result.predicted_intensities,
            result.training_loss_history,
            result.validation_loss_history,
        )
    )


def _memory_value(stats: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = stats.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError, OverflowError):
                return None
    return None


def _device_memory_snapshot(jax: Any) -> dict[str, Any]:
    """Read allocator statistics without claiming unsupported CPU metrics."""
    records = []
    current_values: list[int] = []
    peak_values: list[int] = []
    for device in jax.devices():
        statistics_error = None
        try:
            raw_stats = device.memory_stats()
        except Exception as exc:  # backend-specific implementations vary
            raw_stats = None
            statistics_error = f"{type(exc).__name__}: {exc}"
        stats = raw_stats if isinstance(raw_stats, Mapping) else {}
        current = _memory_value(
            stats,
            "bytes_in_use",
            "current_bytes_in_use",
            "allocated_bytes",
        )
        peak = _memory_value(
            stats,
            "peak_bytes_in_use",
            "peak_allocated_bytes",
        )
        limit = _memory_value(stats, "bytes_limit", "memory_limit")
        if current is not None:
            current_values.append(current)
        if peak is not None:
            peak_values.append(peak)
        records.append(
            {
                "device": str(device),
                "current_bytes": current,
                "peak_bytes": peak,
                "limit_bytes": limit,
                "statistics_available": bool(stats),
                "statistics_error": statistics_error,
            }
        )
    return {
        "devices": records,
        "aggregate_current_bytes": (
            sum(current_values) if len(current_values) == len(records) else None
        ),
        "aggregate_peak_bytes": (
            sum(peak_values) if len(peak_values) == len(records) else None
        ),
        "scope": (
            "allocator statistics reported by each JAX device; peak values are "
            "process-cumulative and are not reset between benchmark phases"
        ),
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_provenance() -> dict[str, Any]:
    def command(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = command("status", "--porcelain")
    return {
        "commit": command("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _json_value(value: Any) -> Any:
    """Convert NumPy/JAX scalar-like metadata to strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    return _json_value(converted)


def _write_report(report: Mapping[str, Any], output: str | None) -> None:
    payload = json.dumps(_json_value(report), indent=2, sort_keys=True, allow_nan=False)
    if output is None or output == "-":
        print(payload)
        return
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"wrote {path}", file=sys.stderr, flush=True)


def _status(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cli_start = perf_counter()
    environment = _configure_environment(
        precision=args.precision,
        device=args.device,
    )

    runtime_import_start = perf_counter()
    import jax
    import numpy as np

    # These imports deliberately occur after allocator/platform/precision
    # configuration.  Importing a submodule first still executes package
    # ``__init__``, which imports JAX-backed modules.
    from wide_angle_propagation import ptychography_workflow_1d as workflow
    from wide_angle_propagation.ptychography_1d import (
        ConvergenceOptions1D,
        LatticeOptimizationOptions1D,
        prepare_lattice_site_reconstruction_1d,
        run_prepared_lattice_site_reconstruction_1d,
    )
    from wide_angle_propagation.ptychography_ensemble_1d import (
        MultistartOptions1D,
        multistart_site_translation_offsets_1d,
    )

    devices = jax.devices()
    backend = jax.default_backend()
    runtime_import_time = perf_counter() - runtime_import_start
    if args.device == "cpu" and backend != "cpu":
        raise RuntimeError(f"requested CPU but JAX initialized backend {backend!r}")
    if args.device == "gpu" and backend not in {"gpu", "cuda"}:
        raise RuntimeError(f"requested GPU but JAX initialized backend {backend!r}")
    if bool(jax.config.x64_enabled) != (args.precision == "float64"):
        raise RuntimeError("JAX precision policy differs from the requested precision")

    mode = "quick" if args.quick else "notebook"
    config = (
        _quick_config(workflow)
        if args.quick
        else workflow.SiliconGlancingConfig1D()
    )
    minibatch_size = 2 if args.quick else 5
    evaluation_batch_size = 3 if args.quick else 10

    _status(f"[{mode}] building experiment geometry and lattice model")
    workload_start = perf_counter()
    phase_start = perf_counter()
    experiment = workflow.build_silicon_glancing_experiment_1d(config)
    _block_experiment(jax, experiment)
    build_time = perf_counter() - phase_start
    memory_after_build = _device_memory_snapshot(jax)

    _status(f"[{mode}] simulating {args.case!r} diffraction data once")
    phase_start = perf_counter()
    dataset = workflow.simulate_experiment_1d(
        experiment,
        case=args.case,
        batch_size=evaluation_batch_size,
    )
    _block_dataset(jax, dataset)
    simulation_time = perf_counter() - phase_start
    memory_after_simulation = _device_memory_snapshot(jax)

    _status(f"[{mode}] preparing and eagerly compiling reusable executables once")
    phase_start = perf_counter()
    pristine = np.asarray(experiment.pristine_potential)
    positive = pristine[pristine > 0.0]
    if not positive.size:
        raise RuntimeError("the workflow produced no positive specimen potential")
    potential_max = 2.0 * float(np.max(positive))
    prepared = prepare_lattice_site_reconstruction_1d(
        experiment.lattice_model,
        experiment.input_probes,
        experiment.window_starts,
        experiment.window_length,
        experiment.propagation_kernel,
        experiment.axial_sampling,
        experiment.config.energy_eV,
        dataset.intensities,
        separate_rigid_registration=True,
        maximum_rigid_displacement=0.15,
        maximum_residual_displacement=0.35,
        scan_coordinates=experiment.scan_coordinates,
        detector_angles=experiment.detector_angles,
        validation_indices=np.asarray(experiment.validation_indices),
        audit_indices=np.asarray(experiment.audit_indices),
        excluded_indices=np.asarray(experiment.guard_indices),
        potential_max=potential_max,
        minibatch_size=minibatch_size,
        evaluation_batch_size=evaluation_batch_size,
        rematerialize=True,
    )
    _block_prepared(jax, prepared)
    preparation_wall_time = perf_counter() - phase_start
    memory_after_preparation = _device_memory_snapshot(jax)

    translation_limit = float(prepared.maximum_rigid_displacement)
    offset_options = MultistartOptions1D(
        n_starts=args.starts,
        base_seed=args.seed,
        initial_translation_half_width_A=(translation_limit, translation_limit),
        minimum_accepted_starts=1,
        minimum_accepted_fraction=0.0,
    )
    offsets = np.asarray(multistart_site_translation_offsets_1d(offset_options))
    validation_interval = args.validation_interval
    convergence = ConvergenceOptions1D(min_updates=args.updates + 1)
    optimization = LatticeOptimizationOptions1D(mode="staged")
    run_records = []
    start_memory = []
    for index, offset in enumerate(offsets):
        _status(f"[{mode}] reconstruction start {index + 1}/{args.starts}")
        phase_start = perf_counter()
        result = run_prepared_lattice_site_reconstruction_1d(
            prepared,
            initial_rigid_displacement=offset,
            updates=args.updates,
            validation_interval=validation_interval,
            training_diagnostic_scan_count=args.training_diagnostic_scans,
            seed=args.seed + index,
            progress=False,
            convergence=convergence,
            optimization=optimization,
        )
        _block_result(jax, result)
        wall_time = perf_counter() - phase_start
        metadata = result.metadata
        optimization_time = float(metadata["optimization_time_s"])
        reported_run_time = float(metadata["run_time_s"])
        completed_updates = int(result.completed_updates)
        elapsed_history = np.asarray(result.elapsed_time_history, dtype=float)
        loop_to_last_evaluation = (
            float(elapsed_history[-1]) if elapsed_history.size else None
        )
        run_memory = _device_memory_snapshot(jax)
        start_memory.append(run_memory)
        run_records.append(
            {
                "start_index": index,
                "seed": args.seed + index,
                "initial_active_site_translation_A": offset.tolist(),
                "requested_updates": args.updates,
                "completed_updates": completed_updates,
                "wall_time_s": wall_time,
                "reported_run_time_s": reported_run_time,
                "optimization_time_s": optimization_time,
                "optimization_phase_timings_s": dict(
                    metadata["optimization_phase_timings_s"]
                ),
                "optimization_phase_unclassified_time_s": float(
                    metadata["optimization_phase_unclassified_time_s"]
                ),
                "training_diagnostic_scan_count": len(
                    metadata["training_diagnostic_indices"]
                ),
                "training_diagnostic_scan_evaluations": int(
                    metadata["training_diagnostic_scan_evaluations"]
                ),
                "validation_scan_evaluations": int(
                    metadata["validation_scan_evaluations"]
                ),
                "final_full_training_loss": float(
                    metadata["final_full_training_loss"]
                ),
                "loop_to_last_evaluation_time_s": loop_to_last_evaluation,
                "optimization_phase_updates_per_s": (
                    completed_updates / optimization_time
                    if optimization_time > 0.0
                    else None
                ),
                "loop_to_last_evaluation_updates_per_s": (
                    completed_updates / loop_to_last_evaluation
                    if loop_to_last_evaluation is not None
                    and loop_to_last_evaluation > 0.0
                    else None
                ),
                "best_update": int(result.best_update),
                "best_selection_loss": float(metadata["best_metric"]),
                "held_out_audit_loss": float(result.audit_loss),
                "stop_reason": result.stop_reason,
                "converged": bool(result.converged),
                "device_memory_after_synchronized_run": run_memory,
            }
        )
        del result
        gc.collect()

    workload_total_time = perf_counter() - workload_start
    process_to_completed_workload_time = perf_counter() - cli_start
    control_shape = (
        len(prepared.model.control_coordinates_s),
        len(prepared.model.control_coordinates_u),
        2,
    )
    n_sites = int(prepared.model.site_coordinates.shape[0])
    n_control_parameters = int(np.prod(control_shape))
    n_residual_control_dof = n_control_parameters - 2
    n_registration_parameters = 2
    optimization_times = [record["optimization_time_s"] for record in run_records]
    wall_times = [record["wall_time_s"] for record in run_records]
    total_completed_updates = sum(record["completed_updates"] for record in run_records)

    device_descriptions = []
    for device in devices:
        device_descriptions.append(
            {
                "string": str(device),
                "id": int(device.id),
                "local_hardware_id": (
                    int(device.local_hardware_id)
                    if hasattr(device, "local_hardware_id")
                    else None
                ),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "process_index": int(device.process_index),
            }
        )
    backend_client = getattr(devices[0], "client", None)

    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "scientific_comparability": (
            "reduced harness smoke geometry; do not compare with notebook results"
            if args.quick
            else "maintained SiliconGlancingConfig1D notebook geometry"
        ),
        "requested": {
            "updates_per_start": args.updates,
            "starts": args.starts,
            "precision": args.precision,
            "device": args.device,
            "case": args.case,
            "validation_interval": args.validation_interval,
            "effective_validation_interval": validation_interval,
            "training_diagnostic_scans": args.training_diagnostic_scans,
            "seed": args.seed,
            "run_protocol": (
                "screening-style prepared starts with fresh optimizer/RNG state; "
                "no representative checkpoint rerun"
            ),
            "early_stopping_disabled_for_fixed_update_timing": True,
        },
        "runtime": {
            "jax_backend": backend,
            "jax_backend_platform_version": (
                str(backend_client.platform_version)
                if backend_client is not None
                and hasattr(backend_client, "platform_version")
                else None
            ),
            "jax_backend_runtime_type": (
                str(backend_client.runtime_type)
                if backend_client is not None
                and hasattr(backend_client, "runtime_type")
                else None
            ),
            "jax_devices": device_descriptions,
            "jax_x64_enabled": bool(jax.config.x64_enabled),
            "effective_potential_dtype": str(prepared.model.reference_potential.dtype),
            "effective_probe_dtype": str(prepared.input_probe.dtype),
            "environment_configured_before_jax_import": environment,
            "python": sys.version,
            "platform": platform.platform(),
            "software_versions": {
                "wide-angle-propagation": _distribution_version(
                    "wide-angle-propagation"
                ),
                "numpy": _distribution_version("numpy"),
                "jax": _distribution_version("jax"),
                "jaxlib": _distribution_version("jaxlib"),
                "optax": _distribution_version("optax"),
                "abtem": _distribution_version("abtem"),
                "ase": _distribution_version("ase"),
                "scipy": _distribution_version("scipy"),
            },
            "source": _git_provenance(),
        },
        "problem": {
            "prepared_api_version": int(prepared.metadata["prepared_api_version"]),
            "reconstruction_problem_id": prepared.reconstruction_problem_id,
            "reconstructor_id": prepared.reconstructor_id,
            "config": asdict(config),
            "potential_shape": list(prepared.model.reference_potential.shape),
            "input_probe_shape": list(prepared.input_probe.shape),
            "diffraction_shape": list(prepared.measured_intensities.shape),
            "propagation_kernel_shape": list(prepared.propagation_kernel.shape),
            "site_patch_array_shape": list(prepared.model.site_patches.shape),
            "window_length_slices": int(prepared.window_length),
            "n_scans": int(prepared.measured_intensities.shape[0]),
            "n_training_scans": int(prepared.training_indices.size),
            "requested_training_diagnostic_scans": (
                args.training_diagnostic_scans
            ),
            "n_validation_scans": int(prepared.validation_indices.size),
            "n_audit_scans": int(prepared.audit_indices.size),
            "n_excluded_guard_scans": int(prepared.excluded_indices.size),
            "n_variable_sites": n_sites,
            "displacement_control_shape": list(control_shape),
            "n_vacancy_parameters": n_sites,
            "n_raw_displacement_control_parameters": n_control_parameters,
            "n_residual_displacement_control_dof": n_residual_control_dof,
            "n_active_site_registration_parameters": n_registration_parameters,
            "n_specimen_parameters": (
                n_sites + n_residual_control_dof + n_registration_parameters
            ),
            "minibatch_size": int(prepared.minibatch_size),
            "evaluation_batch_size": int(prepared.evaluation_batch_size),
            "rematerialize": bool(prepared.rematerialize),
        },
        "timings": {
            "runtime_import_and_backend_initialization_s": runtime_import_time,
            "experiment_build_s": build_time,
            "synthetic_dataset_simulation_s": simulation_time,
            "prepared_reconstruction_wall_s": preparation_wall_time,
            "prepared_reconstruction_internal_s": float(prepared.preparation_time_s),
            "starts_wall_sum_s": sum(wall_times),
            "workload_total_s": workload_total_time,
            "process_to_completed_workload_s": process_to_completed_workload_time,
            "scope": {
                "experiment_build_s": (
                    "workflow geometry, abTEM templates, finite reference slab, "
                    "and synchronized JAX arrays"
                ),
                "synthetic_dataset_simulation_s": (
                    "one complete workflow simulation and synchronized template/"
                    "exterior diagnostics"
                ),
                "prepared_reconstruction_wall_s": (
                    "validation, transfer, eager compilation, and synchronization "
                    "of reusable renderer/train/prediction executables"
                ),
                "per_start.wall_time_s": (
                    "one run_prepared call plus an explicit result synchronization; "
                    "excludes experiment build, simulation, and preparation"
                ),
                "per_start.optimization_time_s": (
                    "prepared runner optimization phase: initial and scheduled "
                    "train/validation evaluations, optimizer updates, final all-scan "
                    "prediction, and audit evaluation"
                ),
                "per_start.loop_to_last_evaluation_time_s": (
                    "optimization start through the final scheduled evaluation; "
                    "includes initial/scheduled evaluations but excludes post-loop "
                    "all-scan prediction and audit evaluation"
                ),
                "workload_total_s": (
                    "experiment build through all synchronized starts; excludes "
                    "runtime imports/backend initialization and JSON serialization"
                ),
                "process_to_completed_workload_s": (
                    "CLI entry through all synchronized starts; excludes report "
                    "assembly and JSON serialization"
                ),
            },
        },
        "throughput": {
            "total_completed_updates": total_completed_updates,
            "aggregate_optimization_phase_updates_per_s": (
                total_completed_updates / sum(optimization_times)
                if sum(optimization_times) > 0.0
                else None
            ),
            "median_per_start_optimization_phase_updates_per_s": statistics.median(
                record["optimization_phase_updates_per_s"] for record in run_records
            ),
            "scope": (
                "completed optimizer updates divided by optimization_time_s; this "
                "is synchronized end-to-end optimization-phase throughput, not an "
                "isolated compiled-train-step kernel rate"
            ),
        },
        "memory": {
            "after_experiment_build": memory_after_build,
            "after_dataset_simulation": memory_after_simulation,
            "after_preparation": memory_after_preparation,
            "after_each_start": start_memory,
        },
        "starts": run_records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(args)
    _write_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
