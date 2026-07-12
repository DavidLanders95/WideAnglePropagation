#!/usr/bin/env python3
"""Benchmark the prepared sparse atomistic-edit ptychography runtime.

The default uses the maintained side-view silicon geometry. The --quick mode
is an installation smoke test, not a scientific performance reference. One
truth-free problem is prepared, then independent empty-edit starts expose the
first-run compilation cost and warm prepared-runtime cost.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "wide_angle_propagation.atomistic_edit_performance_benchmark"
REPORT_SCHEMA_VERSION = 1


def _positive_integer(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _nonnegative_integer(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return result


def _penalty_path(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use comma-separated positive penalties"
        ) from error
    if (
        not result
        or any(not math.isfinite(item) or item <= 0.0 for item in result)
        or any(left <= right for left, right in zip(result, result[1:]))
    ):
        raise argparse.ArgumentTypeError(
            "penalties must be positive and strictly decreasing"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Time prepared sparse atomistic-edit reconstruction."
    )
    parser.add_argument("--quick", action="store_true", help="use smoke geometry")
    parser.add_argument(
        "--updates", type=_positive_integer, default=40,
        help="Adam updates per continuous stage (default: 40)",
    )
    parser.add_argument(
        "--active-set-iterations", type=_positive_integer, default=16,
        help="active-set steps per penalty (default: 16)",
    )
    parser.add_argument(
        "--starts", type=_positive_integer, default=5,
        help="independent empty-edit starts (default: 5)",
    )
    parser.add_argument(
        "--edit-penalty-path",
        type=_penalty_path,
        default=(1e3, 1e2),
        metavar="L1,L2,...",
        help="strictly decreasing edit penalties (default: 1000,100)",
    )
    parser.add_argument(
        "--precision", choices=("float32", "float64"), default="float64",
        help="JAX precision (default: float64)",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "gpu"), default="auto",
        help="JAX platform selected before import (default: auto)",
    )
    parser.add_argument(
        "--case",
        choices=("vacancy", "vacancy_plus_strain", "strained_surface_defects"),
        default="strained_surface_defects",
        help="synthetic specimen used only to generate measurements",
    )
    parser.add_argument(
        "--training-diagnostic-scans",
        type=_positive_integer,
        default=32,
        help="compatibility value recorded but unused by the full-split solver",
    )
    parser.add_argument(
        "--training-scan-batch-size", type=_positive_integer, default=32,
        help="exact training-scan accumulation batch size (default: 32)",
    )
    parser.add_argument(
        "--seed", type=_nonnegative_integer, default=0,
        help="seed for nonzero starts (default: 0)",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="write JSON to PATH; omit it or use '-' for stdout",
    )
    return parser


def _configure_environment(*, precision: str, device: str) -> dict[str, str]:
    if "jax" in sys.modules:
        raise RuntimeError("JAX was imported before environment configuration")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_ENABLE_X64"] = "true" if precision == "float64" else "false"
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device == "gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
    return {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_ENABLE_X64": os.environ["JAX_ENABLE_X64"],
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "<unset>"),
    }


def _quick_config(workflow: Any) -> Any:
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


def _ready(jax: Any, *values: Any) -> None:
    jax.block_until_ready(values)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    try:
        return _json_value(float(value))
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _write_report(report: Mapping[str, Any], output: str | None) -> None:
    payload = json.dumps(
        _json_value(report), indent=2, sort_keys=True, allow_nan=False
    )
    if output is None or output == "-":
        print(payload)
        return
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"wrote {path}", file=sys.stderr, flush=True)


def _surface_envelope(experiment: Any) -> tuple[float, float]:
    coordinates = experiment.transverse_coordinates
    bottom = max(
        -float(experiment.config.slab_depth_A), float(coordinates[0])
    )
    top = min(
        10.0,
        float(experiment.config.vacuum_above_A),
        float(coordinates[-1]),
    )
    if bottom >= top:
        raise RuntimeError("surface discovery envelope is empty")
    return bottom, top


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    process_start = perf_counter()
    environment = _configure_environment(
        precision=args.precision, device=args.device
    )

    import_start = perf_counter()
    import jax
    import jax.numpy as jnp
    import numpy as np

    from wide_angle_propagation import ptychography_workflow_1d as workflow
    from wide_angle_propagation.ptychography_1d import PtychographyObjective1D
    from wide_angle_propagation.ptychography_atomistic_edit_1d import (
        AtomisticEditOptions1D,
        empty_atomistic_edit_state_1d,
    )
    from wide_angle_propagation.ptychography_atomistic_edit_solver_1d import (
        AtomisticEditSolverOptions1D,
        run_prepared_atomistic_edit_reconstruction_1d,
    )

    devices = jax.devices()
    backend = jax.default_backend()
    import_time = perf_counter() - import_start
    if args.device == "cpu" and backend != "cpu":
        raise RuntimeError(f"requested CPU but initialized {backend!r}")
    if args.device == "gpu" and backend not in {"gpu", "cuda"}:
        raise RuntimeError(f"requested GPU but initialized {backend!r}")
    if bool(jax.config.x64_enabled) != (args.precision == "float64"):
        raise RuntimeError("JAX precision differs from the requested policy")

    mode = "quick" if args.quick else "notebook"
    config = (
        _quick_config(workflow)
        if args.quick
        else workflow.SiliconGlancingConfig1D()
    )
    maximum_removals = 4 if args.quick else 16
    maximum_additions = 2 if args.quick else 8
    batch_size = 3 if args.quick else 10

    print(f"[{mode}] building host geometry", file=sys.stderr, flush=True)
    phase_start = perf_counter()
    experiment = workflow.build_silicon_glancing_experiment_1d(config)
    _ready(
        jax,
        experiment.pristine_potential,
        experiment.input_probes,
        experiment.propagation_kernel,
        experiment.lattice_model.site_patches,
    )
    build_time = perf_counter() - phase_start

    print(
        f"[{mode}] simulating {args.case!r} measurements",
        file=sys.stderr,
        flush=True,
    )
    phase_start = perf_counter()
    dataset = workflow.simulate_experiment_1d(
        experiment, case=args.case, batch_size=batch_size
    )
    _ready(jax, dataset.potential, dataset.intensities)
    simulation_time = perf_counter() - phase_start

    objective = PtychographyObjective1D(
        kind="poisson_deviance",
        electrons_per_pattern=1e6,
        minimum_expected_electrons=1e-9,
        relative_signal_scale=1.0,
    )
    valid = dataset.scan.detector_valid_mask
    if valid is None:
        valid = np.ones_like(np.asarray(dataset.intensities), dtype=bool)
    measurement = workflow.synthetic_noiseless_poisson_measurement_1d(
        experiment,
        dataset.scan,
        objective,
        detector_valid_mask=valid,
        calibration_id="benchmark_synthetic_noiseless_poisson:v1",
    )
    surface_envelope = _surface_envelope(experiment)
    discovery = workflow.build_atomistic_edit_discovery_support_1d(
        experiment, surface_envelope_A=surface_envelope
    )
    model_options = AtomisticEditOptions1D(
        max_host_removals=maximum_removals,
        max_extra_centres=maximum_additions,
        max_scattering_equivalent_per_centre=2.0,
        minimum_separation_A=1.8,
        expected_rms_host_strain=0.03,
        edit_penalty_path=tuple(args.edit_penalty_path),
        discovery_support=discovery,
        enable_material_energy_envelope=False,
    )

    print(f"[{mode}] binding prepared AE problem", file=sys.stderr, flush=True)
    phase_start = perf_counter()
    prepared = workflow.prepare_atomistic_edit_experiment_1d(
        experiment,
        measurement,
        objective,
        model_options,
        surface_envelope_A=surface_envelope,
    )
    _ready(
        jax,
        prepared.model.host_model.reference_potential,
        prepared.model.addition_kernel.unit_integrated_values,
        prepared.probe_rows,
        prepared.measurement.observed_total_electrons,
    )
    preparation_time = perf_counter() - phase_start
    if prepared.metadata.get("truth_fields_read") is not False:
        raise RuntimeError("prepared problem crossed the truth boundary")

    solver_options = AtomisticEditSolverOptions1D(
        ablation="level1_physical",
        maximum_active_set_iterations=args.active_set_iterations,
        joint_refinement_updates=args.updates,
        polish_updates=args.updates,
        debias_updates=args.updates,
        training_scan_batch_size=args.training_scan_batch_size,
        seed=args.seed,
    )
    empty = empty_atomistic_edit_state_1d(prepared.model)
    controls_shape = np.asarray(empty.host_displacement_controls).shape
    displacement_bound = float(
        np.asarray(prepared.model.host_model.maximum_displacement)
    )
    initial_control_std_A = min(0.01, displacement_bound)
    child_seeds = np.random.SeedSequence(args.seed).spawn(args.starts)
    starts = []
    for index, child_seed in enumerate(child_seeds):
        seed = int(child_seed.generate_state(1, dtype=np.uint64)[0])
        if index == 0:
            controls = np.zeros(controls_shape)
        else:
            controls = np.random.default_rng(child_seed).normal(
                0.0, initial_control_std_A, controls_shape
            )
            controls = np.clip(controls, -displacement_bound, displacement_bound)
        initial_state = replace(
            empty,
            host_displacement_controls=jnp.asarray(
                controls,
                dtype=jnp.asarray(empty.host_displacement_controls).dtype,
            ),
        )
        print(
            f"[{mode}] atomistic-edit start {index + 1}/{args.starts}",
            file=sys.stderr,
            flush=True,
        )
        phase_start = perf_counter()
        result = run_prepared_atomistic_edit_reconstruction_1d(
            prepared,
            initial_state=initial_state,
            options=replace(solver_options, seed=seed),
            show_progress=False,
            evaluate_audit=False,
        )
        _ready(
            jax,
            result.debiased_state.host_removal_fractions,
            result.debiased_state.extra_scattering_equivalents,
            result.debiased_state.host_displacement_controls,
            result.debiased_training_objective.total_objective,
        )
        wall_time = perf_counter() - phase_start
        starts.append(
            {
                "index": index,
                "seed": seed,
                "initial_host_control_rms_A": float(
                    np.sqrt(np.mean(controls**2))
                ),
                "wall_time_s": wall_time,
                "includes_lazy_compilation": index == 0,
                "path_points_solved": len(result.path_points),
                "active_set_iterations": sum(
                    point.active_set_iterations for point in result.path_points
                ),
                "birth_events": [
                    event
                    for point in result.path_points
                    for event in point.births
                ],
                "active_parameter_count": int(result.active_parameter_count),
                "validation_count_deviance": float(
                    result.debiased_validation_count_deviance
                ),
                "proposal_grid_kkt_satisfied": bool(
                    result.selected_kkt.satisfied
                ),
                "capacity_exhausted": bool(result.capacity_exhausted),
                "converged": bool(result.converged),
                "stop_reason": result.stop_reason,
            }
        )

    wall_times = [record["wall_time_s"] for record in starts]
    deformation_count = int(prepared.model.deformation_parameter_count)
    edit_capacity_count = maximum_removals + 3 * maximum_additions
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "scientific_comparability": (
            "reduced smoke" if args.quick else "maintained side-view geometry"
        ),
        "requested": {
            "joint_refinement_updates": args.updates,
            "active_set_iterations_per_penalty": args.active_set_iterations,
            "polish_and_debias_updates": args.updates,
            "starts": args.starts,
            "initial_host_control_std_A": initial_control_std_A,
            "edit_penalty_path": list(args.edit_penalty_path),
            "case": args.case,
            "precision": args.precision,
            "device": args.device,
            "seed": args.seed,
            "training_scan_batch_size": args.training_scan_batch_size,
        },
        "runtime": {
            "jax_version": jax.__version__,
            "backend": backend,
            "x64_enabled": bool(jax.config.x64_enabled),
            "devices": [
                {"platform": str(d.platform), "kind": str(d.device_kind)}
                for d in devices
            ],
            "environment": environment,
            "python": sys.version,
            "platform": platform.platform(),
        },
        "problem": {
            "prepared_problem_id": prepared.reconstruction_problem_id,
            "reconstructor_id": prepared.reconstructor_id,
            "model_id": prepared.model.model_id,
            "potential_shape": list(prepared.model.host_model.reference_potential.shape),
            "diffraction_shape": list(
                prepared.measurement.observed_total_electrons.shape),
            "scan_split_counts": {
                "training": int(prepared.training_indices.size),
                "validation": int(prepared.validation_indices.size),
                "audit": int(prepared.audit_indices.size),
                "guard": int(prepared.excluded_indices.size),
            },
            "surface_envelope_A": list(surface_envelope),
            "discovery_anchor_counts": {
                "target": int(np.count_nonzero(discovery.target_mask)),
                "nuisance": int(np.count_nonzero(discovery.nuisance_mask)),
            },
            "host_site_count": int(prepared.model.host_model.site_coordinates.shape[0]),
            "deformation_parameter_count": deformation_count,
            "maximum_host_removals": maximum_removals,
            "maximum_extra_centres": maximum_additions,
            "maximum_total_parameter_count": (
                deformation_count + edit_capacity_count
            ),
            "truth_fields_read_by_preparation": False,
        },
        "timings_s": {
            "import_and_backend": import_time,
            "experiment_build": build_time,
            "synthetic_measurement_simulation": simulation_time,
            "prepared_problem_binding": preparation_time,
            "starts_sum": sum(wall_times),
            "warm_start_median": (
                statistics.median(wall_times[1:])
                if len(wall_times) > 1
                else None
            ),
            "process_to_completed_workload": perf_counter() - process_start,
            "first_start_includes_lazy_compilation": True,
        },
        "starts": starts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _write_report(run_benchmark(args), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
