"""Small silicon-specific facade for authenticated sparse atomic edits.

This module deliberately composes the maintained AE-1/AE-2 boundaries rather
than reimplementing them.  Callers supply calibrated measurements and an
explicit, predeclared edit-penalty path.  No object-presence flag, defect
coordinate, radius, shape, phase, or composition enters the configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import operator
from pathlib import Path
from typing import Any

import numpy as np

from .ptychography_1d import PtychographyMeasurement1D, PtychographyObjective1D
from .ptychography_atomistic_edit_1d import AtomisticEditOptions1D, AtomisticEditState1D
from .ptychography_atomistic_edit_io_1d import (
    load_atomistic_edit_reconstruction_bundle_1d,
    make_atomistic_edit_reconstruction_bundle_1d,
    save_atomistic_edit_reconstruction_bundle_1d,
)
from .ptychography_atomistic_edit_solver_1d import (
    AtomisticEditProgressCallback1D,
    AtomisticEditReconstruction1D,
    AtomisticEditSolverOptions1D,
    PreparedAtomisticEditReconstruction1D,
    run_prepared_atomistic_edit_reconstruction_1d,
)
from .ptychography_workflow_1d import (
    SiliconGlancingExperiment1D,
    build_atomistic_edit_discovery_support_1d,
    plot_atomistic_edit_reconstruction_1d,
    prepare_atomistic_edit_experiment_1d,
    summarize_atomistic_edit_reconstruction_1d,
)


__all__ = [
    "SiliconAtomisticEditConfig1D",
    "SiliconAtomisticEditRun1D",
    "load_silicon_atomistic_edit_run_1d",
    "plot_silicon_atomistic_edit_run_1d",
    "reconstruct_silicon_atomistic_edits_1d",
    "save_silicon_atomistic_edit_run_1d",
    "summarize_silicon_atomistic_edit_run_1d",
]


_FACADE_ID = "silicon_atomistic_edit_workflow_1d:v1"


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(result)


def _positive_scalar(name: str, value: Any) -> float:
    array = np.asarray(value)
    if (
        array.shape != ()
        or np.iscomplexobj(array)
        or isinstance(value, (bool, np.bool_))
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(array)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _penalty_path(value: Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("edit_penalty_path must be a sequence of positive values")
    try:
        result = tuple(
            _positive_scalar(f"edit_penalty_path[{index}]", item)
            for index, item in enumerate(value)
        )
    except TypeError as error:
        raise TypeError(
            "edit_penalty_path must be a sequence of positive values"
        ) from error
    if not result:
        raise ValueError("edit_penalty_path must not be empty")
    if any(left <= right for left, right in zip(result, result[1:])):
        raise ValueError("edit_penalty_path must be strictly decreasing")
    return result


@dataclass(frozen=True)
class SiliconAtomisticEditConfig1D:
    """Object-free material and numerical policy for one silicon AE-2 run.

    The penalty path must be calibrated without inspecting a recovered object.
    The ``max_*`` fields are compilation capacities, not prior atom counts; the
    broad vacuum band is clipped to the simulated geometry. Training gradients
    are accumulated exactly over deterministic scan batches by default so the
    facade remains usable for the full side-view acquisition.
    """

    edit_penalty_path: tuple[float, ...]
    max_host_removals: int = 16
    max_extra_centres: int = 8
    max_scattering_equivalent_per_centre: float = 2.0
    minimum_separation_A: float = 1.8
    expected_rms_host_strain: float = 0.03
    vacuum_discovery_band_A: float = 10.0
    maximum_active_set_iterations: int = 16
    joint_refinement_updates: int = 40
    polish_updates: int = 40
    debias_updates: int = 60
    training_scan_batch_size: int = 32
    seed: int = 0
    show_progress: bool = True
    evaluate_audit: bool = True

    def __post_init__(self) -> None:
        path = _penalty_path(self.edit_penalty_path)
        removals = _nonnegative_integer(
            "max_host_removals", self.max_host_removals
        )
        additions = _nonnegative_integer(
            "max_extra_centres", self.max_extra_centres
        )
        if removals + additions == 0:
            raise ValueError("at least one atomistic-edit capacity must be positive")
        active_iterations = _nonnegative_integer(
            "maximum_active_set_iterations", self.maximum_active_set_iterations
        )
        if active_iterations == 0:
            raise ValueError("maximum_active_set_iterations must be positive")
        object.__setattr__(self, "maximum_active_set_iterations", active_iterations)
        for name in (
            "joint_refinement_updates",
            "polish_updates",
            "debias_updates",
            "seed",
        ):
            object.__setattr__(self, name, _nonnegative_integer(name, getattr(self, name)))
        if self.seed >= 2**64:
            raise ValueError("seed must lie in [0, 2**64)")
        for name in ("show_progress", "evaluate_audit"):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be Boolean")
            object.__setattr__(self, name, bool(value))
        scan_batch_size = _nonnegative_integer(
            "training_scan_batch_size", self.training_scan_batch_size
        )
        if scan_batch_size == 0:
            raise ValueError("training_scan_batch_size must be positive")
        object.__setattr__(
            self, "training_scan_batch_size", scan_batch_size
        )
        object.__setattr__(self, "edit_penalty_path", path)
        object.__setattr__(self, "max_host_removals", removals)
        object.__setattr__(self, "max_extra_centres", additions)
        for name in (
            "max_scattering_equivalent_per_centre",
            "minimum_separation_A",
            "expected_rms_host_strain",
            "vacuum_discovery_band_A",
        ):
            object.__setattr__(
                self, name, _positive_scalar(name, getattr(self, name))
            )


@dataclass(frozen=True, eq=False)
class SiliconAtomisticEditRun1D:
    """Prepared authenticated problem and its single-start AE-2 result."""

    prepared: PreparedAtomisticEditReconstruction1D
    result: AtomisticEditReconstruction1D
    solver_options: AtomisticEditSolverOptions1D
    archive_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedAtomisticEditReconstruction1D):
            raise TypeError("prepared must be PreparedAtomisticEditReconstruction1D")
        if not isinstance(self.result, AtomisticEditReconstruction1D):
            raise TypeError("result must be AtomisticEditReconstruction1D")
        if not isinstance(self.solver_options, AtomisticEditSolverOptions1D):
            raise TypeError("solver_options must be AtomisticEditSolverOptions1D")
        if self.prepared.metadata.get("truth_fields_read") is not False:
            raise ValueError("prepared problem lacks the truth-free facade marker")
        if self.result.prepared_problem_id != self.prepared.reconstruction_problem_id:
            raise ValueError("result does not belong to the prepared problem")
        if not isinstance(self.archive_id, str):
            raise TypeError("archive_id must be text")
        if self.archive_id and (
            len(self.archive_id) != 64
            or any(character not in "0123456789abcdef" for character in self.archive_id)
        ):
            raise ValueError("archive_id must be a lowercase SHA-256 digest")

    @property
    def surface_envelope_A(self) -> tuple[float, float]:
        """Resolved slab-to-vacuum discovery bounds stored by preparation."""
        value = self.prepared.metadata.get("surface_envelope_A")
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("prepared problem lacks resolved surface bounds")
        return float(value[0]), float(value[1])


def _surface_envelope_A(
    experiment: SiliconGlancingExperiment1D,
    config: SiliconAtomisticEditConfig1D,
) -> tuple[float, float]:
    coordinates = np.asarray(experiment.transverse_coordinates, dtype=float)
    if (
        coordinates.ndim != 1
        or coordinates.size < 2
        or np.any(~np.isfinite(coordinates))
        or np.any(np.diff(coordinates) <= 0.0)
    ):
        raise ValueError("experiment transverse coordinates are not a valid grid")
    slab_depth_A = _positive_scalar("slab_depth_A", experiment.config.slab_depth_A)
    declared_vacuum_A = _positive_scalar("vacuum_above_A", experiment.config.vacuum_above_A)
    bottom_A = -slab_depth_A
    grid_bottom_A = float(coordinates[0])
    grid_top_A = float(coordinates[-1])
    tolerance = 1e-9 * max(1.0, abs(bottom_A), abs(grid_bottom_A), abs(grid_top_A))
    if grid_bottom_A > bottom_A + tolerance or grid_top_A <= 0.0:
        raise ValueError(
            "the simulated transverse grid does not contain the declared slab "
            "bottom and positive vacuum"
        )
    top_A = min(config.vacuum_discovery_band_A, declared_vacuum_A, grid_top_A)
    if not np.isfinite(top_A) or top_A <= 0.0 or bottom_A >= top_A:
        raise ValueError("the resolved slab-to-vacuum discovery envelope is empty")
    return float(max(bottom_A, grid_bottom_A)), float(top_A)


def reconstruct_silicon_atomistic_edits_1d(
    experiment: SiliconGlancingExperiment1D,
    measurement: PtychographyMeasurement1D,
    objective: PtychographyObjective1D,
    *,
    config: SiliconAtomisticEditConfig1D,
    progress_callback: AtomisticEditProgressCallback1D | None = None,
) -> SiliconAtomisticEditRun1D:
    """Prepare and run the authenticated silicon AE-2 reconstruction."""
    if not isinstance(experiment, SiliconGlancingExperiment1D):
        raise TypeError("experiment must be a SiliconGlancingExperiment1D")
    if not isinstance(config, SiliconAtomisticEditConfig1D):
        raise TypeError("config must be SiliconAtomisticEditConfig1D")
    surface_envelope_A = _surface_envelope_A(experiment, config)
    discovery = build_atomistic_edit_discovery_support_1d(
        experiment,
        surface_envelope_A=surface_envelope_A,
    )
    model_options = AtomisticEditOptions1D(
        max_host_removals=config.max_host_removals,
        max_extra_centres=config.max_extra_centres,
        max_scattering_equivalent_per_centre=config.max_scattering_equivalent_per_centre,
        minimum_separation_A=config.minimum_separation_A,
        expected_rms_host_strain=config.expected_rms_host_strain,
        edit_penalty_path=config.edit_penalty_path,
        discovery_support=discovery,
        enable_material_energy_envelope=False,
    )
    prepared = prepare_atomistic_edit_experiment_1d(
        experiment, measurement, objective, model_options,
        surface_envelope_A=surface_envelope_A,
    )
    if prepared.metadata.get("truth_fields_read") is not False:
        raise RuntimeError("silicon preparation did not preserve the truth boundary")
    solver_options = AtomisticEditSolverOptions1D(
        ablation="level1_physical",
        maximum_active_set_iterations=config.maximum_active_set_iterations,
        joint_refinement_updates=config.joint_refinement_updates,
        polish_updates=config.polish_updates,
        debias_updates=config.debias_updates,
        training_scan_batch_size=config.training_scan_batch_size,
        seed=config.seed,
    )
    result = run_prepared_atomistic_edit_reconstruction_1d(
        prepared,
        options=solver_options,
        show_progress=config.show_progress,
        evaluate_audit=config.evaluate_audit,
        progress_callback=progress_callback,
    )
    return SiliconAtomisticEditRun1D(prepared, result, solver_options)


def plot_silicon_atomistic_edit_run_1d(
    experiment: SiliconGlancingExperiment1D,
    run: SiliconAtomisticEditRun1D,
    *,
    state: AtomisticEditState1D | None = None,
    truth_state: AtomisticEditState1D | None = None,
    truth_potential: Any | None = None,
):
    """Plot authenticated TARGET influence for the result or one checkpoint."""
    if not isinstance(run, SiliconAtomisticEditRun1D):
        raise TypeError("run must be SiliconAtomisticEditRun1D")
    displayed = run.result if state is None else state
    return plot_atomistic_edit_reconstruction_1d(
        experiment,
        run.prepared,
        displayed,
        truth_state=truth_state,
        truth_potential=truth_potential,
    )


def summarize_silicon_atomistic_edit_run_1d(
    experiment: SiliconGlancingExperiment1D,
    run: SiliconAtomisticEditRun1D,
) -> Mapping[str, Any]:
    """Return the contract-checked sparse-edit and stopping summary."""
    if not isinstance(run, SiliconAtomisticEditRun1D):
        raise TypeError("run must be SiliconAtomisticEditRun1D")
    return summarize_atomistic_edit_reconstruction_1d(
        experiment, run.prepared, run.result
    )


def save_silicon_atomistic_edit_run_1d(
    path: str | Path,
    run: SiliconAtomisticEditRun1D,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Save a non-pickled authenticated facade run through the AE-2 archive."""
    if not isinstance(run, SiliconAtomisticEditRun1D):
        raise TypeError("run must be SiliconAtomisticEditRun1D")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping or None")
    caller_metadata = dict(provenance or {})
    caller_metadata["workflow_facade"] = _FACADE_ID
    bundle = make_atomistic_edit_reconstruction_bundle_1d(
        run.prepared,
        run.result,
        solver_options=run.solver_options,
        provenance=caller_metadata,
    )
    save_atomistic_edit_reconstruction_bundle_1d(path, bundle)


def load_silicon_atomistic_edit_run_1d(
    path: str | Path,
) -> SiliconAtomisticEditRun1D:
    """Authenticate and replay a saved facade run."""
    bundle = load_atomistic_edit_reconstruction_bundle_1d(path)
    caller_metadata = bundle.provenance.get("caller_metadata")
    if (
        not isinstance(caller_metadata, Mapping)
        or caller_metadata.get("workflow_facade") != _FACADE_ID
    ):
        raise ValueError("archive was not created by the silicon AE facade")
    return SiliconAtomisticEditRun1D(
        prepared=bundle.prepared,
        result=bundle.reconstruction,
        solver_options=bundle.solver_options,
        archive_id=bundle.archive_id,
    )
