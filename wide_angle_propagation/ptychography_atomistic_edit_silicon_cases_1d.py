"""Physical, truth-isolated silicon cases for the AE-3 blind benchmark.

The benchmark contract intentionally lives in a separate module.  This file
binds that generic contract to one :class:`SiliconGlancingExperiment1D`, one
object-free AE-1 model, a frozen calibrated-count partition, and private
deterministic seeds.  Only train/validation observations cross the callback
boundary.  Audit observations, generating coordinates, elements, seeds, and
forward-mismatch causes remain behind lazy private factories.

Positive truth edits are rendered by direct, exact-subpixel Kirkland
quadrature.  They do not use the production Lobato addition renderer or image
interpolation.  Propagation is deliberately shared with the maintained
glancing forward model, so these cases are implementation stress tests rather
than experimental validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import operator
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .ptychography_1d import (
    PtychographyMeasurement1D,
    PtychographyObjective1D,
    lattice_site_displacements_1d,
    ptychography_expected_signal_electrons_1d,
    render_lattice_site_potential_1d,
    simulate_glancing_scan_1d,
)
from .ptychography_atomic_validation_1d import (
    AtomicTemplateQuadratureOptions1D,
)
from .ptychography_atomistic_edit_1d import (
    AtomisticEditModel1D,
    AtomisticEditOptions1D,
    AtomisticEditState1D,
    atomistic_edit_addition_positions_1d,
    atomistic_edit_prior_components_1d,
    render_atomistic_edit_potential_1d,
)
from .ptychography_atomistic_edit_benchmarks_1d import (
    ActiveEditMultistartEvidence1D,
    AtomisticEditAblationArm1D,
    AtomisticEditBlindAuditCounts1D,
    AtomisticEditBlindCase1D,
    AtomisticEditBlindCaseRole1D,
    AtomisticEditBlindPrivateTruth1D,
    AtomisticEditBlindPublicProblem1D,
    AtomisticEditBlindReconstruction1D,
    AtomisticEditReconstructionContract1D,
    PhysicalAdmissibilityMetrics1D,
    ResolutionAwareMassMeasure1D,
    validate_atomistic_edit_blind_public_problem_1d,
)
from .ptychography_atomistic_edit_solver_1d import (
    AtomisticEditMultistartReconstruction1D,
    AtomisticEditSolverOptions1D,
    prepare_atomistic_edit_reconstruction_1d,
    run_prepared_atomistic_edit_multistart_reconstruction_1d,
)
from .ptychography_atomistic_truth_1d import (
    DirectAtomicNumericalOptions1D,
    accumulate_weighted_atomic_potential_1d,
    render_direct_atomic_template_1d,
)
from .ptychography_support_contract_1d import LatticeSiteRole1D
from .ptychography_workflow_1d import (
    SiliconGlancingExperiment1D,
    build_atomistic_edit_model_1d,
)


Array = Any


__all__ = [
    "AtomisticEditBlindCountSelectionContract1D",
    "make_atomistic_edit_blind_count_selection_contract_1d",
    "make_silicon_atomistic_edit_blind_cases_1d",
    "make_silicon_atomistic_edit_reconstruction_callback_1d",
    "make_silicon_atomistic_edit_reconstruction_callbacks_1d",
]


BlindCallback1D = Callable[
    [AtomisticEditBlindPublicProblem1D], AtomisticEditBlindReconstruction1D
]

_SELECTION_RULE_ID = (
    "geometry-frozen-train-validation-only;private-audit-post-selection:v1"
)
_NUISANCE_SCOPE_ID = (
    "current-ae2-has-no-fitted-nuisance-block;claim-unavailable:v1"
)
_OBSERVABILITY_RULE_ID = (
    "no-independent-acquisition-observability-evaluated;claim-unavailable:v1"
)
_PRIOR_ID = (
    "count-deviance+one-edit-mass+weak-symmetric-strain+occupancy-hard-core:v1"
)
_SOLVER_SELECTION_ID = (
    "validation-largest-lambda-within-frozen-tolerance;fixed-support-debias:v1"
)


def _readonly(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty stripped string")
    return value


def _positive(name: str, value: Any) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or np.iscomplexobj(array)
        or isinstance(value, (bool, np.bool_))
    ):
        raise TypeError(f"{name} must be a real scalar")
    result = float(array)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _seed(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < 0 or result >= 2**64:
        raise ValueError(f"{name} must lie in [0, 2**64)")
    return int(result)


def _partition(name: str, value: Any, *, n_scan: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must be a one-dimensional integer array")
    converted = np.asarray(array, dtype=np.int64)
    if (
        len(np.unique(converted)) != len(converted)
        or np.any(converted < 0)
        or np.any(converted >= n_scan)
    ):
        raise ValueError(f"{name} contains a duplicate or out-of-range scan")
    return _readonly(converted, dtype=np.int64)


def _digest(*, arrays: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(json.dumps(array.shape).encode("ascii"))
        hasher.update(array.view(np.uint8).tobytes())
    hasher.update(
        json.dumps(
            dict(metadata),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return hasher.hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, eq=False)
class AtomisticEditBlindCountSelectionContract1D:
    """Frozen count calibration and geometry-only scan partition.

    The detector mask and all four partitions are fixed before any private
    specimen is generated.  Audit indices are retained only in this trusted
    orchestration object; public callback inputs contain no audit rows.
    """

    electrons_per_pattern: Array
    detector_valid_mask: Array
    training_indices: Array
    validation_indices: Array
    audit_indices: Array
    excluded_indices: Array
    calibration_id: str
    minimum_expected_electrons: float = 1e-9
    relative_signal_scale: float = 1.0
    poisson_sample: bool = True
    selection_rule_id: str = _SELECTION_RULE_ID
    contract_id: str = field(init=False)

    def __post_init__(self) -> None:
        valid = np.asarray(self.detector_valid_mask)
        if valid.ndim != 2 or valid.dtype != np.bool_:
            raise TypeError("detector_valid_mask must be a two-dimensional Boolean array")
        if not np.any(valid):
            raise ValueError("detector_valid_mask removes every observation")
        n_scan = int(valid.shape[0])
        dose = np.asarray(self.electrons_per_pattern)
        if np.iscomplexobj(dose) or np.issubdtype(dose.dtype, np.bool_):
            raise TypeError("electrons_per_pattern must be real numeric")
        if dose.ndim == 0:
            dose = np.full(n_scan, dose.item(), dtype=np.result_type(dose, float))
        if dose.shape != (n_scan,):
            raise ValueError(
                "electrons_per_pattern must be scalar or have one value per scan"
            )
        dose = np.asarray(dose, dtype=np.float64)
        if np.any(~np.isfinite(dose)) or np.any(dose <= 0.0):
            raise ValueError("electrons_per_pattern must be finite and positive")
        partitions = {
            name: _partition(name, getattr(self, name), n_scan=n_scan)
            for name in (
                "training_indices",
                "validation_indices",
                "audit_indices",
                "excluded_indices",
            )
        }
        if not len(partitions["training_indices"]):
            raise ValueError("training_indices must not be empty")
        if not len(partitions["validation_indices"]):
            raise ValueError("validation_indices must not be empty")
        if not len(partitions["audit_indices"]):
            raise ValueError("audit_indices must not be empty")
        concatenated = np.concatenate(tuple(partitions.values()))
        if len(np.unique(concatenated)) != len(concatenated):
            raise ValueError("training, validation, audit, and excluded scans overlap")
        if not np.array_equal(np.sort(concatenated), np.arange(n_scan)):
            raise ValueError("the four scan partitions must classify every scan exactly once")
        if not np.any(valid[partitions["training_indices"]]):
            raise ValueError("the training split has no valid detector values")
        if not np.any(valid[partitions["validation_indices"]]):
            raise ValueError("the validation split has no valid detector values")
        if not np.any(valid[partitions["audit_indices"]]):
            raise ValueError("the audit split has no valid detector values")
        if not isinstance(self.poisson_sample, (bool, np.bool_)):
            raise TypeError("poisson_sample must be Boolean")
        calibration_id = _identifier("calibration_id", self.calibration_id)
        selection_rule_id = _identifier("selection_rule_id", self.selection_rule_id)
        floor = _positive(
            "minimum_expected_electrons", self.minimum_expected_electrons
        )
        scale = _positive("relative_signal_scale", self.relative_signal_scale)
        valid = _readonly(valid, dtype=bool)
        dose = _readonly(dose, dtype=np.float64)
        identity = _digest(
            arrays={
                "dose": dose,
                "valid": valid,
                **partitions,
            },
            metadata={
                "schema": "atomistic_edit_blind_count_selection_1d:v1",
                "calibration_id": calibration_id,
                "minimum_expected_electrons": floor,
                "relative_signal_scale": scale,
                "poisson_sample": bool(self.poisson_sample),
                "selection_rule_id": selection_rule_id,
                "dark_electrons_per_pixel": 0.0,
                "read_noise_std_electrons": 0.0,
            },
        )
        object.__setattr__(self, "electrons_per_pattern", dose)
        object.__setattr__(self, "detector_valid_mask", valid)
        for name, value in partitions.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "calibration_id", calibration_id)
        object.__setattr__(self, "minimum_expected_electrons", floor)
        object.__setattr__(self, "relative_signal_scale", scale)
        object.__setattr__(self, "poisson_sample", bool(self.poisson_sample))
        object.__setattr__(self, "selection_rule_id", selection_rule_id)
        object.__setattr__(self, "contract_id", identity)

    @property
    def selection_indices(self) -> np.ndarray:
        """Sorted train/validation rows; audit and guard rows are absent."""

        return _readonly(
            np.sort(
                np.concatenate(
                    [
                        np.asarray(self.training_indices),
                        np.asarray(self.validation_indices),
                    ]
                )
            ),
            dtype=np.int64,
        )


def make_atomistic_edit_blind_count_selection_contract_1d(
    experiment: SiliconGlancingExperiment1D,
    *,
    electrons_per_pattern: Any,
    calibration_id: str,
    detector_valid_mask: Any | None = None,
    minimum_expected_electrons: float = 1e-9,
    relative_signal_scale: float = 1.0,
    poisson_sample: bool = True,
) -> AtomisticEditBlindCountSelectionContract1D:
    """Freeze the experiment's geometry-only partition and count calibration."""

    if not isinstance(experiment, SiliconGlancingExperiment1D):
        raise TypeError("experiment must be a SiliconGlancingExperiment1D")
    shape = (len(experiment.window_starts), len(experiment.detector_angles))
    valid = (
        np.ones(shape, dtype=bool)
        if detector_valid_mask is None
        else np.asarray(detector_valid_mask)
    )
    if valid.shape != shape:
        raise ValueError("detector_valid_mask must match the experiment scan/detector shape")
    return AtomisticEditBlindCountSelectionContract1D(
        electrons_per_pattern=electrons_per_pattern,
        detector_valid_mask=valid,
        training_indices=experiment.training_indices,
        validation_indices=experiment.validation_indices,
        audit_indices=experiment.audit_indices,
        excluded_indices=experiment.guard_indices,
        calibration_id=calibration_id,
        minimum_expected_electrons=minimum_expected_electrons,
        relative_signal_scale=relative_signal_scale,
        poisson_sample=poisson_sample,
    )


@dataclass(frozen=True)
class _BoundSiliconBlindContext1D:
    experiment: SiliconGlancingExperiment1D
    model: AtomisticEditModel1D
    count_contract: AtomisticEditBlindCountSelectionContract1D
    reconstruction_contract: AtomisticEditReconstructionContract1D
    quadrature_options: AtomicTemplateQuadratureOptions1D
    truth_numerical_options: DirectAtomicNumericalOptions1D


@dataclass(frozen=True)
class _PrivateTruthRealization1D:
    potential: np.ndarray
    probe_rows: np.ndarray
    additions: ResolutionAwareMassMeasure1D
    removals: ResolutionAwareMassMeasure1D
    host_deformation_rms_A: float
    generating_addition_kernel_id: str | None = None
    generating_element: str | None = None
    mismatch_cause: str | None = None
    axial_depth_uncertainty_A: float | None = None
    slice_thickness_A: float | None = None


def _options_digest(model: AtomisticEditModel1D) -> str:
    options = model.options
    return _digest(
        arrays={
            "target_discovery": options.discovery_support.target_mask,
            "nuisance_discovery": options.discovery_support.nuisance_mask,
            "penalty_path": np.asarray(options.edit_penalty_path, dtype=np.float64),
        },
        metadata={
            "schema": "atomistic_edit_options_public_identity_1d:v1",
            "max_host_removals": options.max_host_removals,
            "max_extra_centres": options.max_extra_centres,
            "max_scattering_equivalent_per_centre": (
                options.max_scattering_equivalent_per_centre
            ),
            "minimum_separation_A": options.minimum_separation_A,
            "expected_rms_host_strain": options.expected_rms_host_strain,
            "energy_envelope": options.enable_material_energy_envelope,
            "discovery_contract_id": options.discovery_support.contract_id,
        },
    )


def _bind_context(
    experiment: SiliconGlancingExperiment1D,
    options: AtomisticEditOptions1D,
    count_contract: AtomisticEditBlindCountSelectionContract1D,
) -> _BoundSiliconBlindContext1D:
    if not isinstance(experiment, SiliconGlancingExperiment1D):
        raise TypeError("experiment must be a SiliconGlancingExperiment1D")
    if not isinstance(options, AtomisticEditOptions1D):
        raise TypeError("options must be an AtomisticEditOptions1D")
    if not isinstance(
        count_contract, AtomisticEditBlindCountSelectionContract1D
    ):
        raise TypeError(
            "count_contract must be AtomisticEditBlindCountSelectionContract1D"
        )
    n_scan = len(experiment.window_starts)
    n_detector = len(experiment.detector_angles)
    if count_contract.detector_valid_mask.shape != (n_scan, n_detector):
        raise ValueError("count contract shape disagrees with the experiment")
    for name in (
        "training_indices",
        "validation_indices",
        "audit_indices",
        "excluded_indices",
    ):
        experiment_name = "guard_indices" if name == "excluded_indices" else name
        if not np.array_equal(
            np.sort(np.asarray(getattr(count_contract, name))),
            np.sort(np.asarray(getattr(experiment, experiment_name))),
        ):
            raise ValueError(
                f"count contract {name} disagrees with the experiment partition"
            )
    if options.enable_material_energy_envelope:
        raise ValueError("AE-3 v1 keeps the material-energy arm blocked")
    if options.max_host_removals < 1:
        raise ValueError("the eight-case suite requires one host-removal slot")
    if options.max_extra_centres < 2:
        raise ValueError("the irregular-cluster case requires at least two addition slots")
    model = build_atomistic_edit_model_1d(experiment, options)
    pristine = np.asarray(experiment.pristine_potential)
    model_reference = np.asarray(model.host_model.reference_potential)
    if pristine.shape != model_reference.shape or not np.array_equal(
        pristine, model_reference
    ):
        raise ValueError(
            "experiment.pristine_potential must exactly equal the nominal host "
            "model reference used by reconstruction"
        )
    probes = np.asarray(experiment.input_probes)
    if (
        probes.shape != (n_scan, n_detector)
        or np.any(~np.isfinite(probes))
        or np.any(np.sum(np.abs(probes) ** 2, axis=1) <= 0.0)
    ):
        raise ValueError(
            "experiment.input_probes must contain one finite, nonzero row per scan"
        )
    quadrature_options = getattr(
        experiment.independent_kirkland_template, "options", None
    )
    if not isinstance(quadrature_options, AtomicTemplateQuadratureOptions1D):
        raise TypeError(
            "experiment.independent_kirkland_template.options must be an "
            "AtomicTemplateQuadratureOptions1D"
        )
    selection_hash = _digest(
        arrays={"count_contract_id": np.frombuffer(
            bytes.fromhex(count_contract.contract_id), dtype=np.uint8
        )},
        metadata={
            "solver_selection": _SOLVER_SELECTION_ID,
            "public_rows": "training_and_validation_only",
            "audit_role": "post_selection_private_evaluation_only",
        },
    )
    prior_hash = _text_digest(
        _PRIOR_ID
        + f";sigma={options.expected_rms_host_strain:.17g}"
        + f";rmin={options.minimum_separation_A:.17g}"
    )
    reconstruction_contract = AtomisticEditReconstructionContract1D(
        model_sha256=model.model_id,
        options_sha256=_options_digest(model),
        prior_sha256=prior_hash,
        selection_rule_sha256=selection_hash,
        nuisance_scope_sha256=_text_digest(_NUISANCE_SCOPE_ID),
        observability_rule_sha256=_text_digest(_OBSERVABILITY_RULE_ID),
        fitted_spatial_dimension=2,
    )
    return _BoundSiliconBlindContext1D(
        experiment=experiment,
        model=model,
        count_contract=count_contract,
        reconstruction_contract=reconstruction_contract,
        quadrature_options=quadrature_options,
        truth_numerical_options=DirectAtomicNumericalOptions1D(
            integration_method="adaptive_factorized_cubature",
            adaptive_relative_tolerance=1e-7,
            adaptive_absolute_l2_tolerance=1e-9,
            adaptive_quadrature_rule="gk21",
            adaptive_max_intervals=4096,
            adaptive_max_evaluations=500_000,
            adaptive_error_safety_factor=4.0,
        ),
    )


def _role_rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, stream]))


def _host_target_model_indices(context: _BoundSiliconBlindContext1D) -> np.ndarray:
    support = context.model.host_model.support_contract
    if support is None:
        raise ValueError("the silicon host model has no support contract")
    modeled = np.asarray(support.modeled_site_indices, dtype=np.int64)
    roles = np.asarray(support.site_role_codes, dtype=np.int64)[modeled]
    result = np.flatnonzero(roles == int(LatticeSiteRole1D.TARGET))
    if not len(result):
        raise ValueError("the host model has no reportable target silicon site")
    return result.astype(np.int64, copy=False)


def _direct_half_shape(context: _BoundSiliconBlindContext1D) -> tuple[int, int]:
    cutoff = context.quadrature_options.cutoff_A
    ds = float(context.experiment.axial_sampling)
    du = float(context.experiment.transverse_sampling)

    def support(cutoff_A: float, sampling_A: float) -> int:
        ratio = cutoff_A / sampling_A
        nearest = round(ratio)
        tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(ratio))
        return int(nearest if abs(ratio - nearest) <= tolerance else math.ceil(ratio))

    return support(cutoff, ds), support(cutoff, du)


def _fully_supported_host_indices(
    context: _BoundSiliconBlindContext1D,
    candidates: np.ndarray,
) -> np.ndarray:
    sites = np.asarray(context.model.host_model.site_coordinates, dtype=float)
    s_A = np.asarray(context.experiment.axial_coordinates, dtype=float)
    u_A = np.asarray(context.experiment.transverse_coordinates, dtype=float)
    ds = float(context.experiment.axial_sampling)
    du = float(context.experiment.transverse_sampling)
    half_s, half_u = _direct_half_shape(context)
    target = np.asarray(
        context.model.support_contract.target_discovery_mask, dtype=bool
    )
    supported = []
    for index in candidates:
        continuous_index = np.asarray(
            [
                (sites[index, 0] - s_A[0]) / ds,
                (sites[index, 1] - u_A[0]) / du,
            ],
            dtype=float,
        )
        anchor = np.floor(continuous_index + 0.5).astype(np.int64)
        if (
            half_s <= anchor[0] < len(s_A) - half_s
            and half_u <= anchor[1] < len(u_A) - half_u
            and target[anchor[0], anchor[1]]
        ):
            supported.append(int(index))
    if not supported:
        raise ValueError(
            "no target host site has full direct-Kirkland kernel containment"
        )
    return np.asarray(supported, dtype=np.int64)


def _offgrid_target_positions(
    context: _BoundSiliconBlindContext1D,
    rng: np.random.Generator,
    count: int,
) -> np.ndarray:
    target = np.asarray(
        context.model.support_contract.target_discovery_mask, dtype=bool
    )
    anchors = np.argwhere(target)
    half_s, half_u = _direct_half_shape(context)
    anchors = anchors[
        (anchors[:, 0] >= half_s)
        & (anchors[:, 0] < target.shape[0] - half_s)
        & (anchors[:, 1] >= half_u)
        & (anchors[:, 1] < target.shape[1] - half_u)
    ]
    if not len(anchors):
        raise ValueError("TARGET discovery has no full-support Kirkland anchor")
    order = rng.permutation(len(anchors))
    axes_s = np.asarray(context.experiment.axial_coordinates, dtype=float)
    axes_u = np.asarray(context.experiment.transverse_coordinates, dtype=float)
    ds = float(context.experiment.axial_sampling)
    du = float(context.experiment.transverse_sampling)
    host_support = context.model.host_model.support_contract
    if host_support is None:
        raise ValueError("the silicon host model has no complete site catalogue")
    # Hard-core truth generation must respect every catalogued pristine atom,
    # including fixed/below-budget sites retained only in the reference
    # potential, not just the locally parameterized host sites.
    hosts = np.asarray(host_support.all_site_coordinates, dtype=float)
    top_host = float(np.max(hosts[:, 1]))
    minimum = float(context.model.options.minimum_separation_A)
    selected: list[np.ndarray] = []

    def candidates(prefer_vacuum: bool) -> None:
        for raw_index in order:
            if len(selected) >= count:
                return
            anchor = anchors[raw_index]
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=2)
            fractions = rng.uniform(0.17, 0.39, size=2) * signs
            position = np.asarray(
                [axes_s[anchor[0]] + fractions[0] * ds,
                 axes_u[anchor[1]] + fractions[1] * du],
                dtype=float,
            )
            if prefer_vacuum and position[1] < top_host + 0.45 * minimum:
                continue
            if np.min(np.linalg.norm(hosts - position[None, :], axis=1)) < minimum:
                continue
            if selected and min(
                np.linalg.norm(position - other) for other in selected
            ) < minimum:
                continue
            selected.append(position)

    candidates(True)
    if len(selected) < count:
        candidates(False)
    if len(selected) < count:
        raise ValueError(
            "surface TARGET support cannot place the requested off-grid atoms "
            "with full kernels and the declared minimum separation"
        )
    return np.asarray(selected[:count], dtype=np.float64)


def _fractional_offset(
    position_A: np.ndarray, context: _BoundSiliconBlindContext1D
) -> tuple[float, float]:
    s_A = np.asarray(context.experiment.axial_coordinates, dtype=float)
    u_A = np.asarray(context.experiment.transverse_coordinates, dtype=float)
    ds = float(context.experiment.axial_sampling)
    du = float(context.experiment.transverse_sampling)
    index = np.asarray(
        [(position_A[0] - s_A[0]) / ds, (position_A[1] - u_A[0]) / du]
    )
    anchor = np.floor(index + 0.5).astype(np.int64)
    anchor_position = np.asarray([s_A[anchor[0]], u_A[anchor[1]]])
    offset = np.asarray(position_A) - anchor_position
    return float(offset[0]), float(offset[1])


def _direct_additions(
    context: _BoundSiliconBlindContext1D,
    positions_A: np.ndarray,
    elements: tuple[str, ...],
) -> tuple[np.ndarray, ResolutionAwareMassMeasure1D, str]:
    if len(positions_A) != len(elements):
        raise RuntimeError("internal direct-addition coordinate mismatch")
    grid = accumulate_weighted_atomic_potential_1d(
        positions_A,
        elements,
        np.ones(len(positions_A), dtype=np.float64),
        s_coordinates_A=context.experiment.axial_coordinates,
        u_coordinates_A=context.experiment.transverse_coordinates,
        options=context.quadrature_options,
        numerical_options=context.truth_numerical_options,
        require_full_kernel_support=True,
        metadata={"scope": "private_positive_edit_truth"},
    )
    host_integral = float(
        context.model.addition_kernel.host_equivalent_integrated_scattering
    )
    masses = []
    template_ids = []
    for position, element in zip(positions_A, elements, strict=True):
        template = render_direct_atomic_template_1d(
            element,
            sampling_s_A=context.experiment.axial_sampling,
            sampling_u_A=context.experiment.transverse_sampling,
            options=context.quadrature_options,
            numerical_options=context.truth_numerical_options,
            fractional_offset_A=_fractional_offset(position, context),
            metadata={"scope": "private_positive_edit_truth"},
        )
        masses.append(template.integrated_scattering / host_integral)
        template_ids.append(template.template_id)
    if np.any(
        np.asarray(masses)
        > context.model.options.max_scattering_equivalent_per_centre + 1e-12
    ):
        raise ValueError(
            "a direct truth atom exceeds the declared conservative one-centre "
            "host-equivalent scattering bound"
        )
    kernel_id = _digest(
        arrays={"template_ids": np.frombuffer(
            "".join(template_ids).encode("ascii"), dtype=np.uint8
        )},
        metadata={
            "renderer": "direct-weighted-Kirkland-quadrature",
            "elements": list(elements),
            "grid_id": grid.grid_id,
        },
    )
    return (
        np.asarray(grid.values, dtype=np.float64),
        ResolutionAwareMassMeasure1D(positions_A, np.asarray(masses)),
        kernel_id,
    )


def _zero_controls(context: _BoundSiliconBlindContext1D) -> np.ndarray:
    host = context.model.host_model
    return np.zeros(
        (
            len(host.control_coordinates_s),
            len(host.control_coordinates_u),
            2,
        ),
        dtype=np.float64,
    )


def _render_host(
    context: _BoundSiliconBlindContext1D,
    *,
    removal_index: int | None = None,
    controls: np.ndarray | None = None,
) -> np.ndarray:
    vacancies = np.zeros(
        len(context.model.host_model.site_coordinates), dtype=np.float64
    )
    if removal_index is not None:
        vacancies[int(removal_index)] = 1.0
    if controls is None:
        controls = _zero_controls(context)
    return np.asarray(
        render_lattice_site_potential_1d(
            context.model.host_model,
            jnp.asarray(vacancies),
            jnp.asarray(controls),
        ),
        dtype=np.float64,
    )


def _smooth_metastable_controls(
    context: _BoundSiliconBlindContext1D,
    rng: np.random.Generator,
) -> np.ndarray:
    host = context.model.host_model
    s = np.asarray(host.control_coordinates_s, dtype=float)
    u = np.asarray(host.control_coordinates_u, dtype=float)
    if len(s) * len(u) < 2:
        raise ValueError("metastable strain needs at least two host controls")
    ss, uu = np.meshgrid(s, u, indexing="ij")
    centre_s = float(rng.uniform(s[0], s[-1])) if len(s) > 1 else float(s[0])
    centre_u = float(rng.uniform(u[0], u[-1])) if len(u) > 1 else float(u[0])
    sigma_s = max(float(np.ptp(s)) / 2.5, context.experiment.axial_sampling)
    sigma_u = max(float(np.ptp(u)) / 2.5, context.experiment.transverse_sampling)
    envelope = np.exp(
        -0.5 * ((ss - centre_s) / sigma_s) ** 2
        -0.5 * ((uu - centre_u) / sigma_u) ** 2
    )
    phase = float(rng.uniform(-math.pi, math.pi))
    controls = np.stack(
        [
            envelope * np.cos((uu - centre_u) / sigma_u + phase),
            0.6 * envelope * np.sin((ss - centre_s) / sigma_s - phase),
        ],
        axis=-1,
    )
    controls -= np.mean(controls, axis=(0, 1), keepdims=True)
    maximum = float(np.asarray(host.maximum_displacement))
    target_peak = min(0.25, 0.6 * maximum)
    peak = float(np.max(np.abs(controls)))
    if peak <= np.finfo(float).eps:
        # This only occurs for a degenerate one-by-N grid; a deterministic
        # gradient remains smooth and is not a rigid translation.
        ramp = np.linspace(-1.0, 1.0, controls.size // 2).reshape(controls.shape[:-1])
        controls[..., 0] = ramp
        controls[..., 1] = -0.4 * ramp
        peak = float(np.max(np.abs(controls)))
    return controls * (target_peak / peak)


def _host_deformation_rms(
    context: _BoundSiliconBlindContext1D, controls: np.ndarray
) -> float:
    host = context.model.host_model
    displacement = lattice_site_displacements_1d(
        jnp.asarray(host.site_coordinates),
        jnp.asarray(controls),
        jnp.asarray(host.control_coordinates_s),
        jnp.asarray(host.control_coordinates_u),
    )
    values = np.asarray(displacement, dtype=float)
    return float(np.sqrt(np.mean(np.sum(values**2, axis=1))))


def _private_realization(
    role: AtomisticEditBlindCaseRole1D,
    private_seed: int,
    context: _BoundSiliconBlindContext1D,
) -> _PrivateTruthRealization1D:
    rng = _role_rng(private_seed, 0)
    pristine = np.asarray(context.experiment.pristine_potential, dtype=np.float64)
    probes = np.asarray(context.experiment.input_probes, dtype=np.complex128)
    empty = ResolutionAwareMassMeasure1D.empty(2)

    if role is AtomisticEditBlindCaseRole1D.PRISTINE_HOST:
        return _PrivateTruthRealization1D(
            potential=np.array(pristine, copy=True),
            probe_rows=np.array(probes, copy=True),
            additions=empty,
            removals=empty,
            host_deformation_rms_A=0.0,
        )

    target_hosts = _host_target_model_indices(context)
    full_hosts = _fully_supported_host_indices(context, target_hosts)
    chosen_host = int(rng.choice(full_hosts))
    host_position = np.asarray(
        context.model.host_model.site_coordinates[chosen_host], dtype=np.float64
    )

    if role is AtomisticEditBlindCaseRole1D.ONE_VACANCY:
        return _PrivateTruthRealization1D(
            potential=_render_host(context, removal_index=chosen_host),
            probe_rows=np.array(probes, copy=True),
            additions=empty,
            removals=ResolutionAwareMassMeasure1D(
                host_position[None, :], np.ones(1)
            ),
            host_deformation_rms_A=0.0,
        )

    if role is AtomisticEditBlindCaseRole1D.ONE_SUBSTITUTION:
        addition_values, additions, kernel_id = _direct_additions(
            context, host_position[None, :], ("Ge",)
        )
        return _PrivateTruthRealization1D(
            potential=_render_host(context, removal_index=chosen_host)
            + addition_values,
            probe_rows=np.array(probes, copy=True),
            additions=additions,
            removals=ResolutionAwareMassMeasure1D(
                host_position[None, :], np.ones(1)
            ),
            host_deformation_rms_A=0.0,
            generating_addition_kernel_id=kernel_id,
            generating_element="Ge",
        )

    if role is AtomisticEditBlindCaseRole1D.METASTABLE_DEFECT:
        controls = _smooth_metastable_controls(context, rng)
        deformation_rms = _host_deformation_rms(context, controls)
        if deformation_rms <= 0.0:
            raise RuntimeError("private smooth-strain generator produced zero deformation")
        return _PrivateTruthRealization1D(
            potential=_render_host(context, controls=controls),
            probe_rows=np.array(probes, copy=True),
            additions=empty,
            removals=empty,
            host_deformation_rms_A=deformation_rms,
        )

    if role is AtomisticEditBlindCaseRole1D.NUISANCE_ONLY_MISMATCH:
        detector_coordinate = np.linspace(-1.0, 1.0, probes.shape[1])
        slope = float(rng.uniform(0.12, 0.24)) * float(rng.choice([-1.0, 1.0]))
        mismatched_probes = probes * np.exp(1j * slope * detector_coordinate)[None, :]
        return _PrivateTruthRealization1D(
            potential=np.array(pristine, copy=True),
            probe_rows=mismatched_probes,
            additions=empty,
            removals=empty,
            host_deformation_rms_A=0.0,
            mismatch_cause="common_probe_phase_ramp_outside_nominal_forward_model",
        )

    if role is AtomisticEditBlindCaseRole1D.IRREGULAR_FINITE_CLUSTER:
        addition_count = min(3, context.model.options.max_extra_centres)
    else:
        addition_count = 1
    positions = _offgrid_target_positions(context, rng, addition_count)
    addition_values, additions, kernel_id = _direct_additions(
        context, positions, tuple("Si" for _ in range(addition_count))
    )
    axial_uncertainty = None
    slice_thickness = None
    if role is AtomisticEditBlindCaseRole1D.AXIALLY_UNRESOLVED_ADDITION:
        slice_thickness = float(context.experiment.axial_sampling)
        axial_uncertainty = 2.5 * slice_thickness
    return _PrivateTruthRealization1D(
        potential=np.array(pristine, copy=True) + addition_values,
        probe_rows=np.array(probes, copy=True),
        additions=additions,
        removals=empty,
        host_deformation_rms_A=0.0,
        generating_addition_kernel_id=kernel_id,
        generating_element="Si",
        axial_depth_uncertainty_A=axial_uncertainty,
        slice_thickness_A=slice_thickness,
    )


def _objective_for_indices(
    context: _BoundSiliconBlindContext1D, indices: np.ndarray
) -> PtychographyObjective1D:
    contract = context.count_contract
    return PtychographyObjective1D(
        kind="poisson_deviance",
        electrons_per_pattern=jnp.asarray(
            np.asarray(contract.electrons_per_pattern)[indices]
        ),
        minimum_expected_electrons=contract.minimum_expected_electrons,
        relative_signal_scale=contract.relative_signal_scale,
    )


def _expected_counts(
    context: _BoundSiliconBlindContext1D,
    realization: _PrivateTruthRealization1D,
    indices: np.ndarray,
) -> np.ndarray:
    experiment = context.experiment
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        intensities = simulate_glancing_scan_1d(
            jnp.asarray(realization.potential),
            jnp.asarray(realization.probe_rows[indices]),
            jnp.asarray(np.asarray(experiment.window_starts)[indices]),
            experiment.window_length,
            jnp.asarray(experiment.propagation_kernel),
            experiment.axial_sampling,
            experiment.config.energy_eV,
            rematerialize=False,
        )
        expected = ptychography_expected_signal_electrons_1d(
            intensities,
            jnp.asarray(realization.probe_rows[indices]),
            _objective_for_indices(context, indices),
        )
    result = np.asarray(expected, dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any(result < 0.0):
        raise FloatingPointError("private forward model produced invalid expected counts")
    return result


def _observed_counts(
    context: _BoundSiliconBlindContext1D,
    realization: _PrivateTruthRealization1D,
    indices: np.ndarray,
    *,
    private_seed: int,
    stream: int,
) -> tuple[np.ndarray, np.ndarray]:
    expected = _expected_counts(context, realization, indices)
    if context.count_contract.poisson_sample:
        observed = _role_rng(private_seed, stream).poisson(expected).astype(np.float64)
    else:
        observed = expected
    valid = np.asarray(context.count_contract.detector_valid_mask)[indices]
    observed = np.where(valid, observed, 0.0)
    return np.asarray(observed, dtype=np.float64), np.asarray(valid, dtype=bool)


def _private_truth(
    context: _BoundSiliconBlindContext1D,
    realization: _PrivateTruthRealization1D,
) -> AtomisticEditBlindPrivateTruth1D:
    return AtomisticEditBlindPrivateTruth1D(
        additions=realization.additions,
        removals=realization.removals,
        host_deformation_rms_A=realization.host_deformation_rms_A,
        host_kernel_id=context.model.addition_kernel.kernel_id,
        generating_addition_kernel_id=(
            realization.generating_addition_kernel_id
        ),
        generating_element=realization.generating_element,
        mismatch_cause=realization.mismatch_cause,
        axial_depth_uncertainty_A=realization.axial_depth_uncertainty_A,
        slice_thickness_A=realization.slice_thickness_A,
    )


def make_silicon_atomistic_edit_blind_cases_1d(
    experiment: SiliconGlancingExperiment1D,
    options: AtomisticEditOptions1D,
    count_contract: AtomisticEditBlindCountSelectionContract1D,
    *,
    private_seeds: Sequence[int],
) -> tuple[AtomisticEditBlindCase1D, ...]:
    """Construct the eight physical AE-3 cases with identical public schema.

    Private seeds are assigned in the frozen benchmark role-catalog order.
    They affect coordinates, count draws, strain, and mismatch realization but
    are absent from the public contract and callback input.
    """

    context = _bind_context(experiment, options, count_contract)
    if isinstance(private_seeds, (str, bytes)):
        raise TypeError("private_seeds must contain eight integer seeds")
    try:
        seeds = tuple(
            _seed(f"private_seeds[{index}]", value)
            for index, value in enumerate(private_seeds)
        )
    except TypeError as error:
        raise TypeError("private_seeds must contain eight integer seeds") from error
    roles = tuple(AtomisticEditBlindCaseRole1D)
    if len(seeds) != len(roles):
        raise ValueError("private_seeds must contain exactly eight values")
    if len(set(seeds)) != len(seeds):
        raise ValueError("private_seeds must be unique")
    selection_indices = np.asarray(count_contract.selection_indices, dtype=np.int64)
    audit_indices = np.asarray(count_contract.audit_indices, dtype=np.int64)
    cases = []
    for serial, (role, private_seed) in enumerate(zip(roles, seeds, strict=True)):
        # Only the selection-visible acquisition is materialized now.  The
        # realization is deterministic and discarded; private audit counts and
        # typed truth are independently rebuilt inside their lazy factories.
        selection_realization = _private_realization(role, private_seed, context)
        selection_counts, selection_valid = _observed_counts(
            context,
            selection_realization,
            selection_indices,
            private_seed=private_seed,
            stream=1,
        )
        del selection_realization
        public_problem = AtomisticEditBlindPublicProblem1D(
            selection_observed_total_electrons=selection_counts,
            selection_valid_mask=selection_valid,
            audit_prediction_shape=(len(audit_indices), selection_counts.shape[1]),
            contract=context.reconstruction_contract,
            public_arrays={},
            public_scalars={},
        )

        def audit_factory(
            *,
            _role: AtomisticEditBlindCaseRole1D = role,
            _private_seed: int = private_seed,
        ) -> AtomisticEditBlindAuditCounts1D:
            realization = _private_realization(_role, _private_seed, context)
            observed, valid = _observed_counts(
                context,
                realization,
                audit_indices,
                private_seed=_private_seed,
                stream=2,
            )
            return AtomisticEditBlindAuditCounts1D(observed, valid)

        def truth_factory(
            *,
            _role: AtomisticEditBlindCaseRole1D = role,
            _private_seed: int = private_seed,
        ) -> AtomisticEditBlindPrivateTruth1D:
            return _private_truth(
                context, _private_realization(_role, _private_seed, context)
            )

        cases.append(
            AtomisticEditBlindCase1D(
                role=role,
                private_case_label=f"private-silicon-ae3-{serial + 1:02d}",
                public_problem=public_problem,
                private_audit_factory=audit_factory,
                private_truth_factory=truth_factory,
            )
        )
    return tuple(cases)


def _relative_validation_indices(
    count_contract: AtomisticEditBlindCountSelectionContract1D,
) -> np.ndarray:
    selection = np.asarray(count_contract.selection_indices, dtype=np.int64)
    lookup = {int(original): index for index, original in enumerate(selection)}
    return np.asarray(
        [lookup[int(index)] for index in count_contract.validation_indices],
        dtype=np.int64,
    )


def _measure_from_state(
    context: _BoundSiliconBlindContext1D, state: AtomisticEditState1D
) -> tuple[ResolutionAwareMassMeasure1D, ResolutionAwareMassMeasure1D]:
    model = context.model
    extra_active = np.asarray(state.extra_active, dtype=bool) & (
        np.asarray(state.extra_scattering_equivalents) > 0.0
    )
    anchors = np.asarray(state.extra_anchor_indices, dtype=np.int64)
    target_extra = np.asarray(
        model.support_contract.target_discovery_mask, dtype=bool
    )[anchors[:, 0], anchors[:, 1]]
    extra_keep = extra_active & target_extra
    extra_positions = np.asarray(
        atomistic_edit_addition_positions_1d(model, state), dtype=float
    )[extra_keep]
    extra_masses = np.asarray(
        state.extra_scattering_equivalents, dtype=float
    )[extra_keep]
    additions = (
        ResolutionAwareMassMeasure1D(extra_positions, extra_masses)
        if len(extra_positions)
        else ResolutionAwareMassMeasure1D.empty(2)
    )

    support = model.host_model.support_contract
    assert support is not None
    modeled = np.asarray(support.modeled_site_indices, dtype=np.int64)
    modeled_roles = np.asarray(support.site_role_codes, dtype=np.int64)[modeled]
    removal_active = np.asarray(state.host_removal_active, dtype=bool) & (
        np.asarray(state.host_removal_fractions) > 0.0
    )
    removal_indices = np.asarray(state.host_removal_indices, dtype=np.int64)
    removal_fractions = np.asarray(state.host_removal_fractions, dtype=float)
    combined: dict[int, float] = {}
    for index, fraction, active in zip(
        removal_indices, removal_fractions, removal_active, strict=True
    ):
        if active and modeled_roles[index] == int(LatticeSiteRole1D.TARGET):
            combined[int(index)] = combined.get(int(index), 0.0) + float(fraction)
    if combined:
        ordered = sorted(combined)
        positions = np.asarray(model.host_model.site_coordinates, dtype=float)[ordered]
        masses = np.asarray([combined[index] for index in ordered], dtype=float)
        removals = ResolutionAwareMassMeasure1D(positions, masses)
    else:
        removals = ResolutionAwareMassMeasure1D.empty(2)
    return additions, removals


def _deformation_rms_for_state(
    context: _BoundSiliconBlindContext1D, state: AtomisticEditState1D
) -> float:
    return _host_deformation_rms(
        context, np.asarray(state.host_displacement_controls, dtype=float)
    )


def _support_distance(
    first: tuple[ResolutionAwareMassMeasure1D, ResolutionAwareMassMeasure1D],
    second: tuple[ResolutionAwareMassMeasure1D, ResolutionAwareMassMeasure1D],
    *,
    scale_A: float,
) -> float:
    distances = []
    for one, two in zip(first, second, strict=True):
        p = np.asarray(one.positions_A, dtype=float)
        q = np.asarray(two.positions_A, dtype=float)
        m = np.asarray(one.masses_host_equivalent, dtype=float)
        n = np.asarray(two.masses_host_equivalent, dtype=float)
        if len(p) != len(q):
            distances.append(1.0 + abs(len(p) - len(q)))
            continue
        if not len(p):
            distances.append(0.0)
            continue
        p_order = np.lexsort((p[:, 1], p[:, 0]))
        q_order = np.lexsort((q[:, 1], q[:, 0]))
        positional = float(
            np.max(np.linalg.norm(p[p_order] - q[q_order], axis=1)) / scale_A
        )
        mass = float(np.max(np.abs(m[p_order] - n[q_order])))
        distances.append(max(positional, mass))
    return float(max(distances, default=0.0))


def _multistart_evidence(
    context: _BoundSiliconBlindContext1D,
    result: AtomisticEditMultistartReconstruction1D,
) -> ActiveEditMultistartEvidence1D:
    measures = tuple(
        _measure_from_state(context, candidate.debiased_state)
        for candidate in result.candidates
    )
    count = len(measures)
    scale = max(
        float(context.experiment.axial_sampling),
        float(context.experiment.transverse_sampling),
    )
    pairwise = np.zeros((count, count), dtype=float)
    for first in range(count):
        for second in range(first + 1, count):
            value = _support_distance(
                measures[first], measures[second], scale_A=scale
            )
            pairwise[first, second] = pairwise[second, first] = value
    medoid = int(np.argmin(np.sum(pairwise, axis=1)))
    if count < 2:
        disposition = "not_assessed"
    elif result.structurally_ambiguous:
        disposition = "ambiguous"
    else:
        disposition = "identifiable"
    return ActiveEditMultistartEvidence1D(
        validation_count_deviances=tuple(
            float(candidate.penalized_validation_count_deviance)
            for candidate in result.candidates
        ),
        total_addition_masses=tuple(
            measure[0].total_mass for measure in measures
        ),
        total_removal_masses=tuple(
            measure[1].total_mass for measure in measures
        ),
        support_distance_to_medoid_resolution_units=tuple(
            float(value) for value in pairwise[:, medoid]
        ),
        selected_start_index=result.selected_start_index,
        ambiguity_disposition=disposition,
    )


def _predict_rows(
    context: _BoundSiliconBlindContext1D,
    state: AtomisticEditState1D,
    indices: np.ndarray,
) -> np.ndarray:
    experiment = context.experiment
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        potential = render_atomistic_edit_potential_1d(context.model, state)
        intensities = simulate_glancing_scan_1d(
            potential,
            jnp.asarray(np.asarray(experiment.input_probes)[indices]),
            jnp.asarray(np.asarray(experiment.window_starts)[indices]),
            experiment.window_length,
            jnp.asarray(experiment.propagation_kernel),
            experiment.axial_sampling,
            experiment.config.energy_eV,
            rematerialize=False,
        )
        expected = ptychography_expected_signal_electrons_1d(
            intensities,
            jnp.asarray(np.asarray(experiment.input_probes)[indices]),
            _objective_for_indices(context, indices),
        )
    return np.asarray(expected, dtype=np.float64)


def make_silicon_atomistic_edit_reconstruction_callback_1d(
    experiment: SiliconGlancingExperiment1D,
    options: AtomisticEditOptions1D,
    count_contract: AtomisticEditBlindCountSelectionContract1D,
    *,
    ablation: AtomisticEditAblationArm1D,
    solver_options: AtomisticEditSolverOptions1D | None = None,
    number_of_starts: int = 2,
    initial_host_control_std_A: float = 0.0,
    show_progress: bool = False,
) -> BlindCallback1D:
    """Adapt the truth-free AE-2 solver to one blind benchmark arm.

    The returned callback accepts only the generic public problem.  It does
    not consume audit observations and it emits no nuisance, observability, or
    archive evidence because AE-2 does not currently establish those claims.
    """

    context = _bind_context(experiment, options, count_contract)
    if not isinstance(ablation, AtomisticEditAblationArm1D):
        raise TypeError("ablation must be AtomisticEditAblationArm1D")
    if ablation is AtomisticEditAblationArm1D.MATERIAL_ENERGY_ENVELOPE:
        raise ValueError("the material-energy ablation is blocked_not_run in AE-3 v1")
    expected_solver_ablation = (
        "edit_only"
        if ablation is AtomisticEditAblationArm1D.COUNT_AND_EDIT
        else "level1_physical"
    )
    base_solver_options = (
        AtomisticEditSolverOptions1D()
        if solver_options is None
        else solver_options
    )
    if not isinstance(base_solver_options, AtomisticEditSolverOptions1D):
        raise TypeError("solver_options must be AtomisticEditSolverOptions1D or None")
    bound_solver_options = replace(
        base_solver_options, ablation=expected_solver_ablation
    )
    starts_count = _seed("number_of_starts", number_of_starts)
    if starts_count < 1:
        raise ValueError("number_of_starts must be positive")
    control_std = float(initial_host_control_std_A)
    if not np.isfinite(control_std) or control_std < 0.0:
        raise ValueError("initial_host_control_std_A must be finite and non-negative")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress must be Boolean")
    selection_indices = np.asarray(count_contract.selection_indices, dtype=np.int64)
    audit_indices = np.asarray(count_contract.audit_indices, dtype=np.int64)
    selection_valid = np.asarray(count_contract.detector_valid_mask)[selection_indices]
    validation_relative = _relative_validation_indices(count_contract)

    def callback(
        problem: AtomisticEditBlindPublicProblem1D,
    ) -> AtomisticEditBlindReconstruction1D:
        validate_atomistic_edit_blind_public_problem_1d(problem)
        if problem.contract != context.reconstruction_contract:
            raise ValueError("public reconstruction contract does not match this adapter")
        if problem.public_arrays or problem.public_scalars:
            raise ValueError("silicon AE-3 v1 accepts no auxiliary public object payload")
        observed = np.asarray(problem.selection_observed_total_electrons)
        valid = np.asarray(problem.selection_valid_mask)
        expected_shape = (len(selection_indices), len(experiment.detector_angles))
        if observed.shape != expected_shape or valid.shape != expected_shape:
            raise ValueError("public selection count shape disagrees with the frozen geometry")
        if not np.array_equal(valid, selection_valid):
            raise ValueError("public valid mask disagrees with the frozen calibration")
        if problem.audit_prediction_shape != (
            len(audit_indices), len(experiment.detector_angles)
        ):
            raise ValueError("public audit prediction shape disagrees with frozen geometry")
        measurement = PtychographyMeasurement1D(
            calibrated_signal_electrons=jnp.asarray(observed),
            observed_total_electrons=jnp.asarray(observed),
            valid_mask=jnp.asarray(valid),
            calibrated_dark_electrons_per_pixel=jnp.zeros_like(
                jnp.asarray(observed)
            ),
            calibrated_read_noise_std_electrons=jnp.zeros_like(
                jnp.asarray(observed)
            ),
            calibration_id=count_contract.calibration_id,
            metadata=MappingProxyType(
                {
                    "schema": "ae3-selection-visible-calibrated-counts:v1",
                    "audit_observations_present": False,
                    "truth_metadata_present": False,
                }
            ),
        )
        prepared = prepare_atomistic_edit_reconstruction_1d(
            context.model,
            jnp.asarray(np.asarray(experiment.input_probes)[selection_indices]),
            jnp.asarray(np.asarray(experiment.window_starts)[selection_indices]),
            experiment.window_length,
            experiment.propagation_kernel,
            experiment.axial_sampling,
            experiment.config.energy_eV,
            measurement,
            _objective_for_indices(context, selection_indices),
            validation_indices=validation_relative,
            audit_indices=(),
            excluded_indices=(),
        )
        multistart = run_prepared_atomistic_edit_multistart_reconstruction_1d(
            prepared,
            number_of_starts=starts_count,
            initial_host_control_std_A=control_std,
            options=bound_solver_options,
            show_progress=bool(show_progress),
        )
        selected = multistart.selected_result
        state = selected.debiased_state
        additions, removals = _measure_from_state(context, state)
        prior = atomistic_edit_prior_components_1d(
            context.model, state, selected.selected_edit_penalty
        )
        return AtomisticEditBlindReconstruction1D(
            predicted_selection_total_electrons=_predict_rows(
                context, state, selection_indices
            ),
            predicted_audit_total_electrons=_predict_rows(
                context, state, audit_indices
            ),
            additions=additions,
            removals=removals,
            deformation_parameter_count=context.model.deformation_parameter_count,
            fitted_spatial_dimension=2,
            # AE-2 retains the signed maximum directional violation (negative
            # means every dormant direction has margin).  The benchmark field
            # is the non-negative violation magnitude.
            maximum_dormant_kkt_violation=max(
                0.0, float(selected.selected_kkt.maximum_dormant_violation)
            ),
            recovered_host_deformation_rms_A=_deformation_rms_for_state(
                context, state
            ),
            multistart=_multistart_evidence(context, multistart),
            physical_metrics=PhysicalAdmissibilityMetrics1D(
                hard_core_overlap_mass=float(np.asarray(prior.hard_core_penalty)),
                host_deformation_roughness=float(
                    np.asarray(prior.elastic_penalty)
                ),
            ),
            observability=None,
            nuisance_attribution=None,
            archive_evidence=None,
        )

    return callback


def make_silicon_atomistic_edit_reconstruction_callbacks_1d(
    experiment: SiliconGlancingExperiment1D,
    options: AtomisticEditOptions1D,
    count_contract: AtomisticEditBlindCountSelectionContract1D,
    *,
    solver_options: AtomisticEditSolverOptions1D | None = None,
    number_of_starts: int = 2,
    initial_host_control_std_A: float = 0.0,
    show_progress: bool = False,
) -> Mapping[AtomisticEditAblationArm1D, BlindCallback1D]:
    """Return exactly the two executable AE-3 v1 reconstruction arms."""

    callbacks = {
        arm: make_silicon_atomistic_edit_reconstruction_callback_1d(
            experiment,
            options,
            count_contract,
            ablation=arm,
            solver_options=solver_options,
            number_of_starts=number_of_starts,
            initial_host_control_std_A=initial_host_control_std_A,
            show_progress=show_progress,
        )
        for arm in (
            AtomisticEditAblationArm1D.COUNT_AND_EDIT,
            AtomisticEditAblationArm1D.LEVEL1_PHYSICAL,
        )
    }
    return MappingProxyType(callbacks)
