"""Noise-aware, truth-free diagnostics for lattice-site ptychography.

The local Fisher calculation in this module is deliberately called a
*sensitivity screen*. Its standard-error bounds condition on every other
parameter component being known. It is a conservative screen of independent
single-site perturbations, not a necessary-and-sufficient test for the
smooth-control model. Correlated vacancy, strain, registration, probe, and
detector nuisance directions still require a marginalized observability audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import operator
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .ptychography_1d import (
    LatticeSiteModel1D,
    LatticeSiteReconstruction1D,
    render_lattice_site_potential_from_displacements_1d,
    simulate_glancing_scan_1d,
)


__all__ = [
    "LatticeSiteSensitivityScreen1D",
    "PoissonCountingModel1D",
    "SensitivityScreenOptions1D",
    "lattice_site_sensitivity_screen_1d",
    "load_lattice_site_sensitivity_screen_1d",
    "save_lattice_site_sensitivity_screen_1d",
    "validate_poisson_counting_model_1d",
]


Array = Any


@dataclass(frozen=True)
class PoissonCountingModel1D:
    """Expected detected-electron scale for an ideal counting detector.

    ``electrons_per_pattern`` is the expected number of signal electrons for
    the known incident-probe norm. ``background_electrons_per_pixel`` is a
    known Poisson background. Read noise, uncertain gain, saturation, and
    fitted backgrounds require a later detector-calibration model and are not
    silently approximated here. ``calibrated=True`` requires a non-empty
    ``calibration_id`` so that the claim retains explicit provenance.
    """

    electrons_per_pattern: float
    background_electrons_per_pixel: float = 0.0
    minimum_expected_electrons: float = 1e-9
    calibrated: bool = False
    calibration_id: str | None = None


@dataclass(frozen=True)
class SensitivityScreenOptions1D:
    """Monte Carlo accuracy and conservative local-sensitivity thresholds."""

    hutchinson_probes: int = 16
    probe_batch_size: int = 2
    evaluation_batch_size: int = 5
    seed: int = 0
    vacancy_standard_error_threshold: float = 0.25
    displacement_standard_error_threshold_A: float = 0.05
    maximum_relative_monte_carlo_error: float = 0.35
    vacancy_threshold: float = 0.5
    vacancy_margin: float = 0.1
    rematerialize: bool = True


@dataclass(frozen=True)
class LatticeSiteSensitivityScreen1D:
    """Conditional local Fisher screen at one reconstructed structure.

    The standard errors are optimistic lower bounds because every other site
    and every other component of the same site, along with all nuisance
    parameters, is held fixed. ``site_sensitive`` must therefore never be
    interpreted as a uniqueness or observability certificate.
    """

    site_coordinates: Array
    fisher_blocks: Array
    fisher_diagonal_relative_error: Array
    vacancy_standard_error_lower_bound: Array
    displacement_standard_error_lower_bound_A: Array
    vacancy_sensitive: Array
    displacement_sensitive: Array
    displacement_applicable: Array
    site_sensitive: Array
    scan_indices: Array
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _finite_real_scalar(
    name: str,
    value: Any,
    *,
    positive: bool,
) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or np.issubdtype(array.dtype, np.bool_)
        or np.iscomplexobj(array)
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise TypeError(f"{name} must be a real numeric scalar")
    resolved = float(array)
    invalid_sign = resolved <= 0.0 if positive else resolved < 0.0
    if not np.isfinite(resolved) or invalid_sign:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return resolved


def validate_poisson_counting_model_1d(model: PoissonCountingModel1D) -> None:
    """Validate an ideal Poisson count model used by Fisher diagnostics."""
    if not isinstance(model, PoissonCountingModel1D):
        raise TypeError("counting_model must be a PoissonCountingModel1D instance")
    _finite_real_scalar(
        "counting_model.electrons_per_pattern",
        model.electrons_per_pattern,
        positive=True,
    )
    _finite_real_scalar(
        "counting_model.minimum_expected_electrons",
        model.minimum_expected_electrons,
        positive=True,
    )
    _finite_real_scalar(
        "counting_model.background_electrons_per_pixel",
        model.background_electrons_per_pixel,
        positive=False,
    )
    if not isinstance(model.calibrated, (bool, np.bool_)):
        raise TypeError("counting_model.calibrated must be a boolean")
    if model.calibration_id is not None and not isinstance(
        model.calibration_id, str
    ):
        raise TypeError("counting_model.calibration_id must be a string or None")
    if isinstance(model.calibration_id, str) and not model.calibration_id.strip():
        raise ValueError("counting_model.calibration_id must not be empty")
    if model.calibrated and model.calibration_id is None:
        raise ValueError(
            "counting_model.calibration_id is required when calibrated=True"
        )


def _floating_host_array(name: str, value: Any, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _coordinate_tolerances(*arrays: np.ndarray) -> tuple[float, float]:
    epsilon = max(np.finfo(array.dtype).eps for array in arrays)
    scale = max(
        1.0,
        *(float(np.max(np.abs(array))) if array.size else 0.0 for array in arrays),
    )
    return 4.0 * epsilon, max(1e-12, 4.0 * epsilon * scale)


def _ordered_site_coordinates_match_1d(first: Any, second: Any) -> bool:
    first_host = np.asarray(first)
    second_host = np.asarray(second)
    if first_host.shape != second_host.shape:
        return False
    if not (
        np.issubdtype(first_host.dtype, np.floating)
        and np.issubdtype(second_host.dtype, np.floating)
    ):
        return False
    rtol, atol = _coordinate_tolerances(first_host, second_host)
    return bool(np.allclose(first_host, second_host, rtol=rtol, atol=atol))


def _validated_reconstruction_site_state_1d(
    model: LatticeSiteModel1D,
    reconstruction: LatticeSiteReconstruction1D,
) -> tuple[Array, Array, Array]:
    if not isinstance(model, LatticeSiteModel1D):
        raise TypeError("model must be a LatticeSiteModel1D instance")
    if not isinstance(reconstruction, LatticeSiteReconstruction1D):
        raise TypeError(
            "reconstruction must be a LatticeSiteReconstruction1D instance"
        )
    _floating_host_array("model.reference_potential", model.reference_potential, 2)
    model_sites = _floating_host_array(
        "model.site_coordinates", model.site_coordinates, 2
    )
    vacancies_host = _floating_host_array(
        "reconstruction.vacancy_fractions",
        reconstruction.vacancy_fractions,
        1,
    )
    sites_host = _floating_host_array(
        "reconstruction.site_coordinates", reconstruction.site_coordinates, 2
    )
    displaced_host = _floating_host_array(
        "reconstruction.displaced_site_coordinates",
        reconstruction.displaced_site_coordinates,
        2,
    )
    if sites_host.shape != (vacancies_host.size, 2):
        raise ValueError("reconstruction site arrays have incompatible shapes")
    if displaced_host.shape != sites_host.shape:
        raise ValueError("displaced_site_coordinates must have shape (n_site, 2)")
    if np.any((vacancies_host < 0.0) | (vacancies_host > 1.0)):
        raise ValueError("reconstruction vacancy fractions must lie in [0, 1]")
    if not _ordered_site_coordinates_match_1d(sites_host, model_sites):
        raise ValueError("reconstruction and model must use identical ordered sites")
    maximum_displacement = _finite_real_scalar(
        "model.maximum_displacement",
        model.maximum_displacement,
        positive=False,
    )
    total_displacement = displaced_host - sites_host
    rtol, atol = _coordinate_tolerances(sites_host, displaced_host)
    bound_tolerance = max(atol, rtol * max(maximum_displacement, 1.0))
    if np.any(np.abs(total_displacement) > maximum_displacement + bound_tolerance):
        raise ValueError(
            "reconstruction site displacements exceed model.maximum_displacement"
        )
    return (
        jnp.asarray(reconstruction.vacancy_fractions),
        jnp.asarray(reconstruction.site_coordinates),
        jnp.asarray(reconstruction.displaced_site_coordinates),
    )


def _validate_options(options: SensitivityScreenOptions1D) -> None:
    if not isinstance(options, SensitivityScreenOptions1D):
        raise TypeError("options must be a SensitivityScreenOptions1D instance")
    n_probes = operator.index(options.hutchinson_probes)
    probe_batch = operator.index(options.probe_batch_size)
    evaluation_batch = operator.index(options.evaluation_batch_size)
    if n_probes < 2 or probe_batch < 1 or evaluation_batch < 1:
        raise ValueError("probe counts and evaluation_batch_size must be positive")
    if n_probes % probe_batch:
        raise ValueError("hutchinson_probes must be divisible by probe_batch_size")
    for name in (
        "vacancy_standard_error_threshold",
        "displacement_standard_error_threshold_A",
        "maximum_relative_monte_carlo_error",
    ):
        value = float(getattr(options, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"options.{name} must be finite and positive")
    threshold = float(options.vacancy_threshold)
    margin = float(options.vacancy_margin)
    if not 0.0 < threshold < 1.0 or not 0.0 <= margin < min(
        threshold, 1.0 - threshold
    ):
        raise ValueError("vacancy_threshold and vacancy_margin are incompatible")
    if not isinstance(options.rematerialize, (bool, np.bool_)):
        raise TypeError("options.rematerialize must be a boolean")
    operator.index(options.seed)


def _scan_indices(indices: Sequence[int] | None, n_scan: int) -> np.ndarray:
    if indices is None:
        resolved = np.arange(n_scan, dtype=np.int32)
    else:
        resolved = np.asarray(indices)
        if resolved.ndim != 1 or not np.issubdtype(resolved.dtype, np.integer):
            raise TypeError("scan_indices must be a one-dimensional integer sequence")
        resolved = resolved.astype(np.int32, copy=False)
    if (
        resolved.size == 0
        or np.unique(resolved).size != resolved.size
        or np.any(resolved < 0)
        or np.any(resolved >= n_scan)
    ):
        raise ValueError("scan_indices must contain unique valid scan indices")
    return resolved


def lattice_site_sensitivity_screen_1d(
    model: LatticeSiteModel1D,
    reconstruction: LatticeSiteReconstruction1D,
    input_probe: Any,
    window_starts: Any,
    window_length: int,
    propagation_kernel: Any,
    slice_thickness: Any,
    energy: Any,
    counting_model: PoissonCountingModel1D,
    *,
    scan_indices: Sequence[int] | None = None,
    detector_mask: Any | None = None,
    options: SensitivityScreenOptions1D | None = None,
) -> LatticeSiteSensitivityScreen1D:
    """Estimate local plug-in Poisson-Fisher sensitivity without truth/data.

    The observable is ``2 * sqrt(expected electron counts)``. Its Jacobian
    Gram matrix is the expected Fisher information for independent Poisson
    counts. Hutchinson pullbacks estimate every site's local ``3 x 3`` block
    without materializing the detector-by-parameter Jacobian. Reported scalar
    lower bounds use only the block diagonal, so every other parameter
    component is conditioned as known.
    """
    options = SensitivityScreenOptions1D() if options is None else options
    validate_poisson_counting_model_1d(counting_model)
    _validate_options(options)

    vacancies, sites, displaced = _validated_reconstruction_site_state_1d(
        model, reconstruction
    )
    site_displacements = displaced - sites

    starts = jnp.asarray(window_starts)
    probes = jnp.asarray(input_probe)
    kernel = jnp.asarray(propagation_kernel)
    if starts.ndim != 1 or not jnp.issubdtype(starts.dtype, jnp.integer):
        raise TypeError("window_starts must be a one-dimensional integer array")
    n_scan = int(starts.shape[0])
    n_detector = int(model.reference_potential.shape[1])
    if probes.ndim == 1:
        if probes.shape[0] != n_detector:
            raise ValueError("input_probe must have detector length")
        probe_rows = jnp.broadcast_to(probes, (n_scan, n_detector))
    elif probes.shape == (n_scan, n_detector):
        probe_rows = probes
    else:
        raise ValueError("input_probe must be 1D or have one row per scan")
    if kernel.shape != (n_detector,):
        raise ValueError("propagation_kernel must have detector length")
    probes_host = np.asarray(probe_rows)
    kernel_host = np.asarray(kernel)
    if not np.issubdtype(probes_host.dtype, np.inexact):
        raise TypeError("input_probe must use a floating or complex dtype")
    if not np.issubdtype(kernel_host.dtype, np.inexact):
        raise TypeError("propagation_kernel must use a floating or complex dtype")
    if not np.all(np.isfinite(probes_host)):
        raise ValueError("input_probe must contain only finite values")
    if not np.all(np.isfinite(kernel_host)):
        raise ValueError("propagation_kernel must contain only finite values")
    incident_norm_host = n_detector * np.sum(
        np.abs(probes_host) ** 2, axis=1
    )
    if np.any(~np.isfinite(incident_norm_host)) or np.any(incident_norm_host <= 0.0):
        raise ValueError("every input probe must have finite positive incident norm")
    selected = _scan_indices(scan_indices, n_scan)

    if detector_mask is None:
        mask = np.ones((n_scan, n_detector), dtype=bool)
    else:
        mask = np.asarray(detector_mask)
        if mask.shape == (n_detector,):
            mask = np.broadcast_to(mask, (n_scan, n_detector))
        if mask.shape != (n_scan, n_detector) or mask.dtype != bool:
            raise TypeError(
                "detector_mask must be boolean with shape (n_detector,) or "
                "(n_scan, n_detector)"
            )
    if not np.any(mask[selected]):
        raise ValueError("detector_mask removes every selected observation")
    mask_device = jnp.asarray(mask)

    dose = jnp.asarray(
        counting_model.electrons_per_pattern, dtype=model.reference_potential.dtype
    )
    background = jnp.asarray(
        counting_model.background_electrons_per_pixel,
        dtype=model.reference_potential.dtype,
    )
    floor = jnp.asarray(
        counting_model.minimum_expected_electrons,
        dtype=model.reference_potential.dtype,
    )
    eval_batch = operator.index(options.evaluation_batch_size)
    probe_batch = operator.index(options.probe_batch_size)

    def poisson_observable(
        vacancy_values: Array,
        displacement_values: Array,
        batch_indices: Array,
        valid_scans: Array,
    ) -> Array:
        potential = render_lattice_site_potential_from_displacements_1d(
            model, vacancy_values, displacement_values
        )
        batch_probes = probe_rows[batch_indices]
        intensities = simulate_glancing_scan_1d(
            potential,
            batch_probes,
            starts[batch_indices],
            window_length,
            kernel,
            slice_thickness,
            energy,
            rematerialize=options.rematerialize,
        )
        # ``jnp.fft.fft`` is unnormalized, so Parseval gives
        # ``sum(I_detector) = n_detector * sum(|probe|**2)`` for this real,
        # elastic forward model.
        incident_norm = n_detector * jnp.sum(
            jnp.abs(batch_probes) ** 2, axis=1, keepdims=True
        )
        expected_signal = dose * intensities / incident_norm
        expected_counts = jnp.maximum(expected_signal + background, floor)
        valid = valid_scans[:, None] & mask_device[batch_indices]
        return jnp.where(valid, 2.0 * jnp.sqrt(expected_counts), 0.0)

    @jax.jit
    def pullback_batch(
        vacancy_values: Array,
        displacement_values: Array,
        batch_indices: Array,
        valid_scans: Array,
        random_vectors: Array,
    ) -> tuple[Array, Array]:
        _, pullback = jax.vjp(
            lambda vacancy, displacement: poisson_observable(
                vacancy,
                displacement,
                batch_indices,
                valid_scans,
            ),
            vacancy_values,
            displacement_values,
        )
        vacancy_gradient, displacement_gradient = jax.vmap(pullback)(
            random_vectors
        )
        return vacancy_gradient, displacement_gradient

    n_site = int(vacancies.size)
    fisher_blocks = np.zeros((n_site, 3, 3), dtype=float)
    diagonal_variance = np.zeros((n_site, 3), dtype=float)
    key = jax.random.key(operator.index(options.seed))
    for begin in range(0, len(selected), eval_batch):
        batch = selected[begin : begin + eval_batch]
        valid_count = len(batch)
        padded = np.pad(batch, (0, eval_batch - valid_count), mode="edge")
        valid = np.arange(eval_batch) < valid_count
        batch_blocks = []
        for _ in range(0, options.hutchinson_probes, probe_batch):
            key, probe_key = jax.random.split(key)
            random_vectors = jax.random.rademacher(
                probe_key,
                (probe_batch, eval_batch, n_detector),
                dtype=vacancies.dtype,
            )
            vacancy_gradient, displacement_gradient = pullback_batch(
                vacancies,
                site_displacements,
                jnp.asarray(padded),
                jnp.asarray(valid),
                random_vectors,
            )
            gradients = np.concatenate(
                [
                    np.asarray(vacancy_gradient)[..., None],
                    np.asarray(displacement_gradient),
                ],
                axis=-1,
            )
            batch_blocks.append(
                np.einsum("psi,psj->psij", gradients, gradients)
            )
        samples = np.concatenate(batch_blocks, axis=0)
        fisher_blocks += np.mean(samples, axis=0)
        diagonal_samples = np.diagonal(samples, axis1=-2, axis2=-1)
        diagonal_variance += np.var(diagonal_samples, axis=0, ddof=1) / len(samples)

    fisher_diagonal = np.diagonal(fisher_blocks, axis1=-2, axis2=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        standard_error = 1.0 / np.sqrt(fisher_diagonal)
        relative_error = np.sqrt(diagonal_variance) / fisher_diagonal
    standard_error[~np.isfinite(standard_error)] = np.inf
    relative_error[~np.isfinite(relative_error)] = np.inf
    monte_carlo_ok = (
        relative_error <= options.maximum_relative_monte_carlo_error
    )
    vacancy_sensitive = (
        standard_error[:, 0] <= options.vacancy_standard_error_threshold
    ) & monte_carlo_ok[:, 0]
    displacement_sensitive = (
        standard_error[:, 1:]
        <= options.displacement_standard_error_threshold_A
    ) & monte_carlo_ok[:, 1:]
    vacancy_host = np.asarray(vacancies)
    confidently_occupied = (
        vacancy_host < options.vacancy_threshold - options.vacancy_margin
    )
    confidently_vacant = (
        vacancy_host > options.vacancy_threshold + options.vacancy_margin
    )
    displacement_applicable = confidently_occupied
    site_sensitive = vacancy_sensitive & (
        confidently_vacant
        | (confidently_occupied & np.all(displacement_sensitive, axis=1))
    )
    return LatticeSiteSensitivityScreen1D(
        site_coordinates=sites,
        fisher_blocks=jnp.asarray(fisher_blocks),
        fisher_diagonal_relative_error=jnp.asarray(relative_error),
        vacancy_standard_error_lower_bound=jnp.asarray(standard_error[:, 0]),
        displacement_standard_error_lower_bound_A=jnp.asarray(
            standard_error[:, 1:]
        ),
        vacancy_sensitive=jnp.asarray(vacancy_sensitive),
        displacement_sensitive=jnp.asarray(displacement_sensitive),
        displacement_applicable=jnp.asarray(displacement_applicable),
        site_sensitive=jnp.asarray(site_sensitive),
        scan_indices=jnp.asarray(selected),
        metadata={
            "diagnostic": (
                "conditional_independent_site_plugin_poisson_fisher_sensitivity"
            ),
            "interpretation": (
                "conservative_screen_not_observability_certificate"
            ),
            "fisher_evaluation": "local_plugin_at_reconstructed_structure",
            "conditioned_parameters": (
                "all_other_site_components_and_nuisance_parameters_held_fixed"
            ),
            "expected_electrons_per_pattern": float(
                counting_model.electrons_per_pattern
            ),
            "poisson_background_electrons_per_pixel": float(
                counting_model.background_electrons_per_pixel
            ),
            "counting_model_calibrated": bool(counting_model.calibrated),
            "calibration_id": counting_model.calibration_id,
            "hutchinson_probes": int(options.hutchinson_probes),
            "probe_batch_size": probe_batch,
            "evaluation_batch_size": eval_batch,
            "seed": int(options.seed),
            "scan_indices": selected.tolist(),
            "masked_fraction": float(1.0 - np.mean(mask[selected])),
        },
    )


def save_lattice_site_sensitivity_screen_1d(
    path: str | Path,
    screen: LatticeSiteSensitivityScreen1D,
) -> None:
    """Save a compact sensitivity report without pickled objects."""
    if not isinstance(screen, LatticeSiteSensitivityScreen1D):
        raise TypeError("screen must be a LatticeSiteSensitivityScreen1D")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        dict(screen.metadata),
        default=lambda value: np.asarray(value).tolist(),
        sort_keys=True,
    )
    np.savez_compressed(
        destination,
        schema_version=np.asarray(1, dtype=np.int64),
        site_coordinates=np.asarray(screen.site_coordinates),
        fisher_blocks=np.asarray(screen.fisher_blocks),
        fisher_diagonal_relative_error=np.asarray(
            screen.fisher_diagonal_relative_error
        ),
        vacancy_standard_error_lower_bound=np.asarray(
            screen.vacancy_standard_error_lower_bound
        ),
        displacement_standard_error_lower_bound_A=np.asarray(
            screen.displacement_standard_error_lower_bound_A
        ),
        vacancy_sensitive=np.asarray(screen.vacancy_sensitive),
        displacement_sensitive=np.asarray(screen.displacement_sensitive),
        displacement_applicable=np.asarray(screen.displacement_applicable),
        site_sensitive=np.asarray(screen.site_sensitive),
        scan_indices=np.asarray(screen.scan_indices),
        metadata_json=np.asarray(metadata),
    )


def load_lattice_site_sensitivity_screen_1d(
    path: str | Path,
) -> LatticeSiteSensitivityScreen1D:
    """Load a sensitivity report written by the matching save helper."""
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema_version"].item()) != 1:
            raise ValueError("unsupported sensitivity-screen schema version")
        return LatticeSiteSensitivityScreen1D(
            site_coordinates=jnp.asarray(data["site_coordinates"]),
            fisher_blocks=jnp.asarray(data["fisher_blocks"]),
            fisher_diagonal_relative_error=jnp.asarray(
                data["fisher_diagonal_relative_error"]
            ),
            vacancy_standard_error_lower_bound=jnp.asarray(
                data["vacancy_standard_error_lower_bound"]
            ),
            displacement_standard_error_lower_bound_A=jnp.asarray(
                data["displacement_standard_error_lower_bound_A"]
            ),
            vacancy_sensitive=jnp.asarray(data["vacancy_sensitive"]),
            displacement_sensitive=jnp.asarray(data["displacement_sensitive"]),
            displacement_applicable=jnp.asarray(data["displacement_applicable"]),
            site_sensitive=jnp.asarray(data["site_sensitive"]),
            scan_indices=jnp.asarray(data["scan_indices"]),
            metadata=json.loads(str(data["metadata_json"].item())),
        )
