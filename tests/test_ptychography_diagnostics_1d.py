"""Noise-scaled sensitivity diagnostics for lattice-site ptychography."""

from dataclasses import replace

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (  # noqa: E402
    fresnel_propagation_kernel_1d,
)
from wide_angle_propagation.ptychography_1d import (  # noqa: E402
    LatticeSiteModel1D,
    LatticeSiteReconstruction1D,
)
from wide_angle_propagation.ptychography_diagnostics_1d import (  # noqa: E402
    PoissonCountingModel1D,
    SensitivityScreenOptions1D,
    lattice_site_sensitivity_screen_1d,
    load_lattice_site_sensitivity_screen_1d,
    save_lattice_site_sensitivity_screen_1d,
    validate_poisson_counting_model_1d,
)


ENERGY = 30e3


def _model_and_result():
    shape = (9, 10)
    patch = np.asarray(
        [[0.0, 0.3, 0.0], [0.1, 2.0, 0.5], [0.0, 0.2, 0.0]],
        dtype=float,
    )
    starts = np.asarray([[1, 2], [5, 6]], dtype=np.int32)
    reference = np.full(shape, 0.02, dtype=float)
    for start in starts:
        reference[
            start[0] : start[0] + 3,
            start[1] : start[1] + 3,
        ] += patch
    sites = np.column_stack(
        [(starts[:, 0] + 1) * 0.4, (starts[:, 1] + 1) * 0.3]
    )
    model = LatticeSiteModel1D(
        reference_potential=jnp.asarray(reference),
        site_coordinates=jnp.asarray(sites),
        site_patches=jnp.asarray(np.stack([patch, patch])),
        patch_starts=jnp.asarray(starts),
        control_coordinates_s=jnp.asarray([0.0, 3.2]),
        control_coordinates_u=jnp.asarray([0.0, 2.7]),
        axial_sampling=0.4,
        transverse_sampling=0.3,
        maximum_displacement=0.5,
    )
    vacancies = np.zeros(2)
    controls = np.zeros((2, 2, 2))
    result = LatticeSiteReconstruction1D(
        potential=reference,
        initial_potential=reference,
        vacancy_fractions=vacancies,
        initial_vacancy_fractions=vacancies,
        displacement_controls=controls,
        initial_displacement_controls=controls,
        site_coordinates=sites,
        displaced_site_coordinates=sites,
        control_coordinates_s=np.asarray([0.0, 3.2]),
        control_coordinates_u=np.asarray([0.0, 2.7]),
        predicted_intensities=np.zeros((1, shape[1])),
        measured_intensities=np.zeros((1, shape[1])),
        window_starts=np.asarray([0]),
        scan_coordinates=np.asarray([0.0]),
        detector_angles=np.arange(shape[1]),
        update_history=np.asarray([0]),
        elapsed_time_history=np.asarray([0.0]),
        training_loss_history=np.asarray([0.0]),
        validation_loss_history=np.asarray([0.0]),
        best_update=0,
        metadata={"best_metric": 0.0, "audit_indices": [0]},
    )
    return model, result


def _screen(dose, *, detector_mask=None, model_and_result=None):
    model, result = (
        _model_and_result() if model_and_result is None else model_and_result
    )
    n_u = model.reference_potential.shape[1]
    u = (jnp.arange(n_u) - n_u // 2) * 0.3
    probe = jnp.exp(-0.5 * ((u + 0.1) / 0.7) ** 2) * jnp.exp(0.2j * u)
    kernel = fresnel_propagation_kernel_1d(n_u, 0.3, 0.4, ENERGY)
    return lattice_site_sensitivity_screen_1d(
        model,
        result,
        probe,
        jnp.asarray([0]),
        4,
        kernel,
        0.4,
        ENERGY,
        PoissonCountingModel1D(electrons_per_pattern=dose),
        scan_indices=[0],
        detector_mask=detector_mask,
        options=SensitivityScreenOptions1D(
            hutchinson_probes=8,
            probe_batch_size=2,
            evaluation_batch_size=1,
            seed=3,
            vacancy_standard_error_threshold=1e6,
            displacement_standard_error_threshold_A=1e6,
            maximum_relative_monte_carlo_error=10.0,
            rematerialize=False,
        ),
    )


def test_poisson_fisher_screen_scales_with_dose_and_rejects_unilluminated_site():
    low = _screen(100.0)
    high = _screen(400.0)

    np.testing.assert_allclose(
        high.fisher_blocks,
        4.0 * np.asarray(low.fisher_blocks),
        rtol=2e-10,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        high.vacancy_standard_error_lower_bound[0],
        0.5 * low.vacancy_standard_error_lower_bound[0],
        rtol=2e-10,
    )
    assert bool(high.vacancy_sensitive[0])
    assert not bool(high.site_sensitive[1])
    assert np.isinf(high.vacancy_standard_error_lower_bound[1])
    assert high.metadata["interpretation"] == (
        "conservative_screen_not_observability_certificate"
    )
    assert high.metadata["fisher_evaluation"] == (
        "local_plugin_at_reconstructed_structure"
    )


def test_sensitivity_screen_rejects_a_fully_masked_detector():
    with pytest.raises(ValueError, match="removes every"):
        _screen(100.0, detector_mask=np.zeros(10, dtype=bool))


def test_sensitivity_screen_round_trip_without_pickle(tmp_path):
    screen = _screen(100.0)
    path = tmp_path / "sensitivity.npz"
    save_lattice_site_sensitivity_screen_1d(path, screen)
    with np.load(path, allow_pickle=False) as data:
        assert all(array.dtype != object for array in data.values())
    loaded = load_lattice_site_sensitivity_screen_1d(path)
    for name in (
        "site_coordinates",
        "fisher_blocks",
        "fisher_diagonal_relative_error",
        "vacancy_standard_error_lower_bound",
        "displacement_standard_error_lower_bound_A",
        "vacancy_sensitive",
        "displacement_sensitive",
        "displacement_applicable",
        "site_sensitive",
        "scan_indices",
    ):
        np.testing.assert_allclose(getattr(loaded, name), getattr(screen, name))
    assert loaded.metadata == screen.metadata


@pytest.mark.parametrize(
    ("model", "exception", "message"),
    [
        (
            PoissonCountingModel1D(electrons_per_pattern=True),
            TypeError,
            "real numeric scalar",
        ),
        (
            PoissonCountingModel1D(
                electrons_per_pattern=100.0,
                background_electrons_per_pixel=1.0 + 0.1j,
            ),
            TypeError,
            "real numeric scalar",
        ),
        (
            PoissonCountingModel1D(
                electrons_per_pattern=100.0,
                calibrated="yes",
                calibration_id="invalid",
            ),
            TypeError,
            "must be a boolean",
        ),
        (
            PoissonCountingModel1D(
                electrons_per_pattern=100.0,
                calibrated=True,
            ),
            ValueError,
            "required when calibrated",
        ),
        (
            PoissonCountingModel1D(
                electrons_per_pattern=100.0,
                calibration_id="  ",
            ),
            ValueError,
            "must not be empty",
        ),
    ],
)
def test_counting_model_validation_is_strict(model, exception, message):
    with pytest.raises(exception, match=message):
        validate_poisson_counting_model_1d(model)


def test_sensitivity_accepts_roundoff_equivalent_ordered_coordinates():
    model, result = _model_and_result()
    rounded_sites = np.asarray(result.site_coordinates, dtype=np.float32)
    rounded_result = replace(
        result,
        site_coordinates=rounded_sites,
        displaced_site_coordinates=rounded_sites,
    )
    screen = _screen(
        100.0,
        model_and_result=(model, rounded_result),
    )
    assert screen.site_coordinates.dtype == jnp.float32


def test_sensitivity_rejects_invalid_reconstruction_state_before_differentiation():
    model, result = _model_and_result()
    invalid = replace(result, vacancy_fractions=np.asarray([1.2, 0.0]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _screen(100.0, model_and_result=(model, invalid))

    integer = replace(result, vacancy_fractions=np.asarray([0, 0], dtype=np.int32))
    with pytest.raises(TypeError, match="floating-point dtype"):
        _screen(100.0, model_and_result=(model, integer))
