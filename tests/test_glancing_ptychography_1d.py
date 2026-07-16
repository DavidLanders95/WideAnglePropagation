"""Tests for the reduced glancing-incidence forward API."""

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation import ptychography_1d as module  # noqa: E402
from wide_angle_propagation.propagation_methods import (  # noqa: E402
    fresnel_propagation_kernel_1d,
)
from wide_angle_propagation.ptychography_1d import (  # noqa: E402
    GlancingSideviewCache1D,
    normalized_amplitude_loss_1d,
    simulate_glancing_scan_1d,
    simulate_glancing_sideview_cache_1d,
)


def _problem():
    sampling = 0.25
    potential = jnp.linspace(0.0, 1.0, 48 * 32).reshape(48, 32)
    coordinates_u = (jnp.arange(32) - 16) * sampling
    centers = np.linspace(-1.0, 1.0, 5)
    probes = jnp.stack(
        [jnp.exp(-0.5 * ((coordinates_u - center) / 1.0) ** 2) for center in centers]
    ).astype(jnp.complex128)
    starts = jnp.asarray([0, 4, 8, 12, 16], dtype=jnp.int32)
    kernel = fresnel_propagation_kernel_1d(32, sampling, sampling, 5e3)
    return potential, probes, starts, kernel, sampling


def test_public_api_contains_only_forward_primitives():
    assert set(module.__all__) == {
        "GlancingSideviewCache1D",
        "normalized_amplitude_loss_1d",
        "simulate_glancing_scan_1d",
        "simulate_glancing_sideview_cache_1d",
    }
    for removed in (
        "PotentialReconstruction1D",
        "GlancingScan1D",
        "reconstruct_potential_1d",
        "save_glancing_scan_1d",
        "load_glancing_scan_1d",
    ):
        assert not hasattr(module, removed)


def test_scan_returns_one_full_detector_row_per_probe():
    potential, probes, starts, kernel, sampling = _problem()
    result = simulate_glancing_scan_1d(
        potential, probes, starts, 24, kernel, sampling, 5e3
    )
    assert result.shape == (5, 32)
    assert np.all(np.asarray(result) >= 0.0)


def test_batched_scan_matches_serial_evaluation():
    potential, probes, starts, kernel, sampling = _problem()
    batched = simulate_glancing_scan_1d(
        potential, probes, starts, 24, kernel, sampling, 5e3
    )
    serial = jnp.concatenate(
        [
            simulate_glancing_scan_1d(
                potential,
                probes[index : index + 1],
                starts[index : index + 1],
                24,
                kernel,
                sampling,
                5e3,
            )
            for index in range(len(starts))
        ],
        axis=0,
    )
    np.testing.assert_allclose(batched, serial, rtol=2e-12, atol=2e-12)


def test_scan_has_finite_direct_potential_gradient():
    potential, probes, starts, kernel, sampling = _problem()

    def objective(values):
        prediction = simulate_glancing_scan_1d(
            values, probes[:2], starts[:2], 24, kernel, sampling, 5e3
        )
        return jnp.sum(prediction * jnp.linspace(0.2, 1.0, prediction.size).reshape(prediction.shape))

    value, gradient = jax.jit(jax.value_and_grad(objective))(potential)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.linalg.norm(np.asarray(gradient)) > 0.0


def test_normalized_amplitude_loss_is_zero_for_equal_data():
    measured = jnp.asarray([[1.0, 4.0, 9.0]])
    assert float(normalized_amplitude_loss_1d(measured, measured)) < 1e-24
    with pytest.raises(ValueError, match="identical shapes"):
        normalized_amplitude_loss_1d(measured, measured[:, :2])


def test_sideview_cache_preserves_selected_scan_order_and_detector_rows():
    potential, probes, starts, kernel, sampling = _problem()
    selected = jnp.asarray([4, 1], dtype=jnp.int32)
    cache = simulate_glancing_sideview_cache_1d(
        potential,
        probes,
        starts,
        24,
        kernel,
        sampling,
        5e3,
        selected,
        transverse_coordinates=(jnp.arange(32) - 16) * sampling,
        scan_coordinates=jnp.linspace(10.0, 20.0, 5),
        axial_stride=3,
        transverse_stride=2,
    )
    assert isinstance(cache, GlancingSideviewCache1D)
    np.testing.assert_array_equal(cache.scan_indices, selected)
    assert cache.detector_intensities.shape == (2, 32)
    assert cache.sideview_intensities.shape[0] == 2
    direct = simulate_glancing_scan_1d(
        potential, probes[selected], starts[selected], 24, kernel, sampling, 5e3
    )
    np.testing.assert_allclose(cache.detector_intensities, direct, rtol=2e-6, atol=2e-6)
