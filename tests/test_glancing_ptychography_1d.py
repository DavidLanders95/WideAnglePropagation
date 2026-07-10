"""Focused tests for direct-potential 1D glancing ptychography."""

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (
    fresnel_propagation_kernel_1d,
    phase_grating_1d_from_projected_potential,
    simulate_glancing_fresnel_baseline_1d,
)
from wide_angle_propagation.ptychography_1d import (
    beam_path_reconstruction_region_1d,
    GlancingScan1D,
    PotentialReconstruction1D,
    load_glancing_scan_1d,
    load_glancing_sideview_cache_1d,
    load_potential_reconstruction_1d,
    normalized_amplitude_loss_1d,
    reconstruct_potential_1d,
    save_glancing_scan_1d,
    save_glancing_sideview_cache_1d,
    save_potential_reconstruction_1d,
    simulate_glancing_scan_1d,
    simulate_glancing_sideview_cache_1d,
)


ENERGY = 30e3
N_U = 32
DU = 0.25
DS = 0.5


def _probe(n_u=N_U, du=DU):
    u = (jnp.arange(n_u) - n_u // 2) * du
    return jnp.exp(-0.5 * ((u + 0.15) / 0.7) ** 2) * jnp.exp(0.35j * u)


def _potential(n_s, n_u=N_U):
    u = (jnp.arange(n_u) - n_u // 2) * DU
    slice_strength = 1.0 + 0.17 * jnp.arange(n_s)
    transverse_profile = 90.0 * jnp.exp(-0.5 * ((u - 0.2) / 0.8) ** 2)
    return slice_strength[:, None] * transverse_profile[None, :]


def _kernel(n_u=N_U, du=DU, ds=DS):
    return fresnel_propagation_kernel_1d(n_u, du, ds, ENERGY)


def test_scan_validates_starts_and_returns_full_detector():
    potential = _potential(7)
    intensities = simulate_glancing_scan_1d(
        potential,
        _probe(),
        jnp.array([0, 2, 4], dtype=jnp.int32),
        3,
        _kernel(),
        DS,
        ENERGY,
    )
    assert intensities.shape == (3, N_U)
    assert np.all(np.asarray(intensities) >= 0.0)

    with pytest.raises(ValueError):
        simulate_glancing_scan_1d(
            potential, _probe(), jnp.array([-1]), 3, _kernel(), DS, ENERGY
        )
    with pytest.raises(ValueError):
        simulate_glancing_scan_1d(
            potential, _probe(), jnp.array([5]), 3, _kernel(), DS, ENERGY
        )
    with pytest.raises((TypeError, ValueError)):
        simulate_glancing_scan_1d(
            potential, _probe(), jnp.array([0.0]), 3, _kernel(), DS, ENERGY
        )


def test_batched_scan_matches_serial_fresnel_baseline():
    potential = _potential(8)
    starts = np.array([0, 2, 5])
    window_length = 3
    batched = simulate_glancing_scan_1d(
        potential,
        _probe(),
        starts,
        window_length,
        _kernel(),
        DS,
        ENERGY,
        rematerialize=True,
    )
    serial = []
    for start in starts:
        _, intensity, _ = simulate_glancing_fresnel_baseline_1d(
            _probe(),
            potential[start : start + window_length],
            DU,
            DS,
            ENERGY,
        )
        serial.append(intensity)
    np.testing.assert_allclose(
        np.asarray(batched), np.asarray(jnp.stack(serial)), rtol=2e-12, atol=2e-12
    )


def test_fftshift_never_reorders_scans():
    potential = _potential(4)
    starts = np.array([0, 3, 1])
    unit_kernel = jnp.ones(N_U, dtype=jnp.complex128)
    actual = simulate_glancing_scan_1d(
        potential, _probe(), starts, 1, unit_kernel, DS, ENERGY
    )
    unshifted = []
    for start in starts:
        exit_wave = _probe() * phase_grating_1d_from_projected_potential(
            potential[start] * DS, ENERGY
        )
        unshifted.append(jnp.abs(jnp.fft.fft(exit_wave)) ** 2)
    expected = jnp.fft.fftshift(jnp.stack(unshifted), axes=-1)
    assert not np.allclose(np.asarray(expected[0]), np.asarray(expected[1]))
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-12)


def test_beam_path_region_tracks_scan_windows_and_overlap_counts():
    u = jnp.linspace(-1.0, 1.0, 9)
    mask, coverage = beam_path_reconstruction_region_1d(
        8,
        u,
        jnp.array([0, 2], dtype=jnp.int32),
        5,
        1.0,
        0.0,
        0.2,
        -1.0,
        radius_waists=1.0,
    )

    assert mask.shape == coverage.shape == (8, 9)
    center = int(np.argmin(np.abs(np.asarray(u))))
    np.testing.assert_array_equal(
        np.asarray(coverage[:, center]), np.array([1, 1, 2, 2, 2, 1, 1, 0])
    )
    assert np.all(np.asarray(mask[:, np.asarray(u) > 0.0]) == 0)
    assert np.all(np.asarray(mask[:7, center]))
    assert not bool(np.asarray(mask[7, center]))


def test_beam_path_region_follows_tilted_centreline():
    u = jnp.arange(-2.0, 2.01, 0.25)
    mask, coverage = beam_path_reconstruction_region_1d(
        5,
        u,
        jnp.array([0], dtype=jnp.int32),
        5,
        1.0,
        -np.arctan(0.5),
        0.01,
        -2.0,
        radius_waists=1.0,
    )

    # The midpoint convention places the centreline at u=-0.25 for s=3.
    u_index = int(np.flatnonzero(np.isclose(np.asarray(u), -0.25))[0])
    assert int(np.asarray(coverage[3, u_index])) == 1
    assert bool(np.asarray(mask[3, u_index]))


def test_masked_potential_has_zero_gradient_outside_mask():
    n_s, n_u = 5, 12
    mask = jnp.zeros((n_s, n_u), dtype=bool).at[1:4, 4:8].set(True)
    candidate = jnp.linspace(10.0, 100.0, n_s * n_u).reshape(n_s, n_u)
    probe = _probe(n_u=n_u, du=0.3)
    kernel = _kernel(n_u=n_u, du=0.3, ds=0.4)

    def objective(full_candidate):
        potential = jnp.where(mask, full_candidate, 0.0)
        intensity = simulate_glancing_scan_1d(
            potential, probe, jnp.array([0, 1]), 4, kernel, 0.4, ENERGY
        )
        return jnp.sum(jnp.sqrt(intensity + 1e-12))

    gradient = jax.grad(objective)(candidate)
    assert np.all(np.isfinite(np.asarray(gradient[mask])))
    np.testing.assert_array_equal(np.asarray(gradient[~mask]), 0.0)


def test_direct_potential_gradient_matches_finite_difference():
    n_s = 4
    u = (jnp.arange(N_U) - N_U // 2) * DU
    profile = jnp.exp(-0.5 * ((u - 0.1) / 0.8) ** 2)
    weights = jnp.linspace(0.2, 1.3, N_U) ** 2

    def objective(value):
        potential = jnp.zeros((n_s, N_U)).at[2].set(value * profile)
        intensity = simulate_glancing_scan_1d(
            potential,
            _probe(),
            jnp.array([0]),
            n_s,
            _kernel(),
            DS,
            ENERGY,
            rematerialize=True,
        )[0]
        return jnp.sum(weights * intensity) / jnp.sum(intensity)

    x0 = jnp.asarray(220.0)
    automatic = jax.grad(objective)(x0)
    step = 1e-2
    finite = (objective(x0 + step) - objective(x0 - step)) / (2 * step)
    assert np.isfinite(np.asarray(automatic))
    np.testing.assert_allclose(np.asarray(automatic), np.asarray(finite), rtol=2e-4, atol=2e-8)


def test_sideview_cache_matches_batch_detector_and_downsamples_intensity():
    potential = _potential(8)
    starts = jnp.array([0, 2, 4], dtype=jnp.int32)
    u = (jnp.arange(N_U) - N_U // 2) * DU
    cache = simulate_glancing_sideview_cache_1d(
        potential,
        _probe(),
        starts,
        4,
        _kernel(),
        DS,
        ENERGY,
        jnp.array([0, 2]),
        transverse_coordinates=u,
        axial_stride=2,
        transverse_stride=2,
    )
    expected = simulate_glancing_scan_1d(
        potential, _probe(), starts, 4, _kernel(), DS, ENERGY
    )
    np.testing.assert_allclose(
        np.asarray(cache.detector_intensities),
        np.asarray(expected[jnp.asarray([0, 2])]),
        rtol=2e-6,
        atol=2e-6,
    )
    assert cache.sideview_wavefields.shape == (2, 2, N_U // 2)
    assert cache.sideview_wavefields.dtype == jnp.complex64
    assert cache.sideview_intensities.dtype == jnp.float32
    full_mean_power = np.sum(np.asarray(cache.sideview_intensities[0])) * 4
    assert full_mean_power > 0.0


def test_scan_cache_and_potential_result_round_trip_without_pickle(tmp_path):
    scan = GlancingScan1D(
        intensities=jnp.arange(12, dtype=jnp.float64).reshape(3, 4),
        window_starts=jnp.array([0, 2, 4]),
        scan_coordinates=jnp.array([1.0, 2.0, 3.0]),
        detector_angles=jnp.linspace(-2.0, 2.0, 4),
        metadata={"energy_eV": 30_000.0},
    )
    scan_path = tmp_path / "scan.npz"
    save_glancing_scan_1d(scan_path, scan)
    with np.load(scan_path, allow_pickle=False) as raw:
        assert raw["metadata_json"].dtype.kind in {"U", "S"}
    loaded_scan = load_glancing_scan_1d(scan_path)
    np.testing.assert_allclose(loaded_scan.intensities, scan.intensities)

    potential = _potential(6, n_u=8)
    starts = jnp.array([0, 2])
    probe = _probe(n_u=8, du=0.4)
    kernel = _kernel(n_u=8, du=0.4, ds=0.5)
    cache = simulate_glancing_sideview_cache_1d(
        potential,
        probe,
        starts,
        4,
        kernel,
        DS,
        ENERGY,
        jnp.array([0, 1]),
        axial_stride=2,
        transverse_stride=2,
        metadata={"model": "truth"},
    )
    cache_path = tmp_path / "sideviews.npz"
    save_glancing_sideview_cache_1d(cache_path, cache)
    loaded_cache = load_glancing_sideview_cache_1d(cache_path)
    np.testing.assert_allclose(loaded_cache.sideview_wavefields, cache.sideview_wavefields)
    assert loaded_cache.metadata == cache.metadata

    mask = jnp.zeros((3, 4), dtype=bool).at[:, 1:3].set(True)
    result = PotentialReconstruction1D(
        potential=jnp.arange(12, dtype=jnp.float64).reshape(3, 4),
        initial_potential=jnp.ones((3, 4)),
        reconstruction_mask=mask,
        axial_coordinates=jnp.arange(3, dtype=jnp.float64),
        transverse_coordinates=jnp.arange(4, dtype=jnp.float64),
        predicted_intensities=scan.intensities,
        measured_intensities=scan.intensities,
        window_starts=scan.window_starts,
        scan_coordinates=scan.scan_coordinates,
        detector_angles=scan.detector_angles,
        update_history=jnp.array([0, 10]),
        training_loss_history=jnp.array([1.0, 0.1]),
        validation_loss_history=jnp.array([1.1, 0.2]),
        best_update=10,
        metadata={"n_unknown_pixels": 6},
    )
    result_path = tmp_path / "result.npz"
    save_potential_reconstruction_1d(result_path, result)
    loaded_result = load_potential_reconstruction_1d(result_path)
    np.testing.assert_allclose(loaded_result.potential, result.potential)
    np.testing.assert_array_equal(loaded_result.reconstruction_mask, result.reconstruction_mask)
    assert loaded_result.best_update == result.best_update


def test_reconstruction_rejects_phase_wrapping_bound():
    n_s, n_u = 5, 12
    initial = jnp.ones((n_s, n_u))
    mask = jnp.ones_like(initial, dtype=bool)
    measured = jnp.ones((2, n_u))
    with pytest.raises(ValueError, match="phase bound"):
        reconstruct_potential_1d(
            initial,
            mask,
            _probe(n_u=n_u, du=0.3),
            jnp.array([0, 1]),
            4,
            _kernel(n_u=n_u, du=0.3, ds=0.4),
            0.4,
            ENERGY,
            measured,
            potential_scale=1.0,
            potential_max=1e9,
            updates=1,
        )


def test_tiny_direct_potential_reconstruction_reduces_loss_and_recovers_shape():
    pytest.importorskip("optax", reason="the ptychography extra is not installed")
    n_s, n_u = 7, 24
    du, ds = 0.3, 0.4
    u = (jnp.arange(n_u) - n_u // 2) * du
    probe = jnp.exp(-0.5 * ((u + 0.1) / 0.65) ** 2) * jnp.exp(0.25j * u)
    kernel = fresnel_propagation_kernel_1d(n_u, du, ds, ENERGY)
    starts = jnp.arange(5)
    mask = jnp.zeros((n_s, n_u), dtype=bool)
    mask = mask.at[1:6, 9:15].set(True)
    s_profile = jnp.exp(-0.5 * ((jnp.arange(n_s) - 3.0) / 1.1) ** 2)
    u_profile = jnp.exp(-0.5 * ((u - 0.15) / 0.55) ** 2)
    target = 650.0 * s_profile[:, None] * u_profile[None, :] * mask
    initial = 60.0 * mask
    measured = simulate_glancing_scan_1d(
        target, probe, starts, 3, kernel, ds, ENERGY
    )
    initial_prediction = simulate_glancing_scan_1d(
        initial, probe, starts, 3, kernel, ds, ENERGY
    )
    initial_loss = normalized_amplitude_loss_1d(initial_prediction, measured)

    result = reconstruct_potential_1d(
        initial,
        mask,
        probe,
        starts,
        3,
        kernel,
        ds,
        ENERGY,
        measured,
        transverse_coordinates=u,
        potential_scale=500.0,
        potential_max=900.0,
        learning_rate_start=4e-2,
        learning_rate_end=5e-4,
        updates=300,
        minibatch_size=5,
        validation_interval=20,
        evaluation_batch_size=5,
        rematerialize=False,
        seed=4,
    )
    recovered_loss = normalized_amplitude_loss_1d(result.predicted_intensities, measured)
    correlation = np.corrcoef(
        np.asarray(result.potential)[np.asarray(mask)],
        np.asarray(target)[np.asarray(mask)],
    )[0, 1]
    assert float(recovered_loss) < 0.05 * float(initial_loss)
    assert correlation > 0.9
