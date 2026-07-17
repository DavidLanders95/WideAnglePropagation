"""Analytic tests for electron-optics and propagation helper functions."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ase")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (
    _slice_phase_grating,
    angular_spectrum_propagation_kernel,
    diffraction_intensity,
    electron_refractive_index,
    electron_refractive_index_squared,
    electron_rest_energy,
    energy2wavelength,
    fourier_propagate,
    fresnel_propagation_kernel,
    get_frequencies,
    get_polynomial_bins,
    smoothstep,
    transverse_frequency_squared,
    wpm_step_adaptive,
)


REST_ENERGY_EV = 510_998.95
REFERENCE_WAVELENGTHS = {
    100e3: 0.03701436613781811,
    200e3: 0.025079340450548007,
    300e3: 0.019687489006848795,
}


def _single_fourier_mode(shape, mode):
    ny, nx = shape
    iy, ix = np.mgrid[:ny, :nx]
    return np.exp(2j * np.pi * (mode[0] * iy / ny + mode[1] * ix / nx))


def test_electron_rest_energy_matches_reference_value():
    assert electron_rest_energy() == pytest.approx(REST_ENERGY_EV, rel=2e-6)


@pytest.mark.parametrize("energy", sorted(REFERENCE_WAVELENGTHS))
def test_relativistic_wavelength_matches_reference_value(energy):
    assert float(energy2wavelength(energy)) == pytest.approx(
        REFERENCE_WAVELENGTHS[energy], rel=2e-6
    )


def test_refractive_index_matches_kg_formula_and_basic_invariants():
    energy = 300e3
    potential = np.array([-100.0, 0.0, 100.0, 20_000.0])
    total_energy = REST_ENERGY_EV + energy
    expected_squared = (
        (total_energy + potential) ** 2 - REST_ENERGY_EV**2
    ) / (total_energy**2 - REST_ENERGY_EV**2)

    actual_squared = np.asarray(
        electron_refractive_index_squared(jnp.asarray(potential), energy)
    )
    actual = np.asarray(electron_refractive_index(jnp.asarray(potential), energy))

    assert actual.shape == potential.shape
    np.testing.assert_allclose(actual_squared, expected_squared, rtol=2e-7)
    np.testing.assert_allclose(actual**2, actual_squared, rtol=2e-15, atol=2e-15)
    assert actual[0] < 1.0
    assert actual[1] == pytest.approx(1.0)
    assert np.all(np.diff(actual) > 0.0)


def test_phase_grating_matches_paraxial_kg_interaction():
    energy = 300e3
    thickness = 0.5
    potential = np.array([[-100.0, 0.0, 20_000.0]])
    wavelength = REFERENCE_WAVELENGTHS[energy]
    total_energy = REST_ENERGY_EV + energy
    n_squared = (
        (total_energy + potential) ** 2 - REST_ENERGY_EV**2
    ) / (total_energy**2 - REST_ENERGY_EV**2)
    expected = np.exp(1j * np.pi * (n_squared - 1.0) * thickness / wavelength)

    actual = _slice_phase_grating(jnp.asarray(potential), thickness, energy)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-6, atol=2e-6)


def test_frequency_grids_use_row_column_sampling_and_cycles_per_length():
    shape = (4, 6)
    sampling = (0.5, 0.25)
    fy = np.fft.fftfreq(shape[0], d=sampling[0])
    fx = np.fft.fftfreq(shape[1], d=sampling[1])
    expected_fy, expected_fx = np.meshgrid(fy, fx, indexing="ij")

    actual_fy, actual_fx = get_frequencies(*shape, sampling)
    np.testing.assert_allclose(np.asarray(actual_fy), expected_fy, atol=1e-14)
    np.testing.assert_allclose(np.asarray(actual_fx), expected_fx, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(transverse_frequency_squared(shape, sampling)),
        expected_fy**2 + expected_fx**2,
        atol=1e-14,
    )


def test_diffraction_intensity_centers_plane_wave_peak():
    shape = (4, 6)
    wave = jnp.ones(shape, dtype=jnp.complex128)
    expected = np.zeros(shape)
    expected[shape[0] // 2, shape[1] // 2] = np.prod(shape) ** 2

    np.testing.assert_allclose(np.asarray(diffraction_intensity(wave)), expected)


def test_fourier_propagate_multiplies_selected_spatial_frequency():
    shape = (5, 6)
    mode = (1, 2)
    multiplier = -0.25 + 0.75j
    field = _single_fourier_mode(shape, mode)
    transfer = np.ones(shape, dtype=np.complex128)
    transfer[mode] = multiplier

    actual = fourier_propagate(jnp.asarray(field), jnp.asarray(transfer))
    np.testing.assert_allclose(
        np.asarray(actual), multiplier * field, rtol=1e-13, atol=1e-13
    )


def test_smoothstep_clips_and_interpolates():
    values = jnp.asarray([-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 2.0])
    expected = np.array([0.0, 0.0, 0.15625, 0.5, 0.84375, 1.0, 1.0])
    np.testing.assert_allclose(np.asarray(smoothstep(values)), expected, atol=1e-15)


def test_polynomial_bins_preserve_endpoints_and_power_spacing():
    bins = np.asarray(get_polynomial_bins(1.0, 1.08, n_bins=5, power=2.0))
    expected = np.array([1.0, 1.005, 1.02, 1.045, 1.08])

    np.testing.assert_allclose(bins, expected, rtol=1e-14, atol=1e-14)
    assert np.all(np.diff(bins) > 0.0)


def test_fresnel_kernel_matches_analytic_expression():
    shape = (4, 6)
    sampling = (0.2, 0.35)
    distance = 0.02
    energy = 200e3
    wavelength = REFERENCE_WAVELENGTHS[energy]
    fy = np.fft.fftfreq(shape[0], d=sampling[0])
    fx = np.fft.fftfreq(shape[1], d=sampling[1])
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    expected = np.exp(2j * np.pi * distance / wavelength) * np.exp(
        -1j * np.pi * wavelength * distance * (fx_grid**2 + fy_grid**2)
    )

    actual = fresnel_propagation_kernel(
        *shape, sampling, z=distance, energy=energy
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(np.abs(np.asarray(actual)), 1.0, atol=1e-14)


def test_angular_spectrum_kernel_handles_propagating_and_evanescent_modes():
    shape = (4, 4)
    sampling = (0.005, 0.005)
    distance = 0.01
    energy = 200e3
    wavelength = REFERENCE_WAVELENGTHS[energy]
    fy = np.fft.fftfreq(shape[0], d=sampling[0])
    fx = np.fft.fftfreq(shape[1], d=sampling[1])
    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    kz_squared = wavelength**-2 - fx_grid**2 - fy_grid**2
    kz = np.sqrt(kz_squared.astype(np.complex128))
    expected = np.exp(2j * np.pi * distance * kz)

    actual = np.asarray(
        angular_spectrum_propagation_kernel(
            *shape, sampling, z=distance, energy=energy
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    propagating = kz_squared >= 0.0
    evanescent = ~propagating
    assert np.any(propagating)
    assert np.any(evanescent)
    np.testing.assert_allclose(np.abs(actual[propagating]), 1.0, atol=1e-14)
    assert np.all(np.abs(actual[evanescent]) < 1.0)
    assert np.all(np.abs(actual[evanescent]) > 0.0)


def test_uniform_index_wpm_step_matches_exact_homogeneous_propagation():
    shape = (4, 6)
    sampling = (0.2, 0.25)
    mode = (1, 1)
    distance = 0.02
    energy = 200e3
    refractive_index = 1.01
    field = _single_fourier_mode(shape, mode)
    n_map = jnp.full(shape, refractive_index)

    actual, *_ = wpm_step_adaptive(
        jnp.asarray(field),
        n_map,
        distance,
        energy,
        sampling,
        n_bins=5,
    )

    fy = np.fft.fftfreq(shape[0], d=sampling[0])[mode[0]]
    fx = np.fft.fftfreq(shape[1], d=sampling[1])[mode[1]]
    k0 = 1.0 / REFERENCE_WAVELENGTHS[energy]
    kz = np.sqrt(refractive_index**2 * k0**2 - fx**2 - fy**2)
    expected = field * np.exp(2j * np.pi * distance * kz)

    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=2e-6, atol=2e-6
    )
