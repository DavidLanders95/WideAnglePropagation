"""Basic correctness tests for maintained multislice propagation methods.

These tests use simple potentials (vacuum, uniform) that have known analytical
solutions, and verify that all propagation methods agree in easy regimes.
No GPU/cupy required; uses synthetic potentials directly.
"""
import pytest
import numpy as np

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ase")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.propagation_methods import (
    angular_spectrum_propagation_kernel,
    energy2wavelength,
    fresnel_propagation_kernel,
    simulate_fresnel_as,
    simulate_fresnel_as_jit,
    simulate_wpm,
    simulate_wpm_jit,
)

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------
ENERGY = 300e3
GPTS = (64, 64)
DZ = 2.0  # Angstrom slice thickness
SAMPLING = (0.1, 0.1)  # Angstrom pixel size
N_SLICES = 2
REST_ENERGY_EV = 510_998.95
WAVELENGTH_300KEV = 0.019687489006848795


def _beam_amplitude(psi_xy, h, k):
    """Return the normalized amplitude of one fftshifted Fourier beam."""
    ny, nx = psi_xy.shape
    spectrum = np.fft.fftshift(np.fft.fft2(psi_xy) / (nx * ny))
    return np.abs(spectrum[ny // 2 + k, nx // 2 + h])


def _make_vacuum_potential():
    """Zero potential (vacuum), shape (N_SLICES, ny, nx)."""
    return jnp.zeros((N_SLICES, *GPTS), dtype=jnp.float64)


def _make_plane_wave():
    return jnp.ones(GPTS, dtype=jnp.complex128)


# ---------------------------------------------------------------------------
# Vacuum propagation: plane wave should be preserved
# ---------------------------------------------------------------------------


def test_all_methods_match_vacuum_plane_wave_solution():
    pot = _make_vacuum_potential()
    plane_wave = _make_plane_wave()
    fresnel_kernel = fresnel_propagation_kernel(
        GPTS[0], GPTS[1], SAMPLING, z=DZ, energy=ENERGY
    )
    angular_kernel = angular_spectrum_propagation_kernel(
        GPTS[0], GPTS[1], SAMPLING, z=DZ, energy=ENERGY
    )

    exit_waves = {
        "Fresnel": simulate_fresnel_as(
            pot, plane_wave, fresnel_kernel, DZ, ENERGY
        )[0],
        "angular spectrum": simulate_fresnel_as(
            pot, plane_wave, angular_kernel, DZ, ENERGY
        )[0],
        "WPM": simulate_wpm(pot, plane_wave, DZ, ENERGY, SAMPLING)[0],
    }
    wavelength = float(energy2wavelength(ENERGY))
    expected = np.asarray(plane_wave) * np.exp(
        2j * np.pi * N_SLICES * DZ / wavelength
    )

    for name, exit_wave in exit_waves.items():
        np.testing.assert_allclose(
            np.asarray(exit_wave),
            expected,
            rtol=1e-11,
            atol=1e-11,
            err_msg=f"{name} should reproduce the analytic vacuum phase",
        )


# ---------------------------------------------------------------------------
# Uniform-medium propagation
# ---------------------------------------------------------------------------


def test_all_methods_match_uniform_medium_plane_wave_solutions():
    shape = (8, 8)
    sampling = (0.2, 0.2)
    thickness = 0.05
    n_slices = 3
    potential_value = 20_000.0
    potential = jnp.full((n_slices, *shape), potential_value)
    plane_wave = jnp.ones(shape, dtype=jnp.complex128)
    fresnel_kernel = fresnel_propagation_kernel(
        *shape, sampling, z=thickness, energy=ENERGY
    )
    angular_kernel = angular_spectrum_propagation_kernel(
        *shape, sampling, z=thickness, energy=ENERGY
    )

    fixed_kernel_results = {
        "Fresnel": simulate_fresnel_as(
            potential, plane_wave, fresnel_kernel, thickness, ENERGY
        ),
        "angular spectrum": simulate_fresnel_as(
            potential, plane_wave, angular_kernel, thickness, ENERGY
        ),
    }
    wpm_result = simulate_wpm(
        potential, plane_wave, thickness, ENERGY, sampling, n_bins=5
    )

    total_energy = REST_ENERGY_EV + ENERGY
    n_squared = (
        (total_energy + potential_value) ** 2 - REST_ENERGY_EV**2
    ) / (total_energy**2 - REST_ENERGY_EV**2)
    refractive_index = np.sqrt(n_squared)
    fixed_kernel_step = np.exp(
        1j * np.pi * (n_squared + 1.0) * thickness / WAVELENGTH_300KEV
    )
    wpm_step = np.exp(
        2j
        * np.pi
        * refractive_index
        * thickness
        / WAVELENGTH_300KEV
    )
    slice_numbers = np.arange(1, n_slices + 1)[:, None, None]
    initial = np.asarray(plane_wave)[None, ...]
    expected_fixed_wavefronts = initial * fixed_kernel_step**slice_numbers
    expected_wpm_wavefronts = initial * wpm_step**slice_numbers

    for name, (exit_wave, _, wavefronts) in fixed_kernel_results.items():
        np.testing.assert_allclose(
            np.asarray(wavefronts),
            expected_fixed_wavefronts,
            rtol=2e-6,
            atol=2e-6,
            err_msg=f"{name} should follow the paraxial uniform-medium phase",
        )
        np.testing.assert_allclose(
            np.asarray(exit_wave),
            expected_fixed_wavefronts[-1],
            rtol=2e-6,
            atol=2e-6,
        )

    wpm_exit, _, wpm_wavefronts = wpm_result
    np.testing.assert_allclose(
        np.asarray(wpm_wavefronts),
        expected_wpm_wavefronts,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(wpm_exit),
        expected_wpm_wavefronts[-1],
        rtol=2e-6,
        atol=2e-6,
    )


def test_jitted_multislice_solvers_match_eager_results():
    shape = (4, 4)
    sampling = (0.2, 0.2)
    thickness = 0.05
    y, x = jnp.mgrid[: shape[0], : shape[1]]
    first_slice = 10.0 + jnp.cos(2.0 * jnp.pi * x / shape[1])
    potential = jnp.stack([first_slice, 0.5 * first_slice])
    probe = jnp.exp(2j * jnp.pi * y / shape[0])
    kernel = angular_spectrum_propagation_kernel(
        *shape, sampling, z=thickness, energy=ENERGY
    )

    eager_fixed = simulate_fresnel_as(
        potential, probe, kernel, thickness, ENERGY
    )
    jitted_fixed = simulate_fresnel_as_jit(
        potential, probe, kernel, thickness, ENERGY
    )
    eager_wpm = simulate_wpm(
        potential, probe, thickness, ENERGY, sampling, n_bins=4
    )
    jitted_wpm = simulate_wpm_jit(
        potential, probe, thickness, ENERGY, sampling, n_bins=4
    )

    for eager, jitted in [
        (eager_fixed, jitted_fixed),
        (eager_wpm, jitted_wpm),
    ]:
        for eager_output, jitted_output in zip(eager, jitted):
            np.testing.assert_allclose(
                np.asarray(jitted_output),
                np.asarray(eager_output),
                rtol=1e-12,
                atol=1e-12,
            )


# ---------------------------------------------------------------------------
# Thin specimen: all methods should agree closely
# ---------------------------------------------------------------------------

class TestThinSpecimenAgreement:
    """For 1 unit cell, all methods should give very similar results."""

    def _make_weak_potential(self):
        """A weak, smooth periodic potential (single Fourier component)."""
        ny, nx = GPTS
        y = np.arange(ny) / ny
        x = np.arange(nx) / nx
        Y, X = np.meshgrid(y, x, indexing="ij")
        V = 5.0 * (1.0 + np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y))
        return jnp.broadcast_to(jnp.array(V), (N_SLICES, ny, nx))

    def test_all_methods_close_thin(self):
        pot = self._make_weak_potential()
        pw = _make_plane_wave()
        fk = fresnel_propagation_kernel(
            GPTS[0], GPTS[1], SAMPLING, z=DZ, energy=ENERGY
        )
        ak = angular_spectrum_propagation_kernel(
            GPTS[0], GPTS[1], SAMPLING, z=DZ, energy=ENERGY
        )

        w_fr, _, _ = simulate_fresnel_as(pot, pw, fk, DZ, ENERGY)
        w_as, _, _ = simulate_fresnel_as(pot, pw, ak, DZ, ENERGY)
        w_wpm, _, _ = simulate_wpm(pot, pw, DZ, ENERGY, SAMPLING)
        amp_fr = _beam_amplitude(np.asarray(w_fr), 0, 0)
        amp_as = _beam_amplitude(np.asarray(w_as), 0, 0)
        amp_wpm = _beam_amplitude(np.asarray(w_wpm), 0, 0)

        # All should agree within 1% for thin specimen
        ref = amp_fr
        for name, amp in [("AS", amp_as), ("WPM", amp_wpm)]:
            rel_err = abs(amp - ref) / max(abs(ref), 1e-12)
            assert rel_err < 0.01, (
                f"{name} vs Fresnel: {amp:.6f} vs {ref:.6f} (rel err {rel_err:.4f})"
            )
