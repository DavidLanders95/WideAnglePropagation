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
    simulate_fresnel_as,
    simulate_wpm,
    fresnel_propagation_kernel,
    angular_spectrum_propagation_kernel,
)
from tests.conftest import beam_amplitude_normalized


# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------
ENERGY = 300e3
GPTS = (64, 64)
DZ = 2.0  # Angstrom slice thickness
SAMPLING = (0.1, 0.1)  # Angstrom pixel size
N_SLICES = 2


def _make_vacuum_potential():
    """Zero potential (vacuum), shape (N_SLICES, ny, nx)."""
    return jnp.zeros((N_SLICES, *GPTS), dtype=jnp.float64)


def _make_constant_potential(V_volts=10.0):
    """Uniform potential (constant V everywhere)."""
    return V_volts * jnp.ones((N_SLICES, *GPTS), dtype=jnp.float64)


def _make_plane_wave():
    return jnp.ones(GPTS, dtype=jnp.complex128)


# ---------------------------------------------------------------------------
# Vacuum propagation: plane wave should be preserved
# ---------------------------------------------------------------------------

class TestVacuumPropagation:
    """Propagating a plane wave through vacuum should preserve it."""

    def test_fresnel_vacuum(self):
        pot = _make_vacuum_potential()
        pw = _make_plane_wave()
        fk = fresnel_propagation_kernel(GPTS[0], GPTS[1], SAMPLING, z=DZ, energy=ENERGY)
        exit_wave, _, _ = simulate_fresnel_as(pot, pw, fk, DZ, ENERGY)
        amp = beam_amplitude_normalized(np.asarray(exit_wave), 0, 0)
        assert abs(amp - 1.0) < 1e-6, f"Fresnel vacuum [0,0] amplitude = {amp}"

    def test_angular_spectrum_vacuum(self):
        pot = _make_vacuum_potential()
        pw = _make_plane_wave()
        ak = angular_spectrum_propagation_kernel(GPTS[0], GPTS[1], SAMPLING, z=DZ, energy=ENERGY)
        exit_wave, _, _ = simulate_fresnel_as(pot, pw, ak, DZ, ENERGY)
        amp = beam_amplitude_normalized(np.asarray(exit_wave), 0, 0)
        assert abs(amp - 1.0) < 1e-6, f"AS vacuum [0,0] amplitude = {amp}"

    def test_wpm_vacuum(self):
        pot = _make_vacuum_potential()
        pw = _make_plane_wave()
        exit_wave, _, _ = simulate_wpm(pot, pw, DZ, ENERGY, SAMPLING)
        amp = beam_amplitude_normalized(np.asarray(exit_wave), 0, 0)
        assert abs(amp - 1.0) < 1e-6, f"WPM vacuum [0,0] amplitude = {amp}"


# ---------------------------------------------------------------------------
# Constant potential: all methods should give the same phase shift
# ---------------------------------------------------------------------------

class TestConstantPotential:
    """Uniform potential produces a known global phase shift."""

    V_CONST = 20.0  # Volts

    def test_fresnel_constant_v(self):
        pot = _make_constant_potential(self.V_CONST)
        pw = _make_plane_wave()
        fk = fresnel_propagation_kernel(GPTS[0], GPTS[1], SAMPLING, z=DZ, energy=ENERGY)
        exit_wave, _, _ = simulate_fresnel_as(pot, pw, fk, DZ, ENERGY)
        # Amplitude of [0,0] beam should still be ~1
        amp = beam_amplitude_normalized(np.asarray(exit_wave), 0, 0)
        assert abs(amp - 1.0) < 1e-4, f"Fresnel const-V [0,0] amplitude = {amp}"

    def test_methods_agree_constant_v(self):
        """All methods should give similar exit waves for constant potential."""
        pot = _make_constant_potential(self.V_CONST)
        pw = _make_plane_wave()
        fk = fresnel_propagation_kernel(GPTS[0], GPTS[1], SAMPLING, z=DZ, energy=ENERGY)
        ak = angular_spectrum_propagation_kernel(GPTS[0], GPTS[1], SAMPLING, z=DZ, energy=ENERGY)

        w_fr, _, _ = simulate_fresnel_as(pot, pw, fk, DZ, ENERGY)
        w_as, _, _ = simulate_fresnel_as(pot, pw, ak, DZ, ENERGY)
        w_wpm, _, _ = simulate_wpm(pot, pw, DZ, ENERGY, SAMPLING)

        # All should produce ~same beam amplitudes for [0,0]
        amps = {
            "fresnel": beam_amplitude_normalized(np.asarray(w_fr), 0, 0),
            "as": beam_amplitude_normalized(np.asarray(w_as), 0, 0),
            "wpm": beam_amplitude_normalized(np.asarray(w_wpm), 0, 0),
        }
        for name, amp in amps.items():
            assert abs(amp - 1.0) < 1e-3, f"{name} const-V [0,0] = {amp}"


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
        fk = fresnel_propagation_kernel(GPTS[0], GPTS[1], SAMPLING, z=DZ, energy=ENERGY)
        ak = angular_spectrum_propagation_kernel(GPTS[0], GPTS[1], SAMPLING, z=DZ, energy=ENERGY)

        w_fr, _, _ = simulate_fresnel_as(pot, pw, fk, DZ, ENERGY)
        w_as, _, _ = simulate_fresnel_as(pot, pw, ak, DZ, ENERGY)
        w_wpm, _, _ = simulate_wpm(pot, pw, DZ, ENERGY, SAMPLING)
        amp_fr = beam_amplitude_normalized(np.asarray(w_fr), 0, 0)
        amp_as = beam_amplitude_normalized(np.asarray(w_as), 0, 0)
        amp_wpm = beam_amplitude_normalized(np.asarray(w_wpm), 0, 0)

        # All should agree within 1% for thin specimen
        ref = amp_fr
        for name, amp in [("AS", amp_as), ("WPM", amp_wpm)]:
            rel_err = abs(amp - ref) / max(abs(ref), 1e-12)
            assert rel_err < 0.01, (
                f"{name} vs Fresnel: {amp:.6f} vs {ref:.6f} (rel err {rel_err:.4f})"
            )
