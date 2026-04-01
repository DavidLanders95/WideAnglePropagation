"""Comprehensive tests for the forward-only KG ODE solver.

Tests cover:
1. Mathematical correctness (vacuum, uniform, analytic solutions)
2. Carrier phase convention (match Fresnel/AS)
3. Intensity conservation (unitarity)
4. Convergence with ODE tolerance
5. Split-step comparison (expected Trotter-error scaling)
6. Stability on fine grids (high k_perp)
7. Alternating (crystal-like) potentials
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.wpm import (
    electron_refractive_index,
    energy2wavelength,
    simulate_fresnel_as,
    simulate_kg_ode_full,
    fresnel_propagation_kernel,
    angular_spectrum_propagation_kernel,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ENERGY = 200e3  # 200 keV
DZ = 2.0  # slice thickness (Å)


def _make_grid(ny, nx, dy, dx):
    x = jnp.arange(nx) * dx
    y = jnp.arange(ny) * dy
    X, Y = jnp.meshgrid(x, y)
    return X, Y


def _plane_wave(ny, nx):
    return jnp.ones((ny, nx), dtype=jnp.complex128) / jnp.sqrt(ny * nx)


def _gaussian_probe(ny, nx, dy, dx, sigma=1.5):
    X, Y = _make_grid(ny, nx, dy, dx)
    cx, cy = nx * dx / 2, ny * dy / 2
    probe = jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (sigma ** 2))
    probe = probe.astype(jnp.complex128)
    return probe / jnp.sqrt(jnp.sum(jnp.abs(probe) ** 2))


def _total_intensity(ew):
    return float(jnp.sum(jnp.abs(ew) ** 2))


# ===================================================================
# 1. Vacuum tests — analytic solutions
# ===================================================================

class TestVacuumPropagation:
    """In vacuum the envelope u(z) = const, so ψ(L) = probe·exp(ik₀L)."""

    ny, nx = 64, 64
    dy, dx = 0.15, 0.15

    def test_plane_wave_amplitude_preserved(self):
        probe = _plane_wave(self.ny, self.nx)
        pot = jnp.zeros((20, self.ny, self.nx))
        ew, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-10, atol=1e-12,
        )
        np.testing.assert_allclose(
            np.abs(np.asarray(ew)),
            np.abs(np.asarray(probe)),
            atol=1e-10,
            err_msg="Vacuum should preserve amplitude",
        )

    def test_plane_wave_carrier_phase(self):
        """Exit wave phase should be k₀·L (carrier from propagation)."""
        N = 20
        probe = _plane_wave(self.ny, self.nx)
        pot = jnp.zeros((N, self.ny, self.nx))
        ew, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-10, atol=1e-12,
        )
        k0 = 2 * jnp.pi / energy2wavelength(ENERGY)
        L = N * DZ
        expected_phase = float(k0 * L) % (2 * jnp.pi)
        actual_phase = float(jnp.angle(ew[self.ny // 2, self.nx // 2])) % (
            2 * jnp.pi
        )
        np.testing.assert_allclose(
            actual_phase, expected_phase, atol=1e-8,
            err_msg="Vacuum carrier phase should be k₀·L",
        )

    def test_vacuum_matches_fresnel(self):
        """In vacuum, ODE and Fresnel should be identical."""
        N = 20
        probe = _plane_wave(self.ny, self.nx)
        pot = jnp.zeros((N, self.ny, self.nx))
        fk = fresnel_propagation_kernel(
            self.ny, self.nx, (self.dy, self.dx), z=DZ, energy=ENERGY,
        )
        ew_f, _, _ = simulate_fresnel_as(pot, probe, fk, DZ, ENERGY)
        ew_o, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-10, atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(ew_o), np.asarray(ew_f), atol=1e-8,
            err_msg="Vacuum: ODE should match Fresnel exactly",
        )

    def test_gaussian_vacuum_intensity(self):
        """Gaussian probe in vacuum: total intensity preserved."""
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)
        pot = jnp.zeros((50, self.ny, self.nx))
        ew, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-10, atol=1e-12,
        )
        np.testing.assert_allclose(
            _total_intensity(ew), 1.0, atol=1e-8,
            err_msg="Vacuum should preserve total intensity",
        )


# ===================================================================
# 2. Uniform potential — commuting operators
# ===================================================================

class TestUniformPotential:
    """Uniform potential: [V, ∇²]=0 so split-step and ODE agree exactly."""

    ny, nx = 64, 64
    dy, dx = 0.15, 0.15

    def test_uniform_matches_fresnel(self):
        """ODE should closely match Fresnel for uniform potential (commuting)."""
        N = 50
        V_mean = 20.0  # Volts
        probe = _plane_wave(self.ny, self.nx)
        pot = jnp.full((N, self.ny, self.nx), V_mean)
        fk = fresnel_propagation_kernel(
            self.ny, self.nx, (self.dy, self.dx), z=DZ, energy=ENERGY,
        )
        ew_f, _, _ = simulate_fresnel_as(pot, probe, fk, DZ, ENERGY)
        ew_o, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-10, atol=1e-12,
        )
        rel_err = float(
            jnp.max(jnp.abs(ew_o - ew_f)) / jnp.max(jnp.abs(ew_f))
        )
        assert rel_err < 1e-3, (
            f"Uniform potential (commuting): rel error {rel_err:.2e} too large"
        )

    def test_uniform_analytic_phase(self):
        """Plane wave in uniform medium: phase = k₀·n·L approximately."""
        N = 50
        V = 20.0
        probe = _plane_wave(self.ny, self.nx)
        pot = jnp.full((N, self.ny, self.nx), V)
        ew, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-10, atol=1e-12,
        )
        k0 = 2 * jnp.pi / energy2wavelength(ENERGY)
        n = electron_refractive_index(V, ENERGY)
        L = N * DZ
        # ODE gives exact exp(i*k0*L + i*k0*(n²-1)*L/2)
        # Analytic forward KG for k_perp=0: exp(i*k0*n*L) approximately
        # These differ by O((n-1)²), which is tiny.
        analytic_phase = k0 * float(n) * L
        ode_phase = float(jnp.angle(ew[self.ny // 2, self.nx // 2]))
        # Compare modulo 2π
        diff = abs((ode_phase - analytic_phase) % (2 * np.pi))
        diff = min(diff, 2 * np.pi - diff)
        assert diff < 0.01, f"Phase error {diff:.4f} rad too large"

    def test_uniform_intensity_conservation(self):
        N = 100
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)
        pot = jnp.full((N, self.ny, self.nx), 50.0)
        ew, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-8, atol=1e-10,
        )
        np.testing.assert_allclose(
            _total_intensity(ew), 1.0, atol=1e-3,
            err_msg="Total intensity should be conserved",
        )


# ===================================================================
# 3. Carrier phase convention
# ===================================================================

class TestCarrierConvention:
    """ODE must include carrier exp(ik₀z) to match Fresnel/AS output."""

    ny, nx = 64, 64
    dy, dx = 0.15, 0.15

    def test_carrier_matches_fresnel_vacuum(self):
        """Vacuum: ODE and Fresnel should have identical carrier phase."""
        N = 10
        probe = _plane_wave(self.ny, self.nx)
        pot = jnp.zeros((N, self.ny, self.nx))

        fk = fresnel_propagation_kernel(
            self.ny, self.nx, (self.dy, self.dx), z=DZ, energy=ENERGY,
        )
        ew_f, _, _ = simulate_fresnel_as(pot, probe, fk, DZ, ENERGY)
        ew_o, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-10, atol=1e-12,
        )
        # Both should have phase k₀·N·dz at the center pixel
        phase_f = float(jnp.angle(ew_f[self.ny // 2, self.nx // 2]))
        phase_o = float(jnp.angle(ew_o[self.ny // 2, self.nx // 2]))
        np.testing.assert_allclose(
            phase_o, phase_f, atol=1e-8,
            err_msg="ODE and Fresnel should have same carrier convention",
        )

    def test_dp_matches_without_carrier_effect(self):
        """DP is |FFT|² — should not depend on the global carrier phase."""
        N = 20
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)
        pot = jnp.full((N, self.ny, self.nx), 15.0)

        fk = fresnel_propagation_kernel(
            self.ny, self.nx, (self.dy, self.dx), z=DZ, energy=ENERGY,
        )
        ew_f, _, _ = simulate_fresnel_as(pot, probe, fk, DZ, ENERGY)
        ew_o, _, dp_o, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-8, atol=1e-10,
        )
        dp_f = float(
            jnp.max(jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(ew_f))) ** 2)
        )
        dp_o_max = float(jnp.max(dp_o))
        ratio = dp_o_max / dp_f
        assert 0.95 < ratio < 1.05, (
            f"DP max ratio ODE/Fresnel = {ratio:.3f}, expected ~1"
        )


# ===================================================================
# 4. Intensity conservation
# ===================================================================

class TestIntensityConservation:
    """Forward KG with real refractive index must conserve ∑|ψ|²."""

    ny, nx = 64, 64
    dy, dx = 0.15, 0.15

    @pytest.mark.parametrize("n_slices", [10, 50, 100])
    def test_gaussian_in_potential(self, n_slices):
        X, Y = _make_grid(self.ny, self.nx, self.dy, self.dx)
        cx, cy = self.nx * self.dx / 2, self.ny * self.dy / 2
        V_col = 100.0 * jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 0.25)
        pot = jnp.stack([V_col] * n_slices)
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)

        ew, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-8, atol=1e-10,
        )
        np.testing.assert_allclose(
            _total_intensity(ew), 1.0, atol=2e-3,
            err_msg=f"Intensity not conserved for {n_slices} slices",
        )

    def test_alternating_potential_intensity(self):
        """Crystal-like alternating potential: intensity should be ~1."""
        X, Y = _make_grid(self.ny, self.nx, self.dy, self.dx)
        cx, cy = self.nx * self.dx / 2, self.ny * self.dy / 2
        slices = []
        for i in range(100):
            if i % 2 == 0:
                V = 150.0 * jnp.exp(
                    -((X - cx) ** 2 + (Y - cy) ** 2) / 0.09
                )
            else:
                V = 3.0 * jnp.ones((self.ny, self.nx))
            slices.append(V)
        pot = jnp.stack(slices)
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)

        ew, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-8, atol=1e-10,
        )
        np.testing.assert_allclose(
            _total_intensity(ew), 1.0, atol=5e-3,
            err_msg="Alternating potential: intensity should be ~1",
        )


# ===================================================================
# 5. ODE tolerance convergence
# ===================================================================

class TestToleranceConvergence:
    """Tighter tolerances should give a more accurate answer."""

    ny, nx = 64, 64
    dy, dx = 0.15, 0.15

    def test_tighter_tol_reduces_error(self):
        N = 30
        X, Y = _make_grid(self.ny, self.nx, self.dy, self.dx)
        cx = self.nx * self.dx / 2
        cy = self.ny * self.dy / 2
        V = 50.0 * jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 0.25)
        pot = jnp.stack([V] * N)
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)

        # Reference: very tight tolerance
        ew_ref, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-12, atol=1e-14,
        )
        # Very loose (should show measurable error)
        ew_loose, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-2, atol=1e-4,
        )
        # Medium
        ew_med, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-6, atol=1e-8,
        )

        err_loose = float(jnp.max(jnp.abs(ew_loose - ew_ref)))
        err_med = float(jnp.max(jnp.abs(ew_med - ew_ref)))

        assert err_med <= err_loose, (
            f"Tighter tol should reduce error: loose={err_loose:.2e}, "
            f"med={err_med:.2e}"
        )


# ===================================================================
# 6. Stability on fine grids (high k_perp)
# ===================================================================

class TestFineGridStability:
    """Fine sampling (0.1 Å) produces large k_perp; dtmax should prevent blowup."""

    ny, nx = 64, 64
    dy, dx = 0.1, 0.1  # Fine grid

    def test_fine_grid_no_blowup(self):
        """ODE on fine grid should not blow up even with loose tolerances."""
        N = 50
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)
        X, Y = _make_grid(self.ny, self.nx, self.dy, self.dx)
        cx, cy = self.nx * self.dx / 2, self.ny * self.dy / 2
        V = 80.0 * jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 0.16)
        pot = jnp.stack([V] * N)

        ew, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-5, atol=1e-7,
        )
        I = _total_intensity(ew)
        assert 0.95 < I < 1.05, (
            f"Fine grid intensity {I:.4f} suggests blowup or decay"
        )

    def test_fine_grid_dp_reasonable(self):
        """DP max on fine grid should be comparable to Fresnel DP max."""
        N = 50
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)
        X, Y = _make_grid(self.ny, self.nx, self.dy, self.dx)
        cx, cy = self.nx * self.dx / 2, self.ny * self.dy / 2
        V = 80.0 * jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 0.16)
        pot = jnp.stack([V] * N)

        fk = fresnel_propagation_kernel(
            self.ny, self.nx, (self.dy, self.dx), z=DZ, energy=ENERGY,
        )
        ew_f, _, _ = simulate_fresnel_as(pot, probe, fk, DZ, ENERGY)
        ew_o, _, dp_o, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-5, atol=1e-7,
        )

        dp_f = float(
            jnp.max(jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(ew_f))) ** 2)
        )
        dp_o_max = float(jnp.max(dp_o))
        ratio = dp_o_max / dp_f
        assert 0.5 < ratio < 2.0, (
            f"Fine grid DP ratio ODE/Fresnel = {ratio:.2f}, indicates blowup"
        )

    def test_strong_potential_stability(self):
        """High-potential (>1000V) fine grid: dtmax must account for potential eigenvalue.

        Without the potential contribution to omega_max, the step size
        exceeds Tsit5's imaginary stability radius and the solution blows up.
        """
        N = 50
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)
        X, Y = _make_grid(self.ny, self.nx, self.dy, self.dx)
        cx, cy = self.nx * self.dx / 2, self.ny * self.dy / 2

        # Strong potential: Au-like peak at ~1200V
        V = 1200.0 * jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 0.5**2))
        pot = jnp.stack([V] * N)

        fk = fresnel_propagation_kernel(
            self.ny, self.nx, (self.dy, self.dx), z=DZ, energy=ENERGY,
        )
        ew_f, _, _ = simulate_fresnel_as(pot, probe, fk, DZ, ENERGY)
        ew_o, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-5, atol=1e-7,
        )

        I_o = _total_intensity(ew_o)
        assert 0.95 < I_o < 1.05, (
            f"Strong-potential intensity {I_o:.4f} suggests blowup"
        )

        amp_ratio = float(jnp.abs(ew_o).max() / jnp.abs(ew_f).max())
        assert 0.5 < amp_ratio < 2.0, (
            f"Strong-potential amp ratio {amp_ratio:.2f} suggests instability"
        )


# ===================================================================
# 7. Split-step comparison (Trotter error scaling)
# ===================================================================

class TestSplitStepComparison:
    """ODE vs Fresnel error should scale with number of slices (Trotter error)."""

    ny, nx = 64, 64
    dy, dx = 0.15, 0.15

    def test_single_slice_small_error(self):
        """Single slice: splitting error is minimal, should nearly match."""
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)
        X, Y = _make_grid(self.ny, self.nx, self.dy, self.dx)
        cx, cy = self.nx * self.dx / 2, self.ny * self.dy / 2
        V = 50.0 * jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 0.25)
        pot = V[None, :]

        fk = fresnel_propagation_kernel(
            self.ny, self.nx, (self.dy, self.dx), z=DZ, energy=ENERGY,
        )
        ew_f, _, _ = simulate_fresnel_as(pot, probe, fk, DZ, ENERGY)
        ew_o, _, _, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-10, atol=1e-12,
        )
        rel_err = float(
            jnp.max(jnp.abs(ew_o - ew_f)) / jnp.max(jnp.abs(ew_f))
        )
        assert rel_err < 0.02, (
            f"Single-slice rel error {rel_err:.4f} too large"
        )

    def test_error_grows_with_slices(self):
        """More slices → more accumulated Trotter error (not a bug)."""
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)
        X, Y = _make_grid(self.ny, self.nx, self.dy, self.dx)
        cx, cy = self.nx * self.dx / 2, self.ny * self.dy / 2
        V = 50.0 * jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 0.25)

        fk = fresnel_propagation_kernel(
            self.ny, self.nx, (self.dy, self.dx), z=DZ, energy=ENERGY,
        )

        errors = []
        for n_slices in [5, 20, 50]:
            pot = jnp.stack([V] * n_slices)
            ew_f, _, _ = simulate_fresnel_as(pot, probe, fk, DZ, ENERGY)
            ew_o, _, _, _ = simulate_kg_ode_full(
                pot, probe, DZ, ENERGY, (self.dy, self.dx),
                rtol=1e-10, atol=1e-12,
            )
            rel_err = float(
                jnp.max(jnp.abs(ew_o - ew_f)) / jnp.max(jnp.abs(ew_f))
            )
            errors.append(rel_err)

        # Error should generally increase with more slices
        assert errors[-1] > errors[0], (
            f"Error should grow with slices: {errors}"
        )
        # But should remain bounded (not blow up)
        assert errors[-1] < 0.5, (
            f"Error for 50 slices = {errors[-1]:.2f} is unreasonably large"
        )


# ===================================================================
# 8. Slice-boundary alignment
# ===================================================================

class TestSliceBoundaryAlignment:
    """Adaptive solves must respect discontinuous slice boundaries exactly."""

    ny, nx = 32, 32
    dy, dx = 0.1, 0.1

    def test_discontinuous_stack_matches_slice_by_slice_composition(self):
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx, sigma=0.4)
        X, Y = _make_grid(self.ny, self.nx, self.dy, self.dx)
        cx, cy = self.nx * self.dx / 2, self.ny * self.dy / 2
        sigma = 0.20

        V_left = 1200.0 * jnp.exp(
            -((X - (cx - 0.5)) ** 2 + (Y - cy) ** 2) / (2 * sigma**2)
        )
        V_right = 1200.0 * jnp.exp(
            -((X - (cx + 0.5)) ** 2 + (Y - cy) ** 2) / (2 * sigma**2)
        )
        pot = jnp.stack([V_left, V_right, V_left, V_right])

        ew_full, _, _, wf_full = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-5, atol=1e-7,
        )

        state = probe
        wf_ref = []
        for i in range(pot.shape[0]):
            ew_slice, _, _, _ = simulate_kg_ode_full(
                pot[i:i + 1], state, DZ, ENERGY, (self.dy, self.dx),
                rtol=1e-5, atol=1e-7,
            )
            wf_ref.append(np.asarray(ew_slice))
            state = ew_slice

        wf_ref = np.stack(wf_ref)

        np.testing.assert_allclose(
            np.asarray(wf_full), wf_ref, rtol=1e-6, atol=1e-7,
            err_msg=(
                "Full-stack solve must match explicit slice-by-slice "
                "composition on a discontinuous potential stack"
            ),
        )
        np.testing.assert_allclose(
            np.asarray(ew_full), wf_ref[-1], rtol=1e-6, atol=1e-7,
            err_msg="Exit wave must equal the final slice-by-slice state",
        )


# ===================================================================
# 9. Wavefronts output
# ===================================================================

class TestWavefronts:
    """Check that intermediate wavefronts are sensible."""

    ny, nx = 32, 32
    dy, dx = 0.2, 0.2

    def test_wavefront_shape(self):
        N = 10
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)
        pot = jnp.zeros((N, self.ny, self.nx))
        _, _, _, wf = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
        )
        assert wf.shape == (N, self.ny, self.nx), (
            f"Wavefronts shape {wf.shape} != expected ({N}, {self.ny}, {self.nx})"
        )

    def test_last_wavefront_is_exit_wave(self):
        N = 10
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)
        pot = jnp.zeros((N, self.ny, self.nx))
        ew, _, _, wf = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-10, atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(wf[-1]), np.asarray(ew), atol=1e-10,
            err_msg="Last wavefront should equal exit wave",
        )

    def test_wavefront_intensity_along_z(self):
        """Intensity should be ~1 at every saved z."""
        N = 20
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)
        X, Y = _make_grid(self.ny, self.nx, self.dy, self.dx)
        cx, cy = self.nx * self.dx / 2, self.ny * self.dy / 2
        V = 30.0 * jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 0.25)
        pot = jnp.stack([V] * N)

        _, _, _, wf = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
            rtol=1e-8, atol=1e-10,
        )
        for i in range(N):
            I = float(jnp.sum(jnp.abs(wf[i]) ** 2))
            assert 0.99 < I < 1.01, (
                f"Intensity at slice {i}: {I:.6f} deviates from 1"
            )


# ===================================================================
# 10. Diffraction pattern
# ===================================================================

class TestDiffractionPattern:
    """DP output should be consistent with FFT of exit wave."""

    ny, nx = 32, 32
    dy, dx = 0.2, 0.2

    def test_dp_matches_manual_fft(self):
        N = 10
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx)
        pot = jnp.full((N, self.ny, self.nx), 10.0)

        ew, _, dp, _ = simulate_kg_ode_full(
            pot, probe, DZ, ENERGY, (self.dy, self.dx),
        )
        dp_manual = jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(ew))) ** 2

        np.testing.assert_allclose(
            np.asarray(dp), np.asarray(dp_manual), rtol=1e-10,
            err_msg="DP should equal |fftshift(fft2(exit_wave))|²",
        )
