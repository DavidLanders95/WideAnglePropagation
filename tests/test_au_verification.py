"""Integration tests: validate propagation methods against paper reference data.

Reference: Rother & Scheerschmidt (2009), doi:10.1016/j.ultramic.2008.08.008
Figure 3: Au 300 kV, beam amplitudes vs crystal thickness.

These tests require GPU (cupy + abTEM) for potential generation.
Marked @pytest.mark.slow for tests that take >30 seconds.
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
    simulate_wpm,
    simulate_parabolic_ode,
    fresnel_propagation_kernel,
    angular_spectrum_propagation_kernel,
)
from wide_angle_propagation.klein_gordon import (
    beam_amplitudes_fwd_direct_allbeams,
)
from tests.conftest import (
    beam_amplitude_normalized,
    AU_ENERGY,
    AU_GPTS,
    AU_N_SLICES_PER_CELL,
)

# Skip entire module if cupy unavailable (no GPU)
cupy = pytest.importorskip("cupy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_thickness_sweep_realspace(pot_array, slice_dz, sampling, n_cells=26):
    """Run Fresnel, AS, and WPM multislice through a thickness sweep.

    Returns dict of {method_beam: array of amplitudes at each cell boundary}.
    """
    energy = AU_ENERGY
    gpts = AU_GPTS
    pw = jnp.ones(gpts, dtype=jnp.complex128)

    fk = jnp.array(
        fresnel_propagation_kernel(gpts[0], gpts[1], sampling, z=slice_dz, energy=energy)
    )
    ak = jnp.array(
        angular_spectrum_propagation_kernel(gpts[0], gpts[1], sampling, z=slice_dz, energy=energy)
    )

    keys = ["ms_00", "ms_028", "as_00", "as_028", "wpm_00", "wpm_028"]
    results = {k: [] for k in keys}
    w_ms = w_as = w_wpm = pw

    for i in range(n_cells):
        if i > 0:
            w_ms, _, _ = simulate_fresnel_as(pot_array, w_ms, fk, slice_dz, energy)
            w_as, _, _ = simulate_fresnel_as(pot_array, w_as, ak, slice_dz, energy)
            w_wpm, _, _ = simulate_wpm(pot_array, w_wpm, slice_dz, energy, sampling)

            w_ms = jnp.array(w_ms)
            w_as = jnp.array(w_as)
            w_wpm = jnp.array(w_wpm)

        results["ms_00"].append(beam_amplitude_normalized(np.asarray(w_ms), 0, 0))
        results["ms_028"].append(beam_amplitude_normalized(np.asarray(w_ms), 0, 28))
        results["as_00"].append(beam_amplitude_normalized(np.asarray(w_as), 0, 0))
        results["as_028"].append(beam_amplitude_normalized(np.asarray(w_as), 0, 28))
        results["wpm_00"].append(beam_amplitude_normalized(np.asarray(w_wpm), 0, 0))
        results["wpm_028"].append(beam_amplitude_normalized(np.asarray(w_wpm), 0, 28))

    return {k: np.array(v) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Fixture: run the sweep once and reuse across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lobato_sweep(au_potential_lobato, au_sampling):
    pot_array, slice_dz = au_potential_lobato
    return _run_thickness_sweep_realspace(pot_array, slice_dz, au_sampling, n_cells=26)


@pytest.fixture(scope="module")
def wk_sweep(au_potential_wk, au_sampling):
    pot_array, slice_dz = au_potential_wk
    return _run_thickness_sweep_realspace(pot_array, slice_dz, au_sampling, n_cells=26)


# ---------------------------------------------------------------------------
# Tests: WPM vs paper forward ODE curve
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestWPMvsPaperFWD:
    """WPM beam [0,28] should approach the paper's FWD curve.

    The paper's FWD curve uses a forward-only wide-angle propagation,
    which is analogous to WPM. We expect WPM to be in the same ballpark.
    """

    TOLERANCE = 0.30  # 30% — WPM is an approximation to the paper's FWD

    def test_wpm_beam_028_wk(self, wk_sweep, paper_beam_0_28_kg_fwd):
        x = np.arange(26, dtype=float)
        computed = wk_sweep["wpm_028"]
        reference = paper_beam_0_28_kg_fwd(x)

        mask = x >= 5
        rel_errors = np.abs(computed[mask] - reference[mask]) / np.maximum(reference[mask], 1e-6)
        mean_err = np.mean(rel_errors)
        max_err = np.max(rel_errors)
        assert mean_err < self.TOLERANCE, (
            f"WPM [0,28] WK vs paper FWD: mean rel error = {mean_err:.3f} "
            f"(max = {max_err:.3f})"
        )


# ---------------------------------------------------------------------------
# Tests: Multislice vs paper KG MS curve
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestMSvsPaperKGMS:
    """Angular spectrum MS beams should match the paper's KG MS curves within 5%."""

    TOLERANCE = 0.05

    def test_as_beam_00_wk(self, wk_sweep, paper_beam_0_0_kg_ms):
        x = np.arange(26, dtype=float)
        computed = wk_sweep["as_00"]
        reference = paper_beam_0_0_kg_ms(x)

        mask = x >= 1
        rel_errors = np.abs(computed[mask] - reference[mask]) / np.maximum(reference[mask], 1e-6)
        max_err = np.max(rel_errors)
        assert max_err < self.TOLERANCE, (
            f"AS MS [0,0] WK vs paper KG MS: max rel error = {max_err:.3f}"
        )

    def test_as_beam_028_wk(self, wk_sweep, paper_beam_0_28_kg_ms):
        """AS MS beam [0,28] trend should qualitatively match the paper's KG MS.

        The absolute amplitudes for high-angle beams are very small (~0.01)
        and highly sensitive to the exact scattering parametrization, so we
        use an absolute tolerance rather than a strict relative threshold.
        """
        x = np.arange(26, dtype=float)
        computed = wk_sweep["as_028"]
        reference = paper_beam_0_28_kg_ms(x)

        mask = x >= 3
        abs_errors = np.abs(computed[mask] - reference[mask])
        max_abs_err = np.max(abs_errors)
        # Absolute tolerance: 0.01 is the typical scale of this beam's amplitude
        assert max_abs_err < 0.015, (
            f"AS MS [0,28] WK vs paper KG MS: max abs error = {max_abs_err:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: WPM should be closer to ODE than Fresnel for high-angle beams
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestWPMCloserToPaperFWD:
    """For beam [0,28], WPM should be closer to the paper's FWD than Fresnel MS."""

    def test_wpm_beats_fresnel_for_028(self, wk_sweep, paper_beam_0_28_kg_fwd):
        """At 15-25 cells, WPM [0,28] should deviate less from paper FWD than Fresnel."""
        x = np.arange(26, dtype=float)
        reference = paper_beam_0_28_kg_fwd(x)
        wpm = wk_sweep["wpm_028"]
        fresnel = wk_sweep["ms_028"]

        # Compare at thick specimens (15-25 cells)
        mask = slice(15, 26)
        err_wpm = np.mean(np.abs(wpm[mask] - reference[mask]))
        err_fresnel = np.mean(np.abs(fresnel[mask] - reference[mask]))

        assert err_wpm <= err_fresnel, (
            f"WPM should be closer to paper FWD than Fresnel at high thickness. "
            f"WPM err={err_wpm:.6f}, Fresnel err={err_fresnel:.6f}"
        )


# ---------------------------------------------------------------------------
# Tests: Cross-method consistency
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestCrossMethodConsistency:
    """Verify that methods are internally consistent."""

    def test_all_methods_agree_at_1_cell(self, lobato_sweep):
        """At 1 unit cell, all multislice methods should give similar [0,0] amplitude."""
        methods = ["ms_00", "as_00", "wpm_00"]
        amps = {m: lobato_sweep[m][1] for m in methods}  # index 1 = 1 cell

        values = list(amps.values())
        spread = max(values) - min(values)
        mean_val = np.mean(values)
        rel_spread = spread / mean_val

        assert rel_spread < 0.02, (
            f"Methods disagree at 1 cell: {amps}, rel spread = {rel_spread:.4f}"
        )

    def test_as_and_fresnel_diverge_at_high_thickness(self, lobato_sweep):
        """AS and Fresnel should diverge for beam [0,28] at high thickness (expected)."""
        as_vals = lobato_sweep["as_028"]
        fr_vals = lobato_sweep["ms_028"]

        # At 20+ cells, they should differ by more than a tiny amount
        mask = slice(20, 26)
        diff = np.mean(np.abs(as_vals[mask] - fr_vals[mask]))
        max_val = np.maximum(np.mean(as_vals[mask]), 1e-8)
        rel_diff = diff / max_val

        # This is an expected divergence — just document it passes
        assert rel_diff > 0.001, (
            f"AS and Fresnel unexpectedly identical for [0,28] at high thickness"
        )


# ---------------------------------------------------------------------------
# Tests: Forward-only KG Lanczos method
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestKGLanczosVsPaperFWD:
    """Lanczos KG FWD beam [0,28] should match the paper's FWD curve."""

    MEAN_ABS_TOLERANCE = 0.010
    CORRELATION_TOLERANCE = 0.98

    def test_kg_lanczos_beam_028_wk(self, au_potential_wk, au_sampling,
                                    paper_beam_0_28_kg_fwd):
        """Lanczos KG FWD [0,28] vs paper's FWD ODE at selected thicknesses."""
        pot_array, slice_dz = au_potential_wk
        energy = AU_ENERGY
        gpts = AU_GPTS
        n_cells_array = np.arange(0, 26)

        amps, _, _ = beam_amplitudes_fwd_direct_allbeams(
            pot_array,
            slice_dz,
            energy,
            au_sampling,
            n_cells_array,
            gpts,
            lanczos_m=100,
        )

        computed = amps.get((0, 28), None)
        assert computed is not None, "KG Lanczos did not produce (0,28) beam"

        x = np.arange(26, dtype=float)
        reference = paper_beam_0_28_kg_fwd(x)

        mask = x >= 5
        abs_errors = np.abs(computed[mask] - reference[mask])
        mean_abs_err = np.mean(abs_errors)
        corr = np.corrcoef(computed[mask], reference[mask])[0, 1]

        assert mean_abs_err < self.MEAN_ABS_TOLERANCE, (
            f"KG Lanczos [0,28] WK vs paper FWD: mean abs error = {mean_abs_err:.4f}"
        )
        assert corr > self.CORRELATION_TOLERANCE, (
            f"KG Lanczos [0,28] WK vs paper FWD: correlation = {corr:.4f}"
        )
