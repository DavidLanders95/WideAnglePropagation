"""Au(100) paper benchmark tests for maintained propagation methods.

The geometry follows the Au 300 kV example of Rother & Scheerschmidt (2009),
doi:10.1016/j.ultramic.2008.08.008. These tests check internal consistency for
the maintained finite-projection Lobato model; they are not regressions against
the paper's scattering-factor parametrisation.

These tests require GPU (cupy + abTEM) for potential generation.
Marked @pytest.mark.slow for tests that take >30 seconds.
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
from tests.conftest import (
    beam_amplitude_normalized,
    AU_ENERGY,
    AU_GPTS,
)

# Skip entire module if cupy unavailable (no GPU)
pytest.importorskip("cupy")


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

        # This expected divergence documents that the two kernels are not identical.
        assert rel_diff > 0.001, (
            f"AS and Fresnel unexpectedly identical for [0,28] at high thickness"
        )
