"""
Tests for the Bloch wave eigenvalue solver.

CPU tests (no GPU required):
  - test_structure_factor_fcc_even_parity
  - test_structure_factor_fcc_mixed_parity
  - test_beam_count_consistency
  - test_thickness_zero_unit_amplitude

GPU tests (skipped when CuPy is unavailable):
  - test_eigenvector_unitarity
  - test_rmse_regression
"""
import importlib.util
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load bloch module without triggering the heavy __init__ (jax, abtem)
# ---------------------------------------------------------------------------
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BLOCH_PATH = os.path.join(_REPO, "wide_angle_propagation", "bloch.py")

spec = importlib.util.spec_from_file_location("bloch", _BLOCH_PATH)
_bloch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_bloch)

solve_bloch_wave_gpu = _bloch.solve_bloch_wave_gpu
_structure_factor   = _bloch._structure_factor
HAS_CUPY            = _bloch.HAS_CUPY
_energy2wavelength  = _bloch._energy2wavelength

# ---------------------------------------------------------------------------
# GPU marker
# ---------------------------------------------------------------------------
gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy/GPU not available")

# ---------------------------------------------------------------------------
# Reference data (used for RMSE regression test)
# ---------------------------------------------------------------------------
_REF_PATH = os.path.join(_REPO, "wide_angle_propagation", "reference_data.py")
_ref_spec = importlib.util.spec_from_file_location("reference_data", _REF_PATH)
_ref = importlib.util.module_from_spec(_ref_spec)
_ref_spec.loader.exec_module(_ref)

AU_300KV_BEAM_00_KG_MS   = _ref.AU_300KV_BEAM_00_KG_MS
AU_300KV_BEAM_028_KG_FWD = _ref.AU_300KV_BEAM_028_KG_FWD


# ---------------------------------------------------------------------------
# Shared fixture: Au FCC crystal + wavelength
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def au_atoms():
    from ase.build import bulk
    atoms = bulk("Au", "fcc", a=4.08, cubic=True)
    atoms.info["thermal_sigma"]    = 0.0
    atoms.arrays["thermal_sigma"]  = np.zeros(len(atoms))
    return atoms


@pytest.fixture(scope="module")
def wavelength_300kv():
    return _energy2wavelength(300e3)


# ---------------------------------------------------------------------------
# FCC fractional coordinates (used for structure-factor tests)
# ---------------------------------------------------------------------------
FCC_FRAC = np.array(
    [[0.0, 0.0, 0.0],
     [0.5, 0.5, 0.0],
     [0.5, 0.0, 0.5],
     [0.0, 0.5, 0.5]],
    dtype=np.float64,
)


# ===========================================================================
# 2a. Structure-factor tests (CPU only)
# ===========================================================================

@pytest.mark.parametrize("hkl", [[2, 0, 0], [1, 1, 1], [2, 2, 0], [0, 2, 2]])
def test_structure_factor_fcc_even_parity(hkl):
    """FCC selection rule: all-even and all-odd reflections give |F| = 4."""
    F = _structure_factor(hkl, FCC_FRAC)
    assert abs(abs(F) - 4.0) < 1e-10, (
        f"Expected |F({hkl})| = 4.0 for all-even FCC reflection, got {abs(F):.6f}"
    )


@pytest.mark.parametrize("hkl", [[1, 0, 0], [2, 1, 0], [1, 1, 0], [0, 1, 2]])
def test_structure_factor_fcc_mixed_parity(hkl):
    """FCC selection rule: mixed-parity reflections give |F| = 0."""
    F = _structure_factor(hkl, FCC_FRAC)
    assert abs(F) < 1e-10, (
        f"Expected |F({hkl})| = 0 for mixed-parity FCC reflection, got {abs(F):.6f}"
    )


# ===========================================================================
# 2b. Unitarity / current conservation (GPU)
# ===========================================================================

@gpu
def test_eigenvector_unitarity(au_atoms, wavelength_300kv):
    """The eigenvector matrix C from eigh must be unitary: max|I - C†C| < 1e-10."""
    result = solve_bloch_wave_gpu(
        g_max_zolz=10,
        g_max_holz=15,
        l_max=8,
        n_beams_max=5000,
        atoms=au_atoms,
        wavelength=wavelength_300kv,
        x=np.arange(0, 6),
        include_eigensystem=True,
    )
    evecs = result["evecs"]          # (N, N) complex
    I     = evecs.conj().T @ evecs
    deviation = np.max(np.abs(I - np.eye(evecs.shape[0])))
    assert deviation < 1e-10, (
        f"Eigenvector matrix not unitary: max|I - C†C| = {deviation:.3e}"
    )


# ===========================================================================
# 2c. Thickness = 0 gives unit amplitude for the central beam (CPU)
# ===========================================================================

def test_thickness_zero_unit_amplitude(au_atoms, wavelength_300kv):
    """At zero thickness the [0,0] beam amplitude must be ~1."""
    result = solve_bloch_wave_gpu(
        g_max_zolz=4.0,
        g_max_holz=5.0,
        l_max=1,
        n_beams_max=500,
        atoms=au_atoms,
        wavelength=wavelength_300kv,
        x=[0, 1, 2],
    )
    assert result["amp_00_coh"][0] == pytest.approx(1.0, abs=0.01), (
        f"amp_00_coh at t=0 should be ~1.0, got {result['amp_00_coh'][0]:.6f}"
    )


# ===========================================================================
# 2d. RMSE regression against Klein-Gordon ODE reference (GPU)
# ===========================================================================

@gpu
def test_rmse_regression(au_atoms, wavelength_300kv):
    """Bloch wave RMSE against KG ODE must be < 0.05 (per beam) / < 0.03 (avg).

    Uses the best published parameters from the issue specification:
    g_max_zolz=15, g_max_holz=25, l_max=10, n_beams_max=12000.
    """
    x = np.arange(0, 26, dtype=np.float64)

    result = solve_bloch_wave_gpu(
        g_max_zolz=15,
        g_max_holz=25,
        l_max=10,
        n_beams_max=12000,
        atoms=au_atoms,
        wavelength=wavelength_300kv,
        x=x,
        paper_00=AU_300KV_BEAM_00_KG_MS,
        paper_028=AU_300KV_BEAM_028_KG_FWD,
    )

    assert result["rmse_00"]  < 0.05, (
        f"rmse_00 = {result['rmse_00']:.4f} ≥ 0.05"
    )
    assert result["rmse_028"] < 0.05, (
        f"rmse_028 = {result['rmse_028']:.4f} ≥ 0.05"
    )
    assert result["rmse_avg"] < 0.03, (
        f"rmse_avg = {result['rmse_avg']:.4f} ≥ 0.03"
    )


# ===========================================================================
# 2e. Beam count consistency (CPU)
# ===========================================================================

def test_beam_count_consistency(au_atoms, wavelength_300kv):
    """n_beams == n_zolz + n_holz, and with small params n_beams < n_beams_max."""
    # Use a large cap so it is not hit; g_max values chosen to give ~480 beams
    n_beams_max = 5000
    result = solve_bloch_wave_gpu(
        g_max_zolz=3.0,
        g_max_holz=3.0,
        l_max=0,
        n_beams_max=n_beams_max,
        atoms=au_atoms,
        wavelength=wavelength_300kv,
        x=[0],
    )
    assert result["n_beams"] == result["n_zolz"] + result["n_holz"], (
        "n_beams must equal n_zolz + n_holz"
    )
    assert result["n_beams"] < n_beams_max, (
        f"n_beams = {result['n_beams']} is not < n_beams_max = {n_beams_max}; "
        "expected the cap not to be reached with small g_max"
    )


def test_beam_count_zero_holz(au_atoms, wavelength_300kv):
    """With l_max=0 the solver should return only ZOLZ beams."""
    result = solve_bloch_wave_gpu(
        g_max_zolz=3.0,
        g_max_holz=3.0,
        l_max=0,
        n_beams_max=1000,
        atoms=au_atoms,
        wavelength=wavelength_300kv,
        x=[0],
    )
    assert result["n_holz"] == 0, (
        f"Expected 0 HOLZ beams with l_max=0, got {result['n_holz']}"
    )
    assert result["n_beams"] == result["n_zolz"]
