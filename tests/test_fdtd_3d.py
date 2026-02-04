import os
import shutil
import subprocess
import time
import shutil as _shutil
import sys

import numpy as np
import pytest
import importlib.util

# Load fdtd_solver directly to avoid importing heavy deps from package __init__
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FDTSOLVER_PATH = os.path.join(REPO_ROOT, "wide_angle_propagation", "fdtd_solver.py")

spec = importlib.util.spec_from_file_location("fdtd_solver", FDTSOLVER_PATH)
fdtd_solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fdtd_solver)
RelativisticSolver3D = fdtd_solver.RelativisticSolver3D


def _copy_source(tmp_path):
    repo_src = os.path.join(
        os.path.dirname(__file__),
        "..",
        "wide_angle_propagation",
        "fdtd_3d.cu",
    )
    repo_src = os.path.abspath(repo_src)
    tmp_src = tmp_path / "fdtd_3d.cu"
    shutil.copyfile(repo_src, tmp_src)
    return tmp_src


def _make_solver(tmp_path):
    src = _copy_source(tmp_path)
    bin_path = tmp_path / "fdtd_3d_bin"
    bin_path.write_text("")
    return RelativisticSolver3D(source_file=str(src), binary_file=str(bin_path))


def _write_fake_snapshots(path, nx, ny, nz, M=1, dt=1e-3, E_sim=1.0):
    vol = nx * ny * nz
    with open(path, "wb") as f:
        f.write(np.array([nx], dtype=np.int32).tobytes())
        f.write(np.array([ny], dtype=np.int32).tobytes())
        f.write(np.array([nz], dtype=np.int32).tobytes())
        f.write(np.array([M], dtype=np.int32).tobytes())
        f.write(np.array([dt], dtype=np.float32).tobytes())
        f.write(np.array([E_sim], dtype=np.float32).tobytes())
        for i in range(M):
            t = np.array([i * dt], dtype=np.float32)
            phi = np.zeros(vol, dtype=np.float32)
            psi = np.zeros(vol, dtype=np.float32)
            f.write(t.tobytes())
            f.write(phi.tobytes())
            f.write(psi.tobytes())


def test_update_macros_basic(tmp_path):
    solver = _make_solver(tmp_path)

    changed = solver._update_macros(
        nx=8,
        ny=9,
        nz=10,
        xmin=-2.0,
        xmax=2.0,
        ymin=-3.0,
        ymax=3.0,
        zmin=-4.0,
        zmax=4.0,
        pml_thick=0.2,
        pml_sigma=70.0,
        pml_theta=0.7,
        pml_power=3.0,
        k_source=5,
        wavelength_angstrom=0.05,
        use_angstrom_units=True,
    )

    assert changed is True
    content = (tmp_path / "fdtd_3d.cu").read_text()
    assert "#define NX 8" in content
    assert "#define NY 9" in content
    assert "#define NZ 10" in content
    assert "#define XMIN -2.0000f" in content
    assert "#define PML_THICK" in content
    assert "0.2000f" in content
    assert "#define K_SOURCE 5" in content


def test_run_writes_potential_and_invokes_binary(tmp_path, monkeypatch):
    solver = _make_solver(tmp_path)

    # Avoid nvcc in tests
    solver.compile = lambda: None

    # Ensure binary is newer than source to skip compile check
    bin_path = solver.binary_file
    future = time.time() + 10
    os.utime(bin_path, (future, future))

    # Prepare fake snapshots.bin so _load_snapshots succeeds
    snap_path = os.path.join(solver.working_dir, "snapshots.bin")
    _write_fake_snapshots(snap_path, nx=4, ny=3, nz=2, M=1, dt=1e-3, E_sim=1.0)

    called = {}

    def fake_run(cmd, check, capture_output, text, cwd):
        called["cmd"] = cmd
        called["cwd"] = cwd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    V_map = np.zeros((2, 3, 4), dtype=np.float32)
    extent = (-1.0, 1.0, -1.0, 1.0, -1.5, 1.5)

    res = solver.run(
        V_map,
        wavelength_angstrom=0.05,
        extent=extent,
        n_frames=1,
        steps_per_frame=1,
        snapshot_count=1,
        pml_thick=0.3,
        k_source=2,
        use_angstrom_units=True,
    )

    assert res is not None
    assert "cmd" in called
    assert called["cwd"] == solver.working_dir

    pot_path = os.path.join(solver.working_dir, "potential.bin")
    assert os.path.exists(pot_path)
    with open(pot_path, "rb") as f:
        header = np.frombuffer(f.read(12), dtype=np.int32)
        assert header.tolist() == [4, 3, 2]


def test_load_snapshots_roundtrip(tmp_path):
    solver = _make_solver(tmp_path)

    snap_path = tmp_path / "snapshots.bin"
    _write_fake_snapshots(snap_path, nx=4, ny=3, nz=2, M=2, dt=2e-3, E_sim=2.5)

    out = solver._load_snapshots(str(snap_path))
    assert out["dt"] == np.float32(2e-3)
    assert out["E_sim"] == np.float32(2.5)
    assert len(out["snapshots"]) == 2
    t0, phi0, psi0 = out["snapshots"][0]
    assert phi0.shape == (2, 3, 4)
    assert psi0.shape == (2, 3, 4)


def _cuda_available():
    return _shutil.which("nvcc") is not None


def _can_run_physics_tests():
    return os.environ.get("RUN_FDTD_3D", "1") == "1" and _cuda_available()


def _run_free_space_case(tmp_path, nx=16, ny=16, nz=24, wavelength=0.1):
    solver = _make_solver(tmp_path)
    extent = (-1.0, 1.0, -1.0, 1.0, -1.5, 1.5)
    V_map = np.zeros((nz, ny, nx), dtype=np.float32)

    res = solver.run(
        V_map,
        wavelength_angstrom=wavelength,
        extent=extent,
        n_frames=2,
        steps_per_frame=50,
        snapshot_count=2,
        pml_thick=0.3,
        k_source=4,
        use_angstrom_units=True,
    )
    return res, extent


def _phase_slope_along_z(field, t, E_sim, extent):
    field_demod = field * np.exp(1j * E_sim * t)

    nz = field_demod.shape[0]
    z = np.linspace(extent[4], extent[5], nz)

    i = field_demod.shape[2] // 2
    j = field_demod.shape[1] // 2
    start = 3
    stop = nz - 3

    phase = np.unwrap(np.angle(field_demod[start:stop, j, i]))
    z_fit = z[start:stop]

    slope, intercept = np.polyfit(z_fit, phase, 1)
    return slope, intercept, phase, z_fit


@pytest.mark.skipif(
    not _can_run_physics_tests(),
    reason="Set RUN_FDTD_3D=1 and ensure nvcc is available to run 3D FDTD physics tests.",
)
def test_free_space_plane_wave_uniformity(tmp_path):
    res, extent = _run_free_space_case(tmp_path)

    assert res is not None
    t, phi, psi = res["snapshots"][-1]
    field = phi + 1j * psi

    # Inspect central z-slice and check uniformity across x/y
    nz = field.shape[0]
    k_mid = nz // 2
    intensity = np.abs(field[k_mid]) ** 2
    mean = intensity.mean()
    std = intensity.std()

    # Expect small variation in free space for a plane wave
    assert mean > 0
    assert std / mean < 0.2


@pytest.mark.skipif(
    not _can_run_physics_tests(),
    reason="Set RUN_FDTD_3D=1 and ensure nvcc is available to run 3D FDTD physics tests.",
)
def test_symmetric_potential_yields_symmetric_intensity(tmp_path):
    solver = _make_solver(tmp_path)

    nx, ny, nz = 16, 16, 24
    extent = (-1.0, 1.0, -1.0, 1.0, -1.5, 1.5)

    # Symmetric Gaussian potential centered in x/y
    x = np.linspace(extent[0], extent[1], nx)
    y = np.linspace(extent[2], extent[3], ny)
    z = np.linspace(extent[4], extent[5], nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    sigma = 0.3
    V = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))
    V_map = np.transpose(V, (2, 1, 0)).astype(np.float32)

    res = solver.run(
        V_map,
        wavelength_angstrom=0.1,
        extent=extent,
        n_frames=2,
        steps_per_frame=50,
        snapshot_count=2,
        pml_thick=0.3,
        k_source=4,
        use_angstrom_units=True,
    )

    assert res is not None
    t, phi, psi = res["snapshots"][-1]
    field = phi + 1j * psi
    k_mid = nz // 2
    intensity = np.abs(field[k_mid]) ** 2

    # Symmetry check across x and y
    assert np.allclose(intensity, intensity[::-1, :], rtol=0.2, atol=1e-3)
    assert np.allclose(intensity, intensity[:, ::-1], rtol=0.2, atol=1e-3)


@pytest.mark.skipif(
    not _can_run_physics_tests(),
    reason="Set RUN_FDTD_3D=1 and ensure nvcc is available to run 3D FDTD physics tests.",
)
def test_dirichlet_boundaries_zero(tmp_path):
    res, _extent = _run_free_space_case(tmp_path)

    assert res is not None
    _t, phi, psi = res["snapshots"][-1]

    # Dirichlet boundaries in Z should remain ~0
    boundary_mag = np.abs(phi[[0, -1], :, :]) + np.abs(psi[[0, -1], :, :])
    assert np.max(boundary_mag) < 1e-3


@pytest.mark.skipif(
    not _can_run_physics_tests(),
    reason="Set RUN_FDTD_3D=1 and ensure nvcc is available to run 3D FDTD physics tests.",
)
def test_free_space_phase_matches_wavenumber(tmp_path):
    wavelength = 0.12
    res, extent = _run_free_space_case(tmp_path, wavelength=wavelength)

    assert res is not None
    t, phi, psi = res["snapshots"][-1]

    # Fit slope; phase should be approximately linear in z (free-space plane wave)
    E_sim = res["E_sim"]
    field = phi + 1j * psi
    slope, _intercept, phase, z_fit = _phase_slope_along_z(field, t, E_sim, extent)
    assert np.isfinite(slope)
    assert np.abs(slope) > 1e-3

    # Check linearity (high correlation with fitted line)
    phase_fit = slope * z_fit + _intercept
    residual = phase - phase_fit
    r2 = 1.0 - (np.sum(residual**2) / np.sum((phase - phase.mean())**2))
    assert r2 > 0.8


@pytest.mark.skipif(
    not _can_run_physics_tests(),
    reason="Set RUN_FDTD_3D=1 and ensure nvcc is available to run 3D FDTD physics tests.",
)
def test_forward_propagation_phase_slope_positive(tmp_path):
    res, extent = _run_free_space_case(tmp_path, wavelength=0.12)

    assert res is not None
    t, phi, psi = res["snapshots"][-1]
    field = phi + 1j * psi
    E_sim = res["E_sim"]

    slope, _intercept, _phase, _z_fit = _phase_slope_along_z(field, t, E_sim, extent)

    # Forward propagation should yield a positive phase slope with increasing z
    assert slope > 0
