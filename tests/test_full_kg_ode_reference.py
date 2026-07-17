"""Behavior and exact-reference tests for the full KG ODE solver.

These tests validate ``simulate_kg_ode_full`` as a true second-order
Klein-Gordon system. They intentionally avoid treating it as a forward-only
multislice approximation.
"""

import numpy as np
import pytest
from scipy.linalg import expm

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ase")
pytest.importorskip("diffrax")

from wide_angle_propagation.propagation_methods import (
    electron_refractive_index,
    energy2wavelength,
    simulate_kg_ode_full,
)


jax.config.update("jax_enable_x64", True)

ENERGY = 200e3
SMALL_DZ = 1.5
SMALL_GPTS = (4, 4)
SMALL_SAMPLING = (0.4, 0.4)


def _make_grid(ny, nx, dy, dx):
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    return np.meshgrid(x, y)


def _plane_wave(ny, nx):
    return np.ones((ny, nx), dtype=np.complex128) / np.sqrt(ny * nx)


def _gaussian_probe(ny, nx, dy, dx, sigma):
    X, Y = _make_grid(ny, nx, dy, dx)
    cx, cy = nx * dx / 2, ny * dy / 2
    probe = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (sigma**2))
    probe = probe.astype(np.complex128)
    return probe / np.sqrt(np.sum(np.abs(probe) ** 2))


def _transverse_frequency_sq_grid(gpts, sampling):
    ny, nx = gpts
    dy, dx = sampling
    ky = np.fft.fftfreq(ny, d=dy)
    kx = np.fft.fftfreq(nx, d=dx)
    Kx, Ky = np.meshgrid(kx, ky)
    return Kx**2 + Ky**2


def _forward_vacuum_phi(probe, energy, sampling):
    ny, nx = probe.shape
    k0 = 1 / float(energy2wavelength(energy))
    k_perp_sq = _transverse_frequency_sq_grid((ny, nx), sampling)
    probe_k = np.fft.fft2(np.asarray(probe))
    kz = np.sqrt(np.array(k0**2 - k_perp_sq, dtype=np.complex128))
    kz = np.where(np.imag(kz) < 0, -kz, kz)
    return np.fft.ifft2(2j * np.pi * kz * probe_k)


def _exact_vacuum_reference(probe, total_thickness, energy, sampling):
    ny, nx = probe.shape
    k0 = 1 / float(energy2wavelength(energy))
    k_perp_sq = _transverse_frequency_sq_grid((ny, nx), sampling)
    probe_k = np.fft.fft2(np.asarray(probe))
    kz = np.sqrt(np.array(k0**2 - k_perp_sq, dtype=np.complex128))
    kz = np.where(np.imag(kz) < 0, -kz, kz)

    phase = np.exp(2j * np.pi * kz * total_thickness)
    exit_wave = np.fft.ifft2(phase * probe_k)
    exit_phi = np.fft.ifft2(2j * np.pi * kz * phase * probe_k)
    return exit_wave, exit_phi


def _small_probe():
    ny, nx = SMALL_GPTS
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    cx = 0.5 * (nx - 1)
    cy = 0.5 * (ny - 1)
    amp = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 2.5)
    phase = np.exp(1j * 0.35 * X - 1j * 0.20 * Y)
    probe = amp * phase
    probe = probe.astype(np.complex128)
    return probe / np.sqrt(np.sum(np.abs(probe) ** 2))


def _small_nonuniform_stack():
    ny, nx = SMALL_GPTS
    y = np.arange(ny) / ny
    x = np.arange(nx) / nx
    X, Y = np.meshgrid(x, y)
    return np.stack([
        35.0 * (
            1.0
            + 0.25 * np.cos(2 * np.pi * X)
            + 0.15 * np.cos(2 * np.pi * Y)
            + 0.10 * np.sin(2 * np.pi * (X + Y))
        ),
        28.0 * (
            1.0
            + 0.20 * np.sin(2 * np.pi * X)
            - 0.10 * np.cos(2 * np.pi * Y)
            + 0.08 * np.sin(2 * np.pi * (X - Y))
        ),
        22.0 * (
            1.0
            - 0.18 * np.cos(2 * np.pi * X)
            + 0.12 * np.sin(2 * np.pi * Y)
        ),
    ])


def _small_discontinuous_stack():
    ny, nx = SMALL_GPTS
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    left = 120.0 * np.exp(-((X - 1.0) ** 2 + (Y - 1.5) ** 2) / 0.8)
    right = 120.0 * np.exp(-((X - 2.5) ** 2 + (Y - 1.5) ** 2) / 0.8)
    return np.stack([left, right, left, right])


def _structure_matrix(n_sq_slice, energy, sampling):
    ny, nx = n_sq_slice.shape
    dy, dx = sampling
    k0 = 1 / float(energy2wavelength(energy))

    U_full = np.fft.fft2(np.asarray(n_sq_slice, dtype=np.complex128))
    U_full = U_full / (ny * nx)

    iy_all, ix_all = np.mgrid[:ny, :nx]
    beam_indices = np.stack([iy_all.ravel(), ix_all.ravel()], axis=1)
    iy = beam_indices[:, 0]
    ix = beam_indices[:, 1]

    diy = (iy[:, None] - iy[None, :]) % ny
    dix = (ix[:, None] - ix[None, :]) % nx

    fy = np.fft.fftfreq(ny, d=dy)
    fx = np.fft.fftfreq(nx, d=dx)
    k_perp_sq = fy[iy] ** 2 + fx[ix] ** 2

    return (2 * np.pi) ** 2 * (
        (k0**2) * U_full[diy, dix] - np.diag(k_perp_sq)
    )


def _exact_full_kg_stack(
    potential,
    probe,
    initial_phi,
    slice_thickness,
    energy,
    sampling,
):
    ny, nx = potential.shape[1:]
    n_beams = ny * nx
    scale = ny * nx

    psi_k = np.fft.fft2(np.asarray(probe)) / scale
    phi_k = np.fft.fft2(np.asarray(initial_phi)) / scale
    state = np.concatenate([psi_k.ravel(), phi_k.ravel()])
    wavefronts = []

    for potential_slice in potential:
        n_sq = np.asarray(
            electron_refractive_index(jnp.asarray(potential_slice), energy) ** 2,
            dtype=np.float64,
        )
        M = _structure_matrix(n_sq, energy, sampling)

        A = np.zeros((2 * n_beams, 2 * n_beams), dtype=np.complex128)
        A[:n_beams, n_beams:] = np.eye(n_beams, dtype=np.complex128)
        A[n_beams:, :n_beams] = -M

        state = expm(slice_thickness * A) @ state
        psi_slice_k = state[:n_beams].reshape((ny, nx))
        wavefronts.append(np.fft.ifft2(psi_slice_k * scale))

    exit_k = state[:n_beams].reshape((ny, nx))
    exit_phi_k = state[n_beams:].reshape((ny, nx))

    exit_wave = np.fft.ifft2(exit_k * scale)
    exit_phi = np.fft.ifft2(exit_phi_k * scale)
    return exit_wave, exit_phi, np.stack(wavefronts)


class TestVacuumReference:
    ny = nx = 16
    dy = dx = 0.2
    dz = 1.5
    n_slices = 8

    def test_matches_exact_spectral_solution(self):
        probe = _gaussian_probe(self.ny, self.nx, self.dy, self.dx, sigma=0.7)
        pot = jnp.zeros((self.n_slices, self.ny, self.nx))

        ew, phi, _, wavefronts = simulate_kg_ode_full(
            pot,
            jnp.asarray(probe),
            self.dz,
            ENERGY,
            (self.dy, self.dx),
            rtol=1e-10,
            atol=1e-12,
        )

        exact_wave, exact_phi = _exact_vacuum_reference(
            probe,
            self.n_slices * self.dz,
            ENERGY,
            (self.dy, self.dx),
        )
        exact_wavefronts = np.stack([
            _exact_vacuum_reference(
                probe,
                (idx + 1) * self.dz,
                ENERGY,
                (self.dy, self.dx),
            )[0]
            for idx in range(self.n_slices)
        ])

        np.testing.assert_allclose(
            np.asarray(ew),
            exact_wave,
            rtol=1e-7,
            atol=1e-8,
            err_msg="Vacuum exit wave should match the exact spectral solution",
        )
        np.testing.assert_allclose(
            np.asarray(phi),
            exact_phi,
            rtol=1e-7,
            atol=1e-8,
            err_msg=(
                "Vacuum exit derivative should match the exact spectral "
                "solution"
            ),
        )
        np.testing.assert_allclose(
            np.asarray(wavefronts),
            exact_wavefronts,
            rtol=1e-7,
            atol=1e-8,
            err_msg="Saved vacuum wavefronts should land on exact slice boundaries",
        )


class TestUniformMediumReference:
    ny = nx = 16
    dy = dx = 0.2
    dz = 1.5

    def test_plane_wave_matches_analytic_uniform_solution(self):
        n_slices = 10
        potential_value = 150.0
        pot = jnp.full((n_slices, self.ny, self.nx), potential_value)

        plane = _plane_wave(self.ny, self.nx)
        k0 = 1 / float(energy2wavelength(ENERGY))
        n = float(electron_refractive_index(potential_value, ENERGY))
        total_thickness = n_slices * self.dz

        expected_wave = plane * np.exp(2j * np.pi * k0 * n * total_thickness)
        expected_phi = 2j * np.pi * k0 * n * expected_wave
        initial_phi = 2j * np.pi * k0 * n * plane

        ew, phi, _, _ = simulate_kg_ode_full(
            pot,
            jnp.asarray(plane),
            self.dz,
            ENERGY,
            (self.dy, self.dx),
            initial_phi=jnp.asarray(initial_phi),
            rtol=1e-10,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            np.asarray(ew),
            expected_wave,
            rtol=1e-7,
            atol=1e-8,
            err_msg="Uniform-medium plane wave should follow the analytic KG phase",
        )
        np.testing.assert_allclose(
            np.asarray(phi),
            expected_phi,
            rtol=1e-7,
            atol=1e-8,
            err_msg=(
                "Uniform-medium exit derivative should match the analytic KG "
                "solution"
            ),
        )


class TestExactReferenceConvergence:
    def test_matches_exact_multislice_matrix_exponential(self):
        potential = _small_nonuniform_stack()
        probe = _small_probe()
        initial_phi = _forward_vacuum_phi(probe, ENERGY, SMALL_SAMPLING)

        exact_wave, exact_phi, exact_wavefronts = _exact_full_kg_stack(
            potential,
            probe,
            initial_phi,
            SMALL_DZ,
            ENERGY,
            SMALL_SAMPLING,
        )

        ew, phi, _, wavefronts = simulate_kg_ode_full(
            jnp.asarray(potential),
            jnp.asarray(probe),
            SMALL_DZ,
            ENERGY,
            SMALL_SAMPLING,
            initial_phi=jnp.asarray(initial_phi),
            rtol=1e-10,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            np.asarray(ew),
            exact_wave,
            rtol=1e-6,
            atol=1e-7,
            err_msg=(
                "Full KG ODE should match the exact slice-wise matrix "
                "exponential reference"
            ),
        )
        np.testing.assert_allclose(
            np.asarray(phi),
            exact_phi,
            rtol=1e-6,
            atol=1e-7,
            err_msg="Exit derivative should match the exact reference",
        )
        np.testing.assert_allclose(
            np.asarray(wavefronts),
            exact_wavefronts,
            rtol=1e-6,
            atol=1e-7,
            err_msg="Saved wavefronts should match the exact reference",
        )


class TestSecondOrderStateHandling:
    def test_slice_by_slice_calls_must_carry_exit_phi(self):
        potential = _small_discontinuous_stack()
        probe = _small_probe()

        full_wave, _, _, _ = simulate_kg_ode_full(
            jnp.asarray(potential),
            jnp.asarray(probe),
            SMALL_DZ,
            ENERGY,
            SMALL_SAMPLING,
            rtol=1e-9,
            atol=1e-11,
        )

        state = jnp.asarray(probe)
        phi = None
        for idx in range(potential.shape[0]):
            state, phi, _, _ = simulate_kg_ode_full(
                jnp.asarray(potential[idx:idx + 1]),
                state,
                SMALL_DZ,
                ENERGY,
                SMALL_SAMPLING,
                initial_phi=phi,
                rtol=1e-9,
                atol=1e-11,
            )

        with_phi_error = float(
            jnp.linalg.norm(state - full_wave) / jnp.linalg.norm(full_wave)
        )

        state = jnp.asarray(probe)
        for idx in range(potential.shape[0]):
            state, _, _, _ = simulate_kg_ode_full(
                jnp.asarray(potential[idx:idx + 1]),
                state,
                SMALL_DZ,
                ENERGY,
                SMALL_SAMPLING,
                rtol=1e-9,
                atol=1e-11,
            )

        without_phi_error = float(
            jnp.linalg.norm(state - full_wave) / jnp.linalg.norm(full_wave)
        )

        assert with_phi_error < 1e-8, (
            f"Carrying exit_phi should reproduce the full-stack solution: "
            f"{with_phi_error:.3e}"
        )
        assert without_phi_error > 1e-4, (
            "Dropping exit_phi should measurably change the second-order KG "
            f"state: {without_phi_error:.3e}"
        )
