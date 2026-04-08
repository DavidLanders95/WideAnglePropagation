"""Focused regressions for the full second-order KG ODE solver."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from wide_angle_propagation.propagation_methods import (
    electron_refractive_index,
    energy2wavelength,
    simulate_kg_ode_full,
)


jax.config.update("jax_enable_x64", True)

ENERGY = 200e3
DZ = 1.5
GPTS = (4, 4)
SAMPLING = (0.4, 0.4)


def _make_small_nonuniform_slice():
    ny, nx = GPTS
    y = np.arange(ny) / ny
    x = np.arange(nx) / nx
    X, Y = np.meshgrid(x, y)
    return 35.0 * (
        1.0
        + 0.25 * np.cos(2 * np.pi * X)
        + 0.15 * np.cos(2 * np.pi * Y)
        + 0.10 * np.sin(2 * np.pi * (X + Y))
    )


def _make_probe():
    ny, nx = GPTS
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


def _k_perp_sq_grid():
    ny, nx = GPTS
    dy, dx = SAMPLING
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    Kx, Ky = np.meshgrid(kx, ky)
    return Kx**2 + Ky**2


def _forward_vacuum_phi(probe, k0):
    k_perp_sq = _k_perp_sq_grid()
    kz = np.sqrt(np.array(k0**2 - k_perp_sq, dtype=np.complex128))
    kz = np.where(np.imag(kz) < 0, -kz, kz)
    return np.fft.ifft2(1j * kz * np.fft.fft2(probe))


def _structure_matrix(n_sq_slice, k0):
    ny, nx = GPTS
    dy, dx = SAMPLING

    U_full = (
        np.fft.fft2(np.asarray(n_sq_slice, dtype=np.complex128))
        / (ny * nx)
    )

    iy_all, ix_all = np.mgrid[:ny, :nx]
    beam_indices = np.stack([iy_all.ravel(), ix_all.ravel()], axis=1)
    iy = beam_indices[:, 0]
    ix = beam_indices[:, 1]

    diy = (iy[:, None] - iy[None, :]) % ny
    dix = (ix[:, None] - ix[None, :]) % nx

    fy = np.fft.fftfreq(ny, d=dy)
    fx = np.fft.fftfreq(nx, d=dx)
    k_perp_sq = (2 * np.pi * fy[iy]) ** 2 + (2 * np.pi * fx[ix]) ** 2

    return (k0**2) * U_full[diy, dix] - np.diag(k_perp_sq)


def _exact_single_slice_full_kg(potential_slice, probe, initial_phi):
    wavelength = float(energy2wavelength(ENERGY))
    k0 = 2 * np.pi / wavelength

    n_sq = np.asarray(
        electron_refractive_index(jnp.asarray(potential_slice), ENERGY) ** 2,
        dtype=np.float64,
    )
    M = _structure_matrix(n_sq, k0)
    n_beams = M.shape[0]

    A = np.zeros((2 * n_beams, 2 * n_beams), dtype=np.complex128)
    A[:n_beams, n_beams:] = np.eye(n_beams, dtype=np.complex128)
    A[n_beams:, :n_beams] = -M

    probe_k = np.fft.fft2(probe) / (GPTS[0] * GPTS[1])
    phi_k = np.fft.fft2(initial_phi) / (GPTS[0] * GPTS[1])
    state0 = np.concatenate([probe_k.ravel(), phi_k.ravel()])

    state1 = expm(DZ * A) @ state0

    exit_k = state1[:n_beams].reshape(GPTS)
    exit_phi_k = state1[n_beams:].reshape(GPTS)

    scale = GPTS[0] * GPTS[1]
    exit_wave = np.fft.ifft2(exit_k * scale)
    exit_phi = np.fft.ifft2(exit_phi_k * scale)
    return exit_wave, exit_phi


def test_full_kg_matches_exact_matrix_exponential_for_one_slice():
    potential_slice = _make_small_nonuniform_slice()
    probe = _make_probe()

    wavelength = float(energy2wavelength(ENERGY))
    k0 = 2 * np.pi / wavelength
    initial_phi = _forward_vacuum_phi(probe, k0)

    exact_wave, exact_phi = _exact_single_slice_full_kg(
        potential_slice,
        probe,
        initial_phi,
    )
    ode_wave, ode_phi, _, _ = simulate_kg_ode_full(
        jnp.asarray(potential_slice[None, :, :]),
        jnp.asarray(probe),
        DZ,
        ENERGY,
        SAMPLING,
        initial_phi=jnp.asarray(initial_phi),
        rtol=1e-10,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(ode_wave),
        exact_wave,
        rtol=1e-6,
        atol=1e-7,
        err_msg=(
            "Full KG ODE should match the exact small-grid matrix "
            "exponential"
        ),
    )
    np.testing.assert_allclose(
        np.asarray(ode_phi),
        exact_phi,
        rtol=1e-6,
        atol=1e-7,
        err_msg="Full KG ODE exit derivative should match the exact reference",
    )


def test_full_kg_matches_sequential_cell_propagation_when_phi_is_carried():
    probe = _make_probe()
    ny, nx = GPTS
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    left = 120.0 * np.exp(-((X - 1.0) ** 2 + (Y - 1.5) ** 2) / 0.8)
    right = 120.0 * np.exp(-((X - 2.5) ** 2 + (Y - 1.5) ** 2) / 0.8)
    potential = jnp.asarray(np.stack([left, right, left, right]))

    full_wave, full_phi, _, full_wavefronts = simulate_kg_ode_full(
        potential,
        jnp.asarray(probe),
        DZ,
        ENERGY,
        SAMPLING,
        rtol=1e-9,
        atol=1e-11,
    )

    state = jnp.asarray(probe)
    phi = None
    sequential = []
    for idx in range(potential.shape[0]):
        state, phi, _, _ = simulate_kg_ode_full(
            potential[idx:idx + 1],
            state,
            DZ,
            ENERGY,
            SAMPLING,
            initial_phi=phi,
            rtol=1e-9,
            atol=1e-11,
        )
        sequential.append(np.asarray(state))

    sequential = np.stack(sequential)

    np.testing.assert_allclose(
        np.asarray(full_wavefronts),
        sequential,
        rtol=1e-6,
        atol=1e-7,
        err_msg=(
            "Full KG wavefronts should match sequential calls when "
            "exit_phi is "
            "fed back as initial_phi"
        ),
    )
    np.testing.assert_allclose(
        np.asarray(full_wave),
        sequential[-1],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(full_phi),
        np.asarray(phi),
        rtol=1e-6,
        atol=1e-7,
    )
