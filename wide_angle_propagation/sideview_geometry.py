"""Sideview geometry helpers for 1D glancing-incidence propagation."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .propagation_methods import energy2wavelength


__all__ = [
    "Line1D",
    "normalize",
    "rotation_2d",
    "line_from_angle",
    "reflect_direction",
    "phase_ramp_for_direction",
    "make_tilted_gaussian_beam_1d",
]


@dataclass(frozen=True)
class Line1D:
    """A sampled line in sideview ``(x, z)`` coordinates."""

    r0: jnp.ndarray
    tangent: jnp.ndarray
    normal: jnp.ndarray
    coords: jnp.ndarray

    def points(self):
        """Return sampled points with shape ``(n, 2)``."""
        return self.r0[None, :] + self.coords[:, None] * self.tangent[None, :]


def normalize(vector, eps: float = 1e-30):
    """Return a unit vector with a small differentiable norm floor."""
    vector = jnp.asarray(vector)
    norm = jnp.sqrt(jnp.sum(vector * vector))
    return vector / jnp.maximum(norm, eps)


def rotation_2d(angle):
    """Return the 2D rotation matrix for ``angle`` radians."""
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    return jnp.array([[c, -s], [s, c]])


def line_from_angle(r0, angle, coords):
    """Return a ``Line1D`` whose tangent is tilted from the vertical ``x`` axis."""
    rot = rotation_2d(angle)
    tangent = normalize(rot @ jnp.array([1.0, 0.0]))
    normal = normalize(rot @ jnp.array([0.0, 1.0]))
    return Line1D(
        r0=jnp.asarray(r0),
        tangent=tangent,
        normal=normal,
        coords=jnp.asarray(coords),
    )


def reflect_direction(direction, normal):
    """Reflect ``direction`` across a line with unit ``normal``."""
    direction = normalize(direction)
    normal = normalize(normal)
    return direction - 2.0 * jnp.dot(direction, normal) * normal


def phase_ramp_for_direction(coords, direction, energy, *, line=None, origin_phase=0.0):
    """Return a plane-wave phase ramp sampled on a sideview line.

    If ``line`` is omitted, ``coords`` are treated as vertical ``x`` positions
    on an upright input/output line.
    """
    wavelength = energy2wavelength(energy)
    k0 = 1.0 / wavelength
    direction = normalize(direction)

    if line is None:
        optical_path = jnp.asarray(coords) * direction[0]
    else:
        optical_path = line.points() @ direction

    return jnp.exp(1j * (2.0 * jnp.pi * k0 * optical_path + origin_phase))


def make_tilted_gaussian_beam_1d(
    coords,
    energy,
    *,
    waist,
    center=0.0,
    tilt=0.0,
    amplitude=1.0,
):
    """Return a Gaussian beam on an upright sideview line.

    The beam direction follows the plan convention
    ``k_hat = (sin(theta), cos(theta))`` in ``(x, z)`` coordinates.
    """
    coords = jnp.asarray(coords)
    envelope = amplitude * jnp.exp(-0.5 * ((coords - center) / waist) ** 2)
    direction = jnp.array([jnp.sin(tilt), jnp.cos(tilt)])
    return envelope * phase_ramp_for_direction(coords, direction, energy)
