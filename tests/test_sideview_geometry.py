import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("ase")
jax.config.update("jax_enable_x64", True)

from wide_angle_propagation.sideview_geometry import (
    line_from_angle,
    make_tilted_gaussian_beam_1d,
    normalize,
    reflect_direction,
    rotation_2d,
)


def test_line_from_angle_points_and_unit_vectors():
    coords = jnp.array([-1.0, 0.0, 1.0])
    line = line_from_angle(jnp.array([2.0, 3.0]), 0.0, coords)

    np.testing.assert_allclose(np.asarray(line.tangent), [1.0, 0.0])
    np.testing.assert_allclose(np.asarray(line.normal), [0.0, 1.0])
    np.testing.assert_allclose(
        np.asarray(line.points()),
        [[1.0, 3.0], [2.0, 3.0], [3.0, 3.0]],
    )


def test_rotation_and_reflection_law():
    rot = rotation_2d(jnp.pi / 2.0)
    np.testing.assert_allclose(np.asarray(rot @ jnp.array([1.0, 0.0])), [0.0, 1.0], atol=1e-12)

    reflected = reflect_direction(jnp.array([1.0, 1.0]), jnp.array([0.0, 1.0]))
    np.testing.assert_allclose(np.asarray(reflected), [1 / np.sqrt(2), -1 / np.sqrt(2)])
    np.testing.assert_allclose(np.linalg.norm(np.asarray(normalize(reflected))), 1.0)


def test_tilted_gaussian_beam_has_finite_gradient_through_tilt():
    coords = jnp.linspace(-2.0, 2.0, 32)

    def objective(tilt):
        beam = make_tilted_gaussian_beam_1d(coords, 200e3, waist=0.7, tilt=tilt)
        return jnp.real(jnp.vdot(beam, beam * coords))

    grad = jax.grad(objective)(0.05)
    assert np.isfinite(np.asarray(grad))
