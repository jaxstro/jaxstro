"""Tests for regular-grid interpolation utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.numerics import regular_grid


def test_bilinear_interp_recovers_affine_surface():
    x = jnp.array([0.0, 1.0, 3.0])
    y = jnp.array([-1.0, 2.0])
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    values = 2.0 + 3.0 * xx - 0.5 * yy
    x_new = jnp.array([0.25, 2.5])
    y_new = jnp.array([0.5, 1.5])

    result = regular_grid.bilinear_interp(x, y, values, x_new, y_new)

    np.testing.assert_allclose(result, 2.0 + 3.0 * x_new - 0.5 * y_new, atol=1e-12)


def test_trilinear_interp_recovers_affine_volume():
    x = jnp.array([0.0, 1.0])
    y = jnp.array([0.0, 2.0])
    z = jnp.array([-1.0, 1.0])
    xx, yy, zz = jnp.meshgrid(x, y, z, indexing="ij")
    values = 1.0 + 2.0 * xx - yy + 0.25 * zz

    result = regular_grid.trilinear_interp(
        x,
        y,
        z,
        values,
        jnp.array([0.25, 0.75]),
        jnp.array([0.5, 1.5]),
        jnp.array([-0.5, 0.5]),
    )

    expected = 1.0 + 2.0 * jnp.array([0.25, 0.75])
    expected = expected - jnp.array([0.5, 1.5]) + 0.25 * jnp.array([-0.5, 0.5])
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_regular_grid_interp_supports_vector_values():
    x = jnp.array([0.0, 1.0])
    y = jnp.array([0.0, 1.0])
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    values = jnp.stack([xx + yy, xx - yy], axis=-1)
    xi = jnp.array([[0.25, 0.5], [0.75, 0.25]])

    result = regular_grid.regular_grid_interp((x, y), values, xi)

    expected = jnp.array([[0.75, -0.25], [1.0, 0.5]])
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_regular_grid_boundary_policies_clamp_fill_and_reject():
    x = jnp.array([0.0, 1.0])
    y = jnp.array([0.0, 1.0])
    values = jnp.array([[0.0, 1.0], [1.0, 2.0]])
    xi = jnp.array([[-0.5, 0.5], [0.5, 1.5]])

    clamped = regular_grid.regular_grid_interp((x, y), values, xi, boundary="clamp")
    filled = regular_grid.regular_grid_interp(
        (x, y),
        values,
        xi,
        boundary="fill",
        fill_value=-99.0,
    )

    np.testing.assert_allclose(clamped, jnp.array([0.5, 1.5]), atol=1e-12)
    np.testing.assert_allclose(filled, jnp.array([-99.0, -99.0]), atol=1e-12)
    with pytest.raises(ValueError, match="outside"):
        regular_grid.regular_grid_interp((x, y), values, xi, boundary="reject")


def test_regular_grid_interp_is_jit_vmap_and_grad_compatible():
    x = jnp.array([0.0, 1.0, 2.0])
    y = jnp.array([0.0, 1.0])
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    values = xx**2 + yy
    xi = jnp.array([[0.25, 0.5], [1.5, 0.25]])

    @jax.jit
    def evaluate(v, points):
        return regular_grid.regular_grid_interp((x, y), v, points)

    result = evaluate(values, xi)
    vmapped = jax.vmap(
        lambda point: regular_grid.regular_grid_interp((x, y), values, point)
    )(xi)
    grad_values = jax.grad(
        lambda v: jnp.sum(regular_grid.regular_grid_interp((x, y), v, xi))
    )(values)

    np.testing.assert_allclose(result, vmapped, atol=1e-12)
    assert jnp.all(jnp.isfinite(grad_values))


def test_regular_grid_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="strictly increasing"):
        regular_grid.regular_grid_interp(
            (jnp.array([0.0, 0.0]),),
            jnp.array([0.0, 1.0]),
            jnp.array([[0.0]]),
        )

    with pytest.raises(ValueError, match="leading grid shape"):
        regular_grid.regular_grid_interp(
            (jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0])),
            jnp.array([0.0, 1.0]),
            jnp.array([[0.5, 0.5]]),
        )

    with pytest.raises(ValueError, match="boundary"):
        regular_grid.regular_grid_interp(
            (jnp.array([0.0, 1.0]),),
            jnp.array([0.0, 1.0]),
            jnp.array([[0.5]]),
            boundary="nearest",
        )
