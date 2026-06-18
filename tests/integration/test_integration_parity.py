# tests/test_integration_parity.py
"""Parity + grad tests for cumulative_trapz against progenax's dx-outside form.

Task 2 (Phase B): jaxstro's uniform-spacing cumulative_trapz path is standardized
on progenax's *dx-outside* semantics (cumsum the trapezoid increments first, then
multiply by dx once). This file proves byte-for-byte parity on shared inputs and
grad correctness for both the ``dx=`` (uniform) and ``x=`` (non-uniform) paths.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.numerics import integration


def _progenax_reference(y, dx, axis=-1):
    """Local copy of progenax's dx-OUTSIDE cumulative_trapezoid (reference semantics).

    out[..., k] = sum_{i<k} 0.5 * (y[..., i] + y[..., i+1]) * dx, out[..., 0] = 0.
    cumsum FIRST, multiply by dx ONCE (dx-outside).
    """
    y = jnp.moveaxis(y, axis, -1)
    inner = jnp.cumsum(0.5 * (y[..., 1:] + y[..., :-1]), axis=-1) * dx
    zero = jnp.zeros(y.shape[:-1] + (1,), dtype=inner.dtype)
    return jnp.moveaxis(jnp.concatenate([zero, inner], axis=-1), -1, axis)


# Arrays chosen to include negatives and non-power-of-two dx; these are exactly the
# kind of inputs where dx-inside vs dx-outside can disagree at the ~1-ulp level.
PARITY_CASES = [
    (jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]), 0.1),
    (jnp.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0]), 0.3),
    (jnp.array([-1.5, -0.5, 2.25, 7.125, -3.0]), 1.7),
    (jnp.linspace(-3.0, 5.0, 17), 0.123456789),
    (jnp.array([1e10, -1e10, 1.0, 2.0, 3.0, -7.0, 4.0]), 0.7),
]


class TestUniformParity:
    """jaxstro dx= path must equal progenax dx-outside form byte-for-byte."""

    @pytest.mark.parametrize("y,dx", PARITY_CASES)
    def test_byte_for_byte(self, y, dx):
        got = integration.cumulative_trapz(y, dx=dx)
        ref = _progenax_reference(y, dx=dx)
        assert jnp.array_equal(got, ref), (
            f"not byte-identical: max|diff|={float(jnp.max(jnp.abs(got - ref))):.3e}"
        )

    def test_byte_for_byte_2d_axis(self):
        y = jnp.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [1.0, -2.0, 3.0, -4.0, 5.0],
            ]
        )
        dx = 0.37
        got = integration.cumulative_trapz(y, dx=dx, axis=1)
        ref = _progenax_reference(y, dx=dx, axis=1)
        assert jnp.array_equal(got, ref)

    def test_starts_at_zero(self):
        y = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0])
        out = integration.cumulative_trapz(y, dx=0.2)
        assert out[0] == 0.0

    def test_default_dx_is_one(self):
        y = jnp.array([1.0, 1.0, 1.0, 1.0])
        out = integration.cumulative_trapz(y)
        # unit-spacing trapezoid of ones: 0, 1, 2, 3
        assert jnp.array_equal(out, jnp.array([0.0, 1.0, 2.0, 3.0]))


class TestNonUniformPath:
    """The x-array (non-uniform) path must still work."""

    def test_matches_cumulative_trapezoid_nonuniform(self):
        x = jnp.array([0.0, 0.5, 1.5, 1.7, 4.0])
        y = jnp.array([1.0, 2.0, 0.5, 3.0, -1.0])
        out = integration.cumulative_trapz(y, x=x)
        # Reference: leading 0 + cumsum of 0.5*(y_i+y_{i+1})*(x_{i+1}-x_i)
        dx = jnp.diff(x)
        incr = 0.5 * (y[:-1] + y[1:]) * dx
        ref = jnp.concatenate([jnp.zeros(1), jnp.cumsum(incr)])
        assert jnp.allclose(out, ref, rtol=0, atol=0)

    def test_ends_at_total_nonuniform(self):
        x = jnp.array([0.0, 0.5, 1.5, 1.7, 4.0])
        y = jnp.array([1.0, 2.0, 0.5, 3.0, -1.0])
        out = integration.cumulative_trapz(y, x=x)
        total = integration.trapz(y, x=x)
        assert jnp.allclose(out[-1], total)


class TestGradChecks:
    """FD vs AD grad-checks for both the dx= and x= paths (~1e-6 rel)."""

    def test_grad_dx_path(self):
        y = jnp.array([0.3, 1.2, -0.7, 2.1, 0.4, -1.1])
        dx = 0.25

        def f(yv):
            return jnp.sum(integration.cumulative_trapz(yv, dx=dx) ** 2)

        ad = jax.grad(f)(y)
        # finite differences
        eps = 1e-6
        fd = np.zeros(y.shape)
        for i in range(y.shape[0]):
            yp = y.at[i].add(eps)
            ym = y.at[i].add(-eps)
            fd[i] = (f(yp) - f(ym)) / (2 * eps)
        np.testing.assert_allclose(np.asarray(ad), fd, rtol=1e-6, atol=1e-6)

    def test_grad_x_path(self):
        x = jnp.array([0.0, 0.4, 1.1, 1.9, 3.0])
        y = jnp.array([1.0, 2.0, 0.5, 3.0, -1.0])

        def f(yv):
            return jnp.sum(integration.cumulative_trapz(yv, x=x) ** 2)

        ad = jax.grad(f)(y)
        eps = 1e-6
        fd = np.zeros(y.shape)
        for i in range(y.shape[0]):
            yp = y.at[i].add(eps)
            ym = y.at[i].add(-eps)
            fd[i] = (f(yp) - f(ym)) / (2 * eps)
        np.testing.assert_allclose(np.asarray(ad), fd, rtol=1e-6, atol=1e-6)

    def test_jacrev_dx_path(self):
        y = jnp.array([0.5, 1.0, -0.5, 2.0])
        dx = 0.5
        jac = jax.jacrev(lambda yv: integration.cumulative_trapz(yv, dx=dx))(y)
        # cumulative_trapz is linear in y, so the Jacobian is constant; verify it
        # matches a finite-difference Jacobian.
        eps = 1e-6
        base = integration.cumulative_trapz(y, dx=dx)
        fd = np.zeros((y.shape[0], y.shape[0]))
        for j in range(y.shape[0]):
            pert = integration.cumulative_trapz(y.at[j].add(eps), dx=dx)
            fd[:, j] = np.asarray((pert - base) / eps)
        np.testing.assert_allclose(np.asarray(jac), fd, rtol=1e-5, atol=1e-7)
