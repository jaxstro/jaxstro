r"""Differentiability contract for the promoted Lane-Emden solver.

The whole reason this solver lives in the shared foundation is that gradients flow
through it -- hydrax and progenax both build differentiable initial conditions on top of
it. Two contracts are locked here against finite differences:

1. ``d xi_1 / d n`` -- the first zero of ``theta`` is a ``diffrax.Event`` root, so its
   gradient in ``n`` comes from the implicit function theorem, NOT from differentiating a
   grid ``argmin`` (which has no gradient). This is the subtle one.
2. ``d m_edge / d n`` -- a plain functional of the integrated solution, differentiable
   through the diffrax solve.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jaxstro.numerics.lane_emden import (  # noqa: E402
    polytrope_xi1,
    solve_isothermal,
    solve_polytrope,
)


def _central_fd(fn, x, h):
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


def test_polytrope_xi1_gradient_is_finite_and_nonzero():
    """The event-root xi_1 must be differentiable in n (implicit function theorem)."""
    g = jax.grad(lambda n: polytrope_xi1(n))(1.5)
    assert jnp.isfinite(g)
    assert g != 0.0


def test_polytrope_xi1_gradient_matches_finite_difference():
    """AD grad of xi_1 in n agrees with a central finite difference."""
    n0 = 1.5
    g_ad = float(jax.grad(lambda n: polytrope_xi1(n))(n0))
    g_fd = float(_central_fd(lambda n: polytrope_xi1(n), n0, 1e-4))
    assert abs(g_ad - g_fd) <= 1e-4 * max(1.0, abs(g_fd))


def test_solve_isothermal_edge_mass_gradient_matches_finite_difference():
    """The Bonnor-Ebert branch is what hydrax differentiates -- lock its AD contract.

    The dimensionless enclosed mass at the truncation radius, ``m(xi_max)``, is
    differentiable in the truncation radius through the diffrax solve (the integration
    bound and the output grid both depend on ``xi_max``).
    """
    xm0 = 3.0

    def edge_mass(xi_max):
        return solve_isothermal(xi_max=xi_max, n_points=400).m[-1]

    g_ad = float(jax.grad(edge_mass)(xm0))
    g_fd = float(_central_fd(edge_mass, xm0, 1e-4))
    assert jnp.isfinite(g_ad)
    assert g_ad != 0.0
    assert abs(g_ad - g_fd) <= 1e-3 * max(1.0, abs(g_fd))


def test_solve_polytrope_edge_mass_gradient_matches_finite_difference():
    """The integrated enclosed mass at the edge is differentiable in n."""
    n0 = 1.5
    xi_max = float(polytrope_xi1(n0))  # static bound; differentiate the solve in n only

    def edge_mass(n):
        return solve_polytrope(n, xi_max=xi_max, n_points=400).m[-1]

    g_ad = float(jax.grad(edge_mass)(n0))
    g_fd = float(_central_fd(edge_mass, n0, 1e-4))
    assert jnp.isfinite(g_ad)
    assert abs(g_ad - g_fd) <= 1e-3 * max(1.0, abs(g_fd))
