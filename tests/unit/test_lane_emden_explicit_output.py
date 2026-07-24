r"""Explicit output points for the Lane-Emden solve.

Downstream solvers place their mesh where the physics needs resolution -- a collapsing
protostellar core needs a geometrically-refined grid, not a uniform one. Such a mesh must
be obtained by *evaluating the ODE at those points*, never by interpolating a uniform
solve onto them: enclosed-mass differences amplify interpolation error by ``1/dxi``, so
interpolating and then differencing destroys convergence (measured downstream in hydrax:
the hydrostatic residual stopped converging and blew up as the mesh refined).

``diffrax.SaveAt`` accepts arbitrary output points, so this costs nothing -- the adaptive
controller holds its tolerance whatever the output grid is.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.numerics.lane_emden import XI_0, solve_isothermal, solve_polytrope


def test_explicit_points_reproduce_the_uniform_path():
    """Passing the uniform grid explicitly must equal the n_points path."""
    n = 200
    xi_out = jnp.linspace(XI_0, 6.0, n)
    default = solve_isothermal(xi_max=6.0, n_points=n)
    explicit = solve_isothermal(xi_max=6.0, xi_out=xi_out)
    assert jnp.allclose(explicit.xi, default.xi, rtol=1e-12)
    assert jnp.allclose(explicit.y, default.y, rtol=1e-10)
    assert jnp.allclose(explicit.m, default.m, rtol=1e-10)


def test_geometric_grid_is_solved_not_interpolated():
    """A geometric mesh must carry the SOLVER's accuracy, not interpolation error.

    The check is independent of any interpolation: on a geometric grid the returned
    ``psi`` must satisfy the origin series to the solver's own tolerance at small ``xi``,
    which an interpolated coarse solve would not.
    """
    xi_out = jnp.geomspace(1e-3, 0.2, 60)
    sol = solve_isothermal(xi_max=0.2, xi_out=xi_out)
    series = sol.xi**2 / 6.0 - sol.xi**4 / 120.0
    assert jnp.allclose(sol.y, series, rtol=1e-5, atol=1e-14)
    assert jnp.allclose(sol.xi, xi_out, rtol=1e-12)


def test_output_points_may_be_nonuniform_and_dense_near_the_centre():
    """The collapse use case: tiny central spacing, large outer spacing."""
    xi_out = jnp.concatenate(
        [jnp.geomspace(1e-4, 1.0, 50), jnp.linspace(1.0, 6.451, 51)[1:]]
    )
    sol = solve_isothermal(xi_max=6.451, xi_out=xi_out)
    assert sol.xi.shape == xi_out.shape
    assert jnp.all(jnp.diff(sol.xi) > 0.0)
    assert jnp.all(jnp.diff(sol.m) > 0.0)  # enclosed mass still strictly increasing
    assert jnp.all(jnp.isfinite(sol.y))


def test_polytrope_accepts_explicit_points_too():
    xi_out = jnp.geomspace(1e-3, float(jnp.pi), 80)
    sol = solve_polytrope(1.0, xi_max=float(jnp.pi), xi_out=xi_out)
    exact = jnp.sin(sol.xi) / sol.xi  # n=1 closed form
    assert jnp.allclose(sol.y, exact, rtol=1e-7, atol=1e-9)


def test_explicit_points_stay_differentiable():
    xi_out = jnp.geomspace(1e-3, 6.0, 64)

    def edge_mass(xi_max):
        return solve_isothermal(xi_max=xi_max, xi_out=xi_out).m[-1]

    grad = jax.grad(edge_mass)(6.0)
    assert jnp.isfinite(grad)


def test_rejects_points_outside_the_integration_range():
    """Silent extrapolation would be a correctness trap; fail loudly instead."""
    with pytest.raises(ValueError, match="xi_out"):
        solve_isothermal(xi_max=6.0, xi_out=jnp.linspace(XI_0, 9.0, 20))
