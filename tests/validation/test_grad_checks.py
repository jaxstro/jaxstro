# tests/test_grad_checks.py
"""
Finite-difference-vs-autodiff (FD-vs-AD) grad-check sweep across EVERY
differentiable public numeric primitive in ``jaxstro.numerics``.

This is the consolidated gradient-validation gate (research-workflow
``gradient-validation``): for each differentiable primitive we build a scalar
loss, differentiate it with ``jax.grad``/``jax.jacrev`` (NEVER ``jacfwd`` or
``hessian`` through a custom_vjp / scan-heavy path), and compare against a
central finite difference of the same loss. A mismatch means the autodiff path
is wrong (silent zero gradient, blocked gradient, sign error, ...).

Some primitives already carry FD-vs-AD checks in their own test modules
(``test_integration_parity.py`` for ``cumulative_trapz``, ``test_sampling.py``
for ``inverse_cdf_draw``, ``test_quadrature.py`` for ``hermite_coefficients``,
``test_numerics.py`` for ``newton_ppf``). This module FILLS THE GAPS so the
coverage across the differentiable surface is complete, and centralizes the
FD helper + x64 guard.

x64 GUARD (fail-loud)
---------------------
These FD-vs-AD checks rely on float64: a float32 central difference of a
smooth function underflows / is swamped by rounding noise at the step sizes
used here, which would make the comparison meaningless (or spuriously pass at
huge tolerance). ``conftest.py`` enables x64 before collection; this module
ASSERTS it at import time so the suite fails loud if x64 ever regresses,
rather than silently degrading these gradient checks.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.numerics import (
    compensated,
    integration,
    interpolation,
    linear_algebra,
    regular_grid,
    rootfinding,
    splines,
    stats,
)

# --- x64 fail-loud guard -----------------------------------------------------
# If x64 ever regresses (conftest not loaded, env override, ...), the FD-vs-AD
# comparisons below become numerically meaningless. Fail the whole module loud.
if jnp.zeros(()).dtype != jnp.float64:
    raise RuntimeError(
        "test_grad_checks requires float64 (x64) to be enabled; got "
        f"{jnp.zeros(()).dtype}. conftest.py should set jax_enable_x64=True "
        "before collection. FD-vs-AD grad-checks are invalid in float32."
    )


def fd_grad(f, x, eps=1e-6):
    """Central finite-difference gradient of scalar-output ``f`` at array ``x``.

    Perturbs each component of ``x`` by ``+/- eps`` and forms the central
    difference. ``x`` may be a scalar or 1D array; returns an array of the same
    shape as ``x``.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    flat = x.ravel()
    grad = np.zeros(flat.shape, dtype=np.float64)
    for i in range(flat.shape[0]):
        plus = flat.at[i].add(eps).reshape(x.shape)
        minus = flat.at[i].add(-eps).reshape(x.shape)
        grad[i] = float((f(plus) - f(minus)) / (2.0 * eps))
    return jnp.asarray(grad.reshape(x.shape))


def assert_grad_matches(f, x, *, eps=1e-6, atol=1e-6, rtol=1e-5):
    """Assert ``jax.grad(f)(x)`` matches the central FD gradient of ``f``."""
    ad = jax.grad(f)(jnp.asarray(x, dtype=jnp.float64))
    fd = fd_grad(f, x, eps=eps)
    np.testing.assert_allclose(np.asarray(ad), np.asarray(fd), atol=atol, rtol=rtol)


# =============================================================================
# stats
# =============================================================================
class TestStatsGradChecks:
    def test_safe_log(self):
        # Inputs well above the eps floor so the clip is inactive.
        x = jnp.array([0.5, 1.0, 3.0])
        assert_grad_matches(lambda v: jnp.sum(stats.safe_log(v)), x)

    def test_safe_exp(self):
        # Below max_exp so the clip is inactive.
        x = jnp.array([-1.0, 0.0, 2.0])
        assert_grad_matches(lambda v: jnp.sum(stats.safe_exp(v)), x)

    def test_safe_div(self):
        num = jnp.array([1.0, 2.0, -3.0])
        den = jnp.array([2.0, 4.0, 5.0])
        # Differentiate wrt numerator and denominator separately.
        assert_grad_matches(lambda n: jnp.sum(stats.safe_div(n, den)), num)
        assert_grad_matches(lambda d: jnp.sum(stats.safe_div(num, d)), den)

    def test_logsumexp(self):
        x = jnp.array([0.1, 1.3, -0.7, 2.2])
        assert_grad_matches(lambda v: stats.logsumexp(v), x)

    def test_gaussian_logpdf(self):
        x = jnp.array([0.2, -0.5, 1.1])
        mu = jnp.array([0.0, 0.3, -0.1])
        sigma = jnp.array([1.0, 0.8, 1.2])
        assert_grad_matches(lambda v: jnp.sum(stats.gaussian_logpdf(v, mu, sigma)), x)
        assert_grad_matches(lambda m: jnp.sum(stats.gaussian_logpdf(x, m, sigma)), mu)


# =============================================================================
# interpolation
# =============================================================================
class TestInterpolationGradChecks:
    def test_interp1d_wrt_y(self):
        x = jnp.linspace(0.0, 1.0, 6)
        x_new = jnp.array([0.13, 0.47, 0.82])
        y0 = jnp.sin(3.0 * x)
        assert_grad_matches(lambda y: jnp.sum(interpolation.interp1d(x, y, x_new)), y0)

    def test_interp1d_wrt_x_new(self):
        x = jnp.linspace(0.0, 1.0, 6)
        y = jnp.sin(3.0 * x)
        xn0 = jnp.array([0.13, 0.47, 0.82])
        assert_grad_matches(lambda xn: jnp.sum(interpolation.interp1d(x, y, xn)), xn0)

    def test_tabulated_function_call(self):
        x = jnp.linspace(0.0, 2.0, 8)
        y = x**2
        table = interpolation.TabulatedFunction1D(x=x, y=y)
        xn0 = jnp.array([0.25, 1.1, 1.7])
        assert_grad_matches(lambda xn: jnp.sum(table(xn)), xn0)

    def test_cubic_hermite_interp_wrt_y_and_derivatives(self):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        y0 = jnp.array([0.0, 0.2, 0.8, 1.0])
        dydx0 = jnp.array([0.1, 0.4, 0.2, 0.1])
        x_new = jnp.array([0.25, 1.25, 2.25])
        assert_grad_matches(
            lambda y: jnp.sum(interpolation.cubic_hermite_interp(x, y, dydx0, x_new)),
            y0,
        )
        assert_grad_matches(
            lambda dydx: jnp.sum(
                interpolation.cubic_hermite_interp(x, y0, dydx, x_new)
            ),
            dydx0,
        )

    def test_monotone_cubic_interp_wrt_y(self):
        x = jnp.array([0.0, 1.0, 2.0, 3.0])
        y0 = jnp.array([0.0, 0.4, 1.4, 3.0])
        x_new = jnp.array([0.25, 1.25, 2.25])
        assert_grad_matches(
            lambda y: jnp.sum(interpolation.monotone_cubic_interp(x, y, x_new)),
            y0,
            eps=1e-5,
            atol=1e-5,
            rtol=1e-5,
        )


# =============================================================================
# regular grid interpolation
# =============================================================================
class TestRegularGridGradChecks:
    def test_regular_grid_interp_wrt_values(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0])
        xx, yy = jnp.meshgrid(x, y, indexing="ij")
        values = xx**2 + yy
        xi = jnp.array([[0.25, 0.5], [1.5, 0.25]])
        assert_grad_matches(
            lambda v: jnp.sum(regular_grid.regular_grid_interp((x, y), v, xi)),
            values,
        )

    def test_regular_grid_interp_wrt_coordinates(self):
        x = jnp.array([0.0, 1.0, 2.0])
        y = jnp.array([0.0, 1.0])
        xx, yy = jnp.meshgrid(x, y, indexing="ij")
        values = xx**2 + yy
        xi = jnp.array([[0.25, 0.5], [1.5, 0.25]])
        assert_grad_matches(
            lambda points: jnp.sum(
                regular_grid.regular_grid_interp((x, y), values, points)
            ),
            xi,
            eps=1e-5,
            atol=1e-5,
            rtol=1e-5,
        )


# =============================================================================
# splines
# =============================================================================
class TestSplineGradChecks:
    def test_bspline_eval_wrt_coefficients(self):
        knots = splines.open_uniform_knots(0.0, 1.0, n_basis=6, degree=3)
        x = jnp.array([0.15, 0.37, 0.81])
        coeffs = jnp.sin(jnp.linspace(0.0, 1.0, 6))
        assert_grad_matches(
            lambda c: jnp.sum(splines.bspline_eval(knots, c, x, degree=3)),
            coeffs,
        )

    def test_bspline_eval_wrt_x(self):
        knots = splines.open_uniform_knots(0.0, 1.0, n_basis=6, degree=3)
        coeffs = jnp.sin(jnp.linspace(0.0, 1.0, 6))
        x0 = jnp.array([0.15, 0.37, 0.81])
        assert_grad_matches(
            lambda x: jnp.sum(splines.bspline_eval(knots, coeffs, x, degree=3)),
            x0,
            eps=1e-5,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_bspline_derivative_wrt_coefficients(self):
        knots = splines.open_uniform_knots(0.0, 1.0, n_basis=6, degree=3)
        x = jnp.array([0.15, 0.37, 0.81])
        coeffs = jnp.sin(jnp.linspace(0.0, 1.0, 6))
        assert_grad_matches(
            lambda c: jnp.sum(splines.bspline_derivative(knots, c, x, degree=3)),
            coeffs,
        )

    def test_fit_bspline_lstsq_wrt_sample_values(self):
        knots = splines.open_uniform_knots(0.0, 1.0, n_basis=5, degree=3)
        x = jnp.linspace(0.0, 1.0, 9)
        y0 = jnp.sin(2.0 * x)
        assert_grad_matches(
            lambda y: jnp.sum(splines.fit_bspline_lstsq(knots, x, y, degree=3)),
            y0,
            eps=1e-5,
            atol=1e-5,
            rtol=1e-5,
        )


# =============================================================================
# integration
# =============================================================================
class TestIntegrationGradChecks:
    def test_trapz_no_x(self):
        y0 = jnp.array([0.0, 1.0, 0.5, 2.0, 1.5])
        assert_grad_matches(lambda y: integration.trapz(y), y0)

    def test_trapz_with_x_wrt_y(self):
        x = jnp.array([0.0, 0.3, 0.7, 1.0, 1.6])
        y0 = jnp.array([1.0, 0.5, 2.0, 1.5, 0.2])
        assert_grad_matches(lambda y: integration.trapz(y, x), y0)

    def test_trapz_with_x_wrt_x(self):
        x0 = jnp.array([0.0, 0.3, 0.7, 1.0, 1.6])
        y = jnp.array([1.0, 0.5, 2.0, 1.5, 0.2])
        assert_grad_matches(lambda xv: integration.trapz(y, xv), x0)

    def test_cumulative_trapz_dx_path(self):
        y0 = jnp.array([0.0, 1.0, 0.5, 2.0, 1.5])
        assert_grad_matches(
            lambda y: jnp.sum(integration.cumulative_trapz(y, dx=0.3)), y0
        )

    def test_cumulative_trapz_x_path(self):
        x = jnp.array([0.0, 0.3, 0.7, 1.0, 1.6])
        y0 = jnp.array([1.0, 0.5, 2.0, 1.5, 0.2])
        assert_grad_matches(lambda y: jnp.sum(integration.cumulative_trapz(y, x)), y0)

    def test_simpson_no_x(self):
        y0 = jnp.array([0.0, 1.0, 0.5, 2.0, 1.5])  # odd count
        assert_grad_matches(lambda y: integration.simpson(y), y0)

    def test_simpson_with_x(self):
        x = jnp.linspace(0.0, 1.0, 5)  # uniform, odd
        y0 = jnp.array([0.2, 1.0, 0.5, 2.0, 1.5])
        assert_grad_matches(lambda y: integration.simpson(y, x), y0)

    def test_cumulative_simpson(self):
        x = jnp.linspace(0.0, 1.0, 7)
        y0 = jnp.sin(x)
        assert_grad_matches(lambda y: jnp.sum(integration.cumulative_simpson(y, x)), y0)


# =============================================================================
# rootfinding  (bisect, newton; newton_ppf already covered in test_numerics)
# =============================================================================
class TestRootfindingGradChecks:
    def test_bisect_grad_wrt_target_is_structurally_zero(self):
        """bisect has a STRUCTURALLY ZERO gradient w.r.t. the target/function.

        The root only enters the iteration through ``jnp.sign(...)`` sign
        comparisons (which half-bracket contains the root). ``sign`` has a zero
        derivative a.e., so ``d(root)/d(c) == 0`` even though the true sensitivity
        d(sqrt(c))/dc = 1/(2 sqrt(c)) is nonzero. This is a known limitation:
        bisect is NOT a differentiable solver w.r.t. the function parameters —
        use ``newton`` when you need d(root)/d(param). We assert the zero here so
        the limitation is documented and a future change that silently alters it
        fails loud.
        """

        def root(c):
            return rootfinding.bisect(lambda x: x**2 - c, 0.0, 2.0, max_steps=60)

        ad = jax.grad(root)(jnp.asarray(2.0))
        assert float(ad) == 0.0

    def test_bisect_grad_wrt_bracket_matches_fd(self):
        """bisect IS differentiable w.r.t. the bracket endpoints a, b.

        The endpoints enter linearly through the midpoints ``0.5*(a+b)``, so
        ``d(root)/d(a)`` is well-defined and matches a finite difference. We use a
        small, fixed ``max_steps`` so the bracket dependence is still live: at
        convergence the output snaps to the true root independent of ``a`` (FD
        ~= 0) while AD retains the early-iteration endpoint sensitivity — a known
        artifact of differentiating a (truncated) bracketing iteration. At low
        step count both agree exactly, which is what this asserts.
        """

        def root_a(a):
            return rootfinding.bisect(lambda x: x**2 - 2.0, a, 2.0, max_steps=8)

        a0 = jnp.asarray(0.5)
        assert_grad_matches(root_a, a0, eps=1e-4, atol=1e-6, rtol=1e-5)

    def test_newton_wrt_target(self):
        def root(c):
            return rootfinding.newton(lambda x: x**2 - c, x0=1.5)

        c0 = jnp.asarray(2.0)
        assert_grad_matches(root, c0, atol=1e-6, rtol=1e-5)

    def test_newton_with_grad_wrt_target(self):
        def root(c):
            return rootfinding.newton_with_grad(
                lambda x: x**2 - c, lambda x: 2.0 * x, x0=1.5
            )

        c0 = jnp.asarray(2.0)
        assert_grad_matches(root, c0, atol=1e-6, rtol=1e-5)

    def test_monotone_inverse_interp_wrt_query(self):
        x = jnp.array([0.0, 1.0, 2.5, 4.0])
        y = jnp.array([0.0, 0.25, 0.75, 1.0])
        query0 = jnp.array([0.1, 0.4, 0.9])
        assert_grad_matches(
            lambda q: jnp.sum(rootfinding.monotone_inverse_interp(x, y, q)),
            query0,
        )


# =============================================================================
# compensated
# =============================================================================
class TestCompensatedGradChecks:
    def test_compensated_sum_array(self):
        x0 = jnp.array([1.0, 2.0, 3.0, 4.0])
        assert_grad_matches(lambda x: compensated.compensated_sum_array(x), x0)

    def test_compensated_sum_variadic(self):
        # Differentiate wrt the first stacked term.
        b = jnp.array([0.5, -1.0])
        a0 = jnp.array([1.0, 2.0])
        assert_grad_matches(lambda a: jnp.sum(compensated.compensated_sum(a, b)), a0)

    def test_compensated_dot(self):
        b = jnp.array([1.0, 1.0, 1.0])
        a0 = jnp.array([2.0, -3.0, 5.0])
        assert_grad_matches(lambda a: compensated.compensated_dot(a, b), a0)


# =============================================================================
# linear_algebra
# =============================================================================
class TestLinearAlgebraGradChecks:
    def test_norm2(self):
        x0 = jnp.array([3.0, 4.0, 1.0])
        assert_grad_matches(lambda x: linear_algebra.norm2(x), x0)

    def test_project_onto_wrt_a(self):
        b = jnp.array([1.0, 2.0, 2.0])
        a0 = jnp.array([0.5, -1.0, 3.0])
        assert_grad_matches(lambda a: jnp.sum(linear_algebra.project_onto(a, b)), a0)

    def test_project_onto_wrt_b(self):
        a = jnp.array([0.5, -1.0, 3.0])
        b0 = jnp.array([1.0, 2.0, 2.0])
        assert_grad_matches(lambda b: jnp.sum(linear_algebra.project_onto(a, b)), b0)

    def test_weighted_lstsq_wrt_observations(self):
        design = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        weights = jnp.array([1.0, 0.5, 2.0, 1.5])
        y0 = jnp.array([1.0, 2.8, 5.2, 7.1])
        assert_grad_matches(
            lambda y: jnp.sum(linear_algebra.weighted_lstsq(design, y, weights)),
            y0,
        )

    def test_svd_solve_wrt_rhs(self):
        A = jnp.array([[3.0, 0.2], [0.2, 1.5]])
        b0 = jnp.array([1.0, 2.0])
        assert_grad_matches(lambda b: jnp.sum(linear_algebra.svd_solve(A, b)), b0)

    def test_covariance_matrix_wrt_samples(self):
        samples0 = jnp.array([[0.0, 1.0], [1.0, 2.0], [3.0, 5.0], [4.0, 8.0]])
        assert_grad_matches(
            lambda x: jnp.sum(linear_algebra.covariance_matrix(x, ddof=1)),
            samples0,
        )

    def test_correlation_from_covariance_wrt_covariance(self):
        cov0 = jnp.array([[2.0, 0.3], [0.3, 1.5]])
        assert_grad_matches(
            lambda cov: jnp.sum(linear_algebra.correlation_from_covariance(cov)),
            cov0,
        )


# =============================================================================
# jacrev smoke: array-valued outputs (NOT jacfwd / hessian)
# =============================================================================
class TestJacrevSweep:
    def test_cumulative_trapz_jacrev(self):
        y = jnp.array([0.0, 1.0, 0.5, 2.0, 1.5])
        jac = jax.jacrev(lambda v: integration.cumulative_trapz(v, dx=0.3))(y)
        assert jac.shape == (y.shape[0], y.shape[0])
        assert jnp.all(jnp.isfinite(jac))

    def test_interp1d_jacrev(self):
        x = jnp.linspace(0.0, 1.0, 6)
        x_new = jnp.array([0.13, 0.47, 0.82])
        y = jnp.sin(3.0 * x)
        jac = jax.jacrev(lambda v: interpolation.interp1d(x, v, x_new))(y)
        assert jac.shape == (x_new.shape[0], y.shape[0])
        assert jnp.all(jnp.isfinite(jac))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
