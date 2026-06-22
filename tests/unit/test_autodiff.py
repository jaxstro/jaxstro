"""Tests for small autodiff product helpers."""

import jax
import jax.numpy as jnp

from jaxstro.numerics import autodiff


class TestFirstOrderProducts:
    """Tests for JVP and VJP helpers."""

    def test_jvp_matches_jax_jvp(self):
        def f(x):
            return jnp.array([x[0] + x[1] ** 2, jnp.sin(x[0])])

        x = jnp.array([0.3, -0.7])
        v = jnp.array([1.5, -0.2])
        primal, tangent = autodiff.jvp(f, x, v)
        expected_primal, expected_tangent = jax.jvp(f, (x,), (v,))
        assert jnp.allclose(primal, expected_primal)
        assert jnp.allclose(tangent, expected_tangent)

    def test_vjp_matches_explicit_jacobian_transpose_product(self):
        def f(x):
            return jnp.array([x[0] + x[1] ** 2, x[0] * x[1]])

        x = jnp.array([0.3, -0.7])
        cotangent = jnp.array([2.0, -1.0])
        value, product = autodiff.vjp(f, x, cotangent)
        jac = jax.jacrev(f)(x)
        assert jnp.allclose(value, f(x))
        assert jnp.allclose(product, jac.T @ cotangent)

    def test_jacobian_vector_product_alias_returns_tangent_only(self):
        def f(x):
            return jnp.array([x[0] * x[1], x[1] ** 3])

        x = jnp.array([2.0, -0.5])
        v = jnp.array([0.25, 1.5])
        assert jnp.allclose(
            autodiff.jacobian_vector_product(f, x, v),
            jax.jacrev(f)(x) @ v,
        )

    def test_vector_jacobian_product_alias_returns_product_only(self):
        def f(x):
            return jnp.array([x[0] ** 2, x[0] - x[1]])

        x = jnp.array([0.4, 1.2])
        cotangent = jnp.array([3.0, -2.0])
        assert jnp.allclose(
            autodiff.vector_jacobian_product(f, x, cotangent),
            jax.jacrev(f)(x).T @ cotangent,
        )


class TestSecondOrderProducts:
    """Tests for HVP, Gauss-Newton, and Fisher-style products."""

    def test_hvp_matches_explicit_hessian_product(self):
        def f(x):
            return jnp.sum(jnp.sin(x) + 0.5 * x**2)

        x = jnp.array([0.2, -0.4, 0.7])
        v = jnp.array([1.0, -2.0, 0.5])
        assert jnp.allclose(autodiff.hvp(f, x, v), jax.hessian(f)(x) @ v)

    def test_gauss_newton_product_matches_dense_jtj_product(self):
        def residuals(x):
            return jnp.array([x[0] + 2.0 * x[1], x[0] ** 2 - x[1]])

        x = jnp.array([0.5, -1.0])
        v = jnp.array([0.3, 0.7])
        jac = jax.jacrev(residuals)(x)
        assert jnp.allclose(
            autodiff.gauss_newton_product(residuals, x, v),
            jac.T @ (jac @ v),
        )

    def test_empirical_fisher_product_matches_mean_outer_score_product(self):
        data = jnp.array([0.0, 1.0, 2.0])

        def score(params, datum):
            return jnp.array([datum - params[0], params[1] * datum])

        params = jnp.array([0.5, 2.0])
        v = jnp.array([1.0, -0.25])
        scores = jax.vmap(lambda datum: score(params, datum))(data)
        expected = scores.T @ (scores @ v) / data.shape[0]
        assert jnp.allclose(
            autodiff.empirical_fisher_product(score, params, data, v),
            expected,
        )

    def test_helpers_are_jit_compatible(self):
        def f(x):
            return jnp.sum(x**3)

        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.5, -1.0])
        result = jax.jit(autodiff.hvp, static_argnames=("f",))(f, x, v)
        assert jnp.all(jnp.isfinite(result))
