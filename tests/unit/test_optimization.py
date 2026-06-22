"""Tests for small JAX-native optimization helpers."""

import jax
import jax.numpy as jnp

from jaxstro.numerics import optimization


class TestRobustLosses:
    """Tests for elementwise robust residual losses."""

    def test_squared_loss_is_half_residual_squared(self):
        residuals = jnp.array([-2.0, 0.0, 3.0])
        expected = jnp.array([2.0, 0.0, 4.5])
        assert jnp.allclose(optimization.squared_loss(residuals), expected)

    def test_huber_loss_matches_quadratic_and_linear_regions(self):
        residuals = jnp.array([-2.0, -0.5, 0.5, 3.0])
        result = optimization.huber_loss(residuals, delta=1.0)
        expected = jnp.array([1.5, 0.125, 0.125, 2.5])
        assert jnp.allclose(result, expected)

    def test_pseudo_huber_loss_is_smooth_and_grad_finite(self):
        residuals = jnp.array([-2.0, -0.1, 0.0, 0.1, 2.0])

        def loss(r):
            return jnp.sum(optimization.pseudo_huber_loss(r, delta=0.7))

        grad = jax.grad(loss)(residuals)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.allclose(grad[2], 0.0)

    def test_losses_support_jit_and_vmap(self):
        residuals = jnp.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]])
        f = jax.jit(jax.vmap(lambda row: optimization.huber_loss(row, delta=1.0)))
        result = f(residuals)
        assert result.shape == residuals.shape
        assert jnp.all(jnp.isfinite(result))


class TestObjectiveSummary:
    """Tests for scalar summaries of residual vectors."""

    def test_unweighted_summary_reports_standard_residual_metrics(self):
        residuals = jnp.array([1.0, -2.0, 2.0])
        summary = optimization.objective_summary(residuals)
        assert jnp.allclose(summary["loss"], 4.5)
        assert jnp.allclose(summary["mean_loss"], 1.5)
        assert jnp.allclose(summary["rmse"], jnp.sqrt(3.0))
        assert jnp.allclose(summary["max_abs_residual"], 2.0)
        assert summary["n"] == 3

    def test_weighted_summary_uses_weights_in_loss_and_rmse(self):
        residuals = jnp.array([1.0, 10.0])
        weights = jnp.array([1.0, 0.0])
        summary = optimization.objective_summary(residuals, weights=weights)
        assert jnp.allclose(summary["loss"], 0.5)
        assert jnp.allclose(summary["mean_loss"], 0.5)
        assert jnp.allclose(summary["rmse"], 1.0)

    def test_summary_is_jit_compatible(self):
        residuals = jnp.array([1.0, -2.0, 3.0])
        summary = jax.jit(optimization.objective_summary)(residuals)
        assert jnp.allclose(summary["loss"], 7.0)


class TestLineSearch:
    """Tests for fixed-iteration Armijo backtracking."""

    def test_armijo_backtracking_accepts_descent_step(self):
        def f(x):
            return jnp.sum((x - 1.0) ** 2)

        x0 = jnp.array([0.0])
        grad0 = jax.grad(f)(x0)
        result = optimization.armijo_backtracking(f, x0, -grad0, grad0)
        assert result.accepted
        assert result.step <= 1.0
        assert result.value < f(x0)

    def test_armijo_backtracking_is_jit_compatible_with_static_objective(self):
        def f(x):
            return jnp.sum((x - 2.0) ** 2)

        x0 = jnp.array([0.0, 3.0])
        grad0 = jax.grad(f)(x0)
        search = jax.jit(
            optimization.armijo_backtracking,
            static_argnames=("f", "max_steps"),
        )
        result = search(f, x0, -grad0, grad0, max_steps=12)
        assert result.accepted
        assert jnp.isfinite(result.value)


class TestConvergenceDiagnostics:
    """Tests for optimizer-agnostic convergence diagnostics."""

    def test_relative_step_norm_uses_scale_floor(self):
        x_old = jnp.array([0.0, 0.0])
        x_new = jnp.array([3.0, 4.0])
        assert jnp.allclose(
            optimization.relative_step_norm(x_new, x_old, scale_floor=1.0),
            5.0,
        )

    def test_convergence_summary_combines_step_gradient_and_loss(self):
        summary = optimization.convergence_summary(
            x_new=jnp.array([1.0, 1.0 + 1e-8]),
            x_old=jnp.array([1.0, 1.0]),
            grad=jnp.array([1e-9, -2e-9]),
            loss_new=jnp.array(1.0),
            loss_old=jnp.array(1.0 + 1e-10),
            step_tol=1e-6,
            grad_tol=1e-6,
            loss_tol=1e-6,
        )
        assert summary["converged"]
        assert summary["step_converged"]
        assert summary["grad_converged"]
        assert summary["loss_converged"]

    def test_gradient_inf_norm_supports_vmap(self):
        grads = jnp.array([[1.0, -3.0], [0.25, -0.5]])
        result = jax.vmap(optimization.gradient_inf_norm)(grads)
        assert jnp.allclose(result, jnp.array([3.0, 0.5]))
