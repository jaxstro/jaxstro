"""Tests for fixed-step ODE integration helpers."""

import jax
import jax.numpy as jnp

from jaxstro.numerics import ode


class TestFixedStepSolvers:
    """Tests for scan-based explicit ODE solvers."""

    def test_euler_integrates_constant_derivative(self):
        def rhs(y, t):
            return jnp.ones_like(y) * 2.0

        result = ode.euler(rhs, y0=jnp.array([1.0]), t0=0.0, dt=0.25, num_steps=4)
        assert result.t.shape == (5,)
        assert result.y.shape == (5, 1)
        assert jnp.allclose(result.y[-1], jnp.array([3.0]))

    def test_midpoint_is_second_order_for_exponential_growth(self):
        def rhs(y, t):
            return y

        result = ode.midpoint(rhs, y0=jnp.array(1.0), t0=0.0, dt=0.01, num_steps=100)
        assert jnp.allclose(result.y[-1], jnp.exp(1.0), rtol=5e-5)

    def test_rk4_is_accurate_for_exponential_growth(self):
        def rhs(y, t):
            return y

        result = ode.rk4(rhs, y0=jnp.array(1.0), t0=0.0, dt=0.1, num_steps=10)
        assert jnp.allclose(result.y[-1], jnp.exp(1.0), rtol=1e-6)

    def test_solve_fixed_step_dispatches_method(self):
        def rhs(y, t):
            return -y

        result = ode.solve_fixed_step(
            rhs,
            y0=jnp.array(2.0),
            t0=0.0,
            dt=0.05,
            num_steps=20,
            method="rk4",
        )
        assert jnp.allclose(result.y[-1], 2.0 * jnp.exp(-1.0), rtol=1e-6)

    def test_rk4_supports_jit_vmap_and_grad(self):
        def final_value(y0):
            def rhs(y, t):
                return -0.5 * y

            return ode.rk4(rhs, y0=y0, t0=0.0, dt=0.1, num_steps=10).y[-1]

        y0 = jnp.array([1.0, 2.0, 3.0])
        values = jax.jit(jax.vmap(final_value))(y0)
        grads = jax.vmap(jax.grad(final_value))(y0)
        assert values.shape == y0.shape
        assert jnp.all(jnp.isfinite(grads))
        assert jnp.allclose(grads, jnp.exp(-0.5), rtol=1e-6)


class TestVelocityVerlet:
    """Tests for separable second-order systems."""

    def test_velocity_verlet_preserves_harmonic_oscillator_energy(self):
        def acceleration(q, t):
            return -q

        result = ode.velocity_verlet(
            acceleration,
            q0=jnp.array([1.0]),
            v0=jnp.array([0.0]),
            t0=0.0,
            dt=0.01,
            num_steps=1000,
        )
        energy = 0.5 * result.v[:, 0] ** 2 + 0.5 * result.q[:, 0] ** 2
        drift = jnp.max(jnp.abs(energy - energy[0]))
        assert drift < 2e-5

    def test_velocity_verlet_is_jit_and_grad_compatible(self):
        def final_position(q0):
            def acceleration(q, t):
                return -q

            result = ode.velocity_verlet(
                acceleration,
                q0=q0,
                v0=jnp.array([0.0]),
                t0=0.0,
                dt=0.05,
                num_steps=20,
            )
            return result.q[-1, 0]

        compiled = jax.jit(final_position)
        q0 = jnp.array([1.0])
        assert jnp.isfinite(compiled(q0))
        assert jnp.isfinite(jax.grad(final_position)(q0)[0])
