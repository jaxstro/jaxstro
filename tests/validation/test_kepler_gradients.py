import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from jaxstro.numerics import universal_kepler_step  # noqa: E402


def _objective(state: jax.Array, mu: jax.Array, dt: jax.Array) -> jax.Array:
    result = universal_kepler_step(state[:3], state[3:], mu, dt)
    weights = jnp.array([0.7, -0.2, 0.4, -0.1, 0.5, 0.3])
    return jnp.dot(
        jnp.concatenate((result.position, result.velocity)),
        weights,
    )


def _final_stumpff_argument(
    state: jax.Array,
    anomaly: jax.Array,
    mu: jax.Array,
) -> jax.Array:
    radius = jnp.linalg.norm(state[:3])
    alpha_bar = 2.0 - radius * jnp.dot(state[3:], state[3:]) / mu
    anomaly_bar = anomaly / jnp.sqrt(radius)
    return alpha_bar * anomaly_bar**2


@pytest.mark.parametrize(
    "state",
    [
        jnp.array([0.7, 0.1, 0.0, -0.1, 1.1, 0.05]),
        jnp.array([1.0, 0.2, 0.1, 0.1, 1.7, 0.2]),
    ],
    ids=["elliptic", "hyperbolic"],
)
def test_universal_kepler_fixed_route_jvp_and_vjp_match_central_difference(
    state: jax.Array,
) -> None:
    mu = jnp.array(1.0)
    dt = jnp.array(0.3)
    direction = jnp.array([0.2, -0.3, 0.1, 0.4, -0.2, 0.35])
    direction = direction / jnp.linalg.norm(direction)
    step = jnp.array(1.0e-6)

    center = universal_kepler_step(state[:3], state[3:], mu, dt)
    plus_state = state + step * direction
    minus_state = state - step * direction
    plus = universal_kepler_step(plus_state[:3], plus_state[3:], mu, dt)
    minus = universal_kepler_step(minus_state[:3], minus_state[3:], mu, dt)

    assert bool(center.valid) and bool(plus.valid) and bool(minus.valid)
    assert int(plus.status) == int(center.status) == int(minus.status)
    assert int(plus.iterations) == int(center.iterations) == int(minus.iterations)
    for route_state, result in (
        (state, center),
        (plus_state, plus),
        (minus_state, minus),
    ):
        z = _final_stumpff_argument(route_state, result.universal_anomaly, mu)
        assert abs(abs(float(z)) - 1.0) > 0.1

    def objective(value: jax.Array) -> jax.Array:
        return _objective(value, mu, dt)

    _, jvp_derivative = jax.jvp(objective, (state,), (direction,))
    _, pullback = jax.vjp(objective, state)
    vjp_gradient = pullback(jnp.array(1.0))[0]
    vjp_derivative = jnp.dot(vjp_gradient, direction)
    finite_difference = (
        objective(plus_state) - objective(minus_state)
    ) / (2.0 * step)
    scale = jnp.maximum(jnp.abs(finite_difference), 1.0e-12)

    assert float(jnp.abs(jvp_derivative - finite_difference) / scale) <= 1.0e-5
    assert float(jnp.abs(vjp_derivative - finite_difference) / scale) <= 1.0e-5
