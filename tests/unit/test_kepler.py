import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402

from jaxstro.numerics import (  # noqa: E402
    KEPLER_STATUS_CONVERGED,
    KEPLER_STATUS_INVALID_INPUT,
    KEPLER_STATUS_MAX_STEPS,
    KEPLER_STATUS_NONFINITE_ITERATION,
    KEPLER_STATUS_SINGULAR_RADIUS,
    UniversalKeplerResult,
    universal_kepler_step,
)
from jaxstro.numerics.kepler import (  # noqa: E402
    _stumpff_pair,
    _stumpff_series,
    _terminal_status,
)


def test_kepler_statuses_are_exhaustive_and_distinct() -> None:
    statuses = {
        KEPLER_STATUS_CONVERGED,
        KEPLER_STATUS_INVALID_INPUT,
        KEPLER_STATUS_NONFINITE_ITERATION,
        KEPLER_STATUS_SINGULAR_RADIUS,
        KEPLER_STATUS_MAX_STEPS,
    }
    assert len(statuses) == 5
    assert UniversalKeplerResult._fields == (
        "position",
        "velocity",
        "universal_anomaly",
        "residual",
        "iterations",
        "status",
        "valid",
    )


@pytest.mark.parametrize(
    ("z", "expected_c", "expected_s"),
    [
        (0.0, 0.5, 1.0 / 6.0),
        (1.0e-16, 0.5, 1.0 / 6.0),
        (-1.0e-16, 0.5, 1.0 / 6.0),
    ],
)
def test_stumpff_limits_are_finite(
    z: float,
    expected_c: float,
    expected_s: float,
) -> None:
    actual_c, actual_s = _stumpff_pair(jnp.asarray(z))
    assert float(actual_c) == pytest.approx(expected_c, abs=1.0e-15)
    assert float(actual_s) == pytest.approx(expected_s, abs=1.0e-15)


def _outer_stumpff_pair(z: jax.Array) -> tuple[jax.Array, jax.Array]:
    root = jnp.sqrt(jnp.abs(z))
    positive = (
        (1.0 - jnp.cos(root)) / z,
        (root - jnp.sin(root)) / root**3,
    )
    negative = (
        (jnp.cosh(root) - 1.0) / (-z),
        (jnp.sinh(root) - root) / root**3,
    )
    return jax.tree.map(lambda pos, neg: jnp.where(z > 0.0, pos, neg), positive, negative)


@pytest.mark.parametrize("boundary", [1.0, -1.0])
@pytest.mark.parametrize("component", [0, 1])
def test_stumpff_series_matches_outer_value_and_derivative(
    boundary: float,
    component: int,
) -> None:
    z = jnp.asarray(boundary)
    series_value = _stumpff_series(z)[component]
    outer_value = _outer_stumpff_pair(z)[component]
    series_derivative = jax.grad(lambda x: _stumpff_series(x)[component])(z)
    outer_derivative = jax.grad(lambda x: _outer_stumpff_pair(x)[component])(z)

    assert float(series_value) == pytest.approx(float(outer_value), abs=1.0e-12)
    assert float(series_derivative) == pytest.approx(
        float(outer_derivative), abs=1.0e-9
    )


def test_universal_kepler_closes_one_circular_period() -> None:
    r0 = jnp.array([1.0, 0.0, 0.0])
    v0 = jnp.array([0.0, 1.0, 0.0])

    result = universal_kepler_step(
        r0,
        v0,
        jnp.array(1.0),
        jnp.array(2.0 * jnp.pi),
    )

    assert bool(result.valid)
    assert int(result.status) == KEPLER_STATUS_CONVERGED
    assert jnp.allclose(result.position, r0, rtol=0.0, atol=1.0e-11)
    assert jnp.allclose(result.velocity, v0, rtol=0.0, atol=1.0e-11)
    assert abs(float(result.residual)) <= 1.0e-12


@pytest.mark.parametrize("mu", [0.0, -1.0, jnp.nan])
def test_universal_kepler_invalid_mu_returns_original_state(mu: float) -> None:
    r0 = jnp.array([1.0, 0.0, 0.0])
    v0 = jnp.array([0.0, 1.0, 0.0])

    result = universal_kepler_step(r0, v0, jnp.asarray(mu), jnp.array(0.1))

    assert not bool(result.valid)
    assert int(result.status) == KEPLER_STATUS_INVALID_INPUT
    assert jnp.array_equal(result.position, r0)
    assert jnp.array_equal(result.velocity, v0)


def _invariant_errors(
    r0: jax.Array,
    v0: jax.Array,
    r1: jax.Array,
    v1: jax.Array,
    mu: float = 1.0,
) -> tuple[float, float]:
    energy_0 = 0.5 * jnp.dot(v0, v0) - mu / jnp.linalg.norm(r0)
    energy_1 = 0.5 * jnp.dot(v1, v1) - mu / jnp.linalg.norm(r1)
    energy_scale = jnp.maximum(jnp.abs(energy_0), mu / jnp.linalg.norm(r0))
    angular_0 = jnp.cross(r0, v0)
    angular_1 = jnp.cross(r1, v1)
    angular_scale = jnp.maximum(jnp.linalg.norm(angular_0), 1.0e-15)
    return (
        float(jnp.abs(energy_1 - energy_0) / energy_scale),
        float(jnp.linalg.norm(angular_1 - angular_0) / angular_scale),
    )


def _reverse_state_error(
    r0: jax.Array,
    v0: jax.Array,
    r1: jax.Array,
    v1: jax.Array,
    dt: float,
) -> float:
    reverse = universal_kepler_step(r1, v1, jnp.array(1.0), jnp.asarray(-dt))
    assert bool(reverse.valid)
    position_error = jnp.linalg.norm(reverse.position - r0) / jnp.linalg.norm(r0)
    velocity_scale = jnp.maximum(jnp.linalg.norm(v0), 1.0)
    velocity_error = jnp.linalg.norm(reverse.velocity - v0) / velocity_scale
    return float(jnp.maximum(position_error, velocity_error))


def test_universal_kepler_closes_eccentric_ellipse() -> None:
    eccentricity = 0.6
    r0 = jnp.array([1.0 - eccentricity, 0.0, 0.0])
    v0 = jnp.array(
        [0.0, jnp.sqrt((1.0 + eccentricity) / (1.0 - eccentricity)), 0.0]
    )

    result = universal_kepler_step(r0, v0, jnp.array(1.0), jnp.array(2.0 * jnp.pi))

    assert bool(result.valid)
    assert jnp.allclose(result.position, r0, rtol=0.0, atol=1.0e-11)
    assert jnp.allclose(result.velocity, v0, rtol=0.0, atol=1.0e-11)
    assert abs(float(result.residual)) <= 1.0e-12


def test_universal_kepler_preserves_hyperbolic_invariants_and_reverses() -> None:
    r0 = jnp.array([1.0, 0.0, 0.0])
    v0 = jnp.array([0.0, 1.6, 0.0])
    dt = 0.75

    result = universal_kepler_step(r0, v0, jnp.array(1.0), jnp.asarray(dt))

    assert bool(result.valid)
    energy_error, angular_error = _invariant_errors(
        r0,
        v0,
        result.position,
        result.velocity,
    )
    assert energy_error <= 1.0e-11
    assert angular_error <= 1.0e-11
    assert abs(float(result.residual)) <= 1.0e-12
    assert _reverse_state_error(r0, v0, result.position, result.velocity, dt) <= 1.0e-11


@pytest.mark.parametrize("energy_sign", [-1.0, 1.0])
def test_universal_kepler_handles_near_parabolic_limits(energy_sign: float) -> None:
    r0 = jnp.array([1.0, 0.0, 0.0])
    v0 = jnp.array([0.0, jnp.sqrt(2.0) * (1.0 + energy_sign * 1.0e-8), 0.0])

    result = universal_kepler_step(r0, v0, jnp.array(1.0), jnp.array(0.2))

    assert bool(result.valid)
    energy_error, angular_error = _invariant_errors(
        r0,
        v0,
        result.position,
        result.velocity,
    )
    assert energy_error <= 1.0e-11
    assert angular_error <= 1.0e-11
    assert abs(float(result.residual)) <= 1.0e-12


def test_universal_kepler_is_scale_equivalent() -> None:
    r0 = jnp.array([1.0, 0.0, 0.0])
    v0 = jnp.array([0.0, 1.6, 0.0])
    base = universal_kepler_step(r0, v0, jnp.array(1.0), jnp.array(0.75))
    length_scale = 17.0
    time_scale = 5.0
    scaled = universal_kepler_step(
        length_scale * r0,
        length_scale / time_scale * v0,
        jnp.array(length_scale**3 / time_scale**2),
        jnp.array(time_scale * 0.75),
    )

    assert bool(base.valid) and bool(scaled.valid)
    assert jnp.allclose(
        scaled.position / length_scale,
        base.position,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert jnp.allclose(
        scaled.velocity / (length_scale / time_scale),
        base.velocity,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_universal_kepler_exhausted_budget_returns_original_state() -> None:
    r0 = jnp.array([1.0, 0.0, 0.0])
    v0 = jnp.array([0.0, 1.6, 0.0])

    result = universal_kepler_step(
        r0,
        v0,
        jnp.array(1.0),
        jnp.array(0.75),
        max_steps=0,
    )

    assert not bool(result.valid)
    assert int(result.status) == KEPLER_STATUS_MAX_STEPS
    assert jnp.array_equal(result.position, r0)
    assert jnp.array_equal(result.velocity, v0)


@pytest.mark.parametrize(
    ("r0", "v0", "mu", "dt"),
    [
        (jnp.zeros(3), jnp.array([0.0, 1.0, 0.0]), 1.0, 0.1),
        (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]), 1.0, jnp.nan),
    ],
)
def test_universal_kepler_rejects_other_invalid_inputs(
    r0: jax.Array,
    v0: jax.Array,
    mu: float,
    dt: float,
) -> None:
    result = universal_kepler_step(r0, v0, jnp.asarray(mu), jnp.asarray(dt))
    assert not bool(result.valid)
    assert int(result.status) == KEPLER_STATUS_INVALID_INPUT
    assert jnp.array_equal(result.position, r0, equal_nan=True)
    assert jnp.array_equal(result.velocity, v0, equal_nan=True)


def test_universal_kepler_types_nonfinite_iteration() -> None:
    r0 = jnp.array([1.0, 0.0, 0.0])
    v0 = jnp.array([0.0, 2.0, 0.0])

    result = universal_kepler_step(r0, v0, jnp.array(1.0), jnp.array(1.0e308))

    assert not bool(result.valid)
    assert int(result.status) == KEPLER_STATUS_NONFINITE_ITERATION
    assert jnp.array_equal(result.position, r0)
    assert jnp.array_equal(result.velocity, v0)


def test_universal_kepler_fails_closed_when_radial_collision_root_is_unresolved() -> None:
    r0 = jnp.array([1.0, 0.0, 0.0])
    v0 = jnp.zeros(3)
    free_fall_collision_time = jnp.pi / (2.0 * jnp.sqrt(2.0))

    result = universal_kepler_step(
        r0,
        v0,
        jnp.array(1.0),
        free_fall_collision_time,
    )

    assert not bool(result.valid)
    assert int(result.status) == KEPLER_STATUS_MAX_STEPS
    assert jnp.array_equal(result.position, r0)
    assert jnp.array_equal(result.velocity, v0)


def test_terminal_status_types_positive_singular_radius_evidence() -> None:
    status = _terminal_status(
        input_valid=jnp.asarray(True),
        finite=jnp.asarray(True),
        singular=jnp.asarray(True),
        converged=jnp.asarray(False),
    )
    assert int(status) == KEPLER_STATUS_SINGULAR_RADIUS


def test_universal_kepler_jit_matches_eager_status_and_state() -> None:
    r0 = jnp.array([1.0, 0.0, 0.0])
    v0 = jnp.array([0.0, 1.6, 0.0])
    eager = universal_kepler_step(r0, v0, jnp.array(1.0), jnp.array(0.75))

    compiled = jax.jit(universal_kepler_step)(
        r0,
        v0,
        jnp.array(1.0),
        jnp.array(0.75),
    )

    assert int(compiled.status) == int(eager.status)
    assert int(compiled.iterations) == int(eager.iterations)
    assert jnp.allclose(compiled.position, eager.position, rtol=0.0, atol=1.0e-13)
    assert jnp.allclose(compiled.velocity, eager.velocity, rtol=0.0, atol=1.0e-13)


def test_universal_kepler_vmap_matches_scalar_lanes() -> None:
    positions = jnp.array([[1.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    velocities = jnp.array([[0.0, 1.6, 0.0], [0.0, 2.0, 0.0]])
    mus = jnp.ones(2)
    dts = jnp.array([0.75, 0.4])

    batched = jax.vmap(universal_kepler_step)(positions, velocities, mus, dts)
    scalar = [
        universal_kepler_step(positions[i], velocities[i], mus[i], dts[i])
        for i in range(2)
    ]

    assert jnp.array_equal(
        batched.status,
        jnp.stack([result.status for result in scalar]),
    )
    assert jnp.array_equal(
        batched.iterations,
        jnp.stack([result.iterations for result in scalar]),
    )
    assert jnp.allclose(
        batched.position,
        jnp.stack([result.position for result in scalar]),
        rtol=0.0,
        atol=1.0e-13,
    )
    assert jnp.allclose(
        batched.velocity,
        jnp.stack([result.velocity for result in scalar]),
        rtol=0.0,
        atol=1.0e-13,
    )
