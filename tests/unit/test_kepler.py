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
)
from jaxstro.numerics.kepler import _stumpff_pair, _stumpff_series  # noqa: E402


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
