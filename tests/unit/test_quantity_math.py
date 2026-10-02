"""Tests for dimension-aware quantity math wrappers."""

import jax.numpy as jnp
import pytest

from jaxstro import quantity as q
from jaxstro.quantity.errors import DimensionError


def test_sqrt_and_square_update_units():
    area = 9.0 * q.cm**2
    length = q.math.sqrt(area)

    assert length.unit == q.cm
    assert length.value == 3.0
    assert q.math.square(2.0 * q.cm).unit == q.cm**2


def test_log_and_exp_require_dimensionless_input():
    logged = q.math.log(q.Quantity(jnp.e, q.dimensionless))
    exponentiated = q.math.exp(q.Quantity(1.0, q.dimensionless))

    assert logged.unit is q.dimensionless
    assert logged.value == pytest.approx(1.0)
    assert exponentiated.unit is q.dimensionless
    with pytest.raises(DimensionError):
        q.math.log(1.0 * q.cm)


def test_log_and_exp_normalize_scaled_dimensionless_input():
    percent = q.Unit("percent", 0.01, q.dimensionless.dimensions)

    assert q.math.log(50.0 * percent).value == pytest.approx(jnp.log(0.5))
    assert q.math.exp(50.0 * percent).value == pytest.approx(jnp.exp(0.5))


def test_trig_functions_accept_tagged_angles():
    assert q.math.sin(90.0 * q.deg).value == pytest.approx(1.0)
    assert q.math.cos(0.0 * q.rad).value == pytest.approx(1.0)
    with pytest.raises(DimensionError):
        q.math.sin(q.Quantity(1.0, q.dimensionless))


def test_scaled_angle_stays_usable_by_trigonometry():
    percent = q.Unit("percent", 0.01, q.dimensionless.dimensions)
    half = 50.0 * percent
    scaled_angles = (
        (90.0 * q.deg) * half,
        half * (90.0 * q.deg),
        (90.0 * q.deg) / (200.0 * percent),
    )

    for scaled in scaled_angles:
        assert scaled.unit.metadata.get("semantic") == "angle"
        assert scaled.to_value(q.rad) == pytest.approx(jnp.pi / 4)
        assert q.math.sin(scaled).value == pytest.approx(jnp.sqrt(0.5))
    assert q.math.sin(q.Quantity(jnp.pi / 4, q.rad)).value == pytest.approx(
        jnp.sqrt(0.5)
    )
    with pytest.raises(DimensionError):
        q.math.log(1.0 * q.rad)
    with pytest.raises(DimensionError):
        q.math.exp(1.0 * q.deg)


def test_sum_and_mean_preserve_units():
    values = jnp.array([1.0, 2.0, 3.0]) * q.cm

    assert q.math.sum(values).unit is q.cm
    assert q.math.sum(values).value == 6.0
    assert q.math.mean(values).unit is q.cm
    assert q.math.mean(values).value == 2.0


def test_where_checks_and_preserves_units():
    cond = jnp.array([True, False])
    chosen = q.math.where(
        cond, jnp.array([1.0, 2.0]) * q.m, jnp.array([50.0, 75.0]) * q.cm
    )

    assert chosen.unit is q.m
    assert jnp.allclose(chosen.value, jnp.array([1.0, 0.75]))
    with pytest.raises(DimensionError):
        q.math.where(cond, 1.0 * q.cm, 1.0 * q.s)
    with pytest.raises(DimensionError):
        q.math.where(cond, 1.0 * q.rad, 1.0 * q.dimensionless)
    angles = q.math.where(
        cond,
        jnp.array([90.0, 180.0]) * q.deg,
        jnp.array([0.0, jnp.pi]) * q.rad,
    )
    assert angles.unit is q.deg
    assert jnp.allclose(angles.value, jnp.array([90.0, 180.0]))
