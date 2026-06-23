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


def test_trig_functions_accept_tagged_angles():
    assert q.math.sin(90.0 * q.deg).value == pytest.approx(1.0)
    assert q.math.cos(0.0 * q.rad).value == pytest.approx(1.0)
    with pytest.raises(DimensionError):
        q.math.sin(q.Quantity(1.0, q.dimensionless))


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
