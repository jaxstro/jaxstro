import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro import quantity as q
from jaxstro.quantity.errors import DimensionError


def test_quantity_mode_finite_interval_restores_all_scientific_units():
    result = quad.integrate(
        lambda x: 2.0 * x * q.s**-1,
        quad.Interval(0.0 * q.cm, 3.0 * q.cm),
        method=quad.GaussKronrod(21),
        epsabs=1e-10 * q.cm**2 / q.s,
        epsrel=1e-10,
        max_evaluations=63,
        max_regions=2,
        gradient="stop",
    )
    expected_unit = q.cm**2 / q.s
    assert result.value.unit == expected_unit
    assert result.error.estimate.unit == expected_unit
    assert result.error.norm.unit == expected_unit
    assert result.tolerance.unit == expected_unit


def test_quantity_mode_accepts_dimensionless_domain_with_dimensionful_integrand():
    result = quad.integrate(
        lambda x: jnp.ones_like(x.value) * q.erg,
        quad.Interval(0.0, 1.0),
        method=quad.GaussKronrod(15),
        epsabs=1e-9 * q.erg,
        epsrel=1e-9,
        max_evaluations=45,
        max_regions=2,
        gradient="stop",
    )
    assert result.value.unit == q.erg
    assert jnp.allclose(result.value.value, 1.0)


def test_fully_infinite_quantity_domain_requires_static_unit():
    domain = quad.Infinite(unit=q.cm)
    assert domain.unit == q.cm


def test_raw_fixed_and_transform_paths_reject_unit_bearing_infinite_domain():
    domain = quad.Infinite(unit=q.cm)
    message = "quantity-valued domains are supported only by quad.integrate"
    with pytest.raises(TypeError, match=message):
        quad.fixed(lambda x: x, domain, rule=quad.TanhSinhRule(3))
    with pytest.raises(TypeError, match=message):
        quad.map_domain(domain, jnp.array([0.0]))


def test_quantity_epsabs_must_match_integral_unit():
    with pytest.raises(DimensionError):
        quad.integrate(
            lambda x: x,
            quad.Interval(0.0 * q.cm, 1.0 * q.cm),
            method=quad.GaussKronrod(15),
            epsabs=1e-6 * q.s,
            epsrel=1e-6,
            max_evaluations=45,
            max_regions=2,
        )
