import inspect

import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quad import adaptive


def _one_dimensional_kwargs():
    return dict(
        method=quad.GaussKronrod(15),
        epsabs=1e-10,
        epsrel=1e-10,
        max_evaluations=45,
        max_regions=2,
    )


def test_facade_preserves_one_dimensional_result_bitwise():
    domain = quad.Interval(0.0, 1.0)
    direct = adaptive.integrate(lambda x: x**2, domain, **_one_dimensional_kwargs())
    facade = quad.integrate(lambda x: x**2, domain, **_one_dimensional_kwargs())
    assert jnp.array_equal(facade.value, direct.value)
    assert jnp.array_equal(facade.status, direct.status)
    assert facade.work == direct.work


def test_facade_requires_one_dimensional_region_capacity():
    with pytest.raises(ValueError, match="max_regions"):
        quad.integrate(
            lambda x: x,
            quad.Interval(0.0, 1.0),
            method=quad.GaussKronrod(15),
            epsabs=1e-8,
            epsrel=1e-8,
            max_evaluations=45,
        )


def test_b0_hyperrectangle_has_no_silent_default_method():
    with pytest.raises(TypeError, match="Phase B method"):
        quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
            method=quad.GaussKronrod(15),
            epsabs=1e-8,
            epsrel=1e-8,
            max_evaluations=64,
        )


def test_facade_exposes_future_capacity_names_without_kwargs_catchall():
    parameters = inspect.signature(quad.integrate).parameters
    assert "max_indices" in parameters
    assert "max_frontier" in parameters
    assert "max_nodes" in parameters
    assert "key" in parameters
    assert not any(p.kind == p.VAR_KEYWORD for p in parameters.values())
