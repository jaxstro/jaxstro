"""Adaptive rules near endpoints: no evaluation at an endpoint by rounding,
and roundoff reported only at original domain endpoints (2026-09-24 audit)."""

import jax.numpy as jnp
import pytest

from jaxstro import quad


def _integrate(fun, domain, method, **overrides):
    options = dict(
        method=method,
        epsabs=0.0,
        epsrel=1e-10,
        max_evaluations=20000,
        max_regions=500,
    )
    options.update(overrides)
    return quad.integrate(fun, domain, **options)


@pytest.mark.parametrize(
    "method", [quad.GaussKronrod(21), quad.AdaptiveTanhSinh(5)], ids=["gk", "ts"]
)
@pytest.mark.parametrize(
    "fun,domain,exact",
    [
        (lambda x: (x - 1.0) ** -0.5, quad.Interval(1.0, 2.0), 2.0),
        (lambda x: (2.0 - x) ** -0.5, quad.Interval(1.0, 2.0), 2.0),
        (lambda x: x**-0.9, quad.Interval(0.0, 1.0), 10.0),
    ],
    ids=["left-sqrt", "right-sqrt", "x^-0.9"],
)
def test_endpoint_singularity_gives_a_finite_value(method, fun, domain, exact) -> None:
    # Every method returned NaN / NONFINITE_INTEGRAND: nodes rounded onto the
    # singular endpoint. The value must now be finite with an honest status.
    result = _integrate(fun, domain, method)
    assert jnp.isfinite(result.value)
    assert result.status != quad.QuadStatus.NONFINITE_INTEGRAND
    if result.status == quad.QuadStatus.CONVERGED:
        assert abs(float(result.value) - exact) <= 1e-8 * exact


def test_tanh_sinh_refines_an_interior_peak() -> None:
    # Nodes landing on interior split boundaries set the roundoff flag, so
    # AdaptiveTanhSinh stopped after one split at a 0.89 relative error.
    width = 1e-4
    exact = (jnp.arctan(0.7 / width) + jnp.arctan(0.3 / width)) / width
    result = _integrate(
        lambda x: 1.0 / (width**2 + (x - 0.3) ** 2),
        quad.Interval(0.0, 1.0),
        quad.AdaptiveTanhSinh(5),
    )
    assert result.status == quad.QuadStatus.CONVERGED
    assert abs(float(result.value) - float(exact)) <= 1e-8 * float(exact)
