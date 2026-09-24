"""Adaptive rules near endpoints, with regions stored as (1 + t, 1 - t).

Regression cases from the 2026-09-24 audit and adversarial review. A result is
honest when CONVERGED implies an error within tolerance; otherwise the
estimate exceeds the tolerance.
"""

import math

import jax.numpy as jnp
import pytest

from jaxstro import quad

TS3 = quad.AdaptiveTanhSinh(3)
TS5 = quad.AdaptiveTanhSinh(5)
GK = quad.GaussKronrod(21)


def _integrate(fun, domain, method, epsrel=1e-10):
    return quad.integrate(
        fun,
        domain,
        method=method,
        epsabs=0.0,
        epsrel=epsrel,
        max_evaluations=20000,
        max_regions=500,
    )


def _assert_honest(result, exact):
    error = abs(float(result.value) - exact)
    assert math.isfinite(float(result.value))
    if result.status == quad.QuadStatus.CONVERGED:
        assert error <= float(result.tolerance)
    else:
        assert float(result.error.norm) > float(result.tolerance)


@pytest.mark.parametrize("method", [GK, TS3, TS5], ids=["gk", "ts3", "ts5"])
@pytest.mark.parametrize(
    "fun,domain,exact",
    [
        (lambda x: (x - 1.0) ** -0.5, quad.Interval(1.0, 2.0), 2.0),
        (lambda x: (2.0 - x) ** -0.5, quad.Interval(1.0, 2.0), 2.0),
        (lambda x: x**-0.9, quad.Interval(0.0, 1.0), 10.0),
    ],
    ids=["left-sqrt", "right-sqrt", "x^-0.9"],
)
def test_finite_endpoint_singularities_are_finite_and_honest(
    method, fun, domain, exact
) -> None:
    # Every method returned NaN here: nodes rounded onto the singular endpoint.
    _assert_honest(_integrate(fun, domain, method), exact)


def test_gauss_kronrod_resolves_a_singularity_at_zero() -> None:
    # Complements keep nodes distinct from 0 down to the smallest normal number.
    result = _integrate(lambda x: x**-0.9, quad.Interval(0.0, 1.0), GK)
    assert result.status == quad.QuadStatus.CONVERGED
    assert abs(float(result.value) - 10.0) <= 1e-9 * 10.0


@pytest.mark.parametrize(
    "fun,domain,exact,epsrel",
    [
        (
            lambda x: x**-0.5 * jnp.exp(-x),
            quad.RightInfinite(0.0),
            math.sqrt(math.pi),
            1e-10,
        ),
        (
            lambda x: (-x) ** -0.5 * jnp.exp(x),
            quad.LeftInfinite(0.0),
            math.sqrt(math.pi),
            1e-10,
        ),
        (
            lambda x: x**-0.9 * jnp.exp(-x),
            quad.RightInfinite(0.0),
            math.gamma(0.1),
            1e-3,
        ),
        (lambda x: x**-1.5, quad.RightInfinite(1.0), 2.0, 1e-12),
    ],
    ids=["sqrt-exp", "left-sqrt-exp", "x^-0.9-exp", "x^-1.5-tail"],
)
def test_improper_domain_singularities_are_honest(fun, domain, exact, epsrel) -> None:
    # The dropped-node patch (31f1c31, reverted) converged falsely on these.
    _assert_honest(_integrate(fun, domain, TS3, epsrel), exact)


@pytest.mark.parametrize("method", [TS3, TS5], ids=["ts3", "ts5"])
def test_tanh_sinh_refines_an_interior_peak(method) -> None:
    # Nodes landing on interior split boundaries stopped tanh-sinh after one
    # split at a 0.89 relative error.
    width = 1e-4
    exact = (math.atan(0.7 / width) + math.atan(0.3 / width)) / width
    result = _integrate(
        lambda x: 1.0 / (width**2 + (x - 0.3) ** 2), quad.Interval(0.0, 1.0), method
    )
    assert result.status == quad.QuadStatus.CONVERGED
    assert abs(float(result.value) - exact) <= 1e-10 * exact


@pytest.mark.parametrize("ulps", [2, 3, 8, 64])
@pytest.mark.parametrize("method", [GK, TS3], ids=["gk", "ts3"])
def test_intervals_a_few_ulps_wide_stay_exact(method, ulps) -> None:
    upper = 1.0 + ulps * float(jnp.spacing(jnp.asarray(1.0)))
    result = _integrate(lambda x: x, quad.Interval(1.0, upper), method)
    exact = 0.5 * (upper * upper - 1.0)
    assert float(result.value) == pytest.approx(exact, rel=1e-12)
