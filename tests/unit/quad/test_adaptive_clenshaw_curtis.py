"""Nested adaptive Clenshaw-Curtis contracts."""

import jax.numpy as jnp

from jaxstro.quad import (
    AdaptiveClenshawCurtis,
    ErrorKind,
    Interval,
    MaxNorm,
    QuadStatus,
    integrate,
)
from jaxstro.quad._adaptive import (
    clenshaw_curtis_pair_data,
    nested_rule_estimate_values,
)
from jaxstro.quad._chebyshev import chebyshev_rule_data
from jaxstro.quad.rules import ClenshawCurtisRule


def _options(**overrides):
    options = dict(
        method=AdaptiveClenshawCurtis(initial_order=17),
        epsabs=1e-11,
        epsrel=1e-11,
        max_evaluations=4096,
        max_regions=128,
        error_norm=MaxNorm(),
    )
    options.update(overrides)
    return options


def test_clenshaw_curtis_pair_reuses_exact_nested_nodes_not_nested_weights() -> None:
    pair = clenshaw_curtis_pair_data(AdaptiveClenshawCurtis(initial_order=17))
    low = chebyshev_rule_data(ClenshawCurtisRule(9))
    assert jnp.array_equal(pair.nodes[pair.low_indices], low.nodes)
    assert jnp.array_equal(pair.low_weights, low.weights)
    assert jnp.any(pair.low_weights != pair.high_weights[pair.low_indices])


def test_clenshaw_curtis_pair_preserves_low_and_high_polynomial_exactness() -> None:
    pair = clenshaw_curtis_pair_data(AdaptiveClenshawCurtis(initial_order=17))
    for degree in range(9):
        expected = 0.0 if degree % 2 else 2.0 / (degree + 1)
        high = jnp.sum(pair.high_weights * pair.nodes**degree)
        low = jnp.sum(pair.low_weights * pair.nodes[pair.low_indices] ** degree)
        assert jnp.allclose(high, expected, atol=2e-13)
        assert jnp.allclose(low, expected, atol=2e-13)


def test_nested_estimator_reports_raw_difference_and_summation_floor() -> None:
    pair = clenshaw_curtis_pair_data(AdaptiveClenshawCurtis(initial_order=17))
    differing = nested_rule_estimate_values(pair.nodes**10, pair)
    constant = nested_rule_estimate_values(jnp.ones_like(pair.nodes), pair)
    assert differing.raw_error > 0.0
    assert differing.error >= differing.raw_error
    assert constant.roundoff_floor > 0.0
    assert constant.error >= constant.roundoff_floor
    assert constant.error >= constant.raw_error


def test_adaptive_clenshaw_curtis_integrates_smooth_and_vector_payloads() -> None:
    smooth = integrate(jnp.exp, Interval(-1.0, 1.0), **_options())
    payload = integrate(
        lambda x: jnp.stack((x**2, x**4), axis=-1),
        Interval(-1.0, 1.0),
        **_options(),
    )
    assert smooth.status == QuadStatus.CONVERGED
    assert jnp.allclose(smooth.value, jnp.e - 1.0 / jnp.e, rtol=2e-11)
    assert payload.status == QuadStatus.CONVERGED
    assert jnp.allclose(payload.value, jnp.asarray([2.0 / 3.0, 2.0 / 5.0]))
    assert payload.error.estimate.shape == (2,)
    assert payload.error.kind == ErrorKind.REFINEMENT_DIFFERENCE
    assert payload.work.evaluations == 17


def test_adaptive_clenshaw_curtis_localizes_breakpoints_without_low_repeats() -> None:
    result = integrate(
        lambda x: jnp.abs(x - 0.1),
        Interval(-1.0, 1.0, breakpoints=(0.1,)),
        **_options(),
    )
    assert result.status == QuadStatus.CONVERGED
    assert jnp.allclose(result.value, 1.01, rtol=2e-11)
    assert result.work.evaluations == 2 * 17
    assert result.work.active_regions == 2


def test_adaptive_clenshaw_curtis_exposes_endpoints_and_capacity_statuses() -> None:
    endpoint_nonfinite = integrate(
        lambda x: 1.0 / jnp.sqrt(1.0 - x**2),
        Interval(-1.0, 1.0),
        **_options(),
    )
    evaluation_limited = integrate(
        lambda x: jnp.abs(x - 0.123),
        Interval(-1.0, 1.0),
        **_options(max_evaluations=17, epsabs=0.0, epsrel=0.0),
    )
    region_limited = integrate(
        lambda x: jnp.abs(x - 0.123),
        Interval(-1.0, 1.0),
        **_options(max_regions=1, epsabs=0.0, epsrel=0.0),
    )
    assert endpoint_nonfinite.status == QuadStatus.NONFINITE_INTEGRAND
    assert evaluation_limited.status == QuadStatus.MAX_EVALUATIONS
    assert evaluation_limited.work.evaluations == 17
    assert region_limited.status == QuadStatus.MAX_REGIONS
    assert region_limited.work.evaluations == 17
