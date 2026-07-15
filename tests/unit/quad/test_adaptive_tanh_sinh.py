"""Adaptive tanh-sinh domain, tail, and work contracts."""

import os
import subprocess
import sys

import jax.numpy as jnp
from jax.scipy.special import gammaln

from jaxstro import quantity
from jaxstro.quad import (
    AdaptiveTanhSinh,
    ErrorKind,
    Infinite,
    Interval,
    LeftInfinite,
    MaxNorm,
    QuadStatus,
    RightInfinite,
    WeightedMeasure,
    integrate,
)
from jaxstro.quad._adaptive import (
    tanh_sinh_estimate_values,
    tanh_sinh_pair_data,
)


def _options(**overrides):
    options = dict(
        method=AdaptiveTanhSinh(initial_level=3),
        epsabs=1e-10,
        epsrel=1e-10,
        max_evaluations=20_000,
        max_regions=256,
        error_norm=MaxNorm(),
    )
    options.update(overrides)
    return options


def _singular_exact(exponent):
    return jnp.sqrt(jnp.pi) * jnp.exp(gammaln(1.0 - exponent) - gammaln(1.5 - exponent))


def test_tanh_sinh_pair_reuses_coarse_nodes_and_separates_error_evidence() -> None:
    pair = tanh_sinh_pair_data(AdaptiveTanhSinh(initial_level=3))
    assert jnp.array_equal(pair.nodes[pair.low_indices], pair.low_nodes)
    estimate = tanh_sinh_estimate_values(jnp.exp(pair.nodes), pair)
    assert estimate.discretization_error.shape == ()
    assert estimate.summation_error > 0.0
    assert estimate.tail_error > 0.0
    assert jnp.allclose(
        estimate.error,
        estimate.discretization_error + estimate.summation_error + estimate.tail_error,
    )


def test_tanh_sinh_pair_ratchets_active_counts_and_terminal_transition() -> None:
    expected = (
        (3, 101, 50, False),
        (4, 203, 101, False),
        (5, 407, 203, True),
    )
    for initial_level, active_count, terminal_index, exhausted in expected:
        pair = tanh_sinh_pair_data(
            AdaptiveTanhSinh(initial_level=initial_level), dtype=jnp.float64
        )
        assert pair.nodes.shape == (active_count,)
        assert pair.terminal_index == terminal_index
        assert bool(pair.dtype_exhausted) is exhausted
        assert not jnp.any(pair.outer_shell)


def test_adaptive_tanh_sinh_integrates_smooth_breakpoint_and_orientation() -> None:
    smooth = integrate(jnp.exp, Interval(-1.0, 1.0), **_options())
    localized = integrate(
        lambda x: jnp.abs(x - 0.1),
        Interval(-1.0, 1.0, breakpoints=(0.1,)),
        **_options(),
    )
    reversed_result = integrate(jnp.exp, Interval(1.0, -1.0), **_options())
    pair = tanh_sinh_pair_data(AdaptiveTanhSinh(initial_level=3))
    assert smooth.status == QuadStatus.CONVERGED
    assert jnp.allclose(smooth.value, jnp.e - 1.0 / jnp.e, rtol=2e-10)
    assert localized.status == QuadStatus.CONVERGED
    assert jnp.allclose(localized.value, 1.01, rtol=2e-10)
    assert localized.work.evaluations == 2 * pair.nodes.shape[0]
    assert jnp.allclose(reversed_result.value, -smooth.value)
    assert smooth.error.kind == ErrorKind.REFINEMENT_DIFFERENCE


def test_adaptive_tanh_sinh_handles_declared_endpoint_singularity_envelopes() -> None:
    cases = (
        (0.5, 1e-9, 1e-8),
        (0.9, 1e-4, 3e-2),
    )
    for exponent, tolerance, relative_envelope in cases:
        result = integrate(
            lambda x, exponent=exponent: (1.0 - x**2) ** (-exponent),
            Interval(-1.0, 1.0),
            **_options(epsabs=tolerance, epsrel=tolerance),
        )
        observed = jnp.abs(result.value - _singular_exact(exponent))
        assert result.status == QuadStatus.ROUNDOFF_LIMITED
        assert result.error.norm > result.tolerance
        assert observed <= relative_envelope * _singular_exact(exponent)

    hardest = integrate(
        lambda x: (1.0 - x**2) ** (-0.99),
        Interval(-1.0, 1.0),
        **_options(epsabs=1e-3, epsrel=1e-3),
    )
    observed = jnp.abs(hardest.value - _singular_exact(0.99))
    assert hardest.status == QuadStatus.ROUNDOFF_LIMITED
    assert hardest.error.norm > hardest.tolerance
    assert observed <= 0.7 * _singular_exact(0.99)


def test_clipped_region_still_converges_when_tolerance_is_already_met() -> None:
    result = integrate(
        lambda x: jnp.ones_like(x),
        Interval(0.0, 1.0, breakpoints=(1.0 - 2.0**-10,)),
        **_options(epsabs=1.0, epsrel=1.0),
    )
    assert result.status == QuadStatus.CONVERGED
    assert result.error.norm <= result.tolerance
    assert jnp.allclose(result.value, 1.0)


def test_adaptive_tanh_sinh_integrates_every_improper_domain_family() -> None:
    right = integrate(lambda x: jnp.exp(-x), RightInfinite(0.0), **_options())
    left = integrate(lambda x: jnp.exp(x), LeftInfinite(0.0), **_options())
    full = integrate(lambda x: jnp.exp(-(x**2)), Infinite(), **_options())
    assert right.status == QuadStatus.CONVERGED
    assert left.status == QuadStatus.CONVERGED
    assert (full.status == QuadStatus.CONVERGED) | (
        (full.status == QuadStatus.ROUNDOFF_LIMITED)
        & (full.error.norm > full.tolerance)
    )
    assert jnp.allclose(right.value, 1.0, rtol=3e-9)
    assert jnp.allclose(left.value, 1.0, rtol=3e-9)
    assert jnp.allclose(full.value, jnp.sqrt(jnp.pi), rtol=3e-9)


def test_adaptive_tanh_sinh_supports_weighted_vector_payloads() -> None:
    measure = WeightedMeasure(
        lambda x, args: args * (1.0 + x),
        density_unit=quantity.dimensionless,
    )
    result = integrate(
        lambda x, _args: jnp.stack((x, x**2), axis=-1),
        Interval(0.0, 1.0),
        args=2.0,
        measure=measure,
        **_options(),
    )
    assert result.status == QuadStatus.CONVERGED
    assert jnp.allclose(result.value, jnp.asarray([5.0 / 3.0, 7.0 / 6.0]))
    assert result.error.estimate.shape == (2,)


def test_adaptive_tanh_sinh_rejects_nonfinite_active_contributions() -> None:
    result = integrate(
        lambda x: jnp.where(x > 0.9, jnp.nan, 1.0),
        Interval(-1.0, 1.0),
        **_options(),
    )
    assert result.status == QuadStatus.NONFINITE_INTEGRAND


def test_adaptive_tanh_sinh_reports_exact_active_work_and_capacities() -> None:
    pair = tanh_sinh_pair_data(AdaptiveTanhSinh(initial_level=3))
    node_cost = pair.nodes.shape[0]
    evaluation_limited = integrate(
        lambda x: jnp.abs(x - 0.123),
        Interval(-1.0, 1.0),
        **_options(
            epsabs=0.0,
            epsrel=0.0,
            max_evaluations=node_cost,
        ),
    )
    region_limited = integrate(
        lambda x: jnp.abs(x - 0.123),
        Interval(-1.0, 1.0),
        **_options(epsabs=0.0, epsrel=0.0, max_regions=1),
    )
    assert evaluation_limited.status == QuadStatus.MAX_EVALUATIONS
    assert evaluation_limited.work.evaluations == node_cost
    assert region_limited.status == QuadStatus.MAX_REGIONS
    assert region_limited.work.evaluations == node_cost


def test_float32_endpoint_singularities_do_not_false_converge_in_subprocess() -> None:
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    program = """
import jax.numpy as jnp
from jaxstro.quad import (
    AdaptiveTanhSinh,
    Interval,
    MaxNorm,
    QuadStatus,
    RightInfinite,
    integrate,
)
from jaxstro.quad._adaptive import tanh_sinh_pair_data

method = AdaptiveTanhSinh(initial_level=3)
node_cost = tanh_sinh_pair_data(method, dtype=jnp.float32).nodes.shape[0]
for domain, fun, expected in (
    (Interval(-1.0, 1.0), jnp.exp, jnp.e - 1.0 / jnp.e),
    (RightInfinite(0.0), lambda x: jnp.exp(-x), 1.0),
):
    result = integrate(
        fun,
        domain,
        method=method,
        epsabs=1e-4,
        epsrel=1e-4,
        max_evaluations=10_000,
        max_regions=128,
        error_norm=MaxNorm(),
    )
    assert result.status == QuadStatus.CONVERGED
    assert jnp.allclose(result.value, expected, rtol=2e-5, atol=2e-5)
    assert result.work.evaluations == node_cost

for exponent in (0.5, 0.9, 0.99):
    result = integrate(
        lambda x, exponent=exponent: (1.0 - x**2) ** (-exponent),
        Interval(-1.0, 1.0),
        method=method,
        epsabs=1e-6,
        epsrel=1e-6,
        max_evaluations=10_000,
        max_regions=128,
        error_norm=MaxNorm(),
    )
    assert result.status == QuadStatus.ROUNDOFF_LIMITED
    assert result.error.norm > result.tolerance
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
