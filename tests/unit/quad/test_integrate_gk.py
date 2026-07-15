"""Public adaptive Gauss-Kronrod integration contracts."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro import quantity
from jaxstro.quad import (
    ErrorKind,
    GaussKronrod,
    Infinite,
    Interval,
    MaxNorm,
    QuadStatus,
    Romberg,
    WeightedMeasure,
    integrate,
)


def _options(pair=21, **overrides):
    options = dict(
        method=GaussKronrod(pair=pair),
        epsabs=1e-11,
        epsrel=1e-11,
        max_evaluations=4096,
        max_regions=128,
        error_norm=MaxNorm(),
    )
    options.update(overrides)
    return options


@pytest.mark.parametrize("pair", [15, 21, 31, 41, 51, 61])
def test_integrate_supports_every_gauss_kronrod_pair(pair) -> None:
    result = integrate(
        lambda x: x**4,
        Interval(-1.0, 1.0),
        **_options(pair),
    )
    assert result.status == QuadStatus.CONVERGED
    assert jnp.allclose(result.value, 2.0 / 5.0, rtol=2e-13, atol=2e-13)
    assert result.error.kind == ErrorKind.EMBEDDED_RULE
    assert jnp.isnan(result.error.confidence_level)
    assert result.error.norm <= result.tolerance
    assert result.work.evaluations == pair
    assert result.work.refinements == 0
    assert result.work.active_regions == 1
    assert result.work.levels == 0
    assert result.work.replicates == 0


def test_integrate_breakpoints_orientation_weight_and_payload() -> None:
    measure = WeightedMeasure(
        lambda x, args: args * (1.0 + x),
        density_unit=quantity.dimensionless,
    )

    def fun(x, args):
        return jnp.stack((x + 1j * args, x**2), axis=-1)

    forward = integrate(
        fun,
        Interval(0.0, 1.0, breakpoints=(0.25, 0.75)),
        args=2.0,
        measure=measure,
        **_options(epsabs=1e-10),
    )
    reverse = integrate(
        fun,
        Interval(1.0, 0.0, breakpoints=(0.75, 0.25)),
        args=2.0,
        measure=measure,
        **_options(epsabs=1e-10),
    )
    assert forward.status == QuadStatus.CONVERGED
    assert forward.value.shape == (2,)
    assert jnp.allclose(forward.value, jnp.asarray([5.0 / 3.0 + 6.0j, 7.0 / 6.0]))
    assert jnp.allclose(reverse.value, -forward.value)
    assert forward.error.estimate.shape == forward.value.shape
    assert jnp.issubdtype(forward.error.estimate.dtype, jnp.floating)
    assert jnp.all(forward.error.estimate >= 0.0)
    assert forward.work.evaluations == 3 * 21


@pytest.mark.parametrize(
    "fun",
    [
        lambda x: jnp.ones_like(x, dtype=jnp.int32),
        lambda x: jnp.ones_like(x, dtype=jnp.bool_),
        lambda x: jnp.ones_like(x, dtype=jnp.float32),
        lambda x: jnp.ones_like(x, dtype=jnp.complex64) * (1.0 + 2.0j),
    ],
)
def test_integrate_normalizes_payload_dtype_for_both_cond_branches(fun) -> None:
    nonzero = integrate(fun, Interval(0.0, 1.0), **_options())
    zero = integrate(fun, Interval(1.0, 1.0), **_options())
    assert nonzero.status == QuadStatus.CONVERGED
    assert zero.status == QuadStatus.CONVERGED
    assert nonzero.value.dtype == zero.value.dtype
    assert nonzero.error.estimate.dtype == zero.error.estimate.dtype
    assert jnp.issubdtype(nonzero.value.dtype, jnp.inexact)
    assert jnp.issubdtype(nonzero.error.estimate.dtype, jnp.floating)
    assert jnp.all(zero.value == 0.0)


def test_integrate_returns_both_capacity_statuses_before_partial_work() -> None:
    def fun(x: jax.Array) -> jax.Array:
        return jnp.abs(x - 0.123)

    evaluation_limited = integrate(
        fun,
        Interval(-1.0, 1.0),
        **_options(max_evaluations=21, max_regions=8, epsabs=0.0, epsrel=0.0),
    )
    region_limited = integrate(
        fun,
        Interval(-1.0, 1.0),
        **_options(max_evaluations=1000, max_regions=1, epsabs=0.0, epsrel=0.0),
    )
    assert evaluation_limited.status == QuadStatus.MAX_EVALUATIONS
    assert evaluation_limited.work.evaluations == 21
    assert region_limited.status == QuadStatus.MAX_REGIONS
    assert region_limited.work.evaluations == 21


def test_integrate_rejects_structural_capacities_before_tracing_user_code() -> None:
    def must_not_trace(_x):
        raise AssertionError("structurally invalid calls must not trace the integrand")

    with pytest.raises(ValueError, match="initial node cost"):
        integrate(
            must_not_trace,
            Interval(-1.0, 1.0),
            **_options(max_evaluations=20),
        )
    with pytest.raises(ValueError, match="initial partition"):
        integrate(
            must_not_trace,
            Interval(-1.0, 1.0, breakpoints=(-0.5, 0.5)),
            **_options(max_regions=2),
        )


def test_integrate_exposes_deterministic_roundoff_statuses() -> None:
    stagnated = integrate(
        lambda x: jnp.abs(x - 0.123),
        Interval(-1.0, 1.0),
        **_options(
            epsabs=0.0,
            epsrel=0.0,
            max_evaluations=20_000,
            max_regions=256,
        ),
    )
    left = jnp.asarray(1.0, dtype=jnp.float64)
    right = jnp.nextafter(left, jnp.asarray(2.0, dtype=jnp.float64))
    adjacent = integrate(
        lambda x: jnp.where((x >= left) & (x <= right), 1e200, 0.0),
        Interval(0.0, 2.0, breakpoints=(left, right)),
        **_options(
            epsabs=0.0,
            epsrel=0.0,
            max_evaluations=20_000,
            max_regions=256,
        ),
    )
    assert stagnated.status == QuadStatus.ROUNDOFF_LIMITED
    assert adjacent.status == QuadStatus.ROUNDOFF_LIMITED
    assert stagnated.work.refinements > adjacent.work.refinements


def test_integrate_invalid_nonfinite_and_zero_width_precedence() -> None:
    invalid = integrate(
        lambda x: jnp.ones_like(x),
        Interval(0.0, 1.0, breakpoints=(2.0,)),
        **_options(),
    )
    nonfinite = integrate(
        lambda x: jnp.where(x > 0.0, jnp.nan, x),
        Interval(-1.0, 1.0),
        **_options(),
    )
    zero = integrate(
        lambda x: jnp.full((x.shape[0], 2), jnp.nan),
        Interval(2.0, 2.0),
        **_options(),
    )
    invalid_zero = integrate(
        lambda x: jnp.ones_like(x),
        Interval(jnp.inf, jnp.inf),
        **_options(),
    )
    assert invalid.status == QuadStatus.INVALID_INPUT
    assert nonfinite.status == QuadStatus.NONFINITE_INTEGRAND
    assert zero.status == QuadStatus.CONVERGED
    assert jnp.array_equal(zero.value, jnp.zeros((2,)))
    assert jnp.array_equal(zero.error.estimate, jnp.zeros((2,)))
    assert zero.work.evaluations == 0
    assert zero.work.active_regions == 0
    assert invalid_zero.status == QuadStatus.INVALID_INPUT


def test_integrate_rejects_unsupported_structure_and_gradient_policy() -> None:
    with pytest.raises(TypeError, match="finite Interval"):
        integrate(lambda x: x, Infinite(), **_options())
    with pytest.raises(TypeError, match="not implemented in Phase A2"):
        integrate(
            lambda x: x,
            Interval(-1.0, 1.0),
            **_options(method=Romberg()),
        )
    with pytest.raises(ValueError, match="Phase A3 replay"):
        integrate(
            lambda x: x,
            Interval(-1.0, 1.0),
            **_options(gradient="replay"),
        )


@pytest.mark.parametrize(
    ("epsabs", "epsrel", "error"),
    [
        (jnp.asarray([1e-8]), 1e-8, ValueError),
        (1e-8 + 0j, 1e-8, TypeError),
    ],
)
def test_integrate_rejects_nonscalar_or_complex_tolerances(
    epsabs, epsrel, error
) -> None:
    with pytest.raises(error):
        integrate(
            lambda x: x,
            Interval(-1.0, 1.0),
            **_options(epsabs=epsabs, epsrel=epsrel),
        )
