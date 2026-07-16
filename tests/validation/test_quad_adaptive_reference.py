"""Independent analytic acceptance envelope for adaptive quadrature.

Thresholds in this module are declarations, not values learned from a run.  A
passing status is never accepted without either an analytic observed-error
envelope or an explicitly enumerated nonconverged status.
"""

import json
from pathlib import Path

import jax.numpy as jnp
import pytest
from scripts.build_quad_adaptive_validation import build_evidence

from jaxstro.quad import (
    AdaptiveClenshawCurtis,
    AdaptiveTanhSinh,
    GaussKronrod,
    Infinite,
    Interval,
    MaxNorm,
    QuadStatus,
    RightInfinite,
    Romberg,
    RombergTanhSinh,
    integrate,
)

METHODS = (
    GaussKronrod(pair=21),
    AdaptiveClenshawCurtis(initial_order=17),
    AdaptiveTanhSinh(initial_level=3),
    Romberg(initial_level=1),
    RombergTanhSinh(initial_level=1),
)
H_ADAPTIVE = METHODS[:3]
TANH_SINH = (METHODS[2], METHODS[4])
TOLERANCE_SWEEP = (1e-4, 1e-7, 1e-10)
EVIDENCE_PATH = Path("docs/validation/quad-adaptive-envelope.json")


def _options(method, tolerance, **overrides):
    options = dict(
        method=method,
        epsabs=tolerance,
        epsrel=tolerance,
        max_evaluations=20_000,
        max_regions=256,
        error_norm=MaxNorm(),
    )
    options.update(overrides)
    return options


def test_tolerance_sweep_evidence_matches_fresh_owner_output() -> None:
    recorded = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert recorded == build_evidence()
    assert len(recorded["records"]) == len(METHODS) * len(TOLERANCE_SWEEP)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("tolerance", TOLERANCE_SWEEP)
def test_smooth_exponential_tolerance_sweep(method, tolerance) -> None:
    expected = jnp.e - 1.0 / jnp.e
    result = integrate(jnp.exp, Interval(-1.0, 1.0), **_options(method, tolerance))
    observed = jnp.abs(result.value - expected)
    assert result.status in (QuadStatus.CONVERGED, QuadStatus.ROUNDOFF_LIMITED)
    assert observed <= max(5e-12, 5.0 * tolerance * expected)
    if result.status == QuadStatus.CONVERGED:
        assert result.error.norm <= result.tolerance
    else:
        assert result.error.norm > result.tolerance
    assert result.work.evaluations > 0


@pytest.mark.parametrize("method", METHODS)
def test_polynomial_complex_and_vector_analytic_oracles(method) -> None:
    result = integrate(
        lambda x: jnp.stack((x**4, jnp.exp(1j * x)), axis=-1),
        Interval(-1.0, 1.0),
        **_options(method, 1e-9),
    )
    expected = jnp.asarray([2.0 / 5.0, 2.0 * jnp.sin(1.0)])
    assert result.status in (QuadStatus.CONVERGED, QuadStatus.ROUNDOFF_LIMITED)
    assert jnp.allclose(result.value, expected, rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize("method", H_ADAPTIVE)
def test_declared_breakpoint_localizes_a_nonsmooth_integrand(method) -> None:
    result = integrate(
        lambda x: jnp.abs(x - 0.1),
        Interval(-1.0, 1.0, breakpoints=(0.1,)),
        **_options(method, 1e-9),
    )
    assert result.status == QuadStatus.CONVERGED
    assert jnp.allclose(result.value, 1.01, rtol=2e-9)


@pytest.mark.parametrize("method", TANH_SINH)
def test_improper_tail_analytic_oracles(method) -> None:
    right = integrate(
        lambda x: jnp.exp(-x),
        RightInfinite(0.0),
        **_options(method, 1e-8),
    )
    full = integrate(
        lambda x: jnp.exp(-(x**2)),
        Infinite(),
        **_options(method, 1e-8),
    )
    for result, expected in ((right, 1.0), (full, jnp.sqrt(jnp.pi))):
        assert result.status in (
            QuadStatus.CONVERGED,
            QuadStatus.ROUNDOFF_LIMITED,
        )
        assert jnp.allclose(result.value, expected, rtol=2e-7, atol=2e-9)


def test_endpoint_singularity_has_an_honest_representability_exit() -> None:
    for method in TANH_SINH:
        result = integrate(
            lambda x: (1.0 - x**2) ** (-0.5),
            Interval(-1.0, 1.0),
            **_options(method, 1e-8),
        )
        assert result.status == QuadStatus.ROUNDOFF_LIMITED
        assert result.error.norm > result.tolerance
        assert jnp.abs(result.value - jnp.pi) <= 2e-7


def test_missed_narrow_feature_documents_estimator_false_convergence() -> None:
    width = 1e-4
    expected = jnp.sqrt(jnp.pi) * width
    result = integrate(
        lambda x: jnp.exp(-(((x - 0.123) / width) ** 2)),
        Interval(-1.0, 1.0),
        **_options(GaussKronrod(pair=21), 1e-10),
    )
    assert result.status == QuadStatus.CONVERGED
    assert result.error.norm <= result.tolerance
    assert jnp.abs(result.value - expected) > 100.0 * result.tolerance


def test_failure_envelope_nonfinite_budget_and_endpoint_exposure() -> None:
    nonfinite = integrate(
        lambda x: jnp.where(x > 0.0, jnp.nan, x),
        Interval(-1.0, 1.0),
        **_options(GaussKronrod(), 1e-8),
    )
    exhausted = integrate(
        lambda x: jnp.abs(x - 0.123),
        Interval(-1.0, 1.0),
        **_options(GaussKronrod(), 0.0, max_evaluations=21),
    )
    endpoint = integrate(
        lambda x: 1.0 / jnp.sqrt(1.0 - x**2),
        Interval(-1.0, 1.0),
        **_options(AdaptiveClenshawCurtis(), 1e-8),
    )
    assert nonfinite.status == QuadStatus.NONFINITE_INTEGRAND
    assert exhausted.status == QuadStatus.MAX_EVALUATIONS
    assert endpoint.status == QuadStatus.NONFINITE_INTEGRAND
