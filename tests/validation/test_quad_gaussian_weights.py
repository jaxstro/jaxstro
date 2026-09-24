"""Gaussian rules keep relative weight accuracy in the tails.

Eigenvector weights are accurate only to about eps absolutely; the engine's
Newton-refined nodes and Christoffel weights must match a 60-digit reference
relative to each weight and integrate tail-dominated functions.
"""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest

from jaxstro import quad
from jaxstro.quad._recurrence import gaussian_rule_data

mp = pytest.importorskip("mpmath")

EPS = float(np.finfo(np.float64).eps)
# Approved 2026-09-24 from measured maxima of 15 eps (nodes) and 9.8 n eps
# (weights) for n <= 128, and 8.4e-15 relative for tail integrals, n <= 256.
NODE_TOL_EPS = 64.0
WEIGHT_TOL_EPS_PER_NODE = 16.0
TAIL_RTOL = 1e-13


def _rule(measure, n):
    data = gaussian_rule_data(quad.GaussianRule(n), measure)
    return np.asarray(data.nodes), np.asarray(data.weights)


def _reference(x0, a, b, mass):
    """Newton-converged nodes and Christoffel weights at 60 digits."""
    n = len(a)
    nodes, weights = [], []
    with mp.workdps(60):
        for start in x0:
            x = mp.mpf(float(start))
            for _ in range(8):
                p_prev, p = mp.mpf(0), 1 / mp.sqrt(mass)
                dp_prev, dp = mp.mpf(0), mp.mpf(0)
                for k in range(n):
                    b_next = b[k] if k < n - 1 else mp.mpf(1)
                    b_prev = b[k - 1] if k > 0 else mp.mpf(0)
                    p_new = ((x - a[k]) * p - b_prev * p_prev) / b_next
                    dp_new = (p + (x - a[k]) * dp - b_prev * dp_prev) / b_next
                    p_prev, p, dp_prev, dp = p, p_new, dp, dp_new
                x -= p / dp
            p_prev, p = mp.mpf(0), 1 / mp.sqrt(mass)
            total = p * p
            for k in range(n - 1):
                b_prev = b[k - 1] if k > 0 else mp.mpf(0)
                p_prev, p = p, ((x - a[k]) * p - b_prev * p_prev) / b[k]
                total += p * p
            nodes.append(x)
            weights.append(1 / total)
    return nodes, weights


FAMILIES = {
    "standard-normal": (
        quad.StandardNormalMeasure(),
        lambda n: ([0] * n, [mp.sqrt(k) for k in range(1, n)], mp.mpf(1)),
    ),
    "legendre": (
        quad.LebesgueMeasure(),
        lambda n: ([0] * n, [k / mp.sqrt(4 * k * k - 1) for k in range(1, n)], 2),
    ),
    "laguerre": (
        quad.LaguerreMeasure(),
        lambda n: ([2 * k + 1 for k in range(n)], [mp.mpf(k) for k in range(1, n)], 1),
    ),
}


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("n", [5, 20, 64, 128])
def test_rule_matches_a_60_digit_reference(family, n) -> None:
    measure, coefficients = FAMILIES[family]
    nodes, weights = _rule(measure, n)
    ref_nodes, ref_weights = _reference(nodes, *coefficients(n))

    for x, w, rx, rw in zip(nodes, weights, ref_nodes, ref_weights):
        node_error = abs(mp.mpf(float(x)) - rx) / max(abs(rx), 1)
        assert node_error <= NODE_TOL_EPS * EPS, (family, n, float(rx))
        if rw > mp.mpf("1e-300"):
            weight_error = abs(mp.mpf(float(w)) - rw) / rw
            assert weight_error <= WEIGHT_TOL_EPS_PER_NODE * n * EPS, (
                family,
                n,
                float(rx),
                float(weight_error),
            )


@pytest.mark.parametrize("n", [3, 17, 100])
def test_chebyshev_first_kind_matches_its_closed_form(n) -> None:
    nodes, weights = _rule(quad.JacobiMeasure(-0.5, -0.5), n)
    expected = np.cos((2 * np.arange(n, 0, -1) - 1) * np.pi / (2 * n))
    np.testing.assert_allclose(nodes, expected, rtol=0, atol=NODE_TOL_EPS * EPS)
    np.testing.assert_allclose(
        weights, np.full(n, np.pi / n), rtol=WEIGHT_TOL_EPS_PER_NODE * n * EPS
    )


TAIL_CASES = [
    # (label, measure, integrand, truth, orders); the orders are those at
    # which the rule's truncation error is below TAIL_RTOL.
    (
        "E[exp(6 g)]",
        quad.StandardNormalMeasure(),
        lambda x: np.exp(6 * x),
        math.exp(18),
        (64, 128, 256),
    ),
    (
        "E[exp(10 g)]",
        quad.StandardNormalMeasure(),
        lambda x: np.exp(10 * x),
        math.exp(50),
        (128, 256),
    ),
    (
        "E[g^80]",
        quad.StandardNormalMeasure(),
        lambda x: x**80,
        float(math.prod(range(1, 80, 2))),
        (64, 128, 256),
    ),
    (
        "int e^{x/2} e^{-x}",
        quad.LaguerreMeasure(),
        lambda x: np.exp(x / 2),
        2.0,
        (64, 128, 256),
    ),
    (
        "int x^30 e^{-x}",
        quad.LaguerreMeasure(),
        lambda x: x**30,
        float(math.factorial(30)),
        (64, 128, 256),
    ),
]


@pytest.mark.parametrize(
    "label,measure,f,truth,n",
    [(c[0], c[1], c[2], c[3], n) for c in TAIL_CASES for n in c[4]],
    ids=[f"{c[0]}-n{n}" for c in TAIL_CASES for n in c[4]],
)
def test_tail_dominated_integrals(label, measure, f, truth, n) -> None:
    del label
    nodes, weights = _rule(measure, n)
    value = float(np.sum(weights * f(nodes)))
    assert abs(value - truth) <= TAIL_RTOL * abs(truth)


@pytest.mark.parametrize("n", [3, 17, 100])
def test_chebyshev_second_kind_matches_its_closed_form(n) -> None:
    nodes, weights = _rule(quad.JacobiMeasure(0.5, 0.5), n)
    theta = np.arange(n, 0, -1) * np.pi / (n + 1)
    np.testing.assert_allclose(nodes, np.cos(theta), rtol=0, atol=NODE_TOL_EPS * EPS)
    np.testing.assert_allclose(
        weights,
        np.pi / (n + 1) * np.sin(theta) ** 2,
        rtol=WEIGHT_TOL_EPS_PER_NODE * n * EPS,
    )


@pytest.mark.parametrize("alpha", [-0.3, -0.7])
def test_jacobi_rules_with_alpha_plus_beta_minus_one_are_exact(alpha) -> None:
    # alpha + beta = -1 made the k = 1 recurrence coefficient 0/0 before
    # 2026-09-24. Check degree-(2n-1) exactness against Beta-function moments.
    beta = -1.0 - alpha
    n = 6
    nodes, weights = _rule(quad.JacobiMeasure(alpha, beta), n)
    assert np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights))
    # Measure (1-x)^alpha (1+x)^beta on [-1, 1]; with x = 2t - 1 the moment
    # of (1+x)^m is 2^(alpha+beta+1+m) B(beta+m+1, alpha+1).
    for m in range(2 * n):
        truth = 2.0 ** (alpha + beta + 1 + m) * math.exp(
            math.lgamma(beta + m + 1)
            + math.lgamma(alpha + 1)
            - math.lgamma(alpha + beta + m + 2)
        )
        value = float(np.sum(weights * (1.0 + nodes) ** m))
        assert abs(value - truth) <= TAIL_RTOL * truth, m


def test_gauss_hermite_nodes_is_the_standard_normal_rule() -> None:
    nodes, weights = quad.gauss_hermite_nodes(40)
    ref_nodes, ref_weights = _rule(quad.StandardNormalMeasure(), 40)
    assert np.array_equal(np.asarray(nodes), ref_nodes)
    assert np.array_equal(np.asarray(weights), ref_weights)
    assert jax.numpy.asarray(nodes).dtype == np.float64
