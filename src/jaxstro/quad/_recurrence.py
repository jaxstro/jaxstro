"""Shared classical recurrence engine for Gaussian quadrature rules."""

import math
from typing import Any

import jax
import jax.numpy as jnp

from .measures import (
    JacobiMeasure,
    LaguerreMeasure,
    LebesgueMeasure,
    PhysicistsHermiteMeasure,
    StandardNormalMeasure,
)
from .rules import FixedRuleData, GaussianRule


def _orthonormal_sweep(x, diagonal, off_diagonal, mass: float):
    """Evaluate the orthonormal three-term recurrence at every ``x``.

    Returns ``(q, dq, total, log_scale)``: ``q`` and ``dq`` are the value and
    derivative of an unnormalized degree-``n`` polynomial with the rule's nodes
    as roots (the recurrence closed with ``b_n = 1``; Newton needs only
    ``q / dq``), and ``total * exp(2 * log_scale)`` is
    ``sum_{k<n} p_k(x) ** 2`` for the orthonormal ``p_k``. All values are
    rescaled whenever they would pass the dtype's fourth-root range, so tail
    nodes where ``p_k`` reaches ``1e85`` and beyond neither overflow nor lose
    relative accuracy.
    """
    order = diagonal.shape[0]
    dtype = x.dtype
    threshold = jnp.asarray(jnp.finfo(dtype).max ** 0.25, dtype=dtype)
    log_threshold = jnp.log(threshold)
    closing = jnp.ones((1,), dtype=dtype)
    next_b = jnp.concatenate((off_diagonal, closing))
    previous_b = jnp.concatenate((jnp.zeros((1,), dtype=dtype), off_diagonal))

    def step(carry, k):
        p_prev, p, dp_prev, dp, total, log_scale = carry
        shift = x - diagonal[k]
        p_next = (shift * p - previous_b[k] * p_prev) / next_b[k]
        dp_next = (p + shift * dp - previous_b[k] * dp_prev) / next_b[k]
        total = jnp.where(k < order - 1, total + p_next**2, total)
        large = jnp.abs(p_next) > threshold
        factor = jnp.where(large, 1.0 / threshold, 1.0)
        log_scale = log_scale + jnp.where(large, log_threshold, 0.0)
        carry = (
            p * factor,
            p_next * factor,
            dp * factor,
            dp_next * factor,
            total * factor**2,
            log_scale,
        )
        return carry, None

    p0 = jnp.full_like(x, 1.0 / math.sqrt(mass))
    zeros = jnp.zeros_like(x)
    initial = (zeros, p0, zeros, zeros, p0**2, zeros)
    (_, q, _, dq, total, log_scale), _ = jax.lax.scan(step, initial, jnp.arange(order))
    return q, dq, total, log_scale


def _golub_welsch(diagonal, off_diagonal, mass: float) -> FixedRuleData:
    """Gaussian rule from the Jacobi matrix, with relatively accurate weights.

    The eigenvalues of the symmetric tridiagonal Jacobi matrix give the nodes
    (Golub & Welsch 1969). The eigenvector weights ``mass * v_0**2`` are
    accurate only to about ``eps`` in absolute terms, so tail weights far below
    ``eps`` can be wrong by many orders of magnitude (standard-normal rule,
    ``n = 256``: ``1.3e-60`` instead of ``1.0e-171``). Each node therefore
    takes one Newton step on the recurrence polynomial, and each weight is the
    Christoffel function ``1 / sum_{k<n} p_k(x_i) ** 2`` of the orthonormal
    polynomials (Gautschi 2004, Orthogonal Polynomials, Thm. 1.46), which keeps
    its relative accuracy in the tails.
    """
    matrix = jnp.diag(diagonal)
    if diagonal.shape[0] > 1:
        matrix = matrix + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
    nodes = jnp.linalg.eigvalsh(matrix)
    q, dq, _, _ = _orthonormal_sweep(nodes, diagonal, off_diagonal, mass)
    newton = nodes - q / dq
    nodes = jnp.where(jnp.isfinite(newton), newton, nodes)
    _, _, total, log_scale = _orthonormal_sweep(nodes, diagonal, off_diagonal, mass)
    weights = jnp.exp(-jnp.log(total) - 2.0 * log_scale)
    return FixedRuleData(
        nodes=nodes,
        weights=weights,
        degree=2 * diagonal.shape[0] - 1,
        nested=False,
    )


def _legendre(order: int):
    dtype = jnp.asarray(0.0).dtype
    diagonal = jnp.zeros((order,), dtype=dtype)
    index = jnp.arange(1, order, dtype=dtype)
    off_diagonal = index / jnp.sqrt(4.0 * index**2 - 1.0)
    return diagonal, off_diagonal, 2.0


def _jacobi(order: int, alpha: float, beta: float):
    dtype = jnp.asarray(float(alpha) + float(beta)).dtype
    total = alpha + beta
    diagonal0 = jnp.asarray((beta - alpha) / (total + 2.0), dtype=dtype)
    index = jnp.arange(1, order, dtype=dtype)
    twice = 2.0 * index + total
    diagonal_rest = (beta**2 - alpha**2) / (twice * (twice + 2.0))
    diagonal = jnp.concatenate((diagonal0[None], diagonal_rest))
    # For k >= 2 every factor is positive. At k = 1 the general formula is
    # 0/0 when alpha + beta = -1 (Gauss-Chebyshev among them), so b_1 uses the
    # form with the common factor (1 + alpha + beta) cancelled (Gautschi 2004,
    # Table 1.1).
    first = jnp.sqrt(
        4.0 * (1.0 + alpha) * (1.0 + beta) / ((2.0 + total) ** 2 * (3.0 + total))
    )
    later = index[1:]
    twice_later = twice[1:]
    rest = (2.0 / twice_later) * jnp.sqrt(
        later
        * (later + alpha)
        * (later + beta)
        * (later + total)
        / ((twice_later - 1.0) * (twice_later + 1.0))
    )
    off_diagonal = jnp.concatenate((jnp.atleast_1d(first), rest))[: order - 1]
    mass = (
        2.0 ** (total + 1.0)
        * math.gamma(alpha + 1.0)
        * math.gamma(beta + 1.0)
        / math.gamma(total + 2.0)
    )
    return diagonal, off_diagonal, mass


def _laguerre(order: int, alpha: float):
    dtype = jnp.asarray(float(alpha)).dtype
    index = jnp.arange(order, dtype=dtype)
    diagonal = 2.0 * index + alpha + 1.0
    positive_index = jnp.arange(1, order, dtype=dtype)
    off_diagonal = jnp.sqrt(positive_index * (positive_index + alpha))
    return diagonal, off_diagonal, math.gamma(alpha + 1.0)


def _physicists_hermite(order: int):
    dtype = jnp.asarray(0.0).dtype
    diagonal = jnp.zeros((order,), dtype=dtype)
    index = jnp.arange(1, order, dtype=dtype)
    return diagonal, jnp.sqrt(0.5 * index), math.sqrt(math.pi)


def _standard_normal(order: int):
    dtype = jnp.asarray(0.0).dtype
    diagonal = jnp.zeros((order,), dtype=dtype)
    index = jnp.arange(1, order, dtype=dtype)
    return diagonal, jnp.sqrt(index), 1.0


def gaussian_rule_data(rule: GaussianRule, measure: Any) -> FixedRuleData:
    """Construct a Gaussian rule matched to a declared classical measure."""
    if isinstance(measure, LebesgueMeasure):
        diagonal, off_diagonal, mass = _legendre(rule.order)
    elif isinstance(measure, JacobiMeasure):
        diagonal, off_diagonal, mass = _jacobi(rule.order, measure.alpha, measure.beta)
        if measure.normalized:
            mass = 1.0
    elif isinstance(measure, LaguerreMeasure):
        diagonal, off_diagonal, mass = _laguerre(rule.order, measure.alpha)
        if measure.normalized:
            mass = 1.0
    elif isinstance(measure, PhysicistsHermiteMeasure):
        diagonal, off_diagonal, mass = _physicists_hermite(rule.order)
        if measure.normalized:
            mass = 1.0
    elif isinstance(measure, StandardNormalMeasure):
        diagonal, off_diagonal, mass = _standard_normal(rule.order)
    else:
        raise TypeError("GaussianRule requires a supported classical measure")
    return _golub_welsch(diagonal, off_diagonal, mass)


def gauss_legendre_nodes(n: int):
    """Return the canonical Gauss-Legendre rule on ``[-1, 1]``."""
    data = gaussian_rule_data(GaussianRule(n), LebesgueMeasure())
    return data.nodes, data.weights


def gauss_laguerre_nodes(n: int):
    """Return the canonical Gauss-Laguerre rule for ``exp(-x)``."""
    if n < 1:
        raise ValueError("gauss_laguerre_nodes requires n >= 1")
    data = gaussian_rule_data(GaussianRule(n), LaguerreMeasure())
    return data.nodes, data.weights


__all__ = ["gauss_laguerre_nodes", "gauss_legendre_nodes", "gaussian_rule_data"]
