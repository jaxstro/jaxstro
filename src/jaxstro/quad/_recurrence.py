"""Shared classical recurrence engine for Gaussian quadrature rules."""

import math
from typing import Any

import jax.numpy as jnp

from .measures import (
    JacobiMeasure,
    LaguerreMeasure,
    LebesgueMeasure,
    PhysicistsHermiteMeasure,
    StandardNormalMeasure,
)
from .rules import FixedRuleData, GaussianRule


def _golub_welsch(diagonal, off_diagonal, mass: float) -> FixedRuleData:
    matrix = jnp.diag(diagonal)
    if diagonal.shape[0] > 1:
        matrix = matrix + jnp.diag(off_diagonal, 1) + jnp.diag(off_diagonal, -1)
    nodes, vectors = jnp.linalg.eigh(matrix)
    weights = jnp.asarray(mass, dtype=nodes.dtype) * vectors[0, :] ** 2
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
    off_diagonal = (2.0 / twice) * jnp.sqrt(
        index
        * (index + alpha)
        * (index + beta)
        * (index + total)
        / ((twice - 1.0) * (twice + 1.0))
    )
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


__all__ = ["gaussian_rule_data"]
