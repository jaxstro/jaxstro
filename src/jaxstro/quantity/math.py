"""Dimension-aware math wrappers for quantities."""

from __future__ import annotations

from fractions import Fraction

import jax.numpy as jnp

from . import units
from .errors import DimensionError
from .quantity import Quantity


def sqrt(x: Quantity) -> Quantity:
    """Square root with rational unit powers."""

    _require_quantity(x, "sqrt")
    return Quantity(jnp.sqrt(x.value), x.unit ** Fraction(1, 2))


def square(x: Quantity) -> Quantity:
    """Square a quantity and its unit."""

    _require_quantity(x, "square")
    return Quantity(jnp.square(x.value), x.unit**2)


def log(x: Quantity) -> Quantity:
    """Natural log of a dimensionless quantity."""

    _require_dimensionless(x, "log")
    return Quantity(jnp.log(x.value), units.dimensionless)


def exp(x: Quantity) -> Quantity:
    """Exponential of a dimensionless quantity."""

    _require_dimensionless(x, "exp")
    return Quantity(jnp.exp(x.value), units.dimensionless)


def sin(x: Quantity) -> Quantity:
    """Sine of a tagged angle quantity."""

    return Quantity(jnp.sin(_angle_radians(x, "sin")), units.dimensionless)


def cos(x: Quantity) -> Quantity:
    """Cosine of a tagged angle quantity."""

    return Quantity(jnp.cos(_angle_radians(x, "cos")), units.dimensionless)


def sum(x: Quantity, axis=None, keepdims: bool = False) -> Quantity:
    """Sum values while preserving the unit."""

    _require_quantity(x, "sum")
    return Quantity(jnp.sum(x.value, axis=axis, keepdims=keepdims), x.unit)


def mean(x: Quantity, axis=None, keepdims: bool = False) -> Quantity:
    """Mean values while preserving the unit."""

    _require_quantity(x, "mean")
    return Quantity(jnp.mean(x.value, axis=axis, keepdims=keepdims), x.unit)


def where(condition, x: Quantity, y: Quantity) -> Quantity:
    """Select between compatible quantities, preserving the left unit."""

    _require_quantity(x, "where")
    _require_quantity(y, "where")
    x._require_compatible(y, "where")
    return Quantity(jnp.where(condition, x.value, y.to_value(x.unit)), x.unit)


def _require_quantity(x, operation: str) -> None:
    if not isinstance(x, Quantity):
        raise TypeError(f"{operation} expects a Quantity.")


def _require_dimensionless(x: Quantity, operation: str) -> None:
    _require_quantity(x, operation)
    if not x.unit.is_dimensionless:
        raise DimensionError(
            f"{operation} requires dimensionless input, got {x.unit}.",
            operation=operation,
            expected=units.dimensionless.dimensions,
            actual=x.unit.dimensions,
        )


def _angle_radians(x: Quantity, operation: str):
    _require_quantity(x, operation)
    if x.unit.metadata.get("semantic") != "angle":
        raise DimensionError(
            f"{operation} requires a tagged angle unit, got {x.unit}.",
            operation=operation,
            expected="angle",
            actual=x.unit.metadata.get("semantic"),
        )
    return x.to_value(units.rad)


__all__ = ["cos", "exp", "log", "mean", "sin", "sqrt", "square", "sum", "where"]
