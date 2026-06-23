"""Serialization helpers for units and quantities."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import jax.numpy as jnp

from . import dimensions as d
from .parser import format_unit, parse_unit
from .quantity import Quantity
from .unit import Unit


def to_dict(quantity: Quantity) -> dict[str, Any]:
    """Serialize a scalar quantity to a JSON-safe dictionary."""

    return {
        "value": _scalar_to_json(quantity.value),
        "unit": unit_to_dict(quantity.unit),
    }


def from_dict(payload: dict[str, Any]) -> Quantity:
    """Deserialize a quantity dictionary produced by :func:`to_dict`."""

    return Quantity(payload["value"], unit_from_dict(payload["unit"]))


def unit_to_dict(unit: Unit) -> str | dict[str, Any]:
    """Serialize a unit compactly when possible, or structurally otherwise."""

    symbol = format_unit(unit)
    try:
        if parse_unit(symbol) == unit:
            return symbol
    except Exception:
        pass
    return {
        "symbol": unit.symbol,
        "scale_cgs": unit.scale_to_cgs,
        "dimensions": dimensions_to_dict(unit.dimensions),
    }


def unit_from_dict(payload: str | dict[str, Any]) -> Unit:
    """Deserialize a compact or structured unit payload."""

    if isinstance(payload, str):
        return parse_unit(payload)
    return Unit(
        payload["symbol"],
        payload["scale_cgs"],
        dimensions_from_dict(payload["dimensions"]),
    )


def dimensions_to_dict(dimension: d.Dimension) -> dict[str, int | str]:
    """Serialize nonzero dimension powers."""

    out: dict[str, int | str] = {}
    for name, exponent in zip(d.DIMENSION_NAMES, dimension.exponents):
        if exponent == 0:
            continue
        out[name] = exponent.numerator if exponent.denominator == 1 else str(exponent)
    return out


def dimensions_from_dict(payload: dict[str, int | str]) -> d.Dimension:
    """Deserialize dimension powers from a mapping."""

    return d.Dimension.from_powers(
        **{key: _parse_fraction(value) for key, value in payload.items()}
    )


def _parse_fraction(value: int | str) -> Fraction:
    return Fraction(value)


def _scalar_to_json(value):
    arr = jnp.asarray(value)
    if arr.ndim != 0:
        raise TypeError(
            "array quantity serialization is not implemented yet; use explicit "
            "array serialization helpers when they are added."
        )
    return arr.item()


__all__ = [
    "dimensions_from_dict",
    "dimensions_to_dict",
    "from_dict",
    "to_dict",
    "unit_from_dict",
    "unit_to_dict",
]
