"""Immutable physical unit objects."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Mapping

from . import dimensions as d
from .errors import DimensionError


def _coerce_power(power: int | Fraction) -> Fraction:
    if isinstance(power, bool) or not isinstance(power, int | Fraction):
        raise DimensionError(
            "Unit powers must be exact rational values; rationalize decimals "
            "before constructing units.",
            operation="unit-power",
            actual=power,
        )
    return Fraction(power)


def _format_power(symbol: str, power: Fraction) -> str:
    if power == 1:
        return symbol
    if power.denominator == 1:
        return f"{symbol}^{power.numerator}"
    return f"{symbol}^({power.numerator}/{power.denominator})"


def _combine_symbols(left: str, right: str, op: str) -> str:
    if left == "1":
        return right if op == "*" else f"1/{right}"
    if right == "1":
        return left
    return f"{left} {right}" if op == "*" else f"{left}/{right}"


@dataclass(frozen=True, eq=False)
class Unit:
    """Concrete unit with a scale to CGS and exact dimensions."""

    symbol: str
    scale_to_cgs: float
    dimensions: d.Dimension
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    __array_priority__ = 1000

    def __post_init__(self) -> None:
        object.__setattr__(self, "scale_to_cgs", float(self.scale_to_cgs))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_dimensionless(self) -> bool:
        return self.dimensions.is_dimensionless

    def is_compatible_with(self, other: "Unit") -> bool:
        return self.dimensions.is_compatible_with(other.dimensions)

    def __mul__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            from .quantity import Quantity

            return Quantity(other, self)
        return _canonicalize(
            Unit(
                _combine_symbols(str(self), str(other), "*"),
                self.scale_to_cgs * other.scale_to_cgs,
                self.dimensions * other.dimensions,
            )
        )

    def __rmul__(self, other):
        from .quantity import Quantity

        return Quantity(other, self)

    def __truediv__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        return _canonicalize(
            Unit(
                _combine_symbols(str(self), str(other), "/"),
                self.scale_to_cgs / other.scale_to_cgs,
                self.dimensions / other.dimensions,
            )
        )

    def __rtruediv__(self, other):
        from .quantity import Quantity

        return Quantity(other, dimensionless / self)

    def __pow__(self, power: int | Fraction) -> "Unit":
        exponent = _coerce_power(power)
        return _canonicalize(
            Unit(
                _format_power(str(self), exponent),
                self.scale_to_cgs ** float(exponent),
                self.dimensions**exponent,
            )
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Unit)
            and self.dimensions == other.dimensions
            and self.scale_to_cgs == other.scale_to_cgs
        )

    def __hash__(self) -> int:
        return hash((self.dimensions, self.scale_to_cgs))

    def __str__(self) -> str:
        return self.symbol

    def __repr__(self) -> str:
        return f"Unit({self.symbol!r})"


dimensionless = Unit("1", 1.0, d.dimensionless, name="dimensionless")


def _canonicalize(unit: Unit) -> Unit:
    """Return a friendly canonical unit for high-value CGS composites."""

    if unit.dimensions == d.dimensionless and math.isclose(unit.scale_to_cgs, 1.0):
        return dimensionless
    if unit.dimensions == d.energy and math.isclose(unit.scale_to_cgs, 1.0):
        return Unit("erg", 1.0, d.energy, name="erg")
    return unit


__all__ = ["Unit", "dimensionless"]
