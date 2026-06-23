"""JAX PyTree quantity values."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import jax

from . import dimensions as d
from .errors import DimensionError
from .unit import Unit
from .unit import dimensionless as dimensionless_unit


def _is_scalar_like(value: Any) -> bool:
    return not isinstance(value, Unit | Quantity)


def _conversion_factor(source: Unit, target: Unit) -> float:
    if not source.is_compatible_with(target):
        raise DimensionError(
            f"Cannot convert from {source} to {target}: incompatible dimensions.",
            operation="convert",
            expected=target.dimensions,
            actual=source.dimensions,
        )
    return source.scale_to_cgs / target.scale_to_cgs


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Quantity:
    """A JAX value carrying one static physical unit."""

    value: Any
    unit: Unit

    __array_priority__ = 1000

    def tree_flatten(self):
        return (self.value,), self.unit

    @classmethod
    def tree_unflatten(cls, unit: Unit, children):
        (value,) = children
        return cls(value, unit)

    def to(self, unit: Unit, *, equivalencies=None) -> "Quantity":
        if equivalencies is not None:
            for equivalency in equivalencies:
                converted = equivalency.convert(self, unit)
                if converted is not None:
                    return converted
        return Quantity(self.value * _conversion_factor(self.unit, unit), unit)

    def to_value(self, unit: Unit, *, equivalencies=None):
        return self.to(unit, equivalencies=equivalencies).value

    def to_cgs(self) -> "Quantity":
        return self.to(_cgs_unit_for(self.unit))

    def to_cgs_value(self):
        return self.value * self.unit.scale_to_cgs

    def _require_compatible(self, other: "Quantity", operation: str) -> None:
        if not self.unit.is_compatible_with(other.unit):
            raise DimensionError(
                f"Cannot {operation} {self.unit} and {other.unit}: "
                "incompatible dimensions.",
                operation=operation,
                expected=self.unit.dimensions,
                actual=other.unit.dimensions,
            )

    def __add__(self, other):
        if isinstance(other, Quantity):
            self._require_compatible(other, "add")
            return Quantity(self.value + other.to_value(self.unit), self.unit)
        if _is_scalar_like(other):
            if not self.unit.is_dimensionless:
                raise DimensionError(
                    f"Cannot add raw scalar to dimensional quantity {self.unit}.",
                    operation="add",
                    expected=self.unit.dimensions,
                    actual=d.dimensionless,
                )
            return Quantity(self.value + other, self.unit)
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Quantity):
            self._require_compatible(other, "subtract")
            return Quantity(self.value - other.to_value(self.unit), self.unit)
        if _is_scalar_like(other):
            if not self.unit.is_dimensionless:
                raise DimensionError(
                    f"Cannot subtract raw scalar from dimensional quantity {self.unit}.",
                    operation="subtract",
                    expected=self.unit.dimensions,
                    actual=d.dimensionless,
                )
            return Quantity(self.value - other, self.unit)
        return NotImplemented

    def __rsub__(self, other):
        if _is_scalar_like(other):
            if not self.unit.is_dimensionless:
                raise DimensionError(
                    f"Cannot subtract dimensional quantity {self.unit} from raw scalar.",
                    operation="subtract",
                    expected=d.dimensionless,
                    actual=self.unit.dimensions,
                )
            return Quantity(other - self.value, self.unit)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, self.unit * other.unit)
        if isinstance(other, Unit):
            return Quantity(self.value, self.unit * other)
        if _is_scalar_like(other):
            return Quantity(self.value * other, self.unit)
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, self.unit / other.unit)
        if isinstance(other, Unit):
            return Quantity(self.value, self.unit / other)
        if _is_scalar_like(other):
            return Quantity(self.value / other, self.unit)
        return NotImplemented

    def __rtruediv__(self, other):
        if _is_scalar_like(other):
            return Quantity(other / self.value, dimensionless_unit / self.unit)
        return NotImplemented

    def __pow__(self, power: int | Fraction) -> "Quantity":
        return Quantity(self.value**power, self.unit**power)


def _cgs_unit_for(unit: Unit) -> Unit:
    from . import units

    if unit.dimensions == d.dimensionless:
        return dimensionless_unit
    if unit.dimensions == d.mass:
        return units.g
    if unit.dimensions == d.length:
        return units.cm
    if unit.dimensions == d.time:
        return units.s
    if unit.dimensions == d.temperature:
        return units.K
    if unit.dimensions == d.energy:
        return units.erg
    if unit.dimensions == d.power:
        return units.erg / units.s
    return Unit(f"cgs({unit})", 1.0, unit.dimensions)


__all__ = ["Quantity"]
