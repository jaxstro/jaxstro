"""JAX PyTree quantity values."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import wraps
from typing import Any

import jax
import jax.numpy as jnp

from . import dimensions as d
from .errors import DimensionError, EquivalencyError
from .unit import Unit
from .unit import dimensionless as dimensionless_unit


def _is_scalar_like(value: Any) -> bool:
    return not isinstance(value, Unit | Quantity)


def _conversion_factor(source: Unit, target: Unit) -> float:
    if not source.is_compatible_with(target):
        same_dimensions = source.dimensions == target.dimensions
        raise DimensionError(
            f"Cannot convert from {source} to {target}: incompatible dimensions "
            "or semantic tags.",
            operation="convert",
            expected=(
                target.metadata.get("semantic")
                if same_dimensions
                else target.dimensions
            ),
            actual=(
                source.metadata.get("semantic")
                if same_dimensions
                else source.dimensions
            ),
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
            equivalency_items = (
                equivalencies
                if isinstance(equivalencies, tuple | list)
                else (equivalencies,)
            )
            for equivalency in equivalency_items:
                converted = equivalency.convert(self, unit)
                if converted is not None:
                    return converted
            if not self.unit.is_compatible_with(unit):
                raise EquivalencyError(
                    f"No provided equivalency can convert {self.unit} to {unit}.",
                    operation="equivalency-convert",
                    expected=unit.dimensions,
                    actual=self.unit.dimensions,
                )
        return Quantity(self.value * _conversion_factor(self.unit, unit), unit)

    def to_value(self, unit: Unit, *, equivalencies=None):
        return self.to(unit, equivalencies=equivalencies).value

    def to_cgs(self) -> "Quantity":
        return self.to(_cgs_unit_for(self.unit))

    def to_cgs_value(self):
        return self.value * self.unit.scale_to_cgs

    def to_basis(self, basis, *, role: str | None = None) -> "Quantity":
        return self.to(basis.unit_for(self, role=role))

    def to_dict(self):
        from .serialization import to_dict

        return to_dict(self)

    @classmethod
    def from_dict(cls, payload):
        from .serialization import from_dict

        return from_dict(payload)

    def _require_compatible(self, other: "Quantity", operation: str) -> None:
        if not self.unit.is_compatible_with(other.unit):
            same_dimensions = self.unit.dimensions == other.unit.dimensions
            raise DimensionError(
                f"Cannot {operation} {self.unit} and {other.unit}: "
                "incompatible dimensions or semantic tags.",
                operation=operation,
                expected=(
                    self.unit.metadata.get("semantic")
                    if same_dimensions
                    else self.unit.dimensions
                ),
                actual=(
                    other.unit.metadata.get("semantic")
                    if same_dimensions
                    else other.unit.dimensions
                ),
            )

    def __add__(self, other):
        if isinstance(other, Quantity):
            self._require_compatible(other, "add")
            return Quantity(self.value + other.to_value(self.unit), self.unit)
        if _is_scalar_like(other):
            if not self.unit.is_compatible_with(dimensionless_unit):
                raise DimensionError(
                    f"Cannot add raw scalar to tagged or dimensional quantity {self.unit}.",
                    operation="add",
                    expected=self.unit.metadata.get("semantic") or self.unit.dimensions,
                    actual=(None if self.unit.is_dimensionless else d.dimensionless),
                )
            return Quantity(
                self.value + other * _conversion_factor(dimensionless_unit, self.unit),
                self.unit,
            )
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, Quantity):
            self._require_compatible(other, "subtract")
            return Quantity(self.value - other.to_value(self.unit), self.unit)
        if _is_scalar_like(other):
            if not self.unit.is_compatible_with(dimensionless_unit):
                raise DimensionError(
                    f"Cannot subtract raw scalar from tagged or dimensional quantity {self.unit}.",
                    operation="subtract",
                    expected=self.unit.metadata.get("semantic") or self.unit.dimensions,
                    actual=(None if self.unit.is_dimensionless else d.dimensionless),
                )
            return Quantity(
                self.value - other * _conversion_factor(dimensionless_unit, self.unit),
                self.unit,
            )
        return NotImplemented

    def __rsub__(self, other):
        if _is_scalar_like(other):
            if not self.unit.is_compatible_with(dimensionless_unit):
                raise DimensionError(
                    f"Cannot subtract tagged or dimensional quantity {self.unit} from raw scalar.",
                    operation="subtract",
                    expected=d.dimensionless,
                    actual=self.unit.metadata.get("semantic") or self.unit.dimensions,
                )
            return Quantity(
                other * _conversion_factor(dimensionless_unit, self.unit) - self.value,
                self.unit,
            )
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


def grad(fun: Callable[[Quantity], Quantity]) -> Callable[[Quantity], Quantity]:
    """Differentiate a scalar Quantity output with respect to one Quantity input."""

    @wraps(fun)
    def differentiated(x: Quantity) -> Quantity:
        if not isinstance(x, Quantity):
            raise TypeError("quantity.grad requires a Quantity input.")

        def scalar_value(value):
            output = fun(Quantity(value, x.unit))
            if not isinstance(output, Quantity):
                raise TypeError("quantity.grad requires a Quantity output.")
            if jnp.shape(output.value) != ():
                raise ValueError("quantity.grad requires a scalar Quantity output.")
            return output.value, output

        (_, output), derivative = jax.value_and_grad(scalar_value, has_aux=True)(
            x.value
        )
        return Quantity(derivative, output.unit / x.unit)

    return differentiated


def _cgs_unit_for(unit: Unit) -> Unit:
    from . import units

    if unit.dimensions == d.dimensionless:
        if unit.metadata.get("semantic") == "angle":
            return units.rad
        if unit.metadata.get("semantic") is not None:
            return Unit(f"cgs({unit})", 1.0, unit.dimensions, metadata=unit.metadata)
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


__all__ = ["Quantity", "grad"]
