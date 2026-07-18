"""Static-unit coordinate adapters for heterogeneous physical axes."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from jaxstro.quantity import Quantity, Unit
from jaxstro.quantity.errors import DimensionError


@dataclass(frozen=True)
class Axis:
    """One finite scalar coordinate interval with a static physical unit."""

    lower: Quantity
    upper: Quantity

    def __post_init__(self) -> None:
        if not isinstance(self.lower, Quantity) or not isinstance(self.upper, Quantity):
            raise TypeError("Axis bounds must be Quantity values")
        if not self.lower.unit.is_compatible_with(self.upper.unit):
            raise DimensionError(
                "Axis bounds must have compatible units.",
                operation="quad-axis",
                expected=self.lower.unit.dimensions,
                actual=self.upper.unit.dimensions,
            )
        if jnp.shape(self.lower.value) or jnp.shape(self.upper.value):
            raise ValueError("Axis bounds must be scalar")

    @property
    def unit(self) -> Unit:
        return self.lower.unit


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CoordinatePoint:
    """Coordinate-last values whose axis units are static PyTree metadata."""

    values: Any
    units: tuple[Unit, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.units, tuple) or not self.units:
            raise ValueError("CoordinatePoint units must be a nonempty tuple")
        if any(not isinstance(unit, Unit) for unit in self.units):
            raise TypeError("CoordinatePoint units must contain Unit values")
        shape = jnp.shape(self.values)
        if not shape or shape[-1] != len(self.units):
            raise ValueError(
                "CoordinatePoint final axis must match the number of units"
            )

    @property
    def shape(self):
        return jnp.shape(self.values)

    @property
    def dimension(self) -> int:
        return len(self.units)

    def axis(self, index: int) -> Quantity:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("CoordinatePoint axis index must be static")
        return Quantity(self.values[..., index], self.units[index])

    def as_quantity(self, unit: Unit) -> Quantity:
        converted = [
            self.axis(index).to_value(unit) for index in range(self.dimension)
        ]
        return Quantity(jnp.stack(converted, axis=-1), unit)

    def tree_flatten(self):
        return (self.values,), self.units

    @classmethod
    def tree_unflatten(cls, units: tuple[Unit, ...], children):
        (values,) = children
        return cls(values, units)


__all__ = ["Axis", "CoordinatePoint"]
