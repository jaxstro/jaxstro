"""Static fixed-quadrature rule declarations and constructed rule data."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
from jaxtyping import Array


class FixedRuleData(NamedTuple):
    """Numerical nodes, weights, and exactness metadata for a fixed rule."""

    nodes: Array
    weights: Array
    degree: int
    nested: bool


class _StaticIntegerRule:
    _field_name: str
    _minimum: int

    def __post_init__(self) -> None:
        value = getattr(self, self._field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < self._minimum
        ):
            requirement = (
                "a positive integer" if self._minimum == 1 else "a nonnegative integer"
            )
            raise ValueError(
                f"{type(self).__name__} {self._field_name} must be {requirement}"
            )

    def tree_flatten(self):
        return (), getattr(self, self._field_name)

    @classmethod
    def tree_unflatten(cls, value, _children):
        return cls(value)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GaussianRule(_StaticIntegerRule):
    order: int
    _field_name = "order"
    _minimum = 1


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ClenshawCurtisRule(_StaticIntegerRule):
    order: int
    _field_name = "order"
    _minimum = 1


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FejerIRule(_StaticIntegerRule):
    order: int
    _field_name = "order"
    _minimum = 1


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FejerIIRule(_StaticIntegerRule):
    order: int
    _field_name = "order"
    _minimum = 1


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TanhSinhRule(_StaticIntegerRule):
    level: int
    _field_name = "level"
    _minimum = 0


__all__ = [
    "ClenshawCurtisRule",
    "FejerIRule",
    "FejerIIRule",
    "FixedRuleData",
    "GaussianRule",
    "TanhSinhRule",
]
