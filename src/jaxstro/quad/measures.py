"""Measure declarations for fixed and adaptive integration."""

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import jax

from jaxstro.quantity import Unit


def _validate_classical_parameter(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= -1.0:
        raise ValueError(f"{label} must be finite and greater than -1")


class _StaticMeasure:
    def tree_flatten(self):
        metadata = tuple(
            (name, getattr(self, name)) for name in self.__dataclass_fields__
        )
        return (), metadata

    @classmethod
    def tree_unflatten(cls, metadata, _children):
        return cls(**dict(metadata))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LebesgueMeasure(_StaticMeasure):
    pass


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class WeightedMeasure(_StaticMeasure):
    density: Callable[[Any, Any], Any]
    density_unit: Unit = field(kw_only=True)
    normalized: bool = field(default=False, kw_only=True)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class JacobiMeasure(_StaticMeasure):
    alpha: float
    beta: float
    normalized: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        _validate_classical_parameter(self.alpha, "Jacobi alpha")
        _validate_classical_parameter(self.beta, "Jacobi beta")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LaguerreMeasure(_StaticMeasure):
    alpha: float = 0.0
    normalized: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        _validate_classical_parameter(self.alpha, "Laguerre alpha")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PhysicistsHermiteMeasure(_StaticMeasure):
    normalized: bool = field(default=False, kw_only=True)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StandardNormalMeasure(_StaticMeasure):
    @property
    def normalized(self) -> bool:
        return True
