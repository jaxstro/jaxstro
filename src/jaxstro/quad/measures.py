"""Measure declarations for fixed and adaptive integration."""

from dataclasses import dataclass, field
from typing import Any, Callable

import jax

from jaxstro.quantity import Unit


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
        if self.alpha <= -1.0 or self.beta <= -1.0:
            raise ValueError("Jacobi alpha and beta must be greater than -1")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LaguerreMeasure(_StaticMeasure):
    alpha: float = 0.0
    normalized: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if self.alpha <= -1.0:
            raise ValueError("Laguerre alpha must be greater than -1")


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
