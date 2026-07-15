"""One-dimensional integration domains."""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Interval:
    lower: Any
    upper: Any
    breakpoints: tuple[Any, ...] = field(default=(), kw_only=True)

    def tree_flatten(self):
        children = (self.lower, self.upper, *self.breakpoints)
        return children, len(self.breakpoints)

    @classmethod
    def tree_unflatten(cls, count: int, children):
        lower, upper, *breakpoints = children
        if len(breakpoints) != count:
            raise ValueError("invalid Interval breakpoint PyTree")
        return cls(lower, upper, breakpoints=tuple(breakpoints))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RightInfinite:
    lower: Any

    def tree_flatten(self):
        return (self.lower,), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(children[0])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LeftInfinite:
    upper: Any

    def tree_flatten(self):
        return (self.upper,), None

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(children[0])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Infinite:
    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, _aux, _children):
        return cls()


def interval_orientation(domain: Interval) -> Array:
    delta = jnp.asarray(domain.upper) - jnp.asarray(domain.lower)
    return jnp.sign(delta)


def sorted_breakpoints(domain: Interval) -> Array:
    dtype = jnp.result_type(domain.lower, domain.upper, *domain.breakpoints, 0.0)
    if not domain.breakpoints:
        return jnp.empty((0,), dtype=dtype)
    values = jnp.asarray(domain.breakpoints, dtype=dtype)
    ascending = jnp.sort(values)
    return jnp.where(interval_orientation(domain) < 0.0, ascending[::-1], ascending)


def interval_is_valid(domain: Interval) -> Array:
    lower = jnp.asarray(domain.lower)
    upper = jnp.asarray(domain.upper)
    finite = jnp.isfinite(lower) & jnp.isfinite(upper)
    points = sorted_breakpoints(domain)
    ascending = jnp.sort(points)
    lo = jnp.minimum(lower, upper)
    hi = jnp.maximum(lower, upper)
    interior = jnp.all((points > lo) & (points < hi))
    unique = jnp.all(jnp.diff(ascending) > 0.0)
    return finite & interior & unique
