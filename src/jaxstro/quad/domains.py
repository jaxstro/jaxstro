"""One-dimensional integration domains."""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array

from jaxstro.quantity import Unit


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
    scale: Any | None = field(default=None, kw_only=True)

    def tree_flatten(self):
        if self.scale is not None:
            return (self.lower, self.scale), True
        return (self.lower,), None

    @classmethod
    def tree_unflatten(cls, has_scale, children):
        if has_scale:
            return cls(children[0], scale=children[1])
        return cls(children[0])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LeftInfinite:
    upper: Any
    scale: Any | None = field(default=None, kw_only=True)

    def tree_flatten(self):
        if self.scale is not None:
            return (self.upper, self.scale), True
        return (self.upper,), None

    @classmethod
    def tree_unflatten(cls, has_scale, children):
        if has_scale:
            return cls(children[0], scale=children[1])
        return cls(children[0])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Infinite:
    unit: Unit | None = None
    scale: Any | None = field(default=None, kw_only=True)

    def tree_flatten(self):
        if self.scale is not None:
            return (self.scale,), (self.unit, True)
        return (), self.unit

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        if isinstance(auxiliary, tuple):
            unit, has_scale = auxiliary
            if has_scale:
                return cls(unit=unit, scale=children[0])
        unit = auxiliary
        return cls(unit=unit)


def improper_scale_value(domain: RightInfinite | LeftInfinite | Infinite):
    """Return the stopped scalar map scale, defaulting to the legacy value."""
    raw_scale = 1.0 if domain.scale is None else domain.scale
    scale = jnp.asarray(raw_scale)
    if scale.ndim != 0:
        raise ValueError("improper-domain scale must be scalar")
    if jnp.issubdtype(scale.dtype, jnp.bool_) or jnp.issubdtype(
        scale.dtype, jnp.complexfloating
    ):
        raise TypeError("improper-domain scale must be real")
    return jax.lax.stop_gradient(scale)


def improper_scale_is_valid(domain: RightInfinite | LeftInfinite | Infinite) -> Array:
    scale = improper_scale_value(domain)
    return jnp.isfinite(scale) & (scale > 0.0)


def interval_orientation(domain: Interval) -> Array:
    delta = jnp.asarray(domain.upper) - jnp.asarray(domain.lower)
    return jnp.sign(delta)


def sorted_breakpoints(domain: Interval) -> Array:
    dtype = jnp.result_type(domain.lower, domain.upper, *domain.breakpoints, 0.0)
    if not domain.breakpoints:
        return jnp.empty((0,), dtype=dtype)
    values = jnp.asarray(domain.breakpoints, dtype=dtype)
    values = jax.lax.stop_gradient(values)
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
