"""Norm and tolerance policies shared by quadrature controllers."""

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array


class ErrorNorm(Protocol):
    def __call__(self, value: Array) -> Array: ...


class _StaticNorm:
    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, _aux, _children):
        return cls()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MaxNorm(_StaticNorm):
    def __call__(self, value: Array) -> Array:
        return jnp.max(jnp.abs(value))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class L1Norm(_StaticNorm):
    def __call__(self, value: Array) -> Array:
        return jnp.sum(jnp.abs(value))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class L2Norm(_StaticNorm):
    def __call__(self, value: Array) -> Array:
        return jnp.sqrt(jnp.sum(jnp.abs(value) ** 2))


def error_norm(error: Array, norm: ErrorNorm) -> Array:
    """Reduce scalar, vector, array, or complex error evidence."""
    return norm(jnp.asarray(error))


def tolerance_threshold(
    value: Array,
    *,
    epsabs: float | Array,
    epsrel: float | Array,
    norm: ErrorNorm,
) -> Array:
    """Return the larger absolute or relative tolerance."""
    value_norm = norm(jnp.asarray(value))
    dtype = jnp.result_type(value_norm, epsabs, epsrel, 0.0)
    absolute = jnp.asarray(epsabs, dtype=dtype)
    relative = jnp.asarray(epsrel, dtype=dtype) * value_norm
    return jnp.maximum(absolute, relative)
