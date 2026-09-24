"""Shared integrand calling, node-axis, and payload-shape helpers."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from jaxstro.quantity import Quantity

from .measures import WeightedMeasure

_QUANTITY_ACTIVATION_MESSAGE = (
    "quantity-valued integrand output requires quad.integrate with quantity "
    "epsabs; quantity epsabs activates a dimensionless quantity domain"
)


def node_weighted_sum(values, weights):
    """Sum ``weights[i] * values[i]`` over the leading node axis of a payload."""
    shape = (weights.shape[0],) + (1,) * (values.ndim - 1)
    return jnp.sum(values * jnp.reshape(weights, shape), axis=0)


def payload_dtype(values_dtype, node_dtype):
    """Real node dtype, or its complex counterpart for a complex payload."""
    if jnp.issubdtype(values_dtype, jnp.complexfloating):
        return jnp.complex64 if node_dtype == jnp.float32 else jnp.complex128
    return node_dtype


def has_explicit_args(args: Any) -> bool:
    return not (isinstance(args, tuple) and len(args) == 0)


def call_integrand(fun: Callable, nodes, args: Any, has_args: bool):
    return fun(nodes, args) if has_args else fun(nodes)


def validate_node_values(values, node_count: int, *, context: str):
    if isinstance(values, Quantity):
        raise TypeError(_QUANTITY_ACTIVATION_MESSAGE)
    values = jnp.asarray(values)
    if values.ndim == 0 or values.shape[0] != node_count:
        raise ValueError(f"{context} integrand output must have a leading node axis")
    return values


def infer_payload_zero(
    fun: Callable,
    *,
    args: Any,
    node_count: int,
    node_dtype,
    context: str = "adaptive quadrature",
):
    """Infer payload shape without numerically evaluating an integrand."""
    has_args = has_explicit_args(args)
    abstract_nodes = jax.ShapeDtypeStruct((node_count,), node_dtype)
    abstract = jax.eval_shape(
        lambda nodes: call_integrand(fun, nodes, args, has_args),
        abstract_nodes,
    )
    if isinstance(abstract, Quantity):
        raise TypeError(_QUANTITY_ACTIVATION_MESSAGE)
    if (
        not hasattr(abstract, "shape")
        or len(abstract.shape) == 0
        or abstract.shape[0] != node_count
    ):
        raise ValueError(f"{context} integrand output must have a leading node axis")
    return jnp.zeros(abstract.shape[1:], dtype=abstract.dtype)


def expand_node_factor(factor, payload_ndim: int):
    factor = jnp.asarray(factor)
    return jnp.reshape(factor, factor.shape + (1,) * (payload_ndim - 1))


def density_values(measure, nodes, args: Any):
    """Evaluate a general density once and preserve the node shape."""
    if isinstance(measure, WeightedMeasure):
        density = jnp.asarray(measure.density(nodes, args))
        if density.ndim == 0:
            return jnp.broadcast_to(density, nodes.shape)
        if density.shape != nodes.shape:
            raise ValueError("weighted measure density must preserve the node shape")
        return density
    return jnp.ones_like(nodes)


__all__ = [
    "call_integrand",
    "node_weighted_sum",
    "payload_dtype",
    "density_values",
    "expand_node_factor",
    "has_explicit_args",
    "infer_payload_zero",
    "validate_node_values",
]
