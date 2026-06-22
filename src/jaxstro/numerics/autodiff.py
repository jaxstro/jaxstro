"""Small autodiff product helpers built on JAX primitives."""

from typing import Callable

import jax
from jaxtyping import Array, Float


def jvp(
    f: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    x: Float[Array, "..."],
    tangent: Float[Array, "..."],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Return ``f(x)`` and the Jacobian-vector product ``J @ tangent``."""
    return jax.jvp(f, (x,), (tangent,))


def vjp(
    f: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    x: Float[Array, "..."],
    cotangent: Float[Array, "..."],
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Return ``f(x)`` and the vector-Jacobian product ``cotangent @ J``."""
    value, pullback = jax.vjp(f, x)
    (product,) = pullback(cotangent)
    return value, product


def jacobian_vector_product(
    f: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    x: Float[Array, "..."],
    tangent: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Return only the Jacobian-vector product ``J @ tangent``."""
    _, product = jvp(f, x, tangent)
    return product


def vector_jacobian_product(
    f: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    x: Float[Array, "..."],
    cotangent: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Return only the vector-Jacobian product ``cotangent @ J``."""
    _, product = vjp(f, x, cotangent)
    return product


def hvp(
    f: Callable[[Float[Array, "..."]], Float[Array, ""]],
    x: Float[Array, "..."],
    tangent: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Return the Hessian-vector product of scalar ``f`` at ``x``."""
    return jax.jvp(jax.grad(f), (x,), (tangent,))[1]


def gauss_newton_product(
    residual_fn: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    x: Float[Array, "..."],
    tangent: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Return ``J.T @ J @ tangent`` for a residual function."""
    _, j_tangent = jvp(residual_fn, x, tangent)
    return vector_jacobian_product(residual_fn, x, j_tangent)


def empirical_fisher_product(
    score_fn: Callable[[Float[Array, "..."], Float[Array, "..."]], Float[Array, "..."]],
    params: Float[Array, "..."],
    data: Float[Array, "..."],
    tangent: Float[Array, "..."],
) -> Float[Array, "..."]:
    """Return the mean empirical Fisher-style product from per-example scores."""
    scores = jax.vmap(lambda datum: score_fn(params, datum))(data)
    return scores.T @ (scores @ tangent) / scores.shape[0]


__all__ = [
    "jvp",
    "vjp",
    "jacobian_vector_product",
    "vector_jacobian_product",
    "hvp",
    "gauss_newton_product",
    "empirical_fisher_product",
]
