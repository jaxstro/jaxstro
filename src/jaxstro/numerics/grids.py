# src/jaxstro/numerics/grids.py
"""Grid construction and conservative binning helpers."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from .checks import try_concrete_bool


def _raise_if_concrete_false(predicate, message: str) -> None:
    """Raise eagerly when a validation predicate is concrete and false."""
    result = try_concrete_bool(jnp.asarray(predicate))
    if result is False:
        raise ValueError(message)


def log_grid(
    start: float | Float[Array, ""],
    stop: float | Float[Array, ""],
    num: int,
    *,
    base: float = 10.0,
) -> Float[Array, " n"]:
    """Return ``num`` logarithmically spaced samples from ``start`` to ``stop``."""
    if num < 2:
        raise ValueError("num must be at least 2")
    start = jnp.asarray(start)
    stop = jnp.asarray(stop)
    base_arr = jnp.asarray(base)
    _raise_if_concrete_false(jnp.all(start > 0.0), "start must be positive")
    _raise_if_concrete_false(jnp.all(stop > 0.0), "stop must be positive")
    _raise_if_concrete_false(jnp.all(base_arr > 0.0), "base must be positive")
    _raise_if_concrete_false(jnp.all(base_arr != 1.0), "base must not be 1")
    log_base = jnp.log(base_arr)
    return jnp.power(
        base_arr,
        jnp.linspace(jnp.log(start) / log_base, jnp.log(stop) / log_base, num),
    )


def geometric_bin_edges(
    start: float | Float[Array, ""],
    stop: float | Float[Array, ""],
    n_bins: int,
    *,
    base: float = 10.0,
) -> Float[Array, " n"]:
    """Return logarithmically spaced bin edges for ``n_bins`` bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    return log_grid(start, stop, n_bins + 1, base=base)


def bin_centers(edges: Float[Array, " n"]) -> Float[Array, " n_minus_1"]:
    """Return arithmetic centers from monotonically increasing bin edges."""
    edges = jnp.asarray(edges)
    if edges.ndim != 1 or edges.shape[0] < 2:
        raise ValueError("edges must be a 1D array with at least two entries")
    _raise_if_concrete_false(
        jnp.all(jnp.diff(edges) > 0.0), "edges must be strictly increasing"
    )
    return 0.5 * (edges[:-1] + edges[1:])


def geometric_bin_centers(edges: Float[Array, " n"]) -> Float[Array, " n_minus_1"]:
    """Return geometric centers from positive, increasing bin edges."""
    edges = jnp.asarray(edges)
    if edges.ndim != 1 or edges.shape[0] < 2:
        raise ValueError("edges must be a 1D array with at least two entries")
    _raise_if_concrete_false(
        jnp.all(edges > 0.0), "geometric bin edges must be positive"
    )
    _raise_if_concrete_false(
        jnp.all(jnp.diff(edges) > 0.0), "edges must be strictly increasing"
    )
    return jnp.sqrt(edges[:-1] * edges[1:])


def conservative_rebin(
    old_edges: Float[Array, " n_old_plus_1"],
    values: Float[Array, " n_old"],
    new_edges: Float[Array, " n_new_plus_1"],
) -> Float[Array, " n_new"]:
    """
    Conservatively redistribute old per-bin totals onto new bin edges.

    ``values`` are interpreted as integrated bin totals, uniformly distributed
    across each old bin. The result preserves the total over the overlap of the
    old and new domains. New bins outside the old domain receive zero
    contribution.
    """
    old_edges = jnp.asarray(old_edges)
    values = jnp.asarray(values)
    new_edges = jnp.asarray(new_edges)
    if old_edges.ndim != 1 or new_edges.ndim != 1:
        raise ValueError("old_edges and new_edges must be 1D arrays")
    if values.ndim != 1:
        raise ValueError("values must be a 1D array")
    if old_edges.shape[0] != values.shape[0] + 1:
        raise ValueError("len(old_edges) must equal len(values) + 1")
    if new_edges.shape[0] < 2:
        raise ValueError("new_edges must contain at least two entries")
    _raise_if_concrete_false(
        jnp.all(jnp.diff(old_edges) > 0.0), "old_edges must be strictly increasing"
    )
    _raise_if_concrete_false(
        jnp.all(jnp.diff(new_edges) > 0.0), "new_edges must be strictly increasing"
    )

    old_lo = old_edges[:-1]
    old_hi = old_edges[1:]
    new_lo = new_edges[:-1]
    new_hi = new_edges[1:]
    widths = old_hi - old_lo
    density = values / widths

    overlap_lo = jnp.maximum(new_lo[:, None], old_lo[None, :])
    overlap_hi = jnp.minimum(new_hi[:, None], old_hi[None, :])
    overlap = jnp.maximum(overlap_hi - overlap_lo, 0.0)
    return overlap @ density


__all__ = [
    "log_grid",
    "geometric_bin_edges",
    "bin_centers",
    "geometric_bin_centers",
    "conservative_rebin",
]
