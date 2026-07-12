"""Explicit PRNG stream and resampling helpers."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, Int, UInt32

from .checks import try_concrete_bool

KeyArray = UInt32[Array, "2"]


@partial(jax.jit, static_argnames=("num",))
def key_stream(key: KeyArray, num: int) -> tuple[KeyArray, UInt32[Array, "num 2"]]:
    """Split ``key`` into a next key and ``num`` subkeys."""
    keys = jrandom.split(key, num + 1)
    return keys[0], keys[1:]


@partial(jax.jit, static_argnames=("num", "start"))
def fold_in_stream(
    key: KeyArray, num: int, *, start: int = 0
) -> UInt32[Array, "num 2"]:
    """Return ``num`` keys made by folding in consecutive integer indices."""
    indices = jnp.arange(start, start + num)
    return jax.vmap(lambda index: jrandom.fold_in(key, index))(indices)


def seed_manifest(
    seed: int,
    *,
    stream: str = "default",
    algorithm: str = "jax.random",
) -> dict[str, int | str]:
    """Return a deterministic seed manifest for logs and reports."""
    return {"algorithm": algorithm, "seed": int(seed), "stream": stream}


def _normalize_weights(weights: Float[Array, "n"]) -> Float[Array, "n"]:
    total = jnp.sum(weights)
    n = weights.shape[0]
    return jnp.where(total > 0.0, weights / total, jnp.ones_like(weights) / n)


def _validate_resampling_inputs(
    weights: Float[Array, "n"], num_samples: int | None
) -> Float[Array, "n"]:
    """Validate the eager boundary while preserving transformed execution."""
    weights = jnp.asarray(weights)
    if weights.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if weights.shape[0] == 0:
        raise ValueError("weights must be nonempty")
    if num_samples is not None and num_samples <= 0:
        raise ValueError("num_samples must be positive")

    is_finite = try_concrete_bool(jnp.all(jnp.isfinite(weights)))
    if is_finite is False:
        raise ValueError("weights must be finite")
    is_nonnegative = try_concrete_bool(jnp.all(weights >= 0.0))
    if is_nonnegative is False:
        raise ValueError("weights must be nonnegative")
    return weights


def _inverse_cdf_indices(
    positions: Float[Array, "m"],
    probabilities: Float[Array, "n"],
) -> Int[Array, "m"]:
    cdf = jnp.cumsum(probabilities)
    return jnp.searchsorted(cdf, positions, side="right")


@partial(jax.jit, static_argnames=("num_samples",))
def _systematic_resample_core(
    key: KeyArray,
    weights: Float[Array, "n"],
    *,
    num_samples: int | None = None,
) -> Int[Array, "m"]:
    probabilities = _normalize_weights(weights)
    n = weights.shape[0] if num_samples is None else num_samples
    offset = jrandom.uniform(key, ()) / n
    positions = offset + jnp.arange(n) / n
    return _inverse_cdf_indices(positions, probabilities)


def systematic_resample(
    key: KeyArray,
    weights: Float[Array, "n"],
    *,
    num_samples: int | None = None,
) -> Int[Array, "m"]:
    """Systematic resampling from finite, nonnegative weights.

    Concrete eager inputs are validated. Value-dependent validation is skipped
    when ``weights`` are traced, so transformed callers own that precondition.
    """
    weights = _validate_resampling_inputs(weights, num_samples)
    return _systematic_resample_core(key, weights, num_samples=num_samples)


@partial(jax.jit, static_argnames=("num_samples",))
def _stratified_resample_core(
    key: KeyArray,
    weights: Float[Array, "n"],
    *,
    num_samples: int | None = None,
) -> Int[Array, "m"]:
    probabilities = _normalize_weights(weights)
    n = weights.shape[0] if num_samples is None else num_samples
    uniforms = jrandom.uniform(key, (n,))
    positions = (jnp.arange(n) + uniforms) / n
    return _inverse_cdf_indices(positions, probabilities)


def stratified_resample(
    key: KeyArray,
    weights: Float[Array, "n"],
    *,
    num_samples: int | None = None,
) -> Int[Array, "m"]:
    """Stratified resampling from finite, nonnegative weights.

    Concrete eager inputs are validated. Value-dependent validation is skipped
    when ``weights`` are traced, so transformed callers own that precondition.
    """
    weights = _validate_resampling_inputs(weights, num_samples)
    return _stratified_resample_core(key, weights, num_samples=num_samples)


@partial(jax.jit, static_argnames=("num_samples",))
def _residual_resample_core(
    key: KeyArray,
    weights: Float[Array, "n"],
    *,
    num_samples: int | None = None,
) -> Int[Array, "m"]:
    probabilities = _normalize_weights(weights)
    n = weights.shape[0] if num_samples is None else num_samples
    expected = n * probabilities
    counts = jnp.floor(expected).astype(jnp.int32)
    deterministic_total = jnp.sum(counts)
    residual_n = n - deterministic_total

    cumulative_counts = jnp.cumsum(counts)
    slots = jnp.arange(n)
    deterministic_indices = jnp.searchsorted(cumulative_counts, slots, side="right")

    residual_weights = expected - counts
    residual_total = jnp.sum(residual_weights)
    residual_probs = jnp.where(
        residual_total > 0.0,
        residual_weights / residual_total,
        probabilities,
    )
    residual_safe_n = jnp.maximum(residual_n, 1)
    offset = jrandom.uniform(key, ()) / residual_safe_n
    residual_rank = slots - deterministic_total
    residual_positions = offset + residual_rank / residual_safe_n
    residual_indices = _inverse_cdf_indices(residual_positions, residual_probs)
    return jnp.where(
        slots < deterministic_total, deterministic_indices, residual_indices
    )


def residual_resample(
    key: KeyArray,
    weights: Float[Array, "n"],
    *,
    num_samples: int | None = None,
) -> Int[Array, "m"]:
    """Residual resampling with deterministic floor counts and a random tail.

    Concrete eager inputs are validated. Value-dependent validation is skipped
    when ``weights`` are traced, so transformed callers own that precondition.
    """
    weights = _validate_resampling_inputs(weights, num_samples)
    return _residual_resample_core(key, weights, num_samples=num_samples)


__all__ = [
    "KeyArray",
    "key_stream",
    "fold_in_stream",
    "seed_manifest",
    "systematic_resample",
    "stratified_resample",
    "residual_resample",
]
