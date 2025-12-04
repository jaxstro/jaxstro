# src/jaxstro/numerics/rng.py
"""
JAX PRNG helper utilities.

These helpers standardize a few common patterns for key management
across the jaxstro ecosystem, without prescribing a particular
randomness design.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

from .types import Array

KeyArray = Array  # For annotations; shape (..., 2) for PRNG keys.


@jax.jit
def split_key(key: KeyArray, num: int) -> KeyArray:
    """
    Split a key into `num` new keys.

    Parameters
    ----------
    key : PRNGKey
        Base key.
    num : int
        Number of new keys.

    Returns
    -------
    keys : ndarray, shape (num, 2)
        New keys.
    """
    return jrandom.split(key, num)


def split_tree(key: KeyArray, shape: Tuple[int, ...]) -> KeyArray:
    """
    Split a key into a tree (multi-dimensional array) of keys.

    Parameters
    ----------
    key : PRNGKey
        Base key.
    shape : tuple of int
        Desired shape of the key array.

    Returns
    -------
    keys : ndarray, shape (*shape, 2)
        Array of keys matching the requested shape.
    """
    num = 1
    for s in shape:
        num *= s
    keys_flat = jrandom.split(key, num)
    return jnp.reshape(keys_flat, shape + keys_flat.shape[1:])


def fold_in_indices(key: KeyArray, indices: Array) -> KeyArray:
    """
    Fold an array of integer indices into a base key to obtain
    an array of independent keys.

    Parameters
    ----------
    key : PRNGKey
        Base key.
    indices : ndarray of int
        Indices to fold in (any shape).

    Returns
    -------
    keys : ndarray, shape indices.shape + (2,)
        Array of derived keys.
    """
    indices = jnp.asarray(indices)

    def make_key(i):
        return jrandom.fold_in(key, i)

    flat = indices.ravel()
    flat_keys = jax.vmap(make_key)(flat)
    return jnp.reshape(flat_keys, indices.shape + flat_keys.shape[1:])


__all__ = ["KeyArray", "split_key", "split_tree", "fold_in_indices"]
