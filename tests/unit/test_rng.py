# tests/test_rng.py
"""
Tests for jaxstro.numerics.rng PRNG helpers.

Covers shape, dtype, and determinism for split_key, split_tree, and
fold_in_indices.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jaxstro.numerics import rng


class TestSplitKey:
    """Tests for split_key."""

    def test_shape(self):
        key = jrandom.PRNGKey(0)
        keys = rng.split_key(key, 5)
        assert keys.shape == (5, 2)

    def test_dtype(self):
        key = jrandom.PRNGKey(0)
        keys = rng.split_key(key, 3)
        assert keys.dtype == key.dtype
        assert jnp.issubdtype(keys.dtype, jnp.unsignedinteger)

    def test_determinism_same_key(self):
        k1 = rng.split_key(jrandom.PRNGKey(7), 4)
        k2 = rng.split_key(jrandom.PRNGKey(7), 4)
        assert jnp.array_equal(k1, k2)

    def test_different_keys_differ(self):
        k1 = rng.split_key(jrandom.PRNGKey(0), 4)
        k2 = rng.split_key(jrandom.PRNGKey(1), 4)
        assert not jnp.array_equal(k1, k2)

    def test_matches_jax_split(self):
        key = jrandom.PRNGKey(42)
        assert jnp.array_equal(rng.split_key(key, 6), jrandom.split(key, 6))


class TestSplitTree:
    """Tests for split_tree."""

    def test_shape_2d(self):
        key = jrandom.PRNGKey(0)
        keys = rng.split_tree(key, (3, 4))
        assert keys.shape == (3, 4, 2)

    def test_shape_3d(self):
        key = jrandom.PRNGKey(0)
        keys = rng.split_tree(key, (2, 3, 5))
        assert keys.shape == (2, 3, 5, 2)

    def test_dtype(self):
        key = jrandom.PRNGKey(0)
        keys = rng.split_tree(key, (2, 2))
        assert keys.dtype == key.dtype

    def test_determinism_same_key(self):
        k1 = rng.split_tree(jrandom.PRNGKey(3), (2, 3))
        k2 = rng.split_tree(jrandom.PRNGKey(3), (2, 3))
        assert jnp.array_equal(k1, k2)

    def test_different_keys_differ(self):
        k1 = rng.split_tree(jrandom.PRNGKey(0), (2, 3))
        k2 = rng.split_tree(jrandom.PRNGKey(9), (2, 3))
        assert not jnp.array_equal(k1, k2)

    def test_keys_are_independent(self):
        # Flattened keys should all differ from each other.
        keys = rng.split_tree(jrandom.PRNGKey(11), (2, 2)).reshape(-1, 2)
        for i in range(keys.shape[0]):
            for j in range(i + 1, keys.shape[0]):
                assert not jnp.array_equal(keys[i], keys[j])


class TestFoldInIndices:
    """Tests for fold_in_indices."""

    def test_shape_1d(self):
        key = jrandom.PRNGKey(0)
        idx = jnp.arange(5)
        keys = rng.fold_in_indices(key, idx)
        assert keys.shape == (5, 2)

    def test_shape_2d(self):
        key = jrandom.PRNGKey(0)
        idx = jnp.arange(6).reshape(2, 3)
        keys = rng.fold_in_indices(key, idx)
        assert keys.shape == (2, 3, 2)

    def test_dtype(self):
        key = jrandom.PRNGKey(0)
        keys = rng.fold_in_indices(key, jnp.arange(3))
        assert keys.dtype == key.dtype

    def test_determinism(self):
        key = jrandom.PRNGKey(5)
        idx = jnp.array([0, 1, 2, 3])
        k1 = rng.fold_in_indices(key, idx)
        k2 = rng.fold_in_indices(key, idx)
        assert jnp.array_equal(k1, k2)

    def test_matches_fold_in(self):
        key = jrandom.PRNGKey(123)
        idx = jnp.array([10, 20, 30])
        keys = rng.fold_in_indices(key, idx)
        for i, v in enumerate([10, 20, 30]):
            assert jnp.array_equal(keys[i], jrandom.fold_in(key, v))

    def test_distinct_indices_distinct_keys(self):
        key = jrandom.PRNGKey(0)
        keys = rng.fold_in_indices(key, jnp.array([0, 1, 2]))
        assert not jnp.array_equal(keys[0], keys[1])
        assert not jnp.array_equal(keys[1], keys[2])

    def test_different_base_keys_differ(self):
        idx = jnp.array([0, 1, 2])
        k1 = rng.fold_in_indices(jrandom.PRNGKey(0), idx)
        k2 = rng.fold_in_indices(jrandom.PRNGKey(1), idx)
        assert not jnp.array_equal(k1, k2)

    def test_jit_compatible(self):
        key = jrandom.PRNGKey(0)
        idx = jnp.arange(4)
        keys = jax.jit(rng.fold_in_indices)(key, idx)
        assert keys.shape == (4, 2)
