"""Tests for explicit random-stream and resampling helpers."""

import jax
import jax.numpy as jnp
import jax.random as jrandom

from jaxstro.numerics import random


class TestKeyStreams:
    """Tests for explicit key-stream helpers."""

    def test_key_stream_returns_next_key_and_requested_subkeys(self):
        key = jrandom.PRNGKey(0)
        next_key, keys = random.key_stream(key, 4)
        assert next_key.shape == (2,)
        assert keys.shape == (4, 2)
        assert jnp.issubdtype(keys.dtype, jnp.unsignedinteger)

    def test_key_stream_is_deterministic(self):
        key = jrandom.PRNGKey(123)
        first = random.key_stream(key, 3)
        second = random.key_stream(key, 3)
        assert jnp.array_equal(first[0], second[0])
        assert jnp.array_equal(first[1], second[1])

    def test_fold_in_stream_matches_shape(self):
        key = jrandom.PRNGKey(5)
        keys = random.fold_in_stream(key, 6, start=10)
        assert keys.shape == (6, 2)
        assert jnp.array_equal(keys[0], jrandom.fold_in(key, 10))

    def test_seed_manifest_is_deterministic(self):
        manifest = random.seed_manifest(42, stream="unit-test")
        assert manifest == random.seed_manifest(42, stream="unit-test")
        assert manifest["seed"] == 42
        assert manifest["stream"] == "unit-test"


class TestResampling:
    """Tests for shape-stable resampling kernels."""

    def test_systematic_resample_degenerate_weight(self):
        key = jrandom.PRNGKey(0)
        weights = jnp.array([0.0, 1.0, 0.0])
        indices = random.systematic_resample(key, weights, num_samples=5)
        assert indices.shape == (5,)
        assert jnp.array_equal(indices, jnp.ones(5, dtype=indices.dtype))

    def test_stratified_resample_degenerate_weight(self):
        key = jrandom.PRNGKey(0)
        weights = jnp.array([0.0, 0.0, 2.0])
        indices = random.stratified_resample(key, weights, num_samples=4)
        assert indices.shape == (4,)
        assert jnp.array_equal(indices, jnp.full((4,), 2, dtype=indices.dtype))

    def test_residual_resample_exact_integer_counts(self):
        key = jrandom.PRNGKey(0)
        weights = jnp.array([0.6, 0.4])
        indices = random.residual_resample(key, weights, num_samples=5)
        assert jnp.array_equal(indices, jnp.array([0, 0, 0, 1, 1], dtype=indices.dtype))

    def test_resamplers_are_jit_compatible_with_static_sample_count(self):
        key = jrandom.PRNGKey(0)
        weights = jnp.array([0.2, 0.3, 0.5])
        sample = jax.jit(random.systematic_resample, static_argnames=("num_samples",))
        indices = sample(key, weights, num_samples=8)
        assert indices.shape == (8,)
        assert jnp.all((indices >= 0) & (indices < weights.shape[0]))
