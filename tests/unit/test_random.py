"""Tests for explicit random-stream and resampling helpers."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

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

    @pytest.mark.parametrize(
        "resampler",
        [
            random.systematic_resample,
            random.stratified_resample,
            random.residual_resample,
        ],
    )
    @pytest.mark.parametrize(
        ("weights", "message"),
        [
            (jnp.array([]), "nonempty"),
            (jnp.ones((2, 2)), "one-dimensional"),
            (jnp.array([0.5, -0.1, 0.6]), "nonnegative"),
            (jnp.array([0.5, jnp.nan, 0.5]), "finite"),
            (jnp.array([0.5, jnp.inf, 0.5]), "finite"),
        ],
    )
    def test_resamplers_reject_invalid_eager_weights(self, resampler, weights, message):
        with pytest.raises(ValueError, match=message):
            resampler(jrandom.PRNGKey(0), weights)

    @pytest.mark.parametrize(
        "resampler",
        [
            random.systematic_resample,
            random.stratified_resample,
            random.residual_resample,
        ],
    )
    @pytest.mark.parametrize("num_samples", [0, -1])
    def test_resamplers_reject_nonpositive_sample_count(self, resampler, num_samples):
        with pytest.raises(ValueError, match="positive"):
            resampler(
                jrandom.PRNGKey(0),
                jnp.array([0.25, 0.75]),
                num_samples=num_samples,
            )

    @pytest.mark.parametrize(
        "resampler",
        [
            random.systematic_resample,
            random.stratified_resample,
            random.residual_resample,
        ],
    )
    def test_resamplers_keep_documented_uniform_zero_total_fallback(self, resampler):
        weights = jnp.zeros(3)
        indices = resampler(jrandom.PRNGKey(9), weights, num_samples=12)
        assert indices.shape == (12,)
        assert jnp.all((indices >= 0) & (indices < weights.shape[0]))
