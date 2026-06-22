# tests/test_grids.py
"""Tests for jaxstro.numerics.grids."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.numerics import grids


class TestLogAndGeometricGrids:
    """Tests for log-spaced grids and geometric bin helpers."""

    def test_log_grid_matches_geomspace(self):
        result = grids.log_grid(1.0, 1000.0, 4)
        expected = jnp.array([1.0, 10.0, 100.0, 1000.0])
        assert jnp.allclose(result, expected)

    def test_log_grid_base_argument(self):
        result = grids.log_grid(1.0, 8.0, 4, base=2.0)
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 4.0, 8.0]))

    def test_geometric_bin_edges_and_centers(self):
        edges = grids.geometric_bin_edges(1.0, 100.0, 2)
        centers = grids.geometric_bin_centers(edges)
        assert jnp.allclose(edges, jnp.array([1.0, 10.0, 100.0]))
        assert jnp.allclose(centers, jnp.array([jnp.sqrt(10.0), jnp.sqrt(1000.0)]))

    def test_linear_bin_centers(self):
        edges = jnp.array([0.0, 1.0, 3.0])
        assert jnp.allclose(grids.bin_centers(edges), jnp.array([0.5, 2.0]))

    def test_rejects_invalid_log_inputs(self):
        with pytest.raises(ValueError, match="positive"):
            grids.log_grid(0.0, 1.0, 4)
        with pytest.raises(ValueError, match="num"):
            grids.log_grid(1.0, 10.0, 1)
        with pytest.raises(ValueError, match="n_bins"):
            grids.geometric_bin_edges(1.0, 10.0, 0)


class TestConservativeRebin:
    """Tests for conservative redistribution of bin totals."""

    def test_preserves_total_when_rebinning_to_finer_grid(self):
        old_edges = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([2.0, 4.0])
        new_edges = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
        result = grids.conservative_rebin(old_edges, values, new_edges)
        assert jnp.allclose(result, jnp.array([1.0, 1.0, 2.0, 2.0]))
        assert jnp.allclose(jnp.sum(result), jnp.sum(values))

    def test_preserves_total_when_rebinning_to_coarser_grid(self):
        old_edges = jnp.array([0.0, 1.0, 2.0, 4.0])
        values = jnp.array([1.0, 3.0, 8.0])
        new_edges = jnp.array([0.0, 2.0, 4.0])
        result = grids.conservative_rebin(old_edges, values, new_edges)
        assert jnp.allclose(result, jnp.array([4.0, 8.0]))
        assert jnp.allclose(jnp.sum(result), jnp.sum(values))

    def test_partial_overlap_drops_outside_domain_explicitly(self):
        old_edges = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([2.0, 4.0])
        new_edges = jnp.array([-1.0, 0.5, 1.5, 3.0])
        result = grids.conservative_rebin(old_edges, values, new_edges)
        assert jnp.allclose(result, jnp.array([1.0, 3.0, 2.0]))

    def test_jit_and_grad_compatible_wrt_values(self):
        old_edges = jnp.array([0.0, 1.0, 2.0])
        new_edges = jnp.array([0.0, 0.5, 1.5, 2.0])

        @jax.jit
        def total(values):
            return jnp.sum(grids.conservative_rebin(old_edges, values, new_edges))

        values = jnp.array([2.0, 4.0])
        assert jnp.allclose(total(values), 6.0)
        assert jnp.allclose(jax.grad(total)(values), jnp.ones_like(values))

    def test_rejects_shape_mismatch_eagerly(self):
        with pytest.raises(ValueError, match="len"):
            grids.conservative_rebin(
                jnp.array([0.0, 1.0]), jnp.ones(2), jnp.arange(3.0)
            )
