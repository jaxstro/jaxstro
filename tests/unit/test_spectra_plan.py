"""Tests for explicit fixed-shape spectral resampling plans."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.spectra import (
    CoveragePolicy,
    PointResamplingMethod,
    SpectralAxis,
    SpectralCoordinate,
    SpectralPlan,
    SpectralSampling,
)


def test_plan_is_a_pytree_with_static_coverage_policy() -> None:
    plan = SpectralPlan(
        target_axis=SpectralAxis.points(
            jnp.array([120.0, 180.0]),
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="nm",
        ),
        coverage_policy=CoveragePolicy.INTERSECTION,
    )

    leaves, treedef = jax.tree_util.tree_flatten(plan)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert len(leaves) == 1
    np.testing.assert_array_equal(rebuilt.target_axis.values, [120.0, 180.0])
    assert rebuilt.coverage_policy is CoveragePolicy.INTERSECTION


def test_plan_rejects_unknown_coverage_policy() -> None:
    axis = SpectralAxis.points(
        jnp.array([120.0, 180.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
    )

    with pytest.raises(ValueError):
        SpectralPlan(target_axis=axis, coverage_policy="extrapolate")


def test_plan_preserves_static_point_method_in_pytree_roundtrip() -> None:
    plan = SpectralPlan(
        target_axis=SpectralAxis.points(
            jnp.array([120.0, 180.0]),
            coordinate=SpectralCoordinate.WAVELENGTH,
            unit="nm",
        ),
        point_method=PointResamplingMethod.MONOTONE_CUBIC,
    )

    leaves, treedef = jax.tree_util.tree_flatten(plan)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert len(leaves) == 1
    assert rebuilt.point_method is PointResamplingMethod.MONOTONE_CUBIC


def test_bin_plan_rejects_monotone_cubic_point_method() -> None:
    axis = SpectralAxis.bins(
        jnp.array([100.0, 200.0, 400.0]),
        coordinate=SpectralCoordinate.WAVELENGTH,
        unit="nm",
        sampling=SpectralSampling.BIN_AVERAGES,
    )

    with pytest.raises(ValueError, match="point_method applies only"):
        SpectralPlan(axis, point_method=PointResamplingMethod.MONOTONE_CUBIC)
