"""Explicit fixed-shape plans for spectral resampling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import jax

from .types import SpectralAxis, SpectralSampling


class CoveragePolicy(StrEnum):
    """Policy for a requested target spectral window."""

    INTERSECTION = "intersection"


class PointResamplingMethod(StrEnum):
    """Interpolation method for point-sampled spectra."""

    LINEAR = "linear"
    MONOTONE_CUBIC = "monotone_cubic"


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpectralPlan:
    """A fixed target axis and explicit fail-closed coverage policy."""

    target_axis: SpectralAxis
    coverage_policy: CoveragePolicy = CoveragePolicy.INTERSECTION
    point_method: PointResamplingMethod = PointResamplingMethod.LINEAR

    def __post_init__(self) -> None:
        coverage_policy = CoveragePolicy(self.coverage_policy)
        point_method = PointResamplingMethod(self.point_method)
        if (
            self.target_axis.sampling is not SpectralSampling.POINTS
            and point_method is not PointResamplingMethod.LINEAR
        ):
            raise ValueError("point_method applies only to point-sampled plans")
        object.__setattr__(self, "coverage_policy", coverage_policy)
        object.__setattr__(self, "point_method", point_method)

    def tree_flatten(self):
        return (self.target_axis,), (self.coverage_policy, self.point_method)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (target_axis,) = children
        coverage_policy, point_method = aux_data
        return cls(
            target_axis=target_axis,
            coverage_policy=coverage_policy,
            point_method=point_method,
        )


__all__ = ["CoveragePolicy", "PointResamplingMethod", "SpectralPlan"]
