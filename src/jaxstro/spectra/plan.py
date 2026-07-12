"""Explicit fixed-shape plans for spectral resampling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import jax

from .types import SpectralAxis


class CoveragePolicy(StrEnum):
    """Policy for a requested target spectral window."""

    INTERSECTION = "intersection"


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SpectralPlan:
    """A fixed target axis and explicit fail-closed coverage policy."""

    target_axis: SpectralAxis
    coverage_policy: CoveragePolicy = CoveragePolicy.INTERSECTION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coverage_policy",
            CoveragePolicy(self.coverage_policy),
        )

    def tree_flatten(self):
        return (self.target_axis,), self.coverage_policy

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (target_axis,) = children
        return cls(target_axis=target_axis, coverage_policy=aux_data)


__all__ = ["CoveragePolicy", "SpectralPlan"]
