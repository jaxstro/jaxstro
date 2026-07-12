"""Deterministic host-side topology selection for atmosphere grids."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import product

import numpy as np

from jaxstro.spectra import SpectrumStatusCode


class TopologyKind(StrEnum):
    """Prepared interpolation topology kind."""

    NONE = "none"
    RECTILINEAR = "rectilinear"
    SIMPLEX = "simplex"


@dataclass(frozen=True)
class GridTopology:
    """Available parameter nodes and an explicit approved-simplex manifest."""

    parameter_names: tuple[str, ...]
    points: tuple[tuple[float, ...], ...]
    approved_simplices: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        rank = len(self.parameter_names)
        if rank < 1 or not self.points:
            raise ValueError("grid topology requires parameters and points")
        if any(len(point) != rank for point in self.points):
            raise ValueError("all topology points must have the same rank")
        if not np.all(np.isfinite(np.asarray(self.points, dtype=float))):
            raise ValueError("topology points must be finite")
        if len(set(self.points)) != len(self.points):
            raise ValueError("topology points must be unique")
        for simplex in self.approved_simplices:
            if len(simplex) != rank + 1 or len(set(simplex)) != len(simplex):
                raise ValueError("simplex indices must define dimension plus one nodes")
            if any(index < 0 or index >= len(self.points) for index in simplex):
                raise ValueError("simplex indices must reference topology points")


@dataclass(frozen=True)
class TopologySelection:
    """Host-selected fixed topology or a structured scientific limitation."""

    kind: TopologyKind
    vertex_indices: tuple[int, ...]
    status: SpectrumStatusCode


def _bounding_pair(values: tuple[float, ...], query: float) -> tuple[float, float]:
    if query == values[-1]:
        return values[-2], values[-1]
    for lower, upper in zip(values[:-1], values[1:], strict=True):
        if lower <= query <= upper:
            return lower, upper
    raise ValueError("query is outside parameter bounds")


def _inside_simplex(vertices: np.ndarray, query: np.ndarray) -> bool:
    edge_matrix = (vertices[1:] - vertices[0]).T
    try:
        tail = np.linalg.solve(edge_matrix, query - vertices[0])
    except np.linalg.LinAlgError:
        return False
    weights = np.concatenate(([1.0 - np.sum(tail)], tail))
    return bool(np.all(weights >= -1.0e-12) and np.all(weights <= 1.0 + 1.0e-12))


def select_topology(
    topology: GridTopology,
    query: tuple[float, ...],
) -> TopologySelection:
    """Select a complete cell first, then an approved containing simplex."""
    if len(query) != len(topology.parameter_names):
        raise ValueError("topology query rank must match parameter names")
    points = np.asarray(topology.points, dtype=float)
    query_array = np.asarray(query, dtype=float)
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    if np.any(query_array < lower) or np.any(query_array > upper):
        return TopologySelection(
            TopologyKind.NONE,
            (),
            SpectrumStatusCode.OUTSIDE_CONVEX_HULL,
        )

    axis_pairs: list[tuple[float, float]] = []
    for dimension, coordinate in enumerate(query):
        values = tuple(sorted(set(points[:, dimension])))
        if len(values) < 2:
            axis_pairs = []
            break
        axis_pairs.append(_bounding_pair(values, coordinate))
    point_index = {point: index for index, point in enumerate(topology.points)}
    if axis_pairs:
        corners = tuple(product(*axis_pairs))
        if all(corner in point_index for corner in corners):
            return TopologySelection(
                TopologyKind.RECTILINEAR,
                tuple(point_index[corner] for corner in corners),
                SpectrumStatusCode.OK,
            )

    for simplex in topology.approved_simplices:
        vertices = points[np.asarray(simplex)]
        if _inside_simplex(vertices, query_array):
            return TopologySelection(
                TopologyKind.SIMPLEX,
                simplex,
                SpectrumStatusCode.OK,
            )
    return TopologySelection(
        TopologyKind.NONE,
        (),
        SpectrumStatusCode.NO_COMPLETE_CELL,
    )


__all__ = [
    "GridTopology",
    "TopologyKind",
    "TopologySelection",
    "select_topology",
]
