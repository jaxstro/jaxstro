"""Prepared fixed-topology spectral parameter interpolation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import jax
import jax.numpy as jnp

from jaxstro.numerics.checks import try_concrete_bool
from jaxstro.numerics.regular_grid import regular_grid_interp

from .types import (
    Spectrum,
    SpectrumResult,
    SpectrumStatus,
    SpectrumStatusCode,
)


class FluxInterpolation(StrEnum):
    """Validated interpolation spaces for spectral flux values."""

    LINEAR = "linear"
    POSITIVE_LOG = "positive_log"
    AMPLITUDE_SHAPE = "amplitude_shape"


def _require_concrete(predicate: Any, message: str) -> None:
    result = try_concrete_bool(predicate)
    if result is False:
        raise ValueError(message)


def _validate_interpolation(
    interpolation: FluxInterpolation | str,
) -> FluxInterpolation:
    policy = FluxInterpolation(interpolation)
    if policy is FluxInterpolation.AMPLITUDE_SHAPE:
        raise ValueError("amplitude-shape interpolation is not validated")
    return policy


def _validate_vertex_values(values: Any, template: Spectrum) -> Any:
    array = jnp.asarray(values)
    if array.ndim < 2 or array.shape[-1] != template.values.shape[0]:
        raise ValueError("vertex values must end with the template spectral shape")
    _require_concrete(
        jnp.all(jnp.isfinite(array)),
        "vertex values must form a finite complete corner grid",
    )
    return array


def _interpolate_values(
    values: Any,
    weights: Any,
    interpolation: FluxInterpolation,
) -> Any:
    if interpolation is FluxInterpolation.LINEAR:
        return weights @ values
    return jnp.exp(weights @ jnp.log(values))


def _spectrum_result(
    template: Spectrum,
    values: Any,
    inside: Any,
    operation: str,
) -> SpectrumResult:
    output_values = jnp.where(inside, values, jnp.nan)
    provenance = replace(
        template.provenance,
        operations=(*template.provenance.operations, operation),
    )
    spectrum = Spectrum(
        axis=template.axis,
        values=output_values,
        semantic=template.semantic,
        provenance=provenance,
    )
    code = jnp.where(
        inside,
        int(SpectrumStatusCode.OK),
        int(SpectrumStatusCode.OUTSIDE_CONVEX_HULL),
    )
    return SpectrumResult(spectrum=spectrum, status=SpectrumStatus(code))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PreparedRectilinearStencil:
    """A complete static-rank parameter cell with vector spectral payloads."""

    parameter_axes: tuple[Any, ...]
    vertex_values: Any
    template: Spectrum
    interpolation: FluxInterpolation = FluxInterpolation.LINEAR

    def __post_init__(self) -> None:
        axes = tuple(jnp.asarray(axis) for axis in self.parameter_axes)
        if not axes:
            raise ValueError("rectilinear stencils require at least one parameter axis")
        for axis in axes:
            if axis.ndim != 1 or axis.shape[0] < 2:
                raise ValueError(
                    "parameter axes must be one-dimensional with two points"
                )
            _require_concrete(
                jnp.all(jnp.isfinite(axis)) & jnp.all(jnp.diff(axis) > 0.0),
                "parameter axes must be finite and strictly increasing",
            )
        values = _validate_vertex_values(self.vertex_values, self.template)
        expected = tuple(axis.shape[0] for axis in axes)
        if values.shape[:-1] != expected:
            raise ValueError(
                f"vertex values leading shape must match axes; expected {expected}"
            )
        interpolation = _validate_interpolation(self.interpolation)
        if interpolation is FluxInterpolation.POSITIVE_LOG:
            _require_concrete(
                jnp.all(values > 0.0),
                "positive-log interpolation requires strictly positive values",
            )
        object.__setattr__(self, "parameter_axes", axes)
        object.__setattr__(self, "vertex_values", values)
        object.__setattr__(self, "interpolation", interpolation)

    def evaluate(self, point: Any) -> SpectrumResult:
        """Evaluate inside this fixed complete cell without topology search."""
        point = jnp.asarray(point)
        if point.shape != (len(self.parameter_axes),):
            raise ValueError("rectilinear query shape must match parameter rank")
        inside = jnp.asarray(True)
        for axis, coordinate in zip(self.parameter_axes, point, strict=True):
            inside = inside & (coordinate >= axis[0]) & (coordinate <= axis[-1])
        source_values = self.vertex_values
        if self.interpolation is FluxInterpolation.POSITIVE_LOG:
            source_values = jnp.log(source_values)
        values = regular_grid_interp(
            self.parameter_axes,
            source_values,
            point,
            boundary="fill",
            fill_value=jnp.nan,
        )
        if self.interpolation is FluxInterpolation.POSITIVE_LOG:
            values = jnp.exp(values)
        return _spectrum_result(
            self.template,
            values,
            inside,
            f"parameter:rectilinear:{self.interpolation.value}",
        )

    def tree_flatten(self):
        return (
            self.parameter_axes,
            self.vertex_values,
            self.template,
        ), self.interpolation

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        parameter_axes, vertex_values, template = children
        return cls(parameter_axes, vertex_values, template, aux_data)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PreparedSimplexStencil:
    """A fixed simplex with host-selected vertices and JAX barycentric weights."""

    vertices: Any
    vertex_values: Any
    template: Spectrum
    interpolation: FluxInterpolation = FluxInterpolation.LINEAR

    def __post_init__(self) -> None:
        vertices = jnp.asarray(self.vertices)
        if vertices.ndim != 2 or vertices.shape[1] < 1:
            raise ValueError("simplex vertices must have shape (n+1, n)")
        if vertices.shape[0] != vertices.shape[1] + 1:
            raise ValueError("simplex requires exactly dimension plus one vertices")
        _require_concrete(
            jnp.all(jnp.isfinite(vertices)), "simplex vertices must be finite"
        )
        edge_matrix = (vertices[1:] - vertices[0]).T
        _require_concrete(
            jnp.abs(jnp.linalg.det(edge_matrix)) > 0.0,
            "simplex edge matrix must be nonsingular",
        )
        values = _validate_vertex_values(self.vertex_values, self.template)
        if values.shape[0] != vertices.shape[0] or values.ndim != 2:
            raise ValueError("simplex values must match the vertex and spectral shapes")
        interpolation = _validate_interpolation(self.interpolation)
        if interpolation is FluxInterpolation.POSITIVE_LOG:
            _require_concrete(
                jnp.all(values > 0.0),
                "positive-log interpolation requires strictly positive values",
            )
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "vertex_values", values)
        object.__setattr__(self, "interpolation", interpolation)

    def evaluate(self, point: Any) -> SpectrumResult:
        """Evaluate barycentric weights inside this prepared simplex."""
        point = jnp.asarray(point)
        dimension = self.vertices.shape[1]
        if point.shape != (dimension,):
            raise ValueError("simplex query shape must match parameter rank")
        edge_matrix = (self.vertices[1:] - self.vertices[0]).T
        tail_weights = jnp.linalg.solve(edge_matrix, point - self.vertices[0])
        weights = jnp.concatenate(
            [jnp.asarray([1.0 - jnp.sum(tail_weights)]), tail_weights]
        )
        inside = jnp.all(weights >= -1.0e-12) & jnp.all(weights <= 1.0 + 1.0e-12)
        values = _interpolate_values(
            self.vertex_values,
            weights,
            self.interpolation,
        )
        return _spectrum_result(
            self.template,
            values,
            inside,
            f"parameter:simplex:{self.interpolation.value}",
        )

    def tree_flatten(self):
        return (self.vertices, self.vertex_values, self.template), self.interpolation

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        vertices, vertex_values, template = children
        return cls(vertices, vertex_values, template, aux_data)


__all__ = [
    "FluxInterpolation",
    "PreparedRectilinearStencil",
    "PreparedSimplexStencil",
]
