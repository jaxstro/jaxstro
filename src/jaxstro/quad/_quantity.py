"""Eager quantity normalization for adaptive quadrature only."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from jaxstro.quantity import Quantity, Unit
from jaxstro.quantity import units as q_units
from jaxstro.quantity.errors import DimensionError

from ._integrand import call_integrand, has_explicit_args
from .domains import Infinite, Interval, LeftInfinite, RightInfinite
from .measures import LebesgueMeasure, WeightedMeasure
from .result import QuadError, QuadResult

_RAW_DOMAIN_MESSAGE = "quantity-valued domains are supported only by quad.integrate"


@dataclass(frozen=True)
class NormalizedCall:
    fun: Callable
    domain: Any
    args: Any
    measure: Any
    epsabs: Any
    epsrel: Any
    result_unit: Unit


def domain_coordinates(domain) -> tuple[Any, ...]:
    if isinstance(domain, Interval):
        return (domain.lower, domain.upper, *domain.breakpoints)
    if isinstance(domain, RightInfinite):
        return (domain.lower,)
    if isinstance(domain, LeftInfinite):
        return (domain.upper,)
    if isinstance(domain, Infinite):
        return ()
    raise TypeError(f"unsupported quadrature domain: {type(domain).__name__}")


def domain_has_quantity(domain) -> bool:
    return any(isinstance(value, Quantity) for value in domain_coordinates(domain)) or (
        isinstance(domain, Infinite) and domain.unit is not None
    )


def validate_raw_domain(domain) -> None:
    if domain_has_quantity(domain):
        raise TypeError(_RAW_DOMAIN_MESSAGE)


def quantity_mode(domain, epsabs) -> bool:
    return domain_has_quantity(domain) or isinstance(epsabs, Quantity)


def _coordinate_unit(domain, epsabs) -> Unit:
    coordinates = domain_coordinates(domain)
    quantities = [value for value in coordinates if isinstance(value, Quantity)]
    if quantities:
        if len(quantities) != len(coordinates):
            raise DimensionError(
                "Quantity quadrature coordinates cannot mix raw and dimensional values.",
                operation="quad-coordinate-normalization",
            )
        unit = quantities[0].unit
        for coordinate in quantities[1:]:
            if not coordinate.unit.is_compatible_with(unit):
                raise DimensionError(
                    "Quadrature coordinates must have compatible dimensions.",
                    operation="quad-coordinate-normalization",
                    expected=unit.dimensions,
                    actual=coordinate.unit.dimensions,
                )
        return unit
    if isinstance(domain, Infinite) and domain.unit is not None:
        return domain.unit
    if isinstance(epsabs, Quantity):
        return q_units.dimensionless
    raise TypeError("quantity mode requires a quantity coordinate or quantity epsabs")


def _coordinate_value(value, unit: Unit):
    if isinstance(value, Quantity):
        return value.to_value(unit)
    if not unit.is_dimensionless:
        raise DimensionError(
            "Dimensional quadrature coordinates must all be quantities.",
            operation="quad-coordinate-normalization",
            expected=unit.dimensions,
        )
    return value


def _normalize_domain(domain, unit: Unit):
    if isinstance(domain, Interval):
        return Interval(
            _coordinate_value(domain.lower, unit),
            _coordinate_value(domain.upper, unit),
            breakpoints=tuple(
                _coordinate_value(point, unit) for point in domain.breakpoints
            ),
        )
    if isinstance(domain, RightInfinite):
        return RightInfinite(_coordinate_value(domain.lower, unit))
    if isinstance(domain, LeftInfinite):
        return LeftInfinite(_coordinate_value(domain.upper, unit))
    if isinstance(domain, Infinite):
        return Infinite()
    raise TypeError(f"unsupported quadrature domain: {type(domain).__name__}")


def _require_quantity_output(output, *, context: str) -> Quantity:
    if not isinstance(output, Quantity):
        raise TypeError(f"quantity-mode {context} must return a Quantity")
    return output


def _infer_output_unit(fun, args, coordinate_unit: Unit) -> Unit:
    has_args = has_explicit_args(args)
    node = Quantity(jax.ShapeDtypeStruct((1,), jnp.float64), coordinate_unit)
    output = jax.eval_shape(
        lambda value: call_integrand(fun, value, args, has_args),
        node,
    )
    return _require_quantity_output(output, context="integrand").unit


def _wrap_integrand(fun, args, coordinate_unit: Unit, output_unit: Unit):
    has_args = has_explicit_args(args)

    def evaluate(nodes, live_args):
        output = call_integrand(
            fun,
            Quantity(nodes, coordinate_unit),
            live_args,
            has_args,
        )
        output = _require_quantity_output(output, context="integrand")
        return output.to_value(output_unit)

    if has_args:
        return evaluate

    def evaluate_without_args(nodes):
        return evaluate(nodes, ())

    return evaluate_without_args


def _wrap_measure(measure, args, coordinate_unit: Unit):
    if isinstance(measure, LebesgueMeasure):
        return measure, q_units.dimensionless
    if not isinstance(measure, WeightedMeasure):
        raise TypeError("adaptive quantity quadrature requires a supported measure")
    node = Quantity(jax.ShapeDtypeStruct((1,), jnp.float64), coordinate_unit)
    abstract = jax.eval_shape(lambda value: measure.density(value, args), node)
    density = _require_quantity_output(abstract, context="weighted density")
    if not density.unit.is_compatible_with(measure.density_unit):
        raise DimensionError(
            "Weighted density output is incompatible with density_unit.",
            operation="quad-density-normalization",
            expected=measure.density_unit.dimensions,
            actual=density.unit.dimensions,
        )

    def raw_density(nodes, live_args):
        output = _require_quantity_output(
            measure.density(Quantity(nodes, coordinate_unit), live_args),
            context="weighted density",
        )
        return output.to_value(measure.density_unit)

    return (
        WeightedMeasure(
            raw_density,
            density_unit=measure.density_unit,
            normalized=measure.normalized,
        ),
        measure.density_unit,
    )


def normalize_call(fun, domain, args, measure, epsabs, epsrel) -> NormalizedCall:
    coordinate_unit = _coordinate_unit(domain, epsabs)
    normalized_domain = _normalize_domain(domain, coordinate_unit)
    integrand_unit = _infer_output_unit(fun, args, coordinate_unit)
    normalized_measure, density_unit = _wrap_measure(measure, args, coordinate_unit)
    result_unit = integrand_unit * coordinate_unit * density_unit
    if not isinstance(epsabs, Quantity):
        raise TypeError("quantity mode requires quantity epsabs")
    normalized_epsabs = epsabs.to_value(result_unit)
    if isinstance(epsrel, Quantity):
        normalized_epsrel = epsrel.to_value(q_units.dimensionless)
    else:
        normalized_epsrel = epsrel
    return NormalizedCall(
        fun=_wrap_integrand(fun, args, coordinate_unit, integrand_unit),
        domain=normalized_domain,
        args=args,
        measure=normalized_measure,
        epsabs=normalized_epsabs,
        epsrel=normalized_epsrel,
        result_unit=result_unit,
    )


def restore_result(result: QuadResult, result_unit: Unit) -> QuadResult:
    return QuadResult(
        value=Quantity(result.value, result_unit),
        error=QuadError(
            estimate=Quantity(result.error.estimate, result_unit),
            norm=Quantity(result.error.norm, result_unit),
            kind=result.error.kind,
            confidence_level=result.error.confidence_level,
        ),
        tolerance=Quantity(result.tolerance, result_unit),
        status=result.status,
        work=result.work,
    )


__all__: list[str] = []
