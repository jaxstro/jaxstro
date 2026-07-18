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
from .coordinates import CoordinatePoint
from .domains import Hyperrectangle, Infinite, Interval, LeftInfinite, RightInfinite
from .measures import LebesgueMeasure, ProductMeasure, WeightedMeasure
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


@dataclass(frozen=True)
class NormalizedMultidimCall:
    fun: Callable
    domain: Hyperrectangle
    args: Any
    method: Any
    measure: Any
    epsabs: Any
    epsrel: Any
    result_unit: Unit


def domain_coordinates(domain) -> tuple[Any, ...]:
    if isinstance(domain, Hyperrectangle):
        return (domain.lower, domain.upper)
    if isinstance(domain, Interval):
        return (domain.lower, domain.upper, *domain.breakpoints)
    if isinstance(domain, RightInfinite):
        return (domain.lower,) + (() if domain.scale is None else (domain.scale,))
    if isinstance(domain, LeftInfinite):
        return (domain.upper,) + (() if domain.scale is None else (domain.scale,))
    if isinstance(domain, Infinite):
        return () if domain.scale is None else (domain.scale,)
    raise TypeError(f"unsupported quadrature domain: {type(domain).__name__}")


def domain_has_quantity(domain) -> bool:
    if isinstance(domain, Hyperrectangle) and domain.axis_units is not None:
        return True
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
    if isinstance(domain, Infinite) and domain.unit is not None:
        for coordinate in quantities:
            if not coordinate.unit.is_compatible_with(domain.unit):
                raise DimensionError(
                    "Infinite-domain scale must match the declared coordinate unit.",
                    operation="quad-coordinate-normalization",
                    expected=domain.unit.dimensions,
                    actual=coordinate.unit.dimensions,
                )
        return domain.unit
    if quantities:
        if len(quantities) != len(coordinates):
            raise DimensionError(
                "Quantity quadrature coordinates cannot mix raw and dimensional values.",
                operation="quad-coordinate-normalization",
            )
        unit = quantities[0].unit
        for coordinate in quantities[1:]:
            if not coordinate.unit.is_compatible_with(unit):
                if (
                    isinstance(domain, (RightInfinite, LeftInfinite, Infinite))
                    and coordinate is domain.scale
                ):
                    raise DimensionError(
                        "Improper-domain scale must match the coordinate unit.",
                        operation="quad-coordinate-normalization",
                        expected=unit.dimensions,
                        actual=coordinate.unit.dimensions,
                    )
                raise DimensionError(
                    "Quadrature coordinates must have compatible dimensions.",
                    operation="quad-coordinate-normalization",
                    expected=unit.dimensions,
                    actual=coordinate.unit.dimensions,
                )
        return unit
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
        return RightInfinite(
            _coordinate_value(domain.lower, unit),
            scale=(
                None if domain.scale is None else _coordinate_value(domain.scale, unit)
            ),
        )
    if isinstance(domain, LeftInfinite):
        return LeftInfinite(
            _coordinate_value(domain.upper, unit),
            scale=(
                None if domain.scale is None else _coordinate_value(domain.scale, unit)
            ),
        )
    if isinstance(domain, Infinite):
        return Infinite(
            scale=(
                None if domain.scale is None else _coordinate_value(domain.scale, unit)
            )
        )
    raise TypeError(f"unsupported quadrature domain: {type(domain).__name__}")


def _multidim_axis_units(domain: Hyperrectangle) -> tuple[Unit, ...]:
    if domain.axis_units is not None:
        return domain.axis_units
    return (q_units.dimensionless,) * domain.dimension


def _multidim_coordinate_unit_product(units: tuple[Unit, ...]) -> Unit:
    product = q_units.dimensionless
    for unit in units:
        product = product * unit
    return product


def _require_quantity_output(output, *, context: str) -> Quantity:
    if not isinstance(output, Quantity):
        raise TypeError(f"quantity-mode {context} must return a Quantity")
    return output


def _infer_output_unit(fun, args, coordinate_unit: Unit) -> Unit:
    has_args = has_explicit_args(args)
    node = Quantity(jax.ShapeDtypeStruct((1,), jnp.float64), coordinate_unit)
    if has_args:
        output = jax.eval_shape(
            lambda value, live_args: call_integrand(fun, value, live_args, has_args),
            node,
            args,
        )
    else:
        output = jax.eval_shape(
            lambda value: call_integrand(fun, value, (), has_args),
            node,
        )
    return _require_quantity_output(output, context="integrand").unit


def _infer_multidim_output_unit(
    fun,
    args,
    units: tuple[Unit, ...],
) -> Unit:
    has_args = has_explicit_args(args)
    point = CoordinatePoint(
        jax.ShapeDtypeStruct((1, len(units)), jnp.float64),
        units,
    )
    if has_args:
        output = jax.eval_shape(
            lambda value, live_args: call_integrand(
                fun,
                value,
                live_args,
                has_args,
            ),
            point,
            args,
        )
    else:
        output = jax.eval_shape(
            lambda value: call_integrand(fun, value, (), has_args),
            point,
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


def _wrap_multidim_integrand(
    fun,
    args,
    units: tuple[Unit, ...],
    output_unit: Unit,
):
    has_args = has_explicit_args(args)

    def evaluate(nodes, live_args):
        output = call_integrand(
            fun,
            CoordinatePoint(nodes, units),
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
    abstract = jax.eval_shape(
        lambda value, live_args: measure.density(value, live_args),
        node,
        args,
    )
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


def _validate_density_unit(output, declared: Unit) -> None:
    density = _require_quantity_output(output, context="weighted density")
    if not density.unit.is_compatible_with(declared):
        raise DimensionError(
            "Weighted density output is incompatible with density_unit.",
            operation="quad-density-normalization",
            expected=declared.dimensions,
            actual=density.unit.dimensions,
        )


def _wrap_multidim_measure(
    measure,
    args,
    units: tuple[Unit, ...],
):
    if isinstance(measure, LebesgueMeasure):
        return measure, q_units.dimensionless
    if isinstance(measure, WeightedMeasure):
        point = CoordinatePoint(
            jax.ShapeDtypeStruct((1, len(units)), jnp.float64),
            units,
        )
        abstract = jax.eval_shape(
            lambda value, live_args: measure.density(value, live_args),
            point,
            args,
        )
        _validate_density_unit(abstract, measure.density_unit)

        def raw_density(nodes, live_args):
            output = _require_quantity_output(
                measure.density(CoordinatePoint(nodes, units), live_args),
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
    if isinstance(measure, ProductMeasure):
        if len(measure.components) != len(units):
            raise ValueError(
                "ProductMeasure requires one component per coordinate axis"
            )
        components: list[LebesgueMeasure | WeightedMeasure] = []
        density_unit = q_units.dimensionless
        for component, unit in zip(measure.components, units, strict=True):
            if isinstance(component, LebesgueMeasure):
                components.append(component)
                continue
            node = Quantity(jax.ShapeDtypeStruct((1,), jnp.float64), unit)
            abstract = jax.eval_shape(
                lambda value, live_args, density=component.density: density(
                    value,
                    live_args,
                ),
                node,
                args,
            )
            _validate_density_unit(abstract, component.density_unit)

            def raw_component_density(
                values,
                live_args,
                *,
                density=component.density,
                axis_unit=unit,
                declared_unit=component.density_unit,
            ):
                output = _require_quantity_output(
                    density(Quantity(values, axis_unit), live_args),
                    context="weighted density",
                )
                return output.to_value(declared_unit)

            components.append(
                WeightedMeasure(
                    raw_component_density,
                    density_unit=component.density_unit,
                    normalized=component.normalized,
                )
            )
            density_unit = density_unit * component.density_unit
        return ProductMeasure(tuple(components)), density_unit
    raise TypeError(
        "multidimensional quantity quadrature requires a finite supported measure"
    )


def normalize_call(fun, domain, args, measure, epsabs, epsrel) -> NormalizedCall:
    coordinate_unit = _coordinate_unit(domain, epsabs)
    if (
        isinstance(domain, (RightInfinite, LeftInfinite, Infinite))
        and not coordinate_unit.is_dimensionless
        and not isinstance(domain.scale, Quantity)
    ):
        raise TypeError(
            "dimensional improper quadrature requires an explicit Quantity scale"
        )
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


def _normalize_rqmc_quantity_bounds(method, *, integrand_unit, result_unit):
    from .qmc import AdaptiveScrambledSobol

    if not isinstance(method, AdaptiveScrambledSobol):
        return method
    selected = (
        method.estimate_bounds
        if method.estimate_bounds is not None
        else method.integrand_bounds
    )
    assert selected is not None
    if not all(isinstance(bound, Quantity) for bound in selected):
        raise TypeError(
            "quantity AdaptiveScrambledSobol bounds must be Quantity values"
        )
    target_unit = (
        result_unit if method.estimate_bounds is not None else integrand_unit
    )
    normalized_bounds = tuple(
        float(jnp.asarray(bound.to_value(target_unit))) for bound in selected
    )
    options = {
        "schedule": method.schedule,
        "scramble": method.scramble,
        "confidence_level": method.confidence_level,
    }
    if method.estimate_bounds is not None:
        options["estimate_bounds"] = normalized_bounds
    else:
        options["integrand_bounds"] = normalized_bounds
    return AdaptiveScrambledSobol(**options)


def normalize_multidim_call(
    fun,
    domain: Hyperrectangle,
    args,
    method,
    measure,
    epsabs,
    epsrel,
) -> NormalizedMultidimCall:
    """Normalize heterogeneous quantity axes onto the raw Phase B engine."""
    units = _multidim_axis_units(domain)
    integrand_unit = _infer_multidim_output_unit(fun, args, units)
    normalized_measure, density_unit = _wrap_multidim_measure(
        measure,
        args,
        units,
    )
    coordinate_unit = _multidim_coordinate_unit_product(units)
    result_unit = integrand_unit * coordinate_unit * density_unit
    if not isinstance(epsabs, Quantity):
        raise TypeError("quantity mode requires quantity epsabs")
    normalized_epsabs = epsabs.to_value(result_unit)
    if isinstance(epsrel, Quantity):
        normalized_epsrel = epsrel.to_value(q_units.dimensionless)
    else:
        normalized_epsrel = epsrel
    normalized_method = _normalize_rqmc_quantity_bounds(
        method,
        integrand_unit=integrand_unit,
        result_unit=result_unit,
    )
    return NormalizedMultidimCall(
        fun=_wrap_multidim_integrand(fun, args, units, integrand_unit),
        domain=Hyperrectangle(domain.lower, domain.upper),
        args=args,
        method=normalized_method,
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
