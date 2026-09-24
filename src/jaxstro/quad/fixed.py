"""Public one-dimensional fixed-quadrature evaluator."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from ._chebyshev import chebyshev_rule_data
from ._integrand import (
    call_integrand,
    density_values,
    has_explicit_args,
    node_weighted_sum,
    validate_node_values,
)
from ._quantity import validate_raw_domain
from ._recurrence import gaussian_rule_data
from ._tanh_sinh import tanh_sinh_rule_data
from .domains import (
    Infinite,
    Interval,
    LeftInfinite,
    RightInfinite,
    interval_is_valid,
    sorted_breakpoints,
)
from .measures import (
    JacobiMeasure,
    LaguerreMeasure,
    LebesgueMeasure,
    PhysicistsHermiteMeasure,
    StandardNormalMeasure,
    WeightedMeasure,
)
from .rules import (
    ClenshawCurtisRule,
    FejerIIRule,
    FejerIRule,
    FixedRuleData,
    GaussianRule,
    TanhSinhRule,
)
from .transforms import DomainMapResult, map_domain, map_interval_replay

Domain = Interval | RightInfinite | LeftInfinite | Infinite
Measure = (
    LebesgueMeasure
    | WeightedMeasure
    | JacobiMeasure
    | LaguerreMeasure
    | PhysicistsHermiteMeasure
    | StandardNormalMeasure
)
Rule = GaussianRule | ClenshawCurtisRule | FejerIRule | FejerIIRule | TanhSinhRule


def _evaluate_mapped(
    fun: Callable,
    args: Any,
    has_args: bool,
    data: FixedRuleData,
    mapped: DomainMapResult,
    measure: Measure,
    zero_width: Any = False,
):
    values = validate_node_values(
        call_integrand(fun, mapped.x, args, has_args),
        data.nodes.shape[0],
        context="fixed quadrature",
    )
    # A zero-width segment has a zero Jacobian: its value is exactly zero, and
    # its bound derivative is the integrand value there. Sanitize only the
    # nonfinite values so 0 * inf cannot poison the result.
    values = jnp.where(zero_width & ~jnp.isfinite(values), 0.0, values)
    density = density_values(measure, mapped.x, args)
    weights = data.weights * mapped.jacobian * density
    value = mapped.orientation * node_weighted_sum(values, weights)
    return jnp.where(mapped.valid, value, jnp.full_like(value, jnp.nan))


def _interval_segments(domain: Interval):
    points = sorted_breakpoints(domain)
    endpoints = jnp.concatenate(
        (
            jnp.asarray(domain.lower)[None],
            points,
            jnp.asarray(domain.upper)[None],
        )
    )
    return jnp.stack((endpoints[:-1], endpoints[1:]), axis=-1)


def _evaluate_interval_segments(
    fun: Callable,
    domain: Interval,
    args: Any,
    has_args: bool,
    data: FixedRuleData,
    measure: Measure,
):
    def evaluate_segment(bounds):
        segment = Interval(bounds[0], bounds[1])
        # The signed affine map keeps d/d(bound) = +-f(bound) at zero width.
        mapped = DomainMapResult(*map_interval_replay(segment, data.nodes))
        return _evaluate_mapped(
            fun,
            args,
            has_args,
            data,
            mapped,
            measure,
            zero_width=bounds[0] == bounds[1],
        )

    values = jax.vmap(evaluate_segment)(_interval_segments(domain))
    value = jnp.sum(values, axis=0)
    valid = interval_is_valid(domain)
    return jnp.where(valid, value, jnp.full_like(value, jnp.nan))


def _gaussian_fixed(
    fun: Callable,
    domain: Domain,
    args: Any,
    has_args: bool,
    rule: GaussianRule,
    measure: Measure,
):
    data = gaussian_rule_data(rule, measure)
    if isinstance(measure, (LebesgueMeasure, JacobiMeasure)):
        if not isinstance(domain, Interval):
            raise TypeError(
                "Gaussian Legendre and Jacobi rules require a finite Interval"
            )
        if isinstance(measure, JacobiMeasure) and domain.breakpoints:
            raise TypeError("JacobiMeasure does not support breakpoints")
        return _evaluate_interval_segments(fun, domain, args, has_args, data, measure)
    if isinstance(measure, LaguerreMeasure):
        if not isinstance(domain, RightInfinite):
            raise TypeError("Gaussian Laguerre rules require RightInfinite")
        lower = jnp.asarray(domain.lower)
        mapped = DomainMapResult(
            x=lower + data.nodes,
            jacobian=jnp.ones_like(data.nodes),
            orientation=jnp.asarray(1.0),
            valid=jnp.isfinite(lower),
        )
        return _evaluate_mapped(fun, args, has_args, data, mapped, measure)
    if isinstance(measure, (PhysicistsHermiteMeasure, StandardNormalMeasure)):
        if not isinstance(domain, Infinite):
            raise TypeError("Gaussian Hermite rules require Infinite")
        mapped = DomainMapResult(
            x=data.nodes,
            jacobian=jnp.ones_like(data.nodes),
            orientation=jnp.asarray(1.0),
            valid=jnp.asarray(True),
        )
        return _evaluate_mapped(fun, args, has_args, data, mapped, measure)
    raise TypeError("GaussianRule requires a supported classical measure")


def fixed(
    fun: Callable,
    domain: Domain,
    *,
    args: Any = (),
    rule: Rule,
    measure: Measure | None = None,
):
    """Evaluate a declared one-dimensional fixed quadrature formula."""
    validate_raw_domain(domain)
    selected_measure: Measure = LebesgueMeasure() if measure is None else measure
    has_args = has_explicit_args(args)
    if isinstance(rule, GaussianRule):
        return _gaussian_fixed(fun, domain, args, has_args, rule, selected_measure)
    if isinstance(rule, (ClenshawCurtisRule, FejerIRule, FejerIIRule)):
        if not isinstance(domain, Interval):
            raise TypeError("Chebyshev rules require a finite Interval")
        if not isinstance(selected_measure, (LebesgueMeasure, WeightedMeasure)):
            raise TypeError(
                "Chebyshev rules require LebesgueMeasure or WeightedMeasure"
            )
        data = chebyshev_rule_data(rule)
        return _evaluate_interval_segments(
            fun, domain, args, has_args, data, selected_measure
        )
    if isinstance(rule, TanhSinhRule):
        if not isinstance(selected_measure, (LebesgueMeasure, WeightedMeasure)):
            raise TypeError("TanhSinhRule requires LebesgueMeasure or WeightedMeasure")
        data = tanh_sinh_rule_data(rule)
        if isinstance(domain, Interval):
            return _evaluate_interval_segments(
                fun, domain, args, has_args, data, selected_measure
            )
        mapped = map_domain(domain, data.nodes)
        return _evaluate_mapped(fun, args, has_args, data, mapped, selected_measure)
    raise TypeError("unsupported fixed quadrature rule")


__all__ = ["fixed"]
