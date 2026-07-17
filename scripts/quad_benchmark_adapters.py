"""Thin, traceable adapters for the Phase A4 quadrature benchmark."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import quadax
from scripts.quad_benchmark_cases import (
    BenchmarkCase,
    LibraryMethod,
    PortableScalar,
)

from jaxstro import quad

MethodConfig = tuple[tuple[str, PortableScalar], ...]


@dataclass(frozen=True)
class RunControls:
    epsabs: float
    epsrel: float
    max_regions: int
    max_evaluations: int = 16384


@dataclass(frozen=True)
class MatchedCapacities:
    initial_segments: int
    jaxstro_max_regions: int
    jaxstro_max_evaluations: int
    quadax_max_ninter: int


class RawBenchmarkResult(NamedTuple):
    value: Any
    error: Any
    status: jax.Array
    reported_evaluations: jax.Array
    normalized_evaluations: jax.Array
    refinements: jax.Array
    active_regions: jax.Array
    levels: jax.Array


@dataclass(frozen=True)
class NormalizedResult:
    value: Any
    error: Any
    converged: bool
    raw_status: int
    semantic_status: str
    reported_evaluations: int
    normalized_evaluations: int
    refinements: int
    active_regions: int
    levels: int


_DEFAULT_CONFIGS: dict[str, dict[LibraryMethod, MethodConfig]] = {
    "jaxstro": {
        LibraryMethod.GAUSS_KRONROD: (("pair", 21),),
        LibraryMethod.CLENSHAW_CURTIS: (("initial_order", 17),),
        LibraryMethod.TANH_SINH: (("initial_level", 3),),
        LibraryMethod.ROMBERG: (("initial_level", 1),),
        LibraryMethod.ROMBERG_TANH_SINH: (("initial_level", 1),),
    },
    "quadax": {
        LibraryMethod.GAUSS_KRONROD: (("order", 21),),
        LibraryMethod.CLENSHAW_CURTIS: (("order", 16),),
        LibraryMethod.TANH_SINH: (("order", 61),),
        LibraryMethod.ROMBERG: (("divmax", 10),),
        LibraryMethod.ROMBERG_TANH_SINH: (("divmax", 10),),
    },
}


def _config_dict(
    library: str,
    family: LibraryMethod,
    config: MethodConfig | None,
) -> dict[str, PortableScalar]:
    selected = _DEFAULT_CONFIGS[library][family] if config is None else config
    return dict(selected)


def matched_capacities(
    case: BenchmarkCase,
    family: LibraryMethod,
    *,
    node_cost: int,
    controls: RunControls,
) -> MatchedCapacities:
    """Match regional storage and guarantee the corresponding Jaxstro work cap."""
    del family
    breakpoints = getattr(case.domain, "breakpoints", ())
    initial_segments = len(breakpoints) + 1
    regional_budget = node_cost * (2 * controls.max_regions - initial_segments)
    return MatchedCapacities(
        initial_segments=initial_segments,
        jaxstro_max_regions=controls.max_regions,
        jaxstro_max_evaluations=max(controls.max_evaluations, regional_budget),
        quadax_max_ninter=controls.max_regions,
    )


def normalize_quadax_evaluations(
    family: LibraryMethod,
    reported: int,
    order: int | None,
) -> int:
    """Convert Quadax's Clenshaw-Curtis interval work to actual node calls."""
    if family is LibraryMethod.CLENSHAW_CURTIS:
        if order is None or reported % order:
            raise ValueError("Quadax Clenshaw-Curtis work is not rule-aligned")
        return (order + 1) * (reported // order)
    return reported


def _jaxstro_method(family: LibraryMethod, config: Mapping[str, Any]):
    constructors = {
        LibraryMethod.GAUSS_KRONROD: quad.GaussKronrod,
        LibraryMethod.CLENSHAW_CURTIS: quad.AdaptiveClenshawCurtis,
        LibraryMethod.TANH_SINH: quad.AdaptiveTanhSinh,
        LibraryMethod.ROMBERG: quad.Romberg,
        LibraryMethod.ROMBERG_TANH_SINH: quad.RombergTanhSinh,
    }
    return constructors[family](**config)


def _node_cost(family: LibraryMethod, config: Mapping[str, Any]) -> int:
    if family is LibraryMethod.GAUSS_KRONROD:
        return int(config.get("pair", config.get("order", 21)))
    if family is LibraryMethod.CLENSHAW_CURTIS:
        return int(config.get("initial_order", int(config.get("order", 16)) + 1))
    if family is LibraryMethod.TANH_SINH:
        return int(config.get("order", 61))
    return 1


def raw_jaxstro(
    case: BenchmarkCase,
    family: LibraryMethod,
    controls: RunControls,
    config: MethodConfig | None = None,
) -> Callable[[jax.Array], RawBenchmarkResult]:
    """Build a device-only Jaxstro benchmark callable."""
    options = _config_dict("jaxstro", family, config)
    evaluation_override = options.pop("max_evaluations", None)
    method = _jaxstro_method(family, options)
    capacities = matched_capacities(
        case,
        family,
        node_cost=_node_cost(family, options),
        controls=controls,
    )

    def run(theta: jax.Array) -> RawBenchmarkResult:
        result = quad.integrate(
            case.fun,
            case.domain,
            args=theta,
            method=method,
            epsabs=jnp.asarray(controls.epsabs),
            epsrel=jnp.asarray(controls.epsrel),
            max_evaluations=(
                capacities.jaxstro_max_evaluations
                if evaluation_override is None
                else int(evaluation_override)
            ),
            max_regions=capacities.jaxstro_max_regions,
            error_norm=quad.MaxNorm(),
            gradient="replay",
        )
        return RawBenchmarkResult(
            value=result.value,
            error=result.error.norm,
            status=result.status,
            reported_evaluations=result.work.evaluations,
            normalized_evaluations=result.work.evaluations,
            refinements=result.work.refinements,
            active_regions=result.work.active_regions,
            levels=result.work.levels,
        )

    return run


def _quadax_interval(case: BenchmarkCase) -> jax.Array:
    domain = case.domain
    if isinstance(domain, quad.Interval):
        return jnp.asarray((domain.lower, *domain.breakpoints, domain.upper))
    if isinstance(domain, quad.RightInfinite):
        return jnp.asarray((domain.lower, jnp.inf))
    if isinstance(domain, quad.LeftInfinite):
        return jnp.asarray((-jnp.inf, domain.upper))
    if isinstance(domain, quad.Infinite):
        return jnp.asarray((-jnp.inf, jnp.inf))
    raise TypeError(f"unsupported benchmark domain: {type(domain).__name__}")


def raw_quadax(
    case: BenchmarkCase,
    family: LibraryMethod,
    controls: RunControls,
    config: MethodConfig | None = None,
) -> Callable[[jax.Array], RawBenchmarkResult]:
    """Build a device-only Quadax benchmark callable."""
    options = _config_dict("quadax", family, config)
    interval = _quadax_interval(case)
    regional = family in {
        LibraryMethod.GAUSS_KRONROD,
        LibraryMethod.CLENSHAW_CURTIS,
        LibraryMethod.TANH_SINH,
    }
    functions = {
        LibraryMethod.GAUSS_KRONROD: quadax.quadgk,
        LibraryMethod.CLENSHAW_CURTIS: quadax.quadcc,
        LibraryMethod.TANH_SINH: quadax.quadts,
        LibraryMethod.ROMBERG: quadax.romberg,
        LibraryMethod.ROMBERG_TANH_SINH: quadax.rombergts,
    }
    selected = functions[family]
    order = int(options["order"]) if "order" in options else None
    initial_segments = len(getattr(case.domain, "breakpoints", ())) + 1

    def run(theta: jax.Array) -> RawBenchmarkResult:
        call_options = dict(options)
        if regional:
            call_options["max_ninter"] = controls.max_regions
        value, info = selected(
            case.fun,
            interval,
            args=(theta,),
            full_output=True,
            epsabs=jnp.asarray(controls.epsabs),
            epsrel=jnp.asarray(controls.epsrel),
            norm=jnp.inf,
            **call_options,
        )
        reported = jnp.asarray(info.neval)
        if family is LibraryMethod.CLENSHAW_CURTIS:
            normalized = reported // order * (order + 1)
        else:
            normalized = reported
        if regional:
            active_regions = jnp.asarray(info.info["ninter"])
            refinements = active_regions - initial_segments
        else:
            active_regions = jnp.asarray(-1, dtype=jnp.int32)
            refinements = jnp.asarray(-1, dtype=jnp.int32)
        unavailable = jnp.asarray(-1, dtype=jnp.int32)
        return RawBenchmarkResult(
            value=value,
            error=info.err,
            status=info.status,
            reported_evaluations=reported,
            normalized_evaluations=normalized,
            refinements=refinements,
            active_regions=active_regions,
            levels=unavailable,
        )

    return run


def _semantic_status(library: str, raw_status: int) -> str:
    if library == "jaxstro":
        try:
            return quad.QuadStatus(raw_status).name.lower()
        except ValueError:
            return f"jaxstro_status_{raw_status}"
    if library != "quadax":
        raise ValueError(f"unknown benchmark library: {library}")
    if raw_status == 0:
        return "converged"
    if raw_status == 2:
        return "max_regions"
    return f"quadax_status_{raw_status}"


def normalize_result(
    result: RawBenchmarkResult,
    *,
    library: str,
    family: LibraryMethod,
) -> NormalizedResult:
    """Synchronize a raw result and create the host-only semantic record."""
    del family
    ready = jax.block_until_ready(result)
    raw_status = int(np.asarray(ready.status))
    semantic_status = _semantic_status(library, raw_status)
    return NormalizedResult(
        value=np.asarray(ready.value),
        error=np.asarray(ready.error),
        converged=semantic_status == "converged",
        raw_status=raw_status,
        semantic_status=semantic_status,
        reported_evaluations=int(np.asarray(ready.reported_evaluations)),
        normalized_evaluations=int(np.asarray(ready.normalized_evaluations)),
        refinements=int(np.asarray(ready.refinements)),
        active_regions=int(np.asarray(ready.active_regions)),
        levels=int(np.asarray(ready.levels)),
    )


def portable_numeric(value: Any) -> Any:
    """Encode numeric evidence without emitting JSON-invalid NaN or infinity."""
    array = np.asarray(jax.device_get(value))
    if array.ndim:
        return [portable_numeric(item) for item in array]
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        return bool(scalar)
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    numeric = float(scalar)
    if np.isfinite(numeric):
        return numeric
    classification = "nan"
    if np.isposinf(numeric):
        classification = "posinf"
    elif np.isneginf(numeric):
        classification = "neginf"
    return {"finite": False, "classification": classification}


__all__ = [
    "MatchedCapacities",
    "NormalizedResult",
    "RawBenchmarkResult",
    "RunControls",
    "matched_capacities",
    "normalize_quadax_evaluations",
    "normalize_result",
    "portable_numeric",
    "raw_jaxstro",
    "raw_quadax",
]
