"""Frozen scientific cases and fairness labels for the Phase A4 benchmark."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np

from jaxstro.quad import Infinite, Interval, LeftInfinite, RightInfinite

Domain = Interval | RightInfinite | LeftInfinite | Infinite
PortableScalar = str | int | float | bool


class ComparisonLabel(str, Enum):
    EXACT = "exact"
    STRONG_MATCH = "strong_match"
    NODE_MATCHED = "node_matched"
    FAMILY_MATCHED = "family_matched"
    CAPABILITY = "capability"


class LibraryMethod(str, Enum):
    GAUSS_KRONROD = "gauss_kronrod"
    CLENSHAW_CURTIS = "clenshaw_curtis"
    TANH_SINH = "tanh_sinh"
    ROMBERG = "romberg"
    ROMBERG_TANH_SINH = "romberg_tanh_sinh"


@dataclass(frozen=True)
class IndependentReference:
    orders: tuple[int, ...]
    values: tuple[float, ...]
    convergence_delta: float
    numpy_version: str


@dataclass(frozen=True)
class TruthProvenance:
    kind: str
    expression: str
    source: str
    reference_version: str
    atol: float
    rtol: float
    reference_orders: tuple[int, ...] = ()
    reference_values: tuple[float, ...] = ()
    convergence_delta: float | None = None
    analytic_crosscheck: float | tuple[float, ...] | None = None


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    family: str
    fun: Callable[[jax.Array, jax.Array], jax.Array]
    domain: Domain
    truth: float | tuple[float, ...] | None
    derivative_truth: float | tuple[float, ...] | None
    truth_provenance: TruthProvenance
    theta: float
    expected: str
    supported_methods: tuple[LibraryMethod, ...]
    reference_fun: Callable[[np.ndarray, float], np.ndarray] | None = None


@dataclass(frozen=True)
class MethodPair:
    family: LibraryMethod
    variant: str
    label: ComparisonLabel
    jaxstro_config: tuple[tuple[str, PortableScalar], ...]
    quadax_config: tuple[tuple[str, PortableScalar], ...]
    note: str


@dataclass(frozen=True)
class BestMethodChoice:
    case: str
    jaxstro_method: LibraryMethod
    jaxstro_config: tuple[tuple[str, PortableScalar], ...]
    quadax_method: LibraryMethod
    quadax_config: tuple[tuple[str, PortableScalar], ...]
    rationale: str
    source: str


def _smooth_exponential(x, theta):
    return jnp.exp(theta * x)


def _vector_polynomial_exponential(x, theta):
    return jnp.stack((x**2, jnp.exp(theta * x)), axis=-1)


def _localized_gaussian(x, theta):
    return jnp.exp(-400.0 * (x - theta) ** 2)


def _breakpoint_kink(x, theta):
    return jnp.abs(x - theta)


def _endpoint_sqrt(x, _theta):
    return jnp.sqrt(x)


def _semi_infinite_exponential(x, theta):
    return jnp.exp(-theta * x)


def _full_line_gaussian(x, theta):
    return jnp.exp(-((x / theta) ** 2))


def _oscillatory_cosine(x, theta):
    return jnp.cos(theta * x)


def _expensive_identity(x, theta):
    phases = jnp.arange(8, dtype=jnp.asarray(x).dtype)
    shifted = jnp.expand_dims(x, axis=-1) + phases
    identities = jnp.sin(shifted) ** 2 + jnp.cos(shifted) ** 2
    return jnp.exp(-theta * x) * jnp.mean(identities, axis=-1)


def _narrow_gaussian(x, _theta):
    return jnp.exp(-10000.0 * (x - 0.501) ** 2)


def _nonfinite_band(x, _theta):
    return jnp.where(jnp.abs(x - 0.5) < 0.05, jnp.nan, jnp.exp(-x))


def _numpy_expensive_identity(x: np.ndarray, theta: float) -> np.ndarray:
    phases = np.arange(8, dtype=np.float64)
    shifted = x[..., None] + phases
    identities = np.sin(shifted) ** 2 + np.cos(shifted) ** 2
    return np.exp(-theta * x) * np.mean(identities, axis=-1)


def _numpy_narrow_gaussian(x: np.ndarray, _theta: float) -> np.ndarray:
    return np.exp(-10000.0 * (x - 0.501) ** 2)


def _finite_legendre_reference(
    fun: Callable[[np.ndarray, float], np.ndarray],
    theta: float,
) -> IndependentReference:
    orders = (256, 512, 1024)
    values = []
    for order in orders:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        x = 0.5 * (nodes + 1.0)
        values.append(float(0.5 * np.sum(weights * fun(x, theta))))
    return IndependentReference(
        orders=orders,
        values=tuple(values),
        convergence_delta=abs(values[-1] - values[-2]),
        numpy_version=np.__version__,
    )


_EXPENSIVE_REFERENCE = _finite_legendre_reference(
    _numpy_expensive_identity,
    1.0,
)
_NARROW_REFERENCE = _finite_legendre_reference(
    _numpy_narrow_gaussian,
    1.0,
)


def independent_gauss_legendre_reference(
    case: BenchmarkCase,
) -> IndependentReference:
    """Recompute a host NumPy reference independently of both JAX adapters."""
    if case.reference_fun is None or not isinstance(case.domain, Interval):
        raise ValueError(f"case has no independent finite reference: {case.name}")
    if case.domain.lower != 0.0 or case.domain.upper != 1.0:
        raise ValueError("independent reference currently requires Interval(0, 1)")
    return _finite_legendre_reference(case.reference_fun, case.theta)


def _analytic(
    expression: str,
    *,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-10,
) -> TruthProvenance:
    return TruthProvenance(
        kind="analytic",
        expression=expression,
        source="analytic definition in the Phase A4 benchmark catalog",
        reference_version="not_applicable",
        atol=atol,
        rtol=rtol,
    )


def _reference(
    expression: str,
    reference: IndependentReference,
    analytic_crosscheck: float,
) -> TruthProvenance:
    return TruthProvenance(
        kind="reference",
        expression=expression,
        source="independent NumPy Gauss-Legendre orders 256, 512, and 1024",
        reference_version=reference.numpy_version,
        atol=1.0e-12,
        rtol=1.0e-10,
        reference_orders=reference.orders,
        reference_values=reference.values,
        convergence_delta=reference.convergence_delta,
        analytic_crosscheck=analytic_crosscheck,
    )


_ALL = tuple(LibraryMethod)
_REGIONAL = (
    LibraryMethod.GAUSS_KRONROD,
    LibraryMethod.CLENSHAW_CURTIS,
    LibraryMethod.TANH_SINH,
)
_IMPROPER = (
    LibraryMethod.TANH_SINH,
    LibraryMethod.ROMBERG_TANH_SINH,
)

_LOCALIZED_CENTER = 0.37
_LOCALIZED_TRUTH = (
    math.sqrt(math.pi)
    / 40.0
    * (math.erf(20.0 * (1.0 - _LOCALIZED_CENTER)) + math.erf(20.0 * _LOCALIZED_CENTER))
)
_NARROW_CENTER = 0.501
_NARROW_ANALYTIC = (
    math.sqrt(math.pi)
    / 200.0
    * (math.erf(100.0 * (1.0 - _NARROW_CENTER)) + math.erf(100.0 * _NARROW_CENTER))
)

CASES = (
    BenchmarkCase(
        "smooth_exponential",
        "smooth_finite",
        _smooth_exponential,
        Interval(0.0, 1.0),
        math.e - 1.0,
        1.0,
        _analytic(r"(e^\theta-1)/\theta"),
        1.0,
        "converged",
        _ALL,
    ),
    BenchmarkCase(
        "vector_polynomial_exponential",
        "vector_output",
        _vector_polynomial_exponential,
        Interval(0.0, 1.0),
        (1.0 / 3.0, math.e - 1.0),
        (0.0, 1.0),
        _analytic(r"[1/3,(e^\theta-1)/\theta]"),
        1.0,
        "converged",
        _ALL,
    ),
    BenchmarkCase(
        "localized_gaussian",
        "localized_feature",
        _localized_gaussian,
        Interval(0.0, 1.0),
        _LOCALIZED_TRUTH,
        None,
        _analytic(r"\int_0^1 e^{-400(x-\theta)^2}\,dx"),
        _LOCALIZED_CENTER,
        "converged",
        _REGIONAL,
    ),
    BenchmarkCase(
        "breakpoint_kink",
        "explicit_breakpoint",
        _breakpoint_kink,
        Interval(0.0, 1.0, breakpoints=(0.3,)),
        0.29,
        -0.4,
        _analytic(r"[\theta^2+(1-\theta)^2]/2"),
        0.3,
        "converged",
        _REGIONAL,
    ),
    BenchmarkCase(
        "endpoint_sqrt",
        "endpoint_singularity",
        _endpoint_sqrt,
        Interval(0.0, 1.0),
        2.0 / 3.0,
        None,
        _analytic(r"\int_0^1\sqrt{x}\,dx=2/3"),
        1.0,
        "converged",
        _IMPROPER,
    ),
    BenchmarkCase(
        "semi_infinite_exponential",
        "semi_infinite",
        _semi_infinite_exponential,
        RightInfinite(0.0, scale=1.0),
        1.0,
        -1.0,
        _analytic(r"\int_0^\infty e^{-\theta x}\,dx=1/\theta"),
        1.0,
        "converged",
        _IMPROPER,
    ),
    BenchmarkCase(
        "full_line_gaussian",
        "full_infinite",
        _full_line_gaussian,
        Infinite(scale=1.0),
        math.sqrt(math.pi),
        math.sqrt(math.pi),
        _analytic(r"\int_{-\infty}^{\infty}e^{-(x/\theta)^2}\,dx"),
        1.0,
        "converged",
        _IMPROPER,
    ),
    BenchmarkCase(
        "oscillatory_cosine",
        "oscillatory",
        _oscillatory_cosine,
        Interval(0.0, 1.0),
        math.sin(50.0) / 50.0,
        (50.0 * math.cos(50.0) - math.sin(50.0)) / 50.0**2,
        _analytic(r"\int_0^1\cos(\theta x)\,dx=\sin(\theta)/\theta"),
        50.0,
        "converged",
        (
            LibraryMethod.GAUSS_KRONROD,
            LibraryMethod.CLENSHAW_CURTIS,
            LibraryMethod.ROMBERG,
        ),
    ),
    BenchmarkCase(
        "expensive_identity",
        "expensive_integrand",
        _expensive_identity,
        Interval(0.0, 1.0),
        _EXPENSIVE_REFERENCE.values[-1],
        2.0 / math.e - 1.0,
        _reference(
            r"e^{-\theta x}\langle\sin^2(x+k)+\cos^2(x+k)\rangle_k",
            _EXPENSIVE_REFERENCE,
            1.0 - math.exp(-1.0),
        ),
        1.0,
        "converged",
        _ALL,
        _numpy_expensive_identity,
    ),
    BenchmarkCase(
        "narrow_gaussian",
        "missed_feature",
        _narrow_gaussian,
        Interval(0.0, 1.0),
        _NARROW_REFERENCE.values[-1],
        None,
        _reference(
            r"e^{-10000(x-0.501)^2}",
            _NARROW_REFERENCE,
            _NARROW_ANALYTIC,
        ),
        1.0,
        "converged_or_exposed",
        _REGIONAL,
        _numpy_narrow_gaussian,
    ),
    BenchmarkCase(
        "nonfinite_band",
        "nonfinite",
        _nonfinite_band,
        Interval(0.0, 1.0),
        None,
        None,
        _analytic(r"\mathrm{NaN}\;\mathrm{for}\;|x-0.5|<0.05"),
        1.0,
        "fail_closed",
        (LibraryMethod.GAUSS_KRONROD,),
    ),
)

METHOD_PAIRS = (
    MethodPair(
        LibraryMethod.GAUSS_KRONROD,
        "pair21",
        ComparisonLabel.EXACT,
        (("pair", 21),),
        (("order", 21),),
        "Same embedded Gauss-Kronrod pair on finite domains.",
    ),
    MethodPair(
        LibraryMethod.CLENSHAW_CURTIS,
        "nodes17",
        ComparisonLabel.NODE_MATCHED,
        (("initial_order", 17),),
        (("order", 16),),
        "Both local rules evaluate 17 nodes; estimators differ.",
    ),
    MethodPair(
        LibraryMethod.TANH_SINH,
        "closest_work",
        ComparisonLabel.FAMILY_MATCHED,
        (("initial_level", 2),),
        (("order", 61),),
        "Closest declared local node budget.",
    ),
    MethodPair(
        LibraryMethod.TANH_SINH,
        "native_default",
        ComparisonLabel.FAMILY_MATCHED,
        (("initial_level", 3),),
        (("order", 61),),
        "Each library's native tanh-sinh default.",
    ),
    MethodPair(
        LibraryMethod.ROMBERG,
        "divmax10",
        ComparisonLabel.STRONG_MATCH,
        (("initial_level", 1), ("max_evaluations", 1025)),
        (("divmax", 10),),
        "Global trapezoid and Richardson capacity match.",
    ),
    MethodPair(
        LibraryMethod.ROMBERG_TANH_SINH,
        "divmax10",
        ComparisonLabel.CAPABILITY,
        (("initial_level", 1),),
        (("divmax", 10),),
        "Quadax adds Richardson extrapolation; Jaxstro does not.",
    ),
)


def _best(
    case: str,
    method: LibraryMethod,
    jaxstro_config: tuple[tuple[str, PortableScalar], ...],
    quadax_config: tuple[tuple[str, PortableScalar], ...],
    rationale: str,
) -> BestMethodChoice:
    return BestMethodChoice(
        case=case,
        jaxstro_method=method,
        jaxstro_config=jaxstro_config,
        quadax_method=method,
        quadax_config=quadax_config,
        rationale=rationale,
        source="frozen from the approved Phase A4 design and Quadax 0.2.13 guidance",
    )


BEST_METHODS = {
    "smooth_exponential": _best(
        "smooth_exponential",
        LibraryMethod.GAUSS_KRONROD,
        (("pair", 21),),
        (("order", 21),),
        "Embedded Gauss-Kronrod is efficient for a smooth finite integrand.",
    ),
    "vector_polynomial_exponential": _best(
        "vector_polynomial_exponential",
        LibraryMethod.GAUSS_KRONROD,
        (("pair", 21),),
        (("order", 21),),
        "Gauss-Kronrod supports the smooth vector payload directly.",
    ),
    "localized_gaussian": _best(
        "localized_gaussian",
        LibraryMethod.CLENSHAW_CURTIS,
        (("initial_order", 17),),
        (("order", 16),),
        "Regional nested refinement exposes localized structure.",
    ),
    "breakpoint_kink": _best(
        "breakpoint_kink",
        LibraryMethod.CLENSHAW_CURTIS,
        (("initial_order", 17),),
        (("order", 16),),
        "The explicit breakpoint isolates the nonsmooth point.",
    ),
    "endpoint_sqrt": _best(
        "endpoint_sqrt",
        LibraryMethod.TANH_SINH,
        (("initial_level", 3),),
        (("order", 61),),
        "Double-exponential nodes target endpoint singular behavior.",
    ),
    "semi_infinite_exponential": _best(
        "semi_infinite_exponential",
        LibraryMethod.TANH_SINH,
        (("initial_level", 3),),
        (("order", 61),),
        "Tanh-sinh supports the declared semi-infinite transform.",
    ),
    "full_line_gaussian": _best(
        "full_line_gaussian",
        LibraryMethod.ROMBERG_TANH_SINH,
        (("initial_level", 1),),
        (("divmax", 10),),
        "Global tanh-sinh refinement supports the full line.",
    ),
    "oscillatory_cosine": _best(
        "oscillatory_cosine",
        LibraryMethod.GAUSS_KRONROD,
        (("pair", 21),),
        (("order", 21),),
        "The shipped general-purpose pair is used before specialist methods exist.",
    ),
    "expensive_identity": _best(
        "expensive_identity",
        LibraryMethod.GAUSS_KRONROD,
        (("pair", 21),),
        (("order", 21),),
        "A high-order local rule amortizes the expensive smooth payload.",
    ),
    "narrow_gaussian": _best(
        "narrow_gaussian",
        LibraryMethod.CLENSHAW_CURTIS,
        (("initial_order", 17),),
        (("order", 16),),
        "The case audits adaptive discovery rather than only local exactness.",
    ),
    "nonfinite_band": _best(
        "nonfinite_band",
        LibraryMethod.GAUSS_KRONROD,
        (("pair", 21),),
        (("order", 21),),
        "The exact-family lane exposes the libraries' different failure semantics.",
    ),
}


__all__ = [
    "BEST_METHODS",
    "CASES",
    "METHOD_PAIRS",
    "BenchmarkCase",
    "BestMethodChoice",
    "ComparisonLabel",
    "IndependentReference",
    "LibraryMethod",
    "MethodPair",
    "TruthProvenance",
    "independent_gauss_legendre_reference",
]
