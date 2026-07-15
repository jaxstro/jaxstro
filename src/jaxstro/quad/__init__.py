"""Canonical namespace for current integration foundations and methods."""

from jaxstro.numerics.quadrature import (
    gauss_hermite_nodes,
    gauss_laguerre_nodes,
    gauss_legendre_nodes,
    hermite_coefficients,
    hermite_e_basis,
)

from ._chebyshev import clenshaw_curtis_nodes
from .domains import (
    Infinite,
    Interval,
    LeftInfinite,
    RightInfinite,
    interval_is_valid,
    interval_orientation,
    sorted_breakpoints,
)
from .fixed import fixed
from .measures import (
    JacobiMeasure,
    LaguerreMeasure,
    LebesgueMeasure,
    PhysicistsHermiteMeasure,
    StandardNormalMeasure,
    WeightedMeasure,
)
from .result import ErrorKind, QuadError, QuadResult, QuadStatus, QuadWork
from .rules import (
    ClenshawCurtisRule,
    FejerIRule,
    FejerIIRule,
    GaussianRule,
    TanhSinhRule,
)
from .sampled import cumulative_simpson, cumulative_trapezoid, simpson, trapezoid
from .tolerance import (
    ErrorNorm,
    L1Norm,
    L2Norm,
    MaxNorm,
    error_norm,
    tolerance_threshold,
)
from .transforms import AffineMapResult, DomainMapResult, map_domain, map_interval

__all__ = [
    "AffineMapResult",
    "clenshaw_curtis_nodes",
    "ClenshawCurtisRule",
    "cumulative_simpson",
    "cumulative_trapezoid",
    "DomainMapResult",
    "ErrorKind",
    "ErrorNorm",
    "FejerIRule",
    "FejerIIRule",
    "fixed",
    "Infinite",
    "Interval",
    "JacobiMeasure",
    "LaguerreMeasure",
    "LebesgueMeasure",
    "LeftInfinite",
    "gauss_hermite_nodes",
    "gauss_laguerre_nodes",
    "gauss_legendre_nodes",
    "GaussianRule",
    "hermite_coefficients",
    "hermite_e_basis",
    "L1Norm",
    "L2Norm",
    "MaxNorm",
    "PhysicistsHermiteMeasure",
    "QuadError",
    "QuadResult",
    "QuadStatus",
    "QuadWork",
    "RightInfinite",
    "StandardNormalMeasure",
    "TanhSinhRule",
    "WeightedMeasure",
    "error_norm",
    "interval_is_valid",
    "interval_orientation",
    "map_interval",
    "map_domain",
    "simpson",
    "sorted_breakpoints",
    "trapezoid",
    "tolerance_threshold",
]
