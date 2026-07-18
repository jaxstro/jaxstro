"""Canonical namespace for current integration foundations and methods."""

from jaxstro.numerics.quadrature import (
    gauss_hermite_nodes,
    hermite_coefficients,
    hermite_e_basis,
)

from ._chebyshev import clenshaw_curtis_nodes
from ._recurrence import gauss_laguerre_nodes, gauss_legendre_nodes
from .coordinates import Axis, CoordinatePoint
from .cubature import AdaptiveCubature, GenzMalik
from .domains import (
    Hyperrectangle,
    Infinite,
    Interval,
    LeftInfinite,
    RightInfinite,
    hyperrectangle_is_valid,
    hyperrectangle_orientation,
    interval_is_valid,
    interval_orientation,
    sorted_breakpoints,
)
from .fixed import fixed
from .integrate import integrate
from .measures import (
    JacobiMeasure,
    LaguerreMeasure,
    LebesgueMeasure,
    PhysicistsHermiteMeasure,
    ProductMeasure,
    StandardNormalMeasure,
    WeightedMeasure,
)
from .methods import (
    AdaptiveClenshawCurtis,
    AdaptiveTanhSinh,
    GaussKronrod,
    Romberg,
    RombergTanhSinh,
)
from .qmc import (
    AdaptiveScrambledSobol,
    DigitalShift,
    LinearMatrixScramble,
    OwenScramble,
    ScrambledSobol,
    Sobol,
)
from .result import ErrorKind, QuadError, QuadResult, QuadStatus, QuadWork
from .rules import (
    ClenshawCurtisRule,
    FejerIIRule,
    FejerIRule,
    GaussianRule,
    TanhSinhRule,
)
from .sampled import cumulative_simpson, cumulative_trapezoid, simpson, trapezoid
from .sparse import AdaptiveSmolyak, Smolyak
from .tensor import AdaptiveTensorClenshawCurtis, TensorProduct
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
    "AdaptiveClenshawCurtis",
    "AdaptiveCubature",
    "AdaptiveSmolyak",
    "AdaptiveScrambledSobol",
    "AdaptiveTensorClenshawCurtis",
    "AdaptiveTanhSinh",
    "AffineMapResult",
    "Axis",
    "clenshaw_curtis_nodes",
    "ClenshawCurtisRule",
    "CoordinatePoint",
    "cumulative_simpson",
    "cumulative_trapezoid",
    "DomainMapResult",
    "DigitalShift",
    "ErrorKind",
    "ErrorNorm",
    "FejerIRule",
    "FejerIIRule",
    "fixed",
    "Hyperrectangle",
    "hyperrectangle_is_valid",
    "hyperrectangle_orientation",
    "Infinite",
    "integrate",
    "Interval",
    "JacobiMeasure",
    "LaguerreMeasure",
    "LebesgueMeasure",
    "LeftInfinite",
    "LinearMatrixScramble",
    "gauss_hermite_nodes",
    "gauss_laguerre_nodes",
    "gauss_legendre_nodes",
    "GaussianRule",
    "GaussKronrod",
    "GenzMalik",
    "hermite_coefficients",
    "hermite_e_basis",
    "L1Norm",
    "L2Norm",
    "MaxNorm",
    "PhysicistsHermiteMeasure",
    "ProductMeasure",
    "OwenScramble",
    "QuadError",
    "QuadResult",
    "QuadStatus",
    "QuadWork",
    "RightInfinite",
    "Romberg",
    "RombergTanhSinh",
    "ScrambledSobol",
    "StandardNormalMeasure",
    "Smolyak",
    "TanhSinhRule",
    "TensorProduct",
    "WeightedMeasure",
    "error_norm",
    "interval_is_valid",
    "interval_orientation",
    "map_interval",
    "map_domain",
    "simpson",
    "Sobol",
    "sorted_breakpoints",
    "trapezoid",
    "tolerance_threshold",
]
