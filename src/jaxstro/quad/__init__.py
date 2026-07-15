"""Canonical namespace for current integration foundations and methods."""

from jaxstro.numerics.integration import cumulative_simpson, simpson
from jaxstro.numerics.integration import (
    cumulative_trapz as cumulative_trapezoid,
)
from jaxstro.numerics.integration import trapz as trapezoid
from jaxstro.numerics.quadrature import (
    clenshaw_curtis_nodes,
    gauss_hermite_nodes,
    gauss_laguerre_nodes,
    gauss_legendre_nodes,
    hermite_coefficients,
    hermite_e_basis,
)

from .result import ErrorKind, QuadError, QuadResult, QuadStatus, QuadWork
from .tolerance import (
    ErrorNorm,
    L1Norm,
    L2Norm,
    MaxNorm,
    error_norm,
    tolerance_threshold,
)

__all__ = [
    "clenshaw_curtis_nodes",
    "cumulative_simpson",
    "cumulative_trapezoid",
    "ErrorKind",
    "ErrorNorm",
    "gauss_hermite_nodes",
    "gauss_laguerre_nodes",
    "gauss_legendre_nodes",
    "hermite_coefficients",
    "hermite_e_basis",
    "L1Norm",
    "L2Norm",
    "MaxNorm",
    "QuadError",
    "QuadResult",
    "QuadStatus",
    "QuadWork",
    "error_norm",
    "simpson",
    "trapezoid",
    "tolerance_threshold",
]
