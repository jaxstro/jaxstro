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

__all__ = [
    "clenshaw_curtis_nodes",
    "cumulative_simpson",
    "cumulative_trapezoid",
    "gauss_hermite_nodes",
    "gauss_laguerre_nodes",
    "gauss_legendre_nodes",
    "hermite_coefficients",
    "hermite_e_basis",
    "simpson",
    "trapezoid",
]
