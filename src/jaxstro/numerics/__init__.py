# src/jaxstro/numerics/__init__.py
"""
Numerical utilities for the jaxstro ecosystem.

This subpackage collects small, JAX-native helpers that are broadly
useful across astrophysical codes, but not tied to any specific
domain (N-body, stellar evolution, hydro, etc.).

Submodules
----------
compensated
    Compensated summation (Neumaier) and related helpers.
stats
    Log/exp-stable scalar ops and simple log-likelihood pieces.
interpolation
    1D interpolation and tabulated function helpers.
rootfinding
    Simple, JAX-friendly root-finding routines.
integration
    Lightweight quadrature and cumulative integral helpers.
quadrature
    Gaussian quadrature factory (Gauss-Legendre, Gauss-Hermite, Hermite basis).
regular_grid
    Static-rank regular-grid interpolation helpers.
sampling
    Differentiable inverse-CDF (PPF) sampling primitives.
splines
    JAX-native 1D B-spline basis and evaluation helpers.
checks
    Numerical validation helpers (finiteness, monotonicity, ranges).
linear_algebra
    Small linear algebra convenience utilities.
rng
    PRNG key management helpers for JAX.
"""

from . import (
    checks,
    compensated,
    integration,
    interpolation,
    linear_algebra,
    quadrature,
    regular_grid,
    rng,
    rootfinding,
    sampling,
    splines,
    stats,
)
from .linear_algebra import (
    add_diagonal_jitter,
    correlation_from_covariance,
    correlation_matrix,
    covariance_matrix,
    is_positive_definite,
    positive_definite_jitter,
    qr_solve,
    svd_solve,
    weighted_lstsq,
)
from .quadrature import (
    clenshaw_curtis_nodes,
    gauss_hermite_nodes,
    gauss_laguerre_nodes,
    gauss_legendre_nodes,
    hermite_coefficients,
    hermite_e_basis,
)
from .regular_grid import bilinear_interp, regular_grid_interp, trilinear_interp
from .rootfinding import (
    bisect_many,
    bracket_expand,
    monotone_inverse_interp,
    newton_ppf,
)
from .sampling import inverse_cdf_draw
from .splines import (
    BSpline1D,
    bspline_basis,
    bspline_derivative,
    bspline_design_matrix,
    bspline_eval,
    fit_bspline_lstsq,
    open_uniform_knots,
)
from .types import Array, ScalarFn

__all__ = [
    "Array",
    "ScalarFn",
    "compensated",
    "stats",
    "interpolation",
    "rootfinding",
    "bracket_expand",
    "bisect_many",
    "newton_ppf",
    "monotone_inverse_interp",
    "integration",
    "quadrature",
    "gauss_legendre_nodes",
    "gauss_hermite_nodes",
    "gauss_laguerre_nodes",
    "clenshaw_curtis_nodes",
    "hermite_e_basis",
    "hermite_coefficients",
    "regular_grid",
    "regular_grid_interp",
    "bilinear_interp",
    "trilinear_interp",
    "sampling",
    "inverse_cdf_draw",
    "splines",
    "BSpline1D",
    "bspline_basis",
    "bspline_derivative",
    "bspline_design_matrix",
    "bspline_eval",
    "fit_bspline_lstsq",
    "open_uniform_knots",
    "checks",
    "linear_algebra",
    "weighted_lstsq",
    "qr_solve",
    "svd_solve",
    "covariance_matrix",
    "correlation_from_covariance",
    "correlation_matrix",
    "is_positive_definite",
    "add_diagonal_jitter",
    "positive_definite_jitter",
    "rng",
]
__version__ = "0.1.0"
