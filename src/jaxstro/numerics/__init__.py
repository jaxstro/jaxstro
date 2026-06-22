# src/jaxstro/numerics/__init__.py
"""
Numerical utilities for the jaxstro ecosystem.

This subpackage collects small, JAX-native helpers that are broadly
useful across astrophysical codes, but not tied to any specific
domain (N-body, stellar evolution, hydro, etc.).

Submodules
----------
autodiff
    JVP, VJP, HVP, and curvature-product helpers.
compensated
    Compensated summation (Neumaier) and related helpers.
distributions
    Generic logpdf, CDF, and inverse-CDF kernels.
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
grids
    Grid construction and conservative binning helpers.
regular_grid
    Static-rank regular-grid interpolation helpers.
special
    Stable special-function kernels and polynomial bases.
sampling
    Differentiable inverse-CDF (PPF) sampling primitives.
splines
    JAX-native 1D B-spline basis and evaluation helpers.
checks
    Numerical validation helpers (finiteness, monotonicity, ranges).
linear_algebra
    Small linear algebra convenience utilities.
optimization
    Loss, line-search, and convergence helpers.
ode
    Fixed-step ODE integration helpers.
operators
    Small PyTree linear operators.
rng
    PRNG key management helpers for JAX.
"""

from . import (
    autodiff,
    checks,
    compensated,
    distributions,
    grids,
    integration,
    interpolation,
    linear_algebra,
    ode,
    operators,
    optimization,
    quadrature,
    regular_grid,
    rng,
    rootfinding,
    sampling,
    special,
    splines,
    stats,
)
from .autodiff import (
    empirical_fisher_product,
    gauss_newton_product,
    hvp,
    jacobian_vector_product,
    jvp,
    vector_jacobian_product,
    vjp,
)
from .grids import (
    bin_centers,
    conservative_rebin,
    geometric_bin_centers,
    geometric_bin_edges,
    log_grid,
)
from .distributions import (
    lognormal_cdf,
    lognormal_logpdf,
    lognormal_ppf,
    normal_cdf,
    normal_logpdf,
    normal_ppf,
    powerlaw_cdf,
    powerlaw_logpdf,
    powerlaw_ppf,
    truncated_normal_cdf,
    truncated_normal_logpdf,
    truncated_normal_ppf,
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
from .optimization import (
    LineSearchResult,
    armijo_backtracking,
    convergence_summary,
    gradient_inf_norm,
    huber_loss,
    objective_summary,
    pseudo_huber_loss,
    relative_step_norm,
    squared_loss,
)
from .ode import (
    ODEResult,
    VerletResult,
    euler,
    euler_step,
    midpoint,
    midpoint_step,
    rk4,
    rk4_step,
    solve_fixed_step,
    velocity_verlet,
)
from .operators import (
    BlockDiagonalOperator,
    DenseOperator,
    DiagonalOperator,
    LinearOperator,
    ProductOperator,
    ScaledOperator,
    SumOperator,
    TransposeOperator,
    add,
    block_diag,
    compose,
    scale,
    transpose,
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
from .sampling import inverse_cdf_draw, stratified_uniform
from .special import (
    chebyshev_t_basis,
    laguerre_basis,
    legendre_basis,
    log_normalize,
    log_planck_lambda_cgs,
    log_planck_nu_cgs,
    normalize_log_weights,
    planck_lambda_cgs,
    planck_nu_cgs,
)
from .splines import (
    BSpline1D,
    adaptive_open_uniform_knots,
    bspline_antiderivative,
    bspline_basis,
    bspline_derivative,
    bspline_design_matrix,
    bspline_eval,
    bspline_eval_deboor,
    bspline_integral,
    bspline_roughness_penalty,
    fit_bspline_lstsq,
    open_uniform_knots,
    tensor_product_design_matrix,
)
from .types import Array, ScalarFn

__all__ = [
    "Array",
    "ScalarFn",
    "autodiff",
    "compensated",
    "distributions",
    "stats",
    "interpolation",
    "rootfinding",
    "bracket_expand",
    "bisect_many",
    "newton_ppf",
    "monotone_inverse_interp",
    "integration",
    "jvp",
    "vjp",
    "jacobian_vector_product",
    "vector_jacobian_product",
    "hvp",
    "gauss_newton_product",
    "empirical_fisher_product",
    "quadrature",
    "grids",
    "log_grid",
    "geometric_bin_edges",
    "bin_centers",
    "geometric_bin_centers",
    "conservative_rebin",
    "normal_logpdf",
    "normal_cdf",
    "normal_ppf",
    "lognormal_logpdf",
    "lognormal_cdf",
    "lognormal_ppf",
    "powerlaw_logpdf",
    "powerlaw_cdf",
    "powerlaw_ppf",
    "truncated_normal_logpdf",
    "truncated_normal_cdf",
    "truncated_normal_ppf",
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
    "stratified_uniform",
    "special",
    "planck_lambda_cgs",
    "log_planck_lambda_cgs",
    "planck_nu_cgs",
    "log_planck_nu_cgs",
    "log_normalize",
    "normalize_log_weights",
    "legendre_basis",
    "chebyshev_t_basis",
    "laguerre_basis",
    "splines",
    "BSpline1D",
    "adaptive_open_uniform_knots",
    "bspline_antiderivative",
    "bspline_basis",
    "bspline_derivative",
    "bspline_design_matrix",
    "bspline_eval",
    "bspline_eval_deboor",
    "bspline_integral",
    "bspline_roughness_penalty",
    "fit_bspline_lstsq",
    "open_uniform_knots",
    "tensor_product_design_matrix",
    "checks",
    "linear_algebra",
    "optimization",
    "ode",
    "operators",
    "weighted_lstsq",
    "qr_solve",
    "svd_solve",
    "covariance_matrix",
    "correlation_from_covariance",
    "correlation_matrix",
    "is_positive_definite",
    "add_diagonal_jitter",
    "positive_definite_jitter",
    "LineSearchResult",
    "squared_loss",
    "huber_loss",
    "pseudo_huber_loss",
    "objective_summary",
    "armijo_backtracking",
    "gradient_inf_norm",
    "relative_step_norm",
    "convergence_summary",
    "ODEResult",
    "VerletResult",
    "euler_step",
    "midpoint_step",
    "rk4_step",
    "euler",
    "midpoint",
    "rk4",
    "solve_fixed_step",
    "velocity_verlet",
    "LinearOperator",
    "DenseOperator",
    "DiagonalOperator",
    "ScaledOperator",
    "SumOperator",
    "ProductOperator",
    "TransposeOperator",
    "BlockDiagonalOperator",
    "scale",
    "add",
    "compose",
    "transpose",
    "block_diag",
    "rng",
]
__version__ = "0.1.0"
