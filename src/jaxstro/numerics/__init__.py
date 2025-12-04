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
checks
    Numerical validation helpers (finiteness, monotonicity, ranges).
linear_algebra
    Small linear algebra convenience utilities.
rng
    PRNG key management helpers for JAX.
"""

from .types import Array, ScalarFn
from . import compensated
from . import stats
from . import interpolation
from . import rootfinding
from . import integration
from . import checks
from . import linear_algebra
from . import rng

__all__ = [
    "Array",
    "ScalarFn",
    "compensated",
    "stats",
    "interpolation",
    "rootfinding",
    "integration",
    "checks",
    "linear_algebra",
    "rng",
]
__version__ = "0.1.0"