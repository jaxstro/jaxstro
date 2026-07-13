"""
jaxstro: core utilities for differentiable astrophysics in JAX.

This package provides shared physical constants, unit systems, and
small utilities used across the jaxstro ecosystem (gravax, startrax,
stellax, nebulax, nucleax, etc.).

The design intent is:
- centralize physical constants and unit definitions,
- keep runtime dependencies minimal,
- avoid any domain-specific simulation logic here.
"""

from . import (
    astrometry,
    atmospheres,
    constants,
    contracts,
    coords,
    geometry,
    numerics,
    params,
    provenance,
    quantity,
    spatial,
    spectra,
    testing,
    units,
)
from .units import DEFAULT as DEFAULT_UNITS

__all__ = [
    "DEFAULT_UNITS",
    "constants",
    "contracts",
    "units",
    "atmospheres",
    "astrometry",
    "numerics",
    "coords",
    "geometry",
    "params",
    "provenance",
    "quantity",
    "spatial",
    "spectra",
    "testing",
]
__version__ = "0.1.0"
