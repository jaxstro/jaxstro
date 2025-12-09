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

from . import constants
from . import units
from . import astrometry
from . import numerics
from . import coords

__all__ = ["constants", "units", "astrometry", "numerics", "coords"]
__version__ = "0.1.0"
