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

from importlib import import_module
from types import ModuleType

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


def __getattr__(name: str) -> ModuleType:
    """Load public submodules on first attribute access."""
    if name in __all__ and name != "DEFAULT_UNITS":
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
