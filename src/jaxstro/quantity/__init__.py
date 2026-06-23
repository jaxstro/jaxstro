"""JAX-aware physical quantities and units."""

from . import astro, dimensions, units
from .astro import AU, Lsun, Msun, Rsun, pc
from .dimensions import (
    Dimension,
    acceleration,
    amount,
    current,
    energy,
    length,
    luminosity,
    mass,
    power,
    temperature,
    time,
    velocity,
)
from .errors import (
    DimensionError,
    EquivalencyError,
    QuantityError,
    UnitConversionError,
    UnitParseError,
    UnitRegistryError,
)
from .parser import format_unit, parse_unit
from .quantity import Quantity
from .registry import UnitRegistry
from .serialization import from_dict, to_dict, unit_from_dict, unit_to_dict
from .unit import Unit
from .units import (
    Hz,
    K,
    cm,
    day,
    deg,
    dimensionless,
    erg,
    g,
    kg,
    km,
    m,
    micron,
    nm,
    rad,
    s,
    um,
    yr,
)

DEFAULT_REGISTRY = astro.ASTRO_REGISTRY
GLOBAL_REGISTRY = UnitRegistry("global", parent=DEFAULT_REGISTRY, mutable=True)


def get_unit(symbol: str, *, registry: UnitRegistry | None = None) -> Unit:
    """Look up a unit by exact symbol in the selected registry."""

    return (registry or GLOBAL_REGISTRY).lookup(symbol)


def register_global_unit(unit: Unit, *, aliases: tuple[str, ...] = ()) -> None:
    """Register a unit globally for interactive sessions and notebooks."""

    GLOBAL_REGISTRY.register(unit, aliases=aliases)


__all__ = [
    "AU",
    "Dimension",
    "DimensionError",
    "EquivalencyError",
    "Hz",
    "K",
    "Lsun",
    "Msun",
    "Quantity",
    "QuantityError",
    "Rsun",
    "DEFAULT_REGISTRY",
    "GLOBAL_REGISTRY",
    "Unit",
    "UnitConversionError",
    "UnitParseError",
    "UnitRegistryError",
    "UnitRegistry",
    "acceleration",
    "amount",
    "cm",
    "current",
    "day",
    "deg",
    "dimensionless",
    "dimensions",
    "energy",
    "erg",
    "g",
    "kg",
    "km",
    "length",
    "luminosity",
    "m",
    "mass",
    "micron",
    "nm",
    "pc",
    "power",
    "rad",
    "s",
    "temperature",
    "time",
    "um",
    "velocity",
    "yr",
    "get_unit",
    "format_unit",
    "parse_unit",
    "register_global_unit",
    "from_dict",
    "to_dict",
    "unit_from_dict",
    "unit_to_dict",
    "astro",
    "units",
]
