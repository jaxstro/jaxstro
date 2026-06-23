"""Astronomical units for :mod:`jaxstro.quantity`."""

from __future__ import annotations

from jaxstro import constants as C

from . import dimensions as d
from .registry import UnitRegistry
from .unit import Unit
from .units import CORE_REGISTRY

AU = Unit("AU", C.AU_CM, d.length, name="astronomical unit")
pc = Unit("pc", C.PC_CM, d.length, name="parsec")
Msun = Unit("Msun", C.MSUN_G, d.mass, name="nominal solar mass")
Rsun = Unit("Rsun", C.RSUN_CM, d.length, name="nominal solar radius")
Lsun = Unit("Lsun", C.LSUN_ERG_S, d.power, name="nominal solar luminosity")

ASTRO_REGISTRY = UnitRegistry(
    "astro",
    units={
        "AU": AU,
        "pc": pc,
        "Msun": Msun,
        "Rsun": Rsun,
        "Lsun": Lsun,
    },
    aliases={"msun": "Msun"},
    parent=CORE_REGISTRY,
)

__all__ = ["ASTRO_REGISTRY", "AU", "Lsun", "Msun", "Rsun", "pc"]
