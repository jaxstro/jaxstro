"""Core CGS/SI units for :mod:`jaxstro.quantity`."""

from __future__ import annotations

import math

from jaxstro import constants as C

from . import dimensions as d
from .registry import UnitRegistry
from .unit import Unit, dimensionless

g = Unit("g", 1.0, d.mass, name="gram")
kg = Unit("kg", 1.0e3, d.mass, name="kilogram")

cm = Unit("cm", 1.0, d.length, name="centimeter")
m = Unit("m", 1.0e2, d.length, name="meter")
km = Unit("km", C.KM_CM, d.length, name="kilometer")
nm = Unit("nm", 1.0e-7, d.length, name="nanometer")
micron = Unit("micron", 1.0e-4, d.length, name="micron")
um = micron

s = Unit("s", 1.0, d.time, name="second")
day = Unit("day", 86400.0, d.time, name="day")
yr = Unit("yr", C.YR_S, d.time, name="Julian year")

K = Unit("K", 1.0, d.temperature, name="kelvin")

erg = Unit("erg", 1.0, d.energy, name="erg")
Hz = Unit("Hz", 1.0, d.time**-1, name="hertz")

rad = Unit(
    "rad",
    1.0,
    d.dimensionless,
    name="radian",
    metadata={"semantic": "angle"},
)
deg = Unit(
    "deg",
    math.pi / 180.0,
    d.dimensionless,
    name="degree",
    metadata={"semantic": "angle"},
)

CORE_REGISTRY = UnitRegistry(
    "core",
    units={
        "1": dimensionless,
        "g": g,
        "kg": kg,
        "cm": cm,
        "m": m,
        "km": km,
        "nm": nm,
        "micron": micron,
        "s": s,
        "day": day,
        "yr": yr,
        "K": K,
        "erg": erg,
        "Hz": Hz,
        "rad": rad,
        "deg": deg,
    },
    aliases={"um": "micron"},
)

__all__ = [
    "CORE_REGISTRY",
    "Hz",
    "K",
    "cm",
    "day",
    "deg",
    "dimensionless",
    "erg",
    "g",
    "kg",
    "km",
    "m",
    "micron",
    "nm",
    "rad",
    "s",
    "um",
    "yr",
]
