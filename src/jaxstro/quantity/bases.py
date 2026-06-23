"""Role-aware quantity unit bases."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import dimensions as d
from . import units
from .astro import AU, Lsun, Msun, Rsun, pc
from .errors import UnitRegistryError
from .unit import Unit


@dataclass(frozen=True)
class UnitPreference:
    """Preferred unit for a named physical role."""

    role: str
    unit: Unit
    description: str = ""


@dataclass(frozen=True)
class UnitBasis:
    """Presentation/conversion profile for quantities."""

    name: str
    roles: Mapping[str, UnitPreference] = field(default_factory=dict)
    dimensions: Mapping[d.Dimension, Unit] = field(default_factory=dict)

    def unit_for(self, quantity: Any = None, *, role: str | None = None) -> Unit:
        if role is not None:
            if role in self.roles:
                return self.roles[role].unit
            suggestions = tuple(
                difflib.get_close_matches(role, sorted(self.roles), n=3)
            )
            message = f"Unknown role {role!r} for basis {self.name!r}."
            if suggestions:
                message += f" Did you mean {', '.join(suggestions)}?"
            raise UnitRegistryError(
                message,
                operation="basis-role",
                expected=suggestions,
                actual=role,
            )
        dimension = _dimension_of(quantity)
        if dimension in self.dimensions:
            return self.dimensions[dimension]
        raise UnitRegistryError(
            f"No default unit for dimension {dimension!r} in basis {self.name!r}.",
            operation="basis-dimension",
            expected=tuple(self.dimensions),
            actual=dimension,
        )


def _dimension_of(value: Any) -> d.Dimension:
    from .quantity import Quantity

    if isinstance(value, Quantity):
        return value.unit.dimensions
    if isinstance(value, Unit):
        return value.dimensions
    if isinstance(value, d.Dimension):
        return value
    raise UnitRegistryError(
        "Basis lookup requires a Quantity, Unit, Dimension, or explicit role.",
        operation="basis-dimension",
        actual=type(value).__name__,
    )


def _prefs(**roles: Unit) -> dict[str, UnitPreference]:
    return {role: UnitPreference(role, unit) for role, unit in roles.items()}


CGS = UnitBasis(
    "CGS",
    roles=_prefs(
        stellar_mass=units.g,
        planet_mass=units.g,
        stellar_radius=units.cm,
        orbital_separation=units.cm,
        orbit=units.cm,
        velocity=units.cm / units.s,
        luminosity=units.erg / units.s,
        time=units.s,
    ),
    dimensions={
        d.mass: units.g,
        d.length: units.cm,
        d.time: units.s,
        d.temperature: units.K,
        d.energy: units.erg,
        d.power: units.erg / units.s,
    },
)

SI = UnitBasis(
    "SI",
    roles=_prefs(
        stellar_mass=units.kg,
        planet_mass=units.kg,
        stellar_radius=units.m,
        orbital_separation=units.m,
        orbit=units.m,
        velocity=units.m / units.s,
        luminosity=units.erg / units.s,
        time=units.s,
    ),
    dimensions={d.mass: units.kg, d.length: units.m, d.time: units.s},
)

STELLAR = UnitBasis(
    "STELLAR",
    roles=_prefs(
        stellar_mass=Msun,
        planet_mass=units.g,
        stellar_radius=Rsun,
        orbital_separation=Rsun,
        orbit=Rsun,
        velocity=units.km / units.s,
        luminosity=Lsun,
        time=units.Myr,
    ),
    dimensions={d.mass: Msun, d.length: Rsun, d.time: units.Myr, d.power: Lsun},
)

PLANETARY = UnitBasis(
    "PLANETARY",
    roles=_prefs(
        stellar_mass=Msun,
        planet_mass=units.g,
        stellar_radius=Rsun,
        orbital_separation=AU,
        orbit=AU,
        velocity=units.km / units.s,
        luminosity=Lsun,
        time=units.yr,
    ),
    dimensions={d.mass: Msun, d.length: AU, d.time: units.yr, d.power: Lsun},
)

STAR_CLUSTER = UnitBasis(
    "STAR_CLUSTER",
    roles=_prefs(
        stellar_mass=Msun,
        planet_mass=units.g,
        stellar_radius=Rsun,
        orbital_separation=pc,
        orbit=pc,
        velocity=units.km / units.s,
        luminosity=Lsun,
        time=units.Myr,
    ),
    dimensions={d.mass: Msun, d.length: pc, d.time: units.Myr, d.power: Lsun},
)

CLOSE_BINARY = PLANETARY
WIDE_BINARY = STAR_CLUSTER
COMPACT_BINARY = UnitBasis(
    "COMPACT_BINARY",
    roles=_prefs(
        stellar_mass=Msun,
        planet_mass=units.g,
        stellar_radius=units.km,
        orbital_separation=units.km,
        orbit=units.km,
        velocity=units.km / units.s,
        luminosity=Lsun,
        time=units.s,
    ),
    dimensions={d.mass: Msun, d.length: units.km, d.time: units.s, d.power: Lsun},
)

DYNAMICAL = STAR_CLUSTER
EXOPLANET = PLANETARY

__all__ = [
    "CGS",
    "CLOSE_BINARY",
    "COMPACT_BINARY",
    "DYNAMICAL",
    "EXOPLANET",
    "PLANETARY",
    "SI",
    "STAR_CLUSTER",
    "STELLAR",
    "UnitBasis",
    "UnitPreference",
    "WIDE_BINARY",
]
