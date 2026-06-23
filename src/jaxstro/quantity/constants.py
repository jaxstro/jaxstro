"""Versioned constants as :class:`jaxstro.quantity.Quantity` objects."""

from __future__ import annotations

from dataclasses import dataclass

from jaxstro import constants as C

from . import units
from .astro import AU as AU_unit
from .astro import Lsun as Lsun_unit
from .astro import Msun as Msun_unit
from .astro import Rsun as Rsun_unit
from .quantity import Quantity


@dataclass(frozen=True)
class ConstantMetadata:
    """Provenance metadata for a quantity constant."""

    key: str
    name: str
    source: str
    version: str
    citation: str
    url: str
    accessed: str
    checked_against: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class ConstantSet:
    """Description of a named constant value set."""

    name: str
    description: str


default_set = ConstantSet(
    "jaxstro-legacy-compatible",
    "Quantity constants mirror jaxstro.constants for backwards-compatible CGS "
    "values. Fundamental exact SI constants and G were checked against CODATA "
    "2022 on 2026-06-23; selected exposed values remain unchanged.",
)

G = Quantity(C.G_CGS, units.cm**3 / units.g / units.s**2)
c = Quantity(C.C_CGS, units.cm / units.s)
h = Quantity(C.H_CGS, units.erg * units.s)
k_B = Quantity(C.K_B, units.erg / units.K)
sigma_sb = Quantity(C.SIGMA_SB, units.erg / units.s / units.cm**2 / units.K**4)

Msun = Quantity(1.0, Msun_unit)
Rsun = Quantity(1.0, Rsun_unit)
Lsun = Quantity(1.0, Lsun_unit)
AU = Quantity(1.0, AU_unit)

_METADATA = {
    "G": ConstantMetadata(
        "G",
        "Newtonian constant of gravitation",
        "CODATA",
        "2018",
        "Tiesinga et al. 2021, Rev. Mod. Phys. 93, 025010",
        "https://doi.org/10.1103/RevModPhys.93.025010",
        "2026-06-23",
        checked_against="CODATA 2022",
        notes="Kept aligned with jaxstro.constants.G_CGS.",
    ),
    "c": ConstantMetadata(
        "c",
        "speed of light in vacuum",
        "CODATA",
        "2018",
        "Tiesinga et al. 2021, Rev. Mod. Phys. 93, 025010",
        "https://doi.org/10.1103/RevModPhys.93.025010",
        "2026-06-23",
        checked_against="CODATA 2022",
        notes="Exact SI constant converted to CGS.",
    ),
    "h": ConstantMetadata(
        "h",
        "Planck constant",
        "CODATA",
        "2018",
        "Tiesinga et al. 2021, Rev. Mod. Phys. 93, 025010",
        "https://doi.org/10.1103/RevModPhys.93.025010",
        "2026-06-23",
        checked_against="CODATA 2022",
        notes="Exact SI constant converted to CGS.",
    ),
    "k_B": ConstantMetadata(
        "k_B",
        "Boltzmann constant",
        "CODATA",
        "2018",
        "Tiesinga et al. 2021, Rev. Mod. Phys. 93, 025010",
        "https://doi.org/10.1103/RevModPhys.93.025010",
        "2026-06-23",
        checked_against="CODATA 2022",
        notes="Exact SI constant converted to CGS.",
    ),
    "sigma_sb": ConstantMetadata(
        "sigma_sb",
        "Stefan-Boltzmann constant",
        "CODATA",
        "2018",
        "Tiesinga et al. 2021, Rev. Mod. Phys. 93, 025010",
        "https://doi.org/10.1103/RevModPhys.93.025010",
        "2026-06-23",
        checked_against="CODATA 2022",
        notes="Kept aligned with jaxstro.constants.SIGMA_SB.",
    ),
    "Msun": ConstantMetadata(
        "Msun",
        "nominal solar mass",
        "IAU",
        "2015 Resolution B3",
        "IAU 2015 Resolution B3",
        "https://www.iau.org/static/resolutions/IAU2015_English.pdf",
        "2026-06-23",
    ),
    "Rsun": ConstantMetadata(
        "Rsun",
        "nominal solar radius",
        "IAU",
        "2015 Resolution B3",
        "IAU 2015 Resolution B3",
        "https://www.iau.org/static/resolutions/IAU2015_English.pdf",
        "2026-06-23",
    ),
    "Lsun": ConstantMetadata(
        "Lsun",
        "nominal solar luminosity",
        "IAU",
        "2015 Resolution B3",
        "IAU 2015 Resolution B3",
        "https://www.iau.org/static/resolutions/IAU2015_English.pdf",
        "2026-06-23",
    ),
    "AU": ConstantMetadata(
        "AU",
        "astronomical unit",
        "IAU",
        "2012 Resolution B2",
        "IAU 2012 Resolution B2",
        "https://www.iau.org/static/resolutions/IAU2012_English.pdf",
        "2026-06-23",
    ),
}

_CONSTANTS = {
    "G": G,
    "c": c,
    "h": h,
    "k_B": k_B,
    "sigma_sb": sigma_sb,
    "Msun": Msun,
    "Rsun": Rsun,
    "Lsun": Lsun,
    "AU": AU,
}


def metadata(key: str) -> ConstantMetadata:
    """Return provenance metadata for a constant key."""

    return _METADATA[key]


def raw_value_cgs(key: str):
    """Return the raw CGS value for a quantity constant."""

    return _CONSTANTS[key].to_cgs_value()


__all__ = [
    "AU",
    "ConstantMetadata",
    "ConstantSet",
    "G",
    "Lsun",
    "Msun",
    "Rsun",
    "c",
    "default_set",
    "h",
    "k_B",
    "metadata",
    "raw_value_cgs",
    "sigma_sb",
]
