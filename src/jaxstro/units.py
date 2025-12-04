# src/jaxstro/units.py

"""
Unit systems for the jaxstro ecosystem.

This module defines generic physical unit systems that encode
mass/length/time scales in CGS, together with a small UnitSystem
dataclass you can reuse in downstream packages.

Key design choices:
    - UnitSystem encodes *physical scales*, not simulation semantics.
    - No global context or code-unit state is managed here.
    - Downstream packages (gravax, startrax, nebulax, ...) are free
      to define their own "code units" built on these base systems.

Example
-------
>>> from jaxstro import units, constants as C
>>> us = units.ASTRO_STELLAR
>>> m_cgs = us.mass_scale_cgs         # 1 Msun in g
>>> r_cgs = us.length_scale_cgs       # 1 Rsun in cm
>>> t_cgs = us.time_scale_cgs         # 1 Myr in s
"""

from dataclasses import dataclass
from typing import Tuple

from . import constants as C


@dataclass(frozen=True)
class UnitSystem:
    """
    Physical unit system defined by base mass/length/time scales in CGS.

    This is intentionally lightweight and domain-agnostic. It does
    not know about gravitational constants, N-body code units, or
    any particular simulation code; it just ties together scales.

    Attributes
    ----------
    name : str
        Descriptive name for the unit system.
    mass_unit : str
        Label for the mass unit (e.g. "g", "Msun").
    length_unit : str
        Label for the length unit (e.g. "cm", "Rsun", "pc", "AU").
    time_unit : str
        Label for the time unit (e.g. "s", "yr", "Myr").
    mass_scale_cgs : float
        Value of 1 [mass_unit] in grams.
    length_scale_cgs : float
        Value of 1 [length_unit] in centimetres.
    time_scale_cgs : float
        Value of 1 [time_unit] in seconds.
    description : str
        Free-form description / intended use.
    """

    name: str
    mass_unit: str
    length_unit: str
    time_unit: str
    mass_scale_cgs: float
    length_scale_cgs: float
    time_scale_cgs: float
    description: str = ""

    # Derived scales / helpers
    @property
    def velocity_scale_cgs(self) -> float:
        """
        Velocity scale in cm/s corresponding to 1 [length_unit] / 1 [time_unit].

        Example: for (pc, Myr), this is the speed corresponding to
        1 pc / 1 Myr in cm/s.
        """
        return self.length_scale_cgs / self.time_scale_cgs

    @property
    def velocity_scale_km_s(self) -> float:
        """
        Velocity scale in km/s corresponding to 1 [length_unit] / 1 [time_unit].
        """
        return self.velocity_scale_cgs / C.KM_CM

    def to_cgs(
        self,
        mass: float,
        length: float,
        time: float,
    ) -> Tuple[float, float, float]:
        """
        Convert dimensionless (code) mass, length, time to CGS.

        Parameters
        ----------
        mass : float
            Value in [mass_unit].
        length : float
            Value in [length_unit].
        time : float
            Value in [time_unit].

        Returns
        -------
        m_cgs, r_cgs, t_cgs : tuple of float
            Values in g, cm, s.
        """
        return (
            mass * self.mass_scale_cgs,
            length * self.length_scale_cgs,
            time * self.time_scale_cgs,
        )

    def from_cgs(
        self,
        mass_cgs: float,
        length_cgs: float,
        time_cgs: float,
    ) -> Tuple[float, float, float]:
        """
        Convert CGS mass, length, time to this unit system.

        Parameters
        ----------
        mass_cgs : float
            Mass in grams.
        length_cgs : float
            Length in centimetres.
        time_cgs : float
            Time in seconds.

        Returns
        -------
        m, r, t : tuple of float
            Values in [mass_unit], [length_unit], [time_unit].
        """
        return (
            mass_cgs / self.mass_scale_cgs,
            length_cgs / self.length_scale_cgs,
            time_cgs / self.time_scale_cgs,
        )

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.name} "
            f"([{self.mass_unit}], [{self.length_unit}], [{self.time_unit}])"
        )


# ---------------------------------------------------------------------------
# Canonical physical unit systems
# ---------------------------------------------------------------------------

# Base CGS system (g, cm, s)
CGS = UnitSystem(
    name="CGS",
    mass_unit="g",
    length_unit="cm",
    time_unit="s",
    mass_scale_cgs=1.0,
    length_scale_cgs=1.0,
    time_scale_cgs=1.0,
    description="Base CGS units (g, cm, s)",
)

# Stellar-structure / SSE-friendly system: Msun, Rsun, Myr
ASTRO_STELLAR = UnitSystem(
    name="Stellar (Msun, Rsun, Myr)",
    mass_unit="Msun",
    length_unit="Rsun",
    time_unit="Myr",
    mass_scale_cgs=C.MSUN_G,
    length_scale_cgs=C.RSUN_CM,
    time_scale_cgs=C.MYR_S,
    description="Convenient for stellar evolution and SSE/BSE models.",
)

# Stellar dynamics system: Msun, pc, Myr
ASTRO_DYNAMICAL = UnitSystem(
    name="Stellar dynamics (Msun, pc, Myr)",
    mass_unit="Msun",
    length_unit="pc",
    time_unit="Myr",
    mass_scale_cgs=C.MSUN_G,
    length_scale_cgs=C.PC_CM,
    time_scale_cgs=C.MYR_S,
    description="Typical for star cluster dynamics / N-body codes.",
)

# Planetary / binary system: Msun, AU, yr
ASTRO_PLANETARY = UnitSystem(
    name="Planetary (Msun, AU, yr)",
    mass_unit="Msun",
    length_unit="AU",
    time_unit="yr",
    mass_scale_cgs=C.MSUN_G,
    length_scale_cgs=C.AU_CM,
    time_scale_cgs=C.YR_S,
    description="Solar system, exoplanets, close binaries.",
)

# Default for generic astro work (you can override per package)
DEFAULT = CGS
