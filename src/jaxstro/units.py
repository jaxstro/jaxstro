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

import jax.numpy as jnp

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

    @property
    def G(self) -> float:
        """
        Gravitational constant in this unit system.

        Computed from CGS value using dimensional analysis.
        G has dimensions [length³ mass⁻¹ time⁻²].

        To convert from CGS (cm³ g⁻¹ s⁻²) to code units (L³ M⁻¹ T⁻²):
            G_code = G_CGS × (mass_cgs/1) × (time_cgs/1)² / (length_cgs/1)³

        Returns
        -------
        float
            G in units of [length³ mass⁻¹ time⁻²]

        Examples
        --------
        >>> from jaxstro.units import ASTRO_DYNAMICAL, ASTRO_PLANETARY, CGS
        >>> ASTRO_DYNAMICAL.G  # ~0.00450 pc³ Msun⁻¹ Myr⁻²
        >>> ASTRO_PLANETARY.G  # ~39.48 AU³ Msun⁻¹ yr⁻² (≈ 4π²)
        >>> CGS.G              # 6.67430e-8 cm³ g⁻¹ s⁻²
        """
        return (
            C.G_CGS
            * self.mass_scale_cgs
            * (self.time_scale_cgs**2)
            / (self.length_scale_cgs**3)
        )

    def convert_length(self, value: float, *, to: "UnitSystem") -> float:
        """
        Convert length from this unit system to another.

        Parameters
        ----------
        value : float
            Length in this unit system.
        to : UnitSystem
            Target unit system.

        Returns
        -------
        float
            Length in target unit system.
        """
        return value * (self.length_scale_cgs / to.length_scale_cgs)

    def convert_mass(self, value: float, *, to: "UnitSystem") -> float:
        """
        Convert mass from this unit system to another.

        Parameters
        ----------
        value : float
            Mass in this unit system.
        to : UnitSystem
            Target unit system.

        Returns
        -------
        float
            Mass in target unit system.
        """
        return value * (self.mass_scale_cgs / to.mass_scale_cgs)

    def convert_time(self, value: float, *, to: "UnitSystem") -> float:
        """
        Convert time from this unit system to another.

        Parameters
        ----------
        value : float
            Time in this unit system.
        to : UnitSystem
            Target unit system.

        Returns
        -------
        float
            Time in target unit system.
        """
        return value * (self.time_scale_cgs / to.time_scale_cgs)

    def convert_velocity(self, value: float, *, to: "UnitSystem") -> float:
        """
        Convert velocity from this unit system to another.

        Parameters
        ----------
        value : float
            Velocity in this unit system.
        to : UnitSystem
            Target unit system.

        Returns
        -------
        float
            Velocity in target unit system.
        """
        return value * (self.velocity_scale_cgs / to.velocity_scale_cgs)

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

# ---------------------------------------------------------------------------
# Short aliases for common use cases
# ---------------------------------------------------------------------------

STELLAR = ASTRO_DYNAMICAL  # Star clusters: Msun, pc, Myr
STAR = ASTRO_STELLAR  # Individual stars: Msun, Rsun, Myr (startrax, stellax)
BINARY = ASTRO_PLANETARY  # Binary stars: Msun, AU, yr
SOLAR = ASTRO_STELLAR  # Stellar structure: Msun, Rsun, Myr
PLANETARY = ASTRO_PLANETARY  # Alias for BINARY

# ---------------------------------------------------------------------------
# Photometric unit systems
# ---------------------------------------------------------------------------


# Resolution tables: unit-label -> CGS scale factor (host-side floats).
# These mirror UnitSystem's pattern: the unit choice is a *static* host-side
# string, and the per-call conversion is a constant float multiply, so all
# methods are jit/vmap/grad-safe (no data-dependent branching on the
# differentiated path).
_LUMINOSITY_SCALES_CGS: dict[str, float] = {
    "Lsun": C.LSUN_ERG_S,  # Lsun -> erg/s
    "cgs": 1.0,  # erg/s (identity)
}
_RADIUS_SCALES_CGS: dict[str, float] = {
    "Rsun": C.RSUN_CM,  # Rsun -> cm
    "cm": 1.0,  # cm (identity)
}
# Linear flux-density scales (Jy / cgs). "AB" is magnitude-based and handled
# via the AB zeropoint, NOT a linear scale; it is intentionally absent here.
_FLUX_SCALES_CGS: dict[str, float] = {
    "Jy": C.JY_CGS,  # Jy -> erg/s/cm^2/Hz
    "cgs": 1.0,  # erg/s/cm^2/Hz (identity)
}


@dataclass(frozen=True)
class PhotometricUnits:
    """
    Photometric unit system for luminosity / radius / flux-density outputs.

    Parallel to :class:`UnitSystem` but for *photometric* quantities that the
    dynamical (mass/length/time) basis cannot represent (L☉, R☉, Jy, AB mag).
    The unit choices are static host-side strings; their CGS scale factors are
    resolved once at construction time into plain Python floats, so every
    conversion method is a constant float multiply — JAX jit/vmap/grad-safe
    with no data-dependent branching on the differentiated path.

    Attributes
    ----------
    luminosity : str
        Luminosity unit, ``"Lsun"`` or ``"cgs"`` (erg/s).
    radius : str
        Radius unit, ``"Rsun"`` or ``"cm"``.
    flux_density : str
        Flux-density unit: ``"Jy"`` | ``"cgs"`` (erg/s/cm²/Hz) | ``"AB"`` (mag).
        ``"AB"`` is magnitude-based; use the ``ab_mag_*`` methods rather than
        the linear ``*_flux`` scale.
    name : str
        Optional descriptive name.

    Notes
    -----
    Resolved scale factors (read-only, set at construction):

    - ``luminosity_scale_cgs`` — value of 1 [luminosity] in erg/s.
    - ``radius_scale_cgs`` — value of 1 [radius] in cm.
    - ``flux_scale_cgs`` — value of 1 [flux_density] in erg/s/cm²/Hz
      (defined for ``"Jy"`` / ``"cgs"``; for ``"AB"`` it is ``NaN`` — AB
      conversions go through the zeropoint methods, and NaN poisons any
      accidental use of the linear ``*_flux`` path).
    """

    luminosity: str = "Lsun"
    radius: str = "Rsun"
    flux_density: str = "Jy"
    name: str = ""

    # Resolved at construction (frozen dataclass; set via object.__setattr__).
    luminosity_scale_cgs: float = 0.0
    radius_scale_cgs: float = 0.0
    flux_scale_cgs: float = 0.0

    def __post_init__(self) -> None:
        if self.luminosity not in _LUMINOSITY_SCALES_CGS:
            raise ValueError(
                f"Unknown luminosity unit: {self.luminosity!r}. "
                f"Choose from {sorted(_LUMINOSITY_SCALES_CGS)}."
            )
        if self.radius not in _RADIUS_SCALES_CGS:
            raise ValueError(
                f"Unknown radius unit: {self.radius!r}. "
                f"Choose from {sorted(_RADIUS_SCALES_CGS)}."
            )
        if self.flux_density not in {"Jy", "cgs", "AB"}:
            raise ValueError(
                f"Unknown flux_density unit: {self.flux_density!r}. "
                "Choose from ['AB', 'Jy', 'cgs']."
            )
        # Resolve scales host-side (constant multiplies downstream).
        object.__setattr__(
            self, "luminosity_scale_cgs", _LUMINOSITY_SCALES_CGS[self.luminosity]
        )
        object.__setattr__(self, "radius_scale_cgs", _RADIUS_SCALES_CGS[self.radius])
        # "AB" is magnitude-based and carries NO linear flux scale. Poison it
        # with NaN so any accidental linear use (to_cgs_flux/from_cgs_flux)
        # fails loud (NaN propagates) instead of silently returning a
        # wrong-by-construction result; AB conversions go through the zeropoint.
        flux_scale = _FLUX_SCALES_CGS.get(self.flux_density, float("nan"))
        object.__setattr__(self, "flux_scale_cgs", flux_scale)

    # --- Luminosity conversions (constant multiplies) ----------------------
    def to_cgs_luminosity(self, value):
        """Convert luminosity from this system to CGS (erg/s)."""
        return value * self.luminosity_scale_cgs

    def from_cgs_luminosity(self, value_cgs):
        """Convert luminosity from CGS (erg/s) to this system."""
        return value_cgs / self.luminosity_scale_cgs

    # --- Radius conversions ------------------------------------------------
    def to_cgs_radius(self, value):
        """Convert radius from this system to CGS (cm)."""
        return value * self.radius_scale_cgs

    def from_cgs_radius(self, value_cgs):
        """Convert radius from CGS (cm) to this system."""
        return value_cgs / self.radius_scale_cgs

    # --- Linear flux-density conversions (Jy / cgs) ------------------------
    def to_cgs_flux(self, value):
        """Convert flux density from this system to CGS (erg/s/cm²/Hz).

        Defined for the linear ``"Jy"``/``"cgs"`` choices; for ``"AB"`` use
        :meth:`ab_mag_to_cgs_flux` instead.
        """
        return value * self.flux_scale_cgs

    def from_cgs_flux(self, value_cgs):
        """Convert flux density from CGS (erg/s/cm²/Hz) to this system."""
        return value_cgs / self.flux_scale_cgs

    # --- AB-magnitude flux conversions (zeropoint-based) -------------------
    def ab_mag_to_cgs_flux(self, mag):
        """AB magnitude -> CGS flux density (erg/s/cm²/Hz).

        f = f_AB(zeropoint) × 10^(-0.4 m), with the AB zeropoint
        3631 Jy = 3.631e-20 erg/s/cm²/Hz (Oke & Gunn 1983). Differentiable
        in ``mag`` (smooth exponential, no branching).
        """
        return C.AB_ZEROPOINT_CGS * (10.0 ** (-0.4 * mag))

    def cgs_flux_to_ab_mag(self, flux_cgs):
        """CGS flux density (erg/s/cm²/Hz) -> AB magnitude.

        m = -2.5 log10(f / f_AB). Differentiable for positive flux.
        """
        return -2.5 * (jnp.log10(flux_cgs / C.AB_ZEROPOINT_CGS))

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.name or 'PhotometricUnits'} "
            f"(L=[{self.luminosity}], R=[{self.radius}], "
            f"F=[{self.flux_density}])"
        )


# Conventional default: solar luminosity/radius, Jy flux density.
SOLAR_PHOTOMETRIC = PhotometricUnits(
    luminosity="Lsun",
    radius="Rsun",
    flux_density="Jy",
    name="Solar photometric (Lsun, Rsun, Jy)",
)

# Pure-CGS photometric system.
CGS_PHOTOMETRIC = PhotometricUnits(
    luminosity="cgs",
    radius="cm",
    flux_density="cgs",
    name="CGS photometric (erg/s, cm, erg/s/cm^2/Hz)",
)


# ---------------------------------------------------------------------------
# Unit system registry and lookup
# ---------------------------------------------------------------------------

UNIT_SYSTEMS: dict[str, UnitSystem] = {
    "cgs": CGS,
    "stellar": STELLAR,
    "star": STAR,
    "binary": BINARY,
    "solar": SOLAR,
    "planetary": PLANETARY,
    "astro_stellar": ASTRO_STELLAR,
    "astro_dynamical": ASTRO_DYNAMICAL,
    "astro_planetary": ASTRO_PLANETARY,
}


def get_units(name: str) -> UnitSystem:
    """
    Get a predefined unit system by name (case-insensitive).

    Parameters
    ----------
    name : str
        Name of the unit system (e.g., "stellar", "binary", "cgs").

    Returns
    -------
    UnitSystem
        The requested unit system.

    Raises
    ------
    KeyError
        If the name is not recognized.

    Examples
    --------
    >>> from jaxstro.units import get_units
    >>> stellar = get_units("stellar")
    >>> stellar.G  # ~0.00450 pc³ Msun⁻¹ Myr⁻²
    """
    key = name.lower()
    if key not in UNIT_SYSTEMS:
        available = list(UNIT_SYSTEMS.keys())
        raise KeyError(f"Unknown unit system: '{name}'. Available: {available}")
    return UNIT_SYSTEMS[key]


__all__ = [
    "UnitSystem",
    "CGS",
    "ASTRO_STELLAR",
    "ASTRO_DYNAMICAL",
    "ASTRO_PLANETARY",
    "DEFAULT",
    # Short aliases
    "STELLAR",
    "STAR",
    "BINARY",
    "SOLAR",
    "PLANETARY",
    # Photometric units
    "PhotometricUnits",
    "SOLAR_PHOTOMETRIC",
    "CGS_PHOTOMETRIC",
    # Lookup
    "UNIT_SYSTEMS",
    "get_units",
]
