# src/jaxstro/constants.py

"""
Physical constants for the jaxstro ecosystem (CGS units).

These values follow IAU / CODATA recommendations and are intended
to be the single source of truth for all downstream packages.

Units:
    - Length: cm
    - Mass:   g
    - Time:   s
    - Energy: erg = g·cm²/s²
    - Pressure: dyn/cm² = g/(cm·s²)

References:
    - CODATA 2018: Tiesinga et al. (2021), Rev. Mod. Phys., 93, 025010
    - IAU 2012 B2: astronomical unit (AU)
    - IAU 2015 B3: nominal solar parameters (M_sun, R_sun, L_sun)
    - Asplund et al. (2009), ARA&A, 47, 481: Solar composition
"""

# ===========================================================================
# Fundamental constants (CODATA 2018)
# ===========================================================================

# Gravitational constant [cm³ g⁻¹ s⁻²]
G_CGS: float = 6.67430e-8

# Speed of light [cm/s] (exact)
C_CGS: float = 2.99792458e10

# Planck constant [erg·s] (exact)
H_CGS: float = 6.62607015e-27

# Boltzmann constant [erg/K] (exact)
K_B: float = 1.380649e-16

# Radiation constant [erg cm⁻³ K⁻⁴]
A_RAD: float = 7.565767e-15

# Avogadro's number [mol⁻¹]
N_A: float = 6.02214076e23

# Electron volt [erg]
EV_ERG: float = 1.602176634e-12

# ===========================================================================
# Atomic and particle masses (CODATA 2018)
# ===========================================================================

# Atomic mass unit [g]
M_U: float = 1.66053906660e-24

# Electron mass [g]
M_E: float = 9.1093837015e-28

# Proton mass [g]
M_P: float = 1.67262192369e-24

# Neutron mass [g]
M_N: float = 1.67492749804e-24

# ===========================================================================
# Solar parameters (IAU 2015 nominal values)
# https://www.iau.org/static/resolutions/IAU2015_English.pdf
# ===========================================================================

MSUN_G: float = 1.9884e33  # Solar mass [g]
RSUN_CM: float = 6.957e10  # Solar radius [cm]
LSUN_ERG_S: float = 3.828e33  # Solar luminosity [erg/s]
TEFF_SUN: float = 5772.0  # Solar effective temperature [K]

# ===========================================================================
# Solar composition (Asplund et al. 2009)
# ===========================================================================

X_SUN: float = 0.7381  # Solar hydrogen mass fraction
Y_SUN: float = 0.2485  # Solar helium mass fraction
Z_SUN: float = 0.0134  # Solar metals mass fraction

# ===========================================================================
# Astronomical distance and time units
# ===========================================================================

# Parsec [cm] (from AU and arcsec definition)
PC_CM: float = 3.0856775814913673e18

# Astronomical unit [cm] (IAU 2012 definition: 149,597,870,700 m)
AU_CM: float = 1.495978707e13

# Time units
MYR_S: float = 3.15576e13  # s in 1 Myr (1e6 tropical years)
YR_S: float = 3.15576e7  # s in 1 yr (tropical year)

# Metric conversion
KM_CM: float = 1.0e5  # 1 km = 1e5 cm

# ===========================================================================
# Radiation / thermodynamics
# ===========================================================================

# Stefan–Boltzmann constant [erg cm⁻² s⁻¹ K⁻⁴]
SIGMA_SB: float = 5.670374419e-5

# ===========================================================================
# Photometric units and zeropoints
# ===========================================================================

# Jansky [erg s⁻¹ cm⁻² Hz⁻¹ per Jy].
# Definition: 1 Jy = 1e-26 W m⁻² Hz⁻¹ (SI) = 1e-23 erg s⁻¹ cm⁻² Hz⁻¹ (CGS),
# since 1 W = 1e7 erg/s and 1 m⁻² = 1e-4 cm⁻² → 1e7 × 1e-4 × 1e-26 = 1e-23.
JY_CGS: float = 1e-23

# AB magnitude system zeropoint flux density [Jy].
# f_AB = 3631 Jy at AB mag 0 (Oke & Gunn 1983, ApJ, 266, 713).
AB_ZEROPOINT_JY: float = 3631.0

# AB zeropoint flux density in CGS [erg s⁻¹ cm⁻² Hz⁻¹].
# = AB_ZEROPOINT_JY × JY_CGS = 3631 × 1e-23 (Oke & Gunn 1983, ApJ, 266, 713).
AB_ZEROPOINT_CGS: float = AB_ZEROPOINT_JY * JY_CGS  # 3.631e-20

# ===========================================================================
# Derived velocity conversions
# ===========================================================================

# 1 pc / Myr in km/s
PC_PER_MYR_TO_KM_PER_S: float = (PC_CM / MYR_S) / KM_CM

# 1 AU / yr in km/s
AU_PER_YR_TO_KM_PER_S: float = (AU_CM / YR_S) / KM_CM

# Inverse conversions
KM_PER_S_TO_PC_PER_MYR: float = 1.0 / PC_PER_MYR_TO_KM_PER_S
KM_PER_S_TO_AU_PER_YR: float = 1.0 / AU_PER_YR_TO_KM_PER_S

# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    # Fundamental constants
    "G_CGS",
    "C_CGS",
    "H_CGS",
    "K_B",
    "A_RAD",
    "N_A",
    "EV_ERG",
    # Particle masses
    "M_U",
    "M_E",
    "M_P",
    "M_N",
    # Solar parameters
    "MSUN_G",
    "RSUN_CM",
    "LSUN_ERG_S",
    "TEFF_SUN",
    # Solar composition
    "X_SUN",
    "Y_SUN",
    "Z_SUN",
    # Distance and time units
    "PC_CM",
    "AU_CM",
    "MYR_S",
    "YR_S",
    "KM_CM",
    # Radiation
    "SIGMA_SB",
    # Photometric units and zeropoints
    "JY_CGS",
    "AB_ZEROPOINT_JY",
    "AB_ZEROPOINT_CGS",
    # Velocity conversions
    "PC_PER_MYR_TO_KM_PER_S",
    "AU_PER_YR_TO_KM_PER_S",
    "KM_PER_S_TO_PC_PER_MYR",
    "KM_PER_S_TO_AU_PER_YR",
]
