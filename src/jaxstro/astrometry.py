# src/jaxstro/astrometry.py

"""
Astrometric constants and helpers.

These values are convenient for converting proper motions and
distances into velocities, and for general angular conversions.

All constants are module-level for direct access:
    >>> from jaxstro.astrometry import K_PROPER_MOTION
    >>> v_kms = mu_mas_yr * K_PROPER_MOTION * d_kpc
"""

# ===========================================================================
# Astrometric constants
# ===========================================================================

# Kilometres per parsec, derived from constants.PC_CM / constants.KM_CM.
# The parsec uses the IAU 2012 exact astronomical unit and the angular
# definition of an arcsecond; this stored value is a compatibility literal.
KM_PER_PC: float = 3.0856775814913673e13

# Milliarcseconds per radian
MAS_PER_RAD: float = 206264806.24709636

# Arcseconds per radian
ARCSEC_PER_RAD: float = 206264.80624709636

# Degrees per radian
DEG_PER_RAD: float = 57.29577951308232

# Years per megayear (exact)
YR_PER_MYR: float = 1.0e6

# Proper-motion conversion [km/s per (mas/yr × kpc)].
# 1 mas/yr at 1 kpc equals 1 AU per Julian year. The exact conventional
# conversion from constants.AU_PER_YR_TO_KM_PER_S is 4.740470463... km/s;
# K_PROPER_MOTION retains the long-standing rounded compatibility literal.
K_PROPER_MOTION: float = 4.74047

# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    "KM_PER_PC",
    "MAS_PER_RAD",
    "ARCSEC_PER_RAD",
    "DEG_PER_RAD",
    "YR_PER_MYR",
    "K_PROPER_MOTION",
]
