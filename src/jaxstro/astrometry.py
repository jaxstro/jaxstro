# src/jaxstro/astrometry.py

"""
Astrometric constants and helpers (IAU-friendly).

These values are convenient for converting proper motions and
distances into velocities, and for general angular conversions.

All constants are module-level for direct access:
    >>> from jaxstro.astrometry import K_PROPER_MOTION
    >>> v_kms = mu_mas_yr * K_PROPER_MOTION * d_kpc
"""

# ===========================================================================
# Astrometric constants
# ===========================================================================

# Kilometres per parsec (IAU 2015, consistent with constants.PC_CM)
KM_PER_PC: float = 3.0856775814913673e13

# Milliarcseconds per radian
MAS_PER_RAD: float = 206264806.24709636

# Arcseconds per radian
ARCSEC_PER_RAD: float = 206264.80624709636

# Degrees per radian
DEG_PER_RAD: float = 57.29577951308232

# Years per megayear (exact)
YR_PER_MYR: float = 1.0e6

# Proper motion constant: km/s per (mas/yr × kpc)
# 1 mas/yr at 1 kpc corresponds to 4.74047 km/s.
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
