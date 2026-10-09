"""Stellar-wind mass-loss laws shared by the ecosystem (jaxstro.winds).

Each law is a pure map from surface quantities in solar units to a mass-loss
rate in Msun/yr. Scheme selection, phase gates and regime dispatch belong to
the consumer (startrax, stellax); jaxstro owns the formulas only.

Laws: Reimers (1975) and Schroeder & Cuntz (2005), in ``jaxstro.winds.cool``.
"""

from jaxstro.winds.cool import (
    REIMERS_COEFFICIENT_MSUN_PER_YR,
    SC05_ETA_FIT_MSUN_PER_YR,
    SC05_TEFF_MAX_K,
    SC05_TEFF_MIN_K,
    reimers_mdot,
    schroeder_cuntz05_mdot,
)

__all__ = [
    "REIMERS_COEFFICIENT_MSUN_PER_YR",
    "SC05_ETA_FIT_MSUN_PER_YR",
    "SC05_TEFF_MAX_K",
    "SC05_TEFF_MIN_K",
    "reimers_mdot",
    "schroeder_cuntz05_mdot",
]
