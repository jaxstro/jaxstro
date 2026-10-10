"""Stellar-wind mass-loss laws shared by the ecosystem (jaxstro.winds).

Each law is a pure map from surface quantities in solar units to a mass-loss
rate in Msun/yr. Scheme selection, phase gates and regime dispatch belong to
the consumer (startrax, stellax); jaxstro owns the formulas only.

Laws: Reimers (1975) and Schroeder & Cuntz (2005), in ``jaxstro.winds.cool``;
Vink et al. (2001) and Nugis & Lamers (2000), in ``jaxstro.winds.hot``;
Bloecker (1995), in ``jaxstro.winds.agb``.
"""

from jaxstro.winds.agb import BLOECKER95_COEFFICIENT, bloecker95_mdot
from jaxstro.winds.cool import (
    REIMERS_COEFFICIENT_MSUN_PER_YR,
    SC05_ETA_FIT_MSUN_PER_YR,
    SC05_TEFF_MAX_K,
    SC05_TEFF_MIN_K,
    reimers_mdot,
    schroeder_cuntz05_mdot,
)
from jaxstro.winds.hot import (
    VINK01_VINF_OVER_VESC_COOL,
    VINK01_VINF_OVER_VESC_HOT,
    VINK01_VINF_Z_EXPONENT,
    VINK01_ZSUN,
    nugis_lamers00_eq1_mdot,
    nugis_lamers00_eq22_mdot,
    vink01_gamma_e,
    vink01_jump_teff_k,
    vink01_log_rho_eq14,
    vink01_log_rho_eq23,
    vink01_mdot_cool,
    vink01_mdot_hot,
)

__all__ = [
    "BLOECKER95_COEFFICIENT",
    "REIMERS_COEFFICIENT_MSUN_PER_YR",
    "SC05_ETA_FIT_MSUN_PER_YR",
    "SC05_TEFF_MAX_K",
    "SC05_TEFF_MIN_K",
    "VINK01_VINF_OVER_VESC_COOL",
    "VINK01_VINF_OVER_VESC_HOT",
    "VINK01_VINF_Z_EXPONENT",
    "VINK01_ZSUN",
    "bloecker95_mdot",
    "nugis_lamers00_eq1_mdot",
    "nugis_lamers00_eq22_mdot",
    "reimers_mdot",
    "schroeder_cuntz05_mdot",
    "vink01_gamma_e",
    "vink01_jump_teff_k",
    "vink01_log_rho_eq14",
    "vink01_log_rho_eq23",
    "vink01_mdot_cool",
    "vink01_mdot_hot",
]
