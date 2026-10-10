"""Hot-star and Wolf-Rayet wind relations: Vink et al. (2001) Eqs. 11, 14, 15, 23, 24, 25 and Nugis & Lamers (2000)
Eqs. 1 and 22.

Units: luminosity and mass in solar units, effective temperature in K, abundances as mass fractions, the rate in
Msun/yr. Each function is one printed equation; scheme selection, the bistability switch, blends between laws and
the choice of metallicity belong to the consumer (stellax ``physics.winds`` builds MESA's composition from these).

Vink, de Koter & Lamers (2001, A&A 369, 574; PDF checked, ``vink2001massloss.pdf``):

- Eq. (24), hot side of the bistability jump, 27 500 < Teff <= 50 000 K, v_inf/v_esc = 2.6 (Galactic):
  log Mdot = -6.697 + 2.194 log(L/1e5) - 1.313 log(M/30) - 1.226 log((v_inf/v_esc)/2.0)
             + 0.933 log(Teff/40000) - 10.92 [log(Teff/40000)]^2 + 0.85 log(Z/Zsun).
- Eq. (25), cool side, 12 500 <= Teff <= 22 500 K, v_inf/v_esc = 1.3 (Galactic):
  log Mdot = -6.688 + 2.210 log(L/1e5) - 1.339 log(M/30) - 1.601 log((v_inf/v_esc)/2.0)
             + 1.07 log(Teff/20000) + 0.85 log(Z/Zsun).
- Eq. (14): log <rho> = -13.636 + 0.889 log(Z/Zsun); Eq. (23): log <rho> = -14.94 + 0.85 log(Z/Zsun) + 3.2 Gamma_e;
  Eq. (15): T_jump [kK] = 61.2 + 2.59 log <rho>; Eq. (11): Gamma_e = 7.66e-5 sigma_e (L/M), sigma_e in cm^2/g.
  The paper's complete recipe (Sect. 8) places the jump with Eqs. (15)-(17) and (23); Eq. (14) is the
  Gamma_e-independent fit of Sect. 5.1.
- Zsun = 0.019 (Sect. 4, Anders & Grevesse 1989); v_inf proportional to Z^0.13 (Leitherer et al. 1992, as quoted in
  Sect. 8), so the Galactic ratios scale as (Z/Zsun)^0.13.
- The paper states validity ranges for Eqs. (24) and (25); the functions here do not enforce them. The lower range
  of Eq. (25) is 12 500 K; below it Sect. 8 gives a different constant for Z/Zsun >~ 1 (Fe III/II jump).

Nugis & Lamers (2000, A&A 360, 227; PDF checked, ``nugis2000wrmassloss.pdf``):

- Eq. (1) (abstract, rounded): Mdot = 1.0e-11 (L/Lsun)^1.29 Y^1.7 Z^0.5, sigma = 0.19 dex, for WN and WC stars.
- Eq. (22) (the fit): log Mdot = -11.00 + 1.29 log L + 1.73 log Y + 0.47 log Z, the same sample; Z is the surface
  metal mass fraction (mainly carbon in WC stars).

Differentiability: smooth in every argument for positive inputs; no input is clipped (a zero Y or Z gives non-finite
logarithms; the caller selects the branch).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

#: Vink et al. 2001, Sect. 4: Zsun = 0.019 (Anders & Grevesse 1989).
VINK01_ZSUN: float = 0.019
#: Galactic v_inf / v_esc on the hot and cool side of the jump (Sect. 8, after Eqs. 24 and 25).
VINK01_VINF_OVER_VESC_HOT: float = 2.6
VINK01_VINF_OVER_VESC_COOL: float = 1.3
#: v_inf proportional to Z^0.13 (Leitherer et al. 1992, quoted in Sect. 8).
VINK01_VINF_Z_EXPONENT: float = 0.13


def vink01_gamma_e(sigma_e: ArrayLike, luminosity: ArrayLike, mass: ArrayLike) -> Array:
    """Eq. (11): Gamma_e = 7.66e-5 sigma_e (L/Lsun)/(M/Msun); ``sigma_e`` in cm^2/g (the paper takes its dependence on
    Teff and composition from Lamers & Leitherer 1993; it is an input here)."""
    return 7.66e-5 * jnp.asarray(sigma_e, float) * jnp.asarray(luminosity, float) / jnp.asarray(mass, float)


def vink01_log_rho_eq14(z_ratio: ArrayLike) -> Array:
    """Eq. (14): log10 of the characteristic density at the jump, -13.636 + 0.889 log(Z/Zsun)."""
    return -13.636 + 0.889 * jnp.log10(jnp.asarray(z_ratio, float))


def vink01_log_rho_eq23(z_ratio: ArrayLike, gamma_e: ArrayLike) -> Array:
    """Eq. (23): log10 of the characteristic density at the jump, -14.94 + 0.85 log(Z/Zsun) + 3.2 Gamma_e."""
    return -14.94 + 0.85 * jnp.log10(jnp.asarray(z_ratio, float)) + 3.2 * jnp.asarray(gamma_e, float)


def vink01_jump_teff_k(log_rho: ArrayLike) -> Array:
    """Eq. (15): the bistability jump temperature [K], 1000 (61.2 + 2.59 log <rho>)."""
    return 1.0e3 * (61.2 + 2.59 * jnp.asarray(log_rho, float))


def vink01_mdot_hot(luminosity: ArrayLike, mass: ArrayLike, teff: ArrayLike, z_ratio: ArrayLike,
                    vinf_over_vesc: ArrayLike) -> Array:
    """Eq. (24), the hot side of the jump [Msun/yr] (module doc). ``z_ratio`` = Z/Zsun; ``vinf_over_vesc`` the wind's
    v_inf/v_esc (Galactic 2.6; scale with (Z/Zsun)^0.13 as the paper says)."""
    lt = jnp.log10(jnp.asarray(teff, float) / 40000.0)
    log_mdot = (-6.697
                + 2.194 * jnp.log10(jnp.asarray(luminosity, float) / 1.0e5)
                - 1.313 * jnp.log10(jnp.asarray(mass, float) / 30.0)
                - 1.226 * jnp.log10(jnp.asarray(vinf_over_vesc, float) / 2.0)
                + 0.933 * lt
                - 10.92 * lt * lt
                + 0.85 * jnp.log10(jnp.asarray(z_ratio, float)))
    return 10.0 ** log_mdot


def vink01_mdot_cool(luminosity: ArrayLike, mass: ArrayLike, teff: ArrayLike, z_ratio: ArrayLike,
                     vinf_over_vesc: ArrayLike) -> Array:
    """Eq. (25), the cool side of the jump [Msun/yr] (module doc). Arguments as ``vink01_mdot_hot``; Galactic
    v_inf/v_esc = 1.3."""
    log_mdot = (-6.688
                + 2.210 * jnp.log10(jnp.asarray(luminosity, float) / 1.0e5)
                - 1.339 * jnp.log10(jnp.asarray(mass, float) / 30.0)
                - 1.601 * jnp.log10(jnp.asarray(vinf_over_vesc, float) / 2.0)
                + 1.07 * jnp.log10(jnp.asarray(teff, float) / 20000.0)
                + 0.85 * jnp.log10(jnp.asarray(z_ratio, float)))
    return 10.0 ** log_mdot


def nugis_lamers00_eq1_mdot(luminosity: ArrayLike, helium_y: ArrayLike, metals_z: ArrayLike) -> Array:
    """Nugis & Lamers 2000 Eq. (1) (rounded), Mdot = 1.0e-11 L^1.29 Y^1.7 Z^0.5 [Msun/yr], L in Lsun."""
    return (1.0e-11 * jnp.asarray(luminosity, float) ** 1.29 * jnp.asarray(helium_y, float) ** 1.7
            * jnp.asarray(metals_z, float) ** 0.5)


def nugis_lamers00_eq22_mdot(luminosity: ArrayLike, helium_y: ArrayLike, metals_z: ArrayLike) -> Array:
    """Nugis & Lamers 2000 Eq. (22) (the fit), log Mdot = -11.00 + 1.29 log L + 1.73 log Y + 0.47 log Z
    [Msun/yr], L in Lsun."""
    return 10.0 ** (-11.00 + 1.29 * jnp.log10(jnp.asarray(luminosity, float))
                    + 1.73 * jnp.log10(jnp.asarray(helium_y, float))
                    + 0.47 * jnp.log10(jnp.asarray(metals_z, float)))
