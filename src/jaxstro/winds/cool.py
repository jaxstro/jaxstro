"""Reimers (1975) and Schroeder & Cuntz (2005) mass-loss laws for cool giants.

Units: luminosity in Lsun, radius in Rsun, mass in Msun, effective temperature in
K; the rate is in Msun/yr. The inputs are the star's instantaneous surface
quantities (the current mass, not the initial mass).

Provenance (moved from startrax 2026-10-09, Anna's decision W0; the formulas and
coefficients are unchanged):

- Reimers: ``Mdot = eta * 4e-13 * L R / M``, with ``eta`` the dimensionless
  efficiency. The coefficient 4e-13 Msun/yr is the one printed in Hurley, Pols &
  Tout 2000 (MNRAS 315, 543) Eq. (106); startrax records it as
  ``VAL-H00-REIMERS-COEFFICIENT`` (verification level cross-verified).
- Schroeder & Cuntz 2005 (ApJL 630, L73) Eq. 4, the physical Reimers relation:
  ``Mdot = eta_SC * L R / M * (Teff / 4000 K)**3.5 * (1 + g_sun / (4300 g))``,
  with g the surface gravity, so ``g_sun / g = R**2 / M`` in solar units. The
  fitted normalisation is ``eta_SC = 8e-14 Msun/yr`` (the paper quotes
  8(+/-1)e-14); startrax record ``VAL-SC05-ETA`` (agent-verified). The paper
  calls the Teff band of relevance 3000-4500 K (Sect. 2), calibrated on
  globular-cluster red giants.

Neither law has a default efficiency here: the caller passes ``eta``. The
constants below are the published values, for callers that adopt them and for
cross-checks against a record.

Differentiability: both laws are smooth in every argument for positive inputs;
``jax.grad`` with respect to ``eta`` is exact (the rate is linear in it). No
input is clipped; a non-positive mass or radius returns non-finite values and
the caller is responsible for the domain.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

#: Reimers coefficient [Msun/yr per (Lsun Rsun / Msun)], Hurley et al. 2000 Eq. (106).
REIMERS_COEFFICIENT_MSUN_PER_YR: float = 4.0e-13

#: Schroeder & Cuntz 2005 fitted normalisation eta_SC [Msun/yr] (paper: 8 +/- 1 e-14).
SC05_ETA_FIT_MSUN_PER_YR: float = 8.0e-14

#: Schroeder & Cuntz 2005, Sect. 2: Teff band of relevance of Eq. 4 [K].
SC05_TEFF_MIN_K: float = 3000.0
SC05_TEFF_MAX_K: float = 4500.0

_SC05_TEFF_NORM_K = 4000.0
_SC05_GRAVITY_DIVISOR = 4300.0


def reimers_mdot(
    luminosity: ArrayLike,
    radius: ArrayLike,
    mass: ArrayLike,
    *,
    eta: ArrayLike,
) -> Array:
    """Reimers mass-loss rate [Msun/yr]: ``eta * 4e-13 * L R / M``.

    Args:
        luminosity: L [Lsun].
        radius: R [Rsun].
        mass: current stellar mass M [Msun].
        eta: dimensionless efficiency (no default; Hurley et al. 2000 adopt 0.5).
    """
    lum = jnp.asarray(luminosity, float)
    rad = jnp.asarray(radius, float)
    mas = jnp.asarray(mass, float)
    return jnp.asarray(eta, float) * REIMERS_COEFFICIENT_MSUN_PER_YR * lum * rad / mas


def schroeder_cuntz05_mdot(
    luminosity: ArrayLike,
    radius: ArrayLike,
    mass: ArrayLike,
    teff: ArrayLike,
    *,
    eta: ArrayLike,
) -> Array:
    """Schroeder & Cuntz (2005) Eq. 4 mass-loss rate [Msun/yr].

    ``eta * L R / M * (Teff/4000 K)**3.5 * (1 + R**2 / (4300 M))``, where
    ``R**2 / M = g_sun / g`` in solar units. The band of relevance is
    ``SC05_TEFF_MIN_K <= teff <= SC05_TEFF_MAX_K``; the function does not
    enforce it, so the caller reports an extrapolation as one.

    Args:
        luminosity: L [Lsun].
        radius: R [Rsun].
        mass: current stellar mass M [Msun].
        teff: effective temperature [K].
        eta: normalisation eta_SC [Msun/yr] (no default; the paper's fit is
            ``SC05_ETA_FIT_MSUN_PER_YR``).
    """
    lum = jnp.asarray(luminosity, float)
    rad = jnp.asarray(radius, float)
    mas = jnp.asarray(mass, float)
    teff_k = jnp.asarray(teff, float)
    gsun_over_g = rad**2 / mas
    return (
        jnp.asarray(eta, float)
        * lum
        * rad
        / mas
        * (teff_k / _SC05_TEFF_NORM_K) ** 3.5
        * (1.0 + gsun_over_g / _SC05_GRAVITY_DIVISOR)
    )
