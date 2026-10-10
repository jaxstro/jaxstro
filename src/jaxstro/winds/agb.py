"""Bloecker (1995) AGB wind: Mdot = eta_B 4.83e-9 M^-2.1 L^2.7 x (Reimers rate at eta = 1).

Units: L [Lsun], R [Rsun], M [Msun]; rate [Msun/yr].

Bloecker (1995, A&A 297, 727) multiplies the Reimers rate by 4.83e-9 M^-2.1 L^2.7. The form below is the one MESA
r26.4.1 implements (``star/private/winds.f90`` ``eval_blocker_wind``: the Reimers rate 4e-13 L R/M at unit efficiency,
times ``Blocker_scaling_factor`` x 4.83e-9 M^-2.1 L^2.7). The paper is NOT on disk (``~/brain/knowledge/library``), so
the coefficients 4.83e-9, -2.1 and 2.7 are checked against MESA's source only; checking them against the paper's
equation is open (needs the paper fetched). ``eta`` has no default.

Differentiability: smooth for positive inputs; linear in ``eta``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from jaxstro.winds.cool import REIMERS_COEFFICIENT_MSUN_PER_YR

#: Bloecker 1995 coefficient of M^-2.1 L^2.7, as MESA's ``eval_blocker_wind``.
BLOECKER95_COEFFICIENT: float = 4.83e-9


def bloecker95_mdot(luminosity: ArrayLike, radius: ArrayLike, mass: ArrayLike, *, eta: ArrayLike) -> Array:
    """Bloecker mass-loss rate [Msun/yr]: ``eta * 4.83e-9 M^-2.1 L^2.7 * 4e-13 L R / M``.

    Args:
        luminosity: L [Lsun]. radius: R [Rsun]. mass: current mass M [Msun].
        eta: dimensionless efficiency (MESA ``Blocker_scaling_factor``; no default).
    """
    lum = jnp.asarray(luminosity, float)
    rad = jnp.asarray(radius, float)
    mas = jnp.asarray(mass, float)
    reimers_unit = REIMERS_COEFFICIENT_MSUN_PER_YR * lum * rad / mas
    return jnp.asarray(eta, float) * BLOECKER95_COEFFICIENT * mas ** -2.1 * lum ** 2.7 * reimers_unit
