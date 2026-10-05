"""Dust optics of a size distribution of homogeneous spheres: kappa_abs, kappa_sca per gram
and the scattering-weighted asymmetry g, from Mie efficiencies (jaxstro.optics.mie).

For one species with bulk density rho and a size distribution given as a^3 dn/d ln a (any
normalisation, for example per H atom) on a grid of radii a, at each wavelength lambda:

    sigma_x = int pi a^2 Q_x(m, 2 pi a / lambda) dn/d ln a d ln a       (x = abs, sca)
    sigma_sca g = int pi a^2 Q_sca g dn/d ln a d ln a
    M = int (4 pi / 3) rho a^3 dn/d ln a d ln a

with the trapezoid rule in ln a (jaxstro.quad.trapezoid) on the given grid. A mixture of
species has kappa_x = sum sigma_x / sum M and g = sum sigma_sca g / sum sigma_sca.
Differentiable in the refractive index, the radii, the size distribution and rho.
Design: docs/plans/2026-10-04-mie-scattering-design.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from jaxstro.optics.mie import mie_efficiencies, mie_term_counts
from jaxstro.quad.sampled import trapezoid


class SpeciesCrossSections(NamedTuple):
    """Per unit of the size distribution's normalisation (for example per H atom)."""

    sigma_abs_cm2: Float[Array, "L"]
    sigma_sca_cm2: Float[Array, "L"]
    sigma_sca_g_cm2: Float[Array, "L"]  # int pi a^2 Q_sca g dn
    mass_g: Float[Array, ""]


class MixtureOptics(NamedTuple):
    kappa_abs_cm2_g: Float[Array, "L"]  # per gram of dust
    kappa_sca_cm2_g: Float[Array, "L"]
    g: Float[Array, "L"]  # scattering-weighted asymmetry


def size_distribution_term_counts(wavelength_cm, radius_cm, refractive_index) -> tuple[int, int]:
    """Static (n_terms, n_start) covering every (radius, wavelength) pair (host side)."""
    lam = jnp.asarray(wavelength_cm, dtype=jnp.float64)
    a = jnp.asarray(radius_cm, dtype=jnp.float64)
    m = jnp.broadcast_to(jnp.asarray(refractive_index, dtype=jnp.complex128), lam.shape)
    x = 2.0 * math.pi * a[None, :] / lam[:, None]
    return mie_term_counts(m[:, None], x)


def species_cross_sections(
    *,
    wavelength_cm: Float[Array, "L"],
    radius_cm: Float[Array, "A"],
    a3_dn_dlna: Float[Array, "A"],
    refractive_index: Complex[Array, "L"],
    rho_g_cm3: float | Array,
    n_terms: int,
    n_start: int,
) -> SpeciesCrossSections:
    """Size-integrated cross sections of one species (module doc). One wavelength at a
    time (lax.map), so memory is n_terms x A; `size_distribution_term_counts` sizes the
    static series."""
    lam = jnp.asarray(wavelength_cm, dtype=jnp.float64)
    a = jnp.asarray(radius_cm, dtype=jnp.float64)
    a3n = jnp.asarray(a3_dn_dlna, dtype=jnp.float64)
    m = jnp.broadcast_to(jnp.asarray(refractive_index, dtype=jnp.complex128), lam.shape)
    ln_a = jnp.log(a)
    weight = jnp.pi * a3n / a  # pi a^2 dn/d ln a

    def one_wavelength(args):
        lam_k, m_k = args
        q = mie_efficiencies(m_k, 2.0 * jnp.pi * a / lam_k, n_terms=n_terms, n_start=n_start)
        integrate = lambda f: trapezoid(f * weight, ln_a)  # noqa: E731
        return integrate(q.q_abs), integrate(q.q_sca), integrate(q.q_sca * q.g)

    s_abs, s_sca, s_sca_g = jax.lax.map(one_wavelength, (lam, m))
    mass = trapezoid(4.0 / 3.0 * jnp.pi * jnp.asarray(rho_g_cm3, dtype=jnp.float64) * a3n, ln_a)
    return SpeciesCrossSections(s_abs, s_sca, s_sca_g, mass)


def mixture_optics(species: Sequence[SpeciesCrossSections]) -> MixtureOptics:
    """kappa per gram of the whole mixture and its scattering-weighted g (module doc)."""
    mass = sum(s.mass_g for s in species)
    s_sca = sum(s.sigma_sca_cm2 for s in species)
    return MixtureOptics(
        sum(s.sigma_abs_cm2 for s in species) / mass,
        s_sca / mass,
        sum(s.sigma_sca_g_cm2 for s in species) / s_sca,
    )


__all__ = [
    "MixtureOptics",
    "SpeciesCrossSections",
    "mixture_optics",
    "size_distribution_term_counts",
    "species_cross_sections",
]
