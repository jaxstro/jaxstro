"""jaxstro.optics.dust_mixture: size integration and mixing against analytic limits."""

import jax.numpy as jnp
import pytest

from jaxstro.optics.dust_mixture import (
    mixture_optics,
    size_distribution_term_counts,
    species_cross_sections,
)
from jaxstro.optics.mie import mie_efficiencies, mie_term_counts

pytestmark = pytest.mark.unit


def _species(lam, a, a3n, m, rho):
    nt, ns = size_distribution_term_counts(lam, a, m)
    return species_cross_sections(wavelength_cm=lam, radius_cm=a, a3_dn_dlna=a3n, refractive_index=m,
                                  rho_g_cm3=rho, n_terms=nt, n_start=ns)


def test_single_size_gives_three_q_over_four_rho_a():
    """All grains of one radius a: kappa_x = pi a^2 Q_x / ((4 pi/3) rho a^3) = 3 Q_x / (4 rho a)."""
    lam = jnp.array([0.5e-4, 2.0e-4])
    a, a3n = jnp.array([0.9e-5, 1.0e-5, 1.1e-5]), jnp.array([0.0, 1.0, 0.0])
    m, rho = jnp.array([1.7 + 0.03j, 1.6 + 0.01j]), 3.0
    mix = mixture_optics([_species(lam, a, a3n, m, rho)])
    x = 2 * jnp.pi * 1.0e-5 / lam
    nt, ns = mie_term_counts(m, x)
    q = mie_efficiencies(m, x, n_terms=nt, n_start=ns)
    assert jnp.allclose(mix.kappa_abs_cm2_g, 3 * q.q_abs / (4 * rho * 1.0e-5), rtol=1e-12)
    assert jnp.allclose(mix.kappa_sca_cm2_g, 3 * q.q_sca / (4 * rho * 1.0e-5), rtol=1e-12)
    assert jnp.allclose(mix.g, q.g, rtol=1e-12)


def test_small_grains_absorb_independently_of_their_size_distribution():
    """Rayleigh limit: kappa_abs = (6 pi / (lambda rho)) Im((m^2-1)/(m^2+2)) for any size
    distribution of grains with x << 1 (here a <= 10 nm at lambda = 10 um, x <= 6e-3)."""
    lam = jnp.array([10.0e-4])
    a = jnp.geomspace(1e-7, 1e-6, 20)
    m, rho = jnp.array([1.7 + 0.3j]), 2.5
    expected = 6 * jnp.pi / (lam * rho) * ((m**2 - 1) / (m**2 + 2)).imag
    for a3n in (a**0.5, a**-1.5):
        mix = mixture_optics([_species(lam, a, a3n, m, rho)])
        assert float(jnp.abs(mix.kappa_abs_cm2_g[0] / expected[0] - 1)) <= 1e-4


def test_mixture_combines_species_by_mass_and_scattering():
    lam = jnp.array([0.55e-4])
    a = jnp.geomspace(1e-6, 1e-4, 30)
    s1 = _species(lam, a, a**0.5, jnp.array([1.7 + 0.03j]), 3.3)
    s2 = _species(lam, a, 2 * a**0.2, jnp.array([2.0 + 0.8j]), 1.8)
    mix = mixture_optics([s1, s2])
    assert jnp.allclose(mix.kappa_abs_cm2_g, (s1.sigma_abs_cm2 + s2.sigma_abs_cm2) / (s1.mass_g + s2.mass_g), rtol=1e-14)
    g_expected = (s1.sigma_sca_g_cm2 + s2.sigma_sca_g_cm2) / (s1.sigma_sca_cm2 + s2.sigma_sca_cm2)
    assert jnp.allclose(mix.g, g_expected, rtol=1e-14)
    assert bool(jnp.all((mix.g > 0) & (mix.g < 1)))
