"""Owner tests of jaxstro.winds.cool (Reimers 1975; Schroeder & Cuntz 2005 Eq. 4).

Each reference is a value written out from the published formula, an exact limit
of it, or an exact scaling. Units: L [Lsun], R [Rsun], M [Msun], Teff [K],
rate [Msun/yr].
"""

import jax
import pytest

from jaxstro.winds import (
    REIMERS_COEFFICIENT_MSUN_PER_YR,
    SC05_ETA_FIT_MSUN_PER_YR,
    SC05_TEFF_MAX_K,
    SC05_TEFF_MIN_K,
    reimers_mdot,
    schroeder_cuntz05_mdot,
)

_reimers = jax.jit(reimers_mdot)
_sc05 = jax.jit(schroeder_cuntz05_mdot)

# Round-off of a product of five factors; the values below are exact decimals or
# computed with plain Python floats, so 1e-13 is ~1000 ulp of margin.
_RTOL = 1.0e-13


@pytest.mark.unit
def test_published_constants():
    """A retyped coefficient: 4e-13 (Hurley+2000 Eq. 106), 8e-14 and 3000-4500 K (SC05)."""
    assert REIMERS_COEFFICIENT_MSUN_PER_YR == 4.0e-13
    assert SC05_ETA_FIT_MSUN_PER_YR == 8.0e-14
    assert (SC05_TEFF_MIN_K, SC05_TEFF_MAX_K) == (3000.0, 4500.0)


@pytest.mark.unit
def test_reimers_hand_value_and_eta_linearity():
    """A wrong power of L, R or M, or a dropped coefficient.

    Hurley's convention eta = 0.5 at L=1000, R=50, M=1 gives 0.5*4e-13*5e4 = 1e-8.
    """
    assert float(_reimers(1000.0, 50.0, 1.0, eta=0.5)) == pytest.approx(
        1.0e-8, rel=_RTOL
    )
    # L R / M scaling: L x2, R x3, M x4 multiplies the rate by 2*3/4.
    base = float(_reimers(1000.0, 50.0, 1.0, eta=0.5))
    scaled = float(_reimers(2000.0, 150.0, 4.0, eta=0.5))
    assert scaled == pytest.approx(1.5 * base, rel=_RTOL)


@pytest.mark.unit
def test_reimers_gradient_in_eta_is_the_coefficient():
    """A rate nonlinear in eta: at L=R=M=1 the d/d eta derivative is exactly 4e-13."""
    grad = jax.grad(lambda e: reimers_mdot(1.0, 1.0, 1.0, eta=e))(0.5)
    assert float(grad) == pytest.approx(4.0e-13, rel=1.0e-15)


@pytest.mark.unit
def test_sc05_hand_value():
    """A wrong Teff exponent, gravity term or normalisation in Eq. 4.

    L=800, R=90, M=1.2, Teff=3800 K, eta=8e-14, evaluated with plain Python
    floats from 8e-14 L R/M (T/4000)^3.5 (1 + R^2/(4300 M)).
    """
    expected = 1.0307840227563399e-08
    got = float(_sc05(800.0, 90.0, 1.2, 3800.0, eta=SC05_ETA_FIT_MSUN_PER_YR))
    assert got == pytest.approx(expected, rel=_RTOL)


@pytest.mark.unit
def test_sc05_exact_limits():
    """A gravity term written as g/g_sun instead of g_sun/g, or a wrong Teff reference.

    At Teff = 4000 K and R^2/M = 4300 (R = 43, M = 0.43: 1849/0.43 = 4300), the
    bracket is exactly 2 and the Teff factor is 1, so Mdot = 2 eta L R/M.
    At large R^2/M the bracket is 1 + R^2/(4300 M) so a factor-2 change in Teff
    changes the rate by 2^3.5 at fixed (L, R, M).
    """
    eta, lum, rad, mas = 8.0e-14, 500.0, 43.0, 0.43
    got = float(_sc05(lum, rad, mas, 4000.0, eta=eta))
    assert got == pytest.approx(2.0 * eta * lum * rad / mas, rel=_RTOL)
    ratio = float(_sc05(lum, rad, mas, 5000.0, eta=eta)) / got
    assert ratio == pytest.approx(1.25**3.5, rel=_RTOL)


@pytest.mark.unit
def test_sc05_reduces_to_reimers_form_without_the_gravity_term():
    """A mismatch between the two laws' L R/M structure.

    As R^2/M -> 0 (compact, high-gravity limit: R = 1e-3, M = 1) the bracket
    tends to 1 and SC05 at Teff = 4000 K equals eta_SC L R/M, the Reimers form
    with eta * 4e-13 = eta_SC. The deviation is R^2/(4300 M) = 2.3e-10.
    """
    lum, rad, mas = 10.0, 1.0e-3, 1.0
    sc = float(_sc05(lum, rad, mas, 4000.0, eta=8.0e-14))
    reimers = float(_reimers(lum, rad, mas, eta=8.0e-14 / 4.0e-13))
    assert sc == pytest.approx(reimers, rel=1.0e-9)


@pytest.mark.unit
def test_sc05_gradient_in_eta_is_the_unit_rate():
    """A rate nonlinear in eta: d(rate)/d(eta) equals the rate at eta = 1 Msun/yr."""
    args = (800.0, 90.0, 1.2, 3800.0)
    grad = jax.grad(lambda e: schroeder_cuntz05_mdot(*args, eta=e))(8.0e-14)
    unit_rate = float(_sc05(*args, eta=1.0))
    assert float(grad) == pytest.approx(unit_rate, rel=_RTOL)
