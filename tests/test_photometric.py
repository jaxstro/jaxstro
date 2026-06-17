# tests/test_photometric.py

"""
Tests for photometric constants and the PhotometricUnits dataclass.

Covers:
    - photometric constants (Jy, AB zeropoint) with their numeric values,
    - PhotometricUnits resolving its luminosity/radius/flux-density choices
      to constant multiplies (host-side floats, mirroring UnitSystem),
    - presets SOLAR_PHOTOMETRIC / CGS_PHOTOMETRIC exist and are instances,
    - flux-density conversion through PhotometricUnits is differentiable
      (finite-difference vs jax.jacrev agree to ~1e-6 relative).
"""

import jax
import jax.numpy as jnp

from jaxstro import constants as C
from jaxstro.units import (
    CGS_PHOTOMETRIC,
    SOLAR_PHOTOMETRIC,
    PhotometricUnits,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_jansky_cgs_value():
    # 1 Jy = 1e-23 erg s^-1 cm^-2 Hz^-1 (= 1e-26 W m^-2 Hz^-1)
    assert C.JY_CGS == 1e-23


def test_ab_zeropoint_jy_value():
    # AB zeropoint = 3631 Jy (Oke & Gunn 1983)
    assert C.AB_ZEROPOINT_JY == 3631.0


def test_ab_zeropoint_cgs_value():
    # 3631 Jy * 1e-23 erg/s/cm^2/Hz/Jy = 3.631e-20 erg s^-1 cm^-2 Hz^-1
    assert C.AB_ZEROPOINT_CGS == 3.631e-20
    assert C.AB_ZEROPOINT_CGS == C.AB_ZEROPOINT_JY * C.JY_CGS


# ---------------------------------------------------------------------------
# PhotometricUnits scale resolution (host-side constant multiplies)
# ---------------------------------------------------------------------------


def test_solar_photometric_scales():
    u = SOLAR_PHOTOMETRIC
    assert isinstance(u, PhotometricUnits)
    assert u.luminosity == "Lsun"
    assert u.radius == "Rsun"
    assert u.flux_density == "Jy"
    # Resolved scale factors are the CGS conversion constants.
    assert u.luminosity_scale_cgs == C.LSUN_ERG_S
    assert u.radius_scale_cgs == C.RSUN_CM
    assert u.flux_scale_cgs == C.JY_CGS


def test_cgs_photometric_scales():
    u = CGS_PHOTOMETRIC
    assert isinstance(u, PhotometricUnits)
    assert u.luminosity == "cgs"
    assert u.radius == "cm"
    assert u.flux_density == "cgs"
    # All CGS -> scale factors are unity.
    assert u.luminosity_scale_cgs == 1.0
    assert u.radius_scale_cgs == 1.0
    assert u.flux_scale_cgs == 1.0


def test_luminosity_roundtrip_solar():
    u = SOLAR_PHOTOMETRIC
    x = 3.0  # 3 Lsun
    cgs = u.to_cgs_luminosity(x)
    # Forward multiply is a single exact op.
    assert cgs == 3.0 * C.LSUN_ERG_S
    # Roundtrip is (x*s)/s — not bit-exact in float, so use a tight tolerance.
    assert jnp.allclose(u.from_cgs_luminosity(cgs), x, rtol=1e-12)


def test_radius_roundtrip_solar():
    u = SOLAR_PHOTOMETRIC
    x = 2.0  # 2 Rsun
    cgs = u.to_cgs_radius(x)
    assert cgs == 2.0 * C.RSUN_CM
    assert jnp.allclose(u.from_cgs_radius(cgs), x, rtol=1e-12)


def test_flux_roundtrip_jy():
    u = SOLAR_PHOTOMETRIC
    x = 5.0  # 5 Jy
    cgs = u.to_cgs_flux(x)
    assert cgs == 5.0 * C.JY_CGS
    assert jnp.allclose(u.from_cgs_flux(cgs), x, rtol=1e-12)


def test_cgs_photometric_identity():
    u = CGS_PHOTOMETRIC
    x = 7.0
    assert u.to_cgs_luminosity(x) == x
    assert u.to_cgs_radius(x) == x
    assert u.to_cgs_flux(x) == x


def test_ab_flux_density_zeropoint():
    # An "AB" flux-density choice uses the AB zeropoint, not a linear Jy scale.
    u = PhotometricUnits(
        luminosity="Lsun", radius="Rsun", flux_density="AB", name="ab-test"
    )
    # AB mag <-> flux density (cgs) via f = zp_cgs * 10^(-0.4 m)
    m = jnp.asarray(0.0)
    f_cgs = u.ab_mag_to_cgs_flux(m)
    assert jnp.allclose(f_cgs, C.AB_ZEROPOINT_CGS)
    # Roundtrip
    m_back = u.cgs_flux_to_ab_mag(f_cgs)
    assert jnp.allclose(m_back, m, atol=1e-10)


# ---------------------------------------------------------------------------
# Differentiability (finite-difference vs jacrev), gate: gradient-validation
# ---------------------------------------------------------------------------


def test_flux_conversion_differentiable_linear():
    u = SOLAR_PHOTOMETRIC

    def f(x):
        # Jy -> cgs flux density: pure constant multiply
        return u.to_cgs_flux(x)

    x0 = jnp.asarray(4.2)
    ad = jax.jacrev(f)(x0)
    h = 1e-4
    fd = (f(x0 + h) - f(x0 - h)) / (2.0 * h)
    assert jnp.allclose(ad, fd, rtol=1e-6, atol=0.0)
    # Analytic slope is exactly JY_CGS
    assert jnp.allclose(ad, C.JY_CGS, rtol=1e-12)


def test_ab_flux_conversion_differentiable():
    u = PhotometricUnits(
        luminosity="Lsun", radius="Rsun", flux_density="AB", name="ab-test"
    )

    def f(m):
        return u.ab_mag_to_cgs_flux(m)

    m0 = jnp.asarray(18.0)
    ad = jax.jacrev(f)(m0)
    h = 1e-4
    fd = (f(m0 + h) - f(m0 - h)) / (2.0 * h)
    assert jnp.allclose(ad, fd, rtol=1e-6, atol=0.0)
