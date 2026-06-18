# tests/test_astrometry.py
"""
Tests for jaxstro.astrometry module.

Verifies astrometric constants and conversions.
"""

import math

import pytest

from jaxstro import astrometry as A
from jaxstro import constants as C


class TestAstrometricConstants:
    """Tests for astrometric constants."""

    def test_km_per_pc(self):
        """KM_PER_PC should be consistent with PC_CM."""
        expected = C.PC_CM / C.KM_CM
        assert A.KM_PER_PC == pytest.approx(expected)

    def test_mas_per_rad(self):
        """MAS_PER_RAD should be 1e3 * 3600 * 180 / pi."""
        expected = 1e3 * 3600 * 180 / math.pi
        assert A.MAS_PER_RAD == pytest.approx(expected, rel=1e-10)

    def test_arcsec_per_rad(self):
        """ARCSEC_PER_RAD should be 3600 * 180 / pi."""
        expected = 3600 * 180 / math.pi
        assert A.ARCSEC_PER_RAD == pytest.approx(expected, rel=1e-10)

    def test_deg_per_rad(self):
        """DEG_PER_RAD should be 180 / pi."""
        expected = 180 / math.pi
        assert A.DEG_PER_RAD == pytest.approx(expected, rel=1e-10)

    def test_yr_per_myr(self):
        """YR_PER_MYR should be exactly 1e6."""
        assert A.YR_PER_MYR == 1e6


class TestProperMotionConstant:
    """Tests for the proper motion constant K."""

    def test_k_proper_motion_value(self):
        """K should be approximately 4.74 km/s per (mas/yr * kpc)."""
        assert A.K_PROPER_MOTION == pytest.approx(4.74047, rel=1e-4)

    def test_k_proper_motion_derivation(self):
        """K = 1 AU/yr in km/s (approximately)."""
        # 1 mas/yr at 1 kpc = 1 AU/yr transverse velocity
        # 1 AU/yr ≈ 4.74 km/s
        assert A.K_PROPER_MOTION == pytest.approx(C.AU_PER_YR_TO_KM_PER_S, rel=1e-3)

    def test_proper_motion_example(self):
        """Verify proper motion conversion for a known case."""
        # A star at 1 kpc with proper motion 1 mas/yr
        # should have transverse velocity ~4.74 km/s
        mu_mas_yr = 1.0
        d_kpc = 1.0
        v_kms = mu_mas_yr * A.K_PROPER_MOTION * d_kpc
        assert v_kms == pytest.approx(4.74, rel=1e-2)


class TestAngularConversions:
    """Tests for angular unit consistency."""

    def test_mas_arcsec_consistency(self):
        """1000 mas = 1 arcsec."""
        assert A.MAS_PER_RAD == pytest.approx(1000 * A.ARCSEC_PER_RAD)

    def test_arcsec_deg_consistency(self):
        """3600 arcsec = 1 degree."""
        assert A.ARCSEC_PER_RAD == pytest.approx(3600 * A.DEG_PER_RAD)

    def test_full_circle(self):
        """360 degrees = 2*pi radians."""
        degrees_per_circle = 360.0
        radians_per_circle = 2 * math.pi
        assert A.DEG_PER_RAD == pytest.approx(degrees_per_circle / radians_per_circle)


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_constants_accessible(self):
        """All expected constants should be accessible."""
        expected = [
            "KM_PER_PC",
            "MAS_PER_RAD",
            "ARCSEC_PER_RAD",
            "DEG_PER_RAD",
            "YR_PER_MYR",
            "K_PROPER_MOTION",
        ]
        for name in expected:
            assert hasattr(A, name), f"Missing: {name}"
