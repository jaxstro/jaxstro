# tests/test_units.py
"""
Tests for jaxstro.units module.

Verifies UnitSystem functionality and predefined unit systems.
"""

import pytest

from jaxstro import constants as C
from jaxstro import units as U


class TestUnitSystem:
    """Tests for the UnitSystem dataclass."""

    def test_cgs_identity(self):
        """CGS system should have all scales = 1."""
        assert U.CGS.mass_scale_cgs == 1.0
        assert U.CGS.length_scale_cgs == 1.0
        assert U.CGS.time_scale_cgs == 1.0

    def test_velocity_scale(self):
        """velocity_scale_cgs should be length/time."""
        for us in [U.CGS, U.ASTRO_STELLAR, U.ASTRO_DYNAMICAL, U.ASTRO_PLANETARY]:
            expected = us.length_scale_cgs / us.time_scale_cgs
            assert us.velocity_scale_cgs == expected

    def test_velocity_scale_km_s(self):
        """velocity_scale_km_s should be velocity_scale_cgs / 1e5."""
        for us in [U.CGS, U.ASTRO_STELLAR, U.ASTRO_DYNAMICAL, U.ASTRO_PLANETARY]:
            expected = us.velocity_scale_cgs / C.KM_CM
            assert us.velocity_scale_km_s == expected


class TestToFromCGS:
    """Tests for unit conversion methods."""

    def test_to_cgs_identity(self):
        """CGS to_cgs should be identity."""
        m, r, t = U.CGS.to_cgs(1.0, 1.0, 1.0)
        assert (m, r, t) == (1.0, 1.0, 1.0)

    def test_from_cgs_identity(self):
        """CGS from_cgs should be identity."""
        m, r, t = U.CGS.from_cgs(1.0, 1.0, 1.0)
        assert (m, r, t) == (1.0, 1.0, 1.0)

    def test_to_cgs_stellar(self):
        """1 Msun, 1 Rsun, 1 Myr in CGS."""
        m, r, t = U.ASTRO_STELLAR.to_cgs(1.0, 1.0, 1.0)
        assert m == C.MSUN_G
        assert r == C.RSUN_CM
        assert t == C.MYR_S

    def test_from_cgs_stellar(self):
        """Convert CGS values to stellar units."""
        m, r, t = U.ASTRO_STELLAR.from_cgs(C.MSUN_G, C.RSUN_CM, C.MYR_S)
        assert m == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert t == pytest.approx(1.0)

    def test_roundtrip(self):
        """to_cgs and from_cgs should be inverses."""
        for us in [U.ASTRO_STELLAR, U.ASTRO_DYNAMICAL, U.ASTRO_PLANETARY]:
            m_orig, r_orig, t_orig = 2.5, 10.0, 0.1
            m_cgs, r_cgs, t_cgs = us.to_cgs(m_orig, r_orig, t_orig)
            m_back, r_back, t_back = us.from_cgs(m_cgs, r_cgs, t_cgs)
            assert m_back == pytest.approx(m_orig)
            assert r_back == pytest.approx(r_orig)
            assert t_back == pytest.approx(t_orig)


class TestPredefinedSystems:
    """Tests for predefined unit systems."""

    def test_astro_stellar_scales(self):
        """ASTRO_STELLAR should use Msun, Rsun, Myr."""
        assert U.ASTRO_STELLAR.mass_scale_cgs == C.MSUN_G
        assert U.ASTRO_STELLAR.length_scale_cgs == C.RSUN_CM
        assert U.ASTRO_STELLAR.time_scale_cgs == C.MYR_S

    def test_astro_dynamical_scales(self):
        """ASTRO_DYNAMICAL should use Msun, pc, Myr."""
        assert U.ASTRO_DYNAMICAL.mass_scale_cgs == C.MSUN_G
        assert U.ASTRO_DYNAMICAL.length_scale_cgs == C.PC_CM
        assert U.ASTRO_DYNAMICAL.time_scale_cgs == C.MYR_S

    def test_astro_planetary_scales(self):
        """ASTRO_PLANETARY should use Msun, AU, yr."""
        assert U.ASTRO_PLANETARY.mass_scale_cgs == C.MSUN_G
        assert U.ASTRO_PLANETARY.length_scale_cgs == C.AU_CM
        assert U.ASTRO_PLANETARY.time_scale_cgs == C.YR_S

    def test_unit_labels(self):
        """Unit systems should have correct labels."""
        assert U.ASTRO_STELLAR.mass_unit == "Msun"
        assert U.ASTRO_STELLAR.length_unit == "Rsun"
        assert U.ASTRO_STELLAR.time_unit == "Myr"

        assert U.ASTRO_DYNAMICAL.length_unit == "pc"
        assert U.ASTRO_PLANETARY.length_unit == "AU"
        assert U.ASTRO_PLANETARY.time_unit == "yr"


class TestUnitSystemImmutability:
    """Tests for UnitSystem immutability."""

    def test_frozen(self):
        """UnitSystem should be frozen (immutable)."""
        with pytest.raises(AttributeError):
            U.CGS.mass_scale_cgs = 2.0

    def test_hashable(self):
        """Frozen dataclass should be hashable."""
        # Should not raise
        hash(U.CGS)
        hash(U.ASTRO_STELLAR)


class TestDefaultSystem:
    """Tests for the DEFAULT unit system."""

    def test_default_is_cgs(self):
        """DEFAULT should be CGS."""
        assert U.DEFAULT is U.CGS
