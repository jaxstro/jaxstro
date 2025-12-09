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

    def test_default_is_dynamical(self):
        """DEFAULT should be ASTRO_DYNAMICAL."""
        assert U.DEFAULT is U.ASTRO_DYNAMICAL


# =============================================================================
# Phase 0 TDD Tests - NEW FEATURES
# =============================================================================


class TestGravitationalConstant:
    """Tests for the G property (computed from CGS constants)."""

    def test_cgs_g_value(self):
        """CGS.G should equal G_CGS = 6.67430e-8 cm³ g⁻¹ s⁻²."""
        assert abs(U.CGS.G - C.G_CGS) < 1e-15

    def test_stellar_g_value(self):
        """STELLAR.G should be ~0.00450 (pc³ Msun⁻¹ Myr⁻²)."""
        # G_code = G_CGS * mass_cgs * time_cgs² / length_cgs³
        expected = C.G_CGS * C.MSUN_G * (C.MYR_S ** 2) / (C.PC_CM ** 3)
        assert abs(U.STELLAR.G - expected) < 1e-10
        # Also verify approximate value
        assert abs(U.STELLAR.G - 0.00450) < 1e-4

    def test_binary_g_value(self):
        """BINARY.G should be ~39.48 (AU³ Msun⁻¹ yr⁻²)."""
        # G_code = G_CGS * mass_cgs * time_cgs² / length_cgs³
        expected = C.G_CGS * C.MSUN_G * (C.YR_S ** 2) / (C.AU_CM ** 3)
        assert abs(U.BINARY.G - expected) < 1e-10
        # Should match 4π² (Kepler's 3rd law)
        assert abs(U.BINARY.G - 39.4784) < 0.01

    def test_solar_g_value(self):
        """SOLAR.G should be computed correctly (Rsun³ Msun⁻¹ Myr⁻²)."""
        # SOLAR = ASTRO_STELLAR (Rsun, Msun, Myr)
        # G_code = G_CGS * mass_cgs * time_cgs² / length_cgs³
        expected = C.G_CGS * C.MSUN_G * (C.MYR_S ** 2) / (C.RSUN_CM ** 3)
        assert abs(U.SOLAR.G - expected) < 1e-10
        # Sanity check - G is ~3.9e20 for (Rsun, Msun, Myr)
        assert U.SOLAR.G > 0
        assert abs(U.SOLAR.G - 3.925e20) < 1e18


class TestSingleQuantityConversions:
    """Tests for convert_length, convert_mass, convert_time, convert_velocity methods."""

    def test_convert_length_stellar_to_cgs(self):
        """1 pc should convert to ~3.086e18 cm."""
        r_cm = U.STELLAR.convert_length(1.0, to=U.CGS)
        assert abs(r_cm - C.PC_CM) < 1e12

    def test_convert_length_binary_to_cgs(self):
        """1 AU should convert to ~1.496e13 cm."""
        r_cm = U.BINARY.convert_length(1.0, to=U.CGS)
        assert abs(r_cm - C.AU_CM) < 1e8

    def test_convert_mass_stellar_to_cgs(self):
        """1 Msun should convert to ~1.9884e33 g."""
        m_g = U.STELLAR.convert_mass(1.0, to=U.CGS)
        assert abs(m_g - C.MSUN_G) < 1e28

    def test_convert_time_stellar_to_cgs(self):
        """1 Myr should convert to ~3.156e13 s."""
        t_s = U.STELLAR.convert_time(1.0, to=U.CGS)
        assert abs(t_s - C.MYR_S) < 1e8

    def test_convert_time_binary_to_cgs(self):
        """1 yr should convert to ~3.156e7 s."""
        t_s = U.BINARY.convert_time(1.0, to=U.CGS)
        assert abs(t_s - C.YR_S) < 1e2

    def test_convert_velocity_stellar_to_cgs(self):
        """1 pc/Myr should convert to pc_cm/myr_s cm/s."""
        v_cgs = U.STELLAR.convert_velocity(1.0, to=U.CGS)
        expected = C.PC_CM / C.MYR_S
        assert abs(v_cgs - expected) < 1e-5

    def test_roundtrip_length_conversion(self):
        """Convert stellar→cgs→stellar should be identity."""
        r_original = 10.0
        r_cgs = U.STELLAR.convert_length(r_original, to=U.CGS)
        r_back = U.CGS.convert_length(r_cgs, to=U.STELLAR)
        assert abs(r_back - r_original) < 1e-10

    def test_roundtrip_mass_conversion(self):
        """Convert stellar→cgs→stellar should be identity."""
        m_original = 5.0
        m_cgs = U.STELLAR.convert_mass(m_original, to=U.CGS)
        m_back = U.CGS.convert_mass(m_cgs, to=U.STELLAR)
        assert abs(m_back - m_original) < 1e-10

    def test_convert_between_systems(self):
        """Convert length from STELLAR to BINARY."""
        # 1 pc in AU
        r_au = U.STELLAR.convert_length(1.0, to=U.BINARY)
        expected = C.PC_CM / C.AU_CM  # ~206265 AU/pc
        assert abs(r_au - expected) < 1


class TestShortAliases:
    """Tests for short alias names: STELLAR, STAR, BINARY, SOLAR, PLANETARY."""

    def test_stellar_is_astro_dynamical(self):
        """STELLAR should alias ASTRO_DYNAMICAL (pc, Msun, Myr)."""
        assert U.STELLAR is U.ASTRO_DYNAMICAL

    def test_star_is_astro_stellar(self):
        """STAR should alias ASTRO_STELLAR (Rsun, Msun, Myr) for startrax/stellax."""
        assert U.STAR is U.ASTRO_STELLAR

    def test_binary_is_astro_planetary(self):
        """BINARY should alias ASTRO_PLANETARY (AU, Msun, yr)."""
        assert U.BINARY is U.ASTRO_PLANETARY

    def test_solar_is_astro_stellar(self):
        """SOLAR should alias ASTRO_STELLAR (Rsun, Msun, Myr)."""
        assert U.SOLAR is U.ASTRO_STELLAR

    def test_planetary_is_binary(self):
        """PLANETARY should alias BINARY."""
        assert U.PLANETARY is U.BINARY


class TestGetUnits:
    """Tests for get_units() lookup function."""

    def test_get_stellar(self):
        """get_units('stellar') should return STELLAR."""
        assert U.get_units("stellar") is U.STELLAR

    def test_get_star(self):
        """get_units('star') should return STAR (for startrax/stellax)."""
        assert U.get_units("star") is U.STAR

    def test_get_binary(self):
        """get_units('binary') should return BINARY."""
        assert U.get_units("binary") is U.BINARY

    def test_get_cgs(self):
        """get_units('cgs') should return CGS."""
        assert U.get_units("cgs") is U.CGS

    def test_get_solar(self):
        """get_units('solar') should return SOLAR."""
        assert U.get_units("solar") is U.SOLAR

    def test_case_insensitive_uppercase(self):
        """get_units should be case-insensitive (STELLAR)."""
        assert U.get_units("STELLAR") is U.STELLAR

    def test_case_insensitive_mixed(self):
        """get_units should be case-insensitive (Stellar)."""
        assert U.get_units("Stellar") is U.STELLAR

    def test_invalid_raises_keyerror(self):
        """get_units with invalid name should raise KeyError."""
        with pytest.raises(KeyError):
            U.get_units("invalid_unit_system")

    def test_get_astro_dynamical(self):
        """get_units('astro_dynamical') should work."""
        assert U.get_units("astro_dynamical") is U.ASTRO_DYNAMICAL


class TestUnitSystemsRegistry:
    """Tests for UNIT_SYSTEMS dictionary."""

    def test_unit_systems_contains_stellar(self):
        """UNIT_SYSTEMS should contain 'stellar'."""
        assert "stellar" in U.UNIT_SYSTEMS

    def test_unit_systems_contains_binary(self):
        """UNIT_SYSTEMS should contain 'binary'."""
        assert "binary" in U.UNIT_SYSTEMS

    def test_unit_systems_contains_cgs(self):
        """UNIT_SYSTEMS should contain 'cgs'."""
        assert "cgs" in U.UNIT_SYSTEMS

    def test_unit_systems_values_are_unit_systems(self):
        """All values in UNIT_SYSTEMS should be UnitSystem instances."""
        for name, system in U.UNIT_SYSTEMS.items():
            assert isinstance(system, U.UnitSystem), f"{name} is not a UnitSystem"
