# tests/test_constants.py
"""
Tests for jaxstro.constants module.

Verifies physical constants against authoritative values and
checks internal consistency of derived quantities.
"""

import math

import pytest

from jaxstro import constants as C


class TestFundamentalConstants:
    """Tests for fundamental physical constants."""

    def test_gravitational_constant(self):
        """G should match CODATA 2018 value."""
        # CODATA 2018: 6.67430(15) × 10^-8 cm³ g⁻¹ s⁻²
        assert C.G_CGS == pytest.approx(6.67430e-8, rel=1e-5)

    def test_speed_of_light(self):
        """c should be exactly 299792458 m/s = 2.99792458e10 cm/s."""
        assert C.C_CGS == 2.99792458e10  # Exact by definition

    def test_planck_constant(self):
        """h should match CODATA 2018 exact value."""
        assert C.H_CGS == 6.62607015e-27  # Exact by definition

    def test_boltzmann_constant(self):
        """k_B should match CODATA 2018 exact value."""
        assert C.K_B == 1.380649e-16  # Exact by definition

    def test_stefan_boltzmann_constant(self):
        """sigma_SB should be consistent with k_B, h, c."""
        # sigma = 2 * pi^5 * k^4 / (15 * h^3 * c^2)
        sigma_computed = (
            2 * math.pi**5 * C.K_B**4 / (15 * C.H_CGS**3 * C.C_CGS**2)
        )
        assert C.SIGMA_SB == pytest.approx(sigma_computed, rel=1e-6)

    def test_radiation_constant(self):
        """a_rad should be 4 * sigma_SB / c."""
        a_computed = 4 * C.SIGMA_SB / C.C_CGS
        assert C.A_RAD == pytest.approx(a_computed, rel=1e-6)


class TestParticleMasses:
    """Tests for particle mass constants."""

    def test_atomic_mass_unit(self):
        """m_u should match CODATA 2018."""
        assert C.M_U == pytest.approx(1.66053906660e-24, rel=1e-10)

    def test_electron_mass(self):
        """m_e should match CODATA 2018."""
        assert C.M_E == pytest.approx(9.1093837015e-28, rel=1e-10)

    def test_proton_mass(self):
        """m_p should match CODATA 2018."""
        assert C.M_P == pytest.approx(1.67262192369e-24, rel=1e-10)

    def test_neutron_mass(self):
        """m_n should match CODATA 2018."""
        assert C.M_N == pytest.approx(1.67492749804e-24, rel=1e-10)

    def test_proton_heavier_than_electron(self):
        """Proton should be ~1836 times heavier than electron."""
        ratio = C.M_P / C.M_E
        assert ratio == pytest.approx(1836.15, rel=1e-4)


class TestSolarParameters:
    """Tests for solar constants."""

    def test_solar_mass(self):
        """M_sun should match IAU 2015 nominal value."""
        # IAU 2015 B3: GM_sun = 1.3271244e26 cm³/s², so M = GM/G
        assert C.MSUN_G == pytest.approx(1.9884e33, rel=1e-4)

    def test_solar_radius(self):
        """R_sun should match IAU 2015 nominal value."""
        assert C.RSUN_CM == pytest.approx(6.957e10, rel=1e-4)

    def test_solar_luminosity(self):
        """L_sun should match IAU 2015 nominal value."""
        assert C.LSUN_ERG_S == pytest.approx(3.828e33, rel=1e-4)

    def test_solar_temperature(self):
        """T_eff_sun should be 5772 K (IAU 2015)."""
        assert C.TEFF_SUN == 5772.0

    def test_stefan_boltzmann_consistency(self):
        """L_sun should be consistent with R_sun, T_sun via Stefan-Boltzmann."""
        # L = 4 * pi * R^2 * sigma * T^4
        L_computed = 4 * math.pi * C.RSUN_CM**2 * C.SIGMA_SB * C.TEFF_SUN**4
        # Allow ~0.5% tolerance due to nominal value definitions
        assert C.LSUN_ERG_S == pytest.approx(L_computed, rel=0.005)


class TestSolarComposition:
    """Tests for solar composition (Asplund et al. 2009)."""

    def test_composition_sums_to_unity(self):
        """X + Y + Z should equal 1."""
        total = C.X_SUN + C.Y_SUN + C.Z_SUN
        assert total == pytest.approx(1.0, rel=1e-10)

    def test_hydrogen_fraction(self):
        """X_sun should be ~0.74."""
        assert C.X_SUN == pytest.approx(0.7381, rel=1e-4)

    def test_helium_fraction(self):
        """Y_sun should be ~0.25."""
        assert C.Y_SUN == pytest.approx(0.2485, rel=1e-4)

    def test_metals_fraction(self):
        """Z_sun should be ~0.013."""
        assert C.Z_SUN == pytest.approx(0.0134, rel=1e-4)


class TestDistanceUnits:
    """Tests for distance unit conversions."""

    def test_parsec(self):
        """1 pc should be ~3.086e18 cm."""
        assert C.PC_CM == pytest.approx(3.0857e18, rel=1e-4)

    def test_au(self):
        """1 AU should be exactly 149597870700 m = 1.495978707e13 cm."""
        assert C.AU_CM == 1.495978707e13  # Exact by IAU definition

    def test_parsec_from_au(self):
        """1 pc = 1 AU / tan(1 arcsec) ≈ 206265 AU."""
        au_per_pc = C.PC_CM / C.AU_CM
        assert au_per_pc == pytest.approx(206265, rel=1e-4)


class TestTimeUnits:
    """Tests for time unit conversions."""

    def test_year(self):
        """1 tropical year should be ~3.1558e7 s."""
        assert C.YR_S == pytest.approx(3.15576e7, rel=1e-4)

    def test_megayear(self):
        """1 Myr should be 1e6 years."""
        assert C.MYR_S == C.YR_S * 1e6


class TestVelocityConversions:
    """Tests for velocity unit conversions."""

    def test_pc_per_myr_to_km_s(self):
        """1 pc/Myr should be ~0.978 km/s."""
        assert C.PC_PER_MYR_TO_KM_PER_S == pytest.approx(0.978, rel=1e-2)

    def test_au_per_yr_to_km_s(self):
        """1 AU/yr should be ~4.74 km/s."""
        assert C.AU_PER_YR_TO_KM_PER_S == pytest.approx(4.74, rel=1e-2)

    def test_velocity_conversion_inverse(self):
        """Forward and inverse conversions should be consistent."""
        assert C.PC_PER_MYR_TO_KM_PER_S * C.KM_PER_S_TO_PC_PER_MYR == pytest.approx(1.0)
        assert C.AU_PER_YR_TO_KM_PER_S * C.KM_PER_S_TO_AU_PER_YR == pytest.approx(1.0)

    def test_velocity_derived_correctly(self):
        """Velocity conversions should match distance/time ratios."""
        computed = (C.PC_CM / C.MYR_S) / C.KM_CM
        assert C.PC_PER_MYR_TO_KM_PER_S == computed
