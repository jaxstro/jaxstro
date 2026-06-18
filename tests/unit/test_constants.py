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
        sigma_computed = 2 * math.pi**5 * C.K_B**4 / (15 * C.H_CGS**3 * C.C_CGS**2)
        assert C.SIGMA_SB == pytest.approx(sigma_computed, rel=1e-6)

    def test_stefan_boltzmann_codata_value(self):
        """sigma_SB should match the CODATA 2018 CGS value exactly."""
        # CODATA 2018: 5.670374419e-8 W m^-2 K^-4 = 5.670374419e-5 erg cm^-2 s^-1 K^-4
        assert C.SIGMA_SB == 5.670374419e-5

    def test_radiation_constant(self):
        """a_rad should be 4 * sigma_SB / c (CODATA-derived, tight).

        Compares as a ratio to dodge pytest.approx's default abs=1e-12 floor,
        which would otherwise swamp these ~1e-14 magnitudes and make the check
        vacuous. A_RAD is stored to match 4*sigma/c to better than 1e-7.
        """
        a_computed = 4 * C.SIGMA_SB / C.C_CGS
        assert C.A_RAD / a_computed == pytest.approx(1.0, rel=1e-7)


class TestElectromagneticAndAtomicConstants:
    """Tests for electromagnetic / atomic-physics constants (CODATA 2018)."""

    def test_fine_structure_constant(self):
        """alpha should match CODATA 2018 value (dimensionless)."""
        # CODATA 2018: 7.2973525693(11) x 10^-3
        assert C.ALPHA_FS == pytest.approx(7.2973525693e-3, rel=1e-10)

    def test_thomson_cross_section(self):
        """sigma_T should match CODATA 2018 value in cm^2."""
        # CODATA 2018: 6.6524587321(60) x 10^-29 m^2 = 6.6524587321e-25 cm^2
        assert C.SIGMA_T == pytest.approx(6.6524587321e-25, rel=1e-9)

    def test_classical_electron_radius(self):
        """r_e should match CODATA 2018 value in cm."""
        # CODATA 2018: 2.8179403262(13) x 10^-15 m = 2.8179403262e-13 cm
        assert C.R_E == pytest.approx(2.8179403262e-13, rel=1e-9)

    def test_elementary_charge_esu(self):
        """e in CGS-Gaussian statcoulomb should be ~4.80320e-10."""
        # e_SI = 1.602176634e-19 C (exact); e_esu = e_SI * c_cgs / 10
        assert C.E_ESU == pytest.approx(4.80320471e-10, rel=1e-8)

    def test_elementary_charge_esu_derivation(self):
        """e_esu should equal e_SI * c_cgs / 10 (SI -> Gaussian-CGS)."""
        e_si = 1.602176634e-19
        assert C.E_ESU == pytest.approx(e_si * C.C_CGS / 10.0, rel=1e-12)

    def test_thomson_from_classical_radius(self):
        """sigma_T = (8/3) pi r_e^2 should reproduce the tabulated value."""
        sigma_computed = (8.0 / 3.0) * math.pi * C.R_E**2
        assert C.SIGMA_T == pytest.approx(sigma_computed, rel=1e-8)

    def test_molar_gas_constant(self):
        """R should match CODATA 2018 value in erg mol^-1 K^-1."""
        # CODATA 2018: 8.314462618 J mol^-1 K^-1 = 8.314462618e7 erg mol^-1 K^-1
        assert C.R_GAS == pytest.approx(8.314462618e7, rel=1e-10)

    def test_molar_gas_constant_from_k_b_n_a(self):
        """R = k_B * N_A (with the CGS erg-based k_B)."""
        assert C.R_GAS == pytest.approx(C.K_B * C.N_A, rel=1e-6)


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

    def test_m_bol_sun_value_and_consistency(self):
        """M_bol_sun should be 4.74 and consistent with the IAU 2015 zero point."""
        from jaxstro.constants import M_BOL_SUN

        assert M_BOL_SUN == 4.74
        # IAU 2015 Res B2: M_bol = -2.5 log10(L / L_0), L_0 = 3.0128e28 W;
        # nominal L_sun = 3.828e26 W
        L_sun_W, L_0_W = 3.828e26, 3.0128e28
        assert abs(M_BOL_SUN - (-2.5 * math.log10(L_sun_W / L_0_W))) < 5e-3


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

    def test_julian_year_exact_seconds(self):
        """YR_S is the JULIAN year: 365.25 d * 86400 s = 31557600 s exactly."""
        assert 365.25 * 86400 == 31557600.0  # arithmetic identity
        assert C.YR_S == 31557600.0  # exact, not tropical (365.2422 d)

    def test_year(self):
        """1 Julian year should be ~3.15576e7 s."""
        assert C.YR_S == pytest.approx(3.15576e7, rel=1e-4)

    def test_megayear_exact_seconds(self):
        """MYR_S is 1e6 Julian years = 1e6 * 31557600 = 3.15576e13 s exactly."""
        assert C.MYR_S == 31557600.0 * 1e6
        assert C.MYR_S == 3.15576e13

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
