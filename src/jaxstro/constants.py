# src/jaxstro/constants.py

"""
Physical constants for the jaxstro ecosystem (CGS units).

These values follow IAU / CODATA recommendations and are intended
to be the single source of truth for all downstream packages.

Units:
    - Length: cm
    - Mass:   g
    - Time:   s
    - Energy: erg = g·cm²/s²
    - Pressure: dyn/cm² = g/(cm·s²)

References:
    - CODATA 2018: Tiesinga et al. (2021), Rev. Mod. Phys., 93, 025010;
      archived by NIST at https://physics.nist.gov/cuu/Constants/archive2018.html
    - IAU 2012 B2: astronomical unit (AU)
    - IAU 2015 B3: nominal solar conversion constants (R_sun, L_sun,
      T_eff,sun, and (GM)_sun; not a solar mass in grams),
      https://www.iau.org/common/Uploaded%20files/IAUGA2015-Resolution-B3-recommended-nominal-conversion.pdf
    - IAU 2015 B2: bolometric-magnitude zero points,
      https://www.iau.org/static/resolutions/IAU2015_English.pdf
    - Asplund et al. (2009), ARA&A, 47, 481: Solar composition
"""

import math

# ===========================================================================
# Fundamental constants (CODATA 2018)
# ===========================================================================

# Gravitational constant [cm³ g⁻¹ s⁻²]
G_CGS: float = 6.67430e-8

# Speed of light [cm/s] (exact)
C_CGS: float = 2.99792458e10

# Planck constant [erg·s] (exact)
H_CGS: float = 6.62607015e-27

# Boltzmann constant [erg/K] (exact)
K_B: float = 1.380649e-16

# Reduced Planck constant ħ = h/(2π) [erg·s]. Exact, since h is exact.
HBAR_CGS: float = H_CGS / (2.0 * math.pi)

# A_RAD moved to the "Radiation / thermodynamics" section below, where it is
# derived from SIGMA_SB (which must be defined first). It is still exported
# under the same name.

# Avogadro's number [mol⁻¹]
N_A: float = 6.02214076e23

# Electron volt [erg]
EV_ERG: float = 1.602176634e-12

# ===========================================================================
# Electromagnetic and atomic-physics constants (CODATA 2018)
# Tiesinga et al. (2021), Rev. Mod. Phys., 93, 025010.
# ===========================================================================

# Fine-structure constant α [dimensionless].
# CODATA 2018: 7.2973525693(11) × 10⁻³ (no unit conversion needed).
ALPHA_FS: float = 7.2973525693e-3

# Elementary charge in CGS-Gaussian units [statC = esu = g^½ cm^(3/2) s⁻¹].
# CODATA 2018 e = 1.602176634e-19 C (exact). Gaussian-CGS conversion:
# e_esu = e_SI × c_cgs / 10 = 1.602176634e-19 × 2.99792458e10 / 10
#       = 4.803204712570…e-10 statC (the standard 4.80320471e-10 esu).
E_ESU: float = 4.803204712570263e-10

# Classical electron radius r_e = e²/(m_e c²) [cm].
# CODATA 2018: 2.8179403262(13) × 10⁻¹⁵ m = 2.8179403262e-13 cm (×1e2 for m→cm).
R_E: float = 2.8179403262e-13

# Thomson cross-section σ_T = (8π/3) r_e² [cm²].
# CODATA 2018: 6.6524587321(60) × 10⁻²⁹ m² = 6.6524587321e-25 cm²
# (×1e4 for the m²→cm² conversion).
SIGMA_T: float = 6.6524587321e-25

# Molar gas constant R = k_B N_A [erg mol⁻¹ K⁻¹].
#
# DERIVED. k_B and N_A are both exact in the revised SI, so their product is
# exact and this is R's definition. CODATA 2018 publishes 8.314462618
# J mol⁻¹ K⁻¹, rounded for print; the product gives 8.31446261815324e7
# erg mol⁻¹ K⁻¹, matching to every published digit.
#
# **Was the literal 8.314462618e7 until 2026-07-30.** Its own comment already
# recorded that it "is not bit-identical to K_B * N_A" — the third constant in
# this module documenting its own imprecision at the callsite and keeping it.
# Same fix, same reason, as SIGMA_SB and A_RAD below.
R_GAS: float = K_B * N_A

# ===========================================================================
# Atomic and particle masses (CODATA 2018)
# ===========================================================================

# Atomic mass unit [g]
M_U: float = 1.66053906660e-24

# Electron mass [g]
M_E: float = 9.1093837015e-28

# Proton mass [g]
M_P: float = 1.67262192369e-24

# Neutron mass [g]
M_N: float = 1.67492749804e-24

# Bohr radius [cm], CODATA 2018: a_0 = 5.29177210903(80)e-9 cm.
# A literal rather than hbar^2/(M_E * E_ESU^2): the derivation reproduces
# CODATA only to the rounding of its inputs, and both existing consumers
# (micrax occupation + coulomb, previously two private copies) carry exactly
# these digits — the literal keeps the consolidation 0 ULP.
A_0_CGS: float = 5.29177210903e-9

# ===========================================================================
# Solar parameters and compatibility scales
# IAU 2015 Resolution B3 (direct locator):
# https://www.iau.org/common/Uploaded%20files/IAUGA2015-Resolution-B3-recommended-nominal-conversion.pdf
# B3 defines exact nominal conversion constants R_sun^N, L_sun^N,
# T_eff,sun^N, and (GM)_sun^N. They are conversion factors, not estimates of
# the true solar properties, and B3 does not define a nominal solar mass [g].
# ===========================================================================

# Legacy solar-mass compatibility scale [g]. It is the rounded result of
# (GM)_sun^N / G_CGS using B3's exact 1.3271244e20 m^3 s^-2 and this module's
# frozen CODATA-2018 G_CGS; it is not an IAU nominal solar mass.
MSUN_G: float = 1.9884e33
# IAU B3 nominal conversion constants, converted exactly from their SI forms.
RSUN_CM: float = 6.957e10
LSUN_ERG_S: float = 3.828e33
TEFF_SUN: float = 5772.0

# Conventionally rounded solar absolute bolometric magnitude [mag].
# IAU 2015 Resolution B2 fixes the absolute-bolometric zero point at
# L_0 = 3.0128e28 W (M_bol = 0 at L_0). With B3's nominal luminosity
# L_sun^N = 3.828e26 W, -2.5 * log10(L_sun^N / L_0) =
# 4.7399959339...; M_BOL_SUN retains the customary 4.74 rounding.
# B2 locator (Resolution B2):
# https://www.iau.org/static/resolutions/IAU2015_English.pdf
M_BOL_SUN: float = 4.74

# ===========================================================================
# Solar composition (Asplund et al. 2009)
# ===========================================================================

X_SUN: float = 0.7381  # Solar hydrogen mass fraction
Y_SUN: float = 0.2485  # Solar helium mass fraction
Z_SUN: float = 0.0134  # Solar metals mass fraction

# ===========================================================================
# Astronomical distance and time units
# ===========================================================================

# Parsec [cm] (from AU and arcsec definition)
PC_CM: float = 3.0856775814913673e18

# Astronomical unit [cm] (IAU 2012 definition: 149,597,870,700 m)
AU_CM: float = 1.495978707e13

# Time units — JULIAN year (IAU): 1 yr = 365.25 d × 86400 s = 31557600 s exactly.
# (This is the Julian year, NOT the tropical year ≈ 365.2422 d; the IAU defines
# the Julian year/century for ephemerides and the light-year.)
YR_S: float = 3.15576e7  # s in 1 Julian yr = 365.25 × 86400 = 31557600 s exactly
MYR_S: float = 3.15576e13  # s in 1 Myr = 1e6 Julian yr = 1e6 × 31557600 s

# Metric conversion
KM_CM: float = 1.0e5  # 1 km = 1e5 cm

# ===========================================================================
# Radiation / thermodynamics
# ===========================================================================

# Stefan–Boltzmann constant σ = 2π⁵k⁴/(15h³c²) [erg cm⁻² s⁻¹ K⁻⁴].
#
# DERIVED, not quoted. Since the 2019 SI redefinition h, k and c are all exact,
# so σ is exact too and this expression is its definition rather than a fit to
# it. CODATA 2018 publishes 5.670374419e-8 W m⁻² K⁻⁴ (Tiesinga et al. 2021,
# Rev. Mod. Phys. 93, 025010), rounded for print; the expression gives
# 5.670374419184e-5 erg cm⁻² s⁻¹ K⁻⁴, matching to every published digit.
#
# **Was the literal 5.670374419e-5 until 2026-07-30**, i.e. the published value
# truncated at ten digits, 3.3e-11 below exact. Changed because this module
# declares itself the ecosystem's single source of truth, so a downstream package
# that derives σ correctly ends up disagreeing with the source of truth — which
# is what happened: `stellax.core.constants` was corrected to derive on the same
# day and then differed from jaxstro by 3e-11 in `A_RAD`. Deriving in BOTH places
# is the only arrangement where they cannot disagree.
SIGMA_SB: float = 2.0 * math.pi**5 * K_B**4 / (15.0 * H_CGS**3 * C_CGS**2)

# Radiation constant a = 4σ/c [erg cm⁻³ K⁻⁴]. Derived from the σ above, so it
# cannot drift from it. Defined here rather than with the fundamental constants
# because it is a radiation quantity and because σ must exist first.
#
# **Was the literal 7.565733250e-15 until 2026-07-30.** The comment at its old
# site already said "the compatibility literal is rounded to its stored decimal
# precision; it is not bit-identical to the expression below" — the defect was
# documented at the callsite and left in place. Exact value 7.565733250280009e-15.
A_RAD: float = 4.0 * SIGMA_SB / C_CGS

# ===========================================================================
# Photometric units and zeropoints
# ===========================================================================

# Jansky [erg s⁻¹ cm⁻² Hz⁻¹ per Jy].
# Definition: 1 Jy = 1e-26 W m⁻² Hz⁻¹ (SI) = 1e-23 erg s⁻¹ cm⁻² Hz⁻¹ (CGS),
# since 1 W = 1e7 erg/s and 1 m⁻² = 1e-4 cm⁻² → 1e7 × 1e-4 × 1e-26 = 1e-23.
JY_CGS: float = 1e-23

# AB magnitude system zeropoint flux density [Jy].
# f_AB = 3631 Jy at AB mag 0 (Oke & Gunn 1983, ApJ, 266, 713).
AB_ZEROPOINT_JY: float = 3631.0

# AB zeropoint flux density in CGS [erg s⁻¹ cm⁻² Hz⁻¹].
# = AB_ZEROPOINT_JY × JY_CGS = 3631 × 1e-23 (Oke & Gunn 1983, ApJ, 266, 713).
AB_ZEROPOINT_CGS: float = AB_ZEROPOINT_JY * JY_CGS  # 3.631e-20

# ===========================================================================
# Derived velocity conversions
# ===========================================================================

# 1 pc / Myr in km/s
PC_PER_MYR_TO_KM_PER_S: float = (PC_CM / MYR_S) / KM_CM

# 1 AU / yr in km/s
AU_PER_YR_TO_KM_PER_S: float = (AU_CM / YR_S) / KM_CM

# Inverse conversions
KM_PER_S_TO_PC_PER_MYR: float = 1.0 / PC_PER_MYR_TO_KM_PER_S
KM_PER_S_TO_AU_PER_YR: float = 1.0 / AU_PER_YR_TO_KM_PER_S

# ===========================================================================
# Public API
# ===========================================================================

__all__ = [
    # Fundamental constants
    "G_CGS",
    "C_CGS",
    "H_CGS",
    "HBAR_CGS",
    "K_B",
    "A_RAD",
    "N_A",
    "EV_ERG",
    # Electromagnetic and atomic-physics constants
    "ALPHA_FS",
    "E_ESU",
    "R_E",
    "SIGMA_T",
    "R_GAS",
    # Particle masses
    "M_U",
    "M_E",
    "M_P",
    "M_N",
    # Solar parameters
    "MSUN_G",
    "RSUN_CM",
    "LSUN_ERG_S",
    "TEFF_SUN",
    "M_BOL_SUN",
    # Solar composition
    "X_SUN",
    "Y_SUN",
    "Z_SUN",
    # Distance and time units
    "PC_CM",
    "AU_CM",
    "MYR_S",
    "YR_S",
    "KM_CM",
    # Radiation
    "SIGMA_SB",
    # Photometric units and zeropoints
    "JY_CGS",
    "AB_ZEROPOINT_JY",
    "AB_ZEROPOINT_CGS",
    # Velocity conversions
    "PC_PER_MYR_TO_KM_PER_S",
    "AU_PER_YR_TO_KM_PER_S",
    "KM_PER_S_TO_PC_PER_MYR",
    "KM_PER_S_TO_AU_PER_YR",
]
