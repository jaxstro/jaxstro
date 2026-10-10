"""Owner tests of jaxstro.winds.hot (Vink et al. 2001 Eqs. 11, 14, 15, 23-25; Nugis & Lamers 2000 Eqs. 1, 22) and
jaxstro.winds.agb (Bloecker 1995 as MESA r26.4.1).

Each law is checked at three points against values written out from the printed equation with plain Python floats
(the literals below; the first point of each Vink law has L, M, T at the equation's pivots so that only the v_inf/v_esc
term survives and can be done by hand: Eq. 24 at L=1e5, M=30, T=4e4, Z=Zsun, v=2.6 gives log Mdot = -6.697 - 1.226
log10(1.3) = -6.83668). Units: L [Lsun], R [Rsun], M [Msun], Teff [K], rate [Msun/yr].
"""

import math

import jax
import pytest

from jaxstro.winds import (
    BLOECKER95_COEFFICIENT,
    VINK01_VINF_OVER_VESC_COOL,
    VINK01_VINF_OVER_VESC_HOT,
    VINK01_VINF_Z_EXPONENT,
    VINK01_ZSUN,
    bloecker95_mdot,
    nugis_lamers00_eq1_mdot,
    nugis_lamers00_eq22_mdot,
    vink01_gamma_e,
    vink01_jump_teff_k,
    vink01_log_rho_eq14,
    vink01_log_rho_eq23,
    vink01_mdot_cool,
    vink01_mdot_hot,
)

_RTOL = 1.0e-13   # a product of a few logs and a power: about 1000 ulp of margin on the literals


@pytest.mark.unit
def test_published_constants():
    """A retyped constant: Zsun 0.019 and the Galactic ratios 2.6 / 1.3, Z^0.13 (Vink 2001 Sects. 4, 8); 4.83e-9."""
    assert (VINK01_ZSUN, VINK01_VINF_OVER_VESC_HOT, VINK01_VINF_OVER_VESC_COOL, VINK01_VINF_Z_EXPONENT) == (
        0.019, 2.6, 1.3, 0.13)
    assert BLOECKER95_COEFFICIENT == 4.83e-9


_HOT = [  # (L, M, Teff, Z/Zsun, v_inf/v_esc) -> Eq. 24
    ((1e5, 30.0, 4.0e4, 1.0, 2.6), 1.4564831010898473e-07),
    ((2.5e5, 40.0, 3.3e4, 0.5, 2.6 * 0.5 ** 0.13), 3.237982460205558e-07),
    ((1e6, 80.0, 3.0e4, 2.0, 2.6), 5.845994032916469e-06),
]
_COOL = [  # Eq. 25
    ((1e5, 30.0, 2.0e4, 1.0, 1.3), 4.0881395862304443e-07),
    ((3e4, 20.0, 1.8e4, 0.2, 1.3 * 0.2 ** 0.13), 1.5636569994564786e-08),
    ((2e5, 50.0, 1.5e4, 1.5, 1.3), 9.90238147122627e-07),
]


@pytest.mark.unit
@pytest.mark.parametrize("args,expected", _HOT)
def test_vink01_eq24_hot_side(args, expected):
    """A wrong coefficient, pivot or sign in Eq. 24 (including the -10.92 [log T/4e4]^2 term at T != 4e4)."""
    assert float(jax.jit(vink01_mdot_hot)(*args)) == pytest.approx(expected, rel=_RTOL)


@pytest.mark.unit
@pytest.mark.parametrize("args,expected", _COOL)
def test_vink01_eq25_cool_side(args, expected):
    """A wrong coefficient, pivot or sign in Eq. 25."""
    assert float(jax.jit(vink01_mdot_cool)(*args)) == pytest.approx(expected, rel=_RTOL)


@pytest.mark.unit
def test_vink01_jump_chain_eqs_11_14_15_23():
    """A wrong constant in the jump-temperature chain. Zsun: Eq. 14 gives log rho = -13.636, Eq. 15 then
    T = 1000 (61.2 + 2.59 (-13.636)) = 25 882.76 K. At Z/Zsun = 0.5 and sigma_e = 0.34, L = 1e5, M = 30: Gamma_e =
    7.66e-5 * 0.34 * 1e5 / 30 = 0.0868133, Eq. 23 gives -14.94 + 0.85 log10(0.5) + 3.2 Gamma_e."""
    assert float(vink01_jump_teff_k(vink01_log_rho_eq14(1.0))) == pytest.approx(25882.76, rel=_RTOL)
    ge = float(vink01_gamma_e(0.34, 1e5, 30.0))
    assert ge == pytest.approx(0.08681333333333335, rel=_RTOL)
    assert float(vink01_log_rho_eq14(0.5)) == pytest.approx(-13.636 + 0.889 * math.log10(0.5), rel=_RTOL)
    assert float(vink01_log_rho_eq23(0.5, ge)) == pytest.approx(-14.94 + 0.85 * math.log10(0.5) + 3.2 * ge, rel=_RTOL)
    assert float(vink01_jump_teff_k(-14.0)) == pytest.approx(1.0e3 * (61.2 - 2.59 * 14.0), rel=_RTOL)


@pytest.mark.unit
def test_vink01_scalings_of_the_equations():
    """The (Z/Zsun)^0.85 and (v_inf/v_esc)^-1.226 / -1.601 dependences of Eqs. 24-25: exact power laws."""
    base = float(vink01_mdot_hot(1e5, 30.0, 4e4, 1.0, 2.0))
    assert float(vink01_mdot_hot(1e5, 30.0, 4e4, 10.0, 2.0)) == pytest.approx(base * 10 ** 0.85, rel=_RTOL)
    assert float(vink01_mdot_hot(1e5, 30.0, 4e4, 1.0, 4.0)) == pytest.approx(base * 2.0 ** -1.226, rel=_RTOL)
    cbase = float(vink01_mdot_cool(1e5, 30.0, 2e4, 1.0, 2.0))
    assert float(vink01_mdot_cool(1e5, 30.0, 2e4, 1.0, 4.0)) == pytest.approx(cbase * 2.0 ** -1.601, rel=_RTOL)
    assert cbase == pytest.approx(10.0 ** -6.688, rel=_RTOL)


_NL = [  # (L, Y, Z) -> (Eq. 1, Eq. 22)
    ((1e5, 0.98, 0.02), (3.851228851915277e-06, 4.328178953890603e-06)),
    ((2e5, 0.5, 0.017), (2.765681486074288e-06, 3.060960990915168e-06)),
    ((5e4, 0.3, 0.4), (9.414720562359939e-07, 9.333816793306169e-07)),
]


@pytest.mark.unit
@pytest.mark.parametrize("args,expected", _NL)
def test_nugis_lamers_eqs_1_and_22(args, expected):
    """A wrong exponent of L, Y or Z in Eq. 1 (1.29, 1.7, 0.5) or Eq. 22 (-11.00, 1.29, 1.73, 0.47); the two printed
    forms differ by about 12 % at Y = 0.98, Z = 0.02, which the literals keep apart."""
    assert float(jax.jit(nugis_lamers00_eq1_mdot)(*args)) == pytest.approx(expected[0], rel=_RTOL)
    assert float(jax.jit(nugis_lamers00_eq22_mdot)(*args)) == pytest.approx(expected[1], rel=_RTOL)


_BL = [  # (L, R, M, eta) -> Mdot
    ((5000.0, 200.0, 1.0, 0.1), 1.8759710574635003e-06),
    ((10000.0, 300.0, 2.0, 0.5), 2.1325803075685924e-05),
    ((3000.0, 150.0, 0.8, 1.0), 4.244905866439106e-06),
]


@pytest.mark.unit
@pytest.mark.parametrize("args,expected", _BL)
def test_bloecker_as_mesa(args, expected):
    """A wrong power of M (-2.1) or L (2.7), a dropped Reimers factor or a wrong eta placement in MESA's
    ``eval_blocker_wind`` form, eta 4.83e-9 M^-2.1 L^2.7 x 4e-13 L R/M (paper check open: module doc of agb.py)."""
    L, R, M, eta = args
    assert float(jax.jit(lambda *a: bloecker95_mdot(a[0], a[1], a[2], eta=a[3]))(L, R, M, eta)) == pytest.approx(
        expected, rel=_RTOL)


@pytest.mark.unit
def test_bloecker_gradient_in_eta_is_the_unit_rate():
    """A rate nonlinear in eta: d(rate)/d(eta) equals the rate at eta = 1."""
    args = (5000.0, 200.0, 1.0)
    grad = jax.grad(lambda e: bloecker95_mdot(*args, eta=e))(0.3)
    assert float(grad) == pytest.approx(float(bloecker95_mdot(*args, eta=1.0)), rel=_RTOL)
