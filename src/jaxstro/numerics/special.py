# src/jaxstro/numerics/special.py
"""Small special-function kernels used across astronomy-facing code."""

import math
from functools import partial

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxstro import constants

from .checks import try_concrete_bool


def _raise_if_concrete_false(predicate, message: str) -> None:
    """Raise eagerly when a validation predicate is concrete and false."""
    result = try_concrete_bool(jnp.asarray(predicate))
    if result is False:
        raise ValueError(message)


def _validate_positive(name: str, value: Float[Array, "..."]) -> None:
    _raise_if_concrete_false(jnp.all(value > 0.0), f"{name} must be positive")


@jax.jit
def _log_expm1_positive(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """Stable ``log(expm1(x))`` for positive ``x``."""
    x_small = jnp.minimum(x, 50.0)
    x_large_safe = jnp.maximum(x, 1e-12)
    small = jnp.log(jnp.expm1(x_small))
    large = x + jnp.log1p(-jnp.exp(-x_large_safe))
    return jnp.where(x < 50.0, small, large)


def log_planck_lambda_cgs(
    wavelength_cm: Float[Array, "..."],
    temperature: Float[Array, "..."],
) -> Float[Array, "..."]:
    """
    Log Planck spectral radiance per wavelength in CGS units.

    Parameters use centimeters and kelvin. The returned log value corresponds
    to ``B_lambda`` in ``erg s^-1 cm^-2 sr^-1 cm^-1``.
    """
    wavelength_cm = jnp.asarray(wavelength_cm)
    temperature = jnp.asarray(temperature)
    _validate_positive("wavelength_cm", wavelength_cm)
    _validate_positive("temperature", temperature)

    x = (
        constants.H_CGS
        * constants.C_CGS
        / (wavelength_cm * constants.K_B * temperature)
    )
    log_prefactor = (
        jnp.log(2.0)
        + jnp.log(constants.H_CGS)
        + 2.0 * jnp.log(constants.C_CGS)
        - 5.0 * jnp.log(wavelength_cm)
    )
    return log_prefactor - _log_expm1_positive(x)


def planck_lambda_cgs(
    wavelength_cm: Float[Array, "..."],
    temperature: Float[Array, "..."],
) -> Float[Array, "..."]:
    """
    Planck spectral radiance per wavelength in CGS units.

    Inputs are wavelength in centimeters and temperature in kelvin. The result
    is ``B_lambda`` in ``erg s^-1 cm^-2 sr^-1 cm^-1``.
    """
    return jnp.exp(log_planck_lambda_cgs(wavelength_cm, temperature))


def log_planck_nu_cgs(
    frequency_hz: Float[Array, "..."],
    temperature: Float[Array, "..."],
) -> Float[Array, "..."]:
    """
    Log Planck spectral radiance per frequency in CGS units.

    Parameters use hertz and kelvin. The returned log value corresponds to
    ``B_nu`` in ``erg s^-1 cm^-2 sr^-1 Hz^-1``.
    """
    frequency_hz = jnp.asarray(frequency_hz)
    temperature = jnp.asarray(temperature)
    _validate_positive("frequency_hz", frequency_hz)
    _validate_positive("temperature", temperature)

    x = constants.H_CGS * frequency_hz / (constants.K_B * temperature)
    log_prefactor = (
        jnp.log(2.0)
        + jnp.log(constants.H_CGS)
        + 3.0 * jnp.log(frequency_hz)
        - 2.0 * jnp.log(constants.C_CGS)
    )
    return log_prefactor - _log_expm1_positive(x)


def planck_nu_cgs(
    frequency_hz: Float[Array, "..."],
    temperature: Float[Array, "..."],
) -> Float[Array, "..."]:
    """
    Planck spectral radiance per frequency in CGS units.

    Inputs are frequency in hertz and temperature in kelvin. The result is
    ``B_nu`` in ``erg s^-1 cm^-2 sr^-1 Hz^-1``.
    """
    return jnp.exp(log_planck_nu_cgs(frequency_hz, temperature))


@partial(jax.jit, static_argnames=("axis",))
def log_normalize(
    log_weights: Float[Array, "..."],
    *,
    axis: int | tuple[int, ...] | None = -1,
) -> Float[Array, "..."]:
    """Return log weights normalized so ``sum(exp(out), axis) == 1``."""
    log_weights = jnp.asarray(log_weights)
    return log_weights - jnn.logsumexp(log_weights, axis=axis, keepdims=True)


@partial(jax.jit, static_argnames=("axis",))
def normalize_log_weights(
    log_weights: Float[Array, "..."],
    *,
    axis: int | tuple[int, ...] | None = -1,
) -> Float[Array, "..."]:
    """Return normalized probabilities from unnormalized log weights."""
    return jnp.exp(log_normalize(log_weights, axis=axis))


def _stack_polynomial_sequence(
    p0: Float[Array, "..."],
    p1: Float[Array, "..."],
    rest: Float[Array, " k ..."],
) -> Float[Array, "... k"]:
    leading = jnp.stack([p0, p1], axis=-1)
    tail = jnp.moveaxis(rest, 0, -1)
    return jnp.concatenate([leading, tail], axis=-1)


@partial(jax.jit, static_argnames=("degree",))
def legendre_basis(
    x: Float[Array, "..."],
    *,
    degree: int,
) -> Float[Array, "... degree"]:
    """Evaluate Legendre polynomials ``P_0`` through ``P_degree``."""
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    x = jnp.asarray(x)
    p0 = jnp.ones_like(x)
    if degree == 0:
        return p0[..., None]
    p1 = x
    if degree == 1:
        return jnp.stack([p0, p1], axis=-1)

    def step(carry, n):
        p_nm1, p_n = carry
        n = n.astype(x.dtype)
        p_np1 = ((2.0 * n + 1.0) * x * p_n - n * p_nm1) / (n + 1.0)
        return (p_n, p_np1), p_np1

    _, rest = jax.lax.scan(step, (p0, p1), jnp.arange(1, degree, dtype=x.dtype))
    return _stack_polynomial_sequence(p0, p1, rest)


@partial(jax.jit, static_argnames=("degree",))
def chebyshev_t_basis(
    x: Float[Array, "..."],
    *,
    degree: int,
) -> Float[Array, "... degree"]:
    """Evaluate Chebyshev polynomials of the first kind ``T_0`` through ``T_degree``."""
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    x = jnp.asarray(x)
    t0 = jnp.ones_like(x)
    if degree == 0:
        return t0[..., None]
    t1 = x
    if degree == 1:
        return jnp.stack([t0, t1], axis=-1)

    def step(carry, _):
        t_nm1, t_n = carry
        t_np1 = 2.0 * x * t_n - t_nm1
        return (t_n, t_np1), t_np1

    _, rest = jax.lax.scan(step, (t0, t1), jnp.arange(1, degree))
    return _stack_polynomial_sequence(t0, t1, rest)


@partial(jax.jit, static_argnames=("degree",))
def laguerre_basis(
    x: Float[Array, "..."],
    *,
    degree: int,
) -> Float[Array, "... degree"]:
    """Evaluate ordinary Laguerre polynomials ``L_0`` through ``L_degree``."""
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    x = jnp.asarray(x)
    l0 = jnp.ones_like(x)
    if degree == 0:
        return l0[..., None]
    l1 = 1.0 - x
    if degree == 1:
        return jnp.stack([l0, l1], axis=-1)

    def step(carry, n):
        l_nm1, l_n = carry
        n = n.astype(x.dtype)
        l_np1 = ((2.0 * n + 1.0 - x) * l_n - n * l_nm1) / (n + 1.0)
        return (l_n, l_np1), l_np1

    _, rest = jax.lax.scan(step, (l0, l1), jnp.arange(1, degree, dtype=x.dtype))
    return _stack_polynomial_sequence(l0, l1, rest)


_RICCATI_WRONSKIAN = -1.0
"""Exact value of ``S_l C_{l-1} - S_{l-1} C_l`` for every order and argument."""

_RICCATI_SEED_MARGIN = 60
"""Orders above ``max(degree, x)`` at which Miller's downward sweep is seeded."""

_RICCATI_RESCALE = 1.0e150
"""Downward-sweep rescale threshold. Miller depends only on ratios, so this is
exact rather than a tolerance."""


def riccati_seed_order(degree: int, x_max: float) -> int:
    """Miller start order clearing both the order and the argument.

    The downward sweep is self-correcting only where ``l > x``; below ``x`` both
    solutions oscillate and the seed error persists instead of damping. Seeding
    only above ``degree`` therefore fails at large argument: measured error
    ``1.7e-5`` at ``l = 1, x = 50`` with a seed of ``l + 60``, while ``l = 14,
    x = 0.5`` was exact.
    """
    return int(math.ceil(max(float(x_max), 0.0))) + degree + _RICCATI_SEED_MARGIN


def riccati_bessel_basis(
    x: Float[Array, "..."],
    *,
    degree: int,
    seed_order: int | None = None,
) -> tuple[Float[Array, "degree ..."], Float[Array, "degree ..."]]:
    """Evaluate Riccati-Bessel ``S_l = x j_l(x)`` and ``C_l = -x y_l(x)``.

    ``S`` uses Miller's downward recurrence and ``C`` an upward recurrence, each
    in the direction where its own solution grows. Upward recurrence for ``S`` is
    unstable once ``l > x`` and returns finite, smooth, wrong values rather than
    failing: measured residual of the Wronskian identity at ``x = 2`` is
    ``1.1e-16`` at ``l = 10`` and ``5.6e+14`` at ``l = 25``.

    Parameters
    ----------
    x
        Argument. Must be positive; ``C_l`` diverges at the origin.
    degree
        Highest order returned. Output leading axis has ``degree + 1`` entries.
    seed_order
        Where the downward sweep starts. Defaults to a margin above ``degree``
        alone, which is **only valid for ``x < degree``**. Pass
        ``riccati_seed_order(degree, x_max)`` when sweeping in ``x``; verify
        with :func:`riccati_wronskian_residual`.

    Returns
    -------
    tuple
        ``(S, C)``, each with leading axis over ``l = 0 .. degree``.
    """
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    x = jnp.asarray(x)
    sin_x = jnp.sin(x)
    cos_x = jnp.cos(x)

    c0 = cos_x
    c1 = cos_x / x + sin_x
    if degree == 0:
        c = c0[None]
    else:

        def c_step(carry, n):
            c_prev, c_curr = carry
            c_next = (2.0 * n + 1.0) / x * c_curr - c_prev
            return (c_curr, c_next), c_next

        _, c_rest = jax.lax.scan(c_step, (c0, c1), jnp.arange(1, degree, dtype=x.dtype))
        c = jnp.concatenate([jnp.stack([c0, c1]), c_rest])

    top = (degree + _RICCATI_SEED_MARGIN) if seed_order is None else int(seed_order)

    def s_step(carry, n):
        s_next, s_curr = carry
        s_prev = (2.0 * n + 1.0) / x * s_curr - s_next
        scale = jnp.where(jnp.abs(s_prev) > _RICCATI_RESCALE, _RICCATI_RESCALE, 1.0)
        return (s_curr / scale, s_prev / scale), s_prev / scale

    seed = (jnp.zeros_like(x), jnp.full_like(x, 1.0e-280))
    _, s_down = jax.lax.scan(s_step, seed, jnp.arange(top, 0, -1, dtype=x.dtype))
    s_asc = s_down[::-1][: degree + 1]
    return s_asc * (sin_x / s_asc[0]), c


def riccati_wronskian_residual(
    s: Float[Array, "degree ..."],
    c: Float[Array, "degree ..."],
) -> Float[Array, "degree_minus_one ..."]:
    """Residual of ``S_l C_{l-1} - S_{l-1} C_l = -1``.

    Exact for every order and argument, so this is a recurrence-stability gate
    rather than an approximation check. An unstable recurrence violates it by
    orders of magnitude while the values themselves stay finite and smooth.
    """
    return jnp.abs(s[1:] * c[:-1] - s[:-1] * c[1:] - _RICCATI_WRONSKIAN)


def riccati_bessel_at_order(
    x: Float[Array, "..."],
    *,
    order: int,
    seed_order: int | None = None,
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Evaluate ``(S_order, C_order)`` without materializing the lower orders.

    Same recurrences and the same seed-order obligation as
    :func:`riccati_bessel_basis`; only the intermediate orders are discarded
    instead of stacked. A caller sweeping one order over many arguments would
    otherwise allocate ``(order + 1, n)`` values it never reads -- roughly
    250 MB for ``order = 30`` at ``n = 1e6``.

    Parameters
    ----------
    x
        Argument. Must be positive.
    order
        The single order to return. Concrete integer.
    seed_order
        See :func:`riccati_bessel_basis`; the same requirement to clear both the
        order and the argument applies.

    Returns
    -------
    tuple
        ``(S_order, C_order)``, each shaped like ``x``.
    """
    if order < 0:
        raise ValueError("order must be nonnegative")
    x = jnp.asarray(x)
    sin_x = jnp.sin(x)
    cos_x = jnp.cos(x)

    c0 = cos_x
    c1 = cos_x / x + sin_x
    if order == 0:
        c = c0
    elif order == 1:
        c = c1
    else:

        def c_step(carry, n):
            c_prev, c_curr = carry
            return (c_curr, (2.0 * n + 1.0) / x * c_curr - c_prev), None

        (_, c), _ = jax.lax.scan(c_step, (c0, c1), jnp.arange(1, order, dtype=x.dtype))

    top = (order + _RICCATI_SEED_MARGIN) if seed_order is None else int(seed_order)

    def s_step(carry, n):
        s_next, s_curr, s_saved = carry
        s_prev = (2.0 * n + 1.0) / x * s_curr - s_next
        s_saved = jnp.where(n == order + 1, s_prev, s_saved)
        scale = jnp.where(jnp.abs(s_prev) > _RICCATI_RESCALE, _RICCATI_RESCALE, 1.0)
        return (s_curr / scale, s_prev / scale, s_saved / scale), None

    seed = (jnp.zeros_like(x), jnp.full_like(x, 1.0e-280), jnp.zeros_like(x))
    (_, s_zero, s_order), _ = jax.lax.scan(
        s_step, seed, jnp.arange(top, 0, -1, dtype=x.dtype)
    )
    if order == 0:
        s_order = s_zero
    return s_order * (sin_x / s_zero), c


__all__ = [
    "planck_lambda_cgs",
    "log_planck_lambda_cgs",
    "planck_nu_cgs",
    "log_planck_nu_cgs",
    "log_normalize",
    "normalize_log_weights",
    "legendre_basis",
    "chebyshev_t_basis",
    "laguerre_basis",
    "riccati_seed_order",
    "riccati_bessel_basis",
    "riccati_bessel_at_order",
    "riccati_wronskian_residual",
]
