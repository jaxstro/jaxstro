# src/jaxstro/numerics/special.py
"""Small special-function kernels used across astronomy-facing code."""

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
]
