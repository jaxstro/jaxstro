"""Generic distribution kernels with explicit support behavior."""

import jax.numpy as jnp
from jax.scipy import special as jsp_special
from jaxtyping import Array, Float

_LOG_SQRT_2PI = 0.5 * jnp.log(2.0 * jnp.pi)
_TAYLOR_THRESHOLD = 1.0e-6


def normal_logpdf(
    x: float | Float[Array, "..."],
    *,
    loc: float | Float[Array, "..."] = 0.0,
    scale: float | Float[Array, "..."] = 1.0,
) -> Float[Array, "..."]:
    """Log-density of a normal distribution."""
    z = (jnp.asarray(x) - loc) / scale
    return -_LOG_SQRT_2PI - jnp.log(scale) - 0.5 * z**2


def normal_cdf(
    x: float | Float[Array, "..."],
    *,
    loc: float | Float[Array, "..."] = 0.0,
    scale: float | Float[Array, "..."] = 1.0,
) -> Float[Array, "..."]:
    """CDF of a normal distribution."""
    z = (jnp.asarray(x) - loc) / scale
    return jsp_special.ndtr(z)


def normal_ppf(
    u: float | Float[Array, "..."],
    *,
    loc: float | Float[Array, "..."] = 0.0,
    scale: float | Float[Array, "..."] = 1.0,
) -> Float[Array, "..."]:
    """Inverse CDF of a normal distribution."""
    return loc + scale * jsp_special.ndtri(jnp.asarray(u))


def lognormal_logpdf(
    x: Float[Array, "..."],
    *,
    loc: float | Float[Array, "..."] = 0.0,
    scale: float | Float[Array, "..."] = 1.0,
) -> Float[Array, "..."]:
    """Log-density of a lognormal distribution for ``log(x) ~ N(loc, scale)``."""
    x = jnp.asarray(x)
    in_support = x > 0.0
    x_safe = jnp.where(in_support, x, 1.0)
    log_x = jnp.log(x_safe)
    logpdf = normal_logpdf(log_x, loc=loc, scale=scale) - log_x
    return jnp.where(in_support, logpdf, -jnp.inf)


def lognormal_cdf(
    x: Float[Array, "..."],
    *,
    loc: float | Float[Array, "..."] = 0.0,
    scale: float | Float[Array, "..."] = 1.0,
) -> Float[Array, "..."]:
    """CDF of a lognormal distribution."""
    x = jnp.asarray(x)
    in_support = x > 0.0
    x_safe = jnp.where(in_support, x, 1.0)
    cdf = normal_cdf(jnp.log(x_safe), loc=loc, scale=scale)
    return jnp.where(in_support, cdf, 0.0)


def lognormal_ppf(
    u: Float[Array, "..."],
    *,
    loc: float | Float[Array, "..."] = 0.0,
    scale: float | Float[Array, "..."] = 1.0,
) -> Float[Array, "..."]:
    """Inverse CDF of a lognormal distribution."""
    return jnp.exp(normal_ppf(u, loc=loc, scale=scale))


def _expm1_over_x(x):
    """Evaluate ``expm1(x) / x`` smoothly through zero."""
    small = jnp.abs(x) < _TAYLOR_THRESHOLD
    x_safe = jnp.where(small, 1.0, x)
    taylor = 1.0 + x / 2.0 + x * x / 6.0
    return jnp.where(small, taylor, jnp.expm1(x_safe) / x_safe)


def _log1p_over_x(x):
    """Evaluate ``log1p(x) / x`` smoothly through zero."""
    small = jnp.abs(x) < _TAYLOR_THRESHOLD
    x_safe = jnp.where(small, 1.0, x)
    taylor = 1.0 - x / 2.0 + x * x / 3.0
    return jnp.where(small, taylor, jnp.log1p(x_safe) / x_safe)


def _powerlaw_integral(lo, hi, exponent):
    """Evaluate ``(hi**exponent - lo**exponent) / exponent`` smoothly."""
    log_width = jnp.log(hi) - jnp.log(lo)
    return lo**exponent * log_width * _expm1_over_x(exponent * log_width)


def _powerlaw_inverse(lo, integral, exponent):
    """Invert ``_powerlaw_integral(lo, x, exponent)`` smoothly."""
    scaled = integral * lo ** (-exponent)
    return jnp.exp(jnp.log(lo) + scaled * _log1p_over_x(exponent * scaled))


def _powerlaw_log_norm(alpha, xmin, xmax):
    exponent = alpha + 1.0
    return -jnp.log(_powerlaw_integral(xmin, xmax, exponent))


def powerlaw_logpdf(
    x: Float[Array, "..."],
    *,
    alpha: float | Float[Array, ""] = -1.0,
    xmin: float | Float[Array, ""] = 1.0,
    xmax: float | Float[Array, ""] = 2.0,
) -> Float[Array, "..."]:
    """Log-density for ``p(x) proportional to x**alpha`` on ``[xmin, xmax]``."""
    x = jnp.asarray(x)
    in_support = (x >= xmin) & (x <= xmax)
    x_safe = jnp.where(in_support, x, xmin)
    logpdf = _powerlaw_log_norm(alpha, xmin, xmax) + alpha * jnp.log(x_safe)
    return jnp.where(in_support, logpdf, -jnp.inf)


def powerlaw_cdf(
    x: Float[Array, "..."],
    *,
    alpha: float | Float[Array, ""] = -1.0,
    xmin: float | Float[Array, ""] = 1.0,
    xmax: float | Float[Array, ""] = 2.0,
) -> Float[Array, "..."]:
    """CDF for a finite-support power-law distribution."""
    x = jnp.asarray(x)
    x_clamped = jnp.clip(x, xmin, xmax)
    exponent = alpha + 1.0
    numerator = _powerlaw_integral(xmin, x_clamped, exponent)
    denominator = _powerlaw_integral(xmin, xmax, exponent)
    return numerator / denominator


def powerlaw_ppf(
    u: Float[Array, "..."],
    *,
    alpha: float | Float[Array, ""] = -1.0,
    xmin: float | Float[Array, ""] = 1.0,
    xmax: float | Float[Array, ""] = 2.0,
) -> Float[Array, "..."]:
    """Inverse CDF for a finite-support power-law distribution."""
    u = jnp.asarray(u)
    exponent = alpha + 1.0
    total = _powerlaw_integral(xmin, xmax, exponent)
    return _powerlaw_inverse(xmin, u * total, exponent)


def truncated_normal_logpdf(
    x: Float[Array, "..."],
    *,
    loc: float | Float[Array, "..."] = 0.0,
    scale: float | Float[Array, "..."] = 1.0,
    low: float | Float[Array, "..."] = -jnp.inf,
    high: float | Float[Array, "..."] = jnp.inf,
) -> Float[Array, "..."]:
    """Log-density of a normal distribution truncated to ``[low, high]``."""
    x = jnp.asarray(x)
    in_support = (x >= low) & (x <= high)
    normalizer = normal_cdf(high, loc=loc, scale=scale) - normal_cdf(
        low,
        loc=loc,
        scale=scale,
    )
    logpdf = normal_logpdf(x, loc=loc, scale=scale) - jnp.log(normalizer)
    return jnp.where(in_support, logpdf, -jnp.inf)


def truncated_normal_cdf(
    x: Float[Array, "..."],
    *,
    loc: float | Float[Array, "..."] = 0.0,
    scale: float | Float[Array, "..."] = 1.0,
    low: float | Float[Array, "..."] = -jnp.inf,
    high: float | Float[Array, "..."] = jnp.inf,
) -> Float[Array, "..."]:
    """CDF of a normal distribution truncated to ``[low, high]``."""
    normalizer = normal_cdf(high, loc=loc, scale=scale) - normal_cdf(
        low,
        loc=loc,
        scale=scale,
    )
    raw = (
        normal_cdf(x, loc=loc, scale=scale) - normal_cdf(low, loc=loc, scale=scale)
    ) / normalizer
    return jnp.clip(raw, 0.0, 1.0)


def truncated_normal_ppf(
    u: Float[Array, "..."],
    *,
    loc: float | Float[Array, "..."] = 0.0,
    scale: float | Float[Array, "..."] = 1.0,
    low: float | Float[Array, "..."] = -jnp.inf,
    high: float | Float[Array, "..."] = jnp.inf,
) -> Float[Array, "..."]:
    """Inverse CDF of a normal distribution truncated to ``[low, high]``."""
    cdf_low = normal_cdf(low, loc=loc, scale=scale)
    cdf_high = normal_cdf(high, loc=loc, scale=scale)
    return normal_ppf(
        cdf_low + jnp.asarray(u) * (cdf_high - cdf_low), loc=loc, scale=scale
    )


__all__ = [
    "normal_logpdf",
    "normal_cdf",
    "normal_ppf",
    "lognormal_logpdf",
    "lognormal_cdf",
    "lognormal_ppf",
    "powerlaw_logpdf",
    "powerlaw_cdf",
    "powerlaw_ppf",
    "truncated_normal_logpdf",
    "truncated_normal_cdf",
    "truncated_normal_ppf",
]
