"""Generic distribution kernels with explicit support behavior."""

import jax.numpy as jnp
from jax.scipy import special as jsp_special
from jaxtyping import Array, Float

_LOG_SQRT_2PI = 0.5 * jnp.log(2.0 * jnp.pi)


def normal_logpdf(
    x: Float[Array, "..."],
    *,
    loc: float | Float[Array, "..."] = 0.0,
    scale: float | Float[Array, "..."] = 1.0,
) -> Float[Array, "..."]:
    """Log-density of a normal distribution."""
    z = (jnp.asarray(x) - loc) / scale
    return -_LOG_SQRT_2PI - jnp.log(scale) - 0.5 * z**2


def normal_cdf(
    x: Float[Array, "..."],
    *,
    loc: float | Float[Array, "..."] = 0.0,
    scale: float | Float[Array, "..."] = 1.0,
) -> Float[Array, "..."]:
    """CDF of a normal distribution."""
    z = (jnp.asarray(x) - loc) / scale
    return jsp_special.ndtr(z)


def normal_ppf(
    u: Float[Array, "..."],
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


def _powerlaw_log_norm(alpha, xmin, xmax):
    exponent = alpha + 1.0
    near_log = jnp.isclose(exponent, 0.0)
    exponent_safe = jnp.where(near_log, 1.0, exponent)
    ordinary = jnp.log(jnp.abs(exponent_safe)) - jnp.log(
        jnp.abs(xmax**exponent_safe - xmin**exponent_safe)
    )
    log_case = -jnp.log(jnp.log(xmax) - jnp.log(xmin))
    return jnp.where(near_log, log_case, ordinary)


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
    near_log = jnp.isclose(exponent, 0.0)
    exponent_safe = jnp.where(near_log, 1.0, exponent)
    ordinary = (x_clamped**exponent_safe - xmin**exponent_safe) / (
        xmax**exponent_safe - xmin**exponent_safe
    )
    log_case = (jnp.log(x_clamped) - jnp.log(xmin)) / (jnp.log(xmax) - jnp.log(xmin))
    return jnp.where(near_log, log_case, ordinary)


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
    near_log = jnp.isclose(exponent, 0.0)
    exponent_safe = jnp.where(near_log, 1.0, exponent)
    ordinary = (
        xmin**exponent_safe + u * (xmax**exponent_safe - xmin**exponent_safe)
    ) ** (1.0 / exponent_safe)
    log_case = xmin * jnp.exp(u * (jnp.log(xmax) - jnp.log(xmin)))
    return jnp.where(near_log, log_case, ordinary)


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
