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

_RICCATI_RESCALE_EXP2 = 500
"""Base-2 exponent of the downward-sweep rescale factor.

A **power of two** rather than a round decimal, so that dividing by it is exact
in binary. The recurrence is linear and homogeneous, so an exact scaling of the
carry scales every later value exactly, and the final normalisation divides the
common factor straight back out: the returned values become provably independent
of how many times the rescale fired. With the previous ``1e150`` each division
rounded, so the result depended on the rescale history at the ``1e-16`` level and
the two implementations here could not stay bit-for-bit equal past one rescale.
"""

_RICCATI_RESCALE = float(2.0**_RICCATI_RESCALE_EXP2)
"""Downward-sweep rescale threshold, ``2**500 ~ 3.3e150``. Miller depends only on
ratios, so this is exact rather than a tolerance."""


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

    # The sweep rescales whenever it would overflow, and each rescale reaches the
    # carry and therefore every LATER value -- but ``scan`` stacks its outputs
    # once and never revisits them, so a rescale reaches nothing already emitted.
    # Going downward the values GROW, so the retained window ``[0, degree]`` is
    # emitted last: a rescale firing at order ``b <= degree`` leaves the orders
    # above ``b`` larger than the orders at or below it by exactly the rescale
    # factor. Every entry stays finite, smooth and individually plausible, so
    # nothing forward-only notices; the Wronskian residual jumps to the rescale
    # factor itself. Measured 2026-08-03 at degree 11 and seed order 2118 (the
    # H-H scattering configuration): 323 of 2000 arguments in ``x`` in
    # ``[1e-3, 5]`` corrupt, worst residual ``1.0e+150``.
    #
    # The cure is to carry the cumulative rescale exponent ALONGSIDE each value
    # and put the retained window back on one scale afterwards. ``e`` is an
    # integer count of base-2 exponent, so it costs one add per step, contributes
    # no tangent, and the reconciliation below is an exact power-of-two scaling.
    #
    # ``riccati_bessel_at_order`` never had this defect: it keeps its one saved
    # value IN THE CARRY, so every later rescale divides it too.
    def s_step(carry, n):
        s_next, s_curr, exponent = carry
        s_prev = (2.0 * n + 1.0) / x * s_curr - s_next
        fired = jnp.abs(s_prev) > _RICCATI_RESCALE
        scale = jnp.where(fired, _RICCATI_RESCALE, 1.0)
        exponent = exponent + jnp.where(fired, _RICCATI_RESCALE_EXP2, 0)
        return (s_curr / scale, s_prev / scale, exponent), (s_prev / scale, exponent)

    seed = (
        jnp.zeros_like(x),
        jnp.full_like(x, 1.0e-280),
        jnp.zeros(jnp.shape(x), dtype=jnp.int32),
    )
    _, (s_down, e_down) = jax.lax.scan(
        s_step, seed, jnp.arange(top, 0, -1, dtype=x.dtype)
    )
    s_asc = s_down[::-1][: degree + 1]
    e_asc = e_down[::-1][: degree + 1]

    # Order 0 is emitted last and so carries the largest exponent; ``delta <= 0``
    # and the reconciliation only ever scales DOWN. It can therefore underflow but
    # never overflow, and underflow here is the genuine float64 representability
    # limit on ``S_degree / S_0``, not an artifact of the rescale.
    s_asc = s_asc * jnp.exp2((e_asc - e_asc[0]).astype(x.dtype))

    # Put the sweep on an O(1) scale before normalising. The seed magnitude is
    # arbitrary for the PRIMAL -- Miller depends only on ratios and ``sin_x /
    # s_asc[0]`` divides it out -- but NOT for the DERIVATIVE, which carries a
    # ``1 / s_asc[0]**2`` term and so squares the sweep's magnitude.
    #
    # Wherever the sweep has little growth (the oscillatory region ``l < x``,
    # which is most of the useful domain) it ends near its ``1e-280`` seed, and
    # ``(1e-280)**2 = 1e-560`` underflows float64 to exactly zero. Until
    # 2026-08-03 this function therefore returned finite VALUES with NaN
    # DERIVATIVES: measured at degree 3, ``dS/dx`` was NaN at every ``x`` from 2
    # to 50 at the seed order :func:`riccati_seed_order` prescribes. Nothing
    # forward-only notices, and there was no derivative test.
    #
    # ``stop_gradient`` is what makes this correct rather than approximate. The
    # scale cancels IDENTICALLY: for any fixed ``M``, ``(s/M) * sin_x / (s_0/M)``
    # is the same function of ``x`` as ``s * sin_x / s_0``. So the derivative does
    # not depend on ``dM/dx``, and holding ``M`` constant is exact. Without
    # ``stop_gradient`` the ``-s dM / M**2`` term reintroduces the identical
    # underflow one level up -- that was tried, and the NaN pattern was unchanged.
    #
    # Rescaling HERE rather than reseeding the sweep is also deliberate: a larger
    # seed makes the sweep cross ``_RICCATI_RESCALE`` at small ``x``, and seeding
    # at ``1.0`` was tried and drove the Wronskian residual to ``1e150`` at
    # ``x = 0.5``. Until 2026-08-03 that was recorded here as a latent
    # inconsistency merely "no longer stepped on" -- it WAS being stepped on, at
    # the production seed order, and the exponent bookkeeping above now removes it
    # rather than avoiding it.
    # Scaling by ``|s_asc[0]|`` specifically -- not by the sweep's maximum -- makes
    # the denominator of the normalisation exactly ``+-1``, so ``1 / s_asc[0]**2``
    # is exactly 1 and underflow is impossible by construction rather than by
    # margin. It also keeps this in lockstep with :func:`riccati_bessel_at_order`,
    # which does the identical arithmetic on the identical values, so the two stay
    # bit-for-bit equal (pinned by ``test_single_order_matches_the_basis_exactly``).
    # **The anchor is the LARGER of orders 0 and 1, never order 0 unconditionally.**
    #
    # Miller's sweep fixes every ratio and no overall scale, so it must be tied
    # to one known value. Tying it to ``S_0 = sin x`` fails at ``x = n pi``,
    # where ``S_0`` vanishes but no higher order does: the sweep's own last step
    # is ``S_0 = (3/x) S_1 - S_2``, and at ``x = pi`` that is
    # ``0.9549 - 0.9549`` -- catastrophic cancellation, so the value the whole
    # array is normalised BY is the one value computed to no significant digits.
    #
    # Measured 2026-08-05 against closed forms at ``x = pi``: ``S_1`` came back
    # 0.297774 against a true 1.000000, a factor of 3.36, and at ``x = 2 pi``
    # every order was NaN. This was live in micrax, whose matching evaluates at
    # ``x = k r_max`` and therefore sweeps through ``n pi`` continuously.
    #
    # ``S_0`` and ``S_1`` cannot both be small: ``S_0 = sin x`` and
    # ``S_1 = sin x / x - cos x``, so ``S_0 = 0`` forces ``S_1 = -cos x = +-1``.
    # Both are exact in closed form, so anchoring on whichever is larger is
    # always well conditioned, and reduces to the old behaviour everywhere the
    # old behaviour was correct.
    s0_true = sin_x
    s1_true = sin_x / x - cos_x
    use1 = jnp.abs(s1_true) > jnp.abs(s0_true)
    if degree == 0:
        # Nothing to normalise: order 0 IS the closed form.
        return s0_true[None], c
    anchor_sweep = jnp.where(use1, s_asc[1], s_asc[0])
    anchor_true = jnp.where(use1, s1_true, s0_true)
    scale = jax.lax.stop_gradient(jnp.abs(anchor_sweep))
    s_asc = s_asc / scale
    anchor_sweep = anchor_sweep / scale
    s_asc = s_asc * (anchor_true / anchor_sweep)
    # Order 0 is taken from its closed form; order 1 is NOT, and the asymmetry is
    # measured rather than stylistic. ``sin x`` never cancels, so ``S_0`` is
    # exact everywhere and the sweep's own value (~2e-4 relative at ``x = n pi``,
    # on a magnitude of 1e-16) is strictly worse.
    #
    # ``S_1 = sin x / x - cos x`` DOES cancel: as ``x -> 0`` both terms approach
    # 1 while the difference is ``x^2/3``, so at ``x = 1e-3`` the relative error
    # is ~3e-10. Overriding order 1 with that closed form was tried and pushed
    # the Wronskian residual from 3.5e-14 to 4.9e-10 over ``x in [1e-3, 5]``,
    # concentrated entirely at ``x < 0.02`` and entirely in the order-1 identity.
    # The sweep is better there, so the sweep keeps it.
    #
    # The closed form is still used for the ANCHOR when ``|S_1| > |S_0|``, which
    # happens only near ``x = n pi`` where ``S_1 -> -cos x = +-1`` and there is
    # no cancellation to suffer.
    return jnp.concatenate([s0_true[None], s_asc[1:]]), c


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

    # No exponent bookkeeping is needed here, unlike in
    # :func:`riccati_bessel_basis`: ``s_saved`` lives in the CARRY, so every
    # rescale after the order was captured divides it along with everything else
    # and it never leaves the common scale. Keeping the two paths equal is why
    # the rescale factor is a power of two -- repeated exact divisions here and a
    # single exact multiply there give bit-identical results.
    # `s_one` rides in the CARRY beside `s_saved` for the same reason it does
    # there: every later rescale must divide it too, or it leaves the common
    # scale. It is captured at `n == 2`, where `s_prev` is `S_1`.
    def s_step(carry, n):
        s_next, s_curr, s_saved, s_one = carry
        s_prev = (2.0 * n + 1.0) / x * s_curr - s_next
        s_saved = jnp.where(n == order + 1, s_prev, s_saved)
        s_one = jnp.where(n == 2, s_prev, s_one)
        scale = jnp.where(jnp.abs(s_prev) > _RICCATI_RESCALE, _RICCATI_RESCALE, 1.0)
        return (s_curr / scale, s_prev / scale, s_saved / scale, s_one / scale), None

    seed = (
        jnp.zeros_like(x),
        jnp.full_like(x, 1.0e-280),
        jnp.zeros_like(x),
        jnp.zeros_like(x),
    )
    (_, s_zero, s_order, s_one), _ = jax.lax.scan(
        s_step, seed, jnp.arange(top, 0, -1, dtype=x.dtype)
    )
    if order == 0:
        s_order = s_zero
    # Same rescale as :func:`riccati_bessel_basis`, for the same reason and with
    # the same arithmetic -- see the long note there. Without it the derivative of
    # ``sin_x / s_zero`` carries ``1 / s_zero**2``, and ``s_zero`` sits near the
    # ``1e-280`` seed wherever the sweep does not grow, so the square underflows
    # to zero and this returns a finite value with a NaN derivative.
    # Same anchor choice as :func:`riccati_bessel_basis`, identical arithmetic
    # so the two stay bit-for-bit equal.
    # **The anchor is the LARGER of orders 0 and 1, never order 0 unconditionally.**
    #
    # Miller's sweep fixes every ratio and no overall scale, so it must be tied
    # to one known value. Tying it to ``S_0 = sin x`` fails at ``x = n pi``,
    # where ``S_0`` vanishes but no higher order does: the sweep's own last step
    # is ``S_0 = (3/x) S_1 - S_2``, and at ``x = pi`` that is
    # ``0.9549 - 0.9549`` -- catastrophic cancellation, so the value the whole
    # array is normalised BY is the one value computed to no significant digits.
    #
    # Measured 2026-08-05 against closed forms at ``x = pi``: ``S_1`` came back
    # 0.297774 against a true 1.000000, a factor of 3.36, and at ``x = 2 pi``
    # every order was NaN. This was live in micrax, whose matching evaluates at
    # ``x = k r_max`` and therefore sweeps through ``n pi`` continuously.
    #
    # ``S_0`` and ``S_1`` cannot both be small: ``S_0 = sin x`` and
    # ``S_1 = sin x / x - cos x``, so ``S_0 = 0`` forces ``S_1 = -cos x = +-1``.
    # Both are exact in closed form, so anchoring on whichever is larger is
    # always well conditioned, and reduces to the old behaviour everywhere the
    # old behaviour was correct.
    s0_true = sin_x
    s1_true = sin_x / x - cos_x
    use1 = jnp.abs(s1_true) > jnp.abs(s0_true)
    if order == 0:
        return s0_true, c
    # Order 1 comes from the sweep, NOT its closed form -- see the note in
    # `riccati_bessel_basis`: `sin x / x - cos x` cancels catastrophically as
    # x -> 0 and is measurably worse than the recurrence below x ~ 0.02.
    anchor_sweep = jnp.where(use1, s_one, s_zero)
    anchor_true = jnp.where(use1, s1_true, s0_true)
    scale = jax.lax.stop_gradient(jnp.abs(anchor_sweep))
    s_order = s_order / scale
    anchor_sweep = anchor_sweep / scale
    return s_order * (anchor_true / anchor_sweep), c


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


def riccati_bessel_log_basis(
    x: Float[Array, "..."],
    *,
    degree: int,
    seed_order: int | None = None,
) -> tuple[
    Float[Array, "degree ..."],
    Float[Array, "degree ..."],
    Float[Array, "degree ..."],
    Float[Array, "degree ..."],
]:
    """``(sign_S, log|S_l|, sign_C, log|C_l|)`` -- the pair, in a representable form.

    Same ``S_l = x j_l(x)``, ``C_l = -x y_l(x)`` as :func:`riccati_bessel_basis`,
    returned as sign and log-magnitude so that neither one can leave float64.

    **Why this exists.** The two solutions run in opposite directions: for
    ``l >> x``, ``S_l ~ x^{l+1}/(2l+1)!!`` underflows while
    ``C_l ~ (2l-1)!!/x^l`` overflows. Any expression that forms them as *values*
    and then combines them -- which is exactly what a direct phase-shift match
    does -- meets ``0 * inf`` or ``inf - inf`` and returns NaN, at arguments
    where the true answer is perfectly well defined and simply very small.

    Measured 2026-08-05 in micrax: at ``T = 3000 K`` the H-H channel integral
    returned NaN for ``l = 68..72`` at 32 and 64 Gauss-Legendre nodes per panel
    and finite values at 16 -- because finer panels place nodes at smaller ``k``,
    and ``S_72(2e-3)`` is ``~1e-391``. In log form that is ``-900``, an ordinary
    double.

    The caller recombines with a common scale, e.g. ``M = max(log|.|)`` over the
    terms it needs, and any term genuinely below the others by more than ~700
    e-folds underflows to exactly zero -- which is the correctly rounded result
    of the stable formula, not a mask over a failure.

    Zeros of ``S_l`` or ``C_l`` give ``-inf`` here, which is the honest log of
    zero and behaves correctly under the same recombination.

    Parameters
    ----------
    x
        Argument. Must be positive.
    degree
        Highest order returned; leading axis has ``degree + 1`` entries.
    seed_order
        Start of the downward sweep for ``S``. See :func:`riccati_seed_order`.
    """
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    x = jnp.asarray(x)
    sin_x = jnp.sin(x)
    cos_x = jnp.cos(x)
    log_x = jnp.log(x)

    # ---- C: upward, the direction in which the dominant solution grows -----
    # Carries a base-2 exponent alongside the value for exactly the reason the
    # `S` sweep in `riccati_bessel_basis` does: the recurrence would otherwise
    # overflow, and an overflow here is representational, not physical.
    c0, c1 = cos_x, cos_x / x + sin_x
    if degree == 0:
        c_vals = c0[None]
        c_exp = jnp.zeros_like(c0)[None]
    else:

        def c_step(carry, n):
            c_prev, c_curr, e = carry
            c_next = (2.0 * n + 1.0) / x * c_curr - c_prev
            fired = jnp.abs(c_next) > _RICCATI_RESCALE
            scale = jnp.where(fired, _RICCATI_RESCALE, 1.0)
            e = e + jnp.where(fired, float(_RICCATI_RESCALE_EXP2), 0.0)
            return (c_curr / scale, c_next / scale, e), (c_next / scale, e)

        zero = jnp.zeros_like(x)
        _, (c_rest, e_rest) = jax.lax.scan(
            c_step, (c0, c1, zero), jnp.arange(1, degree, dtype=x.dtype)
        )
        c_vals = jnp.concatenate([jnp.stack([c0, c1]), c_rest])
        c_exp = jnp.concatenate([jnp.stack([zero, zero]), e_rest])

    sign_c = jnp.sign(c_vals)
    log_c = jnp.log(jnp.abs(c_vals)) + c_exp * jnp.log(2.0)

    # ---- S: the ratio recurrence, which cannot under- or overflow ----------
    # `r_l = S_{l-1} / S_l` satisfies `r_l = (2l+1)/x - 1/r_{l+1}` downward, and
    # every `r_l` is O(l/x) -- an ordinary number at every order. Accumulating
    # `log|S_l| = log|S_0| - sum_m log|r_m|` then reaches arbitrarily small `S`
    # without ever forming it. This is the continued-fraction route, and it is
    # why no rescale bookkeeping is needed on this side.
    top = riccati_seed_order(degree, 0.0) if seed_order is None else int(seed_order)
    top = max(top, degree + 1)

    def r_step(r_next, m):
        r = (2.0 * m + 1.0) / x - 1.0 / r_next
        return r, r

    _, r_desc = jax.lax.scan(
        r_step,
        (2.0 * (top + 1.0) + 1.0) / x,
        jnp.arange(top, 0, -1, dtype=x.dtype),
    )
    r_asc = r_desc[::-1][:degree]  # r_1 .. r_degree (r_1 unused, see below)

    # **Anchor at order 1, never at order 0, and never use r_1.**
    #
    # `r_1 = S_0/S_1` is computed as `3/x - 1/r_2`, and near a zero of
    # `S_0 = sin x` those two terms agree to sixteen digits: the true ratio is
    # ~1e-16 while each term is ~1. Catastrophic cancellation, and the error is
    # 100% relative -- every higher order inherits it through the cumulative
    # sum.
    #
    # Measured 2026-08-05, before this fix: at `x = pi` the reconstruction gave
    # `S_1 = 1.103` against a true `0.2978` (3.7x); at `x = 2 pi`, `-1.471`
    # against `-1.000` (47%). That was live in micrax, where `x = k r_max`
    # sweeps through `n pi` continuously -- at `x = 10 pi` the l = 4 phase shift
    # read 2.2115 against a smooth trend of 2.4022.
    #
    # `S_0` and `S_1` cannot both be small: `S_0 = sin x` and
    # `S_1 = sin x / x - cos x`, so `S_0 = 0` forces `S_1 = -cos x = +-1`. Both
    # are known in closed form, so orders 0 and 1 are written down exactly and
    # the accumulation starts at order 2 using only `r_2, r_3, ...` -- ratios
    # that are O(l/x) and well conditioned wherever this representation is
    # needed (`l > x`, where `S_l` is monotonic and has no zeros).
    s0, s1 = sin_x, sin_x / x - cos_x
    log_s01 = jnp.stack([jnp.log(jnp.abs(s0)), jnp.log(jnp.abs(s1))])
    sign_s01 = jnp.stack([jnp.sign(s0), jnp.sign(s1)])
    if degree == 0:
        log_s, sign_s = log_s01[:1], sign_s01[:1]
    elif degree == 1:
        log_s, sign_s = log_s01, sign_s01
    else:
        r_up = r_asc[1:]  # r_2 .. r_degree; r_1 is deliberately unused
        log_s = jnp.concatenate(
            [log_s01, log_s01[1:2] - jnp.cumsum(jnp.log(jnp.abs(r_up)), axis=0)]
        )
        sign_s = jnp.concatenate(
            [sign_s01, sign_s01[1:2] * jnp.cumprod(jnp.sign(r_up), axis=0)]
        )

    del log_x
    return sign_s, log_s, sign_c, log_c
