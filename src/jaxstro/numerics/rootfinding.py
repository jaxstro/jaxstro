# src/jaxstro/numerics/rootfinding.py
"""
Simple 1D root-finding helpers for JAX.

These are minimal, JAX-native solvers intended for scalar monotone
functions (bisection) or well-behaved functions with good initial
guesses (Newton). For large systems or more sophisticated solves,
prefer optimistix / lineax.

All functions use lax.scan internally with fixed iteration counts,
making them fully compatible with jit, vmap, and grad.

Usage:
    @jax.jit
    def solve(a, b):
        return bisect(lambda x: x**2 - 2.0, a, b)

    # Works with vmap
    jax.vmap(solve)(a_arr, b_arr)

    # Works with grad (differentiating through the solver)
    jax.grad(lambda a: solve(a, 2.0))(1.0)
"""

from typing import Callable, NamedTuple, Optional, Union

import jax
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .checks import try_concrete_bool
from .types import ScalarFn

PROPOSAL_NONE = 0
PROPOSAL_SECANT = 1
PROPOSAL_MIDPOINT = 2
PROPOSAL_LO_ENDPOINT = 3
PROPOSAL_HI_ENDPOINT = 4


class BracketState(NamedTuple):
    """True endpoint evidence for a scalar sign-changing bracket."""

    lo: Float[Array, "..."]
    hi: Float[Array, "..."]
    f_lo: Float[Array, "..."]
    f_hi: Float[Array, "..."]
    bracketed: Array


class BracketProposal(NamedTuple):
    """A deterministic interpolation or safeguarded midpoint proposal."""

    x: Float[Array, "..."]
    kind: Array
    safeguarded: Array


class RootTrace(NamedTuple):
    """Fixed-shape post-update evidence for every scan slot."""

    proposal: Float[Array, " steps"]
    residual: Float[Array, " steps"]
    lo: Float[Array, " steps"]
    hi: Float[Array, " steps"]
    f_lo: Float[Array, " steps"]
    f_hi: Float[Array, " steps"]
    proposal_kind: Array
    executed: Array
    admissible: Array
    converged: Array


class BracketedRootResult(NamedTuple):
    """Value-first scalar root result with explicit failure state."""

    root: Float[Array, ""]
    residual: Float[Array, ""]
    converged: Array
    bracketed: Array
    n_evaluations: Array
    trace: RootTrace


class _RootCarry(NamedTuple):
    bracket: BracketState
    best_x: Float[Array, ""]
    best_fx: Float[Array, ""]
    converged: Array


def _raise_if_concrete_false(predicate, message: str) -> None:
    """Raise eagerly when a validation predicate is concrete and false."""
    result = try_concrete_bool(jnp.asarray(predicate))
    if result is False:
        raise ValueError(message)


def initialize_bracket(
    lo: Union[float, Float[Array, "..."]],
    hi: Union[float, Float[Array, "..."]],
    f_lo: Union[float, Float[Array, "..."]],
    f_hi: Union[float, Float[Array, "..."]],
) -> BracketState:
    """Build typed bracket evidence from already-evaluated endpoints."""
    dtype = jnp.result_type(lo, hi, f_lo, f_hi)
    lo = jnp.asarray(lo, dtype=dtype)
    hi = jnp.asarray(hi, dtype=dtype)
    f_lo = jnp.asarray(f_lo, dtype=dtype)
    f_hi = jnp.asarray(f_hi, dtype=dtype)
    finite = jnp.isfinite(lo) & jnp.isfinite(hi)
    finite &= jnp.isfinite(f_lo) & jnp.isfinite(f_hi)
    sign_change = (f_lo == 0.0) | (f_hi == 0.0)
    sign_change |= jnp.signbit(f_lo) != jnp.signbit(f_hi)
    bracketed = finite & (lo <= hi) & sign_change
    return BracketState(lo, hi, f_lo, f_hi, bracketed)


def update_bracket(
    state: BracketState,
    x: Union[float, Float[Array, "..."]],
    fx: Union[float, Float[Array, "..."]],
    *,
    valid: bool | Array = True,
) -> BracketState:
    """Update one bracket endpoint, or preserve every field when inadmissible.

    ``valid=False`` is an exact no-op. This lets a caller exclude a trial whose
    residual is finite but whose external state is inadmissible.
    """
    x = jnp.asarray(x, dtype=state.lo.dtype)
    fx = jnp.asarray(fx, dtype=state.f_lo.dtype)
    admissible = jnp.asarray(valid, dtype=bool) & state.bracketed
    admissible &= jnp.isfinite(x) & jnp.isfinite(fx)
    admissible &= (x >= state.lo) & (x <= state.hi)

    exact = admissible & (fx == 0.0)
    same_as_lo = jnp.signbit(fx) == jnp.signbit(state.f_lo)
    replace_lo = admissible & ~exact & same_as_lo
    replace_hi = admissible & ~exact & ~same_as_lo

    lo = jnp.where(exact | replace_lo, x, state.lo)
    hi = jnp.where(exact | replace_hi, x, state.hi)
    f_lo = jnp.where(exact, jnp.zeros_like(fx), state.f_lo)
    f_lo = jnp.where(replace_lo, fx, f_lo)
    f_hi = jnp.where(exact, jnp.zeros_like(fx), state.f_hi)
    f_hi = jnp.where(replace_hi, fx, f_hi)
    return BracketState(lo, hi, f_lo, f_hi, state.bracketed)


def propose_bracketed(
    state: BracketState,
    *,
    safeguard_fraction: float | Float[Array, ""] = 0.1,
) -> BracketProposal:
    """Propose a safely interior secant point or deterministic midpoint.

    Proposal kinds are integer constants. Exact endpoint roots take priority,
    with the lower endpoint winning ties. A non-root secant is accepted only
    when its denominator and value are finite, it lies strictly in the bracket,
    and it leaves at least ``safeguard_fraction`` of the current width on both
    sides. Otherwise the midpoint is returned with ``safeguarded=True``.
    """
    dtype = jnp.result_type(state.lo, state.hi, state.f_lo, state.f_hi)
    state = BracketState(
        jnp.asarray(state.lo, dtype=dtype),
        jnp.asarray(state.hi, dtype=dtype),
        jnp.asarray(state.f_lo, dtype=dtype),
        jnp.asarray(state.f_hi, dtype=dtype),
        jnp.asarray(state.bracketed, dtype=bool),
    )
    fraction = jnp.asarray(safeguard_fraction, dtype=state.lo.dtype)
    _raise_if_concrete_false(
        jnp.all((fraction >= 0.0) & (fraction < 0.5)),
        "safeguard_fraction must satisfy 0 <= value < 0.5",
    )

    bracketed = jnp.asarray(state.bracketed, dtype=bool)
    lo_root = bracketed & (state.f_lo == 0.0)
    hi_root = bracketed & ~lo_root & (state.f_hi == 0.0)
    endpoint_root = lo_root | hi_root

    midpoint = 0.5 * state.lo + 0.5 * state.hi
    denominator = state.f_hi - state.f_lo
    scale = jnp.maximum(jnp.abs(state.f_lo), jnp.abs(state.f_hi))
    eps = jnp.finfo(state.lo.dtype).eps
    denominator_ok = jnp.isfinite(denominator)
    denominator_ok &= jnp.abs(denominator) > 8.0 * eps * scale
    denominator_safe = jnp.where(denominator_ok, denominator, 1.0)
    width = state.hi - state.lo
    secant = state.lo - (state.f_lo / denominator_safe) * width
    in_bracket = (secant > state.lo) & (secant < state.hi)
    progress_lo = state.lo + fraction * width
    progress_hi = state.hi - fraction * width
    adequate_progress = (secant >= progress_lo) & (secant <= progress_hi)
    use_secant = bracketed & ~endpoint_root & denominator_ok
    use_secant &= jnp.isfinite(secant) & in_bracket & adequate_progress

    candidate = jnp.where(use_secant, secant, midpoint)
    kind = jnp.where(use_secant, PROPOSAL_SECANT, PROPOSAL_MIDPOINT)
    safeguarded = bracketed & ~endpoint_root & ~use_secant
    candidate = jnp.where(lo_root, state.lo, candidate)
    candidate = jnp.where(hi_root, state.hi, candidate)
    kind = jnp.where(lo_root, PROPOSAL_LO_ENDPOINT, kind)
    kind = jnp.where(hi_root, PROPOSAL_HI_ENDPOINT, kind)
    candidate = jnp.where(bracketed, candidate, jnp.nan)
    kind = jnp.where(bracketed, kind, PROPOSAL_NONE).astype(jnp.int32)
    safeguarded = jnp.where(bracketed, safeguarded, False)
    return BracketProposal(candidate, kind, safeguarded)


def safeguarded_bracketed_root(
    f: ScalarFn,
    lo: Union[float, Float[Array, ""]],
    hi: Union[float, Float[Array, ""]],
    *,
    max_steps: int,
    atol: float | Float[Array, ""] = 0.0,
    rtol: float | Float[Array, ""] = 1.0e-8,
    safeguard_fraction: float | Float[Array, ""] = 0.1,
) -> BracketedRootResult:
    """Solve a scalar sign bracket with guarded secant/midpoint proposals.

    The scan length is static, but ``f`` is evaluated inside ``lax.cond`` only
    while a verified bracket remains unconverged. Convergence means an exact
    zero residual or a bracket half-width no larger than
    ``atol + rtol * abs(best_x)``. Exhaustion returns the best finite evaluated
    point with ``converged=False``. A missing initial bracket returns NaN root
    evidence, never a fabricated solution.

    This is a value-first branch-selected numerical map. It does not implement
    implicit differentiation and makes no derivative claim for parameters
    captured by ``f``.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    lo = jnp.asarray(lo)
    hi = jnp.asarray(hi)
    f_lo = jnp.asarray(f(lo))
    f_hi = jnp.asarray(f(hi))
    bracket = initialize_bracket(lo, hi, f_lo, f_hi)
    atol = jnp.asarray(atol, dtype=bracket.lo.dtype)
    rtol = jnp.asarray(rtol, dtype=bracket.lo.dtype)
    _raise_if_concrete_false(jnp.all(atol >= 0.0), "atol must be nonnegative")
    _raise_if_concrete_false(jnp.all(rtol >= 0.0), "rtol must be nonnegative")

    hi_is_better = jnp.abs(bracket.f_hi) < jnp.abs(bracket.f_lo)
    best_x = jnp.where(hi_is_better, bracket.hi, bracket.lo)
    best_fx = jnp.where(hi_is_better, bracket.f_hi, bracket.f_lo)
    endpoint_root = bracket.bracketed & (best_fx == 0.0)
    initial = _RootCarry(bracket, best_x, best_fx, endpoint_root)
    nan = jnp.asarray(jnp.nan, dtype=bracket.lo.dtype)

    def scan_step(carry: _RootCarry, _):
        active = carry.bracket.bracketed & ~carry.converged
        proposal = propose_bracketed(
            carry.bracket, safeguard_fraction=safeguard_fraction
        )

        def evaluate(x):
            return jnp.asarray(f(x), dtype=bracket.lo.dtype)

        fx = lax.cond(active, evaluate, lambda _: nan, proposal.x)
        executed = active
        admissible = executed & jnp.isfinite(fx)
        updated = update_bracket(carry.bracket, proposal.x, fx, valid=admissible)

        better = admissible & (jnp.abs(fx) < jnp.abs(carry.best_fx))
        best_x_new = jnp.where(better, proposal.x, carry.best_x)
        best_fx_new = jnp.where(better, fx, carry.best_fx)
        tolerance = atol + rtol * jnp.abs(best_x_new)
        width_converged = admissible & (updated.hi - updated.lo <= 2.0 * tolerance)
        converged_now = active & ((admissible & (fx == 0.0)) | width_converged)
        next_carry = _RootCarry(
            updated,
            best_x_new,
            best_fx_new,
            carry.converged | converged_now,
        )

        proposal_x = jnp.where(executed, proposal.x, nan)
        residual = jnp.where(executed, fx, nan)
        trace_lo = jnp.where(executed, updated.lo, nan)
        trace_hi = jnp.where(executed, updated.hi, nan)
        trace_f_lo = jnp.where(executed, updated.f_lo, nan)
        trace_f_hi = jnp.where(executed, updated.f_hi, nan)
        proposal_kind = jnp.where(executed, proposal.kind, PROPOSAL_NONE).astype(
            jnp.int32
        )
        trace = RootTrace(
            proposal_x,
            residual,
            trace_lo,
            trace_hi,
            trace_f_lo,
            trace_f_hi,
            proposal_kind,
            executed,
            admissible,
            converged_now,
        )
        return next_carry, trace

    final, trace = lax.scan(scan_step, initial, None, length=max_steps)
    bracketed = final.bracket.bracketed
    root = jnp.where(bracketed, final.best_x, nan)
    residual = jnp.where(bracketed, final.best_fx, nan)
    n_evaluations = jnp.asarray(2, dtype=jnp.int32) + jnp.sum(
        trace.executed, dtype=jnp.int32
    )
    return BracketedRootResult(
        root,
        residual,
        final.converged,
        bracketed,
        n_evaluations,
        trace,
    )


def bracket_expand(
    f: ScalarFn,
    x0: Union[float, Float[Array, "..."]],
    *,
    step: Union[float, Float[Array, "..."]] = 1.0,
    growth: Union[float, Float[Array, "..."]] = 2.0,
    max_steps: int = 32,
) -> tuple[Float[Array, "..."], Float[Array, "..."], Array]:
    """
    Expand a symmetric bracket around ``x0`` until ``f`` changes sign.

    The expansion uses a fixed ``max_steps`` ``lax.scan`` so it is compatible
    with ``jit`` and ``vmap``. The returned ``found`` mask is ``True`` where
    a sign-changing bracket was found. If no bracket is found, ``lo`` and
    ``hi`` are the final expanded endpoints.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    x0 = jnp.asarray(x0)
    step = jnp.asarray(step)
    growth = jnp.asarray(growth)
    _raise_if_concrete_false(jnp.all(step > 0.0), "step must be positive")
    _raise_if_concrete_false(jnp.all(growth > 1.0), "growth must be greater than 1")

    scan_dtype = jnp.result_type(x0, step, growth, jnp.float32)
    step_ids = jnp.arange(max_steps, dtype=scan_dtype)
    f0 = f(x0)
    init_found = jnp.asarray(f0 == 0.0, dtype=bool)
    init = (x0, x0, init_found)

    def scan_step(carry, k):
        lo, hi, found = carry
        radius = step * jnp.power(growth, k)
        cand_lo = x0 - radius
        cand_hi = x0 + radius
        f_lo = f(cand_lo)
        f_hi = f(cand_hi)
        has_bracket = jnp.sign(f_lo) * jnp.sign(f_hi) <= 0.0

        lo_new = jnp.where(found, lo, cand_lo)
        hi_new = jnp.where(found, hi, cand_hi)
        found_new = found | has_bracket
        return (lo_new, hi_new, found_new), None

    (lo, hi, found), _ = lax.scan(scan_step, init, step_ids)
    return lo, hi, found


def bisect(
    f: ScalarFn,
    a: Union[float, Float[Array, "..."]],
    b: Union[float, Float[Array, "..."]],
    max_steps: int = 50,
) -> Float[Array, "..."]:
    """
    Bisection method for a scalar root of f in [a, b].

    Parameters
    ----------
    f : callable
        Scalar function f(x) -> scalar.
    a : Array
        Left bracket (f(a) and f(b) should have opposite signs).
    b : Array
        Right bracket.
    max_steps : int
        Number of bisection iterations. Default 50 gives ~15 decimal
        digits of precision.

    Returns
    -------
    Array
        Approximate root of f.

    Notes
    -----
    Uses lax.scan with fixed iteration count for full compatibility
    with jit, vmap, and grad. Does not terminate early.

    Example
    -------
    >>> @jax.jit
    ... def find_sqrt2(a, b):
    ...     return bisect(lambda x: x**2 - 2.0, a, b)
    >>> find_sqrt2(1.0, 2.0)  # Returns ~1.414...
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    fa = f(a)
    fb = f(b)

    def step(carry, _):
        a_, b_, fa_, fb_ = carry
        mid = 0.5 * (a_ + b_)
        fm = f(mid)

        # Branchless selection of which half contains the root
        left_has_root = jnp.sign(fa_) * jnp.sign(fm) <= 0

        a_new = jnp.where(left_has_root, a_, mid)
        fa_new = jnp.where(left_has_root, fa_, fm)
        b_new = jnp.where(left_has_root, mid, b_)
        fb_new = jnp.where(left_has_root, fm, fb_)

        return (a_new, b_new, fa_new, fb_new), None

    init_carry = (a, b, fa, fb)
    (a_f, b_f, _, _), _ = lax.scan(step, init_carry, None, length=max_steps)
    return 0.5 * (a_f + b_f)


def bisect_many(
    f: Callable[[Float[Array, "..."]], Float[Array, "..."]],
    a: Union[float, Float[Array, "..."]],
    b: Union[float, Float[Array, "..."]],
    max_steps: int = 50,
) -> Float[Array, "..."]:
    """
    Solve independent bisection brackets with array-shaped endpoints.

    This is an explicit vectorized wrapper around :func:`bisect`. The callable
    ``f`` should accept an array of candidate roots and return an array of
    residuals with the same broadcast shape.
    """
    return bisect(f, a, b, max_steps=max_steps)


def newton(
    f: ScalarFn,
    x0: Union[float, Float[Array, "..."]],
    max_steps: int = 30,
) -> Float[Array, "..."]:
    """
    Newton's method with automatic derivative computation.

    Parameters
    ----------
    f : callable
        Scalar function f(x) -> scalar. Must be differentiable.
    x0 : Array
        Initial guess.
    max_steps : int
        Number of Newton iterations.

    Returns
    -------
    Array
        Approximate root of f.

    Notes
    -----
    Computes the derivative automatically via jax.grad(f). Uses
    lax.scan with fixed iteration count for full compatibility
    with jit, vmap, and grad.

    For custom derivatives (e.g., when the automatic derivative is
    expensive or you have an analytical form), use newton_with_grad.

    Example
    -------
    >>> @jax.jit
    ... def find_sqrt2(x0):
    ...     return newton(lambda x: x**2 - 2.0, x0)
    >>> find_sqrt2(1.5)  # Returns ~1.414...
    """
    x0 = jnp.asarray(x0)
    df = jax.grad(f)

    def step(x, _):
        fx = f(x)
        dfx = df(x)
        # Safe division to avoid NaN if derivative is zero
        dfx_safe = jnp.where(dfx == 0.0, 1.0, dfx)
        x_new = x - fx / dfx_safe
        return x_new, None

    x_f, _ = lax.scan(step, x0, None, length=max_steps)
    return x_f


def newton_with_grad(
    f: ScalarFn,
    df: ScalarFn,
    x0: Union[float, Float[Array, "..."]],
    max_steps: int = 30,
) -> Float[Array, "..."]:
    """
    Newton's method with user-provided derivative.

    Parameters
    ----------
    f : callable
        Scalar function f(x) -> scalar.
    df : callable
        Derivative df(x) -> scalar.
    x0 : Array
        Initial guess.
    max_steps : int
        Number of Newton iterations.

    Returns
    -------
    Array
        Approximate root of f.

    Notes
    -----
    Use this when you have an analytical derivative that's faster
    than automatic differentiation, or when the function is not
    directly differentiable by JAX.

    Uses lax.scan with fixed iteration count for full compatibility
    with jit, vmap, and grad.

    Example
    -------
    >>> @jax.jit
    ... def find_sqrt2(x0):
    ...     return newton_with_grad(
    ...         lambda x: x**2 - 2.0,
    ...         lambda x: 2.0 * x,
    ...         x0
    ...     )
    >>> find_sqrt2(1.5)  # Returns ~1.414...
    """
    x0 = jnp.asarray(x0)

    def step(x, _):
        fx = f(x)
        dfx = df(x)
        # Safe division to avoid NaN if derivative is zero
        dfx_safe = jnp.where(dfx == 0.0, 1.0, dfx)
        x_new = x - fx / dfx_safe
        return x_new, None

    x_f, _ = lax.scan(step, x0, None, length=max_steps)
    return x_f


def newton_ppf(
    u: Union[float, Float[Array, "..."]],
    cdf: ScalarFn,
    x0: Union[float, Float[Array, "..."]],
    lo: Union[float, Float[Array, "..."]],
    hi: Union[float, Float[Array, "..."]],
    pdf: Optional[ScalarFn] = None,
    n_iter: int = 20,
    pdf_floor: float = 1e-30,
) -> Float[Array, "..."]:
    r"""
    Generic percent-point-function (PPF / inverse-CDF) solver via Newton.

    Solves ``cdf(x) = u`` for ``x`` using a fixed number of Newton
    iterations. This is the inverse-CDF inversion at the heart of
    reparameterized sampling (e.g. IMF sampling): given a uniform draw
    ``u`` in ``(0, 1)``, it returns the quantile ``x = F^{-1}(u)``.

    Algorithm
    ---------
    Newton's method applied to the residual ``g(x) = cdf(x) - u``.
    Since ``g'(x) = cdf'(x) = pdf(x)``, each step is::

        x_{k+1} = x_k - (cdf(x_k) - u) / pdf(x_k)

    The PDF (CDF derivative) is supplied via ``pdf`` if available, else
    obtained automatically as ``jax.grad(cdf)`` (vmapped to handle array
    ``x``). The update is clipped to ``[lo, hi]`` after every step so the
    iterate never leaves the support. A small ``pdf_floor`` guards the
    division when the density is near zero (flat-CDF regions), preventing
    NaNs without biasing well-conditioned steps.

    This is deliberately decoupled from any specific distribution: callers
    pass their own ``cdf`` (closing over distribution parameters), an
    initial guess ``x0``, and the support bounds. A good ``x0`` (e.g. a
    closed-form approximate inverse) reduces the iterations needed.

    Parameters
    ----------
    u : Array
        Target CDF value(s) in ``(0, 1)``. Shape broadcasts with ``x0``.
    cdf : callable
        Cumulative distribution function ``cdf(x) -> F(x)``. Must be
        differentiable if ``pdf`` is not supplied.
    x0 : Array
        Initial guess for the quantile(s), same shape as ``u``.
    lo, hi : Array
        Lower/upper bounds of the support; iterates are clipped to
        ``[lo, hi]``.
    pdf : callable, optional
        Density ``pdf(x) = cdf'(x)``. If ``None`` (default), computed via
        automatic differentiation of ``cdf``.
    n_iter : int
        Number of Newton iterations (fixed; NOT convergence-based). Default
        20 gives ample precision for smooth unimodal CDFs from a reasonable
        guess.
    pdf_floor : float
        Additive floor on the density in the denominator (default 1e-30)
        guarding against division by ~0 in flat-CDF regions.

    Returns
    -------
    Array
        Quantile(s) ``x`` such that ``cdf(x) ~= u``.

    Notes
    -----
    Fully compatible with ``jit``, ``vmap``, and ``grad``: uses ``lax.scan``
    with a fixed iteration count (no ``while_loop``), additive (not
    branching) safe division, and ``clip`` for bounds. Gradients flow
    through every iteration, so the result is differentiable BOTH w.r.t.
    ``u`` AND w.r.t. any distribution parameters captured inside ``cdf``
    (or ``pdf``) — verified by FD-vs-AD grad-checks against the analytic
    exponential PPF.

    Example
    -------
    >>> import jax.numpy as jnp
    >>> # Exponential: F(x) = 1 - exp(-lam x), true PPF = -ln(1-u)/lam
    >>> lam = 2.0
    >>> u = jnp.array([0.1, 0.5, 0.9])
    >>> x = newton_ppf(u, lambda x: 1.0 - jnp.exp(-lam * x),
    ...                x0=jnp.ones_like(u), lo=0.0, hi=100.0)
    >>> jnp.allclose(x, -jnp.log1p(-u) / lam, atol=1e-6)
    Array(True, dtype=bool)
    """
    u = jnp.asarray(u)
    x0 = jnp.asarray(x0)
    lo = jnp.asarray(lo)
    hi = jnp.asarray(hi)

    if pdf is None:
        # Elementwise derivative of the (possibly vectorized) CDF. vmap over
        # a raveled view so jax.grad sees a scalar->scalar map, then restore
        # shape. This keeps the AD path correct for array-valued x.
        grad_scalar = jax.grad(lambda xi: jnp.reshape(cdf(xi), ()))

        def pdf_fn(x):
            x_arr = jnp.asarray(x)
            flat = jax.vmap(grad_scalar)(x_arr.ravel())
            return flat.reshape(x_arr.shape)
    else:
        pdf_fn = pdf

    def step(x, _):
        residual = cdf(x) - u
        dens = pdf_fn(x)
        x_new = x - residual / (dens + pdf_floor)
        x_new = jnp.clip(x_new, lo, hi)
        return x_new, None

    x_f, _ = lax.scan(step, x0, None, length=n_iter)
    return x_f


def monotone_inverse_interp(
    x: Float[Array, " n"],
    y: Float[Array, " n"],
    y_new: Union[float, Float[Array, "..."]],
) -> Float[Array, "..."]:
    """
    Invert a strictly monotone tabulated function by linear interpolation.

    ``x`` and ``y`` define a one-dimensional table with strictly increasing
    coordinates and values. Queries outside ``[y[0], y[-1]]`` clamp to the
    corresponding endpoint of ``x``.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    y_new = jnp.asarray(y_new)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")
    if x.shape[0] < 2:
        raise ValueError("x and y must contain at least two samples")

    _raise_if_concrete_false(
        jnp.all(jnp.diff(x) > 0.0), "x must be strictly increasing"
    )
    _raise_if_concrete_false(
        jnp.all(jnp.diff(y) > 0.0), "y must be strictly increasing"
    )

    idx = jnp.searchsorted(y, y_new, side="right") - 1
    idx = jnp.clip(idx, 0, y.shape[0] - 2)

    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    denom = jnp.where(y1 == y0, 1.0, y1 - y0)
    t = (y_new - y0) / denom
    out = x0 + t * (x1 - x0)
    return jnp.where(y_new <= y[0], x[0], jnp.where(y_new >= y[-1], x[-1], out))


# Keep newton_1d as alias for backwards compatibility
newton_1d = newton_with_grad

__all__ = [
    "PROPOSAL_NONE",
    "PROPOSAL_SECANT",
    "PROPOSAL_MIDPOINT",
    "PROPOSAL_LO_ENDPOINT",
    "PROPOSAL_HI_ENDPOINT",
    "BracketState",
    "BracketProposal",
    "RootTrace",
    "BracketedRootResult",
    "initialize_bracket",
    "update_bracket",
    "propose_bracketed",
    "safeguarded_bracketed_root",
    "bracket_expand",
    "bisect",
    "bisect_many",
    "newton",
    "newton_with_grad",
    "newton_1d",
    "newton_ppf",
    "monotone_inverse_interp",
]
