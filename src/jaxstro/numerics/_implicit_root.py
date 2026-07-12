"""Certificate types for explicitly gated implicit scalar roots."""

from typing import Any, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float

DERIVATIVE_STATUS_CERTIFIED = 0
DERIVATIVE_STATUS_PRIMAL_FAILED = 1
DERIVATIVE_STATUS_ASSUMPTIONS_REJECTED = 2
DERIVATIVE_STATUS_NONFINITE = 3
DERIVATIVE_STATUS_RESIDUAL_TOO_LARGE = 4
DERIVATIVE_STATUS_SLOPE_ILL_CONDITIONED = 5
DERIVATIVE_STATUS_BRACKET_TOO_WIDE = 6


class ImplicitRootAssumptions(NamedTuple):
    """Caller assertions that cannot be inferred from one local solve."""

    unique_root: Array
    smooth_branch: Array


class ImplicitRootCertificate(NamedTuple):
    """Array-valued evidence required before exposing an IFT derivative."""

    finite: Array
    primal_converged: Array
    unique_root_asserted: Array
    smooth_branch_asserted: Array
    residual_ok: Array
    slope_ok: Array
    width_ok: Array
    certified: Array
    residual_limit: Float[Array, ""]
    width_limit: Float[Array, ""]
    slope_floor: Float[Array, ""]


class ImplicitRootResult(NamedTuple):
    """Derivative-facing result plus the complete value-solver evidence."""

    root: Float[Array, ""]
    residual: Float[Array, ""]
    slope: Float[Array, ""]
    status: Array
    certified: Array
    certificate: ImplicitRootCertificate
    primal: Any


def _build_implicit_certificate(
    primal: Any,
    slope: Float[Array, ""],
    assumptions: ImplicitRootAssumptions,
    *,
    residual_atol: float | Float[Array, ""],
    residual_rtol: float | Float[Array, ""],
    width_atol: float | Float[Array, ""],
    width_rtol: float | Float[Array, ""],
    slope_floor: float | Float[Array, ""],
) -> tuple[ImplicitRootCertificate, Array]:
    """Construct the deterministic fail-closed derivative truth table."""
    dtype = primal.root.dtype
    slope = jnp.asarray(slope, dtype=dtype)
    residual_limit = jnp.asarray(residual_atol, dtype=dtype)
    residual_limit += jnp.asarray(residual_rtol, dtype=dtype) * primal.residual_scale
    width_limit = jnp.asarray(width_atol, dtype=dtype)
    width_limit += jnp.asarray(width_rtol, dtype=dtype) * jnp.abs(primal.root)
    slope_floor = jnp.asarray(slope_floor, dtype=dtype)
    unique_root = jnp.asarray(assumptions.unique_root, dtype=bool)
    smooth_branch = jnp.asarray(assumptions.smooth_branch, dtype=bool)
    finite = (
        jnp.isfinite(primal.root) & jnp.isfinite(primal.residual) & jnp.isfinite(slope)
    )
    residual_ok = jnp.abs(primal.residual) <= residual_limit
    slope_ok = jnp.abs(slope) >= slope_floor
    width = primal.final_bracket.hi - primal.final_bracket.lo
    width_ok = width <= width_limit
    certified = (
        finite
        & primal.converged
        & unique_root
        & smooth_branch
        & residual_ok
        & slope_ok
        & width_ok
    )
    certificate = ImplicitRootCertificate(
        finite,
        primal.converged,
        unique_root,
        smooth_branch,
        residual_ok,
        slope_ok,
        width_ok,
        certified,
        residual_limit,
        width_limit,
        slope_floor,
    )
    status = jnp.asarray(DERIVATIVE_STATUS_CERTIFIED, dtype=jnp.int32)
    status = jnp.where(~width_ok, DERIVATIVE_STATUS_BRACKET_TOO_WIDE, status)
    status = jnp.where(~slope_ok, DERIVATIVE_STATUS_SLOPE_ILL_CONDITIONED, status)
    status = jnp.where(~residual_ok, DERIVATIVE_STATUS_RESIDUAL_TOO_LARGE, status)
    status = jnp.where(~finite, DERIVATIVE_STATUS_NONFINITE, status)
    status = jnp.where(
        ~(unique_root & smooth_branch),
        DERIVATIVE_STATUS_ASSUMPTIONS_REJECTED,
        status,
    )
    status = jnp.where(
        ~primal.converged, DERIVATIVE_STATUS_PRIMAL_FAILED, status
    ).astype(jnp.int32)
    return certificate, status
