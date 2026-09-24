"""Trace-aware helpers shared by jaxstro subpackages.

A leaf module: it imports only JAX, so ``jaxstro.quad`` and ``jaxstro.numerics``
can both use it without importing each other. ``try_concrete_bool`` stays public
through ``jaxstro.numerics.checks``.
"""

from __future__ import annotations

from typing import Optional

import jax
from jaxtyping import Array, Bool


def try_concrete_bool(predicate: Bool[Array, "..."]) -> Optional[bool]:
    """
    Resolve a JAX boolean predicate to a Python ``bool`` when possible.

    Returns the concrete ``bool`` when ``predicate`` can be evaluated at the
    Python level (eager / non-traced execution), and ``None`` when it cannot —
    i.e. when called inside an active JAX trace (``jit``/``vmap``/``grad``),
    where the value is unknown at trace time.

    This is the building block for *eager/debug* input validation: a library
    function can raise on a concrete bad input but must silently skip the check
    under transformation, because a value-dependent ``raise`` cannot fire on a
    tracer. ``isinstance(x, Tracer)`` alone is insufficient, because a concrete
    array closed over inside a jitted caller is not itself a tracer yet ``bool``
    on it still raises within the active trace; catching the conversion error is
    the robust signal.
    """
    try:
        return bool(predicate)
    except jax.errors.ConcretizationTypeError:
        return None
    except jax.errors.TracerBoolConversionError:
        return None
