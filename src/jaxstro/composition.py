"""Chemical composition by mass fraction: one record for every consumer.

``Composition(X, Y, Z, deuterium_to_hydrogen)`` holds the hydrogen, helium and metal mass
fractions and the deuterium-to-hydrogen NUMBER ratio. Every field is required: a default
composition is how a Population I value reaches a Population III run without anyone
choosing it. Consumers (equation-of-state tables, opacities, protostellar models) read the
fields they need and refuse values outside their own validity; this record checks only
that it is a composition -- fractions in ``[0, 1]`` summing to one, a non-negative D/H.

Added 2026-09-23 for hydrax audit finding F13 (seam S1): one composition record passed
to the EOS, the opacities and the protostellar model, living here so that it survives the
protostellar model's move into stellax (hydrax ADR-0009).

The checks run only on concrete values. A traced composition (a metallicity being
differentiated) passes through unchecked, and its consumer's own validity check is the
guard, as it is for any traced input.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

__all__ = ["Composition", "SUM_TOLERANCE"]

#: Allowed ``|X + Y + Z - 1|``: float64 roundoff on three decimal inputs, not a physical
#: tolerance (0.70 + 0.28 + 0.02 sums to 1 - 1.1e-16).
SUM_TOLERANCE: float = 1.0e-12


class Composition(eqx.Module):
    """Mass fractions ``X, Y, Z`` and the deuterium-to-hydrogen number ratio.

    Attributes:
        X: hydrogen mass fraction.
        Y: helium mass fraction.
        Z: metal mass fraction.
        deuterium_to_hydrogen: D/H by number.
    """

    X: ArrayLike
    Y: ArrayLike
    Z: ArrayLike
    deuterium_to_hydrogen: ArrayLike

    def __check_init__(self) -> None:
        values = (self.X, self.Y, self.Z, self.deuterium_to_hydrogen)
        if any(isinstance(v, jax.core.Tracer) for v in values):
            return
        x, y, z, d_h = (float(jnp.asarray(v)) for v in values)
        for name, value in (("X", x), ("Y", y), ("Z", z)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Composition.{name} must lie in [0, 1], got {value!r}"
                )
        if abs(x + y + z - 1.0) > SUM_TOLERANCE:
            raise ValueError(
                f"Composition mass fractions must sum to one: X + Y + Z = {x + y + z!r} "
                f"(X={x!r}, Y={y!r}, Z={z!r})"
            )
        if not d_h >= 0.0:
            raise ValueError(
                f"Composition.deuterium_to_hydrogen must be >= 0, got {d_h!r}"
            )
