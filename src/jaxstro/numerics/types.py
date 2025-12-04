# src/jaxstro/numerics/types.py
"""
Common type aliases for jaxstro.numerics.

These are lightweight conveniences for annotations across the
numerics subpackage and downstream users.
"""

from typing import Callable, TypeAlias

import jax.numpy as jnp

Array = jnp.ndarray
ScalarFn: TypeAlias = Callable[[Array], Array]

__all__ = ["Array", "ScalarFn"]
