"""Small transformed JAX map used by the beginner documentation."""

import jax
import jax.numpy as jnp


def scaled_luminosity(radius_ratio: float, temperature_ratio: float):
    """Return L/L_ref = (R/R_ref)^2 (T/T_ref)^4."""
    radius = jnp.asarray(radius_ratio)
    temperature = jnp.asarray(temperature_ratio)
    return radius**2 * temperature**4


batched_scaled_luminosity = jax.vmap(scaled_luminosity)
compiled_scaled_luminosity = jax.jit(scaled_luminosity)
