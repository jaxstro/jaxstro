"""Value and derivative contracts for the first onboarding map."""

from __future__ import annotations

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "examples" / "onboarding" / "first_jax_map.py"


def test_scaled_luminosity_value_and_gradient() -> None:
    assert MODULE.is_file(), "the executable onboarding map has not been added"
    scaled_luminosity = importlib.import_module(
        "examples.onboarding.first_jax_map"
    ).scaled_luminosity

    value = scaled_luminosity(2.0, 0.5)
    d_radius, d_temperature = jax.grad(scaled_luminosity, argnums=(0, 1))(2.0, 0.5)
    assert jnp.allclose(value, 0.25)
    assert jnp.allclose(d_radius, 0.25)
    assert jnp.allclose(d_temperature, 2.0)
