# tests/integration/test_params_optax.py
"""
Integration test: a real optax gradient-descent loop drives
:class:`jaxstro.params.Parameterization` to recover a known scalar from
noiseless synthetic data.

This exercises the full free/fixed + unconstrained-transform + flat-vector
bridge against an actual optimizer (not a hand-rolled gradient step): the loop
optimizes the *unconstrained* vector produced by :meth:`to_vector`, reconstructs
the model via :meth:`from_vector` inside the loss, and recovers the true
physical value to ~1e-3.

optax is only required behind the ``[ml]`` extra; the module skips cleanly when
optax is absent.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from jaxstro.params import Parameterization
from jaxstro.params.transforms import Exp

optax = pytest.importorskip("optax")


class _ScaledExp(eqx.Module):
    """Toy model ``y = amplitude * exp(-rate * x)`` with a positive ``rate``.

    ``rate`` is the free parameter we recover; ``amplitude`` is held fixed to
    confirm fixed leaves are carried through untouched.
    """

    rate: jax.Array
    amplitude: jax.Array

    def predict(self, x: jax.Array) -> jax.Array:
        return self.amplitude * jnp.exp(-self.rate * x)


def test_optax_recovers_positive_scalar() -> None:
    """Recover a known positive ``rate`` via optax over the unconstrained vec."""
    x = jnp.linspace(0.0, 3.0, 64)
    true_rate = 1.7
    amplitude = 2.5
    truth = _ScaledExp(rate=jnp.asarray(true_rate), amplitude=jnp.asarray(amplitude))
    data = truth.predict(x)  # noiseless

    # Start far from the truth; Exp keeps rate strictly positive while we
    # descend in unconstrained R.
    init = _ScaledExp(rate=jnp.asarray(0.3), amplitude=jnp.asarray(amplitude))
    param = Parameterization.from_where(
        init, where=lambda m: (m.rate,), transforms=(Exp(),)
    )

    @jax.jit
    def loss(vec: jax.Array) -> jax.Array:
        model = param.from_vector(init, vec)
        return jnp.mean((model.predict(x) - data) ** 2)

    optimizer = optax.adam(learning_rate=5e-2)
    vec = param.to_vector(init)
    opt_state = optimizer.init(vec)

    @jax.jit
    def step(vec, opt_state):
        grads = jax.grad(loss)(vec)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(vec, updates), opt_state

    for _ in range(2000):
        vec, opt_state = step(vec, opt_state)

    recovered = param.from_vector(init, vec)
    assert jnp.allclose(recovered.rate, true_rate, atol=1e-3)
    # Fixed leaf preserved exactly.
    assert jnp.array_equal(recovered.amplitude, truth.amplitude)
    # Positivity held throughout (Exp transform).
    assert recovered.rate > 0.0
    # Residual loss is essentially zero on noiseless data.
    assert float(loss(vec)) < 1e-7
