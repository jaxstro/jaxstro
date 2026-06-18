import jax, jax.numpy as jnp
from jaxstro.params.transforms import Identity, Exp, Softplus, Sigmoid

import pytest
@pytest.mark.parametrize("bij,x", [(Identity(), 0.7), (Exp(), 2.3), (Softplus(), 1.5), (Sigmoid(0.0, 1.0), 0.3)])
def test_roundtrip(bij, x):
    x = jnp.asarray(x)
    assert jnp.allclose(bij.forward(bij.inverse(x)), x, rtol=1e-10)

@pytest.mark.parametrize("bij,u", [(Exp(), 0.4), (Softplus(), -1.2), (Sigmoid(2.0, 5.0), 0.8), (Identity(), 1.1)])
def test_log_det_matches_autodiff(bij, u):
    u = jnp.asarray(u)
    ad = jnp.log(jnp.abs(jax.grad(lambda z: bij.forward(z))(u)))
    assert jnp.allclose(bij.forward_log_det_jacobian(u), ad, rtol=1e-6, atol=1e-8)

def test_sigmoid_bounds():
    s = Sigmoid(2.0, 5.0)
    us = jnp.linspace(-20, 20, 50)
    ys = jax.vmap(s.forward)(us)
    assert jnp.all((ys > 2.0) & (ys < 5.0))

def test_exp_positive():
    assert jnp.all(jax.vmap(Exp().forward)(jnp.linspace(-30, 30, 50)) > 0.0)
