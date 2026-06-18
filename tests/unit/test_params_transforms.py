import jax
import jax.numpy as jnp
import pytest

from jaxstro.params.transforms import Exp, Identity, Sigmoid, Softplus


@pytest.mark.parametrize(
    "bij,x",
    [(Identity(), 0.7), (Exp(), 2.3), (Softplus(), 1.5), (Sigmoid(0.0, 1.0), 0.3)],
)
def test_roundtrip(bij, x):
    x = jnp.asarray(x)
    assert jnp.allclose(bij.forward(bij.inverse(x)), x, rtol=1e-10)


@pytest.mark.parametrize(
    "bij,u",
    [(Exp(), 0.4), (Softplus(), -1.2), (Sigmoid(2.0, 5.0), 0.8), (Identity(), 1.1)],
)
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


# --- extreme-u stability (the regime the analytic log-dets exist for; AD diverges here) ---


@pytest.mark.parametrize("bij", [Identity(), Exp(), Softplus(), Sigmoid(2.0, 5.0)])
@pytest.mark.parametrize("u", [-40.0, -20.0, 20.0, 40.0])
def test_log_det_finite_at_extremes(bij, u):
    """Analytic log-det stays finite where autodiff underflows to -inf/0
    (e.g. Sigmoid u=+40 -> AD gives -inf; Softplus u=+40 -> AD gives 0)."""
    ld = bij.forward_log_det_jacobian(jnp.asarray(u))
    assert jnp.isfinite(ld)


def test_sigmoid_log_det_closed_form_at_extreme():
    """log(hi-lo) + log_sigmoid(u) + log_sigmoid(-u) -> log(hi-lo) - |u| as |u|->inf
    (to O(e^-|u|)); the closed form AD cannot reach (it returns -inf)."""
    s = Sigmoid(2.0, 5.0)
    for u in (-40.0, 40.0):
        expected = jnp.log(3.0) - jnp.abs(jnp.asarray(u))  # log(hi-lo) - |u|
        assert jnp.allclose(
            s.forward_log_det_jacobian(jnp.asarray(u)), expected, atol=1e-14
        )


@pytest.mark.parametrize("x", [1e-12, 1e-6, 1.0, 50.0, 700.0, 740.0])
def test_softplus_inverse_stable_small_and_large_x(x):
    """Softplus.inverse round-trips at both extremes: accurate near 0+ and no
    overflow for large x (log(expm1(x)) would give inf for x>~709)."""
    x = jnp.asarray(x)
    assert jnp.isfinite(Softplus().inverse(x))
    assert jnp.allclose(Softplus().forward(Softplus().inverse(x)), x, rtol=1e-10)
