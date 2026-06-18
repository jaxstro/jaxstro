import jax, jax.numpy as jnp, equinox as eqx
from jaxstro.params import Parameterization
from jaxstro.params.transforms import Exp, Sigmoid

class M(eqx.Module):
    r_h: jax.Array
    Q: jax.Array
def _m(): return M(r_h=jnp.array(1.3), Q=jnp.array(0.4))

def test_transformed_roundtrip_and_bounds():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.r_h, x.Q), transforms=(Exp(), Sigmoid(0.0, 1.0)))
    m2 = p.from_vector(m, p.to_vector(m))
    assert jnp.allclose(m2.r_h, m.r_h) and jnp.allclose(m2.Q, m.Q)
    # vector lives in unconstrained R; perturb and confirm bounds hold after forward
    m3 = p.from_vector(m, p.to_vector(m) + jnp.array([5.0, -8.0]))
    assert m3.r_h > 0.0 and 0.0 < m3.Q < 1.0

def test_log_det_jacobian_sums_per_leaf():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.r_h, x.Q), transforms=(Exp(), Sigmoid(0.0, 1.0)))
    v = p.to_vector(m)
    expected = Exp().forward_log_det_jacobian(v[0]) + Sigmoid(0.0,1.0).forward_log_det_jacobian(v[1])
    assert jnp.allclose(p.log_det_jacobian(v), expected, rtol=1e-8)

def test_end_to_end_grad_through_transformed():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.r_h, x.Q), transforms=(Exp(), Sigmoid(0.0, 1.0)))
    v0 = p.to_vector(m)
    def loss(v): mm = p.from_vector(m, v); return mm.r_h**2 + mm.Q**2
    g = jax.grad(loss)(v0); assert jnp.all(jnp.isfinite(g))

def test_transform_follows_leaf_not_tuple_order():
    """REGRESSION (T1 review [Important]): a transform attached to a leaf via `where`
    must apply to THAT leaf regardless of where-tuple order or field-declaration order.
    Here r_h is declared first but selected SECOND, with Exp; Q selected first with Sigmoid.
    Exp must still govern r_h (so a large unconstrained value -> large positive r_h) and
    Sigmoid must still bound Q in (0,1)."""
    m = _m()
    # where-tuple order (Q, r_h) is the REVERSE of declaration order (r_h, Q)
    p = Parameterization.from_where(m, where=lambda x: (x.Q, x.r_h), transforms=(Sigmoid(0.0, 1.0), Exp()))
    m2 = p.from_vector(m, p.to_vector(m))
    assert jnp.allclose(m2.r_h, m.r_h) and jnp.allclose(m2.Q, m.Q)        # round-trip exact
    # Build the unconstrained vector in PyTree-leaf order and push both leaves up hard:
    big = p.to_vector(m) + 8.0
    m3 = p.from_vector(m, big)
    assert m3.r_h > m.r_h            # Exp governs r_h -> grows
    assert 0.0 < m3.Q < 1.0         # Sigmoid still bounds Q despite +8.0 push
