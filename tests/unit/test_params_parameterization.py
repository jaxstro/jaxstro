import jax, jax.numpy as jnp, equinox as eqx
from jaxstro.params import Parameterization

class Model(eqx.Module):
    a: jax.Array
    b: jax.Array
    name: str = eqx.field(static=True, default="m")

def _m(): return Model(a=jnp.array([1.0, 2.0]), b=jnp.array(3.0))

def test_roundtrip_identity():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.a, x.b))
    m2 = p.from_vector(m, p.to_vector(m))
    assert jnp.array_equal(m2.a, m.a) and jnp.array_equal(m2.b, m.b)

def test_fixed_leaf_preserved():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.a,))   # only a is free
    m2 = p.from_vector(m, p.to_vector(m) + 10.0)
    assert jnp.array_equal(m2.b, m.b)                            # b untouched
    assert jnp.allclose(m2.a, m.a + 10.0)

def test_from_where_equals_from_filter():
    m = _m()
    pw = Parameterization.from_where(m, where=lambda x: (x.a, x.b))
    spec = eqx.tree_at(lambda x: (x.a, x.b), jax.tree_util.tree_map(lambda _: False, eqx.filter(m, eqx.is_array)),
                       replace=(True, True))
    pf = Parameterization.from_filter(m, spec)
    assert jnp.array_equal(pw.to_vector(m), pf.to_vector(m))

def test_vector_length_is_free_param_count():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.a, x.b))
    assert p.to_vector(m).shape == (3,)                         # 2 (a) + 1 (b)

def test_empty_free_set():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: ())
    assert p.to_vector(m).shape == (0,)
    m2 = p.from_vector(m, p.to_vector(m))
    assert jnp.array_equal(m2.a, m.a) and jnp.array_equal(m2.b, m.b)

def test_grad_through_from_vector():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.a, x.b))
    v0 = p.to_vector(m)
    def loss(v): mm = p.from_vector(m, v); return jnp.sum(mm.a**2) + mm.b**2
    g = jax.grad(loss)(v0)
    h = 1e-6
    fd = jnp.array([(loss(v0.at[i].add(h)) - loss(v0.at[i].add(-h)))/(2*h) for i in range(v0.size)])
    assert jnp.allclose(g, fd, rtol=1e-5, atol=1e-6)

def test_jit_and_vmap():
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.a, x.b))
    f = jax.jit(lambda v: p.from_vector(m, v).b)
    assert jnp.allclose(f(p.to_vector(m)), m.b)
    batch = jnp.stack([p.to_vector(m), p.to_vector(m)+1.0])
    out = jax.vmap(lambda v: p.from_vector(m, v).b)(batch)
    assert out.shape == (2,)
