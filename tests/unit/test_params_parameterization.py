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


def test_all_free_with_static_field():
    """All array leaves free, static field present and untouched (design edge case)."""
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: (x.a, x.b))
    assert p.to_vector(m).shape == (3,)
    m2 = p.from_vector(m, p.to_vector(m) + 1.0)
    assert jnp.allclose(m2.a, m.a + 1.0) and jnp.allclose(m2.b, m.b + 1.0)
    assert m2.name == m.name                              # static field preserved


class Nested(eqx.Module):
    inner: Model
    c: jax.Array

def _nested():
    return Nested(inner=Model(a=jnp.array([1.0, 2.0]), b=jnp.array(3.0)), c=jnp.array([4.0, 5.0]))

def test_nested_module_roundtrip_and_partial_free():
    """Free a leaf inside a nested eqx.Module + a top-level leaf; fixed leaf untouched."""
    m = _nested()
    p = Parameterization.from_where(m, where=lambda x: (x.inner.a, x.c))   # inner.b stays fixed
    assert p.to_vector(m).shape == (4,)                                    # 2 (inner.a) + 2 (c)
    m2 = p.from_vector(m, p.to_vector(m) + 10.0)
    assert jnp.allclose(m2.inner.a, m.inner.a + 10.0)
    assert jnp.allclose(m2.c, m.c + 10.0)
    assert jnp.array_equal(m2.inner.b, m.inner.b)                          # fixed nested leaf untouched

def test_from_filter_with_hand_written_spec():
    """from_filter works on an externally-authored bool spec (not built via tree_at)."""
    m = _m()
    # hand-write the free_spec bool PyTree directly: a free, b fixed
    spec = Model(a=True, b=False, name="m")
    p = Parameterization.from_filter(m, spec)
    assert p.to_vector(m).shape == (2,)                                    # only a (shape (2,))
    m2 = p.from_vector(m, p.to_vector(m) + 7.0)
    assert jnp.allclose(m2.a, m.a + 7.0)
    assert jnp.array_equal(m2.b, m.b)
