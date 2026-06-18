import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from jaxstro.params import Parameterization
from jaxstro.params.transforms import Exp, Sigmoid, Softplus


class M(eqx.Module):
    r_h: jax.Array
    Q: jax.Array


def _m():
    return M(r_h=jnp.array(1.3), Q=jnp.array(0.4))


def test_transformed_roundtrip_and_bounds():
    m = _m()
    p = Parameterization.from_where(
        m, where=lambda x: (x.r_h, x.Q), transforms=(Exp(), Sigmoid(0.0, 1.0))
    )
    m2 = p.from_vector(m, p.to_vector(m))
    assert jnp.allclose(m2.r_h, m.r_h) and jnp.allclose(m2.Q, m.Q)
    # vector lives in unconstrained R; perturb and confirm bounds hold after forward
    m3 = p.from_vector(m, p.to_vector(m) + jnp.array([5.0, -8.0]))
    assert m3.r_h > 0.0 and 0.0 < m3.Q < 1.0


def test_log_det_jacobian_sums_per_leaf():
    m = _m()
    p = Parameterization.from_where(
        m, where=lambda x: (x.r_h, x.Q), transforms=(Exp(), Sigmoid(0.0, 1.0))
    )
    v = p.to_vector(m)
    expected = Exp().forward_log_det_jacobian(v[0]) + Sigmoid(
        0.0, 1.0
    ).forward_log_det_jacobian(v[1])
    assert jnp.allclose(p.log_det_jacobian(v), expected, rtol=1e-8)


def test_end_to_end_grad_through_transformed():
    m = _m()
    p = Parameterization.from_where(
        m, where=lambda x: (x.r_h, x.Q), transforms=(Exp(), Sigmoid(0.0, 1.0))
    )
    v0 = p.to_vector(m)

    def loss(v):
        mm = p.from_vector(m, v)
        return mm.r_h**2 + mm.Q**2

    g = jax.grad(loss)(v0)
    assert jnp.all(jnp.isfinite(g))


def test_transform_follows_leaf_not_tuple_order():
    """REGRESSION (T1 review [Important]): a transform attached to a leaf via `where`
    must apply to THAT leaf regardless of where-tuple order or field-declaration order.
    Here r_h is declared first but selected SECOND, with Exp; Q selected first with Sigmoid.
    Exp must still govern r_h (so a large unconstrained value -> large positive r_h) and
    Sigmoid must still bound Q in (0,1)."""
    m = _m()
    # where-tuple order (Q, r_h) is the REVERSE of declaration order (r_h, Q)
    p = Parameterization.from_where(
        m, where=lambda x: (x.Q, x.r_h), transforms=(Sigmoid(0.0, 1.0), Exp())
    )
    m2 = p.from_vector(m, p.to_vector(m))
    assert jnp.allclose(m2.r_h, m.r_h) and jnp.allclose(m2.Q, m.Q)  # round-trip exact
    # Build the unconstrained vector in PyTree-leaf order and push both leaves up hard:
    big = p.to_vector(m) + 8.0
    m3 = p.from_vector(m, big)
    assert m3.r_h > m.r_h  # Exp governs r_h -> grows
    assert 0.0 < m3.Q < 1.0  # Sigmoid still bounds Q despite +8.0 push


# --- Task-3 review regression tests -----------------------------------------


class Multi(eqx.Module):
    a: jax.Array  # (2,) free   -> Exp
    s: jax.Array  # scalar free -> Sigmoid
    v: jax.Array  # (3,) free   -> Softplus


def _multi():
    return Multi(
        a=jnp.array([1.2, 0.7]),
        s=jnp.array(0.3),
        v=jnp.array([2.0, 0.5, 1.1]),
    )


def test_log_det_matches_AD_slogdet_multileaf():
    """log_det_jacobian must equal the AD log|det J| of the full free->physical
    map across MIXED-shape leaves with THREE DIFFERENT bijectors selected in an
    order DIFFERENT from declaration order (a, s, v)."""
    m = _multi()
    # Declaration order is (a, s, v); select in scrambled order (s, v, a).
    p = Parameterization.from_where(
        m,
        where=lambda x: (x.s, x.v, x.a),
        transforms=(Sigmoid(0.0, 1.0), Softplus(), Exp()),
    )

    def free_ravel_forward(v):
        mm = p.from_vector(m, v)
        free = eqx.filter(mm, p.free_spec)
        return ravel_pytree(free)[0]

    v = p.to_vector(m) + jnp.array([0.4, -0.6, 0.2, 0.9, -0.3, 0.15])
    J = jax.jacfwd(free_ravel_forward)(v)
    expected = jnp.linalg.slogdet(J)[1]
    assert jnp.allclose(p.log_det_jacobian(v), expected, rtol=1e-8)


class Inner(eqx.Module):
    s: jax.Array
    v: jax.Array


class Outer(eqx.Module):
    a: jax.Array
    inner: Inner
    b: jax.Array


def test_alignment_nested_module_scrambled_where():
    """Nested module, `where` selecting (inner.v, a, inner.s) -- scrambled vs
    declaration order -- with transforms (Exp, Sigmoid, Softplus). At the
    unconstrained zero vector each free leaf must take ITS bijector's
    forward(0); the fixed leaf b is untouched."""
    m = Outer(
        a=jnp.array(0.9),
        inner=Inner(s=jnp.array(-0.4), v=jnp.array([1.0, 2.0, 3.0])),
        b=jnp.array(7.0),
    )
    p = Parameterization.from_where(
        m,
        where=lambda x: (x.inner.v, x.a, x.inner.s),
        transforms=(Exp(), Sigmoid(0.0, 1.0), Softplus()),
    )
    v0 = jnp.zeros_like(p.to_vector(m))
    m2 = p.from_vector(m, v0)
    # a <- Sigmoid(0,1).forward(0) = 0.5
    assert jnp.allclose(m2.a, jnp.array(0.5))
    # inner.s <- Softplus().forward(0) = log 2 ~ 0.6931
    assert jnp.allclose(m2.inner.s, jnp.log(jnp.array(2.0)))
    # inner.v <- Exp().forward(0) = 1.0 elementwise
    assert jnp.allclose(m2.inner.v, jnp.ones(3))
    # fixed b untouched
    assert jnp.allclose(m2.b, m.b)


def test_grad_through_log_det():
    """jax.grad(log_det_jacobian) must be finite and match a central
    finite-difference (numpyro relies on this gradient)."""
    m = _multi()
    p = Parameterization.from_where(
        m,
        where=lambda x: (x.s, x.v, x.a),
        transforms=(Sigmoid(0.0, 1.0), Softplus(), Exp()),
    )
    v = p.to_vector(m) + jnp.array([0.4, -0.6, 0.2, 0.9, -0.3, 0.15])
    g = jax.grad(p.log_det_jacobian)(v)
    assert jnp.all(jnp.isfinite(g))

    eps = 1e-5
    fd = jnp.array(
        [
            (
                p.log_det_jacobian(v.at[i].add(eps))
                - p.log_det_jacobian(v.at[i].add(-eps))
            )
            / (2 * eps)
            for i in range(v.shape[0])
        ]
    )
    assert jnp.allclose(g, fd, rtol=1e-5, atol=1e-6)


class FFF(eqx.Module):
    x: jax.Array  # free
    y: jax.Array  # fixed
    z: jax.Array  # free


def test_from_filter_transforms_pytree_order_and_errors():
    """from_filter: transforms align by PyTree (leaf) order, NOT selection
    order; and too-few / too-many transforms both raise a clean ValueError."""
    m = FFF(x=jnp.array(0.5), y=jnp.array(9.0), z=jnp.array(1.5))
    # Hand-built bool PyTree: x free, y fixed, z free.
    free_spec = FFF(x=True, y=False, z=True)
    # Two DIFFERENT bijectors, in PyTree order (x -> Exp, z -> Sigmoid).
    p = Parameterization.from_filter(
        m, free_spec, transforms=(Exp(), Sigmoid(0.0, 1.0))
    )
    v0 = jnp.zeros_like(p.to_vector(m))
    m2 = p.from_vector(m, v0)
    # x <- Exp().forward(0) = 1.0 ; z <- Sigmoid(0,1).forward(0) = 0.5
    assert jnp.allclose(m2.x, jnp.array(1.0))
    assert jnp.allclose(m2.z, jnp.array(0.5))
    assert jnp.allclose(m2.y, m.y)  # fixed untouched

    # Too-few transforms (1 for 2 free leaves) -> clean ValueError.
    with pytest.raises(ValueError):
        Parameterization.from_filter(m, free_spec, transforms=(Exp(),))
    # Too-many transforms (3 for 2 free leaves) -> clean ValueError.
    with pytest.raises(ValueError):
        Parameterization.from_filter(
            m, free_spec, transforms=(Exp(), Sigmoid(0.0, 1.0), Softplus())
        )


def test_from_where_single_leaf_with_transform():
    """Single-leaf where + a transform: replace-arity must match (transforms[0])."""
    m = _m()
    p = Parameterization.from_where(m, where=lambda x: x.r_h, transforms=(Exp(),))
    m2 = p.from_vector(m, p.to_vector(m))
    assert jnp.allclose(m2.r_h, m.r_h)  # round-trip
    m3 = p.from_vector(m, p.to_vector(m) + 5.0)
    assert m3.r_h > 0.0  # Exp keeps it positive
    assert jnp.allclose(m3.Q, m.Q)  # Q fixed
