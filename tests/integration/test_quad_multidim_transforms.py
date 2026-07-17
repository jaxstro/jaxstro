import jax
import jax.numpy as jnp

from jaxstro import quad
from jaxstro.quad._multidim import evaluate_multidim, map_hyperrectangle


def test_map_composes_with_jit_and_vmap_over_bounds():
    reference = jnp.array([[0.25, 0.75]])

    @jax.jit
    def mapped(lower, upper):
        return map_hyperrectangle(
            quad.Hyperrectangle(lower, upper),
            reference,
        ).x

    lowers = jnp.array([[0.0, 0.0], [1.0, -1.0]])
    uppers = jnp.array([[1.0, 2.0], [3.0, 1.0]])
    values = jax.vmap(mapped)(lowers, uppers)
    assert values.shape == (2, 1, 2)


def test_traced_nonfinite_bounds_fail_dynamically():
    valid = jax.jit(
        lambda upper: quad.hyperrectangle_is_valid(
            quad.Hyperrectangle(jnp.zeros(2), upper)
        )
    )
    assert not valid(jnp.array([1.0, jnp.inf]))


def test_traced_invalid_domain_propagates_through_point_evaluation():
    reference = jnp.array([[0.25, 0.5], [0.75, 0.5]])

    @jax.jit
    def evaluated(upper):
        return evaluate_multidim(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), upper),
            reference,
            args=(),
            measure=quad.LebesgueMeasure(),
        )

    result = evaluated(jnp.array([1.0, jnp.inf]))
    assert not result.valid
    assert result.nonfinite


def test_evaluator_composes_eager_jit_and_vmap_over_bounds():
    reference = jnp.array([[0.25, 0.5], [0.75, 1.0]])

    def evaluated(lower, upper, scale):
        return evaluate_multidim(
            lambda x, factor: factor * jnp.sum(x, axis=-1),
            quad.Hyperrectangle(lower, upper),
            reference,
            args=scale,
            measure=quad.LebesgueMeasure(),
        )

    eager = evaluated(jnp.zeros(2), jnp.ones(2), jnp.asarray(2.0))
    compiled = jax.jit(evaluated)(
        jnp.zeros(2),
        jnp.ones(2),
        jnp.asarray(2.0),
    )
    for eager_leaf, compiled_leaf in zip(
        jax.tree.leaves(eager),
        jax.tree.leaves(compiled),
        strict=True,
    ):
        assert jnp.array_equal(eager_leaf, compiled_leaf)

    batched = jax.vmap(jax.jit(evaluated))(
        jnp.array([[0.0, 0.0], [1.0, -1.0]]),
        jnp.array([[1.0, 2.0], [3.0, 1.0]]),
        jnp.array([2.0, 0.5]),
    )
    assert batched.values.shape == (2, 2)
    assert batched.weights.shape == (2, 2)
    assert batched.nonfinite.shape == (2,)
    assert batched.valid.shape == (2,)
    assert jnp.array_equal(batched.valid, jnp.array([True, True]))
