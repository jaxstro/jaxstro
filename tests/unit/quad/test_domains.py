import inspect

import jax
import jax.numpy as jnp

from jaxstro import quad


def test_interval_endpoints_and_breakpoints_are_dynamic_pytree_leaves() -> None:
    domain = quad.Interval(0.0, 2.0, breakpoints=(1.5, 0.5))
    leaves, treedef = jax.tree.flatten(domain)
    assert len(leaves) == 4
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert rebuilt.breakpoints == domain.breakpoints


def test_breakpoint_count_is_static_structure() -> None:
    two = jax.tree.structure(quad.Interval(0.0, 1.0, breakpoints=(0.2, 0.8)))
    one = jax.tree.structure(quad.Interval(0.0, 1.0, breakpoints=(0.5,)))
    assert two != one


def test_breakpoints_are_keyword_only() -> None:
    parameter = inspect.signature(quad.Interval).parameters["breakpoints"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_breakpoints_sort_in_oriented_domain_order() -> None:
    forward = quad.Interval(0.0, 2.0, breakpoints=(1.5, 0.5))
    reverse = quad.Interval(2.0, 0.0, breakpoints=(0.5, 1.5))
    assert jnp.array_equal(quad.sorted_breakpoints(forward), jnp.array([0.5, 1.5]))
    assert jnp.array_equal(quad.sorted_breakpoints(reverse), jnp.array([1.5, 0.5]))


def test_duplicate_breakpoints_make_interval_invalid() -> None:
    domain = quad.Interval(0.0, 1.0, breakpoints=(0.5, 0.5))
    assert not bool(quad.interval_is_valid(domain))


def test_affine_map_preserves_orientation_separately_from_jacobian() -> None:
    reference = jnp.array([-1.0, 0.0, 1.0])
    forward = quad.map_interval(quad.Interval(2.0, 4.0), reference)
    reverse = quad.map_interval(quad.Interval(4.0, 2.0), reference)
    assert jnp.array_equal(forward.x, jnp.array([2.0, 3.0, 4.0]))
    assert jnp.array_equal(reverse.x, forward.x)
    assert forward.jacobian == 1.0
    assert reverse.jacobian == 1.0
    assert forward.orientation == 1.0
    assert reverse.orientation == -1.0


def test_zero_width_is_exact_zero_orientation() -> None:
    mapped = quad.map_interval(
        quad.Interval(3.0, 3.0),
        jnp.array([-0.5, 0.5]),
    )
    assert mapped.orientation == 0.0
    assert mapped.jacobian == 0.0
    assert jnp.all(mapped.x == 3.0)


def test_dynamic_endpoints_work_under_jit() -> None:
    evaluate = jax.jit(
        lambda lower, upper: (
            quad.map_interval(
                quad.Interval(lower, upper),
                jnp.array([-1.0, 0.0, 1.0]),
            ).x
        )
    )
    assert jnp.array_equal(evaluate(1.0, 5.0), jnp.array([1.0, 3.0, 5.0]))
