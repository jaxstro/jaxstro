import inspect

import jax
import jax.numpy as jnp
import pytest

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


def test_improper_scale_is_keyword_only_and_preserves_default_pytree_layout() -> None:
    assert (
        inspect.signature(quad.RightInfinite).parameters["scale"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(quad.LeftInfinite).parameters["scale"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert (
        inspect.signature(quad.Infinite).parameters["scale"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )

    assert jax.tree.flatten(quad.RightInfinite(0.0))[0] == [0.0]
    assert jax.tree.flatten(quad.LeftInfinite(0.0))[0] == [0.0]
    assert jax.tree.flatten(quad.Infinite())[0] == []


def test_explicit_improper_scale_is_a_dynamic_pytree_leaf() -> None:
    domains = (
        quad.RightInfinite(0.0, scale=2.0),
        quad.LeftInfinite(0.0, scale=2.0),
        quad.Infinite(scale=2.0),
    )
    for domain in domains:
        leaves, treedef = jax.tree.flatten(domain)
        assert leaves[-1] == 2.0
        assert jax.tree.unflatten(treedef, leaves) == domain


def test_explicit_scale_rescales_every_improper_map() -> None:
    reference = jnp.asarray([0.0])
    right = quad.map_domain(quad.RightInfinite(2.0, scale=3.0), reference)
    left = quad.map_domain(quad.LeftInfinite(2.0, scale=3.0), reference)
    full = quad.map_domain(quad.Infinite(scale=3.0), reference)

    assert right.x[0] == 5.0
    assert right.jacobian[0] == 6.0
    assert left.x[0] == -1.0
    assert left.jacobian[0] == 6.0
    assert full.x[0] == 0.0
    assert full.jacobian[0] == 3.0


def test_omitted_improper_scale_matches_explicit_legacy_unit_scale() -> None:
    reference = jnp.asarray([-0.5, 0.0, 0.5])
    pairs = (
        (quad.RightInfinite(2.0), quad.RightInfinite(2.0, scale=1.0)),
        (quad.LeftInfinite(2.0), quad.LeftInfinite(2.0, scale=1.0)),
        (quad.Infinite(), quad.Infinite(scale=1.0)),
    )
    for legacy, explicit in pairs:
        legacy_map = quad.map_domain(legacy, reference)
        explicit_map = quad.map_domain(explicit, reference)
        assert jnp.array_equal(legacy_map.x, explicit_map.x)
        assert jnp.array_equal(legacy_map.jacobian, explicit_map.jacobian)
        assert legacy_map.valid == explicit_map.valid


def test_invalid_improper_scales_fail_closed_under_jit() -> None:
    validity = jax.jit(
        lambda scale: (
            quad.map_domain(
                quad.Infinite(scale=scale),
                jnp.asarray([0.0]),
            ).valid
        )
    )
    assert validity(1.0)
    assert not validity(0.0)
    assert not validity(-1.0)
    assert not validity(jnp.inf)
    assert not validity(jnp.nan)


def test_array_valued_improper_scale_fails_with_scalar_contract() -> None:
    with pytest.raises(ValueError, match="improper-domain scale must be scalar"):
        quad.map_domain(
            quad.Infinite(scale=jnp.asarray([1.0, 2.0])),
            jnp.asarray([0.0]),
        )


def test_complex_improper_scale_fails_with_real_contract() -> None:
    with pytest.raises(TypeError, match="improper-domain scale must be real"):
        quad.map_domain(
            quad.Infinite(scale=1.0 + 1.0j),
            jnp.asarray([0.0]),
        )


def test_boolean_improper_scale_fails_with_real_contract() -> None:
    with pytest.raises(TypeError, match="improper-domain scale must be real"):
        quad.map_domain(
            quad.Infinite(scale=True),
            jnp.asarray([0.0]),
        )


def test_improper_scale_is_stopped_algorithmic_provenance() -> None:
    derivative = jax.grad(
        lambda scale: quad.map_domain(
            quad.RightInfinite(0.0, scale=scale),
            jnp.asarray([0.0]),
        ).x[0]
    )(2.0)
    assert derivative == 0.0
