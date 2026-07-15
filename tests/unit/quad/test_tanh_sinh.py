"""Fixed tanh-sinh formulas and Phase A domain-map contracts."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.quad import Infinite, Interval, LeftInfinite, RightInfinite, TanhSinhRule
from jaxstro.quad._tanh_sinh import tanh_sinh_rule_data
from jaxstro.quad.transforms import map_domain


@pytest.mark.parametrize(
    ("domain", "fun", "expected", "tolerance"),
    [
        (
            Interval(-1.0, 1.0),
            lambda x: 1.0 / jnp.sqrt(1.0 - x * x),
            jnp.pi,
            2e-7,
        ),
        (RightInfinite(0.0), lambda x: jnp.exp(-x), 1.0, 2e-9),
        (LeftInfinite(0.0), lambda x: jnp.exp(x), 1.0, 2e-9),
        (Infinite(), lambda x: jnp.exp(-(x**2)), jnp.sqrt(jnp.pi), 2e-9),
    ],
)
def test_tanh_sinh_formula_and_domain_maps(domain, fun, expected, tolerance) -> None:
    data = tanh_sinh_rule_data(TanhSinhRule(7))
    mapped = map_domain(domain, data.nodes)
    got = mapped.orientation * jnp.sum(data.weights * mapped.jacobian * fun(mapped.x))
    assert jnp.all(mapped.valid)
    assert jnp.allclose(got, expected, rtol=tolerance, atol=tolerance)


def test_tanh_sinh_reference_rule_is_symmetric_and_nested() -> None:
    coarse = tanh_sinh_rule_data(TanhSinhRule(4))
    fine = tanh_sinh_rule_data(TanhSinhRule(5))
    assert coarse.nested is True
    assert jnp.allclose(coarse.nodes, -coarse.nodes[::-1])
    assert jnp.allclose(coarse.weights, coarse.weights[::-1])
    assert jnp.allclose(coarse.nodes, fine.nodes[::2])


def test_interval_map_preserves_orientation_separately() -> None:
    reference = jnp.asarray([-0.5, 0.5])
    forward = map_domain(Interval(2.0, 6.0), reference)
    reverse = map_domain(Interval(6.0, 2.0), reference)
    assert jnp.array_equal(forward.x, reverse.x)
    assert jnp.array_equal(forward.jacobian, reverse.jacobian)
    assert forward.orientation == 1.0
    assert reverse.orientation == -1.0


def test_tanh_sinh_construction_and_map_compile() -> None:
    evaluate = jax.jit(
        lambda lower: (
            map_domain(
                RightInfinite(lower),
                tanh_sinh_rule_data(TanhSinhRule(3)).nodes,
            ).x
        )
    )
    assert jnp.all(jnp.isfinite(evaluate(jnp.asarray(0.0))))
