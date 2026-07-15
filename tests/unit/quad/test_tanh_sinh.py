"""Fixed tanh-sinh formulas and Phase A domain-map contracts."""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

from jaxstro.quad import Infinite, Interval, LeftInfinite, RightInfinite, TanhSinhRule
from jaxstro.quad._tanh_sinh import _tanh_sinh_lattice_data, tanh_sinh_rule_data
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
    lattice = _tanh_sinh_lattice_data(5, dtype=jnp.float64)
    assert coarse.nested is True
    assert jnp.allclose(coarse.nodes, -coarse.nodes[::-1])
    assert jnp.allclose(coarse.weights, coarse.weights[::-1])
    assert jnp.array_equal(fine.nodes[lattice.coarse_to_fine], coarse.nodes)


@pytest.mark.parametrize(
    ("dtype", "expected_counts"),
    [
        (jnp.float32, (5, 11, 21, 41, 79, 155, 305, 599)),
        (jnp.float64, (7, 13, 25, 51, 101, 203, 407, 809)),
    ],
)
def test_tanh_sinh_active_lattice_is_representable_and_nested(
    dtype, expected_counts
) -> None:
    previous = None
    for level, expected_count in enumerate(expected_counts):
        lattice = _tanh_sinh_lattice_data(level, dtype=dtype)
        active_nodes = lattice.nodes[lattice.active]
        active_weights = lattice.weights[lattice.active]
        assert active_nodes.shape == (expected_count,)
        assert jnp.all(jnp.isfinite(active_nodes))
        assert jnp.all(jnp.abs(active_nodes) < 1.0)
        assert jnp.all(jnp.diff(active_nodes) > 0.0)
        assert jnp.all(jnp.isfinite(active_weights))
        assert jnp.all(active_weights > 0.0)
        assert jnp.all(jnp.isfinite(lattice.nodes))
        assert jnp.all(lattice.nodes[~lattice.active] == 0.25)
        assert jnp.all(lattice.weights[~lattice.active] == 0.0)
        if previous is not None:
            assert jnp.array_equal(
                lattice.compact_nodes[lattice.coarse_to_fine],
                previous.compact_nodes,
            )
        previous = lattice


@pytest.mark.parametrize(
    ("dtype", "level", "expected_count", "expected_terminal", "exhausted"),
    [
        (jnp.float32, 0, 5, 2, False),
        (jnp.float32, 1, 11, 5, True),
        (jnp.float32, 4, 79, 40, True),
        (jnp.float64, 5, 203, 101, False),
        (jnp.float64, 6, 407, 203, True),
        (jnp.float64, 7, 809, 406, True),
    ],
)
def test_tanh_sinh_boundary_fixtures(
    dtype, level, expected_count, expected_terminal, exhausted
) -> None:
    lattice = _tanh_sinh_lattice_data(level, dtype=dtype)
    assert lattice.compact_nodes.shape == (expected_count,)
    assert lattice.terminal_index == expected_terminal
    assert lattice.dtype_exhausted == exhausted


@pytest.mark.parametrize(
    ("dtype", "coarse_level", "largest_new_odd"),
    [
        (jnp.float32, 3, 37),
        (jnp.float64, 6, 401),
    ],
)
def test_tanh_sinh_collision_transitions_keep_mandatory_coarse_nodes(
    dtype, coarse_level, largest_new_odd
) -> None:
    coarse = _tanh_sinh_lattice_data(coarse_level, dtype=dtype)
    fine = _tanh_sinh_lattice_data(coarse_level + 1, dtype=dtype)
    coarse_indices = {int(index) for index in coarse.compact_indices}
    fine_indices = {int(index) for index in fine.compact_indices}
    mapped_coarse = {2 * index for index in coarse_indices}
    new_indices = fine_indices - mapped_coarse
    expected_new = {
        index for index in range(-largest_new_odd, largest_new_odd + 1, 2) if index != 0
    }
    outer_shell = {
        index
        for index in new_indices
        if 2 * int(coarse.terminal_index) < abs(index) < int(fine.terminal_index)
    }
    assert mapped_coarse <= fine_indices
    assert new_indices == expected_new
    assert outer_shell == set()


def test_tanh_sinh_inactive_padded_values_are_neutralized_before_reduction() -> None:
    lattice = _tanh_sinh_lattice_data(5, dtype=jnp.float64)
    values = jnp.where(lattice.nodes == 0.25, jnp.inf, jnp.exp(lattice.nodes))
    safe_values = jnp.where(lattice.active, values, 0.0)
    assert jnp.all(jnp.isfinite(safe_values))
    assert jnp.isfinite(jnp.sum(lattice.weights * safe_values))


def test_tanh_sinh_terminal_evidence_does_not_halve_with_unchanged_extent() -> None:
    level_one = _tanh_sinh_lattice_data(1, dtype=jnp.float64)
    level_two = _tanh_sinh_lattice_data(2, dtype=jnp.float64)
    level_three = _tanh_sinh_lattice_data(3, dtype=jnp.float64)

    assert level_one.terminal_parameter == level_two.terminal_parameter
    assert level_three.terminal_parameter > level_two.terminal_parameter

    alpha = 0.9

    def terminal_density(lattice):
        node = lattice.terminal_node
        return lattice.terminal_density_weight * (1.0 - node) ** (-alpha)

    assert jnp.array_equal(terminal_density(level_one), terminal_density(level_two))
    assert terminal_density(level_three) < terminal_density(level_two)


@pytest.mark.parametrize("alpha", [0.5, 0.9, 0.99])
def test_tanh_sinh_terminal_evidence_exposes_unresolved_endpoint_mass(alpha) -> None:
    coarse = _tanh_sinh_lattice_data(6, dtype=jnp.float64)
    fine = _tanh_sinh_lattice_data(7, dtype=jnp.float64)

    def estimate(lattice):
        values = (1.0 - lattice.compact_nodes) ** (-alpha)
        return jnp.sum(lattice.compact_weights * values)

    exact = 2.0 ** (1.0 - alpha) / (1.0 - alpha)
    observed_error = jnp.abs(exact - estimate(fine))
    adjacent_difference = jnp.abs(estimate(fine) - estimate(coarse))
    terminal_evidence = fine.terminal_density_weight * (
        (1.0 - fine.terminal_node) ** (-alpha) + (1.0 + fine.terminal_node) ** (-alpha)
    )
    assert jnp.isfinite(terminal_evidence)
    assert terminal_evidence > 0.0
    if alpha >= 0.9:
        assert observed_error > adjacent_difference


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


def test_public_tanh_sinh_uses_float32_precision_policy_in_subprocess() -> None:
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    program = """
import jax
import jax.numpy as jnp
from jaxstro.quad import Infinite, TanhSinhRule, fixed
from jaxstro.quad._tanh_sinh import tanh_sinh_rule_data

data = tanh_sinh_rule_data(TanhSinhRule(5))
assert data.nodes.dtype == jnp.float32
assert data.nodes.shape == (155,)
evaluate = jax.jit(lambda: fixed(lambda x: jnp.exp(-(x**2)), Infinite(), rule=TanhSinhRule(5)))
assert jnp.isfinite(evaluate())
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
