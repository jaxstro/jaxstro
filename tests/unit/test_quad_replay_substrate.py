"""Private replay-evidence and mapping substrate contracts."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.quad import Interval
from jaxstro.quad._adaptive import (
    LocalEstimate,
    adaptive_controller,
    reference_partition,
    select_segment,
)
from jaxstro.quad._romberg import (
    romberg_refine,
    romberg_replay_value,
    romberg_tanh_sinh_refine,
    romberg_tanh_sinh_replay_value,
)
from jaxstro.quad.tolerance import MaxNorm
from jaxstro.quad.transforms import map_interval_replay


def test_reference_partition_is_local_to_each_original_segment() -> None:
    domain = Interval(0.0, 4.0, breakpoints=(1.0, 3.0))
    partition = reference_partition(domain)

    assert jnp.array_equal(partition.lower, -jnp.ones(3))
    assert jnp.array_equal(partition.upper, jnp.ones(3))
    assert jnp.array_equal(
        partition.segment_id,
        jnp.arange(3, dtype=jnp.int32),
    )
    assert [
        (
            float(select_segment(domain, index).lower),
            float(select_segment(domain, index).upper),
        )
        for index in range(3)
    ] == [(0.0, 1.0), (1.0, 3.0), (3.0, 4.0)]


def test_reversed_partition_preserves_integration_orientation() -> None:
    domain = Interval(4.0, 0.0, breakpoints=(1.0, 3.0))

    assert [
        (
            float(select_segment(domain, index).lower),
            float(select_segment(domain, index).upper),
        )
        for index in range(3)
    ] == [(4.0, 3.0), (3.0, 1.0), (1.0, 0.0)]


def test_controller_propagates_parent_segment_identity() -> None:
    partition = reference_partition(Interval(0.0, 2.0, breakpoints=(1.0,)))

    def estimate(lower, upper, _segment_id):
        width = upper - lower
        return LocalEstimate(
            value=width,
            error=jnp.asarray(1.0),
            nonfinite=jnp.asarray(False),
        )

    result = adaptive_controller(
        partition,
        estimate,
        node_cost=1,
        max_evaluations=4,
        max_regions=4,
        epsabs=0.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    active_ids = result.region_segment_id[result.region_active]
    assert jnp.all((active_ids == 0) | (active_ids == 1))
    assert jnp.sum(active_ids == 0) + jnp.sum(active_ids == 1) == result.active_regions


def test_signed_replay_map_keeps_reversed_orientation_without_sign() -> None:
    mapped = map_interval_replay(
        Interval(2.0, -1.0),
        jnp.array([-1.0, 0.0, 1.0]),
    )
    assert jnp.allclose(mapped.x, jnp.array([2.0, 0.5, -1.0]))
    assert mapped.jacobian == -1.5
    assert mapped.orientation == 1.0


def test_signed_replay_map_has_leibniz_tangent_at_coincident_bounds() -> None:
    nodes = jnp.array([-0.5, 0.5])
    weights = jnp.ones(2)

    def fixed_formula(lower, upper):
        mapped = map_interval_replay(Interval(lower, upper), nodes)
        return jnp.sum(weights * mapped.x**0 * mapped.jacobian)

    _, lower_tangent = jax.jvp(fixed_formula, (1.0, 1.0), (1.0, 0.0))
    _, upper_tangent = jax.jvp(fixed_formula, (1.0, 1.0), (0.0, 1.0))
    assert lower_tangent == -1.0
    assert upper_tangent == 1.0


@pytest.mark.parametrize("initial_level", [1, 2, 3])
def test_romberg_replay_reconstructs_accepted_diagonal(initial_level) -> None:
    def evaluate(reference):
        return jnp.exp(reference), jnp.asarray(False), jnp.asarray(False)

    primal = romberg_refine(
        evaluate,
        jnp.asarray(0.0),
        initial_level=initial_level,
        max_evaluations=257,
        max_regions=1,
        epsabs=1e-12,
        epsrel=1e-12,
        error_norm=MaxNorm(),
        dtype=jnp.float64,
    )
    replay = romberg_replay_value(
        evaluate,
        jnp.asarray(0.0),
        initial_level=initial_level,
        accepted_level=primal.levels - 1,
        max_evaluations=257,
        dtype=jnp.float64,
    )
    assert jnp.allclose(replay, primal.value, rtol=2e-15, atol=2e-15)


def test_global_tanh_sinh_replay_reconstructs_accepted_weight_row() -> None:
    def evaluate(reference):
        return jnp.exp(-(reference**2)), jnp.asarray(False), jnp.asarray(False)

    primal = romberg_tanh_sinh_refine(
        evaluate,
        jnp.asarray(0.0),
        initial_level=2,
        max_evaluations=801,
        max_regions=1,
        epsabs=1e-11,
        epsrel=1e-11,
        error_norm=MaxNorm(),
        dtype=jnp.float64,
    )
    replay = romberg_tanh_sinh_replay_value(
        evaluate,
        jnp.asarray(0.0),
        initial_level=2,
        accepted_level=primal.levels - 1,
        max_evaluations=801,
        dtype=jnp.float64,
    )
    assert jnp.allclose(replay, primal.value, rtol=2e-15, atol=2e-15)
