"""Fixed-capacity h-adaptive controller state and status contracts."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.quad import L1Norm, MaxNorm, QuadStatus
from jaxstro.quad._adaptive import (
    LocalEstimate,
    ReferencePartition,
    _stagnation_hits,
    _value_stagnation_scale,
    adaptive_controller,
)


def _partition(lower=(-1.0,), upper=(1.0,), *, valid=True):
    return ReferencePartition(
        lower=jnp.asarray(lower),
        upper=jnp.asarray(upper),
        segment_id=jnp.arange(len(lower), dtype=jnp.int32),
        valid=jnp.asarray(valid),
    )


def _quadratic_error_estimator(lower, upper, _segment_id):
    width = upper - lower
    return LocalEstimate(
        value=width,
        error=width**2,
        nonfinite=jnp.asarray(False),
    )


def test_controller_initial_segments_and_immediate_convergence() -> None:
    result = adaptive_controller(
        _partition(lower=(-1.0, 0.0), upper=(0.0, 1.0)),
        _quadratic_error_estimator,
        node_cost=3,
        max_evaluations=30,
        max_regions=4,
        epsabs=3.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    assert result.status == QuadStatus.CONVERGED
    assert jnp.array_equal(result.value, 2.0)
    assert jnp.array_equal(result.error, 2.0)
    assert result.evaluations == 6
    assert result.refinements == 0
    assert result.active_regions == 2


def test_controller_uses_lowest_index_ties_and_exact_work_counts() -> None:
    result = adaptive_controller(
        _partition(),
        _quadratic_error_estimator,
        node_cost=2,
        max_evaluations=100,
        max_regions=3,
        epsabs=0.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    assert result.status == QuadStatus.MAX_REGIONS
    assert result.evaluations == 10
    assert result.refinements == 2
    assert result.active_regions == 3
    assert jnp.array_equal(result.region_lower, jnp.asarray([-1.0, 0.0, -0.5]))
    assert jnp.array_equal(result.region_upper, jnp.asarray([-0.5, 1.0, 0.0]))
    assert jnp.array_equal(result.region_active, jnp.asarray([True, True, True]))
    assert jnp.array_equal(result.value, 2.0)
    assert jnp.array_equal(result.error, 1.5)


def test_controller_stops_before_incomplete_evaluation_batch() -> None:
    result = adaptive_controller(
        _partition(),
        _quadratic_error_estimator,
        node_cost=3,
        max_evaluations=9,
        max_regions=8,
        epsabs=0.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    assert result.status == QuadStatus.MAX_EVALUATIONS
    assert result.evaluations == 9
    assert result.refinements == 1
    assert result.active_regions == 2


def test_controller_reduction_recovers_untouched_small_regions() -> None:
    def cancellation_estimator(lower, upper, _segment_id):
        width = upper - lower
        large_parent = (lower < 0.0) & (width == 1.0)
        small_region = (lower >= 0.0) & (width == 1.0)
        evidence = jnp.where(
            large_parent,
            jnp.asarray(1.0e8, dtype=jnp.float32),
            jnp.where(small_region, jnp.asarray(1.0, dtype=jnp.float32), 0.0),
        )
        return LocalEstimate(evidence, evidence, jnp.asarray(False))

    result = adaptive_controller(
        _partition(lower=(-1.0, 0.0), upper=(0.0, 1.0)),
        cancellation_estimator,
        node_cost=1,
        max_evaluations=20,
        max_regions=3,
        epsabs=0.5,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    assert result.status == QuadStatus.MAX_REGIONS
    assert jnp.array_equal(result.value, jnp.asarray(1.0, dtype=jnp.float32))
    assert jnp.array_equal(result.error, jnp.asarray(1.0, dtype=jnp.float32))


def test_controller_status_precedence_invalid_before_nonfinite() -> None:
    def nonfinite_estimator(lower, upper, _segment_id):
        width = upper - lower
        return LocalEstimate(width, width, jnp.asarray(True))

    invalid = adaptive_controller(
        _partition(valid=False),
        nonfinite_estimator,
        node_cost=1,
        max_evaluations=10,
        max_regions=4,
        epsabs=1.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    nonfinite = adaptive_controller(
        _partition(valid=True),
        nonfinite_estimator,
        node_cost=1,
        max_evaluations=10,
        max_regions=4,
        epsabs=1.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    assert invalid.status == QuadStatus.INVALID_INPUT
    assert nonfinite.status == QuadStatus.NONFINITE_INTEGRAND


@pytest.mark.parametrize(
    ("epsabs", "epsrel"),
    [(jnp.nan, 0.0), (-1.0, 0.0), (0.0, jnp.inf), (0.0, -1.0)],
)
def test_controller_invalid_dynamic_tolerances_have_input_precedence(
    epsabs, epsrel
) -> None:
    result = adaptive_controller(
        _partition(),
        _quadratic_error_estimator,
        node_cost=1,
        max_evaluations=10,
        max_regions=4,
        epsabs=epsabs,
        epsrel=epsrel,
        error_norm=MaxNorm(),
    )
    assert result.status == QuadStatus.INVALID_INPUT


def test_controller_rejects_norm_overflow_before_initial_convergence() -> None:
    maximum = jnp.finfo(jnp.float32).max

    def overflow_estimator(_lower, _upper, _segment_id):
        payload = jnp.asarray([maximum, maximum], dtype=jnp.float32)
        return LocalEstimate(payload, payload, jnp.asarray(False))

    result = adaptive_controller(
        _partition(),
        overflow_estimator,
        node_cost=1,
        max_evaluations=10,
        max_regions=4,
        epsabs=0.0,
        epsrel=1.0,
        error_norm=L1Norm(),
    )
    assert result.status == QuadStatus.NONFINITE_INTEGRAND


def test_controller_rejects_norm_overflow_after_split() -> None:
    maximum = jnp.finfo(jnp.float32).max

    def overflow_after_split(lower, upper, _segment_id):
        width = upper - lower
        child = jnp.asarray([0.3 * maximum, 0.3 * maximum], dtype=jnp.float32)
        initial = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
        error = jnp.where(width < 2.0, child, initial)
        return LocalEstimate(jnp.zeros((2,), dtype=jnp.float32), error, False)

    result = adaptive_controller(
        _partition(),
        overflow_after_split,
        node_cost=1,
        max_evaluations=10,
        max_regions=4,
        epsabs=0.0,
        epsrel=0.0,
        error_norm=L1Norm(),
    )
    assert result.status == QuadStatus.NONFINITE_INTEGRAND


def test_controller_rejects_mismatched_or_nonreal_error_payloads() -> None:
    def mismatched(lower, upper, _segment_id):
        width = upper - lower
        return LocalEstimate(jnp.stack((width, width)), width, jnp.asarray(False))

    def complex_error(lower, upper, _segment_id):
        width = upper - lower
        return LocalEstimate(width, width + 0j, jnp.asarray(False))

    common = dict(
        node_cost=1,
        max_evaluations=10,
        max_regions=4,
        epsabs=1.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    with pytest.raises(ValueError, match="match the value payload shape"):
        adaptive_controller(_partition(), mismatched, **common)
    with pytest.raises(TypeError, match="real floating dtype"):
        adaptive_controller(_partition(), complex_error, **common)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("node_cost", True),
        ("max_evaluations", 0),
        ("max_regions", 2.5),
    ],
)
def test_controller_rejects_invalid_static_capacities(name, value) -> None:
    arguments = dict(node_cost=1, max_evaluations=10, max_regions=4)
    arguments[name] = value
    with pytest.raises(ValueError, match=f"adaptive {name} must be"):
        adaptive_controller(
            _partition(),
            _quadratic_error_estimator,
            **arguments,
            epsabs=1.0,
            epsrel=0.0,
            error_norm=MaxNorm(),
        )


def test_controller_detects_unrepresentable_midpoint() -> None:
    endpoint = jnp.asarray(1.0)
    neighbor = jnp.nextafter(endpoint, jnp.asarray(0.0))
    result = adaptive_controller(
        _partition(lower=(neighbor,), upper=(endpoint,)),
        _quadratic_error_estimator,
        node_cost=1,
        max_evaluations=20,
        max_regions=4,
        epsabs=0.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    assert result.status == QuadStatus.ROUNDOFF_LIMITED
    assert result.refinements == 0


def test_controller_combined_exit_precedence_is_frozen() -> None:
    endpoint = jnp.asarray(1.0)
    neighbor = jnp.nextafter(endpoint, jnp.asarray(0.0))
    all_exhausted = adaptive_controller(
        _partition(lower=(neighbor,), upper=(endpoint,)),
        _quadratic_error_estimator,
        node_cost=1,
        max_evaluations=1,
        max_regions=1,
        epsabs=0.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    both_capacities = adaptive_controller(
        _partition(),
        _quadratic_error_estimator,
        node_cost=1,
        max_evaluations=1,
        max_regions=1,
        epsabs=0.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    assert all_exhausted.status == QuadStatus.ROUNDOFF_LIMITED
    assert both_capacities.status == QuadStatus.MAX_EVALUATIONS


def test_controller_payload_error_summation_and_error_norm_policy() -> None:
    def payload_estimator(lower, upper, _segment_id):
        width = upper - lower
        return LocalEstimate(
            value=jnp.stack((width, 2.0 * width)),
            error=jnp.stack((0.6 * width**2, 0.6 * width**2)),
            nonfinite=jnp.asarray(False),
        )

    max_result = adaptive_controller(
        _partition(),
        payload_estimator,
        node_cost=1,
        max_evaluations=1,
        max_regions=4,
        epsabs=3.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    l1_result = adaptive_controller(
        _partition(),
        payload_estimator,
        node_cost=1,
        max_evaluations=1,
        max_regions=4,
        epsabs=3.0,
        epsrel=0.0,
        error_norm=L1Norm(),
    )
    assert max_result.status == QuadStatus.CONVERGED
    assert l1_result.status == QuadStatus.MAX_EVALUATIONS
    assert jnp.array_equal(max_result.error, jnp.asarray([2.4, 2.4]))


def test_controller_no_improvement_roundoff_threshold_is_exact() -> None:
    def unchanged_estimator(lower, upper, _segment_id):
        width = upper - lower
        return LocalEstimate(width, width, jnp.asarray(False))

    below = adaptive_controller(
        _partition(),
        unchanged_estimator,
        node_cost=1,
        max_evaluations=100,
        max_regions=6,
        epsabs=0.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    at_threshold = adaptive_controller(
        _partition(),
        unchanged_estimator,
        node_cost=1,
        max_evaluations=100,
        max_regions=7,
        epsabs=0.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    assert below.status == QuadStatus.MAX_REGIONS
    assert below.no_improvement_count == 5
    assert at_threshold.status == QuadStatus.ROUNDOFF_LIMITED
    assert at_threshold.no_improvement_count == 6


def test_stagnation_adds_child_scalar_priorities_for_disjoint_payloads() -> None:
    def disjoint_estimator(lower, upper, _segment_id):
        width = upper - lower
        parent_error = jnp.asarray([1.5, 0.0])
        child_error = jnp.where(
            lower < 0.0, jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 1.0])
        )
        error = jnp.where(width == 2.0, parent_error, child_error)
        return LocalEstimate(
            value=jnp.stack((width, width)),
            error=error,
            nonfinite=jnp.asarray(False),
        )

    result = adaptive_controller(
        _partition(),
        disjoint_estimator,
        node_cost=1,
        max_evaluations=10,
        max_regions=2,
        epsabs=0.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    assert result.status == QuadStatus.MAX_REGIONS
    assert result.no_improvement_count == 1


def test_controller_error_growth_roundoff_threshold_is_exact() -> None:
    def growing_estimator(lower, upper, _segment_id):
        width = upper - lower
        return LocalEstimate(width + width**2, 1.0 / width, jnp.asarray(False))

    below = adaptive_controller(
        _partition(),
        growing_estimator,
        node_cost=1,
        max_evaluations=100,
        max_regions=14,
        epsabs=0.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    at_threshold = adaptive_controller(
        _partition(),
        growing_estimator,
        node_cost=1,
        max_evaluations=100,
        max_regions=15,
        epsabs=0.0,
        epsrel=0.0,
        error_norm=MaxNorm(),
    )
    assert below.status == QuadStatus.MAX_REGIONS
    assert below.growth_count == 4
    assert at_threshold.status == QuadStatus.ROUNDOFF_LIMITED
    assert at_threshold.growth_count == 5


def test_stagnation_ratio_boundaries_are_mutation_resistant() -> None:
    parent = jnp.asarray(1.0, dtype=jnp.float64)
    below_no_improvement = jnp.nextafter(
        0.99 * parent, jnp.asarray(0.0, dtype=jnp.float64)
    )
    above_growth = jnp.nextafter(1.01 * parent, jnp.asarray(jnp.inf, dtype=jnp.float64))
    equality = _stagnation_hits(
        value_delta=jnp.asarray(0.0),
        value_scale=jnp.asarray(0.0),
        child_priority=0.99 * parent,
        parent_priority=parent,
        refinements=jnp.asarray(10),
    )
    below = _stagnation_hits(
        value_delta=jnp.asarray(0.0),
        value_scale=jnp.asarray(0.0),
        child_priority=below_no_improvement,
        parent_priority=parent,
        refinements=jnp.asarray(10),
    )
    growth_equality = _stagnation_hits(
        value_delta=jnp.asarray(1.0),
        value_scale=jnp.asarray(0.0),
        child_priority=1.01 * parent,
        parent_priority=parent,
        refinements=jnp.asarray(10),
    )
    growth_above = _stagnation_hits(
        value_delta=jnp.asarray(1.0),
        value_scale=jnp.asarray(0.0),
        child_priority=above_growth,
        parent_priority=parent,
        refinements=jnp.asarray(10),
    )
    assert equality == (True, False)
    assert below == (False, False)
    assert growth_equality == (False, False)
    assert growth_above == (False, True)


def test_stagnation_scale_uses_value_arithmetic_precision() -> None:
    parent = jnp.asarray(1.0, dtype=jnp.float64)
    child = parent + jnp.asarray(1.0e-10, dtype=jnp.float64)
    scale = _value_stagnation_scale(parent, child, MaxNorm())
    assert scale.dtype == jnp.float64
    assert jnp.abs(child - parent) > scale


def test_controller_trace_has_one_loop_body_across_capacities() -> None:
    def trace(max_regions, max_evaluations):
        return str(
            jax.make_jaxpr(
                lambda epsabs: (
                    adaptive_controller(
                        _partition(),
                        _quadratic_error_estimator,
                        node_cost=2,
                        max_evaluations=max_evaluations,
                        max_regions=max_regions,
                        epsabs=epsabs,
                        epsrel=0.0,
                        error_norm=MaxNorm(),
                    ).value
                )
            )(jnp.asarray(0.1))
        )

    small = trace(4, 20)
    large = trace(64, 1000)
    assert small.count("while[") == 1
    assert large.count("while[") == 1
    assert small.count("integer_pow") == large.count("integer_pow")
    assert small.count("integer_pow") > 0
