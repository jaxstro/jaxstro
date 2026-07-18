import itertools

import jax
import jax.numpy as jnp
import pytest

from jaxstro import quad
from jaxstro.quad._sparse import (
    RUNNING,
    adaptive_sparse_controller,
    admissible_forward_neighbors,
    required_frontier_capacity,
    select_profit,
    sparse_termination_status,
)
from jaxstro.quad.measures import LebesgueMeasure
from jaxstro.quad.tolerance import MaxNorm


def _options(**overrides):
    options = dict(
        method=quad.AdaptiveSmolyak(initial_level=1),
        epsabs=1.0e-10,
        epsrel=1.0e-10,
        max_evaluations=512,
        max_indices=8,
        max_frontier=17,
        max_nodes=512,
        gradient="stop",
    )
    options.update(overrides)
    return options


def test_candidate_requires_every_backward_neighbor():
    accepted = {(1, 1), (2, 1)}
    assert (3, 1) in admissible_forward_neighbors(accepted, 2)
    assert (2, 2) not in admissible_forward_neighbors(accepted, 2)


def test_profit_uses_surplus_per_new_node_and_lexicographic_tie():
    index = select_profit(
        ((2, 1), (1, 2)),
        jnp.array([2.0, 2.0]),
        jnp.array([4, 4]),
    )
    assert index == 1


def test_profit_skips_zero_cost_candidate_while_positive_cost_exists():
    slot = select_profit(
        ((1, 2), (2, 1)),
        jnp.array([100.0, 2.0]),
        jnp.array([0, 4]),
    )
    assert slot == 1


@pytest.mark.parametrize("initial_level", (0, -1, True, 2.5))
def test_adaptive_smolyak_rejects_invalid_initial_level(initial_level):
    with pytest.raises(ValueError, match="positive integer"):
        quad.AdaptiveSmolyak(initial_level=initial_level)


def test_frontier_capacity_bound_covers_declared_dimensions_and_counts():
    for dimension, accepted_count in itertools.product(range(2, 17), range(1, 65)):
        assert required_frontier_capacity(dimension, accepted_count) == (
            1 + dimension * accepted_count
        )


def test_adaptive_smolyak_reports_frontier_and_index_exhaustion():
    result = quad.integrate(
        lambda x: jnp.exp(jnp.sum(x, axis=-1)),
        quad.Hyperrectangle(jnp.zeros(3), jnp.ones(3)),
        **_options(
            epsabs=0.0,
            epsrel=0.0,
            max_indices=2,
            max_frontier=7,
        ),
    )
    assert result.status == quad.QuadStatus.MAX_INDICES
    assert result.work.refinements == 1
    assert result.error.kind == quad.ErrorKind.SPARSE_GRID_SURPLUS


def test_replay_evidence_is_downward_closed_and_keeps_formula_coefficients():
    controller = adaptive_sparse_controller(
        lambda x: jnp.exp(-8.0 * x[:, 0]),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        args=(),
        measure=LebesgueMeasure(),
        initial_indices=((1, 1),),
        epsabs=0.0,
        epsrel=0.0,
        max_evaluations=128,
        max_indices=4,
        max_frontier=9,
        max_nodes=128,
        error_norm=MaxNorm(),
        zero=jnp.asarray(0.0),
    )
    active_indices = {
        tuple(map(int, index))
        for index, active in zip(
            controller.evidence.indices,
            controller.evidence.active,
            strict=True,
        )
        if bool(active)
    }
    assert tuple(map(int, controller.evidence.indices[1])) == (2, 1)
    assert len(active_indices) == int(controller.refinements) + 1
    for index in active_indices:
        for axis, component in enumerate(index):
            if component > 1:
                backward = list(index)
                backward[axis] -= 1
                assert tuple(backward) in active_indices
    assert jnp.sum(controller.evidence.coefficients) == pytest.approx(1.0)


def test_insufficient_unique_node_budget_returns_max_evaluations():
    result = quad.integrate(
        lambda x: jnp.exp(jnp.sum(x, axis=-1)),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(
            epsabs=0.0,
            epsrel=0.0,
            max_evaluations=3,
            max_nodes=3,
            max_indices=4,
            max_frontier=9,
        ),
    )
    assert result.status == quad.QuadStatus.MAX_EVALUATIONS
    assert result.work.evaluations <= 3


def test_nonfinite_new_batch_returns_nonfinite_integrand():
    result = quad.integrate(
        lambda x: jnp.where(x[:, 0] == 0.0, jnp.nan, 1.0),
        quad.Hyperrectangle(jnp.zeros(2), jnp.ones(2)),
        **_options(),
    )
    assert result.status == quad.QuadStatus.NONFINITE_INTEGRAND
    assert jnp.isnan(result.value)


def test_traced_nonfinite_bound_precedes_coincident_axis():
    @jax.jit
    def solve(upper):
        return quad.integrate(
            lambda x: jnp.sum(x, axis=-1),
            quad.Hyperrectangle(jnp.zeros(2), upper),
            **_options(),
        )

    result = solve(jnp.array([0.0, jnp.inf]))
    assert result.status == quad.QuadStatus.INVALID_INPUT
    assert result.work.evaluations == 0


def test_zero_volume_evaluates_no_physical_sparse_node():
    calls = 0

    def integrand(x):
        def record(_value):
            nonlocal calls
            calls += 1

        jax.debug.callback(record, x[0, 0])
        return jnp.sum(x, axis=-1)

    result = quad.integrate(
        integrand,
        quad.Hyperrectangle(jnp.array([0.0, 1.0]), jnp.array([2.0, 1.0])),
        **_options(),
    )
    assert calls == 0
    assert result.value == 0.0
    assert result.work.evaluations == 0


def test_status_precedence_and_roundoff_contract():
    common = dict(
        invalid=False,
        nonfinite=False,
        converged=False,
        all_active_roundoff=False,
        evaluation_exhausted=False,
        index_exhausted=False,
    )
    assert sparse_termination_status(**common) == RUNNING
    for field, expected in (
        ("index_exhausted", quad.QuadStatus.MAX_INDICES),
        ("evaluation_exhausted", quad.QuadStatus.MAX_EVALUATIONS),
        ("all_active_roundoff", quad.QuadStatus.ROUNDOFF_LIMITED),
        ("converged", quad.QuadStatus.CONVERGED),
        ("nonfinite", quad.QuadStatus.NONFINITE_INTEGRAND),
        ("invalid", quad.QuadStatus.INVALID_INPUT),
    ):
        flags = common | {field: True}
        assert sparse_termination_status(**flags) == expected

    assert (
        sparse_termination_status(
            invalid=True,
            nonfinite=True,
            converged=True,
            all_active_roundoff=True,
            evaluation_exhausted=True,
            index_exhausted=True,
        )
        == quad.QuadStatus.INVALID_INPUT
    )
