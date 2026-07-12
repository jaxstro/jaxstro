"""Focused contracts for safeguarded scalar root finding."""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.numerics import rootfinding


def test_brent_state_separates_bracket_from_interpolation_history() -> None:
    bracket = rootfinding.initialize_bracket(0.0, 2.0, -2.0, 2.0)
    state = rootfinding.initialize_bracketed_root_state(bracket)

    assert state.bracket == bracket
    assert jnp.isnan(state.history.previous_x)
    assert jnp.isnan(state.history.previous_fx)
    assert jnp.isnan(state.history.previous_previous_x)
    assert not bool(state.history.initialized)
    assert state.status == rootfinding.ROOT_STATUS_RUNNING


def test_invalid_advance_is_exact_full_state_noop() -> None:
    bracket = rootfinding.initialize_bracket(0.0, 2.0, -2.0, 2.0)
    state = rootfinding.initialize_bracketed_root_state(bracket)
    proposal = rootfinding.propose_bracketed(state, safeguard_fraction=0.1)
    updated = rootfinding.advance_bracketed_root(
        state, proposal, jnp.asarray(0.0), valid=False
    )

    comparisons = jax.tree.map(
        lambda expected, actual: jnp.array_equal(expected, actual, equal_nan=True),
        state,
        updated,
    )
    assert all(bool(value) for value in jax.tree.leaves(comparisons))


def test_nonfinite_advance_preserves_evidence_and_sets_status() -> None:
    bracket = rootfinding.initialize_bracket(0.0, 2.0, -2.0, 2.0)
    state = rootfinding.initialize_bracketed_root_state(bracket)
    proposal = rootfinding.propose_bracketed(state, safeguard_fraction=0.1)

    updated = rootfinding.advance_bracketed_root(state, proposal, jnp.nan)

    assert updated.bracket == state.bracket
    assert jax.tree.all(
        jax.tree.map(
            lambda actual, expected: jnp.array_equal(actual, expected, equal_nan=True),
            updated.history,
            state.history,
        )
    )
    assert updated.status == rootfinding.ROOT_STATUS_NONFINITE_EVALUATION


def test_three_distinct_points_enable_inverse_quadratic_proposal() -> None:
    bracket = rootfinding.initialize_bracket(0.0, 1.5, -2.0, 0.25)
    history = rootfinding.BracketHistory(2.0, 2.0, 1.0, False, True)
    state = rootfinding.BracketedRootState(
        bracket, history, rootfinding.ROOT_STATUS_RUNNING
    )

    proposal = rootfinding.propose_bracketed(state, safeguard_fraction=0.01)

    assert proposal.kind == rootfinding.PROPOSAL_INVERSE_QUADRATIC
    assert state.bracket.lo < proposal.x < state.bracket.hi


@pytest.mark.parametrize("valid", [False, jnp.asarray(False)])
def test_invalid_advance_does_not_change_initialized_history(valid) -> None:
    bracket = rootfinding.initialize_bracket(0.0, 2.0, -2.0, 2.0)
    history = rootfinding.BracketHistory(1.0, -1.0, 0.0, False, True)
    state = rootfinding.BracketedRootState(
        bracket, history, rootfinding.ROOT_STATUS_RUNNING
    )
    proposal = rootfinding.BracketProposal(
        jnp.asarray(1.5), jnp.asarray(rootfinding.PROPOSAL_SECANT), jnp.asarray(False)
    )

    updated = rootfinding.advance_bracketed_root(state, proposal, 0.5, valid=valid)

    comparisons = jax.tree.map(jnp.array_equal, state, updated)
    assert all(bool(value) for value in jax.tree.leaves(comparisons))


def test_terminal_statuses_are_explicit() -> None:
    exact_lo = rootfinding.safeguarded_bracketed_root(
        lambda x: x, 0.0, 2.0, max_steps=4
    )
    exact_hi = rootfinding.safeguarded_bracketed_root(
        lambda x: x - 2.0, 0.0, 2.0, max_steps=4
    )
    exact_interior = rootfinding.safeguarded_bracketed_root(
        lambda x: x - 1.0, 0.0, 2.0, max_steps=4
    )
    missing = rootfinding.safeguarded_bracketed_root(
        lambda x: x**2 + 1.0, -1.0, 1.0, max_steps=4
    )
    exhausted = rootfinding.safeguarded_bracketed_root(
        lambda x: x**2 - 2.0, 0.0, 2.0, max_steps=1, atol=0.0, rtol=0.0
    )

    assert exact_lo.status == rootfinding.ROOT_STATUS_EXACT_LO
    assert exact_hi.status == rootfinding.ROOT_STATUS_EXACT_HI
    assert exact_interior.status == rootfinding.ROOT_STATUS_EXACT_INTERIOR
    assert missing.status == rootfinding.ROOT_STATUS_MISSING_BRACKET
    assert exhausted.status == rootfinding.ROOT_STATUS_MAX_STEPS


def test_nonfinite_evaluation_has_distinct_status() -> None:
    def f(x):
        return jnp.where((x > 0.0) & (x < 2.0), jnp.nan, x - 1.0)

    result = rootfinding.safeguarded_bracketed_root(f, 0.0, 2.0, max_steps=4)

    assert result.status == rootfinding.ROOT_STATUS_NONFINITE_EVALUATION
    assert not result.converged
    assert result.bracketed


def test_serialization_complete_result_and_trace_fields() -> None:
    result = rootfinding.safeguarded_bracketed_root(
        lambda x: x - 0.25, 0.0, 1.0, max_steps=3
    )

    assert rootfinding.RootTrace._fields == (
        "proposal",
        "residual",
        "lo",
        "hi",
        "f_lo",
        "f_hi",
        "proposal_kind",
        "executed",
        "admissible",
        "converged",
        "status",
    )
    assert rootfinding.BracketedRootResult._fields == (
        "root",
        "residual",
        "status",
        "converged",
        "bracketed",
        "n_evaluations",
        "residual_scale",
        "final_bracket",
        "trace",
    )
    assert result.residual_scale == 0.75
    assert result.final_bracket == rootfinding.BracketState(
        result.final_bracket.lo,
        result.final_bracket.hi,
        result.final_bracket.f_lo,
        result.final_bracket.f_hi,
        result.final_bracket.bracketed,
    )
