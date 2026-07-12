"""Focused contracts for safeguarded scalar root finding."""

import jax.numpy as jnp

from jaxstro.numerics import rootfinding


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
