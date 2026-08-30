"""Algorithm-level invariants for safeguarded scalar root steps."""

import jax.numpy as jnp
import pytest

from jaxstro.numerics import rootfinding


@pytest.mark.parametrize(
    ("f", "lo", "hi"),
    [
        (lambda x: x - 0.3, 0.0, 1.0),
        (lambda x: x**2 - 2.0, 0.0, 2.0),
        (lambda x: (x - 0.4) ** 3, 0.0, 1.0),
        (lambda x: jnp.abs(x - 0.2) - 0.3, 0.2, 1.0),
        (lambda x: 0.8 - 0.5 * x - x, 0.0, 1.0),
    ],
)
def test_admissible_advances_preserve_true_sign_bracket(f, lo, hi) -> None:
    bracket = rootfinding.initialize_bracket(lo, hi, f(lo), f(hi))
    state = rootfinding.initialize_bracketed_root_state(bracket)

    for _ in range(12):
        proposal = rootfinding.propose_bracketed(state, safeguard_fraction=0.05)
        if int(proposal.kind) == rootfinding.PROPOSAL_NONE:
            break
        state = rootfinding.advance_bracketed_root(state, proposal, f(proposal.x))
        bracket = state.bracket
        assert bracket.lo <= bracket.hi
        assert bool(
            (bracket.f_lo == 0.0)
            | (bracket.f_hi == 0.0)
            | (jnp.signbit(bracket.f_lo) != jnp.signbit(bracket.f_hi))
        )


@pytest.mark.parametrize(
    ("f", "lo", "hi", "expected"),
    [
        (lambda x: x - 0.3, 0.0, 1.0, 0.3),
        (lambda x: x**2 - 2.0, 0.0, 2.0, jnp.sqrt(2.0)),
    ],
)
def test_public_safeguarded_solver_converges_on_analytic_roots(
    f, lo, hi, expected
) -> None:
    result = rootfinding.safeguarded_bracketed_root(
        f, lo, hi, max_steps=64, atol=1.0e-7, rtol=0.0
    )
    assert bool(result.bracketed)
    assert bool(result.converged)
    assert float(jnp.abs(result.root - expected)) <= 2.0e-6
    assert float(jnp.abs(result.residual)) <= 2.0e-6


def test_public_safeguarded_solver_reports_typed_missing_and_nonfinite_cases() -> None:
    missing = rootfinding.safeguarded_bracketed_root(
        lambda x: x**2 + 1.0, -1.0, 1.0, max_steps=8
    )
    nonfinite = rootfinding.safeguarded_bracketed_root(
        lambda x: jnp.where((x > 0.0) & (x < 2.0), jnp.nan, x - 1.0),
        0.0,
        2.0,
        max_steps=8,
    )
    assert missing.status == rootfinding.ROOT_STATUS_MISSING_BRACKET
    assert not bool(missing.converged)
    assert nonfinite.status == rootfinding.ROOT_STATUS_NONFINITE_EVALUATION
    assert not bool(nonfinite.converged)
