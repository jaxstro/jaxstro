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
