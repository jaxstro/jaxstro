"""Executable claim contracts for the theory index."""

from __future__ import annotations

from pathlib import Path

from jaxstro.testing import LIVE_GRAD_CONTRACTS, contract_requires_fd

REPO_ROOT = Path(__file__).resolve().parents[2]
THEORY_INDEX = REPO_ROOT / "docs" / "20-methods" / "methods.md"


def _theory_text() -> str:
    return THEORY_INDEX.read_text(encoding="utf-8")


def test_theory_index_lists_exactly_the_live_gradient_contracts() -> None:
    text = _theory_text()
    expected = {
        "smooth_pathwise",
        "known_zero",
        "known_blocked",
        "surrogate",
        "validation_only",
    }

    assert set(LIVE_GRAD_CONTRACTS) == expected
    assert {contract for contract in expected if contract_requires_fd(contract)} == {
        "smooth_pathwise",
        "known_zero",
    }
    for contract in expected:
        assert f"`{contract}`" in text


def test_theory_index_uses_structured_gradient_contract_fields() -> None:
    text = _theory_text()

    assert "```{list-table} Gradient contracts" in text
    assert "* - Gradient contract" in text
    assert "  - AD expectation" in text
    assert "  - FD role" in text
    assert "  - Inference / claim boundary" in text
    assert "Only a clean `smooth_pathwise` result is inference-ready" in text


def test_theory_index_does_not_claim_universal_differentiability() -> None:
    text = _theory_text()

    forbidden = (
        "every public primitive will be differentiated",
        "Every primitive here is built to a single constraint",
        "it must survive `jax.grad`",
        'A `while_loop` that runs "until converged" has a data-dependent trip count, and JAX cannot differentiate through that cleanly',
    )
    for claim in forbidden:
        assert claim not in text

    assert "classify the transform contract first" in text
    assert "Fixed iteration is necessary, not sufficient" in text
    assert "bisection is a branch-selected forward solve" in text
    assert "Newton can carry a `smooth_pathwise` contract" in text


def test_theory_index_routes_discrete_and_validation_contracts() -> None:
    text = _theory_text()

    assert text.count("[](./discrete-space/spatial.md)") >= 2
    assert "inactive branch can still poison a derivative" in text
    assert "[](../60-validation/validation.md)" in text
    assert "[](../50-api/api.md)" in text
