"""Contracts for the future capabilities and module roadmap."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "70-project/development" / "future-capabilities-roadmap.md"
NUMERICAL_ROADMAP = (
    ROOT / "docs" / "70-project/development" / "numerical-methods-roadmap.md"
)
QMC_PROGRAM_HEADING = "### Active program: `jaxstro.quad`"
QMC_CLAIMS = re.compile(
    r"\b(?:QMC|Sobol|Halton|Latin[- ]hypercube|low[- ]discrepancy|"
    r"quasi[- ](?:random|Monte[- ]Carlo))\b",
    re.IGNORECASE,
)


def _section(text: str, heading: str) -> str:
    match = re.search(
        rf"^{re.escape(heading)}\s*$\n(?P<body>.*?)(?=^###\s|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, heading
    return match.group("body")


def _checked_items(text: str) -> tuple[str, ...]:
    return tuple(
        re.sub(r"\s+", " ", match.group("item")).strip()
        for match in re.finditer(
            r"^- \[[xX]\] (?P<item>.*?)(?=^- \[[ xX]\] |^## |\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
    )


def _qmc_scope_errors(future: str, numerical: str) -> tuple[str, ...]:
    qmc_program = _section(future, QMC_PROGRAM_HEADING)
    errors = [
        f"completed quad QMC item: {item}"
        for item in _checked_items(qmc_program)
        if QMC_CLAIMS.search(item)
    ]
    errors.extend(
        f"completed numerical-roadmap QMC claim: {item}"
        for item in _checked_items(numerical)
        if QMC_CLAIMS.search(item)
    )
    return tuple(errors)


def test_future_capabilities_roadmap_preserves_scope_and_build_advice() -> None:
    text = PAGE.read_text(encoding="utf-8")

    required = (
        "# Future modules and capabilities roadmap",
        "## Existing methods and features",
        "## What Jaxstro should become",
        "## Recommended additions",
        "### Active program: `jaxstro.quad`",
        "### Priority 1: `jaxstro.ml`",
        "### Priority 2: `jaxstro.uncertainty`",
        "### Priority 3: `jaxstro.signal`",
        "### Priority 4: consumer-driven ecosystem adapters",
        "### Priority 5: fields only after two consumers",
        "## What not to add",
        "## Build checklist",
        "predict -> compute -> audit -> state the warranted claim",
        "Informax",
    )
    for phrase in required:
        assert phrase in text

    assert text.count("- [ ]") >= 10
    assert "No homegrown neural-network framework" in text
    assert "No general MCMC, VI, NPE, SBI" in text
    for owner in ("Lineax", "Optimistix", "Quadax", "Diffrax"):
        assert owner in text
    assert "Jaxstro owns the approved quadrature capability program" in text
    assert "Quadax remains a validation and benchmark comparator" in text


def test_future_capabilities_roadmap_is_navigable() -> None:
    myst = (ROOT / "docs" / "myst.yml").read_text(encoding="utf-8")
    manifest = json.loads(
        (ROOT / "docs" / "route-manifest.json").read_text(encoding="utf-8")
    )

    path = "70-project/development/future-capabilities-roadmap.md"
    assert myst.count(path) == 1
    assert manifest[path] == "/future-capabilities-roadmap"


def test_development_log_links_future_capabilities_roadmap() -> None:
    index = (ROOT / "docs" / "70-project/development" / "development.md").read_text(
        encoding="utf-8"
    )
    assert "future-capabilities-roadmap.md" in index


def test_completed_numerical_items_do_not_claim_deferred_qmc_scope() -> None:
    future = PAGE.read_text(encoding="utf-8")
    numerical = NUMERICAL_ROADMAP.read_text(encoding="utf-8")
    qmc_program = _section(future, QMC_PROGRAM_HEADING)

    assert "randomized quasi-Monte Carlo" in qmc_program
    assert not any(QMC_CLAIMS.search(item) for item in _checked_items(qmc_program))
    assert _qmc_scope_errors(future, numerical) == ()


def test_qmc_priority_rejects_uppercase_completed_checkbox_mutation() -> None:
    future = PAGE.read_text(encoding="utf-8")
    numerical = NUMERICAL_ROADMAP.read_text(encoding="utf-8")
    mutated = future.replace(
        "- [ ] Design and approve Phase B hyperrectangle integration, adaptive cubature,",
        "- [X] Design and approve Phase B hyperrectangle integration, adaptive cubature,",
        1,
    )

    assert _qmc_scope_errors(mutated, numerical)


def test_entire_qmc_priority_rejects_any_completed_item_mutation() -> None:
    future = PAGE.read_text(encoding="utf-8")
    numerical = NUMERICAL_ROADMAP.read_text(encoding="utf-8")
    mutated = future.replace(
        "- [ ] Design and approve Phase B",
        "- [x] Design and approve Phase B",
        1,
    )

    assert _qmc_scope_errors(mutated, numerical)


@pytest.mark.parametrize(
    "claim",
    (
        "QMC sequences",
        "Halton sequences",
        "low-discrepancy sequences",
        "quasi-Monte-Carlo sequences",
    ),
)
def test_numerical_roadmap_rejects_deferred_qmc_alias_mutations(claim: str) -> None:
    future = PAGE.read_text(encoding="utf-8")
    numerical = NUMERICAL_ROADMAP.read_text(encoding="utf-8")
    mutated = f"{numerical}\n- [x] **Mutation fixture.** Completed {claim}.\n"

    assert _qmc_scope_errors(future, mutated)
