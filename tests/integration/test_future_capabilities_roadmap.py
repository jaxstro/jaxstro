"""Contracts for the future capabilities and module roadmap."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "70-project/development" / "future-capabilities-roadmap.md"
NUMERICAL_ROADMAP = (
    ROOT / "docs" / "70-project/development" / "numerical-methods-roadmap.md"
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
            r"^- \[x\] (?P<item>.*?)(?=^- \[[ x]\] |^## |\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
    )


def test_future_capabilities_roadmap_preserves_scope_and_build_advice() -> None:
    text = PAGE.read_text(encoding="utf-8")

    required = (
        "# Future modules and capabilities roadmap",
        "## Existing methods and features",
        "## What Jaxstro should become",
        "## Recommended additions",
        "### Priority 1: `jaxstro.ml`",
        "### Priority 2: `jaxstro.numerics.qmc`",
        "### Priority 3: `jaxstro.uncertainty`",
        "### Priority 4: `jaxstro.signal`",
        "### Priority 5: consumer-driven ecosystem adapters",
        "### Priority 6: fields only after two consumers",
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
    qmc_priority = _section(future, "### Priority 2: `jaxstro.numerics.qmc`")

    assert "- [ ] Add reference-checked Sobol construction" in qmc_priority
    assert "- [ ] Add Latin-hypercube construction" in qmc_priority
    qmc_claims = re.compile(
        r"\b(?:quasi[- ](?:random|Monte Carlo)|Sobol|Latin[- ]hypercube)\b",
        re.IGNORECASE,
    )
    conflicting = [
        item for item in _checked_items(numerical) if qmc_claims.search(item)
    ]
    assert conflicting == []
