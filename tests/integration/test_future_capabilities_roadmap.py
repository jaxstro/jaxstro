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
QMC_PROGRAM_HEADING = "### Active program: `jaxstro.quad`"


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


def _premature_quad_completion_errors(future: str) -> tuple[str, ...]:
    qmc_program = _section(future, QMC_PROGRAM_HEADING)
    return tuple(
        item
        for item in _checked_items(qmc_program)
        if (
            "Complete the Phase B exhaustive release gate" in item
            or "Design Phase C native scientific geometries" in item
        )
    )


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


def test_phase_b_qmc_is_implemented_but_release_and_phase_c_remain_open() -> None:
    future = PAGE.read_text(encoding="utf-8")
    numerical = NUMERICAL_ROADMAP.read_text(encoding="utf-8")
    qmc_program = _section(future, QMC_PROGRAM_HEADING)
    checked = _checked_items(qmc_program)

    assert any("bounded sequential randomized QMC" in item for item in checked)
    assert "bounded sequential randomized QMC" in numerical
    assert _premature_quad_completion_errors(future) == ()


def test_phase_b_release_gate_rejects_uppercase_completed_checkbox_mutation() -> None:
    future = PAGE.read_text(encoding="utf-8")
    mutated = future.replace(
        "- [ ] Complete the Phase B exhaustive release gate",
        "- [X] Complete the Phase B exhaustive release gate",
        1,
    )

    assert _premature_quad_completion_errors(mutated)


def test_phase_c_rejects_lowercase_completed_checkbox_mutation() -> None:
    future = PAGE.read_text(encoding="utf-8")
    mutated = future.replace(
        "- [ ] Design Phase C native scientific geometries",
        "- [x] Design Phase C native scientific geometries",
        1,
    )

    assert _premature_quad_completion_errors(mutated)
