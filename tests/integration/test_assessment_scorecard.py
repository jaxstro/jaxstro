"""Contracts for the living package assessment scorecard."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORECARD = ROOT / "docs/90-development-log/package-assessment-scorecard.md"


def test_scorecard_has_grades_evidence_and_promotion_rules() -> None:
    text = SCORECARD.read_text(encoding="utf-8")
    assert "Assessment date: 2026-07-12" in text
    assert "| Dimension | Grade | Evidence |" in text
    for phrase in (
        "# Jaxstro package assessment scorecard",
        "## Grading rubric",
        "## Current grades",
        "## Coverage by scientific area",
        "## Grade-change policy",
        "Deficiency preventing the next grade",
        "Promotion evidence required",
        "Scientific contract registry",
    ):
        assert phrase in text
    assert "| Curriculum concept | B+ |" in text
    assert "| Downstream usefulness | B+ |" in text
    assert text.count("[](../") >= 10


def test_scorecard_is_navigable_and_linked_from_sota() -> None:
    myst = (ROOT / "docs/myst.yml").read_text(encoding="utf-8")
    sota = (ROOT / "docs/90-development-log/sota-assessment.md").read_text(
        encoding="utf-8"
    )
    assert myst.count("90-development-log/package-assessment-scorecard.md") == 1
    assert "package-assessment-scorecard.md" in sota


def test_scorecard_separates_registry_delivery_from_uniform_evidence() -> None:
    text = SCORECARD.read_text(encoding="utf-8")
    assert "Scientific contract registry: implemented" in text
    assert "Evidence depth remains uneven" in text
    assert "Unified evidence infrastructure: implemented" in text
    assert "Build the Scientific contract registry" not in text


def test_scorecard_registry_counts_match_generated_inventory() -> None:
    text = SCORECARD.read_text(encoding="utf-8")
    payload = json.loads(
        (ROOT / "docs/validation/contracts.json").read_text(encoding="utf-8")
    )
    counts = (
        len(payload["modules"]),
        sum(len(module["callables"]) for module in payload["modules"]),
        len(payload["unclassified_callables"]),
        len(payload["inherited_symbols"]),
    )
    for value in counts:
        assert f"| {value} |" in text


def test_scorecard_closes_unified_evidence_phase_from_index() -> None:
    text = SCORECARD.read_text(encoding="utf-8")
    evidence = json.loads(
        (ROOT / "docs/validation/evidence-index.json").read_text(encoding="utf-8")
    )
    classes = {entry["evidence_class"] for entry in evidence["entries"]}
    assert "Unified evidence infrastructure: implemented" in text
    assert "Method-specific scientific thresholds remain method-owned" in text
    assert "Executable foundations curriculum: implemented" in text
    assert f"| {len(evidence['entries'])} | artifacts |" in text
    assert f"| {len(classes)} | evidence classes |" in text


def test_scorecard_closes_executable_curriculum_from_generated_coverage() -> None:
    text = SCORECARD.read_text(encoding="utf-8")
    coverage = json.loads(
        (ROOT / "docs/validation/curriculum-coverage.json").read_text(encoding="utf-8")
    )
    contract_count = sum(item["contract_count"] for item in coverage["units"])
    indexed_count = sum(item["evidence_count"] for item in coverage["units"])
    instructor_count = sum(bool(item["instructor_route"]) for item in coverage["units"])
    assert "Executable foundations curriculum: implemented" in text
    assert f"| {coverage['unit_count']} | investigations |" in text
    assert f"| {contract_count} | contract links |" in text
    assert f"| {indexed_count} | indexed evidence links |" in text
    assert f"| {instructor_count} | instructor routes |" in text
    assert "does not automatically raise every pedagogy grade" in text
