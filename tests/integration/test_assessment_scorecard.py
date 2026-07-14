"""Contracts for the living package assessment scorecard."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORECARD = ROOT / "docs/70-project/development/package-assessment-scorecard.md"


def test_scorecard_has_grades_evidence_and_promotion_rules() -> None:
    text = SCORECARD.read_text(encoding="utf-8")
    assert "Assessment date: 2026-07-14" in text
    assert "| Dimension | Grade | Evidence |" in text
    for phrase in (
        "# Jaxstro research-software assessment scorecard",
        "## Evaluation criteria",
        "## Current grades",
        "## Coverage by research workflow",
        "## Grade-change policy",
        "Deficiency preventing the next grade",
        "Promotion evidence required",
        "Scientific contract registry",
    ):
        assert phrase in text
    assert "| Research workflow coverage | B+ |" in text
    assert "| Contract coverage | B+ |" in text
    assert "| Evidence linkage | A- |" in text
    assert "| Limitation coverage | B+ |" in text
    assert "| Downstream usefulness | B+ |" in text
    assert text.count("[](") >= 10


def test_scorecard_is_navigable_and_linked_from_sota() -> None:
    myst = (ROOT / "docs/myst.yml").read_text(encoding="utf-8")
    sota = (ROOT / "docs/70-project/development/sota-assessment.md").read_text(
        encoding="utf-8"
    )
    assert myst.count("70-project/development/package-assessment-scorecard.md") == 1
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
    assert "Executable research workflow registry: implemented" in text
    assert f"| {len(evidence['entries'])} | artifacts |" in text
    assert f"| {len(classes)} | evidence classes |" in text


def test_scorecard_closes_executable_workflows_from_generated_coverage() -> None:
    text = SCORECARD.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    coverage = json.loads(
        (ROOT / "docs/validation/research-workflow-coverage.json").read_text(
            encoding="utf-8"
        )
    )
    contract_count = sum(item["contract_count"] for item in coverage["workflows"])
    indexed_count = sum(item["evidence_count"] for item in coverage["workflows"])
    assert "Executable research workflow registry: implemented" in text
    assert f"| {coverage['workflow_count']} | investigations |" in text
    assert f"| {contract_count} | contract links |" in text
    assert f"| {indexed_count} | indexed evidence links |" in text
    assert (
        "does not automatically validate any downstream scientific model" in normalized
    )
