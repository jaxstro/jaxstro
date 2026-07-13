"""Contracts for the living package assessment scorecard."""

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
