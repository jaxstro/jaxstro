"""Research-workflow conventions for published scientific guidance."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = ROOT / "docs/40-workflows"


def test_workflow_pages_open_with_research_use_cases() -> None:
    for path in WORKFLOW_ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        if path.name == "investigations.md":
            continue
        assert "Use this page when" in text, path


def test_workflow_sources_avoid_classroom_management_framing() -> None:
    forbidden = ("course", "class", "instructor", "grading", "curric" + "ulum")
    for path in WORKFLOW_ROOT.rglob("*.md"):
        lowered = path.read_text(encoding="utf-8").lower()
        for word in forbidden:
            assert re.search(rf"\b{word}\b", lowered) is None, (path, word)
