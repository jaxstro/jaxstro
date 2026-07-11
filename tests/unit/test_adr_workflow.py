"""Ratchets for the tracked, public architecture-decision workflow."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADR_DIR = REPO_ROOT / "docs" / "30-decisions"
ADR_INDEX = ADR_DIR / "index.md"
MYST_CONFIG = REPO_ROOT / "docs" / "myst.yml"


def _adr_paths() -> list[Path]:
    return sorted(ADR_DIR.glob("[0-9][0-9][0-9][0-9]-*.md"))


def _frontmatter(text: str) -> str:
    assert text.startswith("---\n"), "ADR must start with YAML frontmatter"
    _, frontmatter, _ = text.split("---", 2)
    return frontmatter


def test_public_adr_numbers_are_contiguous_through_current_decision():
    numbers = [int(path.name[:4]) for path in _adr_paths()]
    assert numbers == list(range(1, 15))


def test_public_adrs_carry_required_metadata_and_matching_heading():
    required = (
        "title",
        "description",
        "id",
        "date",
        "status",
        "supersedes",
        "decided_by",
    )
    problems = []
    for path in _adr_paths():
        text = path.read_text(encoding="utf-8")
        metadata = _frontmatter(text)
        for field in required:
            if not re.search(rf"(?m)^{field}:", metadata):
                problems.append(f"{path.name}: missing {field}")
        number = path.name[:4]
        if not re.search(rf"(?m)^id:\s*{number}\s*$", metadata):
            problems.append(f"{path.name}: id does not match filename")
        if not re.search(rf"(?m)^# {number} — ", text):
            problems.append(f"{path.name}: heading does not match filename")
    assert not problems, "\n".join(problems)


def test_every_public_adr_is_indexed_and_in_myst_navigation():
    index = ADR_INDEX.read_text(encoding="utf-8")
    myst = MYST_CONFIG.read_text(encoding="utf-8")
    problems = []
    for path in _adr_paths():
        relative = path.relative_to(ADR_DIR).as_posix()
        if f"./{relative}" not in index:
            problems.append(f"{path.name}: missing from decision index")
        if f"30-decisions/{relative}" not in myst:
            problems.append(f"{path.name}: missing from MyST navigation")
    assert not problems, "\n".join(problems)
