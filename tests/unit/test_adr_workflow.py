"""Ratchets for the tracked, public architecture-decision workflow."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADR_DIR = REPO_ROOT / "docs" / "70-project" / "decisions"
ADR_INDEX = ADR_DIR / "decisions.md"
MYST_CONFIG = REPO_ROOT / "docs" / "myst.yml"
ROUTE_MANIFEST = REPO_ROOT / "docs" / "route-manifest.json"


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
        if not re.search(rf"(?m)^# {number} - ", text):
            problems.append(f"{path.name}: heading does not match filename")
    assert not problems, "\n".join(problems)


def test_every_public_adr_is_indexed_routed_and_hidden_in_primary_navigation():
    import json

    index = ADR_INDEX.read_text(encoding="utf-8")
    myst = MYST_CONFIG.read_text(encoding="utf-8")
    manifest = json.loads(ROUTE_MANIFEST.read_text(encoding="utf-8"))
    problems = []
    for path in _adr_paths():
        relative = path.relative_to(ADR_DIR).as_posix()
        if f"./{relative}" not in index:
            problems.append(f"{path.name}: missing from decision index")
        routed = f"70-project/decisions/{relative}"
        if routed not in manifest:
            problems.append(f"{path.name}: missing from route manifest")
        hidden_entry = f"- file: {routed}\n              hidden: true"
        if hidden_entry not in myst:
            problems.append(f"{path.name}: not hidden in primary navigation")
    assert myst.count("70-project/decisions/decisions.md") == 1
    assert not problems, "\n".join(problems)
