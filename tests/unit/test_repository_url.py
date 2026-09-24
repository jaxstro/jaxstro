"""Every published link points at the project's actual repository."""

import tomllib
from pathlib import Path

from jaxstro._public import REPOSITORY_URL

ROOT = Path(__file__).resolve().parents[2]
UNPUBLISHED = ("_build", "plans", "audits", "superpowers")


def test_repository_url_matches_project_metadata() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert REPOSITORY_URL == project["project"]["urls"]["Homepage"]


def test_published_sources_do_not_link_the_retired_repository() -> None:
    # github.com/drannarosen/jaxstro returned 404 on 2026-09-24.
    retired = "github.com/drannarosen/jaxstro"
    candidates = [*(ROOT / "src").rglob("*.py"), ROOT / "README.md"]
    candidates += [
        path
        for path in (ROOT / "docs").rglob("*")
        if path.suffix in {".md", ".yml", ".json"}
        and not set(path.relative_to(ROOT / "docs").parts) & set(UNPUBLISHED)
    ]
    offenders = [
        str(path.relative_to(ROOT))
        for path in candidates
        if retired in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
