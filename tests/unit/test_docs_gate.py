"""Contracts for the reusable rendered-documentation gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_docs_site.py"
BIBLIOGRAPHY_PAGE = (
    REPO_ROOT / "docs" / "70-project" / "bibliography" / "bibliography.md"
)


def _load_checker():
    assert CHECKER_PATH.is_file(), "rendered-docs checker is not implemented"
    spec = importlib.util.spec_from_file_location("check_docs_site", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_content(content_dir: Path, name: str, location: str) -> None:
    content_dir.mkdir(parents=True, exist_ok=True)
    (content_dir / f"{name}.json").write_text(
        json.dumps({"location": location}), encoding="utf-8"
    )


def test_xref_rejects_duplicate_page_routes(tmp_path: Path) -> None:
    checker = _load_checker()
    content_dir = tmp_path / "content"
    _write_content(content_dir, "one", "/one.md")
    _write_content(content_dir, "two", "/two.md")
    xref = {
        "references": [
            {"kind": "page", "data": "/content/one.json", "url": "/same"},
            {"kind": "page", "data": "/content/two.json", "url": "/same"},
        ]
    }

    with pytest.raises(checker.DocsGateError, match="duplicate page route"):
        checker.extract_page_routes(xref, content_dir)


def test_route_manifest_rejects_root_flat_slug_drift() -> None:
    checker = _load_checker()
    expected = {"70-project/direction/architecture.md": "/architecture"}
    actual = {"70-project/direction/architecture.md": "/architecture-drift"}

    with pytest.raises(checker.DocsGateError, match="route manifest drift"):
        checker.validate_route_manifest(actual, expected)


def test_development_server_routes_remain_unprefixed_with_base_path() -> None:
    checker = _load_checker()

    assert checker.development_server_path("/", "/jaxstro") == "/"
    assert checker.development_server_path("/quantities", "/jaxstro") == ("/quantities")


@pytest.mark.parametrize(
    ("html", "message"),
    [
        ('<h2 id="same">A</h2><div id="same">B</div>', "duplicate HTML id"),
        ('<a href="/missing">missing</a>', "unresolved internal link"),
        ('<a href="/two" target="_blank">two</a>', "internal link opens"),
        ('<img src="figure.webp">', "missing nonempty alt"),
    ],
)
def test_dom_audit_rejects_rendered_contract_violations(
    html: str, message: str
) -> None:
    checker = _load_checker()

    with pytest.raises(checker.DocsGateError, match=message):
        checker.audit_html("/one", html, {"/one", "/two"})


def test_dom_audit_accepts_resolved_accessible_markup() -> None:
    checker = _load_checker()
    html = (
        '<main id="content"><a href="/two#section">two</a>'
        '<a href="https://example.com" target="_blank" rel="noreferrer">source</a>'
        '<img src="figure.webp" alt="Measured interpolation fixture"></main>'
    )

    checker.audit_html("/one", html, {"/one", "/two"})


def test_dom_audit_canonicalizes_configured_base_path() -> None:
    checker = _load_checker()

    checker.audit_html(
        "/one",
        '<a href="/jaxstro/two">two</a>',
        {"/one", "/two"},
        base_path="/jaxstro",
    )
    checker.audit_html(
        "/one",
        '<a href="/two">development route</a>',
        {"/one", "/two"},
        base_path="/jaxstro",
    )


def test_generated_bibliography_owns_the_references_heading() -> None:
    text = BIBLIOGRAPHY_PAGE.read_text(encoding="utf-8")

    assert "## References" not in text
    assert "```{bibliography}" in text
