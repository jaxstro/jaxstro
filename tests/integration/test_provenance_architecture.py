"""Executable ownership contracts for the provenance architecture page."""

from __future__ import annotations

from pathlib import Path

import jaxstro.provenance as runtime_provenance
from jaxstro.testing import ALLOWED_STATUSES, ProvenanceCard, validate_card

REPO_ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_PAGE = REPO_ROOT / "docs" / "20-architecture" / "provenance.md"


def _page_text() -> str:
    return PROVENANCE_PAGE.read_text(encoding="utf-8")


def test_provenance_page_matches_both_installed_ownership_surfaces() -> None:
    text = _page_text()

    for symbol in (
        "ArtifactHash",
        "EnvironmentSnapshot",
        "MethodManifest",
        "hash_artifact",
        "environment_snapshot",
        "manifest_to_json",
        "manifest_to_markdown",
    ):
        assert hasattr(runtime_provenance, symbol)
        assert f"`{symbol}`" in text

    assert ProvenanceCard is not None
    assert validate_card is not None
    assert "`jaxstro.testing`" in text
    assert "`ProvenanceCard`" in text
    assert "`validate_card`" in text


def test_provenance_page_has_a_structured_ownership_comparison() -> None:
    text = _page_text()

    assert "title: Provenance architecture" in text
    assert "```{list-table} Provenance ownership" in text
    assert ":label: tbl-provenance-ownership" in text
    for header in ("Surface", "Question answered", "Inputs", "Output", "Validation"):
        assert f"  - {header}" in text or f"* - {header}" in text
    assert "Runtime manifest" in text
    assert "Source-backed card" in text


def test_provenance_page_states_card_status_and_file_format_boundaries() -> None:
    text = _page_text()

    assert set(ALLOWED_STATUSES) == {
        "verified",
        "needs-check",
        "unverifiable-scanned",
    }
    for status in ALLOWED_STATUSES:
        assert f"`{status}`" in text

    assert "already-parsed mappings" in text
    assert "does not parse YAML" in text
    assert "exact locator" in text
    assert "assertion-bearing pytest node IDs" in text
    assert "Neither surface substitutes for the other" in text


def test_provenance_page_routes_generated_families_and_honest_gaps() -> None:
    text = _page_text()

    for route in (
        "../40-api/provenance/index.md",
        "../40-api/provenance/constants.md",
        "../40-api/provenance/transforms.md",
        "../40-api/provenance/atmospheres.md",
        "../60-validation/index.md",
    ):
        assert f"[]({route})" in text

    assert (
        "Zero registered atmosphere cards do not mean complete atmosphere coverage"
        in text
    )
