"""Executable currency and anchor contracts for the validation page."""

from __future__ import annotations

import re
from pathlib import Path

from jaxstro.testing import resolve_node_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_PAGE = REPO_ROOT / "docs" / "60-validation" / "index.md"


def _validation_text() -> str:
    return VALIDATION_PAGE.read_text(encoding="utf-8")


def _anchor_table_text() -> str:
    text = _validation_text()
    return text.split("## Validation anchors", 1)[1].split(
        "## Local evidence commands", 1
    )[0]


def test_validation_page_describes_current_evidence_in_present_tense() -> None:
    text = _validation_text()

    assert "It will carry" not in text
    assert "The table records" in text
    assert "Every quantitative claim elsewhere in these docs should resolve" not in text
    assert "Each row states the bounded claim its cited tests enforce" in text


def test_validation_page_covers_current_cross_cutting_contracts() -> None:
    text = _anchor_table_text()

    required_claims = (
        "ICRS-to-Galactic",
        "singular coordinate locations",
        "exact fixed-radius pairs",
        "implemented quantity layer",
        "provenance-card registry",
        "Sonora and BSTAR remain `POLICY_NOT_VALIDATED`",
    )
    for claim in required_claims:
        assert claim in text


def test_validation_page_routes_claims_to_their_explanations() -> None:
    text = _validation_text()

    assert "[](../20-methods/methods.md)" in text
    assert "[](../20-methods/discrete-space/spatial.md)" in text
    assert "[](../10-theory/quantities.md)" in text
    assert "[](../40-api/provenance/index.md)" in text


def test_validation_anchor_table_uses_only_resolvable_test_paths() -> None:
    table = _anchor_table_text()
    assert "*" not in table

    paths = set(re.findall(r"tests/[A-Za-z0-9_./-]+\.py", table))
    assert paths
    missing = sorted(path for path in paths if not (REPO_ROOT / path).is_file())
    assert not missing, f"missing validation anchor paths: {missing}"


def test_validation_page_cites_collectable_registry_node_ids() -> None:
    text = _anchor_table_text()
    node_ids = {
        "tests/validation/provenance_cards/test_registry.py::"
        "test_validation_references_collect_and_assert_behavior",
        "tests/validation/provenance_cards/test_registry.py::"
        "test_generated_pages_equal_fresh_rendering",
    }

    for node_id in node_ids:
        assert node_id in text
    assert resolve_node_ids(sorted(node_ids), rootdir=str(REPO_ROOT)) == node_ids
