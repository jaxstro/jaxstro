"""Tests for dependency-light provenance-card validation and rendering."""

from __future__ import annotations

import ast
import inspect

import pytest

from jaxstro.testing import provenance_cards


def _card(**overrides: object) -> dict[str, object]:
    card: dict[str, object] = {
        "id": "galactic-icrs",
        "title": "Galactic / ICRS transform",
        "summary": "IAU 1958 Galactic coordinates expressed in ICRS.",
        "scope": "The fixed rotation matrix and its inverse.",
        "conventions": ["input and output angles are degrees"],
        "sources": [
            {
                "reference": "https://www.iausofa.org/s/manual_c.pdf",
                "locator": "iauG2icrs",
                "supports": "frame convention and rotation coefficients",
            }
        ],
        "code_refs": ["src/jaxstro/coords.py::_GALACTIC_TO_ICRS"],
        "validation": [
            "tests/unit/test_coords.py::TestGalacticEquatorial::test_galactic_center"
        ],
        "status": "verified",
        "deviations": [],
    }
    card.update(overrides)
    return card


@pytest.mark.parametrize(
    "missing",
    [
        "id",
        "title",
        "summary",
        "scope",
        "conventions",
        "sources",
        "code_refs",
        "validation",
        "status",
        "deviations",
    ],
)
def test_validate_card_rejects_each_missing_required_field(missing):
    card = _card()
    del card[missing]

    with pytest.raises(provenance_cards.ProvenanceCardError, match=missing):
        provenance_cards.validate_card(card, context="transforms.yaml")


@pytest.mark.parametrize("status", ["verified", "needs-check", "unverifiable-scanned"])
def test_validate_card_accepts_allowed_statuses(status):
    validated = provenance_cards.validate_card(_card(status=status))
    assert validated.status == status


def test_validate_card_rejects_unknown_status():
    with pytest.raises(provenance_cards.ProvenanceCardError, match="unknown status"):
        provenance_cards.validate_card(_card(status="probably-correct"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sources", [], "source"),
        ("code_refs", [], "code reference"),
        ("validation", [], "validation reference"),
        ("code_refs", ["src/jaxstro/coords.py"], "path::qualname"),
        ("validation", ["tests/unit/test_coords.py"], "path::node"),
    ],
)
def test_verified_card_requires_actionable_evidence(field, value, message):
    with pytest.raises(provenance_cards.ProvenanceCardError, match=message):
        provenance_cards.validate_card(_card(**{field: value}))


def test_validate_card_reports_malformed_nested_source():
    source = {
        "reference": "https://www.iausofa.org/s/manual_c.pdf",
        "supports": "rotation coefficients",
    }
    with pytest.raises(provenance_cards.ProvenanceCardError, match="locator"):
        provenance_cards.validate_card(_card(sources=[source]))


def test_render_registry_sorts_families_and_cards_deterministically():
    families = {
        "transforms": [_card(id="z-transform", title="Z transform"), _card()],
        "constants": [_card(id="fundamental", title="Fundamental constants")],
    }
    titles = {"transforms": "Transforms", "constants": "Constants"}

    first = provenance_cards.render_registry(families, family_titles=titles)
    second = provenance_cards.render_registry(
        dict(reversed(list(families.items()))), family_titles=titles
    )

    assert first == second
    assert list(first) == ["constants.md", "transforms.md", "index.md"]
    assert first["transforms.md"].index("Galactic / ICRS") < first[
        "transforms.md"
    ].index("Z transform")
    assert "\n\n(card-z-transform)=\n" in first["transforms.md"]
    assert first["index.md"].index("Constants") < first["index.md"].index("Transforms")


def test_rendered_card_has_stable_myst_and_reference_formatting():
    rendered = provenance_cards.render_card(_card())

    assert rendered.startswith("(card-galactic-icrs)=\n## Galactic / ICRS transform")
    assert "**Status:** `verified`" in rendered
    assert '<a href="https://www.iausofa.org/s/manual_c.pdf">source</a>' in rendered
    assert "*Locator:* `iauG2icrs`" in rendered
    assert "`src/jaxstro/coords.py::_GALACTIC_TO_ICRS`" in rendered
    assert (
        "`tests/unit/test_coords.py::TestGalacticEquatorial::test_galactic_center`"
        in rendered
    )
    assert rendered.endswith("\n")


def test_rendered_doi_preserves_the_canonical_resolver():
    raw = _card(
        sources=[
            {
                "reference": "https://doi.org/10.1234/example",
                "locator": "Section 1",
                "supports": "the fixture claim",
            }
        ]
    )

    rendered = provenance_cards.render_card(raw)

    assert 'href="https://doi.org/10.1234/example"' in rendered


def test_installed_module_has_no_yaml_import():
    tree = ast.parse(inspect.getsource(provenance_cards))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "yaml" not in imported
