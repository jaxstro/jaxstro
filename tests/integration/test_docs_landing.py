"""Executable currency contracts for the website landing page."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LANDING_PAGE = REPO_ROOT / "docs" / "index.md"


def _landing_text() -> str:
    return LANDING_PAGE.read_text(encoding="utf-8")


def test_landing_leads_with_the_scientific_chain_and_why_jax() -> None:
    text = _landing_text()

    assert "astro-first" in text
    assert "science-general" in text
    assert "evidence-first" in text
    assert "representation -> computation -> audit -> evidence -> claim" in text
    assert "[](./00-start-here/why-jax.md)" in text


def test_landing_distinguishes_current_units_from_deferred_quantity_adoption() -> None:
    text = _landing_text()

    assert "`jaxstro.units` is the current canonical ecosystem contract" in text
    assert "`jaxstro.quantity` is implemented" in text
    assert "ecosystem adoption and any replacement cutover are deferred" in text
    assert "planned `jaxstro.quantity`" not in text


def test_landing_uses_research_questions_and_direct_routes() -> None:
    text = _landing_text()

    card_titles = re.findall(r"^:::\{card\}\s+(.+)$", text, re.MULTILINE)
    assert len(card_titles) >= 5
    assert all(title.endswith("?") for title in card_titles)
    for route in (
        "./10-foundations/foundations.md",
        "./20-methods/methods.md",
        "./40-workflows/workflows.md",
        "./50-api/api.md",
        "./60-validation/validation.md",
    ):
        assert f":link: {route}" in text


def test_landing_is_a_concise_entry_page_not_a_duplicate_toc() -> None:
    text = _landing_text()

    assert len(text.split()) <= 500
    assert "## Routed paths" not in text
    assert "## Scientific capabilities" not in text
    assert "## What jaxstro is *not*" not in text
