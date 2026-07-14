"""Contracts for the beginner JAX onboarding route."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START_HERE = ROOT / "docs" / "00-start-here"

PAGES = {
    "why-jax.md": ("# Why JAX?", "## When JAX is the wrong tool"),
    "jax-from-first-principles.md": (
        "# JAX from first principles",
        "## A Python function, a mathematical map, and a traced program",
    ),
    "ways-to-use-these-docs.md": (
        "# Ways to use these docs",
        "## Research-question first",
    ),
}


def test_beginner_pages_exist_and_name_their_boundaries() -> None:
    for name, phrases in PAGES.items():
        text = (START_HERE / name).read_text(encoding="utf-8")
        for phrase in phrases:
            assert phrase in text


def test_why_jax_does_not_promise_automatic_correctness_or_speed() -> None:
    text = (START_HERE / "why-jax.md").read_text(encoding="utf-8")
    for phrase in (
        "JAX does not make an algorithm correct",
        "JAX does not make every program faster",
        "JAX does not make every derivative scientifically meaningful",
    ):
        assert phrase in text


def test_start_here_pages_have_canonical_routes_and_toc_entries() -> None:
    myst = (ROOT / "docs" / "myst.yml").read_text(encoding="utf-8")
    manifest = json.loads(
        (ROOT / "docs" / "route-manifest.json").read_text(encoding="utf-8")
    )
    routes = {
        "00-start-here/start-here.md": "/start-here",
        "00-start-here/first-research-calculation.md": ("/first-research-calculation"),
        "00-start-here/choose-your-path.md": "/choose-your-path",
        "00-start-here/why-jax.md": "/why-jax",
        "00-start-here/jax-from-first-principles.md": ("/jax-from-first-principles"),
        "00-start-here/ways-to-use-these-docs.md": "/ways-to-use-these-docs",
    }

    assert myst.count("title: Start here") == 1
    for page, route in routes.items():
        assert myst.count(f"file: {page}") == 1
        assert manifest[page] == route
    assert "00-getting-started/index.md" not in manifest
    assert "/how-to-learn" not in manifest.values()


def test_entry_pages_were_moved_out_of_the_old_directories() -> None:
    assert not (ROOT / "docs" / "00-getting-started" / "index.md").exists()
    assert not (ROOT / "docs" / "00-getting-started" / "how-to-learn.md").exists()
    assert not (ROOT / "docs" / "05-foundations" / "choose-your-path.md").exists()
