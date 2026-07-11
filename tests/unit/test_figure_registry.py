"""Contracts for jaxstro's optional documentation-figure laboratory."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")
pytest.importorskip("PIL")

from laboratory.figures.registry import FIGURES  # noqa: E402
from laboratory.figures.style import render_webp_bytes  # noqa: E402


def test_architecture_figure_is_registered_for_the_approved_page() -> None:
    spec = FIGURES["jaxstro-foundation"]

    assert spec.page == "20-architecture/index.md"
    assert spec.seed == 0
    assert spec.stem == "jaxstro-foundation"
    assert spec.site_webp == (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "20-architecture"
        / "figures"
        / "jaxstro-foundation.webp"
    )


def test_architecture_figure_contains_ownership_labels() -> None:
    figure = FIGURES["jaxstro-foundation"].builder()
    labels = {
        text.get_text()
        for text in figure.findobj(match=lambda item: hasattr(item, "get_text"))
    }

    assert {
        "Domain packages",
        "jaxstro foundation",
        "JAX + Equinox + jaxtyping",
        "spatial + atmosphere selection: host-side / discrete boundary",
        "depends on",
        "built on",
    } <= labels


def test_architecture_webp_render_is_deterministic_and_committed() -> None:
    spec = FIGURES["jaxstro-foundation"]
    first = render_webp_bytes(spec.builder())
    second = render_webp_bytes(spec.builder())

    assert first == second
    assert spec.site_webp.is_file()
    assert spec.site_webp.stat().st_size > 10_000
