"""Presentation and accessibility contracts for Foundations figures."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "docs/10-foundations/figures"


def test_foundations_vector_figures_are_accessible_and_white_grounded() -> None:
    for figure in sorted(FIGURES.glob("*.svg")):
        text = figure.read_text(encoding="utf-8")
        assert 'role="img"' in text, figure
        assert 'aria-labelledby="title desc"' in text, figure
        assert '<title id="title">' in text, figure
        assert '<desc id="desc">' in text, figure
        assert 'fill="#ffffff"' in text, figure
        assert "#fce6df" not in text, figure
        assert "#fff1ed" not in text, figure
