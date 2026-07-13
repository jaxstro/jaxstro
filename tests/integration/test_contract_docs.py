"""Website contracts for the generated scientific-contract registry."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_generated_page_is_navigable_and_explains_unverified() -> None:
    page = (ROOT / "docs/40-api/contracts.md").read_text(encoding="utf-8")
    myst = (ROOT / "docs/myst.yml").read_text(encoding="utf-8")
    assert myst.count("40-api/contracts.md") == 1
    assert "# Scientific contract registry" in page
    assert "Unverified does not mean unsupported" in page
    assert "## Transform and AD contracts" in page
    assert "## Unclassified callable surfaces" in page
    assert "tests/validation/test_implicit_root_gradients.py" in page
    assert "physical per-lane skipping" in page
    assert "absence from the table is not a support" in page
