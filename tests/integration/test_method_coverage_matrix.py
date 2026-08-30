from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs/70-project/development/method-coverage.md"


def test_method_coverage_matrix_is_navigable_and_claim_calibrated() -> None:
    text = PAGE.read_text(encoding="utf-8")
    for phrase in (
        "# Method coverage matrix",
        "38 method-guide pages",
        "implemented", 
        "experimental",
        "planned",
        "unclassified public callables",
        "predict -> compute -> audit",
        "does not establish scientific-model adequacy",
    ):
        assert phrase in text

    myst = (ROOT / "docs/myst.yml").read_text(encoding="utf-8")
    routes = json.loads((ROOT / "docs/route-manifest.json").read_text(encoding="utf-8"))
    path = "70-project/development/method-coverage.md"
    assert myst.count(f"file: {path}") == 1
    assert routes[path] == "/method-coverage"
