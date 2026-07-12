"""Repository wiring contracts for the reusable documentation gate."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_docs_gate_is_reused_by_local_and_full_ci_gates() -> None:
    script = REPO_ROOT / "scripts" / "check_docs.sh"
    assert script.is_file()
    assert "myst build --html --ci --strict" in script.read_text(encoding="utf-8")

    local_gate = (REPO_ROOT / "scripts" / "check.sh").read_text(encoding="utf-8")
    full_gate = (REPO_ROOT / ".github" / "workflows" / "full-gate.yml").read_text(
        encoding="utf-8"
    )
    assert "bash scripts/check_docs.sh" in local_gate
    assert "docs:" in full_gate
    assert "bash scripts/check_docs.sh" in full_gate
    assert "mystmd@1.10.1" in full_gate


def test_committed_route_manifest_matches_all_authored_pages() -> None:
    manifest_path = REPO_ROOT / "docs" / "route-manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(manifest) == 63
    assert manifest["index.md"] == "/"
    assert manifest["20-architecture/spectra-data-architecture.md"] == (
        "/spectra-data-architecture"
    )
    assert manifest["95-release/checklist.md"] == "/checklist"
    assert manifest["99-bibliography/index.md"] == "/index-11"
    assert len(manifest.values()) == len(set(manifest.values()))
