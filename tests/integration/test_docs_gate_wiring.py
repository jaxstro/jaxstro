"""Repository wiring contracts for the reusable documentation gate."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_docs_gate_is_reused_by_local_and_full_ci_gates() -> None:
    script = REPO_ROOT / "scripts" / "check_docs.sh"
    assert script.is_file()
    script_text = script.read_text(encoding="utf-8")
    assert "myst build --html --ci --strict" in script_text
    assert (
        'uv run --no-sync python "$ROOT_DIR/scripts/check_docs_site.py"' in script_text
    )

    local_gate = (REPO_ROOT / "scripts" / "check.sh").read_text(encoding="utf-8")
    full_gate = (REPO_ROOT / ".github" / "workflows" / "full-gate.yml").read_text(
        encoding="utf-8"
    )
    docs_job = full_gate.split("  test-matrix:", maxsplit=1)[0]
    assert "bash scripts/check_docs.sh" in local_gate
    assert "docs:" in full_gate
    assert "bash scripts/check_docs.sh" in docs_job
    assert "mystmd@1.10.1" in docs_job
    assert "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b" in docs_job
    assert "uv sync --locked --extra dev" in docs_job


def test_committed_route_manifest_matches_all_authored_pages() -> None:
    manifest_path = REPO_ROOT / "docs" / "route-manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    myst = (REPO_ROOT / "docs" / "myst.yml").read_text(encoding="utf-8")
    authored = set(re.findall(r"^\s*- file:\s+(.+\.md)\s*$", myst, re.MULTILINE))
    assert set(manifest) == authored
    assert manifest["index.md"] == "/"
    assert manifest["20-architecture/spectra-data-architecture.md"] == (
        "/spectra-data-architecture"
    )
    assert manifest["95-release/checklist.md"] == "/checklist"
    assert manifest["99-bibliography/index.md"] == "/index-10"
    assert manifest["60-validation/evidence-index.md"] == "/evidence-index"
    assert manifest["validation/rootfinding-performance.md"] == (
        "/rootfinding-performance"
    )
    assert manifest["validation/implicit-root-gradients.md"] == (
        "/implicit-root-gradients"
    )
    assert manifest["validation/spectra-performance.md"] == "/spectra-performance"
    assert len(manifest.values()) == len(set(manifest.values()))
