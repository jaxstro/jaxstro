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
    injector_calls = list(
        re.finditer(
            r'"\$ROOT_DIR/scripts/inject_docs_accessibility\.py" \\\n'
            r'  "\$ROOT_DIR/docs/_build/html"(?P<check> --check)?',
            script_text,
        )
    )
    assert len(injector_calls) == 2
    inject_call, verify_call = injector_calls
    assert inject_call.group("check") is None
    assert verify_call.group("check") == " --check"

    build_position = script_text.index("myst build --html --ci --strict")
    start_position = script_text.index("exec myst start")
    audit_position = script_text.index('"$ROOT_DIR/scripts/check_docs_site.py"')
    stop_position = script_text.rindex("\nstop_docs_server\n")
    success_position = script_text.index('echo "ALL DOCS GATES PASSED"')

    assert (
        build_position
        < start_position
        < audit_position
        < stop_position
        < inject_call.start()
        < verify_call.start()
        < success_position
    )
    final_artifact_lane = script_text[inject_call.start() : success_position]
    assert "myst build" not in final_artifact_lane
    assert "myst start" not in final_artifact_lane

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

    pages = (REPO_ROOT / ".github" / "workflows" / "pages.yml").read_text(
        encoding="utf-8"
    )
    assert pages.index("run: bash scripts/check_docs.sh") < pages.index(
        "uses: actions/upload-pages-artifact@v5"
    )
    assert "path: docs/_build/html" in pages
    assert '- "scripts/inject_docs_accessibility.py"' in pages


def test_committed_route_manifest_matches_authored_navigation_routes() -> None:
    manifest_path = REPO_ROOT / "docs" / "route-manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    myst = (REPO_ROOT / "docs" / "myst.yml").read_text(encoding="utf-8")
    authored = set(re.findall(r"^\s*- file:\s+(.+\.md)\s*$", myst, re.MULTILINE))
    assert set(manifest) == authored
    assert manifest["index.md"] == "/"
    assert manifest[
        "30-representations/spectra-atmospheres/spectra-data-architecture.md"
    ] == ("/spectra-data-architecture")
    assert manifest["60-validation/validation.md"] == "/validation"
    assert manifest["70-project/project.md"] == "/project"
    assert manifest["70-project/release/checklist.md"] == "/checklist"
    assert manifest["70-project/bibliography/bibliography.md"] == "/bibliography"
    assert manifest["60-validation/evidence-index.md"] == "/evidence-index"
    assert manifest["60-validation/numerical/rootfinding-performance.md"] == (
        "/rootfinding-performance"
    )
    assert manifest["60-validation/numerical/implicit-root-gradients.md"] == (
        "/implicit-root-gradients"
    )
    assert manifest["60-validation/numerical/quadrature-replay-derivatives.md"] == (
        "/quadrature-replay-derivatives"
    )
    assert manifest["60-validation/data/spectra-performance.md"] == (
        "/spectra-performance"
    )
    assert len(manifest.values()) == len(set(manifest.values()))
