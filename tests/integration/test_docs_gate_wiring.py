"""Repository wiring contracts for the reusable documentation gate."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_docs_cli_is_lockfile_backed() -> None:
    package = (REPO_ROOT / "package.json").read_text(encoding="utf-8")
    docs_gate = (REPO_ROOT / "scripts/check_docs.sh").read_text(encoding="utf-8")
    assert '"mystmd": "1.10.1"' in package
    assert (REPO_ROOT / "package-lock.json").is_file()
    assert "npx --no-install myst build --html --ci --strict" in docs_gate
    assert "exec npx --no-install myst start" in docs_gate
    assert "\n  myst build" not in docs_gate
    assert "\n  exec myst start" not in docs_gate

    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "node_modules/" in gitignore


def test_docs_gate_is_reused_by_local_and_full_ci_gates() -> None:
    script = REPO_ROOT / "scripts" / "check_docs.sh"
    assert script.is_file()
    script_text = script.read_text(encoding="utf-8")
    assert "npx --no-install myst build --html --ci --strict" in script_text
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

    build_position = script_text.index(
        "npx --no-install myst build --html --ci --strict"
    )
    start_position = script_text.index("exec npx --no-install myst start")
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
    assert "bash scripts/check_docs.sh" in local_gate
    assert "npm ci --ignore-scripts" in local_gate
    assert local_gate.index("npm ci --ignore-scripts") < local_gate.index(
        "bash scripts/check_docs.sh"
    )
    assert "release-mirror:" in full_gate
    assert "Run the exact local release mirror" in full_gate
    assert "run: bash scripts/check.sh" in full_gate
    assert "scientific-validation:" in full_gate
    assert "pytest tests/validation -q" in full_gate

    pages = (REPO_ROOT / ".github" / "workflows" / "pages.yml").read_text(
        encoding="utf-8"
    )
    assert pages.index("run: bash scripts/check_docs.sh") < pages.index(
        "uses: actions/upload-pages-artifact@v5"
    )
    assert "path: docs/_build/html" in pages
    assert '- "scripts/inject_docs_accessibility.py"' in pages
    assert "npm ci --ignore-scripts" in pages
    assert "npm install --global" not in pages


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
    assert manifest["70-project/release/support.md"] == "/support"
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
    assert manifest["60-validation/numerical/quadrature-performance.md"] == (
        "/quadrature-performance"
    )
    assert manifest["60-validation/data/spectra-performance.md"] == (
        "/spectra-performance"
    )
    assert len(manifest.values()) == len(set(manifest.values()))


def test_quadrature_comparison_claims_are_routed_and_calibrated() -> None:
    performance_payload = json.loads(
        (REPO_ROOT / "docs/validation/quad-performance.json").read_text(
            encoding="utf-8"
        )
    )
    api_text = (REPO_ROOT / "docs/50-api/approximation-integration/quad.md").read_text(
        encoding="utf-8"
    )
    adaptive_text = (
        REPO_ROOT / "docs/20-methods/approximation-integration/adaptive-quadrature.md"
    ).read_text(encoding="utf-8")
    validation_text = (REPO_ROOT / "docs/60-validation/validation.md").read_text(
        encoding="utf-8"
    )
    combined_public_text = "\n".join((api_text, adaptive_text, validation_text))

    first = performance_payload["method_payload"]["baseline"]["records"][0]
    assert "comparison_label" in first
    normalized_api_text = api_text.casefold()
    for phrase in (
        "exact",
        "strong-match",
        "node-matched",
        "family-matched",
        "capability comparison",
        "shipped and validated",
        "benchmarking",
        "alpha",
        "approved but planned",
        "intentionally unsupported",
        "Migrating to `jaxstro.quad`",
        "jaxstro.numerics.integration",
    ):
        assert phrase.casefold() in normalized_api_text
    assert (
        "[quadrature performance and comparison]"
        "(../../60-validation/numerical/quadrature-performance.md)" in adaptive_text
    )
    assert "quad.performance" in validation_text
    for unsupported_claim in (
        "Jaxstro is universally fastest",
        "Jaxstro is universally best",
        "Jaxstro is universally SOTA",
    ):
        assert unsupported_claim not in combined_public_text
    assert "alias-protection floor" in adaptive_text
    assert "17 logical evaluations" in adaptive_text
    assert "alias-protection floor" in api_text
