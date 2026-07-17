"""Contracts for local release and GitHub Pages preparation."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_UV_V8_SHA = "08807647e7069bb48b6ef5acd8ec9567f424441b"


def test_pages_workflow_uses_the_verified_docs_gate_and_site_output() -> None:
    workflow_path = REPO_ROOT / ".github" / "workflows" / "pages.yml"
    assert workflow_path.is_file()
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "BASE_URL: /${{ github.event.repository.name }}" in workflow
    assert "contents: read" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-node@v6" in workflow
    assert 'node-version: "24"' in workflow
    assert "package-manager-cache: false" in workflow
    assert f"astral-sh/setup-uv@{SETUP_UV_V8_SHA}" in workflow
    assert 'python-version: "3.13"' in workflow
    assert "uv sync --locked --extra dev" in workflow
    assert "mystmd@1.10.1" in workflow
    assert "bash scripts/check_docs.sh" in workflow
    assert "actions/upload-pages-artifact@v5" in workflow
    assert "path: docs/_build/html" in workflow
    assert "actions/deploy-pages@v5" in workflow
    assert "environment:" in workflow
    assert "name: github-pages" in workflow

    docs_gate = (REPO_ROOT / "scripts" / "check_docs.sh").read_text(encoding="utf-8")
    assert 'BASE_PATH="${BASE_URL:-}"' in docs_gate
    assert '--base-path "$BASE_PATH"' in docs_gate
    assert "docs/_build/html/index.html" in docs_gate


def test_active_workflows_use_node24_action_releases() -> None:
    workflows = tuple((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows

    deprecated = (
        "actions/checkout@v4",
        "actions/setup-node@v4",
        "actions/upload-pages-artifact@v3",
        "actions/deploy-pages@v4",
        "astral-sh/setup-uv@v6",
    )
    for workflow_path in workflows:
        workflow = workflow_path.read_text(encoding="utf-8")
        for action in deprecated:
            assert action not in workflow, f"{workflow_path.name}: {action}"

        if "actions/setup-node@" in workflow:
            assert "actions/setup-node@v6" in workflow, workflow_path.name
            assert 'node-version: "24"' in workflow, workflow_path.name
            assert "package-manager-cache: false" in workflow, workflow_path.name

        if "astral-sh/setup-uv@" in workflow:
            assert f"astral-sh/setup-uv@{SETUP_UV_V8_SHA}" in workflow, (
                workflow_path.name
            )


def test_exhaustive_test_gates_install_benchmark_only_dependencies() -> None:
    """Fresh exhaustive gates must collect benchmark tests without runtime deps."""
    local_gate = (REPO_ROOT / "scripts" / "check.sh").read_text(encoding="utf-8")
    workflow = (REPO_ROOT / ".github" / "workflows" / "full-gate.yml").read_text(
        encoding="utf-8"
    )
    test_matrix = workflow.split("  test-matrix:", maxsplit=1)[1].split(
        "  full-validation:", maxsplit=1
    )[0]

    sync = "uv sync --locked --extra dev --group benchmark"
    assert sync in local_gate
    assert local_gate.index(sync) < local_gate.index('pytest -m "not slow"')
    assert sync in test_matrix
    assert test_matrix.index(sync) < test_matrix.index('pytest -m "not slow"')


def test_release_checklist_preserves_irreversible_stop_gates() -> None:
    checklist_path = REPO_ROOT / "docs" / "70-project" / "release" / "checklist.md"
    assert checklist_path.is_file()
    checklist = checklist_path.read_text(encoding="utf-8")

    required = (
        "jaxstro vs jaxstro-core",
        "explicit authorization",
        "progenax",
        "PyPI",
        "Zenodo",
        "CITATION.cff",
        "CONTRIBUTING.md",
        "sdist",
        "GitHub Actions",
        "bash scripts/check.sh",
        "bash scripts/check_docs.sh",
    )
    for phrase in required:
        assert phrase in checklist


def test_release_metadata_is_public_and_navigation_includes_checklist() -> None:
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    myst = (REPO_ROOT / "docs" / "myst.yml").read_text(encoding="utf-8")

    assert "cff-version: 1.2.0" in citation
    assert "repository-code: https://github.com/jaxstro/jaxstro" in citation
    assert 'license: "Apache-2.0"' in citation
    assert "bash scripts/check.sh" in contributing
    assert "bash scripts/check_docs.sh" in contributing
    assert "file: 70-project/release/checklist.md" in myst

    release_index = (
        REPO_ROOT / "docs" / "70-project" / "release" / "release.md"
    ).read_text(encoding="utf-8")
    assert "[](./checklist.md)" in release_index
    assert "This section will hold" not in release_index


def test_sdist_excludes_internal_and_nonruntime_workspaces() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    excluded = set(pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"])

    assert {
        "/.github",
        "/AGENTS.md",
        "/CLAUDE.md",
        "/STATUS.md",
        "/docs/audits",
        "/docs/plans",
        "/docs/superpowers",
        "/laboratory",
        "/tests",
    } <= excluded
