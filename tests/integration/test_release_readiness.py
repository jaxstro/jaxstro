"""Contracts for local release and GitHub Pages preparation."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_UV_V8_SHA = "08807647e7069bb48b6ef5acd8ec9567f424441b"


def _steps_run(job: dict) -> list[str]:
    return [step["run"] for step in job["steps"] if "run" in step]


def _check_sh_stages() -> list[str]:
    script = (REPO_ROOT / "scripts" / "check.sh").read_text(encoding="utf-8")
    match = re.search(r"^STAGES=\(([^)]*)\)$", script, flags=re.MULTILINE)
    assert match, "scripts/check.sh must declare STAGES=(...)"
    return match.group(1).split()


def _workflow(name: str) -> dict:
    return yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
    )


def test_full_gate_runs_every_local_stage_as_a_parallel_job() -> None:
    """Owner of the full-gate workflow structure (parsed, not text-matched)."""
    workflow = _workflow("full-gate.yml")
    triggers = workflow[True]  # YAML 1.1 reads the key `on` as True
    assert triggers["push"]["branches"] == ["main"]
    assert workflow["env"]["JAX_ENABLE_X64"] == "1"
    assert set(workflow["jobs"]) == {"stage", "full-gate", "scientific-validation"}

    stage = workflow["jobs"]["stage"]
    assert stage["strategy"]["fail-fast"] is False
    assert stage["strategy"]["matrix"]["stage"] == _check_sh_stages()
    # Earlier steps only record the runner; the gate work is check.sh alone.
    assert _steps_run(stage)[-1] == "bash scripts/check.sh ${{ matrix.stage }}"
    assert sum("check.sh" in run for run in _steps_run(stage)) == 1
    assert stage["timeout-minutes"] <= 30

    summary = workflow["jobs"]["full-gate"]
    assert summary["needs"] == ["stage"]
    assert summary["if"] == "always()"

    validation = workflow["jobs"]["scientific-validation"]
    assert validation["if"] == "github.event_name != 'push'"
    assert "uv run --no-sync pytest tests/validation -q" in _steps_run(validation)


def test_check_sh_test_stages_cover_every_test_tier() -> None:
    tiers = {
        path.name
        for path in (REPO_ROOT / "tests").iterdir()
        if path.is_dir() and any(path.rglob("test_*.py"))
    }
    stages = _check_sh_stages()
    assert {f"tests-{tier}" for tier in tiers} <= set(stages)


def test_fast_gate_runs_on_pushes_to_main_and_on_pull_requests() -> None:
    triggers = _workflow("tests.yml")[True]
    assert triggers["push"]["branches"] == ["main"]
    assert "pull_request" in triggers


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
    assert "node-version-file: .nvmrc" in workflow
    assert "package-manager-cache: false" in workflow
    assert f"astral-sh/setup-uv@{SETUP_UV_V8_SHA}" in workflow
    assert 'python-version: "3.13"' in workflow
    assert "uv sync --locked --extra dev" in workflow
    assert "npm ci --ignore-scripts" in workflow
    assert "npm install --global" not in workflow
    assert "bash scripts/check_docs.sh" in workflow
    assert "actions/upload-pages-artifact@v5" in workflow
    assert "path: docs/_build/html" in workflow
    assert "actions/deploy-pages@v5" in workflow
    assert "environment:" in workflow
    assert "name: github-pages" in workflow
    for path in ('"package.json"', '"package-lock.json"', '".nvmrc"'):
        assert path in workflow

    docs_gate = (REPO_ROOT / "scripts" / "check_docs.sh").read_text(encoding="utf-8")
    assert 'BASE_PATH="${BASE_URL:-}"' in docs_gate
    assert '--base-path "$BASE_PATH"' in docs_gate
    assert "docs/_build/html/index.html" in docs_gate


def test_node_is_pinned_once_in_nvmrc_and_matches_engines() -> None:
    nvmrc = (REPO_ROOT / ".nvmrc").read_text(encoding="utf-8").strip()
    package = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    assert package["engines"]["node"] == f"{nvmrc}.x"


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
            assert "node-version-file: .nvmrc" in workflow, workflow_path.name
            assert "node-version:" not in workflow, workflow_path.name
            assert "package-manager-cache: false" in workflow, workflow_path.name

        if "astral-sh/setup-uv@" in workflow:
            assert f"astral-sh/setup-uv@{SETUP_UV_V8_SHA}" in workflow, (
                workflow_path.name
            )


def test_release_mirror_keeps_benchmark_collection_in_the_local_gate() -> None:
    """The exact local mirror must prepare benchmark-only collection dependencies."""
    local_gate = (REPO_ROOT / "scripts" / "check.sh").read_text(encoding="utf-8")

    sync = "uv sync --locked --extra dev --group benchmark"
    assert sync in local_gate
    assert local_gate.index(sync) < local_gate.index('pytest -m "not slow"')


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
        "/docs/_build",
        "/laboratory",
        "/tests",
        "/.mypy_cache",
        "/.pytest_cache",
        "/**/__pycache__",
    } <= excluded


def test_release_gate_checks_wheel_and_sdist_in_clean_interpreters() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["build-system"]["requires"] == ["hatchling==1.31.0"]

    local_gate = (REPO_ROOT / "scripts" / "check.sh").read_text(encoding="utf-8")
    for phrase in (
        "uv build --python 3.13 -o",
        "scripts/check_distribution.py",
        "--wheel",
        "--sdist",
        "build-provenance.txt",
        "uv venv --python 3.13",
        "WHEEL_VENV",
        "SDIST_VENV",
    ):
        assert phrase in local_gate
