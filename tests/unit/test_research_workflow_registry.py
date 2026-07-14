"""Fail-closed registry and freshness contracts for research workflows."""

from __future__ import annotations

import importlib
import inspect
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from examples.investigations._common import (
    AuditCheck,
    InvestigationMetric,
    InvestigationResult,
    calibrated_claim,
    investigation_report,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_research_workflow_registry.py"
MANIFEST = ROOT / "docs/40-workflows/investigations/registry.json"
COVERAGE = ROOT / "docs/validation/research-workflow-coverage.json"
WORKFLOW_FIELDS = {
    "id",
    "title",
    "page",
    "example",
    "prerequisites",
    "public_apis",
    "contract_ids",
    "evidence_ids",
    "validation_targets",
    "limitations",
}


def _registry_module() -> ModuleType:
    assert SCRIPT.is_file(), "the research-workflow registry script must exist"
    return importlib.import_module("scripts.build_research_workflow_registry")


def _copy_registry_fixture(destination: Path) -> None:
    for relative in (
        "docs/40-workflows/investigations",
        "docs/validation/contracts.json",
        "docs/validation/evidence-index.json",
        "examples/investigations",
        "tests/unit/test_distributions.py",
        "tests/unit/test_regular_grid.py",
        "tests/validation/test_bracketed_root_algorithms.py",
        "tests/validation/test_grad_checks.py",
        "tests/validation/test_implicit_root_gradients.py",
        "docs/10-foundations/mathematical-objects",
        "docs/10-foundations/models-and-computation",
    ):
        source = ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def _mutate_manifest(root: Path, mutate: Callable[[dict], None]) -> None:
    path = root / "docs/40-workflows/investigations/registry.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_investigation_result_requires_units_and_passing_audits() -> None:
    result = InvestigationResult(
        "fixture",
        "predict a finite value",
        (InvestigationMetric("fixture.value", "x", 1.0, "dimensionless"),),
        (AuditCheck("fixture.finite", True, "analytic identity"),),
        "The fixture supports only its stated identity.",
    )
    validate_result(result)
    with pytest.raises(ValueError, match="units"):
        validate_result(
            InvestigationResult(
                "bad",
                "prediction",
                (InvestigationMetric("bad.value", "x", 1.0, ""),),
                (AuditCheck("bad.check", True, "identity"),),
                "claim",
            )
        )


def test_failed_audit_is_visible_and_blocks_positive_claim() -> None:
    checks = (AuditCheck("fixture.failed", False, "independent identity"),)
    claim = calibrated_claim(checks, "This positive claim must not survive.")
    result = InvestigationResult(
        "failed-fixture",
        "predict a passing identity",
        (InvestigationMetric("fixture.value", "x", 2.0, "dimensionless"),),
        checks,
        claim,
    )
    report = investigation_report(result)
    assert "| fixture.failed | fail | independent identity |" in report
    assert "No positive claim is warranted" in report
    assert "This positive claim must not survive" not in report


def test_manifest_uses_schema_two_and_exact_workflow_fields() -> None:
    assert MANIFEST.is_file(), "the route-first workflow manifest must exist"
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(payload) == {"schema_version", "workflows"}
    assert payload["schema_version"] == "2"
    assert payload["workflows"]
    assert all(set(workflow) == WORKFLOW_FIELDS for workflow in payload["workflows"])


def test_registry_public_functions_have_exact_signatures() -> None:
    registry_module = _registry_module()
    expected = {
        "load_and_validate_workflows": "(root: 'Path') -> 'list[dict[str, Any]]'",
        "validate_unique_references": (
            "(identity: 'str', workflow: 'dict[str, Any]', fields: 'Sequence[str]') "
            "-> 'None'"
        ),
        "render_outputs": "(root: 'Path' = PosixPath('"
        + str(ROOT)
        + "')) -> 'dict[Path, str]'",
    }
    for name, signature in expected.items():
        assert str(inspect.signature(getattr(registry_module, name))) == signature
    assert not hasattr(registry_module, "load_and_validate_" + "units")
    assert not hasattr(registry_module, "validate_" + "instructor_route")

    source = SCRIPT.read_text(encoding="utf-8")
    assert "from collections.abc import Sequence" in source


def test_workflow_manifest_resolves_contracts_evidence_and_files() -> None:
    registry_module = _registry_module()
    workflows = registry_module.load_and_validate_workflows(ROOT)
    assert {workflow["id"] for workflow in workflows} == {
        "root-values-and-sensitivities",
        "powerlaw-removable-limit",
        "interpolation-boundary-policies",
    }
    assert all(workflow["contract_ids"] for workflow in workflows)
    assert all("evidence_ids" in workflow for workflow in workflows)
    assert all(workflow["validation_targets"] for workflow in workflows)
    root = next(
        workflow
        for workflow in workflows
        if workflow["id"] == "root-values-and-sensitivities"
    )
    assert root["evidence_ids"] == [
        "rootfinding.implicit-gradients",
        "rootfinding.performance",
    ]
    assert all((ROOT / workflow["page"]).is_file() for workflow in workflows)
    assert all((ROOT / workflow["example"]).is_file() for workflow in workflows)


def test_registry_outputs_are_deterministic_and_fresh() -> None:
    registry_module = _registry_module()
    first = registry_module.render_outputs(ROOT)
    second = registry_module.render_outputs(ROOT)
    assert first == second
    assert all(
        path.read_text(encoding="utf-8") == content for path, content in first.items()
    )
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    assert set(coverage) == {"schema_version", "workflow_count", "workflows"}
    assert coverage["schema_version"] == "2"
    assert coverage["workflow_count"] == 3
    completed = subprocess.run(
        [sys.executable, "scripts/build_research_workflow_registry.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.strip() == "research workflow registry fresh"


def test_duplicate_references_are_rejected() -> None:
    registry_module = _registry_module()
    with pytest.raises(ValueError, match="duplicate research workflow contract_ids"):
        registry_module.validate_unique_references(
            "duplicate-workflow",
            {"contract_ids": ["numerics.root", "numerics.root"]},
            ("contract_ids",),
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["workflows"][0].update(
                page="docs/40-workflows/investigations/missing.md"
            ),
            "research workflow page does not exist",
        ),
        (
            lambda payload: payload["workflows"][0].update(
                public_apis=["jaxstro.not_a_public_api"]
            ),
            "research workflow API and contract paths disagree",
        ),
        (
            lambda payload: payload["workflows"][0].update(
                evidence_ids=["missing.evidence"]
            ),
            "unknown research workflow evidence ids",
        ),
    ],
)
def test_registry_fails_closed_on_reference_drift(
    tmp_path: Path,
    mutate: Callable[[dict], None],
    message: str,
) -> None:
    registry_module = _registry_module()
    _copy_registry_fixture(tmp_path)
    _mutate_manifest(tmp_path, mutate)
    with pytest.raises(ValueError, match=message):
        registry_module.load_and_validate_workflows(tmp_path)
