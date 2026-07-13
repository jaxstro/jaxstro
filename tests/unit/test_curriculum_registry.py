"""Fail-closed registry and freshness contracts for executable curriculum units."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from examples.investigations._common import (
    AuditCheck,
    InvestigationMetric,
    InvestigationResult,
    validate_result,
)
from scripts.build_curriculum_registry import (
    load_and_validate_units,
    render_outputs,
    validate_instructor_route,
)

ROOT = Path(__file__).resolve().parents[2]


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


def test_curriculum_manifest_resolves_contracts_evidence_and_files() -> None:
    units = load_and_validate_units(ROOT)
    assert {unit["id"] for unit in units} == {
        "root-values-and-sensitivities",
        "powerlaw-removable-limit",
        "interpolation-boundary-policies",
    }
    assert all(unit["contract_ids"] for unit in units)
    assert all("evidence_ids" in unit for unit in units)
    assert all(unit["validation_targets"] for unit in units)
    root = next(unit for unit in units if unit["id"] == "root-values-and-sensitivities")
    assert root["evidence_ids"] == [
        "rootfinding.implicit-gradients",
        "rootfinding.performance",
    ]
    assert all((ROOT / unit["page"]).is_file() for unit in units)
    assert all((ROOT / unit["example"]).is_file() for unit in units)


def test_curriculum_registry_outputs_are_deterministic_and_fresh() -> None:
    outputs = render_outputs(ROOT)
    assert all(
        path.read_text(encoding="utf-8") == content for path, content in outputs.items()
    )
    coverage = json.loads(
        (ROOT / "docs/validation/curriculum-coverage.json").read_text(encoding="utf-8")
    )
    assert coverage["schema_version"] == "1"
    assert coverage["unit_count"] == 3
    subprocess.run(
        [sys.executable, "scripts/build_curriculum_registry.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_instructor_route_must_resolve_to_a_repository_page() -> None:
    with pytest.raises(ValueError, match="instructor route does not exist"):
        validate_instructor_route(ROOT, "docs/80-instructor/missing.md")
