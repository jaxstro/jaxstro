"""Deterministic rendering and freshness contracts for evidence artifacts."""

import dataclasses
from pathlib import Path

import pytest

from jaxstro.evidence import (
    ComparisonRecord,
    ComparisonRelation,
    EvidenceArtifact,
    EvidenceStatus,
    MetricRecord,
)
from jaxstro.evidence.files import EvidenceFreshnessError, check_artifact
from jaxstro.evidence.render import (
    artifact_from_dict,
    artifact_to_dict,
    artifact_to_json,
    artifact_to_markdown,
)


@pytest.fixture
def valid_artifact() -> EvidenceArtifact:
    artifact = EvidenceArtifact.fixture(
        "fixture",
        metrics=(
            MetricRecord(
                "root.residual",
                "abs(f(x_star))",
                1.0e-14,
                "function units",
                EvidenceStatus.PASS,
            ),
        ),
    )
    comparison = ComparisonRecord(
        "residual-limit",
        "root.residual",
        ComparisonRelation.LESS_EQUAL,
        1.0e-12,
        "function units",
        EvidenceStatus.PASS,
        atol=1.0e-15,
        rtol=1.0e-6,
        note="Declared residual gate.",
    )
    return dataclasses.replace(artifact, comparisons=(comparison,))


def test_json_and_markdown_are_deterministic(valid_artifact) -> None:
    assert artifact_to_json(valid_artifact) == artifact_to_json(valid_artifact)
    markdown = artifact_to_markdown(valid_artifact)
    assert "| Metric identity | Symbol | Value | Units | Status |" in markdown
    assert "Absolute tolerance" in markdown
    assert "1e-15" in markdown
    assert "1e-06" in markdown
    assert "Declared residual gate." in markdown
    assert "/Users/" not in artifact_to_json(valid_artifact)
    restored = artifact_from_dict(artifact_to_dict(valid_artifact))
    assert artifact_to_json(restored) == artifact_to_json(valid_artifact)


def test_check_artifact_rejects_stale_bytes(
    tmp_path: Path, valid_artifact: EvidenceArtifact
) -> None:
    path = tmp_path / "evidence.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvidenceFreshnessError, match="stale"):
        check_artifact(path, valid_artifact)
