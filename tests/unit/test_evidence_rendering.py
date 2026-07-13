"""Deterministic rendering and freshness contracts for evidence artifacts."""

from pathlib import Path

import pytest

from jaxstro.evidence import EvidenceArtifact, EvidenceStatus, MetricRecord
from jaxstro.evidence.files import EvidenceFreshnessError, check_artifact
from jaxstro.evidence.render import artifact_to_json, artifact_to_markdown


@pytest.fixture
def valid_artifact() -> EvidenceArtifact:
    return EvidenceArtifact.fixture(
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


def test_json_and_markdown_are_deterministic(valid_artifact) -> None:
    assert artifact_to_json(valid_artifact) == artifact_to_json(valid_artifact)
    markdown = artifact_to_markdown(valid_artifact)
    assert "| Metric identity | Symbol | Value | Units | Status |" in markdown
    assert "/Users/" not in artifact_to_json(valid_artifact)


def test_check_artifact_rejects_stale_bytes(
    tmp_path: Path, valid_artifact: EvidenceArtifact
) -> None:
    path = tmp_path / "evidence.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvidenceFreshnessError, match="stale"):
        check_artifact(path, valid_artifact)
