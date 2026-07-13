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


@pytest.mark.parametrize("missing", ["metrics", "comparisons", "method_payload"])
def test_artifact_parser_requires_complete_top_level_schema(
    valid_artifact: EvidenceArtifact, missing: str
) -> None:
    payload = artifact_to_dict(valid_artifact)
    del payload[missing]
    with pytest.raises(ValueError, match="evidence artifact fields"):
        artifact_from_dict(payload)


def test_artifact_parser_rejects_unknown_fields(
    valid_artifact: EvidenceArtifact,
) -> None:
    payload = artifact_to_dict(valid_artifact)
    payload["metricz"] = []
    with pytest.raises(ValueError, match="evidence artifact fields"):
        artifact_from_dict(payload)


def test_artifact_parser_rejects_unknown_nested_fields(
    valid_artifact: EvidenceArtifact,
) -> None:
    payload = artifact_to_dict(valid_artifact)
    payload["metrics"][0]["extra"] = "typo"  # type: ignore[index]
    with pytest.raises(ValueError, match="metric fields"):
        artifact_from_dict(payload)


def test_markdown_escapes_table_cell_pipes_and_newlines() -> None:
    artifact = EvidenceArtifact.fixture(
        "table-cells",
        metrics=(MetricRecord("m|x", "a\nb", 1.0, "function|units"),),
    )
    comparison = ComparisonRecord(
        "gate|one",
        "m|x",
        ComparisonRelation.EQUAL,
        1.0,
        "function|units",
        EvidenceStatus.PASS,
        note="line|one\nline two",
    )
    markdown = artifact_to_markdown(
        dataclasses.replace(artifact, comparisons=(comparison,))
    )
    assert "m\\|x" in markdown
    assert "a b" in markdown
    assert "function\\|units" in markdown
    assert "line\\|one line two" in markdown
