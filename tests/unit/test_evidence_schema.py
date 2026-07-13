"""Unit contracts for the shared scientific-evidence envelope."""

import dataclasses
import math

import pytest

from jaxstro.evidence import (
    ComparisonRecord,
    ComparisonRelation,
    EvidenceArtifact,
    EvidenceStatus,
    MetricRecord,
    validate_artifact,
)


def test_metric_requires_identity_symbol_value_and_units() -> None:
    metric = MetricRecord(
        "root.residual",
        "abs(f(x_star))",
        1.0e-14,
        "function units",
        EvidenceStatus.PASS,
    )
    artifact = EvidenceArtifact.fixture("rootfinding.performance", metrics=(metric,))
    validate_artifact(artifact)
    with pytest.raises(dataclasses.FrozenInstanceError):
        metric.units = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("units", ["", "unitless", None])
def test_metric_rejects_missing_or_ambiguous_units(units) -> None:
    artifact = EvidenceArtifact.fixture(
        "bad",
        metrics=(MetricRecord("m", "m", 1.0, units, EvidenceStatus.INFO),),
    )
    with pytest.raises(ValueError, match="units"):
        validate_artifact(artifact)


@pytest.mark.parametrize("value", [True, "1.0", math.nan, math.inf])
def test_metric_rejects_nonportable_numeric_values(value) -> None:
    artifact = EvidenceArtifact.fixture(
        "bad-number",
        metrics=(MetricRecord("m", "m", value, "dimensionless"),),
    )
    with pytest.raises((TypeError, ValueError), match="numeric|finite"):
        validate_artifact(artifact)


def test_artifact_recursively_freezes_payload() -> None:
    source = {"nested": {"values": [1.0, 2.0]}}
    artifact = EvidenceArtifact.fixture("frozen", method_payload=source)
    source["nested"]["values"].append(3.0)
    assert artifact.method_payload["nested"]["values"] == (1.0, 2.0)
    with pytest.raises(TypeError):
        artifact.method_payload["new"] = 1


def test_artifact_rejects_unknown_schema_version() -> None:
    artifact = dataclasses.replace(
        EvidenceArtifact.fixture("future"), schema_version="999"
    )
    with pytest.raises(ValueError, match="unsupported evidence schema"):
        validate_artifact(artifact)


def test_comparison_units_must_match_metric_units() -> None:
    artifact = EvidenceArtifact.fixture(
        "units",
        metrics=(MetricRecord("elapsed", "t", 1.0, "s"),),
    )
    comparison = ComparisonRecord(
        "elapsed.gate",
        "elapsed",
        ComparisonRelation.LESS_EQUAL,
        2.0,
        "bytes",
        EvidenceStatus.PASS,
    )
    with pytest.raises(ValueError, match="comparison units do not match"):
        validate_artifact(dataclasses.replace(artifact, comparisons=(comparison,)))


def test_inequality_comparison_applies_relative_tolerance() -> None:
    artifact = EvidenceArtifact.fixture(
        "relative-tolerance",
        metrics=(MetricRecord("cost", "C", 10.5, "evaluations"),),
    )
    comparison = ComparisonRecord(
        "cost.gate",
        "cost",
        ComparisonRelation.LESS_EQUAL,
        10.0,
        "evaluations",
        EvidenceStatus.PASS,
        rtol=0.1,
    )
    validate_artifact(dataclasses.replace(artifact, comparisons=(comparison,)))
