"""Unit contracts for the shared scientific-evidence envelope."""

import dataclasses
import math

import pytest

from jaxstro.evidence import (
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
