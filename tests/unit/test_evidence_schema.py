"""Unit contracts for the shared scientific-evidence envelope."""

import dataclasses

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
