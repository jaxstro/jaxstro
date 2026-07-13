"""Fail-closed semantic validation for scientific evidence artifacts."""

from __future__ import annotations

import math

from .schema import (
    ComparisonRelation,
    EvidenceArtifact,
    EvidenceStatus,
)


def validate_artifact(artifact: EvidenceArtifact) -> None:
    """Validate identities, units, finiteness, and comparison truth."""
    for name, value in (
        ("schema version", artifact.schema_version),
        ("artifact id", artifact.artifact_id),
        ("artifact version", artifact.artifact_version),
        ("package version", artifact.package_version),
        ("source revision", artifact.source_revision),
        ("generation command", artifact.generation_command),
        ("precision", artifact.precision),
        ("environment policy", artifact.environment.policy),
    ):
        _require_text(value, name)
    metrics = {}
    for metric in artifact.metrics:
        _require_text(metric.identity, "metric identity")
        if metric.identity in metrics:
            raise ValueError(f"duplicate metric identity: {metric.identity}")
        _require_text(metric.symbol, f"{metric.identity} symbol")
        if not isinstance(metric.units, str) or not metric.units.strip():
            raise ValueError(f"{metric.identity} units must be explicit")
        if metric.units == "unitless":
            raise ValueError("units must use 'dimensionless', not 'unitless'")
        if not isinstance(metric.status, EvidenceStatus):
            raise ValueError(f"unknown evidence status: {metric.status!r}")
        _require_finite_number(metric.value, f"{metric.identity} numeric value")
        if not math.isfinite(metric.value):
            raise ValueError(f"nonfinite metric value: {metric.identity}")
        metrics[metric.identity] = metric
    seen_comparisons: set[str] = set()
    for comparison in artifact.comparisons:
        _require_text(comparison.identity, "comparison identity")
        if comparison.identity in seen_comparisons:
            raise ValueError(f"duplicate comparison identity: {comparison.identity}")
        seen_comparisons.add(comparison.identity)
        if comparison.metric_id not in metrics:
            raise ValueError(f"unknown comparison metric: {comparison.metric_id}")
        _require_text(comparison.units, f"{comparison.identity} units")
        _require_finite_number(comparison.reference, "comparison reference")
        _require_finite_number(comparison.atol, "absolute tolerance")
        _require_finite_number(comparison.rtol, "relative tolerance")
        if comparison.atol < 0.0 or comparison.rtol < 0.0:
            raise ValueError("comparison tolerances must be nonnegative")
        _validate_comparison(metrics[comparison.metric_id].value, comparison)


def _validate_comparison(value: int | float, comparison) -> None:
    if not isinstance(comparison.relation, ComparisonRelation):
        raise ValueError(f"unknown comparison relation: {comparison.relation!r}")
    if not isinstance(comparison.status, EvidenceStatus):
        raise ValueError(f"unknown comparison status: {comparison.status!r}")
    if comparison.relation is ComparisonRelation.INFORMATIONAL:
        passed = None
    elif comparison.relation is ComparisonRelation.LESS_EQUAL:
        passed = value <= comparison.reference + comparison.atol
    elif comparison.relation is ComparisonRelation.GREATER_EQUAL:
        passed = value >= comparison.reference - comparison.atol
    elif comparison.relation is ComparisonRelation.CLOSE:
        passed = math.isclose(
            value,
            comparison.reference,
            abs_tol=comparison.atol,
            rel_tol=comparison.rtol,
        )
    else:
        passed = value == comparison.reference
    expected = (
        EvidenceStatus.INFO
        if passed is None
        else (EvidenceStatus.PASS if passed else EvidenceStatus.FAIL)
    )
    if comparison.status is not expected:
        raise ValueError(
            f"comparison status disagrees with declared relation: {comparison.identity}"
        )


def _require_text(value: object, identity: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{identity} must be nonempty text")


def _require_finite_number(value: object, identity: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{identity} must be an int or float numeric value")
    if not math.isfinite(value):
        raise ValueError(f"{identity} must be finite")
