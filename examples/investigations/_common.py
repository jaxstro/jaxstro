"""Fixed curriculum result records shared by executable investigations."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class InvestigationMetric:
    """One units-explicit measured result."""

    identity: str
    symbol: str
    value: int | float
    units: str


@dataclass(frozen=True)
class AuditCheck:
    """One independent or invariant-based audit check."""

    identity: str
    passed: bool
    evidence: str


@dataclass(frozen=True)
class InvestigationResult:
    """Fixed evidence returned by every executable curriculum unit."""

    unit_id: str
    prediction: str
    metrics: tuple[InvestigationMetric, ...]
    audit_checks: tuple[AuditCheck, ...]
    warranted_claim: str


def validate_result(result: InvestigationResult) -> None:
    """Reject incomplete, duplicate, or nonportable curriculum evidence."""
    for identity, value in (
        ("unit id", result.unit_id),
        ("prediction", result.prediction),
        ("warranted claim", result.warranted_claim),
    ):
        if not value.strip():
            raise ValueError(f"{identity} must be nonempty")
    if not result.metrics or not result.audit_checks:
        raise ValueError("investigation requires metrics and audit checks")
    metric_ids: set[str] = set()
    for metric in result.metrics:
        if metric.identity in metric_ids:
            raise ValueError(f"duplicate metric identity: {metric.identity}")
        metric_ids.add(metric.identity)
        if not metric.symbol.strip():
            raise ValueError(f"metric symbol must be nonempty: {metric.identity}")
        if not metric.units.strip():
            raise ValueError(f"metric units must be explicit: {metric.identity}")
        if isinstance(metric.value, bool) or not isinstance(metric.value, (int, float)):
            raise TypeError(f"metric value must be numeric: {metric.identity}")
        if not math.isfinite(metric.value):
            raise ValueError(f"metric value must be finite: {metric.identity}")
    check_ids: set[str] = set()
    for check in result.audit_checks:
        if check.identity in check_ids:
            raise ValueError(f"duplicate audit identity: {check.identity}")
        check_ids.add(check.identity)
        if not check.evidence.strip():
            raise ValueError(f"audit evidence must be nonempty: {check.identity}")


def metric_table(result: InvestigationResult) -> str:
    """Render measured results in the repository's required table format."""
    validate_result(result)
    lines = [
        "| Metric identity | Symbol | Value | Units |",
        "| --- | --- | ---: | --- |",
    ]
    lines.extend(
        f"| {item.identity} | `{item.symbol}` | {item.value} | {item.units} |"
        for item in result.metrics
    )
    return "\n".join(lines) + "\n"


def calibrated_claim(checks: tuple[AuditCheck, ...], positive_claim: str) -> str:
    """Return a positive claim only when every declared audit passes."""
    failed = [check.identity for check in checks if not check.passed]
    if failed:
        return (
            "No positive claim is warranted because these audits failed: "
            + ", ".join(failed)
            + "."
        )
    return positive_claim


def investigation_report(result: InvestigationResult) -> str:
    """Render prediction, metrics, audits, and the calibrated claim."""
    validate_result(result)
    lines = [
        "## Prediction",
        "",
        result.prediction,
        "",
        "## Metrics",
        "",
        metric_table(result).rstrip(),
        "",
        "## Audit",
        "",
        "| Audit identity | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {check.identity} | {'pass' if check.passed else 'fail'} | {check.evidence} |"
        for check in result.audit_checks
    )
    lines.extend(["", "## State the warranted claim", "", result.warranted_claim, ""])
    return "\n".join(lines)
