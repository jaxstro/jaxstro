"""Deterministic evidence reports for numerical-method trust audits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

Status = Literal["validated", "partial", "deferred"]


@dataclass(frozen=True)
class EvidenceAnchor:
    """A single validation claim and the tests or docs that anchor it."""

    property: str
    measured: str
    tolerance: str
    anchors: tuple[str, ...]


@dataclass(frozen=True)
class MethodEvidence:
    """Evidence summary for one numerical method family."""

    method: str
    summary: str
    status: Status
    anchors: tuple[EvidenceAnchor, ...]
    known_limits: tuple[str, ...] = ()


@dataclass(frozen=True)
class NumericalTrustReport:
    """A deterministic report over numerical-method evidence."""

    title: str
    methods: tuple[MethodEvidence, ...]
    generated_by: str = "jaxstro.testing.reports.v1"


def _anchor_to_dict(anchor: EvidenceAnchor) -> dict[str, object]:
    return {
        "property": anchor.property,
        "measured": anchor.measured,
        "tolerance": anchor.tolerance,
        "anchors": list(anchor.anchors),
    }


def _method_to_dict(method: MethodEvidence) -> dict[str, object]:
    return {
        "method": method.method,
        "summary": method.summary,
        "status": method.status,
        "anchors": [
            _anchor_to_dict(anchor)
            for anchor in sorted(method.anchors, key=lambda item: item.property)
        ],
        "known_limits": list(method.known_limits),
    }


def trust_report_to_dict(report: NumericalTrustReport) -> dict[str, object]:
    """Convert a trust report to a deterministic plain-data dictionary."""
    return {
        "title": report.title,
        "generated_by": report.generated_by,
        "methods": [
            _method_to_dict(method)
            for method in sorted(report.methods, key=lambda item: item.method)
        ],
    }


def trust_report_to_json(report: NumericalTrustReport) -> str:
    """Serialize a trust report as deterministic pretty JSON."""
    return json.dumps(trust_report_to_dict(report), indent=2, sort_keys=True) + "\n"


def trust_report_to_markdown(report: NumericalTrustReport) -> str:
    """Render a trust report as deterministic Markdown."""
    lines = [f"# {report.title}", "", f"`generated_by`: `{report.generated_by}`", ""]
    for method in sorted(report.methods, key=lambda item: item.method):
        lines.extend(
            [
                f"## {method.method}",
                "",
                f"Status: `{method.status}`",
                "",
                method.summary,
                "",
                "| Property | Measured | Tolerance | Anchors |",
                "| --- | --- | --- | --- |",
            ]
        )
        for anchor in sorted(method.anchors, key=lambda item: item.property):
            anchor_text = ", ".join(f"`{item}`" for item in anchor.anchors)
            lines.append(
                "| "
                f"{anchor.property} | "
                f"{anchor.measured} | "
                f"{anchor.tolerance} | "
                f"{anchor_text} |"
            )
        if method.known_limits:
            lines.extend(["", "Known limits:"])
            lines.extend(f"- {item}" for item in method.known_limits)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def default_numerics_trust_report() -> NumericalTrustReport:
    """Return the built-in trust summary for current jaxstro numerics."""
    return NumericalTrustReport(
        title="jaxstro numerical methods trust report",
        methods=(
            MethodEvidence(
                method="splines",
                summary="B-spline basis, de Boor evaluation, calculus helpers, fixed-knot fits, penalties, adaptive knots, and tensor designs.",
                status="validated",
                anchors=(
                    EvidenceAnchor(
                        property="Local basis invariants and FD-vs-AD checks",
                        measured="partition, support, de Boor parity, derivatives, integrals, roughness penalties, quantile knots, tensor designs, least-squares recovery",
                        tolerance="exact identities plus test-specific tolerances",
                        anchors=(
                            "tests/unit/test_splines.py",
                            "tests/validation/test_grad_checks.py",
                        ),
                    ),
                ),
                known_limits=(
                    "Custom VJPs, smoothing-parameter selection, sparse tensor-product storage, and iterative knot optimization remain deferred.",
                ),
            ),
            MethodEvidence(
                method="interpolation",
                summary="Clamped linear interpolation plus shape-preserving cubic tables.",
                status="validated",
                anchors=(
                    EvidenceAnchor(
                        property="Monotone tables avoid overshoot and differentiate inside cells",
                        measured="synthetic Hermite/PCHIP identities",
                        tolerance="exact and FD-vs-AD checks",
                        anchors=(
                            "tests/unit/test_interpolation_shape_preserving.py",
                            "tests/validation/test_grad_checks.py",
                        ),
                    ),
                ),
            ),
            MethodEvidence(
                method="regular_grid",
                summary="Static-rank multilinear interpolation with explicit boundary policy.",
                status="validated",
                anchors=(
                    EvidenceAnchor(
                        property="Affine tables recover exactly and gradients match FD",
                        measured="bilinear, trilinear, and ND grid tests",
                        tolerance="exact synthetic identities",
                        anchors=(
                            "tests/unit/test_regular_grid.py",
                            "tests/validation/test_grad_checks.py",
                        ),
                    ),
                ),
            ),
            MethodEvidence(
                method="rootfinding",
                summary="Fixed-count bracketing/Newton solvers and monotone inverse tables.",
                status="validated",
                anchors=(
                    EvidenceAnchor(
                        property="Smooth Newton/inverse-table paths match FD where differentiable",
                        measured="root and inverse interpolation checks",
                        tolerance="test-specific FD-vs-AD tolerances",
                        anchors=(
                            "tests/unit/test_numerics.py",
                            "tests/validation/test_grad_checks.py",
                        ),
                    ),
                ),
                known_limits=(
                    "Bisection and bracket expansion are value-first for function parameters.",
                ),
            ),
            MethodEvidence(
                method="linear_algebra",
                summary="Small dense fits, solves, covariance/correlation, and jitter helpers.",
                status="validated",
                anchors=(
                    EvidenceAnchor(
                        property="Dense helper contracts hold away from rank/cutoff boundaries",
                        measured="synthetic solves, covariance guards, and PD jitter search",
                        tolerance="exact identities plus FD-vs-AD checks",
                        anchors=(
                            "tests/unit/test_linear_algebra.py",
                            "tests/validation/test_grad_checks.py",
                        ),
                    ),
                ),
            ),
            MethodEvidence(
                method="special",
                summary="CGS Planck kernels, log-weight normalization, and polynomial bases.",
                status="validated",
                anchors=(
                    EvidenceAnchor(
                        property="Special-function identities and gradients are validated",
                        measured="direct Planck formulas, limiting cases, recurrences",
                        tolerance="direct parity and FD-vs-AD checks",
                        anchors=(
                            "tests/unit/test_special.py",
                            "tests/validation/test_grad_checks.py",
                        ),
                    ),
                ),
                known_limits=(
                    "Spherical Bessel functions are deferred until a downstream contract exists.",
                ),
            ),
            MethodEvidence(
                method="grids_sampling",
                summary="Log/geometric grids, conservative rebinning, and stratified uniforms.",
                status="validated",
                anchors=(
                    EvidenceAnchor(
                        property="Grid construction and conservative totals are checked",
                        measured="grid identities, total preservation, one sample per stratum",
                        tolerance="exact synthetic checks plus FD-vs-AD rebin tests",
                        anchors=(
                            "tests/unit/test_grids.py",
                            "tests/unit/test_sampling.py",
                            "tests/validation/test_grad_checks.py",
                        ),
                    ),
                ),
                known_limits=(
                    "Sobol/Halton quasi-random sequences are deferred pending reference validation.",
                ),
            ),
        ),
    )


__all__ = [
    "EvidenceAnchor",
    "MethodEvidence",
    "NumericalTrustReport",
    "trust_report_to_dict",
    "trust_report_to_json",
    "trust_report_to_markdown",
    "default_numerics_trust_report",
]
