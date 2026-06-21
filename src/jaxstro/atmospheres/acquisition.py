"""No-download targeted acquisition decision tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .library import AtmosphereCatalogCoverage


@dataclass(frozen=True)
class AcquisitionDecision:
    """One row in a targeted acquisition decision table."""

    category: str
    coverage_status: str
    action: str
    rationale: str


def plan_targeted_acquisition(
    coverages: Iterable[AtmosphereCatalogCoverage],
) -> tuple[AcquisitionDecision, ...]:
    """Build a concise acquisition decision table without downloading data."""
    rows = tuple(coverages)
    min_teff = min(
        (row.teff_min for row in rows if row.teff_min is not None),
        default=None,
    )
    bosz_rows = [row for row in rows if "bosz" in row.dataset.lower()]
    resolutions = {row.resolution for row in bosz_rows if row.resolution}
    alpha_values = {value for row in bosz_rows for value in row.alpha_m_values}
    carbon_values = {value for row in bosz_rows for value in row.c_m_values}

    return (
        AcquisitionDecision(
            category="white dwarfs",
            coverage_status=_present_status(rows, ("wd", "white_dwarf", "white dwarf")),
            action="justify-before-download",
            rationale="No local processed white-dwarf atmosphere family is represented.",
        ),
        AcquisitionDecision(
            category="WR / wind-dominated very hot stars",
            coverage_status=_present_status(rows, ("wr", "wind")),
            action="justify-before-download",
            rationale="TLUSTY covers hot static atmospheres, not wind-dominated WR models.",
        ),
        AcquisitionDecision(
            category="very cool objects below 900 K",
            coverage_status=(
                "covered" if min_teff is not None and min_teff < 900.0 else "gap"
            ),
            action="justify-before-download",
            rationale="Current Sonora staged coverage starts at 900 K.",
        ),
        AcquisitionDecision(
            category="non-solar BOSZ alpha/carbon",
            coverage_status=(
                "partial"
                if alpha_values == {0.0} and carbon_values == {0.0}
                else "unknown"
            ),
            action="defer",
            rationale="Acquire only if downstream science requires abundance sensitivity.",
        ),
        AcquisitionDecision(
            category="additional BOSZ resolutions",
            coverage_status="partial" if resolutions == {"r10000"} else "unknown",
            action="defer",
            rationale="Acquire only if downstream workflows need another resolution tier.",
        ),
    )


def acquisition_rows_to_markdown(rows: Iterable[AcquisitionDecision]) -> str:
    """Serialize acquisition decisions as a Markdown table."""
    lines = [
        "| category | coverage status | action | rationale |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.category} | {row.coverage_status} | {row.action} | {row.rationale} |"
        )
    return "\n".join(lines) + "\n"


def _present_status(
    rows: tuple[AtmosphereCatalogCoverage, ...],
    tokens: tuple[str, ...],
) -> str:
    for row in rows:
        haystack = " ".join(
            [
                row.dataset,
                *(row.atmosphere_values or ()),
                str(row.provenance),
            ]
        ).lower()
        words = set(re.split(r"[^a-z0-9]+", haystack))
        normalized = haystack.replace("_", " ")
        for token in tokens:
            phrase = token.replace("_", " ")
            if token in words or ((" " in phrase) and phrase in normalized):
                return "present"
    return "gap"


__all__ = [
    "AcquisitionDecision",
    "acquisition_rows_to_markdown",
    "plan_targeted_acquisition",
]
