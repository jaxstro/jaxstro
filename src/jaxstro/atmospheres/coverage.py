"""Deterministic atmosphere catalog coverage summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .library import AtmosphereCatalogCoverage


def summarize_catalog_rows(
    *,
    dataset: str,
    rows: Iterable[dict[str, Any]],
    state: str,
    backend_name: str | None = None,
    catalog_path: str | Path | None = None,
    zarr_path: str | Path | None = None,
) -> AtmosphereCatalogCoverage:
    """Summarize raw catalog rows into one deterministic coverage record."""
    row_tuple = tuple(rows)
    wavelength_min_key = (
        "wavelength_min" if _has_key(row_tuple, "wavelength_min") else "lambda_min"
    )
    wavelength_max_key = (
        "wavelength_max" if _has_key(row_tuple, "wavelength_max") else "lambda_max"
    )
    return AtmosphereCatalogCoverage(
        dataset=dataset,
        state=state,
        n_spectra=len(row_tuple),
        teff_min=_min_float(row_tuple, "teff"),
        teff_max=_max_float(row_tuple, "teff"),
        logg_min=_min_float(row_tuple, "logg"),
        logg_max=_max_float(row_tuple, "logg"),
        m_h_values=_unique_floats(row_tuple, "m_h"),
        alpha_m_values=_unique_floats(row_tuple, "alpha_m"),
        c_m_values=_unique_floats(row_tuple, "c_m"),
        vturb_values_km_s=_unique_floats(row_tuple, "vturb_km_s"),
        c_o_values=_unique_floats(row_tuple, "c_o"),
        cloud_labels=_unique_strings(row_tuple, "cloud_label"),
        atmosphere_values=_unique_strings(row_tuple, "atmosphere"),
        resolution=_single_string(row_tuple, "resolution"),
        wavelength_min=_min_float(row_tuple, wavelength_min_key),
        wavelength_max=_max_float(row_tuple, wavelength_max_key),
        wavelength_unit=_single_string(row_tuple, "wavelength_unit")
        or ("nm" if wavelength_min_key == "lambda_min" else None),
        catalog_path=str(catalog_path) if catalog_path is not None else None,
        zarr_path=str(zarr_path) if zarr_path is not None else None,
        backend_name=backend_name,
        backend_available=backend_name in {"newera", "bosz"},
    )


def coverage_rows_to_json(rows: Iterable[AtmosphereCatalogCoverage]) -> str:
    """Serialize coverage rows as sorted, stable JSON."""
    payload = [_coverage_to_dict(row) for row in _sorted_rows(rows)]
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def coverage_rows_to_markdown(rows: Iterable[AtmosphereCatalogCoverage]) -> str:
    """Serialize coverage rows as a deterministic Markdown table."""
    lines = [
        "| dataset | state | count | Teff K | logg | [M/H] | alpha | wavelength | resolution | backend |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in _sorted_rows(rows):
        lines.append(
            " | ".join(
                [
                    f"| {row.dataset}",
                    row.state,
                    str(row.n_spectra),
                    _range(row.teff_min, row.teff_max),
                    _range(row.logg_min, row.logg_max),
                    _values(row.m_h_values),
                    _values(row.alpha_m_values),
                    _wavelength(row),
                    row.resolution or "",
                    row.backend_name or "",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _coverage_to_dict(row: AtmosphereCatalogCoverage) -> dict[str, Any]:
    return {
        "dataset": row.dataset,
        "state": row.state,
        "n_spectra": row.n_spectra,
        "teff_min": row.teff_min,
        "teff_max": row.teff_max,
        "logg_min": row.logg_min,
        "logg_max": row.logg_max,
        "m_h_values": list(row.m_h_values),
        "alpha_m_values": list(row.alpha_m_values),
        "c_m_values": list(row.c_m_values),
        "vturb_values_km_s": list(row.vturb_values_km_s),
        "c_o_values": list(row.c_o_values),
        "cloud_labels": list(row.cloud_labels),
        "atmosphere_values": list(row.atmosphere_values),
        "wavelength_min": row.wavelength_min,
        "wavelength_max": row.wavelength_max,
        "wavelength_unit": row.wavelength_unit,
        "resolution": row.resolution,
        "backend_name": row.backend_name,
        "backend_available": row.backend_available,
        "catalog_path": row.catalog_path,
        "zarr_path": row.zarr_path,
        "raw_path": row.raw_path,
    }


def _sorted_rows(
    rows: Iterable[AtmosphereCatalogCoverage],
) -> tuple[AtmosphereCatalogCoverage, ...]:
    return tuple(sorted(rows, key=lambda row: row.dataset))


def _has_key(rows: tuple[dict[str, Any], ...], key: str) -> bool:
    return any(key in row and row[key] is not None for row in rows)


def _min_float(rows: tuple[dict[str, Any], ...], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    return min(values) if values else None


def _max_float(rows: tuple[dict[str, Any], ...], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row and row[key] is not None]
    return max(values) if values else None


def _unique_floats(rows: tuple[dict[str, Any], ...], key: str) -> tuple[float, ...]:
    return tuple(
        sorted({float(row[key]) for row in rows if key in row and row[key] is not None})
    )


def _unique_strings(rows: tuple[dict[str, Any], ...], key: str) -> tuple[str, ...]:
    return tuple(
        sorted({str(row[key]) for row in rows if key in row and row[key] is not None})
    )


def _single_string(rows: tuple[dict[str, Any], ...], key: str) -> str | None:
    values = _unique_strings(rows, key)
    if not values:
        return None
    return ",".join(values)


def _range(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return ""
    return f"{lower:g}-{upper:g}"


def _values(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:g}" for value in values)


def _wavelength(row: AtmosphereCatalogCoverage) -> str:
    if row.wavelength_min is None or row.wavelength_max is None:
        return ""
    unit = f" {row.wavelength_unit}" if row.wavelength_unit else ""
    return f"{row.wavelength_min:g}-{row.wavelength_max:g}{unit}"


__all__ = [
    "coverage_rows_to_json",
    "coverage_rows_to_markdown",
    "summarize_catalog_rows",
]
