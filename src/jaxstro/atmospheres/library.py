"""Catalog-first atmosphere library selection.

This module is host-side by design. It ranks local processed or staged
atmosphere datasets without importing optional data dependencies at package
import time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

from .spectra import AtmosphereParams, SpectrumResult


@dataclass(frozen=True)
class AtmosphereCatalogCoverage:
    """Deterministic coverage summary for one atmosphere dataset."""

    dataset: str
    state: str
    n_spectra: int
    teff_min: float | None = None
    teff_max: float | None = None
    logg_min: float | None = None
    logg_max: float | None = None
    m_h_values: tuple[float, ...] = ()
    alpha_m_values: tuple[float, ...] = ()
    c_m_values: tuple[float, ...] = ()
    vturb_values_km_s: tuple[float, ...] = ()
    c_o_values: tuple[float, ...] = ()
    cloud_labels: tuple[str, ...] = ()
    atmosphere_values: tuple[str, ...] = ()
    resolution: str | None = None
    wavelength_min: float | None = None
    wavelength_max: float | None = None
    wavelength_unit: str | None = None
    catalog_path: str | None = None
    zarr_path: str | None = None
    raw_path: str | None = None
    backend_name: str | None = None
    backend_available: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AtmosphereLibraryCandidate:
    """Ranked candidate returned by an atmosphere library query."""

    coverage: AtmosphereCatalogCoverage
    score: float
    rank_reason: str

    @property
    def dataset(self) -> str:
        return self.coverage.dataset

    @property
    def state(self) -> str:
        return self.coverage.state

    @property
    def backend_name(self) -> str | None:
        return self.coverage.backend_name

    @property
    def backend_available(self) -> bool:
        return self.coverage.backend_available


@dataclass(frozen=True)
class AtmosphereSelection:
    """Result of a catalog-first atmosphere library query."""

    status: str
    requested: AtmosphereParams
    candidates: tuple[AtmosphereLibraryCandidate, ...] = ()
    selected: AtmosphereLibraryCandidate | None = None
    message: str = ""


@dataclass(frozen=True)
class AtmosphereLibrary:
    """Small catalog-first selector over local atmosphere datasets."""

    coverages: tuple[AtmosphereCatalogCoverage, ...]
    data_dir: Path | None = None

    @classmethod
    def from_coverages(
        cls,
        coverages: list[AtmosphereCatalogCoverage]
        | tuple[AtmosphereCatalogCoverage, ...],
    ) -> "AtmosphereLibrary":
        """Build a library from already summarized coverage rows."""
        return cls(coverages=tuple(sorted(coverages, key=lambda row: row.dataset)))

    @classmethod
    def from_local(
        cls,
        data_dir: str | Path | None = None,
    ) -> "AtmosphereLibrary":
        """Discover local processed catalogs and staged raw datasets."""
        from . import resolve_data_dir
        from .coverage import summarize_catalog_rows

        root = (
            Path(data_dir).expanduser() if data_dir is not None else resolve_data_dir()
        )
        atmospheres_root = root / "atmospheres"
        coverages: list[AtmosphereCatalogCoverage] = []

        newera_catalog = atmospheres_root / "newera" / "processed" / "catalog.parquet"
        if newera_catalog.exists():
            rows = _read_parquet_rows(newera_catalog)
            coverages.append(
                summarize_catalog_rows(
                    dataset="newera_lowres_v3",
                    rows=rows,
                    state="processed",
                    backend_name="newera",
                    catalog_path=newera_catalog,
                    zarr_path=atmospheres_root
                    / "newera"
                    / "processed"
                    / "newera_lowres_v3.zarr",
                )
            )

        bosz_catalog = (
            atmospheres_root
            / "bosz"
            / "2025-recomputed"
            / "processed"
            / "catalog.parquet"
        )
        if bosz_catalog.exists():
            rows = _read_parquet_rows(bosz_catalog)
            coverages.append(
                summarize_catalog_rows(
                    dataset="bosz_2025_recomputed",
                    rows=rows,
                    state="processed",
                    backend_name="bosz",
                    catalog_path=bosz_catalog,
                    zarr_path=bosz_catalog.parent / "bosz_2025_recomputed.zarr",
                )
            )

        sonora_catalog = (
            atmospheres_root / "sonora" / "2024" / "processed" / "catalog.parquet"
        )
        if sonora_catalog.exists():
            rows = _read_parquet_rows(sonora_catalog)
            coverages.append(
                summarize_catalog_rows(
                    dataset="sonora_2024",
                    rows=rows,
                    state="processed",
                    catalog_path=sonora_catalog,
                    zarr_path=sonora_catalog.parent / "sonora_2024.zarr",
                )
            )

        tlusty_catalog = atmospheres_root / "tlusty" / "processed" / "catalog.parquet"
        if tlusty_catalog.exists():
            rows = _read_parquet_rows(tlusty_catalog)
            for dataset in sorted({str(row.get("dataset", "tlusty")) for row in rows}):
                dataset_rows = [
                    row for row in rows if str(row.get("dataset", "tlusty")) == dataset
                ]
                coverages.append(
                    summarize_catalog_rows(
                        dataset=dataset,
                        rows=dataset_rows,
                        state="processed",
                        catalog_path=tlusty_catalog,
                        zarr_path=tlusty_catalog.parent / "tlusty_flux.zarr",
                    )
                )

        staging_manifest = atmospheres_root / "local-staging-manifest.json"
        if staging_manifest.exists():
            processed_datasets = {coverage.dataset for coverage in coverages}
            coverages.extend(
                coverage
                for coverage in _coverages_from_staging_manifest(staging_manifest)
                if coverage.dataset not in processed_datasets
            )

        return cls(
            coverages=tuple(sorted(coverages, key=lambda row: row.dataset)),
            data_dir=root,
        )

    def coverage(self) -> tuple[AtmosphereCatalogCoverage, ...]:
        """Return deterministic coverage rows."""
        return self.coverages

    def select(
        self,
        params: AtmosphereParams,
        *,
        family: str | None = None,
        resolution: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> AtmosphereSelection:
        """Rank local atmosphere datasets for ``params``."""
        filters = filters or {}
        matches: list[AtmosphereLibraryCandidate] = []
        rejection_reasons: list[str] = []
        for coverage in self.coverages:
            reason = _coverage_rejection_reason(
                coverage,
                params,
                family=family,
                resolution=resolution,
                filters=filters,
            )
            if reason is not None:
                rejection_reasons.append(f"{coverage.dataset}: {reason}")
                continue

            backend_ready = coverage.state == "processed" and coverage.backend_available
            rank_reason = (
                "processed backend covers request"
                if backend_ready
                else "coverage match without backend"
            )
            matches.append(
                AtmosphereLibraryCandidate(
                    coverage=coverage,
                    score=_coverage_score(
                        coverage, params, backend_ready=backend_ready
                    ),
                    rank_reason=rank_reason,
                )
            )

        candidates = tuple(sorted(matches, key=lambda item: (item.score, item.dataset)))
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate.backend_available and candidate.state == "processed"
            ),
            None,
        )
        if selected is not None:
            return AtmosphereSelection(
                status="ok",
                requested=params,
                candidates=candidates,
                selected=selected,
                message=f"Selected {selected.dataset}",
            )
        if candidates:
            return AtmosphereSelection(
                status="backend_unavailable",
                requested=params,
                candidates=candidates,
                selected=None,
                message="Coverage exists, but no processed backend is available",
            )
        return AtmosphereSelection(
            status="no_match",
            requested=params,
            candidates=(),
            selected=None,
            message="; ".join(rejection_reasons) or "No atmosphere catalogs loaded",
        )

    def spectrum(
        self,
        params: AtmosphereParams,
        *,
        family: str | None = None,
        resolution: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> SpectrumResult:
        """Return a spectrum from the selected processed backend."""
        selection = self.select(
            params,
            family=family,
            resolution=resolution,
            filters=filters,
        )
        if selection.selected is None:
            raise RuntimeError(
                f"No processed atmosphere backend matches request: {selection.message}"
            )

        coverage = selection.selected.coverage
        if coverage.backend_name == "newera":
            from .newera import NewEraBackend

            if coverage.catalog_path is None:
                raise RuntimeError(f"{coverage.dataset} has no catalog path")
            return NewEraBackend.open(Path(coverage.catalog_path).parent).spectrum(
                params
            )
        if coverage.backend_name == "bosz":
            from .bosz import BoszBackend

            if coverage.catalog_path is None:
                raise RuntimeError(f"{coverage.dataset} has no catalog path")
            return BoszBackend.open(
                Path(coverage.catalog_path).parent,
                resolution=coverage.resolution or "r10000",
            ).spectrum(params)
        raise RuntimeError(
            "Selected atmosphere catalog has no implemented backend: "
            f"{coverage.dataset}"
        )


def _read_parquet_rows(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        pl = import_module("polars")
    except ImportError as exc:  # pragma: no cover - depends on local extra
        raise ImportError(
            "AtmosphereLibrary.from_local requires the optional data dependencies. "
            "Install or run with: uv run --extra data ..."
        ) from exc
    return tuple(pl.read_parquet(path).to_dicts())


def _coverages_from_staging_manifest(
    manifest_path: Path,
) -> tuple[AtmosphereCatalogCoverage, ...]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets", {})
    rows = []
    for dataset, data in sorted(datasets.items()):
        if dataset.startswith("bosz_"):
            continue
        logg_min = _maybe_float(data.get("logg_min"))
        logg_max = _maybe_float(data.get("logg_max"))
        provenance: dict[str, Any] = {"manifest": str(manifest_path)}
        if dataset == "sonora_2024" and logg_min is not None and logg_min < 3.0:
            logg_min += 2.0
            logg_max = None if logg_max is None else logg_max + 2.0
            provenance["logg_note"] = (
                "local manifest stored log10(g_m_s2); converted to cgs logg "
                "for AtmosphereParams coverage"
            )
        rows.append(
            AtmosphereCatalogCoverage(
                dataset=dataset,
                state="raw",
                n_spectra=int(data.get("n_spectra", data.get("n_flux_files", 0))),
                teff_min=_maybe_float(data.get("teff_min")),
                teff_max=_maybe_float(data.get("teff_max")),
                logg_min=logg_min,
                logg_max=logg_max,
                m_h_values=_float_tuple(data.get("m_h_values", ())),
                vturb_values_km_s=_float_tuple(data.get("vturb_values_km_s", ())),
                cloud_labels=tuple(data.get("cloud_labels", ())),
                atmosphere_values=tuple(data.get("prefixes", ())),
                raw_path=str(manifest_path.parent),
                backend_available=False,
                provenance=provenance,
            )
        )
    return tuple(rows)


def _coverage_rejection_reason(
    coverage: AtmosphereCatalogCoverage,
    params: AtmosphereParams,
    *,
    family: str | None,
    resolution: str | None,
    filters: dict[str, Any],
) -> str | None:
    teff = _as_float(params.teff)
    logg = _as_float(params.logg)
    if not _in_range(teff, coverage.teff_min, coverage.teff_max):
        return "teff outside coverage"
    if not _in_range(logg, coverage.logg_min, coverage.logg_max):
        return "logg outside coverage"
    if coverage.m_h_values and not _contains_close(coverage.m_h_values, params.m_h):
        return "m_h plane unavailable"
    if coverage.alpha_m_values and not _contains_close(
        coverage.alpha_m_values, params.alpha_m
    ):
        return "alpha_m plane unavailable"
    if coverage.c_m_values and not _contains_close(coverage.c_m_values, params.c_m):
        return "c_m plane unavailable"
    if coverage.vturb_values_km_s and not _contains_close(
        coverage.vturb_values_km_s, params.vturb_km_s
    ):
        return "vturb_km_s plane unavailable"
    if family is not None and not _matches_family(coverage, family):
        return "family mismatch"
    if resolution is not None and coverage.resolution != resolution:
        return "resolution mismatch"
    for key, value in filters.items():
        if not _matches_filter(coverage, key, value):
            return f"{key} filter mismatch"
    return None


def _coverage_score(
    coverage: AtmosphereCatalogCoverage,
    params: AtmosphereParams,
    *,
    backend_ready: bool,
) -> float:
    teff = _as_float(params.teff)
    logg = _as_float(params.logg)
    teff_mid = _midpoint(coverage.teff_min, coverage.teff_max, fallback=teff)
    logg_mid = _midpoint(coverage.logg_min, coverage.logg_max, fallback=logg)
    teff_span = max((coverage.teff_max or teff) - (coverage.teff_min or teff), 1.0)
    logg_span = max((coverage.logg_max or logg) - (coverage.logg_min or logg), 1.0)
    distance = abs(teff - teff_mid) / teff_span + abs(logg - logg_mid) / logg_span
    return distance + (0.0 if backend_ready else 1000.0)


def _matches_family(coverage: AtmosphereCatalogCoverage, family: str) -> bool:
    token = family.lower()
    haystack = [
        coverage.dataset,
        coverage.backend_name or "",
        *coverage.atmosphere_values,
        *coverage.cloud_labels,
    ]
    return any(token in str(item).lower() for item in haystack)


def _matches_filter(
    coverage: AtmosphereCatalogCoverage,
    key: str,
    value: Any,
) -> bool:
    values = getattr(coverage, f"{key}_values", None)
    if values is None:
        values = getattr(coverage, key, None)
    if values is None:
        return False
    if isinstance(values, tuple):
        if not values:
            return False
        if all(isinstance(item, float) for item in values):
            return _contains_close(values, value)
        return str(value) in {str(item) for item in values}
    return values == value


def _contains_close(values: tuple[float, ...], value: Any) -> bool:
    target = _as_float(value)
    return any(
        math.isclose(item, target, rel_tol=0.0, abs_tol=1.0e-8) for item in values
    )


def _in_range(value: float, lower: float | None, upper: float | None) -> bool:
    if lower is not None and value < lower:
        return False
    if upper is not None and value > upper:
        return False
    return True


def _midpoint(lower: float | None, upper: float | None, *, fallback: float) -> float:
    if lower is None or upper is None:
        return fallback
    return 0.5 * (lower + upper)


def _as_float(value: Any) -> float:
    return float(value)


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _float_tuple(values: Any) -> tuple[float, ...]:
    return tuple(sorted(float(value) for value in values))


__all__ = [
    "AtmosphereCatalogCoverage",
    "AtmosphereLibrary",
    "AtmosphereLibraryCandidate",
    "AtmosphereSelection",
]
