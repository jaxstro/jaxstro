"""Host-side BOSZ processed-artifact backend and raw-file indexing."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from .spectra import AtmosphereParams, PreparedSpectralGrid, SpectrumResult

DEFAULT_BOSZ_ZARR = "bosz_2025_recomputed.zarr"
DEFAULT_BOSZ_CATALOG = "catalog.parquet"
BOSZ_2025_RECOMPUTED_NOTE = (
    "BOSZ files are served from the bosz2024 archive path; the STScI HLSP page "
    "documents a 2025-09-24 recomputation and advises users to replace earlier "
    "2024 files."
)

_BOSZ_PATTERN = re.compile(
    r"^bosz2024_"
    r"(?P<atmos>mp|ms|ap)_"
    r"t(?P<teff>\d+)_"
    r"g(?P<logg>[+-]\d+\.\d)_"
    r"m(?P<m_h>[+-]\d+\.\d\d)_"
    r"a(?P<alpha_m>[+-]\d+\.\d\d)_"
    r"c(?P<c_m>[+-]\d+\.\d\d)_"
    r"v(?P<vturb_km_s>\d+)_"
    r"(?P<resolution>r(?:orig|\d+))_"
    r"(?P<product>resam|noresam|lineid)"
    r"\.txt\.gz$"
)


@dataclass(frozen=True)
class BoszMetadata:
    """Metadata encoded in a BOSZ 2024/2025 recomputed filename."""

    filename: str
    atmosphere: str
    teff: float
    logg: float
    m_h: float
    alpha_m: float
    c_m: float
    vturb_km_s: float
    resolution: str
    product: str


@dataclass(frozen=True)
class BoszFile:
    """One discovered BOSZ spectrum file."""

    path: Path
    metadata: BoszMetadata


@dataclass(frozen=True)
class BoszIndex:
    """Immutable summary of a local BOSZ raw spectrum directory."""

    root: Path
    files: tuple[BoszFile, ...]
    atmospheres: tuple[str, ...]
    teff_values: tuple[float, ...]
    logg_values: tuple[float, ...]
    m_h_values: tuple[float, ...]
    alpha_m_values: tuple[float, ...]
    c_m_values: tuple[float, ...]
    vturb_values_km_s: tuple[float, ...]
    resolutions: tuple[str, ...]
    products: tuple[str, ...]


@dataclass(frozen=True)
class BoszBackend:
    """Lazy host-side backend for processed BOSZ spectra."""

    processed_dir: Path
    catalog_rows: tuple[dict[str, Any], ...]
    zarr_path: Path
    _store: Any = field(repr=False, compare=False)
    resolution: str = "r10000"
    atmosphere: str = "ap"
    product: str = "resam"

    @classmethod
    def open(
        cls,
        processed_dir: str | os.PathLike[str] | None = None,
        *,
        catalog_name: str = DEFAULT_BOSZ_CATALOG,
        zarr_name: str = DEFAULT_BOSZ_ZARR,
        resolution: str = "r10000",
        atmosphere: str = "ap",
        product: str = "resam",
    ) -> "BoszBackend":
        """Open a processed BOSZ artifact directory.

        Optional data dependencies are imported here, not at package import time.
        """
        pl, zarr = _load_optional_backend_deps()

        if processed_dir is None:
            from . import resolve_data_dir

            root = (
                resolve_data_dir()
                / "atmospheres"
                / "bosz"
                / "2025-recomputed"
                / "processed"
            )
        else:
            root = Path(processed_dir).expanduser()

        catalog_path = root / catalog_name
        zarr_path = root / zarr_name
        if not catalog_path.exists():
            raise FileNotFoundError(f"BOSZ catalog not found: {catalog_path}")
        if not zarr_path.exists():
            raise FileNotFoundError(f"BOSZ Zarr store not found: {zarr_path}")

        catalog_rows = tuple(pl.read_parquet(catalog_path).to_dicts())
        store = zarr.open_group(zarr_path, mode="r")
        return cls(
            processed_dir=root,
            catalog_rows=catalog_rows,
            zarr_path=zarr_path,
            resolution=resolution,
            atmosphere=atmosphere,
            product=product,
            _store=store,
        )

    def prepare(self, params: AtmosphereParams) -> PreparedSpectralGrid:
        """Load the local BOSZ interpolation cell enclosing ``params``."""
        teff = _as_host_float(params.teff, "teff")
        logg = _as_host_float(params.logg, "logg")
        m_h = _as_host_float(params.m_h, "m_h")
        alpha_m = _as_host_float(params.alpha_m, "alpha_m")
        c_m = _as_host_float(params.c_m, "c_m")
        vturb_km_s = _as_host_float(params.vturb_km_s, "vturb_km_s")

        rows = [
            row
            for row in self.catalog_rows
            if str(row["resolution"]) == self.resolution
            and str(row["atmosphere"]) == self.atmosphere
            and str(row["product"]) == self.product
            and math.isclose(float(row["m_h"]), m_h)
            and math.isclose(float(row["alpha_m"]), alpha_m)
            and math.isclose(float(row["c_m"]), c_m)
            and math.isclose(float(row["vturb_km_s"]), vturb_km_s)
        ]
        if not rows:
            raise ValueError(
                "No BOSZ plane for "
                f"resolution={self.resolution}, atmosphere={self.atmosphere}, "
                f"product={self.product}, m_h={m_h}, alpha_m={alpha_m}, "
                f"c_m={c_m}, vturb_km_s={vturb_km_s}"
            )

        teff_pair = _bounding_pair([float(row["teff"]) for row in rows], teff)
        logg_pair = _bounding_pair([float(row["logg"]) for row in rows], logg)
        records = {
            (float(row["teff"]), float(row["logg"])): row
            for row in rows
            if float(row["teff"]) in teff_pair and float(row["logg"]) in logg_pair
        }

        first_record = records.get((teff_pair[0], logg_pair[0]))
        if first_record is None:
            raise ValueError("BOSZ local interpolation cell is incomplete")
        wavelength = self._read_wavelength(first_record)

        flux_rows = []
        for teff_value in teff_pair:
            logg_flux = []
            for logg_value in logg_pair:
                record = records.get((teff_value, logg_value))
                if record is None:
                    raise ValueError(
                        "BOSZ local interpolation cell is incomplete for "
                        f"teff={teff_value}, logg={logg_value}"
                    )
                logg_flux.append(self._read_flux(record))
            flux_rows.append(logg_flux)

        return PreparedSpectralGrid(
            teff=jnp.asarray(teff_pair, dtype=jnp.float64),
            logg=jnp.asarray(logg_pair, dtype=jnp.float64),
            wavelength=jnp.asarray(wavelength, dtype=jnp.float64),
            flux=jnp.asarray(np.asarray(flux_rows), dtype=jnp.float64),
            m_h=m_h,
            alpha_m=alpha_m,
            c_m=c_m,
            vturb_km_s=vturb_km_s,
            wavelength_unit="angstrom",
            flux_unit=str(first_record.get("flux_unit", "bosz_resampled_column0")),
        )

    def spectrum(self, params: AtmosphereParams) -> SpectrumResult:
        """Convenience path: prepare a local cell, then interpolate a spectrum."""
        return self.prepare(params).spectrum(params)

    def _read_wavelength(self, record: dict[str, Any]) -> np.ndarray:
        if "wavelength" in self._store:
            return np.asarray(self._store["wavelength"][:], dtype=np.float64)
        wavelength_group = str(record.get("wavelength_zarr_group", ""))
        if wavelength_group:
            group = self._store
            for part in wavelength_group.split("/"):
                group = group[part]
            return np.asarray(group["wavelength"][:], dtype=np.float64)
        raise ValueError("BOSZ processed artifact does not contain a wavelength grid")

    def _read_flux(self, record: dict[str, Any]) -> np.ndarray:
        group = self._store
        for part in str(record["zarr_group"]).split("/"):
            group = group[part]
        flux = group["flux"][int(record["zarr_row"]), :]
        return np.asarray(flux, dtype=np.float64)


def parse_bosz_filename(path: str | os.PathLike[str]) -> BoszMetadata:
    """Parse metadata from a BOSZ 2024/2025 recomputed spectrum filename."""
    filename = Path(path).name
    match = _BOSZ_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"Not a BOSZ 2024 spectrum filename: {filename}")

    resolution = match.group("resolution")
    return BoszMetadata(
        filename=filename,
        atmosphere=match.group("atmos"),
        teff=float(match.group("teff")),
        logg=float(match.group("logg")),
        m_h=float(match.group("m_h")),
        alpha_m=float(match.group("alpha_m")),
        c_m=float(match.group("c_m")),
        vturb_km_s=float(match.group("vturb_km_s")),
        resolution=resolution,
        product=match.group("product"),
    )


def discover_bosz_files(root: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Discover BOSZ 2024/2025 recomputed spectrum files below ``root``."""
    root_path = Path(root).expanduser()
    candidates = root_path.rglob("bosz2024_*.txt.gz")
    files = []
    for path in candidates:
        try:
            metadata = parse_bosz_filename(path)
        except ValueError:
            continue
        files.append(
            (
                metadata.resolution,
                metadata.m_h,
                metadata.alpha_m,
                metadata.c_m,
                metadata.vturb_km_s,
                metadata.teff,
                metadata.logg,
                metadata.filename,
                path,
            )
        )
    return tuple(row[-1] for row in sorted(files))


def build_bosz_index(root: str | os.PathLike[str]) -> BoszIndex:
    """Build an immutable local index for BOSZ raw spectrum files."""
    root_path = Path(root).expanduser()
    files = tuple(
        BoszFile(path=path, metadata=parse_bosz_filename(path))
        for path in discover_bosz_files(root_path)
    )
    return BoszIndex(
        root=root_path,
        files=files,
        atmospheres=tuple(sorted({entry.metadata.atmosphere for entry in files})),
        teff_values=tuple(sorted({entry.metadata.teff for entry in files})),
        logg_values=tuple(sorted({entry.metadata.logg for entry in files})),
        m_h_values=tuple(sorted({entry.metadata.m_h for entry in files})),
        alpha_m_values=tuple(sorted({entry.metadata.alpha_m for entry in files})),
        c_m_values=tuple(sorted({entry.metadata.c_m for entry in files})),
        vturb_values_km_s=tuple(sorted({entry.metadata.vturb_km_s for entry in files})),
        resolutions=tuple(sorted({entry.metadata.resolution for entry in files})),
        products=tuple(sorted({entry.metadata.product for entry in files})),
    )


def _load_optional_backend_deps():
    try:
        pl = import_module("polars")
        zarr = import_module("zarr")
    except ImportError as exc:  # pragma: no cover - depends on local extra
        raise ImportError(
            "BoszBackend requires the optional data dependencies. "
            "Install or run with: uv run --extra data ..."
        ) from exc
    return pl, zarr


def _as_host_float(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"BoszBackend.prepare expects scalar {name}")
    return float(array)


def _bounding_pair(values: list[float], target: float) -> tuple[float, float]:
    unique = sorted(set(values))
    if len(unique) < 2:
        raise ValueError("BOSZ interpolation requires at least two axis values")
    if target <= unique[0]:
        return unique[0], unique[1]
    if target >= unique[-1]:
        return unique[-2], unique[-1]
    for lower, upper in zip(unique[:-1], unique[1:]):
        if lower <= target <= upper:
            return lower, upper
    raise ValueError(f"Could not bracket target value {target}")


__all__ = [
    "BOSZ_2025_RECOMPUTED_NOTE",
    "DEFAULT_BOSZ_CATALOG",
    "DEFAULT_BOSZ_ZARR",
    "BoszBackend",
    "BoszFile",
    "BoszIndex",
    "BoszMetadata",
    "build_bosz_index",
    "discover_bosz_files",
    "parse_bosz_filename",
]
