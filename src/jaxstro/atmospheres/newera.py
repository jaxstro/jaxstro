"""Host-side PHOENIX/NewEra processed-artifact backend."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from .spectra import AtmosphereParams, PreparedSpectralGrid, SpectrumResult

DEFAULT_NEWERA_ZARR = "newera_lowres_v3.zarr"
DEFAULT_NEWERA_CATALOG = "catalog.parquet"


@dataclass(frozen=True)
class NewEraBackend:
    """Lazy host-side backend for processed NewEra low-resolution spectra."""

    processed_dir: Path
    catalog_rows: tuple[dict[str, Any], ...]
    zarr_path: Path
    _store: Any = field(repr=False, compare=False)

    @classmethod
    def open(
        cls,
        processed_dir: str | os.PathLike[str] | None = None,
        *,
        catalog_name: str = DEFAULT_NEWERA_CATALOG,
        zarr_name: str = DEFAULT_NEWERA_ZARR,
    ) -> "NewEraBackend":
        """Open a processed NewEra artifact directory.

        Optional data dependencies are imported here, not at package import time.
        """
        pl, zarr = _load_optional_backend_deps()

        if processed_dir is None:
            from . import resolve_data_dir

            root = resolve_data_dir() / "atmospheres" / "newera" / "processed"
        else:
            root = Path(processed_dir).expanduser()

        catalog_path = root / catalog_name
        zarr_path = root / zarr_name
        if not catalog_path.exists():
            raise FileNotFoundError(f"NewEra catalog not found: {catalog_path}")
        if not zarr_path.exists():
            raise FileNotFoundError(f"NewEra Zarr store not found: {zarr_path}")

        catalog_rows = tuple(pl.read_parquet(catalog_path).to_dicts())
        store = zarr.open_group(zarr_path, mode="r")
        return cls(
            processed_dir=root,
            catalog_rows=catalog_rows,
            zarr_path=zarr_path,
            _store=store,
        )

    def prepare(self, params: AtmosphereParams) -> PreparedSpectralGrid:
        """Load the local NewEra interpolation cell enclosing ``params``."""
        teff = _as_host_float(params.teff, "teff")
        logg = _as_host_float(params.logg, "logg")
        m_h = _as_host_float(params.m_h, "m_h")
        alpha_m = _as_host_float(params.alpha_m, "alpha_m")

        rows = [
            row
            for row in self.catalog_rows
            if math.isclose(float(row["m_h"]), m_h)
            and math.isclose(float(row["alpha_m"]), alpha_m)
        ]
        if not rows:
            raise ValueError(
                f"No NewEra abundance plane for m_h={m_h}, alpha_m={alpha_m}"
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
            raise ValueError("NewEra local interpolation cell is incomplete")
        wavelength = _wavelength_grid(first_record)

        flux_rows = []
        for teff_value in teff_pair:
            logg_flux = []
            for logg_value in logg_pair:
                record = records.get((teff_value, logg_value))
                if record is None:
                    raise ValueError(
                        "NewEra local interpolation cell is incomplete for "
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
            wavelength_unit="nm",
            flux_unit="source_flux_lambda",
        )

    def spectrum(self, params: AtmosphereParams) -> SpectrumResult:
        """Convenience path: prepare a local cell, then interpolate a spectrum."""
        return self.prepare(params).spectrum(params)

    def _read_flux(self, record: dict[str, Any]) -> np.ndarray:
        group = self._store
        for part in str(record["zarr_group"]).split("/"):
            group = group[part]
        flux = group["flux"][int(record["zarr_row"]), :]
        return np.asarray(flux, dtype=np.float64)


def _load_optional_backend_deps():
    try:
        pl = import_module("polars")
        zarr = import_module("zarr")
    except ImportError as exc:  # pragma: no cover - depends on local extra
        raise ImportError(
            "NewEraBackend requires the optional data dependencies. "
            "Install or run with: uv run --extra data ..."
        ) from exc
    return pl, zarr


def _as_host_float(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"NewEraBackend.prepare expects scalar {name}")
    return float(array)


def _bounding_pair(values: list[float], target: float) -> tuple[float, float]:
    unique = sorted(set(values))
    if len(unique) < 2:
        raise ValueError("NewEra interpolation requires at least two axis values")
    if target <= unique[0]:
        return unique[0], unique[1]
    if target >= unique[-1]:
        return unique[-2], unique[-1]
    for lower, upper in zip(unique[:-1], unique[1:]):
        if lower <= target <= upper:
            return lower, upper
    raise ValueError(f"Could not bracket target value {target}")


def _wavelength_grid(record: dict[str, Any]) -> np.ndarray:
    n_wave = int(record["n_wave"])
    lambda_min = float(record["lambda_min"])
    lambda_step = float(record["lambda_step"])
    return lambda_min + np.arange(n_wave, dtype=np.float64) * lambda_step


__all__ = [
    "DEFAULT_NEWERA_CATALOG",
    "DEFAULT_NEWERA_ZARR",
    "NewEraBackend",
]
