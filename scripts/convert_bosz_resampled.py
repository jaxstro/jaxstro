#!/usr/bin/env python3
"""Convert local BOSZ resampled spectra into processed artifacts.

BOSZ lower-resolution products store a shared wavelength grid separately from
per-spectrum two-column gzip text files. This script converts a selected batch
of ``*_resam.txt.gz`` files into:

- a compressed Zarr v2 store with float32 flux and continuum arrays;
- per-batch Parquet catalog fragments plus a combined catalog;
- JSON validation ledgers for each converted batch.

Raw files are deleted only when ``--delete-raw-after-validate`` is passed and the
batch write/readback validation succeeds.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from jaxstro.atmospheres import (
    BOSZ_2025_RECOMPUTED_NOTE,
    DEFAULT_BOSZ_ZARR,
    parse_bosz_filename,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data/atmospheres/bosz/2025-recomputed/raw/r10000/m+0.00"
DEFAULT_WAVELENGTH = (
    REPO_ROOT
    / "data/atmospheres/bosz/2025-recomputed/wavelength/bosz2024_wave_r10000.txt"
)
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data/atmospheres/bosz/2025-recomputed/processed"
FLOAT32_RTOL = 1.0e-6
FLOAT32_ATOL = 0.0


@dataclass(frozen=True)
class BatchInspection:
    """Shape metadata discovered before conversion."""

    source_count: int
    n_wave: int
    wavelength_min: float
    wavelength_max: float


def _load_optional_data_deps():
    """Import optional data-conversion dependencies with a useful error."""
    try:
        import polars as pl
        import zarr
        from numcodecs import Zstd
    except ImportError as exc:  # pragma: no cover - exercised by users without extra
        raise SystemExit(
            "Install data-conversion dependencies with: uv run --extra data ..."
        ) from exc
    return pl, zarr, Zstd


def _safe_group_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return safe.strip("_") or "spectra"


def _sample_rows(model_count: int) -> tuple[int, ...]:
    if model_count == 1:
        return (0,)
    if model_count == 2:
        return (0, 1)
    return (0, model_count // 2, model_count - 1)


def select_source_files(
    raw_dir: Path, patterns: list[str], limit: int | None
) -> list[Path]:
    files = sorted(raw_dir.rglob("bosz2024_*_resam.txt.gz"))
    if patterns:
        selected = []
        for pattern in patterns:
            selected.extend(
                path for path in files if path.match(pattern) or pattern in path.name
            )
        files = sorted(set(selected))
    if limit is not None:
        files = files[:limit]
    return files


def load_wavelength_grid(wavelength_path: Path) -> np.ndarray:
    wavelength = np.loadtxt(wavelength_path, dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size < 2:
        raise ValueError(f"{wavelength_path}: expected a 1D wavelength grid")
    if not np.all(np.diff(wavelength) > 0.0):
        raise ValueError(
            f"{wavelength_path}: wavelength grid is not strictly increasing"
        )
    return wavelength


def inspect_batch(source_files: list[Path], wavelength: np.ndarray) -> BatchInspection:
    if not source_files:
        raise ValueError("No BOSZ source files selected")

    n_wave = int(wavelength.size)
    for path in source_files:
        metadata = parse_bosz_filename(path)
        if metadata.product != "resam":
            raise ValueError(f"{path.name}: only resampled BOSZ products are supported")
        with gzip.open(path, "rt", encoding="ascii", errors="strict") as handle:
            first = handle.readline().split()
            if len(first) != 2:
                raise ValueError(f"{path.name}: expected two data columns")
            count = 1 + sum(1 for _ in handle)
        if count != n_wave:
            raise ValueError(
                f"{path.name}: expected {n_wave} rows to match wavelength grid, "
                f"got {count}"
            )

    return BatchInspection(
        source_count=len(source_files),
        n_wave=n_wave,
        wavelength_min=float(wavelength[0]),
        wavelength_max=float(wavelength[-1]),
    )


def _read_bosz_columns(path: Path, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    with gzip.open(path, "rt", encoding="ascii", errors="strict") as handle:
        data = np.loadtxt(handle, dtype=np.float64)
    if data.shape != (expected_rows, 2):
        raise ValueError(
            f"{path.name}: expected shape {(expected_rows, 2)}, got {data.shape}"
        )
    return data[:, 0], data[:, 1]


def _write_catalog_fragment(records: list[dict[str, Any]], output_path: Path) -> None:
    pl, _, _ = _load_optional_data_deps()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(records).write_parquet(output_path, compression="zstd")


def rebuild_catalog(processed_dir: Path) -> Path | None:
    """Rebuild the combined catalog from all per-batch Parquet fragments."""
    pl, _, _ = _load_optional_data_deps()
    fragment_dir = processed_dir / "catalog_fragments"
    fragments = sorted(fragment_dir.glob("*.parquet"))
    if not fragments:
        return None

    frames = [pl.read_parquet(path) for path in fragments]
    catalog = pl.concat(frames, how="vertical").sort(
        [
            "resolution",
            "m_h",
            "alpha_m",
            "c_m",
            "vturb_km_s",
            "teff",
            "logg",
            "filename",
        ]
    )
    output_path = processed_dir / "catalog.parquet"
    catalog.write_parquet(output_path, compression="zstd")
    return output_path


def convert_source_files(
    source_files: list[Path],
    *,
    wavelength_path: Path,
    processed_dir: Path,
    zarr_name: str = DEFAULT_BOSZ_ZARR,
    group_name: str = "spectra",
    chunk_models: int = 16,
    zstd_level: int = 3,
    overwrite: bool = False,
    delete_raw_after_validate: bool = False,
) -> dict[str, Any]:
    """Convert one BOSZ resampled batch and optionally delete validated raw files."""
    pl, zarr, Zstd = _load_optional_data_deps()
    _ = pl  # fail early unless the full data extra is present

    wavelength = load_wavelength_grid(wavelength_path)
    source_files = sorted(
        source_files,
        key=lambda path: (
            parse_bosz_filename(path).resolution,
            parse_bosz_filename(path).m_h,
            parse_bosz_filename(path).alpha_m,
            parse_bosz_filename(path).c_m,
            parse_bosz_filename(path).vturb_km_s,
            parse_bosz_filename(path).teff,
            parse_bosz_filename(path).logg,
            path.name,
        ),
    )
    inspection = inspect_batch(source_files, wavelength)

    processed_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = processed_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = processed_dir / zarr_name
    store = zarr.open_group(zarr_path, mode="a", zarr_format=2)
    store.attrs.update(
        {
            "source": "BOSZ 2024 grid, recomputed per STScI 2025 update",
            "source_note": BOSZ_2025_RECOMPUTED_NOTE,
            "wavelength_unit": "angstrom",
            "flux_storage_dtype": "float32",
            "continuum_storage_dtype": "float32",
            "precision_policy": (
                "Raw gzip text columns are parsed as float64 for validation, stored "
                "as float32 to reduce local artifact size, and should be cast to "
                "float64 for interpolation/integration when downstream accuracy "
                "requires it."
            ),
            "column_policy": (
                "For BOSZ lower-resolution resampled products, column 0 and column 1 "
                "are preserved as released and exposed as flux and continuum arrays. "
                "Unit-normalization decisions are left to downstream physical "
                "interfaces."
            ),
            "created_by": "scripts/convert_bosz_resampled.py",
        }
    )

    compressor = Zstd(level=zstd_level)
    if "wavelength" in store:
        existing = np.asarray(store["wavelength"][:], dtype=np.float64)
        if existing.shape != wavelength.shape or not np.allclose(existing, wavelength):
            raise ValueError("Existing BOSZ wavelength grid differs from input")
    else:
        store.create_array(
            "wavelength",
            data=wavelength,
            chunks=(inspection.n_wave,),
            compressor=compressor,
            overwrite=True,
        )

    group_name = _safe_group_name(group_name)
    if group_name in store:
        if not overwrite:
            raise FileExistsError(
                f"{group_name} already exists in {zarr_path}; pass --overwrite to replace"
            )
        del store[group_name]
    group = store.create_group(group_name)
    group.attrs.update(
        {
            "converted_at_utc": datetime.now(UTC).isoformat(),
            "source_count": inspection.source_count,
            "wavelength_path": str(wavelength_path),
        }
    )

    chunk0 = min(chunk_models, inspection.source_count)
    flux = group.create_array(
        "flux",
        shape=(inspection.source_count, inspection.n_wave),
        chunks=(chunk0, inspection.n_wave),
        dtype="float32",
        compressor=compressor,
        overwrite=True,
    )
    continuum = group.create_array(
        "continuum",
        shape=(inspection.source_count, inspection.n_wave),
        chunks=(chunk0, inspection.n_wave),
        dtype="float32",
        compressor=compressor,
        overwrite=True,
    )
    for name in ["teff", "logg", "m_h", "alpha_m", "c_m", "vturb_km_s"]:
        group.create_array(
            name,
            shape=(inspection.source_count,),
            chunks=(min(chunk_models * 16, inspection.source_count),),
            dtype="float64",
            compressor=compressor,
            overwrite=True,
        )

    sample_rows = set(_sample_rows(inspection.source_count))
    sample_flux64: dict[int, np.ndarray] = {}
    sample_continuum64: dict[int, np.ndarray] = {}
    catalog_records: list[dict[str, Any]] = []
    max_abs_err = 0.0
    max_rel_err = 0.0
    sha256_by_file: dict[str, str] = {}

    for row, path in enumerate(source_files):
        sha256 = hashlib.sha256()
        with path.open("rb") as binary_handle:
            for chunk in iter(lambda: binary_handle.read(1024 * 1024), b""):
                sha256.update(chunk)
        sha256_by_file[path.name] = sha256.hexdigest()

        metadata = parse_bosz_filename(path)
        flux64, continuum64 = _read_bosz_columns(path, inspection.n_wave)
        flux32 = flux64.astype(np.float32)
        continuum32 = continuum64.astype(np.float32)
        flux[row, :] = flux32
        continuum[row, :] = continuum32

        for stored32, expected64 in (
            (flux32, flux64),
            (continuum32, continuum64),
        ):
            reread_for_error = stored32.astype(np.float64)
            abs_err = np.max(np.abs(reread_for_error - expected64))
            denom = np.maximum(np.abs(expected64), np.finfo(np.float32).tiny)
            rel_err = np.max(np.abs(reread_for_error - expected64) / denom)
            max_abs_err = max(max_abs_err, float(abs_err))
            max_rel_err = max(max_rel_err, float(rel_err))

        for name, value in (
            ("teff", metadata.teff),
            ("logg", metadata.logg),
            ("m_h", metadata.m_h),
            ("alpha_m", metadata.alpha_m),
            ("c_m", metadata.c_m),
            ("vturb_km_s", metadata.vturb_km_s),
        ):
            group[name][row] = value

        record = asdict(metadata)
        record.update(
            {
                "source_path": str(path),
                "source_size_bytes": path.stat().st_size,
                "source_sha256": sha256_by_file[path.name],
                "n_wave": inspection.n_wave,
                "wavelength_min": inspection.wavelength_min,
                "wavelength_max": inspection.wavelength_max,
                "wavelength_unit": "angstrom",
                "flux_unit": "bosz_resampled_column0",
                "continuum_unit": "bosz_resampled_column1",
                "zarr_group": group_name,
                "zarr_row": row,
            }
        )
        catalog_records.append(record)
        if row in sample_rows:
            sample_flux64[row] = flux64
            sample_continuum64[row] = continuum64

    readback_ok = True
    for row, expected64 in sample_flux64.items():
        stored = flux[row, :].astype(np.float64)
        if not np.allclose(stored, expected64, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL):
            readback_ok = False
            break
    if readback_ok:
        for row, expected64 in sample_continuum64.items():
            stored = continuum[row, :].astype(np.float64)
            if not np.allclose(
                stored, expected64, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL
            ):
                readback_ok = False
                break
    if not readback_ok:
        raise ValueError(f"{group_name}: readback validation failed")

    catalog_fragment = processed_dir / "catalog_fragments" / f"{group_name}.parquet"
    _write_catalog_fragment(catalog_records, catalog_fragment)
    combined_catalog = rebuild_catalog(processed_dir)

    validation = {
        "group_name": group_name,
        "source_count": inspection.source_count,
        "source_files": [path.name for path in source_files],
        "source_sha256": sha256_by_file,
        "wavelength_path": str(wavelength_path),
        "zarr_store": str(zarr_path),
        "zarr_group": group_name,
        "catalog_fragment": str(catalog_fragment),
        "combined_catalog": str(combined_catalog) if combined_catalog else None,
        "n_wave": inspection.n_wave,
        "wavelength_min": inspection.wavelength_min,
        "wavelength_max": inspection.wavelength_max,
        "flux_storage_dtype": "float32",
        "continuum_storage_dtype": "float32",
        "max_abs_float32_roundoff": max_abs_err,
        "max_rel_float32_roundoff": max_rel_err,
        "float32_rtol": FLOAT32_RTOL,
        "readback_ok": readback_ok,
        "raw_deleted": False,
        "validated_at_utc": datetime.now(UTC).isoformat(),
    }

    if delete_raw_after_validate:
        for path in source_files:
            path.unlink()
        validation["raw_deleted"] = True
        validation["deleted_at_utc"] = datetime.now(UTC).isoformat()

    validation_path = validation_dir / f"{group_name}.json"
    validation_path.write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--wavelength", type=Path, default=DEFAULT_WAVELENGTH)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--zarr-name", default=DEFAULT_BOSZ_ZARR)
    parser.add_argument("--group-name", default="bridge_r10000_mplus0_00_v2")
    parser.add_argument("--file", dest="patterns", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunk-models", type=int, default=16)
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delete-raw-after-validate", action="store_true")
    args = parser.parse_args()

    source_files = select_source_files(args.raw_dir, args.patterns, args.limit)
    if args.dry_run:
        print(f"Would convert {len(source_files)} files")
        for path in source_files:
            print(path)
        return

    result = convert_source_files(
        source_files,
        wavelength_path=args.wavelength,
        processed_dir=args.processed_dir,
        zarr_name=args.zarr_name,
        group_name=args.group_name,
        chunk_models=args.chunk_models,
        zstd_level=args.zstd_level,
        overwrite=args.overwrite,
        delete_raw_after_validate=args.delete_raw_after_validate,
    )
    print(
        json.dumps(
            {
                "group_name": result["group_name"],
                "source_count": result["source_count"],
                "n_wave": result["n_wave"],
                "max_rel_float32_roundoff": result["max_rel_float32_roundoff"],
                "raw_deleted": result["raw_deleted"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
