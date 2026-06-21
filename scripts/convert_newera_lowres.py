#!/usr/bin/env python3
"""Convert local PHOENIX/NewEra low-res text spectra into processed artifacts.

The raw NewEra low-resolution files alternate one model metadata line with one
very long flux-vector line. This script converts those source files into:

- a compressed Zarr v2 store with float32 spectra;
- per-file Parquet catalog fragments plus a combined catalog;
- JSON validation ledgers for each converted raw file.

Raw files are deleted only when ``--delete-raw-after-validate`` is passed and the
per-file write/readback validation succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from jaxstro.atmospheres import parse_newera_lowres_filename

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = (
    REPO_ROOT / "data/atmospheres/newera/raw/PHOENIX-NewEraV3-LowRes-SPECTRA"
)
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data/atmospheres/newera/processed"
DEFAULT_MANIFEST = (
    REPO_ROOT / "data/atmospheres/newera/manifest/list_of_available_NewEraV3_models.txt"
)
DEFAULT_ZARR = "newera_lowres_v3.zarr"
FLOAT32_RTOL = 1.0e-6
FLOAT32_ATOL = 0.0


@dataclass(frozen=True)
class SourceInspection:
    """Shape metadata discovered before conversion."""

    source_path: Path
    model_count: int
    n_wave: int
    lambda_min: float
    lambda_max: float
    lambda_step: float


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


def _safe_group_name(path: Path) -> str:
    return path.stem.replace(".", "_").replace("+", "plus").replace("-", "minus")


def parse_model_metadata(
    tokens: list[str], source_path: Path, source_row: int
) -> dict[str, Any]:
    """Parse the fixed-position metadata fields used by NewEra low-res rows."""
    if len(tokens) < 31:
        raise ValueError(
            f"{source_path.name} row {source_row}: expected at least 31 metadata "
            f"fields, got {len(tokens)}"
        )

    filename_meta = parse_newera_lowres_filename(source_path)
    return {
        "source_file": source_path.name,
        "source_row": source_row,
        "version": filename_meta.version,
        "product": filename_meta.product,
        "m_h": filename_meta.m_h,
        "alpha_m": filename_meta.alpha_m,
        "n_wave": int(tokens[8]),
        "lambda_min": float(tokens[9]),
        "lambda_max": float(tokens[10]),
        "lambda_step": float(tokens[11]),
        "teff": float(tokens[12]),
        "logg": float(tokens[13]),
        "mass": float(tokens[14]),
        "row_abundance_anchor": float(tokens[18]),
        "row_alpha_m": float(tokens[19]),
        "raw_metadata": " ".join(tokens),
    }


def inspect_source_file(source_path: Path) -> SourceInspection:
    """Count models and confirm fixed wavelength metadata for one raw text file."""
    model_count = 0
    n_wave: int | None = None
    lambda_min: float | None = None
    lambda_max: float | None = None
    lambda_step: float | None = None

    with source_path.open("rt", encoding="ascii", errors="strict") as handle:
        while True:
            metadata_line = handle.readline()
            if not metadata_line:
                break
            flux_line = handle.readline()
            if not flux_line:
                raise ValueError(f"{source_path.name}: dangling metadata line at EOF")

            metadata = parse_model_metadata(
                metadata_line.split(), source_path, source_row=model_count
            )
            if n_wave is None:
                n_wave = metadata["n_wave"]
                lambda_min = metadata["lambda_min"]
                lambda_max = metadata["lambda_max"]
                lambda_step = metadata["lambda_step"]
            elif (
                n_wave != metadata["n_wave"]
                or not math.isclose(lambda_min, metadata["lambda_min"])
                or not math.isclose(lambda_max, metadata["lambda_max"])
                or not math.isclose(lambda_step, metadata["lambda_step"])
            ):
                raise ValueError(
                    f"{source_path.name} row {model_count}: wavelength metadata changed"
                )

            model_count += 1

    if model_count == 0 or n_wave is None:
        raise ValueError(f"{source_path.name}: no models found")

    return SourceInspection(
        source_path=source_path,
        model_count=model_count,
        n_wave=n_wave,
        lambda_min=float(lambda_min),
        lambda_max=float(lambda_max),
        lambda_step=float(lambda_step),
    )


def _sample_rows(model_count: int) -> tuple[int, ...]:
    if model_count == 1:
        return (0,)
    if model_count == 2:
        return (0, 1)
    return (0, model_count // 2, model_count - 1)


def _write_catalog_fragment(records: list[dict[str, Any]], output_path: Path) -> None:
    pl, _, _ = _load_optional_data_deps()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(records).write_parquet(output_path, compression="zstd")


def rebuild_catalog(processed_dir: Path) -> Path | None:
    """Rebuild the combined catalog from all per-file Parquet fragments."""
    pl, _, _ = _load_optional_data_deps()
    fragment_dir = processed_dir / "catalog_fragments"
    fragments = sorted(fragment_dir.glob("*.parquet"))
    if not fragments:
        return None

    frames = [pl.read_parquet(path) for path in fragments]
    catalog = pl.concat(frames, how="vertical").sort(
        ["m_h", "alpha_m", "teff", "logg", "source_file", "source_row"]
    )
    output_path = processed_dir / "catalog.parquet"
    catalog.write_parquet(output_path, compression="zstd")
    return output_path


def write_manifest_parquet(manifest_path: Path, processed_dir: Path) -> Path | None:
    """Convert the upstream manifest to a compact Parquet table."""
    if not manifest_path.exists():
        return None

    pl, _, _ = _load_optional_data_deps()
    rows = []
    with manifest_path.open("rt", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append(
                {
                    "index": int(parts[0]),
                    "filename": parts[1],
                    "md5": parts[2],
                    "filesize_bytes": int(parts[3]),
                    "download_url": parts[4],
                }
            )

    output_path = processed_dir / "manifest.parquet"
    processed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(output_path, compression="zstd")
    return output_path


def convert_source_file(
    source_path: Path,
    *,
    processed_dir: Path,
    zarr_name: str = DEFAULT_ZARR,
    chunk_models: int = 4,
    zstd_level: int = 3,
    overwrite: bool = False,
    delete_raw_after_validate: bool = False,
) -> dict[str, Any]:
    """Convert one raw NewEra low-res text file and optionally delete it."""
    pl, zarr, Zstd = _load_optional_data_deps()
    _ = pl  # fail early unless the full data extra is present

    processed_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = processed_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = processed_dir / zarr_name
    store = zarr.open_group(zarr_path, mode="a", zarr_format=2)
    store.attrs.update(
        {
            "source": "PHOENIX/NewEra V3 low-resolution spectra",
            "flux_storage_dtype": "float32",
            "precision_policy": (
                "Raw ASCII flux values are parsed as float64 for validation, stored "
                "as float32 to reduce local artifact size, and should be cast to "
                "float64 for interpolation/integration when downstream accuracy "
                "requires it."
            ),
            "created_by": "scripts/convert_newera_lowres.py",
        }
    )

    inspection = inspect_source_file(source_path)
    group_name = _safe_group_name(source_path)
    files_group = store.require_group("files")
    if group_name in files_group:
        if not overwrite:
            raise FileExistsError(
                f"{group_name} already exists in {zarr_path}; pass --overwrite to replace"
            )
        del files_group[group_name]
    group = files_group.create_group(group_name)
    group.attrs.update(
        {
            "source_file": source_path.name,
            "source_size_bytes": source_path.stat().st_size,
            "converted_at_utc": datetime.now(UTC).isoformat(),
        }
    )

    compressor = Zstd(level=zstd_level)
    flux = group.create_array(
        "flux",
        shape=(inspection.model_count, inspection.n_wave),
        chunks=(min(chunk_models, inspection.model_count), inspection.n_wave),
        dtype="float32",
        compressor=compressor,
        overwrite=True,
    )
    for name in ["teff", "logg", "m_h", "alpha_m"]:
        group.create_array(
            name,
            shape=(inspection.model_count,),
            chunks=(min(chunk_models * 16, inspection.model_count),),
            dtype="float64",
            compressor=compressor,
            overwrite=True,
        )
    group.create_array(
        "source_row",
        shape=(inspection.model_count,),
        chunks=(min(chunk_models * 16, inspection.model_count),),
        dtype="int32",
        compressor=compressor,
        overwrite=True,
    )

    sample_rows = set(_sample_rows(inspection.model_count))
    sample_flux64: dict[int, np.ndarray] = {}
    catalog_records: list[dict[str, Any]] = []
    max_abs_err = 0.0
    max_rel_err = 0.0
    sha256 = hashlib.sha256()

    with source_path.open("rb") as binary_handle:
        for chunk in iter(lambda: binary_handle.read(1024 * 1024), b""):
            sha256.update(chunk)

    with source_path.open("rt", encoding="ascii", errors="strict") as handle:
        for row in range(inspection.model_count):
            metadata_line = handle.readline()
            flux_line = handle.readline()
            if not metadata_line or not flux_line:
                raise ValueError(f"{source_path.name}: unexpected EOF at row {row}")

            record = parse_model_metadata(metadata_line.split(), source_path, row)
            flux64 = np.fromstring(flux_line, sep=" ", dtype=np.float64)
            if flux64.size != inspection.n_wave:
                raise ValueError(
                    f"{source_path.name} row {row}: expected {inspection.n_wave} flux "
                    f"values, got {flux64.size}"
                )
            flux32 = flux64.astype(np.float32)
            flux[row, :] = flux32

            reread_for_error = flux32.astype(np.float64)
            abs_err = np.max(np.abs(reread_for_error - flux64))
            denom = np.maximum(np.abs(flux64), np.finfo(np.float32).tiny)
            rel_err = np.max(np.abs(reread_for_error - flux64) / denom)
            max_abs_err = max(max_abs_err, float(abs_err))
            max_rel_err = max(max_rel_err, float(rel_err))

            group["teff"][row] = record["teff"]
            group["logg"][row] = record["logg"]
            group["m_h"][row] = record["m_h"]
            group["alpha_m"][row] = record["alpha_m"]
            group["source_row"][row] = row

            record["zarr_group"] = f"files/{group_name}"
            record["zarr_row"] = row
            catalog_records.append(record)
            if row in sample_rows:
                sample_flux64[row] = flux64

    readback_ok = True
    for row, expected64 in sample_flux64.items():
        stored = flux[row, :].astype(np.float64)
        if not np.allclose(stored, expected64, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL):
            readback_ok = False
            break

    catalog_fragment = processed_dir / "catalog_fragments" / f"{group_name}.parquet"
    _write_catalog_fragment(catalog_records, catalog_fragment)
    combined_catalog = rebuild_catalog(processed_dir)

    validation = {
        "source_file": source_path.name,
        "source_path": str(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_sha256": sha256.hexdigest(),
        "zarr_store": str(zarr_path),
        "zarr_group": f"files/{group_name}",
        "catalog_fragment": str(catalog_fragment),
        "combined_catalog": str(combined_catalog) if combined_catalog else None,
        "model_count": inspection.model_count,
        "n_wave": inspection.n_wave,
        "lambda_min": inspection.lambda_min,
        "lambda_max": inspection.lambda_max,
        "lambda_step": inspection.lambda_step,
        "flux_storage_dtype": "float32",
        "max_abs_float32_roundoff": max_abs_err,
        "max_rel_float32_roundoff": max_rel_err,
        "float32_rtol": FLOAT32_RTOL,
        "readback_ok": readback_ok,
        "raw_deleted": False,
        "validated_at_utc": datetime.now(UTC).isoformat(),
    }

    if not readback_ok:
        raise ValueError(f"{source_path.name}: readback validation failed")

    if delete_raw_after_validate:
        source_path.unlink()
        validation["raw_deleted"] = True
        validation["deleted_at_utc"] = datetime.now(UTC).isoformat()

    validation_path = validation_dir / f"{group_name}.json"
    validation_path.write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    return validation


def select_source_files(
    raw_dir: Path, patterns: list[str], limit: int | None
) -> list[Path]:
    files = sorted(raw_dir.glob("PHOENIX-NewEraV*-LowRes-SPECTRA.Z*.txt"))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--zarr-name", default=DEFAULT_ZARR)
    parser.add_argument("--file", dest="patterns", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--chunk-models", type=int, default=4)
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

    manifest_parquet = write_manifest_parquet(args.manifest, args.processed_dir)
    if manifest_parquet:
        print(f"Wrote manifest catalog: {manifest_parquet}")

    for path in source_files:
        print(f"Converting {path.name}")
        result = convert_source_file(
            path,
            processed_dir=args.processed_dir,
            zarr_name=args.zarr_name,
            chunk_models=args.chunk_models,
            zstd_level=args.zstd_level,
            overwrite=args.overwrite,
            delete_raw_after_validate=args.delete_raw_after_validate,
        )
        print(
            json.dumps(
                {
                    "source_file": result["source_file"],
                    "model_count": result["model_count"],
                    "n_wave": result["n_wave"],
                    "max_rel_float32_roundoff": result["max_rel_float32_roundoff"],
                    "raw_deleted": result["raw_deleted"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
