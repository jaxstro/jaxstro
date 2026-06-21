#!/usr/bin/env python3
"""Convert local TLUSTY flux tar files into processed artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from jaxstro.atmospheres.tlusty import (
    parse_tlusty_float,
    parse_tlusty_flux_filename,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data/atmospheres/tlusty/bstars-2007/vturb-2/raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data/atmospheres/tlusty/processed"
DEFAULT_TLUSTY_ZARR = "tlusty_flux.zarr"
FLOAT32_RTOL = 1.0e-6
FLOAT32_ATOL = float(np.finfo(np.float32).tiny)
_C_NM_S = 2.99792458e17


@dataclass(frozen=True)
class TlustyMember:
    """One TLUSTY gzip member inside a source tar."""

    tar_path: Path
    member_name: str


@dataclass(frozen=True)
class TlustyInspection:
    """Shape metadata discovered before TLUSTY conversion."""

    source_count: int
    n_frequency: int
    frequency_min_hz: float
    frequency_max_hz: float
    wavelength_min_nm: float
    wavelength_max_nm: float


def _load_optional_data_deps():
    try:
        import polars as pl
        import zarr
        from numcodecs import Zstd
    except ImportError as exc:  # pragma: no cover - depends on local extra
        raise SystemExit(
            "Install data-conversion dependencies with: uv run --extra data ..."
        ) from exc
    return pl, zarr, Zstd


def _sample_rows(model_count: int) -> tuple[int, ...]:
    if model_count == 1:
        return (0,)
    if model_count == 2:
        return (0, 1)
    return (0, model_count // 2, model_count - 1)


def _discover_members(
    tar_paths: list[Path], limit: int | None = None
) -> list[TlustyMember]:
    members: list[TlustyMember] = []
    for tar_path in tar_paths:
        with tarfile.open(tar_path) as archive:
            for member in archive.getmembers():
                if member.name.endswith(".flux.gz"):
                    parse_tlusty_flux_filename(member.name)
                    members.append(
                        TlustyMember(tar_path=tar_path, member_name=member.name)
                    )
    members = sorted(
        members,
        key=lambda item: (
            parse_tlusty_flux_filename(item.member_name).prefix,
            parse_tlusty_flux_filename(item.member_name).teff,
            parse_tlusty_flux_filename(item.member_name).logg,
            parse_tlusty_flux_filename(item.member_name).vturb_km_s,
            parse_tlusty_flux_filename(item.member_name).cn_altered,
            item.member_name,
        ),
    )
    return members[:limit] if limit is not None else members


def _read_member(member: TlustyMember) -> tuple[np.ndarray, np.ndarray]:
    frequency: list[float] = []
    flux: list[float] = []
    with tarfile.open(member.tar_path) as archive:
        extracted = archive.extractfile(member.member_name)
        if extracted is None:
            raise ValueError(f"{member.member_name}: could not extract from tar")
        with gzip.GzipFile(fileobj=extracted) as handle:
            for raw_line in handle:
                parts = raw_line.decode("ascii").split()
                if not parts:
                    continue
                if len(parts) != 2:
                    raise ValueError(f"{member.member_name}: expected two columns")
                frequency.append(parse_tlusty_float(parts[0]))
                flux.append(parse_tlusty_float(parts[1]))
    if not frequency:
        raise ValueError(f"{member.member_name}: no flux rows found")
    return np.asarray(frequency, dtype=np.float64), np.asarray(flux, dtype=np.float64)


def _inspect_members(
    members: list[TlustyMember],
) -> tuple[TlustyInspection, np.ndarray]:
    if not members:
        raise ValueError("No TLUSTY flux members selected")
    frequency, _ = _read_member(members[0])
    for member in members[1:]:
        other, _ = _read_member(member)
        if other.shape != frequency.shape or not np.allclose(other, frequency):
            raise ValueError("Selected TLUSTY flux files do not share a frequency grid")
    wavelength_nm = _C_NM_S / frequency
    return (
        TlustyInspection(
            source_count=len(members),
            n_frequency=int(frequency.size),
            frequency_min_hz=float(np.min(frequency)),
            frequency_max_hz=float(np.max(frequency)),
            wavelength_min_nm=float(np.min(wavelength_nm)),
            wavelength_max_nm=float(np.max(wavelength_nm)),
        ),
        frequency,
    )


def _write_catalog_fragment(records: list[dict[str, Any]], output_path: Path) -> None:
    pl, _, _ = _load_optional_data_deps()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(records).write_parquet(output_path, compression="zstd")


def rebuild_catalog(processed_dir: Path) -> Path | None:
    pl, _, _ = _load_optional_data_deps()
    fragment_dir = processed_dir / "catalog_fragments"
    fragments = sorted(fragment_dir.glob("*.parquet"))
    if not fragments:
        return None
    catalog = pl.concat([pl.read_parquet(path) for path in fragments]).sort(
        ["dataset", "prefix", "vturb_km_s", "cn_altered", "teff", "logg", "filename"]
    )
    output_path = processed_dir / "catalog.parquet"
    catalog.write_parquet(output_path, compression="zstd")
    return output_path


def convert_tlusty_flux_tars(
    tar_paths: list[Path],
    *,
    processed_dir: Path,
    dataset: str,
    zarr_name: str = DEFAULT_TLUSTY_ZARR,
    group_name: str = "spectra",
    chunk_models: int = 8,
    zstd_level: int = 3,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert TLUSTY flux tar members into local Zarr/Parquet artifacts."""
    pl, zarr, Zstd = _load_optional_data_deps()
    _ = pl
    tar_paths = [Path(path).expanduser() for path in tar_paths]
    members = _discover_members(tar_paths, limit=limit)
    inspection, frequency = _inspect_members(members)

    processed_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = processed_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = processed_dir / zarr_name
    store = zarr.open_group(zarr_path, mode="a", zarr_format=2)
    store.attrs.update(
        {
            "source": "TLUSTY flux spectra",
            "frequency_unit": "Hz",
            "flux_unit": "erg s-1 cm-2 Hz-1",
            "flux_storage_dtype": "float32",
            "wavelength_metadata_policy": (
                "Wavelength ranges are diagnostic coverage metadata derived from "
                "frequency_hz as c/nu; released frequency_hz remains canonical."
            ),
            "created_by": "scripts/convert_tlusty_flux.py",
        }
    )

    compressor = Zstd(level=zstd_level)
    if "frequency_hz" in store:
        existing = np.asarray(store["frequency_hz"][:], dtype=np.float64)
        if existing.shape != frequency.shape or not np.allclose(existing, frequency):
            raise ValueError("Existing TLUSTY frequency grid differs from input")
    else:
        store.create_array(
            "frequency_hz",
            data=frequency,
            chunks=(inspection.n_frequency,),
            compressor=compressor,
            overwrite=True,
        )

    if group_name in store:
        if not overwrite:
            raise FileExistsError(
                f"{group_name} already exists in {zarr_path}; pass --overwrite"
            )
        del store[group_name]
    group = store.create_group(group_name)
    group.attrs.update(
        {
            "converted_at_utc": datetime.now(UTC).isoformat(),
            "dataset": dataset,
            "source_count": inspection.source_count,
        }
    )

    chunk0 = min(chunk_models, inspection.source_count)
    flux_array = group.create_array(
        "flux_fnu",
        shape=(inspection.source_count, inspection.n_frequency),
        chunks=(chunk0, inspection.n_frequency),
        dtype="float32",
        compressor=compressor,
        overwrite=True,
    )
    for name in ["teff", "logg", "vturb_km_s"]:
        group.create_array(
            name,
            shape=(inspection.source_count,),
            chunks=(min(chunk_models * 16, inspection.source_count),),
            dtype="float64",
            compressor=compressor,
            overwrite=True,
        )

    sha256_by_tar: dict[str, str] = {}
    for tar_path in tar_paths:
        sha256 = hashlib.sha256()
        with tar_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha256.update(chunk)
        sha256_by_tar[str(tar_path)] = sha256.hexdigest()

    sample_rows = set(_sample_rows(inspection.source_count))
    sample_flux64: dict[int, np.ndarray] = {}
    catalog_records: list[dict[str, Any]] = []
    max_abs_err = 0.0
    max_rel_err = 0.0
    for row, member in enumerate(members):
        metadata = parse_tlusty_flux_filename(member.member_name)
        _, flux64 = _read_member(member)
        flux32 = flux64.astype(np.float32)
        flux_array[row, :] = flux32
        reread_for_error = flux32.astype(np.float64)
        abs_err = np.max(np.abs(reread_for_error - flux64))
        denom = np.maximum(np.abs(flux64), np.finfo(np.float32).tiny)
        rel_err = np.max(np.abs(reread_for_error - flux64) / denom)
        max_abs_err = max(max_abs_err, float(abs_err))
        max_rel_err = max(max_rel_err, float(rel_err))

        for name, value in (
            ("teff", metadata.teff),
            ("logg", metadata.logg),
            ("vturb_km_s", metadata.vturb_km_s),
        ):
            group[name][row] = value

        record = asdict(metadata)
        record.update(
            {
                "dataset": dataset,
                "source_tar": str(member.tar_path),
                "source_member": member.member_name,
                "source_tar_sha256": sha256_by_tar[str(member.tar_path)],
                "n_frequency": inspection.n_frequency,
                "frequency_min_hz": inspection.frequency_min_hz,
                "frequency_max_hz": inspection.frequency_max_hz,
                "frequency_unit": "Hz",
                "wavelength_min": inspection.wavelength_min_nm,
                "wavelength_max": inspection.wavelength_max_nm,
                "wavelength_unit": "nm",
                "flux_unit": "erg s-1 cm-2 Hz-1",
                "zarr_group": group_name,
                "zarr_row": row,
            }
        )
        catalog_records.append(record)
        if row in sample_rows:
            sample_flux64[row] = flux64

    readback_ok = True
    for row, expected64 in sample_flux64.items():
        stored = flux_array[row, :].astype(np.float64)
        if not np.allclose(stored, expected64, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL):
            readback_ok = False
            break
    if not readback_ok:
        raise ValueError("TLUSTY readback validation failed")

    catalog_fragment = processed_dir / "catalog_fragments" / f"{group_name}.parquet"
    _write_catalog_fragment(catalog_records, catalog_fragment)
    combined_catalog = rebuild_catalog(processed_dir)
    validation = {
        "dataset": dataset,
        "source_count": inspection.source_count,
        "source_tars": [str(path) for path in tar_paths],
        "source_tar_sha256": sha256_by_tar,
        "zarr_store": str(zarr_path),
        "zarr_group": group_name,
        "catalog_fragment": str(catalog_fragment),
        "combined_catalog": str(combined_catalog) if combined_catalog else None,
        "n_frequency": inspection.n_frequency,
        "frequency_min_hz": inspection.frequency_min_hz,
        "frequency_max_hz": inspection.frequency_max_hz,
        "wavelength_min_nm": inspection.wavelength_min_nm,
        "wavelength_max_nm": inspection.wavelength_max_nm,
        "flux_storage_dtype": "float32",
        "max_abs_float32_roundoff": max_abs_err,
        "max_rel_float32_roundoff": max_rel_err,
        "readback_ok": readback_ok,
        "raw_deleted": False,
        "validated_at_utc": datetime.now(UTC).isoformat(),
    }
    validation_path = validation_dir / f"{group_name}.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--zarr-name", default=DEFAULT_TLUSTY_ZARR)
    parser.add_argument("--group-name", default="spectra")
    parser.add_argument("--chunk-models", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tar_paths = sorted(args.raw_dir.glob("*flux*.tar"))
    members = _discover_members(tar_paths, limit=args.limit)
    if args.dry_run:
        print(f"Would convert {len(members)} TLUSTY flux spectra")
        for member in members[:20]:
            print(f"{member.tar_path}:{member.member_name}")
        return

    result = convert_tlusty_flux_tars(
        tar_paths,
        processed_dir=args.processed_dir,
        dataset=args.dataset,
        zarr_name=args.zarr_name,
        group_name=args.group_name,
        chunk_models=args.chunk_models,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "dataset": result["dataset"],
                "source_count": result["source_count"],
                "n_frequency": result["n_frequency"],
                "readback_ok": result["readback_ok"],
                "raw_deleted": result["raw_deleted"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
