#!/usr/bin/env python3
"""Convert local Sonora 2024 spectra zip files into processed artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from jaxstro.atmospheres.sonora import parse_sonora_2024_filename

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ZIP = REPO_ROOT / "data/atmospheres/sonora/2024/raw/spectra.zip"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data/atmospheres/sonora/2024/processed"
DEFAULT_SONORA_ZARR = "sonora_2024.zarr"
FLOAT32_RTOL = 1.0e-6
FLOAT32_ATOL = 0.0


@dataclass(frozen=True)
class SonoraInspection:
    """Shape metadata discovered before Sonora conversion."""

    source_count: int
    n_wave: int
    wavelength_min: float
    wavelength_max: float


@dataclass(frozen=True)
class SonoraSourceDiscovery:
    """Valid and skipped Sonora source members discovered in a zip archive."""

    valid_members: tuple[str, ...]
    skipped_members: tuple[str, ...]

    @property
    def valid_count(self) -> int:
        return len(self.valid_members)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_members)


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


def _source_members(raw_zip: Path, limit: int | None = None) -> list[str]:
    discovery = discover_sonora_sources(raw_zip, limit=limit)
    return list(discovery.valid_members)


def _is_macos_resource_fork_member(name: str) -> bool:
    return name.startswith("__MACOSX/") or Path(name).name.startswith("._")


def discover_sonora_sources(
    raw_zip: Path,
    *,
    limit: int | None = None,
) -> SonoraSourceDiscovery:
    """Discover valid Sonora spectra and archive members intentionally skipped."""
    with zipfile.ZipFile(raw_zip) as archive:
        names = archive.namelist()

    parsed_members = []
    skipped_members = []
    for name in names:
        if not name.endswith(".spec"):
            continue
        if _is_macos_resource_fork_member(name):
            skipped_members.append(name)
            continue
        try:
            parsed_members.append((name, parse_sonora_2024_filename(name)))
        except ValueError:
            skipped_members.append(name)

    sorted_members = sorted(
        parsed_members,
        key=lambda name: (
            name[1].m_h,
            name[1].c_o,
            name[1].teff,
            name[1].logg,
            name[1].cloud_label,
            name[0],
        ),
    )
    valid_members = tuple(name for name, _ in sorted_members)
    if limit is not None:
        valid_members = valid_members[:limit]
    return SonoraSourceDiscovery(
        valid_members=valid_members,
        skipped_members=tuple(sorted(skipped_members)),
    )


def _read_spectrum_member(
    archive: zipfile.ZipFile,
    member: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    header: list[str] = []
    wavelength: list[float] = []
    flux: list[float] = []
    with archive.open(member) as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii").strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    wavelength.append(float(parts[0]))
                    flux.append(float(parts[1]))
                    continue
                except ValueError:
                    pass
            header.append(line)
    if not wavelength:
        raise ValueError(f"{member}: no numeric spectrum rows found")
    return (
        np.asarray(wavelength, dtype=np.float64),
        np.asarray(flux, dtype=np.float64),
        tuple(header),
    )


def _inspect_members(raw_zip: Path, members: list[str]) -> SonoraInspection:
    if not members:
        raise ValueError("No Sonora spectra selected")
    with zipfile.ZipFile(raw_zip) as archive:
        wavelength, _, _ = _read_spectrum_member(archive, members[0])
    return SonoraInspection(
        source_count=len(members),
        n_wave=int(wavelength.size),
        wavelength_min=float(np.min(wavelength)),
        wavelength_max=float(np.max(wavelength)),
    )


def _write_catalog_fragment(records: list[dict[str, Any]], output_path: Path) -> None:
    pl, _, _ = _load_optional_data_deps()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(records).write_parquet(output_path, compression="zstd")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _unique_sorted(records: list[dict[str, Any]], key: str) -> list[Any]:
    return sorted({record[key] for record in records})


def _update_local_staging_manifest(
    *,
    processed_dir: Path,
    raw_zip: Path,
    zarr_path: Path,
    catalog_path: Path | None,
    validation_path: Path,
    validation: dict[str, Any],
    catalog_records: list[dict[str, Any]],
) -> Path | None:
    atmospheres_root = REPO_ROOT / "data" / "atmospheres"
    manifest_path = atmospheres_root / "local-staging-manifest.json"
    if catalog_path is None or not manifest_path.exists():
        return None
    if not _is_relative_to(processed_dir, atmospheres_root):
        return None

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = payload.setdefault("datasets", {})
    datasets["sonora_2024"] = {
        "state": "processed",
        "n_spectra": validation["source_count"],
        "valid_count": validation["valid_count"],
        "skipped_count": validation["skipped_count"],
        "teff_min": min(record["teff"] for record in catalog_records),
        "teff_max": max(record["teff"] for record in catalog_records),
        "logg_min": min(record["logg"] for record in catalog_records),
        "logg_max": max(record["logg"] for record in catalog_records),
        "m_h_values": _unique_sorted(catalog_records, "m_h"),
        "cloud_labels": _unique_sorted(catalog_records, "cloud_label"),
        "c_o_values": _unique_sorted(catalog_records, "c_o"),
        "wavelength_unit": "micron",
        "wavelength_min": validation["wavelength_min"],
        "wavelength_max": validation["wavelength_max"],
        "flux_unit": "W/m2/m",
        "processed_path": _repo_relative(processed_dir),
        "catalog_path": _repo_relative(catalog_path),
        "zarr_path": _repo_relative(zarr_path),
        "validation_path": _repo_relative(validation_path),
        "raw_path": _repo_relative(raw_zip.parent),
        "source_zip": _repo_relative(raw_zip),
        "source_zip_sha256": validation["source_zip_sha256"],
        "raw_deleted_after_validate": False,
        "skipped_members": validation["skipped_members"],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def rebuild_catalog(processed_dir: Path) -> Path | None:
    pl, _, _ = _load_optional_data_deps()
    fragment_dir = processed_dir / "catalog_fragments"
    fragments = sorted(fragment_dir.glob("*.parquet"))
    if not fragments:
        return None
    catalog = pl.concat([pl.read_parquet(path) for path in fragments]).sort(
        ["m_h", "c_o", "cloud_label", "teff", "logg", "filename"]
    )
    output_path = processed_dir / "catalog.parquet"
    catalog.write_parquet(output_path, compression="zstd")
    return output_path


def convert_sonora_zip(
    raw_zip: Path,
    *,
    processed_dir: Path,
    zarr_name: str = DEFAULT_SONORA_ZARR,
    group_name: str = "spectra",
    chunk_models: int = 32,
    zstd_level: int = 3,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert a Sonora spectra zip into local processed Zarr/Parquet artifacts."""
    pl, zarr, Zstd = _load_optional_data_deps()
    _ = pl

    raw_zip = Path(raw_zip).expanduser()
    discovery = discover_sonora_sources(raw_zip, limit=limit)
    members = list(discovery.valid_members)
    inspection = _inspect_members(raw_zip, members)

    processed_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = processed_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    zarr_path = processed_dir / zarr_name
    store = zarr.open_group(zarr_path, mode="a", zarr_format=2)
    store.attrs.update(
        {
            "source": "Sonora 2024 Diamondback spectra",
            "wavelength_unit": "micron",
            "flux_unit": "W/m2/m",
            "flux_storage_dtype": "float32",
            "precision_policy": (
                "Raw text values are parsed as float64 for validation; spectra are "
                "stored as float32 and wavelength as float64."
            ),
            "created_by": "scripts/convert_sonora_2024.py",
        }
    )

    compressor = Zstd(level=zstd_level)
    with zipfile.ZipFile(raw_zip) as archive:
        wavelength, _, _ = _read_spectrum_member(archive, members[0])
    if "wavelength" in store:
        existing = np.asarray(store["wavelength"][:], dtype=np.float64)
        if existing.shape != wavelength.shape or not np.allclose(existing, wavelength):
            raise ValueError("Existing Sonora wavelength grid differs from input")
    else:
        store.create_array(
            "wavelength",
            data=wavelength,
            chunks=(inspection.n_wave,),
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
            "source_zip": str(raw_zip),
            "source_count": inspection.source_count,
            "skipped_source_count": discovery.skipped_count,
        }
    )

    chunk0 = min(chunk_models, inspection.source_count)
    flux_array = group.create_array(
        "flux",
        shape=(inspection.source_count, inspection.n_wave),
        chunks=(chunk0, inspection.n_wave),
        dtype="float32",
        compressor=compressor,
        overwrite=True,
    )
    coordinate_arrays = {}
    for name in ["teff", "logg", "g_m_s2", "m_h", "c_o"]:
        coordinate_arrays[name] = group.create_array(
            name,
            shape=(inspection.source_count,),
            chunks=(min(chunk_models * 16, inspection.source_count),),
            dtype="float64",
            compressor=compressor,
            overwrite=True,
        )

    sample_rows = set(_sample_rows(inspection.source_count))
    sample_flux64: dict[int, np.ndarray] = {}
    catalog_records: list[dict[str, Any]] = []
    max_abs_err = 0.0
    max_rel_err = 0.0
    sha256 = hashlib.sha256()
    with raw_zip.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)

    with zipfile.ZipFile(raw_zip) as archive:
        for start in range(0, inspection.source_count, chunk_models):
            chunk_members = members[start : start + chunk_models]
            chunk_flux32 = np.empty(
                (len(chunk_members), inspection.n_wave), dtype=np.float32
            )
            chunk_coords: dict[str, list[float]] = {
                "teff": [],
                "logg": [],
                "g_m_s2": [],
                "m_h": [],
                "c_o": [],
            }
            for offset, member in enumerate(chunk_members):
                row = start + offset
                metadata = parse_sonora_2024_filename(member)
                wavelength64, flux64, header = _read_spectrum_member(archive, member)
                if wavelength64.shape != wavelength.shape or not np.allclose(
                    wavelength64, wavelength
                ):
                    raise ValueError(
                        "Selected Sonora spectra do not share a wavelength grid"
                    )
                flux32 = flux64.astype(np.float32)
                chunk_flux32[offset, :] = flux32
                reread_for_error = flux32.astype(np.float64)
                abs_err = np.max(np.abs(reread_for_error - flux64))
                denom = np.maximum(np.abs(flux64), np.finfo(np.float32).tiny)
                rel_err = np.max(np.abs(reread_for_error - flux64) / denom)
                max_abs_err = max(max_abs_err, float(abs_err))
                max_rel_err = max(max_rel_err, float(rel_err))

                chunk_coords["teff"].append(metadata.teff)
                chunk_coords["logg"].append(metadata.logg)
                chunk_coords["g_m_s2"].append(metadata.g_m_s2)
                chunk_coords["m_h"].append(metadata.m_h)
                chunk_coords["c_o"].append(metadata.c_o)

                record = asdict(metadata)
                record.update(
                    {
                        "source_zip": str(raw_zip),
                        "source_member": member,
                        "source_zip_sha256": sha256.hexdigest(),
                        "n_wave": inspection.n_wave,
                        "wavelength_min": inspection.wavelength_min,
                        "wavelength_max": inspection.wavelength_max,
                        "wavelength_unit": "micron",
                        "flux_unit": "W/m2/m",
                        "header": "\n".join(header),
                        "zarr_group": group_name,
                        "zarr_row": row,
                    }
                )
                catalog_records.append(record)
                if row in sample_rows:
                    sample_flux64[row] = flux64

            stop = start + len(chunk_members)
            flux_array[start:stop, :] = chunk_flux32
            for name, values in chunk_coords.items():
                coordinate_arrays[name][start:stop] = np.asarray(
                    values, dtype=np.float64
                )

    readback_ok = True
    for row, expected64 in sample_flux64.items():
        stored = flux_array[row, :].astype(np.float64)
        if not np.allclose(stored, expected64, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL):
            readback_ok = False
            break
    if not readback_ok:
        raise ValueError("Sonora readback validation failed")

    catalog_fragment = processed_dir / "catalog_fragments" / f"{group_name}.parquet"
    _write_catalog_fragment(catalog_records, catalog_fragment)
    combined_catalog = rebuild_catalog(processed_dir)
    validation = {
        "source_zip": str(raw_zip),
        "source_zip_sha256": sha256.hexdigest(),
        "source_count": inspection.source_count,
        "valid_count": discovery.valid_count,
        "skipped_count": discovery.skipped_count,
        "skipped_members": list(discovery.skipped_members),
        "zarr_store": str(zarr_path),
        "zarr_group": group_name,
        "catalog_fragment": str(catalog_fragment),
        "combined_catalog": str(combined_catalog) if combined_catalog else None,
        "n_wave": inspection.n_wave,
        "wavelength_min": inspection.wavelength_min,
        "wavelength_max": inspection.wavelength_max,
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
    manifest_path = _update_local_staging_manifest(
        processed_dir=processed_dir,
        raw_zip=raw_zip,
        zarr_path=zarr_path,
        catalog_path=combined_catalog,
        validation_path=validation_path,
        validation=validation,
        catalog_records=catalog_records,
    )
    if manifest_path is not None:
        validation["local_staging_manifest"] = str(manifest_path)
        validation_path.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-zip", type=Path, default=DEFAULT_RAW_ZIP)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--zarr-name", default=DEFAULT_SONORA_ZARR)
    parser.add_argument("--group-name", default="spectra")
    parser.add_argument("--chunk-models", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    discovery = discover_sonora_sources(args.raw_zip, limit=args.limit)
    members = list(discovery.valid_members)
    if args.dry_run:
        print(
            "Would convert "
            f"valid_count={discovery.valid_count} "
            f"skipped_count={discovery.skipped_count} Sonora spectra"
        )
        for member in discovery.skipped_members:
            print(f"skipped {member}")
        for member in members[:20]:
            print(member)
        return

    result = convert_sonora_zip(
        args.raw_zip,
        processed_dir=args.processed_dir,
        zarr_name=args.zarr_name,
        group_name=args.group_name,
        chunk_models=args.chunk_models,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "source_count": result["source_count"],
                "valid_count": result["valid_count"],
                "skipped_count": result["skipped_count"],
                "n_wave": result["n_wave"],
                "readback_ok": result["readback_ok"],
                "raw_deleted": result["raw_deleted"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
