#!/usr/bin/env python3
"""Convert local TLUSTY flux tar files into processed artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from contextlib import ExitStack
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
    with tarfile.open(member.tar_path) as archive:
        return _read_member_from_archive(archive, member.member_name)


def _read_frequency_from_archive(
    archive: tarfile.TarFile,
    member_name: str,
) -> np.ndarray:
    frequency: list[float] = []
    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise ValueError(f"{member_name}: could not extract from tar")
    with gzip.GzipFile(fileobj=extracted) as handle:
        for raw_line in handle:
            parts = raw_line.decode("ascii").split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"{member_name}: expected two columns")
            frequency.append(parse_tlusty_float(parts[0]))
    if not frequency:
        raise ValueError(f"{member_name}: no flux rows found")
    return np.asarray(frequency, dtype=np.float64)


def _read_member_from_archive(
    archive: tarfile.TarFile,
    member_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    frequency: list[float] = []
    flux: list[float] = []
    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise ValueError(f"{member_name}: could not extract from tar")
    with gzip.GzipFile(fileobj=extracted) as handle:
        for raw_line in handle:
            parts = raw_line.decode("ascii").split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"{member_name}: expected two columns")
            frequency.append(parse_tlusty_float(parts[0]))
            flux.append(parse_tlusty_float(parts[1]))
    if not frequency:
        raise ValueError(f"{member_name}: no flux rows found")
    return np.asarray(frequency, dtype=np.float64), np.asarray(flux, dtype=np.float64)


def _inspect_members(
    members: list[TlustyMember],
) -> tuple[TlustyInspection, np.ndarray]:
    if not members:
        raise ValueError("No TLUSTY flux members selected")
    frequency, _ = _read_member(members[0])
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


def _group_members_by_prefix(
    members: list[TlustyMember],
) -> dict[str, list[TlustyMember]]:
    grouped: dict[str, list[TlustyMember]] = {}
    for member in members:
        prefix = parse_tlusty_flux_filename(member.member_name).prefix
        grouped.setdefault(prefix, []).append(member)
    return dict(sorted(grouped.items()))


def _discover_grid_subgroups(
    members: list[TlustyMember],
    tar_paths: list[Path],
) -> tuple[dict[str, list[TlustyMember]], dict[str, np.ndarray]]:
    members_by_grid: dict[str, list[TlustyMember]] = {}
    frequency_by_grid: dict[str, np.ndarray] = {}
    with ExitStack() as stack:
        archives = {
            tar_path: stack.enter_context(tarfile.open(tar_path))
            for tar_path in tar_paths
        }
        for member in members:
            frequency = _read_frequency_from_archive(
                archives[member.tar_path], member.member_name
            )
            grid_name = _matching_grid_name(frequency_by_grid, frequency)
            if grid_name is None:
                grid_name = f"grid{len(frequency_by_grid):03d}"
                frequency_by_grid[grid_name] = frequency
                members_by_grid[grid_name] = []
            members_by_grid[grid_name].append(member)
    return members_by_grid, frequency_by_grid


def _matching_grid_name(
    frequency_by_grid: dict[str, np.ndarray],
    frequency: np.ndarray,
) -> str | None:
    for grid_name, existing in frequency_by_grid.items():
        if existing.shape == frequency.shape and np.allclose(existing, frequency):
            return grid_name
    return None


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
    dataset: str,
    processed_dir: Path,
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
    datasets[dataset] = {
        "state": "processed",
        "n_flux_files": validation["source_count"],
        "teff_min": min(record["teff"] for record in catalog_records),
        "teff_max": max(record["teff"] for record in catalog_records),
        "logg_min": min(record["logg"] for record in catalog_records),
        "logg_max": max(record["logg"] for record in catalog_records),
        "vturb_values_km_s": _unique_sorted(catalog_records, "vturb_km_s"),
        "prefixes": _unique_sorted(catalog_records, "prefix"),
        "cn_altered": _unique_sorted(catalog_records, "cn_altered"),
        "frequency_unit": "Hz",
        "frequency_min_hz": validation["frequency_min_hz"],
        "frequency_max_hz": validation["frequency_max_hz"],
        "wavelength_unit": "nm",
        "wavelength_min": validation["wavelength_min_nm"],
        "wavelength_max": validation["wavelength_max_nm"],
        "flux_unit": "erg s-1 cm-2 Hz-1",
        "processed_path": _repo_relative(processed_dir),
        "catalog_path": _repo_relative(catalog_path),
        "zarr_path": _repo_relative(zarr_path),
        "zarr_group": validation["zarr_group"],
        "validation_path": _repo_relative(validation_path),
        "source_tars": [
            _repo_relative(Path(path)) for path in validation["source_tars"]
        ],
        "source_tar_sha256": validation["source_tar_sha256"],
        "raw_deleted_after_validate": False,
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
    chunk_models: int = 32,
    zstd_level: int = 3,
    overwrite: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert TLUSTY flux tar members into local Zarr/Parquet artifacts."""
    pl, zarr, Zstd = _load_optional_data_deps()
    _ = pl
    tar_paths = [Path(path).expanduser() for path in tar_paths]
    members = _discover_members(tar_paths, limit=limit)
    members_by_grid, frequency_by_grid = _discover_grid_subgroups(members, tar_paths)
    grid_inspections: dict[str, TlustyInspection] = {}
    for grid_name, grid_members in members_by_grid.items():
        frequency = frequency_by_grid[grid_name]
        wavelength_nm = _C_NM_S / frequency
        grid_inspections[grid_name] = TlustyInspection(
            source_count=len(grid_members),
            n_frequency=int(frequency.size),
            frequency_min_hz=float(np.min(frequency)),
            frequency_max_hz=float(np.max(frequency)),
            wavelength_min_nm=float(np.min(wavelength_nm)),
            wavelength_max_nm=float(np.max(wavelength_nm)),
        )
    source_count = len(members)
    frequency_min_hz = min(
        inspection.frequency_min_hz for inspection in grid_inspections.values()
    )
    frequency_max_hz = max(
        inspection.frequency_max_hz for inspection in grid_inspections.values()
    )
    wavelength_min_nm = min(
        inspection.wavelength_min_nm for inspection in grid_inspections.values()
    )
    wavelength_max_nm = max(
        inspection.wavelength_max_nm for inspection in grid_inspections.values()
    )
    n_frequency_by_subgroup = {
        grid_name: inspection.n_frequency
        for grid_name, inspection in grid_inspections.items()
    }
    n_frequency: int | dict[str, int]
    if len(set(n_frequency_by_subgroup.values())) == 1:
        n_frequency = next(iter(n_frequency_by_subgroup.values()))
    else:
        n_frequency = n_frequency_by_subgroup

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
            "source_count": source_count,
            "storage_layout": "grid subgroups with subgroup-local frequency_hz arrays",
        }
    )

    sha256_by_tar: dict[str, str] = {}
    for tar_path in tar_paths:
        sha256 = hashlib.sha256()
        with tar_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha256.update(chunk)
        sha256_by_tar[str(tar_path)] = sha256.hexdigest()

    sample_flux64: dict[tuple[str, int], np.ndarray] = {}
    flux_arrays: dict[str, Any] = {}
    catalog_records: list[dict[str, Any]] = []
    max_abs_err = 0.0
    max_rel_err = 0.0
    with ExitStack() as stack:
        archives = {
            tar_path: stack.enter_context(tarfile.open(tar_path))
            for tar_path in tar_paths
        }
        for grid_name, grid_members in members_by_grid.items():
            inspection = grid_inspections[grid_name]
            frequency = frequency_by_grid[grid_name]
            grid_group = group.create_group(grid_name)
            grid_group.attrs.update(
                {
                    "grid_name": grid_name,
                    "prefixes": sorted(
                        {
                            parse_tlusty_flux_filename(member.member_name).prefix
                            for member in grid_members
                        }
                    ),
                    "source_count": inspection.source_count,
                    "frequency_unit": "Hz",
                    "flux_unit": "erg s-1 cm-2 Hz-1",
                }
            )
            grid_group.create_array(
                "frequency_hz",
                data=frequency,
                chunks=(inspection.n_frequency,),
                compressor=compressor,
                overwrite=True,
            )
            chunk0 = min(chunk_models, inspection.source_count)
            flux_array = grid_group.create_array(
                "flux_fnu",
                shape=(inspection.source_count, inspection.n_frequency),
                chunks=(chunk0, inspection.n_frequency),
                dtype="float32",
                compressor=compressor,
                overwrite=True,
            )
            flux_arrays[grid_name] = flux_array
            coordinate_arrays = {}
            for name in ["teff", "logg", "vturb_km_s"]:
                coordinate_arrays[name] = grid_group.create_array(
                    name,
                    shape=(inspection.source_count,),
                    chunks=(min(chunk_models * 16, inspection.source_count),),
                    dtype="float64",
                    compressor=compressor,
                    overwrite=True,
                )

            sample_rows = set(_sample_rows(inspection.source_count))
            for start in range(0, inspection.source_count, chunk_models):
                chunk_members = grid_members[start : start + chunk_models]
                chunk_flux32 = np.empty(
                    (len(chunk_members), inspection.n_frequency), dtype=np.float32
                )
                chunk_coords: dict[str, list[float]] = {
                    "teff": [],
                    "logg": [],
                    "vturb_km_s": [],
                }
                for offset, member in enumerate(chunk_members):
                    row = start + offset
                    metadata = parse_tlusty_flux_filename(member.member_name)
                    member_frequency, flux64 = _read_member_from_archive(
                        archives[member.tar_path], member.member_name
                    )
                    if member_frequency.shape != frequency.shape or not np.allclose(
                        member_frequency, frequency
                    ):
                        raise ValueError(
                            "Selected TLUSTY flux files do not share a frequency grid"
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
                    chunk_coords["vturb_km_s"].append(metadata.vturb_km_s)

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
                            "zarr_subgroup": grid_name,
                            "zarr_row": row,
                        }
                    )
                    catalog_records.append(record)
                    if row in sample_rows:
                        sample_flux64[(grid_name, row)] = flux64

                stop = start + len(chunk_members)
                flux_array[start:stop, :] = chunk_flux32
                for name, values in chunk_coords.items():
                    coordinate_arrays[name][start:stop] = np.asarray(
                        values, dtype=np.float64
                    )

    readback_ok = True
    for (grid_name, row), expected64 in sample_flux64.items():
        stored = flux_arrays[grid_name][row, :].astype(np.float64)
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
        "source_count": source_count,
        "source_tars": [str(path) for path in tar_paths],
        "source_tar_sha256": sha256_by_tar,
        "zarr_store": str(zarr_path),
        "zarr_group": group_name,
        "zarr_subgroups": sorted(members_by_grid),
        "catalog_fragment": str(catalog_fragment),
        "combined_catalog": str(combined_catalog) if combined_catalog else None,
        "n_frequency": n_frequency,
        "n_frequency_by_subgroup": n_frequency_by_subgroup,
        "source_count_by_subgroup": {
            grid_name: len(grid_members)
            for grid_name, grid_members in members_by_grid.items()
        },
        "frequency_min_hz": frequency_min_hz,
        "frequency_max_hz": frequency_max_hz,
        "wavelength_min_nm": wavelength_min_nm,
        "wavelength_max_nm": wavelength_max_nm,
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
        dataset=dataset,
        processed_dir=processed_dir,
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
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--zarr-name", default=DEFAULT_TLUSTY_ZARR)
    parser.add_argument("--group-name", default="spectra")
    parser.add_argument("--chunk-models", type=int, default=32)
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
