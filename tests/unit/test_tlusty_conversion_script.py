"""Tests for the local TLUSTY flux data-conversion script."""

from __future__ import annotations

import gzip
import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("polars")
pytest.importorskip("zarr")


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "convert_tlusty_flux.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_tlusty_flux", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gzip_payload(offset: float) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as handle:
        handle.write(
            "\n".join(
                [
                    f"3.0D+15 {1.0 + offset:.4f}D-10",
                    f"2.0D+15 {2.0 + offset:.4f}-100",
                    f"1.0D+15 {3.0 + offset:.4f}D-08",
                    "",
                ]
            ).encode("ascii")
        )
    return buffer.getvalue()


def _add_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def test_converter_writes_processed_artifact_without_deleting_tar(tmp_path):
    converter = _load_converter()
    raw_tar = tmp_path / "BCflux_v10.tar"
    with tarfile.open(raw_tar, "w") as archive:
        _add_member(archive, "BC15000g175v10.flux.gz", _gzip_payload(0.0))
        _add_member(archive, "BC15000g200v10CN.flux.gz", _gzip_payload(1.0))

    processed_dir = tmp_path / "processed"
    result = converter.convert_tlusty_flux_tars(
        [raw_tar],
        processed_dir=processed_dir,
        dataset="tlusty_bstar_synthetic",
        chunk_models=1,
    )

    assert raw_tar.exists()
    assert result["raw_deleted"] is False
    assert result["source_count"] == 2
    assert result["n_frequency"] == 3
    assert result["readback_ok"] is True
    assert (processed_dir / "catalog.parquet").exists()

    import polars as pl
    import zarr

    catalog = pl.read_parquet(processed_dir / "catalog.parquet").sort("logg")
    assert catalog["prefix"].to_list() == ["BC", "BC"]
    assert catalog["cn_altered"].to_list() == [False, True]
    assert catalog["flux_unit"].unique().to_list() == ["erg s-1 cm-2 Hz-1"]
    assert catalog["frequency_unit"].unique().to_list() == ["Hz"]

    subgroup = catalog["zarr_subgroup"].unique().item()
    root = zarr.open_group(processed_dir / converter.DEFAULT_TLUSTY_ZARR, mode="r")
    group = root["spectra"][subgroup]
    np.testing.assert_allclose(group["frequency_hz"][:], [3.0e15, 2.0e15, 1.0e15])
    flux = group["flux_fnu"]
    assert flux.dtype == np.dtype("float32")
    assert flux.shape == (2, 3)
    np.testing.assert_allclose(flux[1, :], [2.0e-10, 0.0, 4.0e-08])
