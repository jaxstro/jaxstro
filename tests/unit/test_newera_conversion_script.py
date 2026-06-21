"""Tests for the local NewEra data-conversion script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("polars")
pytest.importorskip("zarr")


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "convert_newera_lowres.py"
)


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_newera_lowres", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metadata_line(teff: float, logg: float) -> str:
    tokens = [
        "star",
        "BPRP",
        "PHH",
        "20250708",
        "02",
        "PHOENIX1D",
        "0",
        "0.01",
        "4",
        "250.0",
        "2500.0",
        "0.01",
        f"{teff}",
        f"{logg}",
        "1.0",
        "0.0",
        "2.4924e-01",
        "1.3372e-02",
        "7.50",
        "0.00",
        "8.43",
        "7.83",
        "8.69",
        "7.60",
        "7.51",
        "6.34",
        "7.50",
        "0.79",
        "999",
        "999",
        "999",
    ]
    return " ".join(tokens)


def test_converter_writes_zarr_parquet_and_deletes_validated_raw(tmp_path):
    converter = _load_converter()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_file = raw_dir / "PHOENIX-NewEraV3-LowRes-SPECTRA.Z-0.0.txt"
    raw_file.write_text(
        "\n".join(
            [
                _metadata_line(2300.0, 0.0),
                "1.0 2.0 3.0 4.0",
                _metadata_line(2400.0, 0.5),
                "1.5 2.5 3.5 4.5",
                "",
            ]
        ),
        encoding="ascii",
    )

    processed_dir = tmp_path / "processed"
    result = converter.convert_source_file(
        raw_file,
        processed_dir=processed_dir,
        chunk_models=1,
        delete_raw_after_validate=True,
    )

    assert result["raw_deleted"] is True
    assert not raw_file.exists()
    assert result["model_count"] == 2
    assert result["n_wave"] == 4
    assert result["readback_ok"] is True
    assert (processed_dir / "catalog.parquet").exists()

    import zarr

    root = zarr.open_group(processed_dir / converter.DEFAULT_ZARR, mode="r")
    flux = root["files"]["PHOENIXminusNewEraV3minusLowResminusSPECTRA_Zminus0_0"][
        "flux"
    ]
    assert flux.dtype == np.dtype("float32")
    assert flux.shape == (2, 4)
    np.testing.assert_allclose(flux[1, :], np.array([1.5, 2.5, 3.5, 4.5]))
