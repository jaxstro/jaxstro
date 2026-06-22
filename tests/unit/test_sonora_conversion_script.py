"""Tests for the local Sonora 2024 data-conversion script."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("polars")
pytest.importorskip("zarr")


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "convert_sonora_2024.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("convert_sonora_2024", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _spec_text(offset: float) -> str:
    return "\n".join(
        [
            "wavelength (micron) \t flux (W/m2/m)",
            "calculated using grid spacing: synthetic",
            "molecules included: ['h2o']",
            f"250.0 {1.0 + offset}",
            f"249.5 {2.0 + offset}",
            f"249.0 {3.0 + offset}",
            "",
        ]
    )


def _spec_text_with_wavelengths(wavelengths: list[float]) -> str:
    lines = [
        "wavelength (micron) \t flux (W/m2/m)",
        "calculated using grid spacing: synthetic",
        "molecules included: ['h2o']",
    ]
    lines.extend(
        f"{wavelength} {index + 1.0}" for index, wavelength in enumerate(wavelengths)
    )
    return "\n".join([*lines, ""])


def test_source_discovery_skips_macos_resource_fork_members(
    tmp_path, capsys, monkeypatch
):
    converter = _load_converter()
    raw_zip = tmp_path / "spectra.zip"
    with zipfile.ZipFile(raw_zip, "w") as archive:
        archive.writestr("spectra/t900g31nc_m0.0_co1.0.spec", _spec_text(0.0))
        archive.writestr("spectra/t1000g100f2_m0.0_co1.0.spec", _spec_text(1.0))
        archive.writestr("__MACOSX/spectra/._bad.spec", "not a spectrum")

    discovery = converter.discover_sonora_sources(raw_zip)

    assert discovery.valid_count == 2
    assert discovery.skipped_count == 1
    assert discovery.skipped_members == ("__MACOSX/spectra/._bad.spec",)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_sonora_2024.py",
            "--raw-zip",
            str(raw_zip),
            "--dry-run",
        ],
    )
    converter.main()
    output = capsys.readouterr().out
    assert "valid_count=2" in output
    assert "skipped_count=1" in output


def test_converter_writes_processed_artifact_without_deleting_zip(tmp_path):
    converter = _load_converter()
    raw_zip = tmp_path / "spectra.zip"
    with zipfile.ZipFile(raw_zip, "w") as archive:
        archive.writestr("spectra/t900g31nc_m0.0_co1.0.spec", _spec_text(0.0))
        archive.writestr("spectra/t1000g100f2_m0.0_co1.0.spec", _spec_text(1.0))

    processed_dir = tmp_path / "processed"
    result = converter.convert_sonora_zip(
        raw_zip,
        processed_dir=processed_dir,
        chunk_models=1,
    )

    assert raw_zip.exists()
    assert result["raw_deleted"] is False
    assert result["source_count"] == 2
    assert result["n_wave"] == 3
    assert result["readback_ok"] is True
    assert (processed_dir / "catalog.parquet").exists()

    import polars as pl
    import zarr

    catalog = pl.read_parquet(processed_dir / "catalog.parquet").sort("teff")
    assert catalog["g_m_s2"].to_list() == [31.0, 100.0]
    assert catalog["cloud_label"].to_list() == ["nc", "f2"]
    assert catalog["wavelength_unit"].unique().to_list() == ["micron"]
    assert catalog["flux_unit"].unique().to_list() == ["W/m2/m"]

    root = zarr.open_group(processed_dir / converter.DEFAULT_SONORA_ZARR, mode="r")
    np.testing.assert_allclose(root["wavelength"][:], [250.0, 249.5, 249.0])
    flux = root["spectra"]["flux"]
    assert flux.dtype == np.dtype("float32")
    assert flux.shape == (2, 3)
    np.testing.assert_allclose(flux[1, :], [2.0, 3.0, 4.0])


def test_converter_rejects_inconsistent_sonora_wavelength_grid(tmp_path):
    converter = _load_converter()
    raw_zip = tmp_path / "spectra.zip"
    with zipfile.ZipFile(raw_zip, "w") as archive:
        archive.writestr(
            "spectra/t900g31nc_m0.0_co1.0.spec",
            _spec_text_with_wavelengths([250.0, 249.5, 249.0]),
        )
        archive.writestr(
            "spectra/t1000g100f2_m0.0_co1.0.spec",
            _spec_text_with_wavelengths([250.0, 249.0, 248.0]),
        )

    with pytest.raises(ValueError, match="wavelength grid"):
        converter.convert_sonora_zip(
            raw_zip,
            processed_dir=tmp_path / "processed",
        )
