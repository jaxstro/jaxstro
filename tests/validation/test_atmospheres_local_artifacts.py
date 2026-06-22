"""Validation tests for local processed atmosphere artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from jaxstro.atmospheres.library import AtmosphereLibrary

pytest.importorskip("polars")
pytest.importorskip("zarr")


REPO_ROOT = Path(__file__).resolve().parents[2]
ATMOSPHERES_ROOT = REPO_ROOT / "data" / "atmospheres"


@pytest.mark.slow
def test_local_sonora_2024_processed_artifact_is_validated():
    import polars as pl
    import zarr

    raw_zip = ATMOSPHERES_ROOT / "sonora" / "2024" / "raw" / "spectra.zip"
    processed_dir = ATMOSPHERES_ROOT / "sonora" / "2024" / "processed"
    catalog_path = processed_dir / "catalog.parquet"
    zarr_path = processed_dir / "sonora_2024.zarr"
    validation_path = processed_dir / "validation" / "spectra.json"

    assert raw_zip.exists()
    assert catalog_path.exists()
    assert zarr_path.exists()
    assert validation_path.exists()

    catalog = pl.read_parquet(catalog_path)
    assert catalog.height == 1440
    assert catalog["wavelength_unit"].unique().to_list() == ["micron"]
    assert catalog["flux_unit"].unique().to_list() == ["W/m2/m"]
    assert math.isclose(float(catalog["logg"].min()), math.log10(31.0 * 100.0))
    assert math.isclose(float(catalog["logg"].max()), math.log10(3160.0 * 100.0))

    root = zarr.open_group(zarr_path, mode="r")
    flux = root["spectra"]["flux"]
    sample = np.asarray(flux[[0, 720, 1439], :256])
    assert sample.shape == (3, 256)
    assert np.isfinite(sample).all()

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["valid_count"] == 1440
    assert validation["skipped_count"] == 4
    assert validation["readback_ok"] is True
    assert validation["raw_deleted"] is False

    manifest = json.loads(
        (ATMOSPHERES_ROOT / "local-staging-manifest.json").read_text(encoding="utf-8")
    )
    sonora = manifest["datasets"]["sonora_2024"]
    assert sonora["state"] == "processed"
    assert sonora["n_spectra"] == 1440
    assert sonora["catalog_path"] == str(catalog_path.relative_to(REPO_ROOT))
    assert sonora["zarr_path"] == str(zarr_path.relative_to(REPO_ROOT))
    assert math.isclose(sonora["logg_min"], math.log10(31.0 * 100.0))
    assert math.isclose(sonora["logg_max"], math.log10(3160.0 * 100.0))

    coverage = {
        row.dataset: row
        for row in AtmosphereLibrary.from_local(REPO_ROOT / "data").coverage()
    }
    assert coverage["sonora_2024"].state == "processed"
    assert coverage["sonora_2024"].catalog_path == str(catalog_path)
    assert coverage["sonora_2024"].zarr_path == str(zarr_path)
    assert math.isclose(coverage["sonora_2024"].logg_min, math.log10(31.0 * 100.0))


@pytest.mark.slow
def test_local_tlusty_processed_artifacts_are_validated():
    import polars as pl
    import zarr

    expectations = {
        "tlusty_bstar_2007_vturb_2": {
            "group": "bstar_2007_vturb_2",
            "raw_dir": ATMOSPHERES_ROOT / "tlusty" / "bstars-2007" / "vturb-2" / "raw",
            "count": 981,
        },
        "tlusty_bstar_2007_vturb_10_cn": {
            "group": "bstar_2007_vturb_10_cn",
            "raw_dir": ATMOSPHERES_ROOT
            / "tlusty"
            / "bstars-2007"
            / "vturb-10-cn"
            / "raw",
            "count": 551,
        },
        "tlusty_ostar_2002": {
            "group": "ostar_2002",
            "raw_dir": ATMOSPHERES_ROOT / "tlusty" / "ostars-2002" / "raw",
            "count": 690,
        },
    }
    processed_dir = ATMOSPHERES_ROOT / "tlusty" / "processed"
    catalog_path = processed_dir / "catalog.parquet"
    zarr_path = processed_dir / "tlusty_flux.zarr"

    assert catalog_path.exists()
    assert zarr_path.exists()
    catalog = pl.read_parquet(catalog_path)
    root = zarr.open_group(zarr_path, mode="r")

    manifest = json.loads(
        (ATMOSPHERES_ROOT / "local-staging-manifest.json").read_text(encoding="utf-8")
    )
    coverage = {
        row.dataset: row
        for row in AtmosphereLibrary.from_local(REPO_ROOT / "data").coverage()
    }

    for dataset, expected in expectations.items():
        rows = catalog.filter(pl.col("dataset") == dataset)
        assert rows.height == expected["count"]
        assert rows["frequency_unit"].unique().to_list() == ["Hz"]
        assert rows["flux_unit"].unique().to_list() == ["erg s-1 cm-2 Hz-1"]
        assert rows["wavelength_unit"].unique().to_list() == ["nm"]

        raw_tars = sorted(expected["raw_dir"].glob("*flux*.tar"))
        assert raw_tars
        assert all(path.exists() for path in raw_tars)

        validation_path = processed_dir / "validation" / f"{expected['group']}.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        assert validation["dataset"] == dataset
        assert validation["source_count"] == expected["count"]
        assert validation["readback_ok"] is True
        assert validation["raw_deleted"] is False
        assert {
            str(Path(path).resolve()) for path in validation["source_tar_sha256"]
        } == {str(path.resolve()) for path in raw_tars}

        group = root[expected["group"]]
        subgroup_count = 0
        for subgroup in rows["zarr_subgroup"].unique().sort().to_list():
            grid_group = group[subgroup]
            frequency = np.asarray(grid_group["frequency_hz"][:])
            assert frequency.ndim == 1
            assert np.all(frequency > 0.0)
            flux = grid_group["flux_fnu"]
            subgroup_count += flux.shape[0]
            sample_rows = sorted({0, flux.shape[0] // 2, flux.shape[0] - 1})
            sample = np.asarray(flux[sample_rows, : min(256, flux.shape[1])])
            assert sample.shape[0] == len(sample_rows)
            assert np.isfinite(sample).all()
        assert subgroup_count == expected["count"]

        assert manifest["datasets"][dataset]["state"] == "processed"
        assert coverage[dataset].state == "processed"
        assert coverage[dataset].catalog_path == str(catalog_path)
        assert coverage[dataset].zarr_path == str(zarr_path)
