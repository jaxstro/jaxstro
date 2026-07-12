"""Tests for TLUSTY raw metadata and numeric parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jaxstro.atmospheres import AtmosphereParams, AtmosphereQuery
from jaxstro.atmospheres.tlusty import (
    TlustyBackend,
    parse_tlusty_float,
    parse_tlusty_flux_filename,
)
from jaxstro.constants import C_CGS
from jaxstro.spectra import (
    SpectralAxis,
    SpectralCoordinate,
    SpectralPlan,
    SpectrumStatusCode,
)

C_NM_S = C_CGS * 1.0e7


def _write_processed_artifact(
    processed_dir: Path,
    *,
    dataset: str = "tlusty_ostar_2002",
    prefix: str = "G",
    cn_altered: bool = False,
) -> None:
    pl = pytest.importorskip("polars")
    zarr = pytest.importorskip("zarr")
    root = zarr.open_group(processed_dir / "tlusty_flux.zarr", mode="w", zarr_format=2)
    dataset_group = root.create_group(dataset.removeprefix("tlusty_"))
    wavelength_nm = np.array([100.0, 200.0, 400.0])
    frequency_hz = C_NM_S / wavelength_nm
    rows = []
    for row, (teff, logg) in enumerate(
        ((30000.0, 3.0), (30000.0, 4.0), (40000.0, 3.0), (40000.0, 4.0))
    ):
        subgroup_name = f"grid{row:03d}"
        subgroup = dataset_group.create_group(subgroup_name)
        subgroup.create_array("frequency_hz", data=frequency_hz, chunks=(3,))
        desired_flambda = np.array([1.0, 2.0, 3.0]) + row
        h_nu = desired_flambda * wavelength_nm**2 / (4.0 * np.pi * C_NM_S)
        subgroup.create_array(
            "flux_fnu",
            data=h_nu[None, :].astype(np.float32),
            chunks=(1, 3),
        )
        rows.append(
            {
                "filename": f"synthetic-{row}",
                "prefix": prefix,
                "teff": teff,
                "logg": logg,
                "vturb_km_s": 10.0 if "10" in dataset else 2.0,
                "cn_altered": cn_altered,
                "dataset": dataset,
                "frequency_unit": "Hz",
                "flux_unit": "erg s-1 cm-2 Hz-1",
                "zarr_group": dataset.removeprefix("tlusty_"),
                "zarr_subgroup": subgroup_name,
                "zarr_row": 0,
            }
        )
    pl.DataFrame(rows).write_parquet(processed_dir / "catalog.parquet")


def _query(product_id: str = "tlusty-ostar2002:z1") -> AtmosphereQuery:
    return AtmosphereQuery(
        params=AtmosphereParams(teff=35000.0, logg=3.5),
        product_id=product_id,
        family="tlusty",
        spectral_plan=SpectralPlan(
            SpectralAxis.points(
                np.array([100.0, 200.0, 400.0]),
                coordinate=SpectralCoordinate.WAVELENGTH,
                unit="nm",
            )
        ),
        requested_parameter_names=("teff", "logg"),
    )


def test_parse_tlusty_filename_extracts_axes_and_cn_flag():
    metadata = parse_tlusty_flux_filename("BC15000g175v10CN.flux.gz")

    assert metadata.prefix == "BC"
    assert metadata.teff == 15000.0
    assert metadata.logg == 1.75
    assert metadata.vturb_km_s == 10.0
    assert metadata.cn_altered is True


def test_parse_tlusty_filename_rejects_non_flux_name():
    with pytest.raises(ValueError, match="TLUSTY flux"):
        parse_tlusty_flux_filename("BC15000g175v10.11.gz")


def test_parse_tlusty_float_accepts_fortran_and_bare_exponents():
    assert parse_tlusty_float("6.67943694D+16") == 6.67943694e16
    assert parse_tlusty_float("1.4363-100") == 1.4363e-100


def test_tlusty_backend_reads_subgroup_grids_and_converts_h_nu(tmp_path):
    _write_processed_artifact(tmp_path)
    backend = TlustyBackend.open(tmp_path, product_id="tlusty-ostar2002:z1")

    prepared = backend.prepare(_query())
    assert prepared.prepared is not None
    result = prepared.prepared.evaluate(_query().params)

    np.testing.assert_allclose(result.spectrum.axis.values, [100.0, 200.0, 400.0])
    np.testing.assert_allclose(result.spectrum.values, [2.5, 3.5, 4.5], rtol=2e-7)
    assert result.spectrum.provenance.canonical_conversion == (
        "F_nu=4*pi*H_nu; F_lambda=F_nu*c_nm_s/lambda_nm^2"
    )
    assert int(result.status.code) == SpectrumStatusCode.OK


def test_tlusty_exposes_all_27_exact_composition_products() -> None:
    specs = TlustyBackend.product_specs()

    assert len(specs) == 27
    assert len({spec.product_id for spec in specs}) == 27
    assert {spec.prefix for spec in specs if spec.dataset == "tlusty_ostar_2002"} == {
        "C",
        "G",
        "L",
        "S",
        "T",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    }
    assert sum(spec.cn_altered for spec in specs) == 5


def test_tlusty_product_specs_preserve_composition_identity() -> None:
    solar = TlustyBackend.spec_for_product("tlusty-ostar2002:z1")
    metal_free = TlustyBackend.spec_for_product("tlusty-ostar2002:z0")

    assert solar.dataset == metal_free.dataset == "tlusty_ostar_2002"
    assert solar.prefix == "G"
    assert metal_free.prefix == "Z"
    assert solar.z_over_z_sun == 1.0
    assert metal_free.z_over_z_sun == 0.0


def test_tlusty_backend_filters_shared_dataset_by_exact_composition(tmp_path) -> None:
    _write_processed_artifact(tmp_path, prefix="G")
    backend = TlustyBackend.open(tmp_path, product_id="tlusty-ostar2002:z0")

    prepared = backend.prepare(_query("tlusty-ostar2002:z0"))

    assert prepared.status is SpectrumStatusCode.NO_DATASET


def test_tlusty_rejects_product_mismatch_and_noncommon_coverage(tmp_path):
    _write_processed_artifact(tmp_path)
    backend = TlustyBackend.open(tmp_path, product_id="tlusty-ostar2002:z1")

    mismatch = backend.prepare(_query("tlusty-bstar2006:vturb2:z1"))
    outside_query = _query()
    outside_query = AtmosphereQuery(
        params=outside_query.params,
        product_id=outside_query.product_id,
        family=outside_query.family,
        spectral_plan=SpectralPlan(
            SpectralAxis.points(
                np.array([90.0, 100.0]),
                coordinate=SpectralCoordinate.WAVELENGTH,
                unit="nm",
            )
        ),
        requested_parameter_names=outside_query.requested_parameter_names,
    )
    outside = backend.prepare(outside_query)

    assert mismatch.status is SpectrumStatusCode.NO_DATASET
    assert outside.status is SpectrumStatusCode.UNSUPPORTED_SPECTRAL_WINDOW
