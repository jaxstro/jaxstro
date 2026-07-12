"""Host-side TLUSTY raw flux metadata helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from jaxstro.constants import C_CGS
from jaxstro.spectra import (
    FluxInterpolation,
    PreparedRectilinearStencil,
    PreparedSimplexStencil,
    SpectralAxis,
    SpectralCoordinate,
    SpectralSemantic,
    Spectrum,
    SpectrumProvenance,
    SpectrumStatusCode,
    resample_spectrum,
)

from .adapters import PreparationResult, PreparedAtmosphere
from .params import AtmosphereQuery
from .products import ArtifactReport, ProductDescriptor
from .topology import GridTopology, TopologyKind, select_topology

DEFAULT_TLUSTY_ZARR = "tlusty_flux.zarr"
DEFAULT_TLUSTY_CATALOG = "catalog.parquet"
_C_NM_S = C_CGS * 1.0e7
_TLUSTY_FLUX_PATTERN = re.compile(
    r"^(?P<prefix>[A-Z]+)"
    r"(?P<teff>\d+)"
    r"g(?P<logg_code>\d+)"
    r"v(?P<vturb_km_s>\d+)"
    r"(?P<cn>CN)?"
    r"\.flux\.gz$"
)
_BARE_EXPONENT_PATTERN = re.compile(
    r"^(?P<mantissa>[+-]?(?:\d+(?:\.\d*)?|\.\d+))(?P<exponent>[+-]\d+)$"
)


@dataclass(frozen=True)
class TlustyFluxMetadata:
    """Metadata encoded in a TLUSTY flux filename."""

    filename: str
    prefix: str
    teff: float
    logg: float
    vturb_km_s: float
    cn_altered: bool


@dataclass(frozen=True)
class TlustyProductSpec:
    """Identity of one composition-scoped TLUSTY spectral product."""

    product_id: str
    dataset: str
    prefix: str
    cn_altered: bool
    z_over_z_sun: float
    vturb_km_s: float


_OSTAR_COMPOSITIONS = (
    ("z2", "C", 2.0),
    ("z1", "G", 1.0),
    ("z0p5", "L", 0.5),
    ("z0p2", "S", 0.2),
    ("z0p1", "T", 0.1),
    ("z1over30", "V", 1.0 / 30.0),
    ("z0p02", "W", 0.02),
    ("z0p01", "X", 0.01),
    ("z0p001", "Y", 0.001),
    ("z0", "Z", 0.0),
)
_BSTAR_COMPOSITIONS = (
    ("z2", "BC", 2.0),
    ("z1", "BG", 1.0),
    ("z0p5", "BL", 0.5),
    ("z0p2", "BS", 0.2),
    ("z0p1", "BT", 0.1),
    ("z0", "BZ", 0.0),
)
_TLUSTY_PRODUCT_SPECS = (
    *(
        TlustyProductSpec(
            product_id=f"tlusty-ostar2002:{token}",
            dataset="tlusty_ostar_2002",
            prefix=prefix,
            cn_altered=False,
            z_over_z_sun=metallicity,
            vturb_km_s=10.0,
        )
        for token, prefix, metallicity in _OSTAR_COMPOSITIONS
    ),
    *(
        TlustyProductSpec(
            product_id=f"tlusty-bstar2006:vturb2:{token}",
            dataset="tlusty_bstar_2007_vturb_2",
            prefix=prefix,
            cn_altered=False,
            z_over_z_sun=metallicity,
            vturb_km_s=2.0,
        )
        for token, prefix, metallicity in _BSTAR_COMPOSITIONS
    ),
    *(
        TlustyProductSpec(
            product_id=f"tlusty-bstar2006:vturb10:{token}:{variant}",
            dataset="tlusty_bstar_2007_vturb_10_cn",
            prefix=prefix,
            cn_altered=variant == "cn",
            z_over_z_sun=metallicity,
            vturb_km_s=10.0,
        )
        for token, prefix, metallicity in _BSTAR_COMPOSITIONS
        for variant in (("standard", "cn") if prefix != "BZ" else ("standard",))
    ),
)
_TLUSTY_PRODUCT_BY_ID = {spec.product_id: spec for spec in _TLUSTY_PRODUCT_SPECS}


@dataclass(frozen=True)
class TlustyBackend:
    """Exact-product adapter for processed TLUSTY H_nu spectra."""

    processed_dir: Path
    catalog_rows: tuple[dict[str, Any], ...]
    zarr_path: Path
    _store: Any = field(repr=False, compare=False)
    product_id: str = "tlusty-ostar2002:z1"
    approved_simplices: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def open(
        cls,
        processed_dir: str | os.PathLike[str] | None = None,
        *,
        product_id: str,
        catalog_name: str = DEFAULT_TLUSTY_CATALOG,
        zarr_name: str = DEFAULT_TLUSTY_ZARR,
        approved_simplices: tuple[tuple[int, ...], ...] = (),
    ) -> TlustyBackend:
        cls.dataset_for_product(product_id)
        pl, zarr = _load_optional_backend_deps()
        if processed_dir is None:
            from . import resolve_data_dir

            root = resolve_data_dir() / "atmospheres" / "tlusty" / "processed"
        else:
            root = Path(processed_dir).expanduser()
        catalog_path = root / catalog_name
        zarr_path = root / zarr_name
        if not catalog_path.exists():
            raise FileNotFoundError(f"TLUSTY catalog not found: {catalog_path}")
        if not zarr_path.exists():
            raise FileNotFoundError(f"TLUSTY Zarr store not found: {zarr_path}")
        return cls(
            processed_dir=root,
            catalog_rows=tuple(pl.read_parquet(catalog_path).to_dicts()),
            zarr_path=zarr_path,
            _store=zarr.open_group(zarr_path, mode="r"),
            product_id=product_id,
            approved_simplices=approved_simplices,
        )

    @staticmethod
    def product_specs() -> tuple[TlustyProductSpec, ...]:
        """Return all exact TLUSTY products in deterministic order."""
        return _TLUSTY_PRODUCT_SPECS

    @staticmethod
    def spec_for_product(product_id: str) -> TlustyProductSpec:
        try:
            return _TLUSTY_PRODUCT_BY_ID[product_id]
        except KeyError as exc:
            raise ValueError(f"unknown TLUSTY product: {product_id}") from exc

    @staticmethod
    def dataset_for_product(product_id: str) -> str:
        return TlustyBackend.spec_for_product(product_id).dataset

    @staticmethod
    def product_descriptor(product_id: str) -> ProductDescriptor:
        TlustyBackend.dataset_for_product(product_id)
        spec = TlustyBackend.spec_for_product(product_id)
        provenance_id = (
            "tlusty-ostar2002"
            if spec.dataset == "tlusty_ostar_2002"
            else "tlusty-bstar2006"
        )
        return ProductDescriptor(
            product_id=product_id,
            family="tlusty",
            parameter_names=("teff", "logg"),
            topology_policy="complete-cell-or-approved-simplex",
            flux_interpolation_policy="linear",
            provenance_id=provenance_id,
        )

    def describe_product(self) -> ProductDescriptor:
        return self.product_descriptor(self.product_id)

    def validate_artifact(self) -> ArtifactReport:
        valid = self.zarr_path.exists() and bool(self.catalog_rows)
        payload = json.dumps(self.catalog_rows, sort_keys=True, default=str).encode()
        digest = f"sha256:{hashlib.sha256(payload).hexdigest()}" if valid else None
        return ArtifactReport(
            valid=valid,
            digest=digest,
            schema="tlusty-ragged-frequency-v1" if valid else None,
            message="" if valid else "TLUSTY artifact is unavailable or empty",
        )

    def prepare(self, query: AtmosphereQuery) -> PreparationResult:
        descriptor = self.describe_product()
        if query.product_id != descriptor.product_id:
            return PreparationResult.failure(
                SpectrumStatusCode.NO_DATASET,
                "TLUSTY query product does not match the opened dataset",
            )
        spec = self.spec_for_product(self.product_id)
        rows = [
            row
            for row in self.catalog_rows
            if str(row["dataset"]) == spec.dataset
            and str(row["prefix"]) == spec.prefix
            and bool(row["cn_altered"]) is spec.cn_altered
        ]
        if not rows:
            return PreparationResult.failure(
                SpectrumStatusCode.NO_DATASET,
                "No rows for the exact TLUSTY product",
            )
        topology = GridTopology(
            parameter_names=descriptor.parameter_names,
            points=tuple((float(row["teff"]), float(row["logg"])) for row in rows),
            approved_simplices=self.approved_simplices,
        )
        params = query.params
        point = (float(np.asarray(params.teff)), float(np.asarray(params.logg)))
        selection = select_topology(topology, point)
        if selection.status is not SpectrumStatusCode.OK:
            return PreparationResult.failure(
                selection.status, "No valid TLUSTY topology"
            )
        artifact = self.validate_artifact()
        vertex_values = []
        template = None
        for index in selection.vertex_indices:
            record = rows[index]
            subgroup = self._store[str(record["zarr_group"])][
                str(record["zarr_subgroup"])
            ]
            frequency = np.asarray(subgroup["frequency_hz"][:], dtype=float)
            h_nu = np.asarray(
                subgroup["flux_fnu"][int(record["zarr_row"]), :], dtype=float
            )
            wavelength_nm = _C_NM_S / frequency
            order = np.argsort(wavelength_nm)
            wavelength_nm = wavelength_nm[order]
            f_nu = 4.0 * np.pi * h_nu[order]
            f_lambda = f_nu * _C_NM_S / wavelength_nm**2
            provenance = SpectrumProvenance(
                source_id=descriptor.provenance_id,
                product_id=descriptor.product_id,
                native_coordinate="frequency_hz",
                native_density="H_nu",
                native_unit="erg s^-1 cm^-2 Hz^-1",
                canonical_conversion=(
                    "F_nu=4*pi*H_nu; F_lambda=F_nu*c_nm_s/lambda_nm^2"
                ),
                citations=("https://tlusty.oca.eu/tlusty/Tlusty2002/",),
                artifact_digest=artifact.digest,
            )
            native = Spectrum(
                axis=SpectralAxis.points(
                    jnp.asarray(wavelength_nm),
                    coordinate=SpectralCoordinate.WAVELENGTH,
                    unit="nm",
                ),
                values=jnp.asarray(f_lambda),
                semantic=SpectralSemantic.SURFACE_FLUX_LAMBDA,
                provenance=provenance,
            )
            remapped = resample_spectrum(native, query.spectral_plan)
            if int(remapped.status.code) != SpectrumStatusCode.OK:
                return PreparationResult.failure(
                    SpectrumStatusCode.UNSUPPORTED_SPECTRAL_WINDOW,
                    "TLUSTY vertices lack common requested spectral coverage",
                )
            template = remapped.spectrum
            vertex_values.append(template.values)
        if template is None:
            raise ValueError("TLUSTY topology selected no vertices")
        stacked = jnp.stack(vertex_values)
        stencil: PreparedRectilinearStencil | PreparedSimplexStencil
        if selection.kind is TopologyKind.RECTILINEAR:
            selected = [topology.points[i] for i in selection.vertex_indices]
            teff_axis = jnp.asarray(sorted({value[0] for value in selected}))
            logg_axis = jnp.asarray(sorted({value[1] for value in selected}))
            stencil = PreparedRectilinearStencil(
                parameter_axes=(teff_axis, logg_axis),
                vertex_values=stacked.reshape(
                    (teff_axis.shape[0], logg_axis.shape[0], stacked.shape[-1])
                ),
                template=template,
                interpolation=FluxInterpolation.LINEAR,
            )
        else:
            stencil = PreparedSimplexStencil(
                vertices=jnp.asarray(
                    [topology.points[index] for index in selection.vertex_indices]
                ),
                vertex_values=stacked,
                template=template,
            )
        return PreparationResult.success(
            PreparedAtmosphere(
                stencil=stencil,
                parameter_names=descriptor.parameter_names,
                spectral_plan=query.spectral_plan,
                provenance=template.provenance,
            )
        )


def _load_optional_backend_deps():
    try:
        pl = import_module("polars")
        zarr = import_module("zarr")
    except ImportError as exc:  # pragma: no cover
        raise ImportError("TlustyBackend requires optional data dependencies") from exc
    return pl, zarr


def parse_tlusty_flux_filename(path: str | Path) -> TlustyFluxMetadata:
    """Parse TLUSTY ``*.flux.gz`` filename coordinates."""
    filename = Path(path).name
    match = _TLUSTY_FLUX_PATTERN.match(filename)
    if match is None:
        raise ValueError(f"Not a TLUSTY flux filename: {filename}")

    return TlustyFluxMetadata(
        filename=filename,
        prefix=match.group("prefix"),
        teff=float(match.group("teff")),
        logg=float(match.group("logg_code")) / 100.0,
        vturb_km_s=float(match.group("vturb_km_s")),
        cn_altered=match.group("cn") is not None,
    )


def parse_tlusty_float(token: str) -> float:
    """Parse TLUSTY Fortran-style floats, including bare signed exponents."""
    normalized = token.replace("D", "E").replace("d", "E")
    match = _BARE_EXPONENT_PATTERN.match(normalized)
    if match is not None and "E" not in normalized and "e" not in normalized:
        normalized = f"{match.group('mantissa')}E{match.group('exponent')}"
    return float(normalized)


__all__ = [
    "DEFAULT_TLUSTY_CATALOG",
    "DEFAULT_TLUSTY_ZARR",
    "TlustyBackend",
    "TlustyFluxMetadata",
    "TlustyProductSpec",
    "parse_tlusty_float",
    "parse_tlusty_flux_filename",
]
