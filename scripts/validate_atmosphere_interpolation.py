#!/usr/bin/env python3
"""Measure bounded leave-one-node-out atmosphere interpolation policies."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

import jax.numpy as jnp  # noqa: E402

from jaxstro.atmospheres import AtmosphereParams, AtmosphereQuery  # noqa: E402
from jaxstro.atmospheres.bosz import BoszBackend  # noqa: E402
from jaxstro.atmospheres.newera import NewEraBackend  # noqa: E402
from jaxstro.atmospheres.sonora import SonoraBackend  # noqa: E402
from jaxstro.atmospheres.tlusty import TlustyBackend  # noqa: E402
from jaxstro.spectra import (  # noqa: E402
    FluxInterpolation,
    PreparedRectilinearStencil,
    PreparedSimplexStencil,
    SpectralAxis,
    SpectralCoordinate,
    SpectralPlan,
)
from jaxstro.testing.spectral_validation import (  # noqa: E402
    longest_common_positive_slice,
    select_interpolation_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "validation" / "atmosphere-interpolation.json"
REPRESENTATIVE_PRODUCTS = (
    "newera-v3-lowres",
    "bosz-2025-recomputed:ap:r10000:resam",
    "sonora-diamondback-2024:f1:m-0.5:co1",
    "tlusty-ostar2002:z1",
    "tlusty-bstar2006:vturb2:z1",
    "tlusty-bstar2006:vturb10:z1:standard",
)


def _product_rows(adapter: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(adapter, NewEraBackend):
        return tuple(
            row
            for row in adapter.catalog_rows
            if float(row["m_h"]) == 0.0 and float(row["alpha_m"]) == 0.0
        )
    if isinstance(adapter, BoszBackend):
        return tuple(
            row
            for row in adapter.catalog_rows
            if str(row["atmosphere"]) == adapter.atmosphere
            and str(row["resolution"]) == adapter.resolution
            and str(row["product"]) == adapter.product
            and float(row["m_h"]) == 0.0
            and float(row["alpha_m"]) == 0.0
            and float(row["c_m"]) == 0.0
            and float(row["vturb_km_s"]) == 2.0
        )
    if isinstance(adapter, SonoraBackend):
        return tuple(
            row
            for row in adapter.catalog_rows
            if str(row["cloud_label"]) == adapter.cloud_label
            and float(row["m_h"]) == adapter.m_h
            and float(row["c_o"]) == adapter.c_o
        )
    if isinstance(adapter, TlustyBackend):
        spec = adapter.spec_for_product(adapter.product_id)
        return tuple(
            row
            for row in adapter.catalog_rows
            if row["dataset"] == spec.dataset
            and row["prefix"] == spec.prefix
            and bool(row["cn_altered"]) is spec.cn_altered
        )
    return ()


def _native_axis_nm(adapter: Any, row: dict[str, Any]) -> np.ndarray:
    if isinstance(adapter, NewEraBackend):
        return float(row["lambda_min"]) + np.arange(int(row["n_wave"])) * float(
            row["lambda_step"]
        )
    if isinstance(adapter, BoszBackend):
        return adapter._read_wavelength(row) / 10.0
    if isinstance(adapter, SonoraBackend):
        return np.asarray(adapter._store["wavelength"][:], dtype=float) * 1.0e3
    if isinstance(adapter, TlustyBackend):
        subgroup = adapter._store[str(row["zarr_group"])][str(row["zarr_subgroup"])]
        frequency = np.asarray(subgroup["frequency_hz"][:], dtype=float)
        from jaxstro.constants import C_CGS

        return C_CGS * 1.0e7 / frequency
    raise TypeError(type(adapter))


def _query(adapter: Any, row: dict[str, Any], axis_nm: np.ndarray) -> AtmosphereQuery:
    descriptor = adapter.describe_product()
    params = AtmosphereParams(
        teff=float(row["teff"]),
        logg=float(row["logg"]),
        m_h=float(row.get("m_h", 0.0)),
        alpha_m=float(row.get("alpha_m", 0.0)),
        c_m=float(row.get("c_m", 0.0)),
        vturb_km_s=float(row.get("vturb_km_s", 2.0)),
        c_o=float(row.get("c_o", 0.55)),
    )
    lo, hi = float(np.min(axis_nm)), float(np.max(axis_nm))
    target = jnp.linspace(lo, hi, min(64, axis_nm.size))
    return AtmosphereQuery(
        params=params,
        product_id=descriptor.product_id,
        family=descriptor.family,
        cloud_label=getattr(adapter, "cloud_label", None),
        spectral_plan=SpectralPlan(
            SpectralAxis.points(
                target,
                coordinate=SpectralCoordinate.WAVELENGTH,
                unit="nm",
            )
        ),
        requested_parameter_names=descriptor.parameter_names,
    )


def _with_policy(
    prepared: Any,
    policy: FluxInterpolation,
    support: slice,
) -> Any:
    stencil = prepared.stencil
    axis = replace(
        stencil.template.axis,
        values=stencil.template.axis.values[support],
    )
    template = replace(
        stencil.template,
        axis=axis,
        values=stencil.template.values[support],
    )
    if isinstance(stencil, PreparedRectilinearStencil):
        updated = PreparedRectilinearStencil(
            stencil.parameter_axes,
            stencil.vertex_values[..., support],
            template,
            policy,
        )
    elif isinstance(stencil, PreparedSimplexStencil):
        updated = PreparedSimplexStencil(
            stencil.vertices,
            stencil.vertex_values[..., support],
            template,
            policy,
        )
    else:
        raise TypeError(type(stencil))
    return replace(
        prepared,
        stencil=updated,
        spectral_plan=replace(prepared.spectral_plan, target_axis=axis),
    )


def _measure_adapter(adapter: Any) -> dict[str, Any] | None:
    rows = _product_rows(adapter)
    teff_values = sorted({float(row["teff"]) for row in rows})
    logg_values = sorted({float(row["logg"]) for row in rows})
    interior_teff = teff_values[1:-1]
    interior_logg = logg_values[1:-1]
    candidates = []
    for teff in interior_teff:
        at_teff = sorted(
            (
                row
                for row in rows
                if float(row["teff"]) == teff and float(row["logg"]) in interior_logg
            ),
            key=lambda row: (float(row["logg"]), str(row.get("filename", ""))),
        )
        if at_teff:
            candidates.append(at_teff[len(at_teff) // 2])
    for row in candidates[:4]:
        axis = _native_axis_nm(adapter, row)
        query = _query(adapter, row, axis)
        heldout_teff = float(row["teff"])
        heldout_ids = {
            id(candidate)
            for candidate in rows
            if float(candidate["teff"]) == heldout_teff
        }
        reduced = replace(
            adapter,
            catalog_rows=tuple(
                candidate
                for candidate in adapter.catalog_rows
                if id(candidate) not in heldout_ids
            ),
        )
        prediction_preparation = reduced.prepare(query)
        if prediction_preparation.prepared is None:
            print(
                f"  candidate teff={heldout_teff:g} logg={float(row['logg']):g}: "
                f"{prediction_preparation.status.name} "
                f"({prediction_preparation.message})",
                file=sys.stderr,
                flush=True,
            )
            continue
        truth_preparation = adapter.prepare(query)
        if truth_preparation.prepared is None:
            print(
                f"  truth teff={heldout_teff:g} logg={float(row['logg']):g}: "
                f"{truth_preparation.status.name} ({truth_preparation.message})",
                file=sys.stderr,
                flush=True,
            )
            continue
        truth = truth_preparation.prepared.evaluate(query.params).spectrum.values
        try:
            support = longest_common_positive_slice(
                prediction_preparation.prepared.stencil.vertex_values
            )
            linear_prepared = _with_policy(
                prediction_preparation.prepared,
                FluxInterpolation.LINEAR,
                support,
            )
            log_prepared = _with_policy(
                prediction_preparation.prepared,
                FluxInterpolation.POSITIVE_LOG,
                support,
            )
        except ValueError as exc:
            print(
                f"  positive-log infeasible at teff={heldout_teff:g} "
                f"logg={float(row['logg']):g}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        linear = linear_prepared.evaluate(query.params).spectrum.values
        positive_log = log_prepared.evaluate(query.params).spectrum.values
        wavelength = query.spectral_plan.target_axis.values[support]
        selection = select_interpolation_policy(
            wavelength,
            truth[support],
            {"linear": linear, "positive_log": positive_log},
        )
        return {
            "product_id": adapter.describe_product().product_id,
            "status": selection.status,
            "accepted_policy": selection.accepted_policy,
            "holdout_method": "leave-one-teff-slice-out",
            "holdout": {"teff": heldout_teff, "logg": float(row["logg"])},
            "positive_support": {
                "bins": int(wavelength.size),
                "wavelength_min_nm": float(wavelength[0]),
                "wavelength_max_nm": float(wavelength[-1]),
            },
            "metrics": {
                name: metrics.to_dict() for name, metrics in selection.metrics.items()
            },
        }
    return None


def build_manifest() -> dict[str, Any]:
    data = REPO_ROOT / "data" / "atmospheres"
    adapters = (
        NewEraBackend.open(data / "newera" / "processed"),
        BoszBackend.open(
            data / "bosz" / "2025-recomputed" / "processed",
            atmosphere="ap",
            resolution="r10000",
            product="resam",
        ),
        SonoraBackend.open(
            data / "sonora" / "2024" / "processed",
            cloud_label="f1",
            m_h=-0.5,
            c_o=1.0,
        ),
        TlustyBackend.open(
            data / "tlusty" / "processed", product_id="tlusty-ostar2002:z1"
        ),
        TlustyBackend.open(
            data / "tlusty" / "processed",
            product_id="tlusty-bstar2006:vturb2:z1",
        ),
        TlustyBackend.open(
            data / "tlusty" / "processed",
            product_id="tlusty-bstar2006:vturb10:z1:standard",
        ),
    )
    records = {}
    for adapter in adapters:
        product_id = adapter.describe_product().product_id
        print(f"measuring {product_id}", file=sys.stderr, flush=True)
        measured = _measure_adapter(adapter)
        if measured is not None:
            records[product_id] = measured
            print(
                f"measured {product_id}: {measured['status']}",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"excluded {product_id}: no viable bounded holdout",
                file=sys.stderr,
                flush=True,
            )
    return {
        "schema_version": 1,
        "primary_metric": "p95_relative_error",
        "selection_rule": "primary win plus no secondary regression",
        "representative_products": REPRESENTATIVE_PRODUCTS,
        "products": dict(sorted(records.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    measured = build_manifest()
    rendered = json.dumps(measured, indent=2, sort_keys=True) + "\n"
    if args.emit:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(rendered, encoding="utf-8")
        print(
            f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(measured['products'])} products)"
        )
        return 0
    if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
        print("atmosphere interpolation manifest is stale")
        return 1
    print(
        f"atmosphere interpolation manifest fresh ({len(measured['products'])} products)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
