"""Build the deterministic adaptive-quadrature tolerance-sweep evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxstro.quad import (
    AdaptiveClenshawCurtis,
    AdaptiveTanhSinh,
    GaussKronrod,
    Interval,
    MaxNorm,
    Romberg,
    RombergTanhSinh,
    integrate,
)

METHODS = (
    ("gauss-kronrod-21", GaussKronrod(pair=21)),
    ("adaptive-clenshaw-curtis-17", AdaptiveClenshawCurtis(initial_order=17)),
    ("adaptive-tanh-sinh-3", AdaptiveTanhSinh(initial_level=3)),
    ("romberg-1", Romberg(initial_level=1)),
    ("romberg-tanh-sinh-1", RombergTanhSinh(initial_level=1)),
)
TOLERANCES = (1e-4, 1e-7, 1e-10)


def build_evidence() -> dict:
    """Return a stable analytic exponential sweep with complete result evidence."""
    jax.config.update("jax_enable_x64", True)
    expected = jnp.e - 1.0 / jnp.e
    records = []
    for method_name, method in METHODS:
        for requested in TOLERANCES:
            result = integrate(
                jnp.exp,
                Interval(-1.0, 1.0),
                method=method,
                epsabs=requested,
                epsrel=requested,
                max_evaluations=20_000,
                max_regions=256,
                error_norm=MaxNorm(),
            )
            records.append(
                {
                    "method": method_name,
                    "requested_tolerance": requested,
                    "reported_indicator_norm": float(result.error.norm),
                    "observed_absolute_error": float(jnp.abs(result.value - expected)),
                    "status": int(result.status),
                    "evaluations": int(result.work.evaluations),
                    "refinements": int(result.work.refinements),
                    "active_regions": int(result.work.active_regions),
                    "levels": int(result.work.levels),
                }
            )
    return {
        "schema_version": 1,
        "precision": "float64",
        "benchmark": "integral_exp_minus1_plus1",
        "analytic_value": float(expected),
        "claim_boundary": (
            "Analytic acceptance and work evidence only; no universal error bound "
            "or cross-library superiority claim."
        ),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_evidence(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
