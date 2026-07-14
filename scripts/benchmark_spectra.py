#!/usr/bin/env python3
"""Run one bounded real-artifact prepared-spectrum benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import tracemalloc
from pathlib import Path
from typing import Any

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jaxstro import __version__  # noqa: E402
from jaxstro.atmospheres import (  # noqa: E402
    AtmosphereLibrary,
    AtmosphereParams,
    AtmosphereQuery,
)
from jaxstro.evidence import (  # noqa: E402
    EnvironmentRecord,
    EvidenceArtifact,
    EvidenceStatus,
    MetricRecord,
    artifact_from_dict,
    check_artifact,
    emit_artifact,
)
from jaxstro.spectra import (  # noqa: E402
    SpectralAxis,
    SpectralCoordinate,
    SpectralPlan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "validation" / "spectra-performance.json"
REPORT = REPO_ROOT / "docs" / "60-validation" / "data" / "spectra-performance.md"
PRODUCT_ID = "newera-v3-lowres"
SPECTRAL_BINS = 64
BATCH_SIZE = 8
CACHED_REPEATS = 7


def _seconds(callable_: Any) -> tuple[float, Any]:
    start = time.perf_counter()
    value = callable_()
    jax.block_until_ready(value)
    return time.perf_counter() - start, value


def run_benchmark() -> dict[str, Any]:
    """Measure preparation separately from fixed-shape JAX evaluation."""
    library = AtmosphereLibrary.from_local(REPO_ROOT / "data")
    params = AtmosphereParams(teff=2425.0, logg=3.0)
    query = AtmosphereQuery(
        params=params,
        product_id=PRODUCT_ID,
        family="newera",
        spectral_plan=SpectralPlan(
            SpectralAxis.points(
                jnp.linspace(500.0, 2500.0, SPECTRAL_BINS),
                coordinate=SpectralCoordinate.WAVELENGTH,
                unit="nm",
            )
        ),
        requested_parameter_names=("teff", "logg"),
    )

    tracemalloc.start()
    preparation_start = time.perf_counter()
    preparation = library.prepare(query)
    preparation_seconds = time.perf_counter() - preparation_start
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if preparation.prepared is None:
        raise RuntimeError(
            f"benchmark preparation failed: {preparation.status.name} "
            f"{preparation.message}"
        )
    prepared = preparation.prepared

    def evaluate(point):
        return prepared.stencil.evaluate(point).spectrum.values

    compiled = jax.jit(evaluate)
    point = jnp.array([2425.0, 3.0])
    first_jit_seconds, output = _seconds(lambda: compiled(point))
    cached = [_seconds(lambda: compiled(point))[0] for _ in range(CACHED_REPEATS)]

    batch = jnp.column_stack(
        (
            jnp.linspace(2410.0, 2490.0, BATCH_SIZE),
            jnp.full((BATCH_SIZE,), 3.0),
        )
    )
    batched = jax.jit(jax.vmap(evaluate))
    batched_seconds, batched_output = _seconds(lambda: batched(batch))

    return {
        "schema_version": 1,
        "case": {
            "product_id": PRODUCT_ID,
            "spectral_bins": SPECTRAL_BINS,
            "batch_size": BATCH_SIZE,
            "cached_repeats": CACHED_REPEATS,
        },
        "timings_seconds": {
            "host_preparation": preparation_seconds,
            "first_jit_evaluation": first_jit_seconds,
            "cached_evaluation_median": statistics.median(cached),
            "batched_evaluation": batched_seconds,
        },
        "memory_scope": "Python host allocations measured by tracemalloc",
        "host_peak_memory_bytes": peak_memory,
        "output_shape": list(output.shape),
        "batched_output_shape": list(batched_output.shape),
        "dtype": str(output.dtype),
        "jax_backend": jax.default_backend(),
    }


def _validate(payload: dict[str, Any]) -> None:
    timings = payload.get("timings_seconds", {})
    required = {
        "host_preparation",
        "first_jit_evaluation",
        "cached_evaluation_median",
        "batched_evaluation",
    }
    if set(timings) != required or not all(
        isinstance(value, (int, float)) and value >= 0.0 for value in timings.values()
    ):
        raise ValueError("spectra benchmark timings are incomplete or invalid")
    if list(payload.get("output_shape", ())) != [SPECTRAL_BINS]:
        raise ValueError("spectra benchmark output shape is invalid")
    if list(payload.get("batched_output_shape", ())) != [BATCH_SIZE, SPECTRAL_BINS]:
        raise ValueError("spectra benchmark batched output shape is invalid")
    if int(payload.get("host_peak_memory_bytes", 0)) <= 0:
        raise ValueError("spectra benchmark host peak memory is invalid")


def build_artifact(payload: dict[str, Any]) -> EvidenceArtifact:
    """Wrap bounded performance observations without inventing thresholds."""
    symbols = {
        "host_preparation": "t_host,prepare",
        "first_jit_evaluation": "t_wall,first-jit",
        "cached_evaluation_median": "t_wall,warm",
        "batched_evaluation": "t_wall,batch",
    }
    metrics = tuple(
        MetricRecord(
            f"spectra.{name}",
            symbols[name],
            value,
            "s",
            EvidenceStatus.INFO,
        )
        for name, value in sorted(payload["timings_seconds"].items())
    ) + (
        MetricRecord(
            "spectra.host_peak_memory",
            "M_host,peak",
            payload["host_peak_memory_bytes"],
            "bytes",
            EvidenceStatus.INFO,
        ),
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    return EvidenceArtifact(
        schema_version="1",
        artifact_id="spectra.performance",
        artifact_version="1",
        package_version=__version__,
        source_revision=revision,
        generation_command="uv run --no-sync python scripts/benchmark_spectra.py --emit",
        precision=payload["dtype"],
        deterministic_config=tuple(sorted(payload["case"].items())),
        environment=EnvironmentRecord(
            "Backend, dtype, and host-memory scope recorded; performance metrics are informational.",
            (
                ("jax_backend", payload["jax_backend"]),
                ("memory_scope", payload["memory_scope"]),
            ),
        ),
        metrics=metrics,
        limitations=(
            "Wall times depend on hardware and system load.",
            "Host peak memory covers Python allocations measured by tracemalloc.",
        ),
        method_payload=payload,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    measured = run_benchmark()
    _validate(measured)
    artifact = build_artifact(measured)
    if args.emit:
        emit_artifact(OUTPUT, artifact)
        emit_artifact(REPORT, artifact)
        print(
            f"wrote {OUTPUT.relative_to(REPO_ROOT)}: "
            f"prepare={measured['timings_seconds']['host_preparation']:.3f}s, "
            f"first_jit={measured['timings_seconds']['first_jit_evaluation']:.3f}s, "
            f"cached={measured['timings_seconds']['cached_evaluation_median']:.6f}s, "
            f"batch={measured['timings_seconds']['batched_evaluation']:.3f}s"
        )
        return 0

    if not OUTPUT.exists():
        print("spectra performance manifest is missing")
        return 1
    stored_artifact = artifact_from_dict(json.loads(OUTPUT.read_text(encoding="utf-8")))
    _validate(stored_artifact.method_payload)
    check_artifact(REPORT, stored_artifact)
    print(
        "spectra benchmark healthy: "
        f"current prepare={measured['timings_seconds']['host_preparation']:.3f}s, "
        f"first_jit={measured['timings_seconds']['first_jit_evaluation']:.3f}s, "
        f"cached={measured['timings_seconds']['cached_evaluation_median']:.6f}s, "
        f"batch={measured['timings_seconds']['batched_evaluation']:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
