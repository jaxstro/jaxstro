#!/usr/bin/env python3
"""Emit and validate the immutable Phase B4 multidimensional benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scripts.quad_multidim_benchmark_adapters import (  # noqa: E402
    run_comparators,
    validate_comparison_record,
)

from jaxstro import quad  # noqa: E402

COMPARISON_OUTPUT = ROOT / "docs/validation/quad-multidim-comparisons.json"
BASELINE_OUTPUT = ROOT / "docs/validation/quad-multidim-performance-baseline.json"
DIMENSIONS = (2, 4, 8, 16)
LEVEL = 8
WARM_REPEATS = 5
VMAP_BATCHES = (16, 128)
TRIGGERS = {
    "warm_runtime_ratio": 1.50,
    "compiler_cost_ratio": 2.00,
    "memory_family_ratio": 2.00,
    "repeated_scaling_excess": 0.25,
}


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tree_is_clean() -> bool:
    return not _git("status", "--porcelain")


def _environment() -> dict[str, object]:
    device = jax.devices()[0]
    return {
        "backend": jax.default_backend(),
        "device": str(device),
        "device_kind": getattr(device, "device_kind", "unknown"),
        "jax": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jaxlib": jax.lib.__version__,
        "machine": platform.machine(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _ready(value):
    return jax.tree.map(
        lambda leaf: (
            jax.block_until_ready(leaf) if hasattr(leaf, "block_until_ready") else leaf
        ),
        value,
    )


def _measure(call: Callable[[], Any], repeats: int = WARM_REPEATS):
    samples = []
    output = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = _ready(call())
        samples.append(time.perf_counter() - started)
    return output, samples


def _measure_parameter_batch(kernel, parameters, *, repeats: int = WARM_REPEATS):
    per_call_samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for parameter in parameters:
            _ready(kernel(parameter))
        per_call_samples.append((time.perf_counter() - started) / len(parameters))
    return per_call_samples


def _compile_and_warm(call: Callable[[], Any]):
    started = time.perf_counter()
    output = _ready(call())
    compile_seconds = time.perf_counter() - started
    output, samples = _measure(call)
    return output, {
        "compile_seconds": compile_seconds,
        "warm_samples_seconds": samples,
        "warm_median_seconds": statistics.median(samples),
        "warm_mad_seconds": statistics.median(
            abs(value - statistics.median(samples)) for value in samples
        ),
    }


def _solve_factory(dimension: int):
    domain = quad.Hyperrectangle(
        jnp.zeros(dimension, dtype=jnp.float64),
        jnp.ones(dimension, dtype=jnp.float64),
    )
    method = quad.Sobol(level=LEVEL, bits=53)

    def solve(amplitude):
        return quad.integrate(
            lambda x, scale: scale * jnp.exp(-jnp.sum(x, axis=-1)),
            domain,
            args=amplitude,
            method=method,
            epsabs=0.0,
            epsrel=0.0,
            max_evaluations=1 << LEVEL,
            gradient="replay",
        ).value

    return solve


def _benchmark_record(dimension: int) -> dict[str, Any]:
    solve = _solve_factory(dimension)
    truth = math.expm1(-1.0) ** dimension
    amplitude = jnp.asarray(1.0, dtype=jnp.float64)

    scalar = jax.jit(solve)
    scalar_value, scalar_timing = _compile_and_warm(lambda: scalar(amplitude))

    grad_kernel = jax.jit(jax.grad(solve))
    gradient, grad_timing = _compile_and_warm(lambda: grad_kernel(amplitude))

    def jvp():
        return jax.jvp(
            solve,
            (amplitude,),
            (jnp.asarray(1.0, dtype=amplitude.dtype),),
        )[1]

    jvp_kernel = jax.jit(jvp)
    jvp_value, jvp_timing = _compile_and_warm(jvp_kernel)

    vmap_timings = {}
    for batch in VMAP_BATCHES:
        parameters = jnp.linspace(0.5, 1.5, batch, dtype=jnp.float64)
        kernel = jax.jit(jax.vmap(solve))
        _values, timing = _compile_and_warm(lambda: kernel(parameters))
        vmap_timings[str(batch)] = timing

    same_parameters = jnp.ones(64, dtype=jnp.float64)
    changing = jnp.linspace(0.8, 1.2, 64, dtype=jnp.float64)
    same_samples = _measure_parameter_batch(scalar, same_parameters)
    changing_samples = _measure_parameter_batch(scalar, changing)
    same_median = statistics.median(same_samples)
    changing_median = statistics.median(changing_samples)
    repeated_scaling_excess = max(0.0, changing_median / same_median - 1.0)

    value = float(np.asarray(scalar_value))
    gradient_value = float(np.asarray(gradient))
    jvp_float = float(np.asarray(jvp_value))
    point_count = 1 << LEVEL
    memory_proxy_bytes = point_count * dimension * 8 + point_count * 16
    expected_memory_bytes = memory_proxy_bytes
    return {
        "case_id": f"sobol_exponential_d{dimension}",
        "dimension": dimension,
        "dtype": "float64",
        "method": "Sobol",
        "controls": {
            "level": LEVEL,
            "bits": 53,
            "scramble": False,
            "epsabs": 0.0,
            "epsrel": 0.0,
            "gradient": "replay",
        },
        "value": value,
        "truth": truth,
        "truth_error": abs(value - truth),
        "logical_evaluations": point_count,
        "unique_nodes": point_count,
        "regions": 0,
        "indices": 0,
        "replicates": 0,
        "sobol_level": LEVEL,
        "coverage": None,
        "coverage_reason": "deterministic_unscrambled_formula",
        "gradient": gradient_value,
        "jvp": jvp_float,
        "gradient_truth": truth,
        "gradient_error": abs(gradient_value - truth),
        "jvp_error": abs(jvp_float - truth),
        "timings": {
            "scalar": scalar_timing,
            "gradient": grad_timing,
            "jvp": jvp_timing,
            "vmap": vmap_timings,
            "same_domain_repeat_samples_seconds": same_samples,
            "changing_parameter_repeat_samples_seconds": changing_samples,
        },
        "compiler_cost_proxy": (
            scalar_timing["compile_seconds"] / scalar_timing["warm_median_seconds"]
        ),
        "memory_proxy_bytes": memory_proxy_bytes,
        "memory_expected_bytes": expected_memory_bytes,
        "memory_family_ratio": memory_proxy_bytes / expected_memory_bytes,
        "repeated_same_median_seconds": same_median,
        "repeated_changing_median_seconds": changing_median,
        "repeated_scaling_excess": repeated_scaling_excess,
    }


def _trigger_assessment(
    records: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
) -> dict[str, object]:
    scipy_by_case = {
        record["case_id"]: record
        for record in comparisons
        if record["library"] == "scipy" and record["label"] == "exact"
    }
    assessments = []
    fired = False
    for record in records:
        comparator = scipy_by_case[record["case_id"]]
        scalar = record["timings"]["scalar"]
        warm_ratio = scalar["warm_median_seconds"] / comparator["elapsed_seconds"]
        values = {
            "warm_runtime_ratio": warm_ratio,
            "compiler_cost_ratio": None,
            "memory_family_ratio": record["memory_family_ratio"],
            "repeated_scaling_excess": record["repeated_scaling_excess"],
        }
        record_fired = (
            warm_ratio >= TRIGGERS["warm_runtime_ratio"]
            or record["memory_family_ratio"] >= TRIGGERS["memory_family_ratio"]
            or record["repeated_scaling_excess"] >= TRIGGERS["repeated_scaling_excess"]
        )
        fired = fired or record_fired
        assessments.append(
            {
                "case_id": record["case_id"],
                "values": values,
                "fired": record_fired,
                "compiler_ratio_reason": (
                    "SciPy's eager exact-node comparator has no matched JIT "
                    "compiler phase, so no compiler ratio is claimed."
                ),
            }
        )
    return {
        "thresholds": TRIGGERS,
        "records": assessments,
        "trigger_fired": fired,
        "decision": (
            "optimization_addendum_required" if fired else "no_runtime_change"
        ),
    }


def _with_digest(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "payload_sha256": _canonical_sha256(payload)}


def build_artifacts():
    comparisons = [dict(record) for record in run_comparators()]
    for record in comparisons:
        validate_comparison_record(record)
    source_revision = _git("rev-parse", "HEAD")
    source_hashes = {
        "benchmark": _sha256(Path(__file__)),
        "adapters": _sha256(ROOT / "scripts/quad_multidim_benchmark_adapters.py"),
        "truth_evidence": _sha256(ROOT / "docs/validation/quad-multidim-truth.json"),
    }
    common = {
        "schema_version": 1,
        "source_revision": source_revision,
        "source_hashes": source_hashes,
        "environment": _environment(),
    }
    comparison_artifact = _with_digest(
        {
            **common,
            "artifact_id": "quad.multidim.comparisons",
            "claim_boundary": (
                "Only records labelled exact support direct timing ratios. "
                "Family, strong, node, and capability matches are descriptive."
            ),
            "records": comparisons,
        }
    )
    records = [_benchmark_record(dimension) for dimension in DIMENSIONS]
    baseline_artifact = _with_digest(
        {
            **common,
            "artifact_id": "quad.multidim.performance-baseline",
            "manifest": {
                "dimensions": list(DIMENSIONS),
                "modes": [
                    "compile",
                    "scalar",
                    "vmap16",
                    "vmap128",
                    "jvp",
                    "gradient",
                    "same_domain_repeat",
                    "changing_parameter_repeat",
                ],
                "warm_repeats": WARM_REPEATS,
                "immutable": True,
            },
            "records": records,
            "trigger_assessment": _trigger_assessment(records, comparisons),
            "claim_boundary": (
                "Host-specific baseline for the frozen Sobol campaign only; "
                "it is not a universal library ranking."
            ),
        }
    )
    return comparison_artifact, baseline_artifact


def _validate_digest(artifact: dict[str, object]) -> None:
    stored = artifact.get("payload_sha256")
    payload = {key: value for key, value in artifact.items() if key != "payload_sha256"}
    if stored != _canonical_sha256(payload):
        raise ValueError(f"{artifact.get('artifact_id')} payload digest is stale")


def _check() -> None:
    expected_hashes = {
        "benchmark": _sha256(Path(__file__)),
        "adapters": _sha256(ROOT / "scripts/quad_multidim_benchmark_adapters.py"),
        "truth_evidence": _sha256(ROOT / "docs/validation/quad-multidim-truth.json"),
    }
    for path, artifact_id in (
        (COMPARISON_OUTPUT, "quad.multidim.comparisons"),
        (BASELINE_OUTPUT, "quad.multidim.performance-baseline"),
    ):
        artifact = json.loads(path.read_text())
        if artifact["artifact_id"] != artifact_id:
            raise ValueError(f"unexpected artifact identity in {path}")
        _validate_digest(artifact)
        if artifact["source_hashes"] != expected_hashes:
            raise ValueError(f"source hashes are stale in {path}")
        _git("cat-file", "-e", f"{artifact['source_revision']}^{{commit}}")
        print(f"fresh: {path.relative_to(ROOT)}")


def _emit() -> None:
    if not _tree_is_clean():
        raise RuntimeError("evidence emission requires a clean worktree")
    comparisons, baseline = build_artifacts()
    COMPARISON_OUTPUT.write_text(_canonical_json(comparisons))
    BASELINE_OUTPUT.write_text(_canonical_json(baseline))
    print(f"wrote: {COMPARISON_OUTPUT.relative_to(ROOT)}")
    print(f"wrote: {BASELINE_OUTPUT.relative_to(ROOT)}")
    print(
        "trigger decision:",
        baseline["trigger_assessment"]["decision"],
    )


def _explore() -> None:
    _comparisons, baseline = build_artifacts()
    print(_canonical_json(baseline["trigger_assessment"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("baseline",), required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--explore", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.emit and args.allow_dirty:
        parser.error("--allow-dirty is never valid for evidence emission")
    if args.explore and not args.allow_dirty and not _tree_is_clean():
        parser.error("dirty exploratory runs require --allow-dirty")
    if args.emit:
        _emit()
    elif args.check:
        _check()
    else:
        _explore()


if __name__ == "__main__":
    main()
