#!/usr/bin/env python3
"""Measure isolated-process RSS for the Phase B Sobol replay decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jaxstro import quad  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/validation/quad-multidim-memory.json"
RUN_CARD = ROOT / "docs/superpowers/specs/2026-08-29-quad-phase-b-memory-run-card.md"
DIMENSIONS = (2, 4, 8, 16)
LEVELS = (8, 12, 16)
FORMULAS = ("sobol", "scrambled_sobol")
PAYLOADS = ("scalar", "array")
MODES = ("primal", "replay_gradient")
SCRAMBLED_REPLICATES = 8
MATERIAL_INCREMENT_BYTES = 10 * 1024**3
RSS_PATTERN = re.compile(r"\s*(\d+)\s+maximum resident set size")
RECORD_PREFIX = "QUAD_MEMORY_RECORD="


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _ready(value: Any) -> Any:
    return jax.tree.map(
        lambda leaf: (
            jax.block_until_ready(leaf) if hasattr(leaf, "block_until_ready") else leaf
        ),
        value,
    )


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
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _method(formula: str, level: int):
    if formula == "sobol":
        return quad.Sobol(level=level, bits=53), None
    if formula == "scrambled_sobol":
        return (
            quad.ScrambledSobol(level=level, replicates=SCRAMBLED_REPLICATES),
            jax.random.key(20260829),
        )
    raise ValueError(f"unknown formula: {formula}")


def _payload(x, amplitude, payload: str):
    base = amplitude * jnp.exp(-jnp.sum(x, axis=-1))
    if payload == "scalar":
        return base
    if payload == "array":
        return jnp.stack(
            (
                base,
                base * (1.0 + x[..., 0]),
                base * (1.0 + x[..., -1]),
                base * (1.0 + jnp.sum(x * x, axis=-1)),
            ),
            axis=-1,
        )
    raise ValueError(f"unknown payload: {payload}")


def _case_kernel(*, dimension: int, level: int, formula: str, payload: str, mode: str):
    method, key = _method(formula, level)
    domain = quad.Hyperrectangle(
        jnp.zeros(dimension, dtype=jnp.float64),
        jnp.ones(dimension, dtype=jnp.float64),
    )
    point_count = 1 << level
    max_evaluations = point_count * (
        SCRAMBLED_REPLICATES if formula == "scrambled_sobol" else 1
    )

    def solve(amplitude):
        return quad.integrate(
            lambda x, scale: _payload(x, scale, payload),
            domain,
            args=amplitude,
            method=method,
            key=key,
            epsabs=0.0,
            epsrel=0.0,
            max_evaluations=max_evaluations,
            gradient="replay",
        ).value

    if mode == "primal":
        return jax.jit(solve)
    if mode == "replay_gradient":
        return jax.jit(jax.grad(lambda amplitude: jnp.sum(solve(amplitude))))
    raise ValueError(f"unknown mode: {mode}")


def _unsupported_case(formula: str, payload: str) -> str | None:
    if formula == "scrambled_sobol" and payload == "array":
        return "Phase B randomized QMC confidence intervals support real scalar payloads only"
    return None


def run_case(*, dimension: int, level: int, formula: str, payload: str, mode: str):
    """Run one supported case in the current fresh process."""
    unsupported_reason = _unsupported_case(formula, payload)
    case = {
        "dimension": dimension,
        "formula": formula,
        "level": level,
        "mode": mode,
        "payload": payload,
    }
    if unsupported_reason is not None:
        return {
            **case,
            "outcome": "contract_unsupported",
            "reason": unsupported_reason,
        }
    kernel = _case_kernel(**case)
    amplitude = jnp.asarray(1.0, dtype=jnp.float64)
    started = time.perf_counter()
    compiled = _ready(kernel(amplitude))
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warmed = _ready(kernel(amplitude))
    warm_seconds = time.perf_counter() - started
    return {
        **case,
        "compile_seconds": compile_seconds,
        "outcome": "measured",
        "output_shape": list(jnp.shape(compiled)),
        "warm_output_shape": list(jnp.shape(warmed)),
        "warm_seconds": warm_seconds,
    }


def _parse_rss(time_stderr: str) -> int:
    match = RSS_PATTERN.search(time_stderr)
    if match is None:
        raise ValueError(f"could not parse macOS peak RSS:\n{time_stderr}")
    return int(match.group(1))


def _inner_command(case: dict[str, object]) -> tuple[str, ...]:
    return (
        "/usr/bin/time",
        "-l",
        sys.executable,
        str(Path(__file__).resolve()),
        "--case",
        "--dimension",
        str(case["dimension"]),
        "--level",
        str(case["level"]),
        "--formula",
        str(case["formula"]),
        "--payload",
        str(case["payload"]),
        "--mode",
        str(case["mode"]),
    )


def _parse_case_record(stdout: str) -> dict[str, object]:
    for line in stdout.splitlines():
        if line.startswith(RECORD_PREFIX):
            return json.loads(line.removeprefix(RECORD_PREFIX))
    raise ValueError(f"memory case emitted no record:\n{stdout}")


def measure_case(case: dict[str, object]) -> dict[str, object]:
    """Measure one case through macOS time in an isolated child process."""
    command = _inner_command(case)
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"memory case failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    record = _parse_case_record(completed.stdout)
    if record["outcome"] == "measured":
        record["peak_rss_bytes"] = _parse_rss(completed.stderr)
        record["command"] = list(command)
    return record


def _cases() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "dimension": dimension,
            "level": level,
            "formula": formula,
            "payload": payload,
            "mode": mode,
        }
        for dimension in DIMENSIONS
        for level in LEVELS
        for formula in FORMULAS
        for payload in PAYLOADS
        for mode in MODES
    )


def _case_identity(record: dict[str, object]) -> tuple[object, ...]:
    return tuple(
        record[key] for key in ("dimension", "level", "formula", "payload", "mode")
    )


def _paired_materiality(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_identity = {_case_identity(record): record for record in records}
    pairs = []
    for record in records:
        if record["outcome"] != "measured" or record["mode"] != "replay_gradient":
            continue
        primal = by_identity[(*_case_identity(record)[:-1], "primal")]
        if primal["outcome"] != "measured":
            raise ValueError("replay case has no measured primal pair")
        increment = int(record["peak_rss_bytes"]) - int(primal["peak_rss_bytes"])
        pairs.append(
            {
                "dimension": record["dimension"],
                "formula": record["formula"],
                "level": record["level"],
                "payload": record["payload"],
                "primal_peak_rss_bytes": primal["peak_rss_bytes"],
                "replay_peak_rss_bytes": record["peak_rss_bytes"],
                "replay_increment_bytes": increment,
                "material": increment >= MATERIAL_INCREMENT_BYTES,
            }
        )
    return pairs


def build_artifact(records: list[dict[str, object]]) -> dict[str, object]:
    pairs = _paired_materiality(records)
    material_cases = [pair for pair in pairs if pair["material"]]
    return {
        "artifact_id": "quad.multidim.observed-memory",
        "claim_boundary": (
            "Fresh-process peak RSS for the frozen CPU replay campaign. It does "
            "not measure separate device memory or establish a universal memory claim."
        ),
        "device_memory_metric": "unavailable: active CPU backend",
        "environment": _environment(),
        "generator": "scripts/measure_quad_multidim_memory.py",
        "generator_sha256": _sha256(Path(__file__)),
        "materiality": {
            "criterion": "replay_peak_rss_minus_matched_primal_peak_rss >= 10 GiB",
            "increment_bytes": MATERIAL_INCREMENT_BYTES,
            "optimization_warranted": bool(material_cases),
        },
        "pairs": pairs,
        "records": records,
        "run_card": str(RUN_CARD.relative_to(ROOT)),
        "schema_version": 1,
        "source_revision": _git("rev-parse", "HEAD"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case", action="store_true")
    mode.add_argument("--campaign", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--dimension", type=int, choices=DIMENSIONS)
    parser.add_argument("--level", type=int, choices=LEVELS)
    parser.add_argument("--formula", choices=FORMULAS)
    parser.add_argument("--payload", choices=PAYLOADS)
    parser.add_argument("--mode", choices=MODES)
    args = parser.parse_args()
    if args.case:
        required = (args.dimension, args.level, args.formula, args.payload, args.mode)
        if any(value is None for value in required):
            parser.error("--case requires dimension, level, formula, payload, and mode")
        record = run_case(
            dimension=args.dimension,
            level=args.level,
            formula=args.formula,
            payload=args.payload,
            mode=args.mode,
        )
        print(RECORD_PREFIX + json.dumps(record, sort_keys=True, allow_nan=False))
        return 0
    if args.campaign:
        artifact = build_artifact([measure_case(case) for case in _cases()])
        OUTPUT.write_text(_canonical_json(artifact), encoding="utf-8")
        print(f"wrote: {OUTPUT.relative_to(ROOT)}")
        print(
            "optimization warranted:", artifact["materiality"]["optimization_warranted"]
        )
        return 0
    artifact = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if artifact["generator_sha256"] != _sha256(Path(__file__)):
        raise ValueError("observed-memory artifact generator hash is stale")
    if artifact["materiality"]["increment_bytes"] != MATERIAL_INCREMENT_BYTES:
        raise ValueError("observed-memory materiality threshold is stale")
    if artifact["run_card"] != str(RUN_CARD.relative_to(ROOT)):
        raise ValueError("observed-memory run card identity is stale")
    print(f"fresh: {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
