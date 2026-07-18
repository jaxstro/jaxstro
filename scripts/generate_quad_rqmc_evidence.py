#!/usr/bin/env python3
"""Generate and freshness-check the frozen Phase B3 RQMC campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from jaxstro.jaxconfig import enable_high_precision

enable_high_precision()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jaxstro import quad  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "validation" / "quad-rqmc-calibration.json"
BASE_SEED = 20260718
SUBSET_SEED_COUNT = 16

CAMPAIGN = {
    "seed_count": 128,
    "confidence_level": 0.95,
    "records": [
        {
            "case": "separable_polynomial",
            "dimension": 2,
            "method": "fixed",
            "level": 8,
            "replicates": 16,
        },
        {
            "case": "separable_exponential",
            "dimension": 8,
            "method": "fixed",
            "level": 8,
            "replicates": 16,
        },
        {
            "case": "low_effective_dimension",
            "dimension": 16,
            "method": "sequential",
            "schedule": [[6, 8], [7, 16], [8, 32]],
        },
        {
            "case": "rotated_smooth",
            "dimension": 4,
            "method": "sequential",
            "schedule": [[6, 8], [7, 16], [8, 32]],
        },
    ],
}


def _canonical_json(payload) -> str:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_binomial_band(
    *,
    trials: int,
    success_numerator: int,
    success_denominator: int,
    two_sided_confidence_numerator: int,
    two_sided_confidence_denominator: int,
) -> tuple[int, int]:
    success = success_numerator
    failure = success_denominator - success_numerator
    denominator = success_denominator**trials
    tail_denominator = 2 * two_sided_confidence_denominator
    tail_numerator = two_sided_confidence_denominator - two_sided_confidence_numerator

    cumulative = 0
    lower = 0
    for count in range(trials + 1):
        cumulative += (
            math.comb(trials, count) * success**count * failure ** (trials - count)
        )
        if tail_denominator * cumulative >= tail_numerator * denominator:
            lower = count
            break

    cumulative = 0
    upper = trials
    for count in range(trials, -1, -1):
        cumulative += (
            math.comb(trials, count) * success**count * failure ** (trials - count)
        )
        if tail_denominator * cumulative >= tail_numerator * denominator:
            upper = count
            break
    return lower, upper


def _case_definition(case: str, dimension: int):
    if case == "separable_polynomial":
        return (
            lambda x: jnp.prod(x * x, axis=-1),
            (1.0 / 3.0) ** dimension,
            (0.0, 1.0),
        )
    if case == "separable_exponential":
        return (
            lambda x: jnp.exp(-jnp.sum(x, axis=-1)),
            (1.0 - math.exp(-1.0)) ** dimension,
            (math.exp(-dimension), 1.0),
        )
    if case == "low_effective_dimension":
        return (
            lambda x: jnp.exp(x[:, 0] + 0.1 * x[:, 1]),
            math.expm1(1.0) * math.expm1(0.1) / 0.1,
            (1.0, math.exp(1.1)),
        )
    if case == "rotated_smooth":
        center = 0.5 * dimension
        return (
            lambda x: 1.0 + 0.1 * (jnp.sum(x, axis=-1) - center) ** 2,
            1.0 + 0.1 * dimension / 12.0,
            (1.0, 1.0 + 0.1 * center**2),
        )
    raise ValueError(f"unknown RQMC calibration case {case}")


def _method(record, estimate_bounds):
    if record["method"] == "fixed":
        return quad.ScrambledSobol(
            level=record["level"],
            replicates=record["replicates"],
            confidence_level=CAMPAIGN["confidence_level"],
        )
    return quad.AdaptiveScrambledSobol(
        schedule=tuple(tuple(row) for row in record["schedule"]),
        estimate_bounds=estimate_bounds,
        confidence_level=CAMPAIGN["confidence_level"],
    )


def _max_evaluations(record) -> int:
    if record["method"] == "fixed":
        return record["replicates"] * (1 << record["level"])
    final_level, final_replicates = record["schedule"][-1]
    return final_replicates * (1 << final_level)


def _keys(seed_count: int):
    root = jax.random.key(BASE_SEED)
    return jax.vmap(lambda index: jax.random.fold_in(root, index))(
        jnp.arange(seed_count, dtype=jnp.uint32)
    )


def _run_case(record, *, seed_count: int):
    dimension = record["dimension"]
    integrand, truth, estimate_bounds = _case_definition(
        record["case"],
        dimension,
    )
    method = _method(record, estimate_bounds)
    domain = quad.Hyperrectangle(
        jnp.zeros(dimension, dtype=jnp.float64),
        jnp.ones(dimension, dtype=jnp.float64),
    )
    max_evaluations = _max_evaluations(record)

    def solve(key):
        return quad.integrate(
            integrand,
            domain,
            method=method,
            key=key,
            epsabs=0.0,
            epsrel=0.0,
            max_evaluations=max_evaluations,
            gradient="stop",
        )

    results = jax.jit(jax.vmap(solve))(_keys(seed_count))
    estimates = np.asarray(results.value)
    half_widths = np.asarray(results.error.estimate)
    statuses = np.asarray(results.status)
    work = np.asarray(results.work.evaluations)
    covered = np.abs(estimates - truth) <= half_widths
    return {
        "estimates": [float(value) for value in estimates],
        "half_widths": [float(value) for value in half_widths],
        "statuses": [int(value) for value in statuses],
        "work": [int(value) for value in work],
        "truth": float(truth),
        "covered": [bool(value) for value in covered],
    }


def _artifact_record(record):
    result = _run_case(record, seed_count=CAMPAIGN["seed_count"])
    estimates = result["estimates"]
    errors = np.asarray(estimates) - result["truth"]
    covered_count = sum(result["covered"])
    rmse = float(np.sqrt(np.mean(errors * errors)))
    mean_half_width = float(np.mean(result["half_widths"]))
    return {
        **record,
        "seed_count": CAMPAIGN["seed_count"],
        "truth": result["truth"],
        "covered_count": covered_count,
        "coverage": covered_count / CAMPAIGN["seed_count"],
        "acceptance_policy": (
            "two_sided_calibration"
            if record["method"] == "fixed"
            else "lower_validity_bound_with_conservatism_reported"
        ),
        "rmse": rmse,
        "mean_half_width": mean_half_width,
        "mean_half_width_to_rmse": mean_half_width / rmse,
        "estimate_sha256": _canonical_sha256(estimates),
        "estimates": estimates,
        "half_widths": result["half_widths"],
        "statuses": result["statuses"],
        "work": result["work"],
        "reproducibility_subset": {
            "seed_count": SUBSET_SEED_COUNT,
            "estimates": estimates[:SUBSET_SEED_COUNT],
            "half_widths": result["half_widths"][:SUBSET_SEED_COUNT],
        },
    }


def build_artifact():
    lower, upper = _exact_binomial_band(
        trials=CAMPAIGN["seed_count"],
        success_numerator=19,
        success_denominator=20,
        two_sided_confidence_numerator=99,
        two_sided_confidence_denominator=100,
    )
    payload = {
        "artifact_id": "quad.rqmc-calibration",
        "schema_version": "1",
        "generator": "scripts/generate_quad_rqmc_evidence.py",
        "generator_sha256": _file_sha256(Path(__file__)),
        "campaign": CAMPAIGN,
        "primary_point_integrand_evaluations": sum(
            CAMPAIGN["seed_count"] * _max_evaluations(record)
            for record in CAMPAIGN["records"]
        ),
        "binomial_acceptance_band": {
            "nominal_coverage": 0.95,
            "two_sided_confidence": 0.99,
            "trials": CAMPAIGN["seed_count"],
            "count": [lower, upper],
        },
        "records": [_artifact_record(record) for record in CAMPAIGN["records"]],
        "claim_boundary": (
            "Empirical calibration supports the frozen campaign only; fixed-look "
            "Student-t intervals are nominal randomized-replicate intervals, and "
            "sequential intervals require the declared finite estimate bounds."
        ),
    }
    return {**payload, "payload_sha256": _canonical_sha256(payload)}


def _validate_artifact(artifact) -> None:
    if artifact["artifact_id"] != "quad.rqmc-calibration":
        raise ValueError("unexpected RQMC artifact identity")
    if artifact["schema_version"] != "1":
        raise ValueError("unexpected RQMC artifact schema")
    if artifact["campaign"] != CAMPAIGN:
        raise ValueError("RQMC campaign controls are stale")
    if artifact["generator_sha256"] != _file_sha256(Path(__file__)):
        raise ValueError("RQMC generator digest is stale")
    payload = {key: value for key, value in artifact.items() if key != "payload_sha256"}
    if artifact["payload_sha256"] != _canonical_sha256(payload):
        raise ValueError("RQMC payload digest is stale")
    lower, upper = artifact["binomial_acceptance_band"]["count"]
    for record in artifact["records"]:
        fixed_failed = record["method"] == "fixed" and not (
            lower <= record["covered_count"] <= upper
        )
        sequential_failed = (
            record["method"] == "sequential" and record["covered_count"] < lower
        )
        if fixed_failed or sequential_failed:
            raise ValueError(f"{record['case']} coverage lies outside the frozen band")
        if record["estimate_sha256"] != _canonical_sha256(record["estimates"]):
            raise ValueError(f"{record['case']} estimate digest is stale")


def check_artifact() -> None:
    artifact = json.loads(OUTPUT.read_text())
    _validate_artifact(artifact)
    for controls, stored in zip(
        CAMPAIGN["records"],
        artifact["records"],
        strict=True,
    ):
        replay = _run_case(controls, seed_count=SUBSET_SEED_COUNT)
        expected = stored["reproducibility_subset"]
        if replay["estimates"] != expected["estimates"]:
            raise ValueError(
                f"{controls['case']} subset estimates are not reproducible"
            )
        if replay["half_widths"] != expected["half_widths"]:
            raise ValueError(
                f"{controls['case']} subset intervals are not reproducible"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate-slow", action="store_true")
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check == (args.emit or args.regenerate_slow):
        parser.error("choose --check or --regenerate-slow --emit")
    if args.emit and not args.regenerate_slow:
        parser.error("--emit requires --regenerate-slow")
    if args.regenerate_slow and not args.emit:
        parser.error("--regenerate-slow requires --emit")
    if args.check:
        check_artifact()
        print(f"fresh: {OUTPUT.relative_to(REPO_ROOT)}")
        return
    artifact = build_artifact()
    OUTPUT.write_text(_canonical_json(artifact))
    print(f"wrote: {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
