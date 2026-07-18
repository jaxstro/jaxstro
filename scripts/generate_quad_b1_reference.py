#!/usr/bin/env python3
"""Emit or check the closed-form Phase B1 Genz reference artifact."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "tests/validation/data/quad-b1-genz-reference.json"
FORMULA_SET_ID = "genz-unit-hypercube-six-family-v1"
GENERATOR_VERSION = "1.0.0"
PRECISION_DECIMAL_DIGITS = 80
DIMENSIONS = (2, 4, 6, 8)
FAMILIES = (
    "oscillatory",
    "product_peak",
    "corner_peak",
    "gaussian",
    "continuous",
    "discontinuous",
)
FORMULA_IDS = {
    "oscillatory": f"{FORMULA_SET_ID}:oscillatory",
    "product_peak": f"{FORMULA_SET_ID}:product-peak",
    "corner_peak": f"{FORMULA_SET_ID}:corner-peak",
    "gaussian": f"{FORMULA_SET_ID}:gaussian",
    "continuous": f"{FORMULA_SET_ID}:continuous",
    "discontinuous": f"{FORMULA_SET_ID}:discontinuous-first-two-axes",
}


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _mp_fraction(value: Fraction) -> mp.mpf:
    return mp.mpf(value.numerator) / mp.mpf(value.denominator)


def _parameter_fractions(dimension: int) -> tuple[list[Fraction], list[Fraction]]:
    a = [Fraction(35 + 5 * index, 100) for index in range(1, dimension + 1)]
    u = [Fraction(index, dimension + 1) for index in range(1, dimension + 1)]
    return a, u


def _oscillatory(a: list[mp.mpf], u: list[mp.mpf]) -> mp.mpf:
    amplitude = mp.fprod(2 * mp.sin(value / 2) / value for value in a)
    return amplitude * mp.cos(2 * mp.pi * u[0] + mp.fsum(a) / 2)


def _product_peak(a: list[mp.mpf], u: list[mp.mpf]) -> mp.mpf:
    return mp.fprod(
        ai * (mp.atan(ai * (1 - ui)) + mp.atan(ai * ui))
        for ai, ui in zip(a, u, strict=True)
    )


def _corner_peak(a: list[mp.mpf]) -> mp.mpf:
    dimension = len(a)
    alternating_sum = mp.mpf("0")
    for mask in itertools.product((0, 1), repeat=dimension):
        denominator = 1 + mp.fsum(
            ai * selected for ai, selected in zip(a, mask, strict=True)
        )
        alternating_sum += (-1) ** sum(mask) / denominator
    return alternating_sum / (math.factorial(dimension) * mp.fprod(a))


def _gaussian(a: list[mp.mpf], u: list[mp.mpf]) -> mp.mpf:
    return mp.fprod(
        mp.sqrt(mp.pi) / (2 * ai) * (mp.erf(ai * (1 - ui)) + mp.erf(ai * ui))
        for ai, ui in zip(a, u, strict=True)
    )


def _continuous(a: list[mp.mpf], u: list[mp.mpf]) -> mp.mpf:
    return mp.fprod(
        (2 - mp.exp(-ai * ui) - mp.exp(-ai * (1 - ui))) / ai
        for ai, ui in zip(a, u, strict=True)
    )


def _discontinuous(a: list[mp.mpf], u: list[mp.mpf]) -> mp.mpf:
    upper = [u[index] if index < 2 else mp.mpf("1") for index in range(len(a))]
    return mp.fprod(
        mp.expm1(ai * upper_i) / ai for ai, upper_i in zip(a, upper, strict=True)
    )


def _truth(family: str, a: list[mp.mpf], u: list[mp.mpf]) -> mp.mpf:
    if family == "oscillatory":
        return _oscillatory(a, u)
    if family == "product_peak":
        return _product_peak(a, u)
    if family == "corner_peak":
        return _corner_peak(a)
    if family == "gaussian":
        return _gaussian(a, u)
    if family == "continuous":
        return _continuous(a, u)
    if family == "discontinuous":
        return _discontinuous(a, u)
    raise ValueError(f"unknown Genz family: {family}")


def _records() -> list[dict]:
    records = []
    for dimension in DIMENSIONS:
        a_fraction, u_fraction = _parameter_fractions(dimension)
        a = [_mp_fraction(value) for value in a_fraction]
        u = [_mp_fraction(value) for value in u_fraction]
        for family in FAMILIES:
            records.append(
                {
                    "a": [_fraction_text(value) for value in a_fraction],
                    "dimension": dimension,
                    "family": family,
                    "formula_id": FORMULA_IDS[family],
                    "truth_decimal": mp.nstr(
                        _truth(family, a, u),
                        n=PRECISION_DECIMAL_DIGITS,
                        strip_zeros=False,
                    ),
                    "u": [_fraction_text(value) for value in u_fraction],
                }
            )
    return records


def _artifact() -> dict:
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "b4_carry_forward": {
            "adaptive_cubature": {
                "dimensions": [2, 4, 6, 8],
                "required_metrics": [
                    "compile_time",
                    "warm_runtime",
                    "process_memory",
                    "device_memory",
                    "dtype",
                    "payload_shape",
                    "reachable_store_capacity",
                ],
            },
            "adaptive_tensor": {
                "dimensions": [5, 6, 7, 8],
                "required_metrics": [
                    "compile_time",
                    "warm_runtime",
                    "process_memory",
                    "device_memory",
                    "dtype",
                    "payload",
                    "capacity",
                ],
            },
        },
        "formula_set_id": FORMULA_SET_ID,
        "generator": {
            "precision_decimal_digits": PRECISION_DECIMAL_DIGITS,
            "source_sha256": source_sha256,
            "version": GENERATOR_VERSION,
        },
        "manifest": {
            "a_rule": "0.35 + 0.05 * arange(1, dimension + 1)",
            "dimensions": list(DIMENSIONS),
            "families": list(FAMILIES),
            "u_rule": "arange(1, dimension + 1) / (dimension + 1)",
        },
        "records": _records(),
        "schema_version": 1,
    }


def _serialized_artifact() -> bytes:
    return (json.dumps(_artifact(), indent=2, sort_keys=True) + "\n").encode()


def _emit() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(_serialized_artifact())
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


def _check() -> int:
    expected = _serialized_artifact()
    if not OUTPUT_PATH.is_file():
        print(f"missing {OUTPUT_PATH.relative_to(ROOT)}")
        return 1
    actual = OUTPUT_PATH.read_bytes()
    if actual != expected:
        print(f"stale {OUTPUT_PATH.relative_to(ROOT)}")
        return 1
    print(f"fresh {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return _emit() if args.emit else _check()


if __name__ == "__main__":
    raise SystemExit(main())
