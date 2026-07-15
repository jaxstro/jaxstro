"""Independent exactness and pinned-provenance checks for Gauss-Kronrod data."""

import importlib.util
import json
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import jax.numpy as jnp
import pytest

from jaxstro.quad import GaussKronrod
from jaxstro.quad._gk import gauss_kronrod_data

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_quadpack_gk_fixture.py"
REFERENCE = ROOT / "tests" / "fixtures" / "quadpack" / "gk-reference.json"
DEGREES = {
    15: (13, 23),
    21: (19, 31),
    31: (29, 47),
    41: (39, 61),
    51: (49, 77),
    61: (59, 91),
}


def _moment(degree: int) -> float:
    return 0.0 if degree % 2 else 2.0 / (degree + 1)


@pytest.mark.parametrize("pair", [15, 21, 31, 41, 51, 61])
def test_gauss_and_kronrod_rules_meet_declared_polynomial_degrees(pair) -> None:
    data = gauss_kronrod_data(GaussKronrod(pair=pair), dtype=jnp.float64)
    gauss_degree, kronrod_degree = DEGREES[pair]
    for degree in range(kronrod_degree + 1):
        observed = jnp.sum(data.kronrod_weights * data.nodes**degree)
        assert jnp.allclose(observed, _moment(degree), rtol=3e-12, atol=3e-14), (
            pair,
            "kronrod",
            degree,
            observed,
        )
    for degree in range(gauss_degree + 1):
        observed = jnp.sum(data.gauss_weights * data.nodes**degree)
        assert jnp.allclose(observed, _moment(degree), rtol=3e-12, atol=3e-14), (
            pair,
            "gauss",
            degree,
            observed,
        )


def test_generated_quadpack_artifacts_are_fresh() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_quadpack_freshness_checker_rejects_mutated_content(tmp_path) -> None:
    specification = importlib.util.spec_from_file_location("gk_fixture_owner", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    stale = tmp_path / "stale.json"
    stale.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale generated fixture"):
        module._check_or_emit(stale, "expected\n", emit=False)


def test_runtime_positive_tables_match_pinned_netlib_provenance() -> None:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    for pair_text, entry in reference["pairs"].items():
        pair = int(pair_text)
        data = gauss_kronrod_data(GaussKronrod(pair=pair), dtype=jnp.float64)
        xgk = jnp.asarray([float(value) for value in entry["xgk"]])
        wgk = jnp.asarray([float(value) for value in entry["wgk"]])
        aligned_gauss = [0.0] * len(entry["xgk"])
        for one_based_index, value in enumerate(entry["wg"], start=1):
            aligned_gauss[2 * one_based_index - 1] = float(value)
        midpoint = pair // 2
        assert jnp.array_equal(data.nodes[midpoint:], xgk[::-1])
        assert jnp.array_equal(data.kronrod_weights[midpoint:], wgk[::-1])
        assert jnp.array_equal(
            data.gauss_weights[midpoint:], jnp.asarray(aligned_gauss[::-1])
        )


@pytest.mark.parametrize("pair", [15, 21, 31, 41, 51, 61])
def test_declared_degrees_are_maximal_in_high_precision(pair) -> None:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))["pairs"][str(pair)]
    gauss_degree, kronrod_degree = DEGREES[pair]
    xgk = [Decimal(value) for value in reference["xgk"]]
    wgk = [Decimal(value) for value in reference["wgk"]]
    aligned_gauss = [Decimal(0)] * len(xgk)
    for one_based_index, value in enumerate(reference["wg"], start=1):
        aligned_gauss[2 * one_based_index - 1] = Decimal(value)

    def positive_half_moment(weights, degree):
        return 2 * sum(
            weight * node**degree
            for node, weight in zip(xgk[:-1], weights[:-1], strict=True)
        )

    with localcontext() as context:
        context.prec = 80
        gauss_next = gauss_degree + 1
        kronrod_next = kronrod_degree + 1
        assert positive_half_moment(aligned_gauss, gauss_next) != Decimal(2) / Decimal(
            gauss_next + 1
        )
        assert positive_half_moment(wgk, kronrod_next) != Decimal(2) / Decimal(
            kronrod_next + 1
        )
