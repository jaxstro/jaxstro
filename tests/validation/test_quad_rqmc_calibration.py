import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxstro import quad
from jaxstro.quad._qmc_interval import spent_alpha
from jaxstro.quad._scramble import scramble_integers
from jaxstro.quad._sobol import sobol_integer_points, sobol_points

ARTIFACT = Path("docs/validation/quad-rqmc-calibration.json")
GENERATOR = Path("scripts/generate_quad_rqmc_evidence.py")


def _canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_rqmc_artifact_schema_hash_and_frozen_campaign():
    artifact = json.loads(ARTIFACT.read_text())
    assert artifact["schema_version"] == "1"
    assert artifact["artifact_id"] == "quad.rqmc-calibration"
    assert artifact["campaign"]["seed_count"] == 128
    assert artifact["campaign"]["confidence_level"] == 0.95
    assert len(artifact["campaign"]["records"]) == 4
    assert artifact["primary_point_integrand_evaluations"] == 3_145_728
    assert artifact["binomial_acceptance_band"] == {
        "count": [115, 127],
        "nominal_coverage": 0.95,
        "trials": 128,
        "two_sided_confidence": 0.99,
    }
    assert artifact["payload_sha256"] == _canonical_sha256(
        {key: value for key, value in artifact.items() if key != "payload_sha256"}
    )


def test_exact_binomial_band_uses_the_frozen_equal_tail_boundary_convention():
    trials = 128
    denominator = 20**trials

    def numerator(count):
        return math.comb(trials, count) * 19**count

    below_lower = sum(numerator(count) for count in range(115))
    through_lower = below_lower + numerator(115)
    above_upper = numerator(128)
    from_upper = numerator(127) + above_upper

    # Each tail receives exactly 0.005 = 1/200 probability.
    assert 200 * below_lower < denominator
    assert 200 * through_lower >= denominator
    assert 200 * above_upper < denominator
    assert 200 * from_upper >= denominator


def test_rqmc_artifact_is_fresh_and_reproducible_on_frozen_subset():
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_every_empirical_coverage_lies_in_exact_binomial_band():
    artifact = json.loads(ARTIFACT.read_text())
    lower, upper = artifact["binomial_acceptance_band"]["count"]
    for record in artifact["records"]:
        if record["method"] == "fixed":
            assert lower <= record["covered_count"] <= upper
            assert record["acceptance_policy"] == "two_sided_calibration"
        else:
            assert lower <= record["covered_count"] <= 128
            assert (
                record["acceptance_policy"]
                == "lower_validity_bound_with_conservatism_reported"
            )
        assert record["seed_count"] == 128
        assert record["estimate_sha256"] == _canonical_sha256(record["estimates"])
        assert record["reproducibility_subset"]["estimates"] == record["estimates"][:16]


def test_sequential_overcoverage_is_exposed_as_an_efficiency_limitation():
    artifact = json.loads(ARTIFACT.read_text())
    upper = artifact["binomial_acceptance_band"]["count"][1]
    sequential = [
        record for record in artifact["records"] if record["method"] == "sequential"
    ]
    assert all(record["covered_count"] > upper for record in sequential)
    assert all(record["mean_half_width_to_rmse"] > 1000.0 for record in sequential)


def test_deterministic_prefix_retains_unsw_example():
    expected = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.75, 0.25, 0.25),
            (0.25, 0.75, 0.75),
            (0.375, 0.375, 0.625),
            (0.875, 0.875, 0.125),
            (0.625, 0.125, 0.875),
            (0.125, 0.625, 0.375),
            (0.1875, 0.3125, 0.9375),
            (0.6875, 0.8125, 0.4375),
        ),
        dtype=jnp.float64,
    )
    assert jnp.array_equal(sobol_points(4, 3, jnp.float64)[:10], expected)


def test_replicate_stream_replays_and_changes_with_root_key():
    points = sobol_integer_points(6, 4, bits=24)

    def replicate_estimates(root_key):
        estimates = []
        for replicate in range(8):
            key = jax.random.fold_in(root_key, replicate)
            scrambled = scramble_integers(
                points,
                method=quad.LinearMatrixScramble(),
                key=key,
                bits=24,
            ).astype(jnp.float64) / (1 << 24)
            estimates.append(jnp.mean(jnp.prod(scrambled, axis=-1)))
        return jnp.asarray(estimates)

    first = replicate_estimates(jax.random.key(101))
    replay = replicate_estimates(jax.random.key(101))
    alternate = replicate_estimates(jax.random.key(102))
    assert jnp.array_equal(first, replay)
    assert jnp.any(first != alternate)


def test_alpha_allocation_identity_rejects_full_alpha_at_every_look():
    alpha = jnp.asarray(0.05, dtype=jnp.float64)
    allocated = sum(spent_alpha(alpha, inspection) for inspection in range(10000))
    mutated = sum(alpha for _inspection in range(2))
    assert allocated <= alpha
    assert jnp.allclose(allocated, alpha, rtol=2.0e-4)
    assert mutated > alpha
