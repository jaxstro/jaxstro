from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
GENERATOR = ROOT / "scripts/generate_quad_multidim_evidence.py"
TRUTH_ARTIFACT = ROOT / "docs/validation/quad-multidim-truth.json"
REPLAY_ARTIFACT = ROOT / "docs/validation/quad-multidim-replay.json"
GENZ_ARTIFACT = ROOT / "tests/validation/data/quad-b1-genz-reference.json"

COMMON_CONTROLS = {
    "epsabs": 1.0e-9,
    "epsrel": 1.0e-9,
    "gradient": "replay",
    "max_evaluations": 65_536,
}
SPARSE_CONTROLS = {
    **COMMON_CONTROLS,
    "max_indices": 128,
    "max_frontier": 2_049,
    "max_nodes": 65_536,
}
CUBATURE_CONTROLS = {**COMMON_CONTROLS, "max_regions": 1_024}
FROZEN_RECORDS = {
    "tensor_polynomial": (
        2,
        "tensor_gauss_3",
        "TensorProduct",
        COMMON_CONTROLS,
        2.0e-13,
        7,
        "exact tensor polynomial moment",
    ),
    "beta_product": (
        3,
        "smolyak_5",
        "Smolyak",
        SPARSE_CONTROLS,
        2.0e-13,
        0,
        "sparse polynomial moment",
    ),
    "separable_exponential": (
        4,
        "smolyak_5",
        "Smolyak",
        SPARSE_CONTROLS,
        2.0e-7,
        8,
        "fixed sparse accuracy threshold; estimator does not claim convergence",
    ),
    "rotated_smooth": (
        4,
        "cubature",
        "AdaptiveCubature",
        CUBATURE_CONTROLS,
        2.0e-12,
        0,
        "rotated smooth cubature convergence",
    ),
    **{
        family: (
            2,
            "tensor_gauss_12",
            "TensorProduct",
            COMMON_CONTROLS,
            5.0e-5,
            7,
            "Genz reference accuracy threshold",
        )
        for family in (
            "genz_oscillatory",
            "genz_product_peak",
            "genz_corner_peak",
            "genz_gaussian",
        )
    },
    **{
        family: (
            2,
            "cubature",
            "AdaptiveCubature",
            CUBATURE_CONTROLS,
            5.0e-5,
            0,
            "Genz reference accuracy threshold",
        )
        for family in ("genz_continuous", "genz_discontinuous")
    },
    "localized_peak": (
        2,
        "adaptive_tensor",
        "AdaptiveTensorClenshawCurtis",
        COMMON_CONTROLS,
        2.0e-7,
        0,
        "localized adaptive-tensor convergence",
    ),
    "boundary_layer": (
        2,
        "tensor_gauss_20",
        "TensorProduct",
        COMMON_CONTROLS,
        2.0e-7,
        7,
        "fixed boundary-layer accuracy threshold",
    ),
}


def _load_generator():
    spec = importlib.util.spec_from_file_location("quad_multidim_evidence", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generated():
    return _load_generator().build_artifacts()


def test_registry_has_exactly_the_frozen_family_order(generated):
    truth, _replay = generated
    assert tuple(truth["validation_families"]) == (
        "tensor_polynomial",
        "beta_product",
        "separable_exponential",
        "rotated_smooth",
        "genz_oscillatory",
        "genz_product_peak",
        "genz_corner_peak",
        "genz_gaussian",
        "genz_continuous",
        "genz_discontinuous",
        "localized_peak",
        "boundary_layer",
    )
    assert [record["family"] for record in truth["records"]] == list(
        truth["validation_families"]
    )


def test_truth_records_meet_predeclared_thresholds(generated):
    truth, _replay = generated
    for record in truth["records"]:
        expected = FROZEN_RECORDS[record["family"]]
        (
            dimension,
            method_id,
            method,
            controls,
            tolerance,
            status,
            expected_claim,
        ) = expected
        assert record["dimension"] == dimension
        assert record["method_id"] == method_id
        assert record["method"] == method
        assert record["controls"] == controls
        assert record["tolerance"] == tolerance
        assert record["status"] == status
        assert record["expected_claim"] == expected_claim
        assert record["absolute_error"] <= tolerance, record["family"]
        assert record["replay_gradient"] == pytest.approx(
            record["value"], rel=2.0e-12, abs=2.0e-12
        )


def test_nonanalytic_references_have_complete_provenance(generated):
    truth, _replay = generated
    upstream = json.loads(GENZ_ARTIFACT.read_text())
    artifact_sha256 = hashlib.sha256(GENZ_ARTIFACT.read_bytes()).hexdigest()
    references = [
        record["truth_source"]
        for record in truth["records"]
        if record["truth_source"]["kind"] == "reference"
    ]
    assert len(references) == 6
    for source in references:
        assert source["generator"] == "scripts/generate_quad_b1_reference.py"
        assert source["external_owner"]
        assert source["precision_decimal_digits"] == 80
        assert source["parameters"]
        assert (
            source["generator_source_sha256"] == upstream["generator"]["source_sha256"]
        )
        assert source["reference_artifact_sha256"] == artifact_sha256
        assert source["absolute_uncertainty"] > 0.0


def test_astro_fixtures_match_truth_and_quantity_representation(generated):
    _truth, replay = generated
    records = replay["astro_records"]
    assert [record["case_id"] for record in records] == [
        "projected_plummer_aperture",
        "diagonal_gaussian_mass",
        "diagonal_gaussian_second_moment_axis_0",
        "diagonal_gaussian_second_moment_axis_1",
        "population_moment",
        "separable_selection",
    ]
    for record in records:
        assert record["absolute_error"] <= 2.0e-7, record["case_id"]
        assert record["raw_quantity_absolute_difference"] <= 2.0e-12
        assert record["replay_parameter"]
        assert record["replay_gradient"] == pytest.approx(
            record["replay_gradient_truth"],
            rel=2.0e-7,
            abs=2.0e-10,
        )
        assert record["replay_gradient_absolute_error"] <= 2.0e-7
        assert record["specification"]["equation"]
        assert record["specification"]["bounds"]
        assert record["specification"]["truth"]
        assert record["specification"]["units"]


def test_committed_artifacts_are_canonical_and_fresh(generated):
    module = _load_generator()
    truth, replay = generated
    assert TRUTH_ARTIFACT.read_text() == module._canonical_json(truth)
    assert REPLAY_ARTIFACT.read_text() == module._canonical_json(replay)
    assert json.loads(TRUTH_ARTIFACT.read_text())["environment"]["jax_enable_x64"]
