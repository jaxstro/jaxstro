from __future__ import annotations

import copy
import importlib.util
import tomllib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxstro.quad._sobol import sobol_points

ROOT = Path(__file__).parents[2]
ADAPTER_PATH = ROOT / "scripts/quad_multidim_benchmark_adapters.py"


def _load_adapters():
    spec = importlib.util.spec_from_file_location(
        "quad_multidim_benchmark_adapters",
        ADAPTER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_dependencies_are_isolated_from_runtime():
    root = tomllib.loads((ROOT / "pyproject.toml").read_text())
    runtime = root["project"]["dependencies"]
    assert not any(
        dependency.lower().startswith(("scipy", "tasmanian", "torchquad"))
        for dependency in runtime
    )
    assert root["dependency-groups"]["benchmark-multidim"] == ["scipy==1.16.0"]

    tasmanian = tomllib.loads(
        (
            ROOT / "laboratory/quad-multidim-comparison/tasmanian/pyproject.toml"
        ).read_text()
    )
    torchquad = tomllib.loads(
        (
            ROOT / "laboratory/quad-multidim-comparison/torchquad/pyproject.toml"
        ).read_text()
    )
    assert tasmanian["project"]["dependencies"] == ["tasmanian==8.2"]
    assert tasmanian["tool"]["uv"]["extra-build-dependencies"]["tasmanian"] == [
        "cmake>=3.25,<5"
    ]
    assert "torchquad==0.5.0" in torchquad["project"]["dependencies"]
    assert "jax==0.10.1" in torchquad["project"]["dependencies"]
    assert "jaxlib==0.10.1" in torchquad["project"]["dependencies"]


def test_isolated_locks_preserve_comparator_and_jax_pins():
    tasmanian_lock = (
        ROOT / "laboratory/quad-multidim-comparison/tasmanian/uv.lock"
    ).read_text()
    torchquad_lock = (
        ROOT / "laboratory/quad-multidim-comparison/torchquad/uv.lock"
    ).read_text()
    assert 'name = "tasmanian"\nversion = "8.2"' in tasmanian_lock
    assert 'name = "torchquad"\nversion = "0.5.0"' in torchquad_lock
    assert 'name = "jax"\nversion = "0.10.1"' in torchquad_lock
    assert 'name = "jaxlib"\nversion = "0.10.1"' in torchquad_lock


def test_comparison_schema_rejects_missing_calibration():
    adapters = _load_adapters()
    record = adapters.scipy_sobol_record(2)
    missing_description = copy.deepcopy(record)
    del missing_description["controls"]["matched_control_description"]
    with pytest.raises(ValueError, match="matched-control"):
        adapters.validate_comparison_record(missing_description)

    unsupported = copy.deepcopy(record)
    unsupported["label"] = "better"
    with pytest.raises(ValueError, match="unsupported comparison label"):
        adapters.validate_comparison_record(unsupported)


@pytest.mark.parametrize("dimension", [2, 4, 8, 16])
def test_scipy_sobol_adapter_is_bit_exact(dimension):
    adapters = _load_adapters()
    record = adapters.scipy_sobol_record(dimension)
    scipy_points = adapters.qmc.Sobol(
        dimension,
        scramble=False,
        bits=53,
    ).random_base2(8)
    jaxstro_points = np.asarray(sobol_points(8, dimension, jnp.float64, bits=53))
    assert np.array_equal(scipy_points, jaxstro_points)
    assert record["label"] == "exact"
    assert record["evaluations"] == 256


def test_external_comparator_adapters_execute_and_match_truth():
    adapters = _load_adapters()
    records = [
        adapters.scipy_cubature_record(),
        adapters.tasmanian_sparse_record(),
        adapters.torchquad_tensor_record(),
    ]
    for record in records:
        adapters.validate_comparison_record(record)
        assert record["truth_error"] < 2.0e-6
        assert record["elapsed_seconds"] > 0.0
    torchquad = records[-1]
    assert torchquad["controls"]["gradient"] == pytest.approx(
        torchquad["value"],
        rel=2.0e-6,
    )
