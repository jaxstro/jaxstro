"""Calibrated external adapters for the Phase B4 multidimensional benchmark."""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import numpy as np
import scipy
from scipy import integrate as scipy_integrate
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"exact", "strong-match", "node-matched", "family-matched", "capability"}


class ComparisonRecord(TypedDict):
    library: str
    version: str
    label: Literal[
        "exact",
        "strong-match",
        "node-matched",
        "family-matched",
        "capability",
    ]
    case_id: str
    controls: dict[str, object]
    value: object
    truth_error: float
    evaluations: int | None
    elapsed_seconds: float


def validate_comparison_record(record: ComparisonRecord) -> None:
    """Reject uncalibrated or structurally incomplete comparison claims."""
    required = {
        "library",
        "version",
        "label",
        "case_id",
        "controls",
        "value",
        "truth_error",
        "evaluations",
        "elapsed_seconds",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"comparison record missing fields: {sorted(missing)}")
    if record["label"] not in LABELS:
        raise ValueError(f"unsupported comparison label: {record['label']}")
    description = record["controls"].get("matched_control_description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("comparison record requires a matched-control description")
    if record["elapsed_seconds"] < 0.0:
        raise ValueError("comparison elapsed_seconds must be nonnegative")
    if record["truth_error"] < 0.0 or not math.isfinite(record["truth_error"]):
        raise ValueError("comparison truth_error must be finite and nonnegative")


def scipy_cubature_record() -> ComparisonRecord:
    """Run SciPy's Genz-Malik family on the frozen rotated-smooth case."""
    dimension = 4
    truth = 1.0 + 0.1 * dimension / 12.0

    def fun(x):
        return 1.0 + 0.1 * (np.sum(x, axis=-1) - 0.5 * dimension) ** 2

    started = time.perf_counter()
    result = scipy_integrate.cubature(
        fun,
        np.zeros(dimension),
        np.ones(dimension),
        rule="genz-malik",
        atol=1.0e-9,
        rtol=1.0e-9,
        max_subdivisions=1_024,
    )
    elapsed = time.perf_counter() - started
    value = float(np.asarray(result.estimate))
    record: ComparisonRecord = {
        "library": "scipy",
        "version": scipy.__version__,
        "label": "family-matched",
        "case_id": "rotated_smooth_d4",
        "controls": {
            "rule": "genz-malik",
            "atol": 1.0e-9,
            "rtol": 1.0e-9,
            "max_subdivisions": 1_024,
            "matched_control_description": (
                "Same 4D unit hyperrectangle, integrand, tolerances, and "
                "Genz-Malik rule family; regional controller details differ."
            ),
        },
        "value": value,
        "truth_error": abs(value - truth),
        "evaluations": None,
        "elapsed_seconds": elapsed,
    }
    validate_comparison_record(record)
    return record


def scipy_sobol_record(dimension: int, *, level: int = 8) -> ComparisonRecord:
    """Run the exact unscrambled Sobol point set used by Jaxstro."""
    started = time.perf_counter()
    points = qmc.Sobol(
        dimension,
        scramble=False,
        bits=53,
    ).random_base2(level)
    values = np.exp(-np.sum(points, axis=-1))
    value = float(np.mean(values))
    elapsed = time.perf_counter() - started
    truth = math.expm1(-1.0) ** dimension
    record: ComparisonRecord = {
        "library": "scipy",
        "version": scipy.__version__,
        "label": "exact",
        "case_id": f"sobol_exponential_d{dimension}",
        "controls": {
            "dimension": dimension,
            "level": level,
            "scramble": False,
            "bits": 53,
            "matched_control_description": (
                "Bit-exact 53-bit unscrambled Sobol nodes, equal-weight mean, "
                "same ordering, domain, and separable exponential integrand."
            ),
        },
        "value": value,
        "truth_error": abs(value - truth),
        "evaluations": 1 << level,
        "elapsed_seconds": elapsed,
    }
    validate_comparison_record(record)
    return record


def _run_isolated(project: Path, code: str) -> dict[str, object]:
    command = (
        "uv",
        "run",
        "--python",
        "3.11",
        "--project",
        str(project),
        "--locked",
        "python",
        "-c",
        code,
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def tasmanian_sparse_record() -> ComparisonRecord:
    """Run one node-matched-purpose sparse polynomial moment in isolation."""
    project = ROOT / "laboratory/quad-multidim-comparison/tasmanian"
    code = r"""
import importlib.metadata as metadata, json, time
from pathlib import Path
import numpy as np
import TasmanianConfig as config
lib = Path(config.__file__).resolve().parents[2]
config.__path_libsparsegrid__ = str(lib / "libtasmaniansparsegrid.dylib")
config.__path_libdream__ = str(lib / "libtasmaniandream.dylib")
config.__path_libcaddons__ = str(lib / "libtasmaniancaddons.dylib")
import Tasmanian
started = time.perf_counter()
grid = Tasmanian.SparseGrid()
grid.makeGlobalGrid(2, 0, 4, "level", "clenshaw-curtis")
raw_points = grid.getPoints()
weights = grid.getQuadratureWeights() / 4.0
points = 0.5 * (raw_points + 1.0)
value = float(np.dot(weights, np.prod(1.0 + points + points**2, axis=1)))
print(json.dumps({"version": metadata.version("Tasmanian"), "value": value,
                  "evaluations": int(points.shape[0]),
                  "elapsed_seconds": time.perf_counter() - started}))
"""
    payload = _run_isolated(project, code)
    truth = (11.0 / 6.0) ** 2
    record: ComparisonRecord = {
        "library": "tasmanian",
        "version": str(payload["version"]),
        "label": "strong-match",
        "case_id": "tensor_polynomial_d2",
        "controls": {
            "depth": 4,
            "type": "level",
            "rule": "clenshaw-curtis",
            "macos_relocation_workaround": True,
            "matched_control_description": (
                "Same 2D unit hyperrectangle and polynomial moment with a "
                "Clenshaw-Curtis sparse family; index conventions are library-specific."
            ),
        },
        "value": payload["value"],
        "truth_error": abs(float(cast(Any, payload["value"])) - truth),
        "evaluations": int(cast(Any, payload["evaluations"])),
        "elapsed_seconds": float(cast(Any, payload["elapsed_seconds"])),
    }
    validate_comparison_record(record)
    return record


def torchquad_tensor_record() -> ComparisonRecord:
    """Run a JAX-backed differentiable Gaussian tensor formula in isolation."""
    project = ROOT / "laboratory/quad-multidim-comparison/torchquad"
    code = r"""
import importlib.metadata as metadata, json, logging, time
import jax, jax.numpy as jnp
from loguru import logger
from torchquad import GaussLegendre
logger.remove()
method = GaussLegendre()
domain = jnp.asarray([[0.0, 1.0], [0.0, 1.0]])
def solve(amplitude):
    return method.integrate(
        lambda x: amplitude * jnp.prod(1.0 + x + x*x, axis=-1),
        dim=2, N=144, integration_domain=domain, backend="jax")
started = time.perf_counter()
value, gradient = jax.value_and_grad(solve)(jnp.asarray(1.0))
jax.block_until_ready(value)
print(json.dumps({"version": metadata.version("torchquad"),
                  "jax_version": jax.__version__,
                  "value": float(value), "gradient": float(gradient),
                  "evaluations": 144,
                  "elapsed_seconds": time.perf_counter() - started}))
"""
    payload = _run_isolated(project, code)
    truth = (11.0 / 6.0) ** 2
    record: ComparisonRecord = {
        "library": "torchquad",
        "version": str(payload["version"]),
        "label": "node-matched",
        "case_id": "tensor_polynomial_d2",
        "controls": {
            "backend": "jax",
            "jax_version": payload["jax_version"],
            "method": "GaussLegendre",
            "nodes_per_axis": 12,
            "gradient": payload["gradient"],
            "matched_control_description": (
                "Same 12-point-per-axis Gauss-Legendre tensor formula, "
                "2D unit hyperrectangle, integrand, and amplitude derivative."
            ),
        },
        "value": payload["value"],
        "truth_error": abs(float(cast(Any, payload["value"])) - truth),
        "evaluations": int(cast(Any, payload["evaluations"])),
        "elapsed_seconds": float(cast(Any, payload["elapsed_seconds"])),
    }
    validate_comparison_record(record)
    return record


def run_comparators() -> list[ComparisonRecord]:
    records = [
        scipy_cubature_record(),
        *(scipy_sobol_record(dimension) for dimension in (2, 4, 8, 16)),
        tasmanian_sparse_record(),
        torchquad_tensor_record(),
    ]
    for record in records:
        validate_comparison_record(record)
    return records


__all__ = [
    "ComparisonRecord",
    "run_comparators",
    "scipy_cubature_record",
    "scipy_sobol_record",
    "tasmanian_sparse_record",
    "torchquad_tensor_record",
    "validate_comparison_record",
]
