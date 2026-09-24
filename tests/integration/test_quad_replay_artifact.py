import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from jaxstro.evidence import artifact_from_dict, artifact_to_dict, check_artifact
from tests.validation._freshness import Origin, assert_fresh

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/generate_quad_replay_evidence.py"
# The artifact records platform "macOS-26.1-arm64-arm-64bit-Mach-O".
ARTIFACT_ORIGIN = Origin(system="macOS", machine="arm64")
# Measured errors computed from other recorded fields; the gates' pass flags
# and the values they are computed from are still compared everywhere.
DERIVED_ERROR_FIELDS = (
    "observed",
    "observed_primal_error",
    "primal_relative_error",
    "derivative_relative_error",
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("quad_replay_evidence", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves its module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quad_replay_derivative_artifact_is_fresh():
    module = _load_generator()
    payload, environment = module.run_evidence()
    module._validate(payload)
    current = module.build_artifact(payload, environment)
    stored = artifact_from_dict(json.loads(module.OUTPUT.read_text(encoding="utf-8")))
    check_artifact(module.REPORT, stored)
    assert_fresh(
        artifact_to_dict(stored)["method_payload"],
        artifact_to_dict(current)["method_payload"],
        origin=ARTIFACT_ORIGIN,
        derived=DERIVED_ERROR_FIELDS,
    )


def test_evidence_index_is_fresh_with_quad_replay_artifact():
    subprocess.run(
        [sys.executable, "scripts/build_evidence_index.py", "--check"],
        cwd=ROOT,
        check=True,
    )
