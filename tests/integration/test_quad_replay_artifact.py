import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_quad_replay_derivative_artifact_is_fresh():
    subprocess.run(
        [sys.executable, "scripts/generate_quad_replay_evidence.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_evidence_index_is_fresh_with_quad_replay_artifact():
    subprocess.run(
        [sys.executable, "scripts/build_evidence_index.py", "--check"],
        cwd=ROOT,
        check=True,
    )
