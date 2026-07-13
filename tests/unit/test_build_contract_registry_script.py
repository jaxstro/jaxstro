"""Freshness contract for generated scientific-contract artifacts."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_contract_registry_artifacts_are_fresh() -> None:
    subprocess.run(
        [sys.executable, "scripts/build_contract_registry.py", "--check"],
        cwd=ROOT,
        check=True,
    )
