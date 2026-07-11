"""Executable contracts for the Getting Started website page."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[2] / "docs/00-getting-started/index.md"


def _worked_example() -> str:
    source = PAGE.read_text().split(
        "## A first worked example: safe math + a root-find", maxsplit=1
    )[1]
    return source.split("```python", maxsplit=1)[1].split("```", maxsplit=1)[0]


def _run_with_probe(source: str) -> dict[str, float]:
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    return json.loads(proc.stdout.splitlines()[-1])


def test_getting_started_uses_a_parameter_independent_newton_path():
    source = _worked_example()

    assert "from jaxstro.numerics.rootfinding import newton" in source
    assert "x0=2.0" in source
    assert "bisect" not in source


def test_getting_started_example_matches_analytic_and_finite_difference_checks():
    source = _worked_example()
    result = _run_with_probe(
        source
        + textwrap.dedent(
            """
            import json
            print(json.dumps({
                "scale_heights": float(scale_heights),
                "analytic_scale_heights": float(analytic_scale_heights),
                "ad_grad": float(ad_grad),
                "fd_grad": float(fd_grad),
            }))
            """
        )
    )

    assert result["scale_heights"] == pytest.approx(2.3025850929940455)
    assert result["analytic_scale_heights"] == pytest.approx(result["scale_heights"])
    assert result["ad_grad"] == pytest.approx(-10.0, rel=1e-12)
    assert result["fd_grad"] == pytest.approx(result["ad_grad"], rel=1e-7)
