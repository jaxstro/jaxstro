"""Executable contracts for the numerical claims in README quick-start examples."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
DOCS_AUDIT = ROOT / "docs/audits/2026-07-11-docs-currency-audit.md"


def _python_block_after(heading: str) -> str:
    source = README.read_text()
    section = source.split(heading, maxsplit=1)[1]
    return section.split("```python", maxsplit=1)[1].split("```", maxsplit=1)[0]


def _run_readme_program(source: str) -> dict[str, object]:
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
    return json.loads(proc.stdout)


def test_readme_coordinate_block_is_standalone_and_executable():
    source = _python_block_after("### Coordinate transforms")
    result = _run_readme_program(
        source
        + textwrap.dedent(
            """
            import json
            print(json.dumps({
                "shape": list(ra_dec.shape),
                "parallax_mas": float(parallax_mas[0]),
            }))
            """
        )
    )

    assert result["shape"] == [2, 2]
    assert result["parallax_mas"] == pytest.approx(10.0)


def test_readme_compensated_sum_block_is_standalone_and_backend_honest():
    source = _python_block_after("### Compensated summation")
    result = _run_readme_program(
        source
        + textwrap.dedent(
            """
            import json
            print(json.dumps({
                "standard": float(standard),
                "compensated": float(compensated),
            }))
            """
        )
    )

    assert result["compensated"] == pytest.approx(2.0)
    assert result["standard"] != result["compensated"]


def test_readme_scopes_jax_transform_claims_and_uses_current_units_language():
    readme = README.read_text()
    audit = DOCS_AUDIT.read_text()

    assert "Everything works with `jax.jit`, `jax.vmap`, and `jax.grad`." not in readme
    assert "Full compatibility with `jit`, `vmap`, and `grad`" not in readme
    assert "Root-finding (fully differentiable)" not in readme
    assert "Rounded solar-mass conversion [g]" in readme
    assert "legacy solar-mass scale" not in audit


def test_readme_precision_constants_units_and_coordinates():
    result = _run_readme_program(
        """
        from jaxstro.jaxconfig import enable_high_precision
        enable_high_precision()

        import json
        import jax
        import jax.numpy as jnp
        from jaxstro import constants as C, units as U
        from jaxstro.coords import (
            compute_parallax,
            galactic_to_equatorial,
            sky_tangent,
        )

        mass = C.MSUN_G
        radius = C.RSUN_CM
        escape_speed_km_s = jnp.sqrt(2.0 * C.G_CGS * mass / radius) / C.KM_CM

        positions_pc = jnp.array([[1.0, 0.5, -0.2], [0.0, 1.0, 0.3]])
        ra_dec = sky_tangent(
            positions_pc,
            distance_pc=1000.0,
            ra_center_deg=180.0,
        )
        ra, dec = galactic_to_equatorial(45.0, 30.0)
        parallax_mas = compute_parallax(
            jnp.zeros((1, 3)),
            distance_pc=100.0,
        )

        print(json.dumps({
            "x64": bool(jax.config.x64_enabled),
            "escape_speed_km_s": float(escape_speed_km_s),
            "G_dynamical": U.ASTRO_DYNAMICAL.G,
            "ra_dec_shape": list(ra_dec.shape),
            "galactic_transform_finite": bool(jnp.isfinite(ra) & jnp.isfinite(dec)),
            "parallax_mas": float(parallax_mas[0]),
        }))
        """
    )

    assert result["x64"] is True
    assert result["escape_speed_km_s"] == pytest.approx(617.7, rel=2e-3)
    assert result["G_dynamical"] == pytest.approx(0.00450, rel=2e-3)
    assert result["ra_dec_shape"] == [2, 2]
    assert result["galactic_transform_finite"] is True
    assert result["parallax_mas"] == pytest.approx(10.0)


def test_readme_rootfinding_and_compensated_sum():
    result = _run_readme_program(
        """
        from jaxstro.jaxconfig import enable_high_precision
        enable_high_precision()

        import json
        import jax.numpy as jnp
        from jaxstro.numerics.compensated import compensated_sum_array
        from jaxstro.numerics.rootfinding import bisect, newton

        root_bisect = bisect(lambda x: x**2 - 2.0, a=1.0, b=2.0)
        root_newton = newton(lambda x: x**2 - 2.0, x0=1.5)
        terms = jnp.array([1e16, 1.0, -1e16, 1.0])

        print(json.dumps({
            "root_bisect": float(root_bisect),
            "root_newton": float(root_newton),
            "compensated_sum": float(compensated_sum_array(terms)),
        }))
        """
    )

    assert result["root_bisect"] == pytest.approx(2.0**0.5, rel=1e-12)
    assert result["root_newton"] == pytest.approx(2.0**0.5, rel=1e-12)
    assert result["compensated_sum"] == pytest.approx(2.0)
