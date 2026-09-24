"""ADR 0015: solver dependencies are imported only by the modules that use them."""

import subprocess
import sys

import pytest

SOLVER_DEPENDENCIES = ("diffrax", "optimistix", "sympy")
LIGHT_MODULES = (
    "jaxstro",
    "jaxstro.units",
    "jaxstro.constants",
    "jaxstro.coords",
    "jaxstro.numerics",
    "jaxstro.quad",
)


def _loaded_after_import(module: str) -> list[str]:
    program = (
        f"import sys, {module}\n"
        f"print(','.join(d for d in {SOLVER_DEPENDENCIES!r} if d in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    return [name for name in completed.stdout.strip().split(",") if name]


@pytest.mark.parametrize("module", LIGHT_MODULES)
def test_light_modules_load_no_solver_dependency(module: str) -> None:
    assert _loaded_after_import(module) == []


def test_lane_emden_names_stay_public_through_numerics() -> None:
    from jaxstro import numerics
    from jaxstro.numerics import lane_emden

    assert numerics.solve_isothermal is lane_emden.solve_isothermal
    assert numerics.lane_emden is lane_emden
    assert "solve_polytrope" in numerics.__all__
