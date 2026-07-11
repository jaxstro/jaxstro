"""Public-coordinate AD-vs-FD contracts away from geometric singularities."""

import jax.numpy as jnp
import pytest

from jaxstro.coords import compute_parallax, sky_tangent
from jaxstro.testing import Case, audit_entry_point

_CASES = (
    Case(
        id="coords.sky_tangent.distance",
        direction="distance->sky_tangent",
        fn=lambda distance_pc: sky_tangent(
            jnp.array([[10.0, -4.0, 2.0]]),
            distance_pc=distance_pc,
            ra_center_deg=180.0,
            dec_center_deg=20.0,
            warn_large_field=False,
        ),
        param="distance_pc",
        theta0=1000.0,
        tol=1e-5,
    ),
    Case(
        id="coords.compute_parallax.distance",
        direction="distance->parallax",
        fn=lambda distance_pc: compute_parallax(
            jnp.array([[10.0, -4.0, 2.0]]), distance_pc=distance_pc
        ),
        param="distance_pc",
        theta0=1000.0,
        tol=1e-5,
    ),
)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.id)
def test_public_coordinate_gradient_contract(case):
    """Interior public coordinate contracts agree with central FD truth."""
    result = audit_entry_point(case)
    assert result.status == "clean"
