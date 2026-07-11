"""Public-coordinate AD-vs-FD contracts away from geometric singularities."""

import jax.numpy as jnp
import pytest

from jaxstro.coords import (
    cartesian_to_spherical,
    cluster_to_galactic_cartesian,
    compute_parallax,
    compute_proper_motions,
    equatorial_to_galactic,
    galactic_to_equatorial,
    sky_tangent,
    spherical_to_cartesian,
    zenith_parallactic,
)
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
    Case(
        id="coords.cluster_to_galactic_cartesian.distance",
        direction="distance->galactic_cartesian",
        fn=lambda distance_pc: cluster_to_galactic_cartesian(
            jnp.array([[10.0, -4.0, 2.0]]),
            l_center_deg=41.0,
            b_center_deg=-18.0,
            distance_pc=distance_pc,
        ),
        param="distance_pc",
        theta0=1000.0,
        tol=1e-5,
    ),
    Case(
        id="coords.galactic_to_equatorial.latitude",
        direction="galactic_latitude->icrs",
        fn=lambda b_deg: jnp.stack(
            galactic_to_equatorial(jnp.array([41.0]), jnp.array([b_deg]))
        ),
        param="b_deg",
        theta0=18.0,
        tol=1e-5,
    ),
    Case(
        id="coords.equatorial_to_galactic.declination",
        direction="icrs_declination->galactic",
        fn=lambda dec_deg: jnp.stack(
            equatorial_to_galactic(jnp.array([130.0]), jnp.array([dec_deg]))
        ),
        param="dec_deg",
        theta0=-12.0,
        tol=1e-5,
    ),
    Case(
        id="coords.cartesian_to_spherical.scale",
        direction="cartesian_scale->spherical",
        fn=lambda scale: jnp.stack(
            cartesian_to_spherical(scale * jnp.array([[1.0, 2.0, 3.0]]))
        ),
        param="scale",
        theta0=2.0,
        tol=1e-5,
    ),
    Case(
        id="coords.spherical_to_cartesian.theta",
        direction="polar_angle->cartesian",
        fn=lambda theta: spherical_to_cartesian(
            jnp.array([3.0]), jnp.array([theta]), jnp.array([0.7])
        ),
        param="theta",
        theta0=1.1,
        tol=1e-5,
    ),
    Case(
        id="coords.zenith_parallactic.hour_angle",
        direction="hour_angle->observing_geometry",
        fn=lambda hour_angle: jnp.stack(
            zenith_parallactic(hour_angle, dec=-0.2, lat=-0.5)
        ),
        param="hour_angle",
        theta0=0.4,
        tol=1e-5,
    ),
    Case(
        id="coords.compute_proper_motions.distance",
        direction="distance->proper_motion",
        fn=lambda distance_pc: jnp.stack(
            compute_proper_motions(
                jnp.array([[10.0, -4.0, 2.0]]),
                jnp.array([[40.0, -15.0, 8.0]]),
                distance_pc=distance_pc,
                ra_center_deg=130.0,
                dec_center_deg=-20.0,
                psi_deg=15.0,
            )
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
