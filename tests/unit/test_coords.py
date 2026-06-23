"""
Tests for jaxstro.coords module.

TDD: Write tests FIRST, verify they FAIL, then implement.
"""

import jax
import jax.numpy as jnp

# These imports should fail until implementation exists
from jaxstro.coords import (
    cartesian_to_spherical,
    cluster_to_galactic_cartesian,
    compute_parallax,
    compute_proper_motions,
    equatorial_to_galactic,
    galactic_to_equatorial,
    sky_tangent,
    spherical_to_cartesian,
)


class TestSkyTangent:
    """Tests for sky_tangent coordinate transform."""

    def test_center_returns_center(self):
        """Star at cluster center should return pointing center."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        result = sky_tangent(
            positions, distance_pc=1000.0, ra_center_deg=180.0, dec_center_deg=0.0
        )
        assert result.shape == (1, 2)
        assert jnp.abs(result[0, 0] - 180.0) < 1e-6  # RA
        assert jnp.abs(result[0, 1] - 0.0) < 1e-6  # Dec

    def test_east_offset(self):
        """Star 1 pc East at 1 kpc should have ~0.057 deg RA offset."""
        positions = jnp.array([[1.0, 0.0, 0.0]])  # 1 pc East
        result = sky_tangent(
            positions, distance_pc=1000.0, ra_center_deg=180.0, dec_center_deg=0.0
        )
        # 1/1000 rad ≈ 0.057 deg
        expected_ra_offset = jnp.rad2deg(1.0 / 1000.0)
        assert jnp.abs(result[0, 0] - (180.0 + expected_ra_offset)) < 0.01

    def test_north_offset(self):
        """Star 1 pc North at 1 kpc should have ~0.057 deg Dec offset."""
        positions = jnp.array([[0.0, 1.0, 0.0]])  # 1 pc North
        result = sky_tangent(
            positions, distance_pc=1000.0, ra_center_deg=180.0, dec_center_deg=0.0
        )
        expected_dec_offset = jnp.rad2deg(1.0 / 1000.0)
        assert jnp.abs(result[0, 1] - expected_dec_offset) < 0.01

    def test_los_offset_no_change(self):
        """Star offset along LOS should have minimal RA/Dec change."""
        positions = jnp.array([[0.0, 0.0, 10.0]])  # 10 pc along LOS
        result = sky_tangent(
            positions, distance_pc=1000.0, ra_center_deg=180.0, dec_center_deg=0.0
        )
        assert jnp.abs(result[0, 0] - 180.0) < 0.01
        assert jnp.abs(result[0, 1] - 0.0) < 0.01

    def test_batch_processing(self):
        """Should handle batch of positions."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        result = sky_tangent(positions, distance_pc=1000.0)
        assert result.shape == (4, 2)

    def test_differentiable_wrt_distance(self):
        """Should be differentiable w.r.t. distance."""
        positions = jnp.array([[1.0, 0.0, 0.0]])

        def loss(distance_pc):
            coords = sky_tangent(positions, distance_pc=distance_pc)
            return coords[0, 0]  # RA

        grad_fn = jax.grad(loss)
        grad = grad_fn(1000.0)
        assert jnp.isfinite(grad)

    def test_roll_angle(self):
        """Roll angle should rotate E-N frame."""
        positions = jnp.array([[1.0, 0.0, 0.0]])  # 1 pc in x

        # No roll: offset in RA
        result_no_roll = sky_tangent(positions, distance_pc=1000.0, psi_deg=0.0)

        # 90 deg roll: offset should now be in Dec
        result_roll = sky_tangent(positions, distance_pc=1000.0, psi_deg=90.0)

        # With roll, RA should be near center, Dec should have offset
        assert jnp.abs(result_no_roll[0, 0] - 180.0) > 0.01  # RA offset
        # Roll should change the output (different coordinate mapping)
        assert not jnp.allclose(result_no_roll, result_roll)


class TestGalacticEquatorial:
    """Tests for Galactic <-> Equatorial conversions."""

    def test_galactic_center(self):
        """Galactic center (l=0, b=0) should be at RA~266.4, Dec~-29."""
        l = jnp.array([0.0])
        b = jnp.array([0.0])
        ra, dec = galactic_to_equatorial(l, b)
        assert jnp.abs(ra[0] - 266.4) < 0.5
        assert jnp.abs(dec[0] - (-29.0)) < 0.5

    def test_north_galactic_pole(self):
        """NGP (l=any, b=90) should be at RA~192.9, Dec~27.1."""
        l = jnp.array([0.0])
        b = jnp.array([90.0])
        ra, dec = galactic_to_equatorial(l, b)
        assert jnp.abs(ra[0] - 192.9) < 0.5
        assert jnp.abs(dec[0] - 27.1) < 0.5

    def test_roundtrip_galactic_equatorial(self):
        """Converting galactic -> equatorial -> galactic should be identity."""
        l_orig = jnp.array([45.0, 180.0, 270.0])
        b_orig = jnp.array([30.0, -45.0, 60.0])

        ra, dec = galactic_to_equatorial(l_orig, b_orig)
        l_back, b_back = equatorial_to_galactic(ra, dec)

        # Allow for wrapping at 360
        l_diff = jnp.minimum(jnp.abs(l_back - l_orig), 360 - jnp.abs(l_back - l_orig))
        assert jnp.all(l_diff < 1e-8)
        assert jnp.allclose(b_back, b_orig, atol=1e-8)

    def test_batch_processing(self):
        """Should handle arrays."""
        l = jnp.array([0.0, 90.0, 180.0, 270.0])
        b = jnp.array([0.0, 0.0, 0.0, 0.0])
        ra, dec = galactic_to_equatorial(l, b)
        assert ra.shape == (4,)
        assert dec.shape == (4,)


class TestSphericalCartesian:
    """Tests for spherical <-> Cartesian conversions."""

    def test_x_axis(self):
        """Point on +x axis: r=1, theta=pi/2, phi=0."""
        positions = jnp.array([[1.0, 0.0, 0.0]])
        r, theta, phi = cartesian_to_spherical(positions)
        assert jnp.abs(r[0] - 1.0) < 1e-10
        assert jnp.abs(theta[0] - jnp.pi / 2) < 1e-10
        assert jnp.abs(phi[0] - 0.0) < 1e-10

    def test_y_axis(self):
        """Point on +y axis: r=1, theta=pi/2, phi=pi/2."""
        positions = jnp.array([[0.0, 1.0, 0.0]])
        r, theta, phi = cartesian_to_spherical(positions)
        assert jnp.abs(r[0] - 1.0) < 1e-10
        assert jnp.abs(theta[0] - jnp.pi / 2) < 1e-10
        assert jnp.abs(phi[0] - jnp.pi / 2) < 1e-10

    def test_z_axis(self):
        """Point on +z axis: r=1, theta=0, phi=any."""
        positions = jnp.array([[0.0, 0.0, 1.0]])
        r, theta, phi = cartesian_to_spherical(positions)
        assert jnp.abs(r[0] - 1.0) < 1e-10
        assert jnp.abs(theta[0] - 0.0) < 1e-10  # theta=0 at +z

    def test_roundtrip_cartesian_spherical(self):
        """Converting cartesian -> spherical -> cartesian should be identity."""
        positions_orig = jnp.array(
            [
                [1.0, 2.0, 3.0],
                [-1.0, 0.5, 2.0],
                [0.0, 0.0, 5.0],
            ]
        )
        r, theta, phi = cartesian_to_spherical(positions_orig)
        positions_back = spherical_to_cartesian(r, theta, phi)
        assert jnp.allclose(positions_back, positions_orig, atol=1e-10)


class TestParallax:
    """Tests for parallax computation."""

    def test_1kpc_gives_1mas(self):
        """Star at 1 kpc should have parallax ~1 mas."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        parallax = compute_parallax(positions, distance_pc=1000.0)
        assert jnp.abs(parallax[0] - 1.0) < 0.01

    def test_10kpc_gives_0p1mas(self):
        """Star at 10 kpc should have parallax ~0.1 mas."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        parallax = compute_parallax(positions, distance_pc=10000.0)
        assert jnp.abs(parallax[0] - 0.1) < 0.01

    def test_100pc_gives_10mas(self):
        """Star at 100 pc should have parallax ~10 mas."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        parallax = compute_parallax(positions, distance_pc=100.0)
        assert jnp.abs(parallax[0] - 10.0) < 0.1

    def test_batch_processing(self):
        """Should handle arrays."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],  # Slight offset
                [0.0, 0.0, 100.0],  # LOS offset
            ]
        )
        parallax = compute_parallax(positions, distance_pc=1000.0)
        assert parallax.shape == (3,)
        # All should be approximately 1 mas (small internal offsets)
        assert jnp.all(jnp.abs(parallax - 1.0) < 0.2)


class TestProperMotions:
    """Tests for proper motion computation."""

    def test_tangential_velocity_at_1kpc(self):
        """100 km/s tangential at 1 kpc should give ~21 mas/yr."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        velocities = jnp.array([[100.0, 0.0, 0.0]])  # 100 km/s in x (tangential)
        mu_ra, mu_dec = compute_proper_motions(
            positions, velocities, distance_pc=1000.0
        )
        # mu = v / (4.74 * d) = 100 / (4.74 * 1) ≈ 21.1 mas/yr
        expected = 100.0 / 4.74047
        assert jnp.abs(mu_ra[0] - expected) < 1.0

    def test_radial_velocity_gives_zero_pm(self):
        """Purely radial velocity should give zero proper motion."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        velocities = jnp.array(
            [[0.0, 0.0, -100.0]]
        )  # 100 km/s toward observer (radial)
        mu_ra, mu_dec = compute_proper_motions(
            positions, velocities, distance_pc=1000.0
        )
        # Radial motion -> zero proper motion
        assert jnp.abs(mu_ra[0]) < 1.0
        assert jnp.abs(mu_dec[0]) < 1.0

    def test_y_velocity_gives_dec_pm(self):
        """Velocity in y should give proper motion in Dec."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        velocities = jnp.array([[0.0, 100.0, 0.0]])  # 100 km/s in y
        mu_ra, mu_dec = compute_proper_motions(
            positions, velocities, distance_pc=1000.0
        )
        expected = 100.0 / 4.74047
        assert jnp.abs(mu_dec[0] - expected) < 1.0

    def test_batch_processing(self):
        """Should handle arrays."""
        positions = jnp.zeros((10, 3))
        velocities = jnp.ones((10, 3)) * 50.0
        mu_ra, mu_dec = compute_proper_motions(
            positions, velocities, distance_pc=1000.0
        )
        assert mu_ra.shape == (10,)
        assert mu_dec.shape == (10,)


class TestDifferentiability:
    """Tests for JAX differentiability."""

    def test_sky_tangent_grad(self):
        """sky_tangent should be differentiable."""
        positions = jnp.array([[1.0, 1.0, 0.0]])

        def loss(distance_pc):
            coords = sky_tangent(positions, distance_pc=distance_pc)
            return jnp.sum(coords)

        grad = jax.grad(loss)(1000.0)
        assert jnp.isfinite(grad)

    def test_parallax_grad(self):
        """compute_parallax should be differentiable."""
        positions = jnp.array([[0.0, 0.0, 0.0]])

        def loss(distance_pc):
            return jnp.sum(compute_parallax(positions, distance_pc=distance_pc))

        grad = jax.grad(loss)(1000.0)
        assert jnp.isfinite(grad)

    def test_proper_motion_grad(self):
        """compute_proper_motions should be differentiable."""
        positions = jnp.array([[0.0, 0.0, 0.0]])
        velocities = jnp.array([[100.0, 50.0, 0.0]])

        def loss(distance_pc):
            mu_ra, mu_dec = compute_proper_motions(
                positions, velocities, distance_pc=distance_pc
            )
            return jnp.sum(mu_ra) + jnp.sum(mu_dec)

        grad = jax.grad(loss)(1000.0)
        assert jnp.isfinite(grad)


class TestJITCompilation:
    """Tests for JIT compilation."""

    def test_sky_tangent_jit(self):
        """sky_tangent should be JIT-compilable."""
        positions = jnp.array([[1.0, 0.0, 0.0]])

        @jax.jit
        def compute(positions, distance_pc):
            return sky_tangent(positions, distance_pc=distance_pc)

        result = compute(positions, 1000.0)
        assert result.shape == (1, 2)

    def test_parallax_jit(self):
        """compute_parallax should be JIT-compilable."""
        positions = jnp.array([[0.0, 0.0, 0.0]])

        @jax.jit
        def compute(positions, distance_pc):
            return compute_parallax(positions, distance_pc=distance_pc)

        result = compute(positions, 1000.0)
        assert result.shape == (1,)


class TestClusterToGalacticCartesian:
    """Tests for the cluster->heliocentric-Galactic-Cartesian placement transform."""

    def test_center_toward_galactic_center(self):
        """Center star at (l=0, b=0, 1 kpc) -> (1000, 0, 0) pc toward GC."""
        uvw = cluster_to_galactic_cartesian(
            jnp.array([[0.0, 0.0, 0.0]]), 0.0, 0.0, 1000.0
        )
        assert jnp.allclose(uvw[0], jnp.array([1000.0, 0.0, 0.0]), atol=1e-6)

    def test_center_at_l90(self):
        """Center at (l=90, b=0, 1 kpc) -> (0, 1000, 0) pc."""
        uvw = cluster_to_galactic_cartesian(
            jnp.array([[0.0, 0.0, 0.0]]), 90.0, 0.0, 1000.0
        )
        assert jnp.allclose(uvw[0], jnp.array([0.0, 1000.0, 0.0]), atol=1e-6)

    def test_center_at_ngp(self):
        """Center at (b=90, 1 kpc) -> (0, 0, 1000) pc toward NGP."""
        uvw = cluster_to_galactic_cartesian(
            jnp.array([[0.0, 0.0, 0.0]]), 0.0, 90.0, 1000.0
        )
        assert jnp.allclose(uvw[0], jnp.array([0.0, 0.0, 1000.0]), atol=1e-6)

    def test_los_offset_increases_distance(self):
        """A +z (LOS) offset increases heliocentric distance by exactly that amount."""
        uvw = cluster_to_galactic_cartesian(
            jnp.array([[0.0, 0.0, 5.0]]), 12.0, -3.0, 1000.0
        )
        r, _, _ = cartesian_to_spherical(uvw)
        assert jnp.abs(float(r[0]) - 1005.0) < 1e-6

    def test_east_offset_increases_l(self):
        """A +x (East) offset increases Galactic longitude l."""
        base = cluster_to_galactic_cartesian(
            jnp.array([[0.0, 0.0, 0.0]]), 30.0, 0.0, 1000.0
        )
        east = cluster_to_galactic_cartesian(
            jnp.array([[2.0, 0.0, 0.0]]), 30.0, 0.0, 1000.0
        )
        _, _, phi_base = cartesian_to_spherical(base)
        _, _, phi_east = cartesian_to_spherical(east)
        assert float(phi_east[0]) > float(phi_base[0])

    def test_roundtrip_to_lbd(self):
        """Recover (l, b, d) of the center via cartesian_to_spherical."""
        uvw = cluster_to_galactic_cartesian(
            jnp.array([[0.0, 0.0, 0.0]]), 47.0, 18.0, 1500.0
        )
        r, theta, phi = cartesian_to_spherical(uvw)
        d = float(r[0])
        b = 90.0 - float(jnp.rad2deg(theta[0]))
        ll = float(jnp.rad2deg(phi[0])) % 360.0
        assert abs(d - 1500.0) < 1e-6
        assert abs(b - 18.0) < 1e-6
        assert abs(ll - 47.0) < 1e-6

    def test_differentiable_wrt_placement(self):
        """Differentiable in distance and (l, b) — the placement leaves."""
        pos = jnp.array([[1.0, -2.0, 3.0]])

        def loss(args):
            l0, b0, d0 = args
            uvw = cluster_to_galactic_cartesian(pos, l0, b0, d0)
            return jnp.sum(uvw**2)

        g = jax.grad(loss)(jnp.array([30.0, 10.0, 1000.0]))
        assert jnp.all(jnp.isfinite(g))
        assert jnp.abs(g[2]) > 0.0  # distance gradient nonzero

    def test_jit_and_batch(self):
        """JIT-compilable over a batch of stars."""
        pos = jax.random.normal(jax.random.PRNGKey(0), (16, 3)) * 2.0

        @jax.jit
        def compute(p):
            return cluster_to_galactic_cartesian(p, 25.0, 5.0, 800.0)

        out = compute(pos)
        assert out.shape == (16, 3)
