"""Tests for generic vector geometry helpers."""

import jax
import jax.numpy as jnp

from jaxstro import geometry


class TestVectorGeometry:
    """Tests for vector normalization and angular distance."""

    def test_normalize_returns_unit_vectors_and_norms(self):
        vectors = jnp.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
        unit, norm = geometry.normalize(vectors, axis=-1, return_norm=True)
        assert jnp.allclose(norm, jnp.array([[5.0], [2.0]]))
        assert jnp.allclose(jnp.linalg.norm(unit, axis=-1), jnp.ones(2))

    def test_angular_distance_between_axes(self):
        x = jnp.array([1.0, 0.0, 0.0])
        y = jnp.array([0.0, 1.0, 0.0])
        assert jnp.allclose(geometry.angular_distance(x, y), 0.5 * jnp.pi)


class TestRotations:
    """Tests for rotation matrices and quaternions."""

    def test_axis_angle_rotation_matrix_rotates_about_z(self):
        matrix = geometry.rotation_matrix(jnp.array([0.0, 0.0, 1.0]), 0.5 * jnp.pi)
        vector = jnp.array([1.0, 0.0, 0.0])
        assert jnp.allclose(matrix @ vector, jnp.array([0.0, 1.0, 0.0]), atol=1e-12)
        assert jnp.allclose(matrix.T @ matrix, jnp.eye(3), atol=1e-12)

    def test_quaternion_rotation_matches_rotation_matrix(self):
        axis = jnp.array([0.0, 0.0, 1.0])
        angle = 0.25 * jnp.pi
        vector = jnp.array([1.0, 2.0, 0.0])
        quat = geometry.quaternion_from_axis_angle(axis, angle)
        matrix = geometry.rotation_matrix(axis, angle)
        assert jnp.allclose(
            geometry.quaternion_rotate(quat, vector),
            matrix @ vector,
            atol=1e-12,
        )

    def test_quaternion_inverse_round_trip(self):
        quat = geometry.quaternion_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), 0.7)
        vector = jnp.array([0.0, 2.0, -1.0])
        rotated = geometry.quaternion_rotate(quat, vector)
        restored = geometry.quaternion_rotate(
            geometry.quaternion_conjugate(quat), rotated
        )
        assert jnp.allclose(restored, vector, atol=1e-12)


class TestRigidTransforms:
    """Tests for rigid transform helpers."""

    def test_rigid_transform_inverse_round_trip(self):
        rotation = geometry.rotation_matrix(jnp.array([0.0, 0.0, 1.0]), 0.3)
        translation = jnp.array([1.0, -2.0, 0.5])
        points = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0]])
        transformed = geometry.rigid_transform(points, rotation, translation)
        inv_rotation, inv_translation = geometry.invert_rigid(rotation, translation)
        restored = geometry.rigid_transform(transformed, inv_rotation, inv_translation)
        assert jnp.allclose(restored, points, atol=1e-12)

    def test_compose_rigid_matches_sequential_application(self):
        r1 = geometry.rotation_matrix(jnp.array([0.0, 0.0, 1.0]), 0.2)
        t1 = jnp.array([1.0, 0.0, 0.0])
        r2 = geometry.rotation_matrix(jnp.array([0.0, 1.0, 0.0]), -0.4)
        t2 = jnp.array([0.0, 2.0, 0.0])
        point = jnp.array([0.5, -1.0, 2.0])
        composed_r, composed_t = geometry.compose_rigid(r1, t1, r2, t2)
        sequential = geometry.rigid_transform(
            geometry.rigid_transform(point, r2, t2), r1, t1
        )
        composed = geometry.rigid_transform(point, composed_r, composed_t)
        assert jnp.allclose(composed, sequential, atol=1e-12)


class TestGeometryTransforms:
    """Tests for JAX transform compatibility."""

    def test_jit_vmap_and_grad(self):
        axis = jnp.array([0.0, 0.0, 1.0])
        vectors = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        def rotate_sum(angle, vector):
            matrix = geometry.rotation_matrix(axis, angle)
            return jnp.sum(matrix @ vector)

        values = jax.jit(jax.vmap(lambda vector: rotate_sum(0.2, vector)))(vectors)
        grad = jax.grad(lambda angle: rotate_sum(angle, vectors[0]))(jnp.array(0.2))
        assert values.shape == (2,)
        assert jnp.isfinite(grad)
