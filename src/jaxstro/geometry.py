"""Generic vector geometry and rigid-transform helpers."""

import jax.numpy as jnp
from jaxtyping import Array, Float


def _unit_vector(
    vector: Float[Array, "..."],
    *,
    axis: int = -1,
    eps: float = 0.0,
) -> Float[Array, "..."]:
    vector = jnp.asarray(vector)
    norm = jnp.linalg.norm(vector, axis=axis, keepdims=True)
    return vector / jnp.maximum(norm, eps)


def normalize(
    vectors: Float[Array, "..."],
    *,
    axis: int = -1,
    eps: float = 0.0,
    return_norm: bool = False,
) -> Float[Array, "..."] | tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Normalize vectors along ``axis`` and optionally return their norms."""
    vectors = jnp.asarray(vectors)
    norm = jnp.linalg.norm(vectors, axis=axis, keepdims=True)
    unit = vectors / jnp.maximum(norm, eps)
    if return_norm:
        return unit, norm
    return unit


def angular_distance(
    a: Float[Array, "..."],
    b: Float[Array, "..."],
    *,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Return the angle between vectors in radians."""
    a_unit = _unit_vector(a, axis=axis)
    b_unit = _unit_vector(b, axis=axis)
    dot = jnp.sum(a_unit * b_unit, axis=axis)
    return jnp.arccos(jnp.clip(dot, -1.0, 1.0))


def rotation_matrix(
    axis: Float[Array, "3"],
    angle: Float[Array, ""],
) -> Float[Array, "3 3"]:
    """Return the right-handed axis-angle rotation matrix."""
    unit = _unit_vector(axis)
    x, y, z = unit
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    one_c = 1.0 - c
    return jnp.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def quaternion_from_axis_angle(
    axis: Float[Array, "3"],
    angle: Float[Array, ""],
) -> Float[Array, "4"]:
    """Return a unit quaternion ``[w, x, y, z]`` from axis-angle inputs."""
    unit = _unit_vector(axis)
    half = 0.5 * angle
    return jnp.concatenate([jnp.cos(half)[None], jnp.sin(half) * unit])


def quaternion_conjugate(q: Float[Array, "4"]) -> Float[Array, "4"]:
    """Return the conjugate of quaternion ``[w, x, y, z]``."""
    q = jnp.asarray(q)
    return jnp.concatenate([q[:1], -q[1:]])


def quaternion_multiply(
    left: Float[Array, "4"],
    right: Float[Array, "4"],
) -> Float[Array, "4"]:
    """Hamilton product for quaternions stored as ``[w, x, y, z]``."""
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return jnp.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quaternion_rotate(
    q: Float[Array, "4"],
    vector: Float[Array, "3"],
) -> Float[Array, "3"]:
    """Rotate a 3-vector by quaternion ``q``."""
    q_unit = _unit_vector(q)
    pure = jnp.concatenate([jnp.zeros(1, dtype=jnp.asarray(vector).dtype), vector])
    rotated = quaternion_multiply(
        quaternion_multiply(q_unit, pure),
        quaternion_conjugate(q_unit),
    )
    return rotated[1:]


def rigid_transform(
    points: Float[Array, "... 3"],
    rotation: Float[Array, "3 3"],
    translation: Float[Array, "3"],
) -> Float[Array, "... 3"]:
    """Apply ``rotation @ point + translation`` to one or more 3D points."""
    return jnp.einsum("ij,...j->...i", rotation, points) + translation


def invert_rigid(
    rotation: Float[Array, "3 3"],
    translation: Float[Array, "3"],
) -> tuple[Float[Array, "3 3"], Float[Array, "3"]]:
    """Return the inverse rigid transform."""
    inv_rotation = rotation.T
    inv_translation = -(inv_rotation @ translation)
    return inv_rotation, inv_translation


def compose_rigid(
    outer_rotation: Float[Array, "3 3"],
    outer_translation: Float[Array, "3"],
    inner_rotation: Float[Array, "3 3"],
    inner_translation: Float[Array, "3"],
) -> tuple[Float[Array, "3 3"], Float[Array, "3"]]:
    """Compose ``outer(inner(point))`` for rigid transforms."""
    rotation = outer_rotation @ inner_rotation
    translation = outer_rotation @ inner_translation + outer_translation
    return rotation, translation


__all__ = [
    "normalize",
    "angular_distance",
    "rotation_matrix",
    "quaternion_from_axis_angle",
    "quaternion_conjugate",
    "quaternion_multiply",
    "quaternion_rotate",
    "rigid_transform",
    "invert_rigid",
    "compose_rigid",
]
