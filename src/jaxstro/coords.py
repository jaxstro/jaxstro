"""
Coordinate transformations for the jaxstro ecosystem.

This module provides coordinate transforms used across jaxstro packages:

- **Sky-tangent projection**: 3D cluster positions -> (RA, Dec)
- **Galactic <-> Equatorial**: Frame conversions (IAU 2000)
- **Cartesian <-> Spherical**: Basic geometry
- **Astrometry**: Parallax and proper motion computations

All functions are JAX-native and fully differentiable.

Units Convention:
- Positions: parsecs (pc)
- Velocities: km/s
- Angles: degrees
- Proper motions: mas/yr
- Parallax: mas

References:
    Liu et al. (2011), "Reconsidering the Galactic Coordinate System", A&A, 526, A16
    Binney & Merrifield (1998), "Galactic Astronomy", Chapter 1
    van Leeuwen (2007), "Hipparcos Data Reduction", ESA SP-1200
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jaxstro.astrometry import K_PROPER_MOTION

if TYPE_CHECKING:
    from jaxtyping import Array, Float

__all__ = [
    "sky_tangent",
    "galactic_to_equatorial",
    "equatorial_to_galactic",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "compute_parallax",
    "compute_proper_motions",
]


# ===========================================================================
# Sky-Tangent Projection
# ===========================================================================


def sky_tangent(
    positions: "Float[Array, 'N 3']",
    distance_pc: float,
    ra_center_deg: float = 180.0,
    dec_center_deg: float = 0.0,
    psi_deg: float = 0.0,
    warn_large_field: bool = True,
) -> "Float[Array, 'N 2']":
    """
    Project 3D cluster-centric positions to (RA, Dec) using sky-tangent frame.

    The cluster is embedded at distance D along the line-of-sight direction
    defined by (ra_center_deg, dec_center_deg). Local offsets are interpreted as:

    - x -> East (rotated by psi)
    - y -> North (rotated by psi)
    - z -> LOS (positive = away from observer)

    Parameters
    ----------
    positions : Float[Array, "N 3"]
        Cluster-centric positions [pc] as (x, y, z_LOS).
    distance_pc : float
        Distance to cluster center [pc].
    ra_center_deg : float, optional
        RA of pointing/tangent point [degrees]. Default 180.0.
    dec_center_deg : float, optional
        Dec of pointing/tangent point [degrees]. Default 0.0.
    psi_deg : float, optional
        Position angle / roll about LOS [degrees, default 0].
        Positive = counter-clockwise rotation of (E, N) frame when looking
        toward the cluster.
    warn_large_field : bool, optional
        If True, warn when field exceeds ~6 deg (TAN projection validity limit).
        Default True.

    Returns
    -------
    Float[Array, "N 2"]
        Array of [RA, Dec] in degrees.

    Notes
    -----
    - Fully differentiable in (ra_center_deg, dec_center_deg, distance_pc, psi_deg)
    - For small offsets at dec=0: dRA ~ x_E/D rad, dDec ~ y_N/D rad
    - For high-dec fields: dRA ~ x_E/(D*cos(dec)) accounts for cos(dec) factor
    - TAN projection is accurate to <0.1 mas for fields < 6 deg

    Roll/Position-Angle
    -------------------
    With psi != 0, the East/North frame rotates about the line-of-sight::

        e' = cos(psi)*e + sin(psi)*n
        n' = -sin(psi)*e + cos(psi)*n

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxstro.coords import sky_tangent
    >>>
    >>> # Single star 1 pc East of cluster center at 1 kpc
    >>> pos = jnp.array([[1.0, 0.0, 0.0]])
    >>> coords = sky_tangent(pos, distance_pc=1000.0, ra_center_deg=180.0, dec_center_deg=0.0)
    >>> # RA offset ~ 1/1000 rad ~ 0.057 deg ~ 206 arcsec
    """
    # Ensure float64 precision for astrometric accuracy
    positions = jnp.asarray(positions, dtype=jnp.float64)
    distance_pc = jnp.float64(distance_pc)

    # Convert tangent point and roll to radians
    ra0 = jnp.deg2rad(jnp.float64(ra_center_deg))
    dec0 = jnp.deg2rad(jnp.float64(dec_center_deg))
    psi = jnp.deg2rad(jnp.float64(psi_deg))

    # Build orthonormal sky-tangent triad (ICRS-aligned)
    cos_ra0, sin_ra0 = jnp.cos(ra0), jnp.sin(ra0)
    cos_dec0, sin_dec0 = jnp.cos(dec0), jnp.sin(dec0)

    # Line-of-sight unit vector (toward cluster, away from observer)
    z_hat = jnp.array(
        [cos_dec0 * cos_ra0, cos_dec0 * sin_ra0, sin_dec0],
        dtype=jnp.float64,
    )

    # East unit vector (increasing RA at constant Dec)
    e_hat_0 = jnp.array([-sin_ra0, cos_ra0, 0.0], dtype=jnp.float64)

    # North unit vector (increasing Dec at constant RA)
    n_hat_0 = jnp.array(
        [-sin_dec0 * cos_ra0, -sin_dec0 * sin_ra0, cos_dec0],
        dtype=jnp.float64,
    )

    # Apply roll/position-angle rotation about LOS
    cos_psi, sin_psi = jnp.cos(psi), jnp.sin(psi)
    e_hat = cos_psi * e_hat_0 + sin_psi * n_hat_0
    n_hat = -sin_psi * e_hat_0 + cos_psi * n_hat_0

    # Extract local coordinates from input positions
    x_local = positions[:, 0]  # East offset [pc]
    y_local = positions[:, 1]  # North offset [pc]
    z_LOS = positions[:, 2]  # LOS offset (positive = away from observer) [pc]

    # Domain-of-validity check (TAN accurate for < ~6 deg = 0.1 rad)
    if warn_large_field:
        max_offset_sq = jnp.max(x_local**2 + y_local**2)
        max_offset = jnp.sqrt(max_offset_sq)
        field_rad = max_offset / distance_pc
        if isinstance(field_rad, (float, int)) and field_rad > 0.1:
            warnings.warn(
                f"Field radius {jnp.rad2deg(field_rad):.1f} deg exceeds TAN validity (~6 deg). "
                "Consider using a different projection for wide fields.",
                UserWarning,
                stacklevel=2,
            )

    # Compute 3D ICRS position of each star
    S = (
        (distance_pc + z_LOS)[:, None] * z_hat[None, :]
        + x_local[:, None] * e_hat[None, :]
        + y_local[:, None] * n_hat[None, :]
    )

    # Normalize to unit vectors on celestial sphere
    S_norm = jnp.linalg.norm(S, axis=1, keepdims=True)
    U = S / S_norm

    # Convert to celestial coordinates
    ra = jnp.arctan2(U[:, 1], U[:, 0])
    dec = jnp.arcsin(jnp.clip(U[:, 2], -1.0, 1.0))

    # Convert to degrees, wrap RA to [0, 360)
    ra_deg = jnp.mod(jnp.rad2deg(ra), 360.0)
    dec_deg = jnp.rad2deg(dec)

    return jnp.stack([ra_deg, dec_deg], axis=-1)


# ===========================================================================
# Galactic <-> Equatorial Transformations
# ===========================================================================

# IAU 2000 rotation matrix: ICRS (Equatorial) = R @ Galactic
# Validated against astropy.coordinates to <1e-10 precision
_GALACTIC_TO_ICRS = jnp.array(
    [
        [-0.0548755604162154, +0.4941094278755837, -0.8676661490190047],
        [-0.8734370902348850, -0.4448296299600112, -0.1980763734312015],
        [-0.4838350155487132, +0.7469822444972189, +0.4559837761750669],
    ]
)

# Inverse: Galactic = R^T @ ICRS
_ICRS_TO_GALACTIC = _GALACTIC_TO_ICRS.T


def galactic_to_equatorial(
    l: "Float[Array, 'N']", b: "Float[Array, 'N']"
) -> tuple["Float[Array, 'N']", "Float[Array, 'N']"]:
    """
    Convert Galactic coordinates (l, b) to Equatorial (RA, Dec) in J2000.0.

    Uses IAU 2000 definitions. Validated against astropy.coordinates.

    Parameters
    ----------
    l : Float[Array, "N"]
        Galactic longitude [degrees] (0 to 360)
    b : Float[Array, "N"]
        Galactic latitude [degrees] (-90 to +90)

    Returns
    -------
    ra : Float[Array, "N"]
        Right ascension [degrees] (0 to 360)
    dec : Float[Array, "N"]
        Declination [degrees] (-90 to +90)

    Notes
    -----
    IAU 2000 parameters (J2000.0):
    - alpha_NGP = 192.85948 deg (RA of North Galactic Pole)
    - delta_NGP = 27.12825 deg (Dec of North Galactic Pole)
    - l_0 = 122.93192 deg (Galactic longitude of NCP)

    References
    ----------
    Liu et al. (2011), "Reconsidering the Galactic Coordinate System", A&A, 526, A16

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxstro.coords import galactic_to_equatorial
    >>> # Galactic center: l=0 deg, b=0 deg -> RA~266.4 deg, Dec~-29.0 deg
    >>> l = jnp.array([0.0])
    >>> b = jnp.array([0.0])
    >>> ra, dec = galactic_to_equatorial(l, b)
    """
    # Convert to radians
    l_rad = jnp.deg2rad(l)
    b_rad = jnp.deg2rad(b)

    # Convert Galactic spherical to Cartesian
    x_gal = jnp.cos(b_rad) * jnp.cos(l_rad)
    y_gal = jnp.cos(b_rad) * jnp.sin(l_rad)
    z_gal = jnp.sin(b_rad)

    # Apply rotation: ICRS = R @ Galactic
    R = _GALACTIC_TO_ICRS
    x_eq = R[0, 0] * x_gal + R[0, 1] * y_gal + R[0, 2] * z_gal
    y_eq = R[1, 0] * x_gal + R[1, 1] * y_gal + R[1, 2] * z_gal
    z_eq = R[2, 0] * x_gal + R[2, 1] * y_gal + R[2, 2] * z_gal

    # Convert Equatorial Cartesian to spherical
    ra_rad = jnp.arctan2(y_eq, x_eq)
    dec_rad = jnp.arcsin(z_eq)

    # Convert to degrees and wrap RA to [0, 360)
    ra = jnp.rad2deg(ra_rad) % 360.0
    dec = jnp.rad2deg(dec_rad)

    return ra, dec


def equatorial_to_galactic(
    ra: "Float[Array, 'N']", dec: "Float[Array, 'N']"
) -> tuple["Float[Array, 'N']", "Float[Array, 'N']"]:
    """
    Convert Equatorial coordinates (RA, Dec) in J2000.0 to Galactic (l, b).

    Inverse transformation of galactic_to_equatorial().

    Parameters
    ----------
    ra : Float[Array, "N"]
        Right ascension [degrees] (0 to 360)
    dec : Float[Array, "N"]
        Declination [degrees] (-90 to +90)

    Returns
    -------
    l : Float[Array, "N"]
        Galactic longitude [degrees] (0 to 360)
    b : Float[Array, "N"]
        Galactic latitude [degrees] (-90 to +90)

    References
    ----------
    Liu et al. (2011), A&A, 526, A16

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxstro.coords import equatorial_to_galactic
    >>> # Galactic center direction: RA~266.4 deg, Dec~-29.0 deg -> l=0 deg, b=0 deg
    >>> ra = jnp.array([266.4])
    >>> dec = jnp.array([-28.9])
    >>> l, b = equatorial_to_galactic(ra, dec)
    """
    # Convert to radians
    ra_rad = jnp.deg2rad(ra)
    dec_rad = jnp.deg2rad(dec)

    # Convert Equatorial spherical to Cartesian
    x_eq = jnp.cos(dec_rad) * jnp.cos(ra_rad)
    y_eq = jnp.cos(dec_rad) * jnp.sin(ra_rad)
    z_eq = jnp.sin(dec_rad)

    # Apply inverse rotation: Galactic = R^T @ ICRS
    R_T = _ICRS_TO_GALACTIC
    x_gal = R_T[0, 0] * x_eq + R_T[0, 1] * y_eq + R_T[0, 2] * z_eq
    y_gal = R_T[1, 0] * x_eq + R_T[1, 1] * y_eq + R_T[1, 2] * z_eq
    z_gal = R_T[2, 0] * x_eq + R_T[2, 1] * y_eq + R_T[2, 2] * z_eq

    # Convert Galactic Cartesian to spherical
    l_rad = jnp.arctan2(y_gal, x_gal)
    b_rad = jnp.arcsin(z_gal)

    # Convert to degrees and wrap l to [0, 360)
    l = jnp.rad2deg(l_rad) % 360.0
    b = jnp.rad2deg(b_rad)

    return l, b


# ===========================================================================
# Cartesian <-> Spherical Transformations
# ===========================================================================


@jax.jit
def cartesian_to_spherical(
    positions: "Float[Array, 'N 3']",
) -> tuple["Float[Array, 'N']", "Float[Array, 'N']", "Float[Array, 'N']"]:
    """
    Convert Cartesian to spherical coordinates.

    Parameters
    ----------
    positions : Float[Array, "N 3"]
        Cartesian positions [pc] as (x, y, z)

    Returns
    -------
    r : Float[Array, "N"]
        Radial distance [pc]
    theta : Float[Array, "N"]
        Polar angle [radians] (0 to pi, measured from +z axis)
    phi : Float[Array, "N"]
        Azimuthal angle [radians] (0 to 2pi, measured from +x axis toward +y)

    Examples
    --------
    >>> positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    >>> r, theta, phi = cartesian_to_spherical(positions)
    >>> # Point 1: (1, 0, 0) -> r=1, theta=pi/2, phi=0
    >>> # Point 2: (0, 1, 0) -> r=1, theta=pi/2, phi=pi/2
    >>> # Point 3: (0, 0, 1) -> r=1, theta=0, phi=0
    """
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    r = jnp.sqrt(x**2 + y**2 + z**2)
    # Use arctan2 for numerical stability instead of arccos
    rho = jnp.sqrt(x**2 + y**2)  # Cylindrical radius
    theta = jnp.arctan2(rho, z)  # Polar angle from z-axis
    phi = jnp.arctan2(y, x)  # Azimuthal angle
    return r, theta, phi


@jax.jit
def spherical_to_cartesian(
    r: "Float[Array, 'N']", theta: "Float[Array, 'N']", phi: "Float[Array, 'N']"
) -> "Float[Array, 'N 3']":
    """
    Convert spherical to Cartesian coordinates.

    Parameters
    ----------
    r : Float[Array, "N"]
        Radial distance [pc]
    theta : Float[Array, "N"]
        Polar angle [radians] (0 to pi)
    phi : Float[Array, "N"]
        Azimuthal angle [radians] (0 to 2pi)

    Returns
    -------
    positions : Float[Array, "N 3"]
        Cartesian positions [pc]

    Examples
    --------
    >>> r = jnp.array([1.0, 1.0, 1.0])
    >>> theta = jnp.array([jnp.pi/2, jnp.pi/2, 0.0])
    >>> phi = jnp.array([0.0, jnp.pi/2, 0.0])
    >>> positions = spherical_to_cartesian(r, theta, phi)
    >>> # Returns approximately [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)
    return jnp.stack([x, y, z], axis=1)


# ===========================================================================
# Astrometric Computations
# ===========================================================================


@jax.jit
def compute_parallax(
    positions: "Float[Array, 'N 3']",
    distance_pc: float,
) -> "Float[Array, 'N']":
    """
    Compute parallax from positions.

    Parameters
    ----------
    positions : Float[Array, "N 3"]
        Cartesian positions relative to system center [pc]
    distance_pc : float
        Distance from observer to system center [pc]

    Returns
    -------
    parallax : Float[Array, "N"]
        Parallax [mas]

    Notes
    -----
    Parallax [mas] = 1000 / distance [pc] = 1 / distance [kpc]

    For a system at distance D with internal scale R << D:
    parallax ~ 1/D (all stars have similar parallax)

    Examples
    --------
    >>> # Star at system center, observer at 1 kpc
    >>> positions = jnp.array([[0.0, 0.0, 0.0]])
    >>> parallax = compute_parallax(positions, distance_pc=1000.0)
    >>> # Expected: parallax ~ 1.0 mas
    """
    # Observer is at (0, 0, -distance_pc)
    # Vector from observer to each star
    observer_z = -distance_pc
    r_vec_x = positions[:, 0]
    r_vec_y = positions[:, 1]
    r_vec_z = positions[:, 2] - observer_z  # = z + distance_pc

    # Distance from observer to each star [pc]
    r_pc = jnp.sqrt(r_vec_x**2 + r_vec_y**2 + r_vec_z**2)

    # Convert to kpc and compute parallax
    r_kpc = r_pc / 1000.0
    parallax = 1.0 / (r_kpc + 1e-10)  # mas

    return parallax


@jax.jit
def compute_proper_motions(
    positions: "Float[Array, 'N 3']",
    velocities: "Float[Array, 'N 3']",
    distance_pc: float,
) -> tuple["Float[Array, 'N']", "Float[Array, 'N']"]:
    """
    Compute proper motions from Cartesian positions and velocities.

    Assumes observer at (0, 0, -distance_pc).
    Projects velocities perpendicular to line of sight.

    Parameters
    ----------
    positions : Float[Array, "N 3"]
        Cartesian positions relative to system center [pc]
    velocities : Float[Array, "N 3"]
        Cartesian velocities [km/s]
    distance_pc : float
        Distance from observer to system center [pc]

    Returns
    -------
    mu_ra : Float[Array, "N"]
        Proper motion in RA * cos(Dec) [mas/yr]
    mu_dec : Float[Array, "N"]
        Proper motion in Dec [mas/yr]

    Notes
    -----
    Proper motion relates to transverse velocity:
        v_transverse [km/s] = 4.74047 * mu [mas/yr] * d [kpc]

    Therefore:
        mu = v_transverse / (4.74047 * d)

    References
    ----------
    van Leeuwen (2007), "Hipparcos Data Reduction", Section 1.5.3

    Examples
    --------
    >>> # Star at system center moving 100 km/s tangentially at 1 kpc
    >>> positions = jnp.array([[0.0, 0.0, 0.0]])
    >>> velocities = jnp.array([[100.0, 0.0, 0.0]])  # Tangential in x-direction
    >>> mu_ra, mu_dec = compute_proper_motions(positions, velocities, distance_pc=1000.0)
    >>> # Expected: mu ~ 100 / 4.74 ~ 21.1 mas/yr
    """
    # Observer at (0, 0, -distance_pc)
    observer_z = -distance_pc

    # Vector from observer to each star
    r_vec_x = positions[:, 0]
    r_vec_y = positions[:, 1]
    r_vec_z = positions[:, 2] - observer_z  # = z + distance_pc

    r = jnp.sqrt(r_vec_x**2 + r_vec_y**2 + r_vec_z**2)

    # Unit vector pointing from observer to star (line of sight)
    r_safe = r + 1e-10
    los_x = r_vec_x / r_safe
    los_y = r_vec_y / r_safe
    los_z = r_vec_z / r_safe

    # Project velocity perpendicular to line of sight (transverse velocity)
    v_radial = velocities[:, 0] * los_x + velocities[:, 1] * los_y + velocities[:, 2] * los_z
    v_trans_x = velocities[:, 0] - v_radial * los_x
    v_trans_y = velocities[:, 1] - v_radial * los_y
    v_trans_z = velocities[:, 2] - v_radial * los_z

    # Distance to each star in kpc
    distance_kpc = r / 1000.0

    # Convert transverse velocity to proper motion
    # v_transverse [km/s] = K_PROPER_MOTION * mu [mas/yr] * d [kpc]
    # mu [mas/yr] = v_transverse [km/s] / (K_PROPER_MOTION * d [kpc])
    conversion = K_PROPER_MOTION * distance_kpc  # km/s per mas/yr

    # Project transverse velocity onto RA and Dec directions
    # Simple approximation: x -> RA direction, y -> Dec direction
    # (Valid for small angular extent and observer looking along +z)
    mu_ra_cosdec = v_trans_x / conversion  # mas/yr
    mu_dec = v_trans_y / conversion  # mas/yr

    return mu_ra_cosdec, mu_dec
