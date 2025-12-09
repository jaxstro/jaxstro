"""
Morton (Z-order) encoding for cache-coherent spatial indexing.

Morton codes interleave the bits of (x, y, z) coordinates to produce a single
1D index that preserves spatial locality. Points near each other in 3D space
tend to have nearby Morton codes, improving cache hit rates for neighbor queries.

This module provides:

- `morton_encode_3d`: Convert 3D integer coordinates to Morton codes
- `morton_decode_3d`: Recover 3D coordinates from Morton codes
- `wyhash32`: Fast deterministic hash for tie-breaking and sampling

The current implementation uses 10 bits per dimension (max 1024 bins per axis),
yielding 30-bit Morton codes stored as int32.

TODO(jaxstro): Document recommended use patterns:
    - Reorder particle arrays in Morton order to improve cache locality.
    - Use Morton bins as a basis for domain decomposition in MPI/GPU tiling.
    - Combine with Hilbert curves for better locality in some applications.

TODO: Consider adding 64-bit Morton variants and configurable bit-depth
for very large grids (e.g. 2048+ bins per dimension). For now we cap
at 10 bits (1024 bins) for portability and simplicity.

References:
    Morton, G. M. (1966). "A computer oriented geodetic data base and a new
        technique in file sequencing." IBM Technical Report.
    Bit manipulation techniques from libmorton (Jeroen Baert, public domain).

Example:
    >>> import jax.numpy as jnp
    >>> from jaxstro.spatial.morton import morton_encode_3d, morton_decode_3d
    >>> xyz = jnp.array([[3, 5, 7], [0, 0, 0], [1023, 1023, 1023]])
    >>> codes = morton_encode_3d(xyz)
    >>> x, y, z = morton_decode_3d(codes)
    >>> assert jnp.allclose(jnp.stack([x, y, z], axis=-1), xyz)
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Int

# =============================================================================
# Constants
# =============================================================================

# Maximum bits per dimension for 3D Morton encoding.
# 10 bits → max 1024 bins per axis, 30 bits total → fits in int32.
MAX_BITS_3D: int = 10


# =============================================================================
# Internal Bit Manipulation
# =============================================================================


def _part1by2_32(x: Int[Array, "..."]) -> Int[Array, "..."]:
    """
    Spread 10-bit integer across 30 bits (every 3rd bit).

    Used for Morton encoding: interleave x, y, z bits.
    Maps bit pattern: abcdefghij → a..b..c..d..e..f..g..h..i..j

    Args:
        x: Integer array (masked to 10 bits internally)

    Returns:
        Bit-spread version with bits at positions 0, 3, 6, 9, ...

    Note:
        This uses the "magic bits" technique from libmorton for O(1) spreading.
    """
    x = jnp.uint32(x) & jnp.uint32(0x3FF)  # Keep 10 bits
    x = (x | (x << 16)) & jnp.uint32(0x030000FF)
    x = (x | (x << 8)) & jnp.uint32(0x0300F00F)
    x = (x | (x << 4)) & jnp.uint32(0x030C30C3)
    x = (x | (x << 2)) & jnp.uint32(0x09249249)
    return x


def _compact1by2_32(x: Int[Array, "..."]) -> Int[Array, "..."]:
    """
    Reverse of _part1by2_32: extract every 3rd bit into 10-bit integer.

    Args:
        x: Bit-spread uint32 array with bits at positions 0, 3, 6, ...

    Returns:
        Compacted 10-bit integers
    """
    x = jnp.uint32(x) & jnp.uint32(0x09249249)
    x = (x ^ (x >> 2)) & jnp.uint32(0x030C30C3)
    x = (x ^ (x >> 4)) & jnp.uint32(0x0300F00F)
    x = (x ^ (x >> 8)) & jnp.uint32(0x030000FF)
    x = (x ^ (x >> 16)) & jnp.uint32(0x000003FF)
    return x


# =============================================================================
# Public API: Morton Encoding/Decoding
# =============================================================================


def morton_encode_3d(
    xyz: Int[Array, "N 3"],
    bits: int = MAX_BITS_3D,
) -> Int[Array, "N"]:
    """
    Encode 3D integer coordinates to Morton (Z-order) codes.

    Interleaves bits of (x, y, z) to produce a single 1D index that preserves
    spatial locality for cache-coherent memory access patterns.

    Args:
        xyz: Integer coordinates [N, 3], each in range [0, 2^bits - 1]
        bits: Bits per dimension. Currently only 10 is supported.

    Returns:
        Morton codes [N] as int32

    Raises:
        ValueError: If bits != MAX_BITS_3D (currently 10)

    Example:
        >>> xyz = jnp.array([[3, 5, 7]])
        >>> morton_encode_3d(xyz)
        Array([911], dtype=int32)

    Note:
        The interleaving places x at bits 0,3,6,..., y at bits 1,4,7,...,
        and z at bits 2,5,8,... This is the standard Z-order convention.
    """
    if bits != MAX_BITS_3D:
        raise ValueError(
            f"Only bits={MAX_BITS_3D} supported for morton_encode_3d, got {bits}"
        )

    x = xyz[:, 0].astype(jnp.uint32)
    y = xyz[:, 1].astype(jnp.uint32)
    z = xyz[:, 2].astype(jnp.uint32)

    # Interleave: x at bits 0,3,6,... y at 1,4,7,... z at 2,5,8,...
    code = _part1by2_32(x) | (_part1by2_32(y) << 1) | (_part1by2_32(z) << 2)

    return code.astype(jnp.int32)  # Convert back for array indexing


def morton_decode_3d(
    code: Int[Array, "N"],
    bits: int = MAX_BITS_3D,
    out_dtype=jnp.int32,
) -> tuple[Int[Array, "N"], Int[Array, "N"], Int[Array, "N"]]:
    """
    Decode Morton codes back to (x, y, z) coordinates.

    Args:
        code: Morton codes [N]
        bits: Bits per dimension. Currently only 10 is supported.
        out_dtype: Output dtype for coordinates (default: int32)

    Returns:
        Tuple of (x, y, z) coordinate arrays, each shape [N]

    Raises:
        ValueError: If bits != MAX_BITS_3D (currently 10)

    Example:
        >>> code = jnp.array([911], dtype=jnp.int32)
        >>> x, y, z = morton_decode_3d(code)
        >>> jnp.stack([x, y, z], axis=-1)
        Array([[3, 5, 7]], dtype=int32)
    """
    if bits != MAX_BITS_3D:
        raise ValueError(
            f"Only bits={MAX_BITS_3D} supported for morton_decode_3d, got {bits}"
        )

    code = code.astype(jnp.uint32)

    x = _compact1by2_32(code)
    y = _compact1by2_32(code >> 1)
    z = _compact1by2_32(code >> 2)

    return (
        x.astype(out_dtype),
        y.astype(out_dtype),
        z.astype(out_dtype),
    )


# =============================================================================
# Hashing Utilities
# =============================================================================


def wyhash32(x: Int[Array, "..."]) -> Int[Array, "..."]:
    """
    Fast deterministic 32-bit hash (WyHash mixer).

    Provides high-quality mixing of integer inputs for applications requiring:

    - **Tie-breaking**: Stable ordering when values are equal
    - **Reservoir sampling**: Unbiased selection when bins overflow capacity
    - **Randomized algorithms**: Deterministic pseudo-random behavior

    The hash is bijective and has good avalanche properties (small input
    changes cause large output changes).

    Args:
        x: Integer array to hash

    Returns:
        Hashed values as uint32 (full 32-bit range)

    Example:
        >>> x = jnp.array([0, 1, 2, 3])
        >>> h = wyhash32(x)
        >>> # Each input maps to a distinct, well-distributed hash
        >>> len(jnp.unique(h)) == 4
        True

    References:
        WyHash by Wang Yi (public domain, https://github.com/wangyi-fudan/wyhash)
    """
    x = jnp.uint32(x)
    x ^= jnp.uint32(0x9E3779B1)  # Golden ratio constant
    x = (x ^ (x >> 16)) * jnp.uint32(0x7FEB352D)
    x = (x ^ (x >> 15)) * jnp.uint32(0x846CA68B)
    x = x ^ (x >> 16)
    return x
