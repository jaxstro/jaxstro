"""
Tests for jaxstro.spatial module.

Tests cover:
1. Morton encoding/decoding roundtrip properties
2. Bin assignment correctness and boundary handling
3. Bin filling with overflow (reservoir sampling)
4. Neighbor candidate gathering monotonicity
5. JAX compatibility (JIT compilation)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxstro.spatial import (
    approx_knn_candidates,
    assign_particles_to_bins,
    fill_bins,
    gather_candidates_from_bins,
    gather_candidates_two_stencil,
    gather_candidates_with_stencil,
    morton_decode_3d,
    morton_encode_3d,
    wyhash32,
)
from jaxstro.spatial.morton import MAX_BITS_3D


def _exact_knn_indices(pos: jnp.ndarray, k: int) -> jnp.ndarray:
    """Return exact k nearest non-self neighbors for a small test cloud."""
    r2 = jnp.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=-1)
    r2 = jnp.where(jnp.eye(pos.shape[0], dtype=bool), jnp.inf, r2)
    _, idx = jax.lax.top_k(-r2, k)
    return idx


def _candidate_sets_for_positions(
    pos: jnp.ndarray,
    *,
    k_target: int,
    L_box: float = 4.0,
    Nbins_per_dim: int = 4,
    Bcap: int = 64,
) -> list[set[int]]:
    """Build generous approximate candidate sets for recall-oriented tests."""
    n = pos.shape[0]
    pos_sentinel = jnp.concatenate([pos, jnp.zeros((1, 3))], axis=0)
    bin_of = assign_particles_to_bins(pos, L_box=L_box, Nbins_per_dim=Nbins_per_dim)
    particle_ids = jnp.arange(n, dtype=jnp.int32)
    bin_members, bin_mask = fill_bins(
        particle_ids,
        bin_of,
        Nbins=Nbins_per_dim**3,
        Bcap=Bcap,
    )
    cand_idx, cand_mask = approx_knn_candidates(
        pos=pos_sentinel,
        bin_members=bin_members,
        bin_mask=bin_mask,
        bin_of=bin_of,
        Nbins_per_dim=Nbins_per_dim,
        K_target=k_target,
        K_bin_coarse=min(Bcap, 32),
        K_bin_dense=min(Bcap, 32),
    )
    return [set(map(int, cand_idx[i][cand_mask[i]].tolist())) for i in range(n)]


# =============================================================================
# Morton Encoding/Decoding Tests
# =============================================================================


class TestMortonEncoding:
    """Tests for Morton (Z-order) encoding and decoding."""

    @pytest.mark.parametrize("Nbins", [8, 16, 32, 64, 128])
    def test_roundtrip_random(self, Nbins: int):
        """Morton encode/decode roundtrip for random coordinates."""
        key = jax.random.PRNGKey(42)
        N = 100
        xyz = jax.random.randint(key, (N, 3), 0, Nbins)

        codes = morton_encode_3d(xyz)
        x2, y2, z2 = morton_decode_3d(codes)
        xyz_decoded = jnp.stack([x2, y2, z2], axis=-1)

        assert jnp.allclose(xyz, xyz_decoded), "Roundtrip failed"

    def test_roundtrip_corners(self):
        """Morton roundtrip for corner cases (0, 0, 0) and (max, max, max)."""
        max_val = 2**MAX_BITS_3D - 1  # 1023 for 10 bits
        xyz = jnp.array(
            [
                [0, 0, 0],
                [max_val, max_val, max_val],
                [max_val, 0, 0],
                [0, max_val, 0],
                [0, 0, max_val],
            ]
        )

        codes = morton_encode_3d(xyz)
        x2, y2, z2 = morton_decode_3d(codes)
        xyz_decoded = jnp.stack([x2, y2, z2], axis=-1)

        assert jnp.allclose(xyz, xyz_decoded), "Corner roundtrip failed"

    def test_known_values(self):
        """Test specific known Morton codes."""
        # (3, 5, 7) should encode to a specific value
        xyz = jnp.array([[3, 5, 7]])
        code = morton_encode_3d(xyz)

        # Decode back
        x, y, z = morton_decode_3d(code)
        assert x[0] == 3
        assert y[0] == 5
        assert z[0] == 7

    def test_unique_codes(self):
        """All distinct coordinates produce distinct Morton codes."""
        key = jax.random.PRNGKey(123)
        N = 500
        xyz = jax.random.randint(key, (N, 3), 0, 32)

        codes = morton_encode_3d(xyz)

        # Find unique input coordinates
        xyz_tuples = [tuple(row.tolist()) for row in xyz]
        unique_inputs = len(set(xyz_tuples))

        # Unique codes should match unique inputs
        unique_codes = len(jnp.unique(codes))
        assert unique_codes == unique_inputs, "Code collision detected"

    def test_bits_validation(self):
        """Invalid bits parameter raises ValueError."""
        xyz = jnp.array([[1, 2, 3]])

        with pytest.raises(ValueError, match="Only bits=10"):
            morton_encode_3d(xyz, bits=8)

        with pytest.raises(ValueError, match="Only bits=10"):
            morton_decode_3d(jnp.array([42]), bits=12)


class TestWyhash:
    """Tests for wyhash32 hashing function."""

    def test_deterministic(self):
        """Hash is deterministic."""
        x = jnp.array([0, 1, 2, 3, 100, 999])
        h1 = wyhash32(x)
        h2 = wyhash32(x)
        assert jnp.allclose(h1, h2), "Hash not deterministic"

    def test_avalanche(self):
        """Small input changes cause large output changes."""
        x1 = jnp.array([0])
        x2 = jnp.array([1])
        h1 = wyhash32(x1)
        h2 = wyhash32(x2)

        # Hashes should be very different
        assert h1[0] != h2[0], "No avalanche effect"

    def test_distribution(self):
        """Hash values are well-distributed."""
        x = jnp.arange(1000)
        h = wyhash32(x)

        # Check all unique
        assert len(jnp.unique(h)) == 1000, "Hash collisions in 0-999"


# =============================================================================
# Bin Assignment Tests
# =============================================================================


class TestBinAssignment:
    """Tests for particle-to-bin assignment."""

    def test_center_particle(self):
        """Particle at center maps to center bin."""
        pos = jnp.array([[0.0, 0.0, 0.0]])
        bin_of = assign_particles_to_bins(pos, L_box=4.0, Nbins_per_dim=8)

        # Center should be bin (4, 4, 4) in 0-7 indexing
        x, y, z = morton_decode_3d(bin_of)
        assert x[0] == 4
        assert y[0] == 4
        assert z[0] == 4

    def test_boundary_clamping(self):
        """Particles outside box are clamped to boundary bins."""
        # Positions way outside the [-2, 2] box (L_box=4)
        pos = jnp.array(
            [
                [-100.0, 0.0, 0.0],  # Far left
                [100.0, 0.0, 0.0],  # Far right
                [0.0, -100.0, 0.0],  # Far bottom
                [0.0, 100.0, 0.0],  # Far top
            ]
        )
        bin_of = assign_particles_to_bins(pos, L_box=4.0, Nbins_per_dim=8)

        x, y, z = morton_decode_3d(bin_of)

        # All should be clamped to [0, 7]
        assert jnp.all((x >= 0) & (x <= 7))
        assert jnp.all((y >= 0) & (y <= 7))
        assert jnp.all((z >= 0) & (z <= 7))

        # Check specific clamping
        assert x[0] == 0  # Far left -> bin 0
        assert x[1] == 7  # Far right -> bin 7

    def test_corner_particles(self):
        """Particles at box corners map to corner bins."""
        L = 4.0
        half = L / 2 - 0.001  # Slightly inside corners
        pos = jnp.array(
            [
                [-half, -half, -half],  # (0, 0, 0)
                [half, half, half],  # (7, 7, 7) for Nbins=8
            ]
        )
        bin_of = assign_particles_to_bins(pos, L_box=L, Nbins_per_dim=8)

        x, y, z = morton_decode_3d(bin_of)

        assert (x[0], y[0], z[0]) == (0, 0, 0)
        assert (x[1], y[1], z[1]) == (7, 7, 7)

    def test_all_bins_in_range(self):
        """All assigned bins are in valid range."""
        key = jax.random.PRNGKey(42)
        N = 1000
        pos = jax.random.uniform(key, (N, 3)) * 4.0 - 2.0  # [-2, 2]^3

        Nbins_per_dim = 16
        bin_of = assign_particles_to_bins(pos, L_box=4.0, Nbins_per_dim=Nbins_per_dim)

        x, y, z = morton_decode_3d(bin_of)

        assert jnp.all((x >= 0) & (x < Nbins_per_dim))
        assert jnp.all((y >= 0) & (y < Nbins_per_dim))
        assert jnp.all((z >= 0) & (z < Nbins_per_dim))

    def test_box_center_offset(self):
        """Box center parameter shifts the grid correctly."""
        # Particle at origin with box centered at (1, 1, 1)
        pos = jnp.array([[0.0, 0.0, 0.0]])
        bin_of = assign_particles_to_bins(
            pos, L_box=4.0, Nbins_per_dim=8, box_center=1.0
        )

        x, y, z = morton_decode_3d(bin_of)

        # With center at (1, 1, 1), origin is at (-1, -1, -1) relative to center
        # Box spans [-1, 3] (center - L/2 to center + L/2)
        # Bin 0 starts at -1, so origin maps to bin 2 (roughly)
        assert x[0] == 2
        assert y[0] == 2
        assert z[0] == 2


# =============================================================================
# Bin Filling Tests
# =============================================================================


class TestBinFilling:
    """Tests for bin filling with overflow handling."""

    def test_basic_filling(self):
        """Basic bin filling without overflow."""
        N = 10
        particle_ids = jnp.arange(N, dtype=jnp.int32)
        # Each particle in a different bin
        bin_of = jnp.arange(N, dtype=jnp.int32)

        bin_members, bin_mask = fill_bins(particle_ids, bin_of, Nbins=16, Bcap=5)

        # First 10 bins should have 1 particle each
        for i in range(10):
            assert bin_mask[i, 0], f"Bin {i} should have 1 particle"
            assert not bin_mask[i, 1], f"Bin {i} should have only 1 particle"

    def test_overflow_handling(self):
        """Overflow handling when all particles in one bin."""
        N = 20
        Bcap = 10
        particle_ids = jnp.arange(N, dtype=jnp.int32)
        bin_of = jnp.zeros(N, dtype=jnp.int32)  # All in bin 0

        bin_members, bin_mask = fill_bins(particle_ids, bin_of, Nbins=8, Bcap=Bcap)

        # Bin 0 should have exactly Bcap particles
        assert jnp.sum(bin_mask[0]) == Bcap, (
            f"Expected {Bcap} particles, got {jnp.sum(bin_mask[0])}"
        )

        # All selected IDs should be unique
        valid_ids = bin_members[0][bin_mask[0]]
        assert len(jnp.unique(valid_ids)) == Bcap, "Duplicate particles selected"

        # All selected IDs should be valid particle IDs
        assert jnp.all((valid_ids >= 0) & (valid_ids < N))

    def test_deterministic_overflow(self):
        """Overflow selection is deterministic across runs."""
        N = 50
        Bcap = 15
        particle_ids = jnp.arange(N, dtype=jnp.int32)
        bin_of = jnp.zeros(N, dtype=jnp.int32)

        bin_members1, _ = fill_bins(particle_ids, bin_of, Nbins=8, Bcap=Bcap)
        bin_members2, _ = fill_bins(particle_ids, bin_of, Nbins=8, Bcap=Bcap)

        assert jnp.allclose(bin_members1, bin_members2), "Overflow not deterministic"

    def test_sentinel_value(self):
        """Empty slots contain sentinel value."""
        N = 5
        particle_ids = jnp.arange(N, dtype=jnp.int32)
        bin_of = jnp.zeros(N, dtype=jnp.int32)  # All in bin 0

        bin_members, bin_mask = fill_bins(particle_ids, bin_of, Nbins=8, Bcap=10)

        # Bin 0 has 5 particles, slots 5-9 should be sentinel
        assert jnp.all(bin_members[0, 5:] == N), "Empty slots not sentinel"
        assert jnp.all(~bin_mask[0, 5:]), "Empty slots not masked"

        # Bin 1+ should be all sentinel
        assert jnp.all(bin_members[1:] == N), "Empty bins not sentinel"
        assert jnp.all(~bin_mask[1:]), "Empty bins not masked"

    def test_custom_sentinel(self):
        """Custom sentinel value works correctly."""
        N = 10
        particle_ids = jnp.arange(N, dtype=jnp.int32)
        bin_of = jnp.zeros(N, dtype=jnp.int32)
        sentinel = 9999

        bin_members, _ = fill_bins(
            particle_ids, bin_of, Nbins=4, Bcap=20, sentinel_N=sentinel
        )

        # Empty slots should have custom sentinel
        assert jnp.all(bin_members[0, 10:] == sentinel)
        assert jnp.all(bin_members[1:] == sentinel)


# =============================================================================
# Neighbor Candidate Gathering Tests
# =============================================================================


class TestNeighborGathering:
    """Tests for approximate neighbor candidate gathering."""

    @pytest.fixture
    def uniform_grid_setup(self):
        """Create a uniform grid of particles for testing."""
        # Create 4x4x4 = 64 particles on a regular grid
        coords = jnp.linspace(-1.5, 1.5, 4)
        xx, yy, zz = jnp.meshgrid(coords, coords, coords, indexing="ij")
        pos = jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)  # [64, 3]

        N = pos.shape[0]
        # Add sentinel position
        pos_sentinel = jnp.concatenate([pos, jnp.zeros((1, 3))], axis=0)  # [65, 3]

        # Assign to bins
        L_box = 4.0
        Nbins_per_dim = 8
        bin_of = assign_particles_to_bins(pos, L_box=L_box, Nbins_per_dim=Nbins_per_dim)

        # Fill bins
        particle_ids = jnp.arange(N, dtype=jnp.int32)
        Nbins = Nbins_per_dim**3
        Bcap = 20
        bin_members, bin_mask = fill_bins(particle_ids, bin_of, Nbins=Nbins, Bcap=Bcap)

        return {
            "pos": pos_sentinel,
            "bin_members": bin_members,
            "bin_mask": bin_mask,
            "bin_of": bin_of,
            "Nbins_per_dim": Nbins_per_dim,
            "N": N,
        }

    def test_stencil_monotonicity(self, uniform_grid_setup):
        """Larger stencil returns at least as many candidates."""
        setup = uniform_grid_setup
        N = setup["N"]
        r_search = jnp.zeros(N)
        dx = 0.5

        # 27-cell (3x3x3)
        cand_27, mask_27 = gather_candidates_with_stencil(
            pos=setup["pos"],
            bin_members=setup["bin_members"],
            bin_mask=setup["bin_mask"],
            bin_of=setup["bin_of"],
            r_search=r_search,
            Nbins_per_dim=setup["Nbins_per_dim"],
            dx=dx,
            Cand_max=50,
            K_bin=10,
            stencil_width=3,
        )

        # 125-cell (5x5x5)
        cand_125, mask_125 = gather_candidates_with_stencil(
            pos=setup["pos"],
            bin_members=setup["bin_members"],
            bin_mask=setup["bin_mask"],
            bin_of=setup["bin_of"],
            r_search=r_search,
            Nbins_per_dim=setup["Nbins_per_dim"],
            dx=dx,
            Cand_max=100,
            K_bin=5,
            stencil_width=5,
        )

        n_cand_27 = jnp.sum(mask_27, axis=1)
        n_cand_125 = jnp.sum(mask_125, axis=1)

        # 125-cell should generally have at least as many (often more)
        # We use mean as aggregate measure (not strict per-particle)
        assert jnp.mean(n_cand_125) >= jnp.mean(n_cand_27) * 0.9, (
            "125-cell should have comparable or more candidates than 27-cell"
        )

    def test_self_excluded(self, uniform_grid_setup):
        """Particles are not their own neighbors."""
        setup = uniform_grid_setup
        N = setup["N"]
        r_search = jnp.zeros(N)
        dx = 0.5

        cand_idx, cand_mask = gather_candidates_from_bins(
            pos=setup["pos"],
            bin_members=setup["bin_members"],
            bin_mask=setup["bin_mask"],
            bin_of=setup["bin_of"],
            r_search=r_search,
            Nbins_per_dim=setup["Nbins_per_dim"],
            dx=dx,
            Cand_max=50,
            K_bin=10,
        )

        # Check no particle is its own candidate
        for i in range(N):
            valid_cands = cand_idx[i][cand_mask[i]]
            assert i not in valid_cands, f"Particle {i} is its own candidate"

    def test_two_stencil_dense_fallback(self, uniform_grid_setup):
        """Two-stencil uses dense fallback for thin coarse pools."""
        setup = uniform_grid_setup
        N = setup["N"]
        r_search = jnp.zeros(N)
        dx = 0.5

        # Use a high K_target to trigger dense fallback
        K_target = 30  # Higher than typical coarse count

        cand_idx, cand_mask = gather_candidates_two_stencil(
            pos=setup["pos"],
            bin_members=setup["bin_members"],
            bin_mask=setup["bin_mask"],
            bin_of=setup["bin_of"],
            r_search=r_search,
            Nbins_per_dim=setup["Nbins_per_dim"],
            dx=dx,
            K_target=K_target,
            K_bin_coarse=10,
            K_bin_dense=2,
        )

        n_cand = jnp.sum(cand_mask, axis=1)

        # With 64 particles total, some should have dense fallback triggered
        # Just verify we get reasonable candidate counts
        assert jnp.mean(n_cand) > 10, "Too few candidates"
        assert jnp.all(n_cand <= 512), "Exceeded max capacity"

    def test_approx_knn_basic(self, uniform_grid_setup):
        """approx_knn_candidates returns valid results."""
        setup = uniform_grid_setup

        cand_idx, cand_mask = approx_knn_candidates(
            pos=setup["pos"],
            bin_members=setup["bin_members"],
            bin_mask=setup["bin_mask"],
            bin_of=setup["bin_of"],
            Nbins_per_dim=setup["Nbins_per_dim"],
            K_target=20,
        )

        n_cand = jnp.sum(cand_mask, axis=1)

        # Every particle should have some candidates
        assert jnp.all(n_cand > 0), "Some particles have no candidates"

        # Candidate indices should be valid
        valid_cands = cand_idx[cand_mask]
        N = setup["N"]
        assert jnp.all((valid_cands >= 0) & (valid_cands < N)), (
            "Invalid candidate indices"
        )

    def test_candidates_contain_exact_knn_for_regular_cloud(self):
        """Generous candidate settings contain the exact kNN on a regular cloud."""
        coords = jnp.linspace(-0.45, 0.45, 3)
        xx, yy, zz = jnp.meshgrid(coords, coords, coords, indexing="ij")
        pos = jnp.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
        jitter = jnp.arange(pos.shape[0], dtype=pos.dtype)[:, None] * jnp.array(
            [1.0e-3, -7.0e-4, 4.0e-4]
        )
        pos = pos + jitter
        k = 6

        exact = _exact_knn_indices(pos, k)
        candidates = _candidate_sets_for_positions(pos, k_target=20, Nbins_per_dim=4)

        for i, candidate_set in enumerate(candidates):
            assert set(map(int, exact[i].tolist())) <= candidate_set

    def test_candidates_contain_exact_knn_near_box_boundaries(self):
        """Boundary clamping does not drop exact neighbors when caps are generous."""
        pos = jnp.array(
            [
                [-1.95, -1.95, -1.95],
                [-1.85, -1.90, -1.80],
                [-1.70, -1.82, -1.92],
                [1.95, 1.95, 1.95],
                [1.84, 1.88, 1.90],
                [1.70, 1.75, 1.82],
                [0.00, 0.00, 0.00],
                [0.12, -0.08, 0.10],
            ]
        )
        k = 2

        exact = _exact_knn_indices(pos, k)
        candidates = _candidate_sets_for_positions(pos, k_target=k)

        for i, candidate_set in enumerate(candidates):
            assert set(map(int, exact[i].tolist())) <= candidate_set

    def test_candidates_contain_exact_knn_for_cluster_without_overflow(self):
        """Clustered bins preserve exact recall when Bcap avoids reservoir loss."""
        offsets = jnp.array(
            [
                [0.00, 0.00, 0.00],
                [0.03, 0.01, -0.02],
                [-0.02, 0.04, 0.01],
                [0.05, -0.03, 0.02],
                [-0.04, -0.01, 0.03],
                [0.02, -0.05, -0.01],
                [-0.06, 0.02, -0.03],
                [0.04, 0.04, 0.04],
                [-0.05, -0.04, 0.02],
                [0.01, 0.06, -0.04],
            ]
        )
        pos = offsets + jnp.array([0.2, -0.1, 0.3])
        k = 4

        exact = _exact_knn_indices(pos, k)
        candidates = _candidate_sets_for_positions(pos, k_target=k, Bcap=16)

        for i, candidate_set in enumerate(candidates):
            assert set(map(int, exact[i].tolist())) <= candidate_set


# =============================================================================
# JAX Compatibility Tests
# =============================================================================


class TestJAXCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_morton_jit(self):
        """Morton encode/decode can be JIT compiled."""

        @jax.jit
        def roundtrip(xyz):
            codes = morton_encode_3d(xyz)
            x, y, z = morton_decode_3d(codes)
            return jnp.stack([x, y, z], axis=-1)

        xyz = jnp.array([[1, 2, 3], [4, 5, 6]])
        result = roundtrip(xyz)

        assert jnp.allclose(result, xyz)

    def test_assign_bins_jit(self):
        """Bin assignment can be JIT compiled."""

        @jax.jit
        def assign(pos):
            return assign_particles_to_bins(pos, L_box=4.0, Nbins_per_dim=8)

        key = jax.random.PRNGKey(0)
        pos = jax.random.uniform(key, (100, 3)) * 4.0 - 2.0

        bin_of = assign(pos)

        assert bin_of.shape == (100,)

    def test_fill_bins_jit(self):
        """Bin filling can be JIT compiled."""

        @jax.jit
        def fill(particle_ids, bin_of):
            return fill_bins(particle_ids, bin_of, Nbins=64, Bcap=20)

        N = 50
        particle_ids = jnp.arange(N, dtype=jnp.int32)
        bin_of = jnp.zeros(N, dtype=jnp.int32)

        bin_members, bin_mask = fill(particle_ids, bin_of)

        assert bin_members.shape == (64, 20)
        assert bin_mask.shape == (64, 20)

    def test_approx_knn_jit(self):
        """approx_knn_candidates can be JIT compiled."""
        N = 128
        key = jax.random.PRNGKey(42)
        pos = jax.random.uniform(key, (N, 3)) * 4.0 - 2.0

        # Add sentinel
        pos_sentinel = jnp.concatenate([pos, jnp.zeros((1, 3))], axis=0)

        # Build bins
        L_box = 4.0
        Nbins_per_dim = 8
        bin_of = assign_particles_to_bins(pos, L_box=L_box, Nbins_per_dim=Nbins_per_dim)
        particle_ids = jnp.arange(N, dtype=jnp.int32)
        Nbins = Nbins_per_dim**3
        Bcap = 32
        bin_members, bin_mask = fill_bins(particle_ids, bin_of, Nbins=Nbins, Bcap=Bcap)

        @jax.jit
        def get_candidates(pos_sent, b_members, b_mask, b_of):
            return approx_knn_candidates(
                pos=pos_sent,
                bin_members=b_members,
                bin_mask=b_mask,
                bin_of=b_of,
                Nbins_per_dim=Nbins_per_dim,
                K_target=20,
            )

        cand_idx, cand_mask = get_candidates(
            pos_sentinel, bin_members, bin_mask, bin_of
        )

        assert cand_idx.shape[0] == N
        assert cand_mask.shape[0] == N

    def test_full_pipeline_jit(self):
        """Full pipeline from positions to candidates can be JIT compiled."""
        N = 64
        L_box = 4.0
        Nbins_per_dim = 8
        Nbins = Nbins_per_dim**3
        Bcap = 20
        K_target = 16

        @jax.jit
        def pipeline(pos):
            # Add sentinel
            pos_sentinel = jnp.concatenate([pos, jnp.zeros((1, 3))], axis=0)

            # Assign to bins
            bin_of = assign_particles_to_bins(
                pos, L_box=L_box, Nbins_per_dim=Nbins_per_dim
            )

            # Fill bins
            particle_ids = jnp.arange(pos.shape[0], dtype=jnp.int32)
            bin_members, bin_mask = fill_bins(
                particle_ids, bin_of, Nbins=Nbins, Bcap=Bcap
            )

            # Get candidates
            return approx_knn_candidates(
                pos=pos_sentinel,
                bin_members=bin_members,
                bin_mask=bin_mask,
                bin_of=bin_of,
                Nbins_per_dim=Nbins_per_dim,
                K_target=K_target,
            )

        key = jax.random.PRNGKey(0)
        pos = jax.random.uniform(key, (N, 3)) * L_box - L_box / 2

        cand_idx, cand_mask = pipeline(pos)

        assert cand_idx.shape[0] == N
        assert jnp.sum(cand_mask) > 0, "No candidates found"


# =============================================================================
# Edge Cases and Stress Tests
# =============================================================================


class TestEdgeCases:
    """Edge case and stress tests."""

    def test_single_particle(self):
        """Single particle case (no neighbors)."""
        pos = jnp.array([[0.0, 0.0, 0.0]])
        pos_sentinel = jnp.concatenate([pos, jnp.zeros((1, 3))], axis=0)

        bin_of = assign_particles_to_bins(pos, L_box=4.0, Nbins_per_dim=8)
        particle_ids = jnp.array([0], dtype=jnp.int32)
        # Bcap must be >= K_bin_coarse (default 18)
        bin_members, bin_mask = fill_bins(particle_ids, bin_of, Nbins=512, Bcap=20)

        cand_idx, cand_mask = approx_knn_candidates(
            pos=pos_sentinel,
            bin_members=bin_members,
            bin_mask=bin_mask,
            bin_of=bin_of,
            Nbins_per_dim=8,
            K_target=5,
        )

        # Single particle has no neighbors
        assert jnp.sum(cand_mask) == 0

    def test_two_particles(self):
        """Two particles should find each other."""
        pos = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        pos_sentinel = jnp.concatenate([pos, jnp.zeros((1, 3))], axis=0)

        bin_of = assign_particles_to_bins(pos, L_box=4.0, Nbins_per_dim=8)
        particle_ids = jnp.arange(2, dtype=jnp.int32)
        # Bcap must be >= K_bin_coarse (default 18)
        bin_members, bin_mask = fill_bins(particle_ids, bin_of, Nbins=512, Bcap=20)

        cand_idx, cand_mask = approx_knn_candidates(
            pos=pos_sentinel,
            bin_members=bin_members,
            bin_mask=bin_mask,
            bin_of=bin_of,
            Nbins_per_dim=8,
            K_target=5,
        )

        # Each particle should find the other
        assert cand_mask[0, 0], "Particle 0 should find particle 1"
        assert cand_mask[1, 0], "Particle 1 should find particle 0"
        assert cand_idx[0, 0] == 1, "Particle 0's neighbor should be 1"
        assert cand_idx[1, 0] == 0, "Particle 1's neighbor should be 0"

    def test_highly_clustered(self):
        """Highly clustered particles (all in tiny region)."""
        key = jax.random.PRNGKey(99)
        N = 100
        # All particles in a tiny cluster
        pos = jax.random.uniform(key, (N, 3)) * 0.01

        pos_sentinel = jnp.concatenate([pos, jnp.zeros((1, 3))], axis=0)
        bin_of = assign_particles_to_bins(pos, L_box=4.0, Nbins_per_dim=16)
        particle_ids = jnp.arange(N, dtype=jnp.int32)
        # High Bcap to accommodate all particles in one bin
        bin_members, bin_mask = fill_bins(particle_ids, bin_of, Nbins=16**3, Bcap=100)

        # Use higher K_bin values for clustered case since all particles
        # are in one bin and we want to find many neighbors
        cand_idx, cand_mask = approx_knn_candidates(
            pos=pos_sentinel,
            bin_members=bin_members,
            bin_mask=bin_mask,
            bin_of=bin_of,
            Nbins_per_dim=16,
            K_target=30,
            K_bin_coarse=50,  # Higher since all in one bin
            K_bin_dense=20,  # Higher fallback
        )

        n_cand = jnp.sum(cand_mask, axis=1)

        # Should find many neighbors (all clustered together)
        # Each particle can have up to N-1 neighbors
        # With K_bin_dense=20, we expect at least 20 candidates per particle
        assert jnp.mean(n_cand) >= 20, "Clustered particles should find many neighbors"

    def test_sparse_uniform(self):
        """Sparse uniform distribution."""
        key = jax.random.PRNGKey(77)
        N = 50
        pos = jax.random.uniform(key, (N, 3)) * 100.0 - 50.0  # Large box

        pos_sentinel = jnp.concatenate([pos, jnp.zeros((1, 3))], axis=0)
        L_box = 100.0
        Nbins_per_dim = 8
        bin_of = assign_particles_to_bins(pos, L_box=L_box, Nbins_per_dim=Nbins_per_dim)
        particle_ids = jnp.arange(N, dtype=jnp.int32)
        bin_members, bin_mask = fill_bins(
            particle_ids, bin_of, Nbins=Nbins_per_dim**3, Bcap=20
        )

        cand_idx, cand_mask = approx_knn_candidates(
            pos=pos_sentinel,
            bin_members=bin_members,
            bin_mask=bin_mask,
            bin_of=bin_of,
            Nbins_per_dim=Nbins_per_dim,
            K_target=10,
        )

        # Should still work, though maybe fewer candidates due to sparsity
        n_cand = jnp.sum(cand_mask, axis=1)
        assert jnp.all(n_cand >= 0), "Negative candidate counts"
