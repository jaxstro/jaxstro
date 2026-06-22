"""Tests for structured 1D mesh helpers."""

import jax
import jax.numpy as jnp

from jaxstro.numerics import meshes


class TestStructuredMesh1D:
    """Tests for mesh construction and geometry."""

    def test_structured_edges_and_mesh_geometry(self):
        edges = meshes.structured_edges_1d(0.0, 2.0, n_cells=4)
        mesh = meshes.Mesh1D(edges)
        assert jnp.allclose(edges, jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]))
        assert jnp.allclose(mesh.centers, jnp.array([0.25, 0.75, 1.25, 1.75]))
        assert jnp.allclose(mesh.widths, jnp.full((4,), 0.5))
        assert jnp.allclose(mesh.volumes, mesh.widths)

    def test_face_geometry_has_positions_and_unit_areas(self):
        mesh = meshes.Mesh1D(jnp.array([0.0, 1.0, 3.0]))
        faces = meshes.face_geometry_1d(mesh.edges)
        assert jnp.allclose(faces.positions, mesh.edges)
        assert jnp.allclose(faces.areas, jnp.ones(3))

    def test_cell_neighbors_use_minus_one_boundary_sentinel(self):
        neighbors = meshes.cell_neighbors_1d(4)
        assert jnp.array_equal(neighbors.left, jnp.array([-1, 0, 1, 2]))
        assert jnp.array_equal(neighbors.right, jnp.array([1, 2, 3, -1]))


class TestFiniteVolumeHelpers:
    """Tests for finite-volume stencil helpers."""

    def test_divergence_of_constant_flux_is_zero(self):
        edges = jnp.array([0.0, 0.5, 1.0, 2.0])
        face_flux = jnp.ones(4) * 3.0
        assert jnp.allclose(meshes.divergence_1d(face_flux, edges), jnp.zeros(3))

    def test_divergence_of_linear_flux_is_one(self):
        edges = jnp.linspace(0.0, 1.0, 6)
        face_flux = edges
        assert jnp.allclose(meshes.divergence_1d(face_flux, edges), jnp.ones(5))

    def test_cell_to_face_average_copies_boundaries_and_averages_interior(self):
        values = jnp.array([1.0, 3.0, 7.0])
        faces = meshes.cell_to_face_average(values)
        assert jnp.allclose(faces, jnp.array([1.0, 2.0, 5.0, 7.0]))


class TestConservativeRemap:
    """Tests for conservative remapping of cell averages."""

    def test_conservative_remap_preserves_integral(self):
        old_edges = jnp.array([0.0, 1.0, 3.0])
        new_edges = jnp.array([0.0, 0.5, 2.0, 3.0])
        averages = jnp.array([2.0, 4.0])
        remapped = meshes.conservative_remap_1d(old_edges, averages, new_edges)
        old_total = jnp.sum(averages * jnp.diff(old_edges))
        new_total = jnp.sum(remapped * jnp.diff(new_edges))
        assert jnp.allclose(new_total, old_total)

    def test_conservative_remap_preserves_constant_average(self):
        old_edges = jnp.array([0.0, 1.0, 2.0])
        new_edges = jnp.array([0.0, 0.25, 1.25, 2.0])
        remapped = meshes.conservative_remap_1d(
            old_edges,
            jnp.full((2,), 3.5),
            new_edges,
        )
        assert jnp.allclose(remapped, jnp.full((3,), 3.5))

    def test_mesh_helpers_are_jit_and_grad_compatible(self):
        old_edges = jnp.array([0.0, 1.0, 2.0])
        new_edges = jnp.array([0.0, 0.5, 1.5, 2.0])

        def total_after_remap(values):
            remapped = meshes.conservative_remap_1d(old_edges, values, new_edges)
            return jnp.sum(remapped * jnp.diff(new_edges))

        values = jnp.array([1.0, 2.0])
        assert jnp.isfinite(jax.jit(total_after_remap)(values))
        assert jnp.all(jnp.isfinite(jax.grad(total_after_remap)(values)))
