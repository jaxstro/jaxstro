"""Executable contracts for the spatial theory chapter."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from jaxstro.spatial import gather_pairs_within_radius

REPO_ROOT = Path(__file__).resolve().parents[2]
SPATIAL_PAGE = REPO_ROOT / "docs" / "10-theory" / "spatial.md"
MYST_CONFIG = REPO_ROOT / "docs" / "myst.yml"


def _page_text() -> str:
    return SPATIAL_PAGE.read_text(encoding="utf-8")


def test_spatial_chapter_is_in_navigation_and_uses_public_api() -> None:
    text = _page_text()
    navigation = MYST_CONFIG.read_text(encoding="utf-8")

    assert "10-theory/spatial.md" in navigation
    for symbol in (
        "assign_particles_to_bins",
        "fill_bins",
        "fill_bins_exact",
        "approx_knn_candidates",
        "gather_pairs_within_radius",
    ):
        assert f"`{symbol}" in text


def test_spatial_chapter_states_candidate_exactness_and_overflow_contracts() -> None:
    text = _page_text()

    assert "Candidate does not mean neighbor" in text
    assert "`fill_bins` cannot certify full recall after capacity overflow" in text
    assert "exact only when `did_overflow` is false" in text
    assert "0 < |x_i - x_j| <= cutoff" in text
    assert "`cell_size >= cutoff`" in text
    assert "host-side, discrete preprocessing" in text
    assert "positive power of two" in text
    assert "{cite:t}`Morton1966`" in text


def test_worked_exact_radius_example_matches_documented_neighbors() -> None:
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.25, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    neighbors, mask, did_overflow = gather_pairs_within_radius(
        positions,
        origin=jnp.array([0.0, 0.0, 0.0]),
        cell_size=0.5,
        cutoff=0.5,
        k_max=5,
        Bcap=6,
        dims=(4, 2, 2),
    )

    focal_neighbors = set(map(int, neighbors[0][mask[0]].tolist()))
    assert focal_neighbors == {1, 2, 4}
    assert not bool(did_overflow)


def test_spatial_chapter_embeds_registered_accessible_figure() -> None:
    text = _page_text()

    assert "./figures/spatial-neighbor-contracts.webp" in text
    assert ":name: fig-spatial-neighbor-contracts" in text
    assert (
        ":alt: Two-panel spatial-neighbor diagram comparing a grid candidate "
        "pool with exact cutoff-filtered neighbors"
    ) in text
