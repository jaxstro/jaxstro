"""Tests for host-side atmosphere topology selection."""

from __future__ import annotations

from jaxstro.atmospheres import GridTopology, TopologyKind, select_topology
from jaxstro.spectra import SpectrumStatusCode


def test_complete_rectilinear_cell_is_selected_before_simplex() -> None:
    topology = GridTopology(
        parameter_names=("teff", "logg"),
        points=((5000.0, 4.0), (6000.0, 4.0), (5000.0, 5.0), (6000.0, 5.0)),
        approved_simplices=((0, 1, 2),),
    )

    selected = select_topology(topology, (5500.0, 4.5))

    assert selected.kind is TopologyKind.RECTILINEAR
    assert selected.vertex_indices == (0, 2, 1, 3)
    assert selected.status is SpectrumStatusCode.OK


def test_approved_simplex_is_selected_deterministically_for_sparse_cell() -> None:
    topology = GridTopology(
        parameter_names=("teff", "logg"),
        points=((5000.0, 4.0), (6000.0, 4.0), (5000.0, 5.0)),
        approved_simplices=((1, 2, 0),),
    )

    selected = select_topology(topology, (5250.0, 4.25))

    assert selected.kind is TopologyKind.SIMPLEX
    assert selected.vertex_indices == (1, 2, 0)
    assert selected.status is SpectrumStatusCode.OK


def test_sparse_cell_without_approved_simplex_fails_closed() -> None:
    topology = GridTopology(
        parameter_names=("teff", "logg"),
        points=((5000.0, 4.0), (6000.0, 4.0), (5000.0, 5.0)),
    )

    selected = select_topology(topology, (5250.0, 4.25))

    assert selected.kind is TopologyKind.NONE
    assert selected.vertex_indices == ()
    assert selected.status is SpectrumStatusCode.NO_COMPLETE_CELL


def test_query_outside_convex_bounds_returns_structured_status() -> None:
    topology = GridTopology(
        parameter_names=("teff", "logg"),
        points=((5000.0, 4.0), (6000.0, 4.0), (5000.0, 5.0)),
        approved_simplices=((0, 1, 2),),
    )

    selected = select_topology(topology, (7000.0, 4.5))

    assert selected.kind is TopologyKind.NONE
    assert selected.status is SpectrumStatusCode.OUTSIDE_CONVEX_HULL


def test_topology_rejects_wrong_rank_and_invalid_manifest() -> None:
    import pytest

    with pytest.raises(ValueError, match="same rank"):
        GridTopology(parameter_names=("teff", "logg"), points=((5000.0,),))
    with pytest.raises(ValueError, match="simplex indices"):
        GridTopology(
            parameter_names=("teff", "logg"),
            points=((5000.0, 4.0), (6000.0, 4.0), (5000.0, 5.0)),
            approved_simplices=((0, 1, 9),),
        )
