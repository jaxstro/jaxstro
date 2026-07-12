"""Validation contracts for NewEra canonical preparation."""

from __future__ import annotations

from jaxstro.atmospheres import NewEraBackend


def test_newera_descriptor_freezes_source_and_interpolation_policy() -> None:
    descriptor = NewEraBackend.product_descriptor()

    assert descriptor.product_id == "newera-v3-lowres"
    assert descriptor.parameter_names == ("teff", "logg")
    assert descriptor.topology_policy == "complete-cell-or-approved-simplex"
    assert descriptor.flux_interpolation_policy == "linear"
    assert descriptor.provenance_id == "newera-v3-lowres"
