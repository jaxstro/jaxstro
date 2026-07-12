"""Validation contracts for distinct TLUSTY products."""

from __future__ import annotations

from jaxstro.atmospheres.tlusty import TlustyBackend


def test_tlusty_descriptors_freeze_product_specific_policy() -> None:
    assert len(TlustyBackend.product_specs()) == 27
    for product_id in (spec.product_id for spec in TlustyBackend.product_specs()):
        descriptor = TlustyBackend.product_descriptor(product_id)
        assert descriptor.parameter_names == ("teff", "logg")
        expected = "linear" if product_id.startswith("tlusty-ostar2002:") else None
        assert descriptor.flux_interpolation_policy == expected
        assert descriptor.provenance_id in {"tlusty-ostar2002", "tlusty-bstar2006"}
