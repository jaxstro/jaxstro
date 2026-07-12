"""Validation contracts for distinct TLUSTY products."""

from __future__ import annotations

from jaxstro.atmospheres.tlusty import TlustyBackend


def test_tlusty_descriptors_freeze_product_specific_policy() -> None:
    for product_id in (
        "tlusty-ostar2002",
        "tlusty-bstar2006-vturb2",
        "tlusty-bstar2006-vturb10-cn",
    ):
        descriptor = TlustyBackend.product_descriptor(product_id)
        assert descriptor.parameter_names == ("teff", "logg")
        assert descriptor.flux_interpolation_policy == "linear"
        assert descriptor.provenance_id in {"tlusty-ostar2002", "tlusty-bstar2006"}
