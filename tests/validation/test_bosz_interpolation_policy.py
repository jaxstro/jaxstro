"""Validation contracts for explicit BOSZ products."""

from __future__ import annotations

from jaxstro.atmospheres import BoszBackend


def test_bosz_descriptor_encodes_atmosphere_resolution_and_product() -> None:
    descriptor = BoszBackend.product_descriptor("ap", "r10000", "resam")

    assert descriptor.product_id == "bosz-2025-recomputed:ap:r10000:resam"
    assert descriptor.parameter_names == ("teff", "logg")
    assert descriptor.flux_interpolation_policy == "linear"
