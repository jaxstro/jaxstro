"""Validation contracts for Sonora Diamondback preparation."""

from __future__ import annotations

from jaxstro.atmospheres.sonora import SonoraBackend


def test_sonora_descriptor_separates_cloud_metallicity_and_c_o() -> None:
    descriptor = SonoraBackend.product_descriptor("f1", 0.0, 1.0)

    assert descriptor.product_id == "sonora-diamondback-2024:f1:m+0:co1"
    assert descriptor.parameter_names == ("teff", "logg")
    assert descriptor.flux_interpolation_policy is None
