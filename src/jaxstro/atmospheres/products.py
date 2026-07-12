"""Hashable atmosphere product and artifact contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductDescriptor:
    """One exact atmosphere product and its validated numerical policies."""

    product_id: str
    family: str
    parameter_names: tuple[str, ...]
    topology_policy: str
    flux_interpolation_policy: str | None
    provenance_id: str

    def __post_init__(self) -> None:
        required = (
            self.product_id,
            self.family,
            self.topology_policy,
            self.provenance_id,
        )
        if any(not value.strip() for value in required):
            raise ValueError("product descriptor fields must be non-empty")
        if not self.parameter_names or any(
            not name.strip() for name in self.parameter_names
        ):
            raise ValueError("product descriptor requires parameter names")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("product parameter names must be unique")


@dataclass(frozen=True)
class ArtifactReport:
    """Host-side validation evidence for one processed artifact."""

    valid: bool
    digest: str | None = None
    schema: str | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.valid and (not self.digest or not self.schema):
            raise ValueError("valid artifact reports require digest and schema")


__all__ = ["ArtifactReport", "ProductDescriptor"]
