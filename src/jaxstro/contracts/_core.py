"""Contracts for public single-file modules and a local manifest constructor."""

from .schema import ExecutionBoundary, MaturityLevel, ModuleContract


def module_contract(
    name: str,
    ownership: str,
    non_ownership: str,
    *,
    boundary: ExecutionBoundary = ExecutionBoundary.RUNTIME,
    dimensions: str = "Caller-owned units and dimensions.",
    maturity: MaturityLevel = MaturityLevel.VALIDATED,
) -> ModuleContract:
    """Construct a module record while keeping ownership text explicit."""
    return ModuleContract(
        id=name,
        import_path=f"jaxstro.{name}",
        ownership=ownership,
        non_ownership=non_ownership,
        intended_uses=("Shared differentiable scientific infrastructure",),
        execution_boundary=boundary,
        dimensional_policy=dimensions,
        maturity=maturity,
    )


CORE_CONTRACTS = (
    module_contract(
        "astrometry",
        "Astrometric constants and transforms.",
        "Survey or population models.",
    ),
    module_contract(
        "constants",
        "Source-backed physical constants.",
        "Runtime source lookup.",
        boundary=ExecutionBoundary.STATIC,
    ),
    module_contract("coords", "Coordinate transformations.", "Domain frame selection."),
    module_contract(
        "geometry", "Generic geometric transformations.", "Domain geometry policy."
    ),
    module_contract(
        "jaxconfig",
        "Explicit JAX precision configuration.",
        "Import-time global configuration.",
        boundary=ExecutionBoundary.STATIC,
    ),
    module_contract(
        "provenance",
        "Runtime artifact manifests.",
        "Scientific-source validation.",
        boundary=ExecutionBoundary.TOOLING,
    ),
    module_contract(
        "units",
        "Canonical ecosystem unit systems.",
        "Hidden domain unit defaults.",
        boundary=ExecutionBoundary.STATIC,
    ),
)
