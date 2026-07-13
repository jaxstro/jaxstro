"""Contracts for public single-file modules and a local manifest constructor."""

from .schema import ExecutionBoundary, MaturityLevel, ModuleContract


def module_contract(
    name: str,
    ownership: str,
    non_ownership: str,
    intended_use: str,
    dimensions: str,
    *,
    boundary: ExecutionBoundary = ExecutionBoundary.RUNTIME,
    maturity: MaturityLevel = MaturityLevel.VALIDATED,
) -> ModuleContract:
    """Construct a module record while keeping ownership text explicit."""
    return ModuleContract(
        id=name,
        import_path=f"jaxstro.{name}",
        ownership=ownership,
        non_ownership=non_ownership,
        intended_uses=(intended_use,),
        execution_boundary=boundary,
        dimensional_policy=dimensions,
        maturity=maturity,
    )


CORE_CONTRACTS = (
    module_contract(
        "astrometry",
        "Astrometric constants.",
        "Survey or population models.",
        "Converting proper-motion and angular units.",
        "Angles in radians/degrees and proper motion in mas/yr as named.",
    ),
    module_contract(
        "constants",
        "Source-backed physical constants.",
        "Runtime source lookup.",
        "Shared physical constants and nominal conversions.",
        "CGS unless the symbol explicitly names another unit.",
        boundary=ExecutionBoundary.STATIC,
    ),
    module_contract(
        "coords",
        "Coordinate transformations.",
        "Domain frame selection.",
        "Astrometric coordinate conversion.",
        "Positions in pc, velocities in km/s, angles in degrees, proper motions in mas/yr, and parallax in mas where documented.",
    ),
    module_contract(
        "geometry",
        "Generic geometric transformations.",
        "Domain geometry policy.",
        "Vector, rotation, and rigid-transform operations.",
        "Caller-owned coordinate units; angles follow each function contract.",
    ),
    module_contract(
        "jaxconfig",
        "Explicit JAX precision configuration.",
        "Import-time global configuration.",
        "Enabling float64 and highest matmul precision.",
        "No physical dimensions.",
        boundary=ExecutionBoundary.STATIC,
    ),
    module_contract(
        "provenance",
        "Runtime artifact manifests.",
        "Scientific-source validation.",
        "Recording deterministic computational provenance.",
        "Metric units remain explicit in producer-owned payloads.",
        boundary=ExecutionBoundary.TOOLING,
    ),
    module_contract(
        "units",
        "Canonical ecosystem unit systems.",
        "Hidden domain unit defaults.",
        "Explicit conversion among named unit systems.",
        "CGS is canonical; named systems declare mass, length, and time scales.",
        boundary=ExecutionBoundary.STATIC,
    ),
    module_contract(
        "contracts",
        "Scientific contract vocabulary, validation, and rendering.",
        "Runtime scientific acceptance or automatic certification.",
        "Auditing public ownership, transforms, AD semantics, and evidence.",
        "Metadata only; scientific units are recorded by owned contracts.",
        boundary=ExecutionBoundary.TOOLING,
    ),
    module_contract(
        "evidence",
        "Portable computational-evidence schemas, validation, and rendering.",
        "Method-specific scientific thresholds or source-provenance claims.",
        "Recording and auditing numerical metrics and comparisons.",
        "Every metric carries explicit producer-owned units.",
        boundary=ExecutionBoundary.TOOLING,
    ),
)
