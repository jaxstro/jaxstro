from jaxstro.contracts import ExecutionBoundary
from jaxstro.contracts._core import module_contract

MODULE_CONTRACT = module_contract(
    "atmospheres",
    "Catalog and artifact preparation plus evidence-gated evaluation.",
    "Photometry or model validity.",
    "Preparing and evaluating evidence-approved atmosphere spectra.",
    "Source coordinates and flux semantics are explicit per product.",
    boundary=ExecutionBoundary.MIXED,
)
