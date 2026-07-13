from jaxstro.contracts import ExecutionBoundary
from jaxstro.contracts._core import module_contract

MODULE_CONTRACT = module_contract(
    "atmospheres",
    "Catalog and artifact preparation plus evidence-gated evaluation.",
    "Photometry or model validity.",
    boundary=ExecutionBoundary.MIXED,
)
