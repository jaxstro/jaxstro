from jaxstro.contracts import ExecutionBoundary
from jaxstro.contracts._core import module_contract

MODULE_CONTRACT = module_contract(
    "spatial",
    "Spatial indexing, candidates, and exact pairs.",
    "Force or encounter semantics.",
    boundary=ExecutionBoundary.MIXED,
)
