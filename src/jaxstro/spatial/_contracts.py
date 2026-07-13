from jaxstro.contracts import ExecutionBoundary
from jaxstro.contracts._core import module_contract

MODULE_CONTRACT = module_contract(
    "spatial",
    "Spatial indexing, candidates, and exact pairs.",
    "Force or encounter semantics.",
    "Fixed-shape spatial candidate and exact-pair mechanics.",
    "Coordinates use caller-owned consistent length units.",
    boundary=ExecutionBoundary.MIXED,
)
