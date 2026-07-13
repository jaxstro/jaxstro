from jaxstro.contracts import ExecutionBoundary
from jaxstro.contracts._core import module_contract

MODULE_CONTRACT = module_contract(
    "testing",
    "Validation and provenance tooling.",
    "Runtime scientific acceptance.",
    boundary=ExecutionBoundary.TOOLING,
)
