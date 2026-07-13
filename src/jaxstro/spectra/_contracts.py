from jaxstro.contracts import ExecutionBoundary
from jaxstro.contracts._core import module_contract

MODULE_CONTRACT = module_contract(
    "spectra",
    "Generic spectral representations and remapping.",
    "Filters, photometry, or instruments.",
    boundary=ExecutionBoundary.MIXED,
)
