from jaxstro.contracts._core import module_contract
from jaxstro.contracts.schema import MaturityLevel

MODULE_CONTRACT = module_contract(
    "quad",
    (
        "Canonical integration namespace, typed configuration, and current "
        "sampled/fixed helper facade."
    ),
    (
        "Adaptive controllers, quantity-valued integration, physical-model "
        "policy, inference, ODE solving, or scientific acceptance."
    ),
    (
        "Configuring domains, measures, tolerances, results, and current "
        "integration helpers."
    ),
    (
        "Caller-owned units; Phase A0 records metadata but performs no "
        "quantity integration."
    ),
    maturity=MaturityLevel.EXPERIMENTAL,
)
