from jaxstro.contracts._core import module_contract

MODULE_CONTRACT = module_contract(
    "params",
    "Selective PyTree/vector parameter bridges.",
    "Inference algorithms or identifiability.",
    "Mapping selected array leaves to unconstrained parameter vectors.",
    "Leaf units remain caller-owned through transformations.",
)
