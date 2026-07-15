from dataclasses import replace

from jaxstro.contracts._core import module_contract
from jaxstro.contracts.schema import (
    ADSemantics,
    BoundaryContract,
    CallableContract,
    EvidenceKind,
    EvidenceReference,
    FailureMode,
    MaturityLevel,
    SupportLevel,
    TransformContract,
)

_FIXED_UNIT_EVIDENCE = EvidenceReference(
    id="quad-fixed-unit",
    kind=EvidenceKind.UNIT_TEST,
    target="tests/unit/quad/test_fixed.py",
    claim=(
        "Analytic exactness, domains, measures, payloads, breakpoints, "
        "orientation, zero width, and structural failures are executable."
    ),
)

_FIXED_TRANSFORM_EVIDENCE = EvidenceReference(
    id="quad-fixed-transforms",
    kind=EvidenceKind.INTEGRATION_TEST,
    target="tests/integration/test_quad_fixed_transforms.py",
    claim=(
        "JIT, VMAP, real and complex payloads, explicit-parameter gradients, "
        "and moving-bound gradients are executable."
    ),
)

_FIXED_CONTRACT = CallableContract(
    id="quad-fixed",
    import_path="jaxstro.quad.fixed",
    purpose=(
        "Evaluate one declared fixed quadrature formula over a supported "
        "one-dimensional domain and measure."
    ),
    domain=(
        "Raw-array scalar bounds, static rule and measure types, and an "
        "integrand with a leading node axis."
    ),
    outputs="An array value with the integrand node axis reduced.",
    ad_semantics=ADSemantics.SMOOTH_PATHWISE,
    precision=(
        "The active JAX precision policy; scientific reference validation uses float64."
    ),
    maturity=MaturityLevel.EXPERIMENTAL,
    transforms=(
        TransformContract(
            "jax.jit",
            SupportLevel.SUPPORTED,
            conditions=(
                "Rule type, rule order or level, measure type, breakpoint "
                "count, and payload shape remain static."
            ),
            evidence_ids=("quad-fixed-transforms",),
        ),
        TransformContract(
            "jax.vmap",
            SupportLevel.SUPPORTED,
            conditions="Batch explicit arguments or numerical bounds.",
            evidence_ids=("quad-fixed-transforms",),
        ),
    ),
    boundaries=(
        BoundaryContract(
            "Unsupported rule, domain, and measure pairings raise eagerly.",
            FailureMode.RAISES,
            evidence_ids=("quad-fixed-unit",),
        ),
        BoundaryContract(
            "Value-dependent invalid finite domains return NaN under tracing.",
            FailureMode.RETURNS_NAN,
            evidence_ids=("quad-fixed-unit",),
        ),
    ),
    evidence=(_FIXED_UNIT_EVIDENCE, _FIXED_TRANSFORM_EVIDENCE),
    limitations=(
        "A fixed rule does not estimate truncation error or select its order.",
        "Quantity-valued integration and adaptive replay derivatives are not implemented.",
    ),
    cost_notes=(
        "Integrand work is the static node count multiplied by the static "
        "number of finite breakpoint segments."
    ),
)

MODULE_CONTRACT = replace(
    module_contract(
        "quad",
        (
            "Canonical sampled-data integration, fixed one-dimensional "
            "quadrature, typed domains, measures, rules, and result evidence."
        ),
        (
            "Adaptive controllers, quantity-valued integration, physical-model "
            "policy, inference, ODE solving, or scientific acceptance."
        ),
        (
            "Integrating sampled arrays and evaluating declared fixed formulas "
            "over supported one-dimensional domains."
        ),
        "Caller-owned units; Phase A1 accepts raw arrays only.",
        maturity=MaturityLevel.EXPERIMENTAL,
    ),
    callables=(_FIXED_CONTRACT,),
)
