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
        "Quantity-valued fixed integration is not implemented.",
    ),
    cost_notes=(
        "Integrand work is the static node count multiplied by the static "
        "number of finite breakpoint segments."
    ),
)

_ADAPTIVE_TRANSFORM_EVIDENCE = EvidenceReference(
    id="quad-adaptive-transforms",
    kind=EvidenceKind.INTEGRATION_TEST,
    target="tests/integration/test_quad_adaptive_transforms.py",
    claim=(
        "JIT, VMAP, replay and stopped gradients, payload shapes, and bounded "
        "controller execution are executable."
    ),
)

_ADAPTIVE_VALIDATION_EVIDENCE = EvidenceReference(
    id="quad-adaptive-validation",
    kind=EvidenceKind.VALIDATION_TEST,
    target="tests/validation/test_quad_adaptive_reference.py",
    claim=(
        "Analytic, complex, vector, breakpoint, improper-domain, singular, "
        "nonfinite, and exhausted-budget cases are executable."
    ),
)

_ADAPTIVE_ARTIFACT_EVIDENCE = EvidenceReference(
    id="quad-adaptive-envelope",
    kind=EvidenceKind.ARTIFACT,
    target="docs/validation/quad-adaptive-envelope.json",
    claim=(
        "Deterministic tolerance sweeps record requested tolerances, observed "
        "errors, statuses, and logical work for all five method families."
    ),
)

_ADAPTIVE_REPLAY_EVIDENCE = EvidenceReference(
    id="quad-adaptive-replay",
    kind=EvidenceKind.VALIDATION_TEST,
    target="tests/validation/test_quad_replay_derivatives.py",
    claim=(
        "All five methods have analytic, frozen-formula, adaptive-rerun, "
        "stability-ladder, failure-status, and quantity-rescaling evidence."
    ),
)

_ADAPTIVE_REPLAY_ARTIFACT = EvidenceReference(
    id="quad-replay-derivatives",
    kind=EvidenceKind.ARTIFACT,
    target="docs/validation/quad-replay-derivatives.json",
    claim=(
        "Deterministic first-order replay measurements, units, gates, accepted "
        "regions or levels, and limitations are freshness checked."
    ),
)

_ADAPTIVE_QUANTITY_EVIDENCE = EvidenceReference(
    id="quad-adaptive-quantity",
    kind=EvidenceKind.INTEGRATION_TEST,
    target="tests/integration/test_quad_quantity_transforms.py",
    claim=(
        "The alpha quantity adapter preserves physical values, derivative "
        "scaling, density units, and JIT and VMAP composition."
    ),
)

_ADAPTIVE_CONTRACT = CallableContract(
    id="quad-integrate",
    import_path="jaxstro.quad.integrate",
    purpose=(
        "Adaptively estimate a one-dimensional integral with typed error, "
        "status, logical-work, replay-derivative, and optional quantity evidence."
    ),
    domain=(
        "Supported one-dimensional domain and measure pairings, a static "
        "method declaration and capacities, scalar real tolerances or the alpha "
        "quantity boundary, and an integrand with a leading node axis."
    ),
    outputs=(
        "A QuadResult containing the exact adaptive primal value, bounded "
        "evidence records, and a replay derivative on value."
    ),
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
                "Method configuration, capacities, breakpoint count, and "
                "payload shape remain static."
            ),
            evidence_ids=("quad-adaptive-transforms",),
        ),
        TransformContract(
            "jax.vmap",
            SupportLevel.SUPPORTED,
            conditions="Each batch member runs one independent bounded controller.",
            evidence_ids=("quad-adaptive-transforms",),
        ),
        TransformContract(
            "jvp",
            SupportLevel.CONDITIONAL,
            conditions=(
                "First-order replay of value on a successful solve with "
                "parameters passed through explicit args or smooth finite bounds."
            ),
            evidence_ids=("quad-adaptive-replay", "quad-replay-derivatives"),
        ),
        TransformContract(
            "vjp",
            SupportLevel.CONDITIONAL,
            conditions=(
                "Project value or a floating diagnostic; full integer-bearing "
                "result Jacobians are outside the contract."
            ),
            evidence_ids=("quad-adaptive-replay",),
        ),
        TransformContract(
            "jacfwd/jacrev",
            SupportLevel.CONDITIONAL,
            conditions=(
                "Apply to value only, using JAX realified conventions for complex maps."
            ),
            evidence_ids=("quad-adaptive-replay",),
        ),
    ),
    boundaries=(
        BoundaryContract(
            "Unsupported method, domain, measure, breakpoint, or capacity declarations raise eagerly.",
            FailureMode.RAISES,
            evidence_ids=("quad-adaptive-validation",),
        ),
        BoundaryContract(
            "Dynamic invalid, nonfinite, roundoff-limited, or exhausted cases return a typed status.",
            FailureMode.STRUCTURED_RESULT,
            evidence_ids=("quad-adaptive-validation",),
        ),
    ),
    evidence=(
        _ADAPTIVE_TRANSFORM_EVIDENCE,
        _ADAPTIVE_VALIDATION_EVIDENCE,
        _ADAPTIVE_ARTIFACT_EVIDENCE,
        _ADAPTIVE_REPLAY_EVIDENCE,
        _ADAPTIVE_REPLAY_ARTIFACT,
        _ADAPTIVE_QUANTITY_EVIDENCE,
    ),
    limitations=(
        "Estimator convergence is not a universal bound on true error.",
        "Related rules can miss the same unresolved narrow feature.",
        "Replay is the default first-order derivative of the accepted fixed formula; gradient=stop remains explicit.",
        "Quantity-aware adaptive integration is alpha and opt-in.",
        "Direct Quantity-PyTree quotient-unit Jacobians and higher derivatives are not claimed.",
        "Multidimensional integration is not implemented.",
        "No performance-superiority claim is established.",
    ),
    cost_notes=(
        "Regional logical work is node_count * (initial_regions + 2 * refinements); "
        "global methods report their finest completed active grid."
    ),
)

MODULE_CONTRACT = replace(
    module_contract(
        "quad",
        (
            "Canonical sampled-data integration, fixed and adaptive one-dimensional "
            "quadrature, typed domains, measures, methods, and result evidence."
        ),
        (
            "Multidimensional integration, direct Quantity-PyTree quotient-unit "
            "Jacobians, physical-model policy, inference, ODE solving, or "
            "scientific acceptance."
        ),
        (
            "Integrating sampled arrays and evaluating declared fixed formulas "
            "or bounded adaptive controllers over supported one-dimensional domains."
        ),
        "Raw kernels with an alpha quantity adapter owned only by quad.integrate.",
        maturity=MaturityLevel.EXPERIMENTAL,
    ),
    callables=(_FIXED_CONTRACT, _ADAPTIVE_CONTRACT),
)
