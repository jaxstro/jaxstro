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
        "stability-ladder, and failure-status evidence; representative "
        "Gauss-Kronrod quantity-rescaling evidence covers the alpha adapter."
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
        "scaling, density units, improper-map scale representation, and JIT "
        "and VMAP composition."
    ),
)

_ADAPTIVE_PERFORMANCE_EVIDENCE = EvidenceReference(
    id="quad-performance",
    kind=EvidenceKind.ARTIFACT,
    target="docs/validation/quad-performance.json",
    claim=(
        "Matched Jaxstro and Quadax correctness, work, transformation, and "
        "timing evidence supports only the labeled records and recorded "
        "hardware envelope, not universal superiority."
    ),
)

_MULTIDIM_TRUTH_EVIDENCE = EvidenceReference(
    id="quad-multidim-truth",
    kind=EvidenceKind.ARTIFACT,
    target="docs/validation/quad-multidim-truth.json",
    claim=(
        "Independent analytic and numerical truths exercise tensor, cubature, "
        "sparse-grid, deterministic Sobol, and randomized Sobol values, "
        "statuses, estimator kinds, and logical work on finite hyperrectangles."
    ),
)

_MULTIDIM_REPLAY_EVIDENCE = EvidenceReference(
    id="quad-multidim-replay",
    kind=EvidenceKind.ARTIFACT,
    target="docs/validation/quad-multidim-replay.json",
    claim=(
        "First-order accepted-formula replay is checked for parameters, "
        "bounds, measures, randomized formulas, and heterogeneous quantity axes."
    ),
)

_RQMC_CALIBRATION_EVIDENCE = EvidenceReference(
    id="quad-rqmc-calibration",
    kind=EvidenceKind.ARTIFACT,
    target="docs/validation/quad-rqmc-calibration.json",
    claim=(
        "Frozen real-scalar coverage campaigns exercise fixed-look Student-t "
        "and bounded sequential empirical-Bernstein uncertainty contracts."
    ),
)

_MULTIDIM_COMPARISON_EVIDENCE = EvidenceReference(
    id="quad-multidim-comparisons",
    kind=EvidenceKind.ARTIFACT,
    target="docs/validation/quad-multidim-comparisons.json",
    claim=(
        "Comparator records carry explicit exact, strong-match, node-match, "
        "family-match, capability, or execution-model-unmatched labels."
    ),
)

_MULTIDIM_PERFORMANCE_EVIDENCE = EvidenceReference(
    id="quad-multidim-performance",
    kind=EvidenceKind.ARTIFACT,
    target="docs/validation/quad-multidim-performance-baseline.json",
    claim=(
        "The immutable host baseline records predeclared timing, compiler-cost, "
        "repeat-scaling, and analytic memory-proxy triggers without a universal "
        "performance claim."
    ),
)

_ADAPTIVE_CONTRACT = CallableContract(
    id="quad-integrate",
    import_path="jaxstro.quad.integrate",
    purpose=(
        "Estimate a supported one-dimensional or finite-hyperrectangle integral "
        "with method-specific error, status, logical-work, replay-derivative, "
        "random-state, and optional quantity evidence."
    ),
    domain=(
        "Supported one-dimensional domains or finite Hyperrectangle dimensions, "
        "a static method declaration and capacities, scalar real tolerances or "
        "the alpha quantity boundary, and an integrand with a leading node axis. "
        "Randomized confidence intervals require real scalar outputs."
    ),
    outputs=(
        "A QuadResult containing the primal estimate, family-specific error "
        "evidence, status and work records, and a first-order replay derivative "
        "on value."
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
                "Method configuration, dimension, capacities, breakpoint count, "
                "payload shape, QMC schedule, and scramble family remain static."
            ),
            evidence_ids=("quad-adaptive-transforms",),
        ),
        TransformContract(
            "jax.vmap",
            SupportLevel.SUPPORTED,
            conditions=(
                "Each batch member runs one independent bounded controller or "
                "formula; cost-sensitive heterogeneous cubature batches should "
                "use lax.map for physical per-lane skipping."
            ),
            evidence_ids=("quad-adaptive-transforms",),
        ),
        TransformContract(
            "jvp",
            SupportLevel.CONDITIONAL,
            conditions=(
                "First-order accepted-formula replay of value with parameters "
                "passed through explicit args or supported smooth bounds; "
                "controller decisions, accepted structures, confidence "
                "construction, and diagnostics are stopped."
            ),
            evidence_ids=(
                "quad-adaptive-replay",
                "quad-replay-derivatives",
                "quad-multidim-replay",
            ),
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
        _ADAPTIVE_PERFORMANCE_EVIDENCE,
        _MULTIDIM_TRUTH_EVIDENCE,
        _MULTIDIM_REPLAY_EVIDENCE,
        _RQMC_CALIBRATION_EVIDENCE,
        _MULTIDIM_COMPARISON_EVIDENCE,
        _MULTIDIM_PERFORMANCE_EVIDENCE,
    ),
    limitations=(
        "Estimator convergence is not a universal bound on true error.",
        "Related rules can miss the same unresolved narrow feature.",
        "Replay is the default first-order derivative of the accepted fixed formula; gradient=stop remains explicit.",
        "Quantity-aware adaptive integration is alpha and opt-in.",
        "Dimensional improper domains require an explicit positive physical scale.",
        "Direct Quantity-PyTree quotient-unit Jacobians and higher derivatives are not claimed.",
        "Multidimensional geometry is restricted to finite hyperrectangles.",
        "TensorProduct and deterministic Sobol provide no runtime error estimate.",
        "Adaptive tensor, cubature, and sparse-grid estimators are method-specific numerical evidence, not universal true-error bounds.",
        "Randomized confidence intervals require real scalar outputs and the declared independent scramble construction.",
        "Observed process and device peak-memory certification remains pending.",
        "No universal performance-superiority claim is established.",
    ),
    cost_notes=(
        "Regional logical work is node_count * (initial_regions + 2 * refinements); "
        "global methods report their finest completed active grid, sparse methods "
        "report unique nodes, and QMC reports point-integrand evaluations across "
        "all accepted replicates."
    ),
)

MODULE_CONTRACT = replace(
    module_contract(
        "quad",
        (
            "Canonical sampled-data integration, fixed and adaptive "
            "one-dimensional quadrature, finite-hyperrectangle tensor, cubature, "
            "sparse-grid, and randomized QMC methods, typed domains, measures, "
            "and result evidence."
        ),
        (
            "Non-hyperrectangular geometries, direct Quantity-PyTree quotient-unit "
            "Jacobians, higher derivatives, physical-model policy, inference, "
            "ODE solving, or scientific acceptance."
        ),
        (
            "Integrating sampled arrays and evaluating declared fixed formulas "
            "or bounded adaptive controllers over supported one-dimensional "
            "domains and finite hyperrectangles."
        ),
        "Raw kernels with an alpha quantity adapter, heterogeneous coordinate "
        "normalization, and unit restoration owned only by quad.integrate.",
        maturity=MaturityLevel.EXPERIMENTAL,
    ),
    callables=(_FIXED_CONTRACT, _ADAPTIVE_CONTRACT),
)
