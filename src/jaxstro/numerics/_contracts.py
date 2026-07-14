from dataclasses import replace

from jaxstro.contracts import (
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
from jaxstro.contracts._core import module_contract


def _root_evidence(
    name: str,
    target: str,
    claim: str,
) -> EvidenceReference:
    kind = (
        EvidenceKind.UNIT_TEST
        if target.startswith("tests/unit/")
        else EvidenceKind.VALIDATION_TEST
    )
    return EvidenceReference(f"root.{name}", kind, target, claim)


_ROOT_PERFORMANCE_GATES = tuple(
    f"{case}.hybrid-no-more-evaluations"
    for case in (
        "flat_slope",
        "linear",
        "monotone_kink",
        "oscillatory_fixed_point_residual",
        "quadratic",
    )
)


def _root_performance_evidence(name: str) -> EvidenceReference:
    return EvidenceReference(
        f"root.{name}.performance",
        EvidenceKind.BENCHMARK,
        "scripts/benchmark_rootfinding.py",
        "All registered analytic cases satisfy the hybrid evaluation-count gate.",
        artifact_id="rootfinding.performance",
        evidence_class="computational",
        artifact_comparison_ids=_ROOT_PERFORMANCE_GATES,
    )


def _value_root_contract(name: str, purpose: str) -> CallableContract:
    target = (
        "tests/unit/test_bracketed_root.py"
        if name == "map_safeguarded_bracketed_root"
        else "tests/unit/test_numerics.py"
    )
    evidence = _root_evidence(
        name,
        target,
        "The registered solver's JIT/batching, terminal status, trace, and analytic-root behavior.",
    )
    performance = _root_performance_evidence(name)
    return CallableContract(
        id=f"numerics.{name}",
        import_path=f"jaxstro.numerics.{name}",
        purpose=purpose,
        domain="Finite scalar endpoints enclosing an exact root or a typed missing-bracket result.",
        outputs="Fixed-shape value-first bracket evidence and terminal status.",
        ad_semantics=ADSemantics.VALUE_FIRST,
        precision="float32 and float64; tolerances are caller-owned.",
        maturity=MaturityLevel.VALIDATED,
        transforms=(
            TransformContract(
                "jit", SupportLevel.SUPPORTED, evidence_ids=(evidence.id,)
            ),
            TransformContract(
                "vmap",
                SupportLevel.CONDITIONAL,
                conditions="Values and shapes are preserved, but physical per-lane skipping is not guaranteed.",
                evidence_ids=(evidence.id,),
            ),
        ),
        boundaries=(
            BoundaryContract(
                "Missing sign bracket or nonfinite admissible evaluation.",
                FailureMode.STRUCTURED_RESULT,
                (evidence.id,),
            ),
        ),
        evidence=(evidence, performance),
        limitations=("No implicit-root derivative claim.",),
        cost_notes="Use lax.map when physical per-lane skipping of expensive residuals matters.",
    )


_implicit_evidence = _root_evidence(
    "implicit_bracketed_root",
    "tests/validation/test_implicit_root_gradients.py",
    "Certified sensitivities agree with analytic and central finite differences.",
)
_IMPLICIT_GATES = tuple(
    f"{case}.{gate}"
    for case in ("exponential", "linear", "quadratic")
    for gate in (
        "absolute_residual.gate",
        "bracket_width.gate",
        "relative_ad_analytic_error.gate",
        "relative_ad_fd_error.gate",
        "slope_magnitude.gate",
        "certificate.gate",
    )
)
_implicit_artifact_evidence = EvidenceReference(
    "root.implicit_bracketed_root.certification",
    EvidenceKind.ARTIFACT,
    "scripts/benchmark_implicit_root.py",
    "All registered primal and derivative certificate comparisons pass.",
    artifact_id="rootfinding.implicit-gradients",
    evidence_class="computational",
    artifact_comparison_ids=_IMPLICIT_GATES,
)

ROOT_CALLABLES: tuple[CallableContract, ...] = (
    _value_root_contract(
        "safeguarded_bracketed_root", "Auditable scalar safeguarded root solve."
    ),
    _value_root_contract(
        "map_safeguarded_bracketed_root",
        "Mapped safeguarded scalar root solves with per-lane control flow.",
    ),
    CallableContract(
        id="numerics.implicit_bracketed_root",
        import_path="jaxstro.numerics.implicit_bracketed_root",
        purpose="Fail-closed implicit-function derivative for a certified scalar root.",
        domain="Caller asserts a unique root and smooth branch; numerical certificate gates must pass.",
        outputs="ImplicitRootResult with primal evidence and derivative certificate.",
        ad_semantics=ADSemantics.CERTIFIED_IMPLICIT,
        precision="float32 and float64; certificate tolerances are explicit.",
        maturity=MaturityLevel.VALIDATED,
        transforms=(
            TransformContract(
                "jit", SupportLevel.SUPPORTED, evidence_ids=(_implicit_evidence.id,)
            ),
        ),
        boundaries=(
            BoundaryContract(
                "Rejected assumption or numerical certificate.",
                FailureMode.RETURNS_NAN,
                (_implicit_evidence.id,),
            ),
        ),
        evidence=(_implicit_evidence, _implicit_artifact_evidence),
        limitations=(
            "Requires a unique mathematical root.",
            "Requires a smooth selected branch and adequate slope conditioning.",
        ),
        cost_notes="Runs the safeguarded primal before exposing a custom-root derivative.",
    ),
)


def _primitive(name: str, purpose: str) -> CallableContract:
    evidence = EvidenceReference(
        f"root.{name}",
        EvidenceKind.UNIT_TEST,
        "tests/unit/test_bracketed_root.py",
        "Deterministic low-level bracket behavior.",
    )
    return CallableContract(
        id=f"numerics.{name}",
        import_path=f"jaxstro.numerics.{name}",
        purpose=purpose,
        domain="Finite scalar bracket evidence.",
        outputs="Fixed-shape bracket state or proposal.",
        ad_semantics=ADSemantics.VALUE_FIRST,
        precision="float32 and float64.",
        maturity=MaturityLevel.VALIDATED,
        evidence=(evidence,),
        limitations=("Primary purpose is auditable forward-value control flow.",),
    )


ROOT_CALLABLES += (
    _primitive(
        "initialize_bracket",
        "Validate endpoint evidence and initialize a sign bracket.",
    ),
    _primitive(
        "update_bracket", "Update one bracket side without losing valid evidence."
    ),
    _primitive(
        "propose_bracketed",
        "Choose deterministic safeguarded interpolation or midpoint fallback.",
    ),
)

_kepler_value_evidence = EvidenceReference(
    "numerics.universal_kepler_step.value",
    EvidenceKind.UNIT_TEST,
    "tests/unit/test_kepler.py",
    "Elliptic, near-parabolic, hyperbolic, reverse, scale, transform, and typed-failure behavior.",
)
_kepler_gradient_evidence = EvidenceReference(
    "numerics.universal_kepler_step.fixed_route_ad",
    EvidenceKind.VALIDATION_TEST,
    "tests/validation/test_kepler_gradients.py",
    "Fixed-route JVP and VJP directional derivatives agree with central finite differences.",
)
_kepler_fixed_route = (
    "Continuous state on one converged route with fixed shape, iteration budget, "
    "status path, and Stumpff branch."
)
KEPLER_CALLABLES: tuple[CallableContract, ...] = (
    CallableContract(
        id="numerics.universal_kepler_step",
        import_path="jaxstro.numerics.universal_kepler_step",
        purpose="Propagate a relative Cartesian two-body state across any Newtonian conic.",
        domain="Finite relative Cartesian vectors, positive mu, signed finite dt, and mutually consistent caller-owned units.",
        outputs="UniversalKeplerResult with Cartesian state, residual, iteration count, and exhaustive status.",
        ad_semantics=ADSemantics.SMOOTH_PATHWISE,
        precision="float32 and float64; invariant and derivative validation uses float64.",
        maturity=MaturityLevel.VALIDATED,
        transforms=(
            TransformContract(
                "jit",
                SupportLevel.SUPPORTED,
                evidence_ids=(_kepler_value_evidence.id,),
            ),
            TransformContract(
                "vmap",
                SupportLevel.SUPPORTED,
                evidence_ids=(_kepler_value_evidence.id,),
            ),
            TransformContract(
                "jvp",
                SupportLevel.CONDITIONAL,
                conditions=_kepler_fixed_route,
                evidence_ids=(_kepler_gradient_evidence.id,),
            ),
            TransformContract(
                "vjp",
                SupportLevel.CONDITIONAL,
                conditions=_kepler_fixed_route,
                evidence_ids=(_kepler_gradient_evidence.id,),
            ),
        ),
        boundaries=(
            BoundaryContract(
                "Invalid input, nonfinite iteration, singular radius, or exhausted iteration budget.",
                FailureMode.STRUCTURED_RESULT,
                (_kepler_value_evidence.id,),
            ),
        ),
        evidence=(_kepler_value_evidence, _kepler_gradient_evidence),
        limitations=(
            "No derivative claim across status, iteration-count, Stumpff-route, conic-label, or collision boundaries.",
            "No implicit-root derivative claim; AD follows the finite executed Newton map.",
            "Units, object identity, encounter selection, and state-commit policy belong to callers.",
        ),
        cost_notes="Runs a fixed 12-slot Newton scan by default; converged lanes freeze numerically.",
    ),
)


def _smooth_callable(
    name: str,
    purpose: str,
    domain: str,
    target: str,
    *,
    boundaries: tuple[BoundaryContract, ...] = (),
    public_path: str | None = None,
    limitations: tuple[str, ...] = (),
) -> CallableContract:
    evidence = EvidenceReference(
        f"numerics.{name}",
        EvidenceKind.VALIDATION_TEST,
        target,
        "Analytic values and central finite differences validate the smooth domain.",
    )
    linked = tuple(replace(item, evidence_ids=(evidence.id,)) for item in boundaries)
    return CallableContract(
        id=f"numerics.{name}",
        import_path=public_path or f"jaxstro.numerics.{name}",
        purpose=purpose,
        domain=domain,
        outputs="JAX array values.",
        ad_semantics=ADSemantics.SMOOTH_PATHWISE,
        precision="float32 and float64; validation uses float64.",
        maturity=MaturityLevel.VALIDATED,
        boundaries=linked,
        evidence=(evidence,),
        limitations=limitations,
    )


_grid_boundaries = (
    BoundaryContract(
        "clamp policy holds queries at the nearest boundary.", FailureMode.SATURATES
    ),
    BoundaryContract(
        "reject policy raises for invalid concrete queries.", FailureMode.RAISES
    ),
)
EXEMPLAR_CALLABLES = tuple(
    _smooth_callable(
        name,
        f"Finite power-law {purpose}.",
        "Finite ordered support with positive bounds; smooth through alpha = -1.",
        "tests/validation/test_grad_checks.py",
        boundaries=(boundary,),
    )
    for name, purpose, boundary in (
        (
            "powerlaw_logpdf",
            "log density and normalization",
            BoundaryContract(
                "Outside x support returns negative infinity.",
                FailureMode.RETURNS_SENTINEL,
            ),
        ),
        (
            "powerlaw_cdf",
            "cumulative distribution",
            BoundaryContract(
                "Outside x support clamps to zero or one.", FailureMode.SATURATES
            ),
        ),
        (
            "powerlaw_ppf",
            "inverse cumulative distribution",
            BoundaryContract(
                "Quantile input is defined on the closed unit interval.",
                FailureMode.UNDEFINED,
            ),
        ),
    )
) + (
    _smooth_callable(
        "interp1d",
        "Piecewise-linear interpolation.",
        "Ordered one-dimensional nodes; query sensitivities are smooth only off knots and clamp boundaries.",
        "tests/validation/test_grad_checks.py",
        boundaries=(
            BoundaryContract(
                "Queries outside the table clamp to endpoint values.",
                FailureMode.SATURATES,
            ),
        ),
        public_path="jaxstro.numerics.interpolation.interp1d",
        limitations=("AD evidence covers interior off-knot, branch-stable queries.",),
    ),
    _smooth_callable(
        "monotone_cubic_interp",
        "Shape-preserving cubic interpolation.",
        "Strictly ordered nodes and monotone data on a branch-stable PCHIP slope selection.",
        "tests/validation/test_grad_checks.py",
        public_path="jaxstro.numerics.interpolation.monotone_cubic_interp",
        limitations=(
            "Sign, plateau, knot, and overshoot branch boundaries are nonsmooth.",
        ),
    ),
    _smooth_callable(
        "regular_grid_interp",
        "Static-rank tensor-product interpolation.",
        "Strictly increasing grid axes and explicit boundary policy.",
        "tests/validation/test_grad_checks.py",
        boundaries=_grid_boundaries,
        limitations=(
            "Coordinate derivatives are claimed only inside branch-stable cells.",
        ),
    ),
)

MODULE_CONTRACT = module_contract(
    "numerics",
    "Generic numerical mechanics.",
    "Domain acceptance, retry policy, or physical state.",
    "Reusable numerical maps for differentiable science.",
    "Caller-owned units; each callable declares dimensional behavior.",
)
MODULE_CONTRACT = replace(
    MODULE_CONTRACT,
    callables=ROOT_CALLABLES + KEPLER_CALLABLES + EXEMPLAR_CALLABLES,
)
