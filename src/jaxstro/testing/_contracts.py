from dataclasses import replace

from jaxstro.contracts import (
    ADSemantics,
    CallableContract,
    EvidenceKind,
    EvidenceReference,
    ExecutionBoundary,
    MaturityLevel,
)
from jaxstro.contracts._core import module_contract

MODULE_CONTRACT = module_contract(
    "testing",
    "Validation and provenance tooling.",
    "Runtime scientific acceptance.",
    "Auditing derivatives, numerical evidence, and provenance cards.",
    "Every reported metric retains producer-declared units.",
    boundary=ExecutionBoundary.TOOLING,
)


def _tool(name: str, purpose: str, target: str) -> CallableContract:
    evidence = EvidenceReference(
        f"testing.{name}",
        EvidenceKind.INTEGRATION_TEST,
        target,
        "Deterministic validation-tool behavior.",
    )
    return CallableContract(
        id=f"testing.{name}",
        import_path=f"jaxstro.testing.{name}",
        purpose=purpose,
        domain="Validation and documentation tooling inputs.",
        outputs="Structured validation evidence.",
        ad_semantics=ADSemantics.VALIDATION_ONLY,
        precision="Tool-owned and explicit.",
        maturity=MaturityLevel.VALIDATED,
        evidence=(evidence,),
        limitations=("Does not determine downstream scientific acceptance.",),
    )


MODULE_CONTRACT = replace(
    MODULE_CONTRACT,
    callables=(
        _tool(
            "compare_gradients",
            "Compare automatic and finite-difference derivatives.",
            "tests/integration/test_grad_audit.py",
        ),
        _tool(
            "validate_card",
            "Validate source-backed provenance cards.",
            "tests/validation/provenance_cards/test_registry.py",
        ),
        _tool(
            "render_registry",
            "Render deterministic provenance-card pages.",
            "tests/validation/provenance_cards/test_registry.py",
        ),
    ),
)
