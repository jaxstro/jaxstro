from __future__ import annotations

from jaxstro.contracts import (
    EvidenceKind,
    ExecutionBoundary,
    MaturityLevel,
    SupportLevel,
    get_callable_contract,
    get_module_contract,
)
from jaxstro.contracts.profiles import QUALIFIED_CORE_MODULES_V1, QUALIFIED_CORE_V1


def test_qualified_core_v1_is_evidence_complete() -> None:
    assert QUALIFIED_CORE_MODULES_V1 == ("jaxstro.units",)
    assert QUALIFIED_CORE_V1 == (
        "jaxstro.numerics.safeguarded_bracketed_root",
        "jaxstro.numerics.implicit_bracketed_root",
        "jaxstro.numerics.universal_kepler_step",
    )
    module = get_module_contract(QUALIFIED_CORE_MODULES_V1[0])
    assert module.maturity is MaturityLevel.VALIDATED
    assert len(module.evidence) == 1
    assert module.evidence[0].kind is EvidenceKind.UNIT_TEST
    assert module.evidence[0].target == "tests/unit/test_units.py"
    assert module.non_ownership
    assert module.execution_boundary is ExecutionBoundary.STATIC
    assert "CGS" in module.dimensional_policy
    for path in QUALIFIED_CORE_V1:
        contract = get_callable_contract(path)
        assert contract.maturity is MaturityLevel.VALIDATED
        assert contract.evidence and contract.limitations and contract.boundaries
        assert any(
            item.kind is EvidenceKind.VALIDATION_TEST for item in contract.evidence
        )
    kepler = get_callable_contract("jaxstro.numerics.universal_kepler_step")
    transforms = {item.transform: item.support for item in kepler.transforms}
    assert transforms == {
        "jit": SupportLevel.SUPPORTED,
        "vmap": SupportLevel.SUPPORTED,
        "jvp": SupportLevel.CONDITIONAL,
        "vjp": SupportLevel.CONDITIONAL,
    }
