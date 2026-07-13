"""Unit contracts for scientific-contract vocabulary and records."""

import dataclasses

import pytest

from jaxstro.contracts import (
    ADSemantics,
    CallableContract,
    EvidenceKind,
    EvidenceReference,
    MaturityLevel,
    SupportLevel,
    TransformContract,
)


def test_callable_contract_is_frozen_and_uses_validated_vocabulary() -> None:
    evidence = EvidenceReference(
        "root.value.quadratic",
        EvidenceKind.VALIDATION_TEST,
        "tests/validation/test_bracketed_root_algorithms.py",
        "analytic quadratic root",
    )
    contract = CallableContract(
        id="numerics.safeguarded_bracketed_root",
        import_path="jaxstro.numerics.safeguarded_bracketed_root",
        purpose="Auditable value-first scalar root solve.",
        domain="Finite scalar endpoints and residuals.",
        outputs="BracketedRootResult",
        transforms=(
            TransformContract(
                "jit", SupportLevel.SUPPORTED, evidence_ids=(evidence.id,)
            ),
        ),
        ad_semantics=ADSemantics.VALUE_FIRST,
        precision="float32 and float64",
        maturity=MaturityLevel.VALIDATED,
        evidence=(evidence,),
    )

    assert contract.evidence[0].id == "root.value.quadratic"
    assert SupportLevel.UNVERIFIED is not SupportLevel.UNSUPPORTED
    assert SupportLevel.CONDITIONAL is not SupportLevel.SUPPORTED
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.purpose = "changed"  # type: ignore[misc]
