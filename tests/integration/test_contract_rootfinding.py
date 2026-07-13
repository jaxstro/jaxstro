"""Scientific-contract assertions for scalar rootfinding."""

from jaxstro.contracts import ADSemantics, SupportLevel, get_callable_contract


def test_value_root_separates_transform_and_cost_claims() -> None:
    record = get_callable_contract("jaxstro.numerics.safeguarded_bracketed_root")
    transforms = {item.transform: item for item in record.transforms}
    assert record.ad_semantics is ADSemantics.VALUE_FIRST
    assert transforms["jit"].support is SupportLevel.SUPPORTED
    assert transforms["vmap"].support is SupportLevel.CONDITIONAL
    assert "physical per-lane skipping" in transforms["vmap"].conditions
    assert "lax.map" in record.cost_notes


def test_implicit_root_requires_certification() -> None:
    record = get_callable_contract("jaxstro.numerics.implicit_bracketed_root")
    assert record.ad_semantics is ADSemantics.CERTIFIED_IMPLICIT
    assert any("unique" in item.lower() for item in record.limitations)
    assert any(item.kind.value == "validation_test" for item in record.evidence)
