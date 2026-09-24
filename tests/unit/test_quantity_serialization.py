"""Tests for quantity and unit serialization."""

import json

import jax.numpy as jnp
import pytest

from jaxstro import quantity as q


def test_compact_known_unit_quantity_dict_round_trips():
    quantity = q.Quantity(1.5, q.Msun / q.yr)
    payload = q.to_dict(quantity)

    assert payload == {"value": 1.5, "unit": "Msun/yr"}
    restored = q.from_dict(payload)
    assert restored.value == 1.5
    assert restored.unit == q.Msun / q.yr


def test_structured_custom_unit_fallback_round_trips():
    unit = q.Unit("code_mass", q.Msun.scale_to_cgs, q.Msun.dimensions)
    payload = q.unit_to_dict(unit)

    assert payload == {
        "symbol": "code_mass",
        "scale_cgs": q.Msun.scale_to_cgs,
        "dimensions": {"mass": 1},
    }
    assert q.unit_from_dict(payload) == unit


def test_serialization_is_deterministic_and_json_safe_for_scalars():
    quantity = q.Quantity(2.0, q.parse_unit("cm^0.5"))
    first = q.to_dict(quantity)
    second = q.to_dict(q.from_dict(first))

    assert first == second
    assert json.loads(json.dumps(first)) == first


def test_array_values_are_rejected_until_helpers_exist():
    quantity = q.Quantity(jnp.array([1.0, 2.0]), q.cm)

    with pytest.raises(TypeError, match="array quantity serialization"):
        q.to_dict(quantity)


def test_quantity_methods_delegate_to_serialization_helpers():
    quantity = q.Quantity(3.0, q.km / q.s)

    assert quantity.to_dict() == {"value": 3.0, "unit": "km/s"}
    assert q.Quantity.from_dict(quantity.to_dict()).unit == q.km / q.s


def test_unit_to_dict_does_not_hide_unrelated_errors(monkeypatch):
    """Only parse and registry failures fall back to the structured form."""
    from jaxstro.quantity import serialization

    def broken(symbol):
        raise TypeError("bug inside the parser")

    monkeypatch.setattr(serialization, "parse_unit", broken)
    with pytest.raises(TypeError, match="bug inside the parser"):
        serialization.unit_to_dict(q.parse_unit("cm"))


def test_unit_to_dict_falls_back_when_the_symbol_does_not_parse(monkeypatch):
    from jaxstro.quantity import serialization
    from jaxstro.quantity.errors import UnitParseError

    def unparseable(symbol):
        raise UnitParseError("unknown symbol")

    monkeypatch.setattr(serialization, "parse_unit", unparseable)
    payload = serialization.unit_to_dict(q.parse_unit("cm"))
    assert isinstance(payload, dict) and payload["scale_cgs"] == 1.0
