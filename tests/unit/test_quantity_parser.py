"""Tests for unit parsing and canonical formatting."""

from fractions import Fraction

import pytest

from jaxstro import quantity as q
from jaxstro.quantity.errors import UnitParseError


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("cm", q.cm),
        ("km/s", q.km / q.s),
        ("Msun/yr", q.Msun / q.yr),
        ("erg / s / cm^2 / Hz", q.erg / q.s / q.cm**2 / q.Hz),
        ("g cm^2 s^-2", q.erg),
        ("(km/s)^2", (q.km / q.s) ** 2),
        ("cm^(1/2)", q.cm ** Fraction(1, 2)),
        ("cm^0.5", q.cm ** Fraction(1, 2)),
        ("sqrt(cm)", q.cm ** Fraction(1, 2)),
        ("(erg/cm^3)^0.5", (q.erg / q.cm**3) ** Fraction(1, 2)),
    ],
)
def test_accepted_unit_strings(expr, expected):
    assert q.parse_unit(expr) == expected


def test_unknown_symbol_has_suggestion():
    with pytest.raises(UnitParseError) as exc:
        q.parse_unit("Msum")

    assert "Msun" in exc.value.expected
    assert "Did you mean" in str(exc.value)


def test_arbitrary_function_names_are_rejected():
    with pytest.raises(UnitParseError, match="Unknown unit symbol"):
        q.parse_unit("sin(cm)")


def test_unclean_decimal_exponent_rejected():
    with pytest.raises(UnitParseError, match="Write it as a rational exponent"):
        q.parse_unit("cm^0.333333333")


def test_malformed_parentheses_are_rejected():
    with pytest.raises(UnitParseError, match="Expected"):
        q.parse_unit("(cm/s")


def test_canonical_formatting_is_deterministic():
    assert q.format_unit(q.parse_unit("cm^0.5")) == "cm^(1/2)"
    assert q.format_unit(q.parse_unit("g cm^2 s^-2")) == "erg"
    assert q.format_unit(q.parse_unit("(km/s)^2")) == "(km/s)^2"
