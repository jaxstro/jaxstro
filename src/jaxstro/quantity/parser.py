"""Strict parser and formatter for quantity units."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from .errors import UnitParseError, UnitRegistryError
from .registry import UnitRegistry
from .unit import Unit

_MAX_DECIMAL_DENOMINATOR = 16
_DECIMAL_TOL = 1.0e-12
_TOKEN_RE = re.compile(
    r"\s*(?:(?P<number>-?(?:\d+(?:\.\d*)?|\.\d+))|"
    r"(?P<symbol>[A-Za-z_][A-Za-z0-9_]*|µm)|(?P<op>\*\*|[*/^()]))"
)


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str


def parse_unit(expr: str, *, registry: UnitRegistry | None = None) -> Unit:
    """Parse a strict unit expression into a :class:`Unit`."""

    from . import GLOBAL_REGISTRY

    parser = _Parser(_tokenize(expr), registry or GLOBAL_REGISTRY)
    unit = parser.parse_expression()
    if parser.peek() is not None:
        token = parser.peek()
        assert token is not None
        raise UnitParseError(
            f"Unexpected token {token.value!r} at the end of unit expression.",
            operation="unit-parse",
            actual=token.value,
        )
    return unit


def format_unit(unit: Unit) -> str:
    """Return the deterministic canonical string for a unit."""

    return str(unit)


def _tokenize(expr: str) -> list[_Token]:
    tokens: list[_Token] = []
    pos = 0
    while pos < len(expr):
        match = _TOKEN_RE.match(expr, pos)
        if match is None:
            raise UnitParseError(
                f"Unexpected character {expr[pos]!r} in unit expression.",
                operation="unit-tokenize",
                actual=expr[pos],
            )
        pos = match.end()
        if match.lastgroup == "number":
            tokens.append(_Token("number", match.group("number")))
        elif match.lastgroup == "symbol":
            tokens.append(_Token("symbol", match.group("symbol")))
        else:
            op = match.group("op")
            tokens.append(_Token(op, op))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Token], registry: UnitRegistry) -> None:
        self.tokens = tokens
        self.registry = registry
        self.pos = 0

    def peek(self) -> _Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def pop(self) -> _Token:
        token = self.peek()
        if token is None:
            raise UnitParseError(
                "Unexpected end of unit expression.", operation="unit-parse"
            )
        self.pos += 1
        return token

    def accept(self, kind: str) -> bool:
        token = self.peek()
        if token is not None and token.kind == kind:
            self.pop()
            return True
        return False

    def expect(self, kind: str) -> _Token:
        token = self.peek()
        if token is None:
            raise UnitParseError(
                f"Expected {kind!r}, got end of expression.",
                operation="unit-parse",
                expected=kind,
                actual=None,
            )
        self.pos += 1
        if token.kind != kind:
            raise UnitParseError(
                f"Expected {kind!r}, got {token.value!r}.",
                operation="unit-parse",
                expected=kind,
                actual=token.value,
            )
        return token

    def parse_expression(self) -> Unit:
        unit = self.parse_power()
        while (token := self.peek()) is not None and token.kind != ")":
            if self.accept("*"):
                unit = unit * self.parse_power()
            elif self.accept("/"):
                unit = unit / self.parse_power()
            elif token.kind in {"symbol", "("}:
                unit = unit * self.parse_power()
            else:
                break
        return unit

    def parse_power(self) -> Unit:
        unit = self.parse_primary()
        token = self.peek()
        if token is not None and token.kind in {"^", "**"}:
            self.pop()
            unit = unit ** self.parse_exponent()
        return unit

    def parse_primary(self) -> Unit:
        token = self.pop()
        if token.kind == "symbol":
            if token.value == "sqrt":
                self.expect("(")
                unit = self.parse_expression()
                self.expect(")")
                return unit ** Fraction(1, 2)
            try:
                return self.registry.lookup(token.value)
            except UnitRegistryError as exc:
                message = f"Unknown unit symbol {token.value!r}."
                if exc.expected:
                    message += f" Did you mean {', '.join(exc.expected)}?"
                raise UnitParseError(
                    message,
                    operation="unit-parse",
                    expected=exc.expected,
                    actual=token.value,
                ) from exc
        if token.kind == "(":
            unit = self.parse_expression()
            self.expect(")")
            return unit
        raise UnitParseError(
            f"Expected unit symbol or parenthesized expression, got {token.value!r}.",
            operation="unit-parse",
            expected="unit",
            actual=token.value,
        )

    def parse_exponent(self) -> Fraction:
        if self.accept("("):
            numerator = self.expect("number").value
            if self.accept("/"):
                denominator = self.expect("number").value
                self.expect(")")
                return Fraction(int(numerator), int(denominator))
            self.expect(")")
            return _parse_number_exponent(numerator)
        return _parse_number_exponent(self.expect("number").value)


def _parse_number_exponent(text: str) -> Fraction:
    if "." not in text:
        return Fraction(int(text), 1)
    exact = Fraction(text)
    rational = exact.limit_denominator(_MAX_DECIMAL_DENOMINATOR)
    if abs(float(exact - rational)) > _DECIMAL_TOL:
        raise UnitParseError(
            f"Decimal exponent {text!r} is not cleanly rationalizable. "
            "Write it as a rational exponent such as ^(1/3).",
            operation="unit-parse-exponent",
            actual=text,
        )
    return rational


__all__ = ["format_unit", "parse_unit"]
