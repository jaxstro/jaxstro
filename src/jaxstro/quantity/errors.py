"""Structured exceptions for :mod:`jaxstro.quantity`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(eq=False)
class QuantityError(Exception):
    """Base class for quantity-system errors with inspectable context."""

    message: str
    operation: str | None = None
    expected: Any = None
    actual: Any = None

    def __str__(self) -> str:
        return self.message


class DimensionError(QuantityError):
    """Raised when dimension arithmetic or compatibility checks fail."""


class UnitConversionError(QuantityError):
    """Raised when units cannot be converted exactly or by equivalency."""


class UnitParseError(QuantityError):
    """Raised when a unit expression cannot be parsed."""


class UnitRegistryError(QuantityError):
    """Raised when a unit registry lookup or mutation fails."""


class EquivalencyError(QuantityError):
    """Raised when an equivalency cannot perform a requested conversion."""
