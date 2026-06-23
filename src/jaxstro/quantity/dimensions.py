"""Exact physical dimensions for :mod:`jaxstro.quantity`.

The canonical exponent order is mass, length, time, temperature, current,
amount, and luminosity. Exponents are always :class:`fractions.Fraction`
objects so equality, hashing, powers, parsing, and serialization are stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar

from .errors import DimensionError

DIMENSION_NAMES: tuple[str, ...] = (
    "mass",
    "length",
    "time",
    "temperature",
    "current",
    "amount",
    "luminosity",
)
_N_DIMS = len(DIMENSION_NAMES)


def _coerce_exponent(value: int | Fraction) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, int | Fraction):
        raise DimensionError(
            "Dimension exponents must be exact rational values; rationalize "
            "decimals before constructing dimensions.",
            operation="dimension-exponent",
            actual=value,
        )
    return Fraction(value)


@dataclass(frozen=True, eq=False)
class Dimension:
    """Fixed-vector exact physical dimension."""

    exponents: tuple[Fraction, ...]
    name: str | None = None

    _names: ClassVar[tuple[str, ...]] = DIMENSION_NAMES

    def __post_init__(self) -> None:
        if len(self.exponents) != _N_DIMS:
            raise DimensionError(
                f"Expected {_N_DIMS} dimension exponents, got {len(self.exponents)}.",
                operation="dimension-init",
                expected=_N_DIMS,
                actual=len(self.exponents),
            )
        object.__setattr__(
            self, "exponents", tuple(_coerce_exponent(x) for x in self.exponents)
        )

    @classmethod
    def from_powers(cls, **powers: int | Fraction) -> "Dimension":
        """Construct a dimension from named exact powers."""

        unknown = set(powers) - set(cls._names)
        if unknown:
            raise DimensionError(
                f"Unknown dimension names: {', '.join(sorted(unknown))}.",
                operation="dimension-from-powers",
                actual=tuple(sorted(unknown)),
            )
        values = [Fraction(0) for _ in cls._names]
        for key, value in powers.items():
            values[cls._names.index(key)] = _coerce_exponent(value)
        return cls(tuple(values))

    @classmethod
    def dimensionless(cls) -> "Dimension":
        """Return the canonical dimensionless singleton."""

        return dimensionless

    @property
    def is_dimensionless(self) -> bool:
        return all(power == 0 for power in self.exponents)

    def is_compatible_with(self, other: "Dimension") -> bool:
        return self == other

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Dimension) and self.exponents == other.exponents

    def __hash__(self) -> int:
        return hash(self.exponents)

    def __mul__(self, other: "Dimension") -> "Dimension":
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def __truediv__(self, other: "Dimension") -> "Dimension":
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def __pow__(self, power: int | Fraction) -> "Dimension":
        exponent = _coerce_exponent(power)
        return Dimension(tuple(value * exponent for value in self.exponents))

    def __repr__(self) -> str:
        if self.name is not None:
            return f"Dimension({self.name})"
        terms = [
            f"{name}^{power}" if power != 1 else name
            for name, power in zip(self._names, self.exponents)
            if power
        ]
        return (
            "Dimension(dimensionless)"
            if not terms
            else "Dimension(" + " ".join(terms) + ")"
        )


dimensionless = Dimension((Fraction(0),) * _N_DIMS, name="dimensionless")
mass = Dimension.from_powers(mass=1)
length = Dimension.from_powers(length=1)
time = Dimension.from_powers(time=1)
temperature = Dimension.from_powers(temperature=1)
current = Dimension.from_powers(current=1)
amount = Dimension.from_powers(amount=1)
luminosity = Dimension.from_powers(luminosity=1)

velocity = length / time
acceleration = velocity / time
energy = mass * (length**2) / (time**2)
power = energy / time

__all__ = [
    "DIMENSION_NAMES",
    "Dimension",
    "acceleration",
    "amount",
    "current",
    "dimensionless",
    "energy",
    "length",
    "luminosity",
    "mass",
    "power",
    "temperature",
    "time",
    "velocity",
]
