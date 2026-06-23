"""Layered unit registries for :mod:`jaxstro.quantity`."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Mapping

from .errors import UnitRegistryError
from .unit import Unit


@dataclass(frozen=True)
class UnitRegistry:
    """Exact-symbol unit registry with optional parent lookup."""

    name: str
    units: Mapping[str, Unit] = field(default_factory=dict)
    aliases: Mapping[str, str] = field(default_factory=dict)
    parent: "UnitRegistry | None" = None
    mutable: bool = False

    def lookup(self, symbol: str) -> Unit:
        key = self.aliases.get(symbol, symbol)
        if key in self.units:
            return self.units[key]
        if self.parent is not None:
            try:
                return self.parent.lookup(symbol)
            except UnitRegistryError:
                pass
        suggestions = self.suggest(symbol)
        message = f"Unknown unit symbol {symbol!r} in registry {self.name!r}."
        if suggestions:
            message += f" Did you mean {', '.join(suggestions)}?"
        raise UnitRegistryError(
            message,
            operation="unit-lookup",
            expected=tuple(suggestions),
            actual=symbol,
        )

    def suggest(self, symbol: str, *, n: int = 3) -> tuple[str, ...]:
        choices = set(self.units) | set(self.aliases)
        if self.parent is not None:
            choices |= set(self.parent.symbols())
        return tuple(difflib.get_close_matches(symbol, sorted(choices), n=n))

    def symbols(self) -> tuple[str, ...]:
        symbols = set(self.units) | set(self.aliases)
        if self.parent is not None:
            symbols |= set(self.parent.symbols())
        return tuple(sorted(symbols))

    def scoped(
        self,
        name: str,
        *,
        units: Mapping[str, Unit] | None = None,
        aliases: Mapping[str, str] | None = None,
    ) -> "UnitRegistry":
        """Create a scoped child registry without mutating this registry."""

        return UnitRegistry(
            name=name,
            units=dict(units or {}),
            aliases=dict(aliases or {}),
            parent=self,
        )

    def register(self, unit: Unit, *, aliases: tuple[str, ...] = ()) -> None:
        """Register a unit in a mutation-controlled registry."""

        if not self.mutable:
            raise UnitRegistryError(
                f"Registry {self.name!r} is immutable; create a scoped registry instead.",
                operation="unit-register",
                actual=self.name,
            )
        dict.__setitem__(self.units, unit.symbol, unit)
        for alias in aliases:
            dict.__setitem__(self.aliases, alias, unit.symbol)


__all__ = ["UnitRegistry"]
