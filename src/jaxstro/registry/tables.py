"""Registry-level provenance tables: implementations, evidence, methods, claims.

A source bundle records what one paper wrote. These four tables record what a
package did with it: the code that realises an equation (``implementations.toml``),
the gate that measured it (``evidence.toml``), the numerical scheme that combines
several equations (``methods.toml``) and the statements its documentation makes
(``claims.toml``). They sit at the registry root beside ``symbols.toml`` because
they are joins across bundles, not properties of one source.

Added 2026-10-02 for hydrax (ADR-0010), whose 21 implementation, 18 evidence and
10 method rows lived in YAML beside the registry with a hand-written join checker,
because this schema had no record type for them. Each record follows the bundle
records' rules: frozen, every key known, every status from a closed vocabulary.
The cross-table references are resolved by :mod:`jaxstro.registry.links`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .records import (
    RegistryError,
    _check_member,
    _optional_string,
    _reject_unknown_keys,
    _require,
    _string_tuple,
)

#: An implementation or method row. ``retired`` rows are history and may name
#: code that no longer exists; ``planned`` rows name code that does not exist yet.
IMPLEMENTATION_STATUSES = frozenset({"planned", "implemented", "verified", "retired"})
METHOD_STATUSES = IMPLEMENTATION_STATUSES

#: An evidence row. Only ``passing`` and ``failing`` claim that a gate runs now,
#: so only those must name one. ``refuted`` records a measurement that disproved
#: what it was built to show; it is kept because the refutation is a result.
EVIDENCE_STATUSES = frozenset({"planned", "passing", "failing", "retired", "refuted"})
LIVE_EVIDENCE_STATUSES = frozenset({"passing", "failing"})
LIVE_IMPLEMENTATION_STATUSES = frozenset({"implemented", "verified"})

#: A documentation claim. ``substantiated`` requires evidence; the link check
#: further requires at least one cited row to be ``passing``.
CLAIM_STATUSES = frozenset(
    {"substantiated", "unsubstantiated", "needs-source-verification"}
)


def _required_text(payload: dict[str, Any], key: str, *, where: str) -> str:
    value = _require(payload, key, where)
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{where}: {key} must be a non-empty string")
    return value


def _dotted_name(value: str, *, key: str, where: str) -> str:
    if not all(part.isidentifier() for part in value.split(".")):
        raise RegistryError(f"{where}: {key}={value!r} is not a dotted Python name")
    return value


def _relative_path(value: str, *, key: str, where: str) -> str:
    """A repository-relative POSIX path that cannot leave the repository."""
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != str(path):
        raise RegistryError(
            f"{where}: {key} entry {value!r} must be a normalised relative path"
        )
    return value


def _test_node(value: str, *, where: str) -> str:
    """``tests/x/test_y.py`` or ``tests/x/test_y.py::test_name[...]``."""
    file_part, *nodes = value.split("::")
    _relative_path(file_part, key="gates", where=where)
    if not file_part.endswith(".py"):
        raise RegistryError(
            f"{where}: gate {value!r} must name a .py test file, optionally ::node"
        )
    if any(not node for node in nodes):
        raise RegistryError(f"{where}: gate {value!r} has an empty node id")
    return value


@dataclass(frozen=True)
class ImplementationRecord:
    """The code that realises one or more registered equations.

    The equation -> implementation link is owned by the equation's
    ``implementation_ids``; this row does not repeat it, because a link stored in
    both directions disagreed with itself in four of hydrax's rows (2026-10-02).
    """

    id: str
    name: str
    status: str
    module: str
    symbol: str
    evidence_ids: tuple[str, ...] = ()
    note: str | None = None

    _KEYS = frozenset(
        {"id", "name", "status", "module", "symbol", "evidence_ids", "note"}
    )

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> ImplementationRecord:
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        symbol = _required_text(payload, "symbol", where=where)
        if not symbol.isidentifier():
            raise RegistryError(f"{where}: symbol={symbol!r} is not an identifier")
        return cls(
            id=_required_text(payload, "id", where=where),
            name=_required_text(payload, "name", where=where),
            status=_check_member(
                _require(payload, "status", where),
                IMPLEMENTATION_STATUSES,
                "status",
                where,
            ),
            module=_dotted_name(
                _required_text(payload, "module", where=where),
                key="module",
                where=where,
            ),
            symbol=symbol,
            evidence_ids=_string_tuple(payload, "evidence_ids", where=where),
            note=_optional_string(payload, "note", where=where),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    """One measurement that substantiates equations or methods.

    ``criterion`` is the pass condition frozen before the run; ``result`` is what
    the run measured; ``gates`` are the test files or node ids that reproduce it.
    A live row (``passing`` or ``failing``) must carry all three: a gate with no
    stated criterion cannot be told apart from a tolerance chosen after the fact.
    """

    id: str
    name: str
    status: str
    equation_ids: tuple[str, ...] = ()
    method_ids: tuple[str, ...] = ()
    gates: tuple[str, ...] = ()
    criterion: str | None = None
    result: str | None = None
    artifacts: tuple[str, ...] = ()
    note: str | None = None

    _KEYS = frozenset(
        {
            "id",
            "name",
            "status",
            "equation_ids",
            "method_ids",
            "gates",
            "criterion",
            "result",
            "artifacts",
            "note",
        }
    )

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> EvidenceRecord:
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        record = cls(
            id=_required_text(payload, "id", where=where),
            name=_required_text(payload, "name", where=where),
            status=_check_member(
                _require(payload, "status", where), EVIDENCE_STATUSES, "status", where
            ),
            equation_ids=_string_tuple(payload, "equation_ids", where=where),
            method_ids=_string_tuple(payload, "method_ids", where=where),
            gates=tuple(
                _test_node(gate, where=where)
                for gate in _string_tuple(payload, "gates", where=where)
            ),
            criterion=_optional_string(payload, "criterion", where=where),
            result=_optional_string(payload, "result", where=where),
            artifacts=tuple(
                _relative_path(path, key="artifacts", where=where)
                for path in _string_tuple(payload, "artifacts", where=where)
            ),
            note=_optional_string(payload, "note", where=where),
        )
        if record.status in LIVE_EVIDENCE_STATUSES:
            missing = [
                key
                for key, value in (
                    ("gates", record.gates),
                    ("criterion", record.criterion),
                    ("result", record.result),
                )
                if not value
            ]
            if missing:
                raise RegistryError(
                    f"{where}: a {record.status!r} evidence row needs {missing}"
                )
        return record


@dataclass(frozen=True)
class MethodRecord:
    """A named numerical scheme: the equations it discretises and its sources.

    ``formulation`` is the package's own label for the family the scheme
    belongs to (hydrax: ``godunov-ale``, ``shared``); it is free text because the
    vocabulary is domain-specific.
    """

    id: str
    name: str
    status: str
    formulation: str
    equation_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    note: str | None = None

    _KEYS = frozenset(
        {"id", "name", "status", "formulation", "equation_ids", "source_ids", "note"}
    )

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> MethodRecord:
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        return cls(
            id=_required_text(payload, "id", where=where),
            name=_required_text(payload, "name", where=where),
            status=_check_member(
                _require(payload, "status", where), METHOD_STATUSES, "status", where
            ),
            formulation=_required_text(payload, "formulation", where=where),
            equation_ids=_string_tuple(payload, "equation_ids", where=where),
            source_ids=_string_tuple(payload, "source_ids", where=where),
            note=_optional_string(payload, "note", where=where),
        )


@dataclass(frozen=True)
class ClaimRecord:
    """A statement the documentation makes, and the evidence behind it."""

    id: str
    statement: str
    status: str
    evidence_ids: tuple[str, ...] = ()

    _KEYS = frozenset({"id", "statement", "status", "evidence_ids"})

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> ClaimRecord:
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        record = cls(
            id=_required_text(payload, "id", where=where),
            statement=_required_text(payload, "statement", where=where),
            status=_check_member(
                _require(payload, "status", where), CLAIM_STATUSES, "status", where
            ),
            evidence_ids=_string_tuple(payload, "evidence_ids", where=where),
        )
        if record.status == "substantiated" and not record.evidence_ids:
            raise RegistryError(f"{where}: a substantiated claim needs evidence_ids")
        return record
