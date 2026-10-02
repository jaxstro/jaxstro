"""Load a whole registry root and resolve every reference between its records.

:func:`~jaxstro.registry.load_source` checks one bundle against itself: its
equations must cite it, their symbols and coefficients must be declared in it.
It does not resolve an id that points outside the bundle, and it cannot see a
duplicate equation id in a second bundle. This module does both, across the
bundles and the registry-level tables of :mod:`jaxstro.registry.tables`.

Two kinds of check, with different failure contracts:

* **Data joins** (:func:`load_registry`) raise. A dangling id is wrong whatever
  the environment, and a provenance graph that loads with one is the failure
  hydrax measured on 2026-08-28: five equations cited an evidence id that existed
  in no file, green for months.
* **Environment checks** (:func:`unresolved_implementations`,
  :func:`missing_files`) return their failures. Whether a module imports or a
  file exists depends on where the check runs, so the caller chooses when to
  run them and how to report them.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from .loader import (
    ATLAS_DECISIONS_FILENAME,
    ATLAS_RELATIONS_FILENAME,
    CLAIMS_FILENAME,
    DERIVED_MODELS_FILENAME,
    EVIDENCE_FILENAME,
    IMPLEMENTATIONS_FILENAME,
    METHODS_FILENAME,
    SYMBOLS_FILENAME,
    available_bibkeys,
    load_claims,
    load_evidence,
    load_implementations,
    load_methods,
    load_source,
)
from .records import EquationRecord, RegistryError, ScalarBinding, SourceBundle
from .tables import (
    LIVE_EVIDENCE_STATUSES,
    LIVE_IMPLEMENTATION_STATUSES,
    ClaimRecord,
    EvidenceRecord,
    ImplementationRecord,
    MethodRecord,
)

#: Every file the loaders read at a registry root. Any other ``.toml`` there is
#: refused: a misspelled ``implementation.toml`` would otherwise be skipped as an
#: absent optional table.
ROOT_FILENAMES = frozenset(
    {
        SYMBOLS_FILENAME,
        ATLAS_DECISIONS_FILENAME,
        ATLAS_RELATIONS_FILENAME,
        DERIVED_MODELS_FILENAME,
        IMPLEMENTATIONS_FILENAME,
        EVIDENCE_FILENAME,
        METHODS_FILENAME,
        CLAIMS_FILENAME,
    }
)


@dataclass(frozen=True)
class Registry:
    """Every bundle and table under one registry root, with all ids resolved.

    ``equations`` is the union of the bundles' equations, keyed by id, and
    ``equation_bundles`` names the bundle each one lives in. Constructed by
    :func:`load_registry`, which guarantees the union has no duplicate.
    """

    root: Path
    bundles: dict[str, SourceBundle]
    equations: dict[str, EquationRecord]
    equation_bundles: dict[str, str]
    implementations: dict[str, ImplementationRecord]
    evidence: dict[str, EvidenceRecord]
    methods: dict[str, MethodRecord]
    claims: dict[str, ClaimRecord]


_Row = TypeVar("_Row")


def _optional_table(
    root: Path, filename: str, loader: Callable[[Path], dict[str, _Row]]
) -> dict[str, _Row]:
    return loader(root) if (root / filename).exists() else {}


def load_registry(registry_root: Path) -> Registry:
    """Load every bundle and table under ``registry_root`` and resolve all links.

    Each of ``implementations.toml``, ``evidence.toml``, ``methods.toml`` and
    ``claims.toml`` is optional as a file; a reference into an absent table
    dangles and is refused like any other. Raises :class:`RegistryError` listing
    every failure found, not only the first.
    """
    root = Path(registry_root)
    stray = sorted(
        path.name for path in root.glob("*.toml") if path.name not in ROOT_FILENAMES
    )
    if stray:
        raise RegistryError(
            f"{root}: unrecognised registry files {stray}; "
            f"known files are {sorted(ROOT_FILENAMES)}"
        )
    bibkeys = available_bibkeys(root)
    if not bibkeys:
        raise RegistryError(f"{root}: no source bundles under sources/")
    bundles = {bibkey: load_source(root, bibkey) for bibkey in bibkeys}

    equations: dict[str, EquationRecord] = {}
    equation_bundles: dict[str, str] = {}
    duplicates: list[str] = []
    for bibkey, bundle in bundles.items():
        for equation_id, equation in bundle.equations.items():
            if equation_id in equation_bundles:
                duplicates.append(
                    f"{equation_id}: registered in both "
                    f"{equation_bundles[equation_id]} and {bibkey}"
                )
                continue
            equations[equation_id] = equation
            equation_bundles[equation_id] = bibkey
    if duplicates:
        raise RegistryError(
            "an equation id must be unique across a registry root:\n  - "
            + "\n  - ".join(duplicates)
        )

    registry = Registry(
        root=root,
        bundles=bundles,
        equations=equations,
        equation_bundles=equation_bundles,
        implementations=_optional_table(
            root, IMPLEMENTATIONS_FILENAME, load_implementations
        ),
        evidence=_optional_table(root, EVIDENCE_FILENAME, load_evidence),
        methods=_optional_table(root, METHODS_FILENAME, load_methods),
        claims=_optional_table(root, CLAIMS_FILENAME, load_claims),
    )
    failures = _link_failures(registry)
    if failures:
        raise RegistryError(
            f"{root}: {len(failures)} unresolved links:\n  - " + "\n  - ".join(failures)
        )
    return registry


def _dangling(
    owner: str,
    field: str,
    ids: tuple[str, ...],
    known: Mapping[str, object],
    where: str,
) -> list[str]:
    return [
        f"{owner}: {field} {item!r} names no {where}"
        for item in ids
        if item not in known
    ]


def _link_failures(registry: Registry) -> list[str]:
    bundles, equations = registry.bundles, registry.equations
    implementations, evidence = registry.implementations, registry.evidence
    failures: list[str] = []
    for bibkey, bundle in registry.bundles.items():
        for equation in bundle.equations.values():
            failures += _dangling(
                equation.id, "source_id", equation.source_ids, bundles, "source bundle"
            )
            failures += _dangling(
                equation.id,
                "implementation_id",
                equation.implementation_ids,
                implementations,
                "implementation row",
            )
            for symbol, binding in equation.symbol_bindings.items():
                if isinstance(binding, ScalarBinding):
                    failures += _dangling(
                        f"{equation.id} binding {symbol}",
                        "source_id",
                        binding.source_ids,
                        bundles,
                        "source bundle",
                    )
        for coefficient in bundle.coefficients.values():
            failures += _dangling(
                f"{bibkey} {coefficient.id}",
                "source_id",
                coefficient.source_ids,
                bundles,
                "source bundle",
            )
    for implementation in implementations.values():
        failures += _dangling(
            implementation.id,
            "evidence_id",
            implementation.evidence_ids,
            evidence,
            "evidence row",
        )
    for gate in evidence.values():
        failures += _dangling(
            gate.id, "equation_id", gate.equation_ids, equations, "equation"
        )
        failures += _dangling(
            gate.id, "method_id", gate.method_ids, registry.methods, "method row"
        )
    for method in registry.methods.values():
        failures += _dangling(
            method.id, "equation_id", method.equation_ids, equations, "equation"
        )
        failures += _dangling(
            method.id, "source_id", method.source_ids, bundles, "source bundle"
        )
    for claim in registry.claims.values():
        failures += _dangling(
            claim.id, "evidence_id", claim.evidence_ids, evidence, "evidence row"
        )
        if claim.status == "substantiated" and not any(
            evidence[item].status == "passing"
            for item in claim.evidence_ids
            if item in evidence
        ):
            failures.append(
                f"{claim.id}: substantiated, but none of its evidence rows is passing"
            )
    return failures


def unresolved_implementations(registry: Registry) -> tuple[str, ...]:
    """Live implementation rows whose ``module.symbol`` does not import.

    Imports each named module. ``retired`` rows are history and may name deleted
    code; ``planned`` rows name code that does not exist yet; neither is checked.
    """
    failures: list[str] = []
    for row in registry.implementations.values():
        if row.status not in LIVE_IMPLEMENTATION_STATUSES:
            continue
        try:
            module = importlib.import_module(row.module)
        except Exception as error:  # noqa: BLE001 - reported to the caller
            failures.append(f"{row.id}: module {row.module} does not import ({error})")
            continue
        if not hasattr(module, row.symbol):
            failures.append(f"{row.id}: {row.module} has no {row.symbol}")
    return tuple(failures)


def missing_files(registry: Registry, repo_root: Path) -> tuple[str, ...]:
    """Files a live evidence row names that do not exist under ``repo_root``.

    Checks the file part of every gate and every artifact of each ``passing`` or
    ``failing`` row. A node id after ``::`` is not resolved: that needs pytest
    collection, which the consumer's own suite already performs.
    """
    root = Path(repo_root)
    failures: list[str] = []
    for row in registry.evidence.values():
        if row.status not in LIVE_EVIDENCE_STATUSES:
            continue
        paths = [gate.partition("::")[0] for gate in row.gates] + list(row.artifacts)
        failures += [
            f"{row.id}: {path} does not exist"
            for path in paths
            if not (root / path).exists()
        ]
    return tuple(failures)
