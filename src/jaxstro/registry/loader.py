"""TOML loader and bibkey resolver for the equation registry.

TOML, not YAML: ``yaml.safe_load`` parses ``1e-13`` as the *string* ``'1e-13'``
and ``no`` as ``False``. For a registry whose content is physical coefficients
that is disqualifying. ``tomllib`` is stdlib from Python 3.11, so this costs no
dependency.

Every entry point takes a ``registry_root`` argument. Nothing here knows it
lives inside startrax.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .records import (
    AtlasDecisionRecord,
    AtlasRelationRecord,
    CaveatRecord,
    CoefficientRecord,
    DerivedModelRecord,
    EquationRecord,
    RegistryError,
    SourceBundle,
    SourceRecord,
    SymbolRecord,
)

SOURCES_DIRNAME = "sources"
SYMBOLS_FILENAME = "symbols.toml"
ATLAS_DECISIONS_FILENAME = "atlas_decisions.toml"
ATLAS_RELATIONS_FILENAME = "atlas_relations.toml"
DERIVED_MODELS_FILENAME = "derived_models.toml"


def default_registry_root() -> Path:
    """Refuses. **jaxstro ships registry machinery, never registry data.**

    In startrax this returned the directory the module ships in, because the
    machinery and that package's science data lived together. Here they do not:
    every consuming package owns the source bundles for the papers it cites, and
    every other entry point already takes a ``registry_root``, so there is
    nothing sensible to default to.

    Returning ``Path(__file__).parent`` would hand back a directory containing no
    ``sources/`` at all, and the failure would surface later as a confusing
    "no registry source" error naming a path the caller never chose. Refusing
    here names the real mistake at the point it is made.
    """
    raise RegistryError(
        "jaxstro.registry ships machinery, not data: there is no default root. "
        "Pass the registry_root of the package whose sources you want -- e.g. "
        "hydrax.registry.registry_root()."
    )


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RegistryError(f"registry file not found: {path}")
    with path.open("rb") as handle:
        try:
            return tomllib.load(handle)
        except tomllib.TOMLDecodeError as error:  # pragma: no cover - malformed input
            raise RegistryError(f"{path}: malformed TOML -- {error}") from error


def _read_symbols(path: Path, *, label: str) -> dict[str, SymbolRecord]:
    payload = _read_toml(path)
    table: dict[str, SymbolRecord] = {}
    for index, item in enumerate(payload.get("symbol", ())):
        record = SymbolRecord.from_toml(item, where=f"{label} symbol[{index}]")
        if record.id in table:
            raise RegistryError(f"{path}: duplicate symbol id {record.id!r}")
        table[record.id] = record
    return table


def load_symbol_table(registry_root: Path) -> dict[str, SymbolRecord]:
    """The shared symbol table.

    Symbols here are *physical quantities*, not properties of a paper: if two
    papers both use ``L`` they must mean the same thing, or a dimensional check
    across them means nothing.
    """
    path = Path(registry_root) / SYMBOLS_FILENAME
    table = _read_symbols(path, label=path.name)
    if not table:
        raise RegistryError(f"{path}: declares no symbols")
    return table


def load_atlas_decisions(registry_root: Path) -> dict[str, AtlasDecisionRecord]:
    """Load researcher-authorized non-source Atlas choices fail-closed."""

    path = Path(registry_root) / ATLAS_DECISIONS_FILENAME
    payload = _read_toml(path)
    records: dict[str, AtlasDecisionRecord] = {}
    for index, item in enumerate(payload.get("decision", ())):
        record = AtlasDecisionRecord.from_toml(
            item, where=f"{ATLAS_DECISIONS_FILENAME} decision[{index}]"
        )
        if record.id in records:
            raise RegistryError(
                f"{ATLAS_DECISIONS_FILENAME}: duplicate decision id {record.id!r}"
            )
        records[record.id] = record
    if not records:
        raise RegistryError(f"{ATLAS_DECISIONS_FILENAME}: declares no decisions")
    return records


def load_atlas_relations(registry_root: Path) -> dict[str, AtlasRelationRecord]:
    """Load the fail-closed relation-to-oracle ownership catalog."""
    payload = _read_toml(Path(registry_root) / ATLAS_RELATIONS_FILENAME)
    records: dict[str, AtlasRelationRecord] = {}
    for index, item in enumerate(payload.get("relation", ())):
        record = AtlasRelationRecord.from_toml(
            item, where=f"{ATLAS_RELATIONS_FILENAME} relation[{index}]"
        )
        if record.relation_id in records:
            raise RegistryError(
                f"{ATLAS_RELATIONS_FILENAME}: duplicate relation {record.relation_id!r}"
            )
        records[record.relation_id] = record
    if not records:
        raise RegistryError(f"{ATLAS_RELATIONS_FILENAME}: declares no relations")
    return records


def load_derived_models(registry_root: Path) -> dict[str, DerivedModelRecord]:
    """Load project-authored model authorities fail-closed."""

    payload = _read_toml(Path(registry_root) / DERIVED_MODELS_FILENAME)
    records: dict[str, DerivedModelRecord] = {}
    for index, item in enumerate(payload.get("derived_model", ())):
        record = DerivedModelRecord.from_toml(
            item, where=f"{DERIVED_MODELS_FILENAME} derived_model[{index}]"
        )
        if record.id in records:
            raise RegistryError(
                f"{DERIVED_MODELS_FILENAME}: duplicate derived-model id {record.id!r}"
            )
        records[record.id] = record
    if not records:
        raise RegistryError(f"{DERIVED_MODELS_FILENAME}: declares no derived models")
    return records


def load_source_symbols(registry_root: Path, bibkey: str) -> dict[str, SymbolRecord]:
    """Symbols belonging to one paper and no other.

    Optional, and usually absent. It exists because some papers define
    intermediate quantities that are neither shared physical quantities nor
    fitted coefficients -- Sander & Vink 2020's ``Gamma_e,b`` is the output of
    another equation in the same paper. Those must not enter the registry-wide
    table, whose value is precisely that a name means one thing everywhere.

    A source-local symbol may not redeclare a shared name: shadowing the
    meaning of ``L`` for one paper is the exact failure the shared table
    prevents.
    """
    path = source_directory(registry_root, bibkey) / SYMBOLS_FILENAME
    if not path.exists():
        return {}
    return _read_symbols(path, label=f"{bibkey}/{SYMBOLS_FILENAME}")


def source_directory(registry_root: Path, bibkey: str) -> Path:
    """Where one paper's records live. The bibkey is the join key."""
    if not bibkey or "/" in bibkey or bibkey.startswith("."):
        raise RegistryError(f"invalid bibkey {bibkey!r}")
    return Path(registry_root) / SOURCES_DIRNAME / bibkey


def load_source(registry_root: Path, bibkey: str) -> SourceBundle:
    """Load one paper's directory into records.

    Raises on anything malformed. A provenance system that silently half-works
    produces confident wrong answers, which is worse than having none.
    """
    directory = source_directory(registry_root, bibkey)
    if not directory.is_dir():
        raise RegistryError(
            f"no registry source {bibkey!r} under {Path(registry_root) / SOURCES_DIRNAME}"
        )

    source_payload = _read_toml(directory / "source.toml")
    source = SourceRecord.from_toml(
        source_payload.get("source", {}), where=f"{bibkey}/source.toml"
    )
    if source.id != bibkey:
        raise RegistryError(
            f"{bibkey}/source.toml: id={source.id!r} does not match its directory name"
        )
    if not (source.pdf_sha256 or source.pdf_unavailable):
        raise RegistryError(
            f"{bibkey}/source.toml: needs a pdf_sha256 or an explicit "
            "pdf_unavailable reason -- a silent gap is how three owners ended up "
            "citing a design document instead of a paper"
        )

    coefficients: dict[str, CoefficientRecord] = {}
    coefficients_path = directory / "coefficients.toml"
    if coefficients_path.exists():
        payload = _read_toml(coefficients_path)
        for index, item in enumerate(payload.get("coefficient", ())):
            coefficient = CoefficientRecord.from_toml(
                item, where=f"{bibkey}/coefficients.toml coefficient[{index}]"
            )
            if coefficient.id in coefficients:
                raise RegistryError(
                    f"{bibkey}: duplicate coefficient id {coefficient.id!r}"
                )
            coefficients[coefficient.id] = coefficient

    caveats: list[CaveatRecord] = []
    caveats_path = directory / "caveats.toml"
    if caveats_path.exists():
        payload = _read_toml(caveats_path)
        caveats = [
            CaveatRecord.from_toml(item, where=f"{bibkey}/caveats.toml caveat[{index}]")
            for index, item in enumerate(payload.get("caveat", ()))
        ]

    equations: dict[str, EquationRecord] = {}
    equations_path = directory / "equations.toml"
    if equations_path.exists():
        payload = _read_toml(equations_path)
        for index, item in enumerate(payload.get("equation", ())):
            equation = EquationRecord.from_toml(
                item, where=f"{bibkey}/equations.toml equation[{index}]"
            )
            if equation.id in equations:
                raise RegistryError(f"{bibkey}: duplicate equation id {equation.id!r}")
            if bibkey not in equation.source_ids:
                raise RegistryError(
                    f"{equation.id}: lives in {bibkey}/ but does not cite it in source_ids"
                )
            equations[equation.id] = equation

    shared = load_symbol_table(registry_root)
    local = load_source_symbols(registry_root, bibkey)
    shadowed = {symbol.name for symbol in local.values()} & {
        symbol.name for symbol in shared.values()
    }
    if shadowed:
        raise RegistryError(
            f"{bibkey}/{SYMBOLS_FILENAME}: {sorted(shadowed)} redeclare shared "
            "symbol names -- a name must mean one thing across the registry"
        )
    symbols = {**shared, **local}

    bundle = SourceBundle(
        source=source,
        equations=equations,
        coefficients=coefficients,
        caveats=tuple(caveats),
        symbols=symbols,
    )
    # fail at load, not at first use
    derived_equation_ids = {
        derived_id
        for equation in equations.values()
        for derived_id in equation.derived_symbols.values()
    }
    for equation in equations.values():
        bundle.symbols_for(equation)
        bundle.coefficients_for(equation)
        bundle.derived_equations_for(equation)
        if source.verification == "researcher-verified" and equation.implementation_ids:
            if not source.source_note_ref:
                raise RegistryError(
                    f"{bibkey}/source.toml: researcher-verified implemented source "
                    "needs source_note_ref"
                )
            if not source.source_version:
                raise RegistryError(
                    f"{bibkey}/source.toml: researcher-verified implemented source "
                    "needs source_version"
                )
            if equation.id not in derived_equation_ids and equation.validity is None:
                raise RegistryError(
                    f"{equation.id}: researcher-verified implemented equation needs validity"
                )
    return bundle


def available_bibkeys(registry_root: Path) -> tuple[str, ...]:
    """Every source directory in this registry, sorted."""
    sources = Path(registry_root) / SOURCES_DIRNAME
    if not sources.is_dir():
        return ()
    return tuple(
        sorted(
            entry.name
            for entry in sources.iterdir()
            if entry.is_dir() and not entry.name.startswith((".", "_"))
        )
    )


@dataclass(frozen=True)
class ResolvedSource:
    """What one bibkey resolves to, for a human or an agent asking about a paper.

    The point is that the researcher and two different coding assistants stop
    each having to *find* the paper, and possibly landing in different places.
    One key, one lookup.
    """

    bibkey: str
    bundle: SourceBundle
    registry_directory: Path
    pdf_path: Path | None
    pdf_status: str
    source_note_path: Path | None


def resolve(
    registry_root: Path,
    bibkey: str,
    *,
    library_root: Path | None = None,
    notes_root: Path | None = None,
) -> ResolvedSource:
    """Resolve a bibkey to its records and, if given a library, its PDF.

    ``library_root`` and ``notes_root`` are arguments rather than constants:
    the PDF library is external to any package (in this project it is the
    brain, which is read-only from here), and a registry that hard-coded a path
    to it would not survive being moved to another package.
    """
    bundle = load_source(registry_root, bibkey)
    directory = source_directory(registry_root, bibkey)

    pdf_path: Path | None = None
    if bundle.source.pdf_unavailable:
        pdf_status = "unavailable"
    elif library_root is None:
        pdf_status = "not-searched"
    else:
        candidate = Path(library_root) / f"{bibkey}.pdf"
        if candidate.exists():
            pdf_path, pdf_status = candidate, "found"
        else:
            pdf_status = "missing"

    note_path: Path | None = None
    if notes_root is not None:
        candidate = Path(notes_root) / f"{bibkey}.md"
        if candidate.exists():
            note_path = candidate

    return ResolvedSource(
        bibkey=bibkey,
        bundle=bundle,
        registry_directory=directory,
        pdf_path=pdf_path,
        pdf_status=pdf_status,
        source_note_path=note_path,
    )
