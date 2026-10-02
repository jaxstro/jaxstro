---
title: Equation registry
---

# Equation registry

## Owner import path

`jaxstro.registry`

## Purpose

Record types, TOML loaders and checks for a package's equation provenance.
jaxstro ships the machinery and no data: every loader takes a `registry_root`,
and each consuming package owns the root for the papers it cites (hydrax
ADR-0010). A root holds one bundle per source and, since 2026-10-02, four
registry-level tables that join the bundles to the package's code, gates,
methods and documentation claims.

## Public records and callables

Source bundles: `SourceRecord`, `EquationRecord`, `CoefficientRecord`,
`CaveatRecord`, `SymbolRecord`, `AnchorRecord`, `SourceBundle`, `load_source`,
`load_symbol_table`, `load_source_symbols`, `available_bibkeys`,
`source_directory`, `resolve`, `ResolvedSource`, `coefficient_value`,
`binding_value`, `default_registry_root` (which refuses; there is no default
root). Startrax Atlas tables: `AtlasDecisionRecord`, `AtlasRelationRecord`,
`DerivedModelRecord` and their loaders.

Registry-level tables: `ImplementationRecord`, `EvidenceRecord`,
`MethodRecord`, `ClaimRecord`, `load_implementations`, `load_evidence`,
`load_methods`, `load_claims`. Whole-root loading and checks: `Registry`,
`load_registry`, `unresolved_implementations`, `missing_files`. Bibliography:
`bibtex_entry`, `bibliography`. The SymPy oracle lives in
`jaxstro.registry.symbolic` and is not re-exported.

### Layout of a registry root

```text
<registry_root>/
  symbols.toml              shared physical symbols ([[symbol]])
  implementations.toml      [[implementation]]   optional
  evidence.toml             [[evidence]]         optional
  methods.toml              [[method]]           optional
  claims.toml               [[claim]]            optional
  sources/<bibkey>/
    source.toml             [source]
    equations.toml          [[equation]]         optional
    coefficients.toml       [[coefficient]]      optional
    caveats.toml            [[caveat]]           optional
    symbols.toml            source-local symbols, optional
```

A source the package cites but registers no equation from is a bundle with only
`source.toml`. Any other `.toml` file at the root is refused, so a misspelled
table name cannot pass as an absent one.

### Registry-level tables

| table | required fields | optional fields | `status` values |
|---|---|---|---|
| `implementation` | `id`, `name`, `status`, `module` (dotted), `symbol` (identifier) | `evidence_ids`, `note` | `planned`, `implemented`, `verified`, `retired` |
| `evidence` | `id`, `name`, `status` | `equation_ids`, `method_ids`, `gates`, `criterion`, `result`, `artifacts`, `note` | `planned`, `passing`, `failing`, `retired`, `refuted` |
| `method` | `id`, `name`, `status`, `formulation` | `equation_ids`, `source_ids`, `note` | `planned`, `implemented`, `verified`, `retired` |
| `claim` | `id`, `statement`, `status` | `evidence_ids` | `substantiated`, `unsubstantiated`, `needs-source-verification` |

An evidence row with status `passing` or `failing` must state `gates` (test
files or node ids, `tests/x/test_y.py::test_z`), `criterion` (the pass condition
fixed before the run) and `result` (what the run measured). `gates` and
`artifacts` are normalised repository-relative paths. A `substantiated` claim
needs `evidence_ids`. The equation-to-implementation link is held once, by the
equation's `implementation_ids`.

### Bibliographic fields

`source.toml` may carry `bib_type` (`article`, `book`, `incollection`,
`inproceedings`, `misc`, `phdthesis`, `techreport`, `unpublished`), `authors`,
`title`, `journal`, `volume`, `pages`, `year` and `arxiv`, all strings, beside
the existing `doi`. They are all-or-nothing: a source that declares any of them
must declare `bib_type`, `authors`, `title` and `year`. `bibliography(root)`
renders every source that declares them, keyed by its bibkey, so a `{cite}`
key is the bundle directory name.

## Shape and dtype expectations

Records are frozen dataclasses of strings, tuples of strings and floats parsed
from TOML. Nothing here is an array.

## JAX transforms and AD classification

Host-side provenance data. No JAX transform applies and there is no AD claim.
Production code reads cited scalars through `coefficient_value`,
`binding_value` and `jaxstro.registry.access.validity_limit` at import time,
outside traced code.

## Failure behavior

Every loader fails closed with `RegistryError`: a missing required field, an
unknown key, a status outside its vocabulary, a duplicate id or an empty table.
`load_registry` additionally refuses an equation id registered in two bundles
and every reference that resolves to nothing, and lists all of them in one
error:

- equation `source_ids`, coefficient `source_ids` and scalar-binding
  `source_ids` to a bundle;
- equation `implementation_ids` to an implementation row;
- implementation `evidence_ids` and claim `evidence_ids` to an evidence row;
- evidence and method `equation_ids` to an equation in any bundle;
- evidence `method_ids` to a method row, method `source_ids` to a bundle;
- a `substantiated` claim to at least one `passing` evidence row.

`unresolved_implementations` and `missing_files` depend on the environment, so
they return their failures rather than raise: live implementations
(`implemented`, `verified`) must import, and the gate files and artifacts of
live evidence rows must exist under the given repository root.

## Contract and evidence links

`tests/unit/test_registry_tables.py` covers each loader, the link check and the
BibTeX rendering against the fixture root in
`tests/unit/fixtures/registry_provenance/`. Runtime artifact manifests are a
different evidence class; see [](./provenance.md).

## Canonical import example

```python
from pathlib import Path

from jaxstro.registry import load_registry, missing_files, unresolved_implementations

registry = load_registry(Path("src/mypackage/registry"))
problems = unresolved_implementations(registry) + missing_files(registry, Path("."))
```
