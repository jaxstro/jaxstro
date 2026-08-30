"""The production-safe equation-registry data API.

An equation is otherwise written in three places that must agree by hand -- the
paper it came from, the JAX kernel implementing it, and the LaTeX on the page
describing it. Nothing forces the three to match, and this project has paid for
it. Here the equation is written once, with its source, and the rest is
printed.

The registry is an **oracle by default and a generator by opt-in**: registering
an equation adds a check without changing what ships, so a registry that turns
out wrong for some equation costs a red test rather than a regression.

One deliberate exception, added 2026-08-02: a **cited scalar** may be read by
shipping code through :mod:`startrax.registry.access`. Holding the true number
here while source holds a hand-copied twin does not remove the divergence
failure, it relocates it -- measured, when correcting Pauli's Z_sun by hand
across five sites missed a sixth.

The SymPy-backed oracle and documentation compiler live in
:mod:`startrax.registry.symbolic` and are deliberately *not* re-exported here.
Production code can therefore load registry data without importing symbolic
machinery. Tests and documentation must opt into that layer explicitly.

**This is the ecosystem home** (ADR-0010, 2026-08-29). startrax was the pilot and
its design is carried here **verbatim** -- the six modules were copied
byte-identically, because its own note required that the move to ``jaxstro`` be
a move rather than a rewrite. Two edits were made deliberately and are recorded
in that ADR: this paragraph, and :func:`~jaxstro.registry.default_registry_root`,
which now refuses rather than returning a data-less directory.

**jaxstro ships machinery, never data.** Every loader entry point takes a
``registry_root``; each consuming package owns the source bundles for the papers
*it* cites. hydrax is the first consumer built on this layer.

``startrax.registry`` is **frozen**: it keeps working, and schema changes land
here only. Its 109 import sites across 76 files migrate as their own task.
"""

from __future__ import annotations

from .access import binding_value, coefficient_value
from .loader import (
    ResolvedSource,
    available_bibkeys,
    default_registry_root,
    load_atlas_decisions,
    load_atlas_relations,
    load_derived_models,
    load_source,
    load_source_symbols,
    load_symbol_table,
    resolve,
    source_directory,
)
from .records import (
    AnchorRecord,
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

__all__ = [
    "AnchorRecord",
    "AtlasDecisionRecord",
    "AtlasRelationRecord",
    "CaveatRecord",
    "CoefficientRecord",
    "DerivedModelRecord",
    "EquationRecord",
    "RegistryError",
    "ResolvedSource",
    "SourceBundle",
    "SourceRecord",
    "SymbolRecord",
    "available_bibkeys",
    "binding_value",
    "coefficient_value",
    "default_registry_root",
    "load_atlas_decisions",
    "load_atlas_relations",
    "load_derived_models",
    "load_source",
    "load_source_symbols",
    "load_symbol_table",
    "resolve",
    "source_directory",
]
