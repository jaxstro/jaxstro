"""Reading cited scalars out of the registry, for code that ships.

Phases 1-3 of the registry programme were scoped "oracle by default": the
registry *checks* the implementation and does not feed it. This module is the
narrow, deliberate exception, and the reason is measured.

On 2026-08-02 Pauli et al. 2025 Eq. 5 was found dividing by
``jaxstro.constants.Z_SUN`` = 0.0134 (Asplund+2009 total metals) where the
paper's own denominator is the solar iron-group abundance 0.014 (Magg et al.
2022, PDF p. 2) -- a different value of a different quantity, making every rate
+3.84% high. The fix was then typed by hand into five more places, and one of
them was missed: the recipe metadata went on declaring 0.0134 while the channel
had moved to 0.014. A registry that holds the true number while source holds a
hand-copied twin has not removed the failure, only relocated it.

So: a *cited scalar* is read from the registry. That is all this does. It does
NOT execute generated code, and it does not move the settled decision that
"generated code is committed, never generated at import" -- ``lambdify`` uses
``exec`` and stays out of the import path. Reading a float from TOML is a
categorically cheaper thing than running generated source.

Costs, honestly:

* the registry TOML becomes load-bearing for ``import startrax.winds``. It
  ships -- ``[tool.hatch.build.targets.wheel] include = ["/src/startrax"]``
  and nothing in the exclude list touches ``*.toml``.
* a malformed registry becomes an ImportError rather than a test failure.
  That is the intended trade: a provenance system that silently half-works
  produces confident wrong answers.

Each source is parsed once per process and cached, so the cost is one TOML
read per paper actually used, not per call.

Like the rest of ``registry/``, this imports nothing from ``startrax``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .loader import default_registry_root, load_source
from .records import AnchorRecord, RegistryError, ScalarBinding, SourceBundle


@lru_cache(maxsize=None)
def _bundle(registry_root: str, bibkey: str) -> SourceBundle:
    """One parse per (root, paper) per process."""
    return load_source(Path(registry_root), bibkey)


def _root(registry_root: Path | None) -> str:
    return str(default_registry_root() if registry_root is None else registry_root)


def coefficient_table(
    bibkey: str,
    prefix: str,
    *,
    rows: int,
    columns: tuple[str, ...],
    registry_root: Path | None = None,
) -> list[list[float]]:
    """A dense (rows x columns) table assembled from per-cell records.

    Cell ids follow ``VAL-<prefix><NN>-<COLUMN>`` with ``NN`` 1-based and
    zero-padded to two digits; a cell the record does not register is 0.0
    (Hurley Appendix A/B registers only meaningful cells). Introduced by the
    2026-08-28 M2 cutover so ``foundations/coefficients.py`` BUILDS its
    matrices from the record instead of restating 455 hand-typed cells the
    record already held (diffed 690/690 at registration).
    """

    bundle = _bundle(_root(registry_root), bibkey)
    table: list[list[float]] = []
    found = 0
    for row in range(1, rows + 1):
        cells: list[float] = []
        for column in columns:
            record = bundle.coefficients.get(f"{prefix}{row:02d}-{column}")
            if record is None:
                cells.append(0.0)
            else:
                cells.append(float(record.value))
                found += 1
        table.append(cells)
    if found == 0:
        raise KeyError(f"no {prefix}* cells found in {bibkey}; wrong prefix or record")
    return table


def coefficient_value(
    bibkey: str, coefficient_id: str, *, registry_root: Path | None = None
) -> float:
    """One fitted coefficient, as a plain float.

    Raises rather than defaulting. A silent fallback here would reintroduce
    exactly the hand-copied number this exists to remove -- and would do it
    invisibly, which is worse than the original.
    """
    bundle = _bundle(_root(registry_root), bibkey)
    record = bundle.coefficients.get(coefficient_id)
    if record is None:
        available = ", ".join(sorted(bundle.coefficients)) or "(none)"
        raise RegistryError(
            f"{bibkey}: no coefficient {coefficient_id!r}. Available: {available}"
        )
    return record.value


def coefficient_sigma(
    bibkey: str, coefficient_id: str, *, registry_root: Path | None = None
) -> float:
    """A coefficient's published 1-sigma fit uncertainty.

    A different quantity from the relation's observed scatter, which belongs to
    the equation -- see :func:`equation_scatter_dex`. Conflating them overstates
    a prediction's precision by about a factor of five on Pauli's relation
    (sigma ~0.09 against 0.43 dex of scatter).
    """
    bundle = _bundle(_root(registry_root), bibkey)
    record = bundle.coefficients.get(coefficient_id)
    if record is None:
        raise RegistryError(f"{bibkey}: no coefficient {coefficient_id!r}")
    if record.sigma is None:
        raise RegistryError(
            f"{coefficient_id}: no published sigma. Do not substitute a guess; "
            "a fabricated uncertainty is worse than an absent one."
        )
    return record.sigma


def equation_scatter_dex(
    bibkey: str, equation_id: str, *, registry_root: Path | None = None
) -> float:
    """The relation's observed RMS scatter, in dex."""
    bundle = _bundle(_root(registry_root), bibkey)
    equation = bundle.equations.get(equation_id)
    if equation is None:
        raise RegistryError(f"{bibkey}: no equation {equation_id!r}")
    if equation.scatter_dex is None:
        raise RegistryError(f"{equation_id}: declares no scatter_dex")
    return equation.scatter_dex


def binding_value(
    bibkey: str, equation_id: str, symbol: str, *, registry_root: Path | None = None
) -> float:
    """A scalar an equation binds to one of its symbols.

    Distinct from a coefficient on purpose. ``Zsun`` is not a *fitted* number
    of Pauli's relation; it is the normalisation the relation was fitted
    AGAINST, supplied by a different paper (Magg et al. 2022). Substituting a
    code-wide preference for it changes the physics without refitting the
    exponent, which is the bug this module was written for.
    """
    bundle = _bundle(_root(registry_root), bibkey)
    equation = bundle.equations.get(equation_id)
    if equation is None:
        available = ", ".join(sorted(bundle.equations)) or "(none)"
        raise RegistryError(
            f"{bibkey}: no equation {equation_id!r}. Available: {available}"
        )
    binding = equation.symbol_bindings.get(symbol)
    if not isinstance(binding, ScalarBinding):
        raise RegistryError(
            f"{equation_id}: symbol {symbol!r} has no bound scalar value"
        )
    return binding.value


def validity_limit(
    bibkey: str,
    equation_id: str,
    axis: str,
    edge: str,
    *,
    registry_root: Path | None = None,
) -> float:
    """One edge of an equation's declared source domain.

    The same argument as ``binding_value``, one level out. ``records.py`` has
    parsed ``[equation.validity.limits.<axis>]`` since the schema existed, and
    until 2026-08-20 nothing could read it -- so every domain gate in the
    package was NECESSARILY hand-typed, and ~40% of them restated a bound the
    TOML already declared. ``kernels/agb.py`` typed ``teff >= 2200.0`` and
    ``<= 3000.0`` beside a record carrying exactly those numbers.

    That is this module's own thesis applied to bounds rather than
    coefficients: a registry that holds the true number while source holds a
    hand-copied twin has not removed the failure, only relocated it. A domain
    bound is as much a published quantity as a fitted exponent -- it is where
    the authors say their fit stops being true.

    ``axis`` is the limit name as declared (``"Z"``, ``"teff_k"``,
    ``"mass_msun"``); ``edge`` is ``"lower"`` or ``"upper"``. Fails closed on
    an undeclared axis or an open edge rather than returning a sentinel: an
    absent bound is not an infinite one.
    """
    if edge not in ("lower", "upper"):
        raise RegistryError(f"edge must be 'lower' or 'upper', got {edge!r}")
    bundle = _bundle(_root(registry_root), bibkey)
    equation = bundle.equations.get(equation_id)
    if equation is None:
        available = ", ".join(sorted(bundle.equations)) or "(none)"
        raise RegistryError(
            f"{bibkey}: no equation {equation_id!r}. Available: {available}"
        )
    if equation.validity is None:
        raise RegistryError(f"{equation_id}: declares no validity record")
    limit = equation.validity.limits.get(axis)
    if limit is None:
        declared = ", ".join(sorted(equation.validity.limits)) or "(none)"
        raise RegistryError(
            f"{equation_id}: declares no {axis!r} limit. Declared axes: {declared}"
        )
    value = getattr(limit, edge)
    if value is None:
        raise RegistryError(
            f"{equation_id}.{axis}: declares no {edge} bound (the interval is "
            "open on that side; an absent bound is not an infinite one)"
        )
    return float(value)


def validity_nodes(
    bibkey: str,
    equation_id: str,
    axis: str,
    *,
    registry_root: Path | None = None,
) -> tuple[float, ...]:
    """The DISCRETE grid an equation is published on, when it has one.

    The sibling of ``validity_limit`` for sources that deploy at named points
    rather than across a range. Fails closed when the axis declares only an
    interval: silently returning the endpoints would turn a two-node grid into
    a continuum, which is exactly the closure the caller must not make by
    accident.
    """

    bundle = _bundle(_root(registry_root), bibkey)
    equation = bundle.equations.get(equation_id)
    if equation is None:
        available = ", ".join(sorted(bundle.equations)) or "(none)"
        raise RegistryError(
            f"{bibkey}: no equation {equation_id!r}. Available: {available}"
        )
    if equation.validity is None:
        raise RegistryError(f"{equation_id}: declares no validity record")
    limit = equation.validity.limits.get(axis)
    if limit is None:
        declared = ", ".join(sorted(equation.validity.limits)) or "(none)"
        raise RegistryError(
            f"{equation_id}: declares no {axis!r} limit. Declared axes: {declared}"
        )
    if limit.nodes is None:
        raise RegistryError(
            f"{equation_id}.{axis}: declares an interval, not a grid. An "
            "interval's endpoints are not its nodes."
        )
    return limit.nodes


def anchor_table(
    bibkey: str, equation_id: str, *, registry_root: Path | None = None
) -> tuple[AnchorRecord, ...]:
    """The paper-derived values an equation is checked against.

    Anchors exist to catch a transcription error in the registry itself. They
    are also, for a DIGITISED table, the published data -- ``vink2002lbv``
    carries 72 anchor rows reproducing Fig. 4 in full, while ``winds/lbv_v1.py``
    hand-types the same 72 ordinates with no registry call at all. Exposing the
    rows lets the one owner feed the runtime instead of merely auditing it.

    Returns them in declaration order. Callers that need a specific traversal
    must key on each row's ``inputs``, never on position: the order is a
    property of the file, not of the physics.
    """
    bundle = _bundle(_root(registry_root), bibkey)
    equation = bundle.equations.get(equation_id)
    if equation is None:
        available = ", ".join(sorted(bundle.equations)) or "(none)"
        raise RegistryError(
            f"{bibkey}: no equation {equation_id!r}. Available: {available}"
        )
    return tuple(equation.anchors)
