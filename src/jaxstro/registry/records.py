"""Record types for the equation registry.

Stdlib-only, because a record is data: which paper, which page, which
coefficient, which caveat. The symbolic layer that turns an ``expression``
string into sympy lives in ``symbolic.py``; the split is organisation, not a
dependency boundary.

Field vocabulary follows glassbox's claim schema (``id``, ``source_ids``,
``verification``, ``implementation_ids``) so that adopting glassbox later means
pointing a tool at files that already exist rather than migrating.

This module imports nothing from ``startrax``: the machinery is
package-agnostic by construction so promoting it to ``jaxstro`` is a move, not
a rewrite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# controlled vocabularies
# ---------------------------------------------------------------------------

#: How a record's content was established. The *labour* of verification is
#: separate from the *authority* of approval: an assistant may reach
#: ``agent-verified`` and no further. Only a human writes ``source-verified``,
#: and approval lives in its own file bound to a content hash.
VERIFICATION_STATUSES = frozenset(
    {
        "self-verified",  # mechanical: a sympy identity residual is zero
        "agent-verified",  # the assistant extracted it from a source
        "cross-verified",  # two independent agent evidence packets agree
        "researcher-verified",  # researcher accepted a recorded verification packet
        "source-verified",  # a human directly inspected the rendered primary source
        "needs-source-verification",  # nobody has
    }
)
# Verification has two useful rungs before a direct human source read:
# ``agent-verified`` means an agent inspected the pinned primary source at the
# registered locator; ``cross-verified`` means two independent agent evidence
# packets agree; ``researcher-verified`` means the researcher accepted the
# evidence packet.  The latter is approval, not a claim that the researcher
# visually re-read every PDF.  A supplied extraction without either independent
# check remains ``needs-source-verification``.

#: What the registry stores for an equation, which decides what can be
#: generated from it. Coverage is always reported per kind so an unregistered
#: iterative solve cannot be confused with one nobody got to.
REPRESENTATIONS = frozenset(
    {
        "symbolic",  # a sympy expression -> LaTeX + kernel + partials
        "literal-coefficient",  # a table of cited values -> LaTeX table + data module
        "algorithmic",  # a description + implementation binding -> citation only
    }
)

#: The physical role of a wind record.  This deliberately says what a source
#: *is*, without granting it dispatch ownership: selection and smoothing stay
#: in ``startrax.winds.spec`` / prepared plans.
WIND_KINDS = frozenset(
    {
        "rate_relation",
        "transition_criterion",
        "domain_guard",
        "correction_or_normalization",
    }
)

#: Outputs a record can emit. ``latex`` costs nothing and is never declined;
#: ``oracle`` is the default for anything expressible because it adds a check
#: without changing what ships.
EMITS = frozenset({"latex", "oracle", "kernel", "partials", "identity"})

#: ``identity`` is a DERIVATION check and is different in kind from ``oracle``.
#: An oracle proves an implementation matches the expression the registry holds;
#: an identity proves the expression is *algebraically true* -- that the stated
#: relation follows from the ones it was derived from. Neither subsumes the
#: other: an implementation can faithfully realise a wrong derivation, and a
#: correct derivation can be implemented wrongly. Carried over from hydrax's
#: registry (ADR-0007) when it merged into this one (ADR-0010), because it was
#: the one check startrax's design did not have.

#: What was done about a restriction the paper states. A ``deferred`` or
#: ``rejected`` caveat requires a reason -- that is the field which would have
#: made the 1-kelvin defect visible on the day it was written.
CAVEAT_STATUSES = frozenset({"enforced", "deferred", "rejected"})

#: Whether the range and regime declared by a paper are actively enforced by
#: the production owner, merely reported, or intentionally unavailable pending
#: a paper-backed reconstruction.  This is distinct from an individual caveat:
#: a domain is the compact, machine-readable answer to "where may I evaluate
#: this relation?" while caveats retain the paper's qualifications and reasons.
VALIDITY_STATUSES = frozenset({"enforced", "reported", "quarantined"})


class RegistryError(ValueError):
    """A malformed registry raises.

    A provenance system that silently half-works is worse than none, because it
    produces confident wrong answers.
    """


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise RegistryError(f"{where}: missing required field {key!r}")
    return mapping[key]


def _check_member(value: str, allowed: frozenset[str], key: str, where: str) -> str:
    if value not in allowed:
        raise RegistryError(f"{where}: {key}={value!r} is not one of {sorted(allowed)}")
    return value


def _finite_number(value: Any, *, name: str, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryError(f"{where}: {name} must be a finite numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise RegistryError(f"{where}: {name} must be a finite numeric value")
    return result


def _optional_string(payload: dict[str, Any], key: str, *, where: str) -> str | None:
    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, str):
        raise RegistryError(f"{where}: {key} must be a string")
    return value


def _optional_member(
    payload: dict[str, Any], key: str, allowed: frozenset[str], *, where: str
) -> str | None:
    if key not in payload:
        return None
    return _check_member(payload[key], allowed, key, where)


def _string_tuple(payload: dict[str, Any], key: str, *, where: str) -> tuple[str, ...]:
    if key not in payload:
        return ()
    value = payload[key]
    if not isinstance(value, list):
        raise RegistryError(f"{where}: {key} must be an array of non-empty strings")
    if any(not isinstance(item, str) or not item for item in value):
        raise RegistryError(f"{where}: {key} must be an array of non-empty strings")
    return tuple(value)


def _reject_unknown_keys(
    payload: dict[str, Any], allowed: frozenset[str], *, where: str
) -> None:
    """Record tables fail closed on ANY unknown key (task 3.4, 2026-08-28).

    Before this, unknown keys loaded silently — which is exactly how seven
    dead schema fields (zero readers, ~320 TOML lines) accumulated. Bindings
    were already strict; this extends the same rule to every record table.
    """
    unknown = set(payload) - allowed
    if unknown:
        raise RegistryError(f"{where}: unknown keys {sorted(unknown)}")


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceRecord:
    """A bibliographic *pointer*, never a PDF.

    Provenance is public and in-repo; the library stays outside it. ``id`` is
    the bibkey -- the join key across the brain library, the brain source note
    and this directory.
    """

    id: str
    verification: str
    source_version: str | None = None
    doi: str | None = None
    reference: str | None = None
    pdf_sha256: str | None = None
    pdf_unavailable: str | None = None
    #: Stable pointer to the human-readable source note.  The URI is a join,
    #: not a second numerical authority: the values remain in this registry.
    source_note_ref: str | None = None

    _KEYS = frozenset(
        {
            "id",
            "verification",
            "source_version",
            "doi",
            "reference",
            "pdf_sha256",
            "pdf_unavailable",
            "source_note_ref",
        }
    )

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> SourceRecord:
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        record = cls(
            id=_require(payload, "id", where),
            verification=_check_member(
                _require(payload, "verification", where),
                VERIFICATION_STATUSES,
                "verification",
                where,
            ),
            source_version=payload.get("source_version"),
            doi=payload.get("doi"),
            reference=payload.get("reference"),
            pdf_sha256=payload.get("pdf_sha256"),
            pdf_unavailable=payload.get("pdf_unavailable"),
            source_note_ref=payload.get("source_note_ref"),
        )
        return record


@dataclass(frozen=True)
class AtlasDecisionRecord:
    """A researcher-authorized Atlas policy, distinct from a source equation.

    A paper record says what its authors wrote.  This record says what
    Startrax deliberately chooses when the source leaves a seam open.  Keeping
    the two types apart makes a closure queryable without mislabelling it as a
    paper result.
    """

    id: str
    treatment: str
    route_id: str
    relation_ids: tuple[str, ...]
    expression: str
    source_equation_refs: tuple[str, ...]
    rationale: str

    _KEYS = frozenset(
        {
            "id",
            "treatment",
            "route_id",
            "relation_ids",
            "expression",
            "source_equation_refs",
            "rationale",
        }
    )

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> AtlasDecisionRecord:
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        treatment = _require(payload, "treatment", where)
        if treatment not in {
            "PHYSICS_CLOSURE",
            "PHYSICS_EXTRAPOLATED",
            "ATLAS_INTERPOLATION",
            "EXPERIMENTAL_EXCEPTION",
        }:
            raise RegistryError(
                f"{where}: treatment={treatment!r} is not a registered Atlas decision treatment"
            )
        record = cls(
            id=_require(payload, "id", where),
            treatment=treatment,
            route_id=_require(payload, "route_id", where),
            relation_ids=_string_tuple(payload, "relation_ids", where=where),
            expression=_require(payload, "expression", where),
            source_equation_refs=_string_tuple(
                payload, "source_equation_refs", where=where
            ),
            rationale=_require(payload, "rationale", where),
        )
        # A decision may directly govern a researcher-derived model (for
        # example a same-event input projection) without pretending that the
        # composition itself is a source relation.  Derived-model coverage is
        # audited separately.
        if not record.source_equation_refs:
            raise RegistryError(f"{where}: source_equation_refs must not be empty")
        return record


@dataclass(frozen=True)
class AtlasRelationRecord:
    """One complete provenance/test contract for a frozen Atlas relation."""

    relation_id: str
    route_ids: tuple[str, ...]
    oracle_kind: str
    source_equation_refs: tuple[str, ...]
    decision_id: str | None = None
    evaluator: str | None = None

    _KEYS = frozenset(
        {
            "relation_id",
            "route_ids",
            "oracle_kind",
            "source_equation_refs",
            "decision_id",
            "evaluator",
        }
    )

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> "AtlasRelationRecord":
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        oracle_kind = _require(payload, "oracle_kind", where)
        if oracle_kind not in {
            "source_oracle",
            "closure_oracle",
            "stateful_oracle",
            "algorithmic",
        }:
            raise RegistryError(f"{where}: invalid oracle_kind={oracle_kind!r}")
        record = cls(
            relation_id=_require(payload, "relation_id", where),
            route_ids=_string_tuple(payload, "route_ids", where=where),
            oracle_kind=oracle_kind,
            source_equation_refs=_string_tuple(
                payload, "source_equation_refs", where=where
            ),
            decision_id=_optional_string(payload, "decision_id", where=where),
            evaluator=_optional_string(payload, "evaluator", where=where),
        )
        if not record.route_ids or not record.source_equation_refs:
            raise RegistryError(
                f"{where}: route_ids and source_equation_refs must not be empty"
            )
        if oracle_kind == "algorithmic":
            if record.evaluator is not None:
                raise RegistryError(
                    f"{where}: algorithmic relation must not claim an evaluator"
                )
        elif record.evaluator is None:
            raise RegistryError(f"{where}: executable relation needs evaluator")
        if oracle_kind == "closure_oracle" and record.decision_id is None:
            raise RegistryError(f"{where}: closure_oracle needs a typed decision_id")
        return record


@dataclass(frozen=True)
class DerivedModelRecord:
    """One Startrax-authored executable model with independent provenance axes."""

    id: str
    status: str
    route_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    source_authority: str | None
    source_support: str | None
    researcher_review_status: str | None
    source_equation_refs: tuple[str, ...]
    source_validity: str
    closure_validity: str
    policy: str

    # The second half of this set is read by RAW-TOML gates, not by this
    # dataclass: test_wind_atlas_registry pins `domain.teff_max_k` to the
    # Pauli boundary and asserts on `inputs`/`closure_validity`, reading the
    # file with tomllib directly. They are live schema with readers, so the
    # strictness gate admits them; only genuinely unread keys are refused.
    _KEYS = frozenset(
        {
            "id",
            "status",
            "route_ids",
            "relation_ids",
            "source_authority",
            "source_support",
            "researcher_review_status",
            "source_equation_refs",
            "source_validity",
            "closure_validity",
            "policy",
            "output",
            "limitations",
            "owner",
            "inputs",
            "validation",
            "decision_id",
            "parameters",
            "domain",
        }
    )

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> "DerivedModelRecord":
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        status = _require(payload, "status", where)
        if status not in {"ACTIVE", "APPROVED_NOT_IMPLEMENTED", "RETIRED"}:
            raise RegistryError(f"{where}: invalid derived-model status={status!r}")
        authority = _optional_string(payload, "source_authority", where=where)
        if authority not in {None, "researcher_derived"}:
            raise RegistryError(
                f"{where}: invalid derived-model source_authority={authority!r}"
            )
        source_support = _optional_string(payload, "source_support", where=where)
        if source_support not in {None, "source_backed", "not_source_backed"}:
            raise RegistryError(
                f"{where}: invalid derived-model source_support={source_support!r}"
            )
        if authority == "researcher_derived" and source_support is None:
            raise RegistryError(
                f"{where}: researcher-derived model must declare source_support"
            )
        review_status = _optional_string(
            payload, "researcher_review_status", where=where
        )
        if review_status not in {None, "pending", "approved", "verified"}:
            raise RegistryError(
                f"{where}: invalid derived-model researcher_review_status={review_status!r}"
            )
        if authority == "researcher_derived" and review_status is None:
            raise RegistryError(
                f"{where}: researcher-derived model must declare researcher_review_status"
            )
        if authority != "researcher_derived" and review_status is not None:
            raise RegistryError(
                f"{where}: non-researcher-derived model must not declare researcher_review_status"
            )
        if source_support == "source_backed" and not _string_tuple(
            payload, "source_equation_refs", where=where
        ):
            raise RegistryError(
                f"{where}: source_backed model needs source_equation_refs"
            )
        return cls(
            id=_require(payload, "id", where),
            status=status,
            route_ids=_string_tuple(payload, "route_ids", where=where),
            relation_ids=_string_tuple(payload, "relation_ids", where=where),
            source_authority=authority,
            source_support=source_support,
            researcher_review_status=review_status,
            source_equation_refs=_string_tuple(
                payload, "source_equation_refs", where=where
            ),
            source_validity=_require(payload, "source_validity", where),
            closure_validity=_require(payload, "closure_validity", where),
            policy=_require(payload, "policy", where),
        )


@dataclass(frozen=True)
class SymbolRecord:
    """A physical quantity, shared across sources.

    Two names, because they serve two printers: ``name`` is a code-safe Python
    identifier (``lambdify`` dummifies anything that is not) and ``latex`` is
    display only.
    """

    id: str
    name: str
    latex: str
    unit: str
    description: str | None = None
    assumptions: tuple[str, ...] = ()

    _KEYS = frozenset({"id", "name", "latex", "unit", "description", "assumptions"})

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> SymbolRecord:
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        name = _require(payload, "name", where)
        if not name.isidentifier():
            raise RegistryError(
                f"{where}: symbol name {name!r} is not a Python identifier; "
                "lambdify would dummify it out of the generated code"
            )
        return cls(
            id=_require(payload, "id", where),
            name=name,
            latex=_require(payload, "latex", where),
            unit=_require(payload, "unit", where),
            description=payload.get("description"),
            assumptions=tuple(payload.get("assumptions", ())),
        )


@dataclass(frozen=True)
class CoefficientRecord:
    """One fitted number, citable on its own, carrying its published sigma.

    ``symbol`` is the code-safe name the equation's expression refers to;
    ``name`` is the human description. ``sigma`` is the coefficient's own fit
    uncertainty -- a different quantity from the relation's observed scatter,
    which lives on the equation.
    """

    id: str
    symbol: str
    name: str
    value: float
    unit: str
    verification: str
    sigma: float | None = None
    latex: str | None = None
    note: str | None = None
    source_ids: tuple[str, ...] = ()

    _KEYS = frozenset(
        {
            "id",
            "symbol",
            "name",
            "value",
            "unit",
            "verification",
            "sigma",
            "latex",
            "note",
            "source_ids",
        }
    )

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> CoefficientRecord:
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        symbol = _require(payload, "symbol", where)
        if not symbol.isidentifier():
            raise RegistryError(
                f"{where}: coefficient symbol {symbol!r} is not an identifier"
            )
        value = _require(payload, "value", where)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RegistryError(
                f"{where}: coefficient value {value!r} is {type(value).__name__}, not a number "
                "(this is the failure mode TOML was chosen to avoid)"
            )
        sigma = payload.get("sigma")
        if sigma is not None and (not isinstance(sigma, (int, float)) or sigma <= 0.0):
            raise RegistryError(
                f"{where}: sigma must be a positive number, got {sigma!r}"
            )
        return cls(
            id=_require(payload, "id", where),
            symbol=symbol,
            name=_require(payload, "name", where),
            value=float(value),
            unit=_require(payload, "unit", where),
            verification=_check_member(
                _require(payload, "verification", where),
                VERIFICATION_STATUSES,
                "verification",
                where,
            ),
            sigma=None if sigma is None else float(sigma),
            latex=payload.get("latex"),
            note=payload.get("note"),
            source_ids=tuple(payload.get("source_ids", ())),
        )


@dataclass(frozen=True)
class CaveatRecord:
    """A restriction the paper states, and what was done about it."""

    text: str
    locator: str
    status: str
    reason: str | None = None
    implementation_note: str | None = None

    _KEYS = frozenset({"text", "locator", "status", "reason", "implementation_note"})

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> CaveatRecord:
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        record = cls(
            text=_require(payload, "text", where),
            locator=_require(payload, "locator", where),
            status=_check_member(
                _require(payload, "status", where), CAVEAT_STATUSES, "status", where
            ),
            reason=payload.get("reason"),
            implementation_note=payload.get("implementation_note"),
        )
        if record.status != "enforced" and not record.reason:
            raise RegistryError(
                f"{where}: a {record.status!r} caveat requires a reason -- "
                "an undocumented omission is how the 1-kelvin floor shipped"
            )
        return record


@dataclass(frozen=True)
class AnchorRecord:
    """A value taken from the paper, used to check the registry against it.

    The only check that catches a transcription error in the registry itself,
    which is the failure mode the registry would otherwise introduce.
    """

    inputs: dict[str, float]
    expect: float
    tolerance: float
    locator: str

    _KEYS = frozenset({"inputs", "expect", "tolerance", "locator"})

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> AnchorRecord:
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        return cls(
            inputs=dict(_require(payload, "inputs", where)),
            expect=float(_require(payload, "expect", where)),
            tolerance=float(_require(payload, "tolerance", where)),
            locator=_require(payload, "locator", where),
        )


@dataclass(frozen=True)
class ValidityLimit:
    """One declared numerical domain of an equation, interval or discrete grid.

    ``nodes`` exists because several sources publish a GRID, not an interval:
    Sabhahit (2023) deploys at five metallicities and its own caveat says the
    Galactic value "cannot select S23"; Sabhahit (2022) runs two, Galactic and
    LMC. An interval cannot express that, so before 2026-08-25 the only way to
    enforce a grid was a hand-typed tuple in source -- which is what
    ``vms_s23/constants.py`` did, in a module that binds everything else.
    """

    lower: float | None = None
    upper: float | None = None
    unit: str | None = None
    nodes: tuple[float, ...] | None = None

    _KEYS = frozenset({"lower", "upper", "unit", "nodes"})

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> ValidityLimit:
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        lower = payload.get("lower")
        upper = payload.get("upper")
        raw_nodes = payload.get("nodes")
        for name, value in (("lower", lower), ("upper", upper)):
            if value is not None and (
                not isinstance(value, (int, float)) or isinstance(value, bool)
            ):
                raise RegistryError(f"{where}: {name} must be numeric when present")
        nodes: tuple[float, ...] | None = None
        if raw_nodes is not None:
            if not isinstance(raw_nodes, list) or not raw_nodes:
                raise RegistryError(f"{where}: nodes must be a non-empty list")
            for value in raw_nodes:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise RegistryError(f"{where}: every node must be numeric")
            nodes = tuple(sorted(float(v) for v in raw_nodes))
            if len(set(nodes)) != len(nodes):
                raise RegistryError(f"{where}: nodes contains duplicates")
        if lower is None and upper is None and nodes is None:
            raise RegistryError(f"{where}: needs at least one of lower, upper or nodes")
        if lower is not None and upper is not None and lower > upper:
            raise RegistryError(f"{where}: lower exceeds upper")
        # A record must not disagree with ITSELF. Declaring a grid AND an
        # interval that does not span it is the two-owner hazard one level in:
        # a consumer reading the interval and a consumer reading the grid would
        # enforce different domains from the same record.
        if nodes is not None:
            if lower is not None and lower != nodes[0]:
                raise RegistryError(
                    f"{where}: lower {lower} is not the smallest node {nodes[0]}"
                )
            if upper is not None and upper != nodes[-1]:
                raise RegistryError(
                    f"{where}: upper {upper} is not the largest node {nodes[-1]}"
                )
        return cls(
            lower=None if lower is None else float(lower),
            upper=None if upper is None else float(upper),
            unit=payload.get("unit"),
            nodes=nodes,
        )


@dataclass(frozen=True)
class ValidityRecord:
    """Machine-readable regime and declared input limits for one relation."""

    regime: str
    status: str
    limits: dict[str, ValidityLimit] = field(default_factory=dict)
    implementation_note: str | None = None

    _KEYS = frozenset({"regime", "status", "limits", "implementation_note"})

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> ValidityRecord:
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        limits = {
            name: ValidityLimit.from_toml(value, where=f"{where}.limits.{name}")
            for name, value in dict(payload.get("limits", {})).items()
        }
        if not limits:
            raise RegistryError(f"{where}: needs at least one declared limit")
        return cls(
            regime=_require(payload, "regime", where),
            status=_check_member(
                _require(payload, "status", where), VALIDITY_STATUSES, "status", where
            ),
            limits=limits,
            implementation_note=payload.get("implementation_note"),
        )


@dataclass(frozen=True)
class ScalarBinding:
    """One cited scalar assigned to an equation symbol."""

    value: float
    verification: str | None = None
    source_ids: tuple[str, ...] = ()
    note: str | None = None


@dataclass(frozen=True)
class DriverBinding:
    """A semantic input owner, rather than a scalar convention."""

    driver: str
    note: str | None = None


SymbolBinding = ScalarBinding | DriverBinding


def _parse_symbol_binding(payload: Any, *, symbol: str, where: str) -> SymbolBinding:
    if not isinstance(payload, dict):
        raise RegistryError(f"{where}: binding {symbol!r} must be a table")
    if ("value" in payload) == ("driver" in payload):
        raise RegistryError(
            f"{where}: binding {symbol!r} needs exactly one of value or driver"
        )
    if "value" in payload:
        allowed = {"value", "verification", "source_ids", "note"}
        unknown = set(payload) - allowed
        if unknown:
            raise RegistryError(
                f"{where}: binding {symbol!r} has unknown keys {sorted(unknown)}"
            )
        return ScalarBinding(
            value=_finite_number(payload["value"], name="binding value", where=where),
            verification=_optional_member(
                payload, "verification", VERIFICATION_STATUSES, where=where
            ),
            source_ids=_string_tuple(payload, "source_ids", where=where),
            note=_optional_string(payload, "note", where=where),
        )
    driver = payload["driver"]
    if not isinstance(driver, str) or not driver:
        raise RegistryError(
            f"{where}: binding {symbol!r} driver must be a non-empty string"
        )
    allowed = {"driver", "note"}
    unknown = set(payload) - allowed
    if unknown:
        raise RegistryError(
            f"{where}: binding {symbol!r} has unknown keys {sorted(unknown)}"
        )
    return DriverBinding(
        driver=driver, note=_optional_string(payload, "note", where=where)
    )


@dataclass(frozen=True)
class EquationRecord:
    """The paper's formula, verbatim, as a string.

    Domain clamps stay out of ``expression``: they are policy owned by the
    domain layer, not part of the paper's equation. Baking one in is precisely
    the conflation that let a 1-kelvin floor look like physics.
    """

    id: str
    name: str
    representation: str
    equation_ref: str
    emits: tuple[str, ...]
    source_ids: tuple[str, ...]
    symbol_ids: tuple[str, ...] = ()
    coefficient_ids: tuple[str, ...] = ()
    implementation_ids: tuple[str, ...] = ()
    expression: str | None = None
    returns: str | None = None
    wind_kind: str | None = None
    applies_to: tuple[str, ...] = ()
    scatter_dex: float | None = None
    symbol_bindings: dict[str, SymbolBinding] = field(default_factory=dict)
    #: Map an input symbol to the registered equation that produces it.  It
    #: makes compositions such as Sander Eq. 28 <- Eqs. 30--32 auditable,
    #: rather than encoding that graph in a test-local dictionary.
    derived_symbols: dict[str, str] = field(default_factory=dict)
    validity: ValidityRecord | None = None
    anchors: tuple[AnchorRecord, ...] = ()
    #: Expressions that MUST each simplify to zero. Required when ``emits``
    #: contains ``identity``. A list because one registered relation usually has
    #: several things worth proving about it -- the core-formation slope has a
    #: closed form AND two pinned endpoints, and they are the same claim.
    identity: tuple[str, ...] = ()
    #: Substitutions applied before simplifying, as ``{expression: replacement}``.
    #: A derivation often holds only GIVEN a closure -- the mass-coordinate slope
    #: identity holds given ``Derivative(m(r), r) = 4 pi r^2 rho`` -- and stating
    #: the hypothesis as data is strictly more informative than burying it in a
    #: Python callable's ``.subs()`` call, which is where it lived before.
    identity_given: dict[str, str] = field(default_factory=dict)
    note: str | None = None

    _KEYS = frozenset(
        {
            "id",
            "identity",
            "identity_given",
            "name",
            "representation",
            "equation_ref",
            "emits",
            "source_ids",
            "symbol_ids",
            "coefficient_ids",
            "implementation_ids",
            "expression",
            "returns",
            "wind_kind",
            "applies_to",
            "scatter_dex",
            "symbol_bindings",
            "derived_symbols",
            "validity",
            "anchor",
            "note",
        }
    )

    @classmethod
    def from_toml(cls, payload: dict[str, Any], *, where: str) -> EquationRecord:
        _reject_unknown_keys(payload, cls._KEYS, where=where)
        representation = _check_member(
            _require(payload, "representation", where),
            REPRESENTATIONS,
            "representation",
            where,
        )
        emits = tuple(_require(payload, "emits", where))
        for emit in emits:
            _check_member(emit, EMITS, "emits", where)

        equation_ref = _require(payload, "equation_ref", where)
        # locator-specific: can the researcher confirm or refute this in under
        # a minute? A bare citation cannot be re-checked.
        if not any(
            token in equation_ref
            for token in ("Eq", "eq.", "Table", "Fig", "Sect", "Sec.", "p.", "pp.")
        ):
            raise RegistryError(
                f"{where}: equation_ref {equation_ref!r} names no equation, page, "
                "table, figure or section"
            )

        expression = payload.get("expression")
        if representation == "symbolic" and not expression:
            raise RegistryError(
                f"{where}: representation='symbolic' requires an expression"
            )
        if representation != "symbolic" and expression:
            raise RegistryError(
                f"{where}: representation={representation!r} must not carry an expression"
            )
        if "oracle" in emits and not expression:
            raise RegistryError(
                f"{where}: emits 'oracle' but has no expression to check"
            )
        identity = tuple(payload.get("identity", ()) or ())
        if "identity" in emits and not identity:
            raise RegistryError(
                f"{where}: emits 'identity' but has no identity residual to check"
            )
        if identity and "identity" not in emits:
            raise RegistryError(
                f"{where}: carries an identity but does not emit 'identity', so "
                "nothing would ever check it"
            )
        identity_given = payload.get("identity_given", {}) or {}
        if identity_given and not identity:
            raise RegistryError(
                f"{where}: identity_given without an identity to apply it to"
            )

        try:
            scatter_dex = (
                None
                if "scatter_dex" not in payload
                else _finite_number(
                    payload["scatter_dex"], name="scatter_dex", where=where
                )
            )
        except RegistryError as error:
            raise RegistryError(
                f"{where}: scatter_dex must be finite and non-negative"
            ) from error
        if scatter_dex is not None and scatter_dex < 0.0:
            raise RegistryError(f"{where}: scatter_dex must be finite and non-negative")
        symbol_bindings = {
            symbol: _parse_symbol_binding(binding, symbol=symbol, where=where)
            for symbol, binding in dict(payload.get("symbol_bindings", {})).items()
        }
        anchors = tuple(
            AnchorRecord.from_toml(item, where=f"{where} anchor[{index}]")
            for index, item in enumerate(payload.get("anchor", ()))
        )
        return cls(
            id=_require(payload, "id", where),
            name=_require(payload, "name", where),
            representation=representation,
            equation_ref=equation_ref,
            emits=emits,
            source_ids=tuple(_require(payload, "source_ids", where)),
            symbol_ids=tuple(payload.get("symbol_ids", ())),
            coefficient_ids=tuple(payload.get("coefficient_ids", ())),
            implementation_ids=tuple(payload.get("implementation_ids", ())),
            expression=expression,
            returns=payload.get("returns"),
            wind_kind=_optional_member(payload, "wind_kind", WIND_KINDS, where=where),
            applies_to=_string_tuple(payload, "applies_to", where=where),
            scatter_dex=scatter_dex,
            symbol_bindings=symbol_bindings,
            derived_symbols=dict(payload.get("derived_symbols", {})),
            validity=(
                None
                if "validity" not in payload
                else ValidityRecord.from_toml(
                    payload["validity"], where=f"{where}.validity"
                )
            ),
            anchors=anchors,
            identity=identity,
            identity_given=dict(identity_given),
            note=payload.get("note"),
        )


@dataclass(frozen=True)
class SourceBundle:
    """Everything one paper contributed: one directory, one verification act."""

    source: SourceRecord
    equations: dict[str, EquationRecord]
    coefficients: dict[str, CoefficientRecord]
    caveats: tuple[CaveatRecord, ...]
    symbols: dict[str, SymbolRecord]

    def symbols_for(self, equation: EquationRecord) -> dict[str, SymbolRecord]:
        """Symbol records an equation declares, keyed by their code-safe name."""
        resolved: dict[str, SymbolRecord] = {}
        for symbol_id in equation.symbol_ids:
            if symbol_id not in self.symbols:
                raise RegistryError(
                    f"{equation.id}: symbol {symbol_id!r} is used but never declared"
                )
            record = self.symbols[symbol_id]
            resolved[record.name] = record
        return resolved

    def coefficients_for(
        self, equation: EquationRecord
    ) -> dict[str, CoefficientRecord]:
        """Coefficient records an equation declares, keyed by expression symbol."""
        resolved: dict[str, CoefficientRecord] = {}
        for coefficient_id in equation.coefficient_ids:
            if coefficient_id not in self.coefficients:
                raise RegistryError(
                    f"{equation.id}: coefficient {coefficient_id!r} resolves to nothing"
                )
            record = self.coefficients[coefficient_id]
            resolved[record.symbol] = record
        return resolved

    def derived_equations_for(
        self, equation: EquationRecord
    ) -> dict[str, EquationRecord]:
        """Registered sub-relations supplying an equation's named inputs."""
        resolved: dict[str, EquationRecord] = {}
        names = {symbol.name for symbol in self.symbols_for(equation).values()}
        for symbol, equation_id in equation.derived_symbols.items():
            if symbol not in names:
                raise RegistryError(
                    f"{equation.id}: derived symbol {symbol!r} is not a declared symbol"
                )
            if equation_id not in self.equations:
                raise RegistryError(
                    f"{equation.id}: derived symbol {symbol!r} references unknown "
                    f"equation {equation_id!r}"
                )
            if not self.equations[equation_id].returns:
                raise RegistryError(
                    f"{equation.id}: derived equation {equation_id!r} has no declared return"
                )
            resolved[symbol] = self.equations[equation_id]
        return resolved
