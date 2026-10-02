"""Registry-level provenance tables, the cross-registry link check, and BibTeX.

Each test copies ``fixtures/registry_provenance`` (a root in which every id
resolves) and breaks one thing, so a refusal is attributable to that one change.
The valid-fixture test is the control: without it a refusal could come from the
fixture itself.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from jaxstro.registry import (
    RegistryError,
    SourceRecord,
    bibliography,
    bibtex_entry,
    load_claims,
    load_evidence,
    load_implementations,
    load_methods,
    load_registry,
    load_source,
    missing_files,
    unresolved_implementations,
)

FIXTURE = Path(__file__).parent / "fixtures" / "registry_provenance"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    target = tmp_path / "registry"
    shutil.copytree(FIXTURE, target)
    return target


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, f"fixture edit target {old!r} not in {path.name}"
    path.write_text(text.replace(old, new, 1))


def _append(path: Path, text: str) -> None:
    path.write_text(path.read_text() + "\n" + text)


# ---------------------------------------------------------------------------
# the control
# ---------------------------------------------------------------------------


def test_the_fixture_registry_loads_with_every_link_resolved(root: Path) -> None:
    registry = load_registry(root)
    assert sorted(registry.bundles) == ["alpha2001", "beta2002", "derivations"]
    assert registry.equation_bundles == {
        "EQ-ALPHA-001": "alpha2001",
        "EQ-BETA-001": "beta2002",
        "EQ-DERIV-001": "derivations",
    }
    assert sorted(registry.implementations) == ["IMPL-ALPHA", "IMPL-RETIRED"]
    assert sorted(registry.evidence) == ["EVID-ALPHA", "EVID-OLD"]
    assert list(registry.methods) == ["METHOD-ALPHA"]
    assert list(registry.claims) == ["CLAIM-ALPHA"]
    alpha = registry.evidence["EVID-ALPHA"]
    assert alpha.gates == (
        "tests/validation/test_alpha.py::test_order",
        "tests/unit/test_alpha.py",
    )
    assert alpha.artifacts == ("validation/plots/alpha.png",)


# ---------------------------------------------------------------------------
# the four loaders: malformed rows, unknown statuses, unknown keys
# ---------------------------------------------------------------------------

LOADERS = {
    "implementations.toml": load_implementations,
    "evidence.toml": load_evidence,
    "methods.toml": load_methods,
    "claims.toml": load_claims,
}

REFUSED_EDITS = [
    # (file, old, new, phrase the refusal must contain)
    (
        "implementations.toml",
        'status = "implemented"',
        'status = "done"',
        "status='done'",
    ),
    (
        "implementations.toml",
        'name = "Alpha kernel"\n',
        "",
        "missing required field 'name'",
    ),
    (
        "implementations.toml",
        'module = "jaxstro.registry"',
        'module = "jaxstro/registry"',
        "dotted",
    ),
    (
        "implementations.toml",
        'symbol = "load_registry"',
        'symbol = "a.b"',
        "identifier",
    ),
    (
        "implementations.toml",
        'note = "Points',
        'notes = "Points',
        "unknown keys ['notes']",
    ),
    ("evidence.toml", 'status = "passing"', 'status = "green"', "status='green'"),
    ("evidence.toml", 'criterion = "Order', 'gate = "Order', "unknown keys ['gate']"),
    (
        "evidence.toml",
        'gates = ["tests/validation',
        'gates = ["/abs/tests/validation',
        "relative path",
    ),
    (
        "evidence.toml",
        '"tests/unit/test_alpha.py"]',
        '"tests/unit/test_alpha.txt"]',
        ".py test file",
    ),
    ("evidence.toml", 'test_order"', 'test_order::"', "empty node id"),
    (
        "evidence.toml",
        '["validation/plots/alpha.png"]',
        '["../alpha.png"]',
        "relative path",
    ),
    (
        "evidence.toml",
        'equation_ids = ["EQ-ALPHA-001", "EQ-BETA-001"]',
        'equation_ids = "EQ-ALPHA-001"',
        "array",
    ),
    ("methods.toml", 'status = "verified"', 'status = "passing"', "status='passing'"),
    (
        "methods.toml",
        'formulation = "godunov-ale"\n',
        "",
        "missing required field 'formulation'",
    ),
    ("claims.toml", 'status = "substantiated"', 'status = "true"', "status='true'"),
    (
        "claims.toml",
        'statement = "The alpha scheme is second order."',
        'statement = ""',
        "non-empty",
    ),
]


@pytest.mark.parametrize(("filename", "old", "new", "phrase"), REFUSED_EDITS)
def test_a_malformed_row_is_refused(
    root: Path, filename: str, old: str, new: str, phrase: str
) -> None:
    LOADERS[filename](root)  # loads before the edit
    _edit(root / filename, old, new)
    with pytest.raises(RegistryError) as caught:
        LOADERS[filename](root)
    assert phrase in str(caught.value)


@pytest.mark.parametrize("missing", ["gates", "criterion", "result"])
def test_a_passing_evidence_row_must_state_gate_criterion_and_result(
    root: Path, missing: str
) -> None:
    path = root / "evidence.toml"
    lines = path.read_text().splitlines()
    first_row_end = lines.index("[[evidence]]", 1)
    kept = [
        line
        for index, line in enumerate(lines)
        if not (index < first_row_end and line.startswith(f"{missing} ="))
    ]
    path.write_text("\n".join(kept) + "\n")
    with pytest.raises(RegistryError, match=f"needs \\['{missing}'\\]"):
        load_evidence(root)


def test_a_retired_evidence_row_needs_no_criterion_or_result(root: Path) -> None:
    assert load_evidence(root)["EVID-OLD"].criterion is None


@pytest.mark.parametrize("filename", sorted(LOADERS))
def test_a_duplicate_id_an_empty_table_and_a_stray_key_are_refused(
    root: Path, filename: str
) -> None:
    path = root / filename
    original = path.read_text()
    first_row = "[[" + original.split("[[", 2)[1]
    path.write_text(original + "\n" + first_row)
    with pytest.raises(RegistryError, match="duplicate"):
        LOADERS[filename](root)
    path.write_text("")
    with pytest.raises(RegistryError, match="declares no"):
        LOADERS[filename](root)
    path.write_text("[[rows]]\nid = 'X'\n")
    with pytest.raises(RegistryError, match="unknown top-level keys"):
        LOADERS[filename](root)
    path.unlink()
    with pytest.raises(RegistryError, match="not found"):
        LOADERS[filename](root)


def test_a_substantiated_claim_without_evidence_is_refused(root: Path) -> None:
    _edit(root / "claims.toml", 'evidence_ids = ["EVID-ALPHA"]\n', "")
    with pytest.raises(RegistryError, match="substantiated claim needs evidence_ids"):
        load_claims(root)


# ---------------------------------------------------------------------------
# the link check
# ---------------------------------------------------------------------------

DANGLING_EDITS = [
    (
        "sources/alpha2001/equations.toml",
        '["IMPL-ALPHA"]',
        '["IMPL-NONE"]',
        "EQ-ALPHA-001: implementation_id 'IMPL-NONE' names no implementation row",
    ),
    (
        "sources/derivations/equations.toml",
        '["derivations"]',
        '["derivations", "nobody1999"]',
        "EQ-DERIV-001: source_id 'nobody1999' names no source bundle",
    ),
    (
        "sources/alpha2001/coefficients.toml",
        '"beta2002"]',
        '"gamma2003"]',
        "alpha2001 VAL-ALPHA-A: source_id 'gamma2003' names no source bundle",
    ),
    (
        "implementations.toml",
        '["EVID-ALPHA"]',
        '["EVID-NONE"]',
        "IMPL-ALPHA: evidence_id 'EVID-NONE' names no evidence row",
    ),
    (
        "evidence.toml",
        '"EQ-BETA-001"]',
        '"EQ-NONE-001"]',
        "EVID-ALPHA: equation_id 'EQ-NONE-001' names no equation",
    ),
    (
        "evidence.toml",
        '["METHOD-ALPHA"]',
        '["METHOD-NONE"]',
        "EVID-ALPHA: method_id 'METHOD-NONE' names no method row",
    ),
    (
        "methods.toml",
        '["EQ-ALPHA-001"]',
        '["EQ-NONE-001"]',
        "METHOD-ALPHA: equation_id 'EQ-NONE-001' names no equation",
    ),
    (
        "methods.toml",
        '["alpha2001"]',
        '["REF-ALPHA-2001"]',
        "METHOD-ALPHA: source_id 'REF-ALPHA-2001' names no source bundle",
    ),
    (
        "claims.toml",
        '["EVID-ALPHA"]',
        '["EVID-ALPHA", "EVID-NONE"]',
        "CLAIM-ALPHA: evidence_id 'EVID-NONE' names no evidence row",
    ),
]


@pytest.mark.parametrize(("filename", "old", "new", "failure"), DANGLING_EDITS)
def test_a_dangling_reference_is_refused(
    root: Path, filename: str, old: str, new: str, failure: str
) -> None:
    _edit(root / filename, old, new)
    with pytest.raises(RegistryError, match="unresolved links") as caught:
        load_registry(root)
    assert failure in str(caught.value)


def test_every_dangling_reference_is_reported_not_only_the_first(root: Path) -> None:
    _edit(root / "methods.toml", '["alpha2001"]', '["nobody1999"]')
    _edit(root / "claims.toml", '["EVID-ALPHA"]', '["EVID-ALPHA", "EVID-NONE"]')
    with pytest.raises(RegistryError, match="2 unresolved links") as caught:
        load_registry(root)
    assert "nobody1999" in str(caught.value) and "EVID-NONE" in str(caught.value)


def test_a_substantiated_claim_needs_a_passing_evidence_row(root: Path) -> None:
    _edit(root / "claims.toml", '["EVID-ALPHA"]', '["EVID-OLD"]')
    with pytest.raises(RegistryError, match="none of its evidence rows is passing"):
        load_registry(root)


def test_an_equation_id_is_unique_across_bundles(root: Path) -> None:
    _edit(
        root / "sources/derivations/equations.toml", '"EQ-DERIV-001"', '"EQ-ALPHA-001"'
    )
    with pytest.raises(
        RegistryError, match="registered in both alpha2001 and derivations"
    ):
        load_registry(root)


def test_an_absent_table_is_optional_but_references_into_it_dangle(root: Path) -> None:
    (root / "claims.toml").unlink()
    assert load_registry(root).claims == {}
    (root / "implementations.toml").unlink()
    with pytest.raises(RegistryError, match="names no implementation row"):
        load_registry(root)


def test_an_unrecognised_file_at_the_root_is_refused(root: Path) -> None:
    (root / "implementations.toml").rename(root / "implementation.toml")
    with pytest.raises(RegistryError, match="unrecognised registry files"):
        load_registry(root)


# ---------------------------------------------------------------------------
# environment checks: they report, the caller decides
# ---------------------------------------------------------------------------


def test_only_live_implementations_must_import(root: Path) -> None:
    assert unresolved_implementations(load_registry(root)) == ()
    _edit(
        root / "implementations.toml",
        'symbol = "load_registry"',
        'symbol = "no_such_symbol"',
    )
    assert unresolved_implementations(load_registry(root)) == (
        "IMPL-ALPHA: jaxstro.registry has no no_such_symbol",
    )
    _edit(
        root / "implementations.toml",
        'module = "jaxstro.registry"',
        'module = "jaxstro.no_such_module"',
    )
    (failure,) = unresolved_implementations(load_registry(root))
    assert failure.startswith(
        "IMPL-ALPHA: module jaxstro.no_such_module does not import"
    )


def test_only_live_evidence_files_must_exist(root: Path, tmp_path: Path) -> None:
    registry = load_registry(root)
    repo = tmp_path / "repo"
    assert missing_files(registry, repo) == (
        "EVID-ALPHA: tests/validation/test_alpha.py does not exist",
        "EVID-ALPHA: tests/unit/test_alpha.py does not exist",
        "EVID-ALPHA: validation/plots/alpha.png does not exist",
    )
    for path in (
        "tests/validation/test_alpha.py",
        "tests/unit/test_alpha.py",
        "validation/plots/alpha.png",
    ):
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).touch()
    # the retired row names a deleted file and is not checked
    assert missing_files(registry, repo) == ()


# ---------------------------------------------------------------------------
# BibTeX fields on SourceRecord, and their rendering
# ---------------------------------------------------------------------------

_BIB = {
    "id": "alpha2001",
    "verification": "agent-verified",
    "bib_type": "article",
    "authors": "Alpha, A.",
    "title": "T",
    "year": "2001",
}


@pytest.mark.parametrize(
    ("change", "phrase"),
    [
        ({"authors": None}, "needs ['authors']"),
        (
            {
                "bib_type": None,
                "authors": None,
                "title": None,
                "year": None,
                "journal": "J",
            },
            "needs ['bib_type', 'authors', 'title', 'year']",
        ),
        ({"bib_type": "paper"}, "bib_type='paper'"),
        ({"year": "01"}, "four digits"),
        ({"year": 2001}, "year must be a string"),
        ({"title": "An {Unclosed Title"}, "unbalanced braces"),
        ({"title": "A } early close {"}, "unbalanced braces"),
    ],
)
def test_partial_or_malformed_bibliographic_fields_are_refused(
    change: dict, phrase: str
) -> None:
    payload = {
        key: value for key, value in {**_BIB, **change}.items() if value is not None
    }
    with pytest.raises(RegistryError) as caught:
        SourceRecord.from_toml(payload, where="source.toml")
    assert phrase in str(caught.value)


def test_a_source_without_bibliographic_fields_still_loads_and_has_no_entry(
    root: Path,
) -> None:
    source = load_source(root, "derivations").source
    assert not source.has_bibliography
    with pytest.raises(RegistryError, match="no BibTeX entry"):
        bibtex_entry(source)


def test_the_bibtex_entry_is_keyed_by_bibkey_with_fields_in_fixed_order(
    root: Path,
) -> None:
    assert bibtex_entry(load_source(root, "beta2002").source) == (
        "@misc{beta2002,\n"
        "  author = {Beta, B.},\n"
        "  title = {A Beta Data Release},\n"
        "  year = {2002a},\n"
        "  eprint = {0201.00001},\n"
        "  archivePrefix = {arXiv}\n"
        "}"
    )
    alpha = bibtex_entry(load_source(root, "alpha2001").source)
    assert alpha.startswith(
        "@article{alpha2001,\n  author = {Alpha, A. and Gamma, G.},\n"
    )
    assert "  title = {{An} Alpha Paper},\n" in alpha
    assert alpha.endswith(
        "  doi = {10.0000/alpha.2001},\n  url = {https://doi.org/10.0000/alpha.2001}\n}"
    )


def test_the_bibliography_holds_every_citable_source_sorted(root: Path) -> None:
    text = bibliography(root)
    keys = [
        line.split("{", 1)[1].rstrip(",")
        for line in text.splitlines()
        if line.startswith("@")
    ]
    assert keys == ["alpha2001", "beta2002"]
    assert text.endswith("}\n")
