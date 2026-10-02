"""BibTeX rendering of registry sources, so a consumer's bibliography is generated.

A package that cites a paper in its documentation and registers equations from
it otherwise keeps the paper's metadata twice: once in the source bundle that
carries its verification status, once in a hand-kept bibliography. hydrax kept
29 such rows in ``references.yaml`` beside 15 bundles (2026-10-02). With the
bibliographic fields on :class:`~jaxstro.registry.SourceRecord`, the ``.bib`` file
is a printing of the bundles and the two cannot disagree.

The entry key is the source id (the bibkey that names the bundle directory), so
a ``{cite}`` key and a registry bibkey are the same string.
"""

from __future__ import annotations

from pathlib import Path

from .loader import available_bibkeys, load_source
from .records import RegistryError, SourceRecord


def bibtex_entry(source: SourceRecord) -> str:
    """One BibTeX entry, keyed by the source id.

    Fields in a fixed order: ``author``, ``title``, ``journal``, ``volume``,
    ``pages``, ``year``, ``doi``, then ``url`` from the DOI and ``eprint`` with
    ``archivePrefix = arXiv`` from ``arxiv``. Absent optional fields are omitted.
    Raises for a source that declares no bibliographic fields.
    """
    if not source.has_bibliography:
        raise RegistryError(
            f"{source.id}: declares no bibliographic fields (bib_type, authors, "
            "title, year), so it has no BibTeX entry"
        )
    fields: list[tuple[str, str | None]] = [
        ("author", source.authors),
        ("title", source.title),
        ("journal", source.journal),
        ("volume", source.volume),
        ("pages", source.pages),
        ("year", source.year),
        ("doi", source.doi),
    ]
    if source.doi:
        fields.append(("url", f"https://doi.org/{source.doi}"))
    if source.arxiv:
        fields += [("eprint", source.arxiv), ("archivePrefix", "arXiv")]
    body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields if value)
    return f"@{source.bib_type}{{{source.id},\n{body}\n}}"


def bibliography(registry_root: Path) -> str:
    """Every citable source under ``registry_root``, sorted by bibkey.

    A source without bibliographic fields (a derivation bundle, which has no
    authors) is not citable and is left out; ``SourceRecord`` enforces that a
    source declares all of the required fields or none, so a source cannot drop
    out of the bibliography by missing one of them.
    """
    sources = [
        load_source(registry_root, bibkey).source
        for bibkey in available_bibkeys(registry_root)
    ]
    entries = [bibtex_entry(source) for source in sources if source.has_bibliography]
    return "\n\n".join(entries) + "\n" if entries else ""
