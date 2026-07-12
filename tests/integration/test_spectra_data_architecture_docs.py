"""Executable contracts for the spectra data-architecture page."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from jaxstro.atmospheres import (
    STATUS_MISSING_ABUNDANCE,
    STATUS_OK,
    STATUS_OUT_OF_GRID,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs" / "20-architecture" / "spectra-data-architecture.md"
DESIGN_RECORD = (
    REPO_ROOT
    / "laboratory"
    / "jaxtroviz"
    / "design"
    / "2026-07-11-spectra-runtime-boundary.md"
)


def _page_text() -> str:
    return PAGE.read_text(encoding="utf-8")


def _portable_python_block() -> str:
    match = re.search(
        r"## Portable JAX-side example.*?```python\n(?P<code>.*?)\n```",
        _page_text(),
        re.DOTALL,
    )
    assert match is not None, "page needs a portable standalone Python example"
    return match.group("code")


def test_portable_prepared_grid_example_executes_without_local_artifacts() -> None:
    namespace: dict[str, object] = {}
    exec(compile(_portable_python_block(), str(PAGE), "exec"), namespace)

    midpoint = namespace["midpoint"]
    outside = namespace["outside"]
    wrong_plane = namespace["wrong_plane"]
    np.testing.assert_allclose(midpoint.spectrum.flux_lambda, [2.5, 3.5, 4.5])
    np.testing.assert_allclose(outside.spectrum.flux_lambda, [1.5, 2.5, 3.5])
    assert int(midpoint.status.code) == STATUS_OK
    assert int(outside.status.code) == STATUS_OUT_OF_GRID
    assert int(wrong_plane.status.code) == STATUS_MISSING_ABUNDANCE
    assert float(namespace["local_slope"]) == pytest.approx(0.002)


def test_page_classifies_every_nonportable_fence_and_ownership_boundary() -> None:
    text = _page_text()

    assert "```{code-block} text\n:caption: Interface notation, not Python" in text
    assert (
        text.count("**Execution contract — local processed artifacts required.**") == 2
    )
    assert "```{list-table} Spectra execution and ownership boundaries" in text
    assert ":label: tbl-spectra-execution-boundaries" in text
    for phrase in (
        "Catalog discovery and candidate ranking",
        "Artifact opening and local-cell preparation",
        "Prepared-grid interpolation",
        "Synthetic photometry and interpretation",
        "Atmosphere support remains in progress",
        "Sonora and TLUSTY do not yet have runtime backends",
    ):
        assert phrase in text


def test_page_embeds_registered_figure_and_evidence_routes() -> None:
    text = _page_text()

    assert "./figures/spectra-runtime-boundary.webp" in text
    assert ":name: fig-spectra-runtime-boundary" in text
    assert (
        ":alt: Three-stage spectra workflow from host-side catalog and artifact "
        "preparation through a JAX-ready local grid to downstream observables"
    ) in text
    assert DESIGN_RECORD.is_file()
    assert "[](../40-api/index.md#jaxstro-atmospheres)" in text
    assert "[](../60-validation/index.md)" in text
    assert "[](./atmosphere-capabilities.md)" in text
