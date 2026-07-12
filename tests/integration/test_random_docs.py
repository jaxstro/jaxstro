"""Executable pedagogy contracts for the random-stream theory page."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RANDOM_PAGE = REPO_ROOT / "docs" / "10-theory" / "random.md"
BIBLIOGRAPHY = REPO_ROOT / "docs" / "99-bibliography" / "references.bib"


def _page_text() -> str:
    return RANDOM_PAGE.read_text(encoding="utf-8")


def _first_python_block() -> str:
    match = re.search(r"```python\n(?P<code>.*?)\n```", _page_text(), re.DOTALL)
    assert match is not None, "random page needs a standalone Python example"
    return match.group("code")


def test_random_example_is_standalone_and_executable() -> None:
    block = _first_python_block()
    for definition in ("key =", "weights =", "num_samples ="):
        assert definition in block

    namespace: dict[str, object] = {}
    exec(compile(block, str(RANDOM_PAGE), "exec"), namespace)

    subkeys = np.asarray(namespace["subkeys"])
    folded = np.asarray(namespace["folded"])
    manifest = namespace["manifest"]
    systematic = np.asarray(namespace["systematic"])
    stratified = np.asarray(namespace["stratified"])
    residual = np.asarray(namespace["residual"])
    replay = np.asarray(namespace["replay"])

    assert subkeys.shape == (3, 2)
    assert folded.shape == (3, 2)
    assert manifest == {
        "algorithm": "jax.random",
        "seed": 17,
        "stream": "particle-filter",
    }
    for indices in (systematic, stratified, residual):
        assert indices.shape == (5,)
        assert np.all((indices >= 0) & (indices < 3))
    np.testing.assert_array_equal(replay, systematic)
    np.testing.assert_array_equal(np.bincount(residual, minlength=3), [2, 2, 1])


def test_random_page_names_live_execution_contracts() -> None:
    text = _page_text()

    assert "```{list-table} Randomness and resampling contracts" in text
    assert ":label: tbl-random-contracts" in text
    for phrase in (
        "caller owns `next_key`",
        "integer indices",
        "`num_samples` is static",
        "zero-total fallback",
        "eager validation is skipped while weights are traced",
        "finite, nonnegative weights under `jax.jit`",
        "`validation_only`",
    ):
        assert phrase in text


def test_random_claims_have_primary_provenance() -> None:
    text = _page_text()
    bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")

    assert "{cite:t}`JAXJEP263`" in text
    assert "{cite:t}`DoucCappeMoulines2005`" in text
    assert "@misc{JAXJEP263" in bibliography
    assert "docs.jax.dev/en/latest/jep/263-prng.html" in bibliography
    assert "@inproceedings{DoucCappeMoulines2005" in bibliography
    assert "10.1109/ISPA.2005.195385" in bibliography


def test_random_page_routes_to_api_validation_and_contract_taxonomy() -> None:
    text = _page_text()

    assert "[](../40-api/index.md#jaxstro-numerics-random)" in text
    assert "[](../60-validation/index.md)" in text
    assert "[](./index.md#gradient-contracts)" in text
